#!/usr/bin/env python3
"""VM<vm-id> tiny-graph probe / host resource observer. No vLLM import.

The native observer ABI is exercised directly: this is not a fabricated
WorkerObserver bank or an engine/controller qualification receipt.
"""
import argparse
import csv
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

IMAGE = 'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
CUPTI = 'e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e'
REPLAYS = 3
CAP_MIB = 3072
STOP_MIB = 2816  # leave room for context/native allocations between samples


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def grade(r):
    """FAIL wins; missing coverage/unsupported observation is INCONCLUSIVE."""
    fails = list(r.get('failures', []))
    missing = list(r.get('missing', []))
    for name in ('output_exact', 'serialized_stable', 'cleanup_ok'):
        if r.get(name) is False:
            fails.append(name)
        elif r.get(name) is not True:
            missing.append(name)
    if r.get('serialized_bytes', 0) <= 0:
        missing.append('serialized_bytes')
    if r.get('replays') != REPLAYS:
        missing.append('replay_census')
    static, window = r.get('static', {}), r.get('window', {})
    if not static.get('complete') or not window.get('complete'):
        missing.append('native_observation_incomplete')
    context = window.get('expected_context', {})
    if context.get('contradictory') or window.get('counts', {}).get('known_wrong_context_records', 0):
        fails.append('wrong_context')
    if context.get('records', 0) and (context.get('is_green') is False or context.get('sm_mismatched', 0)):
        fails.append('wrong_green_or_sm')
    if context.get('is_green') is not True or context.get('sm_matching', 0) < 1:
        missing.append('green_context_evidence')
    if any(window.get('flags', {}).values()) or window.get('reasons'):
        missing.append('native_loss_or_unknown')
    slots = window.get('graph_slots', [])
    ss = static.get('graph_slots', [])
    if len(slots) != 1 or len(ss) != 1:
        missing.append('one_static_graph')
    else:
        slot = slots[0]
        required = ss[0].get('required', {})
        if required.get('kernel', 0) < 1:
            missing.append('no_kernel')
        for kind, field in (('kernel', 'kernel_executions'), ('memcpy', 'copy_executions'), ('memset', 'memset_executions')):
            n = required.get(kind, 0)
            bounds = slot.get('node_execution_bounds', {}).get(kind, {})
            if slot.get('required', {}).get(kind) != n:
                fails.append('static_inventory_changed')
            if slot.get(field) != n * REPLAYS:
                missing.append(kind + '_record_count')
            if n and (bounds.get('min') != REPLAYS or bounds.get('max') != REPLAYS):
                missing.append(kind + '_node_multiplicity')
    verdict = 'FAIL' if fails else 'INCONCLUSIVE' if missing else 'PASS'
    return {'verdict': verdict, 'rc': {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[verdict],
            'failures': sorted(set(fails)), 'missing': sorted(set(missing))}


def checked(rc, action):
    if rc:
        raise RuntimeError(action + ':' + str(int(rc)))


def graph_bytes(lib, handle):
    f = lib.s5_graph_bytes
    f.argtypes = [C.c_uint64, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
    f.restype = C.c_int
    size = C.c_size_t()
    checked(f(handle, None, 0, C.byref(size)), 'serialize_size')
    if not 0 < size.value <= 16 * 1024 * 1024:
        raise RuntimeError('serializer_bound')
    buf = C.create_string_buffer(size.value)
    checked(f(handle, buf, len(buf), C.byref(size)), 'serialize_bytes')
    return bytes(buf.raw[:size.value])  # stays in process; never log/hash raw args


def probe(modules, native, out):
    r = {'schema': 'vm<vm-id>-job1/1', 'image': IMAGE, 'failures': [], 'missing': [],
         'scope': 'single GPU synthetic graph; no engine, collective, performance or q17b GO claim',
         'late_enable': 'observer starts before green creation; kernel activity enabled only after capture+instantiate',
         'memory_fraction': 0.125, 'allocator_config': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
         'replays': 0, 'started_wall_s': time.time()}
    session = split = torch = graph = None
    cuda_ready = False
    window_token = None
    try:
        sys.path.insert(0, str(modules))
        import torch
        # First CUDA allocation/control operation in this process. Native
        # context memory is additionally supervised by the host sidecar.
        torch.cuda.set_per_process_memory_fraction(0.125, 0)
        cuda_ready = True
        r['software'] = {'torch': str(torch.__version__), 'torch_cuda': torch.version.cuda}
        if torch.cuda.device_count() != 1:
            raise RuntimeError('requires_one_visible_gpu')
        props = torch.cuda.get_device_properties(0)
        r['device'] = {'name': props.name, 'total_memory': props.total_memory,
                       'capability': list(torch.cuda.get_device_capability(0)),
                       'sms': props.multi_processor_count}
        if '3090' not in props.name or r['device']['capability'] != [8, 6]:
            raise RuntimeError('unsupported_device')
        receipt = json.loads((native / 'q17-helper-build.json').read_text())
        if receipt['image'] != IMAGE or receipt['cupti_library_sha256'] != CUPTI:
            raise RuntimeError('build_identity')
        for name, digest in receipt['files'].items():
            if sha(native / name) != digest:
                raise RuntimeError('build_pin')
        for name in ('flash_s5_green.cu', 'flash_s5_replay_observer.cpp', 'flash_s5_graph_bytes.cpp'):
            if sha(modules / 'native' / name) != receipt['files'][name]:
                raise RuntimeError('module_source_pin')
        r['build'] = receipt
        wheel = Path('/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13')
        if sha(wheel) != CUPTI:
            raise RuntimeError('cupti_pin')
        import replay_observer as ro
        import green_context as gc
        # Explicit pinned path avoids depending on an unrelated profiler to
        # have loaded CUPTI first. This constructor verifies the library hash.
        binding = ro.ReplayBinding(str(native / 'flash_s5_replay_observer.so'),
                                  receipt['files']['flash_s5_replay_observer.so'],
                                  receipt['files']['flash_s5_replay_observer.cpp'], CUPTI,
                                  cupti_library_path=str(wheel))
        r['observer_binding'] = binding.identity()
        limits = ro.ObserverLimits(buffer_count=8, buffer_bytes=1024*1024,
                                   max_context_records=128, max_kernel_records=4096,
                                   max_copy_records=1024, max_graph_events=4096,
                                   max_graphs=2, max_nodes_per_graph=64,
                                   max_nodes_total=128, max_links=1024)
        session = ro.ObserverSession(binding, limits)
        session.start()
        lib, token = binding._lib, session._token
        r['observer_host_reserved_bytes'] = session._native_reserved
        split = gc.open_split(0, env={'FLASH_S5_DECODE_SMS': '44'}, torch_module=torch,
                              binding=gc.DriverBinding(str(native / 'flash_s5_green.so')))
        r['split'] = {'requested': 44, 'decode': split.decode_sms,
                      'prefill': split.prefill_sms, 'total': split.total_sms}
        # Tiny fixed storage; no cuBLAS workspaces or model allocations.
        x = torch.zeros(4096, device='cuda', dtype=torch.float32)
        y = torch.zeros_like(x)
        torch.cuda.synchronize()
        with torch.cuda.stream(split.decode_stream):
            for _ in range(3):
                y.zero_()
                x.add_(1.0)
                y.add_(x)
        split.decode_stream.synchronize()
        x.zero_()
        torch.cuda.synchronize()
        time.sleep(2)  # ensure the 1s host monitor sees our live CUDA context
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=split.decode_stream):
            y.zero_()
            x.add_(1.0)
            y.add_(x)
        graph.instantiate()
        torch.cuda.synchronize()
        handle = int(graph.raw_cuda_graph())
        before = graph_bytes(lib, handle)
        r['serialized_bytes'] = len(before)
        r['serialized_stable'] = before == graph_bytes(lib, handle)
        checked(lib.s5ro_bind_context(token, split.decode_context_id, split.decode_sms), 'bind_context')
        count = C.c_uint32()
        checked(lib.s5ro_bind_graph(token, handle, 0, C.byref(count)), 'bind_graph')
        r['static'] = session._report_call(lib.s5ro_static, (token,))
        ro._validate_static(r['static'])
        start = C.c_uint64()
        checked(lib.s5ro_begin_window(token, 1, C.byref(start)), 'begin_window')
        window_token = start.value
        with torch.cuda.stream(split.decode_stream):
            for _ in range(REPLAYS):
                graph.replay()
                r['replays'] += 1
        split.decode_stream.synchronize()
        r['window'] = session._report_call(lib.s5ro_end_window, (token, window_token, 1), retry=False)
        window_token = None
        ro._validate_window(r['window'], limits, 1)
        r['serialized_stable'] &= before == graph_bytes(lib, handle)
        # Capture itself records but does not execute. Only the three replays
        # increment x after reset. Actual payload values never enter receipts.
        r['output_exact'] = bool(torch.all(x == REPLAYS).item() and torch.all(y == REPLAYS).item())
        r['allocator_peak_allocated'] = torch.cuda.max_memory_allocated(0)
        r['allocator_peak_reserved'] = torch.cuda.max_memory_reserved(0)
        if r['allocator_peak_reserved'] > 64 * 1024 * 1024:
            r['failures'].append('unexpected_allocator_growth')
        time.sleep(1)  # post-work sample; no timing/performance interpretation
    except Exception as exc:
        # Short native codes/types only; no pointers, graph parameters or
        # arbitrary library exception strings in logs.
        code = getattr(exc, 'code', None)
        if isinstance(exc, RuntimeError) and exc.args and str(exc.args[0]).split(':')[0] in {
                'serialize_size', 'serialize_bytes', 'serializer_bound', 'bind_context', 'bind_graph',
                'begin_window', 'requires_one_visible_gpu', 'unsupported_device',
                'build_identity', 'build_pin', 'module_source_pin', 'cupti_pin'}:
            code = str(exc.args[0])
        r['error'] = {'type': type(exc).__name__, 'code': code}
        r['missing'].append('probe_exception')
        if code in ('build_identity', 'build_pin', 'module_source_pin', 'cupti_pin'):
            r['failures'].append('identity')
    finally:
        cleanup = []
        try:
            if cuda_ready:
                torch.cuda.synchronize()
            if session is not None and window_token is not None:
                session._report_call(session.binding._lib.s5ro_end_window,
                                     (session._token, window_token, 1), retry=False)
        except Exception:
            cleanup.append('window_drain')
        try:
            if session is not None:
                session.close()
        except Exception:
            cleanup.append('observer_close')
        try:
            if graph is not None:
                graph.reset()
            if split is not None:
                split.close()
        except Exception:
            cleanup.append('graph_split_close')
        r['cleanup_ok'] = not cleanup
        r['cleanup_errors'] = cleanup
        r['ended_wall_s'] = time.time()
        r['grade'] = grade(r)
        write(out / 'probe.json', r)
    return r['grade']['rc']


def command(argv):
    r = subprocess.run(argv, capture_output=True, text=True, timeout=8)
    return r.returncode, r.stdout


def parse_apps(text):
    rows = []
    for row in csv.reader(text.splitlines()):
        if not row or not any(x.strip() for x in row):
            continue
        if len(row) != 4:
            raise ValueError('bad_app_row')
        gpu, pid, name, used = (x.strip() for x in row)
        rows.append({'gpu_uuid': gpu, 'pid': int(pid), 'process_name': name,
                     'used_mib': int(used)})
    return rows


def monitor(out):
    """One-second all-app evidence; only our named container can be stopped."""
    summary = {'schema': 'vm<vm-id>-memory/1', 'samples': 0, 'own_samples': 0,
               'own_peak_mib': 0, 'errors': 0, 'stop_triggered': False,
               'cap_mib': CAP_MIB, 'stop_mib': STOP_MIB,
               'basis': '1s sampled process memory, not a hard instantaneous native-memory cap; Vulkan may be absent from compute-apps'}
    deadline = time.monotonic() + 240
    consecutive_errors = 0
    with (out / 'compute-apps.jsonl').open('x') as f:
        while True:
            tick = time.monotonic()
            row = {'wall_s': time.time()}
            try:
                rc, text = command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'])
                row['query_rc'] = rc
                if rc:
                    raise ValueError('smi_failed')
                row['apps'] = parse_apps(text)
                top_rc, top = command(['sudo', '-n', 'docker', 'top', 'flashnext-job1', '-eo', 'pid'])
                pids = {int(x.strip()) for x in top.splitlines()[1:] if x.strip().isdigit()} if top_rc == 0 else set()
                row['own_pids'] = sorted(pids)
                own = [x for x in row['apps'] if x['pid'] in pids]
                row['own_mib'] = sum(x['used_mib'] for x in own)
                rc, gpu = command(['nvidia-smi', '--query-gpu=timestamp,uuid,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'])
                row['gpu_query_rc'], row['gpu_csv'] = rc, gpu.strip()
                if rc:
                    summary['errors'] += 1
                summary['samples'] += 1
                if own:
                    summary['own_samples'] += 1
                    summary['own_peak_mib'] = max(summary['own_peak_mib'], row['own_mib'])
                if row['own_mib'] >= STOP_MIB:
                    summary['stop_triggered'] = True
                    command(['sudo', '-n', 'docker', 'stop', '-t', '5', 'flashnext-job1'])
                consecutive_errors = 0
            except (ValueError, subprocess.TimeoutExpired, OSError):
                summary['errors'] += 1
                consecutive_errors += 1
                row['error'] = 'sampling_failed'
                if consecutive_errors >= 3:
                    summary['stop_triggered'] = True
                    command(['sudo', '-n', 'docker', 'stop', '-t', '5', 'flashnext-job1'])
            f.write(json.dumps(row, sort_keys=True) + '\n')
            f.flush()
            if (out / 'monitor.stop').exists():
                break
            if time.monotonic() >= deadline:
                summary['stop_triggered'] = True
                command(['sudo', '-n', 'docker', 'stop', '-t', '5', 'flashnext-job1'])
                break
            time.sleep(max(0, 1 - (time.monotonic() - tick)))
    write(out / 'monitor.json', summary)
    return 0


def finalize(out, process_rc):
    try:
        r = json.loads((out / 'probe.json').read_text())
        g = grade(r)
    except (OSError, ValueError, TypeError, KeyError):
        g = {'verdict': 'INCONCLUSIVE', 'rc': 2, 'failures': [], 'missing': ['probe_result']}
    try:
        m = json.loads((out / 'monitor.json').read_text())
    except (OSError, ValueError):
        m = {}
    if m.get('stop_triggered') or m.get('own_peak_mib', 0) > CAP_MIB:
        g['failures'].append('resource_stop_or_cap')
    if process_rc in (139, 134):
        g['failures'].append('native_process_crash')
    elif process_rc not in (0, 1, 2):
        g['missing'].append('container_abnormal_exit')
    elif process_rc != g['rc']:
        g['missing'].append('process_verdict_mismatch')
    if not m.get('own_samples') or m.get('errors', 1):
        g['missing'].append('memory_sampling')
    g['verdict'] = 'FAIL' if g['failures'] else 'INCONCLUSIVE' if g['missing'] else 'PASS'
    g['rc'] = {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[g['verdict']]
    g.update({'schema': 'vm<vm-id>-job1-result/1', 'process_rc': process_rc, 'memory': m,
              'scope': 'functional small-graph late-CUPTI/serializer check only; shared GPU, no timing or engine qualification'})
    write(out / 'result.json', g)
    print('VMID_JOB1 ' + g['verdict'])
    return g['rc']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('probe', 'monitor', 'finalize'), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--modules', type=Path)
    p.add_argument('--native', type=Path)
    p.add_argument('--process-rc', type=int, default=2)
    a = p.parse_args()
    if a.mode == 'probe':
        return probe(a.modules, a.native, a.out)
    if a.mode == 'monitor':
        return monitor(a.out)
    return finalize(a.out, a.process_rc)


if __name__ == '__main__':
    sys.exit(main())
