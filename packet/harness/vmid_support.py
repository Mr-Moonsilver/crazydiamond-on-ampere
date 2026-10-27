"""New VM jobs share bounded process supervision; job1 remains immutable."""
import json
from pathlib import Path
import subprocess
import time
from vmid_job1 import IMAGE, CUPTI, CAP_MIB, STOP_MIB, sha, write, command, parse_apps, graph_bytes, checked


def verdict(failures, missing):
    return {'verdict': 'FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS',
            'rc': 1 if failures else 2 if missing else 0,
            'failures': sorted(set(failures)), 'missing': sorted(set(missing))}


def combine(rows):
    return verdict([f'{i}:{x}' for i,r in enumerate(rows) for x in r['failures']],
                   [f'{i}:{x}' for i,r in enumerate(rows) for x in r['missing']])


def monitor(out, container):
    """One-second all-app evidence; only our named container can be stopped."""
    if container not in ('flashnext-job2', 'flashnext-job2b', 'flashnext-job3', 'flashnext-job4', 'flashnext-job5', 'flashnext-job5b', 'flashnext-job6', 'flashnext-job7', 'flashnext-job7b', 'flashnext-job8','flashnext-job9','flashnext-job9b','flashnext-job10','flashnext-job11','flashnext-job10b'):
        raise ValueError('unowned_container')
    summary = {'schema': 'vm<vm-id>-memory/1', 'samples': 0, 'own_samples': 0,
               'own_peak_mib': 0, 'errors': 0, 'stop_triggered': False,
               'cap_mib': CAP_MIB, 'stop_mib': STOP_MIB,
               'basis': '1s sampled process memory, not a hard instantaneous native-memory cap; Vulkan may be absent from compute-apps'}
    deadline = time.monotonic() + (540 if container in ('flashnext-job4','flashnext-job6', 'flashnext-job7', 'flashnext-job7b', 'flashnext-job8','flashnext-job9','flashnext-job9b','flashnext-job10','flashnext-job11','flashnext-job10b') else 240)
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
                top_rc, top = command(['sudo', '-n', 'docker', 'top', container, '-eo', 'pid'])
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
                    command(['sudo', '-n', 'docker', 'stop', '-t', '5', container])
                consecutive_errors = 0
            except (ValueError, subprocess.TimeoutExpired, OSError):
                summary['errors'] += 1
                consecutive_errors += 1
                row['error'] = 'sampling_failed'
                if consecutive_errors >= 3:
                    summary['stop_triggered'] = True
                    command(['sudo', '-n', 'docker', 'stop', '-t', '5', container])
            f.write(json.dumps(row, sort_keys=True) + '\n')
            f.flush()
            if (out / 'monitor.stop').exists():
                break
            if time.monotonic() >= deadline:
                summary['stop_triggered'] = True
                command(['sudo', '-n', 'docker', 'stop', '-t', '5', container])
                break
            time.sleep(max(0, 1 - (time.monotonic() - tick)))
    write(out / 'monitor.json', summary)
    return 0


def finalize(out, process_rc, grade, job):
    try:
        g = grade(json.loads((out / 'probe.json').read_text()))
    except (OSError, ValueError, TypeError, KeyError):
        g = verdict([], ['probe_result'])
    try:
        m = json.loads((out / 'monitor.json').read_text())
    except (OSError, ValueError):
        m = {}
    if m.get('stop_triggered') or m.get('own_peak_mib', 0) > CAP_MIB:
        g['failures'].append('resource_stop_or_cap')
    if process_rc in (139, 134):
        g['failures'].append('native_process_crash')
    elif process_rc not in (0, 1, 2) or process_rc != g['rc']:
        g['missing'].append('process_verdict_mismatch')
    if not m.get('own_samples') or m.get('errors', 1):
        g['missing'].append('memory_sampling')
    result = verdict(g['failures'], g['missing'])
    result.update(schema='vm<vm-id>-' + job + '/1', process_rc=process_rc, memory=m)
    write(out / 'result.json', result)
    print('VMID_' + job.upper() + ' ' + result['verdict'])
    return result['rc']


def observation_grade(r, replays, expected):
    fails, missing = list(r.get('failures', [])), list(r.get('missing', []))
    for name in ('output_exact', 'serialized_stable', 'cleanup_ok'):
        if r.get(name) is False:
            fails.append(name)
        elif r.get(name) is not True:
            missing.append(name)
    if r.get('serialized_bytes', 0) <= 0:
        missing.append('serialization')
    if r.get('replays') != replays:
        missing.append('replay_census')
    static, window = r.get('static', {}), r.get('window', {})
    if not static.get('complete') or not window.get('complete'):
        missing.append('observation_incomplete')
    context = window.get('expected_context', {})
    if (context.get('contradictory') or context.get('sm_mismatched', 0)
            or window.get('counts', {}).get('known_wrong_context_records', 0)):
        fails.append('wrong_context')
    if context.get('records', 0) and context.get('is_green') is False:
        fails.append('not_green')
    if context.get('is_green') is not True or context.get('sm_matching', 0) < 1:
        missing.append('green_context_evidence')
    if any(window.get('flags', {}).values()) or window.get('reasons'):
        missing.append('native_loss_or_unknown')
    slots, ss = window.get('graph_slots', []), static.get('graph_slots', [])
    if len(slots) != 1 or len(ss) != 1:
        missing.append('one_static_graph')
    else:
        for kind, count in expected.items():
            if ss[0].get('node_types', {}).get(kind) != count:
                fails.append('wrong_static_' + kind)
        for kind, field in (('kernel', 'kernel_executions'), ('memcpy', 'copy_executions'), ('memset', 'memset_executions')):
            n = expected[kind]
            bounds = slots[0].get('node_execution_bounds', {}).get(kind, {})
            if slots[0].get('required', {}).get(kind) != n:
                fails.append('inventory_changed_' + kind)
            if slots[0].get(field) != n * replays:
                missing.append(kind + '_record_count')
            if n and (bounds.get('min') != replays or bounds.get('max') != replays):
                missing.append(kind + '_node_multiplicity')
    return verdict(fails, missing)
