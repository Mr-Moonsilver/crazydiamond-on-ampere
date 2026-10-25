#!/usr/bin/env python3
"""Standalone E3 green-context probe. GPU SLOT REQUIRED; never controls an engine.

Rung 1 of the E3 ladder: build the reviewed helper ``native/flash_s5_green.cu``
inside the pinned image with the same sm_86 recipe as ``s5_clock.build_command``,
open the declared fixed split on every requested GPU, measure SM membership, and
time a decode-sized bf16 matmul on the ordinary, decode and prefill streams.
Evidence only; no performance claim beyond the recorded numbers.

Run inside the pinned image (the queue does exactly this):

    FLASH_S5_DECODE_SMS=44 python3 /h/e3_green_probe.py \
        --native /n --out /q --build /q/flash_s5_green.so --devices 0,1,2,3

``--native`` is the checksummed E3_NATIVE directory (green_context.py,
e3_types.py, native/flash_s5_green.cu, SHA256SUMS); ``--out`` is a fresh
writable directory. ``--plan`` is CPU-only and imports neither torch nor
green_context. Importing this module has no CUDA side effect.

Outputs (exclusive/atomic create; a previous run is never overwritten):
  <out>/gpu-<n>.json   one record per requested device
  <out>/summary.json   frozen verdict + reasons
The final stdout line is ``E3_GREEN_PROBE <PASS|FAIL|INCONCLUSIVE> ...``.
No device addresses, secrets or hostnames are written; strings are scrubbed.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time

# Exact reuse of the reviewed S5 build recipe (no duplicate flag list here).
from s5_clock import build_command, json_safe

DEFAULT_DEVICES = '0,1,2,3'
DEFAULT_ROUNDS = 3
DEFAULT_SIZE = 4096
DEFAULT_ITERS = 20
ABI_VERSION = 1

# Error kinds that mean "the probe could not measure this", not "the device is
# wrong". Any other recorded kind is hard evidence and grades FAIL.
INCONCLUSIVE_KINDS = frozenset({'abi', 'build', 'capability', 'config', 'internal',
                                'interrupted', 'missing'})


# -- sanitation --------------------------------------------------------------

def scrub_text(value, limit=4000):
    """Remove hostnames, IPv4 literals and pointer-like addresses from text."""
    text = str(value)
    try:
        host = socket.gethostname()
    except Exception:  # pragma: no cover - no hostname is not a probe failure
        host = ''
    if host:
        text = text.replace(host, '<host>')
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<ip>', text)
    text = re.sub(r'\b0x[0-9a-fA-F]{6,}\b', '<addr>', text)
    return text[:limit]


def scrub_obj(value):
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {key: scrub_obj(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub_obj(item) for item in value]
    return value


# -- build -------------------------------------------------------------------

def _tool_version(command):
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=10)
        text = ((done.stdout or '') + (done.stderr or '')).strip()
        return scrub_text(text, 2000) if text else ''
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 'unavailable: ' + scrub_text(exc)


def build(source, library):
    """Compile the helper; return an evidence record; never touch the engine.

    ``s5_clock.build_command`` is the reviewed recipe; a pre-existing library is
    refused rather than replaced. A load-time ABI query is recorded without any
    CUDA call (``s5_abi`` returns a constant). ``available`` is True only when a
    library was produced; ``abi`` must be 1 for the frozen verdict to pass.
    """
    source = Path(source)
    library = Path(library)
    record = {'available': False, 'abi': None, 'rc': None, 'error': None,
              'source': 'native/flash_s5_green.cu', 'source_sha256': None,
              'library': library.name, 'library_sha256': None,
              'nvcc_version': None, 'elf_needed': [],
              'readelf': bool(shutil.which('readelf')),
              'log_tail': None, 'command': None,
              'cudart': 'shared', 'arch': 'sm_86', 'ptx_fallback': False,
              'refused': False}
    if library.exists():
        record.update(refused=True,
                      error='refusing to replace existing library: ' + library.name)
        return record
    if not source.is_file():
        record['error'] = 'helper source is missing: ' + source.name
        return record
    try:
        record['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        command = build_command(source, library, (8, 6), 'shared')
        record['command'] = [str(part) for part in command]
        record['nvcc_version'] = _tool_version(['nvcc', '--version'])
        done = subprocess.run(command, capture_output=True, text=True, timeout=180)
        record['rc'] = done.returncode
        record['log_tail'] = scrub_text((done.stdout or '') + (done.stderr or ''), 6000)
        if done.returncode:
            record['error'] = 'nvcc failed with rc=%d' % done.returncode
            return record
        if not library.is_file():
            record['error'] = 'nvcc reported success but produced no library'
            return record
        record['library_sha256'] = hashlib.sha256(library.read_bytes()).hexdigest()
        record['available'] = True
        try:
            loaded = ctypes.CDLL(str(library))
            loaded.s5_abi.argtypes = []
            loaded.s5_abi.restype = ctypes.c_int
            record['abi'] = int(loaded.s5_abi())
        except Exception as exc:  # library exists but cannot be bound
            record['error'] = 'library load/ABI query failed: ' + scrub_text(exc)
        if record['readelf']:
            try:
                elf = subprocess.run(['readelf', '-d', str(library)],
                                     capture_output=True, text=True, timeout=10)
                record['elf_needed'] = [line.strip() for line in elf.stdout.splitlines()
                                        if 'NEEDED' in line]
            except (OSError, subprocess.TimeoutExpired) as exc:
                record['elf_needed'] = ['readelf failed: ' + scrub_text(exc)]
        return record
    except (OSError, subprocess.TimeoutExpired) as exc:
        record['error'] = 'helper build failed: ' + scrub_text(exc)
        return record


# -- timing ------------------------------------------------------------------

def measure_timings(torch, split, device, size, iters):
    """Same-stream CUDA-event timings after warmup; no tolerance is applied."""
    cuda = torch.cuda
    dtype = torch.bfloat16
    a = torch.randn((size, size), device='cuda:%d' % device, dtype=dtype)
    b = torch.randn((size, size), device='cuda:%d' % device, dtype=dtype)
    streams = {'ordinary': cuda.current_stream(device),
               'decode_stream': split.decode_stream,
               'prefill_stream': split.prefill_stream}
    # Warm every stream (kernel, cuBLAS handle, allocator path) before timing.
    for stream in streams.values():
        with cuda.stream(stream):
            for _ in range(2):
                torch.matmul(a, b)
        stream.synchronize()
    cuda.synchronize(device)
    timings = {'shape': [size, size, size], 'dtype': 'bfloat16', 'iterations': iters}
    for name, stream in streams.items():
        samples = []
        for _ in range(iters):
            start = cuda.Event(enable_timing=True)
            end = cuda.Event(enable_timing=True)
            with cuda.stream(stream):
                start.record()
                torch.matmul(a, b)
                end.record()
            stream.synchronize()
            samples.append(float(start.elapsed_time(end)))
        timings[name] = {'ms': samples, 'total_ms': float(sum(samples)),
                         'median_ms': float(statistics.median(samples)),
                         'min_ms': float(min(samples)), 'max_ms': float(max(samples))}
    # Concurrent solo launches: each stream's own event span plus wall time.
    decode_start = cuda.Event(enable_timing=True)
    decode_end = cuda.Event(enable_timing=True)
    prefill_start = cuda.Event(enable_timing=True)
    prefill_end = cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    with cuda.stream(split.prefill_stream):
        prefill_start.record()
        for _ in range(iters):
            torch.matmul(a, b)
        prefill_end.record()
    with cuda.stream(split.decode_stream):
        decode_start.record()
        for _ in range(iters):
            torch.matmul(a, b)
        decode_end.record()
    cuda.synchronize(device)
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    timings['concurrent'] = {
        'wall_ms': float(wall_ms),
        'decode_stream_ms': float(decode_start.elapsed_time(decode_end)),
        'prefill_stream_ms': float(prefill_start.elapsed_time(prefill_end)),
        'iterations': iters}
    del a, b
    cuda.synchronize(device)
    return timings


# -- per-device measurement --------------------------------------------------

def _error(kind, message):
    return {'kind': kind, 'message': scrub_text(message)}


def classify_exception(exc):
    """Map an exception to an evidence kind; only the named kinds grade FAIL."""
    text = str(exc)
    name = type(exc).__name__
    if name == 'GreenContextError':
        if 'ABI' in text and '!=' in text:
            return 'abi'
        if 'FLASH_S5_GREEN_LIB' in text or 'FLASH_S5_DECODE_SMS' in text:
            return 'config'
        if 'SM probe' in text:
            return 'membership'
        return 'cuda'
    if name == 'OutOfMemoryError':
        return 'launch'
    if name in ('KeyboardInterrupt', 'SystemExit'):
        return 'interrupted'
    if name in ('RuntimeError', 'MemoryError') and ('CUDA' in text or 'cuda' in text
                                                    or 'out of memory' in text.lower()):
        return 'launch'
    return 'internal'


def _measure_device(record, torch, green_context, args):
    """open_split -> membership(rounds) -> describe -> timings -> close."""
    torch.cuda.set_device(record['device'])  # events have no device argument
    requested = green_context.declared_decode_sms()  # fail closed if undeclared
    record['requested_decode_sms'] = requested
    os.environ['FLASH_S5_GREEN_LIB'] = str(args.library)
    split = green_context.open_split(record['device'])
    failure = None
    try:
        membership = split.membership(rounds=args.rounds)
        record['membership'] = membership.as_record()
        record['describe'] = split.describe()
        record['timings'] = measure_timings(torch, split, record['device'],
                                            args.size, args.iters)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if failure is None:
            split.close()  # a close failure is the primary measurement failure
        else:
            try:
                split.close()
            except Exception as close_exc:
                record['errors'].append(
                    _error('cuda', 'close after failure: ' + scrub_text(close_exc)))


# -- frozen verdict ----------------------------------------------------------

def _record_verdict(record):
    """Frozen per-device rules. Returns (verdict, reasons)."""
    build_record = record.get('build') or {}
    if build_record.get('available') is not True:
        return 'INCONCLUSIVE', ['helper not built']
    if build_record.get('abi') != ABI_VERSION:
        return 'INCONCLUSIVE', ['helper ABI is not %d' % ABI_VERSION]
    status = record.get('status')
    if status == 'interrupted':
        return 'INCONCLUSIVE', ['interrupted']
    if status == 'missing':
        return 'INCONCLUSIVE', ['device missing']
    capability = record.get('capability')
    if isinstance(capability, (list, tuple)) and len(capability) == 2:
        capability = [capability[0], capability[1]]
    else:
        capability = None
    if capability != [8, 6]:
        return 'INCONCLUSIVE', ['capability %r != [8, 6]' % (capability,)]
    errors = record.get('errors') or []
    hard = [err for err in errors if err.get('kind') not in INCONCLUSIVE_KINDS]
    if hard:
        return 'FAIL', ['%s: %s' % (err.get('kind'), err.get('message', '')) for err in hard]
    inconclusive = [err for err in errors if err.get('kind') in INCONCLUSIVE_KINDS]
    if inconclusive:
        return 'INCONCLUSIVE', ['%s: %s' % (err.get('kind'), err.get('message', ''))
                                for err in inconclusive]
    if status != 'ok':
        return 'INCONCLUSIVE', ['not measured (status %r)' % (status,)]
    describe = record.get('describe') or {}
    helper = describe.get('helper') or {}
    if helper.get('abi') != ABI_VERSION:
        return 'FAIL', ['loaded helper ABI is not %d' % ABI_VERSION]
    if (build_record.get('source_sha256')
            and helper.get('helper_source_sha256') != build_record.get('source_sha256')):
        return 'FAIL', ['helper source identity mismatch']
    if (build_record.get('library_sha256')
            and helper.get('library_sha256') != build_record.get('library_sha256')):
        return 'FAIL', ['helper library identity mismatch']
    membership = record.get('membership') or {}
    if membership.get('ok') is not True:
        return 'FAIL', ['membership not ok']
    requested = record.get('requested_decode_sms')
    total = describe.get('total_sms')
    if type(requested) is not int or requested < 1:
        return 'FAIL', ['requested decode SM count not recorded']
    if describe.get('decode_sms') != requested:
        return 'FAIL', ['decode SMs %r != requested %d'
                        % (describe.get('decode_sms'), requested)]
    if type(total) is not int:
        return 'FAIL', ['total SM count not recorded']
    if describe.get('prefill_sms') != total - requested:
        return 'FAIL', ['prefill SMs %r != total - requested %d'
                        % (describe.get('prefill_sms'), total - requested)]
    if membership.get('disjoint') is not True:
        return 'FAIL', ['decode/prefill SM sets are not disjoint']
    if (membership.get('decode_seen') != requested
            or membership.get('prefill_seen') != total - requested
            or membership.get('total') != total):
        return 'FAIL', ['membership counts off']
    timings = record.get('timings') or {}
    for key in ('ordinary', 'decode_stream', 'prefill_stream'):
        entry = timings.get(key)
        total_ms = entry.get('total_ms') if isinstance(entry, dict) else None
        if (not isinstance(entry, dict) or not entry.get('ms')
                or not isinstance(total_ms, (int, float)) or total_ms <= 0):
            return 'FAIL', ['timings missing: ' + key]
    concurrent = timings.get('concurrent')
    wall_ms = concurrent.get('wall_ms') if isinstance(concurrent, dict) else None
    decode_ms = concurrent.get('decode_stream_ms') if isinstance(concurrent, dict) else None
    prefill_ms = concurrent.get('prefill_stream_ms') if isinstance(concurrent, dict) else None
    if (not isinstance(concurrent, dict)
            or not isinstance(wall_ms, (int, float)) or wall_ms <= 0
            or not isinstance(decode_ms, (int, float)) or decode_ms <= 0
            or not isinstance(prefill_ms, (int, float)) or prefill_ms <= 0):
        return 'FAIL', ['timings missing: concurrent']
    return 'PASS', []


def verdict(records):
    """Frozen whole-run rules: PASS only if every requested device passes.

    A hard failure on any measured device outranks an inconclusive device, so a
    real CUresult/launch/count failure is never hidden by a missing or
    wrong-capability device. A build failure (nothing measured) or an
    interrupt/missing-device-only run is INCONCLUSIVE. No tolerance is applied.
    """
    if not isinstance(records, (list, tuple)) or not records:
        return {'verdict': 'INCONCLUSIVE', 'reasons': ['no device records']}
    results = [(record.get('device'), _record_verdict(record)) for record in records]
    reasons = []
    for device, (_, why) in results:
        reasons.extend('device %r: %s' % (device, item) for item in why)
    outcomes = [result[0] for _, result in results]
    if 'FAIL' in outcomes:
        overall = 'FAIL'
    elif 'INCONCLUSIVE' in outcomes:
        overall = 'INCONCLUSIVE'
    elif all(outcome == 'PASS' for outcome in outcomes):
        overall = 'PASS'
    else:  # pragma: no cover - _record_verdict only returns those three
        overall = 'INCONCLUSIVE'
    return {'verdict': overall, 'reasons': reasons}


# -- outputs -----------------------------------------------------------------

def write_json(path, value):
    """Atomic exclusive create: a concurrent/killed writer never leaves partial JSON."""
    path = Path(path)
    if path.exists():
        raise FileExistsError('refusing to overwrite ' + path.name)
    temp = path.with_name(path.name + '.tmp')
    if temp.exists():
        raise FileExistsError('refusing to overwrite ' + temp.name)
    text = json.dumps(json_safe(scrub_obj(value)), indent=2, allow_nan=False)
    with open(temp, 'x') as handle:
        handle.write(text + '\n')
    os.replace(temp, path)


def _flush_record(out, record):
    """Write a completed device record once, so a later hang keeps earlier evidence."""
    path = Path(out) / ('gpu-%d.json' % record['device'])
    if not path.exists():
        write_json(path, record)


def _unavailable_build(error):
    return {'available': False, 'abi': None, 'rc': None,
            'error': error.get('message') if isinstance(error, dict) else scrub_text(error),
            'source': 'native/flash_s5_green.cu', 'source_sha256': None,
            'library_sha256': None, 'nvcc_version': None, 'elf_needed': [],
            'log_tail': None, 'command': None, 'refused': False}


def _finish(records, args, devices, environment):
    result = verdict(records)
    requested = next((record.get('requested_decode_sms') for record in records
                      if record.get('requested_decode_sms') is not None), None)
    summary = {'schema': 1, 'verdict': result['verdict'], 'reasons': result['reasons'],
               'requested_devices': devices, 'requested_decode_sms': requested,
               'build': records[0].get('build') if records else None,
               'environment': environment,
               'records': ['gpu-%d.json' % record['device'] for record in records]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for record in records:
        _flush_record(out, record)
    write_json(out / 'summary.json', summary)
    line = 'E3_GREEN_PROBE ' + result['verdict']
    if result['reasons']:
        line += ' ' + '; '.join(result['reasons'][:5])
    print(line, flush=True)
    return {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[result['verdict']]


# -- command line ------------------------------------------------------------

def parse_devices(text):
    devices = [int(part) for part in str(text).split(',')]
    if not devices or any(device < 0 for device in devices) or len(set(devices)) != len(devices):
        raise ValueError('unique nonnegative device ordinals required')
    return devices


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, help='checksummed E3_NATIVE directory')
    parser.add_argument('--out', type=Path, help='fresh writable output directory')
    parser.add_argument('--build', dest='library', type=Path,
                        help='library path to create (refused if it exists)')
    parser.add_argument('--devices', default=DEFAULT_DEVICES,
                        help='visible CUDA ordinals, comma separated (default %s)' % DEFAULT_DEVICES)
    parser.add_argument('--rounds', type=int, default=DEFAULT_ROUNDS,
                        help='membership rounds (default %d)' % DEFAULT_ROUNDS)
    parser.add_argument('--size', type=int, default=DEFAULT_SIZE,
                        help='square matmul side (default %d)' % DEFAULT_SIZE)
    parser.add_argument('--iters', type=int, default=DEFAULT_ITERS,
                        help='timed iterations (default %d)' % DEFAULT_ITERS)
    parser.add_argument('--plan', action='store_true', help='print the CPU-only plan and exit')
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        devices = parse_devices(args.devices)
    except ValueError:
        parser.error('--devices requires unique nonnegative ordinals')
    if args.rounds < 1 or args.size < 1 or args.iters < 1:
        parser.error('--rounds, --size and --iters must be positive')
    if args.plan:
        print(json.dumps({
            'schema': 1,
            'command': build_command('native/flash_s5_green.cu', 'flash_s5_green.so',
                                     (8, 6), 'shared'),
            'devices': devices, 'rounds': args.rounds,
            'matmul': [args.size, args.size, args.size], 'dtype': 'bfloat16',
            'iterations': args.iters, 'decode_sms_env': 'FLASH_S5_DECODE_SMS',
            'gpu_required': True}, indent=2))
        return 0
    if args.native is None or args.out is None or args.library is None:
        parser.error('--native, --out and --build are required together')
    source = Path(args.native) / 'native' / 'flash_s5_green.cu'
    records = [{'schema': 1, 'device': device, 'status': 'new',
                'requested_decode_sms': None, 'capability': None, 'build': None,
                'describe': None, 'membership': None, 'timings': None,
                'errors': []} for device in devices]
    environment = {'python': sys.version.split()[0], 'torch': None, 'torch_cuda': None,
                   'device_count': None,
                   'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    torch = green_context = None
    try:
        import torch as torch_module
        torch = torch_module
        environment['torch'] = getattr(torch, '__version__', None)
        environment['torch_cuda'] = getattr(getattr(torch, 'version', None), 'cuda', None)
        import green_context as green_module
        green_context = green_module
    except Exception as exc:
        fatal = _error('config', 'import failed: ' + scrub_text(exc))
        for record in records:
            record['status'] = 'error'
            record['build'] = _unavailable_build(fatal)
            record['errors'].append(fatal)
        return _finish(records, args, devices, environment)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
        device_count = int(torch.cuda.device_count())
        environment['device_count'] = device_count
    except Exception as exc:
        fatal = _error('config', 'CUDA device query failed: ' + scrub_text(exc))
        for record in records:
            record['status'] = 'error'
            record['build'] = _unavailable_build(fatal)
            record['errors'].append(fatal)
        return _finish(records, args, devices, environment)
    # Capability is checked per device before any split is opened.
    for record in records:
        device = record['device']
        if device >= device_count:
            record['status'] = 'missing'
            record['errors'].append(_error(
                'missing', 'device %d is not visible (count=%d)' % (device, device_count)))
            continue
        try:
            capability = tuple(int(value) for value in
                               torch.cuda.get_device_capability(device))
        except Exception as exc:
            record['status'] = 'missing'
            record['errors'].append(_error(
                'missing', 'capability query failed: ' + scrub_text(exc)))
            continue
        record['capability'] = [capability[0], capability[1]]
        if capability != (8, 6):
            record['status'] = 'capability'
            record['errors'].append(_error(
                'capability', 'device %d capability %r != (8, 6)' % (device, capability)))
    # One helper build, shared by every 8.6 device; never replaced.
    if any(record.get('capability') == [8, 6] for record in records):
        build_record = build(source, args.library)
    else:
        build_record = _unavailable_build('no requested device reports capability (8, 6)')
    for record in records:
        record['build'] = build_record
    for record in records:
        if record['status'] != 'new':
            _flush_record(args.out, record)
            continue
        if build_record.get('available') is not True:
            record['status'] = 'build-failed'
            record['errors'].append(_error(
                'build', build_record.get('error') or 'helper not built'))
            _flush_record(args.out, record)
            continue
        record['status'] = 'measuring'
        try:
            _measure_device(record, torch, green_context, args)
            record['status'] = 'ok'
        except KeyboardInterrupt:
            record['status'] = 'interrupted'
            record['errors'].append(_error('interrupted', 'interrupted during measurement'))
            break
        except Exception as exc:
            record['status'] = 'error'
            record['errors'].append(_error(classify_exception(exc), scrub_text(exc)))
        finally:
            _flush_record(args.out, record)
    return _finish(records, args, devices, environment)


if __name__ == '__main__':
    raise SystemExit(main())
