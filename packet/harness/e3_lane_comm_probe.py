#!/usr/bin/env python3
"""Standalone E3 prefill-lane communicator probe. GPU SLOT REQUIRED; no engine.

Rung 3 of the E3 GPU ladder: prove, on real NCCL and real GPUs, that the lane's
independent communicators do what ``patches/s5_engine/lane_comm.py`` claims,
with no vLLM engine running. Two phases, each its own pair of worker processes
(torch.multiprocessing spawn, one per visible GPU) inside one container:

  tp  label ``tp-pair-1,3-real``      TP=2 PP=1, real TP group
      1. PyNccl all-reduce of (chunk_tokens x hidden_width) bf16 on an
         explicitly created torch.cuda.Stream (the green prefill stream stand-in;
         recorded as such), exact against a torch reference, event-timed.
      2. The same all-reduce interleaved with a 4096^3 bf16 matmul loop on a
         second stream of the same process, lane-local spans recorded.
      3. tp_scope routing: the lane communicator serves the group call and a
         spy on the stock device communicator records zero calls.
  pp  label ``two-card-PP-synthetic``  TP=1 PP=2, synthetic pair-1,3 topology
      1. tagged PP round trips, payload and header equality.
      2. sender/receiver sequence mismatch is refused/quarantined by the
         module's own rule, bounded polling (never a hang).
      3. typed PPConsumerReceipt from a receiver-queried CUDA event releases the
         send ticket; wrong seq / route / proof kind are refused and leave the
         ticket unreleased.
      4. bytes_owned() inventory per rank.

``--phase {tp,pp,both}`` selects the phases; ``both`` (default) runs them in
order, each a separate spawn launch. ``--plan`` is CPU-only and imports neither
torch nor vLLM. Importing this module has no CUDA side effect.

Outputs (exclusive/atomic create; nothing is overwritten):
  <out>/<phase>-rank-<r>.json   one record per phase and rank
  <out>/summary.json            frozen verdict + reasons
The final stdout line is ``E3_LANE_COMM_PROBE <PASS|FAIL|INCONCLUSIVE> ...``.
No device addresses, secrets, hostnames or payloads are written.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import sys
import time

RANKS = (0, 1)
PHASES = ('tp', 'pp')
LABELS = {'tp': 'tp-pair-1,3-real', 'pp': 'two-card-PP-synthetic'}
PP_TOPOLOGY = ('synthetic pair-1,3 PP (production ranks ordered 0,2,1,3; '
               'cross-stage 0->1 and 2->3)')
DEFAULT_DEVICES = '0,1'
DEFAULT_MAX_TOKENS = 2048
DEFAULT_ITERS = 20
DEFAULT_TOKEN_COUNTS = (1, 64, 512, 2048)
DEFAULT_POLL_TIMEOUT = 30.0
DEFAULT_RECEIPT_TIMEOUT = 60.0
DEFAULT_PHASE_TIMEOUT = 600.0
WARMUP_ITERS = 2
MATMUL_SIZE = 4096
SYNTHETIC_UID = 'q17c-synthetic'
PEER_WORKER_RANK = 1
PEER_STORAGE_EPOCH = 1
PEER_CONTEXT_ID = 1
ROUTE_GENERATION = 7

# Independent reviewed pins of the modules under test (same convention as rung 1:
# the checksummed directory is not merely self-signed).
NATIVE_PINS = {
    'lane_comm.py': 'ae86ed9b5799fc6de8cb8159913999eaf9c047838604e9ec10a7506db15fb14f',
    'e3_types.py': '2a9f08b94d7e383efd358f8b044648cf52a65d76f71119a9ebdc89e5f1d04068',
    'green_context.py': '26fdd7f650a17bd4382a46b0875d83f5cfde3287ee5221bc600c3870773dc7d9',
}

# Error kinds that mean "the probe could not measure this", not "the lane is
# wrong". Any other recorded kind is hard evidence and grades FAIL.
INCONCLUSIVE_KINDS = frozenset({'capability', 'config', 'import', 'internal',
                                'interrupted', 'missing', 'timeout'})


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


def _error(kind, message):
    return {'kind': kind, 'message': scrub_text(message)}


def classify_exception(exc):
    """Map an exception to an evidence kind; only named kinds grade FAIL."""
    name = type(exc).__name__
    text = str(exc)
    low = text.lower()
    if name in ('KeyboardInterrupt', 'SystemExit'):
        return 'interrupted'
    if name in ('ImportError', 'ModuleNotFoundError'):
        return 'import'
    if name == 'LaneCommError':
        return 'protocol'
    if name in ('OutOfMemoryError', 'MemoryError') or 'out of memory' in low:
        return 'launch'
    if 'nccl' in low or 'cuda' in low or 'device-side assert' in low:
        return 'nccl'
    if 'timeout' in name.lower() or 'timed out' in low:
        return 'timeout'
    return 'internal'


# -- native identity ---------------------------------------------------------

def verify_native(root):
    """Exact reviewed inventory/pins of the checksummed lane module directory."""
    root = Path(root)
    files = list(root.rglob('*'))
    if any(path.is_symlink() for path in files):
        raise ValueError('native bundle contains a symlink')
    actual = {str(path.relative_to(root)) for path in files if path.is_file()}
    if actual != set(NATIVE_PINS) | {'SHA256SUMS'}:
        raise ValueError('native bundle file inventory differs')
    expected = ''.join(f'{sha}  {name}\n' for name, sha in sorted(NATIVE_PINS.items()))
    if (root / 'SHA256SUMS').read_text() != expected:
        raise ValueError('native manifest differs from reviewed pins')
    for name, sha in NATIVE_PINS.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
            raise ValueError('native source pin differs: ' + name)
    return dict(NATIVE_PINS)


# -- outputs -----------------------------------------------------------------

def write_json(path, value):
    """Atomic exclusive create: a concurrent/killed writer never leaves partial JSON."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError('refusing to overwrite ' + path.name)
    temp = path.with_name(path.name + '.tmp')
    if temp.exists() or temp.is_symlink():
        raise FileExistsError('refusing to overwrite ' + temp.name)
    text = json.dumps(scrub_obj(value), indent=2, allow_nan=False)
    with open(temp, 'x') as handle:
        handle.write(text + '\n')
    os.link(temp, path)  # exclusive publication; retain the staging artifact


def _read_json(path):
    try:
        with open(path) as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


# -- frozen verdict (pure) ---------------------------------------------------

_POSITIVE = lambda value: type(value) in (int, float) and math.isfinite(value) and value > 0


def _phase_tp_failures(record):
    failures = []
    if record.get('tp_world_size') != 2:
        failures.append('tp world size is not 2')
    if record.get('pp_world_size') != 1:
        failures.append('pp world size is not 1')
    if record.get('pp_channel') is not False:
        failures.append('TP=2/PP=1 built a PP channel')
    if not isinstance(record.get('stream_basis'), str) or not record.get('stream_basis'):
        failures.append('lane stream basis not recorded')
    if record.get('warm') is not True or record.get('tp_warmed') is not True:
        failures.append('lane communicators were not warmed')
    iterations = record.get('iterations')
    if type(iterations) is not int or iterations < 1:
        failures.append('invalid timing iteration count')
    cases = record.get('allreduce')
    counts = [case.get('chunk_tokens') for case in cases] if isinstance(cases, list) else []
    if counts != list(DEFAULT_TOKEN_COUNTS):
        failures.append('all-reduce chunk_tokens coverage differs')
    else:
        for case in cases:
            name = 'all-reduce[%r]' % case.get('chunk_tokens')
            if case.get('exact') is not True or case.get('timed_exact') is not True:
                failures.append(name + ': result not exact')
            if case.get('hidden_width') != record.get('hidden_width'):
                failures.append(name + ': hidden width differs')
            samples = case.get('ms')
            if (not isinstance(samples, list) or len(samples) != iterations
                    or not all(_POSITIVE(value) for value in samples)):
                failures.append(name + ': timing samples invalid')
            for key in ('median_ms', 'min_ms', 'max_ms'):
                if not _POSITIVE(case.get(key)):
                    failures.append(name + ': %s invalid' % key)
    overlap = record.get('overlap')
    if not isinstance(overlap, dict):
        failures.append('overlap evidence missing')
    else:
        if overlap.get('allreduce_exact') is not True or overlap.get('matmul_finite') is not True:
            failures.append('overlap arithmetic not proven')
        if overlap.get('iterations') != iterations:
            failures.append('overlap iteration count differs')
        if overlap.get('matmul') != [MATMUL_SIZE, MATMUL_SIZE, MATMUL_SIZE]:
            failures.append('overlap matmul shape differs')
        for key in ('wall_ms', 'matmul_stream_ms', 'allreduce_stream_ms'):
            if not _POSITIVE(overlap.get(key)):
                failures.append('overlap %s invalid' % key)
        if not isinstance(overlap.get('basis'), str) or 'not simultaneous' not in overlap['basis']:
            failures.append('overlap basis not labelled')
    scope = record.get('scope')
    if not isinstance(scope, dict):
        failures.append('tp_scope evidence missing')
    else:
        inside = scope.get('inside')
        if not isinstance(inside, dict):
            inside = {}
        if inside.get('is_lane') is not True or inside.get('installed_is_lane') is not True:
            failures.append('tp_scope did not install the lane communicator')
        if inside.get('lane_collectives_delta') != 1:
            failures.append('lane communicator did not serve the in-scope collective')
        if inside.get('exact') is not True:
            failures.append('in-scope collective result not exact')
        if inside.get('stock_calls') != 0 or scope.get('stock_calls') != 0:
            failures.append('stock device communicator was touched')
        if scope.get('restored') is not True:
            failures.append('stock device communicator not restored')
        if scope.get('reentrant_refused') is not True:
            failures.append('nested tp_scope not refused')
    return failures


def _phase_pp_failures(record):
    failures = []
    rank = record.get('rank')
    expected_role = {0: 'sender', 1: 'receiver'}.get(rank)
    if record.get('label') != LABELS['pp']:
        failures.append('synthetic PP label missing')
    if record.get('topology') != PP_TOPOLOGY:
        failures.append('synthetic PP topology not recorded')
    if record.get('tp_world_size') != 1:
        failures.append('tp world size is not 1')
    if record.get('pp_world_size') != 2:
        failures.append('pp world size is not 2')
    if record.get('role') != expected_role:
        failures.append('PP role is not %r' % expected_role)
    if record.get('warm') is not True or record.get('warmed') is not True:
        failures.append('PP channel was not warmed')
    if record.get('bind_before_live') is not True:
        failures.append('consumer binding was not made before a live ticket')
    cases = record.get('cases')
    counts = [case.get('num_tokens') for case in cases] if isinstance(cases, list) else []
    if counts != [min(value, record.get('max_tokens') or 0)
                  for value in DEFAULT_TOKEN_COUNTS]:
        failures.append('PP num_tokens coverage differs')
    else:
        for case in cases:
            name = 'pp[%r]' % case.get('num_tokens')
            if case.get('released') is not True:
                failures.append(name + ': slot not released')
            if rank == 1:
                if case.get('payload_equal') is not True:
                    failures.append(name + ': payload mismatch')
                if case.get('header_equal') is not True:
                    failures.append(name + ': header round trip differs')
                if case.get('consumer_event_query') is not True:
                    failures.append(name + ': consumer event not queried complete')
                if case.get('consumer_finite') is not True:
                    failures.append(name + ': consumer result nonfinite')
                if case.get('receipt_written') is not True:
                    failures.append(name + ': consumer receipt not written')
            else:
                if case.get('payload_sent') is not True:
                    failures.append(name + ': payload not sent')
                if case.get('header_sent') is not True:
                    failures.append(name + ': header not sent')
                if case.get('receipt_queries') is not True:
                    failures.append(name + ': receipt lacks queried completion')
                if case.get('receipt_verified') is not True:
                    failures.append(name + ': consumer receipt not verified')
    ownership = record.get('ownership')
    if not isinstance(ownership, dict):
        failures.append('ownership proof evidence missing')
    elif rank == 0:
        for key in ('wrong_seq_refused', 'wrong_route_refused', 'wrong_kind_refused',
                    'stayed_unreleased', 'correct_release'):
            if ownership.get(key) is not True:
                failures.append('ownership %s not proven' % key)
    else:
        if ownership.get('consumer_event_query') is not True:
            failures.append('ownership consumer event not queried')
    mismatch = record.get('mismatch')
    if not isinstance(mismatch, dict):
        failures.append('header mismatch evidence missing')
    else:
        if mismatch.get('deadlock') is not False or mismatch.get('bounded') is not True:
            failures.append('header mismatch was not bounded (possible deadlock)')
        if rank == 1:
            if mismatch.get('refused') is not True:
                failures.append('header mismatch was not refused')
            if mismatch.get('quarantined') is not True:
                failures.append('header mismatch did not quarantine the channel')
        else:
            if mismatch.get('completed') is not True:
                failures.append('sender send did not complete')
        if (type(mismatch.get('txn_seq')) is not int
                or mismatch.get('sent_seq') != mismatch.get('expected_seq', -2) + 1):
            failures.append('header mismatch injection not recorded')
    owned = record.get('bytes_owned')
    if (type(owned) is not int or owned <= 0
            or record.get('bytes_owned_matches') is not True
            or owned != record.get('bytes_owned_expected')):
        failures.append('bytes_owned inventory missing')
    return failures


def _observed_bad_arithmetic(record):
    # Explicit mismatches survive later interruption; absent measurements do not
    # turn a bounded timeout into a numerical FAIL.
    for case in record.get('allreduce') or []:
        if case.get('exact') is False or case.get('timed_exact') is False:
            return True
    for case in record.get('cases') or []:
        if any(case.get(k) is False for k in ('payload_equal', 'header_equal', 'consumer_finite')):
            return True
    overlap = record.get('overlap') or {}
    scope = (record.get('scope') or {}).get('inside') or {}
    return (any(overlap.get(k) is False for k in ('allreduce_exact', 'matmul_finite'))
            or scope.get('exact') is False)


def _record_verdict(record):
    """Frozen per-rank rules. Returns (verdict, reasons)."""
    errors = record.get('errors') or []
    hard = [err for err in errors if err.get('kind') not in INCONCLUSIVE_KINDS]
    if hard:
        return 'FAIL', ['%s: %s' % (err.get('kind'), err.get('message', '')) for err in hard]
    if _observed_bad_arithmetic(record):
        return 'FAIL', ['observed arithmetic mismatch']
    status = record.get('status')
    capability = record.get('capability')
    if isinstance(capability, (list, tuple)) and len(capability) == 2:
        capability = [capability[0], capability[1]]
    else:
        capability = None
    if status in ('missing', 'timeout'):
        return 'INCONCLUSIVE', ['rank not measured (status %r)' % (status,)]
    if status == 'capability':
        return 'INCONCLUSIVE', ['capability %r != [8, 6]' % (capability,)]
    if status == 'interrupted':
        return 'INCONCLUSIVE', ['interrupted']
    if status != 'ok':
        # A rank that failed before it could measure anything reports why; the
        # absent capability is a symptom, never the whole reason.
        inconclusive = [err for err in errors if err.get('kind') in INCONCLUSIVE_KINDS]
        if inconclusive:
            return 'INCONCLUSIVE', ['%s: %s' % (err.get('kind'), err.get('message', ''))
                                    for err in inconclusive]
        return 'INCONCLUSIVE', ['not measured (status %r)' % (status,)]
    inconclusive = [err for err in errors if err.get('kind') in INCONCLUSIVE_KINDS]
    if inconclusive:
        return 'INCONCLUSIVE', ['%s: %s' % (err.get('kind'), err.get('message', ''))
                                for err in inconclusive]
    if capability != [8, 6]:
        return 'INCONCLUSIVE', ['capability %r != [8, 6]' % (capability,)]
    if record.get('cleanup') != 'ok':
        return 'FAIL', ['cleanup not proven']
    phase = record.get('phase')
    if phase == 'tp':
        failures = _phase_tp_failures(record)
    elif phase == 'pp':
        failures = _phase_pp_failures(record)
    else:
        return 'INCONCLUSIVE', ['unknown phase %r' % (phase,)]
    if failures:
        return 'FAIL', failures
    return 'PASS', []


def verdict(records):
    """Frozen whole-run rules: PASS only if both phases PASS on both ranks.

    A hard failure on any measured rank outranks an inconclusive or missing
    rank. Missing/duplicate phase-rank coverage is INCONCLUSIVE unless a
    recorded hard failure already decides the run.
    """
    if not isinstance(records, (list, tuple)) or not records:
        return {'verdict': 'INCONCLUSIVE', 'reasons': ['no phase records']}
    results = []
    coverage_bad = False
    pairs = []
    for record in records:
        if not isinstance(record, dict):
            coverage_bad = True
            continue
        pairs.append((record.get('phase'), record.get('rank')))
        results.append(_record_verdict(record))
    expected = {(phase, rank) for phase in PHASES for rank in RANKS}
    if len(pairs) != len(expected) or set(pairs) != expected:
        coverage_bad = True
    if not coverage_bad:
        pp = {r['rank']: r for r in records if r['phase'] == 'pp'}
        if all(r.get('status') == 'ok' for r in pp.values()):
            for key in ('txn_seq', 'sent_seq', 'expected_seq'):
                if pp[0].get('mismatch', {}).get(key) != pp[1].get('mismatch', {}).get(key):
                    results.append(('FAIL', ['PP mismatch trial identities differ across ranks']))
            for key in ('hidden_width', 'max_tokens'):
                if len({r.get(key) for r in records}) != 1:
                    results.append(('FAIL', ['phase geometry differs across ranks']))
    reasons = []
    if coverage_bad:
        reasons.append('phase/rank inventory incomplete or duplicated')
    for record, (outcome, why) in zip([r for r in records if isinstance(r, dict)], results):
        reasons.extend('%s rank %r: %s' % (record.get('phase', '?'), record.get('rank'), item)
                       for item in why)
    for _, why in results[len(pairs):]:
        reasons.extend(why)
    outcomes = [outcome for outcome, _ in results]
    if 'FAIL' in outcomes:
        overall = 'FAIL'
    elif 'INCONCLUSIVE' in outcomes or coverage_bad:
        overall = 'INCONCLUSIVE'
    elif all(outcome == 'PASS' for outcome in outcomes) and not coverage_bad:
        overall = 'PASS'
    else:  # pragma: no cover - _record_verdict only returns those three
        overall = 'INCONCLUSIVE'
    return {'verdict': overall, 'reasons': reasons}


# -- record helpers ----------------------------------------------------------

def _new_record(phase, rank, device, hidden_width, max_tokens, iterations):
    return {'schema': 1, 'phase': phase, 'rank': rank, 'label': LABELS[phase],
            'device': device, 'status': 'new', 'capability': None,
            'hidden_width': hidden_width, 'max_tokens': max_tokens,
            'iterations': iterations, 'torch': None, 'vllm': None,
            'cleanup': None, 'errors': []}


def _finish_record(record, out):
    path = Path(out) / ('%s-rank-%d.json' % (record['phase'], record['rank']))
    if path.exists() or path.is_symlink():
        return
    write_json(path, record)


def _rank_path(out, phase, rank):
    return Path(out) / ('%s-rank-%d.json' % (phase, rank))


def _synthetic_record(phase, rank, kind, message):
    record = _new_record(phase, rank, rank, None, None, None)
    record['status'] = kind
    record['errors'].append(_error(kind, message))
    return record


# -- in-container rank work --------------------------------------------------

def _imports(native):
    """Import torch, vLLM groups and the lane modules inside the pinned image."""
    import torch
    import vllm
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import (
        init_distributed_environment, initialize_model_parallel,
        get_tp_group, get_pp_group, destroy_model_parallel,
        destroy_distributed_environment)

    sys.path.insert(0, str(Path(native).resolve()))
    import lane_comm
    import e3_types
    lane_file = getattr(lane_comm, '__file__', None)
    types_file = getattr(e3_types, '__file__', None)
    if not lane_file or Path(lane_file).resolve() != (Path(native) / 'lane_comm.py').resolve():
        raise ValueError('imported lane module differs from pinned directory')
    if not types_file or Path(types_file).resolve() != (Path(native) / 'e3_types.py').resolve():
        raise ValueError('imported contract module differs from pinned directory')
    return torch, vllm, PyNcclCommunicator, init_distributed_environment, \
        initialize_model_parallel, get_tp_group, get_pp_group, \
        destroy_model_parallel, destroy_distributed_environment, lane_comm, e3_types


def _pattern(torch, rows, width, device, seed=0):
    """Deterministic bf16 payload: integers 0..199 are exact in bf16."""
    total = int(rows) * int(width)
    index = torch.arange(total, dtype=torch.int32, device=device)
    return ((index + seed) % 200).to(torch.bfloat16).reshape(int(rows), int(width))


def _event_ms(torch, stream, fn, iterations, expected):
    """Event-timed median/min/max over ``iterations`` after warmup."""
    with torch.cuda.stream(stream):
        for _ in range(WARMUP_ITERS):
            fn()
    stream.synchronize()
    samples = []
    exact = True
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record()
            result = fn()
            end.record()
        stream.synchronize()
        samples.append(float(start.elapsed_time(end)))
        exact = bool(torch.equal(result, expected)) and exact
    return {'ms': samples, 'timed_exact': exact, 'total_ms': float(sum(samples)),
            'median_ms': float(statistics.median(samples)),
            'min_ms': float(min(samples)), 'max_ms': float(max(samples))}


def _phase_tp(torch, lane_comm, get_tp_group, get_pp_group, rank, device, stream,
              hidden_width, max_tokens, iterations, token_counts, record):
    tp_group = get_tp_group()
    pp_group = get_pp_group()
    record['tp_world_size'] = int(tp_group.world_size)
    record['pp_world_size'] = int(pp_group.world_size)
    record['stream_basis'] = ('explicit torch.cuda.Stream stands in for the green '
                              'prefill stream')
    record['hidden_width_basis'] = 'owner-declared HC width; reference text_config hc_count=4 * hidden_size=2560'
    comms = lane_comm.PrefillCommunicators.build(
        tp_group, pp_group, device, stream, max_tokens=max_tokens,
        hidden_width=hidden_width, dtype=torch.bfloat16, torch_module=torch)
    record['pp_channel'] = comms.pp is not None
    # The module creates its warm-up tensor on the caller's current stream and
    # consumes it on the lane stream; keep both streams identical so the probe
    # cannot race the module's own warm-up.
    with torch.cuda.stream(stream):
        record['warm'] = bool(comms.warm(stream))
        record['tp_warmed'] = bool(comms.tp.warmed)
    cases = []
    for token_count in token_counts:
        # Build the input on the lane stream too: the module copies it into an
        # owned slot on that stream, so creation and copy are ordered.
        with torch.cuda.stream(stream):
            x = _pattern(torch, token_count, hidden_width, device, rank)
            out = comms.tp.all_reduce(x)
        stream.synchronize()
        if out is None:
            raise lane_comm.LaneCommError('all-reduce returned nothing')
        expected = (_pattern(torch, token_count, hidden_width, device, 0)
                    + _pattern(torch, token_count, hidden_width, device, 1))
        exact = bool(torch.equal(out, expected))
        timing = _event_ms(torch, stream, lambda: comms.tp.all_reduce(x), iterations, expected)
        case = {'chunk_tokens': int(token_count), 'hidden_width': int(hidden_width),
                'dtype': 'bfloat16', 'exact': exact, 'collectives': int(comms.tp.collectives)}
        case.update(timing)
        cases.append(case)
    record['allreduce'] = cases

    # Overlap: interleaved host submissions; lane-local spans only.
    size = MATMUL_SIZE
    a = torch.randn((size, size), device=device, dtype=torch.bfloat16)
    b = torch.randn((size, size), device=device, dtype=torch.bfloat16)
    mat_out = torch.empty_like(a)
    x = _pattern(torch, cases[-1]['chunk_tokens'], hidden_width, device, rank)
    expected_overlap = (_pattern(torch, cases[-1]['chunk_tokens'], hidden_width, device, 0)
                        + _pattern(torch, cases[-1]['chunk_tokens'], hidden_width, device, 1))
    mat_stream = torch.cuda.Stream(device=device)
    torch.cuda.synchronize(device)
    with torch.cuda.stream(mat_stream):
        for _ in range(WARMUP_ITERS):
            torch.matmul(a, b, out=mat_out)
    with torch.cuda.stream(stream):
        for _ in range(WARMUP_ITERS):
            comms.tp.all_reduce(x)
    mat_stream.synchronize()
    stream.synchronize()
    mat_start = torch.cuda.Event(enable_timing=True)
    mat_end = torch.cuda.Event(enable_timing=True)
    lane_start = torch.cuda.Event(enable_timing=True)
    lane_end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(mat_stream):
        mat_start.record()
    with torch.cuda.stream(stream):
        lane_start.record()
    wall_start = time.perf_counter()
    overlap_checks = []
    for _ in range(iterations):
        with torch.cuda.stream(mat_stream):
            torch.matmul(a, b, out=mat_out)
        with torch.cuda.stream(stream):
            reduced = comms.tp.all_reduce(x)
            # Device-side scalar check per result, without a host wait in the loop.
            overlap_checks.append((reduced == expected_overlap).all())
    with torch.cuda.stream(mat_stream):
        mat_end.record()
    with torch.cuda.stream(stream):
        lane_end.record()
    mat_stream.synchronize()
    stream.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    record['overlap'] = {
        'matmul': [size, size, size], 'dtype': 'bfloat16',
        'chunk_tokens': int(cases[-1]['chunk_tokens']), 'hidden_width': int(hidden_width),
        'iterations': iterations, 'wall_ms': float(wall_ms),
        'matmul_stream_ms': float(mat_start.elapsed_time(mat_end)),
        'allreduce_stream_ms': float(lane_start.elapsed_time(lane_end)),
        'matmul_finite': bool(torch.isfinite(mat_out).all().item()),
        'allreduce_exact': all(bool(check.item()) for check in overlap_checks),
        'basis': ('interleaved host submissions; lane-local spans, not '
                  'simultaneous-kernel proof')}
    del a, b, mat_out

    # tp_scope routing with an explicit spy on the stock communicator.
    class _StockSpy:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def all_reduce(self, *args, **kwargs):
            self.calls += 1
            return self.inner.all_reduce(*args, **kwargs)

    stock = tp_group.device_communicator
    spy = _StockSpy(stock)
    tp_group.device_communicator = spy
    inside = {}
    restored = False
    reentrant_refused = False
    try:
        with lane_comm.tp_scope(tp_group, comms.tp) as installed:
            inside['is_lane'] = tp_group.device_communicator is comms.tp
            inside['installed_is_lane'] = installed is comms.tp
            try:
                with lane_comm.tp_scope(tp_group, comms.tp):
                    pass
            except lane_comm.LaneCommError:
                reentrant_refused = True
            with torch.cuda.stream(stream):
                scoped = _pattern(torch, 64, hidden_width, device)
                before = int(comms.tp.collectives)
                group_out = tp_group.all_reduce(scoped)
            stream.synchronize()
            inside['lane_collectives_delta'] = int(comms.tp.collectives) - before
            inside['stock_calls'] = int(spy.calls)
            inside['exact'] = bool(torch.equal(group_out, scoped * 2))
        restored = tp_group.device_communicator is spy
    finally:
        tp_group.device_communicator = stock
    record['scope'] = {'inside': inside, 'restored': restored,
                       'reentrant_refused': reentrant_refused,
                       'stock_calls': int(spy.calls)}
    record['describe'] = comms.describe()


def _write_receipt(out, phase, seq, value):
    write_json(Path(out) / ('%s-receipt-%d.json' % (phase, seq)), value)


def _wait_for_receipt(out, phase, seq, timeout):
    path = Path(out) / ('%s-receipt-%d.json' % (phase, seq))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = _read_json(path)
        if value is not None:
            return value
        time.sleep(0.01)
    return None


def _bind_and_warm(torch, lane_comm, e3_types, channel, stream, record):
    lane = e3_types.LaneId(PEER_WORKER_RANK, e3_types.LaneRole.PREFILL,
                           PEER_STORAGE_EPOCH, PEER_CONTEXT_ID)
    channel.bind_consumer(lane, ROUTE_GENERATION)
    record['bind_before_live'] = (channel.consumer_binding == (lane, ROUTE_GENERATION)
                                  and not channel.tickets)
    # The module's warm-up tensor is allocated on the caller's current stream and
    # then sent/received on the lane stream; make those the same stream.
    with torch.cuda.stream(stream):
        record['warm'] = bool(channel.warm())
    record['warmed'] = bool(channel.warmed)
    return lane


def _decode_receipt(lane_comm, e3_types, data):
    """Reconstruct the typed receipt from plain values; no payload is carried."""
    txn = e3_types.TransactionId(**data['transaction'])
    lane_data = data['lane']
    lane = e3_types.LaneId(lane_data['worker_rank'],
                           e3_types.LaneRole(lane_data['role']),
                           lane_data['storage_epoch'], lane_data['context_id'])
    proof_data = data['proof']
    operation = e3_types.OperationTicket(
        txn, data['worker_rank'], lane, proof_data['op_seq'],
        e3_types.DependencyRole(proof_data['dependency_role']))
    proof = e3_types.CompletionProof(
        operation, e3_types.CompletionKind(proof_data['kind']),
        proof_data['completion_seq'])
    return lane_comm.PPConsumerReceipt(data['stage'], data['seq'],
                                       data['route_generation'], proof)


def _refusal(fn):
    try:
        fn()
        return False
    except Exception as exc:  # the module's own refusal, not an inference
        return type(exc).__name__ == 'LaneCommError'


def _phase_pp(torch, lane_comm, e3_types, PyNcclCommunicator, get_tp_group,
              get_pp_group, rank, device, stream, out, hidden_width, max_tokens,
              token_counts, poll_timeout, receipt_timeout, record):
    tp_group = get_tp_group()
    pp_group = get_pp_group()
    record['tp_world_size'] = int(tp_group.world_size)
    record['pp_world_size'] = int(pp_group.world_size)
    record['topology'] = PP_TOPOLOGY
    record['stream_basis'] = ('explicit torch.cuda.Stream stands in for the green '
                              'prefill stream')
    is_sender = not bool(pp_group.is_last_rank)
    peer = int(pp_group.rank_in_group) + (1 if is_sender else -1)
    pp_nccl = PyNcclCommunicator(group=pp_group.cpu_group, device=device)
    channel = lane_comm.PrefillPPChannel(
        pp_nccl, device, stream, max_tokens=max_tokens, hidden_width=hidden_width,
        dtype=torch.bfloat16, is_sender=is_sender, peer_rank=peer, torch_module=torch)
    record['role'] = 'sender' if is_sender else 'receiver'
    record['peer_rank'] = peer
    lane = _bind_and_warm(torch, lane_comm, e3_types, channel, stream, record)
    cases = []
    keepalive = []
    ownership: dict[str, object] = {'consumer_event_query': True}
    mismatch = None
    seq = 0
    for token_count in token_counts:
        n = min(int(token_count), int(max_tokens))
        txn = e3_types.TransactionId(1, SYNTHETIC_UID, 0, 0, n, seq)
        case: dict[str, object] = {'num_tokens': n, 'seq': seq, 'stage': 0}
        if is_sender:
            with torch.cuda.stream(stream):
                payload = _pattern(torch, n, hidden_width, device, seq)
                ticket = channel.send(payload, txn, 0, seq)
            # Keep the source alive until its slot copy is known complete; the
            # module copies on the lane stream, not the allocating stream.
            keepalive.append(payload)
            receipt_data = _wait_for_receipt(out, 'pp', seq, receipt_timeout)
            if receipt_data is None:
                case.update(payload_equal=None, header_equal=None, released=False,
                            receipt_verified=False, timeout=True)
                cases.append(case)
                mismatch = {'deadlock': False, 'bounded': True, 'completed': bool(ticket.event.query()),
                            'txn_seq': txn.txn_seq,
                            'note': 'peer receipt did not arrive within the bound'}
                record['error_kind_hint'] = 'timeout'
                record['errors'].append(_error('timeout', 'PP receipt wait timed out'))
                break
            if receipt_data.get('status') != 'ok':
                case.update(payload_equal=None, header_equal=None, released=False,
                            receipt_verified=False, timeout=True)
                cases.append(case)
                record['errors'].append(_error(
                    'timeout', 'peer reported ' + str(receipt_data.get('status'))))
                break
            if receipt_data.get('consumer_event_query') is not True:
                raise lane_comm.LaneCommError('receipt lacks a queried consumer event')
            receipt = _decode_receipt(lane_comm, e3_types, receipt_data)
            if seq == 0:
                proof = receipt.proof
                seq_receipt = lane_comm.PPConsumerReceipt(0, seq + 1, ROUTE_GENERATION, proof)
                route_receipt = lane_comm.PPConsumerReceipt(0, seq, ROUTE_GENERATION + 1, proof)
                bad_kind_operation = e3_types.OperationTicket(
                    txn, proof.ticket.worker_rank, proof.ticket.lane,
                    proof.ticket.op_seq, e3_types.DependencyRole.HANDOFF)
                bad_kind_proof = e3_types.CompletionProof(
                    bad_kind_operation, e3_types.CompletionKind.GPU_EVENT,
                    proof.completion_seq)
                outstanding = int(channel.outstanding())
                wrong_seq = _refusal(lambda: channel.release_send(ticket, seq_receipt))
                wrong_route = _refusal(lambda: channel.release_send(ticket, route_receipt))
                wrong_kind = _refusal(
                    lambda: lane_comm.PPConsumerReceipt(0, seq, ROUTE_GENERATION,
                                                        bad_kind_proof))
                stayed = (int(channel.outstanding()) == outstanding
                          and ticket.state != 'retired')
                ownership.update(wrong_seq_refused=wrong_seq,
                                 wrong_route_refused=wrong_route,
                                 wrong_kind_refused=wrong_kind,
                                 stayed_unreleased=stayed,
                                 outstanding=outstanding)
            released = bool(channel.release_send(ticket, receipt))
            ownership['correct_release'] = ownership.get('correct_release', True) and released
            case.update(payload_sent=True, header_sent=True, released=released,
                        receipt_verified=True, receipt_queries=receipt_data.get(
                            'consumer_event_query'))
        else:
            rticket = channel.post_receive(txn, 0, seq, n)
            view = None
            timed_out = True
            deadline = time.monotonic() + poll_timeout
            while time.monotonic() < deadline:
                view = channel.poll_receive(rticket)
                if view is not None:
                    timed_out = False
                    break
                time.sleep(0.002)
            if view is None:
                case.update(payload_equal=None, header_equal=None, released=False,
                            consumer_event_query=False, receipt_written=False, timeout=True)
                cases.append(case)
                _write_receipt(out, 'pp', seq, {'status': 'timeout', 'seq': seq})
                record['errors'].append(_error('timeout', 'PP poll_receive timed out'))
                break
            with torch.cuda.stream(stream):
                expected = _pattern(torch, n, hidden_width, device, seq)
                case['payload_equal'] = bool(torch.equal(view, expected))
            actual = [int(value) for value in channel.header_host[rticket.slot].tolist()]
            case['header_equal'] = (actual == lane_comm.header_values(txn, 0, seq, n))
            with torch.cuda.stream(stream):
                consumed = view.float().sum()
                consumer_event = torch.cuda.Event(enable_timing=False)
                consumer_event.record(stream)
            stream.synchronize()
            queried = bool(consumer_event.query())
            case['consumer_event_query'] = queried
            case['consumer_finite'] = bool(torch.isfinite(consumed).item())
            if not queried:
                raise lane_comm.LaneCommError('consumer event incomplete after synchronization')
            channel.consumer_done(rticket, consumer_event)
            released = bool(channel.release_receive(rticket))
            if not released:
                raise lane_comm.LaneCommError('receive slot did not retire')
            # The receipt carries plain values only; completion was queried here.
            _write_receipt(out, 'pp', seq, {
                'schema': 1, 'kind': 'pp-consumer-receipt', 'status': 'ok',
                'transaction': {'engine_epoch': txn.engine_epoch,
                                'request_uid': txn.request_uid,
                                'request_generation': txn.request_generation,
                                'chunk_start': txn.chunk_start,
                                'chunk_end': txn.chunk_end,
                                'txn_seq': txn.txn_seq},
                'stage': 0, 'seq': seq, 'route_generation': ROUTE_GENERATION,
                'worker_rank': lane.worker_rank,
                'lane': {'worker_rank': lane.worker_rank, 'role': lane.role.value,
                         'storage_epoch': lane.storage_epoch,
                         'context_id': lane.context_id},
                'proof': {'kind': e3_types.CompletionKind.PP_CONSUMER.value,
                          'dependency_role': e3_types.DependencyRole.HANDOFF.value,
                          'op_seq': seq, 'completion_seq': seq},
                'consumer_event_query': queried})
            ownership['consumer_event_query'] = ownership.get('consumer_event_query', True) and queried
            case.update(released=released, receipt_written=True)
        cases.append(case)
        seq += 1

    # Header mismatch: the sender tags seq+1 while the receiver posts seq. The
    # module's own poll_receive rule must quarantine; polling is bounded.
    expected_seq = seq
    n = 1
    txn = e3_types.TransactionId(1, SYNTHETIC_UID, 0, 0, n, 900 + expected_seq)
    mismatch_record = {'expected_seq': expected_seq, 'sent_seq': expected_seq + 1,
                       'txn_seq': txn.txn_seq,
                       'deadlock': False, 'bounded': True}
    if is_sender:
        with torch.cuda.stream(stream):
            payload = _pattern(torch, n, hidden_width, device, seq)
            ticket = channel.send(payload, txn, 0, expected_seq + 1)
        keepalive.append(payload)
        channel.stream.synchronize()
        mismatch_record['completed'] = bool(ticket.event.query())
        mismatch_record['outstanding_after'] = int(channel.outstanding())
    else:
        rticket = channel.post_receive(txn, 0, expected_seq, n)
        started = time.monotonic()
        refused = False
        timed_out = False
        while time.monotonic() - started < poll_timeout:
            try:
                view = channel.poll_receive(rticket)
            except lane_comm.LaneCommError:
                refused = True
                break
            if view is not None:
                break
            time.sleep(0.002)
        else:
            timed_out = True
        elapsed = time.monotonic() - started
        mismatch_record.update(
            refused=refused, timed_out=timed_out, quarantined=channel.quarantined is not None,
            elapsed_s=float(elapsed), deadlock=False, bounded=(elapsed <= poll_timeout))
        if timed_out:
            record['errors'].append(_error('timeout', 'header-mismatch poll did not complete'))
    record['cases'] = cases
    record['ownership'] = ownership
    record['mismatch'] = mismatch_record
    record['describe'] = channel.describe()
    record['bytes_owned'] = int(channel.bytes_owned())
    expected_bytes = channel.slots * (max_tokens * hidden_width * 2 + lane_comm.HEADER_LEN * 8)
    record['bytes_owned_expected'] = int(expected_bytes)
    record['bytes_owned_matches'] = record['bytes_owned'] == record['bytes_owned_expected']
    if not record['bytes_owned_matches']:
        record['errors'].append(_error('inventory', 'bytes_owned differs from the buffer inventory'))
    record['outstanding_after'] = int(channel.outstanding())


# -- rank process entry ------------------------------------------------------

def _run_rank(phase, rank, out, native, device, hidden_width, max_tokens, iterations,
              token_counts, poll_timeout, receipt_timeout):
    record = _new_record(phase, rank, device, hidden_width, max_tokens, iterations)
    torch = None
    destroy = None
    try:
        (torch, vllm, PyNcclCommunicator, init_distributed_environment,
         initialize_model_parallel, get_tp_group, get_pp_group,
         destroy_model_parallel, destroy_distributed_environment,
         lane_comm, e3_types) = _imports(native)
        record['torch'] = getattr(torch, '__version__', None)
        record['vllm'] = getattr(vllm, '__version__', None)
        record['torch_cuda'] = getattr(getattr(torch, 'version', None), 'cuda', None)
        record['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
        torch.cuda.set_device(device)
        capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
        record['capability'] = [capability[0], capability[1]]
        if capability != (8, 6):
            record['status'] = 'capability'
            record['errors'].append(_error(
                'capability', 'device capability %r != (8, 6)' % (capability,)))
            return record
        destroy = (destroy_model_parallel, destroy_distributed_environment)
        world_size = len(RANKS)
        rendezvous = Path(out) / ('%s-rendezvous' % phase)
        init_distributed_environment(
            world_size=world_size, rank=rank, local_rank=device, backend='nccl',
            distributed_init_method='file://' + str(rendezvous))
        tp_size, pp_size = ((2, 1) if phase == 'tp' else (1, 2))
        initialize_model_parallel(tensor_model_parallel_size=tp_size,
                                  pipeline_model_parallel_size=pp_size)
        stream = torch.cuda.Stream(device=device)
        record['distributed'] = {'world_size': world_size, 'rank': rank,
                                 'tp_size': tp_size, 'pp_size': pp_size}
        if phase == 'tp':
            _phase_tp(torch, lane_comm, get_tp_group, get_pp_group, rank, device,
                      stream, hidden_width, max_tokens, iterations, token_counts, record)
        else:
            _phase_pp(torch, lane_comm, e3_types, PyNcclCommunicator, get_tp_group,
                      get_pp_group, rank, device, stream, out, hidden_width,
                      max_tokens, token_counts, poll_timeout, receipt_timeout, record)
        record['status'] = 'ok'
    except KeyboardInterrupt:
        record['status'] = 'interrupted'
        record['errors'].append(_error('interrupted', 'interrupted during measurement'))
    except BaseException as exc:  # record every failure, never a silent exit
        record['status'] = 'error'
        record['errors'].append(_error(classify_exception(exc), scrub_text(exc)))
    finally:
        cleanup = 'not-started'
        try:
            if torch is not None:
                torch.cuda.synchronize()
            if destroy is not None:
                errors = []
                for cleanup_fn in destroy:
                    try:
                        cleanup_fn()
                    except Exception as exc:
                        errors.append(scrub_text(exc))
                if errors:
                    raise RuntimeError('distributed cleanup: ' + '; '.join(errors))
            cleanup = 'ok'
        except BaseException as exc:
            cleanup = 'failed'
            record['errors'].append(_error('cleanup', 'cleanup failed: ' + scrub_text(exc)))
        record['cleanup'] = cleanup
    return record


def _rank_entry(phase, rank, out, native, device, hidden_width, max_tokens,
                iterations, token_counts, poll_timeout, receipt_timeout):
    record = _run_rank(phase, rank, out, native, device, hidden_width, max_tokens,
                       iterations, token_counts, poll_timeout, receipt_timeout)
    _finish_record(record, out)


# -- phase orchestration -----------------------------------------------------

def _launch_phase(phase, args, devices):
    """Run one phase as two spawned worker processes; bound and reap our own."""
    import torch.multiprocessing as torch_mp

    context = torch_mp.get_context('spawn')
    processes = []
    for rank in RANKS:
        process = context.Process(target=_rank_entry, args=(
            phase, rank, str(args.out), str(args.native), int(devices[rank]),
            int(args.hidden_width), int(args.max_tokens), int(args.iters),
            tuple(int(value) for value in DEFAULT_TOKEN_COUNTS),
            float(args.poll_timeout), float(args.receipt_timeout)))
        process.start()
        processes.append((rank, process))
    deadline = time.monotonic() + float(args.phase_timeout)
    for _, process in processes:
        process.join(max(0.1, deadline - time.monotonic()))
    records = []
    for rank, process in processes:
        if process.is_alive():
            process.terminate()
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join(5)
            record = _synthetic_record(phase, rank, 'timeout',
                                       'worker did not finish within the phase bound')
            _finish_record(record, args.out)
        else:
            record = _read_json(_rank_path(args.out, phase, rank))
            if record is None:
                record = _synthetic_record(phase, rank, 'missing', 'worker record missing')
                _finish_record(record, args.out)
            elif record.get('phase') != phase or record.get('rank') != rank:
                record = _synthetic_record(phase, rank, 'internal',
                                           'worker record identity differs')
                _finish_record(record, args.out)
        record = _with_process_exit(record, process.exitcode)
        records.append(record)
    return records


def _with_process_exit(record, exitcode):
    # A published good record cannot hide a later interpreter/allocator abort.
    if exitcode != 0:
        record = dict(record, errors=list(record.get('errors') or []))
        record['errors'].append(_error('interrupted', 'worker exitcode %r' % exitcode))
    record['process_exitcode'] = exitcode
    return record


# -- command line ------------------------------------------------------------

def parse_devices(text):
    devices = [int(part) for part in str(text).split(',')]
    if (len(devices) != len(RANKS) or any(device < 0 for device in devices)
            or len(set(devices)) != len(devices)):
        raise ValueError('exactly two unique nonnegative device ordinals required')
    return devices


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, help='checksummed E3_NATIVE directory')
    parser.add_argument('--out', type=Path, help='fresh writable output directory')
    parser.add_argument('--phase', choices=('tp', 'pp', 'both'), default='both',
                        help='phase to run (default both)')
    parser.add_argument('--devices', default=DEFAULT_DEVICES,
                        help='two visible CUDA ordinals, comma separated (default %s)'
                             % DEFAULT_DEVICES)
    parser.add_argument('--hidden-width', type=int,
                        help='hc*hidden width of the model (required; no default)')
    parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help='PP channel max_tokens (default %d)' % DEFAULT_MAX_TOKENS)
    parser.add_argument('--iters', type=int, default=DEFAULT_ITERS,
                        help='timed iterations (default %d)' % DEFAULT_ITERS)
    parser.add_argument('--poll-timeout', type=float, default=DEFAULT_POLL_TIMEOUT,
                        help='bounded PP poll timeout seconds (default %g)'
                             % DEFAULT_POLL_TIMEOUT)
    parser.add_argument('--receipt-timeout', type=float, default=DEFAULT_RECEIPT_TIMEOUT,
                        help='bounded receipt wait seconds (default %g)'
                             % DEFAULT_RECEIPT_TIMEOUT)
    parser.add_argument('--phase-timeout', type=float, default=DEFAULT_PHASE_TIMEOUT,
                        help='per-phase worker bound seconds (default %g)'
                             % DEFAULT_PHASE_TIMEOUT)
    parser.add_argument('--plan', action='store_true',
                        help='print the CPU-only plan and exit')
    return parser


def _plan(args, devices):
    return {
        'schema': 1, 'phases': list(PHASES) if args.phase == 'both' else [args.phase],
        'labels': dict(LABELS), 'pp_topology': PP_TOPOLOGY,
        'devices': devices, 'hidden_width': args.hidden_width,
        'max_tokens': args.max_tokens, 'iterations': args.iters,
        'token_counts': list(DEFAULT_TOKEN_COUNTS), 'matmul': [MATMUL_SIZE] * 3,
        'dtype': 'bfloat16', 'warmup_iterations': WARMUP_ITERS,
        'poll_timeout_s': args.poll_timeout, 'receipt_timeout_s': args.receipt_timeout,
        'phase_timeout_s': args.phase_timeout,
        'routes': ['vllm.distributed.parallel_state.init_distributed_environment',
                   'vllm.distributed.parallel_state.initialize_model_parallel'],
        'native_pins': dict(NATIVE_PINS), 'gpu_required': True}


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        devices = parse_devices(args.devices)
    except ValueError:
        parser.error('--devices requires exactly two unique nonnegative ordinals')
    if (args.max_tokens != DEFAULT_MAX_TOKENS or args.iters < 1
            or any(not math.isfinite(v) or v <= 0 for v in
                   (args.poll_timeout, args.receipt_timeout, args.phase_timeout))):
        parser.error('--max-tokens must be 2048 and iterations/time bounds positive finite')
    if args.plan:
        print(json.dumps(_plan(args, devices), indent=2))
        return 0
    if args.native is None or args.out is None or args.hidden_width is None:
        parser.error('--native, --out and --hidden-width are required together')
    if args.hidden_width < 1:
        parser.error('--hidden-width must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    if any(p.name.startswith(('tp-', 'pp-', 'summary.json', 'probe.claim'))
           for p in args.out.iterdir()):
        parser.error('refusing existing probe artifacts')
    write_json(args.out / 'probe.claim', {'schema': 1, 'phases': args.phase})
    try:
        verify_native(args.native)
    except Exception as exc:
        record = _synthetic_record('tp', 0, 'config', 'native identity refused: ' + scrub_text(exc))
        records = [record]
        for phase in PHASES:
            for rank in RANKS:
                if phase == 'tp' and rank == 0:
                    continue
                records.append(_synthetic_record(phase, rank, 'missing',
                                                 'native identity refused'))
        result = verdict(records)
        _write_summary(args, records, result, list(PHASES))
        print('E3_LANE_COMM_PROBE %s %s' % (result['verdict'], '; '.join(result['reasons'][:5])),
              flush=True)
        return {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[result['verdict']]
    phases = list(PHASES) if args.phase == 'both' else [args.phase]
    records = []
    for phase in phases:
        records.extend(_launch_phase(phase, args, devices))
    result = verdict(records)
    _write_summary(args, records, result, phases)
    line = 'E3_LANE_COMM_PROBE %s labels=%s,%s' % (
        result['verdict'], LABELS['tp'], LABELS['pp'])
    if result['reasons']:
        line += ' ' + '; '.join(result['reasons'][:5])
    print(line, flush=True)
    return {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[result['verdict']]


def _write_summary(args, records, result, phases=None):
    phases = phases if phases is not None else [record.get('phase') for record in records]
    summary = {'schema': 1, 'verdict': result['verdict'], 'reasons': result['reasons'],
               'phases': list(phases), 'labels': dict(LABELS),
               'pp_topology': PP_TOPOLOGY,
               'requested_devices': parse_devices(args.devices),
               'hidden_width': args.hidden_width, 'max_tokens': args.max_tokens,
               'iterations': args.iters, 'token_counts': list(DEFAULT_TOKEN_COUNTS),
               'native_pins': dict(NATIVE_PINS),
               'process_exits': {f"{r.get('phase')}-rank-{r.get('rank')}": r.get('process_exitcode') for r in records},
               'records': ['%s-rank-%d.json' % (record.get('phase'), record.get('rank'))
                           for record in records]}
    write_json(Path(args.out) / 'summary.json', summary)


if __name__ == '__main__':
    raise SystemExit(main())
