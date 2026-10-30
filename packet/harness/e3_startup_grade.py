#!/usr/bin/env python3
"""Frozen startup/capture grade for the q17b decode-only diagnostic.

This grades only the narrow startup scope: native green helper identity, the
four-worker full-route start, the split decode graph-bank capture, inventory,
allocator/device observations and authenticated status.  It does NOT qualify
replay placement, output equality, speed, the full E3 package or armability.
No torch, CUDA, docker, engine or network access: inputs are owner artifacts.

A contradiction is FAIL.  Missing or explicitly unknown evidence is
INCONCLUSIVE (never PASS); FAIL dominates a concurrent INCONCLUSIVE.  The
requested decode split is the declared 44; the driver's rounded allocation is
reported separately and is not treated as a second request.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

RANKS = (0, 1, 2, 3)
STORAGE_EPOCH = 1
REQUESTED_DECODE_SMS = 44
ALLOCATOR_KEYS = (
    'allocated_current_bytes', 'reserved_current_bytes',
    'allocated_peak_bytes', 'reserved_peak_bytes',
    'inactive_split_current_bytes', 'inactive_split_peak_bytes',
)
STREAM_CONTRACT = {
    'bank': 'decode-split',
    'stream_label': 'decode-split-stream',
    'context_label': 'decode-split-context',
}
NULL = object()


def _int(value):
    return type(value) is int


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _digest(value):
    return (type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


class Grade:
    def __init__(self):
        self.failures = []
        self.missing = []
        self.measured = {'requested_decode_sms': REQUESTED_DECODE_SMS,
                         'decode_sms': None, 'prefill_sms': None, 'total_sms': None,
                         'basis': None}
        self.ranks = {}
        # Kept outside the serialized rank record so a rank with an unknown
        # memory observation still contributes its measured split to the
        # cross-rank consistency check.
        self.split_by_rank = {}
        self.memory_intervals = []
        self.observed_splits = []

    def fail(self, reason):
        if reason not in self.failures:
            self.failures.append(reason)

    def absent(self, reason):
        if reason not in self.missing:
            self.missing.append(reason)

    def verdict(self):
        if self.failures:
            return 'FAIL'
        if self.missing:
            return 'INCONCLUSIVE'
        return 'PASS'

    def record(self):
        return {'schema': 1,
                'scope': ('startup/capture only: native green helper load, four-worker start, '
                          'full-route status and split graph-bank capture, inventory/memory, authenticated status; '
                          'not replay placement, output equality, speed, full E3 or armability'),
                'verdict': self.verdict(),
                'failures': self.failures,
                'missing': self.missing,
                'measured': self.measured,
                'ranks': self.ranks,
                'requested_decode_sms_is_unchanged': self.measured['requested_decode_sms'] == REQUESTED_DECODE_SMS,
                'measured_decode_sms_recorded_separately': self.measured['decode_sms'] is not None,
                'full_package_armable': False}


def _snapshot(grade, record, label):
    if type(record) is not dict or not record:
        grade.absent(label + ': missing')
        return None
    if record.get('schema') != 1:
        grade.fail(label + ': schema is not 1')
    if 'quarantined' not in record:
        grade.fail(label + ': quarantine state is missing')
    elif record['quarantined'] is not False:
        grade.fail(label + ': controller quarantined')
    snapshot = copy.deepcopy(record.get('snapshot'))
    if type(snapshot) is not dict:
        grade.absent(label + ': no snapshot')
        return None
    if snapshot.get('route') != 'full':
        grade.fail(label + ': route is not full')
    if snapshot.get('storage_epoch') != STORAGE_EPOCH:
        grade.fail(label + ': storage epoch is not 1')
    if not _int(snapshot.get('route_generation')) or snapshot['route_generation'] < 0:
        grade.fail(label + ': route generation missing or invalid')
    rows = snapshot.get('ranks')
    if type(rows) is not list or len(rows) != len(RANKS):
        grade.fail(label + ': worker coverage is not exactly four ranks')
        return snapshot
    seen = {}
    for row in rows:
        if type(row) is not dict or not _int(row.get('rank')) or row['rank'] not in RANKS:
            grade.fail(label + ': invalid worker rank')
            continue
        if row['rank'] in seen:
            grade.fail(label + ': duplicate worker rank %d' % row['rank'])
            continue
        seen[row['rank']] = row
    if set(seen) != set(RANKS):
        grade.fail(label + ': missing worker rank(s) ' + repr(sorted(set(RANKS) - set(seen))))
    snapshot['_rank_map'] = seen
    return snapshot


def _evidence(grade, evidence, library_sha256):
    if type(evidence) is not dict:
        grade.absent('declaration evidence is not an object')
        return None, None
    helper = evidence.get('helper_source_sha256')
    if not _digest(helper):
        grade.fail('declaration helper source sha is invalid')
    shas = evidence.get('generated_source_shas')
    build_only = evidence.get('build_only_source_shas')
    if type(shas) is not dict or not shas or not all(_digest(v) for v in shas.values()):
        grade.fail('declaration generated source shas are missing or invalid')
    if type(build_only) is not dict or not build_only or not all(_digest(v) for v in build_only.values()):
        grade.fail('declaration build-only source shas are missing or invalid')
    if not _digest(library_sha256):
        grade.fail('declared compiled helper library sha is invalid')
    pins = evidence.get('pins')
    if type(pins) is not dict or not pins or not all(_digest(v) for v in pins.values()):
        grade.fail('declaration transform pins are missing or invalid')
    return helper, library_sha256


def _rank(grade, rank, row, snapshot, helper, library):
    label = 'rank %d' % rank
    inv, ctr = row.get('inventory'), row.get('counters')
    if type(inv) is not dict or type(ctr) is not dict:
        grade.absent(label + ': inventory/counters missing')
        return
    if inv.get('mode') != 'decode-only-probe' or ctr.get('mode') != 'decode-only-probe':
        grade.fail(label + ': diagnostic mode differs')
    for key, expected in (('route', snapshot.get('route')),
                          ('route_generation', snapshot.get('route_generation')),
                          ('storage_epoch', snapshot.get('storage_epoch'))):
        if inv.get(key) != expected or ctr.get(key) != expected:
            grade.fail(label + ': %s differs from the bound snapshot' % key)
    if ctr.get('state') != 'ready':
        grade.fail(label + ': state is not ready')
    if 'quarantined' not in ctr:
        grade.fail(label + ': worker quarantine state is missing')
    elif ctr['quarantined'] is not None:
        grade.fail(label + ': worker quarantined')
    if (not _int(ctr.get('in_flight')) or ctr['in_flight'] != 0
            or ctr.get('sampling_pending') is not False
            or ctr.get('transition_pending') is not False):
        grade.fail(label + ': worker is not idle at startup status')
    if not _int(ctr.get('transition_count')) or ctr['transition_count'] != 0:
        grade.fail(label + ': transition occurred during startup/capture')
    if ctr.get('invalidation_reasons') != []:
        grade.fail(label + ': capture interval was invalidated')

    if inv.get('helper_source_sha256') != helper:
        grade.fail(label + ': loaded helper source differs from declaration')
    if inv.get('library_sha256') != library:
        grade.fail(label + ': loaded helper library differs from declaration')
    binding = inv.get('binding_identity')
    if type(binding) is not dict:
        grade.fail(label + ': binding identity missing')
    else:
        if binding.get('library_sha256') != library:
            grade.fail(label + ': binding library sha differs')
        if binding.get('helper_source_sha256') != helper:
            grade.fail(label + ': binding helper source differs')
        if binding.get('abi') != 1:
            grade.fail(label + ': green helper ABI is not 1')
    if type(inv.get('identity_scope')) is not str or not inv['identity_scope']:
        grade.fail(label + ': identity scope is not declared')

    split = inv.get('split')
    if type(split) is not dict:
        grade.fail(label + ': split inventory missing')
    else:
        if split.get('requested_decode_sms') != REQUESTED_DECODE_SMS:
            grade.fail(label + ': requested decode SMS is not the declared 44')
        decode, prefill, total = (split.get('decode_sms'), split.get('prefill_sms'),
                                  split.get('total_sms'))
        if not all(_int(v) and v >= 0 for v in (decode, prefill, total)) or total <= 0 or decode <= 0:
            grade.fail(label + ': measured split counts are invalid')
        elif decode < REQUESTED_DECODE_SMS or prefill <= 0:
            grade.fail(label + ': measured allocation is smaller than requested')
        elif decode + prefill != total:
            grade.fail(label + ': measured split counts do not sum to total')
        else:
            grade.measured.update(decode_sms=decode, prefill_sms=prefill,
                                  total_sms=total, basis=split.get('basis'))
            grade.ranks.setdefault('measured_decode_sms', {})[str(rank)] = decode
            grade.split_by_rank[rank] = (decode, prefill, total)
            grade.observed_splits.append((decode, prefill, total))
        if split.get('basis') != 'measured':
            grade.fail(label + ': split membership is not measured')
        member = split.get('membership')
        if type(member) is not dict:
            grade.fail(label + ': measured membership record missing')
        elif not (_int(member.get('decode_seen')) and _int(member.get('prefill_seen'))
                  and _int(member.get('total')) and member.get('decode_seen') == decode
                  and member.get('prefill_seen') == prefill and member.get('total') == total
                  and member.get('disjoint') is True and member.get('ok') is True
                  and type(member.get('rounds')) is list and member['rounds']):
            grade.fail(label + ': measured SM membership is inconsistent')
        if type(member) is dict:
            rounds = member.get('rounds')
            if (type(rounds) is not list or len(rounds) != 2
                    or not all(type(xs) is list and xs and _int(limit)
                               and all(_int(x) and 0 < x <= limit for x in xs)
                               and max(xs) <= limit <= sum(xs)
                               for xs, limit in zip(rounds, (decode, prefill)))):
                grade.fail(label + ': malformed membership rounds')
        helper_record = split.get('helper')
        if type(helper_record) is not dict:
            grade.fail(label + ': split helper identity missing')
        else:
            if helper_record.get('library_sha256') != library:
                grade.fail(label + ': split helper library differs')
            if helper_record.get('helper_source_sha256') != helper:
                grade.fail(label + ': split helper source differs')
            if helper_record.get('abi') != 1:
                grade.fail(label + ': split helper ABI is not 1')

    descriptors = inv.get('descriptors')
    captured = inv.get('descriptors_captured')
    if type(descriptors) is not list or not descriptors:
        grade.fail(label + ': split descriptor inventory is empty')
        descriptors = []
    labels = []
    for desc in descriptors:
        if type(desc) is not dict:
            grade.fail(label + ': malformed split descriptor')
            continue
        name = desc.get('label')
        if type(name) is not str or not name or name in labels:
            grade.fail(label + ': duplicate or missing descriptor label')
            continue
        labels.append(name)
        if desc.get('cg_mode') != 'full':
            grade.fail(label + ': descriptor is not FULL')
        if not _int(desc.get('num_tokens')) or desc['num_tokens'] <= 0:
            grade.fail(label + ': descriptor token count is invalid')
        if desc.get('num_reqs') is not None and (not _int(desc['num_reqs']) or desc['num_reqs'] < 0):
            grade.fail(label + ': descriptor request count is invalid')
        reqs, tokens, uniform = desc.get('num_reqs'), desc.get('num_tokens'), desc.get('uniform_token_count')
        if uniform is not None and (not _int(uniform) or uniform != 1):
            grade.fail(label + ': descriptor is not uniform decode')
        if _int(reqs) and _int(tokens) and not 0 < reqs <= tokens:
            grade.fail(label + ': descriptor request count exceeds tokens or is empty')
        if name != f'full:tokens={tokens}:reqs={reqs}:uniform={uniform}':
            grade.fail(label + ': descriptor label disagrees with fields')
    if not _int(captured) or captured <= 0:
        grade.fail(label + ': no descriptor was captured')
    elif captured != len(descriptors):
        grade.fail(label + ': captured count differs from descriptor inventory')
    if inv.get('raw_graphs_retained') is not True:
        grade.fail(label + ': raw split graphs were not retained')

    measured = inv.get('split_pool_bytes')
    capture = inv.get('capture_memory')
    if measured is None:
        # The probe leaves an unmeasured delta unknown rather than guessing;
        # unknown is missing evidence, not a contradiction.
        grade.absent(label + ': split pool bytes unknown')
    elif not _int(measured) or measured < 0:
        grade.fail(label + ': split pool bytes invalid')
    if capture is None:
        grade.absent(label + ': capture memory record missing')
    elif type(capture) is not dict:
        grade.fail(label + ': capture memory record malformed')
    else:
        before, after, delta = (capture.get('free_before_bytes'),
                                capture.get('free_after_bytes'),
                                capture.get('delta_bytes'))
        if any(v is None for v in (before, after, delta)):
            grade.absent(label + ': capture memory sample unknown')
        elif not all(_int(v) and v >= 0 for v in (before, after, delta)) or before < after:
            grade.fail(label + ': capture memory samples are invalid')
        elif delta != before - after:
            grade.fail(label + ': capture memory delta is inconsistent')
        elif measured is not None and delta != measured:
            grade.fail(label + ': capture memory delta differs from split pool bytes')
        if type(capture.get('basis')) is not str or not capture['basis']:
            grade.fail(label + ': capture memory basis is not declared')

    checks = inv.get('capture_stream_checks')
    if type(checks) is not list or not checks:
        grade.fail(label + ': capture stream checks are empty')
    else:
        checked = []
        for check in checks:
            if type(check) is not dict:
                grade.fail(label + ': malformed capture stream check')
                continue
            name = check.get('descriptor')
            if name in checked or name not in labels:
                grade.fail(label + ': capture stream check has an unknown or duplicate descriptor')
                continue
            checked.append(name)
            for key, value in STREAM_CONTRACT.items():
                if check.get(key) != value:
                    grade.fail(label + ': capture stream check %s differs' % key)
            if check.get('vllm_matches_owner') is not True or check.get('torch_matches_owner') is not True:
                grade.fail(label + ': capture stream agreement failed')
            if type(check.get('basis')) is not str or not check['basis']:
                grade.fail(label + ': capture stream check basis is not declared')
        if set(checked) != set(labels) or len(checked) != len(labels):
            grade.fail(label + ': capture stream checks do not cover every descriptor exactly once')

    memory = row.get('memory')
    if type(memory) is not dict:
        grade.absent(label + ': allocator/device memory snapshot missing')
        return
    if memory.get('schema') != 1:
        grade.fail(label + ': memory snapshot schema is not 1')
    if memory.get('synchronized_by_snapshot') is not False:
        grade.fail(label + ': memory snapshot claims synchronization')
    if type(memory.get('peak_scope')) is not str or not memory['peak_scope']:
        grade.fail(label + ': allocator peak scope is not declared')
    if type(memory.get('device_scope')) is not str or not memory['device_scope']:
        grade.fail(label + ': device sample scope is not declared')
    if type(memory.get('errors')) is not list or memory['errors']:
        grade.absent(label + ': memory query errors or missing error record')
    wall, end = memory.get('wall_s'), memory.get('sample_end_wall_s')
    mono, mono_end = memory.get('monotonic_ns'), memory.get('sample_end_monotonic_ns')
    if not all(_number(x) and x > 0 for x in (wall, end)) or not all(_int(x) and x > 0 for x in (mono, mono_end)):
        grade.absent(label + ': memory timestamps missing or invalid')
    elif end < wall or mono_end < mono:
        grade.fail(label + ': memory timestamps reversed')
    else:
        grade.memory_intervals.append((wall, end))
    allocator = memory.get('allocator')
    if type(allocator) is not dict:
        grade.absent(label + ': allocator observation missing')
    else:
        for key in ALLOCATOR_KEYS:
            value = allocator.get(key)
            if value is None:
                grade.absent(label + ': allocator %s unknown' % key)
            elif not _int(value) or value < 0:
                grade.fail(label + ': allocator %s invalid' % key)
        if (_int(allocator.get('allocated_peak_bytes'))
                and _int(allocator.get('allocated_current_bytes'))
                and allocator['allocated_peak_bytes'] < allocator['allocated_current_bytes']):
            grade.fail(label + ': allocated current exceeds lifetime peak')
        if (_int(allocator.get('reserved_peak_bytes'))
                and _int(allocator.get('reserved_current_bytes'))
                and allocator['reserved_peak_bytes'] < allocator['reserved_current_bytes']):
            grade.fail(label + ': reserved current exceeds lifetime peak')
        if (_int(allocator.get('inactive_split_peak_bytes'))
                and _int(allocator.get('inactive_split_current_bytes'))
                and allocator['inactive_split_peak_bytes'] < allocator['inactive_split_current_bytes']):
            grade.fail(label + ': inactive split current exceeds lifetime peak')
        for smaller, larger in (('allocated_current_bytes', 'reserved_current_bytes'),
                                ('allocated_peak_bytes', 'reserved_peak_bytes'),
                                ('inactive_split_current_bytes', 'reserved_current_bytes'),
                                ('inactive_split_peak_bytes', 'reserved_peak_bytes')):
            if all(_int(allocator.get(k)) for k in (smaller, larger)) and allocator[smaller] > allocator[larger]:
                grade.fail(label + ': contradictory allocator bounds ' + smaller)
    free, total = memory.get('device_free_bytes'), memory.get('device_total_bytes')
    if free is None or total is None:
        grade.absent(label + ': device memory point sample unknown')
    elif not _int(free) or not _int(total) or total <= 0 or not 0 <= free <= total:
        grade.fail(label + ': device memory point sample invalid')
    if type(capture) is dict and _int(total):
        if any(_int(capture.get(k)) and capture[k] > total for k in ('free_before_bytes', 'free_after_bytes')):
            grade.fail(label + ': capture free memory exceeds device total')
    grade.ranks.setdefault('rank', {})[str(rank)] = {
        'route_generation': ctr.get('route_generation'),
        'descriptors_captured': captured,
        'split_pool_bytes': measured,
        'measured_split': ({'decode_sms': split.get('decode_sms'),
                            'prefill_sms': split.get('prefill_sms'),
                            'total_sms': split.get('total_sms')}
                           if type(split) is dict else None),
        'allocator': ({k: allocator.get(k) for k in ALLOCATOR_KEYS}
                      if type(allocator) is dict else None),
        'device_free_bytes': free,
        'device_total_bytes': total,
        'peak_scope': memory.get('peak_scope'),
        'device_scope': memory.get('device_scope'),
        'allocator_unknown': [k for k in ALLOCATOR_KEYS
                              if type(allocator) is not dict or allocator.get(k) is None],
        'device_sample_unknown': free is None or total is None,
        'memory_errors': memory.get('errors') if type(memory.get('errors')) is list else 'invalid',
    }


def _samples(grade, samples):
    if samples is None:
        grade.absent('timestamped device samples missing')
        return
    rows = samples
    if type(rows) is not list or not rows:
        grade.absent('timestamped device samples empty')
        return
    last = rows[-1]
    owned_stop = (type(last) is dict and last.get('kind') == 'stopped'
                  and last.get('clock') == 'unix_realtime_seconds'
                  and _number(last.get('wall_s')))
    if not owned_stop:
        grade.absent('device sampler has no owned clean timestamped stop')
    valid = 0
    previous_end = None
    first_start = None
    for row in rows:
        if type(row) is dict and row.get('kind') == 'stopped' and row is last:
            continue
        if type(row) is not dict or row.get('error') is not None:
            grade.absent('device sampler reports an error or malformed row')
            continue
        gpus = row.get('gpus') if type(row) is dict else None
        if type(gpus) is not dict or set(gpus) != {'0', '1', '2', '3'}:
            grade.absent('device sample lacks complete GPU coverage')
            continue
        # Only bracketed real-time samples count as timestamped evidence.
        if (row.get('clock') != 'unix_realtime_seconds'
                or not _number(row.get('start_s')) or not _number(row.get('end_s'))
                or row['start_s'] > row['end_s']):
            grade.absent('device sample has no valid clock bracket')
            continue
        if (not _number(row.get('elapsed_s')) or row['elapsed_s'] < 0
                or abs((row['end_s'] - row['start_s']) - row['elapsed_s']) > 0.1):
            grade.absent('device sampler clock bracket is inconsistent')
            continue
        if previous_end is not None and row['start_s'] < previous_end:
            grade.fail('device samples are not ordered')
        previous_end = row['end_s']
        if first_start is None:
            first_start = row['start_s']
        if owned_stop and last['wall_s'] < row['end_s']:
            grade.fail('device sampler stop precedes its samples')
        valid += 1
        for gpu in gpus.values():
            if (type(gpu) is not dict or not _int(gpu.get('used_mib'))
                    or not _int(gpu.get('total_mib')) or gpu['total_mib'] <= 0
                    or not 0 <= gpu['used_mib'] <= gpu['total_mib']):
                grade.fail('device sample is invalid')
                return
    if first_start is not None and owned_stop:
        if any(start < first_start or end > last['wall_s'] for start, end in grade.memory_intervals):
            grade.absent('worker memory observations are outside the device sampling interval')
    if not valid:
        grade.absent('no valid four-GPU device sample')
    grade.ranks['device_samples'] = {'valid': valid, 'total_rows': len(rows),
                                     'claimed_peak': False, 'owned_clean_stop': owned_stop}


def grade(open_record, status_record, evidence, library_sha256, samples=None):
    result = Grade()
    opened = _snapshot(result, open_record, 'open')
    status = _snapshot(result, status_record, 'status')
    helper, library = _evidence(result, evidence, library_sha256)
    if opened is None or status is None:
        result.absent('open and status snapshots are both required')
        _samples(result, samples)
        return result.record()
    if (type(open_record.get('run_id')) is not str or not open_record['run_id']
            or status_record.get('run_id') != open_record.get('run_id')):
        result.fail('status run id differs from open')
    if status_record.get('action') != 'status':
        result.fail('status action differs')
    if status_record.get('command_seq') != 1:
        result.fail('status is not the first command after open')
    if open_record.get('action') != 'open' or open_record.get('command_seq') != 0:
        result.fail('open record is not the initial command')
    if not _int(open_record.get('engine_epoch')) or open_record['engine_epoch'] <= 0:
        result.fail('open engine epoch is invalid')
    if status_record.get('engine_epoch') != open_record.get('engine_epoch'):
        result.fail('engine incarnation changed between open and status')
    if open_record.get('paused') is not False or status_record.get('paused') is not False:
        result.fail('engine was paused during startup/capture')
    for key in ('route', 'route_generation', 'storage_epoch'):
        if open_record.get('snapshot', {}).get(key) != status_record.get('snapshot', {}).get(key):
            result.fail('snapshot %s changed between open and status' % key)
    if helper is None or library is None:
        result.absent('declaration identity unavailable')
        _samples(result, samples)
        return result.record()
    descriptor_sets = []
    for phase, snap in (('open', opened), ('status', status)):
        ranks = snap.get('_rank_map')
        for rank in RANKS:
            row = ranks.get(rank) if type(ranks) is dict else None
            if row is None:
                result.fail(phase + ' lacks rank %d' % rank)
                continue
            _rank(result, rank, row, snap, helper, library)
            inv = row.get('inventory')
            if type(inv) is dict and type(inv.get('descriptors')) is list:
                descriptor_sets.append(json.dumps(inv['descriptors'], sort_keys=True))
    if len(set(descriptor_sets)) > 1:
        result.fail('descriptor inventory differs across ranks or startup snapshots')
    if len(set(result.observed_splits)) > 1:
        result.fail('measured split allocation differs between ranks')
    _samples(result, samples)
    return result.record()


def _load(path, label, result):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        result.absent(label + ': unreadable (' + type(exc).__name__ + ')')
        return NULL
    return value


def _load_samples(path, result):
    """The sampler writes one JSON object per line; blank lines are ignored."""
    try:
        text = Path(path).read_text()
    except OSError as exc:
        result.absent('samples: unreadable (' + type(exc).__name__ + ')')
        return NULL
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            result.absent('samples: unreadable (malformed line)')
            return NULL
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open', required=True)
    parser.add_argument('--status', required=True)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--library-sha256', required=True)
    parser.add_argument('--samples', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    probe = Grade()
    opened = _load(args.open, 'open', probe)
    status = _load(args.status, 'status', probe)
    evidence = _load(args.evidence, 'evidence', probe)
    samples = _load_samples(args.samples, probe)
    if any(value is NULL for value in (opened, status, evidence, samples)):
        record = probe.record()
        record['measured'] = {'requested_decode_sms': REQUESTED_DECODE_SMS,
                              'decode_sms': None, 'prefill_sms': None,
                              'total_sms': None, 'basis': None}
        record['requested_decode_sms_is_unchanged'] = True
        record['measured_decode_sms_recorded_separately'] = False
    else:
        record = grade(opened, status, evidence, args.library_sha256, samples)
    try:
        with args.out.open('x') as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write('\n')
    except OSError as exc:
        print(json.dumps({'error': 'result-write-failed-' + type(exc).__name__}), file=sys.stderr)
        return 2
    print(record['verdict'], 'startup/capture-only',
          'requested_decode_sms=%d' % record['measured']['requested_decode_sms'],
          'measured_decode_sms=%s' % record['measured']['decode_sms'])
    if record['failures']:
        print('FAIL ' + '; '.join(record['failures'][:8]), file=sys.stderr)
    if record['missing']:
        print('MISSING ' + '; '.join(record['missing'][:8]), file=sys.stderr)
    return {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[record['verdict']]


if __name__ == '__main__':
    raise SystemExit(main())
