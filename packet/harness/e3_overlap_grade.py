#!/usr/bin/env python3
"""Pure grading for the q18a real E3 overlap run (offline; no engine control).

The module has three jobs, all side-effect free:

1. ``target_definition`` resolves the frozen 80/80 (or W2) denominator policy.
   The repository contains two different target statements and they must not be
   silently mixed:

   * ``s5-w2-target`` -- STALL-PLAN.md section 1 "Target and reference":
     aggregate decode during a fresh prefill >= 95 % of W1 *every stream*,
     worst-stream gap p99 within 20 % of the stream's W1 p99, fresh TTFT
     <= 30.2 s (+10 %).
   * ``s5-overlap-80-80`` -- STALL-PLAN.md:290 (operator targets, 25 Oct
     ~04:00): decode share during a fresh prefill ~80 % of solo (was 95 %),
     prefill ~80 % of solo; 80/80 is a target line, not a gate.

   Neither label is guessed.  The caller must name one, and any numeric
   override that differs from the named policy is reported as a conflict.

2. ``grade`` checks the overlap evidence: accepted prefill segments with real
   copy/attention/STATE_WRITE/PP proofs, decode replay progress *during* those
   segments, useful completed prefill work, all-rank ownership/epoch binding,
   sparse/dense/serialized route coverage, an explicitly reported fallback, the
   frozen denominators (actual full-GPU W1 baseline and matched split solo),
   the unchanged 512 MiB reserve under measured pressure, and exact cleanup.

3. ``candidate_readiness`` refuses an unarmable package (the current full12
   package is ``armable:false``), a fabricated/unreviewed readiness receipt,
   ``require_native_ready`` bypasses, and any launcher pin taken from the
   environment instead of the trusted bundled ``identity_declare`` pin.

Verdict rule (frozen): FAIL for known data/ownership/context contradictions
and hard-policy failures; INCONCLUSIVE for incomplete controls, missing
scope/route/clock proof or unbounded measurement. The 80/80 target shortfall
is reported separately from evidence qualification. Invalid intervals are
retained, never filtered.

usage (composition helper):
  e3_overlap_grade.py --w1 W1.json --overlap OVERLAP.json \
      --samples SAMPLES.timed.jsonl [--cleanup CLEANUP.json] \
      [--readiness READINESS.json] [--generation GENERATION.json] \
      [--package-manifest SHA256SUMS] [--reserve-mib N] --out GRADE.json
"""
import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path
import sys

SCHEMA = 1
MAX_SHARE = 4.0
DEFAULT_RESERVE_MIB = 512

TARGET_POLICIES = {
    's5-w2-target': {
        'source': ('STALL-PLAN.md section "1. Target and reference" (clean '
                   'reference 24 Oct 09:04 UTC, overlay-v7): W2 aggregate '
                   'decode >=95 % of W1 every stream; worst-stream gap p99 '
                   'within 20 % of W1 p99; fresh TTFT <=30.2 s (+10 %).'),
        'decode_share_min': 0.95,
        'decode_denominator': 'w1_full',
        'gap_p99_ratio_max': 1.20,
        'fresh_ttft_s_max': 30.2,
        'prefill_share_min': None,
        'prefill_denominator': None,
        'per_stream_decode_required': True,
        'effect_envelope': {'min_effect': 0.05, 'min_segment_ns': 1,
                            'min_decode_arrivals': 1, 'min_useful_tokens': 1},
    },
    's5-overlap-80-80': {
        'source': ('STALL-PLAN.md:290 (operator targets 25 Oct ~04:00): '
                   '"decode share during a fresh prefill ~80 % of solo (was '
                   '95 %), prefill ~80 % of solo; aim for the maximum, 80/80 '
                   'is a target line, not a gate."'),
        'decode_share_min': 0.80,
        'decode_denominator': 'w1_full',
        'gap_p99_ratio_max': None,
        'fresh_ttft_s_max': None,
        'prefill_share_min': 0.80,
        'prefill_denominator': 'prefill_split',
        'per_stream_decode_required': False,
        'effect_envelope': {'min_effect': 0.05, 'min_segment_ns': 1,
                            'min_decode_arrivals': 1, 'min_useful_tokens': 1},
    },
}

DENOMINATOR_NOTE = (
    'decode denominator is the actual full-GPU decode-only W1 baseline measured '
    'in this queue, never the split/slice baseline; the matched split solo is '
    'reported alongside. The 80/80 wording says only "of solo": this grader '
    'uses matched split solo for the prefill ratio and reports the full-GPU '
    'prefill solo separately; the target label must state the chosen basis.')
CONFLICT_NOTE = (
    'source definitions conflict: STALL-PLAN section 1 declares 95 % of W1 '
    'with a 30.2 s TTFT bound, while STALL-PLAN:290 declares an 80/80 target '
    'line "of solo". An explicit target_label (s5-w2-target or '
    's5-overlap-80-80) is required; no default is guessed.')

PROOF_COMPLETION = ('queried', 'queried_event')
ROUTE_LABELS = ('sparse', 'dense', 'serialized')
RANK_SET = {0, 1, 2, 3}


def _is_int(value):
    return type(value) is int


def _is_num(value):
    return type(value) in (int, float) and math.isfinite(float(value))


def _sha256_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _num(value):
    return float(value) if _is_num(value) else None


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


# ---------------------------------------------------------------------------
# target policy
# ---------------------------------------------------------------------------

def target_definition(label, declared=None):
    """Resolve one frozen policy without guessing.

    ``declared`` is the optional evidence/manifest ``target`` object.  A
    numeric field that differs from the named policy is a source conflict and
    refuses the grade instead of silently preferring either statement.
    """
    declared = declared if type(declared) is dict else {}
    problems = []
    if not isinstance(label, str) or not label:
        return {'ok': False, 'label': None, 'problems': ['target_label_required'],
                'policies': sorted(TARGET_POLICIES), 'conflict': CONFLICT_NOTE,
                'declared': declared, 'denominator_note': DENOMINATOR_NOTE}
    policy = TARGET_POLICIES.get(label)
    if policy is None:
        return {'ok': False, 'label': label,
                'problems': ['unknown_target_label'],
                'policies': sorted(TARGET_POLICIES), 'conflict': CONFLICT_NOTE,
                'declared': declared, 'denominator_note': DENOMINATOR_NOTE}
    for key in ('decode_share_min', 'prefill_share_min', 'gap_p99_ratio_max',
                'fresh_ttft_s_max'):
        if key not in declared or declared[key] is None:
            continue
        if policy[key] is None or not _is_num(declared[key]) \
                or abs(float(declared[key]) - policy[key]) > 1e-9:
            problems.append('target_definition_conflict:' + key)
    if 'decode_denominator' in declared and \
            declared['decode_denominator'] != policy['decode_denominator']:
        problems.append('target_definition_conflict:decode_denominator')
    if 'prefill_denominator' in declared and policy['prefill_denominator'] \
            is not None and declared['prefill_denominator'] \
            != policy['prefill_denominator']:
        problems.append('target_definition_conflict:prefill_denominator')
    resolved = dict(policy)
    resolved.update({'ok': not problems, 'label': label,
                     'problems': sorted(set(problems)),
                     'conflict': CONFLICT_NOTE if problems else None,
                     'denominator_note': DENOMINATOR_NOTE,
                     'declared': declared})
    return resolved


# ---------------------------------------------------------------------------
# segment proofs
# ---------------------------------------------------------------------------

def _fail(failures, text):
    if text not in failures:
        failures.append(text)


def _miss(missing, text):
    if text not in missing:
        missing.append(text)


def _int_field(obj, key, minimum, missing, where):
    if type(obj) is not dict or key not in obj or not _is_int(obj.get(key)) \
            or obj[key] < minimum:
        _miss(missing, 'missing_or_invalid:' + where + '.' + key)
        return None
    return obj[key]


def _require_true(obj, key, missing, where):
    if type(obj) is not dict or obj.get(key) is not True:
        _miss(missing, 'unproved:' + where + '.' + key)
        return False
    return True


def _check_rank_proofs(rank_row, segment, policy, failures, missing, owner):
    """Validate one rank's copy/attention/state/PP/replay/ownership proof."""
    rank = rank_row.get('rank')
    where = 'rank%d.seg%d' % (rank, segment.get('segment_seq', -1))
    ownership = rank_row.get('ownership') or {}
    for key in ('worker_incarnation', 'engine_epoch', 'storage_epoch',
                'route_generation'):
        if key not in ownership:
            _miss(missing, 'missing_proof:' + where + '.ownership.' + key)
        elif key in owner:
            if ownership[key] != owner[key]:
                _fail(failures, 'stale_proof:' + where + '.ownership.' + key)
            supplied = (segment.get('top') or {}).get('owner') or {}
            if key in supplied and supplied[key] != owner[key]:
                _fail(failures, 'stale_proof:' + where + '.ownership.' + key)
        else:
            _miss(missing, 'missing_proof:' + where + '.declared_owner.' + key)
    copy = rank_row.get('copy')
    if type(copy) is not dict:
        _miss(missing, 'missing_proof:' + where + '.copy')
    else:
        for key, minimum in (('buckets', 1), ('bytes', 1), ('events', 1)):
            value = _int_field(copy, key, minimum, missing, where + '.copy')
            if value is None and key in copy:
                _fail(failures, 'near_tie_envelope:' + where + '.copy.' + key)
        completion = copy.get('completion')
        if completion == 'enqueued' or completion == 'wait_enqueued':
            _fail(failures, 'unqueried_completion:' + where + '.copy')
        elif completion not in PROOF_COMPLETION and copy.get('queried') is not True:
            _miss(missing, 'unproved:' + where + '.copy.completion')
    attention = rank_row.get('attention')
    if type(attention) is not dict:
        _miss(missing, 'missing_proof:' + where + '.attention')
    else:
        value = _int_field(attention, 'ops', 1, missing, where + '.attention')
        if value is None and 'ops' in attention:
            _fail(failures, 'near_tie_envelope:' + where + '.attention.ops')
        _require_true(attention, 'complete', missing, where + '.attention')
    state = rank_row.get('state_write')
    if type(state) is not dict:
        _miss(missing, 'missing_proof:' + where + '.state_write')
    else:
        proof = state.get('proof')
        if proof != 'queried_event' or state.get('queried') is not True:
            _fail(failures, 'unqueried_state_write:' + where)
        writers = state.get('writers')
        if type(writers) is not list or not writers \
                or any(not isinstance(w, str) or not w for w in writers):
            _miss(missing, 'missing_proof:' + where + '.state_write.writers')
        _int_field(state, 'joined_streams', 1, missing,
                   where + '.state_write')
    pp = rank_row.get('pp')
    if type(pp) is not dict:
        _miss(missing, 'missing_proof:' + where + '.pp')
    else:
        _require_true(pp, 'receive', missing, where + '.pp')
        _require_true(pp, 'consume', missing, where + '.pp')
    replay = rank_row.get('replay')
    if type(replay) is not dict:
        _miss(missing, 'missing_proof:' + where + '.replay')
        return
    before = replay.get('submitted_before')
    after = replay.get('submitted_after')
    cbefore = replay.get('completed_before')
    cafter = replay.get('completed_after')
    if not all(_is_int(v) for v in (before, after, cbefore, cafter)):
        _miss(missing, 'missing_proof:' + where + '.replay.counters')
        return
    if min(before, after, cbefore, cafter) < 0 or cbefore > before or cafter > after:
        _fail(failures, 'impossible_replay_counters:' + where)
    if after <= before or cafter <= cbefore:
        _fail(failures, 'serialized_false_positive:' + where
              + '.no_decode_replay_progress_in_segment')


def _segment_records(overlap, policy, failures, missing):
    """Check the flat segment list and return coverage/progress facts."""
    segments = overlap.get('segments')
    if type(segments) is not list or not segments:
        _miss(missing, 'missing_proof:overlap.segments')
        return {'accepted': 0, 'fallback': 0, 'routes': {k: 0 for k in ROUTE_LABELS},
                'arrivals': 0, 'useful_tokens': 0, 'completed_prefills': 0,
                'interval_counters_reset': [], 'segment_ranks': 0}
    accepted = fallback = arrivals = useful = completed = 0
    routes = {key: 0 for key in ROUTE_LABELS}
    previous_seq = 0
    last = {}
    owners = overlap.get('owners')
    if type(owners) is not dict or set(owners) != {str(r) for r in RANK_SET}:
        _miss(missing, 'missing_proof:overlap.four_worker_owners')
        owners = {}
    for index, segment in enumerate(segments):
        if type(segment) is not dict:
            _miss(missing, 'missing_proof:segment_not_object:%d' % index)
            continue
        seq = segment.get('segment_seq')
        where = 'segment%d' % index
        if not _is_int(seq) or seq <= previous_seq:
            _fail(failures, 'context_contradiction:segment_order:%d' % index)
        previous_seq = seq if _is_int(seq) else previous_seq
        start = segment.get('start_monotonic_ns')
        end = segment.get('end_monotonic_ns')
        if not _is_int(start) or not _is_int(end) or end <= start:
            _miss(missing, 'missing_proof:' + where + '.clock')
        elif segment.get('accepted') is True and \
                (end - start) < policy['effect_envelope']['min_segment_ns']:
            _fail(failures, 'near_tie_envelope:' + where + '.duration')
        stock = segment.get('stock_forward_used') is True or \
            segment.get('path') == 'stock_forward'
        is_accepted = segment.get('accepted') is True
        if is_accepted and stock:
            _fail(failures, 'whole_stock_forward_fallback_claimed_as_overlap:'
                  + where)
        if segment.get('claims_progress') is True and stock:
            _fail(failures, 'offered_only_work_claim:' + where)
        if stock:
            fallback += 1
            routes['serialized'] += 1
        elif is_accepted:
            accepted += 1
            path = segment.get('path')
            if path != 'e3_overlap':
                _fail(failures, 'overlap_path_not_native:' + where)
            route = segment.get('copy_route')
            if route in ('serialized', 'sparse_uva'):
                _fail(failures, 'unqualified_transport_admitted_to_overlap:' + where)
            elif route not in ('sparse', 'dense'):
                _miss(missing, 'missing_proof:' + where + '.copy_route')
            else:
                routes[route] += 1
            segment_arrivals = segment.get('decode_arrivals')
            if not _is_int(segment_arrivals):
                _miss(missing, 'missing_proof:' + where + '.decode_arrivals')
            elif segment_arrivals < policy['effect_envelope']['min_decode_arrivals']:
                _fail(failures, 'serialized_false_positive:' + where
                      + '.decode_arrivals')
            else:
                arrivals += segment_arrivals
            committed_before = segment.get('committed_tokens_before')
            committed_after = segment.get('committed_tokens_after')
            useful_before = segment.get('useful_tokens_before')
            useful_after = segment.get('useful_tokens_after')
            counters = (committed_before, committed_after, useful_before,
                        useful_after)
            if not all(_is_int(v) for v in counters):
                _miss(missing, 'missing_proof:' + where + '.useful_counters')
            else:
                if committed_after < committed_before \
                        or useful_after < useful_before:
                    _fail(failures, 'counter_reset:' + where)
                committed_delta = committed_after - committed_before
                useful_delta = useful_after - useful_before
                if useful_delta < policy['effect_envelope']['min_useful_tokens']:
                    _fail(failures, 'near_tie_envelope_offered_only:' + where)
                useful += max(0, useful_delta)
                if segment.get('prefill_completed') is True:
                    completed += 1
            ranks = segment.get('ranks')
            if type(ranks) is not list or len(ranks) != 4 \
                    or any(type(r) is not dict or not _is_int(r.get('rank'))
                           for r in ranks) \
                    or {r['rank'] for r in ranks} != RANK_SET:
                _fail(failures, 'ownership_contradiction:' + where
                      + '.four_rank_proof')
                continue
            for rank_row in sorted(ranks, key=lambda r: r['rank']):
                owner = owners.get(str(rank_row['rank']))
                _check_rank_proofs(rank_row, segment, policy, failures, missing,
                                   owner if type(owner) is dict else {})
                rank = rank_row['rank']
                for key, value in (('engine_epoch', overlap.get('engine_epoch')),
                                   ('storage_epoch', overlap.get('storage_epoch')),
                                   ('route_generation',
                                    overlap.get('route_generation'))):
                    seen = (rank_row.get('ownership') or {}).get(key)
                    if seen is not None and value is not None and seen != value:
                        _fail(failures, 'stale_proof:%s.rank%d.%s'
                              % (where, rank, key))
                replay = rank_row.get('replay') or {}
                after = replay.get('submitted_after')
                if _is_int(after):
                    prior = last.get(rank)
                    if prior is not None and after < prior:
                        _fail(failures, 'counter_reset:%s.rank%d.replay'
                              % (where, rank))
                    last[rank] = after
        elif segment.get('accepted') is False:
            pass
        else:
            _miss(missing, 'missing_proof:' + where + '.accepted')
        if segment.get('fallback_claimed_as_overlap') is True:
            _fail(failures, 'whole_stock_forward_fallback_claimed_as_overlap:'
                  + where)
    if not accepted:
        if fallback:
            _miss(missing, 'no_accepted_overlap_segment:fallback_only')
        else:
            _miss(missing, 'no_accepted_overlap_segment')
    return {'accepted': accepted, 'fallback': fallback, 'routes': routes,
            'arrivals': arrivals, 'useful_tokens': useful,
            'completed_prefills': completed}


# ---------------------------------------------------------------------------
# main grade
# ---------------------------------------------------------------------------

def _baseline(evidence, key, expected_route, failures, missing):
    baselines = evidence.get('baselines') or {}
    row = baselines.get(key)
    if type(row) is not dict or not row:
        # An absent/empty lane is missing evidence, not a contradicting route.
        _miss(missing, 'baseline_missing:' + key)
        return None
    route = row.get('route')
    if route != expected_route:
        _fail(failures, 'wrong_baseline_route:%s=%r' % (key, route))
    if row.get('valid') is not True:
        _miss(missing, 'baseline_invalid:' + key)
    return row


def _metric(row, key):
    if type(row) is not dict:
        return None
    value = row.get(key)
    return float(value) if _is_num(value) else None


def _cleanup_checks(cleanup, failures, missing):
    if type(cleanup) is not dict:
        _miss(missing, 'cleanup_missing')
        return
    state = cleanup.get('engine_state')
    if state not in ('inactive', 'failed'):
        if state is None:
            _miss(missing, 'cleanup.engine_state_unproven')
        else:
            _fail(failures, 'cleanup_engine_not_stopped:' + str(state))
    if cleanup.get('engine_stopped') is not True:
        _miss(missing, 'cleanup.engine_stopped_unproved')
    if cleanup.get('production_restored') is True:
        _fail(failures, 'production_restored_without_operator')
    if cleanup.get('automatic_back') is True:
        _fail(failures, 'automatic_production_back')
    found = cleanup.get('launcher_found_sha256')
    restored = cleanup.get('launcher_restored_sha256')
    if found is None or restored is None:
        _miss(missing, 'cleanup.launcher_restore_unproved')
    elif found != restored:
        _fail(failures, 'cleanup_launcher_not_restored_bytewise')
    dfound = cleanup.get('defaults_found_sha256')
    drestored = cleanup.get('defaults_restored_sha256')
    if dfound is None or drestored is None:
        _miss(missing, 'cleanup.defaults_restore_unproved')
    elif dfound != drestored:
        _fail(failures, 'cleanup_defaults_not_restored_bytewise')
    if cleanup.get('restore_withheld') is True:
        _fail(failures, 'cleanup_restore_withheld_under_live_engine')


def _memory_checks(evidence, failures, missing, reserve_mib):
    memory = evidence.get('memory') or {}
    if 'reserve_mib' not in memory:
        _miss(missing, 'reserve_declaration_missing')
        declared = None
    else:
        declared = memory.get('reserve_mib')
    if declared is not None and (not _is_int(declared)
                                 or declared != DEFAULT_RESERVE_MIB):
        _fail(failures, 'reserve_changed:%r' % (declared,))
    samples = memory.get('samples')
    if type(samples) is not list or not samples:
        _miss(missing, 'memory_samples_missing')
    else:
        under = [s for s in samples if type(s) is dict and s.get('pressure') is True]
        if not under:
            _miss(missing, 'memory_pressure_samples_missing')
        else:
            minima = {}
            for sample in under:
                indexes = []
                for gpu in sample.get('gpus') or []:
                    if type(gpu) is not dict or not _is_int(gpu.get('index')) \
                            or not _is_num(gpu.get('free_mib')):
                        _miss(missing, 'memory_sample_incomplete')
                        continue
                    index = gpu['index']
                    indexes.append(index)
                    minima[index] = min(minima.get(index, float('inf')),
                                        float(gpu['free_mib']))
                if len(indexes) != len(RANK_SET) or set(indexes) != RANK_SET:
                    _miss(missing, 'memory_sample_four_gpus_required')
            memory['min_free_mib_under_pressure'] = minima
            if not minima:
                _miss(missing, 'memory_sample_incomplete')
            for index, free in sorted(minima.items()):
                if free < DEFAULT_RESERVE_MIB:
                    _fail(failures, '512MiB_reserve_breached:gpu%d:%.1f'
                          % (index, free))
    allocator = memory.get('allocator_peaks')
    if type(allocator) is not dict or not allocator:
        _miss(missing, 'allocator_peaks_missing')
    else:
        ranks = set()
        for key, value in allocator.items():
            label = str(key)
            for prefix in ('gpu', 'rank'):
                if label.startswith(prefix):
                    label = label[len(prefix):]
                    break
            if label.isdigit() and (type(value) is dict and bool(value)
                                    or _is_int(value) and value >= 0):
                ranks.add(int(label))
        if ranks != RANK_SET or len(allocator) != len(RANK_SET):
            _miss(missing, 'allocator_peaks_four_ranks_required')


def grade(evidence, reserve_mib=DEFAULT_RESERVE_MIB):
    """Grade composed overlap evidence.  Pure: no I/O, no engine access."""
    failures, missing, shortfall = [], [], []
    if type(evidence) is not dict:
        return {'schema': SCHEMA, 'verdict': 'INCONCLUSIVE',
                'failures': [], 'missing': ['evidence_not_object'],
                'shortfall': [], 'target': None}
    # Derived owners/minima must not alter retained controller/native receipts.
    evidence = copy.deepcopy(evidence)
    if evidence.get('schema') != SCHEMA:
        _fail(failures, 'evidence_schema_differs')
    target = target_definition(
        (evidence.get('target') or {}).get('label')
        or evidence.get('target_label'), evidence.get('target'))
    if not target.get('ok'):
        for problem in target.get('problems') or ['target_label_required']:
            _miss(missing, problem)
        return {'schema': SCHEMA, 'verdict': 'FAIL' if failures else 'INCONCLUSIVE',
                'failures': failures, 'missing': sorted(set(missing)),
                'shortfall': [], 'target': target,
                'scope': 'target not resolved; no denominator guessed'}

    overlap = evidence.get('overlap') or {}
    readiness = evidence.get('readiness')
    if readiness is not None:
        if type(readiness) is not dict or not readiness.get('verdict'):
            _miss(missing, 'candidate_readiness_unproved')
        elif readiness.get('verdict') != 'PASS':
            _fail(failures, 'candidate_readiness_not_pass:'
                  + str(readiness.get('verdict')))
    w1_full = _baseline(evidence, 'w1_full', 'full', failures, missing)
    w1_split = _baseline(evidence, 'w1_split', 'split', failures, missing)
    prefill_full = _baseline(evidence, 'prefill_full', 'full', failures, missing)
    prefill_split = _baseline(evidence, 'prefill_split', 'split', failures,
                              missing)
    for name, row in (('w1_full', w1_full), ('w1_split', w1_split),
                      ('prefill_full', prefill_full),
                      ('prefill_split', prefill_split)):
        if row is None:
            continue
        serving = row.get('serving_baseline')
        if name.endswith('_full'):
            if serving is not True:
                _fail(failures, 'wrong_baseline_route:%s_not_serving' % name)
        elif serving is True:
            _fail(failures, 'wrong_baseline_route:%s_claimed_serving' % name)
        for key in ('decode_tokens_per_s', 'prefill_tokens_per_s',
                    'workload_sha256'):
            if key in row and row[key] is None:
                _miss(missing, 'baseline_incomplete:%s.%s' % (name, key))

    def _same_context(rows, tag):
        """Full and matched-split lanes must describe one measured context."""
        values = [row.get('workload_sha256') if type(row) is dict else None
                  for row in rows]
        if any(value is None for value in values):
            _miss(missing, 'workload_context_missing:' + tag)
            return None
        if len(set(values)) != 1:
            _fail(failures, 'baseline_context_mismatch:' + tag)
        return values[0]

    w1_context = _same_context([w1_full, w1_split], 'w1')
    overlap_context = _same_context([prefill_full, prefill_split], 'overlap')
    osha = overlap.get('workload_sha256')
    if osha is None:
        _miss(missing, 'overlap_workload_sha256_missing')
    elif overlap_context is not None and osha != overlap_context:
        _fail(failures, 'baseline_context_mismatch:overlap_lane')
    if w1_context is not None and overlap_context is not None:
        evidence['workload_context'] = {'w1': w1_context,
                                        'overlap': overlap_context}
    if overlap.get('engine_crash') is True:
        _miss(missing, 'engine_crash_partial_evidence')
    if overlap.get('valid') is not True:
        _miss(missing, 'overlap_lane_not_valid')
    invalid = overlap.get('invalid_intervals')
    if type(invalid) is not list:
        _miss(missing, 'invalid_intervals_not_retained')
    else:
        for index, row in enumerate(invalid):
            if type(row) is not dict or not row.get('reasons'):
                _miss(missing, 'invalid_interval_reasons_missing:%d' % index)
    basis = overlap.get('summary_basis') or {}
    if basis.get('filtered_invalid') is True:
        _fail(failures, 'invalid_intervals_filtered_from_result')
    if basis.get('retry_selected') is True or basis.get('picked') is True:
        _fail(failures, 'calibration_retry_or_picking')
    claims = overlap.get('claims') or {}
    if claims.get('retry_selected') is True or claims.get('survivor_averaged') is True:
        _fail(failures, 'calibration_retry_or_survivor_averaging')
    if claims.get('offered_work') is True:
        _fail(failures, 'speed_claim_from_offered_work')
    if overlap.get('interval_counters') is not None:
        counters = overlap.get('interval_counters')
        if type(counters) is not dict or not _is_int(counters.get('reset_count')):
            _miss(missing, 'interval_counter_proof_missing')
        elif counters['reset_count'] > 0:
            _fail(failures, 'counter_reset:interval_counters')

    coverage = _segment_records(overlap, target, failures, missing)
    useful_final = _metric(overlap, 'useful_completed_tokens')
    prompt_tokens = _metric(overlap, 'prompt_tokens')
    if useful_final is None or prompt_tokens is None:
        _miss(missing, 'useful_work_accounting_missing')
    else:
        if coverage['useful_tokens'] and abs(coverage['useful_tokens']
                                             - useful_final) > 0:
            _fail(failures, 'useful_token_accounting_differs')
        if useful_final < prompt_tokens:
            _miss(missing, 'prefill_not_completed')
    completed_prefills = overlap.get('completed_prefills')
    if not _is_int(completed_prefills):
        _miss(missing, 'completed_prefills_missing')
    elif completed_prefills < 1:
        _miss(missing, 'prefill_not_completed')
    _memory_checks(evidence, failures, missing, reserve_mib)
    _cleanup_checks(evidence.get('cleanup'), failures, missing)

    full_decode = _metric(w1_full, 'decode_tokens_per_s')
    split_decode = _metric(w1_split, 'decode_tokens_per_s')
    overlap_decode = _metric(overlap, 'decode_tokens_per_s')
    full_gap = _metric(w1_full, 'gap_p99_ms')
    overlap_gap = _metric(overlap, 'gap_p99_ms')
    prefill_split_tps = _metric(prefill_split, 'prefill_tokens_per_s')
    prefill_full_tps = _metric(prefill_full, 'prefill_tokens_per_s')
    overlap_prefill = _metric(overlap, 'prefill_tokens_per_s')
    if full_decode is None or full_decode <= 0:
        _miss(missing, 'zero_decode_denominator')
    if overlap_decode is None or overlap_decode <= 0:
        _miss(missing, 'overlap_decode_rate_missing')
    decode_share = _ratio(overlap_decode, full_decode)
    split_share = _ratio(split_decode, full_decode)
    prefill_share = _ratio(overlap_prefill, prefill_split_tps)
    prefill_of_full = _ratio(overlap_prefill, prefill_full_tps)
    fresh_ttft = _metric(overlap, 'fresh_ttft_s')
    append_ttft = _metric(overlap, 'append_ttft_s')

    metrics = {
        'decode_tokens_per_s': {'overlap': overlap_decode,
                                'w1_full': full_decode,
                                'w1_split': split_decode},
        'decode_share_of_full': decode_share,
        'decode_share_of_matched_split_solo': _ratio(overlap_decode, split_decode),
        'decode_share_min': target['decode_share_min'],
        'split_solo_share_of_full': split_share,
        'gap_p99_ms': {'overlap': overlap_gap, 'w1_full': full_gap},
        'gap_p99_ratio': _ratio(overlap_gap, full_gap),
        'prefill_tokens_per_s': {'overlap': overlap_prefill,
                                 'prefill_full': prefill_full_tps,
                                 'prefill_split': prefill_split_tps},
        'prefill_share_of_matched_split_solo': prefill_share,
        'prefill_share_of_full_solo': prefill_of_full,
        'prefill_share_min': target['prefill_share_min'],
        'fresh_ttft_s': fresh_ttft,
        'append_ttft_s': append_ttft,
        'useful_completed_tokens': useful_final,
        'corpus_prompt_tokens': prompt_tokens,
        'useful_token_rate': _ratio(useful_final,
                                    _metric(overlap, 'prefill_wall_s')),
        'valid_completed_turns': overlap.get('valid_completed_turns'),
        'cancellations': overlap.get('cancellations'),
        'errors': overlap.get('errors'),
        'allocator': (evidence.get('memory') or {}).get('allocator_peaks'),
        'workload_context': evidence.get('workload_context'),
        'denominator_note': DENOMINATOR_NOTE,
    }
    if decode_share is None:
        pass
    elif decode_share < target['decode_share_min']:
        shortfall.append('decode_share_below_target:%.4f<%.4f'
                         % (decode_share, target['decode_share_min']))
    if target.get('per_stream_decode_required'):
        baseline_rates = (w1_full or {}).get('stream_rates_tokens_per_s')
        overlap_rates = overlap.get('stream_rates_tokens_per_s')
        if type(baseline_rates) is not list or type(overlap_rates) is not list \
                or len(baseline_rates) != 8 or len(overlap_rates) != 8:
            _miss(missing, 'per_stream_decode_rates_missing')
        else:
            shares = []
            for index, (base, seen) in enumerate(zip(baseline_rates,
                                                     overlap_rates)):
                ratio = _ratio(_num(seen), _num(base))
                if ratio is None:
                    _miss(missing, 'per_stream_decode_rate_zero:%d' % index)
                    continue
                shares.append(ratio)
                if ratio < target['decode_share_min']:
                    shortfall.append('stream_decode_share_below_target:%d:%.4f'
                                     % (index, ratio))
            metrics['per_stream_decode_share_min'] = min(shares) if shares else None
    if target.get('gap_p99_ratio_max') is not None:
        if overlap_gap is None or full_gap is None or full_gap <= 0:
            _miss(missing, 'gap_p99_proof_missing')
        elif overlap_gap > full_gap * target['gap_p99_ratio_max']:
            shortfall.append('gap_p99_above_target:%.3f>%.3f'
                             % (overlap_gap,
                                full_gap * target['gap_p99_ratio_max']))
    if target.get('fresh_ttft_s_max') is not None:
        if fresh_ttft is None:
            _miss(missing, 'fresh_ttft_missing')
        elif fresh_ttft > target['fresh_ttft_s_max']:
            shortfall.append('fresh_ttft_above_target:%.3f>%.2f'
                             % (fresh_ttft, target['fresh_ttft_s_max']))
    if target.get('prefill_share_min') is not None:
        if prefill_split_tps is None or prefill_split_tps <= 0:
            _miss(missing, 'zero_prefill_denominator')
        elif prefill_share is None:
            _miss(missing, 'overlap_prefill_rate_missing')
        elif prefill_share < target['prefill_share_min']:
            shortfall.append('prefill_share_below_target:%.4f<%.4f'
                             % (prefill_share, target['prefill_share_min']))
    envelope = target['effect_envelope']
    if claims.get('speed_claim') is True:
        effect = abs((decode_share or 0.0) - 1.0)
        if effect < envelope['min_effect']:
            _fail(failures, 'near_tie_envelope:speed_claim_inside_envelope')
        if decode_share is None or decode_share < target['decode_share_min']:
            _fail(failures, 'near_tie_envelope:speed_claim_without_target_evidence')
    if coverage['fallback']:
        coverage['note'] = ('explicit stock-forward fallback segments are '
                            'retained and measured; they are never counted as '
                            'accepted overlap work')
    routes = {
        'sparse': coverage['routes']['sparse'],
        'dense': coverage['routes']['dense'],
        'serialized': coverage['routes']['serialized'],
        'stock_forward_fallback': coverage['fallback'],
        'accepted_segments': coverage['accepted'],
        'decode_arrivals_during_segments': coverage['arrivals'],
        'note': ('naturally occurring dense/sparse/serialized copies are '
                 'reported by count; no route mix is invented or filtered'),
    }
    # The operator's80/80 line is a target, not an ownership/correctness gate.
    # Keep its shortfall explicit without calling a valid measurement corrupt.
    hard_target_failure = bool(shortfall) and target['label'] == 's5-w2-target'
    verdict = ('FAIL' if failures or hard_target_failure else
               'INCONCLUSIVE' if missing else 'PASS')
    return {
        'schema': SCHEMA,
        'verdict': verdict,
        'failures': sorted(set(failures)),
        'missing': sorted(set(missing)),
        'shortfall': sorted(set(shortfall)),
        'target': target,
        'target_met': None if failures or missing else not bool(shortfall),
        'target_status': ('UNPROVEN' if failures or missing else
                          'SHORTFALL' if shortfall else 'MET'),
        'metrics': metrics,
        'routes': routes,
        'coverage': coverage,
        'baselines': {'w1_full': w1_full, 'w1_split': w1_split,
                      'prefill_full': prefill_full,
                      'prefill_split': prefill_split},
        'memory': evidence.get('memory'),
        'cleanup': evidence.get('cleanup'),
        'scope': ('real E3 decode-during-prefill evidence only; performance '
                  'target marker, never an output-correctness verdict'),
    }


# ---------------------------------------------------------------------------
# candidate readiness
# ---------------------------------------------------------------------------

def _trusted_launcher_sha():
    try:
        import identity_declare
        return identity_declare.S5_LAUNCHER_SHA
    except Exception:  # noqa: BLE001 - missing bundle is an unproved pin
        return None


def _receipt_digest(receipt):
    core = {key: value for key, value in receipt.items()
            if key not in ('receipt_sha256',)}
    return _sha256_json(core)


def candidate_readiness(generation, readiness, *, launcher_sha,
                        package_sha256=None, manifest_entries=None,
                        env_launcher_sha=None, trusted_launcher_sha=None):
    """Refuse unarmable/fabricated packages and arbitrary env launcher pins."""
    failures, missing = [], []
    generation = generation if type(generation) is dict else {}
    readiness = readiness if type(readiness) is dict else {}
    trusted = trusted_launcher_sha or _trusted_launcher_sha()
    if trusted is None:
        _miss(missing, 'trusted_launcher_pin_unavailable')
    if generation.get('armable') is not True:
        _fail(failures, 'candidate_not_armable:armable_false')
    if generation.get('default_feature') != 'FLASH_S5_OVERLAP=0':
        _fail(failures, 'package_default_not_off')
    if generation.get('require_native_ready') is not True:
        _fail(failures, 'require_native_ready_not_declared')
    if generation.get('bypass_require_native_ready') is True:
        _fail(failures, 'require_native_ready_bypass_requested')
    if readiness.get('require_native_ready') is not True:
        _miss(missing, 'readiness.require_native_ready_unproved')
    if readiness.get('reviewed') is not True:
        _miss(missing, 'readiness.review_not_proved')
    if not isinstance(readiness.get('reviewed_by'), str) \
            or not readiness.get('reviewed_by'):
        _miss(missing, 'readiness.reviewer_missing')
    for key in ('native_ready', 'full_native_overlap', 'sparse_dma_lease',
                'armable'):
        if readiness.get(key) is not True:
            _miss(missing, 'readiness.' + key + '_unproved')
    package = package_sha256 or readiness.get('package_sha256')
    if package is None:
        _miss(missing, 'package_sha256_missing')
    else:
        if generation.get('package_sha256') not in (None, package):
            _fail(failures, 'package_sha_mismatch:generation')
        if readiness.get('package_sha256') not in (None, package):
            _fail(failures, 'package_sha_mismatch:readiness')
    if readiness.get('receipt_sha256') is not None:
        if readiness['receipt_sha256'] != _receipt_digest(readiness):
            _fail(failures, 'readiness_receipt_digest_differs')
    else:
        _miss(missing, 'readiness_receipt_digest_missing')
    entries = manifest_entries or {}
    reviewed_path = readiness.get('path')
    if entries:
        if not isinstance(reviewed_path, str) or not reviewed_path:
            _miss(missing, 'readiness.path_missing')
        else:
            name = reviewed_path.lstrip('*')
            if name not in entries:
                _fail(failures, 'readiness_receipt_not_in_package_manifest')
            elif readiness.get('content_sha256') is not None \
                    and entries[name] != readiness['content_sha256']:
                _fail(failures, 'readiness_receipt_content_differs')
    else:
        _miss(missing, 'package_manifest_missing')
    if trusted is not None:
        if launcher_sha != trusted:
            _fail(failures, 'candidate_launcher_not_trusted_pin')
        if readiness.get('launcher_sha256') not in (None, trusted):
            _fail(failures, 'readiness_launcher_pin_differs')
    if env_launcher_sha is not None and env_launcher_sha != trusted:
        _fail(failures, 'arbitrary_env_launcher_pin_refused')
    verdict = ('FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS')
    return {'schema': SCHEMA, 'verdict': verdict,
            'failures': sorted(set(failures)),
            'missing': sorted(set(missing)),
            'trusted_launcher_sha256': trusted,
            'package_sha256': package,
            'scope': ('source-declared reviewed inputs only; no ready receipt '
                      'is invented and require_native_ready is never bypassed')}


# ---------------------------------------------------------------------------
# composition helper / CLI
# ---------------------------------------------------------------------------

def _read_json(path):
    if path is None:
        return None
    return json.loads(Path(path).read_text())


def _read_manifest(path):
    entries = {}
    if path is None:
        return entries
    for line in Path(path).read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            entries[fields[1].lstrip('*')] = fields[0]
    return entries


def _load_samples(path):
    rows = []
    if path is None:
        return rows
    for line in Path(path).read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if type(row) is not dict:
            continue
        if row.get('kind') == 'stopped' or row.get('event') == 'stopped':
            continue
        raw = row.get('gpus')
        gpus = []
        if type(raw) is list:
            gpus = raw
        elif type(raw) is dict:
            # e2_memory_attribution writes {'0': {used_mib,total_mib}, ...};
            # convert to the per-GPU free-memory shape the reserve check uses.
            for key, value in sorted(raw.items()):
                if type(value) is not dict:
                    continue
                used = value.get('used_mib')
                total = value.get('total_mib')
                if not _is_int(used) or not _is_int(total) \
                        or not 0 <= used <= total:
                    continue
                gpus.append({'index': int(key), 'used_mib': used,
                             'total_mib': total,
                             'free_mib': float(total - used)})
        elif type(row.get('per_gpu')) is list:
            gpus = row['per_gpu']
        rows.append({'wall_s': row.get('start_s', row.get('wall_s')),
                     'gpus': gpus,
                     'pressure': bool(row.get('pressure'))})
    return rows


def _w1_route(w1, label):
    summary = (w1 or {}).get('summary') or {}
    routes = summary.get('routes') or {}
    row = routes.get(label) or {}
    rates = []
    for repeat in (w1 or {}).get('repeats') or []:
        if type(repeat) is not dict or repeat.get('valid') is not True \
                or repeat.get('route') != label:
            continue
        metrics = repeat.get('metrics') or {}
        if type(metrics.get('stream_rates_tokens_per_s')) is list:
            rates.extend(metrics['stream_rates_tokens_per_s'])
    per_stream = []
    for value in rates:
        if _is_num(value):
            per_stream.append(float(value))
    valid = [r for r in (w1 or {}).get('repeats') or []
             if type(r) is dict and r.get('route') == label and r.get('valid')]
    declared = [r for r in (w1 or {}).get('repeats') or []
                if type(r) is dict and r.get('route') == label]
    layouts = [(r.get('metrics') or {}).get('stream_rates_tokens_per_s') for r in valid]
    per_stream = ([statistics.median(values) for values in zip(*layouts)]
                  if layouts and all(type(row) is list and len(row) == 8
                                     and all(_is_num(x) for x in row) for row in layouts)
                  else None)
    gaps = []
    for repeat in valid:
        for stream in repeat.get('streams') or []:
            window = (stream or {}).get('window') or {}
            gaps.extend(g for g in window.get('gaps_ms') or [] if _is_num(g))
    return {
        'route': label,
        'valid': bool(valid) and len(valid) == len(declared),
        'repeats_valid': len(valid),
        'repeats_declared': len(declared),
        'invalid_repeats': len(declared) - len(valid),
        'decode_tokens_per_s': row.get('aggregate_tokens_per_s_median'),
        'stream_rates_tokens_per_s': per_stream,
        'gap_p99_ms': row.get('gap_ms_p99', row.get('gap_p99_ms')),
        'serving_baseline': label == 'full',
        'workload_sha256': (w1 or {}).get('config', {}).get('WORKLOAD_SHA256'),
        'scope': 'decode-only W1 baseline, retained repeats',
    }


def compose(w1, overlap, samples=None, cleanup=None, readiness=None,
            generation=None, manifest_entries=None):
    """Build the evidence object accepted by :func:`grade`."""
    lanes = (overlap or {}).get('lanes') or {}
    target_label = (overlap or {}).get('target_label')
    evidence = {
        'schema': SCHEMA,
        'run_id': (overlap or {}).get('run_id'),
        'target': {'label': target_label},
        'baselines': {
            'w1_full': _w1_route(w1, 'full'),
            'w1_split': _w1_route(w1, 'split'),
            'prefill_full': (lanes.get('prefill_full') or {}),
            'prefill_split': (lanes.get('prefill_split') or {}),
        },
        'overlap': {**((overlap or {}).get('overlap') or {})},
        'memory': (overlap or {}).get('memory') or {},
        'cleanup': cleanup,
    }
    if samples is not None:
        evidence['memory'] = {**(evidence.get('memory') or {}),
                              'samples': samples}
    evidence['overlap']['target_label'] = target_label
    evidence['readiness'] = None
    if readiness is not None:
        evidence['readiness'] = candidate_readiness(
            generation, readiness,
            launcher_sha=(overlap or {}).get('launcher_sha256')
            or (readiness or {}).get('launcher_sha256'),
            package_sha256=(overlap or {}).get('package_sha256'),
            manifest_entries=manifest_entries,
            env_launcher_sha=(overlap or {}).get('env_launcher_sha256'))
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--w1', type=Path, required=True)
    parser.add_argument('--overlap', type=Path, required=True)
    parser.add_argument('--samples', type=Path)
    parser.add_argument('--cleanup', type=Path)
    parser.add_argument('--readiness', type=Path)
    parser.add_argument('--generation', type=Path)
    parser.add_argument('--package-manifest', type=Path)
    parser.add_argument('--reserve-mib', type=int, default=DEFAULT_RESERVE_MIB)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    w1 = _read_json(args.w1)
    overlap = _read_json(args.overlap)
    samples = _load_samples(args.samples)
    cleanup = _read_json(args.cleanup)
    readiness = _read_json(args.readiness)
    generation = _read_json(args.generation)
    entries = _read_manifest(args.package_manifest)
    evidence = compose(w1, overlap, samples=samples, cleanup=cleanup,
                       readiness=readiness, generation=generation,
                       manifest_entries=entries)
    result = grade(evidence, reserve_mib=args.reserve_mib)
    with args.out.open('x') as handle:
        json.dump(result, handle, sort_keys=True)
        handle.write('\n')
    print('e3_overlap_grade ' + result['verdict'])
    return {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[result['verdict']]


if __name__ == '__main__':
    sys.exit(main())
