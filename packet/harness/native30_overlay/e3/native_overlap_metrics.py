"""Native overlap measurement algebra (CPU only; no torch, CUDA or vLLM import).

This module is the single place that decides what counts as an *accepted*
native overlap sample.  It has no runtime dependency, so the harness can import
it outside the engine process and every rule below is exercised by CPU
fixtures.

Rules encoded here, from the reviewed brief:

* Completion is a **queried** worker completion report (real completion proofs,
  observed by the engine tick), never a submission counter.  A row that was not
  queried is retained descriptively and cannot close a window.
* ``overlap_begin`` is the outer observation boundary *before* admission.  It is
  reported separately and never used as the headline denominator (it would
  include idle head/tail).
* The headline window is the actual fresh-prefill interval: from the first
  accepted native segment of that chunk through the final queried state-write
  observation.  The append window is computed separately, and its completions
  are selected by the *chunk's own request uid* -- never by a transport label.
* Serialized or UVA transport is retained descriptively but is never counted
  as admitted protected overlap.
* Invalid repeats and invalid intervals are retained; nothing is filtered,
  retried or picked.  A summary exists only when every declared repeat exists.
* Engine and client must prove the same ``CLOCK_MONOTONIC`` domain: boot id and
  time namespace identity plus a timestamp sampled inside the request's
  client send/receive bracket.  An unproved domain is INCONCLUSIVE.  A CUDA
  elapsed duration is never compared to a client monotonic timestamp.
"""
import math

SCHEMA = 1

CLOCK_BASIS = 'clock-monotonic'
ENGINE_SAMPLE_BASIS = 'engine-clock-monotonic-sample'
CLIENT_BASIS = 'client-time-monotonic_ns'

ACCEPTED_TRANSPORT = 'native-split'
DESCRIPTIVE_TRANSPORTS = ('serialized', 'uva', 'fallback')

OUTER_BASIS = 'overlap_begin..overlap_end outer observation boundary'
FRESH_BASIS = 'first accepted native fresh segment .. final queried state write'
APPEND_BASIS = 'first accepted native append segment .. final queried state write'

TARGET_LABEL = 's5-overlap-80-80'
TARGET_DECODE_SHARE_MIN = 0.80
TARGET_PREFILL_SHARE_MIN = 0.80

# Fields that only exist as a *duration* of device work.  They are never
# comparable to a host monotonic timestamp; naming them here is a refusal, not
# a conversion.
NON_COMPARABLE_BASES = ('cuda-event', 'cuda-elapsed', 'device-clock', 'gpu-clock')

MAX_BOUNDARY_NS = 10 * 60 * 1000 * 1000 * 1000  # ten minutes; a sane bracket


class OverlapMetricsError(ValueError):
    """Malformed measurement input (not an invalid-but-retained sample)."""


def clock_identity(proc=None):
    """Read this host's boot id and time namespace (local, read-only)."""
    import os
    root = proc or '/proc'
    try:
        with open(os.path.join(root, 'sys/kernel/random/boot_id'), 'r') as handle:
            boot_id = handle.read().strip()
        time_ns = os.readlink(os.path.join(root, 'self/ns/time'))
    except OSError as exc:
        return {'proved': False,
                'reason': 'clock-identity-unreadable:' + type(exc).__name__,
                'boot_id': None, 'time_namespace': None}
    if not boot_id or not time_ns:
        return {'proved': False, 'reason': 'clock-identity-empty',
                'boot_id': None, 'time_namespace': None}
    return {'proved': True, 'boot_id': boot_id, 'time_namespace': time_ns}


def prove_clock_domain(engine, expected, client_send_ns, client_recv_ns):
    """Prove that an engine sample lies in the client's monotonic domain.

    ``engine`` is the clock record the engine sampled while handling the
    command; ``expected`` is the client's own :func:`clock_identity` result.
    The sample must lie inside ``[client_send_ns, client_recv_ns]`` and name
    the same boot id and time namespace.  The bracket width is the uncertainty.

    An unproved or unknown domain returns ``proved: False`` with reasons; it
    never raises and never substitutes client time for an engine timestamp.
    """
    reasons = []
    send = client_send_ns if type(client_send_ns) is int else None
    recv = client_recv_ns if type(client_recv_ns) is int else None
    if type(engine) is not dict:
        reasons.append('engine-clock-record-missing')
        engine = {}
    basis = engine.get('basis')
    if basis in NON_COMPARABLE_BASES:
        reasons.append('device-elapsed-not-comparable-to-client-monotonic')
    elif basis != CLOCK_BASIS:
        reasons.append('engine-clock-basis-not-monotonic')
    if send is None or recv is None or send <= 0 or recv <= 0:
        reasons.append('client-request-bracket-missing')
    elif recv < send or recv - send > MAX_BOUNDARY_NS:
        reasons.append('client-request-bracket-invalid')
    if type(expected) is not dict or expected.get('proved') is not True:
        reasons.append('client-clock-identity-unproved')
    expected = expected if type(expected) is dict else {}
    boot_id = engine.get('boot_id')
    time_namespace = engine.get('time_namespace')
    if type(boot_id) is not str or not boot_id:
        reasons.append('engine-boot-id-missing')
    elif boot_id != expected.get('boot_id'):
        reasons.append('boot-id-differs')
    if type(time_namespace) is not str or not time_namespace:
        reasons.append('engine-time-namespace-missing')
    elif time_namespace != expected.get('time_namespace'):
        reasons.append('time-namespace-differs')
    sample = engine.get('sample_ns')
    if type(sample) is not int or sample <= 0:
        reasons.append('engine-clock-sample-missing')
    elif send is not None and recv is not None and send > 0 and recv > 0:
        if not (send <= sample <= recv):
            reasons.append('engine-sample-outside-request-bracket')
    proved = not reasons
    return {
        'schema': SCHEMA,
        'proved': proved,
        'reasons': reasons,
        'basis': CLOCK_BASIS if proved else 'unproved',
        'engine_basis': basis,
        'boot_id': boot_id if type(boot_id) is str else None,
        'time_namespace': time_namespace if type(time_namespace) is str else None,
        'engine_sample_ns': sample if type(sample) is int else None,
        'client_send_ns': send,
        'client_recv_ns': recv,
        'bracket_ns': ((recv - send) if send is not None and recv is not None
                       and recv >= send else None),
        'uncertainty_ns': ((recv - send) if send is not None and recv is not None
                           and recv >= send else None),
        'uncertainty_basis': ('engine sample is known to lie in the client '
                              'request bracket; the offset is bounded by the '
                              'bracket width'),
        'note': ('device elapsed durations are never compared to client '
                 'monotonic timestamps'),
    }


def _completion_rows(completions):
    """Validate queried completion evidence; retain unqueried rows."""
    if type(completions) is not list:
        raise OverlapMetricsError('completions-not-a-list')
    rows = []
    for index, row in enumerate(completions):
        if type(row) is not dict:
            raise OverlapMetricsError('completion-row-not-an-object')
        queried = row.get('queried') is True
        proofs = row.get('proofs')
        state_writes = []
        if type(proofs) is list:
            for proof in proofs:
                if type(proof) is dict and proof.get('kind') == 'state_write':
                    stamp = proof.get('completed_ns')
                    if type(stamp) is int and stamp > 0:
                        state_writes.append(stamp)
        rows.append({
            'index': index,
            'request': row.get('request'),
            'request_generation': row.get('request_generation'),
            'worker_rank': row.get('worker_rank'),
            'queried': queried,
            'completed_sequence': row.get('completed_sequence'),
            'state_write_ns': max(state_writes) if state_writes else None,
        })
    return rows


def _segment(raw):
    """Normalize one engine-reported segment without dropping invalid ones."""
    reasons = []
    if type(raw) is not dict:
        return {'valid': False, 'accepted': False,
                'reasons': ['segment-not-an-object']}
    out = dict(raw)
    accepted = raw.get('accepted') is True
    transport = raw.get('transport')
    start = raw.get('start_monotonic_ns')
    end = raw.get('end_monotonic_ns')
    if transport in DESCRIPTIVE_TRANSPORTS:
        reasons.append('transport-not-accepted-overlap:' + str(transport))
        accepted = False
    elif transport != ACCEPTED_TRANSPORT:
        reasons.append('transport-unproved')
        accepted = False
    if type(start) is not int or start <= 0:
        reasons.append('segment-start-unproved')
        accepted = False
    if type(end) is not int or end <= 0:
        reasons.append('segment-end-unproved')
        accepted = False
    if type(start) is int and type(end) is int and start > 0 and end > 0 and end <= start:
        reasons.append('segment-window-empty')
        accepted = False
    if raw.get('accepted') is not True:
        reasons.append('engine-did-not-accept-segment')
    if raw.get('state_write') is not True:
        reasons.append('segment-state-write-unproved')
        accepted = False
    if raw.get('decode_progress') is not None:
        value = raw.get('decode_progress')
        if type(value) is not int or value <= 0:
            reasons.append('segment-decode-progress-not-positive')
            accepted = False
    out.update({'valid': not reasons, 'accepted': accepted,
                'reasons': sorted(set(reasons))})
    return out


def _boundary(receipt):
    """Engine sample of an accepted command; the domain proof is separate."""
    if type(receipt) is not dict:
        return None, ['boundary-receipt-missing']
    sample = receipt.get('observed_clock_ns')
    if type(sample) is not int or sample <= 0:
        return None, ['boundary-sample-missing']
    return sample, []


def _window(label, basis, family, completions):
    """One prefill family's real interval, bounded by queried state writes."""
    uids = {item.get('request') for item in family
            if type(item.get('request')) is str}
    starts = [item['start_monotonic_ns'] for item in family
              if type(item.get('start_monotonic_ns')) is int]
    # Current source frames carry generations. Never borrow a completion from
    # another incarnation of a reused UID. Legacy all-absent rows stay readable.
    generation_bound = any(item.get('request_generation') is not None for item in family)
    keys = {(item.get('request'), item.get('request_generation')) for item in family}
    rows = [row for row in completions
            if row['queried'] and ((row.get('request'), row.get('request_generation')) in keys
                                  if generation_bound else row.get('request') in uids)]
    writes = [row['state_write_ns'] for row in rows
              if type(row.get('state_write_ns')) is int]
    if not starts:
        return {'basis': basis, 'valid': False,
                'reasons': [label + '-start-unproved'],
                'start_ns': None, 'end_ns': None, 'wall_s': None,
                'request_uids': sorted(uids)}
    if not writes:
        return {'basis': basis, 'valid': False,
                'reasons': [label + '-state-write-unproved'],
                'start_ns': min(starts), 'end_ns': None, 'wall_s': None,
                'request_uids': sorted(uids)}
    start, end = min(starts), max(writes)
    reasons = []
    if end <= start:
        reasons.append(label + '-window-empty')
    useful = sum(item.get('chunk_tokens') or 0 for item in family
                 if item.get('accepted') is True)
    return {'basis': basis, 'valid': not reasons, 'reasons': reasons,
            'start_ns': start, 'end_ns': end,
            'wall_s': round((end - start) / 1e9, 9) if end > start else None,
            'first_accepted_work_ns': start,
            'final_queried_state_write_ns': end,
            'useful_tokens': useful,
            'chunks': len(family),
            'request_uids': sorted(uids)}


def measure_windows(*, begin_receipt, end_receipt, begin_proof, end_proof,
                    segments, completions, decode_progress,
                    boundary_uncertainty_ns=None):
    """Build outer/fresh/append windows and their validity.

    ``begin_receipt``/``end_receipt`` carry ``observed_clock_ns`` sampled by the
    engine inside the client request bracket; ``begin_proof``/``end_proof`` are
    :func:`prove_clock_domain` results for those commands.  ``segments`` are
    engine-reported accepted segments (real admitted intervals).  ``completions``
    are queried worker reports observed by the engine.  ``decode_progress`` is
    the actual client-visible decode completion delta across the window (an
    int; not a rate).
    """
    reasons = []
    outer_start, missing = _boundary(begin_receipt)
    reasons.extend(missing)
    outer_end, missing = _boundary(end_receipt)
    reasons.extend(missing)
    for label, proof in (('begin', begin_proof), ('end', end_proof)):
        if type(proof) is not dict or proof.get('proved') is not True:
            reasons.append(label + '-clock-domain-unproved')
    normalized = [_segment(raw) for raw in (segments or [])]
    if not normalized:
        reasons.append('no-engine-segments')
    accepted = [item for item in normalized if item.get('accepted')]
    if not accepted:
        reasons.append('no-accepted-native-segment')
    completion_rows = _completion_rows(completions or [])
    queried = [row for row in completion_rows if row['queried']]
    if not queried:
        reasons.append('no-queried-worker-completion')
    if decode_progress is None or type(decode_progress) is not int:
        reasons.append('decode-progress-not-queried')
    elif decode_progress <= 0:
        reasons.append('decode-progress-not-positive')
    if outer_start is not None and outer_end is not None and outer_end <= outer_start:
        reasons.append('outer-window-empty')

    fresh = [item for item in accepted if item.get('prefill') == 'fresh']
    appended = [item for item in accepted if item.get('prefill') == 'append']
    fresh_window = _window('fresh-prefill', FRESH_BASIS, fresh, queried)
    append_window = _window('append-prefill', APPEND_BASIS, appended, queried)
    uncertainty = boundary_uncertainty_ns
    if type(uncertainty) is not int:
        uncertainty = max([(begin_proof or {}).get('uncertainty_ns') or 0,
                           (end_proof or {}).get('uncertainty_ns') or 0])
    outer = {
        'basis': OUTER_BASIS,
        'valid': not reasons,
        'reasons': sorted(set(reasons)),
        'start_ns': outer_start,
        'end_ns': outer_end,
        'wall_s': (round((outer_end - outer_start) / 1e9, 9)
                   if outer_start is not None and outer_end is not None
                   and outer_end > outer_start else None),
        'boundary_uncertainty_ns': uncertainty,
        'uncertainty_basis': ('engine observation samples lie inside the '
                              'client request brackets; the offset is bounded '
                              'by the wider bracket'),
        'headline': False,
        'excluded_from_headline': ('admission, idle head and terminal tail; '
                                   'never used as the throughput denominator'),
    }
    for item in normalized:
        if not item.get('valid'):
            reasons.extend(item.get('reasons') or [])
    if not fresh_window.get('valid'):
        reasons.extend(fresh_window.get('reasons') or [])
    return {
        'schema': SCHEMA,
        'headline': 'fresh_prefill',
        'outer': outer,
        'fresh_prefill': fresh_window,
        'append_prefill': append_window,
        'segments': normalized,
        'invalid_segments': [item for item in normalized if not item.get('valid')],
        'serialized_descriptive': [item for item in normalized
                                   if item.get('transport') in DESCRIPTIVE_TRANSPORTS],
        'completions': completion_rows,
        'unqueried_completions': [row for row in completion_rows
                                  if not row['queried']],
        'decode_progress': decode_progress,
        'boundary_uncertainty_ns': uncertainty,
        'reasons': sorted(set(reasons)),
        'valid': (not reasons and fresh_window.get('valid') is True
                  and outer.get('valid') is True),
        'summary_basis': {'all_declared_repeats_retained': True,
                          'filtered_invalid': False, 'retry_selected': False,
                          'picked': False},
    }


def decode_share(*, decode_ns, window_ns, decode_progress):
    """Decode share of a single valid window; ``None`` when unproved."""
    if type(decode_progress) is not int or decode_progress <= 0:
        return None
    if type(decode_ns) not in (int, float) or type(window_ns) not in (int, float):
        return None
    if window_ns <= 0 or decode_ns < 0:
        return None
    return round(float(decode_ns) / float(window_ns), 9)


def throughput(tokens, wall_s):
    """Tokens/s with explicit unavailability instead of an invented zero."""
    if type(tokens) is not int or tokens <= 0:
        return None
    if type(wall_s) not in (int, float) or not math.isfinite(wall_s) or wall_s <= 0:
        return None
    return round(tokens / float(wall_s), 6)


def summarize_repeats(repeats, declared):
    """Retain every declared repeat; never select a survivor."""
    if type(declared) is not int or declared < 1:
        raise OverlapMetricsError('invalid-declared-repeat-count')
    if type(repeats) is not list:
        raise OverlapMetricsError('repeats-not-a-list')
    seen = set()
    for row in repeats:
        if type(row) is not dict or type(row.get('repeat')) is not int:
            raise OverlapMetricsError('invalid-repeat-row')
        index = row['repeat']
        if index < 0 or index >= declared or index in seen:
            raise OverlapMetricsError('repeat-index-out-of-range-or-duplicate')
        seen.add(index)
    if len(seen) != declared:
        raise OverlapMetricsError('declared-repeats-missing')
    invalid = [dict(row) for row in repeats if row.get('valid') is not True]
    valid = [row for row in repeats if row.get('valid') is True]
    values = sorted(row['value'] for row in valid
                    if type(row.get('value')) in (int, float))
    aggregate = None
    if not invalid and len(values) == declared:
        middle = len(values) // 2
        aggregate = (values[middle] if declared % 2
                     else (values[middle - 1] + values[middle]) / 2.0)
    return {
        'declared': declared, 'retained': len(repeats),
        'valid': len(valid), 'invalid': len(invalid),
        'invalid_repeats': invalid,
        'all_valid': not invalid and len(valid) == declared,
        'aggregate': aggregate,
        'summary_basis': {'all_declared_repeats_retained': True,
                          'filtered_invalid': False, 'retry_selected': False,
                          'picked': False, 'first_survivor': False},
        'repeats': [dict(row) for row in repeats],
    }


def _share(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 9)


def target_marker(*, decode_share_value, prefill_share_value, label,
                  decode_denominator, prefill_denominator):
    """One named target result; a marker never becomes an ownership gate."""
    if decode_share_value is None or prefill_share_value is None:
        status = 'INCONCLUSIVE'
    elif (decode_share_value >= TARGET_DECODE_SHARE_MIN
          and prefill_share_value >= TARGET_PREFILL_SHARE_MIN):
        status = 'PASS'
    else:
        status = 'SHORTFALL'
    return {
        'label': label,
        'status': status,
        'marker_only': True,
        'not_an_ownership_or_output_gate': True,
        'decode_share': decode_share_value,
        'decode_share_min': TARGET_DECODE_SHARE_MIN,
        'decode_shortfall': (round(TARGET_DECODE_SHARE_MIN - decode_share_value, 9)
                             if decode_share_value is not None
                             and decode_share_value < TARGET_DECODE_SHARE_MIN else 0.0),
        'decode_denominator': decode_denominator,
        'prefill_share': prefill_share_value,
        'prefill_share_min': TARGET_PREFILL_SHARE_MIN,
        'prefill_shortfall': (round(TARGET_PREFILL_SHARE_MIN - prefill_share_value, 9)
                              if prefill_share_value is not None
                              and prefill_share_value < TARGET_PREFILL_SHARE_MIN else 0.0),
        'prefill_denominator': prefill_denominator,
    }


def performance_summary(*, overlap, w1_decode_tokens_per_s,
                        split_prefill_tokens_per_s, full_prefill_tokens_per_s,
                        appends_complete, serialized_fallback=False):
    """Compose the named headline result from a valid overlap sample."""
    if type(overlap) is not dict:
        overlap = {}
    reasons = list(overlap.get('reasons') or [])
    if overlap.get('valid') is not True:
        reasons.append('overlap-window-invalid')
    if appends_complete is not True:
        reasons.append('append-not-complete-before-end')
    if serialized_fallback:
        reasons.append('serialized-or-uva-transport-not-accepted-overlap')
    fresh_candidate = overlap.get('fresh_prefill')
    fresh = fresh_candidate if type(fresh_candidate) is dict else {}
    prefill_rate = overlap.get('fresh_prefill_tokens_per_s')
    if type(prefill_rate) not in (int, float):
        prefill_rate = throughput(fresh.get('useful_tokens'), fresh.get('wall_s'))
    # The decode rate is measured during the same fresh window by the client;
    # it is never derived from a submission count.  Without it the share stays
    # None and the marker is INCONCLUSIVE.
    decode_rate = overlap.get('decode_tokens_per_s')
    if type(decode_rate) not in (int, float):
        decode_rate = None
    progress = overlap.get('decode_progress')
    decode_share_value = (None if type(progress) is not int or progress <= 0
                          else _share(decode_rate, w1_decode_tokens_per_s))
    prefill_matched = _share(prefill_rate, split_prefill_tokens_per_s)
    prefill_full_gpu = _share(prefill_rate, full_prefill_tokens_per_s)
    split_cost = _share(split_prefill_tokens_per_s, full_prefill_tokens_per_s)
    policy = target_marker(
        decode_share_value=decode_share_value,
        prefill_share_value=prefill_matched,
        label=TARGET_LABEL,
        decode_denominator='w1_full_decode_only',
        prefill_denominator='prefill_split')
    full_gpu_marker = target_marker(
        decode_share_value=decode_share_value,
        prefill_share_value=prefill_full_gpu,
        label=TARGET_LABEL + ':full-gpu-prefill-basis',
        decode_denominator='w1_full_decode_only',
        prefill_denominator='prefill_full')
    full_gpu_marker['basis'] = 'separately named full-GPU prefill marker'
    valid = (not reasons and decode_share_value is not None
             and prefill_matched is not None)
    return {
        'schema': SCHEMA,
        'valid': valid,
        'reasons': sorted(set(reasons)),
        'headline': {
            'window': 'fresh_prefill',
            'decode_share': decode_share_value,
            'decode_tokens_per_s': decode_rate,
            'w1_decode_tokens_per_s': w1_decode_tokens_per_s,
            'prefill_tokens_per_s': prefill_rate,
            'prefill_denominator': 'prefill_split',
            'prefill_share_matched_split': prefill_matched,
            'prefill_share_full_gpu': prefill_full_gpu,
            'full_gpu_prefill_ratio': split_cost,
            'idle_head_tail_excluded': True,
            'basis': FRESH_BASIS,
        },
        'append_prefill': {'window': 'append_prefill',
                           'retained_separately': True},
        'outer_observation': overlap.get('outer'),
        'target': policy,
        'target_status': policy['status'],
        'target_shortfall': {'decode': policy['decode_shortfall'],
                             'prefill_matched_split': policy['prefill_shortfall']},
        'full_gpu_80_80_marker': full_gpu_marker,
        'serialized_fallback': bool(serialized_fallback),
    }


__all__ = [
    'APPEND_BASIS', 'CLOCK_BASIS', 'DESCRIPTIVE_TRANSPORTS', 'FRESH_BASIS',
    'OverlapMetricsError', 'OUTER_BASIS', 'TARGET_LABEL', 'clock_identity',
    'decode_share', 'measure_windows', 'performance_summary', 'prove_clock_domain',
    'summarize_repeats', 'target_marker', 'throughput',
]
