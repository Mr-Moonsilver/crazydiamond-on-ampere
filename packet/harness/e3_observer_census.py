"""Row4 observer census: the observer_end reply's sizes, ring fills and replay count.

Pure functions over the parsed observer_end worker reply and the per-rank
observer packets the row4 collector already retains.  Nothing here reads the
engine, a file or the network, and nothing here grades placement: the census
names *why* an observer window is incomplete (which ring overflowed, how full
each ring was, how many decode replays the window held) so that row4 never
reports a generic INCONCLUSIVE for a capacity condition.

Receipt keys are the replay observer's own (patches/s5_engine/replay_observer.py
receipt: ``replay.counts``, ``replay.ring_overflows``, ``replay.flags``,
``limits``); correlation-report keys are the native report's own
(flash_s5_replay_observer.cpp ``s5ro_correlation_report``: ``complete``,
``capacity``, ``overflow``, ``errors``, ``open_scopes``, ``boundary_records``,
``activity_records`` and the ``apis``/``external``/``scopes``/``activities``
lists).
"""
import json

SCHEMA = 'q18a-observer-census/1'

# Ring -> (receipt count key, receipt limit key).  The kernel and copy rings
# hold every activity record of the window; the context and resource rings are
# the observer's bookkeeping rings.  ``resource_events`` counts callbacks seen,
# so it may exceed its ring's capacity; the overflow counter is the authority.
RINGS = (
    ('kernel', 'kernel_records', 'max_kernel_records'),
    ('copy', 'copy_records', 'max_copy_records'),
    ('context', 'context_records', 'max_context_records'),
    ('resource', 'resource_events', 'max_graph_events'),
)
# The alias map has an overflow counter but no stored entry count in the receipt.
ALIAS_RING = 'aliases'
# Correlation lists capped at the report's ``capacity`` (kCorrCapacity).  One
# shared ``overflow`` counter covers all three; a list at capacity names it.
CORRELATION_LISTS = (('apis', 'api'), ('external', 'external'), ('scopes', 'scope'))


def _int(value):
    return type(value) is int and value >= 0


def encoded_size(value):
    """Bytes of ``value`` as compact JSON (the frontend's JSONResponse form)."""
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':')).encode('utf-8'))


def field_sizes(result):
    """Per-rank, per-field encoded sizes of one parsed observer_end reply.

    ``result`` is the adapter reply (``worker_reply.workers[].qualification``).
    Unknown shapes are reported as such, never guessed.
    """
    worker_reply = result.get('worker_reply') if type(result) is dict else None
    rows = worker_reply.get('workers') if type(worker_reply) is dict else None
    if type(rows) is not list:
        return {'state': 'unavailable', 'per_rank': {}}
    per_rank = {}
    for row in rows:
        if type(row) is not dict:
            continue
        record = row.get('qualification')
        rank = row.get('worker_rank')
        if type(record) is not dict or type(rank) is not int:
            continue
        fields = {key: encoded_size(value) for key, value in sorted(record.items())}
        per_rank[str(rank)] = {'fields': fields, 'record_bytes': encoded_size(record),
                               'largest_field': max(fields, key=fields.get) if fields else None}
    envelope = encoded_size({key: value for key, value in result.items()
                             if key != 'worker_reply'})
    return {'state': 'measured', 'basis': 'compact JSON per field, UTF-8',
            'envelope_bytes': envelope, 'per_rank': per_rank}


def _receipt_rings(receipt, rank):
    missing = []
    replay = receipt.get('replay') if type(receipt) is dict else None
    counts = replay.get('counts') if type(replay) is dict else None
    rings = replay.get('ring_overflows') if type(replay) is dict else None
    flags = replay.get('flags') if type(replay) is dict else None
    limits = receipt.get('limits') if type(receipt) is dict else None
    if not (type(counts) is dict and type(rings) is dict and type(limits) is dict):
        return None, ['observer:ring-census-unavailable:%d' % rank]
    out = {}
    for ring, count_key, cap_key in RINGS:
        entries, cap, overflow = counts.get(count_key), limits.get(cap_key), rings.get(ring)
        out[ring] = {'entries': entries if _int(entries) else None,
                     'cap': cap if _int(cap) else None,
                     'overflow': overflow if _int(overflow) else None,
                     'count_key': count_key, 'cap_key': cap_key}
        if not _int(overflow):
            missing.append('observer:%s-ring-census-unavailable:%d' % (ring, rank))
        elif overflow:
            missing.append('observer:%s-ring-overflow:%d' % (ring, rank))
    aliases = rings.get(ALIAS_RING)
    out[ALIAS_RING] = {'entries': None, 'cap': None,
                       'overflow': aliases if _int(aliases) else None,
                       'note': 'graph alias map: overflow counter only, no stored entry count'}
    if not _int(aliases):
        missing.append('observer:aliases-ring-census-unavailable:%d' % rank)
    elif aliases:
        missing.append('observer:aliases-ring-overflow:%d' % rank)
    in_window = counts.get('records_in_window')
    permanent = (flags or {}).get('permanent_overflow') if type(flags) is dict else None
    return {'rings': out, 'records_in_window': in_window if _int(in_window) else None,
            'permanent_overflow': permanent if type(permanent) is bool else None}, missing


def _report_lists(report, rank):
    if type(report) is not dict:
        return None, ['observer:native-report-unavailable:%d' % rank]
    missing = []
    capacity = report.get('capacity')
    overflow = report.get('overflow')
    lists = {}
    at_capacity = []
    for key, label in CORRELATION_LISTS:
        rows = report.get(key)
        entries = len(rows) if type(rows) is list else None
        lists[key] = {'entries': entries, 'cap': capacity if _int(capacity) else None}
        if entries is None:
            missing.append('observer:%s-list-unavailable:%d' % (label, rank))
        elif _int(capacity) and entries >= capacity:
            at_capacity.append(label)
    activities = report.get('activities')
    lists['activities'] = {'entries': len(activities) if type(activities) is list else None,
                           'cap': None,
                           'note': 'distinct (correlation, kind, context, stream, graph, name) rows '
                                   'built from the kernel and copy rings; bounded by those rings'}
    if not _int(overflow) or not _int(capacity):
        missing.append('observer:correlation-census-unavailable:%d' % rank)
    elif overflow:
        if at_capacity:
            missing.extend('observer:%s-ring-overflow:%d' % (label, rank) for label in at_capacity)
        else:
            missing.append('observer:correlation-overflow-unattributed:%d' % rank)
    flags = {key: report.get(key) for key in
             ('complete', 'overflow', 'errors', 'open_scopes', 'boundary_records',
              'activity_records', 'capacity', 'reserved_bytes')}
    if report.get('complete') is not True:
        detail = [key for key in ('errors', 'open_scopes', 'boundary_records')
                  if _int(report.get(key)) and report[key]]
        missing.append('observer:correlation-report-incomplete:%d%s'
                       % (rank, (':' + ','.join(detail)) if detail else ''))
    return {'schema': report.get('schema'), 'flags': flags, 'lists': lists}, missing


def replay_bound(observed, *, declared, low, high, rank):
    if type(observed) is not int:
        return {'observed': None, 'declared': declared, 'low': low, 'high': high,
                'within': None}, ['observer:replays-unmeasured:%d' % rank]
    within = low <= observed <= high
    reasons = []
    if observed < low:
        reasons.append('observer:replays-below-declared:%d:%d<%d' % (rank, observed, low))
    elif observed > high:
        reasons.append('observer:replays-above-tolerance:%d:%d>%d' % (rank, observed, high))
    return {'observed': observed, 'declared': declared, 'low': low, 'high': high,
            'within': within}, reasons


def census(packets, observed_replays, *, ranks, declared, low, high):
    """Per-rank ring fills, correlation-list fills and the replay bound.

    ``packets`` are the retained observer_end rows by rank; ``observed_replays``
    is the collector's own per-rank router delta.  Returns the census record
    and the explicit missing reasons, in rank order.
    """
    packets = packets if type(packets) is dict else {}
    observed_replays = observed_replays if type(observed_replays) is dict else {}
    per_rank, missing = {}, []
    for rank in ranks:
        packet = packets.get(rank, packets.get(str(rank)))
        packet = packet if type(packet) is dict else {}
        rings, ring_missing = _receipt_rings(packet.get('receipt'), rank)
        report, report_missing = _report_lists(packet.get('native_report'), rank)
        observed = observed_replays.get(rank, observed_replays.get(str(rank)))
        bound, bound_missing = replay_bound(observed, declared=declared, low=low,
                                            high=high, rank=rank)
        per_rank[str(rank)] = {'observer_rings': rings, 'native_report': report,
                               'replays': bound}
        missing.extend(ring_missing + report_missing + bound_missing)
    return {'schema': SCHEMA, 'per_rank': per_rank,
            'replay_bound': {'declared': declared, 'low': low, 'high': high},
            'missing': missing}


__all__ = ['SCHEMA', 'census', 'encoded_size', 'field_sizes', 'replay_bound']
