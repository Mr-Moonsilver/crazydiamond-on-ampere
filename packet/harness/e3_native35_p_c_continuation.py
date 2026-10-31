#!/usr/bin/env python3
"""Native35 P-c: the declared row4 capacity continuation for W1/W2.

Lead decision 31 Oct 2026 (gate design, P-c only).  W1/W2 run when row4 is
PASS, or when row4 is INCONCLUSIVE and all of these hold:

- the client window is complete: token_bound, 8/8 streams opened, every stream
  gained >= PLACEMENT_WINDOW_TOKENS, no early finish and no stream error;
- every row4 missing reason is an observer-capacity reason (classified below)
  and at least one is a primary capacity reason;
- row4 failures are empty;
- no engine crash, no current-run engine error marker, NRestarts unchanged,
  the engine healthy, and the row4 cleanup restored route full with typed acks;
- row1 and rule4 PASS as now (the queue's own gate loop, unchanged).

Row4 stays INCONCLUSIVE in queue-grade and in its record.  gates.json records
``row4_capacity_continuation`` with the matched reasons and this receipt's
sha256; any other row4 outcome withholds W1/W2 exactly as before.

Rationale: row4's correctness proof for native35 is R and O (2/2 PASS with the
O3-form collector).  In the speed arm row4 is the overlap health check, and
observer capacity is not a correctness signal.

The reason strings are the ones the unchanged graders actually produce
(test_native35_p_c_continuation runs the real census, the real driver
collection and the real e3_q18a_qualification grading over source-produced
packets):

primary (counter-based)
  ``collect:observer-end-response-too-large``  the observer_end reply was over
      the spool ceiling; it must be the only missing reason and the retained
      transport receipt must show the refusal (body over the ceiling or not
      drained to EOF);
  ``collector:observer:<ring>-ring-overflow:<r>``  a replay-observer ring
      (kernel, copy, context, resource, aliases) or a correlation list (api,
      external, scope) at capacity with its overflow counter > 0.

secondary, per rank r, only when rank r has a primary reason
  activity-ring overflow (kernel, copy, context, resource, aliases):
    ``row4:r<r>:composition:raw_not_dischargeable:<native reasons>`` where the
    native reasons include ``ring_overflow`` and are a subset of
    RAW_CAPACITY_REASONS.  ``unattributed_expected_context`` is the documented
    dischargeable shape (native_graph_placement DISCHARGE_NATIVE) that only
    becomes non-dischargeable because ``ring_overflow`` joins it.
    ``missing_execution`` and ``incomplete_record`` are accepted only when that
    rank's kernel or copy ring dropped records: an exercised node without its
    records is then not separable from the drop.
  correlation-list overflow (api, external, scope):
    ``collector:observer:correlation-report-incomplete:<r>`` without a detail
    suffix (no errors, open scopes or boundary records), and
    ``row4:r<r>:composition:baseline:<reason>`` for reasons in
    CORRELATION_CAPACITY_REASONS, which must include ``correlation_overflow``.
  either: ``row4:r<r>:binding`` and ``row4:r<r>:composed-inc:`` (the composer
    returns no binding and no reason list on these early INC returns).

run level, only with a per-rank primary and every per-rank reason matched
  ``collector:phase-incomplete``, ``collector:native-prefill-ledger-unproved``
  (the prefill ledger is constructed only when every rank's packet regrades
  PASS).

Everything else, including the replay bound, ``*-unavailable`` reasons,
``correlation-overflow-unattributed`` and any detail-suffixed report
incompleteness, withholds.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

from e3_native_qualification_driver import (
    OBSERVER_END_RESPONSE_MAX as CEILING_BYTES, PLACEMENT_STREAMS as STREAMS,
    PLACEMENT_WINDOW_TOKENS as WINDOW_TOKENS)

SCHEMA = 'q18a-native35-p-c-row4-capacity-continuation/1'
RECEIPT_NAME = 'row4-capacity-continuation.json'
ROW4_NAME = 'qual-row4.json'
# The engine-errors marker set of the queue (q18a_native35_p_c.draft.sh capture_logs).
ERROR_MARKER_ERE = 'Traceback|AssertionError|CUDA error|Xid|ContractError|EngineDeadError|FLASH_S5_.*FAIL'

ACTIVITY_RINGS = ('kernel', 'copy', 'context', 'resource', 'aliases')
RECORD_RINGS = ('kernel', 'copy')
CORRELATION_LISTS = ('api', 'external', 'scope')
RAW_CAPACITY_REASONS = frozenset({'ring_overflow', 'unattributed_expected_context'})
RAW_DROP_REASONS = frozenset({'missing_execution', 'incomplete_record'})
CORRELATION_CAPACITY_REASONS = frozenset({
    'correlation_overflow', 'correlation_incomplete', 'decode_grade_inconclusive',
    'prefill_grade_inconclusive', 'whole_raw_incomplete', 'whole_raw_loss'})
TOO_LARGE = 'collect:observer-end-response-too-large'
RUN_LEVEL = ('collector:phase-incomplete', 'collector:native-prefill-ledger-unproved')

_PRIMARY = re.compile(r'^collector:observer:(%s)-ring-overflow:([0-3])$'
                      % '|'.join(ACTIVITY_RINGS + CORRELATION_LISTS))
_REPORT_INCOMPLETE = re.compile(r'^collector:observer:correlation-report-incomplete:([0-3])$')
_RANK = re.compile(r'^row4:r([0-3]):(.*)$')
RULE = ('W1/W2 run when row4 is PASS, or INCONCLUSIVE with a complete client window, '
        'only observer-capacity missing reasons, no failures, no engine crash or error '
        'marker, unchanged NRestarts, a healthy engine and a typed full-route cleanup; '
        'row4 stays INCONCLUSIVE (lead decision 31 Oct 2026, native35 P-c only)')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _window_checks(window, opened, cleanup, *, window_tokens=WINDOW_TOKENS, streams=STREAMS):
    window = window if type(window) is dict else {}
    opened = opened if type(opened) is dict else {}
    cleanup = cleanup if type(cleanup) is dict else {}
    rows = window.get('streams')
    rows = rows if type(rows) is list else []
    deltas = [row.get('completion_delta') for row in rows if type(row) is dict]
    return {
        'client_window_complete': window.get('complete') is True,
        'client_window_token_bound': window.get('reason') == 'token_bound',
        'client_window_streams': len(rows) == streams and len(deltas) == streams,
        'client_window_tokens': (len(deltas) == streams and all(
            type(value) is int and value >= window_tokens for value in deltas)),
        'client_window_no_early_finish': window.get('early_finish') == [],
        'client_window_no_stream_error': window.get('errors_before_end') == [],
        'client_open_complete': opened.get('complete') is True and opened.get('reasons') == [],
        'client_open_all_streams': opened.get('opened_streams') == streams,
        'cleanup_restored_full': (cleanup.get('restored') is True
                                  and cleanup.get('route') == 'full'),
        'cleanup_typed_acks': cleanup.get('typed_acks') is True,
    }


def _engine_checks(engine):
    engine = engine if type(engine) is dict else {}
    start, now = engine.get('nrestarts_start'), engine.get('nrestarts')
    return {
        'engine_not_crashed': engine.get('crashed') is False,
        'engine_healthy': engine.get('healthy') is True,
        'engine_nrestarts_unchanged': (type(start) is str and start.isdigit()
                                       and start == now),
        'engine_no_error_marker': engine.get('error_markers') == 0,
    }


def _transport_refused(evidence):
    transport = (evidence or {}).get('observer_end_transport') if type(evidence) is dict else None
    rows = (transport or {}).get('receipts') if type(transport) is dict else None
    if type(rows) is not list or not rows or type(rows[-1]) is not dict:
        return False
    last = rows[-1]
    return (last.get('worker_action') == 'observer_end'
            and last.get('state') == 'refused:too-large'
            and last.get('ceiling_bytes') == CEILING_BYTES
            and type(last.get('body_bytes')) is int
            and (last['body_bytes'] > CEILING_BYTES or last.get('drained_to_eof') is False)
            and type(last.get('body_sha256')) is str and len(last['body_sha256']) == 64)


def _classify_reasons(missing):
    """Match every reason to a declared capacity rule, or leave it unmatched."""
    matched, unmatched = [], []
    primaries = {}
    for reason in missing:
        found = _PRIMARY.match(reason) if type(reason) is str else None
        if found:
            primaries.setdefault(int(found.group(2)), set()).add(found.group(1))
    for reason in missing:
        rule = None
        if type(reason) is not str:
            unmatched.append(reason)
            continue
        primary = _PRIMARY.match(reason)
        if primary:
            rule = 'primary:%s-ring-overflow' % primary.group(1)
        report = _REPORT_INCOMPLETE.match(reason)
        if report:
            rings = primaries.get(int(report.group(1)), set())
            if rings & set(CORRELATION_LISTS):
                rule = 'secondary:correlation-report-incomplete-at-cap'
        ranked = _RANK.match(reason)
        if ranked:
            rings = primaries.get(int(ranked.group(1)), set())
            tail = ranked.group(2)
            if rings and tail in ('binding', 'composed-inc:'):
                rule = 'secondary:composer-early-inc'
            elif rings & set(ACTIVITY_RINGS) and tail.startswith(
                    'composition:raw_not_dischargeable:'):
                native = set(tail[len('composition:raw_not_dischargeable:'):].split(','))
                allowed = set(RAW_CAPACITY_REASONS)
                if rings & set(RECORD_RINGS):
                    allowed |= RAW_DROP_REASONS
                if 'ring_overflow' in native and native <= allowed:
                    rule = 'secondary:raw-ring-overflow-not-dischargeable'
            elif rings & set(CORRELATION_LISTS) and tail.startswith('composition:baseline:'):
                if tail[len('composition:baseline:'):] in CORRELATION_CAPACITY_REASONS:
                    rule = 'secondary:correlation-overflow-baseline'
        if rule is None:
            unmatched.append(reason)
        else:
            matched.append({'reason': reason, 'rule': rule})
    # A correlation baseline reason set must name the overflow itself.
    for rank, rings in primaries.items():
        if rings & set(CORRELATION_LISTS):
            prefix = 'row4:r%d:composition:baseline:' % rank
            baseline = [row for row in matched if row['reason'].startswith(prefix)]
            if baseline and prefix + 'correlation_overflow' not in [row['reason'] for row in baseline]:
                for row in baseline:
                    matched.remove(row)
                    unmatched.append(row['reason'])
    # Run-level reasons only follow fully matched per-rank capacity reasons.
    run_level = [reason for reason in unmatched if reason in RUN_LEVEL]
    if primaries and len(run_level) == len(unmatched):
        for reason in run_level:
            unmatched.remove(reason)
            matched.append({'reason': reason, 'rule': 'run-level:capacity-consequence'})
    return matched, unmatched, sorted(primaries)


def classify(record, *, engine, window_tokens=WINDOW_TOKENS, streams=STREAMS):
    """Eligibility of one retained qual-row4 record for the capacity continuation."""
    record = record if type(record) is dict else {}
    verdict = record.get('verdict')
    failures = record.get('failures')
    missing = record.get('missing')
    missing = list(missing) if type(missing) is list else None
    refusal = record.get('refusal_evidence')
    if type(refusal) is dict:
        path = 'refusal'
        window, opened = refusal.get('client_window'), refusal.get('open')
        cleanup = record.get('refusal_cleanup')
    else:
        path = 'collected'
        evidence = record.get('evidence') if type(record.get('evidence')) is dict else {}
        window, opened = evidence.get('client_window'), evidence.get('client_open')
        cleanup = evidence.get('placement_cleanup')
    checks = {'row4_inconclusive': verdict == 'INCONCLUSIVE',
              'row4_failures_empty': failures == [],
              'row4_missing_present': bool(missing)}
    checks.update(_window_checks(window, opened, cleanup, window_tokens=window_tokens,
                                 streams=streams))
    checks.update(_engine_checks(engine))
    matched, unmatched, ranks = [], list(missing or []), []
    if path == 'refusal':
        if missing == [TOO_LARGE] and _transport_refused(refusal):
            matched, unmatched = [{'reason': TOO_LARGE, 'rule': 'primary:observer-end-over-spool-ceiling'}], []
        checks['primary_capacity_reason'] = bool(matched)
    elif missing:
        matched, unmatched, ranks = _classify_reasons(missing)
        checks['primary_capacity_reason'] = bool(ranks)
    else:
        checks['primary_capacity_reason'] = False
    checks['every_reason_capacity'] = bool(missing) and not unmatched
    eligible = all(value is True for value in checks.values())
    return {'schema': SCHEMA, 'eligible': eligible, 'path': path, 'rule': RULE,
            'row4_verdict': verdict, 'row4_failures': copy.deepcopy(failures),
            'row4_missing': copy.deepcopy(missing), 'matched': matched,
            'unmatched': unmatched, 'capacity_ranks': ranks, 'checks': checks,
            'engine': copy.deepcopy(engine), 'window_tokens': window_tokens,
            'streams': streams, 'error_marker_ere': ERROR_MARKER_ERE}


def continuation_view(gates, gate_path):
    """Re-prove a ``row4_capacity_continuation`` gates.json from its retained files.

    Returns the gates view with row4 read as PASS for the unchanged native
    performance gate, or None when anything differs.  The receipt is re-derived
    from the retained qual-row4.json; a cached eligible flag is never trusted.
    """
    if type(gates) is not dict or gates.get('row4_capacity_continuation') is not True:
        return None
    outcomes = gates.get('outcomes')
    if type(outcomes) is not dict or outcomes.get('row4') != 'INCONCLUSIVE':
        return None
    folder = Path(gate_path).parent
    try:
        data = (folder / RECEIPT_NAME).read_bytes()
        row4_bytes = (folder / ROW4_NAME).read_bytes()
        receipt = json.loads(data)
        row4 = json.loads(row4_bytes)
    except (OSError, ValueError):
        return None
    if (type(receipt) is not dict or receipt.get('schema') != SCHEMA
            or receipt.get('eligible') is not True
            or sha(data) != gates.get('row4_capacity_continuation_sha256')
            or sha(row4_bytes) != receipt.get('row4_record_sha256')):
        return None
    again = classify(row4, engine=receipt.get('engine'))
    reasons = [row['reason'] for row in receipt.get('matched') or []]
    if (again['eligible'] is not True or again['matched'] != receipt.get('matched')
            or gates.get('row4_capacity_matched') != reasons):
        return None
    view = {key: copy.deepcopy(value) for key, value in gates.items()
            if not key.startswith('row4_capacity_')}
    view['outcomes'] = dict(outcomes, row4='PASS')
    return view


def _engine_arg(args):
    markers = args.error_markers
    return {'crashed': {'0': False, '1': True}.get(args.engine_crashed),
            'healthy': {'0': False, '1': True}.get(args.healthy),
            'nrestarts_start': args.nrestarts_start, 'nrestarts': args.nrestarts,
            'error_markers': int(markers) if markers.isdigit() else markers,
            'error_marker_basis': 'grep -Ec over the current-run engine-stream.log and engine-stream.err'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--row4', required=True)
    p.add_argument('--engine-crashed', required=True)
    p.add_argument('--healthy', required=True)
    p.add_argument('--nrestarts-start', required=True)
    p.add_argument('--nrestarts', required=True)
    p.add_argument('--error-markers', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    engine = _engine_arg(a)
    try:
        data = Path(a.row4).read_bytes()
        row4 = json.loads(data)
        result = classify(row4, engine=engine)
        result.update(row4_record=Path(a.row4).name, row4_record_sha256=sha(data))
    except (OSError, ValueError) as exc:
        result = {'schema': SCHEMA, 'eligible': False, 'rule': RULE, 'engine': engine,
                  'error': 'row4-record-unreadable:' + type(exc).__name__}
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True, indent=1)
        handle.write('\n')
    print('NATIVE35_P_C_ROW4_CAPACITY_CONTINUATION '
          + ('ELIGIBLE' if result['eligible'] else 'WITHHELD'))
    return 0 if result['eligible'] else 2


if __name__ == '__main__':
    sys.exit(main())
