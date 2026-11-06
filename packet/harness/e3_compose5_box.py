#!/usr/bin/env python3
"""q18a-compose-residue box arms and grade: compose-4's box over this table (harness/compose5_arms.py). Discovery mode. dnr added a per-row replay seed and the cap off; hold added the per-row native_hold, acknowledged;
coverage-residue adds the per-row native_residue, acknowledged, keeps the engine's decline record (with its prefill coverage counter) in each status row, and grades coverage per replay arm.

Every arm kind compose-5 runs, compose-4's box already runs: the live arm for sm, fon and foff, the replay arm for cmp and ny, the one-stream arm. Only the table and the names differ, so the table is bound in here
and nothing that ran is edited. The rule is compose5_rule.py on the fetched record; this grade only says the record is whole.
"""
from contextlib import contextmanager
from functools import wraps

import compose5_arms as T
import e3_compose4_box as _box

ARM_SCHEMA = 'compose5-arm/1'
GRADE_SCHEMA = 'compose5-grade/1'
SCOPE = 'the yield priced under the closed-loop replay and in a frozen pair, on one engine: crash, OOM, stock prefill and the price read by compose5_rule.py; not a release assessment and not a numerics result'
NOT_COVERED = _box.NOT_COVERED
replay = _box.replay
_report = _box.report


def report(result):
    return _report(result).replace('q18a-compose-4', 'q18a-compose-residue', 1).replace('compose4_rule.py', 'compose5_rule.py')


def __getattr__(name):
    """compose-4's box by its names, for a caller that only reads them."""
    return getattr(_box, name)


_status_row = _box.status_row


def status_row(tag, reply):
    """compose-4's status row, and the engine's decline record beside it: a1's yield_view keeps a fixed list of keys and drops it, so dnr's
    records had no decline counters. Bounded counts, no request UID; the coverage counter is in it."""
    row = _status_row(tag, reply)
    source = reply.get('append_yield') if type(reply) is dict else None
    if type(row.get('append_yield')) is dict and type(source) is dict:
        row['append_yield']['declines'] = source.get('declines')
    return row


@contextmanager
def _bound():
    """This table and these names in compose-4's box for one call, and back: a suite that runs both boxes in one process sees each with its own."""
    names = dict(T=T, ARM_SCHEMA=ARM_SCHEMA, GRADE_SCHEMA=GRADE_SCHEMA, SCOPE=SCOPE, report=report, status_row=status_row)
    saved = {name: getattr(_box, name) for name in names}
    for name, value in names.items():
        setattr(_box, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(_box, name, value)


def _under_this_table(fn):
    @wraps(fn)
    def call(*args, **kwargs):
        with _bound():
            return fn(*args, **kwargs)
    return call


_grade, _smoke, _replay_arm = _box.grade, _box.smoke_check, _box.replay_arm


def cap_selected(rec):
    """What each rank's lane_chunk answered at the arm's select: requested is the row's flag, and the arm's flushes delta reads 0 with the cap on too (c4b). None if a rank did not answer."""
    select = (rec.get('reserved_cap') or {}).get('select') or {}
    states = [((select.get(str(rank)) or {}).get('after') or {}).get('reserved_cap') for rank in range(4)]
    if not all(type(state) is dict and type(state.get('selected')) is bool for state in states):
        return None
    return any(state['selected'] or state.get('cap_bytes') for state in states)


COVERAGE_SCHEMA = 's5-prefill-coverage/1'


def _coverage(status):
    found = (((status or {}).get('append_yield') or {}).get('declines') or {}).get('coverage')
    return found if type(found) is dict and found.get('schema') == COVERAGE_SCHEMA else None


def coverage_delta(start, end):
    """The engine's prefill coverage counter between two status rows: rows and requests by live|idle, native|stock, site. None if either
    row lacks it, or a count went down (another engine)."""
    a, b = _coverage(start), _coverage(end)
    if a is None or b is None:
        return None
    out = dict(steps=b.get('steps', 0) - a.get('steps', 0), errors=b.get('errors', 0) - a.get('errors', 0), open_native=b.get('open_native'))
    for table in ('rows', 'requests'):
        out[table] = {}
        for state in ('live', 'idle'):
            out[table][state] = {}
            for path in ('native', 'stock'):
                was = ((a.get(table) or {}).get(state) or {}).get(path) or {}
                now = ((b.get(table) or {}).get(state) or {}).get(path) or {}
                diff = {site: now.get(site, 0) - was.get(site, 0) for site in sorted(set(now) | set(was))}
                if any(type(v) is not int or v < 0 for v in diff.values()) or out['steps'] < 0 or out['errors'] < 0:
                    return None
                out[table][state][path] = {site: v for site, v in diff.items() if v}
    live, idle = out['rows']['live'], out['rows']['idle']
    native, stock = sum(live['native'].values()), sum(live['stock'].values())
    out['live_native_share'] = round(native / (native + stock), 6) if native + stock else None
    native += sum(idle['native'].values())
    stock += sum(idle['stock'].values())
    out['all_native_share'] = round(native / (native + stock), 6) if native + stock else None      # the lead's 01:10 target: 1.0 both
    return out


def coverage_windows(arms):
    """Per replay arm: its entry to the next arm's entry (the residue setting holds until the next arm's stage command, so the turns
    still open at 'after' stay in), the last arm to its own 'after'."""
    by = {x.get('label'): x for x in arms if x.get('label')}
    order = [row['label'] for row in T.arms()]
    status = lambda rec, label: next((s for s in (rec or {}).get('statuses') or () if s.get('label') == label), None)
    out = {}
    for k, row in enumerate(T.arms()):
        rec = by.get(row['label'])
        if row['kind'] != 'replay' or rec is None:
            continue
        following = by.get(order[k + 1]) if k + 1 < len(order) else None
        end, to = status(following, 'entry'), 'next-entry'
        if end is None:
            end, to = next((s for s in reversed(rec.get('statuses') or ()) if str(s.get('label')).startswith('after')), None), 'after'
        delta = coverage_delta(status(rec, 'entry'), end)
        out[row['label']] = None if delta is None else dict(delta, to=to)
    return out


def grade(arms, *args, **kwargs):
    """compose-4's grade, and each arm's bound acknowledged: a launch that ignored it would read as this run with no sign of it."""
    with _bound():
        result = _grade(arms, *args, **kwargs)
    by = {x.get('label'): x for x in arms if x.get('label')}
    for row in T.arms():
        rec = by.get(row['label'])
        if rec is not None and (rec.get('inwait') or {}).get('bound_acknowledged') != row['chunk_bound']:
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':chunk-bound-not-acknowledged'})
        if rec is not None and (rec.get('inwait') or {}).get('hold_acknowledged') != row['native_hold']:
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':native-hold-not-acknowledged'})
        if rec is not None and (rec.get('inwait') or {}).get('residue_acknowledged') != row['native_residue']:
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':native-residue-not-acknowledged'})
        if rec is not None and cap_selected(rec):
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':reserved-cap-selected'})
    # coverage-residue: the operator's rule, measured. On a residue arm any stock prefill row fails it, with a decoder live or idle (the lead,
    # 6 Nov 01:10: 100 % of all prefill rows), the last-row handback by its own name (native rows produce no logits: the one path this build
    # leaves stock).
    result['coverage'] = coverage_windows(arms)
    rows = T.by_label()
    for label, delta in result['coverage'].items():
        fails = []
        if delta is None:
            fails.append('coverage-not-read')
        else:
            if delta['errors']:
                fails.append('coverage-errors')
            for state, name in (('live', 'with-decoder-live'), ('idle', 'idle')):
                stock = delta['rows'][state]['stock']
                if rows[label]['native_residue'] and any(v for site, v in stock.items() if site != 'handback'):
                    fails.append('stock-prefill-' + name)
                if rows[label]['native_residue'] and stock.get('handback'):
                    fails.append('handback-row-' + name)
        result['failures'] = sorted(set(result['failures']) | {label + ':' + f for f in fails})
    result['verdict'] = 'FAIL' if result['failures'] else 'INC' if result['missing'] else 'PASS'
    return result


def smoke_check(rec, row=None):
    """compose-4's smoke, and what this table adds: the bound acknowledged, chunks above 1024 rows really run, the host filter engaged on every rank."""
    with _bound():
        ok, reasons = _smoke(rec, row)
    if not isinstance(rec, dict):
        return ok, reasons
    row = row or T.by_label()[T.SMOKE_ARM]
    # dnr: the cap is off by design (the table's note). compose-4's box is frozen and asks for it, so its reason is replaced by the reverse: requested on an arm of this table is the failure.
    reasons = [r for r in reasons if r != 'smoke-reserved-cap-not-requested']
    if (rec.get('reserved_cap') or {}).get('requested') is not False:
        reasons.append('smoke-reserved-cap-not-off')
    if cap_selected(rec) is not False:
        reasons.append('smoke-reserved-cap-selected')
    if (rec.get('inwait') or {}).get('bound_acknowledged') != row['chunk_bound']:
        reasons.append('smoke-chunk-bound-not-acknowledged')
    if (rec.get('inwait') or {}).get('hold_acknowledged') != row['native_hold']:
        reasons.append('smoke-native-hold-not-acknowledged')
    if (rec.get('inwait') or {}).get('residue_acknowledged') != row['native_residue']:
        reasons.append('smoke-native-residue-not-acknowledged')
    # A silent fallback to 1024 would read as no gain. The arm's count holds the bulk, its returners and the append, so the line is the bulk's count at 1024: b13's 40 sit well under it.
    prompt, chunks = (rec.get('fresh') or {}).get('prompt_tokens_observed'), (rec.get('progress_delta') or {}).get('chunks')
    if type(prompt) is not int or type(chunks) is not int or chunks >= len(T.planned_chunks(prompt, 1024)):
        reasons.append('smoke-chunk-rows-not-requested')
    hf = (rec.get('cow_host_filter') or {}).get('counters') or {}
    kept, skipped = hf.get('kept'), hf.get('skipped')
    if not (type(kept) is dict and type(skipped) is dict and len(skipped) == 4 and all((kept.get(rank) or 0) + (skipped.get(rank) or 0) for rank in skipped)):
        reasons.append('smoke-host-filter-not-engaged')
    if any((hf.get('unfiltered') or {}).values()):
        reasons.append('smoke-host-filter-unfiltered')
    return not reasons, sorted(set(reasons))


run_arm = _under_this_table(_box.run_arm)


def replay_arm(a, row, client, poll_client, *, run=None):
    """compose-4's replay arm with this row's seed: in the run itself, and in the journal header, which reads the table's REPLAY."""
    base = run or _box.replay.run
    cfg = T.replay_config(row)

    def seeded(*args, **kwargs):
        return base(*args, cfg=cfg, **kwargs)
    saved = T.REPLAY
    T.REPLAY = cfg
    try:
        with _bound():
            return _replay_arm(a, row, client, poll_client, run=seeded)
    finally:
        T.REPLAY = saved


one_stream_arm = _under_this_table(_box.one_stream_arm)


def main(argv=None):
    # grade and smoke inside compose-4's main are this module's, for the call
    names = dict(grade=grade, smoke_check=smoke_check, replay_arm=replay_arm)
    saved = {name: getattr(_box, name) for name in names}
    with _bound():
        for name, value in names.items():
            setattr(_box, name, value)
        try:
            return _box.main(argv)
        finally:
            for name, value in saved.items():
                setattr(_box, name, value)

if __name__ == '__main__':
    raise SystemExit(main())
