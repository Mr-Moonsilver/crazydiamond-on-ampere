#!/usr/bin/env python3
"""q18a-stage1-step box arms and grade: L1 lazy PP receive and L2 decode merge against the base, all C1 defer, on alpha-next-c1b's engine.

Supersedes e3_alpha_next_c1_box (imported as ``c1``, unchanged) by name; its arm and grade run with the names below bound in their place for one call (c1's ``_bound``). The arm
table is stage1_step_arms (one declared table, 18 core arms, no tail): an arm not in it is refused before any traffic.

The arm: c1's (alpha-next's arm with C1's stage control), every arm C1 defer. Control order: pause, native_progress {burst}, native_qsa_fused {true}, native_layer_group {size},
native_stage_local {local: true, overlap: true, defer_bind: true, lazy_recv, decode_merge}, release_auto_arm, resume, then status only (every action through the real
probe-control route; ACTIONS below is the list the router test holds). Every arm sends both fields. The reply's values are recorded in ``step`` {requested_lazy, requested_merge,
replied, acknowledged_lazy, acknowledged_merge} and beside the others in ``stage_switch``; the arm never refuses a different acknowledgement: the record must reach the grade. The
status view adds ``native_stage_local.overlap.decode_step`` flat: ``step_lazy_installed``, ``step_lazy``, ``step_merge_installed``, ``step_merge``, ``step_<counter>``. No logprobs
on any request: native_auto_arm refuses them (q18a-alpha-next-c1b's lesson: they made every arm stock).

The grade: c1's grade over this table (every c1 gate kept: crash-free, TP commitment, group and C0 and C1 engagement, defer_bind acknowledged and engaged, one command in flight per
stage, OOM, torn lines, signatures) with c1b's C1-aware rules bound as attn-meta's box did. Then this queue's gate:
  selection  every arm's record carries what it sent and what the engine acknowledged (else FAIL HARNESS); the engine's reply and its status before and after the fresh prompt show
             the table's values, both levers installed (else FAIL ENGINE);
  counters   from the status delta of the fresh prompt: soff must move neither lazy_marked nor merge_steps, sl1 must not move merge_steps (FAIL ENGINE: a selection leak); a selected
             lever that never moved is INC (coverage not reached); a missing counter is INC.
Nothing else is a gate. The verdict is PASS, FAIL or INC. Numerics are NOT covered: every selected arm is a speed read; batch 8 against 2 x 4 changes batch composition, so OUTSIDE on
merged arms is expected and is not a defect.

Reads, never gates (step_read, stage1_step_read): per arm the decode ratio, live prefill tok/s, per-step thread time on the stage-1 ranks (full and empty execute_model, sample_tokens,
pair ms, CPU, pp_recv block, attention metadata, device forward), the decode cohorts from the flash-sched stats lines, stage-1 and stage-0 FIFO wait per chunk, pending replies, TTFT,
memory (c1's reads), and a pair table against the matched base arm. The first arm is cold (T.COLD_ARM): its pair is marked, c1's handling unchanged.

Two grades as c1: ``grade`` over the core arms (the queue verdict), ``read`` over every arm (a read; here the same arms). Receipts hold counts, hashes, identities, token ids and host
clock stamps, never prompts or auth.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import e3_alpha_next_c1_box as c1
import e3_alpha_next_c1b_box as c1b
import stage1_step_arms as T
import stage1_step_read as fread

an, progress, ap, base = c1.an, c1.progress, c1.ap, c1.base
rule = c1b.rule
_bound = c1._bound
STAGE_CONTROL = c1.STAGE_CONTROL
GRADE_SCHEMA = 'stage1-step-grade/1'
READ_SCHEMA = 'stage1-step-read/1'
STEP_COUNTERS = ('lazy_marked', 'merge_steps', 'merge_on', 'merge_off')
# Every action the arm's client sends, with its fields, in order (auth aside): each needs a real-router row (A1-h's lesson).
ACTIONS = (('pause', ()), ('native_progress', ('mode',)), ('native_qsa_fused', ('fused',)), ('native_layer_group', ('size',)),
           ('native_stage_local', ('local', 'overlap', 'defer_bind', 'lazy_recv', 'decode_merge')), ('release_auto_arm', ()), ('resume', ()),
           ('status', ()))
AXIS = ('alpha-next\'s C0 on/off names and c1\'s C1-on/off names in this record (over_c0_off, c0_decode_delta, vs_c1_off, memory_c0_on_vs_off, min_free_below_c0_off_mib, C1-on/C1-off '
        'table headings) name the selected arm (sl1, sl12, sl12a) against its matched base (soff); every arm is C1 defer, C0 local')
OBSERVE_ROLE = {'soff': 'measurement (C1 defer, base)', 'sl1': 'measurement (C1 defer, L1)', 'sl12': 'measurement (C1 defer, L1 + L2 during native)', 'sl12a': 'measurement (C1 defer, L1 + L2 always)'}
NOT_COVERED = ('numerics: this queue runs no fixed-schedule check; selected arms are speed reads, batch 8 against 2 x 4 changes batch composition so OUTSIDE on merged arms is expected and '
               'is not a defect, and the fixed-schedule check owed for C1 defer stays owed')

# --- the table as alpha-next's and c1's code read it ------------------------------------------------------

def table_view(sel=None):
    """This table as alpha-next's code reads alpha_next_arms. For the grade (``sel`` None): alpha-next's candidate axis is the lever (c0 'off' = soff, the controls; c0 'on' = the rest);
    ``selection`` and ``label`` take either name, and axis 'on' is sl12 at the core rounds, sl12a at the always round. C0 maps every value to stage-local True. For one arm (``sel``): the
    label and selection carry ``sel`` whatever value alpha-next passes ('on', C0 local); the record keeps c0 'on'."""
    names = {k: getattr(T, k) for k in c1._TABLE_NAMES}
    sel_of = dict({s: s for s in T.SELECTIONS}, off='soff', on='sl12')

    def axis(value):
        return 'off' if value == 'soff' else 'on'

    def arms(part=None):
        return [dict(r, c0=axis(r['c1'])) for r in T.arms(part)]

    def parse_selection(token):
        parsed = T.parse_selection(token)
        return None if parsed is None else (axis(parsed[0]), parsed[1])

    def label(value, size, cohort, index):
        chosen = sel_of[value]
        return T.label('sl12a' if chosen == 'sl12' and index == T.ALWAYS_ROUND else chosen, size, cohort, index)
    names.update(arms=arms, by_label=lambda: {r['label']: r for r in arms()}, C0={'on': True, 'off': True}, parse_selection=parse_selection,
                 selection=lambda value, size: T.selection(sel_of[value], size), label=label)
    if sel is not None:
        names.update(selection=lambda _c0, size: T.selection(sel, size),
                     label=lambda _c0, size, cohort, index: T.label(sel, size, cohort, index), C0={'on': True})
    return SimpleNamespace(**names)


# --- the arm --------------------------------------------------------------------------------------------

def stage_view(status):
    """c1's view plus the decode_step block, flat. A status without the block reads None (the grade refuses the arm)."""
    row = c1.stage_view(status)
    if row is None:
        return None
    block = status[STAGE_CONTROL]['overlap'].get('decode_step')
    block = block if type(block) is dict else {}
    counters = block.get('counters') if type(block.get('counters')) is dict else {}
    return dict(row, step_lazy_installed=block.get('lazy_recv_installed'), step_lazy=block.get('lazy_recv'), step_merge_installed=block.get('decode_merge_installed'),
                step_merge=block.get('decode_merge'), **{'step_' + k: counters.get(k) for k in STEP_COUNTERS})


def stage_delta(after, before):
    out = c1.stage_delta(after, before)
    out.update({'step_' + k: progress.delta(after, before, 'step_' + k) for k in STEP_COUNTERS})
    return out


def release_step(lazy, merge):
    """c1's release (C1 defer) with ``lazy_recv`` and ``decode_merge`` added to the stage command. The reply is recorded, never refused here."""
    extra = dict(lazy_recv=lazy, decode_merge=merge)

    def factory(size, local):
        release = c1.release_c1(True, True)(size, local)

        def run(client, record, mode):
            replies = {}

            class Selecting:
                def __getattr__(self, name):
                    return getattr(client, name)

                def call(self, action, **fields):
                    if action == STAGE_CONTROL:
                        fields = dict(fields, **extra)
                    reply = client.call(action, **fields)
                    if action == STAGE_CONTROL:
                        replies['stage'] = reply
                    return reply
            try:
                release(Selecting(), record, mode)
            finally:
                reply = replies.get('stage')
                replied = type(reply) is dict
                record['step'] = dict(requested_lazy=lazy, requested_merge=merge, replied=replied, acknowledged_lazy=reply.get('lazy_recv') if replied else None,
                                      acknowledged_merge=reply.get('decode_merge') if replied else None)
                if replied and type(record.get('stage_switch')) is dict:
                    record['stage_switch'].update(lazy_recv=reply.get('lazy_recv'), decode_merge=reply.get('decode_merge'))
        return run
    return factory


def stage_reasons(lazy, merge):
    inner = c1.stage_reasons_c1(True)

    def reasons(record, local):
        out = inner(record, local)
        views = {s.get('label'): s for s in record.get('progress') or []}
        before = (views.get('before-fresh') or {}).get(STAGE_CONTROL)
        after = (views.get('after-fresh') or {}).get(STAGE_CONTROL)
        if before and after and any(v.get('step_lazy') is not lazy or v.get('step_merge') != merge for v in (before, after)):
            out.append('stage-step-selection-not-this-arm')
        return out
    return reasons


def _arm_print(*args, **kwargs):
    text = ' '.join(str(x) for x in args)
    print(text.replace('ALPHA_NEXT_ARM ', 'STAGE1_STEP_ARM ', 1), **kwargs)


def arm_names(sel):
    lazy, merge = T.LAZY_RECV[sel], T.DECODE_MERGE[sel]
    return dict(T=table_view(sel), release_alpha_next=release_step(lazy, merge), stage_reasons=stage_reasons(lazy, merge),
                stage_view=stage_view, stage_delta=stage_delta, print=_arm_print)


def run_arm(a):
    """alpha-next's arm, C0 local ('on'), C1 defer, with this arm's lever selection."""
    args = argparse.Namespace(**dict(vars(a), stage_local='on'))
    with _bound(c1, dict(T=T)), _bound(an, arm_names(a.c1)):
        return an.run_arm(args)


# --- the gate ---------------------------------------------------------------------------------------------

def selection_check(arm, row):
    """(reasons, classification) of an arm whose selection is not proved to be the table's: c1's defer_bind_check's shape. An arm whose stage command never got a reply is left to
    alpha-next's rules (it is incomplete already)."""
    lazy, merge = row['lazy_recv'], row['decode_merge']
    rec = arm.get('step')
    if type(rec) is not dict or not {'requested_lazy', 'requested_merge', 'acknowledged_lazy', 'acknowledged_merge', 'replied'} <= set(rec):
        return ['arm-record-without-the-acknowledged-step-selection'], 'HARNESS'
    if rec['requested_lazy'] is not lazy or rec['requested_merge'] != merge:
        return ['arm-requested-another-step-selection-than-the-table'], 'HARNESS'
    if rec['replied'] is not True:
        return [], None
    reasons = []
    if rec['acknowledged_lazy'] is not lazy:
        reasons.append('lazy-recv-acknowledged-differs-from-requested')
    if rec['acknowledged_merge'] != merge:
        reasons.append('decode-merge-acknowledged-differs-from-requested')
    views = {v.get('label'): (v.get(STAGE_CONTROL) or {}) for v in arm.get('progress') or [] if type(v) is dict}
    for point in ('before-fresh', 'after-fresh'):
        view = views.get(point)
        if not view:
            continue                      # alpha-next's own reasons cover a missing status view
        if view.get('step_lazy_installed') is not True or view.get('step_merge_installed') is not True:
            reasons.append('step-lever-not-installed-at-' + point)
        elif view.get('step_lazy') is not lazy:
            reasons.append('lazy-recv-status-differs-from-requested-at-' + point)
        elif view.get('step_merge') != merge:
            reasons.append('decode-merge-status-differs-from-requested-at-' + point)
    return reasons, 'ENGINE' if reasons else None


def counter_check(arm, row):
    """(failures, inconclusive, counters): the status counters from the arm's stage delta, both ways."""
    delta = arm.get('stage_delta') or {}
    counters = {k: delta.get('step_' + k) for k in STEP_COUNTERS}
    if any(type(v) is not int for v in counters.values()):
        return [], ['step-counters-missing'], counters
    fails, incs = [], []
    for key, selected in (('lazy_marked', row['lazy_recv']), ('merge_steps', row['decode_merge'] != 'off')):
        if selected and counters[key] < 1:
            incs.append('step-selected-lever-never-moved:' + key)
        if not selected and counters[key] != 0:
            fails.append('step-unselected-lever-moved:' + key)
    return fails, incs, counters


def arm_window(progress_row):
    """(start_ns, end_ns) of the arm's native chunks from the grader's own rows, or None."""
    chunks = (progress_row or {}).get('chunks') or ()
    starts = [c['first_submit_ns'] for c in chunks if type(c.get('first_submit_ns')) is int]
    ends = [c['last_apply_ns'] for c in chunks if type(c.get('last_apply_ns')) is int]
    return (min(starts), max(ends)) if starts and ends else None


def gate(result, arms):
    """This queue's gate over c1's result, in place: the selection and the counters, per declared arm with a record."""
    failures, missing = set(result['failures']), set(result['missing'])
    classes = set(result['classification'] or ())
    declared, records = T.by_label(), {a.get('label'): a for a in arms if a.get('label')}
    rows = {}
    for name, v in result['validity'].items():
        row, arm = declared.get(name), records.get(name)
        if row is None or arm is None:
            continue
        reasons, cls = selection_check(arm, row)
        bad, inc, counters = counter_check(arm, row)
        if bad:
            reasons, cls = reasons + bad, cls or 'ENGINE'
        rec = arm.get('step') or {}
        v['step'] = dict(rec, table_lazy=row['lazy_recv'], table_merge=row['decode_merge'], counters=counters, reasons=reasons + inc)
        rows[name] = dict(c1=row['c1'], lazy_recv=row['lazy_recv'], decode_merge=row['decode_merge'], acknowledged_lazy=rec.get('acknowledged_lazy'),
                          acknowledged_merge=rec.get('acknowledged_merge'), counters=counters, reasons=reasons + inc)
        if reasons:
            v['reasons'] = list(v.get('reasons') or []) + reasons
            v['verdict'], v['classification'] = 'FAIL', cls
            failures.update(name + ':' + x for x in reasons)
            classes.add(cls)
        if inc:
            v['reasons'] = list(v.get('reasons') or []) + inc
            if v.get('verdict') != 'FAIL':
                v['verdict'] = 'INC'
            missing.update(name + ':' + x for x in inc)
    return failures, missing, classes, rows


# --- the reads --------------------------------------------------------------------------------------------

def _pair_row(nums, reads, on, off):
    one, zero = nums.get(on) or {}, nums.get(off) or {}
    rone, rzero = reads.get(on) or {}, reads.get(off) or {}
    vs = one.get('vs_c1_off') or {}

    def low_free(n):
        values = [v.get('min_free_mib') for v in (n.get('ranks') or {}).values() if v.get('min_free_mib') is not None]
        return min(values) if values else None

    def fifo(cell, stage):
        return ((cell.get('fifo') or {}).get(stage) or {}).get('median_s')

    def pair_ms(cell):
        return fread.slower((cell.get('step') or {}), 'pair_ms')
    return dict(on=on, off=off, cold_pair=T.COLD_ARM in (on, off), prefill_ratio=vs.get('prefill_ratio'), ttft_ratio=vs.get('ttft_ratio'),
                decode_ratio_delta=vs.get('decode_ratio_delta'), on_decode_ratio=one.get('decode_ratio'), off_decode_ratio=zero.get('decode_ratio'),
                on_prefill_tok_s=one.get('prefill_tok_s'), off_prefill_tok_s=zero.get('prefill_tok_s'), on_pair_ms=pair_ms(rone), off_pair_ms=pair_ms(rzero),
                on_stage1_fifo_s=fifo(rone, '1'), off_stage1_fifo_s=fifo(rzero, '1'), on_stage0_fifo_s=fifo(rone, '0'), off_stage0_fifo_s=fifo(rzero, '0'),
                on_min_free_mib=low_free(one), off_min_free_mib=low_free(zero))


def step_read(result, logs, arms):
    """The reads over the arms with a record. Never a verdict input: a read that cannot run says so in ``error``."""
    known = {a.get('label'): a for a in arms if a.get('label')}
    out = dict(is_gate=False, arms={}, pairs=[], error=None)
    try:
        records, errors = progress.read_lines(logs)
        flushes, problems = progress.assemble(records)
        intact, torn = c1.overlap_rows(logs)
        owner, faults = c1.overlap_owners([text for _s, text in intact])
        sched = ap.read_stamped(logs, lambda line: fread.SCHED_MARK in line)
        handlers = ap.read_stamped(logs, lambda line: fread.HANDLER_MARK in line)
        bytxn, workers = fread.index_rows(flushes, owner)
        out.update(progress_line_errors=len(errors), assemble_problems=len(problems), overlap_faults=len(faults), torn_overlap_lines=len(torn))
        rows = (result.get('progress') or {}).get('arms') or {}
        windows, counts = {}, {}
        for row in T.arms():
            name = row['label']
            arm = known.get(name)
            if arm is None:
                continue
            txns = fread.counted_txns(rows.get(name))
            window = arm_window(rows.get(name))
            if window:
                windows[name], counts[name] = window, len((rows.get(name) or {}).get('chunks') or ())
            out['arms'][name] = dict(
                c1=row['c1'], lazy_recv=row['lazy_recv'], decode_merge=row['decode_merge'], cold=name == T.COLD_ARM, counted_chunks=len(txns),
                fifo={str(stage): fread.fifo_arm(txns, bytxn, workers, stage) for stage in (0, 1)}, pending=fread.pending_arm(txns, bytxn),
                sched=fread.sched_read([text for stamp, text in sched if ap.in_span(stamp, arm)]))
        handler = fread.handler_read([text for _s, text in handlers], windows, counts)
        out['handler_malformed'] = handler['malformed']
        for name, cell in handler['arms'].items():
            out['arms'][name]['handler'] = cell
        steps = fread.step_rows([text for _s, text in handlers], windows)
        for name, cell in steps.items():
            out['arms'][name]['step'] = cell
    except (KeyError, TypeError, ValueError, OSError) as exc:
        out['error'] = '%s: %s' % (type(exc).__name__, exc)
    nums = ((result.get('arm_numbers') or {}).get('arms')) or {}
    for row in T.arms():
        partner = T.matched_off(row)
        if partner:
            out['pairs'].append(dict(_pair_row(nums, out['arms'], row['label'], partner), selection=row['c1'], band=row['band'], cohort=row['cohort'], index=row['index']))
    return out


# --- the grade --------------------------------------------------------------------------------------------

def grade(arms, ingress, checks, logs, watches=None, part='core'):
    """c1's grade over this table with c1b's C1-aware rules bound; then this queue's gate and reads, and the record fields."""
    c1b._C1B.clear()
    try:
        with _bound(c1b, dict(T=T)), _bound(c1, dict(T=T, table_view=table_view, OBSERVE_ROLE=OBSERVE_ROLE)), \
                _bound(progress, dict(progress_receipts=c1b.progress_receipts)), _bound(an, dict(ring_read=c1b.ring_read)):
            result = c1.grade(arms, ingress, checks, logs, watches, part=part)
        open_rows = dict(c1b._C1B.get('open_rows') or {})
    finally:
        c1b._C1B.clear()
    failures, missing, classes, rows = gate(result, arms)
    declared = T.by_label()
    for name, cell in result['table'].items():
        row = declared.get(name)
        if row:
            cell.update(lazy_recv=row['lazy_recv'], decode_merge=row['decode_merge'], c1=row['c1'])
    engine_open_rows = sorted(name for name, f in open_rows.items() if set(f['reasons']) & set(rule.ENGINE_REASONS))
    verdict = 'FAIL' if failures else 'INC' if missing else 'PASS'
    if verdict == 'FAIL':
        classes = classes or {'HARNESS'}
    elif verdict == 'INC' and engine_open_rows:
        classes = {'ENGINE'}           # check 4 of the open-rows rule: INC ENGINE, not FAIL (lead, c1b)
    else:
        classes = set()
    result.update(schema=GRADE_SCHEMA if part == 'core' else READ_SCHEMA, verdict=verdict, classification=sorted(classes) or None,
                  failures=sorted(failures), missing=sorted(missing), axis=AXIS,
                  step_gate=dict(arms=rows, counters=list(STEP_COUNTERS), selection_is_gate=True, counters_are_gate=True),
                  step_read=step_read(result, logs, arms), numerics_covered=False, not_covered=NOT_COVERED,
                  open_rows_rule=dict(rule=rule.RULE, supersedes=rule.SUPERSEDES, arms=open_rows, engine_classified_arms=engine_open_rows),
                  basis='one engine, FLASH_S5_LAZY_RECV=1 FLASH_S5_DECODE_MERGE=1 for the whole run, every arm C1 defer with fused, group size 2 and its lever selection by typed control; band 1 at '
                        '65,445 tokens over two counterbalanced rounds and one always-merged pair, band 2 matched at 131,094',
                  scope='L1 lazy PP receive and L2 decode merge against C1 defer, on alpha-next-c1b\'s engine; selection, counters, crash, OOM and one in flight are gated; speed, decode, TTFT, '
                        'FIFO wait, per-step thread time, cohorts, pending replies and memory are reads; not a release assessment and not a numerics result')
    if part != 'core':
        result['read_only'] = 'a read over every arm; the queue verdict is the core grade (stage1-step-grade.json)'
    return result


# --- the report ---------------------------------------------------------------------------------------------

_cell = c1._cell


def _step_cell(arm, field):
    """The field per stage-1 rank, 'rank 2 / rank 3'."""
    cell = arm.get('step') or {}
    return ' / '.join(_cell((cell.get(rank) or {}).get(field), 3) for rank in ('2', '3'))


def report(result):
    with _bound(c1, dict(T=T, table_view=table_view)):
        text = c1.report(result)
    text = text.replace('# q18a-alpha-next-c1 report', '# q18a-stage1-step report', 1)
    text = text.replace('# q18a-alpha-next-c1 read', '# q18a-stage1-step read', 1)
    gate_rows = (result.get('step_gate') or {}).get('arms') or {}
    read = result.get('step_read') or {}
    lines = ['', '## Stage-1 per-step levers', '', 'Not covered: ' + result['not_covered'] + '.', '', 'Axis: ' + result['axis'] + '.', '',
             '### Selection and counters (the gate)', '',
             '| arm | lazy | merge | ack lazy | ack merge | lazy marked | merge steps | merge on | merge off | reasons |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for name, row in gate_rows.items():
        c = row['counters']
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, row['lazy_recv'], row['decode_merge'], row['acknowledged_lazy'], row['acknowledged_merge'], _cell(c.get('lazy_marked')), _cell(c.get('merge_steps')),
            _cell(c.get('merge_on')), _cell(c.get('merge_off')), '; '.join(row['reasons']) or '-'))
    lines += ['', '### Pairs, selected arm against its matched base (reads, never gates)', '',
              '| band | cohort | round | sel | on prefill tok/s | off prefill tok/s | ratio | TTFT ratio | on decode | off decode | on pair ms | off pair ms | on S1 FIFO s | off S1 FIFO s | on S0 FIFO s | off S0 FIFO s | on min free MiB | off min free MiB | cold |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for p in read.get('pairs') or ():
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            p['band'], p['cohort'], p['index'], p['selection'], _cell(p['on_prefill_tok_s']), _cell(p['off_prefill_tok_s']), _cell(p['prefill_ratio'], 4), _cell(p['ttft_ratio'], 4),
            _cell(p['on_decode_ratio'], 4), _cell(p['off_decode_ratio'], 4), _cell(p['on_pair_ms'], 3), _cell(p['off_pair_ms'], 3), _cell(p['on_stage1_fifo_s'], 4), _cell(p['off_stage1_fifo_s'], 4),
            _cell(p['on_stage0_fifo_s'], 4), _cell(p['off_stage0_fifo_s'], 4), _cell(p['on_min_free_mib']), _cell(p['off_min_free_mib']), 'cold arm' if p['cold_pair'] else '-'))
    lines += ['', '### Per arm, stage-1 ranks 2 / 3 (reads; ms per RPC)', '', 'Read error: %s.' % (read.get('error') or 'none'), '',
              '| arm | counted chunks | S1 FIFO p50 s | full e n | empty e n | full e wall | full e CPU | empty e wall | sample wall | pair ms | pp_recv block mean | pp_recv block p90 | prep_attn_meta | device forward p50 | thread share | decoded even / odd | cohorts last |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, a in (read.get('arms') or {}).items():
        sched = a.get('sched') or {}
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s / %s | %s |' % (
            name, a['counted_chunks'], _cell(a['fifo']['1']['median_s'], 4), _step_cell(a, 'full_n'), _step_cell(a, 'empty_n'), _step_cell(a, 'full_wall_ms'), _step_cell(a, 'full_cpu_ms'),
            _step_cell(a, 'empty_wall_ms'), _step_cell(a, 'sample_wall_ms'), _step_cell(a, 'pair_ms'), _step_cell(a, 'pp_recv_block_ms'), _step_cell(a, 'pp_recv_block_p90_ms'),
            _step_cell(a, 'prep_attn_meta_ms'), _step_cell(a, 'dev_forward_p50_ms'), _step_cell(a, 'thread_share'), _cell(sched.get('dec_even')), _cell(sched.get('dec_odd')), sched.get('cohorts_last') or '-'))
    lines += ['', '### Open rows (%s, supersedes %s)' % (rule.RULE, rule.SUPERSEDES), '', '| arm | flushes checked | flagged | reasons |', '|---|---|---|---|']
    for name, f in sorted((result.get('open_rows_rule') or {}).get('arms', {}).items()):
        lines.append('| %s | %d | %d | %s |' % (name, f['checked'], len(f['flagged']), ', '.join(f['reasons']) or '-'))
    head, sep, tail = text.partition('\n## Mechanism\n')
    return head + '\n'.join(lines) + '\n' + sep + tail if sep else text + '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('arm')
    r.add_argument('--url', required=True)
    r.add_argument('--model', required=True)
    r.add_argument('--run-id', required=True)
    r.add_argument('--epoch', type=int, required=True)
    r.add_argument('--c1', choices=T.SELECTIONS, required=True, help='this arm\'s lever selection: soff, sl1, sl12 or sl12a; C1 defer in every arm')
    r.add_argument('--size', type=int, choices=T.SIZES, required=True)
    r.add_argument('--cohort', choices=progress.COHORTS, required=True)
    r.add_argument('--index', type=int, choices=T.INDEXES, required=True)
    r.add_argument('--workload', required=True)
    r.add_argument('--control-timeout-s', type=float, default=120.0)
    r.add_argument('--out', required=True)
    for action, text in (('grade', 'the core grade: the queue verdict'), ('read', 'every arm: a read only')):
        g = sub.add_parser(action, help=text)
        g.add_argument('--arm', action='append', required=True)
        g.add_argument('--ingress', action='append', default=[], help='label=path')
        g.add_argument('--watch', action='append', default=[], help='label=engine-watch record path')
        g.add_argument('--check', action='append', default=[], help='label=path')
        g.add_argument('--log', action='append', default=[], help='followed engine log (stdout/stderr)')
        g.add_argument('--report', help='write the report here')
        g.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if Path(a.out).exists():
        return 2
    if a.action == 'arm':
        return run_arm(a)
    arms = [base.load(x) or {'label': None, 'complete': False, 'reasons': ['arm-artifact-unreadable:' + Path(x).name]}
            for x in a.arm]
    result = grade(arms, ap.load_pairs(a.ingress), ap.load_pairs(a.check), a.log, ap.load_pairs(a.watch),
                   part='core' if a.action == 'grade' else 'all')
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    if a.report:
        with open(a.report, 'x') as handle:
            handle.write(report(result))
    print('STAGE1_STEP_%s %s' % ('GRADE' if a.action == 'grade' else 'READ', result['verdict']))
    if a.action == 'read':
        return 0 if result['verdict'] == 'PASS' else 2      # a read: never 1, the queue does not count it
    return {'PASS': 0, 'FAIL': 1}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
