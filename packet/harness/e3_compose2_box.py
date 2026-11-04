#!/usr/bin/env python3
"""q18a-compose-2 box arms and grade: attn-meta and stage1-step's L1 + L2 stacked, every prefill arm C1 defer at group size 2. Discovery mode.

Supersedes e3_alpha_next_c1_box (``c1``) by name, as e3_stage1_step_box (``sb``) does, and uses sb's arm parts, gate and read and e3_attn_meta_box's (``am``) select, end read, check and decode-only arm,
sb's with this table bound. Nothing of theirs is edited.

The arm: alpha-next's arm with c1's stage control. Control order: pause, native_progress {burst}, native_qsa_fused {true}, native_layer_group {size},
native_stage_local {local: true, overlap: true, defer_bind: true, lazy_recv, decode_merge}, qualification_worker attn_meta_select {on}, release_auto_arm, resume, then status; after the arm's own record
one end read (pause, attn_meta_read, resume). ACTIONS below is the list the router smoke holds. The stage reply is recorded, never refused here; the attn-meta select raises when it does not prove itself,
as in its own box. The decode-only baselines are am's decode arm, unchanged (G1, C1 off, no lever field: L1 and L2 at their defaults, off).

The append rule: alpha-next's arm marks itself incomplete when the stock append moves a fused, group or stage counter. L1 marks every RPC of an arm that selects it, the append's too, so
``step_lazy_marked`` moves there by design (q18a-stage1-step's box records: the only key that moved). append_rule exempts the ``step_*`` keys and only those; the delta stays in the record, and any other
counter that moves keeps the reason and the INC (the lead's scope, 4 Nov).

The grade: c1's grade over this table with c1b's C1-aware rules and sb's gate (selection and counters), then am's check (selection, engagement, leak, order lines, startup lines) on every arm and the
decode arms' verdict. Speed is never a gate. Numerics are not covered: batch 8 against 2 x 4 changes batch composition.

Reads, never gates (compose_read, sb's step_read): per arm and per cell live and solo prefill tok/s, decode ratio on its own window and against the fixed reference (decode-am1), stage-1 thread share and
device forward, prep_attn_meta, merge counters, attn-meta batched counts; the band 1 live stack: each lever's gain on b0 and both against the product of the two.
"""
import argparse
import json
from pathlib import Path

import compose2_arms as T
import e3_alpha_next_c1_box as c1
import e3_alpha_next_c1b_box as c1b
import e3_attn_meta_box as am
import e3_stage1_step_box as sb
from types import SimpleNamespace

an, progress, ap, base = c1.an, c1.progress, c1.ap, c1.base
fread = sb.fread
rule = c1b.rule
_bound = c1._bound
STAGE_CONTROL = c1.STAGE_CONTROL
GRADE_SCHEMA = 'compose-2-grade/1'
READ_SCHEMA = 'compose-2-read/1'
ARM_SCHEMA = 'compose-2-arm/1'
APPEND_REASON = 'append-moved-the-fused-group-or-stage-counters'
ACTIONS = (('pause', ()), ('native_progress', ('mode',)), ('native_qsa_fused', ('fused',)), ('native_layer_group', ('size',)),
           ('native_stage_local', ('local', 'overlap', 'defer_bind', 'lazy_recv', 'decode_merge')), ('qualification_worker', ('worker_action', 'payload')),
           ('release_auto_arm', ()), ('resume', ()), ('status', ()))
AXIS = ("alpha-next's C0 on/off names and c1's C1-on/off names in this record (over_c0_off, c0_decode_delta, vs_c1_off, memory_c0_on_vs_off, min_free_below_c0_off_mib, C1-on/C1-off table headings) name "
        'a lever arm (am, mg, amg) against its matched b0; every arm is C1 defer, C0 local')
OBSERVE_ROLE = {'b0': 'measurement (C1 defer, levers off)', 'am': 'measurement (C1 defer, attn-meta)', 'mg': 'measurement (C1 defer, L1 + L2 during native)',
                'amg': 'measurement (C1 defer, attn-meta and L1 + L2 during native)'}
NOT_COVERED = ('numerics: this queue runs no fixed-schedule check; lever arms are speed reads, batch 8 against 2 x 4 changes batch composition so OUTSIDE on merged arms is expected and is not a defect, '
               'and the fixed-schedule check owed for C1 defer stays owed')
BASIS = ('one engine, FLASH_ATTN_META_GDN_BATCH=1 FLASH_S5_LAZY_RECV=1 FLASH_S5_DECODE_MERGE=1 for the whole run, every prefill arm C1 defer with fused, group size 2 and its lever selection by typed '
         'control; two decode-only baselines first; band 1 at 65,445 tokens live over two counterbalanced rounds and solo once, band 2 live at 131,094')
SCOPE = ("attn-meta and L1 + L2 stacked on alpha-next-c1b's engine; selection, counters, crash, OOM and one in flight are gated; speed, decode, per-step thread time and memory are reads; "
         'not a release assessment and not a numerics result')


def table_view(sel=None):
    """This table as alpha-next's code reads alpha_next_arms: the candidate axis is lever arm against b0 (c0 'off' = b0; c0 'on' = am, mg, amg, and 'on' names amg). For one arm the label and selection
    carry ``sel`` whatever value alpha-next passes; the record keeps c0 'on'."""
    names = {k: getattr(T, k) for k in c1._TABLE_NAMES}
    sel_of = dict({s: s for s in T.SELECTIONS}, off='b0', on='amg')

    def axis(value):
        return 'off' if value == 'b0' else 'on'

    def arms(part=None):
        return [dict(r, c0=axis(r['c1'])) for r in T.arms(part)]

    def parse_selection(token):
        parsed = T.parse_selection(token)
        return None if parsed is None else (axis(parsed[0]), parsed[1])
    names.update(arms=arms, by_label=lambda: {r['label']: r for r in arms()}, C0={'on': True, 'off': True}, parse_selection=parse_selection,
                 selection=lambda value, size: T.selection(sel_of[value], size),
                 label=lambda value, size, cohort, index: T.label(sel_of[value], size, cohort, index))
    if sel is not None:
        names.update(selection=lambda _c0, size: T.selection(sel, size),
                     label=lambda _c0, size, cohort, index: T.label(sel, size, cohort, index), C0={'on': True})
    return SimpleNamespace(**names)


def sb_bindings():
    """sb's names over this table, for one call."""
    return dict(T=T, table_view=table_view, OBSERVE_ROLE=OBSERVE_ROLE, AXIS=AXIS, NOT_COVERED=NOT_COVERED, GRADE_SCHEMA=GRADE_SCHEMA, READ_SCHEMA=READ_SCHEMA)


# --- the arm --------------------------------------------------------------------------------------------

def release_compose(row):
    """sb's release (c1's C1 defer with both step fields on the stage command) with attn_meta_select just before release_auto_arm; the record must show the select ran."""
    inner = sb.release_step(row['lazy_recv'], row['decode_merge'])

    def factory(size, local):
        run_inner = inner(size, local)

        def run(client, record, mode):
            run_inner(am._Selecting(client, row['attn'], record), record, mode)
            if (record.get('attn_meta') or {}).get('select') is None:
                raise ValueError('attn-meta-select-not-sent')
        return run
    return factory


def _arm_print(*args, **kwargs):
    text = ' '.join(str(x) for x in args)
    print(text.replace('ALPHA_NEXT_ARM ', 'COMPOSE2_ARM ', 1), **kwargs)


def arm_names(sel, row):
    return dict(T=table_view(sel), release_alpha_next=release_compose(row), stage_reasons=sb.stage_reasons(row['lazy_recv'], row['decode_merge']),
                stage_view=sb.stage_view, stage_delta=sb.stage_delta, print=_arm_print)


def append_rule(record):
    """Exempt the step_* keys from alpha-next's append rule, in place (module text). True when the reason went away; the delta stays in the record either way."""
    if APPEND_REASON not in (record.get('reasons') or ()):
        return False
    stage = record.get('stage_delta_append') or {}
    other = [record.get('fused_delta_append') or {}, record.get('group_delta_append') or {}, {k: v for k, v in stage.items() if not k.startswith('step_')}]
    if any(v != 0 for block in other for v in block.values()):
        return False
    record['reasons'] = [r for r in record['reasons'] if r != APPEND_REASON]
    record['append_step_exempt'] = {k: v for k, v in stage.items() if k.startswith('step_') and v != 0}
    record['complete'] = not record['reasons'] and not record.get('error')
    return True


def run_arm(a):
    """alpha-next's arm, C0 local ('on'), C1 defer, with this arm's selection; the append rule; then attn-meta's end read. ``a.out`` = the arm's record + attn_meta."""
    name = T.label(a.c1, a.size, a.cohort, a.index)
    row = T.by_label().get(name)
    if row is None:
        record, rc = dict(schema=ARM_SCHEMA, label=name, complete=False, reasons=['arm-not-in-the-declared-table']), 2
    else:
        inner = a.out + '.inner.json'
        args = argparse.Namespace(**dict(vars(a), stage_local='on', out=inner))
        with _bound(c1, dict(T=T)), _bound(an, arm_names(a.c1, row)):
            rc = an.run_arm(args)
        record = base.load(inner) or dict(label=name, complete=False, reasons=['arm-record-unreadable'])
        if append_rule(record) and rc == 2 and record['complete']:
            rc = 0
        block = record.setdefault('attn_meta', dict(requested=row['attn'], select=None))
        block['end'] = am.end_read(a)
        block['deltas'] = am.deltas(block.get('select'), (block['end'] or {}).get('read'))
        block['table'] = row['attn']
        record.update(compose2_schema=ARM_SCHEMA, inner_record=Path(inner).name, levers=dict(attn=row['attn'], lazy_recv=row['lazy_recv'], decode_merge=row['decode_merge']))
    with open(a.out, 'x') as handle:
        json.dump(record, handle, sort_keys=True)
    print('COMPOSE2_ARM_RECORD %s %s' % (name, 'COMPLETE' if record.get('complete') is True else 'INCOMPLETE'))
    return 0 if record.get('complete') is True and rc == 0 else 2


# --- the gate ---------------------------------------------------------------------------------------------

def attn_gate(arms, decode_records, ingress, watches, logs, want):
    """am's per-arm check over the prefill arms, the decode arms' verdict and am's startup check; (failures, missing, classes, per-arm facts, startup)."""
    records, attn = {a.get('label'): a for a in arms if a.get('label')}, {}
    failures, missing, classes = set(), set(), set()
    for row in want:
        rec = records.get(row['label'])
        if rec is None:
            continue                       # c1's grade already counts the missing record
        f, m, facts = am.attn_check(rec, row['attn'])
        attn[row['label']] = dict(facts, failures=f, missing=m)
        failures.update(f)
        missing.update(m)
        if f:
            classes.add('HARNESS' if any(x.endswith('requested-differs-from-table') for x in f) else 'ENGINE')
    for d in T.DECODE_ARMS:
        rec = decode_records.get(d['label'])
        if rec is None:
            missing.add(d['label'] + ':no-arm-record')
            continue
        f, m = am.decode_verdict(rec, (ingress or {}).get(d['label']), (watches or {}).get(d['label']))
        af, amiss, facts = am.attn_check(rec, d['attn'])
        attn[d['label']] = dict(facts, failures=af, missing=amiss)
        failures.update(f + af)
        missing.update(m + amiss)
        if f or af:
            classes.add('ENGINE')
    start = am.startup_check(logs)
    failures.update(start['failures'])
    missing.update(start['missing'])
    if start['failures']:
        classes.add('ENGINE')
    return failures, missing, classes, attn, start


# --- the reads --------------------------------------------------------------------------------------------

def _ratio(x, y):
    return round(x / y, 4) if type(x) in (int, float) and type(y) in (int, float) and y else None


def compose_read(result, decode_records, attn):
    """The lead's table: per arm, per (selection, band, cohort), the decode baselines and the stack. A read."""
    table, gate_rows, step_arms = result.get('table') or {}, (result.get('step_gate') or {}).get('arms') or {}, (result.get('step_read') or {}).get('arms') or {}
    rate = {d['label']: ((decode_records.get(d['label']) or {}).get('decode_only') or {}).get('aggregate_tokens_per_s') for d in T.DECODE_ARMS}
    reference = rate[T.REFERENCE_ARM]
    arms = {}
    for row in T.arms():
        cell, step, facts = table.get(row['label']) or {}, (step_arms.get(row['label']) or {}).get('step') or {}, attn.get(row['label']) or {}
        during = cell.get('decode_during_tok_s')
        arms[row['label']] = dict(
            c1=row['c1'], band=row['band'], cohort=row['cohort'], index=row['index'], cold=row['label'] == T.COLD_ARM, prefill_tok_s=cell.get('prefill_tok_s'), decode_during_tok_s=during,
            decode_ratio_own_window=cell.get('decode_ratio'), decode_vs_fixed_reference=_ratio(during, reference), decode_vs_am0=_ratio(during, rate['decode-am0']),
            stage1_thread_share=fread.slower(step, 'thread_share'), stage1_device_forward_p50_ms=fread.slower(step, 'dev_forward_p50_ms'), stage1_pair_ms=fread.slower(step, 'pair_ms'),
            stage1_prep_attn_meta_ms=fread.slower(step, 'prep_attn_meta_ms'), stage1_full_steps=fread.slower(step, 'full_n'), merge=(gate_rows.get(row['label']) or {}).get('counters'),
            attn_batched=facts.get('batched'), attn_fallback=facts.get('fallback'))
    cells = {}
    for name, r in arms.items():
        cells.setdefault((r['c1'], r['band'], r['cohort']), []).append(r)
    medians = {'%s band %d %s' % key: {k: progress.median([r[k] for r in rows if type(r[k]) in (int, float)] or [None]) if any(type(r[k]) in (int, float) for r in rows) else None
                                       for k in ('prefill_tok_s', 'decode_ratio_own_window', 'decode_vs_fixed_reference', 'stage1_thread_share', 'stage1_device_forward_p50_ms', 'stage1_pair_ms')}
               for key, rows in sorted(cells.items())}

    def live(sel):
        return (medians.get('%s band 1 live' % sel) or {}).get('prefill_tok_s')
    gain_am, gain_mg, gain_amg = _ratio(live('am'), live('b0')), _ratio(live('mg'), live('b0')), _ratio(live('amg'), live('b0'))
    stack = dict(am_on_b0=gain_am, mg_on_b0=gain_mg, amg_on_b0=gain_amg, product_of_the_two=round(gain_am * gain_mg, 4) if gain_am and gain_mg else None,
                 amg_over_product=_ratio(gain_amg, gain_am * gain_mg) if gain_am and gain_mg else None)
    return dict(is_gate=False, reference_arm=T.REFERENCE_ARM, decode_only={k: v for k, v in rate.items()}, am1_over_am0=_ratio(rate['decode-am1'], rate['decode-am0']), arms=arms, cells=medians, stack=stack)


# --- the grade --------------------------------------------------------------------------------------------

def grade(arms, ingress, checks, logs, watches=None, part='core'):
    """sb's grade over this table, then am's check and the decode arms; the reads and the record fields."""
    known = [a for a in arms if a.get('label')]
    decode_records = {a['label']: a for a in known if a.get('kind') == 'decode-only'}
    prefill = [a for a in arms if a.get('kind') != 'decode-only']
    with _bound(sb, sb_bindings()):
        result = sb.grade(prefill, ingress, checks, logs, watches, part=part)
    afail, amiss, acls, attn, start = attn_gate(prefill, decode_records, ingress, watches, logs, T.arms(None if part == 'all' else T.GRADED))
    failures, missing = set(result['failures']) | afail, set(result['missing']) | amiss
    classes = set(result['classification'] or ()) | acls
    for name, cell in result['table'].items():
        row = T.by_label().get(name)
        if row:
            cell.update(attn=row['attn'])
    verdict = 'FAIL' if failures else 'INC' if missing else 'PASS'
    if verdict == 'FAIL':
        classes = classes or {'HARNESS'}
    elif verdict == 'INC' and result['open_rows_rule']['engine_classified_arms']:
        classes = {'ENGINE'}
    else:
        classes = set()
    result.update(verdict=verdict, classification=sorted(classes) or None, failures=sorted(failures), missing=sorted(missing), attn_meta=attn, attn_meta_is_gate=True, startup=start,
                  compose_read=compose_read(result, decode_records, attn), basis=BASIS, scope=SCOPE,
                  append_step_exempt={a['label']: a['append_step_exempt'] for a in known if a.get('append_step_exempt')})
    if part != 'core':
        result['read_only'] = 'a read over every arm; the queue verdict is the core grade (compose-2-grade.json)'
    return result


# --- the report ---------------------------------------------------------------------------------------------

_cell = c1._cell


def report(result):
    with _bound(sb, sb_bindings()):
        text = sb.report(result)
    text = text.replace('# q18a-stage1-step report', '# q18a-compose-2 report', 1).replace('# q18a-stage1-step read', '# q18a-compose-2 read', 1)
    read = result['compose_read']
    lines = ['## compose-2 reads', '', 'Stack, band 1 live medians: %s.' % json.dumps(read['stack'], sort_keys=True), '',
             'Decode-only tok/s: %s; am1 over am0 %s; fixed reference %s.' % (json.dumps(read['decode_only'], sort_keys=True), _cell(read['am1_over_am0'], 4), read['reference_arm']), '',
             'Append rule, step_* exempted, delta kept: %s.' % (json.dumps(result['append_step_exempt'], sort_keys=True) or '{}'), '',
             '| arm | prefill tok/s | decode ratio own | decode vs reference | S1 thread share | S1 forward p50 ms | S1 pair ms | prep_attn_meta ms | full steps | merge steps | batched (rank 2) | cold |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, r in read['arms'].items():
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, _cell(r['prefill_tok_s']), _cell(r['decode_ratio_own_window'], 4), _cell(r['decode_vs_fixed_reference'], 4), _cell(r['stage1_thread_share'], 3), _cell(r['stage1_device_forward_p50_ms'], 3),
            _cell(r['stage1_pair_ms'], 3), _cell(r['stage1_prep_attn_meta_ms'], 3), _cell(r['stage1_full_steps']), _cell((r['merge'] or {}).get('merge_steps')),
            _cell((r['attn_batched'] or [None] * 3)[2]), 'cold arm' if r['cold'] else '-'))
    lines += ['', '| cell | prefill tok/s | decode ratio own | decode vs reference | S1 thread share | S1 forward p50 ms | S1 pair ms |', '|---|---|---|---|---|---|---|']
    for name, m in read['cells'].items():
        lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (name, _cell(m['prefill_tok_s']), _cell(m['decode_ratio_own_window'], 4), _cell(m['decode_vs_fixed_reference'], 4),
                                                              _cell(m['stage1_thread_share'], 3), _cell(m['stage1_device_forward_p50_ms'], 3), _cell(m['stage1_pair_ms'], 3)))
    head, sep, tail = text.partition('\n## Stage-1 per-step levers\n')
    return head + '\n' + '\n'.join(lines) + '\n' + sep + tail if sep else text + '\n' + '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('arm')
    r.add_argument('--url', required=True)
    r.add_argument('--model', required=True)
    r.add_argument('--run-id', required=True)
    r.add_argument('--epoch', type=int, required=True)
    r.add_argument('--c1', choices=T.SELECTIONS, required=True, help="this arm's lever selection: b0, am, mg or amg; C1 defer in every arm")
    r.add_argument('--size', type=int, choices=T.SIZES, required=True)
    r.add_argument('--cohort', choices=progress.COHORTS, required=True)
    r.add_argument('--index', type=int, choices=T.INDEXES, required=True)
    r.add_argument('--workload', required=True)
    r.add_argument('--control-timeout-s', type=float, default=120.0)
    r.add_argument('--out', required=True)
    d = sub.add_parser('decode-arm')
    for flag in ('--url', '--model', '--run-id', '--workload', '--out'):
        d.add_argument(flag, required=True)
    d.add_argument('--epoch', type=int, required=True)
    d.add_argument('--label', choices=tuple(x['label'] for x in T.DECODE_ARMS), required=True)
    d.add_argument('--attn', choices=('on', 'off'), required=True)
    d.add_argument('--control-timeout-s', type=float, default=120.0)
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
    if a.action == 'decode-arm':
        return am.run_decode_arm(a)
    arms = [base.load(x) or {'label': None, 'complete': False, 'reasons': ['arm-artifact-unreadable:' + Path(x).name]} for x in a.arm]
    result = grade(arms, ap.load_pairs(a.ingress), ap.load_pairs(a.check), a.log, ap.load_pairs(a.watch), part='core' if a.action == 'grade' else 'all')
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    if a.report:
        with open(a.report, 'x') as handle:
            handle.write(report(result))
    print('COMPOSE2_%s %s' % ('GRADE' if a.action == 'grade' else 'READ', result['verdict']))
    if a.action == 'read':
        return 0 if result['verdict'] == 'PASS' else 2      # a read: never 1, the queue does not count it
    return {'PASS': 0, 'FAIL': 1}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
