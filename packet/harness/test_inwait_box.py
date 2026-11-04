#!/usr/bin/env python3
"""q18a-inwait box: the table, the arm's selection, the append rule, the gate both ways, the reads. CPU only; no pass here is box evidence. New logic only (the operator's discovery mode)."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import test_alpha_next_c1_queue as tq   # first: puts tools/ and the engine sources on the path
import test_attn_meta_box as tam
import inwait_arms as T
import e3_alpha_next_c1_box as c1box
import e3_attn_meta_box as am
import e3_inwait_box as box
from unittest import mock

an = c1box.an
tap = tq.tap
ROOT = Path(__file__).resolve().parents[1]
ORDER = ['ps0.g2-solo-0', 'ps1.g2-solo-0', 'ps0.g2-live-0', 'ps1.g2-live-0', 'ps1.g2-live-1', 'ps0.g2-live-1', 'pw.g2-solo-0', 'pw.g2-live-0', 'pw.g2-live-1']
QUIET = dict(lazy_marked=0, merge_steps=0, merge_on=0, merge_off=0)
MOVED = dict(lazy_marked=4, merge_steps=4, merge_on=1, merge_off=1)
STEP_VIEW = dict(defer_bind_installed=True, defer_bind_selected=True, step_lazy_installed=True, step_lazy=False, step_merge_installed=True, step_merge='off', **{'step_' + k: 0 for k in QUIET})


class Table(unittest.TestCase):
    def test_the_arms_in_order(self):
        self.assertEqual(T.labels(), ORDER)
        self.assertEqual(T.labels('tail'), [])
        self.assertEqual(T.COLD_ARM, ORDER[0])
        self.assertEqual([d['label'] for d in T.DECODE_ARMS], ['decode-am1'])
        self.assertEqual(T.REFERENCE_ARM, 'decode-am1')

    def test_each_selection_is_its_levers_and_c1_defer_group_2(self):
        for r in T.arms():
            self.assertEqual((r['attn'], r['lazy_recv'], r['decode_merge'], r['inwait']), (True, True, 'chunk', True), r['label'])
            self.assertIs(r['next_levers'], True, r['label'])
            self.assertIs(r['prepare_slices'], r['c1'] != 'ps0', r['label'])
            self.assertIs(r['score_warm'], r['c1'] == 'pw', r['label'])
            self.assertEqual((r['overlap'], r['defer_bind'], r['size']), (True, True, 2), r['label'])

    def test_the_view(self):
        view = box.table_view()
        self.assertEqual({r['c0'] for r in view.arms() if r['c1'] != 'ps0'}, {'on'})
        self.assertEqual({r['c0'] for r in view.arms() if r['c1'] == 'ps0'}, {'off'})
        self.assertEqual(view.label('on', 2, 'live', 1), 'ps1.g2-live-1')
        self.assertEqual(view.label('off', 2, 'live', 1), 'ps0.g2-live-1')
        self.assertEqual(box.table_view('ps0').label('on', 2, 'live', 1), 'ps0.g2-live-1')


class Client(tq.FakeClient):
    def call(self, action, **fields):
        reply = super().call(action, **fields)
        if action == 'native_stage_local':
            reply.update(lazy_recv=fields.get('lazy_recv'), decode_merge=fields.get('decode_merge'), inwait=fields.get('inwait'), early_successor=fields.get('early_successor'))
        return reply


class ArmControl(unittest.TestCase):
    def release(self, sel, client):
        row = next(r for r in T.arms() if r['c1'] == sel)
        order = []

        def select(_client, attn, record):
            order.append((attn, [a for a, _ in client.calls][-1]))
            record['attn_meta'] = dict(requested=attn, select={'0': {}})

        def slots(_client, attn, wanted, record):
            order.append(('levers', wanted, [a for a, _ in client.calls][-1]))
            for name, on in wanted.items():
                record[name] = dict(requested=on, select={'0': {}})
        with box._bound(an, box.arm_names(sel, row)), mock.patch.object(am, 'select_into', select), mock.patch.object(box, 'levers_into', slots):
            record = {}
            an.release_alpha_next(2, True)(client, record, 'burst')
        return record, order

    def test_every_selection_sends_its_fields_and_selects_inside_the_pause(self):
        for sel in T.SELECTIONS:
            client = Client()
            record, order = self.release(sel, client)
            self.assertEqual(dict(client.calls)['native_stage_local'], dict(local=True, overlap=True, defer_bind=True, lazy_recv=T.LAZY_RECV[sel], decode_merge=T.DECODE_MERGE[sel], inwait=T.INWAIT[sel], early_successor=T.NEXT[sel]))
            self.assertEqual(record['inwait'], dict(requested=T.INWAIT[sel], acknowledged=T.INWAIT[sel], early_requested=T.NEXT[sel], early_acknowledged=T.NEXT[sel]))
            self.assertEqual(dict(client.calls)['native_stage_local']['early_successor'], T.NEXT[sel])
            self.assertEqual(record['step'], dict(requested_lazy=T.LAZY_RECV[sel], requested_merge=T.DECODE_MERGE[sel], replied=True, acknowledged_lazy=T.LAZY_RECV[sel], acknowledged_merge=T.DECODE_MERGE[sel]))
            self.assertEqual(order, [('levers', dict(slots_fast=T.NEXT[sel], prepare_slices=T.SLICES[sel], score_warm=T.WARM[sel]), 'native_stage_local'), (T.ATTN[sel], 'native_stage_local')], sel)    # after the stage command, before release_auto_arm

    def test_the_actions_are_the_list_the_router_smoke_holds(self):
        client = Client()
        self.release('ps1', client)
        self.assertEqual([a for a, _ in client.calls], [a for a, _ in box.ACTIONS if a not in ('status', 'qualification_worker')])

    def test_no_select_is_an_error(self):
        row = next(r for r in T.arms() if r['c1'] == 'ps1')

        class Skips(Client):
            def __init__(self):
                super().__init__()

        with box._bound(an, box.arm_names('ps1', row)), mock.patch.object(am, '_Selecting', lambda client, attn, record: client), mock.patch.object(box, 'levers_into', lambda *a: None):
            with self.assertRaises(ValueError):
                an.release_alpha_next(2, True)(Skips(), {}, 'burst')


class AppendRule(unittest.TestCase):
    def record(self, **stage):
        return dict(reasons=[box.APPEND_REASON], complete=False, fused_delta_append=dict(a=0), group_delta_append=dict(b=0), stage_delta_append=dict(dict(c=0, step_lazy_marked=39), **stage))

    def test_only_step_keys_moved_is_exempt_and_the_delta_stays(self):
        record = self.record()
        self.assertIs(box.append_rule(record), True)
        self.assertEqual((record['reasons'], record['complete'], record['append_step_exempt'], record['stage_delta_append']['step_lazy_marked']), ([], True, dict(step_lazy_marked=39), 39))

    def test_any_other_counter_keeps_the_reason(self):
        for name, edit in (('stage', dict(stage_delta_append=dict(c=1, step_lazy_marked=39))), ('fused', dict(fused_delta_append=dict(a=1))), ('group', dict(group_delta_append=dict(b=2)))):
            record = dict(self.record(), **edit)
            self.assertIs(box.append_rule(record), False, name)
            self.assertEqual((record['reasons'], record['complete']), ([box.APPEND_REASON], False), name)
            self.assertNotIn('append_step_exempt', record)

    def test_another_reason_or_an_error_stays_incomplete(self):
        record = dict(self.record(), reasons=[box.APPEND_REASON, 'other'])
        box.append_rule(record)
        self.assertEqual((record['reasons'], record['complete']), (['other'], False))
        record = dict(self.record(), error=dict(type='x'))
        box.append_rule(record)
        self.assertIs(record['complete'], False)

    def test_the_real_arm_end_to_end_takes_rc_0(self):
        row = T.by_label()['ps1.g2-live-0']
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / 'arm.json')

            def inner(args):
                Path(args.out).write_text(json.dumps(self.record()))
                return 2
            args = mock.Mock(spec=['c1', 'size', 'cohort', 'index', 'out'], c1='ps1', size=2, cohort='live', index=0, out=out)
            with mock.patch.object(an, 'run_arm', inner), mock.patch.object(am, 'end_read', lambda a: dict(read=None)), mock.patch.object(box.argparse, 'Namespace', lambda **k: mock.Mock(out=k['out'])), \
                    mock.patch.object(box, 'vars', lambda _a: dict(out=out + '.inner.json'), create=True):
                self.assertEqual(box.run_arm(args), 0)
            record = json.loads(Path(out).read_text())
            self.assertIs(record['complete'], True)
            self.assertEqual(record['levers'], dict(attn=row['attn'], lazy_recv=True, decode_merge='chunk', inwait=True, next_levers=True, prepare_slices=row['prepare_slices'], score_warm=row['score_warm']))


def moved_of(row):
    return dict(QUIET, lazy_marked=4 if row['lazy_recv'] else 0, **({k: MOVED[k] for k in ('merge_steps', 'merge_on', 'merge_off')} if row['decode_merge'] != 'off' else {}))


class OpenRows(unittest.TestCase):
    """check_arm_inwait over the two flushes the rule reads; READING-box-1 item 1."""

    def flushes(self, carried):
        own = [dict(seq=1, t_submit=10, t_apply=100), dict(seq=2, t_submit=20, t_apply=90)]
        next_rows = carried + [dict(seq=9, t_submit=500, t_apply=600, origin='tick')]
        return [dict(line=1, identity=dict(txn=5, open_rows=len(carried)), rows=dict(n=own)),
                dict(line=2, identity=dict(txn=6, open_rows=0), rows=dict(n=next_rows))]

    OVERLAP = {5: (frozenset({1, 2}), 6), 6: (frozenset({3, 4, 9}), None)}

    def test_a_row_sent_from_the_wait_after_the_last_apply_is_carried(self):
        flushes = self.flushes([dict(seq=3, t_submit=95, t_apply=200, origin='wait'), dict(seq=4, t_submit=101, t_apply=210, origin='wait')])
        self.assertEqual(box.rule.check_arm(flushes, self.OVERLAP)['reasons'], ['c1-open-rows-not-carried'])
        out = box.check_arm_inwait(flushes, self.OVERLAP)
        self.assertEqual(out['reasons'], [])
        self.assertEqual((out['flagged'][0]['carried_seqs'], out['flagged'][0]['carried_from_wait']), ([3, 4], 1))

    def test_a_row_sent_by_the_tick_after_the_last_apply_stays_inc(self):
        flushes = self.flushes([dict(seq=3, t_submit=95, t_apply=200, origin='wait'), dict(seq=4, t_submit=101, t_apply=210, origin='tick')])
        self.assertEqual(box.check_arm_inwait(flushes, self.OVERLAP)['reasons'], ['c1-open-rows-not-carried'])

    def test_a_carried_row_that_is_not_the_successors_stays_inc(self):
        flushes = self.flushes([dict(seq=3, t_submit=95, t_apply=200, origin='wait'), dict(seq=7, t_submit=101, t_apply=210, origin='wait')])
        self.assertEqual(box.check_arm_inwait(flushes, self.OVERLAP)['reasons'], ['c1-open-rows-not-carried'])

    def test_too_few_rows_stays_inc(self):
        flushes = self.flushes([dict(seq=3, t_submit=95, t_apply=200, origin='wait')])
        flushes[0]['identity']['open_rows'] = 3
        self.assertEqual(box.check_arm_inwait(flushes, self.OVERLAP)['reasons'], ['c1-open-rows-not-carried'])

    def test_a_clean_flush_is_the_rule_unchanged(self):
        flushes = self.flushes([dict(seq=3, t_submit=95, t_apply=200, origin='wait')])
        self.assertEqual(box.check_arm_inwait(flushes, self.OVERLAP), box.rule.check_arm(flushes, self.OVERLAP))


def synthetic():
    emit, arms = tq.synthetic_run(rows=[dict(r, c1='defer') for r in T.arms()])
    declared = T.by_label()
    for a in arms:
        row = declared[a['label']]
        lazy, merge = row['lazy_recv'], row['decode_merge']
        a['step'] = dict(requested_lazy=lazy, requested_merge=merge, replied=True, acknowledged_lazy=lazy, acknowledged_merge=merge)
        a['stage_switch'].update(lazy_recv=lazy, decode_merge=merge)
        a['stage_delta'].update({'step_' + k: v for k, v in moved_of(row).items()})
        on, nxt = row['inwait'], row['next_levers']
        a['inwait'] = dict(requested=on, acknowledged=on, early_requested=nxt, early_acknowledged=nxt)
        a['slots_fast'] = dict(requested=nxt, calls={str(r): 40 if nxt else 0 for r in range(4)})
        per = lambda v: {str(r): v for r in range(4)}
        cut, warm = row['prepare_slices'], row['score_warm']
        a['prepare_slices'] = dict(requested=cut, counters=dict(sliced=per(60 if cut else 0), yields=per(70 if cut else 0), peek_missing=per(0)))
        a['score_warm'] = dict(requested=warm, counters=dict(allocs=per(1 if warm else 0), flushes=per(0), rewarms=per(0), errors=per(0)))
        a['stage_delta'].update({'step_iw_' + k: 0 for k in box.IW_DELTA})
        a['stage_delta'].update(step_iw_applied=30 if on else 0, step_iw_direct_opens=1 if on else 0, step_iw_direct_releases=1 if on else 0, step_iw_early_begins=30 if nxt else 0)
        a['progress'] = [dict(label=point, native_stage_local=dict(STEP_VIEW, step_lazy=lazy, step_merge=merge, step_iw_selected=on)) for point in ('before-fresh', 'after-fresh')]
        a['attn_meta'] = tam.block(row['attn'], [40] * 4 if row['attn'] else [0] * 4)
    return emit, arms


_RUN = {}


def run_of():
    if 'run' not in _RUN:
        _RUN['run'] = synthetic()
    emit, arms = _RUN['run']
    return emit, copy.deepcopy(arms)


def decode_records(rates=(440.0,)):
    out = []
    for d, rate in zip(T.DECODE_ARMS, rates):
        record = tam.decode_record(d, rate)
        record['attn_meta'] = tam.block(d['attn'], [40] * 4 if d['attn'] else [0] * 4)
        record.update(wall_start_s=1000.0, wall_end_s=1060.0)
        out.append(record)
    return out


def full_run():
    emit, arms = run_of()
    return emit, decode_records() + arms


def grade_with(root, emit, arms, startup=tam.STARTUP, box_module=box):
    q = Path(tempfile.mkdtemp(dir=root))
    tap.write_streams(q, emit.lines)
    extra = q / 'startup.log'
    extra.write_text(''.join('INFO x %s\n' % line for line in startup))
    logs = [str(q / 'engine-stream.log'), str(q / 'engine-stream.err'), str(extra)]
    return box_module.grade(arms, {a['label']: tap.PASS for a in arms}, tap.CHECKS, logs, {a['label']: tap.ALIVE for a in arms})


class Gate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def grade(self, emit, arms, **kw):
        return grade_with(self.tmp.name, emit, arms, **kw)

    def edited(self, name, edit):
        emit, arms = full_run()
        edit(next(a for a in arms if a['label'] == name))
        return self.grade(emit, arms)

    def test_the_record_passes_with_every_section(self):
        result = self.grade(*full_run())
        self.assertEqual(result['verdict'], 'PASS', (result['failures'], result['missing']))
        self.assertEqual(result['schema'], 'inwait-grade/1')
        self.assertIs(result['numerics_covered'], False)
        self.assertEqual(sorted(result['step_gate']['arms']), sorted(ORDER))
        self.assertEqual(sorted(result['attn_meta']), sorted(ORDER + ['decode-am1']))
        self.assertIsNone(result['step_read']['error'])
        self.assertEqual(len(result['step_read']['pairs']), 6)     # ps1 and pw, live x2 and solo
        read = result['compose_read']
        self.assertEqual(sorted(read['arms']), sorted(ORDER))
        cell = read['arms']['ps1.g2-live-0']
        self.assertIsNotNone(cell['prefill_tok_s'])
        self.assertIsNotNone(cell['decode_ratio_own_window'])
        self.assertEqual(cell['merge']['merge_steps'], 4)
        self.assertEqual(cell['attn_batched'], [40] * 4)
        during = result['table']['ps1.g2-live-0']['decode_during_tok_s']
        self.assertEqual(cell['decode_vs_fixed_reference'], round(during / T.FIXED_REFERENCE_TOK_S, 4))
        self.assertEqual(cell['decode_vs_this_run_am1'], round(during / 440.0, 4))
        self.assertEqual(cell['inwait']['applied'], 30)
        self.assertIsNotNone(read['pair']['live_on_over_off'])
        self.assertIn('ps1 band 1 live', read['cells'])
        text = box.report(result)
        self.assertTrue(text.startswith('# q18a-inwait'), text[:80])
        self.assertIn('## inwait reads', text)
        self.assertIn('Not covered: numerics', text)

    def test_attn_leak_and_not_engaged(self):
        result = self.edited('ps1.g2-live-1', lambda a: a['attn_meta'].update(tam.block(True, [0, 40, 40, 40])))
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('ps1.g2-live-1:attn-meta-not-engaged', result['missing'])

    def test_step_gate_kept(self):
        result = self.edited('ps0.g2-live-0', lambda a: a['stage_delta'].update(step_merge_steps=0))
        self.assertIn('ps0.g2-live-0:step-selected-lever-never-moved:merge_steps', result['missing'])

    def test_inwait_check(self):
        result = self.edited('ps1.g2-live-0', lambda a: a['stage_delta'].update(step_iw_applied=0))
        self.assertIn('ps1.g2-live-0:inwait-selected-applied-nothing-in-the-wait', result['failures'])
        result = self.edited('ps0.g2-live-0', lambda a: a['prepare_slices']['counters']['sliced'].update({'0': 2}))
        self.assertIn('ps0.g2-live-0:prepare-slices-off-moved-its-counter', result['failures'])
        result = self.edited('ps1.g2-live-0', lambda a: a['prepare_slices']['counters']['sliced'].update({'3': 0}))
        self.assertIn('ps1.g2-live-0:prepare-slices-selected-not-engaged', result['missing'])
        result = self.edited('ps1.g2-live-0', lambda a: a['prepare_slices'].pop('counters'))
        self.assertIn('ps1.g2-live-0:prepare-slices-counters-absent', result['missing'])
        result = self.edited('ps1.g2-live-0', lambda a: a['score_warm']['counters']['allocs'].update({'0': 1}))
        self.assertIn('ps1.g2-live-0:score-warm-off-moved-its-counter', result['failures'])
        result = self.edited('pw.g2-live-0', lambda a: a['score_warm']['counters']['errors'].update({'2': 1}))
        self.assertIn('pw.g2-live-0:score-warm-errors', result['missing'])
        result = self.edited('ps1.g2-live-0', lambda a: a['stage_delta'].update(step_iw_early_begins=0))
        self.assertIn('ps1.g2-live-0:early-successor-selected-not-engaged', result['missing'])
        result = self.edited('ps1.g2-live-1', lambda a: a['slots_fast'].update(calls={'0': 0, '1': 40, '2': 40, '3': 40}))
        self.assertIn('ps1.g2-live-1:slots-fast-selected-not-engaged', result['missing'])
        result = self.edited('ps1.g2-live-1', lambda a: a['inwait'].update(early_acknowledged=False))
        self.assertIn('ps1.g2-live-1:early-successor-selection-not-acknowledged', result['failures'])
        result = self.edited('ps1.g2-live-1', lambda a: a['inwait'].update(acknowledged=False))
        self.assertIn('ps1.g2-live-1:inwait-selection-not-acknowledged', result['failures'])
        result = self.edited('ps1.g2-solo-0', lambda a: a['stage_delta'].update(step_iw_applied=0))
        self.assertEqual(result['verdict'], 'PASS', 'a solo arm may apply nothing in the wait')

    def test_decode_arm_missing_or_a_leak(self):
        emit, arms = run_of()
        result = self.grade(emit, arms)
        self.assertIn('decode-am1:no-arm-record', result['missing'])

    def test_startup_line_missing_is_inc(self):
        result = self.grade(*full_run(), startup=tam.STARTUP[:2] + tam.STARTUP[3:])
        self.assertIn('attn-meta-startup-line-missing:rank-2', result['missing'])

    def test_a_kept_append_reason_stays_inc(self):
        result = self.edited('ps1.g2-live-1', lambda a: a.update(complete=False, reasons=[box.APPEND_REASON]))
        self.assertNotEqual(result['verdict'], 'PASS')


if __name__ == '__main__':
    unittest.main()
