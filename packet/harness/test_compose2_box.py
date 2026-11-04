#!/usr/bin/env python3
"""q18a-compose-2 box: the table, the arm's selection, the append rule, the gate both ways, the reads. CPU only; no pass here is box evidence. New logic only (the operator's discovery mode)."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import test_alpha_next_c1_queue as tq   # first: puts tools/ and the engine sources on the path
import test_attn_meta_box as tam
import compose2_arms as T
import e3_alpha_next_c1_box as c1box
import e3_attn_meta_box as am
import e3_compose2_box as box
from unittest import mock

an = c1box.an
tap = tq.tap
ROOT = Path(__file__).resolve().parents[1]
ORDER = ['b0.g2-solo-0', 'amg.g2-solo-0', 'b0.g2-live-0', 'am.g2-live-0', 'mg.g2-live-0', 'amg.g2-live-0', 'amg.g2-live-1', 'mg.g2-live-1', 'am.g2-live-1', 'b0.g2-live-1', 'amg.g2-live-2', 'b0.g2-live-2']
QUIET = dict(lazy_marked=0, merge_steps=0, merge_on=0, merge_off=0)
MOVED = dict(lazy_marked=4, merge_steps=4, merge_on=1, merge_off=1)
STEP_VIEW = dict(defer_bind_installed=True, defer_bind_selected=True, step_lazy_installed=True, step_lazy=False, step_merge_installed=True, step_merge='off', **{'step_' + k: 0 for k in QUIET})


class Table(unittest.TestCase):
    def test_the_arms_in_order(self):
        self.assertEqual(T.labels(), ORDER)
        self.assertEqual(T.labels('tail'), [])
        self.assertEqual(T.COLD_ARM, ORDER[0])
        self.assertEqual([d['label'] for d in T.DECODE_ARMS], ['decode-am0', 'decode-am1'])
        self.assertEqual(T.REFERENCE_ARM, 'decode-am1')

    def test_each_selection_is_its_levers_and_c1_defer_group_2(self):
        want = dict(b0=(False, False, 'off'), am=(True, False, 'off'), mg=(False, True, 'chunk'), amg=(True, True, 'chunk'))
        for r in T.arms():
            self.assertEqual((r['attn'], r['lazy_recv'], r['decode_merge']), want[r['c1']], r['label'])
            self.assertEqual((r['overlap'], r['defer_bind'], r['size']), (True, True, 2), r['label'])

    def test_the_view(self):
        view = box.table_view()
        self.assertEqual({r['c0'] for r in view.arms() if r['c1'] != 'b0'}, {'on'})
        self.assertEqual({r['c0'] for r in view.arms() if r['c1'] == 'b0'}, {'off'})
        self.assertEqual(view.label('on', 2, 'live', 1), 'amg.g2-live-1')
        self.assertEqual(view.label('off', 2, 'live', 1), 'b0.g2-live-1')
        self.assertEqual(box.table_view('mg').label('on', 2, 'live', 1), 'mg.g2-live-1')


class Client(tq.FakeClient):
    def call(self, action, **fields):
        reply = super().call(action, **fields)
        if action == 'native_stage_local':
            reply.update(lazy_recv=fields.get('lazy_recv'), decode_merge=fields.get('decode_merge'))
        return reply


class ArmControl(unittest.TestCase):
    def release(self, sel, client):
        row = next(r for r in T.arms() if r['c1'] == sel)
        order = []

        def select(_client, attn, record):
            order.append((attn, [a for a, _ in client.calls][-1]))
            record['attn_meta'] = dict(requested=attn, select={'0': {}})
        with box._bound(an, box.arm_names(sel, row)), mock.patch.object(am, 'select_into', select):
            record = {}
            an.release_alpha_next(2, True)(client, record, 'burst')
        return record, order

    def test_every_selection_sends_its_fields_and_selects_inside_the_pause(self):
        for sel in T.SELECTIONS:
            client = Client()
            record, order = self.release(sel, client)
            self.assertEqual(dict(client.calls)['native_stage_local'], dict(local=True, overlap=True, defer_bind=True, lazy_recv=T.LAZY_RECV[sel], decode_merge=T.DECODE_MERGE[sel]))
            self.assertEqual(record['step'], dict(requested_lazy=T.LAZY_RECV[sel], requested_merge=T.DECODE_MERGE[sel], replied=True, acknowledged_lazy=T.LAZY_RECV[sel], acknowledged_merge=T.DECODE_MERGE[sel]))
            self.assertEqual(order, [(T.ATTN[sel], 'native_stage_local')], sel)    # after the stage command, before release_auto_arm

    def test_the_actions_are_the_list_the_router_smoke_holds(self):
        client = Client()
        self.release('amg', client)
        self.assertEqual([a for a, _ in client.calls], [a for a, _ in box.ACTIONS if a not in ('status', 'qualification_worker')])

    def test_no_select_is_an_error(self):
        row = next(r for r in T.arms() if r['c1'] == 'am')

        class Skips(Client):
            def __init__(self):
                super().__init__()

        with box._bound(an, box.arm_names('am', row)), mock.patch.object(am, '_Selecting', lambda client, attn, record: client):
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
        row = T.by_label()['mg.g2-live-0']
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / 'arm.json')

            def inner(args):
                Path(args.out).write_text(json.dumps(self.record()))
                return 2
            args = mock.Mock(spec=['c1', 'size', 'cohort', 'index', 'out'], c1='mg', size=2, cohort='live', index=0, out=out)
            with mock.patch.object(an, 'run_arm', inner), mock.patch.object(am, 'end_read', lambda a: dict(read=None)), mock.patch.object(box.argparse, 'Namespace', lambda **k: mock.Mock(out=k['out'])), \
                    mock.patch.object(box, 'vars', lambda _a: dict(out=out + '.inner.json'), create=True):
                self.assertEqual(box.run_arm(args), 0)
            record = json.loads(Path(out).read_text())
            self.assertIs(record['complete'], True)
            self.assertEqual(record['levers'], dict(attn=row['attn'], lazy_recv=True, decode_merge='chunk'))


def moved_of(row):
    return dict(QUIET, lazy_marked=4 if row['lazy_recv'] else 0, **({k: MOVED[k] for k in ('merge_steps', 'merge_on', 'merge_off')} if row['decode_merge'] != 'off' else {}))


def synthetic():
    emit, arms = tq.synthetic_run(rows=[dict(r, c1='defer') for r in T.arms()])
    declared = T.by_label()
    for a in arms:
        row = declared[a['label']]
        lazy, merge = row['lazy_recv'], row['decode_merge']
        a['step'] = dict(requested_lazy=lazy, requested_merge=merge, replied=True, acknowledged_lazy=lazy, acknowledged_merge=merge)
        a['stage_switch'].update(lazy_recv=lazy, decode_merge=merge)
        a['stage_delta'].update({'step_' + k: v for k, v in moved_of(row).items()})
        a['progress'] = [dict(label=point, native_stage_local=dict(STEP_VIEW, step_lazy=lazy, step_merge=merge)) for point in ('before-fresh', 'after-fresh')]
        a['attn_meta'] = tam.block(row['attn'], [40] * 4 if row['attn'] else [0] * 4)
    return emit, arms


_RUN = {}


def run_of():
    if 'run' not in _RUN:
        _RUN['run'] = synthetic()
    emit, arms = _RUN['run']
    return emit, copy.deepcopy(arms)


def decode_records(rates=(400.0, 440.0)):
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
        self.assertEqual(result['schema'], 'compose-2-grade/1')
        self.assertIs(result['numerics_covered'], False)
        self.assertEqual(sorted(result['step_gate']['arms']), sorted(ORDER))
        self.assertEqual(sorted(result['attn_meta']), sorted(ORDER + ['decode-am0', 'decode-am1']))
        self.assertIsNone(result['step_read']['error'])
        self.assertEqual(len(result['step_read']['pairs']), 8)     # am x2, mg x2, amg live x2, amg solo, amg band 2
        read = result['compose_read']
        self.assertEqual(read['am1_over_am0'], round(440.0 / 400.0, 4))
        self.assertEqual(sorted(read['arms']), sorted(ORDER))
        cell = read['arms']['amg.g2-live-0']
        self.assertIsNotNone(cell['prefill_tok_s'])
        self.assertIsNotNone(cell['decode_ratio_own_window'])
        self.assertEqual(cell['merge']['merge_steps'], 4)
        self.assertEqual(cell['attn_batched'], [40] * 4)
        during = result['table']['amg.g2-live-0']['decode_during_tok_s']
        self.assertEqual(cell['decode_vs_fixed_reference'], round(during / 440.0, 4))
        self.assertIsNotNone(read['stack']['amg_on_b0'])
        self.assertIsNotNone(read['stack']['product_of_the_two'])
        self.assertIn('amg band 1 live', read['cells'])
        text = box.report(result)
        self.assertTrue(text.startswith('# q18a-compose-2'), text[:80])
        self.assertIn('## compose-2 reads', text)
        self.assertIn('Not covered: numerics', text)

    def test_attn_leak_and_not_engaged(self):
        result = self.edited('b0.g2-live-1', lambda a: a['attn_meta'].update(tam.block(False, [7, 0, 0, 0])))
        self.assertEqual((result['verdict'], result['classification']), ('FAIL', ['ENGINE']))
        self.assertIn('b0.g2-live-1:attn-meta-selection-leak', result['failures'])
        result = self.edited('amg.g2-live-1', lambda a: a['attn_meta'].update(tam.block(True, [0, 40, 40, 40])))
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('amg.g2-live-1:attn-meta-not-engaged', result['missing'])

    def test_step_gate_kept(self):
        result = self.edited('am.g2-live-0', lambda a: a['stage_delta'].update(step_merge_steps=1))
        self.assertIn('am.g2-live-0:step-unselected-lever-moved:merge_steps', result['failures'])
        result = self.edited('mg.g2-live-0', lambda a: a['stage_delta'].update(step_merge_steps=0))
        self.assertIn('mg.g2-live-0:step-selected-lever-never-moved:merge_steps', result['missing'])

    def test_decode_arm_missing_or_a_leak(self):
        emit, arms = run_of()
        result = self.grade(emit, decode_records()[1:] + arms)
        self.assertIn('decode-am0:no-arm-record', result['missing'])
        result = self.edited('decode-am0', lambda a: a['attn_meta'].update(tam.block(False, [1, 0, 0, 0])))
        self.assertIn('decode-am0:attn-meta-selection-leak', result['failures'])

    def test_startup_line_missing_is_inc(self):
        result = self.grade(*full_run(), startup=tam.STARTUP[:2] + tam.STARTUP[3:])
        self.assertIn('attn-meta-startup-line-missing:rank-2', result['missing'])

    def test_a_kept_append_reason_stays_inc(self):
        result = self.edited('mg.g2-live-1', lambda a: a.update(complete=False, reasons=[box.APPEND_REASON]))
        self.assertNotEqual(result['verdict'], 'PASS')


if __name__ == '__main__':
    unittest.main()
