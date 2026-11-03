#!/usr/bin/env python3
"""q18a-stage1-step box: the declared table, the arm's stage command, the status view, the gate, the reads. CPU only; no pass here is box evidence. New logic only
(the operator's discovery mode, 3 Nov 2026): no mutant classes, no fixture-leaf declarations.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import test_alpha_next_c1_queue as tq   # first: puts tools/ and the engine sources on the path
import alpha_next_arms as A
import e3_alpha_next_c1_box as c1box
import e3_stage1_step_box as box
import stage1_step_arms as T
import stage1_step_read as fread

an = c1box.an
tap = tq.tap
ROOT = Path(__file__).resolve().parents[1]
ROUND0 = ['soff.g2-solo-0', 'sl1.g2-solo-0', 'sl12.g2-solo-0', 'sl12.g2-live-0', 'sl1.g2-live-0', 'soff.g2-live-0']
ROUND1 = ['soff.g2-live-1', 'sl1.g2-live-1', 'sl12.g2-live-1', 'sl12.g2-solo-1', 'soff.g2-solo-1']
BAND2 = ['soff.g2-solo-2', 'sl12.g2-solo-2', 'sl12.g2-live-2', 'sl1.g2-live-2', 'soff.g2-live-2']
ORDER = ROUND0 + ROUND1 + BAND2 + ['soff.g2-live-3', 'sl12a.g2-live-3']
QUIET = dict(lazy_marked=0, merge_steps=0, merge_on=0, merge_off=0)
MOVED = dict(lazy_marked=4, merge_steps=4, merge_on=1, merge_off=1)
GOOD = dict(defer_bind_installed=True, defer_bind_selected=True, step_lazy_installed=True, step_lazy=False, step_merge_installed=True, step_merge='off', **{'step_' + k: 0 for k in QUIET})


def moved_of(row):
    return dict(QUIET, lazy_marked=4 if row['lazy_recv'] else 0, **({k: MOVED[k] for k in ('merge_steps', 'merge_on', 'merge_off')} if row['decode_merge'] != 'off' else {}))


class Table(unittest.TestCase):
    def test_the_arms_in_order_and_no_tail(self):
        self.assertEqual(T.labels(), ORDER)
        self.assertEqual(T.labels('tail'), [])
        self.assertEqual(T.COLD_ARM, ORDER[0])
        T.check()

    def test_every_arm_is_c1_defer_with_group_2_and_its_selection(self):
        want = dict(soff=(False, 'off'), sl1=(True, 'off'), sl12=(True, 'chunk'), sl12a=(True, 'always'))
        for r in T.arms():
            self.assertEqual((r['overlap'], r['defer_bind'], r['size'], r['stage_local']), (True, True, 2, True), r['label'])
            self.assertEqual((r['lazy_recv'], r['decode_merge']), want[r['label'].split('.')[0]], r['label'])
            self.assertEqual(r['band'], 2 if r['index'] == 2 else 1, r['label'])

    def test_every_selected_arm_is_matched_in_its_band_cohort_and_round(self):
        rows = T.by_label()
        for name, row in rows.items():
            partner = T.matched_off(row)
            if row['c1'] != 'soff':
                self.assertEqual(partner, 'soff.' + name.split('.', 1)[1])
                self.assertIn(partner, rows)
            else:
                self.assertIsNone(partner)

    def test_standing_constants_are_alpha_nexts(self):
        self.assertEqual((T.STANDING_DECODERS, T.STANDING_GREEDY, T.STANDING_SAMPLED), (A.STANDING_DECODERS, A.STANDING_GREEDY, A.STANDING_SAMPLED))

    def test_the_view(self):
        view = box.table_view()
        self.assertEqual([r['label'] for r in view.arms()], ORDER)
        self.assertEqual({r['c0'] for r in view.arms() if r['c1'] != 'soff'}, {'on'})
        self.assertEqual({r['c0'] for r in view.arms() if r['c1'] == 'soff'}, {'off'})
        self.assertEqual(view.label('on', 2, 'live', 3), 'sl12a.g2-live-3')
        self.assertEqual(view.label('on', 2, 'live', 1), 'sl12.g2-live-1')
        self.assertEqual(view.label('sl1', 2, 'live', 1), 'sl1.g2-live-1')
        self.assertEqual(box.table_view('sl1').label('on', 2, 'live', 1), 'sl1.g2-live-1')


class Client(tq.FakeClient):
    """c1's fake with the step echo; ``reply_lazy``: a value to acknowledge instead, or MISSING."""

    def __init__(self, reply_lazy=None, **kw):
        super().__init__(**kw)
        self.reply_lazy = reply_lazy

    def call(self, action, **fields):
        reply = super().call(action, **fields)
        if action == 'native_stage_local' and self.reply_lazy is not self.MISSING:
            reply.update(lazy_recv=fields.get('lazy_recv') if self.reply_lazy is None else self.reply_lazy, decode_merge=fields.get('decode_merge'))
        return reply


class ArmControl(unittest.TestCase):
    def release(self, sel, client):
        with box._bound(an, box.arm_names(sel)):
            record = {}
            try:
                an.release_alpha_next(2, True)(client, record, 'burst')
                return record, None
            except ValueError as exc:
                return record, str(exc)

    def test_every_selection_sends_its_fields(self):
        for sel, lazy, merge in (('soff', False, 'off'), ('sl1', True, 'off'), ('sl12', True, 'chunk'), ('sl12a', True, 'always')):
            client = Client()
            record, error = self.release(sel, client)
            self.assertIsNone(error)
            self.assertEqual(dict(client.calls)['native_stage_local'], dict(local=True, overlap=True, defer_bind=True, lazy_recv=lazy, decode_merge=merge))
            self.assertEqual(record['step'], dict(requested_lazy=lazy, requested_merge=merge, replied=True, acknowledged_lazy=lazy, acknowledged_merge=merge))

    def test_the_actions_are_the_list_the_router_smoke_holds(self):
        client = Client()
        self.release('sl12', client)
        self.assertEqual([a for a, _ in client.calls], [a for a, _ in box.ACTIONS if a != 'status'])

    def test_another_acknowledgement_is_recorded_and_left_to_the_grade(self):
        record, error = self.release('sl1', Client(reply_lazy=False))
        self.assertIsNone(error)
        self.assertIs(record['step']['acknowledged_lazy'], False)

    def test_a_reply_without_overlap_is_refused_and_still_resumes(self):
        client = Client(overlap_reply=False)
        record, error = self.release('soff', client)
        self.assertEqual(error, 'native-stage-overlap-selection-unproved')
        self.assertEqual(client.calls[-1][0], 'resume')
        self.assertIs(record['step']['replied'], True)


def status(lazy=False, merge='off', counters=None, block=True):
    out = tq.status(defer=True)
    if block:
        out['native_stage_local']['overlap']['decode_step'] = dict(lazy_recv_installed=True, lazy_recv=lazy, decode_merge_installed=True, decode_merge=merge, counters=dict(QUIET, **(counters or {})))
    return out


class Status(unittest.TestCase):
    def test_the_block_flat_and_its_delta(self):
        view = box.stage_view(status(True, 'chunk', dict(merge_steps=3)))
        self.assertEqual((view['step_lazy_installed'], view['step_lazy'], view['step_merge_installed'], view['step_merge'], view['step_merge_steps']), (True, True, True, 'chunk', 3))
        delta = box.stage_delta(box.stage_view(status(True, 'chunk', MOVED)), box.stage_view(status(True, 'chunk')))
        self.assertEqual({k: delta['step_' + k] for k in MOVED}, MOVED)

    def test_a_status_without_the_block_reads_none(self):
        view = box.stage_view(status(block=False))
        self.assertEqual((view['step_lazy_installed'], view['step_lazy']), (None, None))


def synthetic():
    """c1's synthetic run over this table: every arm observed C1 defer; each record with its step block and counters."""
    emit, arms = tq.synthetic_run(rows=[dict(r, c1='defer') for r in T.arms()])
    declared = T.by_label()
    for a in arms:
        row = declared[a['label']]
        lazy, merge = row['lazy_recv'], row['decode_merge']
        a['step'] = dict(requested_lazy=lazy, requested_merge=merge, replied=True, acknowledged_lazy=lazy, acknowledged_merge=merge)
        a['stage_switch'].update(lazy_recv=lazy, decode_merge=merge)
        a['stage_delta'].update({'step_' + k: v for k, v in moved_of(row).items()})
        view = dict(GOOD, step_lazy=lazy, step_merge=merge)
        a['progress'] = [dict(label=point, native_stage_local=dict(view)) for point in ('before-fresh', 'after-fresh')]
    return emit, arms


_RUN = {}


def run_of():
    if 'run' not in _RUN:
        _RUN['run'] = synthetic()
    emit, arms = _RUN['run']
    return emit, copy.deepcopy(arms)


class Gate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def grade(self, emit, arms):
        q = Path(tempfile.mkdtemp(dir=self.tmp.name))
        tap.write_streams(q, emit.lines)
        return box.grade(arms, {a['label']: tap.PASS for a in arms}, tap.CHECKS, [str(q / 'engine-stream.log'), str(q / 'engine-stream.err')], {a['label']: tap.ALIVE for a in arms})

    def edited(self, name, **fields):
        emit, arms = run_of()
        next(a for a in arms if a['label'] == name).update(fields)
        return self.grade(emit, arms)

    def test_the_record_passes_with_every_section(self):
        emit, arms = run_of()
        result = self.grade(emit, arms)
        self.assertEqual(result['verdict'], 'PASS', (result['failures'], result['missing']))
        self.assertEqual(result['schema'], 'stage1-step-grade/1')
        self.assertIs(result['numerics_covered'], False)
        self.assertEqual(sorted(result['step_gate']['arms']), sorted(ORDER))
        read = result['step_read']
        self.assertIsNone(read['error'])
        self.assertEqual(read['is_gate'], False)
        self.assertEqual(len(read['pairs']), 11)              # 6 sl12, 4 sl1, 1 sl12a: every read below is per arm
        for name in ORDER:
            if '-live-' in name:                              # a solo arm cancels its decoders: no decode to read
                self.assertIsNotNone(result['arm_numbers']['arms'][name].get('decode_ratio'), name)
            self.assertGreater(read['arms'][name]['fifo']['1']['chunks'], 0, name)
            self.assertIsNotNone(read['arms'][name]['fifo']['1']['median_s'], name)
        text = box.report(result)
        self.assertIn('Not covered: numerics', text)
        self.assertIn('OUTSIDE on merged arms is expected', text)

    def test_selection_faults_fail(self):
        good = dict(requested_lazy=True, requested_merge='off', replied=True, acknowledged_lazy=True, acknowledged_merge='off')
        result = self.edited('sl1.g2-live-0', step=dict(good, acknowledged_lazy=False))
        self.assertEqual((result['verdict'], result['classification']), ('FAIL', ['ENGINE']))
        self.assertIn('sl1.g2-live-0:lazy-recv-acknowledged-differs-from-requested', result['failures'])
        result = self.edited('sl1.g2-live-1', step=dict(good, acknowledged_merge='chunk'))
        self.assertIn('sl1.g2-live-1:decode-merge-acknowledged-differs-from-requested', result['failures'])
        result = self.edited('sl12.g2-live-0', step=dict(good, requested_merge='off'))
        self.assertEqual(result['classification'], ['HARNESS'])
        result = self.edited('soff.g2-solo-2', progress=[dict(label='before-fresh', native_stage_local=GOOD), dict(label='after-fresh', native_stage_local=dict(GOOD, step_lazy=True))])
        self.assertIn('soff.g2-solo-2:lazy-recv-status-differs-from-requested-at-after-fresh', result['failures'])

    def test_a_record_that_cannot_say_fails_harness(self):
        emit, arms = run_of()
        del next(a for a in arms if a['label'] == 'sl1.g2-solo-0')['step']
        result = self.grade(emit, arms)
        self.assertEqual((result['verdict'], result['classification']), ('FAIL', ['HARNESS']))

    def test_counters_both_ways(self):
        for name, key, reason in (('soff.g2-live-1', 'lazy_marked', 'lazy_marked'), ('soff.g2-live-1', 'merge_steps', 'merge_steps'), ('sl1.g2-live-1', 'merge_steps', 'merge_steps')):
            emit, arms = run_of()
            next(a for a in arms if a['label'] == name)['stage_delta']['step_' + key] = 1
            result = self.grade(emit, arms)
            self.assertIn('%s:step-unselected-lever-moved:%s' % (name, reason), result['failures'])
        for name, key in (('sl1.g2-solo-1' if 'sl1.g2-solo-1' in ORDER else 'sl1.g2-solo-0', 'lazy_marked'), ('sl12.g2-live-1', 'merge_steps'), ('sl12a.g2-live-3', 'merge_steps')):
            emit, arms = run_of()
            next(a for a in arms if a['label'] == name)['stage_delta']['step_' + key] = 0
            result = self.grade(emit, arms)
            self.assertEqual(result['verdict'], 'INC', result['failures'])
            self.assertIn('%s:step-selected-lever-never-moved:%s' % (name, key), result['missing'])

    def test_a_death_and_two_in_flight_stay_failures(self):
        emit, arms = run_of()
        with tempfile.TemporaryDirectory() as tmp:
            q = Path(tmp)
            tap.write_streams(q, emit.lines)
            watches = {a['label']: tap.ALIVE for a in arms}
            watches['sl12.g2-live-2'] = dict(tap.ALIVE, verdict='DEAD', exit_code=1)
            result = box.grade(arms, {a['label']: tap.PASS for a in arms}, tap.CHECKS, [str(q / 'engine-stream.log'), str(q / 'engine-stream.err')], watches)
        self.assertEqual(result['verdict'], 'FAIL', result['missing'])
        rows = [dict(r, c1='defer') for r in T.arms()]
        name = 'sl12.g2-live-0'
        before = tq.synthetic_run(rows=rows[:ORDER.index(name)])[0].txn
        emit, arms = tq.synthetic_run(rows=rows, arm_kwargs={name: dict(breach=(before + 2, 0, 'segment', 3))})
        _, good = run_of()
        for a, g in zip(arms, good):
            a.update({k: g[k] for k in ('step', 'progress')})
            a['stage_switch'].update(g['stage_switch'])
            a['stage_delta'].update(g['stage_delta'])
        v = self.grade(emit, arms)['validity'][name]
        self.assertEqual((v['verdict'], v['classification']), ('FAIL', 'ENGINE'))
        self.assertIn('two-commands-in-flight-on-stage-0', v['signatures'])


MS = 1_000_000


class Reads(unittest.TestCase):
    def test_the_fifo_wait_is_the_slower_ranks_per_chunk(self):
        bytxn = {1: [dict(seq=1, stage=1, kind='segment', segment='layer', t_sent=0), dict(seq=2, stage=1, kind='segment', segment='suffix', t_sent=10 * MS),
                     dict(seq=3, stage=0, kind='segment', segment='layer', t_sent=0)]}
        workers = {(1, 2): dict(t_enter=5 * MS), (1, 3): dict(t_enter=8 * MS), (2, 2): dict(t_enter=15 * MS), (2, 3): dict(t_enter=11 * MS), (3, 0): dict(t_enter=MS)}
        got = fread.fifo_arm([1], bytxn, workers, 1)
        self.assertEqual((got['chunks'], got['median_s'], got['is_gate']), (1, 0.01, False))     # rank 2: 5 + 5 ms; rank 3: 8 + 1 ms
        self.assertEqual(got['by_kind_median_s'], {'layer': 0.005, 'suffix': 0.005})
        self.assertEqual(fread.fifo_arm([1], bytxn, workers, 0)['median_s'], 0.001)

    def test_pending_and_polls(self):
        got = fread.pending_arm([1], {1: [dict(stage=1, outcome='pending'), dict(stage=0, outcome='ok'), dict(kind='poll')]})
        self.assertEqual((got['pending_stage1']['total'], got['pending_stage0']['total'], got['polls']['total']), (1, 0, 1))

    def test_the_decode_cohorts(self):
        line = "x flash-sched stats step %d: {'D': 0, 'dec_even': %d, 'dec_odd': %d} cohorts even=%d odd=%d none=0 decode_share_3s=None"
        got = fread.sched_read([line % (200, 8, 8, 4, 4), 'flash-sched stats broken', line % (400, 808, 8, 8, 0)])
        self.assertEqual((got['dec_even'], got['dec_odd'], got['cohorts_last'], got['malformed'], got['is_gate']), (800, 0, 'even=8 odd=0 none=0', 1, False))
        self.assertIsNone(fread.sched_read([line % (200, 8, 8, 4, 4)])['dec_even'])

    def test_the_step_rows(self):
        seams = ['pp_recv', 'forward', 'prep_attn_meta']
        fields = ['kind', 't_enter', 'wall_ns', 'cpu_ns', 'dev_forward_ns', 'seams']
        rows = [['e', 1000, 8 * MS, 6 * MS, 5 * MS, [[0, 2 * MS, 2 * MS, MS, 1], [1, 4 * MS, 4 * MS, 4 * MS, 1], [2, 3 * MS, 3 * MS, 3 * MS, 1]]],
                ['e', 2000, MS, MS, None, [[0, MS, MS, MS, 1]]], ['s', 3000, 2 * MS, MS, None, []], ['s', 10 ** 10, 90 * MS, MS, None, []]]
        line = 'flash-s5 handler ' + json.dumps(dict(head=dict(fields=dict(m=fields), seams=seams), base_ns=0, rank=2, rows=dict(m=rows)))
        got = fread.step_rows([line, 'flash-s5 handler {bad'], {'a': (0, 10 ** 9)})['a']['2']
        self.assertEqual((got['full_n'], got['empty_n'], got['sample_n'], got['full_wall_ms'], got['empty_wall_ms'], got['sample_wall_ms'], got['pair_ms']), (1, 1, 1, 8.0, 1.0, 2.0, 10.0))
        self.assertEqual((got['pp_recv_block_ms'], got['prep_attn_meta_ms'], got['dev_forward_p50_ms']), (0.5, 3.0, 5.0))
        self.assertEqual(fread.slower(dict({'2': dict(pair_ms=10.0), '3': dict(pair_ms=11.0)}), 'pair_ms'), 11.0)

    def test_the_handler_read(self):
        head = dict(head=dict(fields=dict(m=['kind', 't_enter', 'wall_ns', 'cpu_ns'])), base_ns=0, rank=2, rows=dict(m=[['e', 1000, 2 * MS, MS], ['s', 2000, 4 * MS, MS]]))
        got = fread.handler_read(['flash-s5 handler ' + json.dumps(head), 'flash-s5 handler {bad'], {'a': (0, 10 * 10 ** 9)}, {'a': 2})
        self.assertEqual(got['malformed'], 1)
        self.assertEqual(got['arms']['a']['2']['execute_model']['wall_ms_median'], 2.0)
        self.assertEqual(got['arms']['a']['2']['sample_tokens']['per_chunk'], 0.5)

    def test_the_read_never_raises_into_the_grade(self):
        got = box.step_read(dict(progress={}), ['/nonexistent/stage1-step-log'], [])
        self.assertEqual((got['is_gate'], got['arms']), (False, {}))


if __name__ == '__main__':
    unittest.main()
