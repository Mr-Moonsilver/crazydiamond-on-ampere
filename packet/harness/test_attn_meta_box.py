#!/usr/bin/env python3
"""q18a-attn-meta box: the declared table, the arm's attn-meta select inside c1's release, the end read, the grade (c1's
over this table plus the attn-meta rules) and the reads. CPU only; no pass here is box evidence.

Classes:
* Table: amendment A's cells and counts, the order (interleaved, B = A reversed, P2/MP2 last), the index encoding, the
  matched pairs (P -> R, MP -> M; M -> R, MP -> P, MP2 -> P2 for attn-meta), c1's view over this table, greedy constants.
* Release: c1's real release chain (alpha-next's release_alpha_next with c1's names and this package's release_c1) over a
  fake session client and a fake worker adapter: the select goes inside the pause, after native_stage_local and before
  release_auto_arm, once; a rank not selected or with order not ok refuses (and still resumes); the end read pauses,
  reads and resumes. Mutant: a release that skips the select is caught.
* Rules: attn_check / startup_check / deltas on their own, each verdict branch.
* Grade: c1's synthetic run (test_alpha_next_c1_queue.synthetic_run, its real emitters) over this table's core arms with
  attn-meta blocks and the two decode arms: PASS; an off-arm leak FAILs ENGINE; an on-arm without batches is INC
  attn-meta-not-engaged; a missing startup line INC, a wrong one FAIL; the reads' three retentions and summary (a)/(b).
  Mutants: the leak rule and the engagement rule removed are each caught.
"""
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import test_alpha_next_c1_queue as tq   # first: puts tools/ and the engine sources on the path
import alpha_next_arms as A
import attn_meta_arms as T
import e3_alpha_next_c1_box as c1box
import e3_attn_meta_box as box

an = c1box.an


class Table(unittest.TestCase):
    def test_cells_counts_and_order(self):
        cells = [(r['cell'], r['cohort'], r['band'], r['part']) for r in T.arms()]
        self.assertEqual(cells, [('R', 'live', 1, 'core'), ('M', 'live', 1, 'core'), ('P', 'live', 1, 'core'),
                                 ('MP', 'live', 1, 'core'), ('MP', 'live', 1, 'core'), ('P', 'live', 1, 'core'),
                                 ('M', 'live', 1, 'core'), ('R', 'live', 1, 'core'), ('M', 'solo', 1, 'core'),
                                 ('R', 'solo', 1, 'core'), ('MP', 'solo', 1, 'core'), ('P', 'solo', 1, 'core'),
                                 ('MP', 'live', 2, 'core'), ('P2', 'live', 1, 'tail'), ('MP2', 'live', 1, 'tail')])
        self.assertEqual([d['label'] for d in T.DECODE_ARMS], ['decode-am0', 'decode-am1'])
        self.assertEqual(len(T.arms()) + len(T.DECODE_ARMS), 17)

    def test_index_encoding_and_pairs(self):
        rows = T.by_label()
        for r in rows.values():
            self.assertIs(r['attn'], T.CELLS[r['cell']][0])
            self.assertEqual((r['c1'], r['size']), T.CELLS[r['cell']][1:])
        self.assertEqual(T.matched_off(rows['c1defer.g1-live-3']), 'c1off.g1-live-3')          # MP -> M
        self.assertEqual(T.matched_off(rows['c1defer.g1-solo-0']), 'c1off.g1-solo-0')          # P -> R
        self.assertIsNone(T.matched_off(rows['c1defer.g1-live-2']))                            # MP band 2: none
        self.assertIsNone(T.matched_off(rows['c1defer.g2-live-0']))                            # tail: none
        self.assertEqual(T.matched_attn_off(rows['c1off.g1-live-4']), 'c1off.g1-live-1')       # M -> R
        self.assertEqual(T.matched_attn_off(rows['c1defer.g2-live-3']), 'c1defer.g2-live-0')   # MP2 -> P2
        self.assertEqual(T.band1_glob(), '[0134]')
        self.assertEqual([T.band_of_index(i) for i in T.INDEXES], [1, 1, 2, 1, 1])

    def test_standing_constants_are_alpha_nexts(self):
        self.assertEqual((T.STANDING_DECODERS, T.STANDING_GREEDY, T.GREEDY_TEMPERATURE, T.STANDING_SAMPLED),
                         (A.STANDING_DECODERS, A.STANDING_GREEDY, A.GREEDY_TEMPERATURE, A.STANDING_SAMPLED))

    def test_c1_view_reads_this_table(self):
        with c1box._bound(c1box, dict(T=T)):
            view = c1box.table_view()
            self.assertEqual([r['label'] for r in view.arms()], T.labels())
            self.assertEqual({r['c0'] for r in view.arms() if r['c1'] == 'defer'}, {'on'})
            self.assertEqual(c1box.required_labels(), T.labels('core'))
        self.assertIs(c1box.T, tq.T)                                                           # restored

    def test_malformed_tables_refuse(self):
        for patch in (dict(ORDER=T.ORDER[:-1]), dict(ORDER=T.ORDER[:-2] + T.ORDER[-1:] + T.ORDER[-2:-1] + ((
                'core', 'R', 'live', 0),)), dict(BAND_SIZE={'core': {1: 2, 2: 2}, 'tail': {1: 1, 2: 1}})):
            with self.subTest(patch=list(patch)), mock.patch.multiple(T, **patch):
                with mock.patch.object(T, 'ARM_TABLE', T._table()):
                    self.assertRaises(AssertionError, T.check)


# --- the release ----------------------------------------------------------------------------------------

def counters(selected, batched=0, off_calls=0, mode=0):
    return dict(schema='attn-meta-select/1', batch_schema='attn-meta-gdn-batch/1', selected=selected, batched=batched,
                fallback={'mode': mode, 'capture': 0}, off_calls=off_calls, selections=0)


ORDER_OK = dict(outer='Recorder', selector_outer=False, wrapped_by_other=True, ok=True)


class FakeWorkerControl:
    """The worker adapter's reply shape (native_qualification_frontend._result over the driver's census rows)."""

    def __init__(self, state=None, bad_rank=None, order_ok=True):
        self.state = state if state is not None else {r: counters(False) for r in box.RANKS}
        self.bad_rank, self.order_ok, self.commands = bad_rank, order_ok, []

    def worker(self, command):
        self.commands.append(command)
        rows = []
        for rank in box.RANKS:
            before = copy.deepcopy(self.state[rank])
            if command['worker_action'] == box.SELECT and rank != self.bad_rank:
                self.state[rank]['selected'] = command['payload']['on']
            order = dict(ORDER_OK, ok=self.order_ok)
            q = dict(worker_action=command['worker_action'], engine_supplied_verdict=False, order=order)
            if command['worker_action'] == box.SELECT:
                q.update(before=before, after=copy.deepcopy(self.state[rank]))
            else:
                q.update(counters=copy.deepcopy(self.state[rank]))
            rows.append(dict(worker_rank=rank, qualification=q))
        return dict(schema=1, action=box.WORKER_ACTION, run_id=command['run_id'], command_seq=command['command_seq'],
                    client_id=command['client_id'], engine_epoch=command['engine_epoch'],
                    worker_action=command['worker_action'], worker_reply=dict(qualification=rows))


class Session(tq.FakeClient):
    def __init__(self, control, **kw):
        super().__init__(**kw)
        self.control, self.run_id, self.epoch = control, 'run', 7

    def call(self, action, **fields):
        return super().call(action, **fields)


class Release(unittest.TestCase):
    def release(self, attn, c1='defer', control=None, size=1):
        control = control or FakeWorkerControl()
        client = Session(control)
        client.calls_ref = client.calls
        record = {}
        original = control.worker

        def worker(command):
            client.calls.append(('worker:' + command['worker_action'], command['payload']))
            return original(command)
        control.worker = worker
        try:
            with c1box._bound(c1box, box._bindings(attn)), c1box._bound(an, c1box.arm_names(c1)):
                an.release_alpha_next(size, True)(client, record, 'burst')
            error = None
        except ValueError as exc:
            error = str(exc)
        return client, record, error, control

    def test_select_inside_the_pause_before_release(self):
        for attn in (True, False):
            client, record, error, control = self.release(attn)
            self.assertIsNone(error)
            self.assertEqual([a for a, _ in client.calls],
                             ['pause', 'native_progress', 'native_qsa_fused', 'native_layer_group', 'native_stage_local',
                              'worker:attn_meta_select', 'release_auto_arm', 'resume'])
            self.assertEqual(dict(client.calls)['worker:attn_meta_select'], {'on': attn})
            self.assertEqual({v['after']['selected'] for v in record['attn_meta']['select'].values()}, {attn})
            self.assertEqual(sorted(record['attn_meta']['select']), ['0', '1', '2', '3'])
            command = control.commands[0]
            self.assertEqual(set(command), {'schema', 'action', 'run_id', 'command_seq', 'client_id', 'engine_epoch',
                                            'worker_action', 'payload'})
            self.assertEqual((command['action'], command['engine_epoch']), ('qualification_worker', 7))
        self.assertIs(c1box.release_c1, box.C1_RELEASE)                                        # restored

    def test_a_rank_not_selected_or_order_not_ok_refuses_and_resumes(self):
        for control, text in ((FakeWorkerControl(bad_rank=2), 'attn-meta-selection-unproved'),
                              (FakeWorkerControl(order_ok=False), 'attn-meta-order-not-ok')):
            client, record, error, _ = self.release(True, control=control)
            self.assertEqual(error, text)
            self.assertEqual(client.calls[-1][0], 'resume')
            self.assertNotIn('release_auto_arm', [a for a, _ in client.calls])

    def test_reply_shape_refusals(self):
        control = FakeWorkerControl()
        w = box.Worker(control, 'run', 7)
        good = control.worker(dict(schema=1, action=box.WORKER_ACTION, run_id='run', command_seq=1, client_id='c',
                                   engine_epoch=7, worker_action=box.SELECT, payload={'on': True}))
        self.assertEqual(sorted(box.rank_rows(good, box.SELECT, 7)), [0, 1, 2, 3])
        for edit, text in ((lambda r: r.update(engine_epoch=8), 'epoch'),
                           (lambda r: r['worker_reply']['qualification'].pop(), 'ranks'),
                           (lambda r: r['worker_reply']['qualification'][0]['qualification'].update(engine_supplied_verdict=True), 'row'),
                           (lambda r: r.update(worker_action=box.READ), 'identity')):
            bad = copy.deepcopy(good)
            edit(bad)
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, text):
                box.rank_rows(bad, box.SELECT, 7)
        self.assertEqual(w.seq, 0)

    def test_mutant_release_without_the_select_is_caught(self):
        class Skip(box._Selecting):
            def call(self, action, **fields):
                return self._client.call(action, **fields)
        with mock.patch.object(box, '_Selecting', Skip):
            client, record, error, _ = self.release(True)
        self.assertEqual(error, 'attn-meta-select-not-sent')

    def test_end_read_pauses_reads_resumes(self):
        control = FakeWorkerControl({r: counters(True, batched=5) for r in box.RANKS})
        calls = []

        class Client:
            def __init__(self, url, run_id, epoch=None, timeout_s=None):
                self.control, self.epoch = control, epoch

            def pause(self):
                calls.append('pause')

            def resume(self):
                calls.append('resume')
                return {'resumed': True}
        a = types.SimpleNamespace(url='u', run_id='run', epoch=7, control_timeout_s=5)
        with mock.patch.object(box, 'NativeRuntimeClient', Client):
            out = box.end_read(a)
        self.assertEqual(calls, ['pause', 'resume'])
        self.assertEqual((out['paused'], out['resumed']), (True, True))
        self.assertEqual({v['counters']['batched'] for v in out['read'].values()}, {5})
        self.assertEqual(control.commands[0]['payload'], {})


# --- the rules ------------------------------------------------------------------------------------------

def block(attn, batched_end, start=0, order_ok=True, selected_end=None, requested=None):
    sel = {str(r): dict(before=counters(not attn), after=counters(attn, batched=start), order=dict(ORDER_OK, ok=order_ok))
           for r in box.RANKS}
    end = {str(r): dict(counters=counters(attn if selected_end is None else selected_end, batched=b, off_calls=3, mode=1),
                        order=dict(ORDER_OK, ok=order_ok))
           for r, b in zip(box.RANKS, batched_end)}
    out = dict(requested=attn if requested is None else requested, select=sel, end=dict(read=end, paused=True, resumed=True))
    out['deltas'] = box.deltas(sel, end)
    return out


class Rules(unittest.TestCase):
    def check(self, attn, **kw):
        return box.attn_check(dict(label='x', attn_meta=block(attn, **kw)), attn)

    def test_on_engaged_off_clean(self):
        self.assertEqual(self.check(True, batched_end=[3, 3, 4, 4])[:2], ([], []))
        self.assertEqual(self.check(False, batched_end=[0, 0, 0, 0])[:2], ([], []))
        f, m, facts = self.check(True, batched_end=[3, 3, 4, 4], start=1)
        self.assertEqual(facts['batched'], [2, 2, 3, 3])
        self.assertEqual(facts['fallback']['0'], {'capture': 0, 'mode': 1})

    def test_each_branch(self):
        self.assertEqual(self.check(True, batched_end=[3, 0, 4, 4])[1], ['x:attn-meta-not-engaged'])
        self.assertEqual(self.check(False, batched_end=[0, 0, 1, 0])[0], ['x:attn-meta-selection-leak'])
        self.assertIn('x:attn-meta-order', self.check(True, batched_end=[1] * 4, order_ok=False)[0])
        self.assertIn('x:attn-meta-selection-changed', self.check(True, batched_end=[1] * 4, selected_end=False)[0])
        self.assertIn('x:attn-meta-requested-differs-from-table', self.check(True, batched_end=[1] * 4, requested=False)[0])
        self.assertEqual(box.attn_check(dict(label='x', attn_meta=dict(requested=True)), True)[1], ['x:attn-meta-counters-missing'])
        b = block(True, [1] * 4)
        b['end']['read'].pop('3')
        b['deltas'] = box.deltas(b['select'], b['end']['read'])
        self.assertEqual(box.attn_check(dict(label='x', attn_meta=b), True)[1], ['x:attn-meta-counters-missing'])

    def startup(self, lines):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'log'
            p.write_text(''.join('INFO x flash-attn-meta installed schema=attn-meta-select/1 rank=%s selector_outer=%d '
                                 'wrapped_by_other=%d selected=%d\n' % row for row in lines))
            return box.startup_check([str(p)])

    def test_startup_lines(self):
        good = [(str(r), 1, 0, 0) for r in box.RANKS]
        self.assertEqual(self.startup(good)[('failures')], [])
        self.assertEqual(self.startup(good)['missing'], [])
        self.assertEqual(self.startup(good[:3])['missing'], ['attn-meta-startup-line-missing:rank-3'])
        self.assertEqual(self.startup(good[:3] + [('3', 0, 1, 0)])['failures'], ['attn-meta-startup-order:rank-3'])
        self.assertEqual(self.startup(good + [('None', 1, 0, 0)])['missing'], ['attn-meta-startup-line-rank-unknown:None'])


# --- the grade ------------------------------------------------------------------------------------------

STARTUP = ['flash-attn-meta installed schema=attn-meta-select/1 rank=%d selector_outer=1 wrapped_by_other=0 selected=0' % r
           for r in box.RANKS]
_RUN = {}


def run():
    if 'core' not in _RUN:
        _RUN['core'] = tq.synthetic_run(rows=T.arms(T.GRADED))
    emit, arms = _RUN['core']
    return emit, copy.deepcopy(arms)


def decode_record(d, rate):
    return dict(schema=box.ARM_SCHEMA, kind='decode-only', label=d['label'], complete=True, reasons=[],
                decode_only=dict(aggregate_tokens_per_s=rate, stream_rates_tokens_per_s=[rate / 8] * 8, gap_ms_p99=50.0),
                attn_meta=block(d['attn'], [7] * 4 if d['attn'] else [0] * 4))


class GradeBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def grade(self, edit=None, startup=STARTUP, decode=(400.0, 410.0)):
        emit, arms = run()
        rows = T.by_label()
        for a in arms:
            r = rows[a['label']]
            if r['size'] == 1:      # G1 is the ungrouped Step A stream: Step B's rule wants every group counter still
                a['group_delta'] = dict.fromkeys(a['group_delta'], 0)
            a['attn_meta'] = block(r['attn'], [9] * 4 if r['attn'] else [0] * 4)
        arms += [decode_record(d, rate) for d, rate in zip(T.DECODE_ARMS, decode)]
        if edit:
            edit({a['label']: a for a in arms})
        q = Path(tempfile.mkdtemp(dir=self.tmp.name))
        head = [['log', '2026-11-01T11:50:00.000000000Z (Worker_TP%d pid=%d) INFO 11-01 11:50:00 [attn_meta_select.py:155] '
                 '%s\n' % (i % 2, 201 + i, t)] for i, t in enumerate(startup)]
        tq.tap.write_streams(q, head + list(emit.lines))
        ingress = {a['label']: tq.tap.PASS for a in arms}
        watches = {a['label']: tq.tap.ALIVE for a in arms}
        logs = [str(q / 'engine-stream.log'), str(q / 'engine-stream.err')]
        return box.grade(arms, ingress, tq.tap.CHECKS, logs, watches, part='core')


class Grade(GradeBase):
    def test_core_passes_with_the_reads(self):
        result = self.grade()
        self.assertEqual((result['verdict'], result['failures'], result['missing']), ('PASS', [], []), result.get('failures'))
        self.assertEqual(result['schema'], box.GRADE_SCHEMA)
        self.assertEqual(sorted(result['attn_meta']), sorted(T.labels('core') + ['decode-am0', 'decode-am1']))
        reads = result['attn_meta_reads']
        r = reads['arms']['c1off.g1-live-0']
        self.assertEqual(r['cell'], 'R')
        self.assertTrue(r['same_arm'])
        self.assertEqual(r['retention_vs_standalone_same_attn'], r['retention_vs_frozen_reference'])
        m = reads['arms']['c1off.g1-live-3']
        self.assertFalse(m['same_arm'])
        self.assertEqual(m['retention_vs_frozen_reference'], round(m['decode_during_tok_s'] / 400.0, 4))
        self.assertEqual(m['retention_vs_standalone_same_attn'], round(m['decode_during_tok_s'] / 410.0, 4))
        self.assertEqual(reads['decode_only']['m_over_r'], round(410.0 / 400.0, 4))
        self.assertEqual(sorted(reads['cells']), ['M', 'MP', 'P', 'R'])
        self.assertIn('cell', reads['summary']['b_best_prefill_overall'])
        self.assertIs(reads['is_gate'], False)
        text = box.report(result)
        self.assertIn('(same arm)', text)

    def test_leak_fails_engine_and_not_engaged_is_inc(self):
        def leak(arms):
            arms['c1off.g1-live-1']['attn_meta'] = block(False, [0, 0, 2, 0])
        result = self.grade(leak)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('c1off.g1-live-1:attn-meta-selection-leak', result['failures'])
        self.assertEqual(result['classification'], ['ENGINE'])

        def idle(arms):
            arms['c1defer.g1-live-4']['attn_meta'] = block(True, [0] * 4)
        result = self.grade(idle)
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('c1defer.g1-live-4:attn-meta-not-engaged', result['missing'])

    def test_decode_arm_rules(self):
        def idle(arms):
            arms['decode-am1']['attn_meta'] = block(True, [0] * 4)
        self.assertIn('decode-am1:attn-meta-not-engaged', self.grade(idle)['missing'])

        def gone(arms):
            arms['decode-am0']['complete'] = False
        self.assertIn('decode-am0:decode-arm-incomplete', self.grade(gone)['missing'])

    def test_startup_lines_gate(self):
        self.assertIn('attn-meta-startup-line-missing:rank-2', self.grade(startup=STARTUP[:2] + STARTUP[3:])['missing'])
        bad = STARTUP[:3] + [STARTUP[3].replace('selector_outer=1 wrapped_by_other=0', 'selector_outer=0 wrapped_by_other=1')]
        self.assertIn('attn-meta-startup-order:rank-3', self.grade(startup=bad)['failures'])

    def test_mutants_caught(self):
        def leak(arms):
            arms['c1off.g1-live-1']['attn_meta'] = block(False, [0, 0, 2, 0])

        def idle(arms):
            arms['c1defer.g1-live-4']['attn_meta'] = block(True, [0] * 4)
        real = box.attn_check

        def no_leak_rule(record, declared):
            f, m, facts = real(record, declared)
            return [x for x in f if not x.endswith('selection-leak')], m, facts

        def no_engagement_rule(record, declared):
            f, m, facts = real(record, declared)
            return f, [x for x in m if not x.endswith('not-engaged')], facts
        for mutant, edit, verdict in ((no_leak_rule, leak, 'FAIL'), (no_engagement_rule, idle, 'INC')):
            with self.subTest(mutant=mutant.__name__):
                with mock.patch.object(box, 'attn_check', mutant):
                    caught = self.grade(edit)['verdict'] != verdict
                self.assertTrue(caught, 'the mutant must change the verdict (the real rule is what holds it)')


if __name__ == '__main__':
    unittest.main()
