#!/usr/bin/env python3
"""Cross-arm quiet-SPLIT table (tools/q2024p_j2_quiet_split_reader.py, brief 13 item 1). CPU only, hub only.

Synthetic records (engine-log lines ``<stamp> flash-s5 decode-timeline {json}`` in fresh mkdtemp roots, left in place):
one n3 arm, one sp arm and a control after them, four ranks. Covers every class (Q_full, Q_split per stage, C_split,
B_split), each refusal, the hard stop (a native session on a control generation), the TP stage value (max, never the
sum), the consistency check (0.5 ms on 3 of 4 ranks, reference pooled from --reference roots), the stamp budget and
the attn-meta state. Mutants, each caught: the handler-flags filter dropped, the arm by the stamp without the log lag,
the control tied by time instead of route generation, the stage value as a sum, the native-in-control stop dropped,
the consistency bound widened.
Brief 14 (REVIEW-qsplit3 finding 3): consistency is PASS / FAIL / NOT_TESTABLE with its reasons (NOT_TESTABLE blocks
names only on a 3-rank excess of the control); "counted" needs a completed stage without an abort (v3) or a drain
(v2); a pooled cell never mixes cg_mode / bank_sms and N needs one key; the ordering check, the prompt + 4,096 history
bound, the first-token cache check, attn-meta engagement and the warm; the cross-checks only at 120k on 1, 2, 4 live.
Mutants, each caught: NOT_TESTABLE folded into FAIL, the abort ignored in counted, the key filter dropped, the
history slack widened, the order bound dropped, the cache bound dropped.
"""
import contextlib
import datetime
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
TOOL = next(p for p in (HERE.parent / 'tools' / 'q2024p_j2_quiet_split_reader.py', HERE / 'tools' / 'q2024p_j2_quiet_split_reader.py')
            if p.is_file())
_spec = importlib.util.spec_from_file_location('q2024p_j2_quiet_split_reader', TOOL)
qs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qs)

W0 = 1794600000.0
T = 10 ** 12
META = dict(actual_rows=1, actual_tokens=1, uid_digest='u' * 64, history_tokens=[120000], bank='split', route_generation=5,
            decode_sms=24, bank_sms=24, stream=7, cg_mode='full', num_tokens=1, num_reqs=1, uniform_token_count=1,
            max_query_len=1, num_ubatches=None, forward_calls=1, handler_flags=0, completion='own-forward-end-event-query')
FIELDS = dict(d=list(qs.bf.D_FIELDS), p=['phase', 'layer', 'group', 'bucket', 'stream', 'start_ms', 'end_ms'],
              s=['complete', 'counters'])


def iso(wall):
    return datetime.datetime.fromtimestamp(wall, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + '000Z'


class Rec:
    def __init__(self):
        self.lines, self.seq, self.samples, self.e = [], {}, {}, 1000

    def row(self, rank, wall, D, *, scope='outside', rows=1, bank='split', gen=5, flags=0, calls=1, history=120000,
            device=True, nt=None, sessions_s=None, seq=None, cg=None, sms=None):
        """One session with one d row on ``rank``; returns the row's (rank, seq)."""
        if seq is None:
            self.e += 1
            seq = self.e
        meta = dict(META, actual_rows=rows, num_reqs=rows, num_tokens=nt or qs.PADDED[rows], bank=bank,
                    bank_sms=sms or (24 if bank == 'split' else 82), cg_mode=cg or ('full' if bank == 'split' else 'FULL'),
                    route_generation=gen, handler_flags=flags, forward_calls=calls,
                    history_tokens=[history] * rows if history else [])
        d = [seq, meta, 0.0, D, int(D * 1e6) if device else None, 3000000, 100000, 0, 3, int(D * 1000) + 5]
        pid = 10 + rank
        k = (pid, scope)
        n = self.samples.get(k, 0) + 1
        self.samples[k] = n
        body = dict(side='worker', rank=rank, base_ns=T + n, engine_epoch=1, stage=rank // 2, dropped=0, fields=FIELDS,
                    rows=dict(d=[d], s=[[True, sessions_s or {}]]), schema='s5-decode-timeline/1', line=0, part=0, parts=1)
        if scope == 'outside':
            body.update(scope='outside-native-progress', sample='outside-%d' % n, anchor_sample='outside-%d' % n,
                        engine_send_ns=T, execute_seq=seq)
        else:
            body.update(txn=n, sample=n, anchor_sample=n, route_generation=gen)
        rseq = self.seq.get(pid, 0)
        self.seq[pid] = rseq + 1
        ident = '{"pid":%d,"seq":%d,"sha":"s%d-%d",' % (pid, rseq, pid, n)
        self.lines.append('%s flash-s5 decode-timeline %s' % (iso(wall), ident + json.dumps(body, separators=(',', ':'))[1:]))
        return (rank, seq)

    def write(self, arms=(('n3.g2-live-0', W0, W0 + 500, True), ('sp.g2-live-1', W0 + 600, W0 + 1100, False)),
              control=None):
        root = Path(tempfile.mkdtemp(prefix='qsread-'))
        for name, start, end, attn in arms:
            sel = {str(r): dict(after=dict(selected=attn)) for r in range(4)}
            (root / ('arm-%s.json' % name)).write_text(json.dumps(dict(label=name, wall_start_s=start, wall_end_s=end,
                                                                        attn_meta=dict(requested=attn, select=sel))))
        (root / 'engine.log').write_text('\n'.join(self.lines) + '\n')
        if control is not None:
            (root / 'quiet-split.json').write_text(json.dumps(control))
        return root


def stage(live, gen, prompt=120000, **over):
    s = dict(live=live, prompt_tokens=prompt, split_generation=gen, window=dict(wall_s=25), merge_steps=900,
             completed=True, drain='disconnect', drain_s=0.4, history_tokens=[prompt + 2, prompt + 1500],
             pre_hold_tokens_max=3, streams=[dict(first_token_s=0.8) for _ in range(live)])
    s.update(over)
    return s


CONTROL = dict(schema='quiet-split-box/3', verdict='PASS', native_admitted_before=4, native_admitted_after=4,
               attn_meta=dict(selected=True, order_ok=True, batched_delta={str(r): 40 for r in range(4)}),
               warm=[dict(index=i, status=200, prompt_tokens=120000) for i in range(8)],
               stages=[stage(4, 20), stage(1, 22)])


def standard(q_full=8.0, c_split=11.4, busy=14.3, q_split=11.2, n_busy=60, n_ctl=14, rec=None, peers=None, n_c=6):
    """4 live: Q_full / C_split / B_split in the n3 arm on every rank, the control's stage 0 (4 live) after the sp arm."""
    rec = rec or Rec()
    peers = peers or {}
    for rank in range(4):
        bump = peers.get(rank, 0.0)
        for i in range(10):
            rec.row(rank, W0 + 10 + i, q_full + bump, bank='full', rows=4)
        for i in range(n_c):
            rec.row(rank, W0 + 30 + i, c_split + bump, rows=4)
        for i in range(n_busy):
            rec.row(rank, W0 + 50 + i, busy + bump, scope='native', rows=4)
        for i in range(n_ctl):
            rec.row(rank, W0 + 1200 + i, q_split + bump, rows=4, gen=20)
        for i in range(n_ctl):
            rec.row(rank, W0 + 1300 + i, 9.0 + bump, rows=1, gen=22)
    return rec


def read(root, **kw):
    return qs.read(str(root), **kw)


def live(doc, n):
    return next(r for r in doc['three'] if r['live'] == n)


class Classes(unittest.TestCase):
    def test_each_class_and_its_cell(self):
        doc = read(standard().write(control=CONTROL))
        self.assertEqual(doc['row_counts'], dict(Q_full=40, Q_split=112, C_split=24, B_split=240))
        r0 = live(doc, 4)['ranks'][0]
        self.assertEqual((r0['Q_full']['p50'], r0['C_split']['p50'], r0['B_split']['p50']), (8.0, 11.4, 14.3))
        self.assertEqual(r0['Q_split'][0]['p50'], 11.2)
        self.assertEqual(live(doc, 1)['ranks'][0]['Q_split'][1]['p50'], 9.0)
        cell = next(c for c in doc['table'] if c['cls'] == 'Q_split' and c['rank'] == 0 and c['stage'] == 0)
        self.assertEqual((cell['bank'], cell['bank_sms'], cell['cg_mode'], cell['num_tokens'], cell['actual_rows'], cell['band']),
                         ('split', 24, 'full', 4, 4, '97152-up'))
        chk = live(doc, 4)['control_stages'][0]
        self.assertEqual(chk['differences'][0], dict(S=3.2, N=3.1))
        self.assertEqual(doc['attn_meta']['control'], 'on')
        self.assertEqual(doc['attn_meta']['n3_arms']['n3.g2-live-0'], dict(requested=True, selected=['True']))

    def test_sp_arm_rows_refused_and_kept_counted(self):
        rec = standard()
        for rank in range(4):
            rec.row(rank, W0 + 700, 7.0, bank='full', rows=4)
        doc = read(rec.write(control=CONTROL))
        self.assertEqual(doc['refusals'].get('not_n3_arm:sp.g2-live-1'), 4)

    def test_tp_stage_value_is_the_peer_max(self):
        doc = read(standard(peers={1: 0.5, 3: 0.2}).write(control=CONTROL))
        sv = live(doc, 4)['stage_values']
        self.assertEqual(sv['B_split'], {0: 14.8, 1: 14.5})
        self.assertEqual(sv['Q_split'][0], {0: 11.7, 1: 11.4})

    def test_attn_meta_off_named_for_a_v1_control(self):
        v1 = dict(CONTROL, schema='quiet-split-box/1')
        v1.pop('attn_meta')
        doc = read(standard().write(control=v1))
        self.assertTrue(doc['attn_meta']['control'].startswith('off (not selected'))


class Refusals(unittest.TestCase):
    def test_each_refusal_by_its_first_failing_field(self):
        rec = standard()
        rec.row(0, W0 + 40, 9.0, flags=8, rows=4)
        rec.row(0, W0 + 40, 9.0, calls=2, rows=4)
        rec.row(0, W0 + 40, 9.0, history=0, rows=4)
        rec.row(0, W0 + 40, 9.0, device=False, rows=4)
        rec.row(0, W0 + 1, 9.0, rows=4)                       # inside the lag at the arm start: boundary
        rec.row(0, W0 + 1250, 9.0, rows=4, gen=99)            # after the arms, no control stage
        rec.row(0, W0 + 1250, 9.0, rows=4, gen=20, bank='full')
        rec.row(0, W0 + 550, 9.0, rows=4)                     # between the arms
        doc = read(rec.write(control=CONTROL))
        for why in ('handler_flags_not_0', 'forward_calls_not_1', 'no_history', 'no_device_ns', 'arm_boundary',
                    'after_arms_no_control_stage', 'control_row_not_split', 'no_arm'):
            self.assertEqual(doc['refusals'].get(why), 1, why)

    def test_conflicting_d_rows_never_resolved(self):
        rec = standard()
        rec.row(2, W0 + 60, 9.0, rows=4, seq=5000)
        rec.row(2, W0 + 61, 9.5, rows=4, seq=5000)
        doc = read(rec.write(control=CONTROL))
        self.assertEqual(doc['refusals'].get('conflicting_d_row'), 1)

    def test_native_session_on_a_control_generation_is_a_hard_stop(self):
        rec = standard()
        rec.row(1, W0 + 1210, 12.0, scope='native', rows=4, gen=20)
        doc = read(rec.write(control=CONTROL))
        self.assertEqual([h['why'] for h in doc['hard_stops']], ['native_session_in_control'])
        self.assertEqual(live(doc, 4)['ranks'][1]['Q_split'][0]['n'], 14)

    def test_without_quiet_split_json_no_q_split(self):
        doc = read(standard().write())
        self.assertEqual(doc['row_counts']['Q_split'], 0)
        self.assertEqual(doc['refusals']['after_arms_no_control_stage'], 112)
        self.assertIsNone(doc['attn_meta']['control'])


class Consistency(unittest.TestCase):
    def test_pass_within_half_a_ms_on_three_ranks(self):
        doc = read(standard(c_split=11.4, q_split=11.2).write(control=CONTROL))
        chk = live(doc, 4)['control_stages'][0]
        self.assertEqual((chk['consistency']['value'], chk['order']['value'], chk['names_allowed']), ('PASS', 'OK', True))

    def test_fail_beyond_half_a_ms(self):
        doc = read(standard(c_split=11.4, q_split=10.7).write(control=CONTROL))
        chk = live(doc, 4)['control_stages'][0]
        self.assertEqual((chk['consistency']['value'], chk['consistency']['ranks_within']), ('FAIL', 0))
        self.assertTrue(chk['consistency']['blocks_names'])
        self.assertFalse(chk['names_allowed'])

    def test_reference_rows_pooled_and_minimum_counts(self):
        thin = standard()
        root = thin.write(control=CONTROL)
        doc = read(root)
        thin1 = live(doc, 1)['control_stages'][1]['consistency']                         # no C_split at 1 live
        self.assertEqual((thin1['value'], thin1['blocks_names']), ('NOT_TESTABLE', False))
        self.assertIn('rank0:reference-n0<3', thin1['why'])
        ref = Rec()
        for rank in range(4):
            for i in range(5):
                ref.row(rank, W0 + 20 + i, 9.2, rows=1)
        doc = read(root, references=[str(ref.write())])
        chk = live(doc, 1)['control_stages'][1]
        self.assertEqual((chk['consistency']['value'], live(doc, 1)['ranks'][0]['C_split']['n']), ('PASS', 5))
        few = read(standard(n_ctl=11).write(control=CONTROL))
        self.assertIn('rank0:control-n11<12', live(few, 4)['control_stages'][0]['consistency']['why'])


class Labels(unittest.TestCase):
    """Brief 14: the three labels the review names, and the stage lines."""

    def test_not_testable_is_not_fail_and_prints_its_reasons(self):
        doc = read(standard(n_c=3).write(control=CONTROL))
        c = live(doc, 4)['control_stages'][0]['consistency']
        self.assertEqual((c['value'], c['blocks_names'], c['testable_ranks']), ('NOT_TESTABLE', False, 0))
        self.assertTrue(any(w.startswith('rank0:reference-n3<5') for w in c['why']), c['why'])
        self.assertTrue(live(doc, 4)['control_stages'][0]['names_allowed'])

    def test_not_testable_blocks_on_a_three_rank_excess(self):
        doc = read(standard(n_c=3, q_split=12.2).write(control=CONTROL))
        chk = live(doc, 4)['control_stages'][0]
        self.assertEqual((chk['consistency']['value'], chk['consistency']['blocks_names']), ('NOT_TESTABLE', True))
        self.assertEqual(chk['consistency']['ranks_control_slower'], 4)
        self.assertFalse(chk['names_allowed'])
        thin = read(standard(n_c=2, q_split=12.2).write(control=CONTROL))       # n < 3: names allowed
        self.assertFalse(live(thin, 4)['control_stages'][0]['consistency']['blocks_names'])

    def test_counted_needs_a_completed_stage(self):
        failed = dict(CONTROL, verdict='PARTIAL', stages=[
            stage(4, 20, completed=False, abort=dict(reason='decoder-stalled-in-hold', phase='hold')),
            stage(1, 22, abort=dict(reason='odd', phase='hold'))])          # completed but an abort: never counted
        doc = read(standard().write(control=failed))
        s0, s1 = doc['control']['stages']
        self.assertEqual((s0['counted'], s0['not_counted'], s0['abort']), (False, ['not-completed', 'abort'],
                                                                            'decoder-stalled-in-hold'))
        self.assertEqual((s1['counted'], s1['not_counted']), (False, ['abort']))
        self.assertIn('not counted: not-completed,abort', live(doc, 4)['control_stages'][0]['names_blocked_by'])
        nodrain = dict(CONTROL, stages=[stage(4, 20, drain=None), stage(1, 22)])
        self.assertEqual(read(standard().write(control=nodrain))['control']['stages'][0]['not_counted'], ['no-drain'])

    def test_counted_for_a_v2_record(self):
        """cxqsplit3's schema: its abort field is the drain mode; a failed stage has none."""
        v2 = dict(CONTROL, schema='quiet-split-box/2',
                  stages=[dict(stage(4, 20), abort='max_tokens'), dict(stage(1, 22), abort=None)])
        for s in v2['stages']:
            s.pop('completed')
            s.pop('drain')
        doc = read(standard().write(control=v2))
        s0, s1 = doc['control']['stages']
        self.assertEqual((s0['counted'], s0['drain'], s1['counted'], s1['not_counted']), (True, 'max_tokens', False, ['no-drain']))

    def test_cells_never_pool_two_graph_keys(self):
        rec = standard()
        for i in range(3):
            rec.row(0, W0 + 1250 + i, 10.0, rows=4, gen=20, cg='piecewise')
        doc = read(rec.write(control=CONTROL))
        q = live(doc, 4)['ranks'][0]['Q_split'][0]
        self.assertEqual((q['p50'], q['mixed']), (None, {'full/24': 14, 'piecewise/24': 3}))
        chk = live(doc, 4)['control_stages'][0]
        self.assertIn('rank0:control-mixed', chk['consistency']['why'])
        self.assertIsNone(chk['differences'][0]['N'])

    def test_n_refused_across_keys(self):
        rec = Rec()
        standard(rec=rec, n_busy=0)
        for rank in range(4):
            for i in range(60):
                rec.row(rank, W0 + 50 + i, 14.3, scope='native', rows=4, sms=32)
        doc = read(rec.write(control=CONTROL))
        d = live(doc, 4)['control_stages'][0]['differences'][0]
        self.assertIsNone(d['N'])
        self.assertEqual(d['N_refused'], 'cg_mode/bank_sms differ: Q_split full/24, B_split full/32')
        self.assertEqual(d['S'], 3.2)

    def test_order_broken_blocks_names(self):
        doc = read(standard(q_split=7.5, c_split=7.6).write(control=CONTROL))
        chk = live(doc, 4)['control_stages'][0]
        self.assertEqual((chk['order']['value'], chk['consistency']['value']), ('BROKEN', 'PASS'))
        self.assertFalse(chk['names_allowed'])
        self.assertIn('order BROKEN', chk['names_blocked_by'])

    def test_stage_lines(self):
        cases = (
            (dict(history_tokens=[120002, 124097]), 'history'), (dict(pre_hold_tokens_max=601), 'history'),
            (dict(streams=[dict(first_token_s=0.8)] * 3 + [dict(first_token_s=12.0)]), 'cache'),
            (dict(streams=[]), 'cache'))
        for over, line in cases:
            with self.subTest(over=over):
                ctl = dict(CONTROL, stages=[stage(4, 20, **over), stage(1, 22)])
                s0 = read(standard().write(control=ctl))['control']['stages'][0]
                self.assertFalse(s0['matched'])
                self.assertTrue(s0['lines'][line].startswith('out'), s0['lines'])
        edge = dict(CONTROL, stages=[stage(4, 20, history_tokens=[120000, 124096], pre_hold_tokens_max=600,
                                           streams=[dict(first_token_s=10.0)] * 4), stage(1, 22)])
        self.assertTrue(read(standard().write(control=edge))['control']['stages'][0]['matched'])
        noread = dict(CONTROL, attn_meta=dict(selected=True, order_ok=True))
        s0 = read(standard().write(control=noread))['control']['stages'][0]
        self.assertEqual(s0['lines']['attn_meta'], 'out (attn-meta selected, engagement not read)')
        cold = dict(CONTROL, warm=[dict(index=0, status=200, prompt_tokens=120001)])
        self.assertTrue(read(standard().write(control=cold))['control']['stages'][0]['lines']['warm'].startswith('out'))

    def test_cross_checks_only_at_120k_on_1_2_4(self):
        rec = standard()
        for rank in range(4):
            for i in range(14):
                rec.row(rank, W0 + 1400 + i, 15.0, rows=6, gen=24)
                rec.row(rank, W0 + 1500 + i, 10.8, rows=4, gen=26, history=72000)
        ctl = dict(CONTROL, stages=[stage(4, 20), stage(1, 22), stage(6, 24), stage(4, 26, prompt=72000)])
        doc = read(rec.write(control=ctl))
        for live_n, idx in ((6, 2), (4, 3)):
            chk = live(doc, live_n)['control_stages'][idx]
            self.assertEqual((chk['order']['value'], chk['consistency']['value']), ('NOT_APPLICABLE', 'NOT_APPLICABLE'))
            self.assertTrue(chk['names_allowed'], chk['names_blocked_by'])

    def test_text_prints_values_and_reasons(self):
        doc = read(standard(n_c=3).write(control=CONTROL))
        text = qs.text(doc)
        self.assertIn('consistency NOT_TESTABLE', text)
        self.assertIn('reference-n3<5', text)
        self.assertIn('synthetic random-id prompts', text)
        self.assertIn('order OK', text)


class Budget(unittest.TestCase):
    def test_per_rank_numbers_and_bytes_split_at_the_control(self):
        rec = standard()
        rec.row(3, W0 + 70, 9.0, rows=4, sessions_s=dict(log_budget_exhausted=1))
        doc = read(rec.write(control=CONTROL))
        b = doc['budget']['per_rank']
        self.assertEqual((b['0']['native_max'], b['0']['outside_max']), (60, 44))
        self.assertEqual(b['3']['log_budget_exhausted'], 1)
        self.assertGreater(b['0']['bytes_control'], 0)
        self.assertGreater(b['0']['bytes_before_control'], b['0']['bytes_control'])
        self.assertEqual(b['0']['lines'], 10 + 6 + 60 + 28)


class Main(unittest.TestCase):
    def test_fresh_out_and_both_files(self):
        root = standard().write(control=CONTROL)
        out = Path(tempfile.mkdtemp(prefix='qsread-out-'))
        with contextlib.redirect_stdout(io.StringIO()) as text:
            self.assertEqual(qs.main(['--root', str(root), '--out', str(out)]), 0)
        self.assertIn('QUIET-SPLIT CROSS-ARM TABLE', text.getvalue())
        self.assertTrue((out / 'quiet-split-read.json').is_file() and (out / 'quiet-split-read.txt').is_file())
        with self.assertRaises(SystemExit):
            qs.main(['--root', str(root), '--out', str(out)])


def mutant(old, new):
    text = TOOL.read_text()
    assert text.count(old) == 1, old
    spec = importlib.util.spec_from_loader('qs_mutant', loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(TOOL)
    exec(compile(text.replace(old, new), str(TOOL), 'exec'), module.__dict__)
    return module


def caught(module, test):
    """True when ``test`` (a Classes / Refusals / Consistency method) fails with the mutant bound as ``qs``."""
    global qs
    saved, qs = qs, module
    try:
        case = test.__self__.__class__(test.__name__)
        result = unittest.TestResult()
        case.run(result)
        return bool(result.failures or result.errors)
    finally:
        qs = saved


class Mutants(unittest.TestCase):
    def check(self, old, new, *tests):
        m = mutant(old, new)
        for t in tests:
            self.assertTrue(caught(m, t), (new, t.__name__))

    def test_flags_filter_dropped(self):
        self.check("    if meta.get('handler_flags') != 0:\n", "    if False:\n",
                   Refusals('test_each_refusal_by_its_first_failing_field').test_each_refusal_by_its_first_failing_field)

    def test_arm_without_the_log_lag(self):
        self.check('    lo, hi = stamp - LOG_LAG_S, stamp\n', '    lo, hi = stamp, stamp\n',
                   Refusals('test_each_refusal_by_its_first_failing_field').test_each_refusal_by_its_first_failing_field)

    def test_control_by_time_not_generation(self):
        self.check("        if gen in gens:\n            if meta['bank'] != 'split':",
                   "        if r['stamp'] is not None and r['stamp'] > last_end:\n            gen = next(iter(gens), gen)\n"
                   "            if meta['bank'] != 'split':",
                   Refusals('test_each_refusal_by_its_first_failing_field').test_each_refusal_by_its_first_failing_field)

    def test_stage_value_as_sum(self):
        self.check('        out[stage] = None if va is None or vb is None else max(va, vb)\n',
                   '        out[stage] = None if va is None or vb is None else va + vb\n',
                   Classes('test_tp_stage_value_is_the_peer_max').test_tp_stage_value_is_the_peer_max)

    def test_native_in_control_not_stopped(self):
        self.check("        if gen in gens and r['scope'] == 'native':", "        if False:",
                   Refusals('test_native_session_on_a_control_generation_is_a_hard_stop')
                   .test_native_session_on_a_control_generation_is_a_hard_stop)

    def test_not_testable_folded_into_fail(self):
        self.check("        value = 'NOT_TESTABLE'\n        blocks = excess >= 3\n", "        value = 'FAIL'\n        blocks = True\n",
                   Labels('test_not_testable_is_not_fail_and_prints_its_reasons').test_not_testable_is_not_fail_and_prints_its_reasons)

    def test_abort_ignored_in_counted(self):
        self.check("('abort', 'abort' in s)", "('abort', False)",
                   Labels('test_counted_needs_a_completed_stage').test_counted_needs_a_completed_stage)

    def test_key_filter_dropped(self):
        self.check("    if len(keys) > 1:\n", "    if False:\n",
                   Labels('test_cells_never_pool_two_graph_keys').test_cells_never_pool_two_graph_keys)

    def test_history_slack_widened(self):
        self.check('HISTORY_SLACK = 4096 ', 'HISTORY_SLACK = 1 << 30 ', Labels('test_stage_lines').test_stage_lines)

    def test_order_bound_dropped(self):
        self.check("        elif full['p50'] < q['p50'] <= busy['p50'] + ORDER_MS:", "        elif True:",
                   Labels('test_order_broken_blocks_names').test_order_broken_blocks_names)

    def test_cache_bound_dropped(self):
        self.check('FIRST_TOKEN_S = 10.0 ', 'FIRST_TOKEN_S = 1e9 ', Labels('test_stage_lines').test_stage_lines)

    def test_consistency_bound_widened(self):
        self.check('CONSIST_MS = 0.5\n', 'CONSIST_MS = 1.0\n',
                   Consistency('test_fail_beyond_half_a_ms').test_fail_beyond_half_a_ms)


if __name__ == '__main__':
    unittest.main()
