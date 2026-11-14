#!/usr/bin/env python3
"""Lever 2 pair reader (tools/q2024p_j2_lever2_reader.py). CPU only, hub only; synthetic records in fresh mkdtemp roots
(left in place). The decode-timeline rows come from test_q2024p_j2_quiet_split_reader's Rec (its receipt format), the
anatomy chunk lines and the construction inventory are written here. Covers: B_split / C_split / Q_full p50 per rank
and live count with the bank_sms key (24 off, 42 on), the stage value as the max of the peers, off less on; native
chunk rate per rank and width band with its change in percent; cancelled chunks and chunks outside an n3 arm refused;
the overlap record from the inventory JSON, else from the log line; the decision inputs at 4 live. Mutants, each
caught: the stage value as a sum, the chunk rate over busy_us instead of the device window, cancelled chunks kept.
Appends (per-step brief): burst / all split direct, stranded, +quota, no-lookup; burst end inclusive; the mechanism's
burst preferred, the turns' total else; window and window-fix counters as n3-arm deltas. Mutants: exclusive burst end,
absolute counters, stranded by the last lookup.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TOOL = HERE.parent / 'tools' / 'q2024p_j2_lever2_reader.py'
_spec = importlib.util.spec_from_file_location('q2024p_j2_lever2_reader', TOOL)
l2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(l2)
import test_q2024p_j2_quiet_split_reader as qst  # noqa: E402

W0 = qst.W0
T = 10 ** 12


def anatomy(rank, wall, rows, window_ms, busy_us, *, seq, cancelled=0, start=0):
    base = T + seq * 10 ** 9
    c = [40, 40, 0, 0, 0, int(window_ms * 1000), busy_us, 10, 0, 5, cancelled, 0, 0]
    body = dict(side='worker', rank=rank, base_ns=base, line=seq, part=0, parts=1, txn=seq, chunk_start=start,
                chunk_end=start + rows, request_generation=1, schema='s5-native-anatomy/1', rows=dict(c=[c]))
    ident = '{"pid":%d,"seq":%d,"sha":"a%d-%d",' % (50 + rank, seq, rank, seq)
    return '%s flash-s5 anatomy %s' % (qst.iso(wall), ident + json.dumps(body, separators=(',', ':'))[1:])


def packet(sms, busy4, rate_ms, *, inventory=True, extra_lines=(), q_full=8.0):
    """One n3 arm with Q_full / C_split / B_split at 1 and 4 live on four ranks, bulk and early native chunks."""
    rec = qst.Rec()
    for rank in range(4):
        for i in range(12):
            rec.row(rank, W0 + 10 + i, q_full, bank='full', rows=4)
            rec.row(rank, W0 + 30 + i, busy4 - 2.0, rows=4, sms=sms)
        for i in range(60):
            rec.row(rank, W0 + 50 + i, busy4 + (0.4 if rank == 1 else 0.0), scope='native', rows=4, sms=sms)
            rec.row(rank, W0 + 120 + i, busy4 - 3.0, scope='native', rows=1, sms=sms)
    root = rec.write()
    lines = []
    for rank in range(4):
        for i in range(20):
            lines.append(anatomy(rank, W0 + 200 + i, 2048, rate_ms, 9000, seq=i))
            lines.append(anatomy(rank, W0 + 230 + i, 4048, 2 * rate_ms, 18000, seq=100 + i))
        lines.append(anatomy(rank, W0 + 260, 2048, 1.0, 9000, seq=300, cancelled=1))     # refused
        lines.append(anatomy(rank, W0 + 700, 2048, 1.0, 9000, seq=301))                  # in the sp arm: refused
    lines.extend(extra_lines)
    with open(root / 'engine.log', 'a') as handle:
        handle.write('\n'.join(lines) + '\n')
    if inventory and sms != 24:
        rows = {str(r): dict(native_inventory=dict(
            schema='s5-native-construction/1', worker_rank=r,
            split=dict(requested_decode_sms=24, decode_sms=24, decode_overlap=dict(
                schema='s5-decode-overlap/1', own_decode_sms=24, borrowed_prefill_sms=sms - 24, superset=True,
                pair_delta_bytes=2097152)))) for r in range(4)}
        (root / 'startup-identity.json').write_text(json.dumps(dict(rows=rows)))
    return root


class Pair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.off = packet(24, 14.3, 270.0)
        cls.on = packet(42, 12.8, 290.0)
        cls.doc = l2.read(str(cls.off), str(cls.on))

    def test_busy_split_per_rank_and_stage(self):
        four = next(r for r in self.doc['compare']['forwards']['B_split'] if r['live'] == 4)
        r0 = four['ranks'][0]
        self.assertEqual((r0['off']['p50'], r0['on']['p50'], r0['off_less_on']), (14.3, 12.8, 1.5))
        self.assertEqual((r0['off']['key'], r0['on']['key']), ('full/24', 'full/42'))
        self.assertEqual(r0['off']['n'], 60)
        # stage 0 = max(rank 0, rank 1): rank 1 is 0.4 ms slower in both packets
        self.assertEqual((four['stages'][0]['off'], four['stages'][0]['on'], four['stages'][0]['off_less_on']),
                         (14.7, 13.2, 1.5))
        one = next(r for r in self.doc['compare']['forwards']['B_split'] if r['live'] == 1)
        self.assertEqual(one['ranks'][2]['off_less_on'], 1.5)
        empty = next(r for r in self.doc['compare']['forwards']['B_split'] if r['live'] == 6)
        self.assertEqual((empty['ranks'][0]['off']['n'], empty['ranks'][0]['off_less_on']), (0, None))

    def test_other_classes(self):
        four = {c: next(r for r in self.doc['compare']['forwards'][c] if r['live'] == 4) for c in l2.CLASSES}
        self.assertEqual(four['Q_full']['ranks'][3]['off_less_on'], 0.0)
        self.assertEqual(four['C_split']['ranks'][3]['off']['p50'], 12.3)
        self.assertEqual(four['C_split']['ranks'][3]['on']['key'], 'full/42')

    def test_native_rate_and_refusals(self):
        nat = self.doc['compare']['native']['0']
        self.assertEqual(nat['1025-2048']['off']['chunks'], 20)
        self.assertAlmostEqual(nat['1025-2048']['off']['rows_per_s_p50'], 2048 / 0.270, places=2)
        self.assertEqual(nat['1025-2048']['rate_change_pct'], round(100 * (270 / 290 - 1), 2))
        self.assertEqual(nat['2049-4096']['on']['chunks'], 20)
        refusals = self.doc['off']['native']['refusals']
        self.assertEqual(refusals, dict(cancelled_or_errors=4, not_inside_n3_arm=4))

    def test_overlap_record_and_decision_inputs(self):
        self.assertEqual(self.doc['off']['overlap']['source'], 'absent')
        on = self.doc['on']['overlap']
        self.assertEqual(on['source'], 'inventory')
        self.assertEqual(sorted(on['per_rank']), ['0', '1', '2', '3'])
        self.assertEqual(on['per_rank']['2']['borrowed_prefill_sms'], 18)
        d = self.doc['compare']['decision_inputs']
        self.assertEqual(d['b_split_4_live_off_less_on_ms'], {0: 1.5, 1: 1.5})
        self.assertEqual(d['native_bulk_rate_change_pct']['3'], round(100 * (270 / 290 - 1), 2))
        text = l2.text(self.doc)
        self.assertIn('decision inputs: B_split 4 live off-on stage0 1.500 stage1 1.500 ms', text)
        self.assertIn('r0 14.300 (n 60, full/24) | 12.800 (n 60, full/42) | 1.500', text)

    def test_log_line_fallback(self):
        line = '%s flash-s5 decode-overlap {"schema":"s5-decode-overlap/1","borrowed_prefill_sms":18}' % qst.iso(W0 + 1)
        root = packet(42, 12.8, 290.0, inventory=False, extra_lines=[line])
        rec = l2.overlap_records(str(root))
        self.assertEqual((rec['source'], rec['log_lines'][0]['borrowed_prefill_sms']), ('log', 18))

    def test_main_writes_fresh_out_only(self):
        out = Path(tempfile.mkdtemp(prefix='l2read-')) / 'out'
        with mock.patch('sys.stdout'):
            self.assertEqual(l2.main(['--off', str(self.off), '--on', str(self.on), '--out', str(out)]), 0)
        self.assertEqual(json.loads((out / 'lever2.json').read_text())['schema'], l2.SCHEMA)
        with self.assertRaises(SystemExit):
            l2.main(['--off', str(self.off), '--on', str(self.on), '--out', str(out)])


class Mutants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.off = packet(24, 14.3, 270.0)
        cls.on = packet(42, 12.8, 290.0)

    def test_stage_sum_is_caught(self):
        with mock.patch.object(l2.qs, 'stage_value', lambda per: {s: sum((per[r]['p50'] or 0) for r in rs)
                                                                  for s, rs in l2.qs.STAGES.items()}):
            doc = l2.read(str(self.off), str(self.on))
        four = next(r for r in doc['compare']['forwards']['B_split'] if r['live'] == 4)
        self.assertNotEqual(four['stages'][0]['off'], 14.7)

    def test_rate_over_busy_is_caught(self):
        real = l2.bf.at
        with mock.patch.object(l2.bf, 'at', lambda base, off: real(base, off) if off == 0 else real(base, 9000)):
            doc = l2.read(str(self.off), str(self.on))
        self.assertNotAlmostEqual(doc['compare']['native']['0']['1025-2048']['off']['rows_per_s_p50'], 2048 / 0.270,
                                  places=2)

    def test_cancelled_kept_is_caught(self):
        real = l2.bf.num
        with mock.patch.object(l2.bf, 'num', lambda v: 0 if v == 1 else real(v)):
            doc = l2.read(str(self.off), str(self.on))
        self.assertNotEqual(doc['off']['native']['refusals'], dict(cancelled_or_errors=4, not_inside_n3_arm=4))



def lookup_line(wall_s, hexid, reason, bulk):
    stamp = '2026-11-14T12:%02d:%02d.%06d000Z' % (int(wall_s) // 60, int(wall_s) % 60, int(wall_s * 1e6) % 10 ** 6)
    body = dict(request_uid='cmpl-q18a-%s-0-ab12' % hexid, reason=reason, bulk_uid=bulk, queue_position=0)
    return '%s (EngineCore pid=1) INFO x [scheduler_hooks.py:1] flash-s5 n3-yield-lookup %s' % (stamp, json.dumps(body))


def status(windows, next_left, ends, fix=None):
    qp = dict(yield_window=dict(windows=windows, appends=windows, next_left=next_left, reentries_avoided=0, ends=ends,
                                per_window={'1': windows}))
    if fix is not None:
        qp['yield_window_fix'] = fix
    return dict(native_admission=dict(queue_policy=qp))


def append_root(*, mechanism=True, fix=False):
    """Two n3 arms: direct, stranded, stranded+quota and no-lookup burst appends, one at t = 150 (inclusive) and one
    after the burst; a fresh turn and the header are not appends. Engine-cumulative window counters."""
    root = Path(tempfile.mkdtemp(prefix='l2app-'))
    turns = {'n3.g2-live-0': [('append', 10.0, 2.0, 'aa'), ('append', 20.0, 14.0, 'bb'), ('fresh', 30.0, 9.0, 'ff'),
                              ('append', 150.0, 3.0, 'cc')],
             'n3.g2-live-1': [('append', 40.0, 21.0, 'dd'), ('append', 60.0, 1.0, 'ee'), ('append', 200.0, 5.0, 'gg')]}
    for arm, rows in turns.items():
        lines = [json.dumps(dict(label=arm, replay_start_s=1000.0, header=True))]
        lines += [json.dumps(dict(agent=0, kind=k, t=t, ttft=ttft, client_request_id='q18a-' + h)) for k, t, ttft, h in rows]
        (root / ('arm-%s.json.turns.jsonl' % arm)).write_text('\n'.join(lines) + '\n')
    log = [lookup_line(1, 'aa', 'admission-plan', 'B1'),
           lookup_line(2, 'bb', 'planned', 'B1'), lookup_line(9, 'bb', 'admission-plan', 'B2'),
           lookup_line(3, 'cc', 'admission', 'B2'),
           lookup_line(4, 'dd', 'planned', 'B2'), lookup_line(5, 'dd', 'quota-active', 'B2'),
           lookup_line(20, 'dd', 'admission-plan', 'B3'),
           lookup_line(21, 'gg', 'admission-plan', 'B3'), 'not a json line flash-s5 n3-yield-lookup {']
    (root / 'engine.log').write_text('\n'.join(log) + '\n')
    zero = dict(begins=0, own_pause_passed=0, still_refused=0, reoffered=0)
    (root / 'arm-n3.g2-live-0.json').write_text(json.dumps(dict(statuses=[
        status(1, 0, {'not-served': 1}, zero if fix else None), status(3, 1, {'not-served': 1, 'next-left': 1})])))
    (root / 'arm-n3.g2-live-1.json').write_text(json.dumps(dict(statuses=[
        status(3, 1, {}), status(5, 1, {'not-served': 2, 'next-left': 1, 'none-waiting': 1},
                                 dict(begins=4, own_pause_passed=3, still_refused=1, reoffered=1) if fix else None)])))
    if mechanism:
        (root / 'n3j2-mechanism.json').write_text(json.dumps(dict(facts=dict(burst_append=dict(
            agent_s=40.0, turns=5, mean_ttft_s=8.0, cap_agent_s=244.94)))))
    return root


class Appends(unittest.TestCase):
    """Per-step brief: burst-append agent-s and TTFT beside decode ms, the stranded split, the window counters."""

    @classmethod
    def setUpClass(cls):
        cls.base = l2.side(str(packet(24, 14.3, 270.0)))

    def test_split_and_scopes(self):
        a = l2.appends(append_root())
        b = a['burst']
        self.assertEqual(b['total'], dict(turns=5, agent_s=41.0, mean_ttft_s=8.2, p50_ttft_s=3.0))
        self.assertEqual((b['direct']['turns'], b['direct']['agent_s']), (2, 5.0))           # aa, cc (t = 150 inclusive)
        self.assertEqual((b['stranded']['turns'], b['stranded']['agent_s']), (1, 14.0))      # bb
        self.assertEqual((b['stranded+quota']['turns'], b['stranded+quota']['agent_s']), (1, 21.0))  # dd
        self.assertEqual((b['no-lookup']['turns'], b['no-lookup']['agent_s']), (1, 1.0))     # ee
        self.assertEqual((a['all']['total']['turns'], a['all']['direct']['turns']), (6, 3))  # + gg at t 200
        self.assertEqual(a['mechanism_burst']['agent_s'], 40.0)
        self.assertEqual(a['missing'], [])

    def test_window_counters_are_arm_deltas(self):
        a = l2.appends(append_root())
        self.assertEqual(a['window']['windows'], 4)
        self.assertEqual(a['window']['next_left'], 1)
        self.assertEqual(a['window']['ends'], {'not-served': 1, 'next-left': 1, 'none-waiting': 1})
        self.assertIsNone(a['window_fix'])
        on = l2.appends(append_root(fix=True))
        self.assertEqual(on['window_fix'], dict(begins=4, own_pause_passed=3, still_refused=1, reoffered=1))

    def test_compare_and_text(self):
        off, on = append_root(), append_root(mechanism=False, fix=True)
        with mock.patch.object(l2, 'side', lambda root, log=None: dict(
                appends=l2.appends(root, log), forwards=self.base['forwards'], native=self.base['native'],
                row_counts={}, refusals={}, hard_stops=[], overlap=dict(source='absent', per_rank={}), root=str(root))):
            doc = l2.read(str(off), str(on))
        c = doc['compare']
        self.assertEqual(c['burst']['off'], dict(agent_s=40.0, mean_ttft_s=8.0, turns=5, source='mechanism'))
        self.assertEqual(c['burst']['on'], dict(agent_s=41.0, mean_ttft_s=8.2, turns=5, source='turns'))
        self.assertEqual(c['decision_inputs']['burst_append_agent_s_on_less_off'], 1.0)
        self.assertEqual(c['decision_inputs']['burst_cap_agent_s'], 244.94)
        text = l2.text(doc)
        self.assertIn('burst-append agent-s on-off 1.000 (off 40.000, on 41.000; cap 244.940); mean TTFT on-off 0.200 s',
                      text)
        self.assertIn('on window fix: {', text)
        self.assertIn('off burst: total 5, 41.000, 8.200 / 3.000; direct 2, 5.000', text)

    def test_absent_records(self):
        root = Path(tempfile.mkdtemp(prefix='l2app-empty-'))
        a = l2.appends(root)
        self.assertEqual((a['burst'], a['window'], a['mechanism_burst']), ({}, None, None))
        self.assertEqual(len(a['missing']), 2)


class AppendMutants(unittest.TestCase):
    def test_exclusive_burst_end_is_caught(self):
        with mock.patch.object(l2, 'BURST_S', (0.0, 149.999)):
            with self.assertRaises(AssertionError):
                Appends('test_split_and_scopes').test_split_and_scopes()

    def test_absolute_counters_are_caught(self):
        real = l2._counter_delta
        with mock.patch.object(l2, '_counter_delta', lambda a, b: real(None, b)):
            with self.assertRaises(AssertionError):
                Appends('test_window_counters_are_arm_deltas').test_window_counters_are_arm_deltas()

    def test_stranded_by_last_lookup_is_caught(self):
        real = l2._admission_class
        with mock.patch.object(l2, '_admission_class', lambda found: real(found[-1:] if found else found)):
            with self.assertRaises(AssertionError):
                Appends('test_split_and_scopes').test_split_and_scopes()


if __name__ == '__main__':
    unittest.main()
