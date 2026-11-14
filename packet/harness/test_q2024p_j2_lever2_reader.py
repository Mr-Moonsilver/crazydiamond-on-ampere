#!/usr/bin/env python3
"""Lever 2 pair reader (tools/q2024p_j2_lever2_reader.py). CPU only, hub only; synthetic records in fresh mkdtemp roots
(left in place). The decode-timeline rows come from test_q2024p_j2_quiet_split_reader's Rec (its receipt format), the
anatomy chunk lines and the construction inventory are written here. Covers: B_split / C_split / Q_full p50 per rank
and live count with the bank_sms key (24 off, 42 on), the stage value as the max of the peers, off less on; native
chunk rate per rank and width band with its change in percent; cancelled chunks and chunks outside an n3 arm refused;
the overlap record from the inventory JSON, else from the log line; the decision inputs at 4 live. Mutants, each
caught: the stage value as a sum, the chunk rate over busy_us instead of the device window, cancelled chunks kept.
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


if __name__ == '__main__':
    unittest.main()
