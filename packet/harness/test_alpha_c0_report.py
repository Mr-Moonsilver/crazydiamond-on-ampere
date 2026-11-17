#!/usr/bin/env python3
"""The C0 report's arithmetic on one hand-computed chunk. The real rows come from the box's progress lines."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import alpha_c0_report as report

MS = 1_000_000


def native(seq, stage, layer, submit, recv, apply_):
    row = dict(seq=seq, kind='layer', stage=stage, layer=layer, t_submit=submit, t_apply=apply_)
    row.update({'t_recv%d' % k: recv[k] for k in range(4)})
    return row


def handler(seq, kind, enter, leave):
    return dict(seq=seq, kind=kind, t_enter=enter, t_exit=leave, t_emit=leave)


class OneChunk(unittest.TestCase):
    def setUp(self):
        # Stage 0's command: own ranks reply at 10 ms, non-target at 12 ms, applied at 13 ms.
        # Stage 1's command: 20 ms to 40 ms, own ranks reply at 34 ms, all at 34 ms, applied at 40 ms.
        rows = [native(1, 0, 2, 0, [10 * MS, 10 * MS, 12 * MS, 12 * MS], 13 * MS),
                native(2, 1, 3, 20 * MS, [34 * MS, 34 * MS, 30 * MS, 34 * MS], 40 * MS)]
        # A decode RPC is ready at 22 ms; rank 2 is inside the native handler until 30 ms and rank 3 enters at 25 ms.
        stock = [dict(seq=3, kind='execute_model', t_submit=22 * MS, t_recv=45 * MS, t_take=46 * MS)]
        self.chunk = dict(identity=dict(chunk_start=0, chunk_end=1024, decode_live=8), rows=dict(n=rows, s=stock))
        workers = {}
        per_rank = {rank: [] for rank in range(4)}

        def add(rank, row):
            workers[(row['seq'], rank)] = row
            per_rank[rank].append(row)
        add(2, handler(2, 'flash_s5_submit', 21 * MS, 30 * MS))
        add(2, handler(3, 'execute_model', 30 * MS, 35 * MS))
        add(3, handler(3, 'execute_model', 25 * MS, 28 * MS))
        self.out = report.chunk_report(self.chunk, workers, {r: report.RankRows(v) for r, v in per_rank.items()})

    def test_owner_and_stage_time(self):
        out = self.out
        self.assertAlmostEqual(out['wall_s'], 0.040)
        self.assertAlmostEqual(out['owner_s'], 0.001 + 0.006)
        self.assertAlmostEqual(out['stage']['0']['own'], 0.010)
        self.assertAlmostEqual(out['stage']['0']['tax'], 0.002)
        self.assertAlmostEqual(out['stage']['1']['own'], 0.014)
        self.assertAlmostEqual(out['stage']['1']['tax'], 0.0)
        self.assertAlmostEqual(out['stages_both_s'], 0.0)
        self.assertAlmostEqual(out['stages_neither_s'], 0.040 - 0.013 - 0.020)

    def test_main_thread(self):
        rank2 = self.out['main_thread']['2']
        self.assertAlmostEqual(rank2['native_s'], 0.009)
        self.assertAlmostEqual(rank2['stock_s'], 0.005)
        self.assertAlmostEqual(rank2['idle_s'], 0.040 - 0.014)

    def test_decode_delay_is_the_slowest_rank_and_names_the_native_part(self):
        self.assertEqual(self.out['decode_delay_ms'], [8.0])
        self.assertAlmostEqual(self.out['decode_delay_behind_native_s']['2'], 0.008)
        self.assertAlmostEqual(self.out['decode_delay_behind_native_s']['3'], 0.0)

    def test_handlers_split_qsa_from_the_rest(self):
        self.assertEqual(self.out['handler_s']['qsa'], [0.009])
        self.assertEqual(self.out['handler_s']['other'], [])

    def test_service_model_scales_to_1024_rows(self):
        service = self.out['service_ms_per_1024']
        self.assertAlmostEqual(service['stage0'], 13.0)
        self.assertAlmostEqual(service['stage1'], 20.0)
        self.assertAlmostEqual(service['limit'], 20.0)
        self.assertAlmostEqual(self.out['interval_ms_per_1024'], 40.0)


if __name__ == '__main__':
    unittest.main()
