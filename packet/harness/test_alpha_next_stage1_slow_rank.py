#!/usr/bin/env python3
"""alpha-next 4C.2: the stage-1 anatomy report's slower TP rank per chunk and per band (CPU, synthetic handler rows).

Rows are the report's own main-row shape (rank, seq, wall_ns, handle_output_ns, seams with excl_ns); the seq -> chunk map is the
shape seq_context returns. A per-sequence majority can name one rank while the per-chunk sum names the other; the test holds
such a chunk, a tie, a sequence only one rank logged (not counted), an unassigned sequence, and two bands.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'tools'), HERE]
import stage1_anatomy_report as rep  # noqa: E402


def row(rank, seq, wall_ms, seams=None, handle_ms=0):
    return dict(rank=rank, seq=seq, wall_ns=int(wall_ms * 1e6), handle_output_ns=int(handle_ms * 1e6), kind='e', flags=0,
                observer_ns=0, seams={k: dict(excl_ns=int(v * 1e6), ns=int(v * 1e6), cpu_ns=0) for k, v in (seams or {}).items()})


class SlowerRankPerChunk(unittest.TestCase):
    def setUp(self):
        w = {}
        rows = []
        # Chunk A (band 0-16384): rank 2 slower on 2 of 3 sequences by 1 ms each, rank 3 slower on one by 10 ms -> rank 3 per chunk.
        for seq, (r2, r3) in zip((1, 2, 3), ((11, 10), (11, 10), (10, 20))):
            rows += [row(2, seq, r2, {'forward': r2 - 5}), row(3, seq, r3, {'forward': r3 - 15 if r3 == 20 else r3 - 5,
                                                                              'pp_recv_wait': 10 if r3 == 20 else 0})]
            w[seq] = dict(arm='g2-live-0', chunk_start=1024, chunk_end=2048)
        # A sequence only rank 2 logged: not counted.
        rows.append(row(2, 4, 99))
        w[4] = dict(arm='g2-live-0', chunk_start=1024, chunk_end=2048)
        # Chunk B (beyond the last band, whatever band table is loaded: e3_alpha_perf_box widens it at import): a tie.
        self.beyond = rep.box.BANDS[-1][1] + 1000
        rows += [row(2, 5, 7, handle_ms=1), row(3, 5, 8)]
        w[5] = dict(arm='g2-live-0', chunk_start=self.beyond, chunk_end=self.beyond + 1024)
        # Chunk C (band 16384-32768), another arm: rank 2 slower, carried by handle_output-free 'execute'.
        rows += [row(2, 6, 30, {'execute': 25}), row(3, 6, 12, {'execute': 7})]
        w[6] = dict(arm='g2-solo-0', chunk_start=20000, chunk_end=21024)
        # Unassigned sequence (no chunk flush): skipped.
        rows += [row(2, 7, 50), row(3, 7, 1)]
        self.rows, self.where = rows, w
        self.out = rep.slower_rank_per_chunk(rows, w, '1')

    def test_chunk_table(self):
        table = {(c['arm'], c['chunk_start']): c for c in self.out['chunks']}
        self.assertEqual(set(table), {('g2-live-0', 1024), ('g2-live-0', self.beyond), ('g2-solo-0', 20000)})
        a = table[('g2-live-0', 1024)]
        self.assertEqual((a['sequences'], a['slower_rank'], a['band']), (3, 3, rep.box.band_of(1024)))
        self.assertAlmostEqual(a['handler_ms']['2'], 32.0)
        self.assertAlmostEqual(a['handler_ms']['3'], 40.0)
        self.assertAlmostEqual(a['difference_ms'], 8.0)
        self.assertEqual(a['carrier'], 'pp_recv_wait')
        b = table[('g2-live-0', self.beyond)]
        self.assertEqual((b['slower_rank'], b['carrier'], b['band']), (None, None, 'beyond-%d' % rep.box.BANDS[-1][1]))
        c = table[('g2-solo-0', 20000)]
        self.assertEqual((c['slower_rank'], c['carrier']), (2, 'execute'))

    def test_per_sequence_majority_differs_from_per_chunk(self):
        _, summary = rep.slower_rank(self.rows, '1')
        # Per sequence rank 2 is slower more often (seqs 1, 2, 6, 7); per chunk rank 3 owns the live chunk.
        self.assertGreater(summary['slower_rank_counts']['2'], summary['slower_rank_counts']['3'])
        self.assertEqual(self.out['per_arm']['g2-live-0']['slower_rank_counts'], {'2': 0, '3': 1, 'tie': 1})

    def test_per_band_and_arm_rollups(self):
        bands = self.out['per_band']
        self.assertEqual(sum(b['chunks'] for b in bands.values()), 3)
        self.assertEqual(bands['beyond-%d' % rep.box.BANDS[-1][1]]['slower_rank_counts']['tie'], 1)
        self.assertEqual(self.out['per_arm']['g2-solo-0']['slower_rank_counts'], {'2': 1, '3': 0, 'tie': 0})
        self.assertEqual(self.out['per_arm']['g2-solo-0']['carrier_counts'], {'execute': 1})
        self.assertEqual(self.out['ranks'], [2, 3])

    def test_stage_zero_has_no_rows_here(self):
        out = rep.slower_rank_per_chunk(self.rows, self.where, '0')
        self.assertEqual((out['chunks'], out['per_band']), ([], {}))

    def test_band_names_follow_bands(self):
        self.assertEqual(rep.band_name(self.beyond), 'beyond-%d' % rep.box.BANDS[-1][1])
        self.assertEqual(rep.band_name(20000), rep.box.band_of(20000))
        self.assertIsNone(rep.band_name(None))


if __name__ == '__main__':
    unittest.main()
