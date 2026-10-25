"""Offline tests for e2_drain_freshness.py: the drained proof is FRESH only when every worker's final FLASH_E2_STATE
rpc_sequence equals its latest FLASH_E2_COW seq and no COW line follows it (Codex, 25 Oct 2026, external f020e3a review)."""
import json, sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_drain_freshness as f
import e2_state_grade as g
from test_e2_state_grade import state_line

RANKS = f.RANKS


def cow_line(rank, seq, pairs=0):
    return f'2026-10-25T13:01:00Z ({rank} pid=7) INFO runner_lifecycle.py:76] FLASH_E2_COW seq={seq} pairs={pairs} views=0 cumulative_pairs=3 cumulative_views=3'


class FreshnessTests(unittest.TestCase):
    def test_all_fresh(self):
        lines = []
        for r in RANKS: lines += [cow_line(r, 119), state_line(r, rpc_sequence=119), cow_line(r, 120), state_line(r, rpc_sequence=120)]
        out = f.grade(lines)
        self.assertEqual(out['verdict'], 'FRESH'); self.assertEqual(out['inconclusive'], [])
        self.assertTrue(all(w['freshness'] == 'fresh' for w in out['workers'].values()))
        self.assertEqual(g.drained(lines)['verdict'], 'PASS')

    def test_stale_by_sequence_is_inconclusive_even_when_drained_passes(self):
        lines = []
        for r in RANKS: lines += [cow_line(r, 10), state_line(r, rpc_sequence=10)]
        lines.append(cow_line('Worker_PP0_TP1', 11))   # a later completed RPC without a new state record
        self.assertEqual(g.drained(lines)['verdict'], 'PASS')   # the reproduced gap: stale proof passes the drained grader
        out = f.grade(lines)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE')
        self.assertEqual(out['workers']['Worker_PP0_TP1']['freshness'], 'stale')
        self.assertTrue(any('11 > final STATE rpc_sequence 10' in s for s in out['inconclusive']))
        self.assertEqual(out['workers']['Worker_PP0_TP0']['freshness'], 'fresh')

    def test_state_ahead_of_every_cow_is_inconclusive(self):
        lines = []
        for r in RANKS: lines += [cow_line(r, 20), state_line(r, rpc_sequence=20)]
        lines.append(state_line('Worker_PP0_TP0', rpc_sequence=21))   # a state record with no completed COW seq 21
        out = f.grade(lines)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE')
        self.assertEqual(out['workers']['Worker_PP0_TP0']['freshness'], 'stale')
        self.assertTrue(any('rpc_sequence 21 has no completed COW seq (latest 20)' in s for s in out['inconclusive']))

    def test_stale_by_order_same_sequence(self):
        lines = []
        for r in RANKS: lines += [cow_line(r, 7), state_line(r, rpc_sequence=7)]
        lines.append(cow_line('Worker_PP1_TP0', 7))
        out = f.grade(lines)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE')
        self.assertTrue(any('follows the final STATE line' in s for s in out['inconclusive']))

    def test_missing_rank_and_no_cow_are_inconclusive_not_fail(self):
        lines = [cow_line(r, 3) for r in RANKS[:3]] + [state_line(r, rpc_sequence=3) for r in RANKS[:3]]
        out = f.grade(lines)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE')
        self.assertEqual(out['workers']['Worker_PP1_TP1']['freshness'], 'missing')
        lines2 = [state_line(r, rpc_sequence=3) for r in RANKS]
        out2 = f.grade(lines2)
        self.assertEqual(out2['verdict'], 'INCONCLUSIVE')
        self.assertTrue(all(w['freshness'] == 'unproven' for w in out2['workers'].values()))
        self.assertNotEqual(out['verdict'], 'FAIL'); self.assertNotEqual(out2['verdict'], 'FAIL')

    def test_malformed_final_state_is_inconclusive_and_conservative_fields_reported(self):
        lines = []
        for r in RANKS: lines += [cow_line(r, 5), state_line(r, rpc_sequence=5, pending_producer_pages=2, metadata_pages=9)]
        out = f.grade(lines)
        self.assertEqual(out['verdict'], 'FRESH')
        self.assertEqual(out['workers']['Worker_PP0_TP0']['pending_producer_pages'], 2)
        self.assertEqual(out['workers']['Worker_PP0_TP0']['metadata_pages'], 9)
        lines.append('2026-10-25T13:02:00Z (Worker_PP0_TP0 pid=7) INFO runner_lifecycle.py:80] FLASH_E2_STATE {not json')
        out = f.grade(lines)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE')
        self.assertEqual(out['workers']['Worker_PP0_TP0']['freshness'], 'missing')
        self.assertEqual(out['workers']['Worker_PP0_TP0']['malformed_state_records'], 1)


if __name__ == '__main__':
    unittest.main()
