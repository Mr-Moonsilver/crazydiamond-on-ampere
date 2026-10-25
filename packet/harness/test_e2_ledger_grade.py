"""e2_ledger_grade: free reserve at the sampled peak, matched-baseline overhead, cumulative counters (CPU only)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_ledger_grade as g

RANKS = g.RANKS


def ledger(peak=22.0, retries=2, ooms=0, used=22.5, rows=True):
    return dict(per_rank={r: (dict(alloc_retries=retries, ooms=ooms, reserved_gib=peak, allocated_gib=peak - 1, peak_gib=peak) if rows else {}) for r in RANKS},
                gpus=[dict(index=i, bus='x', total_gib=24.0, used_gib=used) for i in range(4)])


def peaks(used_mib=23040, n=4, total_mib=24576):
    return dict(gpus={str(i): dict(max_used_mib=used_mib, total_mib=total_mib) for i in range(n)}, samples=100)


class LedgerGradeTests(unittest.TestCase):
    def test_pass_with_reserve_free_and_retries_within_baseline(self):
        out = g.grade(ledger(retries=1), ledger(peak=22.3, retries=3), peaks(), ledger(peak=21.1, retries=4, used=23.5))
        self.assertEqual(out['verdict'], 'PASS', out)
        self.assertEqual(out['ranks'][RANKS[0]]['retries_phase_delta'], 2)
        self.assertEqual(out['gpus']['0']['free_at_peak_gib'], 1.5)

    def test_reserve_is_free_memory_at_peak_not_a_baseline_allowance(self):
        # Codex counterexample: peak used = baseline used + 0.49 GiB with only 11 MiB free must FAIL
        base = ledger(peak=21.1, retries=4, used=23.5)
        self.assertEqual(g.grade(base, base, peaks(used_mib=24576 - 11), base)['verdict'], 'FAIL')
        self.assertEqual(g.grade(base, base, peaks(used_mib=24576 - 512), base)['verdict'], 'PASS')
        self.assertEqual(g.grade(base, base, peaks(used_mib=24576 - 511), base)['verdict'], 'FAIL')

    def test_missing_start_rows_is_noted_not_invented(self):
        out = g.grade(ledger(rows=False), ledger(retries=3), peaks(), ledger(retries=4, used=23.5))
        self.assertEqual(out['verdict'], 'PASS')
        self.assertNotIn('retries_phase_delta', out['ranks'][RANKS[0]])
        self.assertEqual(out['ranks'][RANKS[0]]['retries_since_start'], 3)
        self.assertTrue(any('phase start' in n for n in out['notes']))

    def test_missing_baseline_peaks_or_end_rows_is_inconclusive(self):
        self.assertEqual(g.grade(ledger(), ledger(), peaks(), None)['verdict'], 'INCONCLUSIVE')
        self.assertEqual(g.grade(ledger(), ledger(), None, ledger())['verdict'], 'INCONCLUSIVE')
        end = ledger(); del end['per_rank'][RANKS[3]]
        self.assertEqual(g.grade(ledger(), end, peaks(), ledger())['verdict'], 'INCONCLUSIVE')
        self.assertEqual(g.grade(ledger(), ledger(), peaks(n=3), ledger())['verdict'], 'INCONCLUSIVE')
        self.assertEqual(g.grade(ledger(), ledger(), peaks(), dict(per_rank={}, gpus=[]))['verdict'], 'INCONCLUSIVE')

    def test_ooms_backwards_counter_and_retry_excess_fail(self):
        self.assertEqual(g.grade(ledger(), ledger(ooms=1), peaks(), ledger(retries=4))['verdict'], 'FAIL')
        self.assertEqual(g.grade(ledger(retries=5), ledger(retries=2), peaks(), ledger(retries=4))['verdict'], 'FAIL')
        self.assertEqual(g.grade(ledger(retries=0), ledger(retries=5), peaks(), ledger(retries=4))['verdict'], 'FAIL')
        out = g.grade(ledger(retries=0), ledger(peak=23.0, retries=4), peaks(), ledger(peak=21.1, retries=4))
        self.assertEqual(out['verdict'], 'PASS'); self.assertAlmostEqual(out['ranks'][RANKS[0]]['peak_delta_gib'], 1.9)

    def test_reserve_only_mode_reports_without_attributing(self):
        # no control: PASS on the reserve alone, the missing control is a note; retries above a reference stay a note
        out = g.grade(ledger(), ledger(retries=9), peaks(), None, reserve_only=True)
        self.assertEqual(out['verdict'], 'PASS'); self.assertEqual(out['mode'], 'reserve-only')
        self.assertTrue(any('matched control' in n for n in out['notes']))
        out = g.grade(ledger(), ledger(retries=9), peaks(), ledger(retries=4), reserve_only=True)
        self.assertEqual(out['verdict'], 'PASS'); self.assertTrue(any('not attributed' in n for n in out['notes']))
        # the reserve still decides in reserve-only mode
        self.assertEqual(g.grade(ledger(), ledger(), peaks(used_mib=24576 - 100), None, reserve_only=True)['verdict'], 'FAIL')
        # full mode without a control stays INCONCLUSIVE
        self.assertEqual(g.grade(ledger(), ledger(), peaks(), None)['verdict'], 'INCONCLUSIVE')

    def test_matched_point_compares_cumulative_retries_without_peaks(self):
        # Codex q16 v3 review P2-2: shipping-vs-control is a regression comparison only at a matched point (same steps,
        # same seeds since start); the phase sampler is still running there, so peaks are optional, not INCONCLUSIVE
        ctrl = ledger(retries=4, peak=21.1)
        out = g.grade(ledger(rows=False), ledger(retries=4, peak=21.3), None, ctrl, matched=True)
        self.assertEqual((out['verdict'], out['mode']), ('PASS', 'matched'), out)
        self.assertTrue(any('matched point' in n for n in out['notes']))
        self.assertEqual(out['ranks'][RANKS[0]]['baseline_retries'], 4)
        self.assertAlmostEqual(out['ranks'][RANKS[0]]['peak_delta_gib'], 0.2)
        self.assertEqual(g.grade(ledger(rows=False), ledger(retries=5), None, ctrl, matched=True)['verdict'], 'FAIL')
        self.assertEqual(g.grade(ledger(rows=False), ledger(ooms=1, retries=4), None, ctrl, matched=True)['verdict'], 'FAIL')
        # a matched point still needs the control record, four END rows; peaks when given still grade the reserve
        self.assertEqual(g.grade(ledger(rows=False), ledger(retries=4), None, None, matched=True)['verdict'], 'INCONCLUSIVE')
        self.assertEqual(g.grade(ledger(rows=False), ledger(retries=4), peaks(used_mib=24576 - 100), ctrl, matched=True)['verdict'], 'FAIL')
        self.assertEqual(g.grade(ledger(rows=False), ledger(retries=4), peaks(), ctrl, matched=True)['verdict'], 'PASS')
        with self.assertRaises(ValueError): g.grade(ledger(), ledger(), peaks(), ctrl, reserve_only=True, matched=True)

    def test_retry_only_separates_attributable_failures_from_hard_ones(self):
        ctrl = ledger(retries=4)
        out = g.grade(ledger(rows=False), ledger(retries=5), None, ctrl, matched=True)
        self.assertEqual((out['verdict'], out['retry_only']), ('FAIL', True))
        out = g.grade(ledger(rows=False), ledger(retries=5, ooms=1), None, ctrl, matched=True)
        self.assertEqual((out['verdict'], out['retry_only']), ('FAIL', False))
        self.assertFalse(g.grade(ledger(retries=5), ledger(retries=2), peaks(), ctrl)['retry_only'])          # counter backwards
        self.assertFalse(g.grade(ledger(), ledger(), peaks(used_mib=24576 - 100), ctrl, matched=True)['retry_only'])  # reserve
        self.assertFalse(g.grade(ledger(rows=False), ledger(retries=4), None, ctrl, matched=True)['retry_only'])     # PASS

    def test_matched_window_delta_when_both_start_records_have_rows(self):
        ctrl_start, ctrl = ledger(retries=1), ledger(retries=4)
        # same cumulative total, larger window delta than the control: FAIL (candidate started lower, gained more in the window)
        out = g.grade(ledger(retries=0), ledger(retries=4), None, ctrl, matched=True, baseline_start=ctrl_start)
        self.assertEqual(out['verdict'], 'FAIL', out); self.assertEqual(out['ranks'][RANKS[0]]['baseline_retries_delta'], 3)
        self.assertEqual(g.grade(ledger(retries=1), ledger(retries=4), None, ctrl, matched=True, baseline_start=ctrl_start)['verdict'], 'PASS')
        # a START record without rows on either side: no window comparison, noted; the cumulative comparison still decides
        out = g.grade(ledger(rows=False), ledger(retries=4), None, ctrl, matched=True, baseline_start=ctrl_start)
        self.assertEqual(out['verdict'], 'PASS'); self.assertTrue(any('matched-window delta not available' in n for n in out['notes']))
        out = g.grade(ledger(retries=1), ledger(retries=4), None, ctrl, matched=True, baseline_start=ledger(rows=False))
        self.assertEqual(out['verdict'], 'PASS'); self.assertNotIn('baseline_retries_delta', out['ranks'][RANKS[0]])


if __name__ == '__main__':
    unittest.main()
