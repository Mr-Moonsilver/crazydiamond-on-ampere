"""e2_ledger_grade: the declared-reserve memory contract on ledger records and sampled peaks (CPU only)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_ledger_grade as g

RANKS = g.RANKS


def ledger(peak=22.0, retries=2, ooms=0, used=22.5):
    return dict(per_rank={r: dict(alloc_retries=retries, ooms=ooms, reserved_gib=peak, allocated_gib=peak - 1, peak_gib=peak) for r in RANKS},
                gpus=[dict(index=i, bus='x', total_gib=24.0, used_gib=used) for i in range(4)])


def peaks(used_mib=23040, n=4):
    return dict(gpus={str(i): dict(max_used_mib=used_mib, total_mib=24576) for i in range(n)}, samples=100)


class LedgerGradeTests(unittest.TestCase):
    def test_pass_inside_reserve_and_retry_budget(self):
        out = g.grade(ledger(retries=1), ledger(peak=22.3, retries=3), peaks(), ledger(peak=22.0, retries=4, used=22.5))
        self.assertEqual(out['verdict'], 'PASS', out)
        self.assertEqual(out['ranks'][RANKS[0]]['retries_delta'], 2)

    def test_missing_baseline_peaks_or_rows_is_inconclusive(self):
        self.assertEqual(g.grade(ledger(), ledger(), peaks(), None)['verdict'], 'INCONCLUSIVE')
        self.assertEqual(g.grade(ledger(), ledger(), None, ledger())['verdict'], 'INCONCLUSIVE')
        end = ledger(); del end['per_rank'][RANKS[3]]
        self.assertEqual(g.grade(ledger(), end, peaks(), ledger())['verdict'], 'INCONCLUSIVE')
        self.assertEqual(g.grade(ledger(), ledger(), peaks(n=3), ledger())['verdict'], 'INCONCLUSIVE')

    def test_ooms_backwards_counter_peak_and_retry_excess_fail(self):
        self.assertEqual(g.grade(ledger(), ledger(ooms=1), peaks(), ledger())['verdict'], 'FAIL')
        self.assertEqual(g.grade(ledger(retries=5), ledger(retries=2), peaks(), ledger())['verdict'], 'FAIL')
        self.assertEqual(g.grade(ledger(), ledger(peak=22.6), peaks(), ledger(peak=22.0))['verdict'], 'FAIL')
        self.assertEqual(g.grade(ledger(retries=0), ledger(retries=6), peaks(), ledger(retries=4))['verdict'], 'FAIL')
        self.assertEqual(g.grade(ledger(), ledger(), peaks(used_mib=23700), ledger(used=22.5))['verdict'], 'FAIL')

    def test_reserve_bytes_parameter_scales_the_margin(self):
        self.assertEqual(g.grade(ledger(), ledger(peak=22.4), peaks(), ledger(peak=22.0), reserve_bytes=256 * 1024 ** 2)['verdict'], 'FAIL')
        self.assertEqual(g.grade(ledger(), ledger(peak=22.4), peaks(), ledger(peak=22.0), reserve_bytes=512 * 1024 ** 2)['verdict'], 'PASS')


if __name__ == '__main__':
    unittest.main()
