"""Native worker packet proof: source-produced ledgers, actual ctypes reads."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'harness'), str(ROOT / 'patches/s5_engine')]
from e3_native_placement_evidence import regrade_packet
from native_placement_evidence_support import packet


class PacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = packet()

    def setUp(self):
        self.value = copy.deepcopy(self.original)

    def grade(self):
        return regrade_packet(self.value, rank=0, engine_epoch=3,
            run_id='run-1', window_id='win-1', route_generation=7)

    def test_actual_ctypes_read_under_64mib_and_paired_raw_graph_proof_pass(self):
        before = copy.deepcopy(self.value)
        grade = self.grade()
        self.assertEqual(grade['verdict'], 'PASS', grade)
        self.assertEqual(grade['composed']['raw']['verdict'], 'INCONCLUSIVE')
        self.assertEqual(grade['composed']['graph']['by_kind']['memset'], 0)
        self.assertLess(self.value['native_report_read']['required_bytes'], 64*1024**2)
        self.assertEqual(self.value, before)

    def test_retained_window_after_typed_full_return_keeps_both_generations(self):
        self.value.update(route='full', current_route='full', current_route_generation=8)
        self.assertEqual(self.grade()['verdict'], 'PASS')
        self.assertEqual(self.value['route_generation'], 7)
        self.assertEqual(self.value['current_route_generation'], 8)

    def test_foreign_later_generation_not_a_window_receipt(self):
        self.value.update(current_route='full', current_route_generation=9)
        self.assertIn('current_route_bound', self.grade()['failures'])

    def test_missing_metadata_is_inconclusive_and_never_a_file_provider(self):
        self.value.pop('native_report_read')
        self.value['native_read'] = dict(bytes=64*1024**2, completed=True,
            source='retained-split-bank', basis='actual-native-read')
        self.assertIn('native_report_read', self.grade()['missing'])

    def test_failure_read_cannot_reuse_success(self):
        self.value['native_report_read']['completed'] = False
        self.assertIn('native_read_completed', self.grade()['failures'])

    def test_read_digest_cannot_borrow_a_different_native_report(self):
        self.value['native_report']['reserved_bytes'] += 1
        self.assertIn('native_read_digest', self.grade()['failures'])

    def test_capacity_and_actual_bytes_are_separate(self):
        for key, value in [('capacity_bytes', 1024), ('limit_bytes', 1024),
                           ('required_bytes', 64*1024**2+1)]:
            with self.subTest(key=key):
                self.value = copy.deepcopy(self.original)
                self.value['native_report_read'][key] = value
                self.assertEqual(self.grade()['verdict'], 'FAIL')

    def test_wrong_function_or_schema_is_not_a_correlation_read(self):
        for key, value in [('function','s5ro_window'), ('schema','s5ro-correlation/1')]:
            with self.subTest(key=key):
                self.value = copy.deepcopy(self.original)
                self.value['native_report_read'][key] = value
                self.assertEqual(self.grade()['verdict'], 'FAIL')

    def test_raw_fail_precedes_absent_read_metadata(self):
        self.value['receipt']['verdict'] = 'FAIL'
        self.value.pop('native_report_read')
        self.assertIn('raw_placement_fail', self.grade()['failures'])

    def test_native_context_failure_precedes_missing_read_metadata(self):
        self.value['native_report']['activities'][0]['context_id'] = 999
        self.value.pop('native_report_read')
        self.assertEqual(self.grade()['verdict'], 'FAIL')

    def test_no_correlation_or_prefill_receipt_never_qualifies(self):
        for key in ('native_report', 'native_prefill_accounting', 'decode_declared'):
            with self.subTest(key=key):
                self.value = copy.deepcopy(self.original)
                self.value.pop(key)
                self.assertEqual(self.grade()['verdict'], 'INCONCLUSIVE')

    def test_cached_composed_pass_is_ignored(self):
        self.value['composed_placement'] = {'verdict': 'PASS'}
        self.value['native_report']['overflow'] = 1
        self.assertEqual(self.grade()['verdict'], 'FAIL')

    def test_rank_run_epoch_or_window_cannot_be_rekeyed(self):
        for key, value in [('rank', 1), ('engine_epoch', 4), ('run_id', 'foreign'),
                           ('window_id','foreign'), ('route_generation',8)]:
            with self.subTest(key=key):
                kwargs = dict(rank=0, engine_epoch=3, run_id='run-1',
                              window_id='win-1', route_generation=7)
                kwargs[key] = value
                self.assertEqual(regrade_packet(self.value, **kwargs)['verdict'], 'FAIL')

    def test_all_four_source_rank_packets_pass_independently(self):
        for rank in range(4):
            result = regrade_packet(packet(rank), rank=rank, engine_epoch=3,
                run_id='run-1', window_id='win-1', route_generation=7)
            self.assertEqual(result['verdict'], 'PASS', result)


if __name__ == '__main__':
    unittest.main()
