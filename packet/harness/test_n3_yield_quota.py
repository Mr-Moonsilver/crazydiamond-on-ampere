"""Quota observation and eight real host/worker yield cycles; device leaves declared."""
from pathlib import Path
import inspect
import json
import sys
import textwrap
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'harness'), str(ROOT/'patches/s5_engine')]
import engine_driver as ed
import scheduler_hooks as sh
import test_append_yield as a1
import test_n3_junction_yield as junction


class QuotaReceipts(unittest.TestCase):
    def setUp(self):
        self.logger = mock.Mock()
        patch = mock.patch.object(ed, '_progress_logger', return_value=self.logger)
        patch.start(); self.addCleanup(patch.stop)

    def test_queue_prefix_eight_explicit_bulk_and_no_cache_or_allocation(self):
        c, s, manager, bulk = junction.candidate()
        requests = [a1.Req('waiter-%d' % i, 12000) for i in range(12)]
        s.skipped_waiting.items[:] = requests[:3]
        s.waiting.items[:] = requests[3:]
        with mock.patch.object(manager, 'get_computed_blocks', side_effect=AssertionError('no lookup')):
            c._yield_quota_receipts(s, bulk, 8, 8)
            c._yield_quota_receipts(s, bulk, 8, 8)
        self.assertEqual(self.logger.info.call_count, 8)
        rows = [json.loads(call.args[1]) for call in self.logger.info.call_args_list]
        self.assertEqual([row['request_uid'] for row in rows], [r.request_id for r in requests[:8]])
        for row in rows:
            self.assertEqual({k: row[k] for k in ('reason', 'bulk_uid', 'quota_used', 'quota_limit',
                              'queue_snapshot_limit', 'eligibility', 'hit_end', 'junction', 'basis')},
                dict(reason='quota-active', bulk_uid=bulk.request_id, quota_used=8, quota_limit=8,
                     queue_snapshot_limit=8, eligibility='not-evaluated', hit_end=None, junction=None,
                     basis='yield-quota-observation'))
        self.assertEqual(rows[0]['queue'], 'skipped_waiting')
        self.assertEqual(rows[3]['queue'], 'waiting')
        self.assertEqual(rows[3]['queue_position'], 0)
        self.assertEqual(c.yield_receipts['duplicate'], 8)
        self.assertIsNone(c.yield_refusals)
        self.assertEqual(manager.alloc_calls, [])

    def test_new_waiter_same_queue_length_is_observed_and_bulk_is_never_its_own_waiter(self):
        c, s, manager, bulk = junction.candidate()
        first = a1.Req('first', 12000)
        s.waiting.items[:] = [bulk, first]
        c._yield_quota_receipts(s, bulk, 8, 8)
        second = a1.Req('second', 12000)
        s.waiting.items[1] = second
        c._yield_quota_receipts(s, bulk, 8, 8)
        self.assertEqual([json.loads(call.args[1])['request_uid'] for call in self.logger.info.call_args_list],
                         ['first', 'second'])

    def test_snapshot_never_reads_a_ninth_queue_entry(self):
        c, s, manager, bulk = junction.candidate()
        requests = [a1.Req('bounded-%d' % i, 12000) for i in range(8)]
        class Queue:
            def __iter__(self):
                yield from requests
                raise AssertionError('observer read a ninth queue entry')
        s.skipped_waiting = Queue()
        c._yield_quota_receipts(s, bulk, 8, 8)
        self.assertEqual(c.yield_receipts['errors'], 0)
        self.assertEqual(self.logger.info.call_count, 8)

    def test_flag_off_logging_fault_and_shared_limit_are_report_only(self):
        c, s, manager, bulk = junction.candidate()
        s.waiting.items[:] = [a1.Req('waiting', 12000)]
        c.junction_yield = False
        c._yield_quota_receipts(s, bulk, 8, 8)
        self.assertEqual(self.logger.info.call_count, 0)
        c.junction_yield = True
        self.logger.info.side_effect = OSError('declared quota log fault')
        c._yield_quota_receipts(s, bulk, 8, 8)
        self.assertEqual(c.yield_receipts['errors'], 1)
        c._yield_receipt_seen.clear()
        c.yield_receipts['written'] = 4096
        c._yield_quota_receipts(s, bulk, 8, 8)
        self.assertEqual(c.yield_receipts['dropped'], 1)
        self.assertIsNone(c.yield_refusals)
        self.assertEqual(manager.alloc_calls, [])

    def test_invalid_snapshot_failure_cannot_change_admission_or_quota(self):
        c, s, manager, bulk = junction.candidate()
        class BrokenQueue:
            def __iter__(self):
                raise RuntimeError('declared queue observer fault')
        s.skipped_waiting = BrokenQueue()
        c._yield_quota_receipts(s, bulk, 8, 8)
        self.assertEqual(c.yield_receipts['errors'], 1)
        self.assertIsNone(c.yield_refusals)
        self.assertEqual(bulk.num_computed_tokens, 0)
        self.assertEqual(manager.alloc_calls, [])


class EightYieldCycles(unittest.TestCase):
    def test_eight_closes_and_reopens_then_ninth_waits_without_a_frontier_move(self):
        for driver in ('native', 'staged'):
            with self.subTest(driver=driver), mock.patch.object(ed, '_progress_logger', return_value=mock.Mock()) as log:
                rig = a1.Rig(cached={}, per_bulk=8, driver=driver)
                rig.c.junction_yield = True
                rig.bulk.num_tokens = rig.bulk.num_prompt_tokens = 80
                rig.bulk.all_token_ids = list(range(80))
                self.assertIsNotNone(rig.admit())
                for index in range(9):
                    append = rig.add_held('quota-return-%d' % index, 6, cached=4)
                    rig.finish_chunk()
                    frontier = rig.bulk.num_computed_tokens
                    self.assertEqual(frontier, (index+1)*4)
                    if index == 8:
                        self.assertIsNone(rig.d.append_yield_phase())
                        self.assertEqual(rig.d.append_yield_counters['selected'], 8)
                        self.assertEqual(rig.d.append_yield_counters['reopened'], 8)
                        self.assertEqual(rig.d.append_yield_counters['rows_closed'], 8)
                        receipts = [json.loads(c.args[1]) for c in log.return_value.info.call_args_list
                                    if c.args[0] == 'flash-s5 n3-yield-lookup %s']
                        quota = [r for r in receipts if r['reason'] == 'quota-active']
                        self.assertTrue(any(r['request_uid'] == append.request_id and r['quota_used'] == 8 for r in quota))
                        self.assertEqual(append.num_computed_tokens, 0)
                        continue
                    self.assertEqual(rig.d.append_yield_phase(), 'closing')
                    a1.to_append(rig)
                    transactions = a1.run_append(rig)
                    self.assertEqual(transactions[-1].chunk_end, 5)
                    self.assertEqual(rig.bulk.num_computed_tokens, frontier)
                    a1.to_full(rig)
                    rig.stock_handback(append)
                    rig.stock_finish(append)
                    a1.to_resume(rig)
                    opened = rig.admit()
                    self.assertEqual((opened.request_uid, opened.chunk_start), (rig.bulk.request_id, frontier))


class QuotaMutants(unittest.TestCase):
    def caught(self, cls, method, old, new, case):
        source = textwrap.dedent(inspect.getsource(getattr(cls, method)))
        self.assertEqual(source.count(old), 1)
        namespace = dict(vars(sh if cls is sh.PrefillAdmissionController else ed))
        exec(compile(source.replace(old, new, 1), '<quota-mutant>', 'exec'), namespace)
        with mock.patch.object(cls, method, namespace[method]):
            result = unittest.TestResult(); case.run(result)
        self.assertTrue(result.failures, 'mutant must fail an assertion')
        self.assertEqual(result.errors, [])

    def test_missing_quota_receipt_mutant(self):
        self.caught(ed.NativeSegmentDriver, '_yield_offer',
                    'controller._yield_quota_receipts(self.scheduler, bulk, self._yield_counts[key], config.per_bulk)',
                    'pass', EightYieldCycles('test_eight_closes_and_reopens_then_ninth_waits_without_a_frontier_move'))

    def test_snapshot_bound_mutant(self):
        self.caught(sh.PrefillAdmissionController, '_yield_quota_receipts',
                    'islice(chain.from_iterable(queues), 8)', 'islice(chain.from_iterable(queues), 9)',
                    QuotaReceipts('test_queue_prefix_eight_explicit_bulk_and_no_cache_or_allocation'))

    def test_quota_gate_ninth_yield_mutant(self):
        self.caught(ed.NativeSegmentDriver, '_yield_offer',
                    'if self._yield_counts.get(key, 0) >= config.per_bulk:',
                    'if self._yield_counts.get(key, 0) > config.per_bulk:',
                    EightYieldCycles('test_eight_closes_and_reopens_then_ninth_waits_without_a_frontier_move'))


if __name__ == '__main__':
    unittest.main()
