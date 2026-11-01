"""Real generated native attention/state machine; only CUDA objects are leaves."""
import json
import unittest
from unittest.mock import patch
import test_e3_sparse_history_dma as base
import qsa_operation as qo


class StartupPolicyTests(unittest.TestCase):
    def test_default_and_explicit_policy(self):
        self.assertEqual(qo.native_qsa_settings({}), ('current', False))
        self.assertEqual(qo.native_qsa_settings({qo.NATIVE_QSA_POLICY_ENV: 'dense',
            qo.NATIVE_QSA_TRACE_ENV: '1'}), ('dense', True))

    def test_invalid_declarations_refuse(self):
        for env in ({qo.NATIVE_QSA_POLICY_ENV: 'DENSE'},
                    {qo.NATIVE_QSA_TRACE_ENV: 'yes'},
                    {qo.NATIVE_QSA_POLICY_ENV: ''}):
            with self.assertRaises(qo.OperationError):
                qo.native_qsa_settings(env)


class DenseBoundaryTests(base._GeneratedHarness, unittest.TestCase):
    for _name in ('make_operation', 'hold_for', 'bind', 'prepare', 'open_layer'):
        locals()[_name] = getattr(base.GeneratedSparseTests, _name)

    def operation_at(self, pages, policy, trace=False):
        self.device.native_qsa_policy = policy
        self.device.native_qsa_trace = trace
        if trace:
            def event(stream):
                e = self.device.record(stream)
                e.elapsed_time = lambda other: 0.25
                return e
            self.device.timing_event = event
        mapping = tuple(range(1, pages + 1))
        operation, lane, tx, sources = self.prepare({'qsa.0': mapping})
        hold = self.open_layer(operation, lane, tx, 'qsa.0', sources['qsa.0'],
                               mapping, min_pages=24)
        return operation, sources['qsa.0'], hold

    def execute(self, pages, policy, trace=False):
        operation, source, hold = self.operation_at(pages, policy, trace)
        logical = base.T((8, 5), device='cuda:0', dtype='torch.int32',
                         values=[0, 1, 4048, 4049, 4] * 8)
        values = logical.values[:]
        query = base.T((8, 6, 256), device='cuda:0')
        output = base.T((8, 6, 256), device='cuda:0')
        rows = base.T((8,), device='cuda:0', dtype='torch.int64', values=list(range(8)))
        generator = self.attention(query, source, logical, self.table, rows,
                                  pages - 1, self.arena, True, output, e3op=operation)
        self.run_generator(operation, generator)
        operation.end_layer('qsa.0')
        operation.end_forward()
        self.device.ready_all()
        self.assertTrue(operation.drain())
        self.assertEqual(logical.values, values, 'policy must never mutate stock logical indices')
        return operation, hold

    def test_boundary_24_dense_for_both_policies_25_differs(self):
        for pages in (24, 25):
            for policy in ('current', 'dense'):
                with self.subTest(pages=pages, policy=policy):
                    self.setUp()
                    op, source, hold = self.operation_at(pages, policy)
                    self.assertEqual(op.sparse_selection_eligible(pages, 24),
                                     pages > 24 and policy == 'current')

    def test_dense_25_49_128_pages_keep_four_page_and_single_bucket_bounds(self):
        for pages in (25, 49, 128):
            with self.subTest(pages=pages):
                self.setUp()
                op, hold = self.execute(pages, 'dense')
                self.assertEqual(op.stats['buckets'], (pages + 23) // 24)
                self.assertTrue(all(1 <= n <= 4 for _, _, n in self.device.page_copies))
                self.assertEqual(sum(n for _, _, n in self.device.page_copies), pages)
                self.assertEqual(self.device.index_copies, [])
                self.assertEqual(self.device.host_reads, [])
                self.assertEqual(self.device.copies, [])
                self.assertEqual(op.pending_buckets, 0)
                self.assertEqual(self.device.synchronized, 0)
                self.assertFalse(hold.released)
                op.release()
                self.assertTrue(hold.released)

    def test_dense_revalidates_the_retained_source_before_any_copy(self):
        op, source, hold = self.operation_at(25, 'dense')
        hold._allocator.refs.clear()
        with self.assertRaises(RuntimeError):
            op.fill_bucket(self.arena, 0, 24, self.table)
        self.assertFalse(self.device.page_copies)

    def test_current_policy_still_runs_union_and_sparse_copies(self):
        op, hold = self.execute(25, 'current')
        self.assertTrue(self.device.index_copies)
        self.assertTrue(self.device.copies)
        self.assertFalse(self.device.page_copies)
        self.assertEqual(op._current_run().sparse.route, 'sparse_dma')
        op.release()

    def test_no_source_hold_is_not_a_dense_fallback(self):
        op, source, hold = self.operation_at(25, 'dense')
        op._current_run().sparse_hold = None
        with self.assertRaises(base.shd.SparseFallbackRequired):
            op.sparse_selection_eligible(25, 24)
        self.assertFalse(self.device.page_copies)

    def test_policy_latched_not_reconsulted_per_chunk(self):
        op, source, hold = self.operation_at(25, 'dense')
        self.device.native_qsa_policy = 'current'
        self.assertFalse(op.sparse_selection_eligible(25, 24))

    def test_trace_dense_fields_and_event_queries_only(self):
        op, hold = self.execute(25, 'dense', trace=True)
        with self.assertLogs(qo.__name__, level='INFO') as log:
            op.release()
        row = json.loads(log.output[0].split('flash-s5 native43_qsa ', 1)[1])
        self.assertEqual(row['route'], 'dense_dma')
        self.assertEqual(row['history'], 25 * 4048)
        self.assertIsNone(row['selected_count'])
        self.assertEqual(row['descriptors'], 7)
        self.assertEqual(row['waves'], 2)
        self.assertEqual(row['bytes'], 25 * 4048 * 1024)
        self.assertEqual(row['union_ms'], 0)
        self.assertTrue(row['gpu_timing_complete'])
        self.assertGreater(row['copy_ms'], 0)
        self.assertGreater(row['attention_ms'], 0)
        self.assertGreater(row['cpu_plan_ns'], 0)
        self.assertGreater(row['cpu_submit_ns'], 0)
        self.assertEqual(self.device.synchronized, 0)

    def test_trace_current_and_marker_limit(self):
        op, hold = self.execute(25, 'current', trace=True)
        op.trace.MAX_MARKERS = op.trace.markers
        self.assertIsNone(op.trace.begin('copy', self.device.copy))
        with self.assertLogs(qo.__name__, level='INFO') as log:
            op.release()
        row = json.loads(log.output[0].split('flash-s5 native43_qsa ', 1)[1])
        self.assertEqual(row['route'], 'sparse_dma')
        self.assertIsNotNone(row['selected_count'])
        self.assertGreater(row['union_ms'], 0)
        self.assertGreater(row['descriptors'], 0)
        self.assertFalse(row['gpu_timing_complete'])
        self.assertEqual(row['dropped_markers'], 1)


class BoundaryStopTests(unittest.TestCase):
    def test_stop_is_after_real_four_rank_release_and_certified_cancel(self):
        import os
        from test_e3_r4_worker_bridge import Stack
        with patch.dict(os.environ, FLASH_S5_NATIVE_QSA_STOP_AFTER='101200',
                        FLASH_S5_NATIVE_QSA_TRACE='1'):
            stack = Stack()
        # Geometry leaf only: shrink the frontier from 101200 to this fixture's
        # 4 rows; execute real controller/driver/bridges/dispatchers and events.
        self.assertIsNotNone(stack.driver._native43)
        stack.driver._native43['stop_at'] = 4
        for _ in range(600):
            stack.complete_everything()
            stack.advance()
            if stack.driver._last_native_closure is not None:
                break
        checkpoint = stack.driver._native43['checkpoint']
        self.assertEqual(checkpoint['chunk_end'], 4)
        self.assertTrue(checkpoint['exact'])
        self.assertTrue(stack.driver._last_native_closure['source_leases']['clean'])
        self.assertFalse(stack.driver._last_native_closure['pp_receipts_pending'])
        self.assertTrue(all(r['request_closed'] for r in stack.driver._last_native_closure['rows']))
        self.assertEqual({r['worker_rank'] for r in checkpoint['worker_releases']}, {0,1,2,3})
        self.assertTrue(all(r['released'] for r in checkpoint['worker_releases']))
        self.assertTrue(stack.driver._last_native_closure['complete'])
        self.assertTrue(stack.all_dispatchers_reset())
        self.assertIsNone(stack.driver._retained)
        for _ in range(400):
            stack.complete_everything()
            stack.advance()
            if not stack.driver._split_ready and stack.driver._transition is None:
                break
        self.assertFalse(stack.driver._split_ready)
        self.assertIsNone(stack.driver._transition)
        self.assertEqual(set(stack.driver._last_transition_acks), {0,1,2,3})
        self.assertTrue(all(a.applied_bank == 0 for a in stack.driver._last_transition_acks.values()))
        self.assertEqual(stack.controller.commits, 1)

    def test_real_planner_has_exactly_four_chunks_after_arena_to_cutoff(self):
        from scheduler_hooks import staged_chunk_plan
        from test_native37_robust import geometry_split
        chunks = staged_chunk_plan(0, 197269, 1024, 64, geometry_split(197270))
        self.assertTrue(chunks)
        end = 0
        after = []
        for rows in chunks:
            end += rows
            if 97152 < end <= 101200:
                after.append(end)
        self.assertEqual(after, [98176, 99200, 100224, 101200])

    def test_clock_proof_accepts_foreign_identity_and_rejects_outside_bracket(self):
        import native43_boundary_clock as clock
        packet = dict(observed_clock_ns=15,
                      clock=dict(basis='clock-monotonic', boot_id='other', time_namespace='other'))
        with patch.object(clock, 'view', return_value=dict(boot_id='client', time_namespace='client')):
            self.assertTrue(clock.prove(packet, 10, 20)['proved'])
            self.assertFalse(clock.prove(packet, 16, 20)['proved'])
            packet['clock']['basis'] = 'cuda-event'
            self.assertFalse(clock.prove(packet, 10, 20)['proved'])


if __name__ == '__main__':
    unittest.main()
