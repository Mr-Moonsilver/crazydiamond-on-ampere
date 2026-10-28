"""Bounded copy waves: real planner/operation, simulated device boundary only."""
import dataclasses
import struct
import unittest
import test_e3_sparse_history_dma as f

shd, qo, et = f.shd, f.qo, f.et


class PlannerTests(unittest.TestCase):
    def test_strict_cap_unchanged_streaming_preserves_exact_rows(self):
        mapping = tuple(reversed(range(128)))
        versions = (1,) * 128
        for count in (256, 257, 512, 97152, 97153, 200000):
            selected = tuple(range(0, count * 2, 2))
            strict = shd.make_sparse_plan(selected, mapping, versions, 1)
            self.assertEqual(strict.refused, count > 256)
            plan = shd.stream_sparse_plan(strict) if strict.refused else strict
            self.assertEqual(plan.selected, selected)
            self.assertLessEqual(plan.descriptor_high_water, 256)
            actual = []
            for bucket in plan.buckets:
                waves = bucket.waves() if isinstance(bucket, shd.SparseStreamBucket) else (bucket.pieces,)
                cursor = 0
                for wave in waves:
                    self.assertLessEqual(len(wave), 256)
                    for p in wave:
                        self.assertEqual(p.arena_row, cursor)
                        cursor += p.rows
                        actual.extend(range(p.source_row, p.source_row + p.rows))
                self.assertEqual(cursor, bucket.rows)
            expected = [mapping[t // shd.PAGE_ROWS] * shd.PAGE_ROWS + t % shd.PAGE_ROWS for t in selected]
            self.assertEqual(actual, expected)
            self.assertEqual(plan.bucket_rows, tuple(min(shd.ARENA_ROWS, count-lo)
                             for lo in range(0, count, shd.ARENA_ROWS)))

    def test_unsupported_mapping_never_streams(self):
        plan = shd.make_sparse_plan((0,), (0,), (1,), 1, null_page=0)
        with self.assertRaises(shd.SparsePlanError):
            shd.stream_sparse_plan(plan)

    def test_identity_validation_retained(self):
        plan = shd.stream_sparse_plan(shd.make_sparse_plan(tuple(range(0, 600, 2)), (1,), (1, 1), 1))
        for args in (((0,), (1, 1), 1), ((1,), (1, 2), 1), ((1,), (1, 1), 2)):
            with self.assertRaises(shd.SourceRefused):
                shd.validate_plan_source(plan, *args)


class OperationTests(f._GeneratedHarness, unittest.TestCase):
    make_operation = f.GeneratedSparseTests.make_operation
    hold_for = f.GeneratedSparseTests.hold_for
    bind = f.GeneratedSparseTests.bind
    prepare = f.GeneratedSparseTests.prepare
    open_layer = f.GeneratedSparseTests.open_layer

    def start(self, count=1025):
        mapping = tuple(range(1, 26))  # real row1 crosses the 24-page threshold
        op, lane, tx, sources = self.prepare({'qsa.0': mapping})
        hold = self.open_layer(op, lane, tx, 'qsa.0', sources['qsa.0'], mapping, min_pages=24)
        selected = list(range(0, count*2, 2))
        logical = f.T((1, count+1), dtype='torch.int32', values=selected+[-1])
        query = f.T((1, f.QUERY_HEADS, f.HEAD_DIM))
        output = f.T(query.shape)
        rows = f.T((1,), dtype='torch.int64', values=[0])
        op.attach_continuation(self.attention(query, sources['qsa.0'], logical,
            self.table, rows, 24, self.arena, True, output, e3op=op))
        return op, hold

    def test_credit_waits_for_actual_wave_query(self):
        op, hold = self.start()
        # The staged selection may complete; the first copy event remains pending.
        marker = op.advance()
        self.assertTrue(marker.sparse_copy_pending)
        op._layers['qsa.0'].sparse.wave_event.ready = False
        before = (len(self.device.copies), len(self.device.events), len(op._records))
        for _ in range(8):
            self.assertFalse(op.submission_credit().can_submit)
            with self.assertRaises(et.E3Pending):
                op.advance()
        self.assertEqual(before, (len(self.device.copies), len(self.device.events), len(op._records)))
        self.assertEqual(hold.stats['published'], 0)
        self.assertEqual(len(self.kernel.calls), 0)
        self.device.ready_all()
        self.assertTrue(op.submission_credit().can_submit)
        op.advance()
        self.assertEqual(len(self.device.copies), 512)

    def test_cancel_unpublished_prefix_retains_until_query(self):
        op, hold = self.start()
        op.advance()
        event = op._layers['qsa.0'].sparse.wave_event
        event.ready = False
        op.cancel('between-copy-waves')
        with self.assertRaises(qo.OperationCancelled):
            op.advance()
        self.assertFalse(op.drain())
        with self.assertRaises(et.E3Pending):
            op.release()
        self.assertEqual(hold.stats['published'], 0)
        self.assertFalse(hold.releasable)
        event.ready = True
        self.assertTrue(op.drain())
        self.assertEqual(hold.stats['abandoned'], 1)
        self.assertEqual(hold.stats['consumed'], 0)
        self.assertEqual(len(self.device.copies), 256)
        self.assertEqual(len(self.kernel.calls), 0)
        op.release()
        self.assertTrue(hold._released)

    def test_cancel_after_query_before_next_submission(self):
        op, hold = self.start()
        op.advance()
        self.assertTrue(op.submission_credit().can_submit)
        op.cancel('after-query')
        self.assertTrue(op.drain())
        op.release()
        self.assertEqual(hold.stats['abandoned'], 1)
        self.assertEqual(len(self.device.copies), 256)

    def test_partial_publish_and_legacy_cap_still_refused(self):
        op, hold = self.start()
        op.advance()
        with self.assertRaises(shd.SourceRefused):
            hold.copy_recorded(0)
        with self.assertRaises(shd.SourceRefused):
            hold.publish(0)
        with self.assertRaises(shd.SourceRefused):
            hold.abandon(0)
        with self.assertRaises(shd.SourceRefused):
            hold.begin_bucket(1, 300, 300, 300*shd.ROW_BYTES)
        with self.assertRaises(shd.SourceRefused):
            hold.stream_wave_recorded(0, 257, 257)

    def test_original_attention_boundaries_and_ticket_bound(self):
        op, hold = self.start(1025)
        # Exactly four tickets remain sufficient, independent of wave count.
        op.max_tickets = 4
        waves = []
        while True:
            n = len(self.device.copies)
            try:
                value = op.advance()
            except StopIteration:
                break
            waves.append(len(self.device.copies)-n)
            if getattr(value, 'sparse_copy_pending', False):
                self.assertEqual(hold.stats['published'], 0)
                self.assertEqual(len(self.kernel.calls), 0)
        self.assertEqual(waves, [256, 256, 256, 256, 1])
        self.assertEqual(len(self.kernel.calls), 1)
        self.assertEqual(len(op._records), 4)
        _, args, kwargs = self.kernel.calls[0]
        self.assertTrue(kwargs['FIRST_BUCKET'])
        self.assertTrue(kwargs['LAST_BUCKET'])
        self.assertEqual(self.arena.storage.rows[:1025],
                         [struct.pack('<Q', shd.PAGE_ROWS+i*2)*4 for i in range(1025)])
        op.end_layer('qsa.0'); op.end_forward()
        self.assertTrue(op.drain()); op.release()
        self.assertEqual(hold.stats['buckets'], 1)
        self.assertEqual(hold.stats['consumed'], 1)

    def test_event_query_failure_keeps_ownership_quarantined(self):
        op, hold = self.start()
        op.advance()
        event = op._layers['qsa.0'].sparse.wave_event
        def broken():
            raise RuntimeError('query failed')
        event.query = broken  # Device-boundary failure injection, not runtime patch.
        with self.assertRaises(RuntimeError):
            op.poll()
        self.assertFalse(op.drain())
        with self.assertRaises(et.E3Pending):
            op.release()
        self.assertFalse(hold._released)


if __name__ == '__main__':
    unittest.main()
