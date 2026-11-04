"""inwait build-3 (DESIGN-3): PREPARE in slices and the warm score block, worker side, and the engine's retry of a pending PREPARE."""
import types
import unittest
from unittest import mock

import test_e3_lane_chunk as tlc
import test_e3_lane_dispatch as ld
import test_inwait_engine as ie
import test_e3_stage_overlap as c1
import lane_chunk as lc
import lane_dispatch as dp
from e3_types import E3Pending
from wire_v1 import SegmentKind


class Reset:
    def setUp(self):
        super().setUp()
        saved = dict(lc.PREPARE_SLICES), dict(lc.SCORE_WARM)
        self.addCleanup(lambda: (lc.PREPARE_SLICES.update(saved[0]), lc.SCORE_WARM.update(saved[1])))
        lc.PREPARE_SLICES.update(selected=False, sliced=0, yields=0, peek_missing=0)
        lc.SCORE_WARM.update(selected=False, bytes=0, allocs=0, flushes=0, rewarms=0, errors=0, retries_seen=None)


class Dispatch(Reset, unittest.TestCase):
    def fake_torch(self):
        import contextlib
        return types.SimpleNamespace(inference_mode=contextlib.nullcontext,
                                     cuda=types.SimpleNamespace(stream=lambda _s: contextlib.nullcontext(), Event=lambda enable_timing=False: ld.FakeEvent()))

    def dispatcher(self, can=True):
        d, lane, channel, server, admission, operation = ld.make(torch=self.fake_torch())
        factory, log = d.factory, []
        chunk = factory.admit(admission)

        def steps(_admission):
            log.append('private-1')
            yield
            log.append('private-2')
            yield
            log.append('shared')
            return chunk
        factory.can_slice = lambda _a: can
        factory.admit_steps = steps
        factory.admit = lambda _a: log.append('whole') or chunk
        d.open(ld.TXN, admission, operation)
        d.ring_waiting = lambda: self.waiting
        self.waiting = True
        return d, log, chunk, server

    def test_off_runs_the_whole_admit(self):
        d, log, chunk, server = self.dispatcher()
        self.assertEqual(d.submit(ld.cmd('prepare', 0))['result']['prepared'], True)
        self.assertEqual(log, ['whole'])
        self.assertEqual(lc.PREPARE_SLICES['sliced'], 0)

    def test_first_chunk_is_not_sliced(self):
        lc.select_prepare_slices(True)
        d, log, chunk, server = self.dispatcher(can=False)
        d.submit(ld.cmd('prepare', 0))
        self.assertEqual(log, ['whole'])

    def test_two_pending_then_accepted(self):
        lc.select_prepare_slices(True)
        d, log, chunk, server = self.dispatcher()
        command = ld.cmd('prepare', 0)
        for want in (['private-1'], ['private-1', 'private-2']):
            with self.assertRaises(E3Pending):
                d.submit(command)
            self.assertEqual(log, want)
            self.assertIsNone(d.chunk)
            self.assertIsNone(d.quarantined)
            self.assertEqual(server.calls, [], 'nothing parked before the last slice')
            self.assertTrue(d.prepare_resuming(command))
        ack = d.submit(command)
        self.assertEqual(log, ['private-1', 'private-2', 'shared'])
        self.assertIs(d.chunk, chunk)
        self.assertEqual(ack['result']['prepared'], True)
        self.assertEqual((lc.PREPARE_SLICES['sliced'], lc.PREPARE_SLICES['yields']), (1, 2))
        self.assertFalse(d.prepare_resuming(command))
        self.assertEqual(d.submit(command), ack, 'a replay answers from the record')

    def test_nothing_waiting_needs_no_resend(self):
        lc.select_prepare_slices(True)
        d, log, chunk, server = self.dispatcher()
        self.waiting = False
        d.submit(ld.cmd('prepare', 0))
        self.assertEqual(log, ['private-1', 'private-2', 'shared'])
        self.assertEqual((lc.PREPARE_SLICES['sliced'], lc.PREPARE_SLICES['yields']), (1, 0))

    def test_a_different_command_is_refused(self):
        lc.select_prepare_slices(True)
        d, log, chunk, server = self.dispatcher()
        with self.assertRaises(E3Pending):
            d.submit(ld.cmd('prepare', 0))
        self.assertFalse(d.prepare_resuming(ld.cmd('prepare', 1)))
        with self.assertRaises(dp.LaneDispatchError):
            d.submit(ld.cmd('prepare', 1))
        self.assertIsNone(d.quarantined, 'only private state had moved')

    def test_stop_between_slices_closes_as_before_prepare(self):
        lc.select_prepare_slices(True)
        for boundaries in (1, 2):
            d, log, chunk, server = self.dispatcher()
            for _ in range(boundaries):
                with self.assertRaises(E3Pending):
                    d.submit(ld.cmd('prepare', 0))
            d.cancel('stop')
            self.assertIsNone(d.quarantined)
            self.assertTrue(d.drain())
            self.assertTrue(d.release())
            self.assertEqual(server.calls, [], 'never parked, never aborted')
            self.assertEqual(d.factory.calls, [], 'the metadata gate was never taken')
            self.assertIsNone(d._slices)

    def test_a_failure_in_the_last_slice_quarantines(self):
        lc.select_prepare_slices(True)
        d, log, chunk, server = self.dispatcher()
        self.waiting = False

        def steps(_admission):
            yield
            yield
            raise lc.LaneChunkError('shared state failed')
            yield  # noqa: unreachable, keeps it a generator
        d.factory.admit_steps = steps
        with self.assertRaises(dp.LaneDispatchError):
            d.submit(ld.cmd('prepare', 0))
        self.assertIsNotNone(d.quarantined)

    def test_no_ring_says_waiting_and_counts(self):
        d = ld.make(torch=self.fake_torch())[0]
        with mock.patch.dict(dp._RING, clear=True):
            self.assertTrue(d.ring_waiting())
        self.assertEqual(lc.PREPARE_SLICES['peek_missing'], 1)


class Peek(Reset, unittest.TestCase):
    def ring(self, written, read):
        import contextlib
        meta = [written, read]

        @contextlib.contextmanager
        def get_metadata(index):
            self.assertEqual(index, 3)
            yield meta
        return types.SimpleNamespace(buffer=types.SimpleNamespace(get_metadata=get_metadata), current_idx=3, local_reader_rank=0)

    def test_written_and_unread(self):
        self.assertTrue(dp.ring_message_waiting(self.ring(1, 0)))
        self.assertFalse(dp.ring_message_waiting(self.ring(0, 0)))
        self.assertFalse(dp.ring_message_waiting(self.ring(1, 1)))

    def test_the_ring_comes_from_the_running_rpc_frame(self):
        mq = self.ring(1, 0)

        class WorkerProc:
            rpc_broadcast_mq = mq

            def _execute_worker_rpc(self, fn):
                return fn()
        with mock.patch.dict(dp._RING, clear=True):
            self.assertIs(WorkerProc()._execute_worker_rpc(dp._worker_ring), mq)
            self.assertIs(dp._worker_ring(), mq, 'kept')


class Warm(Reset, unittest.TestCase):
    def factory(self, retries=lambda: 0):
        torch = mock.Mock()
        torch.uint8 = 'uint8'
        torch.cuda.memory_stats = lambda device: dict(num_alloc_retries=retries())
        factory = lc.LaneChunkFactory.__new__(lc.LaneChunkFactory)
        factory.torch, factory.runner = torch, types.SimpleNamespace(device='cuda:0')
        return factory, torch

    def admission(self, start=1024, prompt=65444):
        return types.SimpleNamespace(chunk_start=start, prompt_end=prompt)

    def select(self):
        with mock.patch.object(lc, '_flush_allocator') as flush:
            lc.select_score_warm(True)
            lc.select_score_warm(True)
            self.assertEqual(flush.call_count, 1, 'only off to on')

    def test_size(self):
        # 1,024 rows x width 16,384 x 4 = 64 MiB, plus 16 MiB slack, on the 32 MiB step.
        self.assertEqual(lc.score_warm_bytes(65444), 96 * 1024 * 1024)
        self.assertEqual(lc.score_warm_bytes(131328), 160 * 1024 * 1024)

    def test_off_does_nothing(self):
        factory, torch = self.factory()
        self.assertFalse(factory.warm_scores(self.admission()))
        torch.empty.assert_not_called()

    def test_not_on_the_first_chunk(self):
        self.select()
        factory, torch = self.factory()
        self.assertFalse(factory.warm_scores(self.admission(start=0)))
        torch.empty.assert_not_called()

    def test_once_per_size_and_again_after_a_retry(self):
        self.select()
        count = [0]
        factory, torch = self.factory(lambda: count[0])
        factory.warm_scores(self.admission())
        factory.warm_scores(self.admission())
        self.assertEqual(torch.empty.call_count, 1)
        factory.warm_scores(self.admission(prompt=131328))
        self.assertEqual(torch.empty.call_count, 2, 'a longer prompt')
        factory.warm_scores(self.admission())
        self.assertEqual(torch.empty.call_count, 2, 'a shorter one fits')
        count[0] = 1
        factory.warm_scores(self.admission())
        self.assertEqual(torch.empty.call_count, 3, 'the retry emptied the cache')
        self.assertEqual((lc.SCORE_WARM['allocs'], lc.SCORE_WARM['rewarms']), (3, 1))

    def test_a_failure_is_counted_never_raised(self):
        self.select()
        factory, torch = self.factory()
        torch.empty.side_effect = RuntimeError('oom')
        self.assertFalse(factory.warm_scores(self.admission()))
        self.assertEqual(lc.SCORE_WARM['errors'], 1)


class Factory(Reset, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        self.factory = self.make(self.calls)

    def make(self, calls):
        torch = tlc.fake_torch()
        return lc.LaneChunkFactory(tlc.fake_runner(torch, calls), ports=tlc.fake_ports(calls), torch_module=torch)

    def later(self, factory, calls):
        factory.complete_chunk(factory.admit(tlc.admission()), consumers_complete=True)
        calls.clear()
        return tlc.admission(chunk_start=8, chunk_end=11, txn_seq=2, block_tables=((5, 6, 7, 8), (9, 10, 11), ()), new_block_ids_to_zero=(6,))

    def test_can_slice_is_later_chunks_of_the_row_only(self):
        self.assertFalse(self.factory.can_slice(tlc.admission()), 'nothing registered')
        next_chunk = self.later(self.factory, self.calls)
        self.assertTrue(self.factory.can_slice(next_chunk))
        self.assertFalse(self.factory.can_slice(tlc.admission(chunk_start=8, chunk_end=11, txn_seq=2, request_uid='req-b')))
        self.assertFalse(self.factory.can_slice(tlc.admission(chunk_start=8, chunk_end=11, txn_seq=2, request_generation=1)))
        self.factory.admit(next_chunk)
        self.assertFalse(self.factory.can_slice(next_chunk), 'the metadata gate is held')

    def test_nothing_shared_moves_before_the_last_slice(self):
        steps = self.factory.admit_steps(self.later(self.factory, self.calls))
        for _ in range(2):
            next(steps)
            self.assertEqual([c for c in self.calls if c[0] not in ('build',)], [], 'only the private metadata build')
            self.assertIsNone(self.factory._active_metadata_chunk)
        self.assertEqual([c[0] for c in self.calls], ['build'])
        with self.assertRaises(StopIteration) as done:
            next(steps)
        tags = [c[0] for c in self.calls]
        self.assertLess(tags.index('build'), tags.index('append'))
        self.assertLess(tags.index('append'), tags.index('align-pre'), 'the row before the pre-copy, as in admit')
        self.assertIs(self.factory._active_metadata_chunk, done.exception.value)

    def test_the_chunk_is_the_one_admit_builds(self):
        other_calls = []
        other = self.make(other_calls)
        whole = other.admit(self.later(other, other_calls))
        steps = self.factory.admit_steps(self.later(self.factory, self.calls))
        next(steps), next(steps)
        with self.assertRaises(StopIteration) as done:
            next(steps)
        sliced = done.exception.value
        self.assertEqual(sorted(self.calls, key=repr), sorted(other_calls, key=repr), 'the same calls, reordered only')
        self.assertEqual((sliced.req_index, sliced.qsa_host_plan, sliced.native_metadata_generation, sliced.forward_context[2:]),
                         (whole.req_index, whole.qsa_host_plan, whole.native_metadata_generation, whole.forward_context[2:]))
        self.assertEqual((sliced.input_ids.values, sliced.positions.values, sliced.batch.num_tokens),
                         (whole.input_ids.values, whole.positions.values, whole.batch.num_tokens))
        self.assertEqual((self.factory.chunks, self.factory.request.chunks), (other.chunks, other.request.chunks))
        self.assertEqual(self.factory.native_attn_groups[0][0].get_metadata_builder(0).mamba_aligned_state_indices,
                         other.native_attn_groups[0][0].get_metadata_builder(0).mamba_aligned_state_indices)

    def test_abandoned_between_slices_the_row_still_closes(self):
        steps = self.factory.admit_steps(self.later(self.factory, self.calls))
        next(steps)
        steps.close()
        self.assertIsNone(self.factory._active_metadata_chunk)
        self.assertEqual(self.factory.release(), 'req-a')


class SlicedPrepare:
    """The composed engine with every later PREPARE answering pending twice, per rank, as the worker does at its boundaries."""
    PENDS = 2

    def network(self, **kwargs):
        network = super().network(**kwargs)
        self.pends = {}
        for rank, worker in network.workers.items():
            dispatcher = worker.dispatcher
            real = dispatcher.submit

            def submit(command, real=real, rank=rank):
                key = (rank, command.transaction, command.op_seq)
                if command.kind == 'prepare' and command.transaction.chunk_start > 0 and self.pends.get(key, 0) < self.PENDS:
                    self.pends[key] = self.pends.get(key, 0) + 1
                    raise E3Pending('test: prepare slice')
                return real(command)
            dispatcher.submit = submit
        return network


class StopSweepSliced(SlicedPrepare, ie.Selected, c1.StopSweep):
    pass


class StopSweepSlicedEarly(SlicedPrepare, ie.Early, c1.StopSweep):
    pass


class StopSweepOnePeer(StopSweepSlicedEarly):
    """One TP peer takes PREPARE whole while the other is in slices: the peers' peeks need not agree."""

    def network(self, **kwargs):
        network = super().network(**kwargs)
        for rank in (1, 3):
            worker = network.workers[rank]
            real = worker.dispatcher.submit
            worker.dispatcher.submit = lambda command, real=real, keep=self.pends, rank=rank: (keep.setdefault((rank, command.transaction, command.op_seq), 99), real(command))[1]
        return network


class Engine(SlicedPrepare, ie.Base):
    def test_a_request_finishes_with_the_retries_and_each_prepare_accepted_once(self):
        network = self.network(direct=True)
        network.run_request()
        self.assert_closed(network)
        self.assertTrue(self.pends, 'later chunks answered pending')
        self.assertEqual(set(self.pends.values()), {2})
        sent = [c1.stage_of_command(env)[1] for env in network.sent_envelopes if env[0] == 'flash_s5_submit']
        prepares = [m for m in sent if m.kind is SegmentKind.PREPARE and m.transaction.chunk_start > 0]
        self.assertTrue(prepares)
        for command in prepares:
            same = [m for m in prepares if m.transaction == command.transaction and m.op_seq == command.op_seq and m.stage == command.stage]
            self.assertEqual(len(same), 3, 'the first send and two resends, same command')

    def test_in_the_wait_the_resends_go_from_the_wait_too(self):
        network = self.run_in_wait()
        self.same_as_serial(network)
        self.assert_closed(network)
        self.assertEqual(set(self.pends.values()), {2})
        sent = [c1.stage_of_command(env)[1] for env in network.sent_envelopes if env[0] == 'flash_s5_submit']
        distinct = {(m.transaction, m.stage) for m in sent if m.kind is SegmentKind.PREPARE and m.transaction.chunk_start > 0}
        self.assertGreater(network.driver.inwait_counters['wait_sent']['prepare'], len(distinct), 'more wait sends than commands: resends')


if __name__ == '__main__':
    unittest.main()
