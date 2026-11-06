"""coverage-throughput build-2 (a), prepare-ahead, worker side: a second private builder set, a later chunk's slices 1-2 staged
early, and the real PREPARE running slice 3 alone. Hub, fake torch (test_e3_lane_chunk's rig), no GPU.

What this proves: off is today's factory (one set, nothing staged, admit_steps unchanged); staging moves nothing shared (no
request row, pages, gate or pre-copy) and builds into the set the live chunk does not hold; an adopted chunk is the one the
three slices build, reached with slice 3's calls only; any mismatch drops the staged chunk and the slices run as today; the
dispatcher stages only once the live chunk's state_finalize is accepted, under the lane stream, and an adopted PREPARE is
accepted at once even with a message waiting in the ring.
What it does not: device ordering itself (the fake torch has no streams), the engine side that sends the admission early.
"""
import types
import unittest

import test_e3_lane_chunk as tlc
import test_e3_lane_dispatch as ld
import test_inwait_slices as sl
import lane_chunk as lc

SHARED = ('append', 'align-pre', 'register', 'update', 'zero', 'cow')


class Reset(sl.Reset):
    def setUp(self):
        super().setUp()
        saved = {k: (dict(v) if isinstance(v, dict) else v) for k, v in lc.PREPARE_AHEAD.items()}
        self.addCleanup(lambda: lc.PREPARE_AHEAD.update(saved))
        lc.PREPARE_AHEAD.update(selected=False, staged=0, adopted=0, dropped=0, refused={}, errors=0)


def later(chunk_start=8, chunk_end=11, txn_seq=2, **kw):
    kw.setdefault('block_tables', ((5, 6, 7, 8), (9, 10, 11), ()))
    kw.setdefault('new_block_ids_to_zero', (6,))
    return tlc.admission(chunk_start=chunk_start, chunk_end=chunk_end, txn_seq=txn_seq, **kw)


class Factory(Reset, unittest.TestCase):
    def make(self, calls, ahead):
        torch = tlc.fake_torch()
        return lc.LaneChunkFactory(tlc.fake_runner(torch, calls), ports=tlc.fake_ports(calls), torch_module=torch,
                                   prepare_ahead=ahead)

    def live_first(self, factory):
        """The first chunk admitted and still live (its kernels would be reading set 0)."""
        return factory.admit(tlc.admission())

    def test_off_is_one_set_and_stages_nothing(self):
        calls = []
        f = self.make(calls, False)
        self.assertEqual((f.native_builder_sets, len(f.builder_sets)), (1, 1))
        first = self.live_first(f)
        self.assertEqual(first.builder_set, 0)
        self.assertEqual(f.stage_ahead(later()), dict(staged=False, why='not-selected'))
        f.complete_chunk(first, consumers_complete=True)
        self.assertIsNone(f.adopt(later()))
        self.assertEqual((lc.PREPARE_AHEAD['staged'], lc.PREPARE_AHEAD['adopted'], lc.PREPARE_AHEAD['dropped']), (0, 0, 0))
        steps = f.admit_steps(later())
        next(steps), next(steps)
        with self.assertRaises(StopIteration) as done:
            next(steps)
        self.assertEqual(done.exception.value.builder_set, 0, 'one set: every chunk on it')

    def test_prepare_ahead_must_be_a_bool(self):
        with self.assertRaises(lc.LaneChunkError):
            self.make([], 1)

    def test_on_builds_a_second_private_set(self):
        f = self.make([], True)
        self.assertEqual(f.native_builder_sets, 2)
        a, b = f.builder_sets
        self.assertIs(a, f.native_attn_groups)
        ba, bb = a[0][0].get_metadata_builder(0), b[0][0].get_metadata_builder(0)
        self.assertIsNot(ba, bb)
        for name, value in vars(ba).items():
            other = vars(bb)[name]
            if hasattr(value, 'values') or isinstance(value, list):
                self.assertIsNot(value, other, 'mutable builder field %s is shared between the sets' % name)

    def test_staging_moves_nothing_shared_and_uses_the_other_set(self):
        calls = []
        f = self.make(calls, True)
        first = self.live_first(f)
        calls.clear()
        self.assertEqual(f.stage_ahead(later()), dict(staged=True, why='staged'))
        self.assertEqual([c[0] for c in calls if c[0] in SHARED], [], 'staging touched shared state: %r' % calls)
        self.assertIs(f._active_metadata_chunk, first, 'the live chunk still holds the gate')
        self.assertEqual(f._staged[1].builder_set, 1)
        self.assertEqual(f.stage_ahead(later()), dict(staged=True, why='already'))
        self.assertEqual(lc.PREPARE_AHEAD['staged'], 1)

    def test_adopted_chunk_equals_the_sliced_one_with_slice_3_calls_only(self):
        calls, other_calls = [], []
        f, g = self.make(calls, True), self.make(other_calls, True)
        for factory in (f, g):
            factory.complete_chunk(self.live_first(factory), consumers_complete=True)
        # f stages while nothing is live, g runs today's three slices
        f.stage_ahead(later())
        calls.clear(), other_calls.clear()
        adopted = f.adopt(later())
        steps = g.admit_steps(later())
        next(steps), next(steps)
        slice3_start = len(other_calls)
        with self.assertRaises(StopIteration) as done:
            next(steps)
        sliced = done.exception.value
        self.assertIsNotNone(adopted)
        self.assertEqual(sorted(calls, key=repr), sorted(other_calls[slice3_start:], key=repr), 'adopt runs slice 3 only')
        self.assertEqual((adopted.req_index, adopted.qsa_host_plan, adopted.native_metadata_generation, adopted.forward_context[2:]),
                         (sliced.req_index, sliced.qsa_host_plan, sliced.native_metadata_generation, sliced.forward_context[2:]))
        self.assertEqual((adopted.input_ids.values, adopted.positions.values, adopted.batch.num_tokens),
                         (sliced.input_ids.values, sliced.positions.values, sliced.batch.num_tokens))
        self.assertEqual((adopted.builder_set, sliced.builder_set), (1, 1))
        self.assertIs(f._active_metadata_chunk, adopted)
        self.assertEqual((f.chunks, f.request.chunks), (g.chunks, g.request.chunks))
        self.assertEqual(lc.PREPARE_AHEAD['adopted'], 1)

    def test_a_different_admission_drops_and_the_slices_run(self):
        calls = []
        f = self.make(calls, True)
        f.complete_chunk(self.live_first(f), consumers_complete=True)
        f.stage_ahead(later())
        self.assertIsNone(f.adopt(later(chunk_end=10)))
        self.assertIsNone(f._staged)
        self.assertEqual(lc.PREPARE_AHEAD['dropped'], 1)
        steps = f.admit_steps(later(chunk_end=10))
        next(steps), next(steps)
        with self.assertRaises(StopIteration) as done:
            next(steps)
        self.assertEqual(done.exception.value.admission.chunk_end, 10)

    def test_adopt_waits_for_the_gate(self):
        f = self.make([], True)
        first = self.live_first(f)
        f.stage_ahead(later())
        self.assertIsNone(f.adopt(later()), 'the live chunk still holds the gate: not adopted')
        self.assertEqual(lc.PREPARE_AHEAD['dropped'], 1)
        f.complete_chunk(first, consumers_complete=True)

    def test_refusals(self):
        f = self.make([], True)
        self.assertEqual(f.stage_ahead(later())['why'], 'not-a-later-chunk', 'nothing registered')
        self.live_first(f)
        self.assertEqual(f.stage_ahead(tlc.admission())['why'], 'not-a-later-chunk', 'a first chunk')
        self.assertEqual(f.stage_ahead(later(request_uid='req-b'))['why'], 'not-a-later-chunk')
        self.assertEqual(f.stage_ahead(later(request_generation=1))['why'], 'not-a-later-chunk')
        self.assertEqual(f.stage_ahead('x')['why'], 'not-an-admission')
        self.assertEqual(lc.PREPARE_AHEAD['refused'], {'not-a-later-chunk': 4, 'not-an-admission': 1})

    def test_sets_alternate_across_adopted_and_sliced_chunks(self):
        f = self.make([], True)
        seen = []
        chunk = self.live_first(f)
        seen.append(chunk.builder_set)
        for i, ahead in enumerate((True, False, True)):
            adm = later(chunk_start=8 + i, chunk_end=9 + i, txn_seq=2 + i, new_block_ids_to_zero=())
            if ahead:
                self.assertTrue(f.stage_ahead(adm)['staged'])
            f.complete_chunk(chunk, consumers_complete=True)
            chunk = f.adopt(adm)
            if chunk is None:
                steps = f.admit_steps(adm)
                next(steps), next(steps)
                with self.assertRaises(StopIteration) as done:
                    next(steps)
                chunk = done.exception.value
            seen.append(chunk.builder_set)
        self.assertEqual(seen, [0, 1, 0, 1])

    def test_a_failure_stages_nothing_and_never_raises(self):
        f = self.make([], True)
        self.live_first(f)
        original = lc.LaneChunk.build_metadata
        self.addCleanup(setattr, lc.LaneChunk, 'build_metadata', original)
        lc.LaneChunk.build_metadata = lambda self, precopy=True: (_ for _ in ()).throw(RuntimeError('boom'))
        self.assertEqual(f.stage_ahead(later()), dict(staged=False, why='error'))
        self.assertIsNone(f._staged)
        self.assertEqual(lc.PREPARE_AHEAD['errors'], 1)

    def test_release_drops_the_staged_chunk(self):
        f = self.make([], True)
        f.complete_chunk(self.live_first(f), consumers_complete=True)
        f.stage_ahead(later())
        self.assertEqual(f.release(), 'req-a')
        self.assertIsNone(f._staged)
        self.assertEqual(lc.PREPARE_AHEAD['dropped'], 1)


class Dispatch(Reset, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.d, _lane, _ch, _server, self.admission, self.operation = ld.make(torch=sl.Dispatch.fake_torch(self))
        self.staged, self.adopted = [], []
        f = self.d.factory
        f.stage_ahead = lambda adm: self.staged.append(adm) or dict(staged=True, why='staged')
        f._refuse_ahead = lambda why: dict(staged=False, why=why)
        f.can_slice = lambda _a: True
        f.admit_steps = lambda _a: (_ for _ in ()).throw(AssertionError('sliced path must not run on an adopted PREPARE'))

    def test_no_transaction_open_stages(self):
        self.assertEqual(self.d.stage_ahead('next')['why'], 'staged')
        self.assertEqual(self.staged, ['next'])

    def test_live_chunk_must_be_finalized_first(self):
        self.d.open(ld.TXN, self.admission, self.operation)
        self.assertEqual(self.d.stage_ahead('next')['why'], 'live-chunk-not-finalized')
        self.d.accepted[9] = (types.SimpleNamespace(kind='state_finalize'), {})
        self.assertEqual(self.d.stage_ahead('next')['why'], 'staged')
        self.assertEqual(self.staged, ['next'])

    def test_staging_runs_on_the_lane_prefill_stream(self):
        import contextlib
        inside = []

        @contextlib.contextmanager
        def stream(s):
            inside.append(s)
            try:
                yield
            finally:
                inside.pop()
        self.d.torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext,
                                             cuda=types.SimpleNamespace(stream=stream, Event=lambda enable_timing=False: ld.FakeEvent()))
        seen = []
        self.d.factory.stage_ahead = lambda adm: seen.append(list(inside)) or dict(staged=True, why='staged')
        self.d.stage_ahead('next')
        self.assertEqual(seen, [[self.d.native.split.prefill_stream]])

    def test_quarantined_refuses(self):
        self.d.quarantined = 'x'
        self.assertEqual(self.d.stage_ahead('next')['why'], 'quarantined')

    def test_an_adopted_prepare_is_accepted_at_once_with_the_ring_waiting(self):
        lc.select_prepare_slices(True)
        chunk = self.d.factory.admit(self.admission)
        self.d.factory.adopt = lambda adm: self.adopted.append(adm) or chunk
        self.d.factory.admit = lambda _a: (_ for _ in ()).throw(AssertionError('whole admit must not run'))
        self.d.ring_waiting = lambda: True
        self.d.open(ld.TXN, self.admission, self.operation)
        self.assertEqual(self.d.submit(ld.cmd('prepare', 0))['result']['prepared'], True)
        self.assertEqual(self.adopted, [self.admission])
        self.assertEqual(lc.PREPARE_SLICES['yields'], 0)

    def test_no_staged_chunk_runs_todays_path(self):
        chunk = self.d.factory.admit(self.admission)
        self.d.factory.adopt = lambda adm: None
        whole = []
        self.d.factory.admit = lambda a: whole.append(a) or chunk
        self.d.open(ld.TXN, self.admission, self.operation)
        self.assertEqual(self.d.submit(ld.cmd('prepare', 0))['result']['prepared'], True)
        self.assertEqual(whole, [self.admission])


if __name__ == '__main__':
    unittest.main()
