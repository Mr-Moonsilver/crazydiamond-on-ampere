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



def record(chunk_start=8, chunk_end=11, seq=2):
    import wire_v1 as w
    from e3_types import TransactionId
    return w.AdmissionRecord(
        transaction=TransactionId(1, 'req-a', 0, chunk_start, chunk_end, seq), prompt_end=20, state_epoch=9, committed_end=chunk_start,
        block_tables=((5, 6, 7), (9, 10), (), (0, 0)), slot_mappings=((0, 1, 2), (3, 4), (), ()),
        new_block_ids_to_zero=(), cow_pairs=(), input_ids=tuple(range(chunk_end - chunk_start)), ngram_context=(70, 71, 72))


class Flag(unittest.TestCase):
    def test_worker_reader(self):
        self.assertFalse(lc.prepare_ahead_from_env({}))
        self.assertFalse(lc.prepare_ahead_from_env({'FLASH_S5_PREPARE_AHEAD': '0'}))
        self.assertTrue(lc.prepare_ahead_from_env({'FLASH_S5_PREPARE_AHEAD': '1'}))
        with self.assertRaises(lc.LaneChunkError):
            lc.prepare_ahead_from_env({'FLASH_S5_PREPARE_AHEAD': 'yes'})

    def test_engine_needs_the_overlap(self):
        import stage_overlap as so
        from engine_driver import DriverError
        on = {'FLASH_S5_PREPARE_AHEAD': '1', 'FLASH_S5_STAGE_LOCAL': '1', 'FLASH_S5_STAGE_OVERLAP': '1'}
        self.assertTrue(so.prepare_ahead_enabled(on))
        self.assertFalse(so.prepare_ahead_enabled({}))
        with self.assertRaises(DriverError):
            so.prepare_ahead_enabled({'FLASH_S5_PREPARE_AHEAD': '1', 'FLASH_S5_STAGE_LOCAL': '1'})
        with self.assertRaises(DriverError):
            so.prepare_ahead_enabled(dict(on, FLASH_S5_PREPARE_AHEAD='2'))


class Scoped(Reset, unittest.TestCase):
    def test_ahead_defaults_to_none_and_the_repr_is_unchanged(self):
        import stage_wire as sw
        plain = sw.StageScoped('m', (0, 1))
        self.assertIsNone(plain.ahead)
        self.assertEqual(repr(plain), 'StageScoped(str, ranks=(0, 1), local_release=False)')
        self.assertIn(', ahead', repr(sw.StageScoped('m', (0, 1), ahead=record())))

    def test_ahead_must_be_an_admission_record(self):
        import stage_wire as sw
        from e3_types import E3ContractError
        with self.assertRaises(E3ContractError):
            sw.StageScoped('m', (0,), ahead=lc)

    def bridge(self, rank=0, raises=False):
        order = []

        def stage(admission):
            order.append(('stage', admission.chunk_start))
            if raises:
                raise RuntimeError('boom')
            return dict(staged=True)
        return types.SimpleNamespace(rank=rank, dispatcher=types.SimpleNamespace(stage_ahead=stage)), order

    def test_staged_after_the_answer_and_the_answer_kept(self):
        import stage_wire as sw
        b, order = self.bridge()
        answer = sw.stage_rpc(b, 'submit', sw.StageScoped('cmd', (0,), ahead=record()), lambda m: order.append(('answer', m)) or 'ack')
        self.assertEqual(answer, 'ack')
        self.assertEqual(order, [('answer', 'cmd'), ('stage', 8)])

    def test_no_ahead_stages_nothing_and_a_non_target_neither(self):
        import stage_wire as sw
        b, order = self.bridge()
        sw.stage_rpc(b, 'submit', sw.StageScoped('cmd', (0,)), lambda m: 'ack')
        self.assertIs(sw.stage_rpc(b, 'submit', sw.StageScoped('cmd', (1,), ahead=record()), lambda m: 'ack'), sw.STAGE_SKIP)
        self.assertEqual(order, [])

    def test_a_staging_failure_never_reaches_the_command(self):
        import stage_wire as sw
        b, order = self.bridge(raises=True)
        self.assertEqual(sw.stage_rpc(b, 'submit', sw.StageScoped('cmd', (0,), ahead=record()), lambda m: 'ack'), 'ack')
        self.assertEqual(lc.PREPARE_AHEAD['errors'], 1)

    def test_the_worker_stages_the_converted_admission(self):
        import stage_wire as sw
        from worker_bridge import admission_to_chunk
        seen = []
        b = types.SimpleNamespace(rank=0, dispatcher=types.SimpleNamespace(stage_ahead=lambda a: seen.append(a)))
        rec = record()
        sw.stage_rpc(b, 'submit', sw.StageScoped('cmd', (0,), ahead=rec), lambda m: 'ack')
        self.assertEqual(seen, [admission_to_chunk(rec)], 'the same conversion OPEN uses, so adopt can match exactly')


class Carry(unittest.TestCase):
    """The engine side: which envelopes carry the successor's record."""

    def sender(self, *, on=True, successor=True, opened=(), phase='segments', dead=False):
        import stage_progress as sp
        frame_k = types.SimpleNamespace(dead=False, opened={0, 1})
        nxt = types.SimpleNamespace(dead=dead, opened=set(opened),
                                    stage={0: dict(open_request=types.SimpleNamespace(admission='rec-s0')),
                                           1: dict(open_request=types.SimpleNamespace(admission='rec-s1'))})
        driver = types.SimpleNamespace(prepare_ahead=on, _scoped_frame=frame_k, _scoped_stage=1,
                                       _frames=[frame_k, nxt] if successor else [frame_k])
        driver.__dict__['phase'] = phase
        return sp._ScopedSender(driver, None, (2, 3))

    def cmd(self, kind):
        import wire_v1 as w
        return types.SimpleNamespace(kind=getattr(w.SegmentKind, kind))

    def test_state_finalize_carries_the_stage_s_record(self):
        self.assertEqual(self.sender()._ahead('flash_s5_submit', self.cmd('STATE_FINALIZE')), 'rec-s1')

    def test_other_commands_do_not(self):
        s = self.sender()
        for kind in ('PREPARE', 'LAYER', 'SUFFIX', 'PP_SEND'):
            self.assertIsNone(s._ahead('flash_s5_submit', self.cmd(kind)), kind)
        self.assertIsNone(s._ahead('flash_s5_open', 'x'))

    def test_polls_and_release_only_from_finalize_on(self):
        self.assertIsNone(self.sender(phase='segments')._ahead('flash_s5_poll', 'p'))
        self.assertEqual(self.sender(phase='finalize')._ahead('flash_s5_poll', 'p'), 'rec-s1')
        self.assertEqual(self.sender(phase='release')._ahead('flash_s5_release', 'r'), 'rec-s1')

    def test_off_no_successor_opened_or_dead_carry_nothing(self):
        fin = self.cmd('STATE_FINALIZE')
        self.assertIsNone(self.sender(on=False)._ahead('flash_s5_submit', fin))
        self.assertIsNone(self.sender(successor=False)._ahead('flash_s5_submit', fin))
        self.assertIsNone(self.sender(opened=(1,))._ahead('flash_s5_submit', fin))
        self.assertEqual(self.sender(opened=(0,))._ahead('flash_s5_submit', fin), 'rec-s1', 'stage 0 opened says nothing of stage 1')
        self.assertIsNone(self.sender(dead=True)._ahead('flash_s5_submit', fin))



class Startup(Reset, unittest.TestCase):
    """install_native_lane reads FLASH_S5_PREPARE_AHEAD into the worker factory (the scaffolding of
    test_e3_r4_worker_bridge's install_native_lane test, with the factory's arguments recorded)."""

    def install(self, env):
        import sys
        import native_startup
        import s5_startup
        from unittest.mock import patch
        server = types.SimpleNamespace(operation_device=None, runtime=types.SimpleNamespace(
            state=types.SimpleNamespace(worker_rank=0, prefill=None)))
        runner = types.SimpleNamespace(device='cuda:0', _flash_s5_worker=server)
        native = types.SimpleNamespace(split=types.SimpleNamespace(prefill_stream='prefill'),
                                       comms=types.SimpleNamespace(pp=None), copy_stream=None)
        calls = []

        def factory(runner, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(inventory=lambda: {})

        class StartupDispatcher:
            def __init__(self, *a, **kw):
                self.transaction, self.channel = None, None

            def open(self, *a, **kw):
                return {'opened': True}

            def submit(self, command):
                return {'op_seq': command.op_seq}
        modules = {name: types.ModuleType(name) for name in (
            'vllm', 'vllm.models', 'vllm.models.qwen4_exp', 'vllm.models.qwen4_exp.nvidia', 'vllm.models.qwen4_exp.nvidia.qsa')}
        with patch.object(native_startup, 'install_worker_native', lambda *a, **kw: native), \
             patch.object(lc, 'LaneChunkFactory', factory), \
             patch('lane_dispatch.LaneDispatcher', StartupDispatcher), \
             patch.dict(sys.modules, modules):
            s5_startup.install_native_lane(runner, env=dict(env, FLASH_S5_OVERLAP='0'))
        return calls

    def test_off_is_todays_call(self):
        self.assertEqual(self.install({}), [dict(chunk_tokens=1024)])
        self.assertFalse(lc.PREPARE_AHEAD['selected'])

    def test_on_builds_the_second_set(self):
        self.assertEqual(self.install({'FLASH_S5_PREPARE_AHEAD': '1'}), [dict(chunk_tokens=1024, prepare_ahead=True)])
        self.assertTrue(lc.PREPARE_AHEAD['selected'])


if __name__ == '__main__':
    unittest.main()
