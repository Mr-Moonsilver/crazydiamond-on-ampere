"""CPU-only acceptance for the native retained-storage decode route (E3).

The route under test is the real one: a retained ``WorkerState`` decode lane,
a real ``GraphRouter`` over a real ``SplitDecodeBank``/``SplitCaptureBackend``,
and the generated ``check_route``/``replay_fullgraph`` seams driven by an
ordinary ``SchedulerOutput`` that carries neither ``flash_e3_transaction`` nor
``flash_e3_decode_ticket``.  The split manager is a fake whose
``run_fullgraph`` calls are counted, and every CUDA event is a controllable
fake, so this suite proves the call path, ownership and completion discipline
only.  It is not native, GPU, SM-containment or armability evidence, and a
green run here says nothing about device behaviour.
"""
import contextlib
from pathlib import Path
import sys
import types
import unittest
import pickle
import time
from dataclasses import replace
from queue import Queue
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import graph_hooks  # noqa: E402
import lane_hooks  # noqa: E402
import split_bank  # noqa: E402
import wire_v1 as w  # noqa: E402
from e3_types import (  # noqa: E402
    DecodeStorageOwner,
    E3ContractError,
    E3Pending,
)
from graph_router import (  # noqa: E402
    CompilationIdentity,
    ExecutionDescriptor,
    GraphRouter,
    GraphRoutingError,
    GraphUnclean,
    NativeDecodeRoute,
    NativeReplayResult,
    TransitionAuthorization,
    UnsupportedRoute,
)
from lane_state import (  # noqa: E402
    BatchRow,
    MetadataSnapshot,
    WorkerState,
)
from split_bank import (  # noqa: E402
    RouteController,
    SplitCaptureBackend,
    SplitDecodeBank,
    SplitGraphHandle,
)

# The integration section drives the real bridge/driver over the reviewed
# four-rank stack; only its split-object factory is replaced by the native one.
import test_e3_r4_worker_bridge as bridge_suite  # noqa: E402

FULL_POOL_ID = 2 * 1000003
SPLIT_POOL_ID = 1 * 1000003
NS = types.SimpleNamespace


# --- fake device objects ----------------------------------------------------

class _Event:
    """A device event whose host query is controlled by the test."""

    def __init__(self, bus):
        self.bus = bus
        self.stream = None
        self.complete = False
        self.fail_query = False
        self.raw_result = None
        bus.events.append(self)

    def record(self, stream=None):
        self.stream = stream

    def query(self):
        if self.fail_query:
            raise RuntimeError('event query failed')
        if self.raw_result is not None:
            return self.raw_result
        return self.complete

    def synchronize(self):
        self.complete = True
        self.bus.syncs.append(self)


class _Bus:
    def __init__(self):
        self.events = []
        self.syncs = []

    def event(self):
        return _Event(self)

    def complete_all(self, complete=True):
        for event in self.events:
            event.complete = complete

    def sync_streams(self):
        return tuple(event.stream for event in self.syncs)


class _Torch:
    """CUDA surface for the real split bank/controller over fake events."""

    def __init__(self, bus, split):
        self.bus = bus
        self.split = split
        self.cuda = NS(
            Event=lambda enable_timing=False: bus.event(),
            current_stream=self._current_stream,
            stream=self._stream_scope,
            graph_pool_handle=lambda: ('fake-pool', 1),
            synchronize=lambda index=None: None,
        )
        self.accelerator = NS(synchronize=lambda: None,
                              get_memory_info=lambda: (10_000, 20_000))
        self.inference_mode = lambda: contextlib.nullcontext()
        self._stack = [split.decode_stream]

    def _current_stream(self, device=None):
        return self._stack[-1]

    @contextlib.contextmanager
    def _stream_scope(self, stream):
        self._stack.append(stream)
        try:
            yield stream
        finally:
            self._stack.pop()


class _Mode:
    def __init__(self, name):
        self.name = name


class _Vllm:
    """The minimal pinned descriptor mirror ``from_vllm`` consumes."""

    def __init__(self, tokens=1, reqs=1, uniform=None):
        self.cg_mode = _Mode('FULL')
        self.num_tokens = tokens
        self.num_reqs = reqs
        self.uniform_token_count = tokens if uniform is None else uniform
        self.max_query_len = tokens
        self.num_active_loras = 0
        self.num_ubatches = 1


class _StockManager:
    """The untouched stock manager: one FULL descriptor set, its own pool."""

    pool = (2, 0)

    def __init__(self, descriptors=None):
        self._capture_descs = {
            _Mode('FULL'): list(descriptors or [_Vllm(1), _Vllm(2)])}
        self.fullgraph_calls = []

    def run_fullgraph(self, descriptor):
        self.fullgraph_calls.append(descriptor)
        return _Output('stock', descriptor.num_tokens)


class _SplitManager:
    """Stands in for the second stock ModelCudaGraphManager instance."""

    def __init__(self):
        self.fullgraph_calls = []

    def run_fullgraph(self, descriptor):
        self.fullgraph_calls.append(descriptor)
        return _Output('split', descriptor.num_tokens)


class _Output:
    def __init__(self, source, tokens):
        self.source = source
        self.shape = (1, tokens)


# --- native rig -------------------------------------------------------------

def build_native(rank, state, bus, *, max_pending=8, descriptors=None):
    """Real router/controller/bank over fake native streams and events."""
    split = NS(
        decode_stream=bridge_suite.FakeStream(1000 + rank),
        prefill_stream=bridge_suite.FakeStream(2000 + rank),
        decode_stream_token=1000 + rank,
        decode_context_id=state.decode_owner.context_id,
        decode_sms=16,
        requested_decode_sms=16,
    )
    torch = _Torch(bus, split)
    runner = NS(
        main_stream=bridge_suite.FakeStream(3000 + rank),
        output_copy_stream=bridge_suite.FakeStream(4000 + rank),
        pp_handler=NS(main_stream=bridge_suite.FakeStream(5000 + rank),
                      broadcast_stream=bridge_suite.FakeStream(6000 + rank)),
        cudagraph_manager=_StockManager(descriptors),
        lora_config=None,
        vllm_config='config',
        device=f'cuda:{rank}',
    )
    bank = SplitDecodeBank(runner, split, torch_module=torch)
    manager = _SplitManager()
    bank.manager = manager
    bank.captured = True
    bank.pool = SPLIT_POOL_ID
    for descriptor, local in bank.pairs:
        bank.handles[local] = SplitGraphHandle(descriptor, local, bank)
    backend = SplitCaptureBackend(bank)
    controller = RouteController(runner, bank, torch_module=torch)
    router = GraphRouter(
        state, state.decode.handle,
        CompilationIdentity('test-digest', 'full', (1,), 1),
        backend=backend, full_context_id=0, full_pool_id=FULL_POOL_ID,
        full_basis='estimate', max_native_pending=max_pending)
    router.add_bank('decode-split', bank.execution_context('decode-split'),
                    descriptors=tuple(bank.descriptors()))
    router.capture_at_startup()
    router.probe = lambda: (bank.split.decode_context_id,
                            bank.split.decode_stream_token)
    runner._flash_e3_graph_router = router
    return NS(state=state, bus=bus, split=split, runner=runner, torch=torch,
              bank=bank, manager=manager, backend=backend,
              controller=controller, router=router)


def native_route_objects(rank, state, bus, torch):
    """``test_e3_r4_worker_bridge.route_objects`` with the native backend."""
    rig = build_native(rank, state, bus)
    return rig.split, rig.controller, rig.router, rig.backend


def native_rig(rank=0, *, engine_epoch=2, max_pending=8, descriptors=None):
    state = WorkerState(rank, engine_epoch=engine_epoch)
    state.retain_decode_storage(context_id=100 + rank)
    return build_native(rank, state, _Bus(), max_pending=max_pending,
                        descriptors=descriptors)


def graph_bank(rig):
    """The router's captured bank for the declared split route."""
    return rig.router.banks['decode-split']


def bank_context(rig):
    return graph_bank(rig).context


def bank_descriptors(rig):
    return graph_bank(rig).descriptors


def switch(rig, target):
    """Apply one drained router transition with an explicit authorization."""
    if target == 'split':
        name, context_id = 'decode-split', bank_context(rig).context_id
    else:
        name, context_id = 'ordinary', rig.router.full_context_id
    seq = rig.router._last_transition_seq + 1
    authorization = TransitionAuthorization(
        rig.router.engine_epoch, context_id,
        rig.router.handle.storage_epoch, seq, 'test drained switch')
    return rig.router.transition(name, authorization)


def runner_of(rig):
    return NS(_flash_e3_graph_router=rig.router)


def scheduler_output():
    """An ordinary decode step: no transaction and no per-step ticket."""
    output = NS()
    assert not hasattr(output, 'flash_e3_transaction')
    assert not hasattr(output, 'flash_e3_decode_ticket')
    return output


def stock_call(rig):
    def stock(batch_desc):
        rig.bank.stock.fullgraph_calls.append(batch_desc)
        return _Output('stock', getattr(batch_desc, 'num_tokens', 0))
    return stock


def replay_one(rig, tokens=1, sched=None):
    """Run the generated check_route/replay_fullgraph pair for one step."""
    sched = sched if sched is not None else scheduler_output()
    descriptor = _Vllm(tokens)
    decision = graph_hooks.check_route(runner_of(rig), sched, descriptor,
                                       1, tokens, tokens)
    output = graph_hooks.replay_fullgraph(runner_of(rig), sched, descriptor,
                                          stock_call(rig))
    return decision, output


def park_prefill(state, uid='native-prefill', tokens=4):
    transaction = state.new_transaction(uid, 1, 0, tokens)
    snapshot = MetadataSnapshot.capture(
        transaction, (BatchRow(uid, 1, 0, 0, tokens, 0),))
    state.prefill.reserve(transaction, metadata=snapshot, context_id=9)
    state.prefill.ready()
    return transaction


# --- generated seams --------------------------------------------------------

class NativeRouteSeamTests(unittest.TestCase):
    """The generated seams route ordinary decode without an anchor transaction."""

    def test_ordinary_step_uses_the_typed_native_route_and_stock_full(self):
        rig = native_rig()
        sched = scheduler_output()
        descriptor = _Vllm(1)
        decision = graph_hooks.check_route(runner_of(rig), sched, descriptor,
                                           1, 1, 1)
        self.assertIsInstance(decision, NativeDecodeRoute)
        self.assertEqual(decision.mode, 'ordinary')
        self.assertIsNone(decision.bank)
        self.assertEqual(decision.context_id, rig.router.full_context_id)
        self.assertEqual(decision.pool_id, FULL_POOL_ID)
        self.assertEqual(decision.engine_epoch, rig.state.engine_epoch)
        self.assertEqual(decision.worker_rank, rig.state.worker_rank)
        self.assertEqual(decision.incarnation, rig.state.incarnation)
        self.assertEqual(decision.storage_epoch,
                         rig.state.decode_owner.storage_epoch)
        self.assertEqual(decision.route_generation, 0)

        output = graph_hooks.replay_fullgraph(runner_of(rig), sched, descriptor,
                                             stock_call(rig))
        self.assertEqual(output.source, 'stock')
        self.assertEqual(rig.bank.stock.fullgraph_calls, [descriptor])
        # No green graph was touched and no transaction/ticket was invented.
        self.assertEqual(rig.manager.fullgraph_calls, [])
        self.assertEqual(rig.bus.events, [])
        self.assertFalse(hasattr(sched, 'flash_e3_transaction'))
        self.assertFalse(hasattr(sched, 'flash_e3_decode_ticket'))
        self.assertIsNone(graph_hooks._transaction(sched, rig.router))

    def test_full_split_full_actually_switches_the_manager_calls(self):
        rig = native_rig()
        descriptor = _Vllm(1)
        stock = stock_call(rig)

        ordinary = graph_hooks.check_route(runner_of(rig), scheduler_output(),
                                           descriptor, 1, 1, 1)
        self.assertEqual(ordinary.mode, 'ordinary')
        graph_hooks.replay_fullgraph(runner_of(rig), scheduler_output(),
                                     descriptor, stock)
        self.assertEqual(len(rig.bank.stock.fullgraph_calls), 1)
        self.assertEqual(rig.manager.fullgraph_calls, [])

        switch(rig, 'split')
        split_sched = scheduler_output()
        decision = graph_hooks.check_route(runner_of(rig), split_sched,
                                           descriptor, 1, 1, 1)
        self.assertEqual(decision.mode, 'split')
        self.assertEqual(decision.bank, 'decode-split')
        self.assertEqual(decision.context_id, bank_context(rig).context_id)
        self.assertEqual(decision.pool_id, bank_context(rig).pool_id)
        self.assertEqual(decision.route_generation, 1)
        with split_bank.decode_scope(rig.runner):
            output = graph_hooks.replay_fullgraph(
                runner_of(rig), split_sched, descriptor, stock)
        self.assertEqual(output.source, 'split')
        # The actual retained split manager replayed the actual descriptor.
        self.assertEqual(rig.manager.fullgraph_calls,
                         [rig.bank.pairs[0][0]])
        self.assertEqual(len(rig.bank.stock.fullgraph_calls), 1)
        self.assertEqual(len(rig.bus.events), 1)
        self.assertIs(rig.bus.events[0].stream, rig.split.decode_stream)
        self.assertEqual(rig.router.native_pending, 1)
        self.assertEqual(rig.router.replay_watermark, 0)

        rig.bus.complete_all(True)
        self.assertEqual(len(rig.router.poll_native_replays()), 1)
        self.assertEqual(rig.router.replay_watermark, 1)
        switch(rig, 'ordinary')
        decision = graph_hooks.check_route(runner_of(rig), scheduler_output(),
                                           descriptor, 1, 1, 1)
        self.assertEqual(decision.mode, 'ordinary')
        graph_hooks.replay_fullgraph(runner_of(rig), scheduler_output(),
                                     descriptor, stock)
        self.assertEqual(len(rig.bank.stock.fullgraph_calls), 2)
        self.assertEqual(rig.manager.fullgraph_calls,
                         [rig.bank.pairs[0][0]])

    def test_two_decode_steps_share_one_stable_storage_owner(self):
        rig = native_rig(descriptors=[_Vllm(1), _Vllm(2)])
        switch(rig, 'split')
        owner = rig.state.decode_owner
        handle = rig.state.decode.handle
        d1, o1 = replay_one(rig, tokens=1)
        d2, o2 = replay_one(rig, tokens=2)
        self.assertEqual(o1.source, 'split')
        self.assertEqual(o2.source, 'split')
        self.assertEqual([d1.mode, d2.mode], ['split', 'split'])
        for decision in (d1, d2):
            self.assertEqual(decision.storage_epoch, owner.storage_epoch)
            self.assertEqual(decision.route_generation, 1)
            self.assertEqual(decision.engine_epoch, rig.state.engine_epoch)
            self.assertEqual(decision.incarnation, rig.state.incarnation)
        self.assertIs(rig.state.decode_owner, owner)
        self.assertIs(rig.state.decode.handle, handle)
        self.assertEqual(rig.manager.fullgraph_calls,
                         [rig.bank.pairs[0][0], rig.bank.pairs[1][0]])
        self.assertEqual(rig.router.native_pending, 2)
        self.assertEqual(rig.router.replay_watermark, 0)
        self.assertEqual(len(rig.bus.events), 2)
        self.assertIsNot(rig.bus.events[0], rig.bus.events[1])
        rig.bus.complete_all(True)
        self.assertEqual(len(rig.router.poll_native_replays()), 2)
        self.assertEqual(rig.router.replay_watermark, 2)
        census = rig.router.native_census()
        self.assertEqual(census['submitted'], 2)
        self.assertEqual(census['watermark'], 2)
        self.assertEqual(census['pending'], 0)
        self.assertEqual(census['storage_epoch'], owner.storage_epoch)

    def test_zero_token_batch_is_a_declared_noop(self):
        rig = native_rig()
        descriptor = _Vllm(0, reqs=0, uniform=0)
        self.assertIsNone(graph_hooks.check_route(
            runner_of(rig), scheduler_output(), descriptor, 0, 0, 0))
        self.assertEqual(rig.manager.fullgraph_calls, [])
        self.assertEqual(rig.bus.events, [])
        self.assertEqual(rig.router.replay_watermark, 0)
        switch(rig, 'split')
        self.assertIsNone(graph_hooks.check_route(
            runner_of(rig), scheduler_output(), descriptor, 0, 0, 0))
        self.assertEqual(rig.manager.fullgraph_calls, [])
        self.assertEqual(rig.router.native_pending, 0)

    def test_non_full_descriptor_is_noop_on_full_and_refused_on_split(self):
        rig = native_rig()
        piecewise = _Vllm(1)
        piecewise.cg_mode = _Mode('PIECEWISE')
        self.assertIsNone(graph_hooks.check_route(
            runner_of(rig), scheduler_output(), piecewise, 1, 1, 1))
        self.assertEqual(rig.bank.stock.fullgraph_calls, [])
        switch(rig, 'split')
        with self.assertRaises(UnsupportedRoute):
            graph_hooks.check_route(runner_of(rig), scheduler_output(),
                                    piecewise, 1, 1, 1)
        self.assertEqual(rig.manager.fullgraph_calls, [])
        with self.assertRaises(UnsupportedRoute):
            rig.router.native_decode_route(
                ExecutionDescriptor('piecewise', 1, 1, 1))

    def test_active_prefill_refuses_full_route_and_split_keeps_decode(self):
        rig = native_rig()
        park_prefill(rig.state)
        with self.assertRaises(UnsupportedRoute):
            graph_hooks.check_route(runner_of(rig), scheduler_output(),
                                    _Vllm(1), 1, 1, 1)
        self.assertEqual(rig.bank.stock.fullgraph_calls, [])
        switch(rig, 'split')
        # The parked prefill does not turn the decode step into a stock replay.
        decision, output = replay_one(rig, tokens=1)
        self.assertEqual(decision.mode, 'split')
        self.assertEqual(output.source, 'split')
        self.assertEqual(len(rig.manager.fullgraph_calls), 1)

    def test_uncaptured_descriptor_is_refused_without_a_replay(self):
        rig = native_rig()
        switch(rig, 'split')
        with self.assertRaises(UnsupportedRoute):
            graph_hooks.check_route(runner_of(rig), scheduler_output(),
                                    _Vllm(5), 1, 5, 5)
        self.assertEqual(rig.manager.fullgraph_calls, [])
        with self.assertRaises(UnsupportedRoute):
            rig.router.native_replay(
                ExecutionDescriptor('full', 5, 1, 5),
                observed_context=bank_context(rig).context_id,
                observed_stream=bank_context(rig).stream_token)
        self.assertEqual(rig.manager.fullgraph_calls, [])
        self.assertEqual(rig.router.native_pending, 0)

    def test_legacy_transaction_router_still_refuses_without_a_transaction(self):
        # Removing the persistent owner makes the seam treat the router as a
        # non-native one again: the reviewed refusal must remain.
        rig = native_rig()
        switch(rig, 'split')
        rig.state.decode_owner = None
        with self.assertRaises(GraphRoutingError):
            graph_hooks.check_route(runner_of(rig), scheduler_output(),
                                    _Vllm(1), 1, 1, 1)
        self.assertEqual(rig.manager.fullgraph_calls, [])


class NativeExecuteSeamTests(unittest.TestCase):
    """A's frameless execute seam hands real decode to the native route."""

    def make_runtime(self, state):
        adapters = NS(get=lambda: None, set=lambda value: None)
        workspace = NS(enter=lambda value: None, exit=lambda value: None)
        runtime = lane_hooks.LaneRuntime(state, forward_context=adapters,
                                         workspace=workspace)
        return runtime

    def test_native_decode_keeps_one_dispatch_owner(self):
        rig = native_rig()
        runtime = self.make_runtime(rig.state)
        runner = runner_of(rig)
        runner._flash_s5_native_decode = True
        runner._flash_e3_lane = runtime
        sched = scheduler_output()
        calls = []

        def stock(scheduler_output, *args):
            calls.append((scheduler_output, args))
            return 'stock-output'

        self.assertEqual(lane_hooks.execute_with_e3(runner, stock, sched),
                         'stock-output')
        self.assertEqual(len(calls), 1)
        self.assertEqual(runtime.real_frames, 1)
        self.assertFalse(hasattr(sched, 'flash_e3_transaction'))

        runtime.dispatcher.active = True
        with self.assertRaises(E3Pending):
            lane_hooks.execute_with_e3(runner, stock, sched)
        runtime.dispatcher.active = False
        self.assertEqual(len(calls), 1)

    def test_native_decode_is_flag_and_router_gated(self):
        runner = NS(_flash_s5_native_decode=True,
                    _flash_e3_graph_router=NS())
        self.assertFalse(lane_hooks.native_decode_ready(runner))
        runner._flash_e3_graph_router = NS(native_decode_route=lambda value: value)
        self.assertTrue(lane_hooks.native_decode_ready(runner))
        del runner._flash_s5_native_decode
        self.assertFalse(lane_hooks.native_decode_ready(runner))

        rig = native_rig()
        runtime = self.make_runtime(rig.state)
        runner = runner_of(rig)
        runner._flash_e3_lane = runtime

        def stock(scheduler_output, *args):
            return 'stock-output'

        with self.assertRaises(E3ContractError):
            lane_hooks.execute_with_e3(runner, stock, scheduler_output())


# --- replay completion ------------------------------------------------------

class NativeReplayCompletionTests(unittest.TestCase):
    """A host enqueue is not completion; only a queried event retires work."""

    def split_rig(self, **kwargs):
        rig = native_rig(**kwargs)
        switch(rig, 'split')
        return rig

    def test_unqueried_replay_blocks_transition_and_shutdown(self):
        rig = self.split_rig()
        _decision, output = replay_one(rig)
        self.assertEqual(output.source, 'split')
        self.assertEqual(rig.router.replay_watermark, 0)
        self.assertEqual(rig.router.native_pending, 1)
        self.assertEqual(rig.router.poll_native_replays(), ())
        self.assertEqual(rig.router.replay_watermark, 0)
        self.assertEqual(rig.router.native_pending, 1)
        with self.assertRaises(E3Pending):
            switch(rig, 'ordinary')
        self.assertEqual(rig.router.current_mode, 'decode-split')
        self.assertEqual(rig.controller.route, 'full')
        with self.assertRaises(E3Pending):
            rig.router.shutdown()

    def test_queried_completion_advances_the_watermark_once(self):
        rig = self.split_rig()
        replay_one(rig)
        event = rig.bus.events[0]
        self.assertFalse(event.query())
        event.complete = True
        completed = rig.router.poll_native_replays()
        self.assertEqual(len(completed), 1)
        self.assertEqual(rig.router.replay_watermark, 1)
        self.assertEqual(rig.router.native_pending, 0)
        self.assertEqual(rig.router.poll_native_replays(), ())
        self.assertEqual(rig.router.replay_watermark, 1)
        records = rig.router.timing_records
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].basis, 'unmeasured')
        switch(rig, 'ordinary')
        self.assertEqual(rig.router.route_generation, 2)

    def test_query_failure_and_non_boolean_quarantine(self):
        rig = self.split_rig()
        replay_one(rig)
        rig.bus.events[0].fail_query = True
        with self.assertRaises(GraphRoutingError):
            rig.router.poll_native_replays()
        self.assertTrue(rig.router.unclean)
        self.assertEqual(rig.router.replay_watermark, 0)

        rig = self.split_rig()
        replay_one(rig)
        rig.bus.events[0].raw_result = 1
        with self.assertRaises(GraphRoutingError):
            rig.router.poll_native_replays()
        self.assertTrue(rig.router.unclean)
        self.assertEqual(rig.router.replay_watermark, 0)

    def test_pending_replays_are_bounded_and_apply_backpressure(self):
        rig = self.split_rig(max_pending=2,
                             descriptors=[_Vllm(1), _Vllm(2), _Vllm(3)])
        replay_one(rig, tokens=1)
        replay_one(rig, tokens=2)
        with self.assertRaises(E3Pending):
            rig.router.native_replay(
                bank_descriptors(rig)[2],
                observed_context=bank_context(rig).context_id,
                observed_stream=bank_context(rig).stream_token)
        self.assertEqual(len(rig.manager.fullgraph_calls), 2)
        self.assertEqual(rig.router.native_pending, 2)
        self.assertEqual(len(rig.bus.events), 2)
        # Only the first event completed: one lease retires, capacity returns.
        rig.bus.events[0].complete = True
        self.assertEqual(len(rig.router.poll_native_replays(limit=1)), 1)
        self.assertEqual(rig.router.replay_watermark, 1)
        result = rig.router.native_replay(
            bank_descriptors(rig)[2],
            observed_context=bank_context(rig).context_id,
            observed_stream=bank_context(rig).stream_token)
        self.assertIsInstance(result, NativeReplayResult)
        self.assertEqual(rig.router.native_pending, 2)
        self.assertEqual(len(rig.manager.fullgraph_calls), 3)

    def test_wrong_observed_context_or_stream_never_enqueues(self):
        rig = self.split_rig()
        descriptor = bank_descriptors(rig)[0]
        with self.assertRaises(GraphRoutingError):
            rig.router.native_replay(
                descriptor,
                observed_context=bank_context(rig).context_id + 1,
                observed_stream=bank_context(rig).stream_token)
        with self.assertRaises(GraphRoutingError):
            rig.router.native_replay(
                descriptor,
                observed_context=bank_context(rig).context_id,
                observed_stream=bank_context(rig).stream_token + 1)
        self.assertEqual(rig.manager.fullgraph_calls, [])
        self.assertEqual(rig.router.native_pending, 0)
        self.assertEqual(rig.bus.events, [])

    def test_replay_on_the_wrong_current_stream_is_refused_and_quarantined(self):
        rig = self.split_rig()
        descriptor = bank_descriptors(rig)[0]
        with rig.torch.cuda.stream(rig.split.prefill_stream):
            with self.assertRaises(Exception):
                rig.router.native_replay(
                    descriptor,
                    observed_context=bank_context(rig).context_id,
                    observed_stream=bank_context(rig).stream_token)
        self.assertEqual(rig.manager.fullgraph_calls, [])
        self.assertTrue(rig.router.unclean)
        self.assertEqual(rig.router.native_pending, 0)

    def test_event_on_another_stream_and_missing_query_quarantine(self):
        class WrongToken:
            def __init__(self, inner):
                self.inner = inner

            def replay_with_event(self, *args):
                output, event = self.inner.replay_with_event(*args)
                event.stream_token = 424242
                return output, event

        rig = self.split_rig()
        descriptor = bank_descriptors(rig)[0]
        real = rig.router.backend
        rig.router.backend = WrongToken(real)
        with self.assertRaises(GraphRoutingError):
            rig.router.native_replay(
                descriptor,
                observed_context=bank_context(rig).context_id,
                observed_stream=bank_context(rig).stream_token)
        self.assertTrue(rig.router.unclean)
        self.assertEqual(rig.router.native_pending, 0)

        class NoEvent:
            def replay_with_event(self, *args):
                return _Output('split', 1), object()

        rig = native_rig()
        switch(rig, 'split')
        rig.router.backend = NoEvent()
        with self.assertRaises(GraphRoutingError):
            rig.router.native_replay(
                bank_descriptors(rig)[0],
                observed_context=bank_context(rig).context_id,
                observed_stream=bank_context(rig).stream_token)
        self.assertTrue(rig.router.unclean)

    def test_enqueue_failure_quarantines_instead_of_claiming_completion(self):
        class FailEnqueue:
            def __init__(self, inner):
                self.inner = inner

            def replay_with_event(self, *args):
                raise RuntimeError('synthetic enqueue failure')

        rig = self.split_rig()
        rig.router.backend = FailEnqueue(rig.router.backend)
        with self.assertRaises(RuntimeError):
            rig.router.native_replay(
                bank_descriptors(rig)[0],
                observed_context=bank_context(rig).context_id,
                observed_stream=bank_context(rig).stream_token)
        self.assertTrue(rig.router.unclean)
        self.assertEqual(rig.router.replay_watermark, 0)
        self.assertEqual(rig.router.native_pending, 0)

    def test_foreign_storage_owner_and_stale_engine_epoch_are_refused(self):
        rig = native_rig()
        owner = rig.state.decode_owner
        rig.state.decode_owner = DecodeStorageOwner(
            rig.state.worker_rank + 1, rig.state.incarnation,
            owner.storage_epoch, owner.context_id)
        with self.assertRaises(GraphUnclean):
            rig.router.native_decode_route(bank_descriptors(rig)[0])
        rig.state.decode_owner = owner

        rig.state.engine_epoch = rig.state.engine_epoch + 1
        with self.assertRaises(GraphUnclean):
            rig.router.native_decode_route(bank_descriptors(rig)[0])
        rig.state.engine_epoch = rig.state.engine_epoch - 1

        rig.state.decode.state = 'reserved'
        with self.assertRaises(GraphUnclean):
            rig.router.native_decode_route(bank_descriptors(rig)[0])
        rig.state.decode.state = 'retained'
        self.assertEqual(rig.router.native_decode_route(
            bank_descriptors(rig)[0]).mode, 'ordinary')


# --- real router/bridge/driver ---------------------------------------------

def make_native_stack():
    with mock.patch.object(bridge_suite, 'route_objects',
                           native_route_objects):
        return bridge_suite.Stack()


class NativeBridgeIntegrationTests(unittest.TestCase):
    """The real driver/bridge/routers with fake native event ports."""

    def finish_chunk(self, stack):
        stack.drive_to_finalize()
        stack.complete_everything()
        stack.drive(lambda: stack.driver.tx is None)
        self.assertIsNone(stack.driver.error)

    def split_ready(self, stack):
        stack.drive(lambda: stack.driver._split_ready)
        for rank in stack.driver.ranks:
            self.assertEqual(stack.worker(rank).controller.route, 'split')
            self.assertEqual(stack.worker(rank).router.current_mode,
                             'decode-split')
        return stack.driver.engine_epoch

    def split_issued(self, stack):
        """Bootstrap + split computed on every worker, ACK not yet consumed."""
        stack.driver.advance()
        stack.driver.advance()
        self.assertFalse(stack.driver._split_ready)
        for rank in stack.driver.ranks:
            self.assertEqual(stack.worker(rank).controller.route, 'split')
            self.assertEqual(stack.worker(rank).router.current_mode,
                             'decode-split')
        return stack.driver.engine_epoch

    def full_command(self, stack, rank=0, **overrides):
        worker = stack.worker(rank)
        values = dict(
            engine_epoch=stack.driver.engine_epoch,
            expected_route_generation=worker.controller.route_generation,
            route_generation=worker.controller.route_generation + 1,
            target_bank=w.ROUTE_BANK_FULL,
            storage_epochs=tuple(
                stack.worker(item).state.decode.handle.storage_epoch
                for item in stack.driver.ranks),
            decode_replay_watermarks=tuple(
                stack.worker(item).router.replay_watermark
                for item in stack.driver.ranks))
        values.update(overrides)
        return w.TransitionCommand(**values)

    def decode_step(self, worker, tokens=1):
        """Run one generated native decode step for a bridge-backed worker."""
        runner = worker.controller.runner
        sched = scheduler_output()
        descriptor = _Vllm(tokens)

        def reject(batch_desc):
            raise AssertionError('stock FULL manager must not run on the split route')

        before = len(worker.bus.events)
        with split_bank.decode_scope(runner):
            decision = graph_hooks.check_route(runner, sched, descriptor,
                                               1, tokens, tokens)
            output = graph_hooks.replay_fullgraph(runner, sched, descriptor,
                                                  reject)
        self.assertFalse(hasattr(sched, 'flash_e3_transaction'))
        self.assertFalse(hasattr(sched, 'flash_e3_decode_ticket'))
        created = worker.bus.events[before:]
        self.assertEqual(len(created), 1)
        return decision, output, created[0]

    def test_pending_native_replay_blocks_the_return_until_queried(self):
        stack = make_native_stack()
        self.finish_chunk(stack)
        self.split_ready(stack)
        self.assertIsNotNone(stack.controller.admit(stack.scheduler))
        self.finish_chunk(stack)
        replay_events = {}
        for rank in stack.driver.ranks:
            decision, output, event = self.decode_step(stack.worker(rank))
            self.assertEqual(decision.mode, 'split')
            self.assertEqual(output.source, 'split')
            self.assertEqual(stack.worker(rank).router.native_pending, 1)
            replay_events[rank] = event

        self.assertIsNotNone(stack.controller.admit(stack.scheduler))
        stack.drive_to_finalize()
        stack.complete_everything()
        # The test owns device completion: the two native replay events stay
        # unqueried even though every consumer tail event has synchronized.
        for event in replay_events.values():
            event.complete = False
        opens = sum(1 for envelope in stack.sender.envelopes
                    if envelope[0] == 'flash_s5_open')
        watermarks = tuple(worker.router.replay_watermark
                           for worker in (stack.worker(rank)
                                          for rank in stack.driver.ranks))
        self.assertEqual(watermarks, (0, 0, 0, 0))

        # No stock work may run while the handback is owed, and the driver
        # must represent the typed pending rather than a fatal transport error.
        stack.drive(lambda: stack.driver._return_pending == 'handback')
        self.assertIsNone(stack.driver.error)
        self.assertIsNotNone(stack.driver._transition)
        for rank in stack.driver.ranks:
            self.assertEqual(stack.worker(rank).controller.route, 'split')
        self.assertEqual(sum(1 for envelope in stack.sender.envelopes
                             if envelope[0] == 'flash_s5_open'), opens)

        for event in replay_events.values():
            event.complete = True
        stack.drive(lambda: stack.driver._split_ready is False)
        self.assertIsNone(stack.driver.error)
        for rank in stack.driver.ranks:
            worker = stack.worker(rank)
            self.assertEqual(worker.controller.route, 'full')
            self.assertEqual(worker.router.current_mode, 'ordinary')
            # The acknowledged census is the worker's queried watermark.
            self.assertGreaterEqual(worker.router.replay_watermark, 1)
            self.assertGreaterEqual(stack.driver._decode_watermarks[rank], 1)
        self.assertEqual(stack.driver.route_generation, 2)
        self.assertIsNone(stack.driver._return_pending)

    def pending_return(self, pending_ranks=(1,)):
        stack = make_native_stack()
        self.finish_chunk(stack)
        self.split_ready(stack)
        self.assertIsNotNone(stack.controller.admit(stack.scheduler))
        self.finish_chunk(stack)
        events = {rank: self.decode_step(stack.worker(rank))[2]
                  for rank in stack.driver.ranks}
        self.assertIsNotNone(stack.controller.admit(stack.scheduler))
        stack.drive_to_finalize()
        stack.complete_everything()
        for rank, event in events.items():
            event.complete = rank not in pending_ranks
        stack.drive(lambda: stack.driver._transition is not None)
        return stack, events

    def test_mixed_rank_pending_retries_exact_command_and_switches_each_rank_once(self):
        stack, events = self.pending_return()
        command = stack.driver._transition
        self.assertEqual(stack.worker(0).controller.route, 'full')
        self.assertEqual(stack.worker(1).controller.route, 'split')
        self.assertTrue(stack.driver._split_ready)  # no all-rank ACK yet
        for _ in range(3):
            self.assertEqual(stack.driver.advance(), (None, False))
        self.assertEqual(stack.driver._transition, command)
        for rank in (0, 2, 3):
            self.assertEqual(len(stack.worker(rank).controller.transitions), 2)
        events[1].complete = True
        stack.drive(lambda: not stack.driver._split_ready)
        self.assertEqual(stack.driver.route_generation, 2)
        self.assertIsNone(stack.driver.error)
        self.assertIsNone(stack.driver._return_pending)
        for worker in stack.workers.values():
            self.assertEqual(worker.controller.route, 'full')
            self.assertEqual(len(worker.controller.transitions), 2)
        retried = [args[0] for method, args, _, _ in stack.sender.envelopes
                   if method == 'flash_s5_transition' and args[0].target_bank == w.ROUTE_BANK_FULL]
        self.assertGreater(len(retried), 1)
        self.assertTrue(all(value == command for value in retried))
        self.assertTrue(all(ack.drained_streams > 0
                            for ack in stack.driver._last_transition_acks.values()))

    def test_pending_transition_has_one_deadline_not_one_per_retry(self):
        stack, _ = self.pending_return()
        deadline = stack.driver._transition_deadline
        stack.driver.advance()
        self.assertEqual(stack.driver._transition_deadline, deadline)
        stack.driver.clock = lambda: deadline + 1
        with self.assertRaisesRegex(bridge_suite.ed.DriverError, 'deadline expired'):
            stack.driver.advance()
        self.assertIsNotNone(stack.driver.error)
        self.assertIsNotNone(stack.driver._transition)
        self.assertTrue(stack.driver._split_ready)

    def test_pause_mid_partial_transition_keeps_retrying_until_all_ranks_drain(self):
        stack, events = self.pending_return()
        command = stack.driver._transition
        stack.driver.stop('pause')
        self.assertEqual(stack.driver._transition, command)
        self.assertTrue(stack.driver.active)
        self.assertIsNone(stack.driver.error)
        events[1].complete = True
        stack.drive(lambda: not stack.driver.active)
        self.assertIsNone(stack.driver.stop_kind)
        self.assertFalse(stack.driver._split_ready)
        self.assertTrue(stack.driver.finish_pause())
        for worker in stack.workers.values():
            self.assertEqual(worker.controller.route, 'full')

    def test_exact_retry_cache_refuses_changed_command_and_route(self):
        stack, _ = self.pending_return()
        worker = stack.worker(0)
        command = stack.driver._transition
        ack = worker.flash_s5_transition(command)
        self.assertIs(ack, worker.flash_s5_transition(command))
        changed = replace(command, decode_replay_watermarks=(1, 0, 0, 0))
        with self.assertRaises(bridge_suite.WorkerBridgeError):
            worker.flash_s5_transition(changed)
        worker.controller.route = 'split'
        with self.assertRaisesRegex(bridge_suite.WorkerBridgeError, 'exact retry'):
            worker.flash_s5_transition(command)
        self.assertTrue(worker.router.unclean)

    def test_pending_is_success_value_on_owned_fifo_and_missing_rank_blocks_commit(self):
        from reply_transport import ReplyTransport
        class Reader:
            def __init__(self):
                self.queue = Queue()
            def dequeue(self, timeout):
                return self.queue.get(timeout=timeout)
        stack, events = self.pending_return()
        command = stack.driver._transition
        readers = {rank: Reader() for rank in stack.driver.ranks}
        transport = ReplyTransport(readers, 'success', handoff_idle=True)
        def send(_):
            for rank in stack.driver.ranks:
                reply = stack.worker(rank).flash_s5_transition(command)
                # Exercise actual serialization and SUCCESS envelopes. A pending
                # drain must never poison the transport as a worker failure.
                readers[rank].queue.put(('success', pickle.loads(pickle.dumps(reply))))
        try:
            sequence = transport.submit('transition', send, timeout_s=2)
            until = time.monotonic() + 1
            while not transport._calls[sequence].ready and time.monotonic() < until:
                transport.poll(8)
                time.sleep(.001)
            replies, _ = transport.take(sequence)
            self.assertEqual([type(value) for value in replies],
                             [w.TransitionAck, w.TransitionPending,
                              w.TransitionAck, w.TransitionAck])
            self.assertIsNone(transport._failure)
            with self.assertRaisesRegex(bridge_suite.ed.DriverError, 'cover every rank'):
                stack.driver._apply('transition', replies[:-1])
            self.assertEqual(stack.driver.route_generation, 1)
            stack.driver._apply('transition', replies)
            self.assertEqual(stack.driver.route_generation, 1)
            events[1].complete = True
        finally:
            self.assertTrue(transport.close(1)['clean'])

    def test_route_refusals_name_the_actual_owner_and_mutate_nothing(self):
        stack = make_native_stack()
        self.split_issued(stack)
        epoch = stack.driver.engine_epoch
        rank = stack.driver.ranks[0]
        worker = stack.worker(rank)
        storage = list(self.full_command(stack).storage_epochs)
        watermarks = list(self.full_command(stack).decode_replay_watermarks)
        generation = worker.controller.route_generation
        cases = [
            dict(engine_epoch=epoch + 1),
            dict(storage_epochs=tuple(value + 1 if index == rank else value
                                      for index, value in enumerate(storage))),
            dict(expected_route_generation=generation - 1,
                 route_generation=generation),
            dict(decode_replay_watermarks=tuple(
                value + 1 if index == rank else value
                for index, value in enumerate(watermarks))),
        ]
        for overrides in cases:
            with self.assertRaises(bridge_suite.WorkerBridgeError):
                worker.flash_s5_transition(self.full_command(stack, **overrides))
            self.assertEqual(worker.controller.route, 'split')
            self.assertEqual(worker.router.current_mode, 'decode-split')
            self.assertEqual(worker.controller.route_generation, 1)
            self.assertEqual(worker.router.route_generation, 1)

        owner = worker.state.decode_owner
        worker.state.decode_owner = DecodeStorageOwner(
            owner.worker_rank + 1, owner.incarnation, owner.storage_epoch,
            owner.context_id)
        with self.assertRaises(bridge_suite.WorkerBridgeError):
            worker.flash_s5_transition(self.full_command(stack))
        self.assertEqual(worker.controller.route, 'split')
        worker.state.decode_owner = owner

        ack = worker.flash_s5_transition(self.full_command(stack))
        self.assertEqual(ack.target_bank, w.ROUTE_BANK_FULL)
        self.assertEqual(ack.applied_route_generation, generation + 1)
        self.assertEqual(ack.decode_replay_watermark, 0)
        self.assertEqual(worker.controller.route, 'full')
        self.assertEqual(worker.router.current_mode, 'ordinary')

    def test_partial_switch_is_quarantined_and_never_acknowledged(self):
        stack = make_native_stack()
        self.split_issued(stack)
        rank = stack.driver.ranks[0]
        worker = stack.worker(rank)

        def refuse(*args, **kwargs):
            raise RuntimeError('synthetic router refusal')

        worker.router.transition = refuse
        with self.assertRaises(bridge_suite.WorkerBridgeError):
            worker.flash_s5_transition(self.full_command(stack))
        self.assertTrue(worker.router.unclean)
        self.assertEqual(worker.controller.route, 'full')
        with self.assertRaises(bridge_suite.WorkerBridgeError):
            worker.flash_s5_transition(self.full_command(stack))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
