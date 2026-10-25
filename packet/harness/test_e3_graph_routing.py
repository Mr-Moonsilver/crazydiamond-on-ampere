"""CPU-only tests for E3 fixed-split graph routing (B).

No torch, no vLLM, no CUDA, no device.  Graphs, streams and events are opaque
tokens in a fake backend; A's real lane identities and exclusion rules are used
so graph keys and lifetime guards are exercised against the merged contract.
"""
import ast
import dataclasses
import hashlib
import os
from pathlib import Path
import sys
import textwrap
import threading
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import e3_types  # noqa: E402
import graph_hooks  # noqa: E402
import lane_hooks  # noqa: E402
import runner_hooks  # noqa: E402
from e3_types import (  # noqa: E402
    CompletionKind,
    CompletionProof,
    DependencyRole,
    E3Pending,
    LaneRole,
    OperationTicket,
)
from graph_router import (  # noqa: E402
    BankBudget,
    CompilationIdentity,
    ExecutionDescriptor,
    ExecutionContext,
    GraphBudgetReport,
    GraphKey,
    GraphRouter,
    GraphRoutingError,
    GraphUnclean,
    ReplayResult,
    TimingRecord,
    TransitionAuthorization,
    UnsupportedRoute,
)
from lane_state import (  # noqa: E402
    GRAPH_POOL_MEASURED_BYTES,
    BatchRow,
    LaneHandle,
    MetadataSnapshot,
    ScratchSpec,
    WorkerState,
)

SOURCE = ROOT / 'patches/rowcache/model_runner_v2.py.patched'
NS = types.SimpleNamespace

CONTEXT_ID = 7
STREAM = 0x9000
POOL = 0x5000
RESERVED = 8_000_000


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def descriptor(tokens=2, reqs=1, uniform=1, mode='full', loras=0, ubatches=1):
    return ExecutionDescriptor(mode, tokens, reqs, uniform, None, loras, ubatches)


def compilation(digest='cfg-v16-no-v16g', mode='full', sizes=(2, 4)):
    return CompilationIdentity(digest, mode, tuple(sizes), 1, (0,), False)


def split_context(name='green34', context_id=CONTEXT_ID, requested=34,
                  measured=34, pool_id=POOL, stream=STREAM,
                  reserved=RESERVED, basis='estimate'):
    return ExecutionContext(name, context_id, requested, measured, pool_id,
                            stream, reserved, basis)


def full_context():
    return ExecutionContext('full', 0, 132, 132, 0x1000, 0x100,
                            GRAPH_POOL_MEASURED_BYTES, 'measured')


class FakeOutput:
    def __init__(self, tokens):
        self.sampled_token_ids = [1] * tokens
        self.hidden_states = ('hidden', tokens)
        self.shape = (1, tokens)


class FakeBackend:
    """Opaque graph/event/stream tokens; every call is recorded, never waited."""

    def __init__(self):
        self.capture_calls = []
        self.replay_calls = []
        self.destroyed = []
        self.fail_capture_at: int | None = None
        self.fail_replay = False
        self.fail_destroy = False

    def capture(self, name, key, lane, context):
        if self.fail_capture_at is not None and len(self.capture_calls) == self.fail_capture_at:
            raise RuntimeError('fake capture failure')
        handle = ('graph', name, key.context_id, key.descriptor.num_tokens,
                  len(self.capture_calls))
        self.capture_calls.append((name, key, lane, context, handle))
        return handle

    def replay(self, handle, key, lane, context):
        if self.fail_replay:
            raise RuntimeError('fake replay enqueue failure')
        self.replay_calls.append((handle, key, lane, context))
        return FakeOutput(key.descriptor.num_tokens)

    def destroy_graph(self, handle):
        if self.fail_destroy:
            raise RuntimeError('fake destroy failure')
        self.destroyed.append(('graph', handle))

    def destroy_event(self, token):
        self.destroyed.append(('event', token))

    def destroy_stream(self, token):
        self.destroyed.append(('stream', token))


class FakePP:
    def __init__(self):
        self.ring = [('prev', 0), ('prev', 1)]
        self.calls = []

    def get_prev_sampled_outputs(self):
        self.calls.append('prev')
        return self.ring.pop(0)

    def broadcast(self, *args, **kwargs):
        self.calls.append('broadcast')


def decode_worker(rank=0, context_id=CONTEXT_ID, uid='decode-request'):
    worker = WorkerState(rank)
    transaction = worker.new_transaction(uid, 1, 0, 2)
    handle = worker.decode.reserve(transaction, context_id=context_id)
    worker.decode.ready()
    return worker, transaction, handle


def activate_prefill(worker, uid='prefill-request', context_id=9, tokens=4):
    transaction = worker.new_transaction(uid, 1, 0, tokens)
    snapshot = MetadataSnapshot.capture(
        transaction, (BatchRow(uid, 1, 0, 0, tokens, 0),))
    handle = worker.prefill.reserve(transaction, metadata=snapshot,
                                    context_id=context_id)
    worker.prefill.ready()
    return transaction, snapshot, handle


def decode_ticket(worker, transaction, op_seq=0,
                  role=DependencyRole.CONSUMER):
    return OperationTicket(transaction, worker.worker_rank,
                           worker.decode.lane_id, op_seq, role)


def prefill_ticket(worker, transaction, op_seq=0):
    return OperationTicket(transaction, worker.worker_rank,
                           worker.prefill.lane_id, op_seq,
                           DependencyRole.PRODUCER)


def finish(worker, ticket, seq=0, kind=CompletionKind.GPU_EVENT, lane=None):
    proof = CompletionProof(ticket, kind, seq)
    (lane or worker.decode).complete(proof)
    return proof


def attach_probe(router):
    def probe():
        if router.current_mode in router.banks:
            context = router.banks[router.current_mode].context
            return context.context_id, context.stream_token
        return router.full_context_id, full_context().stream_token
    router.probe = probe
    return router


def make_router(*, worker_handle=None, backend=None, banks=None, capture=True,
                compilation_identity=None, reserve_required=100, **kwargs):
    if worker_handle is None:
        worker_handle = decode_worker()
    if len(worker_handle) == 3:
        worker, _transaction, handle = worker_handle
    else:
        worker, handle = worker_handle
    backend = FakeBackend() if backend is None else backend
    router = GraphRouter(
        worker, handle, compilation_identity or compilation(), backend=backend,
        full_context_id=0, full_pool_id=0x1000,
        reserve_required_bytes=reserve_required, **kwargs)
    if banks is None:
        banks = [('split', split_context(), (descriptor(),))]
    for name, context, descriptors in banks:
        router.add_bank(name, context, descriptors=descriptors)
    if capture:
        router.capture_at_startup()
    attach_probe(router)
    return NS(router=router, worker=worker, handle=handle, backend=backend,
              banks=banks, context=banks[0][1] if banks else None)


def auth(router, target_context=CONTEXT_ID, seq=10, engine=1, storage=None,
         reason='test authorization'):
    return TransitionAuthorization(
        engine, target_context,
        router.handle.storage_epoch if storage is None else storage,
        seq, reason)


def transition(rig, target='split', target_context=None, seq=10):
    context_id = (rig.banks[0][1].context_id if target_context is None
                  else target_context)
    return rig.router.transition(target, auth(rig.router, context_id, seq))


def replay(rig, transaction=None, ticket=None, desc=None, context=None):
    worker, router = rig.worker, rig.router
    transaction = transaction or worker.decode.transaction
    if ticket is None:
        ticket = decode_ticket(worker, transaction)
        worker.decode.submit(ticket)
    desc = desc or descriptor()
    context = context or rig.banks[0][1]
    return router.replay(transaction, desc, ticket,
                         observed_context=context.context_id,
                         observed_stream=context.stream_token), ticket


_A_CACHE = {}


def a_sources():
    if not _A_CACHE:
        pinned = SOURCE.read_text()
        execution = runner_hooks.runner_execution(pinned)
        startup = runner_hooks.runner_startup(execution)
        _A_CACHE['pinned'] = pinned
        _A_CACHE['e3'] = lane_hooks.generate(pinned)
        _A_CACHE['e3-e2-exec'] = lane_hooks.generate(execution)
        _A_CACHE['e3-e2-startup'] = lane_hooks.generate(startup)
    return _A_CACHE


def method_source(generated, name):
    tree = ast.parse(generated)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'GPUModelRunner':
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    lines = generated.splitlines(keepends=True)
                    return textwrap.dedent(
                        ''.join(lines[item.lineno - 1:item.end_lineno]))
    raise AssertionError('method not found: ' + name)


class TestGraphIdentity(unittest.TestCase):
    def test_key_binds_every_component(self):
        rig = make_router()
        bank = rig.router.banks['split']
        desc = descriptor()
        key = bank.key_for(desc)
        self.assertEqual(key.descriptor, desc)
        self.assertEqual(key.compilation, compilation())
        self.assertEqual(key.worker_rank, 0)
        self.assertEqual(key.context_id, CONTEXT_ID)
        self.assertEqual(key.measured_sm, 34)
        self.assertEqual(key.storage_epoch, rig.handle.storage_epoch)
        self.assertEqual(key.pool_id, POOL)
        variants = [
            GraphKey(descriptor(tokens=3), key.compilation, key.worker_rank,
                     key.context_id, key.measured_sm, key.storage_epoch, key.pool_id),
            GraphKey(desc, compilation('different-digest'), key.worker_rank,
                     key.context_id, key.measured_sm, key.storage_epoch, key.pool_id),
            GraphKey(desc, key.compilation, 1, key.context_id, key.measured_sm,
                     key.storage_epoch, key.pool_id),
            GraphKey(desc, key.compilation, key.worker_rank, CONTEXT_ID + 1,
                     key.measured_sm, key.storage_epoch, key.pool_id),
            GraphKey(desc, key.compilation, key.worker_rank, key.context_id, 33,
                     key.storage_epoch, key.pool_id),
            GraphKey(desc, key.compilation, key.worker_rank, key.context_id,
                     key.measured_sm, key.storage_epoch + 1, key.pool_id),
            GraphKey(desc, key.compilation, key.worker_rank, key.context_id,
                     key.measured_sm, key.storage_epoch, key.pool_id + 1),
        ]
        for variant in variants:
            self.assertNotEqual(key, variant)

    def test_requested_sm_is_not_proof(self):
        rig = make_router(banks=[('split', split_context(requested=34, measured=33),
                                  (descriptor(),))])
        bank = rig.router.banks['split']
        key = bank.key_for(descriptor())
        self.assertEqual(key.measured_sm, 33)
        self.assertEqual(bank.context.requested_sm, 34)
        self.assertNotEqual(key.measured_sm, bank.context.requested_sm)
        with self.assertRaises(GraphRoutingError):
            split_context(measured=0)
        with self.assertRaises(GraphRoutingError):
            split_context(requested=0)

    def test_no_pci_or_ordinal_in_key(self):
        names = {field.name for field in dataclasses.fields(GraphKey)}
        self.assertEqual(names, {'descriptor', 'compilation', 'worker_rank',
                                 'context_id', 'measured_sm', 'storage_epoch',
                                 'pool_id'})
        self.assertFalse(names & {'device', 'ordinal', 'pci', 'bus'})
        text = (ROOT / 'patches/s5_engine/graph_router.py').read_text().lower()
        self.assertNotIn('pci_bus', text)
        self.assertNotIn('cuda.device', text)

    def test_wrong_context_split_epoch_cannot_reuse(self):
        left = make_router()
        right = make_router(
            worker_handle=decode_worker(rank=0, context_id=8),
            banks=[('green44', split_context('green44', context_id=8,
                                             requested=44, measured=44,
                                             pool_id=POOL + 1), (descriptor(),))])
        bank_left = left.router.banks['split']
        bank_right = right.router.banks['green44']
        graph_left = bank_left.graph_for(descriptor(),
                                         storage_epoch=left.handle.storage_epoch,
                                         worker_rank=0)
        graph_right = bank_right.graph_for(descriptor(),
                                           storage_epoch=right.handle.storage_epoch,
                                           worker_rank=0)
        self.assertIsNotNone(graph_left)
        self.assertIsNotNone(graph_right)
        self.assertNotEqual(graph_left.handle, graph_right.handle)
        self.assertNotEqual(graph_left.key, graph_right.key)
        self.assertNotEqual(graph_left.key.context_id, graph_right.key.context_id)
        with self.assertRaises(GraphRoutingError):
            bank_left.graph_for(descriptor(),
                                storage_epoch=left.handle.storage_epoch + 1,
                                worker_rank=0)
        with self.assertRaises(GraphRoutingError):
            bank_left.graph_for(descriptor(),
                                storage_epoch=left.handle.storage_epoch,
                                worker_rank=1)
        self.assertIsNone(bank_left.graph_for(descriptor(tokens=9),
                                              storage_epoch=left.handle.storage_epoch,
                                              worker_rank=0))

    def test_two_splits_in_one_router_never_share_graphs(self):
        rig = make_router(banks=[
            ('green34', split_context('green34', context_id=7, measured=34,
                                      pool_id=0x5000), (descriptor(),)),
            ('green44', split_context('green44', context_id=8, measured=44,
                                      pool_id=0x5001), (descriptor(),)),
        ])
        router = rig.router
        self.assertIsNot(router.banks['green34'].pool,
                         router.banks['green44'].pool)
        left = router.banks['green34'].graphs[descriptor()]
        right = router.banks['green44'].graphs[descriptor()]
        self.assertNotEqual(left.key, right.key)
        self.assertNotEqual(left.handle, right.handle)
        first = transition(rig, 'green34', target_context=7, seq=10)
        self.assertEqual(first.target_mode, 'green34')
        self.assertEqual(router.supported_route(rig.worker.decode.transaction,
                                                descriptor()).bank, 'green34')
        second = router.transition('green44', auth(router, 8, seq=11))
        self.assertEqual(second.target_mode, 'green44')
        self.assertEqual(router.supported_route(rig.worker.decode.transaction,
                                                descriptor()).bank, 'green44')

    def test_descriptor_conversion_and_validation(self):
        dispatched = NS(cg_mode=NS(name='FULL'), num_tokens=4, num_reqs=2,
                        uniform_token_count=2, max_query_len=2,
                        num_active_loras=0, num_ubatches=1)
        converted = ExecutionDescriptor.from_vllm(dispatched)
        self.assertEqual(converted, ExecutionDescriptor('full', 4, 2, 2, 2))
        with self.assertRaises(GraphRoutingError):
            ExecutionDescriptor('FULL', 4, 2)
        with self.assertRaises(GraphRoutingError):
            ExecutionDescriptor('full', True, 2)
        with self.assertRaises(GraphRoutingError):
            ExecutionDescriptor('full', -1, 2)
        with self.assertRaises(GraphRoutingError):
            ExecutionDescriptor.from_vllm(NS(num_tokens=1))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            converted.num_tokens = 5

    def test_compilation_identity_is_strict(self):
        with self.assertRaises(GraphRoutingError):
            CompilationIdentity('', 'full', (2,), 1)
        with self.assertRaises(GraphRoutingError):
            CompilationIdentity('digest', 'full', (2, 2), 1)
        with self.assertRaises(GraphRoutingError):
            CompilationIdentity('digest', 'full', (0,), 1)
        with self.assertRaises(GraphRoutingError):
            CompilationIdentity('digest', 'full', (2,), 0)
        self.assertEqual(compilation(), compilation())


class TestCaptureAndCoverage(unittest.TestCase):
    def test_capture_uses_A_private_storage_and_deterministic_order(self):
        descs = (descriptor(2, 1, 1), descriptor(4, 2, 2))
        rig = make_router(banks=[('split', split_context(), descs)])
        backend = rig.backend
        self.assertEqual(len(backend.capture_calls), 2)
        self.assertEqual(rig.router.banks['split'].coverage, descs)
        for index, (name, key, lane, context, handle) in enumerate(backend.capture_calls):
            self.assertEqual(name, 'split')
            self.assertIs(lane, rig.handle)
            self.assertIs(context, rig.banks[0][1])
            self.assertEqual(key.descriptor, descs[index])
            self.assertEqual(key.storage_epoch, rig.handle.storage_epoch)
        binding = rig.worker.decode.graph_pools[rig.handle.storage_epoch]['split']
        self.assertEqual(binding.address, POOL)
        self.assertEqual(binding.nbytes, RESERVED)
        self.assertEqual(binding.basis, 'estimate')
        coverage = rig.router.descriptor_coverage()
        self.assertEqual(coverage, {'split': descs})
        self.assertEqual(rig.router.status().idle, True)

    def test_per_split_capture_uses_matching_lane_handle(self):
        left = make_router()
        right = make_router(
            worker_handle=decode_worker(rank=3, context_id=8),
            banks=[('green44', split_context('green44', context_id=8,
                                             pool_id=POOL + 1), (descriptor(),))])
        self.assertIs(left.backend.capture_calls[0][2], left.handle)
        self.assertIs(right.backend.capture_calls[0][2], right.handle)
        self.assertIsNot(left.handle, right.handle)
        self.assertNotEqual(left.backend.capture_calls[0][1],
                            right.backend.capture_calls[0][1])
        self.assertEqual(right.backend.capture_calls[0][1].worker_rank, 3)

    def test_duplicate_pool_rejected(self):
        worker, transaction, handle = decode_worker()
        router = GraphRouter(worker, handle, compilation(),
                             backend=FakeBackend(), full_context_id=0,
                             full_pool_id=0x1000)
        router.add_bank('green34', split_context(), descriptors=(descriptor(),))
        with self.assertRaises(GraphRoutingError):
            router.add_bank('green44', split_context('green44', context_id=8,
                                                     pool_id=POOL),
                            descriptors=(descriptor(),))

    def test_context_identity_collisions_rejected(self):
        worker, transaction, handle = decode_worker()
        router = GraphRouter(worker, handle, compilation(),
                             backend=FakeBackend(), full_context_id=0,
                             full_pool_id=0x1000)
        with self.assertRaises(GraphRoutingError):
            router.add_bank('split', split_context(context_id=0),
                            descriptors=(descriptor(),))
        router.add_bank('split', split_context(), descriptors=(descriptor(),))
        with self.assertRaises(GraphRoutingError):
            router.add_bank('other', split_context('other', pool_id=0x6000),
                            descriptors=(descriptor(),))

    def test_capture_failure_quarantines_and_is_not_idle(self):
        backend = FakeBackend()
        backend.fail_capture_at = 1
        rig = make_router(banks=[('split', split_context(),
                                  (descriptor(2), descriptor(4)))],
                          backend=backend, capture=False)
        with self.assertRaises(RuntimeError):
            rig.router.capture_at_startup()
        bank = rig.router.banks['split']
        self.assertEqual(bank.state, 'quarantined')
        self.assertIn('capture failed', bank.quarantine_reason)
        status = rig.router.status()
        self.assertTrue(status.unclean)
        self.assertFalse(status.idle)
        with self.assertRaises(GraphUnclean):
            rig.router.supported_route(rig.worker.decode.transaction, descriptor())
        with self.assertRaises(GraphUnclean):
            rig.router.transition('ordinary', auth(rig.router, 0, seq=10))
        # Teardown is still explicit and destroys the partial graph.
        record = rig.router.shutdown()
        self.assertIn('graph', record.order)
        self.assertTrue(rig.router.status().unclean)
        self.assertFalse(rig.router.status().idle)

    def test_double_capture_rejected(self):
        rig = make_router()
        with self.assertRaises(GraphRoutingError):
            rig.router.capture_at_startup()

    def test_empty_or_duplicate_declarations_rejected(self):
        worker, transaction, handle = decode_worker()
        router = GraphRouter(worker, handle, compilation(),
                             backend=FakeBackend(), full_context_id=0,
                             full_pool_id=0x1000)
        with self.assertRaises(GraphRoutingError):
            router.add_bank('split', split_context(), descriptors=())
        with self.assertRaises(GraphRoutingError):
            router.add_bank('split', split_context(),
                            descriptors=(descriptor(), descriptor()))
        with self.assertRaises(GraphRoutingError):
            router.add_bank('ordinary', split_context(), descriptors=(descriptor(),))

    def test_capture_requires_backend_and_budget(self):
        worker, transaction, handle = decode_worker()
        router = GraphRouter(worker, handle, compilation(),
                             full_context_id=0, full_pool_id=0x1000)
        with self.assertRaises(GraphRoutingError):
            router.add_bank('split', split_context(), descriptors=(descriptor(),))
        poor = split_context(reserved=0)
        with self.assertRaises(GraphRoutingError):
            router.add_bank('split', poor, descriptors=(descriptor(),))

    def test_router_requires_live_decode_lane_handle(self):
        worker, transaction, handle = decode_worker()
        foreign = LaneHandle(worker.decode.lane_id, object())
        with self.assertRaises(GraphRoutingError):
            GraphRouter(worker, foreign, compilation(), backend=FakeBackend(),
                        full_context_id=0, full_pool_id=0x1000)
        with self.assertRaises(GraphRoutingError):
            GraphRouter(worker, handle, compilation(),
                        backend=NS(), full_context_id=0, full_pool_id=0x1000)


class TestRouting(unittest.TestCase):
    def test_ordinary_when_no_prefill(self):
        rig = make_router()
        decision = rig.router.supported_route(rig.worker.decode.transaction,
                                              descriptor())
        self.assertEqual(decision.mode, 'ordinary')
        self.assertIsNone(decision.bank)
        self.assertIn('no active prefill', decision.reason)

    def test_split_route_after_explicit_transition(self):
        rig = make_router()
        transition(rig)
        decision = rig.router.supported_route(rig.worker.decode.transaction,
                                              descriptor())
        self.assertEqual(decision.mode, 'split')
        self.assertEqual(decision.bank, 'split')

    def test_uncaptured_shape_requires_explicit_ordinary_transition(self):
        rig = make_router()
        transition(rig)
        missing = descriptor(tokens=9)
        with self.assertRaises(UnsupportedRoute):
            rig.router.supported_route(rig.worker.decode.transaction, missing)
        transition(rig, 'ordinary', target_context=0, seq=11)
        self.assertEqual(rig.router.supported_route(
            rig.worker.decode.transaction, missing).mode, 'ordinary')
        self.assertEqual(rig.backend.replay_calls, [])

    def test_unsupported_route_fails_closed_during_prefill(self):
        rig = make_router()
        activate_prefill(rig.worker)
        with self.assertRaises(UnsupportedRoute):
            rig.router.supported_route(rig.worker.decode.transaction, descriptor())
        transition(rig)
        with self.assertRaises(UnsupportedRoute):
            rig.router.supported_route(rig.worker.decode.transaction,
                                       descriptor(tokens=9))
        decision = rig.router.supported_route(rig.worker.decode.transaction,
                                              descriptor())
        self.assertEqual(decision.mode, 'split')

    def test_prefill_transaction_is_never_a_decode_route(self):
        rig = make_router()
        prefill_transaction, _, _ = activate_prefill(rig.worker)
        transition(rig)
        with self.assertRaises(UnsupportedRoute):
            rig.router.supported_route(prefill_transaction, descriptor(mode='piecewise'))
        with self.assertRaises(UnsupportedRoute):
            rig.router.supported_route(prefill_transaction, descriptor())

    def test_stale_lane_identity_refuses_route(self):
        rig = make_router()
        transition(rig)
        # A release cannot happen while graph pools remain; simulate the only
        # way the handle can change (C re-reserving without retiring banks).
        rig.worker.decode.handle = LaneHandle(rig.worker.decode.lane_id, object())
        with self.assertRaises(GraphUnclean):
            rig.router.supported_route(rig.worker.decode.transaction, descriptor())

    def test_route_requires_transaction_and_descriptor_types(self):
        rig = make_router()
        with self.assertRaises(GraphRoutingError):
            rig.router.supported_route('not-a-transaction', descriptor())
        with self.assertRaises(GraphRoutingError):
            rig.router.supported_route(rig.worker.decode.transaction, 'not-a-descriptor')


class TestReplayCompletion(unittest.TestCase):
    def split_rig(self):
        rig = make_router()
        transition(rig)
        return rig

    def test_replay_is_enqueue_not_completion(self):
        rig = self.split_rig()
        result, ticket = replay(rig)
        self.assertIsInstance(result, ReplayResult)
        self.assertEqual(result.output.shape, (1, 2))
        self.assertEqual(rig.router.pending_replays, 1)
        self.assertEqual(rig.router.timing_records, ())
        self.assertTrue(rig.router.banks['split'].pool.active)
        self.assertFalse(rig.router.status().idle)
        self.assertEqual(len(rig.backend.replay_calls), 1)
        self.assertEqual(rig.backend.replay_calls[0][0][0], 'graph')
        self.assertIs(rig.backend.replay_calls[0][2], rig.handle)

    def test_replay_ticket_validation(self):
        rig = self.split_rig()
        worker = rig.worker
        transaction = worker.decode.transaction
        other = worker.new_transaction('other-request', 1, 0, 1)
        cases = [
            decode_ticket(worker, other),
            OperationTicket(transaction, worker.worker_rank,
                            e3_types.LaneId(worker.worker_rank, LaneRole.PREFILL,
                                            rig.handle.storage_epoch, 9),
                            0, DependencyRole.CONSUMER),
            OperationTicket(transaction, worker.worker_rank,
                            worker.decode.lane_id, 0, DependencyRole.PRODUCER),
        ]
        for ticket in cases:
            with self.assertRaises(GraphRoutingError):
                rig.router.replay(transaction, descriptor(), ticket,
                                  observed_context=CONTEXT_ID,
                                  observed_stream=STREAM)
        unsubmitted = decode_ticket(worker, transaction)
        with self.assertRaises(GraphRoutingError):
            rig.router.replay(transaction, descriptor(), unsubmitted,
                              observed_context=CONTEXT_ID,
                              observed_stream=STREAM)

    def test_replay_wrong_context_or_stream_refused(self):
        rig = self.split_rig()
        worker = rig.worker
        ticket = decode_ticket(worker, worker.decode.transaction)
        worker.decode.submit(ticket)
        with self.assertRaises(GraphRoutingError):
            rig.router.replay(worker.decode.transaction, descriptor(), ticket,
                              observed_context=CONTEXT_ID + 1,
                              observed_stream=STREAM)
        with self.assertRaises(GraphRoutingError):
            rig.router.replay(worker.decode.transaction, descriptor(), ticket,
                              observed_context=CONTEXT_ID,
                              observed_stream=STREAM + 1)
        self.assertFalse(rig.router.banks['split'].pool.active)

    def test_completion_requires_recorded_gpu_event(self):
        rig = self.split_rig()
        result, ticket = replay(rig)
        unrecorded = CompletionProof(ticket, CompletionKind.GPU_EVENT, 0)
        with self.assertRaises(GraphRoutingError):
            rig.router.complete_replay(result, unrecorded)
        self.assertTrue(rig.router.status().unclean)
        self.assertFalse(rig.router.status().idle)
        self.assertTrue(rig.router.banks['split'].pool.active)

    def test_completion_wrong_kind_quarantines(self):
        rig = self.split_rig()
        result, ticket = replay(rig)
        finish(rig.worker, ticket, 0, CompletionKind.COPY_EVENT)
        proof = rig.worker.decode.completed[-1]
        with self.assertRaises(GraphRoutingError):
            rig.router.complete_replay(result, proof)
        self.assertTrue(rig.router.status().unclean)

    def test_completion_wrong_ticket_quarantines(self):
        rig = self.split_rig()
        result, ticket = replay(rig)
        finish(rig.worker, ticket, 0)
        other = OperationTicket(ticket.transaction, ticket.worker_rank,
                                ticket.lane, 1, ticket.dependency_role)
        with self.assertRaises(GraphRoutingError):
            rig.router.complete_replay(
                result, CompletionProof(other, CompletionKind.GPU_EVENT, 1))
        self.assertTrue(rig.router.status().unclean)

    def test_valid_completion_releases_pool_and_records_timing(self):
        rig = self.split_rig()
        result, ticket = replay(rig)
        finish(rig.worker, ticket, 0)
        record = rig.router.complete_replay(
            result, rig.worker.decode.completed[-1], gpu_elapsed_ns=1_234)
        self.assertEqual(record.lease_id, result.lease.lease_id)
        self.assertEqual(record.context_id, CONTEXT_ID)
        self.assertEqual(record.storage_epoch, rig.handle.storage_epoch)
        self.assertEqual(record.worker_rank, 0)
        self.assertEqual(record.host_submit_seq, 1)
        self.assertEqual(record.completion_seq, 0)
        self.assertEqual(record.gpu_elapsed_ns, 1_234)
        self.assertEqual(record.basis, 'measured')
        self.assertEqual(rig.router.pending_replays, 0)
        self.assertFalse(rig.router.banks['split'].pool.active)
        self.assertEqual(len(rig.router.timing_records), 1)
        self.assertTrue(rig.router.status().idle)
        with self.assertRaises(GraphRoutingError):
            rig.router.complete_replay(result, rig.worker.decode.completed[-1])

    def test_unmeasured_timing_is_explicit(self):
        rig = self.split_rig()
        result, ticket = replay(rig)
        finish(rig.worker, ticket, 0)
        record = rig.router.complete_replay(result, rig.worker.decode.completed[-1])
        self.assertIsNone(record.gpu_elapsed_ns)
        self.assertEqual(record.basis, 'unmeasured')

    def test_single_replay_per_pool(self):
        rig = self.split_rig()
        first, first_ticket = replay(rig)
        second_ticket = decode_ticket(rig.worker, rig.worker.decode.transaction,
                                      op_seq=1)
        rig.worker.decode.submit(second_ticket)
        with self.assertRaises(E3Pending):
            rig.router.replay(rig.worker.decode.transaction, descriptor(),
                              second_ticket, observed_context=CONTEXT_ID,
                              observed_stream=STREAM)
        self.assertEqual(rig.router.pending_replays, 1)
        finish(rig.worker, first_ticket, 0)
        rig.router.complete_replay(first, rig.worker.decode.completed[-1])
        second = rig.router.replay(rig.worker.decode.transaction, descriptor(),
                                   second_ticket, observed_context=CONTEXT_ID,
                                   observed_stream=STREAM)
        self.assertEqual(rig.router.pending_replays, 1)
        self.assertEqual(second.lease.host_submit_seq, 2)
        self.assertEqual(len(rig.backend.capture_calls), 1)
        self.assertEqual(rig.backend.replay_calls[1][0],
                         rig.backend.replay_calls[0][0])

    def test_enqueue_failure_quarantines_without_false_idle(self):
        rig = self.split_rig()
        rig.backend.fail_replay = True
        ticket = decode_ticket(rig.worker, rig.worker.decode.transaction)
        rig.worker.decode.submit(ticket)
        with self.assertRaises(RuntimeError):
            rig.router.replay(rig.worker.decode.transaction, descriptor(), ticket,
                              observed_context=CONTEXT_ID,
                              observed_stream=STREAM)
        self.assertTrue(rig.router.status().unclean)
        self.assertFalse(rig.router.status().idle)
        self.assertEqual(rig.router.pending_replays, 1)
        self.assertTrue(rig.router.banks['split'].pool.active)
        with self.assertRaises(E3Pending):
            rig.router.shutdown()
        proof = finish(rig.worker, ticket)
        rig.router.shutdown(proofs=(proof,))

    def test_cancel_without_proof_is_unclean_and_pool_retained(self):
        rig = self.split_rig()
        result, ticket = replay(rig)
        self.assertFalse(rig.router.cancel_replay(result))
        self.assertTrue(rig.router.status().unclean)
        self.assertFalse(rig.router.status().idle)
        self.assertTrue(rig.router.banks['split'].pool.active)
        self.assertEqual(rig.router.pending_replays, 1)
        self.assertEqual(rig.router.timing_records, ())
        with self.assertRaises(E3Pending):
            rig.router.shutdown()
        finish(rig.worker, ticket, 0)
        self.assertTrue(rig.router.cancel_replay(result, rig.worker.decode.completed[-1]))
        self.assertEqual(rig.router.pending_replays, 0)
        self.assertFalse(rig.router.banks['split'].pool.active)
        rig.router.shutdown()
        self.assertFalse(rig.router.status().idle)


class TestTransitions(unittest.TestCase):
    def test_transition_needs_explicit_authorization(self):
        rig = make_router()
        with self.assertRaises(GraphRoutingError):
            rig.router.transition('split', None)
        with self.assertRaises(GraphRoutingError):
            rig.router.transition('split', NS(engine_epoch=1))
        with self.assertRaises(GraphRoutingError):
            rig.router.transition('split', auth(rig.router, engine=2))
        with self.assertRaises(GraphRoutingError):
            rig.router.transition('split', auth(rig.router, target_context=8))
        with self.assertRaises(GraphRoutingError):
            rig.router.transition('split', auth(rig.router, storage=999))
        with self.assertRaises(GraphRoutingError):
            rig.router.transition('nowhere', auth(rig.router))
        self.assertEqual(rig.router.mode, 'ordinary')

    def test_transition_authorization_must_advance(self):
        rig = make_router()
        transition(rig, seq=10)
        with self.assertRaises(GraphRoutingError):
            rig.router.transition('ordinary', auth(rig.router, 0, seq=10))
        record = rig.router.transition('ordinary', auth(rig.router, 0, seq=11))
        self.assertEqual(record.transition_seq, 2)
        self.assertEqual(record.source_mode, 'split')
        self.assertEqual(record.target_mode, 'ordinary')

    def test_transition_rejects_outstanding_replay(self):
        rig = make_router()
        transition(rig)
        result, ticket = replay(rig)
        with self.assertRaises(E3Pending):
            rig.router.transition('ordinary', auth(rig.router, 0, seq=11))
        self.assertEqual(rig.router.mode, 'split')
        finish(rig.worker, ticket, 0)
        rig.router.complete_replay(result, rig.worker.decode.completed[-1])
        record = rig.router.transition('ordinary', auth(rig.router, 0, seq=11))
        self.assertEqual(record.target_mode, 'ordinary')

    def test_transition_rejects_copy_and_pp_users(self):
        rig = make_router()
        rig.router.register_external_user('copy', 'copy-token')
        with self.assertRaises(E3Pending):
            transition(rig)
        self.assertEqual(rig.router.mode, 'ordinary')
        rig.router.release_external_user('copy', 'copy-token')
        rig.router.register_external_user('pp', 'pp-slot')
        with self.assertRaises(E3Pending):
            transition(rig)
        self.assertEqual(rig.router.mode, 'ordinary')
        rig.router.release_external_user('pp', 'pp-slot')
        self.assertEqual(transition(rig).target_mode, 'split')
        self.assertEqual(len(rig.router.external), 0)
        self.assertEqual(rig.router.external.snapshot(), ())

    def test_transition_rejects_worker_operations(self):
        rig = make_router()
        ticket = decode_ticket(rig.worker, rig.worker.decode.transaction)
        rig.worker.decode.submit(ticket)
        with self.assertRaises(E3Pending):
            transition(rig)
        finish(rig.worker, ticket, 0)
        prefill_transaction, _, _ = activate_prefill(rig.worker)
        prefill = prefill_ticket(rig.worker, prefill_transaction)
        rig.worker.prefill.submit(prefill)
        with self.assertRaises(E3Pending):
            transition(rig)
        finish(rig.worker, prefill, 0, lane=rig.worker.prefill)
        self.assertEqual(transition(rig).target_mode, 'split')

    def test_capture_in_progress_blocks_transition(self):
        rig = make_router()
        rig.router.banks['split'].state = 'capturing'
        with self.assertRaises(E3Pending):
            rig.router.transition('ordinary', auth(rig.router, 0))

    def test_target_bank_must_be_captured(self):
        rig = make_router(capture=False)
        with self.assertRaises(GraphRoutingError):
            transition(rig)
        self.assertEqual(rig.router.mode, 'ordinary')

    def test_unknown_external_kind_rejected(self):
        rig = make_router()
        with self.assertRaises(GraphRoutingError):
            rig.router.register_external_user('prefill', 'x')
        rig.router.register_external_user('copy', 'x')
        with self.assertRaises(GraphRoutingError):
            rig.router.release_external_user('copy', 'missing')


class TestTeardown(unittest.TestCase):
    def test_shutdown_refuses_outstanding_replay(self):
        rig = make_router()
        transition(rig)
        result, ticket = replay(rig)
        with self.assertRaises(E3Pending):
            rig.router.shutdown()
        self.assertFalse(rig.backend.destroyed)
        self.assertTrue(rig.router.status().shutdown_pending)
        self.assertFalse(rig.router.status().idle)
        finish(rig.worker, ticket, 0)
        rig.router.complete_replay(result, rig.worker.decode.completed[-1])
        record = rig.router.shutdown()
        self.assertIn('graph', record.order)
        self.assertTrue(rig.router.status().idle)

    def test_shutdown_refuses_external_users(self):
        rig = make_router()
        rig.router.register_external_user('copy', 'copy')
        with self.assertRaises(E3Pending):
            rig.router.shutdown()
        self.assertFalse(rig.backend.destroyed)
        rig.router.release_external_user('copy', 'copy')
        rig.router.shutdown()
        self.assertTrue(rig.router.status().idle)

    def test_shutdown_order_and_transport(self):
        released = []
        rig = make_router(banks=[('split', split_context(),
                                  (descriptor(2), descriptor(4)))],
                          transport_release=lambda: released.append('transport'))
        bank = rig.router.banks['split']
        bank.register_event('event-1')
        bank.register_event('event-2')
        bank.register_stream('stream-1')
        handles = [graph.handle for graph in bank.graphs.values()]
        record = rig.router.shutdown()
        self.assertEqual(rig.backend.destroyed, [
            ('graph', handles[1]), ('graph', handles[0]),
            ('event', 'event-2'), ('event', 'event-1'),
            ('stream', 'stream-1'),
        ])
        self.assertEqual(record.order, ('graph', 'graph', 'event', 'event',
                                        'stream', 'pool', 'free', 'transport'))
        self.assertEqual(released, ['transport'])
        self.assertTrue(rig.router.status().idle)
        rig.worker.decode.release()
        self.assertEqual(rig.worker.decode.state, 'free')

    def test_shutdown_before_capture_frees_registered_pool(self):
        rig = make_router(capture=False)
        record = rig.router.shutdown()
        self.assertEqual(record.order, ('pool', 'free'))
        self.assertEqual(rig.router.banks['split'].pool.state, 'freed')
        self.assertTrue(rig.router.status().idle)

    def test_shutdown_is_idempotent(self):
        rig = make_router()
        first = rig.router.shutdown()
        self.assertEqual(rig.router.shutdown(), first)
        self.assertEqual(len(rig.backend.destroyed), 1)

    def test_destroy_failure_quarantines_and_keeps_pool(self):
        rig = make_router()
        rig.backend.fail_destroy = True
        with self.assertRaises(RuntimeError):
            rig.router.shutdown()
        status = rig.router.status()
        self.assertTrue(status.unclean)
        self.assertFalse(status.idle)
        self.assertEqual(rig.router.banks['split'].pool.state, 'idle')
        with self.assertRaises(E3Pending):
            rig.worker.decode.release()

    def test_no_free_or_rebind_before_completion(self):
        rig = make_router()
        transition(rig)
        result, ticket = replay(rig)
        bank = rig.router.banks['split']
        pool = bank.pool
        with self.assertRaises(E3Pending):
            pool.destroy()
        with self.assertRaises(GraphRoutingError):
            pool.free()
        with self.assertRaises(E3Pending):
            pool.rebind(rig.handle.storage_epoch + 1)
        with self.assertRaises(E3Pending):
            bank.rebind(LaneHandle(rig.worker.decode.lane_id, object()))
        with self.assertRaises(E3Pending):
            rig.router.shutdown()
        finish(rig.worker, ticket, 0)
        rig.router.complete_replay(result, rig.worker.decode.completed[-1])
        rig.router.shutdown()
        pool.rebind(rig.handle.storage_epoch + 1)
        self.assertEqual(pool.owner_epoch, rig.handle.storage_epoch + 1)

    def test_empty_bank_cannot_be_rebound_with_graphs(self):
        rig = make_router()
        bank = rig.router.banks['split']
        with self.assertRaises(E3Pending):
            bank.rebind(LaneHandle(rig.worker.decode.lane_id, object()))
        empty = make_router(capture=False).router.banks['split']
        replacement = LaneHandle(rig.worker.decode.lane_id, object())
        self.assertIs(empty.rebind(replacement), replacement)

    def test_shutdown_in_quarantine_finishes_storage_without_false_idle(self):
        backend = FakeBackend()
        backend.fail_capture_at = 1
        rig = make_router(banks=[('split', split_context(),
                                  (descriptor(2), descriptor(4)))],
                          backend=backend, capture=False)
        with self.assertRaises(RuntimeError):
            rig.router.capture_at_startup()
        rig.router.shutdown()
        self.assertFalse(rig.router.status().idle)
        self.assertTrue(rig.router.status().unclean)
        self.assertEqual(rig.router.banks['split'].pool.state, 'freed')


class TestBudget(unittest.TestCase):
    def test_report_covers_full_and_split_banks(self):
        rig = make_router()
        report = rig.router.budget(reserve_measured_bytes=150)
        self.assertEqual([bank.kind for bank in report.banks], ['full', 'split'])
        full = report.banks[0]
        split = report.banks[1]
        self.assertEqual(full.allocated_bytes, GRAPH_POOL_MEASURED_BYTES)
        self.assertEqual(full.basis, 'measured')
        self.assertFalse(full.new_allocation)
        self.assertIsNone(split.allocated_bytes)
        self.assertEqual(split.reserved_bytes, RESERVED)
        self.assertEqual(split.basis, 'estimate')
        self.assertTrue(split.new_allocation)
        self.assertEqual(report.existing_bytes, GRAPH_POOL_MEASURED_BYTES)
        self.assertEqual(report.new_bytes, RESERVED)
        self.assertEqual(report.reserve_required_bytes, 100)

    def test_reserve_shortfall_grants_no_new_storage(self):
        rig = make_router()
        report = rig.router.budget(reserve_measured_bytes=50)
        self.assertEqual(report.funding, 'shortfall')
        self.assertFalse(report.new_allowed)
        with self.assertRaises(GraphRoutingError):
            report.require_funded()
        rig.router.banks['split'].allocated_bytes = 1_000
        report = rig.router.budget(reserve_measured_bytes=50)
        with self.assertRaises(GraphRoutingError):
            report.require_bounded(10_000_000)
        self.assertEqual(report.new_bytes, RESERVED)

    def test_unmeasured_reserve_grants_no_new_storage(self):
        rig = make_router()
        report = rig.router.budget()
        self.assertEqual(report.funding, 'unmeasured')
        self.assertFalse(report.new_allowed)
        with self.assertRaises(GraphRoutingError):
            report.require_funded()

    def test_measured_reserve_funds_and_bounds(self):
        rig = make_router()
        rig.router.banks['split'].allocated_bytes = 4_000_000
        rig.router.banks['split'].context = dataclasses.replace(rig.context, basis='measured')
        report = rig.router.budget(reserve_measured_bytes=150)
        self.assertEqual(report.funding, 'funded')
        self.assertTrue(report.new_allowed)
        report.require_funded().require_bounded(10_000_000)
        with self.assertRaises(GraphRoutingError):
            report.require_bounded(7_999_999)

    def test_bounded_refuses_unmeasured_new_allocation(self):
        rig = make_router()
        report = rig.router.budget(reserve_measured_bytes=150)
        with self.assertRaises(GraphRoutingError):
            report.require_bounded(10_000_000)
        rig.router.banks['split'].allocated_bytes = 4_000_000
        report = rig.router.budget(reserve_measured_bytes=150)
        with self.assertRaises(GraphRoutingError):
            report.require_bounded(10_000_000)
        rig.router.banks['split'].context = dataclasses.replace(rig.context, basis='measured')
        rig.router.budget(reserve_measured_bytes=150).require_bounded(10_000_000)

    def test_duplicate_pool_budget_rejected(self):
        with self.assertRaises(GraphRoutingError):
            GraphBudgetReport((
                BankBudget('a', 'split', 1, 5, 1, 1, 'estimate', True),
                BankBudget('b', 'split', 2, 5, 1, 1, 'estimate', True),
            ), 0, None)
        with self.assertRaises(GraphRoutingError):
            BankBudget('a', 'other', 1, 5, 1, 1, 'estimate', True)
        with self.assertRaises(GraphRoutingError):
            GraphBudgetReport((), -1, None)


class TestTiming(unittest.TestCase):
    def test_no_timing_record_before_completion(self):
        rig = make_router()
        transition(rig)
        result, ticket = replay(rig)
        self.assertEqual(rig.router.timing_records, ())
        self.assertEqual(rig.router.timing_for(CONTEXT_ID), ())
        finish(rig.worker, ticket, 0)
        rig.router.complete_replay(result, rig.worker.decode.completed[-1])
        self.assertEqual(len(rig.router.timing_for(CONTEXT_ID)), 1)
        self.assertEqual(rig.router.timing_for(CONTEXT_ID + 1), ())

    def test_timing_records_have_no_cross_context_anchor(self):
        names = {field.name for field in dataclasses.fields(TimingRecord)}
        self.assertEqual(names, {'lease_id', 'context_id', 'storage_epoch',
                                 'worker_rank', 'host_submit_seq',
                                 'completion_seq', 'gpu_elapsed_ns', 'basis'})
        self.assertFalse(names & {'timestamp', 'anchor', 'clock', 'epoch_ns'})

    def test_no_forbidden_instrumentation_in_router(self):
        forbidden = ('synchronize(', 'globaltimer', 'perf_counter', 'time.time(',
                     'cudaDeviceSynchronize', 'cudaStreamSynchronize',
                     'get_prev_sampled_outputs', 'torch.cuda', 'import torch',
                     'wait_event(')
        text = (ROOT / 'patches/s5_engine/graph_router.py').read_text()
        for token in forbidden:
            self.assertNotIn(token, text, f'graph_router.py contains {token!r}')
        hook_text = (ROOT / 'patches/s5_engine/graph_hooks.py').read_text()
        for token in ('globaltimer', 'perf_counter', 'time.time(',
                      'cudaDeviceSynchronize', 'cudaStreamSynchronize',
                      'get_prev_sampled_outputs', 'torch.cuda', 'wait_event('):
            self.assertNotIn(token, hook_text, f'graph_hooks.py contains {token!r}')

    def test_independent_routers_overlap_without_pool_alias(self):
        barrier = threading.Barrier(2)

        class BarrierBackend(FakeBackend):
            def replay(self, handle, key, lane, context):
                self.replay_calls.append((handle, key, lane, context))
                barrier.wait(timeout=5)
                return FakeOutput(key.descriptor.num_tokens)

        backend = BarrierBackend()
        left = make_router(backend=backend,
                           worker_handle=decode_worker(rank=0, context_id=7),
                           banks=[('green34', split_context('green34', 7, 34, 34,
                                                            0x5000), (descriptor(),))])
        right = make_router(backend=backend,
                            worker_handle=decode_worker(rank=1, context_id=8),
                            banks=[('green44', split_context('green44', 8, 44, 44,
                                                             0x5001), (descriptor(),))])
        transition(left, 'green34', target_context=7)
        transition(right, 'green44', target_context=8)
        left_ticket = decode_ticket(left.worker, left.worker.decode.transaction)
        right_ticket = decode_ticket(right.worker, right.worker.decode.transaction)
        left.worker.decode.submit(left_ticket)
        right.worker.decode.submit(right_ticket)
        results, errors = {}, []

        def run(name, rig, ticket):
            try:
                results[name] = rig.router.replay(
                    rig.worker.decode.transaction, descriptor(), ticket,
                    observed_context=rig.banks[0][1].context_id,
                    observed_stream=rig.banks[0][1].stream_token)
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append((name, exc))

        threads = [threading.Thread(target=run, args=('left', left, left_ticket)),
                   threading.Thread(target=run, args=('right', right, right_ticket))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertEqual(errors, [])
        self.assertEqual(set(results), {'left', 'right'})
        self.assertTrue(left.router.banks['green34'].pool.active)
        self.assertTrue(right.router.banks['green44'].pool.active)
        self.assertNotEqual(left.router.banks['green34'].pool.pool_id,
                            right.router.banks['green44'].pool.pool_id)
        finish(left.worker, left_ticket, 0)
        left.router.complete_replay(results['left'],
                                    left.worker.decode.completed[-1])
        self.assertEqual(right.router.pending_replays, 1)
        self.assertEqual(right.router.timing_records, ())
        finish(right.worker, right_ticket, 0)
        right.router.complete_replay(results['right'],
                                     right.worker.decode.completed[-1])
        self.assertEqual(len(left.router.timing_records), 1)
        self.assertEqual(len(right.router.timing_records), 1)


class TestSerializedStorage(unittest.TestCase):
    def test_shared_serialized_storage_is_real_exclusion(self):
        worker, decode_transaction, handle = decode_worker()
        shared_decode = ScratchSpec('staged_decode', 'shared', 'serialized', 4096,
                                    shared=True,
                                    reuse_condition='one staged write at a time')
        shared_prefill = ScratchSpec('staged_prefill', 'shared', 'serialized', 4096,
                                     shared=True,
                                     reuse_condition='one staged write at a time')
        worker.decode.bind_scratch(shared_decode, 2_000_000)
        prefill_transaction, snapshot, prefill_handle = activate_prefill(worker)
        worker.prefill.bind_scratch(shared_prefill, 2_000_000)
        self.assertEqual(worker.decode.buffers['staged_decode'].address,
                         worker.prefill.buffers['staged_prefill'].address)
        GraphRouter(worker, handle, compilation(), backend=FakeBackend(),
                    full_context_id=0, full_pool_id=0x1000)
        prefill = prefill_ticket(worker, prefill_transaction)
        worker.prefill.submit(prefill)
        decode = decode_ticket(worker, decode_transaction)
        with self.assertRaises(E3Pending):
            worker.decode.submit(decode)
        self.assertEqual(worker.decode.outstanding, {})
        finish(worker, prefill, 0, lane=worker.prefill)
        worker.decode.submit(decode)
        self.assertIn(decode, worker.decode.outstanding)
        finish(worker, decode, 0)
        self.assertEqual(worker.decode.outstanding, {})


class TestGeneratedSource(unittest.TestCase):
    def setUp(self):
        self.sources = a_sources()

    def test_pins_accept_only_reviewed_A_compositions(self):
        for kind, pinned in graph_hooks.A_SOURCE_SHA256.items():
            self.assertEqual(sha(self.sources[kind]), pinned)
            self.assertEqual(graph_hooks.source_kind(self.sources[kind]), kind)
        with self.assertRaises(ValueError):
            graph_hooks.source_kind(self.sources['pinned'])
        execution = runner_hooks.runner_execution(self.sources['pinned'])
        with self.assertRaises(ValueError):
            graph_hooks.source_kind(execution)
        with self.assertRaises(ValueError):
            graph_hooks.source_kind(self.sources['e3'] + '\n')
        generated = graph_hooks.generate(self.sources['e3'])
        with self.assertRaises(ValueError):
            graph_hooks.source_kind(generated)
        with self.assertRaises(ValueError):
            graph_hooks.generate(generated)

    def test_generate_and_revert_exact_for_every_composition(self):
        for kind, a_source in self.sources.items():
            if kind == 'pinned':
                continue
            generated = graph_hooks.generate(a_source)
            ast.parse(generated)
            self.assertEqual(graph_hooks.revert(generated), a_source)
            self.assertEqual(graph_hooks.revert(generated, expected=a_source), a_source)
            with self.assertRaises(ValueError):
                graph_hooks.revert(generated, expected=self.sources['pinned'])
            self.assertEqual(generated.count('_flash_e3_graph.'), 4)
            self.assertEqual(generated.count('_flash_e3_hooks.'), 13)
            self.assertEqual(generated.count('_flash_e3_graph.shutdown_banks(self)'), 1)
        with self.assertRaises(ValueError):
            graph_hooks.revert(self.sources['e3'])

    def test_anchor_drift_and_duplication_rejected(self):
        a_source = self.sources['e3']
        mutated = a_source.replace(
            graph_hooks._CHECK_OLD,
            '        if batch_desc.num_tokens == 1:\n', 1)
        with mock.patch.dict(graph_hooks.A_SOURCE_SHA256,
                             {'e3': sha(mutated)}, clear=True):
            with self.assertRaises(ValueError):
                graph_hooks.generate(mutated)
        duplicated = a_source.replace(
            graph_hooks._CHECK_OLD,
            graph_hooks._CHECK_OLD + graph_hooks._CHECK_OLD, 1)
        with mock.patch.dict(graph_hooks.A_SOURCE_SHA256,
                             {'e3': sha(duplicated)}, clear=True):
            with self.assertRaises(ValueError):
                graph_hooks.generate(duplicated)
        no_import = a_source.replace(graph_hooks._IMPORT_OLD, '', 1)
        with mock.patch.dict(graph_hooks.A_SOURCE_SHA256,
                             {'e3': sha(no_import)}, clear=True):
            with self.assertRaises(ValueError):
                graph_hooks.generate(no_import)
        e2 = self.sources['e3-e2-exec']
        no_shutdown = e2.replace(graph_hooks._SHUTDOWN_E2_OLD,
                                 '        torch.accelerator.synchronize()\n', 1)
        with mock.patch.dict(graph_hooks.A_SOURCE_SHA256,
                             {'e3-e2-exec': sha(no_shutdown)}, clear=True):
            with self.assertRaises(ValueError):
                graph_hooks.generate(no_shutdown)

    def test_placement_and_order(self):
        for kind, a_source in self.sources.items():
            if kind == 'pinned':
                continue
            generated = graph_hooks.generate(a_source)
            check = generated.index('_flash_e3_graph.check_route(')
            zero = generated.index('        if batch_desc.num_tokens == 0:\n')
            self.assertLess(check, zero)
            self.assertEqual(generated.count(
                '            model_output = _flash_e3_graph.replay_fullgraph('), 1)
            self.assertEqual(generated.count('self.cudagraph_manager.run_fullgraph'), 1)
            self.assertNotIn(graph_hooks._REPLAY_OLD, generated)
            capture = generated.index('_flash_e3_graph.capture_banks(self)')
            stock_capture = generated.index(
                '                    self.cudagraph_manager.capture(')
            self.assertLess(stock_capture, capture)
            shutdown_source = method_source(generated, 'shutdown')
            self.assertIn('_flash_e3_graph.shutdown_banks(self)', shutdown_source)
            sync = shutdown_source.index('torch.accelerator.synchronize()')
            hook = shutdown_source.index('_flash_e3_graph.shutdown_banks(self)')
            self.assertLess(sync, hook)
            if '_flash_e2_lifecycle' in a_source:
                self.assertLess(hook, shutdown_source.index('_e2 = getattr'))
            else:
                self.assertLess(hook, shutdown_source.index('self.cudagraph_manager = None'))

    def test_composes_with_A_without_replacing_context_management(self):
        for kind, a_source in self.sources.items():
            if kind == 'pinned':
                continue
            generated = graph_hooks.generate(a_source)
            self.assertEqual(generated.count('_flash_e3_hooks.execute_with_e3'), 1)
            self.assertEqual(generated.count('set_forward_context'),
                             a_source.count('set_forward_context'))
            self.assertEqual(generated.count('get_prev_sampled_outputs'),
                             a_source.count('get_prev_sampled_outputs'))
            self.assertEqual(generated.count('pp_handler'),
                             a_source.count('pp_handler'))
            self.assertEqual(generated.count('_swap_prefill'),
                             a_source.count('_swap_prefill'))
            self.assertEqual(method_source(generated, 'execute_model'),
                             method_source(a_source, 'execute_model'))
            self.assertEqual(method_source(generated, 'sample_tokens'),
                             method_source(a_source, 'sample_tokens'))
            self.assertEqual(generated.count('current_stream'),
                             a_source.count('current_stream'))
            self.assertEqual(generated.count('HostDispatcher'),
                             a_source.count('HostDispatcher'))
            self.assertEqual(generated.count('StreamContext'),
                             a_source.count('StreamContext'))
            self.assertEqual(generated.count('_flash_e3_graph.shutdown_banks(self)'), 1)
            self.assertEqual(generated.count('_flash_e3_graph.capture_banks(self)'), 1)

    def test_generated_source_adds_no_sync_or_ring_calls(self):
        for kind, a_source in self.sources.items():
            if kind == 'pinned':
                continue
            generated = graph_hooks.generate(a_source)
            for token in ('synchronize(', 'globaltimer', 'perf_counter',
                          'time.time(', 'get_prev_sampled_outputs',
                          'cudaDeviceSynchronize', 'cudaStreamSynchronize'):
                self.assertEqual(generated.count(token), a_source.count(token), token)
            for inserted in (graph_hooks._CHECK_NEW, graph_hooks._REPLAY_NEW,
                             graph_hooks._CAPTURE_NEW):
                self.assertNotIn('synchronize(', inserted)
                self.assertNotIn('get_prev_sampled_outputs', inserted)
            self.assertEqual(
                graph_hooks._SHUTDOWN_PLAIN_NEW.count('synchronize('),
                graph_hooks._SHUTDOWN_PLAIN_OLD.count('synchronize('))
            self.assertEqual(
                graph_hooks._SHUTDOWN_E2_NEW.count('synchronize('),
                graph_hooks._SHUTDOWN_E2_OLD.count('synchronize('))

    def test_default_off_seams_are_inert(self):
        generated = graph_hooks.generate(self.sources['e3'])
        ast.parse(generated)
        worker, transaction, handle = decode_worker()
        runner = NS()
        scheduler_output = NS(flash_e3_transaction=transaction)
        self.assertIsNone(graph_hooks.check_route(
            runner, scheduler_output, descriptor(), 1, 2, 1))
        stock_calls = []

        def stock(batch_desc):
            stock_calls.append(batch_desc)
            return FakeOutput(2)

        self.assertEqual(graph_hooks.replay_fullgraph(
            runner, scheduler_output, descriptor(), stock).shape, (1, 2))
        self.assertEqual(len(stock_calls), 1)
        with mock.patch.dict(os.environ, {'FLASH_S5_OVERLAP': '0'}):
            self.assertIsNone(graph_hooks.capture_banks(runner))
        self.assertIsNone(graph_hooks.shutdown_banks(runner))

    def test_enabled_capture_needs_C_builder_and_captures_once(self):
        rig = make_router(capture=False)
        runner = NS(_flash_e3_graph_builder=lambda _: rig.router)
        with mock.patch.dict(os.environ, {'FLASH_S5_OVERLAP': '1'}):
            with self.assertRaises(GraphRoutingError):
                graph_hooks.capture_banks(NS())
            self.assertIs(graph_hooks.capture_banks(runner), rig.router)
            self.assertTrue(runner._flash_e3_graph_captured)
            with self.assertRaises(GraphRoutingError):
                graph_hooks.capture_banks(runner)
        with mock.patch.dict(os.environ, {'FLASH_S5_OVERLAP': '1'}):
            with self.assertRaises(GraphRoutingError):
                graph_hooks.capture_banks(NS(_flash_e3_graph_builder=lambda _: NS()))

    def test_shutdown_banks_pending_then_idempotent(self):
        rig = make_router()
        transition(rig)
        result, ticket = replay(rig)
        runner = NS(_flash_e3_graph_router=rig.router)
        with self.assertRaises(E3Pending):
            graph_hooks.shutdown_banks(runner)
        self.assertFalse(getattr(runner, '_flash_e3_graph_shutdown', False))
        finish(rig.worker, ticket, 0)
        rig.router.complete_replay(result, rig.worker.decode.completed[-1])
        record = graph_hooks.shutdown_banks(runner)
        self.assertIn('graph', record.order)
        self.assertTrue(runner._flash_e3_graph_shutdown)
        self.assertIsNone(graph_hooks.shutdown_banks(runner))

    def test_router_type_is_checked(self):
        with self.assertRaises(GraphRoutingError):
            graph_hooks.check_route(NS(_flash_e3_graph_router=NS()),
                                    NS(flash_e3_transaction=None),
                                    descriptor(), 1, 2, 1)
        with self.assertRaises(GraphRoutingError):
            graph_hooks.shutdown_banks(NS(_flash_e3_graph_router=NS()))


class TestHookRuntime(unittest.TestCase):
    def split_rig(self):
        rig = make_router()
        transition(rig)
        return rig

    def test_check_route_ordinary_and_split(self):
        rig = make_router()
        worker = rig.worker
        scheduler_output = NS(flash_e3_transaction=worker.decode.transaction)
        decision = graph_hooks.check_route(
            NS(_flash_e3_graph_router=rig.router), scheduler_output,
            descriptor(), 1, 2, 1)
        self.assertEqual(decision.mode, 'ordinary')
        transition(rig)
        with self.assertRaises(GraphRoutingError):
            # check_route now requires C's registered decode ticket.
            graph_hooks.check_route(NS(_flash_e3_graph_router=rig.router),
                                    scheduler_output, descriptor(), 1, 2, 1)
        scheduler_output.flash_e3_decode_ticket = decode_ticket(
            worker, worker.decode.transaction)
        decision = graph_hooks.check_route(
            NS(_flash_e3_graph_router=rig.router), scheduler_output,
            descriptor(), 1, 2, 1)
        self.assertEqual(decision.mode, 'split')

    def test_check_route_zero_tokens_returns_none(self):
        rig = self.split_rig()
        scheduler_output = NS(flash_e3_transaction=rig.worker.decode.transaction)
        # The stock engine returns before dispatch; split mode must not demand a
        # ticket or replay a graph for an empty batch.
        self.assertIsNone(graph_hooks.check_route(
            NS(_flash_e3_graph_router=rig.router), scheduler_output,
            descriptor(tokens=0, reqs=0, uniform=0), 0, 0, 0))
        self.assertEqual(rig.backend.replay_calls, [])

    def test_check_route_rejects_prefill_full_graph(self):
        rig = make_router()
        prefill_transaction, _, _ = activate_prefill(rig.worker)
        scheduler_output = NS(flash_e3_transaction=prefill_transaction)
        with self.assertRaises(GraphRoutingError):
            graph_hooks.check_route(NS(_flash_e3_graph_router=rig.router),
                                    scheduler_output, descriptor(),
                                    1, 2, 1)
        self.assertIsNone(graph_hooks.check_route(
            NS(_flash_e3_graph_router=rig.router), scheduler_output,
            descriptor(mode='piecewise'), 1, 2, 1))

    def test_replay_fullgraph_ordinary_is_stock(self):
        rig = make_router()
        scheduler_output = NS(flash_e3_transaction=rig.worker.decode.transaction)
        calls = []

        def stock(batch_desc):
            calls.append(batch_desc)
            return FakeOutput(2)

        output = graph_hooks.replay_fullgraph(
            NS(_flash_e3_graph_router=rig.router), scheduler_output,
            descriptor(), stock)
        self.assertEqual(len(calls), 1)
        self.assertEqual(output.shape, (1, 2))
        self.assertEqual(rig.backend.replay_calls, [])

    def test_replay_fullgraph_split_uses_backend_and_output(self):
        rig = self.split_rig()
        worker = rig.worker
        ticket = decode_ticket(worker, worker.decode.transaction)
        worker.decode.submit(ticket)
        scheduler_output = NS(flash_e3_transaction=worker.decode.transaction,
                              flash_e3_decode_ticket=ticket)
        calls = []

        def stock(batch_desc):
            calls.append(batch_desc)
            return FakeOutput(99)

        output = graph_hooks.replay_fullgraph(
            NS(_flash_e3_graph_router=rig.router), scheduler_output,
            descriptor(), stock)
        self.assertEqual(calls, [])
        self.assertEqual(output.shape, (1, 2))
        self.assertEqual(output.sampled_token_ids, [1, 1])
        self.assertEqual(len(rig.backend.replay_calls), 1)
        self.assertEqual(rig.router.pending_replays, 1)

    def test_replay_fullgraph_prefill_never_replays_decode(self):
        rig = make_router()
        prefill_transaction, _, _ = activate_prefill(rig.worker)
        scheduler_output = NS(flash_e3_transaction=prefill_transaction)
        calls = []

        def stock(batch_desc):
            calls.append(batch_desc)
            return FakeOutput(4)

        output = graph_hooks.replay_fullgraph(
            NS(_flash_e3_graph_router=rig.router), scheduler_output,
            descriptor(tokens=4, reqs=1, uniform=4, mode='piecewise'), stock)
        self.assertEqual(output.shape, (1, 4))
        self.assertEqual(len(calls), 1)
        self.assertEqual(rig.backend.replay_calls, [])
        with self.assertRaises(GraphRoutingError):
            graph_hooks.replay_fullgraph(
                NS(_flash_e3_graph_router=rig.router), scheduler_output,
                descriptor(tokens=4, reqs=1, uniform=4, mode='full'), stock)

    def test_prefill_and_split_do_not_touch_pp_ring(self):
        rig = self.split_rig()
        pp = FakePP()
        runner = NS(_flash_e3_graph_router=rig.router, pp_handler=pp)
        worker = rig.worker
        ticket = decode_ticket(worker, worker.decode.transaction)
        worker.decode.submit(ticket)
        decode_output = NS(flash_e3_transaction=worker.decode.transaction,
                           flash_e3_decode_ticket=ticket)
        graph_hooks.replay_fullgraph(runner, decode_output, descriptor(),
                                     lambda desc: self.fail('stock decode used'))
        self.assertEqual(pp.calls, [])
        prefill_transaction, _, _ = activate_prefill(worker)
        prefill_output = NS(flash_e3_transaction=prefill_transaction)
        graph_hooks.replay_fullgraph(runner, prefill_output,
                                     descriptor(tokens=4, reqs=1, uniform=4,
                                                mode='piecewise'),
                                     lambda desc: FakeOutput(4))
        self.assertEqual(pp.calls, [])

    def test_ordinary_stock_ring_order_unchanged(self):
        rig = make_router()
        pp = FakePP()
        runner = NS(_flash_e3_graph_router=rig.router, pp_handler=pp)
        scheduler_output = NS(flash_e3_transaction=rig.worker.decode.transaction)

        def stock(batch_desc):
            return pp.get_prev_sampled_outputs()

        output = graph_hooks.replay_fullgraph(runner, scheduler_output,
                                              descriptor(), stock)
        self.assertEqual(output, ('prev', 0))
        self.assertEqual(pp.calls, ['prev'])
        self.assertEqual(len(pp.ring), 1)

    def test_split_route_requires_probe_and_matching_values(self):
        rig = self.split_rig()
        worker = rig.worker
        ticket = decode_ticket(worker, worker.decode.transaction)
        worker.decode.submit(ticket)
        scheduler_output = NS(flash_e3_transaction=worker.decode.transaction,
                              flash_e3_decode_ticket=ticket)
        rig.router.probe = None
        with self.assertRaises(GraphRoutingError):
            graph_hooks.replay_fullgraph(NS(_flash_e3_graph_router=rig.router),
                                         scheduler_output, descriptor(),
                                         lambda desc: FakeOutput(2))
        rig.router.probe = lambda: (CONTEXT_ID + 1, STREAM)
        with self.assertRaises(GraphRoutingError):
            graph_hooks.replay_fullgraph(NS(_flash_e3_graph_router=rig.router),
                                         scheduler_output, descriptor(),
                                         lambda desc: FakeOutput(2))
        rig.router.probe = lambda: (CONTEXT_ID, STREAM)
        output = graph_hooks.replay_fullgraph(
            NS(_flash_e3_graph_router=rig.router), scheduler_output,
            descriptor(), lambda desc: self.fail('stock decode used'))
        self.assertEqual(output.shape, (1, 2))

    def test_full_gpu_route_during_prefill_fails_closed(self):
        rig = make_router()
        activate_prefill(rig.worker)
        scheduler_output = NS(
            flash_e3_transaction=rig.worker.decode.transaction,
            flash_e3_decode_ticket=None)
        with self.assertRaises(UnsupportedRoute):
            graph_hooks.check_route(NS(_flash_e3_graph_router=rig.router),
                                    scheduler_output, descriptor(), 1, 2, 1)
        with self.assertRaises(UnsupportedRoute):
            graph_hooks.replay_fullgraph(NS(_flash_e3_graph_router=rig.router),
                                         scheduler_output, descriptor(),
                                         lambda desc: self.fail('silent full replay'))


if __name__ == '__main__':
    unittest.main()


class ReviewRegressions(unittest.TestCase):
    def test_missing_transaction_never_replays_stock_while_prefill_parked(self):
        rig = make_router()
        transition(rig)
        activate_prefill(rig.worker)
        runner = NS(_flash_e3_graph_router=rig.router)
        calls = []
        with self.assertRaises(GraphRoutingError):
            graph_hooks.replay_fullgraph(runner, NS(), descriptor(), calls.append)
        with self.assertRaises(GraphRoutingError):
            graph_hooks.check_route(runner, NS(), descriptor(), 1, 2, 1)
        self.assertEqual(calls, [])

    def test_no_ordinary_transition_with_parked_prefill(self):
        rig = make_router()
        transition(rig)
        activate_prefill(rig.worker)
        with self.assertRaises(E3Pending):
            transition(rig, 'ordinary', target_context=0, seq=11)

    def test_allocated_bytes_cannot_exceed_reservation(self):
        rig = make_router()
        rig.router.banks['split'].context = dataclasses.replace(rig.context, basis='measured')
        rig.router.banks['split'].allocated_bytes = RESERVED + 1
        with self.assertRaises(GraphRoutingError):
            rig.router.budget(reserve_measured_bytes=150).require_bounded(RESERVED * 2)
