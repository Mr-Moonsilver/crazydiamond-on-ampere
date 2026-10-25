"""CPU-only tests for E3 execution state (A's files plus pinned transforms).

No torch, no vLLM, no CUDA: runner methods are sliced from the generated
source and executed with small fakes, and all storage facts are addresses.
"""
import ast
import dataclasses
import hashlib
import os
from pathlib import Path
import sys
import textwrap
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import e3_types  # noqa: E402
import lane_hooks  # noqa: E402
import lane_state  # noqa: E402
import runner_hooks  # noqa: E402
from e3_types import (  # noqa: E402
    ARENA_BYTES,
    ARENA_PAGES,
    BUCKETS_AT_200K,
    CompletionKind,
    CompletionProof,
    DependencyRole,
    E3ContractError,
    E3Pending,
    HISTORIES_PER_RANK,
    LaneId,
    LaneRole,
    OperationTicket,
    PAGE_BYTES,
    PAGE_TOKENS,
    PollResult,
    TransactionId,
)
from lane_state import (  # noqa: E402
    ARENA_BYTES as STATE_ARENA_BYTES,
    GRAPH_POOL_MEASURED_BYTES,
    HC_BLOCK_OUTPUT_BYTES,
    HC_HIDDEN_BYTES,
    HC_INJECTION_BYTES,
    HISTORIES_PER_RANK as STATE_HISTORIES,
    HISTORY_BYTES_PER_RANK,
    HISTORY_PAGES_AT_200K,
    PP_HIDDEN_BYTES,
    QSA_ACCUM_BYTES,
    TOPK_BYTES,
    BatchRow,
    BufferBinding,
    HcTupleBinding,
    InventoryEntry,
    InventoryReport,
    MetadataSnapshot,
    PrivateUidBank,
    QsaHistories,
    QsaHistory,
    ScratchSpec,
    WorkerState,
    estimated_e3_inventory,
    prefill_scratch_specs,
)

SOURCE = ROOT / 'patches/rowcache/model_runner_v2.py.patched'

NS = types.SimpleNamespace


class ForwardContext:
    def __init__(self, value='decode-ctx'):
        self.value = value
        self.history = []

    def get(self):
        return self.value

    def set(self, value):
        self.history.append(value)
        self.value = value


class Workspace:
    def __init__(self, current='decode-ws'):
        self.stack = [current]
        self.calls = []

    def current(self):
        return self.stack[-1]

    def enter(self, workspace):
        self.calls.append(('enter', workspace))
        self.stack.append(workspace)

    def exit(self, workspace):
        if not self.stack or self.stack[-1] != workspace:
            raise AssertionError('workspace exit out of order')
        self.calls.append(('exit', workspace))
        self.stack.pop()


def worker(uid='req-a', generation=1, engine_epoch=3):
    return WorkerState(0, engine_epoch=engine_epoch, weights=object(),
                       weights_address=4096, weights_bytes=8_388_608)


def metadata(state, uid='req-a', generation=1, start=0, end=4, slot=(0,)):
    transaction = state.new_transaction(uid, generation, start, end)
    rows = (BatchRow(uid, generation, 0, start, end, 0),)
    snapshot = MetadataSnapshot.capture(transaction, rows, block_ids=(10, 11),
                                        slot_mappings=slot)
    return transaction, snapshot


def runtime(state, *, context='decode-ctx', workspace='decode-ws'):
    return lane_hooks.LaneRuntime(state, submit_backend=NS(submit=lambda ticket: None, poll=lambda txn: ()),
                       forward_context=ForwardContext(context),
                       workspace=Workspace(workspace),
                       prefill_forward_context='prefill-ctx',
                       prefill_workspace='prefill-ws',
                       uid_namespace=NS(QSA_REQ_UID=[77, 77]), prefill_uid_buffer=[-2, -2])


def open_prefill(rt, state, uid='req-a', generation=1, start=0, end=4,
                 bindings=None, private_attrs=None):
    transaction, snapshot = metadata(state, uid, generation, start, end)
    if bindings is None:
        bindings = ((ScratchSpec('prefill_input_ids', 'prefill', 'metadata', 4096),
                     500_000),)
    if private_attrs is None:
        private_attrs = {'input_buffers': {'owner': 'prefill'}}
    handle = rt.open_prefill(transaction, snapshot, bindings=bindings,
                             private_attrs=private_attrs)
    return transaction, snapshot, handle


def open_decode(rt, state, uid='decode-u', generation=1):
    transaction, snapshot = metadata(state, uid, generation, 0, 1, (0,))
    state.decode.reserve(transaction, metadata=snapshot,
                         bindings=((ScratchSpec('decode_ids', 'decode', 'metadata',
                                                128), 900_000),))
    state.decode.ready()
    return transaction, snapshot


def ticket(rt, transaction, op_seq=0, role=DependencyRole.PRODUCER,
           lane_role=LaneRole.PREFILL):
    lane = rt.state.lane(lane_role)
    return OperationTicket(transaction, rt.state.worker_rank, lane.lane_id,
                           op_seq, role)


def proof(ticket_value, kind=CompletionKind.GPU_EVENT, seq=0):
    return CompletionProof(ticket_value, kind, seq)


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


def load_method(generated, name, extra=None):
    namespace = dict(extra or {})
    namespace.setdefault('_flash_e3_hooks', lane_hooks)
    source = 'from __future__ import annotations\n' + method_source(generated, name)
    exec(source, namespace)
    return namespace[name]


class TestContractTypes(unittest.TestCase):
    def test_immutable_and_redacted(self):
        transaction = TransactionId(3, 'request-secret', 2, 0, 4, 9)
        self.assertNotIn('request-secret', repr(transaction))
        self.assertIn('<len 14>', repr(transaction))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            transaction.chunk_start = 1
        lane = LaneId(0, LaneRole.PREFILL, 2, 7)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            lane.storage_epoch = 3
        self.assertNotIn('request-secret', repr(BatchRow('request-secret', 1, 0, 0, 4, 0)))

    def test_strict_validation(self):
        with self.assertRaises(E3ContractError):
            TransactionId(3, 'u', True, 0, 4, 0)
        with self.assertRaises(E3ContractError):
            TransactionId(3, 'u', 1, 0, 0, 0)
        with self.assertRaises(E3ContractError):
            TransactionId(-1, 'u', 1, 0, 4, 0)
        with self.assertRaises(E3ContractError):
            LaneId(0, LaneRole.DECODE, 0, 0)
        with self.assertRaises(E3ContractError):
            LaneId(0, 'decode', 1, 0)
        with self.assertRaises(E3ContractError):
            PollResult([], 0, False)

    def test_ticket_identity(self):
        transaction = TransactionId(3, 'u', 1, 0, 4, 0)
        lane = LaneId(0, LaneRole.PREFILL, 1, 0)
        ticket_value = OperationTicket(transaction, 0, lane, 0, DependencyRole.PRODUCER)
        self.assertEqual(ticket_value.sequence_key, (0, 'prefill', 0))
        with self.assertRaises(E3ContractError):
            OperationTicket(transaction, 1, lane, 0, DependencyRole.PRODUCER)
        with self.assertRaises(E3ContractError):
            OperationTicket(transaction, 0, lane, -1, DependencyRole.PRODUCER)

    def test_no_rpc_ack_completion(self):
        names = {kind.name for kind in CompletionKind}
        values = {kind.value for kind in CompletionKind}
        self.assertNotIn('RPC_ACK', names)
        self.assertNotIn('rpc_ack', values)
        self.assertEqual(names, {'GPU_EVENT', 'COPY_EVENT', 'PP_CONSUMER', 'STATE_WRITE'})

    def test_constants_match_ledger(self):
        self.assertEqual(PAGE_TOKENS, 4048)
        self.assertEqual(PAGE_BYTES, 4_145_152)
        self.assertEqual(ARENA_PAGES, 24)
        self.assertEqual(ARENA_BYTES, 99_483_648)
        self.assertEqual(BUCKETS_AT_200K, (24, 24, 2))
        self.assertEqual(HISTORIES_PER_RANK, 6)
        self.assertEqual(STATE_ARENA_BYTES, ARENA_BYTES)
        self.assertEqual(STATE_HISTORIES, HISTORIES_PER_RANK)


class TestIdentityChecks(unittest.TestCase):
    def test_transaction_is_not_a_batch_row(self):
        state = worker()
        transaction, snapshot = metadata(state)
        moved = TransactionId(transaction.engine_epoch, transaction.request_uid,
                              transaction.request_generation, transaction.chunk_start,
                              transaction.chunk_end, transaction.txn_seq + 7)
        state.check_transaction(moved)
        self.assertTrue(transaction.same_request(moved))

        stale_engine = TransactionId(transaction.engine_epoch + 1, transaction.request_uid,
                                     transaction.request_generation, 0, 4,
                                     transaction.txn_seq)
        with self.assertRaises(E3ContractError):
            state.check_transaction(stale_engine)

        state.declare_request(transaction.request_uid, 2)
        with self.assertRaises(E3ContractError):
            state.check_transaction(transaction)
        with self.assertRaises(E3ContractError):
            state.new_transaction(transaction.request_uid, 1, 0, 4)

        wrong_row = BatchRow(transaction.request_uid, 9, 0, 0, 4, 0)
        with self.assertRaises(E3ContractError):
            MetadataSnapshot.capture(transaction, (wrong_row,))

    def test_generation_cannot_advance_under_leased_prefill(self):
        state = worker()
        rt = runtime(state)
        open_prefill(rt, state)
        with self.assertRaises(E3Pending):
            state.new_transaction('req-a', 2, 0, 4)
        self.assertEqual(state.declare_request('other-uid', 4), 4)

    def test_metadata_capture_order_and_counts(self):
        state = worker()
        transaction, snapshot = metadata(state)
        self.assertEqual(snapshot.rows[0].chunk_end, 4)
        with self.assertRaises(E3ContractError):
            MetadataSnapshot.capture(transaction, (), slot_mappings=())
        with self.assertRaises(E3ContractError):
            MetadataSnapshot.capture(transaction, snapshot.rows, slot_mappings=(0, 1))
        out_of_order = BatchRow('req-a', 1, 3, 0, 4, 0)
        with self.assertRaises(E3ContractError):
            MetadataSnapshot.capture(transaction, (out_of_order,))

    def test_prefill_metadata_is_required(self):
        state = worker()
        rt = runtime(state)
        transaction = state.new_transaction('req-a', 1, 0, 4)
        with self.assertRaises(E3ContractError):
            rt.open_prefill(transaction, None)
        other, other_snapshot = metadata(state, start=4, end=8)
        with self.assertRaises(E3ContractError):
            rt.open_prefill(transaction, other_snapshot)


class TestSinglePrefillAndEpochs(unittest.TestCase):
    def test_one_active_prefill_globally(self):
        state = worker()
        rt = runtime(state)
        first, _, _ = open_prefill(rt, state)
        second, second_snapshot = metadata(state, 'req-b', 1, 0, 2)
        with self.assertRaises(E3Pending):
            rt.open_prefill(second, second_snapshot)
        self.assertEqual(state.active_prefill, first)

    def test_release_requires_proof_then_advances_epoch(self):
        state = worker()
        rt = runtime(state)
        transaction, _, old_handle = open_prefill(rt, state)
        ticket_value = rt.submit(LaneRole.PREFILL, transaction, 0,
                                 DependencyRole.PRODUCER)
        with self.assertRaises(E3Pending):
            rt.release(transaction)
        with self.assertRaises(E3Pending):
            state.prefill.reserve(transaction, metadata=None)
        rt.release(transaction, (proof(ticket_value),))
        self.assertIsNone(state.active_prefill)
        self.assertEqual(state.prefill.state, 'free')

        next_transaction, next_snapshot = metadata(state, start=4, end=8)
        new_handle = rt.open_prefill(next_transaction, next_snapshot,
                                     private_attrs={})
        self.assertEqual(new_handle.storage_epoch, old_handle.storage_epoch + 1)
        stale = OperationTicket(next_transaction, 0, old_handle.lane_id, 0,
                                DependencyRole.PRODUCER)
        with self.assertRaises(E3ContractError):
            state.prefill.submit(stale)

    def test_stale_batch_row_rejected(self):
        state = worker()
        rt = runtime(state)
        transaction, snapshot, _ = open_prefill(rt, state)
        state.prefill.note_batch(snapshot)
        self.assertEqual(state.prefill._latest_batch.batch_seq, 2)
        with self.assertRaises(E3ContractError):
            state.prefill.check_batch(snapshot)
        state.prefill.check_batch(state.prefill._latest_batch)

        newer = MetadataSnapshot.capture(transaction, snapshot.rows,
                                         slot_mappings=(0,))
        with self.assertRaises(E3ContractError):
            state.prefill.note_batch(newer)  # immutable admission block IDs cannot disappear
        with self.assertRaises(E3ContractError):
            state.prefill.check_batch(snapshot)


class TestCompletionProofs(unittest.TestCase):
    def test_release_needs_the_matching_proof(self):
        state = worker()
        rt = runtime(state)
        transaction, _, _ = open_prefill(rt, state)
        ticket_value = rt.submit(LaneRole.PREFILL, transaction, 0,
                                 DependencyRole.PRODUCER)
        wrong_op = OperationTicket(transaction, 0, state.prefill.lane_id, 5,
                                   DependencyRole.PRODUCER)
        with self.assertRaises(E3ContractError):
            rt.release(transaction, (proof(wrong_op),))
        wrong_epoch = OperationTicket(
            transaction, 0,
            LaneId(0, LaneRole.PREFILL, state.prefill.storage_epoch, 99),
            0, DependencyRole.PRODUCER)
        with self.assertRaises(E3ContractError):
            rt.release(transaction, (proof(wrong_epoch),))
        rt.release(transaction, (proof(ticket_value),))

    def test_operations_are_bounded_and_quarantine_on_failure(self):
        events = []

        class Backend:
            def submit(self, ticket_value):
                events.append(ticket_value)

            def poll(self, transaction):
                return ()

        state = worker()
        rt = runtime(state)
        rt.backend = Backend()
        transaction, _, _ = open_prefill(rt, state)
        first = rt.submit(LaneRole.PREFILL, transaction, 0,
                          DependencyRole.PRODUCER)
        with self.assertRaises(E3ContractError):
            rt.submit(LaneRole.PREFILL, transaction, 0, DependencyRole.PRODUCER)
        result = rt.poll(transaction)
        self.assertEqual(result, PollResult((), 1, False))
        self.assertEqual(events, [first])

        rt.fail_completion(transaction, first, 'engine lost the copy')
        self.assertTrue(state.prefill.failed)
        self.assertEqual(state.prefill.state, 'quarantined')
        self.assertIsNotNone(state.prefill.handle)
        failed = rt.poll(transaction)
        self.assertEqual(failed, PollResult((), 1, True))
        with self.assertRaises(E3ContractError):
            rt.release(transaction, (proof(first),))
        with self.assertRaises(E3Pending):
            rt.open_prefill(transaction, state.prefill.metadata)

    def test_backend_exception_quarantines(self):
        class Backend:
            def submit(self, ticket_value):
                raise RuntimeError('transport down')

            def poll(self, transaction):
                return ()

        state = worker()
        rt = runtime(state)
        rt.backend = Backend()
        transaction, _, _ = open_prefill(rt, state)
        with self.assertRaises(RuntimeError):
            rt.submit(LaneRole.PREFILL, transaction, 0, DependencyRole.PRODUCER)
        self.assertTrue(state.prefill.failed)
        self.assertEqual(rt.poll(transaction), PollResult((), 1, True))

    def test_completion_sequence_is_monotonic(self):
        state = worker()
        rt = runtime(state)
        transaction, _, _ = open_prefill(rt, state)
        first = rt.submit(LaneRole.PREFILL, transaction, 0, DependencyRole.PRODUCER)
        second = rt.submit(LaneRole.PREFILL, transaction, 1, DependencyRole.PRODUCER)
        state.prefill.complete(proof(first, seq=5))
        with self.assertRaises(E3ContractError):
            state.prefill.complete(proof(second, seq=4))
        self.assertEqual(len(state.prefill.outstanding), 1)


class TestAbortRetention(unittest.TestCase):
    def test_abort_keeps_storage_until_completion(self):
        state = worker()
        rt = runtime(state)
        transaction, _, handle = open_prefill(rt, state)
        ticket_value = rt.submit(LaneRole.PREFILL, transaction, 0,
                                 DependencyRole.PRODUCER)
        rt.abort(transaction)
        self.assertEqual(state.prefill.state, 'retiring')
        with self.assertRaises(E3ContractError):
            rt.submit(LaneRole.PREFILL, transaction, 1, DependencyRole.PRODUCER)
        with self.assertRaises(E3Pending):
            rt.release(transaction)
        self.assertEqual(state.prefill.handle, handle)
        self.assertTrue(state.prefill.buffers)
        rt.release(transaction, (proof(ticket_value),))
        self.assertIsNone(state.prefill.handle)
        self.assertFalse(state.prefill.buffers)


class TestPersistentCommit(unittest.TestCase):
    def test_commit_is_exactly_once_and_proof_gated(self):
        state = worker()
        rt = runtime(state)
        transaction, _, _ = open_prefill(rt, state)
        rt.propose_state(transaction, {'cursor': [1, 2]})
        ticket_value = rt.submit(LaneRole.PREFILL, transaction, 0,
                                 DependencyRole.COMMIT)

        with self.assertRaises(E3ContractError):
            rt.commit(transaction, None)
        with self.assertRaises(E3ContractError):
            rt.commit(transaction, proof(ticket_value, CompletionKind.GPU_EVENT, 0))

        other = TransactionId(transaction.engine_epoch, transaction.request_uid,
                              transaction.request_generation, transaction.chunk_start,
                              transaction.chunk_end, transaction.txn_seq + 50)
        foreign_txn = OperationTicket(other, 0, state.prefill.lane_id, 0,
                                      DependencyRole.COMMIT)
        with self.assertRaises(E3ContractError):
            rt.commit(transaction, proof(foreign_txn, CompletionKind.STATE_WRITE, 0))

        decode_lane = LaneId(0, LaneRole.DECODE, 1, 0)
        foreign_lane = OperationTicket(transaction, 0, decode_lane, 0,
                                       DependencyRole.COMMIT)
        with self.assertRaises(E3ContractError):
            rt.commit(transaction, proof(foreign_lane, CompletionKind.STATE_WRITE, 0))

        non_commit = OperationTicket(transaction, 0, state.prefill.lane_id, 1,
                                     DependencyRole.CONSUMER)
        with self.assertRaises(E3ContractError):
            rt.commit(transaction, proof(non_commit, CompletionKind.STATE_WRITE, 0))

        state.prefill.complete(proof(ticket_value, CompletionKind.STATE_WRITE, 0))
        record = rt.commit(transaction, proof(ticket_value, CompletionKind.STATE_WRITE, 0))
        self.assertEqual(record.value, (('cursor', (1, 2)),))
        self.assertEqual(state.prefill.persistent.applied, 1)
        self.assertEqual(state.prefill.persistent.epoch, 1)
        self.assertIs(rt.commit(transaction,
                                proof(ticket_value, CompletionKind.STATE_WRITE, 0)),
                      record)
        self.assertEqual(state.prefill.persistent.applied, 1)
        with self.assertRaises(E3ContractError):
            rt.propose_state(transaction, 'again')

    def test_abort_cancels_pending(self):
        state = worker()
        rt = runtime(state)
        transaction, _, _ = open_prefill(rt, state)
        rt.propose_state(transaction, 'candidate')
        rt.abort(transaction)
        self.assertIsNone(state.prefill.persistent.pending)
        with self.assertRaises(E3ContractError):
            rt.propose_state(transaction, 'late')


class TestLaneBindings(unittest.TestCase):
    def test_cross_lane_aliasing_is_rejected(self):
        state = worker()
        rt = runtime(state)
        transaction = state.new_transaction('req-a', 1, 0, 4)
        state.decode.reserve(transaction, bindings=(
            (ScratchSpec('decode_ids', 'decode', 'metadata', 1024), 1_000_000),))
        state.decode.ready()
        snapshot = MetadataSnapshot.capture(
            transaction, (BatchRow('req-a', 1, 0, 0, 4, 0),))
        with self.assertRaises(E3ContractError):
            state.prefill.reserve(
                transaction, metadata=snapshot, bindings=(
                    (ScratchSpec('prefill_ids', 'prefill', 'metadata', 1024),
                     1_000_512),))

    def test_shared_serialized_storage_is_explicit(self):
        state = worker()
        transaction, snapshot = metadata(state, 'decode-u', 1, 0, 1, (0,))
        state.decode.reserve(transaction, metadata=snapshot)
        state.decode.ready()
        spec_a = ScratchSpec('staged_a', 'decode', 'serialized', 4096, shared=True,
                             reuse_condition='one staged write at a time')
        spec_b = ScratchSpec('staged_b', 'decode', 'serialized', 4096, shared=True,
                             reuse_condition='one staged write at a time')
        state.decode.bind_scratch(spec_a, 2_000_000)
        state.decode.bind_scratch(spec_b, 2_000_000)
        self.assertEqual(state.decode.buffers['staged_a'].address,
                         state.decode.buffers['staged_b'].address)

        plain = ScratchSpec('plain', 'decode', 'serialized', 4096)
        with self.assertRaises(E3ContractError):
            state.decode.bind_scratch(plain, 2_000_000)

    def test_shared_weights_identity(self):
        weights = object()
        state = WorkerState(0, engine_epoch=2, weights=weights,
                            weights_address=1234, weights_bytes=8_000_000)
        self.assertIs(state.weights_token, weights)
        self.assertIs(state.weights_token, state.weights_object)
        self.assertEqual(state.weights_bytes, 8_000_000)

    def test_six_histories_only(self):
        base = 2_000_000_000

        def histories(count):
            return QsaHistories(tuple(
                QsaHistory(f'layer-{index}', 0,
                           base + index * HISTORY_PAGES_AT_200K * PAGE_BYTES,
                           HISTORY_PAGES_AT_200K * PAGE_BYTES)
                for index in range(count)))

        valid = histories(6)
        self.assertEqual(valid.bytes, HISTORY_BYTES_PER_RANK)
        with self.assertRaises(E3ContractError):
            histories(5)
        with self.assertRaises(E3ContractError):
            histories(7)
        duplicate = list(valid.histories)
        duplicate[1] = QsaHistory('layer-0', 0, duplicate[1].address,
                                  duplicate[1].nbytes)
        with self.assertRaises(E3ContractError):
            QsaHistories(tuple(duplicate))
        overlap = list(valid.histories)
        overlap[1] = QsaHistory('layer-b', 0, overlap[0].address,
                                overlap[1].nbytes)
        with self.assertRaises(E3ContractError):
            QsaHistories(tuple(overlap))

        state = worker()
        rt = runtime(state)
        open_prefill(rt, state)
        state.prefill.bind_histories(valid)
        self.assertEqual(len(state.inventory().entries), 8)  # weights + scratch + 6 histories

    def test_histories_are_prefill_only(self):
        state = worker()
        transaction = state.new_transaction('req-a', 1, 0, 4)
        state.decode.reserve(transaction, metadata=MetadataSnapshot.capture(
            transaction, (BatchRow('req-a', 1, 0, 0, 4, 0),)))
        histories = QsaHistories(tuple(
            QsaHistory(f'layer-{index}', 0,
                       3_000_000_000 + index * HISTORY_PAGES_AT_200K * PAGE_BYTES,
                       HISTORY_PAGES_AT_200K * PAGE_BYTES)
            for index in range(6)))
        with self.assertRaises(E3ContractError):
            state.decode.bind_histories(histories)

    def test_private_uid_bank(self):
        bank = PrivateUidBank()
        self.assertEqual(bank.new(3), 1)
        target = [9, 9]
        bank.write(target, [3, 4])
        self.assertEqual(target, [1, -2])
        self.assertEqual(bank.snapshot(), ((3, 1),))

    def test_uid_buffers_are_lane_private(self):
        state = worker(); rt = runtime(state)
        open_prefill(rt, state)
        rt._preparing_prefill = True
        self.assertFalse(rt.new_uid(NS(), 0))
        self.assertFalse(rt.write_uids(NS(), NS(idx_mapping_np=[0])))
        self.assertEqual(state.prefill.uid.snapshot(), ())
        with self.assertRaises(E3ContractError):
            other = runtime(worker())
            tx, md = metadata(other.state)
            other.open_prefill(tx, md, private_attrs={'_qsa_uid_slots': []})


class TestSnapshots(unittest.TestCase):
    def test_metadata_snapshot_sequence(self):
        state = worker()
        transaction, snapshot = metadata(state)
        sequenced = snapshot.with_sequence(4)
        self.assertEqual(snapshot.batch_seq, 0)
        self.assertEqual(sequenced.batch_seq, 4)
        with self.assertRaises(E3ContractError):
            snapshot.with_sequence(-1)

    def test_recurrent_snapshot_is_frozen(self):
        state = worker()
        rt = runtime(state)
        transaction, _, _ = open_prefill(rt, state)
        source = {'cursor': [1, 2], 'flags': {'b': True}}
        snapshot = state.prefill.bind_recurrent(source)
        source['cursor'].append(3)
        source['flags']['c'] = True
        self.assertEqual(snapshot.transaction, transaction)
        self.assertEqual(snapshot.values, (('cursor', (1, 2)), ('flags', (('b', True),))))

    def test_hc_materializes_once_with_pp_proof(self):
        state = worker()
        rt = runtime(state)
        transaction, _, _ = open_prefill(rt, state)
        binding = state.prefill.bind_hc(['hidden'], ['block'], ['inject'], 7,
                                        (10, 20, 30))
        self.assertIsInstance(binding, HcTupleBinding)
        self.assertNotIn('hidden', repr(binding))
        ticket_value = ticket(rt, transaction, 0)
        with self.assertRaises(E3ContractError):
            state.prefill.materialize_hc(proof(ticket_value, CompletionKind.GPU_EVENT, 0))
        self.assertFalse(state.prefill._hc_materialized)
        ticket_value = OperationTicket(transaction, 0, state.prefill.lane_id, 0, DependencyRole.HANDOFF)
        state.prefill.submit(ticket_value)
        state.prefill.complete(proof(ticket_value, CompletionKind.PP_CONSUMER, 0))
        self.assertIs(state.prefill.materialize_hc(
            proof(ticket_value, CompletionKind.PP_CONSUMER, 0)), binding)
        with self.assertRaises(E3ContractError):
            state.prefill.materialize_hc(
                proof(ticket_value, CompletionKind.PP_CONSUMER, 1))

    def test_decode_and_prefill_state_are_separate(self):
        state = worker()
        rt = runtime(state)
        prefill_txn, prefill_meta, _ = open_prefill(rt, state)
        decode_txn, decode_meta = open_decode(rt, state)
        state.prefill.bind_recurrent([1])
        state.decode.bind_recurrent([2])
        self.assertEqual(state.prefill.recurrent.values, (1,))
        self.assertEqual(state.decode.recurrent.values, (2,))
        self.assertIs(state.prefill.metadata, prefill_meta)
        self.assertIs(state.decode.metadata, decode_meta)
        self.assertIsNot(state.prefill.metadata, state.decode.metadata)
        self.assertNotEqual(prefill_txn, decode_txn)


class TestInventory(unittest.TestCase):
    def test_static_categories_and_budget(self):
        report = estimated_e3_inventory()
        categories = report.by_category()
        self.assertEqual(categories['graph_pool'],
                         (GRAPH_POOL_MEASURED_BYTES, GRAPH_POOL_MEASURED_BYTES))
        self.assertEqual(categories['arena'], (0, ARENA_BYTES))
        self.assertEqual(categories['qsa_history'], (0, HISTORY_BYTES_PER_RANK))
        self.assertEqual(categories['weights'], (0, 0))
        self.assertEqual(categories['metadata'], (24_780, 0))
        expected_new = (24_780 + HC_HIDDEN_BYTES + HC_BLOCK_OUTPUT_BYTES +
                        HC_INJECTION_BYTES + TOPK_BYTES + QSA_ACCUM_BYTES +
                        PP_HIDDEN_BYTES + GRAPH_POOL_MEASURED_BYTES)
        self.assertEqual(report.new_bytes, expected_new)
        with self.assertRaises(E3ContractError):
            report.require_bounded(expected_new)  # unknown workspace bytes are not zero
        with self.assertRaises(E3ContractError):
            report.require_bounded(expected_new - 1)
        self.assertEqual(report.existing_bytes,
                         ARENA_BYTES + HISTORY_BYTES_PER_RANK +
                         GRAPH_POOL_MEASURED_BYTES)

    def test_unmeasured_workspaces_stay_zero(self):
        report = estimated_e3_inventory()
        self.assertEqual(set(report.unmeasured()),
                         {'shared_weights', 'marlin_workspace', 'green_context',
                          'communicator_buffers'})
        for entry in report.entries:
            if entry.name in report.unmeasured():
                self.assertEqual(entry.nbytes, 0)
        with self.assertRaises(E3ContractError):
            estimated_e3_inventory(extra_history_window_bytes=1)
        measured = estimated_e3_inventory(marlin_workspace_bytes=1024)
        entry = [e for e in measured.entries if e.name == 'marlin_workspace'][0]
        self.assertEqual((entry.basis, entry.nbytes), ('estimate', 1024))

    def test_duplicate_inventory_entry_rejected(self):
        entry = InventoryEntry('x', 'owner', 'category', 1, 'estimate', True)
        with self.assertRaises(E3ContractError):
            InventoryReport((entry, entry))

    def test_dynamic_inventory_and_graph_retirement(self):
        state = WorkerState(0, engine_epoch=4, weights=object(), weights_bytes=100)
        rt = runtime(state)
        transaction, _, _ = open_prefill(rt, state)
        pool = state.prefill.register_graph_pool(
            'decode_split', 4_000_000, GRAPH_POOL_MEASURED_BYTES)
        entries = {entry.name: entry for entry in state.inventory().entries}
        self.assertEqual(entries['decode_split'].category, 'graph_pool')
        self.assertTrue(entries['decode_split'].new_allocation)
        self.assertEqual(entries['shared_weights'].nbytes, 100)
        self.assertIn('prefill_input_ids', entries)
        with self.assertRaises(E3ContractError):
            state.prefill.register_graph_pool(
                'decode_split', 4_000_000, GRAPH_POOL_MEASURED_BYTES)
        with self.assertRaises(E3Pending):
            rt.release(transaction)
        self.assertEqual(state.prefill.retire_graph_pools(), (pool,))
        with self.assertRaises(E3ContractError):
            state.prefill.retire_graph_pools(epoch=state.prefill.storage_epoch - 1)
        rt.release(transaction)
        names = {entry.name for entry in state.inventory().entries}
        self.assertNotIn('decode_split', names)
        self.assertIn('shared_weights', names)

    def test_scratch_specs_are_immutable_and_sized(self):
        specs = prefill_scratch_specs(1024)
        self.assertEqual(len(specs), 12)
        self.assertEqual(sum(spec.nbytes for spec in specs if spec.category == 'metadata'),
                         24_780)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            specs[0].nbytes = 1
        with self.assertRaises(E3ContractError):
            ScratchSpec('x', 'prefill', 'metadata', 0)


class TestHostDispatcher(unittest.TestCase):
    def test_restores_context_and_workspace_on_exception(self):
        context = ForwardContext('decode-ctx')
        workspace = Workspace('decode-ws')
        dispatcher = lane_hooks.HostDispatcher(context, workspace)
        with self.assertRaises(RuntimeError):
            with dispatcher.enter(None, 'prefill-ctx', 'prefill-ws'):
                self.assertEqual(context.get(), 'prefill-ctx')
                self.assertEqual(workspace.current(), 'prefill-ws')
                raise RuntimeError('forward failed')
        self.assertEqual(context.get(), 'decode-ctx')
        self.assertEqual(workspace.current(), 'decode-ws')
        self.assertFalse(dispatcher.active)
        self.assertEqual(workspace.calls, [('enter', 'prefill-ws'),
                                           ('exit', 'prefill-ws')])

    def test_concurrent_python_forward_is_rejected(self):
        dispatcher = lane_hooks.HostDispatcher(ForwardContext(), Workspace())
        with dispatcher.enter(None, 'a', 'wa'):
            with self.assertRaises(E3Pending):
                with dispatcher.enter(None, 'b', 'wb'):
                    self.fail('nested forward installed')
        self.assertFalse(dispatcher.active)

    def test_missing_adapter_rejected(self):
        with self.assertRaises(E3ContractError):
            lane_hooks.HostDispatcher(None, Workspace())


class _Runner:
    def __init__(self):
        self.input_buffers: dict = {}
        self._qsa_uid_slots: list = []
        self.execute_model_state: object = None
        self._flash_e3_lane: object = None


def _prefill_stock(runner, seen, output='out', pending=True):
    def stock(scheduler_output, *args):
        seen.append({'buffers': runner.input_buffers,
                     'uids': runner._qsa_uid_slots,
                     'state': runner.execute_model_state})
        if pending:
            runner.execute_model_state = 'prefill-pending'
        return output
    return stock


class TestExecutionRouting(unittest.TestCase):
    def test_prefill_swap_and_restore(self):
        state = worker()
        rt = runtime(state)
        transaction, _, _ = open_prefill(rt, state)
        rt.submit(LaneRole.PREFILL, transaction, 0, DependencyRole.CONSUMER)
        runner = _Runner()
        runner.input_buffers = {'owner': 'decode'}
        runner._qsa_uid_slots = [7, 7]
        runner.execute_model_state = 'decode-pending'
        seen = []
        output = rt.execute(runner, _prefill_stock(runner, seen), None, transaction)
        self.assertEqual(output, 'out')
        self.assertEqual(seen, [{'buffers': {'owner': 'prefill'},
                                 'uids': [7, 7], 'state': None}])
        self.assertEqual(rt.prefill_execution, 'prefill-pending')
        self.assertEqual(runner.input_buffers, {'owner': 'decode'})
        self.assertEqual(runner._qsa_uid_slots, [7, 7])
        self.assertEqual(runner.execute_model_state, 'decode-pending')
        self.assertEqual(rt.dispatcher.forward_context.get(), 'decode-ctx')
        self.assertEqual(rt.dispatcher.workspace.current(), 'decode-ws')

    def test_forward_exception_restores_everything(self):
        state = worker()
        rt = runtime(state)
        transaction, _, _ = open_prefill(rt, state)
        rt.submit(LaneRole.PREFILL, transaction, 0, DependencyRole.CONSUMER)
        runner = _Runner()
        runner.input_buffers = {'owner': 'decode'}
        runner._qsa_uid_slots = [7, 7]
        runner.execute_model_state = 'decode-pending'

        def stock(scheduler_output, *args):
            runner.input_buffers = {'owner': 'mutated'}
            runner.execute_model_state = 'prefill-pending'
            raise RuntimeError('kernel failed')

        with self.assertRaises(RuntimeError):
            rt.execute(runner, stock, None, transaction)
        self.assertEqual(runner.input_buffers, {'owner': 'decode'})
        self.assertEqual(runner.execute_model_state, 'decode-pending')
        self.assertEqual(rt.prefill_execution, 'prefill-pending')
        self.assertTrue(state.prefill.failed)
        self.assertEqual(rt.dispatcher.forward_context.get(), 'decode-ctx')

    def test_missing_prefill_execution_state_fails(self):
        state = worker()
        rt = runtime(state)
        transaction, _, _ = open_prefill(rt, state)
        rt.submit(LaneRole.PREFILL, transaction, 0, DependencyRole.CONSUMER)
        runner = _Runner()
        runner.input_buffers = {'owner': 'decode'}
        runner._qsa_uid_slots = [7, 7]
        runner.execute_model_state = 'decode-pending'
        with self.assertRaises(E3ContractError):
            rt.execute(runner, _prefill_stock(runner, [], pending=False), None,
                       transaction)
        self.assertEqual(runner.execute_model_state, 'decode-pending')

    def test_decode_execution_does_not_touch_prefill_private_state(self):
        state = worker()
        rt = runtime(state)
        open_prefill(rt, state)
        decode_transaction, _ = open_decode(rt, state)
        runner = _Runner()
        runner.input_buffers = {'owner': 'decode'}
        runner._qsa_uid_slots = [7, 7]
        runner.execute_model_state = None
        seen = []

        def stock(scheduler_output, *args):
            seen.append(dict(runner.input_buffers))
            runner.input_buffers = {'owner': 'decode-mutated'}
            return 'decode-out'

        output = rt.execute(runner, stock, None, decode_transaction)
        self.assertEqual(output, 'decode-out')
        self.assertEqual(seen, [{'owner': 'decode'}])
        self.assertEqual(runner.input_buffers, {'owner': 'decode-mutated'})
        self.assertEqual(rt.private_attrs['input_buffers'], {'owner': 'prefill'})
        self.assertIsNone(rt.prefill_execution)
        self.assertEqual(rt.real_frames, 1)

    def test_decode_execution_requires_the_decode_lane(self):
        state = worker()
        rt = runtime(state)
        transaction, _ = metadata(state, 'decode-u', 1, 0, 1, (0,))
        with self.assertRaises(E3ContractError):
            rt.execute(_Runner(), lambda *a: 'nope', None, transaction)


class TestBoundedRuntime(unittest.TestCase):
    def test_poll_does_not_wait_and_submit_is_bounded(self):
        class Backend:
            def __init__(self):
                self.submitted = []
                self.ready = []
                self.events = []

            def submit(self, ticket_value):
                self.submitted.append(ticket_value)

            def poll(self, transaction):
                self.events.append('poll')
                return tuple(self.ready)

        state = worker()
        rt = runtime(state)
        backend = Backend()
        rt.backend = backend
        transaction, _, _ = open_prefill(rt, state)
        ticket_value = rt.submit(LaneRole.PREFILL, transaction, 0,
                                 DependencyRole.PRODUCER)
        self.assertEqual(backend.submitted, [ticket_value])
        self.assertEqual(rt.poll(transaction), PollResult((), 1, False))
        backend.ready.append(proof(ticket_value, seq=1))
        self.assertEqual(rt.poll(transaction),
                         PollResult((proof(ticket_value, seq=1),), 0, False))
        rt.release(transaction)
        self.assertEqual(state.prefill.state, 'free')

    def test_unknown_role_is_rejected(self):
        state = worker()
        rt = runtime(state)
        transaction, _, _ = open_prefill(rt, state)
        with self.assertRaises(E3ContractError):
            rt.submit('prefill', transaction, 0, DependencyRole.PRODUCER)

    def test_poll_of_unknown_transaction_is_rejected(self):
        state = worker()
        rt = runtime(state)
        transaction, snapshot = metadata(state)
        with self.assertRaises(E3ContractError):
            rt.poll(transaction)
        rt.open_prefill(transaction, snapshot)
        other, _ = metadata(state, start=4, end=8)
        with self.assertRaises(E3ContractError):
            rt.poll(other)


def _execute(stock, output, dummy=False, profile=False, **fields):
    runner = _Runner()
    seen = []

    def wrapped(scheduler_output, *args):
        seen.append((scheduler_output, args))
        return stock(scheduler_output, *args)

    result = lane_hooks.execute_with_e3(runner, wrapped, output,
                                        dummy_run=dummy, is_profile=profile, **fields)
    return runner, seen, result


class TestWarmupRouting(unittest.TestCase):
    def test_off_mode_is_stock(self):
        calls = []

        def stock(*args):
            calls.append(args)
            return 'out'

        runner = _Runner()
        self.assertEqual(lane_hooks.execute_with_e3(runner, stock, NS(), 'inter'),
                         'out')
        self.assertEqual(len(calls), 1)

    def test_vllm_warmup_passes_through_but_real_frames_do_not(self):
        state = worker()
        rt = runtime(state)
        runner = _Runner()
        runner._flash_e3_lane = rt
        calls = []

        def stock(*args):
            calls.append(args)
            return 'warm'

        warm = NS(num_scheduled_tokens={'_warmup_x': 1})
        self.assertEqual(lane_hooks.execute_with_e3(runner, stock, warm), 'warm')
        self.assertEqual(rt.warmup_rpcs, 1)
        self.assertEqual(rt.real_frames, 0)
        with self.assertRaises(E3ContractError):
            lane_hooks.execute_with_e3(
                runner, stock, NS(num_scheduled_tokens={'real-request': 1}))
        self.assertEqual(len(calls), 1)

    def test_mixed_or_late_warmup_ids_fail(self):
        state = worker()
        rt = runtime(state)
        runner = _Runner()
        runner._flash_e3_lane = rt

        def stock(*args):
            return 'x'

        with self.assertRaises(E3ContractError):
            lane_hooks.execute_with_e3(
                runner, stock,
                NS(scheduled_new_reqs=[NS(req_id='_warmup_a'),
                                       NS(req_id='request-b')]))
        rt.real_frames = 1
        with self.assertRaises(E3ContractError):
            lane_hooks.execute_with_e3(
                runner, stock, NS(num_scheduled_tokens={'_warmup_x': 1}))

    def test_e2_frame_blocks_warmup_discrimination(self):
        state = worker()
        rt = runtime(state)
        runner = _Runner()
        runner._flash_e3_lane = rt
        with self.assertRaises(E3ContractError):
            lane_hooks.execute_with_e3(
                runner, lambda *a: 'x',
                NS(num_scheduled_tokens={'_warmup_x': 1}, flash_e2_frame=object()))

    def test_dummy_and_profile_bypass(self):
        state = worker()
        rt = runtime(state)
        runner = _Runner()
        runner._flash_e3_lane = rt
        self.assertEqual(lane_hooks.execute_with_e3(
            runner, lambda *a: 'd', NS(), dummy_run=True), 'd')
        self.assertEqual(lane_hooks.execute_with_e3(
            runner, lambda *a: 'p', NS(), is_profile=True), 'p')
        self.assertEqual(rt.real_frames, 0)
        self.assertEqual(rt.warmup_rpcs, 0)

    def test_capture_flags_on_real_frame_fail(self):
        state = worker()
        rt = runtime(state)
        runner = _Runner()
        runner._flash_e3_lane = rt
        with self.assertRaises(E3ContractError):
            lane_hooks.execute_with_e3(
                runner, lambda *a: 'x', NS(), skip_attn_for_dummy_run=True)

    def test_prefill_transaction_is_routed(self):
        state = worker()
        rt = runtime(state)
        transaction, _, _ = open_prefill(rt, state)
        rt.submit(LaneRole.PREFILL, transaction, 0, DependencyRole.CONSUMER)
        runner = _Runner()
        runner._flash_e3_lane = rt
        runner.input_buffers = {'owner': 'decode'}
        runner._qsa_uid_slots = [7, 7]
        runner.execute_model_state = 'decode-pending'
        seen = []

        def stock(scheduler_output, *args):
            seen.append(runner.input_buffers)
            runner.execute_model_state = 'prefill-pending'
            return 'prefill-out'

        output = lane_hooks.execute_with_e3(
            runner, stock, NS(flash_e3_transaction=transaction))
        self.assertEqual(output, 'prefill-out')
        self.assertEqual(seen, [{'owner': 'prefill'}])
        self.assertEqual(rt.prefill_execution, 'prefill-pending')
        self.assertEqual(runner.input_buffers, {'owner': 'decode'})

    def test_invalid_transaction_is_rejected(self):
        state = worker()
        rt = runtime(state)
        runner = _Runner()
        runner._flash_e3_lane = rt
        with self.assertRaises(E3ContractError):
            lane_hooks.execute_with_e3(
                runner, lambda *a: 'x', NS(flash_e3_transaction='not-a-txn'))


class TestInitLane(unittest.TestCase):
    def test_off_is_inert(self):
        runner = _Runner()
        with mock.patch.dict(os.environ, {'FLASH_S5_OVERLAP': '0'}):
            self.assertIsNone(lane_hooks.init_lane(runner, {}))
        self.assertIsNone(getattr(runner, '_flash_e3_lane', None))

    def test_on_requires_builder_and_runtime(self):
        runner = _Runner()
        with mock.patch.dict(os.environ, {'FLASH_S5_OVERLAP': '1'}):
            with self.assertRaises(E3ContractError):
                lane_hooks.init_lane(runner, {})
            runner._flash_e3_build_state = lambda r, kv: 'not-a-runtime'
            with self.assertRaises(E3ContractError):
                lane_hooks.init_lane(runner, {})

        state = worker()
        rt = runtime(state)
        runner = _Runner()
        runner._flash_e3_build_state = lambda r, kv: rt
        with mock.patch.dict(os.environ, {'FLASH_S5_OVERLAP': '1'}):
            self.assertIs(lane_hooks.init_lane(runner, {'kv': 1}), rt)
            self.assertIs(runner._flash_e3_lane, rt)
            with self.assertRaises(E3ContractError):
                lane_hooks.init_lane(runner, {'kv': 1})


class TestSourceTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()
        cls.generated = lane_hooks.generate(cls.source)

    def test_source_is_the_pinned_runner(self):
        self.assertTrue(SOURCE.is_file())
        self.assertEqual(hashlib.sha256(self.source.encode()).hexdigest(),
                         lane_hooks.PINNED_SHA256)
        self.assertEqual(lane_hooks.source_kind(self.source), 'pinned')

    def test_all_seams_present_and_parse(self):
        ast.parse(self.generated)
        self.assertEqual(self.generated.count('_flash_e3_hooks.'), 13)
        for name in ('before_prepare_inputs', 'prepare_attn_uids', 'new_uid',
                     'write_uids', 'host_copies', 'check_remove', 'check_add',
                     'check_update', 'note_batch', 'check_sample_tokens',
                     'defer_num_computed', 'init_lane'):
            self.assertIn('_flash_e3_hooks.' + name, self.generated)
        self.assertIn('def _flash_e3_stock_execute_model(', self.generated)
        self.assertIn('vllm.v1.flash_e3.lane_hooks', self.generated)

    def test_revert_recovers_pinned_bytes(self):
        self.assertEqual(lane_hooks.revert(self.generated), self.source)
        self.assertEqual(lane_hooks.revert(self.generated, expected=self.source),
                         self.source)
        with self.assertRaises(ValueError):
            lane_hooks.revert(self.generated + '\n')

    def test_fail_closed_on_unreviewed_or_double_transform(self):
        with self.assertRaises(ValueError):
            lane_hooks.generate(self.source + '\n')
        with self.assertRaises(ValueError):
            lane_hooks.generate(self.generated)
        with self.assertRaises(ValueError):
            lane_hooks.revert(self.source)

    def test_fail_closed_on_missing_and_duplicate_anchors(self):
        anchor = '''        copy_kv_cache_blocks_inplace(views, num_blocks, copies)
'''
        modified = self.source.replace(anchor, '', 1)
        digest = hashlib.sha256(modified.encode()).hexdigest()
        with mock.patch.object(lane_hooks, 'PINNED_SHA256', digest):
            with self.assertRaises(ValueError):
                lane_hooks.generate(modified)

        duplicated = self.source + '    def sample_tokens(\n'
        digest = hashlib.sha256(duplicated.encode()).hexdigest()
        with mock.patch.object(lane_hooks, 'PINNED_SHA256', digest):
            with self.assertRaises(ValueError):
                lane_hooks.generate(duplicated)

    def test_replace_helper(self):
        with self.assertRaises(ValueError):
            lane_hooks._replace('one', 'two', 'three')
        with self.assertRaises(ValueError):
            lane_hooks._replace('one one', 'one', 'two')
        self.assertEqual(lane_hooks._replace('one', 'one', 'two'), 'two')

    def test_e2_composition_order(self):
        e2_source = runner_hooks.runner_execution(self.source)
        self.assertIn('@execute_with_e2\n', e2_source)
        composed = lane_hooks.generate(e2_source)
        self.assertEqual(lane_hooks.source_kind(e2_source), 'e2')
        with self.assertRaises(ValueError):
            lane_hooks.source_kind(composed)
        self.assertIn('_flash_e3_stock_execute_model', composed)
        self.assertEqual(lane_hooks.revert(composed, expected=e2_source), e2_source)
        with self.assertRaises(ValueError):
            lane_hooks.revert(composed)
        with self.assertRaises(ValueError):
            runner_hooks.runner_execution(composed)

    def test_warmup_id_helper_never_logs(self):
        class Request:
            def __init__(self, req_id):
                self.req_id = req_id

            def __repr__(self):
                raise AssertionError('request repr must not be used')

        output = NS(scheduled_new_reqs=[Request('_warmup_a')],
                    scheduled_cached_reqs=NS(req_ids=['_warmup_b']),
                    num_scheduled_tokens={'_v2_mixed_warmup_c': 1},
                    finished_req_ids={'_warmup_d'})
        self.assertEqual(lane_hooks.warmup_request_ids(output),
                         ('_warmup_a', '_warmup_b', '_v2_mixed_warmup_c',
                          '_warmup_d'))
        self.assertEqual(lane_hooks.warmup_request_ids(NS()), ())


class TestPinnedMethodExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generated = lane_hooks.generate(SOURCE.read_text())

    def test_prepare_attn_uses_private_uids_only_for_prefill(self):
        class BlockTables:
            def gather_block_tables(self, *a, **kw): return 'block-tables'
            def compute_slot_mappings(self, *a, **kw): return 'slot-mappings'
        state = worker(); rt = runtime(state); open_prefill(rt, state)
        calls = []
        runner = NS(_flash_e3_lane=rt, block_tables=BlockTables(), pcp_manager=None,
                    _qsa_write_req_uids=lambda batch: calls.append(rt.uid_namespace.QSA_REQ_UID))
        batch = NS(idx_mapping_np=[0, 1], idx_mapping='idx', num_reqs_after_padding=2,
                   query_start_loc='qsl', positions='pos', num_tokens_after_padding=4)
        method = load_method(self.generated, 'prepare_attn')
        rt._preparing_prefill = True
        with rt._swap_prefill(runner):
            method(runner, batch)
        self.assertIs(calls[0], rt.prefill_uid_buffer)
        rt._preparing_prefill = False
        method(runner, batch)
        self.assertIs(calls[1], rt.uid_namespace.QSA_REQ_UID)
        self.assertIsNot(calls[0], calls[1])

    def test_postprocess_defers_only_while_preparing_prefill(self):
        calls = []

        def post_update(*args):
            calls.append(args)

        method = load_method(self.generated, 'postprocess_num_computed_tokens',
                             {'post_update_num_computed_tokens': post_update})
        state = worker()
        rt = runtime(state)
        open_prefill(rt, state)
        input_batch = NS(idx_mapping=[0], query_start_loc=[0, 4],
                         idx_mapping_np=[0], query_start_loc_np=[0, 4])
        runner = NS(req_states=NS(num_computed_tokens=NS(gpu='gpu')),
                    _flash_e3_lane=rt)

        rt._preparing_prefill = True
        method(runner, input_batch)
        self.assertEqual(calls, [])
        self.assertEqual(rt._pending_num_computed, ((0,), (0, 4)))
        rt._preparing_prefill = False
        method(runner, input_batch)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ([0], 'gpu', [0, 4]))

    def test_sample_tokens_rejects_prefill_execution_slot(self):
        method = load_method(self.generated, 'sample_tokens')
        rt = runtime(worker()); rt.prefill_execution = object()
        runner = NS(execute_model_state=rt.prefill_execution, _flash_e3_lane=rt)
        with self.assertRaises(E3ContractError): method(runner, None)
        runner.execute_model_state = None
        self.assertIsNone(method(runner, None))  # parked prefill must not block decode
        runner.execute_model_state = object()
        rt.check_sample_tokens(runner)

    def test_qsa_new_uid_prefill_and_decode_paths(self):
        fake_torch = NS(full=lambda shape, fill, **kwargs: [-2] * shape[0], int32='int32')
        method = load_method(self.generated, '_qsa_new_uid', {'torch': fake_torch})
        state = worker(); rt = runtime(state); open_prefill(rt, state)
        runner = NS(max_num_reqs=3, device='cpu', _flash_e3_lane=rt)
        rt._preparing_prefill = True
        method(runner, 0)
        self.assertEqual(runner._qsa_uid_slots, [1, -2, -2])
        rt._preparing_prefill = False
        method(runner, 1)
        self.assertEqual(runner._qsa_uid_slots, [1, 2, -2])
        self.assertEqual(runner._qsa_uid_next, 2)

    def test_remove_request_blocks_active_prefill_only(self):
        calls = []
        method = load_method(self.generated, '_remove_request')
        state = worker()
        rt = runtime(state)
        transaction, _, _ = open_prefill(rt, state)

        def remove(name):
            def inner(req_id):
                calls.append((name, req_id))
                return 0
            return inner

        runner = NS(_flash_e3_lane=rt,
                    model_state=NS(remove_request=remove('model')),
                    req_states=NS(remove_request=remove('req')),
                    pooling_runner=None, pp_handler=None, encoder_cache=None,
                    prompt_logprobs_worker=None,
                    lora_state=NS(remove_request=remove('lora')))
        with self.assertRaises(E3Pending):
            method(runner, transaction.request_uid)
        self.assertEqual(calls, [])
        self.assertTrue(method(runner, 'other-request'))
        self.assertEqual(calls, [('model', 'other-request'), ('req', 'other-request'),
                                 ('lora', 'other-request')])

    def test_host_copies_and_update_and_add_seams(self):
        state = worker()
        rt = runtime(state)
        transaction, _, _ = open_prefill(rt, state)
        runner = NS(_flash_e3_lane=rt)
        self.assertEqual(lane_hooks.host_copies(runner, [1, 2], ['v'], 7),
                         (2, 1, 7))
        self.assertEqual(rt._host_copies, [(2, 1, 7)])
        lane_hooks.check_add(runner, NS(req_id=transaction.request_uid,
                                        request_generation=1))
        with self.assertRaises(E3ContractError):
            lane_hooks.check_add(runner, NS(req_id=transaction.request_uid,
                                            request_generation=0))

        ticket_value = rt.submit(LaneRole.PREFILL, transaction, 0,
                                 DependencyRole.PRODUCER)
        with self.assertRaises(E3Pending):
            lane_hooks.check_update(
                runner, NS(block_table_updates={0: [10, 99]}))
        rt.mutation_guard = lambda output, transaction: True  # C supplies disjoint-lease proof
        lane_hooks.check_update(runner, NS(block_table_updates={0: [99]}))
        rt.release(transaction, (proof(ticket_value),))

    def test_before_prepare_inputs_checks_leased_batch(self):
        state = worker()
        rt = runtime(state)
        transaction, snapshot, _ = open_prefill(rt, state)
        runner = NS(_flash_e3_lane=rt, execute_model_state=None)
        rt._preparing_prefill = True
        lane_hooks.before_prepare_inputs(
            runner, NS(), NS(req_ids=(transaction.request_uid,), idx_mapping_np=[0]))
        with self.assertRaises(E3ContractError):
            lane_hooks.before_prepare_inputs(
                runner, NS(), NS(req_ids=('other',), idx_mapping_np=[0]))
        rt._preparing_prefill = False
        runner.execute_model_state = 'decode-pending'
        with self.assertRaises(E3Pending):
            lane_hooks.before_prepare_inputs(runner, NS(), NS(req_ids=()))

    def test_note_batch_and_uid_write_seams(self):
        state = worker()
        rt = runtime(state)
        transaction, _, _ = open_prefill(rt, state)
        runner = NS(_flash_e3_lane=rt)
        snapshot = lane_hooks.note_batch(
            runner, NS(flash_e3_transaction=transaction),
            NS(req_ids=(transaction.request_uid,), idx_mapping_np=[0],
               num_computed_prefill_tokens_np=[0]))
        self.assertEqual(snapshot.rows[0].request_uid, transaction.request_uid)
        other, _ = metadata(state, start=4, end=8)
        self.assertIsNone(lane_hooks.note_batch(
            runner, NS(flash_e3_transaction=other),
            NS(req_ids=(transaction.request_uid,), idx_mapping_np=[0],
               num_computed_prefill_tokens_np=[0])))
        with self.assertRaises(E3ContractError):
            lane_hooks.note_batch(
                runner, NS(flash_e3_transaction=transaction),
                NS(req_ids=('unknown-request',), idx_mapping_np=[0],
                   num_computed_prefill_tokens_np=[0]))
        rt._preparing_prefill = True
        self.assertFalse(lane_hooks.prepare_attn_uids(
            runner, NS(idx_mapping_np=[0])))
        self.assertEqual(rt.prefill_uid_buffer, [-2, -2])
        self.assertFalse(lane_hooks.new_uid(runner, 0))
        self.assertFalse(lane_hooks.write_uids(runner, NS(idx_mapping_np=[0])))
        self.assertEqual(rt.prefill_uid_buffer, [-2, -2])
        self.assertTrue(lane_hooks.defer_num_computed(
            runner, NS(idx_mapping_np=[0], query_start_loc_np=[0, 4])))
        self.assertEqual(rt._pending_num_computed, ((0,), (0, 4)))


class TestReviewRegressions(unittest.TestCase):
    def test_default_off_decorators_stay_on_original_methods(self):
        def decorators(source):
            cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'GPUModelRunner')
            return {n.name: [ast.unparse(d) for d in n.decorator_list]
                    for n in cls.body if isinstance(n, ast.FunctionDef)}
        source = SOURCE.read_text(); before = decorators(source)
        after = decorators(lane_hooks.generate(source))
        self.assertEqual(after['sample_tokens'], before['sample_tokens'])
        self.assertEqual(after['_flash_e3_stock_execute_model'], before['execute_model'])
        self.assertEqual(after['execute_model'], ['torch.inference_mode()'])

    def test_e2_source_markers_do_not_bypass_source_pin(self):
        original = runner_hooks.runner_execution(SOURCE.read_text())
        with self.assertRaises(ValueError):
            lane_hooks.generate(original + '\n# unreviewed change\n')
        full = runner_hooks.runner_startup(original)
        generated = lane_hooks.generate(full)
        self.assertEqual(lane_hooks.revert(generated, expected=full), full)
        self.assertLess(generated.index('_install_flash_e2(self, is_profiling)'),
                        generated.index('_flash_e3_hooks.init_lane(self, kv_caches_dict, is_profiling)'))

    def test_workspace_partial_enter_restores_context(self):
        class Partial(Workspace):
            def enter(self, value):
                super().enter(value)
                raise RuntimeError('partial workspace failure')
        fc = ForwardContext(); ws = Partial(); dispatcher = lane_hooks.HostDispatcher(fc, ws)
        with self.assertRaises(RuntimeError):
            with dispatcher.enter(None, 'prefill', 'pw'): pass
        self.assertEqual(fc.get(), 'decode-ctx'); self.assertEqual(ws.current(), 'decode-ws')
        self.assertFalse(dispatcher.active)

    def test_forward_context_set_failure_restores_and_clears_active(self):
        class Partial(ForwardContext):
            def set(self, value):
                super().set(value)
                if value == 'prefill': raise RuntimeError('partial context failure')
        fc = Partial(); dispatcher = lane_hooks.HostDispatcher(fc, Workspace())
        with self.assertRaises(RuntimeError):
            with dispatcher.enter(None, 'prefill', 'pw'): pass
        self.assertEqual(fc.get(), 'decode-ctx'); self.assertFalse(dispatcher.active)

    def test_native_uid_writer_targets_prefill_buffer_then_restores_decode(self):
        class View:
            def __init__(self, parent, section): self.parent, self.section = parent, section
            def copy_(self, other): self.parent.values[self.section] = list(other.values)
        class Buffer:
            def __init__(self, values): self.values = list(values)
            def fill_(self, value): self.values[:] = [value] * len(self.values)
            def __getitem__(self, key):
                if isinstance(key, slice): return View(self, key)
                if isinstance(key, list): return Buffer(self.values[k] for k in key)
                return self.values[key]
            def __setitem__(self, key, value): self.values[key] = value
        state = worker(); rt = runtime(state); tx, _, _ = open_prefill(rt, state)
        decode_buf, prefill_buf = Buffer([88, 88]), Buffer([-2, -2])
        rt.uid_namespace.QSA_REQ_UID = decode_buf; rt.prefill_uid_buffer = prefill_buf
        runner = NS(_flash_e3_lane=rt, _qsa_uid_slots=Buffer([41, 42]), _qsa_uid_next=42)
        method = load_method(lane_hooks.generate(SOURCE.read_text()), '_qsa_write_req_uids', {'_qsa_fc': rt.uid_namespace})
        batch = NS(num_reqs=1, idx_mapping=[1])
        rt._prefill_running = True
        with rt._swap_prefill(runner): method(runner, batch)
        rt._prefill_running = False
        self.assertEqual(prefill_buf.values, [42, -2]); self.assertEqual(decode_buf.values, [88, 88])
        method(runner, NS(num_reqs=1, idx_mapping=[0]))
        self.assertEqual(decode_buf.values, [41, -2]); self.assertEqual(prefill_buf.values, [42, -2])
        rt.release(tx)
        open_prefill(rt, state, uid='next-request')
        new_uid = load_method(lane_hooks.generate(SOURCE.read_text()), '_qsa_new_uid')
        rt._prefill_running = True
        new_uid(runner, 1)
        self.assertEqual(runner._qsa_uid_slots.values, [41, 43])

    def test_state_proof_must_be_submitted_and_completed_before_commit(self):
        state = worker(); rt = runtime(state); tx, _, _ = open_prefill(rt, state)
        rt.propose_state(tx, (1,))
        t = ticket(rt, tx, role=DependencyRole.COMMIT); done = proof(t, CompletionKind.STATE_WRITE)
        with self.assertRaises(E3ContractError): rt.commit(tx, done)
        state.prefill.submit(t)
        with self.assertRaises(E3ContractError): rt.commit(tx, done)
        state.prefill.complete(done)
        later = rt.submit(LaneRole.PREFILL, tx, 1, DependencyRole.CONSUMER)
        with self.assertRaises(E3Pending): rt.commit(tx, done)
        state.prefill.complete(proof(later, seq=1))
        self.assertEqual(rt.commit(tx, done).value, (1,))

    def test_cpu_snapshots_reject_native_mutable_aliases(self):
        state = worker(); rt = runtime(state); tx, snap, _ = open_prefill(rt, state)
        inner = [1]; state.prefill.bind_recurrent((inner,)); inner.append(2)
        self.assertEqual(state.prefill.recurrent.values, ((1,),))
        with self.assertRaises(E3ContractError): state.prefill.bind_recurrent(bytearray(2))
        with self.assertRaises(E3ContractError): MetadataSnapshot(tx, list(snap.rows))

    def test_history_capacity_comes_from_allocation_not_200k_slice(self):
        for pages in (50, 65, 935, 1340):
            self.assertEqual(QsaHistory('layer', 0, 4096, pages * PAGE_BYTES).nbytes, pages * PAGE_BYTES)
        with self.assertRaises(E3ContractError): QsaHistory('layer', 0, 4096, 935 * PAGE_BYTES + 1)

    def test_update_guard_is_before_every_worker_mutation(self):
        generated = lane_hooks.generate(SOURCE.read_text())
        source = method_source(generated, 'update_requests')
        self.assertLess(source.index('_flash_e3_hooks.check_update'), source.index('table_updates ='))
        state = worker(); rt = runtime(state); tx, _, _ = open_prefill(rt, state)
        rt.submit(LaneRole.PREFILL, tx, 0, DependencyRole.CONSUMER)
        for fields in ({'block_table_updates': {'req': ([10], [20])}},
                       {'new_block_ids_to_zero': [10]}, {'kv_cache_block_copies': [object()]},
                       {'scheduled_cached_reqs': NS(new_block_ids=[([10],)])}):
            with self.assertRaises(E3Pending): rt.check_update(NS(), NS(**fields))
        rt.mutation_guard = lambda output, txn: txn == tx
        rt.check_update(NS(), NS(kv_cache_block_copies=[object()]))

    def test_shared_storage_exclusion_is_enforced_at_submission(self):
        state = worker(); rt = runtime(state)
        shared = ScratchSpec('arena', 'shared', 'serialized', 100, shared=True, reuse_condition='consumer complete')
        tx, _, _ = open_prefill(rt, state, bindings=[(shared, 1000)])
        dtx, _ = open_decode(rt, state)
        state.decode.bind_scratch(shared, 1000)
        pre = rt.submit(LaneRole.PREFILL, tx, 0, DependencyRole.CONSUMER)
        with self.assertRaises(E3Pending): rt.submit(LaneRole.DECODE, dtx, 0, DependencyRole.CONSUMER)
        state.prefill.complete(proof(pre))
        rt.submit(LaneRole.DECODE, dtx, 0, DependencyRole.CONSUMER)

    def test_operation_credits_are_finite(self):
        state = worker(); state.max_outstanding = 2; rt = runtime(state)
        tx, _, _ = open_prefill(rt, state)
        rt.submit(LaneRole.PREFILL, tx, 0, DependencyRole.CONSUMER)
        rt.submit(LaneRole.PREFILL, tx, 1, DependencyRole.CONSUMER)
        with self.assertRaises(E3Pending): rt.submit(LaneRole.PREFILL, tx, 2, DependencyRole.CONSUMER)

    def test_profile_startup_does_not_construct_runtime(self):
        with mock.patch.dict(os.environ, {'FLASH_S5_OVERLAP': '1'}):
            self.assertIsNone(lane_hooks.init_lane(NS(), {}, is_profiling=True))


    def test_module_scratch_rebinding_restores_decode_and_retains_prefill(self):
        state = worker(); rt = runtime(state); tx, md = metadata(state)
        owner = NS(scratch={'lane': 'decode'})
        binding = lane_hooks.AttributeBinding(owner, 'scratch', {'lane': 'prefill'})
        rt.open_prefill(tx, md, private_attrs={}, object_bindings=(binding,))
        runner = NS(); before = owner.scratch
        with rt._swap_prefill(runner):
            self.assertEqual(owner.scratch, {'lane': 'prefill'})
            owner.scratch = {'lane': 'replacement'}
        self.assertIs(owner.scratch, before)
        self.assertEqual(binding.value, {'lane': 'replacement'})
        with rt._swap_prefill(runner): self.assertIs(owner.scratch, binding.value)

    def test_abort_retains_pending_outputs_until_consumer_completes(self):
        state = worker(); rt = runtime(state); tx, _, _ = open_prefill(rt, state)
        held = object(); rt.prefill_execution = held
        t = rt.submit(LaneRole.PREFILL, tx, 0, DependencyRole.CONSUMER)
        rt.abort(tx)
        self.assertIs(rt.prefill_execution, held)
        with self.assertRaises(E3Pending): rt.release(tx)
        rt.release(tx, [proof(t)])
        self.assertIsNone(rt.prefill_execution)

    def test_distinct_tensor_views_cannot_alias_uid_storage(self):
        class Tensor:
            device = 'cuda:0'
            def untyped_storage(self): return NS(data_ptr=lambda: 4096, nbytes=lambda: 100)
        rt = runtime(worker()); open_prefill(rt, rt.state)
        rt.uid_namespace.QSA_REQ_UID = Tensor(); rt.prefill_uid_buffer = Tensor()
        with self.assertRaises(E3ContractError):
            with rt._swap_prefill(NS()): pass

    def test_admission_truth_is_not_overwritten_by_current_batch(self):
        state = worker(); rt = runtime(state); tx, md, _ = open_prefill(rt, state)
        before = state.prefill._latest_batch
        with self.assertRaises(E3ContractError):
            rt.note_batch(NS(), NS(flash_e3_transaction=tx),
                          NS(req_ids=[tx.request_uid], idx_mapping_np=[9]))
        self.assertIs(state.prefill._latest_batch, before)
        self.assertIs(state.prefill.metadata, md)

    def test_poll_rejects_unbounded_completion_iterators(self):
        state = worker(); rt = runtime(state); tx, _, _ = open_prefill(rt, state)
        rt.backend = NS(poll=lambda transaction: iter(()))
        with self.assertRaises(E3ContractError): rt.poll(tx)
        self.assertTrue(state.prefill.failed)

    def test_submit_does_not_lose_completion_in_different_backend(self):
        state = worker(); rt = runtime(state); tx, _, _ = open_prefill(rt, state)
        with self.assertRaises(E3ContractError):
            rt.submit(LaneRole.PREFILL, tx, 0, DependencyRole.CONSUMER, backend=object())
        self.assertFalse(state.prefill.outstanding)


    def test_decode_cannot_enter_under_prefill_python_context(self):
        state = worker(); rt = runtime(state); open_prefill(rt, state)
        dtx, _ = open_decode(rt, state)
        with rt.dispatcher.enter(rt.prefill_handle, 'prefill', 'pw'):
            with self.assertRaises(E3Pending): rt.execute(NS(), lambda *a: self.fail('ran'), NS(), dtx)

    def test_foreign_thread_cannot_enter_dispatcher(self):
        import threading
        rt = runtime(worker()); errors = []
        def run():
            try:
                with rt.dispatcher.enter(None, 'foreign', 'workspace'): pass
            except E3ContractError: errors.append('refused')
        t = threading.Thread(target=run); t.start(); t.join(2)
        self.assertEqual(errors, ['refused']); self.assertFalse(rt.dispatcher.active)

    def test_prepare_and_execute_need_registered_completion_ownership(self):
        state = worker(); rt = runtime(state); tx, _, _ = open_prefill(rt, state)
        with self.assertRaises(E3ContractError): rt.execute(NS(), lambda *a: None, NS(), tx)
        with self.assertRaises(E3ContractError): rt.prepare_prefill(NS(), NS(), NS(), NS())
        rt.submit(LaneRole.PREFILL, tx, 0, DependencyRole.PRODUCER)
        runner = NS(input_buffers={'decode': True})
        def prepare(output, batch, descriptor):
            self.assertIs(rt.uid_namespace.QSA_REQ_UID, rt.prefill_uid_buffer)
            self.assertEqual(runner.input_buffers, {'owner': 'prefill'})
            return 'prepared'
        runner.prepare_inputs = prepare
        self.assertEqual(rt.prepare_prefill(runner, NS(), NS(), NS()), 'prepared')
        self.assertEqual(runner.input_buffers, {'decode': True})
        with self.assertRaises(E3Pending): rt.release(tx)


    def test_enabled_startup_rejects_null_backend(self):
        rt = runtime(worker()); rt.backend = lane_hooks._NullBackend()
        runner = NS(_flash_e3_build_state=lambda runner, caches: rt)
        with mock.patch.dict(os.environ, {'FLASH_S5_OVERLAP': '1'}):
            with self.assertRaises(E3ContractError): lane_hooks.init_lane(runner, {})
        self.assertFalse(hasattr(runner, '_flash_e3_lane'))


if __name__ == '__main__':
    unittest.main()
