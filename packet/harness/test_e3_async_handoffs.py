"""C's async handoff contract: CPU state machine, leases and source seams.

Everything here runs offline with deterministic fakes. It is not GPU evidence:
the native completion backend, shared-memory transport, CUDA/PP buffers and the
first four-worker engine smoke remain owner-run acceptance steps.
"""
import ast
from contextlib import ExitStack
from enum import Enum
import hashlib
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import async_handoff as ah
import executor_hooks as eh
import lane_hooks
import reply_transport
import scheduler_hooks as sh
import s5_startup
from e3_types import (
    CompletionKind,
    CompletionProof,
    DependencyRole,
    LaneId,
    LaneRole,
    OperationTicket,
    TransactionId,
)


class SUCCESS(Enum):
    OK = 1


def tx(seq=1, gen=1):
    return TransactionId(7, 'req-abcdef', gen, 0, 1024, seq)


def lanes(count=4):
    return {rank: LaneId(rank, LaneRole.PREFILL, 3, 11) for rank in range(count)}


def ticket(transaction, rank=0, seq=1, role=DependencyRole.PRODUCER, lane=None):
    lane = lane or LaneId(rank, LaneRole.PREFILL, 3, 11)
    return OperationTicket(transaction, rank, lane, seq, role)


def proof(item, kind=CompletionKind.GPU_EVENT, seq=1):
    return CompletionProof(item, kind, seq)


class FakeQueue:
    """MessageQueue-shaped blocking reader; one receiver thread owns it."""

    def __init__(self):
        self.q = queue.Queue()
        self._flash_s5_receiver_owner = None

    def dequeue(self, timeout=None):
        return self.q.get(timeout=timeout)

    def put(self, value):
        self.q.put(value)


class FakeScheduler:
    def __init__(self, pool):
        self._flash_s5_leases = None
        self.kv_cache_manager = type('Manager', (), {'block_pool': pool})()
        self._flash_s5_handoff = None


class FakeBlock:
    def __init__(self, block_id, ref_cnt=1, block_hash=None):
        self.block_id = block_id
        self.ref_cnt = ref_cnt
        self.block_hash = block_hash
        self.is_null = False
        self.pool = None


class FakePool:
    def __init__(self):
        self._flash_s5_leases = None
        self.freed = []

    def free_blocks(self, blocks):
        self.freed.extend(blocks)


class ConfigTests(unittest.TestCase):
    def test_default_off_is_identity_and_settings_fail_closed(self):
        cfg = ah.HandoffConfig.from_env({})
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg, ah.HandoffConfig.disabled())
        for env, pattern in (
                ({ah.OVERLAP_ENV: 'yes'}, 'must be 0 or 1'),
                ({ah.SPARSE_OVERLAP_ENV: '1'}, 'sparse-UVA'),
                ({ah.TRANSPORT_ENV: 'sparse_uva'}, 'unqualified overlap transport'),
                ({ah.COMMAND_SLOTS_ENV: '11'}, 'exceeds the pinned broadcast ring'),
                ({ah.REPLY_TIMEOUT_ENV: '0'}, 'positive')):
            with self.assertRaisesRegex(ah.ConfigurationError, pattern):
                ah.HandoffConfig.from_env(env)
        with self.assertRaises(ah.ConfigurationError):
            ah.HandoffConfig(True, ah.QUALIFIED_TRANSPORT, 11, 1, 1, 1024)

    def test_command_capacity_comes_from_the_actual_ring(self):
        ring = type('Q', (), {'buffer': type('B', (), {'max_chunks': 10})()})()
        self.assertEqual(ah.derive_command_capacity(ring), 10)
        self.assertEqual(ah.derive_command_capacity(ring, 7), 7)
        self.assertEqual(ah.derive_command_capacity(ring, 99), 10)
        with self.assertRaises(ah.ConfigurationError):
            ah.derive_command_capacity(type('Q', (), {})())
        with self.assertRaises(ah.ConfigurationError):
            ah.derive_command_capacity(ring, 0)

    def test_sparse_and_missing_dense_serialize_and_are_counted(self):
        sparse = ah.admit_route(overlap_enabled=True,
                                transport=ah.QUALIFIED_TRANSPORT,
                                sparse_active=True, dense_available=True)
        self.assertIs(sparse.route, ah.Route.SERIAL)
        self.assertTrue(sparse.serialized_fallback)
        self.assertEqual(sparse.sparse_histories, 6)
        missing = ah.admit_route(overlap_enabled=True,
                                 transport=ah.QUALIFIED_TRANSPORT,
                                 sparse_active=False, dense_available=False)
        self.assertIs(missing.route, ah.Route.SERIAL)
        self.assertFalse(missing.serialized_fallback)
        dense = ah.admit_route(overlap_enabled=True,
                               transport=ah.QUALIFIED_TRANSPORT,
                               sparse_active=False, dense_available=True)
        self.assertIs(dense.route, ah.Route.DENSE_DMA)

    def test_enabled_build_rejects_fixture_and_declares_budget(self):
        with self.assertRaises(ah.ConfigurationError):
            ah.build_composition({ah.OVERLAP_ENV: '1'}, transport=object(),
                                 backend=ah.FixtureCompletionBackend(),
                                 lanes=lanes(), native_allocations={'uid': 1})
        self.assertIsNone(ah.build_composition({}, transport=object(),
                                               backend=object(), lanes=lanes()))
        budget = ah.allocation_budget(1024, 10)
        self.assertEqual(budget.unmeasured(), ('shared_weights', 'marlin_workspace',
                                               'green_context', 'communicator_buffers',
                                               'prefill_uid_buffer'))
        names = {entry.name for entry in budget.entries}
        self.assertTrue({'prefill_pp_send_recv', 'command_metadata',
                         'telemetry_ring', 'prefill_uid_buffer'} <= names)


class TelemetryTests(unittest.TestCase):
    def test_metadata_only_bounded_and_payload_refused(self):
        t = ah.Telemetry(limit=2)
        t.record('reply_received', ack_lag_s=0.25, reason_code='not_ready')
        with self.assertRaises(ah.HandoffError):
            t.record('bad', token_ids=(1, 2))
        with self.assertRaises(ah.HandoffError):
            t.record('bad', header='secret')
        with self.assertRaises(ah.HandoffError):
            t.record('bad', reason_code='not-a-code')
        with self.assertRaises(ah.HandoffError):
            t.record('bad', ack_lag_s=float('inf'))
        t.record('one'); t.record('two'); t.record('three')
        self.assertEqual(len(t), 2)
        self.assertEqual(t.dropped, 2)


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.transaction = tx()
        self.t = ticket(self.transaction, seq=1)

    def test_ack_is_not_completion_or_publication(self):
        lease = ah.TransactionLease(self.transaction, lanes(), max_outstanding=4,
                                    clock=lambda: 0.0)
        lease.submit(self.t)
        lease.acknowledge(self.transaction, 1, True, worker_rank=0)
        self.assertEqual(lease.acked, 1)
        self.assertEqual(lease.pending, 1)
        with self.assertRaises(ah.HandoffError):
            lease.commit(proof(self.t, CompletionKind.STATE_WRITE))
        lease.complete(proof(self.t))
        self.assertEqual(lease.pending, 0)
        self.assertIsNot(lease.phase, ah.Phase.COMMITTED)
        with self.assertRaises(ah.HandoffError):
            lease.commit(proof(self.t))
        with self.assertRaises(ah.HandoffError):
            lease.commit(CompletionProof(self.t, CompletionKind.GPU_EVENT, 2))

    def test_commit_requires_commit_role_and_publication_is_once(self):
        lease = ah.TransactionLease(self.transaction, lanes(), max_outstanding=4,
                                    clock=lambda: 0.0)
        commit_tickets = [ticket(self.transaction, rank=r, seq=1,
                                 role=DependencyRole.COMMIT) for r in range(4)]
        for item in commit_tickets:
            lease.submit(item)
            lease.acknowledge(self.transaction, 1, True, worker_rank=item.worker_rank)
        states = [proof(t, CompletionKind.STATE_WRITE, seq=1) for t in commit_tickets]
        with self.assertRaises(ah.HandoffError):
            lease.commit(states[0])
        for state in states:
            lease.complete(state)
        self.assertEqual(lease.commit(states[0]), 1)
        with self.assertRaises(ah.HandoffError):
            lease.commit(states[0])

    def test_stale_duplicate_and_late_proofs_quarantine_without_mutation(self):
        lease = ah.TransactionLease(self.transaction, lanes(), max_outstanding=4,
                                    clock=lambda: 0.0)
        lease.submit(self.t)
        lease.acknowledge(self.transaction, 1, True, worker_rank=0)
        lease.complete(proof(self.t, seq=2))
        with self.assertRaises(ah.HandoffError):
            lease.complete(proof(self.t, seq=2))
        self.assertIs(lease.phase, ah.Phase.QUARANTINED)
        self.assertEqual(lease.quarantine, 'late_reply')
        foreign = ticket(tx(seq=2), seq=1)
        with self.assertRaises(ah.HandoffError):
            ah.TransactionLease(tx(), lanes(), max_outstanding=1,
                                clock=lambda: 0.0).complete(proof(foreign))

    def test_retired_lease_rejects_direct_late_reply(self):
        lease = ah.TransactionLease(self.transaction, lanes(), max_outstanding=4,
                                    clock=lambda: 0.0)
        lease.submit(self.t)
        lease.acknowledge(self.transaction, 1, True, worker_rank=0)
        lease.complete(proof(self.t))
        lease.start_stop('cancel')
        before = (lease.acked, lease.pending)
        with self.assertRaises(ah.HandoffError):
            lease.acknowledge(self.transaction, 1, True, worker_rank=0)
        self.assertIs(lease.phase, ah.Phase.QUARANTINED)
        self.assertEqual(before, (lease.acked, lease.pending))


class LifetimeTests(unittest.TestCase):
    def test_each_category_requires_its_consumer_proof(self):
        t = ticket(tx())
        ledger = ah.LifetimeLedger()
        ledger.acquire(t, ah.LeaseCategory.SOURCE_HISTORY, 'qsa-history')
        with self.assertRaises(ah.HandoffError):
            ledger.release(t, ah.LeaseCategory.SOURCE_HISTORY,
                           proof(t, CompletionKind.GPU_EVENT))
        with self.assertRaises(ah.HandoffError):
            ledger.release(t, ah.LeaseCategory.SOURCE_HISTORY,
                           CompletionProof(ticket(tx(seq=2)), CompletionKind.COPY_EVENT, 1))
        ledger.release(t, ah.LeaseCategory.SOURCE_HISTORY,
                       proof(t, CompletionKind.COPY_EVENT))
        with self.assertRaises(ah.HandoffError):
            ledger.release(t, ah.LeaseCategory.SOURCE_HISTORY,
                           proof(t, CompletionKind.COPY_EVENT))
        self.assertEqual(ledger.rejected_late, 1)
        ledger.note_cow(2, 2)
        ledger.assert_matched_cow()
        ledger.note_cow(1, 0)
        with self.assertRaises(ah.E3Pending):
            ledger.assert_matched_cow()
        ledger.cow_completions = ledger.cow_requests
        ledger.lease_caused_cow = 1
        with self.assertRaises(ah.HandoffError):
            ledger.assert_matched_cow()

    def test_frontier_publishes_once_and_truncates_unpublished(self):
        frontier = ah.CommittedFrontier()
        a, b = tx(seq=1), tx(seq=2)
        frontier.reserve(a, [10, 11, 12])
        self.assertEqual(frontier.lookup([10, 11, 12, 13]), ())
        frontier.publish(a)
        self.assertEqual(frontier.lookup([10, 11, 12, 13]), (10, 11, 12))
        with self.assertRaises(ah.E3Pending):
            frontier.shareable([10, 99])
        frontier.reserve(b, [99])
        self.assertEqual(frontier.lookup([10, 99]), (10,))
        with self.assertRaises(ah.HandoffError):
            frontier.publish(a)

    def test_prefill_pp_channel_has_independent_order_and_no_sample_ring(self):
        channel = ah.PrefillPPChannel(stages=2, max_outstanding=1)
        t1, t2 = ticket(tx(1), seq=1), ticket(tx(2), seq=1)
        channel.send(t1, 0)
        with self.assertRaises(ah.E3Pending):
            channel.send(t2, 0)
        with self.assertRaises(ah.HandoffError):
            channel.consume(t1, 0, proof(t1, CompletionKind.GPU_EVENT))
        channel.consume(t1, 0, proof(t1, CompletionKind.PP_CONSUMER))
        channel.send(t2, 1)
        self.assertEqual(channel.sends, 2)
        self.assertEqual(channel.consumed, 1)
        with self.assertRaises(ah.HandoffError):
            channel.get_prev_sampled_outputs()


class SchedulerLeaseTests(unittest.TestCase):
    def setUp(self):
        self.pool = FakePool()
        self.scheduler = FakeScheduler(self.pool)
        self.guard = sh.SchedulerLeases().attach(self.scheduler, self.pool)
        self.transaction = tx()
        self.t = ticket(self.transaction)
        self.blocks = [FakeBlock(1), FakeBlock(2)]

    def test_free_is_deferred_without_refcount_or_cow_change(self):
        self.guard.lease_blocks(self.t, ah.LeaseCategory.ARENA, [1, 2])
        live = sh.hold_leased_blocks(self.pool, self.blocks)
        self.assertEqual(live, ())
        self.assertEqual(self.pool.freed, [])
        self.assertEqual([b.ref_cnt for b in self.blocks], [1, 1])
        self.assertTrue(sh.block_held(self.pool, self.blocks[0]))
        self.assertEqual(self.guard.lease_caused_cow, 0)
        with self.assertRaises(ah.E3ContractError):
            self.guard.release(self.t, ah.LeaseCategory.ARENA,
                               proof(self.t, CompletionKind.COPY_EVENT))
        self.guard.release(self.t, ah.LeaseCategory.ARENA,
                           proof(self.t, CompletionKind.GPU_EVENT, 2))
        self.assertEqual(sorted(b.block_id for b in self.pool.freed), [1, 2])
        self.assertEqual([b.ref_cnt for b in self.blocks], [1, 1])
        with self.assertRaises(ah.E3ContractError):
            self.guard.release(self.t, ah.LeaseCategory.ARENA,
                               proof(self.t, CompletionKind.GPU_EVENT, 3))
        self.assertEqual(self.guard.rejected_late, 1)

    def test_committed_block_can_be_touched_uncommitted_waits(self):
        self.guard.lease_blocks(self.t, ah.LeaseCategory.ARENA, [1])
        with self.assertRaises(ah.E3Pending):
            sh.before_touch(self.pool, [self.blocks[0]])
        self.guard.publish(self.t)
        sh.before_touch(self.pool, [self.blocks[0]])

    def test_mutation_guard_refuses_other_leases_and_audits_copies(self):
        other_t = ticket(tx(seq=2))
        self.guard.lease_blocks(self.t, ah.LeaseCategory.ARENA, [1, 2])
        self.guard.lease_blocks(other_t, ah.LeaseCategory.ARENA, [3])
        out = type('Out', (), {
            'block_table_updates': ({'new_block_ids': [3]},),
            'new_block_ids_to_zero': (), 'kv_cache_block_copies': ()})()
        self.assertFalse(self.guard.mutation_guard(out, self.transaction))
        out.block_table_updates = ({'new_block_ids': [2]},)
        self.assertFalse(self.guard.mutation_guard(out, self.transaction))
        self.guard.publish(self.t)
        copy = type('Copy', (), {'src_block_id': 2, 'dst_block_id': 4})()
        out.block_table_updates = ()
        out.kv_cache_block_copies = (copy,)
        self.assertTrue(self.guard.mutation_guard(out, self.transaction))
        self.assertEqual(self.guard.cow_requested, 1)
        self.assertEqual(self.guard.cow_completed, 0)
        self.assertEqual(self.guard.lease_caused_cow, 0)

    def test_allocation_of_a_leased_block_and_unclean_shutdown_refused(self):
        self.guard.lease_blocks(self.t, ah.LeaseCategory.ARENA, [1])
        with self.assertRaises(ah.E3ContractError):
            sh.note_allocated(self.pool, [self.blocks[0]])
        with self.assertRaises(ah.E3ContractError):
            sh.before_scheduler_shutdown(self.scheduler)
        sh.hold_leased_blocks(self.pool, self.blocks)
        self.guard.release(self.t, ah.LeaseCategory.ARENA,
                           proof(self.t, CompletionKind.GPU_EVENT))
        self.assertEqual(sh.before_scheduler_shutdown(self.scheduler), dict(clean=True, held=0, pending_free=0))


class SourceTransformTests(unittest.TestCase):
    def test_scheduler_and_pool_transforms_are_exact_reversible_and_drift_closed(self):
        ref = ROOT / 'reference/vllm-src'
        for path, gen, rev, digest in (
                ('v1/core/sched/scheduler.py', sh.generate_scheduler, sh.revert_scheduler,
                 sh.SCHEDULER_E2_SHA256),
                ('v1/core/block_pool.py', sh.generate_block_pool, sh.revert_block_pool,
                 sh.BLOCK_POOL_E2_SHA256)):
            source = (ref / path).read_text()
            # Use the reviewed E2 composition as the input, not raw upstream.
            import mirror_hooks, telemetry_hooks
            if path.endswith('scheduler.py'):
                e2 = mirror_hooks.scheduler(telemetry_hooks.scheduler(source))
            else:
                e2 = mirror_hooks.block_pool(telemetry_hooks.block_pool(source))
            self.assertEqual(hashlib.sha256(e2.encode()).hexdigest(), digest)
            out = gen(e2)
            self.assertEqual(rev(out), e2)
            self.assertGreaterEqual(out.count('_flash_s5_hooks.'), 3)

    def test_worker_typed_rpc_dispatch_and_unknown_method_rejected(self):
        class Worker:
            def __init__(self):
                self.seen = []
                class Adapter:
                    def __init__(adapter, seen):
                        adapter.seen = seen

                    def open(adapter, message):
                        adapter.seen.append(message)
                        return 'ok'
                self._flash_s5_worker = Adapter(self.seen)
        worker = Worker()
        message = ah.OpenPrefill(tx(), 1024)
        self.assertEqual(eh.worker_rpc(worker, 'open', message), 'ok')
        self.assertEqual(worker.seen, [message])
        with self.assertRaises(ValueError):
            eh.worker_rpc(worker, 'unknown', message)
        with self.assertRaises(ValueError):
            eh.worker_rpc(worker, 'open', {'kind': 'open'})

    def test_local_future_never_drains_other_sequences(self):
        class FakeTransport:
            def __init__(self):
                self.taken = []
                self._calls = {0: type('C', (), {'ready': True})(),
                               1: type('C', (), {'ready': True})()}

            def poll(self, budget):
                return 0

            def call_ready(self, sequence):
                return self._calls[sequence].ready

            def take(self, sequence):
                self.taken.append(sequence)
                return sequence, {}
        executor = type('E', (), {'_flash_s5_transport': FakeTransport()})()
        future = eh.LocalFuture(executor, 1)
        self.assertTrue(future.done())
        self.assertEqual(future.result(timeout=0), 1)
        self.assertEqual(executor._flash_s5_transport.taken, [1])


class WorkerAdapterTests(unittest.TestCase):
    def test_open_submit_poll_commit_ack_sequences_are_independent(self):
        transaction = tx()
        backend = ah.FixtureCompletionBackend(delay=1, kind=CompletionKind.STATE_WRITE)
        handoff = ah.WorkerHandoff(transaction, backend=backend)
        self.assertEqual(handoff.apply(ah.OpenPrefill(transaction, 1024)).accepted, True)
        item = ticket(transaction, seq=1, role=DependencyRole.COMMIT)
        ack = handoff.apply(ah.SubmitOperation(transaction, 1, DependencyRole.COMMIT))
        self.assertTrue(ack.accepted)
        backend.submit(item)
        reports = handoff.apply(ah.PollOperations(transaction, 4))
        self.assertEqual(len(reports.proofs), 1)
        self.assertEqual(reports.proofs[0].kind, CompletionKind.STATE_WRITE)
        self.assertEqual(
            handoff.apply(ah.CommitTransaction(transaction,
                                               reports.proofs[0].completion_seq)).published,
            True)
        with self.assertRaises(ah.HandoffError):
            handoff.apply(ah.SubmitOperation(transaction, 2, DependencyRole.PRODUCER))

    def test_foreign_or_unbounded_messages_quarantine(self):
        transaction = tx()
        handoff = ah.WorkerHandoff(transaction, backend=ah.FixtureCompletionBackend())
        with self.assertRaises(ah.HandoffError):
            handoff.apply(ah.OpenPrefill(tx(seq=2), 1024))
        self.assertEqual(handoff.quarantine, 'stale_identity')
        handoff = ah.WorkerHandoff(transaction, backend=ah.FixtureCompletionBackend())
        handoff.apply(ah.OpenPrefill(transaction, 1024))
        with self.assertRaises(ah.HandoffError):
            handoff.apply(ah.PollOperations(transaction, 0))


class DispatcherTests(unittest.TestCase):
    def setUp(self):
        self.queues = {rank: FakeQueue() for rank in range(4)}
        self.config = ah.HandoffConfig(True, ah.QUALIFIED_TRANSPORT, 4, 5.0, 2.0, 1024)
        self.dispatcher = ah.HandoffDispatcher(self.config, allow_fixture=True)
        self.transport = reply_transport.ReplyTransport(
            self.queues, SUCCESS.OK, handoff_idle=True, max_calls=8)
        self.dispatcher.install_transport(self.transport)
        self.backend = ah.FixtureCompletionBackend(delay=3)
        self.dispatcher.install_backend(self.backend)
        self.transaction = tx()

    def tearDown(self):
        self.transport.close(1.0)

    def pump(self, predicate, deadline=2.0):
        end = time.monotonic() + deadline
        while time.monotonic() < end:
            report = self.dispatcher.poll(8)
            if predicate(report):
                return report
            time.sleep(0.005)
        self.fail('reply/completion did not arrive')

    def operation(self, rank=0, seq=1, role=DependencyRole.PRODUCER):
        tickets = tuple(ticket(self.transaction, rank_, seq, role)
                        for rank_ in range(4))
        for item in tickets:
            self.backend.submit(item)

        def send(_sequence):
            for rank_ in range(4):
                self.queues[rank_].put((SUCCESS.OK, ah.SubmitAck(self.transaction, seq, True)))
        self.dispatcher.submit(tickets, send, kind='operation')
        return tickets

    def test_four_rank_ack_ordering_and_no_publication_from_acks(self):
        self.dispatcher.open(self.transaction, lanes())
        tickets = self.operation()
        self.pump(lambda report: self.dispatcher.active.acked == 4)
        self.assertEqual(self.dispatcher.active.pending, 4)
        with self.assertRaises(ah.HandoffError):
            self.dispatcher.commit(proof(tickets[0], CompletionKind.STATE_WRITE))
        self.pump(lambda report: self.dispatcher.active.pending == 0)
        self.assertEqual(self.dispatcher.active.pending, 0)
        self.assertIsNot(self.dispatcher.active.phase, ah.Phase.COMMITTED)

    def test_one_active_transaction_and_sparse_or_missing_dense_refused(self):
        self.dispatcher.open(self.transaction, lanes())
        with self.assertRaises(ah.E3Pending):
            self.dispatcher.open(tx(seq=2), lanes())
        self.dispatcher.stop('cancel')
        self.assertEqual(self.dispatcher.drain(), ah.Phase.RETIRED)
        with self.assertRaises(ah.E3Pending):
            self.dispatcher.open(tx(seq=3), lanes(), sparse_active=True)
        with self.assertRaises(ah.E3Pending):
            self.dispatcher.open(tx(seq=4), lanes(), dense_available=False)

    def test_backpressure_does_not_grow_a_command_backlog(self):
        queues = {rank: FakeQueue() for rank in range(4)}
        tiny = reply_transport.ReplyTransport(queues, SUCCESS.OK,
                                              handoff_idle=True, max_calls=1)
        dispatcher = ah.HandoffDispatcher(self.config, allow_fixture=True)
        dispatcher.install_transport(tiny)
        dispatcher.install_backend(self.backend)
        dispatcher.open(self.transaction, lanes())
        ticket_ = tuple(ticket(self.transaction, rank, 1) for rank in range(4))
        for item in ticket_:
            self.backend.submit(item)
        dispatcher.submit(ticket_, lambda _s: None, kind='operation')
        second = tuple(ticket(self.transaction, rank, 2) for rank in range(4))
        with self.assertRaises(reply_transport.Backpressure):
            dispatcher.submit(second, lambda _s: None, kind='operation')
        self.assertEqual(dispatcher.active.pending, 4)
        self.assertEqual(len(dispatcher.active.submitted), 4)
        tiny.close(1.0)

    def test_partial_send_quarantines_and_stop_is_idempotent(self):
        self.dispatcher.open(self.transaction, lanes())
        ticket_ = tuple(ticket(self.transaction, rank, 1) for rank in range(4))
        ticket2 = tuple(ticket(self.transaction, rank, 2) for rank in range(4))
        for item in (*ticket_, *ticket2):
            self.backend.submit(item)

        def broken(_sequence):
            self.queues[0].put((SUCCESS.OK, ah.SubmitAck(self.transaction, 1, True)))
            raise RuntimeError('partial broadcast')
        with self.assertRaises(RuntimeError):
            self.dispatcher.submit(ticket_, broken, kind='operation')
        self.assertEqual(self.dispatcher.stop('shutdown'), ah.Phase.QUARANTINED)
        self.assertEqual(self.dispatcher.stop('shutdown'), ah.Phase.QUARANTINED)
        self.assertEqual(self.dispatcher.active.quarantine, 'transport_failure')


class ExecutedTransformTests(unittest.TestCase):
    def _function(self, source, name):
        tree = ast.parse(source)
        node = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == name)
        namespace = {'_flash_s5_hooks': sh}
        exec('from __future__ import annotations\n' + ast.unparse(node), namespace)
        return namespace[name]

    def test_generated_free_blocks_defers_then_frees_once(self):
        from pathlib import Path
        import mirror_hooks
        import telemetry_hooks
        source = sh.generate_block_pool(mirror_hooks.block_pool(
            telemetry_hooks.block_pool(
                (ROOT / 'reference/vllm-src/v1/core/block_pool.py').read_text())))

        class Queue:
            def __init__(self):
                self.blocks = []

            def prepend_n(self, blocks):
                self.blocks[:0] = list(blocks)

            def append_n(self, blocks):
                self.blocks.extend(blocks)

            def remove(self, block):
                self.blocks.remove(block)

        class Pool:
            free_blocks = self._function(source, 'free_blocks')

            def __init__(self):
                self._flash_s5_leases = None
                self.enable_caching = False
                self._evict_first = set()
                self.free_block_queue = Queue()
                self.metrics_collector = None
                self._flash_e2_journal = None

        class Block:
            def __init__(self, block_id):
                self.block_id = block_id
                self.ref_cnt = 1
                self.is_null = False
                self.block_hash = None
                self.pool = None

        pool = Pool()
        guard = sh.SchedulerLeases().attach(type('S', (), {'kv_cache_manager': type('M', (), {'block_pool': pool})()})(), pool)
        t = ticket(tx())
        blocks = [Block(1), Block(2)]
        guard.lease_blocks(t, ah.LeaseCategory.ARENA, [1, 2])
        pool.free_blocks(blocks)
        self.assertEqual(pool.free_block_queue.blocks, [])
        self.assertEqual([b.ref_cnt for b in blocks], [1, 1])
        guard.release(t, ah.LeaseCategory.ARENA,
                      proof(t, CompletionKind.GPU_EVENT))
        self.assertEqual(sorted(b.block_id for b in pool.free_block_queue.blocks), [1, 2])
        self.assertEqual([b.ref_cnt for b in blocks], [0, 0])
        self.assertEqual(guard.lease_caused_cow, 0)

    def test_generated_writability_unchanged_and_touch_refuses_uncommitted_lease(self):
        from pathlib import Path
        import mirror_hooks
        import telemetry_hooks
        source = sh.generate_block_pool(mirror_hooks.block_pool(
            telemetry_hooks.block_pool(
                (ROOT / 'reference/vllm-src/v1/core/block_pool.py').read_text())))
        writable = self._function(source, 'is_block_writable')
        touch = self._function(source, 'touch')

        class Queue:
            def __init__(self):
                self.blocks = []

            def remove(self, block):
                self.blocks.remove(block)

        class Pool:
            def __init__(self):
                self._flash_s5_leases = None
                self._evict_first = set()
                self.free_block_queue = Queue()
                self.metrics_collector = None

        Pool.is_block_writable = writable
        Pool.touch = touch

        class Block:
            def __init__(self, block_id):
                self.block_id = block_id
                self.ref_cnt = 1
                self.block_hash = None
                self.is_null = False
                self.pool = None

        pool = Pool()
        guard = sh.SchedulerLeases().attach(type('S', (), {'kv_cache_manager': type('M', (), {'block_pool': pool})()})(), pool)
        t = ticket(tx())
        block = Block(4)
        guard.lease_blocks(t, ah.LeaseCategory.ARENA, [4])
        self.assertTrue(pool.is_block_writable(block))
        with self.assertRaises(ah.E3Pending):
            pool.touch([block])
        self.assertEqual(block.ref_cnt, 1)
        guard.publish(t)
        pool.touch([block])
        self.assertEqual(block.ref_cnt, 2)
        self.assertFalse(pool.is_block_writable(block))



class ReviewRegressions(unittest.TestCase):
    def test_completion_before_ack_then_cancel_preserves_retiring(self):
        transaction = tx()
        lease = ah.TransactionLease(transaction, lanes(1), max_outstanding=2)
        t = ticket(transaction)
        lease.submit(t)
        lease.start_stop('cancel')
        lease.complete(proof(t))
        self.assertIs(lease.phase, ah.Phase.RETIRING)
        with self.assertRaises(ah.E3Pending):
            lease.drain()
        lease.acknowledge(transaction, 1, worker_rank=0)
        self.assertIs(lease.drain(), ah.Phase.RETIRED)
        with self.assertRaises(ah.HandoffError):
            lease.complete(proof(t, seq=2))

    def test_single_rank_commit_cannot_publish_four_rank_transaction(self):
        lease = ah.TransactionLease(tx(), lanes(), max_outstanding=4)
        t = ticket(tx(), role=DependencyRole.COMMIT)
        lease.submit(t)
        lease.complete(proof(t, CompletionKind.STATE_WRITE))
        lease.acknowledge(tx(), 1, worker_rank=0)
        with self.assertRaises(ah.E3Pending):
            lease.commit(proof(t, CompletionKind.STATE_WRITE))

    def test_quarantined_completion_does_not_reactivate_transaction(self):
        lease = ah.TransactionLease(tx(), lanes(1), max_outstanding=1)
        t = ticket(tx())
        lease.submit(t)
        lease._quarantine('transport_failure')
        with self.assertRaises(ah.HandoffError):
            lease.complete(proof(t))
        self.assertEqual(lease.pending, 1)
        self.assertIs(lease.phase, ah.Phase.QUARANTINED)

    def test_control_reply_must_validate_every_rank(self):
        d = ah.HandoffDispatcher(ah.HandoffConfig.from_env({ah.OVERLAP_ENV: '1'}), allow_fixture=True)
        d.open(tx(), lanes())
        acks = [ah.CommitAck(tx(), 1, True)] * 3 + [ah.CommitAck(tx(), 1, False)]
        with self.assertRaises(ah.HandoffError):
            d._apply_reply('commit', tx(), (), acks)
        with self.assertRaises(ah.HandoffError):
            d._apply_reply('cancel', tx(), (), [ah.CancelAck(tx(), 'retiring')])

    def test_none_future_result_and_aggregation_are_cached(self):
        class Transport:
            _calls = {0: type('Call', (), {'ready': True})()}
            count = 0
            def poll(self, budget): pass
            def call_ready(self, sequence): return self._calls[sequence].ready
            def take(self, sequence):
                self.count += 1
                del self._calls[sequence]
                return [None], {}
        transport = Transport()
        executor = type('Executor', (), {'_flash_s5_transport': transport})()
        aggregates = []
        def aggregate(value):
            aggregates.append(value)
            return value[0]
        future = eh.LocalFuture(executor, 0, aggregate)
        self.assertIsNone(future.result())
        self.assertTrue(future.done())
        self.assertIsNone(future.result())
        self.assertEqual(transport.count, 1)
        self.assertEqual(len(aggregates), 1)

    def test_generated_shutdown_guards_both_response_queue_aliases(self):
        ref = ROOT / 'reference/vllm-src' / eh.EXECUTOR_PATH
        generated = eh.generate_executor(ref.read_text())
        tree = ast.parse(generated)
        shutdown = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_flash_s5_stock_shutdown')
        # Execute the actual transformed shutdown with an unclean receiver.
        events = []
        class MQ:
            def shutdown(self): events.append('destroy')
        class Hooks:
            def before_shutdown(self, executor): events.append('close'); return False, {}
            def retain_mq(self, executor, worker): return True
        class Log:
            def debug(self, *args): pass
            def debug_once(self, *args): pass
        shutdown.decorator_list = []
        module = ast.Module(body=[shutdown], type_ignores=[])
        env = {'_flash_s5_hooks': Hooks(), 'logger': Log()}
        # Name annotations in pinned source may be postponed in its module.
        exec('from __future__ import annotations\n' + ast.unparse(module), env)
        executor = type('Executor', (), {})()
        executor.shutting_down = False
        executor.workers = []
        executor.rpc_broadcast_mq = None
        executor.response_mqs = [MQ()]
        env['_flash_s5_stock_shutdown'](executor)
        self.assertEqual(events, ['close'])
        self.assertEqual(len(executor.response_mqs), 1)

    def test_deferred_request_retried_after_last_consumer(self):
        pool = FakePool()
        request = type('Request', (), {'request_id': 'r'})()
        scheduler = FakeScheduler(pool)
        scheduler.kv_cache_manager.get_block_ids = lambda rid: ((4,),)
        freed = []
        scheduler._free_request_blocks = freed.append
        guard = sh.SchedulerLeases().attach(scheduler, pool)
        t = ticket(tx())
        guard.lease_blocks(t, ah.LeaseCategory.SOURCE_HISTORY, [4])
        self.assertTrue(guard.defer_request_free(scheduler, request))
        self.assertEqual(freed, [])
        guard.release(t, ah.LeaseCategory.SOURCE_HISTORY, proof(t, CompletionKind.COPY_EVENT))
        self.assertEqual(freed, [request])
        self.assertFalse(guard.deferred_requests)
        self.assertTrue(guard.shutdown()['clean'])

    def test_after_schedule_rejection_is_not_silently_ignored(self):
        pool = FakePool()
        scheduler = FakeScheduler(pool)
        guard = sh.SchedulerLeases().attach(scheduler, pool)
        guard.lease_blocks(ticket(tx()), ah.LeaseCategory.ARENA, [4])
        out = type('Output', (), {'new_block_ids_to_zero': [4]})()
        with self.assertRaises(ah.E3Pending):
            sh.after_schedule(scheduler, out)

    def test_direct_worker_method_rejects_foreign_message_before_runtime(self):
        runtime = type('Runtime', (), {'abort': lambda *args: self.fail('runtime called')})()
        adapter = ah.RuntimeWorkerHandoff(runtime, tx(), metadata=object())
        with self.assertRaises(ah.HandoffError):
            adapter.cancel(ah.CancelTransaction(tx(seq=2), 'cancel'))


class AdditionalReviewRegressions(unittest.TestCase):
    def test_lower_operation_sequence_rejected_even_if_not_duplicate(self):
        lease = ah.TransactionLease(tx(), lanes(1), max_outstanding=3)
        lease.submit(ticket(tx(), seq=9))
        with self.assertRaises(ah.HandoffError):
            lease.submit(ticket(tx(), seq=8))

    def test_stop_blocks_new_admission_without_an_active_transaction(self):
        dispatcher = ah.HandoffDispatcher(ah.HandoffConfig.from_env({ah.OVERLAP_ENV: '1'}), allow_fixture=True)
        dispatcher.stop('shutdown')
        with self.assertRaises(ah.E3Pending):
            dispatcher.open(tx(), lanes())


class RuntimeCommitRecordTests(unittest.TestCase):
    def test_missing_native_commit_record_cannot_fabricate_publication(self):
        class Runtime:
            def commit(self, transaction, proof): return None
        adapter = ah.RuntimeWorkerHandoff(Runtime(), tx(), metadata=object())
        t = ticket(tx(), role=DependencyRole.COMMIT)
        adapter.completions = [proof(t, CompletionKind.STATE_WRITE)]
        with self.assertRaises(ah.HandoffError):
            adapter.commit(ah.CommitTransaction(tx(), 1))
        self.assertIsNot(adapter.phase, ah.Phase.COMMITTED)


if __name__ == '__main__':
    unittest.main()
