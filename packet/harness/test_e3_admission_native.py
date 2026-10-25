"""E3 pre-schedule admission, bounded wire driver, strict sender and startup shape.

Offline CPU fixtures over the pinned E2 scheduler sources and deterministic
fakes. This is not native execution evidence: no GPU, no vLLM process, no
shared-memory ring, no PP traffic.
"""
from pathlib import Path
import hashlib
import types
from dataclasses import replace
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import async_handoff as ah
import engine_driver as ed
import executor_hooks as eh
import mirror_hooks
import scheduler_hooks as sh
import s5_startup
import telemetry_hooks
import wire_v1 as w
from e3_types import (
    CompletionKind,
    CompletionProof,
    DependencyRole,
    E3ContractError,
    E3Pending,
    LaneId,
    LaneRole,
    OperationTicket,
    TransactionId,
)
from reply_transport import Backpressure

UID = 'req-e3-admission'


# --- source transform -------------------------------------------------------

class TransformTests(unittest.TestCase):
    def test_admission_hooks_are_inserted_and_reverted_exactly(self):
        ref = ROOT / 'reference/vllm-src'
        source = (ref / sh.SCHEDULER_PATH).read_text()
        e2 = mirror_hooks.scheduler(telemetry_hooks.scheduler(source))
        self.assertEqual(hashlib.sha256(e2.encode()).hexdigest(), sh.SCHEDULER_E2_SHA256)
        out = sh.generate_scheduler(e2)
        self.assertIn('_flash_s5_hooks.flash_s5_admit(self)', out)
        self.assertEqual(out.count('_flash_s5_hooks.parked_count(self)'), 2)
        self.assertEqual(sh.revert_scheduler(out), e2)
        self.assertGreaterEqual(out.count('_flash_s5_hooks.'), 5)
        # The block-pool transform is untouched by admission.
        pool = mirror_hooks.block_pool(telemetry_hooks.block_pool(
            (ref / sh.BLOCK_POOL_PATH).read_text()))
        self.assertEqual(sh.revert_block_pool(sh.generate_block_pool(pool)), pool)


# --- fakes ------------------------------------------------------------------

class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.is_null = False


class Request:
    def __init__(self, uid=UID, tokens=10, computed=0, generation=3, epoch=1):
        self.request_id = uid
        self.all_token_ids = list(range(tokens))
        self.num_tokens = tokens
        self.num_computed_tokens = computed
        self.request_generation = generation
        self.status = 'waiting'
        self.state_epoch = epoch
        self.new_block_ids_to_zero = ()


class Queue:
    def __init__(self, requests=()):
        self.items = list(requests)

    def peek_request(self):
        return self.items[0]

    def remove_request(self, request):
        self.items.remove(request)

    def prepend_request(self, request):
        self.items.insert(0, request)

    def __len__(self):
        return len(self.items)

    def __bool__(self):
        return bool(self.items)


class Manager:
    def __init__(self, cached=0, fail=False):
        self.cached = cached
        self.fail = fail
        self.rows = {}
        self.alloc_calls = []
        self.cache_calls = []
        self.lookup_calls = 0
        self.next_block = 1

    def get_computed_blocks(self, request):
        self.lookup_calls += 1
        return None, self.cached, 0

    def allocate_slots(self, request, num_new, **kwargs):
        self.alloc_calls.append((request, num_new, kwargs))
        if self.fail:
            return None
        row = self.rows.get(request.request_id, ())
        row += tuple(range(self.next_block, self.next_block + num_new))
        self.next_block += num_new
        self.rows[request.request_id] = row
        return object()

    def get_block_ids(self, request_id):
        return (self.rows[request_id],)

    def cache_blocks(self, request, num_computed):
        self.cache_calls.append((request.request_id, num_computed))


class Scheduler:
    def __init__(self, manager, requests=(), max_num=8):
        self.kv_cache_manager = manager
        self.waiting = Queue(requests)
        self.running = []
        self.max_num_running_reqs = max_num
        self.max_num_scheduled_tokens = 1024
        self._flash_s5_admission_metadata = lambda request, allocated, start, end: dict(
            new_block_ids_to_zero=(), cow_pairs=(), ngram_context=(),
            unpublished_blocks=manager.rows[request.request_id][- (end - start):])
        self.scheduler_reserve_full_isl = False
        self._pause_state = None
        self.connector = None
        self.vllm_config = None

    def _select_waiting_queue_for_scheduling(self):
        return self.waiting or None

    def _is_blocked_waiting_status(self, status):
        return False


def make_controller(scheduler, *, chunk=4, ranks=(0, 1), max_history=64,
                    leases=None, frontier=None):
    controller = sh.PrefillAdmissionController(
        scheduler, leases=leases or sh.SchedulerLeases(),
        frontier=frontier or ah.CommittedFrontier(), engine_epoch=7,
        chunk_tokens=chunk, ranks=ranks, max_history=max_history)
    return controller.attach()


def make_ticket(transaction, rank, op_seq, role):
    return OperationTicket(transaction, rank,
                           LaneId(rank, LaneRole.PREFILL, 5, 0), op_seq, role)


def make_proof(transaction, rank, op_seq, role, kind, completion_seq=1):
    return CompletionProof(make_ticket(transaction, rank, op_seq, role), kind,
                           completion_seq)


def make_report(transaction, rank, completed, proofs, *, pending=(),
                epoch=1):
    completed = tuple(sorted(completed))
    return w.WorkerCompletionReport(
        transaction, rank, completed, tuple(sorted(pending)),
        max([p.completion_seq for p in proofs] + [0]), tuple(proofs),
        w.CompletionCursor('commit', 0, 0), epoch)


def reports_for(transaction, *, bound_op=0, epoch=1):
    rank0 = make_report(transaction, 0, (0, 1), (
        make_proof(transaction, 0, bound_op, DependencyRole.PRODUCER,
                   CompletionKind.GPU_EVENT),
        make_proof(transaction, 0, 1, DependencyRole.COMMIT,
                   CompletionKind.STATE_WRITE)))
    rank1 = make_report(transaction, 1, (0,),
                        (make_proof(transaction, 1, 0, DependencyRole.COMMIT,
                                    CompletionKind.STATE_WRITE),))
    return replace(rank0, state_epoch=epoch), replace(rank1, state_epoch=epoch)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.request = Request()
        self.manager = Manager()
        self.scheduler = Scheduler(self.manager, [self.request])
        self.controller = make_controller(self.scheduler)

    def test_new_request_is_admitted_before_the_waiting_loop(self):
        record = self.scheduler._flash_s5_admission.admit(self.scheduler)
        self.assertEqual(record.chunk_tokens, 4)
        self.assertEqual(record.committed_end, 0)
        self.assertEqual(record.remaining_tokens, 6)
        self.assertEqual(record.transaction.chunk_start, 0)
        self.assertEqual(record.transaction.chunk_end, 4)
        self.assertEqual(record.flat_blocks(), (1, 2, 3, 4))
        request, num_new, kwargs = self.manager.alloc_calls[0]
        self.assertIs(request, self.request)
        self.assertEqual(num_new, 4)
        self.assertTrue(kwargs['delay_cache_blocks'])
        self.assertEqual(kwargs['num_lookahead_tokens'], 0)
        self.assertEqual(self.manager.cache_calls, [])
        self.assertFalse(self.scheduler.waiting)
        self.assertEqual(self.controller.parked_count(), 1)
        self.assertEqual(self.controller.leases.held_blocks(), (1, 2, 3, 4))
        self.assertEqual(sh.parked_count(self.scheduler), 1)
        self.assertEqual(sh.flash_s5_admit(self.scheduler), None)   # one at a time

    def test_warm_prefix_starts_after_the_cached_blocks(self):
        self.manager.cached = 4
        record = self.controller.admit(self.scheduler)
        self.assertEqual(record.transaction.chunk_start, 4)
        self.assertEqual(record.transaction.chunk_end, 8)
        self.assertEqual(self.manager.lookup_calls, 1)
        _request, num_new, kwargs = self.manager.alloc_calls[0]
        self.assertEqual(num_new, 4)
        self.assertEqual(kwargs['num_new_computed_tokens'], 4)
        self.assertIsNone(kwargs['new_computed_blocks'])

    def test_all_but_last_token_cached_is_left_to_stock(self):
        self.manager.cached = 9
        self.assertIsNone(self.controller.admit(self.scheduler))
        self.assertEqual(self.manager.alloc_calls, [])
        self.assertTrue(self.scheduler.waiting)

    def test_chunk_never_contains_the_last_prompt_token(self):
        request = Request(tokens=5)
        scheduler = Scheduler(Manager(), [request])
        controller = make_controller(scheduler, chunk=8)
        record = controller.admit(scheduler)
        self.assertEqual(record.committed_end, 0)
        self.assertEqual(record.remaining_tokens, 1)

    def test_allocation_failure_keeps_the_request_waiting(self):
        self.manager.fail = True
        self.assertIsNone(self.controller.admit(self.scheduler))
        self.assertEqual(self.controller.alloc_failures, 1)
        self.assertEqual(self.controller.leases.held_blocks(), ())
        self.assertTrue(self.scheduler.waiting)
        self.assertFalse(self.manager.cache_calls)

    def test_refuses_blocked_and_connector_requests(self):
        self.scheduler.connector = object()
        self.assertIsNone(self.controller.admit(self.scheduler))
        self.assertEqual(self.manager.alloc_calls, [])

    def test_uncommitted_blocks_cannot_be_shared(self):
        self.controller.admit(self.scheduler)
        with self.assertRaises(E3Pending):
            self.controller.leases.before_touch([Block(2)])
        with self.assertRaises(E3Pending):
            self.controller.frontier.shareable((1, 2))
        self.assertEqual(self.controller.frontier.lookup((1, 2, 3, 4)), ())

    def test_commit_publishes_once_and_returns_one_token(self):
        record = self.controller.admit(self.scheduler)
        transaction = record.transaction
        ticket = make_ticket(transaction, 0, 0, DependencyRole.PRODUCER)
        self.controller.bind(ticket)
        rank0, rank1 = reports_for(transaction)
        self.controller.note_report(rank0)
        self.assertFalse(self.controller.ready_to_commit(transaction))
        self.controller.note_report(rank1)
        self.assertFalse(self.controller.ready_to_commit(transaction))
        for report in (rank0, rank1):
            self.controller.note_worker_release(w.ReleaseAck(transaction, report.worker_rank,
                record.state_epoch, report.completed_sequence))
        self.assertTrue(self.controller.ready_to_commit(transaction))
        proof = make_proof(transaction, 0, 0, DependencyRole.PRODUCER,
                           CompletionKind.GPU_EVENT)
        request = self.controller.commit(transaction, proof)
        self.assertIs(request, self.request)
        self.assertEqual(self.manager.cache_calls, [(UID, 4)])
        self.assertEqual(request.num_computed_tokens, 4)
        self.assertEqual(self.scheduler.waiting.items, [request])
        self.assertIsNone(self.controller.parked)
        self.assertEqual(self.controller.leases.held_blocks(), ())
        self.assertEqual(self.controller.frontier.shareable((1, 2)), (1, 2))
        self.assertEqual(self.controller.leases.before_touch([Block(2)]) is None, True)
        with self.assertRaises(E3ContractError):
            self.controller.commit(transaction, proof)

    def test_two_full_transactions_advance_the_same_request(self):
        first = self.controller.admit(self.scheduler)
        ticket = make_ticket(first.transaction, 0, 0, DependencyRole.PRODUCER)
        self.controller.bind(ticket)
        for report in reports_for(first.transaction):
            self.controller.note_report(report)
            self.controller.note_worker_release(w.ReleaseAck(first.transaction, report.worker_rank,
                first.state_epoch, report.completed_sequence))
        self.controller.commit(first.transaction, make_proof(
            first.transaction, 0, 0, DependencyRole.PRODUCER,
            CompletionKind.GPU_EVENT))
        second = self.controller.admit(self.scheduler)
        self.assertEqual(second.transaction.chunk_start, 4)
        self.assertEqual(second.transaction.chunk_end, 8)
        self.assertEqual(second.transaction.txn_seq, first.transaction.txn_seq + 1)

    def test_independent_rank_delays_gate_publication(self):
        record = self.controller.admit(self.scheduler)
        transaction = record.transaction
        self.controller.bind(make_ticket(transaction, 0, 0,
                                         DependencyRole.PRODUCER))
        rank0, rank1 = reports_for(transaction)
        self.controller.note_report(rank0)
        self.assertFalse(self.controller.ready_to_commit(transaction))
        self.controller.note_report(rank1)
        self.assertFalse(self.controller.ready_to_commit(transaction))
        for report in (rank0, rank1):
            self.controller.note_worker_release(w.ReleaseAck(transaction, report.worker_rank,
                record.state_epoch, report.completed_sequence))
        self.assertTrue(self.controller.ready_to_commit(transaction))

    def test_state_epoch_is_validated_per_chunk(self):
        record = self.controller.admit(self.scheduler)
        bad = make_report(record.transaction, 0, (), (), epoch=9)
        with self.assertRaises(E3ContractError):
            self.controller.note_report(bad)
        self.assertEqual(self.controller.state_epoch_rejections, 1)

    def test_pp_transfer_needs_its_consumer_proof(self):
        record = self.controller.admit(self.scheduler)
        transaction = record.transaction
        self.controller.bind(make_ticket(transaction, 0, 0,
                                         DependencyRole.PRODUCER))
        for report in reports_for(transaction):
            self.controller.note_report(report)
            self.controller.note_worker_release(w.ReleaseAck(transaction, report.worker_rank,
                record.state_epoch, report.completed_sequence))
        self.controller.note_pp_send(transaction, 0)
        self.assertFalse(self.controller.ready_to_commit(transaction))
        with self.assertRaises(E3ContractError):
            self.controller.note_pp_consumer(transaction, 0, 0)
        self.controller.note_pp_consumer(transaction, 0, 1)
        self.assertTrue(self.controller.ready_to_commit(transaction))
        with self.assertRaises(E3ContractError):
            self.controller.note_pp_consumer(transaction, 3, 1)

    def test_never_submitted_cancellation_frees_the_reservation(self):
        record = self.controller.admit(self.scheduler)
        returned = self.controller.cancel(record.transaction, never_submitted=True)
        self.assertIs(returned, self.request)
        self.assertEqual(self.controller.leases.held_blocks(), ())
        self.assertIsNone(self.controller.parked)
        # The request is handed back unqueued: the engine's abort path owns it.
        self.assertEqual(self.scheduler.waiting.items, [])

    def test_bound_cancellation_requires_its_completion_proof(self):
        record = self.controller.admit(self.scheduler)
        transaction = record.transaction
        self.controller.bind(make_ticket(transaction, 0, 0,
                                         DependencyRole.PRODUCER))
        with self.assertRaises(E3ContractError):
            self.controller.cancel(transaction, never_submitted=True)
        with self.assertRaises(E3ContractError):
            self.controller.cancel(transaction)
        proof = make_proof(transaction, 0, 0, DependencyRole.PRODUCER,
                           CompletionKind.GPU_EVENT)
        with self.assertRaises(E3Pending):
            self.controller.cancel(transaction, proof=proof)
        self.assertTrue(self.controller.leases.held_blocks())
        for report in reports_for(transaction):
            self.controller.note_report(report)
            self.controller.note_worker_release(w.ReleaseAck(transaction, report.worker_rank,
                record.state_epoch, report.completed_sequence))
        self.controller.cancel(transaction, proof=proof)
        self.assertEqual(self.controller.leases.held_blocks(), ())

    def test_cancellation_cannot_abandon_a_pp_transfer(self):
        record = self.controller.admit(self.scheduler)
        transaction = record.transaction
        self.controller.bind(make_ticket(transaction, 0, 0,
                                         DependencyRole.PRODUCER))
        self.controller.note_pp_send(transaction, 0)
        proof = make_proof(transaction, 0, 0, DependencyRole.PRODUCER,
                           CompletionKind.GPU_EVENT)
        with self.assertRaises(E3ContractError):
            self.controller.cancel(transaction, proof=proof)

    def test_history_is_bounded(self):
        controller = make_controller(Scheduler(Manager(), [Request()]),
                                     max_history=1)
        for _ in range(2):
            record = controller.admit(controller._scheduler)
            ticket = make_ticket(record.transaction, 0, 0, DependencyRole.PRODUCER)
            controller.bind(ticket)
            for report in reports_for(record.transaction, epoch=record.state_epoch):
                controller.note_report(report)
                controller.note_worker_release(w.ReleaseAck(record.transaction, report.worker_rank,
                    record.state_epoch, report.completed_sequence))
            controller.commit(record.transaction, make_proof(
                record.transaction, 0, 0, DependencyRole.PRODUCER,
                CompletionKind.GPU_EVENT))
        self.assertEqual(len(controller._committed), 1)
        self.assertEqual(controller.summary()['commits'], 2)

    def test_missing_allocator_metadata_refuses_before_allocating(self):
        del self.scheduler._flash_s5_admission_metadata
        with self.assertRaises(E3ContractError):
            self.controller.admit(self.scheduler)
        self.assertEqual(self.manager.alloc_calls, [])
        self.assertTrue(self.scheduler.waiting)

    def test_post_allocation_metadata_failure_quarantines_the_request(self):
        self.scheduler._flash_s5_admission_metadata = lambda *args: {}
        with self.assertRaises(E3ContractError):
            self.controller.admit(self.scheduler)
        self.assertTrue(self.controller.parked.failed)
        self.assertFalse(self.scheduler.waiting)
        self.assertIsNone(self.controller.admit(self.scheduler))

    def test_alignment_zero_and_errors_are_not_bypassed(self):
        self.scheduler._mamba_block_aligned_split = lambda *args, **kw: 0
        self.assertIsNone(self.controller.admit(self.scheduler))
        self.assertEqual(self.manager.alloc_calls, [])
        def broken(*args, **kw):
            raise TypeError('fixture')
        self.scheduler._mamba_block_aligned_split = broken
        with self.assertRaises(TypeError):
            self.controller.admit(self.scheduler)
        self.assertEqual(self.manager.alloc_calls, [])

    def test_scheduler_budget_and_streaming_slots_are_respected(self):
        self.scheduler.num_waiting_for_streaming_input = 8
        self.assertIsNone(self.controller.admit(self.scheduler))
        self.scheduler.num_waiting_for_streaming_input = 0
        self.scheduler.max_num_scheduled_tokens = 2
        self.assertEqual(self.controller.admit(self.scheduler).chunk_tokens, 2)

    def test_all_group_positions_survive_and_nulls_are_not_leased(self):
        self.manager.get_block_ids = lambda req: ((), (0, 0, 1, 2, 3, 4), ())
        record = self.controller.admit(self.scheduler)
        self.assertEqual(record.block_tables, ((), (0, 0, 1, 2, 3, 4), ()))
        self.assertEqual(self.controller.leases.held_blocks(), (1, 2, 3, 4))
        entry = self.controller.leases._held[1][0]
        self.assertIs(entry['category'], ah.LeaseCategory.SOURCE_HISTORY)

    def test_pending_work_with_state_proof_cannot_publish(self):
        record = self.controller.admit(self.scheduler)
        a, b = reports_for(record.transaction)
        self.controller.note_report(replace(a, pending_op_seqs=(2,)))
        self.controller.note_report(b)
        self.assertFalse(self.controller.ready_to_release(record.transaction))
        with self.assertRaises(E3ContractError):
            self.controller.note_worker_release(w.ReleaseAck(record.transaction, 0, 1,
                                                               a.completed_sequence))

    def test_completed_watermark_cannot_regress(self):
        record = self.controller.admit(self.scheduler)
        a, _ = reports_for(record.transaction)
        self.controller.note_report(a)
        with self.assertRaises(E3ContractError):
            self.controller.note_report(make_report(record.transaction, 0, (), ()))


# --- PP ledger --------------------------------------------------------------

class PPTransferLedgerTests(unittest.TestCase):
    def setUp(self):
        self.ledger = ed.PPTransferLedger(2, stage1_ranks=(1,))
        self.transaction = TransactionId(0, UID, 0, 0, 4, 1)

    def sender_proof(self, seq=0, rank=0):
        return make_proof(self.transaction, rank, seq, DependencyRole.HANDOFF,
                          CompletionKind.GPU_EVENT)

    def consumer_proof(self, seq=0, rank=1):
        return make_proof(self.transaction, rank, seq, DependencyRole.HANDOFF,
                          CompletionKind.PP_CONSUMER)

    def test_release_needs_both_proofs(self):
        self.ledger.note_sent(self.transaction, 0)
        with self.assertRaises(E3Pending):
            self.ledger.release_send(self.transaction, 0)
        self.ledger.note_sender_complete(self.transaction, 0, self.sender_proof())
        with self.assertRaises(E3Pending):
            self.ledger.release_send(self.transaction, 0)
        self.ledger.note_consumer(self.transaction, 0, 1, self.consumer_proof())
        self.assertTrue(self.ledger.releasable(self.transaction, 0))
        consumer = self.ledger.release_send(self.transaction, 0)
        self.assertEqual(consumer.kind, CompletionKind.PP_CONSUMER)
        with self.assertRaises(E3ContractError):
            self.ledger.release_send(self.transaction, 0)

    def test_identity_and_stage_are_validated(self):
        self.ledger.note_sent(self.transaction, 0)
        with self.assertRaises(ed.DriverError):
            self.ledger.note_consumer(self.transaction, 0, 0, self.consumer_proof())
        with self.assertRaises(ed.DriverError):
            self.ledger.note_consumer(self.transaction, 0, 1,
                                      self.consumer_proof(rank=0))
        other = TransactionId(0, UID, 0, 0, 4, 9)
        with self.assertRaises(ed.DriverError):
            self.ledger.note_consumer(other, 0, 1, self.consumer_proof())
        with self.assertRaises(ed.DriverError):
            self.ledger.note_sender_complete(
                self.transaction, 0,
                make_proof(self.transaction, 0, 0, DependencyRole.HANDOFF,
                           CompletionKind.COPY_EVENT))
        with self.assertRaises(ed.DriverError):
            self.ledger.note_consumer(self.transaction, 5, 1,
                                      self.consumer_proof(seq=5))

    def test_transfer_sequences_are_explicit(self):
        self.ledger.note_sent(self.transaction, 3)
        with self.assertRaises(ed.DriverError):
            self.ledger.note_sent(self.transaction, 3)
        self.assertEqual(self.ledger.outstanding(self.transaction), (3,))

    def test_sender_proof_must_come_from_the_sending_lane(self):
        self.ledger.note_sent(self.transaction, 0)
        with self.assertRaises(ed.DriverError):
            self.ledger.note_sender_complete(self.transaction, 0,
                                             self.sender_proof(rank=1))
        self.assertFalse(self.ledger.releasable(self.transaction, 0))


# --- bounded driver ---------------------------------------------------------

class FakeTransport:
    def __init__(self):
        self._calls = {}
        self.kinds = {}
        self.ready = {}
        self.next = 0

    def submit(self, kind, send, ranks=None, timeout_s=300.0, aggregate=None):
        sequence = self.next
        self.next += 1
        send(sequence)
        self._calls[sequence] = types.SimpleNamespace(ready=False)
        self.kinds[sequence] = kind
        return sequence

    def poll(self, budget=8):
        return 0

    def take(self, sequence):
        del self._calls[sequence]
        return self.ready.pop(sequence), {}

    def reply(self, values, sequence=None):
        sequence = min(self._calls) if sequence is None else sequence
        self._calls[sequence].ready = True
        self.ready[sequence] = values


class FakeSender:
    def __init__(self):
        self.envelopes = []
        self.checks = []

    def check(self, envelope):
        self.checks.append(envelope)

    def __call__(self, envelope):
        self.envelopes.append(envelope)


class FakeExecutor:
    def __init__(self):
        self._flash_s5_transport = FakeTransport()
        self._flash_s5_bounded_send = FakeSender()

    def last_command(self):
        method, (message,), _kwargs, _rank = self._flash_s5_bounded_send.envelopes[-1]
        return method, message


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.request = Request(tokens=12)
        self.manager = Manager()
        self.scheduler = Scheduler(self.manager, [self.request])
        self.controller = make_controller(self.scheduler, chunk=4)
        self.record = self.controller.admit(self.scheduler)
        self.executor = FakeExecutor()
        self.driver = ed.NativeSegmentDriver(
            self.controller, executor=self.executor, scheduler=self.scheduler,
            engine=types.SimpleNamespace(), layer_count=4, layer_split=2,
            stage_of=((0, 0), (1, 1)), reply_timeout_s=5.0).install()

    def acks(self):
        transaction = self.record.transaction
        return [w.OpenAck(transaction, LaneId(0, LaneRole.PREFILL, 5, 0), 0, 1, 2, True),
                w.OpenAck(transaction, LaneId(1, LaneRole.PREFILL, 5, 0), 1, 1, 2, True)]

    def acceptances(self, command):
        values = []
        for rank in self.driver.ranks:
            values.append(w.SegmentAcceptance(
                command.transaction, rank, command.kind, command.stage,
                command.op_seq, accepted=rank in command.ranks,
                skipped=rank not in command.ranks))
        return values

    def completion_reports(self):
        reports = []
        for rank in self.driver.ranks:
            tickets = sorted((t for (r, op), t in self.driver.accepted_tickets.items()
                              if r == rank), key=lambda t: t.op_seq)
            proofs = []
            for t in tickets:
                desc = self.driver._accepted_descriptors[(rank, t.op_seq)]
                kind = (CompletionKind.STATE_WRITE if desc.kind is w.SegmentKind.STATE_FINALIZE
                        else CompletionKind.PP_CONSUMER if desc.kind is w.SegmentKind.PP_RECEIVE
                        else CompletionKind.GPU_EVENT)
                proofs.append(CompletionProof(t, kind, len(proofs) + 1))
            reports.append(make_report(self.record.transaction, rank,
                tuple(t.op_seq for t in tickets), proofs))
        return reports

    def finish_release(self):
        method, request = self.executor.last_command()
        self.assertEqual(method, 'flash_s5_commit')
        self.assertEqual(self.manager.cache_calls, [])
        self.executor._flash_s5_transport.reply([
            w.ReleaseAck(request.transaction, rank, request.state_epoch, seq)
            for rank, seq in request.completed_sequences])
        self.driver.advance()


    def test_one_bounded_command_per_tick(self):
        self.driver.advance()
        method, message = self.executor.last_command()
        self.assertEqual(method, 'flash_s5_open')
        self.assertIsInstance(message, w.OpenRequest)
        self.assertEqual(len(self.executor._flash_s5_bounded_send.envelopes), 1)
        self.assertEqual(self.executor._flash_s5_bounded_send.checks,
                         self.executor._flash_s5_bounded_send.envelopes)

        self.executor._flash_s5_transport.reply(self.acks())
        self.driver.advance()
        method, command = self.executor.last_command()
        self.assertEqual(method, 'flash_s5_submit')
        self.assertIs(command.kind, w.SegmentKind.PREPARE)
        self.assertIs(command.kind, w.SegmentKind.PREPARE)
        self.assertEqual(len(self.executor._flash_s5_bounded_send.envelopes), 2)
        self.assertEqual(self.driver.bound_ticket, None)

    def test_full_plan_publishes_only_after_every_proof(self):
        self.driver.advance()
        self.executor._flash_s5_transport.reply(self.acks())
        seen = []
        for _ in range(40):
            self.driver.advance()
            if self.driver.phase != 'segments':
                break
            method, command = self.executor.last_command()
            self.assertEqual(method, 'flash_s5_submit')
            seen.append(command.kind)
            self.executor._flash_s5_transport.reply(self.acceptances(command))
        self.assertEqual(seen, [
            w.SegmentKind.PREPARE, w.SegmentKind.PREPARE, w.SegmentKind.PP_RECEIVE,
            w.SegmentKind.PREFIX,
            w.SegmentKind.LAYER, w.SegmentKind.LAYER, w.SegmentKind.SUFFIX,
            w.SegmentKind.PP_SEND, w.SegmentKind.STATE_FINALIZE, w.SegmentKind.PREFIX,
            w.SegmentKind.LAYER, w.SegmentKind.LAYER, w.SegmentKind.SUFFIX,
            w.SegmentKind.STATE_FINALIZE])
        self.assertEqual(self.driver.phase, 'finalize')
        # One poll per tick while completion reports are missing.
        self.driver.advance()
        method, message = self.executor.last_command()
        self.assertEqual(method, 'flash_s5_poll')
        self.assertIsInstance(message, ah.PollOperations)

        transaction = self.record.transaction
        rank0, rank1 = self.completion_reports()
        self.executor._flash_s5_transport.reply([rank0, rank1])
        self.driver.advance()
        self.finish_release()
        self.assertIsNone(self.driver.tx)
        self.assertEqual(self.driver.phase, 'idle')
        self.assertIsNone(self.controller.parked)
        self.assertEqual(self.manager.cache_calls, [(UID, 4)])
        self.assertEqual(self.scheduler.waiting.items, [self.request])
        self.assertEqual(self.driver.pp.summary(), dict(sent=1, released=1))
        self.assertTrue(self.driver.finish_pause())

    def test_piggybacked_early_state_proof_is_rejected(self):
        self.driver.advance()
        self.executor._flash_s5_transport.reply(self.acks())
        self.driver.advance()
        _method, command = self.executor.last_command()
        values = self.acceptances(command)
        forged = make_report(command.transaction, 0, (command.op_seq,), (
            make_proof(command.transaction, 0, command.op_seq, DependencyRole.COMMIT,
                       CompletionKind.STATE_WRITE),))
        values[0] = replace(values[0], report=forged)
        self.executor._flash_s5_transport.reply(values)
        with self.assertRaises(ed.DriverError):
            self.driver.advance()
        self.assertEqual(self.manager.cache_calls, [])
        self.assertTrue(self.controller.leases.held_blocks())

    def test_pending_retry_reuses_the_same_op_seq(self):
        self.driver.advance()
        self.executor._flash_s5_transport.reply(self.acks())
        self.driver.advance()
        _method, command = self.executor.last_command()
        first_seq = command.op_seq
        pending = [w.SegmentAcceptance(self.record.transaction, rank,
                                       command.kind, command.stage, command.op_seq,
                                       pending=rank in command.ranks,
                                       skipped=rank not in command.ranks)
                   for rank in self.driver.ranks]
        self.executor._flash_s5_transport.reply(pending)
        self.driver.advance()
        _method, retried = self.executor.last_command()
        self.assertEqual(retried.op_seq, first_seq)
        self.assertEqual(retried.descriptor, command.descriptor)
        self.assertEqual(self.driver.index, 0)
        self.assertEqual(self.driver.trackers[0].last_accepted, -1)

    def test_wrong_target_behavior_is_refused(self):
        self.driver.advance()
        self.executor._flash_s5_transport.reply(self.acks())
        self.driver.advance()
        _method, command = self.executor.last_command()
        values = self.acceptances(command)
        values[1] = w.SegmentAcceptance(self.record.transaction, 1, command.kind,
                                        command.stage, command.op_seq,
                                        accepted=True)
        self.executor._flash_s5_transport.reply(values)
        with self.assertRaises(ed.DriverError):
            self.driver.advance()

    def test_stop_without_proof_never_releases_the_lease(self):
        self.driver.advance()
        self.executor._flash_s5_transport.reply(self.acks())
        self.driver.advance()
        _method, command = self.executor.last_command()
        self.executor._flash_s5_transport.reply(self.acceptances(command))
        self.driver.advance()
        self.driver.stop('abort')
        # No completion report yet: the driver polls and keeps ownership.
        self.driver.advance()
        self.assertIsNotNone(self.driver.tx)
        self.assertEqual(self.controller.leases.held_blocks(), (1, 2, 3, 4))

    def test_stop_before_any_acceptance_cancels_the_reservation(self):
        self.driver.stop('abort')
        self.driver.advance()
        self.assertIsNone(self.driver.tx)
        self.assertEqual(self.controller.leases.held_blocks(), ())

    def test_stop_with_an_unconsumed_pp_transfer_quarantines(self):
        self.driver.advance()
        self.executor._flash_s5_transport.reply(self.acks())
        for _ in range(20):
            self.driver.advance()
            _method, command = self.executor.last_command()
            self.executor._flash_s5_transport.reply(self.acceptances(command))
            if command.kind is w.SegmentKind.PP_SEND:
                break
        self.driver.stop('pause')
        with self.assertRaises(ed.DriverError):
            self.driver.advance()
        self.assertEqual(self.controller.leases.held_blocks(), (1, 2, 3, 4))
        self.assertEqual(self.driver.pp.summary(), dict(sent=1, released=0))

    def test_engine_abort_hook_stops_the_parked_chunk(self):
        engine = self.driver.engine
        self.assertIsNone(eh.before_abort(engine, ('other-request',)))
        self.assertEqual(eh.before_abort(engine, (UID,)), 'idle')
        self.assertEqual(self.driver.stop_kind, 'abort')
        self.driver.advance()
        self.assertIsNone(self.controller.parked)
        self.assertEqual(self.controller.leases.held_blocks(), ())

    def test_reset_hook_waits_for_parked_retirement(self):
        engine = self.driver.engine
        with self.assertRaises(E3Pending):
            eh.before_reset_prefix(engine)
        self.driver.advance()
        self.assertIsNone(eh.before_reset_prefix(engine))

    def test_pause_hooks_drain_then_report_clean(self):
        engine = self.driver.engine
        self.assertEqual(eh.before_pause(engine, 'abort'), 'idle')
        self.driver.advance()
        self.assertTrue(eh.before_finish_pause(engine))

    def test_transition_must_echo_the_exact_generation(self):
        command = self.driver.issue_transition(
            expected_route_generation=0, target_bank=1,
            storage_epochs=(5, 6), decode_replay_watermarks=(0, 0))
        self.assertIsInstance(command, w.TransitionCommand)
        method, message = self.executor.last_command()
        self.assertEqual(method, 'flash_s5_transition')
        self.assertIs(message, command)
        stale = w.TransitionAck(0, 0, 1, 1, (5, 6), 1, 1, (5, 6))
        self.executor._flash_s5_transport.reply([stale, stale])
        with self.assertRaises(ed.DriverError):
            self.driver.advance()
        self.assertIsNotNone(self.driver._transition)
        self.assertEqual(self.driver.route_generation, 0)

    def test_applied_transition_advances_the_route(self):
        self.driver.issue_transition(expected_route_generation=0, target_bank=1,
                                     storage_epochs=(5, 6),
                                     decode_replay_watermarks=(0, 0))
        acks = [w.TransitionAck(7, 0, 1, 1, (5, 6), 1, 1, (5, 6), rank)
                for rank in (0, 1)]
        self.executor._flash_s5_transport.reply(acks)
        self.driver.advance()
        self.assertIsNone(self.driver._transition)
        self.assertEqual(self.driver.route_generation, 1)

    def test_native_hook_does_not_drop_ordinary_decode_output(self):
        output = types.SimpleNamespace(total_num_scheduled_tokens=4)
        self.assertFalse(eh.park_prefill(self.driver.engine, output))

    def test_duplicate_rank_cannot_supply_two_acceptances(self):
        self.driver.advance()
        self.executor._flash_s5_transport.reply(self.acks())
        self.driver.advance()
        _, command = self.executor.last_command()
        first = self.acceptances(command)[0]
        self.executor._flash_s5_transport.reply([first, first])
        with self.assertRaises(ed.DriverError):
            self.driver.advance()
        self.assertEqual(self.manager.cache_calls, [])

    def test_paused_layer_repeats_cursor_with_new_operation_sequence(self):
        self.driver.advance()
        self.executor._flash_s5_transport.reply(self.acks())
        for _ in range(10):
            self.driver.advance()
            _, command = self.executor.last_command()
            if command.kind is w.SegmentKind.LAYER:
                break
            self.executor._flash_s5_transport.reply(self.acceptances(command))
        index = self.driver.index
        values = [replace(a, paused=True) if a.accepted else a
                  for a in self.acceptances(command)]
        self.executor._flash_s5_transport.reply(values)
        self.driver.advance()
        _, following = self.executor.last_command()
        self.assertEqual(self.driver.index, index)
        self.assertEqual(following.descriptor.layer_start, command.descriptor.layer_start)
        self.assertEqual(following.op_seq, command.op_seq + 1)

    def test_report_cannot_name_an_unaccepted_operation(self):
        self.driver.advance()
        self.executor._flash_s5_transport.reply(self.acks())
        self.driver.advance()
        fabricated = make_report(self.record.transaction, 0, (900,), ())
        with self.assertRaises(ed.DriverError):
            self.driver._ingest_report(fabricated)
        self.assertEqual(self.manager.cache_calls, [])

    def test_open_ack_is_not_a_safe_local_cancel(self):
        self.driver.advance()
        self.driver.stop('abort')
        self.executor._flash_s5_transport.reply(self.acks())
        self.driver.advance()
        self.assertTrue(self.controller.leases.held_blocks())
        self.assertEqual(self.manager.cache_calls, [])

    def test_duplicate_valid_transition_ack_does_not_advance_generation(self):
        self.driver.issue_transition(expected_route_generation=0, target_bank=1,
            storage_epochs=(5, 4), decode_replay_watermarks=(0, 0))
        ack = w.TransitionAck(7, 0, 1, 1, (5, 4), 1, 1, (5, 4), 0)
        self.executor._flash_s5_transport.reply([ack, ack])
        with self.assertRaises(ed.DriverError):
            self.driver.advance()
        self.assertEqual(self.driver.route_generation, 0)

    def test_one_transition_ack_cannot_move_all_ranks(self):
        self.driver.issue_transition(expected_route_generation=0, target_bank=1,
            storage_epochs=(5, 6), decode_replay_watermarks=(0, 0))
        ack = w.TransitionAck(7, 0, 1, 1, (5, 6), 1, 1, (5, 6), 0)
        with self.assertRaises(ed.DriverError):
            self.driver.accept_transition_ack(ack)
        self.assertEqual(self.driver.route_generation, 0)


# --- strict bounded sender --------------------------------------------------

class Ring:
    def __init__(self, *, max_chunk_bytes=1024, n_reader=2, written=0, reads=(0, 0)):
        self.max_chunk_bytes = max_chunk_bytes
        self.n_reader = n_reader
        self.meta = bytearray([written, *reads])

    def get_metadata(self, index):
        meta = self.meta

        class Context:
            def __enter__(self):
                return meta

            def __exit__(self, *exc):
                return False

        return Context()


class QueueMQ:
    def __init__(self, ring=None, *, remote=0):
        self.buffer = ring
        self.current_idx = 0
        self.n_remote_reader = remote
        self.shutting_down = False
        self._is_writer = True
        self.items = []

    def enqueue(self, obj):
        self.items.append(obj)


class BoundedSenderTests(unittest.TestCase):
    def test_writable_ring_accepts_one_write(self):
        queue = QueueMQ(Ring())
        sender = eh.BoundedSender(queue)
        self.assertTrue(sender.check(('m', (), {}, None)))
        sender(('m', (), {}, None))
        self.assertEqual(len(queue.items), 1)
        self.assertEqual(sender.sent, 1)

    def test_full_ring_is_backpressure_not_a_write(self):
        queue = QueueMQ(Ring(written=1, reads=(0, 0)))
        sender = eh.BoundedSender(queue)
        with self.assertRaises(Backpressure):
            sender.check(('m', (), {}, None))
        self.assertEqual(queue.items, [])
        self.assertEqual(sender.refused_full, 1)

    def test_remote_readers_and_oversize_are_refused(self):
        with self.assertRaises(Backpressure):
            eh.BoundedSender(QueueMQ(Ring(), remote=1)).check(('m', (), {}, None))
        sender = eh.BoundedSender(QueueMQ(Ring(max_chunk_bytes=8)))
        with self.assertRaises(Backpressure):
            sender.check(('m', (), {}, None))
        self.assertEqual(sender.refused_large, 1)

    def test_single_writer_ownership(self):
        queue = QueueMQ(Ring())
        eh.BoundedSender(queue)
        with self.assertRaises(E3ContractError):
            eh.BoundedSender(queue)

    def test_unserializable_envelope_is_refused(self):
        sender = eh.BoundedSender(QueueMQ(Ring()))
        with self.assertRaises(E3ContractError):
            sender.check(('m', (lambda: None,), {}, None))


# --- startup shape ----------------------------------------------------------

class StartupShapeTests(unittest.TestCase):
    def test_native_readiness_still_refuses(self):
        with self.assertRaises(s5_startup.StartupError):
            s5_startup.require_native_ready()

    def test_shape_is_discovered_only_from_a_declared_two_stage_model(self):
        self.assertIsNone(s5_startup._native_execution_shape(types.SimpleNamespace(),
                                                             (0, 1), env={}))
        engine = types.SimpleNamespace(vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(num_hidden_layers=8)),
            parallel_config=types.SimpleNamespace(pipeline_parallel_size=2)))
        env = {'VLLM_PP_LAYER_PARTITION': '3,5'}
        self.assertEqual(s5_startup._native_execution_shape(engine, (0, 1), env=env),
                         (8, 3, ((0, 0), (1, 1))))
        self.assertEqual(s5_startup._native_execution_shape(engine, (0, 1, 2, 3), env=env),
                         (8, 3, ((0, 0), (1, 0), (2, 1), (3, 1))))
        engine.vllm_config.parallel_config.pipeline_parallel_size = 1
        with self.assertRaises(s5_startup.StartupError):
            s5_startup._native_execution_shape(engine, (0, 1), env=env)


if __name__ == '__main__':
    unittest.main()
