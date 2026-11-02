"""Engine-side E3 segment driver: admission, typed driving loop, publication.

``s5_startup.install_engine`` constructs this object from the generated
``EngineCore.__init__`` seam (``executor_hooks.after_engine_startup``).  It owns
no device and no worker objects; it coordinates the workers through the typed
RPCs installed on ``GPUWorker`` and through one owned reply receiver.

The loop is deliberately split so an engine step never waits for a forward:

* :meth:`SegmentDriver.park` runs at the schedule seam.  A chunk the scheduler
  marked as admitted is taken out of the stock step (the whole step must be that
  one request), its metadata is captured once, and the dispatcher opens the
  transaction.  The step returns without launching stock work.
* :meth:`SegmentDriver.advance` runs at the top of each later step.  Each call
  polls the owned receiver (bounded) and either submits at most one typed
  command or consumes the replies that already arrived, then returns to decode.
  It never waits for a layer or a forward.
* When every rank's state-write proof and commit acknowledgment are in, the
  parked output goes through ``scheduler.update_from_output`` exactly once and
  is returned to the engine loop, so the last prompt token was sampled by the
  workers' ordinary sampling path.
* On abort/preempt/reset/pause/disconnect the driver keeps polling until every
  operation completes, then cancels the workers and releases its leases. It
  never drops ownership to make the stop look clean.

Admission is fail-closed.  The E2 scheduler integration that would emit
``SchedulerOutput.flash_s5_prefill_admission`` (and park the request before its
optimistic publication) is not implemented in this increment; without that
marker this driver never parks anything.  See the pass report's Open items.
"""
import math
import time
import copy
import hashlib
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field

from async_handoff import (
    CancelTransaction,
    CommitTransaction,
    DispatchReport,
    OpenPrefill,
    PollOperations,
    SubmitOperation,
    TransactionLease,
)
from e3_types import (
    ARENA_PAGES,
    CompletionKind,
    CompletionProof,
    DependencyRole,
    E3ContractError,
    E3Pending,
    LaneRole,
    OperationTicket,
    PAGE_TOKENS,
    TransactionId,
)
from lane_comm import PPConsumerReceipt
from lane_state import BatchRow, MetadataSnapshot, _freeze
from native_measurement import (PROGRESS_N, emit_receipt_lines,NativeMeasurementCensus, encode_report_vector,
                                decode_report_vector, ControlPathCounters, QsaControlCounters,
                                PROGRESS_DIRECT_METHODS, PROGRESS_META_BYTES, PROGRESS_META_KEY, PROGRESS_MODES,
                                ProgressTrace, StepAnatomy, encode_progress_meta)
from reply_transport import Backpressure, TransportError
from scheduler_hooks import (
    ADMISSION_PHASES,
    NATIVE_ADMISSION_PHASES,
    LeaseCategory,
    PrefillAdmissionController,
)
from wire_v1 import (
    BootstrapAck,
    BootstrapCommand,
    CancelAck,
    CancelRequest,
    PPReceiptAck,
    PPReceiptDelivery,
    PeerBinding,
    PeerBindingAck,
    PollRequest,
    QsaProfileGrant,
    QsaProfileQuery,
    QsaScheduleShape,
    LAYER_GROUP_SIZES,
    qsa_profile_agreement,
    ReleaseRequest,
    ReleaseAck,
    OpenAck,
    OpenRequest,
    OpSeqTracker,
    ROUTE_BANK_FULL,
    ROUTE_BANK_SPLIT,
    SegmentAcceptance,
    SegmentCommand,
    SegmentDescriptor,
    SegmentKind,
    TransitionAck,
    TransitionCommand,
    TransitionPending,
    WorkerCompletionReport,
    LeanCompletionReport,
    LeanStream,
)


def _wire_ready():
    if SegmentDescriptor is None:
        raise DriverError('wire-v1 is not packaged; the native driver cannot run')


# Reported when a chunk cannot be parked; the engine then runs the stock path.
_STOP_KINDS = ('abort', 'cancel', 'disconnect', 'preempt', 'reset', 'pause',
               'shutdown')

# Every journal/evidence timestamp in this module is a host ``CLOCK_MONOTONIC``
# nanosecond observation taken on the engine thread.  A CUDA event elapsed time
# is never stored here: it is not a host timestamp and cannot bound a window.
CLOCK_BASIS = 'clock-monotonic'
EVIDENCE_TIME_BASIS = 'engine-monotonic-observation-of-queried-report'
MAX_EVIDENCE_WINDOW_NS = 10 * 60 * 1000 * 1000 * 1000

# One bounded named utility carries exactly these actions.  Reads settle from
# queried engine state; every worker query and every route mutation is a
# deferred intent driven from the engine's own tick.
CONTROL_READ_ACTIONS = ('status', 'snapshot', 'transition_acks',
                        'bootstrap_acks', 'overlap_evidence', 'cancel')
# Admission is a typed EngineCore-owned intent, not a boolean or an env switch.
# Arming is synchronous engine state (no transport), like a read, but unlike a
# read it mutates the policy decision and therefore refuses while owned work is
# live.
CONTROL_STATE_ACTIONS = ('arm_admission', 'disarm_admission')
CONTROL_WORKER_ACTIONS = ('identity', 'census', 'barrier', 'drain', 'poll_native',
                          'qualification')
CONTROL_ROUTE_ACTIONS = ('bootstrap_engine_owner', 'enter_split', 'leave_split')
CONTROL_ACTIONS = (CONTROL_READ_ACTIONS + CONTROL_STATE_ACTIONS
                   + CONTROL_WORKER_ACTIONS + CONTROL_ROUTE_ACTIONS)
_CONTROL_TIMEOUT_ACTIONS = frozenset(CONTROL_WORKER_ACTIONS + CONTROL_ROUTE_ACTIONS)

# Worker observation actions that a paused LIVE ordinary decode cohort must not
# block: a frozen cohort is not native work.  The driver must still be idle and
# own no chunk/source/PP state, and teacher configuration stays stricter (it
# changes the qualification target, so it needs a configurable worker).
CONTROL_LIVE_COHORT_ACTIONS = ('identity', 'census', 'barrier', 'drain', 'poll_native')
CONTROL_LIVE_COHORT_QUALIFICATION = ('teacher_evidence', 'observer_begin',
                                     'observer_end', 'row1_evidence')


def _bounded_value(value, depth=0):
    """Plain, bounded copy for the journal and for a worker command payload.

    The journal is evidence, so nothing device- or object-shaped is stored: only
    observation values that a later reader can compare without trusting a repr.
    """
    if depth > 3:
        raise DriverError('bounded value is nested too deeply')
    if value is None or type(value) in (bool, int, str):
        if type(value) is str and len(value) > 4096:
            raise DriverError('bounded value string is too long')
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise DriverError('bounded value float is not finite')
        return value
    if type(value) in (list, tuple):
        if len(value) > 64:
            raise DriverError('bounded value sequence is too long')
        return [_bounded_value(item, depth + 1) for item in value]
    if type(value) is dict:
        if len(value) > 64:
            raise DriverError('bounded value mapping is too long')
        out = {}
        for key, item in value.items():
            if type(key) is not str or not key or len(key) > 64:
                raise DriverError('bounded value key is invalid')
            out[key] = _bounded_value(item, depth + 1)
        return out
    if type(value) in (set, frozenset):
        if len(value) > 64:
            raise DriverError('bounded value set is too long')
        return sorted((_bounded_value(item, depth + 1) for item in value),
                      key=lambda item: repr(item))
    raise DriverError('bounded value type is not plain: ' + type(value).__name__)



class DriverError(E3ContractError):
    pass


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise DriverError(name + ' must be a positive int')
    return value


@dataclass(frozen=True)
class PrefillAdmission:
    """What the scheduler must hand the engine to park one bulk prefill chunk.

    ``state_value`` is the host/model persistent state the worker commits
    exactly once.  It is never invented by the driver: admission without it is
    refused, and A's persistent-state record only accepts frozen CPU values.
    ``grammar_output`` is the grammar bitmask the ordinary sampling path
    consumes for the chunk (``None`` when no structured request).
    """

    transaction: TransactionId
    block_ids: tuple
    state_value: object
    grammar_output: object = None
    sparse_active: bool = False
    slot_mappings: tuple = ()
    num_computed: tuple = ()

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise DriverError('admission needs a TransactionId')
        if type(self.block_ids) is not tuple or not self.block_ids:
            raise DriverError('admission needs the leased block ids as a tuple')
        if any(type(b) is not int or b <= 0 for b in self.block_ids):
            raise DriverError('admission block ids must be positive ints')
        if len(set(self.block_ids)) != len(self.block_ids):
            raise DriverError('admission has duplicate blocks')
        object.__setattr__(self, 'state_value', _freeze(self.state_value))
        if self.state_value is None:
            raise DriverError('admission needs the admitted state value')
        if type(self.sparse_active) is not bool:
            raise DriverError('sparse_active must be bool')
        if type(self.slot_mappings) is not tuple or type(self.num_computed) is not tuple:
            raise DriverError('admission mappings must be tuples')


@dataclass
class ParkedBatch:
    """The chunk the engine removed from the stock step and owes one update."""

    admission: PrefillAdmission
    scheduler_output: object
    metadata: MetadataSnapshot
    transaction: TransactionId
    tickets: list = field(default_factory=list)
    output: object = None


class SegmentDriver:
    """Bounded, fail-closed driver for one parked prefill at a time."""

    def __init__(self, composition, *, executor, scheduler, engine=None,
                 clock=time.monotonic, poll_budget=8):
        if composition is None:
            raise DriverError('driver needs an enabled composition')
        self.composition = composition
        self.dispatcher = composition.dispatcher
        self.executor = executor
        self.scheduler = scheduler
        self.engine = engine
        self.clock = clock
        self.poll_budget = _positive(poll_budget, 'poll budget')
        self.parked: ParkedBatch | None = None
        self.phase = 'idle'
        self.stop_kind = None
        self._stop_phase = None
        self.error = None
        self._polls_sent = 0
        queues = getattr(executor, 'response_mqs', None) or ()
        self.ranks = tuple(range(len(queues)))
        if not self.ranks:
            raise DriverError('executor has no response queues for the typed RPCs')
        output_rank = getattr(executor, 'output_rank', 0)
        if type(output_rank) is not int or output_rank not in self.ranks:
            raise DriverError('executor output rank is not a live rank')
        self.output_rank = output_rank

    # -- installation -----------------------------------------------------
    def install(self):
        engine = self.engine
        if engine is None:
            raise DriverError('driver needs the engine object to install on')
        if getattr(engine, '_flash_s5_driver', None) is not None:
            raise DriverError('engine already has an E3 segment driver')
        if getattr(self.scheduler, '_flash_s5_driver', None) is not None:
            raise DriverError('scheduler already has an E3 segment driver')
        engine._flash_s5_driver = self
        self.scheduler._flash_s5_driver = self
        return self

    # -- schedule seam ----------------------------------------------------
    def park(self, scheduler_output):
        """Take an admitted bulk prefill out of the stock step, or return None."""
        if self.error is not None:
            raise DriverError(self.error)
        admission = getattr(scheduler_output, 'flash_s5_prefill_admission', None)
        if admission is None:
            return None
        if self.parked is not None or self.phase != 'idle':
            raise E3Pending('one prefill transaction is already parked')
        if not isinstance(admission, PrefillAdmission):
            raise DriverError('scheduler admission is not a PrefillAdmission record')
        transaction = admission.transaction
        counts = getattr(scheduler_output, 'num_scheduled_tokens', None)
        if type(counts) is not dict or not counts:
            raise DriverError('parked step has no scheduled tokens')
        if transaction.request_uid not in counts:
            raise DriverError('admitted request is not in the scheduled batch')
        if len(counts) != 1:
            raise DriverError(
                'extraction of a bulk prefill from a mixed decode batch needs the '
                'E2 scheduler integration; it is not implemented')
        scheduled = counts[transaction.request_uid]
        total = getattr(scheduler_output, 'total_num_scheduled_tokens', None)
        if type(scheduled) is not int or scheduled < 1 or total != scheduled:
            raise DriverError('parked step must schedule exactly the admitted chunk')
        if (transaction.chunk_end - transaction.chunk_start) != scheduled:
            raise DriverError('admitted chunk does not match the scheduled tokens')
        for name in ('scheduled_spec_decode_tokens', 'scheduled_encoder_inputs',
                     'preempted_req_ids', 'finished_req_ids'):
            if getattr(scheduler_output, name, None):
                raise DriverError(
                    'parking refuses a step with ' + name + '; the scheduler '
                    'integration must exclude them')
        metadata = self._snapshot(admission)
        lease = self.dispatcher.open(
            transaction, {rank: None for rank in self.ranks}, metadata=metadata,
            sparse_active=admission.sparse_active, dense_available=True)
        if not isinstance(lease, TransactionLease):
            raise DriverError('dispatcher open did not return a transaction lease')
        try:
            self.composition.frontier.reserve(transaction, admission.block_ids)
        except BaseException:
            lease._quarantine('stale_identity')
            raise
        self.parked = ParkedBatch(admission, scheduler_output, metadata, transaction)
        self._polls_sent = 0
        self.phase = 'open'
        return self.parked

    def _snapshot(self, admission):
        transaction = admission.transaction
        num_computed = admission.num_computed[0] if admission.num_computed else 0
        row = BatchRow(transaction.request_uid, transaction.request_generation, 0,
                       transaction.chunk_start, transaction.chunk_end, num_computed)
        return MetadataSnapshot.capture(
            transaction, (row,), block_ids=admission.block_ids,
            slot_mappings=admission.slot_mappings,
            num_computed=admission.num_computed)

    # -- step seam --------------------------------------------------------
    def advance(self):
        """One bounded step; returns ``(engine_outputs, True)`` on completion."""
        if self.error is not None:
            raise DriverError(self.error)
        if self.parked is None:
            return None
        report = self.dispatcher.poll(self.poll_budget)
        if not isinstance(report, DispatchReport):
            raise DriverError('dispatcher poll returned no report')
        if report.failed:
            self.error = 'E3 handoff was quarantined'
            raise DriverError(self.error)
        try:
            if self.stop_kind is not None:
                return self._drive_stop()
            return getattr(self, '_drive_' + self.phase)()
        except E3Pending:
            raise
        except BaseException:
            self.error = 'driver operation failed; ownership retained'
            if self.dispatcher.active is not None:
                self.dispatcher.active._quarantine('backend_failure')
            raise

    def _replies(self, kind):
        """Applied replies of one kind; phases advance on their own replies only."""
        return self.dispatcher.reply_count(kind)

    def _batch(self):
        parked = self.parked
        if parked is None:
            raise DriverError('no parked prefill transaction')
        return parked

    def _lease(self):
        parked = self._batch()
        lease = self.dispatcher.active
        if lease is None or lease.transaction != parked.transaction:
            raise DriverError('dispatcher lost the parked transaction')
        return lease

    def _send(self, message, kind, tickets=None):
        """Broadcast one typed command; every rank replies with its own ack.

        This requires a separately installed bounded sender (not implemented
        natively here), without registering a second reply expectation:
        the dispatcher owns the only one. The payload carries no rank-specific
        field, so one object is broadcast (the MQ copies it per worker). The RPC
        method is ``flash_s5_<kind>``; ``submit`` is the one method whose reply
        kind is ``operation``, and it must carry every rank's ticket.
        """
        method = 'flash_s5_' + kind
        transport_kind = 'operation' if kind == 'submit' else kind
        if transport_kind == 'operation' and tickets is None:
            raise DriverError('an operation submission needs one ticket per rank')

        bounded_send = getattr(self.executor, '_flash_s5_bounded_send', None)
        if not callable(bounded_send):
            raise DriverError('native bounded command sender is not installed')

        def send(_sequence):
            bounded_send((method, (message,), {}, None))

        self.dispatcher.submit(tickets, send, kind=transport_kind, ranks=self.ranks)
        return True

    def _tickets(self, op_seq, role, lanes):
        transaction = self._batch().transaction
        return {rank: OperationTicket(transaction, rank, lanes[rank], op_seq, role)
                for rank in self.ranks}

    # -- commit path ------------------------------------------------------
    def _drive_open(self):
        parked = self._batch()
        lease = self._lease()
        if lease.unbound:
            row = parked.metadata.rows[0]
            self._send(OpenPrefill(parked.transaction,
                                   row.chunk_end - row.chunk_start,
                                   parked.metadata), 'open')
            self.phase = 'open_sent'
        return None

    def _drive_open_sent(self):
        if self._replies('open') < 1:
            return None
        lease = self._lease()
        if lease.unbound:
            raise DriverError('open reply did not bind every worker lane')
        lanes = dict(lease.lanes)
        tickets = self._tickets(0, DependencyRole.PRODUCER, lanes)
        # Lease the arena blocks this parked chunk may still write, under the
        # operation identity the workers will actually complete.
        self.composition.scheduler_leases.lease_blocks(
            tickets[self.ranks[0]], LeaseCategory.ARENA,
            self._batch().admission.block_ids)
        self._batch().tickets.append(tickets)
        self._send(SubmitOperation(
            self._batch().transaction, 0, DependencyRole.PRODUCER,
            scheduler_output=self._batch().scheduler_output,
            grammar_output=self._batch().admission.grammar_output), 'submit',
            tuple(tickets[rank] for rank in self.ranks))
        self.phase = 'producer_sent'
        return None

    def _drive_producer_sent(self):
        if self._replies('operation') < 1:
            return None
        lease = self._lease()
        if lease.acked < len(self.ranks):
            return None
        tickets = self._tickets(1, DependencyRole.COMMIT, dict(lease.lanes))
        self._batch().tickets.append(tickets)
        self._send(SubmitOperation(
            self._batch().transaction, 1, DependencyRole.COMMIT,
            state_value=self._batch().admission.state_value), 'submit',
            tuple(tickets[rank] for rank in self.ranks))
        self.phase = 'commit_op_sent'
        return None

    def _drive_commit_op_sent(self):
        if self._replies('operation') < 2:
            return None
        lease = self._lease()
        if lease.acked < 2 * len(self.ranks):
            return None
        self._poll()
        return None

    def _poll(self):
        self._send(PollOperations(self._batch().transaction, self.poll_budget), 'poll')
        self._polls_sent += 1
        self.phase = 'poll_sent'

    def _drive_poll_sent(self):
        if self._replies('poll') < self._polls_sent:
            return None
        lease = self._lease()
        if lease.pending:
            self._poll()
            return None
        proof = self._commit_proof(lease)
        self.dispatcher.commit(proof)
        self._send(CommitTransaction(self._batch().transaction,
                                     proof.completion_seq,
                                     self._batch().admission.state_value), 'commit')
        self.phase = 'commit_ack_sent'
        return None

    def _commit_proof(self, lease):
        proofs = [p for p in lease.completed
                  if p.kind is CompletionKind.STATE_WRITE
                  and p.ticket.dependency_role is DependencyRole.COMMIT
                  and p.ticket.worker_rank == self.ranks[0]]
        if not proofs:
            raise DriverError('no recorded state-write completion to publish with')
        return proofs[-1]

    def _drive_commit_ack_sent(self):
        if self._replies('commit') < 1:
            return None
        parked = self._batch()
        lease = self._lease()
        output = None
        for rank, value in lease.worker_outputs.items():
            if rank == self.output_rank:
                output = value
            elif value is not None:
                raise DriverError('a non-output rank returned a model output')
        if output is None:
            raise DriverError(
                'no detached model output from the output rank; the parked chunk '
                'cannot be published through the scheduler')
        # A terminal method may mutate and then raise. Never call it twice.
        if getattr(parked, 'update_attempted', False):
            raise DriverError('scheduler publication was already attempted')
        parked.update_attempted = True
        outputs = self.scheduler.update_from_output(parked.scheduler_output, output)
        self.composition.frontier.publish(parked.transaction)
        self._release_leases(lease, publish=True)
        self.dispatcher.drain()
        self.parked = None
        self.phase = 'idle'
        return outputs, True

    def _release_leases(self, lease, *, publish):
        tickets = self._batch().tickets
        if not tickets:
            return
        ticket = tickets[0][self.ranks[0]]
        proof = next((p for p in lease.completed if p.ticket == ticket), None)
        if proof is None:
            raise DriverError('arena lease has no recorded completion proof')
        leases = self.composition.scheduler_leases
        if publish:
            leases.publish(ticket)
        leases.release(ticket, LeaseCategory.ARENA, proof)

    # -- stop path --------------------------------------------------------
    def stop(self, kind, *, drain=False):
        """Stop new work; completion polling continues until the chunk retires."""
        if self.parked is None:
            return None
        if kind not in _STOP_KINDS:
            raise DriverError('unknown stop kind')
        phase = self.dispatcher.stop(kind, drain=drain)
        if self.stop_kind is None:
            self.stop_kind = kind
            self._stop_phase = 'poll'
        return phase

    def _drive_stop(self):
        parked = self._batch()
        kind = self.stop_kind
        if kind is None:
            raise DriverError('stop phase without a stop kind')
        lease = self._lease()
        if lease.pending:
            # Keep polling until every submitted operation has a completion;
            # one bounded poll per step, and only after the last reply was read.
            if self._replies('poll') >= self._polls_sent:
                self._poll()
            return None
        if lease.unbound:
            # An unsent open has no remote resources; a sent open must receive
            # all its replies, then cancel every worker even with no operations.
            if self.dispatcher._inflight:
                return None
            self.composition.frontier.cancel(parked.transaction)
            self.dispatcher.drain()
            self.parked = None
            self.phase = 'idle'
            self.stop_kind = None
            self._stop_phase = None
            return None
        if self._stop_phase != 'cancel_sent':
            self._send(CancelTransaction(parked.transaction, kind), 'cancel')
            self._stop_phase = 'cancel_sent'
            return None
        if self._replies('cancel') < 1:
            return None
        self._release_leases(lease, publish=False)
        self.composition.frontier.cancel(parked.transaction)
        try:
            self.dispatcher.drain()
        except E3Pending:
            return None
        self.parked = None
        self.phase = 'idle'
        self.stop_kind = None
        self._stop_phase = None
        return None

    def finish_pause(self):
        if self.parked is not None:
            raise DriverError('pause requires the asynchronous drain to retire the chunk')
        report = self.composition.scheduler_leases.shutdown()
        if not report['clean']:
            raise DriverError('pause with outstanding E3 block leases')
        return True

    @property
    def active(self):
        return self.parked is not None


class PPTransferLedger:
    """Identity-validated PP transfer bookkeeping for the two-stage split.

    A send is releasable only after BOTH the sender's local completion proof
    and the receiver's remote ``PP_CONSUMER`` proof for the same transaction,
    stage and transfer sequence. Cancellation does not skip either.
    """

    def __init__(self, stages=2, *, stage1_ranks=None):
        if stages != 2:
            raise DriverError('the first cut has exactly two PP stages')
        _wire_ready()
        self.stages = stages
        self.stage1_ranks = None if stage1_ranks is None else tuple(stage1_ranks)
        self._transfers = {}

    def note_sent(self, transaction, seq, sender_rank=0):
        if type(transaction) is not TransactionId:
            raise DriverError('PP send needs a TransactionId')
        if type(seq) is not int or seq < 0:
            raise DriverError('PP transfer seq must be a nonnegative int')
        key = (transaction, seq)
        if key in self._transfers:
            raise DriverError('PP transfer seq already used')
        self._transfers[key] = dict(sender_rank=sender_rank, sender_done=None,
                                    consumer=None, released=False)
        return key

    def _entry(self, transaction, seq):
        key = (transaction, seq)
        entry = self._transfers.get(key)
        if entry is None:
            raise DriverError('unknown PP transfer sequence')
        return key, entry

    def note_sender_complete(self, transaction, seq, proof):
        """The sender's own operation completed (a device-event proof)."""
        if type(proof) is not CompletionProof:
            raise DriverError('PP sender completion needs a CompletionProof')
        if proof.ticket.transaction != transaction:
            raise DriverError('PP sender proof belongs to another transaction')
        if proof.kind is not CompletionKind.GPU_EVENT:
            raise DriverError('PP sender proof is not a device completion')
        if proof.ticket.dependency_role is not DependencyRole.HANDOFF:
            raise DriverError('PP sender proof is not a handoff operation')
        key, entry = self._entry(transaction, seq)
        if proof.ticket.worker_rank != entry['sender_rank']:
            raise DriverError('PP sender proof comes from another lane')
        if entry['sender_done'] is not None:
            if entry['sender_done'] != proof:
                raise DriverError('PP sender completion changed')
            return key
        entry['sender_done'] = proof
        return key

    def note_consumer(self, transaction, seq, stage, proof):
        """The receiver proved it consumed this exact transfer."""
        if stage != 1:
            raise DriverError('PP consumer proof belongs to stage 1')
        if type(proof) is not CompletionProof:
            raise DriverError('PP consumer completion needs a CompletionProof')
        if proof.ticket.transaction != transaction:
            raise DriverError('PP consumer proof belongs to another transaction')
        if proof.kind is not CompletionKind.PP_CONSUMER:
            raise DriverError('PP consumer proof has the wrong kind')
        if proof.ticket.dependency_role is not DependencyRole.HANDOFF:
            raise DriverError('PP consumer proof is not a handoff operation')
        if self.stage1_ranks is not None and proof.ticket.worker_rank not in self.stage1_ranks:
            raise DriverError('PP consumer proof comes from a sender lane')
        key, entry = self._entry(transaction, seq)
        if entry['consumer'] is not None:
            if entry['consumer'] != proof:
                raise DriverError('PP consumer completion changed')
            return key
        entry['consumer'] = proof
        return key

    def releasable(self, transaction, seq):
        key, entry = self._entry(transaction, seq)
        return entry['sender_done'] is not None and entry['consumer'] is not None

    def release_send(self, transaction, seq):
        """Consume the transfer's release exactly once; never bypass proofs."""
        key, entry = self._entry(transaction, seq)
        if entry['released']:
            raise E3ContractError('PP transfer was already released')
        if not self.releasable(transaction, seq):
            raise E3Pending('PP transfer lacks its sender/consumer completion')
        entry['released'] = True
        return entry['consumer']

    def outstanding(self, transaction):
        return tuple(sorted(seq for (txn, seq), entry in self._transfers.items()
                            if txn == transaction and not entry['released']))

    def summary(self):
        return dict(sent=len(self._transfers),
                    released=sum(1 for e in self._transfers.values() if e['released']))


_SEGMENT_ROLE = {
    SegmentKind.PREPARE: DependencyRole.PRODUCER,
    SegmentKind.PREFIX: DependencyRole.PRODUCER,
    SegmentKind.LAYER: DependencyRole.PRODUCER,
    SegmentKind.SUFFIX: DependencyRole.PRODUCER,
    SegmentKind.STATE_FINALIZE: DependencyRole.COMMIT,
    SegmentKind.PP_SEND: DependencyRole.HANDOFF,
    SegmentKind.PP_RECEIVE: DependencyRole.HANDOFF,
}


def _continuation(kind, cursor):
    """The host continuation position a TP peer's accepted cursor names.

    After STATE_FINALIZE a worker reports 'commit' once its own state event
    already completed and 'state_finalize' until then. That difference is
    rank-local device completion, not a later host position, so both are the
    same continuation for that command. Every other field and phase is kept.
    """
    if cursor is None:
        return None
    phase = cursor.phase
    if kind is SegmentKind.STATE_FINALIZE and phase == 'commit':
        phase = 'state_finalize'
    return (phase, cursor.layer, cursor.bucket)


def _agreement_detail(command, targets, accepted):
    """One bounded diagnostic for a refused TP agreement; built only on raise."""
    def flag(value):
        return int(bool(value))
    tx = command.transaction
    rows = []
    for item in accepted:
        if item.worker_rank not in targets:
            continue
        cursor = item.cursor
        rows.append('r%d:acc=%d,pend=%d,paused=%d,cursor=%s,profile=%d,tail=%d' % (
            item.worker_rank, flag(item.accepted), flag(item.pending), flag(item.paused),
            'none' if cursor is None else '%s/%d/%d' % (cursor.phase, cursor.layer, cursor.bucket),
            flag(item.profile_boundary), flag(item.tail_boundary)))
    uid = hashlib.sha256(str(tx.request_uid).encode()).hexdigest()[:16]
    return (' [command=%s stage=%d layer=%s op_seq=%d txn_seq=%d chunk=%d-%d uid_sha=%s %s]' % (
        getattr(command.kind, 'value', str(command.kind)), command.stage,
        command.descriptor.layer_start, command.op_seq, tx.txn_seq, tx.chunk_start,
        tx.chunk_end, uid, ' '.join(rows)))[:1024]


@dataclass
class PPSubmissionObligation:
    """Accepted sends require matching receive submissions even during stop.

    Commands are the original typed, idempotent worker commands. Acceptance
    does not imply device completion; the existing PP ledger owns that proof.
    """
    send: SegmentCommand
    route_generation: int
    receive: SegmentCommand | None = None
    send_accepted: set = field(default_factory=set)
    receive_accepted: set = field(default_factory=set)

    def __post_init__(self):
        if (type(self.send) is not SegmentCommand
                or self.send.kind is not SegmentKind.PP_SEND or self.send.stage != 0
                or type(self.route_generation) is not int or self.route_generation < 0):
            raise DriverError('PP submission obligation requires its typed sender/route')


@dataclass
class TPTailSubmissionObligation:
    """A partially accepted tail command must reach every active TP peer.

    The preceding all-rank tail boundary proves both peers observed their
    final consumer before this command is issued. The original command is
    retained for exact idempotent retries during a stop.
    """
    command: SegmentCommand
    boundary_op_seq: int
    route_generation: int
    accepted: set = field(default_factory=set)

    def __post_init__(self):
        if (type(self.command) is not SegmentCommand
                or self.command.kind is not SegmentKind.LAYER
                or type(self.boundary_op_seq) is not int
                or self.boundary_op_seq >= self.command.op_seq
                or type(self.route_generation) is not int
                or self.route_generation < 0):
            raise DriverError('TP tail obligation needs its typed boundary and route')


_PROGRESS_LOG = None
# Plan entries a burst owner wait may advance; every other kind stays on the tick.
_WAIT_SEGMENT_KINDS = frozenset((SegmentKind.PREFIX, SegmentKind.LAYER, SegmentKind.SUFFIX))


def _progress_logger():
    global _PROGRESS_LOG
    if _PROGRESS_LOG is None:
        # vLLM's init_logger is the path s5_startup's INFO line is proven on.
        try:
            from vllm.logger import init_logger
        except ImportError:
            import logging
            _PROGRESS_LOG = logging.getLogger(__name__)
        else:
            _PROGRESS_LOG = init_logger(__name__)
    return _PROGRESS_LOG


def _progress_outcome(kind, value):
    """Descriptive reply class for a receipt; never consulted by the driver."""
    try:
        items = tuple(value)
        if kind == 'segment':
            if any(getattr(item, 'pending', False) for item in items):
                return 'pending'
            accepted = [item for item in items if getattr(item, 'accepted', False)]
            if any(item.profile_boundary for item in accepted):
                return 'boundary'
            if any(item.tail_boundary for item in accepted):
                return 'tail-ready'
            if any(item.paused for item in accepted):
                return 'paused'
            return 'accepted'
        if kind == 'profile':
            states = {getattr(item, 'status', None) for item in items} - {'skipped'}
            return 'pending' if 'pending' in states else 'ready'
        return kind
    except Exception:
        return 'unknown'


class FusedQsa:
    """Step A: one bounded worker command per dense QSA layer.

    Installed at startup by FLASH_S5_QSA_FUSED=1, off otherwise. ``selected`` is
    the per-arm choice, changed only while paused, idle and FULL, and read once
    per transaction before OPEN. A selected transaction has no stepped fallback.
    """
    # At most this many accepted commands not yet host-queried complete, per peer.
    # Held-g R1: a command takes about 4 ms of device time (12 ms on a QSA layer) against a
    # cadence of 14 ms live and 5 ms solo, so a deeper queue only lengthens a stop's drain.
    RUN_AHEAD = 4
    # No accepted op and no completion on any rank for this long fails closed.
    # The reply timeout's value: a worker command may legitimately take a while.
    STALL_S = 300.

    def __init__(self, run_ahead=RUN_AHEAD, stall_s=STALL_S):
        self.selected = False
        self.run_ahead = run_ahead
        self.stall_s = stall_s
        self.counters = dict(selections=0, chunks=0, grants=0, fused_commands=0,
                             run_ahead_polls=0, ledger_checks=0, stops=0,
                             stop_rounds_max=0, stop_ms_max=0)

    def snapshot(self):
        return dict(selected=self.selected, run_ahead=self.run_ahead, stall_s=self.stall_s,
                    **self.counters)


def fused_expected_shape(valid_tokens):
    """The engine's own expectation, from the admission alone."""
    pages = (max(valid_tokens, 1) - 1) // PAGE_TOKENS + 1
    counts = (1,) * ((pages + ARENA_PAGES - 1) // ARENA_PAGES)
    return QsaScheduleShape('dense_dma', counts, counts)


class LayerGroups:
    """Step B: consecutive LAYER entries of a stage as one bounded worker command.

    Installed at startup by FLASH_S5_LAYER_GROUPS=1, off otherwise. ``size`` is the per-arm
    choice, changed only while paused, idle and FULL, read once per transaction before OPEN.
    Size 1 is Step A's command stream.
    """
    # At most this many accepted commands not yet host-queried complete, per peer. The bound
    # is in commands, so a larger group needs a shallower queue to keep a stop's drain where
    # Step A has it. Held-g R1 device time per command: about 4 ms a GDN layer, 12 ms a QSA
    # layer. Worst queue: size 1, 4 commands = 4 layers (up to about 24 ms for 3 GDN + 1 QSA);
    # size 2, 3 commands = 6 layers (about 40 ms); size 4, 2 commands = 8 layers (about 48 ms).
    RUN_AHEAD = {1: FusedQsa.RUN_AHEAD, 2: 3, 4: 2}

    def __init__(self):
        self.size = 1
        self.counters = dict(selections=0, chunks=0, plan_commands=0, group_commands=0,
                             group_layers=0, group_layers_max=0)

    def snapshot(self):
        return dict(size=self.size, run_ahead=self.RUN_AHEAD[self.size], **self.counters)


def layer_group_plan(low, high, qsa_layers, size):
    """[(start, end), ...] over one stage's layers [low, high).

    A group closes at ``size`` or right after a QSA layer, so a group holds at most one QSA
    layer, last, and the stage end closes whatever is left. QSA positions come from the
    PREPARE plan, never from the layer number.
    """
    groups = []
    start = low
    for layer in range(low, high):
        if layer + 1 - start == size or layer in qsa_layers or layer + 1 == high:
            groups.append((start, layer + 1))
            start = layer + 1
    return groups


class NativeProgress:
    """Held-f: native progress receipts and the switchable candidate (owner thread only).

    ``installed`` is the startup setting (``trace`` or ``burst``); ``mode`` is
    the per-arm selection, ``current`` until a typed ``native_progress`` control
    sets ``burst`` (only when installed ``burst``). Recording happens only while
    an auto-arm-owned native chunk is open; a manual cycle is never recorded and
    never changed. Receipts are host CLOCK_MONOTONIC stamps, one line per chunk.
    """
    WAIT_SEND_LIMIT = 8
    DIRECT_KINDS = frozenset(('segment', 'profile', 'poll'))
    MODES = ('current', 'burst')
    # Held-g L3, the per-arm ``idle`` selection: ``step`` is held-f's engine
    # step pacing; ``wait`` lets an owner step that stock work could not use
    # wait at most IDLE_WAIT_S for a native reply instead (see idle_wait).
    IDLE = ('step', 'wait')
    IDLE_WAIT_S = 0.001
    # Held-g L4, the per-arm ``replies`` selection, applied at each chunk's
    # OPEN: ``full`` is every earlier reply; ``lean`` sends completion deltas.
    REPLIES = ('full', 'lean')

    def __init__(self, installed, executor, *, driver=None, clock_ns=time.monotonic_ns):
        if installed not in PROGRESS_MODES:
            raise DriverError('invalid native progress setting')
        self.installed = installed
        self.mode = 'current'
        self.idle = 'step'
        self.replies = 'full'
        self.executor = executor
        self.driver = driver
        self.clock_ns = clock_ns
        # Engine owner thread of the open chunk (begin runs inside advance) and
        # the kind of the last stock RPC submitted while recording.
        self.owner = None
        self.last_stock_kind = None
        self.trace = ProgressTrace('engine')
        self.recording = False
        self.identity = None
        self.native = {}
        # The row whose reply is being applied: the apply that ends the chunk
        # (release) is recorded at the flush, inside its own line.
        self.applying = None
        self.counters = dict.fromkeys((
            'chunks', 'commands', 'direct_commands', 'stock_rpcs', 'waits',
            'wait_sends', 'wait_capped', 'line_errors', 'mode_changes',
            'idle_steps', 'idle_timeouts', 'lean_chunks', 'lean_reports',
            'lean_tokens'), 0)
        # Held-g R5: the engine step loop of the recording chunk.
        self.steps = StepAnatomy(self.counters, clock_ns=clock_ns)

    def snapshot(self):
        return dict(schema='s5-native-progress-status/1', installed=self.installed,
                    mode=self.mode, idle=self.idle, replies=self.replies,
                    recording=self.recording,
                    wait_send_limit=self.WAIT_SEND_LIMIT,
                    direct_kinds=sorted(self.DIRECT_KINDS),
                    trace_lines=self.trace.lines_out, trace_dropped=self.trace.dropped_total,
                    **self.counters)

    def burst(self):
        return self.recording and self.mode == 'burst'

    def wait_allowed(self):
        """Owner thread, burst chunk open, and the ring's last stock RPC is a sample."""
        import threading
        return (self.burst() and self.driver is not None
                and threading.get_ident() == self.owner
                and self.last_stock_kind == 'sample_tokens')

    def meta(self, sequence, direct):
        return encode_progress_meta(sequence, direct)

    @staticmethod
    def with_meta(envelope, meta):
        method, args, kwargs, rank = envelope
        merged = dict(kwargs or {})
        merged[PROGRESS_META_KEY] = meta
        return (method, args, merged, rank)

    def begin(self, identity):
        import threading
        self.recording = True
        self.identity = identity
        self.native = {}
        self.applying = None
        self.owner = threading.get_ident()
        self.last_stock_kind = None
        self.executor._flash_s5_progress_recording = self
        self.steps.reset()

    def applied(self, row):
        """Record a native row after its apply, unless the chunk flush took it."""
        if self.applying is row:
            self.applying = None
            row[PROGRESS_N['t_apply']] = self.clock_ns()
            self.trace.record('n', *row)

    def end(self, logger):
        """Flush this chunk's lines and stop recording; never raises."""
        was = self.recording
        self.recording = False
        self.executor._flash_s5_progress_recording = None
        if not was:
            return
        self.counters['chunks'] += 1
        self.owner = None
        self.last_stock_kind = None
        row, self.applying = self.applying, None
        try:
            if row is not None:
                row[PROGRESS_N['t_apply']] = self.clock_ns()
                self.trace.record('n', *row)
            # Alpha-next PLAN 2: one atomic write per part through the process receipt writer.
            emit_receipt_lines(self.trace.lines(**(self.identity or {}), mode=self.mode,
                                                open_rows=len(self.native)), logger)
        except Exception:
            self.counters['line_errors'] += 1
        try:
            # Held-g R5: the chunk's step anatomy, on its own schema.
            emit_receipt_lines(self.steps.lines(**(self.identity or {}), mode=self.mode, idle=self.idle,
                                                replies=self.replies), logger)
        except Exception:
            self.counters['line_errors'] += 1
        self.identity = None
        self.native = {}


class NativeSegmentDriver:
    """Wire-v1 bounded driver: one segment per tick, no whole-forward call.

    The scheduler admits the chunk pre-schedule (see
    :class:`~scheduler_hooks.PrefillAdmissionController`); this driver never
    parks a ``SchedulerOutput`` and never returns fabricated model outputs.
    Each ``advance`` drains ready replies and then submits at most one wire
    ``SegmentCommand`` (or one poll). Completion is only the controller's
    publication after every rank's host-queried ``STATE_WRITE`` and every PP
    consumer proof; the final token is then handed back to stock decode.
    """

    def __init__(self, controller, *, executor, scheduler, engine=None,
                 layer_count, layer_split, stage_of, clock=time.monotonic,
                 poll_budget=8, reply_timeout_s=300.0, stop_budget=64,
                 stop_tick_budget=8, monotonic_ns=None, journal_limit=1024,
                 control_timeout_s=None):
        _wire_ready()
        if type(controller) is not PrefillAdmissionController:
            raise DriverError('native driver needs the admission controller')
        if executor is None or scheduler is None:
            raise DriverError('native driver needs the executor and scheduler')
        if controller._scheduler is not scheduler:
            raise DriverError('admission controller belongs to another scheduler')
        if type(layer_count) is not int or layer_count < 2:
            raise DriverError('layer_count must cover at least two layers')
        if type(layer_split) is not int or not 0 < layer_split < layer_count:
            raise DriverError('the layer split must leave layers on both stages')
        stage_of = tuple(stage_of)
        if not stage_of or any(len(item) != 2 for item in stage_of):
            raise DriverError('stage_of is a (rank, stage) tuple per worker')
        ranks = tuple(sorted(rank for rank, _ in stage_of))
        if ranks != controller.ranks:
            raise DriverError('stage_of must cover every admitted rank')
        stages = {stage for _, stage in stage_of}
        if stages != {0, 1}:
            raise DriverError('the first cut has exactly two PP stages')
        self.controller = controller
        self.executor = executor
        self.scheduler = scheduler
        self.engine = engine
        self.layer_count = layer_count
        self.layer_split = layer_split
        self.stage_of = dict(stage_of)
        self.ranks = ranks
        self.stage_ranks = {0: tuple(r for r in ranks if self.stage_of[r] == 0),
                            1: tuple(r for r in ranks if self.stage_of[r] == 1)}
        if not self.stage_ranks[0] or not self.stage_ranks[1]:
            raise DriverError('both PP stages need at least one rank')
        self.stage_layers = {0: (0, layer_split), 1: (layer_split, layer_count)}
        self.clock = clock
        self.poll_budget = _positive(poll_budget, 'poll budget')
        if (type(reply_timeout_s) not in (int, float)
                or not math.isfinite(reply_timeout_s) or reply_timeout_s <= 0):
            raise DriverError('reply timeout must be positive')
        self.reply_timeout_s = float(reply_timeout_s)
        if len(self.stage_ranks[0]) != len(self.stage_ranks[1]):
            raise DriverError('TP rank count differs across PP stages')
        self.pp_peers = dict(zip(self.stage_ranks[0], self.stage_ranks[1]))
        self.pp_ledgers = {sender: PPTransferLedger(2, stage1_ranks=(receiver,))
                           for sender, receiver in self.pp_peers.items()}
        self.pp = self.pp_ledgers[self.stage_ranks[0][0]]  # compatibility summary only
        self.stop_budget = _positive(stop_budget, 'stop budget')
        self.stop_tick_budget = _positive(stop_tick_budget, 'stop tick budget')
        self.engine_epoch = controller.engine_epoch
        self.measurement_census = NativeMeasurementCensus(
            engine_epoch=self.engine_epoch, ranks=self.ranks,
            stage_ranks=self.stage_ranks, pp_senders=self.stage_ranks[0],
            pool_chunk_capacity=True)
        self.route_generation = 0
        import os
        stop_at = os.environ.get('FLASH_S5_NATIVE_QSA_STOP_AFTER', '0')
        if stop_at not in ('0', '101200'):
            raise DriverError('native43-stop-frontier-invalid')
        if stop_at != '0' and os.environ.get('FLASH_S5_NATIVE_QSA_TRACE', '0') != '1':
            raise DriverError('native43-stop-needs-trace')
        self._native43 = (None if stop_at == '0' else
            dict(stop_at=int(stop_at), checkpoint=None, cancel_started=False))
        self.tx = None
        self.admission = None
        self.lanes = {}
        self.trackers = {}
        self._last_lane_sequences = {}
        self.tickets = {}
        self.accepted_tickets = {}
        self._accepted_descriptors = {}
        self.phase = 'idle'
        self.plan = []
        self.index = 0
        self.bound_ticket = None
        self.pp_send_ticket = None
        self.last_ticket = {0: None, 1: None}
        self.transfer_seq = 0
        self.reports = {}
        # Held-g L4: this transaction's reply format and the per-rank streams.
        self._replies = 'full'
        self._lean_streams = {}
        self.stop_kind = None
        self.error = None
        self._inflight = {}
        self._retry = None
        self._profile_query = None
        self._profile_observed = {}
        self._profile_grants = {}
        self._tail_boundary = None
        self._tail_submission = None
        self._profile_deadline = None
        self._profile_poll_count = 0
        # Engine-lifetime count, not a count inferred from the raw ring/window.
        self._profile_mismatches = 0
        self._profile_timeouts = 0
        # The raw journal is a bounded recent-event ring. Preserve the first
        # causal profile refusal and the latest one independently so cleanup
        # status can identify both peers even after a long row evicts the ring.
        self._profile_refusal_first = None
        self._profile_refusal_latest = None
        self._profile_refusal_count = 0
        self._stock_steps_skipped = 0
        self._profile_failure = None
        self._profile_failed_request = None
        self._transition = None
        self._transition_retry = False
        self._transition_deadline = None
        self._last_transition_acks = {}
        self.pp_send_seq = None
        self.pp_receive_seq = None
        self.receiver_ticket = None
        self._pp_submission = None
        self._bindings = {}
        self._bind_index = 0
        self._pending_binding = None
        self._receipts = {}
        self._receipt_inflight = None
        self._receipt_acked = set()
        self._receipt_last = None
        self._consumer_proofs = {}
        self._pp_noted = False
        self._stop_acks = {}
        self._stop_rounds = 0
        self._stop_ticks = 0
        self._cancel_targets = frozenset()
        self._release_acks = {}
        self._release_attempts = 0
        # The exact worker request row a certified nonterminal all-rank release
        # left open.  It is bound from the real ReleaseAck rows and the final
        # queried reports, never inferred from a queue size or ``active`` flag.
        # It survives ``_reset``: the row belongs to the request *between*
        # chunks, not to the chunk whose storage was just released.  A same
        # request reopen moves it back to an active transaction; only an
        # all-rank typed terminal release/cancel clears it.
        self._retained = None
        self._retained_cancel_request = None
        self._retained_cancel_acks = {}
        self._last_native_closure = None
        # Engine-owner bootstrap and route completion.  These survive a chunk
        # reset: the worker storage epoch and replay census are worker-stable.
        self._bootstrapped = False
        self._bootstrap_command = None
        self._storage_epochs = {}
        self._decode_watermarks = {}
        self._split_ready = False
        self._return_pending = None
        # Host-monotonic journal of real admitted segments, accepted replies,
        # queried STATE_WRITE reports, PP receipts, source holds and committed
        # frontiers.  It is preserved across a chunk reset (only chunk *state*
        # is dropped); the window read refuses a non-monotonic or stale epoch.
        if monotonic_ns is not None and not callable(monotonic_ns):
            raise DriverError('monotonic_ns must be callable or None')
        self._monotonic_ns = monotonic_ns if monotonic_ns is not None else time.monotonic_ns
        if type(journal_limit) is not int or journal_limit < 8:
            raise DriverError('journal_limit must be an int of at least 8')
        self.journal_limit = journal_limit
        self._journal = deque(maxlen=journal_limit)
        self._journal_dropped = 0
        self.clock_basis = CLOCK_BASIS
        self.control_path = ControlPathCounters(self.engine_epoch)
        self.qsa_control = QsaControlCounters(self.engine_epoch, self.stage_layers)
        # One bounded control intent at a time.  The utility dispatch only
        # registers it; every submission/settlement happens on the engine tick.
        timeout = reply_timeout_s if control_timeout_s is None else control_timeout_s
        if (type(timeout) not in (int, float) or not math.isfinite(timeout)
                or timeout <= 0):
            raise DriverError('control timeout must be positive')
        self._control_timeout_s = float(timeout)
        self._control = None
        self._control_future = None
        self._control_seq = None
        self._control_deadline = None
        self._control_reply = None
        self._control_orphans = []
        self._last_bootstrap_acks = {}
        self._last_bootstrap_observed_ns = None
        self._last_transition_observed_ns = None
        self._last_census = None
        self._last_census_observed_ns = None
        # Explicit native-control route ownership: set only when an explicit
        # ``enter_split`` control intent settles on the split route.  An idle
        # pause then drains the frozen cohort without abandoning the selected
        # route.  It is route-stable state, cleared by the all-rank FULL
        # transition; chunk reset never touches it.
        self._control_route = None
        # The engine-owned native-admission intent.  ``None`` is the default
        # (stock admission for every request); a typed record is the only way
        # to park a native chunk.  It outlives chunk reset by design.
        self._admission_intent = None
        self._admission_seq = 0
        # Held-f: absent unless FLASH_S5_NATIVE_PROGRESS installed it.
        self.progress = None
        self._progress_origin = 'tick'
        self._in_wait = False
        # Step A: absent unless FLASH_S5_QSA_FUSED installed it.
        self.qsa_fused = None
        self._tx_fused = False
        self._fused_mark = None
        self._fused_mark_at = None
        self._fused_stop_at = None
        # Step B: absent unless FLASH_S5_LAYER_GROUPS installed it.
        self.layer_groups = None
        self._tx_group = 1

    def install_layer_groups(self):
        """Step B startup seam; the per-arm choice is the native_layer_group control."""
        if self.layer_groups is not None:
            raise DriverError('layer groups are already installed')
        if self.qsa_fused is None:
            raise DriverError('layer groups require the fused QSA switch')
        self.layer_groups = LayerGroups()
        return self.layer_groups

    def _layer_group_control(self, command):
        """Typed per-arm size; only installed, only paused idle FULL, above 1 only fused."""
        if self.error is not None:
            raise DriverError(self.error)
        if set(command) - {'action', 'size', 'engine_epoch'}:
            raise DriverError('native-layer-group-command-fields-differ')
        epoch = command.get('engine_epoch')
        if epoch is not None and (type(epoch) is not int or epoch != self.engine_epoch):
            raise DriverError('native-control-engine-epoch-differs')
        size = command.get('size')
        if type(size) is not int or size not in LAYER_GROUP_SIZES:
            raise DriverError('native-layer-group-invalid')
        if size != 1 and not self.qsa_fused.selected:
            raise DriverError('native-layer-group-requires-fused')
        policy = getattr(self, 'auto_arm', None)
        recording = self.progress is not None and self.progress.recording
        if (not self._engine_paused() or self._split_ready or self.tx is not None
                or self.controller.active or self._inflight or self._control is not None
                or self._route_exchange() is not None or self._return_pending is not None
                or self._retained is not None or self._admission_intent is not None
                or policy is None or policy.active or recording):
            raise DriverError('native-layer-group-requires-paused-idle-full')
        previous = self.layer_groups.size
        self.layer_groups.size = size
        self.layer_groups.counters['selections'] += int(previous != size)
        now = int(self._monotonic_ns())
        return {'schema': 1, 'action': 'native_layer_group', 'size': size, 'previous': previous,
                'engine_epoch': self.engine_epoch, 'route': 'full',
                'route_generation': self.route_generation,
                'observed_clock_ns': now, 'clock': self._clock_record(now)}

    def _group_stage_plan(self, stage, qsa_layers):
        """Replace one stage's LAYER entries by its groups, once its peers agreed at PREPARE."""
        low, high = self.stage_layers[stage]
        at = [index for index, entry in enumerate(self.plan)
              if entry['kind'] is SegmentKind.LAYER and entry['stage'] == stage]
        if at != list(range(at[0], at[0] + high - low)) or at[0] < self.index:
            raise DriverError('a layer group plan needs the stage whole and unsent')
        size = self._tx_group
        groups = layer_group_plan(low, high, qsa_layers, size)
        self.plan[at[0]:at[-1] + 1] = [
            dict(kind=SegmentKind.LAYER, stage=stage, layer=start, layer_end=end, group=size)
            for start, end in groups]
        self.layer_groups.counters['plan_commands'] += len(groups)

    def install_qsa_fused(self):
        """Step A startup seam; the per-arm choice is the native_qsa_fused control."""
        if self.qsa_fused is not None:
            raise DriverError('fused QSA is already installed')
        if getattr(self, 'auto_arm', None) is None:
            raise DriverError('fused QSA requires the auto-arm policy')
        self.qsa_fused = FusedQsa()
        return self.qsa_fused

    def _fused_control(self, command):
        """Typed per-arm selection; only installed, only paused idle FULL."""
        if self.error is not None:
            raise DriverError(self.error)
        if set(command) - {'action', 'fused', 'engine_epoch'}:
            raise DriverError('native-qsa-fused-command-fields-differ')
        epoch = command.get('engine_epoch')
        if epoch is not None and (type(epoch) is not int or epoch != self.engine_epoch):
            raise DriverError('native-control-engine-epoch-differs')
        fused = command.get('fused')
        if type(fused) is not bool:
            raise DriverError('native-qsa-fused-invalid')
        if not fused and self.layer_groups is not None and self.layer_groups.size != 1:
            raise DriverError('native-layer-group-requires-fused')
        policy = getattr(self, 'auto_arm', None)
        recording = self.progress is not None and self.progress.recording
        if (not self._engine_paused() or self._split_ready or self.tx is not None
                or self.controller.active or self._inflight or self._control is not None
                or self._route_exchange() is not None or self._return_pending is not None
                or self._retained is not None or self._admission_intent is not None
                or policy is None or policy.active or recording):
            raise DriverError('native-qsa-fused-requires-paused-idle-full')
        previous = self.qsa_fused.selected
        self.qsa_fused.selected = fused
        self.qsa_fused.counters['selections'] += int(previous != fused)
        now = int(self._monotonic_ns())
        return {'schema': 1, 'action': 'native_qsa_fused', 'fused': fused, 'previous': previous,
                'engine_epoch': self.engine_epoch, 'route': 'full',
                'route_generation': self.route_generation,
                'observed_clock_ns': now, 'clock': self._clock_record(now)}

    def _fused_agree(self, descriptor, targets, accepted):
        """Both stage peers' commitment for this command. PREPARE also grants the layers."""
        items = [item for item in accepted if item.worker_rank in targets]
        self.qsa_fused.counters['ledger_checks'] += 1
        if len({item.collectives for item in items}) != 1 or items[0].collectives is None:
            raise DriverError('tp-ledger-mismatch' + _agreement_detail(self._command, targets, accepted))
        if descriptor.kind is SegmentKind.LAYER and self._command.profile is not None:
            self.qsa_fused.counters['fused_commands'] += 1
        if self._tx_group != 1 and descriptor.kind is SegmentKind.LAYER:
            layers = descriptor.layer_end - descriptor.layer_start
            counters = self.layer_groups.counters
            counters['group_commands'] += 1
            counters['group_layers'] += layers
            counters['group_layers_max'] = max(counters['group_layers_max'], layers)
        if descriptor.kind is not SegmentKind.PREPARE:
            return
        profiles = {item.dense_profiles for item in items}
        if len(profiles) != 1 or None in profiles:
            raise DriverError('fused-profile-disagreement' + _agreement_detail(self._command, targets, accepted))
        expected = fused_expected_shape(self.tx.chunk_end)
        low, high = self.stage_layers[descriptor.stage]
        rows = profiles.pop()
        for layer, shape in rows:
            if not low <= layer < high or shape != expected:
                raise DriverError('fused-profile-differs-from-the-admission')
            query = QsaProfileQuery(self.tx, self.admission.state_epoch, descriptor.stage, layer,
                                    self._command.op_seq, tuple(targets))
            self._profile_grants[(descriptor.stage, layer)] = QsaProfileGrant(query, shape, fused=True)
            self.qsa_fused.counters['grants'] += 1
        if self._tx_group != 1:
            self._group_stage_plan(descriptor.stage, {layer for layer, _ in rows})

    def _fused_hold(self, entry):
        """Run-ahead bound without worker pending: a LAYER waits on the peers' reports."""
        if entry['kind'] is not SegmentKind.LAYER:
            return False
        limit = self.qsa_fused.run_ahead
        if self._tx_group != 1:
            limit = LayerGroups.RUN_AHEAD[self._tx_group]
        return any(rank in self.reports and len(self.reports[rank].pending_op_seqs) >= limit
                   for rank in self.stage_ranks[entry['stage']])

    def _fused_watchdog(self):
        """Fail closed on no progress anywhere. A collective stuck on the device has
        no safe release: ownership stays retained and no device wait is issued."""
        mark = (len(self.accepted_tickets),
                tuple(sorted((rank, report.completed_sequence) for rank, report in self.reports.items())))
        now = self.clock()
        if mark != self._fused_mark or self._fused_mark_at is None:
            self._fused_mark, self._fused_mark_at = mark, now
            return
        if now - self._fused_mark_at > self.qsa_fused.stall_s:
            self._record('tp_collective_stall', transaction=self.tx.txn_seq,
                         accepted=mark[0], completed=[list(row) for row in mark[1]])
            raise DriverError('tp-collective-stall')

    def install_progress(self, installed):
        """Held-f startup seam: receipts (``trace``) or receipts + candidate (``burst``)."""
        if self.progress is not None:
            raise DriverError('native progress is already installed')
        if getattr(self, 'auto_arm', None) is None:
            raise DriverError('native progress requires the auto-arm policy')
        self.progress = NativeProgress(installed, self.executor, driver=self)
        self.executor._flash_s5_progress_recording = None
        return self.progress

    def _wait_permit(self, nested=False):
        """The burst in-wait permit: everything advance() does besides _drain +
        _drive is a no-op, the chunk is policy-owned, and only an in-chunk
        PREFIX/LAYER/SUFFIX segment or a QSA profile query is in flight or next.
        PREPARE (controller.bind), PP, STATE_FINALIZE, finalize polls, receipts,
        release, open and bind stay on the tick, before schedule()."""
        progress = self.progress
        if (progress is None or not progress.burst() or (self._in_wait and not nested)
                or self.error is not None or self.stop_kind is not None
                or self._profile_failure is not None or self._control is not None
                or self._control_orphans or self._route_exchange() is not None
                or self._return_pending is not None or not self._bootstrapped
                or not self._split_ready or self.tx is None or not self._progress_owned()):
            return False
        inflight = tuple(self._inflight.values())
        if self.phase == 'profile':
            return inflight in ((), ('profile',))
        if self.phase != 'segments' or len(inflight) > 1:
            return False
        if inflight:
            if self._tx_fused and inflight == ('poll',):
                return True
            return (inflight == ('segment',) and self._command is not None
                    and self._command.kind in _WAIT_SEGMENT_KINDS)
        if self._retry is not None:
            return self._retry.kind in _WAIT_SEGMENT_KINDS
        return (self.index < len(self.plan)
                and self.plan[self.index]['kind'] in _WAIT_SEGMENT_KINDS)

    def progress_in_wait(self):
        """One permitted native advance inside an owner stock wait (held-f burst).

        The same _drain + _drive that advance() reaches under the permit, on the
        owner thread. Returns the commands sent (0 or 1), or None when the
        permit does not hold or the ring refused (the caller then waits as
        stock does). Error semantics are advance()'s.
        """
        # The pair-boundary condition (owner thread, last stock RPC a sample)
        # is re-read before the drain and again before the send: only the
        # owner thread submits execute_model/sample_tokens, and it is here.
        if not self._wait_permit() or not self.progress.wait_allowed():
            return None
        self._in_wait = True
        self._progress_origin = 'wait'
        try:
            self._drain()
            if (self._inflight or not self._wait_permit(nested=True)
                    or not self.progress.wait_allowed()):
                return 0
            self._drive()
            return int(bool(self._inflight))
        except Backpressure:
            return None
        except E3Pending:
            raise
        except BaseException:
            self.error = 'native driver failed; ownership retained'
            raise
        finally:
            self._in_wait = False
            self._progress_origin = 'tick'

    def idle_wait_permit(self):
        """Held-g L3, driver side: an idle-wait arm's open auto-owned chunk owes a reply.

        Everything advance() could do this step is done: no control, route,
        return, stop or startup exchange is pending, the engine is not paused,
        and a native reply is owed (the next advance needs it). The engine
        side (executor_hooks._stock_step_idle) proves a stock step would
        schedule and send nothing.
        """
        progress = self.progress
        return (progress is not None and progress.idle == 'wait' and progress.recording
                and self.error is None and self.stop_kind is None
                and self._profile_failure is None and self._control is None
                and not self._control_orphans and self._route_exchange() is None
                and self._return_pending is None and self._bootstrapped
                and self._split_ready and self.tx is not None and self._progress_owned()
                and bool(self._inflight) and not self._engine_paused())

    def idle_wait(self):
        """Held-g L3: block at most IDLE_WAIT_S for any reply to land; never polls."""
        progress = self.progress
        transport = getattr(self.executor, '_flash_s5_transport', None)
        landed = transport.wait_landed(NativeProgress.IDLE_WAIT_S)
        progress.counters['idle_steps'] += 1
        progress.counters['idle_timeouts'] += int(not landed)
        return landed

    def _progress_control(self, command):
        """Typed per-arm mode switch; only installed burst, only paused idle FULL."""
        progress = self.progress
        if progress is None or progress.installed != 'burst':
            raise DriverError('native-control-unknown-action')
        if self.error is not None:
            raise DriverError(self.error)
        if set(command) - {'action', 'mode', 'engine_epoch', 'idle', 'replies'}:
            raise DriverError('native-progress-command-fields-differ')
        epoch = command.get('engine_epoch')
        if epoch is not None and (type(epoch) is not int or epoch != self.engine_epoch):
            raise DriverError('native-control-engine-epoch-differs')
        mode = command.get('mode')
        if mode not in NativeProgress.MODES:
            raise DriverError('native-progress-mode-unknown')
        # Held-g: an arm that names no idle selection gets held-f's step pacing.
        idle = command.get('idle', 'step')
        if idle not in NativeProgress.IDLE:
            raise DriverError('native-progress-idle-unknown')
        replies = command.get('replies', 'full')
        if replies not in NativeProgress.REPLIES:
            raise DriverError('native-progress-replies-unknown')
        policy = getattr(self, 'auto_arm', None)
        if (not self._engine_paused() or self._split_ready or self.tx is not None
                or self.controller.active or self._inflight or self._control is not None
                or self._route_exchange() is not None or self._return_pending is not None
                or self._retained is not None or self._admission_intent is not None
                or policy is None or policy.active or progress.recording):
            raise DriverError('native-progress-requires-paused-idle-full')
        previous = progress.mode
        progress.mode = mode
        progress.counters['mode_changes'] += int(previous != mode)
        previous_idle, progress.idle = progress.idle, idle
        previous_replies, progress.replies = progress.replies, replies
        now = int(self._monotonic_ns())
        return {'schema': 1, 'action': 'native_progress', 'mode': mode, 'previous': previous,
                'idle': idle, 'previous_idle': previous_idle,
                'replies': replies, 'previous_replies': previous_replies,
                'installed': progress.installed, 'engine_epoch': self.engine_epoch,
                'route': 'full', 'route_generation': self.route_generation,
                'observed_clock_ns': now, 'clock': self._clock_record(now)}

    def _progress_owned(self):
        """True only while the auto-arm policy owns the open native chunk."""
        policy = getattr(self, 'auto_arm', None)
        intent = self._admission_intent
        return (policy is not None and not policy.manual_owned and policy.phase == 'native'
                and type(intent) is dict and intent.get('origin') == 'auto-arm')

    def _open_replies(self):
        """Held-g L4: the opening chunk's reply format; lean only when the arm
        selected it and the auto-arm policy owns the chunk."""
        if (self.progress is not None and self.progress.replies == 'lean'
                and self._progress_owned()):
            return 'lean'
        return 'full'

    def _progress_decode_live(self):
        try:
            return sum(1 for r in self.scheduler.running
                       if r.num_computed_tokens >= r.num_prompt_tokens)
        except Exception:
            return None

    def install(self):
        engine, scheduler = self.engine, self.scheduler
        if engine is None:
            raise DriverError('driver needs the engine object to install on')
        if getattr(engine, '_flash_s5_driver', None) is not None:
            raise DriverError('engine already has an E3 segment driver')
        if getattr(scheduler, '_flash_s5_driver', None) is not None:
            raise DriverError('scheduler already has an E3 segment driver')
        engine._flash_s5_driver = self
        scheduler._flash_s5_driver = self
        return self

    @property
    def live(self):
        """Owned chunk/route work; a retained request row is not a chunk."""
        return (self.tx is not None or self.controller.active
                or self._return_pending is not None
                or self._transition is not None
                or self._bootstrap_command is not None
                or self._profile_failure is not None)

    @property
    def active(self):
        # A retained worker request row is real owned work between chunks even
        # though no chunk, GPU replay or paused ordinary batch is outstanding.
        return self.live or self._retained is not None

    def retained_request(self):
        """The exact transaction whose worker request row is still owned."""
        row = self._retained
        return None if row is None else row['transaction']

    def retained_row(self):
        """Bounded status row for the retained metadata; no request UID."""
        row = self._retained
        if row is None:
            return None
        transaction = row['transaction']
        return {
            'transaction': transaction.txn_seq,
            'request_generation': transaction.request_generation,
            'chunk_start': transaction.chunk_start,
            'chunk_end': transaction.chunk_end,
            'state_epoch': row['state_epoch'],
            'ranks': list(self.ranks),
            'completed_sequences': [row['completed_sequences'][rank]
                                    for rank in self.ranks],
            'cancel_pending': self._retained_cancel_request is not None,
            # Replied ranks are not closed ranks: a partial reply (or a rank
            # that still owns the row) must never read as a clean closure.
            'replied_ranks': sorted(self._retained_cancel_acks),
            'closed_ranks': sorted(rank for rank, ack
                                   in self._retained_cancel_acks.items()
                                   if ack.closed),
            'closure': ('closed' if (len(self._retained_cancel_acks) == len(self.ranks)
                                     and all(ack.closed for ack
                                             in self._retained_cancel_acks.values()))
                        else 'open'),
        }

    def native_work_identity(self):
        transaction = self.tx if self.tx is not None else self.retained_request()
        if transaction is None:
            return None
        return {'transaction': transaction.txn_seq,
                'request_uid_sha256': hashlib.sha256(transaction.request_uid.encode()).hexdigest(),
                'request_generation': transaction.request_generation,
                'chunk_start': transaction.chunk_start, 'chunk_end': transaction.chunk_end}

    # -- schedule seam ----------------------------------------------------
    def park(self, scheduler_output):
        """Pre-schedule admission owns parking; a step marker is refused."""
        if getattr(scheduler_output, 'flash_s5_prefill_admission', None) is not None:
            raise DriverError('the legacy post-schedule admission marker is gone')
        return False

    # -- bounded transport ------------------------------------------------
    def _bounded(self):
        sender = getattr(self.executor, '_flash_s5_bounded_send', None)
        if not callable(sender):
            raise DriverError('native bounded command sender is not installed')
        return sender

    def _submit(self, kind, envelope, ranks, *, timeout_s=None):
        with self.control_path.span('prepare_ns'):
            return self._submit_observed(kind, envelope, ranks, timeout_s=timeout_s)

    def _submit_observed(self, kind, envelope, ranks, *, timeout_s=None):
        transport = getattr(self.executor, '_flash_s5_transport', None)
        if transport is None:
            raise DriverError('owned reply transport is not installed')
        sender = self._bounded()
        check = getattr(sender, 'check', None)
        progress = self.progress if self.progress is not None and self.progress.recording else None
        direct = progress is not None and progress.burst() and kind in progress.DIRECT_KINDS
        if direct and envelope[0] not in PROGRESS_DIRECT_METHODS:
            raise DriverError('native-progress-direct-method-not-eligible')
        if progress is not None:
            submitted_ns = time.monotonic_ns()
            # Same pickled size as the real meta: the size precheck stays exact.
            checked = NativeProgress.with_meta(envelope, bytes(PROGRESS_META_BYTES))
        else:
            checked = envelope
        def precheck():
            with self.control_path.span('serialize_ns'):
                return check(checked)
        def send(_sequence):
            with self.control_path.span('serialize_ns'):
                if progress is None:
                    return sender(envelope)
                return sender(NativeProgress.with_meta(envelope, progress.meta(_sequence, direct)))
        extra = {'direct': True} if direct else {}
        sequence = transport.submit(kind, send,
                                    ranks=tuple(ranks),
                                    timeout_s=(self.reply_timeout_s if timeout_s is None
                                               else timeout_s), aggregate=None,
                                    precheck=precheck if callable(check) else None, **extra)
        self.control_path.values['ops_count'] += 1
        if progress is not None:
            segment = op_seq = stage = layer = None
            if kind == 'segment' and self._command is not None:
                segment = self._command.kind.value
                op_seq, stage = self._command.op_seq, self._command.stage
                layer = self._command.descriptor.layer_start
            elif kind == 'profile':
                query = envelope[1][0]
                op_seq, stage, layer = query.boundary_op_seq, query.stage, query.layer
            progress.native[sequence] = [sequence, kind, segment, op_seq, stage, layer,
                                         self._progress_origin, submitted_ns, time.monotonic_ns(),
                                         None, None, None, None, None, None, int(direct)]
            progress.counters['commands'] += 1
            progress.counters['direct_commands'] += int(direct)
        if kind == 'profile':
            query = envelope[1][0]
            self.qsa_control.add(query.stage, query.layer, profile_polls=1,
                                 profile_retry_polls=int(self._profile_poll_count > 0))
            self._profile_poll_count += 1
        elif kind == 'segment' and self._command.kind is SegmentKind.LAYER:
            tail_command = (self._tail_submission is not None
                            and self._tail_submission.command == self._command)
            self.qsa_control.add(self._command.stage, self._command.descriptor.layer_start,
                layer_commands=1,
                bucket_commands=int(self._command.profile is not None and not tail_command),
                tail_commands=int(tail_command))
        self._inflight[sequence] = kind
        return sequence

    def _drain(self):
        with self.control_path.span('poll_ns'):
            return self._drain_observed()

    def _drain_observed(self):
        transport = getattr(self.executor, '_flash_s5_transport', None)
        if transport is None:
            return 0
        transport.poll(self.poll_budget)
        applied = 0
        progress = self.progress
        for sequence, kind in list(self._inflight.items()):
            call = transport._calls.get(sequence)
            if call is None or not call.ready:
                continue
            row = None if progress is None else progress.native.pop(sequence, None)
            registered = None if row is None else getattr(call, 'registered_s', None)
            value, _lag = transport.take(sequence)
            del self._inflight[sequence]
            if row is not None:
                for index, rank in enumerate(self.ranks[:4]):
                    lag = _lag.get(rank) if type(_lag) is dict else None
                    if type(lag) in (int, float) and type(registered) is float:
                        row[PROGRESS_N['t_recv0'] + index] = int((registered + lag) * 1e9)
                row[PROGRESS_N['outcome']] = _progress_outcome(kind, value)
                progress.applying = row
            with self.control_path.span('validate_ns'):
                self._apply(kind, value, sequence)
            applied += 1
            if row is not None:
                progress.applied(row)
        return applied

    def _apply(self, kind, values, sequence=None):
        if kind == 'bootstrap':
            self._apply_bootstrap(values)
        elif kind == 'open':
            self._apply_open(values)
        elif kind == 'bind':
            self._apply_bind(values)
        elif kind == 'segment':
            self._apply_segments(values)
        elif kind == 'profile':
            self._apply_profile(values)
        elif kind == 'poll':
            self._apply_poll(values)
        elif kind == 'receipt':
            self._apply_receipt(values)
        elif kind == 'cancel':
            self._apply_cancel(values)
        elif kind == 'release':
            self._apply_release(values)
        elif kind == 'retained_cancel':
            self._apply_retained_cancel(values)
        elif kind == 'native_overlap':
            self._apply_native_overlap(values, sequence)
        elif kind == 'transition':
            acks = tuple(values)
            if len(acks) != len(self.ranks):
                raise DriverError('transition reply does not cover every rank')
            command = self._transition
            if command is None:
                raise DriverError('no transition is outstanding')
            if {getattr(a, 'worker_rank', None) for a in acks} != set(self.ranks):
                raise DriverError('transition reply ranks are duplicated or missing')
            for ack in acks:
                if type(ack) not in (TransitionAck, TransitionPending) or not ack.matches(command):
                    raise DriverError('transition acknowledgment is stale or mistyped')
                if type(ack) is TransitionPending:
                    continue
                if (ack.decode_replay_watermark
                        < command.decode_replay_watermarks[ack.worker_rank]):
                    raise DriverError('transition replay census went backwards')
            if any(type(ack) is TransitionPending for ack in acks):
                self._transition_retry = True
                return
            # The worker reports its own queried census; it is not an echo.
            for ack in acks:
                self._decode_watermarks[ack.worker_rank] = ack.decode_replay_watermark
            applied = {ack.applied_route_generation for ack in acks}
            if applied != {command.route_generation}:
                raise DriverError('transition acknowledgments disagree on the route generation')
            self.route_generation = acks[0].applied_route_generation
            self._split_ready = command.target_bank == ROUTE_BANK_SPLIT
            if command.target_bank == ROUTE_BANK_FULL:
                self._return_pending = None
                self._control_route = None
                if self._admission_intent is not None:
                    # A full return is the end of the native phase: the intent
                    # may not survive into stock sampling as a stale arm.
                    closed = self._admission_intent
                    self._admission_intent = None
                    self._record('admission_disarmed', phase=closed['phase'],
                                 reason='route-returned-full',
                                 route_generation=self.route_generation)
            self._transition = None
            self._transition_retry = False
            self._transition_deadline = None
            self._last_transition_acks = {ack.worker_rank: ack for ack in acks}
            entry = self._record(
                'transition',
                target_bank=command.target_bank,
                applied_route_generation=self.route_generation,
                acks=[{'worker_rank': ack.worker_rank,
                       'applied_route_generation': ack.applied_route_generation,
                       'applied_bank': ack.applied_bank,
                       'decode_replay_watermark': ack.decode_replay_watermark,
                       'drained_streams': ack.drained_streams}
                      for ack in acks])
            self._last_transition_observed_ns = entry['observed_ns']
            self._control_notice('transition')
        else:
            raise DriverError('unknown wire reply kind')

    def _apply_bootstrap(self, values):
        acks = tuple(values)
        command = self._bootstrap_command
        if command is None:
            raise DriverError('no bootstrap is outstanding')
        if len(acks) != len(self.ranks):
            raise DriverError('bootstrap reply does not cover every rank')
        if {getattr(a, 'worker_rank', None) for a in acks} != set(self.ranks):
            raise DriverError('bootstrap reply ranks are duplicated or missing')
        for ack in acks:
            if type(ack) is not BootstrapAck or not ack.matches(command):
                raise DriverError('bootstrap acknowledgment is stale or mistyped')
            if not ack.bound:
                raise DriverError('worker refused the engine owner bootstrap')
            if ack.lane.role is not LaneRole.DECODE:
                raise DriverError('bootstrap acknowledged a non-decode storage lane')
            self._storage_epochs[ack.worker_rank] = ack.lane.storage_epoch
            self._decode_watermarks[ack.worker_rank] = ack.decode_replay_watermark
        self._bootstrapped = True
        self._bootstrap_command = None
        self._last_bootstrap_acks = {ack.worker_rank: ack for ack in acks}
        entry = self._record(
            'bootstrap',
            ranks=[ack.worker_rank for ack in acks],
            lane_storage_epochs=[ack.lane.storage_epoch for ack in acks],
            decode_replay_watermarks=[ack.decode_replay_watermark for ack in acks],
            incarnations=[ack.incarnation for ack in acks])
        self._last_bootstrap_observed_ns = entry['observed_ns']
        self._control_notice('bootstrap')

    def _apply_open(self, values):
        acks = tuple(values)
        if len(acks) != len(self.ranks):
            raise DriverError('open reply does not cover every rank')
        request = self.open_request
        seen = set()
        for ack in acks:
            if type(ack) is not OpenAck or not ack.matches(request):
                raise DriverError('open acknowledgment is stale or mistyped')
            if not ack.accepted:
                raise DriverError('worker refused the chunk open')
            if ack.lane.worker_rank in seen:
                raise DriverError('duplicate open acknowledgment')
            seen.add(ack.lane.worker_rank)
            self.lanes[ack.lane.worker_rank] = ack.lane
            prior_lane, last = self._last_lane_sequences.get(
                ack.lane.worker_rank, (None, -1))
            self.trackers[ack.lane.worker_rank] = OpSeqTracker(
                last if prior_lane == ack.lane else -1)
            if ack.pp_stages != 2 or ack.lane.role.value != 'prefill':
                raise DriverError('open acknowledged the wrong lane or PP topology')
            if ack.stage != self.stage_of[ack.lane.worker_rank]:
                raise DriverError('worker stage disagrees with the split')
        if self._retained is not None:
            row = self._retained
            if (not row['transaction'].same_request(self.tx)
                    or row['transaction'] == self.tx):
                # The worker open moved a *different* request/generation onto
                # the retained row.  Refuse instead of guessing whose row it is.
                raise DriverError('open does not continue the retained request row')
            self._record('request_row_continued', transaction=self.tx.txn_seq,
                         previous_transaction=row['transaction'].txn_seq,
                         request_generation=self.tx.request_generation,
                         state_epoch=self.admission.state_epoch,
                         previous_state_epoch=row['state_epoch'],
                         chunk_start=self.tx.chunk_start,
                         chunk_end=self.tx.chunk_end,
                         ranks=list(self.ranks))
            # Every rank accepted this chunk's open; the row now belongs to the
            # active transaction and is released/closed by the normal paths.
            self._retained = None
            self._retained_cancel_request = None
            self._retained_cancel_acks = {}
        self.plan = self._build_plan()
        # The scheduler hold may only be validated for lanes that actually
        # received this admitted frame; bind their exact identities now.
        self.controller.note_source_owners(self.tx, self.lanes, self.layer_count)
        self._bindings = {}
        self._bind_index = 0
        self._pending_binding = None
        self.phase = 'bind'
        self._record('chunk_open', transaction=self.tx.txn_seq,
                     request=self.tx.request_uid,
                     request_generation=self.tx.request_generation,
                     chunk_start=self.tx.chunk_start,
                     chunk_end=self.tx.chunk_end,
                     state_epoch=acks[0].state_epoch,
                     committed_end=self.admission.committed_end,
                     prompt_end=self.admission.prompt_end,
                     remaining_tokens=self.admission.remaining_tokens,
                     terminal=self.admission.terminal,
                     stages=[self.stage_of[rank] for rank in sorted(self.lanes)],
                     lane_storage_epochs=[self.lanes[rank].storage_epoch
                                          for rank in sorted(self.lanes)])

    def _apply_bind(self, values):
        acks = tuple(values)
        binding = self._pending_binding
        if binding is None or len(acks) != 1:
            raise DriverError('peer binding reply does not match the targeted sender')
        ack = acks[0]
        if type(ack) is not PeerBindingAck or not ack.matches(binding):
            raise DriverError('peer binding acknowledgment is stale or mistyped')
        if not ack.bound:
            raise DriverError('sender refused the peer binding')
        if ack.receiver_lane != self.lanes[self.pp_peers[binding.sender_rank]]:
            raise DriverError('sender bound an unopened receiver lane')
        self._bindings[binding.sender_rank] = ack
        self._pending_binding = None
        self._bind_index += 1
        if self._bind_index == len(self.stage_ranks[0]):
            self.phase = 'segments'

    def _next_binding(self):
        while self._bind_index < len(self.stage_ranks[0]):
            sender = self.stage_ranks[0][self._bind_index]
            if sender in self._bindings:
                self._bind_index += 1
                continue
            receiver = self.pp_peers[sender]
            binding = PeerBinding(self.tx, sender, self.lanes[receiver],
                                  self.route_generation)
            self._pending_binding = binding
            self._submit('bind', ('flash_s5_bind', (binding,), {}, sender),
                         (sender,))
            return True
        self.phase = 'segments'
        return False

    def _build_plan(self):
        """Bounded segment plan: stage 0 then the explicit PP transfer."""
        seq = self.transfer_seq
        self.transfer_seq += 1
        plan = [dict(kind=SegmentKind.PREPARE, stage=0),
                dict(kind=SegmentKind.PREPARE, stage=1),
                dict(kind=SegmentKind.PREFIX, stage=0)]
        for stage in (0, 1):
            if stage == 1:
                plan.append(dict(kind=SegmentKind.PREFIX, stage=1))
            for layer in range(*self.stage_layers[stage]):
                plan.append(dict(kind=SegmentKind.LAYER, stage=stage, layer=layer))
            plan.append(dict(kind=SegmentKind.SUFFIX, stage=stage))
            if stage == 0:
                plan.append(dict(kind=SegmentKind.PP_SEND, stage=0, seq=seq))
                plan.append(dict(kind=SegmentKind.PP_RECEIVE, stage=1, seq=seq))
            plan.append(dict(kind=SegmentKind.STATE_FINALIZE, stage=stage))
        return plan

    def _apply_segments(self, values):
        accepted = tuple(values)
        if (len(accepted) != len(self.ranks)
                or {getattr(a, 'worker_rank', None) for a in accepted} != set(self.ranks)):
            raise DriverError('segment reply does not cover every rank')
        targets = self._targets
        pending = False
        reports = []
        for item in accepted:
            if type(item) is not SegmentAcceptance:
                raise DriverError('segment reply is not a typed acceptance')
            if item.worker_rank not in self.ranks:
                raise DriverError('acceptance comes from an unknown rank')
            if (item.transaction != self.tx or item.kind is not self._command.kind
                    or item.stage != self._command.stage or item.op_seq != self._command.op_seq):
                raise DriverError('reply identity differs from the outstanding command')
            if item.report is not None:
                reports.append(item.report)
            elif self._replies == 'lean' and self.progress is not None:
                # Held-g L4: nothing changed on that rank since its last report.
                self.progress.counters['lean_tokens'] += 1
            if item.worker_rank not in targets:
                if not item.skipped:
                    raise DriverError('a non-target lane executed the segment')
                continue
            if item.skipped:
                raise DriverError('a target lane skipped the segment')
            if item.pending:
                pending = True
                continue
            if not item.accepted:
                raise DriverError('worker refused the segment')
            if not item.matches(self._command):
                raise DriverError('acceptance does not match the command')
        # Record each accepted target even if a peer is pending: a retry
        # must not forget work already submitted on the first rank.
        descriptor = self._command.descriptor
        accepted_ranks = {item.worker_rank for item in accepted if item.accepted}
        for rank in accepted_ranks:
            tracker = self.trackers[rank]
            first = tracker.accepted(self._command.op_seq, descriptor)
            if first:
                self.tickets[rank] = OperationTicket(
                    self.tx, rank, self.lanes[rank], self._command.op_seq,
                    _SEGMENT_ROLE[descriptor.kind])
                self.accepted_tickets[(rank, self._command.op_seq)] = self.tickets[rank]
                self._accepted_descriptors[(rank, self._command.op_seq)] = descriptor
                self._last_lane_sequences[rank] = (self.lanes[rank], self._command.op_seq)
                if descriptor.kind is SegmentKind.PP_SEND:
                    self.pp_ledgers[rank].note_sent(self.tx, descriptor.seq, rank)
        if descriptor.kind in (SegmentKind.PP_SEND, SegmentKind.PP_RECEIVE):
            obligation = self._pp_submission
            if (obligation is None or obligation.send.transaction != self.tx
                    or obligation.route_generation != self.route_generation):
                raise DriverError('PP acceptance lacks its bound submission obligation')
            expected = (obligation.send if descriptor.kind is SegmentKind.PP_SEND
                        else obligation.receive)
            if expected != self._command:
                raise DriverError('PP acceptance differs from its retained command')
            seen = (obligation.send_accepted if descriptor.kind is SegmentKind.PP_SEND
                    else obligation.receive_accepted)
            seen.update(accepted_ranks)
        tail_submission = self._tail_submission
        if tail_submission is not None and tail_submission.command == self._command:
            if (tail_submission.route_generation != self.route_generation
                    or self._command.transaction != self.tx):
                raise DriverError('TP tail acceptance differs from its route/transaction')
            tail_submission.accepted.update(accepted_ranks)
        if pending:
            if self._command.kind is SegmentKind.LAYER:
                tail_command = (tail_submission is not None
                                and tail_submission.command == self._command)
                self.qsa_control.add(self._command.stage, self._command.descriptor.layer_start,
                    layer_pending_rounds=1,
                    bucket_pending_rounds=int(self._command.profile is not None and not tail_command),
                    tail_pending_rounds=int(tail_command))
            self._retry = self._command
            for report in reports:
                self._ingest_report(report)
            return
        paused = {item.paused for item in accepted if item.accepted}
        cursors = {_continuation(descriptor.kind, item.cursor)
                   for item in accepted if item.accepted}
        if len(paused) != 1 or len(cursors) != 1:
            raise DriverError('TP workers disagree on the layer continuation'
                              + _agreement_detail(self._command, targets, accepted))
        boundaries = {item.profile_boundary for item in accepted if item.accepted}
        if len(boundaries) != 1:
            raise DriverError('TP workers disagree on the QSA submission boundary'
                              + _agreement_detail(self._command, targets, accepted))
        tail_boundaries = {item.tail_boundary for item in accepted if item.accepted}
        if len(tail_boundaries) != 1:
            raise DriverError('TP workers disagree on the QSA tail boundary'
                              + _agreement_detail(self._command, targets, accepted))
        representative = targets[0]
        self.last_ticket[descriptor.stage] = self.tickets[representative]
        if descriptor.kind is SegmentKind.PREPARE and self.bound_ticket is None:
            self.controller.bind(self.tickets[representative])
            self.bound_ticket = self.tickets[representative]
        if descriptor.kind is SegmentKind.PP_SEND:
            self.controller.note_pp_send(self.tx, descriptor.seq)
            self.pp_send_ticket = self.tickets[representative]
        if descriptor.kind is SegmentKind.PP_RECEIVE:
            self.receiver_ticket = self.tickets[targets[0]]
        self._retry = None
        if paused != {True}:
            self.index += 1
        if self.index == len(self.plan):
            self.phase = 'finalize'
        if self._tx_fused:
            self._fused_agree(descriptor, targets, accepted)
        self._record('segment', transaction=self.tx.txn_seq,
                     op_seq=self._command.op_seq,
                     # Never ``kind``: that field carries the journal entry
                     # type the evidence builder keys on.
                     segment_kind=getattr(descriptor.kind, 'value',
                                          str(descriptor.kind)),
                     stage=descriptor.stage,
                     layer_start=descriptor.layer_start,
                     layer_end=descriptor.layer_end,
                     **({'layer_group': descriptor.layer_group} if self._tx_group != 1 else {}),
                     targets=list(targets),
                     # The route is checked by the engine at this reply; it is
                     # descriptive of where the operation actually ran, never a
                     # substitute for the queried completion below.
                     transport=('native-split' if self._split_ready else 'serialized'))
        # Completions piggybacked on this reply are host-queried facts too.
        # Publication stays in the drive path, never inside acceptance.
        for report in reports:
            self._ingest_report(report)

        if descriptor.kind is SegmentKind.LAYER and self._command.profile is not None:
            tail_command = (tail_submission is not None
                            and tail_submission.command == self._command)
            self.qsa_control.add(descriptor.stage, descriptor.layer_start,
                                 bucket_accepted_commands=int(not tail_command),
                                 tail_accepted_commands=int(tail_command))
        if boundaries == {True}:
            if self._command.profile is not None:
                raise DriverError('QSA profile boundary repeated after its grant')
            self._profile_query = QsaProfileQuery(
                self.tx, self.admission.state_epoch, descriptor.stage,
                descriptor.layer_start, self._command.op_seq, tuple(targets))
            self.qsa_control.add(descriptor.stage, descriptor.layer_start, boundary_commands=1)
            self._profile_observed = {}
            self._profile_poll_count = 0
            self._profile_deadline = self.clock() + self.reply_timeout_s
            self.phase = 'profile'
        if tail_boundaries == {True}:
            if descriptor.kind is not SegmentKind.LAYER or paused != {True}:
                raise DriverError('QSA tail boundary is not a paused layer')
            if self._tail_boundary is not None:
                raise DriverError('QSA tail boundary repeated before its tail')
            self._tail_boundary = (self.tx, descriptor.stage,
                                   descriptor.layer_start, self._command.op_seq,
                                   self.route_generation)
            self.qsa_control.add(descriptor.stage, descriptor.layer_start,
                                 tail_boundaries=1)
            self._record('qsa_tail_boundary', transaction=self.tx.txn_seq,
                         stage=descriptor.stage, layer=descriptor.layer_start,
                         op_seq=self._command.op_seq, ranks=list(targets))
        if tail_submission is not None and tail_submission.command == self._command:
            if tail_submission.accepted != set(targets) or paused != {False}:
                raise DriverError('TP tail did not finish on every active-stage peer')
            self._tail_boundary = None
            self._tail_submission = None

    def _apply_profile(self, values):
        query = self._profile_query
        if (query is None or query.transaction != self.tx
                or query.ranks != self.stage_ranks[query.stage]):
            raise DriverError('QSA profile reply has no current active-stage query')
        verdict, grant = qsa_profile_agreement(query, tuple(values), self.ranks)
        for item in values:
            if item.worker_rank in query.ranks:
                self.qsa_control.add(query.stage, query.layer,
                                     profile_worker_host_ns=item.worker_host_ns)
        for item in values:
            prior = self._profile_observed.get(item.worker_rank)
            if prior is not None and prior.status == 'ready' and prior != item:
                raise DriverError('queried QSA schedule changed within its boundary')
            self._profile_observed[item.worker_rank] = item
        # The deadline covers the whole agreement, including a late ready
        # response. Each query gets only the remaining budget in _drive.
        if self.clock() >= self._profile_deadline:
            self._refuse_profile('native-schedule-profile-timeout')
            return
        if verdict == 'pending':
            self.qsa_control.add(query.stage, query.layer, profile_pending_rounds=1)
            return
        if verdict == 'unsupported-native-schedule-profile':
            self._profile_mismatches += 1
            self._refuse_profile(verdict)
            return
        self.qsa_control.add(query.stage, query.layer,
            agreed_buckets=len(grant.shape.bucket_waves),
            agreed_copy_waves=sum(grant.shape.bucket_waves))
        self._profile_grants[(query.stage, query.layer)] = grant
        self._profile_query = None
        self._profile_observed = {}
        self._profile_deadline = None
        self._profile_poll_count = 0
        self.phase = 'segments'

    def _refuse_profile(self, reason):
        # No bucket has been submitted. The accepted producer/selection and
        # every source/PP owner still go through the existing typed stop path.
        self._profile_failure = (self.tx, reason)
        if reason == 'native-schedule-profile-timeout':
            self._profile_timeouts += 1
            field = 'profile_timeouts'
        else:
            field = 'profile_mismatches'
        self.qsa_control.add(self._profile_query.stage, self._profile_query.layer,
                             **{field: 1})
        self.stop_kind = 'abort'
        self._return_pending = 'unsupported-profile'
        entry = self._record('schedule_profile_refused', transaction=self.tx.txn_seq,
                     stage=self._profile_query.stage,
                     layer=self._profile_query.layer, reason=reason,
                     mismatch_count=self._profile_mismatches, timeout_count=self._profile_timeouts,
                     peer_profiles=[{
                         'rank': rank,
                         'status': getattr(self._profile_observed.get(rank), 'status', 'unobserved'),
                         'route': (item.shape.route if item is not None and item.shape is not None else None),
                         'bucket_count': (len(item.shape.bucket_waves) if item is not None and item.shape is not None else None),
                         'bucket_waves': (list(item.shape.bucket_waves) if item is not None and item.shape is not None else None),
                         'selected_count': getattr(item, 'selected_count', None),
                         'worker_host_ns': getattr(item, 'worker_host_ns', None),
                     } for rank in self._profile_query.ranks
                       for item in (self._profile_observed.get(rank),)])
        self._profile_refusal_count += 1
        if self._profile_refusal_first is None:
            self._profile_refusal_first = copy.deepcopy(entry)
        self._profile_refusal_latest = copy.deepcopy(entry)

    def _profile_refusal_receipt(self):
        return {
            'schema': 's5-schedule-profile-refusal/1',
            'count': self._profile_refusal_count,
            'first': copy.deepcopy(self._profile_refusal_first),
            'latest': copy.deepcopy(self._profile_refusal_latest),
            'basis': 'engine-lifetime first/latest typed refusals; independent of the recent-event journal',
        }

    def _ingest_report(self, report):
        if type(report) is LeanCompletionReport:
            return self._ingest_lean(report)
        if type(report) is not WorkerCompletionReport:
            raise DriverError('completion report is mistyped')
        if report.transaction != self.tx:
            raise DriverError('completion report belongs to another transaction')
        if report.worker_rank not in self.lanes:
            raise DriverError('report rank has no opened lane')
        for op in report.completed_op_seqs + report.pending_op_seqs:
            if (report.worker_rank, op) not in self.accepted_tickets:
                raise DriverError('report names an unaccepted operation')
        for proof in report.proofs:
            key = (report.worker_rank, proof.ticket.op_seq)
            if proof.ticket != self.accepted_tickets.get(key):
                raise DriverError('proof differs from the issued ticket or lane epoch')
            kind = self._accepted_descriptors[key].kind
            allowed = {CompletionKind.GPU_EVENT}
            if kind is SegmentKind.STATE_FINALIZE:
                allowed.add(CompletionKind.STATE_WRITE)
            if kind is SegmentKind.PP_RECEIVE:
                allowed.add(CompletionKind.PP_CONSUMER)
            if proof.kind not in allowed:
                raise DriverError('proof role is impossible for the issued operation')
        self.controller.note_report(report)
        self.reports[report.worker_rank] = report
        if self._replies == 'lean':
            # A full report (a cancel ack) ends this rank's lean stream.
            self._lean_streams.setdefault(report.worker_rank, LeanStream()).closed = True
        self._record('report', transaction=report.transaction.txn_seq,
                     worker_rank=report.worker_rank,
                     completed_sequence=report.completed_sequence,
                     completed_op_seqs=list(report.completed_op_seqs),
                     pending_op_seqs=list(report.pending_op_seqs),
                     failed_code=report.failed_code,
                     # The worker's own host-queried proof; a submission counter
                     # is never consulted here.
                     state_written=report.state_written(),
                     proofs=[{'kind': getattr(proof.kind, 'value', str(proof.kind)),
                              'op_seq': proof.ticket.op_seq,
                              'completion_seq': proof.completion_seq}
                             for proof in report.proofs])
        self._collect_pp_proofs(report)

    def _ingest_lean(self, report):
        """Held-g L4: check one rank's delta in O(delta) and extend its accumulated report.

        The issued-ticket checks are the full report's, over the new items; the
        stream checks the delta against everything applied before. Every
        consumer of ``self.reports`` and the controller reads the cumulative
        report the stream returns. The journal records the delta and its
        stream header; the census merges it into the same per-rank sets.
        """
        if self._replies != 'lean':
            raise DriverError('lean report in a full-reply transaction')
        if report.transaction != self.tx:
            raise DriverError('completion report belongs to another transaction')
        rank = report.worker_rank
        if rank not in self.lanes:
            raise DriverError('report rank has no opened lane')
        for op in report.completed_op_seqs + report.pending_op_seqs:
            if (rank, op) not in self.accepted_tickets:
                raise DriverError('report names an unaccepted operation')
        for proof in report.proofs:
            key = (rank, proof.ticket.op_seq)
            if proof.ticket != self.accepted_tickets.get(key):
                raise DriverError('proof differs from the issued ticket or lane epoch')
            kind = self._accepted_descriptors[key].kind
            allowed = {CompletionKind.GPU_EVENT}
            if kind is SegmentKind.STATE_FINALIZE:
                allowed.add(CompletionKind.STATE_WRITE)
            if kind is SegmentKind.PP_RECEIVE:
                allowed.add(CompletionKind.PP_CONSUMER)
            if proof.kind not in allowed:
                raise DriverError('proof role is impossible for the issued operation')
        stream = self._lean_streams.get(rank)
        if stream is None:
            stream = self._lean_streams[rank] = LeanStream()
        try:
            merged = stream.extend(report)
        except E3ContractError as exc:
            raise DriverError('lean report refused: ' + str(exc)) from exc
        self.controller.note_report(merged)
        self.reports[rank] = merged
        if self.progress is not None:
            self.progress.counters['lean_reports'] += 1
        completed, proofs, hosts, digest = stream.end()
        self._record('report', transaction=report.transaction.txn_seq,
                     worker_rank=rank,
                     completed_sequence=merged.completed_sequence,
                     completed_op_seqs=list(report.completed_op_seqs),
                     pending_op_seqs=list(report.pending_op_seqs),
                     failed_code=report.failed_code,
                     state_written=stream.state_written,
                     proofs=[{'kind': getattr(proof.kind, 'value', str(proof.kind)),
                              'op_seq': proof.ticket.op_seq,
                              'completion_seq': proof.completion_seq}
                             for proof in report.proofs],
                     # The vectors above are this delta; the census merges it.
                     lean={'base_completed': report.base_completed,
                           'base_proofs': report.base_proofs,
                           'base_hosts': report.base_hosts,
                           'completed': completed, 'proofs': proofs, 'hosts': hosts,
                           'digest': digest.hex()[:16]})
        self._collect_pp_proofs(report)

    def _apply_poll(self, values):
        reports = tuple(values)
        if (len(reports) != len(self.ranks)
                or {getattr(r, 'worker_rank', None) for r in reports} != set(self.ranks)):
            raise DriverError('poll reply does not cover every rank')
        for report in reports:
            self._ingest_report(report)
        if self.stop_kind is not None:
            return

    def _apply_release(self, values):
        acks = tuple(values)
        if (self.phase != 'release' or len(acks) != len(self.ranks)
                or any(type(a) is not ReleaseAck for a in acks)
                or {a.worker_rank for a in acks} != set(self.ranks)):
            raise DriverError('release lacks the complete typed all-rank reply')
        expected = dict(self.release_request.completed_sequences)
        terminal = self.release_request.terminal
        for ack in acks:
            if (ack.transaction != self.tx or ack.state_epoch != self.admission.state_epoch
                    or ack.completed_sequence != expected[ack.worker_rank]):
                raise DriverError('release acknowledgment differs from the final report')
            if not terminal and ack.request_closed:
                # A nonterminal release must not drop the request row: the next
                # chunk of this request needs it for LaneRequest.update().
                raise DriverError('nonterminal release closed the request row')
        for ack in acks:
            prior = self._release_acks.get(ack.worker_rank)
            if prior is not None and ack != prior:
                # A rank that already certified this chunk answers a broadcast
                # retry with the identical row, never a different one.
                raise DriverError('release retry contradicts an earlier accepted row')
            complete = ack.closed if terminal else ack.released
            if complete:
                self._release_acks[ack.worker_rank] = ack
        if self.stop_kind is not None:
            # A release that raced the stop certifies closure only; it is not a
            # publication. The cancellation path owns the terminal row.
            return
        if set(self._release_acks) != set(self.ranks):
            # Bounded retry: the not-closed workers kept ownership and the row.
            return
        for rank in self.ranks:
            self.controller.note_worker_release(self._release_acks[rank])
        # Retire the scheduler source hold: capture its own description and the
        # controller's retained publication row *before* ``commit`` clears the
        # parked chunk.  Both are queried facts of the real allocator/hold.
        hold = self.controller.source_hold(self.tx)
        hold_record = (hold.describe()
                       if hold is not None and callable(getattr(hold, 'describe', None))
                       else None)
        released = [{'worker_rank': rank,
                     'released': self._release_acks[rank].released,
                     'closed': self._release_acks[rank].closed,
                     'completed_sequence': self._release_acks[rank].completed_sequence}
                    for rank in self.ranks]
        transaction, admission = self.tx, self.admission
        self.controller.commit(self.tx, self._producer_proof())
        frontier = dict(getattr(self.controller, '_committed', {}).get(transaction, {}))
        self._record('chunk_commit', transaction=transaction.txn_seq,
                     request=transaction.request_uid,
                     chunk_start=transaction.chunk_start,
                     chunk_end=transaction.chunk_end,
                     committed_end=admission.committed_end,
                     state_epoch=admission.state_epoch,
                     terminal=admission.terminal,
                     worker_releases=released)
        self._record('source_hold', transaction=transaction.txn_seq,
                     description=hold_record,
                     retired=True)
        self._record('committed_frontier', transaction=transaction.txn_seq,
                     frontier={'committed_end': frontier.get('committed_end'),
                               'remaining': frontier.get('remaining'),
                               'state_epoch': frontier.get('state_epoch'),
                               'ranks': list(frontier.get('ranks') or ()),
                               'pp_seqs': list(frontier.get('pp_seqs') or ())}
                     if frontier else None)
        if terminal:
            # The request row closed with this chunk; no retained metadata may
            # survive into stock decode or a later admission of the same UID.
            self._retained = None
            self._retained_cancel_request = None
            self._retained_cancel_acks = {}
        else:
            # Every rank certified ``released=True, request_closed=False``
            # against the complete final report.  Bind that exact row here,
            # before ``_reset`` drops the chunk's own state.
            self._retained = {
                'transaction': transaction,
                'state_epoch': admission.state_epoch,
                'reports': dict(self.reports),
                'completed_sequences': {
                    rank: self._release_acks[rank].completed_sequence
                    for rank in self.ranks},
            }
            self._retained_cancel_request = None
            self._retained_cancel_acks = {}
            self._record('request_row_retained', transaction=transaction.txn_seq,
                         request_generation=transaction.request_generation,
                         state_epoch=admission.state_epoch,
                         chunk_start=transaction.chunk_start,
                         chunk_end=transaction.chunk_end,
                         ranks=list(self.ranks),
                         completed_sequences=[
                             self._retained['completed_sequences'][rank]
                             for rank in self.ranks])
        self._reset()
        if self._native43 is not None and transaction.chunk_end >= self._native43['stop_at']:
            self._native43['checkpoint'] = dict(transaction=transaction.txn_seq,
                exact=not terminal and transaction.chunk_end == self._native43['stop_at'],
                chunk_start=transaction.chunk_start, chunk_end=transaction.chunk_end,
                engine_observed_ns=self._monotonic_ns(), worker_releases=released,
                state_epoch=admission.state_epoch)
        if terminal:
            # The chunk left the final prompt token for stock decode.  Stock
            # work on the full bank may not start before every rank drained
            # the split route and acknowledged the return.
            self._note_terminal_route('handback')
        return

    def _send_release(self):
        self._release_attempts += 1
        if self._release_attempts > self.stop_budget:
            raise DriverError('release did not close worker ownership within the bound')
        # Closed rows from an earlier attempt are kept: the retry is still one
        # all-rank broadcast (every worker executes it and replies), and the
        # already-closed workers answer with their identical closed row.
        # Terminality is the scheduler's admission fact, never inferred from a
        # reply: the chunk that leaves the final prompt token for stock must
        # close the worker request row before it acknowledges.
        lean_streams = ()
        if self._replies == 'lean':
            lean_streams = tuple((r,) + self._lean_streams[r].end() for r in self.ranks)
        self.release_request = ReleaseRequest(
            self.tx, self.admission.state_epoch,
            tuple((r, self.reports[r].completed_sequence) for r in self.ranks),
            self.admission.terminal, lean_streams)
        self._submit('release', ('flash_s5_release', (self.release_request,), {}, None),
                     self.ranks)

    def _collect_pp_proofs(self, report):
        for proof in report.proofs:
            descriptor = self._accepted_descriptors[(report.worker_rank, proof.ticket.op_seq)]
            if proof.kind is CompletionKind.PP_CONSUMER:
                sender = next((s for s, peer in self.pp_peers.items()
                               if peer == report.worker_rank), None)
                if sender is None or descriptor.kind is not SegmentKind.PP_RECEIVE:
                    raise DriverError('PP consumer has no admitted sender peer')
                if self.pp_send_seq is None or descriptor.seq != self.pp_send_seq:
                    raise DriverError('consumer completion predates or differs from the sender submission')
                if descriptor.seq != self.pp_receive_seq:
                    raise DriverError('consumer completion names another transfer sequence')
                self.pp_ledgers[sender].note_consumer(self.tx, descriptor.seq, 1, proof)
                self._consumer_proofs[sender] = proof
            elif (proof.kind is CompletionKind.GPU_EVENT
                  and descriptor.kind is SegmentKind.PP_SEND):
                self.pp_ledgers[report.worker_rank].note_sender_complete(
                    self.tx, descriptor.seq, proof)
        for sender in self.pp_peers:
            self._maybe_queue_receipt(sender)

    def _maybe_queue_receipt(self, sender):
        """Queue the typed receipt only once both queried proofs are present."""
        if sender in self._receipt_acked or sender in self._receipts:
            return False
        seq = self.pp_send_seq
        if seq is None or sender not in self._consumer_proofs:
            return False
        ledger = self.pp_ledgers[sender]
        if not ledger.releasable(self.tx, seq):
            return False
        binding = self._bindings[sender]
        self._receipts[sender] = PPConsumerReceipt(
            0, seq, binding.route_generation, self._consumer_proofs[sender])
        return True

    def _next_receipt(self):
        """Submit at most one targeted receipt delivery; True when work was sent."""
        sender = self._receipt_inflight
        if sender is not None:
            return True
        for candidate in sorted(self._receipts):
            if candidate in self._receipt_acked:
                continue
            delivery = PPReceiptDelivery(self.tx, candidate,
                                         (self._receipts[candidate],))
            self._receipt_inflight = candidate
            self._receipt_last = delivery
            self._submit('receipt', ('flash_s5_receipt', (delivery,), {}, candidate),
                         (candidate,))
            return True
        return False

    def _apply_receipt(self, values):
        acks = tuple(values)
        delivery = self._receipt_last
        sender = self._receipt_inflight
        if delivery is None or sender is None or len(acks) != 1:
            raise DriverError('PP receipt reply has no outstanding delivery')
        ack = acks[0]
        if type(ack) is not PPReceiptAck or not ack.matches(delivery):
            raise DriverError('PP receipt acknowledgment is stale or mistyped')
        if not ack.accepted:
            raise DriverError('sender refused the PP consumer receipt')
        ledger = self.pp_ledgers[sender]
        seq = delivery.receipts[0].seq
        if sender not in self._receipt_acked:
            # The sender accepted the queried consumer proof: only now may its
            # local PP slot be released and only now is the transfer consumed.
            ledger.release_send(self.tx, seq)
            self._receipt_acked.add(sender)
            self._record('pp_receipt', transaction=self.tx.txn_seq,
                         sender_rank=sender, seq=seq,
                         completion_seq=delivery.receipts[0].proof.completion_seq,
                         accepted=True)
        self._receipt_inflight = None
        self._receipt_last = None
        if (not self._pp_noted
                and all(not item.outstanding(self.tx)
                        for item in self.pp_ledgers.values())):
            self.controller.note_pp_consumer(self.tx, seq, 1)
            self._pp_noted = True

    def _producer_proof(self):
        if self.bound_ticket is None:
            raise DriverError('no bound producer ticket')
        report = self.reports.get(self.bound_ticket.worker_rank)
        if report is None:
            raise DriverError('producer rank did not report completion')
        for proof in report.proofs:
            if proof.ticket == self.bound_ticket and proof.kind is CompletionKind.GPU_EVENT:
                return proof
        raise DriverError('producer ticket has no device completion proof')

    # -- step seam --------------------------------------------------------
    def advance(self, *, deliver_outputs=True):
        """One bounded tick; only a drained refusal emits a request ERROR.

        Successful model output continues to belong to the ordinary handback.
        """
        if self.error is not None:
            raise DriverError(self.error)
        try:
            self._drain()
            self.pump_control()
            if (self._native43 is not None and self._native43['checkpoint'] is not None
                    and not self._native43['cancel_started']):
                self._native43['cancel_started'] = True
                self.stop('cancel')
                # Keep this very tick out of schedule(): stop_kind is not
                # itself a stock-step bypass result. Idle ticks finish stop.
                return (None, False)
            if self._profile_failure is not None and deliver_outputs:
                output = self._finish_profile_failure()
                if output is not None:
                    return output
            if self.stop_kind is not None:
                self._drive_stop()
                return (None, False) if self._profile_failure is not None else None
            if self._drive_route():
                # A route command is outstanding: the worker is draining or
                # switching streams/banks, so no stock work may be launched
                # into the old route.  The engine skips this step instead.
                return (None, False)
            if self._native43 is not None and self._native43['checkpoint'] is not None:
                # Certified stop progresses above, but no chunk N+1 or stock
                # continuation may run while the diagnostic client inspects it.
                return (None, False)
            if self._split_hold_timeout():
                # native39: stock work held past the bound with no native work
                # in flight; the FULL return has started, skip this step.
                return (None, False)
            if self.tx is None:
                if not self.controller.active:
                    return None
                self._begin()
            self._drive()
            return None
        except Backpressure:
            # Ring full: no expectation was registered; retry on the next tick.
            return None
        except E3Pending:
            raise
        except BaseException:
            self.error = 'native driver failed; ownership retained'
            raise

    def _publish_profile_failure(self):
        """Publish once after certified cancellation, independently of route return.

        EngineCoreProc's existing output queue owns normal client delivery. The
        FULL-return guards still own scheduling; a delayed route ack must not
        keep an already closed failed request waiting for its first token.
        """
        if self._profile_failure is None:
            return False
        queue = getattr(self.engine, 'output_queue', None)
        if queue is None:
            return False  # Embedding EngineCore uses the step-result channel.
        if self.tx is not None or self.controller.active or self._retained is not None:
            return False
        transaction, reason = self._profile_failure
        request = self._profile_failed_request
        if request is None:
            return False
        if request.request_id != transaction.request_uid:
            raise DriverError('QSA refusal request identity changed')
        if not self._last_native_closure or not self._last_native_closure.get('complete'):
            raise DriverError('QSA refusal publication requires certified closure')
        from vllm.v1.request import RequestStatus
        from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
        packet = getattr(self, '_profile_failure_packet', None)
        if packet is None:
            self.scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ERROR)
            output = EngineCoreOutput(request_id=request.request_id, new_token_ids=[],
                                      finish_reason=FinishReason.ERROR, stop_reason=reason)
            packet = (request.client_index, EngineCoreOutputs(outputs=[output]))
            self._profile_failure_packet = packet
        try:
            queue.put_nowait(packet)
        except __import__('queue').Full:
            return False
        self._record('profile_failure_published', transaction=transaction.txn_seq,
                     reason=reason, channel='engine-output-queue',
                     full_return_pending=self._return_pending is not None)
        self._profile_failure = None
        self._profile_failed_request = None
        self._profile_failure_packet = None
        return True

    def _finish_profile_failure(self):
        """Embedding fallback; process mode publishes directly after closure."""
        if getattr(self.engine, 'output_queue', None) is not None:
            self._publish_profile_failure()
            return None
        if (self.tx is not None or self.controller.active or self._retained is not None
                or self._return_pending is not None or self._route_exchange() is not None
                or self._inflight or self.stop_kind is not None):
            return None
        transaction, reason = self._profile_failure
        request = self._profile_failed_request
        if request is None or request.request_id != transaction.request_uid:
            raise DriverError('QSA profile refusal lacks its retired request identity')
        from vllm.v1.request import RequestStatus
        from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
        self.scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ERROR)
        output = EngineCoreOutput(request_id=request.request_id, new_token_ids=[],
                                  finish_reason=FinishReason.ERROR, stop_reason=reason)
        result = {request.client_index: EngineCoreOutputs(outputs=[output])}
        self._profile_failure = None
        self._profile_failed_request = None
        return result, False

    # -- engine-owner bootstrap and route completion ----------------------
    def _route_exchange(self):
        """The route command whose reply is still owed, if any."""
        return self._bootstrap_command or self._transition

    def _drive_return(self):
        """One bounded step towards the drained all-rank full-bank return."""
        if self._route_exchange() is not None or self._inflight:
            return
        if self._retained is not None:
            # The full-bank return may not abandon a worker request row the
            # engine still owns.  Closure is driven by the stop/cancel path.
            raise DriverError('full route return requires the retained row to close')
        if not self._split_ready:
            # Already home (or never left); the return is satisfied.
            self._return_pending = None
            return
        try:
            self._issue_transition(ROUTE_BANK_FULL)
        except E3Pending:
            # A worker still owns an unqueried native replay: the bridge
            # refused the drained switch before mutating any route.  Keep the
            # handback pending and skip stock work for this tick; the next tick
            # retries with the workers' advertised census.
            self._return_pending = 'handback'

    def _issue_transition(self, target_bank):
        """Send one transition built from the census the workers acknowledged."""
        if set(self._storage_epochs) != set(self.ranks) \
                or set(self._decode_watermarks) != set(self.ranks):
            raise DriverError('route census is incomplete; refusing a transition')
        if self.ranks != tuple(range(len(self.ranks))):
            # Wire-v1 carries one entry per rank position, so the admitted
            # ranks must be the contiguous worker set.
            raise DriverError('route ranks are not contiguous for a rank-indexed command')
        self.issue_transition(
            expected_route_generation=self.route_generation,
            target_bank=target_bank,
            storage_epochs=tuple(self._storage_epochs[rank] for rank in self.ranks),
            decode_replay_watermarks=tuple(self._decode_watermarks[rank]
                                           for rank in self.ranks))

    def _drive_route(self):
        """Advance bootstrap/split/return; True when stock work must skip."""
        exchange = self._route_exchange()
        if exchange is not None:
            if self._transition is not None and self._transition_retry and not self._inflight:
                self._send_transition()
            # The reply deadline bounds this; the engine must not admit stock
            # work into a route that is being drained or switched underneath it.
            return True
        if not self._bootstrapped:
            if not self.controller.active:
                # Nothing is admitted: the engine does not bind worker route
                # ownership until the first chunk actually needs the split.
                return False
            if self._inflight:
                return True
            command = BootstrapCommand(self.engine_epoch, self.ranks)
            self._bootstrap_command = command
            try:
                self._submit('bootstrap',
                             ('flash_s5_bootstrap', (command,), {}, None),
                             self.ranks)
            except BaseException:
                self._bootstrap_command = None
                raise
            return True
        if self._return_pending is not None:
            self._drive_return()
            return True
        if self.controller.active and not self._split_ready:
            if self._inflight:
                return True
            try:
                self._issue_transition(ROUTE_BANK_SPLIT)
            except E3Pending:
                # Same typed pending as the handback: no route was mutated,
                # no stock work may run, and a later tick retries.
                return True
            return True
        return False

    def _begin(self):
        admission = self.controller.admission()
        row = self._retained
        if row is not None:
            # Only the next chunk of the SAME request/generation may move the
            # retained row through the existing LaneRequest.update() path.  A
            # foreign request, a reused UID with a new generation, or a state
            # epoch that did not advance is refused before any worker open.
            if not row['transaction'].same_request(admission.transaction):
                raise DriverError('a retained request row belongs to another request')
            if admission.state_epoch <= row['state_epoch']:
                raise DriverError('a retained request row has no newer state epoch')
        self.tx = admission.transaction
        self.admission = admission
        self._replies = self._open_replies()
        self._lean_streams = {}
        if self._replies == 'lean' and self.progress is not None:
            self.progress.counters['lean_chunks'] += 1
        self._tx_fused = (self.qsa_fused is not None and self.qsa_fused.selected
                          and self._progress_owned())
        self._fused_mark = self._fused_mark_at = self._fused_stop_at = None
        if self._tx_fused:
            self.qsa_fused.counters['chunks'] += 1
        self._tx_group = (self.layer_groups.size
                          if self._tx_fused and self.layer_groups is not None else 1)
        if self._tx_group != 1:
            self.layer_groups.counters['chunks'] += 1
        self.open_request = OpenRequest(admission, self.ranks,
                                        admission.state_epoch, 1, self._replies,
                                        qsa_fused=self._tx_fused,
                                        layer_group=self._tx_group)
        self.phase = 'open'
        if self.progress is not None and self._progress_owned():
            transaction = admission.transaction
            self.progress.begin(dict(txn=transaction.txn_seq,
                                     request_generation=transaction.request_generation,
                                     chunk_start=transaction.chunk_start,
                                     chunk_end=transaction.chunk_end,
                                     decode_live=self._progress_decode_live()))
        self.plan = []
        self.index = 0
        self._profile_query = None
        self._profile_observed = {}
        self._profile_grants = {}
        self._tail_boundary = None
        self._tail_submission = None
        self._profile_deadline = None
        self._profile_poll_count = 0
        self.reports = {}
        self.accepted_tickets = {}
        self._accepted_descriptors = {}
        self.bound_ticket = None
        self.pp_send_ticket = None
        self.receiver_ticket = None
        self._pp_submission = None
        self.pp_send_seq = None
        self.pp_receive_seq = None
        self.last_ticket = {0: None, 1: None}
        self._bindings = {}
        self._bind_index = 0
        self._pending_binding = None
        self._receipts = {}
        self._receipt_inflight = None
        self._receipt_acked = set()
        self._receipt_last = None
        self._consumer_proofs = {}
        self._pp_noted = False
        self._release_acks = {}
        self._release_attempts = 0
        self._stop_acks = {}
        self._cancel_targets = frozenset()
        self._stop_ticks = 0

    def _drive(self):
        if self._tx_fused:
            self._fused_watchdog()
        if self.phase == 'profile':
            if not self._inflight:
                remaining = self._profile_deadline - self.clock()
                if remaining <= 0:
                    self._refuse_profile('native-schedule-profile-timeout')
                    return
                self._submit('profile', ('flash_s5_poll',
                    (self._profile_query,), {}, None), self.ranks, timeout_s=remaining)
            return
        if self.phase == 'open':
            if not self._inflight:
                self._submit('open', ('flash_s5_open', (self.open_request,), {}, None),
                             self.ranks)
            return
        if self.phase == 'bind':
            if self._inflight:
                return
            self._next_binding()
            return
        if self.phase == 'segments':
            if self._inflight:
                return
            if self._retry is not None:
                self._send_segment(self._retry, self._targets)
                return
            if self.index >= len(self.plan):
                self.phase = 'finalize'
                return
            if self._tx_fused and self._fused_hold(self.plan[self.index]):
                self.qsa_fused.counters['run_ahead_polls'] += 1
                self._submit('poll', ('flash_s5_poll',
                                      (PollRequest(self.tx, self.admission.state_epoch,
                                                   self.poll_budget),), {}, None),
                             self.ranks)
                return
            self._send_plan_entry(self.plan[self.index])
            return
        if self.phase == 'finalize':
            if self._inflight:
                return
            # A sender's PP slot is released only after it accepted the
            # receiver's queried consumer receipt, before publication.
            if self._next_receipt():
                return
            if self._drained() and self.controller.ready_to_release(self.tx):
                self.phase = 'release'
                self._send_release()
                return
            self._submit('poll', ('flash_s5_poll',
                                  (PollRequest(self.tx, self.admission.state_epoch,
                                               self.poll_budget),), {}, None),
                         self.ranks)
            return
        if self.phase == 'release':
            if self._inflight:
                return
            # Not-closed acknowledgments keep worker ownership; retry within
            # the same bound instead of treating an ACK as closure.
            self._send_release()
            return
        raise DriverError('unknown driver phase')

    def _send_plan_entry(self, entry):
        kind = entry['kind']
        stage = entry['stage']
        depends = ()
        # Each worker consumes only its own prior ticket; this tuple names
        # all stage-local predecessors and never fabricates a peer event.
        depends = tuple(self.tickets[r] for r in self.stage_ranks[stage]
                        if r in self.tickets)
        layer = entry.get('layer')
        descriptor = SegmentDescriptor(
            self.tx, kind, stage, depends_on=depends,
            layer_start=layer, layer_end=entry.get('layer_end', None if layer is None else layer + 1),
            seq=entry.get('seq'), layer_group=entry.get('group', 1))
        targets = self.stage_ranks[stage]
        tracker = self.trackers[targets[0]]
        op_seq = tracker.propose(descriptor)
        for rank in targets[1:]:
            if self.trackers[rank].propose(descriptor) != op_seq:
                raise DriverError('lane op_seq proposals diverged')
        # Only a stage's first PREPARE carries the engine's actual source lease;
        # a missing/stale/foreign hold refuses before any GPU submission.
        source = (self._prepare_source(targets)
                  if kind is SegmentKind.PREPARE else None)
        profile = (self._profile_grants.get((stage, descriptor.layer_end - 1))
                   if kind is SegmentKind.LAYER else None)
        command = SegmentCommand(descriptor, op_seq, targets, source=source, profile=profile)
        boundary = self._tail_boundary
        if kind is SegmentKind.LAYER and boundary is not None:
            if boundary != (self.tx, stage, layer, boundary[3], self.route_generation):
                raise DriverError('TP tail boundary belongs to another layer or route')
            if self._tail_submission is not None:
                if self._tail_submission.command != command:
                    raise DriverError('TP tail command replaced its retained obligation')
            else:
                self._tail_submission = TPTailSubmissionObligation(
                    command, boundary[3], self.route_generation)
        if self._tx_fused and profile is not None:
            # A fused layer contains its TP tail: an accepted peer waits in the
            # collective, so the same command reaches every peer before any cancel.
            if self._tail_submission is None:
                self._tail_submission = TPTailSubmissionObligation(
                    command, profile.query.boundary_op_seq, self.route_generation)
            elif self._tail_submission.command != command:
                raise DriverError('fused layer replaced its retained TP obligation')
        elif self._tx_group != 1 and kind is SegmentKind.LAYER:
            # A group without a QSA layer still holds collectives, so it carries the same
            # obligation. Its boundary is this stage's last accepted command (PREPARE at least).
            if self._tail_submission is None:
                self._tail_submission = TPTailSubmissionObligation(
                    command, self.tickets[targets[0]].op_seq, self.route_generation)
            elif self._tail_submission.command != command:
                raise DriverError('layer group replaced its retained TP obligation')
        if kind is SegmentKind.PP_SEND:
            self.pp_send_seq = descriptor.seq
            if self._pp_submission is None:
                self._pp_submission = PPSubmissionObligation(command, self.route_generation)
            elif self._pp_submission.send != command:
                raise DriverError('PP send replaced a retained submission obligation')
        if kind is SegmentKind.PP_RECEIVE:
            self.pp_receive_seq = descriptor.seq
            obligation = self._pp_submission
            if (obligation is None or obligation.send.transaction != self.tx
                    or obligation.send.descriptor.seq != descriptor.seq
                    or obligation.route_generation != self.route_generation
                    or obligation.send_accepted != set(self.stage_ranks[0])):
                raise DriverError('PP receive requires matching accepted sends')
            if obligation.receive is not None and obligation.receive != command:
                raise DriverError('PP receive replaced a retained submission command')
            obligation.receive = command
        self._send_segment(command, targets)

    def _prepare_source(self, targets):
        """The parked chunk's actual immutable lease for a PREPARE command.

        The descriptor is produced from the scheduler's retained block objects
        and its exact admission; it is refused when missing, released, foreign
        to this transaction/state epoch, unbound from the opened lanes, or not a
        subset of this admission's own rows for the target ranks. The engine
        keeps ownership of the block objects throughout.
        """
        hold = self.controller.source_hold(self.tx)
        if hold is None:
            raise DriverError('PREPARE has no scheduler-owned source hold')
        if hold.transaction != self.tx:
            raise DriverError('source hold belongs to another transaction')
        if self.admission is None or hold.state_epoch != self.admission.state_epoch:
            raise DriverError('source hold state epoch is stale')
        owners = hold.owners
        if owners is None:
            raise DriverError('source hold owners are not bound to the opened lanes')
        if {owner.worker_rank for owner in owners} != set(self.ranks):
            raise DriverError('source hold owners do not cover every opened rank')
        try:
            descriptor = hold.descriptor()
            descriptor.validate_frame(self.admission, tuple(targets))
        except E3ContractError as exc:
            raise DriverError('source hold descriptor was refused: ' + str(exc)) from exc
        return descriptor

    def _send_segment(self, command, targets):
        self._command = command
        self._targets = tuple(targets)
        self._submit('segment', ('flash_s5_submit', (command,), {}, None), self.ranks)

    # -- stop path --------------------------------------------------------
    def stop(self, kind, *, drain=False):
        if kind not in _STOP_KINDS:
            raise DriverError('unknown stop kind')
        route_pending = (self._return_pending is not None
                         or self._transition is not None
                         or self._bootstrap_command is not None)
        if self.tx is None and not self.controller.active and not route_pending:
            if self._retained is not None:
                # Between chunks the request row is owned with no chunk, GPU
                # replay or parked batch to cancel.  Close it through the same
                # bounded typed worker path; only then is a full return owed.
                if self.stop_kind is None:
                    self.stop_kind = kind
                if kind == 'pause' and self._control_owns_route():
                    # A control intent owns this split route: the row closes,
                    # the route is not abandoned and no full return is owed.
                    self._record('control_route_pause_kept',
                                 reason=self._control_route.get('intent'),
                                 stop_kind=kind,
                                 retained_transaction=self._retained['transaction'].txn_seq,
                                 route_generation=self.route_generation)
                elif self._split_ready:
                    self._return_pending = 'stop'
                self.pump_stop(self.stop_tick_budget)
                return self.phase
            if not self._split_ready:
                return None
            if kind == 'pause' and self._control_owns_route():
                # An explicit native-control intent selected this split route.
                # A pause freezes the ordinary cohort; it is not permission to
                # abandon the route (which would change generation and make the
                # observation refuse).  Nothing is owed and nothing is parked.
                self._record('control_route_pause_kept',
                             reason=self._control_route.get('intent'),
                             stop_kind=kind,
                             route_generation=self.route_generation)
                return self.phase
            # Idle on the split route with no chunk to cancel: the return is
            # still owed before stock may touch the full bank again.
            self._return_pending = 'stop'
        if self.stop_kind is None:
            self.stop_kind = kind
        # Bounded pump: the engine seam sees one call, never a device wait.
        self.pump_stop(self.stop_tick_budget)
        return self.phase

    def pump_stop(self, budget=None):
        """Run at most ``budget`` bounded ticks; never wait for a forward."""
        budget = self.stop_tick_budget if budget is None else _positive(
            budget, 'stop pump budget')
        for _ in range(budget):
            if not self.active:
                return True
            # Utility/pause callers do not own EngineCore's output channel.
            # Leave a certified profile-refusal output for the engine step.
            self.advance(deliver_outputs=False)
        return not self.active

    def _drive_stop(self):
        if self._inflight and any(kind != 'poll'
                                  for kind in self._inflight.values()):
            # Waiting for a reply is not a cancellation round: the owned
            # transport's reply deadline bounds it, not the engine's tick rate.
            return
        if self._transition is not None:
            # A stop cannot abandon a partially applied route or report a
            # clean pause while ready ranks and a draining rank disagree.
            # Finish the exact outstanding command under its original deadline
            # before cancellation/full-route retirement can progress.
            if self._transition_retry:
                self._send_transition()
            return
        if self.tx is None:
            if self._retained is not None:
                # The request row outlived its chunk.  It is not a parked
                # admission and not a published chunk: close the exact row the
                # workers certified before retiring any parked next chunk.
                self._drive_retained_cancel()
                return
            # Admission happened pre-schedule; the driver may not have begun.
            # Closing admissions retires the parked chunk without publishing.
            if self.controller.active:
                admission = self.controller.admission()
                self.controller.cancel(admission.transaction,
                                       never_submitted=True)
                self._record('cancel', transaction=admission.transaction.txn_seq,
                             never_submitted=True, closed_ranks=[],
                             reason=self.stop_kind)
                self._reset()
                self._note_terminal_route('cancel')
            if self._return_pending is not None:
                self._drive_return()
                return
            # Nothing is owned and the full route is restored: stop is done.
            self.stop_kind = None
            return
        if self.phase == 'open' and self._retained is not None:
            # The ring refused this continuation's open, so no worker saw this
            # chunk: a sent open is still in _inflight above, and its reply moves
            # the row.  The workers hold the previous chunk's row and would
            # quarantine a cancel for a chunk they never opened.  Close that row
            # first; a later tick retires this chunk as never submitted, through
            # the ordinary all-rank cancel below (every rank answers never_open).
            self._drive_retained_cancel()
            return
        if self._drive_tail_stop_submission():
            return
        if self._drive_pp_stop_submission():
            return
        if self._receipts_pending():
            self._stop_round()
            self._next_receipt()
            return
        if self._drive_partial_stop_submission():
            return
        closed = self._closed_rows()
        if set(closed) == set(self.ranks):
            self._retire_cancel(closed)
            return
        # One bounded cancellation/poll tick. The command is one all-rank
        # broadcast: every worker executes it and replies, so the expected reply
        # ranks are always every rank (a subset would desynchronize the owned
        # FIFO transport). Already-closed ranks answer without new ownership
        # and their earlier terminal row is kept.
        self._stop_round()
        reason = self.stop_kind
        self._cancel_targets = frozenset(self.ranks)
        self._submit('cancel', ('flash_s5_cancel',
                                (CancelRequest(self.tx, self.admission.state_epoch,
                                               reason),), {}, None),
                     self.ranks)

    def _drive_partial_stop_submission(self):
        """Finish a segment one TP peer accepted while another answered pending.

        The exact retained command (same transaction, op_seq and descriptor) is
        replayed to its targets under the stop-round bound, as the tail and PP
        obligations above are, so every target holds it before the cancel.
        Never a new command.  A rank that already accepted it answers from its
        existing record.  Without this, a stop after a partial PREPARE closed
        every rank and then refused retirement (accepted operations without a
        bound producer), keeping ownership.
        """
        command = self._retry
        if command is None:
            return False
        targets = self._targets
        # op_seq numbering is per lane, so another stage may hold the same number.
        accepted = {rank for rank, op_seq in self.accepted_tickets
                    if op_seq == command.op_seq and rank in targets}
        if not accepted or set(targets) <= accepted:
            return False
        if command.transaction != self.tx or self._command != command:
            raise DriverError('stop found a foreign partially accepted command')
        self._stop_round()
        self._record('stop_partial_replay', transaction=self.tx.txn_seq,
                     op_seq=command.op_seq,
                     segment_kind=getattr(command.kind, 'value', str(command.kind)),
                     accepted_ranks=sorted(accepted),
                     pending_ranks=sorted(set(targets) - accepted),
                     reason=self.stop_kind)
        self._send_segment(command, targets)
        return True

    def _drive_pp_stop_submission(self):
        """Finish a partially accepted transfer before cancelling either end.

        Never finish unrelated layers or wait for send completion before the
        receive. Missing RPC replies remain transport failures. A responsive
        pending peer is retried under the unchanged stop-round bound.
        """
        obligation = self._pp_submission
        if obligation is None or not obligation.send_accepted:
            return False
        if (obligation.send.transaction != self.tx
                or obligation.route_generation != self.route_generation):
            raise DriverError('stop found a foreign PP submission obligation')
        if obligation.send_accepted != set(self.stage_ranks[0]):
            self._stop_round()
            self._send_segment(obligation.send, self.stage_ranks[0])
            return True
        if obligation.receive_accepted != set(self.stage_ranks[1]):
            self._stop_round()
            if obligation.receive is None:
                self._send_plan_entry(dict(kind=SegmentKind.PP_RECEIVE,
                                           stage=1, seq=obligation.send.descriptor.seq))
            else:
                self._send_segment(obligation.receive, self.stage_ranks[1])
            return True
        return False

    def _drive_tail_stop_submission(self):
        """Settle any accepted tail entry before cancelling its TP peer.

        A stop before the all-rank tail command has no tail obligation. Once a
        rank accepted that command, the exact same typed command is retried to
        every active-stage peer until both accepted it. The normal cancellation
        path then queries completion; a missing reply remains a transport
        failure, never a fabricated clean close.
        """
        obligation = self._tail_submission
        if obligation is None or not obligation.accepted:
            return False
        command = obligation.command
        if (command.transaction != self.tx
                or obligation.route_generation != self.route_generation):
            raise DriverError('stop found a foreign TP tail obligation')
        targets = self.stage_ranks[command.stage]
        if obligation.accepted != set(targets):
            self._stop_round()
            self._send_segment(command, targets)
            return True
        return False

    def _drive_retained_cancel(self):
        """Close the exact retained request row through the typed worker path.

        One all-rank ``CancelRequest`` for the retained transaction and its own
        state epoch; every rank executes the broadcast and replies.  The command
        is immutable and retried unchanged until every rank certifies closure or
        the stop bound refuses (ownership is kept in both cases).
        """
        row = self._retained
        if row is None:
            return
        if self._inflight and any(kind != 'poll'
                                  for kind in self._inflight.values()):
            return
        acks = self._retained_cancel_acks
        if set(acks) == set(self.ranks) and all(ack.closed for ack in acks.values()):
            self._retire_retained_cancel()
            return
        self._stop_round()
        request = self._retained_cancel_request
        if request is None:
            request = CancelRequest(row['transaction'], row['state_epoch'],
                                    self.stop_kind or 'cancel')
            self._retained_cancel_request = request
        self._submit('retained_cancel',
                     ('flash_s5_cancel', (request,), {}, None), self.ranks)

    def _apply_retained_cancel(self, values):
        """One bounded all-rank close of the retained request row."""
        row = self._retained
        if row is None:
            raise DriverError('retained cancellation has no owned request row')
        acks = tuple(values)
        if (len(acks) != len(self.ranks)
                or any(type(ack) is not CancelAck for ack in acks)
                or {ack.worker_rank for ack in acks} != set(self.ranks)):
            raise DriverError('retained cancellation lacks the typed all-rank reply')
        for ack in acks:
            if (ack.transaction != row['transaction']
                    or ack.state_epoch != row['state_epoch']):
                raise DriverError(
                    'retained cancellation binds another transaction/state epoch')
            expected = row['completed_sequences'][ack.worker_rank]
            prior = self._retained_cancel_acks.get(ack.worker_rank)
            if ack == prior:
                continue
            if ack.disposition == 'quarantined':
                # Keep the row and the quarantine disposition; never report a
                # clean close from a worker that still owns storage.
                self._retained_cancel_acks[ack.worker_rank] = ack
                raise DriverError('worker quarantined the retained row; ownership kept')
            if not ack.closed:
                # The rank still owns the row; keep waiting/retrying.
                self._retained_cancel_acks[ack.worker_rank] = ack
                continue
            if not self._retained_close_certified(ack, expected):
                raise DriverError(
                    'retained row closed without its certified queried report')
            if prior is not None and prior.closed:
                # The rank already certified this exact row closed; the
                # broadcast copy confirms it and never replaces the terminal row.
                continue
            self._retained_cancel_acks[ack.worker_rank] = ack

    def _retained_close_certified(self, ack, expected):
        """One closed ack that binds the retained transaction and final report."""
        # This worker positively retained the row; zero-history never_open
        # contradicts that binding. A retry must return its typed close report.
        if ack.disposition == 'never_open':
            return False
        report = ack.report
        row = self._retained
        return (ack.disposition == 'released' and report is not None
                and report.transaction == row['transaction']
                and report.state_epoch == row['state_epoch']
                and report.worker_rank == ack.worker_rank
                and report == row['reports'][ack.worker_rank]
                and ack.completed_op_seqs == report.completed_op_seqs
                and not ack.pending_op_seqs
                and report.completed_sequence == expected
                and ack.completed_sequence == expected)

    def _retire_retained_cancel(self):
        """Forget a row every rank certified closed; no publication/frontier move."""
        row = self._retained
        acks = self._retained_cancel_acks
        if row is None or set(acks) != set(self.ranks) \
                or any(not ack.closed for ack in acks.values()):
            raise DriverError('retained cancellation lacks an all-rank close disposition')
        self._record('request_row_closed', transaction=row['transaction'].txn_seq,
                     request_generation=row['transaction'].request_generation,
                     state_epoch=row['state_epoch'],
                     chunk_start=row['transaction'].chunk_start,
                     chunk_end=row['transaction'].chunk_end,
                     ranks=list(self.ranks),
                     reason=self.stop_kind,
                     dispositions=sorted({ack.disposition for ack in acks.values()}))
        self._remember_native_closure(row['transaction'], acks, row['reports'], self.stop_kind)
        self._retained = None
        self._retained_cancel_request = None
        self._retained_cancel_acks = {}

    def _fused_stop_round(self):
        """A fused stop waits for queued device work, and rounds run about every millisecond,
        so a round count cannot bound it. The stall bound does, with ownership retained."""
        now = self.clock()
        if self._fused_stop_at is None:
            self._fused_stop_at = now
            self.qsa_fused.counters['stops'] += 1
        counters = self.qsa_fused.counters
        elapsed = now - self._fused_stop_at
        counters['stop_rounds_max'] = max(counters['stop_rounds_max'], self._stop_ticks)
        counters['stop_ms_max'] = max(counters['stop_ms_max'], int(elapsed * 1e3))
        if elapsed > self.qsa_fused.stall_s:
            raise DriverError('cancellation did not close worker ownership within the bound')

    def _stop_round(self):
        """Count one submitted stop round; refuse past the bound, keep ownership."""
        self._stop_ticks += 1
        if self._tx_fused:
            return self._fused_stop_round()
        if self._stop_ticks > self.stop_budget:
            raise DriverError('cancellation did not close worker ownership within the bound')

    def _receipts_pending(self):
        return any(sender not in self._receipt_acked for sender in self._receipts)

    def _closed_rows(self):
        """Every rank's certified-closed terminal row (release or cancel)."""
        rows = {rank: ack for rank, ack in self._release_acks.items() if ack.closed}
        for rank, ack in self._stop_acks.items():
            if ack.closed and rank not in rows:
                rows[rank] = ack
        return rows

    def _apply_cancel(self, values):
        acks = tuple(values)
        targets = self._cancel_targets or frozenset(self.ranks)
        if (len(acks) != len(targets)
                or {getattr(a, 'worker_rank', None) for a in acks} != set(targets)
                or any(type(a) is not CancelAck for a in acks)):
            raise DriverError('cancel reply does not cover every outstanding rank')
        closed = self._closed_rows()
        for ack in acks:
            if ack.transaction != self.tx or ack.state_epoch != self.admission.state_epoch:
                raise DriverError('cancel acknowledgment belongs to another transaction/epoch')
            if ack.worker_rank in closed:
                # The rank already certified closure; its broadcast copy may
                # only confirm it and never replaces the terminal row.
                if not ack.closed:
                    raise DriverError('a closed worker reports cancellation ownership again')
                continue
            if ack.disposition == 'quarantined':
                self._stop_acks[ack.worker_rank] = ack
                raise DriverError('worker retained quarantined work; ownership kept')
            prior = self._stop_acks.get(ack.worker_rank)
            if prior is not None:
                if (ack.completed_sequence < prior.completed_sequence
                        or not set(prior.completed_op_seqs)
                        <= set(ack.completed_op_seqs)
                        or not set(prior.pending_op_seqs) >= set(ack.pending_op_seqs)):
                    raise DriverError('cancellation report regressed')
            if ack.report is not None:
                self._ingest_report(ack.report)
            elif ack.disposition != 'never_open':
                raise DriverError('cancellation closed a worker without a queried report')
            self._stop_acks[ack.worker_rank] = ack

    def _remember_native_closure(self, transaction, acks, reports, reason):
        """Retain the already-certified all-rank close; no new completion proof.

        Pure metadata after the existing closure/publication guards. Failure to
        observe this record never changes execution or retries an operation.
        It survives _reset and is bound to the exact transaction, not a last
        active scope or a guessed request identity.
        """
        try:
            rows = []
            for rank in self.ranks:
                ack = acks[rank]
                report = getattr(ack, 'report', None) or reports.get(rank)
                rows.append({'worker_rank': rank, 'typed': type(ack).__name__,
                    'transaction': ack.transaction.txn_seq,
                    'request_generation': ack.transaction.request_generation,
                    'state_epoch': ack.state_epoch, 'closed': ack.closed,
                    'released': ack.released, 'request_closed': ack.request_closed,
                    'disposition': getattr(ack, 'disposition', None),
                    'completed_sequence': ack.completed_sequence,
                    'report': (None if report is None else {
                        'worker_rank': report.worker_rank,
                        'transaction': report.transaction.txn_seq,
                        'request_generation': report.transaction.request_generation,
                        'state_epoch': report.state_epoch,
                        'completed_sequence': report.completed_sequence,
                        'completed_op_seqs': list(report.completed_op_seqs),
                        'pending_op_seqs': list(report.pending_op_seqs),
                        'proofs': [{'kind': proof.kind.value,
                                    'op_seq': proof.ticket.op_seq,
                                    'completion_seq': proof.completion_seq}
                                   for proof in report.proofs]})})
            self._last_native_closure = {
                'schema': 's5-native-certified-close/1',
                'engine_epoch': self.engine_epoch, 'transaction': transaction.txn_seq,
                'request_generation': transaction.request_generation,
                'request_uid_sha256': hashlib.sha256(transaction.request_uid.encode()).hexdigest(),
                'chunk_start': transaction.chunk_start, 'chunk_end': transaction.chunk_end,
                'observed_ns': int(self._monotonic_ns()), 'clock_basis': self.clock_basis,
                'reason': reason, 'rows': rows,
                'source_leases': self.controller.leases.shutdown(),
                'pp_receipts_pending': self._receipts_pending(),
                'basis': 'existing typed all-rank close guards and their same queried reports',
                'complete': True}
        except BaseException as exc:
            self._last_native_closure = {'schema': 's5-native-certified-close/1',
                'complete': False, 'error_type': type(exc).__name__}

    def _retire_cancel(self, acks):
        """Terminal cancellation row: storage/request closure, no publication."""
        if set(acks) != set(self.ranks):
            raise DriverError('cancellation lacks an all-rank close disposition')
        if any(not ack.closed for ack in acks.values()):
            raise DriverError('cancellation retirement requires closed worker ownership')
        if self.bound_ticket is None:
            if self.accepted_tickets:
                raise DriverError('accepted operations exist without a bound producer')
            request = self.controller.cancel(self.tx, never_submitted=True)
            if self._profile_failure is not None:
                self._profile_failed_request = request
            self._remember_native_closure(self.tx, acks, self.reports, self.stop_kind)
            self._reset()
            self._note_terminal_route('cancel')
            self._publish_profile_failure()
            return
        # Cancellation is its own retirement path: reports supply completion
        # identity, but state_written()/publication is never consulted here.
        # A row is either this path's CancelAck or a closed ReleaseAck from a
        # normal release that raced the stop (it carries no report of its own).
        for rank in self.ranks:
            ack = acks[rank]
            report = getattr(ack, 'report', None) or self.reports.get(rank)
            if report is None:
                raise DriverError('cancellation retirement needs a queried report per rank')
            if (report.pending_op_seqs or report.failed
                    or report.completed_sequence != ack.completed_sequence):
                raise DriverError('cancellation report is not a clean terminal row')
            row = ack if type(ack) is ReleaseAck else ack.release_ack(
                report.completed_sequence)
            self.controller.note_worker_release(row)
        request = self.controller.cancel(self.tx, proof=self._producer_proof())
        if self._profile_failure is not None:
            self._profile_failed_request = request
        self._record('cancel', transaction=self.tx.txn_seq, never_submitted=False,
                     closed_ranks=[rank for rank in self.ranks if acks[rank].closed],
                     reason=self.stop_kind)
        self._remember_native_closure(self.tx, acks, self.reports, self.stop_kind)
        self._reset()
        self._note_terminal_route('cancel')
        self._publish_profile_failure()

    def _control_owns_route(self):
        """True while an explicit native-control intent owns the split route."""
        record = self._control_route
        return (record is not None and self._split_ready
                and record.get('engine_epoch') == self.engine_epoch
                and record.get('route_generation') == self.route_generation)

    def _split_hold_timeout(self):
        """End an idle physical-split stock hold by returning FULL (native40).

        The held stock work then runs on FULL, as after a terminal handback;
        the intent closes on the FULL ack ('route-returned-full').  Only with
        nothing owned: no chunk, no parked admission, no retained worker row,
        no route exchange or pending return.
        """
        intent = self._admission_intent
        if (not self._split_ready or self.tx is not None
                or self.controller.active or self._retained is not None
                or self.stop_kind is not None or self._control is not None or self._inflight
                or self._return_pending is not None or self._route_exchange() is not None):
            return False
        query = getattr(self.controller, 'expired_split_hold', None)
        if not callable(query) or self._engine_paused() or not self._scheduler_holds_requests():
            # Paused, or every held request left (finished/aborted client):
            # nothing is stranded, the applied route stays as its owner set it.
            return False
        now = int(self._monotonic_ns())
        hold = query(now, self.engine_epoch, self.route_generation)
        if hold is None:
            return False
        receipt = self.controller.note_split_hold_timeout(self.scheduler, hold, now, intent)
        self._record('admission_hold_timeout', phase=None if intent is None else intent['phase'],
                     seq=None if intent is None else intent['seq'],
                     admission_armed=intent is not None, reason=receipt['reason'],
                     held_since_ns=hold['since_ns'], held_ns=now - hold['since_ns'],
                     bound_ns=self.controller.split_hold_bound_ns, holds=hold['holds'],
                     route_generation=self.route_generation)
        self._note_terminal_route(receipt['reason'])
        self._drive_return()
        return True

    def _scheduler_holds_requests(self):
        scheduler = self.scheduler
        requests = getattr(scheduler, 'requests', None)
        if type(requests) is dict:
            return bool(requests)
        unfinished = getattr(scheduler, 'get_num_unfinished_requests', None)
        return callable(unfinished) and unfinished() > 0

    def _note_terminal_route(self, reason):
        """A chunk retired for good: the split route now owes a full return."""
        if self._split_ready:
            self._return_pending = reason

    def pp_sent(self):
        if self.tx is None:
            return False
        return any(ledger.outstanding(self.tx) for ledger in self.pp_ledgers.values())

    def _pp_consumed(self):
        return not self.pp_sent()

    def _drained(self):
        """Every accepted op on every rank; a producer is not a terminal fence."""
        if set(self.reports) != set(self.ranks):
            return False
        for rank in self.ranks:
            report = self.reports[rank]
            expected = {op for r, op in self.accepted_tickets if r == rank}
            if (report.failed or report.pending_op_seqs
                    or not expected <= set(report.completed_op_seqs)):
                return False
        return True

    def _reset(self):
        """Drop one chunk's ownership.  Route state is worker-stable, not per-chunk.

        ``_bootstrapped``/``_storage_epochs``/``_decode_watermarks``/
        ``_split_ready``/``_return_pending`` deliberately survive: the storage
        epoch and replay census belong to the worker incarnation, and a chunk
        boundary is never a route boundary.  ``_retained`` also survives: a
        nonterminal release leaves the worker request row owned by the request,
        and only a same-request reopen or a typed all-rank close clears it.
        """
        if self.progress is not None:
            self.progress.end(_progress_logger())
        self._tx_fused = False
        self._tx_group = 1
        self.tx = None
        self.admission = None
        self.phase = 'idle'
        self.plan = []
        self.index = 0
        self._retry = None
        self._profile_query = None
        self._profile_observed = {}
        self._profile_grants = {}
        self._tail_boundary = None
        self._tail_submission = None
        self._profile_deadline = None
        self._profile_poll_count = 0
        self.stop_kind = None
        self._command = None
        self._targets = ()
        self.lanes = {}
        self.trackers = {}
        self.tickets = {}
        self.accepted_tickets = {}
        self._accepted_descriptors = {}
        self.reports = {}
        self._replies = 'full'
        self._lean_streams = {}
        self.bound_ticket = None
        self.pp_send_ticket = None
        self.receiver_ticket = None
        self._pp_submission = None
        self.pp_send_seq = None
        self.pp_receive_seq = None
        self._bindings = {}
        self._bind_index = 0
        self._pending_binding = None
        self._receipts = {}
        self._receipt_inflight = None
        self._receipt_acked = set()
        self._receipt_last = None
        self._consumer_proofs = {}
        self._pp_noted = False
        self._release_acks = {}
        self._release_attempts = 0
        self._stop_acks = {}
        self._stop_rounds = 0
        self._stop_ticks = 0
        self._cancel_targets = frozenset()

    def finish_pause(self):
        # A pause utility may arrive while a route/worker intent is registered:
        # settle what the owned transport already holds, then advance one stage.
        self._drain()
        self.pump_control()
        # Only a live chunk or an already-started bounded stop is cancelled
        # here.  A registered control intent owns a route transition, not
        # stock work; it keeps its ownership and settles from the typed reply
        # or its absolute deadline, so pause never waits on it forever.
        if self.tx is not None or self.stop_kind is not None:
            if self.stop_kind is None:
                self.stop('pause')
            else:
                self.pump_stop(self.stop_tick_budget)
            if self.active:
                raise E3Pending('pause waits for bounded native cancellation to close')
        report = self.controller.leases.shutdown()
        if not report['clean']:
            raise DriverError('pause with outstanding E3 block leases')
        return True

    # -- bounded native-control utility -----------------------------------
    #
    # ``EngineCore.flash_s5_native_control(command)`` is the only engine entry
    # point.  It never pumps, polls or waits: a read settles synchronously from
    # queried engine state, and every worker query / route mutation registers
    # one intent that the engine's own tick (``advance``/``pump_control``)
    # submits and settles from the owned reply transport.  Client cancellation
    # never drops route or worker ownership; the absolute deadline or the typed
    # reply settles it.

    def control_pending(self):
        return self._control is not None

    def control_blocks_stock(self):
        """True while stock work may not be launched into this engine tick.

        Every control intent is admitted only against a paused engine, so a
        registered intent never holds stock work that is allowed to run; the
        route exchange predicates are kept because an intent may have settled
        while the engine-owned transition it issued is still outstanding.
        """
        if self._route_exchange() is not None or self._return_pending is not None:
            return True
        return self._control is not None

    def admission_intent(self):
        """The typed engine-owned admission intent, or ``None`` when off."""
        intent = self._admission_intent
        return None if intent is None else dict(intent)

    def _state_control(self, action, command):
        """Arm/disarm the typed admission intent; never a raw boolean.

        Both actions are synchronous engine state: no worker RPC, no device
        wait and no poll loop.  They refuse while a chunk, route exchange or
        registered control intent owns work, so disabling can never silently
        park an admitted request.
        """
        now = int(self._monotonic_ns())
        epoch = command.get('engine_epoch')
        if epoch is not None and (type(epoch) is not int or epoch != self.engine_epoch):
            # A stale client may not arm or disarm the current engine's phase.
            raise DriverError('native-control-engine-epoch-differs')
        if action == 'arm_admission':
            previous = None
            self._arm_admission(command.get('phase'))
        elif action == 'disarm_admission':
            previous = self._disarm_admission('explicit')
        else:
            raise DriverError('native-control-state-action-not-implemented')
        return {'schema': 1, 'action': action, 'engine_epoch': self.engine_epoch,
                'route': 'split' if self._split_ready else 'full',
                'route_generation': self.route_generation,
                'admission_intent': self.admission_intent(),
                'previous_admission_intent': previous,
                'observed_clock_ns': now,
                'clock': self._clock_record(now),
                'clock_basis': self.clock_basis}

    def _arm_admission(self, phase):
        policy = getattr(self, 'auto_arm', None)
        auto = (phase == 'auto-prefill' and policy is not None
                and policy.active and policy._internal)
        if not auto:
            if phase not in ADMISSION_PHASES:
                raise DriverError('native-admission-phase-unknown')
            if phase not in NATIVE_ADMISSION_PHASES:
                raise DriverError('native-admission-phase-not-native')
        if not self._engine_paused():
            # The phase is chosen on a frozen engine and takes effect when the
            # caller resumes; arming mid-step would park into live sampling.
            raise DriverError('native-admission-requires-paused-engine')
        if (self._control is not None or self._inflight
                or self._route_exchange() is not None
                or self._return_pending is not None):
            raise DriverError('native-admission-requires-idle-control-transport')
        if self.tx is not None or self.controller.active:
            raise DriverError('native-admission-arm-with-owned-chunk')
        if not self._split_ready:
            raise DriverError('native-admission-requires-split-route')
        existing = self._admission_intent
        if existing is not None:
            if existing['phase'] != phase:
                raise DriverError('native-admission-intent-change-with-owned-intent')
            # Idempotent same-phase intent may continue the SAME retained
            # request row; it grants nothing about a different request.
            return dict(existing)
        if self._retained is not None:
            # Arming a new phase cannot silently abandon a row the workers
            # still own for the previous request.
            raise DriverError('native-admission-arm-with-retained-request-row')
        self._admission_seq += 1
        intent = {'schema': 1,
                  'owner': 'engine-core',
                  'engine_epoch': self.engine_epoch,
                  'phase': phase,
                  'route': 'split',
                  'route_generation': self.route_generation,
                  'armed': True,
                  'seq': self._admission_seq,
                  'armed_ns': int(self._monotonic_ns())}
        if auto:
            intent.update(origin='auto-arm', selection_seq=policy.selection_seq,
                          request_uid=policy._request().request_id)
        self._admission_intent = intent
        self._record('admission_armed', phase=phase, seq=intent['seq'],
                     route_generation=self.route_generation)
        return dict(intent)

    def _disarm_admission(self, reason):
        intent = self._admission_intent
        if self._retained is not None:
            # Disarm is a phase change even when no intent is currently armed:
            # it must see actual row closure (through the stop/cancel path)
            # instead of reporting a clean phase while workers own a row.
            raise DriverError('native-admission-disable-with-retained-request-row')
        if intent is None:
            return None
        if (self._control is not None or self._inflight
                or self._route_exchange() is not None
                or self._return_pending is not None):
            raise DriverError('native-admission-requires-idle-control-transport')
        if self.tx is not None or self.controller.active:
            raise DriverError('native-admission-disable-with-owned-work')
        self._admission_intent = None
        self._record('admission_disarmed', phase=intent['phase'], seq=intent['seq'],
                     reason=reason, route_generation=self.route_generation)
        return dict(intent)

    def submit_control(self, command):
        policy = getattr(self, 'auto_arm', None)
        if policy is not None:
            return policy.manual_control(command, self._submit_control)
        return self._submit_control(command)

    def _submit_control(self, command):
        if type(command) is not dict:
            raise DriverError('native-control-command-not-an-object')
        action = command.get('action')
        if action == 'native_progress' and self.progress is not None:
            return self._progress_control(command)
        if action == 'native_qsa_fused' and self.qsa_fused is not None:
            return self._fused_control(command)
        if action == 'native_layer_group' and self.layer_groups is not None:
            return self._layer_group_control(command)
        if action not in CONTROL_ACTIONS:
            raise DriverError('native-control-unknown-action')
        if action == 'cancel':
            # Cancellation is always available: it marks the registered intent
            # and reports ownership, never drops or settles anything itself.
            return self._cancel_control()
        if action in CONTROL_READ_ACTIONS:
            # Reads settle synchronously from queried engine state and never
            # take ownership, so they stay available while an intent (even an
            # abandoned one) is registered.
            return self._read_control(action, command)
        if self.error is not None:
            raise DriverError(self.error)
        if action in CONTROL_STATE_ACTIONS:
            return self._state_control(action, command)
        if self._control is not None:
            raise DriverError('native-control-busy')
        self._admit_control(action, command)
        future = Future()
        self._control_future = future
        self._control = {'action': action,
                         'command': dict(command),
                         'stage': 'pending',
                         'inner': None,
                         'notices': [],
                         'started_ns': int(self._monotonic_ns()),
                         'start_route': ('split' if self._split_ready else 'full'),
                         'start_route_generation': self.route_generation}
        self._control_deadline = self.clock() + self._control_timeout_s
        self._control_seq = None
        self._control_reply = None
        return future

    def _admit_control(self, action, command):
        epoch = command.get('engine_epoch')
        if epoch is not None and epoch != self.engine_epoch:
            raise DriverError('native-control-engine-epoch-differs')
        if action == 'qualification':
            worker_action = command.get('worker_action')
            from native_overlap_worker import QUALIFICATION_ACTIONS
            if worker_action not in QUALIFICATION_ACTIONS:
                raise DriverError('native-qualification-action-unknown')
        if self._route_exchange() is not None or self._return_pending is not None:
            raise DriverError('native-control-route-exchange-outstanding')
        if action in CONTROL_ROUTE_ACTIONS:
            # A route intent needs the scheduler paused and no live chunk
            # transaction.  Frozen requests may still be owned by the scheduler:
            # the all-rank transition drains their streams, it does not require
            # them to be finished, so route admission is pause + idle driver,
            # never an empty scheduler.
            if not self._engine_paused():
                raise DriverError('native-control-route-requires-paused-engine')
            if self.tx is not None or self._inflight:
                raise DriverError('native-control-requires-idle-driver')
            if self._retained is not None:
                # A route change to/from the full bank is not allowed to
                # abandon a worker request row; close it through the stop path.
                raise DriverError('native-control-requires-retained-row-closure')
            return
        if self.live or self._inflight:
            # A live native chunk, source hold, PP transfer or route exchange
            # is never bypassed just because the cohort is ordinary decode.
            # A retained request row is idle metadata: observation may query it
            # and the worker census reports it, but it is never a chunk.
            raise DriverError('native-control-requires-idle-driver')
        if not self._engine_paused():
            raise DriverError('native-control-requires-paused-engine')
        if action == 'qualification':
            worker_action = command.get('worker_action')
            if worker_action in ('teacher_configure', 'row1_configure'):
                # Changing the qualification target needs a genuinely
                # configurable (empty) worker, not a frozen live cohort.
                if not self._engine_quiescent():
                    raise DriverError(
                        'native-' + worker_action.replace('_', '-')
                        + '-requires-paused-empty-engine')
                return
            if worker_action in CONTROL_LIVE_COHORT_QUALIFICATION:
                return
            raise DriverError('native-qualification-action-unknown')
        if action in CONTROL_LIVE_COHORT_ACTIONS:
            # Census/barrier/drain/poll observe native ownership only.  The
            # paused ordinary cohort is reported separately in the row and is
            # never a reason to refuse a native query or to call a drain dirty.
            return
        raise DriverError('native-control-requires-paused-empty-engine')

    def _engine_paused(self):
        engine = self.engine
        paused = getattr(engine, 'is_scheduler_paused', None)
        return callable(paused) and paused() is True

    def _engine_quiescent(self):
        if not self._engine_paused():
            return False
        scheduler = getattr(self.engine, 'scheduler', None)
        # The pinned scheduler deliberately reports zero under PAUSED_ALL;
        # actual registered requests still exist and cannot be reconfigured.
        requests = getattr(scheduler, 'requests', None)
        if requests is not None and (type(requests) is not dict or requests):
            return False
        unfinished = getattr(scheduler, 'get_num_unfinished_requests', None)
        if callable(unfinished) and unfinished() != 0:
            return False
        return True

    def _issue_bootstrap_control(self):
        """Owner bootstrap for an explicit control intent.

        The generated tick only bootstraps lazily when a chunk is admitted;
        an explicit prepare must bind worker route ownership before any chunk
        exists, so it sends the same typed command from the same engine-owned
        identity and waits for the same queried all-rank acknowledgments.
        """
        if self._bootstrap_command is not None or self._inflight:
            raise DriverError('native-control-bootstrap-already-outstanding')
        command = BootstrapCommand(self.engine_epoch, self.ranks)
        self._bootstrap_command = command
        try:
            self._submit('bootstrap',
                         ('flash_s5_bootstrap', (command,), {}, None),
                         self.ranks)
        except BaseException:
            self._bootstrap_command = None
            raise

    def _cancel_control(self):
        control = self._control
        now = int(self._monotonic_ns())
        if control is None:
            return {'schema': 1, 'action': 'cancel', 'engine_epoch': self.engine_epoch,
                    'cancelled': False, 'reason': 'no-native-control-intent-pending',
                    'ownership_retained': False,
                    'observed_clock_ns': now,
                    'clock': self._clock_record(now)}
        control['cancelled'] = True
        return {'schema': 1, 'action': 'cancel', 'engine_epoch': self.engine_epoch,
                'cancelled': True, 'intent': control['action'],
                'intent_stage': control['stage'],
                'ownership_retained': True,
                'reason': ('the engine keeps worker/route ownership until the typed '
                           'reply or the absolute deadline settles the intent'),
                'observed_clock_ns': now,
                'clock': self._clock_record(now)}

    def _read_control(self, action, command):
        now = int(self._monotonic_ns())
        if action in ('status', 'snapshot'):
            return self._status_record(now)
        if action == 'transition_acks':
            rows = self.transition_ack_rows()
            return {'schema': 1, 'action': action, 'engine_epoch': self.engine_epoch,
                    'complete': set(self._last_transition_acks) == set(self.ranks),
                    'transition_pending': self._transition is not None,
                    'acks': rows, 'observed_clock_ns': now,
                    'clock_basis': self.clock_basis}
        if action == 'bootstrap_acks':
            rows = self.bootstrap_ack_rows()
            return {'schema': 1, 'action': action, 'engine_epoch': self.engine_epoch,
                    'complete': set(self._last_bootstrap_acks) == set(self.ranks),
                    'bootstrapped': self._bootstrapped,
                    'acks': rows, 'observed_clock_ns': now,
                    'clock_basis': self.clock_basis}
        if action == 'overlap_evidence':
            record = self.overlap_evidence(command.get('start_ns'),
                                           command.get('end_ns'))
            record.update({'action': action, 'clock': self._clock_record(now),
                           'observed_clock_ns': now})
            return record
        raise DriverError('native-control-read-action-not-implemented')

    # -- queried status and typed acknowledgments ------------------------
    def _native_admission_record(self):
        """The admission controller's typed counts and newest refusal (native37)."""
        query = getattr(self.controller, 'native_admission_record', None)
        return query() if callable(query) else None

    def note_stock_step_skipped(self):
        """Engine hook bypassed one stock step; utility pumps do not count."""
        self._stock_steps_skipped += 1

    def _status_record(self, now=None):
        result = self._status_record_base(now)
        policy = getattr(self, 'auto_arm', None)
        if policy is not None:
            result['auto_arm'] = policy.snapshot()
        if self.progress is not None:
            result['native_progress'] = self.progress.snapshot()
        if self.qsa_fused is not None:
            result['native_qsa_fused'] = self.qsa_fused.snapshot()
        if self.layer_groups is not None:
            result['native_layer_groups'] = self.layer_groups.snapshot()
        return result

    def _status_record_base(self, now=None):
        now = int(self._monotonic_ns()) if now is None else now
        epochs = set(self._storage_epochs.values())
        census = self._last_census
        return {
            'schema': 1,
            'mode': 'native-overlap',
            'control_path': self.control_path.snapshot(),
            'stock_steps_skipped': self._stock_steps_skipped,
            'stock_prefix_deferrals': self.controller.leases.stock_prefix_deferrals,
            'native_admission': self._native_admission_record(),
            'engine_epoch': self.engine_epoch,
            'ranks': list(self.ranks),
            'route': 'split' if self._split_ready else 'full',
            'route_generation': self.route_generation,
            'bootstrapped': bool(self._bootstrapped),
            'startup_owner_ready': getattr(self.engine, '_flash_s5_startup_ready', None),
            'transition_pending': self._transition is not None,
            'bootstrap_pending': self._bootstrap_command is not None,
            'return_pending': self._return_pending,
            'active': bool(self.active),
            'live': bool(self.live),
            'controller_active': bool(self.controller.active),
            'retained_row': self.retained_row(),
            'owned_work': {
                # A retained request row is distinct from an outstanding native
                # chunk, a GPU replay and a paused ordinary batch.  These
                # queried facts are reported separately, never inferred from
                # ``active`` or a scheduler-queue size.
                'chunk': self.tx is not None,
                'parked_admission': bool(self.controller.active),
                'retained_request_row': self._retained is not None,
                'route_exchange': self._route_exchange() is not None,
                'return_pending': self._return_pending,
                'control_intent': (None if self._control is None
                                   else self._control['action']),
            },
            'chunk': None if self.tx is None else self.tx.txn_seq,
            'native_work_identity': self.native_work_identity(),
            'last_native_closure': copy.deepcopy(self._last_native_closure),
            'source_leases': self.controller.leases.shutdown(),
            'phase': self.phase,
            'paused': (self.engine is not None
                       and callable(getattr(self.engine, 'is_scheduler_paused', None))
                       and self.engine.is_scheduler_paused() is True),
            'scheduler_requests': (self.engine.scheduler.get_num_unfinished_requests()
                                   if self.engine is not None
                                   and callable(getattr(self.engine.scheduler,
                                                        'get_num_unfinished_requests', None))
                                   else None),
            'storage_epoch': next(iter(epochs)) if len(epochs) == 1 else None,
            'storage_epochs_by_rank': {rank: self._storage_epochs[rank]
                                       for rank in sorted(self._storage_epochs)},
            'decode_watermarks_by_rank': {rank: self._decode_watermarks[rank]
                                          for rank in sorted(self._decode_watermarks)},
            'reports': sorted(int(rank) for rank in self.reports),
            'schedule_profile_mismatches': self._profile_mismatches,
            'schedule_profile_timeouts': self._profile_timeouts,
            'schedule_profile_refusals': self._profile_refusal_receipt(),
            'qsa_control': self.qsa_control.snapshot(),
            'native43_boundary': copy.deepcopy(self._native43),
            'control_pending': None if self._control is None else self._control['action'],
            'admission_intent': self.admission_intent(),
            'control_route': (None if self._control_route is None
                              else dict(self._control_route)),
            'workers': census,
            'workers_observed_ns': self._last_census_observed_ns,
            'journal': {'entries': len(self._journal),
                        'dropped': self._journal_dropped,
                        'limit': self.journal_limit,
                        'clock_basis': self.clock_basis},
            'clock': self._clock_record(now),
            'observed_clock_ns': now,
        }

    def transition_ack_rows(self):
        rows = []
        for rank in sorted(self._last_transition_acks):
            ack = self._last_transition_acks[rank]
            rows.append({
                'typed': type(ack).__name__,
                'worker_rank': ack.worker_rank,
                'engine_epoch': ack.engine_epoch,
                'expected_route_generation': ack.expected_route_generation,
                'route_generation': ack.route_generation,
                'applied_route_generation': ack.applied_route_generation,
                'target_bank': ack.target_bank,
                'applied_bank': ack.applied_bank,
                'applied_storage_epochs': list(ack.applied_storage_epochs),
                'decode_replay_watermark': ack.decode_replay_watermark,
                # The worker's RouteController drained-stream count, never an echo.
                'drained_streams': ack.drained_streams,
                'observed_ns': self._last_transition_observed_ns,
                'clock_basis': self.clock_basis,
            })
        return rows

    def bootstrap_ack_rows(self):
        rows = []
        for rank in sorted(self._last_bootstrap_acks):
            ack = self._last_bootstrap_acks[rank]
            rows.append({
                'typed': type(ack).__name__,
                'worker_rank': ack.worker_rank,
                'engine_epoch': ack.engine_epoch,
                'bound': ack.bound,
                'incarnation': ack.incarnation,
                'lane_worker_rank': ack.lane.worker_rank,
                'storage_epoch': ack.lane.storage_epoch,
                'route_generation': ack.route_generation,
                'decode_replay_watermark': ack.decode_replay_watermark,
                'observed_ns': self._last_bootstrap_observed_ns,
                'clock_basis': self.clock_basis,
            })
        return rows

    # -- control state machine -------------------------------------------
    def control_tick(self):
        """One bounded engine tick for a registered control intent.

        Consumes what the owned reply transport already holds and advances the
        intent by at most one stage.  It never polls a worker, never waits and
        never pumps a decode reply: the reply is taken from the same transport
        the stock step consumes on a later tick.
        """
        self._drain()
        self.pump_control()

    def pump_control(self):
        """Advance the one registered intent; bounded, never waits."""
        control = self._control
        if control is None:
            return
        try:
            self._advance_control(control)
        except Backpressure:
            return
        except BaseException as error:
            retain = (self._route_exchange() is not None
                      or self._return_pending is not None
                      or isinstance(error, TransportError))
            self._fail_control(
                'native-control-failed:%s:%s' % (type(error).__name__,
                                                 str(error)[:160]),
                retain=retain)
            if retain:
                raise

    def _advance_control(self, control):
        if (self._control_deadline is not None
                and self.clock() > self._control_deadline):
            raise DriverError('native-control-deadline-expired')
        if (self._transition is not None and self._transition_retry
                and not self._inflight):
            # The reviewed exact-command transition retry, shared with the
            # engine's own route drive: one absolute deadline, one command.
            self._send_transition()
            return
        inner = control.get('inner')
        if inner is not None:
            self._advance_inner(control, inner)
            return
        action = control['action']
        if action in CONTROL_WORKER_ACTIONS:
            if control['stage'] == 'pending':
                self._submit_worker_control(control, self._worker_command(control))
                return
            if self._control_reply is None:
                return
            reply, self._control_reply = self._control_reply, None
            self._settle_worker_control(control, reply)
            return
        if action == 'bootstrap_engine_owner':
            if self._bootstrapped:
                self._settle_route_control(control)
                return
            if self._bootstrap_command is None and not self._inflight:
                self._issue_bootstrap_control()
            return
        if action == 'enter_split':
            if not self._bootstrapped:
                if self._bootstrap_command is None and not self._inflight:
                    self._issue_bootstrap_control()
                return
            if (set(self._storage_epochs) != set(self.ranks)
                    or set(self._decode_watermarks) != set(self.ranks)):
                control['inner'] = {'action': 'census', 'stage': 'submit',
                                    'command': self._worker_command(control, 'census')}
                return
            if self._split_ready and self._route_exchange() is None:
                self._settle_route_control(control)
                return
            if self._route_exchange() is None and not self._inflight:
                try:
                    self._issue_transition(ROUTE_BANK_SPLIT)
                except E3Pending:
                    return
            return
        if action == 'leave_split':
            if not self._split_ready:
                if self._route_exchange() is None:
                    self._settle_route_control(control)
                return
            if self._route_exchange() is None and not self._inflight:
                try:
                    self._issue_transition(ROUTE_BANK_FULL)
                except E3Pending:
                    return
            return
        raise DriverError('native-control-action-not-implemented')

    def _worker_command(self, control, action=None):
        """Build the worker command from engine-owned identity, not client echo."""
        action = action or control['action']
        command = {'action': action, 'engine_epoch': self.engine_epoch,
                   'route_generation': self.route_generation}
        source = control['command']
        if action == 'poll_native':
            limit = source.get('limit', 8)
            if type(limit) is not int or not 1 <= limit <= 64:
                raise DriverError('native-poll-limit-invalid')
            command['limit'] = limit
        if action == 'drain':
            budget = source.get('budget', 32)
            if type(budget) is not int or not 1 <= budget <= 64:
                raise DriverError('native-drain-budget-invalid')
            command['budget'] = budget
        if action == 'qualification':
            payload = source.get('payload') or {}
            if type(payload) is not dict:
                raise DriverError('native-qualification-payload-not-an-object')
            bounded = _bounded_value(payload)
            if type(bounded) is not dict:
                raise DriverError('native-qualification-payload-unbounded')
            # The payload is copied first and the engine-owned identity is
            # written last: a payload can never rename the action or the epoch.
            command = dict(bounded)
            command.update({'action': source.get('worker_action'),
                            'engine_epoch': self.engine_epoch,
                            'route_generation': self.route_generation})
        return command

    def _submit_worker_control(self, control, command):
        if self._control_seq is not None:
            raise DriverError('native-control-command-already-outstanding')
        remaining = self._control_deadline - self.clock()
        if remaining <= 0:
            raise DriverError('native-control-deadline-expired')
        sequence = self._submit(
            'native_overlap',
            ('flash_s5_native_overlap', (command,), {}, None),
            self.ranks, timeout_s=remaining)
        control['stage'] = 'reply'
        self._control_seq = sequence

    def _advance_inner(self, control, inner):
        if inner['stage'] == 'submit':
            self._submit_worker_control(control, inner['command'])
            inner['stage'] = 'reply'
            return
        if self._control_reply is None:
            return
        reply, self._control_reply = self._control_reply, None
        rows = self._validate_worker_rows(reply, inner['action'])
        self._apply_census_rows(rows)
        control['inner'] = None
        self._control_seq = None

    def _validate_worker_rows(self, reply, action):
        """One typed, complete, same-epoch row per rank from the owned reply."""
        from native_overlap_worker import (
            ROW_BOOL_FIELDS,
            ROW_FIELDS,
            ROW_INT_FIELDS,
            ROW_LIST_FIELDS,
            ROW_NULLABLE,
            ROW_STR_FIELDS,
        )
        if type(reply) not in (list, tuple):
            raise DriverError('native worker reply is not a row list')
        expected = set(ROW_FIELDS)
        rows = []
        seen = set()
        for item in reply:
            if type(item) is not dict or set(item) != expected:
                raise DriverError('native worker row is not the exact typed schema')
            for key in ROW_INT_FIELDS:
                if type(item[key]) is not int or item[key] < 0:
                    raise DriverError('native worker row integer invalid: ' + key)
            for key in ROW_STR_FIELDS:
                if type(item[key]) is not str or not item[key]:
                    raise DriverError('native worker row string invalid: ' + key)
            for key in ROW_BOOL_FIELDS:
                if type(item[key]) is not bool:
                    raise DriverError('native worker row boolean invalid: ' + key)
            for key in ROW_LIST_FIELDS:
                if type(item[key]) is not list:
                    raise DriverError('native worker row list invalid: ' + key)
            for key in ROW_NULLABLE:
                if item[key] is not None and type(item[key]) is not dict:
                    raise DriverError('native worker row mapping invalid: ' + key)
            if item['action'] not in (action, 'census'):
                raise DriverError('native worker row answered a different action')
            if item['schema'] != 1 or item['engine_epoch_ok'] is not True:
                raise DriverError('native worker row is not a same-epoch reading')
            rank = item['worker_rank']
            if rank not in self.ranks or rank in seen:
                raise DriverError('native worker reply ranks are duplicated or unknown')
            seen.add(rank)
            if item['engine_epoch'] != self.engine_epoch:
                raise DriverError('native worker row engine epoch differs')
            if item['route_generation'] != self.route_generation:
                raise DriverError('native worker row route generation differs')
            known = self._storage_epochs.get(rank)
            if known is not None and item['storage_epoch'] != known:
                raise DriverError('native worker row storage epoch differs')
            if item['decode_eligible'] > item['live_decode_requests']:
                raise DriverError('native worker row decode census exceeds its live requests')
            if item['live_batch_rows'] > item['live_decode_requests']:
                raise DriverError('native worker row batch census exceeds its live requests')
            if item['retained_active_rows'] > item['retained_rows']:
                raise DriverError('native worker row retained-active count exceeds retained rows')
            rows.append(dict(item))
        if seen != set(self.ranks):
            raise DriverError('native worker reply does not cover every rank')
        return sorted(rows, key=lambda row: row['worker_rank'])

    def _apply_native_overlap(self, values, sequence):
        if self._control is None:
            self._record('control_orphan', sequence=sequence,
                         reason='no-control-intent-registered')
            return
        if sequence != self._control_seq:
            self._record('control_orphan', sequence=sequence,
                         expected=self._control_seq,
                         reason='reply-sequence-differs')
            return
        if self._control_reply is not None:
            raise DriverError('native control already holds an unclaimed reply')
        self._control_reply = values

    def _control_notice(self, kind):
        control = self._control
        if control is None:
            return
        control['notices'].append(kind)

    def _apply_census_rows(self, rows):
        for row in rows:
            self._storage_epochs[row['worker_rank']] = row['storage_epoch']
            self._decode_watermarks[row['worker_rank']] = row['decode_replay_watermark']
        self._last_census = rows
        self._last_census_observed_ns = int(self._monotonic_ns())
        self._record('worker_census', ranks=[row['worker_rank'] for row in rows],
                     storage_epochs=[row['storage_epoch'] for row in rows],
                     decode_replay_watermarks=[row['decode_replay_watermark']
                                               for row in rows],
                     pending_native_replays=[row['pending_native_replays']
                                             for row in rows],
                     retained_rows=[row['retained_rows'] for row in rows])

    def _settle_worker_control(self, control, reply):
        action = control['action']
        rows = self._validate_worker_rows(reply, action)
        self._apply_census_rows(rows)
        result = {'schema': 1, 'action': action,
                  'engine_epoch': self.engine_epoch,
                  'ranks': list(self.ranks),
                  'workers': rows,
                  'clock': self._clock_record(int(self._monotonic_ns())),
                  'clock_basis': self.clock_basis,
                  'snapshot': self._status_record()}
        if action in ('barrier', 'drain'):
            for row in rows:
                if not row['barrier_clean']:
                    raise DriverError('native-barrier-not-clean:rank-%d'
                                      % row['worker_rank'])
            if action == 'drain':
                for row in rows:
                    if (row['pending_native_replays'] or row['retained_rows']
                            or row['pending_prefill_streams']
                            or row['pending_output_copies']):
                        raise DriverError('native-drain-owned-work-remains:rank-%d'
                                          % row['worker_rank'])
            self._record('worker_barrier', action=action,
                         ranks=[row['worker_rank'] for row in rows],
                         drained_streams=[row['drained_streams'] for row in rows])
        if action == 'poll_native':
            self._record('worker_poll_native',
                         completed=[row['polled_replays'] for row in rows],
                         pending=[row['pending_native_replays'] for row in rows])
        if action == 'qualification':
            result['qualification'] = rows
            result['graded_by_engine'] = False
        self._finish_control(control, result)

    def _settle_route_control(self, control):
        action = control['action']
        route_after = 'split' if self._split_ready else 'full'
        if action == 'enter_split' and self._split_ready:
            # Explicit control now owns this route across pauses and chunks.
            self._control_route = {'owner': 'native-control',
                                   'intent': action,
                                   'engine_epoch': self.engine_epoch,
                                   'route_generation': self.route_generation,
                                   'since_ns': int(self._monotonic_ns())}
        elif action == 'leave_split' and not self._split_ready:
            self._control_route = None
        result = {'schema': 1, 'action': action,
                  'engine_epoch': self.engine_epoch,
                  'bootstrapped': bool(self._bootstrapped),
            'startup_owner_ready': getattr(self.engine, '_flash_s5_startup_ready', None),
                  'route_before': control['start_route'],
                  'route_after': 'split' if self._split_ready else 'full',
                  'route_generation_before': control['start_route_generation'],
                  'route_generation_after': self.route_generation,
                  'changed': (control['start_route'] !=
                              ('split' if self._split_ready else 'full')),
                  'admission_intent': self.admission_intent(),
                  'bootstrap_acks': self.bootstrap_ack_rows(),
                  'transition_acks': self.transition_ack_rows(),
                  'typed_acks': True,
                  'clock': self._clock_record(int(self._monotonic_ns())),
                  'clock_basis': self.clock_basis,
                  'snapshot': self._status_record()}
        self._record('control_route_settled', action=action,
                     route_after=result['route_after'],
                     route_generation=result['route_generation_after'],
                     transition_acks=len(result['transition_acks']),
                     bootstrap_acks=len(result['bootstrap_acks']))
        self._finish_control(control, result)

    def _finish_control(self, control, result):
        future = self._control_future
        cancelled = bool(control.get('cancelled'))
        self._control = None
        self._control_future = None
        self._control_seq = None
        self._control_deadline = None
        self._control_reply = None
        result['cancelled_by_client'] = cancelled
        if cancelled:
            result['client_await_abandoned'] = True
        if future is not None and not future.done():
            future.set_result(result)

    def _fail_control(self, reason, *, retain):
        control = self._control
        future = self._control_future
        self._control = None
        self._control_future = None
        self._control_seq = None
        self._control_deadline = None
        self._control_reply = None
        if control is not None:
            self._record('control_failed', attempted=control['action'],
                         retained=bool(retain))
        if future is not None and not future.done():
            future.set_exception(DriverError(reason))
        if retain:
            self.error = reason + '; route or worker ownership retained'

    # -- host-monotonic journal ------------------------------------------
    def _clock_record(self, sample_ns):
        record = {'basis': self.clock_basis, 'sample_ns': sample_ns,
                  'proc': 'engine'}
        try:
            import os
            with open('/proc/sys/kernel/random/boot_id', 'r') as handle:
                boot_id = handle.read().strip()
            record['boot_id'] = boot_id or None
            record['time_namespace'] = os.readlink('/proc/self/ns/time')
        except OSError as exc:
            record['boot_id'] = None
            record['time_namespace'] = None
            record['unavailable'] = type(exc).__name__
        return record

    def _record(self, entry_kind, **fields):
        if type(entry_kind) is not str or not entry_kind:
            raise DriverError('journal kind required')
        if len(self._journal) == self._journal.maxlen:
            self._journal_dropped += 1
        entry = {'kind': entry_kind,
                 'observed_ns': int(self._monotonic_ns()),
                 'clock_basis': self.clock_basis,
                 'engine_epoch': int(self.engine_epoch),
                 'route_generation': int(self.route_generation),
                 'route': 'split' if self._split_ready else 'full'}
        for key, value in fields.items():
            # Only source-validated report vectors get the lossless page schema.
            # Other metadata keeps the original strict64-element refusal.
            if (entry_kind == 'report' and key in ('completed_op_seqs', 'pending_op_seqs', 'proofs')
                    and type(value) in (list, tuple) and len(value) > 64):
                # Preserve each original leaf's depth/type/string limits. The
                # explicit page tree, separately width/height bounded by its
                # encoder, replaces only the overlong outer source vector.
                entry[key] = encode_report_vector([_bounded_value(item, 1) for item in value])
            else:
                entry[key] = _bounded_value(value)
        self._journal.append(entry)
        self.measurement_census.ingest(entry)
        return entry

    # -- overlap evidence from the journal -------------------------------
    def overlap_evidence(self, start_ns, end_ns):
        """Bounded host-monotonic evidence over real journaled observations."""
        if type(start_ns) is not int or type(end_ns) is not int:
            raise DriverError('overlap-window-not-integers')
        if start_ns <= 0 or end_ns <= 0 or end_ns <= start_ns:
            raise DriverError('overlap-window-empty-or-negative')
        if end_ns - start_ns > MAX_EVIDENCE_WINDOW_NS:
            raise DriverError('overlap-window-exceeds-bound')
        now = int(self._monotonic_ns())
        # The client samples are host monotonic values; a value after the
        # engine's own sample (or one far behind it) proves the two clocks are
        # not the same domain, and the window is refused rather than shifted.
        if end_ns > now:
            raise DriverError('overlap-journal-clock-mismatch:end-after-engine-sample')
        if now - end_ns > MAX_EVIDENCE_WINDOW_NS:
            raise DriverError('overlap-journal-clock-mismatch:end-before-engine-sample')
        entries = [entry for entry in self._journal
                   if start_ns <= entry['observed_ns'] <= end_ns]
        if not entries:
            raise DriverError('overlap-journal-empty-window')
        stale = [entry for entry in entries
                 if entry['clock_basis'] != self.clock_basis
                 or entry['engine_epoch'] != self.engine_epoch]
        if stale:
            raise DriverError('overlap-journal-stale-epoch-or-clock')
        for first, second in zip(entries, entries[1:]):
            if second['observed_ns'] < first['observed_ns']:
                raise DriverError('overlap-journal-not-monotonic')
        complete = (self._journal_dropped == 0
                    or (self._journal and self._journal[0]['observed_ns'] <= start_ns))

        chunks = {}
        completions = []
        receipts = []
        handles = []
        frontiers = []
        cancels = []
        for entry in entries:
            kind = entry['kind']
            txn = entry.get('transaction')
            if kind == 'chunk_open' and type(txn) is int:
                chunk = chunks.setdefault(txn, {})
                chunk.update({key: entry.get(key) for key in
                              ('request', 'request_generation', 'chunk_start',
                               'chunk_end', 'state_epoch', 'committed_end',
                               'prompt_end', 'remaining_tokens', 'terminal')})
                chunk['begin_ns'] = entry['observed_ns']
                chunk['segments'] = []
                chunk['routes'] = set()
                chunk['state_written'] = {}
                chunk['reports'] = 0
            elif kind == 'segment' and type(txn) is int and txn in chunks:
                chunks[txn]['segments'].append(entry)
                chunks[txn]['routes'].add(entry.get('route'))
            elif kind == 'report' and type(txn) is int and txn in chunks:
                chunks[txn]['reports'] += 1
                if entry.get('state_written') is True:
                    chunks[txn]['state_written'][entry.get('worker_rank')] = True
                for proof in decode_report_vector(entry.get('proofs') or []):
                    if type(proof) is not dict:
                        continue
                    completions.append({
                        'request': chunks[txn].get('request'),
                        'worker_rank': entry.get('worker_rank'),
                        'queried': True,
                        'completed_sequence': entry.get('completed_sequence'),
                        'state_written': entry.get('state_written') is True,
                        'observed_ns': entry['observed_ns'],
                        'clock_basis': entry['clock_basis'],
                        'proofs': [{'kind': proof.get('kind'),
                                    'op_seq': proof.get('op_seq'),
                                    'completion_seq': proof.get('completion_seq'),
                                    # The engine tick's observation of a queried
                                    # report; never a device timestamp.
                                    'completed_ns': entry['observed_ns'],
                                    'time_basis': EVIDENCE_TIME_BASIS}],
                    })
            elif kind == 'chunk_commit' and type(txn) is int and txn in chunks:
                chunks[txn]['commit_ns'] = entry['observed_ns']
                chunks[txn]['worker_releases'] = entry.get('worker_releases')
            elif kind == 'pp_receipt':
                receipts.append({'transaction': txn, 'sender_rank': entry.get('sender_rank'),
                                 'seq': entry.get('seq'),
                                 'completion_seq': entry.get('completion_seq'),
                                 'observed_ns': entry['observed_ns'],
                                 'clock_basis': entry['clock_basis']})
            elif kind == 'source_hold':
                handles.append({'transaction': txn,
                                'description': entry.get('description'),
                                'retired': entry.get('retired'),
                                'observed_ns': entry['observed_ns'],
                                'clock_basis': entry['clock_basis']})
            elif kind == 'committed_frontier':
                frontiers.append({'transaction': txn,
                                  'frontier': entry.get('frontier'),
                                  'observed_ns': entry['observed_ns'],
                                  'clock_basis': entry['clock_basis']})
            elif kind == 'cancel':
                cancels.append({'transaction': txn,
                                'never_submitted': entry.get('never_submitted'),
                                'closed_ranks': entry.get('closed_ranks'),
                                'observed_ns': entry['observed_ns'],
                                'clock_basis': entry['clock_basis']})

        segments = []
        useful_tokens = 0
        commits = 0
        wall_ns = 0
        for txn in sorted(chunks):
            chunk = chunks[txn]
            begin = chunk.get('begin_ns')
            commit = chunk.get('commit_ns')
            written = chunk.get('state_written') or {}
            all_written = set(written) == set(self.ranks) and all(written.values())
            routes = chunk.get('routes') or set()
            transport = 'native-split' if routes == {'split'} else 'serialized'
            tokens = None
            if type(chunk.get('chunk_start')) is int and type(chunk.get('chunk_end')) is int:
                tokens = chunk['chunk_end'] - chunk['chunk_start']
            accepted = (type(commit) is int and type(begin) is int
                        and commit > begin and all_written
                        and chunk.get('remaining_tokens') is not None)
            if accepted:
                commits += 1
                useful_tokens += tokens or 0
                wall_ns += commit - begin
            segments.append({
                'transaction': txn,
                'request': chunk.get('request'),
                'prefill': ('fresh' if chunk.get('committed_end') == 0 else 'append'),
                'committed_end': chunk.get('committed_end'),
                'prompt_end': chunk.get('prompt_end'),
                'chunk_start': chunk.get('chunk_start'),
                'chunk_end': chunk.get('chunk_end'),
                'chunk_tokens': tokens,
                'terminal': chunk.get('terminal'),
                'start_monotonic_ns': begin,
                'end_monotonic_ns': commit,
                'transport': transport,
                'accepted': accepted,
                'state_write': all_written,
                'state_write_ranks': sorted(int(rank) for rank in written),
                'report_count': chunk.get('reports'),
                'segment_count': len(chunk.get('segments') or []),
                'decode_progress': None,
                'decode_progress_basis': 'not-queried-engine-side',
                'worker_releases': chunk.get('worker_releases'),
                'clock_basis': self.clock_basis,
                'engine_epoch': self.engine_epoch,
            })

        route_generations = sorted({entry['route_generation'] for entry in entries})
        epochs = set(self._storage_epochs.values())
        incarnations = sorted({row['incarnation'] for row in (self._last_census or [])
                               if type(row.get('incarnation')) is str})
        record = {
            'schema': 1,
            'start_ns': start_ns,
            'end_ns': end_ns,
            'observed_clock_ns': now,
            'clock_basis': self.clock_basis,
            'clock': self._clock_record(now),
            'engine_epoch': self.engine_epoch,
            'route': 'split' if self._split_ready else 'full',
            'route_generation': self.route_generation,
            'route_generations_in_window': route_generations,
            'storage_epoch': next(iter(epochs)) if len(epochs) == 1 else None,
            'worker_incarnation': incarnations[0] if len(incarnations) == 1 else None,
            'segments': segments,
            'completions': completions,
            'queried_reports': [],
            'pp_consumer_receipts': receipts,
            'source_holds': handles,
            'committed_frontiers': frontiers,
            'cancellations': cancels,
            'errors': [],
            'decode_progress': None,
            'decode_progress_basis': ('decode completion is a client-visible delta; '
                                      'the engine journal does not infer it'),
            'prefill': {
                'useful_completed_tokens': useful_tokens,
                'completed_prefills': commits,
                'prefill_wall_ns': wall_ns,
                'prefill_wall_s': round(wall_ns / 1e9, 9) if wall_ns else None,
                'prompt_tokens': None,
                'prompt_tokens_basis': 'per-request prompt length is not a driver fact',
                'basis': ('engine journal: admitted chunk token spans and host-monotonic '
                          'chunk open..commit observations; not device elapsed time'),
            },
            'serialized_transport': any(segment['transport'] != 'native-split'
                                        for segment in segments),
            'journal': {'entries': len(self._journal), 'dropped': self._journal_dropped,
                        'selected': len(entries), 'limit': self.journal_limit,
                        'complete_in_window': bool(complete)},
            'workers': self._last_census,
            'workers_observed_ns': self._last_census_observed_ns,
            'measurement': self.measurement_census.evidence(start_ns, end_ns, now_ns=now),
            'control_path': self.control_path.snapshot(),
            'transition_acks': self.transition_ack_rows(),
            'native_owned': bool(self.active),
        }
        record['measurement']['schedule_profile'] = {
            'mismatches': self._profile_mismatches, 'timeouts': self._profile_timeouts,
            'refusals': self._profile_refusal_receipt(),
            'basis': 'engine-lifetime-monotonic-count'}
        record['measurement']['qsa_control'] = self.qsa_control.snapshot()
        record['measurement']['stock_steps_skipped'] = {
            'count': self._stock_steps_skipped,
            'basis': 'engine-lifetime hook bypasses; subtract matching status snapshots'}
        record['measurement']['stock_prefix_deferrals'] = {
            'count': self.controller.leases.stock_prefix_deferrals,
            'basis': 'engine-lifetime stock cached-prefix request deferrals; subtract matching status snapshots'}
        record['native_admission'] = self._native_admission_record()
        record['measurement']['native_admission'] = record['native_admission']
        reasons = []
        if not complete:
            reasons.append('journal-evicted-before-window')
        if not segments:
            reasons.append('no-admitted-segment-in-window')
        if not any(segment['accepted'] for segment in segments):
            reasons.append('no-accepted-segment-in-window')
        if not completions:
            reasons.append('no-queried-worker-completion-in-window')
        record['reasons'] = reasons
        record['construction_valid'] = not reasons
        return record

    # -- green/full route transition --------------------------------------
    def issue_transition(self, *, expected_route_generation, target_bank,
                         storage_epochs, decode_replay_watermarks):
        if self._transition is not None or self._inflight:
            raise DriverError('a transition or command is already outstanding')
        if expected_route_generation != self.route_generation:
            raise DriverError('transition starts from a stale route generation')
        command = TransitionCommand(
            engine_epoch=self.engine_epoch,
            expected_route_generation=expected_route_generation,
            route_generation=expected_route_generation + 1,
            target_bank=target_bank,
            storage_epochs=tuple(storage_epochs),
            decode_replay_watermarks=tuple(decode_replay_watermarks))
        self._transition = command
        self._transition_retry = True
        self._transition_deadline = self.clock() + self.reply_timeout_s
        self._send_transition()
        return command

    def _send_transition(self):
        command = self._transition
        if command is None or self._inflight:
            raise DriverError('transition retry has no exclusive command ownership')
        remaining = self._transition_deadline - self.clock()
        if remaining <= 0:
            raise DriverError('all-rank transition drain deadline expired; ownership retained')
        self._submit('transition', ('flash_s5_transition', (command,), {}, None),
                     self.ranks, timeout_s=remaining)
        self._transition_retry = False

    def accept_transition_ack(self, ack):
        raise DriverError('transition needs the complete all-rank reply, not one ack')


__all__ = ['DriverError', 'NativeSegmentDriver', 'PPTransferLedger',
           'ParkedBatch', 'PrefillAdmission', 'SegmentDriver']
