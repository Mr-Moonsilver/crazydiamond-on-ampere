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
    CompletionKind,
    CompletionProof,
    DependencyRole,
    E3ContractError,
    E3Pending,
    LaneRole,
    OperationTicket,
    TransactionId,
)
from lane_comm import PPConsumerReceipt
from lane_state import BatchRow, MetadataSnapshot, _freeze
from reply_transport import Backpressure
from scheduler_hooks import LeaseCategory, PrefillAdmissionController
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
)


def _wire_ready():
    if SegmentDescriptor is None:
        raise DriverError('wire-v1 is not packaged; the native driver cannot run')


# Reported when a chunk cannot be parked; the engine then runs the stock path.
_STOP_KINDS = ('abort', 'cancel', 'disconnect', 'preempt', 'reset', 'pause',
               'shutdown')


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
                 stop_tick_budget=8):
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
        self.route_generation = 0
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
        self.stop_kind = None
        self.error = None
        self._inflight = {}
        self._retry = None
        self._transition = None
        self._transition_retry = False
        self._transition_deadline = None
        self._last_transition_acks = {}
        self.pp_send_seq = None
        self.pp_receive_seq = None
        self.receiver_ticket = None
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
        # Engine-owner bootstrap and route completion.  These survive a chunk
        # reset: the worker storage epoch and replay census are worker-stable.
        self._bootstrapped = False
        self._bootstrap_command = None
        self._storage_epochs = {}
        self._decode_watermarks = {}
        self._split_ready = False
        self._return_pending = None

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
    def active(self):
        return (self.tx is not None or self.controller.active
                or self._return_pending is not None
                or self._transition is not None
                or self._bootstrap_command is not None)

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
        transport = getattr(self.executor, '_flash_s5_transport', None)
        if transport is None:
            raise DriverError('owned reply transport is not installed')
        sender = self._bounded()
        check = getattr(sender, 'check', None)
        if callable(check):
            check(envelope)      # backpressure before any expectation is registered
        sequence = transport.submit(kind, lambda _sequence: sender(envelope),
                                    ranks=tuple(ranks),
                                    timeout_s=(self.reply_timeout_s if timeout_s is None
                                               else timeout_s), aggregate=None)
        self._inflight[sequence] = kind
        return sequence

    def _drain(self):
        transport = getattr(self.executor, '_flash_s5_transport', None)
        if transport is None:
            return 0
        transport.poll(self.poll_budget)
        applied = 0
        for sequence, kind in list(self._inflight.items()):
            call = transport._calls.get(sequence)
            if call is None or not call.ready:
                continue
            value, _lag = transport.take(sequence)
            del self._inflight[sequence]
            self._apply(kind, value)
            applied += 1
        return applied

    def _apply(self, kind, values):
        if kind == 'bootstrap':
            self._apply_bootstrap(values)
        elif kind == 'open':
            self._apply_open(values)
        elif kind == 'bind':
            self._apply_bind(values)
        elif kind == 'segment':
            self._apply_segments(values)
        elif kind == 'poll':
            self._apply_poll(values)
        elif kind == 'receipt':
            self._apply_receipt(values)
        elif kind == 'cancel':
            self._apply_cancel(values)
        elif kind == 'release':
            self._apply_release(values)
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
            self._transition = None
            self._transition_retry = False
            self._transition_deadline = None
            self._last_transition_acks = {ack.worker_rank: ack for ack in acks}
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
        self.plan = self._build_plan()
        # The scheduler hold may only be validated for lanes that actually
        # received this admitted frame; bind their exact identities now.
        self.controller.note_source_owners(self.tx, self.lanes, self.layer_count)
        self._bindings = {}
        self._bind_index = 0
        self._pending_binding = None
        self.phase = 'bind'

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
                dict(kind=SegmentKind.PP_RECEIVE, stage=1, seq=seq),
                dict(kind=SegmentKind.PREFIX, stage=0)]
        for stage in (0, 1):
            if stage == 1:
                plan.append(dict(kind=SegmentKind.PREFIX, stage=1))
            for layer in range(*self.stage_layers[stage]):
                plan.append(dict(kind=SegmentKind.LAYER, stage=stage, layer=layer))
            plan.append(dict(kind=SegmentKind.SUFFIX, stage=stage))
            if stage == 0:
                plan.append(dict(kind=SegmentKind.PP_SEND, stage=0, seq=seq))
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
        if pending:
            self._retry = self._command
            for report in reports:
                self._ingest_report(report)
            return
        paused = {item.paused for item in accepted if item.accepted}
        cursors = {item.cursor for item in accepted if item.accepted}
        if len(paused) != 1 or len(cursors) != 1:
            raise DriverError('TP workers disagree on the layer continuation')
        representative = targets[0]
        self.last_ticket[descriptor.stage] = self.tickets[representative]
        if descriptor.kind is SegmentKind.PREPARE and self.bound_ticket is None:
            self.controller.bind(self.tickets[representative])
            self.bound_ticket = self.tickets[representative]
        if descriptor.kind is SegmentKind.PP_SEND:
            self.controller.note_pp_send(self.tx, descriptor.seq)
            for sender, ledger in self.pp_ledgers.items():
                ledger.note_sent(self.tx, descriptor.seq, sender)
            self.pp_send_ticket = self.tickets[representative]
        if descriptor.kind is SegmentKind.PP_RECEIVE:
            self.receiver_ticket = self.tickets[targets[0]]
        self._retry = None
        if paused != {True}:
            self.index += 1
        if self.index == len(self.plan):
            self.phase = 'finalize'
        # Completions piggybacked on this reply are host-queried facts too.
        # Publication stays in the drive path, never inside acceptance.
        for report in reports:
            self._ingest_report(report)

    def _ingest_report(self, report):
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
        self.controller.commit(self.tx, self._producer_proof())
        self._reset()
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
        self.release_request = ReleaseRequest(
            self.tx, self.admission.state_epoch,
            tuple((r, self.reports[r].completed_sequence) for r in self.ranks),
            self.admission.terminal)
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
    def advance(self):
        """One bounded tick. Returns no model output: the handback owns it."""
        if self.error is not None:
            raise DriverError(self.error)
        try:
            self._drain()
            if self.stop_kind is not None:
                self._drive_stop()
                return None
            if self._drive_route():
                # A route command is outstanding: the worker is draining or
                # switching streams/banks, so no stock work may be launched
                # into the old route.  The engine skips this step instead.
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

    # -- engine-owner bootstrap and route completion ----------------------
    def _route_exchange(self):
        """The route command whose reply is still owed, if any."""
        return self._bootstrap_command or self._transition

    def _drive_return(self):
        """One bounded step towards the drained all-rank full-bank return."""
        if self._route_exchange() is not None or self._inflight:
            return
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
        self.tx = admission.transaction
        self.admission = admission
        self.open_request = OpenRequest(admission, self.ranks,
                                        admission.state_epoch, 1)
        self.phase = 'open'
        self.plan = []
        self.index = 0
        self.reports = {}
        self.accepted_tickets = {}
        self._accepted_descriptors = {}
        self.bound_ticket = None
        self.pp_send_ticket = None
        self.receiver_ticket = None
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
            layer_start=layer, layer_end=None if layer is None else layer + 1,
            seq=entry.get('seq'))
        targets = self.stage_ranks[stage]
        tracker = self.trackers[targets[0]]
        op_seq = tracker.propose(descriptor)
        for rank in targets[1:]:
            if self.trackers[rank].propose(descriptor) != op_seq:
                raise DriverError('lane op_seq proposals diverged')
        command = SegmentCommand(descriptor, op_seq, targets)
        if kind is SegmentKind.PP_SEND:
            self.pp_send_seq = descriptor.seq
        if kind is SegmentKind.PP_RECEIVE:
            self.pp_receive_seq = descriptor.seq
        self._send_segment(command, targets)

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
            if not self._split_ready:
                return None
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
            self.advance()
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
            # Admission happened pre-schedule; the driver may not have begun.
            # Closing admissions retires the parked chunk without publishing.
            if self.controller.active:
                admission = self.controller.admission()
                self.controller.cancel(admission.transaction,
                                       never_submitted=True)
                self._reset()
                self._note_terminal_route('cancel')
            if self._return_pending is not None:
                self._drive_return()
                return
            # Nothing is owned and the full route is restored: stop is done.
            self.stop_kind = None
            return
        if self._receipts_pending():
            self._stop_round()
            self._next_receipt()
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

    def _stop_round(self):
        """Count one submitted stop round; refuse past the bound, keep ownership."""
        self._stop_ticks += 1
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

    def _retire_cancel(self, acks):
        """Terminal cancellation row: storage/request closure, no publication."""
        if set(acks) != set(self.ranks):
            raise DriverError('cancellation lacks an all-rank close disposition')
        if any(not ack.closed for ack in acks.values()):
            raise DriverError('cancellation retirement requires closed worker ownership')
        if self.bound_ticket is None:
            if self.accepted_tickets:
                raise DriverError('accepted operations exist without a bound producer')
            self.controller.cancel(self.tx, never_submitted=True)
            self._reset()
            self._note_terminal_route('cancel')
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
        self.controller.cancel(self.tx, proof=self._producer_proof())
        self._reset()
        self._note_terminal_route('cancel')

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
        boundary is never a route boundary.
        """
        self.tx = None
        self.admission = None
        self.phase = 'idle'
        self.plan = []
        self.index = 0
        self._retry = None
        self.stop_kind = None
        self._command = None
        self._targets = ()
        self.lanes = {}
        self.trackers = {}
        self.tickets = {}
        self.accepted_tickets = {}
        self._accepted_descriptors = {}
        self.reports = {}
        self.bound_ticket = None
        self.pp_send_ticket = None
        self.receiver_ticket = None
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
        if self.active:
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
