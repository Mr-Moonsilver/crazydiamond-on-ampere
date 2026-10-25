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
    DependencyRole,
    E3ContractError,
    E3Pending,
    OperationTicket,
    TransactionId,
)
from lane_state import BatchRow, MetadataSnapshot, _freeze
from scheduler_hooks import LeaseCategory

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


__all__ = ['DriverError', 'ParkedBatch', 'PrefillAdmission', 'SegmentDriver']
