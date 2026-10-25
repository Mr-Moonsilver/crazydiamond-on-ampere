"""C: bounded asynchronous prefill handoff, lifetimes and engine dispatch.

This module is the CPU-only contract C owns. It never imports torch or vLLM and
performs no I/O, so the state machine, the admission rules and the completion
ledger can be exercised with deterministic fakes. It is *not* an engine overlay:
the real completion backend, native shared-memory transport and CUDA/PP evidence
remain owner-run acceptance steps (see ``docs/e3-briefs/async-handoffs.md``).

Boundaries this module enforces, independently of the wire:

* An RPC acknowledgment (:meth:`HandoffDispatcher.acknowledge`) is not a GPU
  completion and never releases a source, arena or PP lease. Only A's
  :class:`~e3_types.CompletionProof` does that.
* Publication is exactly-once and only after every outstanding operation of the
  transaction has a recorded completion plus the explicit commit-role proof.
* One active bulk-prefill transaction globally in this increment. Decode remains
  ready while that transaction is parked; the dispatcher never waits for a
  whole prefill chunk.
* An ambiguous reply (timeout, transport failure, late/stale reply, malformed
  typed payload) quarantines the transaction. Late replies cannot mutate a
  reused allocation incarnation; storage release still needs every proof.
* Sparse-UVA fallback is never admitted into protected overlap. It is counted
  and serialized before admission, and a dense-DMA qualification is required.

Every telemetry record is metadata only: transaction/operation sequences, byte
counts, state names and durations. Request UIDs are never logged (only their
length), and no token, text, tensor or provider header is accepted.
"""
from dataclasses import dataclass, field
from enum import Enum
import math
import threading
import time
from collections import deque

from e3_types import (
    CompletionKind,
    CompletionProof,
    DependencyRole,
    E3ContractError,
    E3Pending,
    HISTORIES_PER_RANK,
    LaneId,
    LaneRole,
    MAX_ACTIVE_PREFILL,
    OperationTicket,
    PollResult,
    TransactionId,
)
from lane_state import (InventoryEntry, InventoryReport, PP_HIDDEN_BYTES,
                        estimated_e3_inventory)
from reply_transport import Backpressure, ReplyTransport, TransportError


class HandoffError(E3ContractError):
    """An async-handoff identity, admission or lifetime rule was violated."""


class ConfigurationError(HandoffError):
    """An overlap setting is unqualified, contradictory or incomplete."""


# --- declared settings ------------------------------------------------------

OVERLAP_ENV = 'FLASH_S5_OVERLAP'
TRANSPORT_ENV = 'FLASH_S5_TRANSPORT'
SPARSE_OVERLAP_ENV = 'FLASH_S5_SPARSE_OVERLAP'
COMMAND_SLOTS_ENV = 'FLASH_S5_COMMAND_SLOTS'
REPLY_TIMEOUT_ENV = 'FLASH_S5_REPLY_TIMEOUT_S'
ACK_TIMEOUT_ENV = 'FLASH_S5_ACK_TIMEOUT_S'
CHUNK_TOKENS_ENV = 'FLASH_S5_CHUNK_TOKENS'
BACKEND_ENV = 'FLASH_S5_COMPLETION_BACKEND'

QUALIFIED_TRANSPORT = 'dense_dma'
# Pinned MessageQueue default ring size (shm_broadcast.MessageQueue max_chunks).
PINNED_RING_SLOTS = 10
DEFAULT_COMMAND_SLOTS = PINNED_RING_SLOTS
DEFAULT_REPLY_TIMEOUT_S = 300.0
DEFAULT_ACK_TIMEOUT_S = 30.0
DEFAULT_CHUNK_TOKENS = 1024

# C's new allocation classes, kept separate from A's inventory so neither is
# mistaken for existing storage or for the unmet 512-MiB reserve.
PREFILL_PP_BUFFERS_BYTES = 2 * PP_HIDDEN_BYTES      # independent send + recv
COMMAND_METADATA_BYTES_PER_SLOT = 4096
TELEMETRY_RECORDS = 2048
TELEMETRY_BYTES_PER_RECORD = 192


class Route(str, Enum):
    DENSE_DMA = 'dense_dma'
    SERIAL = 'serial'


class Phase(str, Enum):
    IDLE = 'idle'
    OPEN = 'open'
    SUBMITTED = 'submitted'
    COMPLETING = 'completing'
    COMMITTED = 'committed'
    RETIRING = 'retiring'
    RETIRED = 'retired'
    QUARANTINED = 'quarantined'


class StopKind(str, Enum):
    ABORT = 'abort'
    CANCEL = 'cancel'
    DISCONNECT = 'disconnect'
    PREEMPT = 'preempt'
    RESET = 'reset'
    PAUSE = 'pause'
    SHUTDOWN = 'shutdown'


_STOP_KINDS = frozenset(k.value for k in StopKind)
_CONTROL_KINDS = frozenset({'open', 'poll', 'commit', 'cancel'})
# A stop that ends with its retired lease; pause/reset also resume after the
# engine's own drain.  Shutdown keeps refusing admission until the process exits.
_RESUMABLE_STOPS = frozenset({'abort', 'cancel', 'disconnect', 'preempt',
                              'reset', 'pause'})


def _positive_float(name, value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f'{name} must be a positive finite number')
    return float(value)


def _positive_int(name, value):
    if type(value) is not int or value < 1:
        raise ConfigurationError(f'{name} must be a positive int')
    return value


@dataclass(frozen=True)
class HandoffConfig:
    """Validated opt-in settings. Disabled is the identity configuration."""

    enabled: bool
    transport: str
    command_slots: int
    reply_timeout_s: float
    ack_timeout_s: float
    chunk_tokens: int

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ConfigurationError('enabled must be bool')
        if self.transport not in (QUALIFIED_TRANSPORT,):
            raise ConfigurationError('only the qualified dense-DMA transport is admitted')
        _positive_int('command_slots', self.command_slots)
        if self.command_slots > PINNED_RING_SLOTS:
            raise ConfigurationError(
                'command_slots exceeds the pinned broadcast ring; derive from transport state')
        _positive_float('reply_timeout_s', self.reply_timeout_s)
        _positive_float('ack_timeout_s', self.ack_timeout_s)
        _positive_int('chunk_tokens', self.chunk_tokens)

    @classmethod
    def from_env(cls, env):
        """Parse settings; anything unqualified fails closed while enabled.

        With ``FLASH_S5_OVERLAP`` unset or 0 the ordinary serial path is the
        identity result. Sparse-UVA overlap is rejected even when overlap is
        off because no configuration may silently turn it on later.
        """
        raw = env.get(OVERLAP_ENV, '0')
        if raw not in ('0', '1'):
            raise ConfigurationError(f'{OVERLAP_ENV} must be 0 or 1')
        sparse = env.get(SPARSE_OVERLAP_ENV, '0')
        if sparse not in ('0', '1'):
            raise ConfigurationError(f'{SPARSE_OVERLAP_ENV} must be 0 or 1')
        if sparse == '1':
            raise ConfigurationError('sparse-UVA fallback is not qualified for protected overlap')
        enabled = raw == '1'
        transport = env.get(TRANSPORT_ENV, QUALIFIED_TRANSPORT)
        if transport != QUALIFIED_TRANSPORT:
            raise ConfigurationError('unqualified overlap transport: ' + transport)
        def number(name, default, cast):
            value = env.get(name)
            if value is None:
                return default
            try:
                return cast(value)
            except (TypeError, ValueError):
                raise ConfigurationError(f'invalid {name}') from None
        slots = number(COMMAND_SLOTS_ENV, DEFAULT_COMMAND_SLOTS, int)
        reply_timeout = number(REPLY_TIMEOUT_ENV, DEFAULT_REPLY_TIMEOUT_S, float)
        ack_timeout = number(ACK_TIMEOUT_ENV, DEFAULT_ACK_TIMEOUT_S, float)
        chunk_tokens = number(CHUNK_TOKENS_ENV, DEFAULT_CHUNK_TOKENS, int)
        backend = env.get(BACKEND_ENV, 'native')
        if enabled and backend != 'native':
            raise ConfigurationError('enabled overlap requires the native completion backend')
        return cls(enabled, transport, slots, reply_timeout, ack_timeout, chunk_tokens)

    @classmethod
    def disabled(cls):
        return cls(False, QUALIFIED_TRANSPORT, DEFAULT_COMMAND_SLOTS,
                   DEFAULT_REPLY_TIMEOUT_S, DEFAULT_ACK_TIMEOUT_S, DEFAULT_CHUNK_TOKENS)


def allocation_budget(chunk_tokens=DEFAULT_CHUNK_TOKENS, command_slots=DEFAULT_COMMAND_SLOTS,
                      graph_split_bytes=None):
    """C's new allocations plus A's inventory, all with explicit basis.

    Estimates are estimates. Nothing here is funded by the unmet 512-MiB
    reserve or by the existing 24-page arena, and no entry is a throughput
    claim. ``graph_split_bytes`` is passed through to A's inventory so B's
    bank has one owner.
    """
    if type(chunk_tokens) is not int or chunk_tokens < 1:
        raise HandoffError('chunk_tokens must be a positive int')
    if type(command_slots) is not int or command_slots < 1:
        raise HandoffError('command_slots must be a positive int')
    base = estimated_e3_inventory(chunk_tokens=chunk_tokens,
                                  **({} if graph_split_bytes is None
                                     else {'graph_split_bytes': graph_split_bytes}))
    entries = list(base.entries)
    entries.extend((
        InventoryEntry('prefill_pp_send_recv', 'prefill', 'pp_handoff',
                       PREFILL_PP_BUFFERS_BYTES, 'estimate', True,
                       'independent prefill PP buffers; released after PP consumer proof'),
        InventoryEntry('command_metadata', 'executor', 'host_metadata',
                       command_slots * COMMAND_METADATA_BYTES_PER_SLOT,
                       'estimate', True, 'bounded command ring; fixed slot count'),
        InventoryEntry('telemetry_ring', 'executor', 'host_metadata',
                       TELEMETRY_RECORDS * TELEMETRY_BYTES_PER_RECORD,
                       'estimate', True, 'metadata-only, bounded ring'),
        InventoryEntry('prefill_uid_buffer', 'prefill', 'metadata', 0,
                       'unmeasured', True,
                       'C-supplied native QSA batch UID storage; measure before arming'),
    ))
    return InventoryReport(tuple(entries))


def derive_command_capacity(broadcast_queue, requested=None):
    """Derive admission capacity from the actual broadcast ring state.

    The pinned ``MessageQueue`` exposes ``buffer.max_chunks`` (10 by default).
    A requested lower bound may only reduce capacity; it can never exceed what
    the transport can hold without blocking, and an opaque queue is refused
    rather than guessed.
    """
    buffer = getattr(broadcast_queue, 'buffer', None)
    ring = getattr(buffer, 'max_chunks', None)
    if type(ring) is not int or ring < 1:
        raise ConfigurationError('cannot derive command capacity from this transport')
    capacity = ring
    if requested is not None:
        capacity = min(capacity, _positive_int('requested command slots', requested))
    return capacity


@dataclass(frozen=True)
class AdmittedRoute:
    route: Route
    sparse_histories: int
    serialized_fallback: bool
    reason: str

    def __post_init__(self):
        if type(self.route) is not Route:
            raise HandoffError('route must be Route')
        if type(self.sparse_histories) is not int or self.sparse_histories < 0:
            raise HandoffError('sparse_histories must be a nonnegative int')
        if type(self.serialized_fallback) is not bool:
            raise HandoffError('serialized_fallback must be bool')


def admit_route(*, overlap_enabled, transport, sparse_active, dense_available):
    """Route one transaction before admission; never silently join sparse.

    Sparse-UVA prefill is counted (:class:`AdmittedRoute.sparse_histories`) and
    serialized, not staged through the protected overlap. A missing dense
    history falls back to the ordinary serial path rather than pretending a
    device union is DMA staging.
    """
    if type(sparse_active) is not bool or type(dense_available) is not bool:
        raise HandoffError('admission flags must be bool')
    if overlap_enabled:
        if transport != QUALIFIED_TRANSPORT:
            raise HandoffError('unqualified transport cannot be admitted')
        if sparse_active:
            return AdmittedRoute(Route.SERIAL, HISTORIES_PER_RANK, True,
                                 'sparse_uva_serialized_not_overlap')
        if not dense_available:
            return AdmittedRoute(Route.SERIAL, HISTORIES_PER_RANK, False,
                                 'dense_history_unavailable_serial')
        return AdmittedRoute(Route.DENSE_DMA, 0, False, 'dense_dma_qualified')
    return AdmittedRoute(Route.SERIAL, HISTORIES_PER_RANK if sparse_active else 0,
                         False, 'overlap_disabled')


# --- metadata-only telemetry ------------------------------------------------

_TELEMETRY_FIELDS = frozenset({
    'txn_seq', 'op_seq', 'worker_rank', 'role', 'state', 'phase', 'route',
    'outstanding_bytes', 'ack_lag_s', 'decode_ready_to_launch_s', 'pending',
    'submitted', 'completed', 'committed', 'quarantined', 'reason_code',
    'sparse_serialized', 'cow', 'arena_bytes', 'pp_outstanding', 'command_slots',
    'drain_state', 'request_uid_len', 'generation', 'chunk_tokens',
})
_REASON_CODES = frozenset({
    'sparse_uva_serialized', 'dense_unavailable', 'overlap_disabled',
    'reply_timeout', 'transport_failure', 'late_reply', 'stale_identity',
    'cancel', 'abort', 'disconnect', 'preempt', 'reset', 'pause', 'shutdown',
    'backend_failure', 'commit_once', 'publication', 'prefix_unpublished',
    'not_ready', 'clean', 'unclean', 'sparse_uva_serialized_not_overlap',
    'dense_history_unavailable_serial',
})


class Telemetry:
    """Bounded metadata-only ring. Rejects payloads, request ids and tensors."""

    def __init__(self, limit=TELEMETRY_RECORDS):
        _positive_int('telemetry limit', limit)
        self.limit = limit
        self._records = deque(maxlen=limit)
        self._lock = threading.Lock()
        self.dropped = 0

    def record(self, event, **fields):
        if not isinstance(event, str) or not event or len(event) > 48:
            raise HandoffError('telemetry event must be a short string')
        clean = {}
        for key, value in fields.items():
            if key not in _TELEMETRY_FIELDS:
                raise HandoffError('telemetry field is not metadata: ' + str(key))
            if key == 'reason_code':
                if value not in _REASON_CODES:
                    raise HandoffError('unknown telemetry reason code')
            elif key in ('role', 'state', 'phase', 'route', 'drain_state'):
                if not isinstance(value, str) or len(value) > 32 or any(
                        ord(c) < 32 for c in value):
                    raise HandoffError('telemetry enum text is invalid')
            elif key in ('request_uid_len',):
                if type(value) is not int or value < 0:
                    raise HandoffError('request_uid_len must be a nonnegative int')
            elif type(value) not in (int, float, bool):
                raise HandoffError('telemetry values must be numeric metadata')
            elif isinstance(value, float) and not math.isfinite(value):
                raise HandoffError('telemetry floats must be finite')
            clean[key] = value
        record = {'event': event, **clean}
        with self._lock:
            if len(self._records) == self.limit:
                self.dropped += 1
            self._records.append(record)
        return record

    def snapshot(self):
        with self._lock:
            return tuple(self._records)

    def __len__(self):
        return len(self._records)


# --- typed wire messages ----------------------------------------------------

@dataclass(frozen=True, repr=False)
class WorkerMessage:
    """Base for the versioned S5 worker RPCs (open/submit/poll/commit/cancel)."""
    kind: str = field(init=False, default='')

    def validate(self):
        raise NotImplementedError


def _check_transaction(name, transaction):
    if type(transaction) is not TransactionId:
        raise HandoffError(f'{name} must carry a TransactionId')
    return transaction


@dataclass(frozen=True)
class OpenPrefill(WorkerMessage):
    transaction: TransactionId
    chunk_tokens: int
    # Immutable CPU admission snapshot. It travels with the command because the
    # worker reserves its private lane from it; it is never re-derived on the
    # worker from the already-mutated batch.
    metadata: object = None
    kind: str = field(default='open', init=False)

    def validate(self):
        _check_transaction('open', self.transaction)
        _positive_int('chunk_tokens', self.chunk_tokens)
        return self


@dataclass(frozen=True)
class SubmitOperation(WorkerMessage):
    transaction: TransactionId
    op_seq: int
    dependency_role: DependencyRole
    # Producer role: the parked stock schedule this worker must execute, plus
    # the grammar output the ordinary sampling path consumes. Commit role: the
    # model/host state value the scheduler admitted, applied exactly once.
    scheduler_output: object = None
    grammar_output: object = None
    state_value: object = None
    kind: str = field(default='submit', init=False)

    def validate(self):
        _check_transaction('submit', self.transaction)
        if type(self.op_seq) is not int or self.op_seq < 0:
            raise HandoffError('submit op_seq must be a nonnegative int')
        if type(self.dependency_role) is not DependencyRole:
            raise HandoffError('submit dependency role is invalid')
        return self


@dataclass(frozen=True)
class PollOperations(WorkerMessage):
    transaction: TransactionId
    budget: int
    kind: str = field(default='poll', init=False)

    def validate(self):
        _check_transaction('poll', self.transaction)
        _positive_int('poll budget', self.budget)
        return self


@dataclass(frozen=True)
class CommitTransaction(WorkerMessage):
    transaction: TransactionId
    completion_seq: int
    state_value: object = None
    kind: str = field(default='commit', init=False)

    def validate(self):
        _check_transaction('commit', self.transaction)
        if type(self.completion_seq) is not int or self.completion_seq < 0:
            raise HandoffError('commit completion_seq must be a nonnegative int')
        return self


@dataclass(frozen=True)
class CancelTransaction(WorkerMessage):
    transaction: TransactionId
    reason_code: str
    kind: str = field(default='cancel', init=False)

    def validate(self):
        _check_transaction('cancel', self.transaction)
        if self.reason_code not in _STOP_KINDS:
            raise HandoffError('cancel reason code is invalid')
        return self


@dataclass(frozen=True, repr=False)
class SubmitAck:
    """An RPC acknowledgment: not a completion and not a commit."""
    transaction: TransactionId
    op_seq: int
    accepted: bool

    def __post_init__(self):
        _check_transaction('ack', self.transaction)
        if type(self.op_seq) is not int or self.op_seq < 0:
            raise HandoffError('ack op_seq must be a nonnegative int')
        if type(self.accepted) is not bool:
            raise HandoffError('ack.accepted must be bool')


@dataclass(frozen=True, repr=False)
class OpenAck:
    """The lane each worker actually reserved for this transaction.

    The engine admits with unbound ranks because worker lane identities are
    created by A inside each worker process. This reply is what binds them; it
    is neither a completion nor a publication.
    """
    transaction: TransactionId
    lane: LaneId
    accepted: bool

    def __post_init__(self):
        _check_transaction('open ack', self.transaction)
        if type(self.lane) is not LaneId:
            raise HandoffError('open ack must carry a LaneId')
        if self.lane.role is not LaneRole.PREFILL:
            raise HandoffError('open ack must carry the prefill lane')
        if type(self.accepted) is not bool:
            raise HandoffError('open ack.accepted must be bool')


@dataclass(frozen=True)
class CompletionReport:
    transaction: TransactionId
    proofs: tuple
    pending: int
    failed: bool

    def __post_init__(self):
        _check_transaction('completion report', self.transaction)
        if type(self.proofs) is not tuple:
            raise HandoffError('completion proofs must be an immutable tuple')
        for proof in self.proofs:
            if type(proof) is not CompletionProof:
                raise HandoffError('completion report entries must be CompletionProof')
            if proof.ticket.transaction != self.transaction:
                raise HandoffError('completion proof belongs to another transaction')
        if type(self.pending) is not int or self.pending < 0:
            raise HandoffError('completion report pending must be a nonnegative int')
        if type(self.failed) is not bool:
            raise HandoffError('completion report failed must be bool')


@dataclass(frozen=True)
class CommitAck:
    transaction: TransactionId
    epoch: int
    published: bool
    # Only the output-producing rank carries one. It is the ordinary sampling
    # result for the parked chunk, not a proof and not a second publication.
    output: object = None

    def __post_init__(self):
        _check_transaction('commit ack', self.transaction)
        if type(self.epoch) is not int or self.epoch < 1:
            raise HandoffError('commit ack epoch must be positive')
        if type(self.published) is not bool:
            raise HandoffError('commit ack published must be bool')


@dataclass(frozen=True)
class CancelAck:
    transaction: TransactionId
    state: str

    def __post_init__(self):
        _check_transaction('cancel ack', self.transaction)
        if self.state not in ('retiring', 'retired', 'quarantined'):
            raise HandoffError('cancel ack state is invalid')


def validate_typed_reply(expected, message):
    """Validate an S5 payload against its expected transaction identity.

    Stock wire tuples carry no call id, so FIFO correlation remains audited
    separately. A typed payload that contradicts the expected generation,
    transaction sequence or operation sequence is stale and must quarantine.
    """
    if type(expected) is not TransactionId:
        raise HandoffError('expected transaction is required')
    _check_transaction('reply', getattr(message, 'transaction', None))
    if message.transaction != expected:
        raise HandoffError('typed reply belongs to a stale transaction')
    if (message.transaction.request_key != expected.request_key or
            message.transaction.txn_seq != expected.txn_seq):
        raise HandoffError('typed reply identity is stale')
    return message


# --- completion backends ----------------------------------------------------

class CompletionBackend:
    """C's completion source. Implementations must be bounded and nonblocking."""

    native_verified = False

    def submit(self, ticket):
        raise NotImplementedError

    def poll(self, transaction):
        raise NotImplementedError


class FixtureCompletionBackend(CompletionBackend):
    """Deterministic fake used only by CPU tests.

    Each ticket is completed after ``delay`` polls; ``fail`` makes the next
    poll raise. Completions are independent of the RPC acknowledgment, which is
    the property the state machine must not confuse.
    """

    native_verified = False
    fixture_only = True

    def __init__(self, delay=1, fail=False, kind=CompletionKind.GPU_EVENT):
        if type(delay) is not int or delay < 0:
            raise HandoffError('fixture delay must be a nonnegative int')
        if type(fail) is not bool:
            raise HandoffError('fixture fail must be bool')
        if type(kind) is not CompletionKind:
            raise HandoffError('fixture kind must be CompletionKind')
        self.delay = delay
        self.fail = fail
        self.kind = kind
        self.seq = 0
        self._submitted = {}
        self._polls = {}

    def submit(self, ticket):
        if type(ticket) is not OperationTicket:
            raise HandoffError('backend submit needs OperationTicket')
        if ticket in self._submitted:
            raise HandoffError('operation submitted twice')
        self._submitted[ticket] = None
        self._polls[ticket] = 0
        return ticket

    def poll(self, transaction):
        if self.fail:
            raise RuntimeError('fixture backend failure')
        ready = []
        for ticket in list(self._submitted):
            if ticket.transaction != transaction:
                continue
            self._polls[ticket] += 1
            if self._polls[ticket] >= self.delay:
                del self._submitted[ticket]
                del self._polls[ticket]
                self.seq += 1
                ready.append(CompletionProof(ticket, self.kind, self.seq))
        return tuple(ready)


def require_native_backend(backend):
    """Enabled startup must not accept the null or fixture backend."""
    if not isinstance(backend, CompletionBackend):
        raise ConfigurationError('completion backend must implement CompletionBackend')
    if getattr(backend, 'fixture_only', False) or not backend.native_verified:
        raise ConfigurationError(
            'enabled overlap requires a native, device-event completion backend')


# --- transaction state machine ---------------------------------------------

@dataclass
class _Operation:
    ticket: OperationTicket
    submitted_s: float
    ack: SubmitAck | None = None
    proof: CompletionProof | None = None

    @property
    def acked(self):
        return self.ack is not None

    @property
    def completed(self):
        return self.proof is not None


class TransactionLease:
    """One prefill transaction's RPC/ack/completion/publication state.

    ``acknowledge`` only records a reply. ``complete`` needs a real proof.
    ``commit`` requires every operation acked and completed plus one explicit
    commit-role state-write proof, and publishes exactly once. Any reply that
    does not match a currently outstanding ticket quarantines the lease; the
    caller keeps storage leased until an explicit recovery.
    """

    def __init__(self, transaction, lanes, *, max_outstanding, clock=time.monotonic,
                 telemetry=None, lane_categories=None):
        _check_transaction('lease', transaction)
        if type(lanes) is not dict or not lanes:
            raise HandoffError('lease needs a rank -> LaneId map')
        for rank, lane in lanes.items():
            if type(rank) is not int:
                raise HandoffError('invalid lane map')
            if lane is None:
                # Admission happens on the engine before any worker has
                # reserved its lane; bind_lanes completes the map.
                continue
            if type(lane) is not LaneId or lane.worker_rank != rank:
                raise HandoffError('invalid lane map')
        self.transaction = transaction
        self.lanes = dict(lanes)
        self.max_outstanding = _positive_int('max_outstanding', max_outstanding)
        self.clock = clock
        self.telemetry = telemetry
        self.phase = Phase.OPEN
        self.outstanding = {}
        self.completed = []
        self.submitted = []
        self.acked = 0
        self.commit_proof = None
        self.commit_epoch = 0
        self.worker_outputs = {}
        self.stop_kind = None
        self.quarantine = None
        self._last_op = {}
        self._last_completion = {}
        self._submitted_s = clock()
        self._completion_bytes = 0
        self._lane_categories = dict(lane_categories or {})

    # -- identity helpers -------------------------------------------------
    @property
    def active(self):
        return self.phase in (Phase.OPEN, Phase.SUBMITTED, Phase.COMPLETING)

    @property
    def retired(self):
        return self.phase in (Phase.RETIRED, Phase.QUARANTINED)

    @property
    def pending(self):
        return len(self.outstanding)

    @property
    def request_uid_len(self):
        return len(self.transaction.request_uid)

    @property
    def unbound(self):
        return tuple(sorted(rank for rank, lane in self.lanes.items() if lane is None))

    def bind_lanes(self, lanes):
        """Bind each rank to the prefill LaneId its worker actually reserved.

        A rank can be bound once; the value must come from that rank's own
        ``OpenAck``. Rebinding would launder a stale storage/context epoch.
        """
        if type(lanes) is not dict or set(lanes) != set(self.lanes):
            raise HandoffError('lane binding must cover every leased rank once')
        for rank, lane in lanes.items():
            if type(lane) is not LaneId or lane.worker_rank != rank:
                raise HandoffError('bound lane disagrees with its rank')
            if lane.role is not LaneRole.PREFILL:
                raise HandoffError('bound lane is not the prefill lane')
            current = self.lanes[rank]
            if current is not None and current != lane:
                self._quarantine('stale_identity')
                raise HandoffError('lane was already bound to another epoch')
        for rank, lane in lanes.items():
            self.lanes[rank] = lane
        return dict(self.lanes)

    def _quarantine(self, reason):
        if self.phase is Phase.QUARANTINED:
            return self.quarantine
        self.quarantine = reason
        self.phase = Phase.QUARANTINED
        self._emit('lease_quarantined', phase=Phase.QUARANTINED.value,
                   reason_code=reason, pending=len(self.outstanding))
        return reason

    def _emit(self, event, **fields):
        if self.telemetry is not None:
            self.telemetry.record(event, txn_seq=self.transaction.txn_seq,
                                  generation=self.transaction.request_generation,
                                  request_uid_len=self.request_uid_len, **fields)

    def _require_live(self, allow_retiring=False):
        if not self.active and not (allow_retiring and self.phase is Phase.RETIRING):
            if self.phase is Phase.QUARANTINED:
                raise HandoffError('transaction is quarantined')
            raise HandoffError('transaction is not active')
        if self.quarantine is not None:
            raise HandoffError('transaction is quarantined')

    # -- ack / completion / commit ---------------------------------------
    def submit(self, ticket):
        self._require_live()
        if self.phase is Phase.COMMITTED:
            raise HandoffError('transaction is already committed')
        if type(ticket) is not OperationTicket:
            raise HandoffError('submit needs OperationTicket')
        if ticket.transaction != self.transaction:
            raise HandoffError('operation belongs to another transaction')
        if ticket.worker_rank not in self.lanes:
            raise HandoffError('operation rank is not in the lease')
        lane = self.lanes[ticket.worker_rank]
        if lane is None:
            raise HandoffError('operation submitted before the worker lane was bound')
        if ticket.lane != lane:
            raise HandoffError('operation lane/storage epoch is stale')
        key = ticket.lane  # op_seq is the value, never part of its monotonic key.
        previous = self._last_op.get(key)
        if previous is not None and ticket.op_seq <= previous:
            raise HandoffError('operation sequence is not monotonic')
        if len(self.outstanding) >= self.max_outstanding:
            raise E3Pending('transaction operation credit exhausted')
        self._last_op[key] = ticket.op_seq
        operation = _Operation(ticket, self.clock())
        self.outstanding[ticket] = operation
        self.submitted.append(operation)
        self.phase = Phase.SUBMITTED
        self._emit('operation_submitted', op_seq=ticket.op_seq,
                   worker_rank=ticket.worker_rank,
                   role=ticket.lane.role.value, state=self.phase.value)
        return ticket

    def acknowledge(self, transaction, op_seq, accepted=True, worker_rank=None):
        """Record an RPC reply. It cannot complete or publish anything."""
        if self.phase in (Phase.RETIRED, Phase.QUARANTINED):
            raise HandoffError('reply after transaction retirement cannot mutate it')
        if transaction != self.transaction:
            self._quarantine('late_reply')
            raise HandoffError('late reply cannot be reassociated')
        if not accepted:
            self._quarantine('backend_failure')
            raise HandoffError('worker rejected the operation')
        if worker_rank is not None and type(worker_rank) is not int:
            self._quarantine('stale_identity')
            raise HandoffError('acknowledgment rank is invalid')
        matches = [op for op in self.submitted
                   if op.ticket.op_seq == op_seq
                   and (worker_rank is None or op.ticket.worker_rank == worker_rank)]
        if len(matches) != 1:
            self._quarantine('stale_identity')
            raise HandoffError('acknowledgment names no single outstanding operation')
        operation = matches[0]
        if operation.ack is not None:
            self._quarantine('stale_identity')
            raise HandoffError('duplicate acknowledgment would relabel a reply')
        operation.ack = SubmitAck(transaction, op_seq, True)
        self.acked += 1
        lag = max(0.0, self.clock() - operation.submitted_s)
        self._emit('operation_acked', op_seq=op_seq,
                   worker_rank=operation.ticket.worker_rank, ack_lag_s=lag,
                   pending=len(self.outstanding))
        return operation.ack

    def complete(self, proof):
        self._require_live(allow_retiring=True)
        if type(proof) is not CompletionProof:
            raise HandoffError('completion needs CompletionProof')
        if proof.ticket.transaction != self.transaction:
            self._quarantine('late_reply')
            raise HandoffError('completion belongs to another transaction')
        operation = self.outstanding.get(proof.ticket)
        if operation is None:
            self._quarantine('late_reply')
            raise HandoffError('completion has no outstanding operation')
        if proof.ticket.lane != self.lanes.get(proof.ticket.worker_rank):
            self._quarantine('stale_identity')
            raise HandoffError('completion lane/storage epoch is stale')
        if operation.proof is not None:
            self._quarantine('stale_identity')
            raise HandoffError('duplicate completion')
        key = proof.ticket.lane
        if proof.completion_seq <= self._last_completion.get(key, -1):
            self._quarantine('stale_identity')
            raise HandoffError('completion sequence is not monotonic')
        self._last_completion[key] = proof.completion_seq
        operation.proof = proof
        del self.outstanding[proof.ticket]
        self.completed.append(proof)
        if self.phase is not Phase.RETIRING:
            self.phase = Phase.COMPLETING
        self._emit('operation_completed', op_seq=proof.ticket.op_seq,
                   worker_rank=proof.ticket.worker_rank,
                   pending=len(self.outstanding), state=self.phase.value)
        return proof

    def commit(self, proof):
        """Publish exactly once; every outstanding operation must be complete."""
        if self.phase is Phase.COMMITTED:
            self._emit('commit_replay_rejected', state=self.phase.value,
                       reason_code='commit_once')
            raise HandoffError('transaction was already published')
        self._require_live()
        if type(proof) is not CompletionProof:
            raise HandoffError('commit needs a real CompletionProof')
        if proof.kind is not CompletionKind.STATE_WRITE:
            raise HandoffError('commit proof must be a state-write completion')
        if proof.ticket.transaction != self.transaction:
            raise HandoffError('commit proof belongs to another transaction')
        if proof.ticket.dependency_role is not DependencyRole.COMMIT:
            raise HandoffError('commit proof is not a commit-role operation')
        if proof not in self.completed:
            raise HandoffError('commit proof was not a recorded completion')
        if self.outstanding:
            raise E3Pending('commit cannot publish while operations remain outstanding')
        for operation in self.submitted:
            if not operation.completed:
                raise E3Pending('commit needs every operation completion')
            if not operation.acked:
                raise E3Pending('an operation completed without acknowledgment')
        if len(self.completed) != len(self.submitted):
            raise E3Pending('completion set does not cover every submitted operation')
        committed_ranks = {p.ticket.worker_rank for p in self.completed
                           if p.kind is CompletionKind.STATE_WRITE
                           and p.ticket.dependency_role is DependencyRole.COMMIT}
        if committed_ranks != set(self.lanes):
            raise E3Pending('commit requires state-write completion on every rank')
        self.commit_proof = proof
        self.commit_epoch += 1
        self.phase = Phase.COMMITTED
        self._emit('transaction_committed', committed=1, state=self.phase.value,
                   reason_code='publication')
        return self.commit_epoch

    def start_stop(self, kind):
        if kind not in _STOP_KINDS:
            raise HandoffError('unknown stop kind')
        if self.phase in (Phase.COMMITTED, Phase.RETIRED, Phase.QUARANTINED):
            return self.phase
        self.stop_kind = kind
        self.phase = Phase.RETIRING
        self._emit('transaction_stopping', state=self.phase.value,
                   reason_code=kind, pending=len(self.outstanding))
        return self.phase

    def drain(self):
        """Retire only when every submitted operation has a completion proof.

        An unclean drain stays retirable but keeps quarantined outstanding
        identities; the caller must not release storage in that state.
        """
        if self.quarantine is not None:
            raise HandoffError('quarantined transaction cannot retire storage')
        if self.outstanding or any(not op.acked for op in self.submitted):
            raise E3Pending('transaction has outstanding operations or acknowledgments')
        self.phase = Phase.RETIRED if self.quarantine is None else Phase.QUARANTINED
        self._emit('transaction_drained', state=self.phase.value,
                   reason_code='clean' if self.quarantine is None else self.quarantine)
        return self.phase


# --- engine-level dispatcher ------------------------------------------------

@dataclass(frozen=True)
class DispatchReport:
    polled: int
    ready: tuple
    pending: int
    decode_ready: bool
    failed: bool
    phase: str


class HandoffDispatcher:
    """Single host-owner driver for one parked prefill transaction.

    It submits a bounded operation, returns to decode readiness without waiting
    for a full prefill layer, polls only local completion state, and publishes
    exactly once after every required completion. It never calls
    ``FutureWrapper.result()`` or an MQ ``dequeue`` itself; the owned receiver
    (``reply_transport``) does that on separate threads.
    """

    def __init__(self, config, *, clock=time.monotonic, telemetry=None,
                 allow_fixture=False):
        if type(config) is not HandoffConfig:
            raise ConfigurationError('dispatcher needs HandoffConfig')
        self.config = config
        self.clock = clock
        self.telemetry = telemetry or Telemetry()
        self.allow_fixture = allow_fixture
        self.owner_thread = threading.get_ident()
        self.active = None
        self.backend = None
        self._transport = None
        self._inflight = {}
        self._decode_ready_s = None
        self._stopping = None
        self._retired = 0
        self._quarantined = 0
        self.replies = {}

    def assert_owner(self):
        if threading.get_ident() != self.owner_thread:
            raise HandoffError('handoff dispatcher called from a foreign host thread')

    def install_transport(self, transport):
        if not self.config.enabled:
            raise ConfigurationError('overlap is disabled')
        self.assert_owner()
        if self._transport is not None:
            raise HandoffError('transport is already installed')
        if not isinstance(transport, ReplyTransport):
            raise ConfigurationError('dispatcher needs the owned ReplyTransport')
        if transport._owner_thread != self.owner_thread:
            raise ConfigurationError('transport must be created by the dispatcher thread')
        self._transport = transport
        return transport

    def install_backend(self, backend):
        if not self.config.enabled:
            raise ConfigurationError('overlap is disabled')
        self.assert_owner()
        if not self.allow_fixture:
            require_native_backend(backend)
        else:
            if not isinstance(backend, CompletionBackend):
                raise ConfigurationError('backend must implement CompletionBackend')
        self.backend = backend
        return backend

    # -- admission --------------------------------------------------------
    def open(self, transaction, lanes, *, metadata=None, sparse_active=False,
             dense_available=True):
        self.assert_owner()
        if not self.config.enabled:
            raise ConfigurationError('overlap is disabled')
        if self._stopping is not None:
            raise E3Pending('dispatcher admission stopped')
        if self.active is not None and self.active.phase is not Phase.RETIRED:
            raise E3Pending('one active prefill transaction is already leased')
        if MAX_ACTIVE_PREFILL != 1:
            raise HandoffError('this increment admits exactly one active prefill')
        route = admit_route(overlap_enabled=True, transport=self.config.transport,
                            sparse_active=sparse_active, dense_available=dense_available)
        if route.route is not Route.DENSE_DMA:
            self.telemetry.record('route_serialized', route=route.route.value,
                                  reason_code=route.reason, sparse_serialized=int(route.serialized_fallback))
            raise E3Pending('prefill cannot enter protected overlap: ' + route.reason)
        lease = TransactionLease(transaction, lanes,
                                 max_outstanding=min(self.config.chunk_tokens, 16),
                                 clock=self.clock, telemetry=self.telemetry)
        if metadata is not None and not getattr(metadata, 'rows', None):
            raise HandoffError('prefill metadata snapshot is empty')
        self.active = lease
        self.replies.clear()
        self._decode_ready_s = None
        self.telemetry.record('transaction_opened', txn_seq=transaction.txn_seq,
                              generation=transaction.request_generation,
                              request_uid_len=len(transaction.request_uid),
                              route=route.route.value, chunk_tokens=self.config.chunk_tokens)
        return lease

    def submit(self, ticket, transport_submit, *, kind='operation', ranks=None):
        """Register a bounded operation and broadcast it once.

        ``transport_submit`` receives the local sequence and must enqueue the
        command; the reply expectation is already registered by the transport.
        A full command ring raises :class:`Backpressure`; it never grows a
        backlog or blocks decode.
        """
        self.assert_owner()
        lease = self.active
        if lease is None:
            raise HandoffError('no active prefill transaction')
        if not callable(transport_submit):
            raise HandoffError('submit needs a bounded transport sender')
        if self._transport is None:
            raise HandoffError('no owned reply transport is installed')
        if kind == 'operation':
            if not lease.active:
                raise HandoffError('no active prefill transaction')
            tickets = (ticket,) if type(ticket) is OperationTicket else tuple(ticket)
            if not tickets or any(type(item) is not OperationTicket for item in tickets):
                raise HandoffError('submit needs OperationTicket(s)')
            if {item.worker_rank for item in tickets} != set(lease.lanes):
                raise HandoffError('one operation ticket is required per leased rank')
            if len({item.op_seq for item in tickets}) != 1:
                raise HandoffError('one rank-set operation must share its op_seq')
            for item in tickets:
                if lease.lanes.get(item.worker_rank) is None:
                    raise HandoffError('operation ticket names an unbound lane')
                if lease.lanes[item.worker_rank] != item.lane:
                    raise HandoffError('operation ticket lane is stale')
        elif kind in _CONTROL_KINDS:
            # A committing or stopping transaction still needs completion polls,
            # the commit acknowledgment and the cancel command; refusing them
            # would strand worker storage or drop the publication reply.
            if lease.phase in (Phase.RETIRED, Phase.QUARANTINED):
                raise HandoffError('transaction is not accepting control messages')
            if ticket is not None and ticket != ():
                raise HandoffError('a control message carries no operation ticket')
            tickets = ()
        else:
            raise HandoffError('unknown prefill submission kind')
        expected = tuple(sorted(lease.lanes)) if ranks is None else tuple(ranks)
        if len(expected) != len(set(expected)) or set(expected) != set(lease.lanes):
            raise HandoffError('prefill operation must reach every leased rank once')
        if len(tickets) != len({item.worker_rank for item in tickets}):
            raise HandoffError('duplicate rank ticket')
        saved = (dict(lease.outstanding), list(lease.submitted), dict(lease._last_op), lease.phase)
        if kind == 'operation':
            try:
                for item in tickets:
                    lease.submit(item)
            except BaseException:
                lease.outstanding, lease.submitted, lease._last_op, lease.phase = saved
                raise
        try:
            sequence = self._transport.submit(
                'prefill-' + kind, transport_submit, ranks=expected,
                timeout_s=self.config.reply_timeout_s + self.config.ack_timeout_s)
        except Backpressure:
            # No broadcast was issued. Restore unsent tickets and sequence credits.
            lease.outstanding, lease.submitted, lease._last_op, lease.phase = saved
            self.telemetry.record('command_ring_full', route=self.config.transport,
                                  command_slots=self.config.command_slots,
                                  pending=lease.pending)
            raise
        except BaseException:
            lease._quarantine('transport_failure')
            raise
        self._inflight[sequence] = (kind, lease.transaction, tickets)
        if not tickets:
            return None
        return tickets[0] if len(tickets) == 1 else tickets

    def _ranks(self, lease):
        return tuple(sorted(lease.lanes))

    def mark_decode_ready(self):
        self.assert_owner()
        self._decode_ready_s = self.clock()
        self.telemetry.record('decode_ready', state=(self.active.phase.value
                                                     if self.active else Phase.IDLE.value))
        return self._decode_ready_s

    def mark_launch(self):
        """Record host decode-ready-to-launch latency under prefill submission."""
        self.assert_owner()
        if self._decode_ready_s is None:
            return None
        latency = max(0.0, self.clock() - self._decode_ready_s)
        self.telemetry.record('prefill_submitted_to_launch',
                              decode_ready_to_launch_s=latency)
        self._decode_ready_s = None
        return latency

    def poll(self, budget=8):
        """Bounded local poll; returns decode readiness without waiting."""
        self.assert_owner()
        if type(budget) is not int or budget < 1:
            raise HandoffError('poll budget must be a positive int')
        lease = self.active
        if lease is None:
            return DispatchReport(0, (), 0, True, False, Phase.IDLE.value)
        if lease.quarantine is not None:
            return DispatchReport(0, (), lease.pending, True, True,
                                  Phase.QUARANTINED.value)
        try:
            self._drain_replies(budget)
        except HandoffError:
            lease._quarantine('stale_identity')
        except TransportError:
            lease._quarantine('transport_failure')
            self._quarantined += 1
        ready = () if lease.quarantine is not None else self._poll_completions(lease, budget)
        return DispatchReport(len(ready), tuple(ready), lease.pending, True,
                              lease.quarantine is not None, lease.phase.value)

    def _drain_replies(self, budget):
        """Consume ready typed replies without touching unresolved futures."""
        if self._transport is None:
            return
        self._transport.poll(budget)
        for sequence, context in list(self._inflight.items()):
            call = self._transport._calls.get(sequence)
            if call is None or not call.ready:
                continue
            value, lag = self._transport.take(sequence)
            del self._inflight[sequence]
            # Per-rank lags are metadata; record the worst finite value only.
            worst = max((float(v) for v in lag.values() if math.isfinite(v)),
                        default=0.0)
            self.telemetry.record('reply_received', ack_lag_s=worst)
            self._apply_reply(context[0], context[1], context[2], value)
            self.replies[context[0]] = self.replies.get(context[0], 0) + 1

    @property
    def total_replies(self):
        """Applied reply count; lets the engine driver step without blocking."""
        return sum(self.replies.values())

    def reply_count(self, kind):
        return self.replies.get(kind, 0)

    def _apply_reply(self, kind, transaction, tickets, value):
        lease = self.active
        if lease is None or lease.transaction != transaction:
            # A reply for a retired/cancelled identity never mutates a newer one.
            self.telemetry.record('late_reply_rejected', reason_code='late_reply')
            self._quarantined += 1
            return
        if kind == 'open':
            acks = value if isinstance(value, (list, tuple)) else (value,)
            if len(acks) != len(lease.lanes):
                raise HandoffError('open reply does not cover every rank')
            binding = {}
            for rank, ack in zip(sorted(lease.lanes), acks):
                if type(ack) is not OpenAck:
                    raise HandoffError('open reply is not a typed lane acknowledgment')
                validate_typed_reply(transaction, ack)
                if not ack.accepted:
                    raise HandoffError('worker refused the prefill reservation')
                binding[rank] = ack.lane
            lease.bind_lanes(binding)
            return
        if kind == 'operation':
            acks = value if isinstance(value, (list, tuple)) else (value,)
            if len(acks) != len(lease.lanes):
                raise HandoffError('operation reply does not cover every rank')
            if len(acks) == 1:
                by_rank = {tickets[0].worker_rank: acks[0]}
            else:
                by_rank = dict(zip(sorted(lease.lanes), acks))
            for rank, ack in by_rank.items():
                if type(ack) is not SubmitAck:
                    raise HandoffError('operation reply is not a typed acknowledgment')
                validate_typed_reply(transaction, ack)
                lease.acknowledge(transaction, ack.op_seq, ack.accepted, worker_rank=rank)
            return
        if kind == 'poll':
            reports = value if isinstance(value, (list, tuple)) else (value,)
            if len(reports) != len(lease.lanes):
                raise HandoffError('poll reply does not cover every rank')
            for rank, report in zip(sorted(lease.lanes), reports):
                if type(report) is not CompletionReport:
                    raise HandoffError('poll reply is not a completion report')
                validate_typed_reply(transaction, report)
                for proof in report.proofs:
                    if proof.ticket.worker_rank != rank:
                        raise HandoffError('completion arrived through another rank')
                    lease.complete(proof)
            return
        if kind in ('commit', 'cancel'):
            acks = value if isinstance(value, (list, tuple)) else (value,)
            if len(acks) != len(lease.lanes):
                raise HandoffError('control reply does not cover every rank')
            ranks = sorted(lease.lanes)
            if kind == 'commit':
                epochs = {ack.epoch for ack in acks if type(ack) is CommitAck}
                if len(epochs) != 1:
                    raise HandoffError('commit epochs differ across workers')
                for rank, ack in zip(ranks, acks):
                    if type(ack) is not CommitAck:
                        raise HandoffError('control reply has wrong type')
                    validate_typed_reply(transaction, ack)
                    if not ack.published:
                        raise HandoffError('worker did not publish the transaction')
                    if ack.output is not None:
                        lease.worker_outputs[rank] = ack.output
            else:
                for ack in acks:
                    if type(ack) is not CancelAck:
                        raise HandoffError('control reply has wrong type')
                    validate_typed_reply(transaction, ack)
            return
        raise HandoffError('unknown prefill reply kind')

    def _poll_completions(self, lease, budget):
        if self.backend is None:
            return ()
        try:
            proofs = self.backend.poll(lease.transaction) or ()
        except BaseException:
            lease._quarantine('backend_failure')
            self._quarantined += 1
            return ()
        if not isinstance(proofs, (tuple, list)) or len(proofs) > budget:
            lease._quarantine('backend_failure')
            self._quarantined += 1
            return ()
        applied = []
        for proof in proofs:
            try:
                lease.complete(proof)
            except HandoffError:
                self._quarantined += 1
                return tuple(applied)
            applied.append(proof)
        return tuple(applied)

    def commit(self, proof):
        self.assert_owner()
        lease = self.active
        if lease is None:
            raise HandoffError('no active prefill transaction')
        epoch = lease.commit(proof)
        self._retired += 0
        return epoch

    def drain(self):
        """Retire a stopped transaction; refuses while work is outstanding."""
        self.assert_owner()
        lease = self.active
        if lease is None:
            return Phase.IDLE
        if self._inflight:
            raise E3Pending('RPC acknowledgments still outstanding')
        if not (lease.phase is Phase.RETIRING or lease.phase is Phase.COMMITTED):
            raise E3Pending('only a stopping or committed transaction drains')
        phase = lease.drain()
        self._retired += 1
        self.active = lease if phase is Phase.QUARANTINED else None
        if (phase is Phase.RETIRED and self._stopping is not None
                and self._stopping.value in _RESUMABLE_STOPS):
            # The stopped request retired cleanly; a later request may admit
            # again. Engine shutdown keeps the dispatcher closed.
            self._stopping = None
        self.telemetry.record('transaction_retired', state=phase.value,
                              reason_code='clean' if phase is Phase.RETIRED else lease.quarantine)
        return phase

    def stop(self, kind, *, drain=False):
        """Stop new submissions; the caller drains/quarantines consumers."""
        self.assert_owner()
        self._stopping = StopKind(kind)
        lease = self.active
        if lease is None:
            return Phase.IDLE
        lease.start_stop(kind)
        self.telemetry.record('dispatcher_stopping', state=kind,
                              drain_state='draining' if drain else 'quarantined')
        if drain:
            try:
                return self.drain()
            except E3Pending:
                return lease.phase
        return lease.phase

    @property
    def decode_ready(self):
        """Decode never waits for the parked prefill output or a whole layer."""
        return self._stopping is None

    @property
    def transport(self):
        return self._transport


# --- lifetime ledger --------------------------------------------------------

class LeaseCategory(str, Enum):
    SOURCE_HISTORY = 'source_history'
    ARENA = 'arena'
    PP_BUFFER = 'pp_buffer'
    STATE_BUFFER = 'state_buffer'


_ALLOWED_PROOF = {
    LeaseCategory.SOURCE_HISTORY: frozenset({CompletionKind.COPY_EVENT}),
    LeaseCategory.ARENA: frozenset({CompletionKind.GPU_EVENT}),
    LeaseCategory.PP_BUFFER: frozenset({CompletionKind.PP_CONSUMER}),
    LeaseCategory.STATE_BUFFER: frozenset({CompletionKind.STATE_WRITE}),
}


class LifetimeLedger:
    """Protects source/arena/PP/state storage until its consumer completes.

    Leases hold *existing* allocations; acquiring one never allocates and never
    triggers a copy-on-write. Release requires the consumer-kind completion
    proof, and an unknown/stale proof leaves the lease outstanding. Deferred
    free is preferred over new refcounts: E2's ``is_block_writable`` and CoW
    counts stay those of the stock block pool.
    """

    def __init__(self):
        self._leases = {}
        self._retired = set()
        self.cow_requests = 0
        self.cow_completions = 0
        self.lease_caused_cow = 0
        self.rejected_late = 0

    def acquire(self, ticket, category, owner):
        if type(ticket) is not OperationTicket:
            raise HandoffError('lease needs an OperationTicket')
        if type(category) is not LeaseCategory:
            raise HandoffError('lease category is invalid')
        if not isinstance(owner, str) or not owner:
            raise HandoffError('lease owner must be a short name')
        key = (ticket, category)
        if key in self._leases:
            raise HandoffError('operation already leases this category')
        if ticket in self._retired:
            raise HandoffError('late lease cannot resurrect a retired operation')
        self._leases[key] = owner
        return key

    def release(self, ticket, category, proof):
        if type(proof) is not CompletionProof:
            raise HandoffError('lease release needs a CompletionProof')
        if proof.ticket != ticket:
            raise HandoffError('release proof belongs to another operation')
        if proof.kind not in _ALLOWED_PROOF[category]:
            raise HandoffError('release proof kind does not consume this lease')
        key = (ticket, category)
        if key not in self._leases:
            self.rejected_late += 1
            raise HandoffError('release proof has no outstanding lease')
        del self._leases[key]
        if category is LeaseCategory.SOURCE_HISTORY:
            self._retired.add(ticket)
        return key

    def outstanding(self, category=None):
        return tuple(k for k in self._leases if category is None or k[1] is category)

    def assert_live(self, ticket, category):
        if (ticket, category) not in self._leases:
            raise E3Pending('lease is not held')
        return True

    def note_cow(self, requested, completed):
        if type(requested) is not int or type(completed) is not int or requested < 0 or completed < 0:
            raise HandoffError('CoW counts must be nonnegative ints')
        if completed > requested:
            raise HandoffError('more CoW completions than requests')
        self.cow_requests += requested
        self.cow_completions += completed

    def assert_matched_cow(self):
        if self.cow_requests != self.cow_completions:
            raise E3Pending('CoW copies remain outstanding')
        if self.lease_caused_cow:
            raise HandoffError('a lease created a copy-on-write')
        return True


# --- prefix sharing ---------------------------------------------------------

class CommittedFrontier:
    """Offline frontier model; NOT wired to the native cache lookup/allocation epoch.

    Blocks visible to a prefix-sharing request: committed bytes only.

    A parked transaction's blocks are recorded when they are submitted and
    become shareable exactly at commit. A request that would share an
    unpublished block waits or recomputes; it never reads optimistic storage.
    """

    def __init__(self):
        self._epoch = {}
        self._published = {}

    def reserve(self, transaction, block_ids):
        if type(transaction) is not TransactionId:
            raise HandoffError('frontier needs a TransactionId')
        block_ids = tuple(block_ids)
        if any(type(b) is not int or b < 0 for b in block_ids):
            raise HandoffError('block ids must be nonnegative ints')
        if transaction in self._published:
            raise HandoffError('transaction already published its threshold')
        committed = {b for blocks in self._published.values() for b in blocks}
        if committed.intersection(block_ids):
            raise HandoffError('allocation reuse requires explicit frontier retirement')
        self._epoch[transaction] = block_ids
        return block_ids

    def publish(self, transaction):
        if transaction not in self._epoch:
            raise HandoffError('transaction never reserved a frontier')
        if transaction in self._published:
            raise HandoffError('frontier publishes exactly once')
        self._published[transaction] = self._epoch[transaction]
        return self._published[transaction]

    def retire_blocks(self, block_ids):
        """Offline allocator callback model; not wired to native incarnations."""
        ids = set(block_ids)
        if any(ids.intersection(blocks) for txn, blocks in self._epoch.items()
               if txn not in self._published):
            raise E3Pending('cannot retire blocks of an unpublished transaction')
        for txn, blocks in list(self._published.items()):
            remaining = tuple(b for b in blocks if b not in ids)
            if remaining:
                self._published[txn] = remaining
                self._epoch[txn] = remaining
            else:
                del self._published[txn]
                self._epoch.pop(txn, None)

    def cancel(self, transaction):
        if transaction in self._published:
            raise HandoffError('cannot cancel a published frontier')
        self._epoch.pop(transaction, None)

    def shareable(self, block_ids):
        committed = {b for blocks in self._published.values() for b in blocks}
        requested = tuple(block_ids)
        if any(type(b) is not int or b < 0 for b in requested):
            raise HandoffError('block ids must be nonnegative ints')
        if any(b not in committed for b in requested):
            raise E3Pending('prefix share touches unpublished blocks; wait or recompute')
        return requested

    def lookup(self, requested):
        """Truncate a cache hit at the first unpublished block."""
        committed = {b for blocks in self._published.values() for b in blocks}
        hit = []
        for block in requested:
            if block not in committed:
                break
            hit.append(block)
        return tuple(hit)

    def committed_blocks(self):
        return tuple(sorted({b for blocks in self._published.values() for b in blocks}))


# --- dedicated prefill PP channel ------------------------------------------

class PrefillPPChannel:
    """Independent prefill PP traffic; decode's sample ring is not touched.

    The stock ``PPHandler`` owns a FIFO keyed by decode steps and
    ``get_prev_sampled_outputs``. This channel has its own buffers and order,
    bounds outstanding sends, and never calls that method or advances the
    decode ring.
    """

    def __init__(self, stages, *, max_outstanding=2, clock=time.monotonic,
                 telemetry=None):
        if type(stages) is not int or stages < 1:
            raise HandoffError('stages must be a positive int')
        if type(max_outstanding) is not int or max_outstanding < 1:
            raise HandoffError('max_outstanding must be a positive int')
        self.stages = stages
        self.max_outstanding = max_outstanding
        self.clock = clock
        self.telemetry = telemetry
        self._outstanding = {}
        self._order = deque()
        self.sends = 0
        self.consumed = 0

    def send(self, ticket, stage):
        if type(ticket) is not OperationTicket:
            raise HandoffError('PP send needs an OperationTicket')
        if type(stage) is not int or not 0 <= stage < self.stages:
            raise HandoffError('PP stage is out of range')
        if len(self._outstanding) >= self.max_outstanding:
            raise E3Pending('prefill PP send credit exhausted')
        key = (ticket, stage)
        if key in self._outstanding:
            raise HandoffError('prefill PP operation sent twice')
        self._outstanding[key] = self.clock()
        self._order.append(key)
        self.sends += 1
        if self.telemetry is not None:
            self.telemetry.record('pp_send', op_seq=ticket.op_seq,
                                  worker_rank=ticket.worker_rank,
                                  pp_outstanding=len(self._outstanding))
        return key

    def consume(self, ticket, stage, proof):
        if type(proof) is not CompletionProof or proof.kind is not CompletionKind.PP_CONSUMER:
            raise HandoffError('PP consumption needs a PP-consumer completion proof')
        key = (ticket, stage)
        if key not in self._outstanding:
            raise HandoffError('PP completion has no outstanding send')
        if proof.ticket != ticket:
            raise HandoffError('PP completion belongs to another operation')
        del self._outstanding[key]
        self._order.remove(key)
        self.consumed += 1
        return key

    def outstanding(self):
        return tuple(self._order)

    # Deliberately absent: sample-ring access and get_prev_sampled_outputs.
    def get_prev_sampled_outputs(self):
        raise HandoffError('prefill PP traffic must not read decode sample-ring state')


# --- worker-side typed RPC adapter -----------------------------------------

class WorkerHandoff:
    """Worker-side open/submit/poll/commit/cancel adapter.

    ACK, GPU completion and publication stay distinct: ``submit`` returns an
    acknowledgment, ``poll`` returns proofs, and ``commit`` is refused without
    an explicit commit-role state-write proof recorded by the caller.
    """

    fixture_only = True  # Metadata model, not a native worker execution adapter.

    def __init__(self, transaction, *, backend, now=time.monotonic):
        _check_transaction('worker handoff', transaction)
        if not isinstance(backend, CompletionBackend):
            raise ConfigurationError('worker handoff needs a completion backend')
        self.transaction = transaction
        self.backend = backend
        self.now = now
        self.phase = Phase.OPEN
        self.last_op = -1
        self.acks = 0
        self.completions = []
        self.pending_operations = {}
        self.quarantine = None

    def apply(self, message):
        if type(message) is OpenPrefill:
            return self.open(message)
        if type(message) is SubmitOperation:
            return self.submit(message)
        if type(message) is PollOperations:
            return self.poll(message)
        if type(message) is CommitTransaction:
            return self.commit(message)
        if type(message) is CancelTransaction:
            return self.cancel(message)
        raise HandoffError('unknown S5 worker message')

    def _check(self, message):
        message.validate()
        if message.transaction != self.transaction:
            self.quarantine = 'stale_identity'
            raise HandoffError('typed message belongs to another transaction')
        if self.quarantine is not None:
            raise HandoffError('worker handoff is quarantined')

    def open(self, message):
        self._check(message)
        if self.phase is not Phase.OPEN:
            raise HandoffError('prefill transaction is already open')
        return SubmitAck(self.transaction, 0, True)

    def submit(self, message):
        self._check(message)
        if self.phase not in (Phase.OPEN, Phase.SUBMITTED):
            raise HandoffError('prefill transaction is not accepting submissions')
        if message.op_seq <= self.last_op:
            self.quarantine = 'stale_identity'
            raise HandoffError('worker operation sequence is not monotonic')
        self.last_op = message.op_seq
        self.phase = Phase.SUBMITTED
        self.acks += 1
        self.pending_operations[message.op_seq] = message.dependency_role
        return SubmitAck(self.transaction, message.op_seq, True)

    def poll(self, message):
        self._check(message)
        proofs = self.backend.poll(self.transaction)
        if not isinstance(proofs, (tuple, list)) or len(proofs) > message.budget:
            self.quarantine = 'backend_failure'
            raise HandoffError('worker backend poll is unbounded')
        applied = []
        for proof in proofs:
            if type(proof) is not CompletionProof or proof.ticket.transaction != self.transaction:
                self.quarantine = 'stale_identity'
                raise HandoffError('worker backend returned a foreign completion')
            if self.pending_operations.get(proof.ticket.op_seq) is not proof.ticket.dependency_role:
                self.quarantine = 'stale_identity'
                raise HandoffError('proof has no submitted operation')
            del self.pending_operations[proof.ticket.op_seq]
            applied.append(proof)
            self.completions.append(proof)
        return CompletionReport(self.transaction, tuple(applied), len(self.pending_operations), False)

    def commit(self, message):
        self._check(message)
        if self.phase not in (Phase.OPEN, Phase.SUBMITTED):
            raise HandoffError('worker cannot publish after stop/commit')
        match = [p for p in self.completions
                 if p.completion_seq == message.completion_seq
                 and p.kind is CompletionKind.STATE_WRITE
                 and p.ticket.dependency_role is DependencyRole.COMMIT]
        if not match or self.pending_operations:
            raise HandoffError('commit lacks completions for submitted operations')
        if self.phase is Phase.COMMITTED:
            raise HandoffError('worker commit is exactly-once')
        self.phase = Phase.COMMITTED
        return CommitAck(self.transaction, 1, True)

    def cancel(self, message):
        self._check(message)
        if self.phase is Phase.COMMITTED:
            self.phase = Phase.RETIRING
        else:
            self.phase = Phase.RETIRING
        return CancelAck(self.transaction, self.phase.value)


class RuntimeWorkerHandoff:
    """Bind the typed worker RPCs to A's ``LaneRuntime``.

    A owns storage and completion recording; C owns the RPC contract and the
    publication decision. ``open`` prepares the lane, ``submit`` returns only
    after A recorded the ticket (still not a completion), ``poll`` translates
    A's ``PollResult`` into metadata proofs, and ``commit`` requires a proof
    recorded in this adapter. The real backend and native UID/workspace are
    supplied by C startup; this class allocates nothing.
    """

    def __init__(self, runtime, transaction, *, metadata, role=LaneRole.PREFILL,
                 prepare_context=None, prepare_workspace=None, bindings=(),
                 private_attrs=None, clock=time.monotonic):
        _check_transaction('runtime worker handoff', transaction)
        self.runtime = runtime
        self.transaction = transaction
        self.metadata = metadata
        self.role = role
        self.prepare_context = prepare_context
        self.prepare_workspace = prepare_workspace
        self.bindings = tuple(bindings)
        self.private_attrs = dict(private_attrs or {})
        self.clock = clock
        self.completions = []
        self.last_op = -1
        self.last_ticket = None
        self.last_ack = None
        self.phase = Phase.OPEN
        self.quarantine = None

    def apply(self, message):
        message.validate()
        if message.transaction != self.transaction:
            self.quarantine = 'stale_identity'
            raise HandoffError('typed message belongs to another transaction')
        if type(message) is OpenPrefill:
            return self.open(message)
        if type(message) is SubmitOperation:
            return self.submit(message)
        if type(message) is PollOperations:
            return self.poll(message)
        if type(message) is CommitTransaction:
            return self.commit(message)
        if type(message) is CancelTransaction:
            return self.cancel(message)
        raise HandoffError('unknown S5 worker message')

    def _check(self, message, expected):
        if type(message) is not expected:
            raise HandoffError('worker method/message type mismatch')
        message.validate()
        if self.quarantine is not None or message.transaction != self.transaction:
            self.quarantine = 'stale_identity'
            raise HandoffError('worker transaction is stale or quarantined')

    def open(self, message):
        self._check(message, OpenPrefill)
        if self.phase is not Phase.OPEN:
            raise HandoffError('worker transaction is already opened')
        metadata = getattr(message, 'metadata', None)
        if metadata is not None and metadata != self.metadata:
            raise HandoffError('open message metadata differs from the admitted snapshot')
        handle = self.runtime.open_prefill(
            self.transaction, self.metadata, bindings=self.bindings,
            private_attrs=self.private_attrs)
        if getattr(handle, 'lane_id', None) is None or handle.lane_id.role is not LaneRole.PREFILL:
            raise HandoffError('runtime open did not return a prefill lane')
        self.phase = Phase.OPEN
        return OpenAck(self.transaction, handle.lane_id, True)

    def submit(self, message):
        self._check(message, SubmitOperation)
        if self.phase not in (Phase.OPEN, Phase.SUBMITTED):
            raise HandoffError('worker is not accepting operations')
        if message.op_seq <= self.last_op:
            raise HandoffError('worker operation sequence is not monotonic')
        self.last_op = message.op_seq
        ticket = self.runtime.submit(self.role, self.transaction, message.op_seq,
                                     message.dependency_role)
        self.last_ticket = ticket
        self.last_ack = SubmitAck(self.transaction, message.op_seq, True)
        self.phase = Phase.SUBMITTED
        return self.last_ack

    def poll(self, message):
        self._check(message, PollOperations)
        result = self.runtime.poll(self.transaction)
        if not isinstance(result, PollResult):
            raise HandoffError('runtime poll did not return PollResult')
        if len(result.completed) > message.budget:
            self.quarantine = 'backend_failure'
            raise HandoffError('runtime poll returned an unbounded batch')
        self.completions.extend(result.completed)
        return CompletionReport(self.transaction, tuple(result.completed),
                                result.pending, result.failed)

    def commit(self, message):
        self._check(message, CommitTransaction)
        if self.phase not in (Phase.OPEN, Phase.SUBMITTED):
            raise HandoffError('worker cannot publish after stop/commit')
        match = [p for p in self.completions
                 if p.completion_seq == message.completion_seq
                 and p.kind is CompletionKind.STATE_WRITE
                 and p.ticket.dependency_role is DependencyRole.COMMIT]
        if not match:
            raise HandoffError('commit has no recorded state-write completion')
        record = self.runtime.commit(self.transaction, match[-1])
        if self.phase is Phase.COMMITTED:
            raise HandoffError('worker commit is exactly-once')
        epoch = getattr(record, 'epoch', None)
        if (getattr(record, 'transaction', None) != self.transaction
                or type(epoch) is not int or epoch < 1):
            self.quarantine = 'stale_identity'
            raise HandoffError('runtime returned no valid commit record')
        self.phase = Phase.COMMITTED
        return CommitAck(self.transaction, epoch, True)

    def cancel(self, message):
        self._check(message, CancelTransaction)
        self.runtime.abort(self.transaction)
        self.phase = Phase.RETIRING
        return CancelAck(self.transaction, self.phase.value)


# --- startup composition ----------------------------------------------------

class Composition:
    """What enabled startup installs on the engine/executor/scheduler.

    Every native object is injected by the owner-reviewed builder; nothing here
    allocates. The composition records its allocation estimates, route, and the
    exact seams it expects A/B to supply, so a missing piece fails startup
    instead of silently running the null fixture.
    """

    def __init__(self, *, config, dispatcher, ledger, frontier, pp_channel,
                 allocation, route, runtime=None, scheduler=None, executor=None,
                 scheduler_leases=None):
        if type(config) is not HandoffConfig or not config.enabled:
            raise ConfigurationError('composition needs an enabled configuration')
        if type(dispatcher) is not HandoffDispatcher:
            raise ConfigurationError('composition needs the handoff dispatcher')
        for name, value, kind in (('ledger', ledger, LifetimeLedger),
                                  ('frontier', frontier, CommittedFrontier),
                                  ('pp_channel', pp_channel, PrefillPPChannel)):
            if not isinstance(value, kind):
                raise ConfigurationError(f'composition {name} is invalid')
        self.config = config
        self.dispatcher = dispatcher
        self.ledger = ledger
        self.frontier = frontier
        self.pp_channel = pp_channel
        self.allocation = allocation
        self.route = route
        self.runtime = runtime
        self.scheduler = scheduler
        self.executor = executor
        self.scheduler_leases = scheduler_leases

    def attach(self, engine=None, scheduler=None, executor=None):
        """Attach the composition to the actual objects, refused twice."""
        if engine is not None:
            if getattr(engine, '_flash_s5_handoff', None) is not None:
                raise ConfigurationError('engine already has an E3 handoff')
            engine._flash_s5_handoff = self.dispatcher
        if scheduler is not None:
            if getattr(scheduler, '_flash_s5_leases', None) is not None:
                raise ConfigurationError('scheduler already has E3 leases')
            if self.scheduler_leases is not None:
                self.scheduler_leases.attach(scheduler, scheduler.kv_cache_manager.block_pool)
                scheduler._flash_s5_ledger = self.ledger
            else:
                scheduler._flash_s5_lease_ledger = self.ledger
            scheduler._flash_s5_frontier = self.frontier
            scheduler._flash_s5_handoff = self.dispatcher
            self.scheduler = scheduler
        if executor is not None:
            if getattr(executor, '_flash_s5_handoff', None) is not None:
                raise ConfigurationError('executor already has an E3 handoff')
            executor._flash_s5_handoff = self.dispatcher
            self.executor = executor
        return self

    def inventory(self):
        return self.allocation

    def settings(self):
        return {
            OVERLAP_ENV: '1',
            TRANSPORT_ENV: self.config.transport,
            SPARSE_OVERLAP_ENV: '0',
            COMMAND_SLOTS_ENV: str(self.config.command_slots),
            REPLY_TIMEOUT_ENV: repr(self.config.reply_timeout_s),
            ACK_TIMEOUT_ENV: repr(self.config.ack_timeout_s),
            CHUNK_TOKENS_ENV: str(self.config.chunk_tokens),
            BACKEND_ENV: 'native',
        }


def build_composition(env, *, transport, backend, lanes, transaction=None, allocation=None,
                      allow_fixture=False, clock=time.monotonic, telemetry=None,
                      native_allocations=None, runtime=None, scheduler_leases=None,
                      side='worker'):
    """Enabled-startup composition; fixture mode exists only for CPU tests.

    Native allocation, the real QSA UID namespace and the native completion
    backend are caller-supplied. Enabled startup rejects the fixture backend and
    an empty native allocation record, so a fake cannot be mistaken for an
    armable engine. ``transaction`` is optional: the dispatcher creates the
    per-prefill TransactionId when ``open`` is called; a supplied value is
    validated but never invented here.

    ``side`` names which process this composition belongs to. The engine has no
    device: it coordinates typed worker RPCs and owns no A ``LaneRuntime`` or
    private UID buffer, so those worker-only checks are skipped there. The
    engine still refuses the fixture backend and an empty allocation inventory.
    """
    if side not in ('worker', 'engine'):
        raise ConfigurationError('composition side must be worker or engine')
    config = HandoffConfig.from_env(env)
    if not config.enabled:
        return None
    if transaction is not None:
        _check_transaction('build composition', transaction)
    if not allow_fixture:
        require_native_backend(backend)
        if not native_allocations:
            raise ConfigurationError(
                'enabled startup requires C native allocations (UID buffer, PP buffers)')
        if side == 'worker' and (
                runtime is None or getattr(runtime, 'uid_namespace', None) is None
                or getattr(runtime, 'prefill_uid_buffer', None) is None):
            raise ConfigurationError(
                'enabled startup requires A runtime with UID namespace and private buffer')
    if side == 'engine' and lanes is None:
        lanes = {}
    dispatcher = HandoffDispatcher(config, clock=clock, telemetry=telemetry,
                                   allow_fixture=allow_fixture)
    dispatcher.install_transport(transport)
    dispatcher.install_backend(backend)
    telemetry = dispatcher.telemetry
    ledger = LifetimeLedger()
    frontier = CommittedFrontier()
    channel = PrefillPPChannel(stages=2, telemetry=telemetry)  # PP24/24; four TP ranks are not four stages.
    route = admit_route(overlap_enabled=True, transport=config.transport,
                        sparse_active=False, dense_available=True)
    budget = allocation if allocation is not None else allocation_budget(config.chunk_tokens,
                                                                         config.command_slots)
    return Composition(config=config, dispatcher=dispatcher, ledger=ledger,
                       frontier=frontier, pp_channel=channel, allocation=budget,
                       route=route, runtime=runtime,
                       scheduler_leases=scheduler_leases)


def routing_contract():
    """The seams A and B must supply to C's composition (metadata only)."""
    return {
        'lane_runtime': ('open_prefill', 'prepare_prefill', 'submit', 'poll',
                         'propose_state', 'commit', 'abort', 'release'),
        'graph_router': ('supported_route', 'add_bank', 'capture_at_startup',
                         'replay', 'complete_replay', 'transition', 'shutdown'),
        'executor': ('collective_rpc', 'unique_reply_rank', 'aggregate_order'),
        'scheduler': ('mutation_guard', 'defer_request_free', 'committed_frontier'),
        'pp': ('dedicated_prefill_buffers', 'bounded_outstanding', 'no_sample_ring'),
    }
