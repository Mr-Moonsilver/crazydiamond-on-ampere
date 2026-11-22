"""E3 shared transaction, lane and operation-ticket identity (CPU-only contract).

This module is the frozen interface A supplies to B (graph routing) and C
(executor/engine handoffs). It imports neither torch nor vLLM and performs no
I/O, so B/C can use it in CPU fixtures and in generated runtime code.

Identity rules from ``docs/e3-briefs/README.md``:

* A transaction is identified by engine epoch, request UID and generation,
  chunk bounds and a monotonic transaction sequence -- never by a reusable
  batch row.
* A lane is identified by worker rank, decode/prefill role, storage epoch and
  context identity.
* An operation ticket carries the transaction, rank, lane, a monotonically
  ordered operation sequence and a dependency role. An RPC acknowledgment is
  *not* a completion: :class:`CompletionProof` has no RPC-ack kind.

Everything here is immutable (frozen) and validated at construction. Request
UIDs are redacted from ``repr`` because the E2/E3 runtime rule is that request
IDs are never logged.
"""
from dataclasses import dataclass, field
from enum import Enum

# One active bulk-prefill transaction globally in the first fixed-split
# increment. Adaptive/multi-prefill switching is later work.
MAX_ACTIVE_PREFILL = 1

# Main QSA KV row: 1 local head x (K+V) x 256 x 2 bytes = 1024 B.
PAGE_TOKENS = 4048
PAGE_BYTES = PAGE_TOKENS * 1024              # 4_145_152
ARENA_PAGES = 24                             # existing E2 arena, reused
ARENA_BYTES = ARENA_PAGES * PAGE_BYTES       # 99_483_648
BUCKETS_AT_200K = (24, 24, 2)                # baseline bucket order, 50 pages
HISTORIES_PER_RANK = 6                       # six independent QSA histories


@dataclass(frozen=True)
class TensorFacts:
    """Geometry/ownership facts for one bound tensor (never its storage).

    Mirrors the E2 serial-DMA record so the separately mounted E3 package does
    not import an E2 runtime module.  ``address`` is for internal overlap
    checks only and is never written to reports or logs.
    """
    device: str
    shape: tuple
    strides: tuple
    dtype: str
    item_bytes: int
    pinned: bool
    contiguous: bool
    address: int = field(repr=False)

    def __post_init__(self):
        if (type(self.device) is not str or type(self.dtype) is not str
                or type(self.shape) is not tuple or type(self.strides) is not tuple
                or len(self.shape) != len(self.strides)
                or any(type(n) is not int or n < 0 for n in self.shape + self.strides)
                or type(self.item_bytes) is not int or self.item_bytes <= 0
                or type(self.address) is not int or self.address < 0
                or type(self.pinned) is not bool or type(self.contiguous) is not bool):
            raise E3ContractError('invalid immutable tensor geometry')

    @property
    def nbytes(self):
        n = self.item_bytes
        for dimension in self.shape:
            n *= dimension
        return n


class E3ContractError(ValueError):
    """A supplied identity, ownership or completion value violates the contract."""


class E3Pending(RuntimeError):
    """An owned operation is still outstanding; retry after its completion proof."""


class LaneRole(str, Enum):
    DECODE = 'decode'
    PREFILL = 'prefill'


class DependencyRole(str, Enum):
    PRODUCER = 'producer'       # writes storage or source pages
    CONSUMER = 'consumer'       # reads storage; release waits for this proof
    HANDOFF = 'handoff'         # PP/transfer consumer
    COMMIT = 'commit'           # state publication boundary


class CompletionKind(str, Enum):
    """Kinds of real completion C may turn into a proof.

    Deliberately absent: an RPC acknowledgment. ``FutureWrapper.result()`` or
    a worker method returning is not a GPU completion and cannot release a
    source/arena lease.
    """
    GPU_EVENT = 'gpu_event'
    COPY_EVENT = 'copy_event'
    PP_CONSUMER = 'pp_consumer'
    STATE_WRITE = 'state_write'


# Inclusive origin of every integer field. Validation is strict about ``bool``
# (a bool is an int subclass) so a caller cannot smuggle a flag as a sequence.
_INT_MIN = {
    'engine_epoch': 0,
    'request_generation': 0,
    'chunk_start': 0,
    'chunk_end': 1,
    'txn_seq': 0,
    'worker_rank': 0,
    'storage_epoch': 1,
    'context_id': 0,
    'op_seq': 0,
    'completion_seq': 0,
    'pending': 0,
}


def _int(name, value):
    if type(value) is not int:
        raise E3ContractError(f'{name} must be an int, not {type(value).__name__}')
    if value < _INT_MIN[name]:
        raise E3ContractError(f'{name} must be >= {_INT_MIN[name]}')
    return value


def _text(name, value):
    if not isinstance(value, str) or not value:
        raise E3ContractError(f'{name} must be a non-empty string')
    if len(value) > 256 or any(ord(c) < 32 for c in value):
        raise E3ContractError(f'{name} contains unsupported characters')
    return value


def _role(name, value, enum):
    if type(value) is not enum:
        raise E3ContractError(f'{name} must be {enum.__name__}')
    return value


@dataclass(frozen=True, repr=False)
class TransactionId:
    """Immutable identity of one prefill/decode chunk transaction."""
    engine_epoch: int
    request_uid: str
    request_generation: int
    chunk_start: int
    chunk_end: int
    txn_seq: int

    def __post_init__(self):
        _int('engine_epoch', self.engine_epoch)
        _text('request_uid', self.request_uid)
        _int('request_generation', self.request_generation)
        _int('chunk_start', self.chunk_start)
        _int('chunk_end', self.chunk_end)
        _int('txn_seq', self.txn_seq)
        if self.chunk_end <= self.chunk_start:
            raise E3ContractError('chunk_end must exceed chunk_start')

    @property
    def request_key(self):
        """The reusable-request identity, without a chunk or txn sequence."""
        return self.engine_epoch, self.request_uid, self.request_generation

    def same_request(self, other):
        return isinstance(other, TransactionId) and self.request_key == other.request_key

    def __repr__(self):
        return (f'TransactionId(engine_epoch={self.engine_epoch}, '
                f'request_uid=<len {len(self.request_uid)}>, '
                f'request_generation={self.request_generation}, '
                f'chunk_start={self.chunk_start}, chunk_end={self.chunk_end}, '
                f'txn_seq={self.txn_seq})')


@dataclass(frozen=True, repr=False)
class LaneId:
    """Immutable storage/context identity for one worker lane.

    ``storage_epoch`` changes on every lane rebind; ``context_id`` is B's
    green/full execution-context identity. A handle carrying an older epoch is
    never interchangeable with the current one.
    """
    worker_rank: int
    role: LaneRole
    storage_epoch: int
    context_id: int

    def __post_init__(self):
        _int('worker_rank', self.worker_rank)
        _role('role', self.role, LaneRole)
        _int('storage_epoch', self.storage_epoch)
        _int('context_id', self.context_id)

    def __repr__(self):
        return (f'LaneId(worker_rank={self.worker_rank}, role={self.role.value}, '
                f'storage_epoch={self.storage_epoch}, context_id={self.context_id})')


@dataclass(frozen=True, repr=False)
class DecodeStorageOwner:
    """The worker's persistent decode storage for its whole incarnation.

    Unlike a prefill lane, this owner is retained once at worker startup and is
    never reserved by a request/chunk transaction: captured decode graphs stay
    bound to exactly this storage epoch while chunks come and go.  The engine
    binds its epoch to ``incarnation`` exactly once; a foreign or late rebind is
    refused by :class:`lane_state.WorkerState`.
    """
    worker_rank: int
    incarnation: str
    storage_epoch: int
    context_id: int

    def __post_init__(self):
        _int('worker_rank', self.worker_rank)
        _text('incarnation', self.incarnation)
        _int('storage_epoch', self.storage_epoch)
        _int('context_id', self.context_id)

    @property
    def lane_id(self):
        return LaneId(self.worker_rank, LaneRole.DECODE, self.storage_epoch,
                      self.context_id)

    def __repr__(self):
        return (f'DecodeStorageOwner(worker_rank={self.worker_rank}, '
                f'incarnation=<len {len(self.incarnation)}>, '
                f'storage_epoch={self.storage_epoch}, context_id={self.context_id})')


@dataclass(frozen=True, repr=False)
class OperationTicket:
    """One bounded submission. It is not a completion or a commit."""
    transaction: TransactionId
    worker_rank: int
    lane: LaneId
    op_seq: int
    dependency_role: DependencyRole

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise E3ContractError('ticket.transaction must be TransactionId')
        _int('worker_rank', self.worker_rank)
        if type(self.lane) is not LaneId:
            raise E3ContractError('ticket.lane must be LaneId')
        _int('op_seq', self.op_seq)
        _role('dependency_role', self.dependency_role, DependencyRole)
        if self.lane.worker_rank != self.worker_rank:
            raise E3ContractError('ticket worker rank disagrees with lane')

    @property
    def sequence_key(self):
        """Per-worker, per-lane monotonic operation ordering."""
        return self.worker_rank, self.lane.role.value, self.op_seq

    def __repr__(self):
        return (f'OperationTicket(txn_seq={self.transaction.txn_seq}, '
                f'worker_rank={self.worker_rank}, role={self.lane.role.value}, '
                f'storage_epoch={self.lane.storage_epoch}, op_seq={self.op_seq}, '
                f'dependency={self.dependency_role.value})')


@dataclass(frozen=True)
class CompletionProof:
    """C's proof that one submitted operation actually completed.

    Only :class:`CompletionKind` values exist; in particular there is no
    ``rpc_ack`` kind. A proof for a different transaction/lane/epoch is
    rejected by the lane adapter, and so is a proof with no outstanding match.
    """
    ticket: OperationTicket
    kind: CompletionKind
    completion_seq: int

    def __post_init__(self):
        if type(self.ticket) is not OperationTicket:
            raise E3ContractError('proof.ticket must be OperationTicket')
        _role('kind', self.kind, CompletionKind)
        _int('completion_seq', self.completion_seq)

    def __repr__(self):
        return (f'CompletionProof(txn_seq={self.ticket.transaction.txn_seq}, '
                f'op_seq={self.ticket.op_seq}, kind={self.kind.value}, '
                f'completion_seq={self.completion_seq})')


@dataclass(frozen=True)
class PollResult:
    """Bounded, nonblocking completion query result handed to C."""
    completed: tuple
    pending: int
    failed: bool

    def __post_init__(self):
        if type(self.completed) is not tuple:
            raise E3ContractError('completed must be an immutable tuple')
        for proof in self.completed:
            if type(proof) is not CompletionProof:
                raise E3ContractError('completed entries must be CompletionProof')
        _int('pending', self.pending)
        if type(self.failed) is not bool:
            raise E3ContractError('failed must be bool')
