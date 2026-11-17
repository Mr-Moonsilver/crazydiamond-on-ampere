"""E3 lane storage and worker-state adapter (offline, CPU-only).

Owns what A's brief requires: immutable shared weights, per-request persistent
state, per-lane scratch, graph-stable buffer registration, explicitly serialized
shared storage, the existing 24-page arena and six independent QSA histories.
It imports neither torch nor vLLM; buffers are recorded by address/size facts and
lifetimes, not touched.

Publication boundaries are deliberate:

* :meth:`LaneStorage.submit` records a ticket; it never returns a completion.
* :meth:`LaneStorage.complete` accepts only C's :class:`CompletionProof`.
* :meth:`LaneStorage.release` requires every submitted operation to have a proof.
* :meth:`PersistentState.commit` applies a pending persistent value exactly once.
* :meth:`LaneStorage.abort` keeps consumed storage leased until completion.
* A failed completion quarantines the lane; release and rebind stay refused.

This is the contract B and C import. It is not an engine overlay and establishes
nothing about CUDA completion, aliasing or speed on a GPU.
"""
from dataclasses import dataclass, field, replace
import uuid

from e3_types import (
    ARENA_BYTES,
    ARENA_PAGES,
    BUCKETS_AT_200K,
    CompletionKind,
    CompletionProof,
    DecodeStorageOwner,
    DependencyRole,
    E3ContractError,
    E3Pending,
    HISTORIES_PER_RANK,
    LaneId,
    LaneRole,
    PAGE_BYTES,
    TransactionId,
)

# 200k prompt = 50 pages; six independent histories per rank on each worker.
HISTORY_PAGES_AT_200K = 50
HISTORY_BYTES_PER_RANK = HISTORIES_PER_RANK * HISTORY_PAGES_AT_200K * PAGE_BYTES  # 1_243_545_600

# Sized components from reviews/codex-20261025s-s5-engine-design.md section 7.
HC_HIDDEN_BYTES = 1024 * 4 * 2560 * 2          # 20 MiB
HC_BLOCK_OUTPUT_BYTES = 1024 * 2560 * 2        # 5 MiB
HC_INJECTION_BYTES = 1024 * 4 * 2              # 8 KiB
TOPK_BYTES = 1024 * 2052 * 4                   # 8.015625 MiB
QSA_ACCUM_BYTES = 12_681_216                   # 12.09375 MiB
PP_HIDDEN_BYTES = 20_971_520                   # 20 MiB
GRAPH_POOL_MEASURED_BYTES = 150_323_856        # 0.14 GiB measured, docs/MEMORY-LEDGER.md

BASES = ('measured', 'estimate', 'existing', 'declared-zero', 'unmeasured')
_SERIALIZED = 'serialized'


def _int(name, value, minimum=0):
    if type(value) is not int:
        raise E3ContractError(f'{name} must be an int, not {type(value).__name__}')
    if value < minimum:
        raise E3ContractError(f'{name} must be >= {minimum}')
    return value


def _text(name, value):
    if not isinstance(value, str) or not value:
        raise E3ContractError(f'{name} must be a non-empty string')
    return value


def _ints(name, values, minimum=0):
    values = tuple(values)
    for value in values:
        _int(name, value, minimum)
    return values


def _freeze(value):
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, set):
        return tuple(sorted(_freeze(v) for v in value))
    if value is None or type(value) in (str, int, float, bool, bytes):
        return value
    raise E3ContractError('mutable/native state must use an explicitly owned buffer, not a CPU snapshot')


@dataclass(frozen=True, repr=False)
class BufferBinding:
    """Address/size facts for one lane-owned buffer; the bytes are not read."""
    name: str
    owner: str
    category: str
    address: int
    nbytes: int
    basis: str = 'estimate'
    shared: bool = False
    reuse_condition: str | None = None

    def __post_init__(self):
        _text('name', self.name)
        _text('owner', self.owner)
        _text('category', self.category)
        _int('address', self.address)
        _int('nbytes', self.nbytes, 1)
        if self.basis not in BASES:
            raise E3ContractError(f'unknown basis {self.basis!r}')
        if type(self.shared) is not bool:
            raise E3ContractError('shared must be bool')
        if self.shared and not self.reuse_condition:
            raise E3ContractError('shared serialized storage needs a reuse condition')
        if self.reuse_condition is not None:
            _text('reuse_condition', self.reuse_condition)

    @property
    def span(self):
        return self.address, self.address + self.nbytes

    def overlaps(self, other):
        lo, hi = self.span
        olo, ohi = other.span
        return max(lo, olo) < min(hi, ohi)

    def __repr__(self):
        return (f'BufferBinding(name={self.name!r}, owner={self.owner!r}, '
                f'category={self.category!r}, address={self.address}, nbytes={self.nbytes}, '
                f'basis={self.basis!r}, shared={self.shared})')


@dataclass(frozen=True)
class ScratchSpec:
    """A declared scratch allocation before an address is bound."""
    name: str
    owner: str
    category: str
    nbytes: int
    basis: str = 'estimate'
    shared: bool = False
    reuse_condition: str | None = None

    def __post_init__(self):
        _text('name', self.name)
        _text('owner', self.owner)
        _text('category', self.category)
        _int('nbytes', self.nbytes, 1)
        if self.basis not in BASES:
            raise E3ContractError(f'unknown basis {self.basis!r}')
        if type(self.shared) is not bool:
            raise E3ContractError('shared must be bool')
        if self.shared and not self.reuse_condition:
            raise E3ContractError('shared scratch needs a reuse condition')

    def bind(self, address):
        return BufferBinding(self.name, self.owner, self.category, address, self.nbytes,
                             self.basis, self.shared, self.reuse_condition)


@dataclass(frozen=True)
class InventoryEntry:
    name: str
    owner: str
    category: str
    nbytes: int
    basis: str
    new_allocation: bool
    reuse_condition: str | None = None

    def __post_init__(self):
        _text('name', self.name)
        _text('owner', self.owner)
        _text('category', self.category)
        _int('nbytes', self.nbytes)
        if self.basis not in BASES:
            raise E3ContractError(f'unknown basis {self.basis!r}')
        if type(self.new_allocation) is not bool:
            raise E3ContractError('new_allocation must be bool')
        if self.basis == 'unmeasured' and self.nbytes != 0:
            raise E3ContractError('unmeasured entries record zero bytes')
        if self.basis == 'declared-zero' and self.nbytes != 0:
            raise E3ContractError('declared-zero entries record zero bytes')


@dataclass(frozen=True)
class InventoryReport:
    entries: tuple

    def __post_init__(self):
        if type(self.entries) is not tuple:
            raise E3ContractError('entries must be an immutable tuple')
        seen = set()
        for entry in self.entries:
            if type(entry) is not InventoryEntry:
                raise E3ContractError('entries must be InventoryEntry')
            key = (entry.category, entry.name, entry.owner)
            if key in seen:
                raise E3ContractError('duplicate inventory entry')
            seen.add(key)

    @property
    def total_bytes(self):
        return sum(e.nbytes for e in self.entries)

    @property
    def new_bytes(self):
        return sum(e.nbytes for e in self.entries if e.new_allocation)

    @property
    def existing_bytes(self):
        return sum(e.nbytes for e in self.entries if not e.new_allocation)

    def by_category(self):
        out = {}
        for entry in self.entries:
            new, old = out.get(entry.category, (0, 0))
            out[entry.category] = ((new + entry.nbytes) if entry.new_allocation else new,
                                   old if entry.new_allocation else old + entry.nbytes)
        return out

    def unmeasured(self):
        return tuple(e.name for e in self.entries if e.basis == 'unmeasured')

    def require_bounded(self, limit):
        _int('limit', limit)
        if any(e.new_allocation and e.basis == 'unmeasured' for e in self.entries):
            raise E3ContractError('new allocation budget has unmeasured components')
        if self.new_bytes > limit:
            raise E3ContractError(f'E3 new allocation {self.new_bytes} exceeds budget {limit}')
        return self


@dataclass(frozen=True, repr=False)
class BatchRow:
    request_uid: str
    request_generation: int
    row: int
    chunk_start: int
    chunk_end: int
    num_computed: int

    def __post_init__(self):
        _text('request_uid', self.request_uid)
        _int('request_generation', self.request_generation)
        _int('row', self.row)
        _int('chunk_start', self.chunk_start)
        _int('chunk_end', self.chunk_end)
        _int('num_computed', self.num_computed)
        if self.chunk_end <= self.chunk_start:
            raise E3ContractError('batch row chunk is empty')

    def __repr__(self):
        return (f'BatchRow(request_uid=<len {len(self.request_uid)}>, '
                f'request_generation={self.request_generation}, row={self.row}, '
                f'chunk_start={self.chunk_start}, chunk_end={self.chunk_end}, '
                f'num_computed={self.num_computed})')


@dataclass(frozen=True)
class MetadataSnapshot:
    """Immutable CPU metadata captured once at admission."""
    transaction: TransactionId
    rows: tuple
    block_ids: tuple = ()
    slot_mappings: tuple = ()
    num_computed: tuple = ()
    batch_seq: int = 0

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise E3ContractError('snapshot transaction must be TransactionId')
        if any(type(x) is not tuple for x in (self.rows, self.block_ids, self.slot_mappings, self.num_computed)):
            raise E3ContractError('snapshot collections must be immutable tuples')
        if not self.rows or any(type(r) is not BatchRow for r in self.rows):
            raise E3ContractError('snapshot needs immutable batch rows')
        _int('batch_seq', self.batch_seq)
        _ints('block_id', self.block_ids)
        _ints('slot_mapping', self.slot_mappings, -1)
        _ints('num_computed', self.num_computed)
        for i, row in enumerate(self.rows):
            if (row.row, row.request_uid, row.request_generation, row.chunk_start, row.chunk_end) != (
                    i, self.transaction.request_uid, self.transaction.request_generation,
                    self.transaction.chunk_start, self.transaction.chunk_end):
                raise E3ContractError('snapshot row differs from transaction')
        if any(x and len(x) != len(self.rows) for x in (self.slot_mappings, self.num_computed)):
            raise E3ContractError('snapshot mapping lengths differ')

    @classmethod
    def capture(cls, transaction, rows, *, block_ids=(), slot_mappings=(), num_computed=()):
        if type(transaction) is not TransactionId:
            raise E3ContractError('snapshot.transaction must be TransactionId')
        rows = tuple(rows)
        if not rows:
            raise E3ContractError('metadata snapshot needs at least one row')
        checked = []
        for index, row in enumerate(rows):
            if type(row) is not BatchRow:
                raise E3ContractError('snapshot rows must be BatchRow')
            if row.row != index:
                raise E3ContractError('snapshot rows must be in batch order')
            if (row.request_uid, row.request_generation) != (
                    transaction.request_uid, transaction.request_generation):
                raise E3ContractError('row request identity differs from transaction')
            if (row.chunk_start, row.chunk_end) != (
                    transaction.chunk_start, transaction.chunk_end):
                raise E3ContractError('row chunk differs from transaction')
            checked.append(row)
        block_ids = _ints('block_id', block_ids)
        slot_mappings = _ints('slot_mapping', slot_mappings, -1)
        num_computed = _ints('num_computed', num_computed)
        if slot_mappings and len(slot_mappings) != len(checked):
            raise E3ContractError('slot mapping count differs from rows')
        if num_computed and len(num_computed) != len(checked):
            raise E3ContractError('computed count differs from rows')
        return cls(transaction, tuple(checked), block_ids, slot_mappings, num_computed)

    def with_sequence(self, sequence):
        _int('sequence', sequence)
        return replace(self, batch_seq=sequence)

    def check(self, rows):
        if tuple(rows) != self.rows:
            raise E3ContractError('stale batch row')
        return self


@dataclass(frozen=True)
class RecordedProof:
    """A proof kept in a commit record; the ticket is not replayed."""
    transaction: TransactionId
    kind: CompletionKind
    completion_seq: int


@dataclass(frozen=True)
class CommitRecord:
    transaction: TransactionId
    value: object
    epoch: int
    proof: RecordedProof


@dataclass(frozen=True)
class PersistentPending:
    transaction: TransactionId
    value: object


class PersistentState:
    """Per-lane persistent state with exactly-once commit.

    ``propose`` stages a candidate; ``commit`` applies it once. A duplicate
    ``commit`` for the same transaction returns the recorded result without
    re-applying. A proof is required and must be a real state-write completion
    for this lane/transaction; an RPC ack cannot be constructed as one.
    """

    def __init__(self, lane):
        self.lane = lane
        self.epoch = 0
        self.committed = None
        self.pending = None
        self.applied = 0

    def reset(self):
        self.epoch = 0
        self.committed = None
        self.pending = None
        self.applied = 0

    def propose(self, transaction, value):
        self.lane._require_active(transaction)
        if self.lane._aborted:
            raise E3ContractError('transaction is aborted; no new persistent state')
        if self.committed is not None and self.committed.transaction == transaction:
            raise E3ContractError('persistent state was already committed for this transaction')
        self.pending = PersistentPending(transaction, _freeze(value))

    def cancel_pending(self):
        self.pending = None

    def commit(self, transaction, proof):
        if type(transaction) is not TransactionId:
            raise E3ContractError('commit transaction must be TransactionId')
        if (self.committed is not None and
                self.committed.transaction == transaction):
            # Exactly-once delivery: a duplicate acknowledgement is not a
            # second application of the persistent state.
            return self.committed
        self.lane._require_active(transaction)
        if type(proof) is not CompletionProof:
            raise E3ContractError('persistent commit requires a CompletionProof')
        if proof.kind is not CompletionKind.STATE_WRITE:
            raise E3ContractError('persistent commit requires a state-write completion')
        if proof.ticket.transaction != transaction:
            raise E3ContractError('completion proof belongs to another transaction')
        if proof.ticket.lane != self.lane.lane_id:
            raise E3ContractError('completion proof belongs to another lane')
        if proof.ticket.dependency_role is not DependencyRole.COMMIT:
            raise E3ContractError('completion proof is not a commit-role operation')
        if proof not in self.lane.completed:
            raise E3ContractError('commit needs a recorded completion of a submitted operation')
        if self.lane.outstanding:
            raise E3Pending('commit cannot publish while operations remain outstanding')
        if self.pending is None or self.pending.transaction != transaction:
            raise E3ContractError('no pending persistent state for this transaction')
        self.epoch += 1
        recorded = RecordedProof(transaction, proof.kind, proof.completion_seq)
        self.committed = CommitRecord(transaction, self.pending.value, self.epoch, recorded)
        self.pending = None
        self.applied += 1
        return self.committed

    def snapshot(self):
        value = self.committed.value if self.committed is not None else None
        transaction = self.committed.transaction if self.committed is not None else None
        return self.epoch, transaction, value


@dataclass(frozen=True, repr=False)
class QsaHistory:
    layer_name: str
    rank: int
    address: int
    nbytes: int

    def __post_init__(self):
        _text('layer_name', self.layer_name)
        _int('rank', self.rank)
        _int('address', self.address)
        _int('nbytes', self.nbytes, 1)
        if self.nbytes % PAGE_BYTES:
            raise E3ContractError('history allocation must contain complete discovered pages')

    @property
    def span(self):
        return self.address, self.address + self.nbytes

    def __repr__(self):
        return (f'QsaHistory(layer_name={self.layer_name!r}, rank={self.rank}, '
                f'address={self.address}, nbytes={self.nbytes})')


@dataclass(frozen=True)
class QsaHistories:
    """Exactly six independent histories per rank; no extra window."""
    histories: tuple

    def __post_init__(self):
        if type(self.histories) is not tuple or len(self.histories) != HISTORIES_PER_RANK:
            raise E3ContractError('exactly six independent QSA histories are required')
        names, spans = set(), []
        for history in self.histories:
            if type(history) is not QsaHistory:
                raise E3ContractError('histories must be QsaHistory')
            if history.layer_name in names:
                raise E3ContractError('duplicate QSA layer history')
            names.add(history.layer_name)
            lo, hi = history.span
            if any(max(lo, a) < min(hi, b) for a, b in spans):
                raise E3ContractError('QSA histories alias')
            spans.append((lo, hi))
        if len({history.rank for history in self.histories}) != 1:
            raise E3ContractError('histories belong to different ranks')

    @property
    def bytes(self):
        return sum(h.nbytes for h in self.histories)


@dataclass(frozen=True, repr=False)
class HcTupleBinding:
    """Private delayed HC tuple: hidden, block output, injection, cursor."""
    hidden: object = field(repr=False)
    block_output: object = field(repr=False)
    injection: object = field(repr=False)
    cursor: int
    addresses: tuple

    def __post_init__(self):
        _int('cursor', self.cursor)
        if type(self.addresses) is not tuple or len(self.addresses) != 3:
            raise E3ContractError('HC addresses must be a 3-tuple')
        _ints('HC address', self.addresses)

    def __repr__(self):
        return f'HcTupleBinding(cursor={self.cursor}, addresses={self.addresses})'


@dataclass(frozen=True)
class RecurrentSnapshot:
    transaction: TransactionId
    values: tuple


@dataclass(frozen=True)
class GraphPoolBinding:
    name: str
    lane: LaneId
    address: int
    nbytes: int
    basis: str

    def __post_init__(self):
        _text('name', self.name)
        if type(self.lane) is not LaneId:
            raise E3ContractError('graph pool needs a LaneId')
        _int('address', self.address)
        _int('nbytes', self.nbytes, 1)
        if self.basis not in BASES:
            raise E3ContractError('unknown graph-pool basis')


class PrivateUidBank:
    """Prefill-owned QSA UID slots; never the decode graph's global buffer."""

    def __init__(self):
        self._next = 0
        self._slots = {}

    def new(self, req_index):
        _int('req_index', req_index)
        self._next = self._next % (2 ** 31 - 2) + 1
        self._slots[req_index] = self._next
        return self._next

    def write(self, target, indices):
        for index in indices:
            _int('req_index', index)
        for position in range(len(target)):
            target[position] = -2
        for position, index in enumerate(indices):
            target[position] = self._slots.get(int(index), -2)

    def snapshot(self):
        return tuple(sorted(self._slots.items()))


@dataclass(frozen=True, repr=False)
class LaneHandle:
    """Opaque, immutable handle B/C hold. Rebinding creates a new epoch."""
    _lane: LaneId
    _token: object = field(compare=False, repr=False)

    @property
    def lane_id(self):
        return self._lane

    @property
    def role(self):
        return self._lane.role

    @property
    def worker_rank(self):
        return self._lane.worker_rank

    @property
    def storage_epoch(self):
        return self._lane.storage_epoch

    @property
    def context_id(self):
        return self._lane.context_id

    def __repr__(self):
        return (f'LaneHandle(worker_rank={self.worker_rank}, role={self.role.value}, '
                f'storage_epoch={self.storage_epoch}, context_id={self.context_id})')


class LaneStorage:
    """One lane's mutable storage and lifetime state."""

    def __init__(self, worker, role, context_id):
        self.worker = worker
        self.role = role
        self.context_id = context_id
        self.state = 'free'
        self.storage_epoch = 0
        self.transaction = None
        self.metadata = None
        self.handle = None
        self.buffers = {}
        self.scratch = {}
        self.owner = None
        self.outstanding = {}
        self.completed = []
        self.graph_pools = {}
        self.histories = None
        self.hc = None
        self.recurrent = None
        self.uid = PrivateUidBank()
        self.persistent = PersistentState(self)
        self._aborted = False
        self._quarantine = None
        self._last_op_seq = -1
        self._last_completion_seq = -1
        self._batch_seq = 0
        self._latest_batch = None
        self._hc_materialized = False

    @property
    def lane_id(self):
        if self.storage_epoch < 1:
            raise E3ContractError('free lane has no storage epoch')
        return LaneId(self.worker.worker_rank, self.role, self.storage_epoch,
                      self.context_id)

    @property
    def active(self):
        return self.state in ('retained', 'reserved', 'ready', 'in_use', 'retiring')

    @property
    def failed(self):
        """True after a failed completion; the lane keeps every lease."""
        return self._quarantine is not None

    def _require_active(self, transaction=None):
        if not self.active:
            raise E3ContractError('lane is not reserved')
        if self._quarantine is not None:
            raise E3ContractError('lane is quarantined')
        if transaction is not None and transaction != self.transaction:
            raise E3ContractError('transaction does not own this lane')

    def _check_ticket(self, ticket):
        if self.handle is None:
            raise E3ContractError('lane has no handle')
        if ticket.lane != self.handle.lane_id:
            raise E3ContractError('ticket lane/storage epoch is stale')
        if ticket.transaction != self.transaction:
            raise E3ContractError('ticket transaction is stale')
        if ticket.worker_rank != self.worker.worker_rank:
            raise E3ContractError('ticket worker rank is foreign')

    def reserve(self, transaction, *, metadata=None, bindings=(), context_id=None):
        if self.state != 'free':
            raise E3Pending('lane is still leased; release needs completion proof')
        if self.outstanding:
            raise E3Pending('lane has outstanding operations')
        if type(transaction) is not TransactionId:
            raise E3ContractError('reserve needs TransactionId')
        self.worker.check_transaction(transaction)
        if self.role is LaneRole.PREFILL:
            if self.worker.active_prefill is not None:
                raise E3Pending('one active prefill transaction is already leased')
            if metadata is None:
                raise E3ContractError('prefill reservation needs an immutable metadata snapshot')
        if metadata is not None:
            if type(metadata) is not MetadataSnapshot:
                raise E3ContractError('metadata must be MetadataSnapshot')
            if metadata.transaction != transaction:
                raise E3ContractError('metadata snapshot belongs to another transaction')
        self._drop_bindings()
        self.storage_epoch += 1
        if context_id is not None:
            _int('context_id', context_id)
            self.context_id = context_id
        self.transaction = transaction
        self.metadata = metadata
        self.state = 'reserved'
        self.outstanding = {}
        self.completed = []
        self.graph_pools = {}
        self.histories = None
        self.hc = None
        self.recurrent = None
        self.uid = PrivateUidBank()
        self.persistent.reset()
        self._aborted = False
        self._quarantine = None
        self._last_op_seq = -1
        self._last_completion_seq = -1
        self._batch_seq = 0
        self._latest_batch = None
        self._hc_materialized = False
        self.handle = LaneHandle(self.lane_id, object())
        for item in bindings:
            if isinstance(item, ScratchSpec):
                raise E3ContractError('bindings need (ScratchSpec, address) pairs')
            spec, address = item
            self.bind_scratch(spec, address)
        if self.role is LaneRole.PREFILL:
            self.worker.active_prefill = transaction
        return self.handle

    def retain(self, *, context_id=None, incarnation=None):
        """Reserve the decode lane once, outside every request transaction.

        The decoded-graph bank is captured against this storage epoch and keeps
        it for the worker's incarnation.  A prefill/request transaction never
        owns or advances it, so a route transition cannot invalidate a captured
        graph.  Idempotent for the identical owner; any other state is refused.
        """
        if self.role is not LaneRole.DECODE:
            raise E3ContractError('only the decode lane retains persistent storage')
        if type(incarnation) is not str or not incarnation:
            raise E3ContractError('retained decode storage needs a worker incarnation')
        _text('incarnation', incarnation)
        if self.state == 'retained':
            owner = getattr(self, 'owner', None)
            if (owner is not None and owner.incarnation == incarnation
                    and (context_id is None or owner.context_id == context_id)):
                return owner
            raise E3ContractError('decode storage is already retained by another owner')
        if self.state != 'free':
            raise E3Pending('decode lane is still leased by a request transaction')
        if self.outstanding:
            raise E3Pending('decode lane has outstanding operations')
        if self.transaction is not None:
            raise E3ContractError('decode lane still carries a request transaction')
        if context_id is not None:
            _int('context_id', context_id)
            self.context_id = context_id
        self.storage_epoch += 1
        self.transaction = None
        self.metadata = None
        self.state = 'retained'
        self.outstanding = {}
        self.completed = []
        self.graph_pools = {}
        self.histories = None
        self.hc = None
        self.recurrent = None
        self.uid = PrivateUidBank()
        self.persistent.reset()
        self._aborted = False
        self._quarantine = None
        self._last_op_seq = -1
        self._last_completion_seq = -1
        self._batch_seq = 0
        self._latest_batch = None
        self._hc_materialized = False
        self.handle = LaneHandle(self.lane_id, object())
        self.owner = DecodeStorageOwner(self.worker.worker_rank, incarnation,
                                        self.storage_epoch, self.context_id)
        return self.owner

    def ready(self):
        if self.state != 'reserved':
            raise E3ContractError('only a reserved lane can become ready')
        self.state = 'ready'

    def bind_scratch(self, spec, address):
        self._require_active()
        if type(spec) is not ScratchSpec:
            raise E3ContractError('bind_scratch needs ScratchSpec')
        if spec.name in self.buffers:
            raise E3ContractError('scratch name already bound')
        binding = spec.bind(address)
        self.worker.register_binding(binding, self.role)
        self.buffers[spec.name] = binding
        self.scratch[spec.name] = spec
        return binding

    def _drop_bindings(self):
        self.worker.drop_bindings(self.role)
        self.buffers = {}
        self.scratch = {}

    def register_graph_pool(self, name, address, nbytes, basis='estimate'):
        self._require_active()
        if self.handle is None:
            raise E3ContractError('lane has no handle')
        _text('name', name)
        if name in self.graph_pools.get(self.storage_epoch, {}):
            raise E3ContractError('graph pool already registered for this epoch')
        pool = GraphPoolBinding(name, self.handle.lane_id, address, nbytes, basis)
        self.graph_pools.setdefault(self.storage_epoch, {})[name] = pool
        return pool

    def retire_graph_pools(self, epoch=None):
        if self.state == 'free':
            raise E3ContractError('no graph pools on a free lane')
        epoch = self.storage_epoch if epoch is None else epoch
        _int('epoch', epoch)
        if epoch != self.storage_epoch:
            raise E3ContractError('graph pool epoch is stale')
        if self.outstanding:
            raise E3Pending('graph users are still outstanding')
        pools = tuple(self.graph_pools.pop(epoch, {}).values())
        return pools

    def note_batch(self, snapshot):
        self._require_active()
        if type(snapshot) is not MetadataSnapshot:
            raise E3ContractError('note_batch needs MetadataSnapshot')
        if snapshot.transaction != self.transaction:
            raise E3ContractError('batch belongs to another transaction')
        if self.metadata is not None and replace(snapshot, batch_seq=0) != replace(self.metadata, batch_seq=0):
            raise E3ContractError('batch differs from immutable admission metadata')
        self._batch_seq += 1
        self._latest_batch = snapshot.with_sequence(self._batch_seq)
        return self._latest_batch

    def check_batch(self, snapshot):
        self._require_active()
        if type(snapshot) is not MetadataSnapshot:
            raise E3ContractError('check_batch needs MetadataSnapshot')
        if self._latest_batch is None or snapshot != self._latest_batch:
            raise E3ContractError('stale batch row')
        return snapshot

    def submit(self, ticket):
        self._require_active()
        if self._aborted:
            raise E3ContractError('transaction is aborted; no new submissions')
        if self.state not in ('ready', 'in_use'):
            raise E3ContractError('lane is not ready for submission')
        self._check_ticket(ticket)
        if len(self.outstanding) >= self.worker.max_outstanding:
            raise E3Pending('lane operation credit exhausted')
        if ticket.op_seq <= self._last_op_seq:
            raise E3ContractError('operation sequence is not monotonic')
        other_role = LaneRole.DECODE if self.role is LaneRole.PREFILL else LaneRole.PREFILL
        other = self.worker.lane(other_role)
        if other.outstanding and any(a.overlaps(b) for a in self.buffers.values()
                                     for b in other.buffers.values()):
            raise E3Pending('shared serialized storage still has another lane consumer')
        self._last_op_seq = ticket.op_seq
        self.outstanding[ticket] = None
        self.state = 'in_use'
        return ticket

    def complete(self, proof):
        if type(proof) is not CompletionProof:
            raise E3ContractError('completion requires CompletionProof')
        if not self.active or self.handle is None:
            raise E3ContractError('completion on a lane without active storage')
        if proof.ticket not in self.outstanding:
            raise E3ContractError('completion proof has no outstanding operation')
        if proof.ticket.lane != self.handle.lane_id:
            raise E3ContractError('completion proof belongs to a stale lane epoch')
        if proof.completion_seq <= self._last_completion_seq:
            raise E3ContractError('completion sequence is not monotonic')
        self._last_completion_seq = proof.completion_seq
        del self.outstanding[proof.ticket]
        self.completed.append(proof)
        return proof

    def fail(self, ticket, reason='failed completion'):
        if ticket not in self.outstanding:
            raise E3ContractError('failed completion names no outstanding operation')
        return self.quarantine(reason)

    def quarantine(self, reason='failed completion'):
        """Keep every binding and lease; only an explicit recovery may clear this."""
        if self.state == 'free':
            raise E3ContractError('no lane storage to quarantine')
        self._quarantine = reason or 'failed completion'
        self.state = 'quarantined'
        return self._quarantine

    def abort(self, reason='aborted'):
        if self.state == 'free':
            raise E3ContractError('no transaction to abort')
        if self._quarantine is not None:
            raise E3ContractError('lane is quarantined')
        self._aborted = True
        self.persistent.cancel_pending()
        if self.state != 'retiring':
            self.state = 'retiring'
        return reason

    def release(self, proofs=()):
        if self.state == 'quarantined':
            raise E3ContractError('quarantined lane cannot release storage')
        if self.state == 'free':
            raise E3ContractError('lane is already free')
        if self.handle is None:
            raise E3ContractError('lane has no handle')
        if not isinstance(proofs, (tuple, list)):
            raise E3ContractError('release proofs must be a sequence')
        for proof in proofs:
            if type(proof) is not CompletionProof:
                raise E3ContractError('release needs CompletionProof values')
            if proof.ticket in self.outstanding:
                self.complete(proof)
            elif proof.ticket.lane != self.handle.lane_id:
                raise E3ContractError('release proof belongs to a stale lane epoch')
            else:
                raise E3ContractError('release proof has no outstanding operation')
        if self.outstanding:
            raise E3Pending(f'{len(self.outstanding)} operation(s) lack completion proof')
        if self.graph_pools.get(self.storage_epoch):
            raise E3Pending('graph pools still reference this storage epoch')
        transaction = self.transaction
        self._drop_bindings()
        self.state = 'free'
        self.transaction = None
        self.metadata = None
        self.handle = None
        self.histories = None
        self.hc = None
        self.recurrent = None
        self.persistent.reset()
        self._aborted = False
        if self.worker.active_prefill == transaction:
            self.worker.active_prefill = None
        return transaction

    def bind_histories(self, histories):
        self._require_active()
        if self.role is not LaneRole.PREFILL:
            raise E3ContractError('only the prefill lane owns retained histories')
        if type(histories) is not QsaHistories:
            raise E3ContractError('bind_histories needs QsaHistories')
        if any(h.rank != self.worker.worker_rank for h in histories.histories):
            raise E3ContractError('history rank differs from the worker')
        self.histories = histories
        return histories

    def bind_recurrent(self, values):
        self._require_active()
        if self.transaction is None:
            raise E3ContractError('no active transaction')
        self.recurrent = RecurrentSnapshot(self.transaction, _freeze(values))
        return self.recurrent

    def bind_hc(self, hidden, block_output, injection, cursor, addresses):
        self._require_active()
        self.hc = HcTupleBinding(hidden, block_output, injection, cursor, tuple(addresses))
        self._hc_materialized = False
        return self.hc

    def materialize_hc(self, proof):
        self._require_active()
        if self.hc is None:
            raise E3ContractError('no delayed HC tuple is bound')
        if self.handle is None:
            raise E3ContractError('lane has no handle')
        if self._hc_materialized:
            raise E3ContractError('delayed HC tuple was already materialized')
        if type(proof) is not CompletionProof:
            raise E3ContractError('HC handoff needs a PP-consumer completion proof')
        if proof.kind is not CompletionKind.PP_CONSUMER:
            raise E3ContractError('HC handoff needs a PP-consumer completion proof')
        if proof.ticket.transaction != self.transaction:
            raise E3ContractError('HC completion belongs to another transaction')
        if proof.ticket.lane != self.handle.lane_id:
            raise E3ContractError('HC completion belongs to another lane')
        if proof not in self.completed or proof.ticket.dependency_role is not DependencyRole.HANDOFF:
            raise E3ContractError('HC handoff requires a recorded submitted handoff completion')
        self._hc_materialized = True
        return self.hc


class WorkerState:
    """Decode and prefill lanes on one worker, plus shared immutable weights."""

    def __init__(self, worker_rank, *, engine_epoch=1, weights=None,
                 weights_address=0, weights_bytes=0, max_outstanding=16,
                 engine_bound=True, incarnation=None):
        _int('worker_rank', worker_rank)
        _int('engine_epoch', engine_epoch)
        _int('weights_address', weights_address)
        _int('weights_bytes', weights_bytes)
        _int('max_outstanding', max_outstanding, 1)
        if type(engine_bound) is not bool:
            raise E3ContractError('engine_bound must be bool')
        if incarnation is None:
            incarnation = uuid.uuid4().hex
        _text('incarnation', incarnation)
        if not engine_bound and engine_epoch != 0:
            raise E3ContractError('an unbound worker cannot carry an engine epoch')
        if engine_bound and engine_epoch < 1:
            raise E3ContractError('a bound worker needs a positive engine epoch')
        self.max_outstanding = max_outstanding
        self.worker_rank = worker_rank
        self.engine_epoch = engine_epoch
        self.engine_bound = engine_bound
        self.incarnation = incarnation
        self.decode_owner = None
        self.weights_object = weights
        self.weights_address = weights_address
        self.weights_bytes = weights_bytes
        self.decode = LaneStorage(self, LaneRole.DECODE, 0)
        self.prefill = LaneStorage(self, LaneRole.PREFILL, 1)
        self.active_prefill = None
        self._generations = {}
        self._txn_seq = 0
        self._bindings = {}
        self._declared = []

    def retain_decode_storage(self, *, context_id, incarnation=None):
        """Retain the worker's decode storage for its whole incarnation.

        The returned :class:`DecodeStorageOwner` is the only storage a decode
        graph may be captured against.  No request transaction is involved.
        """
        if incarnation is None:
            incarnation = self.incarnation
        if incarnation != self.incarnation:
            raise E3ContractError('decode storage must belong to this incarnation')
        owner = self.decode.retain(context_id=context_id, incarnation=incarnation)
        if self.decode_owner is not None and self.decode_owner != owner:
            raise E3ContractError('decode storage owner changed for one incarnation')
        self.decode_owner = owner
        return owner

    def bind_engine_owner(self, engine_epoch, *, incarnation=None):
        """Bind this worker incarnation to one engine epoch, exactly once."""
        _int('engine_epoch', engine_epoch)
        if engine_epoch < 1:
            raise E3ContractError('engine epoch must be positive')
        if incarnation is not None and incarnation != self.incarnation:
            raise E3ContractError('engine owner names a foreign worker incarnation')
        if self.engine_bound:
            if engine_epoch == self.engine_epoch:
                return self.engine_epoch
            raise E3ContractError('worker is already bound to another engine epoch')
        self.engine_epoch = engine_epoch
        self.engine_bound = True
        return self.engine_epoch

    def lane(self, role):
        if role is LaneRole.DECODE:
            return self.decode
        if role is LaneRole.PREFILL:
            return self.prefill
        raise E3ContractError('unknown lane role')

    @property
    def weights_token(self):
        """The shared immutable weights object; identity is the contract."""
        return self.weights_object

    def declare_request(self, request_uid, generation):
        _text('request_uid', request_uid)
        _int('request_generation', generation)
        previous = self._generations.get(request_uid)
        if previous is not None:
            if generation < previous:
                raise E3ContractError('request generation went backwards')
            if generation > previous and self.active_prefill is not None:
                prefill_uid = self.active_prefill.request_uid
                if prefill_uid == request_uid:
                    raise E3Pending('cannot start a new generation while prefill is leased')
        self._generations[request_uid] = generation
        return generation

    def current_generation(self, request_uid):
        _text('request_uid', request_uid)
        if request_uid not in self._generations:
            raise E3ContractError('unknown request UID')
        return self._generations[request_uid]

    def check_transaction(self, transaction):
        if type(transaction) is not TransactionId:
            raise E3ContractError('transaction must be TransactionId')
        if not self.engine_bound:
            raise E3ContractError('worker engine owner is not bound')
        if transaction.engine_epoch != self.engine_epoch:
            raise E3ContractError('transaction engine epoch is stale')
        if self._generations.get(transaction.request_uid) != transaction.request_generation:
            raise E3ContractError('transaction request generation is stale')
        return transaction

    def new_transaction(self, request_uid, generation, chunk_start, chunk_end):
        self.declare_request(request_uid, generation)
        self._txn_seq += 1
        transaction = TransactionId(self.engine_epoch, request_uid, generation,
                                    chunk_start, chunk_end, self._txn_seq)
        self.check_transaction(transaction)
        return transaction

    def register_binding(self, binding, role):
        if type(binding) is not BufferBinding:
            raise E3ContractError('register_binding needs BufferBinding')
        key = (role, binding.name)
        if key in self._bindings:
            raise E3ContractError('buffer name already bound on this lane')
        for other in self._bindings.values():
            if not binding.overlaps(other):
                continue
            serialized = (binding.shared and other.shared and
                          binding.category == _SERIALIZED and other.category == _SERIALIZED and
                          binding.reuse_condition == other.reuse_condition and
                          binding.reuse_condition)
            if not serialized:
                raise E3ContractError('lane scratch aliases another buffer')
        self._bindings[key] = binding
        return binding

    def drop_bindings(self, role):
        for key in [k for k in self._bindings if k[0] is role]:
            del self._bindings[key]

    def declare_inventory(self, entry):
        if type(entry) is not InventoryEntry:
            raise E3ContractError('declare_inventory needs InventoryEntry')
        self._declared.append(entry)
        return entry

    def inventory(self):
        entries = [InventoryEntry('shared_weights', 'model', 'weights', self.weights_bytes,
                                  'existing' if self.weights_bytes else 'unmeasured', False,
                                  'immutable, shared by decode and prefill')]
        for lane in (self.decode, self.prefill):
            for binding in lane.buffers.values():
                entries.append(InventoryEntry(binding.name, binding.owner, binding.category,
                                              binding.nbytes, binding.basis, True,
                                              binding.reuse_condition))
            if lane.histories is not None:
                for history in lane.histories.histories:
                    entries.append(InventoryEntry(history.layer_name, 'model.qsa_host_kv',
                                                  'qsa_history', history.nbytes, 'existing',
                                                  False, 'lease held until C completion proof'))
            for pool in lane.graph_pools.get(lane.storage_epoch, {}).values():
                entries.append(InventoryEntry(pool.name, f'{lane.role.value}_lane',
                                              'graph_pool', pool.nbytes, pool.basis, True,
                                              'destroy after replay completion'))
        entries.extend(self._declared)
        return InventoryReport(tuple(entries))


def topk_bytes(chunk_tokens):
    """The private top-k scratch: 1024 rows' worth, or more above 1024 rows (decode-phase dp5); lane_private cuts its view from it."""
    return max(TOPK_BYTES, chunk_tokens * 2052 * 4)


def prefill_scratch_specs(chunk_tokens=1024):
    """Private prefill CPU/GPU metadata buffers, sized for one bounded chunk."""
    _int('chunk_tokens', chunk_tokens, 1)
    return (
        ScratchSpec('prefill_input_ids', 'prefill', 'metadata', chunk_tokens * 4,
                    'estimate', reuse_condition='private chunk only; recopied at admission'),
        ScratchSpec('prefill_positions', 'prefill', 'metadata', chunk_tokens * 8,
                    'estimate', reuse_condition='private chunk only; recopied at admission'),
        ScratchSpec('prefill_seq_lens', 'prefill', 'metadata', chunk_tokens * 4,
                    'estimate', reuse_condition='private chunk only; recopied at admission'),
        ScratchSpec('prefill_query_start_loc', 'prefill', 'metadata', (chunk_tokens + 1) * 4,
                    'estimate', reuse_condition='private chunk only; recopied at admission'),
        ScratchSpec('prefill_block_table', 'prefill', 'metadata', HISTORY_PAGES_AT_200K * 4,
                    'estimate', reuse_condition='private chunk only; recopied at admission'),
        ScratchSpec('prefill_logits_indices', 'prefill', 'metadata', chunk_tokens * 4,
                    'estimate', reuse_condition='private chunk only; recopied at admission'),
        ScratchSpec('prefill_hc_hidden', 'prefill', 'hc', HC_HIDDEN_BYTES, 'estimate',
                    reuse_condition='materialize once at PP handoff'),
        ScratchSpec('prefill_hc_block_output', 'prefill', 'hc', HC_BLOCK_OUTPUT_BYTES,
                    'estimate', reuse_condition='materialize once at PP handoff'),
        ScratchSpec('prefill_hc_injection', 'prefill', 'hc', HC_INJECTION_BYTES, 'estimate',
                    reuse_condition='materialize once at PP handoff'),
        ScratchSpec('prefill_qsa_topk', 'prefill', 'qsa_scratch', topk_bytes(chunk_tokens), 'estimate',
                    reuse_condition='one active prefill; released after consumer proof'),
        ScratchSpec('prefill_qsa_accum', 'prefill', 'qsa_scratch', QSA_ACCUM_BYTES, 'estimate',
                    reuse_condition='one active prefill; released after consumer proof'),
        ScratchSpec('prefill_pp_handoff', 'prefill', 'pp_handoff', PP_HIDDEN_BYTES, 'estimate',
                    reuse_condition='send storage released after receiver consumption'),
    )


def estimated_e3_inventory(*, chunk_tokens=1024, graph_split_bytes=GRAPH_POOL_MEASURED_BYTES,
                           graph_full_bytes=GRAPH_POOL_MEASURED_BYTES,
                           weights_bytes=0, weights_basis='unmeasured',
                           marlin_workspace_bytes=None, green_context_bytes=None,
                           communicator_bytes=None, ple_prefetch_bytes=0,
                           extra_history_window_bytes=0, arena_bytes=ARENA_BYTES):
    """Static allocation report by category, with explicit basis.

    The graph split-bank figure is an estimate (the memory ledger measures only
    the existing decode graph pool), and Marlin/green/communicator bytes are
    unmeasured. No entry grants headroom: the earlier 2.7 GiB fit was withdrawn.
    """
    _int('graph_split_bytes', graph_split_bytes)
    _int('graph_full_bytes', graph_full_bytes)
    _int('weights_bytes', weights_bytes)
    _int('ple_prefetch_bytes', ple_prefetch_bytes)
    _int('extra_history_window_bytes', extra_history_window_bytes)
    entries = [
        InventoryEntry('shared_weights', 'model', 'weights', weights_bytes,
                       weights_basis, False, 'immutable, shared by both lanes'),
        InventoryEntry('qsa_arena', 'prefill', 'arena', arena_bytes, 'existing', False,
                       'existing 24-page E2 arena; no new window'),
        InventoryEntry('qsa_histories_200k', 'model.qsa_host_kv', 'qsa_history',
                       HISTORY_BYTES_PER_RANK, 'existing', False,
                       'six independent histories; lease held until completion proof'),
        InventoryEntry('prefill_metadata', 'prefill', 'metadata',
                       sum(spec.nbytes for spec in prefill_scratch_specs(chunk_tokens)
                           if spec.category == 'metadata'), 'estimate', True,
                       'private chunk; recopied at admission'),
        InventoryEntry('prefill_hc', 'prefill', 'hc',
                       HC_HIDDEN_BYTES + HC_BLOCK_OUTPUT_BYTES + HC_INJECTION_BYTES,
                       'estimate', True, 'materialize once at PP handoff'),
        InventoryEntry('prefill_qsa_scratch', 'prefill', 'qsa_scratch',
                       topk_bytes(chunk_tokens) + QSA_ACCUM_BYTES, 'estimate', True,
                       'one active prefill; consumer proof before reuse'),
        InventoryEntry('prefill_pp_handoff', 'prefill', 'pp_handoff', PP_HIDDEN_BYTES,
                       'estimate', True, 'send storage released after consumption'),
        InventoryEntry('decode_full_graph_pool', 'decode', 'graph_pool', graph_full_bytes,
                       'measured', False, 'ordinary full-GPU decode graph'),
        InventoryEntry('decode_split_graph_bank', 'decode', 'graph_pool', graph_split_bytes,
                       'estimate', True, 'fixed split bank; measure and fund before arming'),
        InventoryEntry('extra_history_window', 'prefill', 'qsa_history',
                       extra_history_window_bytes, 'declared-zero', False,
                       'no additional history-sized window'),
        InventoryEntry('ple_prefetch', 'prefill', 'ple', ple_prefetch_bytes, 'declared-zero',
                       True, 'initial prefill does inline UVA lookup; decode keeps captured path'),
    ]
    for name, value, owner in (('marlin_workspace', marlin_workspace_bytes, 'prefill'),
                               ('green_context', green_context_bytes, 'prefill'),
                               ('communicator_buffers', communicator_bytes, 'prefill')):
        if value is None:
            entries.append(InventoryEntry(name, owner, 'workspace', 0, 'unmeasured', True,
                                          'measure at startup before arming'))
        else:
            _int(name, value)
            entries.append(InventoryEntry(name, owner, 'workspace', value, 'estimate', True,
                                          'lane-private; released after consumer proof'))
    return InventoryReport(tuple(entries))


__all__ = [
    'ARENA_BYTES', 'ARENA_PAGES', 'BUCKETS_AT_200K', 'BatchRow', 'BufferBinding',
    'CommitRecord', 'GraphPoolBinding', 'GRAPH_POOL_MEASURED_BYTES',
    'DecodeStorageOwner',
    'HC_BLOCK_OUTPUT_BYTES', 'HC_HIDDEN_BYTES',
    'HC_INJECTION_BYTES', 'HcTupleBinding', 'HISTORIES_PER_RANK', 'HISTORY_BYTES_PER_RANK',
    'HISTORY_PAGES_AT_200K', 'InventoryEntry', 'InventoryReport', 'LaneHandle', 'LaneStorage',
    'MetadataSnapshot', 'PP_HIDDEN_BYTES', 'PersistentState', 'PrivateUidBank', 'QSA_ACCUM_BYTES',
    'QsaHistories', 'QsaHistory', 'RecurrentSnapshot', 'ScratchSpec', 'TOPK_BYTES', 'WorkerState',
    'estimated_e3_inventory', 'prefill_scratch_specs',
]
