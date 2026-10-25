"""E3 fixed-split decode graph/context routing (offline, CPU-only contract).

This module is B's half of the E3 contract.  It imports A's frozen identities
from :mod:`e3_types`/:mod:`lane_state`; it does not redefine them and it never
imports torch or vLLM.  Graphs, streams and events are opaque tokens handled by
a backend C supplies.  Nothing here captures, replays, allocates or synchronizes
on a real device.

Scope: one declared fixed split (one green/graph bank for decode) plus the
ordinary full-GPU decode route that already exists in the pinned runner.
Adaptive split selection is deliberately absent.

Identity
--------
A graph is bound to the exact :class:`GraphKey` it was captured under:
execution descriptor, compilation identity, worker rank, B's context identity,
the *measured* SM allocation, A's lane storage epoch and the graph pool.  A
lookup that matches the shape but any of those identity fields returns no graph
-- there is no implicit fallback to a graph captured for another context,
split, address set or storage epoch.

Lifetimes
---------
*Capture* happens once at startup.  A failed or cancelled capture quarantines
the bank; no partially captured graph is routable.
*Replay* is an enqueue, not a completion.  A replay lease is only released by
C's recorded :class:`CompletionProof`; host submission and GPU completion are
kept distinct.  A graph pool has at most one active replay, and banks must not
share a pool even though the serial engine did.
*Transition* between the ordinary full-GPU route and the fixed split is an
explicit C-authorized operation and is refused while any replay, copy, PP or
worker operation is outstanding.
*Shutdown* is refused until C has drained consumers, then destroys graphs,
events and streams in a deterministic order before the pools are freed and the
pinned transport hook runs.

A CPU fake backend proves these contracts only.  It says nothing about CUDA
graph placement, SM containment, PP parity, memory or speed.
"""
from collections import deque
from dataclasses import dataclass, field

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
from lane_state import GRAPH_POOL_MEASURED_BYTES, LaneHandle, WorkerState

CG_MODES = ('none', 'piecewise', 'full')
_BASES = ('measured', 'estimate', 'existing', 'declared-zero', 'unmeasured')
_BACKEND_METHODS = ('capture', 'replay', 'destroy_graph', 'destroy_event', 'destroy_stream')


class GraphRoutingError(E3ContractError):
    """A graph/context route violates the fixed-split contract."""


class UnsupportedRoute(GraphRoutingError):
    """The requested shape/context/split has no routable graph bank."""


class GraphUnclean(GraphRoutingError):
    """Capture, replay or teardown failed; the router is not idle."""


def _int(name, value, minimum=0):
    if type(value) is not int:
        raise GraphRoutingError(f'{name} must be an int, not {type(value).__name__}')
    if value < minimum:
        raise GraphRoutingError(f'{name} must be >= {minimum}')
    return value


def _optional_int(name, value, minimum=0):
    if value is None:
        return None
    return _int(name, value, minimum)


def _text(name, value):
    if not isinstance(value, str) or not value:
        raise GraphRoutingError(f'{name} must be a non-empty string')
    if len(value) > 256 or any(ord(c) < 32 for c in value):
        raise GraphRoutingError(f'{name} contains unsupported characters')
    return value


def _bool(name, value):
    if type(value) is not bool:
        raise GraphRoutingError(f'{name} must be bool')
    return value


def _ints(name, values, minimum=0):
    if type(values) is not tuple:
        raise GraphRoutingError(f'{name} must be an immutable tuple')
    for value in values:
        _int(name, value, minimum)
    return values


@dataclass(frozen=True)
class ExecutionDescriptor:
    """Local, torch-free mirror of the pinned ``BatchExecutionDescriptor``.

    C converts the descriptor actually dispatched by the engine with
    :meth:`from_vllm`; B never guesses a captured shape from a batch row.
    """
    cg_mode: str
    num_tokens: int
    num_reqs: int | None
    uniform_token_count: int | None = None
    max_query_len: int | None = None
    num_active_loras: int = 0
    num_ubatches: int = 1

    def __post_init__(self):
        if self.cg_mode not in CG_MODES:
            raise GraphRoutingError(f'unknown cudagraph mode {self.cg_mode!r}')
        _int('num_tokens', self.num_tokens)
        _optional_int('num_reqs', self.num_reqs)
        _optional_int('uniform_token_count', self.uniform_token_count)
        _optional_int('max_query_len', self.max_query_len)
        _int('num_active_loras', self.num_active_loras)
        _int('num_ubatches', self.num_ubatches, 1)

    @classmethod
    def from_vllm(cls, descriptor):
        """Convert the pinned descriptor/enum without importing vLLM."""
        mode = getattr(descriptor, 'cg_mode', None)
        name = getattr(mode, 'name', mode)
        if not isinstance(name, str):
            raise GraphRoutingError('dispatched descriptor has no cudagraph mode')
        try:
            return cls(
                name.lower(),
                int(getattr(descriptor, 'num_tokens')),
                _optional_attr(descriptor, 'num_reqs'),
                _optional_attr(descriptor, 'uniform_token_count'),
                _optional_attr(descriptor, 'max_query_len'),
                int(getattr(descriptor, 'num_active_loras', 0) or 0),
                int(getattr(descriptor, 'num_ubatches', 1) or 1),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise GraphRoutingError('unconvertible dispatched descriptor') from exc


def _optional_attr(obj, name):
    value = getattr(obj, name, None)
    return None if value is None else int(value)


@dataclass(frozen=True)
class CompilationIdentity:
    """Identity of the compilation/capture configuration that produced a graph."""
    digest: str
    cudagraph_mode: str
    capture_sizes: tuple
    decode_query_len: int
    lora_capture_cases: tuple = (0,)
    varlen_decode: bool = False

    def __post_init__(self):
        _text('digest', self.digest)
        if self.cudagraph_mode not in CG_MODES:
            raise GraphRoutingError(f'unknown compilation cudagraph mode {self.cudagraph_mode!r}')
        _ints('capture_sizes', self.capture_sizes, 1)
        if len(set(self.capture_sizes)) != len(self.capture_sizes):
            raise GraphRoutingError('capture sizes must be unique')
        _int('decode_query_len', self.decode_query_len, 1)
        _ints('lora_capture_cases', self.lora_capture_cases)
        _bool('varlen_decode', self.varlen_decode)


@dataclass(frozen=True)
class ExecutionContext:
    """One device execution context/split and the graph pool captured in it.

    ``measured_sm`` is the SM count actually reported at startup; a requested
    green count is not evidence.  ``stream_token`` identifies the actual stream
    the engine and PP handler cache, so replay cannot silently run on another
    stream while the cached stream is still used elsewhere.
    """
    name: str
    context_id: int
    requested_sm: int
    measured_sm: int
    pool_id: int
    stream_token: int
    reserved_bytes: int = 0
    basis: str = 'estimate'

    def __post_init__(self):
        _text('name', self.name)
        _int('context_id', self.context_id)
        _int('requested_sm', self.requested_sm, 1)
        _int('measured_sm', self.measured_sm, 1)
        _int('pool_id', self.pool_id)
        _int('stream_token', self.stream_token)
        _int('reserved_bytes', self.reserved_bytes)
        if self.basis not in _BASES:
            raise GraphRoutingError(f'unknown execution-context basis {self.basis!r}')


@dataclass(frozen=True)
class GraphKey:
    """Exact identity of one captured graph.  No field is an implicit default."""
    descriptor: ExecutionDescriptor
    compilation: CompilationIdentity
    worker_rank: int
    context_id: int
    measured_sm: int
    storage_epoch: int
    pool_id: int

    def __post_init__(self):
        if type(self.descriptor) is not ExecutionDescriptor:
            raise GraphRoutingError('graph key needs an ExecutionDescriptor')
        if type(self.compilation) is not CompilationIdentity:
            raise GraphRoutingError('graph key needs a CompilationIdentity')
        _int('worker_rank', self.worker_rank)
        _int('context_id', self.context_id)
        _int('measured_sm', self.measured_sm, 1)
        _int('storage_epoch', self.storage_epoch, 1)
        _int('pool_id', self.pool_id)


class GraphPool:
    """One graph pool with an explicit single-replay exclusion rule."""

    def __init__(self, name, pool_id, context, owner_epoch):
        self.name = _text('name', name)
        self.pool_id = _int('pool_id', pool_id)
        self.context = context
        if type(context) is not ExecutionContext:
            raise GraphRoutingError('graph pool needs an ExecutionContext')
        self.context_id = context.context_id
        self.owner_epoch = _int('owner_epoch', owner_epoch, 1)
        self._active = None
        self._destroyed = False
        self._freed = False

    @property
    def active(self):
        return self._active is not None

    @property
    def state(self):
        if self._freed:
            return 'freed'
        if self._destroyed:
            return 'destroyed'
        return 'busy' if self.active else 'idle'

    def acquire(self, lease_id):
        _int('lease_id', lease_id, 1)
        if self._destroyed or self._freed:
            raise GraphRoutingError('graph pool is destroyed')
        if self._active is not None:
            raise E3Pending('graph pool already has an active replay')
        self._active = lease_id
        return lease_id

    def release(self, lease_id):
        if self._active != lease_id:
            raise GraphRoutingError('lease does not own this graph pool')
        self._active = None

    def destroy(self):
        if self._destroyed:
            raise GraphRoutingError('graph pool is already destroyed')
        if self.active:
            raise E3Pending('no pool destruction while a replay is outstanding')
        self._destroyed = True

    def free(self):
        if not self._destroyed:
            raise GraphRoutingError('pool must be destroyed before it is freed')
        if self._freed:
            raise GraphRoutingError('graph pool is already freed')
        self._freed = True

    def rebind(self, new_epoch):
        """Advance the pool epoch; only after teardown, never under a user."""
        _int('new_epoch', new_epoch, 1)
        if not self._freed:
            raise E3Pending('no pointer rebinding before destruction and free')
        if new_epoch <= self.owner_epoch:
            raise GraphRoutingError('rebind epoch must advance')
        self.owner_epoch = new_epoch
        return self.owner_epoch


@dataclass(frozen=True, repr=False)
class CapturedGraph:
    key: GraphKey
    handle: object = field(compare=False, repr=False)
    ordinal: int = 0

    def __post_init__(self):
        if type(self.key) is not GraphKey:
            raise GraphRoutingError('captured graph needs a GraphKey')
        _int('ordinal', self.ordinal)

    def __repr__(self):
        return (f'CapturedGraph(context_id={self.key.context_id}, '
                f'descriptor={self.key.descriptor.cg_mode}/{self.key.descriptor.num_tokens}, '
                f'ordinal={self.ordinal})')


class GraphBank:
    """Captured decode graphs for one context/split and one storage epoch."""

    def __init__(self, name, context, lane, compilation, pool, backend):
        self.name = _text('name', name)
        if type(context) is not ExecutionContext:
            raise GraphRoutingError('graph bank needs an ExecutionContext')
        if type(lane) is not LaneHandle:
            raise GraphRoutingError('graph bank needs A\'s LaneHandle')
        if lane.role is not LaneRole.DECODE:
            raise GraphRoutingError('graphs are captured on the decode lane')
        if type(compilation) is not CompilationIdentity:
            raise GraphRoutingError('graph bank needs a CompilationIdentity')
        if type(pool) is not GraphPool:
            raise GraphRoutingError('graph bank needs a GraphPool')
        if pool.context_id != context.context_id:
            raise GraphRoutingError('graph pool and bank contexts disagree')
        for method in _BACKEND_METHODS:
            if not callable(getattr(backend, method, None)):
                raise GraphRoutingError(f'capture backend is missing {method}()')
        self.context = context
        self.lane = lane
        self.compilation = compilation
        self.pool = pool
        self.backend = backend
        self.state = 'empty'
        self.graphs = {}
        self.descriptors = ()
        self.events = []
        self.streams = []
        self.allocated_bytes = None
        self.quarantine_reason = None

    @property
    def ready(self):
        return self.state == 'ready'

    @property
    def coverage(self):
        return self.descriptors

    def key_for(self, descriptor):
        if type(descriptor) is not ExecutionDescriptor:
            raise GraphRoutingError('graph lookup needs an ExecutionDescriptor')
        return GraphKey(descriptor, self.compilation, self.lane.worker_rank,
                        self.context.context_id, self.context.measured_sm,
                        self.lane.storage_epoch, self.pool.pool_id)

    def register_event(self, token):
        if self.state not in ('empty', 'ready'):
            raise GraphRoutingError('events are registered before/with capture')
        self.events.append(token)
        return token

    def register_stream(self, token):
        if self.state not in ('empty', 'ready'):
            raise GraphRoutingError('streams are registered before/with capture')
        self.streams.append(token)
        return token

    def capture(self, descriptors):
        """Capture the declared descriptors in order; all-or-quarantine."""
        if self.state != 'empty':
            raise GraphRoutingError(f'bank {self.name!r} is not empty')
        descriptors = tuple(descriptors)
        if not descriptors:
            raise GraphRoutingError('a graph bank needs at least one descriptor')
        if len(set(descriptors)) != len(descriptors):
            raise GraphRoutingError('duplicate descriptor in graph bank')
        self.state = 'capturing'
        try:
            for ordinal, descriptor in enumerate(descriptors):
                if type(descriptor) is not ExecutionDescriptor:
                    raise GraphRoutingError('bank descriptors must be ExecutionDescriptor')
                key = self.key_for(descriptor)
                handle = self.backend.capture(self.name, key, self.lane, self.context)
                self.graphs[descriptor] = CapturedGraph(key, handle, ordinal)
                self.descriptors = self.descriptors + (descriptor,)
        except BaseException as exc:
            self.state = 'quarantined'
            self.quarantine_reason = f'capture failed: {type(exc).__name__}'
            raise
        self.state = 'ready'
        return self.coverage

    def graph_for(self, descriptor, *, storage_epoch, worker_rank):
        """Return the graph for the exact current identity, or ``None``.

        A shape-only match whose stored key names another context, split,
        pool or storage epoch is refused rather than reused.
        """
        if self.state != 'ready':
            return None
        graph = self.graphs.get(descriptor)
        if graph is None:
            return None
        expected = GraphKey(descriptor, self.compilation, worker_rank,
                            self.context.context_id, self.context.measured_sm,
                            storage_epoch, self.pool.pool_id)
        if graph.key != expected:
            raise GraphRoutingError(
                'captured graph key does not match the current lane/context identity')
        return graph

    def quarantine(self, reason='graph bank failed'):
        if self.state == 'destroyed':
            raise GraphRoutingError('destroyed graph bank cannot be quarantined')
        self.state = 'quarantined'
        self.quarantine_reason = _text('reason', reason)
        return self.quarantine_reason

    def rebind(self, new_lane):
        """Rebind the bank handle only after its graphs and replay are gone."""
        if self.pool.active or self.graphs:
            raise E3Pending('no graph rebinding while graphs or a replay are live')
        if type(new_lane) is not LaneHandle or new_lane.role is not LaneRole.DECODE:
            raise GraphRoutingError('rebind needs a decode LaneHandle')
        self.lane = new_lane
        return new_lane

    def destroy(self):
        """Destroy graphs, events, streams and the pool in a safe order."""
        if self.state == 'destroyed':
            raise GraphRoutingError('graph bank is already destroyed')
        if self.state == 'capturing':
            raise E3Pending('cannot destroy a bank while capture is in progress')
        if self.pool.active:
            raise E3Pending('cannot destroy a bank while a replay is outstanding')
        order = []
        for graph in sorted(self.graphs.values(), key=lambda g: -g.ordinal):
            self.backend.destroy_graph(graph.handle)
            order.append('graph')
        for token in reversed(self.events):
            self.backend.destroy_event(token)
            order.append('event')
        for token in reversed(self.streams):
            self.backend.destroy_stream(token)
            order.append('stream')
        self.pool.destroy()
        order.append('pool')
        self.state = 'destroyed'
        return tuple(order)


@dataclass(frozen=True)
class RouteDecision:
    """Explicit result of a supported-route query handed to C before admission."""
    mode: str
    bank: str | None
    descriptor: ExecutionDescriptor
    reason: str

    def __post_init__(self):
        if self.mode not in ('split', 'ordinary'):
            raise GraphRoutingError(f'unknown route mode {self.mode!r}')
        if type(self.descriptor) is not ExecutionDescriptor:
            raise GraphRoutingError('route decision needs an ExecutionDescriptor')
        if self.mode == 'split':
            if not isinstance(self.bank, str) or not self.bank:
                raise GraphRoutingError('split route names a bank')
        elif self.bank is not None:
            raise GraphRoutingError('ordinary route names no bank')
        _text('reason', self.reason)


@dataclass(frozen=True)
class ReplayLease:
    """One host-submitted replay awaiting C's GPU completion proof."""
    lease_id: int
    bank_name: str
    key: GraphKey
    ticket: OperationTicket
    context_id: int
    stream_token: int
    host_submit_seq: int
    storage_epoch: int

    def __post_init__(self):
        _int('lease_id', self.lease_id, 1)
        _text('bank_name', self.bank_name)
        if type(self.key) is not GraphKey:
            raise GraphRoutingError('lease needs a GraphKey')
        if type(self.ticket) is not OperationTicket:
            raise GraphRoutingError('lease needs an OperationTicket')
        _int('context_id', self.context_id)
        _int('stream_token', self.stream_token)
        _int('host_submit_seq', self.host_submit_seq, 1)
        _int('storage_epoch', self.storage_epoch, 1)


@dataclass(frozen=True, repr=False)
class ReplayResult:
    """A replay enqueue plus its opaque engine output."""
    lease: ReplayLease
    output: object = field(compare=False, repr=False)

    def __post_init__(self):
        if type(self.lease) is not ReplayLease:
            raise GraphRoutingError('replay result needs a ReplayLease')

    def __repr__(self):
        return f'ReplayResult(lease={self.lease!r})'


@dataclass(frozen=True)
class TimingRecord:
    """Lane-local timing, created only after a recorded completion proof."""
    lease_id: int
    context_id: int
    storage_epoch: int
    worker_rank: int
    host_submit_seq: int
    completion_seq: int
    gpu_elapsed_ns: int | None
    basis: str

    def __post_init__(self):
        _int('lease_id', self.lease_id, 1)
        _int('context_id', self.context_id)
        _int('storage_epoch', self.storage_epoch, 1)
        _int('worker_rank', self.worker_rank)
        _int('host_submit_seq', self.host_submit_seq, 1)
        _int('completion_seq', self.completion_seq)
        if self.gpu_elapsed_ns is not None:
            _int('gpu_elapsed_ns', self.gpu_elapsed_ns)
        if self.basis not in ('measured', 'unmeasured'):
            raise GraphRoutingError('timing basis is measured or unmeasured')


@dataclass(frozen=True)
class TransitionAuthorization:
    """C's explicit authorization for one completed full-GPU <-> split switch."""
    engine_epoch: int
    target_context_id: int
    target_storage_epoch: int
    completion_seq: int
    reason: str

    def __post_init__(self):
        _int('engine_epoch', self.engine_epoch)
        _int('target_context_id', self.target_context_id)
        _int('target_storage_epoch', self.target_storage_epoch, 1)
        _int('completion_seq', self.completion_seq)
        _text('reason', self.reason)


@dataclass(frozen=True)
class TransitionRecord:
    transition_seq: int
    source_mode: str
    target_mode: str
    completion_seq: int


@dataclass(frozen=True)
class ShutdownRecord:
    order: tuple
    banks: tuple


@dataclass(frozen=True)
class BankBudget:
    """Allocated/reserved graph-pool bytes for one bank."""
    name: str
    kind: str
    context_id: int
    pool_id: int
    allocated_bytes: int | None
    reserved_bytes: int
    basis: str
    new_allocation: bool

    def __post_init__(self):
        _text('name', self.name)
        if self.kind not in ('full', 'split'):
            raise GraphRoutingError('bank kind is full or split')
        _int('context_id', self.context_id)
        _int('pool_id', self.pool_id)
        _optional_int('allocated_bytes', self.allocated_bytes)
        _int('reserved_bytes', self.reserved_bytes)
        if self.basis not in _BASES:
            raise GraphRoutingError('unknown bank budget basis')
        _bool('new_allocation', self.new_allocation)
        if self.basis == 'unmeasured' and self.allocated_bytes not in (None, 0):
            raise GraphRoutingError('unmeasured banks record no allocation')


@dataclass(frozen=True)
class GraphBudgetReport:
    """Graph-pool budget by bank; a reserve shortfall funds no new storage."""
    banks: tuple
    reserve_required_bytes: int
    reserve_measured_bytes: int | None

    def __post_init__(self):
        if type(self.banks) is not tuple:
            raise GraphRoutingError('banks must be an immutable tuple')
        for bank in self.banks:
            if type(bank) is not BankBudget:
                raise GraphRoutingError('budget entries must be BankBudget')
        _int('reserve_required_bytes', self.reserve_required_bytes)
        _optional_int('reserve_measured_bytes', self.reserve_measured_bytes)
        seen = set()
        for bank in self.banks:
            if bank.pool_id in seen:
                raise GraphRoutingError('graph pools must not be shared between banks')
            seen.add(bank.pool_id)

    @property
    def new_banks(self):
        return tuple(bank for bank in self.banks if bank.new_allocation)

    @property
    def new_bytes(self):
        return sum(bank.reserved_bytes for bank in self.new_banks)

    @property
    def existing_bytes(self):
        return sum(bank.reserved_bytes for bank in self.banks if not bank.new_allocation)

    @property
    def funding(self):
        if self.reserve_measured_bytes is None:
            return 'unmeasured'
        return 'funded' if self.reserve_measured_bytes >= self.reserve_required_bytes else 'shortfall'

    @property
    def new_allowed(self):
        return self.funding == 'funded'

    def require_funded(self):
        funding = self.funding
        if funding != 'funded':
            raise GraphRoutingError(
                f'graph-bank reserve is {funding}; no new storage is granted')
        return self

    def require_bounded(self, limit):
        _int('limit', limit)
        for bank in self.new_banks:
            if bank.allocated_bytes is None or bank.basis != 'measured':
                raise GraphRoutingError('new graph bank allocation is unmeasured')
            if bank.allocated_bytes > bank.reserved_bytes:
                raise GraphRoutingError('graph allocation exceeds its declared reservation')
        if self.new_banks:
            # A reserve shortfall grants no new storage budget, even when the
            # requested byte total happens to fit an arbitrary limit.
            self.require_funded()
        if self.new_bytes > limit:
            raise GraphRoutingError(
                f'graph-bank allocation {self.new_bytes} exceeds budget {limit}')
        return self


class ExternalUsers:
    """Copy/PP users C registers so transitions and teardown cannot race them."""
    KINDS = ('copy', 'pp')

    def __init__(self):
        self._users = []

    def register(self, kind, token):
        if kind not in self.KINDS:
            raise GraphRoutingError(f'unknown external user kind {kind!r}')
        entry = (kind, token)
        if entry in self._users:
            raise GraphRoutingError('external user is already registered')
        self._users.append(entry)
        return entry

    def release(self, kind, token):
        entry = (kind, token)
        if entry not in self._users:
            raise GraphRoutingError('external user is not registered')
        self._users.remove(entry)

    def __len__(self):
        return len(self._users)

    def snapshot(self):
        return tuple(self._users)


@dataclass(frozen=True)
class RouterStatus:
    state: str
    mode: str
    pending_replays: int
    external_users: int
    bank_states: tuple
    shutdown_pending: bool
    unclean: bool

    @property
    def idle(self):
        if self.unclean or self.shutdown_pending:
            return False
        if self.state == 'destroyed':
            return True
        if self.state != 'ready':
            return False
        return (self.pending_replays == 0 and self.external_users == 0
                and all(state in ('empty', 'ready', 'destroyed')
                        for _, state in self.bank_states))


class GraphRouter:
    """Fixed-split graph/context router for one worker's decode lane.

    C constructs one router per worker after it holds a stable decode lane
    handle (A's epoch must not advance while the banks exist), registers the
    declared split bank(s), and calls :meth:`capture_at_startup` after the
    ordinary engine capture.  C then drives :meth:`supported_route`,
    :meth:`replay`, completion polling via A and :meth:`transition`, and
    finally :meth:`shutdown` after every consumer has drained.
    """

    def __init__(self, worker, decode_handle, compilation, *, engine_epoch=None,
                 backend=None, full_context_id, full_pool_id,
                 full_allocated_bytes=GRAPH_POOL_MEASURED_BYTES,
                 full_basis='measured', reserve_required_bytes=0,
                 transport_release=None):
        if type(worker) is not WorkerState:
            raise GraphRoutingError('router needs A\'s WorkerState')
        if type(decode_handle) is not LaneHandle:
            raise GraphRoutingError('router needs A\'s decode LaneHandle')
        if decode_handle.role is not LaneRole.DECODE:
            raise GraphRoutingError('router handle must be the decode lane')
        if worker.decode.handle is not decode_handle:
            raise GraphRoutingError('router handle is not the active decode lane handle')
        if not worker.decode.active:
            raise GraphRoutingError('decode lane must be reserved before graph routing')
        if decode_handle.storage_epoch < 1:
            raise GraphRoutingError('decode lane needs a storage epoch')
        if type(compilation) is not CompilationIdentity:
            raise GraphRoutingError('router needs a CompilationIdentity')
        self.worker = worker
        self.handle = decode_handle
        self.compilation = compilation
        self.engine_epoch = (worker.engine_epoch if engine_epoch is None
                             else _int('engine_epoch', engine_epoch))
        if backend is not None:
            for method in _BACKEND_METHODS:
                if not callable(getattr(backend, method, None)):
                    raise GraphRoutingError(f'capture backend is missing {method}()')
        self.backend = backend
        self.full_context_id = _int('full_context_id', full_context_id)
        self.full_pool_id = _int('full_pool_id', full_pool_id)
        self.full_allocated_bytes = _int('full_allocated_bytes', full_allocated_bytes, 1)
        if full_basis not in _BASES:
            raise GraphRoutingError('unknown full-bank basis')
        self.full_basis = full_basis
        self.reserve_required_bytes = _int('reserve_required_bytes', reserve_required_bytes)
        self.transport_release = transport_release
        self.banks = {}
        self.pools = {}
        self.external = ExternalUsers()
        self.current_mode = 'ordinary'
        self.calibrated = False
        self.state = 'ready'
        self.probe = None
        self._pending = {}
        self._timing = deque(maxlen=1024)
        self._transitions = deque(maxlen=1024)
        self._transition_seq = 0
        self._lease_seq = 0
        self._host_seq = 0
        self._last_transition_seq = -1
        self._quarantine = None
        self._shutdown_pending = False
        self._shutdown_record = None
        self._pending_descriptors = {}

    # --- inventory / capture --------------------------------------------
    @property
    def unclean(self):
        return self._quarantine is not None

    @property
    def mode(self):
        """'ordinary' (full GPU) or a registered split bank name."""
        return self.current_mode

    def add_bank(self, name, context, descriptors=()):
        """Declare and register one split bank before startup capture."""
        self._require_ready()
        _text('name', name)
        if name == 'ordinary':
            raise GraphRoutingError("'ordinary' is reserved for the full-GPU route")
        if name in self.banks:
            raise GraphRoutingError('graph bank already registered')
        if type(context) is not ExecutionContext:
            raise GraphRoutingError('bank needs an ExecutionContext')
        if self.calibrated:
            raise GraphRoutingError('banks must be declared before startup capture')
        if context.context_id != self.handle.context_id:
            raise GraphRoutingError('bank context differs from A lane identity')
        if context.pool_id == self.full_pool_id or context.pool_id in self.pools:
            raise GraphRoutingError('graph pools must not be shared between banks')
        if context.context_id == self.full_context_id:
            raise GraphRoutingError('split context id collides with the full-GPU context')
        if any(bank.context.context_id == context.context_id for bank in self.banks.values()):
            raise GraphRoutingError('split context id is already registered')
        if context.reserved_bytes < 1:
            raise GraphRoutingError('split bank needs a reserved byte budget')
        if self.backend is None:
            raise GraphRoutingError('split capture needs C\'s capture backend')
        descriptors = tuple(descriptors)
        if not descriptors:
            raise GraphRoutingError('a split bank needs at least one descriptor')
        for descriptor in descriptors:
            if type(descriptor) is not ExecutionDescriptor:
                raise GraphRoutingError('bank descriptors must be ExecutionDescriptor')
        if len(set(descriptors)) != len(descriptors):
            raise GraphRoutingError('duplicate descriptor in bank declaration')
        pool = GraphPool(name, context.pool_id, context, self.handle.storage_epoch)
        # A owns the lane inventory and refuses release while this is present.
        self.worker.decode.register_graph_pool(
            name, context.pool_id, context.reserved_bytes, context.basis)
        bank = GraphBank(name, context, self.handle, self.compilation, pool, self.backend)
        self.banks[name] = bank
        self.pools[context.pool_id] = pool
        self._pending_descriptors[name] = descriptors
        return bank

    def capture_at_startup(self, descriptors=None):
        """Capture every declared bank after the ordinary engine capture."""
        self._require_ready()
        if self.calibrated:
            raise GraphRoutingError('graph banks were already captured')
        if descriptors is not None:
            if set(descriptors) != set(self.banks):
                raise GraphRoutingError('capture descriptors must name the declared banks')
        captured = []
        for name, bank in self.banks.items():
            declared = self._pending_descriptors[name]
            try:
                bank.capture(declared)
            except BaseException:
                self.state = 'quarantined'
                self._quarantine = f'capture failed in bank {name!r}'
                raise
            captured.append(name)
        self.calibrated = True
        return tuple(captured)

    # --- routing --------------------------------------------------------
    def supported_route(self, transaction, descriptor):
        """Bounded, nonblocking route query for C before admission.

        Returns a :class:`RouteDecision`.  With an active prefill the only
        supported route is an exact split-bank graph; anything else raises so
        the caller cannot silently replay full-GPU decode during overlap.
        """
        self._require_ready()
        if type(transaction) is not TransactionId:
            raise GraphRoutingError('route query needs a TransactionId')
        if type(descriptor) is not ExecutionDescriptor:
            raise GraphRoutingError('route query needs an ExecutionDescriptor')
        self._require_decode_lane()
        if transaction == self.worker.active_prefill:
            raise UnsupportedRoute('prefill cannot be routed to the decode graph bank')
        prefill_active = self.worker.active_prefill is not None
        if not prefill_active:
            if self.current_mode == 'ordinary':
                return RouteDecision('ordinary', None, descriptor,
                                     'no active prefill; ordinary full-GPU decode')
            bank = self.banks[self.current_mode]
            if bank.ready and bank.graph_for(
                    descriptor, storage_epoch=self.handle.storage_epoch,
                    worker_rank=self.handle.worker_rank) is not None:
                return RouteDecision('split', bank.name, descriptor,
                                     'declared split; no active prefill')
            raise UnsupportedRoute('unsupported split descriptor; complete an explicit ordinary transition first')
        if self.current_mode == 'ordinary':
            raise UnsupportedRoute(
                'full-GPU decode route is not allowed while prefill is active')
        bank = self.banks[self.current_mode]
        if not bank.ready:
            raise UnsupportedRoute(f'bank {bank.name!r} is {bank.state}')
        if bank.graph_for(descriptor, storage_epoch=self.handle.storage_epoch,
                          worker_rank=self.handle.worker_rank) is None:
            raise UnsupportedRoute(
                f'descriptor is not captured in split bank {bank.name!r}')
        return RouteDecision('split', bank.name, descriptor,
                             'overlap fixed split; exact captured descriptor')

    def replay(self, transaction, descriptor, ticket, *,
               observed_context, observed_stream):
        """Enqueue one split-bank replay; never a completion."""
        decision = self.supported_route(transaction, descriptor)
        if decision.mode != 'split':
            raise GraphRoutingError('replay requested on the ordinary route')
        bank = self.banks[decision.bank]
        if type(ticket) is not OperationTicket:
            raise GraphRoutingError('split replay needs a registered OperationTicket')
        if ticket.transaction != transaction:
            raise GraphRoutingError('replay ticket belongs to another transaction')
        if ticket.lane != self.handle.lane_id:
            raise GraphRoutingError('replay ticket names a stale lane')
        if ticket.dependency_role is not DependencyRole.CONSUMER:
            raise GraphRoutingError('decode graph replay is a consumer operation')
        if ticket not in self.worker.decode.outstanding:
            raise GraphRoutingError('replay ticket was not submitted through A')
        if (type(observed_context) is not int or type(observed_stream) is not int):
            raise GraphRoutingError('replay needs C\'s observed context and stream')
        if (observed_context != bank.context.context_id
                or observed_stream != bank.context.stream_token):
            raise GraphRoutingError('replay would run on the wrong context or stream')
        self._require_decode_lane()
        graph = bank.graph_for(descriptor, storage_epoch=self.handle.storage_epoch,
                               worker_rank=self.handle.worker_rank)
        if graph is None:
            raise UnsupportedRoute('no captured graph for the dispatched descriptor')
        if self.backend is None:
            raise GraphRoutingError('split replay needs C\'s capture backend')
        self._lease_seq += 1
        lease_id = self._lease_seq
        bank.pool.acquire(lease_id)
        self._host_seq += 1
        lease = ReplayLease(lease_id, bank.name, graph.key, ticket,
                            bank.context.context_id, bank.context.stream_token,
                            self._host_seq, self.handle.storage_epoch)
        # Keep ownership even if enqueue raises after partially submitting work.
        self._pending[lease_id] = (lease, bank)
        try:
            output = self.backend.replay(graph.handle, graph.key, bank.lane,
                                         bank.context)
        except BaseException:
            self._quarantine_bank(bank, 'replay enqueue failed; completion unknown')
            raise
        self._pending[lease_id] = (lease, bank)
        return ReplayResult(lease, output)

    def complete_replay(self, lease_or_result, proof, *, gpu_elapsed_ns=None):
        """Release a replay only after C's recorded GPU completion proof."""
        lease, bank = self._pending_lease(lease_or_result)
        if type(proof) is not CompletionProof:
            self._quarantine_bank(bank, 'completion is not a CompletionProof')
            raise GraphRoutingError('replay completion needs a CompletionProof')
        if proof.kind is not CompletionKind.GPU_EVENT:
            self._quarantine_bank(bank, 'replay completion is not a GPU event')
            raise GraphRoutingError('decode graph replay needs a GPU-event completion')
        if proof.ticket != lease.ticket:
            self._quarantine_bank(bank, 'completion ticket mismatch')
            raise GraphRoutingError('completion proof names another operation')
        if proof not in self.worker.decode.completed:
            self._quarantine_bank(bank, 'completion was not recorded by A')
            raise GraphRoutingError('replay completion is not recorded on the lane')
        if gpu_elapsed_ns is not None:
            _int('gpu_elapsed_ns', gpu_elapsed_ns)
        bank.pool.release(lease.lease_id)
        del self._pending[lease.lease_id]
        record = TimingRecord(lease.lease_id, lease.context_id, lease.storage_epoch,
                              self.handle.worker_rank, lease.host_submit_seq,
                              proof.completion_seq, gpu_elapsed_ns,
                              'measured' if gpu_elapsed_ns is not None else 'unmeasured')
        self._timing.append(record)
        return record

    def cancel_replay(self, lease_or_result, proof=None):
        """Cancel without a proof quarantines; with a proof it is a completion."""
        lease, bank = self._pending_lease(lease_or_result)
        if proof is None:
            self._quarantine_bank(bank, 'replay cancelled without completion')
            return False
        self.complete_replay(lease, proof)
        return True

    @property
    def timing_records(self):
        """Only completed replays; a pending lease has no timing record."""
        return tuple(self._timing)

    def timing_for(self, context_id):
        _int('context_id', context_id)
        return tuple(record for record in self._timing
                     if record.context_id == context_id)

    @property
    def pending_replays(self):
        return len(self._pending)

    def descriptor_coverage(self):
        return {name: bank.coverage for name, bank in self.banks.items()}

    # --- transitions ----------------------------------------------------
    def transition(self, target, authorization):
        """Switch route only with C's explicit, completed authorization."""
        self._require_ready()
        if type(authorization) is not TransitionAuthorization:
            raise GraphRoutingError('a transition needs C\'s TransitionAuthorization')
        if authorization.engine_epoch != self.engine_epoch:
            raise GraphRoutingError('transition authorization is for another engine epoch')
        if authorization.completion_seq <= self._last_transition_seq:
            raise GraphRoutingError('transition authorization is stale')
        self._require_decode_lane()
        if target == 'ordinary':
            if self.worker.active_prefill is not None:
                raise E3Pending('retire parked prefill before returning to full-GPU decode')
            if authorization.target_context_id != self.full_context_id:
                raise GraphRoutingError('ordinary transition names another context')
        elif target in self.banks:
            bank = self.banks[target]
            if not bank.ready or not bank.coverage:
                raise GraphRoutingError('target split bank is not captured')
            if authorization.target_context_id != bank.context.context_id:
                raise GraphRoutingError('transition names another split context')
        else:
            raise GraphRoutingError(f'unknown transition target {target!r}')
        if authorization.target_storage_epoch != self.handle.storage_epoch:
            raise GraphRoutingError('transition names a stale storage epoch')
        self._require_drained('transition')
        source = self.current_mode
        self.current_mode = target
        self._last_transition_seq = authorization.completion_seq
        self._transition_seq += 1
        record = TransitionRecord(self._transition_seq, source, target,
                                  authorization.completion_seq)
        self._transitions.append(record)
        return record

    @property
    def transitions(self):
        return tuple(self._transitions)

    # --- external users -------------------------------------------------
    def register_external_user(self, kind, token):
        self._require_ready()
        return self.external.register(kind, token)

    def release_external_user(self, kind, token):
        return self.external.release(kind, token)

    # --- teardown -------------------------------------------------------
    def shutdown(self, proofs=()):
        """Destroy banks after C drains consumers; never claim a false idle."""
        if self._shutdown_record is not None:
            return self._shutdown_record
        if self.state == 'destroyed':
            raise GraphRoutingError('router is already destroyed')
        if not isinstance(proofs, (tuple, list)):
            raise GraphRoutingError('shutdown proofs must be a sequence')
        # Close the admission window before the drain check: a concurrent
        # route/replay now fails instead of racing graph destruction.
        self._shutdown_pending = True
        for proof in proofs:
            if type(proof) is not CompletionProof:
                raise GraphRoutingError('shutdown proofs are CompletionProof values')
            match = [lease for lease, _ in self._pending.values()
                     if lease.ticket == proof.ticket]
            if not match:
                raise GraphRoutingError('shutdown proof has no pending replay')
            self.complete_replay(match[0], proof)
        try:
            self._require_drained('shutdown')
        except E3Pending:
            self._shutdown_pending = True
            raise
        order = []
        try:
            for name in reversed(tuple(self.banks)):
                bank = self.banks[name]
                order.extend(bank.destroy())
            # A's inventory retires a whole storage epoch at once, so retire
            # after every bank has dropped its graph, events and stream.
            if self.banks:
                retired = self.worker.decode.retire_graph_pools(self.handle.storage_epoch)
                retired_names = {pool.name for pool in retired}
                missing = set(self.banks) - retired_names
                if missing:
                    raise GraphRoutingError('A lane did not hold every retired graph pool')
                for name in reversed(tuple(self.banks)):
                    self.banks[name].pool.free()
                    order.append('free')
            if self.transport_release is not None:
                self.transport_release()
                order.append('transport')
        except BaseException as exc:
            self.state = 'quarantined'
            self._quarantine = f'shutdown failed: {type(exc).__name__}'
            raise
        self._shutdown_pending = False
        self.state = 'destroyed'
        self._shutdown_record = ShutdownRecord(tuple(order), tuple(self.banks))
        return self._shutdown_record

    # --- budget ---------------------------------------------------------
    def budget(self, *, reserve_measured_bytes=None, reserve_required_bytes=None):
        """Allocated/reserved graph bytes by bank, including the full bank.

        The split bank is a new allocation with an estimate basis; the caller
        must supply its measured allocation and a funded reserve before arming.
        """
        required = (self.reserve_required_bytes if reserve_required_bytes is None
                    else _int('reserve_required_bytes', reserve_required_bytes))
        entries = [BankBudget('decode_full_graph_pool', 'full', self.full_context_id,
                              self.full_pool_id, self.full_allocated_bytes,
                              self.full_allocated_bytes, self.full_basis, False)]
        for bank in self.banks.values():
            entries.append(BankBudget(
                bank.name, 'split', bank.context.context_id, bank.pool.pool_id,
                bank.allocated_bytes, bank.context.reserved_bytes,
                bank.context.basis, True))
        return GraphBudgetReport(tuple(entries), required, reserve_measured_bytes)

    def status(self):
        bank_states = tuple((name, bank.state) for name, bank in self.banks.items())
        if self.state == 'destroyed' and self.unclean:
            state = 'destroyed-unclean'
        elif self._shutdown_pending:
            state = 'shutdown-pending'
        elif self.unclean:
            state = 'quarantined'
        elif self.worker.decode.outstanding or self.worker.prefill.outstanding:
            state = 'busy'
        else:
            state = self.state
        return RouterStatus(state, self.current_mode, len(self._pending),
                            len(self.external), bank_states,
                            self._shutdown_pending, self.unclean)

    # --- helpers --------------------------------------------------------
    def _pending_lease(self, lease_or_result):
        lease = (lease_or_result.lease
                 if type(lease_or_result) is ReplayResult else lease_or_result)
        if type(lease) is not ReplayLease:
            raise GraphRoutingError('expected a ReplayLease or ReplayResult')
        entry = self._pending.get(lease.lease_id)
        if entry is None:
            raise GraphRoutingError('replay lease is not outstanding')
        if entry[0] != lease:
            raise GraphRoutingError('replay lease does not match the outstanding operation')
        return entry

    def _quarantine_bank(self, bank, reason):
        if bank.state != 'quarantined':
            bank.quarantine(reason)
        self._quarantine = reason
        self.state = 'quarantined'

    def _require_ready(self):
        if self.state == 'destroyed':
            raise GraphRoutingError('router is destroyed')
        if self._quarantine is not None or self.state == 'quarantined':
            raise GraphUnclean(f'graph router is unclean: {self._quarantine}')
        if self._shutdown_pending:
            raise E3Pending('graph router shutdown already requested; drain consumers')

    def _require_decode_lane(self):
        if self.worker.decode.handle is not self.handle:
            raise GraphUnclean('decode lane storage epoch/handle changed; graph storage is stale')
        if not self.worker.decode.active or self.worker.decode.state not in ('ready', 'in_use'):
            raise GraphRoutingError('decode lane is not ready for graph replay')
        if self.worker.decode.failed:
            raise GraphUnclean('decode lane is quarantined')

    def _require_drained(self, action):
        if self._pending:
            raise E3Pending(f'{len(self._pending)} replay(s) outstanding before {action}')
        for bank in self.banks.values():
            if bank.pool.active:
                raise E3Pending(f'graph pool {bank.name!r} still has a user before {action}')
            if bank.state == 'capturing':
                raise E3Pending(f'capture in progress before {action}')
        if len(self.external):
            raise E3Pending(f'external copy/PP users outstanding before {action}')
        if self.worker.decode.outstanding:
            raise E3Pending(f'decode operations outstanding before {action}')
        if self.worker.prefill.outstanding:
            raise E3Pending(f'prefill operations outstanding before {action}')
        return True


__all__ = [
    'BankBudget', 'CG_MODES', 'CapturedGraph', 'CompilationIdentity', 'ExecutionDescriptor',
    'ExecutionContext', 'ExternalUsers', 'GraphBank', 'GraphBudgetReport', 'GraphKey',
    'GraphPool', 'GraphRouter', 'GraphRoutingError', 'GraphUnclean', 'ReplayLease',
    'ReplayResult', 'RouteDecision', 'RouterStatus', 'ShutdownRecord', 'TimingRecord',
    'TransitionAuthorization', 'TransitionRecord', 'UnsupportedRoute',
]
