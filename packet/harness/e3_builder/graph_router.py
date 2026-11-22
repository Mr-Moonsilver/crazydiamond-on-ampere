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


def descriptor_name(descriptor):
    """Stable, request-free label for counters and diagnostics."""
    if type(descriptor) is not ExecutionDescriptor:
        raise GraphRoutingError('descriptor label needs an ExecutionDescriptor')
    return (f'{descriptor.cg_mode}:tokens={descriptor.num_tokens}:'
            f'reqs={descriptor.num_reqs}:uniform={descriptor.uniform_token_count}')


@dataclass(frozen=True)
class NoPrefillRoute:
    """Typed route identity for the decode-only diagnostic mode.

    It replaces the prefill transaction that :class:`GraphRouter` requires for
    split routing.  It is *not* a fabricated transaction: it names no request,
    chunk or engine transaction, only the split bank, its context/pool, the
    stable decode storage epoch and the route generation it was issued under.
    """
    bank: str
    context_id: int
    pool_id: int
    storage_epoch: int
    route_generation: int

    def __post_init__(self):
        _text('bank', self.bank)
        _int('context_id', self.context_id)
        _int('pool_id', self.pool_id)
        _int('storage_epoch', self.storage_epoch, 1)
        _int('route_generation', self.route_generation)


@dataclass(frozen=True)
class ProbeGraphKey:
    """Backend key for one diagnostic split graph; no lane/compilation ticket."""
    descriptor: ExecutionDescriptor
    storage_epoch: int
    route_generation: int

    def __post_init__(self):
        if type(self.descriptor) is not ExecutionDescriptor:
            raise GraphRoutingError('probe graph key needs an ExecutionDescriptor')
        _int('storage_epoch', self.storage_epoch, 1)
        _int('route_generation', self.route_generation)


class DecodeProbeRouter:
    """Split-only router for the decode-only diagnostic mode.

    It captures the declared FULL descriptors through the supplied backend into
    the split bank (which owns a separate graph pool) and replays only exact
    captured descriptors.  There is no prefill transaction, no operation
    ticket, no ordinary-route fallback and no completion proof: the diagnostic
    interval is bounded by the drained route transitions, and every host replay
    is counted per bank, stream and descriptor so the GPU test can read it.
    """
    def __init__(self, name, context, descriptors, backend, *, storage_epoch,
                 route_generation=0, context_factory=None):
        self.name = _text('name', name)
        if type(context) is not ExecutionContext:
            raise GraphRoutingError('decode probe needs an ExecutionContext')
        descriptors = tuple(descriptors)
        if not descriptors:
            raise GraphRoutingError('a decode probe bank needs at least one descriptor')
        for descriptor in descriptors:
            if type(descriptor) is not ExecutionDescriptor:
                raise GraphRoutingError('probe descriptors must be ExecutionDescriptor')
        if len(set(descriptors)) != len(descriptors):
            raise GraphRoutingError('duplicate descriptor in decode probe bank')
        for method in _BACKEND_METHODS:
            if not callable(getattr(backend, method, None)):
                raise GraphRoutingError(f'capture backend is missing {method}()')
        if context_factory is not None and not callable(context_factory):
            raise GraphRoutingError('context factory must be callable')
        self.context = context
        self.descriptors = descriptors
        self.backend = backend
        self.storage_epoch = _int('storage_epoch', storage_epoch, 1)
        self.route_generation = _int('route_generation', route_generation)
        self.context_factory = context_factory
        self.state = 'empty'
        self.handles = {}
        self.replays = 0
        self.warmup_replays = 0
        self.replays_by_bank = {self.name: 0}
        self.replays_by_stream = {}
        self.replays_by_descriptor = {}
        self.quarantine_reason = None

    # -- identity ---------------------------------------------------------
    def _require_ready(self):
        if self.state == 'destroyed':
            raise GraphRoutingError('decode probe bank is destroyed')
        if self.state == 'quarantined':
            raise GraphUnclean(f'decode probe bank is unclean: {self.quarantine_reason}')
        if self.state != 'ready':
            raise GraphRoutingError('decode probe bank is not captured')

    def _quarantine(self, reason):
        self.state = 'quarantined'
        self.quarantine_reason = reason

    def _key(self, descriptor):
        return ProbeGraphKey(descriptor, self.storage_epoch, self.route_generation)

    def no_prefill_guard(self):
        """The typed guard used in place of an anchor transaction."""
        if self.state == 'destroyed':
            raise GraphRoutingError('decode probe router is destroyed')
        if self.state == 'quarantined':
            raise GraphUnclean(f'decode probe router is unclean: {self.quarantine_reason}')
        return NoPrefillRoute(self.name, self.context.context_id, self.context.pool_id,
                              self.storage_epoch, self.route_generation)

    def set_route_generation(self, generation):
        _int('route_generation', generation)
        if self.state == 'destroyed':
            raise GraphRoutingError('decode probe router is destroyed')
        if generation != self.route_generation + 1:
            raise GraphRoutingError('probe route generation must advance exactly once')
        self.route_generation = generation
        return generation

    # -- capture / replay -------------------------------------------------
    def capture_at_startup(self):
        if self.state != 'empty':
            raise GraphRoutingError('decode probe bank was already captured')
        self.state = 'capturing'
        try:
            for descriptor in self.descriptors:
                handle = self.backend.capture(self.name, self._key(descriptor), None,
                                              self.context)
                if handle is None:
                    raise GraphRoutingError('split capture returned no graph handle')
                self.handles[descriptor] = handle
            if self.context_factory is not None:
                fresh = self.context_factory()
                if type(fresh) is not ExecutionContext:
                    raise GraphRoutingError('context factory did not return an ExecutionContext')
                if (fresh.context_id, fresh.pool_id, fresh.stream_token) != (
                        self.context.context_id, self.context.pool_id,
                        self.context.stream_token):
                    raise GraphRoutingError('measured split context changed identity')
                self.context = fresh
        except BaseException as exc:
            self._quarantine(f'capture failed: {type(exc).__name__}')
            raise
        self.state = 'ready'
        return tuple(self.descriptors)

    def supported_route(self, transaction, descriptor):
        """Exact captured descriptor on the declared split; nothing else."""
        self._require_ready()
        if type(transaction) is not NoPrefillRoute:
            raise GraphRoutingError('decode probe route needs the typed no-prefill guard')
        if type(descriptor) is not ExecutionDescriptor:
            raise GraphRoutingError('decode probe route needs an ExecutionDescriptor')
        if (transaction.bank != self.name or transaction.context_id != self.context.context_id
                or transaction.pool_id != self.context.pool_id):
            raise GraphRoutingError('no-prefill guard names another bank/context/pool')
        if transaction.storage_epoch != self.storage_epoch:
            raise GraphRoutingError('no-prefill guard names a stale storage epoch')
        if transaction.route_generation != self.route_generation:
            raise GraphRoutingError('no-prefill guard names a stale route generation')
        if descriptor not in self.handles:
            raise UnsupportedRoute('descriptor is not captured in the decode probe bank')
        return RouteDecision('split', self.name, descriptor,
                             'decode-only probe: exact captured FULL descriptor; no prefill')

    def replay(self, transaction, descriptor, *, observed_context, observed_stream):
        decision = self.supported_route(transaction, descriptor)
        if type(observed_context) is not int or type(observed_stream) is not int:
            raise GraphRoutingError('decode probe replay needs C\'s observed context and stream')
        if (observed_context != self.context.context_id
                or observed_stream != self.context.stream_token):
            raise GraphRoutingError('decode probe replay would run on the wrong context or stream')
        handle = self.handles[descriptor]
        try:
            output = self.backend.replay(handle, self._key(descriptor), None, self.context)
        except BaseException as exc:
            self._quarantine(f'replay failed: {type(exc).__name__}')
            raise
        self.replays += 1
        self.replays_by_bank[self.name] = self.replays_by_bank.get(self.name, 0) + 1
        self.replays_by_stream[observed_stream] = self.replays_by_stream.get(observed_stream, 0) + 1
        key = decision.descriptor
        self.replays_by_descriptor[key] = self.replays_by_descriptor.get(key, 0) + 1
        return output

    def warmup(self):
        """Replay every captured descriptor once; counted apart from measurement."""
        self._require_ready()
        for descriptor in self.descriptors:
            try:
                self.backend.replay(self.handles[descriptor], self._key(descriptor),
                                    None, self.context)
            except BaseException as exc:
                self._quarantine(f'warmup failed: {type(exc).__name__}')
                raise
            self.warmup_replays += 1
        return self.warmup_replays

    def counters(self):
        return {
            'state': self.state,
            'bank': self.name,
            'storage_epoch': self.storage_epoch,
            'route_generation': self.route_generation,
            'replays': self.replays,
            'replays_by_bank': dict(self.replays_by_bank),
            'replays_by_stream': dict(self.replays_by_stream),
            'replays_by_descriptor': {descriptor_name(descriptor): count
                                      for descriptor, count in self.replays_by_descriptor.items()},
            'warmup_replays': self.warmup_replays,
        }

    def shutdown(self):
        """Destroy this router's graph handles; the bank/pool follow in split_bank."""
        if self.state == 'destroyed':
            return ()
        order = []
        for descriptor in reversed(self.descriptors):
            handle = self.handles.pop(descriptor, None)
            if handle is not None:
                self.backend.destroy_graph(handle)
                order.append('graph')
        self.state = 'destroyed'
        return tuple(order)


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


@dataclass(frozen=True, repr=False)
class NativeDecodeRoute:
    """Typed decision for one ordinary decode step under the retained owner.

    It carries no request transaction and no per-step OperationTicket: only the
    engine epoch, worker incarnation, persistent decode storage epoch and the
    route generation bound by E's bootstrap.  ``mode='ordinary'`` means the
    stock full-GPU manager remains the exact route; ``mode='split'`` names the
    actual split bank that must replay the descriptor.
    """
    mode: str
    bank: str | None
    descriptor: ExecutionDescriptor
    context_id: int
    pool_id: int
    storage_epoch: int
    route_generation: int
    engine_epoch: int
    worker_rank: int
    incarnation: str = field(repr=False)
    reason: str = ''

    def __post_init__(self):
        if self.mode not in ('split', 'ordinary'):
            raise GraphRoutingError(f'unknown native decode route mode {self.mode!r}')
        if type(self.descriptor) is not ExecutionDescriptor:
            raise GraphRoutingError('native decode route needs an ExecutionDescriptor')
        if self.mode == 'split':
            if not isinstance(self.bank, str) or not self.bank:
                raise GraphRoutingError('native split route names a bank')
        elif self.bank is not None:
            raise GraphRoutingError('native ordinary route names no bank')
        _int('context_id', self.context_id)
        _int('pool_id', self.pool_id)
        _int('storage_epoch', self.storage_epoch, 1)
        _int('route_generation', self.route_generation)
        _int('engine_epoch', self.engine_epoch, 1)
        _int('worker_rank', self.worker_rank)
        _text('incarnation', self.incarnation)
        if self.reason:
            _text('reason', self.reason)

    def __repr__(self):
        return (f'NativeDecodeRoute(mode={self.mode!r}, bank={self.bank!r}, '
                f'descriptor={descriptor_name(self.descriptor)!r}, '
                f'storage_epoch={self.storage_epoch}, '
                f'route_generation={self.route_generation}, '
                f'engine_epoch={self.engine_epoch}, rank={self.worker_rank}, '
                f'incarnation=<len {len(self.incarnation)}>)')


@dataclass(frozen=True)
class NativeReplayLease:
    """One host-submitted native decode replay awaiting its queried event.

    Unlike :class:`ReplayLease` it has no request transaction and no operation
    ticket: the persistent decode storage owner is the identity.  Completion is
    only ever a queried CUDA event (see :meth:`GraphRouter.poll_native_replays`).
    """
    lease_id: int
    bank_name: str
    key: GraphKey
    context_id: int
    stream_token: int
    host_submit_seq: int
    storage_epoch: int
    engine_epoch: int
    route_generation: int

    def __post_init__(self):
        _int('lease_id', self.lease_id, 1)
        _text('bank_name', self.bank_name)
        if type(self.key) is not GraphKey:
            raise GraphRoutingError('native lease needs a GraphKey')
        _int('context_id', self.context_id)
        _int('stream_token', self.stream_token)
        _int('host_submit_seq', self.host_submit_seq, 1)
        _int('storage_epoch', self.storage_epoch, 1)
        _int('engine_epoch', self.engine_epoch, 1)
        _int('route_generation', self.route_generation)


@dataclass(frozen=True, repr=False)
class NativeReplayResult:
    """A native replay enqueue plus its opaque engine output."""
    lease: NativeReplayLease
    output: object = field(compare=False, repr=False)

    def __post_init__(self):
        if type(self.lease) is not NativeReplayLease:
            raise GraphRoutingError('native replay result needs a NativeReplayLease')

    def __repr__(self):
        return f'NativeReplayResult(lease={self.lease!r})'


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
                 transport_release=None, max_native_pending=8):
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
        if engine_epoch is not None and worker.engine_bound and engine_epoch != worker.engine_epoch:
            raise GraphRoutingError('router engine epoch differs from the bound worker')
        # An unbound worker/router cannot authorise a route until the engine's
        # bootstrap binds this incarnation exactly once.
        self._engine_bound = bool(worker.engine_bound)
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
        self._pending_context_factories = {}
        # Native retained-storage replays: no transaction, no ticket, one real
        # CUDA event each.  ``_native_pending`` is bounded so an unpolled worker
        # applies backpressure instead of accumulating unknown work.
        self._native_pending = {}
        self._native_submitted = 0
        self._native_completion_seq = 0
        self.max_native_pending = _int('max_native_pending', max_native_pending, 1)
        # The worker's queried census of actually completed split replays under
        # its persistent decode storage.  Transitions are fenced on this value.
        self.replay_watermark = 0

    @property
    def route_generation(self):
        """Local route generation; every applied transition advances it by one."""
        return self._transition_seq

    # --- engine owner ----------------------------------------------------
    @property
    def engine_owner_bound(self):
        return self._engine_bound

    def bind_engine_owner(self, engine_epoch, *, incarnation=None):
        """Bind this router to the engine owner once, after A bound the worker."""
        _int('engine_epoch', engine_epoch, 1)
        if incarnation is not None and incarnation != self.worker.incarnation:
            raise GraphRoutingError('engine owner names a foreign worker incarnation')
        if not self.worker.engine_bound:
            raise GraphRoutingError('worker engine owner is not bound')
        if engine_epoch != self.worker.engine_epoch:
            raise GraphRoutingError('router engine epoch differs from the worker owner')
        if self._engine_bound:
            if engine_epoch == self.engine_epoch:
                return self.engine_epoch
            raise GraphRoutingError('router is already bound to another engine epoch')
        self.engine_epoch = engine_epoch
        self._engine_bound = True
        return self.engine_epoch

    # --- inventory / capture --------------------------------------------
    @property
    def unclean(self):
        return self._quarantine is not None

    @property
    def mode(self):
        """'ordinary' (full GPU) or a registered split bank name."""
        return self.current_mode

    def add_bank(self, name, context, descriptors=(), *, context_factory=None):
        """Declare and register one split bank before startup capture.

        ``context_factory`` (optional) returns the freshly measured context
        after capture; the router validates that its identity/pool/stream agree
        and records the measured bytes in place of the pre-capture estimate.
        """
        self._require_ready()
        _text('name', name)
        if name == 'ordinary':
            raise GraphRoutingError("'ordinary' is reserved for the full-GPU route")
        if name in self.banks:
            raise GraphRoutingError('graph bank already registered')
        if context_factory is not None and not callable(context_factory):
            raise GraphRoutingError('bank context factory must be callable')
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
        self._pending_context_factories[name] = context_factory
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
                factory = self._pending_context_factories.get(name)
                if factory is not None:
                    measured = factory()
                    if type(measured) is not ExecutionContext:
                        raise GraphRoutingError('bank context factory did not return an ExecutionContext')
                    if (measured.context_id, measured.pool_id, measured.stream_token,
                            measured.measured_sm) != (
                            bank.context.context_id, bank.context.pool_id,
                            bank.context.stream_token, bank.context.measured_sm):
                        raise GraphRoutingError('measured split context changed identity')
                    bank.context = measured
                    bank.pool.context = measured
                    bank.allocated_bytes = measured.reserved_bytes
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

    # --- native retained-storage decode ---------------------------------
    def _require_native_owner(self):
        """The persistent storage owner bound once by E's bootstrap."""
        self._require_ready()
        if not self._engine_bound or not getattr(self.worker, 'engine_bound', False):
            raise E3Pending('native decode route before the engine owner is bound')
        if self.engine_epoch != self.worker.engine_epoch:
            raise GraphUnclean('router and worker name different engine epochs')
        owner = getattr(self.worker, 'decode_owner', None)
        if owner is None:
            raise GraphUnclean('native decode route has no persistent storage owner')
        if owner.lane_id != self.handle.lane_id:
            raise GraphUnclean('persistent decode storage owner changed; graph storage is stale')
        if self.worker.decode.state != 'retained':
            raise GraphUnclean('decode storage is not persistently retained')
        self._require_decode_lane()
        return owner

    def native_decode_route(self, descriptor):
        """Typed, transaction-free route for one ordinary decode SchedulerOutput.

        Returns ``None`` only for a declared no-op: a non-FULL descriptor while
        the full route is restored and no prefill is parked (stock piecewise or
        eager decode remains untouched).  Anything unsupported on the split
        route raises; there is no silent stock fallback on a green stream.
        """
        owner = self._require_native_owner()
        if type(descriptor) is not ExecutionDescriptor:
            raise GraphRoutingError('native decode route needs an ExecutionDescriptor')
        if descriptor.cg_mode != 'full':
            if self.current_mode == 'ordinary' and self.worker.active_prefill is None:
                return None
            raise UnsupportedRoute(
                'non-FULL descriptor cannot dispatch on the retained split route')
        if descriptor.num_tokens == 0:
            # Declared no-op: the stock engine returns before any graph dispatch.
            return None
        prefill_active = self.worker.active_prefill is not None
        if self.current_mode == 'ordinary':
            if prefill_active:
                raise UnsupportedRoute(
                    'full-GPU decode route is not allowed while prefill is active')
            return NativeDecodeRoute(
                'ordinary', None, descriptor, self.full_context_id, self.full_pool_id,
                self.handle.storage_epoch, self._transition_seq, self.engine_epoch,
                owner.worker_rank, owner.incarnation,
                'full route restored; stock full-GPU decode')
        bank = self.banks.get(self.current_mode)
        if bank is None:
            raise GraphUnclean('retained route names an unregistered bank')
        if not bank.ready:
            raise UnsupportedRoute(f'bank {bank.name!r} is {bank.state}')
        if bank.graph_for(descriptor, storage_epoch=self.handle.storage_epoch,
                          worker_rank=self.handle.worker_rank) is None:
            raise UnsupportedRoute(
                f'descriptor is not captured in split bank {bank.name!r}')
        return NativeDecodeRoute(
            'split', bank.name, descriptor, bank.context.context_id,
            bank.context.pool_id, self.handle.storage_epoch, self._transition_seq,
            self.engine_epoch, owner.worker_rank, owner.incarnation,
            'retained split storage; exact captured descriptor')

    def native_replay(self, descriptor, *, observed_context, observed_stream):
        """Enqueue the actual split-bank replay and its per-replay event.

        There is no ticket: the returned lease names the persistent storage
        owner, engine epoch and route generation.  The lease stays pending
        until :meth:`poll_native_replays` queries its event complete.
        """
        decision = self.native_decode_route(descriptor)
        if decision is None or decision.mode != 'split':
            raise GraphRoutingError('native split replay requested on the ordinary route')
        # Retire any already-complete leases first; pending ones stay bounded.
        self.poll_native_replays()
        if len(self._native_pending) >= self.max_native_pending:
            raise E3Pending(
                f'{len(self._native_pending)} native replay(s) pending; backpressure')
        bank = self.banks[decision.bank]
        if type(observed_context) is not int or type(observed_stream) is not int:
            raise GraphRoutingError('native replay needs C\'s observed context and stream')
        if (observed_context != bank.context.context_id
                or observed_stream != bank.context.stream_token):
            raise GraphRoutingError('native replay would run on the wrong context or stream')
        graph = bank.graph_for(descriptor, storage_epoch=self.handle.storage_epoch,
                               worker_rank=self.handle.worker_rank)
        if graph is None:
            raise UnsupportedRoute('no captured graph for the dispatched descriptor')
        backend = self.backend
        enqueue = getattr(backend, 'replay_with_event', None)
        if not callable(enqueue):
            raise GraphRoutingError(
                'split backend cannot record a per-replay completion event')
        self._lease_seq += 1
        self._host_seq += 1
        lease = NativeReplayLease(
            self._lease_seq, bank.name, graph.key, bank.context.context_id,
            bank.context.stream_token, self._host_seq, self.handle.storage_epoch,
            self.engine_epoch, self._transition_seq)
        try:
            output, event = enqueue(graph.handle, graph.key, bank.lane, bank.context)
        except BaseException:
            self._quarantine_bank(bank, 'native replay enqueue failed; completion unknown')
            raise
        if not callable(getattr(event, 'query', None)):
            self._quarantine_bank(bank, 'native replay has no queried event')
            raise GraphRoutingError('native replay needs a real CUDA event')
        if getattr(event, 'stream_token', None) != bank.context.stream_token:
            self._quarantine_bank(bank, 'native replay event names another stream')
            raise GraphRoutingError('native replay event was recorded on another stream')
        self._native_pending[lease.lease_id] = (lease, bank, event)
        self._native_submitted += 1
        return NativeReplayResult(lease, output)

    def poll_native_replays(self, *, limit=None):
        """Query pending replay events; advance the watermark only on completion."""
        self._require_ready()
        if limit is not None and (type(limit) is not int or limit < 1):
            raise GraphRoutingError('poll limit must be a positive int')
        completed = []
        for lease_id in tuple(self._native_pending):
            if limit is not None and len(completed) >= limit:
                break
            lease, bank, event = self._native_pending[lease_id]
            try:
                done = event.query()
            except BaseException as exc:
                self._quarantine_bank(bank, 'native replay event query failed')
                raise GraphRoutingError('native replay completion query failed') from exc
            if type(done) is not bool:
                self._quarantine_bank(bank, 'native replay event query is not boolean')
                raise GraphRoutingError('native replay completion query is not a boolean')
            if not done:
                continue
            del self._native_pending[lease_id]
            self._native_completion_seq += 1
            self.replay_watermark += 1
            self._timing.append(TimingRecord(
                lease.lease_id, lease.context_id, lease.storage_epoch,
                self.handle.worker_rank, lease.host_submit_seq,
                self._native_completion_seq, None, 'unmeasured'))
            completed.append(lease)
        return tuple(completed)

    @property
    def native_pending(self):
        """Unqueried native replay leases; never cleared without a completion."""
        return len(self._native_pending)

    def native_census(self):
        """Queried-completion census G can read without a parallel wrapper."""
        return {
            'watermark': self.replay_watermark,
            'pending': len(self._native_pending),
            'submitted': self._native_submitted,
            'max_pending': self.max_native_pending,
            'storage_epoch': self.handle.storage_epoch,
            'route_generation': self._transition_seq,
            'engine_epoch': self.engine_epoch,
        }

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
        self.replay_watermark += 1
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
        return len(self._pending) + len(self._native_pending)

    def descriptor_coverage(self):
        return {name: bank.coverage for name, bank in self.banks.items()}

    # --- transitions ----------------------------------------------------
    def validate_transition(self, target, authorization):
        """Non-mutating preflight for one route switch; returns ``target``.

        The worker bridge proves the locally drained switch before it touches
        the stream/bank, so a refused authorization never leaves a half-switch.
        """
        self._require_ready()
        if not self._engine_bound:
            raise GraphRoutingError('route transition before the engine owner is bound')
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
        if target == self.current_mode:
            # The bridge must not acknowledge a switch that changed no bank;
            # this is checked after the transient drain/capture conditions so
            # an unquiesced router still reports "pending", not "stale".
            raise GraphRoutingError('route transition would not change the bank')
        return target

    def transition(self, target, authorization):
        """Switch route only with C's explicit, completed authorization."""
        self.validate_transition(target, authorization)
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
        elif (self.worker.decode.outstanding or self.worker.prefill.outstanding
              or self._native_pending):
            state = 'busy'
        else:
            state = self.state
        return RouterStatus(state, self.current_mode,
                            len(self._pending) + len(self._native_pending),
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

    def quarantine(self, reason):
        """Poison this router's route ownership; never report a partial switch.

        Used by the worker bridge when a transition cannot be proven complete
        after it started mutating local route state.  Banks keep their graphs
        and leases: recovery is an operator decision, not a silent fallback.
        """
        if self.state == 'destroyed':
            raise GraphRoutingError('destroyed router cannot be quarantined')
        reason = _text('reason', reason)
        for bank in self.banks.values():
            if bank.state not in ('destroyed', 'quarantined'):
                bank.quarantine(reason)
        self._quarantine = reason
        self.state = 'quarantined'
        return reason

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
        if not self.worker.decode.active or self.worker.decode.state not in (
                'retained', 'ready', 'in_use'):
            raise GraphRoutingError('decode lane is not ready for graph replay')
        if self.worker.decode.failed:
            raise GraphUnclean('decode lane is quarantined')

    def _require_drained(self, action):
        if self._pending:
            raise E3Pending(f'{len(self._pending)} replay(s) outstanding before {action}')
        if self._native_pending:
            # Native replays join the route's drain: a transition, shutdown or
            # reuse may not cross an event that has not been queried complete.
            raise E3Pending(
                f'{len(self._native_pending)} native replay(s) outstanding before {action}')
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
    'BankBudget', 'CG_MODES', 'CapturedGraph', 'CompilationIdentity', 'DecodeProbeRouter',
    'ExecutionDescriptor', 'ExecutionContext', 'ExternalUsers', 'GraphBank',
    'GraphBudgetReport', 'GraphKey', 'GraphPool', 'GraphRouter', 'GraphRoutingError',
    'GraphUnclean', 'NativeDecodeRoute', 'NativeReplayLease', 'NativeReplayResult',
    'NoPrefillRoute', 'ProbeGraphKey', 'ReplayLease', 'ReplayResult',
    'RouteDecision', 'RouterStatus', 'ShutdownRecord', 'TimingRecord',
    'TransitionAuthorization', 'TransitionRecord', 'UnsupportedRoute', 'descriptor_name',
]
