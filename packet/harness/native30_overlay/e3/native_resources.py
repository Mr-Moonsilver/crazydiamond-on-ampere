"""Worker-owned native construction for the E3 overlap lane.

``build_worker_runtime`` is called by ``s5_startup.install_builder``'s builder,
which the *generated* runner invokes immediately before A's ``init_lane`` at the
end of model-runner construction.  It is the real worker startup path: it
discovers the worker rank/device and the canonical QSA UID namespace, allocates
the private prefill UID buffer and scratch inventory, installs the event
completion backend and the typed worker RPC adapter.  The engine epoch stays
unbound; ``install_worker_native`` later retains the persistent decode storage
and installs B's real split-bank graph builder before the engine's capture call.

It allocates storage, therefore it is only ever called with
``FLASH_S5_OVERLAP=1``.  Everything vLLM/torch specific is imported lazily so
the module loads on the CPU hub.

Deliberately *not* here (see Open items in the pass report):

* discovering model-specific private recurrent/HC/module tensors (the caller's
  ``private_attrs`` stay empty rather than guessing attribute names);
* committing real model/host state (the commit value is supplied by admission);
* the model-forward callable for split-graph capture (``_flash_s5_graph_capture``);
* cross-layer host continuation: the generated attention yields at a bucket,
  but the native prefix/layer/suffix dispatcher is not installed. The old
  whole-forward runner seam refuses even in fixtures unless explicitly replaced.
"""
import hashlib
import importlib
import time
from dataclasses import dataclass

from async_handoff import (
    CommitAck,
    CompletionReport,
    HandoffConfig,
    HandoffError,
    PollOperations,
    RuntimeWorkerHandoff,
)
from e3_types import E3ContractError, E3Pending
from graph_router import CompilationIdentity
from lane_hooks import LaneRuntime
from lane_state import (
    GRAPH_POOL_MEASURED_BYTES,
    WorkerState,
    prefill_scratch_specs,
)
from native_backend import (
    EventCompletionBackend,
    NativeDevice,
)
from qsa_operation import (
    RUNNER_SEAM,
    NativeQsaDevice,
    OperationError,
    QsaPrefillOperation,
    metadata_plan_provider,
)

ENGINE_EPOCH = 1                   # legacy default; the native worker binds later
PREFLIGHT_CONTEXT_ID = 1
DECODE_CONTEXT_ID = 1          # B requires the split context to equal A's lane.
FULL_CONTEXT_ID = 0            # the pinned stock decode execution context
SPLIT_RESERVED_BYTES = GRAPH_POOL_MEASURED_BYTES
WORKSPACE_LANE = 1


@dataclass(frozen=True)
class FullGraphIdentity:
    """Identity of the stock full-GPU decode bank, with an honest basis.

    The pool number comes from the stock manager that actually captured the
    bank; it is never a placeholder.  ``allocated_bytes`` remains the reviewed
    budget constant because this seam has no pre/post ``capture_model`` memory
    observation to attribute; it is therefore an estimate, not a measured
    allocation, and cannot supply qualification.  ``context_id`` is B's
    declared primary-context scope label, not a CUPTI/CUDA context identity.
    """
    context_id: int
    pool_id: int
    allocated_bytes: int
    basis: str = 'estimate'


class NativeResourcesError(E3ContractError):
    pass


# --- discovery --------------------------------------------------------------

def _worker_rank(runner):
    config = getattr(runner, 'vllm_config', None)
    parallel = getattr(config, 'parallel_config', None)
    rank = getattr(parallel, 'rank', None)
    if rank is None:
        rank = getattr(runner, 'worker_rank', None)
    if type(rank) is not int or rank < 0:
        raise NativeResourcesError('cannot discover the worker rank for E3 startup')
    return rank


def _runner_device(runner):
    device = getattr(runner, 'device', None)
    if device is None:
        raise NativeResourcesError('runner reports no CUDA device')
    return device


def _max_num_reqs(runner):
    value = getattr(runner, 'max_num_reqs', None)
    if type(value) is not int or value < 1:
        raise NativeResourcesError('runner reports no max_num_reqs for the UID buffer')
    return value


def _import_forward_context():
    """Pure import leaf; constructing/binding QSA state remains GPU-owned."""
    return importlib.import_module('vllm.forward_context')


def _forward_context_module():
    module = _import_forward_context()
    if not hasattr(module, 'QSA_REQ_UID'):
        raise NativeResourcesError(
            'vllm.forward_context has no QSA_REQ_UID buffer; the pinned QSA '
            'forward-context module is required for lane-private request UIDs')
    if not hasattr(module, '_forward_context'):
        raise NativeResourcesError(
            'vllm.forward_context has no module-level _forward_context')
    return module


def _workspace_module():
    return importlib.import_module('vllm.v1.worker.workspace')


def response_success():
    """The actual ``WorkerProc.ResponseStatus.SUCCESS`` enum, never a literal."""
    module = importlib.import_module('vllm.v1.executor.multiproc_executor')
    status = getattr(getattr(module, 'WorkerProc', None), 'ResponseStatus', None)
    success = getattr(status, 'SUCCESS', None)
    if success is None:
        raise NativeResourcesError(
            'pinned WorkerProc.ResponseStatus.SUCCESS is unavailable')
    return success


def _weights(runner):
    getter = getattr(runner, 'get_model', None)
    model = getter() if callable(getter) else getattr(runner, 'model', None)
    address, total = 0, 0
    parameters = getattr(model, 'parameters', None)
    if callable(parameters):
        try:
            for tensor in parameters():
                pointer = int(tensor.data_ptr())
                length = int(tensor.numel() * tensor.element_size())
                address = pointer if address == 0 else min(address, pointer)
                total += length
        except Exception:
            address, total = 0, 0
    return model, address, total


def compilation_identity(runner):
    config = getattr(runner, 'compilation_config', None)
    if config is None:
        vllm_config = getattr(runner, 'vllm_config', None)
        config = getattr(vllm_config, 'compilation_config', None)
    mode = getattr(config, 'cudagraph_mode', None)
    name = getattr(mode, 'name', mode)
    mode_name = name.lower() if isinstance(name, str) and name.lower() in (
        'none', 'piecewise', 'full') else 'full'
    sizes = tuple(int(size) for size in (getattr(config, 'cuda_graph_sizes', ()) or ()))
    query_len = getattr(runner, 'decode_query_len', 1)
    if type(query_len) is not int or query_len < 1:
        query_len = 1
    digest = hashlib.sha256(
        f'{mode_name}:{sorted(sizes)}:{query_len}'.encode()).hexdigest()
    return CompilationIdentity(digest, mode_name, sizes or (1,), query_len)


def full_pool_id(runner):
    """The stock full-GPU graph pool's actual numeric identity.

    vLLM returns either an int or a two-int tuple; both are accepted, and
    anything else is refused rather than guessed.  The split bank applies the
    same formula to its separately allocated pool, so the two can never be
    mistaken for one another.
    """
    manager = getattr(runner, 'cudagraph_manager', None)
    if manager is None:
        raise NativeResourcesError('runner has no stock cudagraph manager for the full pool')
    pool = getattr(manager, 'pool', None)
    if isinstance(pool, tuple):
        if len(pool) != 2 or any(type(part) is not int for part in pool):
            raise NativeResourcesError('stock graph pool identity is not a pair of ints')
        return int(pool[0]) * 1000003 + int(pool[1])
    if type(pool) is int:
        return pool
    raise NativeResourcesError('stock graph pool identity is unavailable')


def full_graph_identity(runner):
    """Identity of the stock full-GPU decode bank; bytes stay an estimate."""
    return FullGraphIdentity(FULL_CONTEXT_ID, full_pool_id(runner),
                             GRAPH_POOL_MEASURED_BYTES, 'estimate')


class _ForwardContextAdapter:
    """A's HostDispatcher adapter over ``vllm.forward_context``'s module global."""

    def __init__(self, module):
        self.module = module

    def get(self):
        return getattr(self.module, '_forward_context', None)

    def set(self, value):
        self.module._forward_context = value


class _WorkspaceLaneAdapter:
    """A's HostDispatcher adapter over ``use_workspace_lane``."""

    def __init__(self, module, lane):
        self.module = module
        self.lane = lane
        self._stack = []

    def enter(self, workspace):
        if workspace != self.lane:
            raise NativeResourcesError('prefill workspace lane differs from startup')
        manager = self.module.use_workspace_lane(workspace)
        manager.__enter__()
        self._stack.append(manager)

    def exit(self, workspace):
        if not self._stack:
            raise NativeResourcesError('workspace exit without enter')
        self._stack.pop().__exit__(None, None, None)


def _ensure_workspace_lanes(module, device, lane):
    manager = module.current_workspace_manager()
    if getattr(manager, '_num_lanes', 0) > lane:
        return manager
    # Reinitializing drops the existing manager and its captured addresses.
    raise NativeResourcesError('workspace lanes must be provisioned before model allocation/capture')


def _prefill_context(module, runner):
    factory = getattr(module, 'create_forward_context', None)
    config = getattr(runner, 'vllm_config', None)
    if callable(factory) and config is not None:
        return factory(None, config)
    raise NativeResourcesError(
        'vllm.forward_context.create_forward_context is required to build the '
        'prefill forward context')


def _assert_distinct(decode, prefill):
    if hasattr(decode, 'data_ptr') and hasattr(prefill, 'data_ptr'):
        if int(decode.data_ptr()) == int(prefill.data_ptr()):
            raise NativeResourcesError('prefill UID buffer aliases decode storage')
    elif decode is prefill:
        raise NativeResourcesError('prefill UID buffer aliases decode storage')


def _allocate_scratch(device, chunk_tokens):
    bindings = []
    inventory = []
    buffers = []
    for spec in prefill_scratch_specs(chunk_tokens):
        buffer = device.allocate(spec.nbytes, name=spec.name)
        buffers.append(buffer)
        bindings.append((spec, buffer.address))
        inventory.append((spec.name, buffer.nbytes, spec.basis))
    return tuple(bindings), tuple(inventory), tuple(buffers)


# --- typed worker RPC adapter ----------------------------------------------

class WorkerHandoffServer:
    """The ``_flash_s5_worker`` object the generated worker RPCs dispatch to.

    One transaction at a time.  The producer submission drives the generated
    runner seam ``_flash_s5_execute_prefill_operation``. That seam refuses until
    the native bounded dispatcher is installed. The standalone generated QSA
    generator owns a producer -> CPU-source DMA -> attention continuation.  No stock
    ``sample_tokens`` or output conversion belongs to this path.  The commit
    role records a device event for its own ticket only when the lane backend
    maps that role; a generic event is never turned into a state-write proof.
    """

    def __init__(self, runtime, *, bindings=(), private_attrs=None,
                 operation_device=None, clock=time.monotonic, row1_factory=None,
                 row1_store=None):
        if not isinstance(runtime, LaneRuntime):
            raise NativeResourcesError('worker server needs A LaneRuntime')
        self.runtime = runtime
        self.bindings = tuple(bindings)
        self.private_attrs = dict(private_attrs or {})
        self.operation_device = operation_device
        self.clock = clock
        self.row1_factory = row1_factory
        self.row1_store = row1_store
        self.row1_provider = None
        self._row1_scopes = {}
        self._row1_run_id = None
        self.worker = None
        self.handoff = None
        self.operation = None
        self.last_output = None
        self.started_s = clock()
        self.executions = 0
        self._operation_layers = ()
        self._deferred_proofs = []
        # The exact metadata snapshot of the frame this worker opened; the
        # source binding's per-layer tables are bound against it, never against
        # a later or coalesced snapshot.
        self._metadata_snapshot = None

    def attach_worker(self, worker):
        """The generated GPU worker attaches itself before any typed RPC."""
        self.worker = worker
        return self

    # -- typed protocol ---------------------------------------------------
    def open(self, message):
        if self.handoff is not None:
            raise HandoffError('a prefill transaction is already open on this worker')
        if getattr(message, 'metadata', None) is None:
            raise HandoffError('open needs the immutable admission metadata snapshot')
        transaction = message.transaction
        # The engine owns the transaction identity; this worker only records the
        # request generation it carries. Re-declaring the same generation is
        # inert; a newer generation while a prefill is leased is refused by A.
        self.runtime.state.declare_request(transaction.request_uid,
                                           transaction.request_generation)
        handoff = RuntimeWorkerHandoff(
            self.runtime, transaction, metadata=message.metadata,
            bindings=self.bindings, private_attrs=self.private_attrs,
            clock=self.clock)
        ack = handoff.open(message)
        self.handoff = handoff
        lane = self.runtime.state.prefill
        if lane.handle is None or lane.transaction != transaction:
            raise HandoffError('opened handoff has no active prefill lane')
        try:
            provider = metadata_plan_provider(message.metadata)
        except OperationError:
            provider = None  # refused at bind_plan, before any DMA
        self.operation = QsaPrefillOperation(
            transaction, lane.lane_id, device=self.operation_device,
            plan_provider=provider, clock=self.clock)
        self._metadata_snapshot = message.metadata
        self.last_output = None
        return ack

    def submit(self, message):
        handoff = self._handoff(message.transaction)
        ack = handoff.submit(message)
        if message.dependency_role is _producer_role():
            if getattr(message, 'scheduler_output', None) is None:
                raise HandoffError(
                    'producer submit needs the parked scheduler output to execute')
            self._execute_prefill(message)
        elif message.dependency_role is _commit_role():
            # The event is recorded after the forward launch and after any
            # earlier event on the same stream; it is not a separate kernel.
            ticket = handoff.last_ticket
            if ticket is None:
                raise HandoffError('commit submit recorded no outstanding ticket')
            self._record_launch(ticket)
        return ack

    def poll(self, message):
        report = self._handoff(message.transaction).poll(message)
        self._deferred_proofs.extend(report.proofs)
        if self.operation is not None:
            # Internal operation tickets are a separate namespace. Do not send
            # them as engine tickets, or expose a launch proof as a drain proof.
            self.operation.poll()
            if self.operation._failed:
                raise HandoffError('QSA operation is quarantined')
            if not self.operation.drain():
                return CompletionReport(report.transaction, (),
                                        report.pending + max(1, self.operation.pending),
                                        report.failed)
        proofs, self._deferred_proofs = tuple(self._deferred_proofs), []
        return CompletionReport(report.transaction, proofs, report.pending, report.failed)

    def commit(self, message):
        handoff = self._handoff(message.transaction)
        state_value = getattr(message, 'state_value', None)
        if state_value is None:
            raise HandoffError(
                'commit needs the admitted host/model state value; the worker '
                'must not invent a persistent state')
        if self.operation is not None and not self.operation.drain():
            raise E3Pending('commit waits for outstanding QSA operation events')
        self.runtime.propose_state(message.transaction, state_value)
        ack = handoff.commit(message)
        output = self.last_output
        self._release_if_clean(message.transaction)
        return _with_output(ack, output)

    def cancel(self, message):
        handoff = self._handoff(message.transaction)
        if self.operation is not None:
            # Stop new submissions; submitted consumers stay owned until their
            # events are observed by the polls below.
            self.operation.cancel('transaction cancelled')
        ack = handoff.cancel(message)
        # Completion polling continues after the stop; one bounded poll lets an
        # already-ready event retire the lane in this call. Anything still
        # outstanding keeps storage leased rather than pretending it drained.
        handoff.poll(PollOperations(message.transaction, 8))
        if self.operation is not None:
            self.operation.poll()
        self._release_if_clean(message.transaction)
        return ack

    def clear_native(self, transaction):
        """Drop the old handoff/operation after the native bridge closed the row.

        The legacy typed protocol keeps its ``commit``/``cancel`` behaviour; only
        the native bridge calls this, and only after its dispatcher drained and
        released. A terminal release has also closed the worker request row; a
        nonterminal one leaves that row to the chunk factory for ``update()``.
        """
        if self.handoff is None or self.handoff.transaction != transaction:
            raise HandoffError('no worker handoff for this transaction')
        self.handoff = None
        self.operation = None
        self.last_output = None
        self._deferred_proofs = []
        self._operation_layers = ()
        self._metadata_snapshot = None
        # A row1 record store passed to this server survives clear_native: the
        # raw qualification records must remain queryable after release.
        return True

    # -- helpers ----------------------------------------------------------
    def _handoff(self, transaction):
        if self.handoff is None or self.handoff.transaction != transaction:
            raise HandoffError('no worker handoff for this transaction')
        return self.handoff

    def _execute_prefill(self, message):
        """Drive one parked chunk through the generated operation seam.

        This method never launches the stock worker forward itself, never
        samples a token and never converts a segment output.  The generated
        runner seam installs the bounded QSA operation on the real layers; the
        pinned layer/attention call sites enqueue bounded copies and record
        their producer/copy/consumer events.
        """
        from s5_startup import require_native_ready
        require_native_ready()
        if self.operation is None:
            raise HandoffError('prefill submission has no open worker operation')
        handoff = self.handoff
        if handoff is None:
            raise HandoffError('prefill submission has no open worker handoff')
        if getattr(message, 'grammar_output', None) is not None:
            raise HandoffError(
                'bounded prefill operation cannot consume a grammar output or '
                'return a stock sampled segment')
        worker = self.worker
        runner = getattr(worker, 'model_runner', None)
        seam = getattr(runner, RUNNER_SEAM, None)
        if not callable(seam):
            raise NativeResourcesError(
                'generated runner has no ' + RUNNER_SEAM +
                ' seam; the stock whole-forward path is refused')
        scheduler_output = message.scheduler_output
        current = getattr(scheduler_output, 'flash_e3_transaction', None)
        if current is not None and current != message.transaction:
            raise HandoffError('parked scheduler output carries another transaction')
        scheduler_output.flash_e3_transaction = message.transaction
        # Clear before the seam so a detached output the forward's output path
        # delivers during it is what commit publishes; the stock sampler is
        # never called and a forward that delivers nothing leaves ``None``.
        self.last_output = None
        seam(scheduler_output)
        # The forward is enqueued (not synchronized) on this worker's stream;
        # record the producer ticket's device event, which the backend maps as
        # GPU_EVENT.  This is not a state-write or PP-consumer proof.
        ticket = handoff.last_ticket
        if ticket is None:
            raise HandoffError('producer submit recorded no outstanding ticket')
        self._record_launch(ticket)
        self.executions += 1

    # -- generated operation seam -----------------------------------------
    def begin_prefill_operation(self, runner, scheduler_output):
        """Install the bounded operation on every QSA layer of this forward."""
        operation = self.operation
        if operation is None:
            raise HandoffError('no QSA operation is bound to this transaction')
        if self._operation_layers:
            raise HandoffError('a QSA operation is already installed on this worker')
        transaction = getattr(scheduler_output, 'flash_e3_transaction', None)
        if transaction != operation.transaction:
            raise HandoffError('scheduler output does not carry the operation transaction')
        layers = _qsa_layers(runner)
        binding = getattr(scheduler_output, 'flash_e3_source_binding', None)
        try:
            if binding is not None:
                # The bridge proved and retained this exact frame's pinned
                # histories; bind their tables, epochs and versions explicitly.
                bind_operation_sources(operation, layers, binding,
                                       self._metadata_snapshot)
                if self.row1_factory is not None:
                    operation.qualification = self.row1_factory(operation)
            else:
                for name, layer in layers:
                    operation.register_source(name, layer._qsa_host_kv)
            operation.begin_forward(layers=tuple(name for name, _ in layers))
        except BaseException:
            operation.cancel('operation layer binding failed')
            raise
        installed = []
        try:
            for name, layer in layers:
                if getattr(layer, '_flash_s5_operation', None) is not None:
                    raise NativeResourcesError(
                        'QSA layer already carries an S5 operation: ' + name)
                layer._flash_s5_operation = operation
                installed.append(layer)
        except BaseException:
            for layer in installed:
                if getattr(layer, '_flash_s5_operation', None) is operation:
                    delattr(layer, '_flash_s5_operation')
            operation.cancel('operation layer install failed')
            raise
        self._operation_layers = tuple(layers)
        return operation

    def end_prefill_operation(self, runner):
        """Uninstall after a completed forward; every QSA layer must have run."""
        operation = self.operation
        layers, self._operation_layers = self._operation_layers, ()
        self._uninstall(layers, operation)
        if operation is None:
            raise HandoffError('no QSA operation is bound to this transaction')
        operation.end_forward()
        return operation

    def abort_prefill_operation(self, runner):
        """Uninstall after a failed forward; keep consumers owned until drained."""
        operation = self.operation
        layers, self._operation_layers = self._operation_layers, ()
        self._uninstall(layers, operation)
        if operation is not None:
            operation.cancel('prefill forward failed')
        return operation

    def _uninstall(self, layers, operation):
        for _, layer in layers:
            if getattr(layer, '_flash_s5_operation', None) is operation:
                delattr(layer, '_flash_s5_operation')

    def accept_detached_output(self, output):
        """Record the detached chunk output the PP/transport worker delivers.

        D2 does not implement native output detachment or a PP sample ring.
        This seam is where that path will deliver its result; the stock
        sampler is never called here, and a missing output leaves ``None``.
        """
        if self.operation is None:
            raise HandoffError('no open operation can accept a detached output')
        self.last_output = output
        return output

    def operation_inventory(self):
        """Metadata-only operation inventory; no device query or readback."""
        if self.operation is None:
            return None
        inventory = self.operation.inventory()
        inventory['transaction_seq'] = self.operation.transaction.txn_seq
        inventory['pending_buckets'] = self.operation.pending_buckets
        return inventory

    def configure_row1(self, *, enabled, run_id, engine_epoch, route_generation):
        """Scope the serial oracle to a paused/drained qualification phase.

        The native qualification action owns the collective pause/drain. This
        second boundary checks the worker's actual idle prefill ownership and
        bound epoch, and the installed provider queries its real route. Stores
        remain retained after disable; an old run id is never recycled.
        """
        if (type(enabled) is not bool or type(run_id) is not str
                or not 1 <= len(run_id) <= 128):
            raise HandoffError('row1 scope needs a bounded run id and bool')
        state = self.runtime.state
        if (not state.engine_bound or engine_epoch != state.engine_epoch
                or type(engine_epoch) is not int):
            raise HandoffError('row1 scope has a stale engine epoch')
        if type(route_generation) is not int or route_generation < 0:
            raise HandoffError('row1 scope needs a route generation')
        lane = state.prefill
        dispatcher = getattr(self, '_flash_s5_dispatcher', None)
        bridge = getattr(self, '_flash_s5_bridge', None)
        if (self.runtime.prefill_execution is not None
                or (bridge is not None and (bridge.active
                    or bridge._retained is not None
                    or bridge._source_binding is not None
                    or bridge._source_close is not None))
                or self.handoff is not None or self.operation is not None
                or lane.handle is not None or lane.outstanding
                or (dispatcher is not None and (
                    dispatcher.transaction is not None
                    or dispatcher.quarantined is not None))):
            raise E3Pending('row1 scope waits for closed prefill ownership')
        provider = self.row1_provider
        if not callable(provider):
            raise HandoffError('row1 scoped provider is unavailable')
        if self._row1_run_id == run_id:
            scope = self._row1_scopes[run_id]
            if (scope['engine_epoch'] != engine_epoch
                    or route_generation < scope['route_generation']
                    or (enabled and route_generation != scope['route_generation'])):
                raise HandoffError('row1 scope identity changed')
            if scope['enabled'] is enabled:
                return self.row1_evidence(run_id=run_id)
            if enabled:
                raise HandoffError('a disabled row1 scope cannot be reopened')
            # Verify the native route is still the one that was qualified.
            provider(None, engine_epoch, route_generation)
            self.row1_factory = None
            scope['enabled'] = False
            scope['disabled_route_generation'] = route_generation
            return self.row1_evidence(run_id=run_id)
        if not enabled:
            raise HandoffError('row1 disable names an unknown scope')
        if run_id in self._row1_scopes:
            raise HandoffError('a retained row1 scope cannot be reused')
        if self.row1_factory is not None:
            raise HandoffError('another row1 scope is still enabled')
        if len(self._row1_scopes) >= 8:
            raise HandoffError('row1 scope capacity reached; evidence retained')
        from row1_oracle import Row1RecordStore
        store = Row1RecordStore()
        factory, layers = provider(store, engine_epoch, route_generation)
        if (not callable(factory) or type(layers) is not tuple or not layers
                or any(type(name) is not str or not name for name in layers)
                or len(set(layers)) != len(layers)):
            raise HandoffError('row1 provider has no exact local QSA census')
        scope = dict(engine_epoch=engine_epoch, route_generation=route_generation,
                     local_layers=layers, enabled=True, store=store)
        self._row1_scopes[run_id] = scope
        self._row1_run_id = run_id
        self.row1_store, self.row1_factory = store, factory
        return self.row1_evidence(run_id=run_id)

    def row1_evidence(self, *, run_id):
        scope = self._row1_scopes.get(run_id)
        if scope is None:
            raise HandoffError('row1 evidence names an unknown scope')
        state = self.runtime.state
        enabled = (scope['enabled'] and self._row1_run_id == run_id
                   and self.row1_factory is not None)
        return dict(schema='s5-native-row1-store/1', run_id=run_id,
                    rank=state.worker_rank, worker_incarnation=state.incarnation,
                    engine_epoch=scope['engine_epoch'],
                    route_generation=scope['route_generation'],
                    disabled_route_generation=scope.get('disabled_route_generation'),
                    local_layers=list(scope['local_layers']),
                    enabled=enabled,
                    raw=self._query_row1_store(scope['store'], enabled=enabled))

    def row1_records(self, *, layer=None, limit=None):
        """The bounded raw comparison store, queryable after release."""
        if self.row1_store is None:
            return ()
        return self.row1_store.records(layer=layer, limit=limit)

    def query_row1(self, *, layer=None, limit=None):
        """A bounded summary plus the raw records for one layer or all layers."""
        return self._query_row1_store(self.row1_store,
            enabled=self.row1_factory is not None, layer=layer, limit=limit)

    @staticmethod
    def _query_row1_store(store, *, enabled, layer=None, limit=None):
        if store is None:
            return dict(enabled=False, comparisons=0, layers=0, failures=0,
                        capacity=0, records=(), layer_records=(),
                        failure_records=())
        summary = dict(store.summary(), enabled=enabled)
        summary['records'] = tuple(r.as_dict() for r in
                                   store.records(layer=layer, limit=limit))
        summary['layer_records'] = tuple(r.as_dict() for r in
                                         store.layer_records(layer=layer, limit=limit))
        summary['failure_records'] = tuple(r.as_dict() for r in
                                           store.failures(layer=layer, limit=limit))
        return summary

    def _record_launch(self, ticket):
        recorder = getattr(self.runtime.backend, 'record_launch', None)
        if recorder is None:
            raise NativeResourcesError(
                'worker completion backend cannot record a launch event')
        recorder(ticket)

    def _release_if_clean(self, transaction):
        if self.operation is not None:
            try:
                self.operation.release()
            except E3Pending:
                # Copy/consumer events still outstanding: keep every owner.
                return False
            self.operation = None
        try:
            self.runtime.release(transaction)
        except E3Pending:
            # Outstanding completions keep storage leased; poll keeps driving.
            return False
        self.handoff = None
        return True


def bind_operation_sources(operation, layers, binding, snapshot):
    """Bind the bridge's worker-local source lease to the QSA operation.

    ``binding`` is produced by :func:`native_source.bind_local_sources` over the
    exact frame this worker opened and the actual pinned ``_qsa_host_kv``
    tensors. Every local QSA layer must appear exactly once, own that same
    tensor, and carry its epoch/version census. Anything missing, extra or
    foreign refuses before the first copy can be planned.
    """
    if operation is None:
        raise HandoffError('source binding needs the open QSA operation')
    if binding is None:
        raise HandoffError('source binding needs the bridge-bound lease')
    if binding.transaction != operation.transaction:
        raise HandoffError('source binding belongs to another transaction')
    if binding.rank != operation.lane.worker_rank:
        raise HandoffError('source binding belongs to another worker rank')
    if snapshot is None:
        raise HandoffError('source binding needs the opened admission snapshot')
    names = tuple(name for name, _ in layers)
    if len(set(names)) != len(names):
        raise HandoffError('a QSA layer appears twice in the operation binding')
    if tuple(sorted(names)) != binding.layers:
        raise HandoffError('source binding does not cover exactly the QSA layers')
    for name, layer in layers:
        source = binding.sources.get(name)
        if source is None or source is not getattr(layer, '_qsa_host_kv', None):
            raise HandoffError('source binding does not own this layer history')
        if name not in binding.source_epochs or name not in binding.versions:
            raise HandoffError('source binding has no epoch/version census')
    operation.plan_provider = metadata_plan_provider(
        snapshot, layer_tables=binding.layer_tables)
    operation.state_epoch = binding.state_epoch
    for name, _ in layers:
        operation.register_source(
            name, binding.sources[name],
            source_epoch=binding.source_epochs[name],
            versions=binding.versions[name])
        # Attach the admitted hold now, not only when the layer driver first
        # runs this layer: a cancel before that point still releases every
        # retained history, so the lane can close its registry views.
        hold = binding.holds.get(name)
        if hold is None:
            raise HandoffError('source binding has no hold for ' + name)
        operation.attach_sparse_source(name, hold)
    return operation


def _qsa_layers(runner):
    """Every offloaded QSA layer that must run through the operation."""
    model = getattr(runner, 'model', None)
    if not callable(getattr(model, 'modules', None)):
        raise NativeResourcesError('runner model cannot enumerate QSA layers')
    layers = []
    seen = set()
    for module in model.modules():
        if not getattr(module, '_qsa_kv_offload', False):
            continue
        name = getattr(module, 'layer_name', None)
        source = getattr(module, '_qsa_host_kv', None)
        if not isinstance(name, str) or not name or name in seen:
            raise NativeResourcesError('QSA layer identity is missing or duplicated')
        if source is None:
            raise NativeResourcesError(
                'QSA layer has no pinned host KV pool: ' + name)
        seen.add(name)
        layers.append((name, module))
    if not layers:
        raise NativeResourcesError(
            'prefill transaction has no offloaded QSA layer to operate on')
    return tuple(layers)


def _with_output(ack, output):
    return CommitAck(ack.transaction, ack.epoch, ack.published, output)


def _producer_role():
    from e3_types import DependencyRole

    return DependencyRole.PRODUCER


def _commit_role():
    from e3_types import DependencyRole

    return DependencyRole.COMMIT


# --- startup ----------------------------------------------------------------

def build_worker_runtime(runner, kv_caches, *, env=None, clock=time.monotonic):
    """Construct this worker's native E3 state; called by the generated runner."""
    config = HandoffConfig.from_env(env)
    if not config.enabled:
        raise NativeResourcesError('worker runtime requires FLASH_S5_OVERLAP=1')
    from s5_startup import require_native_ready
    require_native_ready()
    device = NativeDevice(device=_runner_device(runner))
    rank = _worker_rank(runner)
    forward_context = _forward_context_module()
    workspace = _workspace_module()
    max_reqs = _max_num_reqs(runner)

    uid_buffer = device.allocate_i32(max_reqs, name='flash_s5_prefill_uid')
    _assert_distinct(getattr(forward_context, 'QSA_REQ_UID', None), uid_buffer.tensor)
    bindings, scratch_inventory, scratch_buffers = _allocate_scratch(device, config.chunk_tokens)
    _ensure_workspace_lanes(workspace, device, WORKSPACE_LANE)

    model, weights_address, weights_bytes = _weights(runner)
    # No engine epoch is known at worker construction.  The engine's bootstrap
    # binds this incarnation exactly once before any transaction is admitted.
    state = WorkerState(rank, engine_epoch=0, engine_bound=False, weights=model,
                        weights_address=weights_address,
                        weights_bytes=weights_bytes,
                        max_outstanding=min(config.chunk_tokens, 16))
    backend = EventCompletionBackend(device, clock=clock)
    runtime = LaneRuntime(
        state, forward_context=_ForwardContextAdapter(forward_context),
        workspace=_WorkspaceLaneAdapter(workspace, WORKSPACE_LANE),
        submit_backend=backend,
        prefill_forward_context=_prefill_context(forward_context, runner),
        prefill_workspace=WORKSPACE_LANE,
        uid_namespace=forward_context, prefill_uid_buffer=uid_buffer.tensor)
    # Address tuples are not owners: keep every tensor alive for the runtime.
    runtime._native_owned_buffers = (uid_buffer, *scratch_buffers)
    operation_device = NativeQsaDevice(_runner_device(runner), clock=clock, env=env)
    server = WorkerHandoffServer(runtime, bindings=bindings,
                                 operation_device=operation_device, clock=clock)

    runner._flash_s5_worker = server
    runner._flash_s5_native = {
        'rank': rank,
        'engine_epoch': 0,
        'engine_bound': False,
        'incarnation': state.incarnation,
        'uid_bytes': uid_buffer.nbytes,
        'scratch': scratch_inventory,
        'workspace_lane': WORKSPACE_LANE,
        'weights_bytes': weights_bytes,
        'operation': 'qsa_dense_bounded',
        'native_qualified': False,
    }
    # The real split-bank graph builder is installed by install_worker_native
    # after the bank and the retained decode storage exist.
    return runtime


__all__ = [
    'DECODE_CONTEXT_ID', 'ENGINE_EPOCH', 'FULL_CONTEXT_ID', 'FullGraphIdentity',
    'NativeResourcesError', 'PREFLIGHT_CONTEXT_ID', 'SPLIT_RESERVED_BYTES',
    'WORKSPACE_LANE', 'WorkerHandoffServer', 'build_worker_runtime',
    'compilation_identity', 'full_graph_identity', 'full_pool_id',
    'response_success',
]
