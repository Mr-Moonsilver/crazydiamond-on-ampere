"""Worker-owned native construction for the E3 overlap lane.

``build_worker_runtime`` is called by ``s5_startup.install_builder``'s builder,
which the *generated* runner invokes immediately before A's ``init_lane`` at the
end of model-runner construction.  It is the real worker startup path: it
discovers the worker rank/device and the canonical QSA UID namespace, allocates
the private prefill UID buffer and scratch inventory, installs the event
completion backend and the typed worker RPC adapter, and installs B's graph
builder before the engine's graph-capture call.

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

from async_handoff import (
    CommitAck,
    CompletionReport,
    HandoffConfig,
    HandoffError,
    PollOperations,
    RuntimeWorkerHandoff,
)
from e3_types import E3ContractError, E3Pending
from graph_router import (
    CompilationIdentity,
    ExecutionContext,
    GraphRouter,
    GraphRoutingError,
)
from lane_hooks import LaneRuntime
from lane_state import (
    GRAPH_POOL_MEASURED_BYTES,
    WorkerState,
    prefill_scratch_specs,
)
from native_backend import (
    EventCompletionBackend,
    NativeDevice,
    NativeGraphCaptureBackend,
)
from qsa_operation import (
    RUNNER_SEAM,
    NativeQsaDevice,
    OperationError,
    QsaPrefillOperation,
    metadata_plan_provider,
)

ENGINE_EPOCH = 1
PREFLIGHT_CONTEXT_ID = 1
DECODE_CONTEXT_ID = 1          # B requires the split context to equal A's lane.
FULL_CONTEXT_ID = 0            # placeholder; see Open items.
FULL_POOL_ID = 0               # placeholder; see Open items.
SPLIT_POOL_ID = 1
SPLIT_RESERVED_BYTES = GRAPH_POOL_MEASURED_BYTES
WORKSPACE_LANE = 1


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


def _forward_context_module():
    module = importlib.import_module('vllm.forward_context')
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


def _compilation_identity(runner):
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
                 operation_device=None, clock=time.monotonic):
        if not isinstance(runtime, LaneRuntime):
            raise NativeResourcesError('worker server needs A LaneRuntime')
        self.runtime = runtime
        self.bindings = tuple(bindings)
        self.private_attrs = dict(private_attrs or {})
        self.operation_device = operation_device
        self.clock = clock
        self.worker = None
        self.handoff = None
        self.operation = None
        self.last_output = None
        self.started_s = clock()
        self.executions = 0
        self._operation_layers = ()
        self._deferred_proofs = []

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
        try:
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


# --- graph builder ----------------------------------------------------------

def _reserve_decode_anchor(state):
    """Give B's router the stable decode lane handle it requires.

    A reserves a lane per transaction; B captures once and needs the handle to
    stay alive.  The anchor is an explicit startup identity, listed in Open
    items because per-request decode lanes are not yet reconciled with it.
    """
    if state.decode.active:
        return state.decode.handle
    transaction = state.new_transaction('__flash_s5_decode_anchor__', 0, 0, 1)
    handle = state.decode.reserve(transaction, context_id=DECODE_CONTEXT_ID)
    state.decode.ready()
    return handle


def _split_descriptors(runner):
    value = getattr(runner, '_flash_s5_split_descriptors', ()) or ()
    return tuple(value)


def install_graph_builder(runner, runtime, device, *, config, clock=time.monotonic):
    """Install the graph-builder scaffold, never a native qualified split."""
    from s5_startup import require_native_ready
    require_native_ready()
    state = runtime.state

    def build(current):
        if getattr(current, '_flash_e3_graph_builder_used', False):
            raise GraphRoutingError('E3 graph builder was already used')
        capture = getattr(current, '_flash_s5_graph_capture', None)
        backend = NativeGraphCaptureBackend(device, capture_callable=capture,
                                            clock=clock)
        handle = _reserve_decode_anchor(state)
        compilation = _compilation_identity(current)
        router = GraphRouter(
            state, handle, compilation, backend=backend,
            full_context_id=FULL_CONTEXT_ID, full_pool_id=FULL_POOL_ID,
            full_allocated_bytes=GRAPH_POOL_MEASURED_BYTES, full_basis='measured')
        stream_token = device.current_stream_token()
        setattr(router, 'probe',
                lambda: (handle.context_id, device.current_stream_token()))
        descriptors = _split_descriptors(current)
        if descriptors:
            context = ExecutionContext(
                'decode-split', handle.context_id, requested_sm=device.sm_count(),
                measured_sm=device.sm_count(), pool_id=SPLIT_POOL_ID,
                stream_token=stream_token, reserved_bytes=SPLIT_RESERVED_BYTES,
                basis='estimate')
            router.add_bank('decode-split', context, descriptors)
        current._flash_e3_graph_builder_used = True
        current._flash_e3_graph_builder_stream = stream_token
        return router

    runner._flash_e3_graph_builder = build
    return build


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
    state = WorkerState(rank, engine_epoch=ENGINE_EPOCH, weights=model,
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
    operation_device = NativeQsaDevice(_runner_device(runner), clock=clock)
    server = WorkerHandoffServer(runtime, bindings=bindings,
                                 operation_device=operation_device, clock=clock)

    runner._flash_s5_worker = server
    runner._flash_s5_native = {
        'rank': rank,
        'engine_epoch': ENGINE_EPOCH,
        'uid_bytes': uid_buffer.nbytes,
        'scratch': scratch_inventory,
        'workspace_lane': WORKSPACE_LANE,
        'weights_bytes': weights_bytes,
        'operation': 'qsa_dense_bounded',
        'native_qualified': False,
    }
    install_graph_builder(runner, runtime, device, config=config, clock=clock)
    return runtime


__all__ = [
    'DECODE_CONTEXT_ID', 'ENGINE_EPOCH', 'FULL_CONTEXT_ID', 'FULL_POOL_ID',
    'NativeResourcesError', 'PREFLIGHT_CONTEXT_ID', 'SPLIT_POOL_ID',
    'WORKSPACE_LANE', 'WorkerHandoffServer', 'build_worker_runtime',
    'install_graph_builder', 'response_success',
]
