"""E3 worker-native startup composition and the transition RPC (D4).

Called once per GPU worker from the runner's E3 startup, after the stock
cudagraph manager exists and before capture. It needs the green split created
at worker init (``worker_hooks.after_runner_init``) and composes, in order:

1. the split decode bank objects (capture happens later through B's
   ``capture_banks`` seam);
2. the prefill-lane communicators (PyNccl TP, tagged PP channel), warmed in a
   fixed rank order on the prefill stream;
3. the lane-private PLE prefetch scratch for the one PLE layer on this stage;
4. a ``LaneForward`` factory for the worker operation server;
5. the graph-router builder that registers the split bank with B.

``transition_rpc`` is the worker side of ``flash_s5_transition``: it validates
the engine's command, performs the drained stream/bank switch through the
``RouteController`` and acknowledges only afterwards with the new route
generation. No per-step ticket.

Everything here records bytes it allocated (PP buffers, PLE scratch, measured
split pool after capture) as an inventory; nothing claims funding.
"""
import time

from e3_types import E3ContractError
from lane_comm import PrefillCommunicators
from lane_hooks import LaneRuntime
from lane_private import LanePrivateBindings, admitted_chunk
from model_hooks import LaneForward, PlePrivateScratch, PrefillStreamOwners, ple_embedding
from native_resources import compilation_identity, full_graph_identity
from split_bank import install as install_split_bank

SPLIT_BANK_NAME = 'decode-split'
FULL_ROUTE = 'full'
SPLIT_ROUTE = 'split'


class NativeStartupError(E3ContractError):
    pass


def _torch():
    import torch

    return torch


def _groups(tp_group, pp_group):
    if tp_group is not None and pp_group is not None:
        return tp_group, pp_group
    from vllm.distributed.parallel_state import get_pp_group, get_tp_group

    return (tp_group if tp_group is not None else get_tp_group(),
            pp_group if pp_group is not None else get_pp_group())


def _model_dtype(runner, model):
    config = getattr(getattr(runner, 'vllm_config', None), 'model_config', None)
    dtype = getattr(config, 'dtype', None)
    if dtype is not None:
        return dtype
    parameters = getattr(model, 'parameters', None)
    if callable(parameters):
        for parameter in parameters():
            return parameter.dtype
    raise NativeStartupError('cannot discover the model dtype for the PP channel')


def _hc_width(model):
    config = getattr(model, 'config', None)
    hc = getattr(config, 'hc_count', None)
    hidden = getattr(config, 'hidden_size', None)
    if type(hc) is not int or type(hidden) is not int or hc < 1 or hidden < 1:
        raise NativeStartupError('model config lacks hc_count/hidden_size for the PP channel')
    return hc * hidden


def _ple_scratch(model, chunk_tokens, torch, device, owners):
    found = []
    for layer_idx in range(int(model.start_layer), int(model.end_layer)):
        embedding = ple_embedding(model, layer_idx)
        if embedding is not None:
            found.append((layer_idx, embedding))
    if not found:
        return None, None
    if len(found) > 1:
        raise NativeStartupError('more than one PLE layer on this stage is not supported')
    layer_idx, embedding = found[0]
    try:
        stream = owners.replacement(embedding, '_prefetch_stream')
    except Exception as exc:
        # No primary-context fallback: the native lane refuses until the PLE
        # prefetch stream is owned by the prefill green context.
        raise NativeStartupError(
            'PLE prefetch stream is not owned by the prefill green context') from exc
    return layer_idx, PlePrivateScratch(embedding, chunk_tokens, torch_module=torch,
                                        device=device, stream=stream)


class WorkerNative:
    """Everything the lane needs on one worker, with its byte inventory."""

    def __init__(self, runner, split, bank, backend, controller, comms, ple_layer, ple_scratch,
                 chunk_tokens, tp_group, torch, *, state=None, compilation=None,
                 full_identity=None, decode_owner=None, prefill_streams=None,
                 private_bindings=None):
        self.runner = runner
        self.split = split
        self.bank = bank
        self.backend = backend
        self.controller = controller
        self.comms = comms
        self.ple_layer = ple_layer
        self.ple_scratch = ple_scratch
        self.chunk_tokens = chunk_tokens
        self.tp_group = tp_group
        self.torch = torch
        self.state = state
        self.compilation = compilation
        self.full_identity = full_identity
        self.decode_owner = decode_owner
        self.prefill_streams = prefill_streams
        self.private_bindings = private_bindings
        self.observation = None
        self.router = None
        self.host_submit_ns = []

    def drain_prefill_streams(self):
        """Drain every owned prefill side stream; returns the count."""
        if self.prefill_streams is None:
            return 0
        return self.prefill_streams.drain()

    def lane_forward(self, inputs, *, forward_context=None, attention_driver=None, clock=None):
        kwargs = dict(stream=self.split.prefill_stream, tp_group=self.tp_group,
                      tp_comm=self.comms.tp, forward_context=forward_context,
                      ple_scratch=self.ple_scratch, attention_driver=attention_driver,
                      torch_module=self.torch, prefill_owners=self.prefill_streams)
        if self.private_bindings is not None:
            # The admitted chunk is the authority for the request UID and the
            # chunk's top-k capacity; a generic callable is refused.
            chunk = admitted_chunk(forward_context, runner=self.runner)
            kwargs['private_scope'] = self.private_bindings.chunk_scope(chunk).scope
        if clock is not None:
            kwargs['clock'] = clock
        forward = LaneForward(self.runner.model, inputs, **kwargs)
        self.observe_prefill(forward, 'lane_forward')
        return forward

    def observe_prefill(self, target, kind):
        """Attach a source observer only inside its explicitly owned window."""
        if self.observation is not None:
            self.observation.attach(target, kind)
        return target

    def before_route_change(self, target):
        """Retire any owned observation before a drained FULL handback."""
        owner = getattr(self.runner, '_flash_s5_replay_observer', None)
        if target == FULL_ROUTE and owner is not None:
            owner.before_native_full_return(self)

    def inventory(self):
        record = {
            'split': self.split.describe(),
            'split_bank_bytes': self.bank.measured_bytes,
            'split_bank_basis': 'measured' if self.bank.measured_bytes is not None else 'uncaptured',
            'pp_channel_bytes': self.comms.pp.bytes_owned() if self.comms.pp is not None else 0,
            'ple_scratch_bytes': self.ple_scratch.bytes if self.ple_scratch is not None else 0,
            'ple_layer': self.ple_layer,
            'prefill_streams': (self.prefill_streams.record()
                                if self.prefill_streams is not None else None),
            'private_bindings': (self.private_bindings.audit()
                                 if self.private_bindings is not None else None),
            'communicators': self.comms.describe(),
            'chunk_tokens': self.chunk_tokens,
            'route': self.controller.route,
            'route_generation': self.controller.route_generation,
        }
        return record

    def graph_builder(self, *, router_class=None, full_context_id, full_pool_id,
                      full_allocated_bytes, full_basis='estimate', engine_epoch=None):
        """B's ``runner._flash_e3_graph_builder``: register the split bank once."""
        if router_class is None:
            from graph_router import GraphRouter as router_class

        def build(current):
            if getattr(current, '_flash_e3_graph_builder_used', False):
                raise NativeStartupError('E3 graph builder was already used')
            state = self.state
            if state is None:
                server = getattr(current, '_flash_s5_worker', None)
                runtime = getattr(server, 'runtime', None)
                state = getattr(runtime, 'state', None)
            handle = getattr(getattr(state, 'decode', None), 'handle', None)
            owner = getattr(state, 'decode_owner', None)
            if state is None or handle is None or owner is None:
                raise NativeStartupError(
                    'graph builder needs the retained decode storage owner')
            if handle.lane_id != owner.lane_id:
                raise NativeStartupError('decode handle is not the retained storage owner')
            if state.decode.state != 'retained':
                raise NativeStartupError('decode storage is not persistently retained')
            compilation = self.compilation
            if compilation is None:
                raise NativeStartupError('graph builder needs the compilation identity')
            router = router_class(state, handle, compilation, engine_epoch=engine_epoch,
                                  backend=self.backend, full_context_id=full_context_id,
                                  full_pool_id=full_pool_id,
                                  full_allocated_bytes=full_allocated_bytes,
                                  full_basis=full_basis)
            # The pre-capture context carries the real measured SM count, pool
            # and stream identity; the factory refreshes only byte totals after
            # capture and must agree on every identity field.
            context = self.bank.execution_context(SPLIT_BANK_NAME)
            router.add_bank(SPLIT_BANK_NAME, context, self.bank.descriptors(),
                            context_factory=lambda: self.bank.execution_context(
                                SPLIT_BANK_NAME))
            split = self.split
            torch = self.torch

            def probe():
                stream = torch.cuda.current_stream(self.runner.device)
                return (split.decode_context_id, int(getattr(stream, 'cuda_stream', -1)))

            router.probe = probe
            self.router = router
            current._flash_e3_graph_builder_used = True
            current._flash_e3_graph_router_split = self
            return router

        return build


def install_worker_native(runner, *, chunk_tokens, env=None, torch_module=None,
                          pynccl_class=None, set_graph_pool_id=None, tp_group=None,
                          pp_group=None, clock=time.monotonic):
    """Compose the worker-native lane objects on a runner with its stock manager."""
    if type(chunk_tokens) is not int or chunk_tokens < 1:
        raise NativeStartupError('chunk_tokens must be a positive int')
    if getattr(runner, '_flash_s5_native', None) is not None and isinstance(
            runner._flash_s5_native, WorkerNative):
        raise NativeStartupError('worker-native lane already installed')
    split = getattr(runner, '_flash_s5_split', None)
    if split is None or getattr(split, 'state', None) != 'open':
        raise NativeStartupError('worker-native startup needs the open green split from worker init')
    torch = torch_module if torch_module is not None else _torch()
    model = getattr(runner, 'model', None)
    if model is None:
        raise NativeStartupError('runner has no model')
    tp_group, pp_group = _groups(tp_group, pp_group)

    bank, backend, controller = install_split_bank(
        runner, split, torch_module=torch, set_graph_pool_id=set_graph_pool_id, clock=clock)
    server = getattr(runner, '_flash_s5_worker', None)
    runtime = getattr(server, 'runtime', None)
    if not isinstance(runtime, LaneRuntime):
        raise NativeStartupError('worker-native startup needs the real LaneRuntime')
    state = getattr(runtime, 'state', None)
    if state is None:
        raise NativeStartupError(
            'worker-native startup needs the worker runtime state from build_worker_runtime')
    # Persistent decode storage: one owner for the worker incarnation, outside
    # every prefill/request transaction. Captured graphs stay bound to it.
    decode_owner = state.retain_decode_storage(context_id=split.decode_context_id)
    if decode_owner.context_id != split.decode_context_id:
        raise NativeStartupError('retained decode context differs from the green split')
    compilation = compilation_identity(runner)
    full_identity = full_graph_identity(runner)
    # Exact prefill side-stream owners this stage. Discovery is cheap and
    # fails before any native resource is created; the auxiliary streams are
    # created after the comms warm-up below.
    try:
        prefill_streams = PrefillStreamOwners.discover(runner, split)
    except Exception as exc:
        raise NativeStartupError(
            'prefill side-stream discovery failed: ' + type(exc).__name__) from exc
    comms = PrefillCommunicators.build(
        tp_group, pp_group, runner.device, split.prefill_stream, max_tokens=chunk_tokens,
        hidden_width=_hc_width(model), dtype=_model_dtype(runner, model),
        pynccl_class=pynccl_class, torch_module=torch)
    comms.warm(split.prefill_stream)
    # Distinct prefill-green auxiliaries, retained for the worker incarnation.
    # A helper without the prefill extension refuses here (no fallback).
    prefill_streams.prepare()
    ple_layer, ple_scratch = _ple_scratch(model, chunk_tokens, torch, runner.device,
                                          prefill_streams)
    private_bindings = LanePrivateBindings.discover(
        runner, runtime, chunk_tokens=chunk_tokens, torch_module=torch,
        ple_scratch=ple_scratch)
    native = WorkerNative(runner, split, bank, backend, controller, comms, ple_layer, ple_scratch,
                          chunk_tokens, tp_group, torch, state=state,
                          compilation=compilation, full_identity=full_identity,
                          decode_owner=decode_owner, prefill_streams=prefill_streams,
                          private_bindings=private_bindings)
    runner._flash_s5_native = native
    runner._flash_s5_compilation = compilation
    runner._flash_s5_comms = comms
    runner._flash_s5_ple_scratch = ple_scratch
    runner._flash_s5_lane_forward = native.lane_forward
    runner._flash_s5_inventory = native.inventory
    # The generated A wrapper may hand an ordinary decode step (no request
    # transaction) to B only after this real retained-storage composition.
    runner._flash_s5_native_decode = True
    # Install the real split-bank builder once; never overwrite another one.
    if getattr(runner, '_flash_e3_graph_builder', None) is not None:
        raise NativeStartupError('runner already carries an E3 graph builder')
    runner._flash_e3_graph_builder = native.graph_builder(
        full_context_id=full_identity.context_id,
        full_pool_id=full_identity.pool_id,
        full_allocated_bytes=full_identity.allocated_bytes,
        full_basis=full_identity.basis)
    return native


# --- transition RPC ---------------------------------------------------------------

_REQUIRED_COMMAND = ('engine_epoch', 'expected_route_generation', 'route_generation', 'target')


def transition_rpc(runner, command, *, engine_epoch=None, clock=time.monotonic):
    """Worker side of ``flash_s5_transition``; ACK only after the drained switch."""
    native = getattr(runner, '_flash_s5_native', None)
    if not isinstance(native, WorkerNative):
        raise NativeStartupError('transition before worker-native startup')
    if not isinstance(command, dict):
        raise NativeStartupError('transition command must be a mapping')
    for key in _REQUIRED_COMMAND:
        if key not in command:
            raise NativeStartupError(f'transition command lacks {key!r}')
    if engine_epoch is not None and command['engine_epoch'] != engine_epoch:
        raise NativeStartupError('transition command names another engine epoch')
    controller = native.controller
    if command['expected_route_generation'] != controller.route_generation:
        raise NativeStartupError('transition command expects another route generation')
    if command['route_generation'] != controller.route_generation + 1:
        raise NativeStartupError('transition command must advance the route generation by one')
    target = command['target']
    if target == SPLIT_BANK_NAME:
        target = SPLIT_ROUTE
    if target not in (FULL_ROUTE, SPLIT_ROUTE):
        raise NativeStartupError(f'unknown transition target {target!r}')
    storage_epoch = command.get('storage_epoch')
    lane = getattr(runner, '_flash_e3_lane', None)
    handle = getattr(getattr(getattr(lane, 'state', None), 'decode', None), 'handle', None)
    if storage_epoch is not None and handle is not None and storage_epoch != handle.storage_epoch:
        raise NativeStartupError('transition command names a stale storage epoch')
    started = clock()
    native.before_route_change(target)
    record = controller.enter_split() if target == SPLIT_ROUTE else controller.leave_split()
    router = getattr(runner, '_flash_e3_graph_router', None)
    authorization = command.get('authorization')
    if router is not None and authorization is not None:
        router.transition(SPLIT_BANK_NAME if target == SPLIT_ROUTE else 'ordinary', authorization)
    return {
        'ack': 'flash_s5_transition',
        'route': controller.route,
        'route_generation': controller.route_generation,
        'drained_streams': record.drained_streams,
        'switch_seconds': record.seconds,
        'ack_seconds': clock() - started,
        'decode_replay_watermark': command.get('decode_replay_watermark'),
        'storage_epoch': storage_epoch,
    }


# --- decode-only diagnostic startup ----------------------------------------------

_PROBE_FORBIDDEN = ('_flash_s5_native', '_flash_s5_comms', '_flash_s5_ple_scratch',
                    '_flash_s5_lane_forward', '_flash_s5_dispatcher', '_flash_s5_worker',
                    '_flash_e3_lane', '_flash_e3_build_state')


def install_decode_probe(runner, *, env=None, torch_module=None, set_graph_pool_id=None,
                         clock=time.monotonic, evidence=None):
    """Compose ONLY the green split, the split bank and the drained controller.

    Sibling of :func:`install_worker_native` for the decode-only diagnostic
    mode: no prefill communicator, PLE scratch, lane forward or dispatcher is
    constructed.  The probe flag/overlap conflict and the declared SM count are
    validated before the first allocation, and the diagnostic gate is checked
    against this runtime's pins, helper source and generated bytes.
    """
    import os

    from decode_probe import DecodeProbe, mode_settings, require_decode_probe_ready
    from graph_router import DecodeProbeRouter

    env = os.environ if env is None else env
    settings = mode_settings(env)
    if settings is None:
        raise NativeStartupError('decode probe startup needs ' + 'FLASH_S5_DECODE_ONLY_PROBE=1')
    require_decode_probe_ready(evidence)
    for name in _PROBE_FORBIDDEN:
        if getattr(runner, name, None) is not None:
            raise NativeStartupError('decode probe must not carry the lane object ' + name)
    if getattr(runner, '_flash_s5_decode_probe', None) is not None:
        raise NativeStartupError('decode probe already installed')
    split = getattr(runner, '_flash_s5_split', None)
    if split is None or getattr(split, 'state', None) != 'open':
        raise NativeStartupError('decode probe needs the open green split from worker init')
    declared_lib = env.get('FLASH_S5_GREEN_LIB_SHA256')
    helper_identity = split.binding.identity()
    if (not isinstance(declared_lib, str) or len(declared_lib) != 64
            or not isinstance(helper_identity, dict) or helper_identity.get('abi') != 1
            or declared_lib != helper_identity.get('library_sha256')):
        raise NativeStartupError('compiled helper library does not match declared sha256')
    if split.requested_decode_sms != settings.decode_sms:
        raise NativeStartupError('green split was not created with the declared decode SM count')
    torch = torch_module if torch_module is not None else _torch()
    if set_graph_pool_id is None and torch_module is None:
        from vllm.distributed.device_communicators import pynccl_allocator
        previous_pool = pynccl_allocator._graph_pool_id
        set_graph_pool_id = lambda _stock_pool: pynccl_allocator.set_graph_pool_id(previous_pool)
    bank, backend, controller = install_split_bank(
        runner, split, torch_module=torch, set_graph_pool_id=set_graph_pool_id, clock=clock)
    context = bank.execution_context(SPLIT_BANK_NAME)
    router = DecodeProbeRouter(SPLIT_BANK_NAME, context, bank.descriptors(), backend,
                               storage_epoch=settings.storage_epoch,
                               route_generation=0,
                               context_factory=lambda: bank.execution_context(SPLIT_BANK_NAME))
    probe = DecodeProbe(runner, split, bank, backend, controller, router, settings,
                        clock=clock, torch_module=torch)
    runner._flash_s5_decode_probe = probe
    runner._flash_s5_decode_probe_counters = probe.counters
    runner._flash_s5_decode_probe_events = probe.events
    runner._flash_s5_inventory = probe.inventory
    runner._flash_e3_graph_router = router
    try:
        probe.capture()
        runner._flash_s5_decode_probe_inventory = probe.inventory()
    except BaseException:
        probe.quarantined = 'capture failed; retained ownership pending drained shutdown'
        raise
    return probe


def probe_transition_rpc(runner, command, *, engine_epoch=None, clock=time.monotonic):
    """Worker side of ``flash_s5_transition`` for the decode-only diagnostic."""
    from decode_probe import DecodeProbe

    probe = getattr(runner, '_flash_s5_decode_probe', None)
    if not isinstance(probe, DecodeProbe):
        raise NativeStartupError('transition before decode probe startup')
    if not isinstance(command, dict):
        raise NativeStartupError('transition command must be a mapping')
    for key in _REQUIRED_COMMAND:
        if key not in command:
            raise NativeStartupError(f'transition command lacks {key!r}')
    for field in ('engine_epoch', 'expected_route_generation', 'route_generation', 'storage_epoch'):
        if type(command.get(field)) is not int or command[field] < (0 if field == 'expected_route_generation' else 1):
            raise NativeStartupError('transition integer field invalid: ' + field)
    if engine_epoch is not None and command['engine_epoch'] != engine_epoch:
        raise NativeStartupError('transition command names another engine epoch')
    if command['expected_route_generation'] != probe.route_generation:
        raise NativeStartupError('transition command expects another route generation')
    if command['route_generation'] != probe.route_generation + 1:
        raise NativeStartupError('transition command must advance the route generation by one')
    target = command['target']
    if target == SPLIT_BANK_NAME:
        target = SPLIT_ROUTE
    if target not in (FULL_ROUTE, SPLIT_ROUTE):
        raise NativeStartupError(f'unknown transition target {target!r}')
    storage_epoch = command.get('storage_epoch')
    if storage_epoch is not None and storage_epoch != probe.storage_epoch:
        raise NativeStartupError('transition command names a stale storage epoch')
    if command.get('authorization') is not None:
        raise NativeStartupError('decode probe has no prefill-transition authorization')
    started = clock()
    record = probe.transition(target)
    return {
        'ack': 'flash_s5_transition',
        'mode': 'decode-only-probe',
        'route': probe.route,
        'route_generation': probe.route_generation,
        'drained_streams': record.drained_streams,
        'switch_seconds': record.seconds,
        'ack_seconds': clock() - started,
        'storage_epoch': probe.storage_epoch,
    }


__all__ = ['FULL_ROUTE', 'NativeStartupError', 'SPLIT_BANK_NAME', 'SPLIT_ROUTE', 'WorkerNative',
           'install_decode_probe', 'install_worker_native', 'probe_transition_rpc',
           'transition_rpc']
