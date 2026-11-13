"""C startup: native worker construction and engine-side composition.

Enabled startup requires a reviewed immutable source/build closure before
allocation or transport ownership. Missing closure still refuses. Source
construction and hardware qualification are separate; the owner queue grades
exactness, numerics, placement, performance and lifetime on the actual engine.
Two different processes, two different startup paths:

* :func:`install_builder` is called from the generated model runner at the end
  of construction, immediately before A's ``init_lane``. Its builder calls
  :func:`native_resources.build_worker_runtime`, which allocates this worker's
  private UID buffer and scratch, discovers the device/rank/weights, installs
  the device-event completion backend and the typed worker RPC adapter, and
  installs B's graph builder before the engine captures graphs. There is no
  injected factory: the worker constructs its own objects from the real seam.
* :func:`install_engine` is called from the generated ``EngineCore.__init__``.
  It takes sole reply-transport ownership on the executor, installs the
  engine-side relay backend, builds the composition over the TP ranks (worker
  lane identities are bound later from each worker's open acknowledgment),
  attaches it to the engine/scheduler/executor, and installs the segment
  driver that park/advance call from the generated step seams.

Nothing here is imported by torch/vLLM at module load time; the native pieces
import lazily. Both paths are inert unless ``FLASH_S5_OVERLAP=1``.
"""
import os
import time

from async_handoff import (
    Composition,
    HandoffConfig,
    HandoffError,
    build_composition,
)
from decode_probe import mode_settings, require_decode_probe_ready
from lane_hooks import LaneRuntime
from lane_state import estimated_e3_inventory
from scheduler_hooks import SchedulerLeases

OVERLAP_ENV = 'FLASH_S5_OVERLAP'
_BUILDER_ATTR = '_flash_e3_build_state'
_COMPOSITION_ATTR = '_flash_s5_composition'


class StartupError(HandoffError):
    pass


def require_native_ready():
    """Require the reviewed immutable implementation/build closure, no bypass.

    The returned capability is for the owner's candidate queue only. Hardware
    exactness/numerics/placement/performance/lifetime are graded by that queue,
    never asserted by this startup source check.
    """
    from native_readiness import require_implementation_closure, NativeReadinessError
    try:
        return require_implementation_closure()
    except (NativeReadinessError, OSError, ValueError, TypeError, KeyError) as exc:
        raise StartupError(str(exc)) from None


def _enabled(env):
    value = env.get(OVERLAP_ENV, '0')
    if value not in ('0', '1'):
        raise StartupError('invalid boolean setting: ' + OVERLAP_ENV)
    return value == '1'


def row1_qualification_types():
    """Actual provider import path, callable without constructing GPU owners.

    Packaging binds the complete native row1 backend to its explicit E3 owner.
    The live provider invokes this only after its existing pause/drain fences.
    """
    from row1_exactness import TorchRow1Backend
    from row1_oracle import Row1Qualifier
    return TorchRow1Backend, Row1Qualifier


def install_builder(runner, kv_caches, is_profiling=False, *, env=None):
    """Install the builder A's ``init_lane`` calls at the end of runner init."""
    env = os.environ if env is None else env
    if not is_profiling and env.get('FLASH_ATTN_META_GDN_BATCH', '0') != '0':
        # attn-meta (default off): its instance wrapper goes on first, before any RPC (handler anatomy installs at the
        # first RPC and wraps it); a refusal fails startup, never a skipped wrapper. Unset or 0: no import.
        from attn_meta_select import startup as _attn_meta_startup
        _attn_meta_startup(runner, env=env)
    probe = mode_settings(env)
    if probe is not None:
        # Diagnostic mode: the split (worker_hooks) and later the split bank
        # (native_startup.install_decode_probe) are the whole construction.
        require_decode_probe_ready()
        if not is_profiling:
            from split_bank import full_descriptors
            full_descriptors(runner.cudagraph_manager)
        return None
    if is_profiling or not _enabled(env):
        return None
    require_native_ready()
    if getattr(runner, _BUILDER_ATTR, None) is not None:
        raise StartupError('E3 lane builder is already installed')

    def build(current, caches):
        if getattr(current, _BUILDER_ATTR, None) is not build:
            raise StartupError('E3 lane builder was replaced during startup')
        from native_resources import build_worker_runtime

        runtime = build_worker_runtime(current, caches, env=env)
        if not isinstance(runtime, LaneRuntime):
            raise StartupError('native worker construction returned no LaneRuntime')
        install_native_lane(current, env=env)
        return runtime

    runner._flash_e3_build_state = build
    return build


def _native_execution_shape(engine, ranks, *, env=None):
    """Use EngineCore.vllm_config and the declared PP partition, never guess."""
    env = os.environ if env is None else env
    config = getattr(engine, 'vllm_config', None)
    model_config = getattr(config, 'model_config', None)
    hf_config = getattr(model_config, 'hf_text_config', None)
    if hf_config is None:
        hf_config = getattr(model_config, 'hf_config', None)
    layer_count = getattr(hf_config, 'num_hidden_layers', None)
    parallel = getattr(config, 'parallel_config', None)
    pp_size = getattr(parallel, 'pipeline_parallel_size', None)
    partition = env.get('VLLM_PP_LAYER_PARTITION', '')
    if type(layer_count) is not int or type(pp_size) is not int or not partition:
        return None
    try:
        parts = tuple(int(v) for v in partition.split(','))
    except ValueError:
        raise StartupError('invalid declared PP layer partition') from None
    if (pp_size != 2 or len(parts) != 2 or min(parts) < 1
            or sum(parts) != layer_count or len(ranks) % 2):
        raise StartupError('declared model shape is not the supported two-stage split')
    if ranks != tuple(range(len(ranks))):
        raise StartupError('rank map must use the pinned executor order')
    per_stage = len(ranks) // 2
    return layer_count, parts[0], tuple((rank, rank // per_stage) for rank in ranks)


def install_native_lane(runner, *, env=None, clock=time.monotonic):
    """D4: split bank, lane communicators, chunk factory and segment dispatcher.

    Runs after D's worker runtime/server exist and before graph capture. The
    green split itself was created at worker init (``worker_hooks``); the QSA
    operation device gets its compute/copy streams bound here; the dispatcher
    replaces the refusing whole-forward seam for typed segments.
    """
    env = os.environ if env is None else env
    from native_startup import install_worker_native
    from lane_chunk import LaneChunkFactory
    from lane_dispatch import LaneDispatcher
    from native_measurement import install_receipt_writer

    # Alpha-next PLAN 2: every receipt of this worker process goes out through
    # one atomic write per record. Nothing is opened or written until a receipt.
    install_receipt_writer()

    config = HandoffConfig.from_env(env)
    native = install_worker_native(runner, chunk_tokens=config.chunk_tokens, env=env, clock=clock)
    server = getattr(runner, '_flash_s5_worker', None)
    if server is None:
        raise StartupError('native lane needs the worker handoff server')
    device = getattr(server, 'operation_device', None)
    if device is not None and callable(getattr(device, 'bind_streams', None)):
        import torch

        copy_stream = torch.cuda.Stream(device=runner.device)
        native.copy_stream = copy_stream
        device.bind_streams(native.split.prefill_stream, copy_stream)

    def driver_factory(operation, qsa_module, forward_context_getter):
        from qsa_driver import QsaLayerDriver

        native.observe_prefill(operation, 'qsa_operation')
        driver = QsaLayerDriver(
            operation,
            staged_operation=qsa_module._flash_s5_staged_operation,
            sparse_staged_operation=getattr(
                qsa_module, '_flash_s5_sparse_staged_operation', None),
            source_holds=getattr(operation, 'native_source_holds', None),
            forward_context=forward_context_getter)
        return native.observe_prefill(driver, 'qsa_driver')

    def qsa_module():
        import vllm.models.qwen4_exp.nvidia.qsa as module

        return module

    def forward_context_getter():
        from vllm.forward_context import get_forward_context

        return get_forward_context()

    # The oracle is always OFF at startup. Only the paused/drained native
    # qualification action can install a fresh run-bound store and factory.
    if env.get('FLASH_S5_ROW1_QUALIFY', '0') != '0':
        raise StartupError('row1 qualification requires scoped native control')

    def row1_provider(store, engine_epoch, route_generation):
        state = server.runtime.state
        controller = native.controller
        if (not state.engine_bound or state.engine_epoch != engine_epoch
                or controller.route_generation != route_generation):
            raise StartupError('row1 qualification needs the current engine and route')
        if store is not None and controller.route != 'split':
            raise StartupError('row1 qualification enable needs the split route')
        # Collective pause is the action owner's responsibility. This local
        # fence independently queries every actual decode/prefill consumer.
        drained = controller.drain()
        if type(drained) is not int or drained <= 0:
            raise StartupError('row1 qualification decode drain is unproved')
        if native.prefill_streams is None:
            raise StartupError('row1 qualification prefill owners are unavailable')
        drained = native.drain_prefill_streams()
        if type(drained) is not int or drained < 0:
            raise StartupError('row1 qualification prefill drain is unproved')
        from native_resources import _qsa_layers
        layers = tuple(name for name, _ in _qsa_layers(runner))
        if store is None:
            return None, layers
        TorchRow1Backend, Row1Qualifier = row1_qualification_types()

        def row1_factory(operation):
            def epochs():
                if (state.engine_epoch != engine_epoch
                        or controller.route_generation != route_generation):
                    raise StartupError('row1 scope changed during qualification')
                return dict(worker_incarnation=state.incarnation,
                            engine_epoch=engine_epoch,
                            route_generation=route_generation)
            return Row1Qualifier(
                store, transaction=operation.transaction,
                lane=operation.lane, backend=TorchRow1Backend(), epochs=epochs)
        return row1_factory, layers

    server.row1_provider = row1_provider

    from lane_chunk import PREPARE_AHEAD, prepare_ahead_from_env
    ahead = prepare_ahead_from_env(env)
    PREPARE_AHEAD['selected'] = ahead
    # Off: the same call as before build-2, so the factory and every fake of it see today's arguments.
    factory = (LaneChunkFactory(runner, chunk_tokens=config.chunk_tokens, prepare_ahead=True) if ahead
               else LaneChunkFactory(runner, chunk_tokens=config.chunk_tokens))
    # One per-worker private metadata generation, allocated once at startup.
    # Its byte count and full builder-field classification are included in
    # the existing native inventory/ledger; no per-chunk storage is retained.
    native.native_builder_inventory = factory.inventory
    dispatcher = LaneDispatcher(runner, native, factory, server=server, qsa_module=qsa_module(),
                                driver_factory=driver_factory,
                                forward_context_getter=forward_context_getter)
    server._flash_s5_dispatcher = dispatcher
    runner._flash_s5_dispatcher = dispatcher
    # The typed wire-v1 RPCs route to this bridge, which is the only adapter
    # over the constructed dispatcher. It allocates no storage of its own.
    from worker_bridge import NativeWorkerBridge

    bridge = NativeWorkerBridge(dispatcher, server=server)
    server._flash_s5_bridge = bridge
    runner._flash_s5_bridge = bridge
    from native_measurement import progress_mode
    try:
        progress = progress_mode(env)
    except ValueError as exc:
        raise StartupError(str(exc)) from None
    if progress is not None:
        # Held-g R1: device spans of this worker's lane commands (receipts only).
        dispatcher.spans = _device_spans(bridge.rank)
    return dispatcher


def _device_spans(rank):
    import torch
    from executor_hooks import _progress_logger
    from native_measurement import DeviceSpans, emit_receipt_lines

    return DeviceSpans(lambda: torch.cuda.Event(enable_timing=True), rank=rank,
                       emit=lambda text: _progress_logger().info(text),
                       emit_lines=lambda texts: emit_receipt_lines(texts, _progress_logger()))


def install_engine(engine, *, env=None, clock=time.monotonic):
    """Install the engine-side composition at the generated ``__init__`` seam."""
    env = os.environ if env is None else env
    auto = env.get('FLASH_S5_AUTO_ARM', '0')
    control_inwait = env.get('FLASH_S5_CONTROL_INWAIT', '0')
    if control_inwait not in ('0', '1') or (control_inwait == '1' and auto != '1'):
        raise StartupError('FLASH_S5_CONTROL_INWAIT needs boolean setting and FLASH_S5_AUTO_ARM=1')
    control_o3 = env.get('FLASH_S5_CONTROL_O3_INWAIT', '0')
    if control_o3 not in ('0', '1') or (control_o3 == '1' and
            (control_inwait != '1' or auto != '1' or env.get('FLASH_S5_O3', '0') != '1')):
        raise StartupError('FLASH_S5_CONTROL_O3_INWAIT needs CONTROL_INWAIT=1, AUTO_ARM=1 and O3=1')
    if auto not in ('0', '1'):
        raise StartupError('invalid boolean setting: FLASH_S5_AUTO_ARM')
    if auto == '1' and (mode_settings(env) is not None or not _enabled(env)):
        raise StartupError('auto-arm requires the native overlap composition')
    from native_measurement import progress_mode
    try:
        progress = progress_mode(env)
    except ValueError as exc:
        raise StartupError(str(exc)) from None
    fused = env.get('FLASH_S5_QSA_FUSED', '0')
    if fused not in ('0', '1'):
        raise StartupError('invalid boolean setting: FLASH_S5_QSA_FUSED')
    if fused == '1' and (auto != '1' or env.get('FLASH_S5_NATIVE_QSA_POLICY', 'current') != 'dense'):
        raise StartupError('FLASH_S5_QSA_FUSED needs FLASH_S5_AUTO_ARM=1 and the dense QSA policy')
    if fused == '1' and (env.get('FLASH_S5_NATIVE_QSA_TRACE', '0') != '0'
                         or env.get('FLASH_S5_NATIVE_QSA_STOP_AFTER', '0') != '0'):
        raise StartupError('FLASH_S5_QSA_FUSED refuses the QSA readback diagnostics')
    groups = env.get('FLASH_S5_LAYER_GROUPS', '0')
    if groups not in ('0', '1'):
        raise StartupError('invalid boolean setting: FLASH_S5_LAYER_GROUPS')
    # returning-append (Option A, A1): all four yield bounds at launch, or none.
    from scheduler_hooks import AppendYieldConfig
    from e3_types import E3ContractError
    try:
        append_yield = AppendYieldConfig.from_env(env)
    except E3ContractError as exc:
        raise StartupError(str(exc)) from None
    if append_yield is not None and auto != '1':
        raise StartupError('FLASH_S5_APPEND_YIELD_* needs FLASH_S5_AUTO_ARM=1')
    if groups == '1' and fused != '1':
        raise StartupError('FLASH_S5_LAYER_GROUPS needs FLASH_S5_QSA_FUSED=1')
    # alpha-next-c1: the C1 rebind switches are refused unless C1 is on (fail closed, never silently ignored).
    for name in ('FLASH_S5_STAGE_OVERLAP_DEFER_BIND', 'FLASH_S5_STAGE_OVERLAP_OBSERVE'):
        value = env.get(name, '0')
        if value not in ('0', '1'):
            raise StartupError('invalid boolean setting: ' + name)
        if value == '1' and env.get('FLASH_S5_STAGE_OVERLAP', '0') != '1':
            raise StartupError(name + ' needs FLASH_S5_STAGE_OVERLAP=1')
    # stage1-step: refused unless the deferred bind is on (fail closed, never silently ignored).
    for name in ('FLASH_S5_LAZY_RECV', 'FLASH_S5_DECODE_MERGE', 'FLASH_S5_INWAIT'):
        value = env.get(name, '0')
        if value not in ('0', '1'):
            raise StartupError('invalid boolean setting: ' + name)
        if value == '1' and env.get('FLASH_S5_STAGE_OVERLAP_DEFER_BIND', '0') != '1':
            raise StartupError(name + ' needs FLASH_S5_STAGE_OVERLAP_DEFER_BIND=1')
    if mode_settings(env) is not None:
        # The decode-only probe has no engine-side lane composition.
        return None
    if not _enabled(env):
        return None
    require_native_ready()
    if getattr(engine, _COMPOSITION_ATTR, None) is not None:
        raise StartupError('engine already carries an E3 composition')
    executor = getattr(engine, 'model_executor', None)
    scheduler = getattr(engine, 'scheduler', None)
    if executor is None or scheduler is None:
        raise StartupError('engine startup seam ran before executor/scheduler construction')
    manager = getattr(scheduler, 'kv_cache_manager', None)
    if manager is None or getattr(manager, 'block_pool', None) is None:
        raise StartupError('engine startup requires the E2 block pool for E3 leases')
    queues = getattr(executor, 'response_mqs', None)
    if not queues:
        raise StartupError('engine startup needs the executor reply queues')
    ranks = tuple(range(len(queues)))
    shape = _native_execution_shape(engine, ranks, env=env)
    if shape is None:
        raise StartupError('native engine requires the declared PP model shape')

    from executor_hooks import install
    from native_backend import RemoteCompletionBackend
    from native_resources import response_success

    config = HandoffConfig.from_env(env)
    # The two-FIFO reply receiver exists only when the burst candidate is installed.
    extra = {'direct_replies': True} if progress == 'burst' else {}
    transport = install(executor, config=config, success_status=response_success(),
                        clock=clock, **extra)
    backend = RemoteCompletionBackend(config, transport=transport)
    allocation = estimated_e3_inventory(chunk_tokens=config.chunk_tokens)
    composition = build_composition(
        env, transport=transport, backend=backend, lanes={},
        native_allocations=allocation, allocation=allocation, clock=clock,
        side='engine', scheduler_leases=SchedulerLeases())
    if not isinstance(composition, Composition):
        raise StartupError('enabled engine startup produced no composition')
    composition.attach(engine=engine, scheduler=scheduler, executor=executor)

    from engine_driver import NativeSegmentDriver
    from admission_metadata import install_admission_metadata
    from scheduler_hooks import PrefillAdmissionController, QSA_STAGE_MIN_ROWS

    layer_count, layer_split, stage_of = shape
    # Created once by this EngineCore process, passed to all worker opens.
    # It is not a model/request attribute and cannot silently default to zero.
    import uuid
    # Bootstrap/binding contracts require a positive owner epoch; the random
    # draw is masked to 63 bits and must never land on the unbound sentinel 0.
    epoch = (uuid.uuid4().int & ((1 << 63) - 1)) or 1
    engine._flash_s5_engine_epoch = epoch
    # FLASH_S5_STAGE_OVERLAP: unset or 0 constructs the frozen controller, unchanged.
    controller_class = PrefillAdmissionController
    overlap = False
    if env.get('FLASH_S5_STAGE_OVERLAP', '0') != '0':
        from stage_overlap import StagedAdmissionController, stage_overlap_enabled
        overlap = stage_overlap_enabled(env)
        if overlap:
            controller_class = StagedAdmissionController
    controller = controller_class(
        scheduler, leases=composition.scheduler_leases,
        frontier=composition.frontier,
        engine_epoch=epoch,
        chunk_tokens=config.chunk_tokens, ranks=ranks, clock=clock,
        minimum_stage_rows=QSA_STAGE_MIN_ROWS).attach()
    # The only construction site of the production admission-metadata
    # provider: it captures allocation side effects around the controller's
    # allocate_slots call and owns CoW endpoint retention until commit/cancel.
    install_admission_metadata(scheduler)
    # FLASH_S5_STAGE_LOCAL: unset or 0 constructs the frozen driver, unchanged.
    driver_class = NativeSegmentDriver
    if env.get('FLASH_S5_STAGE_LOCAL', '0') != '0':
        from stage_progress import StagedSegmentDriver, stage_local_enabled
        if stage_local_enabled(env):
            driver_class = StagedSegmentDriver
    if overlap:
        from stage_overlap import OverlapSegmentDriver
        driver_class = OverlapSegmentDriver
    # host-gap batch (FLASH_S5_SMALL_PIECE_BATCH): unset or 0 keeps the class above; 1 needs the overlap driver.
    if env.get('FLASH_S5_SMALL_PIECE_BATCH', '0') != '0':
        from small_piece_batch import driver_class as small_piece_driver, small_piece_batch_enabled
        if small_piece_batch_enabled(env):
            if not overlap:
                raise RuntimeError('FLASH_S5_SMALL_PIECE_BATCH=1 needs FLASH_S5_STAGE_OVERLAP=1')
            driver_class = small_piece_driver()
    driver = driver_class(
        controller, executor=executor, scheduler=scheduler, engine=engine,
        layer_count=layer_count, layer_split=layer_split, stage_of=stage_of,
        clock=clock, reply_timeout_s=config.reply_timeout_s,
        # Host-monotonic observation clock for the control journal; the driver
        # never substitutes a CUDA elapsed duration for a host timestamp.
        monotonic_ns=time.monotonic_ns).install()
    if overlap:
        # coverage-throughput build-2 (a): the successor's admission rides its stage's finalize, polls and release.
        from stage_overlap import prepare_ahead_enabled
        driver.prepare_ahead = prepare_ahead_enabled(env)
    engine._flash_s5_composition = composition
    engine._flash_s5_admission = controller
    engine._flash_s5_driver = driver
    from native_measurement import install_receipt_writer, receipt_path
    # Alpha-next PLAN 2: the engine's receipts (progress, anatomy, stage-local)
    # go out through one atomic write per record. Nothing is written until a receipt.
    install_receipt_writer()
    bootstrap_full_startup(engine, driver, timeout_s=config.reply_timeout_s)
    if auto == '1':
        # Receipts are possible from here on: say whether their path is atomic.
        # Graders mark every receipt-derived read INC on unsafe, never tolerate it.
        try:
            from vllm.logger import init_logger
        except ImportError:
            import logging
            init_logger = logging.getLogger
        init_logger(__name__).info('flash-s5 startup receipt_path=%s', receipt_path())
    if auto == '1':
        from native_auto_arm import install
        install(engine, driver, env)
    if append_yield is not None:
        controller.append_yield = append_yield
    if progress is not None:
        # Held-f receipts (and the burst candidate) attach to the auto policy only.
        driver.install_progress(progress)
    if fused == '1':
        driver.install_qsa_fused()
    if groups == '1':
        driver.install_layer_groups()
    return composition


def bootstrap_full_startup(engine, driver, *, timeout_s, clock=time.monotonic):
    """Engine-owned cold FULL fence; completes before EngineCore's ready message.

    There is no request/client or main-loop reentry here. The same driver and
    sole reply receiver pump the existing bounded control state machine.
    """
    if getattr(engine, '_flash_s5_startup_ready', None) is not None:
        raise StartupError('startup owner already published')
    scheduler = engine.scheduler
    if getattr(scheduler, 'requests', None) or getattr(engine, 'batch_queue', None):
        raise StartupError('startup owner requires an empty engine')
    old_pause = scheduler.pause_state
    scheduler.set_pause_state(type(old_pause).PAUSED_ALL)
    deadline = clock() + timeout_s
    records = {}
    try:
        for action in ('bootstrap_engine_owner', 'barrier', 'enter_split', 'leave_split'):
            future = driver.submit_control({'action': action, 'engine_epoch': driver.engine_epoch})
            while not future.done():
                if clock() >= deadline:
                    raise StartupError('startup FULL owner fence timed out')
                driver.control_tick()
                if not future.done():
                    time.sleep(0.001)
            records[action] = future.result()
            if action == 'barrier':
                rows = records[action].get('workers')
                zeros = ('live_decode_requests', 'running_requests', 'prefill_requests',
                         'retained_rows', 'pending_native_replays',
                         'pending_prefill_streams', 'pending_output_copies')
                if (type(rows) is not list or len(rows) != len(driver.ranks)
                        or {row.get('worker_rank') for row in rows} != set(driver.ranks)
                        or any(row.get('engine_epoch') != driver.engine_epoch
                               or row.get('route') != 'full'
                               or row.get('route_generation') != 0
                               or row.get('barrier_clean') is not True
                               or (row.get('live_decode') or {}).get('available') is not True
                               or any(type(row.get(k)) is not int or row[k] != 0 for k in zeros)
                               for row in rows)):
                    raise StartupError('startup FULL owner fence needs empty drained workers')
        final = records['leave_split']
        if (not driver._bootstrapped or final.get('route_after') != 'full'
                or final.get('typed_acks') is not True
                or len(final.get('transition_acks', ())) != len(driver.ranks)
                or driver.admission_intent() is not None):
            raise StartupError('startup FULL owner proof incomplete')
        engine._flash_s5_startup_ready = dict(engine_epoch=driver.engine_epoch,
            bootstrap_acks=driver.bootstrap_ack_rows(),
            full_transition={key: final[key] for key in ('engine_epoch', 'route_after',
                'route_generation_after', 'typed_acks', 'transition_acks')})
    except BaseException:
        # Do not resume an incompletely bound generation; init must fail closed.
        engine.model_executor.shutdown()
        raise
    else:
        scheduler.set_pause_state(old_pause)
    from vllm.logger import init_logger
    init_logger(__name__).info(
        'flash-s5 startup owner bound engine_epoch=%s ranks=%s route_after=%s route_generation_after=%s',
        driver.engine_epoch, list(driver.ranks), final['route_after'], final['route_generation_after'])
    return engine._flash_s5_startup_ready


def settings(env=None):
    """Metadata-only settings echo; no secret or path is read."""
    env = os.environ if env is None else env
    config = HandoffConfig.from_env(env)
    from native_readiness import implementation_status
    ready = implementation_status() if config.enabled else {'native_ready': False}
    return dict(enabled=config.enabled, transport=config.transport,
                command_slots=config.command_slots,
                reply_timeout_s=config.reply_timeout_s,
                ack_timeout_s=config.ack_timeout_s,
                chunk_tokens=config.chunk_tokens,
                sparse_overlap=bool(config.enabled and ready['native_ready']),
                backend=('native-owner-test' if ready['native_ready'] else 'unimplemented')
                        if config.enabled else 'inert')


__all__ = ['OVERLAP_ENV', 'StartupError', 'install_builder', 'install_engine', 'install_native_lane',
           'require_decode_probe_ready', 'require_native_ready', 'settings']
