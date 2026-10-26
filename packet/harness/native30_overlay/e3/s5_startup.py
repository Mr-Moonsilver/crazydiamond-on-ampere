"""C startup: native worker construction and engine-side composition.

Production enabled startup is hard-refused before allocation or transport
ownership: bounded native execution, admission and green capture are missing.
The construction paths below are exercised only by explicitly marked CPU fixtures.
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
    """Hard refusal until the missing native mechanisms have been implemented.

    No environment escape hatch: tests may explicitly mock this function to
    exercise controller scaffolding, which is not native integration evidence.
    """
    raise StartupError('E3 native integration incomplete: admission, bounded execution, '
                       'green graph capture and source/PP ownership are not implemented')


def _enabled(env):
    value = env.get(OVERLAP_ENV, '0')
    if value not in ('0', '1'):
        raise StartupError('invalid boolean setting: ' + OVERLAP_ENV)
    return value == '1'


def install_builder(runner, kv_caches, is_profiling=False, *, env=None):
    """Install the builder A's ``init_lane`` calls at the end of runner init."""
    env = os.environ if env is None else env
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

        return QsaLayerDriver(operation, staged_operation=qsa_module._flash_s5_staged_operation,
                              forward_context=forward_context_getter)

    def qsa_module():
        import vllm.models.qwen4_exp.nvidia.qsa as module

        return module

    def forward_context_getter():
        from vllm.forward_context import get_forward_context

        return get_forward_context()

    factory = LaneChunkFactory(runner)
    dispatcher = LaneDispatcher(runner, native, factory, server=server, qsa_module=qsa_module(),
                                driver_factory=driver_factory,
                                forward_context_getter=forward_context_getter)
    server._flash_s5_dispatcher = dispatcher
    runner._flash_s5_dispatcher = dispatcher
    return dispatcher


def install_engine(engine, *, env=None, clock=time.monotonic):
    """Install the engine-side composition at the generated ``__init__`` seam."""
    env = os.environ if env is None else env
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
    transport = install(executor, config=config, success_status=response_success(),
                        clock=clock)
    backend = RemoteCompletionBackend(config)
    allocation = estimated_e3_inventory(chunk_tokens=config.chunk_tokens)
    composition = build_composition(
        env, transport=transport, backend=backend, lanes={},
        native_allocations=allocation, allocation=allocation, clock=clock,
        side='engine', scheduler_leases=SchedulerLeases())
    if not isinstance(composition, Composition):
        raise StartupError('enabled engine startup produced no composition')
    composition.attach(engine=engine, scheduler=scheduler, executor=executor)

    from engine_driver import NativeSegmentDriver
    from scheduler_hooks import PrefillAdmissionController

    layer_count, layer_split, stage_of = shape
    # Created once by this EngineCore process, passed to all worker opens.
    # It is not a model/request attribute and cannot silently default to zero.
    import uuid
    epoch = uuid.uuid4().int & ((1 << 63) - 1)
    engine._flash_s5_engine_epoch = epoch
    controller = PrefillAdmissionController(
        scheduler, leases=composition.scheduler_leases,
        frontier=composition.frontier,
        engine_epoch=epoch,
        chunk_tokens=config.chunk_tokens, ranks=ranks, clock=clock).attach()
    driver = NativeSegmentDriver(
        controller, executor=executor, scheduler=scheduler, engine=engine,
        layer_count=layer_count, layer_split=layer_split, stage_of=stage_of,
        clock=clock, reply_timeout_s=config.reply_timeout_s).install()
    engine._flash_s5_composition = composition
    engine._flash_s5_admission = controller
    engine._flash_s5_driver = driver
    return composition


def settings(env=None):
    """Metadata-only settings echo; no secret or path is read."""
    env = os.environ if env is None else env
    config = HandoffConfig.from_env(env)
    return dict(enabled=config.enabled, transport=config.transport,
                command_slots=config.command_slots,
                reply_timeout_s=config.reply_timeout_s,
                ack_timeout_s=config.ack_timeout_s,
                chunk_tokens=config.chunk_tokens,
                sparse_overlap=False, backend='unimplemented' if config.enabled else 'inert')


__all__ = ['OVERLAP_ENV', 'StartupError', 'install_builder', 'install_engine', 'install_native_lane',
           'require_decode_probe_ready', 'require_native_ready', 'settings']
