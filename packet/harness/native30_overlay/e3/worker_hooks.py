"""Generated-source seams for the GPU worker and the cudagraph manager (D4).

Two pinned files are transformed, each with unique anchors and an exact inverse:

``v1/worker/gpu_worker.py``
  * one extra workspace lane under ``FLASH_S5_OVERLAP=1`` (before the model is
    built, so the lane exists before any model workspace user and capture);
  * ``after_runner_init``: creates the declared green split right after the
    runner is constructed (device already selected) and installs the split bank
    objects on the runner;
  * ``decode_scope`` around the worker's ``execute_model`` and ``sample_tokens``
    so that in split mode the whole decode call, including PP send/recv, runs
    with the green decode stream current.

``v1/worker/gpu/cudagraph_utils.py``
  * the stock capture takes its capture context from the manager when one is
    installed (``_flash_s5_capture_context``); the stock manager has none and
    keeps its fresh side stream.

Every seam is inert with the feature off: the extra lane is not added, no
split is created, ``decode_scope`` is a null context and the capture context
is ``None``.
"""
import ast
import hashlib
import os

PINS = {
    'v1/worker/gpu_worker.py': '596f5b6e97be754f84b04c17bc0b96fd92d2f05ee86e0459aa515ffa48d8c5cb',
    'v1/worker/gpu/cudagraph_utils.py': 'b1423812da6d07933bbaf014fa21236c944af1c5f8e553aefb2de89af8afb384',
}
# Read-only behavior dependency; not a transformed source.
CAPTURE_REFERENCE_PINS = {
    'utils/torch_utils.py': 'eec8b1e3c7c673d3d171b92530f8b07acabdad5b0e3ab07cc0e717f827af8e5a',
}

# executor_hooks.generate_worker output (C's RPC methods); the worker seams
# compose after C in the overlay, so that composition is an accepted input too.
WORKER_AFTER_C_SHA256 = '4bda8ffb267e3f66874e7ffc13c35aeac23f738c0c5cf11a617e5a71f6b9bbca'
ACCEPTED = {path: (digest,) for path, digest in PINS.items()}
ACCEPTED['v1/worker/gpu_worker.py'] = (PINS['v1/worker/gpu_worker.py'], WORKER_AFTER_C_SHA256)

# --- gpu_worker.py ----------------------------------------------------------
_W_IMPORT_OLD = 'from vllm.v1.worker.workspace import init_workspace_manager\n'
_W_IMPORT_NEW = _W_IMPORT_OLD + 'import vllm.v1.flash_e3.worker_hooks as _flash_s5_worker\n'

_W_LANES_OLD = '            _num_workspace_lanes(self.vllm_config, self.use_v2_model_runner),\n'
_W_LANES_NEW = ('            _flash_s5_worker.workspace_lanes(\n'
                '                _num_workspace_lanes(self.vllm_config, self.use_v2_model_runner)),\n')

_W_RUNNER_OLD = ('            self.model_runner: GPUModelRunner = GPUModelRunnerV2(  # type: ignore\n'
                 '                self.vllm_config, self.device\n'
                 '            )\n')
_W_RUNNER_NEW = _W_RUNNER_OLD + '            _flash_s5_worker.after_runner_init(self)\n'

_W_SAMPLE_OLD = '        return self.model_runner.sample_tokens(grammar_output)\n'
_W_SAMPLE_NEW = ('        with _flash_s5_worker.decode_scope(self):\n'
                 '            _flash_s5_worker.observe_sample(self)\n'
                 '            return self.model_runner.sample_tokens(grammar_output)\n'
                 '\n'
                 '    def flash_s5_decode_probe(self, command):\n'
                 '        return _flash_s5_worker.decode_probe_rpc(self, command)\n'
                 '\n'
                 '    def flash_s5_native_overlap(self, command):\n'
                 '        return _flash_s5_worker.native_overlap_rpc(self, command)\n')

_W_EXEC_OLD = ('    @torch.inference_mode()\n'
               '    @with_gpu_sync_check\n'
               '    def execute_model(\n'
               '        self, scheduler_output: "SchedulerOutput"\n'
               '    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:\n')
_W_EXEC_NEW = (_W_EXEC_OLD +
               '        _flash_s5_worker.observe_scheduler(self, scheduler_output)\n'
               '        with _flash_s5_worker.decode_scope(self):\n'
               '            return self._flash_s5_stock_execute_model(scheduler_output)\n'
               '\n'
               '    def _flash_s5_stock_execute_model(\n'
               '        self, scheduler_output: "SchedulerOutput"\n'
               '    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:\n')

WORKER_FORWARD = (
    (_W_IMPORT_OLD, _W_IMPORT_NEW),
    (_W_LANES_OLD, _W_LANES_NEW),
    (_W_RUNNER_OLD, _W_RUNNER_NEW),
    (_W_SAMPLE_OLD, _W_SAMPLE_NEW),
    (_W_EXEC_OLD, _W_EXEC_NEW),
)

# --- cudagraph_utils.py -----------------------------------------------------
_C_IMPORT_OLD = 'from vllm.utils.torch_utils import current_stream\n'
_C_IMPORT_NEW = (_C_IMPORT_OLD +
                 'from vllm.v1.flash_e3.worker_hooks import capture_context as _flash_s5_capture_context\n'
                 'from vllm.v1.flash_e3.worker_hooks import capture_graph as _flash_s5_capture_graph, finish_graph as _flash_s5_finish_graph\n'
                 'from vllm.v1.flash_e3.worker_hooks import checked_capture_stream as _flash_s5_checked_capture_stream\n')
_C_CAPTURE_OLD = '        with graph_capture(device=self.device):\n'
_C_CAPTURE_NEW = ('        with graph_capture(\n'
                  '            device=self.device,\n'
                  '            graph_capture_context=_flash_s5_capture_context(self),\n'
                  '        ):\n')

_C_GRAPH_OLD = '                        graph = torch.cuda.CUDAGraph()\n'
_C_GRAPH_NEW = '                        graph = _flash_s5_capture_graph(self)\n'
_C_STREAM_OLD = '                            graph, self.pool, stream=current_stream()\n'
_C_STREAM_NEW = ('                            graph, self.pool,\n'
                 '                            stream=_flash_s5_checked_capture_stream(self, current_stream(), desc)\n')
_C_STORE_OLD = '                        self.graphs[desc] = graph\n'
_C_STORE_NEW = '                        _flash_s5_finish_graph(self, graph)\n' + _C_STORE_OLD

CUDAGRAPH_FORWARD = (
    (_C_IMPORT_OLD, _C_IMPORT_NEW),
    (_C_CAPTURE_OLD, _C_CAPTURE_NEW),
    (_C_GRAPH_OLD, _C_GRAPH_NEW),
    (_C_STREAM_OLD, _C_STREAM_NEW),
    (_C_STORE_OLD, _C_STORE_NEW),
)

TRANSFORMS = {
    'v1/worker/gpu_worker.py': WORKER_FORWARD,
    'v1/worker/gpu/cudagraph_utils.py': CUDAGRAPH_FORWARD,
}
# Presence markers of THIS module's seams (C's worker transform uses other names).
_MARKER = {
    'v1/worker/gpu_worker.py': '_flash_s5_worker.',
    'v1/worker/gpu/cudagraph_utils.py': '_flash_s5_capture_context(',
}


def _replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('source anchor changed or duplicated: ' +
                         old[:80].replace('\n', '\\n'))
    return text.replace(old, new, 1)


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def generate_one(path, source):
    """Transform one pinned source; refuse drift or double application."""
    pinned = PINS.get(path)
    if pinned is None:
        raise ValueError('worker hooks do not transform ' + path)
    if _MARKER[path] in source:
        raise ValueError('source already carries worker hooks: ' + path)
    if _sha(source) not in ACCEPTED[path]:
        raise ValueError('source is not the pinned bytes: ' + path)
    text = source
    for old, new in TRANSFORMS[path]:
        text = _replace(text, old, new)
    try:
        ast.parse(text)
    except SyntaxError as exc:
        raise ValueError('generated source syntax invalid: ' + path) from exc
    return text


def revert_one(path, source, expected=None):
    """Exact inverse of :func:`generate_one`."""
    if path not in TRANSFORMS:
        raise ValueError('worker hooks do not transform ' + path)
    if _MARKER[path] not in source:
        raise ValueError('source does not carry worker hooks: ' + path)
    text = source
    for old, new in reversed(TRANSFORMS[path]):
        text = _replace(text, new, old)
    if expected is not None and text != expected:
        raise ValueError('revert did not recover the supplied source')
    if _sha(text) not in ACCEPTED[path]:
        raise ValueError('revert did not recover the pinned bytes: ' + path)
    return text


def generate(files):
    """``{path: source}`` for the pinned inputs -> transformed outputs."""
    out = {}
    for path in PINS:
        if path not in files:
            raise ValueError('missing pinned input ' + path)
        out[path] = generate_one(path, files[path])
    return out


# --- runtime seams (called by generated source) -----------------------------

def _enabled(env=None):
    env = os.environ if env is None else env
    return env.get('FLASH_S5_OVERLAP', '0') == '1'


PROBE_ENV = 'FLASH_S5_DECODE_ONLY_PROBE'


def _split_enabled(env=None):
    """True when either the overlap path or the decode-only probe needs a split.

    The default-off pair takes no import and no allocation.  With either flag
    set, ``decode_probe.mode_settings`` validates the flag conflict and the
    explicit decode SM declaration before the split is opened.
    """
    env = os.environ if env is None else env
    if env.get('FLASH_S5_OVERLAP', '0') == '0' and env.get(PROBE_ENV, '0') == '0':
        return False
    from decode_probe import mode_settings

    if mode_settings(env) is not None:
        return True
    return _enabled(env)


def workspace_lanes(stock_lanes, env=None):
    """One dedicated prefill lane on top of the stock count when enabled."""
    if type(stock_lanes) is not int or stock_lanes < 1:
        raise ValueError('stock workspace lane count must be a positive int')
    if _split_enabled(env):  # rejects a flag conflict before workspace allocation
        return stock_lanes + 1 if _enabled(env) else stock_lanes
    return stock_lanes


def prefill_workspace_lane(stock_lanes):
    """The lane index the prefill lane owns: always the extra, last lane."""
    return stock_lanes


def capture_context(manager):
    """Capture context for a cudagraph manager; stock managers return None."""
    factory = getattr(manager, '_flash_s5_capture_context', None)
    if factory is None:
        return None
    context = factory()
    if context is None:
        raise ValueError('split capture context factory returned None')
    return context


def after_runner_init(worker, env=None):
    """Create the declared green split and install the split bank objects.

    In the decode-only probe mode only the split is created here; the split
    bank is installed later by ``install_decode_probe`` once the stock manager
    exists.
    """
    if not _split_enabled(env):
        return None
    runner = getattr(worker, 'model_runner', None)
    if runner is None:
        raise ValueError('worker has no model runner')
    if getattr(runner, '_flash_s5_split', None) is not None:
        raise ValueError('green split already created for this runner')
    from decode_probe import mode_settings, require_decode_probe_ready
    if mode_settings(os.environ if env is None else env) is not None:
        require_decode_probe_ready()
        values = os.environ if env is None else env
        lib = values.get('FLASH_S5_GREEN_LIB')
        digest = values.get('FLASH_S5_GREEN_LIB_SHA256')
        if not lib or not digest or len(digest) != 64:
            raise ValueError('decode probe needs the declared helper library sha256')
        with open(lib, 'rb') as handle:
            if hashlib.sha256(handle.read()).hexdigest() != digest:
                raise ValueError('decode probe helper library pin differs')
    from observer_runtime import start_worker_observer
    observer = start_worker_observer(worker, runner, env)
    from green_context import open_split

    device = getattr(worker, 'device', None)
    index = getattr(device, 'index', None)
    if index is None:
        import torch

        index = torch.cuda.current_device()
    split = None
    try:
        split = open_split(int(index), env=env)
        if not split.membership().ok:
            raise ValueError('green split membership failed')
    except BaseException:
        if split is not None:
            split.close()
        if observer is not None:
            observer.shutdown()
        raise
    runner._flash_s5_split = split
    runner._flash_s5_native_identity = split.describe()
    if _enabled(env):
        from native_overlap_worker import install_census

        install_census(runner)
    return split


def install_split_bank(runner, *, set_graph_pool_id=None):
    """Build the bank/backend/controller once the stock manager exists.

    Called from the runner's E3 startup (after ``initialize_kv_cache`` built the
    stock cudagraph manager and before capture); the bank captures through B's
    ``capture_banks`` seam.
    """
    from split_bank import install

    split = getattr(runner, '_flash_s5_split', None)
    if split is None:
        raise ValueError('split bank needs the green split created at worker init')
    return install(runner, split, set_graph_pool_id=set_graph_pool_id)


def decode_scope(worker):
    """Null context when off; the green decode stream when in split mode."""
    runner = getattr(worker, 'model_runner', None)
    if runner is None or getattr(runner, '_flash_s5_decode_stream', None) is None:
        from contextlib import nullcontext

        return nullcontext()
    from split_bank import decode_scope as scope

    return scope(runner)


__all__ = ['ACCEPTED', 'CUDAGRAPH_FORWARD', 'PINS', 'PROBE_ENV', 'WORKER_AFTER_C_SHA256', 'TRANSFORMS', 'WORKER_FORWARD', 'after_runner_init',
           'capture_context', 'decode_scope', 'generate', 'generate_one',
           'install_split_bank', 'native_overlap_rpc', 'observe_sample',
           'observe_scheduler', 'prefill_workspace_lane', 'revert_one',
           'workspace_lanes']


def decode_probe_rpc(worker, command):
    """Ordinary ordered worker RPC, separate from unqualified overlap transport.

    Owner must submit at an engine dispatch boundary and collect every rank's
    ACK before admitting the next request/step. A partial ACK set invalidates the
    whole interval. No scheduler/admission/reply ownership is installed here.
    """
    from decode_probe import mode_settings
    from native_startup import NativeStartupError, probe_transition_rpc
    if mode_settings() is None:
        raise NativeStartupError('decode probe RPC requires diagnostic mode')
    runner = getattr(worker, 'model_runner', None)
    probe = getattr(runner, '_flash_s5_decode_probe', None)
    if probe is None or type(command) is not dict:
        raise NativeStartupError('decode probe RPC before capture or invalid command')
    if type(getattr(worker, 'rank', None)) is not int:
        raise NativeStartupError('worker rank unavailable')
    action = command.get('action')
    if action == 'route_invariant':
        from route_invariant import Snapshot, InvariantUnavailable
        if (set(command) != {'action','engine_epoch','phase'}
                or command['engine_epoch'] != getattr(probe, '_rpc_engine_epoch', None)
                or command['phase'] not in ('capture','entered','measured','returned')):
            raise NativeStartupError('route invariant identity differs')
        probe._require_not_down()
        if (probe.in_flight or getattr(runner, 'execute_model_state', None) is not None
                or not probe.controller.admitting):
            raise NativeStartupError('route invariant requires completed sampling')
        drained=probe.controller.drain()
        if type(drained) is not int or drained<=0:
            raise NativeStartupError('route invariant drain unproved')
        phase=command['phase']
        expected_route='full' if phase in ('capture','returned') else 'split'
        if probe.route!=expected_route:
            raise NativeStartupError('route invariant wrong boundary')
        if phase=='capture':
            if getattr(probe,'_byte_snapshot',None) is not None:
                raise NativeStartupError('route snapshot already owned')
            try:
                probe._byte_snapshot=Snapshot(probe)
                probe._byte_snapshot_missing=None
            except (InvariantUnavailable, AttributeError) as exc:
                probe._byte_snapshot=None
                probe._byte_snapshot_missing=(str(exc) if isinstance(exc,InvariantUnavailable)
                                              else 'route-invariant-source-unavailable')
            result={'schema':1,'phase':phase,'verdict':'INCONCLUSIVE' if probe._byte_snapshot is None else 'PASS',
                    'missing':probe._byte_snapshot_missing}
        else:
            snap=getattr(probe,'_byte_snapshot',None)
            result=(snap.check(phase) if snap is not None else
                    {'schema':1,'phase':phase,'verdict':'INCONCLUSIVE',
                     'missing':getattr(probe,'_byte_snapshot_missing','snapshot-not-captured')})
            if phase=='returned':probe._byte_snapshot=None
        return dict(result,rank=worker.rank,ack='flash_s5_route_invariant',
                    engine_epoch=command['engine_epoch'],storage_epoch=probe.storage_epoch,route_generation=probe.route_generation,
                    route=probe.route,drained_streams=drained)
    if action == 'teacher_evidence':
        from teacher_decode import worker_evidence
        if (set(command) != {'action', 'engine_epoch'}
                or type(command['engine_epoch']) is not int or command['engine_epoch'] < 1
                or command['engine_epoch'] != getattr(probe, '_rpc_engine_epoch', None)):
            raise NativeStartupError('teacher evidence identity differs')
        probe._require_not_down()
        if (probe.in_flight or getattr(runner, 'execute_model_state', None) is not None
                or not getattr(probe.controller, 'admitting', False)):
            raise NativeStartupError('teacher evidence requires completed sampling')
        drained = probe.controller.drain()
        if type(drained) is not int or drained <= 0:
            raise NativeStartupError('teacher evidence drain unproved')
        return dict(worker_evidence(runner), rank=worker.rank,
                    ack='flash_s5_teacher_evidence', engine_epoch=command['engine_epoch'],
                    storage_epoch=probe.storage_epoch, route_generation=probe.route_generation,
                    route=probe.route, drained_streams=drained)
    if action == 'teacher_configure':
        from teacher_decode import configure_worker
        if (set(command) != {'action', 'engine_epoch', 'enabled', 'snapshot_max_bytes'}
                or type(command['snapshot_max_bytes']) is not int
                or not 64*1024**2<=command['snapshot_max_bytes']<=2*1024**3
                or type(command['enabled']) is not bool
                or type(command['engine_epoch']) is not int
                or command['engine_epoch'] < 1
                or command['engine_epoch'] != getattr(probe, '_rpc_engine_epoch', None)):
            raise NativeStartupError('teacher configuration identity differs')
        probe._require_not_down()
        if (probe.route != 'full' or probe.in_flight
                or getattr(runner, 'execute_model_state', None) is not None
                or not getattr(probe.controller, 'admitting', False)
                or getattr(getattr(runner, 'req_states', None), 'req_id_to_index', None) != {}):
            raise NativeStartupError('teacher configuration needs empty full worker')
        try:
            drained = probe.controller.drain()
            if type(drained) is not int or drained <= 0:
                raise NativeStartupError('teacher configuration drain unproved')
            result = configure_worker(runner, command['enabled'], rank=worker.rank)
            probe._snapshot_max_bytes=command['snapshot_max_bytes']
            result['snapshot_max_bytes']=command['snapshot_max_bytes']
        except BaseException:
            probe.quarantined = 'teacher configuration failed'
            raise
        return dict(result, rank=worker.rank, ack='flash_s5_teacher_configure',
                    engine_epoch=command['engine_epoch'], drained_streams=drained)
    if action == 'status':
        if set(command) != {'action'}:
            raise NativeStartupError('status has unexpected fields')
        return {'rank': worker.rank, 'inventory': probe.inventory(),
                'counters': probe.counters(), 'memory': probe.memory_snapshot()}
    if action in ('observer_begin', 'observer_end','launch_begin','launch_end'):
        fields = ({'action', 'engine_epoch', 'run_id', 'window_id'}
                  if action in ('observer_begin', 'observer_end') else {'action', 'engine_epoch'})
        if set(command) != fields:
            raise NativeStartupError('observer fields differ')
        owner = getattr(runner, '_flash_s5_replay_observer', None)
        if owner is None:
            raise NativeStartupError('observer is not declared')
        epoch = command['engine_epoch']
        bound = getattr(probe, '_rpc_engine_epoch', None)
        if type(epoch) is not int or epoch < 1 or bound != epoch:
            raise NativeStartupError('observer engine epoch is not bound by a prior barrier')
        try:
            if action=='launch_begin':return owner.launch_begin(probe,epoch)
            if action=='launch_end':return owner.launch_end(probe,epoch)
            if action == 'observer_begin':
                return owner.begin(probe, epoch, run_id=command['run_id'], window_id=command['window_id'])
            if owner.window_identity is not None and owner.window_identity != (command['run_id'], command['window_id']):
                raise NativeStartupError('observer window identity changed')
            return owner.end(probe, epoch)
        except BaseException:
            probe.quarantined = 'diagnostic observer operation failed'
            raise
    if action == 'barrier':
        if set(command) != {'action', 'engine_epoch'}:
            raise NativeStartupError('barrier fields differ')
        epoch = command['engine_epoch']
        if type(epoch) is not int or epoch < 1:
            raise NativeStartupError('barrier needs positive engine epoch')
        bound = getattr(probe, '_rpc_engine_epoch', None)
        if bound is not None and epoch != bound:
            raise NativeStartupError('barrier engine epoch changed')
        probe._require_not_down()
        if (probe.in_flight or getattr(runner, 'execute_model_state', None) is not None
                or not getattr(probe.controller, 'admitting', False)):
            raise NativeStartupError('barrier has unresolved worker work')
        try:
            drained = probe.controller.drain()
        except BaseException:
            probe.quarantined = 'diagnostic barrier failed'
            raise
        probe._rpc_engine_epoch = epoch
        return {'rank': worker.rank, 'ack': 'flash_s5_probe_barrier',
                'mode': 'decode-only-probe', 'engine_epoch': epoch,
                'storage_epoch': probe.storage_epoch, 'route': probe.route,
                'route_generation': probe.route_generation, 'drained_streams': drained}
    if action != 'transition':
        raise NativeStartupError('unknown decode probe RPC action')
    expected = {'action', 'engine_epoch', 'expected_route_generation', 'route_generation',
                'storage_epoch', 'target'}
    if set(command) != expected:
        raise NativeStartupError('transition fields differ')
    epoch = command['engine_epoch']
    if type(epoch) is not int or epoch < 1:
        raise NativeStartupError('transition needs positive engine epoch')
    bound = getattr(probe, '_rpc_engine_epoch', None)
    if bound is not None and epoch != bound:
        raise NativeStartupError('transition engine epoch changed')
    ack = probe_transition_rpc(runner, {k: v for k,v in command.items() if k != 'action'},
                               engine_epoch=epoch)
    probe._rpc_engine_epoch = epoch
    ack.update(rank=worker.rank, engine_epoch=epoch)
    return ack


def observe_scheduler(worker, scheduler_output):
    runner = getattr(worker, 'model_runner', None)
    probe = getattr(runner, '_flash_s5_decode_probe', None)
    if probe is not None:
        probe.observe_scheduler(scheduler_output)
    # The native lane has its own dispatch census; it never reuses the probe's
    # (the probe refuses a runner that carries the native lane).
    census = getattr(runner, '_flash_s5_native_census', None)
    if census is not None:
        census.observe_scheduler(scheduler_output)


def observe_sample(worker):
    census = getattr(getattr(worker, 'model_runner', None),
                     '_flash_s5_native_census', None)
    if census is not None:
        census.observe_sample()


def native_overlap_rpc(worker, command):
    """Typed native-control RPC at the worker's ordinary ordered RPC boundary.

    All state stays on the worker: this handler only queries the worker's own
    constructed lane and runner rows, or delegates an exact qualification
    action to ``native_qualification.handle_worker``.  It owns no engine queue
    and never reports a completion that was not queried.
    """
    from native_overlap_worker import handle

    return handle(worker, command)


def capture_graph(manager, *, torch_module=None):
    """Retain raw full/split definitions only in the declared decode diagnostic."""
    if torch_module is None:
        import torch as torch_module
    retained = os.environ.get(PROBE_ENV, '0') == '1'
    if retained:
        manager._flash_s5_keep_graphs = True
        return torch_module.cuda.CUDAGraph(keep_graph=True)
    return torch_module.cuda.CUDAGraph()


def finish_graph(manager, graph):
    if getattr(manager, '_flash_s5_keep_graphs', False):
        # keep_graph=True preserves nodes; instantiate before timing/pool accounting.
        graph.instantiate()


def checked_capture_stream(manager, vllm_stream, descriptor, *, torch_module=None):
    """Check the stream argument immediately before the actual graph context.

    The vLLM helper has already run at the generated call site. Do not call it
    again, repair its cache or change streams here. Stock/default-off is an
    identity function with no torch/driver queries. Native handles stay private.
    """
    if (os.environ.get(PROBE_ENV, '0') != '1'
            or getattr(manager, '_flash_s5_capture_context', None) is None):
        return vllm_stream
    from decode_probe import DecodeProbeError
    from graph_router import ExecutionDescriptor, descriptor_name
    owner = getattr(manager, '_flash_s5_capture_owner', None)
    if owner is None or getattr(owner, 'state', None) != 'open':
        raise DecodeProbeError('split capture has no open owning context')
    if torch_module is None:
        import torch as torch_module
    expected = getattr(owner, 'decode_stream_token', None)
    owning_stream = getattr(owner, 'decode_stream', None)
    torch_stream = torch_module.cuda.current_stream(manager.device)
    observed = (getattr(owning_stream, 'cuda_stream', None),
                getattr(vllm_stream, 'cuda_stream', None),
                getattr(torch_stream, 'cuda_stream', None))
    if (type(expected) is not int or expected <= 0
            or any(type(value) is not int or value != expected for value in observed)):
        raise DecodeProbeError('split capture stream agreement failed')
    local = ExecutionDescriptor.from_vllm(descriptor)
    if local.cg_mode != 'full':
        raise DecodeProbeError('split diagnostic capture is not FULL')
    label = descriptor_name(local)
    checks = getattr(manager, '_flash_s5_capture_stream_checks', None)
    if checks is None:
        checks = {}
        manager._flash_s5_capture_stream_checks = checks
    if label in checks:
        raise DecodeProbeError('duplicate split capture stream check')
    checks[label] = {
        'descriptor': label, 'bank': 'decode-split',
        'stream_label': 'decode-split-stream',
        'context_label': 'decode-split-context',
        'vllm_matches_owner': True, 'torch_matches_owner': True,
        'basis': 'checked at torch.cuda.graph stream argument; not node placement',
    }
    return vllm_stream
