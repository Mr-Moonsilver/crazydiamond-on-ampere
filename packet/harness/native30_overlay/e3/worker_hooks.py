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
# executor_hooks.generate_worker output (C's RPC methods); the worker seams
# compose after C in the overlay, so that composition is an accepted input too.
WORKER_AFTER_C_SHA256 = '95f03df33d31af06d27b3dd7ffb9fbc7d45a90466f354e2a45972752e29cd0b6'
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
                 '            return self.model_runner.sample_tokens(grammar_output)\n')

_W_EXEC_OLD = ('    @torch.inference_mode()\n'
               '    @with_gpu_sync_check\n'
               '    def execute_model(\n'
               '        self, scheduler_output: "SchedulerOutput"\n'
               '    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:\n')
_W_EXEC_NEW = (_W_EXEC_OLD +
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
                 'from vllm.v1.flash_e3.worker_hooks import capture_context as _flash_s5_capture_context\n')
_C_CAPTURE_OLD = '        with graph_capture(device=self.device):\n'
_C_CAPTURE_NEW = ('        with graph_capture(\n'
                  '            device=self.device,\n'
                  '            graph_capture_context=_flash_s5_capture_context(self),\n'
                  '        ):\n')

CUDAGRAPH_FORWARD = (
    (_C_IMPORT_OLD, _C_IMPORT_NEW),
    (_C_CAPTURE_OLD, _C_CAPTURE_NEW),
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


def workspace_lanes(stock_lanes, env=None):
    """One dedicated prefill lane on top of the stock count when enabled."""
    if type(stock_lanes) is not int or stock_lanes < 1:
        raise ValueError('stock workspace lane count must be a positive int')
    return stock_lanes + 1 if _enabled(env) else stock_lanes


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
    """Create the declared green split and install the split bank objects."""
    if not _enabled(env):
        return None
    runner = getattr(worker, 'model_runner', None)
    if runner is None:
        raise ValueError('worker has no model runner')
    if getattr(runner, '_flash_s5_split', None) is not None:
        raise ValueError('green split already created for this runner')
    from green_context import open_split

    device = getattr(worker, 'device', None)
    index = getattr(device, 'index', None)
    if index is None:
        import torch

        index = torch.cuda.current_device()
    split = open_split(int(index), env=env)
    try:
        split.membership()
    except BaseException:
        split.close()
        raise
    runner._flash_s5_split = split
    runner._flash_s5_native_identity = split.describe()
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


__all__ = ['ACCEPTED', 'CUDAGRAPH_FORWARD', 'PINS', 'WORKER_AFTER_C_SHA256', 'TRANSFORMS', 'WORKER_FORWARD', 'after_runner_init',
           'capture_context', 'decode_scope', 'generate', 'generate_one',
           'install_split_bank', 'prefill_workspace_lane', 'revert_one', 'workspace_lanes']
