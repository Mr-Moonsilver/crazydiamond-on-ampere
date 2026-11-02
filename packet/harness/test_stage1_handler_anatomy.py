"""Stage-1 stock-handler anatomy (``FLASH_S5_HANDLER_ANATOMY``): CPU tests over the real worker hooks.

The generated ``WorkerProc`` reply path (``_execute_worker_rpc``, ``handle_output``, ``enqueue_output``,
``async_output_busy_loop``) is executed from the pinned vLLM source through ``executor_hooks.generate_executor``
(as in ``test_e3_native_reply_hol``), with the real ``executor_hooks`` worker functions and the real
``handler_anatomy`` recorder. Everything below the GPU worker is a declared leaf
(``harness/stage1_anatomy_fixture_leaves.json``): the GPU worker and model runner (V1 and V2 method names and call
order, read from ``reference/vllm-src``), the PP/TP group coordinators, comm handles, the async output object and
torch.cuda (events and streams on a fake device clock: ``elapsed_time`` raises before both events query True,
``synchronize`` raises always and is recorded as a violation).

Run alone: ``cd harness && python3 test_stage1_handler_anatomy.py``.
"""
import ast
from contextlib import ExitStack
from functools import partial
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
import threading
import textwrap
import time
import traceback
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'patches/s5_engine'), str(ROOT / 'harness'), str(ROOT / 'tools')]
import executor_hooks as eh  # noqa: E402
import handler_anatomy as ha  # noqa: E402
import native_measurement as nm  # noqa: E402

BASE = '5f46c07'  # claude/step-b: the default-off reference for executor_hooks
REFERENCE = ROOT / 'reference/vllm-src'
WORKER_METHODS = ('_execute_worker_rpc', 'handle_output', 'enqueue_output', 'async_output_busy_loop')
OUTPUT_RANK = 2   # PP1 TP0: the stock execute/sample reply rank
MS = 1000000
HANDLER_PREFIX = 'flash-s5 handler '  # the schema's line prefix, held here so the off run needs no import


# --- generated WorkerProc ------------------------------------------------------

class Stop(BaseException):
    """Ends the async output loop; BaseException so the generated ``except Exception`` passes it."""


class AsyncModelRunnerOutput:
    """Leaf base class: the generated ``enqueue_output`` only tests ``isinstance`` against it."""


class Status:
    SUCCESS = 'SUCCESS'
    FAILURE = 'FAILURE'


def generated_executor():
    source = (REFERENCE / eh.EXECUTOR_PATH).read_text()
    if hashlib.sha256(source.encode()).hexdigest() != eh.EXECUTOR_SHA256:
        raise unittest.SkipTest('pinned vLLM executor source is not available')
    return eh.generate_executor(source)


def proc_class(hooks, generated=None):
    """The generated WorkerProc reply-path methods bound to ``hooks`` on a fixture class."""
    tree = ast.parse(generated if generated is not None else generated_executor())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WorkerProc')
    found = {n.name: n for n in owner.body if isinstance(n, ast.FunctionDef)}
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
                        + [found[name] for name in WORKER_METHODS], type_ignores=[])
    scope = {'_flash_s5_hooks': hooks, 'Any': object, 'AsyncModelRunnerOutput': AsyncModelRunnerOutput,
             'WorkerProc': types.SimpleNamespace(ResponseStatus=Status), 'partial': partial,
             'traceback': traceback, 'cloudpickle': None,
             'logger': types.SimpleNamespace(exception=lambda *a, **k: None)}
    exec(compile(ast.fix_missing_locations(module), eh.EXECUTOR_PATH, 'exec'), scope)
    return type('GeneratedWorkerProc', (), {name: scope[name] for name in WORKER_METHODS})


# --- leaf torch.cuda on a fake device clock ------------------------------------

class Device:
    def __init__(self):
        self.now = 0
        self.calls = []
        self.violations = []

    def advance(self, ns=10 ** 12):
        self.now += ns


class Stream:
    def __init__(self, device, name):
        self.device, self.name, self.tail = device, name, 0

    def query(self):
        self.device.calls.append('Stream.query')
        return self.device.now >= self.tail

    def synchronize(self):
        self.device.violations.append('Stream.synchronize')
        raise AssertionError('host sync on a stream')

    def launch(self, ns):
        """Fixture only: enqueue ``ns`` of device work (the leaf runner's kernels)."""
        self.tail = max(self.tail, self.device.now) + int(ns)


class Event:
    def __init__(self, device, enable_timing=False):
        self.device, self.timing, self.stamp = device, enable_timing, None

    def record(self, stream=None):
        self.device.calls.append('Event.record')
        stream = stream if stream is not None else self.device.current
        self.stamp = max(stream.tail, self.device.now)

    def query(self):
        self.device.calls.append('Event.query')
        return self.stamp is not None and self.device.now >= self.stamp

    def elapsed_time(self, end):
        self.device.calls.append('Event.elapsed_time')
        if not (self.query() and end.query()):
            self.device.violations.append('elapsed_time on an unresolved pair')
            raise RuntimeError('cudaErrorNotReady')
        return (end.stamp - self.stamp) / 1e6

    def synchronize(self):
        self.device.violations.append('Event.synchronize')
        raise AssertionError('host sync on an event')


class LeafTorch:
    """torch with only the cuda calls the recorder may make; anything else is an AttributeError."""

    def __init__(self):
        self.device = Device()
        self.device.current = Stream(self.device, 'main')
        device = self.device

        def synchronize(*args):
            device.violations.append('torch.cuda.synchronize')
            raise AssertionError('host sync')
        self.cuda = types.SimpleNamespace(Event=lambda enable_timing=False: Event(device, enable_timing),
                                          current_stream=lambda *a: device.current, synchronize=synchronize)

    @property
    def stream(self):
        return self.device.current


# --- leaf PP/TP groups ---------------------------------------------------------

def spin(seconds):
    end = time.thread_time() + seconds
    while time.thread_time() < end:
        pass


class Work:
    """torch.distributed Work leaf: ``wait`` blocks the host for ``delay`` (a gloo wait), no CPU."""

    def __init__(self, delay):
        self.delay, self.waited = delay, 0

    def wait(self, timeout=None):
        time.sleep(self.delay)
        self.waited += 1
        return True

    def is_completed(self):
        return self.waited > 0


class Group:
    """GroupCoordinator leaf: the attributes and the two PP methods gpu_worker.Worker calls."""

    def __init__(self, rank_in_group, world_size, recv_block=0.0, wait=0.0):
        self.rank_in_group, self.world_size = rank_in_group, world_size
        self.is_first_rank = rank_in_group == 0
        self.is_last_rank = rank_in_group == world_size - 1
        self.recv_block, self.wait = recv_block, wait
        self._pending_isends = []
        self.sent = []

    def irecv_tensor_dict(self, src=None, all_gather_group=None, all_gather_tensors=None):
        # recv_object(src) blocks the host until the previous stage's metadata arrives.
        time.sleep(self.recv_block)
        tensors = {'hidden_states': 'h', 'residual': 'r'}

        def postprocess():
            tensors['residual'] = 'gathered'
        return tensors, [Work(self.wait), Work(0.0)], [postprocess]

    def isend_tensor_dict(self, tensor_dict, dst=None, all_gather_group=None, all_gather_tensors=None):
        handles = [Work(0.0), Work(self.wait), Work(0.0)]
        self._pending_isends.append(handles)
        self.sent.append(tensor_dict)
        return handles


class ParallelState:
    """vllm.distributed.parallel_state leaf: one rank's groups."""

    def __init__(self, pp, tp):
        self.pp, self.tp = pp, tp

    def module(self):
        m = types.ModuleType('vllm.distributed.parallel_state')
        m.get_pp_group = lambda: self.pp
        m.get_tp_group = lambda: self.tp
        return m


# --- leaf worker and model runners ---------------------------------------------

class Tensors:
    def __init__(self, tensors):
        self.tensors = tensors


class AsyncTensors:
    """gpu_worker.AsyncIntermediateTensors leaf: the handles are waited when ``tensors`` is first read."""

    def __init__(self, tensors, comm_handles, comm_postprocess):
        self._tensors, self._handles, self._post, self._waited = tensors, comm_handles, comm_postprocess, False

    @property
    def tensors(self):
        if not self._waited:
            for handle in self._handles:
                handle.wait()
            for fn in self._post:
                fn()
            self._waited = True
        return self._tensors


class Output:
    """ModelRunnerOutput leaf."""

    def __init__(self, tokens):
        self.tokens = tokens


class AsyncOutput(AsyncModelRunnerOutput):
    """AsyncGPUModelRunnerOutput / gpu.async_utils.AsyncOutput leaf: get_output blocks on the copy event."""

    def __init__(self, tokens, copy_wait):
        self.tokens, self.copy_wait = tokens, copy_wait

    def get_output(self):
        time.sleep(self.copy_wait)  # stands for copy_event.synchronize(), vLLM's own sync
        return Output(self.tokens)


class StopOutput(AsyncModelRunnerOutput):
    def get_output(self):
        raise Stop()


class Model:
    def __init__(self, cost=0.0005):
        self.cost = cost

    def compute_logits(self, hidden_states):
        spin(self.cost)
        return 'logits'


class Costs:
    prep = 0.002          # CPU: input preparation
    launch = 0.001        # CPU: forward launch
    forward_dev = 10 * MS  # device: decode forward
    sample = 0.001        # CPU: sampler launch
    sample_dev = 2 * MS    # device: sampling
    book = 0.0005         # CPU: bookkeeping (async)
    tolist = 0.004        # blocked: bookkeeping .tolist() sync (async off)
    tok_recv = 0.003      # blocked: stage 0 sampled-token receive
    copy_wait = 0.004     # blocked: get_output copy sync
    recv_block = 0.008    # blocked: PP metadata receive
    handle_wait = 0.002   # blocked: PP comm handle wait


class Fast(Costs):
    prep = launch = sample = book = tolist = tok_recv = copy_wait = recv_block = handle_wait = 0.0


class RunnerV1:
    """gpu_model_runner.GPUModelRunner leaf: execute_model -> _model_forward (-> handle waits) -> compute_logits;
    sample_tokens -> _sample -> PP token broadcast -> _bookkeeping_sync (reference lines 4187-4866)."""

    def __init__(self, torch, pp, async_scheduling, costs=Costs):
        self.torch, self.pp, self.use_async_scheduling, self.c = torch, pp, async_scheduling, costs
        self.model = Model()
        self.execute_model_state = None
        self.fail_forward = False
        self.calls = []

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        if self.execute_model_state is not None:
            raise RuntimeError('State error: sample_tokens() must be called after execute_model() returns None.')
        self.calls.append('execute_model')
        spin(self.c.prep)
        hidden = self._model_forward(input_ids=None, positions=None, intermediate_tensors=intermediate_tensors)
        if not self.pp.is_last_rank:
            return Tensors(hidden)
        logits = self.model.compute_logits(hidden)
        self.execute_model_state = (scheduler_output, logits)
        return None

    def _model_forward(self, input_ids=None, positions=None, intermediate_tensors=None, inputs_embeds=None, **kw):
        self.calls.append('_model_forward')
        if intermediate_tensors is not None:
            intermediate_tensors.tensors
        if self.fail_forward:
            raise RuntimeError('fixture forward failure')
        spin(self.c.launch)
        self.torch.stream.launch(self.c.forward_dev)
        return {'hidden_states': 'h'}

    def sample_tokens(self, grammar_output):
        self.calls.append('sample_tokens')
        if self.execute_model_state is None:
            if self.use_async_scheduling and not self.pp.is_last_rank:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            return Output(None)
        scheduler_output, logits = self.execute_model_state
        self.execute_model_state = None
        sampled = self._sample(logits, None)
        if self.use_async_scheduling and self.pp.world_size > 1 and self.pp.is_last_rank:
            self._pp_broadcast_prev_sampled_token_ids(sampled)
        tokens = self._bookkeeping_sync(scheduler_output, sampled, logits, None, 1)
        if not self.use_async_scheduling:
            return Output(tokens)
        return AsyncOutput(tokens, self.c.copy_wait)

    def _sample(self, logits, spec_decode_metadata):
        self.calls.append('_sample')
        spin(self.c.sample)
        self.torch.stream.launch(self.c.sample_dev)
        return 'sampled'

    def _bookkeeping_sync(self, scheduler_output, sampler_output, logits, hidden_states, num_scheduled_tokens):
        self.calls.append('_bookkeeping_sync')
        if self.use_async_scheduling:
            spin(self.c.book)
        else:
            time.sleep(self.c.tolist)  # _to_list: the GPU->CPU sync of sync scheduling
        return [[1]]

    def _pp_broadcast_prev_sampled_token_ids(self, sampled_token_ids):
        self.calls.append('_pp_broadcast_prev_sampled_token_ids')
        spin(0.0002)

    def _pp_receive_prev_sampled_token_ids_to_input_batch(self):
        self.calls.append('_pp_receive_prev_sampled_token_ids_to_input_batch')
        time.sleep(self.c.tok_recv)


class GraphManager:
    """gpu/cudagraph_utils.CudaGraphManager leaf: run_fullgraph replays on the current stream."""

    def __init__(self, torch, costs):
        self.torch, self.c = torch, costs

    def run_fullgraph(self, batch_desc):
        spin(self.c.launch)
        self.torch.stream.launch(self.c.forward_dev)
        return 'hidden'

    def run_pw_graph(self, model, model_inputs):
        raise AssertionError('fixture decode runs FULL graphs')


class PPHandler:
    """gpu/pp_utils.PPHandler leaf."""

    def __init__(self, costs):
        self.c = costs

    def receive(self, input_batch):
        time.sleep(self.c.tok_recv)
        return True

    def broadcast(self, sampled_token_ids, num_sampled, num_rejected, input_batch):
        spin(0.0002)


class ModelState:
    def prepare_attn(self, *args, **kwargs):
        spin(0.0005)
        return 'attn'


class RunnerV2:
    """gpu/model_runner.GPUModelRunner leaf: execute_model -> update/add_requests -> prepare_inputs -> prepare_attn ->
    model_state.prepare_attn -> intermediate copy (handle waits) -> cudagraph_manager.run_fullgraph; sample_tokens ->
    sample (compute_logits) -> pp_handler.broadcast -> postprocess_sampled (reference lines 1785-2325)."""

    def __init__(self, torch, pp, async_scheduling, costs=Costs):
        self.torch, self.pp, self.use_async_scheduling, self.c = torch, pp, async_scheduling, costs
        self.model = Model()
        self.cudagraph_manager = GraphManager(torch, costs)
        self.pp_handler = PPHandler(costs) if pp.world_size > 1 else None
        self.model_state = ModelState()
        self.execute_model_state = None

    def update_requests(self, scheduler_output):
        spin(0.0003)

    def add_requests(self, scheduler_output):
        spin(0.0002)

    def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
        spin(self.c.prep / 2)
        return 'input_batch'

    def prepare_attn(self, input_batch):
        spin(0.0003)
        return 'block_tables', 'slot_mappings'

    def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False, **kw):
        self.add_requests(scheduler_output)
        self.update_requests(scheduler_output)
        input_batch = self.prepare_inputs(scheduler_output, None, None)
        self.prepare_attn(input_batch)
        self.model_state.prepare_attn(input_batch)
        if not self.pp.is_first_rank:
            intermediate_tensors.tensors
        hidden = self.cudagraph_manager.run_fullgraph('batch_desc')
        self.execute_model_state = (input_batch, hidden)
        if not self.pp.is_last_rank:
            return Tensors({'hidden_states': hidden})
        return None

    def sample(self, hidden_states, input_batch, grammar_output):
        self.model.compute_logits(hidden_states)
        spin(self.c.sample)
        self.torch.stream.launch(self.c.sample_dev)
        return 'sampler_output', 1, 0

    def postprocess_sampled(self, idx_mapping, sampled_token_ids, num_sampled, num_rejected, query_start_loc):
        spin(0.0002)

    def sample_tokens(self, grammar_output):
        if self.execute_model_state is None:
            return None
        input_batch, hidden = self.execute_model_state
        self.execute_model_state = None
        if not self.pp.is_last_rank:
            self.pp_handler.receive(input_batch)
            return Output(None)
        sampler_output, num_sampled, num_rejected = self.sample(hidden, input_batch, None)
        if self.pp_handler is not None:
            self.pp_handler.broadcast(sampler_output, num_sampled, num_rejected, input_batch)
        out = AsyncOutput([[1]], self.c.copy_wait)
        self.postprocess_sampled(None, sampler_output, num_sampled, num_rejected, None)
        return out


class Worker:
    """gpu_worker.Worker leaf: execute_model/sample_tokens in the reference order (lines 1121-1220)."""

    def __init__(self, state, runner):
        self.state, self.model_runner = state, runner
        self._pp_send_work = []

    def execute_model(self, scheduler_output):
        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []
        intermediate_tensors = None
        pp = self.state.pp
        if not pp.is_first_rank:
            tensor_dict, comm_handles, comm_postprocess = pp.irecv_tensor_dict(
                all_gather_group=self.state.tp, all_gather_tensors={})
            intermediate_tensors = AsyncTensors(tensor_dict, comm_handles, comm_postprocess)
        output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
        if not isinstance(output, Tensors):
            return output
        handles = pp.isend_tensor_dict(output.tensors, all_gather_group=self.state.tp, all_gather_tensors={})
        self._pp_send_work = handles[1:]
        return None

    def sample_tokens(self, grammar_output):
        return self.model_runner.sample_tokens(grammar_output)

    def take_draft_token_ids(self):
        return None

    def flash_s5_release(self, message):
        return None


class Writer:
    """Worker response MQ leaf: records each enqueued result."""

    def __init__(self):
        self.results = []

    def enqueue(self, result):
        self.results.append(result)


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class HookClock:
    """Deterministic ``time`` for executor_hooks only: progress stamps become a pure function of the call order."""

    def __init__(self):
        self.ns = 10 ** 12

    def monotonic_ns(self):
        self.ns += 1000
        return self.ns

    def __getattr__(self, name):
        return getattr(time, name)


# --- the rig: one worker process (rank) ----------------------------------------

def stage_of(rank):
    return rank // 2


class Rig:
    """One rank: generated WorkerProc + leaf worker/runner, real hooks and recorder, captured log lines."""

    def __init__(self, stack, rank, *, anatomy=True, async_scheduling=True, runner='v1', hooks=eh, clock=None,
                 costs=Costs):
        self.rank, self.hooks = rank, hooks
        self.torch = LeafTorch()
        pp = Group(stage_of(rank), 2, recv_block=costs.recv_block, wait=costs.handle_wait)
        tp = Group(rank % 2, 2)
        self.state = ParallelState(pp, tp)
        env = {'FLASH_S5_NATIVE_PROGRESS': 'burst', 'FLASH_S5_AUTO_ARM': '1'}
        if anatomy:
            env['FLASH_S5_HANDLER_ANATOMY'] = '1'
        stack.enter_context(mock.patch.dict(os.environ, env))
        if not anatomy:
            os.environ.pop('FLASH_S5_HANDLER_ANATOMY', None)
        vllm = types.ModuleType('vllm')
        vllm.__path__ = []
        distributed = types.ModuleType('vllm.distributed')
        distributed.__path__ = []
        platforms = types.ModuleType('vllm.platforms')
        platforms.current_platform = types.SimpleNamespace(set_device=lambda device: None)
        stack.enter_context(mock.patch.dict(sys.modules, {
            'vllm': vllm, 'vllm.distributed': distributed, 'vllm.distributed.parallel_state': self.state.module(),
            'vllm.platforms': platforms}))
        # One worker process per rig: the process-global hook state starts unread and is left unread.
        stack.enter_context(mock.patch.object(hooks, '_WORKER_PROGRESS', None))
        if hasattr(hooks, '_ANATOMY'):
            stack.enter_context(mock.patch.object(hooks, '_ANATOMY', None))
        if hasattr(ha, '_load_torch'):
            stack.enter_context(mock.patch.object(ha, '_load_torch', lambda: self.torch))
        if clock is not None:
            stack.enter_context(mock.patch.object(hooks, 'time', clock))
        logger = hooks._progress_logger()
        self.capture = Capture()
        logger.addHandler(self.capture)
        level = logger.level
        logger.setLevel(logging.INFO)
        stack.callback(logger.setLevel, level)
        stack.callback(logger.removeHandler, self.capture)
        cls = RunnerV1 if runner == 'v1' else RunnerV2
        self.runner = cls(self.torch, pp, async_scheduling, costs)
        self.worker = Worker(self.state, self.runner)
        proc = proc_class(hooks)()
        proc.rank = rank
        proc.worker = self.worker
        proc.use_async_scheduling = async_scheduling
        proc.async_output_queue = Queue()
        proc.worker_response_mq = Writer()
        self.proc = proc
        self.thread = None
        if async_scheduling:
            self.thread = threading.Thread(target=self._async_loop, daemon=True)
            self.thread.start()
            stack.callback(self.close)
        self.seq = 100

    def _async_loop(self):
        try:
            self.proc.async_output_busy_loop()
        except Stop:
            pass

    def close(self):
        if self.thread is not None and self.thread.is_alive():
            self.proc.async_output_queue.put(StopOutput())
            self.thread.join(5)

    def rpc(self, method, args=(), output_rank=OUTPUT_RANK, meta=True):
        self.seq += 1
        kwargs = {nm.PROGRESS_META_KEY: nm.encode_progress_meta(self.seq, False)} if meta else {}
        self.proc._execute_worker_rpc((method, args, kwargs, output_rank))
        return self.seq

    def step(self, meta=True):
        """One decode step: execute_model then sample_tokens, both replying from the output rank."""
        return self.rpc('execute_model', ('so',), meta=meta), self.rpc('sample_tokens', (None,), meta=meta)

    def replies(self, n, timeout=5):
        end = time.monotonic() + timeout
        while len(self.proc.worker_response_mq.results) < n and time.monotonic() < end:
            time.sleep(0.001)
        return list(self.proc.worker_response_mq.results)

    def release(self, advance=True):
        if advance:
            self.torch.device.advance()
        self.rpc('flash_s5_release', (None,), output_rank=None)

    def handler_lines(self):
        return [m for m in self.capture.messages if m.startswith(HANDLER_PREFIX)]

    def progress_lines(self):
        return [m for m in self.capture.messages if m.startswith(nm.PROGRESS_LINE)]

    @property
    def recorder(self):
        return self.hooks._ANATOMY or None


def parse(lines):
    """Reassemble handler lines (all parts) into (heads, main rows, output rows) by field name."""
    by_line = {}
    for text in lines:
        body = json.loads(text[len(ha.LINE_PREFIX):])
        by_line.setdefault((body['rank'], body['line']), []).append(body)
    heads, main, out = [], [], []
    for key, parts in sorted(by_line.items()):
        parts.sort(key=lambda b: b['part'])
        assert [b['part'] for b in parts] == list(range(parts[0]['parts'])), key
        head = parts[0]['head']
        heads.append(head)
        for body in parts:
            base = body['base_ns']
            for row in body['rows'].get('m', []):
                r = dict(zip(ha.M_FIELDS, row))
                r['seams'] = {ha.SEAMS[c[0]]: dict(zip(ha.SEAM_FIELDS[1:], c[1:])) for c in r['seams']}
                r['base_ns'] = base
                main.append(r)
            for row in body['rows'].get('a', []):
                r = dict(zip(ha.A_FIELDS, row))
                r['base_ns'] = base
                out.append(r)
    return heads, main, out


def instance_state(rig):
    """What the off proof compares: every instance dict the recorder could touch, and their classes' dicts."""
    objs = dict(proc=rig.proc, worker=rig.worker, runner=rig.runner, model=rig.runner.model, pp=rig.state.pp,
                tp=rig.state.tp, mq=rig.proc.worker_response_mq, queue=rig.proc.async_output_queue)
    for name in ('cudagraph_manager', 'pp_handler', 'model_state'):
        if getattr(rig.runner, name, None) is not None:
            objs[name] = getattr(rig.runner, name)
    snap = {}
    for name, obj in objs.items():
        snap[name] = {k: id(v) for k, v in vars(obj).items() if k not in ('_pending_isends', 'sent', 'results',
                                                                          '_flash_s5_reply_lock',
                                                                          'calls', 'execute_model_state',
                                                                          '_pp_send_work', 'queue', 'mutex',
                                                                          'not_empty', 'not_full', 'all_tasks_done',
                                                                          'unfinished_tasks')}
        snap[name + ':class'] = sorted(vars(type(obj)))
    return snap


class Case(unittest.TestCase):
    def rig(self, rank, **kwargs):
        stack = ExitStack()
        self.addCleanup(stack.close)
        return Rig(stack, rank, **kwargs)


# --- scenarios -----------------------------------------------------------------

class Off(Case):
    def test_switch_unset_installs_nothing_and_writes_no_handler_line(self):
        rig = self.rig(OUTPUT_RANK, anatomy=False)
        before = instance_state(rig)
        for _ in range(3):
            rig.step()
        rig.replies(6)
        rig.release()
        self.assertEqual(rig.handler_lines(), [])
        self.assertFalse(rig.hooks._ANATOMY)
        self.assertEqual(instance_state(rig), before)
        self.assertTrue(rig.progress_lines())
        self.assertEqual(rig.torch.device.calls, [])

    def test_off_in_a_fresh_process_never_imports_handler_anatomy(self):
        result = offrun_subprocess('head', anatomy=False)
        self.assertFalse(result['imported'], result)
        self.assertTrue(result['unchanged'], result)
        self.assertEqual(result['handler_lines'], 0)
        self.assertGreater(len(result['progress']), 0)

    def test_progress_lines_byte_identical_to_step_b_off_and_on(self):
        reference = offrun_subprocess('base', anatomy=False)
        off = offrun_subprocess('head', anatomy=False)
        on = offrun_subprocess('head', anatomy=True)
        self.assertEqual(off['progress'], reference['progress'])
        self.assertEqual(on['progress'], reference['progress'])
        self.assertEqual(off['replies'], reference['replies'])
        self.assertEqual(on['replies'], reference['replies'])
        self.assertGreater(on['handler_lines'], 0)
        self.assertTrue(on['imported'])


class Stage1OutputRankAsync(Case):
    def setUp(self):
        self.r = self.rig(2)
        self.seqs = [self.r.step() for _ in range(3)]
        self.r.replies(6)
        self.r.release()
        self.heads, self.main, self.out = parse(self.r.handler_lines())

    def test_rows_seams_and_device_times(self):
        self.assertEqual(len(self.heads), 1)
        head = self.heads[0]
        self.assertEqual(head['identity']['rank'], 2)
        self.assertEqual(head['identity']['pp_rank'], 1)
        self.assertTrue(head['identity']['is_last_rank'])
        self.assertTrue(head['identity']['async_scheduling'])
        rows = {r['seq']: r for r in self.main}
        self.assertEqual(sorted(rows), sorted(s for pair in self.seqs for s in pair))
        for execute, sample in self.seqs:
            e, s = rows[execute], rows[sample]
            self.assertEqual((e['kind'], s['kind'], e['out'], s['out']), ('e', 's', 1, 1))
            self.assertEqual(set(e['seams']), {'pp_recv', 'pp_recv_wait', 'pp_recv_post', 'execute', 'forward', 'logits'})
            self.assertEqual(set(s['seams']), {'sample_tokens', 'sample', 'pp_tok_send', 'bookkeeping'})
            # PP metadata receive: blocked (sleep), low CPU.
            recv = e['seams']['pp_recv']
            self.assertGreaterEqual(recv['ns'], Costs.recv_block * 1e9)
            self.assertLess(recv['cpu_ns'], recv['ns'] / 2)
            # Handle waits nest in the forward: the forward's exclusive time excludes them.
            fwd = e['seams']['forward']
            self.assertGreaterEqual(e['seams']['pp_recv_wait']['ns'], Costs.handle_wait * 1e9)
            self.assertLessEqual(fwd['excl_ns'], fwd['ns'] - e['seams']['pp_recv_wait']['ns'])
            self.assertGreaterEqual(fwd['cpu_ns'], Costs.launch * 1e9 * 0.9)
            # Device times, read lazily after both events queried True.
            self.assertEqual(e['dev_forward_ns'], Costs.forward_dev)
            self.assertEqual(s['dev_sample_ns'], Costs.sample_dev)
            # The device clock stands still until the release: only the first launch finds the stream idle.
            self.assertEqual((e['idle_before_launch'], e['idle_after_launch']), (int(execute == self.seqs[0][0]), 0))
            # Parts add up: exclusive seams + observer never exceed the wall.
            for r in (e, s):
                excl = sum(c['excl_ns'] for c in r['seams'].values())
                self.assertLessEqual(excl + r['observer_ns'], r['wall_ns'] + 50000)
                self.assertGreater(r['observer_ns'], 0)
                self.assertLessEqual(r['cpu_ns'], r['wall_ns'] + 1000000)
        self.assertEqual(self.r.torch.device.violations, [])
        self.assertEqual(head['counters']['observer_errors'], 0)
        self.assertTrue(set(self.r.torch.device.calls) <= {'Event.record', 'Event.query', 'Event.elapsed_time',
                                                            'Stream.query'})

    def test_async_output_rows_keyed_by_sequence(self):
        outs = {r['seq']: r for r in self.out}
        for execute, sample in self.seqs:
            self.assertIn(sample, outs)
            a = outs[sample]
            self.assertEqual(a['thread'], 0)
            self.assertGreaterEqual(a['get_output_ns'], Costs.copy_wait * 1e9)
            self.assertLess(a['get_output_cpu_ns'], a['get_output_ns'])
            self.assertIsNotNone(a['mq_write_ns'])
            self.assertIsNotNone(a['t_take'])
            # execute_model's None reply also goes through the async FIFO: an output row, no get_output.
            self.assertIn(execute, outs)
            self.assertIsNone(outs[execute]['get_output_ns'])
        rows = {r['seq']: r for r in self.main}
        self.assertTrue(all(rows[s]['handle_output_ns'] is not None for pair in self.seqs for s in pair))
        self.assertEqual(self.heads[0]['counters']['async_unmatched'], 0)

    def test_observer_cost_and_calibration_in_header(self):
        cal = self.heads[0]['calibration']
        self.assertEqual((cal['probes'], cal['event_pairs']), (ha.CAL_PROBES, ha.CAL_EVENT_PAIRS))
        self.assertGreater(cal['probe_ns'], 0)
        self.assertGreaterEqual(cal['event_pair_ns'], 0)
        installed = set(self.heads[0]['installed'])
        self.assertTrue({'pp.irecv_tensor_dict', 'runner.execute_model', 'runner._model_forward', 'runner._sample',
                         'runner._bookkeeping_sync', 'model.compute_logits', 'proc.worker_response_mq.enqueue',
                         'proc.async_output_queue.get'} <= installed, installed)
        self.assertEqual(sum(1 for r in self.main if r['flags'] & ha.FLAG_INSTALL), 1)

    def test_wrappers_are_on_instances_never_classes(self):
        for obj in (self.r.runner, self.r.runner.model, self.r.state.pp, self.r.proc.worker_response_mq):
            cls = type(obj)
            for name, value in vars(obj).items():
                if callable(value) and name in vars(cls):
                    self.assertIsNot(value, vars(cls)[name])
            for name in ('execute_model', 'sample_tokens', '_model_forward', 'irecv_tensor_dict', 'compute_logits',
                         'enqueue'):
                if name in vars(cls):
                    self.assertTrue(getattr(vars(cls)[name], '__qualname__', '').startswith(cls.__name__ + '.'),
                                    'class attribute replaced: %s.%s' % (cls.__name__, name))
        self.assertIn('_model_forward', vars(self.r.runner))
        self.assertIn('irecv_tensor_dict', vars(self.r.state.pp))
        # A second recorder is never installed: the process global stays the first.
        first = self.r.hooks._ANATOMY
        self.r.step()
        self.assertIs(self.r.hooks._ANATOMY, first)


class Stage1NonOutputRank(Case):
    def test_no_reply_path_rows(self):
        r = self.rig(3)
        seqs = [r.step() for _ in range(2)]
        r.release()
        heads, main, out = parse(r.handler_lines())
        self.assertEqual(out, [])
        self.assertEqual(sorted(x['seq'] for x in main), sorted(s for p in seqs for s in p))
        self.assertTrue(all(x['out'] == 0 and x['handle_output_ns'] is None for x in main))
        self.assertEqual(heads[0]['identity']['tp_rank'], 1)
        self.assertTrue(all('pp_recv' in x['seams'] for x in main if x['kind'] == 'e'))


class Stage0Sends(Case):
    def test_send_handles_waited_next_step_and_token_receive(self):
        r = self.rig(0)
        seqs = [r.step() for _ in range(3)]
        r.release()
        heads, main, out = parse(r.handler_lines())
        rows = {x['seq']: x for x in main}
        self.assertFalse(heads[0]['identity']['is_last_rank'])
        for index, (execute, sample) in enumerate(seqs):
            e, s = rows[execute], rows[sample]
            self.assertIn('pp_send', e['seams'])
            self.assertNotIn('pp_recv', e['seams'])
            self.assertNotIn('logits', e['seams'])
            # The previous step's send handles are waited at the top of this one (worker.execute_model).
            if index:
                self.assertEqual(e['seams']['pp_send_wait']['n'], 2)
                self.assertGreaterEqual(e['seams']['pp_send_wait']['ns'], Costs.handle_wait * 1e9)
            else:
                self.assertNotIn('pp_send_wait', e['seams'])
            self.assertEqual(set(s['seams']), {'sample_tokens', 'pp_tok_recv'})
            self.assertGreaterEqual(s['seams']['pp_tok_recv']['ns'], Costs.tok_recv * 1e9)
        self.assertEqual(out, [])
        # The worker's pending handle list holds proxies that delegate to the real handles.
        self.assertTrue(all(isinstance(h, ha._Handle) for h in r.worker._pp_send_work))
        self.assertTrue(all(isinstance(h, Work) for h in r.state.pp._pending_isends[-1]))


class AsyncOff(Case):
    def test_sync_bookkeeping_and_main_thread_reply(self):
        r = self.rig(2, async_scheduling=False)
        seqs = [r.step() for _ in range(2)]
        r.release()
        heads, main, out = parse(r.handler_lines())
        self.assertFalse(heads[0]['identity']['async_scheduling'])
        self.assertNotIn('proc.async_output_queue.get', heads[0]['installed'])
        rows = {x['seq']: x for x in main}
        outs = {x['seq']: x for x in out}
        for execute, sample in seqs:
            book = rows[sample]['seams']['bookkeeping']
            self.assertGreaterEqual(book['ns'], Costs.tolist * 1e9)
            self.assertLess(book['cpu_ns'], book['ns'] / 2)
            self.assertNotIn('pp_tok_send', rows[sample]['seams'])
            a = outs[sample]
            self.assertEqual(a['thread'], 1)
            self.assertIsNone(a['get_output_ns'])
            self.assertIsNone(a['t_take'])
            self.assertGreaterEqual(rows[sample]['handle_output_ns'], a['mq_write_ns'])
        # Four step replies and the release's own (output_rank None).
        self.assertEqual(len(r.proc.worker_response_mq.results), 5)


class V2Runner(Case):
    def test_v2_names_fire(self):
        r = self.rig(2, runner='v2')
        seqs = [r.step() for _ in range(2)]
        r.replies(4)
        r.release()
        heads, main, out = parse(r.handler_lines())
        rows = {x['seq']: x for x in main}
        installed = set(heads[0]['installed'])
        self.assertTrue({'cgm.run_fullgraph', 'runner.sample', 'pph.broadcast', 'pph.receive', 'runner.prepare_inputs',
                         'mstate.prepare_attn', 'runner.postprocess_sampled'} <= installed, installed)
        self.assertIn('runner._model_forward', heads[0]['missing'])
        for execute, sample in seqs:
            self.assertEqual(set(rows[execute]['seams']), {'pp_recv', 'pp_recv_wait', 'pp_recv_post', 'execute',
                                                           'prep_states', 'prep_inputs', 'prep_attn', 'prep_attn_meta',
                                                           'forward'})
            self.assertEqual(rows[execute]['seams']['prep_states']['n'], 2)
            self.assertEqual(set(rows[sample]['seams']), {'sample_tokens', 'sample', 'logits', 'pp_tok_send',
                                                          'post_sampled'})
            self.assertEqual(rows[execute]['dev_forward_ns'], Costs.forward_dev)
            self.assertGreaterEqual({x['seq']: x for x in out}[sample]['get_output_ns'], Costs.copy_wait * 1e9)


class ForwardFailure(Case):
    def test_exception_path_unchanged_and_state_reset(self):
        r = self.rig(2)
        good = r.step()
        r.runner.fail_forward = True
        failed = r.rpc('execute_model', ('so',))
        r.runner.fail_forward = False
        rec = r.recorder
        self.assertIsNone(rec.row)
        self.assertEqual(rec.stack, [])
        self.assertEqual(rec.active, [0] * len(ha.SEAMS))
        after = r.step()
        results = r.replies(5)
        # The failure wire is the stock one: FAILURE with the exception's text, in FIFO order.
        self.assertEqual(results[2], (Status.FAILURE, 'fixture forward failure'))
        self.assertEqual([x[0] for x in results], [Status.SUCCESS] * 2 + [Status.FAILURE] + [Status.SUCCESS] * 2)
        r.release()
        heads, main, out = parse(r.handler_lines())
        rows = {x['seq']: x for x in main}
        self.assertTrue(rows[failed]['flags'] & ha.FLAG_FAILED)
        self.assertIn('pp_recv', rows[failed]['seams'])
        self.assertIn('forward', rows[failed]['seams'])
        self.assertIsNotNone(rows[failed]['wall_ns'])
        for seq in good + after:
            self.assertFalse(rows[seq]['flags'] & (ha.FLAG_FAILED | ha.FLAG_ABORTED))
        self.assertEqual(heads[0]['counters']['failed_rows'], 1)
        self.assertEqual(r.torch.device.violations, [])

    def test_exception_on_a_non_output_rank_closes_as_aborted_at_the_next_rpc(self):
        r = self.rig(3)
        r.runner.fail_forward = True
        failed = r.rpc('execute_model', ('so',))
        r.runner.fail_forward = False
        r.step()
        r.release()
        heads, main, out = parse(r.handler_lines())
        rows = {x['seq']: x for x in main}
        self.assertTrue(rows[failed]['flags'] & ha.FLAG_ABORTED)
        self.assertEqual(heads[0]['counters']['aborted_rows'], 1)


class Bounds(Case):
    def test_pool_rows_and_unresolved_at_flush(self):
        with mock.patch.object(ha, 'EVENT_PAIRS', 2), mock.patch.object(ha, 'MAX_ROWS', 5):
            r = self.rig(2)
            for _ in range(4):
                r.step()
            r.replies(8)
            r.release(advance=False)          # nothing resolved: device clock never advanced
            heads, main, out = parse(r.handler_lines())
            c = heads[0]['counters']
            self.assertEqual(len(main) + len(out), 5)
            self.assertEqual(c['rows_dropped'], 16 - 5)
            self.assertEqual(c['events_pool_empty'], 8 - 2)
            self.assertEqual(c['events_unresolved_at_flush'], 2)
            self.assertTrue(all(x['dev_forward_ns'] is None and x['dev_sample_ns'] is None for x in main))
            self.assertEqual(len(r.recorder.pending), 2)
            self.assertEqual(r.torch.device.violations, [])
            # Later the pairs resolve without a wait and return to the pool.
            r.torch.device.advance()
            r.rpc('execute_model', ('so',))
            self.assertEqual(len(r.recorder.pool), 1)   # 2 returned, 1 taken by this forward
            self.assertEqual(r.recorder.counters['events_late'], 2)
            self.assertEqual(r.torch.device.violations, [])

    def test_line_size_bound(self):
        r = self.rig(2, costs=Fast)
        for _ in range(60):
            r.step()
        r.replies(120)
        r.release()
        lines = r.handler_lines()
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(len(x.encode()) < nm.PROGRESS_LINE_BYTES for x in lines))
        heads, main, out = parse(lines)
        self.assertEqual(len(main), 120)
        self.assertEqual(len(out), 120)


def method_text(text, head):
    start = text.index(head)
    end = text.find('\n    def ', start + len(head))
    return text[start:end if end > 0 else len(text)]


class LeafNames(unittest.TestCase):
    """Every wrapped name and every leaf method name exists in the pinned vLLM source with the same owner."""

    OWNERS = {
        'runner': [('v1/worker/gpu_model_runner.py', 'GPUModelRunner'), ('v1/worker/gpu/model_runner.py',
                                                                          'GPUModelRunner')],
        'pp': [('distributed/parallel_state.py', 'GroupCoordinator')],
        'cgm': [('v1/worker/gpu/cudagraph_utils.py', None)],
        'split_cgm': [('v1/worker/gpu/cudagraph_utils.py', None)],
        'pph': [('v1/worker/gpu/pp_utils.py', 'PPHandler')],
        'mstate': [('v1/worker/gpu/model_states/default.py', None), ('v1/worker/gpu/model_states/interface.py', None)],
        'model': [('models/qwen4_exp/nvidia/model.py', None), ('model_executor/models/qwen3_5.py', None)],
    }

    def defs(self, path, cls):
        file = REFERENCE / path
        if not file.is_file():
            return None
        tree = ast.parse(file.read_text())
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and (cls is None or n.name == cls)]
        return {f.name for n in nodes for f in n.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def test_targets_exist_in_reference(self):
        if not REFERENCE.is_dir():
            self.skipTest('reference/vllm-src not available')
        for owner, attr, seam, mode in ha.TARGETS:
            found = [p for p, c in self.OWNERS[owner] if attr in (self.defs(p, c) or ())]
            self.assertTrue(found, '%s.%s not in %s' % (owner, attr, self.OWNERS[owner]))

    def test_worker_order_matches_reference(self):
        text = (REFERENCE / 'v1/worker/gpu_worker.py').read_text()
        body = method_text(text, '    def execute_model(\n        self, scheduler_output: "SchedulerOutput"')
        order = [body.index(s) for s in ('handle.wait()', 'irecv_tensor_dict(', 'self.model_runner.execute_model(',
                                         'isend_tensor_dict(', 'self._pp_send_work = handles[1:]')]
        self.assertEqual(order, sorted(order))
        v1 = (REFERENCE / 'v1/worker/gpu_model_runner.py').read_text()
        body = method_text(v1, '    def sample_tokens(\n')
        order = [body.index(x) for x in ('self._sample(', '_pp_broadcast_prev_sampled_token_ids(',
                                         'self._bookkeeping_sync(', 'AsyncGPUModelRunnerOutput(')]
        self.assertEqual(order, sorted(order))
        v2 = (REFERENCE / 'v1/worker/gpu/model_runner.py').read_text()
        body = method_text(v2, '    def sample_tokens(\n')
        order = [body.index(x) for x in ('self.sample(', 'self.pp_handler.broadcast(', 'AsyncOutput(',
                                         'self.postprocess_sampled(')]
        self.assertEqual(order, sorted(order))


class FreezeProof(unittest.TestCase):
    """tools/stage1_anatomy_freeze.py over this tree: the source proof passes and every negative control refuses."""

    def test_source_proof_and_negative_controls(self):
        base_hooks_source()
        import stage1_anatomy_freeze as freeze
        source = freeze.source_level()
        self.assertEqual(source['default_off']['verdict'], 'PASS')
        self.assertEqual(source['default_off']['functions'], 5)
        self.assertEqual(source['default_off']['removal_count'], 6)
        self.assertEqual(source['imports']['site'], 'executor_hooks.py:_anatomy')
        self.assertEqual(set(freeze.negative_controls().values()), {'REFUSED'})


class Mutants(unittest.TestCase):
    """The scenarios discriminate: each mutant makes its scenario fail on the rule (not by a broken fixture)."""

    LEAF_CLASSES = (RunnerV1, RunnerV2, Model, Group, Writer, GraphManager, PPHandler, ModelState, Work, Worker)

    def fails(self, scenario, *patches):
        type(scenario).setUpClass()
        saved = {cls: dict(vars(cls)) for cls in self.LEAF_CLASSES}
        try:
            with ExitStack() as stack:
                for target, name, value in patches:
                    stack.enter_context(mock.patch.object(target, name, value))
                result = unittest.TestResult()
                scenario.run(result)
        finally:
            for cls, attrs in saved.items():
                for name in set(vars(cls)) - set(attrs):
                    delattr(cls, name)
                for name, value in attrs.items():
                    if name not in ('__dict__', '__weakref__') and vars(cls).get(name) is not value:
                        setattr(cls, name, value)
        self.assertEqual(result.testsRun, 1)
        self.assertTrue(result.failures or result.errors, 'mutant survived: %s' % scenario.id())
        for _, trace in result.failures + result.errors:
            self.assertNotRegex(trace.strip().splitlines()[-1], r'^(TypeError|AttributeError|NameError)', trace)

    def test_removed_seam(self):
        targets = tuple(row for row in ha.TARGETS if row[1] != 'irecv_tensor_dict')
        self.fails(Stage1OutputRankAsync('test_rows_seams_and_device_times'), (ha, 'TARGETS', targets))

    def test_events_read_early(self):
        real = ha.Recorder._resolve

        def early(self):
            for row, slot, pair in list(self.pending):
                try:
                    row[slot] = int(pair[0].elapsed_time(pair[1]) * 1e6)
                except Exception:  # noqa: BLE001 - the mutant hides its own error like the recorder does
                    self.counters['observer_errors'] += 1
            real(self)
        self.fails(Stage1OutputRankAsync('test_rows_seams_and_device_times'), (ha.Recorder, '_resolve', early))

    def test_host_sync_added(self):
        real = ha.Recorder._resolve

        def synced(self):
            for _, _, pair in list(self.pending):
                try:
                    pair[1].synchronize()
                except Exception:  # noqa: BLE001
                    pass
            real(self)
        self.fails(Stage1OutputRankAsync('test_rows_seams_and_device_times'), (ha.Recorder, '_resolve', synced))

    def test_wrapped_on_the_class(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(ha.Recorder.install)))
        hits = 0
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'setattr'
                    and isinstance(node.args[0], ast.Name) and node.args[0].id == 'target'):
                node.args[0] = ast.Call(func=ast.Name('type', ast.Load()), args=[node.args[0]], keywords=[])
                node.args[2] = ast.Call(func=ast.Name('staticmethod', ast.Load()), args=[node.args[2]], keywords=[])
                hits += 1
        self.assertEqual(hits, 1)
        scope = dict(vars(ha))
        exec(compile(ast.fix_missing_locations(tree), 'mutant', 'exec'), scope)
        self.fails(Stage1OutputRankAsync('test_wrappers_are_on_instances_never_classes'),
                   (ha.Recorder, 'install', scope['install']))


# --- default-off / byte-identity subprocess ------------------------------------

def base_hooks_source():
    """executor_hooks.py at claude/step-b (BASE), or skip when git or the commit is unavailable."""
    try:
        return subprocess.check_output(['git', 'show', BASE + ':patches/s5_engine/executor_hooks.py'], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        raise unittest.SkipTest('git base %s not available' % BASE)


def offrun_subprocess(which, anatomy):
    """``which`` is 'head' (this tree's executor_hooks) or 'base' (BASE's, loaded from git in memory)."""
    if which == 'base':
        base_hooks_source()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', STAGE1_HOOKS=which, STAGE1_ANATOMY='1' if anatomy else '0')
    env.pop('FLASH_S5_HANDLER_ANATOMY', None)
    done = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--offrun'], env=env,
                          capture_output=True, text=True, timeout=120)
    if done.returncode:
        raise AssertionError('offrun failed rc=%d\n%s' % (done.returncode, done.stderr[-4000:]))
    return json.loads(done.stdout.strip().splitlines()[-1])


def offrun_main():
    """In a fresh interpreter: one rank, deterministic hook clock, a fixed RPC script; print what the proof reads."""
    if os.environ['STAGE1_HOOKS'] == 'base':
        hooks = types.ModuleType('executor_hooks')
        hooks.__file__ = 'git:%s:patches/s5_engine/executor_hooks.py' % BASE
        sys.modules['executor_hooks'] = hooks
        exec(compile(base_hooks_source(), hooks.__file__, 'exec'), hooks.__dict__)
    else:
        hooks = eh
        assert Path(hooks.__file__).resolve() == ROOT / 'patches/s5_engine/executor_hooks.py', hooks.__file__
    anatomy = os.environ['STAGE1_ANATOMY'] == '1'
    global ha
    if not anatomy:
        # This file imported handler_anatomy at its top: drop it, so "imported" below means this run did.
        sys.modules.pop('handler_anatomy', None)
        ha = types.SimpleNamespace()
    with ExitStack() as stack:
        rig = Rig(stack, OUTPUT_RANK, anatomy=anatomy, hooks=hooks, clock=HookClock(), async_scheduling=False)
        before = instance_state(rig)
        before = instance_state(rig)
        for _ in range(3):
            rig.step()
        rig.rpc('take_draft_token_ids', (), meta=True)
        rig.release()
        rig.step(meta=False)
        rig.release()
        imported = 'handler_anatomy' in sys.modules
        unchanged = instance_state(rig) == before
        result = dict(imported=imported, unchanged=unchanged, handler_lines=len(rig.handler_lines()),
                      progress=rig.progress_lines(),
                      replies=[[x[0], type(x[1]).__name__, getattr(x[1], 'tokens', None)]
                               for x in rig.proc.worker_response_mq.results],
                      hooks_file=hooks.__file__)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    if '--offrun' in sys.argv:
        offrun_main()
    else:
        unittest.main()
