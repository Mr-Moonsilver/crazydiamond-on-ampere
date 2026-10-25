"""CPU-only tests for the E3 native integration (worker + engine startup seams).

No torch, no vLLM, no CUDA, no network.  The real modules under
``patches/s5_engine`` run unmodified; only their process/device boundaries are
faked:

* a fake ``torch`` module provides allocations, events and a stream token;
* fake ``vllm.forward_context`` / ``vllm.v1.worker.workspace`` modules give
  worker startup its UID namespace, prefill context and workspace lanes;
* a fake ``vllm.v1.executor.multiproc_executor`` supplies the actual
  ``WorkerProc.ResponseStatus.SUCCESS`` identity that enabled startup requires;
* four fake GPU workers, each with its own lane runtime, execute the typed RPCs
  through the generated ``executor_hooks.worker_rpc`` entry point, and their
  forwards route through A's real ``execute_with_e3`` / ``LaneRuntime`` path;
* a fake executor's broadcast ring and response MQs feed the real
  ``ReplyTransport`` receiver threads.

These controller fixtures explicitly mock the production readiness refusal;
they are not native startup proof. Within those fixtures: model-runner construction installs and calls the builder,
``EngineCore.__init__`` installs the composition and driver, and the generated
``park_prefill`` / ``before_step_with_batch_queue`` hooks drive the chunk.  GPU
evidence, real capture and production scheduler admission remain owner-run and
are named in the pass report's Open items.
"""
import os
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import async_handoff as ah  # noqa: E402
import engine_driver as ed  # noqa: E402
import executor_hooks as eh  # noqa: E402
import lane_hooks as lh  # noqa: E402
import lane_state as ls  # noqa: E402
import make_s5_overlap as make  # noqa: E402
import native_backend as nb  # noqa: E402
import native_resources as nr  # noqa: E402
import s5_startup as su  # noqa: E402
from e3_types import (  # noqa: E402
    CompletionKind,
    OperationTicket,
    DependencyRole,
    E3ContractError,
    E3Pending,
    TransactionId,
)
from lane_state import BatchRow, MetadataSnapshot  # noqa: E402


# --- fake device ------------------------------------------------------------

class FakeTensor:
    """A device allocation stand-in: unique address, real byte accounting."""
    _next_address = 1

    def __init__(self, length, element_size, device=None):
        self._length = length
        self._element_size = element_size
        self.device = device
        self._address = FakeTensor._next_address * 0x1000
        FakeTensor._next_address += 1
        self._storage = types.SimpleNamespace(
            data_ptr=lambda: self._address,
            nbytes=lambda: self._length * self._element_size)

    def numel(self):
        return self._length

    def element_size(self):
        return self._element_size

    def data_ptr(self):
        return self._address

    def untyped_storage(self):
        return self._storage


class FakeEvent:
    def __init__(self, cuda):
        self.cuda = cuda
        self.ready = False
        self.recorded = 0

    def record(self, stream=None):
        self.recorded += 1
        self.ready = self.cuda.auto_ready

    def query(self):
        return bool(self.ready)


class FakeCuda:
    def __init__(self):
        self.auto_ready = True
        self.events = []
        self.streams = []
        self.graphs = []

    def Event(self, enable_timing=False):
        event = FakeEvent(self)
        self.events.append(event)
        return event

    def current_device(self):
        return 0

    def current_stream(self, device=None):
        return types.SimpleNamespace(cuda_stream=7)

    def get_device_properties(self, device=None):
        return types.SimpleNamespace(multi_processor_count=132)

    def memory_allocated(self, device=None):
        return 0

    def synchronize(self, device=None):
        for event in self.events:
            event.ready = True

    def Stream(self, device=None):
        stream = types.SimpleNamespace(cuda_stream=100 + len(self.streams))
        self.streams.append(stream)
        return stream

    def CUDAGraph(self):
        graph = types.SimpleNamespace(replays=0)
        self.graphs.append(graph)
        return graph

    def complete_all(self):
        for event in self.events:
            event.ready = True


class FakeTorch:
    """The subset of torch the native backends touch, with no CUDA."""

    def __init__(self, cuda=None):
        self.cuda = cuda or FakeCuda()
        self.uint8 = 'uint8'
        self.int32 = 'int32'

    def empty(self, shape, dtype=None, device=None):
        return FakeTensor(shape[0], 1, device)

    def full(self, shape, fill, dtype=None, device=None):
        tensor = FakeTensor(shape[0], 4, device)
        tensor.fill = fill
        return tensor

    def device(self, kind, index):
        return f'{kind}:{index}'


# --- fake bounded-operation device (CPU evidence only) ---------------------

class OpEvent:
    def __init__(self, device='cuda:0', ready=False):
        self.device = device
        self.ready = ready
        self.queried = 0

    def query(self):
        self.queried += 1
        return bool(self.ready)


class FakeQsaTensor:
    """Shape/stride/ownership facts without torch or device memory."""

    def __init__(self, shape, device='cpu', dtype='torch.bfloat16',
                 pinned=True, address=0x1000, item_bytes=2):
        self.shape = tuple(shape)
        self.device = device
        self.dtype = dtype
        self._pinned = pinned
        self._address = address
        self._item = item_bytes

    def stride(self):
        strides = []
        running = 1
        for size in reversed(self.shape):
            strides.append(running)
            running *= size
        return tuple(reversed(strides))

    def element_size(self):
        return self._item

    def is_pinned(self):
        return self._pinned

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return self._address

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError('fake tensor supports contiguous slices only')
        pages = len(range(*key.indices(self.shape[0])))
        return FakeQsaTensor((pages, *self.shape[1:]), self.device, self.dtype,
                             self._pinned, self._address, self._item)


class FakeQsaDevice:
    device = 'cuda:0'
    """Delayed-event device port; no CUDA, no synchronization primitive."""

    compute = 'compute'
    copy = 'copy'

    def __init__(self, auto_ready=True):
        self.auto_ready = auto_ready
        self.events = []
        self.copies = []
        self.waits = []

    def facts(self, tensor):
        from e3_types import TensorFacts
        return TensorFacts(str(tensor.device), tuple(tensor.shape),
                           tuple(tensor.stride()), str(tensor.dtype),
                           tensor.element_size(), tensor.is_pinned(),
                           tensor.is_contiguous(), tensor.data_ptr())

    def record(self, stream):
        event = OpEvent(ready=self.auto_ready)
        self.events.append((stream, event))
        return event

    def query(self, event):
        return event.query()

    def wait_event(self, stream, event):
        self.waits.append((stream, event))

    def copy_pages(self, source, arena, source_page, arena_page, pages):
        if not 1 <= pages <= 4:
            raise AssertionError('copy exceeded the four-page bound')
        self.copies.append((source_page, arena_page, pages))

    def complete_all(self):
        for _, event in self.events:
            event.ready = True


class FakeQsaLayer:
    def __init__(self, name='model.layers.0.attn'):
        self.layer_name = name
        self._qsa_kv_offload = True
        self._qsa_host_kv = FakeQsaTensor((64, 1, 4048, 512), address=0x1000)


class FakeModel:
    def __init__(self, layers):
        self._layers = list(layers)

    def modules(self):
        return [self, *self._layers]


# --- fake pinned vLLM modules ----------------------------------------------

class FakeForwardContext(types.ModuleType):
    def __init__(self):
        super().__init__('vllm.forward_context')
        self.QSA_REQ_UID = [77, 77]
        self._forward_context = 'decode-ctx'
        self.created = []

    def create_forward_context(self, attn_metadata, vllm_config):
        self.created.append(attn_metadata)
        return 'prefill-ctx'


class FakeWorkspaceModule(types.ModuleType):
    def __init__(self):
        super().__init__('vllm.v1.worker.workspace')
        self.manager = types.SimpleNamespace(_num_lanes=2, _num_ubatches=1)
        self.init_calls = []
        self.entered = []
        self.exited = []

    def current_workspace_manager(self):
        return self.manager

    def init_workspace_manager(self, device=None, num_ubatches=1, num_lanes=1):
        self.init_calls.append((num_ubatches, num_lanes))
        self.manager._num_lanes = num_lanes
        return self.manager

    def use_workspace_lane(self, lane):
        module = self

        class _Lane:
            def __enter__(self):
                module.entered.append(lane)
                return self

            def __exit__(self, *exc):
                module.exited.append(lane)
                return False

        return _Lane()


def fake_vllm(forward_context, workspace, success):
    """Module tree for ``import vllm...``; ``success`` is the enum identity."""

    class ResponseStatus:
        SUCCESS = success

    class WorkerProc:
        pass

    WorkerProc.ResponseStatus = ResponseStatus

    vllm = types.ModuleType('vllm')
    v1 = types.ModuleType('vllm.v1')
    executor = types.ModuleType('vllm.v1.executor')
    multiproc = types.ModuleType('vllm.v1.executor.multiproc_executor')
    multiproc.WorkerProc = WorkerProc
    worker = types.ModuleType('vllm.v1.worker')
    vllm.v1 = v1
    v1.executor = executor
    v1.worker = worker
    executor.multiproc_executor = multiproc
    return {'vllm': vllm, 'vllm.v1': v1, 'vllm.v1.executor': executor,
            'vllm.v1.executor.multiproc_executor': multiproc,
            'vllm.v1.worker': worker, 'vllm.forward_context': forward_context,
            'vllm.v1.worker.workspace': workspace}


# --- fake host context ------------------------------------------------------

class HostForwardContext:
    def __init__(self, current='decode-ctx'):
        self.current = current
        self.history = []

    def get(self):
        return self.current

    def set(self, value):
        self.history.append(value)
        self.current = value


class HostWorkspace:
    def __init__(self, current='decode-ws'):
        self.stack = [current]
        self.calls = []

    def enter(self, workspace):
        self.calls.append(('enter', workspace))
        self.stack.append(workspace)

    def exit(self, workspace):
        if not self.stack or self.stack[-1] != workspace:
            raise AssertionError('workspace exit out of order')
        self.calls.append(('exit', workspace))
        self.stack.pop()


# --- fake queue/executor ----------------------------------------------------

class FakeMQ:
    """Message-queue compatible reader for one rank's reply channel."""

    def __init__(self):
        self._items = []
        self._condition = threading.Condition()
        self.buffer = types.SimpleNamespace(max_chunks=10)
        self._flash_s5_receiver_owner = None

    def put(self, value):
        with self._condition:
            self._items.append(value)
            self._condition.notify_all()

    def dequeue(self, timeout=None):
        with self._condition:
            if not self._items:
                self._condition.wait(timeout)
            if not self._items:
                raise TimeoutError('response queue empty')
            return self._items.pop(0)

    def empty(self):
        with self._condition:
            return not self._items


class FakeBroadcast:
    def __init__(self, executor):
        from contextlib import nullcontext
        self.executor = executor
        self.current_idx = 0
        self.buffer = types.SimpleNamespace(max_chunks=10, max_chunk_bytes=1 << 20,
            n_reader=4, get_metadata=lambda index: nullcontext(bytearray((0, 0, 0, 0, 0))))

    def enqueue(self, item):
        self.executor.enqueue(item)


class _Delivery:
    def __init__(self, item, ranks):
        self.item = item
        self.remaining = set(ranks)


class FakeExecutor:
    """Broadcast ring + per-rank reply MQs; delivery is explicit and ordered."""

    def __init__(self, workers, success):
        self.workers = workers
        self.success = success
        self.error = object()
        self.response_mqs = [FakeMQ() for _ in workers]
        self.rpc_broadcast_mq = FakeBroadcast(self)
        self._flash_s5_bounded_send = self.enqueue  # in-memory fixture only
        self.output_rank = 0
        self.futures_queue = []
        self.is_failed = False
        self.queued = []
        self.pending = []
        self.failures = []
        self.held_replies = []

    def enqueue(self, item):
        self.queued.append(item)
        self.pending.append(_Delivery(item, range(len(self.workers))))

    def _invoke(self, rank, item):
        method, args, kwargs, _ = item
        worker = self.workers[rank]
        try:
            return self.success, getattr(worker, method)(*args, **kwargs)
        except BaseException as exc:  # Reported like WorkerProc error status.
            self.failures.append((rank, method, exc))
            return self.error, None

    def _run(self, rank, item):
        self.response_mqs[rank].put(self._invoke(rank, item))

    def deliver(self, ranks=None, hold=(), withhold=()):
        """Deliver pending broadcasts.

        ``hold`` ranks are neither executed nor answered; ``withhold`` ranks run
        their RPC but their reply stays local until :meth:`release_hold`.
        """
        delivered = set()
        for delivery in list(self.pending):
            for rank in sorted(delivery.remaining):
                if ranks is not None and rank not in ranks:
                    continue
                if rank in hold:
                    continue
                delivery.remaining.discard(rank)
                if rank in withhold:
                    # Run the RPC now; only its reply is withheld.
                    self.held_replies.append((rank, self._invoke(rank,
                                                                 delivery.item)))
                else:
                    delivered.add(rank)
                    self._run(rank, delivery.item)
        self.pending = [d for d in self.pending if d.remaining]
        return delivered

    def release_hold(self, ranks):
        """Deliver held replies and execute broadcasts still held for ``ranks``."""
        ranks = set(ranks)
        for held in list(self.held_replies):
            if held[0] in ranks:
                self.held_replies.remove(held)
                self.response_mqs[held[0]].put(held[1])
        delivered = set()
        for delivery in list(self.pending):
            for rank in sorted(delivery.remaining & ranks):
                delivery.remaining.discard(rank)
                delivered.add(rank)
                self._run(rank, delivery.item)
        self.pending = [d for d in self.pending if d.remaining]
        return delivered

    def fail_rank(self, rank):
        """One worker reports the failure status for its next reply."""
        for delivery in list(self.pending):
            if rank in delivery.remaining:
                delivery.remaining.discard(rank)
                self.response_mqs[rank].put((self.error, None))
        self.pending = [d for d in self.pending if d.remaining]

    def settle(self, ranks=None, timeout=2.0):
        """Wait until every delivered rank reply reached the owned receiver."""
        transport = self.transport
        ranks = range(len(self.workers)) if ranks is None else ranks
        ranks = set(ranks)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with transport._condition:
                inflight = any(rank in ranks for rank, _ in transport._inflight)
                poisoned = transport._failure is not None
            if poisoned:
                # A failed worker poisons the receiver: no later reply is read,
                # and the driver's next poll raises the quarantine itself.
                return True
            if not inflight and all(self.response_mqs[rank].empty()
                                    for rank in ranks):
                return True
            time.sleep(0.001)
        self.raise_worker_failures()
        raise AssertionError('the delivered replies never reached the receiver')

    @property
    def transport(self):
        transport = getattr(self, '_flash_s5_transport', None)
        if transport is None:
            raise AssertionError('owned transport was not installed')
        return transport

    def raise_worker_failures(self):
        if self.failures:
            rank, method, exc = self.failures[0]
            raise AssertionError(
                f'worker {rank} raised in {method}: {exc!r}') from exc

    def walk(self, engine, limit=128):
        """Drive the generated step seam until it returns the parked output."""
        for _ in range(limit):
            if self.pending:
                delivered = self.deliver()
                self.settle(delivered)
                self.raise_worker_failures()
            result = eh.before_step_with_batch_queue(engine)
            if result is not None:
                return result
        self.raise_worker_failures()
        raise AssertionError('driver did not finish within the step budget')


# --- fake scheduler ---------------------------------------------------------

class FakePool:
    def __init__(self):
        self.freed = []

    def free_blocks(self, blocks):
        self.freed.extend(list(blocks))


class FakeScheduler:
    def __init__(self):
        self.block_pool = FakePool()
        self.kv_cache_manager = types.SimpleNamespace(block_pool=self.block_pool)
        self.update_calls = []
        self.outputs = {'published': 1}

    def update_from_output(self, scheduler_output, output):
        self.update_calls.append((scheduler_output, output))
        return self.outputs


class FakeEngine:
    def __init__(self, executor, scheduler):
        self.model_executor = executor
        self.scheduler = scheduler


# --- fake worker ------------------------------------------------------------

class ForwardOutput:
    def __init__(self, rank, sample=None):
        self.rank = rank
        self.sample = sample

    def __repr__(self):
        return f'ForwardOutput(rank={self.rank})'


class FakeRunner:
    """The generated model runner: routes through A's real execute wrapper.

    ``_flash_s5_execute_prefill_operation`` is a fixture-only driver.
    The generated runner now refuses stock forward; this fixture does not
    establish native layer continuation or sample-ring safety.
    The stock forward stands in for the pinned QSA layer call sites: it drives
    the real operation state machine with fake tensors/events before recording
    the stock frame.  This is CPU fake-CUDA evidence, not native qualification.
    """

    def __init__(self, runtime, rank, calls, layers=(), output_rank=0):
        self._flash_e3_lane = runtime
        self.execute_model_state = None
        self.rank = rank
        self.calls = calls
        self.output_rank = output_rank
        self.model = FakeModel(layers)

    def _flash_s5_execute_prefill_operation(self, scheduler_output):
        server = self._flash_s5_worker
        server.begin_prefill_operation(self, scheduler_output)
        try:
            output = self.execute_model(scheduler_output)
        except BaseException:
            server.abort_prefill_operation(self)
            raise
        server.end_prefill_operation(self)
        return output

    def execute_model(self, scheduler_output, *args, **kwargs):
        self.calls.append(('execute_model',
                           scheduler_output.flash_e3_transaction))

        def stock(scheduler_output, *rest):
            self._run_qsa_operation()
            self.execute_model_state = types.SimpleNamespace(pending=True)
            self.calls.append(('stock_forward',
                               scheduler_output.flash_e3_transaction))
            return ForwardOutput(self.rank)

        return lh.execute_with_e3(self, stock, scheduler_output)

    def _run_qsa_operation(self):
        """Fixture stand-in for the pinned layer's generated call sites."""
        server = self._flash_s5_worker
        operation = server.operation
        if operation is None or not self.model._layers:
            return
        layer = self.model._layers[0]
        name = layer.layer_name
        # Explicit fixture table; no production flat-ID-to-QSA inference.
        operation.plan_provider = lambda layer_name, n: (10, 11)[:n]
        arena = FakeQsaTensor((24, 1, 4048, 512), device='cuda:0',
                              pinned=False, address=0x9000)
        table = FakeQsaTensor((16,), device='cuda:0', dtype='torch.int64',
                              pinned=False, address=0x7000)
        operation.begin_layer(name, layer._qsa_host_kv)
        operation.producer_submitted(name, layer._qsa_host_kv)
        operation.bind_plan(name, table, 2, 2 * 4048, arena)
        for bucket in operation._layers[name].buckets:
            staged = arena[:bucket.pages]
            operation.fill_bucket(staged, bucket.lo, bucket.hi, table)
            operation.after_attention_bucket(staged, table)
        operation.end_layer(name)
        # Fixture-only stand-in for the not-yet-implemented asynchronous
        # output detachment.  Only the PP output rank delivers a chunk result;
        # no stock sampler or PP sample ring is touched.
        if self.rank == self.output_rank:
            server.accept_detached_output(
                ForwardOutput(self.rank, sample='detached'))


class FakeGPUWorker:
    def __init__(self, runner, rank, *, output_rank, calls):
        self.model_runner = runner
        self.rank = rank
        self.is_output_rank = rank == output_rank
        self.calls = calls
        self.samples = []

    def execute_model(self, scheduler_output, non_block=False):
        self.calls.append(('worker_execute',
                           scheduler_output.flash_e3_transaction))
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output=None):
        self.samples.append(grammar_output)
        if not self.is_output_rank:
            return None
        return ForwardOutput(self.rank, sample='sampled-token')


def make_worker_stack(rank, *, output_rank=0, cuda=None):
    """One worker as the generated runner/GPU worker would build it."""
    cuda = cuda or FakeCuda()
    calls = []
    device = nb.NativeDevice(torch_module=FakeTorch(cuda), device='cuda:0')
    backend = nb.EventCompletionBackend(device)
    state = ls.WorkerState(rank, engine_epoch=1, weights=object(),
                           weights_address=4096, weights_bytes=1 << 20,
                           max_outstanding=16)
    namespace = types.SimpleNamespace(QSA_REQ_UID=[77, 77])
    runtime = lh.LaneRuntime(
        state, submit_backend=backend,
        forward_context=HostForwardContext(),
        workspace=HostWorkspace(),
        prefill_forward_context='prefill-ctx',
        prefill_workspace='prefill-ws',
        uid_namespace=namespace, prefill_uid_buffer=[-2, -2])
    bindings = ((ls.ScratchSpec('prefill_input_ids', 'prefill', 'metadata', 4096),
                 500_000),)
    server = nr.WorkerHandoffServer(runtime, bindings=bindings,
                                    operation_device=FakeQsaDevice())
    runner = FakeRunner(runtime, rank, calls, layers=(FakeQsaLayer(),),
                        output_rank=output_rank)
    runner._flash_s5_worker = server
    worker = FakeGPUWorker(runner, rank, output_rank=output_rank, calls=calls)
    for name in ('open', 'submit', 'poll', 'commit', 'cancel'):
        setattr(worker, 'flash_s5_' + name,
                (lambda method: lambda message: eh.worker_rpc(
                    worker, method, message))(name))
    return types.SimpleNamespace(rank=rank, cuda=cuda, device=device,
                                 backend=backend, state=state, runtime=runtime,
                                 server=server, runner=runner, worker=worker,
                                 calls=calls)


def make_transaction(uid='req-a', seq=1, start=0, end=8, generation=0):
    return TransactionId(1, uid, generation, start, end, seq)


class FakeSchedulerOutput:
    def __init__(self, uid, tokens, admission=None):
        self.num_scheduled_tokens = {uid: tokens}
        self.total_num_scheduled_tokens = tokens
        self.flash_s5_prefill_admission = admission
        self.flash_e3_transaction = None


def admission(transaction, *, blocks=(10, 11), grammar=None, sparse=False):
    return ed.PrefillAdmission(transaction, tuple(blocks), 'state-v1',
                               grammar_output=grammar, sparse_active=sparse)


def build_legacy_fixture_engine(engine):
    """Exercise the earlier post-schedule controller explicitly, NOT startup.

    Native startup no longer silently selects this controller when its real
    model/config discovery fails. These fake-worker lifetime regressions stay
    useful and separate from wire-v1 admission/driver tests.
    """
    from async_handoff import HandoffConfig, build_composition
    from lane_state import estimated_e3_inventory
    from scheduler_hooks import SchedulerLeases
    env = {'FLASH_S5_OVERLAP': '1'}
    config = HandoffConfig.from_env(env)
    transport = eh.install(engine.model_executor, config=config,
                           success_status=nr.response_success())
    backend = nb.RemoteCompletionBackend(config)
    allocation = estimated_e3_inventory(chunk_tokens=config.chunk_tokens)
    composition = build_composition(env, transport=transport, backend=backend, lanes={},
        native_allocations=allocation, allocation=allocation, side='engine',
        scheduler_leases=SchedulerLeases())
    composition.attach(engine=engine, scheduler=engine.scheduler, executor=engine.model_executor)
    driver = ed.SegmentDriver(composition, executor=engine.model_executor,
        scheduler=engine.scheduler, engine=engine).install()
    engine._flash_s5_composition = composition
    return composition


class EngineFixture:
    """Earlier post-schedule controller; explicit CPU fixture, not native startup."""

    def __init__(self, *, workers=4, cuda=None):
        self.workers = [make_worker_stack(rank, output_rank=0, cuda=cuda)
                        for rank in range(workers)]
        self.success = object()
        self.executor = FakeExecutor([w.worker for w in self.workers],
                                     self.success)
        self.scheduler = FakeScheduler()
        self.engine = FakeEngine(self.executor, self.scheduler)
        self.forward_context = FakeForwardContext()
        self.workspace = FakeWorkspaceModule()
        self.modules = fake_vllm(self.forward_context, self.workspace,
                                 self.success)
        with mock.patch.dict(sys.modules, self.modules), \
                mock.patch.dict(os.environ, {'FLASH_S5_OVERLAP': '1'}):
            with mock.patch.object(nb.RemoteCompletionBackend, 'native_verified', True):
                self.composition = build_legacy_fixture_engine(self.engine)

    @property
    def driver(self):
        return self.engine._flash_s5_driver

    def output(self, transaction=None, *, tokens=8, grammar=None, sparse=False):
        transaction = transaction or make_transaction()
        return transaction, FakeSchedulerOutput(
            transaction.request_uid, tokens,
            admission(transaction, grammar=grammar, sparse=sparse))

    def park(self, **kwargs):
        transaction, output = self.output(**kwargs)
        assert eh.park_prefill(self.engine, output) is True
        return transaction, output, self.driver.parked

    def run(self, **kwargs):
        transaction, output, _ = self.park(**kwargs)
        result = self.executor.walk(self.engine)
        return transaction, output, result


@mock.patch('s5_startup.require_native_ready', lambda: None)
@mock.patch('s5_startup.install_native_lane', lambda runner, env=None, clock=None: None)  # D4 lane: fixture-only bypass
class StartupSeamTests(unittest.TestCase):
    def test_worker_builder_is_installed_and_constructs_the_runtime(self):
        class Runner:
            def __init__(self):
                self.vllm_config = types.SimpleNamespace(
                    parallel_config=types.SimpleNamespace(rank=2),
                    compilation_config=types.SimpleNamespace(
                        cudagraph_mode=types.SimpleNamespace(name='full'),
                        cuda_graph_sizes=(1, 2)))
                self.device = 'cuda:0'
                self.max_num_reqs = 8
                self.decode_query_len = 1
                self.model = None

        runner = Runner()
        forward_context = FakeForwardContext()
        workspace = FakeWorkspaceModule()
        modules = fake_vllm(forward_context, workspace, object())
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.dict(sys.modules, {'torch': FakeTorch()}), \
                mock.patch.dict(os.environ, {'FLASH_S5_OVERLAP': '1'}):
            builder = su.install_builder(runner, {'layer': object()},
                                         env={'FLASH_S5_OVERLAP': '1'})
            self.assertTrue(callable(builder))
            runtime = lh.init_lane(runner, {'layer': object()})
        self.assertIsInstance(runtime, lh.LaneRuntime)
        self.assertEqual(runtime.state.worker_rank, 2)
        self.assertIsInstance(runtime.backend, nb.EventCompletionBackend)
        self.assertEqual(forward_context.created, [None])
        self.assertEqual(workspace.init_calls, [])
        self.assertIs(runner._flash_s5_worker.runtime, runtime)
        self.assertEqual(runner._flash_s5_native['rank'], 2)
        self.assertEqual(runner._flash_s5_native['workspace_lane'],
                         nr.WORKSPACE_LANE)
        self.assertGreater(runner._flash_s5_native['uid_bytes'], 0)
        self.assertTrue(callable(runner._flash_s5_worker.open))
        self.assertTrue(callable(runner._flash_e3_graph_builder))

    def test_split_capture_fails_closed_without_the_model_forward_callable(self):
        runner = types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(
                parallel_config=types.SimpleNamespace(rank=0),
                compilation_config=types.SimpleNamespace(
                    cudagraph_mode=types.SimpleNamespace(name='full'),
                    cuda_graph_sizes=(1,))),
            device='cuda:0', max_num_reqs=4, decode_query_len=1, model=None)
        forward_context = FakeForwardContext()
        workspace = FakeWorkspaceModule()
        modules = fake_vllm(forward_context, workspace, object())
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.dict(sys.modules, {'torch': FakeTorch()}), \
                mock.patch.dict(os.environ, {'FLASH_S5_OVERLAP': '1'}):
            su.install_builder(runner, {}, env={'FLASH_S5_OVERLAP': '1'})
            lh.init_lane(runner, {})
        router = runner._flash_e3_graph_builder(runner)
        self.assertEqual(router.current_mode, 'ordinary')
        descriptor = object()
        context = types.SimpleNamespace(reserved_bytes=1 << 20,
                                        context_id=router.handle.context_id)
        with self.assertRaisesRegex(ah.ConfigurationError,
                                    '_flash_s5_graph_capture'):
            router.backend.capture('decode-split', descriptor,
                                   router.handle, context)

    def test_disabled_startup_is_inert_on_both_seams(self):
        runner = types.SimpleNamespace()
        self.assertIsNone(su.install_builder(runner, {}, env={}))
        self.assertIsNone(getattr(runner, '_flash_e3_build_state', None))
        engine = types.SimpleNamespace()
        self.assertIsNone(su.install_engine(engine, env={}))
        self.assertIsNone(getattr(engine, '_flash_s5_composition', None))

    def test_engine_startup_installs_composition_and_driver_once(self):
        fixture = EngineFixture()
        self.assertIsInstance(fixture.composition, ah.Composition)
        self.assertIs(fixture.engine._flash_s5_handoff,
                      fixture.composition.dispatcher)
        self.assertIs(fixture.scheduler._flash_s5_leases,
                      fixture.composition.scheduler_leases)
        self.assertIs(fixture.driver.dispatcher, fixture.composition.dispatcher)
        self.assertEqual(fixture.driver.ranks, (0, 1, 2, 3))
        self.assertTrue(fixture.executor._flash_s5_stock_forbidden)
        with mock.patch.dict(sys.modules, fixture.modules), \
                mock.patch.dict(os.environ, {'FLASH_S5_OVERLAP': '1'}):
            with self.assertRaisesRegex(su.StartupError, 'already carries'):
                eh.after_engine_startup(fixture.engine)
            with self.assertRaisesRegex(ed.DriverError, 'already has'):
                ed.SegmentDriver(fixture.composition,
                                 executor=fixture.executor,
                                 scheduler=fixture.scheduler,
                                 engine=fixture.engine).install()


@mock.patch('s5_startup.require_native_ready', lambda: None)
@mock.patch.dict(nb._ROLE_KIND, {DependencyRole.COMMIT: CompletionKind.STATE_WRITE})
class SegmentDriverTests(unittest.TestCase):
    def test_full_chunk_publishes_once_through_the_scheduler(self):
        fixture = EngineFixture()
        transaction, output, result = fixture.run()
        self.assertEqual(result, (fixture.scheduler.outputs, True))
        self.assertEqual(len(fixture.scheduler.update_calls), 1)
        called_output, model_output = fixture.scheduler.update_calls[0]
        self.assertIs(called_output, output)
        self.assertEqual(model_output.rank, fixture.executor.output_rank)
        self.assertIsNone(fixture.driver.parked)
        self.assertEqual(fixture.driver.phase, 'idle')
        self.assertEqual(fixture.composition.scheduler_leases.held_blocks(), ())
        self.assertIsNone(fixture.composition.dispatcher.active)
        self.assertEqual(fixture.composition.frontier.committed_blocks(),
                         (10, 11))
        # D2 has no stock sampling and no PP sample ring in the chunk path.
        self.assertEqual([w.worker.samples for w in fixture.workers],
                         [[], [], [], []])
        for stack in fixture.workers:
            frames = [c for c in stack.calls if c[0] == 'stock_forward']
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0][1], transaction)
            self.assertEqual(stack.calls[0][0], 'execute_model')
            self.assertNotIn('worker_execute', [c[0] for c in stack.calls])
            # Decode's module context and workspace are restored after the chunk.
            self.assertEqual(stack.runtime.dispatcher.forward_context.current,
                             'decode-ctx')
            self.assertEqual(stack.runtime.dispatcher.workspace.stack,
                             ['decode-ws'])
            self.assertEqual(stack.runtime.dispatcher.workspace.calls,
                             [('enter', 'prefill-ws'), ('exit', 'prefill-ws')])
            # The decode UID namespace and the private prefill buffer are
            # restored/untouched by the operation path.
            self.assertEqual(stack.runtime.uid_namespace.QSA_REQ_UID, [77, 77])
            self.assertEqual(stack.runtime.prefill_uid_buffer, [-2, -2])
            self.assertIsNone(stack.runner.execute_model_state)
        # The fixture PP channel carried nothing; this is NOT native sample-ring proof.
        self.assertEqual(fixture.composition.pp_channel.sends, 0)
        self.assertEqual(fixture.composition.pp_channel.consumed, 0)

    def test_operation_completion_is_not_hidden_by_ready_lane_events(self):
        stack = make_worker_stack(0)
        txn = make_transaction()
        metadata = MetadataSnapshot.capture(
            txn, (BatchRow('req-a', 0, 0, 0, 8, 0),), block_ids=(10, 11))
        stack.worker.flash_s5_open(ah.OpenPrefill(txn, 8, metadata))
        stack.server.operation_device.auto_ready = False
        output = FakeSchedulerOutput('req-a', 8)
        message = ah.SubmitOperation(txn, 0, DependencyRole.PRODUCER,
                                     scheduler_output=output)
        stack.worker.flash_s5_submit(message)
        report = stack.worker.flash_s5_poll(ah.PollOperations(txn, 8))
        self.assertEqual(report.proofs, ())
        self.assertGreater(report.pending, 0)
        self.assertTrue(stack.server._deferred_proofs)
        stack.server.operation_device.complete_all()
        report = stack.worker.flash_s5_poll(ah.PollOperations(txn, 8))
        self.assertEqual(len(report.proofs), 1)
        self.assertEqual(report.pending, 0)
        self.assertEqual(stack.worker.flash_s5_poll(ah.PollOperations(txn, 8)).proofs, ())

    def test_producer_and_commit_proofs_are_equal_on_every_rank(self):
        fixture = EngineFixture()
        fixture.run()
        completions = [proof for stack in fixture.workers
                       for proof in stack.state.prefill.completed]
        self.assertEqual(len(completions), 2 * len(fixture.workers))
        for proof in completions:
            self.assertEqual(proof.completion_seq, proof.ticket.op_seq + 1)
        producer = {proof.ticket.worker_rank: proof.completion_seq
                    for proof in completions
                    if proof.ticket.dependency_role is DependencyRole.PRODUCER}
        commit = {proof.ticket.worker_rank: proof.completion_seq
                  for proof in completions
                  if proof.ticket.dependency_role is DependencyRole.COMMIT}
        self.assertEqual(set(producer.values()), {1})
        self.assertEqual(set(commit.values()), {2})
        self.assertEqual(set(producer), set(range(len(fixture.workers))))
        self.assertEqual(set(commit), set(range(len(fixture.workers))))

    def test_no_publication_before_every_worker_proof(self):
        fixture = EngineFixture()
        for stack in fixture.workers:
            stack.cuda.auto_ready = False
        transaction, _, _ = fixture.park()
        for _ in range(40):
            if fixture.executor.pending:
                fixture.executor.deliver()
                fixture.executor.settle()
            self.assertIsNone(fixture.driver.advance())
            if fixture.driver.phase == 'poll_sent':
                break
        self.assertEqual(fixture.driver.phase, 'poll_sent')
        self.assertEqual(fixture.scheduler.update_calls, [])
        self.assertEqual(fixture.composition.frontier.committed_blocks(), ())
        self.assertEqual(fixture.driver.parked.transaction, transaction)
        # Complete one rank only: the lease still holds every arena block.
        fixture.workers[0].cuda.complete_all()
        for _ in range(6):
            if fixture.executor.pending:
                fixture.executor.deliver()
                fixture.executor.settle()
            self.assertIsNone(fixture.driver.advance())
        self.assertEqual(fixture.scheduler.update_calls, [])
        self.assertEqual(fixture.composition.scheduler_leases.held_blocks(),
                         (10, 11))
        for stack in fixture.workers:
            stack.cuda.complete_all()
        result = fixture.executor.walk(fixture.engine)
        self.assertEqual(result, (fixture.scheduler.outputs, True))
        self.assertEqual(len(fixture.scheduler.update_calls), 1)

    def test_completion_is_recorded_before_the_rpc_ack(self):
        fixture = EngineFixture()
        transaction, output = fixture.output()
        fixture.driver.park(output)
        self.assertIsNone(fixture.driver.advance())       # broadcasts open
        delivered = fixture.executor.deliver()
        fixture.executor.settle(delivered)
        self.assertIsNone(fixture.driver.advance())       # applies open
        # The producer submit runs on every worker but rank 1's ACK is withheld.
        delivered = fixture.executor.deliver(withhold=(1,))
        fixture.executor.settle(delivered)
        self.assertEqual(len(fixture.executor.held_replies), 1)
        call = next(iter(fixture.executor.transport._calls.values()))
        self.assertEqual(call.kind, 'prefill-operation')
        self.assertFalse(call.ready)
        lease = fixture.driver.dispatcher.active
        self.assertEqual(lease.acked, 0)
        self.assertEqual(lease.pending, len(fixture.workers))
        self.assertEqual(lease.completed, [])
        # Rank 1 already recorded its device event, but no engine poll saw it.
        outstanding = fixture.workers[1].backend.outstanding(transaction)
        self.assertEqual(len(outstanding), 1)
        self.assertGreater(len(fixture.workers[1].cuda.events), 0)
        # The engine lease still knows neither the ACK nor the completion.
        self.assertIsNone(fixture.driver.advance())
        # The call is correlated per rank: a partial reply set applies nothing.
        self.assertEqual(lease.acked, 0)
        fixture.executor.release_hold((1,))
        fixture.executor.settle()
        self.assertIsNone(fixture.driver.advance())
        self.assertEqual(lease.acked, len(fixture.workers))

    def test_admission_is_fail_closed(self):
        fixture = EngineFixture()
        self.assertIsNone(fixture.driver.park(FakeSchedulerOutput('req-a', 8)))
        transaction, output = fixture.output()
        output.num_scheduled_tokens = {transaction.request_uid: 8, 'other': 1}
        with self.assertRaisesRegex(ed.DriverError, 'mixed decode batch'):
            fixture.driver.park(output)
        self.assertIsNone(fixture.driver.parked)
        with self.assertRaisesRegex(ed.DriverError, 'state value'):
            ed.PrefillAdmission(make_transaction(), (10,), None)
        with self.assertRaisesRegex(ed.DriverError, 'positive ints'):
            ed.PrefillAdmission(make_transaction(), (0,), object())

    def test_sparse_admission_is_refused_and_counted(self):
        fixture = EngineFixture()
        transaction = make_transaction(uid='req-sparse', seq=1)
        output = FakeSchedulerOutput('req-sparse', 8,
                                     admission(transaction, sparse=True))
        with self.assertRaises(E3Pending):
            fixture.driver.park(output)
        records = [r for r in fixture.composition.dispatcher.telemetry.snapshot()
                   if r['event'] == 'route_serialized']
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['reason_code'],
                         'sparse_uva_serialized_not_overlap')
        self.assertIsNone(fixture.driver.parked)

    def test_prefix_blocks_are_unshareable_until_commit(self):
        fixture = EngineFixture()
        transaction, _ = fixture.output()
        fixture.driver.park(_)
        with self.assertRaises(E3Pending):
            fixture.composition.frontier.shareable((10, 11))
        fixture.executor.walk(fixture.engine)
        self.assertEqual(fixture.composition.frontier.shareable((10, 11)),
                         (10, 11))
        self.assertEqual(fixture.composition.frontier.lookup((10, 11, 99)),
                         (10, 11))

    def test_arena_blocks_can_be_reused_after_retirement(self):
        fixture = EngineFixture()
        fixture.run(transaction=make_transaction(seq=1))
        fixture.composition.frontier.retire_blocks((10, 11))
        fixture.run(transaction=make_transaction(seq=2))
        self.assertEqual(len(fixture.scheduler.update_calls), 2)
        self.assertEqual(fixture.composition.scheduler_leases.held_blocks(), ())
        self.assertEqual(fixture.composition.dispatcher.reply_count('commit'), 1)
        self.assertIsNone(fixture.driver.parked)


@mock.patch('s5_startup.require_native_ready', lambda: None)
@mock.patch.dict(nb._ROLE_KIND, {DependencyRole.COMMIT: CompletionKind.STATE_WRITE})
class StopPathTests(unittest.TestCase):
    def _drive_until(self, fixture, predicate, limit=64):
        for _ in range(limit):
            if fixture.executor.pending:
                fixture.executor.deliver()
                fixture.executor.settle()
            fixture.driver.advance()
            fixture.executor.raise_worker_failures()
            if predicate():
                return
        fixture.executor.raise_worker_failures()
        self.fail('stop path did not retire')

    def test_abort_withholds_publication_and_retires_storage(self):
        fixture = EngineFixture()
        transaction, output, _ = fixture.park()
        for _ in range(3):
            if fixture.executor.pending:
                fixture.executor.deliver()
                fixture.executor.settle()
            fixture.driver.advance()
        self.assertEqual(fixture.driver.phase, 'commit_op_sent')
        result = eh.before_abort(fixture.engine, [transaction.request_uid])
        self.assertEqual(result.name, 'RETIRING')
        self._drive_until(fixture, lambda: fixture.driver.parked is None)
        self.assertEqual(fixture.scheduler.update_calls, [])
        self.assertEqual(fixture.composition.frontier.committed_blocks(), ())
        self.assertIsNone(fixture.composition.dispatcher.active)
        self.assertEqual(fixture.composition.scheduler_leases.held_blocks(), ())
        self.assertIsNone(fixture.driver.stop_kind)
        for stack in fixture.workers:
            self.assertIsNone(stack.server.handoff)
            self.assertEqual(stack.state.prefill.state, 'free')

    def test_preempt_pause_reset_and_shutdown_keep_polling(self):
        stops = {
            'preempt': lambda f: f.driver.stop('preempt'),
            'pause': lambda f: eh.before_pause(f.engine, 'abort'),
            'reset': lambda f: eh.before_reset_prefix(f.engine),
            'shutdown': lambda f: eh.before_engine_shutdown(f.engine),
        }
        for name, stop in stops.items():
            with self.subTest(stop=name):
                fixture = EngineFixture()
                fixture.park()
                fixture.executor.deliver()      # nothing queued: open not sent
                fixture.executor.settle()
                self.assertIsNone(fixture.driver.advance())   # sends open
                if name in ('reset', 'shutdown'):
                    with self.assertRaises(E3Pending):
                        stop(fixture)
                else:
                    stop(fixture)
                self._drive_until(fixture,
                                  lambda: fixture.driver.parked is None)
                self.assertEqual(fixture.scheduler.update_calls, [])
                self.assertIsNone(fixture.composition.dispatcher.active)
                self.assertEqual(
                    fixture.composition.scheduler_leases.held_blocks(), ())

    def test_a_preempted_chunk_retires_and_the_next_one_admits(self):
        fixture = EngineFixture()
        fixture.park()
        self.assertIsNone(fixture.driver.advance())       # broadcasts open
        delivered = fixture.executor.deliver()
        fixture.executor.settle(delivered)
        self.assertIsNone(fixture.driver.advance())       # sends producer submit
        self.assertEqual(fixture.driver.phase, 'producer_sent')
        fixture.driver.stop('preempt')
        self._drive_until(fixture, lambda: fixture.driver.parked is None)
        self.assertIsNone(fixture.composition.dispatcher._stopping)
        # The stop belonged to that request; a later chunk is admitted again.
        _, _, result = fixture.run(transaction=make_transaction(uid='req-b',
                                                                seq=2))
        self.assertEqual(result, (fixture.scheduler.outputs, True))
        self.assertEqual(len(fixture.scheduler.update_calls), 1)

    def test_stop_before_any_operation_retires_locally(self):
        fixture = EngineFixture()
        fixture.park()
        fixture.driver.stop('shutdown')
        self._drive_until(fixture, lambda: fixture.driver.parked is None)
        self.assertEqual(fixture.scheduler.update_calls, [])
        self.assertIsNone(fixture.composition.dispatcher.active)
        self.assertEqual(fixture.executor.queued, [])

    def test_partial_send_never_advances_the_transaction(self):
        fixture = EngineFixture()
        fixture.park()
        self.assertIsNone(fixture.driver.advance())       # broadcast open
        delivered = fixture.executor.deliver(ranks=(0, 1, 2))
        fixture.executor.settle(delivered)
        for _ in range(8):
            self.assertIsNone(fixture.driver.advance())
        self.assertEqual(fixture.scheduler.update_calls, [])
        lease = fixture.composition.dispatcher.active
        self.assertIsNotNone(lease)
        # The open reply was partial, so no worker lane was bound at all.
        self.assertEqual(lease.unbound, (0, 1, 2, 3))
        self.assertEqual(fixture.composition.scheduler_leases.held_blocks(), ())

    def test_worker_failure_quarantines_instead_of_publishing(self):
        fixture = EngineFixture()
        fixture.park()
        self.assertIsNone(fixture.driver.advance())       # broadcasts open
        delivered = fixture.executor.deliver()
        fixture.executor.settle(delivered)
        self.assertIsNone(fixture.driver.advance())       # sends producer submit
        self.assertEqual(fixture.driver.phase, 'producer_sent')
        fixture.executor.fail_rank(3)
        with self.assertRaisesRegex(ed.DriverError, 'quarantined'):
            for _ in range(16):
                if fixture.executor.pending:
                    fixture.executor.deliver()
                    fixture.executor.settle()
                fixture.driver.advance()
        self.assertEqual(fixture.scheduler.update_calls, [])
        self.assertEqual(fixture.composition.frontier.committed_blocks(), ())
        self.assertEqual(fixture.composition.scheduler_leases.held_blocks(),
                         (10, 11))

    def test_finish_pause_refuses_while_a_chunk_is_parked(self):
        fixture = EngineFixture()
        fixture.park()
        with self.assertRaisesRegex(ed.DriverError, 'asynchronous drain'):
            eh.before_finish_pause(fixture.engine)


@mock.patch('s5_startup.require_native_ready', lambda: None)
@mock.patch.dict(nb._ROLE_KIND, {DependencyRole.COMMIT: CompletionKind.STATE_WRITE})
class GeneratedSeamTests(unittest.TestCase):
    def test_core_transform_installs_the_native_startup_and_step_seams(self):
        source = (ROOT / 'reference/vllm-src/v1/engine/core.py').read_text()
        generated = eh.generate_core(source)
        self.assertIn('_flash_s5_hooks.after_engine_startup(self)', generated)
        self.assertIn('_flash_s5_hooks.park_prefill(self, scheduler_output)',
                      generated)
        self.assertIn('_flash_s5_hooks.before_step_with_batch_queue(self)',
                      generated)
        self.assertIn('return None, False', generated)
        for seam in ('before_abort', 'before_pause', 'before_finish_pause',
                     'before_reset_prefix', 'before_engine_shutdown'):
            self.assertIn('_flash_s5_hooks.' + seam, generated)
        self.assertEqual(eh.revert_core(generated, source), source)

    def test_worker_transform_exposes_only_the_typed_rpcs(self):
        source = (ROOT / 'reference/vllm-src/v1/worker/gpu_worker.py').read_text()
        generated = eh.generate_worker(source)
        for name in ('open', 'submit', 'poll', 'commit', 'cancel'):
            self.assertIn('def flash_s5_' + name + '(', generated)
        self.assertEqual(eh.revert_worker(generated, source), source)

    def test_generated_worker_rpc_reaches_the_native_adapter(self):
        stack = make_worker_stack(1, output_rank=0)
        transaction = make_transaction()
        metadata = MetadataSnapshot.capture(
            transaction, (BatchRow('req-a', 0, 0, 0, 8, 0),), block_ids=(10, 11))
        ack = stack.worker.flash_s5_open(ah.OpenPrefill(transaction, 8, metadata))
        self.assertIsInstance(ack, ah.OpenAck)
        self.assertEqual(ack.lane.worker_rank, 1)
        self.assertIs(stack.server.worker, stack.worker)
        self.assertEqual(stack.state.prefill.state, 'ready')

    def test_generated_worker_rpc_refuses_a_wrong_method(self):
        stack = make_worker_stack(0, output_rank=0)
        transaction = make_transaction()
        metadata = MetadataSnapshot.capture(
            transaction, (BatchRow('req-a', 0, 0, 0, 8, 0),), block_ids=(10, 11))
        message = ah.OpenPrefill(transaction, 8, metadata)
        with self.assertRaisesRegex(E3ContractError, 'disagree'):
            eh.worker_rpc(stack.worker, 'submit', message)


class PackageRecordTests(unittest.TestCase):
    def test_new_modules_are_mounted_and_the_record_stays_unarmable(self):
        for name in ('native_backend', 'native_resources', 'engine_driver',
                     'qsa_operation'):
            self.assertIn(name, make.E3_MODULES)
        source = (ROOT / 'patches/s5_engine/make_s5_overlap.py').read_text()
        self.assertIn("'armable': False", source)
        for gap in ('flash_s5_prefill_admission', '_flash_s5_graph_capture',
                    'SOURCE_HISTORY',
                    'prefix/layer/suffix dispatcher is wired',
                    'flat block ids are not a QSA history map'):
            self.assertIn(gap, source)

    def test_generated_package_imports_the_new_modules(self):
        for name in ('native_backend', 'native_resources', 'engine_driver'):
            packaged = make.package(
                (ROOT / 'patches/s5_engine' / (name + '.py')).read_text())
            self.assertIn('from .async_handoff import', packaged)
            compile(packaged, name + '.py', 'exec')


class NativeRefusalReviewTests(unittest.TestCase):
    def test_enable_refuses_before_cuda_allocation_or_transport(self):
        with mock.patch.object(nb, 'NativeDevice') as allocate, mock.patch.object(eh, 'install') as transport:
            with self.assertRaisesRegex(su.StartupError, 'incomplete'):
                su.install_builder(types.SimpleNamespace(), {}, env={'FLASH_S5_OVERLAP':'1'})
            with self.assertRaisesRegex(su.StartupError, 'incomplete'):
                su.install_engine(types.SimpleNamespace(), env={'FLASH_S5_OVERLAP':'1'})
            with self.assertRaisesRegex(su.StartupError, 'incomplete'):
                nr.build_worker_runtime(types.SimpleNamespace(), {}, env={'FLASH_S5_OVERLAP':'1'})
            allocate.assert_not_called(); transport.assert_not_called()

    def test_no_env_override_can_bypass_refusal(self):
        with self.assertRaises(su.StartupError):
            su.install_engine(object(), env={'FLASH_S5_OVERLAP':'1', 'FLASH_S5_NATIVE_READY':'1'})

    def test_scratch_ownership_retains_every_allocation(self):
        import gc, weakref
        torch = FakeTorch()
        refs = []
        allocate = torch.empty
        def tracked(*args, **kwargs):
            tensor = allocate(*args, **kwargs); refs.append(weakref.ref(tensor)); return tensor
        torch.empty = tracked
        bindings, inventory, owned = nr._allocate_scratch(nb.NativeDevice(torch, 'cuda:0'), 1024)
        gc.collect()
        self.assertEqual(len(bindings), len(owned))
        self.assertTrue(all(ref() is not None for ref in refs))
        self.assertEqual({address for spec, address in bindings}, {b.address for b in owned})

    def test_existing_workspace_is_never_reinitialized(self):
        workspace = FakeWorkspaceModule(); workspace.manager._num_lanes = 1
        original = workspace.manager
        with self.assertRaisesRegex(nr.NativeResourcesError, 'before model'):
            nr._ensure_workspace_lanes(workspace, object(), 1)
        self.assertIs(original, workspace.manager)
        self.assertEqual(workspace.init_calls, [])

    def test_generic_event_cannot_certify_state_write(self):
        stack = make_worker_stack(0)
        txn = make_transaction()
        stack.state.declare_request(txn.request_uid, txn.request_generation)
        meta = MetadataSnapshot.capture(txn, (BatchRow('req-a', 0, 0, 0, 8, 0),))
        handle = stack.runtime.open_prefill(txn, meta)
        ticket = OperationTicket(txn, 0, handle.lane_id, 0, DependencyRole.COMMIT)
        with self.assertRaises(ah.ConfigurationError):
            stack.backend.submit(ticket)
        self.assertFalse(stack.backend.native_verified)
        self.assertFalse(nb.RemoteCompletionBackend(ah.HandoffConfig.from_env({'FLASH_S5_OVERLAP': '1'})).native_verified)

    def test_callable_alone_does_not_enable_fake_split_capture(self):
        device = nb.NativeDevice(FakeTorch(), 'cuda:0')
        callback = mock.Mock()
        backend = nb.NativeGraphCaptureBackend(device, callback)
        with self.assertRaisesRegex(ah.ConfigurationError, 'green context'):
            backend.capture('split', object(), object(), types.SimpleNamespace(reserved_bytes=1))
        callback.assert_not_called()
        self.assertEqual(device.torch.cuda.graphs, [])


@mock.patch('s5_startup.require_native_ready', lambda: None)
@mock.patch.dict(nb._ROLE_KIND, {DependencyRole.COMMIT: CompletionKind.STATE_WRITE})
class ControllerReviewTests(unittest.TestCase):
    def drive_to(self, fixture, phase):
        for _ in range(100):
            if fixture.driver.phase == phase: return
            if fixture.executor.pending:
                fixture.executor.deliver(); fixture.executor.settle()
            fixture.driver.advance()
        self.fail('phase not reached: ' + phase)

    def test_no_prefix_publication_until_commit_ack_all_ranks(self):
        fixture = EngineFixture(); fixture.park()
        self.drive_to(fixture, 'commit_ack_sent')
        self.assertEqual(fixture.composition.frontier.committed_blocks(), ())
        delivered = fixture.executor.deliver(withhold=(3,)); fixture.executor.settle(delivered)
        for _ in range(3): fixture.driver.advance()
        self.assertEqual(fixture.scheduler.update_calls, [])
        self.assertEqual(fixture.composition.frontier.committed_blocks(), ())
        fixture.executor.release_hold((3,)); fixture.executor.settle()
        fixture.driver.advance()
        self.assertEqual(len(fixture.scheduler.update_calls), 1)

    def test_opened_but_unsubmitted_workers_receive_cancel(self):
        fixture = EngineFixture(); txn, _, _ = fixture.park()
        fixture.driver.advance()  # open enqueued
        fixture.executor.deliver(); fixture.executor.settle()
        fixture.driver.stop('abort')
        for _ in range(50):
            if fixture.executor.pending:
                fixture.executor.deliver(); fixture.executor.settle()
            fixture.driver.advance()
            if not fixture.driver.active: break
        self.assertFalse(fixture.driver.active)
        self.assertTrue(all(w.server.handoff is None and w.state.prefill.state == 'free' for w in fixture.workers))
        self.assertNotIn(txn, fixture.composition.frontier._epoch)
        self.assertTrue(any(item[0] == 'flash_s5_cancel' for item in fixture.executor.queued))

    def test_new_transaction_does_not_inherit_old_reply_counts(self):
        fixture = EngineFixture(); fixture.run()
        fixture.composition.frontier.retire_blocks((10,11))
        fixture.park(transaction=make_transaction(seq=2))
        self.assertEqual(fixture.composition.dispatcher.total_replies, 0)
        fixture.driver.advance()  # second open queued, not delivered
        for _ in range(3): fixture.driver.advance()
        self.assertEqual(fixture.driver.phase, 'open_sent')
        self.assertEqual(len(fixture.scheduler.update_calls), 1)

    def test_scheduler_exception_never_retries_partial_publication(self):
        fixture = EngineFixture(); fixture.park()
        self.drive_to(fixture, 'commit_ack_sent')
        fixture.executor.deliver(); fixture.executor.settle()
        calls = []
        def bad(*args):
            calls.append(1); raise RuntimeError('scheduler mutated then failed')
        fixture.scheduler.update_from_output = bad
        with self.assertRaises(RuntimeError): fixture.driver.advance()
        with self.assertRaises(ed.DriverError): fixture.driver.advance()
        self.assertEqual(calls, [1])
        self.assertEqual(fixture.composition.frontier.committed_blocks(), ())
        self.assertEqual(fixture.composition.scheduler_leases.held_blocks(), (10,11))

    def test_native_sender_cannot_fall_back_to_blocking_enqueue(self):
        fixture = EngineFixture(); fixture.park()
        del fixture.executor._flash_s5_bounded_send
        with self.assertRaisesRegex(ed.DriverError, 'bounded command'):
            fixture.driver.advance()
        self.assertEqual(fixture.executor.queued, [])

    def test_admission_rejects_duplicate_blocks_and_freezes_state(self):
        with self.assertRaisesRegex(ed.DriverError, 'duplicate'):
            ed.PrefillAdmission(make_transaction(), (10,10), 'state')
        value = {'nested':[1]}
        admitted = ed.PrefillAdmission(make_transaction(), (10,), value)
        value['nested'].append(2)
        self.assertEqual(admitted.state_value, (('nested',(1,)),))

    def test_poll_requires_every_rank_and_preserves_proof_origin(self):
        fixture = EngineFixture(); txn, _, _ = fixture.park()
        self.drive_to(fixture, 'poll_sent')
        dispatcher = fixture.composition.dispatcher
        reports = [w.server.poll(ah.PollOperations(txn, 8)) for w in fixture.workers]
        with self.assertRaisesRegex(ah.HandoffError, 'every rank'):
            dispatcher._apply_reply('poll', txn, (), reports[:-1])
        with self.assertRaisesRegex(ah.HandoffError, 'another rank'):
            dispatcher._apply_reply('poll', txn, (), [reports[1], reports[0], *reports[2:]])

    def test_commit_rejects_different_worker_epochs(self):
        fixture = EngineFixture(); txn, _, _ = fixture.park()
        self.drive_to(fixture, 'commit_ack_sent')
        with self.assertRaisesRegex(ah.HandoffError, 'epochs differ'):
            fixture.composition.dispatcher._apply_reply('commit', txn, (),
                [ah.CommitAck(txn, 1, True), ah.CommitAck(txn, 2, True),
                 ah.CommitAck(txn, 1, True), ah.CommitAck(txn, 1, True)])
        self.assertEqual(fixture.composition.frontier.committed_blocks(), ())

    def test_stop_failure_quarantines_instead_of_repeating_side_effect(self):
        fixture = EngineFixture(); fixture.park(); fixture.driver.stop('abort')
        with mock.patch.object(fixture.driver, '_drive_stop', side_effect=RuntimeError('stop failed')) as stop:
            with self.assertRaises(RuntimeError): fixture.driver.advance()
            with self.assertRaises(ed.DriverError): fixture.driver.advance()
            self.assertEqual(stop.call_count, 1)
        self.assertEqual(fixture.composition.dispatcher.active.phase, ah.Phase.QUARANTINED)

    def test_reused_blocks_need_an_allocator_retirement_observation(self):
        fixture = EngineFixture(); fixture.run()
        with self.assertRaisesRegex(ah.HandoffError, 'retirement'):
            fixture.park(transaction=make_transaction(seq=2))
        self.assertEqual(fixture.composition.dispatcher.active.phase, ah.Phase.QUARANTINED)


if __name__ == '__main__':
    unittest.main()
