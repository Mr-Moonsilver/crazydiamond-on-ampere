"""Native reply head-of-line blocking behind async decode output (held-f repro).

The generated WorkerProc reply path (``_execute_worker_rpc``, ``handle_output``,
``enqueue_output``, ``async_output_busy_loop``, ``worker_busy_loop``) is executed
from the pinned vLLM source through ``executor_hooks.generate_executor``. The
engine side is the real owned ``ReplyTransport`` and the generated
``collective_rpc``/``StockLocalFuture``. Four worker processes are four thread
pairs (RPC loop + async output loop), as in ``WorkerProc``.

Declared leaves (real engine object -> leaf, why equivalent for this question):

* ``MessageQueue`` broadcast/response rings -> ``queue.Queue`` FIFOs. The
  question is reply *order*; the shm ring is FIFO per reader and the response
  ring is single-writer/single-reader FIFO, which a Queue reproduces. Each
  broadcast reader gets its own unpickled copy, as from the shm ring.
* ``GPUWorker.sample_tokens`` -> returns an ``AsyncModelRunnerOutput`` leaf whose
  ``get_output`` waits on an Event. The real ``AsyncOutput.get_output`` blocks on
  ``copy_event.synchronize()`` (the decode batch's device completion); the Event
  stands for that completion and is released by the test.
* ``GPUWorker.flash_s5_submit`` -> returns a typed tuple at once, stamping the
  host time it ran. The real handler enqueues kernels and returns without a
  device wait (``lane_dispatch.submit``); only its reply path is under test.
* ``vllm.platforms.current_platform`` -> a no-op ``set_device`` (no CUDA here).
* ``WorkerProc.ResponseStatus`` -> two sentinel objects (the transport only
  compares identity with the SUCCESS value it was given).
* The process-global worker progress trace (``executor_hooks._worker_progress``,
  one per worker process) -> one trace per rank in this single test process.
"""
import ast
from contextlib import nullcontext
from functools import partial
import hashlib
from pathlib import Path
import pickle
from queue import Empty, Queue
import sys
import threading
import time
import traceback
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'patches/s5_engine'), str(ROOT / 'harness')]
import async_handoff as ah  # noqa: E402
import engine_driver as ed  # noqa: E402
import executor_hooks as eh  # noqa: E402
import native_measurement as nm  # noqa: E402

REFERENCE = ROOT / 'reference/vllm-src' / eh.EXECUTOR_PATH
WORKER_METHODS = ('_execute_worker_rpc', 'handle_output', 'enqueue_output',
                  'async_output_busy_loop', 'worker_busy_loop')
STAGE_OF = {0: 0, 1: 0, 2: 1, 3: 1}
OUTPUT_RANK = 2  # world_size - tp_size: PP1 TP0, the stock execute/sample reply rank


class Stop(BaseException):
    """Ends a fixture loop; BaseException so the generated ``except Exception`` passes it."""


class AsyncModelRunnerOutput:
    """Leaf base class: the generated code only tests ``isinstance`` against it."""


class Status:
    SUCCESS = object()
    FAILURE = object()


def generated_executor():
    source = REFERENCE.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != eh.EXECUTOR_SHA256:
        raise unittest.SkipTest('pinned vLLM executor source is not available')
    return eh.generate_executor(source)


def worker_proc_class(generated=None):
    """The generated WorkerProc reply-path methods bound on a fixture class."""
    text = generated if generated is not None else generated_executor()
    tree = ast.parse(text)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WorkerProc')
    found = {n.name: n for n in owner.body if isinstance(n, ast.FunctionDef)}
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')],
                                             level=0)] + [found[name] for name in WORKER_METHODS],
                        type_ignores=[])
    scope = {'_flash_s5_hooks': eh, 'Any': object, 'AsyncModelRunnerOutput': AsyncModelRunnerOutput,
             'WorkerProc': types.SimpleNamespace(ResponseStatus=Status), 'partial': partial,
             'traceback': traceback, 'cloudpickle': None,
             'logger': types.SimpleNamespace(exception=lambda *a, **k: None)}
    exec(compile(ast.fix_missing_locations(module), eh.EXECUTOR_PATH, 'exec'), scope)
    return type('GeneratedWorkerProc', (), {name: scope[name] for name in WORKER_METHODS})


class Inbox:
    """One worker's broadcast reader (shm ring leaf)."""

    def __init__(self):
        self.q = Queue()

    def dequeue(self, indefinite=False, timeout=None):
        item = self.q.get()
        if item is Stop:
            raise Stop()
        return item


class Reader:
    """Engine-side response reader for one rank (single-reader FIFO leaf)."""

    def __init__(self):
        self.q = Queue()

    def dequeue(self, timeout):
        try:
            return self.q.get(timeout=timeout)
        except Empty:
            raise TimeoutError from None

    def shutdown(self):
        pass


class Writer:
    """Worker-side response writer for one rank; records emit times."""

    def __init__(self, reader, clock):
        self.reader, self.clock, self.emitted = reader, clock, []

    def enqueue(self, result):
        if result is Stop:
            raise Stop()
        self.emitted.append((self.clock(), result[1]))
        self.reader.q.put(result)


class DecodeOutput(AsyncModelRunnerOutput):
    def __init__(self, done, label):
        self.done, self.label = done, label

    def get_output(self):
        if not self.done.wait(10):
            raise RuntimeError('fixture decode completion never released')
        return ('decode', self.label)


class GPUWorkerLeaf:
    def __init__(self, rank, clock):
        self.rank, self.clock = rank, clock
        self.decode_done = {}
        self.ran = []

    def sample_tokens(self, label):
        done = self.decode_done.setdefault(label, threading.Event())
        self.ran.append((self.clock(), 'sample_tokens', label))
        return DecodeOutput(done, label)

    def flash_s5_submit(self, message):
        self.ran.append((self.clock(), 'flash_s5_submit', message))
        return ('accepted', self.rank, message, 'skipped' if STAGE_OF[self.rank] != message[1] else 'target')


class Broadcast:
    """The engine's one broadcast ring, fanned out to every worker inbox."""

    def __init__(self, inboxes):
        self.inboxes = inboxes
        self.sent = []
        self.current_idx = 0
        self.n_remote_reader = 0
        self.buffer = types.SimpleNamespace(max_chunks=10, max_chunk_bytes=1024 ** 2, n_reader=4,
                                            get_metadata=lambda i: nullcontext(bytearray(5)))

    def enqueue(self, item):
        # The shm ring delivers one serialized envelope; each reader unpickles
        # its own copy (a worker popping a kwarg never affects a peer).
        wire = pickle.dumps(item)
        self.sent.append(pickle.loads(wire))
        for inbox in self.inboxes:
            inbox.q.put(pickle.loads(wire))

    def shutdown(self):
        pass


class Network:
    """Four generated WorkerProc reply paths behind the real owned transport."""

    def __init__(self, test, generated=None, clock=time.monotonic):
        self.clock = clock
        proc_class = worker_proc_class(generated)
        self.readers = [Reader() for _ in range(4)]
        inboxes = [Inbox() for _ in range(4)]
        self.procs, self.threads = [], []
        platforms = types.ModuleType('vllm.platforms')
        platforms.current_platform = types.SimpleNamespace(set_device=lambda device: None)
        saved = {name: sys.modules.get(name) for name in ('vllm', 'vllm.platforms')}
        sys.modules.setdefault('vllm', types.ModuleType('vllm'))
        sys.modules['vllm.platforms'] = platforms

        def restore():
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        test.addCleanup(restore)
        for rank in range(4):
            proc = proc_class()
            proc.rank = rank
            proc.worker = GPUWorkerLeaf(rank, clock)
            proc.use_async_scheduling = True
            proc.async_output_queue = Queue()
            proc.worker_response_mq = Writer(self.readers[rank], clock)
            proc.rpc_broadcast_mq = inboxes[rank]
            self.procs.append(proc)
            for target in (proc.worker_busy_loop, proc.async_output_busy_loop):
                thread = threading.Thread(target=self._run, args=(target,), daemon=True)
                thread.start()
                self.threads.append(thread)
        self.executor = types.SimpleNamespace(
            workers=[], response_mqs=self.readers, rpc_broadcast_mq=Broadcast(inboxes),
            futures_queue=[], is_failed=False)
        self.executor.collective_rpc = types.MethodType(stock_collective_rpc(), self.executor)

        def stock_reader(*args, **kwargs):
            raise AssertionError('stock reply reader forbidden under the owned transport')
        self.executor._flash_s5_stock_collective_rpc = stock_reader
        self.transport = eh.install(self.executor, config=ah.HandoffConfig.from_env({'FLASH_S5_OVERLAP': '1'}),
                                    success_status=Status.SUCCESS)
        test.addCleanup(self.close)

    @staticmethod
    def _run(target):
        try:
            target()
        except Stop:
            pass

    def close(self):
        for proc in self.procs:
            proc.rpc_broadcast_mq.q.put(Stop)
            proc.async_output_queue.put(StopOutput())
        for event_map in (p.worker.decode_done for p in self.procs):
            for event in event_map.values():
                event.set()
        for thread in self.threads:
            thread.join(2)
        self.transport.close(1)

    def native_submit(self, op_seq, stage, meta=False, direct=False):
        """What NativeSegmentDriver._submit does: all-rank replies, one broadcast."""
        envelope = ('flash_s5_submit', ((op_seq, stage),), {}, None)

        def send(sequence):
            wire = (ed.NativeProgress.with_meta(envelope, nm.encode_progress_meta(sequence, direct))
                    if meta else envelope)
            self.executor.rpc_broadcast_mq.enqueue(wire)
        extra = {'direct': True} if direct else {}
        return self.transport.submit('segment', send, ranks=(0, 1, 2, 3), timeout_s=10, **extra)

    def worker_traces(self, test):
        """Leaf: one worker trace per rank (see module docstring)."""
        from collections import deque
        states = {rank: (nm.ProgressTrace('worker', rank), deque()) for rank in range(4)}
        original = eh._worker_progress
        eh._worker_progress = lambda rank: states[rank]
        test.addCleanup(setattr, eh, '_worker_progress', original)
        return states

    def wait_ready(self, sequence, timeout):
        end = self.clock() + timeout
        while self.clock() < end:
            self.transport.poll(8)
            if self.transport.call_ready(sequence):
                return True
            time.sleep(0.001)
        return False

    def replied_ranks(self, sequence):
        self.transport.poll(8)
        call = self.transport._calls.get(sequence)
        return set() if call is None else set(call.replies)


class StopOutput(AsyncModelRunnerOutput):
    def get_output(self):
        raise Stop()


def stock_collective_rpc():
    """The generated E3 ``collective_rpc`` wrapper (owned-transport branch)."""
    text = generated_executor()
    tree = ast.parse(text)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiprocExecutor')
    node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == 'collective_rpc')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')],
                                             level=0), node], type_ignores=[])
    scope = {'_flash_s5_hooks': eh}
    exec(compile(ast.fix_missing_locations(module), eh.EXECUTOR_PATH, 'exec'), scope)
    return scope['collective_rpc']


class NativeReplyHeadOfLine(unittest.TestCase):
    HOLD_S = 0.25

    def test_current_rank2_native_reply_waits_for_pending_decode_output(self):
        net = Network(self)
        # One decode batch in flight: the stock sample reply comes from rank 2 only.
        decode = net.executor.collective_rpc('sample_tokens', args=('b1',), non_block=True,
                                             unique_reply_rank=OUTPUT_RANK, timeout=10)
        sent = net.clock()
        native = net.native_submit(op_seq=7, stage=0)
        time.sleep(self.HOLD_S)
        # Ranks 0, 1 and 3 have no stock reply ahead of it; rank 2 does.
        self.assertEqual(net.replied_ranks(native), {0, 1, 3})
        self.assertFalse(net.transport.call_ready(native))
        ran = [t for t, name, _ in net.procs[OUTPUT_RANK].worker.ran if name == 'flash_s5_submit']
        self.assertEqual(len(ran), 1)
        self.assertLess(ran[0] - sent, self.HOLD_S / 2,
                        'rank 2 executed the native command at once; only its reply is held')
        net.procs[OUTPUT_RANK].worker.decode_done.setdefault('b1', threading.Event()).set()
        self.assertTrue(net.wait_ready(native, 5))
        values, lag = net.transport.take(native)
        self.assertEqual([v[1] for v in values], [0, 1, 2, 3])
        self.assertGreaterEqual(lag[OUTPUT_RANK], self.HOLD_S)
        self.assertTrue(all(lag[r] < self.HOLD_S / 2 for r in (0, 1, 3)), lag)
        self.assertEqual(decode.result(5), ('decode', 'b1'))
        # Reply order on rank 2 is RPC order: decode output first, then the native ACK.
        emitted = [value for _, value in net.procs[OUTPUT_RANK].worker_response_mq.emitted]
        self.assertEqual(emitted[0], ('decode', 'b1'))
        self.assertEqual(emitted[1][0], 'accepted')

    def test_current_each_native_command_pays_the_decode_period(self):
        """One-in-flight native commands behind a steady decode stream: one per period."""
        net = Network(self)
        period, commands = 0.04, 6
        worker = net.procs[OUTPUT_RANK].worker
        stop = threading.Event()
        decodes = []

        def decode_stream():
            # Two decode batches in flight; each completes one period after the
            # previous one, and the engine resolves it (as step_with_batch_queue).
            index, pending = 0, []
            while not stop.is_set():
                label = 'b%d' % index
                pending.append((label, net.executor.collective_rpc(
                    'sample_tokens', args=(label,), non_block=True,
                    unique_reply_rank=OUTPUT_RANK, timeout=10)))
                if len(pending) >= 2:
                    time.sleep(period)
                    done, future = pending.pop(0)
                    worker.decode_done.setdefault(done, threading.Event()).set()
                    decodes.append(future.result(5))
                index += 1
            for event in list(worker.decode_done.values()):
                event.set()
            for done, future in pending:
                worker.decode_done.setdefault(done, threading.Event()).set()
                decodes.append(future.result(5))
        engine = threading.Thread(target=decode_stream, daemon=True)
        start = net.clock()
        engine.start()
        try:
            time.sleep(period / 2)
            for op_seq in range(commands):
                native = net.native_submit(op_seq=op_seq, stage=op_seq % 2)
                self.assertTrue(net.wait_ready(native, 5))
                net.transport.take(native)
            elapsed = net.clock() - start
        finally:
            stop.set()
            engine.join(5)
        # Each command waited for at least one pending decode output on rank 2.
        self.assertGreaterEqual(elapsed, commands * period * 0.8)
        self.assertTrue(all(value[0] == 'decode' for value in decodes))


def rows(trace, tag):
    names = nm.PROGRESS_ROWS[tag]
    return [dict(zip(names, row[1:])) for row in trace.rows if row[0] == tag]


class ProgressReceipts(unittest.TestCase):
    """Held-f receipts over the generated worker loop and the real transport."""
    HOLD_S = 0.25

    def test_trace_separates_rank2_reply_wait_from_worker_host_time(self):
        net = Network(self)
        states = net.worker_traces(self)
        progress = ed.NativeProgress('trace', net.executor)
        progress.begin(dict(txn=1))
        decode = net.executor.collective_rpc('sample_tokens', args=('b1',), non_block=True,
                                             unique_reply_rank=OUTPUT_RANK, timeout=10)
        native = net.native_submit(op_seq=7, stage=0, meta=True)
        time.sleep(self.HOLD_S)
        net.procs[OUTPUT_RANK].worker.decode_done.setdefault('b1', threading.Event()).set()
        self.assertTrue(net.wait_ready(native, 5))
        net.transport.take(native)
        self.assertEqual(decode.result(5), ('decode', 'b1'))
        # The reserved kwarg reached every worker loop and no worker method saw it
        # (the leaf methods take no keyword arguments).
        self.assertEqual([sorted(item[2]) for item in net.executor.rpc_broadcast_mq.sent],
                         [[nm.PROGRESS_META_KEY]] * 2)
        for rank in range(4):
            got = {row['kind']: row for row in rows(states[rank][0], 'r')}
            self.assertEqual(set(got), {'sample_tokens', 'flash_s5_submit'}, rank)
            sample, submit = got['sample_tokens'], got['flash_s5_submit']
            self.assertEqual((sample['seq'], submit['seq']), (decode.sequence, native))
            self.assertTrue(sample['t_enter'] <= sample['t_exit'] and submit['t_enter'] <= submit['t_exit'])
            self.assertEqual(submit['direct'], 0)
            self.assertEqual(submit['replied'], 1)
            wait_ns = submit['t_emit'] - submit['t_exit']
            if rank == OUTPUT_RANK:
                # Head-of-line wait: emit after the decode output it queued behind.
                self.assertEqual(sample['replied'], 1)
                self.assertGreaterEqual(sample['t_emit'], sample['t_exit'] + int(self.HOLD_S * 0.9e9))
                self.assertGreaterEqual(wait_ns, int(self.HOLD_S * 0.9e9))
                self.assertGreaterEqual(submit['t_emit'], sample['t_emit'])
            else:
                # Not the stock output rank: no reply, row recorded at handler exit.
                self.assertEqual((sample['replied'], sample['t_emit']), (0, None))
                self.assertLess(wait_ns, int(self.HOLD_S * 0.5e9))
        engine_stock = rows(progress.trace, 's')
        waits = rows(progress.trace, 'w')
        self.assertEqual([(r['seq'], r['kind']) for r in engine_stock], [(decode.sequence, 'sample_tokens')])
        self.assertTrue(engine_stock[0]['t_submit'] <= engine_stock[0]['t_recv'] <= engine_stock[0]['t_take'])
        self.assertGreaterEqual(engine_stock[0]['t_recv'] - engine_stock[0]['t_submit'], int(self.HOLD_S * 0.9e9))
        self.assertEqual([(w['seq'], w['kind'], w['sends'], w['capped']) for w in waits],
                         [(decode.sequence, 'sample_tokens', 0, 0)])
        self.assertEqual(progress.counters['stock_rpcs'], 1)
        # A flush is bounded parts of one line number; the trace is then empty.
        lines = states[OUTPUT_RANK][0].lines()
        self.assertTrue(lines and all(len(t) < nm.PROGRESS_LINE_BYTES for t in lines))
        self.assertEqual(states[OUTPUT_RANK][0].rows, [])

    def test_unrecorded_rpcs_carry_no_meta_and_leave_no_rows(self):
        net = Network(self)
        states = net.worker_traces(self)
        decode = net.executor.collective_rpc('sample_tokens', args=('b1',), non_block=True,
                                             unique_reply_rank=OUTPUT_RANK, timeout=10)
        native = net.native_submit(op_seq=7, stage=0)
        net.procs[OUTPUT_RANK].worker.decode_done.setdefault('b1', threading.Event()).set()
        self.assertTrue(net.wait_ready(native, 5))
        net.transport.take(native)
        self.assertEqual(decode.result(5), ('decode', 'b1'))
        self.assertTrue(all(item[2] == {} for item in net.executor.rpc_broadcast_mq.sent))
        self.assertIsNone(decode._progress)
        for rank in range(4):
            # Ordered emits still keep the FIFO bookkeeping aligned (None entries).
            self.assertEqual(rows(states[rank][0], 'r'), [])
            self.assertEqual(list(states[rank][1]), [])

    def test_line_parts_stay_below_the_docker_split_and_reassemble(self):
        trace = nm.ProgressTrace('engine', limit=nm.PROGRESS_MAX_ROWS)
        for seq in range(1500):
            trace.record('n', seq, 'segment', seq, seq % 2, seq % 48, 'tick', 10**15 + seq * 1000,
                         10**15 + seq * 1000 + 5, None, None, 10**15 + seq * 1000 + 900, None,
                         10**15 + seq * 1000 + 990, 'accepted', 0)
        for seq in range(5000):
            trace.record('s', seq, 'sample_tokens', 10**15 + 1, 10**15 + 2, 10**15 + 3)
        self.assertEqual(trace.dropped, 1500 + 5000 - nm.PROGRESS_MAX_ROWS)
        lines = trace.lines(txn=3, chunk_start=0, chunk_end=1024)
        bodies = [__import__('json').loads(t[len(nm.PROGRESS_LINE):]) for t in lines]
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(len(t) < nm.PROGRESS_LINE_BYTES for t in lines), max(map(len, lines)))
        self.assertEqual([b['part'] for b in bodies], list(range(len(bodies))))
        self.assertEqual({(b['line'], b['parts'], b['base_ns'], b['txn']) for b in bodies},
                         {(0, len(bodies), 10**15, 3)})
        self.assertEqual(bodies[0]['dropped'], 1500 + 5000 - nm.PROGRESS_MAX_ROWS)
        self.assertTrue(all('fields' not in b and 'dropped' not in b for b in bodies[1:]))
        joined = [r for b in bodies for r in b['rows'].get('n', [])]
        self.assertEqual([r[0] for r in joined], list(range(1500)))
        self.assertEqual(joined[1][6:8], [1, 1])  # microsecond offsets from base_ns
        self.assertEqual(sum(len(b['rows'].get('s', [])) for b in bodies), nm.PROGRESS_MAX_ROWS - 1500)
        self.assertEqual(trace.lines(), [])

    def test_setting_and_meta_contract(self):
        self.assertIsNone(nm.progress_mode({}))
        self.assertIsNone(nm.progress_mode({nm.PROGRESS_ENV: '0'}))
        self.assertEqual(nm.progress_mode({nm.PROGRESS_ENV: 'burst', 'FLASH_S5_AUTO_ARM': '1'}), 'burst')
        for env in ({nm.PROGRESS_ENV: 'trace'}, {nm.PROGRESS_ENV: 'trace', 'FLASH_S5_AUTO_ARM': '0'},
                    {nm.PROGRESS_ENV: 'on', 'FLASH_S5_AUTO_ARM': '1'}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                nm.progress_mode(env)
        import pickle
        sizes = {len(pickle.dumps(nm.encode_progress_meta(v, d))) for v in (0, 1, 2 ** 40, 2 ** 64 - 1)
                 for d in (False, True)}
        self.assertEqual(sizes, {len(pickle.dumps(bytes(nm.PROGRESS_META_BYTES)))})
        self.assertEqual(nm.decode_progress_meta(nm.encode_progress_meta(12345, True)), (12345, True))
        for bad in (b'', bytes(8), bytes(8) + b'\x02', 'x' * 9):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                nm.decode_progress_meta(bad)


if __name__ == '__main__':
    unittest.main()
