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
* ``GPUWorker.flash_s5_poll`` / ``flash_s5_fail`` -> raise at once: a refused
  native command (direct-eligible / not eligible); only the failure wire and
  the refusal before dispatch are under test.
* ``vllm.platforms.current_platform`` -> a no-op ``set_device`` (no CUDA here).
* ``WorkerProc.ResponseStatus`` -> two sentinel objects (the transport only
  compares identity with the SUCCESS value it was given).
* The process-global worker progress trace (``executor_hooks._worker_progress``,
  one per worker process) -> one trace per rank in this single test process.
* ``NativeSegmentDriver`` state for the pair-boundary round -> ``PairLeaf``. The
  real ``progress_in_wait``/``_wait_permit``/``_progress_owned``/``_route_exchange``
  run on it unchanged; it holds every field they read (checked against their
  AST). Its ``_drain``/``_drive`` apply one reply / record one send with the
  real index and in-flight bookkeeping; the transport and worker behind them
  are what the other classes here and the connected probe cover.
"""
import ast
from contextlib import nullcontext
from functools import partial
import hashlib
import inspect
from pathlib import Path
import pickle
import textwrap
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

    def flash_s5_poll(self, message):
        self.ran.append((self.clock(), 'flash_s5_poll', message))
        raise RuntimeError('fixture worker refusal')

    def flash_s5_fail(self, message):
        self.ran.append((self.clock(), 'flash_s5_fail', message))
        raise RuntimeError('fixture worker refusal')


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

    def __init__(self, test, generated=None, clock=time.monotonic, direct=False):
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
        extra = {'direct_replies': True} if direct else {}
        self.transport = eh.install(self.executor, config=ah.HandoffConfig.from_env({'FLASH_S5_OVERLAP': '1'}),
                                    success_status=Status.SUCCESS, **extra)
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

    def native_submit(self, op_seq, stage, meta=False, direct=False, method='flash_s5_submit'):
        """What NativeSegmentDriver._submit does: all-rank replies, one broadcast."""
        envelope = (method, ((op_seq, stage),), {}, None)

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
            trace.record('n', seq, 'segment', 'layer', seq, seq % 2, seq % 48, 'tick', 10**15 + seq * 1000,
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
        at = nm.PROGRESS_N
        self.assertEqual(joined[1][at['t_submit']:at['t_sent'] + 1], [1, 1])  # µs offsets from base_ns
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


class DirectReplies(unittest.TestCase):
    """Held-f burst piece (a): worker RPC-thread replies and the two-FIFO receiver."""
    HOLD_S = 0.25

    def wait_failure(self, net, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with net.transport._condition:
                if net.transport._failure is not None:
                    return net.transport._failure
            time.sleep(0.002)
        return None

    def test_direct_reply_passes_the_pending_decode_output(self):
        net = Network(self, direct=True)
        states = net.worker_traces(self)
        decode = net.executor.collective_rpc('sample_tokens', args=('b1',), non_block=True,
                                             unique_reply_rank=OUTPUT_RANK, timeout=10)
        start = net.clock()
        native = net.native_submit(op_seq=7, stage=0, meta=True, direct=True)
        self.assertTrue(net.wait_ready(native, self.HOLD_S / 2))
        self.assertLess(net.clock() - start, self.HOLD_S / 2)
        values, lag = net.transport.take(native)
        self.assertEqual([v[1] for v in values], [0, 1, 2, 3])
        self.assertTrue(all(lag[r] < self.HOLD_S / 2 for r in range(4)), lag)
        self.assertFalse(decode.done())
        net.procs[OUTPUT_RANK].worker.decode_done.setdefault('b1', threading.Event()).set()
        self.assertEqual(decode.result(5), ('decode', 'b1'))
        # On rank 2 the native ACK left first, then the decode output it overtook.
        emitted = [value for _, value in net.procs[OUTPUT_RANK].worker_response_mq.emitted]
        self.assertIsInstance(emitted[0], eh.DirectReply)
        self.assertEqual(emitted[0].value[0], 'accepted')
        self.assertEqual(emitted[1], ('decode', 'b1'))
        submit = next(r for r in rows(states[OUTPUT_RANK][0], 'r') if r['kind'] == 'flash_s5_submit')
        self.assertEqual((submit['direct'], submit['replied']), (1, 1))
        self.assertLess(submit['t_emit'] - submit['t_exit'], int(self.HOLD_S * 0.5e9))
        self.assertEqual(list(states[OUTPUT_RANK][1]), [])

    def test_each_class_stays_fifo_under_a_decode_stream(self):
        net = Network(self, direct=True)
        period, commands = 0.04, 12
        worker = net.procs[OUTPUT_RANK].worker
        stop = threading.Event()
        decodes, labels = [], []

        def decode_stream():
            index, pending = 0, []
            while not stop.is_set():
                label = 'b%d' % index
                labels.append(label)
                pending.append((label, net.executor.collective_rpc(
                    'sample_tokens', args=(label,), non_block=True,
                    unique_reply_rank=OUTPUT_RANK, timeout=10)))
                if len(pending) >= 2:
                    time.sleep(period)
                    done, future = pending.pop(0)
                    worker.decode_done.setdefault(done, threading.Event()).set()
                    decodes.append(future.result(5))
                index += 1
            for done, future in pending:
                worker.decode_done.setdefault(done, threading.Event()).set()
                decodes.append(future.result(5))
        engine = threading.Thread(target=decode_stream, daemon=True)
        engine.start()
        try:
            time.sleep(period / 2)
            start = net.clock()
            got = []
            for op_seq in range(commands):
                native = net.native_submit(op_seq=op_seq, stage=op_seq % 2, meta=True, direct=True)
                self.assertTrue(net.wait_ready(native, 5))
                values, _lag = net.transport.take(native)
                got.append([v[2] for v in values])
            elapsed = net.clock() - start
        finally:
            stop.set()
            engine.join(5)
        # Every native reply matched its own command on every rank, in order,
        # and the commands no longer pay one decode period each.
        self.assertEqual(got, [[(op, op % 2)] * 4 for op in range(commands)])
        self.assertLess(elapsed, commands * period * 0.5)
        self.assertEqual(decodes, [('decode', label) for label in labels[:len(decodes)]])
        self.assertIsNone(net.transport._failure)

    def test_wire_with_no_expectation_in_its_class_poisons(self):
        for injected, expect_direct in ((eh.DirectReply(0, ('accepted', 0, None, 'x')), False),
                                        (('decode', 'stray'), True)):
            with self.subTest(injected=type(injected).__name__):
                net = Network(self, direct=True)
                if expect_direct:
                    net.native_submit(op_seq=1, stage=1, meta=True, direct=True)
                else:
                    net.executor.collective_rpc('sample_tokens', args=('held',), non_block=True,
                                                unique_reply_rank=0, timeout=10)
                # Rank 0's receiver sees a wire of the other class first.
                net.readers[0].q.put((Status.SUCCESS, injected))
                self.assertEqual(self.wait_failure(net, 2), 'reply_correlation_failure')

    def direct_calls(self, net, count):
        """Direct expectations on rank 0 whose broadcast is withheld (the wire is injected)."""
        return [net.transport.submit('segment', lambda _seq: None, ranks=(0,), timeout_s=10, direct=True)
                for _ in range(count)]

    def test_direct_reply_is_accepted_only_as_the_exact_direct_head(self):
        # Codex conformance (85a3098): reordered same-class replies were swapped
        # silently. Each DirectReply carries its registered sequence.
        net = Network(self, direct=True)
        s1, s2 = self.direct_calls(net, 2)
        net.readers[0].q.put((Status.SUCCESS, eh.DirectReply(s1, 'one')))
        net.readers[0].q.put((Status.SUCCESS, eh.DirectReply(s2, 'two')))
        self.assertTrue(net.wait_ready(s1, 2) and net.wait_ready(s2, 2))
        self.assertEqual((net.transport.take(s1)[0], net.transport.take(s2)[0]), (['one'], ['two']))
        self.assertIsNone(net.transport._failure)
        for label, wires in (
                ('reordered', lambda a, b: [eh.DirectReply(b, 'two'), eh.DirectReply(a, 'one')]),
                ('delayed-duplicate', lambda a, b: [eh.DirectReply(a, 'one'), eh.DirectReply(a, 'one')]),
                ('unregistered', lambda a, b: [eh.DirectReply(b + 7, 'x')]),
                ('untyped-sequence', lambda a, b: [eh.DirectReply(str(a), 'one')])):
            with self.subTest(case=label):
                net = Network(self, direct=True)
                a, b = self.direct_calls(net, 2)
                for wire in wires(a, b):
                    net.readers[0].q.put((Status.SUCCESS, wire))
                self.assertEqual(self.wait_failure(net, 2), 'reply_correlation_failure')
                # Nothing was delivered under a wrong sequence, and every pending call is poisoned.
                call = net.transport._calls.get(b)
                self.assertTrue(call is not None and not call.ready)
                with self.assertRaises(eh.TransportError):
                    net.transport.take(b)

    def test_direct_reply_pickles_with_its_sequence(self):
        net = Network(self, direct=True)
        native = net.native_submit(op_seq=4, stage=1, meta=True, direct=True)
        self.assertTrue(net.wait_ready(native, 2))
        wires = [value for proc in net.procs for _, value in proc.worker_response_mq.emitted]
        self.assertEqual(len(wires), 4)
        self.assertTrue(all(isinstance(w, eh.DirectReply) and w.sequence == native for w in wires))
        copy = pickle.loads(pickle.dumps(wires[0]))
        self.assertEqual((type(copy), copy.sequence, copy.value), (eh.DirectReply, native, wires[0].value))

    def test_direct_command_failure_poisons_at_once_ahead_of_decode(self):
        net = Network(self, direct=True)
        net.executor.collective_rpc('sample_tokens', args=('b1',), non_block=True,
                                    unique_reply_rank=OUTPUT_RANK, timeout=10)
        start = time.monotonic()
        net.native_submit(op_seq=3, stage=0, meta=True, direct=True, method='flash_s5_poll')
        self.assertEqual(self.wait_failure(net, self.HOLD_S), 'worker_failure')
        self.assertLess(time.monotonic() - start, self.HOLD_S)
        self.assertTrue(any(name == 'flash_s5_poll' for _, name, _ in net.procs[0].worker.ran))

    def test_worker_refuses_a_direct_bit_on_an_ineligible_method(self):
        net = Network(self, direct=True)
        failures = []
        original = eh.report_worker_failure
        eh.report_worker_failure = lambda rank, method, exc: failures.append((rank, method, str(exc)))
        self.addCleanup(setattr, eh, 'report_worker_failure', original)
        net.native_submit(op_seq=5, stage=0, meta=True, direct=True, method='flash_s5_fail')
        self.assertEqual(self.wait_failure(net, 2), 'worker_failure')
        # Refused before dispatch: the handler never ran, and the refusal took the monitored report.
        self.assertFalse(any(name == 'flash_s5_fail' for p in net.procs for _, name, _ in p.worker.ran))
        self.assertIn('direct reply refused', ' '.join(f[2] for f in failures))
        self.assertEqual(nm.PROGRESS_DIRECT_METHODS, frozenset(('flash_s5_submit', 'flash_s5_poll')))

    def test_off_has_no_direct_fifo_and_refuses_a_direct_submission(self):
        net = Network(self)
        self.assertIsNone(net.transport._direct)
        # Off, every receiver runs the unchanged single-FIFO loop.
        self.assertTrue(all(t._target == net.transport._receive_loop for t in net.transport._threads.values()))
        with self.assertRaises(ValueError):
            net.native_submit(op_seq=1, stage=0, meta=True, direct=True)
        with self.assertRaises(ValueError):
            net.transport.submit('segment', lambda _seq: None, ranks=(0,), timeout_s=1, direct=1)
        self.assertEqual(net.transport._calls, {})


class InWaitLoop(unittest.TestCase):
    """executor_hooks.native_progress_wait edges; the driver is a declared leaf
    (its permit and advance are proven on the connected probe, not here)."""

    class Driver:
        def __init__(self, answers):
            self.answers, self.calls = list(answers), 0

        def progress_in_wait(self):
            self.calls += 1
            return self.answers.pop(0) if self.answers else 1

    def progress(self, answers):
        progress = ed.NativeProgress('burst', types.SimpleNamespace(), driver=self.Driver(answers))
        return progress

    def test_cap_bounds_sends_and_is_counted(self):
        progress = self.progress([])
        self.assertEqual(eh.native_progress_wait(progress, lambda: False, None),
                         (progress.WAIT_SEND_LIMIT, 1))
        self.assertEqual(progress.driver.calls, progress.WAIT_SEND_LIMIT)
        self.assertEqual((progress.counters['wait_sends'], progress.counters['wait_capped']),
                         (progress.WAIT_SEND_LIMIT, 1))

    def test_refused_permit_and_ready_end_the_loop(self):
        progress = self.progress([1, 0, None])
        self.assertEqual(eh.native_progress_wait(progress, lambda: False, None), (1, 0))
        self.assertEqual(progress.driver.calls, 3)
        ready = iter((False, False, True))
        progress = self.progress([1, 1, 1])
        self.assertEqual(eh.native_progress_wait(progress, lambda: next(ready), None), (2, 0))

    def test_deadline_ends_the_loop_without_a_send(self):
        progress = self.progress([0] * 1000)
        start = time.monotonic()
        self.assertEqual(eh.native_progress_wait(progress, lambda: False, start + 0.02), (0, 0))
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertGreater(progress.driver.calls, 1)

    def test_wait_needs_burst_owner_thread_and_a_sample_last(self):
        progress = self.progress([])
        progress.begin(dict(txn=1))
        progress.last_stock_kind = 'sample_tokens'
        self.assertFalse(progress.wait_allowed())  # mode current
        progress.mode = 'burst'
        self.assertTrue(progress.wait_allowed())
        progress.last_stock_kind = 'execute_model'
        self.assertFalse(progress.wait_allowed())
        progress.last_stock_kind = 'sample_tokens'
        other = []
        thread = threading.Thread(target=lambda: other.append(progress.wait_allowed()))
        thread.start()
        thread.join(2)
        self.assertEqual(other, [False])
        progress.end(types.SimpleNamespace(info=lambda text: None))
        self.assertFalse(progress.wait_allowed())


class PairLeaf:
    """NativeSegmentDriver state leaf for one in-wait round.

    ``pair_driver`` adds the real driver functions ``progress_in_wait``,
    ``_wait_permit``, ``_progress_owned`` and ``_route_exchange``. The leaf holds the
    driver fields they read (``InWaitPairBoundary`` checks that set against
    their AST) in the open, auto-owned, mid-chunk state: phase ``segments``, the
    LAYER at ``plan[0]`` in flight, ``plan[1]`` next. Two methods are leaves.
    ``_drain`` applies one scripted reply the way the real ``_drain`` does
    (the replied command leaves ``_inflight`` and ``_apply_segment`` advances
    the plan index), or applies nothing (the reply has not arrived). ``_drive``
    records the send the real one would make for ``plan[index]`` and puts it in
    flight; the index advances only when that reply is applied.
    """
    def __init__(self, upcoming, during_drain=None):
        self.calls = []
        self.during_drain = during_drain
        self.progress = ed.NativeProgress('burst', types.SimpleNamespace(), driver=self)
        self.progress.begin(dict(txn=1))
        self.progress.mode = 'burst'
        self.progress.last_stock_kind = 'sample_tokens'
        self.plan = [dict(kind=ed.SegmentKind.LAYER), dict(kind=upcoming)]
        self.index = 0
        self.phase = 'segments'
        self._command = types.SimpleNamespace(kind=ed.SegmentKind.LAYER)
        self._inflight = {7: 'segment'}
        self._retry = None
        self._in_wait = False
        self._tx_fused = False
        self._progress_origin = 'tick'
        self.error = self.stop_kind = self._profile_failure = None
        self._control = None
        self._control_orphans = ()
        self._bootstrap_command = self._transition = None
        self._return_pending = None
        self._bootstrapped = self._split_ready = True
        self.tx = object()
        self.auto_arm = types.SimpleNamespace(manual_owned=False, phase='native')
        self._admission_intent = dict(origin='auto-arm')

    def _drain(self):
        self.calls.append(('drain', self._in_wait, self._progress_origin))
        if self.during_drain == 'no-reply':
            return
        self._inflight.pop(7)
        self.index += 1
        if self.during_drain == 'stock-execute':
            self.progress.last_stock_kind = 'execute_model'

    def _drive(self):
        kind = self.plan[self.index]['kind']
        self.calls.append(('send', kind))
        self._inflight[8] = 'segment'
        self._command = types.SimpleNamespace(kind=kind)


PAIR_REAL = ('progress_in_wait', '_wait_permit', '_progress_owned', '_route_exchange')


def pair_driver(upcoming, during_drain=None, wait=None):
    """A PairLeaf with the real driver functions, read at call time."""
    real = {name: getattr(ed.NativeSegmentDriver, name) for name in PAIR_REAL}
    if wait is not None:
        real['progress_in_wait'] = wait
    return type('PairDriver', (PairLeaf,), real)(upcoming, during_drain)


class InWaitPairBoundary(unittest.TestCase):
    """The permit and wait_allowed are re-read after _drain and before _drive in
    every in-wait round: a flip between them stops the send.

    Runs the real ``progress_in_wait`` with the real ``_wait_permit`` on
    ``PairLeaf``. The structural flip is the one each chunk makes: the drain
    applies the last LAYER's reply and the next descriptor is a tick-only
    PP_SEND or STATE_FINALIZE. A mutant without the post-drain recheck sends
    there, so the test discriminates.
    """

    def round(self, upcoming, during_drain=None, wait=None):
        leaf = pair_driver(upcoming, during_drain, wait)
        self.assertTrue(leaf._wait_permit())
        self.assertTrue(leaf.progress.wait_allowed())
        out = leaf.progress_in_wait()
        self.assertEqual((leaf._in_wait, leaf._progress_origin), (False, 'tick'))
        return out, leaf

    def test_a_reply_that_ends_the_layers_stops_the_send(self):
        for upcoming in (ed.SegmentKind.PP_SEND, ed.SegmentKind.STATE_FINALIZE):
            with self.subTest(upcoming=upcoming.value):
                out, leaf = self.round(upcoming)
                self.assertEqual(out, 0)
                self.assertEqual(leaf.calls, [('drain', True, 'wait')])
                self.assertFalse(leaf._wait_permit())
                self.assertEqual((leaf._inflight, leaf.index), ({}, 1))

    def test_a_stock_execute_after_the_drain_stops_the_send(self):
        out, leaf = self.round(ed.SegmentKind.LAYER, 'stock-execute')
        self.assertTrue(leaf._wait_permit())
        self.assertFalse(leaf.progress.wait_allowed())
        self.assertEqual((out, leaf.calls), (0, [('drain', True, 'wait')]))

    def test_a_reply_not_yet_arrived_stops_the_send(self):
        out, leaf = self.round(ed.SegmentKind.LAYER, 'no-reply')
        self.assertEqual((out, leaf.calls), (0, [('drain', True, 'wait')]))
        self.assertEqual((leaf._inflight, leaf.index), ({7: 'segment'}, 0))

    def test_a_permit_that_holds_after_the_drain_sends_once(self):
        out, leaf = self.round(ed.SegmentKind.LAYER)
        self.assertEqual(out, 1)
        self.assertEqual(leaf.calls, [('drain', True, 'wait'), ('send', ed.SegmentKind.LAYER)])

    def test_a_refused_permit_never_drains(self):
        leaf = pair_driver(ed.SegmentKind.LAYER)
        leaf._inflight = {7: 'segment', 9: 'segment'}
        self.assertIsNone(leaf.progress_in_wait())
        leaf = pair_driver(ed.SegmentKind.LAYER)
        leaf.progress.last_stock_kind = 'execute_model'
        self.assertIsNone(leaf.progress_in_wait())
        self.assertEqual(leaf.calls, [])

    def test_the_mutant_without_the_post_drain_recheck_sends(self):
        source = textwrap.dedent(inspect.getsource(ed.NativeSegmentDriver.progress_in_wait))
        tree = ast.parse(source)
        body = tree.body[0].body[-1].body  # the try body
        recheck = [node for node in body if isinstance(node, ast.If)
                   and 'nested=True' in ast.unparse(node.test)]
        self.assertEqual(len(recheck), 1)
        body.remove(recheck[0])
        scope = dict(vars(ed))
        exec(compile(ast.fix_missing_locations(tree), 'mutant', 'exec'), scope)
        mutant = scope['progress_in_wait']
        for upcoming in (ed.SegmentKind.PP_SEND, ed.SegmentKind.STATE_FINALIZE):
            out, leaf = self.round(upcoming, wait=mutant)
            self.assertEqual(leaf.calls[-1], ('send', upcoming))
        out, leaf = self.round(ed.SegmentKind.LAYER, 'stock-execute', wait=mutant)
        self.assertEqual(leaf.calls[-1], ('send', ed.SegmentKind.LAYER))

    def test_the_leaf_holds_every_field_the_real_functions_read(self):
        read = set()
        for name in PAIR_REAL:
            tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(ed.NativeSegmentDriver, name))))
            read |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                     and isinstance(node.value, ast.Name) and node.value.id == 'self'}
        leaf = pair_driver(ed.SegmentKind.LAYER)
        missing = sorted(name for name in read if not hasattr(leaf, name))
        self.assertEqual(missing, [])
        self.assertTrue({'_drain', '_drive', '_wait_permit', 'progress'} <= read)


if __name__ == '__main__':
    unittest.main()
