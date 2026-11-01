"""Native43 le-fix: the EngineCore broadcast write cannot outlive a failed executor.

l-e (1 Nov 2026): the busy loop sat in MessageQueue.acquire_write (enqueue with timeout=None) from
~11:03:06 while the fatal channel marked the executor failed and shut it down at 11:03:08; EXECUTOR_FAILED
was never processed and the API never saw ENGINE_CORE_DEAD.  These CPU cases run the actual
executor_hooks.bounded_enqueue, the actual generated stock collective_rpc over the pinned
multiproc_executor.py, the actual owned collective_rpc/ReplyTransport and the actual fatal channel
(monitor_fatal_workers over a real pipe).  The ring is a CPU fixture with the pinned acquire_write
timeout contract (declared in native43_le_bounded_enqueue_fixture_leaves.json); the pinned-image probe
(tools/native43_le_fix_enqueueprobe.py) repeats the l-e chain on vLLM's real shared-memory MessageQueue.
"""
import ast
import collections
from contextlib import nullcontext
import json
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'patches/s5_engine'), str(ROOT / 'harness')]
import async_handoff as ah  # noqa: E402
import executor_hooks as hooks  # noqa: E402
from reply_transport import TransportError, WorkerRPCError  # noqa: E402

REFERENCE = ROOT / 'reference/vllm-src/v1/executor/multiproc_executor.py'
STOCK_WRITE = '        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))\n'


class Ring:
    """CPU stand-in for vLLM's MessageQueue writer side (shm_broadcast.py 648-683, 823-880).

    ``free`` writable blocks; a write into a full ring waits until a reader frees a block.  With a
    timeout, acquire_write raises TimeoutError before claiming a block, so nothing is written.
    """

    def __init__(self, free=0, chunks=10):
        self.free = free
        self.cond = threading.Condition()
        self.writes = []
        self.calls = []
        self.shutting_down = False
        self._is_writer = True
        self.n_remote_reader = 0
        self.current_idx = 0
        self.buffer = SimpleNamespace(max_chunks=chunks, max_chunk_bytes=1024 ** 2, n_reader=4,
                                      get_metadata=lambda i: nullcontext(bytearray(5)))

    def enqueue(self, obj, timeout=None):
        self.calls.append(timeout)
        with self.cond:
            if not self.cond.wait_for(lambda: self.free > 0, timeout):
                raise TimeoutError
            self.free -= 1
            self.writes.append(obj)

    def drain(self, n=1):
        with self.cond:
            self.free += n
            self.cond.notify_all()


def fatal_executor(ring, test):
    """An executor with the actual fatal channel installed (monitor_fatal_workers over one real pipe).

    Cleanup marks it shutting down before the pipe closes, so the receiver ends as an orderly EOF."""
    reader, writer = multiprocessing.Pipe(duplex=False)
    executor = SimpleNamespace(rpc_broadcast_mq=ring, response_mqs=[], is_failed=False, futures_queue=collections.deque(),
                               shutting_down=False, world_size=1, failure_callback=None, shutdowns=[])
    executor.workers = [SimpleNamespace(rank=0, proc=SimpleNamespace(pid=os.getpid()), fatal_reader=reader)]

    def shutdown():
        executor.shutdowns.append(time.monotonic())
        executor.shutting_down = True
    executor.shutdown = shutdown
    old = os.environ.get('FLASH_S5_OVERLAP')
    os.environ['FLASH_S5_OVERLAP'] = '1'
    try:
        hooks.monitor_fatal_workers(executor)
    finally:
        if old is None:
            os.environ.pop('FLASH_S5_OVERLAP')
        else:
            os.environ['FLASH_S5_OVERLAP'] = old
    test.addCleanup(writer.close)
    test.addCleanup(setattr, executor, 'shutting_down', True)
    return executor, writer


def publish_fatal(writer):
    """The worker side of l-e: the record report_worker_failure publishes (it then parks)."""
    record = dict(schema='s5-worker-fatal/1', rank=0, pid=os.getpid(), method='execute_model',
                  error_type='QsaDriverError', code='worker-rpc-exception', cause='Row1OracleError',
                  where='qsa_driver.py:382:_tail', detail='native row1 qualification failed: route_mismatch')
    writer.send_bytes(json.dumps(record).encode())
    return time.monotonic()


def background(fn):
    row = SimpleNamespace(value=None, error=None, ended=None)

    def run():
        try:
            row.value = fn()
        except BaseException as exc:  # noqa: BLE001 - the case inspects it
            row.error = exc
        row.ended = time.monotonic()
    row.thread = threading.Thread(target=run, daemon=True)
    row.thread.start()
    return row


def generated_stock_rpc():
    """The actual generated MultiprocExecutor._flash_s5_stock_collective_rpc over the pinned source."""
    generated = hooks.generate_executor(REFERENCE.read_text())
    tree = ast.parse(generated)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiprocExecutor')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_flash_s5_stock_collective_rpc')
    fn.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(
        body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn],
        type_ignores=[]))
    scope = dict(_flash_s5_hooks=hooks, time=time,
                 FutureWrapper=lambda queue, get_response, aggregate: SimpleNamespace(queue=queue))
    exec(compile(module, 'actual-generated-stock-collective-rpc', 'exec'), scope)
    return scope['_flash_s5_stock_collective_rpc'], generated


class BoundedEnqueueTests(unittest.TestCase):
    def test_overlap_off_is_the_stock_unbounded_call(self):
        ring = Ring(free=1)
        executor = SimpleNamespace(rpc_broadcast_mq=ring, is_failed=False)
        hooks.bounded_enqueue(executor, ring, ('m', (), {}, None))
        self.assertEqual((ring.calls, ring.writes), ([None], [('m', (), {}, None)]))

    def test_a_free_block_is_one_bounded_write(self):
        ring = Ring(free=1)
        executor, _ = fatal_executor(ring, self)
        hooks.bounded_enqueue(executor, ring, 'x')
        self.assertEqual((ring.calls, ring.writes), ([hooks.ENQUEUE_SLICE_S], ['x']))
        self.assertEqual(hooks.ENQUEUE_SLICE_S, 1.0)

    def test_the_le_chain_raises_the_stored_worker_rpc_error_within_the_bound(self):
        ring = Ring(free=0)
        executor, writer = fatal_executor(ring, self)
        row = background(lambda: hooks.bounded_enqueue(executor, ring, 'step', slice_s=0.05))
        time.sleep(0.3)
        self.assertTrue(row.thread.is_alive())  # live executor, full ring: keeps waiting
        sent = publish_fatal(writer)
        row.thread.join(3)
        self.assertFalse(row.thread.is_alive(), 'bounded write still blocked after the fatal record')
        self.assertIsInstance(row.error, WorkerRPCError)
        self.assertIs(row.error, executor._flash_s5_fatal_failure)
        self.assertEqual(row.error.record['error_type'], 'QsaDriverError')
        self.assertLess(row.ended - sent, 0.05 + 1.0)  # one slice plus the fatal thread's hand-off
        self.assertEqual(ring.writes, [])  # a timed-out slice writes nothing
        self.assertEqual(len(executor.shutdowns), 1)  # the fatal channel's own teardown still runs

    def test_overlap_off_reproduces_the_le_hang(self):
        """The pre-fix call (stock enqueue, timeout=None) stays blocked after the same fatal record."""
        ring = Ring(free=0)
        executor, writer = fatal_executor(ring, self)
        row = background(lambda: ring.enqueue('step'))  # the parent's exact call
        publish_fatal(writer)
        time.sleep(1.0)
        self.assertTrue(executor.is_failed)
        self.assertTrue(row.thread.is_alive(), 'the pre-fix write should still be blocked')
        ring.drain()  # release the fixture thread
        row.thread.join(2)
        self.assertFalse(row.thread.is_alive())

    def test_a_live_slow_ring_keeps_waiting_and_writes_once(self):
        ring = Ring(free=0)
        executor, _ = fatal_executor(ring, self)
        row = background(lambda: hooks.bounded_enqueue(executor, ring, 'slow', slice_s=0.05))
        time.sleep(0.4)  # about eight slices
        self.assertTrue(row.thread.is_alive())
        ring.drain()
        row.thread.join(2)
        self.assertIsNone(row.error)
        self.assertEqual(ring.writes, ['slow'])
        self.assertGreater(len(ring.calls), 3)

    def test_the_stock_long_wait_warning_keeps_its_cadence(self):
        clock = SimpleNamespace(now=0.0)

        class ScriptedRing(Ring):
            def enqueue(self, obj, timeout=None):
                self.calls.append(timeout)
                if len(self.calls) <= 130:
                    clock.now += timeout
                    raise TimeoutError
                self.writes.append(obj)
        ring = ScriptedRing()
        executor, _ = fatal_executor(ring, self)
        logged = []
        hooks.bounded_enqueue(executor, ring, 'w', clock=lambda: clock.now,
                              long_wait=(60, hooks.LONG_WAIT_FALLBACK[1], lambda msg, n: logged.append(msg % n)))
        self.assertEqual(ring.writes, ['w'])
        self.assertEqual(len(logged), 2)  # after 60 s and after 120 s of waiting, as acquire_write logs
        self.assertTrue(logged[0].startswith('No available shared memory broadcast block found in 60 seconds'))
        # Without an injected triple the queue's own module supplies interval, text and logger (no import).
        self.assertEqual(hooks._long_wait(Ring())[:2], hooks.LONG_WAIT_FALLBACK)
        module = type(sys)('fixture_shm_broadcast')
        module.VLLM_RINGBUFFER_WARNING_INTERVAL, module.LONG_WAIT_TIME_LOG_MSG = 7, 'msg %d'
        module.logger = SimpleNamespace(info=lambda *a: None)
        sys.modules[module.__name__] = module
        self.addCleanup(sys.modules.pop, module.__name__, None)
        queue_type = type('MessageQueue', (Ring,), {'__module__': module.__name__})
        self.assertEqual(hooks._long_wait(queue_type())[:2], (7, 'msg %d'))

    def test_failed_or_shutting_down_without_a_record(self):
        for setup, kind, text in ((lambda e, r: setattr(e, 'is_failed', True), RuntimeError, 'Executor failed.'),
                                  (lambda e, r: setattr(r, 'shutting_down', True), TransportError, 'shutting down'),
                                  (lambda e, r: setattr(e, 'shutting_down', True), TransportError, 'shutting down')):
            ring = Ring(free=0)
            executor, _ = fatal_executor(ring, self)
            setup(executor, ring)
            with self.assertRaises(kind) as caught:
                hooks.bounded_enqueue(executor, ring, 'z', slice_s=0.01)
            self.assertIn(text, str(caught.exception))
            self.assertEqual(ring.writes, [])


class GeneratedSitesTests(unittest.TestCase):
    def test_the_generated_stock_write_goes_through_the_hook_and_reverts(self):
        rpc, generated = generated_stock_rpc()
        source = REFERENCE.read_text()
        self.assertEqual(source.count(STOCK_WRITE), 1)
        self.assertEqual(generated.count('self.rpc_broadcast_mq.enqueue('), 0)
        self.assertEqual(generated.count('_flash_s5_hooks.bounded_enqueue(self, self.rpc_broadcast_mq,'), 1)
        self.assertEqual(hooks.revert_executor(generated), source)
        ring = Ring(free=1)
        executor = SimpleNamespace(rpc_broadcast_mq=ring, response_mqs=[None], is_failed=False,
                                   futures_queue=collections.deque())
        rpc(executor, 'check_health', non_block=True)
        self.assertEqual((ring.calls, ring.writes), ([None], [('check_health', (), {}, None)]))  # overlap off

    def test_the_generated_stock_rpc_fails_loud_on_the_le_chain(self):
        rpc, _ = generated_stock_rpc()
        ring = Ring(free=0)
        executor, writer = fatal_executor(ring, self)
        executor.response_mqs = [None]
        row = background(lambda: rpc(executor, 'execute_model', non_block=True))
        time.sleep(1.5)  # more than one default slice: a live executor keeps waiting
        self.assertTrue(row.thread.is_alive())
        sent = publish_fatal(writer)
        row.thread.join(4)
        self.assertFalse(row.thread.is_alive(), 'generated stock write still blocked after the fatal record')
        self.assertIs(row.error, executor._flash_s5_fatal_failure)
        self.assertLess(row.ended - sent, 2 * hooks.ENQUEUE_SLICE_S + 0.5)
        self.assertEqual(ring.writes, [])
        self.assertEqual(ring.calls[0], hooks.ENQUEUE_SLICE_S)

    def test_the_owned_route_fails_loud_and_poisons_the_transport(self):
        ring = Ring(free=0)
        executor, writer = fatal_executor(ring, self)

        class Reader:
            def dequeue(self, timeout):
                time.sleep(min(timeout or 0.05, 0.05))
                raise TimeoutError
        executor.response_mqs = [Reader()]
        transport = hooks.install(executor, config=ah.HandoffConfig.from_env({'FLASH_S5_OVERLAP': '1'}),
                                  success_status=object())
        self.addCleanup(transport.close, 1)
        row = background(lambda: hooks.collective_rpc(executor, None, 'execute_model', timeout=30))
        time.sleep(1.5)  # more than one default slice: still waiting on a live executor
        self.assertTrue(row.thread.is_alive())
        sent = publish_fatal(writer)
        row.thread.join(4)
        self.assertFalse(row.thread.is_alive(), 'owned write still blocked after the fatal record')
        self.assertIs(row.error, executor._flash_s5_fatal_failure)
        self.assertLess(row.ended - sent, 2 * hooks.ENQUEUE_SLICE_S + 0.5)
        self.assertEqual(ring.writes, [])
        with self.assertRaises(TransportError):
            hooks.collective_rpc(executor, None, 'later', timeout=1)


if __name__ == '__main__':
    unittest.main()
