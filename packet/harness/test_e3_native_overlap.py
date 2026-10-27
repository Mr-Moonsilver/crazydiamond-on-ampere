"""CPU acceptance for native control completion (E3).

Two distinct process/object owners are exercised for real:

* the **engine owner** is a fake ``EngineCore`` that carries only the generated
  ``flash_s5_native_control`` method (extracted from the pinned generated
  ``v1/engine/core.py`` source and executed here) plus the real
  ``NativeSegmentDriver`` the method dispatches to;
* the **API owner** is a separate fake ``AsyncLLM`` whose ``engine_core``
  exposes ``call_utility_async`` on a distinct object that has no driver
  attribute.  It resolves the utility the way the pinned ``core.py`` does: a
  ``concurrent.futures.Future`` result is deferred through
  ``add_done_callback``.

The workers are the reviewed real ``LaneDispatcher``/``NativeWorkerBridge``
fakes from ``test_e3_r4_worker_bridge``; ``native_overlap_worker.handle`` runs
on them for real, over the driver's owned ``DirectTransport`` FIFO.  No GPU,
CUDA, NCCL, MQ, model forward or real PP traffic is exercised and no pass here
is native evidence.  ``native_ready`` stays ``False`` everywhere.
"""
import ast
import asyncio
import concurrent.futures
import textwrap
import time
import types
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine_driver as ed  # noqa: E402
import executor_hooks as eh  # noqa: E402
import native_overlap_control as nc  # noqa: E402
import native_overlap_metrics as nm  # noqa: E402
import native_overlap_session as nsess  # noqa: E402
import native_overlap_worker as nw  # noqa: E402
import s5_startup  # noqa: E402
import scheduler_hooks as sh  # noqa: E402
import wire_v1 as w  # noqa: E402

from test_e3_native_storage_replay import make_native_stack  # noqa: E402

UID = 'req-r4-bridge'
CORE_PATH = ROOT / 'reference/vllm-src/v1/engine/core.py'


# --- the pinned generated utility seam --------------------------------------

_method_source = None


def generated_control_source():
    """The real ``flash_s5_native_control`` method from the generated core."""
    global _method_source
    if _method_source is not None:
        return _method_source
    text = eh.generate_core(CORE_PATH.read_text())
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'flash_s5_native_control':
            segment = ast.get_source_segment(text, node)
            if segment is None:
                break
            _method_source = textwrap.dedent('    ' + segment)
            return _method_source
    raise AssertionError('the generated core has no native-control utility')


def control_function():
    """Compile exactly the generated method body and return the function."""
    namespace = {'_flash_s5_hooks': eh}
    exec(compile(generated_control_source(), '<generated-core>', 'exec'), namespace)
    function = namespace.get('flash_s5_native_control')
    if function is None:
        raise AssertionError('the generated utility did not compile')
    return function


_pause_methods = None


def generated_pause_methods():
    """The real generated ``pause_scheduler``/``_finish_pause`` methods.

    These are the pinned EngineCore methods after the E3 transform, not a
    frontend boolean fake: ``pause_scheduler`` calls ``before_pause`` and
    ``_finish_pause`` calls ``before_finish_pause``.  The first
    ``pause_scheduler`` (the in-process ``EngineCore``) is the reviewed one.
    """
    global _pause_methods
    if _pause_methods is not None:
        return _pause_methods
    text = eh.generate_core(CORE_PATH.read_text())
    tree = ast.parse(text)
    found = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name in ('pause_scheduler', '_finish_pause')
                and node.name not in found):
            segment = ast.get_source_segment(text, node)
            if segment is None:
                raise AssertionError('generated pause method has no source')
            found[node.name] = textwrap.dedent('    ' + segment)
    missing = {'pause_scheduler', '_finish_pause'} - set(found)
    if missing:
        raise AssertionError('generated core lacks pause methods: %s' % missing)
    import typing
    namespace = {
        '_flash_s5_hooks': eh,
        'PauseMode': typing.Literal['abort', 'keep', 'wait'],
        'get_args': typing.get_args,
        'PauseState': types.SimpleNamespace(UNPAUSED='UNPAUSED',
                                            PAUSED_NEW='PAUSED_NEW',
                                            PAUSED_ALL='PAUSED_ALL'),
        'RequestStatus': types.SimpleNamespace(FINISHED_ABORTED='aborted'),
        'Future': concurrent.futures.Future,
    }
    source = '\n\n'.join(found[name]
                         for name in ('pause_scheduler', '_finish_pause'))
    exec(compile(source, '<generated-core-pause>', 'exec'), namespace)
    _pause_methods = (namespace['pause_scheduler'],
                      namespace['_finish_pause'])
    return _pause_methods


_pause_scheduler, _finish_pause = generated_pause_methods()

CoreClass = type('GeneratedEngineCore', (),
                 {'flash_s5_native_control': control_function(),
                  'pause_scheduler': _pause_scheduler,
                  '_finish_pause': _finish_pause})


# --- two owners -------------------------------------------------------------

class TickCore(CoreClass):
    """The engine owner: generated utility + the real driver it dispatches to."""

    def __init__(self, driver, scheduler=None, paused=True):
        self._flash_s5_driver = driver
        self.scheduler = scheduler
        self.paused = paused
        self.ticks = 0
        self.pause_states = []
        self.rpc_calls = []
        self.model_executor = types.SimpleNamespace(
            collective_rpc=lambda name: self.rpc_calls.append(name))

    def is_scheduler_paused(self):
        return self.paused

    def _reset_caches(self):
        raise AssertionError('the native pause path must not clear caches')

    def resume_scheduler(self):
        self.paused = False
        setter = getattr(self.scheduler, 'set_pause_state', None)
        if callable(setter):
            setter('UNPAUSED')

    def step(self):
        """One real engine tick, through the generated idle-tick seam."""
        self.ticks += 1
        tick = eh.engine_idle_tick(self)
        if tick is not None:
            return
        self._flash_s5_driver.advance()


def _settle(waiter, future):
    if waiter.done():
        return
    try:
        waiter.set_result(future.result())
    except BaseException as error:      # the engine's typed refusal
        waiter.set_exception(error)


class UtilityClient:
    """The API owner's single pinned utility channel (a distinct object)."""

    client_count = 1
    client_index = 0

    def __init__(self, core):
        self.core = core
        self.calls = []
        self.deferred = 0

    async def call_utility_async(self, method, *args):
        self.calls.append((method, args))
        result = getattr(self.core, method)(*args)
        if isinstance(result, concurrent.futures.Future):
            # Pinned core.py: a Future result defers the utility output until
            # the engine's own callback completes it.
            self.deferred += 1
            loop = asyncio.get_running_loop()
            waiter = loop.create_future()
            result.add_done_callback(
                lambda future: loop.call_soon_threadsafe(_settle, waiter, future))
            return await waiter
        return result


class Frontend:
    """A fake ``AsyncLLM``: awaited pause/resume plus the utility channel."""

    def __init__(self, core):
        self.core = core
        self.engine_core = UtilityClient(core)
        self.pause_calls = []

    async def is_scheduler_paused(self):
        return self.core.paused

    async def pause_generation(self, mode, clear_cache):
        if (mode, clear_cache) != ('keep', False):
            raise AssertionError('pause must stay awaited keep/clear_cache=False')
        self.pause_calls.append(('pause', mode, clear_cache))
        # The real generated EngineCore path (before_pause/_finish_pause), not a
        # boolean-only frontend fake.
        self.core.pause_scheduler(mode=mode, clear_cache=clear_cache)

    async def resume_generation(self):
        self.pause_calls.append(('resume',))
        self.core.resume_scheduler()


# --- workers ----------------------------------------------------------------

def attach_runner(worker, *, rows=None, clock=time.monotonic_ns):
    """Give a reviewed fake worker the observed runner rows the RPC reads."""
    native = worker.dispatcher.native
    native.state = worker.state
    native.prefill_streams = None
    native.drain_prefill_streams = lambda: 0
    rows = {} if rows is None else rows
    slots = len(rows)
    runner = types.SimpleNamespace(
        device=None,
        pp_handler=worker.controller.runner.pp_handler,
        execute_model_state=None,
        req_states=types.SimpleNamespace(
            req_id_to_index=dict(rows),
            prefill_len=types.SimpleNamespace(np=[8] * slots),
            num_computed_prefill_tokens=[4] * slots),
        _flash_s5_native=native,
        _flash_s5_bridge=worker._flash_s5_bridge,
    )
    nw.install_census(runner)
    worker.model_runner = runner
    worker.flash_s5_native_overlap = lambda command: nw.handle(
        worker, command, clock=clock)
    return runner


def idle_stack(rows=None):
    """Real driver + four real bridge workers with no admitted chunk."""
    stack = make_native_stack()
    parked = stack.controller.parked
    if parked is not None:
        stack.controller.cancel(parked.transaction, never_submitted=True)
    for worker in stack.workers.values():
        attach_runner(worker, rows=rows)
    return stack


def active_stack(rows=None):
    """Real driver + four real bridge workers with the parked chunk intact."""
    stack = make_native_stack()
    for worker in stack.workers.values():
        attach_runner(worker, rows=rows)
    return stack


def cold_stack(rows=None):
    """The same real stack, but the initial park is handed back to the queue.

    This is what a cold full-route scheduler looks like before any native
    admission owns work: one genuinely waiting request and no parked chunk.
    """
    stack = make_native_stack()
    parked = stack.controller.parked
    if parked is not None:
        request = stack.controller.cancel(parked.transaction, never_submitted=True)
        stack.scheduler.waiting.prepend_request(request)
    for worker in stack.workers.values():
        attach_runner(worker, rows=rows)
    return stack


# The reviewed 02c8ed4 rig from the replay suite: a real handback transition
# left TransitionPending on one rank, with every consumer tail synchronized.
def pending_handback(pending_ranks=(1,)):
    # Imported here so this module's suite does not re-collect that module's
    # test cases; it only borrows the fixture construction.
    from test_e3_native_storage_replay import NativeBridgeIntegrationTests
    rig = NativeBridgeIntegrationTests(
        'test_pending_native_replay_blocks_the_return_until_queried')
    return rig.pending_return(pending_ranks)


def core_for(stack, *, paused=True):
    core = TickCore(stack.driver, stack.scheduler, paused=paused)
    scheduler = stack.scheduler

    def set_pause_state(state):
        core.pause_states.append(state)
        core.paused = state != 'UNPAUSED'

    scheduler.set_pause_state = set_pause_state
    scheduler.pause_state = 'PAUSED_ALL' if paused else 'UNPAUSED'
    stack.driver.engine = core
    frontend = Frontend(core)
    return core, frontend


class Pump:
    """Runs the engine's own ticks until stopped."""

    def __init__(self, core):
        self.core = core
        self.task = None
        self.failure = None

    async def _run(self):
        while True:
            try:
                self.core.step()
            except ed.E3Pending:
                pass
            except BaseException as error:      # surfaced by the test
                self.failure = error
                return
            await asyncio.sleep(0)

    def start(self):
        self.task = asyncio.get_running_loop().create_task(self._run())
        return self

    async def stop(self):
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None


async def pump_until(predicate, core, *, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('engine ticks did not reach the expected state')
        try:
            core.step()
        except ed.E3Pending:
            pass
        await asyncio.sleep(0)


# --- generated utility dispatch --------------------------------------------

class GeneratedDispatchTests(unittest.IsolatedAsyncioTestCase):
    def test_the_generated_core_carries_exactly_one_named_utility(self):
        source = generated_control_source()
        self.assertIn('def flash_s5_native_control(self, command):', source)
        self.assertIn('_flash_s5_hooks.native_control(self, command)', source)
        generated = eh.generate_core(CORE_PATH.read_text())
        self.assertEqual(generated.count('def flash_s5_native_control('), 1)
        # The utility never turns a submission into a completion: no callback
        # is installed and no result is awaited inside the seam itself.
        self.assertNotIn('add_done_callback', source)
        self.assertNotIn('await ', source)

    async def test_two_distinct_owners_serve_reads_synchronously(self):
        stack = idle_stack()
        core, frontend = core_for(stack)
        client = frontend.engine_core
        self.assertIsNot(client, core)
        self.assertFalse(hasattr(client, '_flash_s5_driver'))
        self.assertFalse(hasattr(core, 'call_utility_async'))
        self.assertIsInstance(client, UtilityClient)
        control = nc.NativeControlClient(frontend, timeout_s=5.0)
        status = await control.status()
        self.assertEqual(status['route'], 'full')
        self.assertEqual(status['engine_epoch'], stack.driver.engine_epoch)
        self.assertIsNone(status['control_pending'])
        self.assertIsNone(stack.driver._control)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][0], nc.UTILITY)
        self.assertEqual(client.calls[0][1][0]['action'], 'status')
        self.assertIs(control.last_snapshot, status)

    async def test_an_intent_is_deferred_to_engine_ticks_not_pumped_inline(self):
        stack = idle_stack()
        core, frontend = core_for(stack)
        control = nc.NativeControlClient(frontend, timeout_s=5.0)
        task = asyncio.get_running_loop().create_task(control.census())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        # The utility call is registered but nothing was submitted: the client
        # never pumps the engine or the reply transport.
        self.assertIn(nc.UTILITY, [method for method, _ in frontend.engine_core.calls])
        self.assertEqual(frontend.engine_core.deferred, 1)
        self.assertIsNotNone(stack.driver._control)
        self.assertEqual(
            [envelope[0] for envelope in stack.sender.envelopes], [])
        self.assertFalse(task.done())

        await pump_until(lambda: task.done(), core)
        reply = task.result()
        self.assertEqual(reply['action'], 'census')
        self.assertEqual(reply['ranks'], [0, 1, 2, 3])
        self.assertEqual([row['worker_rank'] for row in reply['workers']],
                         [0, 1, 2, 3])
        self.assertEqual([envelope[0] for envelope in stack.sender.envelopes],
                         ['flash_s5_native_overlap'])
        self.assertIsNone(stack.driver._control)

    def test_the_controlled_engine_is_the_only_engine_owner(self):
        # A caller that fabricates a driver on the client is refused.
        stack = idle_stack()
        core, frontend = core_for(stack)
        frontend.engine_core._flash_s5_driver = stack.driver
        with self.assertRaises(nc.NativeControlError):
            nc.NativeControlClient(frontend, timeout_s=5.0)
        del frontend.engine_core._flash_s5_driver


# --- worker rows ------------------------------------------------------------

class WorkerRowTests(unittest.IsolatedAsyncioTestCase):
    def test_pinned_numpy_request_state_scalars_are_real_decode_census(self):
        import numpy as np
        states = types.SimpleNamespace(req_id_to_index={'r0': 0, 'r1': 1},
            prefill_len=types.SimpleNamespace(np=np.array([1024, 1024], dtype=np.int32)),
            num_computed_prefill_tokens=np.array([1024, 512], dtype=np.int32))
        runner = types.SimpleNamespace(req_states=states)
        census = nw.live_census(runner, None)
        self.assertIsNotNone(census)
        self.assertEqual(census['decode_eligible'], 1)
        self.assertEqual(census['prefill_requests'], 1)
        self.assertIs(type(census['decode_eligible']), int)
        for bad in (np.array([True, True]), np.array([1024., 512.]),
                    np.array([-1, 512], dtype=np.int32)):
            states.num_computed_prefill_tokens = bad
            self.assertIsNone(nw.live_census(runner, None))

    def test_missing_bridge_cannot_certify_zero_retained_native_rows(self):
        stack = idle_stack()
        runner = stack.worker(0).model_runner
        del runner._flash_s5_bridge
        with self.assertRaisesRegex(nw.NativeOverlapUnavailable, 'bridge-unavailable'):
            nw._bridge_rows(runner)

    def test_allocator_and_free_total_readings_keep_their_queried_basis(self):
        class Cuda:
            is_available = staticmethod(lambda: True)
            is_initialized = staticmethod(lambda: True)
            mem_get_info = staticmethod(lambda index: (123, 456))
            memory_stats = staticmethod(lambda index: {
                'allocated_bytes.all.peak': 7, 'reserved_bytes.all.peak': 9,
                'num_alloc_retries': 2})
            memory_allocated = staticmethod(lambda index: 3)
            memory_reserved = staticmethod(lambda index: 5)
        torch = types.SimpleNamespace(cuda=Cuda)
        runner = types.SimpleNamespace(device=types.SimpleNamespace(index=0))
        allocator, memory, reason = nw.measure_memory(runner, torch_module=torch)
        self.assertIsNone(reason)
        self.assertEqual(allocator['basis'], 'torch-cuda-allocator-queried-v1')
        self.assertEqual(allocator['device_index'], 0)
        self.assertEqual(allocator['allocated_bytes'], 3)
        self.assertEqual(allocator['reserved_bytes'], 5)
        self.assertEqual(allocator['allocated_peak_bytes'], 7)
        self.assertEqual(allocator['reserved_peak_bytes'], 9)
        self.assertEqual(allocator['num_alloc_retries'], 2)
        # The free/total reading keeps its own queried basis; it is not the
        # 512 MiB reserve proof and no reserve figure is invented here.
        self.assertEqual(memory, {'basis': 'torch-cuda-mem-get-info-v1',
                                  'device_index': 0, 'free_bytes': 123,
                                  'total_bytes': 456})
        unqueried = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False))
        allocator, memory, reason = nw.measure_memory(runner,
                                                      torch_module=unqueried)
        self.assertIsNone(allocator)
        self.assertIsNone(memory)
        self.assertEqual(reason, 'cuda-not-initialized')

    def worker_row(self, stack, rank=0):
        return nw.handle(stack.worker(rank),
                         {'action': 'census', 'engine_epoch': stack.driver.engine_epoch},
                         clock=time.monotonic_ns)

    def test_row_vocabulary_is_exact_and_engine_bound(self):
        stack = idle_stack()
        row = self.worker_row(stack)
        self.assertEqual(set(row), set(nw.ROW_FIELDS))
        self.assertEqual(row['schema'], 1)
        self.assertEqual(row['engine_epoch_ok'], True)
        self.assertEqual(row['clock_basis'], nw.CLOCK_BASIS)
        self.assertEqual(row['live_decode']['basis'], 'worker-native-live-decode-v1')
        self.assertEqual(row['retained_request_uids'], [])
        # A row missing a field, an extra field, a stale epoch or a duplicate
        # rank is refused before it can become engine state.
        stack.driver._storage_epochs = {rank: row['storage_epoch']
                                        for rank in stack.driver.ranks}
        stack.driver._decode_watermarks = {rank: 0 for rank in stack.driver.ranks}
        good = [dict(row, worker_rank=rank,
                     decode_replay_watermark=0,
                     route_generation=stack.driver.route_generation)
                for rank in stack.driver.ranks]
        self.assertEqual(len(stack.driver._validate_worker_rows(good, 'census')), 4)
        missing = [dict(item) for item in good]
        del missing[0]['pid']
        with self.assertRaises(ed.DriverError):
            stack.driver._validate_worker_rows(missing, 'census')
        extra = [dict(item) for item in good]
        extra[0]['invented'] = True
        with self.assertRaises(ed.DriverError):
            stack.driver._validate_worker_rows(extra, 'census')
        stale = [dict(item) for item in good]
        stale[1]['engine_epoch'] = stack.driver.engine_epoch + 1
        with self.assertRaises(ed.DriverError):
            stack.driver._validate_worker_rows(stale, 'census')
        duplicate = [dict(item) for item in good]
        duplicate[3]['worker_rank'] = 0
        with self.assertRaises(ed.DriverError):
            stack.driver._validate_worker_rows(duplicate, 'census')
        absent = good[:3]
        with self.assertRaises(ed.DriverError):
            stack.driver._validate_worker_rows(absent, 'census')

    def test_partial_and_empty_dispatch_batches_are_preserved(self):
        stack = idle_stack()
        worker = stack.worker(0)
        census = worker.model_runner._flash_s5_native_census
        census.observe_scheduler(types.SimpleNamespace(
            num_scheduled_tokens={UID: 4, 'other-live': 1}))
        self.assertEqual(census.last_batch_req_ids, frozenset({UID, 'other-live'}))
        # Empty dispatched batch: a meaningful no-op, never an error.
        census.observe_scheduler(types.SimpleNamespace(num_scheduled_tokens={}))
        self.assertEqual(census.empty_noops, 1)
        self.assertEqual(census.last_batch_req_ids, frozenset())
        fake = types.SimpleNamespace(req_states=types.SimpleNamespace(
            req_id_to_index={UID: 0, 'other-live': 1},
            prefill_len=types.SimpleNamespace(np=[8, 8]),
            num_computed_prefill_tokens=[4, 8]))
        live = nw.live_census(fake, census)
        self.assertEqual(live['live_requests'], 2)
        self.assertEqual(live['prefill_requests'], 1)
        self.assertEqual(live['decode_eligible'], 1)
        self.assertIs(live['empty_noop'], True)
        self.assertEqual(live['live_request_uids'], sorted([UID, 'other-live']))
        # An empty dispatched batch says nothing about the live request rows:
        # the ordinary cohort is reported separately from the batch observation.
        self.assertEqual(live['scheduled_requests'], 0)
        self.assertEqual(live['scheduled_tokens'], 0)

    async def test_ordinary_live_cohort_does_not_block_a_native_drain(self):
        stack = idle_stack(rows={UID: 0})
        core, frontend = core_for(stack)
        # The real scheduler reports a paused LIVE cohort; the fake scheduler in
        # the controller-only suite omitted this method and masked the refusal.
        stack.scheduler.get_num_unfinished_requests = lambda: 8
        worker = stack.worker(0)
        census = worker.model_runner._flash_s5_native_census
        census.observe_scheduler(types.SimpleNamespace(num_scheduled_tokens={}))
        row = self.worker_row(stack)
        self.assertEqual(row['live_decode_requests'], 1)
        self.assertEqual(row['retained_rows'], 0)
        self.assertEqual(row['retained_active_rows'], 0)
        self.assertEqual(row['retained_request_uids'], [])
        self.assertEqual(row['live_batch_rows'], 0)
        # A census row is not a drain proof; barrier/drain query the streams.

        control = nc.NativeControlClient(frontend, timeout_s=5.0)
        barrier = asyncio.get_running_loop().create_task(control.barrier())
        await pump_until(lambda: barrier.done(), core)
        barrier_reply = barrier.result()
        self.assertEqual([item['barrier_clean'] for item in barrier_reply['workers']],
                         [True, True, True, True])
        task = asyncio.get_running_loop().create_task(control.drain())
        await pump_until(lambda: task.done(), core)
        reply = task.result()
        self.assertEqual(reply['action'], 'drain')
        self.assertEqual([item['retained_rows'] for item in reply['workers']],
                         [0, 0, 0, 0])
        # The frozen cohort is reported, never cleared as a drain side effect.
        self.assertEqual([item['live_decode_requests'] for item in reply['workers']],
                         [1, 1, 1, 1])
        self.assertEqual(stack.worker(0).model_runner.req_states.req_id_to_index,
                         {UID: 0})

    async def test_real_nonterminal_release_retains_the_row_until_typed_close(self):
        stack = make_native_stack()
        for worker in stack.workers.values():
            attach_runner(worker, rows={'ordinary-a': 0, 'ordinary-b': 1})
            runner = worker.model_runner
            runner.req_states.prefill_len.np = [8, 8]
            runner.req_states.num_computed_prefill_tokens = [8, 8]
        # First chunk: nonterminal, so the worker deliberately keeps the request
        # row in the real bridge while its transaction is no longer active.
        stack.drive_to_finalize()
        stack.complete_everything()
        stack.drive(lambda: stack.driver.tx is None)
        self.assertIsNone(stack.driver.error)
        for rank in stack.driver.ranks:
            bridge = stack.worker(rank)._flash_s5_bridge
            self.assertFalse(bridge.active)
            self.assertIsNotNone(bridge._retained)
            self.assertEqual(bridge._retained['transaction'],
                             stack.record.transaction)

        core, frontend = core_for(stack)
        stack.scheduler.get_num_unfinished_requests = lambda: 8
        row = nw.handle(stack.worker(0),
                        {'action': 'barrier',
                         'engine_epoch': stack.driver.engine_epoch},
                        clock=time.monotonic_ns)
        self.assertEqual(row['retained_rows'], 1)
        self.assertEqual(row['retained_active_rows'], 0)
        self.assertEqual(row['retained_request_uids'], [UID])
        self.assertEqual(row['live_decode_requests'], 2)
        self.assertIs(row['barrier_clean'], False)

        control = nc.NativeControlClient(frontend, timeout_s=5.0)
        for action in ('barrier', 'drain'):
            task = asyncio.get_running_loop().create_task(getattr(control, action)())
            await pump_until(lambda: task.done(), core)
            with self.assertRaises(ed.DriverError) as caught:
                task.result()
            self.assertIn('native-barrier-not-clean', str(caught.exception))
            self.assertIsNone(stack.driver.error)
            self.assertIsNotNone(stack.worker(0)._flash_s5_bridge._retained)

        # Only an actual typed close (the existing bridge path) releases it.
        for rank in stack.driver.ranks:
            bridge = stack.worker(rank)._flash_s5_bridge
            ack = bridge.cancel(w.CancelRequest(stack.record.transaction,
                                                stack.record.state_epoch, 'abort'))
            self.assertTrue(ack.closed)
            self.assertIsNone(bridge._retained)
        row = nw.handle(stack.worker(0),
                        {'action': 'barrier',
                         'engine_epoch': stack.driver.engine_epoch},
                        clock=time.monotonic_ns)
        self.assertEqual(row['retained_rows'], 0)
        self.assertIs(row['barrier_clean'], True)
        task = asyncio.get_running_loop().create_task(control.drain())
        await pump_until(lambda: task.done(), core)
        reply = task.result()
        self.assertEqual(reply['action'], 'drain')
        self.assertEqual([item['retained_rows'] for item in reply['workers']],
                         [0, 0, 0, 0])

    async def test_pending_native_replays_refuse_a_barrier(self):
        stack = idle_stack()
        core, frontend = core_for(stack)
        worker = stack.worker(1)
        native = worker.dispatcher.native
        native.router = types.SimpleNamespace(
            native_pending=1, replay_watermark=0,
            poll_native_replays=lambda limit=None: ())
        control = nc.NativeControlClient(frontend, timeout_s=5.0)
        task = asyncio.get_running_loop().create_task(control.barrier())
        await pump_until(lambda: task.done(), core)
        with self.assertRaises(ed.DriverError) as caught:
            task.result()
        self.assertIn('native-barrier-not-clean:rank-1', str(caught.exception))

    async def test_qualification_is_delegated_and_unconstructed_is_unavailable(self):
        stack = idle_stack()
        core, frontend = core_for(stack)
        control = nc.NativeControlClient(frontend, timeout_s=5.0)
        # The other worker's provider is not part of this package: the engine
        # must refuse rather than accept a harness-shaped verdict.
        task = asyncio.get_running_loop().create_task(
            control.qualification('teacher_configure', {'teacher': 'w1'}))
        await pump_until(lambda: task.done(), core)
        with self.assertRaises(ed.DriverError) as caught:
            task.result()
        self.assertIn('native-control-failed', str(caught.exception))
        self.assertIsNone(stack.driver.error)
        # Only the four exact actions reach the worker at all.
        with self.assertRaises(nc.NativeControlError):
            await control.qualification('teacher_verdict', {})


# --- live ordinary cohort vs native ownership ------------------------------

class LiveCohortTests(unittest.IsolatedAsyncioTestCase):
    """A paused LIVE ordinary cohort is observed, never a native owner."""

    async def settle(self, core, coroutine):
        task = asyncio.get_running_loop().create_task(coroutine)
        await pump_until(lambda: task.done(), core)
        return task

    async def test_observation_actions_are_admitted_for_a_live_cohort(self):
        stack = idle_stack(rows={UID: 0})
        core, frontend = core_for(stack)
        # The real scheduler reports a paused live cohort; the admission
        # controller's fake omitted this method and masked the old refusal.
        stack.scheduler.get_num_unfinished_requests = lambda: 8
        control = nc.NativeControlClient(frontend, timeout_s=5.0)

        # Configuring a NEW teacher trace needs a genuinely empty worker.
        task = await self.settle(
            core, control.qualification('teacher_configure', {'teacher': 'w1'}))
        with self.assertRaises(ed.DriverError) as caught:
            task.result()
        self.assertIn('native-teacher-configure-requires-paused-empty-engine',
                      str(caught.exception))
        self.assertIsNone(stack.driver.error)

        # Observation boundaries are admitted for the frozen cohort: they reach
        # the worker's own provider (absent in this package) instead of being
        # refused by a live-cohort rule that does not apply to them.
        for action in ('teacher_evidence', 'observer_begin', 'observer_end', 'row1_evidence'):
            task = await self.settle(
                core, control.qualification(action, {'window': 1}))
            with self.assertRaises(ed.DriverError) as caught:
                task.result()
            message = str(caught.exception)
            self.assertIn('native-control-failed', message)
            self.assertNotIn('requires-paused-empty-engine', message)
            self.assertIsNone(stack.driver.error)

        task = await self.settle(core, control.census())
        reply = task.result()
        self.assertEqual([row['live_decode_requests'] for row in reply['workers']],
                         [1, 1, 1, 1])
        self.assertEqual([row['running_requests'] for row in reply['workers']],
                         [1, 1, 1, 1])
        self.assertEqual([row['retained_rows'] for row in reply['workers']],
                         [0, 0, 0, 0])
        self.assertEqual([row['live_batch_rows'] for row in reply['workers']],
                         [0, 0, 0, 0])
        self.assertEqual(stack.worker(0).model_runner.req_states.req_id_to_index,
                         {UID: 0})

    async def test_row1_configuration_requires_the_empty_paused_boundary(self):
        stack = idle_stack(rows={UID: 0})
        core, frontend = core_for(stack)
        stack.scheduler.get_num_unfinished_requests = lambda: 8
        control = nc.NativeControlClient(frontend, timeout_s=5.0)
        task = await self.settle(core, control.qualification(
            'row1_configure', {'enabled': True, 'run_id': 'native-row1'}))
        with self.assertRaisesRegex(ed.DriverError,
                'row1-configure-requires-paused-empty-engine'):
            task.result()
        self.assertFalse(stack.sender.envelopes)

    async def test_owned_native_work_is_never_bypassed_by_a_live_cohort(self):
        stack = active_stack(rows={UID: 0})
        core, frontend = core_for(stack)
        stack.scheduler.get_num_unfinished_requests = lambda: 8
        self.assertTrue(stack.driver.active)          # the parked native chunk
        self.assertTrue(stack.controller.active)
        control = nc.NativeControlClient(frontend, timeout_s=5.0)
        # Admission is decided at submit time: each action is refused before any
        # worker query is issued, so no engine tick is needed to see it.
        for action in ('census', 'barrier', 'drain', 'poll_native'):
            with self.assertRaises(ed.DriverError) as caught:
                await getattr(control, action)()
            self.assertIn('native-control-requires-idle-driver',
                          str(caught.exception))
            self.assertIsNone(control.pending)
            self.assertIsNone(stack.driver.error)
        self.assertTrue(stack.controller.active)
        self.assertIsNotNone(stack.controller.parked)


# --- the real generated pause path -----------------------------------------

class GeneratedPauseTests(unittest.IsolatedAsyncioTestCase):
    """The pinned ``pause_scheduler``/``_finish_pause`` path, not a bool toggle.

    ``Frontend.pause_generation`` executes the generated EngineCore method:
    ``before_pause`` (driver ``stop``), ``set_pause_state``, then
    ``_finish_pause`` (``before_finish_pause`` and
    ``collective_rpc("synchronize_device")``).
    """

    async def pause(self, frontend):
        return await frontend.pause_generation(mode='keep', clear_cache=False)

    async def test_full_to_split_then_a_split_observation_keeps_its_route(self):
        stack = idle_stack()
        core, frontend = core_for(stack, paused=False)
        driver = stack.driver
        self.assertFalse(core.paused)

        await self.pause(frontend)
        self.assertTrue(core.paused)
        self.assertEqual(core.pause_states, ['PAUSED_ALL'])
        self.assertEqual(core.rpc_calls, ['synchronize_device'])
        self.assertEqual(driver.route_generation, 0)
        self.assertIsNone(driver._return_pending)
        self.assertIsNone(driver._control_route)

        control = nc.NativeControlClient(frontend, timeout_s=5.0)
        task = asyncio.get_running_loop().create_task(control.enter_split())
        await pump_until(lambda: task.done(), core)
        self.assertEqual(task.result()['route_after'], 'split')
        self.assertTrue(driver._split_ready)
        self.assertEqual(driver.route_generation, 1)
        self.assertIsNotNone(driver._control_route)
        self.assertTrue(all(row.drained_streams >= 1
                            for row in driver._last_transition_acks.values()))

        session = nsess.NativeOverlapSession(frontend, control=control,
                                             timeout_s=5.0)
        begin = await session.execute({'schema': 1, 'action': 'overlap_begin',
                                       'run_id': 'run-pause'})
        self.assertEqual(begin['route'], 'split')
        self.assertEqual(begin['engine_epoch'], driver.engine_epoch)

        # A pause arriving while the observation holds the split route is not
        # permission to abandon it: the generated path runs for real, the route
        # and generation do not move, and no full return is owed.
        def transitions():
            return sum(1 for envelope in stack.sender.envelopes
                       if envelope[0] == 'flash_s5_transition')
        before = transitions()
        await self.pause(frontend)
        self.assertEqual(transitions(), before)
        self.assertFalse(driver._return_pending)
        self.assertTrue(driver._control_owns_route())
        self.assertIn('control_route_pause_kept',
                      [entry['kind'] for entry in driver._journal])
        self.assertEqual(core.rpc_calls, ['synchronize_device',
                                          'synchronize_device'])
        self.assertTrue(driver._split_ready)
        self.assertEqual(driver.route_generation, 1)
        self.assertEqual(driver.engine_epoch, stack.controller.engine_epoch)
        status = await control.status()
        self.assertEqual(status['route'], 'split')
        self.assertEqual(status['route_generation'], 1)
        self.assertIsNone(status['return_pending'])
        self.assertEqual(status['control_route']['route_generation'], 1)
        for rank in driver.ranks:
            self.assertEqual(stack.worker(rank).controller.route, 'split')
        end = await session.execute({'schema': 1, 'action': 'overlap_end',
                                     'run_id': 'run-pause'})
        self.assertEqual(end['route'], 'split')
        self.assertFalse(end['native_ready'])
        self.assertFalse(end['armable'])

    async def test_reset_retires_the_controlled_split_route(self):
        stack = idle_stack()
        core, frontend = core_for(stack)
        control = nc.NativeControlClient(frontend, timeout_s=5.0)
        task = asyncio.create_task(control.enter_split())
        await pump_until(lambda: task.done(), core)
        task.result()
        self.assertTrue(stack.driver._control_owns_route())
        stack.driver.stop('reset')
        await pump_until(lambda: not stack.driver.active, core)
        self.assertFalse(stack.driver._split_ready)
        self.assertIsNone(stack.driver._control_route)
        self.assertTrue(all(worker.controller.route == 'full'
                            for worker in stack.workers.values()))

    async def test_a_pending_rank_keeps_the_exact_command_until_it_settles(self):
        stack, events = pending_handback()
        core, frontend = core_for(stack)
        driver = stack.driver
        command = driver._transition
        self.assertIsNotNone(command)
        self.assertTrue(driver._split_ready)
        self.assertEqual(driver.route_generation, 1)
        self.assertEqual(driver._return_pending, 'handback')

        with self.assertRaises(ed.E3Pending):
            await self.pause(frontend)
        # The pause is pending, not silently clean: no synchronize ran, and the
        # exact outstanding command is still the driver's transition.
        self.assertEqual(core.pause_states, ['PAUSED_ALL'])
        self.assertEqual(core.rpc_calls, [])
        self.assertIs(driver._transition, command)
        self.assertTrue(driver._split_ready)
        self.assertEqual(driver.route_generation, 1)
        self.assertIsNone(driver.error)
        resends = [args[0] for method, args, _kwargs, _rank in stack.sender.envelopes
                   if method == 'flash_s5_transition'
                   and args[0].target_bank == w.ROUTE_BANK_FULL]
        self.assertTrue(resends)
        self.assertTrue(all(item is command for item in resends))

        # Only the worker's own queried completion settles it.
        events[1].complete = True
        stack.drive(lambda: driver._transition is None)
        stack.drive(lambda: not driver.active)
        self.assertIsNone(driver.error)
        self.assertFalse(driver._split_ready)
        self.assertEqual(driver.route_generation, 2)
        self.assertIsNone(driver._return_pending)
        self.assertIsNone(driver.stop_kind)

        # The retried pause now completes through the generated finish path:
        # device synchronization, then the full-route facts.
        await self.pause(frontend)
        self.assertEqual(core.rpc_calls, ['synchronize_device'])
        self.assertEqual(driver.engine_epoch, stack.controller.engine_epoch)
        self.assertFalse(driver._split_ready)
        self.assertEqual(driver.route_generation, 2)
        self.assertIsNone(driver._control_route)
        self.assertIsNone(driver.admission_intent())
        status = await nc.NativeControlClient(frontend, timeout_s=5.0).status()
        self.assertEqual(status['route'], 'full')
        self.assertEqual(status['route_generation'], 2)


# --- engine-owned admission intent -----------------------------------------

# Exercise the process-mode override and its real idle callback/step methods.
# The earlier TickCore tests execute only EngineCore.pause_scheduler.
def process_core(stack):
    import queue
    from collections import deque
    from enum import Enum
    from functools import partial
    from typing import Any, Literal, get_args
    class PauseState(Enum):
        UNPAUSED = 0
        PAUSED_NEW = 1
        PAUSED_ALL = 2
    namespace = dict(_flash_s5_hooks=eh, Future=concurrent.futures.Future,
        Any=Any, PauseMode=Literal['abort', 'keep', 'wait'], get_args=get_args,
        PauseState=PauseState, partial=partial, queue=queue, time=time,
        DEBUG=10, logger=types.SimpleNamespace(isEnabledFor=lambda level: False,
                                              debug=lambda *args: None))
    generated = ast.parse(eh.generate_core(CORE_PATH.read_text()))
    def method(tree, owner, name):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        node.decorator_list = []
        exec(compile('from __future__ import annotations\n' + ast.unparse(node),
                     '<pinned-process-method>', 'exec'), namespace)
        return namespace[name]
    class Proc(TickCore):
        def step(self):
            self._process_input_queue()
            return self._process_engine_step()
    for name in ('pause_scheduler', '_pause_complete', 'has_work',
                 '_process_input_queue', '_process_engine_step', '_notify_idle_state_callbacks'):
        setattr(Proc, name, method(generated, 'EngineCoreProc', name))
    Proc._finish_pause = method(generated, 'EngineCore', '_finish_pause')
    core = Proc(stack.driver, stack.scheduler)
    core._idle_state_callbacks = []
    core.engines_running = False
    core.batch_queue = deque()
    core.input_queue, core.aborts_queue, core.output_queue = queue.Queue(), queue.Queue(), queue.Queue()
    core.process_input_queue_block = False  # fixture queue is not socket-fed
    core.is_running = lambda: True
    core.post_step = lambda executed: None
    core.step_fn = types.MethodType(method(generated, 'EngineCore', 'step'), core)
    scheduler = stack.scheduler
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.requests = {}
    scheduler.skipped_waiting = []
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.ec_connector = None
    sched = ast.parse((ROOT / 'reference/vllm-src/v1/core/sched/scheduler.py').read_text())
    scheduler.get_num_unfinished_requests = types.MethodType(
        method(sched, 'Scheduler', 'get_num_unfinished_requests'), scheduler)
    scheduler.has_unfinished_requests = lambda: scheduler.get_num_unfinished_requests() > 0
    scheduler.has_finished_requests = lambda: False
    scheduler.has_requests = types.MethodType(method(sched, 'Scheduler', 'has_requests'), scheduler)
    def pause_state(state):
        scheduler._pause_state = state
        core.paused = state != PauseState.UNPAUSED
        core.pause_states.append(state)
    scheduler.set_pause_state = pause_state
    stack.driver.engine = core
    return core


class ProcessPauseTests(unittest.TestCase):
    def test_process_keep_pause_retires_a_not_yet_submitted_native_chunk(self):
        stack = active_stack()
        core = process_core(stack)
        self.assertTrue(stack.controller.active)
        result = core.pause_scheduler(mode='keep', clear_cache=False)
        self.assertIsNone(result)
        self.assertFalse(stack.controller.active)
        self.assertEqual(core.rpc_calls, ['synchronize_device'])
        self.assertTrue(stack.controller.leases.shutdown()['clean'])

    def test_process_pause_future_waits_for_the_exact_mixed_rank_transition(self):
        stack, events = pending_handback()
        core = process_core(stack)
        command = stack.driver._transition
        result = core.pause_scheduler(mode='keep', clear_cache=False)
        self.assertIsInstance(result, concurrent.futures.Future)
        self.assertFalse(result.done())
        self.assertTrue(core.has_work())
        for _ in range(4):
            core.step()
        self.assertFalse(result.done())
        self.assertEqual(core.rpc_calls, [])
        self.assertIs(stack.driver._transition, command)
        events[1].complete = True
        for _ in range(100):
            core.step()
            if result.done():
                break
        self.assertTrue(result.done())
        self.assertIsNone(result.result())
        self.assertFalse(stack.driver._split_ready)
        self.assertEqual(core.rpc_calls, ['synchronize_device'])
        self.assertIsNone(stack.driver.error)

    def test_paused_zero_scheduler_count_does_not_mean_empty_request_registry(self):
        stack = idle_stack()
        core = process_core(stack)
        core.scheduler.requests = {'ordinary-live-request': object()}
        core.scheduler.set_pause_state(types.SimpleNamespace(name='PAUSED_ALL'))
        # Use the exact enum value already bound in the pinned method globals.
        paused = core.scheduler.get_num_unfinished_requests.__func__.__globals__['PauseState'].PAUSED_ALL
        core.scheduler.set_pause_state(paused)
        self.assertEqual(core.scheduler.get_num_unfinished_requests(), 0)
        self.assertFalse(stack.driver._engine_quiescent())

    def test_process_idle_steps_settle_a_registered_utility(self):
        stack = idle_stack()
        core = process_core(stack)
        core.pause_scheduler(mode='keep', clear_cache=False)
        result = stack.driver.submit_control({'action': 'census'})
        for _ in range(100):
            core.step()
            if result.done():
                break
        self.assertTrue(result.done())
        self.assertEqual(len(result.result()['workers']), 4)


class AdmissionPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def split_control(self, stack, core, frontend):
        control = nc.NativeControlClient(frontend, timeout_s=5.0)
        task = asyncio.get_running_loop().create_task(control.enter_split())
        await pump_until(lambda: task.done(), core)
        self.assertEqual(task.result()['route_after'], 'split')
        return control

    def test_the_phase_vocabulary_is_one_closed_table(self):
        self.assertEqual(nc.ADMISSION_PHASES, sh.ADMISSION_PHASES)
        self.assertEqual(nsess.ADMISSION_PHASES, sh.ADMISSION_PHASES)
        self.assertEqual(nc.NATIVE_ADMISSION_PHASES, sh.NATIVE_ADMISSION_PHASES)
        self.assertEqual(nsess.NATIVE_ADMISSION_PHASES, sh.NATIVE_ADMISSION_PHASES)
        self.assertTrue(set(sh.NATIVE_ADMISSION_PHASES)
                        <= set(sh.ADMISSION_PHASES))
        self.assertNotIn('full-prefill-solo', sh.NATIVE_ADMISSION_PHASES)
        self.assertNotIn('split-w1-decode', sh.NATIVE_ADMISSION_PHASES)

    async def test_cold_stock_preparation_waits_for_the_owned_intent(self):
        stack = cold_stack()
        core, frontend = core_for(stack)
        request = stack.scheduler.waiting.peek_request()
        manager = stack.scheduler.kv_cache_manager
        allocs = len(manager.alloc_calls)
        envelopes = len(stack.sender.envelopes)
        settings = (stack.scheduler.max_num_running_reqs,
                    stack.scheduler.max_num_scheduled_tokens,
                    request.num_tokens, request.num_computed_tokens)

        # Policy off: the waiting request is not parked and no native command is
        # issued for it.
        self.assertIsNone(sh.flash_s5_admit(stack.scheduler))
        self.assertEqual(stack.controller.parked_count(), 0)
        self.assertIs(stack.scheduler.waiting.peek_request(), request)
        self.assertEqual(len(manager.alloc_calls), allocs)
        self.assertEqual(len(stack.sender.envelopes), envelopes)

        control = await self.split_control(stack, core, frontend)
        self.assertIsNone(stack.driver.admission_intent())
        # The split route alone does not enable admission.
        self.assertIsNone(sh.flash_s5_admit(stack.scheduler))
        self.assertEqual(stack.controller.parked_count(), 0)

        armed = await control.arm_admission('split-prefill-solo')
        self.assertEqual(armed['admission_intent']['phase'], 'split-prefill-solo')
        self.assertEqual(armed['admission_intent']['owner'], 'engine-core')
        self.assertEqual(armed['route'], 'split')
        self.assertEqual(armed['engine_epoch'], stack.driver.engine_epoch)
        # One intent at a time: the phase cannot change under an owned intent,
        # while repeating the identical phase is the only accepted retry.
        with self.assertRaises(ed.DriverError) as caught:
            await control.arm_admission('w2-overlap')
        self.assertIn('native-admission-intent-change-with-owned-intent',
                      str(caught.exception))
        again = await control.arm_admission('split-prefill-solo')
        self.assertEqual(again['admission_intent']['seq'],
                         armed['admission_intent']['seq'])

        record = sh.flash_s5_admit(stack.scheduler)
        self.assertIsNotNone(record)
        self.assertIs(stack.controller.parked.request, request)
        self.assertEqual(record.transaction.request_uid, UID)
        self.assertEqual(stack.controller.parked_count(), 1)
        self.assertFalse(stack.scheduler.waiting)
        # The phase intent changes no workload or scheduler setting.
        self.assertEqual(
            (stack.scheduler.max_num_running_reqs,
             stack.scheduler.max_num_scheduled_tokens,
             request.num_tokens, request.num_computed_tokens), settings)

        # A new phase or a disable is refused while that chunk is owned.
        with self.assertRaises(ed.DriverError) as caught:
            await control.arm_admission('w2-overlap')
        self.assertIn('native-admission-arm-with-owned-chunk',
                      str(caught.exception))
        with self.assertRaises(ed.DriverError) as caught:
            await control.disarm_admission()
        self.assertIn('native-admission-disable-with-owned-work',
                      str(caught.exception))
        self.assertIsNotNone(stack.driver.admission_intent())

        # The existing reviewed cancel path hands the request back; only then is
        # the intent removable.
        stack.controller.cancel(record.transaction, never_submitted=True)
        reply = await control.disarm_admission()
        self.assertIsNone(reply['admission_intent'])
        self.assertEqual(reply['previous_admission_intent']['phase'],
                         'split-prefill-solo')
        self.assertEqual(reply['route'], 'split')
        self.assertEqual(reply['route_generation'], 1)
        self.assertIsNone(stack.driver.admission_intent())
        self.assertIn('admission_disarmed',
                      [entry['kind'] for entry in stack.driver._journal])

        # A full return closes the native phase even if nobody disarmed first.
        await control.arm_admission('w2-overlap')
        task = asyncio.get_running_loop().create_task(control.leave_split())
        await pump_until(lambda: task.done(), core)
        left = task.result()
        self.assertEqual(left['route_after'], 'full')
        self.assertIsNone(left['admission_intent'])
        self.assertIsNone(stack.driver.admission_intent())
        self.assertIsNone(stack.driver._control_route)
        self.assertEqual(stack.driver.route_generation, 2)
        closed = [entry for entry in stack.driver._journal
                  if entry['kind'] == 'admission_disarmed'
                  and entry.get('reason') == 'route-returned-full']
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]['phase'], 'w2-overlap')
        with self.assertRaises(ed.DriverError) as caught:
            await control.arm_admission('row1-native')
        self.assertIn('native-admission-requires-split-route',
                      str(caught.exception))
        self.assertIsNone(sh.flash_s5_admit(stack.scheduler))
        self.assertEqual(stack.controller.parked_count(), 0)

    async def test_the_engine_refuses_foreign_epochs_and_unknown_phases(self):
        stack = cold_stack()
        core, frontend = core_for(stack)
        await self.split_control(stack, core, frontend)
        driver = stack.driver
        epoch = driver.engine_epoch
        with self.assertRaises(ed.DriverError) as caught:
            driver.submit_control({'action': 'arm_admission',
                                   'phase': 'split-w1-decode'})
        self.assertIn('native-admission-phase-not-native', str(caught.exception))
        with self.assertRaises(ed.DriverError) as caught:
            driver.submit_control({'action': 'arm_admission', 'phase': 'invented'})
        self.assertIn('native-admission-phase-unknown', str(caught.exception))
        with self.assertRaises(ed.DriverError) as caught:
            driver.submit_control({'action': 'arm_admission',
                                   'phase': 'row1-native',
                                   'engine_epoch': epoch + 1})
        self.assertIn('native-control-engine-epoch-differs',
                      str(caught.exception))
        self.assertIsNone(driver.admission_intent())
        self.assertEqual(stack.controller.parked_count(), 0)
        # Arming is a pause-time decision, never a mid-step policy flip.
        core.paused = False
        try:
            with self.assertRaises(ed.DriverError) as caught:
                driver.submit_control({'action': 'arm_admission',
                                       'phase': 'row1-native'})
            self.assertIn('native-admission-requires-paused-engine',
                          str(caught.exception))
        finally:
            core.paused = True
        self.assertIsNone(driver.admission_intent())
        # The same intent is accepted under this engine's own epoch.
        reply = driver.submit_control({'action': 'arm_admission',
                                       'phase': 'row1-native',
                                       'engine_epoch': epoch})
        self.assertEqual(reply['admission_intent']['phase'], 'row1-native')
        driver.submit_control({'action': 'disarm_admission'})
        self.assertIsNone(driver.admission_intent())

    async def test_the_policy_seam_refuses_a_foreign_or_stale_intent(self):
        stack = cold_stack()
        core, frontend = core_for(stack)
        control = await self.split_control(stack, core, frontend)
        await control.arm_admission('split-prefill-solo')
        driver = stack.driver
        record = dict(driver.admission_intent())

        class IntentOwner:
            """A foreign owner projecting one fabricated intent at the seam."""

            def __init__(self, intent):
                self.intent = intent

            def admission_intent(self):
                return self.intent

        cases = (
            ('not-a-record', 'not a typed record'),
            (dict(record, armed=False), 'not armed'),
            (dict(record, owner='worker'), 'no engine owner'),
            (dict(record, phase='split-w1-decode'), 'names no native phase'),
            (dict(record, phase='invented'), 'names no native phase'),
            (dict(record, engine_epoch=record['engine_epoch'] + 1),
             'another engine epoch'),
            (dict(record, route='full'), 'needs the split route'),
        )
        real_owner = stack.scheduler._flash_s5_driver
        try:
            for intent, message in cases:
                stack.scheduler._flash_s5_driver = IntentOwner(intent)
                with self.assertRaises(sh.E3ContractError) as caught:
                    sh.flash_s5_admit(stack.scheduler)
                self.assertIn(message, str(caught.exception))
                self.assertEqual(stack.controller.parked_count(), 0)
                self.assertTrue(stack.scheduler.waiting)
        finally:
            stack.scheduler._flash_s5_driver = real_owner
        # The real engine-owned intent still admits the same waiting request.
        admitted = sh.flash_s5_admit(stack.scheduler)
        self.assertIsNotNone(admitted)
        self.assertEqual(admitted.transaction.request_uid, UID)

    async def test_pending_route_refuses_admission_arm_before_mutation(self):
        stack = cold_stack()
        core, frontend = core_for(stack)
        await self.split_control(stack, core, frontend)
        driver = stack.driver
        # Typed TransitionPending leaves the exact command owed even with no
        # broadcast currently in flight. It is not an idle control boundary.
        driver._transition = object()
        with self.assertRaises(ed.DriverError):
            driver.submit_control({'action': 'arm_admission',
                                   'phase': 'row1-native'})
        self.assertIsNone(driver.admission_intent())
        self.assertEqual(stack.controller.parked_count(), 0)

    async def test_stale_route_generation_cannot_park_a_waiting_request(self):
        stack = cold_stack()
        core, frontend = core_for(stack)
        control = await self.split_control(stack, core, frontend)
        await control.arm_admission('row1-native')
        stack.driver._admission_intent['route_generation'] -= 1
        with self.assertRaises(sh.E3ContractError):
            sh.flash_s5_admit(stack.scheduler)
        self.assertEqual(stack.controller.parked_count(), 0)

    def test_unbound_policy_seam_keeps_stock_admission_by_default(self):
        stack = cold_stack()
        scheduler = stack.scheduler
        del scheduler._flash_s5_driver
        self.assertIsNone(sh.flash_s5_admit(scheduler))
        self.assertEqual(stack.controller.parked_count(), 0)
        self.assertTrue(scheduler.waiting)

    async def test_session_preparation_selects_the_phase_and_never_arms_on_full(self):
        stack = cold_stack()
        core, frontend = core_for(stack)
        pump = Pump(core).start()
        try:
            await self._phase_selection(stack, frontend)
        finally:
            await pump.stop()
        if pump.failure is not None:
            raise pump.failure

    async def _phase_selection(self, stack, frontend):
        session = nsess.NativeOverlapSession(frontend, timeout_s=5.0)
        for phase in nsess.NATIVE_ADMISSION_PHASES:
            result = await session.execute(
                {'schema': 1, 'action': 'prepare', 'run_id': 'run-phase',
                 'target_route': 'split', 'phase': phase})
            self.assertEqual(result['route'], 'split')
            self.assertTrue(result['admission']['armed'])
            self.assertEqual(result['admission']['admission_intent']['phase'],
                             phase)
            self.assertEqual(stack.driver.admission_intent()['phase'], phase)
            left = await session.execute({'schema': 1, 'action': 'leave_full',
                                          'run_id': 'run-phase'})
            self.assertEqual(left['route'], 'full')
            self.assertFalse(left['admission']['armed'])
            self.assertIsNone(stack.driver.admission_intent())
            self.assertIsNone(stack.driver._control_route)

        # W1 decode-only preparation is a split route with admission OFF.
        result = await session.execute(
            {'schema': 1, 'action': 'prepare', 'run_id': 'run-phase',
             'target_route': 'split', 'phase': 'split-w1-decode'})
        self.assertEqual(result['route'], 'split')
        self.assertFalse(result['admission']['armed'])
        self.assertIsNone(result['admission']['admission_intent'])
        self.assertIsNone(stack.driver.admission_intent())
        self.assertIsNone(sh.flash_s5_admit(stack.scheduler))
        self.assertEqual(stack.controller.parked_count(), 0)

        armed = await session.execute({'schema': 1, 'action': 'arm_admission',
                                       'run_id': 'run-phase',
                                       'phase': 'row1-native'})
        self.assertTrue(armed['armed'])
        self.assertEqual(stack.driver.admission_intent()['phase'], 'row1-native')
        disarmed = await session.execute({'schema': 1, 'action': 'admission',
                                          'run_id': 'run-phase',
                                          'phase': 'split-w1-decode'})
        self.assertFalse(disarmed['armed'])
        self.assertIsNone(stack.driver.admission_intent())
        with self.assertRaises(nsess.NativeOverlapSessionError):
            await session.execute({'schema': 1, 'action': 'arm_admission',
                                   'run_id': 'run-phase',
                                   'phase': 'full-prefill-solo'})
        # A full-route preparation never arms a native phase: it disarms.
        with self.assertRaises(nsess.NativeOverlapSessionError):
            await session.execute({'schema': 1, 'action': 'prepare',
                                   'run_id': 'run-phase',
                                   'target_route': 'full', 'phase': 'w2-overlap'})
        self.assertIsNone(stack.driver.admission_intent())
        # The explicit stock full-route phase is accepted and keeps it off.
        full = await session.execute({'schema': 1, 'action': 'prepare',
                                      'run_id': 'run-phase',
                                      'target_route': 'full',
                                      'phase': 'full-prefill-solo'})
        self.assertEqual(full['route'], 'full')
        self.assertFalse(full['admission']['armed'])
        self.assertEqual(full['admission']['phase'], 'full-prefill-solo')
        self.assertIsNone(stack.driver.admission_intent())
        self.assertIsNone(stack.driver._control_route)


# --- session/failure behaviour ---------------------------------------------

class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stack = idle_stack()
        self.core, self.frontend = core_for(self.stack)
        self.pump = Pump(self.core).start()

    async def asyncTearDown(self):
        await self.pump.stop()
        if self.pump.failure is not None:
            raise self.pump.failure

    async def test_prepare_binds_every_rank_with_typed_acks(self):
        self.core.paused = False           # the API path must actually pause
        control = nc.NativeControlClient(self.frontend, timeout_s=5.0)
        session = nsess.NativeOverlapSession(self.frontend, control=control,
                                             timeout_s=5.0)
        result = await session.execute({'schema': 1, 'action': 'prepare',
                                        'run_id': 'run-a', 'target_route': 'split'})
        self.assertEqual(result['route'], 'split')
        self.assertTrue(result['typed_acks'])
        self.assertEqual(len(result['bootstrap_acks']), 4)
        self.assertEqual(len(result['transition_acks']), 4)
        for row in result['transition_acks']:
            self.assertEqual(row['typed'], 'TransitionAck')
            self.assertEqual(row['applied_bank'], w.ROUTE_BANK_SPLIT)
            self.assertGreaterEqual(row['drained_streams'], 1)
        for rank in self.stack.driver.ranks:
            self.assertEqual(self.stack.worker(rank).controller.route, 'split')
        self.assertFalse(result['native_ready'])
        self.assertFalse(result['armable'])
        self.assertIn('require_native_ready', result['native_ready_reason'])
        # Every pause/resume went through the awaited API path.
        self.assertEqual(result['paused']['mode'], 'keep')
        self.assertIs(result['paused']['clear_cache'], False)
        self.assertTrue(result['paused']['called'])
        self.assertIn(('pause', 'keep', False), self.frontend.pause_calls)
        self.assertIn(('resume',), self.frontend.pause_calls)

        left = await session.execute({'schema': 1, 'action': 'leave_full',
                                      'run_id': 'run-a'})
        self.assertEqual(left['route'], 'full')
        self.assertEqual(left['worker_drain'][0]['drained_streams'] >= 1, True)
        self.assertEqual(self.stack.driver.route_generation, 2)
        self.assertFalse(left['native_ready'])

    async def test_pause_while_a_transition_is_pending_does_not_deadlock(self):
        transport = self.stack.transport
        transport.hold = True
        control = nc.NativeControlClient(self.frontend, timeout_s=5.0)
        task = asyncio.get_running_loop().create_task(control.enter_split())
        await pump_until(lambda: self.stack.driver._bootstrap_command is not None
                         or self.stack.driver._transition is not None, self.core)
        # A pause utility arrives while the engine still owns the route intent.
        self.assertTrue(await self.frontend.is_scheduler_paused())
        self.assertTrue(self.stack.driver.finish_pause())
        self.assertIsNotNone(self.stack.driver._control)
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        transport.deliver_held()
        await pump_until(lambda: task.done(), self.core)
        reply = task.result()
        self.assertEqual(reply['route_after'], 'split')
        self.assertIsNone(self.stack.driver.error)
        self.assertIsNone(self.stack.driver._control)

    async def test_client_timeout_retains_engine_ownership_until_observed(self):
        await self.pump.stop()
        control = nc.NativeControlClient(self.frontend, timeout_s=0.05)
        # The reply is computed but held on the owned transport, so the utility
        # deadline expires while the engine still owns the intent.
        self.stack.transport.hold = True
        with self.assertRaises(nc.NativeControlError) as caught:
            await control.census()
        self.assertIn('retains ownership', str(caught.exception))
        self.assertEqual(control.pending, 'census')
        self.assertIsNotNone(self.stack.driver._control)
        cancel = await control.cancel()
        self.assertTrue(cancel['ownership_retained'])
        self.assertEqual(cancel['intent'], 'census')
        self.assertIsNotNone(self.stack.driver._control)
        # Refused until settlement is observed, never silently re-issued.
        with self.assertRaises(nc.NativeControlError):
            await control.census()
        self.stack.transport.deliver_held()
        await pump_until(lambda: self.stack.driver._control is None, self.core)
        status = await control.status()
        self.assertIsNone(status['control_pending'])
        self.assertIsNone(control.pending)
        task = asyncio.get_running_loop().create_task(control.census())
        await pump_until(lambda: task.done(), self.core)
        reply = task.result()
        self.assertEqual(reply['action'], 'census')
        self.pump = Pump(self.core).start()

    async def test_deadline_expiry_fails_the_intent_without_inventing_a_result(self):
        control = nc.NativeControlClient(self.frontend, timeout_s=5.0)
        task = asyncio.get_running_loop().create_task(control.barrier())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertIsNotNone(self.stack.driver._control)
        self.stack.driver._control_deadline = 0.0
        await pump_until(lambda: task.done(), self.core)
        with self.assertRaises(ed.DriverError) as caught:
            task.result()
        self.assertIn('native-control-failed', str(caught.exception))
        self.assertIsNone(self.stack.driver._control)
        self.assertIsNone(self.stack.driver.error)

    async def test_transport_backpressure_leaves_the_intent_registered(self):
        await self.pump.stop()
        control = nc.NativeControlClient(self.frontend, timeout_s=5.0)
        state = {'calls': 0, 'blocked': True}

        def check(envelope):
            state['calls'] += 1
            if state['blocked']:
                raise ed.Backpressure('synthetic ring full')

        self.stack.sender.check = check
        task = asyncio.get_running_loop().create_task(control.census())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        for _ in range(3):
            self.core.step()
        # Backpressure registers no expectation: nothing was sent, the intent
        # keeps its ownership and a later tick retries the same command.
        self.assertIsNotNone(self.stack.driver._control)
        self.assertEqual(self.stack.sender.envelopes, [])
        self.assertFalse(task.done())
        state['blocked'] = False
        await pump_until(lambda: task.done(), self.core)
        self.assertEqual(task.result()['action'], 'census')
        self.assertEqual(state['calls'], 4)
        self.pump = Pump(self.core).start()


# --- evidence ---------------------------------------------------------------

class EvidenceTests(unittest.IsolatedAsyncioTestCase):
    def finish_chunk(self, stack):
        stack.drive_to_finalize()
        stack.complete_everything()
        stack.drive(lambda: stack.driver.tx is None)
        self.assertIsNone(stack.driver.error)

    def test_fresh_and_append_journal_survive_release_and_build_evidence(self):
        stack = active_stack()
        core, frontend = core_for(stack)
        stack.drive(lambda: stack.driver._split_ready)
        self.assertEqual(stack.worker(0).controller.route, 'split')

        control = nc.NativeControlClient(frontend, timeout_s=5.0)
        session = nsess.NativeOverlapSession(frontend, control=control,
                                             timeout_s=5.0)
        begin = asyncio.run(session.execute({'schema': 1, 'action': 'overlap_begin',
                                             'run_id': 'run-b'}))
        self.assertEqual(begin['route'], 'split')

        self.finish_chunk(stack)                     # fresh chunk on the split
        self.assertIsNotNone(stack.controller.admit(stack.scheduler))
        self.finish_chunk(stack)                     # append chunk on the split

        result = asyncio.run(session.execute(
            {'schema': 1, 'action': 'overlap_end', 'run_id': 'run-b',
             'decode_progress': 4}))
        segments = [segment for segment in result['segments']
                    if segment.get('accepted')]
        self.assertEqual([segment['prefill'] for segment in segments],
                         ['fresh', 'append'])
        self.assertEqual([segment['transport'] for segment in segments],
                         ['native-split', 'native-split'])
        self.assertTrue(all(segment['state_write'] for segment in segments))
        self.assertEqual([segment['state_write_ranks'] for segment in segments],
                         [[0, 1, 2, 3], [0, 1, 2, 3]])
        # Every accepted segment carries the queried per-op reports of all four
        # ranks, not a submission count.
        self.assertTrue(all(segment['report_count'] >= 4 for segment in segments))
        self.assertGreaterEqual(len(result['completions']), 8)
        self.assertEqual({row['worker_rank'] for row in result['completions']},
                         {0, 1, 2, 3})
        self.assertTrue(all(row['queried'] for row in result['completions']))
        # The queried STATE_WRITE settlement lands as a real completion proof in
        # the window; it is not inferred from a submission counter.
        self.assertTrue(any(row['state_write_ns'] for row in result['completions']))
        self.assertTrue(result['source_holds'])
        self.assertTrue(result['committed_frontiers'])
        self.assertEqual(result['journal']['dropped'], 0)
        self.assertTrue(result['journal']['complete_in_window'])
        self.assertTrue(result['enabled'])
        self.assertEqual(result['decode_progress'], 4)
        self.assertGreater(result['fresh_prefill']['useful_tokens'], 0)
        self.assertGreater(result['append_prefill']['useful_tokens'], 0)
        self.assertEqual(result['clock']['begin']['basis'], nm.CLOCK_BASIS)
        self.assertTrue(result['clock']['end']['proved'])
        self.assertFalse(result['native_ready'])
        self.assertFalse(result['armable'])

    def test_evidence_refuses_clock_mismatch_and_missing_telemetry(self):
        stack = active_stack()
        core, frontend = core_for(stack)
        start = time.monotonic_ns() - 1
        self.finish_chunk(stack)
        driver = stack.driver
        now = time.monotonic_ns()
        with self.assertRaises(ed.DriverError):
            driver.overlap_evidence(0, now)
        with self.assertRaises(ed.DriverError):
            driver.overlap_evidence(now, now)
        with self.assertRaises(ed.DriverError):
            driver.overlap_evidence(now - 10, now + 10 ** 9)
        with self.assertRaises(ed.DriverError):
            driver.overlap_evidence(now - ed.MAX_EVIDENCE_WINDOW_NS - 1, now)
        with self.assertRaises(ed.DriverError):
            driver.overlap_evidence(now + 1, now + 2)

        end = time.monotonic_ns() + 1
        record = driver.overlap_evidence(start, min(end, time.monotonic_ns()))
        self.assertTrue(record['journal']['complete_in_window'])

        # A stale epoch already in the window is refused, not relabelled.
        stale = dict(driver._journal[-1])
        stale['engine_epoch'] += 1
        driver._journal.append(stale)
        with self.assertRaises(ed.DriverError):
            driver.overlap_evidence(start, time.monotonic_ns())
        driver._journal.pop()

        # A different clock basis is refused too.
        stale = dict(driver._journal[-1])
        stale['clock_basis'] = 'cuda-elapsed'
        driver._journal.append(stale)
        with self.assertRaises(ed.DriverError):
            driver.overlap_evidence(start, time.monotonic_ns())
        driver._journal.pop()

    def test_serialized_transport_is_descriptive_never_accepted(self):
        metrics = nm.measure_windows(
            begin_receipt={'observed_clock_ns': 10},
            end_receipt={'observed_clock_ns': 20},
            begin_proof={'proved': True, 'uncertainty_ns': 1},
            end_proof={'proved': True, 'uncertainty_ns': 1},
            segments=[{'transaction': 1, 'request': UID, 'prefill': 'fresh',
                       'transport': 'uva', 'accepted': True, 'state_write': True,
                       'start_monotonic_ns': 11, 'end_monotonic_ns': 19,
                       'chunk_tokens': 4}],
            completions=[{'request': UID, 'worker_rank': 0, 'queried': True,
                          'proofs': [{'kind': 'state_write', 'completed_ns': 19}]}],
            decode_progress=2)
        self.assertFalse(metrics['valid'])
        self.assertEqual(metrics['segments'][0]['accepted'], False)
        self.assertIn('transport-not-accepted-overlap:uva',
                      metrics['segments'][0]['reasons'])
        self.assertTrue(metrics['serialized_descriptive'])


# --- metrics ----------------------------------------------------------------

class ClockDomainTests(unittest.TestCase):
    def test_unproved_or_device_clocks_are_refused(self):
        identity = nm.clock_identity()
        self.assertTrue(identity['proved'])
        engine = {'basis': 'clock-monotonic', 'sample_ns': 50,
                  'boot_id': identity['boot_id'],
                  'time_namespace': identity['time_namespace']}
        self.assertTrue(nm.prove_clock_domain(engine, identity, 10, 90)['proved'])
        cuda = dict(engine, basis='cuda-elapsed')
        proof = nm.prove_clock_domain(cuda, identity, 10, 90)
        self.assertFalse(proof['proved'])
        self.assertIn('device-elapsed-not-comparable-to-client-monotonic',
                      proof['reasons'])
        other = dict(engine, boot_id='00000000-0000-0000-0000-000000000000')
        proof = nm.prove_clock_domain(other, identity, 10, 90)
        self.assertFalse(proof['proved'])
        self.assertIn('boot-id-differs', proof['reasons'])
        outside = dict(engine, sample_ns=200)
        proof = nm.prove_clock_domain(outside, identity, 10, 90)
        self.assertFalse(proof['proved'])
        self.assertIn('engine-sample-outside-request-bracket', proof['reasons'])
        self.assertIsNone(nm.decode_share(decode_ns=5, window_ns=10,
                                          decode_progress=None))


# --- default-off and pinned inverses ---------------------------------------

class DefaultOffTests(unittest.TestCase):
    def test_default_off_behaviour_holds(self):
        self.assertFalse(s5_startup._enabled({}))
        self.assertFalse(s5_startup._enabled({'FLASH_S5_OVERLAP': '0'}))
        with self.assertRaises(s5_startup.StartupError):
            s5_startup._enabled({'FLASH_S5_OVERLAP': 'yes'})
        with self.assertRaises(s5_startup.StartupError):
            s5_startup.require_native_ready()
        engine = types.SimpleNamespace()
        self.assertIsNone(s5_startup.install_engine(engine, env={}))
        self.assertFalse(hasattr(engine, '_flash_s5_driver'))
        from worker_hooks import _enabled as worker_enabled
        self.assertFalse(worker_enabled({}))

    def test_generated_sources_keep_their_exact_inverses(self):
        text = CORE_PATH.read_text()
        generated = eh.generate_core(text)
        self.assertEqual(eh.revert_core(generated, expected=text), text)
        worker_path = ROOT / 'reference/vllm-src/v1/worker/gpu_worker.py'
        if worker_path.exists():
            source = worker_path.read_text()
            generated = eh.generate_worker(source)
            self.assertEqual(generated, eh.generate_worker(source))
            self.assertEqual(eh.revert_worker(generated, expected=source), source)


if __name__ == '__main__':
    unittest.main()
