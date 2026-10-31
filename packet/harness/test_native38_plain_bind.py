"""CPU-only owner/PP failure proof. Leaves: native38_fixture_leaves.json."""
import ast
import asyncio
from enum import StrEnum
from functools import partial
import multiprocessing
import logging
import os
from pathlib import Path
import sys
import threading
import traceback
import time
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'patches/s5_engine'), str(ROOT/'harness')]
import executor_hooks as hooks
from reply_transport import ReplyTransport, WorkerRPCError
from s5_startup import bootstrap_full_startup, StartupError
import test_e3_native_overlap as native
from test_s5_reply_transport import Reader, Status


def setUpModule():
    global _logging_leaf_previous
    _logging_leaf_previous = sys.modules.get('vllm.logger')
    sys.modules['vllm.logger'] = SimpleNamespace(init_logger=logging.getLogger)


def tearDownModule():
    if _logging_leaf_previous is None:
        sys.modules.pop('vllm.logger', None)
    else:
        sys.modules['vllm.logger'] = _logging_leaf_previous


class Pause(StrEnum):
    UNPAUSED = 'UNPAUSED'
    PAUSED_ALL = 'PAUSED_ALL'


def generated_rpc():
    source = (ROOT/'reference/vllm-src/v1/executor/multiproc_executor.py').read_text()
    generated = hooks.generate_executor(source)
    if os.environ.get('NATIVE38_RPC_SUBJECT') == 'parent':
        generated = source
    if os.environ.get('NATIVE38_RPC_SUBJECT') != 'parent':
        assert hooks.revert_executor(generated) == source
    tree = ast.parse(generated)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WorkerProc')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_execute_worker_rpc')
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = dict(Any=Any, partial=partial, traceback=traceback, _flash_s5_hooks=hooks,
                     logger=SimpleNamespace(exception=lambda *a: None))
    exec(compile(module, 'actual-generated-worker-rpc', 'exec'), namespace)
    return namespace['_execute_worker_rpc']


def fault_worker(writer, kind):
    hooks.set_fatal_writer(writer)
    def fail():
        if kind == 'E3Pending':
            from e3_types import E3Pending
            raise E3Pending('fixture')
        raise ValueError('fixture')
    worker = SimpleNamespace(rank=0, worker=SimpleNamespace(execute_model=fail))
    generated_rpc()(worker, ('execute_model', (), {}, 1))
    # Model the old busy loop continuing after silently dropping PP0's error.
    if os.environ.get('NATIVE38_RPC_SUBJECT') == 'parent':
        threading.Event().wait()


class OwnerStartup(unittest.TestCase):
    def stack(self):
        stack = native.cold_stack()
        stack.scheduler.waiting.items.clear()
        core, _ = native.core_for(stack, paused=False)
        stack.scheduler.pause_state = Pause.UNPAUSED
        return stack, core

    def test_connected_startup_binds_every_real_bridge_before_plain_route(self):
        stack, core = self.stack()
        from e3_types import E3Pending
        from graph_hooks import check_route
        runner=SimpleNamespace(_flash_e3_graph_router=stack.workers[0].router)
        descriptor=SimpleNamespace(cg_mode='FULL',num_tokens=1,num_reqs=1,
                                   uniform_token_count=1,max_query_len=1)
        with self.assertRaises(E3Pending):
            check_route(runner,SimpleNamespace(),descriptor,1,1,1)
        ready = bootstrap_full_startup(core, stack.driver, timeout_s=2)
        self.assertIsNotNone(check_route(runner,SimpleNamespace(),descriptor,1,1,1))
        self.assertEqual(ready['engine_epoch'], stack.driver.engine_epoch)
        self.assertEqual(len(ready['bootstrap_acks']), 4)
        self.assertEqual(ready['full_transition']['route_after'], 'full')
        self.assertTrue(stack.driver._bootstrapped)
        self.assertFalse(stack.driver._split_ready)
        self.assertIsNone(stack.driver.admission_intent())
        self.assertEqual(stack.scheduler.pause_state, Pause.UNPAUSED)
        self.assertFalse(core.paused)
        self.assertTrue(all(w._flash_s5_bridge.engine_epoch == stack.driver.engine_epoch for w in stack.workers.values()))

    def test_startup_bound_receipt_once_after_proof_is_latched(self):
        stack, core = self.stack()
        logger = Mock()
        def init_logger(name):
            self.assertIsNotNone(core._flash_s5_startup_ready)
            self.assertEqual(stack.scheduler.pause_state, Pause.UNPAUSED)
            return logger
        with patch.dict(sys.modules, {'vllm.logger': SimpleNamespace(init_logger=init_logger)}):
            ready = bootstrap_full_startup(core, stack.driver, timeout_s=2)
            with self.assertRaises(StartupError):
                bootstrap_full_startup(core, stack.driver, timeout_s=2)
        logger.info.assert_called_once_with(
            'flash-s5 startup owner bound engine_epoch=%s ranks=%s route_after=%s route_generation_after=%s',
            stack.driver.engine_epoch, list(stack.driver.ranks), 'full',
            ready['full_transition']['route_generation_after'])

    def test_startup_bound_receipt_absent_on_failure_paths(self):
        cases = ('published', 'nonempty', 'queued', 'pause', 'deadline',
                 'future', 'barrier', 'proof', 'acks', 'resume')
        for case in cases:
            with self.subTest(case=case):
                stack, core = self.stack()
                core.model_executor = SimpleNamespace(shutdown=lambda: None)
                init_logger = Mock()
                if case == 'published': core._flash_s5_startup_ready = {'fixture': True}
                if case == 'nonempty': stack.scheduler.requests = {'fixture': object()}
                if case == 'queued': core.batch_queue = [object()]
                if case == 'acks': stack.driver.bootstrap_ack_rows = Mock(side_effect=ValueError('fixture'))
                pause = stack.scheduler.set_pause_state
                if case in ('pause', 'resume'):
                    def set_pause(state):
                        if case == 'pause' or state == Pause.UNPAUSED: raise ValueError('fixture')
                        return pause(state)
                    stack.scheduler.set_pause_state = set_pause
                submit = stack.driver.submit_control
                def submit_control(command):
                    future = submit(command)
                    def result():
                        row = future.result()
                        if case == 'future': raise ValueError('fixture')
                        if case == 'barrier' and command['action'] == 'barrier':
                            row = dict(row, workers=[])
                        if case == 'proof' and command['action'] == 'leave_split':
                            row = dict(row, typed_acks=False)
                        return row
                    return SimpleNamespace(done=future.done, result=result)
                stack.driver.submit_control = submit_control
                with patch.dict(sys.modules, {'vllm.logger': SimpleNamespace(init_logger=init_logger)}):
                    with self.assertRaises((StartupError, ValueError)):
                        bootstrap_full_startup(core, stack.driver, timeout_s=0 if case == 'deadline' else 2)
                init_logger.assert_not_called()

    def test_second_owner_startup_refuses(self):
        stack, core = self.stack()
        bootstrap_full_startup(core, stack.driver, timeout_s=2)
        with self.assertRaisesRegex(StartupError, 'already published'):
            bootstrap_full_startup(core, stack.driver, timeout_s=2)

    def test_failed_startup_never_publishes_readiness(self):
        stack, core = self.stack()
        shutdown = []
        core.model_executor = SimpleNamespace(shutdown=lambda: shutdown.append(True))
        # Expired owner deadline: no bootstrap ack can publish ready.
        with self.assertRaisesRegex(StartupError, 'timed out'):
            bootstrap_full_startup(core, stack.driver, timeout_s=0)
        self.assertFalse(hasattr(core, '_flash_s5_startup_ready'))
        self.assertEqual(shutdown, [True])
        self.assertTrue(core.paused)

    def test_restart_uses_fresh_worker_set_without_native_client(self):
        first, a = self.stack(); second, b = self.stack()
        second.driver.engine_epoch=11
        second.controller.engine_epoch=11
        ra = bootstrap_full_startup(a, first.driver, timeout_s=2)
        rb = bootstrap_full_startup(b, second.driver, timeout_s=2)
        self.assertNotEqual(ra['engine_epoch'],rb['engine_epoch'])
        self.assertEqual(len(ra['bootstrap_acks']), len(rb['bootstrap_acks']))
        self.assertIsNot(first.workers[0], second.workers[0])
        self.assertTrue(first.driver._bootstrapped and second.driver._bootstrapped)


class FatalReplies(unittest.TestCase):
    def test_non_output_failure_poisons_real_pending_future_before_peer_teardown(self):
        for kind in ('E3Pending', 'ValueError'):
            with self.subTest(kind=kind):
                ctx = multiprocessing.get_context('spawn')
                reader, writer = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=fault_worker, args=(writer, kind))
                proc.start(); writer.close()
                blocked_output = Reader()
                transport = ReplyTransport({1: blocked_output}, Status.SUCCESS, handoff_idle=True)
                sequence = transport.submit('execute_model', lambda seq: None, ranks=(1,), timeout_s=300)
                self.assertTrue(blocked_output.entered.wait(1))
                order = []
                executor = SimpleNamespace(workers=[SimpleNamespace(rank=0, proc=proc, fatal_reader=reader)],
                    _flash_s5_transport=transport, is_failed=False, shutting_down=False,
                    failure_callback=lambda: order.append('notified'))
                def shutdown():
                    order.append('teardown')
                    proc.terminate()
                executor.shutdown = shutdown
                try:
                    with patch.dict(os.environ, {'FLASH_S5_OVERLAP':'1'}):
                        hooks.monitor_fatal_workers(executor)
                    until = time.monotonic()+3
                    while not executor.is_failed and time.monotonic()<until:
                        time.sleep(.002)
                    with self.assertRaises(WorkerRPCError) as caught:
                        hooks.LocalFuture(executor, sequence).result(timeout=.1)
                    self.assertEqual(caught.exception.record['rank'], 0)
                    self.assertEqual(caught.exception.record['error_type'], kind)
                    self.assertEqual(order[0], 'notified')
                    self.assertFalse(transport._calls[sequence].replies)
                finally:
                    proc.terminate(); proc.join(2)
                    blocked_output.q.put((Status.SUCCESS, None))
                    transport.close(1)

    def test_idle_core_failure_notice_keeps_typed_cause(self):
        failure=WorkerRPCError(dict(rank=0,method='execute_model',error_type='E3Pending',code='fixture'))
        core=SimpleNamespace(model_executor=SimpleNamespace(_flash_s5_fatal_failure=failure))
        with self.assertRaises(WorkerRPCError) as result:
            hooks.raise_worker_failure(core)
        self.assertIs(result.exception,failure)

    def test_ready_publication_refuses_unbound_enabled_core(self):
        from e3_types import E3ContractError
        with patch.dict(os.environ,{'FLASH_S5_OVERLAP':'1','FLASH_S5_DECODE_ONLY_PROBE':'0'}):
            with self.assertRaises(E3ContractError):
                hooks.require_startup_owner(SimpleNamespace())
            stack,core=OwnerStartup().stack()
            bootstrap_full_startup(core,stack.driver,timeout_s=2)
            hooks.require_startup_owner(core)

    def test_feature_off_does_not_create_fatal_pipe(self):
        with patch.dict(os.environ, {'FLASH_S5_OVERLAP':'0'}):
            self.assertEqual(hooks.make_fatal_pipe(SimpleNamespace()), (None, None))

    def test_callback_observes_typed_failure_before_shutdown(self):
        seen=[]
        executor=SimpleNamespace(_flash_s5_fatal_lock=threading.Lock(), _flash_s5_fatal_failure=None,
            is_failed=False, failure_callback=lambda:seen.append(executor._flash_s5_fatal_failure))
        hooks.notify_worker_failure(executor,None)
        self.assertIsInstance(seen[0],WorkerRPCError)
        hooks.notify_worker_failure(executor,None)
        self.assertEqual(len(seen),1)


class StartupCoexistence(unittest.IsolatedAsyncioTestCase):
    async def test_later_real_prepare_full_reuses_startup_owner_and_fence(self):
        import native_overlap_session as session
        import native_overlap_control as control
        stack, core = OwnerStartup().stack()
        ready = bootstrap_full_startup(core, stack.driver, timeout_s=2)
        _, frontend = core, native.Frontend(core)
        pump = native.Pump(core).start()
        try:
            owner = control.NativeControlClient(frontend, timeout_s=2)
            native_session = session.NativeOverlapSession(frontend, control=owner)
            result = await native_session.execute(dict(schema=1,action='prepare',
                run_id='native38-cpu',target_route='full',phase='full-prefill-solo'))
            self.assertEqual(result['engine_epoch'], ready['engine_epoch'])
            self.assertTrue(result['typed_acks'])
            self.assertEqual(result['route'],'full')
            self.assertIsNone(result['initial_full_fence'])
            self.assertEqual(len(stack.driver.bootstrap_ack_rows()),4)
        finally:
            await pump.stop()


class Health(unittest.IsolatedAsyncioTestCase):
    def install(self):
        import probe_control
        class Response:
            def __init__(self, data=None, status_code=200):
                self.status_code=status_code
        app=SimpleNamespace()
        def middleware(kind):
            def register(fn):
                app.health=fn
                return fn
            return register
        app.middleware=middleware
        with patch.dict(sys.modules, {'fastapi.responses':SimpleNamespace(JSONResponse=Response)}):
            probe_control.install_owner_health(app)
        return app, Response

    async def test_real_status_proof_is_required_then_cached_without_binding(self):
        app, Response=self.install()
        stack, core=OwnerStartup().stack()
        calls=[]
        async def utility(method, command):
            calls.append((method,dict(command)))
            return stack.driver.submit_control(command)
        client=SimpleNamespace(errored=False,engine_core=SimpleNamespace(call_utility_async=utility))
        request=SimpleNamespace(url=SimpleNamespace(path='/health'),
            app=SimpleNamespace(state=SimpleNamespace(engine_client=client)))
        async def next_handler(request):return Response()
        self.assertEqual((await app.health(request,next_handler)).status_code,503)
        self.assertFalse(stack.driver._bootstrapped)
        bootstrap_full_startup(core,stack.driver,timeout_s=2)
        self.assertEqual((await app.health(request,next_handler)).status_code,200)
        count=len(calls)
        self.assertEqual((await app.health(request,next_handler)).status_code,200)
        self.assertEqual(len(calls),count)
        client.errored=True
        self.assertEqual((await app.health(request,next_handler)).status_code,503)
        client.errored=False
        await app.health(request,next_handler)
        self.assertGreater(len(calls),count)

    async def test_replacement_client_cannot_reuse_old_health_latch(self):
        app,Response=self.install()
        async def absent(*a):return {'bootstrapped':False,'engine_epoch':123}
        request=SimpleNamespace(url=SimpleNamespace(path='/health'),
            app=SimpleNamespace(state=SimpleNamespace(engine_client=None)))
        async def next_handler(request):return Response()
        self.assertEqual((await app.health(request,next_handler)).status_code,503)
        request.app.state.engine_client=SimpleNamespace(errored=False,
            engine_core=SimpleNamespace(call_utility_async=absent))
        self.assertEqual((await app.health(request,next_handler)).status_code,503)


if __name__ == '__main__':
    unittest.main()
