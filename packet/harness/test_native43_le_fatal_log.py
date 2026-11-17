"""Native43 le-fix (b): the parent logs the worker fatal record before the executor teardown it starts.

l-e (1 Nov 2026): the flash-s5 fatal channel shut the executor down at 11:03:08Z and no process logged why.
These cases run the actual generated WorkerProc._execute_worker_rpc in a spawned worker process, the actual
report_worker_failure/monitor_fatal_workers pipe and the le-fix scanner on the logged line.
Leaves: native43_le_fatal_log_fixture_leaves.json.
"""
import ast
from functools import partial
import json
import logging
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'patches/s5_engine'), str(ROOT / 'harness')]
import executor_hooks as hooks  # noqa: E402
import native43_engine_death as death  # noqa: E402
from qsa_driver import QsaDriverError  # noqa: E402
from reply_transport import WorkerRPCError  # noqa: E402
from row1_oracle import Row1OracleError  # noqa: E402

# The unchanged l-e engine-errors grep (harness/runs/q18a_native43_lifecycle.draft.sh capture_logs).
LE_GATE = r'Traceback|AssertionError|CUDA error|Xid|ContractError|EngineDeadError|FLASH_S5_.*FAIL'


def generated_rpc():
    """The actual generated WorkerProc._execute_worker_rpc over the pinned multiproc_executor.py."""
    source = (ROOT / 'reference/vllm-src/v1/executor/multiproc_executor.py').read_text()
    generated = hooks.generate_executor(source)
    assert hooks.revert_executor(generated) == source
    tree = ast.parse(generated)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WorkerProc')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_execute_worker_rpc')
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = dict(Any=Any, partial=partial, traceback=traceback, _flash_s5_hooks=hooks,
                     logger=SimpleNamespace(exception=lambda *a: None))
    exec(compile(module, 'actual-generated-worker-rpc', 'exec'), namespace)
    return namespace['_execute_worker_rpc']


def le_row1_failure():
    """The l-e chain: Row1Qualifier.fail -> Row1OracleError -> qsa_driver QsaDriverError (qsa_driver.py 377-382)."""
    try:
        raise Row1OracleError('row1 qualification failed: route_mismatch')
    except Row1OracleError as exc:
        raise QsaDriverError('native row1 qualification failed: ' + str(exc)) from exc


def fault_worker(writer, kind):
    hooks.set_fatal_writer(writer)

    def execute_model():
        if kind in ('native-layer', 'native-layer-long'):
            import test_e3_lane_dispatch as rig
            # Same fixture setup as DispatcherStage0Tests; the dispatcher and
            # its exception chain are real, only layer execution is replaced.
            d, lane, channel, server, admission, operation = rig.make(True, rig.fake_torch())
            d.open(rig.TXN, admission, operation)
            d.submit(rig.cmd('prepare', 1))
            d.submit(rig.cmd('prefix', 2))
            def layer_failure(index):
                raise RuntimeError(('xé"\\' * 400) if kind.endswith('long') else 'CUDA error: allocation failed (fixture)')
            lane.layer = layer_failure
            d.submit(rig.cmd('layer', 3, layer_start=0, layer_end=1))
        if kind == 'le-row1':
            le_row1_failure()
        if kind == 'own-long':
            raise QsaDriverError('xé"\\' * 400)
        if kind == 'exit':
            os._exit(3)
        raise ValueError('foreign payload text must not travel')
    worker = SimpleNamespace(rank=0, worker=SimpleNamespace(execute_model=execute_model))
    generated_rpc()(worker, ('execute_model', (), {}, 1))


class Capture(logging.Handler):
    def __init__(self, order):
        super().__init__(logging.DEBUG)
        self.order, self.rows = order, []

    def emit(self, record):
        self.rows.append(record)
        self.order.append('logged')


class FatalLog(unittest.TestCase):
    def run_fault(self, kind, *, shutting_down=False):
        """Spawned worker + the actual monitor; returns (order, log rows, executor)."""
        ctx = multiprocessing.get_context('spawn')
        reader, writer = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=fault_worker, args=(writer, kind))
        proc.start()
        writer.close()
        self.addCleanup(lambda: (proc.terminate(), proc.join(5)))
        order = []
        capture = Capture(order)
        logger = logging.getLogger(hooks.__name__)
        logger.addHandler(capture)
        self.addCleanup(logger.removeHandler, capture)
        done = threading.Event()
        executor = SimpleNamespace(workers=[SimpleNamespace(rank=0, proc=proc, fatal_reader=reader)],
                                   is_failed=False, shutting_down=shutting_down,
                                   failure_callback=lambda: order.append('notified'))

        def shutdown():
            order.append('teardown')
            proc.terminate()
            done.set()
        executor.shutdown = shutdown
        os.environ['FLASH_S5_OVERLAP'] = '1'
        try:
            hooks.monitor_fatal_workers(executor)
        finally:
            os.environ.pop('FLASH_S5_OVERLAP', None)
        if not shutting_down:
            self.assertTrue(done.wait(60), 'executor.shutdown() was not called')
        else:
            proc.join(60)
            time.sleep(0.5)
        return order, capture.rows, executor

    def test_le_row1_failure_is_logged_after_notify_before_teardown(self):
        order, rows, executor = self.run_fault('le-row1')
        self.assertEqual(order, ['notified', 'logged', 'teardown'])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].levelno, logging.ERROR)
        line = rows[0].getMessage()
        record = executor._flash_s5_fatal_failure.record
        self.assertIsInstance(executor._flash_s5_fatal_failure, WorkerRPCError)
        self.assertEqual((record['rank'], record['method'], record['error_type'], record['code'], record['cause']),
                         (0, 'execute_model', 'QsaDriverError', 'worker-rpc-exception', 'Row1OracleError'))
        self.assertEqual(record['detail'],
                         'native row1 qualification failed: row1 qualification failed: route_mismatch')
        self.assertRegex(record['where'], r'^test_native43_le_fatal_log\.py:\d+:le_row1_failure$')
        for part in ('FLASH_S5_WORKER_FAILURE rank=0 ', 'method=execute_model', 'error_type=QsaDriverError',
                     'cause=Row1OracleError', "detail='native row1 qualification failed: row1 qualification "
                     "failed: route_mismatch'", 'executor shutdown follows (worker fatal channel)'):
            self.assertIn(part, line)
        self.assertNotIn('\n', line)

    def test_foreign_exception_message_is_not_serialized(self):
        order, rows, executor = self.run_fault('foreign')
        record = executor._flash_s5_fatal_failure.record
        self.assertEqual(record['error_type'], 'ValueError')
        self.assertNotIn('detail', record)
        self.assertNotIn('foreign payload', rows[0].getMessage())
        self.assertIn('detail=None', rows[0].getMessage())
        self.assertEqual(order, ['notified', 'logged', 'teardown'])

    def test_native_layer_quarantine_keeps_the_device_cause_before_teardown(self):
        order, rows, executor = self.run_fault('native-layer')
        record = executor._flash_s5_fatal_failure.record
        self.assertEqual(order, ['notified', 'logged', 'teardown'])
        self.assertEqual(record['error_type'], 'LaneDispatchError')
        self.assertEqual(record['cause'], 'RuntimeError')
        self.assertEqual(record['native_message'], 'RuntimeError: CUDA error: allocation failed (fixture)')
        self.assertIn('lane_dispatch.py:', record['native_trace'])
        self.assertIn(':_do_layer', record['native_trace'])
        self.assertIn(':layer_failure', record['native_trace'])
        self.assertNotIn(str(ROOT), record['native_trace'])
        line = rows[0].getMessage()
        self.assertIn(record['native_message'], line)
        self.assertIn('layer_failure', line)
        self.assertNotIn('\n', line)

    def test_escape_heavy_native_cause_still_fits_the_pipe_and_keeps_the_raising_frame(self):
        _, rows, executor = self.run_fault('native-layer-long')
        record = executor._flash_s5_fatal_failure.record
        self.assertLessEqual(len(json.dumps(record).encode()), hooks.FATAL_RECORD_MAX)
        self.assertTrue(record['native_message'].startswith('RuntimeError:'))
        self.assertTrue(record['native_message'].isascii())
        self.assertIn(':layer_failure', record['native_trace'])
        self.assertIn('native_message=', rows[0].getMessage())

    def test_long_non_ascii_own_message_stays_inside_the_1024_byte_record(self):
        _, rows, executor = self.run_fault('own-long')
        record = executor._flash_s5_fatal_failure.record
        self.assertLessEqual(len(json.dumps(record).encode()), hooks.FATAL_RECORD_MAX)
        self.assertLessEqual(len(record['detail']), hooks.FATAL_OPTIONAL['detail'])
        self.assertTrue(record['detail'].isascii())
        self.assertIn('error_type=QsaDriverError', rows[0].getMessage())

    def test_worker_exit_without_record_logs_unavailable(self):
        order, rows, executor = self.run_fault('exit')
        self.assertEqual(order, ['notified', 'logged', 'teardown'])
        self.assertIn('record=unavailable (EOFError)', rows[0].getMessage())
        self.assertEqual(executor._flash_s5_fatal_failure.record['error_type'], 'WorkerDied')

    def test_orderly_shutdown_eof_stays_silent(self):
        order, rows, _ = self.run_fault('exit', shutting_down=True)
        self.assertEqual((order, rows), ([], []))

    def test_invalid_optional_field_refuses_the_record(self):
        rows = []
        bad = dict(schema='s5-worker-fatal/1', rank=0, pid=1, method='m', error_type='E', code='c', detail=7)
        ctx = multiprocessing.get_context('spawn')
        reader, writer = ctx.Pipe(duplex=False)
        writer.send_bytes(json.dumps(bad).encode())
        writer.close()
        done = threading.Event()
        order = []
        capture = Capture(order)
        logger = logging.getLogger(hooks.__name__)
        logger.addHandler(capture)
        self.addCleanup(logger.removeHandler, capture)
        executor = SimpleNamespace(workers=[SimpleNamespace(rank=0, proc=SimpleNamespace(pid=1), fatal_reader=reader)],
                                   is_failed=False, shutting_down=False, failure_callback=None,
                                   shutdown=lambda: done.set())
        os.environ['FLASH_S5_OVERLAP'] = '1'
        try:
            hooks.monitor_fatal_workers(executor)
        finally:
            os.environ.pop('FLASH_S5_OVERLAP', None)
        self.assertTrue(done.wait(10))
        rows = capture.rows
        self.assertIn('record=unavailable (ValueError)', rows[0].getMessage())

    def test_logged_line_is_caught_by_the_scanner_and_by_the_unchanged_le_gate(self):
        import re
        _, rows, _ = self.run_fault('le-row1')
        # The box line shape: docker timestamp + vLLM's process prefix and formatter around the message.
        line = ('2026-11-01T11:03:08.280000000Z (EngineCore pid=85) ERROR 11-01 11:03:08 [executor_hooks.py:%d] %s'
                % (rows[0].lineno, rows[0].getMessage()))
        self.assertEqual(death.classify(line.encode()), ('hard', 'worker_fatal_record'))
        self.assertRegex(line, LE_GATE)


if __name__ == '__main__':
    unittest.main()
