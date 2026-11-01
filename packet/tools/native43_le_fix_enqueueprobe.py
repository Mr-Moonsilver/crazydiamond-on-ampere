#!/usr/bin/env python3
"""Native43 le-fix bounded-enqueue probe (pinned image, CPU only): the l-e hang on vLLM's real MessageQueue.

The EngineCore broadcast ring is vLLM's real shared-memory MessageQueue (shm_broadcast, 1 local reader,
max_chunks 2), filled so the next write must wait.  The writer is the actual generated stock
MultiprocExecutor collective_rpc (generate_executor over the pinned multiproc_executor.py) or the actual
owned executor_hooks.collective_rpc behind install(); the failure arrives as in l-e, a worker fatal record on
the real monitor_fatal_workers pipe, which marks the executor failed and calls its shutdown.  Each case runs
in its own process (a blocked pre-fix writer spins until that process exits):

  stock-prefix / owned-prefix  parent (299f27e) executor_hooks: the writer is still blocked 6 s after the record;
  stock-fixed / owned-fixed    le-fix executor_hooks: the writer raises the stored WorkerRPCError within the bound;
  slow-fixed                   le-fix, live executor: a real reader frees one block after 3 s; the write completes
                               once, the reader reads it back, and the stock long-wait warning still appears
                               (VLLM_RINGBUFFER_WARNING_INTERVAL=1 for this case).

Leaves (declared in harness/native43_le_bounded_enqueue_fixture_leaves.json): the executor object, its
shutdown (records the call) and the owned route's reply readers.  `--out` writes the record.
"""
import argparse
import ast
import collections
import importlib.util
import json
import logging
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

SOURCE = Path('/source')
PARENT = Path('/parent')
REFERENCE = SOURCE / 'reference/vllm-src/v1/executor/multiproc_executor.py'
SHARED = ('e3_types.py', 'reply_transport.py', 'async_handoff.py')
BOUND_S = 2.0  # one 1 s slice plus the fatal thread's hand-off and scheduling slack
HANG_S = 6.0


def load_hooks(tree):
    sys.path.insert(0, str(SOURCE / 'patches/s5_engine'))
    spec = importlib.util.spec_from_file_location('hooks_' + tree.name, tree / 'patches/s5_engine/executor_hooks.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stock_rpc(hooks):
    generated = hooks.generate_executor(REFERENCE.read_text())
    tree = ast.parse(generated)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiprocExecutor')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_flash_s5_stock_collective_rpc')
    module = ast.fix_missing_locations(ast.Module(
        body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn],
        type_ignores=[]))
    scope = dict(_flash_s5_hooks=hooks, time=time,
                 FutureWrapper=lambda queue, get_response, aggregate: SimpleNamespace(queue=queue))
    exec(compile(module, 'generated-stock-collective-rpc', 'exec'), scope)
    return scope['_flash_s5_stock_collective_rpc'], generated.count('_flash_s5_hooks.bounded_enqueue(')


def ring():
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
    queue = MessageQueue(1, 1, max_chunk_bytes=1024 * 1024, max_chunks=2)
    for i in range(2):
        queue.enqueue(('fill', i))  # both blocks written, none read: the next write must wait
    return queue


def executor_with_channel(hooks, queue):
    reader, writer = multiprocessing.Pipe(duplex=False)
    executor = SimpleNamespace(rpc_broadcast_mq=queue, response_mqs=[None], is_failed=False, world_size=1,
                               futures_queue=collections.deque(), shutting_down=False, failure_callback=None,
                               shutdowns=[])
    executor.workers = [SimpleNamespace(rank=0, proc=SimpleNamespace(pid=os.getpid()), fatal_reader=reader)]

    def shutdown():
        executor.shutdowns.append(time.monotonic())
        executor.shutting_down = True
    executor.shutdown = shutdown
    os.environ['FLASH_S5_OVERLAP'] = '1'
    hooks.monitor_fatal_workers(executor)
    return executor, writer


def publish(writer):
    record = dict(schema='s5-worker-fatal/1', rank=0, pid=os.getpid(), method='execute_model',
                  error_type='QsaDriverError', code='worker-rpc-exception', cause='Row1OracleError',
                  where='qsa_driver.py:382:_tail', detail='native row1 qualification failed: route_mismatch')
    writer.send_bytes(json.dumps(record).encode())
    return time.monotonic()


def case(name):
    import vllm.logger  # noqa: F401  vLLM's logging configuration
    tree = PARENT if name.endswith('prefix') else SOURCE
    hooks = load_hooks(tree)
    queue = ring()
    executor, writer = executor_with_channel(hooks, queue)
    row = dict(case=name, tree=str(tree), writes_before=queue.current_idx)
    if name.startswith('owned'):
        import async_handoff as ah

        class Reader:  # owned-route reply reader: no reply is ever due here
            def dequeue(self, timeout):
                time.sleep(min(timeout or 0.05, 0.05))
                raise TimeoutError
        executor.response_mqs = [Reader()]
        hooks.install(executor, config=ah.HandoffConfig.from_env({'FLASH_S5_OVERLAP': '1'}), success_status=object())
        call = lambda: hooks.collective_rpc(executor, None, 'execute_model', timeout=60)  # noqa: E731
    else:
        rpc, row['generated_bounded_sites'] = stock_rpc(hooks)
        call = lambda: rpc(executor, 'execute_model', non_block=True)  # noqa: E731
    seen = []
    if name == 'slow-fixed':
        class Capture(logging.Handler):
            def emit(self, record):
                seen.append(record.getMessage())
        logging.getLogger('vllm.distributed.device_communicators.shm_broadcast').addHandler(Capture())
    result = SimpleNamespace(error=None, ended=None, value=None)

    def run():
        try:
            result.value = call()
        except BaseException as exc:  # noqa: BLE001
            result.error = exc
        result.ended = time.monotonic()
    thread = threading.Thread(target=run, daemon=True)
    started = time.monotonic()
    thread.start()
    if name == 'slow-fixed':
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
        reader = MessageQueue.create_from_handle(queue.export_handle(), 0)
        time.sleep(3.0)
        row['alive_before_read'] = thread.is_alive()
        first = reader.dequeue(timeout=5)
        thread.join(5)
        second, third = reader.dequeue(timeout=5), reader.dequeue(timeout=5)
        row.update(finished=not thread.is_alive(), error=None if result.error is None else type(result.error).__name__,
                   wait_s=round(result.ended - started, 3) if result.ended else None, read_back=[first, second, third],
                   long_wait_warnings=[m for m in seen if m.startswith('No available shared memory broadcast block')])
        ok = (row['alive_before_read'] and row['finished'] and row['error'] is None and row['wait_s'] >= 3.0
              and third == ('execute_model', (), {}, None) and len(row['long_wait_warnings']) >= 2)
    else:
        time.sleep(1.5)
        row['alive_before_record'] = thread.is_alive()
        sent = publish(writer)
        thread.join(HANG_S)
        row.update(alive_after_s=HANG_S if thread.is_alive() else None, shutdowns=len(executor.shutdowns),
                   is_failed=executor.is_failed,
                   error=None if result.error is None else type(result.error).__name__,
                   error_is_stored_failure=result.error is not None
                   and result.error is getattr(executor, '_flash_s5_fatal_failure', None),
                   raised_after_record_s=None if result.ended is None else round(result.ended - sent, 3))
        if name.endswith('prefix'):
            ok = row['alive_before_record'] and row['alive_after_s'] == HANG_S and row['is_failed'] and row['shutdowns'] == 1
        else:
            ok = (row['alive_before_record'] and row['alive_after_s'] is None and row['error'] == 'WorkerRPCError'
                  and row['error_is_stored_failure'] and row['raised_after_record_s'] < BOUND_S
                  and row['shutdowns'] == 1)
    row['ok'] = bool(ok)
    sys.stdout.write('CASE_RESULT ' + json.dumps(row, sort_keys=True, default=repr) + '\n')
    sys.stdout.flush()
    os._exit(0)  # a blocked pre-fix writer thread would otherwise spin forever


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--case')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.case:
        return case(a.case)
    import hashlib
    shared = {n: [hashlib.sha256((t / 'patches/s5_engine' / n).read_bytes()).hexdigest() for t in (SOURCE, PARENT)]
              for n in SHARED}
    rows = {}
    for name in ('stock-prefix', 'stock-fixed', 'owned-prefix', 'owned-fixed', 'slow-fixed'):
        env = dict(os.environ, VLLM_RINGBUFFER_WARNING_INTERVAL='1') if name == 'slow-fixed' else None
        done = subprocess.run([sys.executable, '-B', __file__, '--case', name], capture_output=True, text=True,
                              timeout=120, env=env)
        (a.out.parent / ('enqueueprobe-%s.log' % name)).write_text(done.stdout + done.stderr)
        found = [json.loads(line.split(' ', 1)[1]) for line in done.stdout.splitlines() if line.startswith('CASE_RESULT ')]
        rows[name] = dict(rc=done.returncode, **(found[0] if found else dict(ok=False)))
    checks = {name: row['ok'] and row['rc'] == 0 for name, row in rows.items()}
    checks['shared_modules_equal'] = all(len(set(v)) == 1 for v in shared.values())
    checks['fixed_stock_has_one_bounded_site'] = rows['stock-fixed'].get('generated_bounded_sites') == 1
    checks['prefix_stock_has_none'] = rows['stock-prefix'].get('generated_bounded_sites') == 0
    out = dict(schema='native43-le-fix-enqueueprobe/1', verdict='PASS' if all(checks.values()) else 'FAIL',
               checks=checks, cases=rows, shared_sha256=shared, bound_s=BOUND_S, hang_s=HANG_S)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True, default=repr) + '\n')
    print('ENQUEUEPROBE', out['verdict'], json.dumps(checks, sort_keys=True))
    return 0 if out['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
