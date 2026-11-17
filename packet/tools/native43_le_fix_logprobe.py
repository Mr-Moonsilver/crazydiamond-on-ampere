#!/usr/bin/env python3
"""Native43 le-fix container probe: the worker fatal record through vLLM's own logging, read by the scanner.

Runs inside the pinned image (CPU only, network none).  Two packages are mounted under vllm.v1:
  flash_e3         the l-e overlay e3/ with executor_hooks.py = make_s5_overlap.package(le-fix source);
  flash_e3_prefix  the l-e overlay e3/ as it ran (a copy).
For each package a child process imports vllm.logger (vLLM's own logging configuration) and installs the
real monitor_fatal_workers.  A spawned worker runs the actual generated WorkerProc._execute_worker_rpc
and raises the l-e chain: the package's Row1OracleError(route_mismatch), then its QsaDriverError.  The child's
stdout/stderr get docker-style timestamps, as the box observer's stream has them, and native43_engine_death
scans the result.  Expected: le-fix DEAD worker_fatal_record, with the line matching the unchanged l-e
engine-errors grep; pre-fix ALIVE, no line (the l-e silence).
"""
import argparse
import ast
from functools import partial
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any

SOURCE = Path('/source')
LE_GATE = r'Traceback|AssertionError|CUDA error|Xid|ContractError|EngineDeadError|FLASH_S5_.*FAIL'


def generated_rpc(hooks):
    source = (SOURCE / 'reference/vllm-src/v1/executor/multiproc_executor.py').read_text()
    generated = hooks.generate_executor(source)
    tree = ast.parse(generated)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WorkerProc')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_execute_worker_rpc')
    namespace = dict(Any=Any, partial=partial, traceback=traceback, _flash_s5_hooks=hooks,
                     logger=SimpleNamespace(exception=lambda *a: None))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), 'generated-rpc', 'exec'),
         namespace)
    return namespace['_execute_worker_rpc']


def fault(package, writer):
    hooks = importlib.import_module('vllm.v1.%s.executor_hooks' % package)
    qsa = importlib.import_module('vllm.v1.%s.qsa_driver' % package)
    oracle = importlib.import_module('vllm.v1.%s.row1_oracle' % package)
    hooks.set_fatal_writer(writer)

    def execute_model():
        try:
            raise oracle.Row1OracleError('row1 qualification failed: route_mismatch')
        except oracle.Row1OracleError as exc:
            raise qsa.QsaDriverError('native row1 qualification failed: ' + str(exc)) from exc
    generated_rpc(hooks)(SimpleNamespace(rank=0, worker=SimpleNamespace(execute_model=execute_model)),
                         ('execute_model', (), {}, 1))


def scenario(package):
    import vllm.logger  # noqa: F401  vLLM's own logging configuration of the "vllm" logger
    hooks = importlib.import_module('vllm.v1.%s.executor_hooks' % package)
    ctx = multiprocessing.get_context('spawn')
    reader, writer = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=fault, args=(package, writer))
    proc.start()
    writer.close()
    done = threading.Event()
    executor = SimpleNamespace(workers=[SimpleNamespace(rank=0, proc=proc, fatal_reader=reader)], is_failed=False,
                               shutting_down=False, failure_callback=lambda: None)
    executor.shutdown = lambda: (proc.terminate(), done.set())
    os.environ['FLASH_S5_OVERLAP'] = '1'
    hooks.monitor_fatal_workers(executor)
    ok = done.wait(120)
    proc.join(10)
    sys.stdout.flush()
    failure = getattr(executor, '_flash_s5_fatal_failure', None)
    print('SCENARIO_RESULT ' + json.dumps(dict(package=package, shutdown_called=ok,
                                               record=None if failure is None else failure.record), sort_keys=True))
    return 0 if ok else 1


def stamp(text):
    lines = []
    for line in text.splitlines():
        ns = time.time_ns()
        lines.append(time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(ns // 10**9)) + '.%09dZ ' % (ns % 10**9) + line)
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario')
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    if args.scenario:
        return scenario(args.scenario)
    sys.path.insert(0, str(SOURCE / 'harness'))
    import native43_engine_death as death
    rows = {}
    for package in ('flash_e3', 'flash_e3_prefix'):
        result = subprocess.run([sys.executable, '-B', __file__, '--scenario', package], capture_output=True,
                                text=True, timeout=600)
        log = args.out.parent / ('logprobe-%s.log' % package)
        log.write_text(stamp(result.stdout + result.stderr))
        scanner = death.Scanner([log])
        scanner.poll(final=True)
        decision = scanner.decide()
        fatal = [line for line in log.read_text().splitlines() if 'FLASH_S5_WORKER_FAILURE' in line]
        reported = [json.loads(line.split('SCENARIO_RESULT ', 1)[1]) for line in result.stdout.splitlines()
                    if line.startswith('SCENARIO_RESULT ')]
        rows[package] = dict(rc=result.returncode, scan_verdict=decision['verdict'],
                             first_death=decision['first_death'], fatal_lines=fatal,
                             le_gate_match=any(re.search(LE_GATE, line) for line in fatal),
                             scenario=reported[0] if reported else None)
    fixed, prefix = rows['flash_e3'], rows['flash_e3_prefix']
    record = (fixed['scenario'] or {}).get('record') or {}
    checks = {
        'fixed_shutdown_called': fixed['rc'] == 0 and (fixed['scenario'] or {}).get('shutdown_called') is True,
        'fixed_one_error_line': len(fixed['fatal_lines']) == 1 and ' ERROR ' in fixed['fatal_lines'][0],
        'fixed_scan_dead_worker_fatal_record': fixed['scan_verdict'] == 'DEAD'
                                               and (fixed['first_death'] or {}).get('kind') == 'worker_fatal_record',
        'fixed_le_gate_matches': fixed['le_gate_match'],
        'fixed_record_cause': (record.get('error_type'), record.get('cause'), record.get('detail')) == (
            'QsaDriverError', 'Row1OracleError',
            'native row1 qualification failed: row1 qualification failed: route_mismatch'),
        'fixed_vllm_format': bool(fixed['fatal_lines']) and re.search(
            r'Z ERROR \d\d-\d\d \d\d:\d\d:\d\d \[executor_hooks\.py:\d+\] FLASH_S5_WORKER_FAILURE rank=0 ',
            fixed['fatal_lines'][0]) is not None,
        'prefix_shutdown_called': prefix['rc'] == 0 and (prefix['scenario'] or {}).get('shutdown_called') is True,
        'prefix_silent': prefix['fatal_lines'] == [] and prefix['scan_verdict'] == 'ALIVE',
    }
    out = dict(schema='native43-le-fix-logprobe/1', verdict='PASS' if all(checks.values()) else 'FAIL',
               checks=checks, packages=rows)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    print('LOGPROBE', out['verdict'], json.dumps(checks, sort_keys=True))
    return 0 if out['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
