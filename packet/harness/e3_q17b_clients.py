#!/usr/bin/env python3
"""One owned controller session: short exactness/placement, then matched W1.

No arbitrary RPC/load command. Auth remains in the existing environment. This
client never starts or stops the engine; the owner queue owns that lifecycle.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time

import e3_probe_rpc as rpc

import e3_decode_correctness as correctness
import e3_w1_decode_driver as w1


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write('\n')


def _audit_binding(call, failures, missing):
    outer = call.get('receipt') or {}
    observer = outer.get('observer')
    core = call.get('receipt_core')
    if type(observer) is not dict or type(core) is not dict:
        missing.append('observer canonical receipt binding absent')
        return
    digest = w1._sha256_json(observer)
    if (outer.get('observer_sha256') != digest or core.get('observer_sha256') != digest
            or call.get('receipt_sha256') != w1._sha256_json(core)
            or call.get('retained_receipt_sha256') != w1._sha256_json(outer)):
        failures.append('observer canonical receipt digest differs')
    for field in ('engine_epoch', 'storage_epoch', 'route_generation_after'):
        if outer.get(field) != core.get(field):
            failures.append('observer canonical identity differs: ' + field)


def placement_grade(data):
    failures, missing = [], []
    runs = data.get('runs', []) if type(data) is dict else []
    split = [r for r in runs if type(r) is dict and r.get('route') == 'split']
    if len(split) != 1:
        return {'verdict': 'INCONCLUSIVE', 'failures': [], 'missing': ['one split correctness run required']}
    calls = [c for c in split[0].get('calls', []) if c.get('action') == 'pause_end']
    if len(calls) != 1 or calls[0].get('lifecycle_ok') is not True:
        return {'verdict': 'INCONCLUSIVE', 'failures': [], 'missing': ['completed split pause_end required']}
    outer = calls[0].get('receipt') or {}
    _audit_binding(calls[0], failures, missing)
    observer = outer.get('observer')
    if type(observer) is not dict:
        return {'verdict': 'INCONCLUSIVE', 'failures': [], 'missing': ['native replay observation absent']}
    rows = observer.get('ranks')
    if (type(rows) is not list or len(rows) != 4
            or any(type(r) is not dict or type(r.get('rank')) is not int for r in rows)
            or {r['rank'] for r in rows} != {0, 1, 2, 3}):
        failures.append('native observer rank coverage differs')
        rows = []
    if observer.get('schema') != 1 or observer.get('stopped') is not True:
        missing.append('observer stop not proved')
    starts = [c for c in split[0].get('calls', []) if c.get('action') == 'pause_enter_split']
    begin_rows = []
    if len(starts) == 1 and starts[0].get('lifecycle_ok') is True:
        _audit_binding(starts[0], failures, missing)
        begin_rows = (starts[0].get('receipt', {}).get('observer') or {}).get('ranks', [])
    if (type(begin_rows) is not list or len(begin_rows) != 4
            or any(type(r) is not dict or type(r.get('rank')) is not int for r in begin_rows)
            or {r['rank'] for r in begin_rows} != {0,1,2,3}):
        missing.append('observer begin binding unavailable')
        begin_rows = []
    begins = {r['rank']:r for r in begin_rows}
    for row in rows:
        rank = row['rank']
        begin = begins.get(rank, {})
        if (begin.get('ack') != 'flash_s5_observer_begin' or begin.get('state') != 'window'
                or type(begin.get('drained_streams')) is not int or begin['drained_streams'] <= 0):
            missing.append(f'rank{rank}: observer begin/drain unproved')
        if any(begin.get(k) != row.get(k) for k in ('worker_incarnation','engine_epoch','storage_epoch','route_generation')):
            failures.append(f'rank{rank}: begin/end observer ownership differs')
        if (row.get('ack') != 'flash_s5_observer_end' or row.get('state') != 'closed'
                or row.get('observer_stopped') is not True
                or type(row.get('drained_streams')) is not int or row['drained_streams'] <= 0):
            missing.append(f'rank{rank}: observer drain/stop/close unproved')
        for field in ('engine_epoch', 'storage_epoch'):
            if row.get(field) != outer.get(field):
                failures.append(f'rank{rank}: {field} differs')
        if row.get('route_generation') != outer.get('route_generation_after'):
            failures.append(f'rank{rank}: route generation differs')
        receipt = row.get('receipt')
        if type(receipt) is not dict:
            missing.append(f'rank{rank}: native receipt absent')
            continue
        if receipt.get('schema') != 's5ro-replay-observer/1':
            failures.append(f'rank{rank}: native receipt schema differs')
        for section in (receipt, receipt.get('static_structure', {}), receipt.get('replay', {})):
            verdict = section.get('verdict') if type(section) is dict else None
            if verdict == 'FAIL':
                failures.append(f'rank{rank}: known graph/context contradiction')
            elif verdict != 'PASS' or section.get('complete') is not True:
                missing.append(f'rank{rank}: placement evidence incomplete')
        proof = receipt.get('caller_proof') or {}
        if (proof.get('worker_rank') != rank
                or proof.get('worker_incarnation') != row.get('worker_incarnation')
                or proof.get('engine_incarnation') != 'engine:' + str(outer.get('engine_epoch'))
                or proof.get('relevant_consumers') != row.get('drained_streams')
                or proof.get('storage_epoch') != outer.get('storage_epoch')
                or proof.get('route_generation') != outer.get('route_generation_after')
                or proof.get('controller_drained') is not True
                or proof.get('route_generation_matches') is not True
                or proof.get('storage_epoch_matches') is not True):
            failures.append(f'rank{rank}: drain proof binding differs')
    if observer.get('complete') is not True:
        missing.append('observer aggregate incomplete')
    return {'schema': 1, 'verdict': 'FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS',
            'failures': sorted(set(failures)), 'missing': sorted(set(missing)),
            'observer_stopped': observer.get('stopped') is True,
            'ranks': sorted(r['rank'] for r in rows),
            'scope': 'actual exercised split-decode graphs, before unprofiled W1'}


def final_status_grade(status, context):
    failures, missing = [], []
    controller = status.get('controller', {}) if type(status) is dict else {}
    if (status.get('read_only') is not True or status.get('sequence_independent') is not True
            or status.get('run_id') != context.get('run_id')):
        missing.append('final read-only owner observation absent')
    if controller.get('engine_epoch') != context.get('engine_epoch'):
        failures.append('final engine epoch differs')
    if status.get('quarantined') is True or controller.get('quarantined') is True:
        failures.append('final controller quarantined')
    if controller.get('paused') is not False:
        missing.append('final resumed engine unproved')
    rows = controller.get('workers')
    if (type(rows) is not list or len(rows) != 4 or
            any(type(r) is not dict or type(r.get('rank')) is not int for r in rows)
            or {r['rank'] for r in rows} != {0,1,2,3}):
        missing.append('final four-worker observation absent'); rows = []
    for row in rows:
        rank = row['rank']; counters = row.get('counters') or {}; inv = row.get('inventory') or {}
        if row.get('status') == 'quarantined' or counters.get('quarantined') is not None:
            failures.append(f'rank{rank}: final quarantine')
        if row.get('status') != 'ok': missing.append(f'rank{rank}: final status unavailable')
        for field, expected in (('route','full'), ('storage_epoch',context.get('storage_epoch')),
                                ('route_generation',context.get('route_generation'))):
            if counters.get(field) != expected or inv.get(field) != expected:
                failures.append(f'rank{rank}: final {field} differs')
        if (type(counters.get('in_flight')) is not int or counters['in_flight'] != 0
                or counters.get('sampling_pending') is not False
                or counters.get('transition_pending') is not False):
            missing.append(f'rank{rank}: final idle unproved')
        observer = inv.get('replay_observer') or {}
        if (observer.get('enabled') is not True or observer.get('closed') is not True
                or observer.get('state') != 'closed'):
            missing.append(f'rank{rank}: observer closed unproved')
        memory = row.get('memory') or {}; allocator = memory.get('allocator') or {}
        if (memory.get('errors') != [] or type(memory.get('device_free_bytes')) is not int
                or type(memory.get('device_total_bytes')) is not int
                or not 0 <= memory['device_free_bytes'] <= memory['device_total_bytes']
                or memory['device_total_bytes'] <= 0
                or any(type(allocator.get(k)) is not int or allocator[k] < 0 for k in (
                    'allocated_current_bytes','reserved_current_bytes','allocated_peak_bytes',
                    'reserved_peak_bytes','inactive_split_current_bytes','inactive_split_peak_bytes'))):
            missing.append(f'rank{rank}: final memory observation incomplete')
    return {'schema':1, 'verdict':'FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS',
            'failures':failures, 'missing':missing,
            'scope':'post-client point observation and allocator lifetime peaks; no reserve threshold'}


class HttpController:
    def __init__(self, url, key):
        rpc.target(url)
        self.url, self.key = url.rstrip('/'), key

    def execute(self, command, timeout_s):
        start = time.monotonic()
        try:
            result = rpc.exchange(command, url=self.url, key=self.key,
                                  timeout_s=timeout_s, raw=False)
        except rpc.ControllerRefusal as exc:
            raise w1.ControllerTransportError('controller_refusal',
                {'kind':'http', 'error':'controller_refusal', 'timeout':False,
                 **exc.record}) from None
        except rpc.TransportError as exc:
            raise w1.ControllerTransportError('http_transport',
                {'kind': 'http', 'error': type(exc).__name__, 'timeout': False}) from None
        return result, {'kind': 'http', 'rc': 0, 'seconds': time.monotonic()-start,
                        'timeout': False, 'stdout_len': 0, 'stderr_len': 0}


def run(args, env=None, controller=None):
    env = dict(os.environ if env is None else env)
    rpc.target(args.url)
    url = args.url.rstrip('/')
    out = Path(args.out)
    out.mkdir(mode=0o700)
    base = Path(__file__).resolve().parent
    command = shlex.join([sys.executable, str(base / 'e3_probe_rpc.py')])
    env.update(RUN_ID=args.run_id, CONTROLLER_CMD=command, PROBE_URL=url,
               API_SOURCE_DIR=str(args.api_source))
    source, problems = correctness.verify_sources(env)
    if problems or source.get('verified') is not True:
        raise ValueError('pinned completion source unavailable or changed')
    if controller is None:
        controller = HttpController(url, env.get('VLLM_API_KEY'))
    context = {}
    cfg = correctness.config_from_env(dict(env, WORKLOAD=str(args.correctness_workload)),
                                      out / 'correctness.json')
    result = correctness.run(cfg, url, args.model, controller,
                             source=source, context=context)
    write(out / 'correctness.json', result)
    placement = placement_grade(result)
    write(out / 'placement.json', placement)
    outcomes = {'correctness': result.get('verdict'), 'placement': placement['verdict']}
    if result.get('verdict') == 'PASS' and placement['verdict'] == 'PASS':
        cfg = w1.config_from_env(dict(env, WORKLOAD=str(args.w1_workload),
            CONTROLLER_TIMEOUT_S='120', READ_TIMEOUT_S='180', DRAIN_TIMEOUT_S='120'),
            out / 'performance.json')
        perf = w1.run(cfg, url, args.model, controller, context=context)
        outcomes['performance'] = perf.get('summary', {}).get('verdict')
        # The W1 summary uses PASS only when every required repeat is valid.
        perf_rc = w1.exit_code(perf)
        outcomes['performance'] = {0: 'PASS', 1: 'FAIL', 2: 'INCONCLUSIVE'}[perf_rc]
    else:
        outcomes['performance'] = 'NOT_RUN'
        write(out / 'performance.json', {'schema': 1, 'verdict': 'NOT_RUN',
              'reason': 'correctness and stopped placement audit must pass first'})
    try:
        status, transport = controller.execute({'schema':1, 'run_id':args.run_id,
                                               'action':'status','read_only':True}, 120)
        status_grade = final_status_grade(status, context)
        write(out / 'final-status.json', {'status':status,'transport':transport,'grade':status_grade})
    except Exception as exc:
        status_grade = {'verdict':'INCONCLUSIVE','missing':['final status unavailable'],
                        'error':type(exc).__name__}
        write(out / 'final-status.json', {'grade':status_grade})
    outcomes['final_status'] = status_grade['verdict']
    verdict = ('FAIL' if 'FAIL' in outcomes.values() else
               'PASS' if all(v == 'PASS' for v in outcomes.values()) else 'INCONCLUSIVE')
    final = {'schema': 1, 'run_id': args.run_id, 'verdict': verdict, 'outcomes': outcomes,
             'engine_epoch': context.get('engine_epoch'),
             'final_route': context.get('route'), 'final_command_seq': context.get('command_seq'),
             'observer_during_performance': False if placement['verdict'] == 'PASS' else None,
             'files': ['correctness.json', 'placement.json', 'performance.json', 'final-status.json']}
    write(out / 'clients-grade.json', final)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--api-source', required=True, type=Path)
    parser.add_argument('--correctness-workload', required=True, type=Path)
    parser.add_argument('--w1-workload', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        # No exception strings: HTTP/config errors could carry user inputs.
        print('Q17B_CLIENTS INCONCLUSIVE ' + type(exc).__name__, file=sys.stderr)
        return 2
    print('Q17B_CLIENTS ' + result['verdict'])
    return {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[result['verdict']]


if __name__ == '__main__':
    raise SystemExit(main())
