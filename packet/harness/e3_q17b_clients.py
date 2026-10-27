#!/usr/bin/env python3
"""One owned controller session: short exactness/placement, then matched W1.

No arbitrary RPC/load command. Auth remains in the existing environment. This
client never starts or stops the engine; the owner queue owns that lifecycle.
"""
import argparse
from dataclasses import replace
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


def replay_multiplicity(begin, end, receipt):
    """Independent router submissions vs every static node's record count.

    Both censuses come from drained worker boundaries, are bound to canonical
    observer receipts, and count successful backend replay calls, not CUPTI
    records. Missing or excess records are incomplete evidence, not model FAIL.
    """
    def require(value):
        if not value:
            raise ValueError('incomplete replay census')

    try:
        a, b = begin['replay_census'], end['replay_census']
        slots = receipt['replay']['graph_slots']
        require(type(slots) is list and slots)
        labels = [slot['label'] for slot in slots]
        require(len(set(labels)) == len(labels))
        for census in (a, b):
            require(type(census) is dict and type(census['graphs']) is dict)
            require(set(census['graphs']) == set(labels))
            require(type(census['total']) is int and census['total'] >= 0)
            require(all(type(n) is int and n >= 0 for n in census['graphs'].values()))
            require(sum(census['graphs'].values()) == census['total'])
        require(b['total'] > a['total'])
        for slot in slots:
            count = b['graphs'][slot['label']] - a['graphs'][slot['label']]
            require(count >= 0)
            for kind, field in (('kernel', 'kernel_executions'),
                                ('memcpy', 'copy_executions'),
                                ('memset', 'memset_executions')):
                required = slot['required'][kind]
                records = slot[field]
                pair = slot['node_execution_bounds'][kind]
                require(type(required) is int and required >= 0)
                require(type(records) is int and records == required * count)
                expected = count if required else 0
                require(all(type(pair[k]) is int and pair[k] == expected for k in ('min', 'max')))
        return True
    except (KeyError, TypeError, ValueError):
        return False

def replay_capacity(begin, end, receipt):
    """Measured inventory/census against fixed storage; never assumes graph growth fits."""
    try:
        limits=receipt['limits']; slots=receipt['replay']['graph_slots']
        demand={'kernel':0,'copy':0}; max_kernel=0
        for slot in slots:
            n=end['replay_census']['graphs'][slot['label']]-begin['replay_census']['graphs'][slot['label']]
            if type(n) is not int or n<0:raise ValueError('invalid census')
            k=slot['required']['kernel']; c=slot['required']['memcpy']+slot['required']['memset']
            demand['kernel']+=k*n; demand['copy']+=c*n; max_kernel=max(max_kernel,k)
        for key in ('max_kernel_records','max_copy_records','max_graph_events'):
            if type(limits[key]) is not int or limits[key]<=0:raise ValueError('invalid limit')
        counts=receipt['replay']['counts']
        events=counts['resource_events']
        if type(events) is not int or events<0:raise ValueError('invalid events')
        fits=(demand['kernel']<=limits['max_kernel_records'] and
              demand['copy']<=limits['max_copy_records'] and events<=limits['max_graph_events'])
        return {'verdict':'PASS' if fits else 'INCONCLUSIVE','required_kernel_records':demand['kernel'],
                'required_copy_records':demand['copy'],'kernel_record_limit':limits['max_kernel_records'],
                'copy_record_limit':limits['max_copy_records'],'resource_events_seen':events,
                'resource_event_limit':limits['max_graph_events'],
                'largest_graph_kernel_nodes':max_kernel,
                'largest_graph_replay_capacity':limits['max_kernel_records']//max_kernel if max_kernel else 0,
                'scope':'inventory-derived ring bound; buffers, aliases and lifetime loss flags are checked separately'}
    except (KeyError,TypeError,ValueError):
        return {'verdict':'INCONCLUSIVE','missing':'capacity evidence unavailable'}


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
    capacities = {}
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
        capacities[str(rank)] = replay_capacity(begin,row,receipt)
        if capacities[str(rank)]['verdict'] != 'PASS':
            missing.append(f'rank{rank}: replay capacity incomplete')
        if not replay_multiplicity(begin, row, receipt):
            missing.append(f'rank{rank}: replay multiplicity incomplete')
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
            'ranks': sorted(r['rank'] for r in rows), 'capacity':capacities,
            'scope': 'actual exercised split-decode graphs, before unprofiled W1'}


def data_invariant_grade(data):
    failures, missing = [], []
    runs=[r for r in data.get('runs',[]) if r.get('route')=='split']
    if len(runs)!=1:
        return {'verdict':'INCONCLUSIVE','failures':[],'missing':['one split cohort required']}
    phases={}
    for call in runs[0].get('calls',[]):
        if not call.get('lifecycle_ok'):continue
        receipt=call.get('receipt') or {}; core=call.get('receipt_core') or {}
        evidence=receipt.get('data_invariant')
        if not evidence:continue
        if (core.get('data_invariant')!=evidence or call.get('receipt_sha256')!=w1._sha256_json(core)):
            failures.append('byte-proof receipt binding differs');continue
        for phase,record in evidence.items():
            if phase in phases:failures.append('duplicate byte-proof phase')
            phases[phase]=record
            ranks=record.get('ranks',[])
            if len(ranks)!=4 or {r.get('rank') for r in ranks}!={0,1,2,3}:
                missing.append('four worker byte proofs required');continue
            for r in ranks:
                if (r.get('engine_epoch')!=core.get('engine_epoch') or r.get('storage_epoch')!=core.get('storage_epoch')
                        or r.get('route')!=('full' if phase in ('capture','returned') else 'split')
                        or r.get('route_generation')!=core.get('route_generation_before' if phase=='capture' else 'route_generation_after')
                        or r.get('phase')!=phase
                        or type(r.get('drained_streams')) is not int or r['drained_streams']<=0):
                    failures.append('byte-proof boundary differs')
                if r.get('verdict')=='FAIL':failures.append('rank%d: %s'%(r['rank'],r.get('failure','byte mismatch')))
                elif r.get('verdict')!='PASS':missing.append('rank%d: byte proof incomplete'%r['rank'])
                elif phase!='capture' and (r.get('local_layers')!=6 or r.get('cohort_size')!=8
                        or type(r.get('history_bytes')) is not int or r['history_bytes']<=0
                        or type(r.get('graph_count')) is not int or r['graph_count']<=0):
                    missing.append('byte proof coverage missing')
    if set(phases)!={'capture','entered','measured','returned'}:missing.append('all drained byte boundaries required')
    return {'schema':1,'verdict':'FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS',
            'failures':failures,'missing':missing,'phases':phases,
            'scope':'immutable cohort prompt-prefix QSA KV/host history and split graph definitions only'}


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
    teacher_diagnostic = getattr(args, 'teacher_diagnostic', False) is True
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
    # Keep the short replay placement window bounded (~40 observed replays in
    # full8). The longer teacher window uses the same immutable bank afterwards,
    # with native execution tracing already stopped and its own route proofs.
    cfg=correctness.config_from_env(dict(env,WORKLOAD=str(args.correctness_workload)),out/'correctness.json')
    result=correctness.run(cfg,url,args.model,controller,source=source,context=context)
    write(out/'correctness.json',result)
    placement=placement_grade(result)
    write(out/'placement.json',placement)
    launch_rows=[]
    for r in result.get('runs',[]):
        if r.get('route')=='split':
            for call in r.get('calls',[]):
                if call.get('action')=='pause_end':
                    for row in (call.get('receipt',{}).get('observer') or {}).get('ranks',[]):
                        launch_rows.append({'rank':row['rank'],'diagnostic':row.get('launch_config')})
    launch_verdicts=[(r['diagnostic'] or {}).get('verdict') for r in launch_rows]
    launch={'schema':1,'gates_w1':False,'ranks':launch_rows,
            'verdict':'DIFFER' if 'DIFFER' in launch_verdicts else
                'PASS' if len(launch_rows)==4 and all(v=='PASS' for v in launch_verdicts) else 'INCONCLUSIVE'}
    write(out/'launch-config.json',launch)
    from e3_teacher_decode_client import Instrument
    instrument=Instrument(args.teacher_baseline,snapshot_max_bytes=args.snapshot_max_bytes)
    cfg=correctness.config_from_env(dict(env,WORKLOAD=str(args.teacher_workload)),out/'teacher-forced.json')
    cfg=replace(cfg,instrument=instrument)
    try:
        teacher=correctness.run(cfg,url,args.model,controller,source=source,context=context)
    except Exception as exc:
        # A result-assembly failure may occur AFTER engine work. Do not call
        # this NOT_RUN, retry it, or erase an already recorded placement FAIL.
        # Exception values can contain request data: retain only their type.
        teacher={'schema':1,'verdict':'INCONCLUSIVE',
                 'verdict_reasons':['teacher_client_exception'],
                 'error_type':type(exc).__name__,
                 'evidence_scope':'teacher execution may have occurred; result unavailable'}
    write(out/'teacher-forced.json',teacher)
    data_exact=data_invariant_grade(teacher)
    write(out/'data-invariant.json',data_exact)
    outcomes={'teacher_forced':teacher.get('verdict'),'data_invariant':data_exact['verdict'],
              'placement':placement['verdict']}
    if not teacher_diagnostic and all(v=='PASS' for v in outcomes.values()):
        cfg=w1.config_from_env(dict(env,WORKLOAD=str(args.w1_workload),
            CONTROLLER_TIMEOUT_S='120',READ_TIMEOUT_S='180',DRAIN_TIMEOUT_S='120'),out/'performance.json')
        perf=w1.run(cfg,url,args.model,controller,context=context)
        outcomes['performance']={0:'PASS',1:'FAIL',2:'INCONCLUSIVE'}[w1.exit_code(perf)]
    else:
        outcomes['performance']='NOT_RUN'
        write(out/'performance.json',{'schema':1,'verdict':'NOT_RUN',
              'reason':('teacher-only diagnostic: W1 unconditionally excluded' if teacher_diagnostic else
                        'scoped byte invariants, rule v3 and complete replay placement must PASS')})
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
    grading_outcomes = {k:v for k,v in outcomes.items()
                        if not (teacher_diagnostic and k == 'performance')}
    verdict = ('FAIL' if 'FAIL' in grading_outcomes.values() else
               'PASS' if all(v == 'PASS' for v in grading_outcomes.values()) else 'INCONCLUSIVE')
    final = {'schema': 1, 'run_id': args.run_id, 'verdict': verdict, 'outcomes': outcomes,
             'engine_epoch': context.get('engine_epoch'),
             'final_route': context.get('route'), 'final_command_seq': context.get('command_seq'),
             'observer_during_performance': False if placement['verdict'] == 'PASS' else None,
             'free_running_evidence_verdict':result.get('verdict'),'launch_config_diagnostic':launch['verdict'],
             'files': ['launch-config.json','teacher-forced.json','data-invariant.json','correctness.json', 'placement.json', 'performance.json', 'final-status.json']}
    if teacher_diagnostic:
        final.update(scope='teacher-row-diagnostic', w1_eligible=False,
                     excluded=['W1','leak','lifetime','canary'],
                     placement_basis='unchanged raw graph-only receipt; no engine offer ledger')
    write(out / 'clients-grade.json', final)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-diagnostic', action='store_true',
                        help='stop after teacher + raw placement evidence; never run W1')
    parser.add_argument('--url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--api-source', required=True, type=Path)
    parser.add_argument('--correctness-workload', required=True, type=Path)
    parser.add_argument('--teacher-workload',required=True,type=Path)
    parser.add_argument('--teacher-baseline',required=True,type=Path)
    parser.add_argument('--snapshot-max-bytes',type=int,default=2*1024**3)
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
