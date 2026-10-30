#!/usr/bin/env python3
"""Native cancellation/drain/reuse proof, beside unchanged leak/canary gates.

A real admitted request is disconnected once while the engine owns native work.
Only that request generation's certified typed all-rank close can qualify it.
There is no general cancel RPC and no active=False -> request_closed inference.
The long pressure/leak and real-route canary are separate queue artifacts.
"""
import argparse
import json
from pathlib import Path
import time

import e3_decode_correctness as streams
import e3_overlap_performance as shared
import e3_native_performance as performance
from e3_native_runtime_client import NativeRuntimeClient
from e3_native_qualification_driver import exception_details


def grade_close(before,after,epoch):
    failures,missing=[],[]
    expected=before.get('native_work_identity') or {}
    close=after.get('last_native_closure') or {}
    def need(name,ok):
        if not ok:missing.append(name)
    need('observed-owned-native-request',before.get('active') is True and bool(expected)
         and (before.get('owned_work') or {}).get('chunk') is True)
    need('certified-close',close.get('schema')=='s5-native-certified-close/1' and close.get('complete') is True)
    for key in ('request_uid_sha256','request_generation'):
        need('same-request:'+key,expected.get(key) is not None and expected.get(key)==close.get(key))
    need('same-engine',before.get('engine_epoch')==epoch==after.get('engine_epoch')==close.get('engine_epoch'))
    need('close-after-observation',type(close.get('observed_ns')) is int
         and type(before.get('observed_clock_ns')) is int and close['observed_ns']>=before['observed_clock_ns'])
    need('actual-abort-source',close.get('reason') in ('abort','cancel','disconnect'))
    need('no-owned-work-after-close',after.get('active') is False and after.get('retained_row') is None)
    need('scheduler-source-leases',type(close.get('source_leases')) is dict
         and close['source_leases'].get('clean') is True and close['source_leases'].get('held')==0
         and close['source_leases'].get('pending_free')==0)
    need('pp-consumers',close.get('pp_receipts_pending') is False)
    rows=close.get('rows')
    need('typed-all-rank-close',type(rows) is list and len(rows)==4
         and {r.get('worker_rank') for r in rows if type(r) is dict}==set(range(4)))
    work=0
    for row in rows if type(rows) is list else []:
        if type(row) is not dict:continue
        rank=row.get('worker_rank');p='rank%s:'%rank
        need(p+'typed',row.get('typed') in ('CancelAck','ReleaseAck'))
        need(p+'closed',row.get('closed') is True and row.get('released') is True and row.get('request_closed') is True)
        need(p+'transaction',row.get('transaction')==close.get('transaction')
             and row.get('request_generation')==close.get('request_generation'))
        report=row.get('report') or {}
        need(p+'same-queried-report',report.get('worker_rank')==rank
             and report.get('transaction')==close.get('transaction')
             and report.get('request_generation')==close.get('request_generation')
             and report.get('state_epoch')==row.get('state_epoch')
             and type(row.get('completed_sequence')) is int
             and row.get('completed_sequence')==report.get('completed_sequence'))
        need(p+'no-pending-operations',report.get('pending_op_seqs')==[])
        completed=report.get('completed_op_seqs')
        proofs=report.get('proofs') or []
        if type(completed) is list:
            work+=len(completed)
            proved={r.get('op_seq') for r in proofs if type(r) is dict and r.get('kind')=='gpu_event'}
            need(p+'queried-gpu-completions',set(completed)==proved)
        else:missing.append(p+'operation-history')
    need('cancellation-exercised-gpu-work',work>0)
    return {'verdict':'FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS',
            'failures':failures,'missing':missing,'completed_operation_count':work,
            'scope':'same native request generation, actual typed close and queried reports, source/PP ownership'}


def _same_workers(before,after):
    a={r['worker_rank']:r for r in before.get('workers') or []}
    b={r['worker_rank']:r for r in after.get('workers') or []}
    return set(a)==set(b)==set(range(4)) and all(
        a[r].get(k) is not None and a[r].get(k)==b[r].get(k)
        for r in a for k in ('worker_incarnation','storage_epoch'))


def run(client,cfg,url,model):
    record={'schema':'s5-native-lifetime/1','run_id':client.run_id,'engine_epoch':client.epoch,
            'failures':[],'missing':[],'scope':'native cancel/drain/reuse; separate unchanged pressure/leak/canary gates'}
    states=[];workload=shared._workload_view(cfg,streams=1,generation_budget=1)
    try:
        record['prepare']=client.prepare('full','full-prefill-solo')
        record['identity_before']=client.call('identity')
        client.enter('split-prefill-solo');client.resume()
        prompts=shared._prefill_prompts(cfg)
        body=shared._request(model,prompts['fresh'],workload,shared._cache_salt(client.run_id,'lifetime-cancel',0))
        states=streams._start_streams(url,model,[body],[{'prompt_sha256':prompts['fresh_sha256'],'prompt_chars':len(prompts['fresh'])}],workload)
        deadline=time.monotonic()+cfg['timeouts']['read_s']
        before=None
        while time.monotonic()<deadline:
            snapshot=client.call('status')
            if (snapshot.get('active') is True and (snapshot.get('owned_work') or {}).get('chunk') is True
                    and snapshot.get('native_work_identity') and states[0].response is not None
                    and states[0].first_advance_ns is None):
                before=snapshot;break
            if states[0].first_advance_ns is not None or states[0].error is not None:break
            time.sleep(.02)
        record['before_disconnect']=before
        if before is None:raise ValueError('native-inflight-cancellation-coverage-unobserved')
        record['client_cancel_alive']=streams._stop_streams({},states,workload)
        record['disconnect_observed_ns']=time.monotonic_ns()
        deadline=time.monotonic()+cfg['timeouts']['drain_s']
        after=None
        while time.monotonic()<deadline:
            after=client.call('status')
            if after.get('active') is False and after.get('retained_row') is None:break
            time.sleep(.02)
        record['after_disconnect']=after
        record['cancel_grade']=grade_close(before,after,client.epoch)
        record['missing'].extend(record['cancel_grade']['missing'])
        record['failures'].extend(record['cancel_grade']['failures'])
        record['drain']=client.pause()
        record['return_full']=client.leave()
        # Exactly one new admitted native request after cancellation. No retry
        # of the cancelled request and no replacement on a coverage refusal.
        reuse=performance.prefill(cfg,client,url,model,'prefill_split','fresh',prompts,
                shared._cache_salt(client.run_id,'lifetime-reuse',0))
        record['reuse']=reuse
        if reuse.get('complete') is not True:record['missing'].append('native-reuse-incomplete')
        record['identity_after']=client.call('identity')
        if not _same_workers(record['identity_before'],record['identity_after']):
            record['failures'].append('persistent-worker-or-storage-changed')
        record['final']=client.call('status')
        if (record['final'].get('route')!='full' or record['final'].get('active') is not False
                or (record['final'].get('source_leases') or {}).get('clean') is not True):
            record['missing'].append('final-owned-source-or-route')
        client.resume()
    except Exception as exc:
        record['error']=exception_details(exc,client.control)
        record['missing'].append('native-lifetime-refused:'+record['error']['error_message'])
    finally:
        if states:streams._stop_streams({},states,workload)
    record['verdict']='FAIL' if record['failures'] else 'INCONCLUSIVE' if record['missing'] else 'PASS'
    return record


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('url','model','run-id','workload','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--epoch',type=int,required=True)
    a=p.parse_args(argv)
    if Path(a.out).exists():return 2
    try:
        problems,cfg=shared.validate_manifest(json.loads(Path(a.workload).read_text()))
        if problems:raise ValueError('native-lifetime-workload-invalid')
        result=run(NativeRuntimeClient(a.url,a.run_id,epoch=a.epoch),cfg,a.url,a.model)
    except Exception as exc:
        result={'verdict':'INCONCLUSIVE','missing':['lifetime-inputs'],'failures':[],'error':exception_details(exc)}
    with open(a.out,'x') as handle:json.dump(result,handle,sort_keys=True)
    print('NATIVE_LIFETIME '+result['verdict'])
    return {'PASS':0,'FAIL':1}.get(result['verdict'],2)


if __name__=='__main__':raise SystemExit(main())
