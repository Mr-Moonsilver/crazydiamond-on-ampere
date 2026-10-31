#!/usr/bin/env python3
"""Source-joined native W2: fixed solo baselines, fresh and append windows.

The frozen q18a workload is unchanged. Each native request has its own route
and generation-bound window because a terminal native request returns FULL.
No observer window spans the long performance run. Raw journal loss remains
retained beside the independent source-joined measurement census.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import e3_decode_correctness as streams
import e3_overlap_performance as shared
import e3_overlap_grade as legacy_grade
from e3_native_runtime_client import NativeRuntimeClient
from e3_native_qualification_driver import exception_details
from native_measurement import select_window_evidence
from native_overlap_metrics import clock_identity, throughput, target_marker


def native_window(record, family, epoch):
    """Recheck the actual source packet, not its cached COMPLETE boolean."""
    missing = []
    begin, end = record.get('begin'),record.get('end')
    if type(begin) is not dict or type(end) is not dict:
        return {'complete':False,'reasons':['native-window-packet-missing']}
    start,stop = begin.get('measure_start_ns'),end.get('measure_end_ns')
    raw, snapshot = end.get('raw_journal_evidence'),begin.get('snapshot')
    view,binding = select_window_evidence(raw,snapshot,start,stop)
    if binding.get('valid') is not True:
        missing.extend(binding.get('reasons') or ['native-source-binding'])
    if (begin.get('engine_epoch') != epoch or end.get('engine_epoch') != epoch
            or type(start) is not int or type(stop) is not int or stop <= start):
        missing.append('native-window-identity')
    expected = clock_identity()
    for name in ('begin','end'):
        proof = (end.get('clock') or {}).get(name) or {}
        if (proof.get('proved') is not True or expected.get('proved') is not True
                or proof.get('boot_id') != expected.get('boot_id')
                or proof.get('time_namespace') != expected.get('time_namespace')):
            missing.append('client-engine-clock:'+name)
    bracket = record.get('client_bracket') or {}
    if not (type(bracket.get('start_ns')) is int and type(bracket.get('end_ns')) is int
            and type(start) is int and type(stop) is int
            and bracket['start_ns'] <= start < stop <= bracket['end_ns']):
        missing.append('client-engine-bracket')
    measurement = raw.get('measurement') if type(raw) is dict else None
    groups = measurement.get('groups') if type(measurement) is dict else None
    if (type(groups) is not list or len(groups)!=1 or groups[0].get('family') != family
            or groups[0].get('complete') is not True):
        missing.append('native-request-family-or-coverage')
        group = {}
    else:
        group = groups[0]
    useful,wall = group.get('useful_tokens'),group.get('wall_s')
    rate = throughput(useful,wall)
    if rate is None:
        missing.append('native-useful-prefill-rate')
    if measurement is None or measurement.get('serialized_transport') is not False:
        missing.append('serialized-or-unproven-native-transport')
    return {'complete':not missing,'reasons':sorted(set(missing)),
            'binding':binding,'group':group,'prefill_tokens_per_s':rate,
            'headline_basis':'first accepted native work .. final first-queried STATE_WRITE observation',
            'raw_journal_retained':True,'view':view}


def _stream_ok(state):
    return (state.first_advance_ns is not None and state.finish_at_ns is not None
            and state.error is None and state.malformed is None and state.done_marker is True)


def first_advance_read_timeout(cfg):
    """Apply only the declared native28 client wait, never a timing allowance."""
    frozen = cfg['timeouts']['read_s']
    override = os.environ.get('Q18A_FIRST_ADVANCE_READ_TIMEOUT_S')
    if override is None:
        return frozen
    if frozen != 180.0 or override != '1200':
        raise ValueError('first-advance-read-timeout-declaration-invalid')
    return 1200.0


def gates_allow_run(gates, gate_path):
    expected={'startup','row1','rule4_controls','rule4_candidate','row4','w1_controls'}
    outcomes=gates.get('outcomes')
    if type(outcomes) is not dict or set(outcomes) != expected:
        return False
    if all(value == 'PASS' for value in outcomes.values()):
        return gates.get('rule4_control_continuation') in (None, False)
    if (gates.get('rule4_control_continuation') is not True
            or any(outcomes[name] != 'PASS'
                   for name in expected-{'rule4_controls','rule4_candidate'})
            or any(outcomes[name] not in ('INCONCLUSIVE', 'FAIL')
                   for name in ('rule4_controls','rule4_candidate'))):
        return False
    path=Path(gate_path).with_name('rule4-continuation.json')
    try:
        data=path.read_bytes()
        receipt=json.loads(data)
    except (OSError, ValueError):
        return False
    raw = {name: outcomes[name] for name in ('rule4_controls', 'rule4_candidate')}
    required = {'control-hard-cap-inconclusive': 'INCONCLUSIVE',
                'split-comparison-fail': 'FAIL'}.get(receipt.get('mode'))
    return (required is not None and all(value == required for value in raw.values())
            and hashlib.sha256(data).hexdigest() == gates.get('rule4_continuation_sha256')
            and receipt.get('eligible') is True
            and receipt.get('raw_verdicts') == raw)


def prefill(cfg, client, url, model, lane, family, prompts, salt, *, decode_states=None):
    native = lane != 'prefill_full'
    phase = 'w2-overlap' if lane=='overlap' else 'split-prefill-solo'
    record = {'lane':lane,'family':family,'complete':False,'native':native}
    state = None
    sampler = shared.CountSampler(decode_states) if decode_states else None
    try:
        if native:
            record['prepare'] = client.prepare('split',phase)
        else:
            record['prepare'] = client.prepare('full','full-prefill-solo')
        if sampler:
            sampler.start()
            sampler.snapshot()
        local_start = time.monotonic_ns()
        if native:
            record['begin'] = client.call('prefill_begin',phase=phase)
        spec = cfg['prefill'][family]
        state,workload = shared._send_prefill(url,model,cfg,lane,0 if family=='fresh' else 1,
            prompts[family],spec['generation_budget'],spec['seed'],salt)
        read_s = cfg['timeouts']['read_s']
        record['first_advance_read_timeout_s'] = read_s
        first = shared._wait_first_advance(state,read_s)
        record['client_finish'] = shared._finish_or_cancel([state],workload)
        if sampler:
            sampler.snapshot()
        if native:
            record['end'] = client.call('prefill_end')
        local_end = time.monotonic_ns()
        refusal = (record.get('end') or {}).get('refusal') if native else None
        record['client_bracket'] = {'start_ns':local_start,'end_ns':local_end}
        record['request'] = {'prompt_sha256':prompts[family+'_sha256'],
            'prompt_tokens_declared':prompts[family+'_tokens'],
            'prompt_tokens_observed':state.prompt_tokens,
            'shared_prefix_tokens':0 if family=='fresh' else prompts['shared_prefix_tokens'],
            'request_sha256':state.request_sha256,'send_at_ns':state.send_at_ns,
            'first_at_ns':first,'finish_at_ns':state.finish_at_ns,
            'ttft_s':(first-state.send_at_ns)/1e9 if first and state.send_at_ns else None,
            'output_completion':state.count(),'error':state.error,'malformed':state.malformed}
        if type(refusal) is dict:
            # Typed engine refusal (native37): the request waited unallocated on
            # the armed split route; name it instead of a generic client failure.
            raise ValueError(str(refusal.get('reason') or 'native-admission-refused:untyped'))
        if not _stream_ok(state) or record['client_finish'].get('timed_out'):
            raise ValueError('prefill-client-completion-unproved')
        if (type(state.prompt_tokens) is not int or state.prompt_tokens < 1
                or state.prompt_tokens != state.prompt_token_ids_len):
            raise ValueError('prefill-prompt-token-geometry-unproven')
        if native:
            proof = native_window(record,family,client.epoch)
            record['native_proof'] = {k:v for k,v in proof.items() if k!='view'}
            record['complete'] = proof['complete']
            if sampler:
                group = proof['group']
                start,end = group.get('start_ns'),group.get('end_ns')
                if type(start) is int and type(end) is int and end>start:
                    record['decode_window'] = shared._stream_window(decode_states,sampler,start,end)
                    reasons = record.setdefault('reasons',[])
                    for row in record['decode_window']['streams']:
                        if row.get('completion_delta',0)<1 or row.get('error_before_window_end'):
                            reasons.append('decode-not-advancing:'+str(row.get('stream')))
                    if any(s.finish_at_ns is not None and s.finish_at_ns < end for s in decode_states):
                        reasons.append('standing-decoder-finished-before-native-headline-end')
                    if reasons:
                        record['complete'] = False
                else:
                    record['complete'] = False
                    record['reasons']=['native-headline-window-missing']
        else:
            # Actual serving baseline, client request -> first generated token.
            # The native matched denominator below is separately named and is
            # never replaced by this end-to-end TTFT rate.
            useful = state.prompt_tokens if family == 'fresh' else None
            ttft = record['request']['ttft_s']
            record['serving_prefill_tokens_per_s'] = throughput(useful,ttft)
            record['complete'] = type(ttft) in (int,float) and ttft>0
            record['serving_rate_basis'] = ('observed fresh prompt tokens / client TTFT' if family=='fresh'
                else 'append TTFT only; prefix reuse is not inferred from nominal suffix size')
        record['drain'] = client.pause()
        record['return_full'] = client.leave()
    except Exception as exc:
        record['error'] = exception_details(exc,client.control)
        record['complete'] = False
        if native:
            record['cleanup'] = return_full_after_error(client)
    finally:
        if sampler:
            sampler.close()
        if state is not None:
            streams._stop_streams({},[state],shared._workload_view(cfg,streams=1))
    return record


def return_full_after_error(client):
    """Best-effort return to FULL after a native lane error (native37 H1).

    pause and leave_full each run in their own try: a pause whose drain proof
    fails must not skip leave_full.  The lane's own error, completeness and the
    run's first_error are never changed here; only this record is added.
    """
    cleanup = {'attempted': True, 'basis': 'best-effort pause then leave_full after a native lane error'}
    for key,call in (('pause',client.pause),('leave_full',client.leave)):
        try:
            cleanup[key] = call()
            cleanup[key+'_error'] = None
        except Exception as exc:  # noqa: BLE001 - recorded, never raised
            cleanup[key] = None
            cleanup[key+'_error'] = exception_details(exc,client.control)
    cleanup['route_after'] = (cleanup['leave_full'] or {}).get('route')
    return cleanup


def grade(record):
    """Report the fixed 80/80 target over complete rows, without survivor selection.

    The operator's target line is descriptive. Exactness, numerical, placement,
    ownership and memory gates remain separate hard gates in the same queue.
    """
    missing,failures = [],[]
    lanes = record.get('lanes') or {}
    proofs = {}
    for lane in shared.LANES:
        rows = lanes.get(lane)
        if type(rows) is not dict or set(rows) != {'fresh','append'}:
            missing.append(lane+':family-coverage')
            continue
        for family,row in rows.items():
            if row.get('complete') is not True:
                missing.append(lane+':'+family+':incomplete')
            if lane!='prefill_full':
                proof = native_window(row,family,record.get('engine_epoch'))
                proofs[(lane,family)] = proof
                missing.extend(lane+':'+family+':'+r for r in proof['reasons'])
                if lane == 'overlap':
                    window = row.get('decode_window') or {}
                    group = proof.get('group') or {}
                    if (window.get('start_ns') != group.get('start_ns')
                            or window.get('end_ns') != group.get('end_ns')):
                        missing.append('decode-window-does-not-match-native-headline:'+family)
    w1 = record.get('w1') or {}
    if (w1.get('summary') or {}).get('verdict')!='COMPLETE':
        missing.append('w1-fixed-repeats-incomplete')
    denominator = legacy_grade._w1_route(w1,'full')
    matched = (lanes.get('prefill_split') or {}).get('fresh') or {}
    overlap = (lanes.get('overlap') or {}).get('fresh') or {}
    full = (lanes.get('prefill_full') or {}).get('fresh') or {}
    split_rate = proofs.get(('prefill_split','fresh'),{}).get('prefill_tokens_per_s')
    candidate_rate = proofs.get(('overlap','fresh'),{}).get('prefill_tokens_per_s')
    decode = overlap.get('decode_window') or {}
    baseline = denominator.get('decode_tokens_per_s')
    dr = decode.get('aggregate_tokens_per_s')
    shares = {'decode':dr/baseline if type(dr) in (int,float) and type(baseline) in (int,float) and baseline>0 else None,
              'prefill':candidate_rate/split_rate if type(candidate_rate) in (int,float) and type(split_rate) in (int,float) and split_rate>0 else None}
    marker = target_marker(decode_share_value=shares['decode'],prefill_share_value=shares['prefill'],
        label='s5-overlap-80-80',decode_denominator='actual full-GPU W1 fixed repeats',
        prefill_denominator='matched split solo fresh native window')
    rates,base = decode.get('stream_rates_tokens_per_s'),denominator.get('stream_rates_tokens_per_s')
    per_stream = [a/b for a,b in zip(rates,base)] if (type(rates) is list and type(base) is list
        and len(rates)==len(base)==8 and all(type(x) in (int,float) and x>0 for x in base)
        and all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in rates)
        and all(math.isfinite(x) for x in base)) else None
    if per_stream is None:
        missing.append('eight-stream-decode-share-unproved')
    if not missing and marker['status'] not in ('PASS','SHORTFALL'):
        missing.append('80-80-marker-incomplete')
    if record.get('first_error'):
        missing.append('declared-run-error-retained')
    serving_rate = full.get('serving_prefill_tokens_per_s')
    return {'schema':'s5-native-performance-grade/1','verdict':'FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS',
        'failures':failures,'missing':sorted(set(missing)),'target':marker,
        'decode_shares_by_stream':per_stream,
        'target_met':None if missing or failures else marker['status']=='PASS',
        'target_is_gate':False,
        'prefill_share_full_serving_ttft_basis':candidate_rate/serving_rate if type(candidate_rate) in (int,float) and type(serving_rate) in (int,float) and serving_rate>0 else None,
        'ttft_s':{lane:{family:(row.get('request') or {}).get('ttft_s') for family,row in rows.items()}
                  for lane,rows in lanes.items() if type(rows) is dict},
        'all_declared_repeats_retained':True,'filtered_invalid':False,'retry_selected':False,'picked':False,
        'scope':'measured native decode-during-prefill and prefill speed/TTFT; release still needs lifetime/memory/canary'}


def run(cfg,client,url,model,w1):
    # Bound conversion of engine/host monotonic observation times to physical
    # memory sampler query brackets. Clock steps invalidate the conversion.
    mono_before=time.monotonic_ns();wall=time.time();mono_after=time.monotonic_ns()
    result = {'schema':'s5-native-performance/1','run_id':client.run_id,'engine_epoch':client.epoch,
              'manifest_sha256':cfg['manifest_sha256'],'lanes':{},'w1':w1,
              'wall_anchor_s':wall,'wall_anchor_ns':(mono_before+mono_after)//2,
              'wall_anchor_uncertainty_ns':mono_after-mono_before}
    prompts = shared._prefill_prompts(cfg)
    for lane in shared.LANES:
        result['lanes'][lane] = {}
        standing=[]
        workload = shared._workload_view(cfg)
        try:
            if lane=='overlap':
                client.prepare('full','full-prefill-solo')
                cohort=streams.build_cohort(workload)
                bodies=[shared._request(model,row['prompt'],workload,shared._cache_salt(client.run_id,lane,i))
                        for i,row in enumerate(cohort)]
                standing=streams._start_streams(url,model,bodies,cohort,workload)
                if streams._wait_open(standing,workload).get('complete') is not True:
                    raise ValueError('w2-standing-cohort-unopened')
            salt=shared._cache_salt(client.run_id,lane,0)
            for family in ('fresh','append'):
                row=prefill(cfg,client,url,model,lane,family,prompts,salt,decode_states=standing)
                result['lanes'][lane][family]=row
                if row.get('complete') is not True:
                    raise ValueError('native-lane-refused:'+lane+':'+family)
        except Exception as exc:
            result['first_error']=exception_details(exc,client.control)
            break
        finally:
            if standing:
                streams._stop_streams({},standing,workload)
    result['engine_epoch']=client.epoch
    result['wall_end_s']=time.time();result['wall_end_ns']=time.monotonic_ns()
    result['grade']=grade(result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--url',required=True);p.add_argument('--model',required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--epoch',type=int,required=True)
    p.add_argument('--workload',required=True);p.add_argument('--w1',required=True);p.add_argument('--gates',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if Path(a.out).exists():return 2
    result={}
    try:
        gates=json.loads(Path(a.gates).read_text())
        if not gates_allow_run(gates, a.gates):
            raise ValueError('same-queue-qualification-gates-not-pass')
        package=shared._package_gate(os.environ)
        if package.get('ok') is not True:
            raise ValueError('reviewed-package-readiness-not-pass')
        data=json.loads(Path(a.workload).read_text())
        problems,cfg=shared.validate_manifest(data)
        if problems or cfg['repeats']!=1:raise ValueError('frozen-native-workload-invalid')
        effective_read_s = first_advance_read_timeout(cfg)
        cfg = dict(cfg, timeouts=dict(cfg['timeouts'], read_s=effective_read_s))
        result=run(cfg,NativeRuntimeClient(a.url,a.run_id,epoch=a.epoch),a.url,a.model,json.loads(Path(a.w1).read_text()))
        result['first_advance_read_timeout_s'] = effective_read_s
    except Exception as exc:
        result={'schema':'s5-native-performance/1','grade':{'verdict':'INCONCLUSIVE','missing':['run-refused'],
                 'failures':[]},'error':exception_details(exc)}
    with open(a.out,'x') as handle:json.dump(result,handle,sort_keys=True)
    verdict=result['grade']['verdict'];print('NATIVE_PERFORMANCE '+verdict)
    return {'PASS':0,'FAIL':1}.get(verdict,2)


if __name__=='__main__':raise SystemExit(main())
