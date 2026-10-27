#!/usr/bin/env python3
"""CUPTI external-ID bridge: deferred joins, no per-offer drains.

VM only. Raw graph placement stays unchanged. Synthetic source-declared
families do not qualify engine call sites. All fault mutations are labelled.
"""
import argparse
import copy
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import threading
import traceback
import vmid_support as s
from vmid_job2 import GraphBinding
from vmid_job6 import HostQueries
from vmid_job8 import FAMILIES
from vmid_job5 import LIMITS, occupancy

MIX=dict(kernel=1386,memcpy=26,memset=7)

def account_module():
    for p in (Path(__file__).resolve().parent/'e3_builder',Path(__file__).resolve().parents[1]/'patches/s5_engine'):
        if (p/'activity_accounting.py').is_file():
            sys.path.insert(0,str(p));break
    import activity_accounting
    return activity_accounting


def case_grade(r):
    f=list(r.get('failures',[]));m=list(r.get('missing',[]))
    account_module()
    import engine_offers as eo
    raw=r.get('raw_placement');engine=r.get('engine_accounting')
    composed=eo.compose_placement(raw,engine)
    mode=r.get('mode')
    for key in ('cleanup_ok','serialized_stable','output_exact','raw_unchanged'):
        if r.get(key) is False:f.append(key)
        elif r.get(key) is not True:m.append(key)
    if mode=='positive':
        if composed.get('verdict')=='FAIL':f.extend(composed.get('reasons',[]))
        elif composed.get('verdict')!='PASS':m.extend(composed.get('reasons',[]) or ['composition_missing'])
        if not isinstance(raw,dict) or raw.get('verdict')!='INCONCLUSIVE':m.append('raw_eager_guard_missing')
        if not isinstance(raw,dict) or len(raw.get('replay',{}).get('graph_slots',[]))!=7:m.append('seven_graph_bank')
        if not engine or engine.get('eager_green_records',0)<1:m.append('nonempty_eager_join')
        if r.get('graph_records')!=56760 or r.get('replays')!=40:m.append('full8_graph_exposure')
        for name,row in r.get('injections',{}).items():
            if not row.get('rejected'):f.append('mutation_accepted:'+name)
        if set(r.get('injections',{}))!={'wrong_window','wrong_rank','one_missing_eager','raw_fail'}:m.append('mutation_coverage')
    elif mode in ('unoffered','primary'):
        wanted='unoffered_eager_activity' if mode=='unoffered' else 'wrong_context'
        if not engine or wanted not in engine.get('failures',[]):f.append('negative_not_rejected:'+wanted)
        if composed.get('verdict')!='FAIL':f.append('composition_lost_failure')
    else:m.append('case_mode')
    if engine and engine.get('native_summary',{}).get('errors',0):m.append('native_errors')
    return s.verdict(f,m)


def grade(r):
    f=list(r.get('failures',[]));m=list(r.get('missing',[]));cases={}
    if [x.get('mode') for x in r.get('cases',[])]!=['positive','unoffered','primary']:m.append('case_coverage')
    for row in r.get('cases',[]):
        g=case_grade(row);name=row['mode'];cases[name]=g
        f.extend(name+':'+x for x in g['failures']);m.extend(name+':'+x for x in g['missing'])
    for name in ('job1','job2b_flat'):
        g=r.get('regressions',{}).get(name,{})
        if g.get('verdict')=='FAIL':f.append(name+':regression')
        elif g.get('verdict')!='PASS':m.append(name+':missing')
    out=s.verdict(f,m);out['cases']=cases;return out


def case(torch,ro,gc,native,receipt,out,mode):
    from types import SimpleNamespace as NS
    import engine_offers as eo
    from activity_accounting import CorrelationBridge
    r=dict(mode=mode,replays=0,failures=[],missing=[],injections={})
    session=split=ledger=None;owners=[]
    try:
        binding=ro.ReplayBinding(str(native/'flash_s5_replay_observer.so'),receipt['files']['flash_s5_replay_observer.so'],receipt['files']['flash_s5_replay_observer.cpp'],s.CUPTI,
            cupti_library_path='/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13')
        session=ro.ObserverSession(binding,ro.ObserverLimits(max_receipt_bytes=64*1024**2));session.start()
        bridge=CorrelationBridge(session,epoch=1,generation=1)
        split=gc.open_split(0,env={'FLASH_S5_DECODE_SMS':'44'},torch_module=torch,binding=gc.DriverBinding(str(native/'flash_s5_green.so')))
        split.retain_allocator_context();ordinary=torch.cuda.Stream();stream=split.decode_stream
        host=HostQueries(native/'vmid_context_probe.so');identity=host.stream(stream)
        context=identity['green_cupti_context_id'];streamid=identity['cupti_stream_id']
        src=torch.full((1024,),7,dtype=torch.int32,pin_memory=True);dst=torch.empty_like(src,pin_memory=True)
        a,b,z,e,f=[torch.zeros((1024,),dtype=torch.int32,device='cuda') for _ in range(5)]
        scratch=torch.empty_like(src,pin_memory=True)
        torch.cuda.synchronize();gl=GraphBinding(native/'vmid_graph_probe.so')
        bank=NS(captured=True,pairs=[],handles={})
        bank.manager=NS(graphs={},_flash_s5_capture_owner=split,_flash_s5_capture_context=lambda:None)
        descs=[];before={}
        for i in range(7 if mode=='positive' else 1):
            own,graph=C.c_uint64(),C.c_uint64()
            s.checked(gl.lib.vmg_create(int(stream.cuda_stream),a.data_ptr(),b.data_ptr(),z.data_ptr(),src.data_ptr(),dst.data_ptr(),1024,1386,26,7,0,C.byref(own),C.byref(graph)),'create')
            owners.append((own.value,graph.value));descriptor=object();local=object()
            descs.append(dict(num_tokens=8*(i+1),num_reqs=8,uniform_token_count=1,num_ubatches=1))
            bank.pairs.append((descriptor,local));bank.handles[local]=NS(local=local,bank=bank,destroyed=False)
            bank.manager.graphs[descriptor]=NS(raw_cuda_graph=lambda g=graph.value:g)
        probe=NS(bank=bank,router=NS(storage_epoch=1,route_generation=1),split=split,
                 storage_epoch=1,route_generation=1,route='split',interval='active')
        session.bind_bank(probe)
        lib=binding._lib
        for i,(_,graph) in enumerate(owners):before[i]=s.graph_bytes(lib,graph)
        eager=C.CDLL(str(native/'vmid_eager_probe.so'))
        eager.vm8_eager_submit.argtypes=[C.c_uint64]*5+[C.c_int]*2;eager.vm8_eager_submit.restype=C.c_int
        def submit(family,n=1,target=stream):
            s.checked(eager.vm8_eager_submit(int(target.cuda_stream),e.data_ptr(),f.data_ptr(),src.data_ptr(),scratch.data_ptr(),family,n),'eager')
        for family in range(1,8):submit(family)
        submit(3,target=ordinary);s.checked(gl.lib.vmg_replay(owners[0][0]),'warm');torch.cuda.synchronize()
        roles={k:dict(context_id=context,stream_id=streamid) for k in ('main','pp','output_copy')}
        session.begin_window(route_generation=1,descriptor=bank.pairs[0][0])
        start=bridge.window()
        wb=eo.bind_window(run_id='vm-job11-'+mode,window_id='short-placement',worker='rank0',rank=0,
            worker_incarnation='vm-worker-1',engine_incarnation='synthetic-no-engine',engine_epoch=1,
            route_generation=1,storage_epoch=1,green_context_id=context,native_start_ns=start['start_ns'],native_end_ns=None,
            observer_identity=binding.identity(),helper_identity=dict(native_static_schema=ro.NATIVE_SCHEMA_STATIC,
                native_window_schema=ro.NATIVE_SCHEMA_WINDOW,compiled_api=130001,runtime_api=130001),
            declared_target='graph:0',descriptor_labels={'graph:'+str(i):eo.descriptor_key(d) for i,d in enumerate(descs)},
            router_census=None,bank_identity_sha256=eo.bank_identity(session._bank_meta),stream_roles=roles,pp_stage=0)
        ledger=eo.EngineOfferLedger(bridge,worker='rank0',rank=0,epoch=1,generation=1,route_generation=1,
            green_context_id=context,identities=roles,pp_stage=0,env={eo.ENABLE_ENV:'1'},binding=wb)
        ledger.activate()
        n=40 if mode=='positive' else 1
        for _ in range(n):
            with ledger.step('execute'):
                with ledger.offer_scope('request_state_table','prep_state'):submit(1,3)
                with ledger.offer_scope('input_attention','prepare_inputs'):submit(4)
                with ledger.offer_scope('input_attention','prepare_attn',branch='ordinary'):submit(5)
                with ledger.offer_scope('forward_dispatch','graph_replay',descriptor=descs[0],census=MIX):
                    s.checked(gl.lib.vmg_replay(owners[0][0]),'replay')
            with ledger.step('sample'):
                with ledger.offer_scope('logits','compute_full'):submit(2,3)
                with ledger.offer_scope('sampler','completion_counts'):submit(7)
                with ledger.offer_scope('output_copy','async_output'):submit(6)
        if mode=='unoffered':submit(3)
        elif mode=='primary':
            with ledger.step('sample'):
                with ledger.offer_scope('logits','compute_full'):submit(3,target=ordinary)
        stream.synchronize();ordinary.synchronize();ledger.deactivate()
        proof=ro.DrainedProof(worker_rank=0,worker_incarnation='vm-worker-1',engine_incarnation='synthetic-no-engine',
            route_generation=1,storage_epoch=1,relevant_consumers=2,controller_drained=True)
        raw=ro._receipt_dict(session.end_window(proof));raw_before=json.dumps(raw,sort_keys=True)
        ledger.finish_window({'total':n,'graphs':{'graph:0':n}})
        engine=ledger.seal();composed=eo.compose_placement(raw,engine)
        r.update(raw_placement=raw,engine_accounting=engine,composed=composed,replays=n,
            graph_records=engine.get('graph_records'),raw_unchanged=json.dumps(raw,sort_keys=True)==raw_before,
            observer_host_reserved_bytes=session._native_reserved,correlation=bridge.report(),
            scope='actual engine ledger and native composer on synthetic source-declared work; no vLLM/model/PP call-site qualification')
        if mode=='positive':
            for name in ('wrong_window','wrong_rank','one_missing_eager','raw_fail'):
                rr=copy.deepcopy(raw);ee=copy.deepcopy(engine)
                if name=='wrong_window':ee['binding']['native_start_ns']+=1
                elif name=='wrong_rank':ee['binding']['rank']+=1
                elif name=='one_missing_eager':ee['eager_green_records']-=1
                else:rr['verdict']='FAIL'
                g=eo.compose_placement(rr,ee)
                r['injections'][name]=dict(rejected=g['verdict']!='PASS',verdict=g['verdict'],reasons=g['reasons'],basis='checker mutation of copied receipts')
        r['serialized_stable']=all(before[i]==s.graph_bytes(lib,g) for i,(_,g) in enumerate(owners))
        r['serialized_bytes']={str(i):len(v) for i,v in before.items()}
        r['output_exact']=bool(torch.all(dst==1393).item() and torch.all(z.cpu()==0).item())
        r['allocator_peak_reserved']=torch.cuda.max_memory_reserved(0)
    except Exception as exc:
        r['missing'].append('case_exception');r['error_type']=type(exc).__name__
        r['error_stack']=[dict(file=Path(x.filename).name,line=x.lineno,function=x.name) for x in traceback.extract_tb(exc.__traceback__)]
    finally:
        errors=[]
        if ledger:ledger.deactivate()
        try:torch.cuda.synchronize()
        except Exception:errors.append('drain')
        try:
            if session:
                if session.state=='window':session.end_window(ro.DrainedProof(worker_rank=0,worker_incarnation='vm-worker-1',engine_incarnation='synthetic-no-engine',route_generation=1,storage_epoch=1,relevant_consumers=2,controller_drained=True))
                if session.state in ('started','bound','ended'):session.stop()
                session.close()
        except Exception:errors.append('observer_close')
        try:
            for own,_ in reversed(owners):s.checked(gl.lib.vmg_close(own),'graph_close')
            if split:split.close()
        except Exception:errors.append('native_close')
        r['cleanup_ok']=not errors;r['cleanup_errors']=errors
    r['grade']=case_grade(r);s.write(out/(mode+'.json'),r);return r


def probe(modules,native,out):
    r=dict(schema='vm<vm-id>-job11/1',image=s.IMAGE,cases=[],regressions={},failures=[],missing=[],memory_fraction=.125)
    try:
        import torch
        torch.cuda.set_per_process_memory_fraction(.125,0);torch.set_num_threads(4)
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)!=(8,6):raise RuntimeError('device')
        sys.path.insert(0,str(modules));import replay_observer as ro;import green_context as gc
        receipt=json.loads((native/'q17-helper-build.json').read_text())
        if receipt['image']!=s.IMAGE or receipt['cupti_library_sha256']!=s.CUPTI:raise RuntimeError('build_pin')
        for name,h in receipt['files'].items():
            if s.sha(native/name)!=h:raise RuntimeError('artifact_pin')
        for name in ('flash_s5_green.cu','flash_s5_replay_observer.cpp','flash_s5_graph_bytes.cpp'):
            if s.sha(modules/'native'/name)!=receipt['files'][name]:raise RuntimeError('source_pin')
        r['build']=receipt
        for mode in ('positive','unoffered','primary'):r['cases'].append(case(torch,ro,gc,native,receipt,out,mode))
        # Unmodified job1 and job2b regressions, isolated processes, same helper
        # binaries. Child-node INCONCLUSIVE remains separate from flat coverage.
        for name in ('job1','job2'):
            d=out/('regression-'+name);d.mkdir()
            cmd=[sys.executable,str(Path(__file__).with_name('vmid_'+name+'.py')),'--mode','probe','--modules',str(modules),'--native',str(native),'--out',str(d)]
            with (d/'process.log').open('x') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
            raw=json.loads((d/'probe.json').read_text())
            if name=='job1':
                import vmid_job1 as reg;g=reg.grade(raw);r['regressions']['job1']=g
            else:
                import vmid_job2 as reg;g=reg.grade(raw);r['regressions']['job2b_flat']=g['q17_flat_node_families'];r['regressions']['job2_child_scope']=g
            if rc!=g['rc']:r['missing'].append(name+':process_verdict')
    except Exception as exc:
        r['error_type']=type(exc).__name__;r['missing'].append('probe_exception')
        r['error_stack']=[dict(file=Path(x.filename).name,line=x.lineno,function=x.name) for x in traceback.extract_tb(exc.__traceback__)]
    r['grade']=grade(r);s.write(out/'probe.json',r);return r['grade']['rc']



def finalize(out,process_rc):
    try:
        raw=json.loads((out/'probe.json').read_text());g=grade(raw)
    except (OSError,ValueError,TypeError,KeyError):
        raw={};g=s.verdict([],['probe_result']);g['cases']={}
    try:m=json.loads((out/'monitor.json').read_text())
    except (OSError,ValueError):m={}
    f=list(g['failures']);missing=list(g['missing'])
    if m.get('stop_triggered') or m.get('own_peak_mib',0)>3072:f.append('resource_stop_or_cap')
    if process_rc in (139,134):f.append('native_process_crash')
    elif process_rc not in (0,1,2) or process_rc!=g['rc']:missing.append('process_verdict_mismatch')
    if not m.get('own_samples') or m.get('errors',1):missing.append('memory_sampling')
    result=s.verdict(f,missing)
    result.update(schema='vm<vm-id>-job11/1',process_rc=process_rc,memory=m,cases=g['cases'],
       regressions=raw.get('regressions',{}),scope='actual engine ledger/composer on synthetic GPU submissions; no engine call-site, model or full12 qualification')
    s.write(out/'result.json',result);print('VMID_JOB11 '+result['verdict']);return result['rc']


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('probe','monitor','finalize'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--modules',type=Path);p.add_argument('--native',type=Path);p.add_argument('--process-rc',type=int,default=2);a=p.parse_args()
    if a.mode=='monitor':return s.monitor(a.out,'flashnext-job11')
    if a.mode=='finalize':return finalize(a.out,a.process_rc)
    return probe(a.modules,a.native,a.out)
if __name__=='__main__':sys.exit(main())
