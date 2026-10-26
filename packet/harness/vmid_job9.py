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


def injections(report,offers,grader):
    """Checker mutations of retained evidence, never simulated GPU failures."""
    checks={}
    def check(name,edit):
        r=copy.deepcopy(report);o=copy.deepcopy(offers);edit(r,o)
        g=grader(r,o,epoch=1,generation=1)
        checks[name]=dict(rejected=g['verdict']!='PASS',verdict=g['verdict'],failures=g['failures'],missing=g['missing'])
    check('missing_join',lambda r,o:r['external'].clear())
    check('duplicate_join',lambda r,o:r['external'].append(dict(r['external'][0])))
    check('unknown_offer',lambda r,o:r['external'][0].update(external_id=999999))
    check('stale_generation',lambda r,o:o[0].update(generation=2))
    check('wrong_thread',lambda r,o:o[0].update(thread=999999))
    check('unclosed_offer',lambda r,o:o[0].update(closed=False))
    check('scope_corruption',lambda r,o:r['scopes'][0].update(action='pop'))
    check('lost_activity',lambda r,o:r['activities'].pop())
    check('duplicate_activity',lambda r,o:r['activities'].append(dict(r['activities'][0])))
    check('overflow',lambda r,o:r.update(overflow=1))
    check('boundary',lambda r,o:r.update(boundary_records=1))
    check('wrong_context',lambda r,o:r['activities'][0].update(context_id=999999))
    check('api_failure',lambda r,o:[a.update(return_value=1) for a in r['apis']])
    return checks


def resource_count(r):
    return r.get('window',{}).get('counts',{}).get('resource_events',0)


def case_grade(r):
    f=list(r.get('failures',[]));m=list(r.get('missing',[]));mode=r.get('mode')
    if mode not in ('positive','unoffered','primary'):m.append('case_mode')
    for key in ('cleanup_ok','serialized_stable','output_exact'):
        if r.get(key) is False:f.append(key)
        elif r.get(key) is not True:m.append(key)
    a=r.get('accounting',{});g=account_module().grade(a.get('native',{}),a.get('offers',[]),epoch=1,generation=1)
    if r.get('limits')!=LIMITS:m.append('default_limits')
    if mode=='positive':
        if r.get('pressure_graphs')!=8 or resource_count(r)<189796:m.append('startup_resource_exposure')
        f.extend(g.get('failures',[]));m.extend(g.get('missing',[]))
        if g.get('verdict')!='PASS' and not f and not m:m.append('accounting_missing')
        if not g.get('wrappers'):m.append('runtime_driver_dedup_not_observed')
        if len({o['thread'] for o in a.get('offers',[])})!=2:m.append('two_submit_threads')
        if not any(o.get('body_exception') and o.get('closed') for o in a.get('offers',[])):m.append('exception_pop_coverage')
        if len(r.get('injections',{}))!=13:m.append('refusal_coverage')
        for name,x in r.get('injections',{}).items():
            if not x.get('rejected'):f.append('mutation_accepted:'+name)
    elif mode in ('unoffered','primary'):
        required='unoffered_activity' if mode=='unoffered' else 'wrong_context'
        if required not in g.get('failures',[]):f.append('negative_not_rejected:'+required)
        if a.get('native',{}).get('complete') is not True:m.append('negative_native_incomplete')
    w=r.get('window',{});counts=w.get('counts',{});ctx=w.get('expected_context',{})
    if ctx.get('is_green') is not True or ctx.get('sm_matching',0)<1:m.append('green_identity')
    if ctx.get('contradictory') or ctx.get('sm_mismatched',0):f.append('green_contradiction')
    if any(v for k,v in w.get('flags',{}).items() if k!='unattributed_expected_context'):m.append('native_flags')
    if a.get('native',{}).get('activity_records')!=counts.get('kernel_records',-1)+counts.get('copy_records',-1):m.append('whole_window_conservation')
    if set(w.get('reasons',[]))-{'unattributed_expected_context'}:m.append('raw_graph_reason')
    if 'unattributed_expected_context' not in w.get('reasons',[]):m.append('raw_graph_guard_missing')
    for key in ('dropped_records_total','buffers_refused','drop_query_errors','parse_errors','callback_errors','unknown_node_records','ambiguous_node_records','kind_mismatch_records','known_wrong_context_records','unknown_activity_records','unknown_resource_events'):
        if counts.get(key)!=0:m.append('native:'+key)
    if not counts.get('ring_overflows') or any(counts['ring_overflows'].values()):m.append('native_ring_loss')
    if r.get('static',{}).get('complete') is not True:m.append('static_incomplete')
    slots=w.get('graph_slots',[]);bank=7 if mode=='positive' else 1;n=40 if mode=='positive' else 1
    if len(slots)!=bank or r.get('replays')!=n:m.append('bank_exposure')
    for i,slot in enumerate(slots):
        for kind,field in (('kernel','kernel_executions'),('memcpy','copy_executions'),('memset','memset_executions')):
            want=n if i==0 else 0
            if (slot.get('required',{}).get(kind)!=MIX[kind] or slot.get(field)!=MIX[kind]*want
                or slot.get('node_execution_bounds',{}).get(kind)!={'min':want,'max':want}):m.append('multiplicity:'+str(i)+':'+kind)
    graph_records={k:sum(x.get('count',0) for x in a.get('native',{}).get('activities',[]) if x.get('graph_id') and x.get('kind')==k) for k in MIX}
    if graph_records!={k:MIX[k]*n for k in MIX}:m.append('graph_correlation_conservation')
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
    from activity_accounting import CorrelationBridge,grade as account_grade
    r=dict(mode=mode,replays=0,failures=[],missing=[],injections={})
    session=split=None;owners=[];window=None;threads=[]
    try:
        binding=ro.ReplayBinding(str(native/'flash_s5_replay_observer.so'),receipt['files']['flash_s5_replay_observer.so'],receipt['files']['flash_s5_replay_observer.cpp'],s.CUPTI,cupti_library_path='/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13')
        limits=ro.ObserverLimits(max_receipt_bytes=64*1024**2);session=ro.ObserverSession(binding,limits);session.start()
        bridge=CorrelationBridge(session,epoch=1,generation=1)
        r['observer_host_reserved_bytes']=session._native_reserved;r['limits']=limits.as_record()
        lib,token=binding._lib,session._token
        host=HostQueries(native/'vmid_context_probe.so')
        split=gc.open_split(0,env={'FLASH_S5_DECODE_SMS':'44'},torch_module=torch,binding=gc.DriverBinding(str(native/'flash_s5_green.so')))
        split.retain_allocator_context();ordinary=torch.cuda.Stream();aux=split.create_decode_aux()
        streams=[split.decode_stream,aux];identities=[host.stream(x) for x in streams]
        context=identities[0]['green_cupti_context_id'];primary=host.stream(ordinary)['primary_cupti_context_id']
        r['context_id']=context;r['primary_context_id']=primary
        src=torch.full((1024,),7,dtype=torch.int32,pin_memory=True);dst=torch.empty_like(src,pin_memory=True)
        a,b,z=[torch.zeros((1024,),dtype=torch.int32,device='cuda') for _ in range(3)]
        buffers=[(torch.zeros_like(a),torch.zeros_like(a),torch.empty_like(src,pin_memory=True)) for _ in range(3)]
        torch.cuda.synchronize();gl=GraphBinding(native/'vmid_graph_probe.so')
        s.checked(lib.s5ro_bind_context(token,split.decode_context_id,44),'bind_context')
        before={};bank=7 if mode=='positive' else 1
        for role,count,stream in (('pressure',8 if mode=='positive' else 0,ordinary),('bound',bank,streams[0])):
            for i in range(count):
                own,graph=C.c_uint64(),C.c_uint64()
                s.checked(gl.lib.vmg_create(int(stream.cuda_stream),a.data_ptr(),b.data_ptr(),z.data_ptr(),src.data_ptr(),dst.data_ptr(),1024,1386,26,7,0,C.byref(own),C.byref(graph)),'create')
                owners.append((role,i,own.value,graph.value))
                if role=='bound':
                    n=C.c_uint32();s.checked(lib.s5ro_bind_graph(token,graph.value,i,C.byref(n)),'bind')
                    before[i]=s.graph_bytes(lib,graph.value)
        target=next(o for role,i,o,g in owners if role=='bound' and i==0)
        eager=C.CDLL(str(native/'vmid_eager_probe.so'));eager.vm8_eager_submit.argtypes=[C.c_uint64]*5+[C.c_int]*2;eager.vm8_eager_submit.restype=C.c_int
        def submit(t,name,n,stream):
            e,f,tmp=buffers[t]
            s.checked(eager.vm8_eager_submit(int(stream.cuda_stream),e.data_ptr(),f.data_ptr(),src.data_ptr(),tmp.data_ptr(),FAMILIES[name][0],n),'submit')
        ready=threading.Barrier(3,timeout=30);start=threading.Event();other_scope=threading.Event();negative_done=threading.Event();errors=[]
        def worker(t):
            try:
                torch.cuda.set_device(0)
                for name in FAMILIES:submit(t,name,1,streams[t])
                streams[t].synchronize();ready.wait()
                if not start.wait(30):raise RuntimeError('window_start_timeout')
                for j in range(40 if mode=='positive' else 1):
                    params=dict(context_id=context,stream_id=identities[t]['cupti_stream_id'])
                    with bridge.offer('tick',{},**params):
                        if t==0:
                            with bridge.offer('graph',MIX,graph=True,copy_kinds=(1,2,8),**params):s.checked(gl.lib.vmg_replay(target),'replay')
                        for name,n in (('h2d',1),('eager_pre',80),('eager_post',83+t*3),('d2d',40),('d2h',1),('memset',2),('helper_sentinel',1)):
                            with bridge.offer(name,{FAMILIES[name][1]:n},copy_kinds=(FAMILIES[name][2],),**params):submit(t,name,n,streams[t])
                        if j==0:
                            with bridge.offer('same_symbol_family_'+str(t),{'kernel':1},**params):submit(t,'eager_pre',1,streams[t])
                            try:
                                with bridge.offer('exception_cleanup',{},**params):raise ValueError('intentional CPU exception')
                            except ValueError:pass
                # Deliberate negative: no offer on this submitting thread;
                # another thread deliberately holds an active scope (TLS isolation).
                if t==0 and mode!='positive':
                    with bridge.offer('negative_other_thread_parent',{},context_id=context,stream_id=identities[t]['cupti_stream_id']):
                        other_scope.set()
                        if not negative_done.wait(30):raise RuntimeError('negative_thread_timeout')
                if t==1 and mode!='positive' and not other_scope.wait(30):raise RuntimeError('other_scope_timeout')
                if t==1 and mode=='unoffered':submit(t,'helper_sentinel',1,streams[t])
                if t==1 and mode=='primary':
                    with bridge.offer('negative_primary',{'kernel':1},context_id=context,stream_id=identities[t]['cupti_stream_id']):submit(2,'helper_sentinel',1,ordinary)
                if t==1:negative_done.set()
            except Exception as exc:errors.append(type(exc).__name__)
        # Warm primary sentinel and graph before tracing; no drains per offer.
        submit(2,'helper_sentinel',1,ordinary);s.checked(gl.lib.vmg_replay(target),'warm');torch.cuda.synchronize()
        threads=[threading.Thread(target=worker,args=(t,),daemon=True) for t in range(2)]
        for t in threads:t.start()
        ready.wait();r['static']=session._report_call(lib.s5ro_static,(token,));ro._validate_static(r['static'])
        begin=C.c_uint64();s.checked(lib.s5ro_begin_window(token,1,C.byref(begin)),'begin');window=begin.value;start.set()
        for t in threads:t.join(90)
        if any(t.is_alive() for t in threads):raise RuntimeError('submit_thread_timeout')
        if errors:raise RuntimeError('submit_thread_failure')
        # Sole drain for the complete deferred batch, after all scopes closed.
        for stream in streams+[ordinary]:stream.synchronize()
        r['replays']=40 if mode=='positive' else 1
        r['window']=session._report_call(lib.s5ro_end_window,(token,window,1),retry=False);window=None
        r['accounting']=bridge.report();r['thread_errors']=errors
        if mode=='positive':r['injections']=injections(r['accounting']['native'],r['accounting']['offers'],account_grade)
        r['serialized_stable']=all(before[i]==s.graph_bytes(lib,g) for role,i,o,g in owners if role=='bound')
        r['serialized_bytes']={str(i):len(v) for i,v in before.items()}
        r['pressure_graphs']=sum(role=='pressure' for role,i,o,g in owners)
        r['occupancy']=occupancy(r['window'])
        nr=r['accounting']['native'];cap=nr['capacity']
        r['correlation_occupancy']={k:dict(count=len(nr[k]),capacity=cap,headroom=cap-len(nr[k])) for k in ('apis','external','scopes')}
        r['output_exact']=bool(torch.all(dst==1393).item() and torch.all(z.cpu()==0).item())
        r['allocator_peak_reserved']=torch.cuda.max_memory_reserved(0)
    except Exception as exc:
        r['missing'].append('case_exception');r['error_type']=type(exc).__name__
        r['error_stack']=[dict(file=Path(x.filename).name,line=x.lineno,function=x.name) for x in traceback.extract_tb(exc.__traceback__)]
    finally:
        if any(t.is_alive() for t in threads):
            # Never destroy observer/graphs under a still-submitting thread.
            # The owned process is the remaining bounded lifetime boundary.
            import os
            r['missing'].append('submitters_not_joined');r['cleanup_ok']=False
            s.write(out/(mode+'-unfinished.json'),r)
            os._exit(2)
        errors=[]
        try:torch.cuda.synchronize()
        except Exception:errors.append('drain')
        try:
            if session:
                if window:session._report_call(binding._lib.s5ro_end_window,(session._token,window,1),retry=False)
                session.close()
        except Exception:errors.append('observer_close')
        try:
            for role,i,o,g in reversed(owners):s.checked(gl.lib.vmg_close(o),'graph_close')
            if split:split.close()
        except Exception:errors.append('native_close')
        r['cleanup_ok']=not errors;r['cleanup_errors']=errors
    r['grade']=case_grade(r);s.write(out/(mode+'.json'),r);return r


def probe(modules,native,out):
    r=dict(schema='vm<vm-id>-job9/1',image=s.IMAGE,cases=[],regressions={},failures=[],missing=[],memory_fraction=.125)
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
    result.update(schema='vm<vm-id>-job9/1',process_rc=process_rc,memory=m,cases=g['cases'],
       regressions=raw.get('regressions',{}),scope='synthetic external-correlation bridge; no engine call-site qualification or graphId-0 waiver')
    s.write(out/'result.json',result);print('VMID_JOB9 '+result['verdict']);return result['rc']


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('probe','monitor','finalize'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--modules',type=Path);p.add_argument('--native',type=Path);p.add_argument('--process-rc',type=int,default=2);a=p.parse_args()
    if a.mode=='monitor':return s.monitor(a.out,'flashnext-job9')
    if a.mode=='finalize':return finalize(a.out,a.process_rc)
    return probe(a.modules,a.native,a.out)
if __name__=='__main__':sys.exit(main())
