#!/usr/bin/env python3
"""VM-only context isolation. No engine; placement guards remain unchanged."""
import argparse
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import vmid_support as s

FAMILIES=('aten','cublas','qsa_gather','composed')
ORIGINS=('plain','green')
REPLAYS=3
QSA_SHA='5a0d962022ec4e47809fa072d2b13745c84413de894048e67f10c3c10a45d51b'
NO_NODE=4294967295


def classify_context(d,ident):
    if ident==d['expected_context_id']:return 'expected_green'
    contexts=d.get('contexts',[])
    expected=[c for c in contexts if c['context_id']==d['expected_context_id']]
    if any(c['parent_context_id']==ident and ident!=0 for c in expected):return 'expected_parent'
    matched=[c for c in contexts if c['context_id']==ident]
    if any(c['is_green'] for c in matched):return 'other_green'
    return 'other_context' if matched else 'unknown'


def joined(d):
    """Join by observer node identity, never enumeration position or symbol."""
    nodes={(n['slot'],n['node_index']):n for n in d.get('nodes',[])}
    result=[]
    for r in d.get('executions',[]):
        n=nodes.get((r['slot'],r['node_index']))
        result.append(dict(r,node=n,activity_context_class=classify_context(d,r['context_id']),
            static_context_applicable=bool(n and n['ctx_semantics']=='explicit_kernel_context'
                                          and n['context_id_available']),
            static_activity_agree=(n['context_id']==r['context_id']) if n and
                n['ctx_semantics']=='explicit_kernel_context' and n['context_id_available'] else None))
    return result


def subgrade(r,static,route,origin):
    fails=[];missing=[];d=r.get('diagnostics',{})
    if r.get('output_exact') is False:fails.append('output_changed')
    elif r.get('output_exact') is not True:missing.append('output_check')
    if r.get('replays')!=REPLAYS:missing.append('replay_census')
    if d.get('complete') is not True:missing.append('context_diagnostics_incomplete')
    nodes=d.get('nodes',[]);keys={(n.get('slot'),n.get('node_index')) for n in nodes}
    required=sum(x.get('required',{}).get('kernel',0) for x in static.get('graph_slots',[]))
    if len(keys)!=len(nodes) or len(nodes)!=required or required==0:missing.append('static_kernel_census')
    counts={k:0 for k in keys};wrong=0
    for n in nodes:
        if n.get('query_rc')!=0:missing.append('node_query')
    for x in d.get('executions',[]):
        key=(x.get('slot'),x.get('node_index'))
        if key not in counts:missing.append('unjoined_kernel');continue
        counts[key]+=x['count']
        if x['context_id']!=d.get('expected_context_id'):wrong+=x['count']
    if any(v!=REPLAYS for v in counts.values()):missing.append('per_node_multiplicity')
    info=r.get('stream',{})
    if route=='green':
        if info.get('green_present') is not True or info.get('green_cupti_context_id')!=d.get('expected_context_id'):
            fails.append('wrong_green_stream')
        if wrong and origin=='green':fails.append('known_wrong_context')
    else:
        if info.get('green_present') is not False:fails.append('negative_stream_not_plain')
        if wrong==0 and origin=='plain':fails.append('ordinary_negative_not_rejected')
    result=s.verdict(fails,missing)
    result.update(current_context_predicate='FAIL' if wrong else 'INCONCLUSIVE' if missing else 'PASS',wrong_context_records=wrong,
        purpose='ordinary-route negative control' if route==origin=='plain' else
            'green-route placement' if route==origin=='green' else 'cross-origin replay diagnostic only')
    return result


def case_grade(r):
    failures=list(r.get('failures',[]));missing=list(r.get('missing',[]))
    for k in ('cleanup_ok','serialized_stable','native_schema_valid','static_census_stable'):
        if r.get(k) is False:failures.append(k)
        elif r.get(k) is not True:missing.append(k)
    if r.get('membership',{}).get('ok') is not True:missing.append('membership_evidence')
    if r.get('split')!=dict(decode=44,prefill=38,total=82):failures.append('split_identity')
    static=r.get('static',{});window=r.get('window',{})
    if static.get('complete') is not True:missing.append('static_incomplete')
    ctx=window.get('expected_context',{})
    if ctx.get('is_green') is False or ctx.get('sm_mismatched',0) or ctx.get('contradictory'):failures.append('context_identity')
    if ctx.get('is_green') is not True or ctx.get('sm_matching',0)<1:missing.append('context_identity_evidence')
    counts=window.get('counts',{})
    for key in ('dropped_records_total','buffers_refused','parse_errors','callback_errors',
                'ambiguous_node_records','unknown_node_records','kind_mismatch_records',
                'incomplete_records','unknown_timestamp_records','invalid_interval_records','boundary_overlap_records',
                'drop_query_errors','unknown_activity_records','unknown_resource_events'):
        if type(counts.get(key)) is not int:missing.append('counter:'+key)
        elif counts[key]:failures.append('loss_or_mapping:'+key)
    allowed={'known_wrong_context','missing_execution','incomplete_record','unattributed_expected_context'}
    if set(window.get('reasons',[]))-allowed:missing.append('native_observation_failure')
    rings=counts.get('ring_overflows',{})
    if set(rings)!={'aliases','resource','context','kernel','copy'}:missing.append('overflow_evidence')
    elif any(rings.values()):failures.append('overflow')
    # The parent window intentionally contains the plain-stream negative.
    # Its unchanged native verdict must reject it; it is never reported PASS.
    if r.get('capture_origin')=='plain' and window.get('flags',{}).get('known_wrong_context') is not True:
        failures.append('native_guard_did_not_reject_negative')
    phases={route:subgrade(r.get('windows',{}).get(route,{}),static,route,r.get('capture_origin')) for route in ('plain','green')}
    for route,g in phases.items():
        failures.extend(route+':'+x for x in g['failures']);missing.extend(route+':'+x for x in g['missing'])
    result=s.verdict(failures,missing);result['windows']=phases
    return result


def grade(r):
    rows=r.get('cases',[]);g=s.combine([case_grade(x) for x in rows])
    if [(x.get('family'),x.get('capture_origin')) for x in rows]!=[(f,o) for f in FAMILIES for o in ORIGINS]:g['missing'].append('case_census')
    result=s.verdict(g['failures']+r.get('failures',[]),g['missing']+r.get('missing',[]))
    result['cases']={f"{x.get('family')}-{x.get('capture_origin')}":case_grade(x) for x in rows}
    return result


class HostQueries:
    def __init__(self,path):
        self.lib=C.CDLL(str(path))
        self.lib.vmc_timestamp.argtypes=[C.POINTER(C.c_uint64)];self.lib.vmc_timestamp.restype=C.c_int
        self.lib.vmc_stream.argtypes=[C.c_uint64,C.POINTER(C.c_uint64),C.c_uint];self.lib.vmc_stream.restype=C.c_int
    def timestamp(self):
        v=C.c_uint64();s.checked(self.lib.vmc_timestamp(C.byref(v)),'timestamp');return v.value
    def stream(self,stream):
        values=(C.c_uint64*7)();s.checked(self.lib.vmc_stream(stream.cuda_stream,values,7),'stream_query')
        return dict(primary_cupti_context_id=values[0],green_present=bool(values[1]),green_cupti_context_id=values[2],
                    cupti_stream_id=values[3],primary_cuda_context_id=values[4],green_driver_id=values[5])


def load_qsa():
    root=Path(__file__).resolve().parent
    p=root/'vmid_sources/ops_qsa.py'
    if not p.exists():p=root.parent/'patches/retain/ops_qsa.py.v8.1'
    if s.sha(p)!=QSA_SHA:raise ValueError('qsa_source_pin')
    spec=importlib.util.spec_from_file_location('vmid_job6_qsa',p)
    if spec is None:
        from importlib.machinery import SourceFileLoader
        spec=importlib.util.spec_from_loader('vmid_job6_qsa',SourceFileLoader('vmid_job6_qsa',str(p)))
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod


def operation(torch,qsa,family):
    # Fixed exact representable arithmetic, <32 MiB persistent GPU payload.
    a=torch.full((4096,),2.,device='cuda');b=torch.full_like(a,3.);v=torch.empty_like(a);red=torch.empty((),device='cuda')
    ga=torch.full((512,512),.25,device='cuda',dtype=torch.bfloat16);gb=torch.full_like(ga,.5);go=torch.empty_like(ga)
    cache=torch.full((2,1,4048,512),.25,device='cuda',dtype=torch.bfloat16)
    arena=torch.zeros((1,1,4048,512),device='cuda',dtype=torch.bfloat16)
    indices=torch.tensor([0,4047,4048,8095],dtype=torch.int64,device='cuda');table=torch.tensor([0,1],dtype=torch.int32,device='cuda')
    def run():
        if family in ('aten','composed'):
            torch.add(a,b,out=v);torch.sum(v,dim=0,out=red)
        if family in ('cublas','composed'):torch.mm(ga,gb,out=go)
        if family in ('qsa_gather','composed'):
            qsa._qsa_gather_selected_host_rows[(4,1)](cache,indices,table,arena,*cache.stride()[:3],*arena.stride()[:3],
                4,PAGE_SIZE=4048,ROW_WIDTH=512,BLOCK_D=512,num_warps=4)
    def check():
        return bool((family not in ('aten','composed') or (torch.all(v==5).item() and red.item()==20480)) and
            (family not in ('cublas','composed') or torch.all(go==64).item()) and
            (family not in ('qsa_gather','composed') or torch.all(arena.view(-1,512)[:4]==.25).item()))
    return run,check


def case(torch,ro,gc,qsa,native,receipt,out,family,origin):
    name=f'{family}-{origin}';r=dict(family=family,capture_origin=origin,replays_per_window=REPLAYS,
        windows={},failures=[],missing=[],started_wall_s=time.time())
    session=split=None;graphs=[];window=None;buffers=[]
    try:
        binding=ro.ReplayBinding(str(native/'flash_s5_replay_observer.so'),receipt['files']['flash_s5_replay_observer.so'],
            receipt['files']['flash_s5_replay_observer.cpp'],s.CUPTI,
            cupti_library_path='/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13')
        limits=ro.ObserverLimits(buffer_count=8,buffer_bytes=1024**2,max_context_records=128,max_kernel_records=8192,
            max_copy_records=1024,max_graph_events=16384,max_graphs=4,max_nodes_per_graph=256,max_nodes_total=1024,max_links=4096)
        session=ro.ObserverSession(binding,limits);session.start();lib,token=binding._lib,session._token
        r['observer_host_reserved_bytes']=session._native_reserved
        lib.s5ro_context_diagnostics.argtypes=[C.c_uint64,C.c_uint64,C.c_uint64,C.c_void_p,C.c_uint64,C.POINTER(C.c_uint64)]
        lib.s5ro_context_diagnostics.restype=C.c_int
        def diag(start=0,end=0):
            d=session._report_call(lib.s5ro_context_diagnostics,(token,start,end));ro._validate_context_diagnostics(d);return d
        host=HostQueries(native/'vmid_context_probe.so')
        split=gc.open_split(0,env={'FLASH_S5_DECODE_SMS':'44'},torch_module=torch,binding=gc.DriverBinding(str(native/'flash_s5_green.so')))
        ordinary=torch.cuda.Stream();r['split']=dict(decode=split.decode_sms,prefill=split.prefill_sms,total=split.total_sms)
        r['membership']=split.membership(rounds=3).as_record()
        r['membership_scope']='helper sentinel on decode/prefill streams only; not library-kernel placement'
        if not r['membership']['ok']:r['failures'].append('sentinel_membership')
        run,check=operation(torch,qsa,family);buffers=[run,check]
        torch.cuda.synchronize()
        # Realistic module/handle warmup on the ordinary route precedes green capture.
        with torch.cuda.stream(ordinary):
            for _ in range(3):run()
        ordinary.synchronize()
        full=torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(full,stream=ordinary):run()
        full.instantiate();graphs.append(full)
        if origin=='green':
            with torch.cuda.stream(split.decode_stream):
                for _ in range(3):run()
            split.decode_stream.synchronize()
            target=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(target,stream=split.decode_stream):run()
            target.instantiate();graphs.append(target)
        else:target=full
        torch.cuda.synchronize();handle=int(target.raw_cuda_graph());before=s.graph_bytes(lib,handle)
        r['serialized_bytes']=len(before)
        s.checked(lib.s5ro_bind_context(token,split.decode_context_id,44),'bind_context')
        n=C.c_uint32();s.checked(lib.s5ro_bind_graph(token,handle,0,C.byref(n)),'bind_graph')
        r['static']=session._report_call(lib.s5ro_static,(token,));ro._validate_static(r['static'])
        r['node_census_before']=diag();s.write(out/(name+'-before.json'),r['node_census_before'])
        r['streams']={key:host.stream(stream) for key,stream in [('plain',ordinary),('green',split.decode_stream)]}
        start=C.c_uint64();s.checked(lib.s5ro_begin_window(token,1,C.byref(start)),'begin_window');window=start.value
        # Native observer stays in its one window. Host-drained timestamp
        # intervals isolate the same graph's two replay routes without resetting losses.
        for route,stream in [('plain',ordinary),('green',split.decode_stream)]:
            torch.cuda.synchronize();begin=host.timestamp()
            with torch.cuda.stream(stream):
                for _ in range(REPLAYS):target.replay()
            stream.synchronize();end=host.timestamp()
            r['windows'][route]=dict(start_ns=begin,end_ns=end,replays=REPLAYS,stream=r['streams'][route],output_exact=check())
        r['window']=session._report_call(lib.s5ro_end_window,(token,window,1),retry=False);window=None
        s.write(out/(name+'-window.json'),r['window']);ro._validate_window(r['window'],limits,1);r['native_schema_valid']=True
        for route,stream in [('plain',ordinary),('green',split.decode_stream)]:
            row=r['windows'][route];d=diag(row['start_ns'],row['end_ns']);row['diagnostics']=d;row['joined']=joined(d)
        after=diag();r['static_census_stable']=r['node_census_before']['nodes']==after['nodes']
        r['serialized_stable']=before==s.graph_bytes(lib,handle)
        r['allocator_peak_reserved']=torch.cuda.max_memory_reserved(0)
        if r['allocator_peak_reserved']>2*1024**3:r['failures'].append('allocator_budget')
    except Exception as exc:
        r['error_type']=type(exc).__name__;r['error_code']=getattr(exc,'code',None)
        r['error_stack']=[dict(file=Path(x.filename).name,line=x.lineno,function=x.name) for x in traceback.extract_tb(exc.__traceback__)]
        if isinstance(exc,torch.cuda.OutOfMemoryError):r['failures'].append('cuda_oom')
        else:r['missing'].append('case_exception')
    finally:
        errors=[];drained=False
        try:torch.cuda.synchronize();drained=True
        except Exception:errors.append('drain')
        try:
            if session and window is not None:
                d=session._report_call(session.binding._lib.s5ro_end_window,(session._token,window,1),retry=False)
                s.write(out/(name+'-cleanup-window.json'),d)
            if session:session.close()
        except Exception:errors.append('observer_close')
        if drained:
            for g in reversed(graphs):
                try:g.reset()
                except Exception:errors.append('graph_reset')
            try:
                if split:split.close()
            except Exception:errors.append('split_close')
        r['cleanup_ok']=not errors;r['cleanup_errors']=errors;r['ended_wall_s']=time.time()
    r['grade']=case_grade(r);s.write(out/(name+'.json'),r);return r


def probe(modules,native,out):
    r=dict(schema='vm<vm-id>-job6/1',image=s.IMAGE,cases=[],failures=[],missing=[],qsa_source_sha256=QSA_SHA,
           scope='single-GPU synthetic library/context attribution; no model, collective, engine or performance qualification',
           windows_basis='two drained timestamp subwindows inside one immutable native observer window; no reset or second observer subscription',
           memory_fraction=.125,started_wall_s=time.time())
    try:
        import torch
        torch.cuda.set_per_process_memory_fraction(.125,0);torch.set_num_threads(4)
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)!=(8,6):raise RuntimeError('device')
        sys.path.insert(0,str(modules));import replay_observer as ro;import green_context as gc
        receipt=json.loads((native/'q17-helper-build.json').read_text())
        if receipt['image']!=s.IMAGE or receipt['cupti_library_sha256']!=s.CUPTI:raise RuntimeError('build_pin')
        for name,digest in receipt['files'].items():
            if s.sha(native/name)!=digest:raise RuntimeError('artifact_pin')
        for name in ('flash_s5_green.cu','flash_s5_replay_observer.cpp','flash_s5_graph_bytes.cpp'):
            if s.sha(modules/'native'/name)!=receipt['files'][name]:raise RuntimeError('source_pin')
        r['build']=receipt;r['software']=dict(torch=str(torch.__version__),cuda=torch.version.cuda)
        qsa=load_qsa();time.sleep(2)
        for family in FAMILIES:
            for origin in ORIGINS:r['cases'].append(case(torch,ro,gc,qsa,native,receipt,out,family,origin))
        time.sleep(1)
    except Exception as exc:
        r['error_type']=type(exc).__name__;r['missing'].append('probe_exception')
        r['error_stack']=[dict(file=Path(x.filename).name,line=x.lineno,function=x.name) for x in traceback.extract_tb(exc.__traceback__)]
    r['grade']=grade(r);r['ended_wall_s']=time.time();s.write(out/'probe.json',r);return r['grade']['rc']


def finalize(out,process_rc):
    try:g=grade(json.loads((out/'probe.json').read_text()))
    except (OSError,ValueError,TypeError,KeyError):g=s.verdict([],['probe_result']);g['cases']={}
    try:m=json.loads((out/'monitor.json').read_text())
    except (OSError,ValueError):m={}
    failures=list(g['failures']);missing=list(g['missing'])
    if m.get('stop_triggered') or m.get('own_peak_mib',0)>3072:failures.append('resource_stop_or_cap')
    if process_rc in (139,134):failures.append('native_process_crash')
    elif process_rc not in (0,1,2) or process_rc!=g['rc']:missing.append('process_verdict_mismatch')
    if not m.get('own_samples') or m.get('errors',1):missing.append('memory_sampling')
    result=s.verdict(failures,missing)
    result.update(schema='vm<vm-id>-job6/1',process_rc=process_rc,memory=m,cases=g['cases'],
        scope='context-isolation experiment; cross-origin replay is diagnostic; no engine or box GO')
    s.write(out/'result.json',result);print('VMID_JOB6 '+result['verdict']);return result['rc']


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('probe','monitor','finalize'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--modules',type=Path);p.add_argument('--native',type=Path)
    p.add_argument('--process-rc',type=int,default=2);a=p.parse_args()
    if a.mode=='monitor':return s.monitor(a.out,'flashnext-job6')
    if a.mode=='finalize':
        return finalize(a.out,a.process_rc)
    return probe(a.modules,a.native,a.out)
if __name__=='__main__':sys.exit(main())
