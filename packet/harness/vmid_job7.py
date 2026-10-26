#!/usr/bin/env python3
"""Real pinned stream-owner methods with small synthetic operands; VM only."""
import argparse
import ctypes as C
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import vmid_support as s
from vmid_job6 import HostQueries, classify_context

FAMILIES=('shared','ple','composed24')
ROUTES=('before','repaired','cross','restored')
REPLAYS=3


def symbol_join(diag, report, inventories):
    # _normalise uses sorted kernel IDs; convert those ordinals back to IDs,
    # then join the independently retained ALL-node enumeration by exact ID.
    from launch_diagnostic import _normalise
    rows=_normalise(report,inventories)
    by_id={}
    for (label,ordinal),row in rows.items():
        nid=sorted(inventories[label]['nodes'])[ordinal]
        if nid in by_id:raise ValueError('duplicate original kernel id')
        by_id[nid]=row['name']
    result=[]
    for n in diag['nodes']:
        symbol=by_id.get(n['graph_node_id'])
        if symbol is None:raise ValueError('node symbol join incomplete')
        result.append({'slot':n['slot'],'node_index':n['node_index'],
                       'graph_node_id':n['graph_node_id'],'symbol':symbol})
    return result


def case_grade(r):
    f=list(r.get('failures',[]));m=list(r.get('missing',[]))
    for k in ('cleanup_ok','owners_restored','serialized_stable','source_pins_verified'):
        if r.get(k) is False:f.append(k)
        elif r.get(k) is not True:m.append(k)
    if r.get('source_scope')!='real owner methods; synthetic small arithmetic and TP1/DP1 PLE fixture':m.append('source_scope')
    if r.get('overlap_selected') is not True:m.append('real_shared_overlap_branch')
    for name in ROUTES:
        w=r.get('windows',{}).get(name,{})
        if w.get('output_exact') is False:f.append(name+':output')
        elif w.get('output_exact') is not True:m.append(name+':output')
        if w.get('complete') is not True:m.append(name+':evidence_incomplete')
        if w.get('replays')!=REPLAYS:m.append(name+':replays')
        counts=w.get('context_counts',{})
        green=counts.get('expected_green',0);parent=counts.get('expected_parent',0)
        other=sum(v for k,v in counts.items() if k not in ('expected_green','expected_parent'))
        if other:f.append(name+':third_context')
        if name=='before':
            if not parent or not green:m.append('mixed_context_reproduction_missing')
            if w.get('predicate')!='FAIL':m.append('negative_guard_not_reproduced')
        elif name=='restored':
            if not parent or green:f.append('ordinary_restore_context')
            if w.get('predicate')!='FAIL':m.append('restored_negative_guard')
        else:
            if parent:f.append(name+':wrong_context')
            if not green:m.append(name+':green_evidence')
            if w.get('predicate')!='PASS':m.append(name+':placement_not_pass')
    native=r.get('native_window',{})
    expected=native.get('expected_context',{})
    if expected.get('is_green') is not True or expected.get('sm_matching',0)<1:m.append('green_identity')
    if expected.get('contradictory') or expected.get('sm_mismatched',0):f.append('green_identity_contradiction')
    for n in (0,2):
        slots=native.get('graph_slots',[])
        if len(slots)!=3 or 'known_wrong_context' not in slots[n].get('reasons',[]):m.append('native_negative_control')
    wrong=sum(sum(v for k,v in w.get('context_counts',{}).items() if k!='expected_green') for w in r.get('windows',{}).values())
    # Native wrong-context totals also include copies/memsets on negative side
    # streams. Kernel-only diagnostics are a lower bound, never equalized by guess.
    native_wrong=native.get('counts',{}).get('known_wrong_context_records')
    if type(native_wrong) is not int or native_wrong<wrong:m.append('native_negative_record_census')
    if any(v for k,v in native.get('flags',{}).items() if k!='known_wrong_context'):m.append('native_loss')
    if set(native.get('reasons',[])) & {'expected_context_not_green','expected_context_sm_mismatch','context_destroyed'}:f.append('native_context_failure')
    if set(native.get('reasons',[]))-{'known_wrong_context','incomplete_record','missing_execution'}:m.append('unexplained_native_reason')
    # Explicitly allow the negative slots' wrong-context reason only. The
    # repaired graph has its own native slot verdict and full multiplicity.
    fixed=r.get('repaired_slot',{})
    if set(fixed.get('reasons',[])) & {'known_wrong_context','expected_context_not_green','expected_context_sm_mismatch','context_destroyed'}:
        f.append('native_repaired_slot')
    if fixed.get('complete') is not True:m.append('native_repaired_slot_incomplete')
    for kind,field in (('kernel','kernel_executions'),('memcpy','copy_executions'),('memset','memset_executions')):
        n=fixed.get('required',{}).get(kind)
        if type(n) is not int:m.append('native_fixed_inventory');continue
        bounds=fixed.get('node_execution_bounds',{}).get(kind,{})
        if fixed.get(field)!=n*REPLAYS*2 or (n and (bounds.get('min')!=REPLAYS*2 or bounds.get('max')!=REPLAYS*2)):
            m.append('native_fixed_'+kind+'_multiplicity')
    if not r.get('symbol_join'):m.append('authoritative_node_symbol_join')
    return s.verdict(f,m)


def grade(r):
    f=list(r.get('failures',[]));m=list(r.get('missing',[]))
    rows=r.get('cases',[])
    if [x.get('family') for x in rows]!=list(FAMILIES):m.append('case_inventory')
    grades={}
    for x in rows:
        name=x.get('family','unknown');g=case_grade(x);grades[name]=g
        f.extend(name+':'+v for v in g['failures']);m.extend(name+':'+v for v in g['missing'])
    for name in ('job1','job2b_flat'):
        g=r.get('regressions',{}).get(name,{})
        if g.get('verdict')=='FAIL':f.append(name+':regression')
        elif g.get('verdict')!='PASS':m.append(name+':regression_missing')
    g=s.verdict(f,m);g['cases']=grades;return g


def fixture(torch, types, family):
    shared_cls,ple_cls=types
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up=torch.full((1280,2560),.001,device='cuda',dtype=torch.bfloat16)
            self.down=torch.full((2560,640),.001,device='cuda',dtype=torch.bfloat16)
            self.gate=torch.full((1,2560),.001,device='cuda',dtype=torch.bfloat16)
        def forward(self,x):
            import torch.nn.functional as F
            z=F.linear(x,self.up);a,b=z.chunk(2,dim=-1)
            return F.linear(F.silu(a)*b,self.down)*torch.sigmoid(F.linear(x,self.gate))
    cfg=SimpleNamespace(moe_parallel_config=SimpleNamespace(enable_eplb=False,
            all2all_backend='allgather_reducescatter',use_fi_nvl_two_sided_kernels=False))
    mlp=MLP();shared=[shared_cls(mlp,cfg,False,lambda:False) for _ in range(2)]
    x=torch.full((4,2560),.125,device='cuda',dtype=torch.bfloat16)
    selected=all(int(o._determine_shared_experts_order(x))==3 for o in shared)
    if not selected:raise RuntimeError('real_shared_overlap_branch_unavailable')
    # No constructor allocates model-sized weights or initializes distributed
    # groups. Real TP1/DP1 helpers, lookup and fork/join methods remain intact.
    ple=ple_cls.__new__(ple_cls);torch.nn.Module.__init__(ple)
    weight=torch.full((32,160),.25,dtype=torch.bfloat16,pin_memory=True)
    ple.weight=torch.nn.Parameter(weight,requires_grad=False)
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    ple._uva_weight=get_accelerator_view_from_cpu_tensor(weight)
    ple.embedding_dim=160;ple._block_d=256;ple.tp_size=1
    ple.etp_data_parallel_size=1;ple.data_parallel_rank=0
    ple.shard_indices=SimpleNamespace(org_vocab_start_index=0,org_vocab_end_index=32)
    ple._prefetch_stream=torch.cuda.Stream()
    ple._prefetch_buffer=torch.empty((4,2,160),device='cuda',dtype=torch.bfloat16)
    ple._output_dim=320
    ids=torch.tensor([[0,1],[2,3],[4,5],[30,31]],device='cuda',dtype=torch.int64)
    ple_out=torch.empty((4,320),device='cuda',dtype=torch.bfloat16)
    outputs=[torch.empty_like(x) for _ in range(24 if family=='composed24' else 1)]
    marker=torch.zeros((32,),device='cuda')
    owners=[]
    if family!='ple':owners.extend((o,'_stream','shared') for o in shared)
    if family!='shared':owners.append((ple,'_prefetch_stream','ple'))
    def run():
        marker.fill_(2)  # main-context sentinel, not part of shared/PLE arithmetic
        if family!='shared':ple.start_prefetch(x,ids)
        if family!='ple':
            for i,out in enumerate(outputs):
                o=shared[i%2]
                if o.maybe_forward_async(x) is not True:raise RuntimeError('shared_path_changed')
                o.wait();out.copy_(o.output)
        if family!='shared':ple._finalize_prefetch(ple._prefetch_buffer,ple_out)
        marker.add_(1)
    def take():
        selected=([marker]+(outputs if family!='ple' else [])+([ple_out] if family!='shared' else []))
        return [t.detach().cpu().clone() for t in selected]
    return owners,run,take,selected


def case(torch,ro,gc,cs,native,receipt,out,family,vconfig):
    r=dict(family=family,failures=[],missing=[],windows={},
      source_scope='real owner methods; synthetic small arithmetic and TP1/DP1 PLE fixture',
      substitutions=['bounded PLE constructor state (32x160 pinned BF16)','synthetic BF16 gated MLP weights; no model load'],
      source_pins=cs.PINS,started_wall_s=time.time())
    session=split=adapter=None;graphs=[];attached=[];window=None
    try:
        types=cs.pinned_types();r['source_pins_verified']=True
        binding=ro.ReplayBinding(str(native/'flash_s5_replay_observer.so'),receipt['files']['flash_s5_replay_observer.so'],
           receipt['files']['flash_s5_replay_observer.cpp'],s.CUPTI,
           cupti_library_path='/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13')
        limits=ro.ObserverLimits(buffer_count=16,buffer_bytes=1024**2,max_context_records=128,max_kernel_records=65536,
             max_copy_records=8192,max_graph_events=131072,max_graphs=8,max_nodes_per_graph=2048,max_nodes_total=8192,max_links=32768)
        session=ro.ObserverSession(binding,limits);session.start();lib,token=binding._lib,session._token
        r['observer_host_reserved_bytes']=session._native_reserved
        lib.s5ro_context_diagnostics.argtypes=[C.c_uint64,C.c_uint64,C.c_uint64,C.c_void_p,C.c_uint64,C.POINTER(C.c_uint64)]
        lib.s5ro_context_diagnostics.restype=C.c_int
        def diag(start=0,end=0):
            d=session._report_call(lib.s5ro_context_diagnostics,(token,start,end));ro._validate_context_diagnostics(d);return d
        host=HostQueries(native/'vmid_context_probe.so')
        split=gc.open_split(0,env={'FLASH_S5_DECODE_SMS':'44'},torch_module=torch,binding=gc.DriverBinding(str(native/'flash_s5_green.so')))
        ordinary=torch.cuda.Stream()
        owners,run,take,r['overlap_selected']=fixture(torch,types,family)
        from vllm.forward_context import set_forward_context
        from vllm.config import CUDAGraphMode
        def capture(stream):
            with set_forward_context(None,vconfig,num_tokens=4,cudagraph_runtime_mode=CUDAGraphMode.NONE):
                with torch.cuda.stream(stream):
                    for _ in range(3):run()
                torch.cuda.synchronize()
                g=torch.cuda.CUDAGraph(keep_graph=True);graphs.append(g)
                with torch.cuda.graph(g,stream=stream):run()
                g.instantiate();stream.synchronize()
                return g
        full=capture(ordinary)
        with torch.cuda.stream(ordinary):full.replay()
        ordinary.synchronize();reference=take()
        bad=capture(split.decode_stream)
        adapter=cs.CaptureStreams(split,owners);adapter.prepare()
        r['streams_before']=[host.stream(o) for o in adapter.originals]
        r['streams_repaired']=[host.stream(o) for o in adapter.streams]
        with adapter.capture():fixed=capture(split.decode_stream)
        adapter.attach_graph(fixed);attached.append(fixed)
        r['owners_restored']=adapter.record()['original_owners_restored']
        r['owner_inventory']=adapter.record()
        restored=capture(ordinary)
        s.checked(lib.s5ro_bind_context(token,split.decode_context_id,44),'bind_context')
        targets=[bad,fixed,restored];inventories={};serialized=[]
        for slot,g in enumerate(targets):
            n=C.c_uint32();handle=g.raw_cuda_graph()
            s.checked(lib.s5ro_bind_graph(token,handle,slot,C.byref(n)),'bind_graph')
            inventories[str(slot)]=session._report_call(lib.s5ro_graph_ids,(handle,))
            serialized.append(s.graph_bytes(lib,handle))
        r['static']=session._report_call(lib.s5ro_static,(token,));ro._validate_static(r['static'])
        r['before']=diag()
        start=C.c_uint64();s.checked(lib.s5ro_begin_window(token,1,C.byref(start)),'begin');window=start.value
        for name,slot,stream in [('before',0,split.decode_stream),('repaired',1,split.decode_stream),('cross',1,ordinary),('restored',2,ordinary)]:
            torch.cuda.synchronize();begin=host.timestamp()
            with torch.cuda.stream(stream):
                for _ in range(REPLAYS):targets[slot].replay()
            stream.synchronize();end=host.timestamp()
            r['windows'][name]=dict(slot=slot,start=begin,end=end,replays=REPLAYS,
               output_exact=all(torch.equal(a,b) for a,b in zip(reference,take())))
        w=session._report_call(lib.s5ro_end_window,(token,window,1),retry=False);window=None
        ro._validate_window(w,limits,1);r['native_window']=w
        r['repaired_slot']=w['graph_slots'][1]
        # One fixed graph appears in two subwindows, therefore native count=6.
        for name,row in r['windows'].items():
            d=diag(row['start'],row['end']);row['diagnostics']=d
            counts={};pernode={}
            for e in d['executions']:
                cls=classify_context(d,e['context_id']);counts[cls]=counts.get(cls,0)+e['count']
                key=(e['slot'],e['node_index']);pernode[key]=pernode.get(key,0)+e['count']
            expected={(n['slot'],n['node_index']) for n in d['nodes'] if n['slot']==row['slot']}
            row['context_counts']=counts;row['complete']=bool(d['complete'] and expected and set(pernode)==expected and all(n==REPLAYS for n in pernode.values()))
            row['predicate']='FAIL' if any(k!='expected_green' and v for k,v in counts.items()) else 'PASS' if counts else 'INCONCLUSIVE'
        launch=session._report_call(lib.s5ro_launch_report,(token,0))
        r['symbol_join']=symbol_join(diag(),launch,inventories)
        r['serialized_stable']=all(a==s.graph_bytes(lib,g.raw_cuda_graph()) for a,g in zip(serialized,targets))
        r['serialized_bytes']=[len(x) for x in serialized]
        r['allocator_peak_reserved']=torch.cuda.max_memory_reserved(0)
    except Exception as exc:
        r['error_type']=type(exc).__name__;r['error_stack']=[dict(file=Path(x.filename).name,line=x.lineno,function=x.name) for x in traceback.extract_tb(exc.__traceback__)]
        if isinstance(exc,torch.cuda.OutOfMemoryError):r['failures'].append('cuda_oom')
        else:r['missing'].append('case_exception')
    finally:
        errors=[];drained=False
        try:torch.cuda.synchronize();drained=True
        except Exception:errors.append('drain')
        try:
            if session and window is not None:session._report_call(session.binding._lib.s5ro_end_window,(session._token,window,1),retry=False)
            if session:session.close()
        except Exception:errors.append('observer_close')
        if drained:
            for g in reversed(graphs):
                try:
                    g.reset()
                    if adapter and g in attached:adapter.retire_graph(g)
                except Exception:errors.append('graph_reset')
            try:
                if adapter:adapter.close_after_graphs()
                if split:split.close()
            except Exception:errors.append('stream_close')
        r['cleanup_ok']=not errors;r['cleanup_errors']=errors;r['ended_wall_s']=time.time()
    r['grade']=case_grade(r);s.write(out/(family+'.json'),r);return r


def probe(modules,native,out):
    r=dict(schema='vm<vm-id>-job7/1',image=s.IMAGE,cases=[],regressions={},failures=[],missing=[],memory_fraction=.125)
    try:
        import torch
        torch.cuda.set_per_process_memory_fraction(.125,0);torch.set_num_threads(4)
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)!=(8,6):raise RuntimeError('device')
        sys.path.insert(0,str(modules));import replay_observer as ro;import green_context as gc;import capture_streams as cs
        receipt=json.loads((native/'q17-helper-build.json').read_text())
        if receipt['image']!=s.IMAGE or receipt['cupti_library_sha256']!=s.CUPTI:raise RuntimeError('build_pin')
        for name,h in receipt['files'].items():
            if s.sha(native/name)!=h:raise RuntimeError('artifact_pin')
        for name in ('flash_s5_green.cu','flash_s5_replay_observer.cpp','flash_s5_graph_bytes.cpp'):
            if s.sha(modules/'native'/name)!=receipt['files'][name]:raise RuntimeError('source_pin')
        r['build']=receipt
        from vllm.config import VllmConfig,set_current_vllm_config
        with set_current_vllm_config(VllmConfig()) as unused:
            # Obtain the same config object without relying on contextmanager return.
            from vllm.config import get_current_vllm_config
            cfg=get_current_vllm_config()
            for family in FAMILIES:r['cases'].append(case(torch,ro,gc,cs,native,receipt,out,family,cfg))
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
    result.update(schema='vm<vm-id>-job7/1',process_rc=process_rc,memory=m,cases=g['cases'],
       regressions=raw.get('regressions',{}),scope='real stream-owner methods with synthetic payloads; no engine/box qualification')
    s.write(out/'result.json',result);print('VMID_JOB7 '+result['verdict']);return result['rc']


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('probe','monitor','finalize'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--modules',type=Path);p.add_argument('--native',type=Path);p.add_argument('--process-rc',type=int,default=2);a=p.parse_args()
    if a.mode=='monitor':return s.monitor(a.out,'flashnext-job7')
    if a.mode=='finalize':return finalize(a.out,a.process_rc)
    return probe(a.modules,a.native,a.out)
if __name__=='__main__':sys.exit(main())
