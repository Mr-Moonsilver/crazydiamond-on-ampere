#!/usr/bin/env python3
"""Graph + declared eager families, same context: positive census, no suppression.

Four sequential synthetic rank equivalents. Counts model full11 exposure,
not a claim that its unnamed graphId-0 records came from these toy operations.
The original observer verdict remains INCONCLUSIVE for graphId-0 work.
"""
import argparse
from collections import Counter
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import vmid_support as s
from vmid_job2 import GraphBinding
from vmid_job6 import HostQueries

REPLAYS=40
# CUDA CUPTI memcpy kinds (not cudaMemcpyKind): HtoD=1, DtoH=2, DtoD=8.
FAMILIES={'eager_pre':(1,'kernel',0),'eager_post':(2,'kernel',0),
          'helper_sentinel':(3,'kernel',0),'h2d':(4,'memcpy',1),
          'd2d':(5,'memcpy',8),'d2h':(6,'memcpy',2),'memset':(7,'memset',0)}

def plan(rank):
    return dict(graph={'kernel':1368 if rank<2 else 1362,'memcpy':26 if rank<2 else 25,'memset':6},
        families={'eager_pre':80,'eager_post':83 if rank<2 else 86,'helper_sentinel':1,
                  'h2d':1,'d2d':40 if rank<2 else 39,'d2h':1,'memset':2},replays=REPLAYS)


def totals(census):
    return Counter({}) if not census else Counter({k:sum(x['count'] for x in census.get('rows',[]) if x['kind']==k) for k in ('kernel','memcpy','memset')})

def signature(row):
    return tuple(row.get(k) for k in ('kind','context_id','stream_id','graph_id','node_id','copy_kind','name'))


def accounting_grade(whole,intervals,context_id,stream_id):
    """All records must belong to exactly one predeclared drained interval.

    This VM-only ledger needs exact expected counts, direction, context and
    stream; it does not grant the engine a graphId-0 exception.
    """
    f=[];m=[];seen=Counter();by_family={};prev=whole.get('start',0)
    def valid(c):
        return (c.get('schema')=='s5ro-activity-census/1' and c.get('complete') is True
                and c.get('boundary_records')==c.get('invalid_records')==c.get('omitted_records')==0
                and type(c.get('rows')) is list and all(type(x.get('count')) is int and x['count']>0 for x in c['rows'])
                and sum(x['count'] for x in c['rows'])==c.get('matched_records'))
    if not valid(whole):m.append('whole_census_incomplete')
    for item in intervals:
        c=item.get('census',{});name=item.get('family','?');expected=item.get('expected',{})
        if not valid(c):m.append(name+':census_incomplete');continue
        if not prev<=c['start']<c['end']<=whole.get('end',0):f.append('overlap_or_outside_interval')
        prev=c['end']
        counts=totals(c)
        if dict(counts)!={k:expected.get(k,0) for k in ('kernel','memcpy','memset')}:f.append(name+':exact_count')
        for row in c['rows']:
            seen[signature(row)]+=row['count']
            if row['context_id']!=context_id:f.append(name+':wrong_context')
            if name!='graph' and row['stream_id']!=stream_id:f.append(name+':wrong_stream')
            if name=='graph':
                if not row['graph_id'] or not row['node_id']:f.append('graph_identity')
            else:
                if row['graph_id'] or row['node_id']:f.append(name+':not_eager')
                spec=FAMILIES.get(name)
                if spec is None:f.append('undeclared_family');continue
                if row['kind']!=spec[1] or row['copy_kind']!=spec[2]:f.append(name+':kind_or_direction')
                if row['kind']=='kernel' and {1:'vm8_pre',2:'vm8_post',3:'vm8_sentinel'}[spec[0]] not in row['name']:f.append(name+':kernel_family')
        target=by_family.setdefault(name,Counter());target.update(counts)
    all_seen=Counter()
    for row in whole.get('rows',[]):all_seen[signature(row)]+=row['count']
    if not m and seen!=all_seen:f.append('unassigned_or_duplicate_activity')
    result=s.verdict(f,m);result['families']={k:dict(v) for k,v in by_family.items()};return result


def case_grade(r):
    f=list(r.get('failures',[]));m=list(r.get('missing',[]));rank=r.get('rank_equivalent')
    if rank not in range(4):return s.verdict(f,m+['rank_equivalent'])
    p=plan(rank)
    for key in ('output_exact','serialized_stable','cleanup_ok'):
        if r.get(key) is False:f.append(key)
        elif r.get(key) is not True:m.append(key)
    if r.get('replays')!=REPLAYS or r.get('plan')!=p:m.append('declared_exposure')
    if r.get('static',{}).get('complete') is not True:m.append('static_incomplete')
    w=r.get('window',{});ctx=w.get('expected_context',{})
    if ctx.get('is_green') is not True or ctx.get('sm_matching',0)<1:m.append('green_identity')
    if ctx.get('contradictory') or ctx.get('sm_mismatched',0):f.append('green_contradiction')
    # Old aggregate guard MUST still refuse eager expected-context work.
    if 'unattributed_expected_context' not in w.get('reasons',[]):f.append('original_guard_not_preserved')
    if set(w.get('reasons',[]))-{'unattributed_expected_context'}:m.append('other_native_reason')
    if any(v for k,v in w.get('flags',{}).items() if k!='unattributed_expected_context'):m.append('native_loss_or_mapping')
    counts=w.get('counts',{})
    for k in ('dropped_records_total','buffers_refused','drop_query_errors','parse_errors','callback_errors',
              'unknown_node_records','ambiguous_node_records','kind_mismatch_records','known_wrong_context_records',
              'unknown_activity_records','unknown_resource_events'):
        if counts.get(k)!=0:m.append('native_counter:'+k)
    if not counts.get('ring_overflows') or any(counts['ring_overflows'].values()):m.append('ring_loss')
    slots=w.get('graph_slots',[])
    if len(slots)!=1:m.append('one_graph')
    else:
        for kind,field in (('kernel','kernel_executions'),('memcpy','copy_executions'),('memset','memset_executions')):
            if (slots[0].get('required',{}).get(kind)!=p['graph'][kind]
                or slots[0].get(field)!=p['graph'][kind]*REPLAYS
                or slots[0].get('node_execution_bounds',{}).get(kind)!={'min':REPLAYS,'max':REPLAYS}):m.append('graph_multiplicity:'+kind)
    try:
        a=accounting_grade(r['positive'],r['intervals'],r['context_id'],r['stream_id'])
        f.extend(a['failures']);m.extend(a['missing'])
        family_totals=a['families']
        order=('h2d','eager_pre','graph','eager_post','d2d','d2h','memset','helper_sentinel')
        if [x.get('family') for x in r['intervals']]!=list(order)*REPLAYS:m.append('family_schedule')
        for item in r['intervals']:
            name=item['family']
            wanted=p['graph'] if name=='graph' else {FAMILIES[name][1]:p['families'][name]}
            if item.get('expected')!=wanted:f.append('family_plan_changed')
        for name,n in p['families'].items():
            if family_totals.get(name,{}).get(FAMILIES[name][1])!=n*REPLAYS:m.append('family_coverage:'+name)
        # Negative 1: same-green undeclared eager work, in a ledger with no offer.
        neg=accounting_grade(r['negative_unoffered'],[],r['context_id'],r['stream_id'])
        if 'unassigned_or_duplicate_activity' not in neg['failures']:f.append('unoffered_negative_not_rejected')
        # Negative 2: explicitly offered sentinel on the primary context.
        q=r['negative_primary'];decl=dict(family='helper_sentinel',expected={'kernel':1},census=q)
        neg=accounting_grade(q,[decl],r['context_id'],r['stream_id'])
        if 'helper_sentinel:wrong_context' not in neg['failures']:f.append('primary_negative_not_rejected')
        if any(x['context_id']!=r.get('primary_context_id') for x in q['rows']):f.append('negative_not_primary_context')
        for key in ('negative_unoffered','negative_primary'):
            q=r[key]
            if not q.get('complete') or totals(q)!=Counter(kernel=1,memcpy=0,memset=0):m.append(key+':coverage')
        if (r['all_activity'].get('complete') is not True
                or r['all_activity'].get('omitted_records')!=0
                or r['all_activity'].get('boundary_records')!=0
                or r['all_activity'].get('invalid_records')!=0):m.append('whole_window_census')
        all_counts=totals(r['all_activity'])
        if all_counts!=totals(r['positive'])+totals(r['negative_unoffered'])+totals(r['negative_primary']):f.append('window_unassigned')
        if (all_counts['kernel']!=counts.get('kernel_records') or all_counts['memcpy']+all_counts['memset']!=counts.get('copy_records')):m.append('native_census_reconciliation')
        if counts.get('unattributed_expected_context')!=(sum(p['families'].values())*REPLAYS+1):m.append('native_eager_reconciliation')
    except (KeyError,TypeError,ValueError):m.append('accounting_evidence_missing');family_totals={}
    out=s.verdict(f,m);out['families']=family_totals;return out


def grade(r):
    f=list(r.get('failures',[]));m=list(r.get('missing',[]));cases={}
    if [x.get('rank_equivalent') for x in r.get('cases',[])]!=list(range(4)):m.append('four_rank_equivalents')
    for row in r.get('cases',[]):
        g=case_grade(row);label=str(row.get('rank_equivalent'));cases[label]=g
        f.extend(label+':'+x for x in g['failures']);m.extend(label+':'+x for x in g['missing'])
    for name in ('job1','job2b_flat'):
        g=r.get('regressions',{}).get(name,{})
        if g.get('verdict')=='FAIL':f.append(name+':regression')
        elif g.get('verdict')!='PASS':m.append(name+':regression_missing')
    out=s.verdict(f,m);out['cases']=cases;return out


def case(torch,ro,gc,native,receipt,out,rank):
    p=plan(rank);r=dict(rank_equivalent=rank,plan=p,replays=0,failures=[],missing=[],intervals=[],
        scope='synthetic declared families and full11-sized graph exposure; not engine operation attribution')
    session=split=None;owner=0;window=None
    try:
        binding=ro.ReplayBinding(str(native/'flash_s5_replay_observer.so'),receipt['files']['flash_s5_replay_observer.so'],
          receipt['files']['flash_s5_replay_observer.cpp'],s.CUPTI,cupti_library_path='/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13')
        limits=ro.ObserverLimits();session=ro.ObserverSession(binding,limits);session.start()
        r['observer_host_reserved_bytes']=session._native_reserved
        lib,token=binding._lib,session._token
        lib.s5ro_activity_census.argtypes=[C.c_uint64]*3+[C.c_void_p,C.c_uint64,C.POINTER(C.c_uint64)];lib.s5ro_activity_census.restype=C.c_int
        host=HostQueries(native/'vmid_context_probe.so')
        split=gc.open_split(0,env={'FLASH_S5_DECODE_SMS':'44'},torch_module=torch,binding=gc.DriverBinding(str(native/'flash_s5_green.so')))
        split.retain_allocator_context();ordinary=torch.cuda.Stream()
        identity=host.stream(split.decode_stream);r['context_id']=identity['green_cupti_context_id'];r['stream_id']=identity['cupti_stream_id'];r['primary_context_id']=host.stream(ordinary)['primary_cupti_context_id']
        src=torch.full((1024,),7,dtype=torch.int32,pin_memory=True);dst=torch.empty_like(src,pin_memory=True)
        a,b,z,e,f=[torch.zeros((1024,),dtype=torch.int32,device='cuda') for _ in range(5)]
        tmp=torch.empty_like(src,pin_memory=True);torch.cuda.synchronize()
        gl=GraphBinding(native/'vmid_graph_probe.so');own,graph=C.c_uint64(),C.c_uint64()
        s.checked(gl.lib.vmg_create(split.decode_stream_token,a.data_ptr(),b.data_ptr(),z.data_ptr(),src.data_ptr(),dst.data_ptr(),1024,
          p['graph']['kernel'],p['graph']['memcpy'],p['graph']['memset'],0,C.byref(own),C.byref(graph)),'graph_create');owner=own.value
        eager=C.CDLL(str(native/'vmid_eager_probe.so'));eager.vm8_eager_submit.argtypes=[C.c_uint64]*5+[C.c_int]*2;eager.vm8_eager_submit.restype=C.c_int
        def submit(name,n,stream):
            s.checked(eager.vm8_eager_submit(int(stream.cuda_stream),e.data_ptr(),f.data_ptr(),src.data_ptr(),tmp.data_ptr(),FAMILIES[name][0],n),'eager_submit')
        # Warm every function before the measured late-enabled window.
        for name in FAMILIES:submit(name,1,split.decode_stream)
        submit('helper_sentinel',1,ordinary);torch.cuda.synchronize()
        s.checked(lib.s5ro_bind_context(token,split.decode_context_id,44),'bind_context');n=C.c_uint32()
        s.checked(lib.s5ro_bind_graph(token,graph.value,0,C.byref(n)),'bind_graph')
        r['static']=session._report_call(lib.s5ro_static,(token,));ro._validate_static(r['static'])
        before=s.graph_bytes(lib,graph.value);r['serialized_bytes']=len(before)
        start=C.c_uint64();s.checked(lib.s5ro_begin_window(token,1,C.byref(start)),'begin');window=start.value
        intervals=[];positive_start=host.timestamp()
        def measure(name,n=1):
            begin=host.timestamp()
            if name=='graph':s.checked(gl.lib.vmg_replay(owner),'replay');expected=p['graph']
            else:submit(name,n,split.decode_stream);expected={FAMILIES[name][1]:n}
            split.decode_stream.synchronize();end=host.timestamp()
            intervals.append(dict(family=name,expected=expected,start=begin,end=end))
        for _ in range(REPLAYS):
            for name in ('h2d','eager_pre'):measure(name,p['families'][name])
            measure('graph')
            for name in ('eager_post','d2d','d2h','memset','helper_sentinel'):measure(name,p['families'][name])
            r['replays']+=1
        positive_end=host.timestamp()
        neg_start=host.timestamp();submit('helper_sentinel',1,split.decode_stream);split.decode_stream.synchronize();neg_end=host.timestamp()
        parent_start=host.timestamp();submit('helper_sentinel',1,ordinary);ordinary.synchronize();parent_end=host.timestamp()
        r['window']=session._report_call(lib.s5ro_end_window,(token,window,1),retry=False);window=None
        ro._validate_window(r['window'],limits,1)
        def census(a,b):return session._report_call(lib.s5ro_activity_census,(token,a,b))
        r['positive']=census(positive_start,positive_end)
        r['negative_unoffered']=census(neg_start,neg_end);r['negative_primary']=census(parent_start,parent_end)
        w=r['window'];r['all_activity']=census(w['window']['start_ns'],w['window']['end_ns'])
        for item in intervals:
            item['census']=census(item.pop('start'),item.pop('end'));r['intervals'].append(item)
        r['serialized_stable']=before==s.graph_bytes(lib,graph.value)
        r['output_exact']=(bool(torch.all(dst==7+p['graph']['kernel']).item())
            and bool(torch.all(z.cpu()==0).item()) and bool(torch.all(f.cpu()==0).item())
            and int(tmp[0])==7+p['families']['eager_pre']+2*p['families']['eager_post']
            and bool(torch.all(tmp[1:]==7).item()))
    except Exception as exc:
        r['missing'].append('phase_exception');r['error_type']=type(exc).__name__
        r['error_stack']=[dict(file=Path(x.filename).name,line=x.lineno,function=x.name) for x in traceback.extract_tb(exc.__traceback__)]
    finally:
        errors=[]
        try:torch.cuda.synchronize()
        except Exception:errors.append('drain')
        try:
            if session:
                if window:session._report_call(binding._lib.s5ro_end_window,(session._token,window,1),retry=False)
                session.close()
        except Exception:errors.append('observer_close')
        try:
            if owner:s.checked(gl.lib.vmg_close(owner),'graph_close')
            if split:split.close()
        except Exception:errors.append('native_close')
        r['cleanup_ok']=not errors;r['cleanup_errors']=errors
    r['grade']=case_grade(r);s.write(out/('rank'+str(rank)+'.json'),r);return r


def probe(modules,native,out):
    r=dict(schema='vm<vm-id>-job8/1',image=s.IMAGE,cases=[],regressions={},failures=[],missing=[],memory_fraction=.125)
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
        for rank in range(4):r['cases'].append(case(torch,ro,gc,native,receipt,out,rank))
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
    result.update(schema='vm<vm-id>-job8/1',process_rc=process_rc,memory=m,cases=g['cases'],
       regressions=raw.get('regressions',{}),scope='declared graph and eager synthetic families; no engine graphId-0 waiver')
    s.write(out/'result.json',result);print('VMID_JOB8 '+result['verdict']);return result['rc']


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('probe','monitor','finalize'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--modules',type=Path);p.add_argument('--native',type=Path);p.add_argument('--process-rc',type=int,default=2);a=p.parse_args()
    if a.mode=='monitor':return s.monitor(a.out,'flashnext-job8')
    if a.mode=='finalize':return finalize(a.out,a.process_rc)
    return probe(a.modules,a.native,a.out)
if __name__=='__main__':sys.exit(main())
