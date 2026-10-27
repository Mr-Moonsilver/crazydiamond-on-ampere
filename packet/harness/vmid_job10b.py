#!/usr/bin/env python3
"""VM-only teacher collector smoke: real trace/top-k kernels, synthetic batches.

No model, graph capture, engine or four-rank qualification. No helper .so is
needed: the pinned image supplies the stock trace replay and top-k kernels.
"""
import argparse
import ast
import gc
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import traceback
import types
import __future__
import vmid_support as s

TEACHER_SHA = '3e8bfd2443a98f6df830c69983cd70b528cf37bdf0aa7514bab9668a4a7bd22c'
PINS = {
    'v1/worker/gpu/input_batch.py':'abfbb9075a80354bd4bb59d2729fcdea3c62eed827067a823761c98356670ad1',
    'v1/worker/gpu/sample/trace_replay.py':'d2a7d3b3b55379efabe4740a4b7f14030126dc11be36f6ea974164ce42316a50',
    'v1/worker/gpu/sample/logprob.py':'f47479a818192acb6b297c79fa67a3b2f9820e3492a394d5adda777e9b7c832e',
    'v1/engine/logprobs.py':'14bbe78d92a2c451d79b3d9dd9d5268bc57e16bad5079e44ce9ed2cd805e873e',
    'logprobs.py':'3cc2ed6cdd30572d29f9b293ddbc14ad318c6b5d5dc7fdd36bee2f8af259b373',
    'entrypoints/openai/completion/serving.py':'75d05ddddf303734a078a3e42fcb173fef58ea56da09bc0b5c7525cb9bbbf527',
}
CASES = ('replicated_shape', 'permuted_partial_padding', 'slot_reuse')
SCOPE = 'single-GPU teacher collector; synthetic eager batches, real stock trace/top-k; no model/graph/PP qualification'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def serializers(root):
    for path,sha in PINS.items():
        if s.sha(root/path)!=sha: raise ValueError('pinned_source_mismatch')
    env={'itertools':itertools,'Logprob':types.SimpleNamespace,
         'FlatLogprobs':type('FlatLogprobs',(),{}),'CompletionLogProbs':types.SimpleNamespace}
    for path,name in [('logprobs.py','append_logprobs_for_next_position'),
                      ('entrypoints/openai/completion/serving.py','_create_completion_logprobs')]:
        nodes=[n for n in ast.walk(ast.parse((root/path).read_text()))
               if isinstance(n,ast.FunctionDef) and n.name==name]
        if len(nodes)!=1:raise ValueError('serialization_seam')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec',
                     flags=__future__.annotations.compiler_flag),env)
    service=types.SimpleNamespace(_get_decoded_token=lambda row,token,tokenizer,**kw:'token_id:'+str(token))
    def serialize(ids,values):
        rows=[]
        env['append_logprobs_for_next_position'](rows,ids,values,[None]*len(ids),1,20)
        out=env['_create_completion_logprobs'](service,[ids[0]],rows,20,None,return_as_token_id=True)
        return out.top_logprobs[0]
    return serialize


def case_grade(r):
    f=[];m=[]
    if r.get('host_snapshot_basis')!='pinned-InputBatch-numpy-int32-bool':m.append('host_snapshot_basis')
    if r.get('name') not in CASES:m.append('case_name')
    for key in ('logits_unchanged','restored','shift_negative_rejected'):
        if r.get(key) is False:f.append(key)
        elif r.get(key) is not True:m.append(key)
    if r.get('overflow') is not False or r.get('capture_defects')!=[]:m.append('capture_incomplete')
    rows=r.get('rows',[]);traces=r.get('trace_digests',[])
    if len(traces)!=8 or len(set(traces))!=8:m.append('trace_coverage')
    expected={(trace,pos) for trace in traces for pos in range(256,512)}
    actual=set()
    for row in rows:
        key=(row.get('trace_sha256'),row.get('position'))
        if key in actual:f.append('duplicate_row')
        actual.add(key)
        if row.get('forced_token_exact') is False:f.append('forced_token_mismatch')
        elif row.get('forced_token_exact') is not True:m.append('forced_token_unproved')
        if row.get('input_position')!=row.get('total_len',0)-1:f.append('input_position')
        a,b=row.get('logprob_sha256'),row.get('serialized_sha256')
        if not all(type(x) is str and len(x)==64 and all(c in '0123456789abcdef' for c in x) for x in (a,b)):
            m.append('row_binding_missing')
        elif a!=b:f.append('row_binding_mismatch')
    if len(rows)!=2048 or actual!=expected:m.append('row_coverage')
    wanted=512 if r.get('name')=='permuted_partial_padding' else 256
    if r.get('calls')!=wanted:m.append('call_coverage')
    return s.verdict(f,m)


def grade(r):
    f=list(r.get('failures',[]));m=list(r.get('missing',[]));cases={}
    if r.get('image')!=s.IMAGE or r.get('teacher_sha256')!=TEACHER_SHA:m.append('identity')
    if r.get('source_pins')!=PINS:m.append('source_pins')
    if r.get('memory_fraction')!=.125:m.append('memory_backstop')
    if r.get('bad_readback_rejected') is not True:m.append('bad_readback_negative')
    if r.get('cleanup_ok') is not True:m.append('cleanup')
    if [c.get('name') for c in r.get('cases',[])]!=list(CASES):m.append('case_coverage')
    for row in r.get('cases',[]):
        g=case_grade(row);cases[row.get('name','unknown')]=g
        f.extend(row.get('name','unknown')+':'+x for x in g['failures'])
        m.extend(row.get('name','unknown')+':'+x for x in g['missing'])
    out=s.verdict(f,m);out['cases']=cases;return out


def run_case(torch,td,serialize,runner,states,name,out):
    from types import SimpleNamespace as NS
    import numpy as np
    td.configure_worker(runner,True,rank=2)
    trace=runner._flash_s5_teacher_trace
    salt=71 if name=='slot_reuse' else 11
    continuations={slot:[(salt+slot*13+p*3)%128 for p in range(768)] for slot in range(8)}
    for slot,tokens in continuations.items(): trace.add_request(slot,NS(trace_decode_token_ids=tokens))
    trace.apply_staged_writes()
    truths={};tops=[];unchanged=True
    order=[7,2,0,5,3,1,6,4] if name=='permuted_partial_padding' else list(range(8))
    groups=[order[:4],order[4:]] if name=='permuted_partial_padding' else [order]
    for pos in range(256,512):
        states.total_len.gpu.copy_(states.prompt_len.gpu+pos)
        for group in groups:
            n=len(group);idx=torch.tensor(group,device='cuda',dtype=torch.int32)
            batch=NS(idx_mapping=idx,expanded_idx_mapping=idx,
                logits_indices=torch.arange(n,device='cuda',dtype=torch.int64),
                cu_num_logits=torch.arange(n+1,device='cuda',dtype=torch.int32),
                expanded_local_pos=torch.zeros(n,device='cuda',dtype=torch.int32),
                num_scheduled_tokens=np.ones(n,dtype=np.int32),
                is_prefilling_np=np.zeros(n,dtype=np.bool_),
                num_reqs=n,num_reqs_after_padding=8,num_tokens=n,num_tokens_after_padding=8,
                num_draft_tokens=0,positions=(states.total_len.gpu[idx]-1))
            runner.cudagraph_manager.dispatch()  # synthetic eager descriptor, no graph claimed
            logits=torch.remainder(torch.arange(128,device='cuda',dtype=torch.float32)[None,:]*17
                                   +idx[:,None]*7+pos*3,128)/32
            before=logits.clone()
            returned=runner.sample(logits,batch,None)[0]
            unchanged=unchanged and torch.equal(before,logits)
            ids=returned.logprobs_tensors.logprob_token_ids.cpu().tolist()
            values=returned.logprobs_tensors.logprobs.cpu().tolist()
            for i,slot in enumerate(group):
                top=serialize(ids[i],values[i]);sha=td.digest(continuations[slot])
                truths[(sha,pos)]=digest([sha,pos,list(top.items())])
                if len(tops)<2: tops.append((sha,pos,top))
    evidence=td.worker_evidence(runner)
    rows=[dict(row,serialized_sha256=truths.get((row['trace_sha256'],row['position']))) for row in evidence['positions']]
    # Same coordinate with a different physical row's numerical values must fail.
    a,b=tops
    wrong=digest([a[0],a[1],list(b[2].items())])
    correct=truths[(a[0],a[1])]
    original=runner._flash_s5_teacher_observer['shadows'][0]['original']
    td.configure_worker(runner,False,rank=2)
    restored=runner.sample is original and runner.sampler.trace_replay_state is None and not hasattr(runner,'_flash_s5_teacher_observer')
    r=dict(host_snapshot_basis='pinned-InputBatch-numpy-int32-bool',name=name,rows=rows,trace_digests=[td.digest(continuations[i]) for i in range(8)],
           calls=len(evidence['calls']),overflow=evidence['overflow'],capture_defects=evidence['capture_defects'],
           logits_unchanged=unchanged,restored=restored,shift_negative_rejected=wrong!=correct,
           descriptor_basis='synthetic eager descriptor; no CUDA graphs or model',
           negative_basis='CPU checker mutations of actual readbacks; not GPU fault injection',
           slot_basis='synthetic slots reused across cases; no scheduler lifetime claim')
    r['grade']=case_grade(r);s.write(out/(name+'.json'),r)
    return r


def probe(modules,out):
    r=dict(schema='vm<vm-id>-job10b/1',image=s.IMAGE,memory_fraction=.125,cases=[],failures=[],missing=[],scope=SCOPE)
    try:
        if s.sha(modules/'teacher_decode.py')!=TEACHER_SHA:raise ValueError('teacher_source_pin')
        r['teacher_sha256']=TEACHER_SHA
        import torch
        torch.cuda.set_per_process_memory_fraction(.125,0);torch.set_num_threads(4)
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)!=(8,6):raise ValueError('device')
        import vllm
        serialize=serializers(Path(vllm.__file__).resolve().parent);r['source_pins']=dict(PINS)
        from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
        sys.path.insert(0,str(modules));import teacher_decode as td
        from types import SimpleNamespace as NS
        states=NS(max_num_reqs=8,device=torch.device('cuda:0'),
            prompt_len=NS(gpu=torch.arange(100,108,device='cuda',dtype=torch.int32)),
            total_len=NS(gpu=torch.zeros(8,device='cuda',dtype=torch.int32)))
        descriptor=NS(cg_mode=NS(name='NONE',value=0),num_tokens=8,num_reqs=8,
                      uniform_token_count=1,max_query_len=1,num_active_loras=0,num_ubatches=1)
        runner=NS(is_last_pp_rank=True,req_states=states,batch_sharder=None,
            _flash_s5_decode_probe=NS(_rpc_engine_epoch=1,route='full',route_generation=0,storage_epoch=1),
            cudagraph_manager=NS(dispatch=lambda:descriptor),
            sampler=NS(trace_replay_state=None,logprobs_mode='raw_logprobs'))
        def sample(logits,batch,grammar):
            sampled=torch.argmax(logits,dim=-1)
            runner.sampler.trace_replay_state.apply_trace(sampled,batch.idx_mapping)
            lp=compute_topk_scores(logits,20,sampled)
            return (NS(sampled_token_ids=sampled[:,None],logprobs_tensors=lp),None,None)
        runner.sample=sample
        time.sleep(2)  # ensure at least one sidecar sample of our context
        for name in CASES:r['cases'].append(run_case(torch,td,serialize,runner,states,name,out))
        state=NS(positions=[dict(batch_row=0,trace_sha256='a'*64,position=256)],capture_defects=set(),capture_overflow=False)
        td.observe_logprobs(state,(NS(logprobs_tensors=None),None,None),0)
        r['bad_readback_rejected']=state.capture_overflow and 'sampler-logprob-binding-unproved' in state.capture_defects
        torch.cuda.synchronize()
        r['allocator_peak_reserved']=torch.cuda.max_memory_reserved()
        del runner,states,state
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();r['cleanup_ok']=True
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
    result.update(schema='vm<vm-id>-job10b/1',process_rc=process_rc,memory=m,cases=g['cases'],scope=SCOPE)
    s.write(out/'result.json',result);print('VMID_JOB10B '+result['verdict']);return result['rc']


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('probe','monitor','finalize'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--modules',type=Path);p.add_argument('--process-rc',type=int,default=2);a=p.parse_args()
    if a.mode=='monitor':return s.monitor(a.out,'flashnext-job10b')
    if a.mode=='finalize':return finalize(a.out,a.process_rc)
    return probe(a.modules,a.out)
if __name__=='__main__':sys.exit(main())
