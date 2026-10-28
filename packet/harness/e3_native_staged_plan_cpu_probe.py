#!/usr/bin/env python3
"""Actual mounted Scheduler/Request CPU split proof; no allocation/model/CUDA.

A real Scheduler instance is made with __new__, with explicit CPU configuration
fields for its pure alignment method. Engine/scheduler construction is not
claimed. Methods, kernels and ready flags are never replaced.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS
import time
from e3_native_import_probe import check,error


def collect():
    import torch
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.flash_e3 import scheduler_hooks as sh
    from vllm.models.qwen4_exp.nvidia import qsa
    assert not torch.cuda.is_initialized()
    method=Scheduler._mamba_block_aligned_split
    methods=(method,sh.PrefillAdmissionController._aligned_split,sh.staged_chunk_plan)
    cases=[]
    begin=time.monotonic_ns()
    for block in (16,176,1024,4048):
        for cap in (64,128,1024):
            for partial in (False,True):
                for initial in (0,98304):
                    request=Request(request_id='native15-plan',prompt_token_ids=list(range(98312)),
                        sampling_params=SamplingParams(max_tokens=1),pooling_params=None)
                    request.num_computed_tokens=initial
                    # Explicit source-method inputs, no Scheduler.__init__,
                    # allocator, model, engine, fake method or device owner.
                    scheduler=Scheduler.__new__(Scheduler)
                    scheduler.cache_config=NS(block_size=block)
                    scheduler.hash_block_size=176
                    scheduler.use_eagle_block_drop=False
                    scheduler.mamba_has_prefill_checkpoint_blocks=False
                    scheduler.mamba_prefill_checkpoint_alignment=block
                    scheduler.mamba_partial_cache_hit=partial
                    scheduler.mamba_fine_grained_prefix_cache=False
                    scheduler.max_num_scheduled_tokens=cap
                    scheduler.scheduler_config=NS(long_prefill_token_threshold=0)
                    before=(request.num_computed_tokens,tuple(request.all_token_ids))
                    def split(start,limit):
                        view=sh._ProjectedPrefillRequest(request,start)
                        return scheduler._mamba_block_aligned_split(view,limit)
                    plan=sh.staged_chunk_plan(initial,98311,cap,sh.QSA_STAGE_MIN_ROWS,split)
                    legacy=[]
                    legacy_position=initial
                    while legacy_position<98311:
                        size=split(legacy_position,min(cap,98311-legacy_position))
                        if size==0:break
                        legacy.append(size);legacy_position+=size
                    if block==4048 and cap==1024 and initial==0:
                        assert plan, 'production-sized block/cap request must have a native plan'
                        if partial:
                            assert 32 in legacy and min(plan)>=64
                            assert plan[-3:]==(992,64,103)
                    position=initial
                    for size in plan:
                        assert 64<=size<=cap
                        assert split(position,size)==size
                        position+=size
                    if plan:assert position==98311
                    assert before==(request.num_computed_tokens,tuple(request.all_token_ids))
                    cases.append(dict(block_size=block,hash_block_size=176,cap=cap,
                        partial_cache=partial,initial_frontier=initial,plan=list(plan),
                        legacy_short_chunks=[n for n in legacy if n<64],
                        verdict='NATIVE_PLAN' if plan else 'STOCK_BEFORE_ALLOCATION',
                        source_unchanged=True,alignment_exact=True))
    # Runtime shape where terminal7 is legal to stock alignment but cannot
    # execute native attention. The new plan leaves64 without changing cap.
    plan=sh.staged_chunk_plan(0,98311,1024,64,lambda start,n:n)
    assert plan[-2:]==(967,64)
    negative=[]
    for name in ('_flash_s5_staged_operation','_flash_s5_sparse_staged_operation'):
        for rows in (1,7,63,64,65):
            helper=getattr(qsa,name)
            try:next(helper(None,NS(_qsa_kv_offload=True,_qsa_stage_min_rows=64),
                None,None,None,None,rows,True,None))
            except RuntimeError as exc:message=str(exc)
            else:raise AssertionError('missing CPU metadata did not refuse')
            expected=('S5 operation requires the staged dense row threshold' if rows<64
                else 'S5 operation requires CPU host metadata; no device readback')
            assert message==expected
            negative.append(dict(helper=name,rows=rows,minimum=64,error=message))
    assert sh.QSA_STAGE_MIN_ROWS==qsa._QSA_STAGE_MIN_ROWS==64
    assert methods==(Scheduler._mamba_block_aligned_split,
        sh.PrefillAdmissionController._aligned_split,sh.staged_chunk_plan)
    assert not torch.cuda.is_initialized()
    return dict(verdict='PASS',actual_request_type=Request.__module__+'.'+Request.__name__,
        actual_scheduler_method=Scheduler.__module__+'.'+Scheduler.__name__+'._mamba_block_aligned_split',
        synthetic_scheduler_config=True,scheduler_initializer_executed=False,
        cases=cases,planned=sum(bool(x['plan']) for x in cases),
        stock_refused=sum(not x['plan'] for x in cases),guard_boundaries=negative,
        native14_short_tail_negative_rejected=True,rebalance_example=list(plan[-2:]),
        no_allocator_constructed=True,no_model_constructed=True,no_methods_patched=True,
        no_kernels_patched=True,no_ready_flags_patched=True,cuda_initialized=False,
        elapsed_ns=time.monotonic_ns()-begin,
        source_sha256={name:hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest()
            for name,obj in (('scheduler',Scheduler),('request',Request),
                ('scheduler_hooks',sh.PrefillAdmissionController),('qsa',qsa))})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=check(json.loads(a.graph.read_text()))
    result['schema']='native15-staged-chunk-plan-cpu/1'
    try:
        if result['verdict']=='PASS':result['staged_plan']=collect()
    except Exception as exc:
        result['failures'].append('staged-chunk-plan-proof')
        result['staged_plan']=dict(verdict='FAIL',exception=error(exc))
    result['verdict']='FAIL' if result['failures'] else 'INCONCLUSIVE' if result['missing'] else 'PASS'
    result['rc']=1 if result['failures'] else 2 if result['missing'] else 0
    result['probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:result[k] for k in ('verdict','rc','counts','failures','missing','cuda_initialized','native7_negative_rejected')}))
    raise SystemExit(result['rc'])


if __name__=='__main__':main()
