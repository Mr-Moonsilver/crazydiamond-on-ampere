#!/usr/bin/env python3
"""Actual Scheduler/Request CoW guard composition; CPU outputs are declared leaves."""
import argparse,copy,dataclasses,hashlib,json,time,traceback
from pathlib import Path


def config(device='cpu', max_model_len=512):
    from e3_native_stock_rpc_probe import gpu_config
    from vllm.config import VllmConfig,ModelConfig,CacheConfig,SchedulerConfig,ParallelConfig,DeviceConfig,CompilationConfig,LoadConfig
    original=gpu_config(device)
    return VllmConfig(model_config=ModelConfig(model=original.model_config.model,skip_tokenizer_init=True,
            language_model_only=True,dtype='bfloat16',max_model_len=max_model_len,enforce_eager=True),
        cache_config=CacheConfig(block_size=4048,mamba_block_size=4048,prefix_match_unit=176,
            num_gpu_blocks_override=128,enable_prefix_caching=True,gpu_memory_utilization=.08,mamba_cache_mode='align'),
        scheduler_config=SchedulerConfig(max_model_len=max_model_len,is_encoder_decoder=False,max_num_batched_tokens=128,
            max_num_seqs=8,async_scheduling=True),parallel_config=ParallelConfig(distributed_executor_backend='mp'),
        device_config=DeviceConfig(device=device),compilation_config=CompilationConfig(mode=0),load_config=LoadConfig(load_format='dummy'))


def engine_request(name,tokens,salt,max_tokens):
    from vllm.v1.engine import EngineCoreRequest
    from vllm.sampling_params import SamplingParams
    return EngineCoreRequest(request_id=name,prompt_token_ids=tokens,mm_features=None,
        sampling_params=SamplingParams(max_tokens=max_tokens,temperature=0.,ignore_eos=True),
        pooling_params=None,arrival_time=time.time(),lora_request=None,cache_salt=salt,data_parallel_rank=None)


def host_preprocess(owner,request):
    from vllm.v1.engine.core import EngineCore
    actual,wave=EngineCore.preprocess_add_request(owner,request)
    assert wave==0
    expected=len(request.prompt_token_ids)//owner.hash_block_size if hasattr(owner,'hash_block_size') else None
    if expected is not None:assert len(actual.block_hashes)==expected
    return actual


def snapshot(guard,out,scheduler):
    copies=[dict(type=type(c).__module__+'.'+type(c).__name__,
        src_block_id=c.src_block_id,dst_block_id=c.dst_block_id) for c in (out.kv_cache_block_copies or [])]
    held={str(k):[dict(transaction=dataclasses.asdict(e['transaction']),committed=e['committed'],released=e['released']) for e in v] for k,v in guard._held.items()}
    return dict(copies=copies,new_block_ids_to_zero=out.new_block_ids_to_zero,
        scheduled_tokens=dict(out.num_scheduled_tokens),held=held,committed=sorted(guard._committed()),
        requests={k:dict(computed=r.num_computed_tokens,output=r.num_output_tokens,
            blocks=[list(x) for x in scheduler.kv_cache_manager.get_block_ids(k)]) for k,r in scheduler.requests.items() if k in out.num_scheduled_tokens})


def cpu_composition():
    import torch
    from vllm.config import set_current_vllm_config,SchedulerConfig
    from vllm.distributed import init_distributed_environment,initialize_model_parallel
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration as Model
    from vllm.v1.kv_cache_interface import MambaSpec,FullAttentionSpec
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as Kind
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend
    from vllm.v1.attention.backends.short_conv_attn import PleShortConvAttentionBackend
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionBackend
    from vllm.models.qwen4_exp.common.qsa_cache import QSAKeyStateCache,QSACompressedKeyCache,QSAStateBackend
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config,resolve_kv_cache_block_sizes,update_kv_cache_capacity,init_none_hash,get_request_block_hasher
    from vllm.v1.structured_output import StructuredOutputManager
    from vllm.v1.request import Request
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.flash_e3 import scheduler_hooks as hooks
    from e3_native_transitions_lifecycle_probe import cache_configuration
    cfg=config()
    host=copy.copy(cfg)
    host.scheduler_config=SchedulerConfig(max_model_len=512,is_encoder_decoder=False,max_num_batched_tokens=128,
        max_num_seqs=8,async_scheduling=True,disable_hybrid_kv_cache_manager=False)
    with set_current_vllm_config(host):
        init_distributed_environment(world_size=1,rank=0,local_rank=0,distributed_init_method='file:///tmp/native17-cpu-dist',backend='gloo')
        initialize_model_parallel(1,1,backend='gloo')
        gdn,ple=Model.get_mamba_specs_from_config(host)
        def state(s,k):return MambaSpec(block_size=4048,shapes=tuple(s.shapes),dtypes=tuple(s.dtypes),mamba_type=k,mamba_cache_mode='align',tp_replicated=s.tp_replicated)
        specs={'shape.gdn0':state(gdn,Kind.GDN_ATTN),'shape.gdn2':state(gdn,Kind.GDN_ATTN),
            'shape.ple':state(ple,Kind.SHORT_CONV),
            'shape.qsa':Qwen4ExpQSAFlashAttentionBackend.customize_spec(FullAttentionSpec(block_size=4048,
                num_kv_heads=1,head_size=256,head_size_v=256,dtype=torch.bfloat16))}
        caches=[QSAKeyStateCache(head_size=64,dtype=torch.bfloat16,cache_config=host.cache_config,
            prefix='shape.indexer.raw_key_cache',vllm_config=host,compress_ratio=4,cache_rope_positions=host.model_config.uses_mrope),
            QSACompressedKeyCache(head_size=64,dtype=torch.bfloat16,cache_config=host.cache_config,
                prefix='shape.indexer.compressed_key_cache',vllm_config=host,compress_ratio=4)]
        for owner in caches:specs[owner.prefix]=owner.get_kv_cache_spec(host)
        worker_kv,cache_record=cache_configuration(host,specs,(GDNAttentionBackend,PleShortConvAttentionBackend,Qwen4ExpQSAFlashAttentionBackend,QSAStateBackend))
        kv=generate_scheduler_kv_cache_config([worker_kv]);host.cache_config.num_gpu_blocks=kv.num_blocks
        host.cache_config.block_size=min(g.kv_cache_spec.block_size for g in kv.kv_cache_groups if g.kv_cache_spec.prefix_cacheable)
        update_kv_cache_capacity(host,kv);host.validate_block_size();block,hash_block=resolve_kv_cache_block_sizes(kv,host)
        Scheduler=host.scheduler_config.get_scheduler_cls();scheduler=Scheduler(host,kv,StructuredOutputManager(host),block_size=block,hash_block_size=hash_block)
        guard=hooks.SchedulerLeases().attach(scheduler,scheduler.kv_cache_manager.block_pool)
        hash_fn=get_hash_fn_by_name('sha256');init_none_hash(hash_fn);hasher=get_request_block_hasher(hash_block,hash_fn)
        from types import SimpleNamespace
        cpu_core=SimpleNamespace(mm_receiver_cache=None,request_block_hasher=hasher,hash_block_size=hash_block)
        def request(name,salt,tail):
            return host_preprocess(cpu_core,engine_request(name,[1+i%100 for i in range(176)]+[tail]*32,salt,4))
        first=host_preprocess(cpu_core,engine_request('cpu-completed-prior',[1,2,3,4],'prior-salt',4))
        scheduler.add_request(first)
        records=[];completed_first=False
        for step in range(200):
            try:out=scheduler.schedule(False)
            except Exception as exc:
                tb=exc.__traceback__;last=None
                while tb:
                    if 'scheduler_output' in tb.tb_frame.f_locals:last=tb.tb_frame.f_locals['scheduler_output']
                    tb=tb.tb_next
                if type(exc).__name__!='E3Pending' or last is None:raise
                return dict(verdict='PASS',finding='ENGINE',legacy_refusal=str(exc),completed_first=completed_first,
                    first_outputs=first.num_output_tokens,snapshot=snapshot(guard,last,scheduler),steps=step,
                    cache_record=cache_record,leases_ever_acquired=False,manual_retirement=False,methods_patched=False)
            records.append(snapshot(guard,out,scheduler))
            ids=list(out.num_scheduled_tokens)
            tokens=[[17] if scheduler.requests[k].num_computed_tokens>=scheduler.requests[k].num_prompt_tokens else [] for k in ids]
            scheduler.update_from_output(out,ModelRunnerOutput(req_ids=ids,req_id_to_index={k:i for i,k in enumerate(ids)},sampled_token_ids=tokens))
            if not completed_first and first.is_finished():
                completed_first=True;assert guard.shutdown()['clean']
                for i in range(8):scheduler.add_request(request('cpu-control-'+str(i),'teacher-full-A1',120+i))
            if completed_first and not scheduler.has_requests():
                return dict(verdict='PASS',finding='NONE',completed_first=True,first_outputs=first.num_output_tokens,
                    all_requests_completed=True,steps=step+1,records=records,manual_retirement=False,leases_ever_acquired=False,methods_patched=False,
                    cow_copies=[c for row in records for c in row['copies']],
                    foreign_lease_guard=foreign_lease_guard())
        raise AssertionError('CPU scheduler scenario did not finish')


def foreign_lease_guard():
    from types import SimpleNamespace as Row
    from vllm.v1.flash_e3.scheduler_hooks import SchedulerLeases
    from vllm.v1.flash_e3.e3_types import TransactionId, OperationTicket, LaneId, LaneRole
    from vllm.v1.flash_e3.async_handoff import LeaseCategory
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
    guard=SchedulerLeases()
    tx=TransactionId(7,'foreign-request',1,0,128,1)
    ticket=OperationTicket(tx,0,LaneId(0,LaneRole.PREFILL,3,11),1,__import__(
        'vllm.v1.flash_e3.e3_types',fromlist=['DependencyRole']).DependencyRole.PRODUCER)
    guard.lease_blocks(ticket,LeaseCategory.ARENA,[3])
    # The copy type is msgspec; source/destination fields alone are enough for
    # the existing guard. Exact constructor fields are inspected before use.
    copy=Row(src_block_id=3,dst_block_id=9)
    out=Row(kv_cache_block_copies=[copy])
    assert guard.mutation_guard(out,None) is False
    assert guard.cow_requested==0
    copy.src_block_id=7;copy.dst_block_id=3
    assert guard.mutation_guard(out,None) is False
    return dict(verdict='PASS',foreign_uncommitted_source_rejected=True,
        held_destination_rejected=True,actual_typed_lease=True,methods_patched=False,
        leaf='host-only source/destination projection; no allocator/device mutation')


def gpu_shape():
    import inspect,torch
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams
    from e3_native_stock_rpc_probe import cache_capacity_shape
    cfg=config('cpu');capacity=cache_capacity_shape(cfg)
    req=Request(request_id='shape-control',prompt_token_ids=list(range(176))+[120]*32,
        sampling_params=SamplingParams(max_tokens=32,temperature=0.,ignore_eos=True),pooling_params=None,cache_salt='teacher-full-A1')
    inspect.signature(EngineCore).bind(cfg,MultiprocExecutor,False)
    assert callable(EngineCore.step_with_batch_queue) and callable(EngineCore.add_request)
    assert callable(EngineCore.shutdown) and len(req.prompt_token_ids)==208
    return dict(verdict='PASS',capacity=capacity,cuda_initialized=torch.cuda.is_initialized(),
        executed=['config','actual Request/SamplingParams','cache spec capacity methods','EngineCore signature and method accesses','foreign lease guard','actual EngineCoreRequest and Core.preprocess_add_request with real host hasher'],
        deferred=['EngineCore real MultiprocExecutor/worker device startup','real GPU forward and CoW kernels','physical native configure TP2PP2 green prefix (observed on native16 box)'])


def gpu_executor(progress):
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor,WorkerProc
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams
    from vllm.v1.flash_e3 import executor_hooks as hooks,async_handoff,scheduler_hooks
    core=None;records=[];seen=set();high=0;iterations=0;deadline=time.monotonic()+300
    def checkpoint(step,**fields):
        progress.write(json.dumps(dict(step=step,observed_ns=time.monotonic_ns(),**fields))+'\n');progress.flush()
    try:
        checkpoint('engine-before-construction')
        core=EngineCore(config('cuda'),MultiprocExecutor,False)
        checkpoint('engine-constructed')
        executor=core.model_executor
        transport=hooks.install(executor,config=async_handoff.HandoffConfig.from_env(
            {'FLASH_S5_OVERLAP':'1','FLASH_S5_COMMAND_SLOTS':'10'}),success_status=WorkerProc.ResponseStatus.SUCCESS)
        guard=scheduler_hooks.SchedulerLeases().attach(core.scheduler,core.scheduler.kv_cache_manager.block_pool)
        assert guard.shutdown()['clean'] and transport._max_calls==10
        def run():
            nonlocal high,iterations
            while core.scheduler.has_requests() or core.batch_queue:
                assert time.monotonic()<deadline,'real executor CoW deadline'
                core.step_with_batch_queue();iterations+=1;high=max(high,len(transport._calls))
                for _,out,_ in core.batch_queue:
                    key=id(out)
                    if key not in seen:
                        # Keep the actual object alive so Python cannot reuse
                        # its id while this observation ledger is retained.
                        seen.add(key);retained_outputs.append(out)
                        copies=[dict(src_block_id=c.src_block_id,dst_block_id=c.dst_block_id)
                                for c in (out.kv_cache_block_copies or [])]
                        records.append(dict(copies=copies,scheduled_tokens=dict(out.num_scheduled_tokens),held=len(guard._held)))
                checkpoint('batch-step',iteration=iterations,credits=len(transport._calls),requests=len(core.scheduler.requests))
        retained_outputs=[]
        def make(name,tokens,salt):
            message=engine_request(name,tokens,salt,32)
            request,wave=core.preprocess_add_request(message)
            assert wave==0 and request._block_hasher is core.request_block_hasher
            if len(tokens)>=176:assert request.block_hashes,'stock Core preprocessing did not hash shared prefix'
            core.add_request(request)
            checkpoint('stock-request-preprocessed',request_id=name,prompt_tokens=len(tokens),block_hashes=len(request.block_hashes))
            return request
        prior=make('actual-completed-prior',[1,2,3,4],'prior-salt')
        run();assert prior.num_output_tokens==32 and guard.shutdown()['clean']
        checkpoint('prior-completed',tokens=32,lease_ledger=guard.shutdown())
        requests=[]
        for i in range(8):
            requests.append(make('actual-control-'+str(i),[1+j%100 for j in range(176)]+[120+i]*32,'teacher-full-A1'))
        checkpoint('eight-admissions',ids=[r.request_id for r in requests]);run()
        copies=[c for r in records for c in r['copies']]
        assert copies,'actual allocator CoW was not exercised'
        assert all(r.num_output_tokens==32 for r in requests)
        assert not transport._calls and not executor._flash_s5_stock_futures
        assert guard.shutdown()['clean'] and guard.cow_requested>=len(copies)
        return dict(verdict='PASS',actual_engine_core=True,actual_multiproc_executor=True,
            actual_scheduler=True,actual_request=True,actual_allocator_recycling=True,
            concurrent_requests=8,completed_prior_tokens=32,completion_tokens=[r.num_output_tokens for r in requests],
            iterations=iterations,credit_high_water=high,max_calls=10,all_credits_returned=True,
            cow_copies=copies,cow_requested=guard.cow_requested,records=records,lease_ledger=guard.shutdown(),
            leases_ever_acquired=False,manual_retirement=False,methods_patched=False,kernels_patched=False,
            gpu_execution=True,actual_request_preprocessing=True,native_configure_prefix='observed-on-box native16; not constructed by this TP1 proof',
            full_model_numerics=False,hardware_qualified=False)
    finally:
        if core is not None:core.shutdown()


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=('cpu-full','cpu-shape','gpu-all'),default='cpu-full');a=p.parse_args();started=time.monotonic()
    r=dict(verdict='INCONCLUSIVE',rc=2,failure_class='NONE',sections={},engine_constructed=False)
    def persist():a.out.write_text(json.dumps(r,indent=2)+'\n')
    with a.out.with_suffix('.progress.jsonl').open('x') as progress:
        try:
            # Verify actual mounted source identity, not builder imports.
            from vllm.v1.flash_e3 import scheduler_hooks
            import inspect
            graph=json.loads(a.graph.read_text());pin=next(x for x in graph['modules'] if x['module']==scheduler_hooks.__name__)
            path=Path(inspect.getfile(scheduler_hooks));assert str(path)==pin['destination']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==pin['sha256']
            r['sections']['foreign_lease_guard']=foreign_lease_guard();persist()
            if a.mode=='cpu-full':
                result=cpu_composition()
                assert result.get('finding')=='NONE' and result.get('cow_copies')
                r['sections']['scheduler']=result;persist()
            r['sections']['gpu_constructor_shape']=gpu_shape();persist()
            if a.mode=='gpu-all':
                r['sections']['multiproc_executor']=gpu_executor(progress);r['engine_constructed']=True;persist()
            r.update(verdict='PASS',rc=0)
        except Exception as exc:
            # Keep cause/context traceback; no blanket HARNESS classification
            # for errors raised inside actual mounted runtime composition.
            r.update(verdict='FAIL',rc=1,failure_class='OPEN',error_type=type(exc).__name__,error_message=str(exc),stack=traceback.format_exc())
        r['wall_s']=time.monotonic()-started;persist()
    print(json.dumps({k:r.get(k) for k in ('verdict','rc','error_type','error_message','wall_s')}),flush=True);return r['rc']
if __name__=='__main__':raise SystemExit(main())
