#!/usr/bin/env python3
"""Connected actual-object transition scenarios; unfinished until self-run.

No GO is issued by this file. The output distinguishes the exact observed
contract from engine/model/TP/PP transitions which one card cannot qualify.
"""
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace as NS


def connected_budget_allocation():
    """Real Request -> planner -> allocator capture -> unsubmitted cancel/reuse.

    There are deliberately no fabricated worker reports or publication calls.
    The abort is the actual unsubmitted admission branch, not normal completion.
    """
    import torch
    from vllm.config import (CacheConfig,ModelConfig,VllmConfig,SchedulerConfig,
        ParallelConfig,DeviceConfig,CompilationConfig)
    from vllm.sampling_params import SamplingParams
    from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpConfig,Qwen4ExpTextConfig
    from vllm.v1.structured_output import StructuredOutputManager
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.request import Request,RequestStatus
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import init_none_hash,get_request_block_hasher
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.sched.interface import PauseState
    from vllm.v1.core.sched.request_queue import create_request_queue,SchedulingPolicy
    from vllm.v1.kv_cache_interface import KVCacheConfig,KVCacheGroupSpec,MambaSpec,FullAttentionSpec
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as Kind
    from vllm.v1.flash_e3 import scheduler_hooks as sh,admission_metadata as am,async_handoff as ah
    from vllm.v1.flash_e3.e3_types import E3ContractError

    hash_fn=get_hash_fn_by_name('sha256')
    init_none_hash(hash_fn)
    hasher=get_request_block_hasher(176,hash_fn)
    block=4048
    groups=[KVCacheGroupSpec(layer_names=['matrix.gdn'],kv_cache_spec=MambaSpec(
        block_size=block,shapes=((4,4),(2,4)),dtypes=(torch.float32,torch.float32),
        mamba_type=Kind.GDN_ATTN,mamba_cache_mode='align')),
        KVCacheGroupSpec(layer_names=['matrix.ple'],kv_cache_spec=MambaSpec(
        block_size=block,shapes=((4,4),),dtypes=(torch.float32,),
        mamba_type=Kind.SHORT_CONV,mamba_cache_mode='align')),
        KVCacheGroupSpec(layer_names=['matrix.qsa'],kv_cache_spec=FullAttentionSpec(
        block_size=block,num_kv_heads=1,head_size=256,dtype=torch.bfloat16))]
    config=KVCacheConfig(num_blocks=6,kv_cache_tensors=[],kv_cache_groups=groups)
    import tempfile
    directory=Path(tempfile.mkdtemp(prefix='matrix-model-',dir='/tmp'))
    text=Qwen4ExpTextConfig(vocab_size=248320,max_position_embeddings=262144,
        ple_layer_ids=[1],ngram_size=3,eos_token_id=248044)
    Qwen4ExpConfig(text_config=text,architectures=['Qwen4ExpForConditionalGeneration']).save_pretrained(directory)
    model_config=ModelConfig(model=str(directory),skip_tokenizer_init=True,
        language_model_only=True,dtype='bfloat16',max_model_len=262144,enforce_eager=True)
    cache=CacheConfig(block_size=block,mamba_block_size=block,enable_prefix_caching=True,
        prefix_match_unit=176,mamba_cache_mode='align')
    cache.num_gpu_blocks=config.num_blocks
    schedule=SchedulerConfig(max_model_len=262144,is_encoder_decoder=False,
        max_num_batched_tokens=1024,max_num_seqs=10,scheduler_reserve_full_isl=False)
    vllm_config=VllmConfig(model_config=model_config,cache_config=cache,
        scheduler_config=schedule,parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device='cpu'),compilation_config=CompilationConfig(mode=0))
    scheduler=Scheduler(vllm_config,config,StructuredOutputManager(vllm_config),
        block_size=block,hash_block_size=176)
    manager=scheduler.kv_cache_manager
    assert type(manager) is KVCacheManager
    assert scheduler.mamba_partial_cache_hit is True
    leases=sh.SchedulerLeases().attach(scheduler,manager.block_pool)
    frontier=ah.CommittedFrontier()
    controller=sh.PrefillAdmissionController(scheduler,leases=leases,frontier=frontier,
        engine_epoch=73,chunk_tokens=1024,ranks=(0,1,2,3),minimum_stage_rows=64).attach()
    provider=am.install_admission_metadata(scheduler)
    methods=(KVCacheManager.allocate_slots,KVCacheManager.free,KVCacheManager.cache_blocks,
        Scheduler._mamba_block_aligned_split,sh.PrefillAdmissionController.admit,
        am.NativeAdmissionMetadata.capture)
    scenarios=[];seen=[]
    for seq in (1,2):
        request=Request(request_id='matrix-request-'+str(seq),prompt_token_ids=[seq]*98312,
            sampling_params=SamplingParams(max_tokens=1),pooling_params=None,block_hasher=hasher)
        scheduler.add_request(request)
        assert request.block_hashes
        initial_free=manager.block_pool.get_num_free_blocks()
        scheduler.max_num_scheduled_tokens=63
        assert controller.admit(scheduler) is None
        assert manager.block_pool.get_num_free_blocks()==initial_free
        assert all(not row for row in manager.get_block_ids(request.request_id))
        assert scheduler.waiting.peek_request() is request
        scheduler.max_num_scheduled_tokens=1024
        record=controller.admit(scheduler)
        assert record is not None and record.chunk_tokens==1024
        assert provider.captures==seq
        ids=tuple(sorted({b for row in record.block_tables for b in row if b>0}))
        assert ids
        source=controller.source_hold(record.transaction)
        assert source is not None and source.released is False
        scheduler.max_num_scheduled_tokens=128
        assert controller.admit(scheduler) is None
        ordinary_step=scheduler.schedule()
        assert request.request_id not in ordinary_step.num_scheduled_tokens
        assert ordinary_step.total_num_scheduled_tokens==0
        assert controller.parked.admission is record and record.chunk_tokens==1024
        assert record.transaction.chunk_end==1024
        # No worker was opened/submitted. This is the real supported abort
        # branch; manufacturing completion or publishing here is forbidden.
        assert controller.parked.bound is False
        returned=controller.cancel(record.transaction,never_submitted=True)
        assert returned is request and controller.parked is None
        assert source._retired and not frontier.committed_blocks()
        finished=scheduler.finish_requests(request.request_id,RequestStatus.FINISHED_ABORTED)
        assert finished==[request] and request.request_id not in scheduler.requests
        assert manager.block_pool.get_num_free_blocks()==initial_free
        scenarios.append(dict(request_index=seq,admitted_rows=1024,
            post_admit_budget=128,admission_immutable=True,refused63_before_allocation=True,
            actual_stock_schedule_executed=True,parked_native_not_truncated=True,actual_scheduler_finish_executed=True,
            real_metadata_capture=True,unsubmitted_cancel=True,real_allocator_free=True,
            physical_ids=list(ids),reused_from_prior=sorted(set(ids)&set(seen))))
        seen.extend(ids)
    assert scenarios[1]['reused_from_prior']
    assert methods==(KVCacheManager.allocate_slots,KVCacheManager.free,KVCacheManager.cache_blocks,
        Scheduler._mamba_block_aligned_split,sh.PrefillAdmissionController.admit,
        am.NativeAdmissionMetadata.capture)
    return dict(verdict='PASS',scenario='connected-cold-admission-budget-abort-reuse',scenarios=scenarios,
        actual_types=[Request.__module__+'.'+Request.__name__,type(scheduler).__module__+'.'+type(scheduler).__name__,
            type(manager).__module__+'.'+type(manager).__name__,type(manager.block_pool).__module__+'.'+type(manager.block_pool).__name__,
            type(provider).__module__+'.'+type(provider).__name__],methods_patched=False,
        completion_reports_constructed=False,manual_frontier_retirement=False,manual_publication=False,
        box_evidence_scope='cold-request preallocation budget eligibility and unsubmitted source/allocator ownership only',
        deferred=['normal completed-chunk worker closure','actual stock final-token forward/sampling',
            'prefix-cache hit state','full39group resource pressure','TP/PP'],
        fixture_leaves=[
            dict(leaf='real Scheduler/ModelConfig/VllmConfig constructors with a local metadata-only Qwen config and CPU device',replaces='EngineCore startup with production weights and TP2PP2 workers',
                equivalence='actual initialized scheduler, queues and allocator; no fabricated capability or skipped constructor; loaded model and distributed execution are not claimed',
                more_convenient_for_full_engine=True,box_evidence=False),
            dict(leaf='three real cache-spec groups with small Mamba tensor shapes and six physical allocator blocks',
                replaces='39 production groups and their configured capacity',
                equivalence='same real allocator/queues/capture/retirement; forces physical reuse, not growing IDs; resource-pressure equivalence is not claimed',
                more_convenient_for_full_engine=True,box_evidence=False),
            dict(leaf='native cancellation caller',replaces='EngineCore driver abort action before worker open',
                equivalence='actual never_submitted cancellation method, unbound hold verified; no completion/report/publication fabricated',
                more_convenient_for_full_engine=True,box_evidence=False)],
        engine_qualified=False,hardware_qualified=False)


class EngineCounterexample(RuntimeError):
    pass


def collect():
    return connected_budget_allocation()


def main():
    import argparse,json,time,traceback
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=('cpu-full','cpu-shape','gpu-all'),default='cpu-full')
    args=parser.parse_args()
    started=time.monotonic()
    result=dict(schema='native15-connected-transitions/1',verdict='INCONCLUSIVE',rc=2,
        mode=args.mode,scenarios={},failures=[],missing=[],failure_class='NONE',
        matrix_complete=False,box_go_allowed=False,gpu_execution=False,methods_patched=False,
        deferred=['completed native interval -> stock final token -> full model sampling',
            'real dispatcher/bridge -> model layer continuation -> completion on GPU',
            'actual cached GPU state hits, copy-wave cancellation/refill and paged-report cancellation',
            'full rule4 sampler/control/candidate plumbing'])
    try:
        graph=json.loads(args.graph.read_text());assert graph['verdict']=='PASS'
        leaves=Path(__file__).with_name('fixture-leaves.json')
        result['fixture_leaves']=json.loads(leaves.read_text())
        assert result['fixture_leaves']['clears_box_hold'] is False
        if args.mode=='gpu-all':
            import torch
            assert torch.cuda.device_count()==1
            torch.cuda.set_per_process_memory_fraction((2*1024**3)/torch.cuda.get_device_properties(0).total_memory,0)
        result['scenarios']['budget_allocation']=collect()
        # Source-origin proof uses actual imported classes, not builder aliases.
        import sys,torch
        if args.mode!='gpu-all':assert not torch.cuda.is_initialized()
        if args.mode in ('cpu-shape','gpu-all'):
            result['scenarios']['sampler_cpu_shape']=sampler_cpu_shape(
                cpu_device_probe=args.mode=='cpu-shape')
            result['predicate_selftest']=predicate_selftest()
        if args.mode=='gpu-all':
            result['scenarios']['sampler_rule4']=connected_sampler_rule4()
            result['gpu_execution']=True
        active={r['module']:r for r in graph['modules'] if r['active_mount']}
        imported=[]
        for name in ('vllm.v1.flash_e3.scheduler_hooks','vllm.v1.flash_e3.admission_metadata',
            'vllm.v1.core.block_pool','vllm.v1.core.sched.scheduler','vllm.v1.core.single_type_kv_cache_manager'):
            mod=sys.modules[name];actual=Path(mod.__file__).resolve();spec=active[name]
            assert actual==Path(spec['destination']).resolve()
            assert hashlib.sha256(actual.read_bytes()).hexdigest()==spec['sha256']
            imported.append(dict(module=name,origin=str(actual),sha256=spec['sha256']))
        result.update(verdict='PASS',rc=0,imported=imported,cuda_initialized=torch.cuda.is_initialized())
        if args.mode=='gpu-all':
            recomputed=regrade_sampler_rule4(result['scenarios']['sampler_rule4'])
            if recomputed['verdict']!='PASS':raise EngineCounterexample('rule4 raw numeric plumbing regrade failed')
            result['scenarios']['sampler_rule4_regrade']=recomputed
    except Exception as exc:
        result['failure_class']='ENGINE' if isinstance(exc,EngineCounterexample) else 'HARNESS'
        result.update(verdict='FAIL',rc=1)
        result['failures']=[result['failure_class']+':'+type(exc).__name__]
        result['exception']=dict(type=type(exc).__name__,message=str(exc)[:1024],
            stack=[dict(file=Path(f.filename).name,function=f.name,line=f.lineno)
                for f in traceback.extract_tb(exc.__traceback__)[-12:]])
    result['wall_s']=time.monotonic()-started
    result['probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:result[k] for k in ('schema','verdict','rc','failure_class','failures','matrix_complete','wall_s')}),flush=True)
    return result['rc']



def sampling_batch(torch, device, request_ids, slots, step):
    """Construct the actual InputBatch; also fully executable in CPU-shape mode."""
    import numpy as np
    from vllm.v1.worker.gpu.input_batch import InputBatch
    n=len(request_ids)
    indices=torch.tensor(slots,dtype=torch.int32,device=device)
    starts_np=np.arange(n+1,dtype=np.int32)
    starts=torch.tensor(starts_np,dtype=torch.int32,device=device)
    positions=torch.full((n,),3+step,dtype=torch.int64,device=device)
    seq=torch.full((n,),4+step,dtype=torch.int32,device=device)
    return InputBatch(req_ids=list(request_ids),num_reqs=n,num_reqs_after_padding=n,
        idx_mapping=indices,idx_mapping_np=np.array(slots,dtype=np.int32),
        expanded_idx_mapping=indices,expanded_local_pos=torch.zeros(n,dtype=torch.int32,device=device),
        num_scheduled_tokens=np.ones(n,dtype=np.int32),num_tokens=n,num_tokens_after_padding=n,
        num_draft_tokens=0,num_draft_tokens_per_req=None,query_start_loc=starts,
        query_start_loc_np=starts_np,seq_lens=seq,seq_lens_cpu_upper_bound=seq.cpu(),
        dcp_local_seq_lens=None,num_computed_tokens_np=np.full(n,3+step,dtype=np.int32),
        prefill_len_np=np.full(n,4,dtype=np.int32),num_computed_prefill_tokens_np=np.full(n,4,dtype=np.int32),
        is_prefilling_np=np.zeros(n,dtype=bool),has_prefill=False,
        input_ids=torch.ones(n,dtype=torch.int32,device=device),positions=positions,
        is_padding=torch.zeros(n,dtype=torch.bool,device=device),
        logits_indices=torch.arange(n,dtype=torch.int64,device=device),
        cu_num_logits=starts,cu_num_logits_np=starts_np,has_structured_output_reqs=False,prompt_lens=None)


def sampler_host_constructor_path(*, cpu_device_probe):
    """Execute CUDA-only lazy host imports even on the CPU platform.

    The stock capability check returns before importing FlashInfer on CPU.
    Import its real backend explicitly, without changing that check/platform.
    UVA allocation still needs a driver; record an actual CPU-device attempt
    rather than building a shim RequestState to bypass it.
    """
    import ast, importlib, json, os, textwrap
    import torch
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState
    modules=[importlib.import_module(name) for name in (
        'flashinfer','flashinfer.jit.env','vllm.v1.attention.backends.flashinfer')]
    env=modules[1]
    cache=Path(env.FLASHINFER_CACHE_DIR)
    workspace=Path(env.FLASHINFER_WORKSPACE_DIR)
    for path in (cache,workspace):
        assert path.is_relative_to('/tmp'), ('cache outside executable tmpfs',str(path))
        path.mkdir(parents=True,exist_ok=True)
        # Actual access catches readonly/cache-policy errors, not a name test.
        with (path/'matrix1b-host-cache-check.json').open('x') as out:
            json.dump({'cache_write':'PASS'},out)
    # Constructor call targets are resolved from the real functions' globals.
    constructors=[]
    pending=[Sampler,RequestState];seen=set()
    while pending:
        cls=pending.pop()
        if cls in seen:continue
        seen.add(cls);method=cls.__init__
        tree=ast.parse(textwrap.dedent(inspect.getsource(method)))
        calls=[]
        for node in ast.walk(tree):
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name):continue
            value=method.__globals__.get(node.func.id)
            if inspect.isclass(value) and value.__module__.startswith('vllm.'):
                calls.append(value.__module__+'.'+value.__name__)
                pending.append(value)
        source=Path(inspect.getfile(cls))
        constructors.append(dict(name=cls.__module__+'.'+cls.__name__,
            signature=str(inspect.signature(cls)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            constructor_calls=sorted(set(calls)),host_module_import_executed=True,
            full_constructor_execution='GPU owner-run (UVA/device allocation)'))
    cpu_attempt={'attempted':False,'reason':'GPU owner path executes the actual CUDA constructors below'}
    if cpu_device_probe:
        try:RequestState(8,512,8,0,64,torch.device('cpu'))
        except RuntimeError as exc:
            message=str(exc)
            assert any(s in message.lower() for s in ('nvidia driver','cuda driver','pin_memory','pin memory','accelerator')),message
            assert torch.cuda.is_initialized() is False
            cpu_attempt=dict(attempted=True,constructed=False,exception=type(exc).__name__,
                message=message[:512],no_shim=True,
                reason='real RequestState CPU construction reaches pinned/UVA allocation requiring a CUDA driver')
        else:
            raise AssertionError('CPU RequestState construction unexpectedly passed; extend CPU constructor execution')
    return dict(verdict='PASS',lazy_host_imports_executed=[m.__name__ for m in modules],
        cache=str(cache),workspace=str(workspace),actual_cache_write=True,
        constructor_index=constructors,cpu_constructor_attempt=cpu_attempt,
        constructors_fully_executed_on_cpu=False,platform_patched=False,
        scope='host import/cache side effects plus actual CPU constructor boundary; full CUDA constructor path owner-run')


def sampler_cpu_shape(*, cpu_device_probe=True):
    import torch
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState
    from vllm.v1.worker.gpu.input_batch import post_update
    from vllm.v1.core.sched.scheduler import Scheduler
    host_constructor=sampler_host_constructor_path(cpu_device_probe=cpu_device_probe)
    rows=[]
    for step in (0,255):
        batch=sampling_batch(torch,'cpu',['shape-'+str(i) for i in range(8)],list(range(7,-1,-1)),step)
        # Actual accesses to every InputBatch attribute the sampler reads.
        actual={name:getattr(batch,name) for name in ('expanded_idx_mapping','idx_mapping',
            'idx_mapping_np','cu_num_logits_np','expanded_local_pos','positions','logits_indices',
            'input_ids','seq_lens','cu_num_logits','num_reqs')}
        assert actual['seq_lens'].tolist()==[4+step]*8
        assert actual['positions'][actual['logits_indices']].tolist()==[3+step]*8
        assert actual['idx_mapping'].tolist()==list(range(7,-1,-1))
        rows.append(dict(step=step,actual_batch_type=type(batch).__module__+'.'+type(batch).__name__,
            seq_lens=actual['seq_lens'].tolist(),slots=actual['idx_mapping'].tolist()))
    methods={name:str(inspect.signature(obj)) for name,obj in (
        ('Sampler',Sampler),('Sampler.__call__',getattr(Sampler,'__call__')),
        ('Sampler.add_request',getattr(Sampler,'add_request')),('RequestState',RequestState),
        ('post_update',post_update),('Scheduler._update_request_with_output',getattr(Scheduler,'_update_request_with_output')))}
    return dict(verdict='PASS',cases=rows,actual_method_signatures=methods,
        host_constructor_path=host_constructor,
        gpu_owned_fixtures_constructed=False,gpu_execution=False,
        scope='CPU-shape: actual imports/method access and full InputBatch CPU fixture construction; GPU RequestState/Sampler allocation deferred to owner')


def connected_sampler_rule4():
    """Real GPU sampling/state-update -> real Request output -> v4 arithmetic.

    Synthetic logits and a short precomputed prefix are convenient model-data
    leaves. This proves plumbing, not actual model numerics or full row2.
    """
    import copy
    import numpy as np
    import torch
    import e3_teacher_decode_rule4 as rule
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request
    from vllm.v1.worker.gpu.states import RequestState
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.input_batch import post_update
    from vllm.v1.core.sched.scheduler import Scheduler
    scheduler=Scheduler.__new__(Scheduler);scheduler.max_model_len=512
    device=torch.device('cuda:0');runs={};sampled_digests={}
    for name in ('full-A1','full-A2','full-C','split-B'):
        state=RequestState(8,512,8,0,64,device)
        sampler=Sampler(8,64,device,state,logprobs_mode='raw_logprobs')
        requests=[];slots=[]
        for i in range(8):
            params=SamplingParams(max_tokens=256,temperature=0.,logprobs=20,seed=17)
            request=Request(request_id=name+'-'+str(i),prompt_token_ids=[1,2,3,4],
                sampling_params=params,pooling_params=None)
            request.num_computed_tokens=3  # declared synthetic precomputed prefix
            state.add_request(request.request_id,4,list(request.all_token_ids),3,256)
            slot=state.req_id_to_index[request.request_id]
            sampler.add_request(slot,4,params)
            requests.append(request);slots.append(slot)
        state.apply_staged_writes();sampler.apply_staged_writes()
        assert not sampler.needs_logits_processing[np.array(slots,dtype=np.int32)].any()
        rows={};tokens=[]
        for step in range(256):
            batch=sampling_batch(torch,device,[r.request_id for r in requests],slots,step)
            logits=torch.linspace(2.,-6.,64,device=device).repeat(8,1)
            # Distinct, bounded per-stream/step inputs; candidate is a small
            # synthetic perturbation, not a purported split engine forward.
            logits=logits+torch.arange(8,device=device).float()[:,None]*.001
            if name=='split-B':logits[:,1]+=.01
            output=sampler(logits,batch)
            assert output.logprobs_tensors is not None
            lp=output.logprobs_tensors
            ids=lp.logprob_token_ids.cpu().tolist();values=lp.logprobs.cpu().tolist()
            sampled=output.sampled_token_ids.cpu().reshape(-1).tolist();tokens.append(sampled)
            post_update(batch.idx_mapping,state.num_computed_tokens.gpu,state.last_sampled_tokens,
                sampler.penalties_state.output_bin_counts,output.sampled_token_ids,
                output.num_sampled,output.num_rejected,batch.query_start_loc,
                state.all_token_ids.gpu,state.total_len.gpu)
            for i,request in enumerate(requests):
                returned,stopped=scheduler._update_request_with_output(request,[int(sampled[i])])
                assert returned==[sampled[i]] and stopped is (step==255)
                top={('token_id:'+str(token)):float(value) for token,value in zip(ids[i],values[i])}
                assert 20<=len(top)<=21
                rows[(i,step)]={'top':top}
        torch.cuda.synchronize()
        assert all(r.num_output_tokens==256 for r in requests)
        assert state.num_computed_tokens.gpu[torch.tensor(slots,device=device)].cpu().tolist()==[259]*8
        assert state.total_len.gpu[torch.tensor(slots,device=device)].cpu().tolist()==[260]*8
        runs[name]=rows;sampled_digests[name]=hashlib.sha256(__import__('json').dumps(tokens).encode()).hexdigest()
        for request in requests:assert state.remove_request(request.request_id) is not None
        assert state.num_reqs==0
    calibration,digest,_=rule.load_calibration()
    controls={a+'|'+b:rule.pair_metrics(runs[a],runs[b]) for a,b in rule.CONTROL_PAIRS}
    maximum={key:max(v[key] for v in controls.values()) for key in rule.CONTINUOUS+rule.COUNTING}
    allowances=rule._allowances(calibration,maximum)
    assert not rule._control_violations(calibration,controls)
    comparisons={}
    for name in ('full-A1','full-A2','full-C'):
        metrics=rule.pair_metrics(runs['split-B'],runs[name],flip_margin_limit=allowances['flip_margin'])
        assert not rule._comparison_violations(name,metrics,allowances)
        comparisons[name]=metrics
    negative=copy.deepcopy(runs['split-B']);first=next(iter(negative))
    winner=max(negative[first]['top'],key=negative[first]['top'].get)
    negative[first]['top'][winner]+=2.
    metrics=rule.pair_metrics(negative,runs['full-A1'],flip_margin_limit=allowances['flip_margin'])
    assert rule._comparison_violations('injected',metrics,allowances)
    return dict(verdict='PASS',scenario='connected-short-sampler-state-request-rule4',
        raw_numeric_rows={name:[dict(stream=key[0],position=key[1],**value) for key,value in mapped.items()]
                          for name,mapped in runs.items()},
        controls=controls,comparisons=comparisons,allowances=allowances,calibration_sha256=digest,
        sampled_digests=sampled_digests,rows_per_run=2048,requests_per_run=8,prompt_rows=4,
        decode_steps=256,gpu_state_update_kernel=True,actual_request_stop=True,mutation_rejected=True,
        full_row2_grade_not_claimed=True,box_evidence=False,
        fixture_leaves=[dict(leaf='synthetic64-vocabulary logits and three precomputed prompt tokens',
            replaces='full model logits and committed native prompt prefix',more_convenient=True,
            equivalence='sampling/state/token-update/comparison method plumbing only; not model accuracy or native-to-stock handoff',box_evidence=False),
            dict(leaf='direct stock scheduler token-output method on a fresh Scheduler view',
                replaces='EngineCore output delivery and scheduler step',more_convenient=True,
                equivalence='same Request append/stop logic; no claim of engine/TP/PP scheduling',box_evidence=False)],
        deferred=['full worker capture/SSE row-mapping and four-rank structural grade','actual model logits','native-to-stock bank transition'])


def regrade_sampler_rule4(record):
    """Pure comparison regrade, retaining the full structural-gate exclusion."""
    import e3_teacher_decode_rule4 as rule
    try:
        mapped={}
        for name in ('full-A1','full-A2','full-C','split-B'):
            rows=record['raw_numeric_rows'][name]
            assert len(rows)==2048
            table={(row['stream'],row['position']):{'top':row['top']} for row in rows}
            assert set(table)=={(i,j) for i in range(8) for j in range(256)}
            mapped[name]=table
        calibration,digest,_=rule.load_calibration()
        controls={a+'|'+b:rule.pair_metrics(mapped[a],mapped[b]) for a,b in rule.CONTROL_PAIRS}
        maximum={key:max(v[key] for v in controls.values()) for key in rule.CONTINUOUS+rule.COUNTING}
        allowances=rule._allowances(calibration,maximum)
        reasons=rule._control_violations(calibration,controls)
        comparisons={}
        for name in ('full-A1','full-A2','full-C'):
            metrics=rule.pair_metrics(mapped['split-B'],mapped[name],flip_margin_limit=allowances['flip_margin'])
            comparisons[name]=metrics;reasons.extend(rule._comparison_violations(name,metrics,allowances))
        # Pure numeric rows do not satisfy the full schema/row-mapping gate.
        # Never manufacture graph, worker rank, lifecycle or capture facts.
        structural=rule.grade({'runs':record['raw_numeric_rows']})
        assert structural['verdict']=='INCONCLUSIVE'
        return dict(verdict='FAIL' if reasons else 'PASS',reasons=reasons,controls=controls,
            comparisons=comparisons,allowances=allowances,calibration_sha256=digest,
            full_row2_verdict=structural['verdict'],box_evidence=False)
    except Exception as exc:
        return dict(verdict='FAIL',failure_class='HARNESS',exception=type(exc).__name__,
                    message=str(exc)[:512],box_evidence=False)


def predicate_selftest():
    """Synthetic receipt mutation test, never scenario/hardware evidence."""
    import copy
    rows=[dict(stream=i,position=j,top={'token_id:'+str(k):-k*.2 for k in range(20)})
          for i in range(8) for j in range(256)]
    record={'raw_numeric_rows':{name:copy.deepcopy(rows) for name in ('full-A1','full-A2','full-C','split-B')}}
    positive=regrade_sampler_rule4(record)
    assert positive['verdict']=='PASS',positive
    assert positive['full_row2_verdict']=='INCONCLUSIVE'
    record['raw_numeric_rows']['split-B'][0]['top']['token_id:0']+=2.
    negative=regrade_sampler_rule4(record)
    assert negative['verdict']=='FAIL' and negative.get('reasons'),negative
    return dict(verdict='PASS',synthetic_receipt=True,box_evidence=False,
                positive=positive,mutation=negative,scope='predicate code path only, no fabricated lifecycle or graph flags')


if __name__=='__main__':raise SystemExit(main())
