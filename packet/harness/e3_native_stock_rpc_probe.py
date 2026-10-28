#!/usr/bin/env python3
"""Source-faithful stock batch-queue/owned-receiver credit lifecycle proof.

CPU MQ and scheduler leaves preserve RPC ordering and discarded exec futures.
No GPU completion, model numerical or production performance claim.
"""
import argparse,hashlib,importlib.util,inspect,json,threading,time,traceback,types
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from queue import Queue,Empty

_PROGRESS=None

def checkpoint(step,**fields):
    if _PROGRESS is not None:
        _PROGRESS.write(json.dumps(dict(step=step,observed_ns=time.monotonic_ns(),**fields))+'\n')
        _PROGRESS.flush()


def collect(graph,*,cpu_only=True):
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor,FutureWrapper,WorkerProc
    from vllm.v1.flash_e3 import executor_hooks as hooks
    import torch
    pins={r['module']:r for r in graph['modules'] if r['active_mount']}
    for cls in (EngineCore,MultiprocExecutor,hooks.LocalFuture):
        module=__import__('sys').modules[cls.__module__];path=Path(inspect.getfile(module))
        pin=pins[module.__name__]
        assert str(path)==pin['destination'] and hashlib.sha256(path.read_bytes()).hexdigest()==pin['sha256']
    assert callable(EngineCore.step_with_batch_queue)
    assert callable(MultiprocExecutor.execute_model) and callable(MultiprocExecutor.sample_tokens)
    assert callable(FutureWrapper.result)
    if cpu_only:assert not torch.cuda.is_initialized()
    return EngineCore,MultiprocExecutor,FutureWrapper,WorkerProc,hooks


class Reader:
    def __init__(self):self.q=Queue();self.readers=set()
    def dequeue(self,timeout=None):
        self.readers.add(threading.get_ident())
        try:return self.q.get(timeout=timeout)
        except Empty:raise TimeoutError from None


class Broadcast:
    def __init__(self,executor,status):self.e=executor;self.status=status;self.writes=[]
    def enqueue(self,item):
        method,args,kwargs,rank=item;self.writes.append(method)
        for r,q in enumerate(self.e.response_mqs):
            if rank is None or rank==r:q.q.put((self.status,(r,method,len(self.writes))))


def scenario(classes,kind,iterations=200):
    Core,Executor,FutureWrapper,WorkerProc,hooks=classes
    e=types.SimpleNamespace(response_mqs=[Reader()],futures_queue=deque(),is_failed=False,
        output_rank=0,kv_output_aggregator=None,ec_output_aggregator=None)
    e.rpc_broadcast_mq=Broadcast(e,WorkerProc.ResponseStatus.SUCCESS)
    e._flash_s5_stock_collective_rpc=types.MethodType(Executor._flash_s5_stock_collective_rpc,e)
    e.execute_model=types.MethodType(Executor.execute_model,e)
    e.sample_tokens=types.MethodType(Executor.sample_tokens,e)
    transport=None
    if kind=='stock':
        e.collective_rpc=e._flash_s5_stock_collective_rpc
    else:
        from vllm.v1.flash_e3.reply_transport import ReplyTransport
        transport=ReplyTransport({0:e.response_mqs[0]},WorkerProc.ResponseStatus.SUCCESS,
            handoff_idle=True,max_calls=10)
        e._flash_s5_transport=transport;e._flash_s5_stock_forbidden=True;e._flash_s5_stock_futures={}
        if kind=='native15':
            spec=importlib.util.spec_from_file_location('vllm.v1.flash_e3.native15_credit_counterexample',
                Path(__file__).with_name('native15_executor_hooks.py'))
            legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
            e.collective_rpc=lambda method,**kw:legacy.collective_rpc(e,e._flash_s5_stock_collective_rpc,method,**kw)
        else:e.collective_rpc=types.MethodType(Executor.collective_rpc,e)
    outputs=[]
    scheduler=types.SimpleNamespace(has_requests=lambda:True,
        schedule=lambda throttle:types.SimpleNamespace(total_num_scheduled_tokens=1,pending_structured_output_tokens=False),
        get_grammar_bitmask=lambda output:None,update_from_output=lambda output,value:outputs.append(value) or {})
    core=types.SimpleNamespace(batch_queue=deque(),batch_queue_size=2,scheduler=scheduler,
        model_executor=e,is_ec_consumer=True,is_pooling_model=False,
        _should_throttle_prefills=lambda:False,log_error_detail=lambda output:nullcontext(),
        capture_iteration_details=lambda output:nullcontext(None),
        _attach_iteration_details=lambda *args:None,_process_aborts_queue=lambda:None)
    high=0;completed=0
    try:
        for i in range(iterations):
            Core.step_with_batch_queue(core);completed+=1
            if transport:high=max(high,len(transport._calls))
        if kind=='native15':raise AssertionError('native15 negative unexpectedly passed')
        for future,_,execute in core.batch_queue:future.result();execute.result()
        if transport:
            assert not transport._calls and not e._flash_s5_stock_futures
        else:assert not e.futures_queue
        assert len(outputs)>=iterations-1
        return dict(verdict='PASS',iterations=completed,sampled_outputs=len(outputs),max_calls=10,
            credit_high_water=high,all_credits_returned=True,broadcasts=len(e.rpc_broadcast_mq.writes),
            actual_stock_batch_queue=True,actual_executor_methods=True,actual_future_wrapper=kind=='stock',
            reply_queue_readers=[len(q.readers) for q in e.response_mqs])
    except Exception as exc:
        if kind!='native15' or type(exc).__name__!='Backpressure' or str(exc)!='unclaimed RPC credit budget':raise
        assert completed<20 and len(transport._calls)==10
        return dict(verdict='PASS',legacy_rejected=True,iterations_before_refusal=completed,
            sampled_outputs=len(outputs),credits_at_refusal=len(transport._calls),error_type=type(exc).__name__,message=str(exc))
    finally:
        if transport:transport.close(1)


def gpu_config(device):
    """Same reduced real Qwen family as matrix2D; transport-only workload."""
    from vllm.config import (LoadConfig,VllmConfig,ModelConfig,CacheConfig,
        SchedulerConfig,ParallelConfig,DeviceConfig,CompilationConfig)
    from e3_native_transitions_lifecycle_probe import config
    original=config(device)
    # Construct configs through their public constructors. Reconstructing an
    # already resolved pydantic dataclass via replace revalidates mutated
    # derived fields such as mm_tensor_ipc=None as user inputs and is invalid.
    cfg=VllmConfig(
        model_config=ModelConfig(model=original.model_config.model,skip_tokenizer_init=True,
            language_model_only=True,dtype='bfloat16',max_model_len=256,enforce_eager=True),
        cache_config=CacheConfig(block_size=4048,mamba_block_size=4048,num_gpu_blocks_override=4,
            enable_prefix_caching=True,gpu_memory_utilization=.08,mamba_cache_mode='align'),
        scheduler_config=SchedulerConfig(max_model_len=256,is_encoder_decoder=False,
            max_num_batched_tokens=128,max_num_seqs=2,async_scheduling=True),
        parallel_config=ParallelConfig(distributed_executor_backend='mp'),
        device_config=DeviceConfig(device=device),compilation_config=CompilationConfig(mode=0),
        load_config=LoadConfig(load_format='dummy'))
    assert cfg.max_concurrent_batches==2
    assert cfg.parallel_config.tensor_parallel_size==cfg.parallel_config.pipeline_parallel_size==1
    return cfg


def gpu_shape(*,cpu_only=True):
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor,WorkerProc
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams
    import inspect,torch
    cuda_before=torch.cuda.is_initialized()
    cfg=gpu_config('cpu')
    request=Request(request_id='credit-shape',prompt_token_ids=[1,2,3,4],
        sampling_params=SamplingParams(max_tokens=32,temperature=0.,ignore_eos=True),pooling_params=None)
    inspect.signature(EngineCore).bind(cfg,MultiprocExecutor,False)
    inspect.signature(EngineCore.add_request).bind(None,request)
    assert callable(EngineCore.step_with_batch_queue) and callable(EngineCore.shutdown)
    assert WorkerProc.ResponseStatus.SUCCESS is not None
    if cpu_only:assert not torch.cuda.is_initialized()
    else:assert torch.cuda.device_count()==1
    return dict(verdict='PASS',batch_queue_size=cfg.max_concurrent_batches,
        venue_preconditions=dict(cpu_only=cpu_only,cuda_initialized_before=cuda_before,
            cuda_initialized_after=torch.cuda.is_initialized(),device_count=torch.cuda.device_count(),
            cuda_uninitialized_required=cpu_only,
            gpu_venue_rule='config/real constructor may initialize CUDA normally; exactly one visible GPU required',
            hub_limit='GPU-less CPU-shape cannot prove driver/device/GPU startup state'),
        executed=['real reduced Qwen config construction','VllmConfig/LoadConfig validation',
                  'Request/SamplingParams construction','EngineCore constructor signature binding',
                  'actual executor/worker/core imports and accessed callable attributes'],
        deferred=['EngineCore.__init__: constructs real MultiprocExecutor and CUDA worker subprocess',
                  'GPUWorker.init_device/load_model/initialize_from_config via real EngineCore',
                  'actual stock Scheduler, CUDA forward/sampling and asynchronous batch queue'])


def gpu_executor():
    import torch
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor,WorkerProc
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams
    from vllm.v1.flash_e3 import executor_hooks as hooks,async_handoff
    cfg=gpu_config('cuda')
    core=None
    try:
        checkpoint('actual-engine-core-before-construction')
        core=EngineCore(cfg,MultiprocExecutor,False)
        checkpoint('actual-engine-core-constructed')
        executor=core.model_executor
        assert type(executor) is MultiprocExecutor and core.batch_queue is not None
        assert not executor.futures_queue and not hasattr(executor,'_flash_s5_transport')
        transport=hooks.install(executor,config=async_handoff.HandoffConfig.from_env(
            {'FLASH_S5_OVERLAP':'1','FLASH_S5_COMMAND_SLOTS':'10'}),
            success_status=WorkerProc.ResponseStatus.SUCCESS)
        assert transport._max_calls==10
        checkpoint('actual-owned-transport-installed',command_slots=transport._max_calls)
        request=Request(request_id='actual-mp-credit-request',prompt_token_ids=[1,2,3,4],
            sampling_params=SamplingParams(max_tokens=32,temperature=0.,ignore_eos=True),pooling_params=None)
        core.add_request(request)
        iterations=0;high=0;deadline=time.monotonic()+180
        while core.scheduler.has_requests() or core.batch_queue:
            assert time.monotonic()<deadline,'actual executor decode deadline'
            core.step_with_batch_queue();iterations+=1;high=max(high,len(transport._calls))
            checkpoint('actual-decode-step',iteration=iterations,credits=len(transport._calls),
                completion_tokens=request.num_output_tokens)
        assert request.num_output_tokens==32 and iterations>=20
        assert not transport._calls and not executor._flash_s5_stock_futures
        return dict(verdict='PASS',actual_executor_type=type(executor).__module__+'.'+type(executor).__name__,
            actual_engine_core=True,request_type=type(request).__module__+'.'+type(request).__name__,
            iterations=iterations,completion_tokens=request.num_output_tokens,
            max_calls=10,credit_high_water=high,all_credits_returned=True,
            gpu_execution=True,full_model_numerics=False,native_prefill=False,
            scope='real TP1 MultiprocExecutor/EngineCore stock batch queue and owned receiver; reduced dummy-weight model only')
    finally:
        if core is not None:core.shutdown()


def main():
    global _PROGRESS
    p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=('cpu-full','cpu-shape','gpu-all'),default='cpu-full');a=p.parse_args();started=time.monotonic()
    result=dict(schema='native16-stock-rpc-credit/1',verdict='INCONCLUSIVE',rc=2,failure_class='NONE',
        sections={},engine_constructed=False,gpu_execution=False,methods_patched=False,command_slots=10)
    _PROGRESS=a.out.with_suffix('.progress.jsonl').open('x')
    try:
        graph=json.loads(a.graph.read_text());classes=collect(graph,cpu_only=a.mode!='gpu-all')
        for kind in ('stock','native15','native16'):
            result['sections'][kind]=scenario(classes,kind)
            checkpoint(kind,receipt=result['sections'][kind])
            print('END '+kind+' PASS',flush=True)
        result['sections']['gpu_constructor_shape']=gpu_shape(cpu_only=a.mode!='gpu-all')
        checkpoint('gpu-constructor-shape',receipt=result['sections']['gpu_constructor_shape'])
        if a.mode=='gpu-all':
            result['sections']['multiproc_executor']=gpu_executor()
            result['engine_constructed']=result['gpu_execution']=True
        import torch
        if a.mode!='gpu-all':assert not torch.cuda.is_initialized()
        result.update(verdict='PASS',rc=0,cuda_initialized=torch.cuda.is_initialized(),
            fixture_leaves=json.loads(Path(__file__).with_name('fixture-leaves.json').read_text()))
    except Exception as exc:
        from e3_native_transitions_lifecycle_probe import exception_chain
        result.update(verdict='FAIL',rc=1,failure_class='OPEN',error_type=type(exc).__name__,
            error_message=str(exc),stack=traceback.format_exc(),exception_chain=exception_chain(exc))
    result['wall_s']=time.monotonic()-started
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('verdict','rc','failure_class','wall_s')}),flush=True)
    _PROGRESS.close()
    return result['rc']


if __name__=='__main__':raise SystemExit(main())
