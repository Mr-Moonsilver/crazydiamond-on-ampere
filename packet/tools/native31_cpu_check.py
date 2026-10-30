"""Real-image CPU exactness/registration/client-error check; explicit comm leaf."""
import hashlib
import ast
from pathlib import Path
import time
from types import SimpleNamespace as NS
from unittest.mock import patch


def run():
    import torch
    import lane_comm as lc
    from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
    from vllm.v1.engine import EngineCoreRequest, EngineCoreOutput, FinishReason
    from vllm.sampling_params import SamplingParams, RequestOutputKind
    assert not torch.cuda.is_initialized()
    op=lc.register_selection_op()
    stream=object();calls=[];authoritative={}
    class Comm:
        disabled=False;world_size=2
        def broadcast(self,tensor,src,stream):
            calls.append((src,stream))
            tensor.copy_(authoritative['selection'])
    group=NS(world_size=2,device_communicator=NS(pynccl_comm=Comm()))
    cuda_leaf=NS(int32=torch.int32,cuda=NS(current_stream=lambda device:stream))
    original=lc.canonical_selection
    def canonical(tensor):original(tensor,group=group,torch_module=cuda_leaf)
    generator=torch.Generator().manual_seed(31)
    values=torch.randn(128,32,generator=generator,dtype=torch.bfloat16)
    query=torch.randn(8,32,generator=generator,dtype=torch.bfloat16)
    def decode(selection):
        chosen=values[selection.long()]
        logits=(chosen.float()*query.float()[:,None,:]).sum(-1)
        return (torch.softmax(logits,-1)[...,None]*chosen.float()).sum(1)
    digests=[]
    with patch.object(lc,'canonical_selection',canonical):
        compiled=torch.compile(lambda selection: (op(selection),selection)[1],backend='eager',fullgraph=True)
        for step in range(32):
            before=torch.stack([torch.randperm(128,generator=generator)[:16] for _ in range(8)]).to(torch.int32)
            after=before.clone();authoritative['selection']=before
            expected=decode(before);compiled(after);actual=decode(after)
            assert torch.equal(after,before) and torch.equal(actual,expected)
            assert torch.equal(actual.argmax(-1),expected.argmax(-1))
            digests.append(hashlib.sha256(actual.numpy().tobytes()).hexdigest())
    assert len(calls)==32 and all(src==0 and used is stream for src,used in calls)
    # Execute both actual generated _run_qsa methods. Only the indexer/model
    # tensor producers and attention kernel are CPU leaves; call sites, guards,
    # ordering and buffer arguments remain the real old/new source.
    root=Path('/source/harness')
    if root.is_dir():
        def load_method(path):
            tree=ast.parse(path.read_text())
            node=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_run_qsa')
            node.decorator_list=[]
            namespace=dict(torch=torch,cast=lambda typ,value:value,FlashAttentionMetadata=object,QSAForwardMetadata=object,Qwen4ExpQSAFlashAttentionImpl=object,get_forward_context=lambda:NS(attn_metadata=metadata),_flash_s5_selection=op)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
            return namespace['_run_qsa']
        before_method=load_method(root/'native31-fixtures/qsa-before.py')
        after_method=load_method(root/'native30_overlay/qsa.py')
        metadata={'layer':NS(num_actual_tokens=8,slot_mapping=None,max_query_len=1),'side':NS(num_actual_tokens=8,token_to_req=None)}
        class Indexer:
            raw_key_cache=NS(prefix='side');packed_output_width=16
            def __call__(self,qk,positions,buffer):buffer.copy_(qk);return buffer
        def forward(layer,q,k,v,cache,main,out,**kwargs):out.copy_(decode(layer.topk_indices_buffer))
        def layer():return NS(layer_name='layer',kv_cache=torch.ones(1),indexer=Indexer(),topk_indices_buffer=torch.empty((8,16),dtype=torch.int32),impl=NS(do_kv_cache_update=lambda *args:None,forward_qsa=forward),_max_decode_query_len=1)
        with patch.object(lc,'canonical_selection',canonical):
            for step in range(32):
                selected=torch.stack([torch.randperm(128,generator=generator)[:16] for _ in range(8)]).to(torch.int32)
                authoritative['selection']=selected
                expected=torch.empty((8,32));actual=torch.empty_like(expected)
                before_method(layer(),selected,None,None,None,None,expected)
                after_method(layer(),selected,None,None,None,None,actual)
                assert torch.equal(actual,expected)
    params=SamplingParams(max_tokens=64,detokenize=False,output_kind=RequestOutputKind.DELTA)
    request=EngineCoreRequest(request_id='native31-cpu',external_req_id='native31-external',prompt_token_ids=[1,2,3],mm_features=None,sampling_params=params,pooling_params=None,arrival_time=time.time(),lora_request=None,cache_salt=None,data_parallel_rank=None)
    collector=RequestOutputCollector(RequestOutputKind.DELTA,request.request_id)
    processor=OutputProcessor(None,log_stats=False);processor.add_request(request,None,queue=collector)
    error=EngineCoreOutput(request_id=request.request_id,new_token_ids=[],finish_reason=FinishReason.ERROR,stop_reason='unsupported-native-schedule-profile')
    processor.process_outputs([error]);output=collector.get_nowait()
    assert output.finished and output.outputs[0].finish_reason=='error'
    from vllm.entrypoints.generate.base.serving import GenerateBaseServing
    from vllm.exceptions import GenerationError
    serving=GenerateBaseServing.__new__(GenerateBaseServing)
    try:
        serving._raise_if_error(output.outputs[0].finish_reason,request.request_id)
    except GenerationError:
        pass
    else:
        raise AssertionError('real API failed to raise typed GenerationError')
    assert not torch.cuda.is_initialized()
    return dict(verdict='PASS',ordinary_decode_exactness='PASS',steps=32,rows=8,output_sha256=digests,compiled_mutation_op='PASS',actual_run_qsa_exactness='PASS' if root.is_dir() else 'NOT_RUN',typed_client_error='PASS',typed_api_error='PASS',cuda_initialized=False,scope='CPU real Torch gather/softmax decode arithmetic and real OutputProcessor; synthetic selected indices and in-process PyNccl/stream leaf; not model/GPU/NCCL latency evidence')
