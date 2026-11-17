#!/usr/bin/env python3
"""Pinned-image CPU tensor oracle. No model load, CUDA, sockets or engine control."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace as NS


def method(source, cls, name, namespace):
    tree=ast.parse(source)
    node=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls
              for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[]
    module=ast.Module(body=[node],type_ignores=[])
    exec('from __future__ import annotations\n'+ast.unparse(module),namespace)
    return namespace[name]


def run(root, out):
    import os
    import numpy as np
    import torch
    import vllm
    from vllm.model_executor.models.interfaces import SupportsMultiModal, _require_is_multimodal
    from vllm.model_executor.models.utils import _merge_multimodal_embeddings
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange, MultiModalKwargsItems
    from vllm.model_executor.models.qwen2_vl import _create_qwen2vl_field_factory
    from vllm.multimodal.utils import get_mm_features_in_window, set_mm_embedding_modality
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
    sys.path.insert(0,str(root/'patches/s5_engine'))
    import native_mm as mm
    import native_mm_overlay
    import model_hooks
    from e3_types import TransactionId
    os.environ['FLASH_S5_NATIVE_MM']='1'
    torch.set_num_threads(1)
    base=Path(vllm.__file__).parent
    modelpath=base/'models/qwen4_exp/nvidia/model.py'
    source=modelpath.read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==model_hooks.PINS['models/qwen4_exp/nvidia/model.py']
    stock_method=method(source,'Qwen4ExpForConditionalGeneration','embed_input_ids',
        dict(torch=torch,_require_is_multimodal=_require_is_multimodal,_merge_multimodal_embeddings=_merge_multimodal_embeddings))
    encoder_source=(base/'v1/worker/gpu/mm/encoder_runner.py').read_text()
    sys.modules['vllm.v1.flash_e3.native_mm']=mm
    prepare=method(native_mm_overlay.encoder_source(encoder_source),'EncoderRunner','prepare_mm_inputs',
        dict(torch=torch,async_tensor_h2d=lambda tensor,device:tensor.clone()))
    gather=method(encoder_source,'EncoderRunner','gather_mm_embeddings',dict(torch=torch,PIN_MEMORY=False,
        get_mm_features_in_window=get_mm_features_in_window,set_mm_embedding_modality=set_mm_embedding_modality))
    weights=(torch.arange(32*2560,dtype=torch.float32).reshape(32,2560)/4096).to(torch.bfloat16)
    embed=lambda ids:torch.nn.functional.embedding(ids.long(),weights)
    stock=NS(_embed_text_input_ids=lambda ids,call,**kw:call(ids),language_model=NS(embed_input_ids=embed),
        language_model_only=False,use_deepstack=False)
    cases=[]
    for tokens,h,w in ((1,2,2),(256,32,32),(1024,64,64)):
      for offset in (201,2033,189000):
       for masked in (False,True):
        values=(torch.arange(tokens*2560,dtype=torch.float32).reshape(tokens,2560)/1024).to(torch.bfloat16)
        length=tokens*2 if masked else tokens
        mask=torch.arange(length)%2==0 if masked else None
        hf_inputs={'image_embeds':values,'image_grid_thw':torch.tensor([[1,h,w]],dtype=torch.int64)}
        fields=_create_qwen2vl_field_factory(2)(hf_inputs)
        item=MultiModalKwargsItems.from_hf_inputs(hf_inputs,fields)['image'][0]
        feature=MultiModalFeatureSpec(data=item,modality='image',
            identifier='fixture-'+str(len(cases)),mm_position=PlaceholderRange(offset,length,mask))
        request=NS(request_id='fixture-'+str(len(cases)),mm_features=[feature],num_prompt_tokens=199999)
        plan=mm.request_plan(request)
        starts=sorted({max(0,offset-4), offset, offset+max(0,length//2), offset+length})
        for start in starts:
          end=min(start+2024,199998)
          tx=TransactionId(1,request.request_id,1,start,end,len(cases)+1)
          payload=plan.chunk(tx,1)
          ids=torch.arange(end-start,dtype=torch.int64)%32
          expected_mask=torch.zeros(end-start,dtype=torch.bool)
          row_values=[]
          for i,p in enumerate(plan.positions):
            if start<=p<end:
              expected_mask[p-start]=True;row_values.append(values[i])
          mm_values=[] if not row_values else [torch.stack(row_values)]
          expected=stock_method(stock,ids,mm_values,is_multimodal=expected_mask)
          lane=NS(inputs={'input_ids':ids},model=NS(embed_input_ids=embed))
          if payload:
            actual=mm.prefix_inputs(lane,NS(mm=payload,transaction=tx,state_epoch=1,input_ids=tuple(ids.tolist())),torch)
            assert torch.equal(actual,expected),'native splice differs from pinned stock embed_input_ids'
            lease=lane._native_mm_lease
            try:lease.release()
            except mm.NativeMMError:pass
            else:raise AssertionError('prefix sources retired without completion')
            lease.bind(NS(query=lambda:True));lease.release()
          else:
            actual=embed(ids);assert torch.equal(actual,expected)
          # Ordinary image-in-handback uses the same existing encoder cache/gather contract.
          cache=EncoderCache();cache.add_request(request.request_id,[feature]);cache.encoder_outputs[feature.identifier]=values
          state=NS(encoder_cache=cache,is_realtime=False,device=torch.device('cpu'))
          cache.encoder_outputs.clear()
          hashes,kwargs=prepare(state,{request.request_id:[0]})
          assert hashes==[] and kwargs==[]
          assert cache.encoder_outputs[feature.identifier].untyped_storage().nbytes()==values.nbytes
          chunks,ordinary_mask=gather(state,[request.request_id],end-start,np.array([end-start]),
              np.array([0,end-start]),np.array([request.num_prompt_tokens]),np.array([start]))
          ordinary=stock_method(stock,ids,chunks,is_multimodal=ordinary_mask)
          assert torch.equal(ordinary,actual),'ordinary handback differs from native splice'
          cache.remove_request(request.request_id)
          assert request.request_id not in cache.mm_features
          cases.append(dict(tokens=tokens,masked=masked,start=start,end=end,rows=int(expected_mask.sum()),equal=True))
        # CPU source mutation cannot rewrite any earlier queued snapshot.
        frozen=plan.data;values.fill_(0);assert mm.request_plan(request).data==frozen
    # Eight split views must not each retain a whole eight-image GPU batch.
    batch=torch.zeros((8*1024,2560),dtype=torch.bfloat16)
    cache=EncoderCache();scheduled={}
    for i,view in enumerate(batch.split(1024)):
        feature=MultiModalFeatureSpec(data={'image_embeds':NS(data=view),'image_grid_thw':NS(data=torch.tensor([1,64,64]))},
            modality='image',identifier='batch-'+str(i),mm_position=PlaceholderRange(0,1024))
        cache.add_request(str(i),[feature]);scheduled[str(i)]=[0]
    state=NS(encoder_cache=cache,device=torch.device('cpu'))
    os.environ['FLASH_S5_NATIVE_MM']='0'
    hashes,kwargs=prepare(state,scheduled)
    assert len(hashes)==8 and len(kwargs)==8 and cache.encoder_outputs=={}
    os.environ['FLASH_S5_NATIVE_MM']='1'
    hashes,kwargs=prepare(state,scheduled)
    assert hashes==[] and kwargs==[]
    storages={v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in cache.encoder_outputs.values()}
    assert len(storages)==8 and sum(storages.values())==40*2**20
    try:mm.stage_image(feature,'cpu',lambda tensor,device:tensor)
    except mm.NativeMMError:pass
    else:raise AssertionError('split view retained an eight-image backing storage')
    bad=NS(request_id='bad',num_prompt_tokens=4,mm_features=[MultiModalFeatureSpec(
        data={'image_embeds':NS(data=torch.full((1,2560),float('nan'),dtype=torch.bfloat16)),
              'image_grid_thw':NS(data=torch.tensor([1,2,2]))},modality='image',identifier='bad',mm_position=PlaceholderRange(0,1))])
    try:mm.request_plan(bad)
    except mm.NativeMMError:pass
    else:raise AssertionError('nonfinite feature admitted')
    assert not torch.cuda.is_initialized()
    result=dict(verdict='PASS',scope='CPU splice/ordinary-gather values only; no GPU/TP/PP/LM answers',
        cases=cases,cuda_initialized=False,private_cache_storage_bytes=sum(storages.values()),
        private_cache_allocations=len(storages),model_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        ordinary_gather_source_sha256=hashlib.sha256(encoder_source.encode()).hexdigest())
    out.write_text(json.dumps(result,indent=2)+'\n')
    print('NATIVE_MM_TENSOR PASS cases='+str(len(cases))+' CUDA_NOT_INITIALIZED')

if __name__=='__main__':
    try:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
        a=p.parse_args();run(a.source,a.out)
    except Exception as exc:
        frame=traceback.extract_tb(exc.__traceback__)[-1]
        print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {exc} at {frame.filename}:{frame.lineno}',file=sys.stderr)
        print('NATIVE_MM_TENSOR HOLD '+type(exc).__name__)
        raise SystemExit(2)
