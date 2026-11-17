#!/usr/bin/env python3
"""Pinned CPU codec/chat-token geometry; model lookup/processor allocation are declared leaves."""
import argparse,hashlib,json,sys
from pathlib import Path
import copy
from types import SimpleNamespace as NS
from unittest.mock import patch,PropertyMock

def main():
    import torch
    from transformers import AutoTokenizer
    from vllm.entrypoints import chat_utils
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.serve.tokenize.protocol import TokenizeChatRequest
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(a.config,local_files_only=True)
    cache=json.loads(a.cache.read_text());response=next(iter(cache['features'].values()));feature=copy.deepcopy(response['content']);feature['image_embeds']['image_grid_thw']=response['image_grid_thw'][0]
    config=NS(is_multimodal_model=True,enable_prompt_embeds=False,hf_config=NS(use_unified_vision_chunk=False),multimodal_config=NS(enable_mm_embeds=True,
        get_limit_per_prompt=lambda modality:0,media_io_kwargs=None,interleave_mm_strings=True),allowed_local_media_path='',allowed_media_domains=None)
    config.get_multimodal_config=lambda:config.multimodal_config
    leaf=type('PlaceholderModel',(),dict(get_placeholder_str=staticmethod(lambda modality,n:'<|vision_start|><|image_pad|><|vision_end|>')))
    from vllm.model_executor.models.qwen2_vl import _create_qwen2vl_field_factory
    processor=NS(info=NS(data_parser=NS(placeholder_metadata_fields=lambda modality:{'image_grid_thw'}),
        ctx=NS(model_config=NS(get_inputs_embeds_size=lambda:2560))))
    processor._get_mm_fields_config=lambda batch,kwargs:_create_qwen2vl_field_factory(2)(batch)
    def ids(messages):
        with patch.object(chat_utils.BaseMultiModalItemTracker,'model_cls',new_callable=PropertyMock,return_value=leaf),patch.object(chat_utils.BaseMultiModalItemTracker,'mm_processor',new_callable=PropertyMock,return_value=processor):
            conversation,data,uuids=chat_utils.parse_chat_messages(messages,config,'string')
        tensor=data['image']['image_embeds'];grid=data['image']['image_grid_thw']
        if isinstance(tensor,list):tensor=tensor[0]
        if isinstance(grid,list):grid=grid[0]
        assert tensor.dtype==torch.bfloat16 and tuple(tensor.shape)==(1024,2560)
        assert hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==response['embedding_sha256']
        assert grid.reshape(-1).tolist()==response['image_grid_thw'][0]
        text=tokenizer.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        text=text.replace('<|image_pad|>','<|image_pad|>'*1024)
        return tokenizer.encode(text,add_special_tokens=False)
    rows=[]
    for tail in (False,True):
        feature_part=feature
        filler=0
        for _ in range(8):
            text=dict(type='text',text='Agent memory-fit fixture.'+' x'*filler+' Keep responding until the output cap.')
            messages=[dict(role='user',content=[text,feature_part] if tail else [feature_part,text])]
            prompt=ids(messages)
            if len(prompt)==190320:break
            filler+=190320-len(prompt)
        assert len(prompt)==190320
        positions=[i for i,t in enumerate(prompt) if t==248056];assert len(positions)==1024
        if tail:assert any(i>=190256 for i in positions)
        body=dict(model='Qwen3.8-Flash-Next',messages=messages,stream=True,max_tokens=8192,temperature=0,ignore_eos=True,
            return_token_ids=True,request_id='vision-fit-fixture',skip_special_tokens=False,chat_template_kwargs=dict(enable_thinking=False))
        ChatCompletionRequest.model_validate(body)
        TokenizeChatRequest.model_validate({k:v for k,v in body.items() if k in ('model','messages','chat_template_kwargs')})
        rows.append(dict(tail=tail,prompt_tokens=len(prompt),image_rows=len(positions),image_start=positions[0],image_end=positions[-1]+1))
    # Exact fixed-work continuation, rather than rerendering a historical assistant message.
    messages=[dict(role='user',content=[feature,dict(type='text',text='first'+' x'*100)])]
    before=ids(messages);reply=tokenizer.encode(' continuation text',add_special_tokens=False)
    image_end=max(i for i,t in enumerate(before) if t==248056)+1
    assert before[image_end]==248054
    retained=tokenizer.decode(before[image_end+1:]+reply,skip_special_tokens=False)
    original_text=messages[-1]['content'][1]['text']
    if retained.startswith(original_text):pass
    elif retained.startswith('\n'+original_text):retained=retained[1:]
    else:raise AssertionError('unmodelled image/text separator')
    after=ids([dict(role='user',content=[feature,dict(type='text',text=retained+'<|im_end|>\n<|im_start|>user\nnext')])])
    assert after[:len(before)+len(reply)]==before+reply,'exact token-prefix continuation changed'
    import vision_fit_8x200k as fit
    def local_post(url,path,body,timeout=60):
        if path=='/tokenize':
            TokenizeChatRequest.model_validate(body)
            actual=ids(body['messages'])
            return dict(tokens=actual,count=len(actual),max_model_len=200000)
        if path=='/detokenize':return dict(prompt=tokenizer.decode(body['tokens'],skip_special_tokens=False))
        raise AssertionError('unexpected CPU tokenizer path')
    args=NS(url='http://CPU-IO-LEAF',model='Qwen3.8-Flash-Next')
    feature_case=dict(content=fit.embedding_content(response),rows=1024,image_sha256='a'*64,feature_sha256=response['embedding_sha256'])
    growth=[];previous=None;reply=tokenizer.encode(' x',add_special_tokens=False)
    assert len(reply)==1
    with patch.object(fit,'post',side_effect=local_post):
        for target in fit.PREFIXES:
            stream=fit.prepare(args,feature_case,target,'growth-'+str(target),previous)
            ChatCompletionRequest.model_validate(dict(model=args.model,messages=stream.messages,max_tokens=stream.output_cap,
                temperature=0,ignore_eos=True,stream=True,stream_options={'include_usage':True},return_token_ids=True,
                request_id=stream.uid,skip_special_tokens=False,chat_template_kwargs={'enable_thinking':False}))
            stream.tokens=reply*stream.output_cap
            growth.append(dict(prompt_tokens=len(stream.prompt),output_cap=stream.output_cap,retained_prefix_equal=True))
            previous=stream
        prime=fit.companion_prime(args,previous,0)
        assert len(prime.prompt)==190144 and prime.prompt[:190080]==previous.prompt[:190080]
        assert prime.output_cap==1
        tail=fit.prepare(args,feature_case,190320,'tail-real-codec',tail=True)
        assert 0<tail.record()['image_tail_overlap']<=64
    assert not torch.cuda.is_initialized()
    a.out.write_text(json.dumps(dict(verdict='PASS',rows=rows,companion_prime=dict(prompt_tokens=len(prime.prompt),shared_prefix_tokens=190080,output_cap=prime.output_cap),reply_prefix_equal=True,growth=growth,cache_tensor_byte_equal=True,tail_image_overlap=tail.record()['image_tail_overlap'],cuda_initialized=False,
        scope='real pinned codec/parser/tokenizer/protocol types; synthetic model placeholder lookup and processor owner; no model/device forward'),indent=2)+'\n')
    print('VISION_FIT_TOKENIZER PASS')

if __name__=='__main__':
    import traceback
    try:main()
    except Exception as exc:
        frame=traceback.extract_tb(exc.__traceback__)[-1];print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {exc} at {frame.filename}:{frame.lineno}',file=sys.stderr);raise SystemExit(2)
