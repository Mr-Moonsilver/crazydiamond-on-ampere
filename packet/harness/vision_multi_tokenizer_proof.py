#!/usr/bin/env python3
"""Pinned CPU multi-image wire/order and exact native chunk geometry; no model forward."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch,PropertyMock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from vision_g5_plan import cases,messages

def main():
 import torch
 from transformers import AutoTokenizer
 from vllm.entrypoints import chat_utils
 from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
 from vllm.model_executor.models.qwen2_vl import _create_qwen2vl_field_factory
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--semantic',type=Path,required=True);p.add_argument('--eight',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 tokenizer=AutoTokenizer.from_pretrained(a.config,local_files_only=True)
 cache={**json.loads(a.semantic.read_text())['features'],**json.loads(a.eight.read_text())['features']}
 config=NS(is_multimodal_model=True,enable_prompt_embeds=False,hf_config=NS(use_unified_vision_chunk=False),multimodal_config=NS(enable_mm_embeds=True,get_limit_per_prompt=lambda modality:0,media_io_kwargs=None,interleave_mm_strings=True),allowed_local_media_path='',allowed_media_domains=None)
 config.get_multimodal_config=lambda:config.multimodal_config
 model=type('PlaceholderModel',(),dict(get_placeholder_str=staticmethod(lambda modality,n:'<|vision_start|><|image_pad|><|vision_end|>')))
 processor=NS(info=NS(data_parser=NS(placeholder_metadata_fields=lambda modality:{'image_grid_thw'}),ctx=NS(model_config=NS(get_inputs_embeds_size=lambda:2560))))
 processor._get_mm_fields_config=lambda batch,kwargs:_create_qwen2vl_field_factory(2)(batch)
 def feature(name):
  response=cache[hashlib.sha256((a.assets/name).read_bytes()).hexdigest()];value=copy.deepcopy(response['content']);value['image_embeds']['image_grid_thw']=response['image_grid_thw'][0];return value
 def ids(case):
  body=messages(case,feature);ChatCompletionRequest.model_validate(dict(model='qwen38-w8a8-tp4',messages=body,max_tokens=128,chat_template_kwargs=dict(enable_thinking=False)))
  with patch.object(chat_utils.BaseMultiModalItemTracker,'model_cls',new_callable=PropertyMock,return_value=model),patch.object(chat_utils.BaseMultiModalItemTracker,'mm_processor',new_callable=PropertyMock,return_value=processor):conversation,data,_=chat_utils.parse_chat_messages(body,config,'string')
  actual=data['image']['image_embeds'];actual=actual if isinstance(actual,list) else [actual]
  expected=[cache[hashlib.sha256((a.assets/name).read_bytes()).hexdigest()] for name in case['images']]
  merged=torch.cat([x.reshape(-1,2560) for x in actual]);parts=[x.contiguous().view(torch.uint8).numpy().tobytes() for x in actual]
  expected_bytes=sum(x['shape'][0] for x in expected)*5120
  assert merged.numel()*2==expected_bytes
  flat=b''.join(parts);offset=0
  for item in expected:
   size=item['shape'][0]*5120;assert hashlib.sha256(flat[offset:offset+size]).hexdigest()==item['embedding_sha256'];offset+=size
  text=tokenizer.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True,enable_thinking=False)
  chunks=text.split('<|image_pad|>');assert len(chunks)==len(expected)+1
  text=chunks[0]+''.join('<|image_pad|>'*item['shape'][0]+chunk for item,chunk in zip(expected,chunks[1:]))
  return tokenizer.encode(text,add_special_tokens=False)
 original=json.loads((a.assets/'cases.json').read_text());result=None
 for repeat in range(1500,2100,16):
  plan=cases(original,repeat);case=plan[-1];tokens=ids(case);positions=[i for i,t in enumerate(tokens) if t==248056];count={i//2048:0 for i in positions}
  for i in positions:count[i//2048]+=1
  cross=any(x+1==y and x//2048!=y//2048 for x,y in zip(positions,positions[1:]))
  if len(tokens)>4096 and cross and max(count.values())>1024:
   result=dict(verdict='PASS',prefix_repeats=repeat,prompt_tokens=len(tokens),image_rows=len(positions),max_chunk_image_rows=max(count.values()),crosses_chunk=True,case_id=case['id'],cache_tensor_byte_equal=True,wire_order_equal=True,cuda_initialized=torch.cuda.is_initialized());break
 assert result is not None and not result['cuda_initialized']
 a.out.write_text(json.dumps(result,indent=2)+'\n');print('VISION_MULTI_TOKENIZER PASS '+json.dumps(result))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  import traceback
  frame=traceback.extract_tb(exc.__traceback__)[-1];print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {exc} at {frame.filename}:{frame.lineno}',file=sys.stderr);raise SystemExit(2)
