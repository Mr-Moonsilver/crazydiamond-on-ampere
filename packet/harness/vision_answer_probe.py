#!/usr/bin/env python3
"""Owner-run trusted embeddings client, or the same raw images on the reference model."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parent))
from vision_encoder_service import error_info
from vision_encoder_backend import MODEL_MANIFEST_SHA


def key(path, name):
    if path.is_symlink() or path.stat().st_mode & 0o777 != 0o600:
        raise ValueError('private0600 key file required')
    fields=dict(line.split('=',1) for line in path.read_text().splitlines() if '=' in line)
    value=fields.get(name,'')
    if not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('missing named API key')
    return value


def post(url, token, body):
    req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={
        'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=300) as response:
        if response.status!=200:raise ValueError('API did not return200')
        return json.loads(response.read(10*2**20))


def validate_embedding(response):
    if (response.get('schema')!='flash-next-image-embeds/1'
            or response.get('manifest_sha256')!=MODEL_MANIFEST_SHA
            or response.get('dtype')!='bfloat16' or len(response.get('shape',[]))!=2
            or type(response['shape'][0]) is not int or type(response['shape'][1]) is not int
            or response['shape'][1]!=2560 or not 0<response['shape'][0]<=1024
            or not re.fullmatch('[0-9a-f]{64}',response.get('embedding_sha256',''))
            or response.get('content',{}).get('type')!='image_embeds'):
        raise ValueError('encoder identity/shape differs')
    return response


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('embeddings','reference'),required=True)
    p.add_argument('--engine-url',required=True);p.add_argument('--engine-key-file',type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--encoder-url',default='http://<encoder-host>:9041')
    p.add_argument('--encoder-key-file',type=Path);p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--embeddings-cache',type=Path);p.add_argument('--embeddings-cache-sha256')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    if (a.embeddings_cache is None)!=(a.embeddings_cache_sha256 is None):
        p.error('--embeddings-cache and its SHA are required together')
    if a.mode=='embeddings' and a.encoder_key_file is None and a.embeddings_cache is None:
        p.error('--encoder-key-file required in embeddings mode')
    engine_key=key(a.engine_key_file,'VLLM_API_KEY')
    encoder_key=key(a.encoder_key_file,'VISION_ENCODER_API_KEY') if a.mode=='embeddings' and a.embeddings_cache is None else None
    cache=None
    if a.embeddings_cache is not None:
        if (a.mode!='embeddings' or a.embeddings_cache.is_symlink()
                or a.embeddings_cache.stat().st_mode & 0o777 != 0o600
                or a.embeddings_cache.stat().st_size>64*2**20):
            raise ValueError('bounded private embedding cache required')
        raw=a.embeddings_cache.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=a.embeddings_cache_sha256:
            raise ValueError('embedding cache SHA differs')
        cache=json.loads(raw)
        if (cache.get('schema')!='vision-embedding-cache/1' or cache.get('manifest_sha256')!=MODEL_MANIFEST_SHA
                or not 1<=len(cache.get('features',{}))<=8):
            raise ValueError('embedding cache identity/inventory differs')
    cases=json.loads(a.cases.read_text())
    if not isinstance(cases,list) or not 1<=len(cases)<=8:raise ValueError('bounded case list required')
    record=dict(schema='vision-answer-probe/1',mode=a.mode,model=a.model,verdict='HOLD',cases=[])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as output:output.write(json.dumps(record)+'\n')
    for case in cases:
        image=(a.cases.parent/case['image']).resolve()
        if not image.is_relative_to(a.cases.parent.resolve()):raise ValueError('case image outside packet')
        raw=image.read_bytes()
        if not 0<len(raw)<=8*2**20:raise ValueError('case image byte bound')
        started=time.monotonic();features=None
        if a.mode=='embeddings':
            response=validate_embedding(cache['features'][hashlib.sha256(raw).hexdigest()] if cache is not None
                else post(a.encoder_url+'/v1/encode',encoder_key,dict(image_base64=base64.b64encode(raw).decode())))
            content=response['content']
            features={k:response[k] for k in ('manifest_sha256','shape','embedding_sha256')}
        else:
            content=dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(raw).decode()))
        answer=post(a.engine_url+'/v1/chat/completions',engine_key,dict(model=a.model,
            messages=[dict(role='user',content=[content,dict(type='text',text=case['question'])])],
            temperature=0,max_tokens=128,seed=17,chat_template_kwargs=dict(enable_thinking=False)))
        text=answer['choices'][0]['message']['content']
        if not isinstance(text,str) or not text.strip():raise ValueError('answer text missing')
        record['cases'].append(dict(id=case['id'],image_sha256=hashlib.sha256(raw).hexdigest(),
            question=case['question'],expected=case['expected'],answer=text,
            features=features,response_model=answer.get('model'),usage=answer.get('usage'),wall_s=time.monotonic()-started))
        a.out.write_text(json.dumps(record,indent=2)+'\n')
    record['verdict']='COMPLETE';a.out.write_text(json.dumps(record,indent=2)+'\n')
    print('VISION_ANSWER_PROBE COMPLETE mode='+a.mode+' cases='+str(len(record['cases'])))
    return 0


if __name__=='__main__':
    try:code=main()
    except SystemExit as exc:
        if exc.code:print('PACKET_CLIENT_ERROR '+json.dumps(error_info(exc)),file=sys.stderr)
        raise
    except Exception as exc:
        print('PACKET_CLIENT_ERROR '+json.dumps(error_info(exc)),file=sys.stderr)
        print('VISION_ANSWER_PROBE HOLD '+type(exc).__name__)
        raise SystemExit(2)
    raise SystemExit(code)
