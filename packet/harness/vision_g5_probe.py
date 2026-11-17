#!/usr/bin/env python3
"""G5 raw-image capture through the declared public/LiteLLM route, with private receipts."""
import argparse,base64,hashlib,json,os,sys,time,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from packet_client_contract import report_exception

def token(path,name):
    if path.is_symlink() or path.stat().st_mode&0o777!=0o600:raise ValueError('key file must be private0600')
    values=dict(line.split('=',1) for line in path.read_text().splitlines() if '=' in line)
    value=values.get(name,'').strip().strip('\"\'')
    if not value or '\n' in value or '\r' in value:raise ValueError('named key missing')
    return value

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',required=True);p.add_argument('--key-file',type=Path,required=True);p.add_argument('--key-name',default='VLLM_API_KEY')
    p.add_argument('--model',default='qwen38-w8a8-tp4');p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--phase',choices=('reference','production'),required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv);os.umask(0o077);a.out.parent.mkdir(parents=True,exist_ok=True)
    record=dict(schema='w27-g5/1',phase=a.phase,verdict='HOLD',cases=[],base_url=a.base_url)
    with a.out.open('x') as f:json.dump(record,f)
    k=token(a.key_file,a.key_name);cases=json.loads(a.cases.read_text())
    if len(cases)!=4 or len({c['id'] for c in cases})!=4:raise ValueError('four fixed cases required')
    for case in cases:
        image=(a.cases.parent/case['image']).resolve()
        if not image.is_relative_to(a.cases.parent.resolve()):raise ValueError('image outside packet')
        raw=image.read_bytes()
        if not 0<len(raw)<=8*2**20:raise ValueError('image byte bound')
        body=dict(model=a.model,messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(raw).decode())),dict(type='text',text=case['question'])])],temperature=0,max_tokens=128,seed=17,chat_template_kwargs=dict(enable_thinking=False))
        started=time.time();request=urllib.request.Request(a.base_url.rstrip('/')+'/v1/chat/completions',json.dumps(body).encode(),{'Authorization':'Bearer '+k,'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=300) as response:
            answer=json.loads(response.read(2**20));text=answer['choices'][0]['message']['content']
            usage=answer.get('usage') or {}
            if not isinstance(text,str) or not text.strip() or type(usage.get('completion_tokens')) is not int:raise ValueError('answer/usage missing')
            record['cases'].append(dict(id=case['id'],image_sha256=hashlib.sha256(raw).hexdigest(),question=case['question'],expected=case['expected'],answer=text,started_wall_ns=int(started*1e9),finished_wall_ns=time.time_ns(),wall_s=time.time()-started,output_tokens=usage['completion_tokens'],call_id=response.headers.get('x-litellm-call-id'),response_model=answer.get('model')))
        a.out.write_text(json.dumps(record,indent=2)+'\n')
    record['verdict']='COMPLETE';a.out.write_text(json.dumps(record,indent=2)+'\n');print('W27_G5_CAPTURE COMPLETE '+a.phase)
    return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:report_exception(exc);print('W27_G5_CAPTURE HOLD incomplete');raise SystemExit(2)
