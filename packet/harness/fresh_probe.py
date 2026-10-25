#!/usr/bin/env python3
"""One fresh 200k completion; no repeat, completion/usage evidence required."""
import json,sys,time,urllib.request,uuid
from pathlib import Path

def main(url,model,out):
    def post(path,body):
        req=urllib.request.Request(url+path,json.dumps(body).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=1800) as r:return json.load(r)
    text=Path('/home/<service-user>/p2p-tests/corpus.txt').read_text()
    toks=post('/tokenize',{'model':model,'prompt':text[:1200000]})['tokens'];assert toks
    toks=(toks*(200000//len(toks)+1))[:200000]
    t=time.monotonic()
    response=post('/v1/completions',{'model':model,'prompt':toks,'max_tokens':1,'min_tokens':1,'ignore_eos':True,
                                  'temperature':0,'cache_salt':uuid.uuid4().hex})
    seconds=time.monotonic()-t
    usage=response['usage'];assert response['choices'][0]['finish_reason'] is not None
    assert usage['prompt_tokens']==200000 and usage['completion_tokens']==1
    cached=usage['prompt_tokens_details']['cached_tokens'];assert cached==0
    Path(out).write_text(json.dumps(dict(seconds=seconds,prompt_tokens=200000,completion_tokens=1,cached_tokens=cached)))
if __name__=='__main__':main(*sys.argv[1:])
