#!/usr/bin/env python3
"""CT-side, CPU-only canary handshake. Does not control engine or traffic routing."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from pp_release_grade import canary


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,obj):
    path=Path(path)
    staged=path.with_name(path.name+'.writing')
    with staged.open('x') as f:
        json.dump(obj,f,indent=2); f.flush(); os.fsync(f.fileno())
    # Atomic, refuses an existing destination. Keep the staging hardlink as
    # evidence; no deletion or replacement of prior handshake artifacts.
    os.link(staged,path)


def engine():
    raw=subprocess.check_output(['docker','inspect','qwen-flash-prod'],timeout=30)
    d=json.loads(raw)[0]
    return d['Id'],d['State']['StartedAt'],d['State']['Running']


def metrics():
    text=urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=10).read().decode()
    values={}
    for name in ('num_requests_running','num_requests_waiting','request_success_total'):
        nums=[float(l.split()[-1]) for l in text.splitlines() if re.match(r'^vllm:'+name+r'(\{| )',l)]
        if not nums or any(not math.isfinite(v) or v<0 for v in nums):
            raise ValueError('missing/invalid engine metric '+name)
        values[name]=sum(nums)
    return values


def status(q, idle=False):
    r=json.loads((q/'canary.request').read_text())
    if (q/'canary.cancel').exists() or time.time()>r['deadline_wall']:
        raise ValueError('canary request expired/cancelled')
    eid,start,running=engine()
    if not running or [eid,start]!=r['engine']:
        raise ValueError('engine changed')
    if digest(Path(r['overlay'])/'core/SHA256SUMS') != r['overlay_core_sha256']:
        raise ValueError('overlay changed')
    m=metrics()
    if idle and (m['num_requests_running'] or m['num_requests_waiting']):
        raise ValueError('engine is not idle')
    return dict(request=r,metrics=m)


def main():
    cmd=sys.argv[1]; q=Path(sys.argv[2]).resolve()
    if cmd=='prepare':
        h,ovl,pin=map(Path,sys.argv[3:6]); n,rounds=map(int,sys.argv[6:8])
        eid,start,running=engine()
        if not running: raise ValueError('engine is stopped')
        write(q/'canary.request',dict(schema='pp-canary/v1',id=uuid.uuid4().hex,
            created_wall=time.time(),deadline_wall=time.time()+2700,N=n,ROUNDS=rounds,
            engine=[eid,start],overlay=str(ovl),overlay_core_sha256=digest(ovl/'core/SHA256SUMS'),
            harness_sha256=digest(h/'SHA256SUMS'),request_sha256=digest(pin),
            identity_before_sha256=digest(q/'canary.before.json')))
        return
    if cmd in ('status','idle'):
        print(json.dumps(status(q,cmd=='idle'))); return
    if cmd=='claim':
        r=status(q,True)['request']; data=json.load(sys.stdin)
        if data['id']!=r['id']: raise ValueError('wrong claim id')
        write(q/'canary.claim',data); return
    if cmd=='archive':
        r=status(q)['request']; expected,size=sys.argv[3],int(sys.argv[4])
        if size<=0 or size>100*2**20: raise ValueError('invalid archive size')
        data=sys.stdin.buffer.read(size+1)
        if len(data)!=size or hashlib.sha256(data).hexdigest()!=expected: raise ValueError('archive mismatch')
        with (q/'canary.raw.tar.gz').open('xb') as f: f.write(data)
        return
    if cmd=='publish':
        r=status(q,True)['request']; data=json.load(sys.stdin)
        if data.get('id')!=r['id'] or data.get('engine')!=r['engine']: raise ValueError('result identity mismatch')
        if digest(q/'canary.raw.tar.gz')!=data['archive_sha256']: raise ValueError('missing/mismatched raw archive')
        claim=json.loads((q/'canary.claim').read_text())
        if claim['id']!=r['id'] or data['metrics_start']!=claim['metrics_start']: raise ValueError('claim mismatch')
        now=metrics()['request_success_total']
        # CT independently rechecks current success counter, against claimed idle start.
        if now!=data['metrics_end']['request_success_total']: raise ValueError('traffic arrived after hub grading')
        delta=now-claim['metrics_start']['request_success_total']
        verdict=canary(data['grade'],r['N'],r['ROUNDS'],delta,data.get('traffic'),r['id'])
        if data.get('runner_rc')!=0 and verdict['verdict']=='PASS':
            verdict=dict(verdict='FAIL',fails=['canary runner failed'])
        data['release_grade']=verdict
        write(q/'canary.result',data); print(verdict['verdict']); return
    if cmd=='grade':
        r=json.loads((q/'canary.request').read_text()); d=json.loads((q/'canary.result').read_text())
        if d['id']!=r['id'] or d['engine']!=r['engine'] or digest(q/'canary.raw.tar.gz')!=d['archive_sha256']:
            raise ValueError('result binding failed')
        claim=json.loads((q/'canary.claim').read_text())
        if claim['id']!=r['id'] or d['metrics_start']!=claim['metrics_start']: raise ValueError('claim mismatch')
        grade=canary(d['grade'],r['N'],r['ROUNDS'],d['metrics_end']['request_success_total']-d['metrics_start']['request_success_total'],d.get('traffic'),r['id'])
        if d['runner_rc']!=0: grade['verdict']='FAIL'
        print(json.dumps(grade,indent=2))
        raise SystemExit({'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[grade['verdict']])
    raise ValueError('unknown bridge command')


if __name__=='__main__':
    try: main()
    except Exception as e:
        # Do not dump docker output, settings, env or authentication material.
        print(json.dumps({'verdict':'INCONCLUSIVE','error':type(e).__name__}),file=sys.stderr)
        raise SystemExit(2)
