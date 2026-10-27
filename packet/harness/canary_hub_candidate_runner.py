#!/usr/bin/env python3
"""Hub client for the bounded CT canary request; owner launches it after queue pinning."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from canary_grade import grade
from pp_release_grade import canary
from canary_relay import Relay
from e3_private_auth import read_private

H=Path(__file__).resolve().parent
SSH=['ssh','-F','/dev/null','-i',str(Path.home()/'.ssh/<ssh-key>'),'-o','IdentitiesOnly=yes',
     '-o','BatchMode=yes','-o','ConnectTimeout=10','root@<pve-host>']


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def safe_path(p,root):
    p=Path(p)
    if not p.is_absolute() or '..' in p.parts or not str(p).startswith(root+'/') or not re.fullmatch(r'[A-Za-z0-9_./-]+',str(p)):
        raise ValueError('invalid path')
    return str(p)


def remote(ct_h,q,cmd,*args,data=None):
    command=shlex.join(['pct','exec','<ct-id>','--','python3',ct_h+'/pp_canary_bridge.py',cmd,q,*map(str,args)])
    p=subprocess.run(SSH+[command],input=data,capture_output=True,timeout=90)
    if p.returncode: raise RuntimeError('CT bridge '+cmd+' failed (output suppressed)')
    return p.stdout


def prepare_client_config(settings, auth_path, out):
    """Private per-run settings, no secret argv or shared configuration write."""
    if not auth_path: raise ValueError('candidate canary private auth file required')
    token=read_private(auth_path)
    candidate_config=out/'candidate-client-config'
    candidate_config.mkdir(mode=0o700)
    candidate_settings=dict(settings)
    candidate_settings['env']=dict(settings.get('env') or {},ANTHROPIC_AUTH_TOKEN=token)
    fd=os.open(candidate_config/'settings.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as handle: json.dump(candidate_settings,handle)
    return candidate_config


def main():
    q=safe_path(sys.argv[1],'/home/<service-user>/s0')
    ct_h=safe_path(sys.argv[2],'/home/<service-user>/harness')
    out=Path(safe_path(sys.argv[3],'<hub-dir>/agents/flash-next'))
    out.mkdir(mode=0o700,parents=False,exist_ok=False)
    subprocess.run(['sha256sum','--quiet','-c','SHA256SUMS'],cwd=H,check=True)
    config=Path('/home/<service-user>/.claude-local')
    settings=json.loads((config/'settings.json').read_text())
    if settings.get('env',{}).get('ANTHROPIC_BASE_URL')!='http://<gateway-host>:8000':
        raise ValueError('Claude local route differs from reviewed gateway')
    config=prepare_client_config(settings,os.environ.get('CANARY_CLIENT_AUTH_FILE'),out)
    client=Path('/home/<service-user>/.local/bin/claude')
    if not client.is_file(): raise ValueError('missing Claude client')
    deadline=time.monotonic()+4*3600
    while True:
        try:
            info=json.loads(remote(ct_h,q,'idle')); break
        except RuntimeError:
            if time.monotonic()>=deadline: raise TimeoutError('no ready idle canary request')
            time.sleep(10)
    r=info['request']
    if r['harness_sha256']!=digest(H/'SHA256SUMS'): raise ValueError('hub/CT harness hashes differ')
    if r['schema']!='pp-canary/v1' or (r['N'],r['ROUNDS'])!=(10,2): raise ValueError('unexpected canary shape')
    claim=dict(id=r['id'],engine=r['engine'],metrics_start=info['metrics'])
    remote(ct_h,q,'claim',data=json.dumps(claim).encode())
    (out/'request.json').write_text(json.dumps(r,indent=2))
    work=out/'work';work.mkdir()
    run=out/'run'
    env=dict(os.environ,CLAUDE_CONFIG_DIR=str(config),CANARY_CONFIG_DIR=str(config),CANARY_WORK_ROOT=str(work))
    # Ensure inherited shell overrides cannot silently divert the configured route.
    env.pop('ANTHROPIC_BASE_URL',None)
    env.pop('ANTHROPIC_AUTH_TOKEN',None)
    env.pop('ANTHROPIC_API_KEY',None)
    relay=Relay('http://<gateway-host>:8000',
                [f'{r["id"]}-r{rd}-s{i}' for rd in (1,2) for i in range(1,11)],out/'relay.jsonl')
    relay.start()
    env.update(CANARY_PROXY_URL=relay.url,CANARY_RUN_ID=r['id'])
    child=None
    try:
        with (out/'runner.log').open('xb') as log:
            child=subprocess.Popen(['bash',str(H/'flash_canary_candidate.sh'),'10','2',str(run)],env=env,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                if time.time()>r['deadline_wall']-60: raise TimeoutError('canary deadline')
                remote(ct_h,q,'status')  # detect cancellation/engine replacement promptly
                time.sleep(5)
            rc=child.returncode
        end=None
        for _ in range(24):
            try: end=json.loads(remote(ct_h,q,'idle')); break
            except RuntimeError: time.sleep(5)
        if end is None: raise RuntimeError('engine did not drain after canary')
        projects=config/'projects'
        d=grade(str(run),str(projects))
        delta=end['metrics']['request_success_total']-info['metrics']['request_success_total']
        traffic=relay.snapshot()
        v=canary(d,10,2,delta,traffic,r['id'])
        # Copy only transcripts belonging to this run's newly allocated directories.
        dest=out/'transcripts';dest.mkdir()
        start=json.loads((run/'run-start.json').read_text())['started_wall']
        for row in (run/'expected.txt').read_text().splitlines():
            wd=row.split()[4]
            if not Path(wd).is_relative_to(work): raise ValueError('unexpected canary work directory')
            name=re.sub(r'[/.]','-',wd); src=projects/name
            (dest/name).mkdir()
            for f in src.glob('*.jsonl'):
                if f.stat().st_mtime>=start: shutil.copy2(f,dest/name/f.name)
        (out/'grade.json').write_text(json.dumps(d,indent=2))
        (out/'release-grade.json').write_text(json.dumps(v,indent=2))
        archive=out/'raw.tar.gz'
        # relay.jsonl exists only once the relay saw a request; its absence is already an INCONCLUSIVE release grade
        # (unattributed traffic) and must not stop the result from being published (q12g: FileNotFoundError here left
        # the gate waiting 45 min for a result that never came).
        with tarfile.open(archive,'x:gz') as tf:
            for name in ('run','transcripts','request.json','grade.json','release-grade.json','runner.log','relay.jsonl'):
                if (out/name).exists(): tf.add(out/name,arcname=name)
        if not (out/'relay.jsonl').exists(): v.setdefault('missing',[]).append('relay journal absent: no canary request reached the relay')
        sha=digest(archive)
        remote(ct_h,q,'archive',sha,archive.stat().st_size,data=archive.read_bytes())
        res=dict(id=r['id'],engine=r['engine'],runner_rc=rc,grade=d,release_grade=v,
                 metrics_start=info['metrics'],metrics_end=end['metrics'],archive_sha256=sha,traffic=traffic)
        remote(ct_h,q,'publish',data=json.dumps(res).encode())
        print('canary hub '+v['verdict'])
        return {'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[v['verdict']]
    finally:
        relay.close()
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL); child.wait()


if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as e:
        print('canary hub INCONCLUSIVE: '+type(e).__name__,file=sys.stderr)
        raise SystemExit(2)
