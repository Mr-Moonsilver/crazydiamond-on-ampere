#!/usr/bin/env python3
"""Owner-only serialized candidate chain. Codex never executes it on the box.

Requires a separate ARM GO and an already stopped engine. Reads the freshly
minted owner-private env file; its value is never in argv or output. This
external runner does not modify the immutable harness or runtime package.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode=True
H=Path('/home/<service-user>/harness/038e5c95e7f8')
S0=Path('/home/<service-user>/s0')
QNAME='q18a-native1'
OVERLAY=Path('/data/vllm/flash-next-tp4/prod/overlay-v16s5-q18a-native1')
BUNDLE='038e5c95e7f8c1120a44713055f48f2c44f9e658ca6522d6272e71114aba4f4b'
MANIFEST='b232748413431bc5ad3cceee767f9c099f865389fdc0d5e1e83caa729dd64784'
GREEN='e06b062263dcc315dec1ab279f1c08abc8f627e4c6f901ef5a740ec307cba41b'
OBSERVER='05a4846d17699a692e923882fcf6c7e8eb4cf486f1888823ac1e4072695ac06d'
LATEST_QUEUE_START=datetime.datetime(2026,10,28,1,45,tzinfo=datetime.timezone.utc).timestamp()
_REDACTIONS=[]  # credential values stay in this owner process's memory only


class Refused(ValueError):
    pass


def require(condition,reason):
    if not condition:raise Refused(reason)


def sha(path):
    require(path.is_file() and not path.is_symlink(),'regular-public-file-required')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_slot(now):
    require(now<=LATEST_QUEUE_START,'ARM-cutoff-passed')


def environment(key,model,cap,original=None):
    require(type(key) is str and re.fullmatch('[0-9a-f]{64}',key) is not None,'fresh-64-hex-key-required')
    require(type(model) is str and model and len(model)<=200 and not any(x in model for x in '\r\n'),'model-required')
    require(type(cap) is int and 64*1024**2<=cap<=2*1024**3,'snapshot-cap-invalid')
    if key not in _REDACTIONS:_REDACTIONS.append(key)
    env=dict(os.environ if original is None else original)
    for name in list(env):
        if name.startswith('Q17_TEST_') or name in ('Q18A_OVERLAP_BOUNDED','S5_LAUNCHER_SHA','CONTROLLER_CMD'):
            env.pop(name)
    env.update(HARNESS_DIR=str(H),S0=str(S0),QNAME=QNAME,
        S5_OVERLAY=str(OVERLAY),S5_GREEN_LIB_SHA256=GREEN,S5_REPLAY_LIB_SHA256=OBSERVER,
        S5_LAUNCHER_NAME='run-flash-s5-test.sh',S5_REQUESTED_DECODE_SMS='44',
        S5_MODEL=model,Q18A_TARGET_LABEL='s5-overlap-80-80',
        Q18A_READINESS=str(OVERLAY/'e3/native-overlap-readiness.json'),
        S5_INVARIANT_MAX_BYTES=str(cap),PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=os.pathsep.join(map(str,(H/'e3_builder',H/'e2_builder',H))),
        PROBE_URL='http://127.0.0.1:8000',VLLM_API_KEY=key)
    return env


def available_memory():
    info={line.split(':')[0]:int(line.split()[1])*1024
          for line in Path('/proc/meminfo').read_text().splitlines()
          if len(line.split())>=2 and line.split()[1].isdigit()}
    available=info['MemAvailable']
    group=Path('/sys/fs/cgroup')
    if (group/'memory.current').is_file() and (group/'memory.max').is_file():
        current=(group/'memory.current').read_text().strip()
        maximum=(group/'memory.max').read_text().strip()
        if current.isdigit() and maximum.isdigit():available=min(available,int(maximum)-int(current))
    return available


def public_output(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=30)
    require(p.returncode==0,'public-preflight-command-failed')
    return p.stdout.strip()


def preflight(cap):
    check_slot(time.time())
    require(type(cap) is int and 64*1024**2<=cap<=2*1024**3,'snapshot-cap-invalid')
    require(os.geteuid()==0,'owner-root-required')
    require(sha(H/'SHA256SUMS')==BUNDLE,'harness-manifest-pin')
    subprocess.run(['sha256sum','--quiet','-c','SHA256SUMS'],cwd=H,check=True,timeout=120)
    require(sha(OVERLAY/'SHA256SUMS')==MANIFEST,'overlay-manifest-pin')
    subprocess.run(['sha256sum','--quiet','-c','SHA256SUMS'],cwd=OVERLAY,check=True,timeout=120)
    require(sha(OVERLAY/'e3/native/flash_s5_green.so')==GREEN,'green-library-pin')
    require(sha(OVERLAY/'e3/native/flash_s5_replay_observer.so')==OBSERVER,'observer-library-pin')
    sys.path[:0]=[str(H/'e3_builder'),str(H)]
    from native_readiness import verify_finalized_package
    from decode_probe import GENERATED_SOURCES
    verify_finalized_package(OVERLAY,GENERATED_SOURCES)
    state=public_output(['systemctl','show','vllm-flash','-p','ActiveState','--value'])
    require(state=='inactive','owner-must-stop-production-first')
    raw=public_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'])
    devices=[tuple(int(part.strip()) for part in line.split(',')) for line in raw.splitlines()]
    require(len(devices)==4 and {i for i,_ in devices}==set(range(4)) and all(v==0 for _,v in devices),'four-idle-gpus-required')
    staged=Path('/etc/default/vllm-flash.next')
    require(not staged.exists() and not staged.is_symlink(),'defaults-next-present')
    for suffix in ('','-requests','.log','-declare.log'):
        p=S0/(QNAME+suffix)
        require(not p.exists() and not p.is_symlink(),'fresh-output-path-required')
    available=available_memory();required=4*cap+5*1024**3
    require(available>=required,'stopped-host-headroom-insufficient')
    print('ARM_GUARD PASS available_bytes='+str(available)+' required_bytes='+str(required),flush=True)


def phase(script,log,env):
    with log.open('xb') as handle:
        return subprocess.run(['bash',str(H/'runs'/script)],env=env,
                              stdout=handle,stderr=subprocess.STDOUT).returncode


def diagnostic(exc):
    message=('input-encoding-invalid' if isinstance(exc,UnicodeError) else str(exc))
    for secret in (*_REDACTIONS,os.environ.get('VLLM_API_KEY')):
        if type(secret) is str and secret:message=message.replace(secret,'***')
    return {'error_type':type(exc).__name__,
            'error_message':message.replace('\n',' ').replace('\r',' ')[:512],
            'error_stack':[{'file':Path(frame.filename).name,'line':frame.lineno,
                            'function':frame.name}
                           for frame in traceback.extract_tb(exc.__traceback__)[-8:]]}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--auth-env-file',type=Path,required=True)
    p.add_argument('--model',default='qwen38-w8a8-tp4')
    p.add_argument('--snapshot-max-bytes',type=int,default=2*1024**3)
    a=p.parse_args(argv)
    preflight(a.snapshot_max_bytes)
    from e3_private_auth import read_private
    env=environment(read_private(a.auth_env_file),a.model,a.snapshot_max_bytes)
    rc=phase('q18a_native1_declare.sh',S0/(QNAME+'-declare.log'),env)
    print('DECLARE rc='+str(rc),flush=True)
    if rc:
        print('OWNER_CHAIN DONE rc='+str(rc)+' queue_started=false',flush=True)
        return rc
    check_slot(time.time())  # declaration does not grant a late queue entry
    env['REQUESTS']=str(S0/(QNAME+'-requests'))
    env['Q']=str(S0/QNAME)
    rc=phase('q18a_native1.draft.sh',S0/(QNAME+'.log'),env)
    print('QUEUE rc='+str(rc),flush=True)
    print('OWNER_CHAIN DONE rc='+str(rc)+' queue_started=true',flush=True)
    return rc


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        # Bounded location/message diagnostics, never key data or source/locals.
        print('OWNER_CHAIN INCONCLUSIVE '+json.dumps(diagnostic(exc),sort_keys=True),flush=True)
        raise SystemExit(2)
