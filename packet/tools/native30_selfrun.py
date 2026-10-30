#!/usr/bin/env python3
"""Run exact frozen native30 bytes in the pinned image, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
IMAGE='sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root):
    rows={}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        pin,name=line.split(None,1);assert sha(root/name)==pin,name;rows[name]=pin
    assert set(rows)=={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'SHA256SUMS'}
def run(bundle,out,name):
    bundle=bundle.resolve();verify(bundle);overlay=bundle/'native30_overlay';verify(overlay)
    out=out.resolve();out.mkdir();out.chmod(0o777)
    command=['sudo','-n','docker','run','--pull=never','--name',name,'--runtime','runc','--network','none','--cpus','4','--memory','4g','--memory-swap','4g','--read-only','--tmpfs','/tmp:rw,exec,nosuid,nodev,size=512m','--ulimit','core=0']
    for setting in ('NVIDIA_VISIBLE_DEVICES=none','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','FLASH_NATIVE30_LOCALIZE=1','FLASH_NATIVE30_SYNC=none','PYTHONPATH=/plugin:/source/harness:/source/harness/e3_builder'):
        command+=['-e',setting]
    mounts={bundle:'/source/harness',bundle/'native30_tools':'/source/tools',bundle/'e3_builder':'/source/patches/s5_engine',bundle/'e3_reference':'/source/reference/vllm-src',overlay/'e3/native30_plugins':'/plugin',overlay/'e3':'/usr/local/lib/python3.12/dist-packages/vllm/v1/flash_e3',overlay/'e2':'/usr/local/lib/python3.12/dist-packages/vllm/v1/flash_e2',overlay/'e3_sources/worker.py':'/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py'}
    for source,target in mounts.items():command+=['-v',f'{source}:{target}:ro']
    command+=['-v',f'{out}:/out','--entrypoint','python3',IMAGE,'-B','/source/tools/native30_cpu_probe.py']
    started=time.monotonic()
    with (out/'container.log').open('x') as log:rc=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
    probe=json.loads((out/'probe.json').read_text()) if (out/'probe.json').exists() else {}
    inspect=json.loads(subprocess.check_output(['sudo','-n','docker','inspect',name],text=True))[0]
    devices=inspect['HostConfig'].get('DeviceRequests') or []
    assert not devices and not inspect['HostConfig'].get('Devices')
    result=dict(probe,schema='native30-selfrun/1',verdict='PASS' if rc==0 and probe.get('verdict')=='PASS' else 'FAIL',classification=None if rc==0 else 'HARNESS',rc=rc,bundle_manifest=sha(bundle/'SHA256SUMS'),overlay_manifest=sha(overlay/'SHA256SUMS'),image=IMAGE,gpu_device_requests=len(devices),box_action=False,container=name,wall_s=round(time.monotonic()-started,3),command=command)
    (out/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result,sort_keys=True));return rc
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args();raise SystemExit(run(a.bundle,a.out,a.name))
