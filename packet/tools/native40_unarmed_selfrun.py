#!/usr/bin/env python3
"""Exact source/overlay CPU self-run. No device, network or container deletion."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from native37_robust_selfrun import IMAGE,overlay_mounts

ROOT=Path(__file__).resolve().parents[1]
SUITES=('test_native40_unarmed_hold','test_native39_trunk','test_native37_robust','test_native38_plain_bind',
        'test_e3_admission_native','test_e3_admission_metadata','test_e3_staged_chunk_plan',
        'test_e3_native_overlap','test_e3_native_storage_replay','test_e3_r4_worker_bridge')
NEGATIVES=('test_native40_unarmed_hold.UnarmedHold.test_all_stock_prefill_branches_hold_without_native_admission',
           'test_native40_unarmed_hold.UnarmedHold.test_unarmed_timeout_returns_full_without_inventing_native_refusal',
           'test_native39_trunk.HeldKinds.test_positive_frontier_and_running_prefill_hold_on_split_even_unarmed')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(bundle,out,name,overlay=None,sources=None,negative=False):
    from build_native30_held import verify
    bundle=bundle.resolve();verify(bundle)
    overlay=overlay.resolve() if overlay else bundle/'native30_overlay'
    verify(overlay)
    out=out.resolve();out.mkdir();out.chmod(0o777)
    sources=sources.resolve() if sources else bundle/'e3_builder'
    script='''import json,sys,unittest
from pathlib import Path

if __name__ == '__main__':
    import torch
    files=list(Path('/source/harness').rglob('*.py'))
    for p in files: compile(p.read_bytes(),str(p),'exec')
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:]))
    assert not torch.cuda.is_initialized()
    Path('/out/suites.json').write_text(json.dumps(dict(tests=r.testsRun,skips=len(r.skipped),failures=len(r.failures),errors=len(r.errors),ok=r.wasSuccessful(),compiled=len(files),cuda_initialized=False)))
    raise SystemExit(0 if r.wasSuccessful() and not r.skipped else 1)
'''
    (out/'suites.py').write_text(script)
    (out/'inner.sh').write_text('cd /source/harness\npython3 -B /out/suites.py '+ ' '.join(NEGATIVES if negative else SUITES)+' > /out/suites.log 2>&1\necho $? > /out/suites.rc\npython3 -B e3_native40_unarmed_probe.py --out /out/connected.json > /out/connected.log 2>&1\necho $? > /out/connected.rc\n')
    c=['sudo','-n','docker','run','--pull=never','--name',name,'--runtime','runc','--network','none','--cpus','4','--memory','6g','--memory-swap','6g','--read-only','--tmpfs','/tmp:rw,exec,nosuid,nodev,size=1g','--ulimit','core=0',
       '-e','VLLM_CACHE_ROOT=/tmp/vllm-cache','-e','NVIDIA_VISIBLE_DEVICES=none','-e','CUDA_VISIBLE_DEVICES=','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/source/harness:/source/harness/e3_builder']
    mounts=dict(overlay_mounts(overlay));mounts.update({bundle:'/source/harness',sources:'/source/patches/s5_engine',
        bundle/'native35_tools':'/source/tools',(ROOT/'reference/vllm-src').resolve():'/source/reference/vllm-src'})
    for s,t in mounts.items():c+=['-v',f'{s}:{t}:ro']
    c+=['-v',f'{out}:/out','--entrypoint','bash',IMAGE,'/out/inner.sh']
    start=time.monotonic()
    with (out/'container.log').open('x') as f:rc=subprocess.call(c,stdout=f,stderr=subprocess.STDOUT)
    inspect=json.loads(subprocess.check_output(['sudo','-n','docker','inspect',name],text=True))[0]
    assert not inspect['HostConfig'].get('DeviceRequests') and not inspect['HostConfig'].get('Devices')
    assert inspect['HostConfig']['NetworkMode']=='none'
    def read(n):return json.loads((out/n).read_text()) if (out/n).exists() else {}
    suites,connected=read('suites.json'),read('connected.json')
    if negative:
        ok=(suites.get('tests')==len(NEGATIVES) and suites.get('failures',0)==len(NEGATIVES)
            and not suites.get('errors') and suites.get('skips')==0
            and connected.get('verdict')=='FAIL' and connected.get('error_type')=='UnsupportedRoute')
    else:ok=suites.get('ok') is True and suites.get('skips')==0 and connected.get('verdict')=='PASS'
    result=dict(schema='native40-selfrun/1',verdict='PASS' if rc==0 and ok else 'FAIL',negative_control=negative,
        classification=None if rc==0 and ok else 'HARNESS',container_rc=rc,suites=suites,connected=connected,
        connected_sha256=sha(out/'connected.json') if connected else None,container=name,image=IMAGE,
        bundle=str(bundle),manifest=sha(bundle/'SHA256SUMS'),overlay=str(overlay),overlay_manifest=sha(overlay/'SHA256SUMS'),
        sources=str(sources),network='none',gpu_execution=False,gpu_device_requests=0,box_action=False,
        wall_s=round(time.monotonic()-start,3),command=c,
        log_shas={p.name:sha(p) for p in out.glob('*.log')})
    (out/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('verdict','container_rc','wall_s')}),sha(out/'result.json'))
    return 0 if result['verdict']=='PASS' else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--overlay',type=Path);p.add_argument('--sources',type=Path);p.add_argument('--negative',action='store_true');a=p.parse_args();raise SystemExit(run(a.bundle,a.out,a.name,a.overlay,a.sources,a.negative))
