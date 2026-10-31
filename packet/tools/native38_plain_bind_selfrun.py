#!/usr/bin/env python3
"""Exact frozen native38 source/overlay, pinned image, CPU and network none."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from native37_robust_selfrun import IMAGE,overlay_mounts

ROOT=Path(__file__).resolve().parents[1]
SUITES=('test_native38_plain_bind','test_e3_native_startup_binding','test_e3_native_overlap',
        'test_e3_probe_control','test_e3_native_qualification_frontend',
        'test_e3_executor_threads','test_s5_reply_transport')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(bundle,out,name):
    from build_native30_held import verify
    verify(bundle);overlay=bundle/'native30_overlay';verify(overlay)
    out.mkdir();out.chmod(0o777)
    command=['sudo','-n','docker','run','--pull=never','--name',name,'--runtime','runc',
        '--network','none','--cpus','4','--memory','6g','--memory-swap','6g','--read-only',
        '--tmpfs','/tmp:rw,exec,nosuid,nodev,size=1g','--ulimit','core=0',
        '-e','NVIDIA_VISIBLE_DEVICES=none','-e','CUDA_VISIBLE_DEVICES=',
        '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/source/harness:/source/harness/e3_builder']
    mounts=dict(overlay_mounts(overlay));mounts.update({bundle:'/source/harness',
        bundle/'e3_builder':'/source/patches/s5_engine',
        bundle/'native35_tools':'/source/tools',
        (ROOT/'reference/vllm-src').resolve():'/source/reference/vllm-src'})
    for source,target in mounts.items():command+=['-v',str(source)+':'+target+':ro']
    command+=['-v',str(out)+':/out:rw']
    code="import sys,unittest,torch,json; from pathlib import Path; files=sorted(Path('/source/harness').rglob('*.py')); [compile(p.read_bytes(),str(p),'exec') for p in files]; Path('/out/compile.json').write_text(json.dumps(dict(files=len(files),failures=[],verdict='PASS'))); assert not torch.cuda.is_initialized(); suite=unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:]); r=unittest.TextTestRunner(verbosity=2).run(suite); import e3_native38_cpu_probe; e3_native38_cpu_probe.run(); assert not torch.cuda.is_initialized(); raise SystemExit(0 if r.wasSuccessful() and not r.skipped else 1)"
    command+=['--entrypoint','python3',IMAGE,'-B','-c',code,*SUITES]
    start=time.monotonic()
    with (out/'container.log').open('x') as log:rc=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
    inspect=json.loads(subprocess.check_output(['sudo','-n','docker','inspect',name],text=True))[0]
    assert not inspect['HostConfig'].get('DeviceRequests') and not inspect['HostConfig'].get('Devices')
    assert inspect['HostConfig']['NetworkMode']=='none'
    import re
    text=(out/'container.log').read_text();matches=re.findall(r'^Ran (\d+) tests? in ',text,re.M)
    record=dict(schema='native38-selfrun/1',verdict='PASS' if rc==0 and matches and '\nOK\n' in text else 'FAIL',
        classification=None if rc==0 else 'HARNESS',compile_proof=json.loads((out/'compile.json').read_text()) if (out/'compile.json').exists() else None,health_proof=json.loads((out/'health-probe.json').read_text()) if (out/'health-probe.json').exists() else None,rc=rc,tests_run=int(matches[-1]) if matches else None,
        skips=0 if 'skipped=' not in text else None,container=name,image=IMAGE,
        bundle=str(bundle),manifest=sha(bundle/'SHA256SUMS'),overlay_manifest=sha(overlay/'SHA256SUMS'),
        gpu_execution=False,network='none',gpu_device_requests=0,wall_s=round(time.monotonic()-start,3),
        command=command,log_sha256=sha(out/'container.log'))
    with (out/'result.json').open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({k:record[k] for k in ('verdict','rc','tests_run','skips','wall_s')}));print('result_sha256',sha(out/'result.json'))
    return 0 if record['verdict']=='PASS' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args();raise SystemExit(run(a.bundle.resolve(),a.out.resolve(),a.name))
