#!/usr/bin/env python3
"""Exact pinned-container CPU gates; no GPU, network or container removal."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from native37_robust_selfrun import IMAGE,overlay_mounts
ROOT=Path(__file__).resolve().parents[1]
SUITES=('test_native41_autoarm','test_native40_unarmed_hold','test_native39_trunk','test_native37_robust','test_native38_plain_bind',
        'test_e3_admission_native','test_e3_admission_metadata','test_e3_staged_chunk_plan',
        'test_e3_native_overlap','test_e3_native_storage_replay','test_e3_r4_worker_bridge')
MODES=('auto','off','abort-before','abort-live','reuse','foreign','bad-graph','refuse','pending-split','pending-full','running-prefill','below-threshold','long-id')
NEGATIVE_TEST='test_native41_autoarm.SettingAndStartup.test_auto_is_bound_after_real_startup_fence'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(bundle,out,name,overlay=None,sources=None,negative=False):
    from build_native30_held import verify
    bundle=bundle.resolve();verify(bundle);overlay=overlay.resolve() if overlay else bundle/'native30_overlay';verify(overlay)
    sources=sources.resolve() if sources else bundle/'e3_builder'
    out=out.resolve();out.mkdir();out.chmod(0o777)
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
    modes=('auto','off') if negative else MODES
    lines=['cd /source/harness', 'python3 -B /out/suites.py '+' '.join((NEGATIVE_TEST,) if negative else SUITES)+' > /out/suites.log 2>&1','echo $? > /out/suites.rc']
    for mode in modes:
        lines+=['FLASH_S5_AUTO_ARM='+('0' if mode=='off' else '1')+' python3 -B e3_native41_auto_probe.py --mode '+mode+' --out /out/'+mode+'.json > /out/'+mode+'.log 2>&1','echo $? > /out/'+mode+'.rc']
    (out/'inner.sh').write_text('\n'.join(lines)+'\n')
    c=['sudo','-n','docker','run','--pull=never','--name',name,'--runtime','runc','--network','none','--cpus','4','--memory','6g','--memory-swap','6g','--read-only','--tmpfs','/tmp:rw,exec,nosuid,nodev,size=1g','--ulimit','core=0',
       '-e','VLLM_CACHE_ROOT=/tmp/vllm-cache','-e','NVIDIA_VISIBLE_DEVICES=none','-e','CUDA_VISIBLE_DEVICES=','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/source/harness:/source/harness/e3_builder']
    mounts=dict(overlay_mounts(overlay));mounts.update({bundle:'/source/harness',sources:'/source/patches/s5_engine',bundle/'native35_tools':'/source/tools',(ROOT/'reference/vllm-src').resolve():'/source/reference/vllm-src'})
    for s,t in mounts.items():c+=['-v',f'{s}:{t}:ro']
    c+=['-v',f'{out}:/out','--entrypoint','bash',IMAGE,'/out/inner.sh']
    start=time.monotonic()
    with (out/'container.log').open('x') as f:rc=subprocess.call(c,stdout=f,stderr=subprocess.STDOUT)
    inspected=json.loads(subprocess.check_output(['sudo','-n','docker','inspect',name],text=True))[0]['HostConfig']
    assert not inspected.get('DeviceRequests') and not inspected.get('Devices') and inspected['NetworkMode']=='none'
    def read(n):return json.loads((out/n).read_text()) if (out/n).exists() else {}
    suites=read('suites.json');cases={mode:read(mode+'.json') for mode in modes}
    if negative:
        ok=(suites.get('tests')==1 and suites.get('failures')==1 and not suites.get('errors') and suites.get('skips')==0
            and cases['auto'].get('verdict')=='FAIL' and cases['auto'].get('error_type')=='AssertionError'
            and cases['auto'].get('error')=='ordinary-requests-trigger-native-cycle' and cases['off'].get('verdict')=='PASS')
    else:ok=suites.get('ok') is True and suites.get('skips')==0 and all(v.get('verdict')=='PASS' for v in cases.values())
    record=dict(schema='native41-selfrun/1',verdict='PASS' if rc==0 and ok else 'FAIL',negative_control=negative,
        classification=None if rc==0 and ok else 'HARNESS-OR-ENGINE-UNCLASSIFIED',container_rc=rc,suites=suites,
        cases={m:dict(verdict=r.get('verdict'),checks=len(r.get('checks',[])),error_type=r.get('error_type'),error=r.get('error'),sha256=sha(out/(m+'.json')) if r else None) for m,r in cases.items()},
        bundle=str(bundle),manifest=sha(bundle/'SHA256SUMS'),overlay=str(overlay),overlay_manifest=sha(overlay/'SHA256SUMS'),
        sources=str(sources),image=IMAGE,network='none',gpu_device_requests=0,gpu_execution=False,box_action=False,
        container=name,wall_s=round(time.monotonic()-start,3),command=c,log_shas={p.name:sha(p) for p in out.glob('*.log')})
    (out/'result.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n');print(record['verdict'],sha(out/'result.json'))
    return 0 if record['verdict']=='PASS' else 1

def compare_off(candidate,parent,out):
    a=json.loads((candidate/'off.json').read_text());b=json.loads((parent/'off.json').read_text())
    assert a['verdict']==b['verdict']=='PASS'
    keys=('trace','request_tokens','rpc_methods','cohort_tokens','ticks','geometry')
    raw=lambda r:json.dumps({k:r[k] for k in keys},sort_keys=True,separators=(',',':')).encode()
    assert raw(a)==raw(b),'default-off behavior differs'
    result=dict(verdict='PASS',schema='native41-default-off-comparison/1',fields=keys,canonical_trace_sha256=hashlib.sha256(raw(a)).hexdigest(),candidate_sha256=sha(candidate/'off.json'),parent_sha256=sha(parent/'off.json'))
    with out.open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--name');p.add_argument('--overlay',type=Path);p.add_argument('--sources',type=Path);p.add_argument('--negative',action='store_true');p.add_argument('--compare-candidate',type=Path);p.add_argument('--compare-parent',type=Path);a=p.parse_args()
    if a.compare_candidate:print(compare_off(a.compare_candidate,a.compare_parent,a.out))
    else:raise SystemExit(run(a.bundle,a.out,a.name,a.overlay,a.sources,a.negative))
