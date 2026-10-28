#!/usr/bin/env python3
"""CPU-only full scenario/shape self-run in the exact image and overlay."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
IMAGE='sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
NAME='flashnext-native15-matrix1-hub-cpu'


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=('cpu-full','cpu-shape'),default='cpu-shape');a=p.parse_args()
    root=a.input.resolve();out=a.out.absolute();assert root.is_dir() and not out.exists() and not out.is_symlink()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    rows={n:d for d,n in (s.split(maxsplit=1) for s in (root/'SHA256SUMS').read_text().splitlines())}
    assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(rows)|{'SHA256SUMS'}
    assert not any(p.is_symlink() for p in root.rglob('*'))
    assert all(sha(root/n)==d for n,d in rows.items())
    graph=json.loads((root/'proof/import-graph.json').read_text());assert graph['verdict']=='PASS'
    out.mkdir(mode=0o700)
    cmd=['timeout','--signal=TERM','--kill-after=30s','360s','docker','run','--pull=never','--name',NAME,
        '--runtime','runc','--network','none','--cpus','2','--memory','4g','--memory-swap','4g','--pids-limit','256',
        '--read-only','--tmpfs','/tmp:rw,exec,nosuid,nodev,size=1024m','--env','NVIDIA_VISIBLE_DEVICES=none',
        '--env','CUDA_VISIBLE_DEVICES=','--env','PYTHONPATH=','--env','PYTHONDONTWRITEBYTECODE=1',
        '--env','HF_HUB_OFFLINE=1','--env','VLLM_CACHE_ROOT=/tmp/vllm','--env','HF_HOME=/tmp/hf',
        '--env','TRITON_CACHE_DIR=/tmp/triton','--mount','type=bind,src='+str(root/'proof')+',dst=/proof,readonly',
        '--mount','type=bind,src='+str(out)+',dst=/out']
    for mount in graph['mounts']:
        if mount['active']:
            source=root/'overlay'/mount['source'];assert source.exists() and not source.is_symlink()
            cmd+=['--mount','type=bind,src='+str(source)+',dst='+mount['destination']+',readonly']
    cmd+=['--entrypoint','python3',IMAGE,'-B','/proof/e3_native_transitions_matrix_probe.py',
        '--graph','/proof/import-graph.json','--out','/out/probe.json','--mode',a.mode]
    (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');started=time.monotonic()
    with (out/'container.log').open('xb') as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
    result=dict(rc=rc,wall_s=time.monotonic()-started,image=IMAGE,mode=a.mode,container=NAME,
        retained=True,gpu_exposed=False,manifest_sha256=sha(root/'SHA256SUMS'))
    if (out/'probe.json').is_file():
        j=json.loads((out/'probe.json').read_text());result.update(verdict=j['verdict'],failure_class=j['failure_class'],probe_sha256=sha(out/'probe.json'))
    else:result.update(verdict='FAIL',failure_class='HARNESS',missing='probe_result')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    raise SystemExit(rc)


if __name__=='__main__':main()
