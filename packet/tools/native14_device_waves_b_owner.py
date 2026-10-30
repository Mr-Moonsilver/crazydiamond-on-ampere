#!/usr/bin/env python3
"""Owner-executed VM GPU probe. No box/engine control; retains its container.

Hard allocator limit 2 GiB; own-process sampled stop at 2816 MiB under a 3072
MiB allowance. Foreign GPU processes are observed but never signalled. A lock
serializes Flash-Next VM jobs. All output paths/names are fresh; no retries.
"""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

IMAGE='sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
NAME='flashnext-native14-device-waves-b'
CAP_MIB, STOP_MIB, TIMEOUT_S = 3072, 2816, 900


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def command(args): return subprocess.run(args,capture_output=True,text=True,timeout=15)
def write(path, value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2,sort_keys=True);f.write('\n')


def docker_command(root, out, graph):
    cmd=['docker','run','--pull=never','--name',NAME,'--runtime','nvidia',
         '--env','NVIDIA_VISIBLE_DEVICES=0','--env','NVIDIA_DRIVER_CAPABILITIES=compute,utility',
         '--network','none','--cpus','4','--memory','4g','--memory-swap','4g',
         '--pids-limit','256','--read-only','--tmpfs','/tmp:rw,exec,nosuid,nodev,size=1024m',
         '--mount','type=bind,src='+str(root/'proof')+',dst=/proof,readonly',
         '--mount','type=bind,src='+str(out)+',dst=/out',
         '--env','CUDA_VISIBLE_DEVICES=0','--env','PYTHONPATH=',
         '--env','PYTHONDONTWRITEBYTECODE=1','--env','HF_HUB_OFFLINE=1',
         '--env','VLLM_CACHE_ROOT=/tmp/vllm','--env','HF_HOME=/tmp/hf','--env','TRITON_CACHE_DIR=/tmp/triton']
    for mount in graph['mounts']:
        if not mount['active']: continue
        src=root/'overlay'/mount['source']
        if not src.exists() or src.is_symlink(): raise ValueError('missing/linked mount')
        cmd+=['--mount','type=bind,src='+str(src)+',dst='+mount['destination']+',readonly']
    return cmd+['--entrypoint','python3',IMAGE,'-B','/proof/e3_native_sparse_waves_device_probe.py',
                '--graph','/proof/import-graph.json','--out','/out/probe.json']


def parse_apps(text):
    rows=[]
    for fields in csv.reader(io.StringIO(text)):
        if not fields: continue
        if len(fields)!=3: raise ValueError('compute-app schema')
        pid,used,name=(v.strip() for v in fields)
        rows.append(dict(pid=int(pid),used_mib=int(used),name=name))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=args.input.resolve();out=args.out.absolute()
    assert root.is_dir() and not out.exists() and not out.is_symlink()
    rows={n:d for d,n in (x.split(maxsplit=1) for x in (root/'SHA256SUMS').read_text().splitlines())}
    files=[x for x in root.rglob('*') if x.is_file()]
    assert not any(x.is_symlink() for x in root.rglob('*'))
    assert {x.relative_to(root).as_posix() for x in files}==set(rows)|{'SHA256SUMS'}
    assert all(sha(root/n)==d for n,d in rows.items())
    graph=json.loads((root/'proof/import-graph.json').read_text())
    assert graph['verdict']=='PASS' and graph['launcher_sha256']=='83ca0128798fe184b87c006149800994c825979ecedae24a72387b4d7a7085a4'
    # Existing lock is retained. Never unlink another owner's lock or outputs.
    lock=open('/opt/flashnext/.job.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    existing=command(['docker','ps','-a','--format','{{.Names}}'])
    assert existing.returncode==0 and NAME not in existing.stdout.splitlines(), 'container name already used'
    running=command(['docker','ps','--format','{{.Names}}'])
    assert running.returncode==0 and not any(x.startswith('flashnext-') for x in running.stdout.splitlines()), 'VM lane occupied'
    dev=command(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader,nounits'])
    assert dev.returncode==0
    devices=list(csv.reader(io.StringIO(dev.stdout)))
    assert len(devices)==1 and '3090' in devices[0][0]
    assert int(devices[0][1])-int(devices[0][2])>=CAP_MIB, 'less than budget free'
    out.mkdir(mode=0o700);os.chmod(out,0o700)
    write(out/'input.json',dict(manifest_sha256=sha(root/'SHA256SUMS'),image=IMAGE,container=NAME,
        cap_mib=CAP_MIB,stop_mib=STOP_MIB,allocator_bytes=2*1024**3,timeout_s=TIMEOUT_S,device=devices))
    start=time.monotonic();peak=0;samples=0;errors=[];stop_reason=None;seen=set()
    with (out/'container.log').open('xb') as log, (out/'monitor.jsonl').open('x') as monitor:
        proc=subprocess.Popen(docker_command(root,out,graph),stdout=log,stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                elapsed=time.monotonic()-start
                top=command(['docker','top',NAME,'-eo','pid'])
                if top.returncode==0:
                    seen.update(int(x.strip()) for x in top.stdout.splitlines()[1:] if x.strip())
                apps=command(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory,process_name','--format=csv,noheader,nounits'])
                try:
                    if apps.returncode: raise RuntimeError('nvidia-smi rc='+str(apps.returncode))
                    raw=parse_apps(apps.stdout)
                    own=[x for x in raw if x['pid'] in seen]
                    own_mib=sum(x['used_mib'] for x in own)
                    if own: samples+=1
                    peak=max(peak,own_mib)
                    monitor.write(json.dumps(dict(elapsed_s=elapsed,own_mib=own_mib,own=own,
                        foreign=[x for x in raw if x['pid'] not in seen]))+'\n');monitor.flush()
                    if own_mib>=STOP_MIB: stop_reason='own_gpu_stop'
                    if top.returncode and elapsed>15 and proc.poll() is None:
                        running_state=command(['docker','inspect','--format','{{.State.Running}}',NAME])
                        if running_state.returncode or running_state.stdout.strip()=='true':
                            raise RuntimeError('cannot identify own PIDs')
                except Exception as exc:
                    errors.append(str(exc)[:256]);stop_reason='monitor_error'
                if elapsed>=TIMEOUT_S: stop_reason='timeout'
                if stop_reason:
                    # Only our uniquely named container; retained, never rm.
                    stopped=command(['docker','stop','--time','10',NAME])
                    if stopped.returncode: errors.append('own-container-stop-failed')
                    break
                time.sleep(.2)
            try:rc=proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                command(['docker','kill',NAME]);rc=proc.wait(timeout=20)
        finally:
            if proc.poll() is None: command(['docker','stop','--time','10',NAME])
    running_state=command(['docker','inspect','--format','{{.State.Running}}',NAME])
    if running_state.returncode==0 and running_state.stdout.strip()=='true':
        command(['docker','stop','--time','10',NAME]);errors.append('container_outlived_client')
    inspect=command(['docker','inspect','--format','{{json .State}}',NAME])
    state=json.loads(inspect.stdout) if inspect.returncode==0 else {}
    write(out/'container-state.json',state)
    probe=json.loads((out/'probe.json').read_text()) if (out/'probe.json').is_file() else None
    failures=[];missing=[]
    if stop_reason: failures.append(stop_reason)
    if rc not in (0,2): failures.append('process_rc:'+str(rc))
    if not probe: missing.append('probe_result')
    elif probe.get('verdict')!='PASS':
        failures.extend(probe.get('failures') or []);missing.extend(probe.get('missing') or [])
        if probe.get('verdict')=='FAIL' and not failures:failures.append('probe_fail')
    if not samples:missing.append('own_gpu_samples')
    if peak>CAP_MIB: failures.append('own_gpu_cap')
    if errors:missing.append('monitor_errors')
    if state.get('Running') is not False:failures.append('container_not_exited')
    final_apps=command(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory,process_name','--format=csv,noheader,nounits'])
    cleanup=False
    if final_apps.returncode==0:
        cleanup=not any(x['pid'] in seen for x in parse_apps(final_apps.stdout))
    if not cleanup:missing.append('own_gpu_cleanup')
    verdict='FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS'
    record=dict(schema='native14-device-owner/1',verdict=verdict,rc={'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[verdict],
        process_rc=rc,failures=failures,missing=missing,own_peak_mib=peak,own_samples=samples,
        stop_reason=stop_reason,monitor_errors=errors,cleanup_ok=cleanup,container_retained=True,
        cap_mib=CAP_MIB,stop_mib=STOP_MIB,image=IMAGE,manifest_sha256=sha(root/'SHA256SUMS'),
        probe_sha256=sha(out/'probe.json') if probe else None,probe_verdict=probe.get('verdict') if probe else None,
        elapsed_s=time.monotonic()-start,box_action=False,engine_constructed=False)
    write(out/'result.json',record)
    print(json.dumps(record),flush=True)
    return record['rc']

if __name__=='__main__':raise SystemExit(main())
