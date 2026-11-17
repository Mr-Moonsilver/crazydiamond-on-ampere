#!/usr/bin/env python3
"""Manual, infra-cleared encoder service admission; no automatic GPU reservation or restart."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from packet_client_contract import report_exception
from vision_encoder_job import physical
from vision_speech_guard import Refused,check_health,validate_clearance


def admission(state,now):
    attempts=[t for t in state.get('attempts',[]) if type(t) in (int,float) and 0<=now-t<3600]
    if len(attempts)>=3:raise Refused('readmission-three-starts-per-hour; remain OFF and alert infra')
    last=state.get('yielded_at')
    if last is not None and (type(last) not in (int,float) or now-last<600):
        raise Refused('readmission-ten-minute-yield-cooldown')
    return attempts


def save(path,state):
    pending=path.with_suffix('.next.json')
    with os.fdopen(os.open(pending,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600),'w') as f:json.dump(state,f)
    pending.replace(path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet',type=Path,required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--clearance',type=Path,required=True)
    p.add_argument('--state',type=Path,default=Path('/opt/flashnext/service-state/admissions.json'))
    p.add_argument('--key-file',type=Path,default=Path('/opt/flashnext/keys/vision-encoder.env'))
    args=p.parse_args(argv)
    clear=validate_clearance(args.clearance,args.run_id)
    if clear.get('service_cleared') is not True:raise Refused('fresh-service-clearance-required')
    subprocess.run(['sha256sum','--strict','--quiet','-c','PACKET-SHA256SUMS'],cwd=args.packet,check=True)
    args.state.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    with (args.state.parent/'admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(args.state.read_text()) if args.state.exists() else {}
        attempts=admission(state,time.time())
        # Re-admissions (not the initial cleared start) need five continuous minutes free.
        if state.get('yielded_at') is not None:
            until=time.monotonic()+300
            while time.monotonic()<until:
                check_health()
                if physical(clear['gpu_uuid'])['free_mib']<4096:raise Refused('readmission-free-headroom-not-stable-five-minutes')
                time.sleep(5)
        state.update(attempts=attempts+[time.time()],run_id=args.run_id)
        save(args.state,state)
        run=Path('/opt/flashnext/runs')/('vision-'+args.run_id)
        command=[sys.executable,'-B',str(args.packet/'harness/vision_encoder_job.py'),
            '--model','/opt/flashnext/models/flash-next-vision-8b333e23','--run-dir',str(run),'--run-id',args.run_id,
            '--clearance',str(args.clearance),'--key-file',str(args.key_file),'--expected-asr',str(args.packet/'asr-expected.txt'),
            '--asr-url','http://<encoder-host>:9030/asr?language=en&output_format=json','--asr-max-seconds','10',
            '--speech-health','http://<encoder-host>:9030/health','--fixture','<hub-dir>/scratch/infra/asr-fixture/asr10.wav',
            '--images',*[str(args.packet/'harness/vision_assets'/n) for n in ('red.png','table.png','diagram.png','screenshot.png','red1024.png')],
            '--service']
        child=subprocess.Popen(command)
        def stop(*_):
            if child.poll() is None:child.terminate()
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:code=child.wait()
        finally:
            if child.poll() is None:
                child.terminate();child.wait(timeout=45)
            state.update(yielded_at=time.time(),last_exit=child.returncode,alert='OFF; manual fresh clearance required')
            save(args.state,state)
        print('VISION_SERVICE_OFF '+json.dumps(dict(run_id=args.run_id,child_rc=code,state_path=str(args.state))),flush=True)
        return code

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:report_exception(exc);print('VISION_SERVICE HOLD admission-or-owned-stop');raise SystemExit(2)
