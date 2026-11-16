#!/usr/bin/env python3
"""Run the unchanged full source/capture identity gate with its own phase budget."""
import argparse,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'harness'))
from packet_client_contract import report_exception


def run(packet,out,phase):
    h=packet/'harness';root=os.environ.get('R','');driver=h/'engine_identity.py'
    def child(name,argv):
        with (out/(name+'.stdout.log')).open('x') as stdout,(out/(name+'.stderr.log')).open('x') as stderr:
            result=subprocess.run([sys.executable,'-B',str(driver),*argv],stdout=stdout,stderr=stderr,check=False)
        if result.returncode:raise RuntimeError('full-identity-'+name+'-rc='+str(result.returncode))
    request=out/'identity-request.json';start=out/'identity-start.json';end=out/'identity-end.json'
    if phase=='start':
        args=['declare',str(request),'--defaults',root+'/etc/default/vllm-flash',
            '--launcher',root+'/data/vllm/flash-next-tp4/prod/run-flash-prod.sh','--setcfg',root+'/tmp/setcfg.py',
            '--image-file',str(packet/'image-id.txt'),'--mount-policy',str(h/'e3_startup_mount_policy.json'),'--timeout','900']
        if root:args+=['--source-mirror',root+'/data/vllm/flash-next-tp4']
        child('declare',args)
        child('capture-start',['capture',str(start),'--request',str(request),'--defaults',root+'/etc/default/vllm-flash','--timeout','900'])
        child('check-start',['check',str(start),str(request)])
    else:
        child('capture-end',['capture',str(end),'--request',str(request),'--defaults',root+'/etc/default/vllm-flash','--timeout','1200'])
        child('check-end',['check',str(start),str(request),str(end)])
    print('VISION_FIT_IDENTITY PASS '+phase)

if __name__=='__main__':
    try:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,required=True)
        p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=('start','end'),required=True)
        a=p.parse_args();run(a.packet,a.out,a.phase)
    except Exception as exc:report_exception(exc);print('VISION_FIT_IDENTITY HOLD '+type(exc).__name__);raise SystemExit(2)
