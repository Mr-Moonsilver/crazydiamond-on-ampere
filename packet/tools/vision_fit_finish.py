#!/usr/bin/env python3
"""Final owned-stop/death/identity gate, with fit-D's prospective stop boundary."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'harness'))
from packet_client_contract import report_exception
from fit_death import Scanner

def finish(root,phase_rc,stop_rc):
    reasons=[];verdict='FIT';read={};death={}
    try:read=json.loads((root/'fit-read.json').read_text())
    except (OSError,ValueError):reasons.append('fit-reader-missing')
    try:
        before=json.loads((root/'input-preflight.json').read_text())['host']['memory_events_last']
        after=read['raw']['host']['memory_events_last']
        if any(type(row[k]) is not int or row[k]<0 for row in (before,after) for k in ('oom','oom_kill')) or any(after[k]<before[k] for k in ('oom','oom_kill')):
            raise ValueError('unknown-or-replaced-CT-cgroup-counters')
        if any(after[k]>before[k] for k in ('oom','oom_kill')):
            verdict='FAIL';reasons.append('host-cgroup-OOM-including-startup')
    except (OSError,ValueError,KeyError,TypeError):reasons.append('startup-through-workload-host-events-missing')
    if read.get('verdict')!='FIT':reasons+=read.get('reasons') or ['workload-not-FIT']
    if phase_rc:reasons.append('owner-phase-rc-'+str(phase_rc))
    if stop_rc or 'SERVE_STOP_STATE READY' not in (root/'owned-stop.log').read_text():reasons.append('owned-stop-unproven')
    for phase in ('start','end'):
        if 'VISION_FIT_IDENTITY PASS '+phase not in (root/(phase+'-identity.stdout.log')).read_text():reasons.append(phase+'-identity-unproven')
    paths=sorted(set(root.glob('engine-stream.log'))|set((root/'serve-engine').glob('engine-final-*.log')))
    if paths:
        scan=Scanner([str(p) for p in paths],str(root/'host-stop-intent.json'));scan.poll(final=True);death=scan.decide()
        if death['verdict']=='DEAD':verdict='FAIL';reasons.append('engine-death-before-owned-stop')
        elif death['verdict']!='ALIVE':reasons.append('engine-death-scan-incomplete')
    else:reasons.append('engine-log-missing')
    if read.get('verdict')=='FAIL':verdict='FAIL'
    if verdict!='FAIL' and reasons:verdict='HOLD'
    return dict(schema='vision-fit-owner-result/1',verdict=verdict,reasons=sorted(set(reasons)),phase_rc=phase_rc,
        stop_rc=stop_rc,death=death,fit=read,scope='box fit with images, encoder link on<vm-id> not tested')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--phase-rc',type=int,required=True);p.add_argument('--stop-rc',type=int,required=True);a=p.parse_args()
    value=finish(a.run,a.phase_rc,a.stop_rc);(a.run/'arm-result.json').write_text(json.dumps(value,indent=2)+'\n')
    print('VISION_FIT_OWNER '+value['verdict']+' '+','.join(value['reasons']))
    return dict(FIT=0,HOLD=2,FAIL=1)[value['verdict']]
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:report_exception(exc);print('VISION_FIT_OWNER HOLD final-reader-unavailable');raise SystemExit(2)
