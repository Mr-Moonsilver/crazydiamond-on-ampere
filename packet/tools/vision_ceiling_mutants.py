#!/usr/bin/env python3
"""CPU-only reverts of the serial sweep's gates, each must fail its focused protocol test."""
import argparse,os,shutil,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
changes=[
 ('speech-pid-disappeared','vision_encoder_ceiling.py',"speech=set(clear['speech_gpu_pids'])","speech=set(apps)-{pid}",'Ceiling.test_card_budget_pid_restart_and_unknown_gpu_owner'),
 ('card-budget','vision_encoder_ceiling.py','if used>22528 or outside>500:','if False:','Ceiling.test_card_budget_pid_restart_and_unknown_gpu_owner'),
 ('ASR-match','vision_encoder_ceiling.py',"if not speech.get('correct') or speech['elapsed_s']>bound:",'if False:','Ceiling.test_first_asr_failure_stops_higher_steps'),
 ('missing-N8','vision_encoder_ceiling.py','for n in (1,2,4,6,8):','for n in (1,2,4,6):','Ceiling.test_whole_serial_gateway_sweep_and_post_stop_asr'),
 ('physical-readout','vision_encoder_ceiling_read.py',"and all(r.get('measured_samples',0)>0 and 0<r.get('owned_physical_peak_mib',0)<=3072 for r in rows)",'and True','WholeOwner.test_actual_owner_argv_all21_encodes_7_asrs_gpu_off_and_final_reader'),
]
for name,module,old,new,test in changes:
 out=a.out/name;shutil.copytree(a.packet,out);path=out/'harness'/module;text=path.read_text()
 if text.count(old)!=1:raise ValueError('mutation anchor differs '+name)
 text=text.replace(old,new)
 path.write_text(text);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',VISION_CEILING_PACKET=str(out),PYTHONPATH=str(out/'harness')+':'+str(out/'tools'))
 r=subprocess.run([sys.executable,'-B','-m','unittest','test_vision_encoder_ceiling.'+test],cwd=out,env=env,capture_output=True,text=True,timeout=40)
 (a.out/(name+'.log')).write_text(r.stdout+r.stderr)
 if r.returncode==0 or 'FAIL' not in r.stderr:raise AssertionError('mutant not caught by assertion: '+name+' '+r.stderr)
 print('MUTANT_CAUGHT '+name,flush=True)
print('VISION_CEILING_MUTANTS PASS '+str(len(changes)))
