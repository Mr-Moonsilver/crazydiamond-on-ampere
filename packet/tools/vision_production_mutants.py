#!/usr/bin/env python3
"""Retained source-revert negatives, actual client/parser protocol per mutated copy; CPU only."""
import argparse,json,os,re,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CASES=(
 ('gateway-auth','vision_public_gateway.py',"if (post or self.path not in ('/health','/metrics','/v1/models')) and not self.authenticated():", "if False and not self.authenticated():",'test_vision_public_gateway.Protocol.test_full_text_image_tool_messages_sse_and_keyless_metadata'),
 ('g5-all-four','vision_g5_probe.py','for case in cases:','for case in cases[:-1]:','test_vision_w27_protocol.WholeProtocol.test_g5_actual_argv_all_seven_cases_final_reader_and_reference_mutant'),
 ('a3-all100','vision_prod_a3.py','todo=[items[i] for i in selected]','todo=[items[i] for i in selected[:-1]]','test_vision_w27_protocol.WholeProtocol.test_a3_frozen_runner_actual100_requests_and_scorer_leaf_to_final_verdict'),
 ('g6-duplicate-owner','vision_g6_capture.py',"if name not in CALLERS or name in result['calls']:","if name not in CALLERS:",'test_vision_w27_protocol.WholeProtocol.test_g6_complete_actual_reader_argv_missing_caller_and_probe_only_negative'),
 ('reader-real-caller','vision_w27_read.py'," or c.get('application_path') is not True",'','test_vision_w27_protocol.WholeProtocol.test_g6_complete_actual_reader_argv_missing_caller_and_probe_only_negative'),
 ('service-clearance','vision_service_owner.py',"if clear.get('service_cleared') is not True:",'if False:','test_vision_w27_protocol.WholeProtocol.test_service_owner_exact_argv_owned_stop_and_admission_refusals'),
 ('restart-unreleased','vision_production_ready.py',"and auto.get('manual_owned') is False",'', 'test_vision_production_ready.Readiness.test_missing_or_zero_selection_never_becomes_ready'),
 ('encoder-memory','vision_encoder_job.py',"sample = check_memory(**physical(clear['gpu_uuid'], pid))","sample = physical(clear['gpu_uuid'], pid)",'test_vision_encoder_protocol.Protocol.test_memory_breach_stops_only_owned_child'),
)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 records=[]
 for name,module,before,after,test in CASES:
  folder=a.out/name;h=folder/'harness';h.mkdir(parents=True)
  for f in (ROOT/'harness').glob('vision_*.py'):shutil.copy2(f,h/f.name)
  for n in ('packet_client_contract.py','test_vision_public_gateway.py','test_vision_w27_protocol.py','test_vision_encoder_protocol.py','test_vision_production_ready.py'):shutil.copy2(ROOT/'harness'/n,h/n)
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(h)+':'+str(ROOT/'tools'))
  argv=[sys.executable,'-B','-m','unittest',test]
  baseline=subprocess.run(argv,env=env,cwd=folder,capture_output=True,text=True,timeout=120)
  (folder/'baseline.log').write_text(baseline.stdout+baseline.stderr)
  if baseline.returncode:raise ValueError('mutant baseline not green:'+name)
  path=h/module;text=path.read_text()
  if text.count(before)!=(2 if name=='encoder-memory' else 1):raise ValueError('mutation anchor not unique:'+name)
  path.write_text(text.replace(before,after))
  mutant=subprocess.run(argv,env=env,cwd=folder,capture_output=True,text=True,timeout=120)
  output=mutant.stdout+mutant.stderr;(folder/'mutant.log').write_text(output)
  caught=mutant.returncode!=0 and 'AssertionError' in output and bool(re.search(r'Ran [1-9][0-9]* test',output))
  records.append(dict(name=name,argv=argv,baseline_rc=baseline.returncode,mutant_rc=mutant.returncode,caught=caught))
  print('VISION_PRODUCTION_MUTANT '+name+' '+('CAUGHT' if caught else 'MISSED'),flush=True)
  if not caught:break
 (a.out/'mutants.json').write_text(json.dumps(records,indent=2)+'\n');okay=len(records)==len(CASES) and all(r['caught'] for r in records)
 print('VISION_PRODUCTION_MUTANTS '+('PASS' if okay else 'FAIL')+' '+str(sum(r['caught'] for r in records))+'/'+str(len(CASES)));return 0 if okay else 1
if __name__=='__main__':raise SystemExit(main())
