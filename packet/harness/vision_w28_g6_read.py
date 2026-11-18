#!/usr/bin/env python3
"""Approval-bound w28 G6: only three named callers may lack a 27B capture."""
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from packet_client_contract import report_exception
from vision_w27_read import REQUIRED_CALLERS,CALLERS,G6_CONTRACT_SHA
UNPAIRED={'<caller-a>.text','<caller-a>.image','<caller-d>'}

def read(pre,post):
 def result(v,r,**kw):return dict(verdict=v,reason=r,approval='APPROVAL-w28-vision-production-20261118:3b2f66d8',contract_sha256=G6_CONTRACT_SHA,calls=rows,**kw)
 rows=[]
 if post.get('phase')!='Flash-Next' or (pre is not None and pre.get('phase')!='27B'):return result('HOLD','caller-phase-differs')
 for name in REQUIRED_CALLERS:
  a=(pre or {}).get('calls',{}).get(name);b=post.get('calls',{}).get(name)
  if b is None or (a is None and name not in UNPAIRED):return result('HOLD','missing-real-caller:'+name)
  if a is None and b.get('timed_out') is not False:return result('HOLD','caller-timeout-not-proven:'+name)
  for c in ([a,b] if a is not None else [b]):
   if c.get('status')=='FAIL' or c.get('timed_out') is True:return result('FAIL','real-caller-failed:'+name)
   if (c.get('status')!='PASS' or c.get('application_path') is not True or type(c.get('output_tokens')) is not int or c['output_tokens']<0
       or not isinstance(c.get('wall_s'),(int,float)) or not math.isfinite(c['wall_s']) or c['wall_s']<0
       or not c.get('fixture_sha256') or not c.get('receipt_path') or not c.get('caller_timeout_contract')):return result('HOLD','unproven-real-caller:'+name)
  if a is not None and a['fixture_sha256']!=b['fixture_sha256']:return result('HOLD','caller-fixture-differs:'+name)
  rows.append(dict(caller=name,reference='27B not captured' if a is None else '27B captured',reference_wall_s=None if a is None else a['wall_s'],reference_output_tokens=None if a is None else a['output_tokens'],reference_receipt=None if a is None else a['receipt_path'],production_wall_s=b['wall_s'],production_output_tokens=b['output_tokens'],production_receipt=b['receipt_path']))
 return result('PASS','six-real-callers-w28-approved-baseline-scope',<caller-c>_text_proof='real summaries; no separate text call',<caller-c>_image='N/A BY DESIGN',supplementary={n:dict(reference=(pre or {}).get('calls',{}).get(n),production=post.get('calls',{}).get(n),scope='REPORT ONLY caller-config probe') for n in CALLERS if n not in REQUIRED_CALLERS})

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--record',type=Path,required=True);p.add_argument('--reference',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
 value=read(json.loads(a.reference.read_text()) if a.reference else None,json.loads(a.record.read_text()));a.out.write_text(json.dumps(value,indent=2)+'\n');print('W28_G6 '+value['verdict']+' '+value['reason']);return dict(PASS=0,HOLD=2,FAIL=1)[value['verdict']]
if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception as exc:report_exception(exc);print('W28_G6 HOLD incomplete-or-exception');raise SystemExit(2)
