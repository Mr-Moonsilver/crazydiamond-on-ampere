#!/usr/bin/env python3
"""Collect owner-run real-caller receipts without replaying a live app twice or copying content."""
import argparse,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from packet_client_contract import report_exception
from vision_w27_read import CALLERS,REQUIRED_CALLERS
FIELDS=('status','application_path','fixture_sha256','wall_s','output_tokens','caller_timeout_contract','timed_out','receipt_path')
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('27B','Flash-Next'),required=True)
 p.add_argument('--owner-record',type=Path,action='append',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
 os.umask(0o077);a.out.parent.mkdir(parents=True,exist_ok=True)
 result=dict(schema='w27-g6/1',phase=a.phase,calls={},sources=[],verdict='HOLD')
 with a.out.open('x') as f:json.dump(result,f)
 for source in a.owner_record:
  value=json.loads(source.read_text())
  if value.get('phase')!=a.phase:raise ValueError('owner capture phase differs')
  for name,call in value.get('calls',{}).items():
   if name not in CALLERS or name in result['calls']:raise ValueError('unknown or duplicate caller receipt:'+name)
   result['calls'][name]={k:call[k] for k in FIELDS if k in call}
  result['sources'].append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
  a.out.write_text(json.dumps(result,indent=2)+'\n')
 missing=sorted(set(REQUIRED_CALLERS)-set(result['calls']));result.update(verdict='COMPLETE' if not missing else 'HOLD',missing_callers=missing)
 a.out.write_text(json.dumps(result,indent=2)+'\n');print('W27_G6_CAPTURE '+result['verdict']+' '+('all-owner-receipts' if not missing else 'missing:'+','.join(missing)))
 return 0 if not missing else 2
if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception as exc:report_exception(exc);print('W27_G6_CAPTURE HOLD exception');raise SystemExit(2)
