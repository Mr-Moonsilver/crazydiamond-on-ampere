#!/usr/bin/env python3
"""Join numeric MM boundary receipts to the known G5 native request; no user text is read from receipts."""
import argparse,hashlib,json,re,sys
from pathlib import Path
PREFIX='flash-s5 native-mm-boundary '
def read(log,record):
 target=next((x for x in record.get('cases',[]) if x.get('kind')=='native-boundary'),None)
 if target is None:return dict(verdict='HOLD',reason='native-G5-case-missing')
 ids=[target.get('request_id'),target.get('response_id')];ids=[x for x in ids if isinstance(x,str)]
 hashes={hashlib.sha256(x.encode()).hexdigest() for x in ids}
 rows=[]
 for line in log:
  if PREFIX not in line:continue
  try:row=json.loads(line.split(PREFIX,1)[1].strip())
  except ValueError:continue
  if row.get('request_sha256') in hashes:rows.append(row)
 if any(type(r.get('worker_rank'))is not int or type(r.get('free_bytes'))is not int for r in rows):return dict(verdict='HOLD',reason='MM-boundary-rank-or-memory-unreadable')
 ranks={r.get('worker_rank') for r in rows};minima={str(rank):min(r['free_bytes'] for r in rows if r.get('worker_rank')==rank) for rank in ranks if rank in range(4)}
 if ranks!=set(range(4)):return dict(verdict='HOLD',reason='native-G5-four-peer-boundaries-missing',rank_minimum_bytes=minima,matched_rows=len(rows))
 for row in rows:
  mm=row.get('native_mm',{})
  if (row.get('schema')!='native-mm-boundary/1' or type(row.get('free_bytes'))is not int
      or mm.get('tower_registered_cuda_bytes')!=0 or type(mm.get('cache_storage_bytes'))is not int
      or mm['cache_storage_bytes']>40*2**20):return dict(verdict='FAIL',reason='MM-memory-cache-or-tower-proof-failed',rank_minimum_bytes=minima)
 if min(minima.values())<40*2**20:return dict(verdict='FAIL',reason='physical-floor-below40MiB',rank_minimum_bytes=minima)
 if not any(r.get('stage')==0 and r.get('rows',0)>1024 for r in rows):return dict(verdict='HOLD',reason='wider-than-one-image-splice-not-shown',rank_minimum_bytes=minima)
 return dict(verdict='PASS',reason='four-peer-MM-boundaries-cache-floor',rank_minimum_bytes=minima,matched_rows=len(rows),max_payload_rows=max(r['rows'] for r in rows),scope='MM prefix boundary samples; G4 supplies full8x200k/preemption/OOM gate')
def main():
 p=argparse.ArgumentParser();p.add_argument('--engine-log',type=Path,required=True);p.add_argument('--record',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 with a.engine_log.open(errors='replace') as lines:result=read(lines,json.loads(a.record.read_text()))
 a.out.write_text(json.dumps(result,indent=2)+'\n');print('W27_G5_BOUNDARY '+result['verdict']+' '+result['reason']);return 0 if result['verdict']=='PASS' else 2
if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception as exc:
  import traceback
  frame=traceback.extract_tb(exc.__traceback__)[-1];print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {exc} at {frame.filename}:{frame.lineno}',file=sys.stderr);print('W27_G5_BOUNDARY HOLD exception');raise SystemExit(2)
