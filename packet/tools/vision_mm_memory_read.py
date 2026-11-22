#!/usr/bin/env python3
"""All-rank MM construction budget; registered tensors only, physical FIT is separate."""
import argparse,json,sys,traceback
from pathlib import Path


def read(value):
    reasons=[];peers={}
    workers=value.get('workers')
    if value.get('verdict')!='PASS' or not isinstance(workers,dict) or set(workers)!={'0','1','2','3'}:
        return dict(verdict='HOLD',reasons=['startup-or-peer-inventory-incomplete'],peers={})
    fields=('tower_registered_cuda_bytes','cache_budget_tokens','cache_budget_bytes','cache_feature_bytes',
        'cache_storage_bytes','encoder_inputs_buffer_bytes','active_splice_device_bytes')
    for rank in range(4):
        inv=(workers[str(rank)].get('native_inventory') or {})
        mm=inv.get('native_mm')
        if (inv.get('worker_rank')!=rank or not isinstance(mm,dict) or mm.get('schema')!='native-mm-inventory/1'
                or any(type(mm.get(k)) is not int or mm[k]<0 for k in fields)):
            reasons.append('rank-'+str(rank)+'-missing-or-unknown-MM-bytes');continue
        peers[str(rank)]=mm
        if (mm['tower_registered_cuda_bytes']!=0 or mm['cache_budget_tokens']!=8192
                or mm['cache_budget_bytes']!=40*2**20 or mm['cache_storage_bytes']>40*2**20
                or mm['cache_feature_bytes']!=mm['cache_storage_bytes']
                or mm.get('gpu_encoder_cache_present') is not (rank<2)
                or mm['encoder_inputs_buffer_bytes']!=(5*2**20 if rank<2 else 0)
                or mm['active_splice_device_bytes']>2*(5*2**20+8192)):
            reasons.append('rank-'+str(rank)+'-MM-budget-differs')
    return dict(verdict='HOLD' if reasons else 'PASS',reasons=reasons,peers=peers,
        scope='registered tower/MM tensor construction only; physical allocator/driver delta and8x200k peak FIT remain separate gates')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--startup-grade',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=read(json.loads(a.startup_grade.read_text()))
    with a.out.open('x') as output:json.dump(result,output,indent=2);output.write('\n')
    print('VISION_MM_MEMORY '+result['verdict']+' '+(','.join(result['reasons']) or 'four-rank-construction-budget'))
    return 0 if result['verdict']=='PASS' else 2

if __name__=='__main__':
    try:code=main()
    except (Exception,SystemExit) as exc:
        if not isinstance(exc,SystemExit) or exc.code:
            frame=traceback.extract_tb(exc.__traceback__)[-1]
            print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {exc} at {frame.filename}:{frame.lineno}',file=sys.stderr)
        if isinstance(exc,SystemExit):raise
        print('VISION_MM_MEMORY HOLD '+type(exc).__name__);raise SystemExit(2)
    raise SystemExit(code)
