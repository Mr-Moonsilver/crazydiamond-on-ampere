#!/usr/bin/env python3
"""Join the small memory workload with real startup/score engagement receipts; not a full G4 grade."""
import argparse,json,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'harness/e3_builder'),str(ROOT/'harness'),str(ROOT/'patches/s5_engine')]
from packet_client_contract import report_exception
from startup_vram_budget import facts


def assess(workload,lines):
    reasons=[];seen={str(r):{} for r in range(4)};score={}
    for line in lines:
        prefix='flash-s5 startup-vram '
        if prefix not in line:continue
        row=json.loads(line.split(prefix,1)[1]);rank=str(row.get('rank'))
        if rank not in seen:continue
        if row.get('schema')=='g4-startup-phase/1':
            label=row['label'];item=seen[rank].setdefault(label,dict(before=0,after=0,error=0,measurements=[],pending=None))
            phase=row['phase'];item[phase]+=1
            if phase in ('before','after'):
                snapshot=row['snapshot'];facts(snapshot)
                if phase=='before':
                    if item['pending'] is not None:reasons.append('startup-phase-overwritten:'+rank+':'+label)
                    item['pending']=snapshot
                elif item['pending'] is None:reasons.append('startup-after-without-before:'+rank+':'+label)
                else:
                    before=item['pending'];a,b=facts(snapshot),facts(before)
                    if (before['device_index']!=snapshot['device_index'] or before['total_bytes']!=snapshot['total_bytes'] or before['observed_ns']>snapshot['observed_ns']):reasons.append('startup-phase-identity:'+rank+':'+label)
                    item['measurements'].append(dict(before=before,after=snapshot,outside_delta_bytes=a['outside_cache_bytes']-b['outside_cache_bytes'],
                        reserved_delta_bytes=snapshot['reserved_bytes']-before['reserved_bytes'],allocated_delta_bytes=snapshot['allocated_bytes']-before['allocated_bytes']))
                    item['pending']=None
        elif row.get('schema')=='g4-score-view/1':
            if rank in score:reasons.append('duplicate-first-score-receipt')
            score[rank]=row
    for rank,phases in seen.items():
        for label in ('native-tp-pp-setup','native-tp-pp-warmup','full-capture','split-capture'):
            row=phases.get(label,{})
            if not row.get('before') or row.get('before')!=row.get('after') or row.get('error'):reasons.append('startup-phase-missing-or-unfinished:'+rank+':'+label)
        custom=[row for label,row in phases.items() if label in ('custom-ar-registration','custom-ar-disabled')]
        if not custom or any(not row['before'] or row['before']!=row['after'] or row['error'] for row in custom):reasons.append('custom-ar-phase-missing:'+rank)
        row=score.get(rank,{})
        if any(row.get(k)!=v for k,v in dict(workspace_lane=1,workspace_bytes=200*2**20,
            score_capacity_bytes=64*2**20,offset_bytes=136*2**20,extra_resident_bytes=0).items()):reasons.append('native-score-not-proven:'+rank)
    if workload.get('verdict')!='FIT':reasons+=workload.get('reasons') or ['workload-not-FIT']
    minimum=workload.get('minimum_mib',{})
    if set(minimum)!=set(seen) or any(type(v) not in (int,float) or not math.isfinite(v) or v<250 for v in minimum.values()):reasons.append('250MiB-target-not-shown')
    if workload.get('raw',{}).get('scope')!='headroom-discriminator-only-not-full-G4':reasons.append('wrong-workload-scope')
    verdict='FAIL' if workload.get('verdict')=='FAIL' else ('HOLD' if reasons else 'PASS')
    return dict(schema='g4-headroom-result/1',verdict=verdict,reasons=sorted(set(reasons)),minimum_mib=minimum,
        startup_phase_counts=seen,first_score=score,scope='headroom discriminator only; full G4 and production qualification remain unrun')


def main():
    p=argparse.ArgumentParser();p.add_argument('--workload',type=Path,required=True);p.add_argument('--engine-log',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('fresh-output-required')
    with a.engine_log.open(errors='replace') as lines:value=assess(json.loads(a.workload.read_text()),lines)
    a.out.write_text(json.dumps(value,indent=2)+'\n');print('G4_HEADROOM '+value['verdict']+' '+','.join(value['reasons']))
    return dict(PASS=0,HOLD=2,FAIL=1)[value['verdict']]
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:report_exception(exc);print('G4_HEADROOM HOLD reader-incomplete');raise SystemExit(2)
