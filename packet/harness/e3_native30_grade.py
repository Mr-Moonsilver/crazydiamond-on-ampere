"""Classify retained localization receipts without printing GPU fault tracebacks."""
import argparse
import json
from pathlib import Path


def grade(log,placement=None,engine_crashed=False):
    ranks={str(r):dict(begins=0,last_begin=None,last_end=None,exceptions=[],labels=[],unmatched={},last_memory=None) for r in range(4)}
    malformed=0
    paths=[Path(log)] if isinstance(log,(str,Path)) else [Path(p) for p in log]
    records=[];missing=[];seen=set();duplicates=0
    for path in paths:
        if not path.is_file():
            missing.append(str(path));continue
        with path.open(errors='replace') as f:
            for line in f:
                start=line.find('NATIVE30 ')
                if start<0:continue
                try:
                    row=json.loads(line[start+len('NATIVE30 '):])
                    if not isinstance(row,dict):raise ValueError('receipt-not-object')
                except ValueError:malformed+=1;continue
                identity=json.dumps(row,sort_keys=True)
                if identity in seen:duplicates+=1;continue
                seen.add(identity);records.append(row)
    # One host monotonic clock; file concatenation alone can put end before begin.
    records.sort(key=lambda r:r.get('monotonic_ns',0))
    for row in records:
        rank=row.get('rank')
        if type(rank) is not int or str(rank) not in ranks:continue
        out=ranks[str(rank)]
        key=str(row.get('pid'))+':'+str(row.get('call_id'))
        if row.get('event')=='begin':
            out['begins']+=1;out['last_begin']=row;out['unmatched'][key]=row
            if row['label'] not in out['labels']:out['labels'].append(row['label'])
        elif row.get('event')=='end':
            out['last_end']=row;out['unmatched'].pop(key,None)
        elif row.get('event')=='barrier-memory':out['last_memory']=row
        elif row.get('event')=='exception':out['exceptions'].append(row)
        elif row.get('event')=='pending':out['unmatched'].pop(key,None)
    for row in ranks.values():
        row['last_unmatched_begin']=next(reversed(row['unmatched'].values()),None) if row['unmatched'] else None
        row['unmatched']=list(row['unmatched'].values())
    fault=engine_crashed or any(v['exceptions'] for v in ranks.values())
    observed=all('FIRST_ADMISSION' in v['labels'] and 'DECODE_STEP' in v['labels'] for v in ranks.values())
    complete=bool(placement and placement.get('verdict')=='PASS')
    paired=all(not r['unmatched'] for r in ranks.values())
    return dict(schema='native30-localization-grade/1',verdict='FAIL' if fault or malformed else 'PASS' if observed and complete and paired and not missing else 'INCONCLUSIVE',
        failure_class='ENGINE' if fault else 'HARNESS' if malformed or not observed or not complete or not paired or missing else 'NONE',
        per_rank=ranks,log_inputs=[str(p) for p in paths],missing_logs=missing,duplicate_receipts=duplicates,malformed_receipts=malformed,engine_crashed=engine_crashed,
        all_rank_admission_and_decode=observed,all_begins_paired=paired,placement_completed=complete,release_qualified=False,
        scope='localization only; last begin is an attribution boundary, not proof of the originating kernel')


def main():
    p=argparse.ArgumentParser();p.add_argument('--log',required=True,action='append');p.add_argument('--placement',required=True);p.add_argument('--out',required=True);p.add_argument('--engine-crashed',choices=('0','1'),required=True);a=p.parse_args()
    try:placement=json.loads(Path(a.placement).read_text())
    except (OSError,ValueError):placement=None
    record=grade(a.log,placement,a.engine_crashed=='1')
    with Path(a.out).open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
    return {'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[record['verdict']]


if __name__=='__main__':raise SystemExit(main())
