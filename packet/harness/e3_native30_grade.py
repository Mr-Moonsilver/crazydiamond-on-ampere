"""Classify retained localization receipts without printing GPU fault tracebacks."""
import argparse
import json
from pathlib import Path


def grade(log,placement=None,engine_crashed=False):
    ranks={str(r):dict(begins=0,last_begin=None,last_end=None,exceptions=[],labels=[]) for r in range(4)}
    malformed=0
    with Path(log).open(errors='replace') as f:
        for line in f:
            start=line.find('NATIVE30 ')
            if start<0:continue
            try:row=json.loads(line[start+len('NATIVE30 '):])
            except ValueError:malformed+=1;continue
            rank=row.get('rank')
            if type(rank) is not int or str(rank) not in ranks:continue
            out=ranks[str(rank)]
            if row.get('event')=='begin':
                out['begins']+=1;out['last_begin']=row
                if row['label'] not in out['labels']:out['labels'].append(row['label'])
            elif row.get('event')=='end':out['last_end']=row
            elif row.get('event')=='exception':out['exceptions'].append(row)
    fault=engine_crashed or any(v['exceptions'] for v in ranks.values())
    observed=all('FIRST_ADMISSION' in v['labels'] and 'DECODE_STEP' in v['labels'] for v in ranks.values())
    complete=bool(placement and placement.get('verdict')=='PASS')
    return dict(schema='native30-localization-grade/1',verdict='FAIL' if fault or malformed else 'PASS' if observed and complete else 'INCONCLUSIVE',
        failure_class='ENGINE' if fault else 'HARNESS' if malformed or not observed or not complete else 'NONE',
        per_rank=ranks,malformed_receipts=malformed,engine_crashed=engine_crashed,
        all_rank_admission_and_decode=observed,placement_completed=complete,release_qualified=False,
        scope='localization only; last begin is an attribution boundary, not proof of the originating kernel')


def main():
    p=argparse.ArgumentParser();p.add_argument('--log',required=True);p.add_argument('--placement',required=True);p.add_argument('--out',required=True);p.add_argument('--engine-crashed',choices=('0','1'),required=True);a=p.parse_args()
    try:placement=json.loads(Path(a.placement).read_text())
    except (OSError,ValueError):placement=None
    log=Path(a.log)
    if log.is_file():record=grade(log,placement,a.engine_crashed=='1')
    else:record=dict(verdict='FAIL' if a.engine_crashed=='1' else 'INCONCLUSIVE',failure_class='ENGINE' if a.engine_crashed=='1' else 'HARNESS',reason='retained-stream-unavailable',release_qualified=False)
    with Path(a.out).open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
    return {'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[record['verdict']]


if __name__=='__main__':raise SystemExit(main())
