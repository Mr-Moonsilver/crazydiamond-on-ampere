"""Stream detailed receipts; keep the unchanged legacy localizer on basic labels."""
import argparse
import json
from pathlib import Path
from e3_native30_grade import grade

def collect(paths,filtered):
    coverage={str(r):dict(admission=None,join=None,end=None,limits=0) for r in range(4)}
    with filtered.open('x') as out:
        for path in paths:
            if not path.is_file():continue
            with path.open(errors='replace') as src:
                for line in src:
                    if 'NATIVE30 ' not in line:continue
                    try:row=json.loads(line.split('NATIVE30 ',1)[1])
                    except ValueError:out.write(line);continue
                    if not isinstance(row,dict):out.write(line);continue
                    ev=row.get('event','');rank=str(row.get('rank'))
                    if rank in coverage:
                        key={'host-policy-admission':'admission','host-policy-join':'join','host-policy-end':'end'}.get(ev)
                        if key:coverage[rank][key]=row
                        if ev=='host-receipt-limit':coverage[rank]['limits']+=1
                    if not ev.startswith('host-'):out.write(line)
    return coverage

def evaluate(paths,placement,engine_crashed,filtered):
    coverage=collect(paths,filtered);record=grade(filtered,placement,engine_crashed)
    good=all(x['admission'] and x['join'] and x['end'] and not x['limits']
             and x['end'].get('seconds_after_join',0)>=30 for x in coverage.values())
    missing=[str(p) for p in paths if not p.is_file()]
    if record['verdict']=='PASS' and (not good or missing):
        record.update(verdict='INCONCLUSIVE',failure_class='HARNESS')
    record.update(log_inputs=[str(p) for p in paths],missing_logs=missing,
                  host_coverage=coverage,host_coverage_complete=good,filtered_labels=str(filtered))
    return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--log',required=True,action='append');p.add_argument('--placement',required=True);p.add_argument('--out',required=True);p.add_argument('--engine-crashed',choices=('0','1'),required=True);a=p.parse_args()
    try:placement=json.loads(Path(a.placement).read_text())
    except (OSError,ValueError):placement=None
    record=evaluate([Path(x) for x in a.log],placement,a.engine_crashed=='1',Path(a.out).with_suffix('.basic.log'))
    with Path(a.out).open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
    return {'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[record['verdict']]

if __name__=='__main__':raise SystemExit(main())
