#!/usr/bin/env python3
"""Summarize every planned case, including missing grades and interrupted jobs."""
import json
from pathlib import Path
import re
import sys

CASES=('terminal-c8-1','terminal-c8-2','terminal-c8-3','moe-off','gdn-only','qsa-dense-only','marlin-green-cap')

def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None

def summarize(root):
    root=Path(root);rcs={}
    try:
        for line in (root/'steps.txt').read_text().splitlines():
            m=re.fullmatch(r'(\S+) rc=(\d+) connections=(8|32)',line)
            if m:rcs[m[1]]=int(m[2])
    except OSError:pass
    rows=[]
    for case in CASES:
        folder=root/case/'model-terminal';grade=read(folder/'grade.json')
        valid=isinstance(grade,dict) and grade.get('verdict') in ('PASS','FAIL','INCONCLUSIVE')
        errors=[read(p) for p in folder.glob('result.json.rank*.error.json')]
        errors=[e for e in errors if isinstance(e,dict)]
        ranks=grade.get('data',{}).get('ranks',[]) if valid else []
        both=len(ranks)==2 and {x.get('rank') for x in ranks if isinstance(x,dict)}=={0,1}
        verdict=grade['verdict'] if valid else 'INCONCLUSIVE'
        if any(e.get('fatal_correctness') for e in errors):verdict='FAIL'
        if verdict=='PASS' and (not both or not all(x.get('completed') is True for x in ranks)):
            verdict='INCONCLUSIVE'
        rc=rcs.get(case)
        if rc is not None and rc!=0 and verdict=='PASS':verdict='INCONCLUSIVE'
        rows.append(dict(case=case,verdict=verdict,grade_present=valid,case_rc=rc,
            reason=grade.get('reason') if valid else 'missing or malformed grade; case absent/unfinished',
            watchdog_hang=any(e.get('watchdog') or 'watchdog' in str(e.get('error','')) for e in errors),
            liveness_complete=both and all(x.get('completed') is True for x in ranks),
            bitwise_first_count=sum(x.get('correctness',{}).get('prefill',{}).get('bitwise_first') is True for x in ranks),
            ranks=[dict(rank=x['rank'],correctness=x.get('correctness')) for x in ranks]))
    v='FAIL' if any(r['verdict']=='FAIL' for r in rows) else 'INCONCLUSIVE' if any(r['verdict']!='PASS' for r in rows) else 'PASS'
    trials=rows[:3]
    return dict(verdict=v,scope='diagnostic cases, not S5 value or release',cases=rows,
        full_c8_trials_planned=3,full_c8_trials_graded=sum(r['grade_present'] for r in trials),
        hang_trials=sum(r['watchdog_hang'] for r in trials),
        bitwise_both_ranks_trials=sum(r['bitwise_first_count']==2 for r in trials))

if __name__=='__main__':
    root=Path(sys.argv[1]);r=summarize(root)
    (root/'summary.json').write_text(json.dumps(r)+'\n')
    print('S5 BISECTION SUMMARY '+r['verdict']+'; planned cases='+str(len(r['cases'])))
    raise SystemExit({'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[r['verdict']])
