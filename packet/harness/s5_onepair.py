#!/usr/bin/env python3
"""Offline plan/grading and GPU entry point for bounded S5 one-pair diagnoses."""
import argparse
import json
import math
from pathlib import Path

CASES = ('decode-alone', 'prefill-alone', 'node-audit', 'collectives', 'prio-overlap')

def audit_grade(rows):
    kernels=[r for r in rows if r.get('type')=='kernel']
    if not kernels or any(r.get('context_known') is not True for r in kernels):
        return dict(verdict='INCONCLUSIVE', reason='kernel context unavailable; do not infer it from replay stream', kernels=len(kernels))
    mismatch=[r['node'] for r in kernels if r.get('context_matches') is not True]
    return dict(verdict='FAIL' if mismatch else 'PASS', scope='kernel-node context only, not event ordering or forward progress',
                kernels=len(kernels), mismatched_nodes=mismatch)

def grade(path):
    path=Path(path); errors=[]; ranks=[]
    for i in range(2):
        ep=Path(str(path)+f'.rank{i}.error.json')
        if ep.exists(): errors.append(json.loads(ep.read_text()))
    if any(e.get('fatal_correctness') for e in errors):
        return dict(verdict='FAIL',reason='explicit correctness failure',errors=errors)
    try:
        d=json.loads(path.read_text());ranks=d['ranks']
        assert len(ranks)==2 and {r['rank'] for r in ranks}=={0,1}
        assert d['pair']=='1,3' and d['case'] in CASES
        assert len({r['pci_bus_id'] for r in ranks})==2
        for r in ranks:
            assert r['completed'] is True and r['case']==d['case']
            assert r['source_sha256'] and r['source_sha256']==ranks[0]['source_sha256']
            assert (r['context'],r['max_connections'])==(ranks[0]['context'],ranks[0]['max_connections'])
            assert json.loads(Path(str(path)+f'.rank{r["rank"]}.progress.json').read_text())['phase']=='completed'
        assert not errors
    except (OSError,ValueError,KeyError,TypeError,AssertionError):
        return dict(verdict='INCONCLUSIVE', reason='incomplete rank/cleanup evidence', errors=errors)
    verdict='PASS'
    if d['case']=='node-audit':
        verdict='INCONCLUSIVE' if any(r['audit']['verdict']!='PASS' for r in ranks) else 'PASS'
    else:
        expected={'decode-alone':{'decode'},'prefill-alone':{'prefill'},'collectives':{'decode','prefill'},'prio-overlap':{'decode'}}[d['case']]
        for r in ranks:
            if set(r['durations_ms'])!=expected:verdict='INCONCLUSIVE'
            for lane,values in r['durations_ms'].items():
                if len(values)!=30 or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values):
                    verdict='INCONCLUSIVE'
            if not r['durations_ms']:verdict='INCONCLUSIVE'
    return dict(verdict=verdict, scope='one-pair diagnostic only; never S5 GO',case=d['case'],ranks=ranks)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--grade',type=Path);ap.add_argument('--case',choices=CASES)
    ap.add_argument('--out',type=Path);ap.add_argument('--config',type=Path)
    ap.add_argument('--overlay-sha',required=False)
    ap.add_argument('--decode-backend',choices=['custom','pynccl'],default='custom')
    ap.add_argument('--context',choices=['green24','prio'],default='green24')
    ap.add_argument('--phase-timeout',type=float,default=120)
    a=ap.parse_args()
    if a.grade:
        r=grade(a.grade);print(json.dumps(r,allow_nan=False));return {'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[r['verdict']]
    if not a.case or not a.out or not a.config:ap.error('case/out/config required for GPU run')
    from s5_onepair_gpu import run
    return run(a)
if __name__=='__main__':raise SystemExit(main())
