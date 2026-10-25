#!/usr/bin/env python3
"""S6 evidence gates: every rank verified; matched W1/W2 cannot regress."""
import json, math, re, sys

def verify(text):
    rows=re.findall(r'(Worker_PP[01]_TP[01]).*S6_VERIFY rows=(\d+) equal=(\d+)',text)
    return bool(rows) and set(r[0] for r in rows)=={f'Worker_PP{p}_TP{t}' for p in range(2) for t in range(2)} and all(int(n)>0 and e=='1' for _,n,e in rows)
def performance(a,b):
    # Raw harness report is stored as summary; retain and recompute raw separately.
    a=a.get('summary',a.get('report',a));b=b.get('summary',b.get('report',b))
    assert a['verdict']==b['verdict']=='PASS' and not a['fails'] and not b['fails']
    checks={}
    for window in ('before','during'):
        checks[window+'_tps']=b[window]['agg_tps']>=a[window]['agg_tps']*.97
        checks[window+'_p99']=b[window]['worst_stream_gap_ms']['p99']<=a[window]['worst_stream_gap_ms']['p99']*1.1
        for av,bv in zip(a[window]['per_stream'],b[window]['per_stream'],strict=True):
            assert av['tps']>0 and bv['tps']>0 and math.isfinite(bv['tps'])
    checks['fresh_ttft']=b['newcomer']['client_ttft_s']<=a['newcomer']['client_ttft_s']*1.1
    checks['fresh_uncached']=a['newcomer']['cached_tokens']==b['newcomer']['cached_tokens']==0
    return checks
if __name__=='__main__':
    try:
        if sys.argv[1]=='verify':ok=verify(open(sys.argv[2]).read());d={'all_four_ranks_byte_equal':ok}
        else:d=performance(*[json.load(open(p)) for p in sys.argv[2:4]]);ok=all(d.values())
        print(json.dumps({'verdict':'PASS' if ok else 'FAIL','checks':d}));sys.exit(0 if ok else 1)
    except Exception:print('{"verdict":"INCONCLUSIVE","reason":"missing or invalid evidence"}');sys.exit(2)
