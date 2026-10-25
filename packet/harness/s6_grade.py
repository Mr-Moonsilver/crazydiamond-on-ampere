#!/usr/bin/env python3
"""S6 ABA gates. A gain must beat both controls; deltas include every stream."""
import json, math, re, sys

def verify(text):
    rows=re.findall(r'(Worker_PP[01]_TP[01]).*S6_VERIFY rows=(\d+) equal=(\d+)',text)
    return bool(rows) and set(r[0] for r in rows)=={f'Worker_PP{p}_TP{t}' for p in range(2) for t in range(2)} and all(int(n)>0 and e=='1' for _,n,e in rows)

def performance(a,b,c):
    for d in (a,b,c):
        assert d['verdict']=='PASS' and not d['fails']
        assert d['streams']==8 and d['stream_prompt']==153537 and d['long_prompt']==200000
        assert d['newcomer']['cached_tokens']==0
        for w in ('before','during'):
            assert len(d[w]['per_stream'])==8
            for key in ('agg_tps',):assert math.isfinite(d[w][key]) and d[w][key]>0
            assert d[w]['worst_stream_gap_ms']['p99']>0
            for s in d[w]['per_stream']:assert math.isfinite(s['tps']) and s['tps']>0
        assert math.isfinite(d['newcomer']['client_ttft_s']) and d['newcomer']['client_ttft_s']>0
    def delta(x,y,z):
        mid=(x+z)/2
        return {'baseline_first':x,'candidate':y,'baseline_last':z,'candidate_pct_vs_mean':100*(y/mid-1),
                'control_spread_pct':100*abs(x-z)/mid}
    checks={};measurements={}
    for w in ('before','during'):
        xs=[d[w]['agg_tps'] for d in (a,b,c)]
        measurements[w]={'aggregate_tps':delta(*xs),
           'per_stream_tps':[delta(*(d[w]['per_stream'][i]['tps'] for d in (a,b,c))) for i in range(8)],
           'worst_p99_ms':delta(*(d[w]['worst_stream_gap_ms']['p99'] for d in (a,b,c)))}
        checks[w+'_aggregate_nonregression']=xs[1]>=max(xs[0],xs[2])*.97
        checks[w+'_each_stream_nonregression']=all(b[w]['per_stream'][i]['tps']>=max(a[w]['per_stream'][i]['tps'],c[w]['per_stream'][i]['tps'])*.97 for i in range(8))
        checks[w+'_p99_nonregression']=b[w]['worst_stream_gap_ms']['p99']<=min(a[w]['worst_stream_gap_ms']['p99'],c[w]['worst_stream_gap_ms']['p99'])*1.1
    ttft=[d['newcomer']['client_ttft_s'] for d in (a,b,c)];measurements['fresh_ttft_s']=delta(*ttft)
    checks['fresh_ttft_nonregression']=ttft[1]<=min(ttft[0],ttft[2])*1.1
    gains={'w2_above_both_controls_2pct':b['during']['agg_tps']>max(a['during']['agg_tps'],c['during']['agg_tps'])*1.02,
           'ttft_below_both_controls_2pct':ttft[1]<min(ttft[0],ttft[2])*.98}
    checks['measured_gain']=any(gains.values())
    return {'verdict':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'gains':gains,'measurements':measurements,
            'scope':'one ABA sequence; observed gain beyond both controls, not a confidence interval'}
if __name__=='__main__':
    try:
        if sys.argv[1]=='verify':
            ok=verify(open(sys.argv[2]).read());d={'verdict':'PASS' if ok else 'FAIL','all_four_ranks_byte_equal':ok}
        else:d=performance(*[json.load(open(p)) for p in sys.argv[2:5]]);ok=d['verdict']=='PASS'
        print(json.dumps(d));sys.exit(0 if ok else 1)
    except Exception:print('{"verdict":"INCONCLUSIVE","reason":"missing or invalid evidence"}');sys.exit(2)
