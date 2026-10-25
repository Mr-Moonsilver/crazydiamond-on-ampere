#!/usr/bin/env python3
"""Rescore replay and fresh/W2 evidence; report gains and model limitations."""
import json,math,sys
from pathlib import Path
from agent_replay import summarize

def grade(q):
    checks={};metrics={}
    for n in (8,10):
        a=json.loads((q/f'base-{n}-replay.json').read_text());b=json.loads((q/f'cost-{n}-replay.json').read_text())
        assert a['config']==b['config'] and a['config']['AGENTS']==n
        sa,sb=[summarize(d['turns'],d['config']['WARMUP'],d['config']['DURATION']) for d in (a,b)]
        for s in (sa,sb):
            assert not s['errors'] and not s['incomplete'] and s['warm_turns']>=20 and s['appends_judged']>=20
            assert all(math.isfinite(s[k]) and s[k]>0 for k in ('turns_per_hour','append_ttft_p95'))
        checks[f'{n}_turns']=sb['turns_per_hour']>=sa['turns_per_hour']*.97
        checks[f'{n}_append_p95']=sb['append_ttft_p95']<=sa['append_ttft_p95']*1.1
        checks[f'{n}_prefix_reuse']=sb['append_full_reuse']>=sa['append_full_reuse']-.02
        metrics[str(n)]={'baseline':sa,'candidate':sb,'turns_gain_pct':100*(sb['turns_per_hour']/sa['turns_per_hour']-1)}
    a,b=[json.loads((q/f'{c}-8-w2.json').read_text()) for c in ('base','cost')]
    assert a['verdict']==b['verdict']=='PASS' and not a['fails'] and not b['fails']
    for w in ('before','during'):
        checks[w+'_tps']=b[w]['agg_tps']>=a[w]['agg_tps']*.97
        checks[w+'_p99']=b[w]['worst_stream_gap_ms']['p99']<=a[w]['worst_stream_gap_ms']['p99']*1.1
    checks['w2_fresh_ttft']=b['newcomer']['client_ttft_s']<=a['newcomer']['client_ttft_s']*1.1
    x,y=[json.loads((q/f'{c}-fresh.json').read_text()) for c in ('base','cost')]
    assert all(d['cached_tokens']==0 and d['completion_tokens']==1 and d['prompt_tokens']==200000 for d in (x,y))
    checks['fresh_200k']=y['seconds']<=x['seconds']*1.1
    metrics['w2']={w:{'baseline':a[w],'candidate':b[w]} for w in ('before','during')}
    metrics['fresh']={'baseline':x,'candidate':y}
    checks['model_exercised']='FLASH_COST step=' in (q/'cost-8.engine.log').read_text()
    return {'verdict':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'metrics':metrics,
            'scope':'nonregression qualification only; a PASS is not evidence of improvement or a calibrated deadline'}
if __name__=='__main__':
    try:d=grade(Path(sys.argv[1]));print(json.dumps(d));sys.exit(0 if d['verdict']=='PASS' else 1)
    except Exception:print('{"verdict":"INCONCLUSIVE","reason":"invalid or missing evidence"}');sys.exit(2)
