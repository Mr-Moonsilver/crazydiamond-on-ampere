#!/usr/bin/env python3
"""S5 frontier/controller CPU rules; GPU work is an explicit separate entry."""
import argparse
import json
import math
from pathlib import Path


def score_profile(data):
    ranks=data['ranks'];assert len(ranks)==2 and {r['rank'] for r in ranks}=={0,1}
    summaries=[]
    for r in ranks:
        assert r['completed'] is True and r['audit']['verdict']=='PASS'
        m=r['modes'];d=m['decode'];p=m['prefill'];o=m['overlap'];solo=m['decode_split']
        values=[d['decode_steps_s'],p['prefill_steps_s_upper'],o['decode_steps_s'],o['gap_p99_ms'],d['gap_p99_ms'],o['prefill_steps_s'],solo['decode_steps_s']]
        assert all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in values)
        assert o['prefill_covers_decode_tail'] and not o['prefill_tail_order_ambiguous']
        summaries.append(dict(rank=r['rank'],decode_share=o['decode_steps_s']/d['decode_steps_s'],
          split_alone_share=solo['decode_steps_s']/d['decode_steps_s'],p99_ratio=o['gap_p99_ms']/d['gap_p99_ms'],
          prefill_share=o['prefill_steps_s']/p['prefill_steps_s_upper'],raw_gap_ms=o['gap_p99_ms']))
    decode=min(r['decode_share'] for r in summaries);gap=max(r['p99_ratio'] for r in summaries)
    pre=min(r['prefill_share'] for r in summaries)
    return dict(verdict='PASS',decode_target_met=decode>=.80,p99_target_met=gap<=1.2,
        both_80_target=decode>=.80 and pre>=.80,prefill_target_proxy=pre>=.80,minimum_decode_share=decode,maximum_p99_ratio=gap,
        minimum_prefill_share=pre,ranks=summaries,
        scope='slice frontier only; 80/80 are soft comparison targets, not a verdict or a TTFT prediction')


class SplitController:
    """Offline policy prototype. Table keys: (active cohort rows, split).

    All entries must be measured on BOTH TP ranks, at relevant history/coverage.
    No invented interpolation or presumed linear SM scaling. Select only at a
    fully fenced boundary; unknown loads retain the safe split and flag unknown.
    Maximize the weaker lane, then the product of both shares; decode breaks
    ties. There is no threshold saturation at 80/80. p99 remains reported.
    """
    def __init__(self,table,safe_split=44):self.table=table;self.current=safe_split;self.pending=None;self.streak=0
    def choose(self,rows,fenced,split_cost_ms,slack_ms):
        if rows==0:return dict(split=self.current,switch=False,reason='no decode; ordinary prefill allowed after fence')
        options=[(split,v) for (load,split),v in self.table.items() if load==rows and
                 v.get('validated') is True and all(math.isfinite(v[k]) and v[k]>0 for k in ('decode_share','prefill_share','p99_ratio'))]
        if not options:return dict(split=self.current,switch=False,reason='infeasible_or_uncalibrated')
        candidate=max(options,key=lambda x:(min(x[1]['decode_share'],x[1]['prefill_share']),
            x[1]['decode_share']*x[1]['prefill_share'],x[1]['decode_share'],-x[0]))[0]
        if candidate==self.pending:self.streak+=1
        else:self.pending=candidate;self.streak=1
        if not fenced:return dict(split=self.current,switch=False,reason='wait_for_both_lane_fences')
        if candidate==self.current:return dict(split=self.current,switch=False,reason='already_selected')
        if split_cost_ms is None or not math.isfinite(split_cost_ms) or split_cost_ms>slack_ms:
            return dict(split=self.current,switch=False,reason='switch_budget_unknown_or_infeasible')
        if candidate<self.current and self.streak<4:return dict(split=self.current,switch=False,reason='downshift_hysteresis')
        self.current=candidate;return dict(split=candidate,switch=True,reason='measured_frontier')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--grade',type=Path)
    p.add_argument('--rung',choices=['profile','bandwidth','bank','model-terminal','profile-terminal']);p.add_argument('--split',type=int,default=24)
    p.add_argument('--rows',type=int,default=4);p.add_argument('--pair',choices=['1,3','0,2'],default='1,3')
    p.add_argument('--out',type=Path);p.add_argument('--config',type=Path)
    a=p.parse_args()
    if a.grade:
        try:
            errors=[json.loads(p.read_text()) for p in a.grade.parent.glob(a.grade.name+'.rank*.error.json')]
            if any(x.get('fatal_correctness') for x in errors):
                print(json.dumps(dict(verdict='FAIL',reason='explicit correctness failure')));return 1
            if errors:raise ValueError('rank error artifact')
            d=json.loads(a.grade.read_text())
            for rank in (0,1):
                assert json.loads(Path(str(a.grade)+f'.rank{rank}.progress.json').read_text())['phase']=='completed'
            if d['rung']=='profile-terminal' and all(x.get('value_probe',{}).get('enabled') is True for x in d['ranks']):
                from s5_value import grade
                r=grade(d);print(json.dumps(r,allow_nan=False));return {'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[r['verdict']]
            if any(x.get('correctness_verified') is False for x in d['ranks']):
                print(json.dumps(dict(verdict='FAIL',reason='concurrent trajectory differs from solo',data=d)));return 1
            if d['rung']=='profile-terminal':assert all(x.get('correctness_verified') is True for x in d['ranks'])
            if d['rung'] in ('profile','profile-terminal'):r=score_profile(d)
            else:
                assert len(d['ranks'])==2 and {x['rank'] for x in d['ranks']}=={0,1} and all(x['completed'] for x in d['ranks'])
                if d['rung']=='bank':assert all(len(x['switches'])==18 and len(x['bank_memory'])==4 for x in d['ranks'])
                if d['rung']=='model-terminal':
                    for x in d['ranks']:
                        m=x['modes']['overlap'];assert x['steps'] in (30,200,400)
                        assert len(m['raw_decode'])==x['steps'] and m['raw_prefill']
                        assert x['audit']['verdict']=='PASS' and x['correctness_verified'] is True
                        assert all(type(y['gpu_ms']) in (int,float) and math.isfinite(y['gpu_ms']) and y['gpu_ms']>0 for key in ('raw_decode','raw_prefill') for y in m[key])
                        if x['ablation']['marlin_cap']:
                            caps=[r for r in x['marlin_cap']['streams'] if r['cap']>0]
                            assert len(caps)>=2 and all(r['calls']>0 and r['intercepted_queries']>=r['calls'] for r in caps)
                if d['rung']=='bandwidth':assert all({'overlap','resident_overlap'}<=x['modes'].keys() for x in d['ranks'])
                r=dict(verdict='PASS',scope='diagnostic completed only; inspect per-rank evidence',data=d)
        except Exception:r=dict(verdict='INCONCLUSIVE',reason='missing, malformed, unproven or incomplete evidence')
        print(json.dumps(r,allow_nan=False));return 0 if r['verdict']=='PASS' else 2
    if not a.rung or not a.out or not a.config:p.error('GPU entry requires rung/out/config')
    if a.split not in (8,12,16,20,24,34,44) or a.rows not in (1,2,3,4,5):p.error('unreviewed split/load')
    from s5_variants_gpu import run
    return run(a)
if __name__=='__main__':raise SystemExit(main())
