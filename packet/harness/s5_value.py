#!/usr/bin/env python3
"""CPU-only value report: matched controls, per-rank spikes, all planned cases."""
import json
import math
from pathlib import Path
import sys
from s5_terminal_stats import percentile

def positive(v):
    if type(v) not in (int,float) or not math.isfinite(v) or v<=0:raise ValueError('nonpositive/nonfinite timing')
    return v

def checked_mode(mode,steps):
    if type(steps) is not int or steps<2:raise ValueError('empty decode window')
    gaps=mode['gpu_completion_gaps_ms']
    if len(gaps)!=steps-1 or len(mode['raw_decode'])!=steps:raise ValueError('incomplete decode window')
    for v in gaps:positive(v)
    positive(mode['decode_steps_s']);positive(mode['gap_p99_ms'])
    return gaps

def comparison(full,solo,overlap,steps):
    f=checked_mode(full,steps);s=checked_mode(solo,steps);g=checked_mode(overlap,steps)
    for k in ('prefill_steps_s','prefill_steps_s_upper'):
        v=overlap[k]
        if type(v) not in (int,float) or not math.isfinite(v) or v<0:raise ValueError('invalid prefill rate')
    if overlap['prefill_steps_s_upper']<overlap['prefill_steps_s']:raise ValueError('inverted rate bounds')
    threshold=2*percentile(s,.5)
    return dict(decode_share_vs_solo=overlap['decode_steps_s']/solo['decode_steps_s'],
        decode_share_vs_full=overlap['decode_steps_s']/full['decode_steps_s'],
        solo_steps_s=solo['decode_steps_s'],full_steps_s=full['decode_steps_s'],
        overlap_steps_s=overlap['decode_steps_s'],solo_gap_p50_ms=percentile(s,.5),
        solo_gap_p99_ms=percentile(s,.99),full_gap_p99_ms=percentile(f,.99),
        gap_p50_ms=percentile(g,.5),gap_p90_ms=percentile(g,.9),gap_p99_ms=percentile(g,.99),
        gap_max_ms=max(g),p99_vs_solo=percentile(g,.99)/percentile(s,.99),
        p99_vs_full=percentile(g,.99)/percentile(f,.99),spike_threshold_ms=threshold,
        spike_count_gt_2x_solo=sum(v>threshold for v in g),gap_count=len(g),
        spike_fraction=sum(v>threshold for v in g)/len(g),
        solo_spike_count_same_threshold=sum(v>threshold for v in s),
        prefill_steps_s=overlap['prefill_steps_s'],prefill_steps_s_upper=overlap['prefill_steps_s_upper'],
        tail_proven=overlap['prefill_covers_decode_tail'] is True and not overlap['prefill_tail_order_ambiguous'])

def grade(data):
    rows=[];verdict='PASS'
    try:
        ranks=data['ranks']
        if len(ranks)!=2 or {r['rank'] for r in ranks}!={0,1}:raise ValueError('both ranks required')
        # A second rank's explicit correctness failure outranks missing timing
        # data on the first rank.
        if any(r.get(k,{}).get('passed') is False for r in ranks for k in ('correctness','noop_correctness','resident_correctness')):
            verdict='FAIL'
        for r in ranks:
            if r.get('completed') is not True or r['audit']['verdict']!='PASS':raise ValueError('completion/node placement missing')
            if r['max_connections'] not in ('8','32'):raise ValueError('unknown connections')
            steps=r['steps'];m=r['modes'];probe=r['value_probe']
            if probe.get('enabled') is not True:raise ValueError('not a value case')
            names=['overlap']+(['overlap_noop'] if probe['prefill_noop'] else [])+(['overlap_resident'] if probe.get('resident_prefill') else [])
            for name in names:
                correctness=r[{'overlap':'correctness','overlap_noop':'noop_correctness','overlap_resident':'resident_correctness'}[name]]
                if correctness['passed'] is False:verdict='FAIL'
                elif correctness['passed'] is not True:raise ValueError('missing correctness')
                if name=='overlap_noop' and r['noop_collective_calls']!=len(m[name]['raw_prefill'])*(1+2*r['spec']['layers']):
                    raise ValueError('no-op not bound to each prefill collective')
                row=dict(rank=r['rank'],connections=int(r['max_connections']),split=r['requested_split'],
                    actual_split=r['actual_split'],mode=name,correctness=correctness['passed'],
                    **comparison(m['decode'],m['decode_split'],m[name],steps))
                row['bitwise_equal']=all(correctness[k]['bitwise_first'] is True and correctness[k]['concurrent_states_match'] is True for k in ('decode','prefill'))
                solo_prefill=m[{'overlap':'prefill','overlap_noop':'noop_prefill','overlap_resident':'resident_prefill'}[name]]
                row['prefill_share']=row['prefill_steps_s']/positive(solo_prefill['prefill_steps_s'])
                row['prefill_share_upper']=row['prefill_steps_s_upper']/solo_prefill['prefill_steps_s']
                row['prefill_vs_original_solo']=row['prefill_steps_s']/positive(m['prefill']['prefill_steps_s'])
                row['vs_80_80_target']='met' if row['decode_share_vs_full']>=.80 and row['prefill_share']>=.80 else 'below'
                row['p99_target_met']=row['p99_vs_full']<=1.2
                row['target_only']=True
                if not row['tail_proven'] and verdict=='PASS':verdict='INCONCLUSIVE'
                rows.append(row)
    except (KeyError,ValueError,TypeError,ZeroDivisionError):
        if verdict!='FAIL':verdict='INCONCLUSIVE'
    return dict(verdict=verdict,ranks=rows,scope='four-layer slice; no-op is attribution only, no full-engine or release verdict',
                fresh_prompt_ttft_s=None,full_engine_leak_gate='not_run',
                spike_rule='strictly greater than twice this rank same-split solo median GPU completion gap')

def summarize(root):
    root=Path(root);cases=[]
    for name in ('green24-c8','green24-c32','green34-c8','green34-c32','qsa-sparse-c8','qsa-copy-kernels-c8'):
            folder=root/name/'profile-terminal'
            errors=[]
            for p in folder.glob('result.json.rank*.error.json'):
                try:errors.append(json.loads(p.read_text()))
                except (OSError,ValueError):pass
            try:
                d=json.loads((folder/'result.json').read_text());r=grade(d)
                stamped=json.loads((folder/'grade.json').read_text())
                if stamped['verdict']=='FAIL':r['verdict']='FAIL'
                elif stamped['verdict']!='PASS' and r['verdict']=='PASS':r['verdict']='INCONCLUSIVE'
            except (OSError,ValueError,KeyError,TypeError):r=dict(verdict='INCONCLUSIVE',ranks=[],reason='missing/malformed artifacts')
            if any(e.get('fatal_correctness') for e in errors):r['verdict']='FAIL'
            cases.append(dict(case=name,**r))
    verdict='FAIL' if any(c['verdict']=='FAIL' for c in cases) else 'INCONCLUSIVE' if any(c['verdict']!='PASS' for c in cases) else 'PASS'
    return dict(verdict=verdict,cases=cases)

def table(summary):
    lines=['case | rank | mode | decode/split solo | decode/full solo | prefill/own solo [upper] | vs 80/80 target | p50 / p99 ms | spikes >2x solo | tail | correct | verdict',
           '--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---']
    for case in summary['cases']:
        if not case['ranks']:lines.append(f"{case['case']} | — | — | — | — | — | — | — | — | — | — | {case['verdict']}")
        for r in case['ranks']:
            lines.append(f"{case['case']} | {r['rank']} | {r['mode']} | {r['decode_share_vs_solo']:.3f} | {r['decode_share_vs_full']:.3f} | {r['prefill_share']:.3f} [{r['prefill_share_upper']:.3f}] | {r['vs_80_80_target']} | {r['gap_p50_ms']:.3f} / {r['gap_p99_ms']:.3f} | {r['spike_count_gt_2x_solo']}/{r['gap_count']} | {r['tail_proven']} | {r['correctness']} | {case['verdict']}")
    return '\n'.join(lines)+'\n\n80/80 is a target, not a verdict. Fresh 200k TTFT and full-engine leak gates: not measured by this slice.\n'

def monitor_lines(case,report):
    lines=[]
    for r in report.get('ranks',[]):
        lines.append(f"S5_VALUE case={case} rank={r['rank']} mode={r['mode']} "
            f"share_solo={r['decode_share_vs_solo']:.4f} share_full={r['decode_share_vs_full']:.4f} "
            f"gap_p50_ms={r['gap_p50_ms']:.3f} gap_p99_ms={r['gap_p99_ms']:.3f} "
            f"spikes_gt_2x={r['spike_count_gt_2x_solo']}/{r['gap_count']} threshold_ms={r['spike_threshold_ms']:.3f} "
            f"prefill_share={r['prefill_share']:.4f} prefill_share_upper={r['prefill_share_upper']:.4f} target80_80={r['vs_80_80_target']} "
            f"bitwise_equal={'y' if r['bitwise_equal'] else 'n'} tail={'y' if r['tail_proven'] else 'n'} verdict={report['verdict']}")
    if not lines:lines.append(f"S5_VALUE case={case} rank=unknown verdict={report['verdict']} metrics=unavailable")
    return lines

if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--case':
        path=Path(sys.argv[3])
        try:r=json.loads(path.read_text())
        except (OSError,ValueError):r=dict(verdict='INCONCLUSIVE',ranks=[])
        print('\n'.join(monitor_lines(sys.argv[2],r)));raise SystemExit(0)
    root=Path(sys.argv[1]);r=summarize(root)
    (root/'summary.json').write_text(json.dumps(r)+'\n');(root/'summary.md').write_text(table(r))
    print(table(r),end='');print('S5 VALUE '+r['verdict'])
    raise SystemExit({'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[r['verdict']])
