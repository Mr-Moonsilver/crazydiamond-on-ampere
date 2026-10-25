"""CPU arithmetic only. No subtraction between CUDA events in different lanes."""
import math

def percentile(xs,f):
    ys=sorted(xs);return ys[min(len(ys)-1,int((len(ys)-1)*f))] if ys else None

def summarize(rows,durations,gaps,tail,mode):
    d,p=rows['decode'],rows['prefill']
    if any(not math.isfinite(v) or v<=0 for v in durations.values()):raise ValueError('invalid lane duration')
    r=dict(raw_decode=d,raw_prefill=p,mode=mode,duration_source='lane-local CUDA events only',
           prefill_covers_decode_tail=tail,prefill_tail_order_ambiguous=not tail,
           cross_lane_ordering='conservative host issue/completion bounds; terminal-event CPU queries for tail')
    if d:
        window=durations['decode']
        inside=[x for x in p if x['submitted_s']>=d[0]['observed_s'] and x['observed_s']<=d[-1]['submitted_s']]
        possible=[x for x in p if x['observed_s']>=d[0]['submitted_s'] and x['submitted_s']<=d[-1]['observed_s']]
        host=[(b['observed_s']-a['observed_s'])*1000 for a,b in zip(d,d[1:])]
        r.update(window_s=window,decode_steps_s=len(d)/window,gpu_completion_gaps_ms=gaps,gap_p99_ms=percentile(gaps,.99),
                 gap_p50_ms=percentile(gaps,.5),gap_max_ms=max(gaps),host_gap_p99_ms=percentile(host,.99),
                 prefill_steps_s=len(inside)/window,prefill_steps_s_upper=len(possible)/window,
                 prefill_boundary_ambiguous=len(possible)-len(inside))
    else:
        r.update(window_s=durations['prefill'],prefill_steps_s=len(p)/durations['prefill'],
                 prefill_steps_s_upper=len(p)/durations['prefill'])
    return r
