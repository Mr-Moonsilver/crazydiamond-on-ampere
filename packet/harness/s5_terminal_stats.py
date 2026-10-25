"""CPU arithmetic only. No subtraction between CUDA events in different lanes."""
import math

def percentile(xs,f):
    ys=sorted(xs);return ys[min(len(ys)-1,int((len(ys)-1)*f))] if ys else None

def tail_coverage(rows,proof):
    """Validate the new workload-tail proof; never reconstruct it for old runs."""
    try:
        if proof['version']!=1 or proof['basis']!='continuous_prefill_backlog':return False
        d,p=rows['decode'],rows['prefill'];q=proof['tail_probe']
        if len(d)<2 or len(p)<2 or [x['step'] for x in p]!=list(range(len(p))):return False
        if [x['step'] for x in d]!=list(range(len(d))):return False
        for lane in (d,p):
            for i,x in enumerate(lane):
                ts=[x[k] for k in ('submitted_s','issued_s','observed_s')]
                if not all(type(t) in (int,float) and math.isfinite(t) for t in ts) or ts!=sorted(ts):return False
                if i and x['submitted_s']<lane[i-1]['observed_s']:return False
        if not (proof['backlog_open_s']<=p[0]['submitted_s']<=p[0]['issued_s']<=d[0]['submitted_s']):return False
        if not (d[-1]['observed_s']<=q['probe_s']<=p[-1]['submitted_s']):return False
        if [x['step'] for x in p if x.get('post_decode_guard') is True]!=[len(p)-1]:return False
        n=q['ticket_step']
        if type(n) is not int or n<0 or n>=len(p)-1:return False
        if q['ticket_submitted_s']!=p[n]['submitted_s'] or q['ticket_submitted_s']>q['probe_s']:return False
        if type(q['start_complete']) is not bool or type(q['end_recorded']) is not bool:return False
        if q['end_recorded'] and type(q['end_complete']) is not bool:return False
        if not q['end_recorded'] and q['end_complete'] is not None:return False
        return True
    except (KeyError,TypeError,IndexError):return False


def summarize(rows,durations,gaps,tail,mode,proof=None):
    d,p=rows['decode'],rows['prefill']
    if any(not math.isfinite(v) or v<=0 for v in durations.values()):raise ValueError('invalid lane duration')
    r=dict(raw_decode=d,raw_prefill=p,mode=mode,duration_source='lane-local CUDA events only',
           prefill_covers_decode_tail=tail,prefill_tail_order_ambiguous=not tail,
           cross_lane_ordering='conservative host issue/completion bounds; terminal-event CPU queries for tail')
    if proof is not None:
        r['tail_proof']=proof
        r['prefill_covers_decode_tail']=tail_coverage(rows,proof)
        r['prefill_tail_order_ambiguous']=not r['prefill_covers_decode_tail']
        q=proof.get('tail_probe') or {}
        r['prefill_device_active_at_tail_probe']=q.get('start_complete') is True and q.get('end_recorded') is True and q.get('end_complete') is False
        r['prefill_tail_dispatch_pending']=q.get('end_recorded') is False
    if d:
        window=durations['decode']
        inside=[x for x in p if not x.get('post_decode_guard') and x['submitted_s']>=d[0]['observed_s'] and x['observed_s']<=d[-1]['submitted_s']]
        possible=[x for x in p if not x.get('post_decode_guard') and x['observed_s']>=d[0]['submitted_s'] and x['submitted_s']<=d[-1]['observed_s']]
        host=[(b['observed_s']-a['observed_s'])*1000 for a,b in zip(d,d[1:])]
        r.update(window_s=window,decode_steps_s=len(d)/window,gpu_completion_gaps_ms=gaps,gap_p99_ms=percentile(gaps,.99),
                 gap_p50_ms=percentile(gaps,.5),gap_max_ms=max(gaps),host_gap_p99_ms=percentile(host,.99),
                 prefill_steps_s=len(inside)/window,prefill_steps_s_upper=len(possible)/window,
                 prefill_boundary_ambiguous=len(possible)-len(inside))
    else:
        r.update(window_s=durations['prefill'],prefill_steps_s=len(p)/durations['prefill'],
                 prefill_steps_s_upper=len(p)/durations['prefill'])
    return r
