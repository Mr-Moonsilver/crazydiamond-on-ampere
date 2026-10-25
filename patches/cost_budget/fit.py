#!/usr/bin/env python3
"""Fit explicitly labelled window-mean copy envelopes and client-gap tail proxy.

No claim that client gaps or 5-second bucket means are individual GPU step times.
"""
import argparse, hashlib, json, math, re
from pathlib import Path
RANKS=['0:0','0:1','1:0','1:1']
BINS=[12,25,37,50,65]
ROW=re.compile(r'pages<=(\d+|max): n=(\d+) rows=([\d.]+) pages=([\d.]+) sparse=\d+% cov=([\w.]+) moved=([\d.]+)p pre=([\d.]+) copy=([\d.]+) attn=([\d.]+) span=([\d.]+) ms')
def quantile(x,q):return sorted(x)[min(len(x)-1,int(q*len(x)))]
def main(paths,out):
    records=[];sources=[];gaps=[];excluded=0
    for p in paths:
        raw=p.read_bytes();sources.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()})
        if p.suffix=='.json':
            d=json.loads(raw);assert d['verdict']=='PASS' and not d['fails']
            # Only fixed-chunk, alternation reference cases listed by the caller.
            chunk=int(re.search(r't(\d+)\.json$',p.name)[1])
            r=d['raw'];m=r['marks'];per=[]
            for events in r['events']:
                advances=[];last=0
                for t,n in events:
                    if n>last:advances.append(t);last=n
                intervals=[(b-a)*1000 for a,b in zip(advances,advances[1:]) if m['p0']<b<=m['p1']]
                last=max((t for t in advances if t<=m['p1']),default=None)
                assert last is not None
                intervals.append((m['p1']-max(last,m['p0']))*1000) # same open-gap rule as stall_trace2
                assert len(intervals)>=5
                per.append(quantile(intervals,.99))
            gaps.append({'chunk':chunk,'gap99_ms':max(per),'decode99_ms':d['before']['worst_stream_gap_ms']['p99'],
                         'history':d['long_prompt'],'path':str(p)})
        else:
            for line in raw.decode().splitlines():
                rank=re.search(r'Worker_PP(\d+)_TP(\d+)',line)
                if not rank:continue
                for b,n,rows,pages,cov,moved,pre,cp,at,span in ROW.findall(line):
                    n=int(n);rows,pages,moved,cp,span=map(float,(rows,pages,moved,cp,span))
                    # Avoid JIT-dominated tiny first-window samples. This is a
                    # declared filter, not trimming individual slow steps.
                    if n<8 or rows<128 or moved<=0 or cp<=0 or span>100:
                        excluded+=1;continue
                    records.append({'rank':':'.join(rank.groups()),'bin':int(b) if b!='max' else 65,
                                    'coverage':min(1.,moved/pages),'copy_ms_per_page':cp/moved,'n':n})
    cells={}
    for rank in RANKS:
        candidates=[r for r in records if r['rank']==rank];assert candidates
        cells[rank]=[]
        for bound in BINS:
            rows=[r for r in candidates if r['bin']==bound]
            measured=bool(rows);rows=rows or candidates
            cells[rank].append({'pages_le':bound,'copy_ms_per_page':quantile([r['copy_ms_per_page'] for r in rows],.95),
                'coverage_upper':quantile([r['coverage'] for r in rows],.95) if measured else 1.,
                'windows':len(rows) if measured else 0})
    def copy_ms(history,coverage=None):
        pages=math.ceil(history/4048)
        return max(6*pages*(coverage if coverage is not None else next(c for c in v if c['pages_le']>=min(65,pages))['coverage_upper'])*
                   next(c for c in v if c['pages_le']>=min(65,pages))['copy_ms_per_page'] for v in cells.values())
    knots={}
    for g in gaps:
        residual=max(0.,g['gap99_ms']-g['decode99_ms']-copy_ms(g['history']))
        knots[g['chunk']]=max(knots.get(g['chunk'],0.),residual)
    monotone=0.;kn=[]
    for c,v in sorted(knots.items()):monotone=max(monotone,v);kn.append([c,monotone])
    model={'schema':'flash-prefill-cost/v1','tail_kind':'window-mean-copy-p95 + per-stream-client-gap-p99 proxy',
      'calibrated_gpu_step_tail':False,'page_size':4048,'qsa_layers_per_rank':6,'max_fit_history':max(g['history'] for g in gaps),
      'copy_cells':cells,'other_ms_knots':kn,'decode_gap99_ms':max(g['decode99_ms'] for g in gaps),
      'sources':sources,'fit_windows':len(records),'excluded_windows':excluded,'tail_cases':gaps,
      'scope':'PP2 TP2 BF16 host KV, no retention, observed x8 rank 1:0; not overlap or deadline guarantee'}
    out.write_text(json.dumps(model,indent=2)+'\n');print(json.dumps({'windows':len(records),'excluded':excluded,'knots':kn,'tail':'proxy, not calibrated GPU p99'}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);p.add_argument('sources',nargs='+',type=Path);a=p.parse_args();main(a.sources,a.out)
