#!/usr/bin/env python3
"""Grade E2 same-input byte evidence only; model-level rule v3 stays separate.

Missing coverage is INCONCLUSIVE; any primary failure or invalid evidence FAILS.
No payloads are loaded or echoed. Default: six layers on each of four workers,
dense >=200k/multiple buckets and a prefix-hit invocation per layer.
"""
import argparse
import json
from pathlib import Path
import re

RANKS={f'PP{p}_TP{t}' for p in (0,1) for t in (0,1)}
COUNTERS=('calls','dense_dma','sparse_uva','empty','buckets','bucket_bytes','output_bytes',
          'max_length','prefix_hit_calls','dense_max_length','dense_multibucket_calls')


def grade(root,min_dense_length=200000,need_prefix=True,require_dense_only=False):
    root=Path(root);fails=[];missing=[];summary={}
    if list(root.glob('*.failure.json')):fails.append('immutable worker failure artifact exists')
    paths=list(root.glob('*.exact.json'))
    if {p.name.removesuffix('.exact.json') for p in paths}-RANKS:
        fails.append('unknown or duplicate worker evidence')
    for rank in sorted(RANKS):
        p=root/(rank+'.exact.json')
        if not p.is_file():missing.append(rank+': no evidence');continue
        try:
            d=json.loads(p.read_text())
            if d['schema']!=1 or d['rank']!=rank or d['failed'] is not False or d['active'] is not False:
                raise ValueError('worker status')
            if d['page_tokens']!=4048 or d['arena_pages']!=24:raise ValueError('baseline arena geometry')
            layers=d['layers'];indices=[]
            if not isinstance(layers,dict) or len(layers)!=6:raise ValueError('layers')
            for name,s in layers.items():
                m=re.findall(r'(?:^|\.)layers\.(\d+)\.',name)
                if len(m)!=1:raise ValueError('layer name')
                indices.append(int(m[0]))
                if any(type(s.get(k)) is not int or s[k]<0 for k in COUNTERS):raise ValueError('counters')
                if s['calls']!=s['dense_dma']+s['sparse_uva']+s['empty']:raise ValueError('route counts')
                if (s['prefix_hit_calls']>s['calls'] or s['dense_multibucket_calls']>s['dense_dma'] or
                        s['dense_max_length']>s['max_length'] or s['buckets']<s['dense_dma']+s['dense_multibucket_calls'] or
                        s['bucket_bytes']%(4048*1024)!=0 or s['bucket_bytes']<s['buckets']*4048*1024 or
                        (s['buckets']==0)!=(s['bucket_bytes']==0)):
                    raise ValueError('counter consistency')
                if not s['calls'] or not s['output_bytes']:missing.append(rank+': QSA output coverage')
                if not s['dense_multibucket_calls'] or s['dense_max_length']<min_dense_length:
                    missing.append(rank+': dense long-context/multibucket coverage')
                if need_prefix and not s['prefix_hit_calls']:missing.append(rank+': prefix-hit coverage')
                if require_dense_only and s['sparse_uva']:
                    fails.append(rank+': sparse fallback observed in declared dense-only phase')
            pp=int(rank[2]);expected=set(range(3+24*pp,24+24*pp,4))
            if set(indices)!=expected:raise ValueError('PP layer placement')
            summary[rank]={k:sum(s[k] for s in layers.values()) for k in ('calls','dense_dma','sparse_uva','buckets','bucket_bytes','output_bytes')}
        except (OSError,ValueError,KeyError,TypeError,AttributeError):
            fails.append(rank+': malformed/inconsistent evidence')
    verdict='FAIL' if fails else 'INCONCLUSIVE' if missing else 'PASS'
    return dict(verdict=verdict,fails=sorted(set(fails)),missing=sorted(set(missing)),workers=summary,
                require_dense_only=require_dense_only,
                scope='same-input QSA bytes; no MoE noise waiver or model qualification')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--min-dense-length',type=int,default=200000)
    p.add_argument('--no-prefix',action='store_true',help='diagnostic only; final gate requires prefix coverage')
    p.add_argument('--require-dense-only',action='store_true',help='dense verification phase: any sparse fallback is FAIL')
    a=p.parse_args()
    if a.min_dense_length<1:p.error('positive minimum required')
    out=grade(a.directory,a.min_dense_length,not a.no_prefix,a.require_dense_only);print(json.dumps(out,sort_keys=True))
    raise SystemExit({'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[out['verdict']])
