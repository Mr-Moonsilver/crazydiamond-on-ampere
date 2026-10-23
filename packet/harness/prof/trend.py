import gzip,json,glob,collections,re
f=sorted(glob.glob('dp0_pp1_tp0*.json.gz'))[0]
ev=json.load(gzip.open(f))['traceEvents']
P=sorted([e for e in ev if e.get('cat')=='user_annotation' and e.get('name','').startswith('execute_context_1')], key=lambda e:e['ts'])
kern=sorted([e for e in ev if e.get('cat')=='kernel'], key=lambda e:e['ts'])
# map kernels to launching cpu op via correlation id
corr={}
for e in ev:
    if e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{}): corr[e['args']['correlation']]=e
cpu=sorted([e for e in ev if e.get('cat')=='cpu_op'], key=lambda e:e['ts'])
import bisect
cts=[c['ts'] for c in cpu]
def parent_op(rt):
    # innermost aten op containing the runtime call
    i=bisect.bisect_right(cts, rt['ts'])-1; best=None
    j=i
    while j>=0 and j>i-400:
        c=cpu[j]
        if c['ts']<=rt['ts']<=c['ts']+c['dur']:
            if best is None or c['dur']<best['dur']: best=c
        j-=1
    return best['name'] if best else '?'
keys=['vectorized_gather_kernel','cutlass::Kernel2','marlin_moe','_qsa_gather_selected_host_rows','_qsa_sparse_staged_gqa_kernel','cross_device_reduce']
rows=[]
gather_ops=collections.Counter()
for idx,s in enumerate(P):
    t0,t1=s['ts'],s['ts']+s['dur']
    c=collections.Counter()
    for k in kern:
        if t0<=k['ts']<t1+20000:
            for kk in keys:
                if kk in k['name']: c[kk]+=k['dur']
            if 'vectorized_gather_kernel' in k['name'] and idx<6:
                rt=corr.get(k.get('args',{}).get('correlation'))
                if rt: gather_ops[parent_op(rt)]+=k['dur']
    rows.append((idx, s['dur']/1000, {kk[:18]:round(c[kk]/1000,1) for kk in keys}))
for r in rows[::8]+[rows[-1]]: print(r)
print('vectorized_gather launched from (first 6 prefill steps, ms):', [(k, round(v/1000/6,1)) for k,v in gather_ops.most_common(6)])
