import gzip,json,glob,collections,sys,re
for f in sorted(glob.glob('dp0_pp*_tp0*.json.gz')):
    d=json.load(gzip.open(f)); ev=d['traceEvents']
    steps=[e for e in ev if e.get('cat')=='user_annotation' and e.get('name','').startswith('execute_context')]
    kern=[e for e in ev if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
    kern.sort(key=lambda e:e['ts'])
    rt=[e for e in ev if e.get('cat') in ('cuda_runtime','cuda_driver')]
    print('=====',f[:12], 'steps', len(steps))
    agg=collections.defaultdict(list)
    for s in steps:
        kind='P' if 'context_1' in s['name'] else 'D'
        t0,t1=s['ts'],s['ts']+s['dur']
        ks=[k for k in kern if k['ts']>=t0 and k['ts']<t1+50000]  # kernels launched in the step may run a bit later; bounded
        ks=[k for k in ks if k['ts']<t1+20000]
        busy=sum(k['dur'] for k in ks)
        syncs=[r for r in rt if t0<=r['ts']<t1 and re.search(r'Synchronize|MemcpyAsync|cudaMemcpy\b|EventSynchronize',r['name']) and r['dur']>200]
        agg[kind].append((s['dur']/1000, busy/1000, len(ks), sum(r['dur'] for r in syncs)/1000, len(syncs)))
    for kind,v in agg.items():
        n=len(v); print(kind, 'n=%d wall %.1f ms  gpu-busy %.1f ms  kernels %d  blocked-in-sync %.1f ms (%d calls)'%(n, sum(x[0] for x in v)/n, sum(x[1] for x in v)/n, sum(x[2] for x in v)//n, sum(x[3] for x in v)/n, sum(x[4] for x in v)//n))
    # top kernels in prefill steps
    P=[s for s in steps if 'context_1' in s['name']]
    if P:
        top=collections.Counter(); 
        for s in P:
            t0,t1=s['ts'],s['ts']+s['dur']
            for k in kern:
                if t0<=k['ts']<t1+20000: top[re.sub(r'<.*','',k['name'])[:70]]+=k['dur']
        tot=sum(top.values())
        for name,us in top.most_common(14): print('   %6.1f ms/step %4.1f%%  %s'%(us/1000/len(P), 100*us/tot, name))
        st=collections.Counter()
        for s in P:
            t0,t1=s['ts'],s['ts']+s['dur']
            for r in rt:
                if t0<=r['ts']<t1: st[r['name']]+=r['dur']
        print('   cuda runtime time per prefill step:', [(k, round(v/1000/len(P),1)) for k,v in st.most_common(6)])
