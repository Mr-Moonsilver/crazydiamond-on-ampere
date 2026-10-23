import gzip,json,glob,collections,re,sys,bisect
f=glob.glob('c512/dp0_pp1_tp1*.json.gz')[0]
ev=json.load(gzip.open(f))['traceEvents']
G=sorted([e for e in ev if e.get('cat')=='gpu_user_annotation' and e.get('name','').startswith('execute_context_1')], key=lambda e:e['ts'])
steps=[]
for g in G:
    if steps and g['ts']<=steps[-1][1]: steps[-1][1]=max(steps[-1][1], g['ts']+g['dur']); continue
    steps.append([g['ts'], g['ts']+g['dur']])
steps=steps[:30]
kern=sorted([e for e in ev if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')], key=lambda e:e['ts'])
rt=sorted([e for e in ev if e.get('cat') in ('cuda_runtime','cuda_driver')], key=lambda e:e['ts'])
cpu=sorted([e for e in ev if e.get('cat') in ('cpu_op','user_annotation')], key=lambda e:e['ts'])
cts=[c['ts'] for c in cpu]; rts=[r['ts'] for r in rt]
corr={r['args']['correlation']:r for r in rt if 'correlation' in r.get('args',{})}
hist=collections.Counter(); prevk=collections.Counter(); nextop=collections.Counter(); rtblock=collections.Counter()
total=0
for t0,t1 in steps:
    ks=[k for k in kern if t0<=k['ts']<t1]
    end=t0; last=None
    for k in ks:
        g=k['ts']-end
        if g>200:
            total+=g
            hist[min(int(g/1000),10)]+=1
            prevk[re.sub(r'\(.*','',re.sub(r'<.*','',last['name']))[:50] if last else 'start']+=g
            # the kernel after the gap was launched by which runtime call / cpu op, and when
            r=corr.get(k.get('args',{}).get('correlation'))
            if r:
                i=bisect.bisect_right(cts, r['ts'])-1; best=None; j=i
                while j>=0 and j>i-4000:
                    c=cpu[j]
                    if c['ts']<=r['ts']<=c['ts']+c['dur'] and c.get('cat')=='cpu_op' and (best is None or c['dur']<best['dur']): best=c
                    j-=1
                nextop[best['name'] if best else '?']+=g
                # long-blocking runtime calls on the CPU just before this launch (sync waits)
                i=bisect.bisect_right(rts, r['ts'])-1
                for j in range(max(0,i-60), i):
                    rr=rt[j]
                    if rr['dur']>500 and rr['ts']+rr['dur']>end: rtblock[rr['name']]+=1
        if k['ts']+k['dur']>end: end=k['ts']+k['dur']; last=k
n=len(steps)
print('GPU idle in gaps >0.2 ms: %.1f ms/step'%(total/n/1000))
print('gap size histogram (ms bucket: count/step):', {k: round(v/n,1) for k,v in sorted(hist.items())})
print('idle after kernel:'); [print('  %5.1f ms  %s'%(v/n/1000,k)) for k,v in prevk.most_common(8)]
print('idle before launch from cpu op:'); [print('  %5.1f ms  %s'%(v/n/1000,k)) for k,v in nextop.most_common(10)]
print('blocking runtime calls preceding gaps:', rtblock.most_common(5))
