import gzip,json,glob,collections,re
f=sorted(glob.glob('dp0_pp1_tp0*.json.gz'))[0]
ev=json.load(gzip.open(f))['traceEvents']
P=sorted([e for e in ev if e.get('cat')=='user_annotation' and e.get('name','').startswith('execute_context_1')], key=lambda e:e['ts'])
kern=sorted([e for e in ev if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')], key=lambda e:e['ts'])
streams=collections.Counter(k.get('args',{}).get('stream') for k in kern)
print('streams', streams.most_common(5))
top=collections.Counter(); idle=0; span=0
for s in P[:40]:
    t0,t1=s['ts'],s['ts']+s['dur']
    ks=[k for k in kern if t0<=k['ts']<t1]
    for k in ks: top[re.sub(r'\(.*','',re.sub(r'<.*','',k['name']))[:60]]+=k['dur']
    # idle on the union of intervals
    iv=sorted((k['ts'],k['ts']+k['dur']) for k in ks)
    cur=t0; busy=0
    for a,b in iv:
        if b<=cur: continue
        busy+=b-max(a,cur); cur=max(cur,b)
    idle+=(t1-t0)-busy; span+=t1-t0
n=40
print('mean step %.1f ms, GPU idle %.1f ms/step (%.0f%%)'%(span/n/1000, idle/n/1000, 100*idle/span))
tot=sum(top.values())
for name,us in top.most_common(30): print('  %6.2f ms  %4.1f%%  %s'%(us/n/1000, 100*us/tot, name))
