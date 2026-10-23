import gzip,json,glob,collections,re,sys
for f in sys.argv[1:]:
    ev=json.load(gzip.open(f))['traceEvents']
    G=sorted([e for e in ev if e.get('cat')=='gpu_user_annotation' and e.get('name','').startswith('execute_context')], key=lambda e:e['ts'])
    kern=sorted([e for e in ev if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')], key=lambda e:e['ts'])
    # merge duplicate annotations (one per stream) into one range per step
    steps=[]
    for g in G:
        if steps and g['name']==steps[-1][0] and g['ts']<=steps[-1][2]: steps[-1][2]=max(steps[-1][2], g['ts']+g['dur']); continue
        steps.append([g['name'], g['ts'], g['ts']+g['dur']])
    agg=collections.defaultdict(lambda:[0,0,0,0]); top=collections.Counter(); nP=0
    for name,t0,t1 in steps:
        kind='P' if 'context_1' in name else 'D'
        ks=[k for k in kern if t0<=k['ts']<t1]
        busy=0; cur=t0
        for k in ks:
            a,b=k['ts'],k['ts']+k['dur']
            if b>cur: busy+=b-max(a,cur); cur=b
        a=agg[kind]; a[0]+=1; a[1]+=t1-t0; a[2]+=busy
        if kind=='P':
            nP+=1
            for k in ks:
                if 'nccl' in k['name'].lower(): continue
                top[re.sub(r'\(.*','',re.sub(r'<.*','',k['name']))[:55]]+=k['dur']
    print('=====', f[:12], {k:(v[0], round(v[1]/v[0]/1000,1), round(v[2]/v[0]/1000,1)) for k,v in agg.items()}, '(count, GPU span ms, GPU busy ms)')
    for name,us in top.most_common(8): print('   %6.2f ms  %s'%(us/nP/1000, name))
