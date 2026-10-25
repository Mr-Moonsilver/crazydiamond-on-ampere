# Inserted into the pinned ops_qsa.py by make_sparse_stage.py.
@triton.jit
def _s6_mark(indices, rows, flags, S: tl.constexpr, K: tl.constexpr,
             N: tl.constexpr, R: tl.constexpr, B: tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    r=tl.load(rows+i//K, i<R*K, other=0)
    v=tl.load(indices+r*S+i%K, i<R*K, other=-1)
    tl.atomic_or(flags+tl.maximum(v,0),1,(i<R*K)&(v>=0)&(v<N),sem='relaxed')

@triton.jit
def _s6_scan(flags, ranks, totals, N: tl.constexpr, B: tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    f=tl.load(flags+i,i<N,other=0)
    p=tl.cumsum(f)
    tl.store(ranks+i,p,i<N)
    tl.store(totals+tl.program_id(0),tl.sum(f,0))

@triton.jit
def _s6_offsets(totals, offsets, info, N: tl.constexpr, G: tl.constexpr, B: tl.constexpr):
    i=tl.arange(0,B); n=tl.load(totals+i,i<G,other=0)
    p=tl.cumsum(n); count=tl.sum(n,0)
    tl.store(offsets+i,p-n,i<G)
    tl.store(info,count);tl.store(info+1,count*5<N*4)

@triton.jit
def _s6_pack(flags, ranks, offsets, tokens, N: tl.constexpr, B: tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    f=tl.load(flags+i,i<N,other=0)
    pos=tl.load(ranks+i,i<N,other=0)+tl.load(offsets+tl.program_id(0))-1
    tl.store(tokens+pos,i,(i<N)&(f!=0))
    tl.store(ranks+i,pos,(i<N)&(f!=0))

@triton.jit
def _s6_remap(indices, rows, ranks, info, out, S: tl.constexpr, O: tl.constexpr,
              K: tl.constexpr, R: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    r=tl.load(rows+i//K,i<R*K,other=0)
    v=tl.load(indices+r*S+i%K,i<R*K,other=-1)
    compact=tl.load(ranks+tl.maximum(v,0),(i<R*K)&(v>=0)&(v<N),other=-1)
    sparse=tl.load(info+1)!=0
    tl.store(out+r*O+i%K,tl.where(sparse,compact,v),i<R*K)

@triton.jit
def _s6_gather(cache,tokens,table,arena,info,
               CB:tl.constexpr,CH:tl.constexpr,CT:tl.constexpr,
               AB:tl.constexpr,AH:tl.constexpr,AT:tl.constexpr,
               LO:tl.constexpr,SIZE:tl.constexpr,N:tl.constexpr,PAGE:tl.constexpr,
               WIDTH:tl.constexpr,B:tl.constexpr):
    # Fixed grid, device bounds. No host count or shape-dependent allocation.
    t=tl.program_id(0);h=tl.program_id(1);base=LO*PAGE
    count=tl.load(info);sparse=tl.load(info+1)!=0
    limit=tl.where(sparse,count,N)
    if base+t<limit:
        logical=tl.load(tokens+base+t) if sparse else base+t
        physical=tl.load(table+logical//PAGE).to(tl.int64)
        d=tl.arange(0,B)
        val=tl.load(cache+physical*CB+h*CH+(logical%PAGE).to(tl.int64)*CT+d,d<WIDTH,other=0)
        tl.store(arena+(t//PAGE).to(tl.int64)*AB+h*AH+(t%PAGE)*AT+d,val,d<WIDTH)


def _s6_union(logical_indices,rows,num_tokens):
    # These buffers belong to this operation/stream. No global cache or cross-
    # request reuse. Sorted ascending packing matches torch.unique(sorted=True).
    b=1024;groups=triton.cdiv(num_tokens,b)
    flags=torch.zeros((num_tokens,),device=rows.device,dtype=torch.int32)
    ranks=torch.empty_like(flags);tokens=torch.empty_like(flags)
    totals=torch.empty((groups,),device=rows.device,dtype=torch.int32)
    offsets=torch.empty_like(totals);info=torch.empty((2,),device=rows.device,dtype=torch.int32)
    width=logical_indices.shape[1]-1
    _s6_mark[(triton.cdiv(rows.numel()*width,b),)](logical_indices,rows,flags,
        logical_indices.stride(0),width,num_tokens,rows.numel(),b)
    _s6_scan[(groups,)](flags,ranks,totals,num_tokens,b)
    _s6_offsets[(1,)](totals,offsets,info,num_tokens,groups,triton.next_power_of_2(groups))
    _s6_pack[(groups,)](flags,ranks,offsets,tokens,num_tokens,b)
    remapped=logical_indices.clone()
    _s6_remap[(triton.cdiv(rows.numel()*width,b),)](logical_indices,rows,ranks,info,remapped,
        logical_indices.stride(0),remapped.stride(0),width,rows.numel(),num_tokens,b)
    return tokens,info,remapped
