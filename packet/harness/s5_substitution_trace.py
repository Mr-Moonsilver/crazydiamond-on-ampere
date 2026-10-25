"""Optional failure-only op localization; never a correctness waiver or timing run."""
from itertools import combinations
from contextlib import contextmanager
from s5_substitution import arrays,capture,tensor_diff


def changed_axes(a,b):
    """Index geometry only, never tensor/token values; axis 0=tokens, last=hidden."""
    import numpy as np
    if (a.name,a.dtype,a.shape)!=(b.name,b.dtype,b.shape):return []
    aa=arrays(a)[0];bb=arrays(b)[0]
    mask=np.any(aa.view(np.uint8).reshape(-1,aa.dtype.itemsize)!=
                bb.view(np.uint8).reshape(-1,bb.dtype.itemsize),axis=1).reshape(a.shape)
    result=[]
    for axis,size in enumerate(a.shape):
        other=tuple(i for i in range(len(a.shape)) if i!=axis)
        counts=mask.sum(axis=other);indices=np.flatnonzero(counts)
        result.append(dict(axis=axis,size=size,distinct_indices=int(indices.size),
            first_indices=indices[:32].tolist(),indices_truncated=indices.size>32,
            max_differences_per_index=int(counts.max(initial=0))))
    return result


def compare_traces(a,b):
    if len(a)!=3 or len(b)!=3:raise ValueError('three traced chunks required')
    rows=[]
    for chunk,(aa,bb) in enumerate(zip(a,b)):
        if len(aa)!=len(bb) or [x.name for x in aa]!=[x.name for x in bb]:
            raise ValueError('trace operation structure changed')
        diffs=[dict(tensor_diff(x,y),changed_axes=changed_axes(x,y) if x.data!=y.data else []) for x,y in zip(aa,bb)]
        changed=[x for x in diffs if not x['equal'] or not x.get('finite')]
        first=changed[0]['name'] if changed else None
        rows.append(dict(chunk=chunk,first_divergence=first,
                         first_layer=int(first.split('/')[1].removeprefix('layer')) if first else None,
                         first_op=first.split('/')[2] if first else None,
                         compared_tensors=len(diffs),differences=changed))
    return rows


class OpTrace:
    def __init__(self,lane=None):self.named=[];self.op=0;self.layer=-1;self.hc_calls=0;self.lane=lane
    def record(self,kind,value):
        import torch
        name=f'op{self.op:02d}/layer{self.layer}/{kind}'
        def save(x,path):
            if isinstance(x,torch.Tensor):self.named.append((path,x.clone()))
            elif isinstance(x,(tuple,list)):
                for i,v in enumerate(x):save(v,path+f'/{i}')
        save(value,name)
        self.op+=1
    def call(self,kind,fn):
        # The slice has two HC calls per layer, in fixed source order.
        if kind=='hc':self.layer=self.hc_calls//2;self.hc_calls+=1
        value=fn()
        if kind=='selection' and self.lane is not None:
            self.record('selection.indices',self.lane.indices)
        self.record(kind,value)
        return value


def traced_qsa(lane,x,i,trace):
    """Failure-only expansion of Slice.qsa, preserving its original arithmetic.

    This slice replaces the learned indexer with fixed synthetic indices.
    There are no indexer scores or top-k ties to diagnose here.
    """
    import torch
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention_staged
    if not lane.pre:raise ValueError('source trace is prefill-only')
    trace.record('qsa.input',x)
    projected=torch.mm(x,lane.w['qsa_in']);trace.record('qsa.projection',projected)
    q,gate=projected[:,:3072],projected[:,3584:]
    q=q.reshape(lane.t,12,256).contiguous();out=torch.empty_like(q)
    trace.record('qsa.indices',lane.indices)
    qsa_sparse_paged_attention_staged(q,lane.kv,lane.indices,lane.table[0],lane.rows,
                                    lane.pages-1,lane.arena,True,out)
    trace.record('qsa.attention',out)
    gated=out.flatten(1)*torch.sigmoid(gate);trace.record('qsa.gated',gated)
    result=torch.mm(gated,lane.w['qsa_out']);trace.record('qsa.output_projection',result)
    return result


@contextmanager
def traced_mm(lane,trace):
    """Solo diagnostic only: no decode thread runs during this wrapper."""
    import torch
    original=torch.mm
    weights={id(value):name for name,value in lane.w.items()}
    def mm(a,b,*args,**kwargs):
        name=weights.get(id(b),'unknown_weight')
        trace.record('mm.'+name+'.input',a)
        result=original(a,b,*args,**kwargs)
        trace.record('mm.'+name+'.output',result)
        return result
    torch.mm=mm
    try:yield
    finally:torch.mm=original


@contextmanager
def traced_gdn_outputs(lane,trace):
    """Observe original FLA/norm calls; do not duplicate their arithmetic."""
    from importlib import import_module
    fla=import_module('vllm.third_party.flash_linear_attention.ops')
    norm=import_module('vllm.third_party.flash_linear_attention.ops.layernorm_guard')
    original_chunk=fla.chunk_gated_delta_rule;original_norm=norm.layer_norm_fwd
    def chunk(*args,**kwargs):
        result=original_chunk(*args,**kwargs)
        trace.record('gdn.fla_output',result[0].reshape(lane.t,24,128))
        trace.record('gdn.final_state',result[1])
        return result
    def layernorm(*args,**kwargs):
        trace.record('gdn.pre_norm',args[0].reshape(lane.t,24,128))
        result=original_norm(*args,**kwargs)
        trace.record('gdn.post_norm',kwargs['out'].reshape(lane.t,24,128))
        return result
    fla.chunk_gated_delta_rule=chunk;norm.layer_norm_fwd=layernorm
    try:yield
    finally:fla.chunk_gated_delta_rule=original_chunk;norm.layer_norm_fwd=original_norm


def diagnose(pre,transport,original_kv,resident_kv,prefetch,stream,progress):
    import torch
    samples=[]
    # Separate 2x2 instrumented trajectories. Clones can change launch timing;
    # this diagnostic never replaces or clears the uninstrumented failed oracle.
    for trial in range(2):
        for kind in ('host','resident'):
            progress.update('terminal-resident-source-trace',trace_path=kind,trace_trial=trial)
            transport.drain();pre.kv=resident_kv if kind=='resident' else original_kv
            transport.prefetch=kind=='resident' and prefetch
            pre.reset();torch.cuda.synchronize();seq=[]
            for _ in range(3):
                trace=OpTrace(pre);original_qsa=pre.qsa
                pre.qsa=lambda x,i:traced_qsa(pre,x,i,trace)
                try:
                    with torch.cuda.stream(stream),traced_mm(pre,trace),traced_gdn_outputs(pre,trace):pre.forward(trace)
                finally:pre.qsa=original_qsa
                stream.synchronize();seq.append(capture(trace.named))
            samples.append((f'{kind}-{trial}',seq))
    comparisons=[dict(left=an,right=bn,chunks=compare_traces(a,b))
                 for (an,a),(bn,b) in combinations(samples,2)]
    return dict(completed=True,comparisons=comparisons,
        scope='failure-only instrumented 2 host + 2 resident trajectories; clone traffic may mask noise; cannot clear primary FAIL',
        granularity='outer operations, Python torch.mm input/output, GDN FLA output/final state/pre/post norm, QSA projection/indices/attention/gate/output; no native-kernel attribution',
        indexer='not executed by this slice; synthetic selected indices compared bitwise; no indexer scores exist',
        axes='axis 0 is token row; last axis is hidden column; intermediate axes (HC) retained; index samples only, no tensor values')
