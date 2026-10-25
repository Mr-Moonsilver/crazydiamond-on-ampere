"""Dense CPU-DMA into the existing arena; offline source generator + slice hook.

No engine patch; no sparse path. CUDA imports are lazy. The helper changes only
source gathering, preserving the pinned attention function's bucket arithmetic.
"""
import ast
import hashlib
import inspect
import math
import textwrap
import time
import types
from s5_dma_window import WindowRefill

BASE_SHA='5a0d962022ec4e47809fa072d2b13745c84413de894048e67f10c3c10a45d51b'
NAME='qsa_sparse_paged_attention_staged'
GATHER='''            physical = block_table_row[page_lo:page_hi].to(torch.int64)
            torch.index_select(kv_cache, 0, physical, out=staged)'''


def generate(source):
    if hashlib.sha256(source.encode()).hexdigest()!=BASE_SHA:raise ValueError('unpinned QSA source')
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==NAME)
    fn=''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    if fn.count(GATHER)!=1:raise ValueError('dense gather anchor')
    fn=fn.replace('def '+NAME+'(', 'def flash_s5_arena_attention(',1)
    fn=fn.replace('    retained_copied: int = 0,\n','    retained_copied: int = 0,\n    *, dma_fill,\n',1)
    fn=fn.replace('    num_rows = rows.numel()\n', '''    if retained is not None or retained_copied != 0:
        raise RuntimeError("arena-direct rejects retained shortcut")
    if os.environ.get("VLLM_QSA_KVO_SPARSE_PREFILL", "1") != "0":
        raise RuntimeError("arena-direct is dense-only; sparse must not be silently disabled")
    num_rows = rows.numel()
''',1)
    fn=fn.replace(GATHER,'            dma_fill(staged, page_lo, page_hi, block_table_row)',1)
    ast.parse(fn);return fn


def slice_function(source):
    """Copy the running Slice.qsa, replacing its staged call only; no globals mutation."""
    tree=ast.parse(textwrap.dedent(source));found=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id==NAME:
            node.func.id='_s5_arena_staged';found.append(node)
    if len(found)!=1:raise ValueError('slice staged call count')
    return compile(ast.fix_missing_locations(tree),'<s5-arena-slice>','exec')


def copy_plan(table,lo,hi,pool_pages,arena_pages):
    if not (0<=lo<hi<=len(table) and hi-lo<=arena_pages):raise ValueError('bucket bounds')
    ids=table[lo:hi]
    if any(type(i) is not int or not 0<=i<pool_pages for i in ids):raise ValueError('physical page bounds')
    runs=[];j=0
    while j<len(ids):
        n=1
        while j+n<len(ids) and n<4 and ids[j+n]==ids[j]+n:n+=1
        runs.append((ids[j],j,n));j+=n
    return runs


def validate(record,mode,pages,page_bytes,layers=6):
    spans=[(lo,min(lo+23,pages)) for lo in range(0,pages,23)]
    rows=mode['raw_prefill'];entries=record['refills']
    expected=[(s,l,lo,hi) for s in range(len(rows)) for l in range(layers) for lo,hi in spans]
    if (not rows or [r['step'] for r in rows]!=list(range(len(rows))) or
        [(r['step'],r['layer'],r['page_lo'],r['page_hi']) for r in entries]!=expected or
        record['mode']!='every_qsa_bucket' or record['layers']!=layers or record['windows']!=0 or
        record['arena_pages']!=23 or record['chunk_cap_pages']!=4 or
        record['max_outstanding_bytes']!=min(23,pages)*page_bytes or
        record['max_buckets_in_flight']!=1 or record['timing_basis']!='whole_bucket_stream_span'):
        raise ValueError('arena DMA scope or bucket coverage')
    for r in entries:
        p=rows[r['step']];n=r['page_hi']-r['page_lo']
        if (r['bytes']!=n*page_bytes or r['chunks']!=(n+3)//4 or
            not p['submitted_s']<=r['start_s']<=r['end_s']<=p['observed_s'] or
            not math.isfinite(r['gpu_copy_ms']) or r['gpu_copy_ms']<=0):
            raise ValueError('arena copy bytes, timing or piece bound')
    if any(a['end_s']>b['start_s'] for a,b in zip(entries,entries[1:])):raise ValueError('arena reuse overlaps')
    total=layers*len(rows)*pages*page_bytes
    if (record['bytes']!=total or record['chunks']!=sum(r['chunks'] for r in entries) or
        not math.isclose(record['gpu_copy_ms'],sum(r['gpu_copy_ms'] for r in entries),rel_tol=1e-10)):
        raise ValueError('arena totals')
    d=mode.get('raw_decode',[])
    inside=[r for r in entries if len(d)>=2 and r['start_s']>=d[0]['observed_s'] and r['end_s']<=d[-1]['submitted_s']]
    chunks=sum(sum(r['step']==s for r in inside)==layers*len(spans) for s in range(len(rows)))
    return dict(prefill_steps=len(rows),refill_count=len(entries),fully_inside_decode_chunks=chunks,
        fully_inside_decode_refills=len(inside),fully_inside_decode_bytes=sum(r['bytes'] for r in inside),
        inside_decode_window=chunks>0)


class ArenaDirect(WindowRefill):
    def __init__(self,pre,original_kv,candidate_view):
        import torch
        import vllm.models.qwen4_exp.nvidia.ops.qsa as module
        if pre.arena.shape[0]!=23:raise ValueError('baseline arena must hold23 pages')
        self.arena_id=pre.arena.data_ptr();self.table_id=pre.table.data_ptr()
        self.mapping=pre.table[0].cpu().tolist() # setup-only authoritative mirror
        self.bucket_checks=0
        super().__init__(pre,original_kv,candidate_view)
        namespace=dict(module.__dict__)
        exec(compile(generate(inspect.getsource(module)),'<s5-arena-attention>','exec'),namespace)
        self.attention=namespace['flash_s5_arena_attention']
        def staged(*args,**kwargs):return self.attention(*args,**kwargs,dma_fill=self.fill)
        ns=dict(self.qsa.__func__.__globals__,_s5_arena_staged=staged)
        exec(slice_function(inspect.getsource(self.qsa.__func__)),ns)
        self.candidate_qsa=types.MethodType(ns[self.qsa.__func__.__name__],pre)

    def attend(self,x,layer):
        if self.pre.kv is not self.resident:return self.qsa(x,layer)
        self.layer=layer
        actual=self.candidate_qsa(x,layer)
        if self.verify_layers:
            self.pre.kv=self.original_kv
            try:expected=self.qsa(x,layer)
            finally:self.pre.kv=self.resident
            if not self.torch.equal(actual.view(self.torch.uint8),expected.view(self.torch.uint8)):
                from s5_segment_gpu import CorrectnessError
                raise CorrectnessError('arena-direct QSA output differs from UVA at layer '+str(layer))
            self.layer_checks+=1
        return actual

    def fill(self,staged,lo,hi,table):
        torch=self.torch
        if (staged.data_ptr()!=self.arena_id or self.pre.arena.data_ptr()!=self.arena_id or
            table.data_ptr()!=self.table_id or staged.shape[0]!=hi-lo):raise ValueError('arena/table ownership changed')
        runs=copy_plan(self.mapping,lo,hi,self.pre.host.shape[0],23)
        start=time.perf_counter()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.begin.record()
            for src,dst,n in runs:staged[dst:dst+n].copy_(self.pre.host[src:src+n],non_blocking=True)
            self.end.record()
        self.end.synchronize()
        torch.cuda.current_stream().wait_event(self.end)
        self.refills.append(dict(step=self.step,layer=self.layer,page_lo=lo,page_hi=hi,
            start_s=start,end_s=time.perf_counter(),bytes=(hi-lo)*self.page_bytes,chunks=len(runs),
            gpu_copy_ms=self.begin.elapsed_time(self.end)))
        if self.verify_layers:
            # Oracle only: copy candidate bytes to CPU, then perform the actual
            # original UVA gather into the SAME arena. No second GPU arena.
            # Attention sees the reference contents only after byte equality;
            # the measured path never runs this branch.
            actual=staged.cpu()
            physical=table[lo:hi].to(torch.int64)
            torch.index_select(self.original_kv,0,physical,out=staged)
            expected=staged.cpu()
            if not torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)):
                from s5_segment_gpu import CorrectnessError
                raise CorrectnessError(f'arena bucket bytes differ: layer{self.layer} pages{lo}:{hi}')
            self.bucket_checks+=1

    def probe_maps(self,stream):
        """Small GPU oracle, outside measurement: identity, fragmented, short tail."""
        torch=self.torch;pre=self.pre
        original_indices=pre.indices.cpu();original_table=pre.table.cpu();pages=pre.pages
        original_mapping=list(self.mapping);original_kv=pre.kv
        # Five MiB oracle-only contiguous projection input, not persistent storage.
        x=pre.x[:,:2560].contiguous();checks=[]
        try:
            self.verify_layers=True;pre.kv=self.resident
            for name,mapping,n in (('identity',list(range(pages)),pages),
                                   ('reverse',list(reversed(range(pages))),pages),
                                   ('fragmented_short',list(range(0,pages,2))+list(range(1,pages,2)),24)):
                pre.pages=n;self.mapping=mapping
                pre.table.copy_(torch.tensor([mapping],dtype=pre.table.dtype,device='cpu'))
                pre.indices.copy_(original_indices)
                pre.indices[:,:-1].remainder_(n*4048)
                pre.indices[:,0]=min(pre.history-1,n*4048-1)
                torch.cuda.synchronize()
                before=self.bucket_checks
                with torch.cuda.stream(stream):self.attend(x,0)
                stream.synchronize()
                checks.append(dict(name=name,pages=n,buckets=self.bucket_checks-before,passed=True))
        finally:
            pre.pages=pages;pre.kv=original_kv;self.mapping=original_mapping
            pre.table.copy_(original_table);pre.indices.copy_(original_indices)
            torch.cuda.synchronize();self.layer_checks=0;self.bucket_checks=0;self.reset_stats()
        if [c['buckets'] for c in checks]!=[3,3,2]:raise ValueError('mapping probe coverage')
        return dict(passed=True,cases=checks,checks=8,expected=8,
                    source='actual UVA gather vs CPU DMA, identical table and input, no output tolerance')

    def report(self):
        return dict(mode='every_qsa_bucket',layers=6,windows=0,arena_pages=23,chunk_cap_pages=4,
            max_buckets_in_flight=1,max_outstanding_bytes=min(23,self.pre.pages)*self.page_bytes,
            timing_basis='whole_bucket_stream_span',
            bytes=sum(r['bytes'] for r in self.refills),chunks=sum(r['chunks'] for r in self.refills),
            gpu_copy_ms=sum(r['gpu_copy_ms'] for r in self.refills),refills=list(self.refills),
            scope='six-QSA synthetic shared source; existing arena, no full-history device window')

    def close(self):
        super().close()
        self.attention=None;self.candidate_qsa=None;self.mapping=None


def check_rank(rank):
    """CPU-only gate; bind every proof to this case and both timing modes."""
    p=rank['value_probe'];m=rank['resident_ledger'];spec=rank['spec']
    if (rank['requested_split']!=34 or spec['layers']!=6 or spec['mix']!=['full_attention']*6 or
        not p['dma_every_layer'] or p.get('dma_batched',False) or p.get('resident_dma',False) or
        p['sparse_prefill']!='0' or m['kv_added_bytes']!=0 or m['added_kv_window_bytes']!=0 or
        m['expected_resident_allocation_delta']!=0 or m['arena_reused'] is not True or
        m['arena_allocated_bytes']!=23*m['page_bytes']):raise ValueError('arena case/ledger mismatch')
    check=rank['resident_substitution'];trials=p['source_trials'];expected=trials*3*6
    for name,n in (('dma_layer_exact',expected),('arena_bucket_exact',expected*((m['pages']+22)//23))):
        c=check[name]
        if c['passed'] is not True or c['checks']!=n or c['expected']!=n:raise ValueError('arena exactness count')
    c=rank['arena_mapping_exact']
    if (c['passed'] is not True or c['checks']!=8 or c['expected']!=8 or
        [(r['name'],r['pages'],r['buckets'],r['passed']) for r in c['cases']]!=
        [('identity',50,3,True),('reverse',50,3,True),('fragmented_short',24,2,True)]):
        raise ValueError('arena mapping/short-tail evidence')
    proof=validate(rank['resident_dma'],rank['modes']['overlap_resident'],m['pages'],m['page_bytes'])
    validate(rank['resident_dma_solo'],rank['modes']['resident_prefill'],m['pages'],m['page_bytes'])
    if proof['inside_decode_window']!=rank['resident_dma']['inside_decode_window']:
        raise ValueError('arena inside-window claim')
    return proof
