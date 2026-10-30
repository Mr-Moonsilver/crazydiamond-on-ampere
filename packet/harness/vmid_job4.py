#!/usr/bin/env python3
"""Four serial QSA-only layers: resident reference vs compact sparse CPU DMA.

No model, engine, TP, GDN/HC/PLE/MoE, projection weights, or speed claim. Uses the
same pinned staged QSA entry point/head geometry as s5_segment_gpu.Slice.qsa.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time
import vmid_support as s
from vmid_qsa_dma import SOURCE_SHA, generate, BucketLease, Cancelled
from vmid_sparse_plan import (PAGE_ROWS,ROW_BYTES,ARENA_PAGES,MAX_PIECES,
                                make_plan,validate_source,cases)

LAYERS=4
QUERY_ROWS=80
TOPK=2051
BOOLS=('bytes_exact','tail_unchanged','output_exact','output_finite','completed')


def grade(r):
    f=list(r.get('failures',[]));m=list(r.get('missing',[]))
    if r.get('source_sha256')!=SOURCE_SHA:m.append('source_pin')
    if r.get('geometry')!=dict(layers=LAYERS,query_rows=QUERY_ROWS,query_heads=12,
            head_dim=256,topk=TOPK,page_rows=PAGE_ROWS,arena_pages=ARENA_PAGES,
            piece_pages=4,descriptor_cap=MAX_PIECES):m.append('geometry')
    expected=[(layer,case[0]) for case in cases() for layer in range(LAYERS)]
    got=r.get('cases',[])
    if [(x.get('layer'),x.get('name')) for x in got]!=expected:m.append('case_layer_coverage')
    specs={x[0]:x for x in cases()}
    for x in got:
        label=f"{x.get('name')}:layer{x.get('layer')}"
        spec=specs.get(x.get('name'))
        if spec is None: f.append(label+':unknown_case');continue
        plan=make_plan(spec[1],spec[2],(1,)*52,1)
        if x.get('selected_rows')!=len(spec[1]):f.append(label+':selection_count')
        if x.get('fallback')!=plan.fallback:f.append(label+':fallback')
        counts=dict(bucket_checks=len(plan.buckets),candidate_kernels=len(plan.buckets),
                    dma_pieces=sum(len(b.pieces) for b in plan.buckets),
                    bytes_checked=sum(b.rows for b in plan.buckets)*ROW_BYTES)
        for key,want in counts.items():
            if type(x.get(key)) is not int or x[key]!=want:m.append(label+':'+key)
        if plan.fallback:
            if x.get('reference_calls')!=0 or x.get('candidate_calls')!=0:f.append(label+':fallback_launched')
            if x.get('output_exact') is not None:f.append(label+':fallback_claims_output')
            continue
        if x.get('reference_calls')!=1 or x.get('candidate_calls')!=1:m.append(label+':attention_calls')
        for key in BOOLS:
            if x.get(key) is False:f.append(label+':'+key)
            elif x.get(key) is not True:m.append(label+':'+key)
    cancel=r.get('cancellation',[])
    if [x.get('layer') for x in cancel]!=list(range(LAYERS)):m.append('cancellation_layers')
    for x in cancel:
        for key in ('cancelled','partial_prefix_exact','partial_tail_poison','consumer_refused',
                    'reuse_refused','output_untouched','drained','refill_output_exact'):
            if x.get(key) is False:f.append(f"cancel{x.get('layer')}:{key}")
            elif x.get(key) is not True:m.append(f"cancel{x.get('layer')}:{key}")
        if x.get('kernels_before_refill')!=0 or x.get('published_before_refill')!=0:f.append('partial_bucket_published')
        if type(x.get('submitted_pieces')) is not int or not 0<x['submitted_pieces']<x.get('planned_pieces',0):
            m.append('mid_plan_cancellation')
    peaks=(r.get('allocator_peak_allocated'),r.get('allocator_peak_reserved'))
    if any(type(x) is not int or x<0 for x in peaks):m.append('allocator_peaks')
    elif not 0<peaks[0]<=peaks[1]<=1024**3:f.append('allocator_growth')
    if r.get('distinct_host_layers') is not True:m.append('independent_layers')
    if r.get('cleanup_ok') is False:f.append('cleanup')
    elif r.get('cleanup_ok') is not True:m.append('cleanup')
    return s.verdict(f,m)


def source_path():
    bundled=Path(__file__).parent/'vmid_sources/ops_qsa.py'
    return bundled if bundled.exists() else Path(__file__).resolve().parents[1]/'patches/retain/ops_qsa.py.v8.1'


def load_attention():
    source=source_path()
    if s.sha(source)!=SOURCE_SHA:raise ValueError('unpinned_qsa_source')
    # Import actual Triton definitions from the reviewed source, not image stock
    # selected by Python's module search path. This does not patch vLLM globally.
    from importlib.machinery import SourceFileLoader
    loader=SourceFileLoader('vmid_pinned_qsa',str(source))
    spec=importlib.util.spec_from_loader(loader.name,loader)
    module=importlib.util.module_from_spec(spec)
    sys.modules[loader.name]=module
    loader.exec_module(module)
    namespace=dict(module.__dict__)
    exec(compile(generate(source.read_text()),'<vm<vm-id>-sparse-dma>','exec'),namespace)
    return module,namespace


def selection(torch,selected):
    # Every selected token appears exactly once across the union. Padding is -1,
    # with the public per-query valid-count column; same TOPK as Slice.qsa.
    if len(selected)>QUERY_ROWS*TOPK:raise ValueError('selection_capacity')
    data=torch.full((QUERY_ROWS,TOPK+1),-1,dtype=torch.int32,device='cpu')
    for row in range(QUERY_ROWS):
        part=selected[row*TOPK:(row+1)*TOPK]
        if part:data[row,:len(part)]=torch.tensor(part,dtype=torch.int32,device='cpu')
        data[row,-1]=len(part)
    return data.to('cuda')


class Fill:
    def __init__(self,torch,host,arena,readback,plan,copy_stream,cancel=False):
        self.t=torch;self.host=host;self.arena=arena;self.readback=readback
        self.plan=plan;self.cs=copy_stream;self.cancel=cancel
        self.lease=BucketLease();self.done=None;self.kernels=0;self.checks=0;self.pieces=0
        self.bytes=0;self.bytes_exact=True;self.tail=True;self.aborted_event=None;self.submitted=[]

    def finish(self):
        if self.done is not None:
            self.done.synchronize();self.lease.complete(self.done);self.done=None

    def __call__(self,staged,selected,compact_start):
        t=self.t
        self.finish()
        validate_source(self.plan,self.plan.mapping,self.plan.versions,self.plan.epoch)
        bucket=self.plan.buckets[self.checks]
        if compact_start!=bucket.compact_start or tuple(selected.cpu().tolist())!=self.plan.selected[compact_start:compact_start+bucket.rows]:
            raise RuntimeError('stock_union_or_bucket_differs')
        self.lease.begin()
        flat=self.arena.view(-1,512);source=self.host.view(-1,512)
        # Cross-stream readiness includes any preceding reference attention.
        self.cs.wait_stream(t.cuda.current_stream())
        chosen=bucket.pieces[:max(1,len(bucket.pieces)//2)] if self.cancel else bucket.pieces
        with t.cuda.stream(self.cs):
            flat.fill_(-3)
            for p in chosen:
                flat[p.arena_row:p.arena_row+p.rows].copy_(source[p.source_row:p.source_row+p.rows],non_blocking=True)
            event=t.cuda.Event();event.record(self.cs)
        if self.cancel:
            self.submitted=list(chosen);self.aborted_event=event
            self.lease.abort(event)
            raise Cancelled('mid_plan_before_publication')
        t.cuda.current_stream().wait_event(event)
        self.readback.copy_(flat,non_blocking=True)
        checked=t.cuda.Event();checked.record();checked.synchronize()
        ids=t.tensor([self.plan.mapping[i//PAGE_ROWS]*PAGE_ROWS+i%PAGE_ROWS
                      for i in self.plan.selected[compact_start:compact_start+bucket.rows]],dtype=t.int64,device='cpu')
        expected=t.index_select(source,0,ids)
        exact=bool(t.equal(self.readback[:bucket.rows].view(t.uint8),expected.view(t.uint8)))
        tail=bool(t.all(self.readback[bucket.rows:]==-3).item())
        self.bytes_exact &= exact;self.tail &= tail
        if not exact or not tail:raise AssertionError('bucket_byte_oracle')
        self.lease.publish(event)
        self.checks+=1;self.pieces+=len(chosen);self.bytes+=bucket.rows*ROW_BYTES

    def kernel_proxy(self,kernel):
        owner=self
        class Proxy:
            def __getitem__(self,grid):
                def launch(*a,**kw):
                    owner.lease.consume()
                    result=kernel[grid](*a,**kw)
                    owner.kernels+=1
                    owner.done=owner.t.cuda.Event();owner.done.record()
                    return result
                return launch
        return Proxy()


def attention(torch,module,ns,fill,q,resident,indices,table,rows,arena,out):
    # Give each invocation a private globals dict. Never replace a live module's
    # kernel or gather, and never change reference arguments or arithmetic.
    import types
    globals_=dict(ns)
    globals_['_qsa_sparse_staged_gqa_kernel']=fill.kernel_proxy(module._qsa_sparse_staged_gqa_kernel)
    fn=types.FunctionType(ns['dma_attention'].__code__,globals_,argdefs=ns['dma_attention'].__defaults__)
    fn(q,resident,indices,table,rows,49,arena,True,out,dma_fill=fill)
    fill.finish()


def probe(outdir):
    r=dict(schema='vm<vm-id>-job4/1',image=s.IMAGE,source_sha256=SOURCE_SHA,cases=[],cancellation=[],failures=[],missing=[],
           geometry=dict(layers=LAYERS,query_rows=QUERY_ROWS,query_heads=12,head_dim=256,topk=TOPK,
                         page_rows=PAGE_ROWS,arena_pages=ARENA_PAGES,piece_pages=4,descriptor_cap=MAX_PIECES),
           scope='four serial QSA-only synthetic layers; actual pinned staged attention, MoE off; no full model, TP, engine leases, overlap or speed claim',
           started_wall_s=time.time())
    torch=None;ready=False
    try:
        import torch
        torch.cuda.set_per_process_memory_fraction(.125,0);ready=True
        torch.set_num_threads(4);torch.manual_seed(4004)
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)!=(8,6):raise RuntimeError('one_sm86_device_required')
        if os.environ.get('VLLM_QSA_KVO_SPARSE_PREFILL')!='1' or os.environ.get('QSA_STAGE_TIMING')!='0':raise RuntimeError('declared_sparse_mode_required')
        module,ns=load_attention()
        hosts=[torch.empty((52,1,PAGE_ROWS,512),dtype=torch.bfloat16,device='cpu',pin_memory=True).normal_(0,.03) for _ in range(LAYERS)]
        r['distinct_host_layers']=len({h.data_ptr() for h in hosts})==LAYERS and all(h.is_pinned() for h in hosts)
        resident=torch.empty_like(hosts[0],device='cuda')
        arena=torch.empty((ARENA_PAGES,1,PAGE_ROWS,512),dtype=torch.bfloat16,device='cuda')
        readback=torch.empty((ARENA_PAGES*PAGE_ROWS,512),dtype=torch.bfloat16,device='cpu',pin_memory=True)
        r['pinned_host_bytes']=sum(h.numel()*h.element_size() for h in hosts)+readback.numel()*2
        r['arena_bytes']=arena.numel()*2;r['resident_reference_bytes']=resident.numel()*2
        rows=torch.arange(QUERY_ROWS,dtype=torch.int64,device='cuda');cs=torch.cuda.Stream()
        seed=torch.randn((QUERY_ROWS,12,256),dtype=torch.bfloat16,device='cuda')*.03
        time.sleep(2)
        for spec in cases():
            name,selected,mapping,fallback=spec
            plan=make_plan(selected,mapping,(1,)*52,1)
            indices=selection(torch,selected);table=torch.tensor(mapping,dtype=torch.int32,device='cuda')
            qr,qc=seed.clone(),seed.clone()
            for layer,host in enumerate(hosts):
                item=dict(name=name,layer=layer,selected_rows=len(selected),fallback=plan.fallback,
                          reference_calls=0,candidate_calls=0,bucket_checks=0,candidate_kernels=0,dma_pieces=0,bytes_checked=0,output_exact=None)
                if not plan.fallback:
                    resident.copy_(host,non_blocking=True)
                    ref=torch.full_like(qr,-7);actual=torch.full_like(qc,-7)
                    module.qsa_sparse_paged_attention_staged(qr,resident,indices,table,rows,49,arena,True,ref)
                    item['reference_calls']=1
                    fill=Fill(torch,host,arena,readback,plan,cs)
                    attention(torch,module,ns,fill,qc,resident,indices,table,rows,arena,actual)
                    torch.cuda.synchronize()
                    item.update(candidate_calls=1,bucket_checks=fill.checks,candidate_kernels=fill.kernels,dma_pieces=fill.pieces,
                                bytes_checked=fill.bytes,bytes_exact=fill.bytes_exact,tail_unchanged=fill.tail,
                                output_exact=bool(torch.equal(ref.view(torch.uint8),actual.view(torch.uint8))),
                                output_finite=bool(torch.isfinite(ref).all() and torch.isfinite(actual).all()),completed=fill.lease.state=='free')
                    qr,qc=ref,actual  # separate four-layer chains, identical initial query only
                r['cases'].append(item);s.write(outdir/f'{name}-layer{layer}.json',item)
        # Cancel an incomplete first bucket before any attention, then drain and
        # refill the same arena fully. No attempt to erase already-issued DMA.
        spec=next(x for x in cases() if x[0]=='arena_25_pages_tail')
        _,selected,mapping,_=spec;plan=make_plan(selected,mapping,(1,)*52,1)
        indices=selection(torch,selected);table=torch.tensor(mapping,dtype=torch.int32,device='cuda')
        for layer,host in enumerate(hosts):
            resident.copy_(host,non_blocking=True)
            expected=torch.empty_like(seed)
            module.qsa_sparse_paged_attention_staged(seed,resident,indices,table,rows,49,arena,True,expected)
            actual=torch.full_like(seed,-7)
            fill=Fill(torch,host,arena,readback,plan,cs,cancel=True)
            rec=dict(layer=layer,cancelled=False)
            try:attention(torch,module,ns,fill,seed,resident,indices,table,rows,arena,actual)
            except Cancelled:rec['cancelled']=True
            for key,action in (('consumer_refused',fill.lease.consume),('reuse_refused',fill.lease.begin)):
                rec[key]=False
                try:action()
                except RuntimeError:rec[key]=True
            rec.update(kernels_before_refill=fill.kernels,published_before_refill=fill.lease.published,
                       submitted_pieces=len(fill.submitted),planned_pieces=len(plan.buckets[0].pieces))
            fill.aborted_event.synchronize();fill.lease.drain()
            rec['drained']=fill.lease.state=='free'
            snapshot=arena.view(-1,512).cpu()
            end=sum(p.rows for p in fill.submitted)
            logical=selected[:end]
            physical=torch.tensor([mapping[i//PAGE_ROWS]*PAGE_ROWS+i%PAGE_ROWS for i in logical],dtype=torch.int64,device='cpu')
            original=torch.index_select(host.view(-1,512),0,physical)
            rec['partial_prefix_exact']=bool(torch.equal(snapshot[:end].view(torch.uint8),original.view(torch.uint8)))
            rec['partial_tail_poison']=bool(torch.all(snapshot[end:]==-3).item())
            rec['output_untouched']=bool(torch.all(actual==-7).item())
            # Reuse the SAME lease/arena, not a replacement that hides stale bytes.
            fill.cancel=False
            attention(torch,module,ns,fill,seed,resident,indices,table,rows,arena,actual)
            torch.cuda.synchronize()
            rec['refill_output_exact']=bool(torch.equal(expected.view(torch.uint8),actual.view(torch.uint8)))
            r['cancellation'].append(rec);s.write(outdir/f'cancel-layer{layer}.json',rec)
            del snapshot,original,physical
        r['allocator_peak_allocated']=torch.cuda.max_memory_allocated(0)
        r['allocator_peak_reserved']=torch.cuda.max_memory_reserved(0)
        time.sleep(1)
    except Exception as exc:
        r['error_type']=type(exc).__name__
        if isinstance(exc,AssertionError) or (torch is not None and isinstance(exc,torch.cuda.OutOfMemoryError)):
            r['failures'].append('oracle_or_allocation_failure')
        else:r['missing'].append('probe_exception')
        # Stack locations (not tensors or payloads) make a failed first run actionable.
        import traceback
        r['error_stack']=[dict(file=Path(x.filename).name,line=x.lineno,function=x.name) for x in traceback.extract_tb(exc.__traceback__)]
    finally:
        try:
            if ready:torch.cuda.synchronize()
            r['cleanup_ok']=True
        except Exception:r['cleanup_ok']=False
        r['ended_wall_s']=time.time();r['grade']=grade(r);s.write(outdir/'probe.json',r)
    return r['grade']['rc']


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('probe','monitor','finalize'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--process-rc',type=int,default=2)
    a=p.parse_args()
    if a.mode=='monitor':return s.monitor(a.out,'flashnext-job4')
    if a.mode=='finalize':return s.finalize(a.out,a.process_rc,grade,'job4')
    return probe(a.out)

if __name__=='__main__':sys.exit(main())
