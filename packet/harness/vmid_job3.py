#!/usr/bin/env python3
"""Serial bounded CPU-source DMA, independent CPU byte oracle, synthetic only."""
import argparse
import json
from pathlib import Path
import sys
import time
import vmid_support as s
from vmid_sparse_plan import (PAGE_ROWS,ROW_BYTES,ARENA_PAGES,PIECE_PAGES,MAX_PIECES,
                                make_plan,validate_source,cases)


def grade(r):
    fails=list(r.get('failures',[]));missing=list(r.get('missing',[]))
    got=r.get('cases',[])
    if [x.get('name') for x in got] != [x[0] for x in cases()]:missing.append('case_coverage')
    declared={x[0]:(len(x[1]),x[3]) for x in cases()}
    for case in got:
        name=case.get('name','unknown')
        n=case.get('selected_rows')
        if (type(n) is not int or type(case.get('expected_fallback')) is not bool
                or (n,case.get('expected_fallback'))!=declared.get(name)):
            fails.append(name+':case_plan_mismatch')
            continue
        numbers=('dma_pieces','dma_bytes','bytes_checked','buckets',
                 'max_piece_bytes','max_source_page_span','max_bucket_pieces')
        if any(type(case.get(k)) is not int or case[k]<0 for k in numbers):
            missing.append(name+':numeric_evidence')
            continue
        if case['expected_fallback']:
            if (case.get('fallback')!='descriptor_cap' or any(case[k] for k in numbers)):
                fails.append(name+':unsafe_fallback')
            continue
        if case.get('fallback'):fails.append(name+':unexpected_fallback')
        for key in ('bytes_exact','tail_unchanged','copy_and_consumer_complete'):
            if case.get(key) is False:fails.append(name+':'+key)
            elif case.get(key) is not True:missing.append(name+':'+key)
        if case['bytes_checked']!=n*ROW_BYTES:missing.append(name+':byte_coverage')
        if case['dma_bytes']!=case['bytes_checked']:missing.append(name+':transfer_coverage')
        if case['buckets']!=(n+ARENA_PAGES*PAGE_ROWS-1)//(ARENA_PAGES*PAGE_ROWS):
            missing.append(name+':bucket_coverage')
        if n and (not case['buckets']<=case['dma_pieces']<=min(n,case['buckets']*MAX_PIECES)
                  or not ROW_BYTES<=case['max_piece_bytes']<=case['dma_bytes']
                  or case['max_piece_bytes']%ROW_BYTES
                  or case['max_source_page_span']<1
                  or not 1<=case['max_bucket_pieces']<=case['dma_pieces']):
            missing.append(name+':piece_coverage')
        if not n and any(case[k] for k in numbers):fails.append(name+':nonempty_empty_plan')
        if case['max_piece_bytes']>PIECE_PAGES*PAGE_ROWS*ROW_BYTES:fails.append(name+':piece_bound')
        if case['max_source_page_span']>PIECE_PAGES:fails.append(name+':source_page_span')
        if case['max_bucket_pieces']>MAX_PIECES:fails.append(name+':submission_bound')
    peaks=(r.get('allocator_peak_allocated'),r.get('allocator_peak_reserved'))
    if any(type(x) is not int or x<0 for x in peaks):missing.append('allocator_peaks')
    elif peaks[0]>peaks[1] or peaks[1]>256*1024**2:fails.append('unexpected_allocator_growth')
    if r.get('cleanup_ok') is False:fails.append('cleanup')
    elif r.get('cleanup_ok') is not True:missing.append('cleanup')
    if r.get('arena_bytes')!=ARENA_PAGES*PAGE_ROWS*ROW_BYTES:missing.append('arena_geometry')
    if (r.get('page_rows'),r.get('row_bytes'),r.get('arena_pages'),r.get('piece_pages'),r.get('descriptor_cap'))!=(PAGE_ROWS,ROW_BYTES,ARENA_PAGES,PIECE_PAGES,MAX_PIECES):
        missing.append('declared_geometry')
    return s.verdict(fails,missing)


def run_case(torch, host, arena, readback, copy_stream, consume_stream, spec):
    name,selected,mapping,expect_fallback=spec
    versions=(1,)*host.shape[0]
    plan=make_plan(selected,mapping,versions,1)
    r=dict(name=name,expected_fallback=expect_fallback,fallback=plan.fallback,
           selected_rows=len(selected),dma_pieces=0,dma_bytes=0,bytes_checked=0,buckets=0,
           max_piece_bytes=0,max_source_page_span=0,max_bucket_pieces=0,
           bytes_exact=True,tail_unchanged=True,copy_and_consumer_complete=True,
           unique_source_pages=len({mapping[t//PAGE_ROWS] for t in selected}))
    if plan.fallback:return r
    source=host.view(-1,ROW_BYTES)
    for bucket in plan.buckets:
        validate_source(plan,mapping,versions,1)
        with torch.cuda.stream(copy_stream):
            arena.fill_(165)  # poison padding, then replace only initialized compact rows
            for piece in bucket.pieces:
                # CPU pinned slices -> existing GPU arena. No GPU UVA gather.
                arena[piece.arena_row:piece.arena_row+piece.rows].copy_(
                    source[piece.source_row:piece.source_row+piece.rows],non_blocking=True)
            copied=torch.cuda.Event();copied.record(copy_stream)
        with torch.cuda.stream(consume_stream):
            consume_stream.wait_event(copied)
            # D2H reads the entire arena, including the untouched tail. This
            # consumer must complete before the next fill can reuse it.
            readback.copy_(arena,non_blocking=True)
            consumed=torch.cuda.Event();consumed.record(consume_stream)
        consumed.synchronize()
        r['copy_and_consumer_complete'] &= bool(copied.query() and consumed.query())
        # Independent oracle: direct logical selection -> CPU physical rows,
        # never consults the DMA piece list when constructing expected bytes.
        part=selected[bucket.compact_start:bucket.compact_start+bucket.rows]
        physical=torch.tensor([mapping[t//PAGE_ROWS]*PAGE_ROWS+t%PAGE_ROWS for t in part],dtype=torch.int64,device='cpu')
        expected=torch.index_select(source,0,physical)
        r['bytes_exact'] &= bool(torch.equal(readback[:bucket.rows],expected))
        r['tail_unchanged'] &= bool(torch.all(readback[bucket.rows:]==165).item())
        r['bytes_checked']+=bucket.rows*ROW_BYTES
        r['buckets']+=1;r['dma_pieces']+=len(bucket.pieces)
        r['max_bucket_pieces']=max(r['max_bucket_pieces'],len(bucket.pieces))
        for piece in bucket.pieces:
            r['dma_bytes']+=piece.rows*ROW_BYTES
            r['max_piece_bytes']=max(r['max_piece_bytes'],piece.rows*ROW_BYTES)
            span=(piece.source_row+piece.rows-1)//PAGE_ROWS-piece.source_row//PAGE_ROWS+1
            r['max_source_page_span']=max(r['max_source_page_span'],span)
        del expected,physical
    return r


def probe(out):
    r={'schema':'vm<vm-id>-job3/1','image':s.IMAGE,'cases':[],'failures':[],'missing':[],
       'scope':'serial synthetic sparse byte transport; no attention/MoE, engine leases, overlap or throughput claim',
       'page_rows':PAGE_ROWS,'row_bytes':ROW_BYTES,'arena_pages':ARENA_PAGES,
       'piece_pages':PIECE_PAGES,'descriptor_cap':MAX_PIECES,'started_wall_s':time.time()}
    torch=None;ready=False
    try:
        import torch
        torch.cuda.set_per_process_memory_fraction(0.125,0)
        ready=True
        torch.set_num_threads(4)
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)!=(8,6):
            raise RuntimeError('one_sm86_device_required')
        r['software']=dict(torch=str(torch.__version__),cuda=torch.version.cuda)
        # 52 physical pages, independent mapping (50 logical pages); host
        # allocation contains unique row IDs plus a nonzero byte pattern.
        host=torch.empty((52,PAGE_ROWS,ROW_BYTES),dtype=torch.uint8,device='cpu',pin_memory=True)
        flat=host.view(-1,ROW_BYTES)
        pattern=(torch.arange(ROW_BYTES,dtype=torch.int32,device='cpu')%251).to(torch.uint8)
        flat.copy_(pattern.expand_as(flat))
        ids=torch.arange(flat.shape[0],dtype=torch.int64,device='cpu')
        for byte in range(8):flat[:,byte]=((ids>>(8*byte))&255).to(torch.uint8)
        del ids,pattern
        arena=torch.empty((ARENA_PAGES*PAGE_ROWS,ROW_BYTES),dtype=torch.uint8,device='cuda')
        readback=torch.empty((ARENA_PAGES*PAGE_ROWS,ROW_BYTES),dtype=torch.uint8,device='cpu',pin_memory=True)
        if not host.is_pinned() or not readback.is_pinned():raise RuntimeError('unpinned_host')
        r['arena_bytes']=arena.numel()
        r['pinned_host_bytes']=host.numel()+readback.numel()
        cs,ds=torch.cuda.Stream(),torch.cuda.Stream()
        torch.cuda.synchronize()
        time.sleep(2)  # make native/allocator footprint visible to 1s sidecar
        for spec in cases():
            item=run_case(torch,host,arena,readback,cs,ds,spec)
            r['cases'].append(item)
            s.write(out/(spec[0]+'.json'),item)
            if not all(item.get(k,True) for k in ('bytes_exact','tail_unchanged','copy_and_consumer_complete')):break
        r['allocator_peak_reserved']=torch.cuda.max_memory_reserved(0)
        r['allocator_peak_allocated']=torch.cuda.max_memory_allocated(0)
        if r['allocator_peak_reserved']>256*1024**2:r['failures'].append('unexpected_allocator_growth')
        time.sleep(1)
    except Exception as exc:
        r['error_type']=type(exc).__name__
        if torch is not None and isinstance(exc,torch.cuda.OutOfMemoryError):
            r['failures'].append('cuda_out_of_memory')
        else:r['missing'].append('probe_exception')
    finally:
        try:
            if ready:torch.cuda.synchronize()
            r['cleanup_ok']=True
        except Exception:r['cleanup_ok']=False
        r['ended_wall_s']=time.time()
        r['grade']=grade(r)
        s.write(out/'probe.json',r)
    return r['grade']['rc']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('probe','monitor','finalize'),required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--process-rc',type=int,default=2)
    a=p.parse_args()
    if a.mode=='monitor':return s.monitor(a.out,'flashnext-job3')
    if a.mode=='finalize':return s.finalize(a.out,a.process_rc,grade,'job3')
    return probe(a.out)


if __name__=='__main__':sys.exit(main())
