"""Resident-source experiment helpers. GPU imports stay inside GPU entry points."""
import math
import os
import time

KINDS=('legacy','kv','ple_prefetch','kv_ple_prefetch','arena_direct')
CASES=('qsa-resident24','qsa-resident34','qsa-resident44','qsa-kv34','qsa-ple34','mixed-resident34','qsa-dma34')
CASE_CONFIG=dict(zip(CASES,((24,'qsa','kv_ple_prefetch'),(34,'qsa','kv_ple_prefetch'),
    (44,'qsa','kv_ple_prefetch'),(34,'qsa','kv'),(34,'qsa','ple_prefetch'),(34,'mixed','kv_ple_prefetch'),(34,'qsa','kv_ple_prefetch'))))
# Opt-in subset case only; do not change the old seven-case matrix.
CASE_CONFIG['dma-every-step34']=(34,'qsa','kv')
CASE_CONFIG['dma-batched34']=(34,'qsa','kv')
CASE_CONFIG['arena-direct34']=(34,'qsa','arena_direct')


def dma_window_proven(mode,step=1):
    d,p=mode['raw_decode'],mode['raw_prefill']
    if len(d)<2 or len(p)<=step:return False
    return p[step]['submitted_s']>=d[0]['observed_s'] and p[step]['observed_s']<=d[-1]['submitted_s']


def ledger(history, page_size, row_bytes, kv_bytes, ple_rows, ple_width, ple_itemsize,
           local_qsa_layers=6, slice_qsa_layers=4, retain_kv=True, prefetch_ple=True):
    if min(history,page_size,row_bytes,kv_bytes,ple_rows,ple_width,ple_itemsize,local_qsa_layers)<1:
        raise ValueError('invalid ledger dimensions')
    pages=(history+page_size-1)//page_size;page_bytes=page_size*row_bytes
    if kv_bytes!=pages*page_bytes:raise ValueError('slice KV allocation disagrees with shape')
    payload=1024*16*ple_width*ple_itemsize
    return dict(history=history,page_size=page_size,pages=pages,page_bytes=page_bytes,
        slice_qsa_layers=slice_qsa_layers,slice_unique_kv_sources=1,
        kv_source_bytes=kv_bytes,kv_added_bytes=kv_bytes if retain_kv else 0,
        ple_toy_host_bytes=ple_rows*ple_width*ple_itemsize,ple_chunk_bytes=payload,
        ple_pinned_buffer_bytes=payload,ple_prefetch_enabled=prefetch_ple,
        ple_transport_raw_device_bytes=payload if ple_itemsize==1 else 0,
        ple_added_device_buffer_bytes=0,ple_device_buffer_reused=True,
        expected_resident_allocation_delta=kv_bytes if retain_kv else 0,
        engine_fp8_ple_host_gib=48.6,engine_local_qsa_layers=local_qsa_layers,
        engine_full_kv_bytes=kv_bytes*local_qsa_layers,
        engine_s3_24_bytes=24*page_bytes*local_qsa_layers,
        engine_s3_40_bytes=40*page_bytes*local_qsa_layers,
        engine_kv_bytes_beyond_s3_24=max(0,pages-24)*page_bytes*local_qsa_layers,
        engine_kv_bytes_beyond_s3_40=max(0,pages-40)*page_bytes*local_qsa_layers,
        engine_staging_arena_bytes=23*page_bytes,
        scope='slice shares one KV source across QSA layers; engine six-layer values are calculated, not allocated')


class PLETransport:
    """Matched UVA vs CPU-packed contiguous H2D, including per-chunk packing.

    The slice keeps its BF16 collective. Byte transport/dequantization models
    FP8 PLE cost; it is not the engine's int8 ETP collective implementation.
    """
    def __init__(self,lane,itemsize=1):
        import torch
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        if itemsize not in (1,2):raise ValueError('PLE itemsize')
        self.lane=lane;self.itemsize=itemsize;self.prefetch=False;self.step=0
        self.calls=0;self.bytes=0;self.pack_seconds=0.;self.pending=False;self.closed=False
        self.base=lane.ple_base.cpu()  # One setup readback; no GPU IDs read in the window.
        if itemsize==1:
            host=torch.empty(lane.ple_host.shape,dtype=torch.uint8,pin_memory=True)
            host.copy_(lane.ple_host.to(torch.float8_e4m3fn).view(torch.uint8))
            lane.ple_host=host;lane.ple_uva=get_accelerator_view_from_cpu_tensor(host)
            self.raw=torch.empty(lane.ple_output.shape,dtype=torch.uint8,device='cuda')
        else:self.raw=lane.ple_output
        # Only prefill needs a pinned packing buffer. No whole-table device copy.
        self.packed=torch.empty(lane.ple_output.shape,dtype=lane.ple_host.dtype,pin_memory=True) if lane.pre else None
        self.copy_done=torch.cuda.Event() if lane.pre else None
        self.original_reset=lane.reset;self.original_ple=lane.ple
        lane.reset=self.reset;lane.ple=self.lookup

    def drain(self):
        if self.pending:self.copy_done.synchronize();self.pending=False

    def close(self):
        """After graph destruction, release pinned ownership with contexts alive.

        Dropping the caller's transport variable is insufficient: lane.reset and
        lane.ple retain this object, which retains the lane and its pinned table.
        Caller must fence BOTH green streams and destroy graphs first. The
        device fence here also covers setup/default-stream uses of these buffers.
        A failed fence leaves ownership intact; the outer watchdog owns hangs.
        """
        if self.closed:return
        import torch
        self.drain();torch.cuda.synchronize()
        lane=self.lane
        lane.reset=self.original_reset;lane.ple=self.original_ple
        lane.ple_uva=None;lane.ple_host=None
        self.packed=None;self.copy_done=None;self.raw=None;self.base=None
        self.original_reset=None;self.original_ple=None;self.lane=None
        self.closed=True

    def reset(self):
        self.drain();self.step=0;self.original_reset()

    def stats(self):return dict(calls=self.calls,h2d_bytes=self.bytes,pack_seconds=self.pack_seconds)

    def lookup(self):
        import torch
        from vllm.models.qwen4_exp.nvidia.ngram_embedding import _lookup_ple_embedding_from_pinned_kernel
        lane=self.lane
        if self.prefetch:
            if not lane.pre:raise RuntimeError('decode PLE prefetch forbidden')
            # The prior copy must release the pinned buffer before the CPU
            # overwrites it, including eager correctness loops without a step fence.
            self.drain();start=time.perf_counter()
            ids=(self.base+self.step*104729)%lane.ple_host.shape[0]
            torch.index_select(lane.ple_host,0,ids,out=self.packed)
            self.pack_seconds+=time.perf_counter()-start
            self.raw.copy_(self.packed,non_blocking=True)
            self.copy_done.record(torch.cuda.current_stream());self.pending=True
            self.calls+=1;self.bytes+=self.raw.numel()*self.raw.element_size()
        else:
            _lookup_ple_embedding_from_pinned_kernel[(lane.ple_ids.numel(),)](
                lane.ple_uva,lane.ple_ids,self.raw,160,0,lane.ple_host.shape[0],BLOCK_D=256)
        if self.itemsize==1:
            # Unit global scale in this synthetic table, identical in both paths.
            lane.ple_output.copy_(self.raw.view(torch.float8_e4m3fn).to(lane.dtype))
        self.step+=1
        return lane.ple_output


def validate_prefetch(stats,steps,payload):
    if stats['calls']!=steps or stats['h2d_bytes']!=steps*payload:
        raise ValueError('PLE prefetch not accounted once per step')
    if not math.isfinite(stats['pack_seconds']) or stats['pack_seconds']<0:raise ValueError('invalid pack time')


def run_control(dec,pre,transport,kind,g,green,sd,sp,ctl,progress,steps,phase,memory,result,dma=False,every_layer=False,batched=False):
    import torch
    import s5_segment_gpu as b
    from s5_terminal_mode import run as terminal_run
    from s5_terminal_correctness import verify
    retain_kv=kind in ('kv','kv_ple_prefetch');prefetch=kind in ('ple_prefetch','kv_ple_prefetch')
    phase('terminal-resident-setup');original_kv=pre.kv
    before=memory()
    resident_kv=(original_kv.view_as(original_kv) if kind=='arena_direct' else
                 pre.host.to('cuda') if retain_kv else original_kv)
    torch.cuda.synchronize();after=memory()
    if after['reserved']>18*2**30:raise b.Inconclusive('resident control exceeds18GiB')
    record=ledger(pre.history,pre.host.shape[2],pre.host[0,0,0].numel()*pre.host.element_size(),
        pre.host.numel()*pre.host.element_size(),pre.ple_host.shape[0],pre.ple_host.shape[1],transport.itemsize,
        slice_qsa_layers=pre.spec['mix'].count('full_attention'),retain_kv=retain_kv,prefetch_ple=prefetch)
    record.update(kind=kind,before=before,after=after,
        allocated_delta=after['allocated']-before['allocated'],reserved_delta=after['reserved']-before['reserved'],
        free_delta=after['free']-before['free'])
    result['resident_ledger']=record;result['resident_before']=before;result['resident_after']=after
    result['resident_bytes']=record['kv_added_bytes']
    if kind=='arena_direct':
        record.update(arena_reused=True,arena_allocated_bytes=pre.arena.numel()*pre.arena.element_size(),
                      added_kv_window_bytes=0,source='existing arena allocation',
                      oracle_gpu_input_bytes=pre.t*2560*2,
                      oracle_host_bucket_snapshots_bytes=2*23*record['page_bytes'])
    import json
    print('S5_MEMORY '+json.dumps(dict(rank=result['rank'],**record)),flush=True)
    window=None
    try:
        if kind=='arena_direct':
            from s5_arena_direct import ArenaDirect
            window=ArenaDirect(pre,original_kv,resident_kv)
            phase('terminal-arena-mapping')
            result['arena_mapping_exact']=window.probe_maps(green.sp)
            window.verify_layers=True
        elif every_layer:
            if not dma or kind!='kv':raise ValueError('every-layer DMA requires KV-only resident control')
            from s5_dma_window import WindowRefill
            window=WindowRefill(pre,original_kv,resident_kv,batched=batched)
            window.verify_layers=True
        from s5_substitution import run as substitution
        phase('terminal-resident-substitution',300)
        trials=int(os.environ.get('S5_SOURCE_TRIALS','3'))
        trace_on_fail=os.environ.get('S5_SOURCE_TRACE_ON_FAIL','0')
        if trials not in (3,6) or trace_on_fail not in ('0','1'):raise ValueError('unreviewed source oracle settings')
        check=substitution(pre,transport,original_kv,resident_kv,prefetch,green.sp,progress,trials)
        if window is not None:
            expected_checks=trials*3*6
            if window.layer_checks!=expected_checks:raise ValueError('missing per-layer exact DMA/UVA comparisons')
            check['dma_layer_exact']=dict(passed=True,checks=window.layer_checks,expected=expected_checks)
            if kind=='arena_direct':
                bucket_expected=expected_checks*((pre.pages+22)//23)
                if window.bucket_checks!=bucket_expected:raise ValueError('missing exact arena bucket checks')
                check['arena_bucket_exact']=dict(passed=True,checks=window.bucket_checks,expected=bucket_expected)
            window.verify_layers=False
        if os.environ.get('S5_MOE','on')=='off':
            check['deterministic_control']=all(p['output_equal'] for p in check['comparisons'])
            check['passed']=check['passed'] and check['deterministic_control']
        from s5_segment import write_json
        error_path=progress.path.with_name(progress.path.name.replace('.progress.json','.error.json'))
        # Preserve a local hard failure even if the peer never reaches the vote.
        if not check['passed']:
            write_json(error_path,dict(fatal_correctness=True,
                error='resident/prefetch substitution differs from matched host sequences',evidence=check))
        # Both ranks must enter diagnostic collectives, even when only one
        # rank fails locally. This CPU Gloo vote is outside timing windows.
        import torch.distributed as dist
        failed=torch.tensor([not check['passed']],dtype=torch.int32,device='cpu')
        dist.all_reduce(failed,op=dist.ReduceOp.MAX,group=ctl)
        any_failed=bool(failed.item())
        check['peer_or_local_failed']=any_failed
        result['resident_substitution']=check
        result['resident_source_equality']=check['passed']
        print('S5_SUBSTITUTION '+json.dumps(dict(rank=result['rank'],passed=check['passed'],
            all_states_and_ple_bitwise=check['all_states_and_ple_bitwise'],
            resident_matches_host_sequences=check['resident_matches_host_sequences'])),flush=True)
        if any_failed:
            if trace_on_fail=='1':
                from s5_substitution_trace import diagnose
                phase('terminal-resident-source-trace',300)
                from s5_failure import diagnostic
                result['resident_source_trace']=diagnostic(error_path,
                    dict(fatal_correctness=True,
                         error='resident/prefetch substitution differs from matched host sequences',evidence=check),
                    lambda:diagnose(pre,transport,original_kv,resident_kv,prefetch,green.sp,progress))
                check['diagnostic_trace']=result['resident_source_trace']
                for pair in result['resident_source_trace'].get('comparisons',[]):
                    for chunk in pair['chunks']:
                        print('S5_SOURCE_TRACE '+json.dumps(dict(rank=result['rank'],left=pair['left'],right=pair['right'],
                            chunk=chunk['chunk'],layer=chunk['first_layer'],op=chunk['first_op'],
                            first_divergence=chunk['first_divergence'])),flush=True)
                        boundary=chunk['layer3_boundary']
                        print('S5_LAYER3_BOUNDARY '+json.dumps(dict(rank=result['rank'],left=pair['left'],right=pair['right'],
                            chunk=chunk['chunk'],complete=boundary['complete'],all_equal=boundary['all_equal'],
                            **boundary['status'])),flush=True)
            error=b.CorrectnessError('resident/prefetch substitution differs from matched host sequences')
            error.evidence=check
            raise error
        phase('terminal-resident-prefill-solo')
        if window is not None:window.reset_stats()
        result['modes']['resident_prefill']=terminal_run(dec,pre,g,sd,sp,ctl,progress,steps,'prefill',False)
        if window is not None:
            if kind=='arena_direct':
                from s5_arena_direct import validate as validate_dma
                validation_args={}
            else:
                from s5_dma_window import validate as validate_dma
                validation_args=dict(batched=batched)
            solo_dma=window.report()
            solo_dma.update(validate_dma(solo_dma,result['modes']['resident_prefill'],pre.pages,record['page_bytes'],**validation_args))
            result['resident_dma_solo']=solo_dma
            window.reset_stats()
        phase('terminal-resident-overlap');start=transport.stats()
        original_forward=pre.forward;dma_stream=torch.cuda.Stream() if dma and not every_layer else None
        dma_events=[];forward_count=0
        def refill_forward(trace):
            nonlocal forward_count
            if forward_count==1:
                # The first prefill has already let decode start. Bound each
                # transfer to four contiguous pages and allow one outstanding.
                dma_stream.wait_stream(torch.cuda.current_stream())
                for lo in range(0,pre.pages,4):
                    hi=min(lo+4,pre.pages)
                    a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(dma_stream):
                        a.record();resident_kv[lo:hi].copy_(pre.host[lo:hi],non_blocking=True);z.record()
                    z.synchronize() # releases this bounded transfer before the next
                    dma_events.append((a,z,(hi-lo)*record['page_bytes']))
                torch.cuda.current_stream().wait_event(dma_events[-1][1])
            forward_count+=1
            return original_forward(trace)
        if dma and not every_layer:
            if not retain_kv:raise ValueError('DMA refill needs retained KV')
            pre.forward=refill_forward
        try:m=terminal_run(dec,pre,g,green.sd,green.sp,ctl,progress,steps,'overlap',False)
        finally:pre.forward=original_forward
        result['modes']['overlap_resident']=m
        if window is not None:
            repeated=window.report()
            repeated.update(validate_dma(repeated,m,pre.pages,record['page_bytes'],**validation_args))
            result['resident_dma']=repeated
        elif dma:
            result['resident_dma']=dict(bytes=sum(n for _,_,n in dma_events),chunks=len(dma_events),
                chunk_cap_pages=4,max_outstanding=1,gpu_copy_ms=sum(a.elapsed_time(z) for a,z,_ in dma_events),
                inside_decode_window=dma_window_proven(m),prefill_step=1,
                scope='forced full refill of prestaged values in step 1, DMA-only primary-context stream, no decode waits')
        end=transport.stats();stats={k:end[k]-start[k] for k in end};result['ple_prefetch_stats']=stats
        if prefetch:validate_prefetch(stats,len(m['raw_prefill']),record['ple_chunk_bytes'])
        elif stats['calls'] or stats['h2d_bytes']:raise ValueError('unexpected PLE prefetch')
        phase('terminal-resident-correctness')
        result['resident_correctness']=verify(dec,pre,g,green.sd,green.sp,(len(m['raw_decode']),len(m['raw_prefill'])),progress)
        record['peak_allocated_after']=torch.cuda.max_memory_allocated()
        record['peak_reserved_after']=torch.cuda.max_memory_reserved()
    finally:
        if window is not None:window.close()
        transport.drain();transport.prefetch=False;pre.kv=original_kv


def close_resident_resources(graphs,contexts,lanes,transports,progress):
    """Successful-run cleanup, outside all measurement windows. No best effort.

    The pinned allocator may record a CUDA event only when a tensor is freed.
    Thus even already-completed copies need their stream/context alive until
    pinned tensors are released AND the allocator's cached events are drained.
    Never destroy contexts after a failed fence or host-cache flush.
    """
    import gc
    import torch
    empty_host_cache=getattr(getattr(torch,'accelerator',None),'empty_host_cache',None)
    if not callable(empty_host_cache):raise RuntimeError('resident cleanup requires torch.accelerator.empty_host_cache')
    for context in contexts:
        context.sd.synchronize();context.sp.synchronize()
    torch.cuda.synchronize();progress.update('cleanup-streams-drained')
    for graph in reversed(graphs):graph.reset()
    progress.update('cleanup-graphs-released')
    for transport in transports:
        if transport is not None:transport.close()
    # The DMA control also uses the host KV allocation. Drop its UVA view first.
    for lane in lanes:
        lane.kv=None;lane.host=None;lane.ple_uva=None;lane.ple_host=None
    gc.collect();torch.cuda.synchronize()
    empty_host_cache();torch.cuda.synchronize()
    progress.update('cleanup-host-cache-drained')
    for context in contexts:context.close()
    progress.update('cleanup-contexts-destroyed')
    return dict(streams_drained=True,graphs_released=True,transports_closed=True,
                pinned_sources_released=True,host_cache_drained_before_contexts=True,
                contexts_destroyed=True)
