"""Repeated CPU-source DMA window control; imports CUDA only when installed.

One device window, a refill before EVERY QSA layer, including solo and oracle
trials. No engine code. Source remains the slice's synthetic shared host table.
"""
import math
import time


def validate(record, mode, pages, page_bytes, layers=6, batched=False):
    """Account every refill, including warm-up/start and post-decode guard work."""
    if record.get('batched',False) is not batched:raise ValueError('DMA fence policy mismatch')
    if batched and (record.get('max_refills_in_flight')!=1 or
                    record.get('max_outstanding_bytes')!=pages*page_bytes or
                    record.get('timing_basis')!='whole_refill_stream_span'):
        raise ValueError('unbounded batched DMA or wrong timing basis')
    steps = len(mode['raw_prefill'])
    if steps < 1 or pages < 1 or page_bytes < 1:raise ValueError('empty DMA workload')
    if [p['step'] for p in mode['raw_prefill']]!=list(range(steps)):
        raise ValueError('prefill steps not contiguous')
    entries = record['refills']
    expected = {(step, layer) for step in range(steps) for layer in range(layers)}
    keys = [(r['step'], r['layer']) for r in entries]
    if (len(keys) != len(expected) or set(keys) != expected or
            record['mode'] != 'every_qsa_layer' or record['layers'] != layers or
            record['windows'] != 1 or record['chunk_cap_pages'] != 4 or
            record['max_outstanding'] != ((pages+3)//4 if batched else 1)):
        raise ValueError('missing, duplicate or incorrectly scoped DMA refill')
    for r in entries:
        p = mode['raw_prefill'][r['step']]
        if (r['bytes'] != pages*page_bytes or r['chunks'] != (pages+3)//4 or
                not p['submitted_s'] <= r['start_s'] <= r['end_s'] <= p['observed_s'] or
                not math.isfinite(r['gpu_copy_ms']) or r['gpu_copy_ms'] <= 0):
            raise ValueError('unbounded, untimed or incomplete layer refill')
    ordered=sorted(entries,key=lambda r:(r['step'],r['layer']))
    if any(a['end_s']>b['start_s'] for a,b in zip(ordered,ordered[1:])):
        raise ValueError('shared-window refills overlap or run out of order')
    total = pages*page_bytes*steps*layers
    if (record['bytes'] != total or record['chunks'] != sum(r['chunks'] for r in entries) or
            not math.isclose(record['gpu_copy_ms'],sum(r['gpu_copy_ms'] for r in entries),rel_tol=1e-10)):
        raise ValueError('DMA totals disagree with layer evidence')
    d = mode.get('raw_decode', [])
    inside = [r for r in entries if len(d) >= 2 and
              r['start_s'] >= d[0]['observed_s'] and r['end_s'] <= d[-1]['submitted_s']]
    whole_chunks=sum(all(any(r['step']==step and r['layer']==layer for r in inside)
                         for layer in range(layers)) for step in range(steps))
    return dict(prefill_steps=steps,refill_count=len(entries),bytes=total,
                fully_inside_decode_refills=len(inside),
                fully_inside_decode_bytes=sum(r['bytes'] for r in inside),
                fully_inside_decode_chunks=whole_chunks,
                inside_decode_window=whole_chunks>0)


class WindowRefill:
    def __init__(self, pre, original_kv, resident_kv, batched=False):
        import torch
        if pre.spec['mix'] != ['full_attention']*6:
            raise ValueError('repeated window case requires six QSA layers')
        if pre.host.device.type != 'cpu' or not pre.host.is_pinned():
            raise ValueError('window refill source must be pinned CPU storage')
        if pre.host.shape != resident_kv.shape or not pre.host.is_contiguous():
            raise ValueError('window source shape/stride')
        self.torch=torch; self.pre=pre; self.original_kv=original_kv; self.resident=resident_kv
        self.batched=batched
        self.forward=pre.forward; self.qsa=pre.qsa
        self.stream=torch.cuda.Stream()
        self.begin=torch.cuda.Event(enable_timing=True); self.end=torch.cuda.Event(enable_timing=True)
        self.page_bytes=pre.host[0].numel()*pre.host.element_size()
        self.verify_layers=False; self.layer_checks=0
        self.reset_stats()
        pre.forward=self.run; pre.qsa=self.attend

    def reset_stats(self):
        self.step=-1; self.refills=[]

    def run(self, trace):
        self.step += 1
        return self.forward(trace)

    def attend(self, x, layer):
        # Host oracle uses exactly the original UVA path, without any DMA.
        if self.pre.kv is not self.resident:
            return self.qsa(x,layer)
        torch=self.torch
        start=time.perf_counter(); milliseconds=0.; chunks=0
        # The previous attention consumer must finish before overwriting this
        # one shared window. This dependency is lane-local; decode never waits.
        self.stream.wait_stream(torch.cuda.current_stream())
        if self.batched:
            with torch.cuda.stream(self.stream):self.begin.record()
        for lo in range(0,self.pre.pages,4):
            hi=min(lo+4,self.pre.pages)
            with torch.cuda.stream(self.stream):
                if not self.batched:self.begin.record()
                self.resident[lo:hi].copy_(self.pre.host[lo:hi],non_blocking=True)
                if not self.batched:self.end.record()
            if not self.batched:
                self.end.synchronize()  # one outstanding piece
                milliseconds += self.begin.elapsed_time(self.end)
            chunks += 1
        if self.batched:
            # One bounded refill, all disjoint pieces on the SAME stream.
            # CPU source and destination stay owned until this terminal fence.
            with torch.cuda.stream(self.stream):self.end.record()
            self.end.synchronize()
            milliseconds=self.begin.elapsed_time(self.end)
        torch.cuda.current_stream().wait_event(self.end)
        self.refills.append(dict(step=self.step,layer=layer,start_s=start,end_s=time.perf_counter(),
            bytes=self.pre.pages*self.page_bytes,chunks=chunks,gpu_copy_ms=milliseconds))
        actual=self.qsa(x,layer)
        if self.verify_layers:
            # Same exact x, selected indices, bucket order and weights. QSA has
            # no persistent writes in this slice. Readbacks are oracle-only.
            self.pre.kv=self.original_kv
            try: expected=self.qsa(x,layer)
            finally:self.pre.kv=self.resident
            if not torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)):
                from s5_segment_gpu import CorrectnessError
                raise CorrectnessError('DMA-window QSA output differs from UVA at layer '+str(layer))
            self.layer_checks += 1
        return actual

    def report(self):
        return dict(mode='every_qsa_layer',layers=6,windows=1,chunk_cap_pages=4,batched=self.batched,
            max_outstanding=(self.pre.pages+3)//4 if self.batched else 1,max_refills_in_flight=1,
            max_outstanding_bytes=(self.pre.pages if self.batched else min(4,self.pre.pages))*self.page_bytes,
            timing_basis='whole_refill_stream_span' if self.batched else 'sum_of_piece_stream_spans',
            bytes=sum(r['bytes'] for r in self.refills),chunks=sum(r['chunks'] for r in self.refills),
            gpu_copy_ms=sum(r['gpu_copy_ms'] for r in self.refills),refills=list(self.refills),
            scope='six-QSA slice, one synthetic host table re-read per layer; not the 24-layer engine')

    def close(self):
        # Drain before dropping closures/events, while green contexts are alive.
        self.stream.synchronize()
        self.torch.cuda.synchronize()
        self.pre.forward=self.forward; self.pre.qsa=self.qsa
        self.begin=None; self.end=None; self.stream=None
        self.forward=None; self.qsa=None; self.pre=None; self.resident=None; self.original_kv=None
