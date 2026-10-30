#!/usr/bin/env python3
"""VM-only device boundary proof with stock minimum64, unchanged live kernels.

Only query geometry and the fixture minimum differ from native14 GPU B. The
copied qsa_case body is checked against that immutable fixture in CPU tests.
Runtime methods/guards/thresholds/ready flags are not patched. No engine claim.
"""
import argparse,inspect,json,time,hashlib
from dataclasses import asdict
from pathlib import Path
import e3_native_sparse_waves_device_probe as base
from e3_native_sparse_waves_device_probe import (PAGE,WIDTH,PAGES,TOPK,FRONTIER,
    make_alignment,make_chunk,mapped_uva,bounded_poll,sha)
SCHEMA='native15-staged-boundaries-device/1'
CASE_ROWS=(64,65,103,967,992)
BASE_SHA='d78888eebf408cde0e9774a5fb8f43cbb5be30337a33aecfb6e5b097f508a254'

def qsa_case(route, *, rows, fragmented=False, cancel=False):
    import torch
    from vllm.models.qwen4_exp.nvidia import qsa
    from vllm.models.qwen4_exp.nvidia.ops import qsa as ops
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
    from vllm.v1.flash_e3 import (qsa_operation as qo, native_source as ns, wire_v1 as wire,
                                row1_oracle as ro, row1_exactness as ex)
    source = torch.empty((PAGES+1, 1, PAGE, WIDTH), dtype=torch.bfloat16,
                         device='cpu', pin_memory=True).normal_(0, .03)
    uva = mapped_uva(source)
    layer = torch.nn.Module()
    layer.layer_name, layer._qsa_kv_offload = 'probe.qsa', True
    layer._qsa_host_kv, layer.kv_cache = source, uva
    layer._qsa_stage_min_rows, layer._qsa_stage_arena_bytes = qsa._QSA_STAGE_MIN_ROWS, 24*PAGE*WIDTH*2
    f = make_alignment(block_size=PAGE, frontier=FRONTIER, tokens=rows, qsa_layer=layer)
    chunk = make_chunk(f, 9, seq=1 if route=='dense_dma' else 2)
    compute, copy = torch.cuda.current_stream(), torch.cuda.Stream()
    device = qo.NativeQsaDevice('cuda:0'); device.bind_streams(compute, copy)
    lane = f.et.LaneId(0, f.et.LaneRole.PREFILL, chunk.admission.state_epoch, 0)
    descriptor = wire.SourceLeaseDescriptor(chunk.admission.transaction, chunk.admission.state_epoch,
                                            tuple(range(1, PAGES+1)), (0,))
    # Only data owners here. bind_local_sources constructs the real allocator,
    # registry and holds; no fake lease or completion provider.
    from vllm.v1.flash_e3.lane_dispatch import LaneDispatcher
    def plan(layer_name, count):
        row = dict(binding.layer_tables)[layer_name]
        assert len(row) == count
        return row
    operation = qo.QsaPrefillOperation(chunk.admission.transaction, lane, device=device, plan_provider=plan)
    operation.state_epoch = chunk.admission.state_epoch
    owner = LaneDispatcher(f.runner, None, None, torch_module=torch)
    owner.operation = operation
    binding = ns.bind_local_sources(owner, chunk.admission, descriptor, rank=0)
    name = layer.layer_name
    store = ro.Row1RecordStore(capacity=64)
    operation.qualification = ro.Row1Qualifier(store, transaction=chunk.admission.transaction,
        lane=lane, backend=ex.TorchRow1Backend(), epochs=lambda: dict(worker_incarnation='device-path',
        engine_epoch=1, route_generation=1))
    operation.register_source(name, binding.sources[name], source_epoch=binding.source_epochs[name],
                              versions=binding.versions[name])
    operation.begin_forward(); operation.begin_layer(name, source); operation.producer_submitted(name, source)
    operation.attach_sparse_source(name, binding.holds[name])
    md = FlashAttentionMetadata(num_actual_tokens=rows, max_query_len=rows,
        query_start_loc=chunk.query_start_loc, max_seq_len=FRONTIER, seq_lens=chunk.seq_lens,
        block_table=chunk.block_tables[2], slot_mapping=torch.arange(rows, dtype=torch.int64, device='cuda:0'),
        use_cascade=False, common_prefix_len=0, cu_prefix_query_lens=None, prefix_kv_lens=None, suffix_kv_lens=None)
    # Exercise the full real generator's native10 rejection before any bind/submission.
    try:
        next(qsa._flash_s5_sparse_staged_operation(operation, layer, None, uva, None, md, rows, True, None))
    except RuntimeError as exc:
        assert str(exc) == 'S5 operation requires CPU host metadata; no device readback'
    else:
        raise AssertionError('native10 metadata admitted')
    host_plan = f.lc.attach_qsa_host_metadata({name:md}, (name,), chunk.query_start_loc_cpu,
                                             chunk.seq_lens_cpu, chunk.admission)
    count = FRONTIER if route=='dense_dma' else 4097 if fragmented else 1000
    packed = torch.full((rows, TOPK+1), -1, dtype=torch.int32, device='cpu')
    packed[:, -1] = 0
    for row, lo in enumerate(range(0, count, TOPK)):
        hi = min(lo+TOPK, count)
        packed[row, :hi-lo] = torch.arange(lo, hi, dtype=torch.int32) * (2 if fragmented else 1)
        packed[row, -1] = hi-lo
    selection = packed.to('cuda:0')
    query = torch.randn((rows, 12, 256), dtype=torch.bfloat16, device='cuda:0') * .03
    output = torch.full_like(query, -7)
    helper = qsa._flash_s5_sparse_staged_operation
    arena_fn = ops.qsa_get_staging_arena
    operation.attach_continuation(helper(operation, layer, query, uva, selection, md, rows, True, output))
    deadline = time.monotonic()+180
    waves = []
    while True:
        if time.monotonic()>deadline: raise TimeoutError('QSA advance deadline')
        try:
            marker = operation.advance()
            if getattr(marker, 'sparse_copy_pending', False):
                sparse = operation._layers[name].sparse
                hold = binding.holds[name]
                waves.append(dict(rows=sparse.wave_rows, descriptors=sparse.wave_descriptors,
                    published=hold.stats['published'], consumers=len(operation._layers[name].consumed)))
                assert hold.stats['published']==0 and not operation._layers[name].consumed
                assert sparse.wave_event is not None
                if cancel:
                    operation.cancel('GPU proof between sparse waves')
                    try: operation.advance()
                    except qo.OperationCancelled: pass
                    else: raise AssertionError('cancelled operation advanced')
                    bounded_poll(operation.drain)
                    proofs = operation.poll()
                    assert hold.stats['abandoned']==1 and hold.stats['published']==0
                    assert hold.stats['copied']==0 and hold.stats['consumed']==0
                    operation.release()
                    binding.close(owner.source_registry)
                    aborted = [x.as_dict() for x in store.layer_records()]
                    assert len(aborted)==1 and aborted[0]['passed'] is False and aborted[0]['aborted']
                    assert aborted[0]['hold_released'] is True
                    assert torch.equal(output, torch.full_like(output,-7))
                    assert owner.source_allocator.inventory()['live']==0
                    assert not owner.source_registry._active
                    return dict(verdict='PASS', cancelled=True, no_partial_publish=True,
                        no_attention_output=True, queried_drain=True, allocator_live=0,
                        source_registry_empty=True, advance_refused=True,
                        layer_records=aborted, waves=waves, completion_proofs=[asdict(x) for x in proofs],
                        operation_source_sha256=sha(qo.__file__), methods_unpatched=True,
                        geometry=dict(pages=PAGES, selected=count, row_step=2))
        except StopIteration: break
        except f.et.E3Pending: time.sleep(.001)
    operation.end_layer(name)
    # Same actual finish_state method, request writer and GPU align operation.
    before_finalize = [x.clone() for x in f.caches]
    assert chunk.finish_state() is True
    joined = torch.cuda.Event(); joined.record(copy); compute.wait_event(joined)
    state_event = torch.cuda.Event(); state_event.record(compute)
    operation.end_forward()
    bounded_poll(state_event.query)
    expected_state = [x.clone() for x in before_finalize]
    destination = chunk.admission.block_tables[0][PAGES-1]
    for x, old in zip(expected_state, before_finalize): x[destination].copy_(old[3])
    assert all(torch.equal(a,b) for a,b in zip(f.caches,expected_state))
    bounded_poll(operation.drain)
    proofs = operation.poll()
    operation.release()  # Actual release checks + Row1Qualifier.finish_layer.
    binding.close(owner.source_registry)
    records = [x.as_dict() for x in store.records()]
    layers = [x.as_dict() for x in store.layer_records()]
    failures = [x.as_dict() for x in store.failures()]
    assert records and len(layers)==1 and not failures
    assert all(x['equal'] is True for x in records)
    assert all(x['different_bytes'] is None for x in records)
    assert layers[0]['passed'] and layers[0]['hold_released'] and layers[0]['route']==route
    assert layers[0]['output_checks']==1 and layers[0]['bucket_checks'] == (2 if route=='dense_dma' else 1)
    assert owner.source_allocator.inventory()['live']==0 and not owner.source_registry._active
    assert all(x.event is not None and x.complete for x in operation._layers[name].consumed)
    assert helper is qsa._flash_s5_sparse_staged_operation and arena_fn is ops.qsa_get_staging_arena
    assert md._qsa_host_meta is host_plan
    assert f.req.num_computed_tokens.gpu[9].item()==FRONTIER
    assert all(torch.equal(a,b) for a,b in zip(f.ordinary, f.ordinary_before))
    return dict(verdict='PASS', route=route, kernel_executed=True, actual_helper=True,
        metadata_class=type(md).__module__+'.'+type(md).__name__, host_plan_bound=True,
        uva_pointer_verified=True, records=records, layer_records=layers, failure_records=failures,
        finish_state=True, finalized_state_bytes_exact=True, native10_negative_rejected=True, state_event_queried=True, copy_event_joined=True,
        operation_drained=True, operation_released=True, allocator_live=0,
        completion_proofs=[asdict(x) for x in proofs], completion_proof_count=len(proofs), source_registry_empty=True, ordinary_tables_unchanged=True,
        methods_unpatched=True, arena_provider_unpatched=True, source_sha256=sha(qsa.__file__),
        ops_sha256=sha(ops.__file__), backend_sha256=sha(inspect.getfile(ex.TorchRow1Backend)),
        fragmented=fragmented, waves=waves, sparse_stats=dict(operation.stats),
        operation_source_sha256=sha(qo.__file__),
        geometry=dict(pages=PAGES, selected=count, row_step=2 if fragmented else 1, page_rows=PAGE, width=WIDTH, rows=rows, heads=12, topk=TOPK,
                      arena_pages=24, pinned_host_bytes=source.numel()*source.element_size()))

def grade(record):
    from e3_native_row1_grade import _digest
    failures=[];missing=[]
    if record.get('schema')!=SCHEMA:failures.append('schema')
    for name in ('mocks_used','patched_methods','patched_kernels','engine_qualified','full_row1_qualified'):
        if record.get(name) is not False:missing.append('scope:'+name)
    if record.get('exceptions'):failures.extend('exception:'+k for k in record['exceptions'])
    if record.get('imports_verified') is not True:missing.append('mounted_imports')
    if record.get('minimum_rows')!=64:failures.append('stock_minimum')
    try:base.require_jit_mount(record.get('jit_mount') or {})
    except Exception:failures.append('jit_mount')
    expected={str(rows)+'_'+route for rows in CASE_ROWS for route in ('dense_dma','sparse_dma')}
    cases=record.get('cases') or {}
    if set(cases)!=expected:missing.append('case_coverage')
    for key,row in cases.items():
        if key not in expected:failures.append('foreign_case:'+key);continue
        if row.get('verdict')!='PASS':missing.append('case:'+key)
        for flag in ('kernel_executed','actual_helper','host_plan_bound','uva_pointer_verified',
            'finish_state','finalized_state_bytes_exact','state_event_queried','copy_event_joined',
            'operation_drained','operation_released','source_registry_empty','ordinary_tables_unchanged',
            'methods_unpatched','arena_provider_unpatched'):
            if row.get(flag) is not True:missing.append(key+':'+flag)
        if row.get('geometry',{}).get('rows')!=int(key.split('_',1)[0]):failures.append(key+':geometry')
        if row.get('allocator_live')!=0:missing.append(key+':allocator_live')
        comparisons=row.get('records') or []
        layers=row.get('layer_records') or []
        route=key.split('_',1)[1]
        expected_buckets=2 if route=='dense_dma' else 1
        buckets=[r for r in comparisons if r.get('operation') in ('dense_page_bucket','sparse_row_bucket')]
        outputs=[r for r in comparisons if r.get('operation')=='qsa_output_before_moe']
        if len(buckets)!=expected_buckets or len(outputs)!=1:missing.append(key+':raw_coverage')
        for raw in comparisons:
            if raw.get('digest')!=_digest(raw):failures.append(key+':comparison_digest')
            if raw.get('equal') is False:failures.append(key+':raw_comparisons')
            elif raw.get('equal') is not True:missing.append(key+':raw_comparisons')
            if type(raw.get('compared_bytes')) is not int or raw['compared_bytes']<=0:missing.append(key+':compared_bytes')
            if raw.get('route')!=route:failures.append(key+':comparison_route')
        for raw in outputs:
            if raw.get('finite') is not True:failures.append(key+':nonfinite_output')
            if raw.get('rows')!=int(key.split('_',1)[0]):missing.append(key+':output_rows')
        for raw in layers:
            if raw.get('digest')!=_digest(raw,layer=True):failures.append(key+':layer_digest')
        if (len(layers)!=1 or layers[0].get('passed') is not True
                or layers[0].get('hold_released') is not True or layers[0].get('route')!=route
                or layers[0].get('bucket_checks')!=expected_buckets or layers[0].get('output_checks')!=1):
            missing.append(key+':source_or_oracle')
        if row.get('failure_records'):failures.append(key+':source_or_oracle')
    if record.get('threshold_negatives')!=[1,7,32,63]:missing.append('native14_short_negatives')
    peak=record.get('gpu_allocator_peak_reserved')
    if type(peak) is not int or peak<0:missing.append('allocator_peak')
    elif peak>2*1024**3:failures.append('allocator_bound')
    verdict='FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS'
    return dict(verdict=verdict,rc={'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[verdict],failures=sorted(set(failures)),missing=sorted(set(missing)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=dict(schema=SCHEMA,cases={},exceptions={},imports_verified=False,minimum_rows=64,
        mocks_used=False,patched_methods=False,patched_kernels=False,engine_qualified=False,
        full_row1_qualified=False,synthetic_ownership_inputs=True,
        out_of_scope=['full model, natural selections, logits, TP/PP, engine row1 verdict',
            'green placement, speed, lifetime, leak and canary'])
    try:
        assert sha(base.__file__)==BASE_SHA
        result['jit_mount']=base.check_jit_mount();base.require_jit_mount(result['jit_mount'])
        result['imports']=base.verify_imports(json.loads(a.graph.read_text()));result['imports_verified']=True
        import torch
        from types import SimpleNamespace as NS
        from vllm.models.qwen4_exp.nvidia import qsa
        assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(8,6)
        torch.cuda.set_per_process_memory_fraction((2*1024**3)/torch.cuda.get_device_properties(0).total_memory,0)
        torch.set_num_threads(4);torch.manual_seed(15015)
        assert qsa._QSA_STAGE_MIN_ROWS==64
        result['threshold_negatives']=[]
        for rows in (1,7,32,63):
            try:next(qsa._flash_s5_sparse_staged_operation(None,
                NS(_qsa_kv_offload=True,_qsa_stage_min_rows=qsa._QSA_STAGE_MIN_ROWS),None,None,None,None,rows,True,None))
            except RuntimeError as exc:assert str(exc)=='S5 operation requires the staged dense row threshold'
            else:raise AssertionError('native14 short rows admitted')
            result['threshold_negatives'].append(rows)
        for rows in CASE_ROWS:
            for route in ('dense_dma','sparse_dma'):
                name=str(rows)+'_'+route;print('BEGIN '+name,flush=True)
                try:
                    result['cases'][name]=qsa_case(route,rows=rows)
                    print('END '+name+' PASS',flush=True)
                except Exception as exc:
                    result['exceptions'][name]=base.err(exc)
                    raise
        result['gpu_allocator_peak_allocated']=torch.cuda.max_memory_allocated()
        result['gpu_allocator_peak_reserved']=torch.cuda.max_memory_reserved()
    except Exception as exc:
        if not result['exceptions']:result['exceptions']['setup']=base.err(exc)
    result.update(grade(result));result['probe_sha256']=sha(__file__);result['graph_sha256']=sha(a.graph)
    with a.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:result[k] for k in ('verdict','rc','failures','missing')}),flush=True)
    return result['rc']


if __name__=='__main__':raise SystemExit(main())
