#!/usr/bin/env python3
"""VM-only real mounted-overlay device proof; no model, patched methods or kernels.

Synthetic ownership/tensor inputs are explicit. Imported production constructors,
methods, generators and kernels are called intact. This is not an engine or
multi-rank row1 certificate. The owner wrapper enforces this job's GPU budget.
"""
import argparse
import ctypes
from dataclasses import asdict
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace as NS

SCHEMA = 'native14-device-waves/1'
STOCK_MAMBA_SHA = '7382bcc8a7b070eaec44339379ebce485777fe23b97e05ee108088faa707c7e6'
TORCH_UTILS_SHA = 'eec8b1e3c7c673d3d171b92530f8b07acabdad5b0e3ab07cc0e717f827af8e5a'
PAGE, WIDTH, PAGES, ROWS, TOPK = 4048, 512, 25, 50, 2051
FRONTIER = PAGE * PAGES
SECTIONS = ('alignment', 'dense_dma', 'sparse_dma', 'sparse_fragmented', 'cancel_fragmented', 'refill_fragmented', 'backend_negative')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def err(exc):
    return dict(error_type=type(exc).__name__, error_message=str(exc)[:1024],
                stack=[dict(file=Path(x.filename).name, function=x.name, line=x.lineno)
                       for x in traceback.extract_tb(exc.__traceback__)[-12:]])


def bounded_poll(fn, seconds=30):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        value = fn()
        if value:
            return value
        time.sleep(.001)
    raise TimeoutError('queried completion deadline')


def verify_imports(graph):
    import sys
    assert graph['verdict'] == 'PASS'
    assert not any(p.endswith(('/e3_builder', '/e2_builder', '/s5_engine', '/e3', '/e2'))
                   for p in sys.path if p), 'builder path present'
    excluded = {x['module'] for x in graph['builder_only_proof']}
    rows = {x['module']: x for x in graph['modules']
            if x['active_mount'] and x['module'] not in excluded}
    result = []
    for name in dict.fromkeys(['vllm.models.qwen4_exp.nvidia.model'] + sorted(rows)):
        if name not in rows:
            continue
        mod = importlib.import_module(name)
        path = Path(mod.__file__).resolve()
        assert path == Path(rows[name]['destination']).resolve(), 'wrong module owner: '+name
        assert sha(path) == rows[name]['sha256'], 'wrong module bytes: '+name
        result.append(dict(module=name, sha256=sha(path), origin=str(path)))
    assert len(result) == 81
    return result


def make_alignment(*, block_size=4, frontier=8, tokens=4, qsa_layer=None):
    """Real lane constructors over explicitly synthetic cache/config owners."""
    import torch
    from vllm.config import CacheConfig
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as Kind
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec, FullAttentionSpec
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    from vllm.v1.worker.gpu.states import RequestState
    from vllm.v1.worker import mamba_utils as stock
    from vllm.v1.flash_e3 import lane_chunk as lc, e3_types as et
    assert sha(stock.__file__) == STOCK_MAMBA_SHA
    capacity, blocks = 10, 32
    device = torch.device('cuda:0')
    specs = [MambaSpec(block_size=block_size, shapes=((4, 4), (2, 4)),
                      dtypes=(torch.float32, torch.float32), mamba_type=Kind.GDN_ATTN,
                      mamba_cache_mode='align'),
             MambaSpec(block_size=block_size, shapes=((4, 4),), dtypes=(torch.float32,),
                       mamba_type=Kind.SHORT_CONV, mamba_cache_mode='align')]
    names = ['probe.gdn', 'probe.ple']
    owners, caches = {}, []
    for name, spec in zip(names, specs):
        owner = torch.nn.Module()
        owner.kv_cache = [torch.arange(blocks * math.prod(s), dtype=torch.float32,
                                      device=device).reshape((blocks,) + s) for s in spec.shapes]
        owners[name] = owner
        caches.extend(owner.kv_cache)
    groups = [KVCacheGroupSpec(layer_names=[n], kv_cache_spec=s) for n, s in zip(names, specs)]
    if qsa_layer is not None:
        groups.append(KVCacheGroupSpec(layer_names=[qsa_layer.layer_name],
            kv_cache_spec=FullAttentionSpec(block_size=PAGE, num_kv_heads=1, head_size=256,
                                           dtype=torch.bfloat16)))
    config = KVCacheConfig(num_blocks=blocks, kv_cache_tensors=[], kv_cache_groups=groups)
    # The full model cannot be constructed within the VM budget. These are
    # concrete ownership fields, not an initialized worker/readiness claim.
    state = MambaHybridModelState.__new__(MambaHybridModelState)
    state.device, state.max_num_reqs = device, capacity
    state._align_mode = CacheConfig(mamba_cache_mode='align').mamba_cache_mode == 'align'
    state._mamba_spec, state._mamba_group_ids = None, []
    state._mamba_state_copy_funcs = state._mamba_ctx = None
    state.model = Qwen4ExpForConditionalGeneration
    state.num_accepted_tokens_gpu = torch.ones(capacity, dtype=torch.int32, device=device)
    state._mamba_state_idx_gpu = torch.zeros(capacity, dtype=torch.int32, device=device)
    state._mamba_src_col_gpu = torch.zeros(capacity, dtype=torch.int32, device=device)
    state._mamba_src_off_gpu = torch.zeros(capacity, dtype=torch.int32, device=device)
    req = RequestState(capacity, frontier+8, tokens, 0, 128, device)
    # Real registration fills slots 9 through 0; no req-state API is replaced.
    for i in range(capacity):
        req.add_request('probe-'+str(i), frontier+1, [1]*(frontier+1), frontier-tokens, 1)
    req.apply_staged_writes()
    ordinary = [torch.arange(capacity*blocks, device=device, dtype=torch.int32).reshape(capacity, blocks)
                for _ in groups]
    sizes = [s.block_size for s in specs] + ([PAGE] if qsa_layer is not None else [])
    model = torch.nn.ModuleList(list(owners.values()) + ([qsa_layer] if qsa_layer is not None else []))
    runner = NS(device=device, model_state=state, req_states=req, model=model,
                kv_cache_config=config,
                vllm_config=NS(compilation_config=NS(static_forward_context=owners)),
                block_tables=NS(num_kv_cache_groups=len(groups), blocks_per_kv_block=[1]*len(groups),
                    block_sizes=sizes, kernel_block_sizes=sizes, _slot_mapping_enabled=[False]*len(groups),
                    block_tables=[NS(gpu=x) for x in ordinary]))
    ports = lc.PinnedPorts.load()  # Actual pinned allocators, including CPU pinning.
    tables = lc.LaneTables(runner, torch_module=torch)
    align = lc.LaneAlign(runner, ports, tables, torch_module=torch)
    assert align.ctx.is_initialized and align.request_slot_capacity == capacity
    assert align.ctx.num_accepted_tokens_out.shape == (capacity,)
    return NS(runner=runner, state=state, req=req, ports=ports, tables=tables, align=align,
              caches=caches, ordinary=ordinary, ordinary_before=[x.clone() for x in ordinary],
              config=config, stock=stock, lc=lc, et=et, frontier=frontier, tokens=tokens)


def make_chunk(f, slot, seq=1):
    import torch
    tx = f.et.TransactionId(1, 'device-path', 1, f.frontier-f.tokens, f.frontier, seq)
    # Non-identity physical table makes req-slot vs batch-row indexing visible.
    row = (3, 5, 1, 6, 2, 7, 4, 8) + tuple(range(9, 32))
    groups = [row, row]
    if len(f.tables.rows) == 3:
        groups.append(tuple(range(1, PAGES+1)))
    admission = f.lc.ChunkAdmission(tx, f.frontier+1, (1,)*f.tokens, (0, 0),
                                   tuple(groups), state_epoch=seq)
    return f.lc.LaneChunk(f.runner, f.ports, f.tables, f.align, admission, slot, torch_module=torch)


def alignment():
    import torch
    f = make_alignment()
    method = f.stock.MambaSpecDecodeGPUContext.run_fused_postprocess_align
    kernel = f.stock.postprocess_mamba_fused_kernel
    cases = []
    for slot in (0, 2, 9):
        for mode in ('copy', 'self', 'unaligned'):
            chunk = make_chunk(f, slot)
            for x in f.caches:
                x.copy_(torch.arange(x.numel(), device=x.device, dtype=x.dtype).reshape(x.shape))
            before = [x.clone() for x in f.caches]
            f.state.num_accepted_tokens_gpu.fill_(1)
            f.state._mamba_state_idx_gpu.fill_(0 if mode != 'self' else 1)
            f.req.num_computed_tokens.gpu.fill_(8 if mode != 'unaligned' else 7)
            accepted_before = f.state.num_accepted_tokens_gpu.clone()
            f.align.ctx.run_fused_postprocess_align(1, f.state.num_accepted_tokens_gpu,
                f.state._mamba_state_idx_gpu, f.req.num_computed_tokens.gpu, chunk.idx_mapping)
            event = torch.cuda.Event(); event.record(); bounded_poll(event.query)
            expected = [x.clone() for x in before]
            if mode == 'copy':
                for x, old in zip(expected, before):
                    x[5].copy_(old[3])
            assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a,b in zip(f.caches, expected))
            assert torch.equal(f.align.ctx.num_accepted_tokens_out, accepted_before)
            assert torch.equal(f.state.num_accepted_tokens_gpu, accepted_before)
            cases.append(dict(slot=slot, mode=mode, num_reqs=1, mapping_shape=[1],
                accepted_shape=[10], snapshot_shape=[10], kernel_executed=True,
                completion_queried=True, all_state_bytes_exact=True, accepted_exact=True))
    groups = f.ports.mamba_groups(f.config)
    funcs = f.state.model.get_mamba_state_copy_funcs({g.mamba_type for g in groups})
    legacy = f.stock.MambaSpecDecodeGPUContext.create(max_num_reqs=1, kv_cache_config=f.config,
        copy_funcs=funcs, device=f.runner.device, make_buffer=f.ports.make_buffer(f.runner.device))
    legacy.initialize_from_forward_context(f.config, f.runner.vllm_config.compilation_config.static_forward_context,
                                          funcs, f.tables.rows)
    try:
        legacy.run_fused_postprocess_align(1, f.state.num_accepted_tokens_gpu,
            f.state._mamba_state_idx_gpu, f.req.num_computed_tokens.gpu, chunk.idx_mapping)
    except RuntimeError as exc:
        assert 'broadcast shape [10]' in str(exc)
        negative = err(exc)
    else:
        raise AssertionError('native11 shape accepted')
    # Actual finish_state -> postprocess -> real RequestState staged write kernel.
    finals = []
    for slot in (0, 2, 9):
        chunk = make_chunk(f, slot)
        f.state._mamba_state_idx_gpu.fill_(0)
        f.state.num_accepted_tokens_gpu.fill_(2)
        f.req.num_computed_tokens.gpu.fill_(4)
        old = [x.clone() for x in f.caches]
        assert chunk.finish_state() is True
        event = torch.cuda.Event(); event.record(); bounded_poll(event.query)
        expected = [x.clone() for x in old]
        for x, prior in zip(expected, old): x[5].copy_(prior[3])
        assert all(torch.equal(a, b) for a,b in zip(f.caches, expected))
        computed = f.req.num_computed_tokens.gpu.cpu().tolist()
        accepted = f.state.num_accepted_tokens_gpu.cpu().tolist()
        assert computed == [8 if i == slot else 4 for i in range(10)]
        assert accepted == [1 if i == slot else 2 for i in range(10)]
        finals.append(dict(slot=slot, actual_finish_state=True, staged_write_exact=True,
                           state_bytes_exact=True, completion_queried=True))
    assert method is f.stock.MambaSpecDecodeGPUContext.run_fused_postprocess_align
    assert kernel is f.stock.postprocess_mamba_fused_kernel
    assert f.state._mamba_ctx is None and f.state._mamba_state_copy_funcs is None
    assert all(torch.equal(a,b) for a,b in zip(f.ordinary, f.ordinary_before))
    return dict(verdict='PASS', cases=cases, finish_state=finals, native11_negative=negative,
                legacy_rejected=True, methods_unpatched=True, kernels_unpatched=True,
                private_rows=1, request_slots=10, ordinary_tables_unchanged=True,
                ordinary_context_unchanged=True, stock_source_sha256=sha(f.stock.__file__),
                lane_source_sha256=sha(f.lc.__file__))


def mapped_uva(source):
    import vllm.utils.torch_utils as utils
    assert sha(utils.__file__) == TORCH_UTILS_SHA
    view = utils.get_accelerator_view_from_cpu_tensor(source)
    fn = ctypes.CDLL('libcudart.so.13').cudaHostGetDevicePointer
    fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    address = ctypes.c_void_p()
    assert fn(ctypes.byref(address), ctypes.c_void_p(source.data_ptr()), 0) == 0
    assert view.is_cuda and view.data_ptr() == address.value
    assert view.shape == source.shape and view.dtype == source.dtype and view.is_contiguous()
    return view


def qsa_case(route, *, fragmented=False, cancel=False):
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
    layer._qsa_stage_min_rows, layer._qsa_stage_arena_bytes = 1, 24*PAGE*WIDTH*2
    f = make_alignment(block_size=PAGE, frontier=FRONTIER, tokens=ROWS, qsa_layer=layer)
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
    md = FlashAttentionMetadata(num_actual_tokens=ROWS, max_query_len=ROWS,
        query_start_loc=chunk.query_start_loc, max_seq_len=FRONTIER, seq_lens=chunk.seq_lens,
        block_table=chunk.block_tables[2], slot_mapping=torch.arange(ROWS, dtype=torch.int64, device='cuda:0'),
        use_cascade=False, common_prefix_len=0, cu_prefix_query_lens=None, prefix_kv_lens=None, suffix_kv_lens=None)
    # Exercise the full real generator's native10 rejection before any bind/submission.
    try:
        next(qsa._flash_s5_sparse_staged_operation(operation, layer, None, uva, None, md, ROWS, True, None))
    except RuntimeError as exc:
        assert str(exc) == 'S5 operation requires CPU host metadata; no device readback'
    else:
        raise AssertionError('native10 metadata admitted')
    host_plan = f.lc.attach_qsa_host_metadata({name:md}, (name,), chunk.query_start_loc_cpu,
                                             chunk.seq_lens_cpu, chunk.admission)
    count = FRONTIER if route=='dense_dma' else 4097 if fragmented else 1000
    packed = torch.full((ROWS, TOPK+1), -1, dtype=torch.int32, device='cpu')
    packed[:, -1] = 0
    for row, lo in enumerate(range(0, count, TOPK)):
        hi = min(lo+TOPK, count)
        packed[row, :hi-lo] = torch.arange(lo, hi, dtype=torch.int32) * (2 if fragmented else 1)
        packed[row, -1] = hi-lo
    selection = packed.to('cuda:0')
    query = torch.randn((ROWS, 12, 256), dtype=torch.bfloat16, device='cuda:0') * .03
    output = torch.full_like(query, -7)
    helper = qsa._flash_s5_sparse_staged_operation
    arena_fn = ops.qsa_get_staging_arena
    operation.attach_continuation(helper(operation, layer, query, uva, selection, md, ROWS, True, output))
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
        geometry=dict(pages=PAGES, selected=count, row_step=2 if fragmented else 1, page_rows=PAGE, width=WIDTH, rows=ROWS, heads=12, topk=TOPK,
                      arena_pages=24, pinned_host_bytes=source.numel()*source.element_size()))


def backend_negative():
    """A real Row1Qualifier mismatch must call differing_bytes and refuse."""
    import torch
    from vllm.v1.flash_e3 import row1_oracle as ro, row1_exactness as ex, e3_types as et
    backend = ex.TorchRow1Backend()
    from vllm.v1.flash_e2.exactness import TorchExactBackend as FrozenBackend
    tx = et.TransactionId(1, 'negative', 1, 0, 1, 1)
    lane = et.LaneId(0, et.LaneRole.PREFILL, 1, 0)
    qualifier = ro.Row1Qualifier(ro.Row1RecordStore(capacity=8), transaction=tx, lane=lane,
        backend=backend, epochs=lambda:dict(worker_incarnation='device-path',engine_epoch=1,route_generation=1))
    try:
        ro.Row1Qualifier(ro.Row1RecordStore(capacity=8), transaction=tx, lane=lane,
                         backend=FrozenBackend(), epochs=lambda: {})
    except ro.Row1OracleError as exc:
        assert 'gather_rows' in str(exc)
    else: raise AssertionError('native9 frozen backend accepted')
    # Explicit raw data owner for the negative oracle only; not a source lease.
    uva = torch.arange(2*WIDTH, dtype=torch.float32, device='cuda:0').reshape(2, WIDTH)
    run = NS(layer_name='negative', state_epoch=1, allocation_epoch=1, source_epoch=1,
             mapping_version=1, uva=uva, row1_failed=None, sparse_hold=None)
    staged = torch.full_like(uva, -1)
    bucket = ro.Row1Bucket(index=0, route='sparse_dma', rows=2, pieces=1, compact_start=0,
                           nbytes=2*WIDTH*4, segments=((0,0,2),))
    try: qualifier.bucket(run, bucket, staged)
    except ro.Row1OracleError as exc:
        assert 'arena_bytes' in str(exc)
    else: raise AssertionError('device oracle mismatch accepted')
    records = [x.as_dict() for x in qualifier.store.records()]
    assert len(records)==1 and records[0]['equal'] is False and records[0]['different_bytes']>0
    assert backend.equal(staged, uva) and backend.differing_bytes(staged, uva)==0
    assert backend.finite(staged) and backend.nbytes(staged)==2*WIDTH*4
    changed = staged.clone(); changed[0,0]=123
    assert backend.differing_bytes(changed, uva)>0 and not backend.equal(changed, uva)
    return dict(verdict='PASS', refused=True, raw=records, device_differing_bytes=True, device_equal_finite_nbytes=True, native9_negative_rejected=True)


def grade(record):
    from e3_native_row1_grade import _digest
    failures, missing = [], []
    if record.get('schema') != SCHEMA: failures.append('schema')
    for key in ('mocks_used','patched_methods','patched_kernels','engine_qualified','full_row1_qualified'):
        if record.get(key) is not False: missing.append('scope:'+key)
    if record.get('exceptions'): failures.extend('exception:'+x for x in record['exceptions'])
    if record.get('imports_verified') is not True: missing.append('mounted_imports')
    mount = record.get('jit_mount')
    if not isinstance(mount, dict): missing.append('jit_mount')
    else:
        try: require_jit_mount(mount)
        except RuntimeError: failures.append('jit_mount_not_executable_writable')
    sections = record.get('sections') or {}
    for name in SECTIONS:
        row = sections.get(name)
        if not isinstance(row, dict) or row.get('verdict') != 'PASS': missing.append(name)
    if isinstance(sections.get('alignment'), dict):
        a = sections.get('alignment') or {}
        cases = a.get('cases') or []
        if {(r.get('slot'), r.get('mode')) for r in cases} != {(s,m) for s in (0,2,9) for m in ('copy','self','unaligned')}:
            missing.append('alignment_case_coverage')
        if len(cases)!=9 or not all(r.get('kernel_executed') is True and r.get('all_state_bytes_exact') is True
                                   and r.get('completion_queried') is True for r in cases): missing.append('alignment_device_evidence')
        if a.get('legacy_rejected') is not True: missing.append('native11_negative')
        finals=a.get('finish_state') or []
        if (len(finals)!=3 or {x.get('slot') for x in finals}!={0,2,9}
            or not all(all(x.get(k) is True for k in ('actual_finish_state','staged_write_exact','state_bytes_exact','completion_queried')) for x in finals)):
            missing.append('finish_state_slots')
        if a.get('methods_unpatched') is False or a.get('kernels_unpatched') is False:
            failures.append('alignment_patched')
        elif a.get('methods_unpatched') is not True or a.get('kernels_unpatched') is not True:
            missing.append('alignment_unpatched_evidence')
    if isinstance(sections.get('backend_negative'), dict):
        negative=sections.get('backend_negative') or {}
        if not all(negative.get(k) is True for k in ('refused','device_differing_bytes','native9_negative_rejected')):
            missing.append('backend_negative_evidence')
        if not any(x.get('equal') is False and type(x.get('different_bytes')) is int and x['different_bytes']>0 for x in negative.get('raw') or []):
            missing.append('backend_negative_raw')
    for name in ('dense_dma','sparse_dma','sparse_fragmented','refill_fragmented'):
        route = 'dense_dma' if name=='dense_dma' else 'sparse_dma'
        row = sections.get(name)
        if not isinstance(row, dict):
            continue  # Unrun section is missing, never evidence of a contradiction.
        for key in ('kernel_executed','actual_helper','host_plan_bound','uva_pointer_verified','finish_state',
                    'finalized_state_bytes_exact','native10_negative_rejected','state_event_queried','copy_event_joined','operation_drained','operation_released',
                    'source_registry_empty','ordinary_tables_unchanged','methods_unpatched','arena_provider_unpatched'):
            if row.get(key) is not True: missing.append(name+':'+key)
        layers = row.get('layer_records') or []
        comparisons = row.get('records') or []
        if not comparisons: missing.append(name+':raw_comparisons')
        elif any(x.get('equal') is False for x in comparisons): failures.append(name+':raw_comparisons')
        elif any(x.get('equal') is not True for x in comparisons): missing.append(name+':comparison_equality')
        bucket_rows=[x for x in comparisons if x.get('operation') in ('dense_page_bucket','sparse_row_bucket')]
        output_rows=[x for x in comparisons if x.get('operation')=='qsa_output_before_moe']
        if (len(bucket_rows)!=(2 if name=='dense_dma' else 1) or len(output_rows)!=1
            or not all(x.get('route')==route and x.get('finite') is True for x in output_rows)
            or not all(type(x.get('compared_bytes')) is int and x['compared_bytes']>0 for x in comparisons)):
            missing.append(name+':comparison_coverage')
        if len(layers)!=1 or not all(x.get('passed') is True and x.get('hold_released') is True and x.get('route')==route for x in layers):
            missing.append(name+':raw_layer')
        for raw in comparisons:
            if raw.get('digest') != _digest(raw): failures.append(name+':comparison_digest')
        for raw in layers:
            if raw.get('digest') != _digest(raw, layer=True): failures.append(name+':layer_digest')
        expected_buckets=2 if name=='dense_dma' else 1
        if not layers or layers[0].get('bucket_checks')!=expected_buckets or layers[0].get('output_checks')!=1:
            missing.append(name+':bucket_output_coverage')
        if row.get('allocator_live') is None: missing.append(name+':allocator_live')
        elif row['allocator_live'] != 0: failures.append(name+':source_or_oracle')
        if row.get('failure_records'): failures.append(name+':source_or_oracle')
    for name in ('sparse_fragmented','refill_fragmented'):
        row=sections.get(name) or {}
        waves=row.get('waves') or []
        stats=row.get('sparse_stats') or {}
        buckets=[r for r in row.get('records',[]) if r.get('operation')=='sparse_row_bucket']
        if (len(buckets)!=1 or buckets[0].get('rows')!=4097
                or buckets[0].get('pieces')!=4097
                or buckets[0].get('compared_bytes')!=4097*1024):
            missing.append(name+':fragmented_raw_coverage')
        if (row.get('fragmented') is not True or len(waves)!=16
            or [w.get('descriptors') for w in waves] != list(range(256,4097,256))
            or any(w.get('published')!=0 or w.get('consumers')!=0 for w in waves)
            or stats.get('sparse_descriptor_high_water')!=256
            or stats.get('sparse_descriptors')!=4097 or stats.get('sparse_buckets')!=1):
            missing.append(name+':bounded_waves')
    cancelled=sections.get('cancel_fragmented') or {}
    for key in ('cancelled','no_partial_publish','no_attention_output','queried_drain',
                'source_registry_empty','advance_refused','methods_unpatched'):
        if cancelled.get(key) is not True: missing.append('cancel_fragmented:'+key)
    if cancelled.get('allocator_live')!=0: missing.append('cancel_fragmented:allocator_live')
    aborted=cancelled.get('layer_records') or []
    if (len(aborted)!=1 or aborted[0].get('passed') is not False
            or not aborted[0].get('aborted') or aborted[0].get('hold_released') is not True):
        missing.append('cancel_fragmented:raw_abort')
    for raw in aborted:
        if raw.get('digest') != _digest(raw,layer=True):
            failures.append('cancel_fragmented:abort_digest')
    peak=record.get('gpu_allocator_peak_reserved')
    if type(peak) is not int or peak<0: missing.append('allocator_peak')
    elif peak>2*1024**3: failures.append('allocator_bound')
    verdict = 'FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS'
    return dict(verdict=verdict, rc={'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[verdict], failures=failures, missing=missing)



def check_jit_mount():
    """Record actual effective flags before importing code that initializes Triton."""
    flags = os.statvfs('/tmp').f_flag
    mount = next((line.split() for line in Path('/proc/mounts').read_text().splitlines()
                  if len(line.split()) >= 4 and line.split()[1] == '/tmp'), None)
    row = dict(path='/tmp', filesystem=mount[2] if mount else None,
               options=mount[3].split(',') if mount else [],
               noexec=bool(flags & os.ST_NOEXEC), readonly=bool(flags & os.ST_RDONLY))
    return row


def require_jit_mount(row):
    if (row.get('filesystem') != 'tmpfs' or row.get('noexec') is not False
            or row.get('readonly') is not False):
        raise RuntimeError('JIT scratch requires writable executable tmpfs')
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    result=dict(schema=SCHEMA, sections={}, exceptions={}, imports_verified=False,
        engine_qualified=False, full_row1_qualified=False, synthetic_ownership_inputs=True,
        mocks_used=False, patched_methods=False, patched_kernels=False,
        out_of_scope=['full model / natural QSA selections / logits', 'TP and PP collectives / sender-consumer retirement',
                      'engine scheduler hold / all-rank release and commit certificate',
                      'LaneForward suffix, dispatcher submit protocol and WorkerHandoffServer uninstall',
                      'row1 collective configure-disable packet / four-rank coverage',
                      'green placement accounting / overlap / speed / lifetime / canary'])
    try:
        result['jit_mount']=check_jit_mount()
        require_jit_mount(result['jit_mount'])
        result['imports']=verify_imports(json.loads(args.graph.read_text()));result['imports_verified']=True
        import torch
        assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(8,6)
        torch.cuda.set_per_process_memory_fraction((2*1024**3)/torch.cuda.get_device_properties(0).total_memory,0)
        torch.set_num_threads(4);torch.manual_seed(12015)
        result['device']=torch.cuda.get_device_name(0)
        for name, fn in [('alignment',alignment),('dense_dma',lambda:qsa_case('dense_dma')),
                         ('sparse_dma',lambda:qsa_case('sparse_dma')),
                         ('sparse_fragmented',lambda:qsa_case('sparse_dma',fragmented=True)),
                         ('cancel_fragmented',lambda:qsa_case('sparse_dma',fragmented=True,cancel=True)),
                         ('refill_fragmented',lambda:qsa_case('sparse_dma',fragmented=True)),
                         ('backend_negative',backend_negative)]:
            print('BEGIN '+name,flush=True)
            try:
                result['sections'][name]=fn()
                print('END '+name+' PASS',flush=True)
            except Exception as exc:
                result['exceptions'][name]=err(exc)
                print('END '+name+' '+type(exc).__name__,flush=True)
                break  # A device failure cannot be followed by trustworthy cases.
        result['gpu_allocator_peak_allocated']=torch.cuda.max_memory_allocated()
        result['gpu_allocator_peak_reserved']=torch.cuda.max_memory_reserved()
    except Exception as exc:
        result['exceptions']['setup']=err(exc)
    result.update(grade(result));result['probe_sha256']=sha(__file__);result['graph_sha256']=sha(args.graph)
    with args.out.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('verdict','rc','failures','missing')}),flush=True)
    return result['rc']

if __name__=='__main__':raise SystemExit(main())
