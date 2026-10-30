#!/usr/bin/env python3
"""Owner-only two-stream selection skew, actual helper/oracle/lifetime proof.

Two logical TP peers share one GPU, so their post-agreement bucket execution is
serialized: the unmodified arena provider is per-device. Selection-copy streams
are independent and actually skewed. No full-engine or TP/NCCL evidence.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback
import e3_native_sparse_waves_device_probe as base
from e3_native_sparse_waves_device_probe import (PAGE, WIDTH, PAGES, TOPK, FRONTIER,
    make_alignment, make_chunk, mapped_uva, bounded_poll, sha)

SCHEMA = 'native18-selection-skew/1'


def host_shape():
    """Execute pre-device imports/contracts; device constructors stay explicit."""
    import ast
    import torch
    from vllm.models.qwen4_exp.nvidia import qsa
    from vllm.v1.flash_e3 import qsa_operation as qo, wire_v1 as wire, e3_types as et
    from vllm.v1.flash_e3.row1_exactness import TorchRow1Backend
    from vllm.v1.flash_e3.row1_oracle import Row1Qualifier, Row1RecordStore
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
    from vllm.v1.flash_e3.lane_dispatch import LaneDispatcher
    namespace = {}
    for function in (peer_program, skew_case, base.make_alignment, base.make_chunk, base.mapped_uva):
        for node in ast.walk(ast.parse(inspect.getsource(function))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<gpu-branch-import>', 'exec'), namespace)
    assert qsa._QSA_STAGE_MIN_ROWS == 64
    # Execute the unchanged alignment helper's non-device configuration
    # constructors too; importing them alone would miss a setup contract.
    from vllm.config import CacheConfig
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as Kind
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec, FullAttentionSpec
    from vllm.v1.flash_e3 import lane_chunk as lc
    specs = [MambaSpec(block_size=PAGE, shapes=((4, 4), (2, 4)),
                      dtypes=(torch.float32, torch.float32), mamba_type=Kind.GDN_ATTN,
                      mamba_cache_mode='align'),
             MambaSpec(block_size=PAGE, shapes=((4, 4),), dtypes=(torch.float32,),
                      mamba_type=Kind.SHORT_CONV, mamba_cache_mode='align')]
    groups = [KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)
              for name, spec in zip(('probe.gdn', 'probe.ple'), specs)]
    groups.append(KVCacheGroupSpec(layer_names=['probe.qsa'],
        kv_cache_spec=FullAttentionSpec(block_size=PAGE, num_kv_heads=1,
                                        head_size=256, dtype=torch.bfloat16)))
    config = KVCacheConfig(num_blocks=32, kv_cache_tensors=[], kv_cache_groups=groups)
    assert len(config.kv_cache_groups) == 3
    assert CacheConfig(mamba_cache_mode='align').mamba_cache_mode == 'align'
    copy_funcs = Qwen4ExpForConditionalGeneration.get_mamba_state_copy_funcs(
        {Kind.GDN_ATTN, Kind.SHORT_CONV})
    assert copy_funcs
    owners = torch.nn.ModuleList([torch.nn.Module() for _ in groups])
    assert len(owners) == 3
    ports = lc.PinnedPorts.load()
    assert callable(ports.mamba_groups)
    tx = et.TransactionId(1, 'cpu-shape', 1, 0, 64, 1)
    query = wire.QsaProfileQuery(tx, 1, 0, 0, 2, (0, 1))
    shape = wire.QsaScheduleShape('sparse_dma', (17,), (17,))
    observations = tuple(wire.QsaProfileObservation(query, rank, 'ready', shape,
                                                   'queried-selection-copy') for rank in (0, 1))
    verdict, grant = wire.qsa_profile_agreement(query, observations, (0, 1))
    assert verdict == 'agreed' and grant.shape == shape
    owner = LaneDispatcher(object(), None, None, torch_module=torch)
    device = qo.NativeQsaDevice('cuda:0')
    operation = qo.QsaPrefillOperation(tx, et.LaneId(0, et.LaneRole.PREFILL, 1, 0),
                                     device=device, plan_provider=lambda name, count: tuple(range(count)))
    store = Row1RecordStore(capacity=64)
    qualifier = Row1Qualifier(store, transaction=tx, lane=operation.lane,
                             backend=TorchRow1Backend(), epochs=lambda: dict(
                                 worker_incarnation='shape', engine_epoch=1, route_generation=1))
    for method in (device.bind_streams, device.query, operation.schedule_profile,
                   operation.authorize_schedule, operation.advance, operation.release,
                   qualifier.bucket, qualifier.finish_layer, owner.open,
                   torch.cuda.Stream, torch.cuda.Event, torch.cuda.stream,
                   torch.cuda._sleep, qsa._flash_s5_sparse_staged_operation):
        assert callable(method)
    md = FlashAttentionMetadata(num_actual_tokens=64, max_query_len=64,
        query_start_loc=torch.tensor([0, 64], dtype=torch.int32), max_seq_len=FRONTIER,
        seq_lens=torch.tensor([FRONTIER], dtype=torch.int32), block_table=torch.zeros((1, 32), dtype=torch.int32),
        slot_mapping=torch.arange(64, dtype=torch.int64), use_cascade=False,
        common_prefix_len=0, cu_prefix_query_lens=None, prefix_kv_lens=None, suffix_kv_lens=None)
    assert md.num_actual_tokens == 64 and not torch.cuda.is_initialized()
    import copy
    peer = json.loads((Path(__file__).parent/'fixtures/native18-skew-predicate.json').read_text())
    fixture = dict(schema=SCHEMA, imports_verified=True, exceptions={}, cases={})
    for ready, name in enumerate(('left_ready', 'right_ready')):
        fixture['cases'][name] = dict(observed_statuses=['ready' if r == ready else 'pending' for r in (0, 1)],
            no_grant_retries=1100, retries_no_submission=True, peers=[copy.deepcopy(peer), copy.deepcopy(peer)],
            compute_streams=[1, 2], copy_streams=[3, 4],
            queried_profile=dict(route='sparse_dma', bucket_submissions=[17], bucket_waves=[17]),
            shared_arena_bucket_execution_serialized=True, box_evidence=False)
    assert grade(fixture)['verdict'] == 'PASS', grade(fixture)
    broken = copy.deepcopy(fixture)
    broken['cases']['left_ready']['peers'][0]['records'][0]['digest'] = '0'*64
    assert grade(broken)['verdict'] == 'FAIL'
    broken = copy.deepcopy(fixture)
    broken['cases']['right_ready']['peers'][1]['operation_released'] = False
    assert grade(broken)['verdict'] != 'PASS'
    broken = copy.deepcopy(fixture)
    broken['cases']['right_ready']['observed_statuses'] = ['ready', 'ready']
    assert grade(broken)['verdict'] != 'PASS'
    return dict(verdict='PASS', cuda_initialized=False, predicate_synthetic_positive_and_negatives=True,
        executed=['all imports from GPU scenario and alignment constructors',
                  'actual stock minimum constant access', 'MambaSpec/FullAttentionSpec/KVCacheGroupSpec/KVCacheConfig/CacheConfig constructors',
                  'stock Mamba copy-function registry; torch Module/ModuleList; PinnedPorts.load',
                  'typed query/shape/observation/grant agreement',
                  'NativeQsaDevice, QsaPrefillOperation, LaneDispatcher constructors',
                  'Row1RecordStore, TorchRow1Backend, Row1Qualifier constructors',
                  'FlashAttentionMetadata constructor on CPU tensors',
                  'every directly used runtime method attribute accessed'],
        deferred=['CUDA streams/events/_sleep and allocations/pinning',
                  'RequestState, LaneTables, LaneAlign and LaneChunk construction requires CUDA allocation/pinning',
                  'UVA mapping and source registration', 'Triton kernels and device oracle',
                  'observed pending selection on each skew direction'],
        venue_preconditions=['one SM86 CUDA device', '3072 MiB cap / 2816 MiB stop / 2 GiB allocator',
                             'exec JIT tmpfs; writable HOME/cache', 'CUDA state observed, no virgin-context assertion'],
        box_evidence=False)

def peer_program(rank, *, delay_cycles=0):
    route, rows, fragmented, cancel = "sparse_dma", 64, True, False
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
    lane = f.et.LaneId(rank, f.et.LaneRole.PREFILL, chunk.admission.state_epoch, 0)
    descriptor = wire.SourceLeaseDescriptor(chunk.admission.transaction, chunk.admission.state_epoch,
                                            tuple(range(1, PAGES+1)), (0, 1))
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
    binding = ns.bind_local_sources(owner, chunk.admission, descriptor, rank=rank)
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
    yield dict(phase='constructed', operation=operation, compute=compute, copy=copy,
               transaction=chunk.admission.transaction, state_epoch=chunk.admission.state_epoch)
    if delay_cycles:
        with torch.cuda.stream(copy):
            torch.cuda._sleep(delay_cycles)
    boundary = operation.advance()
    assert getattr(boundary, 'qsa_schedule_boundary', False)
    assert operation.stats['sparse_buckets'] == 0
    yield dict(phase='boundary', operation=operation, marker=boundary)
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


class SkewUnavailable(RuntimeError):
    """A valid but non-deciding timing observation; never manufacture pending."""


def skew_case(ready_rank, note=lambda phase, **fields: None):
    import torch
    from vllm.v1.flash_e3 import wire_v1 as wire, e3_types as et
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    programs = [peer_program(rank, delay_cycles=0 if rank == ready_rank else 2_000_000_000)
                for rank in (0, 1)]
    constructed = []
    def resume(rank):
        with torch.cuda.stream(streams[rank]):
            return next(programs[rank])
    def profile(rank):
        with torch.cuda.stream(streams[rank]):
            return constructed[rank]['operation'].schedule_profile()
    for rank in (0, 1):
        row = resume(rank)
        assert row['phase'] == 'constructed'
        constructed.append(row)
        note('peer_constructed', rank=rank, compute_stream=int(row['compute'].cuda_stream),
             copy_stream=int(row['copy'].cuda_stream))
    assert constructed[0]['transaction'] == constructed[1]['transaction']
    assert streams[0].cuda_stream != streams[1].cuda_stream
    assert resume(ready_rank)['phase'] == 'boundary'
    bounded_poll(lambda: profile(ready_rank) is not None)
    delayed_rank = 1 - ready_rank
    assert resume(delayed_rank)['phase'] == 'boundary'
    query = wire.QsaProfileQuery(constructed[0]['transaction'], constructed[0]['state_epoch'],
                                 0, 0, 2, (0, 1))
    # op_seq=2 is an explicit host protocol input here; actual dispatcher
    # op_seq/descriptor retry and grant binding are covered by the CPU suite.
    observations = []
    for rank in (0, 1):
        shape = profile(rank)
        observations.append(wire.QsaProfileObservation(query, rank,
            'pending' if shape is None else 'ready', shape,
            None if shape is None else 'queried-selection-copy'))
    statuses = [row.status for row in observations]
    note('queried_selection_skew', statuses=statuses,
         profiles=[asdict(row.shape) if row.shape else None for row in observations])
    if statuses != ['ready' if rank == ready_rank else 'pending' for rank in (0, 1)]:
        raise SkewUnavailable('real selection events did not expose the requested skew: '+repr(statuses))
    assert wire.qsa_profile_agreement(query, tuple(observations), (0, 1)) == ('pending', None)
    ready_op = constructed[ready_rank]['operation']
    before = (dict(ready_op.stats), len(ready_op._records))
    for _ in range(1100):
        with torch.cuda.stream(streams[ready_rank]):
            try:
                ready_op.advance()
            except et.E3Pending:
                pass
            else:
                raise AssertionError('a ready peer advanced before TP agreement')
    assert (dict(ready_op.stats), len(ready_op._records)) == before
    note('pregrant_retries', ready_rank=ready_rank, attempts=1100, unchanged=True)
    bounded_poll(lambda: profile(delayed_rank) is not None)
    observations = tuple(wire.QsaProfileObservation(query, rank, 'ready', profile(rank),
                          'queried-selection-copy') for rank in (0, 1))
    verdict, grant = wire.qsa_profile_agreement(query, observations, (0, 1))
    assert verdict == 'agreed' and grant.shape.bucket_waves == (17,)
    note('profile_agreed', shape=asdict(grant.shape))
    for rank in (0, 1):
        with torch.cuda.stream(streams[rank]):
            constructed[rank]['operation'].authorize_schedule(grant)
    # The stock arena cache is per device. Serialize bucket ownership on this
    # single GPU, retaining both genuine selection streams and actual oracle.
    completed = []
    for rank in (0, 1):
        try:
            resume(rank)
        except StopIteration as end:
            completed.append(end.value)
            note('peer_numerical_and_lifetime', rank=rank, receipt=end.value)
        else:
            raise AssertionError('unexpected peer-program control yield')
    return dict(verdict='PASS', ready_rank=ready_rank, observed_statuses=statuses,
        compute_streams=[int(s.cuda_stream) for s in streams],
        copy_streams=[int(x['copy'].cuda_stream) for x in constructed],
        queried_profile=asdict(grant.shape), no_grant_retries=1100,
        retries_no_submission=True, peers=completed, shared_arena_bucket_execution_serialized=True,
        boundary_op_seq_synthetic=True, full_engine_constructed=False, box_evidence=False)


def grade(record):
    from e3_native_row1_grade import _digest
    failures, missing = [], []
    if record.get('schema') != SCHEMA:
        failures.append('schema')
    if record.get('exceptions'):
        failures.append('scenario_exception')
    if record.get('imports_verified') is not True:
        missing.append('mounted_imports')
    cases = record.get('cases') or {}
    if set(cases) != {'left_ready', 'right_ready'}:
        missing.append('both_skew_directions')
    for name, case in cases.items():
        if name not in ('left_ready', 'right_ready'):
            failures.append('unknown_case'); continue
        ready = 0 if name == 'left_ready' else 1
        if case.get('observed_statuses') != ['ready' if r == ready else 'pending' for r in (0, 1)]:
            missing.append(name+':observed_skew')
        if case.get('no_grant_retries') != 1100 or case.get('retries_no_submission') is not True:
            failures.append(name+':pregrant_submission')
        for field in ('compute_streams', 'copy_streams'):
            streams = case.get(field) or []
            if len(streams) != 2 or any(type(s) is not int or s <= 0 for s in streams) or len(set(streams)) != 2:
                missing.append(name+':'+field)
        profile = case.get('queried_profile') or {}
        if (profile.get('route') != 'sparse_dma'
                or tuple(profile.get('bucket_submissions') or ()) != (17,)
                or tuple(profile.get('bucket_waves') or ()) != (17,)):
            failures.append(name+':profile')
        if case.get('shared_arena_bucket_execution_serialized') is not True or case.get('box_evidence') is not False:
            failures.append(name+':single_card_scope')
        peers = case.get('peers') or []
        if len(peers) != 2:
            missing.append(name+':two_peers')
        for rank, peer in enumerate(peers):
            prefix = name+':'+str(rank)
            for field in ('kernel_executed', 'actual_helper', 'host_plan_bound', 'uva_pointer_verified',
                          'finish_state', 'finalized_state_bytes_exact', 'state_event_queried',
                          'copy_event_joined', 'operation_drained', 'operation_released',
                          'source_registry_empty', 'ordinary_tables_unchanged', 'methods_unpatched',
                          'arena_provider_unpatched'):
                if peer.get(field) is not True:
                    missing.append(prefix+':'+field)
            if peer.get('allocator_live') != 0 or peer.get('failure_records'):
                failures.append(prefix+':lifetime_or_oracle')
            comparisons = peer.get('records') or []
            if len(comparisons) != 2:
                missing.append(prefix+':comparison_coverage')
            for raw in comparisons:
                if raw.get('digest') != _digest(raw) or raw.get('equal') is not True:
                    failures.append(prefix+':raw_comparison')
                if not isinstance(raw.get('compared_bytes'), int) or raw['compared_bytes'] <= 0:
                    missing.append(prefix+':compared_bytes')
            buckets = [r for r in comparisons if r.get('operation') == 'sparse_row_bucket']
            outputs = [r for r in comparisons if r.get('operation') == 'qsa_output_before_moe']
            if len(buckets) != 1 or len(outputs) != 1:
                missing.append(prefix+':bucket_output')
            if outputs and (outputs[0].get('finite') is not True or outputs[0].get('rows') != 64):
                failures.append(prefix+':output')
            layers = peer.get('layer_records') or []
            if len(layers) != 1:
                missing.append(prefix+':layer_record')
            for raw in layers:
                if raw.get('digest') != _digest(raw, layer=True) or raw.get('passed') is not True or raw.get('hold_released') is not True:
                    failures.append(prefix+':layer_lifetime')
            stats = peer.get('sparse_stats') or {}
            if stats.get('copy_events') != 17 or stats.get('selection_copies') != 1:
                failures.append(prefix+':wave_counts')
    verdict = 'FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS'
    return dict(verdict=verdict, rc={'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[verdict],
                failures=sorted(set(failures)), missing=sorted(set(missing)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=('cpu-shape', 'gpu-all'), required=True)
    args = p.parse_args()
    record = dict(schema=SCHEMA, verdict='INCONCLUSIVE', rc=2, failure_class='NONE',
                  cases={}, progress=[], exceptions={}, imports_verified=False, box_evidence=False,
                  engine_constructed=False, gpu_execution=False)
    def persist():
        args.out.write_text(json.dumps(record, indent=2)+'\n')
    started = time.monotonic()
    try:
        record['imports'] = base.verify_imports(json.loads(args.graph.read_text()))
        record['imports_verified'] = True
        record['jit_mount'] = base.check_jit_mount()
        base.require_jit_mount(record['jit_mount'])
        persist()
        if args.mode == 'cpu-shape':
            record['host_shape'] = host_shape()
            record.update(verdict='PASS', rc=0)
        else:
            import torch
            assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (8, 6)
            torch.cuda.set_per_process_memory_fraction((2*1024**3)/torch.cuda.get_device_properties(0).total_memory, 0)
            torch.set_num_threads(4)
            torch.manual_seed(18018)
            for ready in (0, 1):
                name = 'left_ready' if ready == 0 else 'right_ready'
                print('BEGIN '+name, flush=True)
                def note(phase, **fields):
                    record['progress'].append(dict(case=name, phase=phase,
                                                   elapsed_s=time.monotonic()-started, **fields))
                    persist()
                record['cases'][name] = skew_case(ready, note)
                record['gpu_execution'] = True
                persist()
                print('END '+name+' PASS', flush=True)
            record['allocator_peak_reserved'] = torch.cuda.max_memory_reserved()
            assert record['allocator_peak_reserved'] <= 2*1024**3
            record.update(grade(record))
    except SkewUnavailable as exc:
        record.update(verdict='INCONCLUSIVE', rc=2, missing=[str(exc)], failure_class='NONE')
    except Exception as exc:
        record['exceptions']['scenario'] = dict(type=type(exc).__name__, message=str(exc),
                                                 chain=traceback.format_exc())
        record.update(verdict='FAIL', rc=1, failure_class='OPEN')
    record['wall_s'] = time.monotonic() - started
    persist()
    print(json.dumps({k: record.get(k) for k in ('verdict', 'rc', 'failure_class', 'wall_s')}), flush=True)
    return record['rc']


if __name__ == '__main__':
    raise SystemExit(main())
