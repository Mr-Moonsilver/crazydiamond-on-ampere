#!/usr/bin/env python3
"""Actual pinned Mamba snapshot path on CPU; intercept before GPU launch.

Imports use the actual generated overlay layout. Descriptor/owner inputs and
request-slot tensors are small, explicit CPU test data, not a loaded model.
No module, method, kernel or readiness flag is replaced. The sole allocator
adaptation disables pinning on actual CpuGpuBuffer CPU storage (no GPU nodes).
"""
import argparse
import ast
import functools
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace as NS

from e3_native_import_probe import check

STOCK_MAMBA_SHA = '7382bcc8a7b070eaec44339379ebce485777fe23b97e05ee108088faa707c7e6'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BeforeKernel(Exception):
    """Explicit proof boundary, not success from the GPU kernel."""


def launch_line(method):
    source, first = inspect.getsourcelines(method)
    hits = [first + i for i, line in enumerate(source)
            if line.strip().startswith('postprocess_mamba_fused_kernel[grid](')]
    assert len(hits) == 1, 'stock launch boundary changed'
    return hits[0]


def invoke_to_kernel_boundary(ctx, accepted, state_indices, computed, mapping):
    """Call the real method. Trace only to stop before its real GPU launch."""
    method = type(ctx).run_fused_postprocess_align
    boundary = launch_line(method)
    observed = {}
    previous = sys.gettrace()

    def trace(frame, event, arg):
        if frame.f_code is method.__code__ and event == 'line' and frame.f_lineno == boundary:
            local = frame.f_locals
            assert local['num_reqs'] == 1
            assert local['num_accepted_tokens_gpu'] is accepted
            assert local['state_idx_gpu'] is state_indices
            assert local['new_num_computed_tokens_gpu'] is computed
            assert local['idx_mapping'] is mapping
            assert local['num_accepted_tokens_snapshot'] is ctx.num_accepted_tokens_out
            observed.update(num_reqs=1, snapshot_shape=list(ctx.num_accepted_tokens_out.shape),
                            accepted_shape=list(accepted.shape), state_shape=list(state_indices.shape),
                            computed_shape=list(computed.shape), mapping_shape=list(mapping.shape),
                            mapping_values=mapping.tolist(), grid=list(local['grid']),
                            actual_method_entered=True, actual_snapshot_copy_executed=True,
                            kernel_launch_intercepted=True, kernel_executed=False)
            raise BeforeKernel()
        return trace

    try:
        sys.settrace(trace)
        ctx.run_fused_postprocess_align(1, accepted, state_indices, computed, mapping)
    except BeforeKernel:
        assert observed
    finally:
        sys.settrace(previous)
    assert observed, 'method returned before snapshot/launch boundary'
    return observed


def collect():
    import torch
    from vllm.config import CacheConfig
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as Kind
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
    from vllm.v1.utils import CpuGpuBuffer
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    from vllm.v1.worker import mamba_utils as stock
    from vllm.v1.flash_e3 import lane_chunk as lc

    assert sha(stock.__file__) == STOCK_MAMBA_SHA, 'stock Mamba source pin'
    original_method = stock.MambaSpecDecodeGPUContext.run_fused_postprocess_align
    original_kernel = stock.postprocess_mamba_fused_kernel
    device = torch.device('cpu')
    capacity = 10
    cache_config = CacheConfig(mamba_cache_mode='align')
    specs = (
        MambaSpec(block_size=4, shapes=((4, 4), (2, 4)),
                  dtypes=(torch.float32, torch.float32), mamba_type=Kind.GDN_ATTN,
                  mamba_cache_mode='align'),
        MambaSpec(block_size=4, shapes=((4, 4),), dtypes=(torch.float32,),
                  mamba_type=Kind.SHORT_CONV, mamba_cache_mode='align'),
    )
    names = ('cpu.gdn', 'cpu.ple')
    config = KVCacheConfig(num_blocks=8, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)
        for name, spec in zip(names, specs)])
    owners = {}
    caches = []
    for name, spec in zip(names, specs):
        owner = torch.nn.Module()
        owner.kv_cache = [torch.arange(8 * __import__('math').prod(shape), dtype=torch.float32,
                                      device=device).reshape((8,) + shape)
                          for shape in spec.shapes]
        owners[name] = owner
        caches.extend(owner.kv_cache)
    cache_before = [x.clone() for x in caches]
    # Actual class; only the CPU ownership fields used by LaneAlign are
    # populated, explicitly. No model/engine constructor or startup claim.
    state = MambaHybridModelState.__new__(MambaHybridModelState)
    state.device = device
    state.max_num_reqs = capacity
    state._align_mode = cache_config.mamba_cache_mode == 'align'
    state._mamba_spec = None
    state._mamba_group_ids = []
    state._mamba_state_copy_funcs = None
    state._mamba_ctx = None
    state.model = Qwen4ExpForConditionalGeneration  # real classmethod, no model instance
    state.num_accepted_tokens_gpu = torch.arange(1, capacity + 1, dtype=torch.int32, device=device)
    state._mamba_state_idx_gpu = torch.ones(capacity, dtype=torch.int32, device=device)
    state._mamba_src_col_gpu = torch.ones(capacity, dtype=torch.int32, device=device)
    state._mamba_src_off_gpu = torch.zeros(capacity, dtype=torch.int32, device=device)
    computed = torch.full((capacity,), 8, dtype=torch.int32, device=device)
    ordinary_tables = [torch.arange(capacity * 8, dtype=torch.int32, device=device).reshape(capacity, 8)
                       for _ in specs]
    ordinary_before = [x.clone() for x in ordinary_tables]
    runner = NS(device=device, model_state=state, kv_cache_config=config,
                req_states=NS(num_computed_tokens=NS(gpu=computed)),
                vllm_config=NS(compilation_config=NS(static_forward_context=owners)),
                block_tables=NS(num_kv_cache_groups=2, blocks_per_kv_block=[1, 1],
                                kernel_block_sizes=[4, 4], _slot_mapping_enabled=[False, False],
                                block_tables=[NS(gpu=x) for x in ordinary_tables]))
    ports = lc.PinnedPorts.load()
    # Actual CPU+GPU-buffer class with both views on CPU. Pinning would require
    # a CUDA owner despite this being a CPU-only proof; no method is replaced.
    ports.make_buffer = lambda dev: functools.partial(CpuGpuBuffer, device=dev, pin_memory=False)
    tables = lc.LaneTables(runner, torch_module=torch)
    align = lc.LaneAlign(runner, ports, tables, torch_module=torch)
    ctx = align.ctx
    assert ctx.is_initialized  # set only by actual initialize_from_forward_context
    assert ctx.num_accepted_tokens_out.shape == (capacity,)
    assert all(x.shape[0] == 1 for x in tables.rows)
    assert ctx.block_table_ptrs.tolist() == [x.data_ptr() for x in tables.rows]

    # Execute the production idx_mapping initializer expression itself with
    # a CPU device; the active batch has one row even for request slot nine.
    tree = ast.parse(Path(lc.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LaneChunk')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    expr = next(n.value for n in init.body if isinstance(n, ast.Assign)
                and any(isinstance(x, ast.Attribute) and x.attr == 'idx_mapping' for x in n.targets))
    expression = compile(ast.Expression(expr), str(lc.__file__), 'eval')
    rows = []
    for slot in (0, 2, 9):
        accepted = state.num_accepted_tokens_gpu
        accepted[slot].fill_(1)
        before = accepted.clone()
        mapping = eval(expression, {'torch': torch, 'req_index': slot, 'device': device})
        row = invoke_to_kernel_boundary(ctx, accepted, state._mamba_state_idx_gpu, computed, mapping)
        assert torch.equal(ctx.num_accepted_tokens_out, accepted)
        assert torch.equal(accepted, before)
        assert int(mapping[0]) == slot
        row.update(slot=slot, snapshot_exact=True, accepted_unchanged=True,
                   private_batch_rows=1, request_slot_capacity=capacity)
        rows.append(row)
    groups = ports.mamba_groups(config)
    funcs = Qwen4ExpForConditionalGeneration.get_mamba_state_copy_funcs(
        {spec.mamba_type for spec in groups})
    legacy = stock.MambaSpecDecodeGPUContext.create(
        max_num_reqs=1, kv_cache_config=config, copy_funcs=funcs,
        device=device, make_buffer=ports.make_buffer(device))
    legacy.initialize_from_forward_context(config, owners, funcs, tables.rows)
    try:
        invoke_to_kernel_boundary(legacy, state.num_accepted_tokens_gpu,
                                  state._mamba_state_idx_gpu, computed, mapping)
    except RuntimeError as exc:
        assert 'broadcast shape [10]' in str(exc), str(exc)
        negative = dict(rejected=True, error_type=type(exc).__name__, error_message=str(exc)[:256],
                        snapshot_shape=[1], accepted_shape=[10], reached_kernel=False)
    else:
        raise AssertionError('native11 single-slot snapshot was not refused')
    assert state._mamba_ctx is None and state._mamba_state_copy_funcs is None
    assert stock.MambaSpecDecodeGPUContext.run_fused_postprocess_align is original_method
    assert stock.postprocess_mamba_fused_kernel is original_kernel
    assert all(torch.equal(a, b) for a, b in zip(caches, cache_before))
    assert all(torch.equal(a, b) for a, b in zip(ordinary_tables, ordinary_before))
    assert not torch.cuda.is_initialized()
    return dict(verdict='PASS', stock_source_sha256=sha(stock.__file__),
                lane_source_sha256=sha(lc.__file__), actual_context_class=type(ctx).__module__+'.'+type(ctx).__name__,
                actual_method='run_fused_postprocess_align', cases=rows, native11_negative=negative,
                private_tables_exact=True, private_batch_rows=1, request_slot_capacity=capacity,
                ordinary_context_unchanged=True, ordinary_tables_unchanged=True,
                state_caches_unchanged=True, methods_unpatched=True, kernels_unpatched=True,
                ready_flag_derived_from_actual_initialization=True,
                cpu_allocator_pinning=False, synthetic_cpu_ownership_inputs=True,
                model_constructed=False, engine_constructed=False, kernel_executed=False,
                scope='actual imported lane/context construction and stock method through real CPU snapshot copy; '
                      'explicit trace stop before GPU launch; small CPU ownership/descriptor inputs; no GPU/model/state-copy qualification')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = check(json.loads(a.graph.read_text()))
    result['schema'] = 'native12-real-mamba-slot-capacity/1'
    try:
        result['align_capacity'] = collect()
    except Exception as exc:
        result['failures'].append('align_capacity:'+type(exc).__name__)
        result['align_capacity'] = dict(verdict='FAIL', error_type=type(exc).__name__,
            error_message=str(exc)[:512], stack=[dict(file=Path(f.filename).name, function=f.name, line=f.lineno)
                                               for f in traceback.extract_tb(exc.__traceback__)[-8:]])
    import torch
    result['cuda_initialized'] = torch.cuda.is_initialized()
    if result['cuda_initialized']:
        result['failures'].append('CUDA-initialized')
    result['verdict'] = 'FAIL' if result['failures'] else 'INCONCLUSIVE' if result['missing'] else 'PASS'
    result['rc'] = 1 if result['failures'] else 2 if result['missing'] else 0
    result['probe_sha256'] = sha(__file__)
    result['graph_sha256'] = sha(a.graph)
    with a.out.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ('verdict', 'rc', 'counts', 'failures', 'missing', 'cuda_initialized')}))
    return result['rc']


if __name__ == '__main__':
    raise SystemExit(main())
