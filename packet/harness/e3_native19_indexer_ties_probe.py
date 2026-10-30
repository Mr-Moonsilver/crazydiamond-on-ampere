#!/usr/bin/env python3
"""One-card pinned QSA indexer/top-k/union repeatability and fragmentation.

The compressed cache/query scores are synthetic and tie-heavy. Runtime kernels,
the real union scratch allocator and sparse planner are unchanged. This is not
a TP collective, a model forward, or a production numerical certificate.
"""
import hashlib
import json
from pathlib import Path
import time


ROWS = 64
PAGES = 50
PAGE_ROWS = 4048
RATIO = 4
TOPK = 2048
FRONTIER = PAGES * PAGE_ROWS
COMPRESSED_PAGE = PAGE_ROWS // RATIO
REPEATS = 200


def _modules():
    import torch
    from vllm.models.qwen4_exp.nvidia.ops import qsa, qsa_indexer
    from vllm.v1.flash_e3 import qsa_operation, sparse_history_dma
    required = (qsa._s6_union, qsa_indexer.qsa_select_paged_prefill,
                qsa_indexer.expand_qsa_block_indices,
                qsa_operation.NativeQsaDevice.alloc_union_scratch,
                sparse_history_dma.make_sparse_plan,
                sparse_history_dma.stream_sparse_plan)
    assert all(callable(item) for item in required)
    return torch, qsa, qsa_indexer, qsa_operation, sparse_history_dma


def cpu_shape(*, require_uninitialized):
    torch, qsa, idx, op, sparse = _modules()
    initialized = bool(torch.cuda.is_initialized())
    if require_uninitialized:
        assert not initialized
    assert sparse.PAGE_ROWS == PAGE_ROWS
    assert sparse.MAX_LAYER_ROWS >= FRONTIER
    assert idx._TOPK_WORKSPACE_BYTES > 0
    assert callable(op.NativeQsaDevice.bind_streams)
    return dict(verdict='PASS', pages=PAGES, page_rows=PAGE_ROWS,
        compressed_page=COMPRESSED_PAGE, frontier=FRONTIER, rows=ROWS,
        topk=TOPK, compress_ratio=RATIO, packed_width=TOPK + RATIO,
        repeats_per_condition=REPEATS, actual_pinned_functions_accessed=6,
        cuda_initialized=initialized, topk_extension_execution_deferred=True)


def _fragmentation(tokens, sparse):
    selected = tuple(int(x) for x in tokens.cpu().tolist())
    strict = sparse.make_sparse_plan(selected, tuple(range(PAGES)),
        tuple(range(1, PAGES + 1)), 1, valid_tokens=FRONTIER)
    if strict.refused:
        assert strict.fallback.reason == 'descriptor_cap'
        plan = sparse.stream_sparse_plan(strict)
    else:
        plan = strict
    waves = sum(sum(1 for _ in bucket.waves()) if hasattr(bucket, 'waves') else 1
                for bucket in plan.buckets)
    return dict(selected=len(selected), strict_refusal=(strict.fallback.reason if strict.refused else None),
        buckets=len(plan.buckets), descriptors=plan.descriptors,
        waves=waves, descriptor_high_water=plan.descriptor_high_water)


def gpu_case():
    torch, qsa, idx, op, sparse = _modules()
    assert torch.cuda.device_count() == 1
    device = 'cuda:0'
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., (2 * 1024**3) / total), 0)
    torch.manual_seed(19)
    compute, copy, load = (torch.cuda.Stream(device=device) for _ in range(3))
    native = op.NativeQsaDevice(device)
    native.bind_streams(compute, copy)
    # A fixed discrete score vocabulary creates genuine bf16 ties. The two
    # peers receive byte-identical Q/K and page maps; they differ only in
    # stream scheduling and the declared concurrent-load condition.
    with torch.cuda.stream(compute):
        q = torch.randint(-2, 3, (ROWS, 4, 128), device=device, dtype=torch.int16).to(torch.bfloat16)
        keys = torch.randint(-2, 3, (PAGES, COMPRESSED_PAGE, 1, 128),
                             device=device, dtype=torch.int16).to(torch.bfloat16)
        table = torch.arange(PAGES, device=device, dtype=torch.int32).reshape(1, PAGES)
        starts = torch.tensor((0, ROWS), device=device, dtype=torch.int32)
        positions = torch.arange(FRONTIER - ROWS, FRONTIER, device=device, dtype=torch.int32)
        visible = torch.div(positions, RATIO, rounding_mode='floor') + 1
        rows = torch.arange(ROWS, device=device, dtype=torch.int32)
        peers = []
        for _ in (0, 1):
            blocks = torch.empty((ROWS, TOPK // RATIO), device=device, dtype=torch.int32)
            packed = torch.empty((ROWS, TOPK + RATIO), device=device, dtype=torch.int32)
            scratch = type('Scratch', (), {})()
            for name, value in native.alloc_union_scratch(FRONTIER, ROWS,
                    packed.shape[1] - 1, 'torch.int32', packed).items():
                setattr(scratch, name, value)
            peers.append((blocks, packed, scratch))
    torch.cuda.synchronize(device)
    samples, mismatch = [], 0
    for condition in ('idle', 'concurrent_load'):
        for iteration in range(REPEATS):
            if condition == 'concurrent_load':
                with torch.cuda.stream(load):
                    torch.cuda._sleep(100000)
            outputs = []
            for blocks, packed, scratch in peers:
                with torch.cuda.stream(compute):
                    scratch.flags.zero_()
                    idx.qsa_select_paged_prefill(q, keys, table, starts, visible,
                        TOPK, RATIO, ROWS, blocks, FRONTIER)
                    idx.expand_qsa_block_indices(blocks, positions, visible,
                        RATIO, TOPK, packed)
                    tokens, info, remapped = qsa._s6_union(packed, rows,
                                                             FRONTIER, scratch)
                    done = torch.cuda.Event()
                    done.record(compute)
                done.synchronize()
                count = int(info[0].item())
                assert 0 < count <= FRONTIER
                outputs.append(tokens[:count].clone())
            equal = bool(torch.equal(outputs[0], outputs[1]))
            mismatch += int(not equal)
            if iteration in (0, REPEATS - 1):
                samples.append(dict(condition=condition, iteration=iteration,
                    equal=equal, counts=[int(x.numel()) for x in outputs],
                    digests=[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
                             for x in outputs],
                    fragmentation=_fragmentation(outputs[0], sparse)))
    torch.cuda.synchronize(device)
    return dict(verdict='PASS' if mismatch == 0 else 'FAIL',
        repeats_per_condition=REPEATS, conditions=['idle', 'concurrent_load'],
        peer_comparisons=2 * REPEATS, selected_set_mismatches=mismatch,
        samples=samples, topk_and_union_kernels_executed=True,
        actual_native_union_allocator=True, synthetic_tie_scores=True,
        separate_load_stream_no_nccl=True,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--mode', choices=('cpu-shape', 'gpu'), required=True)
    args = parser.parse_args()
    result = dict(schema='native19-indexer-ties/1', verdict='INCONCLUSIVE',
                  mode=args.mode, box_evidence=False, started_ns=time.monotonic_ns())
    try:
        result['cpu_shape'] = cpu_shape(require_uninitialized=args.mode == 'cpu-shape')
        if args.mode == 'gpu':
            result['gpu'] = gpu_case()
            result['verdict'] = result['gpu']['verdict']
        else:
            result['verdict'] = 'PASS'
    except BaseException as exc:
        import traceback
        result.update(verdict='FAIL', error_type=type(exc).__name__,
                      error_message=str(exc), stack=traceback.format_exc())
    result['elapsed_ns'] = time.monotonic_ns() - result['started_ns']
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(result['verdict'], flush=True)
    return {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[result['verdict']]


if __name__ == '__main__':
    raise SystemExit(main())
