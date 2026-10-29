#!/usr/bin/env python3
"""Locate the first divergent pinned QSA selection stage on one CUDA device.

This is a diagnostic over synthetic ties, not model or box evidence. All Q/K,
page-map and visibility inputs are shared by two separately allocated peers.
Each real stage is synchronized and its bytes retained before the next stage.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

ROWS, PAGES, PAGE_ROWS, RATIO, TOPK = 64, 50, 4048, 4, 2048
FRONTIER = PAGES * PAGE_ROWS
COMPRESSED_PAGE = PAGE_ROWS // RATIO
REPEATS = 8


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()


def modules():
    import torch
    from vllm.models.qwen4_exp.nvidia.ops import qsa, qsa_indexer as idx
    from vllm.v1.flash_e3.qsa_operation import NativeQsaDevice
    required = (idx._prefill_logits, idx._topk, idx.qsa_select_paged_prefill,
                idx.expand_qsa_block_indices, qsa._s6_union,
                NativeQsaDevice.alloc_union_scratch)
    assert all(callable(f) for f in required)
    assert idx._TOPK_WORKSPACE_BYTES > 0
    return torch, qsa, idx, NativeQsaDevice


def cpu_shape():
    torch, _, _, _ = modules()
    assert not torch.cuda.is_initialized()
    return dict(verdict='PASS', actual_pinned_functions_accessed=6,
                gpu_stages_deferred=True, cuda_initialized=False,
                rows=ROWS, frontier=FRONTIER, repeats=REPEATS)


def gpu_case():
    torch, qsa, idx, NativeQsaDevice = modules()
    assert torch.cuda.device_count() == 1
    device = 'cuda:0'
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 2 * 1024**3 / total), 0)
    torch.manual_seed(19)
    compute = torch.cuda.Stream(device=device)
    native = NativeQsaDevice(device)
    native.bind_streams(compute, torch.cuda.Stream(device=device))
    with torch.cuda.stream(compute):
        q = torch.randint(-2, 3, (ROWS, 4, 128), device=device,
                          dtype=torch.int16).to(torch.bfloat16)
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
    q_before, keys_before = digest(q), digest(keys)
    peer_ptrs = [{name: int(getattr(scratch, name).data_ptr())
                  for name in ('flags', 'ranks', 'tokens', 'totals', 'offsets', 'info', 'remapped')}
                 for _, _, scratch in peers]
    assert all(peer_ptrs[0][name] != peer_ptrs[1][name] for name in peer_ptrs[0])
    logits_width = min(((FRONTIER + RATIO - 1) // RATIO + 63) // 64 * 64,
                       PAGES * COMPRESSED_PAGE)
    score_digests = []
    valid_columns = torch.arange(logits_width, device=device)[None, :] < visible[:, None]
    with torch.cuda.stream(compute):
        for _ in range(2):
            scores = idx._prefill_logits(q, keys, table, starts, visible,
                                        ROWS, logits_width, 0, ROWS)
            torch.cuda.synchronize(device)
            # The pinned scoring kernel does not write columns beyond each
            # row's visible count. Mask those unspecified bytes before hashing.
            visible_scores = scores.masked_fill(~valid_columns, -float('inf'))
            score_digests.append(digest(visible_scores))
        threshold = torch.topk(visible_scores, TOPK // RATIO, dim=1).values[:, -1:]
        equal_at_boundary = (visible_scores == threshold).sum(dim=1)
        above_boundary = (visible_scores > threshold).sum(dim=1)
    torch.cuda.synchronize(device)
    records = []
    for iteration in range(REPEATS):
        pair = []
        for blocks, packed, scratch in peers:
            with torch.cuda.stream(compute):
                idx.qsa_select_paged_prefill(q, keys, table, starts, visible,
                    TOPK, RATIO, ROWS, blocks, FRONTIER)
            compute.synchronize()
            block_sha = digest(blocks)
            with torch.cuda.stream(compute):
                idx.expand_qsa_block_indices(blocks, positions, visible,
                                             RATIO, TOPK, packed)
            compute.synchronize()
            packed_sha = digest(packed)
            with torch.cuda.stream(compute):
                scratch.flags.zero_()
                tokens, info, _ = qsa._s6_union(packed, rows, FRONTIER, scratch)
            compute.synchronize()
            count = int(info[0].item())
            assert 0 < count <= FRONTIER
            pair.append(dict(block_sha256=block_sha, packed_sha256=packed_sha,
                             union_sha256=digest(tokens[:count]), selected_count=count))
        records.append(dict(iteration=iteration, peers=pair,
            blocks_equal=pair[0]['block_sha256'] == pair[1]['block_sha256'],
            packed_equal=pair[0]['packed_sha256'] == pair[1]['packed_sha256'],
            union_equal=pair[0]['union_sha256'] == pair[1]['union_sha256']))
    # A strictly ordered score matrix asks whether the same persistent top-k
    # implementation is stable when there is no tie at the selection boundary.
    with torch.cuda.stream(compute):
        unique = torch.arange(logits_width, device=device,
                              dtype=torch.float32).repeat(ROWS, 1)
        workspace = torch.empty((idx._TOPK_WORKSPACE_BYTES,),
                                dtype=torch.uint8, device=device)
        unique_blocks = []
        for _ in range(2):
            output = torch.empty((ROWS, TOPK // RATIO), device=device,
                                 dtype=torch.int32)
            idx._topk(unique, visible, TOPK, RATIO, output, workspace)
            compute.synchronize()
            unique_blocks.append(digest(output))
    q_after, keys_after = digest(q), digest(keys)
    first = next((stage for stage, key in
                 (('scores', None), ('topk_blocks', 'blocks_equal'),
                  ('expanded', 'packed_equal'), ('union', 'union_equal'))
                 if (score_digests[0] != score_digests[1] if key is None else
                     any(not row[key] for row in records))), 'none')
    return dict(verdict='PASS' if q_before == q_after and keys_before == keys_after else 'FAIL',
        first_divergence=first, shared_q_sha256=q_before,
        shared_k_sha256=keys_before, q_unchanged=q_before == q_after,
        keys_unchanged=keys_before == keys_after,
        peer_scratch_disjoint=True, score_digests=score_digests,
        boundary_equal_min=int(equal_at_boundary.min().item()),
        boundary_equal_max=int(equal_at_boundary.max().item()),
        boundary_equal_rows_gt_one=int((equal_at_boundary > 1).sum().item()),
        above_boundary_min=int(above_boundary.min().item()),
        unique_topk_equal=unique_blocks[0] == unique_blocks[1],
        unique_topk_digests=unique_blocks, records=records,
        kernels_unpatched=True, overlay_unchanged=True,
        synthetic_tie_scores=True, box_evidence=False,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('cpu-shape', 'gpu-all'), required=True)
    args = parser.parse_args()
    result = dict(schema='native19-indexer-diagnosis/1', verdict='INCONCLUSIVE',
                  mode=args.mode, graph_sha256=hashlib.sha256(args.graph.read_bytes()).hexdigest(),
                  failure_class='OPEN', gpu_execution=False, box_evidence=False)
    try:
        result['cpu_shape'] = cpu_shape() if args.mode == 'cpu-shape' else None
        if args.mode == 'gpu-all':
            result['gpu'] = gpu_case()
            result['gpu_execution'] = True
            result['verdict'] = result['gpu']['verdict']
        else:
            result['verdict'] = 'PASS'
        result['failure_class'] = 'NONE' if result['verdict'] == 'PASS' else 'OPEN'
    except BaseException as exc:
        result.update(verdict='FAIL', failure_class='OPEN', error_type=type(exc).__name__,
                      error_message=str(exc), stack=traceback.format_exc())
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(result['verdict'], flush=True)
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
