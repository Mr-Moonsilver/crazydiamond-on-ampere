#!/usr/bin/env python3
"""Compare actual pinned QSA top-k sets on equal tied and unique scores.

This is a one-card diagnostic with synthetic scores. It does not construct the
model, multiple TP processes, or a box qualification run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

from e3_native19_indexer_diag_probe import (
    ROWS, PAGES, PAGE_ROWS, RATIO, TOPK, FRONTIER, COMPRESSED_PAGE,
    digest, modules,
)

REPEATS = 8
LOAD_CYCLES = 2_000_000_000


def row_sets(first, second, scores, thresholds, visible):
    """Retain every row's set delta and exact boundary-score classification."""
    rows = []
    for row in range(ROWS):
        left, right = first[row], second[row]
        a, b = set(left), set(right)
        changed = sorted(a ^ b)
        valid = all(0 <= index < visible[row] for index in a | b)
        nonboundary = [index for index in changed if
                       not 0 <= index < visible[row] or
                       float(scores[row, index]) != float(thresholds[row])]
        rows.append(dict(row=row, set_equal=a == b,
                         symmetric_difference_size=len(changed),
                         all_differing_at_boundary=not nonboundary,
                         nonboundary_difference_count=len(nonboundary),
                         left_duplicate_count=len(left) - len(a),
                         right_duplicate_count=len(right) - len(b),
                         indices_valid=valid,
                         ordered_equal=left == right))
    return rows


def summary(rows):
    return dict(rows=len(rows), set_different=sum(not r['set_equal'] for r in rows),
                ordered_different=sum(not r['ordered_equal'] for r in rows),
                symdiff_total=sum(r['symmetric_difference_size'] for r in rows),
                nonboundary_total=sum(r['nonboundary_difference_count'] for r in rows),
                invalid_rows=sum(not r['indices_valid'] for r in rows),
                duplicate_rows=sum(bool(r['left_duplicate_count'] or
                                        r['right_duplicate_count']) for r in rows))


def cpu_shape():
    torch, _, idx, _ = modules()
    assert not torch.cuda.is_initialized()
    assert callable(torch.cuda._sleep) and callable(idx._topk)
    bf16 = torch.tensor((1.0, -2.0, 0.5), dtype=torch.bfloat16)
    assert digest(bf16) == hashlib.sha256(bf16.view(torch.uint8).numpy().tobytes()).hexdigest()
    # Execute the actual comparison predicate on CPU, including tie and
    # non-tie differences. Only CUDA kernels are deferred to the VM.
    import numpy as np
    scores = np.zeros((ROWS, 4), dtype=np.float32)
    scores[:, 1] = 1
    scores[:, 2] = 1
    scores[:, 3] = 2
    left = [[3, 1] for _ in range(ROWS)]
    right = [[3, 2] for _ in range(ROWS)]
    tied = row_sets(left, right, scores, [1] * ROWS, [4] * ROWS)
    assert summary(tied)['set_different'] == ROWS
    assert summary(tied)['nonboundary_total'] == 0
    scores[:, 2] = 3
    unique = row_sets(left, right, scores, [1] * ROWS, [4] * ROWS)
    assert summary(unique)['nonboundary_total'] == ROWS
    return dict(verdict='PASS', bf16_storage_hash_executed=True,
                actual_predicate_executed=True, tied_rows=summary(tied),
                unique_rows=summary(unique), gpu_deferred=True,
                cuda_initialized=False)


def gpu_case():
    torch, _, idx, _ = modules()
    assert torch.cuda.device_count() == 1
    device = 'cuda:0'
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 2 * 1024**3 / total), 0)
    torch.manual_seed(19)
    compute = torch.cuda.Stream(device=device)
    load = torch.cuda.Stream(device=device)
    with torch.cuda.stream(compute):
        q = torch.randint(-2, 3, (ROWS, 4, 128), device=device,
                          dtype=torch.int16).to(torch.bfloat16)
        keys = torch.randint(-2, 3, (PAGES, COMPRESSED_PAGE, 1, 128),
                             device=device, dtype=torch.int16).to(torch.bfloat16)
        table = torch.arange(PAGES, device=device, dtype=torch.int32).reshape(1, PAGES)
        starts = torch.tensor((0, ROWS), device=device, dtype=torch.int32)
        positions = torch.arange(FRONTIER - ROWS, FRONTIER,
                                 device=device, dtype=torch.int32)
        visible = torch.div(positions, RATIO, rounding_mode='floor') + 1
        logits_width = min(((FRONTIER + RATIO - 1) // RATIO + 63) // 64 * 64,
                           PAGES * COMPRESSED_PAGE)
        columns = torch.arange(logits_width, device=device)[None, :]
        valid_columns = columns < visible[:, None]
        unique_scores = torch.arange(logits_width, device=device,
                                     dtype=torch.float32).repeat(ROWS, 1)
        workspace = torch.empty((idx._TOPK_WORKSPACE_BYTES,),
                                dtype=torch.uint8, device=device)
        outputs = [torch.empty((ROWS, TOPK // RATIO), device=device,
                               dtype=torch.int32) for _ in range(2)]
    compute.synchronize()
    q_before, keys_before, unique_before = digest(q), digest(keys), digest(unique_scores)
    visible_host = [int(x) for x in visible.cpu().tolist()]
    unique_host = unique_scores.cpu().numpy()
    unique_boundary = [float(n - TOPK // RATIO) for n in visible_host]
    records = []
    baseline_tied_score_digest = None
    for phase in ('idle', 'concurrent_load'):
        for iteration in range(REPEATS):
            if phase == 'concurrent_load':
                with torch.cuda.stream(load):
                    torch.cuda._sleep(LOAD_CYCLES)
                    load_done = torch.cuda.Event()
                    load_done.record(load)
            else:
                load_done = None
            tied_outputs, unique_outputs, score_digests = [], [], []
            # Peer zero repeats on the SAME output/workspace; peer one uses
            # different output storage. Score tensors are freshly produced.
            for peer in (0, 0, 1):
                with torch.cuda.stream(compute):
                    logits = idx._prefill_logits(q, keys, table, starts, visible,
                                                 ROWS, logits_width, 0, ROWS)
                    active = bool(load_done is not None and not load_done.query())
                    idx._topk(logits, visible, TOPK, RATIO, outputs[peer], workspace)
                compute.synchronize()
                scored = logits.masked_fill(~valid_columns, -float('inf'))
                score_digests.append(digest(scored))
                tied_outputs.append(dict(peer=peer, load_active_at_submit=active,
                                         ordered_sha256=digest(outputs[peer]),
                                         rows=outputs[peer].cpu().tolist()))
            for peer in (0, 1):
                with torch.cuda.stream(compute):
                    active = bool(load_done is not None and not load_done.query())
                    idx._topk(unique_scores, visible, TOPK, RATIO,
                              outputs[peer], workspace)
                compute.synchronize()
                unique_outputs.append(dict(peer=peer, load_active_at_submit=active,
                                           ordered_sha256=digest(outputs[peer]),
                                           rows=outputs[peer].cpu().tolist()))
            if load_done is not None:
                load.synchronize()
            assert len(set(score_digests)) == 1, 'peer scored logits differ'
            if baseline_tied_score_digest is None:
                baseline_tied_score_digest = score_digests[0]
            assert score_digests[0] == baseline_tied_score_digest, 'scored logits changed across repeats'
            # Every peer received the same scored matrix. For each changed
            # index, inspect that row's exact float32 value, not a bf16 cast.
            tied_score_host = scored.cpu().numpy()
            tied_boundary = torch.topk(scored, TOPK // RATIO, dim=1).values[:, -1].cpu().tolist()
            repeated = row_sets(tied_outputs[0]['rows'], tied_outputs[1]['rows'],
                                tied_score_host, tied_boundary, visible_host)
            peers = row_sets(tied_outputs[1]['rows'], tied_outputs[2]['rows'],
                             tied_score_host, tied_boundary, visible_host)
            unique = row_sets(unique_outputs[0]['rows'], unique_outputs[1]['rows'],
                              unique_host, unique_boundary, visible_host)
            # Retain per-row facts and compact source digests; no raw 50k-wide
            # scores or selected indices are required in the receipt.
            records.append(dict(phase=phase, iteration=iteration,
                                score_sha256=score_digests[0],
                                tied_ordered_sha256=[x['ordered_sha256'] for x in tied_outputs],
                                unique_ordered_sha256=[x['ordered_sha256'] for x in unique_outputs],
                                load_active_at_submit=[x['load_active_at_submit']
                                                       for x in tied_outputs + unique_outputs],
                                tied_self_repro=dict(summary=summary(repeated), rows=repeated),
                                tied_peer=dict(summary=summary(peers), rows=peers),
                                unique_peer=dict(summary=summary(unique), rows=unique)))
    q_after, keys_after, unique_after = digest(q), digest(keys), digest(unique_scores)
    active_loaded = sum(all(r['load_active_at_submit']) for r in records
                        if r['phase'] == 'concurrent_load')
    inputs_unchanged = (q_before == q_after and keys_before == keys_after
                        and unique_before == unique_after)
    verdict = ('FAIL' if not inputs_unchanged else
               'INCONCLUSIVE' if active_loaded != REPEATS else 'PASS')
    return dict(verdict=verdict,
                shared_q_sha256=q_before, shared_k_sha256=keys_before,
                unique_score_sha256=unique_before,
                scored_logits_sha256=baseline_tied_score_digest,
                inputs_unchanged=inputs_unchanged,
                phases=('idle', 'concurrent_load'), repeats_per_phase=REPEATS,
                loaded_iterations_all_five_active_at_submit=active_loaded,
                load_cycles=LOAD_CYCLES, load_is_single_stream_sleep=True,
                records=records, kernels_unpatched=True, overlay_unchanged=True,
                synthetic_tie_scores=True, box_evidence=False,
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('cpu-shape', 'gpu-all'), required=True)
    args = parser.parse_args()
    result = dict(schema='native19-indexer-set-diagnosis/1', verdict='INCONCLUSIVE',
                  mode=args.mode,
                  graph_sha256=hashlib.sha256(args.graph.read_bytes()).hexdigest(),
                  gpu_execution=False, box_evidence=False)
    try:
        if args.mode == 'cpu-shape':
            result['cpu_shape'] = cpu_shape()
            result['verdict'] = result['cpu_shape']['verdict']
        else:
            result['gpu'] = gpu_case()
            result['gpu_execution'] = True
            result['verdict'] = result['gpu']['verdict']
    except BaseException as exc:
        result.update(verdict='FAIL', failure_class='HARNESS_OR_ENGINE_OPEN',
                      error_type=type(exc).__name__, error_message=str(exc),
                      stack=traceback.format_exc())
    else:
        result['failure_class'] = 'NONE' if result['verdict'] == 'PASS' else 'OPEN'
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(result['verdict'], flush=True)
    return {'PASS': 0, 'INCONCLUSIVE': 2, 'FAIL': 1}[result['verdict']]


if __name__ == '__main__':
    raise SystemExit(main())
