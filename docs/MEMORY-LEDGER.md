# Memory ledger (goal item 8)

One place for every memory consumer on CT <ct-id> (4 × RTX 3090, 24.0 GiB each; host 240 GiB CT limit).
Measured values cite their run; planned items are marked **estimate** until a per-rank ledger measures them.
Tool for the live engine: `harness/memory_ledger.py` (read-only, allowlisted settings only).

## Per GPU (one rank of TP2 × PP2)

| Consumer | GiB | Source / status |
|---|---:|---|
| Model weights | 18.05 (PP0) / 17.99 (PP1) | measured, live engine 24 Oct 15:16 (overlay-v16, text only) |
| KV pool (`--kv-cache-memory-bytes`) | 2.0 | measured (2.15 GB configured) |
| QSA row cache | 0.64 | measured: 6 QSA layers × 110 MiB (11 requests × 10,240 slots) |
| CUDA graphs | 0.14 | measured |
| Unexplained (activations, allocator cache, NCCL, 96 MiB staging arena, context) | 1.0–1.1 | measured, used 21.2–21.3 minus explained 20.8 |
| **Headroom** | **~2.7** | 24.0 − used |
| S3 retained history, full, active prefill | 1.16 at 200k / 1.5 at 262k | **estimate** (six independent QSA histories; Codex: ~1.158 GiB padded at 200k). 24 pages = 0.56, 40 pages = 0.94 measured as configured |
| PLE chunk-row prefetch buffer | ~0.005 (FP8: ~2.5 MiB per 1,024-token chunk; ×2 if double-buffered) | **estimate**; only needed if PLE reads matter (q11s PLE-only control) |
| Vision tower (stage 0 only) | 0.42 (TP-split weights) + encode peak | BF16 weights measured; off in production (`--language-model-only`). q14b on one GPU (whole tower): 0.84 GiB weights, peak reserved 0.93 / 1.10 / 1.45 GiB at 512² / 1024² / 1536² |
| S5 extras: prefill-lane activations/workspace, per-split decode graphs, PyNccl buffers, green-context overhead | unknown | **to measure**: per-rank ledger in the next S5 queue (Codex `codex/s5-resident`) |

Fit on paper (stage 0, worst case): full 200k retention 1.16 + vision 0.42 = 1.58 GiB of ~2.7 → ~1.1 GiB left
for the S5 extras. Without vision ~1.5 GiB. At 262k retention 1.5 GiB leaves ~0.8 (with vision) / ~1.2 (without).
If the S5 extras do not fit: smaller row cache during a prefill (the biggest tunable), fewer KV-pool bytes, or
partial retention (most recent / most selected pages) plus bulk prefetch of the rest.

## Host (CT <ct-id>)

| Consumer | GiB | Source / status |
|---|---:|---|
| PLE table, BF16 | 95.9 | measured (OVERNIGHT-20261024) |
| PLE table, FP8 (`intel-w4a16-fp8ple`) | ~48.6 | q7b 25 Oct 01:48: shmem 219.9 → 172.6 (−47.3), all steps PASS |
| Pinned host QSA K/V pools | 4 × ~31 (≤ 33) | measured (1,340-block pool) |
| Shared memory total | 219.9 (BF16 PLE) / 172.6 (FP8 PLE) | measured |
| Anonymous memory (engine, API, tools) | ≤ 8.2 steady; 20 under the profiler | measured 24 Oct |
| Required container headroom | ≥ 10 | rule (STALL-PLAN): below it, graph capture / profiler tip the CT into thrashing |
| CT limit / host spare | 240 / ~16 beyond the CT | by design |

With FP8 PLE the CT gains ~47 GiB of slack: enough for profiling, larger host KV pools, or a remote vision
service's host side, without raising the CT limit.

## Open measurements

1. Per-rank allocated/reserved/free with S3 retention + S5 overlap active (next S5 queue).
2. The same on the full engine once S5 is integrated; full-engine leak/lifetime gates.
3. Row-cache size vs decode throughput, if the S5 extras need its VRAM.
