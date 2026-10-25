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
| Unexplained at idle after start (context, NCCL, 96 MiB staging arena) | 1.03–1.16 | **measured q15 25 Oct 06:46** (`tmp/q15-sample/mem-after-start.json`): nvidia-smi used 21.8–22.0 minus explained 20.77–20.83. The earlier 0.4–0.5 was from a 24 Oct reading at 21.2–21.3 used |
| Under load (teacher-forced 62.5k probes + 6 decode streams) | used 22.93–23.05; allocator reserved 22.01–22.05, allocated 21.02, peak 21.08 | **measured q15 after-tf**; ≈ 1.0 GiB of the reserved pool is cached-but-unallocated, ≈ 0.9–1.0 GiB sits outside the allocator |
| After a 120k warm/cold prefill and a 90 s drain | used 22.6–23.0; reserved 21.72–21.98, peak 21.13, **alloc_retries 2 on every rank**, 0 OOM | **measured q15 after-drain**: the allocator already freed cached blocks and retried at 120k |
| **Under pressure (two 200k decode streams + the 7568 leak case with 30 % cancels), per rank PP0_TP0 / PP0_TP1 / PP1_TP0 / PP1_TP1** | **used 23.50 / 23.32 / 22.68 / 22.48**; reserved 22.54 / 21.64 / 22.44 / 21.52, allocated 21.00 / 21.00 / 20.93 / 20.93, peak 21.08 / 21.07 / 21.09 / 21.09, **alloc_retries 4 / 5 / 4 / 5** (0 before this phase), 0 OOM; identical after a 90 s drain | **measured q15b 25 Oct 07:39** (`tmp/q15b-sample/mem-under-pressure.json`, `mem-after-drain.json`): the deepest state measured so far; TP0 ranks reserve ≈ 0.9 more than TP1 ranks, PP0 ranks use ≈ 0.8 more than PP1 (both unexplained) |
| **Headroom** | **rank 0 under pressure: ≈ 0.5 device-free, allocator cache 1.5 (TP1 ranks 0.6) and already retrying**; q15 teacher-forced load: ≈ 0.95 free + ≈ 1.0 cached (not 2.7) | measured q15/q15b; the idle "24.0 − used" figure (2.0–2.2) is not available under load; the allocator cache is reclaimable only by torch allocations and only if fragmentation permits |
| S3 retained history, full, active prefill | 1.16 at 200k / 1.5 at 262k | **estimate** (six independent QSA histories; Codex: ~1.158 GiB padded at 200k). 24 pages = 0.56, 40 pages = 0.94 measured as configured |
| E2 per-layer device window (Codex option, 25 Oct): one 50- or 65-page history window per rank shared by the six QSA layers, six logical ownership records, CPU-DMA refill before every layer | 0.193 (50 pages) / 0.251 (65 pages) | **candidate, currently unfunded**: Codex 25 Oct 07:55: at ≈ 0.5 GiB driver-free under pressure (q15b) the 0.5 GiB reserve leaves no funded window, single or double; needs preallocated-fit / reclaim evidence or an approved named trade (operator) before any declaration; costs retention reuse; repeated-DMA cost measured by the slice case `dma-every-step34` (q11y2 MoE-off, 25 Oct): 1,243,545,600 B of H2D per 1024-token chunk per rank at 200k, solo copy 12.9 GB/s on the x8 card / 25.8 GB/s x16 (≈ 97 / 48 ms per chunk), decode share 0.91 (x8 pair) / 1.00 (x16 pair) with 0 spikes, **prefill share 0.64–0.65 of own solo**; per-refill GPU copy 16.1 → 21.3 ms (x8) and 8.0 → 9.2–9.4 ms (x16) solo → overlap, host refill medians 21 → 30 ms (x8 pair) and 13 → 18 ms (x16 pair): the exposed cost is host synchronisation around 13 serial pieces, so Codex's next variant keeps one window and enqueues the pieces before one fence per layer |
| PLE chunk-row prefetch buffer | ~0.005 (FP8: ~2.5 MiB per 1,024-token chunk; ×2 if double-buffered) | **estimate**; only needed if PLE reads matter (q11s PLE-only control) |
| Vision tower (stage 0 only) | 0.42 (TP-split weights) + encode peak | BF16 weights measured; off in production (`--language-model-only`). q14b on one GPU (whole tower): 0.84 GiB weights, peak reserved 0.93 / 1.10 / 1.45 GiB at 512² / 1024² / 1536² |
| S5 extras: prefill-lane activations/workspace, per-split decode graphs, PyNccl buffers, green-context overhead | unknown | **to measure**: per-rank ledger in the next S5 queue (Codex `codex/s5-resident`) |
| Slice residency cost (4-layer S5 slice, one shared 50-page source, per rank; q11u 25 Oct) | +198 MiB allocated and +198 MiB reserved, exactly the copied bytes (no allocator overhead); peak reserved during overlap 2.0 GiB (QSA slice) / 2.6 GiB (mixed slice), slice weights included | measured on both TP pairs; a slice data point, not the engine's S5 extras |
| S5 sized components (design, not measured) | HC hidden 20 MiB, PP hidden 20 MiB per buffer, top-k 8 MiB (×6 if per layer), QSA accumulator 12 MiB, 23-page arena 91 MiB (reuse) | `reviews/codex-20261025s-s5-engine-design.md` §7; budget after retention: 1.54 GiB at 200k / 1.19 at 262k, minus a proposed 0.5 GiB reserve |

Fit, measured (q15 and q15b, 25 Oct): under teacher-forced load the engine uses 22.9–23.05 of 24.0 GiB per rank and the
allocator retried at 120k; under two 200k decode streams plus leak traffic rank 0 reaches 23.5 used with 4–5 allocator
retries on every rank (the deepest measured state, the one every declaration is checked against). Full 200k retention
(1.16) plus the S5 extras does **not** fit in the measured headroom without a
named source per rank: the 2.0 GiB KV pool (9.57× concurrency at 262k is more than the 6–8 agents served), the row
cache (11 requests × 110 MiB per layer), or partial retention (most recent / most selected pages) plus bulk prefetch
of the rest. Every E2/S5 candidate must declare its source in its ledger before it is armed (Codex to propose).
The earlier "~2.7 GiB headroom" fit-on-paper is withdrawn.

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

0. Done 25 Oct (q15): per-rank allocated/reserved/peak/retries on the unmodified production engine after start, under
   the teacher-forced load and after a 120k prefill + drain (`tmp/q15-sample/mem-after-*.json`, log-derived via
   `memory_ledger.py`; nvidia-smi used is direct).
1. Per-rank allocated/reserved/free with S3 retention + S5 overlap active (next S5 queue).
2. The same on the full engine once S5 is integrated; full-engine leak/lifetime gates.
3. Row-cache size vs decode throughput, if the S5 extras need its VRAM.
