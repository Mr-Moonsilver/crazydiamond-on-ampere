# Stock decode-step attention-metadata cost: measure, explain, design (Phase 1)

Branch `claude/attn-meta` (cut from `claude/alpha-next` 48aca6b2). Hub only: no box, no VM, no GPU, no engine
package change. Records: `results/hub/attn-meta/w1..w5`. Status log: `tmp/attn-meta/STATUS.md` (outside the repo).

Box fact being explained (`results/box/q18a-stage1-sweep/README.md`, live, mean per stock decode step, every rank):
`prep_attn_meta` (= `model_state.prepare_attn`, V2) **3.7–3.9 ms** of a **5.6–5.7 ms** `execute_model`. CPU time equals
wall time there, so it is host compute, not a wait. TP2xPP2, 8 decode streams, ~65k native prefill alongside, group
size 1, FULL_DECODE_ONLY.

## 0. Short answer

- Per stock decode step each rank builds metadata for **21 builders** (stage 0: 22): **18 GDN groups** (one GDN layer
  each), the QSA ring group, the QSA compressed half and the FA half of the QSA uniform group, plus PLE short-conv on
  stage 0.
- The 18 GDN builds are **70–75 % of the host cost** on the CPU witness (1.67–1.81 ms of 2.14–2.74 ms warm). Each runs
  the same 9 tiny device kernels (162 per step), 6 of which compute **the same value for all 18 groups** (the per-request
  start block). Only the gather source (the group's block table) and the destination buffers differ.
- It is **not mostly dead work**. GDN computes values the FULL graph reads, just 18x redundantly. Dead-only work is about
  0.4–0.8 ms: the FA build, QSA host checks, PLE host logic, framework objects and the GDN dataclass/Python. This is
  because under FULL replay the returned metadata dict is never read. Only in-place writes into builder persistent
  buffers reach the graph.
- **No host sync** anywhere on the path (all `.item()` read CPU tensors).
- **Chosen for Phase 2: option A, GDN group-batched decode staging.** It computes the start index once, does one stacked
  gather, and makes two foreach copies into each builder's own buffers. CPU witness: stock 2.17–2.83 ms → candidate
  0.60–1.03 ms (p50). That is **−1.6 to −1.9 ms CPU**, with ~162 → 7 device ops (box launch premium removed on top,
  modelled +0.15–0.6 ms). Bit-identical by construction (integer index copies), and equality is tested against stock on
  every metadata field and every builder buffer. Default off (`FLASH_ATTN_META_GDN_BATCH=1` enables). It adds no
  builder field and does not touch `build_attn_metadata` (the native lane's path).

## 1. Where the time goes (from the V2 source in `reference/vllm-src`)

### 1.1 Call path

`GPUModelRunner.execute_model` → `self.model_state.prepare_attn(input_batch, cudagraph_mode, block_tables,
slot_mappings, attn_groups, kv_cache_config)` (wrapped on the box by `engine_offers` as `input_attention/attn_metadata`,
and by the anatomy seam `prep_attn_meta`).

- `Qwen4ExpModelState` inherits `MambaHybridModelState.prepare_attn` (`v1/worker/gpu/model_states/mamba_hybrid.py:232`).
  In FULL it uses `num_reqs = num_reqs_after_padding` and `num_tokens = num_tokens_after_padding`.
- It builds the CPU `query_start_loc_cpu`, `max_query_len` (CPU `.item()`), `max_seq_len` (CPU `.item()`) and
  `is_prefilling`, then runs the align-mode loop.
- `attn_utils.build_attn_metadata` (`v1/worker/gpu/attn_utils.py:392`) loops over **every KV-cache group**. For each
  group it makes a `CommonAttentionMetadata` (block table, slot mapping) and calls
  `group.get_metadata_builder(0).build(common_prefix_len=0, common_attn_metadata=cm)` for each attention group. The
  result is a dict `layer_name → metadata`.

### 1.2 KV-cache groups on the box layout (`kv_cache_utils._get_packed_kv_cache_groups`; witness `builders_stage*`)

| group | spec | builder | layers | where |
|---|---|---|---|---|
| 0–35 | MambaSpec (GDN state, one state per block, block 4048) | `GDNAttentionMetadataBuilder` | 1 each | 18 per stage (0–17 stage 0, 18–35 stage 1) |
| 36 | MambaSpec (PLE short-conv) | `PleShortConvAttentionMetadataBuilder` | 1 | stage 0 only |
| 37 | CircularBufferSpec (QSA ring, block 4) | `QSAMetadataBuilder[ring]` | 6 per stage | both |
| 38 | UniformTypeKVCacheSpecs: FullAttentionSpec + MLA-compressed | `Qwen4ExpQSAMetadataBuilder` (FA) + `QSAMetadataBuilder[compressed]` | 6 per stage | both |

The rank builds metadata only for groups with local layers: 18 GDN + ring + QSA uniform (2 builders) on stage 1, plus
PLE on stage 0. That is 21 `CommonAttentionMetadata` objects per step on stage 1 (22 on stage 0).

### 1.3 Per builder, FULL decode step (all-decode, R = padded requests, T = padded tokens)

**GDN (`v1/attention/backends/gdn_attn.py:209`), x18.**
- `mamba_get_block_table_tensor` (`v1/attention/backends/utils.py:1150ff`, align mode) runs
  `start = (seq_lens-1)//4048; start.clamp_(0); idx = (start.unsqueeze(1)+arange(1)).to(int64); gather(block_table,
  1, idx)`. These are sub, floor_divide, clamp_, arange, add, _to_copy and gather: 7 kernels.
- The split into decodes/prefills is CPU-side.
- FULL branch (`:476–494`) has 2 `copy_` into the builder's persistent `non_spec_state_indices_tensor[:R]` and
  `non_spec_query_start_loc[:R+1]`, plus 2 `fill_` over `[R:R]` and `[R+1:R+1]`. These are empty, since
  num_decodes == batch_size == R.
- Per group that is 9 real device ops, 2 empty fills and 9 views; one `GDNAttentionMetadata`.
- **Per step:** 162 kernels + 36 empty fills + 162 views. Witness census: `_to_copy, add, arange, clamp_, floor_divide,
  gather, sub` 18 each; `copy_` 36, `fill_` 36.
- Changes step to step: `seq_lens` (+1 per step), so `start` changes only when a request crosses a 4048-token block.
  The block table changes only on allocation, admission or reorder. `query_start_loc` for uniform decode is
  `[0..R]`.
- **Identical across the 18 groups:** `seq_lens`, `start`, `idx`, `query_start_loc`, R, T and block size. Only the
  gathered block table (each group's own table) and the destination buffers differ.
- **Graph reads:** exactly the 2 persistent buffers (dataflow: all GDN device ops lie on a path to a graph write; 0 dead).

**PLE short-conv (`v1/attention/backends/short_conv_attn.py`, BaseMamba `_compute_common_metadata`), x1, stage 0.**
- 15 device ops and 13 host ops (`.item()` on CPU, any/eq/gt on CPU masks).
- FULL staging writes `state_indices_tensor_d` and `has_initial_states_d`. This staging is **gated on the config's
  `cudagraph_mode`, read inside `build()`** (see §5).
- Dataflow: 3 device ops and 18 host ops dead. Witness 318–377 µs.

**QSA ring and QSA compressed (`models/qwen4_exp/common/qsa_cache.py`, overlay `nvidia/qsa.py`), x1 each.**
- 8 host ops (CPU-side checks, `.item()` on CPU) plus **one Triton launch** (`_build_qsa_metadata_kernel`) writing the
  builder's own buffers.
- Dataflow: the host ops (26 over the two) are dead under FULL; the Triton launch is needed. Witness 92–144 µs each.

**FA half (`Qwen4ExpQSAMetadataBuilder`), x1.**
- Pure Python object build, no device op. Nothing of it reaches FULL replay. Witness 20–33 µs.

**Framework (`prepare_attn` + `build_attn_metadata`).**
- 21–22 `CommonAttentionMetadata`, 20–21 device `select` views of the block tables, ~25 host slices, 1 CPU `max`, CPU
  `zeros`/`copy_` for `is_prefilling`.
- Dataflow: 18–19 device views and 29–30 host ops dead under FULL. Witness: the remainder 0.15–0.35 ms.

**H2D copies and syncs.**
- `prepare_attn` itself does no H2D copy; `query_start_loc` and `seq_lens` were staged by `prepare_inputs`.
- The meta-device pass raised no "cannot read data" error. Every `_local_scalar_dense` is on a CPU tensor. **No host
  sync.**

### 1.4 Non-FULL paths

- PIECEWISE and eager (prefill, mixed batches, `cudagraph_mode != FULL`) **do** read the returned dict through the
  forward context: GDN `non_spec_state_indices_tensor`, `has_initial_state`, token indices and the prefill fields.
- Any option must therefore gate on `cudagraph_mode == FULL` and an all-decode batch, and fall back to stock otherwise.
- Capture (`for_capture=True`) must stay stock so that graphs are captured on stock buffer addresses.
- The native lane builds through `attn_utils.build_attn_metadata` with private cloned builders
  (`patches/s5_engine/lane_chunk.py:134,685`). It never calls `model_state.prepare_attn`.

## 2. CPU witness over the real builder code

`tools/attn_meta_witness.py`, launched by `tools/attn_meta_selfrun.py`.

**Container.**
- Pinned image `sha256:ad067db5…cd0b` (torch 2.13.0+cu130, triton 3.7.1).
- `--cpus 4 --network none`, no GPU device request (asserted), nice 10 inside, read-only root.
- Hub CPU: AMD Threadripper PRO 3975WX.

**What runs.**
- The real `VllmConfig` from `reference/model-config.json` with box values (max_num_seqs 10, batched tokens 1024, block
  4048, align, FULL_DECODE_ONLY, capture sizes up to 24, PP 24/24, TP2, no spec).
- The real `kv_cache_utils` grouping, real `AttentionGroup`, real `create_metadata_builders`, real InputBatch and real
  `Qwen4ExpModelState.prepare_attn`.
- The overlay `qsa.py` builder class from the r6 build that the box runs (the image's copy differs; §5).

**Workload.**
- 8 requests: `h65k` (65,000 + 997·i), `h110k` (110,000 + 997·i), and `mix`.
- Two passes. `cpu` times the step. `meta` (meta device) counts ops: a device op is a GPU launch on the box, and the
  pass also classifies host ops and views and detects syncs. A backward dataflow pass runs from builder-buffer writes
  and Triton args.

### 2.1 Declared fixture leaves (witness.json `leaves`)

| leaf | fixture | real (box) |
|---|---|---|
| device | `cpu` / `meta` | CUDA sm86; launch cost not measured, counted then modelled |
| Triton launches | launcher stub records (kernel, grid) | real JIT launch (~20–60 µs host each, modelled) |
| VllmConfig | real class, box values; cudagraph_mode set to FULL_DECODE_ONLY because the CPU platform resolves NONE | platform resolves FULL_DECODE_ONLY |
| KV groups / attention groups | real grouping + real builders, CPU tensors | same code, CUDA tensors |
| model_state | `object.__new__(Qwen4ExpModelState)` + the attributes `prepare_attn` reads | full init |
| InputBatch / block tables / slot mappings | real InputBatch with synthetic tables; distinct block ids per group | scheduler-produced |
| `is_arch_support_pdl` | patched False (CPU) | queried on CUDA |
| cudagraph_mode | FULL passed to `prepare_attn` | FULL on decode replay |
| `Qwen4ExpQSAMetadataBuilder` | overlay r6 class text exec'd into the image module | overlay r6 |

### 2.2 Results (w3 PASS rc 0, `results/hub/attn-meta/w3/witness.json`)

Step = one `prepare_attn`. 3,000 iterations; cold = a cache-thrash pass between calls.

| case | warm p50 | warm p90 | cold p50 | GDN x18 | PLE | QSA ring | QSA compr. | FA |
|---|---|---|---|---|---|---|---|---|
| stage0 h65k | 2.56 ms | 3.35 | 3.18 | 1.68 ms (93 µs/group) | 318 µs | 117 | 94 | 26 |
| stage0 h110k | 2.52 | 3.11 | 3.26 | 1.77 (98) | 346 | 129 | 99 | 28 |
| stage0 mix | 2.74 | 3.13 | 3.19 | 1.77 (98) | 377 | 139 | 101 | 33 |
| stage1 h65k | 2.25 | 2.64 | 2.84 | 1.67 (93) | – | 114 | 93 | 20 |
| stage1 h110k | 2.24 | 2.94 | 2.83 | 1.76 (98) | – | 139 | 100 | 27 |
| stage1 mix | 2.14 | 2.55 | 2.74 | 1.81 (101) | – | 144 | 103 | 29 |

**GDN per-build split (cProfile, stage1 mix).** About 33 µs of own Python (slicing, the decode/prefill split,
dataclass) and about 64 µs in `mamba_get_block_table_tensor`.

**Op micro-bench (CPU, per tiny aten op, µs).** add 4.7, floordiv 5.9, copy_ 3.5, empty fill 3.9, arange 2.7,
gather 2.6, select 2.4, slice 2.1.

**History length barely matters (65k vs 110k).** The cost scales with the number of groups and ops, not with
tokens.

### 2.3 Gap to the box's 3.7–3.9 ms (modelled, not measured)

- The cold CPU step is 2.7–3.3 ms. Add the CUDA launch premium over a CPU tiny op: about 162–177 device kernels at
  2–5 µs each, or 0.32–0.87 ms. Add 2 Triton launches at 20–60 µs each, and the `is_arch_support_pdl` and stream
  queries. Total: **≈ 3.1–4.3 ms**, which brackets 3.7–3.9.
- Not explained: the box's stage 0 minus stage 1 difference is only ~0.1 ms, while the witness shows +0.3–0.5 ms for
  PLE on stage 0. Either the box CPU runs the PLE host logic faster than the hub, or PLE's host work overlaps something
  on the box that the seam does not split.
- The box CPU model is not in the local records, so the per-op scale factor is unknown. **Phase 2 must measure the
  device-op count and the per-step host time on a GPU (VM), not assume them.**

## 3. Options, ranked by saving and risk

Savings are host time per stock decode step per rank. "CPU" means measured on the witness. "box" means modelled by
adding the removed launches at 1–4 µs each.

| # | option | saving | risk | bit-identical needs | native lane | memory |
|---|---|---|---|---|---|---|
| **A** | **GDN group-batched decode staging** (FULL, all-decode only) | **CPU −1.6…−1.9 ms** (w4/w5 bench); box ≈ −1.8…−2.5 ms (155 fewer launches) | low–moderate | `non_spec_state_indices_tensor[:R]`, `non_spec_query_start_loc[:R+1]` of every GDN builder, same tensors, same offsets; every GDNAttentionMetadata field; dict order | untouched: no builder field added, stock builders' own buffers only, `build_attn_metadata` unchanged | one `[G,R,W]` int32 stack temp, G=18, R≤24, W=65 (262144/4048 rounded up): ~110 KB worst case per step, freed |
| E | A, but one Triton kernel over a pointer table for all GDN groups | A + ~6 launches (≈20–40 µs box) | moderate (GPU-only to verify; new kernel) | as A | as A | pointer table, 18×8 B |
| F | Build the stock metadata while stage 1 waits on the PP receive (1.6–2.1 ms wall) | up to 1.6–2.1 ms wall on stage 1 only; complementary to A | high: reorders the stage-1 timeline; interacts with the C1 rebind (alpha-next-c1) and the receive path | ordering of buffer writes vs the previous replay (must not overwrite buffers a running graph reads) | none directly | none |
| C | PLE decode fast path (stage 0) | ≈ −0.2…−0.25 ms CPU, stage 0 only | moderate (BaseMamba config-gated staging, `has_initial_states_d`) | `state_indices_tensor_d`, `has_initial_states_d` | PLE builder is cloned for the lane; same rule as A | none |
| B | Skip dead work under FULL: FA build, QSA host checks, framework objects | ≈ −0.2…−0.35 ms CPU | moderate: returned objects become stubs; anything reading the forward context under FULL (hooks, observers, the e3 graph hooks) would see them | QSA Triton writes; nothing else reaches the graph | must not change `build_attn_metadata` (shared) → only via a stock-only wrapper | none |
| D | Skip GDN restaging when unchanged (start index changes once per 4048 tokens) | A + ≈ −0.1…−0.2 ms | high: cache key must cover idx_mapping, block-table rows, R, `seq_lens//4048`, and every other writer of the same buffers (capture, dummy/profile runs); staleness is silent wrong state | as A, plus no stale buffer after any other writer | any shared invalidation state would be new sharing, forbidden | small host mirror |
| G | Async scheduling (build step n+1 during n) | low: the host is the bottleneck, so nothing to hide behind | high | – | – | – |
| H | Re-pack GDN layers into fewer KV groups | none without changing the cache layout: one state per block is the packing's design; still 18 local groups per rank | very high (KV layout, prefix cache, lane) | – | – | – |

**Ranking.** A is first: biggest saving, lowest risk, testable on CPU. F is a separate wall-time lever; do it after
A, and it needs its own design with the C1 rebind owner. C and B are small and could follow A as separate flags. E
only if the VM shows A's 7 remaining ops matter. D, G and H are not recommended.

## 4. Chosen option A: GDN group-batched decode staging

### 4.1 What it does (`tools/attn_meta_gdn_batch_proto.py`, prototype, not packaged)

A drop-in for `MambaHybridModelState.prepare_attn` with the same arguments and result.

1. `eligible_groups` returns None (so the stock path runs unchanged) unless **all** of these hold:
   - `cudagraph_mode == FULL` and not `for_capture`;
   - no spec tokens and `mamba_cache_mode == 'align'`;
   - no recoverssm and no DCP;
   - an all-decode batch: max scheduled tokens 1 and no `is_prefilling` row;
   - every GDN group has exactly one exact-type `GDNAttentionMetadataBuilder` with `use_spec_decode` False,
     `use_full_cuda_graph` True, `R ≤ decode_cudagraph_max_bs` and `num_speculative_blocks == 0`;
   - one common block size;
   - at least 2 such groups and one common block-table width.
2. Stock `prepare_attn` runs for **all other groups** (GDN groups masked to `[]`): PLE, QSA ring/compressed and FA see
   exactly what they saw before.
3. GDN, once:
   - `start = (seq_lens[:R]-1)//bs; start.clamp_(0)`;
   - `idx = start.to(int64).view(1,R,1).expand(G,R,1)`;
   - `states = gather(stack(block_tables[gdn]), 2, idx)`;
   - `_foreach_copy_` into each builder's **own** `non_spec_state_indices_tensor[:R]`;
   - `_foreach_copy_` of `query_start_loc` into each `non_spec_query_start_loc[:R+1]`.
   Stock's two padding fills are empty in this case and are skipped.
4. One `GDNAttentionMetadata` per group with stock's field values (`num_decodes=R`, `num_decode_tokens=T`,
   `num_actual_tokens=T`, tensors = views of the builder buffers, other fields None/0). The dict is rebuilt in stock
   order.

Device ops per step for GDN: 162 kernels → **7** (sub, floor_divide, clamp_, _to_copy, stack, gather, 2 foreach
copies; candidate census in `w5/bench.json`).

### 4.2 Acceptance tests (`harness/test_attn_meta_gdn_batch.py`, run inside the pinned image)

The tests skip on the hub without torch/vllm, so `harness/selftest.py` is unaffected. Container run:
`tools/attn_meta_selfrun.py --mode tests`.

**Equality vs stock on the CPU path.** Builder buffers are poisoned with sentinel −7 / True before each run. The
comparison covers:
- every metadata field and every tensor value;
- the number of distinct metadata objects (layers sharing one object);
- where each tensor lives (builder field + storage offset);
- **every builder persistent buffer in full** (not only `[:R]`);
- dict keys and order.

Cases:
- histories h65k and h110k;
- boundary histories at B−2, B−1, B, B+1, 2B−1, 2B, 16B, 27B+1 (B = 4048);
- single request, 10 requests;
- both stages, with and without `foreach`;
- padding (n < R, `query_start_loc_np` padded to R+1 with num_tokens, as V2 `prepare_inputs` does).

The suite also checks:
- the batched path is really taken: 18 groups, one stack call;
- metadata tensors point into the builder buffers.

**Fallbacks are byte-identical to stock:** NONE and PIECEWISE, a prefill row, an `is_prefilling` row, a spec-decode
builder, a non-align mode, and `for_capture`.

**Default off.** Only the exact string `FLASH_ATTN_META_GDN_BATCH=1` enables it. Unset, '0', '', 'true', 'yes',
'on', ' 1' and '1 ' don't. With the flag unset, the eligibility check is never reached and the output equals stock.

**Mutants: all must be caught.**

| mutant | how it is caught |
|---|---|
| start-without-minus-one | metadata |
| no-clamp | exception |
| rotated-destinations | metadata |
| qsl-not-written | metadata |
| fresh-tensor-in-metadata | metadata |
| num-decode-tokens-unpadded | metadata |
| dict-order-rest-first | keys/order |
| flag-default-on | default-off |
| ignores-capture-gate | capture gate |

**Runs.**
- **w4 FAIL rc 1** (kept). It had a fixture bug: padding didn't pad `query_start_loc_np`, so stock PLE raised a size
  mismatch. Its first order mutant was equivalent (GDN groups already come first in stock order).
- **w5 PASS rc 0** (commit de488405): 14 tests OK, 9/9 mutants caught.

**Bench (w5, CPU p50; stock → candidate foreach / loop):**

| case | stock | foreach | loop |
|---|---|---|---|
| stage0 h65k | 2.59 ms | 0.96 | 0.94 |
| stage0 h110k | 2.83 | 1.03 | 0.97 |
| stage1 h65k | 2.26 | 0.60 | 0.60 |
| stage1 h110k | 2.17 | 0.66 | 0.63 |

(w4: stock 2.29–2.94 → 0.58–1.07.) `_foreach_copy_` vs a loop makes no difference on CPU. On CUDA, foreach fuses the
18 copies into about one launch per list (to be confirmed on the VM).

### 4.3 Phase 2 (not done here; needs the lead's go)

- **Package.** Hook only the stock step: a `Qwen4ExpModelState.prepare_attn` override that calls the batched path
  when the flag is on, else `super()`.
  - It composes with the `engine_offers` wrapper (`input_attention/attn_metadata` wraps the bound
    `self.model_state.prepare_attn`) and the anatomy seam.
  - It must not touch `attn_utils.build_attn_metadata`, the builders, or the lane's private builders. It adds no
    builder fields.
- **VM GPU checks (one run, through the queues).**
  1. A torch-profiler device-op count per decode step, flag off vs on (expect GDN 162 → ~7).
  2. A byte compare of all GDN builder buffers after each step, flag off vs on, over a decode run that crosses block
     boundaries and admits/finishes requests.
  3. Token-identical greedy output, flag off vs on.
  4. `prep_attn_meta` host time per step via the anatomy seam.
  5. Capture and dummy runs stay stock (`for_capture`).
- **Box (later, operator/lead).** The decode tok/s effect depends on whether host prepare stays the critical path after
  the C1 rebind. Do not promise tok/s from the host saving alone.

## 5. Engine-relevant findings (for the lead)

1. **Under FULL decode replay the returned metadata dict is dead.** Only the builders' in-place persistent-buffer
   writes and the QSA Triton writes reach the graph. The dict matters for PIECEWISE and eager only.
2. **No host sync** in `prepare_attn` on the stock decode step.
3. **PLE's FULL staging depends on the config's `compilation_config.cudagraph_mode`, read inside `build()`.** It does
   not depend on the `cudagraph_mode` argument. A config that resolved NONE would leave `state_indices_tensor_d`
   unstaged under a FULL replay. Our CPU platform did that (w1), and the box resolves FULL_DECODE_ONLY.
   **Open risk** (lead 22:3x): no action while the box config is FULL; any NONE or PIECEWISE configuration must
   re-check it. Option A does not touch PLE (PLE runs inside the stock call), so A neither fixes nor worsens it.
4. **The image's `nvidia/qsa.py` `Qwen4ExpQSAMetadataBuilder` differs from the r6 overlay's** (witness
   `overlay_qsa_builder_class_identical: False`). The witness used the overlay class.
5. The box's stage 0 vs stage 1 `prep_attn_meta` differ by only ~0.1 ms, against +0.3–0.5 ms for PLE on the CPU
   witness (§2.3). This is unexplained.

## 5b. Phase 2 (lead GO 22:3x): the overlay, its tests, the VM run

**Code** (claude/attn-meta, not packaged; packaging is Tue daytime on the base the lead picks):

- `patches/s5_engine/attn_meta_batch.py` is option A as overlay code. `install(runner)` sets one **instance**
  attribute, `runner.model_state.prepare_attn`, and never touches a class.
  - **Switch.** `FLASH_ATTN_META_GDN_BATCH`: unset or `0` is off, `1` is on, any other value refuses. When off,
    install returns before importing torch or vLLM.
  - **Install refusals.** It refuses a second install, an instance that already carries a `prepare_attn` (a seam
    installed first would be bypassed), a class that overrides `prepare_attn`, and a model state that is not a
    MambaHybridModelState.
  - **Counters.** One fixed table: batched steps, plus stock fallbacks by reason (13 fixed reasons).
  - **Call-site order for packaging.** Install it before `handler_anatomy` installs its seams. The anatomy
    recorder wraps whatever the instance holds, and `engine_offers` looks the attribute up at each call. The
    capture path (`cudagraph_utils`, mode NONE, `for_capture`) reaches stock through the wrapper.
- `tools/attn_meta_cases.py` holds the shared device-generic fixture and checks. It has its own declared leaves
  on top of the witness's.
- `harness/test_attn_meta_overlay.py` runs against the overlay code itself. It covers:
  - every w5 equality case and fallback, with fallback reasons asserted;
  - a 48-step decode sequence with block crossings, admissions, finishes and padding;
  - install and default-off, including a subprocess check that nothing is imported when off;
  - seam composition;
  - **native isolation:** the lane's private clones (real `lane_chunk.private_attention_groups`) share no storage
    with the stock builders and stay byte-untouched over overlay steps with the switch on. Two native mutants
    must be caught: an overlay that writes into the clones, and a clone function that shares the GDN buffers;
  - 14 overlay mutants.
- `tools/attn_meta_vm_probe.py` is one probe for both devices, CPU on the hub and CUDA on VM <vm-id>. Its phases:
  - **equality;**
  - **sequence:** a captured CUDA graph per fixture reads every GDN persistent buffer and is replayed after
    every step;
  - **opcount:** CUDA profiler kernel, memcpy and memset records plus runtime launch/sync calls; the meta census
    on CPU;
  - **timing:** per call, per builder, and per tiny device op, so the launch premium is measured;
  - **native.**

  Greedy tokens are **not covered** (no model fits one 3090), and that leaf is declared. The VM wrapper and
  stage/run/fetch tools are `tools/attn_meta_vm_run_gpu.py` and `tools/attn_meta_vm_input.py`.

Results are in §6 and `tmp/attn-meta/READY-pre-vm.md`.

### VM <vm-id> run attn-meta-vm-a (03 Nov 23:5x CET, one RTX 3090, CUDA, `results/vm<vm-id>/attn-meta-vm-a`)

**Verdict: FAIL rc 1, as recorded.** The only failures are `opcount host-sync {'cudaDeviceSynchronize': 1}` on both
stages. That count is identical for stock and overlay (2 per profiled call, minus the probe's own flush), and is
most likely the profiler's stop flush. It is a probe counting fault (HARNESS), not a sync inside `prepare_attn`.
Not rerun (one run). Every other check passed:

- **Equality:** 18 cases, 0 differences, on real CUDA builders with real Triton.
- **Sequence:** 48/48 steps per stage. A **captured CUDA graph** reading all 36 GDN buffers was replayed after
  every step and read identical bytes, stock vs overlay. There were 9 block crossings per stage, n in {6, 7, 8},
  and all 48 steps batched.
- **Native:** 154–160 clone tensors, untouched, 0 shared storage, stock written, 3/3 batched.
- **Memory:** own peak 334 MiB (the CUDA context); torch max allocated 2.4 MB.

**Device records per call** (CUDA profiler):

| stage | stock | overlay | Triton launches (both) |
|---|---|---|---|
| 1 | 128 kernels + 36 memcpy | **10 kernels**, 0 memcpy | 2 |
| 0 | 138 kernels + 38 memcpy | **20 kernels + 2 memcpy** | 2 |

**Host µs per call** (p50, device synchronised between calls), on the VM's CPU, not the box's:

| | stock | overlay | saving |
|---|---|---|---|
| stage 1 | 4,891–4,907 | **796–830** | −4.1 ms, −83 % |
| stage 0 | 5,253–5,317 | **1,322–1,356** | −3.9 ms |

**Per builder** (stock, CUDA):
- GDN: **226–232 µs per build ×18 = 4.06–4.18 ms**, about 85 % of the step;
- PLE: 564 µs;
- QSA ring: 207–233 µs;
- QSA compressed: 148–156 µs;
- FA: 22–27 µs.

**Launch premium, measured.** A tiny device op costs **12–15 µs** host time on CUDA here (add 12.3, floordiv
14.7, copy_ 12.7, gather 12.6, arange 13.3; an empty fill 5.9; a view 3.1). The hub's CPU tensors cost 2–6 µs per
op. The modelled 2–5 µs premium per op was too low for this host. The GDN per-build cost on CUDA (226 µs vs ~95 on
CPU) is mostly that premium: 9 launches × ~13 µs.

**Box estimate** (modelled from the VM ratio, not measured). The box measured stock at 3.7–3.9 ms, against the VM's
4.9 ms. Scaling the overlay by the same factor gives about 0.6–0.7 ms on stage 1, a saving of about 3 ms per step.

**Finding 5 is still open.** On the VM, stage 0 minus stage 1 is +0.36–0.43 ms, with PLE at 0.56 ms. That is
consistent with the hub witness and not with the box's ~0.1 ms. The VM does not explain the box's small stage
difference.

## 6. Records

| run | what | verdict |
|---|---|---|
| w1 | witness v1 | PASS rc 0, but flawed: the CPU platform resolved cudagraph_mode NONE, so PLE's FULL staging looked dead. Kept. |
| w2 | witness v2 | FAIL rc 1: import cycle `nvidia.qsa` ↔ `nvidia.model`. Kept. |
| w3 | witness v2 + import fix (0338e1a6) | PASS rc 0. The numbers above. |
| w4 | tests + bench (789737b5) | FAIL rc 1: fixture padding bug and an equivalent mutant. Bench rc 0. |
| w5 | tests + bench (de488405) | PASS rc 0 |

Container START/END CET times and the load at each are in `tmp/attn-meta/STATUS.md`. Sources executed in the image
are recorded as sha256 lists (`sources-sha256.txt`). All workload values are synthetic numbers.
