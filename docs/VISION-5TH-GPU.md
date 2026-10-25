# Vision encoder on a 5th RTX 3090 over the network — preparation (not implemented)

Operator, 24 Oct 2026: keep images out of the serving config while they cost too much performance or VRAM; investigate
a 5th 3090 reached over the network that runs the vision encoder, with the embeddings sent to the language model.
This note gathers what is known, what the engine already supports, and what must be measured before deciding.
Nothing here is deployed.

## What is known (measured or read from the model, 24 Oct)

| Item | Value | Source |
|---|---|---|
| Vision tower weights | 0.84 GiB BF16, ~0.42 GiB per GPU on pipeline stage 0 (first-stage-only allocation) | README, item 5 runs |
| Tower shape | 27 blocks, width 1152, patch 16, spatial merge 2, output width 2560 | model config.json |
| LM hidden size | 2560 (48 layers) | model config.json |
| Tokens per image | one LM token per 32 × 32 pixels: 1024² → 1,024 tokens; 1920 × 1080 → ~2,040 | patch 16 × merge 2 |
| Embedding bytes per image | tokens × 2560 × 2 B = 5 KiB per token: 1024² → 5 MiB; 1080p → ~10 MiB | derived |
| Transfer time at 10 GbE (~1.1 GB/s usable) | ~5 ms (1024²), ~10 ms (1080p) per image | derived; crazydiamond nic0/nic1 are 10 Gb/s (AQC113) |
| Interference today (tower on the serving GPUs) | worst p99 decode gap 22.9 → 221.8 ms during four image requests | item 5 vision-interference.json (review: not a PASS) |
| VRAM headroom | stage 1 peaked at 24.38 of 24.58 GB with the tower allocated | README |
| Isolation blocker | a reused client UUID returned the first image's content | item 5 vision-uuid.json |

## What the engine already supports (vLLM 0.29.1rc1 image)

1. **Encoder disaggregation (EPD).** `vllm/distributed/ec_transfer`: an encoder-cache connector with roles
   `ec_producer` / `ec_consumer` / `ec_both` (`--ec-transfer-config`). Implementations in the image:
   `ECMooncakeConnector` (network transfer through Mooncake, TCP or RDMA), `ECExampleConnector` (shared storage) and a
   CPU variant. The scheduler already consults `ec_connector.ensure_cache_available(...)` before scheduling a
   multimodal request, so the language-model engine can run as a consumer that never runs the tower.
   Topology: an encoder-only vLLM instance (producer) on the 5th GPU; the existing engine (consumer, tower not loaded
   or not run) receives embeddings keyed by the multimodal hash.
2. **Client-side embeddings** (`enable_mm_embeds`): a client may send precomputed image embeddings instead of pixels.
   That would put the encoder behind the gateway instead of inside vLLM. It also lets any client inject arbitrary
   embeddings, so it is acceptable only behind a trusted gateway.

## Decisions it needs (operator)

- Which machine hosts the 5th 3090 and its link to crazydiamond (10 GbE direct, or through the switch).
- EPD inside vLLM (recommended: one cache-identity path, the engine's own scheduler waits for the embedding) vs a
  gateway-side encoder service.

## Measurements to take before deciding (all on the existing box, no new hardware)

1. Tower encode time per image on one 3090, for 512², 1024², 1080p screenshots and 4k photos: alone, and while the
   GPU does nothing else (the 5th GPU's case). Batch-of-4 throughput.
2. Embedding size actually produced per image (check the merger output dtype and token count against the table).
3. What the language model still pays per image with the tower elsewhere: the image tokens' prefill (1–2k tokens on
   the history) — this stays on the serving GPUs and is the same cost as a text append of that length.
4. Consumer-side memory: the encoder cache on the consumer (`--mm-encoder-cache` / encoder cache size) in VRAM or host.
5. Mooncake transfer on this LAN: latency for 5 and 10 MiB payloads over TCP between crazydiamond and the candidate
   host (no RDMA NICs here as far as known).

## Correctness gates (before any image traffic in production)

- Cache identity is the multimodal hash computed by the engine from content + processor settings; client-supplied
  UUIDs are stripped or rejected at the gateway (today's blocker). With EPD the producer and consumer must agree on
  that hash, so it must be the content-derived one, never a client value.
- vision_test.py quality/bound/followup/isolation/uuid on the EPD topology, across users and routes.
- An image latency budget in the interference test (per-stream relative p99, agreed absolute ceiling) — the current
  test only checks errors and progress.
- Encoder host down or slow: requests must fail or wait cleanly; text traffic unaffected (fault-injection test).

## Expected effect

The tower's 0.42 GiB per stage-0 GPU returns to the KV pool / retained history, and the 22.9 → 221.8 ms p99
interference from encoding on the serving GPUs disappears; what remains is the image tokens' prefill, which the
short-first scheduler and S5 treat like any other short append.

## q14b measurement (25 Oct 04:24 UTC, tower only, one RTX 3090 x16, BF16, no LM, no network)

Weights 333 tensors, 897,862,112 B (0.84 GiB) on one GPU, load 2.8 s. Single run after one warm-up (not a distribution).

| Input | Visual tokens | Encode (CUDA events) | Preprocess (CPU) | Peak reserved | Embedding payload | Ideal transfer 1 / 10 Gb/s (one copy) |
|---|---:|---:|---:|---:|---:|---:|
| 512² | 256 | 14.7 ms | 10.3 ms | 0.93 GiB | 1.25 MiB | 10.5 / 1.0 ms |
| 1024² | 1,024 | 66.4 ms | 30.4 ms | 1.10 GiB | 5.0 MiB | 41.9 / 4.2 ms |
| 1536² | 2,304 | 205.1 ms | 70.8 ms | 1.45 GiB | 11.25 MiB | 94.4 / 9.4 ms |

Reading: on 10 GbE the network adds ~5–7 % to the encode time; on 1 GbE it adds ~60 % (more if both TP ranks
need their own copy, or with base64 framing, which is +33 %). The visual tokens still have to be prefilled by the
language model on the four main GPUs. The remote tower takes the encoder's VRAM and compute off them, not the
prefill of the visual tokens. Not yet measured: the remote route, language decode, image quality.
