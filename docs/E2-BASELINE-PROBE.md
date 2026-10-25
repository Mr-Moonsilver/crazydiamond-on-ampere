# E2 baseline probe: runtime facts for the serial retention candidate (spec from Codex, 25 Oct 2026)

Read-only probe to run in the engine-baseline slot (overlay-v16, engine in a test slot, never production). It feeds the
E2-serial design (`reviews/codex-20261025s-s5-engine-design.md` §4–5, §7). Observations are helpful, not a reason to
delay the ordinary numerical baseline (teacher-forced batch invariance), which must stay uninstrumented. If the probe
needs an overlay, identity-pin that probe overlay separately.

**Common fields (every record):** `schema`, `case_id`, `rank`, `pp_rank`, `tp_rank`, `gpu_uuid`, `pci_bus_id`,
`sample_phase`, `monotonic_ns`, `engine_manifest_sha256`.

**Per QSA layer (inventory, once):** `layer_name`, `layer_index`, `group_id`, `host_shape`, `host_stride_bytes`,
`dtype`, `page_tokens`, `page_bytes`, `host_is_pinned`, `opaque_allocation_id`, `host_storage_bytes`,
`gpu_placeholder_bytes`, `arena_bytes`. No raw addresses.

**Mapping samples (CPU scheduler metadata only, no payloads):** `request_alias`, `scheduler_step`, `chunk_start`,
`chunk_end`, `seq_len`, `admitted_cached_tokens`, ordered valid `physical_block_ids` per group,
`valid_tokens_per_page`, `copy_pairs` [{`src`, `dst`, `prefix_tokens`}]. Capture: a fresh 200k prompt at its first
chunk, a page-boundary chunk and its final chunk; a long warm-prefix append at its first chunk and at a page
crossing. Ordered block ids matter (the contiguity histogram is derived offline). Unsupported fields = `null`.

**Worker memory (direct, not log-derived; after capture, after warm-up, after drain):** `allocated_bytes`,
`reserved_bytes`, `peak_allocated_bytes`, `peak_reserved_bytes`, `cuda_free_bytes`, `total_bytes`, with timestamps.
`memory_ledger.py`'s log-derived fields can be stale.

**Optional, quiescent after an 8-stream warm-up:** row cache `capacity_bytes`, `valid_slots_per_owner`
(dummy owner excluded), `counters_delta`, `active_owner_ids` separate from stale retained ownership.

**Rules:** never synchronise GPU reads inside the timed baseline; `teacher_forced.py` asserts cold prefills with fresh
salts, so warm-vs-cold needs `prefix_warm_cold.py` separately.
