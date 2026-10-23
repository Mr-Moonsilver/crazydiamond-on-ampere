#!/usr/bin/env python3
"""Prototype: per-request VRAM cache of recently selected QSA host-KV rows (decode only).

Builds patched copies of the Flash image's qsa.py and ops/qsa.py:
  * bind_kv_cache allocates, per QSA layer, cache_kv [R, SLOTS, kv_heads, 2*head_dim] BF16 and
    tags [R, SLOTS] int32 (-1 = empty) when QSA_ROWCACHE_SLOTS > 0 (R = QSA_ROWCACHE_REQS).
  * decode: a fill kernel copies only the selected rows whose slot does not already hold that
    token from host RAM into the cache (direct-mapped, slot = token % SLOTS, CAS-claimed);
    the attention kernel then reads hits from VRAM and anything else (collisions, requests
    beyond R) from host RAM as before, so results never depend on the hit rate.
  * tags are keyed by batch slot: a request in its first chunk (seq_len == query_len) has its
    slot's tags cleared, so a newly admitted request never sees a previous owner's rows.
usage: make_rowcache.py qsa.image.py ops_qsa.image.py out_qsa.py out_ops_qsa.py
"""
import sys

qsa, ops = open(sys.argv[1]).read(), open(sys.argv[2]).read()


def sub(text, old, new, count=1, first_only=False):
    assert text.count(old) == count, (old[:80], text.count(old))
    return text.replace(old, new, 1 if first_only else -1)


# ---------------- qsa.py: allocation, flush, plumbing ----------------
qsa = sub(qsa, """        self._qsa_host_kv = allocate_registered_host_tensor(shape, torch.bfloat16)
        self._qsa_gpu_slots = kv_cache
        self.kv_cache = get_accelerator_view_from_cpu_tensor(self._qsa_host_kv)
""", """        self._qsa_host_kv = allocate_registered_host_tensor(shape, torch.bfloat16)
        self._qsa_gpu_slots = kv_cache
        self.kv_cache = get_accelerator_view_from_cpu_tensor(self._qsa_host_kv)
        rc_slots = int(os.environ.get("QSA_ROWCACHE_SLOTS", "0"))
        self._qsa_rowcache = None
        if rc_slots > 0:
            rc_reqs = int(os.environ.get("QSA_ROWCACHE_REQS", "8")) + 1  # + dummy row
            dev = kv_cache.device
            row_last = torch.zeros(rc_reqs, dtype=torch.int64, device=dev)
            row_last[-1] = 2**62  # the dummy row is never chosen for a request
            self._qsa_rowcache = (
                torch.empty((rc_reqs, rc_slots, self.num_kv_heads, 2 * self.head_dim), dtype=torch.bfloat16, device=dev),
                torch.full((rc_reqs, rc_slots), -1, dtype=torch.int32, device=dev),   # tags
                torch.zeros(2, dtype=torch.int64, device=dev),                      # [selected rows, misses]
                torch.empty((rc_reqs, rc_slots), dtype=torch.int32, device=dev),    # per-step claims
                torch.full((rc_reqs,), -1, dtype=torch.int32, device=dev),          # row -> owning request
                row_last,                                                           # row -> last step used
                torch.full((1024,), -1, dtype=torch.int32, device=dev),             # batch index -> row
                torch.zeros(1, dtype=torch.int64, device=dev),                      # step counter
            )
            # Request identity for the row cache: the model runner (gpu_model_runner patch) writes a
            # unique id per live request, in batch order, into this per-process buffer every step.
            # Fixed address, so graphs replay with the current ids.
            import vllm.forward_context as _fc
            if getattr(_fc, "QSA_REQ_UID", None) is None:
                _fc.QSA_REQ_UID = torch.full((1024,), -2, dtype=torch.int32, device=dev)
            init_logger(__name__).info(
                "QSA row cache %s: %d requests x %d slots, %.1f MiB VRAM",
                self.layer_name, rc_reqs, rc_slots,
                self._qsa_rowcache[0].numel() * 2 / 2**20)
""")
qsa = sub(qsa, """        if staged is None:
            qsa_sparse_paged_attention(
                query[:num_tokens],
                key_cache,
                value_cache,
                logical_indices,
                attn_metadata.block_table,
                token_to_req,
                use_prefill_config,
                output[:num_tokens],
            )
            return output
""", """        rowcache = getattr(layer, "_qsa_rowcache", None)
        if rowcache is not None and not getattr(__import__("vllm.forward_context").forward_context,
                                                 "QSA_REQ_UID_WRITER", False):
            # Without the runner patch every request would carry the same id: never cache then.
            rowcache = None
        if rowcache is not None:
            # Cache rows belong to requests, not batch indices: with pipeline parallelism two
            # micro-batches alternate, so batch index i is a different request on every other step.
            # A request is identified by the id the model runner assigns it (QSA_REQ_UID, batch
            # order). NOT by its first physical KV page: under prefix caching sessions with the same
            # prompt start share that page, and keying on it let them read each other's rows
            # (cross-session leak, 23 Oct 2026). Each step: find the row owned by each request; give
            # new requests (or ones in their first chunk) the least recently used row and clear its
            # tags. Fixed-shape tensor ops only, so this also replays inside graphs.
            import vllm.forward_context as _fc
            _, tags, _, _, row_owner, row_last, row_map, step = rowcache
            rows = row_owner.shape[0] - 1  # last row is the dummy
            num_reqs = min(attn_metadata.seq_lens.shape[0], attn_metadata.block_table.shape[0], rows)
            query_lens = attn_metadata.query_start_loc[1:num_reqs + 1] - attn_metadata.query_start_loc[:num_reqs]
            owner = _fc.QSA_REQ_UID[:num_reqs]
            active = query_lens > 0
            fresh = active & (attn_metadata.seq_lens[:num_reqs] == query_lens)
            matches = (row_owner[None, :rows] == owner[:, None]) & active[:, None]
            has = matches.any(dim=1) & ~fresh
            match_row = matches.to(torch.int32).argmax(dim=1)
            used = torch.zeros(rows + 1, dtype=torch.int64, device=owner.device).scatter_reduce_(
                0, match_row.long(), has.long(), "amax")
            order = torch.argsort(row_last + used * 2**61)  # unused rows, oldest first; dummy last
            need = active & ~has
            pick = order[(torch.cumsum(need.long(), 0) - 1).clamp(min=0)]
            row = torch.where(has, match_row.long(), torch.where(need, pick, torch.full_like(pick, rows)))
            step.add_(1)
            new_rows = torch.where(need, row, torch.full_like(row, rows))
            tags.index_fill_(0, new_rows, -1)
            row_owner.index_put_((new_rows,), owner)
            row_last.index_put_((row,), step.expand(num_reqs))
            row_last[rows:].fill_(2**62)
            row_owner[rows:].fill_(-1)
            tags[rows:].fill_(-1)
            row_map[:num_reqs].copy_(torch.where(active, row, torch.full_like(row, -1)).to(torch.int32))
            if os.environ.get("QSA_ROWCACHE_DEBUG") and layer.layer_name.split(".")[-3] == "3":
                dbg = getattr(layer, "_qsa_dbg_steps", 0)
                if dbg < 40:
                    layer._qsa_dbg_steps = dbg + 1
                    from vllm.logger import init_logger
                    init_logger(__name__).warning("QSA_ROWCACHE_DEBUG step=%d owner=%s row=%s has=%s",
                                                  dbg, owner.tolist(), row_map[:num_reqs].tolist(), has.tolist())
        # Hit counters are read back (a GPU sync) only while /work/rowcache.report exists, and only on
        # the first QSA layer of each stage; reading them on every layer and prefill step cost ~15-20 %
        # of prefill throughput.
        if (rowcache is not None and use_prefill_config and layer.layer_name.split(".")[-3] in ("3", "27")
                and os.path.exists("/work/rowcache.report")):
            selected, misses = rowcache[2].tolist()
            if selected:
                from vllm.logger import init_logger
                init_logger(__name__).warning(
                    "QSA_ROWCACHE_HITS layer=%s selected=%d misses=%d hit_rate=%.3f",
                    layer.layer_name, selected, misses, 1 - misses / selected)
            rowcache[2].zero_()
        # A/B switch for experiments: the flush above still runs while the cache is off.
        if rowcache is not None and os.path.exists("/work/rowcache.off"):
            rowcache = None
        if staged is None:
            qsa_sparse_paged_attention(
                query[:num_tokens],
                key_cache,
                value_cache,
                logical_indices,
                attn_metadata.block_table,
                token_to_req,
                use_prefill_config,
                output[:num_tokens],
                rowcache=None if use_prefill_config else rowcache,
            )
            return output
""")

# ---------------- ops/qsa.py: kernels ----------------
# 1. attention kernel: extra args + cache-aware loads
ops = sub(ops, """    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,""", """    output_ptr,
    cache_kv_ptr,
    tags_ptr,
    row_map_ptr,
    stride_c_req,
    stride_c_slot,
    stride_c_head,
    num_cache_reqs,
    stride_q_row,
    stride_q_head,
    stride_k_block,""", count=2, first_only=True)  # first = the decode split-k kernel (line ~18)
ops = sub(ops, """    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)""", """    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CACHE_SLOTS: tl.constexpr = 1,
    USE_CACHE: tl.constexpr = False,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)""")
old_loads = """        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(query, keys)"""
new_loads = """        key_ptrs = (
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None]
        )
        value_ptrs = (
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :]
        )
        if USE_CACHE:
            # One load per row either way: cached rows point into VRAM, the rest keep their
            # host-mapped (UVA) address, so shared-memory use matches the uncached kernel.
            mapped = tl.load(row_map_ptr + safe_request)
            cache_req = tl.minimum(tl.maximum(mapped, 0), num_cache_reqs - 1)
            slot = safe_token % CACHE_SLOTS
            in_cache_range = (request >= 0) & (request < num_requests) & (mapped >= 0) & (mapped < num_cache_reqs)
            tag = tl.load(tags_ptr + cache_req * CACHE_SLOTS + slot,
                          mask=valid & in_cache_range, other=-1)
            hit = valid & in_cache_range & (tag == logical_token)
            cache_row = cache_kv_ptr + cache_req * stride_c_req + slot * stride_c_slot + kv_head * stride_c_head
            key_ptrs = tl.where(hit[None, :], cache_row[None, :] + dim_offsets[:, None], key_ptrs)
            value_ptrs = tl.where(hit[:, None], cache_row[:, None] + HEAD_DIM + dim_offsets[None, :], value_ptrs)
        keys = tl.load(key_ptrs, mask=valid[None, :], other=0.0)
        values = tl.load(value_ptrs, mask=valid[:, None], other=0.0)
        scores = tl.dot(query, keys)"""
ops = sub(ops, old_loads, new_loads)

# 2. fill kernel, inserted before the merge kernel
fill_kernel = '''
@triton.jit
def _qsa_rowcache_select(
    indices_ptr, block_table_ptr, token_to_req_ptr, row_map_ptr, row, tile, stride_indices_row, stride_table_req,
    num_requests, num_cache_reqs, num_cache_blocks,
    TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr,
    CACHE_SLOTS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    mapped = tl.load(row_map_ptr + safe_request)
    cache_req = tl.minimum(tl.maximum(mapped, 0), num_cache_reqs - 1)
    valid_count = tl.minimum(tl.load(indices_ptr + row * stride_indices_row + TOPK), TOPK)
    columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    token = tl.load(indices_ptr + row * stride_indices_row + columns, mask=columns < valid_count, other=-1)
    safe_token = tl.maximum(token, 0)
    logical_page = safe_token // PAGE_SIZE
    valid = ((request >= 0) & (request < num_requests) & (mapped >= 0) & (mapped < num_cache_reqs)
             & (token >= 0) & (logical_page < PAGE_TABLE_WIDTH))
    physical_page = tl.load(
        block_table_ptr + safe_request * stride_table_req + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
        mask=valid, other=-1)
    valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
    slot = safe_token % CACHE_SLOTS
    return token, valid, cache_req, slot, tl.maximum(physical_page, 0).to(tl.int64), safe_token % PAGE_SIZE


@triton.jit(do_not_specialize=["num_requests", "num_cache_reqs", "num_cache_blocks"])
def _qsa_rowcache_claim_kernel(
    indices_ptr, block_table_ptr, token_to_req_ptr, row_map_ptr, tags_ptr, claim_ptr,
    stride_indices_row, stride_table_req, num_requests, num_cache_reqs, num_cache_blocks,
    TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr,
    CACHE_SLOTS: tl.constexpr, BLOCK_N: tl.constexpr,
) -> None:
    # Phase 1 of 2: one winner per slot and step. A slot already holding a selected token is
    # kept (priority KEEP); otherwise the largest selected token wins. claim is reset to -1 first.
    token, valid, cache_req, slot, _, _ = _qsa_rowcache_select(
        indices_ptr, block_table_ptr, token_to_req_ptr, row_map_ptr, tl.program_id(0), tl.program_id(1),
        stride_indices_row, stride_table_req, num_requests, num_cache_reqs, num_cache_blocks,
        TOPK, PAGE_SIZE, PAGE_TABLE_WIDTH, CACHE_SLOTS, BLOCK_N)
    offset = cache_req * CACHE_SLOTS + slot
    tag = tl.load(tags_ptr + offset, mask=valid, other=-1)
    priority = tl.where(tag == token, 2147483647, token)
    tl.atomic_max(claim_ptr + offset, priority, mask=valid)


@triton.jit(do_not_specialize=["num_requests", "num_cache_reqs", "num_cache_blocks"])
def _qsa_rowcache_copy_kernel(
    k_cache_ptr, v_cache_ptr, indices_ptr, block_table_ptr, token_to_req_ptr, row_map_ptr,
    cache_kv_ptr, tags_ptr, claim_ptr, counters_ptr,
    stride_k_block, stride_k_token, stride_k_head,
    stride_v_block, stride_v_token, stride_v_head,
    stride_indices_row, stride_table_req,
    stride_c_req, stride_c_slot, stride_c_head,
    num_requests, num_cache_reqs, num_cache_blocks,
    TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr,
    CACHE_SLOTS: tl.constexpr, NUM_KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PUBLISH: tl.constexpr,
) -> None:
    # Phase 2 (PUBLISH=False): the claim winner, if it is a miss, copies its host row; only one
    # token can win a slot, so every write to a slot carries the same data. Phase 3 (PUBLISH=True,
    # a separate launch): the same lanes, decided from the same unchanged tags/claims, set the tag.
    # Keeping the tag writes out of the copy launch means no lane can see a tag whose row is still
    # in flight (that race lost rows on the x8 card, where host reads are slowest).
    token, valid, cache_req, slot, safe_page, page_offset = _qsa_rowcache_select(
        indices_ptr, block_table_ptr, token_to_req_ptr, row_map_ptr, tl.program_id(0), tl.program_id(1),
        stride_indices_row, stride_table_req, num_requests, num_cache_reqs, num_cache_blocks,
        TOPK, PAGE_SIZE, PAGE_TABLE_WIDTH, CACHE_SLOTS, BLOCK_N)
    offset = cache_req * CACHE_SLOTS + slot
    tag = tl.load(tags_ptr + offset, mask=valid, other=-1)
    claim = tl.load(claim_ptr + offset, mask=valid, other=-1)
    miss = valid & (tag != token)
    copy = miss & (claim == token)
    if PUBLISH:
        tl.store(tags_ptr + offset, token, mask=copy)
    else:
        tl.atomic_add(counters_ptr, tl.sum(valid.to(tl.int64), axis=0))
        tl.atomic_add(counters_ptr + 1, tl.sum(miss.to(tl.int64), axis=0))
        dims = tl.arange(0, HEAD_DIM)
        for head in tl.static_range(NUM_KV_HEADS):
            keys = tl.load(k_cache_ptr + safe_page[:, None] * stride_k_block + page_offset[:, None] * stride_k_token
                           + head * stride_k_head + dims[None, :], mask=copy[:, None], other=0.0)
            values = tl.load(v_cache_ptr + safe_page[:, None] * stride_v_block + page_offset[:, None] * stride_v_token
                             + head * stride_v_head + dims[None, :], mask=copy[:, None], other=0.0)
            dest = cache_kv_ptr + cache_req * stride_c_req + slot[:, None] * stride_c_slot + head * stride_c_head
            tl.store(dest + dims[None, :], keys, mask=copy[:, None])
            tl.store(dest + HEAD_DIM + dims[None, :], values, mask=copy[:, None])


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_merge_splitk_kernel('''
ops = sub(ops, '''
@triton.jit(do_not_specialize=["num_rows"])
def _qsa_merge_splitk_kernel(''', fill_kernel)

# 3. wrapper: signature, fill launch, extra kernel args
ops = sub(ops, """    use_prefill_config: bool,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    \"\"\"Run sparse GQA directly over paged BF16 K/V caches.""", """    use_prefill_config: bool,
    out: torch.Tensor | None = None,
    rowcache: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    \"\"\"Run sparse GQA directly over paged BF16 K/V caches.""")
ops = sub(ops, """    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        q.stride(0),""", """    if rowcache is not None:
        cache_kv, tags, counters, claim = rowcache[:4]
        row_map = rowcache[6]
        cache_slots = cache_kv.shape[1]
        fill_n = 32
        fill_grid = (q.shape[0], triton.cdiv(selection_width, fill_n))
        claim.fill_(-1)
        _qsa_rowcache_claim_kernel[fill_grid](
            logical_indices, block_table, token_to_req, row_map, tags, claim,
            logical_indices.stride(0), block_table.stride(0),
            block_table.shape[0], cache_kv.shape[0], k_cache.shape[0],
            TOPK=selection_width, PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1],
            CACHE_SLOTS=cache_slots, BLOCK_N=fill_n, num_warps=1, num_stages=1,
        )
        for publish in (False, True):
            _qsa_rowcache_copy_kernel[fill_grid](
                k_cache, v_cache, logical_indices, block_table, token_to_req, row_map, cache_kv, tags, claim, counters,
                k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
                logical_indices.stride(0), block_table.stride(0),
                cache_kv.stride(0), cache_kv.stride(1), cache_kv.stride(2),
                block_table.shape[0], cache_kv.shape[0], k_cache.shape[0],
                TOPK=selection_width, PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1],
                CACHE_SLOTS=cache_slots, NUM_KV_HEADS=k_cache.shape[2], HEAD_DIM=q.shape[2],
                BLOCK_N=fill_n, PUBLISH=publish, num_warps=1 if publish else 4, num_stages=1,
            )
        cache_args = (cache_kv, tags, row_map, cache_kv.stride(0), cache_kv.stride(1), cache_kv.stride(2),
                      cache_kv.shape[0])
    else:
        cache_slots = 1
        cache_args = (out, out, out, 0, 0, 0, 1)
    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        *cache_args,
        q.stride(0),""")
ops = sub(ops, """        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=partial_warps,
        num_stages=2,
    )
    if num_splits == 1:
        return out
""", """        BLOCK_M=block_m,
        BLOCK_N=block_n,
        CACHE_SLOTS=cache_slots,
        USE_CACHE=rowcache is not None,
        num_warps=partial_warps,
        num_stages=2,
    )
    if num_splits == 1:
        return out
""")

# 4. kernel warmup call (the other caller of the split-k kernel): pass the cache-off arguments
ops = sub(ops, """            partial_lse_ptr,
            output_ptr,
            row_stride,
            head_stride,
            key_cache.stride(0),""", """            partial_lse_ptr,
            output_ptr,
            output_ptr,
            output_ptr,
            output_ptr,
            0,
            0,
            0,
            1,
            row_stride,
            head_stride,
            key_cache.stride(0),""")
assert ops.count("_qsa_sparse_paged_gqa_splitk_kernel[") + ops.count("_qsa_sparse_paged_gqa_splitk_kernel.") == 2

open(sys.argv[3], "w").write(qsa)
open(sys.argv[4], "w").write(ops)
print("ok")
