"""Frozen native17 generated attention functions for historical negatives.

Source: operation_source.py at 57c0d38, applied to the pinned QSA reference.
Only compiled against declared CPU tensor/kernel leaves; not box evidence.
Current native18 connected scenarios must use the current generated functions.
"""

def flash_s5_arena_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table_row: torch.Tensor,
    rows: torch.Tensor,
    max_logical_page: int,
    arena: torch.Tensor,
    use_prefill_config: bool,
    out: torch.Tensor,
    retained: torch.Tensor | None = None,
    retained_copied: int = 0,
    *, e3op,
) -> None:
    """Run the rows in ``rows`` against host-resident K/V via the arena.

    With ``retained`` (the first pages of this request already in VRAM, see qsa_retained_history) that is the first
    bucket; the remaining pages are staged through the arena (dense) as further buckets.

    Caller contract (use the direct path when it cannot be met):
      * ``rows`` belongs to a single request and ``block_table_row`` is that
        request's block table row.
      * ``kv_cache`` is the host (UVA) tensor shaped
        ``[blocks, num_kv_heads, page_size, 2 * head_dim]`` and ``arena`` is a
        GPU tensor of the same dtype whose first dimension is a page count.
    """
    if retained is not None or retained_copied != 0:
        raise RuntimeError("S5 operation rejects retained-bucket shortcut")
    num_rows = rows.numel()
    if num_rows == 0:
        raise RuntimeError("S5 operation received no attention rows")
    pages_in_arena = arena.shape[0]
    page_size = kv_cache.shape[2]
    head_dim = kv_cache.shape[3] // 2
    num_kv_heads = kv_cache.shape[1]
    num_pages = max_logical_page + 1
    unique_tokens = None
    _t = None  # lane-local timing comes from the operation server
    # Long prefills often reuse the same selected keys across adjacent queries.
    # Compact that union before transfer instead of reading every host page.
    # Keep the existing path for small contexts or nearly dense selections.
    if False:  # declared dense transport; never mutate process environment
        selected_rows = logical_indices.index_select(0, rows)
        unique, inverse = torch.unique(
            selected_rows[:, :-1].reshape(-1), sorted=True, return_inverse=True
        )
        has_padding = unique.numel() > 0 and int(unique[0]) == -1
        unique_valid = unique[1:] if has_padding else unique
        if unique_valid.numel() == 0:
            raise RuntimeError("S5 operation received no selected rows")
            return
        if _t is not None:
            _t["coverage"] = unique_valid.numel() / max(1, num_pages * page_size)
        if unique_valid.numel() < 0.8 * num_pages * page_size:
            remapped = logical_indices.clone()
            compact = inverse.reshape(num_rows, -1).to(torch.int32)
            if has_padding:
                compact -= 1
            remapped[rows, :-1] = compact
            logical_indices = remapped
            unique_tokens = unique_valid
            num_pages = triton.cdiv(unique_tokens.numel(), page_size)
    if unique_tokens is not None:
        raise RuntimeError("S5 operation does not support sparse-UVA compaction")
    first_hi = retained.shape[0] if retained is not None else 0
    spans = ([(0, first_hi)] if retained is not None else []) + [
        (lo, min(lo + pages_in_arena, num_pages)) for lo in range(first_hi, num_pages, pages_in_arena)]
    num_buckets = len(spans)
    if _t is not None:
        _t["sparse"] = unique_tokens is not None
        _t["moved"] = num_pages if retained is None else retained_copied + num_pages - first_hi

    group_size = q.shape[1] // num_kv_heads
    # Ampere BF16 tensor-core dot requires at least 16 rows. TP4 has
    # six query heads per local KV head; masked padding preserves the result.
    block_m = max(16, triton.next_power_of_2(group_size))
    selection_width = logical_indices.shape[1] - 1
    block_n, partial_warps, num_tiles, _ = _select_config(
        num_rows, num_kv_heads, use_prefill_config, selection_width
    )

    if num_buckets == 1:
        acc = max_state = norm_state = q
    else:
        acc = torch.empty(
            (num_rows, q.shape[1], q.shape[2]), dtype=torch.float32, device=q.device
        )
        max_state = torch.empty(
            (num_rows, q.shape[1]), dtype=torch.float32, device=q.device
        )
        norm_state = torch.empty_like(max_state)

    for bucket, (page_lo, page_hi) in enumerate(spans):
        from_retained = retained is not None and bucket == 0
        staged = retained if from_retained else arena[: page_hi - page_lo]
        if _t is not None:
            _t["ev"].append(_STAGE_TIMER.mark())
        if from_retained:
            pass  # already in VRAM
        elif unique_tokens is None:
            e3op.fill_bucket(staged, page_lo, page_hi, block_table_row)
        else:
            selected = unique_tokens[page_lo * page_size:page_hi * page_size]
            _qsa_gather_selected_host_rows[(selected.numel(), num_kv_heads)](
                kv_cache, selected, block_table_row, staged,
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                staged.stride(0), staged.stride(1), staged.stride(2),
                selected.numel(), PAGE_SIZE=page_size, ROW_WIDTH=2 * head_dim,
                BLOCK_D=triton.next_power_of_2(2 * head_dim), num_warps=4,
            )
        if _t is not None:
            _t["ev"].append(_STAGE_TIMER.mark())
        k_arena, v_arena = staged.transpose(1, 2).split(head_dim, dim=-1)
        e3op.retain_attention(q, logical_indices, rows, acc, max_state, norm_state, out)
        _qsa_sparse_staged_gqa_kernel[(num_rows, num_kv_heads)](
            q,
            k_arena,
            v_arena,
            logical_indices,
            rows,
            acc,
            max_state,
            norm_state,
            out,
            q.stride(0),
            q.stride(1),
            k_arena.stride(0),
            k_arena.stride(1),
            k_arena.stride(2),
            v_arena.stride(0),
            v_arena.stride(1),
            v_arena.stride(2),
            logical_indices.stride(0),
            out.stride(0),
            out.stride(1),
            num_rows,
            page_lo,
            page_hi - page_lo,
            TOPK=selection_width,
            PAGE_SIZE=page_size,
            GROUP_SIZE=group_size,
            HEAD_DIM=q.shape[2],
            NUM_QUERY_HEADS=q.shape[1],
            NUM_TILES=num_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            FIRST_BUCKET=bucket == 0,
            LAST_BUCKET=bucket == num_buckets - 1,
            num_warps=partial_warps,
            num_stages=2,
        )
        yield e3op.after_attention_bucket(
            staged, block_table_row, q, logical_indices, rows, acc, max_state, norm_state, out)
        if _t is not None:
            _t["ev"].append(_STAGE_TIMER.mark())
    if _t is not None:
        _STAGE_TIMER.end(_t)

def flash_s5_sparse_arena_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table_row: torch.Tensor,
    rows: torch.Tensor,
    max_logical_page: int,
    arena: torch.Tensor,
    use_prefill_config: bool,
    out: torch.Tensor,
    retained: torch.Tensor | None = None,
    retained_copied: int = 0,
    *, e3op,
) -> None:
    """Run the rows in ``rows`` against host-resident K/V via the arena.

    With ``retained`` (the first pages of this request already in VRAM, see qsa_retained_history) that is the first
    bucket; the remaining pages are staged through the arena (dense) as further buckets.

    Caller contract (use the direct path when it cannot be met):
      * ``rows`` belongs to a single request and ``block_table_row`` is that
        request's block table row.
      * ``kv_cache`` is the host (UVA) tensor shaped
        ``[blocks, num_kv_heads, page_size, 2 * head_dim]`` and ``arena`` is a
        GPU tensor of the same dtype whose first dimension is a page count.
    """
    if retained is not None or retained_copied != 0:
        raise RuntimeError("S5 sparse operation rejects retained-bucket shortcut")
    num_rows = rows.numel()
    if num_rows == 0:
        raise RuntimeError("S5 sparse operation received no attention rows")
    pages_in_arena = arena.shape[0]
    page_size = kv_cache.shape[2]
    head_dim = kv_cache.shape[3] // 2
    num_kv_heads = kv_cache.shape[1]
    num_pages = max_logical_page + 1
    unique_tokens = None
    _t = None  # lane-local timing comes from the operation server
    # Long prefills often reuse the same selected keys across adjacent queries.
    # Compact that union before transfer instead of reading every host page.
    # Keep the existing path for small contexts or nearly dense selections.
    if retained is None and e3op.sparse_selection_eligible(num_pages, pages_in_arena):
        # Fixed-capacity sorted union and exact inverse, entirely on the device.
        # Capacity, row count and width are host-known; no CUDA scalar, list or
        # shape-dependent allocation is read or created here.
        capacity = num_pages * page_size
        union = e3op.sparse_union_scratch(capacity, logical_indices, rows)
        unique_tokens, union_info, remapped = _s6_union(
            logical_indices, rows, capacity, union)
        # Bounded D2H into pinned scratch on the copy stream, behind a compute
        # event.  The generator yields until that copy event is host-queried;
        # the count and selection prefix are read only from completed pinned
        # memory.  Host slicing by an already-read count is not a device read.
        _selection = e3op.stage_sparse_selection(unique_tokens, union_info)
        while not e3op.sparse_selection_ready(_selection):
            yield e3op.sparse_selection_pending(_selection)
        selected_count = e3op.sparse_selection_count(_selection)
        if selected_count == 0:
            e3op.sparse_empty()
            out[rows] = 0
            return
        # Integer form of the pinned ``count < 0.8 * capacity`` threshold; it
        # is exact for every integer count and capacity.
        if selected_count * 5 < capacity * 4:
            logical_indices = remapped
            num_pages = triton.cdiv(selected_count, page_size)
        else:
            unique_tokens = None
    e3op.declare_sparse_route("sparse_dma" if unique_tokens is not None else "dense_dma")
    first_hi = retained.shape[0] if retained is not None else 0
    spans = ([(0, first_hi)] if retained is not None else []) + [
        (lo, min(lo + pages_in_arena, num_pages)) for lo in range(first_hi, num_pages, pages_in_arena)]
    num_buckets = len(spans)
    if _t is not None:
        _t["sparse"] = unique_tokens is not None
        _t["moved"] = num_pages if retained is None else retained_copied + num_pages - first_hi

    group_size = q.shape[1] // num_kv_heads
    # Ampere BF16 tensor-core dot requires at least 16 rows. TP4 has
    # six query heads per local KV head; masked padding preserves the result.
    block_m = max(16, triton.next_power_of_2(group_size))
    selection_width = logical_indices.shape[1] - 1
    block_n, partial_warps, num_tiles, _ = _select_config(
        num_rows, num_kv_heads, use_prefill_config, selection_width
    )

    if num_buckets == 1:
        acc = max_state = norm_state = q
    else:
        acc = torch.empty(
            (num_rows, q.shape[1], q.shape[2]), dtype=torch.float32, device=q.device
        )
        max_state = torch.empty(
            (num_rows, q.shape[1]), dtype=torch.float32, device=q.device
        )
        norm_state = torch.empty_like(max_state)

    for bucket, (page_lo, page_hi) in enumerate(spans):
        from_retained = retained is not None and bucket == 0
        staged = retained if from_retained else arena[: page_hi - page_lo]
        if _t is not None:
            _t["ev"].append(_STAGE_TIMER.mark())
        if from_retained:
            pass  # already in VRAM
        elif unique_tokens is None:
            e3op.fill_bucket(staged, page_lo, page_hi, block_table_row)
        else:
            selected = unique_tokens[
                page_lo * page_size:min(page_hi * page_size, selected_count)]
            while True:
                _copy = e3op.fill_sparse_bucket(
                    staged, selected, page_lo * page_size, block_table_row)
                if not getattr(_copy, "sparse_copy_pending", False):
                    break
                yield _copy
        if _t is not None:
            _t["ev"].append(_STAGE_TIMER.mark())
        k_arena, v_arena = staged.transpose(1, 2).split(head_dim, dim=-1)
        _qsa_sparse_staged_gqa_kernel[(num_rows, num_kv_heads)](
            q,
            k_arena,
            v_arena,
            logical_indices,
            rows,
            acc,
            max_state,
            norm_state,
            out,
            q.stride(0),
            q.stride(1),
            k_arena.stride(0),
            k_arena.stride(1),
            k_arena.stride(2),
            v_arena.stride(0),
            v_arena.stride(1),
            v_arena.stride(2),
            logical_indices.stride(0),
            out.stride(0),
            out.stride(1),
            num_rows,
            page_lo,
            page_hi - page_lo,
            TOPK=selection_width,
            PAGE_SIZE=page_size,
            GROUP_SIZE=group_size,
            HEAD_DIM=q.shape[2],
            NUM_QUERY_HEADS=q.shape[1],
            NUM_TILES=num_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            FIRST_BUCKET=bucket == 0,
            LAST_BUCKET=bucket == num_buckets - 1,
            num_warps=partial_warps,
            num_stages=2,
        )
        yield e3op.after_attention_bucket(
            staged, block_table_row, q, logical_indices, rows, acc, max_state, norm_state, out)
        if _t is not None:
            _t["ev"].append(_STAGE_TIMER.mark())
    if _t is not None:
        _STAGE_TIMER.end(_t)
