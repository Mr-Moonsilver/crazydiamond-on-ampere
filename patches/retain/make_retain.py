#!/usr/bin/env python3
"""S3: keep one prefilling request's host K/V history on the GPU across its chunks (QSA staged prefill).

Today every 1,024-token chunk, each QSA layer copies the request's whole history (or its sparse union, 50-66 % at
long histories) from pinned host RAM into the 96 MiB arena again: 18 ms per prefill step on the x16 ranks, 33 ms on
the x8 rank (24 Oct timing). With QSA_RETAIN_HISTORY=1 each layer keeps one request's pages in a VRAM buffer of
QSA_RETAIN_PAGES pages (default 40, ~1 GiB per rank over 6 layers) and copies only what changed since its previous
chunk, then attends over the buffer as the first bucket; pages beyond the buffer go through the arena as today (dense).
v2 (24 Oct): v1 retained up to 65 pages (1.6 GiB per rank) and ran the rank out of VRAM in the next prefill
activation (overlay-v10, OOM at 20 MiB); now a fixed budget plus the arena tail.

Reuse rules (anything else is copied again):
  * same request: the runner's per-request id (QSA_REQ_UID, also used by the row cache; a re-added or resumed request
    gets a new one), read together with the block ids in one small host transfer;
  * same physical block id for that page, and the page was complete (all page_size tokens written) when copied --
    the current chunk's K/V is written to host before attention, so pages touched by it are always recopied.
  * the first chunk a layer sees for a request uses the old path (sparse union) and only arms retention: one-chunk
    appends keep today's cost, retention pays from the second chunk on.
One request per layer at a time. A failed allocation turns retention off for the process (old path). Unset, the patched files behave exactly like their inputs.

usage: make_retain.py ops_qsa.py.in ops_qsa.py.out qsa.py.in qsa.py.out
"""
import difflib, sys

ops_in, ops_out, qsa_in, qsa_out = sys.argv[1:5]


def sub(text, old, new, count=1):
    assert text.count(old) == count, (old[:80], text.count(old))
    return text.replace(old, new)


src = orig = open(ops_in).read()

src = sub(src, """_STAGE_TIMER = _QSAStageTimer() if os.environ.get("QSA_STAGE_TIMING", "0") == "1" else None
""", '''_STAGE_TIMER = _QSAStageTimer() if os.environ.get("QSA_STAGE_TIMING", "0") == "1" else None


class _QSARetainedHistory:
    """QSA_RETAIN_HISTORY=1: one prefilling request's host K/V pages kept in VRAM per layer (see make_retain.py)."""

    def __init__(self, cap_pages):
        self.cap = cap_pages
        self.buf = {}    # host kv_cache data_ptr (one per layer) -> VRAM [cap, heads, page, 2 * head_dim]
        self.state = {}  # data_ptr -> (request id, block ids of the copied pages, pages complete when copied)
        self.failed = False

    def get(self, kv_cache, block_table_row, num_pages, seq_len, uid, ids):
        """Pages [0, min(num_pages, cap)) of request ``uid`` in VRAM, refreshed; (view, pages copied) or (None, 0)."""
        key = kv_cache.data_ptr()
        old_uid, old_ids, complete = self.state.get(key, (None, (), 0))
        if self.failed or uid < 1:
            self.state.pop(key, None)
            return None, 0
        if old_uid != uid:
            self.state[key] = (uid, (), 0)  # first chunk seen: arm only
            return None, 0
        buf = self.buf.get(key)
        if buf is None:
            try:
                buf = torch.empty((self.cap, *kv_cache.shape[1:]), dtype=kv_cache.dtype,
                                  device=block_table_row.device)
            except torch.OutOfMemoryError:
                self.failed = True
                self.state.clear()
                return None, 0
            self.buf[key] = buf
        keep = min(num_pages, self.cap)
        first, lim = 0, min(complete, keep, len(old_ids))
        while first < lim and old_ids[first] == ids[first]:
            first += 1
        if first < keep:
            physical = block_table_row[first:keep].to(torch.int64)
            torch.index_select(kv_cache, 0, physical, out=buf[first:keep])
        self.state[key] = (uid, tuple(ids[:keep]), min(keep, seq_len // kv_cache.shape[2]))
        return buf[:keep], keep - first


_RETAINED = (_QSARetainedHistory(int(os.environ.get("QSA_RETAIN_PAGES", "40")))
             if os.environ.get("QSA_RETAIN_HISTORY", "0") == "1" else None)


def qsa_retained_history(kv_cache, block_table_row, num_pages, seq_len, uid, ids):
    if _RETAINED is None:
        return None, 0
    return _RETAINED.get(kv_cache, block_table_row, num_pages, seq_len, uid, ids)
''')

src = sub(src, """    use_prefill_config: bool,
    out: torch.Tensor,
) -> None:
    \"\"\"Run the rows in ``rows`` against host-resident K/V via the arena.
""", """    use_prefill_config: bool,
    out: torch.Tensor,
    retained: torch.Tensor | None = None,
    retained_copied: int = 0,
) -> None:
    \"\"\"Run the rows in ``rows`` against host-resident K/V via the arena.

    With ``retained`` (the first pages of this request already in VRAM, see qsa_retained_history) that is the first
    bucket; the remaining pages are staged through the arena (dense) as further buckets.
""")


src = sub(src, """    if (num_pages > int(os.environ.get("QSA_SPARSE_MIN_PAGES", pages_in_arena))
""", """    if (retained is None and num_pages > int(os.environ.get("QSA_SPARSE_MIN_PAGES", pages_in_arena))
""")

src = sub(src, """    num_buckets = triton.cdiv(num_pages, pages_in_arena)
    if _t is not None:
        _t["sparse"] = unique_tokens is not None
        _t["moved"] = num_pages
""", """    first_hi = retained.shape[0] if retained is not None else 0
    spans = ([(0, first_hi)] if retained is not None else []) + [
        (lo, min(lo + pages_in_arena, num_pages)) for lo in range(first_hi, num_pages, pages_in_arena)]
    num_buckets = len(spans)
    if _t is not None:
        _t["sparse"] = unique_tokens is not None
        _t["moved"] = num_pages if retained is None else retained_copied + num_pages - first_hi
""")

src = sub(src, """    for bucket in range(num_buckets):
        page_lo = bucket * pages_in_arena
        page_hi = min(page_lo + pages_in_arena, num_pages)
        staged = arena[: page_hi - page_lo]
""", """    for bucket, (page_lo, page_hi) in enumerate(spans):
        from_retained = retained is not None and bucket == 0
        staged = retained if from_retained else arena[: page_hi - page_lo]
""")

src = sub(src, """        if unique_tokens is None:
            physical = block_table_row[page_lo:page_hi].to(torch.int64)
""", """        if from_retained:
            pass  # already in VRAM
        elif unique_tokens is None:
            physical = block_table_row[page_lo:page_hi].to(torch.int64)
""")

open(ops_out, 'w').write(src)
sys.stdout.writelines(difflib.unified_diff(orig.splitlines(True), src.splitlines(True), 'ops_qsa.py.in', 'ops_qsa.py.out'))

src = orig = open(qsa_in).read()
src = sub(src, """    from .ops.qsa import qsa_get_staging_arena, qsa_sparse_paged_attention_staged
""", """    from .ops.qsa import qsa_get_staging_arena, qsa_retained_history, qsa_sparse_paged_attention_staged
""")
src = sub(src, """    rows = torch.arange(start, end, device=query.device, dtype=torch.int64)
    qsa_sparse_paged_attention_staged(
        query[:num_tokens],
        kv_cache,
        logical_indices,
        attn_metadata.block_table[best],
        rows,
        max_logical_page,
        arena,
        use_prefill_config,
        output[:num_tokens],
    )
""", """    rows = torch.arange(start, end, device=query.device, dtype=torch.int64)
    retained, copied = None, 0
    if os.environ.get("QSA_RETAIN_HISTORY", "0") == "1" and not os.path.exists("/work/retain.off"):
        # S3 (make_retain.py): request identity = the runner's per-request id, as for the row cache.
        # /work/retain.off: test-only A/B switch inside one engine process (the launcher removes it).
        import vllm.forward_context as _fc
        uids = getattr(_fc, "QSA_REQ_UID", None)
        if uids is not None and getattr(_fc, "QSA_REQ_UID_WRITER", False) and best < uids.shape[0]:
            table = attn_metadata.block_table[best]
            meta = torch.cat([uids[best:best + 1].to(table.dtype), table[: max_logical_page + 1]]).tolist()
            retained, copied = qsa_retained_history(
                kv_cache, table, max_logical_page + 1, int(lengths[best]), int(meta[0]), meta[1:])
    qsa_sparse_paged_attention_staged(
        query[:num_tokens],
        kv_cache,
        logical_indices,
        attn_metadata.block_table[best],
        rows,
        max_logical_page,
        arena,
        use_prefill_config,
        output[:num_tokens],
        retained=retained,
        retained_copied=copied,
    )
""")
open(qsa_out, 'w').write(src)
sys.stdout.writelines(difflib.unified_diff(orig.splitlines(True), src.splitlines(True), 'qsa.py.in', 'qsa.py.out'))
