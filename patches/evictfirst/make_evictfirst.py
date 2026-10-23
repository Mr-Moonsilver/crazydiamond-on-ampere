#!/usr/bin/env python3
"""Evict superseded Mamba partial tails first (vLLM 0.29.1rc1 image, align mode + --prefix-match-unit).

Each Mamba group (36 here, one per linear-attention layer) keeps a session's resume point as one 1.6 MB state
block, so the 1340-block pool holds only ~37 snapshots. A follow-up turn hits the previous turn's partial tail,
copies it (CoW) and publishes its own later tail. The consumed source is then useless for a linear conversation,
yet its release appended it to the LRU end, so every turn pushed two snapshots to the young end and idle sessions
were evicted after ~10 other turns. This patch demotes the consumed source to the front of the free queue once the
request has published its own later tail. The block keeps its hash (it still hits until reused); only eviction
order changes, so correctness is unaffected.

usage: make_evictfirst.py ORIG_DIR OUT_DIR   (ORIG_DIR holds the image's block_pool.py and
       single_type_kv_cache_manager.py; writes the patched files and .diff next to them in OUT_DIR)"""
import difflib, pathlib, sys

src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)


def sub(text, old, new, count=1):
    assert text.count(old) == count, (old[:80], text.count(old))
    return text.replace(old, new)


bp = (src / "block_pool.py").read_text()
orig_bp = bp
bp = sub(bp, """        self.cached_block_hashes_by_block: dict[int, set[BlockHashWithGroupId]] = {}
""", """        self.cached_block_hashes_by_block: dict[int, set[BlockHashWithGroupId]] = {}
        # Superseded partial-tail sources, freed to the front of the queue
        # (flash-next evict-first patch).
        self._evict_first: set[int] = set()
        self.num_demoted = 0
""")
bp = sub(bp, """    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:""",
         """    def demote(self, block: KVCacheBlock) -> None:
        \"\"\"Make a cached block the next eviction candidate without dropping
        its hash: a superseded partial-tail source still hits until reused.\"\"\"
        if block.is_null:
            return
        self.num_demoted += 1
        if self.num_demoted % 2000 == 1:
            logger.info("evict-first: %d superseded tail blocks demoted",
                        self.num_demoted)
        if block.ref_cnt == 0:
            self.free_block_queue.remove(block)
            self.free_block_queue.prepend_n([block])
        else:
            self._evict_first.add(block.block_id)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:""")
bp = sub(bp, """            for block in ret:
                self._maybe_evict_cached_block(block)
""", """            for block in ret:
                self._evict_first.discard(block.block_id)
                self._maybe_evict_cached_block(block)
""")
bp = sub(bp, """            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
""", """            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            self._evict_first.discard(block.block_id)
            block.ref_cnt += 1
""")
bp = sub(bp, """                if block.block_hash is None or not self.enable_caching:
                    # LIFO reuse of non-cached blocks for better GPU locality.
                    blocks_to_evict_first.append(block)
""", """                if (
                    block.block_hash is None
                    or not self.enable_caching
                    or block.block_id in self._evict_first
                ):
                    # LIFO reuse of non-cached blocks for better GPU locality;
                    # superseded partial tails go here too.
                    self._evict_first.discard(block.block_id)
                    blocks_to_evict_first.append(block)
""")
bp = sub(bp, """        self.cached_block_hashes_by_block.clear()
        if self._reuse_watchers:""", """        self.cached_block_hashes_by_block.clear()
        self._evict_first.clear()
        if self._reuse_watchers:""")

st = (src / "single_type_kv_cache_manager.py").read_text()
orig_st = st
st = sub(st, """            self._producer_partial_tail_reqs: dict[str, tuple[KVCacheBlock, int]] = {}
""", """            self._producer_partial_tail_reqs: dict[str, tuple[KVCacheBlock, int]] = {}
            # Partial tail a request resumed from: (block, hash, tokens).
            self._consumed_tails: dict = {}
""")
st = sub(st, """                    else:
                        self._apply_cow(request_id, block_idx, source_block, cow_block)
                        returned_blocks = [cow_block] + returned_blocks
""", """                    else:
                        self._apply_cow(request_id, block_idx, source_block, cow_block)
                        self._consumed_tails[request_id] = (
                            source_block,
                            source_block.block_hash,
                            source_block.block_hash_num_tokens,
                        )
                        returned_blocks = [cow_block] + returned_blocks
""")
st = sub(st, """            self._producer_partial_tail_reqs.pop(request_id, None)
            # An offer is only guaranteed""", """            self._producer_partial_tail_reqs.pop(request_id, None)
            self._consumed_tails.pop(request_id, None)
            # An offer is only guaranteed""")
st = sub(st, """            partial_hash = self._cache_partial_tail_block(request, num_tokens)
            if partial_hash is not None:
                self.cached_blocks_this_step.add(partial_hash)
""", """            partial_hash = self._cache_partial_tail_block(request, num_tokens)
            if partial_hash is not None:
                self.cached_blocks_this_step.add(partial_hash)
                self._demote_consumed_tail(request)
""")
st = sub(st, """    def new_step_starts(self) -> None:
        self.cached_blocks_this_step.clear()
""", """    def new_step_starts(self) -> None:
        self.cached_blocks_this_step.clear()

    def _demote_consumed_tail(self, request: Request) -> None:
        # flash-next evict-first patch: once this request has published its own
        # later prompt-boundary state, the partial tail it resumed from is
        # superseded for a linear conversation; let it be evicted first.
        consumed = self._consumed_tails.pop(request.request_id, None)
        if consumed is None:
            return
        block, block_hash, num_tokens = consumed
        unit = self.block_pool.hash_block_size
        if (
            block_hash is None
            or block.block_hash != block_hash
            or num_tokens is None
            or num_tokens >= request.num_prompt_tokens // unit * unit
        ):
            return
        self.block_pool.demote(block)
""")

for name, old, new in (("block_pool.py", orig_bp, bp), ("single_type_kv_cache_manager.py", orig_st, st)):
    (out / name).write_text(new)
    (out / (name + ".diff")).write_text("".join(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True), "a/" + name, "b/" + name)))
    print("wrote", out / name)
