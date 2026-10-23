#!/usr/bin/env python3
"""CPU-only check of the evict-first BlockPool changes (run inside the image: CUDA_VISIBLE_DEVICES= python3
test_evictfirst.py DIR, where DIR holds the patched block_pool.py)."""
import importlib.util, sys

sys.modules.pop("vllm.v1.core.block_pool", None)
spec = importlib.util.spec_from_file_location("vllm.v1.core.block_pool", sys.argv[1] + "/block_pool.py")
bp = importlib.util.module_from_spec(spec); sys.modules[spec.name] = bp; spec.loader.exec_module(bp)
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

pool = bp.BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
order = lambda: [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]
raw = lambda i: BlockHash(bytes([i]) * 32)
h = lambda i: make_block_hash_with_group_id(raw(i), 0)

a = pool.get_new_blocks(3)
for i, b in enumerate(a):
    pool._insert_block_hash(h(i), b, num_tokens=16)
pool.free_blocks(a)
print("after freeing 3 cached:", order())
assert order()[-3:] == [b.block_id for b in a]
pool.demote(a[1])                                   # ref 0: moves to the front, keeps its hash
assert order()[0] == a[1].block_id and a[1].block_hash == h(1), order()
assert pool.get_cached_block(raw(1), [0]) is not None
got = pool.get_new_blocks(1)[0]                     # reused first, hash dropped
assert got is a[1] and got.block_hash is None and pool.get_cached_block(raw(1), [0]) is None
print("demote ref0 -> front, reused first, hash dropped: ok")

c = pool.get_new_blocks(1)[0]
pool._insert_block_hash(h(9), c, num_tokens=16)
pool.demote(c)                                      # ref 1: remembered until freed
pool.free_blocks([c])
assert order()[0] == c.block_id and c.block_hash == h(9), order()
print("demote ref1 -> freed to front with hash: ok")

pool.touch([c]); pool.free_blocks([c])              # a new hit clears the mark: back to LRU order
assert order()[-1] == c.block_id, order()
print("touch clears the mark: ok")

d = pool.get_new_blocks(1)[0]
pool._insert_block_hash(h(7), d, num_tokens=16)
pool.demote(d); pool._maybe_evict_cached_block(d); pool.free_blocks([d])
assert order()[0] == d.block_id                     # unhashed goes to the front anyway
assert d.block_id not in pool._evict_first
print("mark cleared on free; all ok; demoted", pool.num_demoted)
