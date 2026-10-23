#!/usr/bin/env python3
"""Numerical check of the row-cache kernels: cached attention must equal uncached attention,
cold (all misses), warm (all hits) and with forced slot collisions."""
import importlib.util, sys, torch

spec = importlib.util.spec_from_file_location("ops_qsa_rc", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
dev = "cuda"
blocks, page, heads, dim, reqs, qheads = 128, 256, 1, 256, 8, 12
host = torch.randn(blocks, heads, page, 2 * dim, dtype=torch.bfloat16, device=dev)
if len(sys.argv) > 2 and sys.argv[2] == "uva":  # host KV as production has it: pinned RAM seen through UVA
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    pinned = host.cpu().pin_memory()
    host = get_accelerator_view_from_cpu_tensor(pinned)
    print("host KV: pinned RAM via UVA")
key_cache, value_cache = host.transpose(1, 2).split(dim, dim=-1)
rows_per_req, topk = 2, 2051
rows = reqs * rows_per_req
q = torch.randn(rows, qheads, dim, dtype=torch.bfloat16, device=dev)
pages_per_req = blocks // reqs
block_table = torch.arange(blocks, dtype=torch.int32, device=dev).view(reqs, pages_per_req)[:, torch.randperm(pages_per_req)].contiguous()
ctx = pages_per_req * page
token_to_req = torch.arange(reqs, dtype=torch.int32, device=dev).repeat_interleave(rows_per_req)
idx = torch.full((rows, topk + 1), -1, dtype=torch.int32, device=dev)
for r in range(rows):
    n = topk - (r * 37) % 100
    n = min(n, 4096)
    idx[r, :n] = torch.randperm(ctx, device=dev)[:n].int()
    idx[r, -1] = n


def run(rowcache, use_prefill=False):
    return m.qsa_sparse_paged_attention(q, key_cache, value_cache, idx, block_table, token_to_req,
                                        use_prefill, rowcache=rowcache)


ref = run(None).float()
for _ in range(5):  # the unmodified (no-cache) path must reproduce itself exactly
    again = run(None).float()
    print(f"nocache_repeat_maxdiff={(again - ref).abs().max():.3e}")
for slots in (4096, 256):  # 256 forces many collisions
    rows = reqs + 3  # more cache rows than requests, mapped in a permuted order
    perm = torch.randperm(rows - 1, device=dev)[:reqs].int()
    row_map = torch.full((1024,), -1, dtype=torch.int32, device=dev); row_map[:reqs] = perm
    cache = (torch.full((rows, slots, heads, 2 * dim), -7.0, dtype=torch.bfloat16, device=dev),  # sentinel
             torch.full((rows, slots), -1, dtype=torch.int32, device=dev),
             torch.zeros(2, dtype=torch.int64, device=dev),
             torch.empty((rows, slots), dtype=torch.int32, device=dev),
             None, None, row_map, None)
    cold = run(cache).float(); warm = run(cache).float()
    for _ in range(3):  # repeated warm calls keep re-claiming colliding slots
        warm = run(cache).float()
    filled = int((cache[1] >= 0).sum())
    R = lambda rq: int(perm[rq])
    bad = sum(int(not torch.equal(cache[0][R(rq), s, 0], host[int(block_table[rq, int(cache[1][R(rq), s]) // page]), 0, int(cache[1][R(rq), s]) % page]))
              for rq in range(reqs) for s in torch.nonzero(cache[1][R(rq)] >= 0).flatten().tolist())
    unmapped_rows_touched = int((cache[1][[r for r in range(rows) if r not in set(perm.tolist())]] >= 0).sum())
    if bad:
        for rq in range(reqs):
            rows_rq = host[block_table[rq].long(), 0].reshape(-1, 2 * dim)  # [ctx, 2*dim] in logical order
            for s_ in torch.nonzero(cache[1][R(rq)] >= 0).flatten().tolist():
                tok = int(cache[1][R(rq), s_]); got = cache[0][R(rq), s_, 0]
                if torch.equal(got, rows_rq[tok]):
                    continue
                exact = torch.nonzero((rows_rq == got).all(dim=1)).flatten().tolist()
                wrong = int((got != rows_rq[tok]).sum())
                wrong_k = int((got[:dim] != rows_rq[tok][:dim]).sum())
                cands = [t for t in exact]
                print(f"  CORRUPT req={rq} slot={s_} tag={tok} wrong_elems={wrong} (K {wrong_k}, V {wrong - wrong_k}) "
                      f"data_equals_token={cands} same_slot={[t % slots == s_ for t in cands]} zeros={int((got == 0).sum())} sentinel={int((got == -7).sum())}")
    print(f"slots={slots} counters(selected,misses)={cache[2].tolist()} filled={filled} corrupt_slots={bad} cold_maxdiff={(cold-ref).abs().max():.3e} warm_maxdiff={(warm-ref).abs().max():.3e}")
    print(f"  unmapped_rows_touched={unmapped_rows_touched}")
    assert bad == 0 and unmapped_rows_touched == 0 and (cold - ref).abs().max() == 0 and (warm - ref).abs().max() == 0
    # every tag must point at a row whose cached data equals the host row
    t = cache[1]
    for rq in range(reqs):
        for s in torch.nonzero(t[R(rq)] >= 0).flatten().tolist():
            tok = int(t[R(rq), s]); pg = int(block_table[rq, tok // page]); off = tok % page
            assert torch.equal(cache[0][R(rq), s, 0], host[pg, 0, off]), (rq, s, tok)
print("rowcache kernels OK")
