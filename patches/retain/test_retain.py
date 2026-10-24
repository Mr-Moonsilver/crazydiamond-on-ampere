#!/usr/bin/env python3
"""CPU test of _QSARetainedHistory (S3) from a patched ops_qsa.py: the class is extracted and run on CPU tensors.

The engine writes each chunk's K/V to host before attention and never rewrites a page of a live request except by
appending; a copied-on-write tail gets a new block id. The test drives exactly that and checks that every returned
buffer equals a fresh copy of the host pages, that only changed pages are copied, and that a new request id (block
ids reused with other content) never sees the previous request's pages.
usage: test_retain.py ops_qsa.py
"""
import ast, sys
from types import SimpleNamespace
from unittest.mock import patch
import torch

tree = ast.parse(open(sys.argv[1]).read())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_QSARetainedHistory")
ns = {"torch": torch}
exec(compile(ast.Module([cls], []), "retain", "exec"), ns)
R = ns["_QSARetainedHistory"]

PAGE, HEADS, W, BLOCKS = 8, 2, 4, 40
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


def run_request(r, kv, uid, table, total, chunk, start=0, expect_first_none=True):
    seq = start
    first = True
    copies = []
    while seq < total:
        new = min(chunk, total - seq)
        kv.view(-1, HEADS, W)  # noqa: shape sanity
        for t in range(seq, seq + new):  # append this chunk's tokens (written before attention)
            kv[table[t // PAGE], :, t % PAGE] = torch.randn(HEADS, W) + uid
        seq += new
        pages = (seq - 1) // PAGE + 1
        view, copied = r.get(kv, table, pages, seq, uid, table[:pages].tolist())
        if first and expect_first_none:
            check(view is None, f"uid {uid}: first chunk must only arm")
        elif view is None:
            fails.append(f"uid {uid}: no buffer at seq {seq}")
        else:
            keep = min(pages, r.cap)
            check(torch.equal(view, kv.index_select(0, table[:keep].long())), f"uid {uid}: stale buffer at seq {seq}")
            copies.append(copied)
        first = False
    return copies


torch.manual_seed(0)
kv = torch.zeros(BLOCKS, HEADS, PAGE, W)
r = R(cap_pages=12)

# 1. long fresh prefill in 5-token chunks: after the arming chunk, each chunk copies only the incomplete tail pages
table = torch.tensor([3, 7, 1, 9, 12, 15, 2, 20, 21, 22, 30, 31], dtype=torch.int32)
c = run_request(r, kv, 5, table, 90, 5)
check(len(c) > 10 and max(c[1:]) <= 2, f"case 1: too many pages copied per chunk: {c}")

# 2. same request, a tail block replaced (copy-on-write): that page and later ones are recopied
seq = 90
pages = (seq - 1) // PAGE + 1
table2 = table.clone(); table2[5] = 33
kv[33] = kv[15]
kv[33, :, 0] += 1.0  # the new block differs
view, copied = r.get(kv, table2, pages, seq, 5, table2[:pages].tolist())
check(view is not None and torch.equal(view, kv.index_select(0, table2[:pages].long())), "case 2: COW page not recopied")
check(copied == pages - 5, f"case 2: expected {pages - 5} pages copied, got {copied}")

# 3. a new request reuses the same block ids with other content: never served the old pages
for b in table.tolist():
    kv[b] = 0
c = run_request(r, kv, 6, table, 60, 7)
check(all(x >= 0 for x in c), "case 3")

# 4. more pages than the buffer: the first cap pages are retained (the caller stages the rest through the arena)
big = torch.arange(14, dtype=torch.int32) + 20
for b in big.tolist():
    kv[b] = torch.randn(HEADS, PAGE, W)
v, _ = r.get(kv, big, 14, 14 * PAGE, 8, big.tolist())
check(v is None, "case 4: first chunk must only arm")
v, n = r.get(kv, big, 14, 14 * PAGE, 8, big.tolist())
check(v is not None and v.shape[0] == 12 and torch.equal(v, kv.index_select(0, big[:12].long())),
      "case 4: over-capacity request must return the first cap pages")
v, n = r.get(kv, big, 14, 14 * PAGE, 8, big.tolist())
check(n == 0, f"case 4: complete pages must not be recopied, copied {n}")

# 5. invalid request id (dummy / missing writer) never retained
v, _ = r.get(kv, table, 2, 16, -2, table[:2].tolist())
check(v is None, "case 5: uid < 1 must not be retained")

# 6. two layers (distinct host tensors) are independent
kv_b = torch.randn(BLOCKS, HEADS, PAGE, W)
c = run_request(r, kv_b, 7, table, 40, 8)
c2 = run_request(r, kv, 7, table, 40, 8)
check(c and c2, "case 6")

# 7. A partial prefix-cache hit starts inside an existing page. Complete prefix pages
# stay immutable; the private partial tail is appended, becomes full, then grows past cap.
r_prefix = R(cap_pages=3)
prefix_kv = torch.randn(BLOCKS, HEADS, PAGE, W)
prefix_before = prefix_kv[table[:2].long()].clone()
c = run_request(r_prefix, prefix_kv, 101, table, 7 * PAGE + 1, 3, start=2 * PAGE + 3)
check(c and c[0] == 3 and all(n == 0 for n in c[1:]), f"case 7: prefix/cap copies {c}")
check(torch.equal(prefix_kv[table[:2].long()], prefix_before), "case 7: shared prefix was modified")

# 8. A -> B -> A with a shared first page but different partial tails. Each owner
# change must re-arm, even when A returns with exactly its old table and UID.
r_shared = R(cap_pages=3)
shared_kv = torch.randn(BLOCKS, HEADS, PAGE, W)
table_a = torch.tensor([0, 1, 2], dtype=torch.int32)
table_b = torch.tensor([0, 4, 5], dtype=torch.int32)
for uid, ids in ((201, table_a), (202, table_b), (201, table_a)):
    v, _ = r_shared.get(shared_kv, ids, 3, 2 * PAGE + 3, uid, ids.tolist())
    check(v is None, f"case 8: owner {uid} did not re-arm")
    v, n = r_shared.get(shared_kv, ids, 3, 2 * PAGE + 3, uid, ids.tolist())
    check(n == 3 and torch.equal(v, shared_kv[ids.long()]), f"case 8: owner {uid} got another owner's pages")

# 9. CoW of the actual partial tail, followed by an append with the same physical
# ID. Reaching an exact page boundary still requires that final tail refresh.
tail_source = shared_kv[2].clone()
shared_kv[6] = tail_source
tail_table = torch.tensor([0, 1, 6], dtype=torch.int32)
shared_kv[6, :, 3:5] += 100
v, n = r_shared.get(shared_kv, tail_table, 3, 2 * PAGE + 5, 201, tail_table.tolist())
check(n == 1 and torch.equal(v, shared_kv[tail_table.long()]), "case 9: partial CoW tail stale")
shared_kv[6, :, 5:] += 200
v, n = r_shared.get(shared_kv, tail_table, 3, 3 * PAGE, 201, tail_table.tolist())
check(n == 1 and torch.equal(v, shared_kv[tail_table.long()]), "case 9: completed tail stale")
v, n = r_shared.get(shared_kv, tail_table, 3, 3 * PAGE, 201, tail_table.tolist())
check(n == 0, "case 9: fully copied page should now be reusable")
check(torch.equal(shared_kv[2], tail_source), "case 9: CoW source modified")

# 10. Resume/cancel + external request-ID reuse: the runner MUST supply a fresh
# internal UID. Recycled physical blocks may now contain entirely different data.
for uid in (203, 204):
    shared_kv[tail_table.long()] += 1000
    v, _ = r_shared.get(shared_kv, tail_table, 3, 3 * PAGE, uid, tail_table.tolist())
    check(v is None, f"case 10: new incarnation {uid} did not re-arm")
    v, n = r_shared.get(shared_kv, tail_table, 3, 3 * PAGE, uid, tail_table.tolist())
    check(n == 3 and torch.equal(v, shared_kv[tail_table.long()]), f"case 10: stale incarnation {uid}")

# 11. An invalid UID must invalidate previously warm state, not just bypass it.
v, _ = r_shared.get(shared_kv, tail_table, 3, 3 * PAGE, -2, tail_table.tolist())
check(v is None, "case 11: invalid UID returned pages")
shared_kv[tail_table.long()] += 1000
v, _ = r_shared.get(shared_kv, tail_table, 3, 3 * PAGE, 204, tail_table.tolist())
check(v is None, "case 11: valid UID after invalidation must re-arm")
v, n = r_shared.get(shared_kv, tail_table, 3, 3 * PAGE, 204, tail_table.tolist())
check(n == 3 and torch.equal(v, shared_kv[tail_table.long()]), "case 11: invalidation left stale pages")

# 12. OOM on a later layer disables reuse on already allocated layers too.
# This checks correctness only; v6 still holds the old allocations (review finding).
oom_kv = torch.zeros_like(shared_kv)
r_shared.get(oom_kv, tail_table, 3, 3 * PAGE, 204, tail_table.tolist())
with patch.object(torch, "empty", side_effect=torch.OutOfMemoryError("injected test OOM")):
    v, n = r_shared.get(oom_kv, tail_table, 3, 3 * PAGE, 204, tail_table.tolist())
check(v is None and n == 0 and r_shared.failed and not r_shared.state, "case 12: OOM must fail closed")
v, _ = r_shared.get(shared_kv, tail_table, 3, 3 * PAGE, 204, tail_table.tolist())
check(v is None, "case 12: old layer still retained after OOM")

# 13. Exercise the actual staged wrapper on CPU, replacing only its Triton launch
# with a probe. Compare every selected K/V row with the host-table oracle, and
# require exactly one visit across retained + arena buckets. This does NOT test
# Triton arithmetic/softmax or CUDA lifetime; those still need a GPU test.
class BucketProbe:
    def __init__(self, host, ids):
        self.host, self.ids = host, ids
        self.seen, self.flags = {}, []

    def __getitem__(self, grid):
        return self.launch

    def launch(self, *args, **kwargs):
        _, keys, values, indices, rows = args[:5]
        lo, width = args[-2:]
        self.flags.append((kwargs["FIRST_BUCKET"], kwargs["LAST_BUCKET"]))
        check(width > 0 and keys.shape[0] == width, "case 13: wrong bucket extent")
        for row in rows.tolist():
            for col, token in enumerate(indices[row, :-1].tolist()):
                if token < 0 or not lo <= token // PAGE < lo + width:
                    continue
                page, offset = divmod(token, PAGE)
                actual = torch.cat((keys[page - lo, offset], values[page - lo, offset]), dim=-1)
                expected = self.host[int(self.ids[page]), :, offset]
                check(torch.equal(actual, expected), f"case 13: wrong K/V for token {token}, bucket {lo}")
                self.seen[row, col] = self.seen.get((row, col), 0) + 1


staged_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "qsa_sparse_paged_attention_staged")
staged_ns = {
    "torch": torch, "_STAGE_TIMER": None,
    "os": SimpleNamespace(environ={"VLLM_QSA_KVO_SPARSE_PREFILL": "0"}),
    "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
                              cdiv=lambda n, d: (n + d - 1) // d),
    "_select_config": lambda *args: (16, 4, 1, 1),
}
exec(compile(ast.Module([staged_node], []), "staged", "exec"), staged_ns)
for cap in (None, 1, 3, 7):
    for arena_pages in (2, 3):
        for pages in (1, 2, 3, 4, 7, 8, 10, 11):
            ids = table[:pages]
            host = torch.randn(BLOCKS, HEADS, PAGE, W)
            # Partial last page; select all live tokens plus padding, in permuted rows.
            seq = pages * PAGE - 3
            indices = torch.full((4, seq + 2), -1, dtype=torch.int32)
            rows = torch.tensor([3, 1], dtype=torch.int64)
            indices[rows, :seq] = torch.arange(seq, dtype=torch.int32)
            indices[rows, -1] = seq
            q = torch.zeros(4, HEADS, W // 2)
            arena = torch.empty(arena_pages, HEADS, PAGE, W)
            retained = None
            if cap is not None:
                retained = host[ids[:min(cap, pages)].long()].clone()
            probe = BucketProbe(host, ids)
            staged_ns["_qsa_sparse_staged_gqa_kernel"] = probe
            staged_ns["qsa_sparse_paged_attention_staged"](
                q, host, indices, ids, rows, pages - 1, arena, True, torch.empty_like(q),
                retained=retained)
            expected_visits = {(row, col): 1 for row in rows.tolist() for col in range(seq)}
            check(probe.seen == expected_visits, f"case 13: missing/duplicate tokens cap={cap}, arena={arena_pages}, pages={pages}")
            check(probe.flags == [(i == 0, i == len(probe.flags) - 1) for i in range(len(probe.flags))],
                  "case 13: FIRST/LAST bucket flags")

# 14. same-uid rollback (Codex review): retain 3 full pages, rewrite page 3, call again with a shorter sequence and
#     identical ids -> must not return the stale page 3 (v2 did); and the call after that re-copies it
if "old_len" in open(sys.argv[1]).read():
    kv = torch.zeros(BLOCKS, HEADS, PAGE, W)
    r = R(cap_pages=8)
    t14 = torch.tensor([4, 5, 6, 7], dtype=torch.int32)
    kv[t14[:3].long()] = torch.randn(3, HEADS, PAGE, W)
    r.get(kv, t14, 3, 3 * PAGE, 77, t14[:3].tolist())          # arm
    v, _ = r.get(kv, t14, 3, 3 * PAGE, 77, t14[:3].tolist())   # retain 3 complete pages
    check(v is not None, "case 14: retained after arming")
    kv[6] = torch.randn(HEADS, PAGE, W)                         # rewrite page 3 (rollback + new tokens)
    v, _ = r.get(kv, t14, 3, 3 * PAGE - 4, 77, t14[:3].tolist())
    check(v is None or torch.equal(v, kv.index_select(0, t14[:3].long())), "case 14: stale page after rollback")
    v, _ = r.get(kv, t14, 3, 3 * PAGE - 2, 77, t14[:3].tolist())
    check(v is None or torch.equal(v, kv.index_select(0, t14[:3].long())), "case 14: stale page after re-arm")
    # 15. allocation failure releases every layer's buffer
    r = R(cap_pages=4)
    kv_a, kv_b = torch.zeros(BLOCKS, HEADS, PAGE, W), torch.zeros(BLOCKS, HEADS, PAGE, W)
    for k in (kv_a, kv_a, kv_b):
        r.get(k, t14, 2, 2 * PAGE, 9, t14[:2].tolist())
    check(len(r.buf) == 1, "case 15: one layer allocated before the failure")
    with patch.object(torch, "empty", side_effect=torch.OutOfMemoryError("synthetic")):
        r.get(kv_b, t14, 2, 2 * PAGE, 9, t14[:2].tolist())
    check(r.failed and not r.buf, "case 15: buffers released after an allocation failure")

print("retain test:", "PASS" if not fails else "FAIL " + "; ".join(fails))
sys.exit(1 if fails else 0)
