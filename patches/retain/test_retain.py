#!/usr/bin/env python3
"""CPU test of _QSARetainedHistory (S3) from a patched ops_qsa.py: the class is extracted and run on CPU tensors.

The engine writes each chunk's K/V to host before attention and never rewrites a page of a live request except by
appending; a copied-on-write tail gets a new block id. The test drives exactly that and checks that every returned
buffer equals a fresh copy of the host pages, that only changed pages are copied, and that a new request id (block
ids reused with other content) never sees the previous request's pages.
usage: test_retain.py ops_qsa.py
"""
import ast, sys
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
            check(torch.equal(view, kv.index_select(0, table[:pages].long())), f"uid {uid}: stale buffer at seq {seq}")
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

# 4. more pages than the buffer: old path (None), and the state is dropped
big = torch.arange(13, dtype=torch.int32)
v, _ = r.get(kv, big, 13, 13 * PAGE, 6, big.tolist())
check(v is None, "case 4: over-capacity request must use the old path")
v, _ = r.get(kv, table, 3, 24, 6, table[:3].tolist())
check(v is None, "case 4: state must be re-armed after the old path")

# 5. invalid request id (dummy / missing writer) never retained
v, _ = r.get(kv, table, 2, 16, -2, table[:2].tolist())
check(v is None, "case 5: uid < 1 must not be retained")

# 6. two layers (distinct host tensors) are independent
kv_b = torch.randn(BLOCKS, HEADS, PAGE, W)
c = run_request(r, kv_b, 7, table, 40, 8)
c2 = run_request(r, kv, 7, table, 40, 8)
check(c and c2, "case 6")

print("retain test:", "PASS" if not fails else "FAIL " + "; ".join(fails))
sys.exit(1 if fails else 0)
