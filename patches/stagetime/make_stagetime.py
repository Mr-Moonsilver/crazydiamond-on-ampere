#!/usr/bin/env python3
"""Lightweight timing of the QSA staged (host-KV) prefill attention (ops/qsa.py of the vLLM image).

QSA_STAGE_TIMING=1 enables it; otherwise the patched file behaves exactly like its input. Pattern adapted from
DeepInfer's graph timer: CUDA events from a reused pool, records read only after their last event completed
(event.query(), never a stream sync), a bounded ring that drops and counts instead of stalling serving. Every 5 s each
worker logs, per history-length bin (pages of 4,048 tokens): calls, query rows, history pages, sparse-union use and
coverage, pages moved, and GPU time split into pre-work (union/remap), host->arena copy, attention, whole call.
Event-record nodes are skipped while a CUDA graph is being captured.

usage: make_stagetime.py ops_qsa.py.in ops_qsa.py.out
"""
import difflib, sys

src = open(sys.argv[1]).read()
orig = src


def sub(text, old, new, count=1):
    assert text.count(old) == count, (old[:80], text.count(old))
    return text.replace(old, new)


src = sub(src, "import os\nimport torch\n", "import collections\nimport os\nimport time\nimport torch\n")

src = sub(src, """def qsa_sparse_paged_attention_staged(
""", '''class _QSAStageTimer:
    """QSA_STAGE_TIMING=1: bounded CUDA-event timing of the staged prefill attention (see make_stagetime.py)."""

    BINS = (12, 25, 37, 50, 1 << 30)  # history pages: <=49k, <=101k, <=150k, <=202k tokens, more
    RING = 4096

    def __init__(self):
        from vllm.logger import init_logger
        self.log = init_logger("vllm.qsa.stagetime")
        self.pending = collections.deque()
        self.free = []
        self.dropped = 0
        self.agg = {}
        self.last = time.monotonic()

    def mark(self):
        e = self.free.pop() if self.free else torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def begin(self, rows, pages):
        if torch.cuda.is_current_stream_capturing():
            return None
        self._drain()
        if len(self.pending) >= self.RING:
            self.dropped += 1
            return None
        return {"rows": rows, "pages": pages, "sparse": False, "coverage": None, "moved": pages, "ev": [self.mark()]}

    def end(self, t):
        t["ev"].append(self.mark())
        self.pending.append(t)

    def _drain(self):
        while self.pending and self.pending[0]["ev"][-1].query():
            t = self.pending.popleft()
            ev = t["ev"]
            b = ev[1:-1]  # per bucket: copy start, copy end (= attention start), attention end
            copy = sum(b[i].elapsed_time(b[i + 1]) for i in range(0, len(b) - 2, 3))
            attn = sum(b[i + 1].elapsed_time(b[i + 2]) for i in range(0, len(b) - 2, 3))
            pre = ev[0].elapsed_time(b[0]) if b else 0.0
            span = ev[0].elapsed_time(ev[-1])
            k = next(x for x in self.BINS if t["pages"] <= x)
            a = self.agg.setdefault(k, [0, 0, 0, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0])
            a[0] += 1; a[1] += t["rows"]; a[2] += t["pages"]; a[3] += t["sparse"]
            a[4] += t["coverage"] or 0.0; a[5] += t["moved"]
            a[6] += pre; a[7] += copy; a[8] += attn; a[9] += span; a[10] += t["coverage"] is not None
            self.free.extend(ev)
        now = time.monotonic()
        if self.agg and now - self.last >= 5.0:
            parts = []
            for k in sorted(self.agg):
                n, rows, pages, sp, cov, moved, pre, copy, attn, span, ncov = self.agg[k]
                parts.append(f"pages<={k if k < 1 << 30 else 'max'}: n={n} rows={rows / n:.0f} pages={pages / n:.1f} "
                             f"sparse={sp / n:.0%} cov={cov / ncov if ncov else float('nan'):.2f} moved={moved / n:.1f}p "
                             f"pre={pre / n:.2f} copy={copy / n:.2f} attn={attn / n:.2f} span={span / n:.2f} ms")
            self.log.info("QSA stage timing (%.1f s, per call): %s | dropped %d", now - self.last, " ; ".join(parts),
                          self.dropped)
            self.agg = {}
            self.last = now


_STAGE_TIMER = _QSAStageTimer() if os.environ.get("QSA_STAGE_TIMING", "0") == "1" else None


def qsa_sparse_paged_attention_staged(
''')

src = sub(src, """    num_pages = max_logical_page + 1
    unique_tokens = None
""", """    num_pages = max_logical_page + 1
    unique_tokens = None
    _t = _STAGE_TIMER.begin(num_rows, num_pages) if _STAGE_TIMER is not None else None
""")

src = sub(src, """        if unique_valid.numel() < 0.8 * num_pages * page_size:
""", """        if _t is not None:
            _t["coverage"] = unique_valid.numel() / max(1, num_pages * page_size)
        if unique_valid.numel() < 0.8 * num_pages * page_size:
""")

src = sub(src, """    num_buckets = triton.cdiv(num_pages, pages_in_arena)

    group_size = q.shape[1] // num_kv_heads
""", """    num_buckets = triton.cdiv(num_pages, pages_in_arena)
    if _t is not None:
        _t["sparse"] = unique_tokens is not None
        _t["moved"] = num_pages

    group_size = q.shape[1] // num_kv_heads
""")

src = sub(src, """        staged = arena[: page_hi - page_lo]
        if unique_tokens is None:
""", """        staged = arena[: page_hi - page_lo]
        if _t is not None:
            _t["ev"].append(_STAGE_TIMER.mark())
        if unique_tokens is None:
""")

src = sub(src, """        k_arena, v_arena = staged.transpose(1, 2).split(head_dim, dim=-1)
        _qsa_sparse_staged_gqa_kernel[(num_rows, num_kv_heads)](
""", """        if _t is not None:
            _t["ev"].append(_STAGE_TIMER.mark())
        k_arena, v_arena = staged.transpose(1, 2).split(head_dim, dim=-1)
        _qsa_sparse_staged_gqa_kernel[(num_rows, num_kv_heads)](
""")

src = sub(src, """            num_warps=partial_warps,
            num_stages=2,
        )


_QSA_ARENA_CACHE""", """            num_warps=partial_warps,
            num_stages=2,
        )
        if _t is not None:
            _t["ev"].append(_STAGE_TIMER.mark())
    if _t is not None:
        _STAGE_TIMER.end(_t)


_QSA_ARENA_CACHE""")

open(sys.argv[2], 'w').write(src)
sys.stdout.writelines(difflib.unified_diff(orig.splitlines(True), src.splitlines(True), 'ops_qsa.py.in', 'ops_qsa.py.out'))
