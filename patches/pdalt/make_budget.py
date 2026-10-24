#!/usr/bin/env python3
"""Scheduler v6 (goal item 3): prefill chunk size by load + a step-time budget with a learned cost model and
infeasibility reporting. On top of make_sfirst.py (scheduler.py.v8 -> v9).

Chunk by load, FLASH_CHUNK_BUSY=N: the engine is configured with a larger chunk (PREFILL_CHUNK, e.g. 2048: a fresh
prompt alone prefills ~10 % faster, sweep 1); while any decoder is running, a step's token budget is capped at N
(e.g. 1024), so decoders never sit behind the large chunk.
Step budget, FLASH_STEP_TARGET_MS=T (with decoders running, on non-decode steps): the prefill budget is chosen so
the step should take about T ms. Cost model: wall ms per prefill token, learned per history bin (32k tokens of the
longest running prefill) as an EWMA (0.8 / 0.2) over previous steps that scheduled >= 128 prefill tokens, where a
step's wall time is the time to the next schedule() call. That proxy over-attributes under pipeline overlap (24 Oct
review); it is conservative (smaller chunks) and its estimates are logged so the error is visible. Budget =
T / (ms per token), rounded down to 16, at least FLASH_STEP_MIN_CHUNK (default 256). When even the minimum chunk is
predicted to exceed T, the step runs the minimum chunk and counts as infeasible (bg_infeasible): the target cannot
be met by slicing at that history length -- the measured reason the plan needs S5.
Stats add bg_budget (last budget), bg_infeasible, bg_k (ms per token per bin). Unset, v8 behaviour is unchanged.
usage: make_budget.py scheduler.py.v8 scheduler.py.v9
"""
import difflib, sys

src = orig = open(sys.argv[1]).read()


def sub(old, new):
    global src
    assert src.count(old) == 1, (old[:80], src.count(old))
    src = src.replace(old, new)


sub("""        self._sched_stats.update(sf_steps=0, sf_skips=0)
""", """        self._sched_stats.update(sf_steps=0, sf_skips=0)
        # flash-next v6 chunk by load + step budget (make_budget.py)
        self._chunk_busy = int(os.environ.get("FLASH_CHUNK_BUSY", "0"))
        self._bg_target = float(os.environ.get("FLASH_STEP_TARGET_MS", "0"))
        self._bg_min = int(os.environ.get("FLASH_STEP_MIN_CHUNK", "256"))
        self._bg_k = {}       # history bin -> EWMA wall ms per prefill token
        self._bg_prev = None  # (prefill tokens, bin, schedule time) of the previous step
        if self._chunk_busy or self._bg_target:
            self._sched_stats.update(bg_budget=0, bg_infeasible=0, bg_k={})
""")

sub("""        # flash-next v5 short-first: while a short prefill is pending, long prefills yield (aging bound)
""", """        # flash-next v6 (make_budget.py): learn ms per prefill token, then size this step's prefill budget
        bg_bin = None
        if self._chunk_busy or self._bg_target:
            now_b = time.monotonic()
            if self._bg_prev is not None and self._bg_prev[0] >= 128:
                k = (now_b - self._bg_prev[2]) * 1000 / self._bg_prev[0]
                old = self._bg_k.get(self._bg_prev[1])
                self._bg_k[self._bg_prev[1]] = k if old is None else 0.8 * old + 0.2 * k
                self._sched_stats["bg_k"] = {b: round(v, 4) for b, v in sorted(self._bg_k.items())}
            hist = max((r.num_computed_tokens for r in self.running if r.is_prefill_chunk), default=0)
            bg_bin = min(hist // 32768, 8)
            if pd_mode != "D" and token_budget > 0 and any(not r.is_prefill_chunk for r in self.running):
                if self._chunk_busy:
                    token_budget = min(token_budget, self._chunk_busy)
                k = self._bg_k.get(bg_bin)
                if self._bg_target and k:
                    want = int(self._bg_target / k) // 16 * 16
                    if want < self._bg_min:
                        self._sched_stats["bg_infeasible"] += 1
                    token_budget = min(token_budget, max(self._bg_min, want))
                self._sched_stats["bg_budget"] = token_budget
            self._bg_t0 = now_b
        # flash-next v5 short-first: while a short prefill is pending, long prefills yield (aging bound)
""")

sub("""        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
""", """        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        if bg_bin is not None:
            ndec_b = sum(1 for r in scheduled_running_reqs if not r.is_prefill_chunk)
            self._bg_prev = (total_num_scheduled_tokens - ndec_b, bg_bin, self._bg_t0) if pd_mode != "D" else None
""")
open(sys.argv[2], "w").write(src)
sys.stdout.writelines(difflib.unified_diff(orig.splitlines(True), src.splitlines(True), "scheduler.py.v8", "scheduler.py.v9"))
