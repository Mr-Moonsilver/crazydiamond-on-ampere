#!/usr/bin/env python3
"""Alternate decode-only and prefill-only steps under pipeline parallelism (vLLM 0.29.1rc1 image,
vllm/v1/core/sched/scheduler.py). Enabled by FLASH_PD_ALTERNATE=1; otherwise scheduling is unchanged.

With PP2 + Model Runner V2 a decoding request may only be scheduled every pp_size steps (sampled-token ring
cadence), while prefill chunks run every step. So once a long prompt arrives every step carries a ~1k-token
chunk (~130 ms) and each decoder gets one token per two such steps (~260 ms measured, vs 15.6 ms alone).
While both decoders and prefill work are pending this patch makes the steps alternate:
  * decode step: some decoder is eligible and the previous step was not a decode step. Prefill is deferred
    through the existing DP-balancing path (no running prefill chunks, no new prompt admission), so the
    batch is decode-only (CUDA graph, QSA row cache);
  * prefill step: everything else; running decoders are skipped (a decoder that became eligible on a
    prefill step waits one step and from then on runs on the decode parity).
Two decode steps never follow each other, so prefill gets at least every other step; decoders keep exactly
the pp_size cadence. A decoder's token then costs one prefill step plus one decode step instead of two
prefill steps.
usage: make_pdalt.py scheduler.image.py out_scheduler.py
"""
import difflib, sys

src = open(sys.argv[1]).read()
orig = src


def sub(text, old, new, count=1):
    assert text.count(old) == count, (old[:80], text.count(old))
    return text.replace(old, new)


src = sub(src, """        self.prefill_capacity_bound = False
        self.scheduler_reserve_full_isl = (
""", """        self.prefill_capacity_bound = False
        # flash-next: alternate decode-only / prefill-only steps under PP (FLASH_PD_ALTERNATE=1).
        self.pd_alternate = self.use_pp and os.environ.get("FLASH_PD_ALTERNATE", "0") == "1"
        self._pd_last_decode = False
        self.scheduler_reserve_full_isl = (
""")

src = sub(src, """        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)
""", """        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)

        # flash-next PD alternation (see make_pdalt.py): "D" = decode-only step, "P" = prefill-only step.
        pd_mode = None
        if self.pd_alternate:
            decoders = [r for r in self.running if not r.is_prefill_chunk]
            prefill_work = bool(self.waiting) or any(r.is_prefill_chunk for r in self.running)
            if decoders and prefill_work:
                eligible = any(self.current_step >= r.next_decode_eligible_step for r in decoders)
                pd_mode = "D" if eligible and not self._pd_last_decode else "P"
                defer_prefills = defer_prefills or pd_mode == "D"
        self._pd_last_decode = pd_mode == "D"
""")

src = sub(src, """            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue
""", """            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue

            if pd_mode == "P" and not request.is_prefill_chunk:
                # flash-next PD alternation: decoders wait for the next decode-only step.
                req_index += 1
                continue
""")

if "\nimport os\n" not in src:
    src = sub(src, "import itertools\n", "import itertools\nimport os\n")

open(sys.argv[2], "w").write(src)
open(sys.argv[2] + ".diff", "w").write("".join(difflib.unified_diff(
    orig.splitlines(True), src.splitlines(True), "a/scheduler.py", "b/scheduler.py")))
print("wrote", sys.argv[2])
