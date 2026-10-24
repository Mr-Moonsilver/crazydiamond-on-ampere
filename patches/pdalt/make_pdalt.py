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

v2 (24 Oct 2026, Codex's audit): prefill work only counts when it is feasible -- a running prefill chunk, or a
waiting / skipped_waiting request AND a free request slot. A prefill step that still schedules nothing (memory or
status blocked) pauses alternation until the waiting set, the running set or a running prefill chunk changes, so a
blocked queue costs at most one empty step instead of every other step. FLASH_SCHED_STATS=N logs counters every N
steps: decode/prefill/plain steps, empty prefill steps, and decoders scheduled on even vs odd steps (PP parity).
FLASH_PARITY_BALANCE=1 (needs PP; independent of FLASH_PD_ALTERNATE): on steps that are not PD-alternation decode
steps, schedule at most ceil(decoders / 2) decoders. With PP2 every decoder runs every other step, so decoders
can pile onto one step parity; a decoder skipped once runs on the other parity from then on, so the two
micro-batches even out at the cost of one step per moved decoder. Only decoders actually scheduled count against
the cap (a decoder rejected later, e.g. by allocation, does not use a slot). Stats then also report the live
cohorts (running decoders by the parity of their last scheduled decode step) and confirmed parity changes.
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
        self._pd_blocked_sig = None  # (waiting, skipped, running) when a prefill step admitted nothing
        self._sched_stats_every = int(os.environ.get("FLASH_SCHED_STATS", "0"))
        self._sched_stats = dict(D=0, P=0, plain=0, empty_P=0, dec_even=0, dec_odd=0, cap_skips=0,
                                 parity_changes=0)
        self.parity_balance = self.use_pp and os.environ.get("FLASH_PARITY_BALANCE", "0") == "1"
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
            running_chunk = any(r.is_prefill_chunk for r in self.running)
            queued = len(self.waiting) + len(self.skipped_waiting)
            free_slot = len(self.running) + self.num_waiting_for_streaming_input < self.max_num_running_reqs
            sig = (len(self.waiting), len(self.skipped_waiting), len(self.running))
            if running_chunk or sig != self._pd_blocked_sig:
                self._pd_blocked_sig = None
            prefill_work = running_chunk or (queued > 0 and free_slot and self._pd_blocked_sig is None)
            if decoders and prefill_work:
                eligible = any(self.current_step >= r.next_decode_eligible_step for r in decoders)
                pd_mode = "D" if eligible and not self._pd_last_decode else "P"
                defer_prefills = defer_prefills or pd_mode == "D"
        self._pd_last_decode = pd_mode == "D"
        # flash-next parity balance: cap decoders per non-"D" step at half of all decoders (rounded up).
        dec_cap = None
        if self.parity_balance and pd_mode != "D":
            ndec_all = sum(1 for r in self.running if not r.is_prefill_chunk)
            dec_cap = (ndec_all + 1) // 2 if ndec_all > 1 else None
        ndec_sched = 0
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

            if dec_cap is not None and not request.is_prefill_chunk and ndec_sched >= dec_cap:
                # flash-next parity balance: this decoder waits a step and moves to the other parity.
                self._sched_stats["cap_skips"] += 1
                req_index += 1
                continue
""")

src = sub(src, """            # Schedule the request.
            scheduled_running_reqs.append(request)
            prefill_scheduled |= request.is_prefill_chunk
""", """            # Schedule the request.
            scheduled_running_reqs.append(request)
            prefill_scheduled |= request.is_prefill_chunk
            if not request.is_prefill_chunk:
                # flash-next: only scheduled decoders count against the parity cap; track each decoder's cohort.
                ndec_sched += 1
                parity = self.current_step % 2
                if getattr(request, "_flash_parity", parity) != parity:
                    self._sched_stats["parity_changes"] += 1
                request._flash_parity = parity
""")

src = sub(src, """        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
""", """        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())

        # flash-next PD alternation v2: a prefill step that admitted nothing pauses alternation until the queue or
        # the running set changes (otherwise a blocked queue empties every other step).
        if pd_mode == "P" and total_num_scheduled_tokens == 0:
            self._pd_blocked_sig = (len(self.waiting), len(self.skipped_waiting), len(self.running))
        if self._sched_stats_every:
            st = self._sched_stats
            st[pd_mode or "plain"] += 1
            st["empty_P"] += pd_mode == "P" and total_num_scheduled_tokens == 0
            ndec = sum(1 for r in scheduled_running_reqs if not r.is_prefill_chunk)
            st["dec_even" if self.current_step % 2 == 0 else "dec_odd"] += ndec
            if self.current_step % self._sched_stats_every == 0:
                cohorts = [getattr(r, "_flash_parity", None) for r in self.running if not r.is_prefill_chunk]
                logger.info("flash-sched stats step %d: %s cohorts even=%d odd=%d none=%d", self.current_step, st,
                            cohorts.count(0), cohorts.count(1), cohorts.count(None))
""")

if "\nimport os\n" not in src:
    src = sub(src, "import itertools\n", "import itertools\nimport os\n")

open(sys.argv[2], "w").write(src)
open(sys.argv[2] + ".diff", "w").write("".join(difflib.unified_diff(
    orig.splitlines(True), src.splitlines(True), "a/scheduler.py", "b/scheduler.py")))
print("wrote", sys.argv[2])
