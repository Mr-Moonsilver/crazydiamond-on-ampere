#!/usr/bin/env python3
"""Scheduler v5 (goal item 3): short prefills before long ones -- an agent's next turn must not wait behind a fresh
200k prompt. Applied on top of make_pdalt.py's output (scheduler.py.v7 -> v8).

Today the scheduler serves running requests first in arrival order, then the waiting queue FCFS, with a 1,024-token
step budget. A long prompt mid-prefill takes the whole budget for ~27 s, so a 2k append arriving behind it (an agent
turn on a cached 150k history) gets its first token only when the long prompt is done.
FLASH_SHORT_FIRST=N (tokens, 0 = off): a request is short when its uncached prefill is at most N tokens -- for a
waiting request, its prompt minus the prefix-cache hit (looked up with the coordinator's side-effect-free
find_longest_cache_hit, memoised per request and length); for a running prefill, its remaining tokens. While a short
prefill is pending, on non-decode steps:
  * running prefill chunks that are not short are skipped this step;
  * waiting requests that are not short are deferred to the step's skipped list (they keep their queue position).
A waiting short request only counts while a request slot is free (else nothing could use the yielded budget).
Aging: after FLASH_SHORT_FIRST_MAX_RUN (default 8) consecutive deferring prefill steps, one step runs without
deferral, so a long prompt keeps at least 1/(MAX_RUN+1) of the prefill steps under a stream of short turns.
v8.1 (Codex round 2): decode-only steps no longer reset the aging counter (with PD alternation every P step started
at 1 and aging never fired); the cache-hit estimate is looked up once per scheduling pass, never memoised across
passes (a memo survived eviction or a sibling publishing the prefix); and a deferring step that schedules no prefill
token at all (the short request could not be admitted: KV, stale output, encoder, preemption ...) makes the next
prefill step run without deferral (sf_empty counts them), so a non-admissible short request cannot stall prefill.
v8.2 (Codex round 3): "no prefill scheduled" is counted from the scheduled requests that are prefill chunks, not as
all tokens minus one per decoder (wrong with draft tokens). Decode scheduling, PD
alternation and the parity cap are unchanged. Stats add sf_steps (steps that deferred) and sf_skips (deferrals).
usage: make_sfirst.py scheduler.py.v7 scheduler.py.v8
"""
import difflib, sys

src = orig = open(sys.argv[1]).read()


def sub(old, new):
    global src
    assert src.count(old) == 1, (old[:80], src.count(old))
    src = src.replace(old, new)


sub("""        self._pd_d_run = 0
        self.scheduler_reserve_full_isl = (
""", """        self._pd_d_run = 0
        # flash-next v5 short-first (make_sfirst.py)
        self._sf_max = int(os.environ.get("FLASH_SHORT_FIRST", "0"))
        self._sf_max_run = int(os.environ.get("FLASH_SHORT_FIRST_MAX_RUN", "8"))
        self._sf_run = 0
        self._sf_memo = {}  # request id -> (num_tokens, uncached tokens), this scheduling pass only
        self._sf_bypass = False
        self._sched_stats.update(sf_steps=0, sf_skips=0, sf_empty=0, sf_aged=0)
        self.scheduler_reserve_full_isl = (
""")

sub("""    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
""", """    def _sf_uncached(self, request):
        # flash-next v5: prefill tokens still to compute (waiting: prompt minus the prefix-cache hit, no side effects)
        if request.num_computed_tokens > 0:
            return request.num_tokens - request.num_computed_tokens
        memo = self._sf_memo.get(request.request_id)
        if memo is not None and memo[0] == request.num_tokens:
            return memo[1]
        hit = 0
        kvm = self.kv_cache_manager
        if kvm.prefix_cache_lookup_enabled(request):
            hit = kvm.coordinator.find_longest_cache_hit(request.block_hashes, request.num_tokens - 1)[1]
        self._sf_memo[request.request_id] = (request.num_tokens, request.num_tokens - hit)
        return request.num_tokens - hit

    def _sf_short_pending(self):
        # flash-next v5: ids of short prefills (waiting and ready, or running mid-prefill)
        short = set()
        admissible = len(self.running) + self.num_waiting_for_streaming_input < self.max_num_running_reqs
        for r in (itertools.chain(self.waiting, self.skipped_waiting) if admissible else ()):
            if not self._is_blocked_waiting_status(r.status) and self._sf_uncached(r) <= self._sf_max:
                short.add(r.request_id)
        for r in self.running:
            if r.is_prefill_chunk and r.num_tokens - r.num_computed_tokens <= self._sf_max:
                short.add(r.request_id)
        return short

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
""")

sub("""        while req_index < len(self.running) and token_budget > 0:
""", """        # flash-next v5 short-first: while a short prefill is pending, long prefills yield (aging bound)
        sf_short = set()
        if self._sf_max and pd_mode != "D":
            self._sf_memo = {}
            sf_short = self._sf_short_pending()
            if sf_short and (self._sf_run >= self._sf_max_run or self._sf_bypass):
                self._sched_stats["sf_aged" if self._sf_run >= self._sf_max_run else "sf_empty"] += 1
                sf_short = set()
            self._sf_bypass = False
            self._sf_run = self._sf_run + 1 if sf_short else 0  # decode-only steps leave the counter alone
            self._sched_stats["sf_steps"] += bool(sf_short)
        while req_index < len(self.running) and token_budget > 0:
""")

sub("""            if dec_cap is not None and not request.is_prefill_chunk and ndec_sched >= dec_cap:
""", """            if (sf_short and request.is_prefill_chunk and request.request_id not in sf_short):
                # flash-next v5 short-first: a long prefill chunk yields this step to a short prefill
                self._sched_stats["sf_skips"] += 1
                req_index += 1
                continue

            if dec_cap is not None and not request.is_prefill_chunk and ndec_sched >= dec_cap:
""")

sub("""                if (
                    request.num_stale_output_tokens > 0
                    and not request.drop_stale_output
                ):""", """                if sf_short and request_id not in sf_short:
                    # flash-next v5 short-first: a long waiting request is deferred to the next step
                    self._sched_stats["sf_skips"] += 1
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                if (
                    request.num_stale_output_tokens > 0
                    and not request.drop_stale_output
                ):""")

sub("""        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
""", """        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        prefill_scheduled_tokens = sum(num_scheduled_tokens.get(r.request_id, 0) for r in itertools.chain(
            scheduled_running_reqs, scheduled_new_reqs, scheduled_resumed_reqs) if r.is_prefill_chunk)
        if sf_short:
            # flash-next v8.2: a deferring step that scheduled no prefill at all -> next prefill step without deferral
            self._sf_bypass = prefill_scheduled_tokens <= 0
""")

open(sys.argv[2], "w").write(src)
sys.stdout.writelines(difflib.unified_diff(orig.splitlines(True), src.splitlines(True), "scheduler.py.v7", "scheduler.py.v8"))
