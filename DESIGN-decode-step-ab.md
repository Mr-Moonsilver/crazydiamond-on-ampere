# DESIGN — decode-step levers A / B (stage-0 launch-ahead, worker-side prepare-ahead), 13 Nov 2026

Status: **design only, not built** (brief item 3). SIZING.md shows why: no record has a handler row in a clean
decode-only window, and in overlap steps the host is ahead of the device on 96–99.9 % of forward launches. A or B can
only pay where the host is behind, and the records show no such step. The investigator's realistic figures
(A 203, B 283 agent-s) assume a host-bound clean step, and that assumption has not been measured. This document states
what each would be, the smaller of the two (B), and the reading of FLASH_S5_DECODE_ANATOMY that decides whether B is
built.

## The step being targeted
Clean two-cohort decode on the native arm: TP2×PP2, async scheduling, stage 0 = ranks 0/1, stage 1 = ranks 2/3 (it
samples). Per engine round, for one cohort:

    core: schedule N  ->  stage 0: prepare N (prep_inputs, prep_attn_meta, prep_attn)  ->  forward₀  ->  pp_send
          stage 1: recv  ->  forward₁  ->  sample_tokens (D's target)  ->  PP broadcast of the tokens  ->  core: update

Stage 0's host prep for step N+1 needs the tokens of step N only for `input_ids`. Positions, slot mapping, block tables,
sequence lengths and attention metadata depend on the step count, not on the token values. vLLM's async scheduling
already lets the core schedule N+1 before N's tokens are back (placeholders). Each worker still runs its prep inside the
RPC for N+1, so on stage 0 the prep follows the arrival of that RPC.

Host time on stage 0 per step, measured in overlap rows (np-off, ranks 0/1 p50): prep_attn_meta 1.23 ms, prep_inputs
0.36 ms, prep_attn 0.24 ms, pp_send 0.31 ms. That is ~2.1 ms. It is off the critical path whenever the device is still
busy when the prep finishes, which is 96 % of overlap launches.

## A — engine-side launch-ahead
Send stage 0's RPC for N+1 as soon as N's stage-0 forward is enqueued, not when the core's batch queue pops N. It is a
change to the core's batch-queue depth on the native arm: max_concurrent_batches is PP = 2 today, and A would make it 3
in clean windows only.
* Touches: EngineCore.step_with_batch_queue (stock) through executor_hooks.before_step_with_batch_queue.
  The scheduler's placeholders then run two steps deep. KV block allocation happens one step earlier. A preemption
  or a finish on N must cancel N+1's prepared batch.
* Risk: high. A third in-flight batch changes stock's scheduler invariants (num_computed_tokens advanced twice before
  either output). The sp arm must not see it, the yield window and lever A run on the same seam, and a preemption
  needs an exact undo. Size: large (engine core + scheduler + driver).

## B — worker-side prepare-ahead (the smaller one)
On stage 0 only, at the end of handling RPC N (after forward₀ is enqueued and pp_send is issued), the worker builds
N+1's input batch for the same request set, assuming each running request advances by exactly one token. Everything is
built except `input_ids`. When RPC N+1 arrives, the worker checks that its scheduler output matches the assumption, then
fills in `input_ids` from the scheduler output and launches. The assumption is: the same request ids in the same order,
num_scheduled_tokens 1 each, no new, finished, preempted or resumed requests, and no block-table change beyond the
appended slot the prepared copy already computed. On any mismatch the prepared state is discarded and stock prep runs
unchanged.
* Flag: FLASH_S5_DECODE_PREPARE_AHEAD, default 0. 1 = native arm only, the decode_step gate (the same per-RPC wire tag
  bit as D), stage 0 only, decode-only steps only (no prefill rows in N or N+1).
* Hooks: worker_hooks around the V2 runner's `prepare_inputs` / attention-metadata build, on the instance. The
  prepared copy lives in a second set of the runner's persistent input buffers. Not checked yet: whether the V2
  runner's async-scheduling buffers can hold it, or whether B needs its own set (a few MB at 8 rows). A CUDA event
  orders the copy with forward N. No change to the core,
  the scheduler or stage 1.
* Identity: the metadata tensors handed to forward N+1 must be bitwise equal to stock's prep for N+1.
  Test: run stock prep and B's prepared-then-validated prep on the same scheduler outputs over a replay-shaped decode
  trace (CPU, the runner's real `prepare_inputs` against the image source) and compare every tensor; mutants for a
  stale block table, a reordered batch and a finished request.
* Cost when it misses: one discarded host build (~1.8 ms of host work that overlapped the device anyway); no device
  work is wasted.
* Size: medium (one worker module plus the tag bit; no engine-core change).

## What decides B (item 4's window)
FLASH_S5_DECODE_ANATOMY gives, per clean 20 s window:
* per rank, the handler rows with `idle before launch` (Stream.query() before forward), host prep seams and dev_forward;
* on the core, one stamp per engine step (the round time).
Build B if **both** hold on stage 0 (ranks 0/1) in the decode-only windows of an off/on pair:
1. the stream is already idle at forward launch on ≥ 30 % of steps (overlap steps: 3.7–3.9 %);
2. round − dev_forward₀ − dev_forward₁ − dev_sample ≥ 1.0 ms at p50, i.e. a gap that host prep could explain.
Value if they hold: the gap removed per clean step × 742 agent-s per ms × 19 % (clean share of n3 decode, 3,547 of
18,685 agent-s). So 1 ms of gap is about 140 agent-s, and B reaches 150 agent-s realistic only if it removes about
1.1 ms per clean step. If (1) or (2) fails, B is dropped, and so is A, since A needs the same host-bound step.

## Not in this design
* Any stock-vLLM tuning (max_concurrent_batches, async scheduling settings): the stock arm stays as it is.
* Stage 1: its prep is on the critical path only behind stage 0's pp_send, which B does not move.
