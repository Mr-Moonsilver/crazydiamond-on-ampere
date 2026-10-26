# E3 correctness protocol v2 (operator decision, 26 Oct 2026 ~13:40 CET)

Supersedes the free-running exact token-ID oracle as the correctness *gate* for q17b and every
later E3 rung. The oracle is kept as recorded evidence. Nothing here adds a tolerance to an
exact gate; it moves the model-level comparison onto the rule the gate spec already names.

## Why (established, not assumed)

1. The production engine is not bitwise reproducible at the model level with MoE on, and the
   mechanism is named: `_count_and_sort_expert_tokens` claims slots with `atomicAdd`, Marlin's
   Stream-K partition depends on tile position, so identical routing gives different last-bit
   results (q11v/q11v2 traces, 25 Oct; Codex `reviews/codex-20261025aa-marlin-position.md`).
   The slice exact oracle passed only with MoE off (q11v B, q11aa).
2. The engine-level numerical gate for exactly this noise already exists and is baselined:
   teacher-forced batch invariance, rule v3 (`harness/teacher_forced.py`; q3d PASS 24 Oct;
   q15 baseline 25 Oct: noise 0.074/0.176, reordered 0.092/0.179, concurrent 0.127/0.137
   against frozen tolerances 0.375/0.5, flips only at near ties; `tmp/q15-sample/tf-baseline.json`).
   `docs/E2-CANDIDATE-GATE.md` lines 53-54: "the exactness rule stays separate from model-level
   noise rules: serial per-layer resident-vs-UVA equality is bitwise; teacher-forced and
   warm/cold use rule v3 for the model-level comparison."
3. q17b-full8 (26 Oct): two full-mode control runs differ on 7/8 streams in free running, with
   6/7 first differences before any pause or route change (independent Opus review,
   `reviews/opus-20261026-q17b-full8-independent-review.md`). The rung-2 oracle
   (`docs/e3-briefs/q17b-short-correctness.md`) was built to report exactly this as
   INCONCLUSIVE rather than invent a tolerance. It did its job; as a gate it cannot pass by
   construction, and every further slot would end INCONCLUSIVE.

## The gate from the next slot on

| # | Check | Rule | Verdict effect |
|---|---|---|---|
| 1 | Data exactness: retained pages, history, recurrent/conv/PLE states, split-bank contents | bitwise, unchanged | FAIL is immutable |
| 2 | Model-level numerical: rule v3, frozen tolerances, q15 baseline | teacher-forced (positions fixed) on the **decode path inside the split window**: forced continuation, per-step logits recorded while the split graphs execute. Control pair under the same instrument must itself be within rule v3 (control stability), else INCONCLUSIVE. Split vs full outside rule v3, or a non-near-tie flip, = FAIL | gates W1 |
| 3 | Launch-config identity: per graph node, full vs split (kernel symbol, grid, block, shared memory, every launch field CUPTI exposes) | exact; result PASS or DIFFER with the kernel list | diagnostic, always reported; does not alone unlock or block W1 (a DIFFER on SM-count-sized grids is information, not corruption) |
| 4 | Placement | per-replay exact record count = static kernels x replays; loss flags retained, never cleared | gates W1 |
| 5 | Free-running exact token oracle (full/full/split/full) | unchanged, recorded | evidence only; INCONCLUSIVE allowed |
| 6 | Leak (partial tail, cross-session), lifetime, canary | unchanged | still binding for any tag |

W1 (performance) runs only on 1, 2 and 4 PASS. No retry, no picking among runs.

Implementation of 2 is Codex's call between: (a) a sampler override inside the existing probe
patch that forces the fixed continuation and records the logits (preferred: full 256-step
coverage, positions fixed, rule v3 semantics exact); (b) free running with per-step top-k
logprobs compared up to the first divergence, the divergence itself required to be a near tie,
compared-position coverage reported and a minimum coverage required for PASS. Either way the
tolerances, probe construction and baseline are rule v3 as frozen; the only new element is
that the forced positions are decoded, not prefilled.

## Not allowed

Engine numerics (batch-invariant or deterministic modes), sampling, the SM split, KV, settings
or any parameter; any widening of rule v3; any retry or selection among runs; clearing loss
flags; counting a node present once per window as placed.

## Consequences accepted

- Residual risk: rule v3 bounds the logit deviation at fixed positions over the probe set; a
  corruption below 0.375 logits at every probed position would pass. Wrong-page or stale-history
  reads produce large deviations, not last-bit ones, so this is the known residual, not a new
  one; the leak canary and live agent replay remain the behavioural backstop.
- Cost: rule v3 took ~20 min in q15 on the prefill path; the decode-path variant inside the
  split window is 256 steps x 8 streams and should be cheaper. Codex estimates before the slot;
  the slot budget includes it.
- Rung 4 (prefill under the split) uses the original `teacher_forced.py` against the q15
  baseline directly; no further design needed there.
- The free-running oracle stops deciding PASS. It could not; keeping it as the gate would have
  burned every slot.

Decision: operator, on the independent review and the 25 Oct record; recorded by Claude.
