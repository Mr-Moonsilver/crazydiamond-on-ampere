# E2-serial candidate: the GPU gate it will face (box-side spec, 25 Oct 2026)

Written so Codex's unarmed candidate queue and the box-side lifecycle meet in the middle. The candidate is the
serial-only retention adapter in the ordinary engine (opt-in switch, default off), delivered as a checksummed overlay
directory generated from pinned sources plus its SHA256SUMS. Nothing here is armed; production stays STOPPED and is
never restored by the gate queue. One GPU job at a time; the gate queue owns the box for its slot.

## Lifecycle (reuse `harness/runs/q12b.sh` + the gate queue pattern of `qN_pp_grouping.draft.sh`, `q10.sh` restore)

1. Preflight with the engine STOPPED: harness bundle checksums; `/tmp/setcfg.py` sha pin; image pin
   (`/tmp/flash-image.txt`); deployed launcher is one of the known shas; candidate overlay = overlay-v16 + the
   generated files only (byte compare of every other file; `SHA256SUMS` + `core/SHA256SUMS` verified); GPUs idle.
2. Identity: explicit `engine_identity.py declare` calls for each engine configuration of the A/B (off and on), or a
   generalized pair declaration Codex adds. The q12b `declare-pair` invocation is specific to the PP-v16g on /
   CoW-verify pair and rejects other settings; it is not reused unchanged. Declarations run from sources with the
   engine stopped (the weight hash read must not press on a running engine's shmem; budget 1800 s). Both engines
   share image, model mount, launcher and partition 24,24; they differ only in the overlay files and the switch.
3. Launcher: install the candidate's launcher only if it needs extra checksummed mounts (q10 pattern: copy of the
   found launcher kept, put back on every exit). Otherwise the production launcher stays untouched.
4. Engine starts: `/usr/local/sbin/flash-reconfig.sh` backs up the CURRENT `/etc/default/vllm-flash` and restarts
   those settings when the new ones fail health, so a failing candidate start would transiently bring up whatever the
   current defaults are. Therefore, before the first candidate start, the queue saves the production defaults to a
   private copy (hash recorded) and writes a declared test-only configuration (candidate overlay, switch off) as the
   current defaults, so any automatic rollback lands on a test engine, never production. Alternatively a no-rollback
   test start (`systemctl restart` + the queue's own health wait and stop on failure). Every start is followed by
   `installed`, `model_is`, image and `ident` checks; every measurement step proves `ident` unchanged before and after.
5. On every exit: engine log saved, engine STOPPED, GPUs proven free, launcher and `/etc/default/vllm-flash`
   restored to the saved production values (byte compare against the private copy), `Q<N> DONE … (engine left
   STOPPED)`. The gate queue never starts the production configuration.

## Gates (hard unless marked target) and the tools that already exist

| Gate | Tool / evidence | Verdict rule |
|---|---|---|
| Startup allocation and ledger declaration | engine log QSA lines + `memory_ledger.py` after capture, plus simultaneous per-worker peaks under load | the candidate declares its incremental bytes plus a reserve; if the window is funded from measured driver-free memory there is no released-bytes requirement, otherwise the released source is named and its delta measured; the measured per-rank delta matches the declaration and simultaneous worker peaks stay inside the reserve; no `alloc_retries` increase vs the q15 baseline at the same load |
| Exactness vs the UVA path | Codex's exactness instrumentation (to deliver): six-layer byte equality of the resident pages, same-bucket QSA output/state equality, copy bytes per chunk | bitwise; one sidecar per rank; FAIL is immutable |
| Numerical equivalence, in-process A/B | `retain_ab.py --inproc URL MODEL SWITCHFILE OUT.json` (cases A 60k fresh, B 150k fresh, C append; on vs off-a, noise off-a vs off-b) | tool's own PASS/FAIL (first divergence per case) |
| Teacher-forced batch invariance with the switch | `teacher_forced.py` with `SWITCH=<file in /work>` and `SWITCH_CMD_ON/OFF` (docker exec); alone/reordered/concurrent/switch-off | rule v3, frozen; baseline = `tmp/q15-sample/tf-baseline.json` |
| Warm/cold prefix | `prefix_warm_cold.py --baseline tmp/q15-sample/warmcold.json` | rule v3 |
| CoW invariants | `QSA_COW_VERIFY=1` run + `cow_invariants.py VERIFY.log` | all 4 ranks, every counter present, violations 0 |
| Leak (partial tail / CoW) | `leak_partial_tail.py` (MODE=inside, UNIQUE_PREFIX=1 for pool pressure, CANCEL_FRAC for aborts) | `leakgrade.py` verdicts; any LEAK or INCONCLUSIVE blocks |
| Leak (cross-session) | `leaktest.py SESSIONS ROUNDS` ("bad 0") | 0 leaks |
| Lifetime / pressure | long run filling the KV pool (preemption, request cancel, UID reuse) with the ledger and the lease/ownership counters sampled before/after | the fixed startup window storage stays allocated by design, so reserved VRAM is not expected to return to the pre-window baseline; what must drain to zero after the run are leases, valid-page ownership records and pending DMA (Codex's lease states); allocated/reserved stay inside the declared budget; 0 OOM |
| Canary | `flash_canary.sh` via the hub relay (`canary_hub_runner.sh`, `canary_grade.py`), 10×2 | tool's PASS |
| Memory ledger | `memory_ledger.py` after capture, after warm-up, after drain (q15 shape) | recorded; compared with q15 rows |
| Performance (target, not gate) | fresh 200k prompt TTFT and long warm-prefix append, switch on vs a matched 200k switch-off control on the same engine (the q15 62.5k probe time is a different workload and is not the denominator); copy bytes and DMA time per chunk | report; 80/80 is the later overlap target, serial-only may be slower |

## What Codex delivers for the queue to be runnable

- Overlay generator output (pinned sources → files + SHA256SUMS) and the exact list of generated files.
- Switch name and mechanism (env + optional switch file under `/work`), default off; `installed` K=V list; the
  identity declaration calls (or generalized pair) for the off and on configurations.
- The exactness rule stays separate from model-level noise rules: serial per-layer resident-vs-UVA equality is
  bitwise; teacher-forced and warm/cold use rule v3 for the model-level comparison.
- Exactness instrumentation and its grader (sidecars, immutable primary FAIL, as in `s5_failure.py`).
- The declared memory source and reserve for the ledger comparison.
- The unarmed queue (`harness/runs/q16_e2_serial.draft.sh`), which the box-side session reviews against this
  lifecycle, bundles, deploys and runs as the next GPU job after any slice job already queued.

## Time budget (lean by day)

Identity declare ≈ 15–30 min (engine stopped), two engine starts ≈ 10 min, A/B + teacher-forced ≈ 25 min, warm/cold
≈ 10 min, CoW/leak ≈ 20 min, lifetime ≈ 20 min, canary ≈ 10 min: about 2 h. Split into a correctness slot
(identity, exactness, A/B, teacher-forced, warm/cold, ledgers) and a hygiene slot (CoW, leaks, lifetime, canary) if the
day needs the GPU in between.
