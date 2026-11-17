# native43 packaging interface

Base: `1ffc394` / held-b `756f826c221c`. Branch `codex/native43-dense-boundary`, worktree `wt-codex-native43`.

Engine (all in `patches/s5_engine/`):

- `qsa_operation.py`: startup policy parser, dense eligibility, bounded stream-event/CPU receipts, boundary hashes.
- `native_resources.py`: passes the actual startup env into NativeQsaDevice.
- `qsa_driver.py`: diagnostic final-chunk selection and QSA-output evidence hooks. Generated ordinary/model arithmetic unchanged.
- `engine_driver.py`: explicit diagnostic cutoff after all-rank commit, existing certified retained-row cancel and FULL return; no stock/chunk continuation afterwards; status includes `native43_boundary`.

Startup env:

- `FLASH_S5_NATIVE_QSA_POLICY=current|dense`, default `current` (latched once).
- `FLASH_S5_NATIVE_QSA_TRACE=0|1`, default `0`; require `1` for this queue.
- `FLASH_S5_NATIVE_QSA_STOP_AFTER=0|101200`, default `0`; this diagnostic queue uses `101200`, exactly four chunks past `24*4048=97152`. It holds the engine after certified cancellation until restart; it is not a production setting.

Harness additions:

- `native43_boundary_client.py`: one real 197270-token fresh prompt, no cohort; uses native session; waits for exact checkpoint + typed all-rank closure + FULL; closes HTTP request. `--url --model --run-id --epoch --pair-id --policy --workload --out [--timeout-s 300]`.
- `native43_boundary_clock.py`: engine sample inside real request bracket, never boot-id/ns equality; record identities as provenance.
- `native43_scheduler_tick_probe.py`: required pinned-container real AsyncScheduler/Core-hook probe, `--out /out/native43-scheduler-probe.json`; reuses native37 construction leaves and explicitly declared pending-stop/checkpoint leaves. No source files are mounted raw into the generated engine namespace; use normal `make_s5_overlap.package` generation.
- `native43_clock_probe.py`: `--serve` stdin/stdout status probe for a host client outside the engine's pinned CPU container; actual driver status clock and actual proc views. Packaging must ensure the two views differ (alternate boot_id bind mount is a declared fixture if needed), zero offsets observed, and execute client `prove()` against this packet.
- `native43_dense_grade.py`: `--current CLIENT --dense CLIENT --current-log LOG [repeat for stderr] --dense-log LOG [repeat] --out JSON`; hashes are INFORMATIONAL; reports stage-0 first-QSA selection comparisons and bounded 256-element output max-abs/relative differences without a numerical failure gate. Deciding evidence is timing/route coverage and certified closure, with no speed target gate. Final readback chunk is excluded from timing.
- `native43_boundary_gate.py`: shared phase budgets and inherited owner weekend window.
- `native43_fixture_leaves.json`: exact synthetic objects and limitations; none is box evidence.
- `test_native43_boundary_queue.py`: queue derivation/argv/grade and real-clock packet seams.
- `test_native43_dense_boundary.py`: generated-attention transport tests and real four-rank cutoff/cancel/FULL CPU gate; clock negatives. Existing `test_e3_qsa_driver.py` and `test_e3_lane_private.py` add the new diagnostic method to their no-device fixtures. `selftest.py` registers both new suites.

Queue derivation: `tools/native43_queue_drafts.py` writes/checks:

- `runs/q18a_native43_dense_boundary.draft.sh`: pair, current then dense, separate fresh process/key/request dirs; output argument is an absolute PREFIX; children are PREFIX-current and PREFIX-dense, requests PREFIX-current-requests and PREFIX-dense-requests. `HARNESS_DIR` is the frozen bundle. Pair grade PREFIX-pair-grade.json. No production start.
- `runs/q18a_native43_{current,dense}.draft.sh`: held-b prefix through startup/bind/identity/ledger/length guard; replace all later workload phases with the single boundary client; keep held-b crash/stop/restore bodies.
- `runs/q18a_native43_{current,dense}_{settings,declare}.sh`: literal policy and trace/stop declarations; model settings, loopback, observer/offers/localize off unchanged. Packaging must prepare BOTH request directories with those exact settings and fresh root-only keys.

Receipts: INFO `flash-s5 native43_qsa JSON`, per layer, emitted at operation release. Max 512 timing markers/operation, drops explicit; no device-wide fences. Records include history, actual route, union-selected count or explicit NOT_OBSERVED when union skipped, descriptors/waves/bytes, CPU plan/submission costs, stream-event union/copy/attention spans, cumulative operation counters and operation open/release stamps. Existing driver status `control_path` and `qsa_control` retain command/profile counters. Final boundary readbacks (selection IDs and QSA output) are bounded to 64 MiB/tensor, hashes only, and explicitly excluded from measured chunks. No policy threshold is changed by the diagnostic.

Packaging owns builder, review closure, freeze, exact-container self-runs, precheck and runner. Source-final SHA and fresh test results follow here when stable. No box/device action by Codex.

CPU results so far: new suites 24/24 (0 skips), existing worker operations 43/43, sparse transport 34/34, QSA driver 10/10, four-rank bridge 42/42. Full strict is running at `tmp/native43-strict-a.log`; packaging should run strict again on the merged final freeze. Expected actual box time is about 20–30 minutes for two startup/identity/boundary/cleanup arms; the shared worst-case phase caps reserve substantially more. No speed claim.

Harness follow-up: pair grading independently revalidates raw checkpoint, exact four typed CancelAck/report rows, source/PP closure, FULL status and the real request bracket; a cached client PASS does not suffice. Timed pre-threshold (97152) and first three post-threshold chunks (98176/99200/100224) must be present for every rank; final 101200 evidence chunk remains excluded.

Development strict run finished 324/325, 208 suites, zero skips: sole failure was test_e3_lane_private.py (three errors from its separate FakeOperation lacking boundary_evidence). That fixture is now updated; its focused rerun passed 29/29, zero skips. Final new suites passed 24/24, zero skips (`tmp/native43-focused-final.log`). No engine changes after bd51de8. Packaging owns the fresh full strict run on merged/frozen bytes.

Owner-review corrections supersede the earlier hash-equality specification: all four findings CONFIRMED in reviews/codex-20261101-native43-owner-findings.md. Immediate cutoff bypass; per-bucket copy spans (including inter-wave host gaps); protected trace emit; all cross-arm hashes informational. Full route/timing record coverage requires 2400 unique layer records per arm; 3000 cap leaves 600 (25%) headroom. Sample output differences cover the first 256 flattened values, not a full-tensor maximum. Final counts follow after focused tests.

Owner-review correction handoff: source includes native43_scheduler_tick_probe.py, which packaging must execute in its exact frozen pinned-image self-run. Standalone generated-module probe PASS at tmp/native43-owner-cpu-e/probe.json; exact old-driver negative FAIL at tmp/native43-owner-cpu-f/probe.json, both no GPU/no network. Summary tmp/native43-owner-cpu-proof.json. Focused correction suites: 29 new + 43 worker operations + 34 sparse DMA + 10 QSA driver + 42 four-rank bridge + 29 private binding = 187 tests, zero skips. Old-method host negative also reproduces extra 4→8 admission. Queue/window/caps unchanged.
