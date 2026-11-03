#!/usr/bin/env bash
# stage1_step_gates.sh: the q18a-stage1-step packaging gate chain (CPU only, hub; no box/VM/GPU action, no push). Derived by name, not edited anywhere: tools/alpha_next_c1b_gates_b.sh (chain b2 ran it) through
# attn-meta's fixed form (claude/attn-meta-c1b: tools/attn_meta_gates_c.sh, chain c ran it) over the stage1-step tool family. Named differences:
#   * the parent is alpha-next-c1b's READY build (chain b2, manifest 6595d6692a0d; tools/stage1_step_parent.json; a placeholder refuses: rc 3, 'parent not pinned');
#   * no carry rule and no selfrun5 steps (they were attn-meta's own); the engine self-run is tools/stage1_step_engine_selfrun.py (c1's seven parts plus fifo/);
#   * base-proof and default-off as attn-meta's (c1's tree-identity modules at fbb9123e; this tree projects byte-identically onto fbb9123e, switch-unset refusals);
#     focused and the composition test through harness/stage1_step_robust_swap.py with --base-receipt;
#   * strict through <hub-dir>/agents/flash-next/fn-main/tools/hub-slot.sh strict 2 (docs/hub-slot.md: at most two heavy strict steps on the hub);
#     parallel inside the slot: tools/strict_parallel.py (test-speed 7c85af5b, byte for byte) with WORKERS and LANES set at launch (tools/stage1_step_launch.sh: 4 workers, tools/strict_lanes3.alpha-next-c1b.json,
#     audited over c1b; a row whose sha moved falls to the serial lane: slower, not wrong). Its log reads as a serial log (the build's regression_from, the flaky rule). No gates_fast tail (lead, 22:05);
#   * focused stays a real run: focused_subset's FOCUSED_BLOCK reads alpha_next_robust_swap's list, and this chain's runner is the superseded one, so no proof by id exists on this tree and the step is not dropped (lead, 22:05);
#   * wait_clear before every container step: it waits until docker ps shows no container whose name contains alpha-next-c1, append-yield, returning-append, attn-meta, stage1-handoff or test-speed- (a failing docker ps counts as not
#     clear). waiting-before / clear-before lines (no rc field) go to the steps file. No clearance file; our own containers are prefixed stage1-step-;
#   * --dry-run: every step's precondition, no container, no build, nothing written but EVID/dryrun.log and EVID/dryrun.json; tools/stage1_step_launch.sh refuses without it;
#   * no vm-carry step: c1b's carry proves the engine bytes did not move; here they move, so it would refuse by construction. The inherited stop line stays the last line.
#   * the static checks run right after the build and before any container (the lead's standing order, 3 Nov): precheck without its selfrun check, tamper, clock scan, compile proof; the full precheck stays at the end. Before strict, the seams
#     check alone over the tree (tools/stage1_step_seams.py, the precheck's own scan), and in the dry run;
#   * tools/stage1_step_targeted.sh runs this file's review_to_stop block after a test-side stop (the lead's targeted route), by its markers.
# Everything else is c1's: the flaky rule (tools/alpha_c1_flaky.py, unchanged: one unchanged rerun of a step whose every parsed failure is the lead's list), the prepare-review
# commit, build, bundle self-run, queue pins write + check, comparison, the two-phase negative (rc 5 at the observation, then tools/stage1_step_gates_cont.sh),
# precheck conf, precheck, staging check, and the stop: 'stopped-before-delivery: VM run pending'.
# One heavy job at a time (the hub is shared), nothing else in parallel, nice 10. No absolute-time window.
# usage: WORKERS=N LANES=FILE [PINS_SUPERSEDE=reason] stage1_step_gates.sh [--dry-run] WT SPEC OUT EVID TAG NOTE
# Stops at the first failing step; every rc is logged in $EVID/gates-$TAG.steps.
set -u
unset STAGE1_STEP_PARENT ALPHA_NEXT_C1_PARENT ALPHA_NEXT_C1B_PARENT FLASH_S5_LAZY_RECV FLASH_S5_DECODE_MERGE   # development pins are for tests only; the switch is unset
DRY=; if [[ ${1:-} == --dry-run ]]; then DRY=1; shift; fi
WT=$1 SPEC=$2 OUT=$3 EVID=$4 TAG=$5 NOTE=$6
HUB_SLOT=<hub-dir>/agents/flash-next/fn-main/tools/hub-slot.sh
STEPS=$EVID/gates-$TAG.steps
mkdir -p "$EVID"
[[ ! -e $STEPS && ! -e $OUT && -f $SPEC && -n ${WORKERS:-} && -f ${LANES:-} ]] || exit 3
[[ $(nice) -ge 10 ]] || renice -n 10 -p $$ >/dev/null
step() { printf '%s %s rc=%s\n' "$(date -u +%H:%M:%SZ)" "$1" "$2" >>"$STEPS"; [[ $2 == 0 ]] || exit "$2"; }
note_rc() { printf '%s %s rc=%s\n' "$(date -u +%H:%M:%SZ)" "$1" "$2" >>"$STEPS"; }
# flaky KIND TARGET: 0 when every failure of the attempt is a declared flaky test; its verdict line goes to the steps file.
flaky() { local v; v=$(python3 -B "$WT/tools/alpha_c1_flaky.py" "$1" "$2" 2>&1); local rc=$?; printf 'flaky %s %s\n' "$1" "$v" >>"$STEPS"; return $rc; }
# >>> wait_clear (harness/test_stage1_step_gates.py runs this block)
clear_ps() { sudo -n docker ps --format '{{.Names}}'; }
wait_clear() {
  local t0=$SECONDS names last=
  while :; do
    if names=$(clear_ps 2>&1); then
      names=$(grep -E 'alpha-next-c1|append-yield|returning-append|attn-meta|stage1-handoff|test-speed-' <<<"$names" | tr '\n' ' ')
    else names=docker-ps-failed; fi
    [[ -z $names ]] && break
    [[ $last == "$names" ]] || printf '%s waiting-before %s: %s\n' "$(date -u +%H:%M:%SZ)" "$1" "$names" >>"$STEPS"
    last=$names
    sleep "${CLEAR_POLL_S:-30}"
  done
  printf '%s clear-before %s waited %ss loadavg %s\n' "$(date -u +%H:%M:%SZ)" "$1" "$((SECONDS - t0))" \
      "$(cut -d' ' -f1-3 /proc/loadavg)" >>"$STEPS"
}
# <<< wait_clear
cd "$WT" || exit 3
ENGINE_SUITES=$(python3 -I -c "import json,sys;print(' '.join(json.load(open(sys.argv[1]))['engine_suites']))" tools/stage1_step_engine_suites.json) || exit 3
FOCUSED_MODULES="test_stage1_step_box test_native43_engine_death_h test_alpha_next_stop_race_h test_e3_c1_open_rows test_alpha_next_c1b_queue test_alpha_next_c1_send_gate test_alpha_next_c1_defer_control_b \
    $ENGINE_SUITES test_stage1_step_queue test_stage1_step_queue_names test_stage1_step_default_off test_stage1_step_robust_swap \
    test_alpha_next_c1_queue test_e3_stage_overlap test_alpha_next_c1_composed_fields test_e3_stage_overlap_control \
    test_c1_rebind_witness test_c1_rebind_on test_alpha_next_c1_defer_control test_alpha_next_c1_rebind_default_off test_alpha_next_c1_negative_guard \
    test_alpha_next_probe_launch test_alpha_c1_queue test_alpha_c1_box test_alpha_c1_report \
    test_alpha_next_queue test_alpha_next_receipts test_alpha_next_receipts_mutants test_alpha_next_stage1_slow_rank \
    test_alpha_next_path_qualification test_alpha_next_stopfix_matrix test_alpha_next_c0_ring test_alpha_next_peak_receipt \
    test_stage1_handler_anatomy test_e3_native_anatomy test_e3_native_lean_replies test_e3_w1_decode_driver_ordered \
    test_native43_le_bounded_enqueue_stepped test_alpha_c0_queue test_e3_stage_local test_e3_stage_local_control \
    test_e3_stage_local_qsa test_alpha_c0_report test_step_b_queue test_e3_native_layer_groups test_step_a_queue \
    test_e3_native_qsa_fused test_step_a_report test_step_a_torn_lines test_alpha_perf_queue test_native41_progress_queue \
    test_e3_native_reply_hol test_native41_box_queue test_native41_autoarm test_native41_unarmed_box test_native41_tp_skew \
    test_alpha_m3_receipts test_native39_trunk test_native39_trunk_queue test_native37_robust test_native38_plain_bind \
    test_e3_admission_native test_e3_staged_chunk_plan test_e3_native_overlap test_e3_native_storage_replay \
    test_e3_r4_worker_bridge test_e3_admission_metadata test_native35_p_c_continuation test_e3_native28_host_guard \
    test_identity_declare test_e3_native_measurement test_native43_dense_boundary test_native43_boundary_queue \
    test_e3_qsa_driver test_e3_lane_private test_e3_sparse_history_dma test_e3_worker_operations test_e3_tp_finalize_skew \
    test_e3_native_measurement_clients test_native35_performance test_native43_engine_death test_native43_le_fatal_log \
    test_native43_le_bounded_enqueue test_native43_le_fix_queue_deltas test_native40_unarmed_hold test_native40_box_queue \
    test_e3_package test_e3_native_schedule_profile test_e3_native_submission_credit test_e3_native_selection_asymmetry \
    test_s5_reply_transport test_e3_executor_threads test_e3_stock_rpc_credits test_e3_async_handoffs test_e3_probe_control \
    test_e3_probe_session test_e3_native_startup_binding test_e3_wire_v1 test_e3_report_validation_cost"
# >>> dry_run (harness/test_stage1_step_gates.py runs this block)
if [[ -n $DRY ]]; then
  LOG=$EVID/dryrun.log
  [[ ! -e $LOG && ! -e $EVID/dryrun.json ]] || { echo 'dry-run refused: a dry run of this tag exists (a new dry run gets a new tag)' >&2; exit 3; }
  WORST=0
  dry() { local name=$1; shift; local out rc; out=$("$@" 2>&1); rc=$?
    printf 'dry %s rc=%s\n' "$name" "$rc" >>"$LOG"; [[ $rc == 0 ]] || { printf '%s\n' "$out" | tail -n 15 | cut -c1-400 >>"$LOG"; WORST=1; }; }
  printf 'dry-run head %s spec %s nice %s loadavg %s\n' "$(git rev-parse HEAD)" "$(sha256sum "$SPEC" | cut -c1-12)" "$(nice)" "$(cut -d' ' -f1-3 /proc/loadavg)" >"$LOG"
  dry tree-clean bash -c '[[ -z $(git status --porcelain -- harness tools patches) ]]'
  dry spec-is-for-this-tree python3 -B tools/stage1_step_dryrun.py spec "$SPEC"
  dry parent-pin python3 -B tools/stage1_step_parent_pin.py --pinned
  dry parent-records python3 -B tools/stage1_step_dryrun.py parent
  dry default-off-rows python3 -B tools/stage1_step_default_off_rows.py --check
  dry default-off bash -c 'python3 -B tools/stage1_step_default_off.py >/dev/null'
  dry review-pins python3 -B tools/stage1_step_review_pins.py --check
  dry queue-texts python3 -B tools/stage1_step_queue.py --check
  dry probe-derived python3 -B tools/stage1_step_tools_derive.py --check
  dry seams python3 -B tools/stage1_step_seams.py
  mkdir -p "$(dirname "$WT")/tmp"
  dry strict-plan bash -c 'cd harness && STAGE1_STEP_BASE_RECEIPT=/nonexistent-plan-only python3 -B "$0/tools/strict_parallel.py" --lanes "$1" --check-plan >/dev/null' "$WT" "$LANES"
  dry syntax bash -c 'bash -n tools/stage1_step_gates.sh && bash -n tools/stage1_step_gates_cont.sh && bash -n tools/stage1_step_targeted.sh && bash -n tools/stage1_step_launch.sh && bash -n tools/stage1_step_launch_cont.sh && bash -n harness/runs/q18a_stage1_step.draft.sh && bash -n harness/runs/q18a_stage1_step_declare.sh'
  dry chain-tools python3 -B tools/stage1_step_dryrun.py tools tools/stage1_step_gates.sh
  dry focused-modules python3 -B tools/stage1_step_dryrun.py modules $FOCUSED_MODULES
  dry superseded-map python3 -B tools/stage1_step_dryrun.py swap
  dry pins-stubs python3 -B tools/stage1_step_dryrun.py pins "${PINS_SUPERSEDE:-}"
  dry hub-slot bash -c 'D=$(mktemp -d) && HUB_SLOT_DIR=$D "$0" strict 2 "$D/slot.log" -- true; rc=$?; rm -rf "$D"; exit $rc' "$HUB_SLOT"
  dry docker-ps bash -c "sudo -n docker ps --format '{{.Names}}' >/dev/null"
  dry docker-image python3 -B tools/stage1_step_dryrun.py image
  dry disk python3 -B tools/stage1_step_dryrun.py disk "$OUT"
  printf 'containers matching the wait now: %s\n' "$(sudo -n docker ps --format '{{.Names}}' 2>/dev/null | grep -E 'alpha-next-c1|append-yield|returning-append|attn-meta|stage1-handoff|test-speed-' | tr '\n' ' ')" >>"$LOG"
  printf 'dry-run rc=%s %s\n' "$WORST" "$(date -u +%H:%M:%SZ)" >>"$LOG"
  bash tools/dryrun-record.sh "$WT" "$WORST" "$LOG" "$EVID/dryrun.json" >>"$EVID/dryrun-record.out" 2>&1 || WORST=2
  cat "$LOG"
  exit "$WORST"
fi
# <<< dry_run
HEAD=$(git rev-parse HEAD)
[[ -z $(git status --porcelain -- harness tools patches) ]] || { echo dirty >>"$STEPS"; exit 3; }
printf 'head %s spec %s note %s\n' "$HEAD" "$(sha256sum "$SPEC" | cut -c1-12)" "$NOTE" >"$STEPS"
printf 'priority nice %s; loadavg %s; workers %s; lanes %s\n' "$(nice)" "$(cut -d' ' -f1-3 /proc/loadavg)" "$WORKERS" "$(sha256sum "$LANES" | cut -c1-12)" >>"$STEPS"

python3 -B tools/stage1_step_parent_pin.py --pinned >"$EVID/parent-pin-$TAG.log" 2>&1
RC=$?; [[ $RC == 0 ]] || echo 'parent not pinned' >>"$STEPS"
step parent-pin "$RC"
BASE_RECEIPT=$EVID/base-proof-$TAG.json
python3 -B tools/stage1_step_base_proof.py --out "$BASE_RECEIPT" >"$EVID/base-proof-$TAG.log" 2>&1
step base-proof $?
python3 -B tools/stage1_step_default_off.py --out "$EVID/default-off-$TAG.json" >"$EVID/default-off-$TAG.log" 2>&1
step default-off $?
python3 -B tools/stage1_step_review_pins.py --check >"$EVID/review-pins-$TAG.log" 2>&1
step review-pins $?
python3 -B tools/stage1_step_seams.py >"$EVID/seams-$TAG.log" 2>&1
step seams-tree $?

cd "$WT/harness"
# Declared (test-speed): the suites mkdtemp in <repo parent>/tmp and one of them makes it; started in parallel it must be there (lanes required_dirs; strict refuses without it).
mkdir -p "$(dirname "$WT")/tmp"
STAGE1_STEP_BASE_RECEIPT=$BASE_RECEIPT python3 -u "$WT/tools/strict_parallel.py" --workers "$WORKERS" --runs-root "$EVID/strict-runs" --lanes "$LANES" --check-plan >"$EVID/strict-plan-$TAG.log" 2>&1
step strict-plan $?
# >>> strict (tools/stage1_step_targeted.sh runs this block)
strict() { { STAGE1_STEP_BASE_RECEIPT=$BASE_RECEIPT "$HUB_SLOT" strict 2 "$EVID/strict-slot.log" -- python3 -u "$WT/tools/strict_parallel.py" --workers "$WORKERS" --runs-root "$EVID/strict-runs" --lanes "$LANES" --record "$1.record.json" 2>&1; echo "rc=$?"; } >"$1"; tail -n 1 "$1" | sed 's/^rc=//'; }
# <<< strict
STRICT_LOG=$EVID/strict-$TAG.log
RC=$(strict "$STRICT_LOG")
if [[ $RC != 0 ]] && flaky strict "$STRICT_LOG"; then
  note_rc strict-1 "$RC"; STRICT_LOG=$EVID/strict-$TAG-2.log; RC=$(strict "$STRICT_LOG"); step strict-2 "$RC"
else step strict "$RC"; fi
# shellcheck disable=SC2086
focused() { { echo "# head $HEAD focused"; python3 -B stage1_step_robust_swap.py --base-receipt "$BASE_RECEIPT" $FOCUSED_MODULES 2>&1; echo "rc=$?"; } >"$1"
  tail -n 1 "$1" | sed 's/^rc=//'; }
FOCUSED_LOG=$EVID/focused-$TAG.log
RC=$(focused "$FOCUSED_LOG")
if [[ $RC != 0 ]] && flaky unittest "$FOCUSED_LOG"; then
  note_rc focused-1 "$RC"; FOCUSED_LOG=$EVID/focused-$TAG-2.log; RC=$(focused "$FOCUSED_LOG"); step focused-2 "$RC"
else step focused "$RC"; fi
# c1's composition suites (both read git; c1's pack tests use a development parent read from the tree), through the superseded
# runner: test_alpha_next_c1_pack holds three of the four declared ids.
{ python3 -B stage1_step_robust_swap.py --base-receipt "$BASE_RECEIPT" test_alpha_next_composition test_alpha_next_c1_pack test_alpha_next_c1b_pack 2>&1
  echo "rc=$?"; } >"$EVID/composition-test-$TAG.log"
step composition-test "$(tail -n 1 "$EVID/composition-test-$TAG.log" | sed 's/^rc=//')"

cd "$WT"
# >>> review_to_stop (tools/stage1_step_targeted.sh runs this block after a test-side stop; it needs HEAD, STRICT_LOG, FOCUSED_LOG, TAG and the functions above)
[[ $(git rev-parse HEAD) == "$HEAD" ]] || { echo "head moved" >>"$STEPS"; exit 4; }
python3 -B tools/build_stage1_step.py --spec "$SPEC" --prepare-review --selftest-log "$STRICT_LOG" \
    --focused-log "$FOCUSED_LOG" >"$EVID/prepare-review-$TAG.log" 2>&1
step prepare-review $?
git add patches/s5_engine/native_candidate_review.json && git commit -q -m "q18a-stage1-step build ($TAG): review prepared over $HEAD

strict $(sha256sum "$STRICT_LOG" | cut -c1-12), focused $(sha256sum "$FOCUSED_LOG" | cut -c1-12).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
step review-commit $?
printf 'review-commit %s\n' "$(git rev-parse HEAD)" >>"$STEPS"

python3 -B tools/build_stage1_step.py --spec "$SPEC" --out "$OUT" --selftest-log "$STRICT_LOG" \
    --focused-log "$FOCUSED_LOG" >"$EVID/build-$TAG.log" 2>&1
step build $?
PIN=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['manifest'])" "$OUT/build.json")
P12=${PIN:0:12}
HELD=$OUT/q18a-stage1-step-bundle-$P12
RUNNER=$OUT/q18a-stage1-step-owner-run-$P12.py
SELFRUN=$OUT/selfrun

# >>> static_checks (harness/test_stage1_step_gates.py holds these before the first container step): the lead's standing order (3 Nov; c1d lost an hour when its precheck failed after a 53-minute engine self-run).
# The whole precheck but its selfrun check (no container has run yet; the full precheck runs again at the end, over the self-run), tamper, clock scan, compile proof. A failure stops the chain here with that rc.
python3 -B tools/stage1_step_precheck_conf.py --build "$OUT/build.json" --out "$OUT/native-precheck-static.conf" >"$OUT/precheck-static-conf.log" 2>&1
step precheck-static-conf $?
ONLY='manifest counts prevdiff sources runner pins review syntax queue credscan keys leaves harness seams' bash tools/owner-precheck/native-precheck.sh "$OUT/native-precheck-static.conf" >"$OUT/precheck-static-$TAG.log" 2>&1
step precheck-static $?

python3 -B tools/stage1_step_overlay_tamper.py --parent "$OUT/parent/native30_overlay" --candidate "$HELD/native30_overlay" \
    --out "$OUT/overlay-tamper.json" >"$OUT/overlay-tamper-stdout.log" 2>&1
step overlay-tamper $?

python3 -B tools/stage1_step_clock_scan.py --bundle "$HELD" --runner "$RUNNER" --setcfg-reference "$WT/deploy/setcfg.py" \
    --out "$OUT/clock-scan.json" >"$OUT/clock-scan-stdout.log" 2>&1
step clock-scan $?

python3 -B tools/native35_compile_proof.py --bundle "$HELD" --runner "$RUNNER" --out "$OUT/compile-proof.json" \
    >"$OUT/compile-proof-stdout.log" 2>&1
step compile-proof $?

# <<< static_checks
wait_clear bundle-selfrun
python3 -B tools/native43_bundle_selfrun.py --spec "$SPEC" --bundle "$HELD" --out "$OUT/selfrun" \
    --name "stage1-step-hub-cpu-$P12" --runner "$RUNNER" >"$OUT/selfrun-stdout.json" 2>&1
step bundle-selfrun $?

# Hub-side queue pins (tools/stage1_step_queue_pins.json): written from this build, then checked against OUT. The lead commits the
# pins file; the chain does not. A re-run over filled pins needs PINS_SUPERSEDE="reason".
python3 -B tools/stage1_step_queue_pins_write.py --out "$OUT" ${PINS_SUPERSEDE:+--supersede "$PINS_SUPERSEDE"} \
    >"$OUT/queue-pins-write.log" 2>&1
step queue-pins-write $?
python3 -B tools/stage1_step_queue_pins_write.py --out "$OUT" --check >"$OUT/queue-pins-check.log" 2>&1
step queue-pins-check $?

# engine DIR LABEL: one engine self-run attempt into $OUT/DIR, container stage1-step-LABEL-cpu-<pin>, stdout in $OUT/DIR-stdout.log.
engine() { python3 -B tools/stage1_step_engine_selfrun.py --spec "$SPEC" --tree "$WT" --bundle "$HELD" --out "$OUT/$1" \
    --name "stage1-step-$2-cpu-$P12" >"$OUT/$1-stdout.log" 2>&1; }
ENGINE=$OUT/engine-selfrun
wait_clear engine-selfrun
engine engine-selfrun engine; RC=$?
if [[ $RC != 0 ]] && flaky engine "$ENGINE"; then
  note_rc engine-selfrun-1 "$RC"; ENGINE=$OUT/engine-selfrun-2; wait_clear engine-selfrun-2; engine engine-selfrun-2 engine-2; step engine-selfrun-2 $?
else step engine-selfrun "$RC"; fi

python3 -B tools/stage1_step_engine_selfrun.py --compare "$ENGINE" --out "$OUT/engine-comparison.json" \
    >"$OUT/engine-comparison-stdout.log" 2>&1
step engine-comparison $?

if python3 -I -c "import json,sys;sys.exit(0 if str(json.load(open(sys.argv[1]))['status']).startswith('AWAITING-OBSERVATION') else 1)" \
    tools/stage1_step_negative_pins.json; then
  # One observation only (the sweep's rule): observe, then stop for review and the pin writer.
  wait_clear engine-negative-observe
  python3 -B tools/stage1_step_engine_selfrun.py --negative --observe --bundle "$HELD" --overlay "$OUT/parent/native30_overlay" \
      --sources "$OUT/parent/e3_builder" --out "$OUT/engine-negative-observe" --name "stage1-step-negative-observe-cpu-$P12" \
      >"$OUT/engine-negative-observe-stdout.log" 2>&1
  ORC=$?
  # INC (rc 4: the wall guard's TIMEOUT or the fail-closed judge) stops the chain INC, never a FAIL and never an observation.
  if [[ $ORC == 4 ]]; then echo 'engine-negative-observe INC: see engine-negative-observe/result.json inc_reasons' >>"$STEPS"; step engine-negative-observe 4; fi
  # --observe exits 1 by design; the step passes when the record's verdict is OBSERVED.
  python3 -I -c "import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))['verdict']=='OBSERVED' else 1)" \
      "$OUT/engine-negative-observe/result.json"
  step engine-negative-observe $?
  echo "pins awaiting review: the observed record $OUT/engine-negative-observe/result.json, classified after review into tools/stage1_step_negative_pins.json" >>"$STEPS"
  step engine-negative-awaiting-pins 5
fi
wait_clear engine-negative
python3 -B tools/stage1_step_engine_selfrun.py --negative --bundle "$HELD" --overlay "$OUT/parent/native30_overlay" \
    --sources "$OUT/parent/e3_builder" --out "$OUT/engine-negative" --name "stage1-step-negative-cpu-$P12" \
    >"$OUT/engine-negative-stdout.log" 2>&1
RC=$?
[[ $RC == 4 ]] && echo 'engine-negative INC: see engine-negative/result.json inc_reasons' >>"$STEPS"
step engine-negative "$RC"

# The conf stays with its build; the delivery tool's caller copies the delivered one to tools/owner-precheck/.
python3 -B tools/stage1_step_precheck_conf.py --build "$OUT/build.json" --selfrun "$SELFRUN/result.json" \
    --out "$OUT/native-precheck.conf" >"$OUT/precheck-conf.log" 2>&1
step precheck-conf $?

bash tools/owner-precheck/native-precheck.sh "$OUT/native-precheck.conf" >"$OUT/precheck-$TAG.log" 2>&1
step precheck $?

# The frozen runner's own staging require blocks over the exact records delivered (Step B's check, evaluated on the hub first).
python3 -B tools/stage1_step_staging_check.py --runner "$RUNNER" --selfrun "$SELFRUN/result.json" \
    --engine "$ENGINE/result.json" --out "$OUT/staging-check.json" >"$OUT/staging-check.log" 2>&1
step staging-check $?

echo 'stopped-before-delivery: VM run pending' >>"$STEPS"
exit 0
# <<< review_to_stop
