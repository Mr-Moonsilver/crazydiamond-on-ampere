#!/usr/bin/env bash
# attn_meta_gates_b.sh: derived by tools/attn_meta_b_derive.py from tools/attn_meta_gates.sh (chain a ran it; never edited): the
# c1b-base family, the parent c1b b2's READY build, focused plus the c1b and _h suites. Everything below is that chain.
# attn_meta_gates.sh: the q18a-attn-meta packaging gate chain (CPU only, hub; no box/VM/GPU action, no push). c1's chain
# (tools/alpha_next_c1_gates_c.sh, which c1's chain b ran; never edited) over the attn-meta tool family, with these changes:
#   * the parent is c1's delivered build (tools/attn_meta_parent.json; the placeholder refuses: rc 3, 'parent not pinned');
#   * c1's composition and rebind-default-off steps are replaced by attn-meta's two proofs (lead 3 Nov 13:3x, option A,
#     condition 2): base-proof (tools/attn_meta_base_proof.py: c1's tree-identity modules at c1's delivery fbb9123e in a detached
#     shared clone; receipt rc and test count, 22 expected) and default-off (tools/attn_meta_default_off.py: this tree projects
#     byte-identically onto fbb9123e and the switch-unset refusals pass). Both receipts are written once under EVID;
#   * focused and the composition test run through harness/attn_meta_robust_swap.py with --base-receipt (the superseded runner:
#     exactly the four declared c1 ids left out, the receipt and default-off PASS in the same run, else INC rc 2 / FAIL rc 1);
#   * the engine self-run is tools/attn_meta_engine_selfrun_b.py (c1's seven parts unchanged plus attn/), compared against c1's
#     frozen engine record; the negative is its two-phase negative over c1's regenerated overlay (tools/attn_meta_negative_pins.json);
#   * two more container steps after the negative: tools/attn_meta_selfrun5.py tests-c and probe-c (unchanged; the suites that
#     need the model config and the overlay qsa, here the bundle's own native30_overlay/qsa.py);
#   * wait_clear (c1's rule, the lead's resume-2 names): before every container step it waits until docker ps shows no container
#     whose name contains alpha-next-c1, append-yield or returning-append (a failing docker ps counts as not clear) AND the
#     clearance file CLEAR (argument 7) exists: the lead's hold on all hub container steps (3 Nov 13:0x) is lifted only on the
#     lead's word, and the file records it. waiting-before / clear-before lines (no rc field) go to the steps file.
# Everything else is c1's: the flaky rule (tools/alpha_c1_flaky.py, unchanged: one unchanged rerun of a step whose every parsed
# failure is the lead's list), the prepare-review commit, build, bundle self-run, queue pins write + check, tamper, clock scan,
# compile proof, precheck conf, precheck, staging check, and the stop: 'stopped-before-delivery: VM run pending'.
# One heavy job at a time (the hub is shared), nothing in parallel. No absolute-time window.
# usage: attn_meta_gates.sh WT SPEC OUT EVID TAG NOTE CLEAR
# Stops at the first failing step; every rc is logged in $EVID/gates-$TAG.steps.
set -u
unset ATTN_META_PARENT ATTN_META_PARENT_B ALPHA_NEXT_C1_PARENT FLASH_ATTN_META_GDN_BATCH   # development pins are for tests only; the switch is unset
WT=$1 SPEC=$2 OUT=$3 EVID=$4 TAG=$5 NOTE=$6 CLEAR=$7
STEPS=$EVID/gates-$TAG.steps
mkdir -p "$EVID"
[[ ! -e $STEPS && ! -e $OUT && -f $SPEC && -n $CLEAR ]] || exit 3
step() { printf '%s %s rc=%s\n' "$(date -u +%H:%M:%SZ)" "$1" "$2" >>"$STEPS"; [[ $2 == 0 ]] || exit "$2"; }
note_rc() { printf '%s %s rc=%s\n' "$(date -u +%H:%M:%SZ)" "$1" "$2" >>"$STEPS"; }
# flaky KIND TARGET: 0 when every failure of the attempt is a declared flaky test; its verdict line goes to the steps file.
flaky() { local v; v=$(python3 -B "$WT/tools/alpha_c1_flaky.py" "$1" "$2" 2>&1); local rc=$?; printf 'flaky %s %s\n' "$1" "$v" >>"$STEPS"; return $rc; }
# >>> wait_clear (tools/attn_meta_gates.sh; harness/test_attn_meta_gates.py runs this block)
clear_ps() { sudo -n docker ps --format '{{.Names}}'; }
wait_clear() {
  local t0=$SECONDS names last=
  while :; do
    if names=$(clear_ps 2>&1); then
      names=$(grep -E 'alpha-next-c1|append-yield|returning-append' <<<"$names" | tr '\n' ' ')
    else names=docker-ps-failed; fi
    [[ -e $CLEAR ]] || names="${names}hold:no-clearance-file"
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
HEAD=$(git rev-parse HEAD)
[[ -z $(git status --porcelain -- harness tools patches) ]] || { echo dirty >>"$STEPS"; exit 3; }
printf 'head %s spec %s\n' "$HEAD" "$(sha256sum "$SPEC" | cut -c1-12)" >"$STEPS"
printf 'priority nice %s; loadavg %s\n' "$(nice)" "$(cut -d' ' -f1-3 /proc/loadavg)" >>"$STEPS"

python3 -B tools/attn_meta_parent_pin_b.py --pinned >"$EVID/parent-pin-$TAG.log" 2>&1
RC=$?; [[ $RC == 0 ]] || echo 'parent not pinned' >>"$STEPS"
step parent-pin "$RC"
BASE_RECEIPT=$EVID/base-proof-$TAG.json
python3 -B tools/attn_meta_base_proof.py --out "$BASE_RECEIPT" >"$EVID/base-proof-$TAG.log" 2>&1
step base-proof $?
python3 -B tools/attn_meta_default_off.py --out "$EVID/default-off-$TAG.json" >"$EVID/default-off-$TAG.log" 2>&1
step default-off $?
python3 -B tools/attn_meta_review_pins.py --check >"$EVID/review-pins-$TAG.log" 2>&1
step review-pins $?

cd "$WT/harness"
strict() { { python3 -u selftest.py 2>&1; echo "rc=$?"; } >"$1"; tail -n 1 "$1" | sed 's/^rc=//'; }
STRICT_LOG=$EVID/strict-$TAG.log
RC=$(strict "$STRICT_LOG")
if [[ $RC != 0 ]] && flaky strict "$STRICT_LOG"; then
  note_rc strict-1 "$RC"; STRICT_LOG=$EVID/strict-$TAG-2.log; RC=$(strict "$STRICT_LOG"); step strict-2 "$RC"
else step strict "$RC"; fi
focused() { { echo "# head $HEAD focused"; python3 -B attn_meta_robust_swap.py --base-receipt "$BASE_RECEIPT" \
    test_attn_meta_box_b test_native43_engine_death_h test_alpha_next_stop_race_h test_e3_c1_open_rows test_alpha_next_c1b_queue test_alpha_next_c1_send_gate test_alpha_next_c1_defer_control_b \
    test_attn_meta_queue test_attn_meta_box test_attn_meta_control test_attn_meta_c1_g1 test_attn_meta_default_off \
    test_attn_meta_robust_swap test_attn_meta_vm_input_b test_attn_meta_vm_input_c \
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
    test_e3_probe_session test_e3_native_startup_binding test_e3_wire_v1 test_e3_report_validation_cost 2>&1; echo "rc=$?"; } >"$1"
  tail -n 1 "$1" | sed 's/^rc=//'; }
FOCUSED_LOG=$EVID/focused-$TAG.log
RC=$(focused "$FOCUSED_LOG")
if [[ $RC != 0 ]] && flaky unittest "$FOCUSED_LOG"; then
  note_rc focused-1 "$RC"; FOCUSED_LOG=$EVID/focused-$TAG-2.log; RC=$(focused "$FOCUSED_LOG"); step focused-2 "$RC"
else step focused "$RC"; fi
# c1's composition suites (both read git; c1's pack tests use a development parent read from the tree), through the superseded
# runner: test_alpha_next_c1_pack holds three of the four declared ids.
{ python3 -B attn_meta_robust_swap.py --base-receipt "$BASE_RECEIPT" test_alpha_next_composition test_alpha_next_c1_pack test_alpha_next_c1b_pack 2>&1
  echo "rc=$?"; } >"$EVID/composition-test-$TAG.log"
step composition-test "$(tail -n 1 "$EVID/composition-test-$TAG.log" | sed 's/^rc=//')"

cd "$WT"
[[ $(git rev-parse HEAD) == "$HEAD" ]] || { echo "head moved" >>"$STEPS"; exit 4; }
python3 -B tools/build_attn_meta_b.py --spec "$SPEC" --prepare-review --selftest-log "$STRICT_LOG" \
    --focused-log "$FOCUSED_LOG" >"$EVID/prepare-review-$TAG.log" 2>&1
step prepare-review $?
git add patches/s5_engine/native_candidate_review.json && git commit -q -m "q18a-attn-meta build ($TAG): review prepared over $HEAD

strict $(sha256sum "$STRICT_LOG" | cut -c1-12), focused $(sha256sum "$FOCUSED_LOG" | cut -c1-12).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
step review-commit $?
printf 'review-commit %s\n' "$(git rev-parse HEAD)" >>"$STEPS"

python3 -B tools/build_attn_meta_b.py --spec "$SPEC" --out "$OUT" --selftest-log "$STRICT_LOG" \
    --focused-log "$FOCUSED_LOG" >"$EVID/build-$TAG.log" 2>&1
step build $?
PIN=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['manifest'])" "$OUT/build.json")
P12=${PIN:0:12}
HELD=$OUT/q18a-attn-meta-bundle-$P12
RUNNER=$OUT/q18a-attn-meta-owner-run-$P12.py

wait_clear bundle-selfrun
python3 -B tools/native43_bundle_selfrun.py --spec "$SPEC" --bundle "$HELD" --out "$OUT/selfrun" \
    --name "attn-meta-hub-cpu-$P12" --runner "$RUNNER" >"$OUT/selfrun-stdout.json" 2>&1
step bundle-selfrun $?

# Hub-side queue pins (tools/attn_meta_queue_pins.json): written from this build, then checked against OUT. The lead commits the
# pins file; the chain does not. A re-run over filled pins needs PINS_SUPERSEDE="reason".
python3 -B tools/attn_meta_queue_pins_write.py --out "$OUT" ${PINS_SUPERSEDE:+--supersede "$PINS_SUPERSEDE"} \
    >"$OUT/queue-pins-write.log" 2>&1
step queue-pins-write $?
python3 -B tools/attn_meta_queue_pins_write.py --out "$OUT" --check >"$OUT/queue-pins-check.log" 2>&1
step queue-pins-check $?

# engine DIR LABEL: one engine self-run attempt into $OUT/DIR, container attn-meta-LABEL-cpu-<pin>, stdout in $OUT/DIR-stdout.log.
engine() { python3 -B tools/attn_meta_engine_selfrun_b.py --spec "$SPEC" --tree "$WT" --bundle "$HELD" --out "$OUT/$1" \
    --name "attn-meta-$2-cpu-$P12" >"$OUT/$1-stdout.log" 2>&1; }
ENGINE=$OUT/engine-selfrun
wait_clear engine-selfrun
engine engine-selfrun engine; RC=$?
if [[ $RC != 0 ]] && flaky engine "$ENGINE"; then
  note_rc engine-selfrun-1 "$RC"; ENGINE=$OUT/engine-selfrun-2; wait_clear engine-selfrun-2; engine engine-selfrun-2 engine-2; step engine-selfrun-2 $?
else step engine-selfrun "$RC"; fi

python3 -B tools/attn_meta_engine_selfrun_b.py --compare "$ENGINE" --out "$OUT/engine-comparison.json" \
    >"$OUT/engine-comparison-stdout.log" 2>&1
step engine-comparison $?

if python3 -I -c "import json,sys;sys.exit(0 if str(json.load(open(sys.argv[1]))['status']).startswith('AWAITING-OBSERVATION') else 1)" \
    tools/attn_meta_negative_pins.json; then
  # One observation only (the sweep's rule): observe, then stop for review and the pin writer.
  wait_clear engine-negative-observe
  python3 -B tools/attn_meta_engine_selfrun_b.py --negative --observe --bundle "$HELD" --overlay "$OUT/parent/native30_overlay" \
      --sources "$OUT/parent/e3_builder" --out "$OUT/engine-negative-observe" --name "attn-meta-negative-observe-cpu-$P12" \
      >"$OUT/engine-negative-observe-stdout.log" 2>&1
  ORC=$?
  # INC (rc 4: the wall guard's TIMEOUT or the fail-closed judge) stops the chain INC, never a FAIL and never an observation.
  if [[ $ORC == 4 ]]; then echo 'engine-negative-observe INC: see engine-negative-observe/result.json inc_reasons' >>"$STEPS"; step engine-negative-observe 4; fi
  # --observe exits 1 by design; the step passes when the record's verdict is OBSERVED.
  python3 -I -c "import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))['verdict']=='OBSERVED' else 1)" \
      "$OUT/engine-negative-observe/result.json"
  step engine-negative-observe $?
  echo "pins awaiting review: the observed record $OUT/engine-negative-observe/result.json, classified after review into tools/attn_meta_negative_pins.json" >>"$STEPS"
  step engine-negative-awaiting-pins 5
fi
wait_clear engine-negative
python3 -B tools/attn_meta_engine_selfrun_b.py --negative --bundle "$HELD" --overlay "$OUT/parent/native30_overlay" \
    --sources "$OUT/parent/e3_builder" --out "$OUT/engine-negative" --name "attn-meta-negative-cpu-$P12" \
    >"$OUT/engine-negative-stdout.log" 2>&1
RC=$?
[[ $RC == 4 ]] && echo 'engine-negative INC: see engine-negative/result.json inc_reasons' >>"$STEPS"
step engine-negative "$RC"

# The suites that need the model config and the overlay qsa (tools/attn_meta_selfrun5.py, unchanged; its own wait rule and 600 s cap).
for MODE in tests-c probe-c; do
  wait_clear "attn-$MODE"
  python3 -B tools/attn_meta_selfrun5.py --mode "$MODE" --name "attn-meta-$MODE-cpu-$P12" --out "$OUT/attn-$MODE" \
      --overlay-qsa "$HELD/native30_overlay/qsa.py" >"$OUT/attn-$MODE-stdout.log" 2>&1
  RC=$?
  python3 -I -c "import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))['verdict']=='PASS' else 1)" "$OUT/attn-$MODE/result.json" \
      || { [[ $RC == 0 ]] && RC=1; }
  step "attn-$MODE" "$RC"
done

python3 -B tools/attn_meta_overlay_tamper_b.py --parent "$OUT/parent/native30_overlay" --candidate "$HELD/native30_overlay" \
    --out "$OUT/overlay-tamper.json" >"$OUT/overlay-tamper-stdout.log" 2>&1
step overlay-tamper $?

python3 -B tools/attn_meta_clock_scan_b.py --bundle "$HELD" --runner "$RUNNER" --setcfg-reference "$WT/deploy/setcfg.py" \
    --out "$OUT/clock-scan.json" >"$OUT/clock-scan-stdout.log" 2>&1
step clock-scan $?

python3 -B tools/native35_compile_proof.py --bundle "$HELD" --runner "$RUNNER" --out "$OUT/compile-proof.json" \
    >"$OUT/compile-proof-stdout.log" 2>&1
step compile-proof $?

# The conf stays with its build; the delivery tool's caller copies the delivered one to tools/owner-precheck/.
python3 -B tools/attn_meta_precheck_conf_b.py --build "$OUT/build.json" --selfrun "$OUT/selfrun/result.json" \
    --out "$OUT/native-precheck.conf" >"$OUT/precheck-conf.log" 2>&1
step precheck-conf $?

bash tools/owner-precheck/native-precheck.sh "$OUT/native-precheck.conf" >"$OUT/precheck-$TAG.log" 2>&1
step precheck $?

# The frozen runner's own staging require blocks over the exact records delivered (Step B's check, evaluated on the hub first).
python3 -B tools/attn_meta_staging_check.py --runner "$RUNNER" --selfrun "$OUT/selfrun/result.json" \
    --engine "$ENGINE/result.json" --out "$OUT/staging-check.json" >"$OUT/staging-check.log" 2>&1
step staging-check $?

echo 'stopped-before-delivery: VM run pending' >>"$STEPS"
exit 0
