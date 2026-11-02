#!/usr/bin/env bash
# alpha-next-c1 packaging gate chain (CPU only, hub; no box/VM/GPU action, no push): tools/alpha_next_gates.sh's chain over the
# alpha-next-c1 tool family, with C1's one change in form (tools/alpha_c1_gates.sh, the lead, 2 Nov): strict, focused and the engine
# self-run each get one unchanged rerun of the whole step when tools/alpha_c1_flaky.py (unchanged; its list is the lead's: the
# test_e3_probe_session driver-composition test, the test_native43_le_bounded_enqueue owned-route test, test_native38_plain_bind
# .FatalReplies, test_e3_native_reply_hol.DirectReplies; alpha-next's gates declare no flaky ids of their own: their load-flaky
# methods are swapped by harness/alpha_next_robust_swap.py) finds every parsed failure of the attempt in that list. Both logs are kept
# (NAME and NAME-2), both rc's are in the steps file, later steps read the passing attempt.
#   parent pin (tools/alpha_next_c1_parent.json; the placeholder refuses: rc 3, 'parent not pinned')
#   -> composition (tools/alpha_next_c1_composition.py) -> rebind default-off (tools/alpha_next_c1_rebind_default_off.py) -> review pins --check -> strict -> focused (alpha-next's list plus C1's,
#   the composition fix's, the C1 rebind's (test_c1_rebind_witness, test_c1_rebind_on), the defer_bind control's, alpha-next's
#   b-chain test_alpha_next_probe_launch and the alpha-next-c1 queue suite, through alpha_next_robust_swap) -> composition tests -> prepare review
#   + commit -> build (with alpha-next-c1-freeze.json) -> bundle self-run -> queue pins write + check -> engine self-run (alpha-next's
#   six containers + C1's overlap container) -> comparison (against alpha-next's pinned frozen record) -> negative (two-phase: while
#   tools/alpha_next_c1_negative_pins.json is the AWAITING-OBSERVATION stub, one --observe and stop, rc 5) -> tamper -> clock scan
#   -> compile proof -> precheck conf -> precheck -> staging check -> stop: 'stopped-before-delivery: VM run pending'.
# The VM step (tools/alpha_next_c1_vm_input.py plain and --stage under VM <vm-id>'s GPU flock, tools/alpha_next_c1_vm_compare.py) is the
# lead's, after this chain; with VM_RECORD set (the compare record) the chain runs the delivery itself.
# One heavy job at a time (the hub is shared), so nothing runs in parallel here. No absolute-time window.
# usage: alpha_next_c1_gates.sh WT SPEC OUT EVID TAG NOTE [DELIVERY]
# Stops at the first failing step; every rc is logged in $EVID/gates-$TAG.steps.
set -u
unset ALPHA_NEXT_C1_PARENT   # development pins are for tests and dry runs only; the chain reads the committed pin
WT=$1 SPEC=$2 OUT=$3 EVID=$4 TAG=$5 NOTE=$6 DELIVERY=${7:-$EVID/delivery-$TAG.json}
STEPS=$EVID/gates-$TAG.steps
mkdir -p "$EVID"
[[ ! -e $STEPS && ! -e $OUT && -f $SPEC ]] || exit 3
step() { printf '%s %s rc=%s\n' "$(date -u +%H:%M:%SZ)" "$1" "$2" >>"$STEPS"; [[ $2 == 0 ]] || exit "$2"; }
note_rc() { printf '%s %s rc=%s\n' "$(date -u +%H:%M:%SZ)" "$1" "$2" >>"$STEPS"; }
# flaky KIND TARGET: 0 when every failure of the attempt is a declared flaky test; its verdict line goes to the steps file.
flaky() { local v; v=$(python3 -B "$WT/tools/alpha_c1_flaky.py" "$1" "$2" 2>&1); local rc=$?; printf 'flaky %s %s\n' "$1" "$v" >>"$STEPS"; return $rc; }
cd "$WT" || exit 3
HEAD=$(git rev-parse HEAD)
[[ -z $(git status --porcelain -- harness tools patches) ]] || { echo dirty >>"$STEPS"; exit 3; }
printf 'head %s spec %s\n' "$HEAD" "$(sha256sum "$SPEC" | cut -c1-12)" >"$STEPS"
printf 'priority nice %s; loadavg %s\n' "$(nice)" "$(cut -d' ' -f1-3 /proc/loadavg)" >>"$STEPS"

python3 -B tools/alpha_next_c1_parent_pin.py --pinned >"$EVID/parent-pin-$TAG.log" 2>&1
RC=$?; [[ $RC == 0 ]] || echo 'parent not pinned' >>"$STEPS"
step parent-pin "$RC"
python3 -B tools/alpha_next_c1_composition.py --out "$EVID/composition-$TAG.json" >"$EVID/composition-$TAG.log" 2>&1
step composition $?
# The C1 rebind's default-off proof (FLASH_S5_STAGE_OVERLAP_DEFER_BIND / _OBSERVE unset: the engine is C1's), the coordinator's tool
# tools/alpha_next_c1_rebind_default_off.py: --out FILE (written once), exit 0 only when it holds. Its record goes to the delivery.
python3 -B tools/alpha_next_c1_rebind_default_off.py --out "$EVID/rebind-default-off-$TAG.json" \
    >"$EVID/rebind-default-off-$TAG.log" 2>&1
step rebind-default-off $?
python3 -B tools/alpha_next_c1_review_pins.py --check >"$EVID/review-pins-$TAG.log" 2>&1
step review-pins $?

cd "$WT/harness"
strict() { { python3 -u selftest.py 2>&1; echo "rc=$?"; } >"$1"; tail -n 1 "$1" | sed 's/^rc=//'; }
STRICT_LOG=$EVID/strict-$TAG.log
RC=$(strict "$STRICT_LOG")
if [[ $RC != 0 ]] && flaky strict "$STRICT_LOG"; then
  note_rc strict-1 "$RC"; STRICT_LOG=$EVID/strict-$TAG-2.log; RC=$(strict "$STRICT_LOG"); step strict-2 "$RC"
else step strict "$RC"; fi
focused() { { echo "# head $HEAD focused"; python3 -B alpha_next_robust_swap.py \
    test_alpha_next_c1_queue test_e3_stage_overlap test_alpha_next_c1_composed_fields test_e3_stage_overlap_control \
    test_c1_rebind_witness test_c1_rebind_on test_alpha_next_c1_defer_control test_alpha_next_probe_launch \
    test_alpha_c1_queue test_alpha_c1_box test_alpha_c1_report \
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
# Their own log: both composition suites read git; the pack's tool tests use a development parent read from the tree.
{ python3 -B -m unittest -v test_alpha_next_composition test_alpha_next_c1_pack 2>&1; echo "rc=$?"; } >"$EVID/composition-test-$TAG.log"
step composition-test "$(tail -n 1 "$EVID/composition-test-$TAG.log" | sed 's/^rc=//')"

cd "$WT"
[[ $(git rev-parse HEAD) == "$HEAD" ]] || { echo "head moved" >>"$STEPS"; exit 4; }
python3 -B tools/build_alpha_next_c1.py --spec "$SPEC" --prepare-review --selftest-log "$STRICT_LOG" \
    --focused-log "$FOCUSED_LOG" >"$EVID/prepare-review-$TAG.log" 2>&1
step prepare-review $?
git add patches/s5_engine/native_candidate_review.json && git commit -q -m "alpha-next-c1 build ($TAG): review prepared over $HEAD

strict $(sha256sum "$STRICT_LOG" | cut -c1-12), focused $(sha256sum "$FOCUSED_LOG" | cut -c1-12).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
step review-commit $?
printf 'review-commit %s\n' "$(git rev-parse HEAD)" >>"$STEPS"

python3 -B tools/build_alpha_next_c1.py --spec "$SPEC" --out "$OUT" --selftest-log "$STRICT_LOG" \
    --focused-log "$FOCUSED_LOG" >"$EVID/build-$TAG.log" 2>&1
step build $?
PIN=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['manifest'])" "$OUT/build.json")
P12=${PIN:0:12}
HELD=$OUT/q18a-alpha-next-c1-bundle-$P12
RUNNER=$OUT/q18a-alpha-next-c1-owner-run-$P12.py

python3 -B tools/native43_bundle_selfrun.py --spec "$SPEC" --bundle "$HELD" --out "$OUT/selfrun" \
    --name "alpha-next-c1-hub-cpu-$P12" --runner "$RUNNER" >"$OUT/selfrun-stdout.json" 2>&1
step bundle-selfrun $?

# Hub-side queue pins (tools/alpha_next_c1_queue_pins.json): bundle, overlay and package identity from this build, then checked
# against OUT. The lead commits the pins file; the chain does not. A re-run over filled pins needs PINS_SUPERSEDE="reason".
python3 -B tools/alpha_next_c1_queue_pins_write.py --out "$OUT" ${PINS_SUPERSEDE:+--supersede "$PINS_SUPERSEDE"} \
    >"$OUT/queue-pins-write.log" 2>&1
step queue-pins-write $?
python3 -B tools/alpha_next_c1_queue_pins_write.py --out "$OUT" --check >"$OUT/queue-pins-check.log" 2>&1
step queue-pins-check $?

# engine DIR LABEL: one engine self-run attempt into $OUT/DIR, container alpha-next-c1-LABEL-cpu-<pin>, stdout in $OUT/DIR-stdout.log.
engine() { python3 -B tools/alpha_next_c1_engine_selfrun.py --spec "$SPEC" --tree "$WT" --bundle "$HELD" --out "$OUT/$1" \
    --name "alpha-next-c1-$2-cpu-$P12" >"$OUT/$1-stdout.log" 2>&1; }
ENGINE=$OUT/engine-selfrun
engine engine-selfrun engine; RC=$?
if [[ $RC != 0 ]] && flaky engine "$ENGINE"; then
  note_rc engine-selfrun-1 "$RC"; ENGINE=$OUT/engine-selfrun-2; engine engine-selfrun-2 engine-2; step engine-selfrun-2 $?
else step engine-selfrun "$RC"; fi

python3 -B tools/alpha_next_c1_engine_selfrun.py --compare "$ENGINE" --out "$OUT/engine-comparison.json" \
    >"$OUT/engine-comparison-stdout.log" 2>&1
step engine-comparison $?

if python3 -I -c "import json,sys;sys.exit(0 if str(json.load(open(sys.argv[1]))['status']).startswith('AWAITING-OBSERVATION') else 1)" \
    tools/alpha_next_c1_negative_pins.json; then
  # One observation only (the sweep's rule): observe, then stop for review and the pin writer.
  python3 -B tools/alpha_next_c1_engine_selfrun.py --negative --observe --bundle "$HELD" --overlay "$OUT/parent/native30_overlay" \
      --sources "$OUT/parent/e3_builder" --out "$OUT/engine-negative-observe" --name "alpha-next-c1-negative-observe-cpu-$P12" \
      >"$OUT/engine-negative-observe-stdout.log" 2>&1
  # --observe exits 1 by design; the step passes when the record's verdict is OBSERVED.
  python3 -I -c "import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))['verdict']=='OBSERVED' else 1)" \
      "$OUT/engine-negative-observe/result.json"
  step engine-negative-observe $?
  echo "pins awaiting review: tools/alpha_next_c1_negative_pins_write.py --observed $OUT/engine-negative-observe/result.json --log $OUT/engine-negative-observe-stdout.log --classify REVIEWED.json --discovery TEXT" >>"$STEPS"
  step engine-negative-awaiting-pins 5
fi
python3 -B tools/alpha_next_c1_engine_selfrun.py --negative --bundle "$HELD" --overlay "$OUT/parent/native30_overlay" \
    --sources "$OUT/parent/e3_builder" --out "$OUT/engine-negative" --name "alpha-next-c1-negative-cpu-$P12" \
    >"$OUT/engine-negative-stdout.log" 2>&1
step engine-negative $?

python3 -B tools/alpha_next_c1_overlay_tamper.py --parent "$OUT/parent/native30_overlay" --candidate "$HELD/native30_overlay" \
    --out "$OUT/overlay-tamper.json" >"$OUT/overlay-tamper-stdout.log" 2>&1
step overlay-tamper $?

python3 -B tools/alpha_next_c1_clock_scan.py --bundle "$HELD" --runner "$RUNNER" --setcfg-reference "$WT/deploy/setcfg.py" \
    --out "$OUT/clock-scan.json" >"$OUT/clock-scan-stdout.log" 2>&1
step clock-scan $?

python3 -B tools/native35_compile_proof.py --bundle "$HELD" --runner "$RUNNER" --out "$OUT/compile-proof.json" \
    >"$OUT/compile-proof-stdout.log" 2>&1
step compile-proof $?

# The conf stays with its build; the delivery tool's caller copies the delivered one to tools/owner-precheck/.
python3 -B tools/alpha_next_c1_precheck_conf.py --build "$OUT/build.json" --selfrun "$OUT/selfrun/result.json" \
    --out "$OUT/native-precheck.conf" >"$OUT/precheck-conf.log" 2>&1
step precheck-conf $?

bash tools/owner-precheck/native-precheck.sh "$OUT/native-precheck.conf" >"$OUT/precheck-$TAG.log" 2>&1
step precheck $?

# The frozen runner's own staging require blocks over the exact records delivered (Step B's check, evaluated on the hub first).
python3 -B tools/alpha_next_c1_staging_check.py --runner "$RUNNER" --selfrun "$OUT/selfrun/result.json" \
    --engine "$ENGINE/result.json" --out "$OUT/staging-check.json" >"$OUT/staging-check.log" 2>&1
step staging-check $?

if [[ -z ${VM_RECORD:-} ]]; then echo 'stopped-before-delivery: VM run pending' >>"$STEPS"; exit 0; fi
python3 -B tools/alpha_next_c1_delivery.py --vm "$VM_RECORD" --build "$OUT/build.json" --freeze "$OUT/alpha-next-c1-freeze.json" \
    --engine "$ENGINE/result.json" --compare "$OUT/engine-comparison.json" \
    --negative "$OUT/engine-negative/result.json" --tamper "$OUT/overlay-tamper.json" --queue "$OUT/selfrun/result.json" \
    --clock-scan "$OUT/clock-scan.json" --compile-proof "$OUT/compile-proof.json" --precheck "$OUT/precheck-$TAG.log" \
    --conf "$OUT/native-precheck.conf" --staging-check "$OUT/staging-check.json" \
    --rebind-default-off "$EVID/rebind-default-off-$TAG.json" --note "$NOTE" --out "$DELIVERY" \
    >"$OUT/delivery-stdout.log" 2>&1
step delivery $?
echo DONE >>"$STEPS"
