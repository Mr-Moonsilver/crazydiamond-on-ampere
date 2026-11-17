#!/usr/bin/env bash
# C1 packaging gate chain (CPU only, hub; no box/VM/GPU action, no push): tools/alpha_c0_gates_g.sh (C0 l's chain) with C1's tools.
# build_alpha_c1.py (C0 l build-l parent); C0 l's focused list through harness/clock_bound.py plus the C1 suites; the merged engine
# self-run (C0's cases and suites plus C1's) with its comparison against C0 l's frozen record, the negative and the overlay tamper
# control; one heavy job at a time (the hub is shared), so nothing runs in parallel here. The runner lineage is build_native43's own
# normalise-back proof over C0 l's runner.
# One change in form (the lead, 2 Nov): strict, focused and the engine self-run each get one unchanged rerun of the whole step when
# tools/alpha_c1_flaky.py finds every failure in the declared flaky list (parsed FAIL:/ERROR: lines only; anything unparsed is not
# flaky). Both logs are kept (NAME and NAME-2) and both rc's are in the steps file; later steps read the passing attempt.
# usage: alpha_c1_gates.sh WT SPEC OUT EVID TAG NOTE [DELIVERY]
#   The VM run needs the built bundle, so with no VM_RECORD in the environment the chain stops after the precheck; the delivery tool
#   is then run by hand with --vm and --staging-check (tools/runner_staging_check.py). The chain stops there on purpose and logs it.
#   strict -> focused -> prepare review + commit -> build (with alpha-c1-freeze.json) -> bundle self-run
#   -> engine self-run -> comparison -> negative -> tamper -> clock scan -> compile proof -> precheck conf -> precheck [-> delivery].
# Stops at the first failing step; every rc is logged in $EVID/gates-$TAG.steps.
set -u
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
printf 'priority nice 0 (normal); loadavg %s\n' "$(cut -d' ' -f1-3 /proc/loadavg)" >>"$STEPS"

cd "$WT/harness"
strict() { { python3 -u selftest.py 2>&1; echo "rc=$?"; } >"$1"; tail -n 1 "$1" | sed 's/^rc=//'; }
STRICT_LOG=$EVID/strict-$TAG.log
RC=$(strict "$STRICT_LOG")
if [[ $RC != 0 ]] && flaky strict "$STRICT_LOG"; then
  note_rc strict-1 "$RC"; STRICT_LOG=$EVID/strict-$TAG-2.log; RC=$(strict "$STRICT_LOG"); step strict-2 "$RC"
else step strict "$RC"; fi
focused() { { echo "# head $HEAD focused"; python3 -B clock_bound.py -m test_alpha_c1_queue test_alpha_c1_box test_alpha_c1_report \
    test_e3_stage_overlap test_e3_stage_overlap_control test_alpha_c0_queue test_e3_stage_local test_e3_stage_local_control test_e3_stage_local_qsa test_alpha_c0_report \
    test_alpha_perf_queue test_native41_progress_queue test_e3_native_reply_hol test_native41_box_queue test_native41_autoarm \
    test_native41_unarmed_box test_native41_tp_skew test_alpha_m3_receipts test_native39_trunk test_native39_trunk_queue \
    test_native37_robust test_native38_plain_bind test_e3_admission_native test_e3_staged_chunk_plan test_e3_native_overlap \
    test_e3_native_storage_replay test_e3_r4_worker_bridge test_e3_admission_metadata test_native35_p_c_continuation \
    test_e3_native28_host_guard test_identity_declare test_e3_native_measurement test_native43_dense_boundary \
    test_native43_boundary_queue test_e3_qsa_driver test_e3_lane_private test_e3_sparse_history_dma test_e3_worker_operations \
    test_e3_tp_finalize_skew test_e3_native_measurement_clients test_native35_performance test_native43_engine_death \
    test_native43_le_fatal_log test_native43_le_bounded_enqueue test_native43_le_fix_queue_deltas test_native40_unarmed_hold \
    test_native40_box_queue test_e3_package test_e3_native_schedule_profile test_e3_native_submission_credit \
    test_e3_native_selection_asymmetry test_s5_reply_transport test_e3_executor_threads test_e3_stock_rpc_credits \
    test_e3_async_handoffs test_e3_probe_control test_e3_probe_session test_e3_native_startup_binding test_e3_wire_v1 \
    test_e3_report_validation_cost 2>&1; echo "rc=$?"; } >"$1"; tail -n 1 "$1" | sed 's/^rc=//'; }
FOCUSED_LOG=$EVID/focused-$TAG.log
RC=$(focused "$FOCUSED_LOG")
if [[ $RC != 0 ]] && flaky unittest "$FOCUSED_LOG"; then
  note_rc focused-1 "$RC"; FOCUSED_LOG=$EVID/focused-$TAG-2.log; RC=$(focused "$FOCUSED_LOG"); step focused-2 "$RC"
else step focused "$RC"; fi

cd "$WT"
[[ $(git rev-parse HEAD) == "$HEAD" ]] || { echo "head moved" >>"$STEPS"; exit 4; }
python3 -B tools/build_alpha_c1.py --spec "$SPEC" --prepare-review --selftest-log "$STRICT_LOG" \
    --focused-log "$FOCUSED_LOG" >"$EVID/prepare-review-$TAG.log" 2>&1
step prepare-review $?
git add patches/s5_engine/native_candidate_review.json && git commit -q -m "C1 build ($TAG): review prepared over $HEAD

strict $(sha256sum "$STRICT_LOG" | cut -c1-12), focused $(sha256sum "$FOCUSED_LOG" | cut -c1-12).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
step review-commit $?
printf 'review-commit %s\n' "$(git rev-parse HEAD)" >>"$STEPS"

python3 -B tools/build_alpha_c1.py --spec "$SPEC" --out "$OUT" --selftest-log "$STRICT_LOG" \
    --focused-log "$FOCUSED_LOG" >"$EVID/build-$TAG.log" 2>&1
step build $?
PIN=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['manifest'])" "$OUT/build.json")
P12=${PIN:0:12}
HELD=$OUT/q18a-alpha-c1-bundle-$P12
RUNNER=$OUT/q18a-alpha-c1-owner-run-$P12.py

python3 -B tools/native43_bundle_selfrun.py --spec "$SPEC" --bundle "$HELD" --out "$OUT/selfrun" \
    --name "alpha-c1-hub-cpu-$P12" --runner "$RUNNER" >"$OUT/selfrun-stdout.json" 2>&1
step bundle-selfrun $?

# engine DIR LABEL: one engine self-run attempt into $OUT/DIR, container alpha-c1-LABEL-cpu-<pin>, stdout in $OUT/DIR-stdout.log.
engine() { python3 -B tools/alpha_c1_engine_selfrun.py --spec "$SPEC" --tree "$WT" --bundle "$HELD" --out "$OUT/$1" \
    --name "alpha-c1-$2-cpu-$P12" >"$OUT/$1-stdout.log" 2>&1; }
ENGINE=$OUT/engine-selfrun
engine engine-selfrun engine; RC=$?
if [[ $RC != 0 ]] && flaky engine "$ENGINE"; then
  note_rc engine-selfrun-1 "$RC"; ENGINE=$OUT/engine-selfrun-2; engine engine-selfrun-2 engine-2; step engine-selfrun-2 $?
else step engine-selfrun "$RC"; fi

python3 -B tools/alpha_c1_engine_selfrun.py --compare "$ENGINE" --out "$OUT/engine-comparison.json" \
    >"$OUT/engine-comparison-stdout.log" 2>&1
step engine-comparison $?

python3 -B tools/alpha_c1_engine_selfrun.py --negative --bundle "$HELD" --overlay "$OUT/parent/native30_overlay" \
    --sources "$OUT/parent/e3_builder" --out "$OUT/engine-negative" --name "alpha-c1-negative-cpu-$P12" \
    >"$OUT/engine-negative-stdout.log" 2>&1
step engine-negative $?

python3 -B tools/alpha_c1_overlay_tamper.py --parent "$OUT/parent/native30_overlay" --candidate "$HELD/native30_overlay" \
    --out "$OUT/overlay-tamper.json" >"$OUT/overlay-tamper-stdout.log" 2>&1
step overlay-tamper $?

python3 -B tools/alpha_c1_clock_scan.py --bundle "$HELD" --runner "$RUNNER" --setcfg-reference "$WT/deploy/setcfg.py" \
    --out "$OUT/clock-scan.json" >"$OUT/clock-scan-stdout.log" 2>&1
step clock-scan $?

python3 -B tools/native35_compile_proof.py --bundle "$HELD" --runner "$RUNNER" --out "$OUT/compile-proof.json" \
    >"$OUT/compile-proof-stdout.log" 2>&1
step compile-proof $?

# The conf stays with its build; the delivery tool's caller copies the delivered one to tools/owner-precheck/.
python3 -B tools/alpha_c1_precheck_conf.py --build "$OUT/build.json" --selfrun "$OUT/selfrun/result.json" \
    --out "$OUT/native-precheck.conf" >"$OUT/precheck-conf.log" 2>&1
step precheck-conf $?

bash tools/owner-precheck/native-precheck.sh "$OUT/native-precheck.conf" >"$OUT/precheck-$TAG.log" 2>&1
step precheck $?

if [[ -z ${VM_RECORD:-} ]]; then echo 'stopped-before-delivery: VM run pending' >>"$STEPS"; exit 0; fi
[[ -n ${STAGING_CHECK:-} ]] || { echo 'delivery needs STAGING_CHECK (tools/runner_staging_check.py record)' >>"$STEPS"; exit 3; }
python3 -B tools/alpha_c1_delivery.py --vm "$VM_RECORD" --build "$OUT/build.json" --freeze "$OUT/alpha-c1-freeze.json" \
    --staging-check "$STAGING_CHECK" --engine "$ENGINE/result.json" --compare "$OUT/engine-comparison.json" \
    --negative "$OUT/engine-negative/result.json" --tamper "$OUT/overlay-tamper.json" --queue "$OUT/selfrun/result.json" \
    --clock-scan "$OUT/clock-scan.json" --compile-proof "$OUT/compile-proof.json" --precheck "$OUT/precheck-$TAG.log" \
    --conf "$OUT/native-precheck.conf" --note "$NOTE" --out "$DELIVERY" >"$OUT/delivery-stdout.log" 2>&1
step delivery $?
echo DONE >>"$STEPS"
