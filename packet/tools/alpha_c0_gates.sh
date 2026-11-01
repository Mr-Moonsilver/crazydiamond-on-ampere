#!/usr/bin/env bash
# C0 packaging gate chain (CPU only, hub; no box/VM/GPU action, no push). tools/alpha_perf_gates.sh's chain with:
# build_alpha_c0.py (alpha-perf build-d parent); alpha-perf's and held-g's focused lists plus the C0 suites; the merged engine
# self-run with its comparison, negative and the overlay tamper control; one heavy job at a time (the hub is shared), so nothing
# runs in parallel here. The runner lineage is build_native43's own normalise-back proof over the alpha-perf runner.
# usage: alpha_c0_gates.sh WT SPEC OUT EVID TAG NOTE [DELIVERY]
#   The VM run needs the built bundle, so with no VM_RECORD in the environment the chain stops after the precheck; the delivery tool
#   is then run by hand with --vm. The chain stops there on purpose and logs it.
#   strict -> focused -> prepare review + commit -> build (with alpha-c0-freeze.json) -> bundle self-run
#   -> engine self-run -> comparison -> negative -> tamper -> clock scan -> compile proof -> precheck conf -> precheck -> delivery.
# Stops at the first failing step; every rc is logged in $EVID/gates-$TAG.steps.
set -u
WT=$1 SPEC=$2 OUT=$3 EVID=$4 TAG=$5 NOTE=$6 DELIVERY=${7:-$EVID/delivery-$TAG.json}
STEPS=$EVID/gates-$TAG.steps
mkdir -p "$EVID"
[[ ! -e $STEPS && ! -e $OUT && -f $SPEC ]] || exit 3
step() { printf '%s %s rc=%s\n' "$(date -u +%H:%M:%SZ)" "$1" "$2" >>"$STEPS"; [[ $2 == 0 ]] || exit "$2"; }
cd "$WT" || exit 3
HEAD=$(git rev-parse HEAD)
[[ -z $(git status --porcelain -- harness tools patches) ]] || { echo dirty >>"$STEPS"; exit 3; }
printf 'head %s spec %s\n' "$HEAD" "$(sha256sum "$SPEC" | cut -c1-12)" >"$STEPS"
printf 'priority nice 0 (normal); loadavg %s\n' "$(cut -d' ' -f1-3 /proc/loadavg)" >>"$STEPS"

cd "$WT/harness"
{ python3 -u selftest.py 2>&1; echo "rc=$?"; } >"$EVID/strict-$TAG.log"
step strict "$(tail -n 1 "$EVID/strict-$TAG.log" | sed 's/^rc=//')"
{ echo "# head $HEAD focused"; python3 -B -m unittest test_alpha_c0_queue test_e3_stage_local test_e3_stage_local_control test_e3_stage_local_qsa test_alpha_c0_report \
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
    test_e3_report_validation_cost 2>&1; echo "rc=$?"; } >"$EVID/focused-$TAG.log"
step focused "$(tail -n 1 "$EVID/focused-$TAG.log" | sed 's/^rc=//')"

cd "$WT"
[[ $(git rev-parse HEAD) == "$HEAD" ]] || { echo "head moved" >>"$STEPS"; exit 4; }
python3 -B tools/build_alpha_c0.py --spec "$SPEC" --prepare-review --selftest-log "$EVID/strict-$TAG.log" \
    --focused-log "$EVID/focused-$TAG.log" >"$EVID/prepare-review-$TAG.log" 2>&1
step prepare-review $?
git add patches/s5_engine/native_candidate_review.json && git commit -q -m "C0 build ($TAG): review prepared over $HEAD

strict $(sha256sum "$EVID/strict-$TAG.log" | cut -c1-12), focused $(sha256sum "$EVID/focused-$TAG.log" | cut -c1-12).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
step review-commit $?
printf 'review-commit %s\n' "$(git rev-parse HEAD)" >>"$STEPS"

python3 -B tools/build_alpha_c0.py --spec "$SPEC" --out "$OUT" --selftest-log "$EVID/strict-$TAG.log" \
    --focused-log "$EVID/focused-$TAG.log" >"$EVID/build-$TAG.log" 2>&1
step build $?
PIN=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['manifest'])" "$OUT/build.json")
P12=${PIN:0:12}
HELD=$OUT/q18a-alpha-c0-bundle-$P12
RUNNER=$OUT/q18a-alpha-c0-owner-run-$P12.py

python3 -B tools/native43_bundle_selfrun.py --spec "$SPEC" --bundle "$HELD" --out "$OUT/selfrun" \
    --name "alpha-c0-hub-cpu-$P12" --runner "$RUNNER" >"$OUT/selfrun-stdout.json" 2>&1
step bundle-selfrun $?

python3 -B tools/alpha_c0_engine_selfrun.py --spec "$SPEC" --tree "$WT" --bundle "$HELD" --out "$OUT/engine-selfrun" \
    --name "alpha-c0-engine-cpu-$P12" >"$OUT/engine-selfrun-stdout.log" 2>&1
step engine-selfrun $?

python3 -B tools/alpha_c0_engine_selfrun.py --compare "$OUT/engine-selfrun" --out "$OUT/engine-comparison.json" \
    >"$OUT/engine-comparison-stdout.log" 2>&1
step engine-comparison $?

python3 -B tools/alpha_c0_engine_selfrun.py --negative --bundle "$HELD" --overlay "$OUT/parent/native30_overlay" \
    --sources "$OUT/parent/e3_builder" --out "$OUT/engine-negative" --name "alpha-c0-negative-cpu-$P12" \
    >"$OUT/engine-negative-stdout.log" 2>&1
step engine-negative $?

python3 -B tools/alpha_c0_overlay_tamper.py --parent "$OUT/parent/native30_overlay" --candidate "$HELD/native30_overlay" \
    --out "$OUT/overlay-tamper.json" >"$OUT/overlay-tamper-stdout.log" 2>&1
step overlay-tamper $?

python3 -B tools/alpha_c0_clock_scan.py --bundle "$HELD" --runner "$RUNNER" --setcfg-reference "$WT/deploy/setcfg.py" \
    --out "$OUT/clock-scan.json" >"$OUT/clock-scan-stdout.log" 2>&1
step clock-scan $?

python3 -B tools/native35_compile_proof.py --bundle "$HELD" --runner "$RUNNER" --out "$OUT/compile-proof.json" \
    >"$OUT/compile-proof-stdout.log" 2>&1
step compile-proof $?

# The conf stays with its build; the delivery tool's caller copies the delivered one to tools/owner-precheck/.
python3 -B tools/alpha_c0_precheck_conf.py --build "$OUT/build.json" --selfrun "$OUT/selfrun/result.json" \
    --out "$OUT/native-precheck.conf" >"$OUT/precheck-conf.log" 2>&1
step precheck-conf $?

bash tools/owner-precheck/native-precheck.sh "$OUT/native-precheck.conf" >"$OUT/precheck-$TAG.log" 2>&1
step precheck $?

if [[ -z ${VM_RECORD:-} ]]; then echo 'stopped-before-delivery: VM run pending' >>"$STEPS"; exit 0; fi
python3 -B tools/alpha_c0_delivery.py --vm "$VM_RECORD" --build "$OUT/build.json" --freeze "$OUT/alpha-c0-freeze.json" \
    --engine "$OUT/engine-selfrun/result.json" --compare "$OUT/engine-comparison.json" \
    --negative "$OUT/engine-negative/result.json" --tamper "$OUT/overlay-tamper.json" --queue "$OUT/selfrun/result.json" \
    --clock-scan "$OUT/clock-scan.json" --compile-proof "$OUT/compile-proof.json" --precheck "$OUT/precheck-$TAG.log" \
    --conf "$OUT/native-precheck.conf" --note "$NOTE" --out "$DELIVERY" >"$OUT/delivery-stdout.log" 2>&1
step delivery $?
echo DONE >>"$STEPS"
