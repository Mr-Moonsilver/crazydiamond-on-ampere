#!/usr/bin/env bash
# alpha-next packaging gate chain, b: tools/alpha_next_gates.sh (chains g1-g4 ran it; superseded here, never edited) with two
# changes: the engine self-run, comparison and negative steps call tools/alpha_next_engine_selfrun_b.py (every probe case through
# harness/alpha_next_probe_launch.py: a real ReceiptWriter whose records are read back with the box grader's parser; g4 engine
# self-run, HARNESS), and the focused list adds test_alpha_next_probe_launch. Everything below is the base chain's text.
# alpha-next packaging gate chain (CPU only, hub; no box/VM/GPU action, no push). tools/alpha_c0_gates_g.sh's chain shape (C0 was the
# first candidate to add overlay modules) over Step B's tool family:
#   build_alpha_next.py (Step B's frozen build e as parent, C0's added-module wrappers, TOOLS_REPLACED step_a_report.py);
#   two steps before strict: the git-based composition proof (tools/alpha_next_composition.py) and the review pins check
#   (tools/alpha_next_review_pins.py --check);
#   the focused list is the union of Step B's (step_b_gates_c.sh) and C0's (alpha_c0_gates_g.sh) lists plus the alpha-next queue suite,
#   the alpha-next stop-fix matrix, C0 ring, peak receipt, receipt writer (PLAN 2, with its mutants), stage-1 slow-rank and 4C.5 path qualification suites, the stage-1 anatomy suite, held-g's anatomy and lean-replies suites
#   and the two load-robust twins; it runs through harness/alpha_next_robust_swap.py, which leaves out the declared load-flaky frozen
#   methods (swapped for their twins), the superseded stage-1 FreezeProof method and step_b_clock_bound's declared real-clock ids;
#   test_alpha_next_composition runs on its own (it needs git), after focused;
#   the engine self-run (Step B's four containers, C0's stage container, the composed container with held-g's probe cases and the
#   composed suites) with its comparison against Step B's frozen record, the negative over the regenerated Step B overlay, and the
#   overlay tamper control;
#   negative pins: while tools/alpha_next_negative_pins.json is the AWAITING-OBSERVATION stub, the negative step runs once with
#   --observe instead (into $OUT/engine-negative-observe) and the chain stops there (rc 5, logged). Then: review the observation,
#   write the pins with one command (tools/alpha_next_negative_pins_write.py --observed .../result.json --log ...-stdout.log
#   --classify REVIEWED.json --discovery TEXT), commit the pins file alone, and run the chain again (new TAG and OUT: the head moved);
#   after the precheck, the staging check (the frozen runner's own staging requires over the exact records, tools/alpha_next_staging_check.py).
# One heavy job at a time (the hub is shared), so nothing runs in parallel here. No absolute-time window: launch time is the lead's.
# usage: alpha_next_gates.sh WT SPEC OUT EVID TAG NOTE [DELIVERY]
#   The VM run needs the built bundle, so with no VM_RECORD in the environment the chain stops after the staging check; the delivery
#   tool is then run by hand with --vm (or with VM_RECORD set, by this chain). The chain stops there on purpose and logs it.
#   composition -> review pins -> strict -> focused -> composition test -> prepare review + commit -> build (with alpha-next-freeze.json)
#   -> bundle self-run -> queue pins write + check (tools/alpha_next_queue_pins.json, committed by the lead) -> engine self-run -> comparison -> negative -> tamper -> clock scan -> compile proof -> precheck conf
#   -> precheck -> staging check -> [delivery when VM_RECORD is set].
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
printf 'priority nice %s; loadavg %s\n' "$(nice)" "$(cut -d' ' -f1-3 /proc/loadavg)" >>"$STEPS"

python3 -B tools/alpha_next_composition.py --out "$EVID/composition-$TAG.json" >"$EVID/composition-$TAG.log" 2>&1
step composition $?
# The review's source and harness pins equal the committed tree (any later source commit means re-running the tool and committing).
python3 -B tools/alpha_next_review_pins.py --check >"$EVID/review-pins-$TAG.log" 2>&1
step review-pins $?

cd "$WT/harness"
{ python3 -u selftest.py 2>&1; echo "rc=$?"; } >"$EVID/strict-$TAG.log"
step strict "$(tail -n 1 "$EVID/strict-$TAG.log" | sed 's/^rc=//')"
{ echo "# head $HEAD focused"; python3 -B alpha_next_robust_swap.py \
    test_alpha_next_queue test_alpha_next_receipts test_alpha_next_receipts_mutants test_alpha_next_stage1_slow_rank \
    test_alpha_next_path_qualification test_alpha_next_probe_launch test_alpha_next_stopfix_matrix test_alpha_next_c0_ring test_alpha_next_peak_receipt \
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
    test_e3_probe_session test_e3_native_startup_binding test_e3_wire_v1 test_e3_report_validation_cost 2>&1; echo "rc=$?"; } >"$EVID/focused-$TAG.log"
step focused "$(tail -n 1 "$EVID/focused-$TAG.log" | sed 's/^rc=//')"
# Its own log: the composition test reads git, and the review's focused basis is one unittest run (focused-$TAG.log).
{ python3 -B -m unittest -v test_alpha_next_composition 2>&1; echo "rc=$?"; } >"$EVID/composition-test-$TAG.log"
step composition-test "$(tail -n 1 "$EVID/composition-test-$TAG.log" | sed 's/^rc=//')"

cd "$WT"
[[ $(git rev-parse HEAD) == "$HEAD" ]] || { echo "head moved" >>"$STEPS"; exit 4; }
python3 -B tools/build_alpha_next.py --spec "$SPEC" --prepare-review --selftest-log "$EVID/strict-$TAG.log" \
    --focused-log "$EVID/focused-$TAG.log" >"$EVID/prepare-review-$TAG.log" 2>&1
step prepare-review $?
git add patches/s5_engine/native_candidate_review.json && git commit -q -m "alpha-next build ($TAG): review prepared over $HEAD

strict $(sha256sum "$EVID/strict-$TAG.log" | cut -c1-12), focused $(sha256sum "$EVID/focused-$TAG.log" | cut -c1-12).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
step review-commit $?
printf 'review-commit %s\n' "$(git rev-parse HEAD)" >>"$STEPS"

python3 -B tools/build_alpha_next.py --spec "$SPEC" --out "$OUT" --selftest-log "$EVID/strict-$TAG.log" \
    --focused-log "$EVID/focused-$TAG.log" >"$EVID/build-$TAG.log" 2>&1
step build $?
PIN=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['manifest'])" "$OUT/build.json")
P12=${PIN:0:12}
HELD=$OUT/q18a-alpha-next-bundle-$P12
RUNNER=$OUT/q18a-alpha-next-owner-run-$P12.py

python3 -B tools/native43_bundle_selfrun.py --spec "$SPEC" --bundle "$HELD" --out "$OUT/selfrun" \
    --name "alpha-next-hub-cpu-$P12" --runner "$RUNNER" >"$OUT/selfrun-stdout.json" 2>&1
step bundle-selfrun $?

# Hub-side queue pins (tools/alpha_next_queue_pins.json): bundle, overlay and package identity from this build, then checked
# against OUT. The lead commits the pins file; the chain does not. A re-run over filled pins needs PINS_SUPERSEDE="reason".
python3 -B tools/alpha_next_queue_pins_write.py --out "$OUT" ${PINS_SUPERSEDE:+--supersede "$PINS_SUPERSEDE"} \
    >"$OUT/queue-pins-write.log" 2>&1
step queue-pins-write $?
python3 -B tools/alpha_next_queue_pins_write.py --out "$OUT" --check >"$OUT/queue-pins-check.log" 2>&1
step queue-pins-check $?

python3 -B tools/alpha_next_engine_selfrun_b.py --spec "$SPEC" --tree "$WT" --bundle "$HELD" --out "$OUT/engine-selfrun" \
    --name "alpha-next-engine-cpu-$P12" >"$OUT/engine-selfrun-stdout.log" 2>&1
step engine-selfrun $?

python3 -B tools/alpha_next_engine_selfrun_b.py --compare "$OUT/engine-selfrun" --out "$OUT/engine-comparison.json" \
    >"$OUT/engine-comparison-stdout.log" 2>&1
step engine-comparison $?

if python3 -I -c "import json,sys;sys.exit(0 if str(json.load(open(sys.argv[1]))['status']).startswith('AWAITING-OBSERVATION') else 1)" \
    tools/alpha_next_negative_pins.json; then
  # One observation only (the sweep's rule): observe, then stop for review and the pin writer.
  python3 -B tools/alpha_next_engine_selfrun_b.py --negative --observe --bundle "$HELD" --overlay "$OUT/parent/native30_overlay" \
      --sources "$OUT/parent/e3_builder" --out "$OUT/engine-negative-observe" --name "alpha-next-negative-observe-cpu-$P12" \
      >"$OUT/engine-negative-observe-stdout.log" 2>&1
  # --observe exits 1 by design; the step passes when the record's verdict is OBSERVED.
  python3 -I -c "import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))['verdict']=='OBSERVED' else 1)" \
      "$OUT/engine-negative-observe/result.json"
  step engine-negative-observe $?
  echo "pins awaiting review: tools/alpha_next_negative_pins_write.py --observed $OUT/engine-negative-observe/result.json --log $OUT/engine-negative-observe-stdout.log --classify REVIEWED.json --discovery TEXT" >>"$STEPS"
  step engine-negative-awaiting-pins 5
fi
python3 -B tools/alpha_next_engine_selfrun_b.py --negative --bundle "$HELD" --overlay "$OUT/parent/native30_overlay" \
    --sources "$OUT/parent/e3_builder" --out "$OUT/engine-negative" --name "alpha-next-negative-cpu-$P12" \
    >"$OUT/engine-negative-stdout.log" 2>&1
step engine-negative $?

python3 -B tools/alpha_next_overlay_tamper.py --parent "$OUT/parent/native30_overlay" --candidate "$HELD/native30_overlay" \
    --out "$OUT/overlay-tamper.json" >"$OUT/overlay-tamper-stdout.log" 2>&1
step overlay-tamper $?

python3 -B tools/alpha_next_clock_scan.py --bundle "$HELD" --runner "$RUNNER" --setcfg-reference "$WT/deploy/setcfg.py" \
    --out "$OUT/clock-scan.json" >"$OUT/clock-scan-stdout.log" 2>&1
step clock-scan $?

python3 -B tools/native35_compile_proof.py --bundle "$HELD" --runner "$RUNNER" --out "$OUT/compile-proof.json" \
    >"$OUT/compile-proof-stdout.log" 2>&1
step compile-proof $?

# The conf stays with its build; the delivery tool's caller copies the delivered one to tools/owner-precheck/.
python3 -B tools/alpha_next_precheck_conf.py --build "$OUT/build.json" --selfrun "$OUT/selfrun/result.json" \
    --out "$OUT/native-precheck.conf" >"$OUT/precheck-conf.log" 2>&1
step precheck-conf $?

bash tools/owner-precheck/native-precheck.sh "$OUT/native-precheck.conf" >"$OUT/precheck-$TAG.log" 2>&1
step precheck $?

# The frozen runner's own staging require blocks over the exact records delivered (Step B's check, evaluated on the hub first).
python3 -B tools/alpha_next_staging_check.py --runner "$RUNNER" --selfrun "$OUT/selfrun/result.json" \
    --engine "$OUT/engine-selfrun/result.json" --out "$OUT/staging-check.json" >"$OUT/staging-check.log" 2>&1
step staging-check $?

if [[ -z ${VM_RECORD:-} ]]; then echo 'stopped-before-delivery: VM run pending' >>"$STEPS"; exit 0; fi
python3 -B tools/alpha_next_delivery.py --vm "$VM_RECORD" --build "$OUT/build.json" --freeze "$OUT/alpha-next-freeze.json" \
    --engine "$OUT/engine-selfrun/result.json" --compare "$OUT/engine-comparison.json" \
    --negative "$OUT/engine-negative/result.json" --tamper "$OUT/overlay-tamper.json" --queue "$OUT/selfrun/result.json" \
    --clock-scan "$OUT/clock-scan.json" --compile-proof "$OUT/compile-proof.json" --precheck "$OUT/precheck-$TAG.log" \
    --conf "$OUT/native-precheck.conf" --staging-check "$OUT/staging-check.json" --note "$NOTE" --out "$DELIVERY" \
    >"$OUT/delivery-stdout.log" 2>&1
step delivery $?
echo DONE >>"$STEPS"
