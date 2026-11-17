#!/usr/bin/env bash
# native43 packaging gate chain (CPU only, hub; no box/VM/GPU action, no push).
# usage: native43-gates.sh WT SPEC OUT EVID TAG
#   focused || strict -> prepare review + commit -> build -> bundle self-run || engine self-run (+ clock split)
#   -> owner-chain seam self-run -> compile proof -> precheck conf -> precheck.
# Stops at the first failing step; every rc is logged in $EVID/gates-$TAG.steps.
set -u
WT=$1 SPEC=$2 OUT=$3 EVID=$4 TAG=$5
BASEDIR=<hub-dir>/agents/flash-next/tmp/e2f-native-offline-base-20261027T211405Z
STEPS=$EVID/gates-$TAG.steps
mkdir -p "$EVID"
[[ ! -e $STEPS && ! -e $OUT && -f $SPEC ]] || exit 3
step() { printf '%s %s rc=%s\n' "$(date -u +%H:%M:%SZ)" "$1" "$2" >>"$STEPS"; [[ $2 == 0 ]] || exit "$2"; }
cd "$WT" || exit 3
HEAD=$(git rev-parse HEAD)
[[ -z $(git status --porcelain -- harness tools patches) ]] || { echo dirty >>"$STEPS"; exit 3; }
printf 'head %s spec %s\n' "$HEAD" "$(sha256sum "$SPEC" | cut -c1-12)" >"$STEPS"
NI='nice -n 10 ionice -c3'

DRY=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['dry_run'])" "$SPEC")
# The two native43 suites exist only in the native43 source; the held-b dry run (dry_run true) has the other 18.
N43_SUITES="test_native43_dense_boundary test_native43_boundary_queue"
[[ $DRY == True ]] && N43_SUITES=
cd "$WT/harness"
{ echo "# head $HEAD focused (dry_run $DRY)"; $NI python3 -B -m unittest test_native39_trunk test_native39_trunk_queue test_native37_robust \
    test_native38_plain_bind test_e3_admission_native test_e3_staged_chunk_plan test_e3_native_overlap \
    test_e3_native_storage_replay test_e3_r4_worker_bridge test_e3_admission_metadata test_native35_p_c_continuation \
    test_e3_native28_host_guard test_identity_declare test_e3_native_measurement \
    $N43_SUITES test_e3_qsa_driver test_e3_lane_private \
    test_e3_sparse_history_dma test_e3_worker_operations 2>&1; echo "rc=$?"; } >"$EVID/focused-$TAG.log" &
FOCUSED=$!
{ $NI python3 -u selftest.py 2>&1; echo "rc=$?"; } >"$EVID/strict-$TAG.log"
wait $FOCUSED
step focused "$(tail -n 1 "$EVID/focused-$TAG.log" | sed 's/^rc=//')"
step strict "$(tail -n 1 "$EVID/strict-$TAG.log" | sed 's/^rc=//')"

cd "$WT"
[[ $(git rev-parse HEAD) == "$HEAD" ]] || { echo "head moved" >>"$STEPS"; exit 4; }
REVIEW_MODE=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['review'])" "$SPEC")
if [[ $REVIEW_MODE == prepare ]]; then
python3 -B tools/build_native43.py --spec "$SPEC" --prepare-review --selftest-log "$EVID/strict-$TAG.log" \
    --focused-log "$EVID/focused-$TAG.log" >"$EVID/prepare-review-$TAG.log" 2>&1
step prepare-review $?
git add patches/s5_engine/native_candidate_review.json && git commit -q -m "native43 build ($TAG): review prepared over $HEAD

strict $(sha256sum "$EVID/strict-$TAG.log" | cut -c1-12), focused $(sha256sum "$EVID/focused-$TAG.log" | cut -c1-12).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
step review-commit $?
printf 'review-commit %s\n' "$(git rev-parse HEAD)" >>"$STEPS"
else
printf 'review %s: committed review inherited, no prepare\n' "$REVIEW_MODE" >>"$STEPS"
fi

$NI python3 -B tools/build_native43.py --spec "$SPEC" --out "$OUT" --selftest-log "$EVID/strict-$TAG.log" \
    --focused-log "$EVID/focused-$TAG.log" >"$EVID/build-$TAG.log" 2>&1
step build $?
PIN=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['manifest'])" "$OUT/build.json")
P12=${PIN:0:12}
NAME=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['name'])" "$OUT/build.json")
T=${NAME#q18a-}
HELD=$OUT/$NAME-bundle-$P12
RUNNER=$OUT/$NAME-owner-run-$P12.py

$NI python3 -B tools/native43_bundle_selfrun.py --spec "$SPEC" --bundle "$HELD" --out "$OUT/selfrun" \
    --name "$T-hub-cpu-$P12" --runner "$RUNNER" >"$OUT/selfrun-stdout.json" 2>&1 &
BUNDLE_SR=$!
$NI python3 -B tools/native43_selfrun.py --spec "$SPEC" --tree "$WT" --bundle "$HELD" --out "$OUT/engine-selfrun" \
    --name "$T-engine-cpu-$P12" >"$OUT/engine-selfrun-stdout.log" 2>&1
ENGINE_RC=$?
wait $BUNDLE_SR
step bundle-selfrun $?
step engine-selfrun $ENGINE_RC

if [[ $DRY == True ]]; then
printf 'owner-seam not applicable: dry run (held-b runner, no pair queue)\n' >>"$STEPS"
else
$NI python3 -B tools/native43_owner_seam.py --bundle "$HELD" --runner "$RUNNER" --base "$BASEDIR" \
    --bundle-record "$OUT/selfrun/result.json" --engine-record "$OUT/engine-selfrun/result.json" \
    --out "$OUT/owner-seam" --name "$T-seam-cpu-$P12" >"$OUT/owner-seam-stdout.log" 2>&1
SEAM_RC=$?
if [[ ${SEAM_SOFT:-0} == 1 && $SEAM_RC != 0 ]]; then
  printf '%s owner-seam rc=%s (SEAM_SOFT: logged, chain continues; dev run only)\n' "$(date -u +%H:%M:%SZ)" "$SEAM_RC" >>"$STEPS"
else
  step owner-seam $SEAM_RC
fi
fi

$NI python3 -B tools/native35_compile_proof.py --bundle "$HELD" --runner "$RUNNER" --out "$OUT/compile-proof.json" \
    >"$OUT/compile-proof-stdout.log" 2>&1
step compile-proof $?

python3 -B tools/native43_precheck_conf.py --spec "$SPEC" --build "$OUT/build.json" --selfrun "$OUT/selfrun/result.json" \
    --out "$WT/tools/owner-precheck/native-precheck.conf.native43-$TAG" >"$OUT/precheck-conf.log" 2>&1
step precheck-conf $?

bash tools/owner-precheck/native-precheck.sh "$WT/tools/owner-precheck/native-precheck.conf.native43-$TAG" \
    >"$OUT/precheck-$TAG.log" 2>&1
step precheck $?
echo DONE >>"$STEPS"
