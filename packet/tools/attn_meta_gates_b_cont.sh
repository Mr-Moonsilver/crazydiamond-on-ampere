#!/usr/bin/env bash
# attn_meta_gates_b_cont.sh: the continuation of attn_meta_gates_b.sh after its negative observation stop (rc 5), derived by
# tools/attn_meta_b_derive.py: the same preamble (wait_clear, step), then the chain's text from the engine negative to its stop line,
# unchanged. Refuses unless the steps file ends at 'engine-negative-awaiting-pins rc=5', the pins are filled and committed (hub-side,
# so the bundle did not move: the build's manifest is re-checked) and the tree is clean. Same usage, same steps file; appended to it.
set -u
unset ATTN_META_PARENT ATTN_META_PARENT_B ALPHA_NEXT_C1_PARENT FLASH_ATTN_META_GDN_BATCH   # development pins are for tests only; the switch is unset
WT=$1 SPEC=$2 OUT=$3 EVID=$4 TAG=$5 NOTE=$6 CLEAR=$7
STEPS=$EVID/gates-$TAG.steps
mkdir -p "$EVID"
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
[[ -f $STEPS && -f $OUT/build.json ]] || exit 3
[[ $(grep -E ' rc=[0-9]+$' "$STEPS" | tail -n 1 | cut -d' ' -f2-) == 'engine-negative-awaiting-pins rc=5' ]] || { echo 'continuation refused: not at the observation stop' >>"$STEPS"; exit 3; }
[[ -z $(git status --porcelain -- harness tools patches) ]] || { echo 'continuation refused: dirty' >>"$STEPS"; exit 3; }
python3 -I -c "import json,sys;sys.exit(1 if str(json.load(open(sys.argv[1]))['status']).startswith('AWAITING-OBSERVATION') else 0)" tools/attn_meta_negative_pins.json \
    || { echo 'continuation refused: pins not filled' >>"$STEPS"; exit 3; }
HEAD=$(git rev-parse HEAD)
printf 'continuation head %s\n' "$HEAD" >>"$STEPS"
PIN=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['manifest'])" "$OUT/build.json")
P12=${PIN:0:12}
HELD=$OUT/q18a-attn-meta-bundle-$P12
RUNNER=$OUT/q18a-attn-meta-owner-run-$P12.py
[[ $(sha256sum "$HELD/SHA256SUMS" | cut -d' ' -f1) == "$PIN" ]] || { echo 'continuation refused: the bundle moved' >>"$STEPS"; exit 3; }
ENGINE=$OUT/engine-selfrun
grep -qE ' engine-selfrun-2 rc=0$' "$STEPS" && ENGINE=$OUT/engine-selfrun-2
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
  python3 -B tools/attn_meta_selfrun5.py --mode "$MODE" --cap 1800 --name "attn-meta-$MODE-cpu-$P12" --out "$OUT/attn-$MODE" \
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
