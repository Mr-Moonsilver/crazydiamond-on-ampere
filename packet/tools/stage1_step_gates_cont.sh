#!/usr/bin/env bash
# stage1_step_gates_cont.sh: the continuation of stage1_step_gates.sh after its negative observation stop (rc 5), attn-meta's form by name: the same preamble (wait_clear,
# step), then the chain's text from the engine negative to its stop line, unchanged (harness/test_stage1_step_gates.py compares them). Refuses unless the steps file ends at
# 'engine-negative-awaiting-pins rc=5', the pins are filled and committed (hub-side, so the bundle did not move: the build's manifest is re-checked) and the tree is clean.
# Same usage, same steps file; appended to it. --dry-run: the same refusals, no container, EVID/dryrun-cont.log and EVID/dryrun-cont.json.
set -u
unset STAGE1_STEP_PARENT ALPHA_NEXT_C1_PARENT ALPHA_NEXT_C1B_PARENT FLASH_S5_LAZY_RECV FLASH_S5_DECODE_MERGE   # development pins are for tests only; the switch is unset
DRY=; if [[ ${1:-} == --dry-run ]]; then DRY=1; shift; fi
WT=$1 SPEC=$2 OUT=$3 EVID=$4 TAG=$5 NOTE=$6
HUB_SLOT=<hub-dir>/agents/flash-next/fn-main/tools/hub-slot.sh
STEPS=$EVID/gates-$TAG.steps
mkdir -p "$EVID"
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
# >>> refusals (harness/test_stage1_step_gates.py runs this block)
refuse() { echo "continuation refused: $1" >>"$STEPS"; exit 3; }
[[ -f $STEPS && -f $OUT/build.json ]] || exit 3
[[ $(grep -E ' rc=[0-9]+$' "$STEPS" | tail -n 1 | cut -d' ' -f2-) == 'engine-negative-awaiting-pins rc=5' ]] || refuse 'not at the observation stop'
[[ -z $(git status --porcelain -- harness tools patches) ]] || refuse dirty
python3 -I -c "import json,sys;sys.exit(1 if str(json.load(open(sys.argv[1]))['status']).startswith('AWAITING-OBSERVATION') else 0)" tools/stage1_step_negative_pins.json \
    || refuse 'pins not filled'
PIN=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['manifest'])" "$OUT/build.json")
P12=${PIN:0:12}
HELD=$OUT/q18a-stage1-step-bundle-$P12
RUNNER=$OUT/q18a-stage1-step-owner-run-$P12.py
[[ $(sha256sum "$HELD/SHA256SUMS" | cut -d' ' -f1) == "$PIN" ]] || refuse 'the bundle moved'
# <<< refusals
if [[ -n $DRY ]]; then
  LOG=$EVID/dryrun-cont.log
  [[ ! -e $LOG && ! -e $EVID/dryrun-cont.json ]] || { echo 'dry-run refused: a continuation dry run of this tag exists' >&2; exit 3; }
  printf 'dry-run head %s continuation refusals passed; bundle %s; containers matching the wait now: %s\n' "$(git rev-parse HEAD)" "$P12" \
      "$(sudo -n docker ps --format '{{.Names}}' 2>/dev/null | grep -E 'alpha-next-c1|append-yield|returning-append|attn-meta|stage1-handoff|test-speed-' | tr '\n' ' ')" >"$LOG"
  sudo -n docker ps --format '{{.Names}}' >/dev/null; RC=$?
  printf 'dry-run rc=%s %s\n' "$RC" "$(date -u +%H:%M:%SZ)" >>"$LOG"
  bash tools/dryrun-record.sh "$WT" "$RC" "$LOG" "$EVID/dryrun-cont.json" >>"$EVID/dryrun-record.out" 2>&1 || RC=2
  cat "$LOG"
  exit "$RC"
fi
printf 'continuation head %s\n' "$(git rev-parse HEAD)" >>"$STEPS"
SELFRUN=$OUT/selfrun
ENGINE=$OUT/engine-selfrun
grep -qE ' engine-selfrun-2 rc=0$' "$STEPS" && ENGINE=$OUT/engine-selfrun-2
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
