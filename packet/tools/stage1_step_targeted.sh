#!/usr/bin/env bash
# stage1_step_targeted.sh: the lead's targeted route after a test-side (HARNESS) stop with unchanged engine bytes: only the failed step and the touched test files, then the prepare-review commit, the bundle rebuild, the self-runs
# and every later step, which are tools/stage1_step_gates.sh's own text (its review_to_stop block, read from the committed file by its markers, nothing copied). Use a full chain when engine bytes move: this refuses them.
#   usage: WORKERS=N LANES=FILE tools/stage1_step_targeted.sh [--dry-run] WT SPEC OUT EVID TAG NOTE PRIOR_EVID PRIOR_TAG -- TOUCHED_FILE...
# The failed step comes from the prior steps file: its last line with an rc must be a non-zero rc of a step this route can rerun (strict, strict-2, focused, focused-2, composition-test); a stop at any later step needs only the block,
# the step reruns inside it. TOUCHED_FILE: repo-relative. Every harness/ or tools/ file moved since the prior chain's head must be named, and only such files; the patches/s5_engine bytes must equal the prior head's except the
# review record. Strict and focused logs that passed in the prior chain are kept as the build's inputs and named in the steps file. The touched test files run through the superseded runner and their log goes to the steps file.
# Refusals rc 3, written to the steps file when it exists. New OUT, new EVID (or a new TAG in the same EVID): nothing of the prior run is touched. --dry-run: the same refusals, no container, EVID/dryrun.log and EVID/dryrun.json.
set -u
unset STAGE1_STEP_PARENT ALPHA_NEXT_C1_PARENT ALPHA_NEXT_C1B_PARENT FLASH_S5_LAZY_RECV FLASH_S5_DECODE_MERGE   # development pins are for tests only; the switch is unset
DRY=; if [[ ${1:-} == --dry-run ]]; then DRY=1; shift; fi
[[ $# -ge 10 && ${9:-} == -- ]] || { echo 'usage: stage1_step_targeted.sh [--dry-run] WT SPEC OUT EVID TAG NOTE PRIOR_EVID PRIOR_TAG -- TOUCHED_FILE...' >&2; exit 2; }
WT=$1 SPEC=$2 OUT=$3 EVID=$4 TAG=$5 NOTE=$6 PRIOR_EVID=$7 PRIOR_TAG=$8
shift 9
TOUCHED=("$@")
GATES=$WT/tools/stage1_step_gates.sh
HUB_SLOT=<hub-dir>/agents/flash-next/fn-main/tools/hub-slot.sh
STEPS=$EVID/gates-$TAG.steps
PRIOR=$PRIOR_EVID/gates-$PRIOR_TAG.steps
mkdir -p "$EVID"
[[ ! -e $STEPS && ! -e $OUT && -f $SPEC && -f $PRIOR && -n ${WORKERS:-} && -f ${LANES:-} ]] || exit 3
[[ $(nice) -ge 10 ]] || renice -n 10 -p $$ >/dev/null
step() { printf '%s %s rc=%s\n' "$(date -u +%H:%M:%SZ)" "$1" "$2" >>"$STEPS"; [[ $2 == 0 ]] || exit "$2"; }
note_rc() { printf '%s %s rc=%s\n' "$(date -u +%H:%M:%SZ)" "$1" "$2" >>"$STEPS"; }
flaky() { local v; v=$(python3 -B "$WT/tools/alpha_c1_flaky.py" "$1" "$2" 2>&1); local rc=$?; printf 'flaky %s %s\n' "$1" "$v" >>"$STEPS"; return $rc; }
block() { sed -n "/^# >>> $1 /,/^# <<< $1\$/p" "$GATES"; }
refuse() { printf 'targeted refused: %s\n' "$1" >>"$STEPS"; echo "targeted refused: $1" >&2; exit 3; }
eval "$(block wait_clear)"
eval "$(block strict)"
cd "$WT" || exit 3
# >>> targeted_refusals (harness/test_stage1_step_gates.py runs this block)
HEAD=$(git rev-parse HEAD)
printf 'head %s spec %s note %s targeted-from %s\n' "$HEAD" "$(sha256sum "$SPEC" | cut -c1-12)" "$NOTE" "$PRIOR_TAG" >"$STEPS"
PRIOR_HEAD=$(sed -n '1s/^head \([0-9a-f]\{40\}\) .*/\1/p' "$PRIOR")
[[ -n $PRIOR_HEAD ]] || refuse 'the prior steps file names no head'
[[ -z $(git status --porcelain -- harness tools patches) ]] || refuse dirty
ENGINE_MOVED=$(git diff --name-only "$PRIOR_HEAD" "$HEAD" -- patches/s5_engine | grep -v '^patches/s5_engine/native_candidate_review.json$' || true)
[[ -z $ENGINE_MOVED ]] || refuse "engine bytes moved, the full chain is the route: $ENGINE_MOVED"
MOVED=$(git diff --name-only "$PRIOR_HEAD" "$HEAD" -- harness tools | grep -vE '^tools/(native43_spec\.stage1-step\.json|stage1_step_cpu_probe\.py|stage1_step_queue_pins\.json|stage1_step_negative_pins\.json)$' || true)
for name in $MOVED; do
  printf '%s\n' "${TOUCHED[@]}" | grep -qxF "$name" || refuse "moved and not named: $name"
done
for name in "${TOUCHED[@]}"; do
  grep -qxF "$name" <<<"$MOVED" || refuse "named and not moved: $name"
done
FAILED=$(grep -E ' rc=[0-9]+$' "$PRIOR" | tail -n 1 | cut -d' ' -f2-)
[[ -n $FAILED && $FAILED != *' rc=0' ]] || refuse 'the prior chain did not stop at a failed step'
FAILED_STEP=${FAILED%% *}
# <<< targeted_refusals
printf 'targeted route: failed step of %s: %s; touched %s\n' "$PRIOR_TAG" "$FAILED" "${TOUCHED[*]}" >>"$STEPS"
BASE_RECEIPT=$PRIOR_EVID/base-proof-$PRIOR_TAG.json
[[ -f $BASE_RECEIPT ]] || refuse 'no prior base-proof receipt'
MODULES=$(printf '%s\n' "${TOUCHED[@]}" | sed -n 's#^harness/\(test_[A-Za-z0-9_]*\)\.py$#\1#p' | tr '\n' ' ')
if [[ -n $DRY ]]; then
  LOG=$EVID/dryrun.log
  [[ ! -e $LOG && ! -e $EVID/dryrun.json ]] || { echo 'dry-run refused: a dry run of this tag exists' >&2; exit 3; }
  printf 'dry-run head %s targeted from %s failed %s touched %s\n' "$HEAD" "$PRIOR_TAG" "$FAILED" "${TOUCHED[*]}" >"$LOG"
  python3 -B tools/stage1_step_review_pins.py --check >>"$LOG" 2>&1; RC=$?
  bash -n "$GATES" || RC=1
  [[ -n $(block review_to_stop) ]] || RC=1
  python3 -B tools/stage1_step_dryrun.py spec "$SPEC" >>"$LOG" 2>&1 || RC=1
  python3 -B tools/stage1_step_dryrun.py image >>"$LOG" 2>&1 || RC=1
  sudo -n docker ps --format '{{.Names}}' >/dev/null 2>&1 || RC=1
  printf 'dry-run rc=%s %s\n' "$RC" "$(date -u +%H:%M:%SZ)" >>"$LOG"
  bash tools/dryrun-record.sh "$WT" "$RC" "$LOG" "$EVID/dryrun.json" >>"$EVID/dryrun-record.out" 2>&1 || RC=2
  cat "$LOG"
  exit "$RC"
fi
python3 -B tools/stage1_step_review_pins.py --check >"$EVID/review-pins-$TAG.log" 2>&1
step review-pins $?
cd "$WT/harness" || exit 3
# The touched test files.
if [[ -n $MODULES ]]; then
  # shellcheck disable=SC2086
  { python3 -B stage1_step_robust_swap.py --base-receipt "$BASE_RECEIPT" $MODULES 2>&1; echo "rc=$?"; } >"$EVID/touched-$TAG.log"
  step touched-tests "$(tail -n 1 "$EVID/touched-$TAG.log" | sed 's/^rc=//')"
fi
# The failed step, when it stands before the block; its logs, or the prior chain's that passed.
STRICT_LOG=$PRIOR_EVID/strict-$PRIOR_TAG.log
[[ -f $PRIOR_EVID/strict-$PRIOR_TAG-2.log ]] && STRICT_LOG=$PRIOR_EVID/strict-$PRIOR_TAG-2.log
FOCUSED_LOG=$PRIOR_EVID/focused-$PRIOR_TAG.log
[[ -f $PRIOR_EVID/focused-$PRIOR_TAG-2.log ]] && FOCUSED_LOG=$PRIOR_EVID/focused-$PRIOR_TAG-2.log
ENGINE_SUITES=$(python3 -I -c "import json,sys;print(' '.join(json.load(open(sys.argv[1]))['engine_suites']))" "$WT/tools/stage1_step_engine_suites.json") || exit 3
eval "$(cd "$WT" && sed -n '/^FOCUSED_MODULES="/,/"$/p' "$GATES")"
case $FAILED_STEP in
  strict|strict-1|strict-2)
    mkdir -p "$(dirname "$WT")/tmp"
    STRICT_LOG=$EVID/strict-$TAG.log
    RC=$(strict "$STRICT_LOG"); step strict "$RC"
    FOCUSED_LOG=$EVID/focused-$TAG.log
    # shellcheck disable=SC2086
    { echo "# head $HEAD focused"; python3 -B stage1_step_robust_swap.py --base-receipt "$BASE_RECEIPT" $FOCUSED_MODULES 2>&1; echo "rc=$?"; } >"$FOCUSED_LOG"
    step focused "$(tail -n 1 "$FOCUSED_LOG" | sed 's/^rc=//')";;
  focused|focused-1|focused-2)
    FOCUSED_LOG=$EVID/focused-$TAG.log
    # shellcheck disable=SC2086
    { echo "# head $HEAD focused"; python3 -B stage1_step_robust_swap.py --base-receipt "$BASE_RECEIPT" $FOCUSED_MODULES 2>&1; echo "rc=$?"; } >"$FOCUSED_LOG"
    step focused "$(tail -n 1 "$FOCUSED_LOG" | sed 's/^rc=//')";;
  composition-test)
    { python3 -B stage1_step_robust_swap.py --base-receipt "$BASE_RECEIPT" test_alpha_next_composition test_alpha_next_c1_pack test_alpha_next_c1b_pack 2>&1; echo "rc=$?"; } >"$EVID/composition-test-$TAG.log"
    step composition-test "$(tail -n 1 "$EVID/composition-test-$TAG.log" | sed 's/^rc=//')";;
  *) printf 'the failed step %s reruns inside the block\n' "$FAILED_STEP" >>"$STEPS";;
esac
printf 'build inputs: strict %s (%s) focused %s (%s)\n' "$STRICT_LOG" "$(sha256sum "$STRICT_LOG" | cut -c1-12)" "$FOCUSED_LOG" "$(sha256sum "$FOCUSED_LOG" | cut -c1-12)" >>"$STEPS"
cd "$WT" || exit 3
eval "$(block review_to_stop)"
