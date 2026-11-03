#!/usr/bin/env bash
"$(dirname "$(readlink -f "$0")")/require-dryrun.sh" "$(dirname "$(dirname "$(readlink -f "$0")")")" "<hub-dir>/agents/flash-next/tmp/stage1-step/evidence/gates-${1:?usage: stage1_step_launch_cont.sh TAG [NOTE]}/dryrun-cont.json" || exit 3
# Launch tools/stage1_step_gates_cont.sh, detached, nice 10, as tools/stage1_step_launch.sh; its end appends 'chain-<tag>-cont rc=<rc> <date>'. The dry run:
#   tools/stage1_step_gates_cont.sh --dry-run WT SPEC OUT EVID TAG NOTE    it writes EVID/dryrun-cont.log and EVID/dryrun-cont.json. No box/VM/GPU action, no push.
set -u
TAG=$1 NOTE=${2:-stage1-step}
WT=$(dirname "$(dirname "$(readlink -f "$0")")")
BASE=<hub-dir>/agents/flash-next/tmp/stage1-step
EVID=$BASE/evidence/gates-$TAG OUT=$BASE/build-$TAG SPEC=$WT/tools/native43_spec.stage1-step.json
RESULT=$BASE/builder-out-chain.txt
N=$((10 - $(nice))); (( N >= 0 )) || N=0
setsid nohup nice -n "$N" bash -c '"$1/tools/stage1_step_gates_cont.sh" "$1" "$2" "$3" "$4" "$5" "$6"; echo "chain-$5-cont rc=$? $(date)" >>"$7"' launch \
    "$WT" "$SPEC" "$OUT" "$EVID" "$TAG" "$NOTE" "$RESULT" >"$EVID/chain-$TAG-cont.out" 2>&1 </dev/null &
echo "launched chain-$TAG-cont pid $! steps $EVID/gates-$TAG.steps result $RESULT"
