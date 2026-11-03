#!/usr/bin/env bash
"$(dirname "$(readlink -f "$0")")/require-dryrun.sh" "$(dirname "$(dirname "$(readlink -f "$0")")")" "<hub-dir>/agents/flash-next/tmp/stage1-step/evidence/gates-${1:?usage: stage1_step_launch.sh TAG [NOTE]}/dryrun.json" || exit 3
# Launch tools/stage1_step_gates.sh, detached (setsid, no terminal, nice 10): it survives this shell; its end appends one line 'chain-<tag> rc=<rc> <date>' to the builder's
# file. The dry run this refuses without (docs/hub-slot.md; a new dry run gets a new tag):
#   WORKERS=4 LANES=$WT/tools/strict_lanes3.alpha-next-c1b.json tools/stage1_step_gates.sh --dry-run WT SPEC OUT EVID TAG NOTE       with the paths below; it writes EVID/dryrun.log and EVID/dryrun.json
# The worker count and the lanes are set here, at launch, never in the chain.
# After the chain's rc 5 at the negative observation, the pins reviewed and committed, tools/stage1_step_launch_cont.sh is the continuation. No box/VM/GPU action, no push.
set -u
TAG=$1 NOTE=${2:-stage1-step}
WT=$(dirname "$(dirname "$(readlink -f "$0")")")
BASE=<hub-dir>/agents/flash-next/tmp/stage1-step
EVID=$BASE/evidence/gates-$TAG OUT=$BASE/build-$TAG SPEC=$WT/tools/native43_spec.stage1-step.json
RESULT=$BASE/builder-out-chain.txt
N=$((10 - $(nice))); (( N >= 0 )) || N=0
WORKERS=${WORKERS:-4} LANES=${LANES:-$WT/tools/strict_lanes3.alpha-next-c1b.json} setsid nohup nice -n "$N" bash -c '"$1/tools/stage1_step_gates.sh" "$1" "$2" "$3" "$4" "$5" "$6"; echo "chain-$5 rc=$? $(date)" >>"$7"' launch \
    "$WT" "$SPEC" "$OUT" "$EVID" "$TAG" "$NOTE" "$RESULT" >"$EVID/chain-$TAG.out" 2>&1 </dev/null &
echo "launched chain-$TAG pid $! steps $EVID/gates-$TAG.steps result $RESULT"
