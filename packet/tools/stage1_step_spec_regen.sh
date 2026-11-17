#!/usr/bin/env bash
# Regenerate and commit the stage1-step packaging spec in one command (c1b's tool by name) (CPU only; no box/VM/GPU action, no push).
#   usage: tools/stage1_step_spec_regen.sh [--pre-queue]
# Full mode (the queue harness has landed): regenerate tools/stage1_step_cpu_probe.py (tools/stage1_step_tools_derive.py, which reads
# the queue suite's count) and commit it if it moved, then write tools/native43_spec.stage1-step.json (tools/stage1_step_spec.py over a
# fresh scratch overlay regeneration; source_commit = HEAD) and commit it. The probe goes first because the spec's source_paths and
# tools are read from git, so the probe must already be committed for the spec to declare it.
# --pre-queue: before the queue harness lands; no probe, the spec's queue-derived fields are null and it cannot build.
# harness/ and patches/ must be committed first (the spec pins HEAD); each commit runs the secret scan on its staged diff.
set -eu
cd "$(dirname "$0")/.."
# The parent must be pinned (tools/stage1_step_parent.json; rc 3 'parent not pinned' otherwise). A development pin
# (STAGE1_STEP_PARENT) is refused here: a committed spec is always over the pinned parent.
[[ -z ${STAGE1_STEP_PARENT:-} ]] || { echo 'STAGE1_STEP_PARENT is set: a committed spec needs the pinned parent' >&2; exit 3; }
python3 -B tools/stage1_step_parent_pin.py --pinned
SCAN=<hub-dir>/agents/flash-next/tmp/alpha-next/scripts/secret_scan.sh
PACK=<hub-dir>/agents/flash-next/tmp/stage1-step/pack
MODE=${1:-}
[[ -z $MODE || $MODE == --pre-queue ]] || { echo "usage: $0 [--pre-queue]" >&2; exit 2; }
[[ -z $(git status --porcelain -- harness patches tools) ]] || { echo 'commit harness/, patches/ and tools/ first' >&2; exit 3; }
commit() {  # commit FILE MESSAGE: only if FILE moved; secret scan on the staged diff first
  git add "$1"
  if git diff --cached --quiet -- "$1"; then echo "unchanged: $1"; return 0; fi
  bash "$SCAN"
  git commit -q -m "$2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "$1"
  echo "committed $1 at $(git rev-parse --short=12 HEAD)"
}
if [[ -z $MODE ]]; then
  python3 -B tools/stage1_step_tools_derive.py
  commit tools/stage1_step_cpu_probe.py "stage1-step pack: CPU probe regenerated (tools/stage1_step_tools_derive.py)"
fi
SCRATCH=$PACK/spec-scratch-$(git rev-parse --short=12 HEAD)-$$
python3 -B tools/stage1_step_spec.py --scratch "$SCRATCH" $MODE
commit tools/native43_spec.stage1-step.json "stage1-step pack: packaging spec regenerated${MODE:+ ($MODE)} over $(git rev-parse --short=12 HEAD)"
python3 -I -c "import json,sys;d=json.load(open(sys.argv[1]));print('spec', d['source_commit'][:12], 'expected_tests', d['expected_tests'], 'engine_changed', len(d['engine_changed']), 'source_paths', len(d['source_paths']))" tools/native43_spec.stage1-step.json
