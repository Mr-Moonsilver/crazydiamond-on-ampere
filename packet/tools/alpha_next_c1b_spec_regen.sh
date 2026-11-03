#!/usr/bin/env bash
# Regenerate and commit the alpha-next-c1b packaging spec in one command (tools/alpha_next_c1_spec_regen.sh for alpha-next-c1b) (CPU only; no box/VM/GPU action, no push).
#   usage: tools/alpha_next_c1b_spec_regen.sh [--pre-queue]
# Full mode (the queue harness has landed): regenerate tools/alpha_next_c1b_cpu_probe.py (tools/alpha_next_c1b_tools_derive.py, which reads
# the queue suite's count) and commit it if it moved, then write tools/native43_spec.alpha-next-c1b.json (tools/alpha_next_c1b_spec.py over a
# fresh scratch overlay regeneration; source_commit = HEAD) and commit it. The probe goes first because the spec's source_paths and
# tools are read from git, so the probe must already be committed for the spec to declare it.
# --pre-queue: before the queue harness lands; no probe, the spec's queue-derived fields are null and it cannot build.
# harness/ and patches/ must be committed first (the spec pins HEAD); each commit runs the secret scan on its staged diff.
set -eu
cd "$(dirname "$0")/.."
# The parent must be pinned (tools/alpha_next_c1b_parent.json; rc 3 'parent not pinned' otherwise). A development pin
# (ALPHA_NEXT_C1B_PARENT) is refused here: a committed spec is always over the pinned parent.
[[ -z ${ALPHA_NEXT_C1B_PARENT:-} ]] || { echo 'ALPHA_NEXT_C1B_PARENT is set: a committed spec needs the pinned parent' >&2; exit 3; }
python3 -B tools/alpha_next_c1b_parent_pin.py --pinned
SCAN=<hub-dir>/agents/flash-next/tmp/alpha-next/scripts/secret_scan.sh
PACK=<hub-dir>/agents/flash-next/tmp/alpha-next-c1b/pack
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
  python3 -B tools/alpha_next_c1b_tools_derive.py
  commit tools/alpha_next_c1b_cpu_probe.py "alpha-next-c1b pack: CPU probe regenerated (tools/alpha_next_c1b_tools_derive.py)"
fi
SCRATCH=$PACK/spec-scratch-$(git rev-parse --short=12 HEAD)-$$
python3 -B tools/alpha_next_c1b_spec.py --scratch "$SCRATCH" $MODE
commit tools/native43_spec.alpha-next-c1b.json "alpha-next-c1b pack: packaging spec regenerated${MODE:+ ($MODE)} over $(git rev-parse --short=12 HEAD)"
python3 -I -c "import json,sys;d=json.load(open(sys.argv[1]));print('spec', d['source_commit'][:12], 'expected_tests', d['expected_tests'], 'engine_changed', len(d['engine_changed']), 'source_paths', len(d['source_paths']))" tools/native43_spec.alpha-next-c1b.json
