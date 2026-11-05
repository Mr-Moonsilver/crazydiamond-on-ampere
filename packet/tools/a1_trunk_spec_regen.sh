#!/usr/bin/env bash
# Regenerate and commit the a1-trunk packaging spec in one command (inwait's tool by name; CPU only; no box/VM/GPU action, no push). The probe goes first: the spec's source_paths and tools are read from git.
set -eu
cd "$(dirname "$0")/.."
[[ -z ${INWAIT_PARENT:-} ]] || { echo 'INWAIT_PARENT is set: a committed spec needs the pinned parent' >&2; exit 3; }
python3 -B tools/inwait_parent_pin.py --pinned
SCAN=<hub-dir>/agents/flash-next/tmp/alpha-next/scripts/secret_scan.sh
PACK=<hub-dir>/agents/flash-next/tmp/a1-trunk/pack
[[ -z $(git status --porcelain -- harness patches tools) ]] || { echo 'commit harness/, patches/ and tools/ first' >&2; exit 3; }
commit() {
  git add "$1"
  if git diff --cached --quiet -- "$1"; then echo "unchanged: $1"; return 0; fi
  bash "$SCAN"
  git commit -q -m "$2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "$1"
  echo "committed $1 at $(git rev-parse --short=12 HEAD)"
}
python3 -B tools/a1_trunk_tools_derive.py
commit tools/a1_trunk_cpu_probe.py "a1-trunk pack: CPU probe regenerated (tools/a1_trunk_tools_derive.py)"
mkdir -p "$PACK"
SCRATCH=$PACK/spec-scratch-$(git rev-parse --short=12 HEAD)-$$
python3 -B tools/a1_trunk_spec.py --scratch "$SCRATCH"
commit tools/native43_spec.a1-trunk.json "a1-trunk pack: packaging spec regenerated over $(git rev-parse --short=12 HEAD)"
python3 -I -c "import json,sys;d=json.load(open(sys.argv[1]));print('spec', d['source_commit'][:12], 'expected_tests', d['expected_tests'], 'engine_changed', len(d['engine_changed']), 'source_paths', len(d['source_paths']))" tools/native43_spec.a1-trunk.json
