#!/usr/bin/env bash
# Regenerate and commit the compose-5 packaging spec in one command (inwait's tool by name; CPU only; no box/VM/GPU action, no push). The probe goes first: the spec's source_paths and tools are read from git.
set -eu
cd "$(dirname "$0")/.."
[[ -z ${INWAIT_PARENT:-} ]] || { echo 'INWAIT_PARENT is set: a committed spec needs the pinned parent' >&2; exit 3; }
python3 -B tools/inwait_parent_pin.py --pinned
SCAN=<hub-dir>/agents/flash-next/tmp/alpha-next/scripts/secret_scan.sh
PACK=<hub-dir>/cargo-target/flash-next/compose-5/pack   # regenerable scratch, sdc by the lead's 5 Nov note
[[ -z $(git status --porcelain -- harness patches tools) ]] || { echo 'commit harness/, patches/ and tools/ first' >&2; exit 3; }
commit() {
  git add "$1"
  if git diff --cached --quiet -- "$1"; then echo "unchanged: $1"; return 0; fi
  bash "$SCAN"
  git commit -q -m "$2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "$1"
  echo "committed $1 at $(git rev-parse --short=12 HEAD)"
}
python3 -B tools/compose5_tools_derive.py
commit tools/compose5_cpu_probe.py "compose-5 pack: CPU probe regenerated (tools/compose5_tools_derive.py)"
mkdir -p "$PACK"
SCRATCH=$PACK/spec-scratch-$(git rev-parse --short=12 HEAD)-$$
python3 -B tools/compose5_spec.py --scratch "$SCRATCH"
commit tools/native43_spec.compose-5.json "compose-5 pack: packaging spec regenerated over $(git rev-parse --short=12 HEAD)"
python3 -I -c "import json,sys;d=json.load(open(sys.argv[1]));print('spec', d['source_commit'][:12], 'expected_tests', d['expected_tests'], 'engine_changed', len(d['engine_changed']), 'source_paths', len(d['source_paths']))" tools/native43_spec.compose-5.json
