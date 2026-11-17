#!/usr/bin/env python3
"""The Step B held bundle, freeze proofs and owner runner: build_native43.py, unchanged, with Step A's frozen build as parent.

build_native43's text is never edited (a past queue ran it). Declared configuration, as module constants, as
tools/build_step_a.py does. The parent is read from Step A's own build.json, named and pinned in tools/step_b_parent.json. Candidate
name q18a-step-b is admitted by a load_spec with the wider name rule; every other assertion, the source freeze, the overlay
regeneration, its negative controls and the runner normalisation are build_native43's.

Added after that build, in the same OUT:
  parent/         the Step A reference: a copy of the parent bundle whose e3_builder sources are asserted equal to Step A's source
                  commit, its overlay regenerated from BASE and asserted byte-identical to the parent's. --negative runs over this.
  step-b-freeze.json   tools/step_b_freeze.py: default-off projections, reach and negative controls against claude/step-a, and the
                  overlay against make_s5_overlap.package(source).
No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import build_native43 as b  # noqa: E402

PARENT_FILE = TOOLS / 'step_b_parent.json'
PINS = json.loads(PARENT_FILE.read_text())
STEP_A = json.loads(Path(PINS['build']).read_text())
assert b.sha(PINS['build']) == PINS['build_sha256'], 'Step A build record changed'
assert STEP_A['schema'] == 'native43-build/1' and STEP_A['name'] == 'q18a-step-a' and STEP_A['manifest'] == PINS['manifest']
b.PARENT = Path(STEP_A['bundle'])
b.HELD = b.PARENT.parent
b.PARENT_MANIFEST = STEP_A['manifest']
b.PARENT_OVERLAY_MANIFEST = STEP_A['overlay_manifest']
b.PARENT_REVIEW = STEP_A['review_sha256']
b.PARENT_BUNDLE_SOURCE = STEP_A['source']
b.PARENT_COMMIT = STEP_A['source']
b.PARENT_CANDIDATE = 'q18a-step-a'
b.PARENT_STEM = 'q18a_step_a'
b.PARENT_RUNNER = Path(STEP_A['runner'])
b.PARENT_RUNNER_SHA = STEP_A['runner_sha256']
b.PARENT_TESTS = STEP_A['expected_selfrun_tests']
ENGINE_COMMIT = STEP_A['source']


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    assert spec.get('schema') == 'native43-packaging-spec/1', 'spec schema'
    assert set(spec) == b.SPEC_KEYS, sorted(set(spec) ^ b.SPEC_KEYS)
    assert spec['candidate'] == 'q18a-step-b' and spec['review'] == 'prepare' and spec['dry_run'] is False
    assert re.fullmatch(r'[0-9a-f]{40}', spec['source_commit'])
    return spec


b.load_spec = load_spec


def reference(out):
    """The Step A reference bundle: parent bytes, engine sources asserted Step A's, overlay regenerated and byte-identical."""
    work = out / 'parent-work'
    work.mkdir()
    bundle = out / 'parent'
    shutil.copytree(b.PARENT, bundle)
    for path in [bundle, *bundle.rglob('*')]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    names = [n for n in b.git('ls-tree', '-r', '--name-only', ENGINE_COMMIT, 'patches/s5_engine').splitlines()
             if n != b.REVIEW_PATH]
    for name in names:
        data = subprocess.check_output(['git', 'show', ENGINE_COMMIT + ':' + name], cwd=b.ROOT)
        target = bundle / 'e3_builder' / Path(name).relative_to('patches/s5_engine')
        assert target.read_bytes() == data, 'parent engine source differs from Step A: ' + name
    closure, _ = b.regenerate_overlay(work, bundle)
    assert b.sha(bundle / 'native30_overlay/SHA256SUMS') == b.PARENT_OVERLAY_MANIFEST, 'regenerated reference differs from the parent overlay'
    assert b.verify(bundle / 'native30_overlay') == b.verify(b.PARENT / 'native30_overlay')
    for path in [bundle, *bundle.rglob('*')]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    return dict(path=str(bundle), engine_commit=b.git('rev-parse', ENGINE_COMMIT).strip(), engine_files=len(names),
                overlay_manifest=b.PARENT_OVERLAY_MANIFEST, overlay_byte_identical_to_parent=True,
                review_sha256=closure['review_sha256'])


def build(spec_path, out, strict_log, focused_log):
    import step_b_freeze as freeze
    rows = b.git('diff', '--name-status', freeze.PARENT, ENGINE_COMMIT, '--', 'patches/s5_engine').splitlines()
    assert set(rows) <= {'M\t' + b.REVIEW_PATH}, ('the freeze parent and Step A\'s build differ in engine bytes', rows)
    source = freeze.source_level()
    result = b.build(spec_path, out, strict_log, focused_log)
    ref = reference(out)
    proof = dict(schema='step-b-freeze/1', verdict='PASS', source=source, negative_controls=freeze.negative_controls(),
                 overlay=freeze.overlay_level(out / 'parent/native30_overlay', Path(result['overlay'])), reference=ref,
                 candidate_manifest=result['manifest'], candidate_overlay_manifest=result['overlay_manifest'],
                 parent_manifest=b.PARENT_MANIFEST, gpu_execution=False, box_action=False)
    b.dump(out / 'step-b-freeze.json', proof)
    print('STEP_B_FREEZE PASS ' + b.sha(out / 'step-b-freeze.json'))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--selftest-log', type=Path)
    p.add_argument('--focused-log', type=Path)
    p.add_argument('--prepare-review', action='store_true')
    p.add_argument('--propose', action='store_true')
    a = p.parse_args()
    if a.propose:
        print(json.dumps(b.propose(b.load_spec(a.spec)), indent=2, sort_keys=True))
    elif a.prepare_review:
        b.dump(b.ROOT / b.REVIEW_PATH, b.prepare_review(b.load_spec(a.spec), a.selftest_log, a.focused_log))
    else:
        assert a.out is not None and a.selftest_log is not None and a.focused_log is not None
        build(a.spec, a.out.resolve(), a.selftest_log.resolve(), a.focused_log.resolve())
