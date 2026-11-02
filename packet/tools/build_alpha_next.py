#!/usr/bin/env python3
"""The alpha-next held bundle, freeze proofs and owner runner: build_native43.py, unchanged, with Step B's frozen build as parent.

build_native43's text is never edited (a past queue ran it). Declared configuration, as module constants, as tools/build_step_b.py
does: the parent is read from Step B's own build.json (chain e, bundle 4b1eccb869db), named and pinned in tools/alpha_next_parent.json
(checked with alpha_next_parent_pin.py --check). Candidate name q18a-alpha-next is admitted by a load_spec with the wider name rule;
every other assertion, the source freeze, the overlay regeneration, its negative controls and the runner normalisation are
build_native43's.

Three declared wrappers, as tools/build_alpha_c0.py (C0 was the first candidate to add overlay files), because alpha-next adds
three default-off modules to Step B's overlay set: e3/stage_progress.py and e3/stage_wire.py (C0, FLASH_S5_STAGE_LOCAL) and
e3/handler_anatomy.py (stage-1 anatomy, FLASH_S5_HANDLER_ANATOMY):
  overlay_level    build_native43's rules on the files the parent has, with exactly those three added and nothing else;
  expected_staging build_native43's rows plus the same three files under native30_overlay;
  stage            build_native43's staging, reporting those three overlay rows as rows the regeneration adds (absent when staged).

Added after that build, in the same OUT:
  parent/         the Step B reference: a copy of the parent bundle whose e3_builder sources are asserted equal to Step B's build
                  source, its overlay regenerated from BASE and asserted byte-identical to the parent's. --negative runs over this.
  alpha-next-freeze.json   tools/alpha_next_freeze.py: the composition proof, the feature freeze records, the default-off
                  projection of this build's own engine changes, negative controls, and the overlay against
                  make_s5_overlap.package(source).
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

PARENT_FILE = TOOLS / 'alpha_next_parent.json'
PINS = json.loads(PARENT_FILE.read_text())
STEP_B = json.loads(Path(PINS['build']).read_text())
assert b.sha(PINS['build']) == PINS['build_sha256'], 'Step B build record changed'
assert STEP_B['schema'] == 'native43-build/1' and STEP_B['name'] == 'q18a-step-b' and STEP_B['manifest'] == PINS['manifest']
b.PARENT = Path(STEP_B['bundle'])
b.HELD = b.PARENT.parent
b.PARENT_MANIFEST = STEP_B['manifest']
b.PARENT_OVERLAY_MANIFEST = STEP_B['overlay_manifest']
b.PARENT_REVIEW = STEP_B['review_sha256']
b.PARENT_BUNDLE_SOURCE = STEP_B['source']
b.PARENT_COMMIT = STEP_B['source']
b.PARENT_CANDIDATE = 'q18a-step-b'
b.PARENT_STEM = 'q18a_step_b'
b.PARENT_RUNNER = Path(STEP_B['runner'])
b.PARENT_RUNNER_SHA = STEP_B['runner_sha256']
b.PARENT_TESTS = STEP_B['expected_selfrun_tests']
ENGINE_COMMIT = STEP_B['source']
NEW_OVERLAY = ('e3/handler_anatomy.py', 'e3/stage_progress.py', 'e3/stage_wire.py')


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    assert spec.get('schema') == 'native43-packaging-spec/1', 'spec schema'
    assert set(spec) == b.SPEC_KEYS, sorted(set(spec) ^ b.SPEC_KEYS)
    assert spec['candidate'] == 'q18a-alpha-next' and spec['review'] == 'prepare' and spec['dry_run'] is False
    assert re.fullmatch(r'[0-9a-f]{40}', spec['source_commit'])
    return spec


b.load_spec = load_spec
_expected_staging = b.expected_staging
_stage = b.stage


def overlay_level(spec, overlay, parent_rows):
    """build_native43's overlay rule plus exactly the declared added modules; ``prove`` runs on the changed ones only."""
    rows = b.verify(overlay)
    added = sorted(set(rows) - set(parent_rows))
    assert added == sorted(NEW_OVERLAY), ('overlay files added', added)
    assert not set(parent_rows) - set(rows), 'overlay file removed'
    moved = sorted(n for n in parent_rows if rows[n] != parent_rows[n])
    engine = sorted(n for n in moved if n not in b.METADATA)
    assert engine == sorted(spec['engine_changed']), ('undeclared overlay engine change', engine, spec['engine_changed'])
    assert not any(n.endswith('.so') for n in moved), 'compiled library moved'
    assert set(spec['overlay_definitions']) == set(engine), sorted(set(spec['overlay_definitions']) ^ set(engine))
    definitions = {path: b.prove((b.PARENT / 'native30_overlay' / path).read_text(), (overlay / path).read_text(), path,
                                 spec['overlay_definitions'][path]) for path in engine}
    return dict(parent_overlay=str(b.PARENT / 'native30_overlay'), parent_overlay_manifest=b.PARENT_OVERLAY_MANIFEST,
                moved=moved, added=added, engine_changed=engine, metadata_moved=sorted(set(moved) - set(engine)),
                definitions=definitions, compiled_libraries_byte_identical=True, byte_identical_to_parent=False)


def expected_staging(spec):
    added, replaced = _expected_staging(spec)
    return added | {'native30_overlay/' + name for name in NEW_OVERLAY}, replaced


def stage(spec, bundle):
    added, replaced = _stage(spec, bundle)
    rows = ['native30_overlay/' + name for name in NEW_OVERLAY]
    assert not any((bundle / row).exists() for row in rows), 'an added overlay file is present before the regeneration'
    return sorted(added + rows), replaced


b.overlay_level = overlay_level
b.expected_staging = expected_staging
b.stage = stage


def reference(out):
    """The Step B reference bundle: parent bytes, engine sources asserted Step B's, overlay regenerated and byte-identical."""
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
        assert target.read_bytes() == data, 'parent engine source differs from Step B: ' + name
    closure, _ = b.regenerate_overlay(work, bundle)
    assert b.sha(bundle / 'native30_overlay/SHA256SUMS') == b.PARENT_OVERLAY_MANIFEST, 'regenerated reference differs from the parent overlay'
    assert b.verify(bundle / 'native30_overlay') == b.verify(b.PARENT / 'native30_overlay')
    for path in [bundle, *bundle.rglob('*')]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    return dict(path=str(bundle), engine_commit=b.git('rev-parse', ENGINE_COMMIT).strip(), engine_files=len(names),
                overlay_manifest=b.PARENT_OVERLAY_MANIFEST, overlay_byte_identical_to_parent=True,
                review_sha256=closure['review_sha256'])


def build(spec_path, out, strict_log, focused_log):
    import alpha_next_freeze as freeze
    source = freeze.source_level()
    result = b.build(spec_path, out, strict_log, focused_log)
    ref = reference(out)
    proof = dict(schema='alpha-next-freeze/1', verdict='PASS', source=source, negative_controls=freeze.negative_controls(),
                 overlay=freeze.overlay_level(out / 'parent/native30_overlay', Path(result['overlay'])), reference=ref,
                 candidate_manifest=result['manifest'], candidate_overlay_manifest=result['overlay_manifest'],
                 parent_manifest=b.PARENT_MANIFEST, gpu_execution=False, box_action=False)
    b.dump(out / 'alpha-next-freeze.json', proof)
    print('ALPHA_NEXT_FREEZE PASS ' + b.sha(out / 'alpha-next-freeze.json'))
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
