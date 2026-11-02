#!/usr/bin/env python3
"""The C1 held bundle, freeze proofs and owner runner: build_native43.py, unchanged, with C0 l's delivered build as parent.

build_native43's text is never edited (a past queue ran it). Declared configuration, as module constants, as
tools/build_alpha_c0.py does: parent = C0 l (tmp/c0c1/pack/build-l, bundle a1526e066c6a, runner 2f64e205, build commit cf0e301,
source commit f37ee41, 342 self-run tests), whose engine is cf0e301's patches/s5_engine byte for byte. Candidate name
q18a-alpha-c1 is admitted by a load_spec with the wider name rule; every other assertion, the source freeze, the overlay
regeneration, its negative controls and the runner normalisation are build_native43's.

Three declared wrappers, as build_alpha_c0's, because C1 adds one overlay file on the parent's set (e3/stage_overlap.py, a
default-off module):
  overlay_level    build_native43's rules on the files the parent has, with exactly that file added and nothing else;
  expected_staging build_native43's rows plus the same file under native30_overlay;
  stage            build_native43's staging, which also reports that overlay row as a row the overlay regeneration adds. It
                   asserts it is absent when staged; the final bundle comparison (the same expected set) asserts it appears.

Added after that build, in the same OUT:
  parent/         the C0 l reference: a copy of the parent bundle whose e3_builder sources are asserted equal to cf0e301's, its
                  overlay regenerated from BASE and asserted byte-identical to the parent's. --negative runs over this.
  alpha-c1-freeze.json   tools/alpha_c1_freeze.py: default-off projections against C0 l (and, through C0's tool, c3f1f64),
                  negative controls, and the overlay against make_s5_overlap.package(source).
No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_native43 as b  # noqa: E402

HELD = Path('<hub-dir>/agents/flash-next/tmp/c0c1/pack/build-l')
SOURCE = 'cf0e3016825e4a96d8a09166fdd982cd7429f976'      # C0 l's build (review) commit = the bundle's provenance source
b.HELD = HELD
b.PARENT = HELD / 'q18a-alpha-c0-bundle-a1526e066c6a'
b.PARENT_MANIFEST = 'a1526e066c6a4ec97d26aafa3560521c7643c3c83897d80e28fd70ab4977d7cd'
b.PARENT_OVERLAY_MANIFEST = '441d185072b36a5782d3999da978f62b0ee2a51df2aa56ef3e8b2630f91746c5'
b.PARENT_REVIEW = '756de51fe4e09e50009fd5c8ef22ffdbe65c3f3a84b189c9678b1d2c7e332430'
b.PARENT_BUNDLE_SOURCE = SOURCE
b.PARENT_COMMIT = SOURCE
b.PARENT_CANDIDATE = 'q18a-alpha-c0'
b.PARENT_STEM = 'q18a_alpha_c0'
b.PARENT_RUNNER = HELD / 'q18a-alpha-c0-owner-run-a1526e066c6a.py'
b.PARENT_RUNNER_SHA = '2f64e205f4d1dd73d6f9b7dfa2e67fe6730aacfb8f9310bc91a75818bfb20040'
b.PARENT_TESTS = 342
ENGINE_COMMIT = SOURCE


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    assert spec.get('schema') == 'native43-packaging-spec/1', 'spec schema'
    assert set(spec) == b.SPEC_KEYS, sorted(set(spec) ^ b.SPEC_KEYS)
    assert spec['candidate'] == 'q18a-alpha-c1' and spec['review'] == 'prepare' and spec['dry_run'] is False
    assert re.fullmatch(r'[0-9a-f]{40}', spec['source_commit'])
    return spec


b.load_spec = load_spec
NEW_OVERLAY = ('e3/stage_overlap.py',)
_expected_staging = b.expected_staging


def overlay_level(spec, overlay, parent_rows):
    """build_native43's overlay rule plus exactly the declared added module; ``prove`` runs on the changed ones only."""
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


_stage = b.stage


def stage(spec, bundle):
    added, replaced = _stage(spec, bundle)
    rows = ['native30_overlay/' + name for name in NEW_OVERLAY]
    assert not any((bundle / row).exists() for row in rows), 'an added overlay file is present before the regeneration'
    return sorted(added + rows), replaced


b.overlay_level = overlay_level
b.expected_staging = expected_staging
b.stage = stage


def reference(out):
    """The C0 l reference bundle: parent bytes, engine sources asserted cf0e301's, overlay regenerated and byte-identical."""
    work = out / 'parent-work'
    work.mkdir()
    bundle = out / 'parent'
    shutil.copytree(b.PARENT, bundle)
    for path in [bundle, *bundle.rglob('*')]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    names = b.git('ls-tree', '-r', '--name-only', ENGINE_COMMIT, 'patches/s5_engine').splitlines()
    for name in names:
        data = subprocess.check_output(['git', 'show', ENGINE_COMMIT + ':' + name], cwd=b.ROOT)
        target = bundle / 'e3_builder' / Path(name).relative_to('patches/s5_engine')
        assert target.read_bytes() == data, 'parent engine source differs from C0 l\'s build commit: ' + name
    closure, _ = b.regenerate_overlay(work, bundle)
    assert b.sha(bundle / 'native30_overlay/SHA256SUMS') == b.PARENT_OVERLAY_MANIFEST, 'regenerated reference differs from the parent overlay'
    assert b.verify(bundle / 'native30_overlay') == b.verify(b.PARENT / 'native30_overlay')
    for path in [bundle, *bundle.rglob('*')]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    return dict(path=str(bundle), engine_commit=b.git('rev-parse', ENGINE_COMMIT).strip(), engine_files=len(names),
                overlay_manifest=b.PARENT_OVERLAY_MANIFEST, overlay_byte_identical_to_parent=True,
                review_sha256=closure['review_sha256'])


def build(spec_path, out, strict_log, focused_log):
    import alpha_c1_freeze as freeze
    result = b.build(spec_path, out, strict_log, focused_log)
    source = freeze.source_level(out / 'default-off-c1-work')
    ref = reference(out)
    proof = dict(schema='alpha-c1-freeze/1', verdict='PASS', source=source, negative_controls=freeze.negative_controls(),
                 overlay=freeze.overlay_level(out / 'parent/native30_overlay', Path(result['overlay'])), reference=ref,
                 candidate_manifest=result['manifest'], candidate_overlay_manifest=result['overlay_manifest'],
                 parent_manifest=b.PARENT_MANIFEST, gpu_execution=False, box_action=False)
    b.dump(out / 'alpha-c1-freeze.json', proof)
    print('ALPHA_C1_FREEZE PASS ' + b.sha(out / 'alpha-c1-freeze.json'))
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
