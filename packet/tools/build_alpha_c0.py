#!/usr/bin/env python3
"""The C0 held bundle, freeze proofs and owner runner: build_native43.py, unchanged, with alpha-perf build-d as parent.

build_native43's text is never edited (a past queue ran it). Declared configuration, as module constants, as
tools/build_alpha_perf.py does: parent = alpha-perf build-d (bundle d6b0f69ead0b, runner b168cec3, source 435dba6), whose engine is
c3f1f64's byte for byte. Candidate name q18a-alpha-c0 is admitted by a load_spec with the wider name rule; every other assertion, the
source freeze, the overlay regeneration, its negative controls and the runner normalisation are build_native43's.

Two declared wrappers, because this is the first candidate to add overlay files on an unchanged parent set (e3/stage_wire.py and
e3/stage_progress.py, both default-off modules):
  overlay_level    build_native43's rules on the files the parent has, with exactly those two added and nothing else;
  expected_staging build_native43's rows plus the same two files under native30_overlay.

Added after that build, in the same OUT:
  parent/         the c3f1f64 reference: a copy of the parent bundle whose e3_builder sources are asserted equal to c3f1f64's, its
                  overlay regenerated from BASE and asserted byte-identical to the parent's. --negative runs over this.
  alpha-c0-freeze.json   tools/alpha_c0_freeze.py: default-off projections and negative controls against c3f1f64, and the
                  overlay against make_s5_overlap.package(source).
No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_native43 as b  # noqa: E402

HELD = Path('<hub-dir>/agents/flash-next/tmp/alpha-perf/build-d')
SOURCE = '435dba6e475e890cedd79f7b65c04369e62fc6ac'
b.HELD = HELD
b.PARENT = HELD / 'q18a-alpha-perf-bundle-d6b0f69ead0b'
b.PARENT_MANIFEST = 'd6b0f69ead0b7d602f5caf879d938ec111162fb3d38fb79e3960b0a6846465f7'
b.PARENT_OVERLAY_MANIFEST = 'cf5acf65164704b866c7fe79b1f0af32951a2b98ce4e9a11e9ae56b6c20eb4fa'
b.PARENT_REVIEW = 'b1c42d08dbb385d9a2d3869eb1a33231b18496c82575ec6704f1acd571966643'
b.PARENT_BUNDLE_SOURCE = SOURCE
b.PARENT_COMMIT = SOURCE
b.PARENT_CANDIDATE = 'q18a-alpha-perf'
b.PARENT_STEM = 'q18a_alpha_perf'
b.PARENT_RUNNER = HELD / 'q18a-alpha-perf-owner-run-d6b0f69ead0b.py'
b.PARENT_RUNNER_SHA = 'b168cec3668159775484f9d369ccae502f11761b737a5187f19d65877c70c282'
b.PARENT_TESTS = 287
ENGINE_COMMIT = 'c3f1f64'


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    assert spec.get('schema') == 'native43-packaging-spec/1', 'spec schema'
    assert set(spec) == b.SPEC_KEYS, sorted(set(spec) ^ b.SPEC_KEYS)
    assert spec['candidate'] == 'q18a-alpha-c0' and spec['review'] == 'prepare' and spec['dry_run'] is False
    assert re.fullmatch(r'[0-9a-f]{40}', spec['source_commit'])
    return spec


b.load_spec = load_spec
NEW_OVERLAY = ('e3/stage_progress.py', 'e3/stage_wire.py')
_expected_staging = b.expected_staging


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


b.overlay_level = overlay_level
b.expected_staging = expected_staging


def reference(out):
    """The c3f1f64 reference bundle: parent bytes, engine sources asserted c3f1f64's, overlay regenerated and byte-identical."""
    work = out / 'parent-work'
    work.mkdir()
    bundle = out / 'parent'
    shutil.copytree(b.PARENT, bundle)
    for path in [bundle, *bundle.rglob('*')]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    names = [n for n in b.git('ls-tree', '-r', '--name-only', ENGINE_COMMIT, 'patches/s5_engine').splitlines()
             if n != b.REVIEW_PATH]
    for name in names:
        import subprocess
        data = subprocess.check_output(['git', 'show', ENGINE_COMMIT + ':' + name], cwd=b.ROOT)
        target = bundle / 'e3_builder' / Path(name).relative_to('patches/s5_engine')
        assert target.read_bytes() == data, 'parent engine source differs from c3f1f64: ' + name
    closure, _ = b.regenerate_overlay(work, bundle)
    assert b.sha(bundle / 'native30_overlay/SHA256SUMS') == b.PARENT_OVERLAY_MANIFEST, 'regenerated reference differs from the parent overlay'
    assert b.verify(bundle / 'native30_overlay') == b.verify(b.PARENT / 'native30_overlay')
    for path in [bundle, *bundle.rglob('*')]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    return dict(path=str(bundle), engine_commit=b.git('rev-parse', ENGINE_COMMIT).strip(), engine_files=len(names),
                overlay_manifest=b.PARENT_OVERLAY_MANIFEST, overlay_byte_identical_to_parent=True,
                review_sha256=closure['review_sha256'])


def build(spec_path, out, strict_log, focused_log):
    import alpha_c0_freeze as freeze
    source = freeze.source_level()
    result = b.build(spec_path, out, strict_log, focused_log)
    ref = reference(out)
    proof = dict(schema='alpha-c0-freeze/1', verdict='PASS', source=source, negative_controls=freeze.negative_controls(),
                 overlay=freeze.overlay_level(out / 'parent/native30_overlay', Path(result['overlay'])), reference=ref,
                 candidate_manifest=result['manifest'], candidate_overlay_manifest=result['overlay_manifest'],
                 parent_manifest=b.PARENT_MANIFEST, gpu_execution=False, box_action=False)
    b.dump(out / 'alpha-c0-freeze.json', proof)
    print('ALPHA_C0_FREEZE PASS ' + b.sha(out / 'alpha-c0-freeze.json'))
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
