#!/usr/bin/env python3
"""The alpha-next-c1 held bundle, freeze proofs and owner runner: build_native43.py, unchanged, with alpha-next's pre-VM build as parent.

build_native43's text is never edited (a past queue ran it). Declared configuration, as module constants, as tools/build_alpha_next.py
does: the parent is read from alpha-next's own build.json, named and pinned in tools/alpha_next_c1_parent.json (written by
tools/alpha_next_c1_parent_pin.py; while that file is the placeholder this tool exits 3, 'parent not pinned'). Candidate name
q18a-alpha-next-c1 is admitted by a load_spec with the wider name rule; every other assertion, the source freeze, the overlay
regeneration, its negative controls and the runner normalisation are build_native43's.

One added overlay module (e3/stage_overlap.py, C1, FLASH_S5_STAGE_OVERLAP, default off), so build_alpha_c1's three declared
wrappers: overlay_level (build_native43's rules plus exactly that file), expected_staging (its row under native30_overlay) and stage
(the row reported as added by the regeneration, absent when staged).

Bundled tools replaced: none. build_native43.stage checks every bundled *.py tool against the committed one, and none differs from
alpha-next's bundle (TOOLS_REPLACED stays empty; the undeclared-replaced-tool negative control below still proves a differing one is
refused). The merge of claude/c1 moved one bundled non-python file, tools/native43_spec.alpha-c0.json (C0 l's final spec); stage does
not check or refresh json tools, so the bundle keeps alpha-next's copy, as every parent build did; nothing in the runner reads it.

Added after that build, in the same OUT:
  parent/   the alpha-next reference: a copy of the parent bundle whose e3_builder sources are asserted equal to alpha-next's build
            source (the review pin aside), its overlay regenerated from BASE and asserted byte-identical to the parent's. --negative
            runs over this.
  alpha-next-c1-freeze.json   tools/alpha_next_c1_freeze.py: C1 off is alpha-next (stage_default_off_c1's projection re-pointed),
            both off reaches alpha-next's own freeze, negative controls, and the overlay against make_s5_overlap.package(source).
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
import alpha_next_c1_parent_pin as parent_pin  # noqa: E402

PINS = parent_pin.load()
import build_native43 as b  # noqa: E402

PARENT_BUILD = json.loads(Path(PINS['build']).read_text())
assert b.sha(PINS['build']) == PINS['build_sha256'], 'alpha-next build record changed'
assert PARENT_BUILD['schema'] == 'native43-build/1' and PARENT_BUILD['name'] == 'q18a-alpha-next'
assert PARENT_BUILD['manifest'] == PINS['manifest'] and PARENT_BUILD['source'] == PINS['source']
b.PARENT = Path(PARENT_BUILD['bundle'])
b.HELD = b.PARENT.parent
b.PARENT_MANIFEST = PARENT_BUILD['manifest']
b.PARENT_OVERLAY_MANIFEST = PARENT_BUILD['overlay_manifest']
b.PARENT_REVIEW = PARENT_BUILD['review_sha256']
b.PARENT_BUNDLE_SOURCE = PARENT_BUILD['source']
b.PARENT_COMMIT = PARENT_BUILD['source']
b.PARENT_CANDIDATE = 'q18a-alpha-next'
b.PARENT_STEM = 'q18a_alpha_next'
b.PARENT_RUNNER = Path(PARENT_BUILD['runner'])
b.PARENT_RUNNER_SHA = PARENT_BUILD['runner_sha256']
b.PARENT_TESTS = PARENT_BUILD['expected_selfrun_tests']
ENGINE_COMMIT = PARENT_BUILD['source']
NEW_OVERLAY = ('e3/stage_overlap.py',)
CANDIDATE = 'q18a-alpha-next-c1'


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    assert spec.get('schema') == 'native43-packaging-spec/1', 'spec schema'
    assert set(spec) == b.SPEC_KEYS, sorted(set(spec) ^ b.SPEC_KEYS)
    assert spec['candidate'] == CANDIDATE and spec['review'] == 'prepare' and spec['dry_run'] is False
    assert re.fullmatch(r'[0-9a-f]{40}', spec['source_commit'])
    return spec


b.load_spec = load_spec


TOOLS_REPLACED = {}
b.TOOLS_REPLACED = TOOLS_REPLACED
_expected_staging = b.expected_staging
_stage = b.stage


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


def stage(spec, bundle):
    added, replaced = _stage(spec, bundle)
    rows = ['native30_overlay/' + name for name in NEW_OVERLAY]
    assert not any((bundle / row).exists() for row in rows), 'an added overlay file is present before the regeneration'
    return sorted(added + rows), replaced


b.overlay_level = overlay_level
b.expected_staging = expected_staging
b.stage = stage


def reference(out):
    """The alpha-next reference bundle: parent bytes, engine sources asserted alpha-next's, overlay regenerated and byte-identical."""
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
        assert target.read_bytes() == data, 'parent engine source differs from alpha-next: ' + name
    closure, _ = b.regenerate_overlay(work, bundle)
    assert b.sha(bundle / 'native30_overlay/SHA256SUMS') == b.PARENT_OVERLAY_MANIFEST, 'regenerated reference differs from the parent overlay'
    assert b.verify(bundle / 'native30_overlay') == b.verify(b.PARENT / 'native30_overlay')
    for path in [bundle, *bundle.rglob('*')]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    return dict(path=str(bundle), engine_commit=b.git('rev-parse', ENGINE_COMMIT).strip(), engine_files=len(names),
                overlay_manifest=b.PARENT_OVERLAY_MANIFEST, overlay_byte_identical_to_parent=True,
                review_sha256=closure['review_sha256'])


def replaced_tool_control(out):
    """A parent bundle whose one bundled python tool differs from the committed one must be refused by stage (build_native43's rule)."""
    held = out / 'negative-replaced-tool-parent'
    shutil.copytree(b.PARENT, held)
    for path in [held, *held.rglob('*')]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    victim = sorted(p for p in (held / 'native35_tools').glob('*.py') if (TOOLS / p.name).is_file())[0]
    with victim.open('a') as handle:
        handle.write('\n# undeclared\n')
    saved, b.PARENT = b.PARENT, held
    try:
        return b.refused(lambda: _stage(dict(tools=[]), out / 'negative-replaced-tool'))
    finally:
        b.PARENT = saved


def build(spec_path, out, strict_log, focused_log):
    import alpha_next_c1_freeze as freeze
    result = b.build(spec_path, out, strict_log, focused_log)
    source = freeze.source_level(out / 'default-off-c1-work')
    ref = reference(out)
    controls = dict(freeze.negative_controls(), **{'undeclared-replaced-tool': replaced_tool_control(out)})
    proof = dict(schema='alpha-next-c1-freeze/1', verdict='PASS', source=source, negative_controls=controls,
                 tools_replaced=dict(declared=[], replaced=sorted(TOOLS_REPLACED)),
                 overlay=freeze.overlay_level(out / 'parent/native30_overlay', Path(result['overlay'])), reference=ref,
                 candidate_manifest=result['manifest'], candidate_overlay_manifest=result['overlay_manifest'],
                 parent_manifest=b.PARENT_MANIFEST, parent_pins_sha256=b.sha(parent_pin.TARGET),
                 dev_parent=bool(PINS.get('dev')), gpu_execution=False, box_action=False)
    b.dump(out / 'alpha-next-c1-freeze.json', proof)
    print('ALPHA_NEXT_C1_FREEZE PASS ' + b.sha(out / 'alpha-next-c1-freeze.json'))
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
