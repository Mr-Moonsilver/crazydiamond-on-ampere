#!/usr/bin/env python3
"""The q18a-inwait held bundle, freeze proofs and owner runner: build_native43.py, unchanged, with alpha-next-c1b's READY build as parent.

tools/build_attn_meta_b.py's configuration (claude/attn-meta-c1b; the module constants on build_native43, never its text) re-pointed: the parent
is read from c1b's own build.json (build-b2), named and pinned in tools/inwait_parent.json (tools/inwait_parent_pin.py; a placeholder exits 3,
'parent not pinned'). Candidate name q18a-inwait. The engine moves (declared in the spec's engine_changed and overlay_definitions), one overlay
modules are added (NEW_OVERLAY: attn-meta's two), none removed. Bundled tools replaced: none (TOOLS_REPLACED empty; the
undeclared-replaced-tool control still proves a differing one is refused).

Added after that build, in the same OUT:
  parent/   the c1b reference: a copy of c1b's bundle whose e3_builder sources are asserted equal to c1b's build source (the review pin aside), its
            overlay regenerated and asserted byte-identical to c1b's. --negative runs over this.
  inwait-freeze.json   tools/inwait_freeze.py: the jump off is c1 (the default-off rows), that reaches c1b's own freeze, negative
            controls, and the overlay against make_s5_overlap.package(source).
No GPU, no box action.
"""
import argparse
import ast
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import inwait_parent_pin as parent_pin  # noqa: E402

PINS = parent_pin.load()
import build_native43 as b  # noqa: E402

PARENT_BUILD = json.loads(Path(PINS['build']).read_text())
assert b.sha(PINS['build']) == PINS['build_sha256'], 'alpha-next-c1b build record changed'
assert PARENT_BUILD['schema'] == 'native43-build/1' and PARENT_BUILD['name'] == 'q18a-alpha-next-c1b'
assert PARENT_BUILD['manifest'] == PINS['manifest'] and PARENT_BUILD['source'] == PINS['source']
b.PARENT = Path(PARENT_BUILD['bundle'])
b.HELD = b.PARENT.parent
b.PARENT_MANIFEST = PARENT_BUILD['manifest']
b.PARENT_OVERLAY_MANIFEST = PARENT_BUILD['overlay_manifest']
b.PARENT_REVIEW = PARENT_BUILD['review_sha256']
b.PARENT_BUNDLE_SOURCE = PARENT_BUILD['source']
b.PARENT_COMMIT = PARENT_BUILD['source']
b.PARENT_CANDIDATE = 'q18a-alpha-next-c1b'
b.PARENT_STEM = 'q18a_alpha_next_c1b'
b.PARENT_RUNNER = Path(PARENT_BUILD['runner'])
b.PARENT_RUNNER_SHA = PARENT_BUILD['runner_sha256']
b.PARENT_TESTS = PARENT_BUILD['expected_selfrun_tests']
ENGINE_COMMIT = PARENT_BUILD['source']
NEW_OVERLAY = ('e3/attn_meta_batch.py', 'e3/attn_meta_select.py')
CANDIDATE = 'q18a-inwait'


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


NON_AST_OVERLAY = frozenset({'run-flash-s5-test.sh'})


def overlay_declaration(path, old, new):
    """Only Python receives an AST proof; the one launcher gets exact byte pins."""
    if path.endswith('.py'):
        return b.changed_definitions(Path(old).read_text(), Path(new).read_text())
    assert path in NON_AST_OVERLAY, 'non-Python overlay file not allowlisted: ' + path
    return dict(format='non-ast-sha256/1', old_sha256=b.sha(old), new_sha256=b.sha(new))


def overlay_file_proof(path, old, new, declared):
    if path.endswith('.py'):
        return b.prove(Path(old).read_text(), Path(new).read_text(), path, declared)
    actual = overlay_declaration(path, old, new)
    assert declared == actual, ('non-AST overlay pin differs', path, declared, actual)
    return dict(actual, changed=['non-ast:sha256'])


_derive_runner = b.derive_runner


def derive_runner(candidate, tests, swaps_spec, pin, review_sha, sources, expect_engine_pins_unchanged):
    """Keep the inherited runner body, including its independent launcher check."""
    owner, _ = _derive_runner(candidate, tests, swaps_spec, pin, review_sha, sources, expect_engine_pins_unchanged)
    assignment = next(n for n in ast.parse(owner).body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'LAUNCHER')
    old = ast.get_source_segment(owner, assignment.value)
    launcher = sources['run-flash-s5-test.sh']
    assert re.fullmatch(r'[0-9a-f]{64}', launcher), 'invalid launcher pin'
    if expect_engine_pins_unchanged:
        assert ast.literal_eval(old) == launcher, 'unchanged launcher pin differs'
    updated = b.replace(owner, dict(LAUNCHER=repr(launcher)))
    assert b.replace(updated, dict(LAUNCHER=old)) == owner, 'runner changed beyond launcher pin'
    compile(updated, candidate + '-owner-run', 'exec')
    diff = [line for line in b.difflib.ndiff(b.PARENT_RUNNER.read_text().splitlines(), updated.splitlines())
            if line.startswith(('- ', '+ ')) and not line[2:].startswith('EXPECTED_SOURCES=')]
    return updated, diff


b.derive_runner = derive_runner


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
    definitions = {path: overlay_file_proof(path, b.PARENT / 'native30_overlay' / path, overlay / path,
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
    """The c1b reference bundle: parent bytes, engine sources asserted c1's, overlay regenerated and byte-identical."""
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
        assert target.read_bytes() == data, 'parent engine source differs from c1b: ' + name
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


REUSED_BASIS = 'NOT a run on this tree: the latest passing log of the parent tree (alpha-next-c1b chain b2), reused under the operator\'s discovery mode (3 Nov 2026)'


def reuse_parent_logs():
    """Discovery mode: the two build inputs are the parent tree's logs, and the review and provenance say so instead of 'fresh ... on the packaged tree'."""
    regression, focused = b.regression_from, b.focused_from
    b.regression_from = lambda log: dict(regression(log), basis=REUSED_BASIS)
    b.focused_from = lambda log, **kw: dict(focused(log), basis=REUSED_BASIS)


def build_discovery(spec_path, out, strict_log, focused_log):
    """build_native43's build alone: the bundle and the runner with all the build's own checks. No inwait-freeze.json and no OUT/parent (the runner reads neither; --negative needs OUT/parent)."""
    return b.build(spec_path, out, strict_log, focused_log)


def build(spec_path, out, strict_log, focused_log):
    import inwait_freeze as freeze
    result = b.build(spec_path, out, strict_log, focused_log)
    source = freeze.source_level(out / 'default-off-inwait-work')
    ref = reference(out)
    controls = dict(freeze.negative_controls(), **{'undeclared-replaced-tool': replaced_tool_control(out)})
    proof = dict(schema='inwait-freeze/1', verdict='PASS', source=source, negative_controls=controls,
                 tools_replaced=dict(declared=[], replaced=sorted(TOOLS_REPLACED)),
                 overlay=freeze.overlay_level(out / 'parent/native30_overlay', Path(result['overlay'])), reference=ref,
                 candidate_manifest=result['manifest'], candidate_overlay_manifest=result['overlay_manifest'],
                 parent_manifest=b.PARENT_MANIFEST, parent_pins_sha256=b.sha(parent_pin.TARGET),
                 dev_parent=bool(PINS.get('dev')), gpu_execution=False, box_action=False)
    b.dump(out / 'inwait-freeze.json', proof)
    print('INWAIT_FREEZE PASS ' + b.sha(out / 'inwait-freeze.json'))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--selftest-log', type=Path)
    p.add_argument('--focused-log', type=Path)
    p.add_argument('--prepare-review', action='store_true')
    p.add_argument('--propose', action='store_true')
    p.add_argument('--discovery', action='store_true', help='discovery mode: --selftest-log and --focused-log are the parent tree\'s, the records say so; no freeze proof, no parent reference')
    a = p.parse_args()
    if a.discovery:
        reuse_parent_logs()
    if a.propose:
        print(json.dumps(b.propose(b.load_spec(a.spec)), indent=2, sort_keys=True))
    elif a.prepare_review:
        b.dump(b.ROOT / b.REVIEW_PATH, b.prepare_review(b.load_spec(a.spec), a.selftest_log, a.focused_log))
    else:
        assert a.out is not None and a.selftest_log is not None and a.focused_log is not None
        (build_discovery if a.discovery else build)(a.spec, a.out.resolve(), a.selftest_log.resolve(), a.focused_log.resolve())
