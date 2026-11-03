#!/usr/bin/env python3
"""The alpha-next-c1b held bundle, freeze proof and owner runner: build_native43.py (never edited) with alpha-next-c1's delivered build-b
as parent. Harness-only: the engine bytes do not move.

Declared configuration, as tools/build_alpha_next_c1.py does: the parent is read from alpha-next-c1's own build.json, named and pinned
in tools/alpha_next_c1b_parent.json (tools/alpha_next_c1b_parent_pin.py; a placeholder exits 3, 'parent not pinned'). Candidate
q18a-alpha-next-c1b is admitted by a load_spec with the wider name rule, which also requires engine_changed == [] and
overlay_definitions == {}. NEW_OVERLAY = (): no overlay file is added or removed. TOOLS_REPLACED = {}.

Why the overlay is not byte-identical to the parent's, and what is proved instead: the c1b harness changes move the source review
(patches/s5_engine/native_candidate_review.json: harness_source_shas, cpu_regression, post_review_cpu and the alpha_next_c1b key;
make_s5_overlap._native_candidate_closure binds every tracked harness file's sha), and the generator embeds that review in the
overlay's closure. So exactly the METADATA rows built on the review move (e3/native-implementation-closure.json; the generation
and readiness records through implementation_review_sha256 / package_sha256 / receipt_sha256), and with them the overlay's
SHA256SUMS. Every other overlay row (engine modules, compiled libraries, plugins, launcher, sources) must be byte-identical.

Declared supersession (SUPERSEDED_CHECK, in the build's freeze proof and alpha-next-c1b-freeze.json): build_native43.build holds,
for a spec with no engine change, 'the regenerated overlay manifest equals the parent's'. That cannot hold when the review moves, so
build() is build_native43.build's own source with that one assertion swapped, exactly once, for:
  every non-METADATA overlay row byte-identical to the parent's (same file set, no .so moved), and every moved METADATA row moved
  only in review-bound keys (metadata_review_only: REVIEW_BOUND_KEYS, the closure's review equal to the bundle's prepared review).
overlay_level (wrapped, as alpha-next-c1's) asserts the first half on its own; the swapped assertion runs both after it. A scratch
overlay with one engine row or one .so changed, or with a metadata row changed outside the review-bound keys, is refused (negative
controls in the freeze proof; harness/test_alpha_next_c1b_pack.py runs them offline over c1's overlay).

Added after the build, in the same OUT:
  parent/   the alpha-next-c1 reference: a copy of the parent bundle whose e3_builder sources are asserted equal to c1's build
            source (the review pin aside), its overlay regenerated from BASE and asserted byte-identical to the parent's. The
            engine negative runs over this.
  alpha-next-c1b-freeze.json   overlay rows vs the parent's (engine rows identical, metadata review-only), engine sources under
            patches/s5_engine at the build source == the parent build source's == the bundle's e3_builder (review excepted),
            the superseded check quoted, negative controls. No GPU, no box action.
"""
import argparse
import inspect
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import alpha_next_c1b_parent_pin as parent_pin  # noqa: E402

PINS = parent_pin.load()
import build_native43 as b  # noqa: E402

PARENT_BUILD = json.loads(Path(PINS['build']).read_text())
assert b.sha(PINS['build']) == PINS['build_sha256'], 'alpha-next-c1 build record changed'
assert PARENT_BUILD['schema'] == 'native43-build/1' and PARENT_BUILD['name'] == 'q18a-alpha-next-c1'
assert PARENT_BUILD['manifest'] == PINS['manifest'] and PARENT_BUILD['source'] == PINS['source']
b.PARENT = Path(PARENT_BUILD['bundle'])
b.HELD = b.PARENT.parent
b.PARENT_MANIFEST = PARENT_BUILD['manifest']
b.PARENT_OVERLAY_MANIFEST = PARENT_BUILD['overlay_manifest']
b.PARENT_REVIEW = PARENT_BUILD['review_sha256']
b.PARENT_BUNDLE_SOURCE = PARENT_BUILD['source']
b.PARENT_COMMIT = PARENT_BUILD['source']
b.PARENT_CANDIDATE = 'q18a-alpha-next-c1'
b.PARENT_STEM = 'q18a_alpha_next_c1'
b.PARENT_RUNNER = Path(PARENT_BUILD['runner'])
b.PARENT_RUNNER_SHA = PARENT_BUILD['runner_sha256']
b.PARENT_TESTS = PARENT_BUILD['expected_selfrun_tests']
ENGINE_COMMIT = PARENT_BUILD['source']
NEW_OVERLAY = ()
CANDIDATE = 'q18a-alpha-next-c1b'
FREEZE = 'alpha-next-c1b-freeze.json'
# Keys of a METADATA json row that are bound to the source review (directly or through the closure's bytes): the only keys
# allowed to move. Any other key moving in any METADATA row refuses the build.
REVIEW_BOUND_KEYS = frozenset({'review', 'review_sha256', 'implementation_review_sha256', 'package_sha256', 'receipt_sha256'})


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    assert spec.get('schema') == 'native43-packaging-spec/1', 'spec schema'
    assert set(spec) == b.SPEC_KEYS, sorted(set(spec) ^ b.SPEC_KEYS)
    assert spec['candidate'] == CANDIDATE and spec['review'] == 'prepare' and spec['dry_run'] is False
    assert spec['engine_changed'] == [] and spec['overlay_definitions'] == {}, 'c1b is harness-only: no engine change declared'
    assert re.fullmatch(r'[0-9a-f]{40}', spec['source_commit'])
    return spec


b.load_spec = load_spec
TOOLS_REPLACED = {}
b.TOOLS_REPLACED = TOOLS_REPLACED


def _diff_paths(old, new, prefix=()):
    """Key paths where two json values differ (dicts walked; any other differing value is one leaf)."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                out.append(prefix + (key,))
            else:
                out += _diff_paths(old[key], new[key], prefix + (key,))
        return out
    return [] if old == new else [prefix]


def metadata_review_only(parent_overlay, overlay, review=None):
    """Problems (empty when it holds): every moved row is a METADATA json row whose differing key paths all end in (or pass
    through, for the closure's 'review') a REVIEW_BOUND_KEYS key; with REVIEW given, the closure's review equals it."""
    rows, parent_rows = b.verify(overlay), b.verify(parent_overlay)
    problems = []
    if set(rows) != set(parent_rows):
        problems.append('overlay-file-set-changed:%s' % sorted(set(rows) ^ set(parent_rows)))
    for name in sorted(n for n in parent_rows if n in rows and rows[n] != parent_rows[n]):
        if name not in b.METADATA or not name.endswith('.json'):
            problems.append('non-metadata-row-moved:' + name)
            continue
        old = json.loads((Path(parent_overlay) / name).read_text())
        new = json.loads((Path(overlay) / name).read_text())
        for path in _diff_paths(old, new):
            if not path or not any(k in REVIEW_BOUND_KEYS for k in path[:1] + path[-1:]):
                problems.append('metadata-key-moved-outside-the-review:%s:%s' % (name, '/'.join(map(str, path))))
    if review is not None:
        closure = json.loads((Path(overlay) / 'e3/native-implementation-closure.json').read_text())
        if closure.get('review') != review:
            problems.append('closure-review-is-not-the-prepared-review')
    return problems


b.metadata_review_only = metadata_review_only     # a name the swapped assertion below reads from build_native43's globals


def overlay_level(spec, overlay, parent_rows):
    """build_native43's overlay rule for an unchanged engine, with metadata allowed to move: no file added or removed, no
    non-METADATA row moved, no compiled library moved."""
    rows = b.verify(overlay)
    assert set(rows) == set(parent_rows), ('overlay file set changed', sorted(set(rows) ^ set(parent_rows)))
    moved = sorted(n for n in parent_rows if rows[n] != parent_rows[n])
    engine = sorted(n for n in moved if n not in b.METADATA)
    assert engine == [] == spec['engine_changed'], ('overlay engine row moved in a harness-only build', engine)
    assert not any(n.endswith('.so') for n in moved), 'compiled library moved'
    assert spec['overlay_definitions'] == {}
    return dict(parent_overlay=str(b.PARENT / 'native30_overlay'), parent_overlay_manifest=b.PARENT_OVERLAY_MANIFEST,
                moved=moved, added=[], engine_changed=[], metadata_moved=moved, definitions={},
                compiled_libraries_byte_identical=True, engine_rows_byte_identical_to_parent=True,
                byte_identical_to_parent=not moved)


b.overlay_level = overlay_level

# ---------------------------------------------------------------- the one declared swap in build_native43.build
SUPERSEDED_CHECK = (
    "    if not spec['engine_changed']:\n"
    "        assert sha(overlay / 'SHA256SUMS') == PARENT_OVERLAY_MANIFEST, 'regenerated overlay differs from held-b'\n",
    "    if not spec['engine_changed']:\n"
    "        assert not metadata_review_only(PARENT / 'native30_overlay', overlay, review), \\\n"
    "            ('regenerated overlay moved beyond the review-bound metadata', metadata_review_only(PARENT / 'native30_overlay', overlay, review))\n",
)


def derive_build(module=b):
    text = inspect.getsource(module.build)
    old, new = SUPERSEDED_CHECK
    assert text.count(old) == 1, 'build_native43.build changed: the superseded check is not there exactly once'
    return text.replace(old, new)


BUILD_TEXT = derive_build()
exec(compile(BUILD_TEXT, 'build_alpha_next_c1b:build_native43.build(swapped)', 'exec'), b.__dict__)   # defines b.build


def superseded_record():
    return dict(where='tools/build_native43.py build()', old=SUPERSEDED_CHECK[0], new=SUPERSEDED_CHECK[1],
                applied_exactly_once=True, build_native43_sha256=b.sha(TOOLS / 'build_native43.py'),
                review_bound_keys=sorted(REVIEW_BOUND_KEYS),
                why='the c1b harness changes move the source review, which the generator embeds in the overlay closure; the '
                    'parent-manifest equality cannot hold, so it is replaced by: no non-METADATA row moved, and every moved '
                    'METADATA row moved only in review-bound keys')


# ---------------------------------------------------------------- after the build
def _writable(root, on=True):
    for path in [root, *root.rglob('*')]:
        path.chmod((0o755 if path.is_dir() else 0o644) if on else (0o555 if path.is_dir() else 0o444))


def reference(out):
    """The alpha-next-c1 reference bundle: parent bytes, engine sources asserted c1's, overlay regenerated and byte-identical."""
    work = out / 'parent-work'
    work.mkdir()
    bundle = out / 'parent'
    shutil.copytree(b.PARENT, bundle)
    _writable(bundle)
    names = [n for n in b.git('ls-tree', '-r', '--name-only', ENGINE_COMMIT, 'patches/s5_engine').splitlines()
             if n != b.REVIEW_PATH]
    for name in names:
        data = subprocess.check_output(['git', 'show', ENGINE_COMMIT + ':' + name], cwd=b.ROOT)
        target = bundle / 'e3_builder' / Path(name).relative_to('patches/s5_engine')
        assert target.read_bytes() == data, 'parent engine source differs from alpha-next-c1: ' + name
    closure, _ = b.regenerate_overlay(work, bundle)
    assert b.sha(bundle / 'native30_overlay/SHA256SUMS') == b.PARENT_OVERLAY_MANIFEST, 'regenerated reference differs from the parent overlay'
    assert b.verify(bundle / 'native30_overlay') == b.verify(b.PARENT / 'native30_overlay')
    _writable(bundle, False)
    return dict(path=str(bundle), engine_commit=b.git('rev-parse', ENGINE_COMMIT).strip(), engine_files=len(names),
                overlay_manifest=b.PARENT_OVERLAY_MANIFEST, overlay_byte_identical_to_parent=True,
                review_sha256=closure['review_sha256'])


def engine_sources(source, held):
    """patches/s5_engine at the build source == at the parent build source == the bundle's e3_builder (the review excepted)."""
    ours = sorted(n for n in b.git('ls-tree', '-r', '--name-only', source, 'patches/s5_engine').splitlines() if n != b.REVIEW_PATH)
    theirs = sorted(n for n in b.git('ls-tree', '-r', '--name-only', ENGINE_COMMIT, 'patches/s5_engine').splitlines()
                    if n != b.REVIEW_PATH)
    assert ours == theirs, ('engine source file set moved', sorted(set(ours) ^ set(theirs)))
    differ = [n for n in ours if b.git('rev-parse', '%s:%s' % (source, n)) != b.git('rev-parse', '%s:%s' % (ENGINE_COMMIT, n))]
    assert not differ, ('engine sources moved vs alpha-next-c1', differ)
    for name in ours:
        rel = Path(name).relative_to('patches/s5_engine')
        assert (held / 'e3_builder' / rel).read_bytes() == (b.PARENT / 'e3_builder' / rel).read_bytes(), 'bundled engine source moved: ' + name
    return dict(source=source, parent_source=ENGINE_COMMIT, files=len(ours), differ=[], review_excepted=b.REVIEW_PATH,
                bundle_e3_builder_equal_to_parent=True)


def overlay_controls(out, overlay, review):
    """The swapped check both ways over scratch copies of the candidate overlay: metadata-only accepted, the rest refused."""
    parent = b.PARENT / 'native30_overlay'
    controls = {}

    def scratch(tag, edit):
        place = out / ('negative-c1b-' + tag)
        shutil.copytree(overlay, place)
        _writable(place)
        edit(place)
        b.manifest(place)
        return metadata_review_only(parent, place, review)

    def append(rel, text):
        return lambda place: (place / rel).write_bytes((place / rel).read_bytes() + text)

    def closure_key(place):
        path = place / 'e3/native-implementation-closure.json'
        data = json.loads(path.read_text())
        data['generated_source_shas'] = dict(data['generated_source_shas'], **{'c1b-undeclared': '0' * 64})
        path.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n')

    so = sorted(p.relative_to(overlay).as_posix() for p in overlay.rglob('*.so'))[0]
    for tag, edit in (('engine-row', append('e3/qsa_operation.py', b'\nC1B_UNDECLARED = 1\n')), ('compiled-library', append(so, b'\0')),
                      ('launcher', append('run-flash-s5-test.sh', b'\n# undeclared\n')), ('metadata-key', closure_key)):
        controls[tag] = 'REFUSED' if scratch(tag, edit) else 'ACCEPTED'
        assert controls[tag] == 'REFUSED', ('the swapped check accepted', tag)
    accepted = scratch('metadata-review-only', lambda place: None)
    assert not accepted, ('the candidate overlay itself is refused', accepted)
    controls['metadata-review-only'] = 'ACCEPTED'
    controls['overlay-level-engine-row'] = b.refused(lambda: overlay_level(dict(engine_changed=[], overlay_definitions={}),
                                                                           out / 'negative-c1b-engine-row', b.verify(parent)))
    return controls


def replaced_tool_control(out):
    """A parent bundle whose one bundled python tool differs from the committed one must be refused by stage (build_native43's rule)."""
    held = out / 'negative-replaced-tool-parent'
    shutil.copytree(b.PARENT, held)
    _writable(held)
    victim = sorted(p for p in (held / 'native35_tools').glob('*.py') if (TOOLS / p.name).is_file())[0]
    with victim.open('a') as handle:
        handle.write('\n# undeclared\n')
    saved, b.PARENT = b.PARENT, held
    try:
        return b.refused(lambda: b.stage(dict(tools=[]), out / 'negative-replaced-tool'))
    finally:
        b.PARENT = saved


def build(spec_path, out, strict_log, focused_log):
    result = b.build(spec_path, out, strict_log, focused_log)
    held = Path(result['bundle'])
    review = json.loads((held / 'e3_builder/native_candidate_review.json').read_text())
    problems = metadata_review_only(b.PARENT / 'native30_overlay', held / 'native30_overlay', review)
    assert not problems, problems
    rows, parent_rows = b.verify(held / 'native30_overlay'), b.verify(b.PARENT / 'native30_overlay')
    moved = sorted(n for n in parent_rows if rows[n] != parent_rows[n])
    ref = reference(out)
    controls = dict(overlay_controls(out, held / 'native30_overlay', review),
                    **{'undeclared-replaced-tool': replaced_tool_control(out)})
    proof = dict(schema='alpha-next-c1b-freeze/1', verdict='PASS',
                 overlay=dict(parent_overlay_manifest=b.PARENT_OVERLAY_MANIFEST, candidate_overlay_manifest=result['overlay_manifest'],
                              rows=len(rows), moved=moved, metadata_moved=moved, non_metadata_moved=[],
                              engine_rows_byte_identical_to_parent=True, metadata_review_only=True,
                              byte_identical_to_parent=not moved),
                 engine_sources=engine_sources(result['source'], held), superseded_check=superseded_record(),
                 negative_controls=controls, reference=ref, tools_replaced=dict(declared=[], replaced=sorted(TOOLS_REPLACED)),
                 candidate_manifest=result['manifest'], parent_manifest=b.PARENT_MANIFEST,
                 parent_pins_sha256=b.sha(parent_pin.TARGET), dev_parent=bool(PINS.get('dev')), gpu_execution=False,
                 box_action=False)
    b.dump(out / FREEZE, proof)
    print('ALPHA_NEXT_C1B_FREEZE PASS ' + b.sha(out / FREEZE))
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
