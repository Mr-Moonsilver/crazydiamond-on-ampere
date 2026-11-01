#!/usr/bin/env python3
"""Native43 held bundle, freeze proof and owner runner, built from a declared source commit.

Parent: the native39 trunk held-b bundle (756f826c221c, built from 75f7189; branch head
1ffc394) and its frozen owner runner (8b52d5eb...).  Input: a candidate spec
(tools/native43_spec.<label>.json) naming
  source_commit   the engine/harness commit to package (Codex's native43 commit, or
                  1ffc394 for the held-b dry run);
  candidate, arm  the queue name and self-run arm; the runner follows the candidate stem;
  declarations    every path changed since 1ffc394 (tools included), every changed
                  definition of each modified Python file, every moved overlay file and
                  its changed definitions;
  review          'inherit' (held-b's committed review, dry run) or 'prepare' (this
                  builder's --prepare-review output, committed on the build branch);
  runner/self-run the self-run test count, the exact runner swaps and the rehearsal;
  checks          derivation checks (--check) the source tree must pass before the build.

Build tree: this worktree's HEAD, clean.  The source commit must be HEAD or an
ancestor of it, and harness/ and patches/s5_engine/ must equal the source commit's
bytes, except the prepared review when the spec says 'prepare'.  Packaging adds only
tools/ (and records outside the bundle).

Overlay: regenerated from BASE exactly as build_native39_trunk.py did
(make_s5_overlap.py --native-candidate, then e3_finalize_package.py) with held-b's
owner-qualified compiled libraries.  The moved overlay files must equal the declared
engine files plus generation metadata.  With no engine change (the dry run) the
regenerated overlay must equal held-b's byte for byte.

Runner: held-b's runner text with the candidate names, H/BUNDLE (and REVIEW and
EXPECTED_SOURCES when the overlay moves), the self-run test count and the declared
swaps, each applied exactly once.  Normalised back, every other byte equals held-b's
runner.  A rehearsal spec derives the native43-named runner from the same parent as a
shadow (compiled, normalised back, diff recorded; never written as a runner).

Usage:
  --spec S --propose                                  print the declaration proposal
  --spec S --prepare-review --selftest-log L --focused-log F
  --spec S --out DIR --selftest-log L --focused-log F (builds; held read-only)
No GPU, no box action.
"""
import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HELD = Path('<hub-dir>/agents/flash-next/tmp/native39-trunk-held-b')
PARENT = HELD / 'q18a-native39-trunk-bundle-756f826c221c'
PARENT_MANIFEST = '756f826c221c026440d7955a9f834fe72db83581439fed9bb69d07d314ea4264'
PARENT_OVERLAY_MANIFEST = 'b8f345948642e98285645990c6b35c54d86a9d10f347611bcfc642aa8d444115'
PARENT_REVIEW = 'fd3fb01086ce641f0586b7a0db4ad894b39c09ce59db52b793d873bfda3d0398'
PARENT_BUNDLE_SOURCE = '75f71897aac4bf0764596c83d6ddbbba4b343301'
PARENT_COMMIT = '1ffc394783d3d67e140056868b50ca2f74c88b3b'
PARENT_CANDIDATE = 'q18a-native39-trunk'
PARENT_STEM = 'q18a_native39_trunk'
PARENT_RUNNER = HELD / 'q18a-native39-trunk-owner-run-756f826c221c.py'
PARENT_RUNNER_SHA = '8b52d5eb39639dee847fb93609f6653ef7e6cdc0a3750ed1a158742a2a90a960'
PARENT_TESTS = 218
BASE = Path('<hub-dir>/agents/flash-next/tmp/e2f-native-offline-base-20261027T211405Z')
BASE_SHA = 'da561411fadbb82922eeb8eafbe9424bc9703aaf8c2030bd813743c692b641a5'
IMAGE = 'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
COMPILED_LIBRARIES = ('flash_s5_green.so', 'flash_s5_replay_observer.so', 'q17-helper-build.json')
PLUGINS = ('native30_host.py', 'native30_policy.py', 'native34_diag.py', 'native34_worker.py', 'native34_control.py',
           'native34_storage.py', 'native34_indexer.py', 'native34_readback.py')
# Generation metadata that moves with any engine-source or review change (trunk METADATA).
METADATA = {'S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json', 'e3/native-overlap-readiness.json',
            'e3/decode-probe-evidence.json', 'core/SHA256SUMS'}
REVIEW_PATH = 'patches/s5_engine/native_candidate_review.json'
SPEC_KEYS = {'schema', 'label', 'dry_run', 'source_commit', 'candidate', 'arm', 'review', 'review_key', 'details',
             'source_paths', 'definitions', 'engine_changed', 'overlay_definitions', 'tools', 'queue',
             'expected_tests', 'runner_swaps', 'rehearsal', 'engine_selfrun', 'bundle_selfrun', 'clock_split', 'precheck',
             'checks'}
sys.path[:0] = [str(ROOT / 'patches/s5_engine'), str(ROOT / 'tools'), str(ROOT / 'harness')]
from build_native30_held import sha, dump, manifest, verify, replace  # noqa: E402
from native37_robust_freeze import prove, changed as changed_definitions  # noqa: E402


# ---------------------------------------------------------------- spec and tree
def load_spec(path):
    spec = json.loads(Path(path).read_text())
    assert spec.get('schema') == 'native43-packaging-spec/1', 'spec schema'
    assert set(spec) == SPEC_KEYS, sorted(set(spec) ^ SPEC_KEYS)
    assert re.fullmatch(r'q18a-native[0-9a-z-]+', spec['candidate']) and spec['review'] in ('inherit', 'prepare')
    assert re.fullmatch(r'[0-9a-f]{40}', spec['source_commit'])
    if spec['dry_run']:
        assert spec['source_commit'] == PARENT_COMMIT and spec['engine_changed'] == [] and spec['review'] == 'inherit'
    return spec


def stem(candidate):
    return candidate.replace('-', '_')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def head():
    dirty = git('status', '--porcelain', '--', 'harness', 'tools', 'patches/s5_engine')
    assert not dirty, 'commit harness/tools/patches before building:\n' + dirty
    return git('rev-parse', 'HEAD').strip()


def tree_matches_source(spec):
    """harness/ and patches/s5_engine/ at HEAD are the source commit's bytes (review excepted)."""
    source = spec['source_commit']
    subprocess.run(['git', 'merge-base', '--is-ancestor', source, 'HEAD'], cwd=ROOT, check=True)
    rows = git('diff', '--name-status', source, 'HEAD', '--', 'harness', 'patches/s5_engine').splitlines()
    allowed = {'M\t' + REVIEW_PATH} if spec['review'] == 'prepare' else set()
    assert set(rows) <= allowed, ('build tree differs from the source commit', rows)
    return dict(source_commit=source, head=git('rev-parse', 'HEAD').strip(), differs=rows)


def source_paths():
    return git('diff', '--name-status', PARENT_COMMIT, 'HEAD', '--', 'patches/s5_engine', 'harness', 'tools').splitlines()


# ---------------------------------------------------------------- freeze (source level)
def compile_proof(paths):
    files = sorted(row.split('\t', 1)[1] for row in paths if row.endswith('.py') and row[0] in 'AM')
    for name in files:
        compile((ROOT / name).read_text(), name, 'exec')
    return dict(files=files, count=len(files), ok=True)


def check_paths(paths, spec):
    assert set(paths) == set(spec['source_paths']), ('undeclared source paths', sorted(set(paths) ^ set(spec['source_paths'])))


def source_level(spec):
    paths = source_paths()
    check_paths(paths, spec)
    modified = {row.split('\t', 1)[1] for row in paths if row.startswith('M\t') and row.endswith('.py')}
    assert modified == set(spec['definitions']), sorted(modified ^ set(spec['definitions']))
    result = {}
    for path, names in sorted(spec['definitions'].items()):
        result[path] = prove(git('show', PARENT_COMMIT + ':' + path), (ROOT / path).read_text(), path, names)
    return dict(parent=PARENT_COMMIT, paths=paths, definitions=result, compile=compile_proof(paths))


def propose(spec):
    """The declaration a reviewer checks and commits into the spec (never applied by itself)."""
    paths = source_paths()
    out = dict(source_paths=paths, definitions={})
    for row in paths:
        status, path = row.split('\t', 1)
        if status == 'M' and path.endswith('.py'):
            out['definitions'][path] = changed_definitions(git('show', PARENT_COMMIT + ':' + path), (ROOT / path).read_text())
    return out


# ---------------------------------------------------------------- review
def regression_from(log):
    text = log.read_text()
    summary = re.findall(r'^selftest: (\d+)/(\d+) cases as required$', text, re.M)
    suites = [json.JSONDecoder().raw_decode(text[m.end():])[0] for m in re.finditer('selftest suite: ', text)]
    assert (len(summary) == 1 and summary[0][0] == summary[0][1] and suites
            and all(s['returncode'] == 0 and not s['skipped'] for s in suites)), \
        'strict regression log must be complete, all-pass, zero skips'
    return dict(cases=int(summary[0][0]), required=int(summary[0][1]), exit_code=0, skips=0, suites=len(suites),
                reported_tests=sum(s['tests_run'] or 0 for s in suites), log_sha256=sha(log),
                basis='fresh full strict registered CPU regression on the packaged tree')


def focused_from(log, basis='the spec\'s focused suites over the packaged tree'):
    text = log.read_text()
    ran = re.findall(r'^Ran (\d+) tests? in [0-9.]+s\n\nOK$', text, re.M)
    assert (len(ran) == 1 and not re.search(r'\.\.\. skipped|^FAILED|^FAIL:|^ERROR:', text, re.M)
            and text.rstrip().endswith('rc=0')), 'focused log must be one all-pass run with zero skips'
    return dict(tests=int(ran[0]), skips=0, verdict='PASS', log_sha256=sha(log),
                basis=basis)


def prepare_review(spec, strict_log, focused_log):
    review = json.loads((ROOT / REVIEW_PATH).read_text())
    if spec['review'] == 'inherit':
        return review
    tracked = git('ls-files', 'harness').splitlines()
    review['source_shas'] = {n: sha(ROOT / 'patches/s5_engine' / n) for n in review['source_shas']}
    review['harness_source_shas'] = {str(Path(n).relative_to('harness')): sha(ROOT / n)
                                     for n in tracked if (ROOT / n).is_file()}
    review['cpu_regression'] = regression_from(strict_log)
    review['post_review_cpu'] = focused_from(focused_log)
    review[spec['review_key']] = details(spec)
    return review


def details(spec):
    return dict(spec['details'], candidate=spec['candidate'], arm=spec['arm'], parent=PARENT_CANDIDATE,
                parent_manifest=PARENT_MANIFEST, parent_commit=PARENT_COMMIT, source_commit=spec['source_commit'],
                engine_files=sorted(spec['engine_changed']), dry_run=spec['dry_run'], device_run='NOT_RUN')


# ---------------------------------------------------------------- runner
def runner_swaps(spec, tests):
    swaps = [tuple(row) for row in spec['runner_swaps']]
    if tests != PARENT_TESTS:
        swaps.insert(0, ("result.get('tests_run',0)==%d" % PARENT_TESTS, "result.get('tests_run',0)==%d" % tests))
    return swaps


def derive_runner(candidate, tests, swaps_spec, pin, review_sha, sources, expect_engine_pins_unchanged):
    original = PARENT_RUNNER.read_text()
    owner = original.replace(PARENT_STEM, stem(candidate)).replace(PARENT_CANDIDATE, candidate)
    swaps = runner_swaps(dict(runner_swaps=swaps_spec), tests)
    for old, new in swaps:
        assert owner.count(old) == 1, ('runner swap must match exactly once', old[:100])
        owner = owner.replace(old, new)
    assignments = {n.targets[0].id: ast.get_source_segment(original, n.value) for n in ast.parse(original).body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    pins = dict(H="Path('/home/<service-user>/harness/" + pin[:12] + "')", BUNDLE=repr(pin))
    if expect_engine_pins_unchanged:
        assert ast.literal_eval(assignments['REVIEW']) == review_sha == PARENT_REVIEW, 'review pin moved'
        assert ast.literal_eval(assignments['EXPECTED_SOURCES']) == sources, 'overlay rows differ from the parent pins'
    else:
        pins.update(REVIEW=repr(review_sha), EXPECTED_SOURCES=repr(sources))
    owner = replace(owner, pins)
    compile(owner, candidate + '-owner-run', 'exec')
    normalized = owner
    for old, new in reversed(swaps):
        normalized = normalized.replace(new, old)
    normalized = normalized.replace(stem(candidate), PARENT_STEM).replace(candidate, PARENT_CANDIDATE)
    assert replace(normalized, {k: assignments[k] for k in pins}) == original, 'unexpected runner body change'
    diff = [line for line in difflib.ndiff(original.splitlines(), owner.splitlines())
            if line.startswith(('- ', '+ ')) and not line[2:].startswith('EXPECTED_SOURCES=')]
    return owner, diff


# ---------------------------------------------------------------- bundle and overlay
def stage(spec, bundle):
    """held-b bundle + the build tree's harness/engine sources + the spec's packaging tools."""
    shutil.copytree(PARENT, bundle)
    for path in [bundle, *bundle.rglob('*')]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    added, replaced = [], []
    for name in git('ls-files', 'harness', 'patches/s5_engine').splitlines():
        src = ROOT / name
        if not src.is_file():
            continue
        target = (bundle / Path(name).relative_to('harness') if name.startswith('harness/')
                  else bundle / 'e3_builder' / Path(name).relative_to('patches/s5_engine'))
        if target.exists():
            if sha(target) != sha(src):
                shutil.copyfile(src, target)
                replaced.append(str(target.relative_to(bundle)))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
            added.append(str(target.relative_to(bundle)))
    tools = bundle / 'native35_tools'
    for directory in sorted(bundle.glob('native3*_tools')):
        for path in sorted(directory.glob('*.py')):
            if (ROOT / 'tools' / path.name).is_file():
                assert sha(path) == sha(ROOT / 'tools' / path.name), 'bundled tool differs from committed tools: ' + path.name
    for name in spec['tools']:
        assert not (tools / name).exists(), 'packaging tool already bundled: ' + name
        shutil.copyfile(ROOT / 'tools' / name, tools / name)
        added.append('native35_tools/' + name)
    shutil.copyfile(bundle / 'e3_builder/native_candidate_review.json', bundle / 'native30-review.json')
    return sorted(added), sorted(replaced)


def expected_staging(spec):
    """Bundle rows the declared source paths must add or replace (harness/engine only; tools are spec['tools'])."""
    added, replaced = set(), set()
    for row in spec['source_paths']:
        status, path = row.split('\t', 1)
        if path.startswith('tools/'):
            continue
        rel = (str(Path(path).relative_to('harness')) if path.startswith('harness/')
               else 'e3_builder/' + str(Path(path).relative_to('patches/s5_engine')))
        assert status in 'AM', row
        (added if status == 'A' else replaced).add(rel)
    if spec['review'] == 'prepare':
        replaced.add('e3_builder/native_candidate_review.json')
    tools = {'tools/' + n for n in spec['tools']}
    declared_tools = {row.split('\t', 1)[1] for row in spec['source_paths'] if row.startswith('A\ttools/')}
    assert tools <= declared_tools, sorted(tools - declared_tools)
    return added | {'native35_tools/' + n for n in spec['tools']}, replaced


def regenerate_overlay(out, bundle):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join(map(str, (bundle / 'e3_builder', bundle / 'e2_builder', bundle))))
    raw, final = out / 'raw', out / 'final'
    command = ['python3', '-B', str(bundle / 'e3_builder/make_s5_overlap.py'), str(BASE), str(raw), '--reference',
               str(bundle / 'e3_reference'), '--base-manifest-sha256', BASE_SHA, '--native-candidate']
    with (out / 'generation.log').open('x') as f:
        subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    for name in COMPILED_LIBRARIES:
        shutil.copyfile(PARENT / 'native30_overlay/e3/native' / name, raw / 'e3/native' / name)
    with (out / 'finalize.log').open('x') as f:
        subprocess.run(['python3', '-B', str(bundle / 'e3_finalize_package.py'), str(raw), str(final)], env=env,
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    from native_readiness import digest, payload_identity, verify_finalized_package
    overlay = bundle / 'native30_overlay'
    for path in final.rglob('*'):
        if path.is_file():
            target = overlay / path.relative_to(final)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    for name in PLUGINS:
        assert sha(bundle / name) == sha(overlay / 'e3/native30_plugins' / name), 'plugin differs: ' + name
    gen = json.loads((overlay / 'S5-OVERLAP-GENERATION.json').read_text())
    ready = json.loads((overlay / 'e3/native-overlap-readiness.json').read_text())
    gen['package_sha256'] = ready['package_sha256'] = payload_identity(overlay)
    ready['receipt_sha256'] = digest({k: v for k, v in ready.items() if k != 'receipt_sha256'})
    dump(overlay / 'S5-OVERLAP-GENERATION.json', gen)
    dump(overlay / 'e3/native-overlap-readiness.json', ready)
    manifest(overlay)
    closure = json.loads((overlay / 'e3/native-implementation-closure.json').read_text())
    verify_finalized_package(overlay, closure['generated_source_shas'])
    return closure, digest


def overlay_level(spec, overlay, parent_rows):
    rows = verify(overlay)
    assert set(rows) == set(parent_rows), ('overlay file set changed', sorted(set(rows) ^ set(parent_rows)))
    moved = sorted(n for n in parent_rows if rows[n] != parent_rows[n])
    engine = sorted(n for n in moved if n not in METADATA)
    assert engine == sorted(spec['engine_changed']), ('undeclared overlay engine change', engine, spec['engine_changed'])
    assert not any(n.endswith('.so') for n in moved), 'compiled library moved'
    if not engine:
        assert not moved, ('metadata moved without an engine change', moved)
    assert set(spec['overlay_definitions']) == set(engine), sorted(set(spec['overlay_definitions']) ^ set(engine))
    definitions = {}
    for path in engine:
        definitions[path] = prove((PARENT / 'native30_overlay' / path).read_text(), (overlay / path).read_text(), path,
                                  spec['overlay_definitions'][path])
    return dict(parent_overlay=str(PARENT / 'native30_overlay'), parent_overlay_manifest=PARENT_OVERLAY_MANIFEST,
                moved=moved, engine_changed=engine, metadata_moved=sorted(set(moved) - set(engine)),
                definitions=definitions, compiled_libraries_byte_identical=True,
                byte_identical_to_parent=not moved)


# ---------------------------------------------------------------- negative controls
def refused(check):
    try:
        check()
    except AssertionError:
        return 'REFUSED'
    raise AssertionError('undeclared change accepted')


def negative_controls(spec, overlay, parent_rows):
    out = {'undeclared-path': refused(lambda: check_paths(source_paths() + ['A\tharness/native43_undeclared.py'], spec))}
    path = 'tools/build_native43.py'
    text = (ROOT / path).read_text()
    out['undeclared-function'] = refused(lambda: prove(text, text + '\ndef native43_undeclared():\n    pass\n', path, []))
    probe = dict(spec, engine_changed=list(spec['engine_changed']))
    scratch = overlay.parent.parent / 'negative-overlay'
    shutil.copytree(overlay, scratch)
    target = scratch / 'e3/qsa_operation.py'
    target.chmod(0o644)
    target.write_text(target.read_text() + '\nNATIVE43_UNDECLARED = 1\n')
    manifest(scratch)
    if 'e3/qsa_operation.py' in spec['engine_changed']:
        out['undeclared-overlay-definition'] = refused(lambda: overlay_level(probe, scratch, parent_rows))
    else:
        out['undeclared-overlay-file'] = refused(lambda: overlay_level(probe, scratch, parent_rows))
    out['runner-swap-twice'] = refused(lambda: derive_runner(spec['candidate'], spec['expected_tests'],
                                                             [["QNAME=", "QNAME="]], PARENT_MANIFEST, PARENT_REVIEW,
                                                             {}, True))
    return out


# ---------------------------------------------------------------- build
def build(spec_path, out, strict_log, focused_log):
    spec = load_spec(spec_path)
    assert not out.exists()
    assert sha(PARENT / 'SHA256SUMS') == PARENT_MANIFEST and sha(PARENT_RUNNER) == PARENT_RUNNER_SHA
    assert sha(BASE / 'SHA256SUMS') == BASE_SHA
    previous = verify(PARENT)
    parent_rows = verify(PARENT / 'native30_overlay')
    assert sha(PARENT / 'native30_overlay/SHA256SUMS') == PARENT_OVERLAY_MANIFEST
    parent_provenance = json.loads((PARENT / 'native30-provenance.json').read_text())
    assert (parent_provenance['source'], parent_provenance['review_sha256'], parent_provenance['candidate']) == \
        (PARENT_BUNDLE_SOURCE, PARENT_REVIEW, PARENT_CANDIDATE)
    source = head()
    tree = tree_matches_source(spec)
    for tool, flag in spec['checks']:
        subprocess.run([sys.executable, '-B', str(ROOT / tool), flag], check=True, cwd=ROOT)
    source_proof = source_level(spec)
    regression, focused = regression_from(strict_log), focused_from(focused_log)
    review = prepare_review(spec, strict_log, focused_log)
    assert json.loads((ROOT / REVIEW_PATH).read_text()) == review, 'prepare and commit the source review before the build'
    if spec['review'] == 'inherit':
        assert sha(ROOT / REVIEW_PATH) == sha(PARENT / 'e3_builder/native_candidate_review.json')
    out.mkdir()
    bundle = out / 'bundle'
    added, replaced = stage(spec, bundle)
    want_added, want_replaced = expected_staging(spec)
    assert set(added) == want_added and set(replaced) == want_replaced, \
        (sorted(set(added) ^ want_added), sorted(set(replaced) ^ want_replaced))
    for name in spec['queue']:
        assert (bundle / 'runs' / name).is_file(), 'queue text missing: ' + name
    for suffix in ('.draft.sh', '_declare.sh'):
        assert stem(spec['candidate']) + suffix in spec['queue'], 'the runner phases need ' + suffix
    closure, digest = regenerate_overlay(out, bundle)
    assert closure['review'] == review
    overlay = bundle / 'native30_overlay'
    overlay_proof = overlay_level(spec, overlay, parent_rows)
    if spec['review'] == 'inherit':
        assert closure['review_sha256'] == PARENT_REVIEW
    if not spec['engine_changed']:
        assert sha(overlay / 'SHA256SUMS') == PARENT_OVERLAY_MANIFEST, 'regenerated overlay differs from held-b'
    provenance = json.loads((bundle / 'native30-provenance.json').read_text())
    provenance.update(source=source, candidate=spec['candidate'], native35_arm=spec['arm'], native43=details(spec),
                      selftest=regression, post_review_cpu=focused, overlay_manifest=sha(overlay / 'SHA256SUMS'),
                      review_sha256=closure['review_sha256'], parent_bundle_manifest=PARENT_MANIFEST,
                      parent_candidate=PARENT_CANDIDATE, parent_bundle_source=PARENT_BUNDLE_SOURCE,
                      source_commit=spec['source_commit'], engine_changed=sorted(spec['engine_changed']),
                      engine_changed_vs_native39_trunk=sorted(spec['engine_changed']),
                      box_ready=not spec['dry_run'], dry_run=spec['dry_run'])
    dump(bundle / 'native30-provenance.json', provenance)
    manifest(bundle)
    pin = sha(bundle / 'SHA256SUMS')
    held = out / (spec['candidate'] + '-bundle-' + pin[:12])
    bundle.rename(held)
    rows = verify(held)
    bundle_changed = sorted(n for n in previous if n in rows and rows[n] != previous[n])
    new = sorted(set(rows) - set(previous))
    removed = sorted(set(previous) - set(rows))
    want_changed = ({'native30-provenance.json'} | want_replaced
                    | ({'native30-review.json'} if spec['review'] == 'prepare' else set())
                    | {'native30_overlay/' + n for n in overlay_proof['moved']}
                    | ({'native30_overlay/SHA256SUMS'} if overlay_proof['moved'] else set()))
    assert set(bundle_changed) == want_changed and set(new) == want_added and not removed, \
        (sorted(set(bundle_changed) ^ want_changed), sorted(set(new) ^ want_added), removed)
    overlay_rows = dict(sorted(verify(held / 'native30_overlay').items()))
    unchanged_engine = not overlay_proof['moved']
    owner, runner_diff = derive_runner(spec['candidate'], spec['expected_tests'], spec['runner_swaps'], pin,
                                       closure['review_sha256'], overlay_rows, unchanged_engine)
    runner = out / (spec['candidate'] + '-owner-run-' + pin[:12] + '.py')
    runner.write_text(owner)
    rehearsal = None
    if spec['rehearsal']:
        r = spec['rehearsal']
        text, diff = derive_runner(r['candidate'], r['expected_tests'], r['runner_swaps'], pin,
                                   closure['review_sha256'], overlay_rows, unchanged_engine)
        rehearsal = dict(candidate=r['candidate'], expected_tests=r['expected_tests'], compiled=True,
                         normalised_back_to_parent=True, sha256=hashlib.sha256(text.encode()).hexdigest(),
                         differences=diff, written=False)
    negatives = negative_controls(spec, held / 'native30_overlay', parent_rows)
    proof = dict(verdict='PASS', classification=None, source=source, tree=tree, parent_bundle=str(PARENT),
                 parent_manifest=PARENT_MANIFEST, parent_commit=PARENT_COMMIT, manifest=pin, changed=bundle_changed,
                 added=new, removed=removed, overlay_regenerated=True, overlay_manifest=sha(held / 'native30_overlay/SHA256SUMS'),
                 overlay=overlay_proof, engine_changed=overlay_proof['engine_changed'],
                 freeze=dict(source=source_proof, negative_controls=negatives),
                 review_sha256=closure['review_sha256'], parent_review_sha256=PARENT_REVIEW,
                 runner=dict(path=str(runner), sha256=sha(runner), parent=str(PARENT_RUNNER), parent_sha256=PARENT_RUNNER_SHA,
                             differences=runner_diff, expected_selfrun_tests=spec['expected_tests'],
                             swaps=[list(s) for s in runner_swaps(spec, spec['expected_tests'])],
                             only_names_pins_and_declared_swaps_changed=True),
                 rehearsal=rehearsal, spec=str(spec_path), spec_sha256=sha(spec_path),
                 gpu_execution=False, box_action=False, details=details(spec))
    dump(out / 'freeze-proof.json', proof)
    result = dict(schema='native43-build/1', name=spec['candidate'], arm=spec['arm'], label=spec['label'],
                  dry_run=spec['dry_run'], bundle=str(held), manifest=pin, overlay=str(held / 'native30_overlay'),
                  overlay_manifest=sha(held / 'native30_overlay/SHA256SUMS'), review_sha256=closure['review_sha256'],
                  runner=str(runner), runner_sha256=sha(runner), runner_size=runner.stat().st_size,
                  checksummed_files=len(rows), source=source, source_commit=spec['source_commit'], held=True,
                  regression=regression, focused=focused, expected_selfrun_tests=spec['expected_tests'], image=IMAGE,
                  parent_manifest=PARENT_MANIFEST, parent_runner=str(PARENT_RUNNER), parent_runner_sha256=PARENT_RUNNER_SHA,
                  parent_bundle=str(PARENT), engine_changed=overlay_proof['engine_changed'],
                  overlay_byte_identical_to_parent=overlay_proof['byte_identical_to_parent'],
                  spec=str(spec_path), spec_sha256=sha(spec_path), freeze_proof_sha256=sha(out / 'freeze-proof.json'))
    dump(out / 'build.json', result)
    for path in [held, *held.rglob('*'), runner]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    print(json.dumps({k: result[k] for k in ('name', 'label', 'manifest', 'overlay_manifest', 'review_sha256',
                                             'runner_sha256', 'expected_selfrun_tests', 'engine_changed',
                                             'overlay_byte_identical_to_parent')}, sort_keys=True))
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
        print(json.dumps(propose(load_spec(a.spec)), indent=2, sort_keys=True))
    elif a.prepare_review:
        dump(ROOT / REVIEW_PATH, prepare_review(load_spec(a.spec), a.selftest_log, a.focused_log))
    else:
        assert a.out is not None and a.selftest_log is not None and a.focused_log is not None
        build(a.spec, a.out.resolve(), a.selftest_log.resolve(), a.focused_log.resolve())
