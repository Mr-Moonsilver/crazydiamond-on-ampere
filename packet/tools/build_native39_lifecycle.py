#!/usr/bin/env python3
"""Native39 lifecycle held bundle and owner runner.

Bundle: the native39 trunk held-b bundle (756f826c221c, source 75f7189) with
its engine overlay (native30_overlay), engine sources (e3_builder) and review
(native30-review.json) byte-identical, so the lifecycle runs the trunk engine
bit for bit; added are the lifecycle queue texts, window, suite, fixture leaves
and tools; changed are selftest.py (one registry row) and the provenance
(candidate, arm, sources, regression).  No overlay is regenerated.

Runner: the frozen trunk held-b owner runner (8b52d5eb...) with names, the
bundle pins (H, BUNDLE), the lifecycle window import and the lifecycle self-run
gate (tests_run, arm, lifecycle queue receipts) changed; REVIEW and
EXPECTED_SOURCES stay the trunk runner's (asserted), and every other byte is
asserted equal after normalisation.  The engine self-run record keeps the trunk
schema (tools/native39_trunk_selfrun.py, unchanged) with label == QNAME.

Usage: --out DIR --selftest-log L --focused-log F   (builds, held read-only)
"""
import argparse
import ast
import difflib
import json
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
PARENT_CANDIDATE = 'q18a-native39-trunk'
PARENT_RUNNER = HELD / 'q18a-native39-trunk-owner-run-756f826c221c.py'
PARENT_RUNNER_SHA = '8b52d5eb39639dee847fb93609f6653ef7e6cdc0a3750ed1a158742a2a90a960'
PARENT_TESTS = 218
IMAGE = 'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
CANDIDATE, ARM, STEM = 'q18a-native39-lifecycle', 'LIFECYCLE', 'q18a_native39_lifecycle'
HARNESS_CHANGED = {'selftest.py'}
HARNESS_ADDED = {'native39_lifecycle_gate.py', 'native39_lifecycle_fixture_leaves.json', 'test_native39_lifecycle_queue.py',
                 'runs/q18a_native39_lifecycle.draft.sh', 'runs/q18a_native39_lifecycle_declare.sh',
                 'runs/q18a_native39_lifecycle_settings.sh'}
TOOLS = ('build_native39_lifecycle.py', 'native39_lifecycle_freeze.py', 'native39_lifecycle_declarations.json',
         'native39_lifecycle_cpu_probe.py', 'native39_lifecycle_bundle_selfrun.py', 'native39_lifecycle_tools_derive.py',
         'native39_lifecycle_queue_drafts.py')
CHECKS = (('tools/native39_lifecycle_queue_drafts.py', '--check'), ('tools/native39_lifecycle_tools_derive.py', '--check'),
          ('tools/native39_trunk_queue_drafts.py', '--check'), ('tools/native39_trunk_tools_derive.py', '--check'))
sys.path[:0] = [str(ROOT / 'patches/s5_engine'), str(ROOT / 'tools'), str(ROOT / 'harness')]
from build_native30_held import sha, dump, manifest, verify, replace  # noqa: E402
import native39_lifecycle_freeze as freeze  # noqa: E402
import native39_lifecycle_gate as window  # noqa: E402

RUNNER_SWAPS = (
    ('    from native39_trunk_gate import trunk_window_ok as window_ok\n',
     '    from native39_lifecycle_gate import lifecycle_window_ok as window_ok\n'),
    ("result.get('native35_arm')=='TRUNK'", "result.get('native35_arm')=='LIFECYCLE'"),
    ("result.get('native39_trunk_queue')=='PASS'",
     "result.get('native39_trunk_queue')=='PASS' and result.get('native39_lifecycle_queue')=='PASS'"),
)


def expected_tests():
    tree = ast.parse((ROOT / 'tools/native39_lifecycle_cpu_probe.py').read_text())
    values = [n.value.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id == 'EXPECTED_TESTS']
    assert len(values) == 1 and type(values[0]) is int
    return values[0]


def runner_swaps():
    return (("result.get('tests_run',0)==%d" % PARENT_TESTS, "result.get('tests_run',0)==%d" % expected_tests()),) + RUNNER_SWAPS


def regression_from(log):
    text = log.read_text()
    summary = re.findall(r'^selftest: (\d+)/(\d+) cases as required$', text, re.M)
    suites = [json.JSONDecoder().raw_decode(text[m.end():])[0] for m in re.finditer('selftest suite: ', text)]
    assert (len(summary) == 1 and summary[0][0] == summary[0][1] and suites
            and all(s['returncode'] == 0 and not s['skipped'] for s in suites)), \
        'strict regression log must be complete, all-pass, zero skips'
    return dict(cases=int(summary[0][0]), required=int(summary[0][1]), exit_code=0, skips=0, suites=len(suites),
                reported_tests=sum(s['tests_run'] or 0 for s in suites), log_sha256=sha(log),
                basis='fresh full strict registered CPU regression on the native39 lifecycle final tree')


def focused_from(log):
    text = log.read_text()
    ran = re.findall(r'^Ran (\d+) tests? in [0-9.]+s\n\nOK$', text, re.M)
    assert (len(ran) == 1 and not re.search(r'\.\.\. skipped|^FAILED|^FAIL:|^ERROR:', text, re.M)
            and text.rstrip().endswith('rc=0')), 'focused log must be one all-pass run with zero skips'
    return dict(tests=int(ran[0]), skips=0, verdict='PASS', log_sha256=sha(log),
                basis='native39 lifecycle and trunk queue suites plus the native35 queue/lifetime/grade, measurement and '
                      'measurement-client, pp_release, partial-tail, identity-declare, private-auth, q18a-guard and native28 '
                      'host-guard suites')


def source_head():
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'harness', 'tools', 'patches/s5_engine'],
                                    cwd=ROOT, text=True)
    assert not dirty, 'commit harness/tools/patches before building:\n' + dirty
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def details():
    return dict(candidate=CANDIDATE, arm=ARM, parent=PARENT_CANDIDATE, parent_manifest=PARENT_MANIFEST,
        parent_bundle_source=PARENT_BUNDLE_SOURCE, engine_files=[],
        scope='the trunk held-b engine byte for byte (overlay, engine sources and review identical) under a lifecycle/leak '
              'queue; observer and offers off as on the trunk',
        queue='q18a-native39-lifecycle: the trunk queue through startup, bind receipt, W2 length guard and row1 (the lane '
              'proof), then native35-L\'s gates (startup, row1) and its native28 lifecycle block between two instruments '
              'receipts: native cancel/drain/reuse on the trunk W2 manifest, typed FULL hygiene, leaktest 8 x 3, pressure2 '
              '2 x 200000 beside the 7568 partial-tail leak at 30 % cancels, pressure completion and lifetime; native35 '
              'lifetime grade at exit; rule4, row4, W1, the plain W1 and the W2 speed rounds are absent',
        known_inconclusive=['pressure-lifetime and therefore native35-lifetime: the trunk engine env has no FLASH_POOL_STATS, so '
                            'no FLASH_PP_LAYOUT / FLASH_POOL_STATS line reaches the unchanged pp_release_grade lifetime oracle',
                            'native30-localize: no row4 placement to read (INCONCLUSIVE HARNESS, as on native36-wd and native35-L)'],
        window=dict(not_before=window.NOT_BEFORE, hard_end=window.HARD_END, active_s=window.LIFECYCLE_ACTIVE_S,
                    cleanup_margin_s=window.LIFECYCLE_CLEANUP_MARGIN_S, latest_entry_utc=window.LATEST_ENTRY_UTC,
                    active_derivation=[list(r) for r in window.ACTIVE_DERIVATION],
                    cleanup_derivation=[list(r) for r in window.CLEANUP_DERIVATION],
                    expected_runtime_s=window.EXPECTED_RUNTIME_S),
        model_settings_changed=False, tail_handback=False, device_run='NOT_RUN',
        fixture_leaves='native39_lifecycle_fixture_leaves.json', owner_runner=CANDIDATE + '-owner-run-<bundle12>.py',
        limitations='CPU fixture resources do not establish native cancellation, KV pressure, partial-tail copies or leak '
                    'behaviour on the device; the lifecycle block has not run on a native engine')


def derive_runner(pin, sources):
    original = PARENT_RUNNER.read_text()
    owner = original.replace('q18a_native39_trunk', STEM).replace('q18a-native39-trunk', CANDIDATE)
    for old, new in runner_swaps():
        assert owner.count(old) == 1, old[:80]
        owner = owner.replace(old, new)
    assignments = {n.targets[0].id: ast.get_source_segment(original, n.value) for n in ast.parse(original).body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    # The engine pins stay the trunk's: same review, same overlay rows.
    assert ast.literal_eval(assignments['EXPECTED_SOURCES']) == sources, 'overlay rows differ from the trunk runner pins'
    assert ast.literal_eval(assignments['REVIEW']) == PARENT_REVIEW
    owner = replace(owner, dict(H="Path('/home/<service-user>/harness/" + pin[:12] + "')", BUNDLE=repr(pin)))
    compile(owner, CANDIDATE + '-owner-run', 'exec')
    normalized = owner
    for old, new in reversed(runner_swaps()):
        normalized = normalized.replace(new, old)
    normalized = normalized.replace(STEM, 'q18a_native39_trunk').replace(CANDIDATE, 'q18a-native39-trunk')
    assert replace(normalized, {k: assignments[k] for k in ('H', 'BUNDLE')}) == original, 'unexpected runner body change'
    diff = [line for line in difflib.ndiff(original.splitlines(), owner.splitlines()) if line.startswith(('- ', '+ '))]
    return owner, diff


def build(out, strict_log, focused_log):
    assert not out.exists()
    assert sha(PARENT / 'SHA256SUMS') == PARENT_MANIFEST and sha(PARENT_RUNNER) == PARENT_RUNNER_SHA
    previous = verify(PARENT)
    parent_overlay = verify(PARENT / 'native30_overlay')
    assert sha(PARENT / 'native30_overlay/SHA256SUMS') == PARENT_OVERLAY_MANIFEST
    parent_provenance = json.loads((PARENT / 'native30-provenance.json').read_text())
    assert (parent_provenance['source'], parent_provenance['review_sha256'], parent_provenance['candidate']) == \
        (PARENT_BUNDLE_SOURCE, PARENT_REVIEW, PARENT_CANDIDATE)
    source = source_head()
    for tool, flag in CHECKS:
        subprocess.run([sys.executable, '-B', str(ROOT / tool), flag], check=True, cwd=ROOT)
    source_proof = freeze.source_level()
    negatives = freeze.negative_controls()
    regression, focused = regression_from(strict_log), focused_from(focused_log)
    out.mkdir()
    bundle = out / 'bundle'
    shutil.copytree(PARENT, bundle)
    for path in [bundle, *bundle.rglob('*')]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    tracked = subprocess.check_output(['git', 'ls-files', 'harness', 'patches/s5_engine'], cwd=ROOT, text=True).splitlines()
    added, replaced = [], []
    for name in tracked:
        src = ROOT / name
        if not src.is_file():
            continue
        if name.startswith('patches/s5_engine/'):
            target = bundle / 'e3_builder' / Path(name).relative_to('patches/s5_engine')
            assert target.is_file() and sha(target) == sha(src), 'engine source differs from the trunk bundle: ' + name
            continue
        target = bundle / Path(name).relative_to('harness')
        if target.exists():
            if sha(target) != sha(src):
                assert str(Path(name).relative_to('harness')) in HARNESS_CHANGED, \
                    'trunk bundle copy differs from the committed tree: ' + name
                shutil.copyfile(src, target)
                replaced.append(str(target.relative_to(bundle)))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
            added.append(str(target.relative_to(bundle)))
    assert set(added) == HARNESS_ADDED, sorted(set(added) ^ HARNESS_ADDED)
    assert set(replaced) == HARNESS_CHANGED, replaced
    tools = bundle / 'native35_tools'
    for directory in sorted(bundle.glob('native3*_tools')):
        for path in sorted(directory.glob('*.py')):
            if (ROOT / 'tools' / path.name).is_file():
                assert sha(path) == sha(ROOT / 'tools' / path.name), 'bundled tool differs from committed tools: ' + path.name
    for name in TOOLS:
        assert not (tools / name).exists()
        shutil.copyfile(ROOT / 'tools' / name, tools / name)
        added.append('native35_tools/' + name)
    provenance = json.loads((bundle / 'native30-provenance.json').read_text())
    assert provenance['overlay_manifest'] == PARENT_OVERLAY_MANIFEST and provenance['native35_arm'] == 'TRUNK'
    provenance.update(source=source, candidate=CANDIDATE, native35_arm=ARM, native39_lifecycle=details(), selftest=regression,
                      post_review_cpu=focused, parent_bundle_manifest=PARENT_MANIFEST, parent_candidate=PARENT_CANDIDATE,
                      parent_bundle_source=PARENT_BUNDLE_SOURCE, engine_changed=[], engine_changed_vs_native39_trunk=[],
                      overlay_byte_identical_to_parent=True, review_byte_identical_to_parent=True)
    dump(bundle / 'native30-provenance.json', provenance)
    manifest(bundle)
    pin = sha(bundle / 'SHA256SUMS')
    held = out / (CANDIDATE + '-bundle-' + pin[:12])
    bundle.rename(held)
    rows = verify(held)
    overlay_rows = verify(held / 'native30_overlay')
    assert overlay_rows == parent_overlay
    bundle_changed = sorted(n for n in previous if n in rows and rows[n] != previous[n])
    new = sorted(set(rows) - set(previous))
    removed = sorted(set(previous) - set(rows))
    assert set(bundle_changed) == freeze.BUNDLE_CHANGED and set(new) == set(added) and not removed, \
        (bundle_changed, sorted(set(new) ^ set(added)), removed)
    bundle_proof = freeze.bundle_level(held)
    owner, runner_diff = derive_runner(pin, dict(sorted(overlay_rows.items())))
    runner = out / (CANDIDATE + '-owner-run-' + pin[:12] + '.py')
    runner.write_text(owner)
    proof = dict(verdict='PASS', classification=None, source=source, parent_bundle=str(PARENT), parent_manifest=PARENT_MANIFEST,
                 parent_bundle_source=PARENT_BUNDLE_SOURCE, manifest=pin, changed=bundle_changed, added=new, removed=removed,
                 overlay_regenerated=False, overlay_manifest=sha(held / 'native30_overlay/SHA256SUMS'),
                 parent_overlay_manifest=PARENT_OVERLAY_MANIFEST, overlay_byte_identical=True, engine_changed=[],
                 freeze=dict(source=source_proof, bundle=bundle_proof, negative_controls=negatives),
                 review_sha256=PARENT_REVIEW, review_byte_identical=True,
                 runner=dict(path=str(runner), sha256=sha(runner), parent=str(PARENT_RUNNER), parent_sha256=PARENT_RUNNER_SHA,
                             differences=runner_diff, expected_selfrun_tests=expected_tests(),
                             review_and_expected_sources_unchanged=True,
                             only_names_pins_window_selfrun_gates_changed=True),
                 gpu_execution=False, box_action=False, details=details())
    dump(out / 'freeze-proof.json', proof)
    result = dict(name=CANDIDATE, arm=ARM, bundle=str(held), manifest=pin, overlay=str(held / 'native30_overlay'),
                  overlay_manifest=sha(held / 'native30_overlay/SHA256SUMS'), review_sha256=PARENT_REVIEW,
                  runner=str(runner), runner_sha256=sha(runner), runner_size=runner.stat().st_size,
                  checksummed_files=len(rows), source=source, held=True, regression=regression, focused=focused,
                  expected_selfrun_tests=expected_tests(), image=IMAGE, parent_manifest=PARENT_MANIFEST,
                  parent_runner=str(PARENT_RUNNER), parent_runner_sha256=PARENT_RUNNER_SHA, parent_bundle=str(PARENT),
                  engine_changed=[], latest_entry_utc=window.LATEST_ENTRY_UTC,
                  freeze_proof_sha256=sha(out / 'freeze-proof.json'))
    dump(out / 'build.json', result)
    for path in [held, *held.rglob('*'), runner]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    print(json.dumps({k: result[k] for k in ('name', 'manifest', 'overlay_manifest', 'review_sha256', 'runner_sha256',
                                             'expected_selfrun_tests', 'latest_entry_utc')}, sort_keys=True))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--selftest-log', type=Path, required=True)
    p.add_argument('--focused-log', type=Path, required=True)
    a = p.parse_args()
    build(a.out, a.selftest_log, a.focused_log)
