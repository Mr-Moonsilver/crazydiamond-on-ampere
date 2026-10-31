#!/usr/bin/env python3
"""Native37 robust held bundle: parent is the frozen native35 P-c held-c bundle.

native37 = held-c plus E1 (stock prefill held on an armed split route, memoised
typed admission refusals, exported counts, prefill_end typed refusal) in
e3/scheduler_hooks.py, e3/engine_driver.py and e3/native_overlap_session.py,
H1 (best-effort FULL return after a native lane error) in
harness/e3_native_performance.py, and the T1/T2 probe, the hub suite, fixture
leaves and tools.

The overlay is regenerated from BASE as in build_native36_w: exactly the three
engine files and the four generation/closure metadata files change; every
other engine file, plugin and compiled library stays byte-identical to held-c.
tools/native37_robust_freeze.py proves at source and overlay level that only
the declared definitions changed.

Not box-ready by design: no owner runner and no queue are cut (the W2 manifest
and every model setting are unchanged); the box bundle waits for the FULL
decode fix to be lined up with this one.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path('<hub-dir>/agents/flash-next/tmp/native35-p-held-c/q18a-native35-p-c-bundle-fe194172a6c8')
PARENT_MANIFEST = 'fe194172a6c83f11ce9bb5cbae306732be52ef08b5f52202bcdcf546310fbc13'
PARENT_OVERLAY_MANIFEST = '32673c374f8b258df54d229eac8bcd456cd633ca51715c939a55f8615194591f'
PARENT_REVIEW = '3f8f7201bd71ffecdd13df9c85f429efa4850a53cf4fb44701ba985a9a8c58a9'
PARENT_SOURCE = '720b1238310de0ba61cc9a2b9f8a7dd8d9c1abb4'
BASE = Path('<hub-dir>/agents/flash-next/tmp/e2f-native-offline-base-20261027T211405Z')
BASE_SHA = 'da561411fadbb82922eeb8eafbe9424bc9703aaf8c2030bd813743c692b641a5'
CANDIDATE = 'q18a-native37-robust'
ENGINE_CHANGED = {'e3/scheduler_hooks.py', 'e3/engine_driver.py', 'e3/native_overlap_session.py'}
METADATA = {'S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json',
            'e3/native-overlap-readiness.json', 'e3/decode-probe-evidence.json'}
SOURCE_CHANGED = {'scheduler_hooks.py', 'engine_driver.py', 'native_overlap_session.py', 'native_candidate_review.json'}
HARNESS_CHANGED = {'e3_native_performance.py', 'test_native35_performance.py', 'selftest.py'}
HARNESS_ADDED = {'e3_native37_admission_probe.py', 'test_native37_robust.py', 'native37_fixture_leaves.json'}
TOOLS = ('native37_robust_selfrun.py', 'native37_robust_freeze.py', 'build_native37_robust.py')
PLUGINS = ('native30_host.py', 'native30_policy.py', 'native34_diag.py', 'native34_worker.py', 'native34_control.py',
           'native34_storage.py', 'native34_indexer.py', 'native34_readback.py')
BUNDLE_CHANGED = ({'native30-provenance.json', 'native30-review.json', 'e3_builder/native_candidate_review.json'}
                  | {'e3_builder/' + n for n in SOURCE_CHANGED - {'native_candidate_review.json'}}
                  | HARNESS_CHANGED
                  | {'native30_overlay/' + n for n in ENGINE_CHANGED | METADATA | {'SHA256SUMS'}})
sys.path[:0] = [str(ROOT / 'patches/s5_engine'), str(ROOT / 'tools'), str(ROOT / 'harness')]
from build_native30_held import sha, dump, manifest, verify  # noqa: E402
from native_readiness import digest, payload_identity, verify_finalized_package  # noqa: E402
import native37_robust_freeze as freeze  # noqa: E402


def regression_from(log):
    text = log.read_text()
    summary = re.findall(r'^selftest: (\d+)/(\d+) cases as required$', text, re.M)
    suites = [json.JSONDecoder().raw_decode(text[m.end():])[0] for m in re.finditer('selftest suite: ', text)]
    assert (len(summary) == 1 and summary[0][0] == summary[0][1] and suites
            and all(s['returncode'] == 0 and not s['skipped'] for s in suites)), \
        'strict regression log must be complete, all-pass, zero skips'
    return dict(cases=int(summary[0][0]), required=int(summary[0][1]), exit_code=0, skips=0, suites=len(suites),
                reported_tests=sum(s['tests_run'] or 0 for s in suites), log_sha256=sha(log),
                basis='fresh full strict registered CPU regression on the native37 robust final tree')


def focused_from(log):
    text = log.read_text()
    ran = re.findall(r'^Ran (\d+) tests? in [0-9.]+s\n\nOK$', text, re.M)
    assert (len(ran) == 1 and not re.search(r'\.\.\. skipped|^FAILED|^FAIL:|^ERROR:', text, re.M)
            and text.rstrip().endswith('rc=0')), 'focused log must be one all-pass run with zero skips'
    return dict(tests=int(ran[0]), skips=0, verdict='PASS', log_sha256=sha(log),
                basis=('test_native37_robust plus every suite that imports scheduler_hooks, engine_driver, '
                       'native_overlap_session, e3_native_performance or e3_native35_performance, in one run'))


def source_head():
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'harness', 'tools', 'patches/s5_engine'],
                                    cwd=ROOT, text=True)
    assert not dirty, 'commit harness/tools/patches before building:\n' + dirty
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def details():
    return dict(
        parent_manifest=PARENT_MANIFEST, parent_candidate='q18a-native35-p-c', parent_source=PARENT_SOURCE,
        candidate=CANDIDATE,
        change=('E1: defer_stock_prefix_hit returns True while an EngineCore native admission intent is armed '
                '(split route), so a request native admission refused waits unallocated in waiting instead of '
                'reaching the worker route guard; a refused staged-plan search is memoised per request object '
                'and planning inputs; typed refusal receipts (reason, start, native end, forced stop, gap, '
                'queries, exhausted) and engine-lifetime counts (admitted, refused, alloc_failures, plan '
                'searches, memo hits, split stock deferrals) are exported in the driver status and overlap '
                'evidence; prefill_end carries a typed refusal, also when the refused window journals nothing '
                '(evidence read refused, status read instead). H1: e3_native_performance.prefill does a '
                'best-effort pause and leave_full after a native lane error, each in its own try, recorded as '
                "record['cleanup']; a typed refusal is the lane error"),
        not_changed=('W2 manifest, model and serve settings, queue texts, owner runner, compiled libraries, '
                     'plugins, every other engine file'),
        capability='ending the native lane at the last forced stop with a FULL stock tail: design only '
                   '(docs/e3-briefs/q18a-native37-tail-handback-design.md)',
        delta_author='Claude Opus 5.5 (claude/native37-robust)', codex_review_of_delta='NOT_RUN',
        review_identity_note=("native_readiness.check_review requires reviewed_by='Codex' for the implementation "
                              'closure; that field is inherited from the native35 review. The native37 delta has '
                              'not been reviewed by Codex.'),
        tests=dict(t1_t2='harness/e3_native37_admission_probe.py (pinned image, tools/native37_robust_selfrun.py)',
                   t3_and_hub='harness/test_native37_robust.py (registered in harness/selftest.py)',
                   fixture_leaves='harness/native37_fixture_leaves.json'),
        box_ready=False, owner_runner=None, device_run='NOT_RUN', vm_input='NONE')


def prepare_review(strict_log, focused_log):
    review = json.loads((ROOT / 'patches/s5_engine/native_candidate_review.json').read_text())
    tracked = subprocess.check_output(['git', 'ls-files', 'harness'], cwd=ROOT, text=True).splitlines()
    review['source_shas'] = {n: sha(ROOT / 'patches/s5_engine' / n) for n in review['source_shas']}
    review['harness_source_shas'] = {str(Path(n).relative_to('harness')): sha(ROOT / n)
                                     for n in tracked if (ROOT / n).is_file()}
    review['cpu_regression'] = regression_from(strict_log)
    review['post_review_cpu'] = focused_from(focused_log)
    review['native37_robust'] = details()
    return review


def build(out, strict_log, focused_log):
    assert not out.exists()
    assert sha(PARENT / 'SHA256SUMS') == PARENT_MANIFEST and sha(BASE / 'SHA256SUMS') == BASE_SHA
    previous = verify(PARENT)
    parent_overlay = verify(PARENT / 'native30_overlay')
    assert sha(PARENT / 'native30_overlay/SHA256SUMS') == PARENT_OVERLAY_MANIFEST
    parent_provenance = json.loads((PARENT / 'native30-provenance.json').read_text())
    assert parent_provenance['source'] == PARENT_SOURCE and parent_provenance['review_sha256'] == PARENT_REVIEW
    source = source_head()
    source_proof = freeze.source_level()
    negatives = freeze.negative_controls()
    regression, focused = regression_from(strict_log), focused_from(focused_log)
    review = prepare_review(strict_log, focused_log)
    assert json.loads((ROOT / 'patches/s5_engine/native_candidate_review.json').read_text()) == review, \
        'prepare and commit the native37 source review before freeze'
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
        target = (bundle / Path(name).relative_to('harness') if name.startswith('harness/')
                  else bundle / 'e3_builder' / Path(name).relative_to('patches/s5_engine'))
        if target.exists():
            if sha(target) != sha(src):
                allowed = (Path(name).name in SOURCE_CHANGED if name.startswith('patches/s5_engine/')
                           else str(Path(name).relative_to('harness')) in HARNESS_CHANGED)
                assert allowed, 'parent copy differs from the committed tree: ' + name
                shutil.copyfile(src, target)
                replaced.append(str(target.relative_to(bundle)))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
            added.append(str(target.relative_to(bundle)))
    assert set(added) == HARNESS_ADDED, sorted(set(added) ^ HARNESS_ADDED)
    assert set(replaced) == ({'e3_builder/' + n for n in SOURCE_CHANGED} | HARNESS_CHANGED), replaced
    tools = bundle / 'native35_tools'
    for directory in sorted(bundle.glob('native3*_tools')):
        for path in sorted(directory.glob('*.py')):
            if (ROOT / 'tools' / path.name).is_file():
                assert sha(path) == sha(ROOT / 'tools' / path.name), 'bundled tool differs from committed tools: ' + path.name
    for name in TOOLS:
        assert not (tools / name).exists()
        shutil.copyfile(ROOT / 'tools' / name, tools / name)
        added.append('native35_tools/' + name)
    assert sha(PARENT / 'native30-review.json') == sha(PARENT / 'e3_builder/native_candidate_review.json')
    shutil.copyfile(bundle / 'e3_builder/native_candidate_review.json', bundle / 'native30-review.json')
    # Regenerate the overlay from BASE exactly as build_native36_w does.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join(map(str, (bundle / 'e3_builder', bundle / 'e2_builder', bundle))))
    raw, final = out / 'raw', out / 'final'
    command = ['python3', '-B', str(bundle / 'e3_builder/make_s5_overlap.py'), str(BASE), str(raw), '--reference',
               str(bundle / 'e3_reference'), '--base-manifest-sha256', BASE_SHA, '--native-candidate']
    with (out / 'generation.log').open('x') as f:
        subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    # Python-only change: retain the exact owner-qualified compiled libraries.
    for name in ('flash_s5_green.so', 'flash_s5_replay_observer.so', 'q17-helper-build.json'):
        shutil.copyfile(PARENT / 'native30_overlay/e3/native' / name, raw / 'e3/native' / name)
    with (out / 'finalize.log').open('x') as f:
        subprocess.run(['python3', '-B', str(bundle / 'e3_finalize_package.py'), str(raw), str(final)], env=env,
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    overlay = bundle / 'native30_overlay'
    for path in final.rglob('*'):
        if path.is_file():
            target = overlay / path.relative_to(final)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    for name in PLUGINS:
        assert sha(ROOT / 'harness' / name) == sha(overlay / 'e3/native30_plugins' / name), 'plugin differs: ' + name
    gen = json.loads((overlay / 'S5-OVERLAP-GENERATION.json').read_text())
    ready = json.loads((overlay / 'e3/native-overlap-readiness.json').read_text())
    gen['package_sha256'] = ready['package_sha256'] = payload_identity(overlay)
    ready['receipt_sha256'] = digest({k: v for k, v in ready.items() if k != 'receipt_sha256'})
    dump(overlay / 'S5-OVERLAP-GENERATION.json', gen)
    dump(overlay / 'e3/native-overlap-readiness.json', ready)
    manifest(overlay)
    closure = json.loads((overlay / 'e3/native-implementation-closure.json').read_text())
    verify_finalized_package(overlay, closure['generated_source_shas'])
    assert closure['review'] == review and closure['review_sha256'] == digest(review) != PARENT_REVIEW
    newov = verify(overlay)
    assert set(newov) == set(parent_overlay), set(newov) ^ set(parent_overlay)
    changed = sorted(n for n in parent_overlay if newov[n] != parent_overlay[n])
    engine_changed = sorted(n for n in changed if n not in METADATA)
    assert set(engine_changed) == ENGINE_CHANGED, engine_changed
    assert not any(n.endswith('.so') for n in changed) and set(changed) - set(engine_changed) <= METADATA, changed
    old_probe = json.loads((PARENT / 'native30_overlay/e3/decode-probe-evidence.json').read_text())
    new_probe = json.loads((overlay / 'e3/decode-probe-evidence.json').read_text())
    assert {k for k in set(old_probe) | set(new_probe) if old_probe.get(k) != new_probe.get(k)} <= {'generated_source_shas'}
    old_closure = json.loads((PARENT / 'native30_overlay/e3/native-implementation-closure.json').read_text())
    assert old_closure['native_source_shas'] == closure['native_source_shas']
    overlay_proof = freeze.overlay_level(overlay)
    provenance = json.loads((bundle / 'native30-provenance.json').read_text())
    assert provenance['candidate'] == 'q18a-native35-p-c' and provenance['overlay_manifest'] == PARENT_OVERLAY_MANIFEST
    provenance.update(source=source, candidate=CANDIDATE, native37_robust=details(), selftest=regression,
                      post_review_cpu=focused, overlay_manifest=sha(overlay / 'SHA256SUMS'),
                      review_sha256=closure['review_sha256'], parent_bundle_manifest=PARENT_MANIFEST,
                      engine_changed=engine_changed, engine_changed_vs_native35_p_c=engine_changed)
    dump(bundle / 'native30-provenance.json', provenance)
    manifest(bundle)
    pin = sha(bundle / 'SHA256SUMS')
    held = out / (CANDIDATE + '-bundle-' + pin[:12])
    bundle.rename(held)
    assert all(sha(path) == sha(held / 'vmid_sources' / path.relative_to(PARENT / 'vmid_sources'))
               for path in (PARENT / 'vmid_sources').rglob('*') if path.is_file()), 'historical VM snapshot changed'
    rows = verify(held)
    bundle_changed = sorted(n for n in previous if n in rows and rows[n] != previous[n])
    new = sorted(set(rows) - set(previous))
    removed = sorted(set(previous) - set(rows))
    assert set(bundle_changed) == BUNDLE_CHANGED and set(new) == set(added) and not removed, \
        (sorted(set(bundle_changed) ^ BUNDLE_CHANGED), new, removed)
    proof = dict(verdict='PASS', classification=None, source=source, parent_bundle=str(PARENT),
                 parent_manifest=PARENT_MANIFEST, parent_source=PARENT_SOURCE, manifest=pin,
                 changed=bundle_changed, added=new, removed=removed, overlay_regenerated=True,
                 overlay_manifest=sha(held / 'native30_overlay/SHA256SUMS'), parent_overlay_manifest=PARENT_OVERLAY_MANIFEST,
                 overlay_changed=changed, engine_changed=engine_changed,
                 freeze=dict(source=source_proof, overlay=overlay_proof, negative_controls=negatives),
                 compiled_libraries_byte_identical=True, plugins_byte_identical=True, native_sources_byte_identical=True,
                 review_sha256=closure['review_sha256'], parent_review_sha256=PARENT_REVIEW,
                 historical_vm_snapshots_byte_identical=True, owner_runner=None, box_ready=False,
                 gpu_execution=False, box_action=False, details=details())
    dump(out / 'freeze-proof.json', proof)
    result = dict(name=CANDIDATE, bundle=str(held), manifest=pin, overlay=str(held / 'native30_overlay'),
                  overlay_manifest=sha(held / 'native30_overlay/SHA256SUMS'), review_sha256=closure['review_sha256'],
                  checksummed_files=len(rows), source=source, held=True, regression=regression, focused=focused,
                  parent_manifest=PARENT_MANIFEST, engine_changed=engine_changed, box_ready=False,
                  freeze_proof_sha256=sha(out / 'freeze-proof.json'))
    dump(out / 'build.json', result)
    for path in [held, *held.rglob('*')]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    p.add_argument('--selftest-log', type=Path, required=True)
    p.add_argument('--focused-log', type=Path, required=True)
    p.add_argument('--prepare-review', action='store_true')
    a = p.parse_args()
    if a.prepare_review:
        dump(ROOT / 'patches/s5_engine/native_candidate_review.json', prepare_review(a.selftest_log, a.focused_log))
    else:
        assert a.out is not None
        build(a.out, a.selftest_log, a.focused_log)
