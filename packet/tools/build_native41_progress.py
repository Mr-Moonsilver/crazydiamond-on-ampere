#!/usr/bin/env python3
"""Regenerate held-f (q18a-native41-progress) and its exact held-e2 (818a4a7) reference from the pinned BASE.

Held-e2's delivered bundle (fc9fee4268dd) supplies the unchanged compiled libraries and the packaging shell.
Adds the held-f progress queue and its owner runner: the runner is held-e2's runner (re-derived here through
held-e's builder and proven byte-identical to the delivered one) with the named held-f swaps. No model setting,
launcher change or device action.
"""
import argparse
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'patches/s5_engine'), str(ROOT / 'harness')]
from build_native30_held import sha, dump, manifest, verify  # noqa: E402
from build_native38_plain_bind import regression_from, focused_from  # noqa: E402
from native_readiness import digest, payload_identity, verify_finalized_package  # noqa: E402
PARENT = '818a4a7876751fcbb677f7b5994f6a1d44e2a185'
SHELL = Path('<hub-dir>/agents/flash-next/tmp/native41-autoarm-held-e2/q18a-native41-autoarm-e-bundle-fc9fee4268dd')
SHELL_SHA = 'fc9fee4268ddac29603b9e538350a1b9444752a4b4c6dff23b864ba3aeaceb9f'
SHELL_RUNNER_SHA = '589e68a4eef7c7b204a31a3a7557ef8028087b78dd73572b9b74e19501d73e7d'
SHELL_CANDIDATE = 'q18a-native41-autoarm-e'
BASE = Path('<hub-dir>/agents/flash-next/tmp/e2f-native-offline-base-20261027T211405Z')
BASE_SHA = 'da561411fadbb82922eeb8eafbe9424bc9703aaf8c2030bd813743c692b641a5'
CANDIDATE = 'q18a-native41-progress'
STEM = 'q18a_native41_progress'
SCOPE = ('Held-f: per-command native progress receipts (engine and worker ProgressTrace lines joined by transport '
         'sequence and rank) and the burst host-progress candidate behind FLASH_S5_NATIVE_PROGRESS=burst (exact-'
         'sequence direct replies for flash_s5_submit/flash_s5_poll on a second per-rank FIFO, bounded owner in-wait '
         'native progress inside the sample_tokens wait), the per-arm native_progress switch on the native router '
         'only with FLASH_S5_AUTO_ARM=1 and FLASH_S5_NATIVE_PROGRESS=burst, and the eight-arm matched timing queue '
         '(current/burst x live/solo, 65,445-token fresh prompt below the 97,152 threshold, eight 16,384-token '
         'decoders); default-off: old receive loop and wait body preserved (AST-identical); new wrapper and shared '
         'worker-output lock overhead declared')
TARGETED = ['startup', 'bind', 'unarmed-stock-prefill', 'current-solo-0', 'burst-solo-0', 'burst-live-0', 'current-live-0',
            'current-live-1', 'burst-live-1', 'burst-solo-1', 'current-solo-1', 'typed-FULL']
META = {'core/SHA256SUMS', 'S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json',
        'e3/native-overlap-readiness.json', 'e3/decode-probe-evidence.json'}
HELD_E_ENGINE_COMMENT = '        # Native41: exact frozen connected auto-arm and status-receipt cases, distinct from queue-leaf tests.'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def paths(commit=None):
    return (git('ls-tree', '-r', '--name-only', commit, 'harness', 'tools', 'patches/s5_engine').splitlines() if commit
            else git('ls-files', 'harness', 'tools', 'patches/s5_engine').splitlines())


def source(commit, name):
    return subprocess.check_output(['git', 'show', commit + ':' + name], cwd=ROOT) if commit else (ROOT / name).read_bytes()


def runtime():
    from native41_progress_freeze import RUNTIME, ADDED
    return RUNTIME, ADDED


def details():
    return dict(base=PARENT, scope=SCOPE, fixture_leaves='native41_progress_fixture_leaves.json', device_execution='NOT_RUN',
                model_settings_changed=False, tail_handback=False, launcher_changed=False, targeted_queue=TARGETED,
                speed_claim=False, progress_setting='FLASH_S5_NATIVE_PROGRESS=burst with FLASH_S5_AUTO_ARM=1 in the queue '
                                                    'EXTRA_ENV only; current or burst per arm')


def review_for(bundle, regression, focused, label):
    p = bundle / 'e3_builder/native_candidate_review.json'
    r = json.loads(p.read_text())
    r['source_shas'] = {n: sha(bundle / 'e3_builder' / n) for n in r['source_shas']}
    r['harness_source_shas'] = {str(Path(n).relative_to('harness')): sha(bundle / Path(n).relative_to('harness'))
                                for n in paths(PARENT if label == 'parent' else None) if n.startswith('harness/')}
    if label != 'parent':
        r['cpu_regression'] = regression
        r['post_review_cpu'] = focused
        r['native41_progress'] = details()
    dump(p, r)
    dump(bundle / 'native30-review.json', r)
    return r


def prepare_review(strict, focused):
    review = json.loads((ROOT / 'patches/s5_engine/native_candidate_review.json').read_text())
    review['source_shas'] = {n: sha(ROOT / 'patches/s5_engine' / n) for n in review['source_shas']}
    review['harness_source_shas'] = {str(Path(n).relative_to('harness')): sha(ROOT / n) for n in paths() if n.startswith('harness/')}
    reg, focus = regression_from(strict), focused_from(focused)
    reg['basis'] = 'fresh full strict held-f CPU regression'
    focus['basis'] = ('held-f progress queue and reply-order suites plus held-e\'s focused native41/native40/native39/'
                      'native38/admission/driver/bridge/storage/router suites')
    review['cpu_regression'] = reg
    review['post_review_cpu'] = focus
    review['native41_progress'] = details()
    return review


def held_e_runner(pin, review_sha, sources):
    """Held-e2's owner runner for these pins, as held-e's builder derives it."""
    import build_native41_autoarm as parent_build
    return parent_build.derive_runner(pin, review_sha, sources)[0]


def expected_tests():
    import build_native41_autoarm as parent_build
    import native41_progress_queue
    return parent_build.expected_tests() + native41_progress_queue.PROGRESS_QUEUE_TESTS


def engine_cases():
    from native41_progress_engine_selfrun import CASES
    return sorted(CASES)


def runner_swaps():
    import build_native41_autoarm as parent_build
    return ((SHELL_CANDIDATE, CANDIDATE), ('q18a_native41_autoarm', STEM),
            ('from native41_box_gate import window_ok', 'from native41_progress_gate import window_ok'),
            ("tests_run',0)==%d and" % parent_build.expected_tests(), "tests_run',0)==%d and" % expected_tests()),
            (HELD_E_ENGINE_COMMENT,
             '        # Held-f: exact frozen connected auto-arm, receipt and native progress cases, distinct from queue-leaf tests.'),
            ("engine.get('schema')=='native41-selfrun/1'", "engine.get('schema')=='native41-progress-selfrun/1'"),
            ("sorted(engine.get('cases') or {})==%r" % parent_build.engine_cases(),
             "sorted(engine.get('cases') or {})==%r" % engine_cases()))


def derive_runner(pin, review_sha, sources):
    old = held_e_runner(pin, review_sha, sources)
    text = old
    for a, b in runner_swaps():
        assert text.count(a) >= 1, a
        if a.startswith('q18a'):
            text = text.replace(a, b)
        else:
            assert text.count(a) == 1, a
            text = text.replace(a, b)
    assert 'autoarm' not in text, 'held-e name left in the held-f runner'
    compile(text, CANDIDATE + '-owner-run', 'exec')
    back = text
    for a, b in reversed(runner_swaps()):
        back = back.replace(b, a)
    assert back == old, 'unexpected held-f runner change'
    return text, old


def stage(out, label, regression, focused):
    parent = label == 'parent'
    bundle = out / label
    shutil.copytree(SHELL, bundle)
    for p in [bundle, *bundle.rglob('*')]:
        p.chmod(0o755 if p.is_dir() else 0o644)
    for n in paths(PARENT if parent else None):
        if n.startswith('harness/'):
            target = bundle / Path(n).relative_to('harness')
        elif n.startswith('patches/s5_engine/'):
            target = bundle / 'e3_builder' / Path(n).relative_to('patches/s5_engine')
        else:
            target = bundle / 'native35_tools' / Path(n).relative_to('tools')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source(PARENT if parent else None, n))
    review = review_for(bundle, regression, focused, label)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join(map(str, (bundle / 'e3_builder', bundle / 'e2_builder', bundle))))
    raw, final = out / (label + '-raw'), out / (label + '-final')
    with (out / (label + '-generation.log')).open('x') as f:
        subprocess.run(['python3', '-B', str(bundle / 'e3_builder/make_s5_overlap.py'), str(BASE), str(raw), '--reference',
                        str(bundle / 'e3_reference'), '--base-manifest-sha256', BASE_SHA, '--native-candidate'],
                       env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    for n in ('flash_s5_green.so', 'flash_s5_replay_observer.so', 'q17-helper-build.json'):
        shutil.copyfile(SHELL / 'native30_overlay/e3/native' / n, raw / 'e3/native' / n)
    with (out / (label + '-finalize.log')).open('x') as f:
        subprocess.run(['python3', '-B', str(bundle / 'e3_finalize_package.py'), str(raw), str(final)], env=env,
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    ov = bundle / 'native30_overlay'
    for p in final.rglob('*'):
        if p.is_file():
            t = ov / p.relative_to(final)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, t)
    gen = json.loads((ov / 'S5-OVERLAP-GENERATION.json').read_text())
    ready = json.loads((ov / 'e3/native-overlap-readiness.json').read_text())
    gen['package_sha256'] = ready['package_sha256'] = payload_identity(ov)
    ready['receipt_sha256'] = digest({k: v for k, v in ready.items() if k != 'receipt_sha256'})
    dump(ov / 'S5-OVERLAP-GENERATION.json', gen)
    dump(ov / 'e3/native-overlap-readiness.json', ready)
    manifest(ov)
    closure = json.loads((ov / 'e3/native-implementation-closure.json').read_text())
    verify_finalized_package(ov, closure['generated_source_shas'])
    assert closure['review'] == review
    # Only held-f's runtime sources and metadata may differ from held-e2's delivered overlay; the regenerated
    # reference must carry held-e2's exact engine bytes.
    RUNTIME, ADDED = runtime()
    before, after = verify(SHELL / 'native30_overlay'), verify(ov)
    added = set(after) - set(before)
    assert not set(before) - set(after), 'no held-e2 overlay file may be removed'
    assert added == (set() if parent else ADDED), sorted(added)
    changed = {n for n in before if before[n] != after[n]}
    assert changed <= (META if parent else RUNTIME | META), sorted(changed - (META if parent else RUNTIME | META))
    assert after['run-flash-s5-test.sh'] == before['run-flash-s5-test.sh'], 'launcher must stay byte-identical'
    ancestry = json.loads((bundle / 'native30-provenance.json').read_text())
    provenance = dict(schema='native41-progress-provenance/1',
                      candidate=CANDIDATE if not parent else 'held-e2-818a4a7-reference',
                      source=git('rev-parse', PARENT if parent else 'HEAD').strip(), overlay_manifest=sha(ov / 'SHA256SUMS'),
                      review_sha256=closure['review_sha256'], parent_source=PARENT,
                      owner_runner=CANDIDATE + '-owner-run-<SUMS12>.py' if not parent else None, box_ready=not parent,
                      native35_arm='TRUNK', parent_bundle_manifest=SHELL_SHA, ancestral_parent_provenance=ancestry,
                      overlay_engine_changed_vs_parent=sorted((changed | added) - META - {'SHA256SUMS'}),
                      engine_changed_vs_parent=[] if parent else sorted(RUNTIME | ADDED),
                      selftest=None if parent else regression, post_review_cpu=None if parent else focused,
                      model_settings_changed=False, tail_handback=False, launcher_changed=False,
                      queue='runs/' + STEM + '.draft.sh' if not parent else None, gpu_work=False, box_action=False)
    dump(bundle / 'native30-provenance.json', provenance)
    manifest(bundle)
    return bundle


def build(out, strict, focused):
    import native41_progress_freeze as freeze
    import native41_progress_queue
    import native41_progress_gate
    assert not out.exists()
    assert not git('status', '--porcelain', '--', 'harness', 'tools', 'patches/s5_engine').strip(), 'commit source before freeze'
    assert sha(SHELL / 'SHA256SUMS') == SHELL_SHA and sha(BASE / 'SHA256SUMS') == BASE_SHA
    verify(SHELL)
    verify(BASE)
    prepared = prepare_review(strict, focused)
    assert json.loads((ROOT / 'patches/s5_engine/native_candidate_review.json').read_text()) == prepared, \
        'prepare and commit review before freeze'
    reg, focus = prepared['cpu_regression'], prepared['post_review_cpu']
    for n, text in (native41_progress_queue.derive() | native41_progress_queue.derive_tools()).items():
        assert (ROOT / n).read_text() == text, n
    # Held-e's runner derivation reproduces held-e2's delivered runner byte for byte.
    shell_runner = SHELL.parent / (SHELL_CANDIDATE + '-owner-run-' + SHELL_SHA[:12] + '.py')
    assert sha(shell_runner) == SHELL_RUNNER_SHA
    shell_review = json.loads((SHELL / 'native30_overlay/e3/native-implementation-closure.json').read_text())['review_sha256']
    assert held_e_runner(SHELL_SHA, shell_review, dict(sorted(verify(SHELL / 'native30_overlay').items()))) == \
        shell_runner.read_text(), 'held-e2 runner derivation differs from the delivered runner'
    out.mkdir()
    parent = stage(out, 'parent', reg, focus)
    bundle = stage(out, 'candidate', reg, focus)
    source_proof = freeze.source_level()
    overlay_proof = freeze.overlay_level(parent / 'native30_overlay', bundle / 'native30_overlay')
    proof = dict(verdict='PASS', source=source_proof, overlay=overlay_proof, negative_controls=freeze.negative_controls(),
                 branch_base=git('rev-parse', PARENT).strip(), parent_bundle_sha256=SHELL_SHA,
                 compiled_libraries_and_plugins_byte_identical=True, parent_overlay_regenerated_from_exact_base=True,
                 model_settings_changed=False, tail_handback=False, launcher_changed=False)
    pin = sha(bundle / 'SHA256SUMS')
    held = out / (CANDIDATE + '-bundle-' + pin[:12])
    bundle.rename(held)
    review_sha = json.loads((held / 'native30_overlay/e3/native-implementation-closure.json').read_text())['review_sha256']
    runner_text, held_e_text = derive_runner(pin, review_sha, dict(sorted(verify(held / 'native30_overlay').items())))
    runner = out / (CANDIDATE + '-owner-run-' + pin[:12] + '.py')
    runner.write_text(runner_text)
    before, after = verify(SHELL), verify(held)
    assert set(before) <= set(after), 'no parent files may be removed'
    proof.update(manifest=pin, parent_manifest=SHELL_SHA, parent_bundle=str(SHELL),
                 bundle_changed=sorted(n for n in before if before[n] != after[n]),
                 bundle_added=sorted(set(after) - set(before)), bundle_removed=[],
                 runner=dict(parent=str(shell_runner), parent_sha256=SHELL_RUNNER_SHA, sha256=sha(runner),
                             held_e_derivation_reproduces_delivered_runner=True,
                             differences=[line for line in difflib.ndiff(held_e_text.splitlines(), runner_text.splitlines())
                                          if line.startswith(('- ', '+ '))]),
                 queue_derivation='native41_progress_queue.derive(): held-e startup/bind/unarmed, eight progress arms '
                                  '(current/burst x live/solo, two rounds), typed FULL, one phase budget',
                 active_s=native41_progress_gate.ACTIVE_S, cleanup_margin_s=native41_progress_gate.CLEANUP_S)
    dump(out / 'freeze-proof.json', proof)
    result = dict(schema='native41-progress-build/1', name=CANDIDATE, arm='TRUNK', verdict='PASS',
                  source=git('rev-parse', 'HEAD').strip(), branch_base=git('rev-parse', PARENT).strip(),
                  bundle=str(held), manifest=pin, overlay=str(held / 'native30_overlay'),
                  overlay_manifest=sha(held / 'native30_overlay/SHA256SUMS'),
                  reference_bundle=str(parent), reference_overlay_manifest=sha(parent / 'native30_overlay/SHA256SUMS'),
                  parent_bundle=str(SHELL), parent_bundle_manifest=SHELL_SHA,
                  regression=reg, focused=focus, files=len(verify(held)), freeze_proof_sha256=sha(out / 'freeze-proof.json'),
                  held=True, box_ready=True, runner=str(runner), runner_sha256=sha(runner), runner_size=runner.stat().st_size,
                  expected_selfrun_tests=expected_tests(), engine_cases=engine_cases(), review_sha256=review_sha,
                  image=__import__('build_native39_trunk').IMAGE, checksummed_files=len(verify(held)),
                  queue=str(held / 'runs' / (STEM + '.draft.sh')), gpu_execution=False, automatic_native_admission=True,
                  progress_setting='FLASH_S5_AUTO_ARM=1 FLASH_S5_NATIVE_PROGRESS=burst in the queue EXTRA_ENV only')
    dump(out / 'build.json', result)
    for d in (held, parent):
        for p in [*d.rglob('*'), d]:
            p.chmod(0o555 if p.is_dir() else 0o444)
    runner.chmod(0o444)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--selftest-log', type=Path, required=True)
    p.add_argument('--focused-log', type=Path, required=True)
    p.add_argument('--prepare-review', action='store_true', help='write the prepared review into patches/s5_engine and stop')
    a = p.parse_args()
    if a.prepare_review:
        dump(ROOT / 'patches/s5_engine/native_candidate_review.json',
             prepare_review(a.selftest_log.resolve(), a.focused_log.resolve()))
    else:
        build(a.out.resolve(), a.selftest_log.resolve(), a.focused_log.resolve())
