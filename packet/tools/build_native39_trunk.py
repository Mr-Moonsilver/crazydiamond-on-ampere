#!/usr/bin/env python3
"""Native39 trunk held bundle and owner runner.

Bundle: the native38 plain-bind held-b bundle (d7a0943b...) plus the native39
tree: the P1/P2 engine sources (scheduler_hooks, engine_driver,
native_overlap_session) and their review, the trunk queue texts, checks,
window, W2 manifest, plain W1 client, suites and fixture leaves, and the trunk
tools.  The overlay is regenerated from BASE exactly as native38 did
(make_s5_overlap.py --native-candidate with the prepared and committed review,
then e3_finalize_package.py); the owner-qualified compiled libraries are kept
byte-identical.

Runner: the frozen P-c owner runner (fe194172a6c8, run on the box 31 Oct 2026)
with names, the bundle/review/overlay pins (H, BUNDLE, REVIEW,
EXPECTED_SOURCES), the trunk window import, the trunk self-run gate and a
second required record (the native39 engine self-run of this bundle) changed;
every other byte is asserted equal after normalisation.  LAUNCHER, OBSERVER,
BASE, IMAGE, the found launcher/defaults pins and every staging/arm step stay
P-c's.

Usage: --prepare-review --selftest-log L --focused-log F  (writes the review)
       --out DIR --selftest-log L --focused-log F          (builds, held read-only)
"""
import argparse
import ast
import difflib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path('<hub-dir>/agents/flash-next/tmp/native38-plain-bind-held-b/q18a-native38-plain-bind-bundle-d7a0943b90ab')
PARENT_MANIFEST = 'd7a0943b90ab9dac43ef206b4d160e27e74db39c0a07685a87728435a4f98468'
PARENT_OVERLAY_MANIFEST = '0ba36718d246d712609171d1ff31b4c9ab7c6cb994232b0ad5cf345db940b4b0'
PARENT_REVIEW = '9527d33cd0b94825140a1c2388ce48cdb51cc341910ed22662c08dacc76bd711'
PARENT_BUNDLE_SOURCE = 'b2c4ec7420e1411b35640fe5af4b64d73f15bb86'
PARENT_CANDIDATE = 'q18a-native38-plain-bind'
BRANCH_BASE = '72ea0baa208e1cfc04446898bbd58a0f3c4f84b8'
BASE = Path('<hub-dir>/agents/flash-next/tmp/e2f-native-offline-base-20261027T211405Z')
BASE_SHA = 'da561411fadbb82922eeb8eafbe9424bc9703aaf8c2030bd813743c692b641a5'
P_C_RUNNER = Path('<hub-dir>/agents/flash-next/tmp/native35-p-held-c/q18a-native35-p-c-owner-run-fe194172a6c8.py')
P_C_RUNNER_SHA = 'da0a936cd186aea7a0bb9c32597738c01e996df9a08be69a3aa02bc776c9181e'
IMAGE = 'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
CANDIDATE, ARM, STEM = 'q18a-native39-trunk', 'TRUNK', 'q18a_native39_trunk'
ENGINE_CHANGED = {'core/scheduler.py', 'e3/engine_driver.py', 'e3/native_overlap_session.py', 'e3/scheduler_hooks.py'}
METADATA = {'S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json', 'e3/native-overlap-readiness.json',
            'e3/decode-probe-evidence.json', 'core/SHA256SUMS'}
SOURCE_CHANGED = {'engine_driver.py', 'native_overlap_session.py', 'scheduler_hooks.py', 'native_candidate_review.json'}
HARNESS_CHANGED = {'selftest.py'}
HARNESS_ADDED = {'e3_native39_trunk_checks.py', 'e3_native39_trunk_probe.py', 'e3_native39_w2_length.py',
                 'fixtures/native39_box_extract.json', 'native39_fixture_leaves.json', 'native39_trunk_gate.py',
                 'runs/q18a_native39_trunk.draft.sh', 'runs/q18a_native39_trunk_declare.sh',
                 'runs/q18a_native39_trunk_settings.sh', 'test_native39_trunk.py', 'test_native39_trunk_queue.py',
                 'w1_trace.py', 'workloads/q18a-w2-overlap-native39.json'}
TOOLS = ('build_native39_trunk.py', 'native39_trunk_freeze.py', 'native39_trunk_declarations.json',
         'native39_trunk_cpu_probe.py', 'native39_trunk_bundle_selfrun.py', 'native39_trunk_tools_derive.py',
         'native39_trunk_selfrun.py', 'native39_trunk_queue_drafts.py', 'native39_w2_length.py', 'native39_box_extract.py')
PLUGINS = ('native30_host.py', 'native30_policy.py', 'native34_diag.py', 'native34_worker.py', 'native34_control.py',
           'native34_storage.py', 'native34_indexer.py', 'native34_readback.py')
BUNDLE_CHANGED = ({'native30-provenance.json', 'native30-review.json', 'e3_builder/native_candidate_review.json'}
                  | {'e3_builder/' + n for n in SOURCE_CHANGED - {'native_candidate_review.json'}}
                  | HARNESS_CHANGED
                  | {'native30_overlay/' + n for n in ENGINE_CHANGED | METADATA | {'SHA256SUMS'}})
CHECKS = (('tools/native39_trunk_queue_drafts.py', '--check'), ('tools/native39_trunk_tools_derive.py', '--check'),
          ('tools/native39_w2_length.py', '--check'))
sys.path[:0] = [str(ROOT / 'patches/s5_engine'), str(ROOT / 'tools'), str(ROOT / 'harness')]
from build_native30_held import sha, dump, manifest, verify, replace  # noqa: E402
from native_readiness import digest, payload_identity, verify_finalized_package  # noqa: E402
import native39_trunk_freeze as freeze  # noqa: E402
import native39_trunk_gate as window  # noqa: E402
import e3_native39_w2_length as w2length  # noqa: E402

# Runner swaps (P-c text with names already changed -> trunk text). Each must match exactly once.
P_C_GATE = ("result.get('tests_run',0)==183 and result.get('final_input_proof') is True and result.get('draft_window_slots')=='PASS' "
            "and result.get('native34_protocol')=='PASS' and result.get('native34_readback')=='PASS' and result.get('native35_builder')=='PASS' "
            "and result.get('native35_arm')=='P-C' and result.get('native35_performance')=='PASS' and result.get('native35_row4_l2')=='PASS' "
            "and result.get('native35_localize_off')=='PASS' and result.get('native35_p_c_row4')=='PASS' and result.get('native35_p_c_continuation')=='PASS'")
RUNNER_SWAPS = (
    ('    from native35p_gate import speed_window_ok as window_ok\n',
     '    from native39_trunk_gate import trunk_window_ok as window_ok\n'),
    ("    p.add_argument('--auth-env-file',type=Path);p.add_argument('--selfrun-record',type=Path)\n",
     "    p.add_argument('--auth-env-file',type=Path);p.add_argument('--selfrun-record',type=Path)\n"
     "    p.add_argument('--engine-selfrun-record',type=Path)\n"),
    ("        require(a.auth_env_file is not None and a.selfrun_record is not None,'staging-auth-and-selfrun-record-required')\n",
     "        require(a.auth_env_file is not None and a.selfrun_record is not None and a.engine_selfrun_record is not None,\n"
     "                'staging-auth-and-selfrun-records-required')\n"),
    ("                and result.get('identity_timeout_negative_rejected') is True,'exact-frozen-selfrun-required')\n",
     "                and result.get('identity_timeout_negative_rejected') is True,'exact-frozen-selfrun-required')\n"
     "        # Native39: the engine self-run of this bundle (R/W/P, native37 T1/T2, suites) in the pinned CPU image.\n"
     "        engine=json.loads(a.engine_selfrun_record.read_text())\n"
     "        require(engine.get('schema')=='native39-trunk-selfrun/1' and engine.get('label')==QNAME\n"
     "                and engine.get('bundle_manifest')==BUNDLE and engine.get('container_rc')==0\n"
     "                and (engine.get('probe39') or {}).get('verdict')=='PASS' and (engine.get('probe37') or {}).get('verdict')=='PASS'\n"
     "                and (engine.get('py_compile') or {}).get('ok') is True\n"
     "                and (engine.get('suites') or {}).get('ok') is True and (engine.get('suites') or {}).get('skipped')==0\n"
     "                and engine.get('gpu_device_requests')==0 and engine.get('network')=='none'\n"
     "                and engine.get('image')==IMAGE and engine.get('box_action') is False,'exact-engine-selfrun-required')\n"),
    ("            selfrun_sha256=sha(a.selfrun_record),declaration_sha256=",
     "            selfrun_sha256=sha(a.selfrun_record),engine_selfrun_sha256=sha(a.engine_selfrun_record),declaration_sha256="),
)


def expected_tests():
    tree = ast.parse((ROOT / 'tools/native39_trunk_cpu_probe.py').read_text())
    values = [n.value.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id == 'EXPECTED_TESTS']
    assert len(values) == 1 and type(values[0]) is int
    return values[0]


def trunk_gate():
    return (P_C_GATE.replace('==183', '==%d' % expected_tests())
            .replace("result.get('native35_arm')=='P-C'", "result.get('native35_arm')=='TRUNK'")
            + " and result.get('native39_trunk_queue')=='PASS'")


def regression_from(log):
    text = log.read_text()
    summary = re.findall(r'^selftest: (\d+)/(\d+) cases as required$', text, re.M)
    suites = [json.JSONDecoder().raw_decode(text[m.end():])[0] for m in re.finditer('selftest suite: ', text)]
    assert (len(summary) == 1 and summary[0][0] == summary[0][1] and suites
            and all(s['returncode'] == 0 and not s['skipped'] for s in suites)), \
        'strict regression log must be complete, all-pass, zero skips'
    return dict(cases=int(summary[0][0]), required=int(summary[0][1]), exit_code=0, skips=0, suites=len(suites),
                reported_tests=sum(s['tests_run'] or 0 for s in suites), log_sha256=sha(log),
                basis='fresh full strict registered CPU regression on the native39 trunk final tree')


def focused_from(log):
    text = log.read_text()
    ran = re.findall(r'^Ran (\d+) tests? in [0-9.]+s\n\nOK$', text, re.M)
    assert (len(ran) == 1 and not re.search(r'\.\.\. skipped|^FAILED|^FAIL:|^ERROR:', text, re.M)
            and text.rstrip().endswith('rc=0')), 'focused log must be one all-pass run with zero skips'
    return dict(tests=int(ran[0]), skips=0, verdict='PASS', log_sha256=sha(log),
                basis='native39 trunk and queue suites plus native37/native38 and the admission, staged-plan, overlap, '
                      'storage-replay, worker-bridge, metadata, P-c queue/continuation, host-guard and identity suites')


def source_head():
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'harness', 'tools', 'patches/s5_engine'],
                                    cwd=ROOT, text=True)
    assert not dirty, 'commit harness/tools/patches before building:\n' + dirty
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def details():
    manifest_decl = json.loads((ROOT / 'harness/workloads/q18a-w2-overlap-native39.json').read_text())['length_declaration']
    return dict(candidate=CANDIDATE, arm=ARM, parent=PARENT_CANDIDATE, parent_manifest=PARENT_MANIFEST,
        branch_base=BRANCH_BASE, engine_files=sorted(ENGINE_CHANGED),
        scope='native37 robust + native38 plain bind + native39 P1/P2 (positive-frontier and running stock prefills held on '
              'an armed split route, typed empty-window prefill_end refusal) under the P-c speed queue',
        queue='q18a-native39-trunk: P-c with a plannable W2 length (guard before row1), loopback API server, bind receipt, '
              'plain W1 control with an idle precondition for W2 and a descriptive trunk summary',
        w2=dict(manifest='workloads/q18a-w2-overlap-native39.json',
                fresh={k: manifest_decl['fresh'][k] for k in ('prompt_tokens_declared', 'tokens', 'residue', 'gap', 'plans')},
                append={k: manifest_decl['append'][k] for k in ('prompt_tokens_declared', 'tokens', 'residue', 'gap', 'plans')},
                rule=manifest_decl['rule']['refused_when'], max_prompt_tokens=w2length.shared.MAX_PROMPT_TOKENS),
        window=dict(not_before=window.NOT_BEFORE, hard_end=window.HARD_END, active_s=window.TRUNK_ACTIVE_S,
                    cleanup_margin_s=window.TRUNK_CLEANUP_MARGIN_S,
                    active_derivation=[list(r) for r in window.ACTIVE_DERIVATION]),
        model_settings_changed=False, tail_handback=False, device_run='NOT_RUN',
        fixture_leaves='native39_fixture_leaves.json', owner_runner='q18a-native39-trunk-owner-run-<bundle12>.py',
        limitations='CPU fixture resources do not establish CUDA exactness, device timing or the bind line on the box')


def prepare_review(strict_log, focused_log):
    review = json.loads((ROOT / 'patches/s5_engine/native_candidate_review.json').read_text())
    tracked = subprocess.check_output(['git', 'ls-files', 'harness'], cwd=ROOT, text=True).splitlines()
    review['source_shas'] = {n: sha(ROOT / 'patches/s5_engine' / n) for n in review['source_shas']}
    review['harness_source_shas'] = {str(Path(n).relative_to('harness')): sha(ROOT / n)
                                     for n in tracked if (ROOT / n).is_file()}
    review['cpu_regression'] = regression_from(strict_log)
    review['post_review_cpu'] = focused_from(focused_log)
    review['native39_trunk'] = details()
    return review


def derive_runner(pin, review_sha, sources):
    original = P_C_RUNNER.read_text()
    owner = original.replace('q18a_native35_p_c', STEM).replace('q18a-native35-p-c', CANDIDATE)
    assert owner.count(P_C_GATE) == 1
    owner = owner.replace(P_C_GATE, trunk_gate())
    for old, new in RUNNER_SWAPS:
        assert owner.count(old) == 1, old[:80]
        owner = owner.replace(old, new)
    owner = replace(owner, dict(H="Path('/home/<service-user>/harness/" + pin[:12] + "')", BUNDLE=repr(pin), REVIEW=repr(review_sha),
                                EXPECTED_SOURCES=repr(sources)))
    compile(owner, CANDIDATE + '-owner-run', 'exec')
    # Normalise back: every other byte is P-c's.
    normalized = owner
    for old, new in reversed(RUNNER_SWAPS):
        normalized = normalized.replace(new, old)
    normalized = normalized.replace(trunk_gate(), P_C_GATE).replace(STEM, 'q18a_native35_p_c').replace(CANDIDATE, 'q18a-native35-p-c')
    assignments = {n.targets[0].id: ast.get_source_segment(original, n.value) for n in ast.parse(original).body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert replace(normalized, {k: assignments[k] for k in ('H', 'BUNDLE', 'REVIEW', 'EXPECTED_SOURCES')}) == original, \
        'unexpected runner body change'
    diff = [line for line in difflib.ndiff(original.splitlines(), owner.splitlines()) if line.startswith(('- ', '+ '))
            and not line[2:].startswith('EXPECTED_SOURCES=')]
    return owner, diff


def build(out, strict_log, focused_log):
    assert not out.exists()
    assert sha(PARENT / 'SHA256SUMS') == PARENT_MANIFEST and sha(BASE / 'SHA256SUMS') == BASE_SHA
    assert sha(P_C_RUNNER) == P_C_RUNNER_SHA
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
    review = prepare_review(strict_log, focused_log)
    assert json.loads((ROOT / 'patches/s5_engine/native_candidate_review.json').read_text()) == review, \
        'prepare and commit the native39 source review before the build'
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
    # Regenerate the overlay from BASE exactly as build_native38_plain_bind does.
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
    assert provenance['candidate'] == PARENT_CANDIDATE and provenance['overlay_manifest'] == PARENT_OVERLAY_MANIFEST
    provenance.update(source=source, candidate=CANDIDATE, native35_arm=ARM, native39_trunk=details(), selftest=regression,
                      post_review_cpu=focused, overlay_manifest=sha(overlay / 'SHA256SUMS'),
                      review_sha256=closure['review_sha256'], parent_bundle_manifest=PARENT_MANIFEST,
                      engine_changed=engine_changed, engine_changed_vs_native38_plain_bind=engine_changed)
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
    overlay_rows = verify(held / 'native30_overlay')
    sources = dict(sorted(overlay_rows.items()))
    owner, runner_diff = derive_runner(pin, closure['review_sha256'], sources)
    runner = out / (CANDIDATE + '-owner-run-' + pin[:12] + '.py')
    runner.write_text(owner)
    proof = dict(verdict='PASS', classification=None, source=source, branch_base=BRANCH_BASE, parent_bundle=str(PARENT),
                 parent_manifest=PARENT_MANIFEST, parent_bundle_source=PARENT_BUNDLE_SOURCE, manifest=pin,
                 changed=bundle_changed, added=new, removed=removed, overlay_regenerated=True,
                 overlay_manifest=sha(held / 'native30_overlay/SHA256SUMS'), parent_overlay_manifest=PARENT_OVERLAY_MANIFEST,
                 overlay_changed=changed, engine_changed=engine_changed,
                 freeze=dict(source=source_proof, overlay=overlay_proof, negative_controls=negatives),
                 compiled_libraries_byte_identical=True, plugins_byte_identical=True, native_sources_byte_identical=True,
                 review_sha256=closure['review_sha256'], parent_review_sha256=PARENT_REVIEW,
                 historical_vm_snapshots_byte_identical=True,
                 runner=dict(path=str(runner), sha256=sha(runner), parent=str(P_C_RUNNER), parent_sha256=P_C_RUNNER_SHA,
                             differences=runner_diff, expected_selfrun_tests=expected_tests(),
                             only_names_pins_window_selfrun_gates_changed=True),
                 gpu_execution=False, box_action=False, details=details())
    dump(out / 'freeze-proof.json', proof)
    result = dict(name=CANDIDATE, arm=ARM, bundle=str(held), manifest=pin, overlay=str(held / 'native30_overlay'),
                  overlay_manifest=sha(held / 'native30_overlay/SHA256SUMS'), review_sha256=closure['review_sha256'],
                  runner=str(runner), runner_sha256=sha(runner), runner_size=runner.stat().st_size,
                  checksummed_files=len(rows), source=source, held=True, regression=regression, focused=focused,
                  expected_selfrun_tests=expected_tests(), image=IMAGE, parent_manifest=PARENT_MANIFEST,
                  engine_changed=engine_changed, freeze_proof_sha256=sha(out / 'freeze-proof.json'))
    dump(out / 'build.json', result)
    for path in [held, *held.rglob('*'), runner]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    print(json.dumps({k: result[k] for k in ('name', 'manifest', 'overlay_manifest', 'review_sha256', 'runner_sha256',
                                             'expected_selfrun_tests')}, sort_keys=True))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
