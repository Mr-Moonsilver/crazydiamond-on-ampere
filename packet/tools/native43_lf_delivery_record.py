#!/usr/bin/env python3
"""Assemble the machine delivery records for the native43 l-f build (results/hub/native43-lifecycle-f/).

tools/native43_lifecycle_delivery_record.py (as run for l-e) for l-f: the same build, freeze, bundle self-run
(frozen-runner probe), engine self-run (with the CT-vs-container clock split), compile proof, precheck, strict and
focused bindings, with
  * the l-f runner lineage (native43-lf-runner-lineage/1: the box-run l-e runner, declared lines and engine rows);
  * the l-f clock scan (native43-lf-clock-scan/1: the lifecycle scan over the l-f texts, the declared death-scan
    resolution consumed, the two l-f negative controls flagged);
  * the identity check against the l-e build source (--le-source, e2826ca): the engine files outside the declared
    delta are l-e's bytes; the delta files (executor_hooks.py, row1_oracle.py, qsa_operation.py) are the source
    commit's and moved only in the declared commits (--delta-commit); the overlay e3/executor_hooks.py is the hooks
    the le-fix bounded-enqueue self-run exercised (--enqueue-selfrun) and the generated e3_sources/executor.py has
    exactly one bounded_enqueue site; the l-f queue texts are the source commit's and their derivation checks; the
    l-e texts in the bundle are the box-run bytes; harness/runs moved only by the three l-f texts.
Writes delivery.json, frozen-gates.json and strict.json (refusing to overwrite).  Every verdict is copied, never set.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native43_delivery_record import load, precheck_summary, sha  # noqa: E402
from native43_lifecycle_delivery_record import gate_window  # noqa: E402

NAME = 'q18a-native43-lifecycle-f'
LF_TEXTS = ('q18a_native43_lifecycle_f.draft.sh', 'q18a_native43_lifecycle_f_declare.sh', 'q18a_native43_lifecycle_f_settings.sh')
# The l-e texts as they ran on the box (bundle 296adc0886b6, run 1): the l-f derivation source.
LE_TEXTS = {'q18a_native43_lifecycle.draft.sh': 'b5a20401f677cae0cc48e5ab026e52c7aab9f456c4ebddd63d597cae7bda072c',
            'q18a_native43_lifecycle_declare.sh': 'b1daded8d3d61812f86d79515878cf439bf7112f3a39a5ee0b75715967187f9d',
            'q18a_native43_lifecycle_settings.sh': 'b01060b41d425e34f3543fc9bf6bbc34db8191af2d68da262da09e28761389f4'}
ENGINE_FILES = ('engine_driver.py', 'native_resources.py', 'qsa_driver.py', 'qsa_operation.py', 'executor_hooks.py',
                'row1_oracle.py')
DELTA_FILES = ('executor_hooks.py', 'qsa_operation.py', 'row1_oracle.py')
BOUNDED_SITE = '_flash_s5_hooks.bounded_enqueue('
REVIEW = 'patches/s5_engine/native_candidate_review.json'
FRESH_BOX_PATHS = tuple(p % NAME for p in (
    '/data/vllm/flash-next-tp4/prod/overlay-v16s5-%s', '/data/vllm/flash-next-tp4/prod/overlay-v16s5-%s-build',
    '/home/<service-user>/s0/%s', '/home/<service-user>/s0/%s-requests', '/home/<service-user>/s0/%s.log', '/home/<service-user>/s0/%s-declare.log',
    '/home/<service-user>/s0/%s-declare-chain.log', '/home/<service-user>/s0/%s-staging.json', '/home/<service-user>/s0/%s-build-logs'))


def identity(bundle, repo, le_source, source, delta_commits, enqueue_selfrun):
    def show(rev, path):
        return subprocess.run(['git', '-C', str(repo), 'show', '%s:%s' % (rev, path)], check=True, capture_output=True).stdout

    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True).stdout
    engine, delta = {}, {}
    for name in ENGINE_FILES:
        data = (bundle / 'e3_builder' / name).read_bytes()
        path = 'patches/s5_engine/' + name
        assert data == show(source, path), name
        if name in DELTA_FILES:
            assert data != show(le_source, path), ('declared delta file equals l-e', name)
            moved_in = git('log', '--format=%H', le_source + '..' + source, '--', path).split()
            assert moved_in and set(moved_in) <= set(delta_commits), (name, moved_in)
            delta[name] = dict(sha256=hashlib.sha256(data).hexdigest(),
                               le_sha256=hashlib.sha256(show(le_source, path)).hexdigest(), commits=moved_in)
        else:
            assert data == show(le_source, path), ('engine file outside the delta moved', name)
        engine[name] = hashlib.sha256(data).hexdigest()
    moved = git('diff', '--name-only', le_source, source).split()
    engine_moved = sorted(m for m in moved if m.startswith('patches/') and m != REVIEW)
    assert engine_moved == sorted('patches/s5_engine/' + n for n in DELTA_FILES), engine_moved
    # The review moved only by prepared-review commits (the l-e build's); the bundle carries l-f's own prepared review.
    review_commits = git('log', '--format=%H %s', le_source + '..' + source, '--', REVIEW).splitlines()
    assert all(' build (' in row and 'review prepared over' in row for row in review_commits), review_commits
    runs_moved = sorted(m for m in moved if m.startswith('harness/runs/'))
    assert runs_moved == sorted('harness/runs/' + n for n in LF_TEXTS), runs_moved
    overlay = bundle / 'native30_overlay'
    hooks = sha(overlay / 'e3/executor_hooks.py')
    enqueue = load(enqueue_selfrun)
    assert enqueue['verdict'] == 'PASS' and enqueue['enqueueprobe']['verdict'] == 'PASS'
    assert hooks == enqueue['packages']['fixed_executor_hooks_sha256'], 'overlay hooks are not the self-run hooks'
    # The overlay file is the generator's output over e3_builder/executor_hooks.py (package-relative imports), so it is
    # bound to the self-run's package bytes above, not to the builder source.
    sites = (overlay / 'e3_sources/executor.py').read_text().count(BOUNDED_SITE)
    assert sites == 1, sites
    queue = {}
    for name in LF_TEXTS:
        path = bundle / 'runs' / name
        assert path.read_bytes() == show(source, 'harness/runs/' + name), name
        queue[name] = sha(path)
    for name, pin in LE_TEXTS.items():
        assert sha(bundle / 'runs' / name) == pin, name
    derived = subprocess.run([sys.executable, '-B', str(Path(repo) / 'tools/native43_lf_queue_drafts.py'), '--check'],
                             capture_output=True, text=True)
    assert derived.returncode == 0, derived.stdout + derived.stderr
    return dict(le_source=le_source, source=source, engine_files=engine, declared_engine_delta=delta,
                delta_commits=list(delta_commits), source_paths_since_le=moved, review_commits_since_le=review_commits,
                overlay_executor_hooks_sha256=hooks, builder_executor_hooks_sha256=sha(bundle / 'e3_builder/executor_hooks.py'),
                enqueue_selfrun=str(enqueue_selfrun),
                enqueue_selfrun_sha256=sha(enqueue_selfrun), generated_executor_bounded_sites=sites,
                lf_queue_texts=queue, lf_queue_derivation=derived.stdout.strip(), le_box_run_texts=dict(LE_TEXTS),
                verdict='PASS')


def record(a):
    b, f, spec = load(a.build), load(a.freeze), load(a.spec)
    ident = identity(Path(b['bundle']), a.repo, a.le_source, spec['source_commit'], a.delta_commit, a.enqueue_selfrun)
    s, e, c = load(a.bundle_selfrun), load(a.engine_selfrun), load(a.compile_proof)
    lineage, k = load(a.runner_lineage), load(a.clock_scan)
    assert not spec['dry_run'] and b['name'] == spec['candidate'] == NAME and b['arm'] == spec['arm'] == 'NATIVE43L'
    assert f['verdict'] == 'PASS' and f['manifest'] == b['manifest'] and f['spec_sha256'] == sha(a.spec)
    assert s['verdict'] == 'PASS' and s['rc'] == 0 and s['bundle_manifest'] == b['manifest'] and s['skips'] == 0
    assert s['frozen_runner_sha256'] == b['runner_sha256'] and s['tests_run'] == b['expected_selfrun_tests']
    assert s['native35_arm'] == 'NATIVE43L' and s['native43_lf_queue'] == 'PASS' and s['native43_lifecycle_queue'] == 'PASS'
    assert s['native43_queue'] == 'PASS' and s['native39_trunk_queue'] == 'PASS'
    assert s['gpu_device_requests'] == 0 and s['overlay_manifest'] == b['overlay_manifest'] and s['image'] == b['image']
    assert s['owner_prelaunch']['verdict'] == 'PASS'
    assert e['schema'] == 'native43-selfrun/1' and e['label'] == b['name'] and e['bundle_manifest'] == b['manifest']
    assert e['verdict'] == 'PASS' and e['container_rc'] == 0 and e['probe39']['verdict'] == 'PASS'
    assert e['probe37']['verdict'] == 'PASS' and e['py_compile']['ok'] and e['suites']['ok'] and e['suites']['skipped'] == 0
    assert e['network'] == 'none' and e['gpu_device_requests'] == 0 and e['image'] == b['image']
    assert e['spec_sha256'] == sha(a.spec)
    assert set(spec['details']['engine_delta']['focused_suites']) - {'test_native43_lf_queue'} <= set(spec['engine_selfrun']['suites'])
    clock = e['clock_split']
    assert clock['verdict'] == 'PASS' and clock['outcomes'] == spec['clock_split']['expect'] and clock['side'] == spec['clock_split']['side']
    assert clock['outcomes']['native_window_clock'] == 'proved' and clock['outcomes']['native_window_clock_pre_fix'] == 'refused'
    assert clock['control_outcomes']['native_window_clock_pre_fix'] == 'proved' and clock['pre_fix']
    assert clock['method'] and clock['split']['boot_ids_differ'] and clock['split']['time_namespaces_differ']
    probes = {row['field']: e[row['field']] for row in spec['engine_selfrun'].get('probes', [])}
    assert all(v['verdict'] == 'PASS' and v['required_ok'] is True for v in probes.values()), probes
    assert lineage['schema'] == 'native43-lf-runner-lineage/1' and lineage['verdict'] == 'PASS'
    assert lineage['runner_sha256'] == b['runner_sha256'] and lineage['review_sha256'] == b['review_sha256']
    assert sorted(lineage['declared_engine_delta_vs_le']) == sorted(spec['details']['engine_delta']['lineage_rows'])
    assert c['verdict'] == 'PASS' and not c['failures'] and c['manifest'] == b['manifest'] and c['runner_sha256'] == b['runner_sha256']
    assert k['schema'] == 'native43-lf-clock-scan/1' and k['verdict'] == 'PASS' and not k['scan']['problems']
    assert k['bundle_manifest'] == b['manifest'] and k['bundle'] == str(Path(b['bundle']).resolve())
    assert k['runner']['sha256'] == b['runner_sha256'] and k['setcfg_reference']
    assert k['scan']['reuse_clock_uses_native43_prove'] is True and k['scan']['cli_resolution']['consumed'] is True
    assert 'native43_row1_qualification.py' in k['scan']['cli_resolution']['values']
    death = k['scan']['death_scan_resolution']
    assert death['consumed'] is True and death['entry_in_closure'] is True
    assert k['negative_controls'] and all(k['negative_controls'].values()) and not k['scan']['direct_references']
    assert len(k['negative_controls']) == 7
    precheck = precheck_summary(a.precheck_log)
    strict = dict(b['regression'], log=str(a.strict_log), verdict='PASS', gpu_execution=False,
                  scope='full strict registered CPU regression on the native43 l-f build tree')
    assert strict['log_sha256'] == sha(a.strict_log)
    focused = dict(b['focused'], log=str(a.focused_log))
    assert focused['log_sha256'] == sha(a.focused_log)
    provenance = load(Path(b['bundle']) / 'native30-provenance.json')
    window = gate_window(Path(b['bundle']))
    assert window['active'] + window['margin'] == s['native43_lifecycle_window_active_and_cleanup_s']
    queue_check = Path(a.bundle_selfrun).with_name('native43-lifecycle-queue-check.json')
    lf_check = Path(a.bundle_selfrun).with_name('native43-lf-queue-check.json')
    runner = Path(b['runner']).name
    row = dict(
        name=b['name'], arm=b['arm'], label=spec['label'], source=b['source'], source_commit=spec['source_commit'],
        details=provenance.get('native43'), bundle=b['bundle'], manifest=b['manifest'], overlay_manifest=b['overlay_manifest'],
        coverage=b['checksummed_files'], review_sha256=b['review_sha256'], image=b['image'],
        runner=b['runner'], runner_sha256=b['runner_sha256'], runner_size=b['runner_size'],
        runner_parent=dict(path=f['runner']['parent'], sha256=f['runner']['parent_sha256']),
        runner_differences=f['runner']['differences'], runner_swaps=len(f['runner']['swaps']), runner_phases=f['runner']['phases'],
        parent=dict(manifest=f['parent_manifest'], commit=f['parent_commit'], review_sha256=f['parent_review_sha256']),
        freeze_proof=str(a.freeze), freeze_proof_sha256=sha(a.freeze), spec=str(a.spec), spec_sha256=sha(a.spec),
        freeze_differences=dict(bundle_changed=f['changed'], bundle_added=f['added'], bundle_removed=f['removed'],
                                overlay_moved=f['overlay']['moved'], engine_changed=f['engine_changed']),
        source_definitions={k2: v['changed'] for k2, v in f['freeze']['source']['definitions'].items()},
        overlay_definitions={k2: v['changed'] for k2, v in f['overlay']['definitions'].items()},
        source_compile=f['freeze']['source']['compile'], negative_controls=f['freeze']['negative_controls'],
        bundle_selfrun=str(a.bundle_selfrun), bundle_selfrun_sha256=sha(a.bundle_selfrun),
        bundle_selfrun_tests=s['tests_run'], bundle_selfrun_skips=s['skips'], bundle_selfrun_container=s['container'],
        bundle_selfrun_wall_s=s['wall_s'],
        lifecycle_queue_cases=sorted(r['case'] for r in s['native43_lifecycle_queue_receipts']),
        lifecycle_queue_check=str(queue_check), lifecycle_queue_check_sha256=sha(queue_check),
        lf_queue_cases=sorted(r['case'] for r in s['native43_lf_queue_receipts']),
        lf_queue_check=str(lf_check), lf_queue_check_sha256=sha(lf_check),
        native43_queue_cases=len(s['native43_queue_cases']), owner_prelaunch=s['owner_prelaunch'],
        engine_selfrun=str(a.engine_selfrun), engine_selfrun_sha256=sha(a.engine_selfrun),
        engine_selfrun_view=dict(probe39=e['probe39']['verdict'], probe37=e['probe37']['verdict'], py_compile=e['py_compile'],
                                 suites=e['suites'], network=e['network'], gpu_device_requests=e['gpu_device_requests']),
        clock_split=clock, engine_selfrun_probes=probes,
        runner_lineage=str(a.runner_lineage), runner_lineage_sha256=sha(a.runner_lineage),
        runner_lineage_view={k2: lineage[k2] for k2 in ('box_runner', 'box_runner_sha256', 'box_runner_lineage',
                                                        'box_runner_lineage_sha256', 'declared_differences',
                                                        'metadata_rows_moved_vs_le', 'le_review_sha256')},
        runner_lineage_engine_delta=lineage['declared_engine_delta_vs_le'],
        engine_queue_identity=ident,
        clock_scan=str(a.clock_scan), clock_scan_sha256=sha(a.clock_scan),
        clock_scan_view=dict(verdict=k['verdict'], client_units=len(k['scan']['client_units']), roots=len(k['scan']['roots']),
                             closure=len(k['scan']['closure']), invocations=len(k['scan']['invocations']),
                             reuse_clock_uses_native43_prove=k['scan']['reuse_clock_uses_native43_prove'],
                             cli_resolution=k['scan']['cli_resolution'], death_scan_resolution=death,
                             negative_controls=sorted(k['negative_controls'])),
        compile_proof=str(a.compile_proof), compile_proof_sha256=sha(a.compile_proof), compile_files=c['files'],
        compile_failures=len(c['failures']),
        precheck=precheck, precheck_conf=str(a.precheck_conf), precheck_conf_sha256=sha(a.precheck_conf),
        precheck_log=str(a.precheck_log), precheck_log_sha256=sha(a.precheck_log),
        lifecycle_bound_s=window['active'] + window['margin'], latest_entry_utc=window['latest'],
        kept_rows=window['kept'], dropped_rows=window['dropped'], expected_runtime=window['expected'],
        fresh_box_paths=list(FRESH_BOX_PATHS) + ['/home/<service-user>/harness/' + b['manifest'][:12]],
        owner_commands=['sudo python3 %s --staging --auth-env-file PRIVATE_AUTH_FILE --selfrun-record EXACT_BUNDLE_SELFRUN_JSON '
                        '--engine-selfrun-record EXACT_ENGINE_SELFRUN_JSON --snapshot-max-bytes 536870912 --sync none' % runner,
                        'owner stops production (vllm-flash inactive/failed, four idle GPUs), then: sudo python3 %s --arm' % runner],
        note=str(a.note), note_sha256=sha(a.note), strict=strict, focused=focused, verdict='PASS', classification=None,
        gpu_execution=False, box_action=False,
        scope='frozen input gates, the fresh strict regression, the pinned-image CPU self-runs (bundle probe, engine + '
              'clock split), the l-f runner lineage proof and the static client-clock scan; no device run; row1 under '
              'dense, cancel/drain/reuse, loud engine death and timing come only from the box run')
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (('delivery.json', row), ('frozen-gates.json', {k2: v for k2, v in row.items() if k2 not in ('strict', 'focused')}),
                        ('strict.json', strict)):
        with (out / name).open('x') as fh:
            json.dump(value, fh, sort_keys=True, indent=2)
            fh.write('\n')
    print(json.dumps(dict(out=str(out), verdict='PASS', manifest=b['manifest'][:12], precheck=precheck,
                          latest_entry_utc=row['latest_entry_utc'], delivery_sha256=sha(out / 'delivery.json')), sort_keys=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('spec', 'build', 'freeze', 'bundle-selfrun', 'engine-selfrun', 'runner-lineage', 'clock-scan', 'compile-proof',
                 'precheck-conf', 'precheck-log', 'strict-log', 'focused-log', 'note', 'out', 'enqueue-selfrun'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--le-source', required=True, help='the l-e build source commit (e2826ca)')
    p.add_argument('--delta-commit', action='append', required=True, help='a commit allowed to move a declared delta file')
    record(p.parse_args())
