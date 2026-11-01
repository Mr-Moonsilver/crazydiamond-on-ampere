#!/usr/bin/env python3
"""Assemble the machine delivery records for the native43 lifecycle build (results/hub/native43-<label>/).

tools/native43_delivery_record.py (as run for n43-f) for the single-queue lifecycle: the same build, freeze,
bundle self-run (frozen-runner probe), engine self-run (with the CT-vs-container clock split), compile proof,
precheck, strict and focused bindings; the runner lineage proof replaces the pair's owner-chain seam (single-queue
runner); the lifecycle client-clock scan replaces the pair's; the identity check binds the four engine files to
Codex d9fb80e/109313a (the n43-f engine) and the three lifecycle queue texts to the source commit and their
derivation.  Writes delivery.json, frozen-gates.json and strict.json (refusing to overwrite).  Every verdict is
copied, never set.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native43_delivery_record import ENGINE_FILES, load, precheck_summary, sha  # noqa: E402

LIFECYCLE_TEXTS = ('q18a_native43_lifecycle.draft.sh', 'q18a_native43_lifecycle_declare.sh',
                   'q18a_native43_lifecycle_settings.sh')
# Source paths that may differ from the Codex engine source: the prepared review (n43-f's, carried) and the
# lifecycle packaging (no engine file, no native43 pair queue text).
REVIEW = 'patches/s5_engine/native_candidate_review.json'
NAME = 'q18a-native43-lifecycle'
# The derivation source: the native39 lifecycle texts as they ran on the box (held-a run 1 bundle).
N39L_TEXTS = ('q18a_native39_lifecycle.draft.sh', 'q18a_native39_lifecycle_declare.sh', 'q18a_native39_lifecycle_settings.sh')
N39L_BUNDLE = Path('<hub-dir>/agents/flash-next/tmp/native39-lifecycle-held-a/q18a-native39-lifecycle-bundle-b90f1e292956')
FRESH_BOX_PATHS = tuple(p.replace('native39-lifecycle', 'native43-lifecycle') for p in (
    '/data/vllm/flash-next-tp4/prod/overlay-v16s5-q18a-native39-lifecycle',
    '/data/vllm/flash-next-tp4/prod/overlay-v16s5-q18a-native39-lifecycle-build',
    '/home/<service-user>/s0/q18a-native39-lifecycle', '/home/<service-user>/s0/q18a-native39-lifecycle-requests',
    '/home/<service-user>/s0/q18a-native39-lifecycle.log', '/home/<service-user>/s0/q18a-native39-lifecycle-declare.log',
    '/home/<service-user>/s0/q18a-native39-lifecycle-declare-chain.log',
    '/home/<service-user>/s0/q18a-native39-lifecycle-staging.json',
    '/home/<service-user>/s0/q18a-native39-lifecycle-build-logs'))


def gate_window(bundle):
    """The lifecycle window from the bundle's own gate modules, in an isolated interpreter (-I -B: nothing is written)."""
    code = ('import json,sys; sys.path.insert(0, sys.argv[1]); import native43_lifecycle_gate as g, native34_gate as w; '
            'print(json.dumps(dict(active=g.LIFECYCLE_ACTIVE_S, margin=g.LIFECYCLE_CLEANUP_MARGIN_S, hard_end=w.HARD_END, '
            'latest=g.LATEST_ENTRY_UTC, kept=list(g.KEPT), dropped=list(g.DROPPED), expected=g.EXPECTED_RUNTIME_S, '
            'same=(g.HARD_END == w.HARD_END and g.NOT_BEFORE == w.NOT_BEFORE))))')
    out = json.loads(subprocess.run([sys.executable, '-I', '-B', '-c', code, str(bundle)], check=True,
                                    capture_output=True, text=True).stdout)
    assert out['same'], out
    return out


def identity(bundle, repo, engine_base, source):
    def show(rev, path):
        return subprocess.run(['git', '-C', str(repo), 'show', '%s:%s' % (rev, path)], check=True, capture_output=True).stdout
    engine = {}
    for name in ENGINE_FILES:
        data = (bundle / 'e3_builder' / name).read_bytes()
        assert data == show(engine_base, 'patches/s5_engine/' + name) == show(source, 'patches/s5_engine/' + name), name
        engine[name] = hashlib.sha256(data).hexdigest()
    queue = {}
    for name in LIFECYCLE_TEXTS:
        path = bundle / 'runs' / name
        assert path.read_bytes() == show(source, 'harness/runs/' + name), name
        queue[name] = sha(path)
    pair = {}
    for path in sorted((bundle / 'runs').glob('q18a_native43_*')):
        if path.name not in LIFECYCLE_TEXTS:
            assert path.read_bytes() == show(engine_base, 'harness/runs/' + path.name), path.name
            pair[path.name] = sha(path)
    moved = subprocess.run(['git', '-C', str(repo), 'diff', '--name-only', engine_base, source], check=True,
                           capture_output=True, text=True).stdout.split()
    engine_moved = [m for m in moved if m.startswith('patches/')]
    runs_moved = [m for m in moved if m.startswith('harness/runs/')]
    assert set(engine_moved) <= {REVIEW}, engine_moved
    assert set(runs_moved) == {'harness/runs/' + n for n in LIFECYCLE_TEXTS + N39L_TEXTS}, runs_moved
    box_run = {}
    for name in N39L_TEXTS:
        data = show(source, 'harness/runs/' + name)
        assert data == (N39L_BUNDLE / 'runs' / name).read_bytes(), name
        box_run[name] = hashlib.sha256(data).hexdigest()
    derived = subprocess.run([sys.executable, '-B', str(Path(repo) / 'tools/native43_lifecycle_queue_drafts.py'), '--check'],
                             capture_output=True, text=True)
    assert derived.returncode == 0, derived.stdout + derived.stderr
    return dict(engine_base=engine_base, source=source, engine_files=engine, lifecycle_queue_texts=queue,
                pair_queue_texts_equal_engine_base=pair, lifecycle_queue_derivation=derived.stdout.strip(),
                derivation_source_texts_equal_n39l_box_run=dict(bundle=str(N39L_BUNDLE), texts=box_run),
                source_paths_since_engine_base=moved, verdict='PASS')


def record(a):
    b, f, spec = load(a.build), load(a.freeze), load(a.spec)
    ident = identity(Path(b['bundle']), a.repo, a.engine_base, spec['source_commit'])
    s, e, c = load(a.bundle_selfrun), load(a.engine_selfrun), load(a.compile_proof)
    lineage, k = load(a.runner_lineage), load(a.clock_scan)
    assert not spec['dry_run'] and b['name'] == spec['candidate'] == NAME and b['arm'] == spec['arm'] == 'NATIVE43L'
    assert f['verdict'] == 'PASS' and f['manifest'] == b['manifest'] and f['spec_sha256'] == sha(a.spec)
    assert s['verdict'] == 'PASS' and s['rc'] == 0 and s['bundle_manifest'] == b['manifest'] and s['skips'] == 0
    assert s['frozen_runner_sha256'] == b['runner_sha256'] and s['tests_run'] == b['expected_selfrun_tests']
    assert s['native35_arm'] == 'NATIVE43L' and s['native43_lifecycle_queue'] == 'PASS'
    assert s['native43_queue'] == 'PASS' and s['native39_trunk_queue'] == 'PASS'
    assert s['gpu_device_requests'] == 0 and s['overlay_manifest'] == b['overlay_manifest'] and s['image'] == b['image']
    assert s['owner_prelaunch']['verdict'] == 'PASS'
    assert e['schema'] == 'native43-selfrun/1' and e['label'] == b['name'] and e['bundle_manifest'] == b['manifest']
    assert e['verdict'] == 'PASS' and e['container_rc'] == 0 and e['probe39']['verdict'] == 'PASS'
    assert e['probe37']['verdict'] == 'PASS' and e['py_compile']['ok'] and e['suites']['ok'] and e['suites']['skipped'] == 0
    assert e['network'] == 'none' and e['gpu_device_requests'] == 0 and e['image'] == b['image']
    clock = e['clock_split']
    assert clock['verdict'] == 'PASS' and clock['outcomes'] == spec['clock_split']['expect'] and clock['side'] == spec['clock_split']['side']
    assert clock['outcomes']['native_window_clock'] == 'proved'
    assert clock['method'] and clock['split']['boot_ids_differ'] and clock['split']['time_namespaces_differ']
    probes = {row['field']: e[row['field']] for row in spec['engine_selfrun'].get('probes', [])}
    assert all(v['verdict'] == 'PASS' and v['required_ok'] is True for v in probes.values()), probes
    assert lineage['schema'] == 'native43-lifecycle-runner-lineage/1' and lineage['verdict'] == 'PASS'
    assert lineage['runner_sha256'] == b['runner_sha256'] and lineage['review_sha256'] == b['review_sha256']
    assert c['verdict'] == 'PASS' and not c['failures'] and c['manifest'] == b['manifest'] and c['runner_sha256'] == b['runner_sha256']
    assert k['schema'] == 'native43-lifecycle-clock-scan/1' and k['verdict'] == 'PASS' and not k['scan']['problems']
    assert k['bundle_manifest'] == b['manifest'] and k['bundle'] == str(Path(b['bundle']).resolve())
    assert k['runner']['sha256'] == b['runner_sha256'] and k['setcfg_reference']
    assert k['scan']['reuse_clock_uses_native43_prove'] is True and k['scan']['cli_resolution']['consumed'] is True
    assert k['negative_controls'] and all(k['negative_controls'].values()) and not k['scan']['direct_references']
    precheck = precheck_summary(a.precheck_log)
    strict = dict(b['regression'], log=str(a.strict_log), verdict='PASS', gpu_execution=False,
                  scope='full strict registered CPU regression on the native43 lifecycle build tree')
    assert strict['log_sha256'] == sha(a.strict_log)
    focused = dict(b['focused'], log=str(a.focused_log))
    assert focused['log_sha256'] == sha(a.focused_log)
    provenance = load(Path(b['bundle']) / 'native30-provenance.json')
    window = gate_window(Path(b['bundle']))
    assert window['active'] + window['margin'] == s['native43_lifecycle_window_active_and_cleanup_s']
    queue_check = Path(a.bundle_selfrun).with_name('native43-lifecycle-queue-check.json')
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
        native43_queue_cases=len(s['native43_queue_cases']), owner_prelaunch=s['owner_prelaunch'],
        engine_selfrun=str(a.engine_selfrun), engine_selfrun_sha256=sha(a.engine_selfrun),
        engine_selfrun_view=dict(probe39=e['probe39']['verdict'], probe37=e['probe37']['verdict'], py_compile=e['py_compile'],
                                 suites=e['suites'], network=e['network'], gpu_device_requests=e['gpu_device_requests']),
        clock_split=clock, engine_selfrun_probes=probes,
        runner_lineage=str(a.runner_lineage), runner_lineage_sha256=sha(a.runner_lineage),
        runner_lineage_view={k2: lineage[k2] for k2 in ('box_runner', 'box_runner_sha256', 'engine_runner', 'engine_runner_sha256',
                                                        'declared_differences', 'metadata_rows_moved_vs_n43f', 'owner_seam')},
        engine_queue_identity=ident,
        clock_scan=str(a.clock_scan), clock_scan_sha256=sha(a.clock_scan),
        clock_scan_view=dict(verdict=k['verdict'], client_units=len(k['scan']['client_units']), roots=len(k['scan']['roots']),
                             closure=len(k['scan']['closure']), invocations=len(k['scan']['invocations']),
                             reuse_clock_uses_native43_prove=k['scan']['reuse_clock_uses_native43_prove'],
                             cli_resolution=k['scan']['cli_resolution'], negative_controls=sorted(k['negative_controls'])),
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
              'clock split), the runner lineage proof and the static client-clock scan; no device run; cancel/drain/reuse '
              'and timing come only from the box run')
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
                 'precheck-conf', 'precheck-log', 'strict-log', 'focused-log', 'note', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--engine-base', required=True, help='Codex commit the four engine files must equal (109313a)')
    record(p.parse_args())
