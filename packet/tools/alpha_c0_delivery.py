#!/usr/bin/env python3
"""Bind the C0 delivery to both exact-byte CPU self-runs, the freezes, the comparison, the negative, the tamper control, the
clock scan, the compile proof, the precheck, the engine diff against c3f1f64 and the staging check (the frozen runner's own
staging requires evaluated over the same two self-run records, tools/runner_staging_check.py). Writes one record; asserts, never
repairs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ALPHA_PERF_MANIFEST = 'd6b0f69ead0b7d602f5caf879d938ec111162fb3d38fb79e3960b0a6846465f7'
REVIEW = 'patches/s5_engine/native_candidate_review.json'
PARENT = 'c3f1f64'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def engine_diff(modules, added):
    """Every path under patches/ that differs from c3f1f64: the declared engine modules, the two new ones and the review pin, nothing else."""
    rows = git('diff', '--name-status', PARENT, 'HEAD', '--', 'patches').splitlines()
    declared = ({'M\tpatches/s5_engine/' + name for name in modules} | {'A\tpatches/s5_engine/' + name for name in added}
                | {'M\t' + REVIEW})
    assert set(rows) == declared, sorted(set(rows) ^ declared)
    text = git('diff', PARENT, 'HEAD', '--', 'patches/s5_engine', ':!' + REVIEW)
    return dict(parent_commit=git('rev-parse', PARENT).strip(), head=git('rev-parse', 'HEAD').strip(), paths_differing=rows,
                stat=git('diff', '--stat', PARENT, 'HEAD', '--', 'patches/s5_engine', ':!' + REVIEW).splitlines(),
                diff_sha256=hashlib.sha256(text.encode()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('build', 'freeze', 'engine', 'compare', 'negative', 'tamper', 'queue', 'clock_scan', 'compile_proof', 'precheck',
                 'conf', 'staging_check', 'note', 'out'):
        p.add_argument('--' + name.replace('_', '-'), dest=name, type=Path, required=True)
    p.add_argument('--vm', type=Path, help='the VM record, when the lead has run it')
    a = p.parse_args()
    sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
    import alpha_c0_freeze as freeze_tool
    import alpha_c0_gate as gate
    b, fz, e, cmp_, neg, tam, q, k, c = [json.loads(path.read_text()) for path in (
        a.build, a.freeze, a.engine, a.compare, a.negative, a.tamper, a.queue, a.clock_scan, a.compile_proof)]
    assert b['schema'] == 'native43-build/1' and b['name'] == 'q18a-alpha-c0' and b['parent_manifest'] == ALPHA_PERF_MANIFEST
    assert e['verdict'] == q['verdict'] == k['verdict'] == fz['verdict'] == cmp_['verdict'] == neg['verdict'] == tam['verdict'] == 'PASS'
    assert e['schema'] == 'alpha-c0-selfrun/1' and e['negative_control'] is False
    assert fz['candidate_manifest'] == b['manifest'] and fz['candidate_overlay_manifest'] == b['overlay_manifest']
    assert fz['parent_manifest'] == ALPHA_PERF_MANIFEST and fz['reference']['overlay_byte_identical_to_parent'] is True
    assert fz['negative_controls'] and set(fz['negative_controls'].values()) == {'REFUSED'}
    assert e['manifest'] == q['bundle_manifest'] == b['manifest'] and e['overlay_manifest'] == b['overlay_manifest']
    assert e['suites']['ok'] is True and e['skips'] == q['skips'] == 0 and all(x['verdict'] == 'PASS' for x in e['cases'].values())
    assert all(x['same'] for x in e['pairs'].values()) and e['pairs']
    assert not cmp_['differs'] and cmp_['candidate_record_sha256'] == sha(a.engine.parent / 'result.json')
    assert neg['negative_control'] is True and neg['manifest'] == b['manifest'] and not neg['reasons'] and neg['pinned_tests'] > 0
    assert tam['candidate_manifest'] == b['overlay_manifest'] and all(x['ok'] for x in tam['cases'])
    assert q['frozen_runner_sha256'] == b['runner_sha256'] and q['tests_run'] == b['expected_selfrun_tests']
    assert q['alpha_c0_queue'] == 'PASS' and q['owner_prelaunch']['verdict'] == 'PASS'
    assert k['schema'] == 'alpha-c0-clock-scan/1' and not k['scan']['problems'] and k['bundle_manifest'] == b['manifest']
    assert k['runner']['sha256'] == b['runner_sha256'] and k['negative_controls'] and all(k['negative_controls'].values())
    proof = json.loads((Path(b['bundle']).parent / 'freeze-proof.json').read_text())
    assert sha(Path(b['bundle']).parent / 'freeze-proof.json') == b['freeze_proof_sha256'] and proof['verdict'] == 'PASS'
    assert sha(Path(b['bundle']) / 'SHA256SUMS') == b['manifest']
    summary = re.findall(r'^SUMMARY checks=(\d+) pass=(\d+) fail=(\d+)$', a.precheck.read_text(), re.M)
    assert len(summary) == 1 and summary[0][0] == summary[0][1] and summary[0][2] == '0', summary
    # The frozen runner's own staging requires over these exact records (tools/runner_staging_check.py; the lead, 2 Nov 2026).
    st = json.loads(a.staging_check.read_text())
    assert st['schema'] == 'runner-staging-check/1' and st['verdict'] == 'PASS' and st['rc'] == 0 and not st['reasons_missing']
    assert st['runner_sha256'] == b['runner_sha256'] == sha(b['runner'])
    assert st['selfrun_sha256'] == sha(a.queue) and st['engine_selfrun_sha256'] == sha(a.engine)
    leaves = json.loads((ROOT / 'harness/alpha_c0_fixture_leaves.json').read_text())
    vm = json.loads(a.vm.read_text()) if a.vm else None
    assert vm is None or (vm.get('verdict') == 'PASS' and vm.get('bundle_manifest', b['manifest']) == b['manifest'])
    source = fz['source']
    row = dict(schema='alpha-c0-delivery/1', verdict='PASS', classification=None, box_action=False, gpu_execution=False,
               name=b['name'], bundle=b['bundle'], manifest=b['manifest'], overlay_manifest=b['overlay_manifest'],
               review_sha256=b['review_sha256'], runner=b['runner'], runner_sha256=b['runner_sha256'], runner_size=b['runner_size'],
               source_commit=b['source_commit'], build_commit=b['source'], parent_bundle=b['parent_bundle'],
               parent_manifest=b['parent_manifest'], freeze_proof_sha256=b['freeze_proof_sha256'],
               engine_changed_vs_parent=b['engine_changed'], regression=b['regression'], focused=b['focused'],
               engine_sources_vs_c3f1f64=engine_diff(freeze_tool.off.PROJECTIONS, freeze_tool.off.NEW_MODULES),
               alpha_c0_freeze=dict(path=str(a.freeze), sha256=sha(a.freeze), verdict=fz['verdict'],
                                  identical_files=source['identical_files'], new_modules=source['new_modules'],
                                  projected_files=sorted(source['projected']), negative_controls=sorted(fz['negative_controls']),
                                  reference=fz['reference']),
               engine_selfrun=dict(path=str(a.engine), sha256=sha(a.engine), verdict=e['verdict'], tests=e['tests'], skips=e['skips'],
                                   cases=sorted(e['cases']), connected_checks=e['connected_checks'],
                                   clock_split=e['clock_split']['verdict'], scheduler_tick_probe=(e.get('scheduler_tick_probe') or {}).get('verdict'),
                                   probe39=e['probe39']['verdict'], probe37=e['probe37']['verdict'], pairs=e['pairs'],
                                   declared_limit=e['declared_limit']),
               comparison=dict(path=str(a.compare), sha256=sha(a.compare), verdict=cmp_['verdict'], compared=cmp_['compared'],
                               parent_record=cmp_['parent_record'], parent_record_sha256=cmp_['parent_record_sha256']),
               negative=dict(path=str(a.negative), sha256=sha(a.negative), verdict=neg['verdict'], pinned_tests=neg['pinned_tests'],
                             tests=neg['suites'].get('tests'), container_rc=neg['container_rc'], overlay_manifest=neg['overlay_manifest']),
               overlay_tamper=dict(path=str(a.tamper), sha256=sha(a.tamper), verdict=tam['verdict'], cases=len(tam['cases'])),
               queue_selfrun=dict(path=str(a.queue), sha256=sha(a.queue), verdict=q['verdict'], rc=q.get('rc'),
                                  tests_run=q['tests_run'], skips=q['skips'], container=q.get('container'),
                                  frozen_runner_sha256=q['frozen_runner_sha256'], gpu_device_requests=q.get('gpu_device_requests')),
               clock_scan=dict(path=str(a.clock_scan), sha256=sha(a.clock_scan), verdict=k['verdict'],
                               negative_controls=sorted(k['negative_controls'])),
               compile_proof=dict(path=str(a.compile_proof), sha256=sha(a.compile_proof), **{x: c.get(x) for x in ('verdict', 'files', 'count', 'failed')}),
               precheck=dict(conf=str(a.conf), conf_sha256=sha(a.conf), log=str(a.precheck), log_sha256=sha(a.precheck),
                             checks=int(summary[0][0]), passed=int(summary[0][1]), failed=0),
               staging_check=dict(path=str(a.staging_check), sha256=sha(a.staging_check), verdict=st['verdict'],
                                  reasons=st['reasons_found'], case_lists=[c2 for blk in st['blocks'] for c2 in blk['case_lists']]),
               vm=dict(path=str(a.vm), sha256=sha(a.vm), verdict=vm.get('verdict'), rc=vm.get('rc'), scope=vm.get('scope')) if vm
               else dict(verdict='NOT_RUN', why='no VM record was given'),
               fixture_leaves=dict(path='harness/alpha_c0_fixture_leaves.json', sha256=sha(ROOT / 'harness/alpha_c0_fixture_leaves.json'),
                                   inherits=leaves['inherits'], names=[x['name'] for x in leaves['leaves']]),
               expected_runtime=gate.EXPECTED_RUNTIME_S, box_budget_s=gate.BOX_BUDGET_S, active_s=gate.ACTIVE_S, cleanup_s=gate.CLEANUP_S,
               last_arm_entry_s=gate.LAST_ARM_ENTRY_S, latest_start_unix=gate.latest_arm(),
               note=str(a.note), note_sha256=sha(a.note), model_settings_changed=False, box_device_run='NOT_RUN')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as handle:
        handle.write(json.dumps(row, sort_keys=True, indent=1) + '\n')
    print(json.dumps({k2: row[k2] for k2 in ('verdict', 'manifest', 'runner_sha256', 'review_sha256')}, sort_keys=True))
    print('delivery sha256 ' + sha(a.out))


if __name__ == '__main__':
    main()
