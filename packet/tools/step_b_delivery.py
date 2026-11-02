#!/usr/bin/env python3
"""Bind the Step B delivery to both exact-byte CPU self-runs, the freezes, the comparison, the negative, the tamper control, the clock
scan, the compile proof, the precheck, the VM run on the frozen bundle and the engine diff against Step A. One record; asserts, never repairs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT_PINS = json.loads((ROOT / 'tools/step_b_parent.json').read_text())
STEP_A_MANIFEST = PARENT_PINS['manifest']
REVIEW = 'patches/s5_engine/native_candidate_review.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def engine_diff(parent, modules):
    """Every path under patches/ that differs from Step A: the declared engine modules and the review pin, nothing else."""
    rows = git('diff', '--name-status', parent, 'HEAD', '--', 'patches').splitlines()
    declared = {'M\tpatches/s5_engine/' + name for name in modules} | {'M\t' + REVIEW}
    assert set(rows) == declared, sorted(set(rows) ^ declared)
    text = git('diff', parent, 'HEAD', '--', 'patches/s5_engine', ':!' + REVIEW)
    return dict(parent_commit=git('rev-parse', parent).strip(), head=git('rev-parse', 'HEAD').strip(), paths_differing=rows,
                stat=git('diff', '--stat', parent, 'HEAD', '--', 'patches/s5_engine', ':!' + REVIEW).splitlines(),
                diff_sha256=hashlib.sha256(text.encode()).hexdigest())


def vm_record(record, compare, manifest):
    """The VM run on the frozen bundle: its own verdict, the bundle it was built from, and what the record says it cannot show."""
    record, compare = Path(record), Path(compare)
    result = json.loads((record / 'result.json').read_text())
    build = json.loads(Path(str(record) + '.build.json').read_text())
    report = json.loads(compare.read_text())
    assert result['verdict'] == 'PASS' and result['probe_verdict'] == 'PASS' and result['rc'] == 0, result['verdict']
    assert build['bundle_manifest_sha256'] == manifest, 'the VM run was not on the frozen bundle'
    assert not any(row['failures'] for row in report['runs'].values())
    return dict(path=str(record), sha256=sha(record / 'result.json'), verdict=result['verdict'], rc=result['rc'],
                own_peak_mib=result['own_peak_mib'], bundle_manifest=build['bundle_manifest_sha256'], compare=str(compare),
                compare_sha256=sha(compare), bytes_verdict=report['bytes_verdict'], deterministic=report['deterministic'],
                cannot_show=report['cannot_show'], scope='one real 3090, seeded dummy weights, one rank')


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('build', 'freeze', 'engine', 'compare', 'negative', 'tamper', 'queue', 'clock_scan', 'compile_proof', 'precheck',
                 'conf', 'note', 'vm', 'vm_compare', 'staging_check', 'staging_predicates', 'out'):
        p.add_argument('--' + name.replace('_', '-'), dest=name, type=Path, required=True)
    a = p.parse_args()
    sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
    import step_b_freeze as freeze_tool
    import step_b_gate as gate
    import step_b_clock_bound as clock_bound
    b, fz, e, cmp_, neg, tam, q, k, c = [json.loads(path.read_text()) for path in (
        a.build, a.freeze, a.engine, a.compare, a.negative, a.tamper, a.queue, a.clock_scan, a.compile_proof)]
    assert b['schema'] == 'native43-build/1' and b['name'] == 'q18a-step-b' and b['parent_manifest'] == STEP_A_MANIFEST
    assert e['verdict'] == q['verdict'] == k['verdict'] == fz['verdict'] == cmp_['verdict'] == neg['verdict'] == tam['verdict'] == 'PASS'
    assert e['schema'] == 'step-b-selfrun/1' and e['negative_control'] is False
    assert fz['candidate_manifest'] == b['manifest'] and fz['candidate_overlay_manifest'] == b['overlay_manifest']
    assert fz['parent_manifest'] == STEP_A_MANIFEST and fz['reference']['overlay_byte_identical_to_parent'] is True
    assert fz['negative_controls'] and set(fz['negative_controls'].values()) == {'REFUSED'}
    assert e['manifest'] == q['bundle_manifest'] == b['manifest'] and e['overlay_manifest'] == b['overlay_manifest']
    assert e['suites']['ok'] is True and e['skips'] == q['skips'] == 0 and all(x['verdict'] == 'PASS' for x in e['cases'].values())
    assert all(x['same'] for x in e['pairs'].values()) and e['pairs']
    assert not cmp_['differs'] and cmp_['candidate_record_sha256'] == sha(a.engine.parent / 'result.json')
    assert neg['negative_control'] is True and neg['manifest'] == b['manifest'] and not neg['reasons'] and neg['pinned_tests'] > 0
    assert tam['candidate_manifest'] == b['overlay_manifest'] and all(x['ok'] for x in tam['cases'])
    assert q['frozen_runner_sha256'] == b['runner_sha256'] and q['tests_run'] == b['expected_selfrun_tests']
    assert q['step_b_queue'] == 'PASS' and q['owner_prelaunch']['verdict'] == 'PASS'
    assert q['clock_bound_frozen_excluded'] == clock_bound.IN_PROBE
    assert k['schema'] == 'step-b-clock-scan/1' and not k['scan']['problems'] and k['bundle_manifest'] == b['manifest']
    assert k['runner']['sha256'] == b['runner_sha256'] and k['negative_controls'] and all(k['negative_controls'].values())
    st = json.loads(a.staging_check.read_text())
    assert st['schema'] == 'step-b-staging-check/1' and st['verdict'] == 'PASS' and all(st['negative_controls_refused'].values())
    assert st['runner_sha256'] == b['runner_sha256'] and st['selfrun_sha256'] == sha(a.queue) and st['engine_selfrun_sha256'] == sha(a.engine)
    sp = json.loads(a.staging_predicates.read_text())
    assert sp['schema'] == 'staging-predicates/1' and sp['verdict'] == 'PASS' and not sp['missing_blocks'] and not sp['repeated_blocks']
    assert sp['runner_sha256'] == b['runner_sha256'] and sp['selfrun_sha256'] == sha(a.queue) and sp['engine_selfrun_sha256'] == sha(a.engine)
    pins = json.loads((ROOT / 'tools/step_b_negative_pins.json').read_text())
    # The ids the engine self-run's dense container left out, from its own suites.log (sha-bound in the dense record), checked
    # against the declared list and its reason.
    dense_log = a.engine.parent / 'dense' / 'engine' / 'suites.log'
    dense_rec = json.loads((a.engine.parent / 'dense' / 'result.json').read_text())
    assert sha(dense_log) == dense_rec['suites']['log_sha256'], 'dense suites log differs from its record'
    left_out = sorted(re.findall(r'^clock-bound, excluded \(declared\): (\S+)$', dense_log.read_text(), re.M))
    assert left_out and all(x in clock_bound.CLOCK_BOUND_FROZEN for x in left_out), left_out
    assert neg['pinned_tests'] == len(pins['tests']) and sha(pins['observation']['observed_json']) == pins['observation']['observed_sha256']
    proof = json.loads((Path(b['bundle']).parent / 'freeze-proof.json').read_text())
    assert sha(Path(b['bundle']).parent / 'freeze-proof.json') == b['freeze_proof_sha256'] and proof['verdict'] == 'PASS'
    assert sha(Path(b['bundle']) / 'SHA256SUMS') == b['manifest']
    summary = re.findall(r'^SUMMARY checks=(\d+) pass=(\d+) fail=(\d+)$', a.precheck.read_text(), re.M)
    assert len(summary) == 1 and summary[0][0] == summary[0][1] and summary[0][2] == '0', summary
    leaves = json.loads((ROOT / 'harness/step_b_fixture_leaves.json').read_text())
    source = fz['source']
    row = dict(schema='step-b-delivery/1', verdict='PASS', classification=None, box_action=False, gpu_execution=False,
               name=b['name'], bundle=b['bundle'], manifest=b['manifest'], overlay_manifest=b['overlay_manifest'],
               review_sha256=b['review_sha256'], runner=b['runner'], runner_sha256=b['runner_sha256'], runner_size=b['runner_size'],
               source_commit=b['source_commit'], build_commit=b['source'], parent_bundle=b['parent_bundle'],
               parent_manifest=b['parent_manifest'], freeze_proof_sha256=b['freeze_proof_sha256'],
               engine_changed_vs_parent=b['engine_changed'], regression=b['regression'], focused=b['focused'],
               engine_sources_vs_step_a=engine_diff(fz['reference']['engine_commit'], freeze_tool.ENGINE),
               step_a_base=dict(build=PARENT_PINS['build'], build_sha256=PARENT_PINS['build_sha256'], manifest=STEP_A_MANIFEST,
                                engine_commit=fz['reference']['engine_commit']),
               step_b_freeze=dict(path=str(a.freeze), sha256=sha(a.freeze), verdict=fz['verdict'],
                                  projected_functions=source['default_off']['functions'],
                                  not_projected=source.get('not_projected'), reference=fz['reference']),
               engine_selfrun=dict(path=str(a.engine), sha256=sha(a.engine), verdict=e['verdict'], tests=e['tests'], skips=e['skips'],
                                   cases=sorted(e['cases']), connected_checks=e['connected_checks'],
                                   clock_split=e['clock_split']['verdict'], scheduler_tick_probe=(e.get('scheduler_tick_probe') or {}).get('verdict'),
                                   probe39=e['probe39']['verdict'], probe37=e['probe37']['verdict'], pairs=e['pairs'],
                                   declared_limit=e['declared_limit'],
                                   dense_left_out={x: clock_bound.CLOCK_BOUND_FROZEN[x] for x in left_out},
                                   dense_suites_log=str(dense_log), dense_suites_log_sha256=sha(dense_log)),
               comparison=dict(path=str(a.compare), sha256=sha(a.compare), verdict=cmp_['verdict'], compared=cmp_['compared'],
                               parent_record=cmp_['parent_record'], parent_record_sha256=cmp_['parent_record_sha256']),
               negative=dict(path=str(a.negative), sha256=sha(a.negative), verdict=neg['verdict'], pinned_tests=neg['pinned_tests'],
                             tests=neg['suites'].get('tests'), container_rc=neg['container_rc'], overlay_manifest=neg['overlay_manifest'],
                             pins=dict(declared='negative pins from observation, 2 Nov', count=len(pins['tests']), cases=len(pins['cases']),
                                       path='tools/step_b_negative_pins.json', sha256=sha(ROOT / 'tools/step_b_negative_pins.json'),
                                       observation=pins['observation'])),
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
                                  requires={k: v['holds'] for k, v in st['requires'].items()},
                                  negative_controls_refused=st['negative_controls_refused'], runner_sha256=st['runner_sha256'],
                                  selfrun_sha256=st['selfrun_sha256'], engine_selfrun_sha256=st['engine_selfrun_sha256']),
               staging_predicates=dict(path=str(a.staging_predicates), sha256=sha(a.staging_predicates), verdict=sp['verdict'],
                                       blocks={k: v['verdict'] for k, v in sp['blocks'].items()},
                                       tool='<hub-dir>/agents/flash-next/tmp/step-a/staging-check/staging_predicates.py',
                                       tool_sha256=sha('<hub-dir>/agents/flash-next/tmp/step-a/staging-check/staging_predicates.py')),
               vm=vm_record(a.vm, a.vm_compare, b['manifest']),
               fixture_leaves=dict(path='harness/step_b_fixture_leaves.json', sha256=sha(ROOT / 'harness/step_b_fixture_leaves.json'),
                                   inherits=leaves['inherits'], names=[x['name'] for x in leaves['leaves']]),
               clock_bound_frozen=dict(reason=clock_bound.REASON, ids=sorted(clock_bound.CLOCK_BOUND_FROZEN),
                                       left_out_of_bundle_selfrun=q['clock_bound_frozen_excluded'],
                                       runner_requirement_removed="result.get('native39_trunk_queue')=='PASS'",
                                       approval='operator, 2 Nov 2026 about 03:05 CET, through the lead'),
               launch_time='governed by the lead at launch (operator, 2 Nov 2026); no hard end, window or latest start in the package',
               expected_runtime=gate.EXPECTED_RUNTIME_S, box_budget_s=gate.BOX_BUDGET_S, active_s=gate.ACTIVE_S, cleanup_s=gate.CLEANUP_S,
               last_arm_entry_s=gate.LAST_ARM_ENTRY_S, trim_g4_repeat_s=gate.TRIM_G4_REPEAT_S,
               note=str(a.note), note_sha256=sha(a.note), model_settings_changed=False, device_run='NOT_RUN')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as handle:
        handle.write(json.dumps(row, sort_keys=True, indent=1) + '\n')
    print(json.dumps({k2: row[k2] for k2 in ('verdict', 'manifest', 'runner_sha256', 'review_sha256')}, sort_keys=True))
    print('delivery sha256 ' + sha(a.out))


if __name__ == '__main__':
    main()
