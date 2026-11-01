#!/usr/bin/env python3
"""Assemble the machine delivery records for a native43 packaging build (results/hub/native43-<label>/).

Reads the build, freeze proof, the bundle self-run (frozen-runner probe), the engine self-run (with the
CT-vs-container clock split), the owner-chain seam self-run, the compile proof, the precheck conf and log,
the strict and focused logs and the delivery note; checks they bind to one bundle and one runner; writes
delivery.json, frozen-gates.json and strict.json (refusing to overwrite).  Every verdict is copied, never set.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def fresh_box_paths(name, manifest):
    prod = '/data/vllm/flash-next-tp4/prod/overlay-v16s5-' + name
    s0 = '/home/<service-user>/s0/' + name
    arms = [s0 + '-' + arm + part for arm in ('current', 'dense') for part in ('', '-requests', '-declare.log', '-declare-chain.log')]
    return [prod, prod + '-build', s0, s0 + '.log', s0 + '-staging.json', s0 + '-pair-grade.json', s0 + '-build-logs',
            *arms, '/home/<service-user>/harness/' + manifest[:12]]


def precheck_summary(log):
    rows = re.findall(r'^SUMMARY checks=(\d+) pass=(\d+) fail=(\d+)$', Path(log).read_text(), re.M)
    assert len(rows) == 1 and rows[0][0] == rows[0][1] and rows[0][2] == '0', rows
    return rows[0][1] + '/' + rows[0][0]


def record(a):
    b, f, spec = load(a.build), load(a.freeze), load(a.spec)
    s, e, o, c = load(a.bundle_selfrun), load(a.engine_selfrun), load(a.owner_seam), load(a.compile_proof)
    assert not spec['dry_run'] and b['name'] == spec['candidate'] and b['arm'] == spec['arm'] == 'NATIVE43'
    assert f['verdict'] == 'PASS' and f['manifest'] == b['manifest'] and f['spec_sha256'] == sha(a.spec)
    assert s['verdict'] == 'PASS' and s['rc'] == 0 and s['bundle_manifest'] == b['manifest'] and s['skips'] == 0
    assert s['frozen_runner_sha256'] == b['runner_sha256'] and s['tests_run'] == b['expected_selfrun_tests']
    assert s['native35_arm'] == 'NATIVE43' and s['native43_queue'] == 'PASS' and s['native39_trunk_queue'] == 'PASS'
    assert s['gpu_device_requests'] == 0 and s['overlay_manifest'] == b['overlay_manifest'] and s['image'] == b['image']
    assert s['owner_prelaunch']['verdict'] == 'PASS'
    assert e['schema'] == 'native43-selfrun/1' and e['label'] == b['name'] and e['bundle_manifest'] == b['manifest']
    assert e['verdict'] == 'PASS' and e['container_rc'] == 0 and e['probe39']['verdict'] == 'PASS'
    assert e['probe37']['verdict'] == 'PASS' and e['py_compile']['ok'] and e['suites']['ok'] and e['suites']['skipped'] == 0
    assert e['network'] == 'none' and e['gpu_device_requests'] == 0 and e['image'] == b['image']
    clock = e['clock_split']
    assert clock['verdict'] == 'PASS' and clock['outcomes'] == spec['clock_split']['expect'] and clock['side'] == spec['clock_split']['side']
    assert o['verdict'] == 'PASS' and o['rc'] == 0 and o['runner_sha256'] == b['runner_sha256'] and o['bundle_manifest'] == b['manifest']
    assert o['network'] == 'none' and o['gpu_device_requests'] == 0 and not o['problems']
    assert c['verdict'] == 'PASS' and not c['failures'] and c['manifest'] == b['manifest'] and c['runner_sha256'] == b['runner_sha256']
    precheck = precheck_summary(a.precheck_log)
    strict = dict(b['regression'], log=str(a.strict_log), verdict='PASS', gpu_execution=False,
                  scope='full strict registered CPU regression on the merged frozen native43 build tree')
    assert strict['log_sha256'] == sha(a.strict_log)
    focused = dict(b['focused'], log=str(a.focused_log))
    assert focused['log_sha256'] == sha(a.focused_log)
    provenance = load(Path(b['bundle']) / 'native30-provenance.json')
    gate = Path(b['bundle']) / 'native43_boundary_gate.py'
    namespace = {}
    exec(compile(gate.read_text().replace('from native34_gate import NOT_BEFORE, HARD_END',
                                          'NOT_BEFORE = HARD_END = None'), str(gate), 'exec'), namespace)
    weekend = {}
    exec(compile((Path(b['bundle']) / 'native34_gate.py').read_text(), 'native34_gate.py', 'exec'), weekend)
    pair_bound = namespace['PAIR_ACTIVE_S'] + namespace['PAIR_KILL_MARGIN_S']
    latest = datetime.datetime.fromtimestamp(weekend['HARD_END'] - pair_bound, datetime.timezone.utc)
    queue_check = Path(a.bundle_selfrun).with_name('native43-queue-check.json')
    runner = Path(b['runner']).name
    row = dict(
        name=b['name'], arm=b['arm'], label=spec['label'], source=b['source'], source_commit=spec['source_commit'],
        details=provenance['native43'], bundle=b['bundle'], manifest=b['manifest'], overlay_manifest=b['overlay_manifest'],
        coverage=b['checksummed_files'], review_sha256=b['review_sha256'], image=b['image'],
        runner=b['runner'], runner_sha256=b['runner_sha256'], runner_size=b['runner_size'],
        runner_parent=dict(path=f['runner']['parent'], sha256=f['runner']['parent_sha256']),
        runner_differences=f['runner']['differences'], runner_swaps=len(f['runner']['swaps']), runner_phases=f['runner']['phases'],
        parent=dict(manifest=f['parent_manifest'], commit=f['parent_commit'], review_sha256=f['parent_review_sha256']),
        freeze_proof=str(a.freeze), freeze_proof_sha256=sha(a.freeze), spec=str(a.spec), spec_sha256=sha(a.spec),
        freeze_differences=dict(bundle_changed=f['changed'], bundle_added=f['added'], bundle_removed=f['removed'],
                                overlay_moved=f['overlay']['moved'], engine_changed=f['engine_changed']),
        source_definitions={k: v['changed'] for k, v in f['freeze']['source']['definitions'].items()},
        overlay_definitions={k: v['changed'] for k, v in f['overlay']['definitions'].items()},
        source_compile=f['freeze']['source']['compile'], negative_controls=f['freeze']['negative_controls'],
        bundle_selfrun=str(a.bundle_selfrun), bundle_selfrun_sha256=sha(a.bundle_selfrun),
        bundle_selfrun_tests=s['tests_run'], bundle_selfrun_skips=s['skips'], bundle_selfrun_container=s['container'],
        bundle_selfrun_wall_s=s['wall_s'], native43_queue_cases=s['native43_queue_cases'],
        native43_queue_check=str(queue_check), native43_queue_check_sha256=sha(queue_check),
        owner_prelaunch=s['owner_prelaunch'],
        engine_selfrun=str(a.engine_selfrun), engine_selfrun_sha256=sha(a.engine_selfrun),
        engine_selfrun_view=dict(probe39=e['probe39']['verdict'], probe37=e['probe37']['verdict'], py_compile=e['py_compile'],
                                 suites=e['suites'], network=e['network'], gpu_device_requests=e['gpu_device_requests']),
        clock_split=clock,
        owner_seam=str(a.owner_seam), owner_seam_sha256=sha(a.owner_seam),
        owner_seam_view=dict(verdict=o['verdict'], negatives=o['negatives'], fixture_leaves=o['fixture_leaves'],
                             real_draft_client_foreground=o['real_draft_client_foreground'],
                             owner_decision=o['owner_decision'], container=o['container'], wall_s=o['wall_s']),
        compile_proof=str(a.compile_proof), compile_proof_sha256=sha(a.compile_proof), compile_files=c['files'],
        compile_failures=len(c['failures']),
        precheck=precheck, precheck_conf=str(a.precheck_conf), precheck_conf_sha256=sha(a.precheck_conf),
        precheck_log=str(a.precheck_log), precheck_log_sha256=sha(a.precheck_log),
        pair_bound_s=pair_bound, latest_entry_utc=latest.strftime('%Y-%m-%dT%H:%M:%SZ'),
        fresh_box_paths=fresh_box_paths(b['name'], b['manifest']),
        owner_commands=['sudo python3 %s --staging --auth-env-file PRIVATE_AUTH_FILE --selfrun-record EXACT_BUNDLE_SELFRUN_JSON '
                        '--engine-selfrun-record EXACT_ENGINE_SELFRUN_JSON --snapshot-max-bytes 536870912 --sync none' % runner,
                        'owner stops production (vllm-flash inactive/failed, four idle GPUs), then: sudo python3 %s --arm' % runner],
        key_policy='one fresh key per pair queue, shared by both arms (owner decision 1 Nov 2026); --staging declares '
                   'both request directories from it',
        note=str(a.note), note_sha256=sha(a.note), strict=strict, focused=focused, verdict='PASS', classification=None,
        gpu_execution=False, box_action=False,
        scope='frozen input gates, the fresh strict regression, the pinned-image CPU self-runs (bundle probe, engine + '
              'clock split, owner-chain seam); no device run; boundary and timing evidence come only from the box run')
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (('delivery.json', row), ('frozen-gates.json', {k: v for k, v in row.items() if k not in ('strict', 'focused')}),
                        ('strict.json', strict)):
        with (out / name).open('x') as fh:
            json.dump(value, fh, sort_keys=True, indent=2)
            fh.write('\n')
    print(json.dumps(dict(out=str(out), verdict='PASS', manifest=b['manifest'][:12], precheck=precheck,
                          latest_entry_utc=row['latest_entry_utc']), sort_keys=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('spec', 'build', 'freeze', 'bundle-selfrun', 'engine-selfrun', 'owner-seam', 'compile-proof',
                 'precheck-conf', 'precheck-log', 'strict-log', 'focused-log', 'note', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    record(p.parse_args())
