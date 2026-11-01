#!/usr/bin/env python3
"""Assemble the machine delivery records for native39 trunk (results/hub/native39-trunk/).

Reads the build, freeze proof, both pinned-image self-runs of the held bundle
(the frozen-runner bundle probe and the engine self-run), the native38 held-b
negative control, the strict and focused logs and the delivery note, checks
they bind to one bundle, and writes delivery.json, frozen-gates.json and
strict.json (refusing to overwrite).  Every verdict is copied, never set.
"""
import argparse
import hashlib
import json
from pathlib import Path

FRESH_BOX_PATHS = ('/data/vllm/flash-next-tp4/prod/overlay-v16s5-q18a-native39-trunk',
                   '/data/vllm/flash-next-tp4/prod/overlay-v16s5-q18a-native39-trunk-build',
                   '/home/<service-user>/s0/q18a-native39-trunk', '/home/<service-user>/s0/q18a-native39-trunk-requests',
                   '/home/<service-user>/s0/q18a-native39-trunk.log', '/home/<service-user>/s0/q18a-native39-trunk-declare.log',
                   '/home/<service-user>/s0/q18a-native39-trunk-declare-chain.log',
                   '/home/<service-user>/s0/q18a-native39-trunk-staging.json',
                   '/home/<service-user>/s0/q18a-native39-trunk-build-logs')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def record(build, freeze, bundle_selfrun, engine_selfrun, negative, strict_log, focused_log, note, out):
    b, f = load(build), load(freeze)
    s, e, n = load(bundle_selfrun), load(engine_selfrun), load(negative)
    assert f['verdict'] == 'PASS' and f['manifest'] == b['manifest'] and b['freeze_proof_sha256'] == sha(freeze)
    assert s['verdict'] == 'PASS' and s['rc'] == 0 and s['bundle_manifest'] == b['manifest']
    assert s['frozen_runner_sha256'] == b['runner_sha256'] and s['tests_run'] == b['expected_selfrun_tests'] and s['skips'] == 0
    assert s['native35_arm'] == 'TRUNK' and s['native39_trunk_queue'] == 'PASS' and s['gpu_device_requests'] == 0
    assert s['overlay_manifest'] == b['overlay_manifest'] and s['image'] == b['image'] and s['owner_prelaunch']['verdict'] == 'PASS'
    assert e['schema'] == 'native39-trunk-selfrun/1' and e['label'] == b['name'] and e['bundle_manifest'] == b['manifest']
    assert e['container_rc'] == 0 and e['probe39']['verdict'] == 'PASS' and e['probe37']['verdict'] == 'PASS'
    assert e['py_compile']['ok'] and e['suites']['ok'] and e['suites']['skipped'] == 0
    assert e['network'] == 'none' and e['gpu_device_requests'] == 0 and e['image'] == b['image']
    assert n['overlay_manifest'] == f['parent_overlay_manifest'] and n['probe39']['verdict'] != 'PASS'
    strict = dict(b['regression'], log=str(strict_log), verdict='PASS', gpu_execution=False,
                  scope='full strict registered CPU regression on the native39 trunk build source tree')
    assert strict['log_sha256'] == sha(strict_log)
    focused = dict(b['focused'], log=str(focused_log))
    assert focused['log_sha256'] == sha(focused_log)
    details = load(Path(b['bundle']) / 'native30-provenance.json')['native39_trunk']
    queue_check = Path(bundle_selfrun).with_name('native39-trunk-queue-check.json')
    row = dict(
        name=b['name'], arm=b['arm'], source=b['source'], details=details,
        bundle=b['bundle'], manifest=b['manifest'], overlay_manifest=b['overlay_manifest'],
        coverage=b['checksummed_files'], review_sha256=b['review_sha256'], image=b['image'],
        runner=b['runner'], runner_sha256=b['runner_sha256'], runner_size=b['runner_size'],
        parent=dict(name='q18a-native38-plain-bind', manifest=b['parent_manifest'],
                    overlay_manifest=f['parent_overlay_manifest'], review_sha256=f['parent_review_sha256']),
        runner_parent=dict(path=f['runner']['parent'], sha256=f['runner']['parent_sha256']),
        runner_differences=f['runner']['differences'],
        freeze_proof=str(freeze), freeze_proof_sha256=sha(freeze),
        freeze_differences=dict(bundle_changed=f['changed'], bundle_added=f['added'], bundle_removed=f['removed'],
                                overlay_changed=f['overlay_changed'], engine_changed=f['engine_changed']),
        source_definitions={k: v['changed'] for k, v in f['freeze']['source']['definitions'].items()},
        overlay_definitions={k: v['changed'] for k, v in f['freeze']['overlay']['definitions'].items()},
        compile_proof=f['freeze']['source']['compile'], negative_controls=f['freeze']['negative_controls'],
        bundle_selfrun=str(bundle_selfrun), bundle_selfrun_sha256=sha(bundle_selfrun), bundle_selfrun_tests=s['tests_run'],
        bundle_selfrun_skips=s['skips'], bundle_selfrun_container=s['container'],
        trunk_queue_check=str(queue_check), trunk_queue_check_sha256=sha(queue_check),
        trunk_queue_cases=sorted(r['case'] for r in s['native39_trunk_queue_receipts']),
        owner_prelaunch=s['owner_prelaunch'],
        engine_selfrun=str(engine_selfrun), engine_selfrun_sha256=sha(engine_selfrun),
        engine_selfrun_view=dict(probe39=e['probe39'], probe37=dict(verdict=e['probe37']['verdict'], t2=e['probe37']['t2']),
                                 py_compile=e['py_compile'], suites=e['suites'], network=e['network'],
                                 gpu_device_requests=e['gpu_device_requests'], wall_s=e['wall_s']),
        negative_control=str(negative), negative_control_sha256=sha(negative),
        negative_control_view=dict(label=n['label'], probe39=n['probe39']['verdict'], probe37=n['probe37']['verdict'],
                                   probe39_first_failures={k: (n['probe39'].get(k) or {}).get('first_failure')
                                                           for k in ('r', 'w', 'p')}),
        fresh_box_paths=list(FRESH_BOX_PATHS) + ['/home/<service-user>/harness/' + b['manifest'][:12]],
        owner_commands=['sudo python3 %s --staging --auth-env-file PRIVATE_AUTH_FILE --selfrun-record EXACT_BUNDLE_SELFRUN_JSON '
                        '--engine-selfrun-record EXACT_ENGINE_SELFRUN_JSON --snapshot-max-bytes 536870912 --sync none'
                        % Path(b['runner']).name,
                        'owner stops production (vllm-flash inactive/failed, four idle GPUs), then: sudo python3 %s --arm'
                        % Path(b['runner']).name],
        note=str(note), note_sha256=sha(note), strict=strict, focused=focused, verdict='PASS', classification=None,
        gpu_execution=False, box_action=False,
        scope='frozen input gates, the fresh strict regression and both pinned-image CPU self-runs; no device run; the speed '
              'numbers come only from the box run; 80/80 is a reported marker, never a gate')
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (('delivery.json', row), ('frozen-gates.json', {k: v for k, v in row.items() if k not in ('strict', 'focused')}),
                        ('strict.json', strict)):
        with (out / name).open('x') as fh:
            json.dump(value, fh, sort_keys=True, indent=2)
            fh.write('\n')
    print(json.dumps(dict(out=str(out), verdict='PASS', manifest=b['manifest'][:12]), sort_keys=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('build', 'freeze', 'bundle-selfrun', 'engine-selfrun', 'negative', 'strict-log', 'focused-log', 'note', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    record(a.build, a.freeze, a.bundle_selfrun, a.engine_selfrun, a.negative, a.strict_log, a.focused_log, a.note, a.out)
