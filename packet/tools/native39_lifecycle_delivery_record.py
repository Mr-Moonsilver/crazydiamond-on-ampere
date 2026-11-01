#!/usr/bin/env python3
"""Assemble the machine delivery records for native39 lifecycle (results/hub/native39-lifecycle/).

Reads the build, freeze proof, both pinned-image self-runs of the held bundle
(the frozen-runner bundle probe and the trunk engine self-run under the lifecycle
label), the compile proof, the owner precheck conf and log, the strict and focused
logs and the delivery note, checks they bind to one bundle, and writes
delivery.json, frozen-gates.json and strict.json (refusing to overwrite).  Every
verdict is copied, never set.  No engine negative control: the engine is the
trunk's byte for byte (freeze proof), so the trunk's native38 negative control
stands for it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

FRESH_BOX_PATHS = ('/data/vllm/flash-next-tp4/prod/overlay-v16s5-q18a-native39-lifecycle',
                   '/data/vllm/flash-next-tp4/prod/overlay-v16s5-q18a-native39-lifecycle-build',
                   '/home/<service-user>/s0/q18a-native39-lifecycle', '/home/<service-user>/s0/q18a-native39-lifecycle-requests',
                   '/home/<service-user>/s0/q18a-native39-lifecycle.log', '/home/<service-user>/s0/q18a-native39-lifecycle-declare.log',
                   '/home/<service-user>/s0/q18a-native39-lifecycle-declare-chain.log',
                   '/home/<service-user>/s0/q18a-native39-lifecycle-staging.json',
                   '/home/<service-user>/s0/q18a-native39-lifecycle-build-logs')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def precheck_summary(log):
    text = Path(log).read_text()
    summary = re.search(r'^SUMMARY checks=(\d+) pass=(\d+) fail=(\d+)$', text, re.M)
    assert summary and summary.group(1) == summary.group(2) and summary.group(3) == '0', log
    return summary.group(2) + '/' + summary.group(1)


def record(build, freeze, bundle_selfrun, engine_selfrun, compile_proof, precheck_conf, precheck_log, strict_log,
           focused_log, note, out):
    b, f, c = load(build), load(freeze), load(compile_proof)
    s, e = load(bundle_selfrun), load(engine_selfrun)
    assert f['verdict'] == 'PASS' and f['manifest'] == b['manifest'] and b['freeze_proof_sha256'] == sha(freeze)
    assert f['engine_changed'] == [] and f['overlay_byte_identical'] is True and f['overlay_regenerated'] is False
    assert b['engine_env_delta'] == {'FLASH_POOL_STATS': '200'}
    assert s['verdict'] == 'PASS' and s['rc'] == 0 and s['bundle_manifest'] == b['manifest']
    assert s['frozen_runner_sha256'] == b['runner_sha256'] and s['tests_run'] == b['expected_selfrun_tests'] and s['skips'] == 0
    assert s['native35_arm'] == 'LIFECYCLE' and s['native39_lifecycle_queue'] == 'PASS' and s['native39_trunk_queue'] == 'PASS'
    assert s['gpu_device_requests'] == 0 and s['overlay_manifest'] == b['overlay_manifest'] and s['image'] == b['image']
    assert s['owner_prelaunch']['verdict'] == 'PASS'
    assert s['container'] == b['name'][len('q18a-'):] + '-hub-cpu-' + b['manifest'][:12]
    assert e['schema'] == 'native39-trunk-selfrun/1' and e['label'] == b['name'] and e['bundle_manifest'] == b['manifest']
    assert e['container_rc'] == 0 and e['probe39']['verdict'] == 'PASS' and e['probe37']['verdict'] == 'PASS'
    assert e['py_compile']['ok'] and e['suites']['ok'] and e['suites']['skipped'] == 0
    assert e['network'] == 'none' and e['gpu_device_requests'] == 0 and e['image'] == b['image'] and e['box_action'] is False
    assert e['overlay_manifest'] == b['overlay_manifest']
    assert c['verdict'] == 'PASS' and c['manifest'] == b['manifest'] and c['runner_sha256'] == b['runner_sha256']
    conf_text = Path(precheck_conf).read_text()
    for key in ('EXPECTED_MANIFEST', 'EXPECTED_RUNNER_SHA'):
        assert re.search(r'(?m)^%s=%s$' % (key, b['manifest'] if key == 'EXPECTED_MANIFEST' else b['runner_sha256']), conf_text)
    assert re.search(r'(?m)^SELFRUN_CONTAINER=%s$' % re.escape(s['container']), conf_text)
    assert re.search(r'(?m)^# conf    .*sha256=%s$' % sha(precheck_conf), Path(precheck_log).read_text()), \
        'the precheck log must name this conf by its sha256'
    strict = dict(b['regression'], log=str(strict_log), verdict='PASS', gpu_execution=False,
                  scope='full strict registered CPU regression on the native39 lifecycle build source tree')
    assert strict['log_sha256'] == sha(strict_log)
    focused = dict(b['focused'], log=str(focused_log))
    assert focused['log_sha256'] == sha(focused_log)
    details = load(Path(b['bundle']) / 'native30-provenance.json')['native39_lifecycle']
    queue_check = Path(bundle_selfrun).with_name('native39-lifecycle-queue-check.json')
    row = dict(
        name=b['name'], arm=b['arm'], source=b['source'], details=details,
        bundle=b['bundle'], manifest=b['manifest'], overlay_manifest=b['overlay_manifest'],
        coverage=b['checksummed_files'], review_sha256=b['review_sha256'], image=b['image'],
        runner=b['runner'], runner_sha256=b['runner_sha256'], runner_size=b['runner_size'],
        latest_entry_utc=b['latest_entry_utc'],
        engine_env_delta=b['engine_env_delta'], telemetry_delta=details['telemetry_delta'],
        parent=dict(name='q18a-native39-trunk', manifest=b['parent_manifest'], bundle=b['parent_bundle'],
                    overlay_manifest=f['parent_overlay_manifest'], review_sha256=f['review_sha256']),
        runner_parent=dict(path=f['runner']['parent'], sha256=f['runner']['parent_sha256']),
        runner_differences=f['runner']['differences'],
        freeze_proof=str(freeze), freeze_proof_sha256=sha(freeze),
        freeze_differences=dict(bundle_changed=f['changed'], bundle_added=f['added'], bundle_removed=f['removed'],
                                overlay_regenerated=f['overlay_regenerated'], engine_changed=f['engine_changed']),
        source_definitions={k: v['changed'] for k, v in f['freeze']['source']['definitions'].items()},
        source_compile=f['freeze']['source']['compile'], negative_controls=f['freeze']['negative_controls'],
        bundle_identity=f['freeze']['bundle'],
        compile_proof=str(compile_proof), compile_proof_sha256=sha(compile_proof), compile_files=c['files'],
        compile_failures=len(c['failures']),
        bundle_selfrun=str(bundle_selfrun), bundle_selfrun_sha256=sha(bundle_selfrun), bundle_selfrun_tests=s['tests_run'],
        bundle_selfrun_skips=s['skips'], bundle_selfrun_container=s['container'], bundle_selfrun_wall_s=s['wall_s'],
        lifecycle_queue_check=str(queue_check), lifecycle_queue_check_sha256=sha(queue_check),
        lifecycle_queue_cases=sorted(r['case'] for r in s['native39_lifecycle_queue_receipts']),
        trunk_queue_cases=sorted(r['case'] for r in s['native39_trunk_queue_receipts']),
        owner_prelaunch=s['owner_prelaunch'],
        engine_selfrun=str(engine_selfrun), engine_selfrun_sha256=sha(engine_selfrun),
        engine_selfrun_view=dict(label=e['label'], probe39=e['probe39'],
                                 probe37=dict(verdict=e['probe37']['verdict'], t2=e['probe37']['t2']),
                                 py_compile=e['py_compile'], suites=e['suites'], network=e['network'],
                                 gpu_device_requests=e['gpu_device_requests'], wall_s=e['wall_s']),
        precheck_conf=str(precheck_conf), precheck_conf_sha256=sha(precheck_conf), precheck=precheck_summary(precheck_log),
        precheck_log=str(precheck_log), precheck_log_sha256=sha(precheck_log),
        fresh_box_paths=list(FRESH_BOX_PATHS) + ['/home/<service-user>/harness/' + b['manifest'][:12]],
        owner_commands=['sudo python3 %s --staging --auth-env-file PRIVATE_AUTH_FILE --selfrun-record EXACT_BUNDLE_SELFRUN_JSON '
                        '--engine-selfrun-record EXACT_ENGINE_SELFRUN_JSON --snapshot-max-bytes 536870912 --sync none'
                        % Path(b['runner']).name,
                        'owner stops production (vllm-flash inactive/failed, four idle GPUs), then: sudo python3 %s --arm'
                        % Path(b['runner']).name],
        note=str(note), note_sha256=sha(note), strict=strict, focused=focused, verdict='PASS', classification=None,
        gpu_execution=False, box_action=False,
        scope='frozen input gates, the fresh strict regression, both pinned-image CPU self-runs, the compile proof and the '
              'owner precheck; no device run; cancellation, KV pressure, partial-tail and leak behaviour come only from the '
              'box run; the engine is the trunk engine in code with one declared telemetry env delta '
              '(FLASH_POOL_STATS=200), which makes no speed claim')
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (('delivery.json', row), ('frozen-gates.json', {k: v for k, v in row.items() if k not in ('strict', 'focused')}),
                        ('strict.json', strict)):
        with (out / name).open('x') as fh:
            json.dump(value, fh, sort_keys=True, indent=2)
            fh.write('\n')
    print(json.dumps(dict(out=str(out), verdict='PASS', manifest=b['manifest'][:12], delivery_sha256=sha(out / 'delivery.json')),
                     sort_keys=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('build', 'freeze', 'bundle-selfrun', 'engine-selfrun', 'compile-proof', 'precheck-conf', 'precheck-log',
                 'strict-log', 'focused-log', 'note', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    record(a.build, a.freeze, a.bundle_selfrun, a.engine_selfrun, a.compile_proof, a.precheck_conf, a.precheck_log,
           a.strict_log, a.focused_log, a.note, a.out)
