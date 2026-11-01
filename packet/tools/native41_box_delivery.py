#!/usr/bin/env python3
"""Bind the native41 box delivery to both exact-byte CPU self-runs, the native40 negative and default-off."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('build', 'engine', 'queue', 'negative', 'default_off', 'precheck', 'conf', 'note', 'out'):
        p.add_argument('--' + name.replace('_', '-'), dest=name, type=Path, required=True)
    a = p.parse_args()
    b, e, q, n, off = [json.loads(path.read_text()) for path in (a.build, a.engine, a.queue, a.negative, a.default_off)]
    assert b['box_ready'] is True and b['runner'] and b['queue']
    assert e['verdict'] == q['verdict'] == n['verdict'] == off['verdict'] == 'PASS'
    assert e['schema'] == n['schema'] == 'native41-selfrun/1' and e['negative_control'] is False
    assert e['manifest'] == q['bundle_manifest'] == b['manifest'] == n['manifest']
    assert e['overlay_manifest'] == q['overlay_manifest'] == b['overlay_manifest']
    assert n['overlay_manifest'] == b['reference_overlay_manifest'], 'negative must run over the native40 reference overlay'
    assert e['suites']['ok'] and e['suites']['skips'] == q['skips'] == 0
    assert sorted(e['cases']) == b['engine_cases'] and all(c['verdict'] == 'PASS' for c in e['cases'].values())
    assert q['frozen_runner_sha256'] == b['runner_sha256']
    assert q['tests_run'] == b['expected_selfrun_tests'] and q['owner_prelaunch']['verdict'] == 'PASS'
    assert n['negative_control'] is True
    assert (n['cases']['auto']['error'], n['cases']['window-auto']['error']) == (
        'ordinary-requests-trigger-native-cycle', 'session-action-not-exposed')
    assert off['candidate_sha256'] == e['cases']['off']['sha256']
    assert re.search(r'^SUMMARY checks=15 pass=15 fail=0$', a.precheck.read_text(), re.M)
    assert sha(Path(b['bundle']) / 'SHA256SUMS') == b['manifest']
    import sys
    sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'harness')]
    import native41_box_gate as gate
    row = dict(b, verdict='PASS', classification=None, box_action=False, gpu_execution=False,
        engine_selfrun=dict(path=str(a.engine), sha256=sha(a.engine), tests=e['suites']['tests'], skips=0,
                            cases={k: dict(verdict=v['verdict'], checks=v['checks']) for k, v in sorted(e['cases'].items())}),
        bundle_selfrun=dict(path=str(a.queue), sha256=sha(a.queue), tests=q['tests_run'], skips=0,
                            container=q['container'], owner_prelaunch=q['owner_prelaunch']),
        negative=dict(path=str(a.negative), sha256=sha(a.negative), verdict=n['verdict'], tests=n['suites']['tests'],
                      failures=n['suites']['failures'],
                      cases={k: dict(verdict=v['verdict'], error=v['error']) for k, v in sorted(n['cases'].items())}),
        default_off=dict(path=str(a.default_off), sha256=sha(a.default_off), canonical_trace_sha256=off['canonical_trace_sha256']),
        precheck=dict(conf=str(a.conf), conf_sha256=sha(a.conf), log=str(a.precheck), sha256=sha(a.precheck)),
        note=str(a.note), note_sha256=sha(a.note),
        expected_box_minutes='%d nominal; watchdog/window: native41_box_gate.ACTIVE_S=%d s plus CLEANUP_S=%d s, '
                             'latest arm %s' % (gate.EXPECTED_RUNTIME_S['nominal'] // 60, gate.ACTIVE_S, gate.CLEANUP_S,
                                                __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime(gate.latest_arm()))),
        model_settings_changed=False, automatic_native_admission='opt-in: FLASH_S5_AUTO_ARM=1 in the queue EXTRA_ENV only',
        owner_commands=['sudo -n python3 ' + b['runner'] + ' --staging --auth-env-file PRIVATE_AUTH_FILE '
                        '--selfrun-record ' + str(a.queue) + ' --engine-selfrun-record ' + str(a.engine)
                        + ' --snapshot-max-bytes 536870912 --sync none',
                        'owner controls stopped/idle arm: sudo -n python3 ' + b['runner'] + ' --arm'])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as handle:
        json.dump(row, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print('native41 box delivery PASS', b['manifest'])


if __name__ == '__main__':
    main()
