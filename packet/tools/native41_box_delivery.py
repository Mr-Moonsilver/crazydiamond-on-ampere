#!/usr/bin/env python3
"""Bind the native41 box delivery to both exact-byte CPU self-runs, the native40 negative, the held-d prior-candidate
negative, default-off and the client-clock scan."""
import argparse
import hashlib
import json
import re
from pathlib import Path


# Held-d's delivered overlay (bundle 2224909057e6): the engine box run 1's fatal
# TP agreement refusal came from, before the held-e STATE_FINALIZE fix.
PRIOR_OVERLAY_MANIFEST = 'a7b73bdf211362a690212902db7b2cd48103578e723fda4b4233334719a045ea'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('build', 'engine', 'queue', 'negative', 'prior_negative', 'default_off', 'clock_scan', 'precheck', 'conf',
                 'note', 'out'):
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
    assert (n['cases']['auto']['error'], n['cases']['receipts-auto']['error']) == (
        'ordinary-requests-trigger-native-cycle', 'auto-receipts-complete')
    assert off['candidate_sha256'] == e['cases']['off']['sha256']
    pn = json.loads(a.prior_negative.read_text())
    assert pn['schema'] == 'native41-selfrun/1' and pn['verdict'] == 'PASS' and pn['negative_kind'] == 'prior-candidate'
    assert pn['manifest'] == b['manifest'] and pn['overlay_manifest'] == PRIOR_OVERLAY_MANIFEST
    assert pn['cases']['auto']['verdict'] == 'PASS' and pn['cases']['tp-skew']['verdict'] == 'FAIL'
    assert pn['cases']['tp-skew']['error_type'] == 'DriverError'
    assert pn['cases']['tp-skew']['error'].startswith('TP workers disagree on the layer continuation')
    assert e['cases']['tp-skew']['verdict'] == 'PASS'
    k = json.loads(a.clock_scan.read_text())
    assert k['schema'] == 'native41-clock-scan/1' and k['verdict'] == 'PASS' and not k['scan']['problems']
    assert k['bundle_manifest'] == b['manifest'] and k['bundle'] == str(Path(b['bundle']).resolve())
    assert k['runner']['sha256'] == b['runner_sha256'] and k['setcfg_reference']
    assert k['scan']['unarmed_client_uses_container_bracket'] is True and not k['scan']['direct_references']
    assert not k['scan']['window_or_evidence_references']
    assert k['negative_controls'] and all(k['negative_controls'].values())
    assert {'e3_native40_unarmed_box', 'e3_native_performance'}.isdisjoint(k['scan']['closure'])
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
        prior_negative=dict(path=str(a.prior_negative), sha256=sha(a.prior_negative), verdict=pn['verdict'],
                            overlay_manifest=pn['overlay_manifest'],
                            cases={k: dict(verdict=v['verdict'], error_type=v['error_type'], error=v['error'])
                                   for k, v in sorted(pn['cases'].items())}),
        default_off=dict(path=str(a.default_off), sha256=sha(a.default_off), canonical_trace_sha256=off['canonical_trace_sha256']),
        clock_scan=dict(path=str(a.clock_scan), sha256=sha(a.clock_scan), verdict=k['verdict'],
                        closure=len(k['scan']['closure']), roots=k['scan']['roots'],
                        reachable_identity_calls=k['scan']['reachable_identity_calls'],
                        clock_identity_readers=k['scan']['clock_identity_readers'],
                        container_proof=k['scan']['container_proof'], negative_controls=sorted(k['negative_controls'])),
        box_actions_beyond_native40=['docker inspect --format {{.State.Pid}} <engine container id> (read-only)',
                                     'docker exec <engine container id> cat /proc/sys/kernel/random/boot_id (read-only)',
                                     'readlink /proc/<init pid>/ns/time and read /proc/<init pid>/timens_offsets (read-only)'],
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
