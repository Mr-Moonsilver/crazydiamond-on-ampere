#!/usr/bin/env python3
"""Bind the held-f box delivery to both exact-byte CPU self-runs, the held-e2-parent negative, the default-off
comparison against held-e2's frozen engine record, the freeze proof and the client-clock scan."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HELD_E2_MANIFEST = 'fc9fee4268ddac29603b9e538350a1b9444752a4b4c6dff23b864ba3aeaceb9f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('build', 'engine', 'queue', 'negative', 'default_off', 'clock_scan', 'precheck', 'conf', 'note', 'out'):
        p.add_argument('--' + name.replace('_', '-'), dest=name, type=Path, required=True)
    a = p.parse_args()
    sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
    import native41_progress_engine_selfrun as engine_tool
    import native41_progress_gate as gate
    b, e, q, n, off = [json.loads(path.read_text()) for path in (a.build, a.engine, a.queue, a.negative, a.default_off)]
    assert b['schema'] == 'native41-progress-build/1' and b['box_ready'] is True and b['runner'] and b['queue']
    assert b['parent_bundle_manifest'] == HELD_E2_MANIFEST
    assert e['verdict'] == q['verdict'] == n['verdict'] == off['verdict'] == 'PASS'
    assert e['schema'] == n['schema'] == engine_tool.SCHEMA and e['negative_control'] is False
    assert q['schema'] == 'native41-progress-bundle-selfrun/1'
    assert e['manifest'] == q['bundle_manifest'] == b['manifest'] == n['manifest']
    assert e['overlay_manifest'] == q['overlay_manifest'] == b['overlay_manifest']
    assert n['negative_control'] is True and n['negative_kind'] == 'held-e2-parent'
    assert n['overlay_manifest'] == b['reference_overlay_manifest'], 'negative must run over the held-e2 reference overlay'
    assert e['suites']['ok'] and e['suites']['skips'] == q['skips'] == 0
    assert sorted(e['cases']) == b['engine_cases'] and all(c['verdict'] == 'PASS' for c in e['cases'].values())
    assert {m: (c['verdict'], c['error_type'], c['error']) for m, c in n['cases'].items()} == engine_tool.NEGATIVE_CASES
    assert not n['reasons'] and n['suites']['tests'] - len(engine_tool.NEGATIVE_TESTS) > 0
    assert q['frozen_runner_sha256'] == b['runner_sha256']
    assert q['tests_run'] == b['expected_selfrun_tests'] and q['owner_prelaunch']['verdict'] == 'PASS'
    assert off['schema'] == 'native41-progress-default-off-comparison/1' and not off['differs']
    assert off['candidate_record_sha256'] == sha(a.engine) and off['parent_record_sha256'] == engine_tool.HELD_E2_ENGINE_SHA
    proof = json.loads((Path(b['bundle']).parent / 'freeze-proof.json').read_text())
    assert sha(Path(b['bundle']).parent / 'freeze-proof.json') == b['freeze_proof_sha256'] and proof['verdict'] == 'PASS'
    assert proof['source']['default_off']['verdict'] == 'PASS' and proof['overlay']['executor_off']['verdict'] == 'PASS'
    k = json.loads(a.clock_scan.read_text())
    assert k['schema'] == 'native41-progress-clock-scan/1' and k['verdict'] == 'PASS' and not k['scan']['problems']
    assert k['bundle_manifest'] == b['manifest'] and k['bundle'] == str(Path(b['bundle']).resolve())
    assert k['runner']['sha256'] == b['runner_sha256'] and k['setcfg_reference']
    assert k['scan']['unarmed_client_uses_container_bracket'] is True and not k['scan']['direct_references']
    assert not k['scan']['window_or_evidence_references']
    assert k['negative_controls'] and all(k['negative_controls'].values())
    assert {'e3_native40_unarmed_box', 'e3_native_performance'}.isdisjoint(k['scan']['closure'])
    assert re.search(r'^SUMMARY checks=15 pass=15 fail=0$', a.precheck.read_text(), re.M)
    assert sha(Path(b['bundle']) / 'SHA256SUMS') == b['manifest']
    row = dict(b, verdict='PASS', classification=None, box_action=False, gpu_execution=False,
        engine_selfrun=dict(path=str(a.engine), sha256=sha(a.engine), tests=e['suites']['tests'], skips=0,
                            cases={m: dict(verdict=v['verdict'], checks=v['checks'], canonical_sha256=v['canonical_sha256'])
                                   for m, v in sorted(e['cases'].items())}),
        bundle_selfrun=dict(path=str(a.queue), sha256=sha(a.queue), tests=q['tests_run'], skips=0,
                            container=q['container'], owner_prelaunch=q['owner_prelaunch']),
        negative=dict(path=str(a.negative), sha256=sha(a.negative), verdict=n['verdict'], kind=n['negative_kind'],
                      suites=n['suites'], outcomes_sha256=n['outcomes_sha256'],
                      cases={m: dict(verdict=v['verdict'], error_type=v['error_type'], error=v['error'])
                             for m, v in sorted(n['cases'].items())}),
        default_off=dict(path=str(a.default_off), sha256=sha(a.default_off), verdict=off['verdict'],
                         compared=sorted(m for m, r in off['modes'].items() if r['compared']),
                         parent_record=off['parent_record'], parent_record_sha256=off['parent_record_sha256']),
        freeze=dict(default_off_functions=proof['source']['default_off']['functions'],
                    executor_reverted_rows=proof['overlay']['executor_off']['reverted_rows']),
        clock_scan=dict(path=str(a.clock_scan), sha256=sha(a.clock_scan), verdict=k['verdict'],
                        closure=len(k['scan']['closure']), roots=k['scan']['roots'],
                        reachable_identity_calls=k['scan']['reachable_identity_calls'],
                        clock_identity_readers=k['scan']['clock_identity_readers'],
                        container_proof=k['scan']['container_proof'], negative_controls=sorted(k['negative_controls'])),
        box_actions_beyond_held_e=[],
        precheck=dict(conf=str(a.conf), conf_sha256=sha(a.conf), log=str(a.precheck), sha256=sha(a.precheck)),
        note=str(a.note), note_sha256=sha(a.note),
        expected_box_minutes='%d nominal; watchdog/window: native41_progress_gate.ACTIVE_S=%d s plus CLEANUP_S=%d s, '
                             'latest arm %s' % (gate.EXPECTED_RUNTIME_S['nominal'] // 60, gate.ACTIVE_S, gate.CLEANUP_S,
                                                time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(gate.latest_arm()))),
        model_settings_changed=False,
        progress_setting='FLASH_S5_AUTO_ARM=1 FLASH_S5_NATIVE_PROGRESS=burst in the queue EXTRA_ENV only; '
                         'each arm selects current or burst through native_progress while paused idle FULL',
        owner_commands=['sudo -n python3 ' + b['runner'] + ' --staging --auth-env-file PRIVATE_AUTH_FILE '
                        '--selfrun-record ' + str(a.queue) + ' --engine-selfrun-record ' + str(a.engine)
                        + ' --snapshot-max-bytes 536870912 --sync none',
                        'owner controls stopped/idle arm: sudo -n python3 ' + b['runner'] + ' --arm'])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as handle:
        json.dump(row, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print('held-f box delivery PASS', b['manifest'])


if __name__ == '__main__':
    main()
