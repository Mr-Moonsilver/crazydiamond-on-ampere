#!/usr/bin/env python3
"""Bind native40 box delivery to both exact-byte CPU gates and parent negative."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('build', 'engine', 'queue', 'negative', 'precheck', 'conf', 'note', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    b, e, q, n = [json.loads(path.read_text()) for path in (a.build, a.engine, a.queue, a.negative)]
    assert b['box_ready'] is True and b['runner'] and b['queue']
    assert e['verdict'] == q['verdict'] == n['verdict'] == 'PASS'
    assert e['manifest'] == q['bundle_manifest'] == b['manifest']
    assert e['overlay_manifest'] == q['overlay_manifest'] == b['overlay_manifest']
    assert e['suites']['ok'] and e['suites']['skips'] == q['skips'] == 0
    assert e['connected']['verdict'] == 'PASS'
    assert q['frozen_runner_sha256'] == b['runner_sha256']
    assert q['tests_run'] == b['expected_selfrun_tests'] and q['owner_prelaunch']['verdict'] == 'PASS'
    assert n['negative_control'] is True and n['connected']['error_type'] == 'UnsupportedRoute'
    assert re.search(r'^SUMMARY checks=15 pass=15 fail=0$', a.precheck.read_text(), re.M)
    assert sha(Path(b['bundle']) / 'SHA256SUMS') == b['manifest']
    row = dict(b, verdict='PASS', classification=None, box_action=False, gpu_execution=False,
        engine_selfrun=dict(path=str(a.engine), sha256=sha(a.engine), tests=e['suites']['tests'], skips=0,
                            connected_checks=len(e['connected']['checks'])),
        bundle_selfrun=dict(path=str(a.queue), sha256=sha(a.queue), tests=q['tests_run'], skips=0,
                            container=q['container'], owner_prelaunch=q['owner_prelaunch']),
        negative=dict(path=str(a.negative), sha256=sha(a.negative), verdict=n['verdict']),
        precheck=dict(conf=str(a.conf), conf_sha256=sha(a.conf), log=str(a.precheck), sha256=sha(a.precheck)),
        note=str(a.note), note_sha256=sha(a.note), expected_box_minutes='45–50 nominal; watchdog/window derive from native40_box_gate.PHASES, plus existing cleanup allowance',
        model_settings_changed=False, automatic_native_admission=False,
        owner_commands=['sudo -n python3 ' + b['runner'] + ' --staging --auth-env-file PRIVATE_AUTH_FILE '
                        '--selfrun-record ' + str(a.queue) + ' --engine-selfrun-record ' + str(a.engine)
                        + ' --snapshot-max-bytes 536870912 --sync none',
                        'owner controls stopped/idle arm: sudo -n python3 ' + b['runner'] + ' --arm'])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as handle:
        json.dump(row, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print('native40 box delivery PASS', b['manifest'])


if __name__ == '__main__':
    main()
