#!/usr/bin/env python3
"""Bind the alpha-perf delivery to both exact-byte CPU self-runs, the freeze proof, the clock scan, the compile proof, the
precheck, the VM record and the engine-sources diff against claude/alpha-trunk. Writes one record; asserts, never repairs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HELD_F_MANIFEST = '15dffe62e5d425c66c06f90a3387f624ebe618ab181d71141f60c881b192c7c8'
REVIEW = 'patches/s5_engine/native_candidate_review.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def engine_diff(trunk):
    """Every path under patches/ that differs from the merged trunk commit; only the review pin may."""
    rows = git('diff', '--name-status', trunk, 'HEAD', '--', 'patches').splitlines()
    assert set(rows) <= {'M\t' + REVIEW}, rows
    head, base = (json.loads(git('show', c + ':' + REVIEW)) for c in ('HEAD', trunk))
    moved = sorted(k for k in set(head) | set(base) if head.get(k) != base.get(k))
    assert head['source_shas'] == base['source_shas'], 'review pins other engine source bytes'
    return dict(trunk_commit=git('rev-parse', trunk).strip(), head=git('rev-parse', 'HEAD').strip(), paths_differing=rows,
                review_keys_differing=moved, engine_source_shas_equal=True,
                meaning='every engine source file is byte-identical to the merged trunk commit; the review pin differs in '
                        'its harness shas, CPU receipts and the alpha_perf details, which each build must')


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('build', 'engine', 'queue', 'clock_scan', 'compile_proof', 'precheck', 'conf', 'note', 'vm', 'out'):
        p.add_argument('--' + name.replace('_', '-'), dest=name, type=Path, required=True)
    p.add_argument('--trunk', required=True, help='the claude/alpha-trunk commit merged')
    a = p.parse_args()
    sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
    import alpha_perf_gate as gate
    b, e, q, k, c = [json.loads(path.read_text()) for path in (a.build, a.engine, a.queue, a.clock_scan, a.compile_proof)]
    vm = json.loads(a.vm.read_text())
    assert b['schema'] == 'native43-build/1' and b['name'] == 'q18a-alpha-perf' and b['parent_manifest'] == HELD_F_MANIFEST
    assert e['verdict'] == q['verdict'] == k['verdict'] == 'PASS'
    assert e['schema'] == 'alpha-perf-selfrun/1' and e['negative_control'] is False
    assert e['manifest'] == q['bundle_manifest'] == b['manifest'] and e['overlay_manifest'] == b['overlay_manifest']
    assert e['suites']['ok'] is True and e['skips'] == q['skips'] == 0 and all(x['verdict'] == 'PASS' for x in e['cases'].values())
    assert q['frozen_runner_sha256'] == b['runner_sha256'] and q['tests_run'] == b['expected_selfrun_tests']
    assert q['alpha_perf_queue'] == 'PASS' and q['owner_prelaunch']['verdict'] == 'PASS'
    assert k['schema'] == 'alpha-perf-clock-scan/1' and not k['scan']['problems'] and k['bundle_manifest'] == b['manifest']
    assert k['runner']['sha256'] == b['runner_sha256'] and k['negative_controls'] and all(k['negative_controls'].values())
    proof = json.loads((Path(b['bundle']).parent / 'freeze-proof.json').read_text())
    assert sha(Path(b['bundle']).parent / 'freeze-proof.json') == b['freeze_proof_sha256'] and proof['verdict'] == 'PASS'
    assert sha(Path(b['bundle']) / 'SHA256SUMS') == b['manifest']
    summary = re.findall(r'^SUMMARY checks=(\d+) pass=(\d+) fail=(\d+)$', a.precheck.read_text(), re.M)
    assert len(summary) == 1 and summary[0][0] == summary[0][1] and summary[0][2] == '0', summary
    leaves = json.loads((ROOT / 'harness/alpha_perf_fixture_leaves.json').read_text())
    assert vm.get('verdict') == 'PASS' and vm.get('bundle_manifest', b['manifest']) == b['manifest']
    row = dict(schema='alpha-perf-delivery/1', verdict='PASS', classification=None, box_action=False, gpu_execution=False,
               name=b['name'], bundle=b['bundle'], manifest=b['manifest'], overlay_manifest=b['overlay_manifest'],
               review_sha256=b['review_sha256'], runner=b['runner'], runner_sha256=b['runner_sha256'], runner_size=b['runner_size'],
               source_commit=b['source_commit'], build_commit=b['source'], parent_bundle=b['parent_bundle'],
               parent_manifest=b['parent_manifest'], freeze_proof_sha256=b['freeze_proof_sha256'],
               engine_changed_vs_held_f=b['engine_changed'], regression=b['regression'], focused=b['focused'],
               engine_sources_vs_alpha_trunk=engine_diff(a.trunk),
               queue_selfrun=dict(path=str(a.queue), sha256=sha(a.queue), verdict=q['verdict'], rc=q.get('rc'),
                                  tests_run=q['tests_run'], skips=q['skips'], container=q.get('container'),
                                  frozen_runner_sha256=q['frozen_runner_sha256'], gpu_device_requests=q.get('gpu_device_requests')),
               engine_selfrun=dict(path=str(a.engine), sha256=sha(a.engine), verdict=e['verdict'], tests=e['tests'], skips=e['skips'],
                                   cases=sorted(e['cases']), connected_checks=e['connected_checks'],
                                   clock_split=e['clock_split']['verdict'], scheduler_tick_probe=(e.get('scheduler_tick_probe') or {}).get('verdict'),
                                   probe39=e['probe39']['verdict'], probe37=e['probe37']['verdict'], declared_limit=e['declared_limit']),
               clock_scan=dict(path=str(a.clock_scan), sha256=sha(a.clock_scan), verdict=k['verdict'],
                               negative_controls=sorted(k['negative_controls'])),
               compile_proof=dict(path=str(a.compile_proof), sha256=sha(a.compile_proof), **{x: c.get(x) for x in ('verdict', 'files', 'count', 'failed')}),
               precheck=dict(conf=str(a.conf), conf_sha256=sha(a.conf), log=str(a.precheck), log_sha256=sha(a.precheck),
                             checks=int(summary[0][0]), passed=int(summary[0][1]), failed=0),
               vm=dict(path=str(a.vm), sha256=sha(a.vm), verdict=vm.get('verdict'), rc=vm.get('rc'), scope=vm.get('scope')),
               fixture_leaves=dict(path='harness/alpha_perf_fixture_leaves.json', sha256=sha(ROOT / 'harness/alpha_perf_fixture_leaves.json'),
                                   inherits=leaves['inherits'], names=[x['name'] for x in leaves['leaves']]),
               expected_runtime=gate.EXPECTED_RUNTIME_S, box_budget_s=gate.BOX_BUDGET_S, active_s=gate.ACTIVE_S, cleanup_s=gate.CLEANUP_S,
               last_arm_entry_s=gate.LAST_ARM_ENTRY_S, latest_start_unix=gate.latest_arm(),
               note=str(a.note), note_sha256=sha(a.note), model_settings_changed=False, device_run='NOT_RUN')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as handle:
        handle.write(json.dumps(row, sort_keys=True, indent=1) + '\n')
    print(json.dumps({k2: row[k2] for k2 in ('verdict', 'manifest', 'runner_sha256', 'review_sha256')}, sort_keys=True))
    print('delivery sha256 ' + sha(a.out))


if __name__ == '__main__':
    main()
