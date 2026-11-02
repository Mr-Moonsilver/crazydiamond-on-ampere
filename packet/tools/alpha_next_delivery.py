#!/usr/bin/env python3
"""Bind the alpha-next delivery to both exact-byte CPU self-runs, the freezes, the comparison, the negative, the tamper control, the
clock scan, the compile proof, the precheck, the staging check, the engine diff against Step B's source and, when the lead has run
it, the VM record. One record; asserts, never repairs.

tools/step_b_delivery.py's assertions over the alpha-next records, with C0's shape where C0 had one (tools/alpha_c0_delivery.py):
the VM record is optional (--vm; absent means NOT_RUN in the record), the engine diff declares the added modules as 'A' rows, and the
freeze is alpha_next_freeze's record ('alpha-next-freeze/1', verdict PASS, written by tools/build_alpha_next.py). The staging check
is tools/alpha_next_staging_check.py (Step B's check). The dense container's left-out ids (declared clock-bound and robust-swap ids)
are read back from its own suites.log and the composed container's declared exclusions from its record. The hub-side queue pins
(tools/alpha_next_queue_pins.json, written by tools/alpha_next_queue_pins_write.py from this build's OUT) must hold exactly this
build's bundle, overlay and package identity (alpha_next_queue_pins_write.check: recomputed from OUT, same OUT, outside the build).
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PARENT_PINS = json.loads((ROOT / 'tools/alpha_next_parent.json').read_text())
STEP_B_MANIFEST = PARENT_PINS['manifest']
STEP_B_SOURCE = PARENT_PINS['source']
REVIEW = 'patches/s5_engine/native_candidate_review.json'
PINS = ROOT / 'tools/alpha_next_negative_pins.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def engine_diff(changed, added):
    """Every path under patches/ that differs from Step B's source: the freeze's changed and new modules and the review pin."""
    rows = git('diff', '--name-status', STEP_B_SOURCE, 'HEAD', '--', 'patches').splitlines()
    declared = ({'M\tpatches/s5_engine/' + n.split('/', 1)[1] for n in changed}
                | {'A\tpatches/s5_engine/' + n.split('/', 1)[1] for n in added} | {'M\t' + REVIEW})
    assert set(rows) == declared, sorted(set(rows) ^ declared)
    text = git('diff', STEP_B_SOURCE, 'HEAD', '--', 'patches/s5_engine', ':!' + REVIEW)
    return dict(parent_commit=git('rev-parse', STEP_B_SOURCE).strip(), head=git('rev-parse', 'HEAD').strip(), paths_differing=rows,
                stat=git('diff', '--stat', STEP_B_SOURCE, 'HEAD', '--', 'patches/s5_engine', ':!' + REVIEW).splitlines(),
                diff_sha256=hashlib.sha256(text.encode()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('build', 'freeze', 'engine', 'compare', 'negative', 'tamper', 'queue', 'clock_scan', 'compile_proof', 'precheck',
                 'conf', 'staging_check', 'note', 'out'):
        p.add_argument('--' + name.replace('_', '-'), dest=name, type=Path, required=True)
    p.add_argument('--vm', type=Path, help='the VM record, when the lead has run it')
    a = p.parse_args()
    sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
    import alpha_next_freeze as freeze_tool
    import alpha_next_gate as gate
    import alpha_next_robust_swap as swap
    import step_b_clock_bound as clock_bound
    b, fz, e, cmp_, neg, tam, q, k, c = [json.loads(path.read_text()) for path in (
        a.build, a.freeze, a.engine, a.compare, a.negative, a.tamper, a.queue, a.clock_scan, a.compile_proof)]
    assert b['schema'] == 'native43-build/1' and b['name'] == 'q18a-alpha-next' and b['parent_manifest'] == STEP_B_MANIFEST
    assert fz['schema'] == 'alpha-next-freeze/1' and fz['verdict'] == 'PASS'
    assert e['verdict'] == q['verdict'] == k['verdict'] == cmp_['verdict'] == neg['verdict'] == tam['verdict'] == 'PASS'
    assert e['schema'] == 'alpha-next-selfrun/1' and e['negative_control'] is False
    assert fz['candidate_manifest'] == b['manifest'] and fz['candidate_overlay_manifest'] == b['overlay_manifest']
    assert fz['parent_manifest'] == STEP_B_MANIFEST and fz['reference']['overlay_byte_identical_to_parent'] is True
    assert fz['negative_controls'] and set(fz['negative_controls'].values()) == {'REFUSED'}
    assert e['manifest'] == q['bundle_manifest'] == b['manifest'] and e['overlay_manifest'] == b['overlay_manifest']
    assert e['suites']['ok'] is True and e['skips'] == q['skips'] == 0 and all(x['verdict'] == 'PASS' for x in e['cases'].values())
    assert all(x['same'] for x in e['pairs'].values()) and e['pairs']
    assert e['dense_left_out']['ok'] is True and e['composed_left_out']
    assert not cmp_['differs'] and cmp_['candidate_record_sha256'] == sha(a.engine.parent / 'result.json')
    assert neg['negative_control'] is True and neg['manifest'] == b['manifest'] and not neg['reasons'] and neg['pinned_tests'] > 0
    assert tam['schema'] == 'alpha-next-overlay-tamper/1' and tam['candidate_manifest'] == b['overlay_manifest']
    assert all(x['ok'] for x in tam['cases'])
    assert q['frozen_runner_sha256'] == b['runner_sha256'] and q['tests_run'] == b['expected_selfrun_tests']
    assert q['alpha_next_queue'] == 'PASS' and q['step_b_queue'] == 'PASS' and q['owner_prelaunch']['verdict'] == 'PASS'
    assert q['clock_bound_frozen_excluded'] == clock_bound.IN_PROBE
    assert k['schema'] == 'alpha-next-clock-scan/1' and not k['scan']['problems'] and k['bundle_manifest'] == b['manifest']
    assert k['runner']['sha256'] == b['runner_sha256'] and k['negative_controls'] and all(k['negative_controls'].values())
    st = json.loads(a.staging_check.read_text())
    assert st['schema'] == 'alpha-next-staging-check/1' and st['verdict'] == 'PASS' and all(st['negative_controls_refused'].values())
    assert st['runner_sha256'] == b['runner_sha256'] == sha(b['runner'])
    assert st['selfrun_sha256'] == sha(a.queue) and st['engine_selfrun_sha256'] == sha(a.engine)
    # The hub-side queue pins (tools/alpha_next_queue_pins.json) hold exactly this build's bundle, overlay and package identity.
    import alpha_next_queue_pins_write as queue_pins
    qp = queue_pins.check(a.build.parent)
    assert qp['harness_bundle_sha256'] == b['manifest'] and qp['overlay_manifest_sha256'] == b['overlay_manifest']
    pins = json.loads(PINS.read_text())
    assert pins['schema'] == 'alpha-next-negative-pins/1' and pins['tests'] and pins['observation'], 'negative pins not observed yet'
    assert neg['pinned_tests'] == len(pins['tests']) and sha(pins['observation']['observed_json']) == pins['observation']['observed_sha256']
    dense_log = a.engine.parent / 'dense' / 'engine' / 'suites.log'
    dense_rec = json.loads((a.engine.parent / 'dense' / 'result.json').read_text())
    assert sha(dense_log) == dense_rec['suites']['log_sha256'] == e['dense_left_out']['suites_log_sha256'], 'dense suites log differs'
    proof = json.loads((Path(b['bundle']).parent / 'freeze-proof.json').read_text())
    assert sha(Path(b['bundle']).parent / 'freeze-proof.json') == b['freeze_proof_sha256'] and proof['verdict'] == 'PASS'
    assert sha(Path(b['bundle']) / 'SHA256SUMS') == b['manifest']
    summary = re.findall(r'^SUMMARY checks=(\d+) pass=(\d+) fail=(\d+)$', a.precheck.read_text(), re.M)
    assert len(summary) == 1 and summary[0][0] == summary[0][1] and summary[0][2] == '0', summary
    vm = json.loads(a.vm.read_text()) if a.vm else None
    assert vm is None or (vm.get('verdict') == 'PASS' and vm.get('bundle_manifest', b['manifest']) == b['manifest'])
    leaves = {n: json.loads((ROOT / 'harness' / n).read_text()) for n in (
        'step_b_fixture_leaves.json', 'alpha_c0_fixture_leaves.json', 'stage1_anatomy_fixture_leaves.json',
        'native41_held_g_fixture_leaves.json', 'alpha_next_stopfix_fixture_leaves.json', 'alpha_next_robust_fixture_leaves.json',
        'alpha_next_receipts_fixture_leaves.json', 'alpha_next_queue_fixture_leaves.json')}
    source = fz['source']
    row = dict(schema='alpha-next-delivery/1', verdict='PASS', classification=None, box_action=False, gpu_execution=False,
               name=b['name'], bundle=b['bundle'], manifest=b['manifest'], overlay_manifest=b['overlay_manifest'],
               review_sha256=b['review_sha256'], runner=b['runner'], runner_sha256=b['runner_sha256'], runner_size=b['runner_size'],
               source_commit=b['source_commit'], build_commit=b['source'], parent_bundle=b['parent_bundle'],
               parent_manifest=b['parent_manifest'], freeze_proof_sha256=b['freeze_proof_sha256'],
               engine_changed_vs_parent=b['engine_changed'], regression=b['regression'], focused=b['focused'],
               engine_sources_vs_step_b=engine_diff(freeze_tool.CHANGED, freeze_tool.NEW),
               step_b_base=dict(build=PARENT_PINS['build'], build_sha256=PARENT_PINS['build_sha256'], manifest=STEP_B_MANIFEST,
                                engine_commit=fz['reference']['engine_commit']),
               alpha_next_freeze=dict(path=str(a.freeze), sha256=sha(a.freeze), verdict=fz['verdict'], schema=fz['schema'],
                                      features=sorted(source.get('features') or {}), not_projected=source.get('not_projected'),
                                      negative_controls=sorted(fz['negative_controls']), tools_replaced=fz.get('tools_replaced'),
                                      reference=fz['reference']),
               engine_selfrun=dict(path=str(a.engine), sha256=sha(a.engine), verdict=e['verdict'], tests=e['tests'], skips=e['skips'],
                                   cases=sorted(e['cases']), connected_checks=e['connected_checks'],
                                   clock_split=e['clock_split']['verdict'], scheduler_tick_probe=(e.get('scheduler_tick_probe') or {}).get('verdict'),
                                   probe39=e['probe39']['verdict'], probe37=e['probe37']['verdict'], pairs=e['pairs'],
                                   declared_limit=e['declared_limit'], dense_left_out=e['dense_left_out'],
                                   composed_suites=e['composed_suites'], composed_left_out=e['composed_left_out']),
               comparison=dict(path=str(a.compare), sha256=sha(a.compare), verdict=cmp_['verdict'], compared=cmp_['compared'],
                               parent_record=cmp_['parent_record'], parent_record_sha256=cmp_['parent_record_sha256']),
               negative=dict(path=str(a.negative), sha256=sha(a.negative), verdict=neg['verdict'], pinned_tests=neg['pinned_tests'],
                             tests=neg['suites'].get('tests'), container_rc=neg['container_rc'], overlay_manifest=neg['overlay_manifest'],
                             pins=dict(count=len(pins['tests']), cases=len(pins['cases']), path='tools/alpha_next_negative_pins.json',
                                       sha256=sha(PINS), observation=pins['observation'], discovery=pins.get('discovery'))),
               queue_pins=qp,
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
                                  requires={k2: v['holds'] for k2, v in st['requires'].items()},
                                  negative_controls_refused=st['negative_controls_refused'], runner_sha256=st['runner_sha256'],
                                  selfrun_sha256=st['selfrun_sha256'], engine_selfrun_sha256=st['engine_selfrun_sha256']),
               vm=dict(path=str(a.vm), sha256=sha(a.vm), verdict=vm.get('verdict'), rc=vm.get('rc'), scope=vm.get('scope')) if vm
               else dict(verdict='NOT_RUN', why='no VM record was given'),
               fixture_leaves={n: dict(sha256=sha(ROOT / 'harness' / n), inherits=v.get('inherits'),
                                       names=[x['name'] for x in v.get('leaves', [])]) for n, v in leaves.items()},
               clock_bound_frozen=dict(reason=clock_bound.REASON, ids=sorted(clock_bound.CLOCK_BOUND_FROZEN),
                                       left_out_of_bundle_selfrun=q['clock_bound_frozen_excluded'],
                                       approval='operator, 2 Nov 2026 about 03:05 CET, through the lead'),
               robust_swap=dict(swapped=dict(swap.ROBUST_SWAP), superseded=dict(swap.SUPERSEDED),
                                module_sha256=sha(ROOT / 'harness/alpha_next_robust_swap.py')),
               expected_runtime=gate.EXPECTED_RUNTIME_S, box_budget_s=gate.BOX_BUDGET_S, active_s=gate.ACTIVE_S, cleanup_s=gate.CLEANUP_S,
               last_arm_entry_s=gate.LAST_ARM_ENTRY_S, last_tail_entry_s=getattr(gate, 'LAST_TAIL_ENTRY_S', None),
               start_window=dict(absolute=None, why='no absolute-time window: launch time is the lead\'s (operator, 2 Nov); '
                                 'the limits are relative to the run\'s own start (box_budget_s, last_arm_entry_s)'),
               note=str(a.note), note_sha256=sha(a.note), model_settings_changed=False, box_device_run='NOT_RUN')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as handle:
        handle.write(json.dumps(row, sort_keys=True, indent=1) + '\n')
    print(json.dumps({k2: row[k2] for k2 in ('verdict', 'manifest', 'runner_sha256', 'review_sha256')}, sort_keys=True))
    print('delivery sha256 ' + sha(a.out))


if __name__ == '__main__':
    main()
