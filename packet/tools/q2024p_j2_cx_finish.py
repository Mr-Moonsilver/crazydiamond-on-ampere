#!/usr/bin/env python3
"""Finish one q2024p-j2 composition packet (a cohort block or the dry packet) after its frozen CPU PASS and precheck
PASS: owner runner negatives (the night-pack cases plus the composition's moved engine modules and suites;
tools/q2024p_j2_yrs_finish.negatives), the owner transport rendered by the standard v26 generator (never executed;
tools/q2024p_j2_hgb_finish.transport unchanged), the wrapper compile pass and the delivery record with
DELIVERY-SHA256SUMS. No box command, GPU, VM, deletion or push.
"""
import argparse
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location('q2024p_j2_np_finish', Path(__file__).resolve().parent / 'q2024p_j2_np_finish.py')
nfin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nfin)
yfin, fin, sha, dump = nfin.yfin, nfin.fin, nfin.sha, nfin.dump
BRANCH = 'claude/n3-compose'
NOMINAL_MIN, HARD_MIN = 54, 81          # j2's four arms at 480 s; the composition adds no arm

CASES = dict(nfin.CASES)
for _module in ('stage_overlap', 'native_overlap_control'):
    CASES['cx_source_%s_sha' % _module] = (lambda m: lambda r: r['hgb_source_modules'][m].update(sha256='0' * 64))(_module)
    CASES['cx_generated_%s_sha' % _module] = (lambda m: lambda r: r['hgb_generated_modules'][m].update(sha256='0' * 64))(_module)
    CASES['cx_source_%s_missing' % _module] = (lambda m: lambda r: r['hgb_source_modules'].pop(m))(_module)
for _suite in ('test_append_no_block_stop', 'test_append_no_block_stop_stack', 'test_n3_decode_groups', 'test_dmp_reader',
               'test_q2024p_j2_cx'):
    CASES['suite_%s_absent' % _suite] = yfin._suite_absent(_suite)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.build.resolve()
    build = json.loads((root / 'build.json').read_text())
    assert build['schema'] == 'q2024p-j2-cx-build/1'
    cpu_path = root / 'frozen-cpu/result.json'
    cpu = json.loads(cpu_path.read_text())
    assert ({'engine_driver', 'scheduler_hooks', 'probe_control', 'ingress_incremental', 'native_auto_arm',
             'small_piece_batch', 'stage_overlap', 'native_overlap_control'}
            <= set(cpu['hgb_source_modules']) & set(cpu['hgb_generated_modules']))
    precheck = json.loads((root / 'precheck/precheck.json').read_text())
    assert precheck['verdict'] == 'PASS' and precheck['cpu_record_sha256'] == sha(cpu_path)
    assert precheck['runner_sha256'] == build['runner_sha256'] == sha(build['runner'])
    yfin.CASES = CASES
    neg = yfin.negatives(build, cpu_path, root / 'cx-negatives')
    moved = fin.transport(build, cpu_path, root / 'transport')
    comp = fin.compile_pass(build, moved, root / 'compile-proof.json')
    delivery = dict(schema='q2024p-j2-cx-delivery/1', state='READY_FOR_OWNER_GATE', exploratory=True,
                    family=build['family'], name=build['name'], variant=build['variant'], tag=build['tag'],
                    flags=build['flags'], diagnostic=build['diagnostic'], liveness_guard=build['liveness_guard'],
                    source=build['source_commit'], branch=BRANCH, pushed=False,
                    bundle=build['bundle'], manifest=build['manifest'], overlay_manifest=build['overlay_manifest'],
                    runner=build['runner'], runner_sha256=build['runner_sha256'], runner_size=build['runner_size'],
                    cpu_record_sha256=sha(cpu_path), cpu_tests=cpu['tests_run'], cpu_skips=0,
                    precheck_sha256=sha(root / 'precheck/precheck.json'),
                    negatives_sha256=sha(root / 'cx-negatives/negatives.json'), negative_cases=len(CASES),
                    transport_sha256=sha(root / 'transport/transport.json'), compile_sha256=sha(root / 'compile-proof.json'),
                    owner_command=moved['owner_command'], nominal_minutes=NOMINAL_MIN, hard_minutes=HARD_MIN,
                    hardware_qualified=False, box_action=False, gpu_action=False)
    dump(root / 'delivery.json', delivery)
    files = [root / 'build.json', cpu_path, root / 'precheck/precheck.json', root / 'cx-negatives/negatives.json',
             root / 'transport/transport.json', root / 'compile-proof.json', root / 'delivery.json', Path(build['runner']),
             Path(build['bundle']) / 'SHA256SUMS'] + sorted(Path(moved['scripts']).glob('*.sh'))
    with (root / 'DELIVERY-SHA256SUMS').open('x') as handle:
        handle.write(''.join(sha(p) + '  ' + str(p) + '\n' for p in files))
    print(json.dumps(dict(delivery, negatives=neg['results'], compile=comp), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
