#!/usr/bin/env python3
"""Finish one q2024p-j2 night-pack packet after its frozen CPU PASS and precheck PASS: owner runner negatives (the ing
packet's cases plus the family reader suite; tools/q2024p_j2_yrs_finish.negatives with this family's cases), the owner
transport rendered by the standard v26 generator (never executed; tools/q2024p_j2_hgb_finish.transport unchanged), the
wrapper compile pass and the delivery record with DELIVERY-SHA256SUMS.
No box command, GPU, VM, deletion or push.
"""
import argparse
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location('q2024p_j2_ing_finish', Path(__file__).resolve().parent / 'q2024p_j2_ing_finish.py')
ifin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ifin)
yfin, fin, sha, dump = ifin.yfin, ifin.fin, ifin.sha, ifin.dump
BRANCH = 'claude/n3-night-pack'

CASES = dict(ifin.CASES)
CASES['suite_test_q2024p_j2_np_reader_absent'] = yfin._suite_absent('test_q2024p_j2_np_reader')
FLAG_FIELDS = ('small_piece_batch', 'yield_reentry_settle', 'no_append_owner', 'yield_window', 'ingress_incremental',
               'ingress_verify_every', 'ingress_verify_async')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.build.resolve()
    build = json.loads((root / 'build.json').read_text())
    assert build['schema'] == 'q2024p-j2-np-build/1'
    cpu_path = root / 'frozen-cpu/result.json'
    cpu = json.loads(cpu_path.read_text())
    assert ({'engine_driver', 'scheduler_hooks', 'probe_control', 'ingress_incremental', 'native_auto_arm',
             'small_piece_batch'} <= set(cpu['hgb_source_modules']) & set(cpu['hgb_generated_modules']))
    precheck = json.loads((root / 'precheck/precheck.json').read_text())
    assert precheck['verdict'] == 'PASS' and precheck['cpu_record_sha256'] == sha(cpu_path)
    assert precheck['runner_sha256'] == build['runner_sha256'] == sha(build['runner'])
    yfin.CASES = CASES
    neg = yfin.negatives(build, cpu_path, root / 'np-negatives')
    moved = fin.transport(build, cpu_path, root / 'transport')
    comp = fin.compile_pass(build, moved, root / 'compile-proof.json')
    delivery = dict(schema='q2024p-j2-np-delivery/1', state='READY_FOR_OWNER_GATE', exploratory=True,
                    family=build['family'], name=build['name'], variant=build['variant'],
                    **{k: build[k] for k in FLAG_FIELDS},
                    prefix_counter=build['prefix_counter'], diagnostic=build['diagnostic'],
                    source=build['source_commit'], branch=BRANCH, pushed=False,
                    bundle=build['bundle'], manifest=build['manifest'], overlay_manifest=build['overlay_manifest'],
                    runner=build['runner'], runner_sha256=build['runner_sha256'], runner_size=build['runner_size'],
                    cpu_record_sha256=sha(cpu_path), cpu_tests=cpu['tests_run'], cpu_skips=0,
                    precheck_sha256=sha(root / 'precheck/precheck.json'),
                    negatives_sha256=sha(root / 'np-negatives/negatives.json'), negative_cases=len(CASES),
                    transport_sha256=sha(root / 'transport/transport.json'), compile_sha256=sha(root / 'compile-proof.json'),
                    owner_command=moved['owner_command'], nominal_minutes=54, hard_minutes=81,
                    hardware_qualified=False, box_action=False, gpu_action=False)
    dump(root / 'delivery.json', delivery)
    files = [root / 'build.json', cpu_path, root / 'precheck/precheck.json', root / 'np-negatives/negatives.json',
             root / 'transport/transport.json', root / 'compile-proof.json', root / 'delivery.json', Path(build['runner']),
             Path(build['bundle']) / 'SHA256SUMS'] + sorted(Path(moved['scripts']).glob('*.sh'))
    with (root / 'DELIVERY-SHA256SUMS').open('x') as handle:
        handle.write(''.join(sha(p) + '  ' + str(p) + '\n' for p in files))
    print(json.dumps(dict(delivery, negatives=neg['results'], compile=comp), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
