#!/usr/bin/env python3
"""Finish one q2024p-j2-yr packet after its frozen CPU PASS and precheck PASS: owner runner negatives (the yrs ones plus
engine_driver, scheduler_hooks and the yield-rule suites; tools/q2024p_j2_yrs_finish.negatives with this packet's cases),
owner transport rendered by the standard v26 generator (never executed; tools/q2024p_j2_hgb_finish.transport unchanged),
wrapper compile pass and the delivery record.
No box command, GPU, VM, deletion or push.
"""
import argparse
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location('q2024p_j2_yrs_finish', Path(__file__).resolve().parent / 'q2024p_j2_yrs_finish.py')
yfin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(yfin)
fin, sha, dump = yfin.fin, yfin.sha, yfin.dump
BRANCH = 'claude/n3-yield-rules'

CASES = dict(yfin.CASES)
for _module in ('engine_driver', 'scheduler_hooks'):
    CASES['yr_source_%s_sha' % _module] = (lambda m: lambda r: r['hgb_source_modules'][m].update(sha256='0' * 64))(_module)
    CASES['yr_generated_%s_sha' % _module] = (lambda m: lambda r: r['hgb_generated_modules'][m].update(sha256='0' * 64))(_module)
    CASES['yr_source_%s_missing' % _module] = (lambda m: lambda r: r['hgb_source_modules'].pop(m))(_module)
for _suite in ('test_yield_no_append_owner', 'test_yield_window', 'test_q2024p_j2_yr_reader'):
    CASES['suite_%s_absent' % _suite] = yfin._suite_absent(_suite)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.build.resolve()
    build = json.loads((root / 'build.json').read_text())
    assert build['schema'] == 'q2024p-j2-yr-build/1'
    cpu_path = root / 'frozen-cpu/result.json'
    cpu = json.loads(cpu_path.read_text())
    assert {'engine_driver', 'scheduler_hooks'} <= set(cpu['hgb_source_modules']) & set(cpu['hgb_generated_modules'])
    precheck = json.loads((root / 'precheck/precheck.json').read_text())
    assert precheck['verdict'] == 'PASS' and precheck['cpu_record_sha256'] == sha(cpu_path)
    assert precheck['runner_sha256'] == build['runner_sha256'] == sha(build['runner'])
    yfin.CASES = CASES
    neg = yfin.negatives(build, cpu_path, root / 'yr-negatives')
    moved = fin.transport(build, cpu_path, root / 'transport')
    comp = fin.compile_pass(build, moved, root / 'compile-proof.json')
    delivery = dict(schema='q2024p-j2-yr-delivery/1', state='READY_FOR_OWNER_GATE', exploratory=True,
                    name=build['name'], variant=build['variant'], small_piece_batch=build['small_piece_batch'],
                    yield_reentry_settle=build['yield_reentry_settle'], no_append_owner=build['no_append_owner'],
                    yield_window=build['yield_window'], diagnostic=build['diagnostic'],
                    source=build['source_commit'], branch=BRANCH, pushed=False,
                    bundle=build['bundle'], manifest=build['manifest'], overlay_manifest=build['overlay_manifest'],
                    runner=build['runner'], runner_sha256=build['runner_sha256'], runner_size=build['runner_size'],
                    cpu_record_sha256=sha(cpu_path), cpu_tests=cpu['tests_run'], cpu_skips=0,
                    precheck_sha256=sha(root / 'precheck/precheck.json'),
                    negatives_sha256=sha(root / 'yr-negatives/negatives.json'), negative_cases=len(CASES),
                    transport_sha256=sha(root / 'transport/transport.json'), compile_sha256=sha(root / 'compile-proof.json'),
                    owner_command=moved['owner_command'], nominal_minutes=54, hard_minutes=81,
                    hardware_qualified=False, box_action=False, gpu_action=False)
    dump(root / 'delivery.json', delivery)
    files = [root / 'build.json', cpu_path, root / 'precheck/precheck.json', root / 'yr-negatives/negatives.json',
             root / 'transport/transport.json', root / 'compile-proof.json', root / 'delivery.json', Path(build['runner']),
             Path(build['bundle']) / 'SHA256SUMS'] + sorted(Path(moved['scripts']).glob('*.sh'))
    with (root / 'DELIVERY-SHA256SUMS').open('x') as handle:
        handle.write(''.join(sha(p) + '  ' + str(p) + '\n' for p in files))
    print(json.dumps(dict(delivery, negatives=neg['results'], compile=comp), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
