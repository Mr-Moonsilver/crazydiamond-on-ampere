#!/usr/bin/env python3
"""Finish one q2024p-j2-yrs packet after its frozen CPU PASS and precheck PASS: owner runner negatives (the hgb ones plus
native_auto_arm and the yield re-entry suite), owner transport rendered by the standard v26 generator (never executed;
tools/q2024p_j2_hgb_finish.transport unchanged), wrapper compile pass and the delivery record.
No box command, GPU, VM, deletion or push.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

_spec = importlib.util.spec_from_file_location('q2024p_j2_hgb_finish', Path(__file__).resolve().parent / 'q2024p_j2_hgb_finish.py')
fin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fin)
sha, dump = fin.sha, fin.dump
BRANCH = 'claude/n3-reentry-fast'


def _suite_absent(suite):
    def change(record):
        record.update(tests_run=record['tests_run'] - record['suite_counts'].pop(suite))
    return change


CASES = {
    'hgb_source_modules_absent': lambda r: r.pop('hgb_source_modules'),
    'hgb_generated_modules_absent': lambda r: r.pop('hgb_generated_modules'),
    'hgb_source_small_piece_batch_sha': lambda r: r['hgb_source_modules']['small_piece_batch'].update(sha256='0' * 64),
    'hgb_generated_s5_startup_sha': lambda r: r['hgb_generated_modules']['s5_startup'].update(sha256='0' * 64),
    'hgb_source_module_missing': lambda r: r['hgb_source_modules'].pop('stage_wire'),
    'yrs_source_native_auto_arm_sha': lambda r: r['hgb_source_modules']['native_auto_arm'].update(sha256='0' * 64),
    'yrs_generated_native_auto_arm_sha': lambda r: r['hgb_generated_modules']['native_auto_arm'].update(sha256='0' * 64),
    'yrs_source_native_auto_arm_missing': lambda r: r['hgb_source_modules'].pop('native_auto_arm'),
    'yrs_generated_native_auto_arm_missing': lambda r: r['hgb_generated_modules'].pop('native_auto_arm'),
    'suite_small_piece_batch_absent': _suite_absent('test_small_piece_batch'),
    'suite_yield_reentry_settle_absent': _suite_absent('test_yield_reentry_settle'),
    'suite_yrs_reader_absent': _suite_absent('test_q2024p_j2_yrs_reader'),
}


def negatives(build, cpu_path, out):
    """The actual owner runner refuses frozen records whose bound modules or suites are absent or altered."""
    bundle, runner = Path(build['bundle']), Path(build['runner'])
    sys.path[:0] = [str(bundle / 'tools'), str(bundle / 'e3_builder'), str(bundle)]
    module = types.ModuleType('yrs_owner_boundary')
    module.__file__ = str(runner)
    exec(compile(runner.read_bytes(), str(runner), 'exec'), module.__dict__)
    module.H = bundle
    module.preflight = lambda cap: None
    calls = []

    class Boundary(Exception):
        pass

    def stop():
        calls.append('cpu_build')
        raise Boundary()
    module.cpu_build = stop
    cpu = json.loads(cpu_path.read_text())
    assert 'native_auto_arm' in cpu['hgb_source_modules'] and 'native_auto_arm' in cpu['hgb_generated_modules']

    def attempt(record):
        try:
            module.main(['--staging', '--auth-env-file', '/public/not-read-auth.env', '--selfrun-record', str(record),
                         '--engine-selfrun-record', str(record), '--model', 'public-precheck-dynamic-model'])
        except Boundary:
            return 'boundary'
        except module.Refused as exc:
            return 'refused:' + str(exc)
        return 'returned'
    out.mkdir()
    results = {'actual': attempt(cpu_path)}
    for name, change in CASES.items():
        record = json.loads(json.dumps(cpu))
        change(record)
        path = out / ('negative-' + name + '.json')
        dump(path, record)
        results[name] = attempt(path)
    assert results['actual'] == 'boundary', results
    assert all(value == 'refused:exact-n3j2-frozen-cpu-required' for key, value in results.items() if key != 'actual'), results
    assert calls == ['cpu_build'], calls
    result = dict(schema='q2024p-j2-yrs-runner-negatives/1', verdict='PASS', runner_sha256=sha(runner),
                  cpu_record_sha256=sha(cpu_path), results=results, box_action=False)
    dump(out / 'negatives.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.build.resolve()
    build = json.loads((root / 'build.json').read_text())
    assert build['schema'] == 'q2024p-j2-yrs-build/1'
    cpu_path = root / 'frozen-cpu/result.json'
    precheck = json.loads((root / 'precheck/precheck.json').read_text())
    assert precheck['verdict'] == 'PASS' and precheck['cpu_record_sha256'] == sha(cpu_path)
    assert precheck['runner_sha256'] == build['runner_sha256'] == sha(build['runner'])
    neg = negatives(build, cpu_path, root / 'yrs-negatives')
    moved = fin.transport(build, cpu_path, root / 'transport')
    comp = fin.compile_pass(build, moved, root / 'compile-proof.json')
    delivery = dict(schema='q2024p-j2-yrs-delivery/1', state='READY_FOR_OWNER_GATE', exploratory=True,
                    name=build['name'], variant=build['variant'], small_piece_batch=build['small_piece_batch'],
                    yield_reentry_settle=build['yield_reentry_settle'], diagnostic=build['diagnostic'],
                    source=build['source_commit'], branch=BRANCH, pushed=False,
                    bundle=build['bundle'], manifest=build['manifest'], overlay_manifest=build['overlay_manifest'],
                    runner=build['runner'], runner_sha256=build['runner_sha256'], runner_size=build['runner_size'],
                    cpu_record_sha256=sha(cpu_path), cpu_tests=json.loads(cpu_path.read_text())['tests_run'], cpu_skips=0,
                    precheck_sha256=sha(root / 'precheck/precheck.json'),
                    negatives_sha256=sha(root / 'yrs-negatives/negatives.json'),
                    transport_sha256=sha(root / 'transport/transport.json'), compile_sha256=sha(root / 'compile-proof.json'),
                    owner_command=moved['owner_command'], nominal_minutes=54, hard_minutes=81,
                    hardware_qualified=False, box_action=False, gpu_action=False)
    dump(root / 'delivery.json', delivery)
    files = [root / 'build.json', cpu_path, root / 'precheck/precheck.json', root / 'yrs-negatives/negatives.json',
             root / 'transport/transport.json', root / 'compile-proof.json', root / 'delivery.json', Path(build['runner']),
             Path(build['bundle']) / 'SHA256SUMS'] + sorted(Path(moved['scripts']).glob('*.sh'))
    with (root / 'DELIVERY-SHA256SUMS').open('x') as handle:
        handle.write(''.join(sha(p) + '  ' + str(p) + '\n' for p in files))
    print(json.dumps(dict(delivery, negatives=neg['results'], compile=comp), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
