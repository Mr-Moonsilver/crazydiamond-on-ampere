#!/usr/bin/env python3
"""Finish one q2024p-j2-dmp packet after its frozen CPU PASS and precheck PASS: runner negatives (the host-gap ones and
the decode-merge ones), owner transport rendered by the standard v26 generator (never executed), wrapper compile pass
and the delivery record. Reuses tools/q2024p_j2_hgb_finish.py's transport and compile pass (their record schemas keep
the hgb names); the waiter cap is the dmp hard budget (96 min = 5760 s; n3j2_gate.BOX_BUDGET_S in the bundle).
No box command, GPU, VM, deletion or push.
"""
import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('q2024p_j2_hgb_finish_base', ROOT / 'q2024p_j2_hgb_finish.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

NOMINAL_MIN, HARD_MIN = 64, 96
base.CAP_S = HARD_MIN * 60
BRANCH = 'claude/n3-decode-merge-probe'
REFUSED = 'refused:exact-n3j2-frozen-cpu-required'


def negatives(build, cpu_path, out):
    """The actual owner runner refuses frozen records whose engine module bindings or probe suites are absent or altered."""
    bundle, runner = Path(build['bundle']), Path(build['runner'])
    sys.path[:0] = [str(bundle / 'tools'), str(bundle / 'e3_builder'), str(bundle)]
    module = types.ModuleType('dmp_owner_boundary')
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
    counts = cpu['suite_counts']

    def attempt(record):
        try:
            module.main(['--staging', '--auth-env-file', '/public/not-read-auth.env', '--selfrun-record', str(record),
                         '--engine-selfrun-record', str(record), '--model', 'public-precheck-dynamic-model'])
        except Boundary:
            return 'boundary'
        except module.Refused as exc:
            return 'refused:' + str(exc)
        return 'returned'

    def drop_suite(name):
        def change(r):
            r['tests_run'] -= r['suite_counts'].pop(name)
        return change
    out.mkdir()
    results = {'actual': attempt(cpu_path)}
    cases = {
        'hgb_source_modules_absent': lambda r: r.pop('hgb_source_modules'),
        'hgb_generated_modules_absent': lambda r: r.pop('hgb_generated_modules'),
        'source_stage_overlap_sha': lambda r: r['hgb_source_modules']['stage_overlap'].update(sha256='0' * 64),
        'generated_stage_overlap_sha': lambda r: r['hgb_generated_modules']['stage_overlap'].update(sha256='0' * 64),
        'source_native_overlap_control_missing': lambda r: r['hgb_source_modules'].pop('native_overlap_control'),
        'generated_executor_hooks_sha': lambda r: r['hgb_generated_modules']['executor_hooks'].update(sha256='0' * 64),
        'suite_decode_merge_probe_absent': drop_suite('test_decode_merge_probe'),
        'suite_dmp_reader_absent': drop_suite('test_dmp_reader'),
        'suite_small_piece_batch_absent': drop_suite('test_small_piece_batch'),
    }
    for name, change in cases.items():
        record = json.loads(json.dumps(cpu))
        change(record)
        path = out / ('negative-' + name + '.json')
        base.dump(path, record)
        results[name] = attempt(path)
    assert results['actual'] == 'boundary', results
    assert all(value == REFUSED for key, value in results.items() if key != 'actual'), results
    assert calls == ['cpu_build'], calls
    result = dict(schema='q2024p-j2-dmp-runner-negatives/1', verdict='PASS', runner_sha256=base.sha(runner),
                  cpu_record_sha256=base.sha(cpu_path), suite_counts=counts, results=results, box_action=False)
    base.dump(out / 'negatives.json', result)
    return result


def transport_tagged(build, cpu_path, out, tag):
    """base.transport with the scripts directory box-<name>-<tag>: a second rendering (same packet, same pins) whose
    chain log is its own. On the box the packet's staging receipt and arm-run log are per QNAME, so a second run of the
    same packet is refused there once the first has staged; the rendering covers a first chain that stopped before."""
    source = inspect.getsource(base.transport)
    assert source.count("'-n1'") == 1, 'transport scripts-dir naming changed'
    space = dict(vars(base))
    exec(compile(source.replace("'-n1'", repr('-' + tag)), base.__file__, 'exec'), space)
    return space['transport'](build, cpu_path, out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.build.resolve()
    build = json.loads((root / 'build.json').read_text())
    assert build['schema'] == 'q2024p-j2-dmp-build/1' and build['hard_minutes'] == HARD_MIN
    assert build['nominal_minutes'] == NOMINAL_MIN
    cpu_path = root / 'frozen-cpu/result.json'
    precheck = json.loads((root / 'precheck/precheck.json').read_text())
    assert precheck['verdict'] == 'PASS' and precheck['cpu_record_sha256'] == base.sha(cpu_path)
    assert precheck['runner_sha256'] == build['runner_sha256'] == base.sha(build['runner'])
    gate = (Path(build['bundle']) / 'n3j2_gate.py').read_text()
    assert 'NOMINAL_S = %d * 60' % NOMINAL_MIN in gate, 'bundle gate budget differs'
    neg = negatives(build, cpu_path, root / 'dmp-negatives')
    moved = base.transport(build, cpu_path, root / 'transport')
    assert moved['waiter_cap_s'] == HARD_MIN * 60
    second = transport_tagged(build, cpu_path, root / 'transport-n2', 'n2')
    assert second['waiter_cap_s'] == HARD_MIN * 60 and second['scripts'].endswith('-n2')
    assert second['runner_sha256'] == moved['runner_sha256'] and second['frozen_cpu_sha256'] == moved['frozen_cpu_sha256']
    comp = base.compile_pass(build, moved, root / 'compile-proof.json')
    cpu = json.loads(cpu_path.read_text())
    delivery = dict(schema='q2024p-j2-dmp-delivery/1', state='READY_FOR_OWNER_GATE', exploratory=True,
                    name=build['name'], variant=build['variant'], probe_selection=build['probe_selection'],
                    decode_merge=build['decode_merge'], settings_added=build['settings_added'],
                    small_piece_batch=build['small_piece_batch'],
                    source=build['source_commit'], branch=BRANCH, pushed=False,
                    bundle=build['bundle'], manifest=build['manifest'], overlay_manifest=build['overlay_manifest'],
                    runner=build['runner'], runner_sha256=build['runner_sha256'], runner_size=build['runner_size'],
                    cpu_record_sha256=base.sha(cpu_path), cpu_tests=cpu['tests_run'], cpu_skips=cpu['skipped'],
                    precheck_sha256=base.sha(root / 'precheck/precheck.json'),
                    negatives_sha256=base.sha(root / 'dmp-negatives/negatives.json'),
                    transport_sha256=base.sha(root / 'transport/transport.json'),
                    compile_sha256=base.sha(root / 'compile-proof.json'),
                    owner_command=moved['owner_command'], waiter_cap_s=moved['waiter_cap_s'],
                    owner_command_n2=second['owner_command'],
                    transport_n2_sha256=base.sha(root / 'transport-n2/transport.json'),
                    n2_scope='same packet and pins, own scripts dir and chain log; the box refuses a second staging of '
                             'the same QNAME (staging receipt / arm-run log exist), so n2 serves a first chain that stopped '
                             'before staging',
                    nominal_minutes=NOMINAL_MIN, hard_minutes=HARD_MIN,
                    review='none for this packet: builder CPU tests only', hardware_qualified=False,
                    box_action=False, gpu_action=False)
    base.dump(root / 'delivery.json', delivery)
    files = [root / 'build.json', cpu_path, root / 'precheck/precheck.json', root / 'dmp-negatives/negatives.json',
             root / 'transport/transport.json', root / 'compile-proof.json', root / 'delivery.json', Path(build['runner']),
             Path(build['bundle']) / 'SHA256SUMS', root / 'transport-n2/transport.json'] + \
        sorted(Path(moved['scripts']).glob('*.sh')) + sorted(Path(second['scripts']).glob('*.sh'))
    with (root / 'DELIVERY-SHA256SUMS').open('x') as handle:
        handle.write(''.join(base.sha(p) + '  ' + str(p) + '\n' for p in files))
    print(json.dumps(dict(delivery, negatives=neg['results'], compile=comp), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
