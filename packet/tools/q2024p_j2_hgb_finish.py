#!/usr/bin/env python3
"""Finish one q2024p-j2-hgb packet after its frozen CPU PASS and precheck PASS: host-gap runner negatives, owner
transport rendered by the standard v26 generator (never executed), wrapper compile pass and the delivery record.
No box command, GPU, VM, deletion or push.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import types

GENERATOR = Path('<scratchpad-root>/scratchpad/'
                 'make-trunk-box-scripts-v26.sh')
GENERATOR_SHA = '09bb1d54a6197ce8792b0362d73b0ffda2b39350a6aa2970972454ef3c1a2eed'
CAP_S = 4860


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(data, indent=2, sort_keys=True) + '\n')


def negatives(build, cpu_path, out):
    """The actual owner runner refuses frozen records whose host-gap module bindings are absent or altered."""
    bundle, runner = Path(build['bundle']), Path(build['runner'])
    sys.path[:0] = [str(bundle / 'tools'), str(bundle / 'e3_builder'), str(bundle)]
    module = types.ModuleType('hgb_owner_boundary')
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
    cases = {
        'hgb_source_modules_absent': lambda r: r.pop('hgb_source_modules'),
        'hgb_generated_modules_absent': lambda r: r.pop('hgb_generated_modules'),
        'hgb_source_small_piece_batch_sha': lambda r: r['hgb_source_modules']['small_piece_batch'].update(sha256='0' * 64),
        'hgb_generated_s5_startup_sha': lambda r: r['hgb_generated_modules']['s5_startup'].update(sha256='0' * 64),
        'hgb_source_module_missing': lambda r: r['hgb_source_modules'].pop('stage_wire'),
        'suite_small_piece_batch_absent': lambda r: (r['suite_counts'].pop('test_small_piece_batch'),
                                                     r.update(tests_run=r['tests_run'] - 26)),
    }
    for name, change in cases.items():
        record = json.loads(json.dumps(cpu))
        change(record)
        path = out / ('negative-' + name + '.json')
        dump(path, record)
        results[name] = attempt(path)
    assert results['actual'] == 'boundary', results
    assert all(value == 'refused:exact-n3j2-frozen-cpu-required' for key, value in results.items() if key != 'actual'), results
    assert calls == ['cpu_build'], calls
    result = dict(schema='q2024p-j2-hgb-runner-negatives/1', verdict='PASS', runner_sha256=sha(runner),
                  cpu_record_sha256=sha(cpu_path), results=results, box_action=False)
    dump(out / 'negatives.json', result)
    return result


def transport(build, cpu_path, out):
    """n3j2_transport.prepare for this packet's name: the v26 generator, its three calendar guards removed."""
    assert sha(GENERATOR) == GENERATOR_SHA, 'generator pin differs'
    name, bundle, runner, manifest = build['name'], Path(build['bundle']), Path(build['runner']), build['manifest']
    runner_sha, cpu_sha = sha(runner), sha(cpu_path)
    assert runner_sha == build['runner_sha256'] and bundle.name == name + '-bundle-' + manifest[:12]
    proof = json.loads(cpu_path.read_text())
    assert proof['verdict'] == 'PASS' and proof['runner_sha256'] == runner_sha and proof['bundle_manifest'] == manifest
    assert proof['container_rc'] == 0 and proof['skipped'] == 0 and proof['box_action'] is False
    out.mkdir()
    scripts = out / ('box-' + name + '-n1')
    scripts.mkdir()
    argv = ['bash', str(GENERATOR), name, str(bundle), manifest, str(runner), runner_sha,
            str(cpu_path), cpu_sha, str(cpu_path), cpu_sha, '2026-11-10T04:15:00+01:00', str(CAP_S)]
    done = subprocess.run(argv, env=dict(os.environ, GEN_OUT_DIR=str(scripts)), capture_output=True, text=True, check=True)
    (out / 'generate.log').write_text(done.stdout + done.stderr)
    chain = scripts / (name + '-chain.sh')
    before = sha(chain)
    lines = chain.read_text().splitlines(True)
    guards = [line for line in lines if line.startswith(('LE=$(date -d ', 'while [ "$(date +%s)" -lt "$NB" ]',
                                                         '[ "$(date +%s)" -lt "$LE" ]'))]
    assert len(guards) == 3
    chain.write_text(''.join(line for line in lines if line not in guards))
    text = chain.read_text()
    assert all(token not in text for token in ('LE=', 'NB=', '$LE', '$NB', 'PAST_LATEST_ENTRY'))
    for label, prefix, expected in (('runner', manifest[:12], runner), ('selfrun', cpu_sha[:12], cpu_path),
                                    ('engine-selfrun', cpu_sha[:12], cpu_path)):
        path = scripts / (name + '-stage-' + label + '-' + prefix + '.sh')
        encoded = path.read_text().split("PAYLOAD=$(cat <<'B64'\n", 1)[1].split('\nB64\n', 1)[0]
        assert base64.b64decode(encoded) == expected.read_bytes(), label
    for mode in ('staging', 'arm-launch'):
        body = (scripts / (name + '-' + mode + '.sh')).read_text()
        assert '/home/<service-user>/s0/' + name + '-owner-run.py' in body and ('--staging' if mode == 'staging' else '--arm') in body
    staging = (scripts / (name + '-staging.sh')).read_text()
    assert '--selfrun-record /home/<service-user>/s0/' + name + '-selfrun-record.json' in staging
    assert '--engine-selfrun-record /home/<service-user>/s0/' + name + '-engine-selfrun-record.json' in staging
    assert 'q18a-box-queue-waiter-v2.sh $Q %d' % CAP_S in text
    assert 'if idle ' in text and 'NOT_IDLE no arm' in text and 'NO_RESTORE' in text
    checked = {}
    for path in sorted(scripts.glob('*.sh')):
        subprocess.run(['bash', '-n', str(path)], capture_output=True, check=True)
        checked[path.name] = sha(path)
    result = dict(schema='q2024p-j2-hgb-owner-transport/1', state='PREPARED_NOT_EXECUTED', name=name,
                  source_commit=build['source_commit'], bundle=str(bundle), bundle_manifest=manifest, runner=str(runner),
                  runner_sha256=runner_sha, frozen_cpu=str(cpu_path), frozen_cpu_sha256=cpu_sha,
                  same_combined_proof_for_both_owner_arguments=True, scripts=str(scripts),
                  owner_command='bash ' + str(scripts / (name + '-seq.sh')), waiter_cap_s=CAP_S, generator_argv=argv,
                  generator_sha256=GENERATOR_SHA, before_chain_sha256=before, chain_sha256=sha(chain),
                  removed_date_guards=guards, bash_n=sorted(checked), script_sha256=checked,
                  box_action=False, gpu_action=False, vm_action=False)
    dump(out / 'transport.json', result)
    return result


def compile_pass(build, transport_record, out):
    """py_compile every Python file of the packet (bundle, runner) and bash -n every shell wrapper, into a fresh cfile dir."""
    bundle, runner = Path(build['bundle']), Path(build['runner'])
    cache = Path(tempfile.mkdtemp(prefix='hgb-compile-', dir=str(out.parent)))
    python = sorted(bundle.rglob('*.py')) + [runner]
    for index, path in enumerate(python):
        py_compile.compile(str(path), cfile=str(cache / ('%05d.pyc' % index)), doraise=True)
    shells = sorted(bundle.rglob('*.sh')) + sorted(Path(transport_record['scripts']).glob('*.sh'))
    for path in shells:
        subprocess.run(['bash', '-n', str(path)], capture_output=True, check=True)
    result = dict(schema='q2024p-j2-hgb-compile/1', verdict='PASS', python_files=len(python), shell_files=len(shells),
                  pyc_scratch=str(cache), runner_compiled=True, wrappers_bash_n=len(list(Path(transport_record['scripts']).glob('*.sh'))))
    dump(out, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.build.resolve()
    build = json.loads((root / 'build.json').read_text())
    cpu_path = root / 'frozen-cpu/result.json'
    precheck = json.loads((root / 'precheck/precheck.json').read_text())
    assert precheck['verdict'] == 'PASS' and precheck['cpu_record_sha256'] == sha(cpu_path)
    assert precheck['runner_sha256'] == build['runner_sha256'] == sha(build['runner'])
    neg = negatives(build, cpu_path, root / 'hgb-negatives')
    moved = transport(build, cpu_path, root / 'transport')
    comp = compile_pass(build, moved, root / 'compile-proof.json')
    delivery = dict(schema='q2024p-j2-hgb-delivery/1', state='READY_FOR_OWNER_GATE', exploratory=True,
                    name=build['name'], variant=build['variant'], small_piece_batch=build['small_piece_batch'],
                    source=build['source_commit'], branch='claude/n3-host-gap-batch', pushed=False,
                    bundle=build['bundle'], manifest=build['manifest'], overlay_manifest=build['overlay_manifest'],
                    runner=build['runner'], runner_sha256=build['runner_sha256'], runner_size=build['runner_size'],
                    cpu_record_sha256=sha(cpu_path), cpu_tests=json.loads(cpu_path.read_text())['tests_run'], cpu_skips=0,
                    precheck_sha256=sha(root / 'precheck/precheck.json'),
                    negatives_sha256=sha(root / 'hgb-negatives/negatives.json'),
                    transport_sha256=sha(root / 'transport/transport.json'), compile_sha256=sha(root / 'compile-proof.json'),
                    owner_command=moved['owner_command'], nominal_minutes=54, hard_minutes=81,
                    hardware_qualified=False, box_action=False, gpu_action=False)
    dump(root / 'delivery.json', delivery)
    files = [root / 'build.json', cpu_path, root / 'precheck/precheck.json', root / 'hgb-negatives/negatives.json',
             root / 'transport/transport.json', root / 'compile-proof.json', root / 'delivery.json', Path(build['runner']),
             Path(build['bundle']) / 'SHA256SUMS'] + sorted(Path(moved['scripts']).glob('*.sh'))
    with (root / 'DELIVERY-SHA256SUMS').open('x') as handle:
        handle.write(''.join(sha(p) + '  ' + str(p) + '\n' for p in files))
    print(json.dumps(dict(delivery, negatives=neg['results'], compile=comp), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
