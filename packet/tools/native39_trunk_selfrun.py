#!/usr/bin/env python3
"""Native39 container-shaped self-run: pinned image, CPU only, network none.

Mounts an overlay exactly as run-flash-s5-test.sh does with CORE_OVERLAY=1 and
S5=1 (native37_robust_selfrun.overlay_mounts, unchanged), plus the harness and
engine sources read-only, then runs inside one container:
  1. py_compile of every declared changed/added file (from the mounted tree),
  2. the native39 R/W/P probe against the mounted overlay,
  3. the native37 T1/T2 probe against the same overlay (regression),
  4. the hub suites the change touches under the image's python, with --suites.
``--label native38-held-b`` runs the probes against the native38 held-b overlay
(negative control); there R/W must FAIL and the run records it unaltered.
No GPU device request, no box action, nothing pushed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from native37_robust_selfrun import IMAGE, overlay_mounts, reference_pins, source_mounts

COMPILED = ('patches/s5_engine/scheduler_hooks.py', 'patches/s5_engine/engine_driver.py',
            'patches/s5_engine/native_overlap_session.py', 'harness/e3_native39_trunk_probe.py',
            'harness/test_native39_trunk.py', 'harness/selftest.py',
            'tools/native39_trunk_selfrun.py', 'tools/native39_trunk_freeze.py',
            'tools/build_native39_trunk.py')
SUITES = ('test_native39_trunk', 'test_native37_robust', 'test_native38_plain_bind',
          'test_e3_admission_native', 'test_e3_staged_chunk_plan', 'test_e3_native_overlap',
          'test_e3_native_storage_replay', 'test_e3_r4_worker_bridge', 'test_e3_admission_metadata')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inner_script(label, suites, compiled):
    files = ' '.join('/source/' + name for name in compiled)
    lines = ['set -u', 'cd /source/harness',
             'python3 -B -c "import py_compile,sys\nfor p in sys.argv[1:]: py_compile.compile(p,cfile=\'/tmp/pyc\',doraise=True)\nprint(\'PY_COMPILE\',len(sys.argv)-1,\'OK\')" '
             + files + ' > /out/py-compile.log 2>&1; echo "py_compile rc=$?" >> /out/py-compile.log',
             'python3 -B e3_native39_trunk_probe.py --label %s --out /out/probe39.json > /out/probe39.log 2>&1; echo "probe39 rc=$?" >> /out/probe39.log' % label,
             'python3 -B e3_native37_admission_probe.py --label %s --out /out/probe37.json > /out/probe37.log 2>&1; echo "probe37 rc=$?" >> /out/probe37.log' % label]
    if suites:
        lines.append('python3 -B -m unittest -v ' + ' '.join(SUITES) + ' > /out/suites.log 2>&1; echo "suites rc=$?" >> /out/suites.log')
    return '\n'.join(lines) + '\n'


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def run(tree, overlay, out, name, label, suites, bundle=None, compiled=COMPILED):
    tree, overlay, out = Path(tree).resolve(), Path(overlay).resolve(), Path(out).resolve()
    bundle = None if bundle is None else Path(bundle).resolve()
    if bundle is not None:
        assert overlay == bundle / 'native30_overlay', 'a bundle self-run mounts its own overlay'
    pins = reference_pins(tree, bundle)
    out.mkdir()
    out.chmod(0o777)
    (out / 'inner.sh').write_text(inner_script(label, suites, compiled))
    command = ['sudo', '-n', 'docker', 'run', '--pull=never', '--name', name, '--runtime', 'runc',
               '--network', 'none', '--cpus', '4', '--memory', '6g', '--memory-swap', '6g', '--read-only',
               '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g', '--ulimit', 'core=0',
               '-e', 'NVIDIA_VISIBLE_DEVICES=none', '-e', 'CUDA_VISIBLE_DEVICES=', '-e', 'PYTHONDONTWRITEBYTECODE=1',
               '-e', 'PYTHONPATH=/source/harness']
    mounts = dict(overlay_mounts(overlay))
    mounts.update(source_mounts(tree, bundle))
    for source, target in mounts.items():
        command += ['-v', '%s:%s:ro' % (source, target)]
    command += ['-v', '%s:/out' % out, '--entrypoint', 'bash', IMAGE, '/out/inner.sh']
    started = time.monotonic()
    with (out / 'container.log').open('x') as log:
        rc = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    inspect = json.loads(subprocess.check_output(['sudo', '-n', 'docker', 'inspect', name], text=True))[0]
    devices = inspect['HostConfig'].get('DeviceRequests') or []
    network = inspect['HostConfig'].get('NetworkMode')
    subprocess.check_call(['sudo', '-n', 'docker', 'rm', name], stdout=subprocess.DEVNULL)
    p39, p37 = load(out / 'probe39.json'), load(out / 'probe37.json')
    compile_log = (out / 'py-compile.log').read_text() if (out / 'py-compile.log').exists() else ''
    suites_view = None
    if suites:
        text = (out / 'suites.log').read_text()
        ran = [line for line in text.splitlines() if line.startswith('Ran ')]
        suites_view = dict(ran=ran[-1] if ran else None, ok=text.rstrip().endswith('suites rc=0'),
                           skipped=text.count('... skipped'), log_sha256=sha(out / 'suites.log'))

    def checks(view):
        return None if view is None else dict(verdict=view.get('verdict'), first_failure=view.get('first_failure'),
                                              checks=[(c['name'], c['verdict']) for c in view.get('checks', [])])
    result = dict(schema='native39-trunk-selfrun/1', label=label, container_rc=rc,
                  probe39=dict(verdict=p39.get('verdict'), rc=p39.get('rc'), error=p39.get('error'),
                               r=checks(p39.get('r')), w=checks(p39.get('w')), p=checks(p39.get('p')),
                               sha256=sha(out / 'probe39.json') if (out / 'probe39.json').exists() else None),
                  probe37=dict(verdict=p37.get('verdict'), rc=p37.get('rc'), error=p37.get('error'),
                               t1=None if 't1' not in p37 else {k: p37['t1'].get(k) for k in (
                                   'verdict', 'cases', 'planned', 'refused', 'stock_scheduled_on_split',
                                   'untyped_refusals', 'rule_mismatches', 'failure_count', 'w2_plannability')},
                               t2=checks(p37.get('t2')),
                               sha256=sha(out / 'probe37.json') if (out / 'probe37.json').exists() else None),
                  mounted=p39.get('mounted'), geometry=p39.get('geometry'),
                  py_compile=dict(ok='PY_COMPILE %d OK' % len(compiled) in compile_log
                                  and compile_log.rstrip().endswith('rc=0'), files=len(compiled)),
                  suites=suites_view, overlay=str(overlay), overlay_manifest=sha(overlay / 'SHA256SUMS'),
                  reference=pins, bundle=None if bundle is None else str(bundle),
                  bundle_manifest=None if bundle is None else sha(bundle / 'SHA256SUMS'),
                  tree_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tree, text=True).strip(),
                  tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain', '--', 'harness', 'patches', 'tools'],
                                                          cwd=tree, text=True).strip()),
                  image=IMAGE, network=network, gpu_device_requests=len(devices), box_action=False,
                  wall_s=round(time.monotonic() - started, 3), command=command)
    (out / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('label', 'container_rc', 'probe39', 'py_compile', 'suites', 'wall_s')},
                     sort_keys=True))
    print(json.dumps({'probe37': {k: result['probe37'][k] for k in ('verdict', 't2')}}, sort_keys=True))
    print('result_sha256', sha(out / 'result.json'))
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--overlay', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--suites', action='store_true')
    p.add_argument('--bundle', type=Path)
    p.add_argument('--dev', action='store_true', help='compile only the files that exist (development runs)')
    a = p.parse_args()
    files = COMPILED
    if a.dev:
        files = tuple(name for name in COMPILED if (a.tree / name).exists())
    raise SystemExit(run(a.tree, a.overlay, a.out, a.name, a.label, a.suites, a.bundle, files))
