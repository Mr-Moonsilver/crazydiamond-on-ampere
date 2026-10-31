#!/usr/bin/env python3
"""Native37 container-shaped self-run: pinned image, CPU only, network none.

Mounts an overlay exactly as run-flash-s5-test.sh does with CORE_OVERLAY=1 and
S5=1 (core, e2, e3 and the e3/qsa source files; no vision, no ngram), plus the
harness and engine sources read-only, then runs inside one container:
  1. py_compile of every declared changed/added file (from the mounted tree),
  2. the T1/T2 admission probe against the mounted overlay,
  3. the hub suites (test_native37_robust and the suites the change touches)
     under the image's python, when --suites is given.
``--label held-c`` runs the same probe against the held-c overlay (negative
control); there the probe must FAIL and the run records it unaltered.
No GPU device request, no box action, nothing pushed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

IMAGE = 'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
V = '/usr/local/lib/python3.12/dist-packages/vllm/v1'
D = '/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia'
COMPILED = ('patches/s5_engine/scheduler_hooks.py', 'patches/s5_engine/engine_driver.py',
            'patches/s5_engine/native_overlap_session.py', 'harness/e3_native_performance.py',
            'harness/e3_native37_admission_probe.py', 'harness/test_native37_robust.py',
            'harness/test_native35_performance.py', 'harness/selftest.py',
            'tools/native37_robust_selfrun.py', 'tools/native37_robust_freeze.py',
            'tools/build_native37_robust.py')
# Pinned vLLM reference files the suites read from /source/reference/vllm-src.
REFERENCE_READS = ('v1/core/sched/scheduler.py', 'v1/core/block_pool.py', 'v1/engine/core.py',
                   'v1/worker/gpu_worker.py', 'models/qwen4_exp/nvidia/qsa.py')
SUITES = ('test_native37_robust', 'test_native35_performance', 'test_e3_staged_chunk_plan',
          'test_e3_admission_native', 'test_e3_native_overlap', 'test_e3_native_measurement_clients')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlay_mounts(overlay):
    o = Path(overlay)
    rows = {o/'core/block_pool.py': V+'/core/block_pool.py',
            o/'core/single_type_kv_cache_manager.py': V+'/core/single_type_kv_cache_manager.py',
            o/'core/scheduler.py': V+'/core/sched/scheduler.py',
            o/'e2': V+'/flash_e2',
            o/'e2_sources/scheduler_output.py': V+'/core/sched/output.py',
            o/'e2_sources/block_table.py': V+'/worker/gpu/block_table.py',
            o/'e3': V+'/flash_e3',
            o/'e3_sources/executor.py': V+'/executor/multiproc_executor.py',
            o/'e3_sources/core.py': V+'/engine/core.py',
            o/'e3_sources/worker.py': V+'/worker/gpu_worker.py',
            o/'e3_sources/cudagraph_utils.py': V+'/worker/gpu/cudagraph_utils.py',
            o/'e3_sources/sampler.py': V+'/worker/gpu/sample/sampler.py',
            o/'e3_sources/async_utils.py': V+'/worker/gpu/async_utils.py',
            o/'e3_sources/api_routers.py': '/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/launchers/api_server/routers.py',
            o/'e3_sources/model.py': D+'/model.py',
            o/'qsa.py': D+'/qsa.py', o/'ops_qsa.py': D+'/ops/qsa.py',
            o/'model_runner_v2.py': V+'/worker/gpu/model_runner.py'}
    for source in rows:
        assert source.exists(), source
    return rows


def inner_script(label, suites):
    compiled = ' '.join('/source/' + name for name in COMPILED)
    lines = ['set -u', 'cd /source/harness',
             'python3 -B -c "import py_compile,sys\nfor p in sys.argv[1:]: py_compile.compile(p,cfile=\'/tmp/pyc\',doraise=True)\nprint(\'PY_COMPILE\',len(sys.argv)-1,\'OK\')" '
             + compiled + ' > /out/py-compile.log 2>&1; echo "py_compile rc=$?" >> /out/py-compile.log',
             'python3 -B e3_native37_admission_probe.py --label %s --out /out/probe.json > /out/probe.log 2>&1; echo "probe rc=$?" >> /out/probe.log' % label]
    if suites:
        lines.append('python3 -B -m unittest -v ' + ' '.join(SUITES) + ' > /out/suites.log 2>&1; echo "suites rc=$?" >> /out/suites.log')
    return '\n'.join(lines) + '\n'


def source_mounts(tree, bundle):
    """The sources at /source: a frozen bundle or a git tree.

    A bundle does not ship the full pinned vLLM reference (no v1/core/sched/scheduler.py or
    block_pool.py), so the suites read it from the tree's pinned reference, the same files the
    hub strict regression reads; reference_pins() records each file read and checks every copy
    the bundle does carry (e3_reference, native31-fixtures/reference) is byte-identical.
    """
    if bundle is not None:
        return {bundle: '/source/harness', bundle/'e3_builder': '/source/patches/s5_engine',
                bundle/'native35_tools': '/source/tools',
                (tree/'reference/vllm-src').resolve(): '/source/reference/vllm-src'}
    return {tree/'harness': '/source/harness', tree/'patches': '/source/patches',
            tree/'tools': '/source/tools', tree/'reference/vllm-src': '/source/reference/vllm-src'}


def reference_pins(tree, bundle):
    ref = (tree/'reference/vllm-src').resolve()
    rows = {}
    for name in REFERENCE_READS:
        row = dict(sha256=sha(ref/name))
        if bundle is not None:
            for copy in ('e3_reference', 'native31-fixtures/reference'):
                if (bundle/copy/name).is_file():
                    assert sha(bundle/copy/name) == row['sha256'], 'bundle reference copy differs: %s/%s' % (copy, name)
                    row[copy] = 'identical'
                else:
                    row[copy] = 'absent'
        rows[name] = row
    return dict(root=str(ref), files=rows)


def run(tree, overlay, out, name, label, suites, bundle=None):
    tree, overlay, out = Path(tree).resolve(), Path(overlay).resolve(), Path(out).resolve()
    bundle = None if bundle is None else Path(bundle).resolve()
    if bundle is not None:
        assert overlay == bundle/'native30_overlay', 'a bundle self-run mounts its own overlay'
    pins = reference_pins(tree, bundle)
    out.mkdir()
    out.chmod(0o777)
    (out/'inner.sh').write_text(inner_script(label, suites))
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
    with (out/'container.log').open('x') as log:
        rc = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    inspect = json.loads(subprocess.check_output(['sudo', '-n', 'docker', 'inspect', name], text=True))[0]
    devices = inspect['HostConfig'].get('DeviceRequests') or []
    network = inspect['HostConfig'].get('NetworkMode')
    subprocess.check_call(['sudo', '-n', 'docker', 'rm', name], stdout=subprocess.DEVNULL)
    probe = json.loads((out/'probe.json').read_text()) if (out/'probe.json').exists() else {}
    compile_log = (out/'py-compile.log').read_text() if (out/'py-compile.log').exists() else ''
    suites_view = None
    if suites:
        text = (out/'suites.log').read_text()
        ran = [line for line in text.splitlines() if line.startswith('Ran ')]
        suites_view = dict(ran=ran[-1] if ran else None, ok=text.rstrip().endswith('suites rc=0'),
                           skipped=text.count('... skipped'), log_sha256=sha(out/'suites.log'))
    result = dict(schema='native37-robust-selfrun/1', label=label, container_rc=rc,
                  probe_verdict=probe.get('verdict'), probe_rc=probe.get('rc'),
                  t1=None if 't1' not in probe else {k: probe['t1'].get(k) for k in (
                      'verdict', 'cases', 'planned', 'refused', 'random_cases', 'random_refused',
                      'stock_scheduled_on_split', 'untyped_refusals', 'rule_mismatches', 'failure_count',
                      'w2_plannability')},
                  t2=None if 't2' not in probe else dict(verdict=probe['t2'].get('verdict'),
                      first_failure=probe['t2'].get('first_failure'),
                      checks=[(c['name'], c['verdict']) for c in probe['t2'].get('checks', [])]),
                  probe_error=probe.get('error'), mounted=probe.get('mounted'), geometry=probe.get('geometry'),
                  py_compile=dict(ok='PY_COMPILE %d OK' % len(COMPILED) in compile_log
                                  and compile_log.rstrip().endswith('rc=0'), files=len(COMPILED)),
                  suites=suites_view, probe_sha256=sha(out/'probe.json') if (out/'probe.json').exists() else None,
                  overlay=str(overlay), overlay_manifest=sha(overlay/'SHA256SUMS'),
                  reference=pins, bundle=None if bundle is None else str(bundle),
                  bundle_manifest=None if bundle is None else sha(bundle/'SHA256SUMS'),
                  tree_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tree, text=True).strip(),
                  tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain', '--', 'harness', 'patches', 'tools'],
                                                          cwd=tree, text=True).strip()),
                  image=IMAGE, network=network, gpu_device_requests=len(devices), box_action=False,
                  wall_s=round(time.monotonic() - started, 3), command=command)
    (out/'result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('label', 'container_rc', 'probe_verdict', 't1', 't2', 'py_compile',
                                              'suites', 'wall_s')}, sort_keys=True))
    print('result_sha256', sha(out/'result.json'))
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
    a = p.parse_args()
    raise SystemExit(run(a.tree, a.overlay, a.out, a.name, a.label, a.suites, a.bundle))
