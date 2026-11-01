#!/usr/bin/env python3
"""Native43 le-fix hub self-run: the le-fix suites and the vLLM-logging probe in the pinned image.

Container-shaped like tools/native37_robust_selfrun.py: pinned image, CPU only, network none, read-only
sources.  It runs:
  1. py_compile of the le-fix files under the image's python;
  2. the le-fix suites test_native43_engine_death, test_native43_le_fix_queue_deltas, test_native43_le_fatal_log
     (all must pass), and a parity check: the neighbouring test_native38_plain_bind (its FatalReplies uses the
     same fatal channel) runs in the same container over this tree and over the parent tree (git archive of
     299f27e), and every test's outcome must be identical.  The parent's container-only failures are not
     le-fix regressions; they are reported, not hidden;
  3. tools/native43_le_fix_logprobe.py with two packages under vllm.v1.  flash_e3 is the l-e overlay e3/
     with executor_hooks.py replaced by make_s5_overlap.package(le-fix source).  flash_e3_prefix is the
     l-e overlay e3/ as it ran.  Both are copies made on the hub after their hashes are checked.
Before the container, it checks on the hub that make_s5_overlap.package(the l-e source at the parent
commit) equals the l-e overlay's e3/executor_hooks.py byte for byte, and that no other e3/ file differs.
No GPU, no box action, no push.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from native37_robust_selfrun import IMAGE  # noqa: E402

V = '/usr/local/lib/python3.12/dist-packages/vllm/v1'
PARENT = '299f27e'
COMPILED = ('harness/native43_engine_death.py', 'harness/test_native43_engine_death.py',
            'harness/test_native43_le_fatal_log.py', 'harness/test_native43_le_fix_queue_deltas.py',
            'patches/s5_engine/executor_hooks.py', 'tools/native43_le_fix_queue_deltas.py',
            'tools/native43_le_fix_logprobe.py', 'tools/native43_le_fix_selfrun.py')
SUITES = ('test_native43_engine_death', 'test_native43_le_fix_queue_deltas', 'test_native43_le_fatal_log')
PARITY = 'test_native38_plain_bind'
OUTCOMES = '''import json, sys, unittest
class R(unittest.TextTestResult):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.rows = {}
    def addSuccess(self, t): super().addSuccess(t); self.rows[t.id()] = 'ok'
    def addError(self, t, e): super().addError(t, e); self.rows[t.id()] = 'error'
    def addFailure(self, t, e): super().addFailure(t, e); self.rows[t.id()] = 'fail'
    def addSkip(self, t, r): super().addSkip(t, r); self.rows[t.id()] = 'skip'
    def addSubTest(self, t, s, e):
        super().addSubTest(t, s, e)
        if e is not None:
            self.rows[s.id()] = 'fail' if issubclass(e[0], t.failureException) else 'error'
            self.rows[t.id()] = 'subtest-not-ok'
# Guarded: spawn-context workers (FatalReplies) re-import this main module as __mp_main__.
if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[1])
    result = unittest.TextTestRunner(resultclass=R, verbosity=2, stream=sys.stderr).run(suite)
    json.dump(result.rows, open(sys.argv[2], 'w'), indent=1, sort_keys=True)
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_sha(folder):
    rows = {}
    for path in sorted(Path(folder).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            rows[str(path.relative_to(folder))] = sha(path)
    return rows


def packages(tree, overlay, out):
    sys.path[:0] = [str(tree / 'patches/s5_engine'), str(tree / 'harness')]
    import make_s5_overlap
    head = subprocess.check_output(['git', 'show', PARENT + ':patches/s5_engine/executor_hooks.py'], cwd=tree, text=True)
    original = (overlay / 'e3/executor_hooks.py').read_text()
    assert make_s5_overlap.package(head) == original, 'the l-e overlay executor_hooks.py is not package(parent source)'
    before = tree_sha(overlay / 'e3')
    prefix, fixed = out / 'pkg-prefix', out / 'pkg-fixed'
    shutil.copytree(overlay / 'e3', prefix)
    shutil.copytree(overlay / 'e3', fixed)
    (fixed / 'executor_hooks.py').write_text(make_s5_overlap.package((tree / 'patches/s5_engine/executor_hooks.py').read_text()))
    assert tree_sha(prefix) == before and tree_sha(overlay / 'e3') == before
    changed = sorted(k for k, v in tree_sha(fixed).items() if before.get(k) != v)
    assert changed == ['executor_hooks.py'], changed
    return dict(overlay_e3_files=len(before), overlay_manifest=sha(overlay / 'SHA256SUMS'),
                parent_package_equals_overlay=True, fixed_changed=changed,
                fixed_executor_hooks_sha256=sha(fixed / 'executor_hooks.py'),
                prefix_executor_hooks_sha256=before['executor_hooks.py']), prefix, fixed


def inner_script():
    compiled = ' '.join('/source/' + name for name in COMPILED)
    return '\n'.join([
        'set -u', 'cd /source/harness',
        'python3 -B -c "import py_compile,sys\nfor p in sys.argv[1:]: py_compile.compile(p,cfile=\'/tmp/pyc\',doraise=True)\n'
        'print(\'PY_COMPILE\',len(sys.argv)-1,\'OK\')" ' + compiled + ' > /out/py-compile.log 2>&1; echo "py_compile rc=$?" >> /out/py-compile.log',
        'python3 -B -m unittest -v ' + ' '.join(SUITES) + ' > /out/suites.log 2>&1; echo "suites rc=$?" >> /out/suites.log',
        'python3 -B /out/outcomes.py %s /out/parity-fixed.json > /out/parity-fixed.log 2>&1' % PARITY,
        'cd /parent/harness && PYTHONPATH=/parent/harness python3 -B /out/outcomes.py %s /out/parity-parent.json '
        '> /out/parity-parent.log 2>&1; cd /source/harness' % PARITY,
        'python3 -B /source/tools/native43_le_fix_logprobe.py --out /out/logprobe.json > /out/logprobe.log 2>&1; '
        'echo "logprobe rc=$?" >> /out/logprobe.log',
    ]) + '\n'


def run(tree, overlay, out, name):
    tree, overlay, out = Path(tree).resolve(), Path(overlay).resolve(), Path(out).resolve()
    out.mkdir()
    started = time.monotonic()
    pins, prefix, fixed = packages(tree, overlay, out)
    for folder in (out, prefix, fixed):
        folder.chmod(0o777 if folder == out else 0o755)
    (out / 'inner.sh').write_text(inner_script())
    (out / 'outcomes.py').write_text(OUTCOMES)
    parent = out / 'parent-tree'
    parent.mkdir()
    archive = subprocess.run(['git', 'archive', PARENT, 'harness', 'patches', 'tools'], cwd=tree, check=True,
                             capture_output=True).stdout
    subprocess.run(['tar', '-x', '-C', str(parent)], input=archive, check=True)
    command = ['sudo', '-n', 'docker', 'run', '--pull=never', '--name', name, '--runtime', 'runc',
               '--network', 'none', '--cpus', '4', '--memory', '6g', '--memory-swap', '6g', '--read-only',
               '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g', '--ulimit', 'core=0',
               '-e', 'NVIDIA_VISIBLE_DEVICES=none', '-e', 'CUDA_VISIBLE_DEVICES=', '-e', 'PYTHONDONTWRITEBYTECODE=1',
               '-e', 'PYTHONPATH=/source/harness']
    mounts = [(tree / 'harness', '/source/harness'), (tree / 'patches', '/source/patches'), (tree / 'tools', '/source/tools'),
              ((tree / 'reference/vllm-src').resolve(), '/source/reference/vllm-src'),
              (fixed, V + '/flash_e3'), (prefix, V + '/flash_e3_prefix'),
              (parent / 'harness', '/parent/harness'), (parent / 'patches', '/parent/patches'),
              (parent / 'tools', '/parent/tools'), ((tree / 'reference/vllm-src').resolve(), '/parent/reference/vllm-src')]
    for source, target in mounts:
        command += ['-v', '%s:%s:ro' % (source, target)]
    command += ['-v', '%s:/out' % out, '--entrypoint', 'bash', IMAGE, '/out/inner.sh']
    with (out / 'container.log').open('x') as log:
        rc = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    inspect = json.loads(subprocess.check_output(['sudo', '-n', 'docker', 'inspect', name], text=True))[0]
    devices = inspect['HostConfig'].get('DeviceRequests') or []
    network = inspect['HostConfig'].get('NetworkMode')
    subprocess.check_call(['sudo', '-n', 'docker', 'rm', name], stdout=subprocess.DEVNULL)
    text = (out / 'suites.log').read_text() if (out / 'suites.log').exists() else ''
    ran = [line for line in text.splitlines() if line.startswith('Ran ')]
    compile_log = (out / 'py-compile.log').read_text() if (out / 'py-compile.log').exists() else ''
    probe = json.loads((out / 'logprobe.json').read_text()) if (out / 'logprobe.json').exists() else {}
    views = {}
    for side in ('fixed', 'parent'):
        path = out / ('parity-%s.json' % side)
        views[side] = json.loads(path.read_text()) if path.exists() else None
    fixed_rows, parent_rows = views['fixed'], views['parent']
    parity = dict(suite=PARITY, parent=PARENT, identical=fixed_rows is not None and fixed_rows == parent_rows,
                  tests=len(fixed_rows or {}),
                  not_ok_both=sorted(k for k, v in (fixed_rows or {}).items() if v != 'ok'
                                     and (parent_rows or {}).get(k) == v),
                  differing=sorted(k for k in set(fixed_rows or {}) | set(parent_rows or {})
                                   if (fixed_rows or {}).get(k) != (parent_rows or {}).get(k)))
    result = dict(schema='native43-le-fix-selfrun/1', container_rc=rc, image=IMAGE, network=network,
                  gpu_device_requests=len(devices), box_action=False,
                  py_compile=dict(ok='PY_COMPILE %d OK' % len(COMPILED) in compile_log
                                  and compile_log.rstrip().endswith('rc=0'), files=len(COMPILED)),
                  suites=dict(names=list(SUITES), ran=ran[-1] if ran else None,
                              ok=text.rstrip().endswith('suites rc=0'), skipped=text.count('... skipped'),
                              log_sha256=sha(out / 'suites.log') if text else None),
                  parity=parity,
                  logprobe=dict(verdict=probe.get('verdict'), checks=probe.get('checks'),
                                sha256=sha(out / 'logprobe.json') if probe else None),
                  packages=pins, overlay=str(overlay),
                  reference=dict(root=str((tree / 'reference/vllm-src').resolve()),
                                 multiproc_executor_sha256=sha(tree / 'reference/vllm-src/v1/executor/multiproc_executor.py')),
                  tree_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tree, text=True).strip(),
                  tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain', '--', 'harness', 'patches',
                                                           'tools'], cwd=tree, text=True).strip()),
                  wall_s=round(time.monotonic() - started, 3), command=command)
    result['verdict'] = 'PASS' if (rc == 0 and result['py_compile']['ok'] and result['suites']['ok']
                                   and probe.get('verdict') == 'PASS' and parity['identical']
                                   and network == 'none' and not devices
                                   and not result['tree_dirty']) else 'FAIL'
    (out / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('verdict', 'container_rc', 'py_compile', 'suites', 'parity', 'logprobe',
                                              'tree_head', 'tree_dirty', 'wall_s')}, sort_keys=True))
    print('result_sha256', sha(out / 'result.json'))
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--overlay', type=Path, required=True, help='the l-e overlay (tmp/native43-lifecycle-e/final)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', required=True)
    a = p.parse_args()
    raise SystemExit(run(a.tree, a.overlay, a.out, a.name))
