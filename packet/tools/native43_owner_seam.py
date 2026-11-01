#!/usr/bin/env python3
"""Native43 owner-chain seam self-run: the frozen owner runner's --staging and --arm paths, end to end,
with the bundle's real pair wrapper, real e3_private_auth and real overlay copy, in the pinned image
(CPU only, network none).  No box action.

What runs unchanged: the frozen runner file (main, preflight, cpu_build, verify_sources, environment,
check_slot, live_guards, stopped_unit, phase, read_private), the bundle's
runs/q18a_native43_dense_boundary.draft.sh (pair wrapper, sha-checked), e3_private_auth.py prepare/check,
native43_boundary_gate.window_ok and native43_dense_grade.py.

Fixture leaves (each replaces a box-only object; declared in the record):
  L1 preflight's two box pins (/data/vllm/flash-next-tp4/prod/run-flash-prod.sh and /etc/default/vllm-flash):
     owner.sha answers the runner's own frozen pins for exactly those two paths; every other sha is real.
  L2 BASE, OVERLAY/RAW/BUILD_LOGS and S0 point into a fresh work root instead of /data/vllm and
     /home/<service-user>/s0; BASE is the hub copy of the same base overlay (its SHA256SUMS pin is checked as is).
  L3 systemctl/nvidia-smi in live_guards(): owner.public_output answers a stopped unit and four idle GPUs.
  L4 the per-arm declares and drafts (box queues: docker identity, systemd, GPUs) are stubs that run only
     the real queues' auth steps: the declare's `e3_private_auth.py prepare` into $S0/$QNAME-requests/private
     and the draft's `e3_private_auth.py check` (q18a_native43_current.draft.sh line 508).  They record the
     sha256 of the synthetic key they saw and which per-arm key variables were visible.  phase() children
     run from a fixture harness F (symlinks to the bundle, runs/ = the wrapper copy + the stubs) with
     HARNESS_DIR=F.
  L5 the stub arms write the queue-grade.json/cleanup.json fields the wrapper's cleanup check reads; the
     pair grade then runs for real and must refuse (no client records exist).
Keys are synthetic (secrets.token_hex), never box keys; only their sha256 appear in the record.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time

IMAGE = 'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
WRAPPER = 'q18a_native43_dense_boundary.draft.sh'
POLICIES = ('current', 'dense')
BOX_PIN_PATHS = ('/data/vllm/flash-next-tp4/prod/run-flash-prod.sh', '/etc/default/vllm-flash')

STUB_DECLARE = r'''#!/usr/bin/env bash
# FIXTURE LEAF L4 (native43 owner seam): stands in for the box declare; only its auth step is real.
set -uo pipefail
H=${HARNESS_DIR:?}; S0=${S0:?}; QNAME=${QNAME:?}
R=$S0/$QNAME-requests
[[ ! -e $R && ! -L $R ]] || exit 3
mkdir -m 0700 "$R" && mkdir -m 0700 "$R/private" || exit 3
python3 "$H/e3_private_auth.py" prepare "$R/private/vllm-api.env" > "$R/auth-prepare.log" 2>&1 || exit 4
printf '{"fixture": "native43-owner-seam", "qname": "%s"}\n' "$QNAME" > "$R/startup.json"
python3 - "$R/seam-declare.json" <<'PY'
import hashlib, json, os, sys
key = os.environ.get('VLLM_API_KEY', '')
json.dump(dict(qname=os.environ['QNAME'], key_sha256=hashlib.sha256(key.encode()).hexdigest(),
               other_key_vars=sorted(n for n in os.environ if n.startswith('VLLM_API_KEY_'))), open(sys.argv[1], 'x'))
PY
'''

STUB_DRAFT = r'''#!/usr/bin/env bash
# FIXTURE LEAF L4/L5 (native43 owner seam): stands in for the box arm; runs the real arm's auth check only.
set -uo pipefail
H=${HARNESS_DIR:?}; R=${REQUESTS:?}; Q=${1:?}
[[ ! -e $Q && ! -L $Q ]] || exit 3
mkdir -m 0700 "$Q" || exit 3
rc=0
python3 "$H/e3_private_auth.py" check "$R/private/vllm-api.env" > "$Q/auth-check.log" 2>&1 || rc=$?
python3 - "$Q" "$R" "$rc" <<'PY'
import hashlib, json, os, sys
q, r, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
key = os.environ.get('VLLM_API_KEY', '')
json.dump(dict(output=q, requests=r, auth_check_rc=rc, key_sha256=hashlib.sha256(key.encode()).hexdigest(),
               other_key_vars=sorted(n for n in os.environ if n.startswith('VLLM_API_KEY_')),
               bounded=os.environ.get('NATIVE43_PAIR_BOUNDED'), pair_id=os.environ.get('NATIVE43_PAIR_ID')),
          open(q + '/seam-arm.json', 'x'))
json.dump(dict(engine_crashed=False, engine_state='inactive', fixture='native43-owner-seam'), open(q + '/queue-grade.json', 'x'))
json.dump(dict(engine_stopped=True, restore_withheld=False, fixture='native43-owner-seam',
               **{k + s: 'fixture-' + k for k in ('launcher', 'defaults') for s in ('_found_sha256', '_restored_sha256')}),
          open(q + '/cleanup.json', 'x'))
PY
exit $rc
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key_sha(key):
    return hashlib.sha256(key.encode()).hexdigest()


def load_owner(path):
    spec = importlib.util.spec_from_file_location('native43_frozen_owner', path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    return owner


def fixture_harness(bundle, root):
    """F: every bundle entry except runs/ as a symlink; runs/ = the wrapper's bundle bytes + the L4 stubs."""
    root.mkdir()
    for entry in bundle.iterdir():
        if entry.name != 'runs':
            (root / entry.name).symlink_to(entry)
    runs = root / 'runs'
    runs.mkdir()
    shutil.copyfile(bundle / 'runs' / WRAPPER, runs / WRAPPER)
    assert sha(runs / WRAPPER) == sha(bundle / 'runs' / WRAPPER)
    for policy in POLICIES:
        (runs / ('q18a_native43_%s_declare.sh' % policy)).write_text(STUB_DECLARE)
        (runs / ('q18a_native43_%s.draft.sh' % policy)).write_text(STUB_DRAFT)
    return root


def install_leaves(owner, work, base, fixture):
    text = Path(owner.__file__).read_text()
    pins = {p: re.search(r"sha\(Path\('" + re.escape(p) + r"'\)\)=='([0-9a-f]{64})'", text).group(1) for p in BOX_PIN_PATHS}
    real_sha = owner.sha
    owner.sha = lambda path: pins[str(path)] if str(path) in pins else real_sha(path)          # L1
    owner.H = Path('/source/harness')
    owner.S0 = work / 's0'
    owner.S0.mkdir(mode=0o700)
    prod = work / 'prod'
    prod.mkdir()
    owner.BASE = base                                                                       # L2
    owner.OVERLAY = prod / owner.OVERLAY.name
    owner.RAW = owner.OVERLAY.with_name(owner.OVERLAY.name + '-build')
    owner.BUILD_LOGS = owner.S0 / (owner.QNAME + '-build-logs')
    gpus = '\n'.join('%d, 0' % i for i in range(4))
    unit = 'ActiveState=inactive\nSubState=dead\nMainPID=0\nControlPID=0'
    owner.public_output = lambda args: unit if args[0] == 'systemctl' else gpus             # L3
    calls = []
    real_run = subprocess.run

    class Shim:                                                                             # L4: phase() children from F
        STDOUT = subprocess.STDOUT

        @staticmethod
        def run(cmd, env=None, **kw):
            assert cmd[0] == 'bash' and cmd[1].startswith(str(owner.H / 'runs') + '/'), cmd
            script = Path(cmd[1]).name
            env = dict(env, HARNESS_DIR=str(fixture))
            calls.append(dict(script=script, args=list(cmd[2:]), qname=env.get('QNAME'),
                              key_vars=sorted(n for n in env if n.startswith('VLLM_API_KEY')),
                              requests=env.get('REQUESTS'), harness_dir_from_runner=str(owner.H)))
            return real_run(['bash', str(fixture / 'runs' / script), *cmd[2:]], env=env, **kw)
    owner.subprocess = Shim
    return calls


def private_env(directory, key):
    directory.mkdir(mode=0o700)
    path = directory / 'vllm-api.env'
    sys.path[:0] = ['/source/harness']
    from e3_private_auth import prepare
    prepare(path, {'VLLM_API_KEY': key})
    return path


def scenario(name, work_root, keys, records, base):
    work = work_root / name
    work.mkdir()
    fixture = fixture_harness(Path('/source/harness'), work / 'F')
    owner = load_owner('/source/owner.py')
    calls = install_leaves(owner, work, base, fixture)
    current = private_env(work / 'auth-current', keys['current'])
    dense = private_env(work / 'auth-dense', keys['dense'])
    out = dict(name=name, calls=calls)
    argv = ['--staging', '--auth-env-file', str(current), '--dense-auth-env-file', str(dense),
            '--selfrun-record', records['bundle'], '--engine-selfrun-record', records['engine']]
    try:
        out['staging_rc'] = owner.main(argv)
    except owner.Refused as exc:
        out['staging_refused'] = str(exc)
        return out, owner
    stage = json.loads((owner.S0 / (owner.QNAME + '-staging.json')).read_text())
    out['staging_receipt'] = {k: stage.get(k) for k in ('verdict', 'declaration_sha256', 'engine_untouched')}
    out['declares'] = {p: json.loads((owner.S0 / (owner.QNAME + '-' + p + '-requests') / 'seam-declare.json').read_text())
                       for p in POLICIES}
    try:
        out['arm_rc'] = owner.main(['--arm'])
    except owner.Refused as exc:
        out['arm_refused'] = str(exc)
        return out, owner
    prefix = owner.S0 / owner.QNAME
    out['arms'] = {p: (json.loads((Path(str(prefix) + '-' + p) / 'seam-arm.json').read_text())
                       if (Path(str(prefix) + '-' + p) / 'seam-arm.json').is_file() else None) for p in POLICIES}
    grade = Path(str(prefix) + '-pair-grade.json')
    out['pair_grade'] = json.loads(grade.read_text()).get('verdict') if grade.is_file() else 'ABSENT'
    out['pair_log_tail'] = (owner.S0 / (owner.QNAME + '.log')).read_text()[-1500:]
    return out, owner


def inner(records):
    work = Path('/out/work')
    work.mkdir()
    base = Path('/base')
    keys = {p: secrets.token_hex(32) for p in POLICIES}
    main, owner = scenario('fresh-key-per-arm', work, keys, records, base)
    prefix = str(owner.S0 / owner.QNAME)
    problems = []
    declares, arms = main.get('declares') or {}, main.get('arms') or {}
    if main.get('staging_rc') != 0:
        problems.append('staging-not-0:%r' % main.get('staging_rc', main.get('staging_refused')))
    for p in POLICIES:
        d, a = declares.get(p) or {}, arms.get(p) or {}
        if d.get('qname') != owner.QNAME + '-' + p or d.get('key_sha256') != key_sha(keys[p]) or d.get('other_key_vars'):
            problems.append('declare-' + p)
        if (a.get('auth_check_rc') != 0 or a.get('key_sha256') != key_sha(keys[p]) or a.get('other_key_vars')
                or a.get('output') != prefix + '-' + p or a.get('requests') != prefix + '-' + p + '-requests'
                or a.get('bounded') != '1'):
            problems.append('arm-' + p)
    pair_calls = [c for c in main['calls'] if c['script'] == WRAPPER]
    if [c['args'] for c in pair_calls] != [[prefix]]:
        problems.append('pair-wrapper-argument')
    if 'VLLM_API_KEY' in sum((c['key_vars'] for c in pair_calls), []):
        problems.append('pair-wrapper-got-a-shared-key')
    if main.get('pair_grade') == 'PASS':
        problems.append('pair-grade-passed-on-fixture-arms')
    same = {p: keys['current'] for p in POLICIES}
    negative, _ = scenario('same-key-both-arms', work, same, records, base)
    if negative.get('staging_refused') != 'fresh-key-per-arm-required':
        problems.append('same-key-negative-not-refused')
    record = dict(schema='native43-owner-seam/1', verdict='FAIL' if problems else 'PASS', problems=problems,
                  scenario=main, negative=dict(name=negative['name'], staging_refused=negative.get('staging_refused')),
                  key_sha256={p: key_sha(k) for p, k in keys.items()}, runner_sha256=sha('/source/owner.py'),
                  wrapper_sha256=sha('/source/harness/runs/' + WRAPPER),
                  fixture_leaves=['L1 box launcher/defaults pins answered for exactly those two paths',
                                  'L2 BASE/OVERLAY/RAW/BUILD_LOGS/S0 in a fresh work root (hub base copy)',
                                  'L3 systemctl/nvidia-smi answers: stopped unit, four idle GPUs',
                                  'L4 per-arm declares/drafts: stubs running only the real auth prepare/check',
                                  'L5 stub arm queue-grade/cleanup fields; the real pair grade must refuse'])
    Path('/out/seam.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(verdict=record['verdict'], problems=problems)))
    return 0 if not problems else 1


def outer(bundle, runner, base, bundle_record, engine_record, out, name):
    bundle, runner, base, out = (Path(p).resolve() for p in (bundle, runner, base, out))
    out.mkdir()
    out.chmod(0o777)
    records = out / 'records'
    records.mkdir()
    shutil.copyfile(bundle_record, records / 'bundle.json')
    shutil.copyfile(engine_record, records / 'engine.json')
    command = ['sudo', '-n', 'docker', 'run', '--pull=never', '--name', name, '--runtime', 'runc', '--network', 'none',
               '--cpus', '4', '--memory', '4g', '--memory-swap', '4g', '--read-only',
               '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=512m', '--ulimit', 'core=0',
               '-e', 'NVIDIA_VISIBLE_DEVICES=none', '-e', 'CUDA_VISIBLE_DEVICES=', '-e', 'PYTHONDONTWRITEBYTECODE=1',
               '-v', f'{bundle}:/source/harness:ro', '-v', f'{bundle / "native35_tools"}:/source/tools:ro',
               '-v', f'{runner}:/source/owner.py:ro', '-v', f'{base}:/base:ro', '-v', f'{out}:/out',
               '--entrypoint', 'python3', IMAGE, '-B', '/source/tools/native43_owner_seam.py', '--inner',
               '--bundle-record', '/out/records/bundle.json', '--engine-record', '/out/records/engine.json']
    started = time.monotonic()
    with (out / 'container.log').open('x') as log:
        rc = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    inspect = json.loads(subprocess.check_output(['sudo', '-n', 'docker', 'inspect', name], text=True))[0]
    host = inspect['HostConfig']
    devices = host.get('DeviceRequests') or []
    seam = json.loads((out / 'seam.json').read_text()) if (out / 'seam.json').is_file() else {}
    result = dict(seam, schema='native43-owner-seam/1', verdict='PASS' if rc == 0 and seam.get('verdict') == 'PASS' else 'FAIL',
                  classification=None if rc == 0 and seam.get('verdict') == 'PASS' else 'HARNESS', rc=rc,
                  bundle_manifest=sha(bundle / 'SHA256SUMS'), runner_sha256=sha(runner), image=IMAGE,
                  network=host.get('NetworkMode'), gpu_device_requests=len(devices), box_action=False, container=name,
                  wall_s=round(time.monotonic() - started, 3), command=command)
    assert not devices and not host.get('Devices')
    subprocess.run(['sudo', '-n', 'docker', 'rm', name], check=True, capture_output=True)
    (out / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ('verdict', 'problems', 'rc', 'wall_s')}))
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inner', action='store_true')
    p.add_argument('--bundle-record', required=True)
    p.add_argument('--engine-record', required=True)
    p.add_argument('--bundle')
    p.add_argument('--runner')
    p.add_argument('--base')
    p.add_argument('--out')
    p.add_argument('--name')
    a = p.parse_args()
    if a.inner:
        raise SystemExit(inner(dict(bundle=a.bundle_record, engine=a.engine_record)))
    raise SystemExit(outer(a.bundle, a.runner, a.base, a.bundle_record, a.engine_record, a.out, a.name))
