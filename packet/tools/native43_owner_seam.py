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
     sha256 of the synthetic key they saw and which per-arm key variables were visible.  Each stub arm also
     starts a background reader and waits for it, as the real draft waits for its foreground boundary client
     (checked statically on the real drafts); at start it records any process of the other arm still alive.
     phase() children run from a fixture harness F (symlinks to the bundle, runs/ = the wrapper copy + the
     stubs) with HARNESS_DIR=F.
  L5 the stub arms write the queue-grade.json/cleanup.json fields the wrapper's cleanup check reads; the
     pair grade then runs for real and must refuse (no client records exist).
Owner decision (1 Nov 2026): one fresh key per pair queue, shared by both arms; --staging declares both request
directories from the one --auth-env-file key (CLI unchanged).  Keys here are synthetic (secrets.token_hex), never
box keys; only their sha256 appear in the record.
"""
import argparse
import hashlib
import importlib.util
import json
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
python3 - "$R/seam-declare.json" <<'SEAM_PY'
import hashlib, json, os, sys
key = os.environ.get('VLLM_API_KEY', '')
json.dump(dict(qname=os.environ['QNAME'], key_sha256=hashlib.sha256(key.encode()).hexdigest(),
               other_key_vars=sorted(n for n in os.environ if n.startswith('VLLM_API_KEY_'))), open(sys.argv[1], 'x'))
SEAM_PY
'''

STUB_DRAFT = r'''#!/usr/bin/env bash
# FIXTURE LEAF L4/L5 (native43 owner seam): stands in for the box arm; runs the real arm's auth check only.
set -uo pipefail
H=${HARNESS_DIR:?}; R=${REQUESTS:?}; Q=${1:?}
[[ ! -e $Q && ! -L $Q ]] || exit 3
mkdir -m 0700 "$Q" || exit 3
export NATIVE43_SEAM_ARM=${Q##*-}
python3 - "$Q" <<'SEAM_PY'
import json, os, sys, time
start, me, alive = time.monotonic_ns(), os.environ['NATIVE43_SEAM_ARM'], []
for pid in filter(str.isdigit, os.listdir('/proc')):
    try:
        env = open('/proc/%s/environ' % pid, 'rb').read().split(b'\0')
    except OSError:
        continue
    if any(e.startswith(b'NATIVE43_SEAM_ARM=') and e.split(b'=', 1)[1].decode() != me for e in env):
        alive.append(int(pid))
json.dump(dict(start_ns=start, other_arm_alive=alive), open(sys.argv[1] + '/seam-start.json', 'x'))
SEAM_PY
python3 -c 'import time; time.sleep(0.5)' &
reader=$!
rc=0
python3 "$H/e3_private_auth.py" check "$R/private/vllm-api.env" > "$Q/auth-check.log" 2>&1 || rc=$?
wait "$reader"
python3 - "$Q" "$R" "$rc" <<'SEAM_PY'
import hashlib, json, os, sys, time
end = time.monotonic_ns()
q, r, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
key = os.environ.get('VLLM_API_KEY', '')
json.dump(dict(output=q, requests=r, auth_check_rc=rc, key_sha256=hashlib.sha256(key.encode()).hexdigest(),
               other_key_vars=sorted(n for n in os.environ if n.startswith('VLLM_API_KEY_')),
               bounded=os.environ.get('NATIVE43_PAIR_BOUNDED'), pair_id=os.environ.get('NATIVE43_PAIR_ID'),
               end_ns=end, **json.load(open(q + '/seam-start.json'))),
          open(q + '/seam-arm.json', 'x'))
json.dump(dict(engine_crashed=False, engine_state='inactive', fixture='native43-owner-seam'), open(q + '/queue-grade.json', 'x'))
json.dump(dict(engine_stopped=True, restore_withheld=False, fixture='native43-owner-seam',
               **{k + s: 'fixture-' + k for k in ('launcher', 'defaults') for s in ('_found_sha256', '_restored_sha256')}),
          open(q + '/cleanup.json', 'x'))
SEAM_PY
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


def scenario(name, work_root, key, records, base, before_staging=None, before_arm=None):
    work = work_root / name
    work.mkdir()
    fixture = fixture_harness(Path('/source/harness'), work / 'F')
    owner = load_owner('/source/owner.py')
    calls = install_leaves(owner, work, base, fixture)
    auth = private_env(work / 'auth', key)
    out = dict(name=name, calls=calls)
    argv = ['--staging', '--auth-env-file', str(auth),
            '--selfrun-record', records['bundle'], '--engine-selfrun-record', records['engine']]
    if before_staging:
        before_staging(owner)
    try:
        out['staging_rc'] = owner.main(argv)
    except Exception as exc:
        out['staging_refused'] = str(exc) if isinstance(exc, owner.Refused) else type(exc).__name__
        return out, owner
    stage = json.loads((owner.S0 / (owner.QNAME + '-staging.json')).read_text())
    out['staging_receipt'] = {k: stage.get(k) for k in ('verdict', 'declaration_sha256', 'engine_untouched')}
    out['declares'] = {p: json.loads((owner.S0 / (owner.QNAME + '-' + p + '-requests') / 'seam-declare.json').read_text())
                       for p in POLICIES}
    if before_arm:
        before_arm(owner)
    try:
        out['arm_rc'] = owner.main(['--arm'])
    except Exception as exc:
        out['arm_refused'] = str(exc) if isinstance(exc, owner.Refused) else type(exc).__name__
        return out, owner
    prefix = str(owner.S0 / owner.QNAME)
    out['arms'] = {p: (json.loads((Path(prefix + '-' + p) / 'seam-arm.json').read_text())
                       if (Path(prefix + '-' + p) / 'seam-arm.json').is_file() else None) for p in POLICIES}
    grade = Path(prefix + '-pair-grade.json')
    out['pair_grade'] = json.loads(grade.read_text()).get('verdict') if grade.is_file() else 'ABSENT'
    out['pair_log_tail'] = (owner.S0 / (owner.QNAME + '.log')).read_text()[-1500:]
    return out, owner


def client_foreground(draft):
    """The real arm runs its boundary client in the foreground: the command ends without '&' and rc=$? follows."""
    lines = Path(draft).read_text().splitlines()
    i = next(j for j, line in enumerate(lines) if 'native43_boundary_client.py' in line)
    while lines[i].rstrip().endswith('\\'):
        i += 1
    return not lines[i].rstrip().endswith('&') and lines[i + 1].startswith('rc=$?')


def existing_dense(owner):
    (owner.S0 / (owner.QNAME + '-dense-requests')).mkdir(mode=0o700)


def other_dense_key(owner):
    path = owner.S0 / (owner.QNAME + '-dense-requests') / 'private/vllm-api.env'
    path.rename(path.with_name('vllm-api.env.seam-original'))
    from e3_private_auth import prepare
    prepare(path, {'VLLM_API_KEY': secrets.token_hex(32)})


def missing_dense(owner):
    requests = owner.S0 / (owner.QNAME + '-dense-requests')
    requests.rename(requests.with_name(requests.name + '.seam-moved'))


NEGATIVES = (('pre-existing-request-dir', dict(before_staging=existing_dense), ('staging_refused', 'fresh-output-path-required')),
             ('arm-key-mismatch', dict(before_arm=other_dense_key), ('arm_refused', 'one-queue-key-required')),
             ('arm-request-dir-missing', dict(before_arm=missing_dense), ('arm_refused', 'FileNotFoundError')))


def inner(records):
    work = Path('/out/work')
    work.mkdir()
    base = Path('/base')
    key = secrets.token_hex(32)
    main, owner = scenario('one-queue-key', work, key, records, base)
    prefix = str(owner.S0 / owner.QNAME)
    problems = []
    declares, arms = main.get('declares') or {}, main.get('arms') or {}
    if main.get('staging_rc') != 0:
        problems.append('staging-not-0:%r' % main.get('staging_rc', main.get('staging_refused')))
    for p in POLICIES:
        d, a = declares.get(p) or {}, arms.get(p) or {}
        if d.get('qname') != owner.QNAME + '-' + p or d.get('key_sha256') != key_sha(key) or d.get('other_key_vars'):
            problems.append('declare-' + p)
        if (a.get('auth_check_rc') != 0 or a.get('key_sha256') != key_sha(key) or a.get('other_key_vars')
                or a.get('output') != prefix + '-' + p or a.get('requests') != prefix + '-' + p + '-requests'
                or a.get('bounded') != '1' or a.get('other_arm_alive')):
            problems.append('arm-' + p)
    if not (arms.get('current') and arms.get('dense') and arms['current']['end_ns'] <= arms['dense']['start_ns']):
        problems.append('arm-2-not-after-arm-1-joined')
    if len({(a or {}).get('pair_id') for a in arms.values()}) != 1:
        problems.append('pair-id-differs')
    pair_calls = [c for c in main['calls'] if c['script'] == WRAPPER]
    if [c['args'] for c in pair_calls] != [[prefix]] or [c['key_vars'] for c in pair_calls] != [['VLLM_API_KEY']]:
        problems.append('pair-wrapper-argument-or-key')
    if main.get('pair_grade') == 'PASS':
        problems.append('pair-grade-passed-on-fixture-arms')
    foreground = {p: client_foreground('/source/harness/runs/q18a_native43_%s.draft.sh' % p) for p in POLICIES}
    if not all(foreground.values()):
        problems.append('real-draft-client-not-foreground')
    negatives = {}
    for name, hooks, (field, want) in NEGATIVES:
        row, _ = scenario(name, work, key, records, base, **hooks)
        negatives[name] = {k: row.get(k) for k in ('staging_rc', 'staging_refused', 'arm_rc', 'arm_refused')}
        if row.get(field) != want or [c for c in row['calls'] if c['script'] == WRAPPER]:
            problems.append('negative-not-refused:' + name)
    record = dict(schema='native43-owner-seam/1', verdict='FAIL' if problems else 'PASS', problems=problems,
                  scenario=main, negatives=negatives, real_draft_client_foreground=foreground,
                  key_sha256=key_sha(key), runner_sha256=sha('/source/owner.py'),
                  wrapper_sha256=sha('/source/harness/runs/' + WRAPPER),
                  owner_decision='one fresh key per pair queue, shared by both arms (1 Nov 2026)',
                  fixture_leaves=['L1 box launcher/defaults pins answered for exactly those two paths',
                                  'L2 BASE/OVERLAY/RAW/BUILD_LOGS/S0 in a fresh work root (hub base copy)',
                                  'L3 systemctl/nvidia-smi answers: stopped unit, four idle GPUs',
                                  'L4 per-arm declares/drafts: stubs running only the real auth prepare/check, '
                                  'each with a joined background reader',
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
    ok = rc == 0 and seam.get('verdict') == 'PASS' and not devices and not host.get('Devices') and host.get('NetworkMode') == 'none'
    result = dict(seam, schema='native43-owner-seam/1', verdict='PASS' if ok else 'FAIL',
                  classification=None if ok else 'HARNESS', rc=rc,
                  bundle_manifest=sha(bundle / 'SHA256SUMS'), runner_sha256=sha(runner), image=IMAGE,
                  network=host.get('NetworkMode'), gpu_device_requests=len(devices), box_action=False, container=name,
                  wall_s=round(time.monotonic() - started, 3), command=command)
    subprocess.run(['sudo', '-n', 'docker', 'rm', name], check=True, capture_output=True)
    (out / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ('verdict', 'problems', 'rc', 'wall_s')}))
    return 0 if ok else 1


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
