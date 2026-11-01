#!/usr/bin/env python3
"""CT-vs-container clock split, reproduced on the hub with two CPU containers.

On the box the queue's clients run as plain python3 in CT <ct-id>, while the engine
runs in a docker container.  The two sides read different clock identities:
- the CT's /proc/sys/kernel/random/boot_id is LXC's per-container value and its
  time namespace is the CT's;
- the engine container reads the real kernel boot id and has its own time namespace.
Every timens offset is zero, so CLOCK_MONOTONIC is one clock on both sides.
native40 went HARNESS-INC on the box because a client compared those identities for
equality (results/box/q18a-native40-unarmed-hold-run-1); its hub self-run had one
shared view and could not see it.

This tool rebuilds that split physically:
- engine container: --network none, its own time namespace, the real boot id.  It
  serves one loopback status route whose clock record comes from the overlay's real
  NativeSegmentDriver._clock_record (overlay mounted as the engine self-run mounts it) and whose engine-side proof comes from the real
  native_overlap_metrics.prove_clock_domain against the engine's own identity
  (as native_overlap_control does inside the engine process).
- client container: joins the engine container's network namespace (loopback only),
  has its own time namespace, and gets an LXC-style boot id: a fresh random UUID
  bind-mounted over /proc/sys/kernel/random/boot_id before every capability is
  dropped (setpriv --bounding-set -all --no-new-privs).  It runs the bundle's own
  client clock checks against the status route, with the request bracket.
- positive control: the same client checks run by docker exec inside the engine
  container (one shared view), where the identity checks must pass.

Checks (each names the real code it runs; the spec states the expected outcome):
  identity_equality    the native40 unarmed-box rule (status clock boot_id and
                       time_namespace equal the client's clock_identity()), verbatim
  prove_clock_domain   native_overlap_metrics.prove_clock_domain(status clock,
                       client clock_identity(), send, recv)
  native_window_clock  e3_native_performance.native_window over a record carrying the
                       engine-side proofs; outcome is whether it names client-engine-clock
  bracket              every engine sample lies inside its client request bracket
  module:<m>.<f>       a bundle module's bracket proof, called as f(snapshot, send, recv),
                       outcome its 'proved' field (native43_boundary_clock.prove)
Verdict PASS: the split is present (boot ids and time namespaces differ, every
timens offset zero), every check has its declared outcome, and the control has
identity checks proved.  No GPU, no box action, no network beyond loopback.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

IMAGE = 'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
PORT = 18043
SAMPLES = 8
CHECKS = ('identity_equality', 'prove_clock_domain', 'native_window_clock', 'bracket')


def view():
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError as exc:
            return 'unreadable:' + type(exc).__name__
    try:
        time_ns = os.readlink('/proc/self/ns/time')
    except OSError as exc:
        time_ns = 'unreadable:' + type(exc).__name__
    offsets = read('/proc/self/timens_offsets')
    zero = all(line.split()[1:] == ['0', '0'] for line in offsets.splitlines()) if ':' not in offsets else False
    return dict(boot_id=read('/proc/sys/kernel/random/boot_id'), time_namespace=time_ns, timens_offsets=offsets,
                timens_offsets_zero=zero, pid=os.getpid())


# ------------------------------------------------------------------ engine side
def serve(port, out):
    # The overlay e3 package mounted where the engine imports it (native37_robust_selfrun.overlay_mounts).
    from vllm.v1.flash_e3 import engine_driver
    from vllm.v1.flash_e3.native_overlap_metrics import clock_identity, prove_clock_domain
    from http.server import BaseHTTPRequestHandler, HTTPServer
    fake = type('ClockOnly', (), {'clock_basis': engine_driver.CLOCK_BASIS})()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == '/shutdown':
                self.send_response(200)
                self.end_headers()
                server.stop = True
                return
            t0 = time.monotonic_ns()
            now = time.monotonic_ns()
            clock = engine_driver.NativeSegmentDriver._clock_record(fake, now)
            t1 = time.monotonic_ns()
            body = json.dumps(dict(observed_clock_ns=now, clock=clock, engine_view=view(),
                                   engine_proof=prove_clock_domain(clock, clock_identity(), t0, t1),
                                   source=dict(engine_driver=engine_driver.__file__,
                                               sha256=hashlib.sha256(Path(engine_driver.__file__).read_bytes()).hexdigest())),
                              sort_keys=True).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(('127.0.0.1', port), Handler)
    server.stop = False
    server.timeout = 1.0
    (Path(out) / 'ready').write_text(json.dumps(view(), sort_keys=True) + '\n')
    deadline = time.monotonic() + 300
    while not server.stop and time.monotonic() < deadline:
        server.handle_request()
    return 0


# ------------------------------------------------------------------ client side
def fetch(port, path='/status'):
    import urllib.request
    send = time.monotonic_ns()
    with urllib.request.urlopen('http://127.0.0.1:%d%s' % (port, path), timeout=10) as r:
        body = r.read()
    recv = time.monotonic_ns()
    return json.loads(body) if body else None, send, recv


def client(port, out, modules):
    sys.path[:0] = ['/source/harness/e3_builder', '/source/harness']
    from native_overlap_metrics import clock_identity, prove_clock_domain
    import e3_native_performance as performance
    local = clock_identity()
    rows = []
    for _ in range(SAMPLES):
        status, send, recv = fetch(port)
        clock = status['clock']
        identity = dict(proved=bool(local.get('boot_id')) and clock.get('boot_id') == local['boot_id']
                        and clock.get('time_namespace') == local.get('time_namespace'))
        domain = prove_clock_domain(clock, local, send, recv)
        record = dict(begin=dict(engine_epoch=1, measure_start_ns=send, snapshot={}),
                      end=dict(engine_epoch=1, measure_end_ns=recv, raw_journal_evidence=None,
                               clock=dict(begin=status['engine_proof'], end=status['engine_proof'])),
                      client_bracket=dict(start_ns=send - 1, end_ns=recv + 1))
        window = performance.native_window(record, 'fresh', 1)
        named = sorted(r for r in window['reasons'] if r.startswith('client-engine-clock'))
        row = dict(send_ns=send, recv_ns=recv, engine_ns=status['observed_clock_ns'],
                   identity_equality=dict(proved=identity['proved']),
                   prove_clock_domain=dict(proved=domain['proved'], reasons=domain['reasons']),
                   native_window_clock=dict(proved=not named, reasons=named),
                   bracket=dict(proved=send <= status['observed_clock_ns'] <= recv),
                   engine_clock=clock, engine_view=status['engine_view'], engine_source=status['source'])
        for name in modules:
            module, function = name.rsplit('.', 1)
            proof = getattr(__import__(module), function)(
                dict(observed_clock_ns=status['observed_clock_ns'], clock=clock), send, recv)
            row['module:' + name] = dict(proved=proof.get('proved') is True,
                                         basis=proof.get('basis'), identity_equality_required=proof.get('identity_equality_required'))
        rows.append(row)
    result = dict(client_view=view(), client_clock_identity=local, samples=rows,
                  outcomes={k: ('proved' if all(r[k]['proved'] for r in rows) else
                                'refused' if not any(r[k]['proved'] for r in rows) else 'mixed')
                            for k in list(CHECKS) + ['module:' + m for m in modules]})
    Path(out).write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    return 0


# ------------------------------------------------------------------ hub orchestration
def docker(*args, **kw):
    return subprocess.run(['sudo', '-n', 'docker', *args], **kw)


def inspect(name):
    row = json.loads(docker('inspect', name, capture_output=True, text=True, check=True).stdout)[0]
    host = row['HostConfig']
    return dict(network=host.get('NetworkMode'), device_requests=len(host.get('DeviceRequests') or []),
                devices=len(host.get('Devices') or []), cap_add=host.get('CapAdd'), read_only=host.get('ReadonlyRootfs'),
                exit_code=row['State'].get('ExitCode'), image=row.get('Image'))


def run(bundle, out, name, expect, modules):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from native37_robust_selfrun import overlay_mounts
    bundle, out = Path(bundle).resolve(), Path(out).resolve()
    overlay = bundle / 'native30_overlay'
    tool = Path(__file__).resolve()
    assert set(expect) == set(CHECKS) | {'module:' + m for m in modules}, sorted(expect)
    out.mkdir()
    for d in ('engine', 'client', 'control'):
        (out / d).mkdir()
        (out / d).chmod(0o777)
    ct_boot = str(uuid.uuid4())
    (out / 'ct-boot-id').write_text(ct_boot + '\n')
    (out / 'ct-boot-id').chmod(0o444)
    common = ['--pull=never', '--runtime', 'runc', '--cpus', '2', '--memory', '3g', '--memory-swap', '3g', '--read-only',
              '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=256m', '--ulimit', 'core=0',
              '-e', 'NVIDIA_VISIBLE_DEVICES=none', '-e', 'CUDA_VISIBLE_DEVICES=', '-e', 'PYTHONDONTWRITEBYTECODE=1',
              '-v', '%s:/n43/clock_split.py:ro' % tool]
    engine_name, client_name = name + '-engine', name + '-client'
    started = time.monotonic()
    commands = {}
    commands['engine'] = ['run', '-d', '--name', engine_name, '--network', 'none', *common,
                          *sum((['-v', '%s:%s:ro' % kv] for kv in overlay_mounts(overlay).items()), []),
                          '-v', '%s:/source/harness:ro' % bundle,
                          '-v', '%s:/out' % (out / 'engine'),
                          '--entrypoint', 'python3', IMAGE, '-B', '/n43/clock_split.py', 'serve', '--port', str(PORT),
                          '--out', '/out']
    docker(*commands['engine'], check=True, stdout=subprocess.DEVNULL)
    deadline = time.monotonic() + 180
    while not (out / 'engine/ready').exists() and time.monotonic() < deadline:
        time.sleep(0.5)
    assert (out / 'engine/ready').exists(), 'engine-side status route did not start'
    client_args = ['/n43/clock_split.py', 'client', '--port', str(PORT)] + sum((['--module', m] for m in modules), [])
    inner = ('mount --bind /n43/boot_id /proc/sys/kernel/random/boot_id && exec setpriv --bounding-set -all '
             '--inh-caps -all --no-new-privs python3 -B ' + ' '.join(client_args) + ' --out /out/client.json')
    commands['client'] = ['run', '--name', client_name, '--network', 'container:' + engine_name, *common,
                          '--cap-add', 'SYS_ADMIN', '--security-opt', 'apparmor=unconfined',
                          '-v', '%s:/n43/boot_id:ro' % (out / 'ct-boot-id'), '-v', '%s:/source/harness:ro' % bundle,
                          '-v', '%s:/out' % (out / 'client'), '--entrypoint', 'sh', IMAGE, '-c', inner]
    with (out / 'client.log').open('x') as log:
        client_rc = docker(*commands['client'], stdout=log, stderr=subprocess.STDOUT).returncode
    # Positive control: the same checks inside the engine container (one shared view).
    commands['control'] = ['exec', engine_name, 'python3', '-B', *client_args, '--out', '/out/control.json']
    with (out / 'control.log').open('x') as log:
        control_rc = docker(*commands['control'], stdout=log, stderr=subprocess.STDOUT).returncode
    shutdown = docker('exec', engine_name, 'python3', '-c',
                      'import urllib.request;urllib.request.urlopen("http://127.0.0.1:%d/shutdown",timeout=10)' % PORT,
                      capture_output=True, text=True)
    docker('wait', engine_name, stdout=subprocess.DEVNULL)
    info = dict(engine=inspect(engine_name), client=inspect(client_name))
    with (out / 'engine.log').open('x') as log:
        docker('logs', engine_name, stdout=log, stderr=subprocess.STDOUT)
    docker('rm', engine_name, client_name, stdout=subprocess.DEVNULL)
    result = assemble(out, expect, modules, client_rc, control_rc, info, commands, shutdown.returncode, ct_boot)
    result['wall_s'] = round(time.monotonic() - started, 3)
    (out / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('verdict', 'reasons', 'outcomes', 'split')}, sort_keys=True))
    return 0 if result['verdict'] == 'PASS' else 1


def assemble(out, expect, modules, client_rc, control_rc, info, commands, shutdown_rc, ct_boot):
    reasons = []
    client = json.loads((out / 'client/client.json').read_text()) if (out / 'client/client.json').exists() else None
    if client is None:
        return dict(schema='native43-clock-split/1', verdict='FAIL', classification='HARNESS',
                    reasons=['client-record-missing'], outcomes=None, split=None, client_rc=client_rc,
                    containers=info, commands=commands)
    cv, ev = client['client_view'], client['samples'][0]['engine_view']
    split = dict(client_boot_id=cv['boot_id'], engine_boot_id=ev['boot_id'], client_time_namespace=cv['time_namespace'],
                 engine_time_namespace=ev['time_namespace'], client_offsets_zero=cv['timens_offsets_zero'],
                 engine_offsets_zero=ev['timens_offsets_zero'], ct_boot_id_bound=cv['boot_id'] == ct_boot,
                 boot_ids_differ=cv['boot_id'] != ev['boot_id'],
                 time_namespaces_differ=cv['time_namespace'] != ev['time_namespace'])
    for key in ('ct_boot_id_bound', 'boot_ids_differ', 'time_namespaces_differ', 'client_offsets_zero', 'engine_offsets_zero'):
        if split[key] is not True:
            reasons.append('split-not-reproduced:' + key)
    for key, want in sorted(expect.items()):
        if client['outcomes'].get(key) != want:
            reasons.append('outcome:%s=%s(expected %s)' % (key, client['outcomes'].get(key), want))
    if client_rc != 0:
        reasons.append('client-rc:%s' % client_rc)
    control = json.loads((out / 'engine/control.json').read_text()) if (out / 'engine/control.json').exists() else None
    control_outcomes = None if control is None else control['outcomes']
    if control is None or control_rc != 0:
        reasons.append('control-missing-or-rc:%s' % control_rc)
    else:
        if control['client_view']['boot_id'] != ev['boot_id'] or control['client_view']['time_namespace'] != ev['time_namespace']:
            reasons.append('control-not-shared-view')
        for key, got in sorted(control_outcomes.items()):
            if got != 'proved':
                reasons.append('control:%s=%s(expected proved)' % (key, got))
    if info['engine']['network'] != 'none' or not str(info['client']['network']).startswith('container:'):
        reasons.append('network-not-isolated')
    if any(info[k]['device_requests'] or info[k]['devices'] for k in info):
        reasons.append('device-request')
    return dict(schema='native43-clock-split/1', verdict='FAIL' if reasons else 'PASS',
                classification='HARNESS' if reasons else None, reasons=reasons, outcomes=client['outcomes'],
                expected=expect, control_outcomes=control_outcomes, split=split, client_rc=client_rc, control_rc=control_rc, shutdown_rc=shutdown_rc,
                containers=info, commands=commands, client_record_sha256=hashlib.sha256(
                    (out / 'client/client.json').read_bytes()).hexdigest(), image=IMAGE, box_action=False,
                gpu_device_requests=sum(info[k]['device_requests'] for k in info))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='mode', required=True)
    s = sub.add_parser('serve')
    s.add_argument('--port', type=int, required=True)
    s.add_argument('--out', required=True)
    c = sub.add_parser('client')
    c.add_argument('--port', type=int, required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--module', action='append', default=[])
    r = sub.add_parser('run')
    r.add_argument('--bundle', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--name', required=True)
    r.add_argument('--expect', required=True, help='JSON {check: proved|refused}')
    r.add_argument('--module', action='append', default=[])
    a = p.parse_args()
    if a.mode == 'serve':
        raise SystemExit(serve(a.port, a.out))
    if a.mode == 'client':
        raise SystemExit(client(a.port, a.out, a.module))
    raise SystemExit(run(a.bundle, a.out, a.name, json.loads(a.expect), a.module))
