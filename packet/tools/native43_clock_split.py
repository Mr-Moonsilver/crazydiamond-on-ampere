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

This tool rebuilds that split physically.
- Engine container: --network none, its own time namespace, the real boot id.  Its
  main process is an engine-side status probe on stdin/stdout, wired to two FIFOs on a
  shared directory.  Engine sides:
    native43-clock-probe   the bundle's harness/native43_clock_probe.py --serve, unchanged
                           (real driver _status_record over the real four-bridge stack)
    driver-clock-record    this tool's `serve`: the overlay's real
                           NativeSegmentDriver._clock_record and the real
                           native_overlap_metrics.prove_clock_domain against the engine's
                           own identity (as native_overlap_control does in the engine)
- Client container: no network at all, its own time namespace, and an LXC-style boot
  id: a fresh random UUID bind-mounted over /proc/sys/kernel/random/boot_id before
  every capability is dropped (setpriv --bounding-set -all --no-new-privs).  It sends
  `status` lines through the FIFOs and brackets each reply with its own
  time.monotonic_ns().
- Positive control: the same client checks by docker exec inside the engine
  container, against a second engine-side process spawned there (one shared view);
  every check must be proved.

Client checks (each runs the bundle's real code; the spec declares the outcome):
  identity_equality    the native40 unarmed-box rule: status clock boot_id and
                       time_namespace equal the client's clock_identity()
  prove_clock_domain   native_overlap_metrics.prove_clock_domain(status clock, client
                       clock_identity(), send, recv)
  native_window_clock  e3_native_performance.native_window over a record carrying the
                       engine-side proofs; outcome is whether it names client-engine-clock
  bracket              every engine sample lies inside its client request bracket
  module:<m>.<f>       a bundle module's bracket proof f(status, send, recv) and its
                       'proved' field (native43_boundary_clock.prove)
For the native43-clock-probe side the engine-side proof is prove_clock_domain over the
packet's own engine view (the identity the engine process read), computed in the client.
Verdict PASS: the split is present (boot ids and time namespaces differ, every timens
offset zero), every check has its declared outcome, and the control proves them all.
No GPU, no box action, no network.
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
SAMPLES = 8
CHECKS = ('identity_equality', 'prove_clock_domain', 'native_window_clock', 'bracket')
SIDES = {'native43-clock-probe': ['python3', '-B', '/source/harness/native43_clock_probe.py', '--serve'],
         'driver-clock-record': ['python3', '-B', '/n43/clock_split.py', 'serve']}
CLIENT_PATH = ['/source/harness/e3_builder', '/source/harness']


def offsets_zero(text):
    return (bool(text) and not text.startswith('unreadable:')
            and all(line.split()[1:] == ['0', '0'] for line in text.splitlines()))


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
    return dict(boot_id=read('/proc/sys/kernel/random/boot_id'), time_namespace=time_ns, timens_offsets=offsets,
                timens_offsets_zero=offsets_zero(offsets), pid=os.getpid())


# ------------------------------------------------------------------ engine side (driver-clock-record)
def serve():
    from vllm.v1.flash_e3 import engine_driver
    from vllm.v1.flash_e3.native_overlap_metrics import clock_identity, prove_clock_domain
    fake = type('ClockOnly', (), {'clock_basis': engine_driver.CLOCK_BASIS})()
    source = dict(engine_driver=engine_driver.__file__,
                  sha256=hashlib.sha256(Path(engine_driver.__file__).read_bytes()).hexdigest())
    for line in sys.stdin:
        if line.strip() != 'status':
            raise SystemExit('unknown probe command')
        t0 = time.monotonic_ns()
        now = time.monotonic_ns()
        clock = engine_driver.NativeSegmentDriver._clock_record(fake, now)
        t1 = time.monotonic_ns()
        print(json.dumps(dict(status=dict(observed_clock_ns=now, clock=clock), engine_view=view(),
                              engine_proof=prove_clock_domain(clock, clock_identity(), t0, t1), source=source),
                         sort_keys=True), flush=True)
    return 0


# ------------------------------------------------------------------ client side
def client(side, out, modules, fifo=None):
    sys.path[:0] = CLIENT_PATH
    from native_overlap_metrics import clock_identity, prove_clock_domain
    import e3_native_performance as performance
    child = None
    if fifo is not None:
        request = open(os.path.join(fifo, 'req.fifo'), 'w')
        reply = open(os.path.join(fifo, 'resp.fifo'), 'r')
    else:
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(CLIENT_PATH), PYTHONDONTWRITEBYTECODE='1')
        child = subprocess.Popen(SIDES[side], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
        request, reply = child.stdin, child.stdout
    local = clock_identity()
    rows = []
    for _ in range(SAMPLES):
        send = time.monotonic_ns()
        request.write('status\n')
        request.flush()
        line = reply.readline()
        recv = time.monotonic_ns()
        packet = json.loads(line)
        status = packet['status']
        clock = status.get('clock') or {}
        engine_view = packet['engine_view']
        engine_proof = packet.get('engine_proof')
        if engine_proof is None:
            engine_identity = dict(proved=True, boot_id=engine_view['boot_id'],
                                   time_namespace=engine_view['time_namespace'])
            engine_proof = prove_clock_domain(clock, engine_identity, send, recv)
        identity = (bool(local.get('boot_id')) and clock.get('boot_id') == local['boot_id']
                    and clock.get('time_namespace') == local.get('time_namespace'))
        domain = prove_clock_domain(clock, local, send, recv)
        record = dict(begin=dict(engine_epoch=1, measure_start_ns=send, snapshot={}),
                      end=dict(engine_epoch=1, measure_end_ns=recv, raw_journal_evidence=None,
                               clock=dict(begin=engine_proof, end=engine_proof)),
                      client_bracket=dict(start_ns=send - 1, end_ns=recv + 1))
        named = sorted(r for r in performance.native_window(record, 'fresh', 1)['reasons']
                       if r.startswith('client-engine-clock'))
        stamp = status.get('observed_clock_ns')
        row = dict(send_ns=send, recv_ns=recv, engine_ns=stamp,
                   identity_equality=dict(proved=identity),
                   prove_clock_domain=dict(proved=domain['proved'], reasons=domain['reasons']),
                   native_window_clock=dict(proved=not named, reasons=named,
                                            engine_proof_proved=engine_proof.get('proved')),
                   bracket=dict(proved=type(stamp) is int and send <= stamp <= recv),
                   engine_clock=clock, engine_view=engine_view,
                   engine_source=packet.get('source') or packet.get('fixture'))
        for name in modules:
            module, function = name.rsplit('.', 1)
            proof = getattr(__import__(module), function)(status, send, recv)
            row['module:' + name] = dict(proved=proof.get('proved') is True, basis=proof.get('basis'),
                                         identity_equality_required=proof.get('identity_equality_required'))
        rows.append(row)
    request.close()
    if child is not None:
        child.wait(timeout=60)
    keys = list(CHECKS) + ['module:' + m for m in modules]
    result = dict(side=side, client_view=view(), client_clock_identity=local, samples=rows,
                  outcomes={k: ('proved' if all(r[k]['proved'] for r in rows) else
                                'refused' if not any(r[k]['proved'] for r in rows) else 'mixed') for k in keys})
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


def run(bundle, tree, out, name, side, expect, modules):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from native37_robust_selfrun import overlay_mounts, source_mounts
    bundle, out = Path(bundle).resolve(), Path(out).resolve()
    assert side in SIDES and set(expect) == set(CHECKS) | {'module:' + m for m in modules}, (side, sorted(expect))
    tool = Path(__file__).resolve()
    out.mkdir()
    for d in ('engine', 'client', 'pipe'):
        (out / d).mkdir()
        (out / d).chmod(0o777)
    for fifo in ('req.fifo', 'resp.fifo'):
        os.mkfifo(out / 'pipe' / fifo)
        (out / 'pipe' / fifo).chmod(0o666)
    ct_boot = str(uuid.uuid4())
    (out / 'ct-boot-id').write_text(ct_boot + '\n')
    (out / 'ct-boot-id').chmod(0o444)
    common = ['--pull=never', '--runtime', 'runc', '--network', 'none', '--cpus', '2', '--memory', '3g',
              '--memory-swap', '3g', '--read-only', '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=256m', '--ulimit', 'core=0',
              '-e', 'NVIDIA_VISIBLE_DEVICES=none', '-e', 'CUDA_VISIBLE_DEVICES=', '-e', 'PYTHONDONTWRITEBYTECODE=1',
              '-e', 'PYTHONPATH=' + os.pathsep.join(CLIENT_PATH),
              '-v', '%s:/n43/clock_split.py:ro' % tool, '-v', '%s:/pipe' % (out / 'pipe')]
    engine_mounts = dict(overlay_mounts(bundle / 'native30_overlay'))
    engine_mounts.update(source_mounts(Path(tree).resolve(), bundle))
    engine_name, client_name = name + '-engine', name + '-client'
    started = time.monotonic()
    commands = dict(engine=['run', '-d', '--name', engine_name, *common,
                            *sum((['-v', '%s:%s:ro' % kv] for kv in engine_mounts.items()), []),
                            '-v', '%s:/out' % (out / 'engine'), '--entrypoint', 'sh', IMAGE, '-c',
                            'exec %s < /pipe/req.fifo > /pipe/resp.fifo 2> /out/engine-side.err' % ' '.join(SIDES[side])])
    docker(*commands['engine'], check=True, stdout=subprocess.DEVNULL)
    client_args = ['/n43/clock_split.py', 'client', '--side', side] + sum((['--module', m] for m in modules), [])
    # Positive control first: the FIFO-side engine process waits until the CT-view client opens the pipes.
    commands['control'] = ['exec', engine_name, 'python3', '-B', *client_args, '--out', '/out/control.json']
    with (out / 'control.log').open('x') as log:
        control_rc = docker(*commands['control'], stdout=log, stderr=subprocess.STDOUT, timeout=600).returncode
    inner = ('mount --bind /n43/boot_id /proc/sys/kernel/random/boot_id && exec setpriv --bounding-set -all '
             '--inh-caps -all --no-new-privs python3 -B ' + ' '.join(client_args) + ' --fifo /pipe --out /out/client.json')
    commands['client'] = ['run', '--name', client_name, *common, '--cap-add', 'SYS_ADMIN',
                          '--security-opt', 'apparmor=unconfined', '-v', '%s:/n43/boot_id:ro' % (out / 'ct-boot-id'),
                          '-v', '%s:/source/harness:ro' % bundle, '-v', '%s:/out' % (out / 'client'),
                          '--entrypoint', 'sh', IMAGE, '-c', inner]
    with (out / 'client.log').open('x') as log:
        client_rc = docker(*commands['client'], stdout=log, stderr=subprocess.STDOUT, timeout=600).returncode
    docker('wait', engine_name, stdout=subprocess.DEVNULL, timeout=120)
    info = dict(engine=inspect(engine_name), client=inspect(client_name))
    with (out / 'engine.log').open('x') as log:
        docker('logs', engine_name, stdout=log, stderr=subprocess.STDOUT)
    docker('rm', engine_name, client_name, stdout=subprocess.DEVNULL)
    result = assemble(out, side, expect, client_rc, control_rc, info, commands, ct_boot)
    result['wall_s'] = round(time.monotonic() - started, 3)
    (out / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('verdict', 'reasons', 'outcomes', 'split')}, sort_keys=True))
    return 0 if result['verdict'] == 'PASS' else 1


def assemble(out, side, expect, client_rc, control_rc, info, commands, ct_boot):
    reasons = []
    path = out / 'client/client.json'
    client = json.loads(path.read_text()) if path.exists() else None
    if client is None:
        return dict(schema='native43-clock-split/1', side=side, verdict='FAIL', classification='HARNESS',
                    reasons=['client-record-missing:rc=%s' % client_rc], outcomes=None, expected=expect, split=None,
                    client_rc=client_rc, control_rc=control_rc, containers=info, commands=commands, box_action=False,
                    gpu_device_requests=sum(info[k]['device_requests'] for k in info))
    cv, ev = client['client_view'], client['samples'][0]['engine_view']
    engine_offsets = ev.get('timens_offsets', ev.get('offsets'))
    split = dict(client_boot_id=cv['boot_id'], engine_boot_id=ev['boot_id'], client_time_namespace=cv['time_namespace'],
                 engine_time_namespace=ev['time_namespace'], client_offsets=cv['timens_offsets'],
                 engine_offsets=engine_offsets, client_offsets_zero=offsets_zero(cv['timens_offsets']),
                 engine_offsets_zero=offsets_zero(engine_offsets), ct_boot_id_bound=cv['boot_id'] == ct_boot,
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
    cpath = out / 'engine/control.json'
    control = json.loads(cpath.read_text()) if cpath.exists() else None
    control_outcomes = None if control is None else control['outcomes']
    if control is None or control_rc != 0:
        reasons.append('control-missing-or-rc:%s' % control_rc)
    else:
        if (control['client_view']['boot_id'] != ev['boot_id']
                or control['client_view']['time_namespace'] != ev['time_namespace']):
            reasons.append('control-not-engine-view')
        for key, got in sorted(control_outcomes.items()):
            if got != 'proved':
                reasons.append('control:%s=%s(expected proved)' % (key, got))
    if info['engine']['network'] != 'none' or info['client']['network'] != 'none':
        reasons.append('network-not-none')
    if any(info[k]['device_requests'] or info[k]['devices'] for k in info):
        reasons.append('device-request')
    return dict(schema='native43-clock-split/1', side=side, verdict='FAIL' if reasons else 'PASS',
                classification='HARNESS' if reasons else None, reasons=reasons, outcomes=client['outcomes'],
                expected=expect, control_outcomes=control_outcomes, split=split, client_rc=client_rc,
                control_rc=control_rc, containers=info, commands=commands,
                client_record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                control_record_sha256=None if control is None else hashlib.sha256(cpath.read_bytes()).hexdigest(),
                image=IMAGE, box_action=False, gpu_device_requests=sum(info[k]['device_requests'] for k in info))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='mode', required=True)
    sub.add_parser('serve')
    c = sub.add_parser('client')
    c.add_argument('--side', choices=sorted(SIDES), required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--module', action='append', default=[])
    c.add_argument('--fifo')
    r = sub.add_parser('run')
    r.add_argument('--bundle', required=True)
    r.add_argument('--tree', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--name', required=True)
    r.add_argument('--side', choices=sorted(SIDES), required=True)
    r.add_argument('--expect', required=True, help='JSON {check: proved|refused}')
    r.add_argument('--module', action='append', default=[])
    a = p.parse_args()
    if a.mode == 'serve':
        raise SystemExit(serve())
    if a.mode == 'client':
        raise SystemExit(client(a.side, a.out, a.module, a.fifo))
    raise SystemExit(run(a.bundle, a.tree, a.out, a.name, a.side, json.loads(a.expect), a.module))
