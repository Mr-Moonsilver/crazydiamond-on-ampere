#!/usr/bin/env python3
"""Plain API prefill on unarmed split with a CT-side container clock proof.

Native40's unarmed flow (frozen: a box queue ran it): real 30 s hold ->
applied FULL -> completion. No native prefill_begin or admission action is
sent. Every native40 check is kept. Only the clock test is replaced. The
native40 module is not imported: its plain request and the reviewed FULL
restoration are copied here verbatim, so its identity comparison cannot be
reached from this client (native41_clock_scan refuses it in the closure).

Native40 compared the engine status clock (boot_id, time namespace) with the
client process's own identity view. On the box the client is plain python3 in the
LXC CT, with an lxcfs boot_id and the init time namespace. The engine is in a
docker container, with the kernel boot_id and its own time namespace. The test
refused although CLOCK_MONOTONIC is one clock (q18a-native40-unarmed-hold
run 1: client-engine-clock-unproved, HARNESS).

Native41 proves the domain from the CT side:
1. Container identity: init pid (docker inspect --format {{.State.Pid}}),
   readlink /proc/<pid>/ns/time, /proc/<pid>/timens_offsets, and the kernel
   boot_id inside the container (docker exec <cid> cat
   /proc/sys/kernel/random/boot_id).
2. Declared zero offsets: the container init's and the client's own
   /proc/self/timens_offsets parse to exactly monotonic 0 0 and boottime 0 0.
   A missing or unparsable file is unproved. The client's own boot_id and
   time namespace are never read.
   The docker calls are read-only: inspect of the init pid, and exec of cat on
   the boot_id file. Nothing else runs in the container.
3. The engine status clock names the container's boot_id and time namespace.
4. native_overlap_metrics.prove_clock_domain: the engine sample of one status
   call lies inside the client's time.monotonic_ns() send/receive bracket.
The proof runs once before the split entry and the plain request (so before
the no-reply-on-split comparison) and once after the completion. Either one
unproved is INCONCLUSIVE/HARNESS with the cause
client-engine-clock-unproved:<phase>:<reasons>. Receipts hold identities,
offsets and counts. They never hold env, request or auth payloads.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.parse

import urllib.request

from e3_native_qualification_driver import exception_details
from e3_native_runtime_client import NativeRuntimeClient
from e3_native39_trunk_checks import full_idle
from native_overlap_metrics import prove_clock_domain

SCHEMA = 'native41-unarmed-box/1'
DOCKER_TIMEOUT_S = 10.0
ZERO_OFFSETS = {'monotonic': [0, 0], 'boottime': [0, 0]}
CONTAINER_ID = re.compile(r'[0-9a-f]{12,64}')
CLOCK_PROOF_BASIS = (
    'CT-side container identity (docker inspect init pid; readlink /proc/<pid>/ns/time; '
    '/proc/<pid>/timens_offsets; docker exec cat /proc/sys/kernel/random/boot_id), zero '
    'monotonic/boottime offsets on the container init and on the client (/proc/self/timens_offsets), '
    'engine status clock naming that boot_id and time namespace, engine sample inside the client '
    'time.monotonic_ns() bracket of one status call (native_overlap_metrics.prove_clock_domain)')


def docker_cli(argv, timeout_s):
    """Run one docker CLI command. Stdout only; stderr and env are never recorded."""
    done = subprocess.run(['docker'] + list(argv), capture_output=True, text=True,
                          timeout=timeout_s, stdin=subprocess.DEVNULL)
    if done.returncode != 0:
        raise OSError('docker-rc-%d' % done.returncode)
    return done.stdout


def parse_timens_offsets(text):
    """Exactly one monotonic and one boottime line of integer secs/nsecs, else None."""
    if type(text) is not str:
        return None
    out = {}
    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        if len(words) != 3 or words[0] not in ZERO_OFFSETS or words[0] in out:
            return None
        try:
            out[words[0]] = [int(words[1]), int(words[2])]
        except ValueError:
            return None
    return out if set(out) == set(ZERO_OFFSETS) else None


def offsets_receipt(path, side):
    """Read one timens_offsets file. Missing, unparsable or nonzero adds a reason."""
    try:
        text = Path(path).read_text()
    except OSError as exc:
        return None, [side + '-timens-offsets-unreadable:' + type(exc).__name__]
    parsed = parse_timens_offsets(text)
    if parsed is None:
        return None, [side + '-timens-offsets-unparsable']
    if parsed != ZERO_OFFSETS:
        return parsed, [side + '-timens-offsets-nonzero']
    return parsed, []


def container_identity(container_id, *, docker=None, proc='/proc', timeout_s=DOCKER_TIMEOUT_S):
    """The engine container's clock identity, read from the CT side."""
    docker = docker or docker_cli
    record = dict(container_id=container_id, init_pid=None, time_namespace=None,
                  timens_offsets=None, boot_id=None, reasons=[])
    reasons = record['reasons']
    if type(container_id) is not str or not CONTAINER_ID.fullmatch(container_id):
        reasons.append('container-id-missing')
        return record
    try:
        text = docker(['inspect', '--format', '{{.State.Pid}}', container_id], timeout_s).strip()
        pid = int(text) if text.isdigit() else None
        if pid is None or pid <= 0:
            reasons.append('container-init-pid-unproved:not-positive-int')
        else:
            record['init_pid'] = pid
    except (OSError, subprocess.SubprocessError) as exc:
        reasons.append('container-init-pid-unproved:' + (str(exc) if str(exc).startswith('docker-rc-')
                                                         else type(exc).__name__))
    pid = record['init_pid']
    if pid is not None:
        try:
            namespace = os.readlink(os.path.join(proc, str(pid), 'ns/time'))
            if namespace:
                record['time_namespace'] = namespace
            else:
                reasons.append('container-time-namespace-empty')
        except OSError as exc:
            reasons.append('container-time-namespace-unreadable:' + type(exc).__name__)
        record['timens_offsets'], more = offsets_receipt(os.path.join(proc, str(pid), 'timens_offsets'),
                                                         'container')
        reasons.extend(more)
    try:
        boot_id = docker(['exec', container_id, 'cat', '/proc/sys/kernel/random/boot_id'], timeout_s).strip()
        if boot_id:
            record['boot_id'] = boot_id
        else:
            reasons.append('container-boot-id-empty')
    except (OSError, subprocess.SubprocessError) as exc:
        reasons.append('container-boot-id-unproved:' + (str(exc) if str(exc).startswith('docker-rc-')
                                                        else type(exc).__name__))
    return record


def client_identity(proc='/proc'):
    """The client's own offsets (checked). Its boot_id and time namespace view is never read."""
    offsets, reasons = offsets_receipt(os.path.join(proc, 'self/timens_offsets'), 'client')
    return dict(timens_offsets=offsets, reasons=reasons,
                view_basis='not read: an LXC CT shows an lxcfs boot_id and its own time namespace')


# Copied verbatim from native40's unarmed client (plain_request) and from
# e3_native39_w1 (STATUS_KEYS, status_view, restore_full), so neither module is
# imported by this client.
STATUS_KEYS = ('route', 'route_generation', 'paused', 'transition_pending', 'control_pending',
               'return_pending', 'scheduler_requests', 'observed_clock_ns', 'clock')


def status_view(status):
    return {k: status.get(k) for k in STATUS_KEYS} if type(status) is dict else None


def restore_full(client):
    """Best-effort FULL return after a W1 error; resumes only on a proven FULL route."""
    out = {'attempted': True, 'basis': 'pause, leave_full, status; resume only when route full and no transition pending'}
    for key, call in (('pause', client.pause), ('leave_full', client.leave)):
        try:
            out[key] = call()
            out[key + '_error'] = None
        except Exception as exc:  # noqa: BLE001 - recorded, never raised
            out[key] = None
            out[key + '_error'] = exception_details(exc, client.control)
    try:
        out['status'] = status_view(client.call('status'))
        out['status_error'] = None
    except Exception as exc:  # noqa: BLE001
        out['status'] = None
        out['status_error'] = exception_details(exc, client.control)
    status = out['status'] or {}
    full = status.get('route') == 'full' and status.get('transition_pending') is False
    out['resume'] = None
    if full and status.get('paused') is True:
        try:
            out['resume'] = client.resume()
        except Exception as exc:  # noqa: BLE001
            out['resume_error'] = exception_details(exc, client.control)
            full = False
    out['route_after'] = status.get('route')
    out['full_restored'] = bool(full)
    return out


def plain_request(url, model, result, timeout_s):
    started = time.monotonic_ns()
    try:
        headers = {'Content-Type': 'application/json'}
        key = os.environ.get('VLLM_API_KEY')
        if key:
            headers['Authorization'] = 'Bearer ' + key
        body = dict(model=model, prompt=[17] * 256, max_tokens=1,
                    temperature=0, ignore_eos=True, stream=False)
        req = urllib.request.Request(url.rstrip('/') + '/v1/completions',
                                     data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_s) as reply:
            data = json.load(reply)
            result.update(http_status=reply.status, completion_tokens=(data.get('usage') or {}).get('completion_tokens'),
                          choices=len(data.get('choices') or []))
    except Exception as exc:
        # Exception messages may echo a URL/payload. Type only.
        result['error_type'] = type(exc).__name__
    finally:
        result.update(sent_ns=started, completed_ns=time.monotonic_ns())


def clock_proof(phase, client, container_id, *, docker=None, proc='/proc', begin=None):
    """One proof: container and client identity, then one bracketed status call."""
    container = container_identity(container_id, docker=docker, proc=proc)
    local = client_identity(proc)
    reasons = container['reasons'] + local['reasons']
    if begin is not None:
        same = ('init_pid', 'time_namespace', 'boot_id')
        if any(container[k] != (begin.get('container') or {}).get(k) for k in same):
            reasons.append('container-identity-changed')
    expected = dict(proved=not reasons, boot_id=container['boot_id'],
                    time_namespace=container['time_namespace'])
    engine, status_error = None, None
    send = time.monotonic_ns()
    try:
        engine = client.call('status').get('clock')
    except Exception as exc:  # noqa: BLE001 - recorded as an unproved domain
        status_error = type(exc).__name__
    recv = time.monotonic_ns()
    # prove_clock_domain's vocabulary is kept verbatim. Here its 'client-clock-identity-unproved'
    # means the expected (container) identity failed one of the reasons listed before it.
    domain = prove_clock_domain(engine, expected, send, recv)
    if status_error:
        reasons.append('engine-status-failed:' + status_error)
    for reason in domain['reasons']:
        if reason not in reasons:
            reasons.append(reason)
    return dict(phase=phase, proved=domain['proved'] is True and not reasons, reasons=reasons,
                container=container, client=local, expected=expected, engine_clock=engine,
                domain=domain, bracket_ns=domain.get('bracket_ns'), basis=CLOCK_PROOF_BASIS)


def require_clock(record, phase, client, a, docker, proc):
    proof = clock_proof(phase, client, a.container_id, docker=docker, proc=proc,
                        begin=record['clock_proofs'].get('begin'))
    record['clock_proofs'][phase] = proof
    if proof['proved'] is not True:
        raise ValueError('client-engine-clock-unproved:%s:%s' % (phase, ','.join(proof['reasons'])))


def run(a, *, client=None, send=None, clock=time.monotonic, sleep=time.sleep, idle=None,
        docker=None, proc='/proc'):
    if urllib.parse.urlsplit(a.url).hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('loopback-required')
    client = client or NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
    send, idle = send or plain_request, idle or full_idle
    record = dict(schema=SCHEMA, verdict='INCONCLUSIVE', failure_class='HARNESS',
                  no_native_arm=True, prompt_tokens=256, output_tokens=1, observations=[], request={},
                  container_id=a.container_id, clock_proofs={}, clock_basis=CLOCK_PROOF_BASIS)
    thread = None
    try:
        record['before'] = idle(a)
        if record['before'].get('verdict') != 'PASS':
            raise ValueError('initial-full-idle-unproved')
        # Shared clock proved before the split entry, the plain request and the
        # no-reply-on-split comparison that relies on it.
        require_clock(record, 'begin', client, a, docker, proc)
        client.enter('split-w1-decode')
        client.resume()
        before = client.call('status')
        if before.get('route') != 'split' or before.get('admission_intent') is not None:
            raise ValueError('unarmed-split-unproved')
        baseline = (before.get('native_admission') or {}).get('split_stock_deferrals')
        if type(baseline) is not int:
            raise ValueError('hold-counter-unavailable')
        generation = before['route_generation']
        record['split_generation'] = generation
        thread = threading.Thread(target=send, args=(a.url, a.model, record['request'], a.timeout_s), daemon=True)
        thread.start()
        deadline = clock() + a.timeout_s
        held = False
        while clock() < deadline:
            status = client.call('status')
            admission = status.get('native_admission') or {}
            receipt = admission.get('last_split_hold_timeout')
            snap = {k: status.get(k) for k in ('route', 'route_generation', 'scheduler_requests',
                    'transition_pending', 'return_pending', 'admission_intent', 'observed_clock_ns')}
            snap.update(deferrals=admission.get('split_stock_deferrals'), timeout=receipt)
            if len(record['observations']) < 256:
                record['observations'].append(snap)
            if status.get('admission_intent') is not None:
                raise ValueError('native-admission-unexpectedly-armed')
            if (status.get('route') == 'split' and status.get('route_generation') == generation
                    and status.get('scheduler_requests') == 1 and type(snap['deferrals']) is int
                    and snap['deferrals'] > baseline):
                held = True
                # Compare only an already completed reply with the engine's
                # earlier status stamp on the proven shared monotonic clock.
                end = record['request'].get('completed_ns')
                stamp = status.get('observed_clock_ns')
                if type(end) is int and type(stamp) is int and end <= stamp:
                    raise ValueError('plain-request-completed-on-split')
            if (status.get('route') == 'full' and status.get('transition_pending') is False
                    and type(receipt) is dict):
                if (not held or receipt.get('reason') != 'unarmed-stock-prefill-hold-timeout'
                        or receipt.get('admission_armed') is not False
                        or receipt.get('route_generation') != generation
                        or receipt.get('engine_epoch') != a.epoch
                        or receipt.get('held_ns', -1) < receipt.get('bound_ns', 1)
                        or receipt.get('bound_ns') != 30_000_000_000
                        or status.get('route_generation', -1) <= generation):
                    raise ValueError('automatic-full-return-unproved')
                record['automatic_return'] = snap
                break
            if record['request'].get('error_type'):
                raise ValueError('plain-client-failed')
            sleep(.5)
        else:
            raise ValueError('automatic-full-return-timeout')
        thread.join(max(0, deadline - clock()))
        request = record['request']
        if (thread.is_alive() or request.get('error_type') or request.get('http_status') != 200
                or request.get('completion_tokens') != 1 or request.get('choices') != 1):
            raise ValueError('plain-completion-unproved')
        # The same proof again at the end: the domain held across the run.
        require_clock(record, 'end', client, a, docker, proc)
        record['after'] = idle(a)
        if record['after'].get('verdict') != 'PASS':
            raise ValueError('final-full-idle-unproved')
        record.update(verdict='PASS', failure_class=None)
    except Exception as exc:
        record.update(cause=str(exc) if type(exc) is ValueError else type(exc).__name__)
    finally:
        if record['verdict'] != 'PASS':
            # Never resume an unknown/split route; use the reviewed restoration.
            record['cleanup'] = restore_full(client)
        if thread:
            thread.join(1)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('url', 'model', 'run-id', 'container-id'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--control-timeout-s', type=float, default=30)
    p.add_argument('--timeout-s', type=float, default=100)
    a = p.parse_args(argv)
    if a.out.exists():
        return 2
    record = run(a)
    with a.out.open('x') as f:
        json.dump(record, f, sort_keys=True, indent=2)
        f.write('\n')
    print('NATIVE41_UNARMED ' + record['verdict'])
    return 0 if record['verdict'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
