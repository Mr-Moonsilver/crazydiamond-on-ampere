#!/usr/bin/env python3
"""Plain API prefill on unarmed split; real 30s hold -> applied FULL -> completion.

No native prefill_begin/admission action is sent. The explicit split command
selects decode-only phase; its existing idle hold timeout performs the return.
Receipts contain counts/identities, never request/auth payloads.
"""
import argparse
import json
import os
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request

from e3_native_runtime_client import NativeRuntimeClient
from e3_native39_trunk_checks import full_idle


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


def run(a, *, client=None, send=None, clock=time.monotonic, sleep=time.sleep, idle=None):
    if urllib.parse.urlsplit(a.url).hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('loopback-required')
    client = client or NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
    send, idle = send or plain_request, idle or full_idle
    record = dict(schema='native40-unarmed-box/1', verdict='INCONCLUSIVE', failure_class='HARNESS',
                  no_native_arm=True, prompt_tokens=256, output_tokens=1, observations=[], request={})
    thread = None
    try:
        record['before'] = idle(a)
        if record['before'].get('verdict') != 'PASS':
            raise ValueError('initial-full-idle-unproved')
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
        # Shared clock is required for the no-reply-on-split comparison.
        control_clock = status.get('clock') or {}
        from e3_native39_w1 import clock_identity
        local = clock_identity()
        if (not local.get('boot_id') or control_clock.get('boot_id') != local['boot_id']
                or control_clock.get('time_namespace') != local.get('time_namespace')):
            raise ValueError('client-engine-clock-unproved')
        record['after'] = idle(a)
        if record['after'].get('verdict') != 'PASS':
            raise ValueError('final-full-idle-unproved')
        record.update(verdict='PASS', failure_class=None)
    except Exception as exc:
        record.update(cause=str(exc) if type(exc) is ValueError else type(exc).__name__)
    finally:
        if record['verdict'] != 'PASS':
            # Never resume an unknown/split route; use the reviewed restoration.
            from e3_native39_w1 import restore_full
            record['cleanup'] = restore_full(client)
        if thread:
            thread.join(1)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('url', 'model', 'run-id'):
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
    print('NATIVE40_UNARMED ' + record['verdict'])
    return 0 if record['verdict'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
