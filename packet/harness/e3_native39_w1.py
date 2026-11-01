#!/usr/bin/env python3
"""Native39 trunk W1: e3_native_w1's fixed requests and windows on a sealed cohort.

Same requests, routes, windows, interval proof and output schema as
e3_native_w1 (the W2 rounds read this file as their W1 denominator).  Added for
Codex's queue review of 1c334bc (items 1 and 2) and the lead's P1 conditions:

Seal, per repeat (FULL and split), against the engine's own state:
  drain  before the cohort opens, status while running shows scheduler_requests
         0 (the previous repeat's cancelled cohort has left the scheduler),
         polled for at most DRAIN_TIMEOUT_S; else the repeat is INVALID.
  entry  status while running: unpaused, route full, no transition or control
         pending, scheduler_requests == 8 (the pinned scheduler's unfinished
         count, waiting + running).  Then the paused FULL barrier: on every
         rank prefill_requests 0, decode_eligible 8, live 8, and
         live_request_uids exactly the cohort's 8 engine ids (each stream's
         SSE response id plus the engine's '-<8 hex>' suffix, one to one).
  begin  the paused barrier on the measured route, before resume: the same.
  end    the paused barrier after the window: the same.
An entry seal failure never enters split.  A begin seal failure returns to
FULL without resuming on split.  Either marks the repeat INVALID and stops the
run (no later repeat).

Cleanup: every exception path makes a best-effort pause, leave_full and
status read, and resumes only when the engine reports route full with no
transition pending.  A split or unknown route stays paused, so the queue's
typed FULL check withholds the plain W1.

Each repeat records its wall span and the cohort's local client ports (for
the access-log ingress check), and a harness CLOCK_MONOTONIC bracket around
its FULL return (for the return-timing reader).

Residual, stated: under PAUSED_ALL the pinned scheduler reports 0 unfinished
requests, so a request that arrives between the unpaused status read and the
pause is invisible to the entry seal.  The ingress check marks that repeat
INVALID afterwards.  Holding such a request safely is the engine's job
(native40 unarmed hold).
"""
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

import e3_native_w1 as base
import e3_w1_decode_driver as w1
from e3_native_runtime_client import NativeRuntimeClient
from e3_native_qualification_driver import exception_details

COHORT = 8
DRAIN_TIMEOUT_S = 30.0
DRAIN_POLL_S = 0.05
ENGINE_SUFFIX = re.compile(r'[0-9a-f]{8}')
STATUS_KEYS = ('route', 'route_generation', 'paused', 'transition_pending', 'control_pending',
               'return_pending', 'scheduler_requests', 'observed_clock_ns', 'clock')


class Unsealed(ValueError):
    pass


def clock_identity():
    record = {'basis': 'CLOCK_MONOTONIC', 'proc': 'harness'}
    try:
        record['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() or None
        record['time_namespace'] = os.readlink('/proc/self/ns/time')
    except OSError as exc:
        record.update(boot_id=None, time_namespace=None, unavailable=type(exc).__name__)
    return record


def response_ids(states):
    ids = [s.response_id for s in states]
    if (len(ids) != COHORT or any(type(i) is not str or not i for i in ids)
            or len(set(ids)) != len(ids)):
        raise Unsealed('cohort-response-ids-unavailable')
    return ids


def client_port(state):
    for sock in (state.cancel_sock, state.sock):
        try:
            return int(sock.getsockname()[1])
        except (AttributeError, OSError, TypeError, IndexError, ValueError):
            continue
    return None


def owner(uid, ids):
    hits = [i for i in ids if type(uid) is str and (
        uid == i or (uid.startswith(i + '-') and ENGINE_SUFFIX.fullmatch(uid[len(i) + 1:])))]
    return hits[0] if len(hits) == 1 else None


def uid_reasons(uids, ids):
    """The engine's live ids must be the cohort's, one each; else the reasons."""
    if type(uids) is not list or len(uids) != len(set(map(str, uids))):
        return ['live-uids-unavailable']
    reasons = []
    owned = [owner(uid, ids) for uid in uids]
    if any(o is None for o in owned):
        reasons.append('foreign-live-uid')
    if sorted(o for o in owned if o is not None) != sorted(ids):
        reasons.append('cohort-uids-not-exact')
    return reasons


def entry_reasons(status):
    reasons = []
    if type(status) is not dict:
        return ['entry-status-unavailable']
    if status.get('paused') is not False:
        reasons.append('entry-status:paused')
    if status.get('route') != 'full':
        reasons.append('entry-status:route')
    if status.get('transition_pending') is not False or status.get('control_pending') is not None:
        reasons.append('entry-status:transition-or-control-pending')
    if status.get('return_pending'):
        reasons.append('entry-status:return-pending')
    if status.get('scheduler_requests') != COHORT:
        reasons.append('entry-status:scheduler-requests')
    return reasons


def seal_reasons(client, reply, ids, route):
    rows = client.workers(reply.get('barrier'))
    reasons = []
    if reply.get('route') != route:
        reasons.append('snapshot:route')
    for rank in sorted(rows):
        row = rows[rank]
        live = row.get('live_decode') or {}
        if row.get('route') != route:
            reasons.append('rank%d:route' % rank)
        if (row.get('prefill_requests') != 0 or row.get('decode_eligible') != COHORT
                or row.get('live_decode_requests') != COHORT or live.get('available') is not True
                or live.get('prefill_requests') != 0):
            reasons.append('rank%d:live-cohort' % rank)
        reasons.extend('rank%d:%s' % (rank, r) for r in uid_reasons(live.get('live_request_uids'), ids))
    return reasons


def status_view(status):
    return {k: status.get(k) for k in STATUS_KEYS} if type(status) is dict else None


def wait_drained(client):
    """Status (running) until scheduler_requests is 0: nothing but the next cohort will be registered."""
    deadline, polls = time.monotonic() + DRAIN_TIMEOUT_S, 0
    while True:
        status = client.call('status')
        polls += 1
        if status.get('paused') is False and status.get('scheduler_requests') == 0:
            return {'polls': polls, 'status': status_view(status)}
        if time.monotonic() >= deadline:
            raise Unsealed('scheduler-not-empty-before-cohort')
        time.sleep(DRAIN_POLL_S)


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


def run(cfg, url, model, client=None):
    client = client or NativeRuntimeClient(url, cfg.run_id, timeout_s=cfg.controller_timeout_s)
    requests, meta = w1.build_requests(cfg.workload)
    _, hashes, digest = w1.build_request_set(model, requests, cfg.workload)
    repeats, abort = [], None
    for repeat, route in enumerate(cfg.workload.route_plan):
        row = {'repeat': repeat, 'route': route, 'valid': False, 'reasons': [],
               'streams': [], 'native_receipts': {}, 'metrics': {}, 'seal': {},
               'cohort': None, 'wall': {'open_s': None, 'end_s': None}}
        repeats.append(row)
        if abort:
            row.update(not_run=True, reasons=['earlier-refusal:' + abort])
            continue
        states, stop, window = [], threading.Event(), None
        row['wall']['open_s'] = time.time()
        try:
            row['native_receipts']['drained'] = wait_drained(client)
            row['native_receipts']['prepare'] = client.prepare('full', 'full-prefill-solo')
            for i, messages in enumerate(requests):
                state = w1.StreamState(i, messages, meta[i], w1.request_body(model, messages, cfg.workload))
                states.append(state)
                state.thread = threading.Thread(target=w1.stream_reader, args=(state, cfg, url, stop), daemon=True)
                state.thread.start()
            deadline = time.monotonic() + cfg.workload.open_timeout_s
            for state in states:
                state.opened.wait(max(0, deadline - time.monotonic()))
            if any(s.first_at_ns is None or s.finish_at_ns is not None or s.error is not None for s in states):
                raise ValueError('native-w1-cohort-not-live')
            ids = response_ids(states)
            row['cohort'] = {'response_ids': ids, 'client_ports': [client_port(s) for s in states]}
            entry = client.call('status')
            row['native_receipts']['entry_status'] = status_view(entry)
            held = client.pause()
            row['native_receipts']['entry'] = held
            row['seal']['entry'] = entry_reasons(entry) + seal_reasons(client, held, ids, 'full')
            if row['seal']['entry']:
                row['reasons'].extend('seal-entry:' + r for r in row['seal']['entry'])
                raise Unsealed('unsealed-entry')
            if route == 'split':
                row['native_receipts']['enter'] = client.enter('split-w1-decode')
            before = client.pause()
            row['native_receipts']['begin'] = before
            row['seal']['begin'] = seal_reasons(client, before, ids, route)
            if row['seal']['begin']:
                row['reasons'].extend('seal-begin:' + r for r in row['seal']['begin'])
                raise Unsealed('unsealed-begin')
            row['native_receipts']['resume'] = client.resume()
            start = time.monotonic_ns()
            deadline_ns = start + int(cfg.workload.window_s * 1e9)
            while time.monotonic_ns() < deadline_ns:
                if any(s.error is not None or s.finish_at_ns is not None for s in states):
                    raise ValueError('native-w1-stream-ended-in-window')
                time.sleep(min(.02, max(0, (deadline_ns - time.monotonic_ns()) / 1e9)))
            end = time.monotonic_ns()
            window = {'measured': True, 'start_ns': start, 'end_ns': end, 'wall_s': (end - start) / 1e9}
            after = client.pause()
            row['native_receipts']['end'] = after
            row['seal']['end'] = seal_reasons(client, after, ids, route)
            row['reasons'].extend('seal-end:' + r for r in row['seal']['end'])
            row['interval'] = base.interval_proof(before, after, client, route)
            row['reasons'].extend(row['interval']['reasons'])
            send_ns = time.monotonic_ns()
            row['native_receipts']['leave_full'] = client.leave()
            row['return_bracket'] = {'send_ns': send_ns, 'reply_ns': time.monotonic_ns(),
                                     'clock': clock_identity()}
            row['native_receipts']['cleanup_resume'] = client.resume()
        except Exception as exc:
            row['error'] = exception_details(exc, client.control)
            kind = 'native-w1-unsealed:' if isinstance(exc, Unsealed) else 'native-w1-refusal:'
            row['reasons'].append(kind + row['error']['error_message'])
            abort = row['error']['error_message']
            row['cleanup'] = restore_full(client)
        finally:
            row['streams'], _ = w1._finalize_streams(cfg, states, stop, window, time.monotonic_ns())
            row['window'] = window
            row['metrics'] = w1._repeat_metrics(row['streams'], window)
            row['exposure'] = w1._exposure(row['streams'], cfg.workload)
            row['reasons'].extend(row['exposure']['reasons'])
            if any(r.get('drain_timeout') for r in row['streams']):
                row['reasons'].append('client-reader-drain-incomplete')
            if [s.get('request_sha256') for s in row['streams']] != hashes:
                row['reasons'].append('request-bytes-changed')
            sealed = all(row['seal'].get(k) == [] for k in ('entry', 'begin', 'end'))
            row['sealed'] = sealed
            row['valid'] = sealed and not row['reasons'] and row.get('interval', {}).get('complete') is True
            row['wall']['end_s'] = time.time()
    complete = len(repeats) == cfg.workload.repeats and all(r['valid'] for r in repeats)
    routes = {}
    for route in ('full', 'split'):
        rows = [r for r in repeats if r['route'] == route]
        valid = bool(rows) and all(r['valid'] for r in rows)
        values = [r['metrics'].get('aggregate_tokens_per_s') for r in rows]
        gaps = [g for r in rows for s in r['streams'] for g in s.get('window', {}).get('gaps_ms', [])]
        routes[route] = {'valid_repeats': [r['repeat'] for r in rows if r['valid']],
            'aggregate_tokens_per_s_median': w1._pct(values, .5) if valid and all(type(v) in (int, float) for v in values) else None,
            'gap_ms_p99': w1._pct(gaps, .99) if valid else None}
    return {'schema': 2, 'mode': 'native-overlap', 'config': {'WORKLOAD_SHA256': cfg.workload.manifest_sha256,
        'REPEATS': cfg.workload.repeats, 'request_set_sha256': digest}, 'repeats': repeats,
        'native39': {'seal': 'entry status + paused FULL barrier, begin and end barriers: exact cohort ids, no prefill',
                     'sealed_repeats': [r['repeat'] for r in repeats if r.get('sealed')],
                     'cleanups': [r['repeat'] for r in repeats if 'cleanup' in r],
                     'full_restored_after_error': all((r['cleanup'] or {}).get('full_restored') is True
                                                      for r in repeats if 'cleanup' in r),
                     'harness_clock': clock_identity()},
        'summary': {'verdict': 'COMPLETE' if complete else 'INCONCLUSIVE', 'routes': routes,
            'totals': {'required': cfg.workload.repeats, 'attempted': len(repeats),
                       'valid': sum(r['valid'] for r in repeats), 'invalid': sum(not r['valid'] for r in repeats), 'contract': 0},
            'all_declared_repeats_retained': True, 'filtered_invalid': False, 'retry_selected': False, 'picked': False}}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        return 2
    url, model, out = argv
    if Path(out).exists():
        return 2
    try:
        cfg = w1.config_from_env(os.environ, out)
        result = run(cfg, url, model)
    except Exception as exc:
        result = {'schema': 2, 'verdict': 'INCONCLUSIVE', 'error': exception_details(exc)}
    with open(out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    complete = result.get('summary', {}).get('verdict') == 'COMPLETE'
    print('NATIVE39_W1 ' + ('COMPLETE' if complete else 'INCONCLUSIVE'))
    return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
