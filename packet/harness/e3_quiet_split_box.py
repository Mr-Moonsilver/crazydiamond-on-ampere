#!/usr/bin/env python3
"""Quiet-SPLIT control (brief 11 C): live decoders on the split route with no native work.

What it measures: decode alone on the SPLIT route with the stage-local selection decode_merge=always (the merge active
on every step whatever the policy phase, stage_overlap._decode_merge_wanted), at 8, 4 and 1 live decoders, HOLD_S s
each. Nothing native runs: the manual lease (any non-read control, native_auto_arm.manual_control) keeps the auto-arm
gate closed ('gate-manual-owned'), no native admission is armed, and every poll proves it.

The flow per stage, as e3_native_w1 and e3_native41_unarmed_box do it:
  1. N streaming decoders opened on the typed FULL route (small token-id prompts, ignore_eos, max_tokens far past the
     hold), each past its first token: their prefill is done before the split, so the split route holds no stock
     prefill (native41: an unarmed SPLIT holds new stock prefill and returns FULL after 30 s by itself);
  2. pause_enter_split('split-w1-decode'), resume; the split route proved: route split, generation advanced,
     admission_intent None, scheduler_requests == N;
  3. HOLD_S s of status polls, each one a runner guard (POLL_GUARDS): route split, the same generation, no
     transition or return pending, admission_intent None, scheduler_requests == N, native admitted unchanged, no
     native chunk, auto-arm idle and manual-owned, the selection still decode_merge=always; at the end every decoder
     advanced inside the window and the engine's merge_steps grew;
  4. pause, leave_full, resume (route full proved); the streams closed; scheduler_requests 0 again.
Before the stages: one stage-local selection while paused idle FULL (local, overlap, defer_bind, lazy_recv as the
n3 selection, decode_merge=always), acknowledged; after them the previous selection's five fields are put back (best
effort, recorded). Any refusal or guard failure: the reviewed FULL restoration (restore_full, never resumes an unknown
or split route), the streams closed, INCONCLUSIVE/HARNESS with the cause.

The declaration (DECLARED) is checked before any engine call: stages exactly 8,4,1, hold in 20..30 s, merge always.
A different declaration exits 2 with no control sent (declaration guard). Receipts hold counts, routes and times;
never env, request bodies or auth (the key is read from VLLM_API_KEY and only used as a header).
Speed is descriptive (aggregate tokens/s, per-stream gaps inside the window); the verdict is the guards only.
"""
import argparse
import http.client
import json
import os
from pathlib import Path
import socket
import threading
import time
import urllib.parse

from e3_native_runtime_client import NativeRuntimeClient
from e3_native_qualification_driver import exception_details

# Copied verbatim from e3_native41_unarmed_box (there from e3_native39_w1): STATUS_KEYS, status_view, restore_full, so
# the import closure stays e3_native_runtime_client's (test_quiet_split_box pins the copy against the source).
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


SCHEMA = 'quiet-split-box/1'
STAGE_CONTROL = 'native_stage_local'
SPLIT_PHASE = 'split-w1-decode'
DECLARED = dict(stages=(8, 4, 1), hold_min_s=20.0, hold_max_s=30.0, decode_merge='always', lazy_recv=True)
PROMPT_TOKENS = 64
MAX_TOKENS = 8192
POLL_S = 0.5
SNAPSHOT_LIMIT = 80
POLL_GUARDS = ('route-left-split', 'route-generation-changed', 'transition-or-return-pending', 'admission-armed',
               'scheduler-requests-not-live', 'native-admitted', 'native-chunk-open', 'auto-arm-active',
               'manual-lease-lost', 'decode-merge-not-always')
SELECTION_FIELDS = ('local', 'overlap', 'defer_bind', 'lazy_recv', 'decode_merge')


class GuardError(ValueError):
    """A runner guard: the record names it in ``cause``."""


def declaration_reasons(stages, hold_s, merge):
    """Empty when the CLI declaration is the declared quiet-SPLIT control, else each difference."""
    out = []
    if tuple(stages) != DECLARED['stages']:
        out.append('stages-not-8-4-1')
    if type(hold_s) not in (int, float) or not DECLARED['hold_min_s'] <= hold_s <= DECLARED['hold_max_s']:
        out.append('hold-not-20-30-s')
    if merge != DECLARED['decode_merge']:
        out.append('decode-merge-not-always')
    return out


def stage_view(status):
    """The stage-local decode-step row of one status reply (None when absent)."""
    try:
        return status['native_stage_local']['overlap']['decode_step']
    except (KeyError, TypeError):
        return None


def poll_reasons(status, *, generation, live, admitted):
    """The runner guards on one status poll inside a hold; empty when quiet SPLIT holds."""
    out = []
    if status.get('route') != 'split':
        out.append('route-left-split')
    if status.get('route_generation') != generation:
        out.append('route-generation-changed')
    if status.get('transition_pending') is not False or status.get('return_pending') not in (None, False):
        out.append('transition-or-return-pending')
    if status.get('admission_intent') is not None:
        out.append('admission-armed')
    if status.get('scheduler_requests') != live:
        out.append('scheduler-requests-not-live')
    if (status.get('native_admission') or {}).get('admitted') != admitted:
        out.append('native-admitted')
    if status.get('chunk') is not None or (status.get('owned_work') or {}).get('chunk') not in (None, False):
        out.append('native-chunk-open')
    arm = status.get('auto_arm')
    if type(arm) is not dict or arm.get('active') is not False:
        out.append('auto-arm-active')
    if type(arm) is not dict or arm.get('manual_owned') is not True:
        out.append('manual-lease-lost')
    step = stage_view(status)
    if type(step) is not dict or step.get('decode_merge') != DECLARED['decode_merge']:
        out.append('decode-merge-not-always')
    return out


def merge_steps(status):
    step = stage_view(status)
    value = ((step or {}).get('counters') or {}).get('merge_steps')
    return value if type(value) is int else None


class Stream:
    """One streaming decoder: first token, chunk times, end. Times on the run's monotonic ns clock."""

    def __init__(self, index):
        self.index = index
        self.first_ns = None
        self.chunks = []
        self.finished_ns = None
        self.error = None
        self.opened = threading.Event()
        self.stop = threading.Event()
        self.thread = None
        self.sock = None

    def view(self, window):
        start, end = window if window else (None, None)
        inside = [t for t in self.chunks if start is not None and start <= t <= end]
        gaps = [(b - a) / 1e6 for a, b in zip(inside, inside[1:])]
        return dict(index=self.index, first=self.first_ns is not None, chunks=len(self.chunks),
                    window_chunks=len(inside), window_gap_ms_max=max(gaps) if gaps else None,
                    window_gap_ms_p50=sorted(gaps)[len(gaps) // 2] if gaps else None,
                    finished_early=self.finished_ns is not None and (end is None or self.finished_ns <= end),
                    error=self.error)


def request_body(model, stage, index):
    # Distinct token-id prompts per stage and stream; nothing from a workload or a user.
    return dict(model=model, prompt=[1000 + 16 * stage + index] * PROMPT_TOKENS, max_tokens=MAX_TOKENS,
                temperature=0, ignore_eos=True, stream=True)


def stream_reader(url, model, stage, stream, now_ns, timeout_s):
    try:
        headers = {'Content-Type': 'application/json'}
        key = os.environ.get('VLLM_API_KEY')
        if key:
            headers['Authorization'] = 'Bearer ' + key
        parts = urllib.parse.urlsplit(url)
        # http.client, not urllib: close_streams shuts this socket down, which ends a read blocked between chunks
        # (a paused engine sends nothing); closing a urllib response from another thread does not (test_quiet_split_box).
        conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout_s)
        try:
            conn.connect()
            stream.sock = conn.sock
            if stream.stop.is_set():
                return
            conn.request('POST', parts.path.rstrip('/') + '/v1/completions',
                         body=json.dumps(request_body(model, stage, stream.index)).encode(), headers=headers)
            reply = conn.getresponse()
            if reply.status != 200:
                stream.error = 'http-%d' % reply.status
                return
            while not stream.stop.is_set():
                line = reply.readline()
                if not line:
                    if not stream.stop.is_set():
                        stream.finished_ns = now_ns()
                    break
                if not line.startswith(b'data: '):
                    continue
                if line.strip() == b'data: [DONE]':
                    stream.finished_ns = now_ns()
                    break
                t = now_ns()
                if stream.first_ns is None:
                    stream.first_ns = t
                    stream.opened.set()
                stream.chunks.append(t)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - type only: a message may echo a URL or payload
        if not stream.stop.is_set():
            stream.error = type(exc).__name__
    finally:
        stream.opened.set()


def start_stream(url, model, stage, index, now_ns, timeout_s):
    stream = Stream(index)
    stream.thread = threading.Thread(target=stream_reader, args=(url, model, stage, stream, now_ns, timeout_s),
                                     daemon=True)
    stream.thread.start()
    return stream


def close_streams(streams, join_s=5.0):
    for s in streams:
        s.stop.set()
        try:
            if s.sock is not None:
                s.sock.shutdown(socket.SHUT_RDWR)     # a reader blocked between chunks (a paused engine) ends here
        except OSError:  # an already closed socket; the join below records a reader that did not end
            pass
    for s in streams:
        if s.thread is not None:
            s.thread.join(join_s)
    return [s.index for s in streams if s.thread is not None and s.thread.is_alive()]


def select(client, fields):
    """One stage-local selection while paused idle FULL: pause, the command, resume (resume always tried)."""
    client.pause()
    try:
        return client.call(STAGE_CONTROL, **fields)
    finally:
        client.resume()


def wait_requests(client, want, deadline, clock, sleep):
    while True:
        status = client.call('status')
        if status.get('scheduler_requests') == want:
            return status
        if clock() >= deadline:
            raise GuardError('scheduler-requests-not-%d' % want)
        sleep(POLL_S)


def run_stage(client, a, live, record, *, starter, clock, sleep, now_ns, admitted):
    row = dict(live=live, reasons=[], snapshots=[], polls=0)
    record['stages'].append(row)
    streams = []
    try:
        for i in range(live):
            streams.append(starter(a.url, a.model, live, i, now_ns, a.stream_timeout_s))
        deadline = clock() + a.open_timeout_s
        for s in streams:
            s.opened.wait(max(0.0, deadline - clock()))
        if any(s.first_ns is None or s.error is not None or s.finished_ns is not None for s in streams):
            raise GuardError('decoders-not-live-on-full')
        before = client.call('status')
        if before.get('route') != 'full' or before.get('scheduler_requests') != live:
            raise GuardError('full-cohort-not-%d' % live)
        client.enter(SPLIT_PHASE)
        client.resume()
        status = client.call('status')
        generation = status.get('route_generation')
        if (status.get('route') != 'split' or type(generation) is not int
                or generation <= before.get('route_generation', generation)
                or status.get('admission_intent') is not None or status.get('scheduler_requests') != live):
            raise GuardError('quiet-split-entry-unproved')
        row['split_generation'] = generation
        merge0 = merge_steps(status)
        start = now_ns()
        end_at = clock() + a.hold_s
        while True:
            reasons = poll_reasons(status, generation=generation, live=live, admitted=admitted)
            row['polls'] += 1
            if len(row['snapshots']) < SNAPSHOT_LIMIT:
                snap = status_view(status)
                snap.update(admitted=(status.get('native_admission') or {}).get('admitted'),
                            merge_steps=merge_steps(status), reasons=reasons)
                row['snapshots'].append(snap)
            if reasons:
                raise GuardError('poll:' + ','.join(reasons))
            if any(s.error is not None or s.finished_ns is not None for s in streams):
                raise GuardError('decoder-ended-in-hold')
            if clock() >= end_at:
                break
            sleep(POLL_S)
            status = client.call('status')
        end = now_ns()
        row['window'] = dict(start_ns=start, end_ns=end, wall_s=(end - start) / 1e9)
        merge1 = merge_steps(status)
        row['merge_steps'] = None if merge0 is None or merge1 is None else merge1 - merge0
        if type(row['merge_steps']) is not int or row['merge_steps'] < 1:
            raise GuardError('merge-not-active-in-hold')
        views = [s.view((start, end)) for s in streams]
        if any(v['window_chunks'] < 1 for v in views):
            raise GuardError('decoder-stalled-in-hold')
        client.pause()
        client.leave()
        client.resume()
        after = client.call('status')
        if after.get('route') != 'full' or after.get('route_generation', -1) <= generation:
            raise GuardError('full-return-unproved')
    finally:
        row['open_after_close'] = close_streams(streams)
        row['streams'] = [s.view((row['window']['start_ns'], row['window']['end_ns']) if 'window' in row else None)
                          for s in streams]
        tokens = sum(v['window_chunks'] for v in row['streams'])
        wall = (row.get('window') or {}).get('wall_s')
        row['aggregate_tokens_per_s'] = tokens / wall if wall else None
    wait_requests(client, 0, clock() + a.drain_timeout_s, clock, sleep)
    if row['open_after_close']:
        raise GuardError('stream-reader-not-closed')
    return row


def run(a, *, client=None, starter=None, clock=time.monotonic, sleep=time.sleep, now_ns=time.monotonic_ns):
    record = dict(schema=SCHEMA, verdict='INCONCLUSIVE', failure_class='HARNESS', declared=dict(DECLARED),
                  requested=dict(stages=list(a.stages), hold_s=a.hold_s, decode_merge=a.decode_merge),
                  no_native_work=True, stages=[], basis=__doc__.split('\n', 2)[0])
    declared = declaration_reasons(a.stages, a.hold_s, a.decode_merge)
    if declared:
        record['cause'] = 'declaration:' + ','.join(declared)
        record['control_sent'] = False
        return record
    if urllib.parse.urlsplit(a.url).hostname not in ('127.0.0.1', 'localhost', '::1'):
        record['cause'] = 'loopback-required'
        record['control_sent'] = False
        return record
    client = client or NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
    starter = starter or start_stream
    selected = None
    try:
        status = client.call('status')
        if (status.get('route') != 'full' or status.get('scheduler_requests') != 0
                or status.get('admission_intent') is not None or (status.get('auto_arm') or {}).get('active') is not False):
            raise GuardError('entry-not-idle-full')
        admitted = (status.get('native_admission') or {}).get('admitted')
        if type(admitted) is not int:
            raise GuardError('native-admitted-unavailable')
        record['native_admitted_before'] = admitted
        reply = select(client, dict(local=True, overlap=True, defer_bind=True, lazy_recv=DECLARED['lazy_recv'],
                                    decode_merge=DECLARED['decode_merge']))
        selected = {k: reply.get(k) for k in SELECTION_FIELDS}
        selected.update({'previous_' + k: reply.get('previous' if k == 'local' else 'previous_' + k)
                         for k in SELECTION_FIELDS})
        record['selection'] = selected
        if (reply.get('local') is not True or reply.get('overlap') is not True or reply.get('defer_bind') is not True
                or reply.get('lazy_recv') is not DECLARED['lazy_recv']
                or reply.get('decode_merge') != DECLARED['decode_merge']):
            raise GuardError('selection-not-acknowledged')
        for live in DECLARED['stages']:
            run_stage(client, a, live, record, starter=starter, clock=clock, sleep=sleep, now_ns=now_ns,
                      admitted=admitted)
        status = client.call('status')
        record['native_admitted_after'] = (status.get('native_admission') or {}).get('admitted')
        if record['native_admitted_after'] != admitted:
            raise GuardError('native-admitted')
        record.update(verdict='PASS', failure_class=None)
    except Exception as exc:  # noqa: BLE001 - recorded; the verdict stays INCONCLUSIVE
        record['cause'] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
    finally:
        if record['verdict'] != 'PASS':
            record['cleanup'] = restore_full(client)
        if record['verdict'] == 'PASS' or record['cleanup'].get('full_restored') is True:
            record['selection_restore'] = restore_selection(client, selected)
        else:
            # select() resumes; never on a route the restoration did not prove FULL.
            record['selection_restore'] = dict(attempted=False, reason='FULL not restored')
    return record


def restore_selection(client, selected):
    """Put the previous selection's five fields back (best effort, recorded, never raised)."""
    if not selected:
        return dict(attempted=False, reason='nothing selected')
    previous = {k: selected.get('previous_' + k) for k in SELECTION_FIELDS}
    if any(v is None for v in previous.values()):
        return dict(attempted=False, reason='previous selection not reported', previous=previous)
    out = dict(attempted=True, previous=previous)
    try:
        reply = select(client, previous)
        out['restored'] = all(reply.get(k) == previous[k] for k in SELECTION_FIELDS)
    except Exception as exc:  # noqa: BLE001
        out.update(restored=False, error=str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('url', 'model', 'run-id'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stages', default='8,4,1')
    p.add_argument('--hold-s', type=float, default=25.0)
    p.add_argument('--decode-merge', default='always')
    p.add_argument('--control-timeout-s', type=float, default=60)
    p.add_argument('--open-timeout-s', type=float, default=60)
    p.add_argument('--drain-timeout-s', type=float, default=60)
    p.add_argument('--stream-timeout-s', type=float, default=120)
    a = p.parse_args(argv)
    try:
        a.stages = tuple(int(x) for x in a.stages.split(','))
    except ValueError:
        a.stages = ()
    if a.out.exists():
        return 2
    record = run(a)
    with a.out.open('x') as f:
        json.dump(record, f, sort_keys=True, indent=2)
        f.write('\n')
    print('QUIET_SPLIT ' + record['verdict'] + ('' if record['verdict'] == 'PASS' else ' ' + record.get('cause', '')))
    return 0 if record['verdict'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
