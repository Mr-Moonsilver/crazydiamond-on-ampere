#!/usr/bin/env python3
"""Quiet-SPLIT control v2 (brief 13 item 2): live decoders on the split route with no native work, at the replay arms'
histories, with attn-meta selected as in the n3 arms.

What it measures: decode alone on the SPLIT route with the stage-local selection decode_merge=always (the merge active
on every step whatever the policy phase, stage_overlap._decode_merge_wanted) and attn_meta_select {on} (the n3 arms'
selection, compose5_arms.ATTN), on the declared PLAN of (prompt tokens, live decoders) stages, HOLD_S s each:
120,000-token histories at 8, 6, 4, 2 and 1 live, then 72,000 at 4 and 1 (the replay arms' busy split forwards have max
histories 62k-134k; REVIEW-qsplit2 finding 2). Nothing native runs: the manual lease (taken by the queue's typed FULL
entry, held as after the sp arm) keeps the auto-arm gate closed ('gate-manual-owned'), no native admission is armed,
and every poll proves it.

Prompts: per stream index i a pseudo-random token-id sequence (random.Random(PROMPT_SEED + i), ids in
[ID_LO, ID_HI)); a 72,000 stage sends the first 72,000 ids of the same stream's 120,000 prompt. Nothing from a
workload or a user.
Warm (once, after the selection, on FULL): each of the eight 120,000-token prompts with max_tokens 1, concurrently;
the prefix cache then holds every stage's prompt (2.5M-token pool, prefix caching on), so a stage's streams reach their
first token without a long prefill and the history stays in its band.

The flow per stage, as e3_native_w1 and e3_native41_unarmed_box do it:
  1. N streaming decoders opened on the typed FULL route (the stage's prompts, ignore_eos, max_tokens MAX_TOKENS), each
     past its first token; at the split no stream may have decoded more than PRE_HOLD_TOKENS (else 'history-off-band');
  2. pause_enter_split('split-w1-decode'), resume; the split route proved: route split, generation advanced,
     admission_intent None, scheduler_requests == N;
  3. HOLD_S s of status polls, each one a runner guard (POLL_GUARDS); at the end every decoder advanced inside the
     window and the engine's merge_steps grew;
  4. pause, leave_full, resume (route full proved); the streams closed (socket shutdown);
  5. the disconnect check: scheduler_requests 0 within ABORT_BOUND_S is abort 'disconnect'. Otherwise the fallback:
     the requests end by themselves at MAX_TOKENS on the FULL route (the stamp samples nothing there: the manual lease is
     held and the route is no longer the manual split); scheduler_requests 0 within drain_timeout_s is abort
     'max_tokens', and the next stage runs; never 0: 'requests-not-drained'. Each stage records its abort and drain_s.
Before the stages, one pause while idle FULL: the stage-local selection (local, overlap, defer_bind, lazy_recv as the
n3 selection, decode_merge=always) and attn_meta_select {on} through the reviewed worker adapter (every rank's
after.selected True, order ok); then the warm. After the stages, one pause while idle FULL: attn_meta_read (engaged:
the batched counter grew on every rank, else 'attn-meta-not-engaged'), the previous stage-local fields and the
previous attn-meta selection put back (best effort, recorded).
Any refusal, guard failure or SIGTERM (the queue's timeout): the reviewed FULL restoration (restore_full, never resumes
an unknown or split route), the streams closed, then a status read proves the split lease released (control_route
None on a FULL route: 'split_lease_released'); INCONCLUSIVE/HARNESS with the cause. After a SIGTERM the control calls
of the cleanup are bounded by CLEANUP_TIMEOUT_S and the best-effort selection restore is skipped (the engine is stopped
next); the record is still written.

The declaration (DECLARED) is checked before any engine call: the plan exactly PLAN, hold in 20..30 s, merge always,
max_tokens MAX_TOKENS. A different declaration exits 2 with no control sent (declaration guard). Receipts hold counts,
routes and times; never env, request bodies or auth (the key is read from VLLM_API_KEY and only used as a header).
Speed is descriptive (aggregate tokens/s, per-stream gaps inside the window); the verdict is the guards only.
"""
import argparse
import http.client
import json
import os
from pathlib import Path
import random
import secrets
import signal
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


# Copied verbatim from e3_attn_meta_box (Worker, rank_rows), so the import closure stays e3_native_runtime_client's
# (test_quiet_split_box pins the copy against the source when present).
WORKER_ACTION = 'qualification_worker'
SELECT, READ = 'attn_meta_select', 'attn_meta_read'
RANKS = (0, 1, 2, 3)


class Worker:
    """attn_meta_select / attn_meta_read over the reviewed qualification worker adapter (its own client identity)."""

    def __init__(self, control, run_id, epoch):
        self.control, self.run_id, self.epoch = control, run_id, epoch
        self.client_id = 'attn-meta-' + secrets.token_hex(8)
        self.seq = 0

    def call(self, worker_action, payload):
        self.seq += 1
        command = dict(schema=1, action=WORKER_ACTION, run_id=self.run_id, command_seq=self.seq, client_id=self.client_id,
                       engine_epoch=self.epoch, worker_action=worker_action, payload=payload)
        return rank_rows(self.control.worker(command), worker_action, self.epoch)


def rank_rows(reply, worker_action, epoch):
    """{rank: the worker's qualification mapping} from the adapter's reply, exact coverage; raises when it does not read."""
    if type(reply) is not dict or reply.get('action') != WORKER_ACTION or reply.get('worker_action') != worker_action:
        raise ValueError('attn-meta-reply-identity')
    if reply.get('engine_epoch') != epoch:
        raise ValueError('attn-meta-reply-epoch')
    rows = (reply.get('worker_reply') or {}).get('qualification')
    if type(rows) is not list or len(rows) != len(RANKS) or any(type(r) is not dict for r in rows):
        raise ValueError('attn-meta-reply-ranks')
    out = {}
    for row in rows:
        q = row.get('qualification')
        rank = row.get('worker_rank')
        if type(q) is not dict or q.get('worker_action') != worker_action or q.get('engine_supplied_verdict') is not False:
            raise ValueError('attn-meta-reply-row')
        out[rank] = {k: q.get(k) for k in ('before', 'after', 'counters', 'order')}
    if set(out) != set(RANKS):
        raise ValueError('attn-meta-reply-ranks')
    return out


SCHEMA = 'quiet-split-box/2'
STAGE_CONTROL = 'native_stage_local'
SPLIT_PHASE = 'split-w1-decode'
HIGH, LOW = 120000, 72000
PLAN = ((HIGH, 8), (HIGH, 6), (HIGH, 4), (HIGH, 2), (HIGH, 1), (LOW, 4), (LOW, 1))
MAX_TOKENS = 4096                 # hold + open at the fastest step (~10 ms) is ~2,800 tokens; an unaborted stream ends
                                  # by itself within ~(4096 - 1,500) x 20 ms ~ 50 s at 8 live (the fallback)
DECLARED = dict(plan=PLAN, hold_min_s=20.0, hold_max_s=30.0, decode_merge='always', lazy_recv=True, attn_meta=True,
                max_tokens=MAX_TOKENS)
PROMPT_SEED, ID_LO, ID_HI = 20261114, 1000, 30000
PRE_HOLD_TOKENS = 600             # tokens a stream may decode on FULL before its split (history stays in its band)
ABORT_BOUND_S = 5.0               # disconnect aborts: scheduler_requests 0 within this after the close
CLEANUP_TIMEOUT_S = 20            # control-call timeout of the cleanup after a SIGTERM
POLL_S = 0.5
SNAPSHOT_LIMIT = 80
POLL_GUARDS = ('route-left-split', 'route-generation-changed', 'transition-or-return-pending', 'admission-armed',
               'scheduler-requests-not-live', 'native-admitted', 'native-chunk-open', 'auto-arm-active',
               'manual-lease-lost', 'decode-merge-not-always')
SELECTION_FIELDS = ('local', 'overlap', 'defer_bind', 'lazy_recv', 'decode_merge')


class GuardError(ValueError):
    """A runner guard: the record names it in ``cause``."""


class Terminated(GuardError):
    """SIGTERM (the queue's timeout) raised in the main thread: the cleanup still runs, the record is still written."""


# SIGTERM state (install_term_handler): raised only while armed; run() disarms it before its cleanup, so a TERM that
# arrives during the restoration is counted and ignored instead of cutting it short.
TERM = dict(count=0, armed=False)


def parse_plan(text):
    """'120000x8,120000x6,...' -> ((120000, 8), ...); () when it does not parse."""
    try:
        return tuple((int(a), int(b)) for a, b in (part.split('x') for part in text.split(',')))
    except ValueError:
        return ()


def declaration_reasons(plan, hold_s, merge, max_tokens):
    """Empty when the CLI declaration is the declared quiet-SPLIT control, else each difference."""
    out = []
    if tuple(plan) != DECLARED['plan']:
        out.append('plan-not-declared')
    if type(hold_s) not in (int, float) or not DECLARED['hold_min_s'] <= hold_s <= DECLARED['hold_max_s']:
        out.append('hold-not-20-30-s')
    if merge != DECLARED['decode_merge']:
        out.append('decode-merge-not-always')
    if max_tokens != DECLARED['max_tokens']:
        out.append('max-tokens-not-%d' % MAX_TOKENS)
    return out


def prompt_ids(index, length):
    """Stream ``index``'s prompt: the first ``length`` ids of its seeded HIGH-token sequence."""
    rng = random.Random(PROMPT_SEED + index)
    return [rng.randrange(ID_LO, ID_HI) for _ in range(HIGH)][:length]


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
        self.opened_ns = None
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
                    before_window=sum(1 for t in self.chunks if start is not None and t < start),
                    window_chunks=len(inside), window_gap_ms_max=max(gaps) if gaps else None,
                    window_gap_ms_p50=sorted(gaps)[len(gaps) // 2] if gaps else None,
                    first_token_s=None if self.first_ns is None or self.opened_ns is None
                    else (self.first_ns - self.opened_ns) / 1e9,
                    finished_early=self.finished_ns is not None and (end is None or self.finished_ns <= end),
                    error=self.error)


def request_body(model, prompt_tokens, index, *, stream=True, max_tokens=MAX_TOKENS):
    return dict(model=model, prompt=prompt_ids(index, prompt_tokens), max_tokens=max_tokens, temperature=0,
                ignore_eos=True, stream=stream)


def _headers():
    headers = {'Content-Type': 'application/json'}
    key = os.environ.get('VLLM_API_KEY')
    if key:
        headers['Authorization'] = 'Bearer ' + key
    return headers


def stream_reader(url, model, prompt_tokens, stream, now_ns, timeout_s):
    try:
        parts = urllib.parse.urlsplit(url)
        # http.client, not urllib: close_streams shuts this socket down, which ends a read blocked between chunks
        # (a paused engine sends nothing); closing a urllib response from another thread does not (test_quiet_split_box).
        conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout_s)
        try:
            conn.connect()
            stream.sock = conn.sock
            if stream.stop.is_set():
                return
            stream.opened_ns = now_ns()
            conn.request('POST', parts.path.rstrip('/') + '/v1/completions',
                         body=json.dumps(request_body(model, prompt_tokens, stream.index)).encode(), headers=_headers())
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


def start_stream(url, model, prompt_tokens, index, now_ns, timeout_s):
    stream = Stream(index)
    stream.thread = threading.Thread(target=stream_reader, args=(url, model, prompt_tokens, stream, now_ns, timeout_s),
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


def warm_one(url, model, index, timeout_s):
    """One non-streaming completion of stream ``index``'s HIGH prompt, max_tokens 1: its prefill cached."""
    parts = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout_s)
    t0 = time.monotonic()
    try:
        conn.request('POST', parts.path.rstrip('/') + '/v1/completions',
                     body=json.dumps(request_body(model, HIGH, index, stream=False, max_tokens=1)).encode(),
                     headers=_headers())
        reply = conn.getresponse()
        data = reply.read()
        out = dict(index=index, status=reply.status, seconds=round(time.monotonic() - t0, 3))
        if reply.status == 200:
            usage = (json.loads(data) or {}).get('usage') or {}
            out.update(prompt_tokens=usage.get('prompt_tokens'),
                       cached_tokens=(usage.get('prompt_tokens_details') or {}).get('cached_tokens'))
        return out
    finally:
        conn.close()


def warm(url, model, indices, timeout_s, warmer=None):
    """Every prompt warmed concurrently; GuardError unless each returned 200 with its prompt's length."""
    warmer = warmer or warm_one
    rows, threads = {}, []

    def one(i):
        try:
            rows[i] = warmer(url, model, i, timeout_s)
        except Exception as exc:  # noqa: BLE001 - type only
            rows[i] = dict(index=i, error=type(exc).__name__)
    for i in indices:
        th = threading.Thread(target=one, args=(i,), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join(timeout_s + 5)
    out = [rows.get(i, dict(index=i, error='warm-timeout')) for i in indices]
    if any(r.get('status') != 200 or r.get('prompt_tokens') != HIGH for r in out):
        raise GuardError('warm-failed')
    return out


def select(client, fields):
    """One stage-local selection while paused idle FULL: pause, the command, resume (resume always tried)."""
    client.pause()
    try:
        return client.call(STAGE_CONTROL, **fields)
    finally:
        client.resume()


def select_with_attn(client, worker, fields, attn):
    """One pause while idle FULL: the stage-local command, then attn_meta_select {on: attn}; resume always tried."""
    client.pause()
    try:
        reply = client.call(STAGE_CONTROL, **fields)
        rows = worker.call(SELECT, {'on': attn})
        return reply, rows
    finally:
        client.resume()


def attn_read(client, worker):
    client.pause()
    try:
        return worker.call(READ, {})
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


def drain(client, a, clock, sleep):
    """After the close: ('disconnect' | 'max_tokens', seconds). The disconnect check, then the defined fallback."""
    t0 = clock()
    while True:
        status = client.call('status')
        if status.get('scheduler_requests') == 0:
            spent = clock() - t0
            return ('disconnect' if spent <= ABORT_BOUND_S else 'max_tokens'), round(spent, 3)
        if clock() - t0 >= a.drain_timeout_s:
            raise GuardError('requests-not-drained')
        sleep(POLL_S)


def run_stage(client, a, prompt_tokens, live, record, *, starter, clock, sleep, now_ns, admitted):
    row = dict(live=live, prompt_tokens=prompt_tokens, reasons=[], snapshots=[], polls=0)
    record['stages'].append(row)
    streams = []
    try:
        for i in range(live):
            streams.append(starter(a.url, a.model, prompt_tokens, i, now_ns, a.stream_timeout_s))
        deadline = clock() + a.open_timeout_s
        for s in streams:
            s.opened.wait(max(0.0, deadline - clock()))
        if any(s.first_ns is None or s.error is not None or s.finished_ns is not None for s in streams):
            raise GuardError('decoders-not-live-on-full')
        before = client.call('status')
        if before.get('route') != 'full' or before.get('scheduler_requests') != live:
            raise GuardError('full-cohort-not-%d' % live)
        row['pre_hold_tokens_max'] = max(len(s.chunks) for s in streams)
        if row['pre_hold_tokens_max'] > PRE_HOLD_TOKENS:
            raise GuardError('history-off-band')
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
        row['history_tokens'] = [prompt_tokens + min(v['before_window'] for v in views),
                                 prompt_tokens + max(v['chunks'] for v in views)]
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
    row['abort'], row['drain_s'] = drain(client, a, clock, sleep)
    if row['open_after_close']:
        raise GuardError('stream-reader-not-closed')
    return row


def engaged(select_rows, end_rows):
    """Per rank: the batched counter's growth from the select's after to the end read (None when unread)."""
    out = {}
    for rank in RANKS:
        s = ((select_rows or {}).get(rank) or {}).get('after') or {}
        e = ((end_rows or {}).get(rank) or {}).get('counters') or {}
        b0, b1 = s.get('batched'), e.get('batched')
        out[str(rank)] = b1 - b0 if type(b0) is int and type(b1) is int else None
    return out


def lease_check(client):
    """After the FULL restoration: the split lease released (control_route None) on a FULL route; never raised."""
    try:
        status = client.call('status')
    except Exception as exc:  # noqa: BLE001
        return dict(split_lease_released=None, error=type(exc).__name__)
    return dict(split_lease_released=status.get('route') == 'full' and status.get('control_route') is None,
                route=status.get('route'), control_route=status.get('control_route'))


def run(a, *, client=None, starter=None, warmer=None, clock=time.monotonic, sleep=time.sleep,
        now_ns=time.monotonic_ns):
    record = dict(schema=SCHEMA, verdict='INCONCLUSIVE', failure_class='HARNESS',
                  declared=dict(DECLARED, plan=[list(x) for x in PLAN]),
                  requested=dict(plan=[list(x) for x in a.plan], hold_s=a.hold_s, decode_merge=a.decode_merge,
                                 max_tokens=a.max_tokens),
                  prompt=dict(seed=PROMPT_SEED, id_range=[ID_LO, ID_HI], high=HIGH, low=LOW, low_is_prefix_of_high=True),
                  no_native_work=True, stages=[], basis=__doc__.split('\n', 3)[1])
    declared = declaration_reasons(a.plan, a.hold_s, a.decode_merge, a.max_tokens)
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
    selected = attn_previous = select_rows = None
    worker = Worker(client.control, a.run_id, a.epoch)
    try:
        status = client.call('status')
        if (status.get('route') != 'full' or status.get('scheduler_requests') != 0
                or status.get('admission_intent') is not None or (status.get('auto_arm') or {}).get('active') is not False):
            raise GuardError('entry-not-idle-full')
        admitted = (status.get('native_admission') or {}).get('admitted')
        if type(admitted) is not int:
            raise GuardError('native-admitted-unavailable')
        record['native_admitted_before'] = admitted
        reply, select_rows = select_with_attn(
            client, worker, dict(local=True, overlap=True, defer_bind=True, lazy_recv=DECLARED['lazy_recv'],
                                 decode_merge=DECLARED['decode_merge']), DECLARED['attn_meta'])
        selected = {k: reply.get(k) for k in SELECTION_FIELDS}
        selected.update({'previous_' + k: reply.get('previous' if k == 'local' else 'previous_' + k)
                         for k in SELECTION_FIELDS})
        record['selection'] = selected
        befores = {((r or {}).get('before') or {}).get('selected') for r in select_rows.values()}
        attn_previous = befores.pop() if len(befores) == 1 else None
        record['attn_meta'] = dict(selected=all(((r or {}).get('after') or {}).get('selected') is True
                                                for r in select_rows.values()),
                                   order_ok=all(((r or {}).get('order') or {}).get('ok') is True
                                                for r in select_rows.values()),
                                   previous=attn_previous)
        if (reply.get('local') is not True or reply.get('overlap') is not True or reply.get('defer_bind') is not True
                or reply.get('lazy_recv') is not DECLARED['lazy_recv']
                or reply.get('decode_merge') != DECLARED['decode_merge']):
            raise GuardError('selection-not-acknowledged')
        if record['attn_meta']['selected'] is not True:
            raise GuardError('attn-meta-selection-unproved')
        if record['attn_meta']['order_ok'] is not True:
            raise GuardError('attn-meta-order-not-ok')
        record['warm'] = warm(a.url, a.model, range(max(n for _, n in PLAN)), a.warm_timeout_s, warmer)
        for prompt_tokens, live in DECLARED['plan']:
            run_stage(client, a, prompt_tokens, live, record, starter=starter, clock=clock, sleep=sleep, now_ns=now_ns,
                      admitted=admitted)
        status = client.call('status')
        record['native_admitted_after'] = (status.get('native_admission') or {}).get('admitted')
        if record['native_admitted_after'] != admitted:
            raise GuardError('native-admitted')
        end_rows = attn_read(client, worker)
        record['attn_meta']['batched_delta'] = engaged(select_rows, end_rows)
        record['attn_meta']['selected_end'] = all(((r or {}).get('counters') or {}).get('selected') is True
                                                  for r in end_rows.values())
        if any(v is None or v <= 0 for v in record['attn_meta']['batched_delta'].values()):
            raise GuardError('attn-meta-not-engaged')
        record.update(verdict='PASS', failure_class=None)
    except Exception as exc:  # noqa: BLE001 - recorded; the verdict stays INCONCLUSIVE
        record['cause'] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
    finally:
        TERM['armed'] = False
        terminated = isinstance(record.get('cause'), str) and record['cause'].startswith('terminated:')
        if terminated:
            client.timeout_s = min(getattr(client, 'timeout_s', CLEANUP_TIMEOUT_S), CLEANUP_TIMEOUT_S)
        if record['verdict'] != 'PASS':
            record['cleanup'] = restore_full(client)
            record['cleanup'].update(lease_check(client))
        if terminated:
            record['selection_restore'] = dict(attempted=False, reason='terminated: the engine is stopped next')
        elif record['verdict'] == 'PASS' or record['cleanup'].get('full_restored') is True:
            record['selection_restore'] = restore_selection(client, worker, selected, attn_previous)
        else:
            # select() resumes; never on a route the restoration did not prove FULL.
            record['selection_restore'] = dict(attempted=False, reason='FULL not restored')
    return record


def restore_selection(client, worker, selected, attn_previous):
    """Put the previous selection's five fields and the previous attn-meta selection back (best effort, recorded)."""
    if not selected:
        return dict(attempted=False, reason='nothing selected')
    previous = {k: selected.get('previous_' + k) for k in SELECTION_FIELDS}
    if any(v is None for v in previous.values()) or type(attn_previous) is not bool:
        return dict(attempted=False, reason='previous selection not reported', previous=previous,
                    attn_previous=attn_previous)
    out = dict(attempted=True, previous=previous, attn_previous=attn_previous)
    try:
        reply, rows = select_with_attn(client, worker, previous, attn_previous)
        out['restored'] = (all(reply.get(k) == previous[k] for k in SELECTION_FIELDS)
                           and all(((r or {}).get('after') or {}).get('selected') is attn_previous for r in rows.values()))
    except Exception as exc:  # noqa: BLE001
        out.update(restored=False, error=str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
    return out


def install_term_handler():
    """SIGTERM raises Terminated in the main thread while armed (the cleanup then runs inside run()); a signal while
    disarmed (run's cleanup, the record write) or a second one is counted and ignored."""
    TERM.update(count=0, armed=True)

    def handler(signum, frame):
        TERM['count'] += 1
        if TERM['armed'] and TERM['count'] == 1:
            raise Terminated('terminated:SIGTERM')
    signal.signal(signal.SIGTERM, handler)
    return TERM


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n', 2)[1])
    for name in ('url', 'model', 'run-id'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--plan', default=','.join('%dx%d' % x for x in PLAN))
    p.add_argument('--hold-s', type=float, default=25.0)
    p.add_argument('--decode-merge', default='always')
    p.add_argument('--max-tokens', type=int, default=MAX_TOKENS)
    p.add_argument('--control-timeout-s', type=float, default=60)
    p.add_argument('--open-timeout-s', type=float, default=60)
    p.add_argument('--warm-timeout-s', type=float, default=420)
    p.add_argument('--drain-timeout-s', type=float, default=150)
    p.add_argument('--stream-timeout-s', type=float, default=180)
    a = p.parse_args(argv)
    a.plan = parse_plan(a.plan)
    if a.out.exists():
        return 2
    term = install_term_handler()
    record = run(a)
    record['sigterm_count'] = term['count']
    with a.out.open('x') as f:
        json.dump(record, f, sort_keys=True, indent=2)
        f.write('\n')
    print('QUIET_SPLIT ' + record['verdict'] + ('' if record['verdict'] == 'PASS' else ' ' + record.get('cause', '')))
    return 0 if record['verdict'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
