#!/usr/bin/env python3
"""Native41 box arms: plain W2 traffic with the manual lease held (stock) or released (auto).

Both arms send the frozen q18a W2 requests through the same plain streaming
clients held-b used: eight standing decoders, then the fresh prompt and its
append. No prepare, admission, prefill_begin or other native command is sent
around the measured prefill. The stock arm keeps the manual lease (auto-arm
suppressed). The auto arm pauses, releases and resumes once while the standing
cohort is open, before its decode-only window. The engine policy alone then
decides whether the fresh prompt runs natively.

Proof is receipts only: a second loopback client reads status (and nothing
else) through the fresh prefill in both arms. The auto cycle is proved from
those status records: a split lane armed by the policy (intent origin
auto-arm, its own selection), the policy's native chunks bound to the selected
request up to this prompt's last native row, no manual lease at any sample,
one closed cycle back on FULL, and the return bounded in ms on the engine
clock. Speed is the client basis for both arms on the same engine: decode
tok/s of the standing streams from fresh send to its first token, prompt
tokens / TTFT, each over the arm's own pre-prefill decode-only window. The
engine native-window basis is not available for auto cycles. Receipts hold
counts, hashes and identities, never prompts or auth.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import e3_decode_correctness as streams
import e3_overlap_performance as shared
from e3_native_qualification_driver import exception_details
from e3_native_runtime_client import NativeRuntimeClient
from native_overlap_metrics import throughput

ARM_SCHEMA = 'native41-autoarm-arm/2'
GRADE_SCHEMA = 'native41-autoarm-grade/2'
PROOF_SCHEMA = 'native41-auto-receipts/1'
KINDS = ('stock', 'auto')
STANDING = 'native41-standing'
SETTLE_S = 5.0
CLOSE_TIMEOUT_S = 60.0
POLL_S = 0.25
# Once the plan's shrinking tail stages are in flight (a native chunk at most
# half the largest one seen; alignment trims full chunks by tens of rows only),
# the poller tightens so the FULL return is bounded
# by milliseconds of sample spacing. Both arms run the same poller; the stock
# arm never has native work, so it stays at POLL_S.
POLL_FAST_S = 0.02
MAX_SAMPLES = 2048
# Generation requests one arm sends: eight standing decoders, fresh, append.
ARM_GENERATIONS = 10
ARM_ACTIONS = {'stock': frozenset(('status',)),
               'auto': frozenset(('status', 'pause', 'release_auto_arm', 'resume'))}
NO_ENGINE_BASIS = 'engine native-window basis not available for auto cycles (no window action on this engine)'


def compact(status):
    """The status fields the receipt proof reads (engine records, never prompts)."""
    if type(status) is not dict:
        return None
    auto = status.get('auto_arm') if type(status.get('auto_arm')) is dict else None
    intent = status.get('admission_intent') if type(status.get('admission_intent')) is dict else None
    admission = status.get('native_admission') if type(status.get('native_admission')) is dict else {}
    work = status.get('native_work_identity') if type(status.get('native_work_identity')) is dict else None
    return {'route': status.get('route'), 'route_generation': status.get('route_generation'),
            'engine_epoch': status.get('engine_epoch'), 'paused': status.get('paused'),
            'scheduler_requests': status.get('scheduler_requests'),
            'observed_clock_ns': status.get('observed_clock_ns'),
            'auto_arm': None if auto is None else {k: auto.get(k) for k in (
                'phase', 'active', 'manual_owned', 'selection_seq', 'completed', 'request_uid')},
            'admission_intent': None if intent is None else {k: intent.get(k) for k in (
                'origin', 'phase', 'route', 'route_generation', 'selection_seq', 'request_uid', 'seq')},
            'admitted': admission.get('admitted'), 'refused': admission.get('refused'),
            'native_work': None if work is None else {k: work.get(k) for k in (
                'request_uid_sha256', 'request_generation', 'transaction', 'chunk_start', 'chunk_end')}}


def stream_ok(state):
    """Copied from native35's performance client (_stream_ok); that module is not imported here."""
    return (state.first_advance_ns is not None and state.finish_at_ns is not None
            and state.error is None and state.malformed is None and state.done_marker is True)


def first_advance_read_timeout(cfg):
    """Copied from native35's performance client: only the declared native28 client wait."""
    frozen = cfg['timeouts']['read_s']
    override = os.environ.get('Q18A_FIRST_ADVANCE_READ_TIMEOUT_S')
    if override is None:
        return frozen
    if frozen != 180.0 or override != '1200':
        raise ValueError('first-advance-read-timeout-declaration-invalid')
    return 1200.0


def status_view(label, status):
    return dict(compact(status) or {}, label=label)


class Poller:
    """Status-only reader on its own loopback client, through the fresh prefill of either arm.

    Keeps every sample whose proof fields changed (bounded), the count of all
    samples and its own action log. Polling spacing only: the proof never reads
    the fast switch.
    """

    def __init__(self, client, *, clock=time.monotonic_ns):
        self.client, self.clock = client, clock
        self.widest = 0
        self.samples, self.actions, self.count, self.errors = [], [], 0, []
        self.stop, self.closed = threading.Event(), threading.Event()
        self.baseline, self.thread, self.fast_after_ns = None, None, None
        self._last = None

    def start(self, baseline):
        self.baseline = baseline
        self.thread = threading.Thread(target=self.run, name='native41-status-poller', daemon=True)
        self.thread.start()

    def keep(self, view):
        key = json.dumps({k: v for k, v in view.items() if k not in ('observed_clock_ns', 'client_ns')},
                         sort_keys=True, default=str)
        if key == self._last:
            if self.samples:
                self.samples[-1]['last_observed_clock_ns'] = view.get('observed_clock_ns')
                self.samples[-1]['repeats'] = self.samples[-1].get('repeats', 0) + 1
            return
        self._last = key
        if len(self.samples) < MAX_SAMPLES:
            self.samples.append(view)
        else:
            self.errors.append('samples-truncated')

    def run(self):
        fast = False
        while not self.stop.is_set():
            try:
                self.actions.append('status')
                view = compact(self.client.call('status'))
                view['client_ns'] = self.clock()
            except Exception as exc:  # noqa: BLE001 - recorded; the proof reads it
                self.errors.append(type(exc).__name__ + ':' + str(exc)[:120])
                self.stop.wait(POLL_S)
                continue
            self.count += 1
            self.keep(view)
            was_fast = fast
            work = view.get('native_work') or {}
            if type(work.get('chunk_start')) is int and type(work.get('chunk_end')) is int:
                rows = work['chunk_end'] - work['chunk_start']
                fast = fast or 0 < 2 * rows <= self.widest
                self.widest = max(self.widest, rows)
            if fast and not was_fast:
                self.fast_after_ns = view.get('observed_clock_ns')
            auto = view.get('auto_arm') or {}
            base = (self.baseline or {}).get('auto_arm') or {}
            if (type(auto.get('completed')) is int and type(base.get('completed')) is int
                    and auto['completed'] > base['completed'] and auto.get('active') is False):
                self.closed.set()
            self.stop.wait(POLL_FAST_S if fast and not self.closed.is_set() else POLL_S)

    def finish(self, timeout_s):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout_s)
        return {'samples_read': self.count, 'samples_kept': len(self.samples), 'samples': self.samples,
                'actions': sorted(set(self.actions)), 'errors': self.errors[:16],
                'alive_after_join': bool(self.thread and self.thread.is_alive()),
                'fast_after_ns': self.fast_after_ns,
                'poll_s': POLL_S, 'poll_fast_s': POLL_FAST_S}


def receipt_proof(kind, before, polled, after, prompt_tokens, actions):
    """The auto cycle (or its absence for stock) from status receipts alone."""
    reasons = []
    b, z = compact(before) or {}, compact(after) or {}
    samples = list(polled.get('samples') or [])
    rows = [b] + samples + [z]
    b_auto, z_auto = b.get('auto_arm') or {}, z.get('auto_arm') or {}
    final_row = prompt_tokens - 1 if type(prompt_tokens) is int else None
    g0, s0, c0 = b.get('route_generation'), b_auto.get('selection_seq'), b_auto.get('completed')
    record = {'schema': PROOF_SCHEMA, 'kind': kind, 'final_native_row': final_row,
              'samples_read': polled.get('samples_read'), 'samples_kept': polled.get('samples_kept'),
              'control_actions': sorted(set(actions)), 'poller_errors': polled.get('errors') or []}
    if polled.get('errors'):
        reasons.append('poller-error')
    if polled.get('alive_after_join'):
        reasons.append('poller-still-running')
    if not set(actions) <= ARM_ACTIONS[kind]:
        reasons.append('arm-sent-a-non-receipt-action')
    if any(type(r.get('engine_epoch')) is not int or r.get('engine_epoch') != b.get('engine_epoch') for r in rows):
        reasons.append('engine-epoch-changed')
    if type(g0) is not int or type(s0) is not int or type(c0) is not int:
        reasons.append('before-status-incomplete')
        return dict(record, reasons=sorted(set(reasons)), complete=False)
    admitted = (z.get('admitted') - b.get('admitted') if type(z.get('admitted')) is int
                and type(b.get('admitted')) is int else None)
    record['admitted_delta'] = admitted
    if kind == 'stock':
        if any((r.get('auto_arm') or {}).get('manual_owned') is not True for r in rows):
            reasons.append('stock-lease-not-held-at-every-sample')
        if (any(r.get('route') != 'full' or r.get('route_generation') != g0 for r in rows)
                or z_auto.get('selection_seq') != s0 or z_auto.get('completed') != c0 or admitted != 0):
            reasons.append('stock-arm-auto-selection-or-native-work-observed')
        return dict(record, reasons=sorted(set(reasons)), complete=not reasons)
    if any((r.get('auto_arm') or {}).get('manual_owned') is not False for r in rows):
        reasons.append('manual-lease-at-some-sample')
    if b.get('route') != 'full' or b_auto.get('active') is not False:
        reasons.append('before-not-released-idle-full')
    engaged = [r for r in samples if r.get('route') == 'split'
               and (r.get('admission_intent') or {}).get('origin') == 'auto-arm']
    intent = (engaged[0].get('admission_intent') or {}) if engaged else {}
    uid = intent.get('request_uid')
    record['engaged'] = None if not engaged else {
        'route_generation': engaged[0].get('route_generation'), 'phase': intent.get('phase'),
        'selection_seq': intent.get('selection_seq'), 'observed_clock_ns': engaged[0].get('observed_clock_ns'),
        'request_uid_sha256': hashlib.sha256(uid.encode()).hexdigest() if type(uid) is str else None}
    if not engaged:
        reasons.append('auto-lane-engagement-not-observed')
    elif (intent.get('phase') != 'auto-prefill' or intent.get('selection_seq') != s0 + 1
          or intent.get('route_generation') != g0 + 1 or engaged[0].get('route_generation') != g0 + 1
          or type(uid) is not str or (engaged[0].get('auto_arm') or {}).get('request_uid') != uid
          or any((r.get('admission_intent') or {}).get('request_uid') not in (None, uid) for r in samples)):
        reasons.append('auto-intent-not-this-policy-selection')
    digest = hashlib.sha256(uid.encode()).hexdigest() if type(uid) is str else None
    work = [r for r in samples if r.get('native_work')]
    ends = [r['native_work'].get('chunk_end') for r in work]
    record['native_chunks_observed'] = len({r['native_work'].get('transaction') for r in work})
    record['max_native_row_observed'] = max([e for e in ends if type(e) is int], default=None)
    if not work:
        reasons.append('native-chunk-not-observed')
    elif any(r['native_work'].get('request_uid_sha256') != digest for r in work):
        reasons.append('native-chunk-of-another-request')
    elif final_row is None or record['max_native_row_observed'] != final_row:
        reasons.append('native-rows-not-this-prompt')
    if type(admitted) is not int or admitted < 1:
        reasons.append('no-native-admission-counted')
    if (z.get('route') != 'full' or z.get('route_generation') != g0 + 2 or z_auto.get('active') is not False
            or z_auto.get('selection_seq') != s0 + 1 or z_auto.get('completed') != c0 + 1
            or z.get('admission_intent') is not None):
        reasons.append('auto-cycle-not-closed-once-on-full')
    # Return: last sample with the prompt's native work open -> first FULL sample
    # at the cycle's return generation (engine clock, sample spacing included).
    open_at = [r.get('last_observed_clock_ns') or r.get('observed_clock_ns') for r in work]
    full = [r for r in samples + [z] if r.get('route') == 'full' and r.get('route_generation') == g0 + 2]
    if work and full and all(type(n) is int for n in open_at) and type(full[0].get('observed_clock_ns')) is int:
        record['return'] = {'upper_bound_ms': (full[0]['observed_clock_ns'] - max(open_at)) / 1e6,
                            'basis': 'engine status clock: last sample with native work open -> first FULL '
                                     'sample at the return generation; includes poll spacing'}
        if record['return']['upper_bound_ms'] < 0:
            reasons.append('return-order-inverted')
    else:
        record['return'] = {'upper_bound_ms': None, 'basis': 'return samples unavailable'}
        reasons.append('return-not-bracketed')
    return dict(record, reasons=sorted(set(reasons)), complete=not reasons)


def release(client, record):
    """Pause (takes the lease), release_auto_arm, resume: the only control around the auto arm."""
    reply = client.pause()
    record['release_pause'] = {'route': reply.get('route'), 'route_generation': reply.get('route_generation')}
    try:
        got = client.call('release_auto_arm')
        record['release'] = {k: got.get(k) for k in ('released', 'engine_epoch')}
    finally:
        record['release_resume'] = client.resume().get('resumed')
    if record['release'].get('released') is not True:
        raise ValueError('auto-arm-release-unproved')


def stream_reasons(window, states, end_ns):
    reasons = []
    for row in window['streams']:
        if row.get('completion_delta', 0) < 1 or row.get('error_before_window_end'):
            reasons.append('decode-not-advancing:' + str(row.get('stream')))
    if any(s.finish_at_ns is not None and s.finish_at_ns < end_ns for s in states):
        reasons.append('standing-decoder-finished-before-window-end')
    return reasons


def request_row(state, first, prompts, family):
    return {'prompt_sha256': prompts[family + '_sha256'], 'prompt_tokens_declared': prompts[family + '_tokens'],
            'prompt_tokens_observed': state.prompt_tokens, 'request_sha256': state.request_sha256,
            'send_at_ns': state.send_at_ns, 'first_at_ns': first, 'finish_at_ns': state.finish_at_ns,
            'ttft_s': (first - state.send_at_ns) / 1e9 if first and state.send_at_ns else None,
            'output_completion': state.count(), 'error': state.error, 'malformed': state.malformed}


def geometry_ok(state):
    return (type(state.prompt_tokens) is int and state.prompt_tokens > 0
            and state.prompt_tokens == state.prompt_token_ids_len)


def arm(a, cfg, client, *, poll_client=None, sleep=time.sleep):
    kind, index = a.kind, a.index
    record = {'schema': ARM_SCHEMA, 'kind': kind, 'index': index, 'complete': False, 'reasons': [],
              'statuses': [], 'run_id': a.run_id, 'engine_epoch': client.epoch,
              'generation_requests_expected': ARM_GENERATIONS, 'manifest_sha256': cfg['manifest_sha256'],
              'native_commands_around_prefill': [], 'engine_basis': NO_ENGINE_BASIS}
    workload = shared._workload_view(cfg)
    prompts = shared._prefill_prompts(cfg)
    salt = shared._cache_salt(a.run_id, 'native41-%s-%d' % (kind, index), 0)
    record['salt_sha256'] = hashlib.sha256(salt.encode()).hexdigest()
    standing, sampler, state, poller = [], None, None, None
    mark = len(getattr(client, 'history', ()))
    record['wall_start_s'] = time.time()
    try:
        package = shared._package_gate(os.environ)
        if package.get('ok') is not True:
            raise ValueError('reviewed-package-readiness-not-pass')
        entry = client.call('status')
        record['statuses'].append(status_view('entry', entry))
        auto = entry.get('auto_arm')
        if type(auto) is not dict:
            raise ValueError('auto-arm-policy-not-installed')
        if (auto.get('manual_owned') is not True or auto.get('active') is not False
                or entry.get('route') != 'full' or entry.get('scheduler_requests') != 0):
            raise ValueError('arm-entry-not-latched-idle-full')
        cohort = streams.build_cohort(workload)
        bodies = [shared._request(a.model, row['prompt'], workload, shared._cache_salt(a.run_id, STANDING, i))
                  for i, row in enumerate(cohort)]
        standing = streams._start_streams(a.url, a.model, bodies, cohort, workload)
        opened = streams._wait_open(standing, workload)
        record['open'] = {k: opened.get(k) for k in ('complete', 'reasons')} | {
            'open_s': [round((s.first_advance_ns - s.send_at_ns) / 1e9, 3) if s.first_advance_ns and s.send_at_ns
                       else None for s in standing]}
        if opened.get('complete') is not True:
            raise ValueError('w2-standing-cohort-unopened')
        if kind == 'auto':
            release(client, record)
        sampler = shared.CountSampler(standing)
        sampler.start()
        sleep(SETTLE_S)
        b0 = sampler.snapshot()
        sleep(cfg['window_s'])
        b1 = sampler.snapshot()
        record['decode_only'] = shared._stream_window(standing, sampler, b0, b1)
        # Standing-stream problems are notes: the client basis is descriptive.
        record['notes'] = ['decode-only:' + r for r in stream_reasons(record['decode_only'], standing, b1)]
        before = client.call('status')
        record['statuses'].append(status_view('before-fresh', before))
        before_auto = before.get('auto_arm') or {}
        if before_auto.get('manual_owned') is not (kind == 'stock') or before_auto.get('active') is not False:
            raise ValueError('arm-lease-state-before-fresh-differs')
        spec = cfg['prefill']['fresh']
        poller = Poller(poll_client)
        poller.start(compact(before))
        sampler.snapshot()
        state, fresh_workload = shared._send_prefill(a.url, a.model, cfg, 'native41-' + kind, 0, prompts['fresh'],
                                                     spec['generation_budget'], spec['seed'], salt)
        first = shared._wait_first_advance(state, cfg['timeouts']['read_s'])
        record['fresh_finish'] = shared._finish_or_cancel([state], fresh_workload)
        sampler.snapshot()
        if kind == 'auto':
            poller.closed.wait(CLOSE_TIMEOUT_S)
        polled = poller.finish(CLOSE_TIMEOUT_S)
        poller = None
        after = client.call('status')
        record['statuses'].append(status_view('after-fresh', after))
        record['fresh'] = request_row(state, first, prompts, 'fresh')
        actions = [h.get('action') for h in list(getattr(client, 'history', ()))[mark:]] + polled['actions']
        record['receipts'] = receipt_proof(kind, before, polled, after, state.prompt_tokens, actions)
        record['polled'] = polled
        if not stream_ok(state) or record['fresh_finish'].get('timed_out'):
            raise ValueError('fresh-client-completion-unproved')
        if not geometry_ok(state):
            raise ValueError('fresh-prompt-token-geometry-unproven')
        ttft = record['fresh']['ttft_s']
        record['client_basis'] = {
            'prefill_tokens_per_s': throughput(state.prompt_tokens, ttft),
            'basis': 'observed fresh prompt tokens / client TTFT (held-b serving basis)',
            'decode_window': shared._stream_window(standing, sampler, state.send_at_ns, first),
            'decode_basis': 'standing streams, fresh send -> fresh first token'}
        record['notes'] += ['client-window:' + r for r in
                            stream_reasons(record['client_basis']['decode_window'], standing, first)]
        # The frozen W2 budget (8,192 per standing stream) can run out inside a long
        # prefill span; this sub-window ends at the first standing finish, so all
        # eight streams are open throughout (descriptive, client clock).
        finishes = [s.finish_at_ns for s in standing if type(s.finish_at_ns) is int]
        cut = min([first] + finishes) if type(first) is int else None
        record['client_basis']['decode_window_all_open'] = (
            shared._stream_window(standing, sampler, state.send_at_ns, cut)
            if type(cut) is int and type(state.send_at_ns) is int and cut > state.send_at_ns else None)
        record['client_basis']['all_open_share_of_span'] = (
            (cut - state.send_at_ns) / (first - state.send_at_ns)
            if record['client_basis']['decode_window_all_open'] and first > state.send_at_ns else None)
        if record['receipts']['complete'] is not True:
            label = 'auto-not-proven:' if kind == 'auto' else 'stock-receipts:'
            record['reasons'] += [label + r for r in record['receipts']['reasons']]
        # The append is the same conversation; its remaining prefill is below the
        # policy minimum, so it stays stock in both arms by design.
        mark_status = client.call('status')
        record['statuses'].append(status_view('before-append', mark_status))
        spec = cfg['prefill']['append']
        app_state, app_workload = shared._send_prefill(a.url, a.model, cfg, 'native41-' + kind, 1, prompts['append'],
                                                       spec['generation_budget'], spec['seed'], salt)
        app_first = shared._wait_first_advance(app_state, cfg['timeouts']['read_s'])
        record['append_finish'] = shared._finish_or_cancel([app_state], app_workload)
        record['append'] = request_row(app_state, app_first, prompts, 'append')
        if not stream_ok(app_state) or record['append_finish'].get('timed_out'):
            record['reasons'].append('append-client-completion-unproved')
        tail = client.call('status')
        record['statuses'].append(status_view('after-append', tail))
        if ((tail.get('auto_arm') or {}).get('selection_seq')
                != (mark_status.get('auto_arm') or {}).get('selection_seq')):
            record['reasons'].append('append-was-selected')
        record['control_actions'] = sorted({h.get('action') for h in list(getattr(client, 'history', ()))[mark:]})
        if not set(record['control_actions']) <= ARM_ACTIONS[kind]:
            record['reasons'].append('arm-sent-a-non-receipt-action')
        record['complete'] = not record['reasons']
    except Exception as exc:  # noqa: BLE001 - retained, never retried
        record['error'] = exception_details(exc, getattr(client, 'control', None))
        record['reasons'].append('arm-error')
        record['complete'] = False
    finally:
        if poller is not None:
            record['polled_after_error'] = {k: v for k, v in poller.finish(5.0).items() if k != 'samples'}
        if sampler:
            sampler.close()
        if state is not None:
            streams._stop_streams({}, [state], shared._workload_view(cfg, streams=1))
        if standing:
            streams._stop_streams({}, standing, workload)
        record['wall_end_s'] = time.time()
    return record


def ratio(a, b):
    return a / b if type(a) in (int, float) and type(b) in (int, float) and b > 0 and math.isfinite(a) else None


def arm_metrics(row):
    only = (row.get('decode_only') or {}).get('aggregate_tokens_per_s')
    client = row.get('client_basis') or {}
    during = (client.get('decode_window') or {}).get('aggregate_tokens_per_s')
    all_open = (client.get('decode_window_all_open') or {}).get('aggregate_tokens_per_s')
    receipts = row.get('receipts') or {}
    return {'kind': row.get('kind'), 'index': row.get('index'), 'complete': row.get('complete'),
            'decode_only_tokens_per_s': only,
            'client': {'prefill_tokens_per_s': client.get('prefill_tokens_per_s'),
                       'decode_during_prefill_tokens_per_s': during, 'decode_share': ratio(during, only),
                       'all_open_decode_tokens_per_s': all_open, 'all_open_decode_share': ratio(all_open, only),
                       'all_open_share_of_span': client.get('all_open_share_of_span'),
                       'ttft_s': (row.get('fresh') or {}).get('ttft_s')},
            'engine': None,
            'per_stream_decode_tokens_per_s': [r.get('rate_tokens_per_s') for r in
                                               (client.get('decode_window') or {}).get('streams') or []],
            'per_stream_all_open_decode_tokens_per_s': [r.get('rate_tokens_per_s') for r in
                                                        (client.get('decode_window_all_open') or {}).get('streams') or []],
            'per_stream_decode_only_tokens_per_s': [r.get('rate_tokens_per_s') for r in
                                                    (row.get('decode_only') or {}).get('streams') or []],
            'auto_receipts_complete': receipts.get('complete') if row.get('kind') == 'auto' else None,
            'return_upper_bound_ms': (receipts.get('return') or {}).get('upper_bound_ms'),
            'native_chunks_observed': receipts.get('native_chunks_observed'),
            'append_ttft_s': (row.get('append') or {}).get('ttft_s'), 'notes': row.get('notes') or []}


def grade(arms, ingress, checks):
    """Auto engagement is graded from receipts; client-basis speed is reported beside stock, never a gate."""
    missing, failures = [], []
    rows = [arm_metrics(r) for r in arms]
    for r in arms:
        label = '%s-%s' % (r.get('kind'), r.get('index'))
        if r.get('complete') is not True:
            missing.extend(label + ':' + x for x in (r.get('reasons') or ['incomplete']))
        if r.get('kind') == 'auto' and (r.get('receipts') or {}).get('complete') is not True:
            missing.append(label + ':auto-engagement-unproven')
    for label, data in sorted(ingress.items()):
        if (data or {}).get('verdict') == 'FAIL':
            failures.append('ingress-' + label + ':INVALID')
        elif (data or {}).get('verdict') != 'PASS':
            missing.append('ingress-' + label + ':' + str((data or {}).get('verdict')))
    for label, data in sorted(checks.items()):
        if (data or {}).get('verdict') == 'FAIL':
            failures.append(label + ':FAIL')
        elif (data or {}).get('verdict') != 'PASS':
            missing.append(label + ':' + str((data or {}).get('verdict')))
    kinds = [r.get('kind') for r in arms]
    if kinds.count('auto') < 1 or kinds.count('stock') < 1:
        missing.append('declared-arm-pair-missing')

    def median(kind, path):
        values = []
        for row in rows:
            value = row
            for key in path:
                value = (value or {}).get(key) if type(value) is dict else None
            if row['kind'] == kind and row['complete'] is True and type(value) in (int, float):
                values.append(value)
        values.sort()
        return None if not values else (values[len(values) // 2] if len(values) % 2
                                         else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2)
    summary = {kind: {'client_prefill_tokens_per_s': median(kind, ('client', 'prefill_tokens_per_s')),
                      'client_decode_during_prefill_tokens_per_s': median(
                          kind, ('client', 'decode_during_prefill_tokens_per_s')),
                      'client_decode_share': median(kind, ('client', 'decode_share')),
                      'client_all_open_decode_share': median(kind, ('client', 'all_open_decode_share')),
                      'client_all_open_share_of_span': median(kind, ('client', 'all_open_share_of_span')),
                      'decode_only_tokens_per_s': median(kind, ('decode_only_tokens_per_s',)),
                      'ttft_s': median(kind, ('client', 'ttft_s'))} for kind in KINDS}
    summary['auto']['return_upper_bound_ms'] = median('auto', ('return_upper_bound_ms',))
    summary['auto_over_stock'] = {
        'client_prefill': ratio(summary['auto']['client_prefill_tokens_per_s'], summary['stock']['client_prefill_tokens_per_s']),
        'client_decode_share': ratio(summary['auto']['client_decode_share'], summary['stock']['client_decode_share'])}
    verdict = 'FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS'
    return {'schema': GRADE_SCHEMA, 'verdict': verdict, 'failures': failures, 'missing': sorted(set(missing)),
            'auto_engaged': [bool((r.get('receipts') or {}).get('complete')) for r in arms if r.get('kind') == 'auto'],
            'arms': rows, 'summary': summary, 'speed_is_gate': False, 'target_is_gate': False,
            'all_declared_arms_retained': True, 'retry_selected': False, 'picked': False,
            'basis': 'client: standing decode tok/s fresh send -> first token over each arm\'s own decode-only '
                     'window; prefill = prompt tokens / TTFT. ' + NO_ENGINE_BASIS,
            'held_b_comparison': 'hub-side: held-b W2 client basis (stock fresh TTFT 25.3 s, 7,797 tok/s; '
                                 'native split 13.3 min) beside these arms',
            'scope': 'auto-arm engagement from plain W2 traffic, proved from status receipts; '
                     'descriptive client-basis speed beside same-queue stock arms'}


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('arm')
    r.add_argument('--url', required=True)
    r.add_argument('--model', required=True)
    r.add_argument('--run-id', required=True)
    r.add_argument('--epoch', type=int, required=True)
    r.add_argument('--kind', choices=KINDS, required=True)
    r.add_argument('--index', type=int, choices=(0, 1), required=True)
    r.add_argument('--workload', required=True)
    r.add_argument('--control-timeout-s', type=float, default=120.0)
    r.add_argument('--out', required=True)
    g = sub.add_parser('grade')
    g.add_argument('--arm', action='append', required=True)
    g.add_argument('--ingress', action='append', default=[], help='label=path')
    g.add_argument('--check', action='append', default=[], help='label=path')
    g.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if Path(a.out).exists():
        return 2
    if a.action == 'arm':
        try:
            data = json.loads(Path(a.workload).read_text())
            problems, cfg = shared.validate_manifest(data)
            if problems or cfg['repeats'] != 1:
                raise ValueError('frozen-native-workload-invalid')
            cfg = dict(cfg, timeouts=dict(cfg['timeouts'], read_s=first_advance_read_timeout(cfg)))
            client = NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
            # The receipt poller's own loopback client: status only, same run and epoch.
            poll = NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
            result = arm(a, cfg, client, poll_client=poll)
        except Exception as exc:  # noqa: BLE001
            result = {'schema': ARM_SCHEMA, 'kind': a.kind, 'index': a.index, 'complete': False,
                      'reasons': ['arm-refused'], 'error': exception_details(exc)}
        with open(a.out, 'x') as handle:
            json.dump(result, handle, sort_keys=True)
        print('NATIVE41_ARM %s-%d %s' % (a.kind, a.index, 'COMPLETE' if result.get('complete') else 'INCOMPLETE'))
        return 0 if result.get('complete') is True else 2
    arms = [load(x) or {'kind': None, 'complete': False, 'reasons': ['arm-artifact-unreadable:' + Path(x).name]}
            for x in a.arm]
    pairs = lambda rows: {x.split('=', 1)[0]: load(x.split('=', 1)[1]) for x in rows}
    result = grade(arms, pairs(a.ingress), pairs(a.check))
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    print('NATIVE41_GRADE ' + result['verdict'])
    return {'PASS': 0, 'FAIL': 1}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
