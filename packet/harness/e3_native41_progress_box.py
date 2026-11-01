#!/usr/bin/env python3
"""Held-f box arms: matched native progress timing, (current | burst) x (live | solo), auto-arm only.

Every arm is held-e's auto arm (e3_native41_autoarm_box): eight standing
plain-client decoders, then a fresh prompt that the engine policy alone runs
natively, then its stock append. Two things differ. First, the release takes
one more typed control between pause and release_auto_arm: native_progress
{mode}, the per-arm selection of the burst candidate, which the engine accepts
only when paused, idle and FULL. Second, the solo arms cancel the standing
streams as soon as the status poller sees the fresh prompt's first native
chunk. The cohort rule needs live decoders at selection and through the split
arming; every later chunk then begins with none. The live arms keep all eight
open through the fresh first token (a declared standing budget of 16,384).

Proof is receipts only. Held-e's status-receipt proof applies unchanged (the
native_progress action is accounted separately). The arm adds:

* the mode switch reply and the engine's native_progress counters before and
  after the fresh prompt;
* live arms: all eight streams open through the first token;
* solo arms: the cancel after engagement, with no reader surviving;
* no KV preemption over the fresh prompt (/metrics).

The grade joins the engine's and workers' per-chunk progress lines from the
followed engine log by transport sequence and assigns chunks to arms by wall
time and mode. It decomposes each chunk into owner cadence, send, worker FIFO,
worker host, reply head-of-line, response MQ and owner apply, and reports
medians per chunk-start band. Only chunks with all eight decoders live count
for live arms; only chunks begun with none count for solo arms. Speed is
descriptive, never a gate. Receipts hold counts, hashes, identities and host
clock stamps, never prompts or auth.
"""
import argparse
import hashlib
import json
import lzma
import math
import os
from pathlib import Path
import re
import threading
import time
import urllib.request

import e3_decode_correctness as streams
import e3_native41_autoarm_box as base
import e3_overlap_performance as shared
from e3_native_qualification_driver import exception_details
from e3_native_runtime_client import NativeRuntimeClient
from native_overlap_metrics import throughput

ARM_SCHEMA = 'native41-progress-arm/1'
GRADE_SCHEMA = 'native41-progress-grade/1'
MODES = ('current', 'burst')
COHORTS = ('live', 'solo')
# ABBA within each cohort and overall (current=A, burst=B): A B B A A B B A.
# Solo first: the first arm's cohort opens cold (~170 s of stock prefill), so
# its streams decode longest before the fresh prompt; a solo arm needs them
# only through the first native chunk.
ARM_ORDER = (('current', 'solo', 0), ('burst', 'solo', 0), ('burst', 'live', 0), ('current', 'live', 0),
             ('current', 'live', 1), ('burst', 'live', 1), ('burst', 'solo', 1), ('current', 'solo', 1))
ACTIONS = frozenset(('status', 'pause', 'native_progress', 'release_auto_arm', 'resume'))
STANDING = base.STANDING
MAX_STANDING_BUDGET = 16384
MODEL_LEN = 262144
PROGRESS_LINE = 'flash-s5 progress '
PROGRESS_SCHEMA = 's5-native-progress/1'
ROWS = {
    'n': ('seq', 'kind', 'segment', 'op_seq', 'stage', 'layer', 'origin', 't_submit', 't_sent',
          't_recv0', 't_recv1', 't_recv2', 't_recv3', 't_apply', 'outcome', 'direct'),
    's': ('seq', 'kind', 't_submit', 't_recv', 't_take'),
    'w': ('seq', 'kind', 't_start', 't_end', 'sends', 'capped'),
    'r': ('seq', 'kind', 't_enter', 't_exit', 't_emit', 'direct', 'replied'),
}
BANDS = ((0, 16384), (16384, 32768), (32768, 49152), (49152, 65536), (65536, 97152))
LOG_LAG_S = 2.0      # docker stamps a line when it reads it: late, never early
COMPONENTS = ('cadence', 'send', 'fifo', 'host', 'hol', 'mq', 'apply')


def label(mode, cohort, index):
    return '%s-%s-%d' % (mode, cohort, index)


def progress_view(status):
    p = status.get('native_progress') if type(status) is dict else None
    if type(p) is not dict:
        return None
    return {k: p.get(k) for k in ('schema', 'installed', 'mode', 'recording', 'chunks', 'commands',
                                  'direct_commands', 'stock_rpcs', 'waits', 'wait_sends', 'wait_capped',
                                  'line_errors', 'mode_changes', 'trace_lines', 'trace_dropped')}


def delta(after, before, key):
    a, b = (after or {}).get(key), (before or {}).get(key)
    return a - b if type(a) is int and type(b) is int else None


def preemption_metrics(url):
    """vllm:num_preemptions_total and kv_cache_usage_perc from /metrics (no auth, loopback)."""
    try:
        text = urllib.request.urlopen(url.rstrip('/') + '/metrics', timeout=10).read().decode()
    except (OSError, ValueError) as exc:
        return {'error': type(exc).__name__}
    out = {}
    for name in ('num_preemptions_total', 'kv_cache_usage_perc'):
        nums = [float(line.split()[-1]) for line in text.splitlines()
                if re.match(r'^vllm:' + name + r'(\{| )', line)]
        out[name] = sum(nums) if nums else None
    return out


class SoloPoller(base.Poller):
    """Held-e's status-only poller; on the first sample with native work it starts
    the solo cancel of the standing streams on its own thread (once)."""

    def __init__(self, client, *, on_native=None, clock=time.monotonic_ns):
        super().__init__(client, clock=clock)
        self.on_native, self.fired = on_native, None

    def keep(self, view):
        super().keep(view)
        if self.on_native is not None and self.fired is None and view.get('native_work'):
            self.fired = {'observed_clock_ns': view.get('observed_clock_ns'), 'client_ns': view.get('client_ns'),
                          'chunk_start': view['native_work'].get('chunk_start'),
                          'chunk_end': view['native_work'].get('chunk_end')}
            threading.Thread(target=self.on_native, name='native41-solo-cancel', daemon=True).start()


def release(client, record, mode):
    """Pause, native_progress {mode}, release_auto_arm, resume: the only control around the arm."""
    reply = client.pause()
    record['release_pause'] = {'route': reply.get('route'), 'route_generation': reply.get('route_generation')}
    try:
        got = client.call('native_progress', mode=mode)
        record['mode_switch'] = {k: got.get(k) for k in ('mode', 'previous', 'installed', 'route',
                                                         'route_generation', 'engine_epoch')}
        got = client.call('release_auto_arm')
        record['release'] = {k: got.get(k) for k in ('released', 'engine_epoch')}
    finally:
        record['release_resume'] = client.resume().get('resumed')
    switch = record['mode_switch']
    if switch.get('mode') != mode or switch.get('installed') != 'burst' or switch.get('route') != 'full':
        raise ValueError('native-progress-mode-unproved')
    if record['release'].get('released') is not True:
        raise ValueError('auto-arm-release-unproved')


def standing_budget(data):
    progress = data.get('progress') if type(data) is dict else None
    budget = (progress or {}).get('standing_generation_budget')
    if (type(progress) is not dict or progress.get('schema') != 'native41-progress-workload/1'
            or type(budget) is not int or not 1 <= budget <= MAX_STANDING_BUDGET):
        raise ValueError('progress-workload-declaration-invalid')
    return budget


def progress_reasons(record, mode, cohort):
    """The arm's own receipts beside held-e's status proof."""
    reasons = []
    history = record.get('control_history') or []
    expected = ['status', 'pause', 'native_progress', 'release_auto_arm', 'resume']
    if [a for a in history if a != 'status'][:4] != expected[1:] or history[:1] != ['status']:
        reasons.append('control-sequence-not-pause-native-progress-release-resume')
    if history.count('native_progress') != 1 or not set(history) <= ACTIONS:
        reasons.append('arm-sent-a-non-receipt-action')
    views = {s.get('label'): s for s in record.get('progress') or []}
    before, after = views.get('before-fresh'), views.get('after-fresh')
    if not before or not after or before.get('installed') != 'burst':
        reasons.append('progress-status-missing')
        return reasons
    if before.get('mode') != mode or after.get('mode') != mode or before.get('recording') or after.get('recording'):
        reasons.append('progress-mode-not-this-arm')
    chunks = delta(after, before, 'chunks')
    record['progress_delta'] = {k: delta(after, before, k) for k in (
        'chunks', 'commands', 'direct_commands', 'stock_rpcs', 'waits', 'wait_sends', 'wait_capped',
        'line_errors', 'trace_lines', 'trace_dropped')}
    d = record['progress_delta']
    if type(chunks) is not int or chunks < 2:
        reasons.append('progress-chunks-not-recorded')
    if d['line_errors'] != 0 or d['trace_dropped'] != 0:
        reasons.append('progress-lines-incomplete')
    if mode == 'burst' and not (type(d['direct_commands']) is int and d['direct_commands'] > 0):
        reasons.append('burst-sent-no-direct-command')
    if mode == 'burst' and cohort == 'live' and not (type(d['wait_sends']) is int and d['wait_sends'] > 0):
        reasons.append('burst-live-sent-nothing-inside-a-wait')
    if mode == 'current' and (d['direct_commands'] != 0 or d['wait_sends'] != 0):
        reasons.append('current-used-the-candidate')
    metrics = record.get('metrics') or {}
    p0 = (metrics.get('before-fresh') or {}).get('num_preemptions_total')
    p1 = (metrics.get('after-fresh') or {}).get('num_preemptions_total')
    if type(p0) is not float or type(p1) is not float:
        reasons.append('preemption-counter-unavailable')
    elif p1 != p0:
        reasons.append('kv-preemption-during-the-fresh-prompt')
    client = record.get('client_basis') or {}
    if cohort == 'live' and client.get('all_open_share_of_span') != 1.0:
        reasons.append('live-cohort-finished-inside-the-prefill')
    if cohort == 'solo':
        cancel = record.get('solo_cancel') or {}
        if cancel.get('triggered') is not True or cancel.get('readers_alive') != []:
            reasons.append('solo-cancel-unproved')
    return reasons


def arm(a, cfg, client, *, poll_client=None, sleep=time.sleep, budget=MAX_STANDING_BUDGET):
    mode, cohort, index = a.mode, a.cohort, a.index
    name = label(mode, cohort, index)
    record = {'schema': ARM_SCHEMA, 'kind': 'auto', 'mode': mode, 'cohort': cohort, 'index': index,
              'label': name, 'complete': False, 'reasons': [], 'statuses': [], 'progress': [], 'metrics': {},
              'run_id': a.run_id, 'engine_epoch': client.epoch,
              'generation_requests_expected': base.ARM_GENERATIONS, 'manifest_sha256': cfg['manifest_sha256'],
              'standing_generation_budget': budget, 'engine_basis': base.NO_ENGINE_BASIS}
    workload = shared._workload_view(cfg, generation_budget=budget)
    prompts = shared._prefill_prompts(cfg)
    # One fresh cache-salt namespace per mode, cohort and round (never a cache-hit arm).
    salt = shared._cache_salt(a.run_id, 'native41-progress-' + name, 0)
    record['salt_sha256'] = hashlib.sha256(salt.encode()).hexdigest()
    standing, sampler, state, poller = [], None, None, None
    mark = len(getattr(client, 'history', ()))
    record['wall_start_s'] = time.time()

    def status(tag):
        reply = client.call('status')
        record['statuses'].append(base.status_view(tag, reply))
        view = progress_view(reply)
        record['progress'].append(dict(view or {}, label=tag))
        return reply
    try:
        if cfg['decode']['prompt_tokens'] + budget > MODEL_LEN:
            raise ValueError('standing-request-exceeds-model-length')
        package = shared._package_gate(os.environ)
        if package.get('ok') is not True:
            raise ValueError('reviewed-package-readiness-not-pass')
        entry = status('entry')
        auto = entry.get('auto_arm')
        if type(auto) is not dict:
            raise ValueError('auto-arm-policy-not-installed')
        if (auto.get('manual_owned') is not True or auto.get('active') is not False
                or entry.get('route') != 'full' or entry.get('scheduler_requests') != 0):
            raise ValueError('arm-entry-not-latched-idle-full')
        view = progress_view(entry)
        if not view or view.get('installed') != 'burst' or view.get('recording') is not False:
            raise ValueError('native-progress-not-installed-burst')
        rows = streams.build_cohort(workload)
        bodies = [shared._request(a.model, row['prompt'], workload, shared._cache_salt(a.run_id, STANDING, i))
                  for i, row in enumerate(rows)]
        standing = streams._start_streams(a.url, a.model, bodies, rows, workload)
        opened = streams._wait_open(standing, workload)
        record['open'] = {k: opened.get(k) for k in ('complete', 'reasons')} | {
            'open_s': [round((s.first_advance_ns - s.send_at_ns) / 1e9, 3) if s.first_advance_ns and s.send_at_ns
                       else None for s in standing]}
        if opened.get('complete') is not True:
            raise ValueError('w2-standing-cohort-unopened')
        release(client, record, mode)
        sampler = shared.CountSampler(standing)
        sampler.start()
        sleep(base.SETTLE_S)
        b0 = sampler.snapshot()
        sleep(cfg['window_s'])
        b1 = sampler.snapshot()
        record['decode_only'] = shared._stream_window(standing, sampler, b0, b1)
        record['notes'] = ['decode-only:' + r for r in base.stream_reasons(record['decode_only'], standing, b1)]
        before = status('before-fresh')
        before_auto = before.get('auto_arm') or {}
        if before_auto.get('manual_owned') is not False or before_auto.get('active') is not False:
            raise ValueError('arm-lease-state-before-fresh-differs')
        record['metrics']['before-fresh'] = preemption_metrics(a.url)
        cancel = record['solo_cancel'] = {'triggered': False}

        def solo_cancel():
            cancel.update(triggered=True, started_ns=time.monotonic_ns())
            cancel['readers_alive'] = streams._cancel_streams(standing, workload)
            cancel['ended_ns'] = time.monotonic_ns()
        spec = cfg['prefill']['fresh']
        poller = SoloPoller(poll_client, on_native=solo_cancel if cohort == 'solo' else None)
        poller.start(base.compact(before))
        sampler.snapshot()
        state, fresh_workload = shared._send_prefill(a.url, a.model, cfg, 'native41-progress-' + name, 0,
                                                     prompts['fresh'], spec['generation_budget'], spec['seed'], salt)
        first = shared._wait_first_advance(state, cfg['timeouts']['read_s'])
        record['fresh_finish'] = shared._finish_or_cancel([state], fresh_workload)
        sampler.snapshot()
        poller.closed.wait(base.CLOSE_TIMEOUT_S)
        polled = poller.finish(base.CLOSE_TIMEOUT_S)
        cancel['trigger'] = poller.fired
        poller = None
        after = status('after-fresh')
        record['metrics']['after-fresh'] = preemption_metrics(a.url)
        record['fresh'] = base.request_row(state, first, prompts, 'fresh')
        history = [h.get('action') for h in list(getattr(client, 'history', ()))[mark:]]
        record['control_history'] = history
        receipt_actions = [x for x in history + polled['actions'] if x != 'native_progress']
        record['receipts'] = base.receipt_proof('auto', before, polled, after, state.prompt_tokens, receipt_actions)
        record['polled'] = polled
        if not base.stream_ok(state) or record['fresh_finish'].get('timed_out'):
            raise ValueError('fresh-client-completion-unproved')
        if not base.geometry_ok(state):
            raise ValueError('fresh-prompt-token-geometry-unproven')
        ttft = record['fresh']['ttft_s']
        finishes = [s.finish_at_ns for s in standing if type(s.finish_at_ns) is int]
        cut = min([first] + finishes) if type(first) is int else None
        if cohort == 'solo' and type(cancel.get('started_ns')) is int:
            cut = min(cut, cancel['started_ns']) if type(cut) is int else None
        record['client_basis'] = {
            'prefill_tokens_per_s': throughput(state.prompt_tokens, ttft),
            'basis': 'observed fresh prompt tokens / client TTFT (descriptive)',
            'decode_window_all_open': (shared._stream_window(standing, sampler, state.send_at_ns, cut)
                                       if type(cut) is int and type(state.send_at_ns) is int
                                       and cut > state.send_at_ns else None),
            'all_open_share_of_span': ((cut - state.send_at_ns) / (first - state.send_at_ns)
                                       if type(cut) is int and type(first) is int
                                       and first > state.send_at_ns and cut > state.send_at_ns else None)}
        if cohort == 'live':
            record['client_basis']['decode_window'] = shared._stream_window(standing, sampler, state.send_at_ns, first)
            record['notes'] += ['client-window:' + r for r in
                                base.stream_reasons(record['client_basis']['decode_window'], standing, first)]
        if record['receipts']['complete'] is not True:
            record['reasons'] += ['auto-not-proven:' + r for r in record['receipts']['reasons']]
        record['reasons'] += progress_reasons(record, mode, cohort)
        # The append is the same conversation; its remaining prefill is below the
        # policy minimum, so it stays stock in every arm by design.
        mark_status = status('before-append')
        spec = cfg['prefill']['append']
        app_state, app_workload = shared._send_prefill(a.url, a.model, cfg, 'native41-progress-' + name, 1,
                                                       prompts['append'], spec['generation_budget'], spec['seed'],
                                                       salt)
        app_first = shared._wait_first_advance(app_state, cfg['timeouts']['read_s'])
        record['append_finish'] = shared._finish_or_cancel([app_state], app_workload)
        record['append'] = base.request_row(app_state, app_first, prompts, 'append')
        if not base.stream_ok(app_state) or record['append_finish'].get('timed_out'):
            record['reasons'].append('append-client-completion-unproved')
        tail = status('after-append')
        if ((tail.get('auto_arm') or {}).get('selection_seq')
                != (mark_status.get('auto_arm') or {}).get('selection_seq')):
            record['reasons'].append('append-was-selected')
        record['control_actions'] = sorted({h.get('action') for h in list(getattr(client, 'history', ()))[mark:]})
        if not set(record['control_actions']) <= ACTIONS:
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


# --- progress lines: parse, join, decompose -----------------------------------

def docker_stamp(text):
    """Epoch seconds of a `docker logs --timestamps` prefix (RFC 3339, ns), else None."""
    m = re.match(r'^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(?:\.(\d+))?Z ', text)
    if not m:
        return None
    import calendar
    whole = calendar.timegm(tuple(int(m.group(i)) for i in range(1, 7)) + (0, 0, 0))
    return whole + (int((m.group(7) or '0')[:9].ljust(9, '0')) / 1e9)


def read_lines(paths):
    """Progress records from the followed engine log(s): (stamp, body) per line, plus parse errors."""
    out, errors = [], []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        opener = lzma.open if path.suffix == '.xz' else open
        with opener(path, 'rt', errors='replace') as handle:
            for line in handle:
                at = line.find(PROGRESS_LINE)
                if at < 0:
                    continue
                try:
                    body = json.loads(line[at + len(PROGRESS_LINE):])
                except ValueError:
                    errors.append('unparsable-progress-line')
                    continue
                if type(body) is not dict or body.get('schema') != PROGRESS_SCHEMA:
                    errors.append('progress-line-schema')
                    continue
                out.append((docker_stamp(line), body))
    return out, errors


def assemble(records):
    """Whole flushes: engine by line number, worker by (rank, line); parts complete or listed."""
    groups, problems = {}, []
    for stamp, body in records:
        key = (body.get('side'), body.get('rank'), body.get('line'), body.get('base_ns'))
        groups.setdefault(key, []).append((stamp, body))
    flushes = []
    for key, parts in groups.items():
        parts.sort(key=lambda p: p[1].get('part', -1))
        bodies = [b for _, b in parts]
        count = bodies[0].get('parts')
        if [b.get('part') for b in bodies] != list(range(count or 0)) or any(b.get('parts') != count for b in bodies):
            problems.append('line-parts-incomplete:%s:%s:%s' % key[:3])
            continue
        head = bodies[0]
        if head.get('fields') != {k: list(v) for k, v in ROWS.items()}:
            problems.append('line-fields-differ:%s:%s:%s' % key[:3])
            continue
        rows = {}
        for b in bodies:
            for tag, items in (b.get('rows') or {}).items():
                rows.setdefault(tag, []).extend(items)
        base_ns = head['base_ns']

        def absolute(tag, item):
            return {n: (base_ns + v * 1000 if n.startswith('t_') and type(v) is int else v)
                    for n, v in zip(ROWS[tag], item)}
        flushes.append(dict(side=key[0], rank=key[1], line=key[2], stamp=min(s for s, _ in parts if s is not None)
                            if any(s is not None for s, _ in parts) else None, dropped=head.get('dropped'),
                            identity={k: v for k, v in head.items() if k not in (
                                'rows', 'fields', 'part', 'parts', 'dropped', 'schema', 'side', 'rank', 'line',
                                'base_ns')},
                            rows={tag: [absolute(tag, item) for item in items] for tag, items in rows.items()}))
    return flushes, problems


def median(values):
    values = sorted(v for v in values if type(v) in (int, float) and math.isfinite(v))
    if not values:
        return None
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def chunk_breakdown(chunk, workers):
    """One chunk's native command chain, critical (last-replying) rank per command, in seconds.

    Inside the chunk the owner runs one command chain: cadence is the time from
    one command's apply to the next submit; the other components follow each
    command from submit to apply. Overlapping commands make the residual
    (wall minus the serial sum) negative; it is reported, never hidden.
    """
    rows = sorted(chunk['rows'].get('n', []), key=lambda r: (r['t_submit'] or 0, r['seq']))
    totals = dict.fromkeys(COMPONENTS, 0.0)
    missing, last_apply, origins, direct = [], None, {}, 0
    for r in rows:
        origins[r['origin']] = origins.get(r['origin'], 0) + 1
        direct += int(r['direct'] == 1)
        if last_apply is not None and type(r['t_submit']) is int:
            totals['cadence'] += max(0, r['t_submit'] - last_apply) / 1e9
        if type(r['t_sent']) is int and type(r['t_submit']) is int:
            totals['send'] += (r['t_sent'] - r['t_submit']) / 1e9
        recv = {k: r['t_recv%d' % k] for k in range(4) if type(r['t_recv%d' % k]) is int}
        if recv:
            rank = max(recv, key=recv.get)
            w = workers.get((r['seq'], rank))
            if w is None or not all(type(w[k]) is int for k in ('t_enter', 't_exit', 't_emit')):
                missing.append(dict(seq=r['seq'], rank=rank, kind=r['kind']))
            else:
                totals['fifo'] += max(0, w['t_enter'] - r['t_sent']) / 1e9
                totals['host'] += (w['t_exit'] - w['t_enter']) / 1e9
                totals['hol'] += (w['t_emit'] - w['t_exit']) / 1e9
                totals['mq'] += max(0, recv[rank] - w['t_emit']) / 1e9
            if type(r['t_apply']) is int:
                totals['apply'] += max(0, r['t_apply'] - recv[rank]) / 1e9
        if type(r['t_apply']) is int:
            last_apply = r['t_apply']
    stock = chunk['rows'].get('s', [])
    waits = chunk['rows'].get('w', [])
    first = rows[0]['t_submit'] if rows and type(rows[0]['t_submit']) is int else None
    end = rows[-1]['t_apply'] if rows and type(rows[-1]['t_apply']) is int else None
    wall = (end - first) / 1e9 if first is not None and end is not None else None
    kinds = sorted({s['kind'] for s in stock})
    roundtrip = {k: median([(s['t_recv'] - s['t_submit']) / 1e9 for s in stock if s['kind'] == k
                            and type(s['t_recv']) is int and type(s['t_submit']) is int]) for k in kinds}
    host = {}
    for k in kinds:
        spans = []
        for s in stock:
            if s['kind'] != k:
                continue
            for rank in range(4):
                w = workers.get((s['seq'], rank))
                if w is not None and w.get('replied') == 1 and type(w['t_exit']) is int and type(w['t_enter']) is int:
                    spans.append((w['t_exit'] - w['t_enter']) / 1e9)
        host[k] = median(spans)
    return dict(commands=len(rows), wall_s=wall, components_s=totals,
                residual_s=None if wall is None else wall - sum(totals.values()),
                origins=origins, direct_commands=direct,
                stock_rpcs=len(stock), stock_by_kind={k: sum(1 for s in stock if s['kind'] == k) for k in kinds},
                stock_roundtrip_s=roundtrip, stock_replying_host_s=host,
                wait_s=sum((w['t_end'] - w['t_start']) / 1e9 for w in waits
                           if type(w['t_end']) is int and type(w['t_start']) is int),
                wait_sends=sum(w['sends'] or 0 for w in waits), waits=len(waits),
                missing_worker_rows=missing, first_submit_ns=first, last_apply_ns=end)


def chunk_label(decode_live, streams_count):
    if decode_live == streams_count:
        return 'live'
    if decode_live == 0:
        return 'solo'
    return 'transition'


def band_of(start):
    for lo, hi in BANDS:
        if lo <= start < hi:
            return '%d-%d' % (lo, hi)
    return 'other'


def progress_receipts(arms, records, parse_errors, streams_count=8):
    """Per-arm chunk breakdowns from the joined lines, with completeness reasons."""
    flushes, problems = assemble(records)
    problems = sorted(set(problems + parse_errors))
    workers = {}
    duplicate = 0
    for f in flushes:
        if f['side'] != 'worker':
            continue
        for r in f['rows'].get('r', []):
            key = (r['seq'], f['rank'])
            duplicate += int(key in workers)
            workers[key] = r
    engine = sorted((f for f in flushes if f['side'] == 'engine'), key=lambda f: f['line'])
    dropped = sum(f['dropped'] or 0 for f in flushes)
    out, owners = {}, {}
    last_line = engine[-1]['line'] if engine else None
    for arm in arms:
        name = arm.get('label')
        start, end = arm.get('wall_start_s'), arm.get('wall_end_s')
        row = dict(label=name, mode=arm.get('mode'), cohort=arm.get('cohort'), reasons=[])
        out[name] = row
        if type(start) not in (int, float) or type(end) not in (int, float):
            row['reasons'].append('arm-span-missing')
            continue
        mine = [f for f in engine if f['stamp'] is not None and start <= f['stamp'] <= end + LOG_LAG_S]
        for f in mine:
            owners.setdefault(id(f), []).append(name)
        row['chunks_logged'] = len(mine)
        expected = (arm.get('progress_delta') or {}).get('chunks')
        if not mine:
            row['reasons'].append('no-progress-line-in-the-arm-span')
            continue
        if expected != len(mine):
            row['reasons'].append('progress-lines-differ-from-the-chunk-counter')
        if any(f['identity'].get('mode') != arm.get('mode') for f in mine):
            row['reasons'].append('progress-line-mode-not-this-arm')
        if any(f['identity'].get('open_rows') != 0 for f in mine):
            row['reasons'].append('progress-line-with-open-rows')
        ordered = sorted(mine, key=lambda f: f['identity'].get('chunk_start', -1))
        starts = [f['identity'].get('chunk_start') for f in ordered]
        ends = [f['identity'].get('chunk_end') for f in ordered]
        prompt = (arm.get('fresh') or {}).get('prompt_tokens_observed')
        if (starts[:1] != [0] or type(prompt) is not int or ends[-1:] != [prompt - 1]
                or any(e != s for e, s in zip(ends, starts[1:]))):
            row['reasons'].append('native-chunks-not-row-0-through-n-minus-1')
        chunks = []
        for f in ordered:
            info = chunk_breakdown(f, workers)
            live = f['identity'].get('decode_live')
            info.update(txn=f['identity'].get('txn'), chunk_start=f['identity'].get('chunk_start'),
                        chunk_end=f['identity'].get('chunk_end'), decode_live=live, line=f['line'],
                        label=chunk_label(live, streams_count), band=band_of(f['identity'].get('chunk_start') or 0))
            chunks.append(info)
        for c, nxt in zip(chunks, chunks[1:]):
            if c['first_submit_ns'] is not None and nxt['first_submit_ns'] is not None:
                c['open_interval_s'] = (nxt['first_submit_ns'] - c['first_submit_ns']) / 1e9
            if c['last_apply_ns'] is not None and nxt['first_submit_ns'] is not None:
                nxt['gap_before_s'] = (nxt['first_submit_ns'] - c['last_apply_ns']) / 1e9
        # A worker flushes at a chunk's release begin, so each release reply row
        # rides on the next flush; only the run's very last release stays unflushed.
        allowed = [m for c in chunks if c['line'] == last_line for m in c['missing_worker_rows']
                   if m['kind'] == 'release']
        other = [m for c in chunks for m in c['missing_worker_rows'] if m not in allowed]
        row['missing_worker_rows'] = len(other)
        row['unflushed_final_release_rows'] = len(allowed)
        if other:
            row['reasons'].append('worker-rows-missing')
        want = 'live' if arm.get('cohort') == 'live' else 'solo'
        counted = [c for c in chunks if c['label'] == want]
        row['chunks'] = chunks
        row['counted_label'] = want
        row['counted_chunks'] = len(counted)
        if arm.get('cohort') == 'live' and len(counted) != len(chunks):
            row['reasons'].append('live-arm-chunk-without-all-decoders')
        if arm.get('cohort') == 'solo':
            labels = [c['label'] for c in chunks]
            first_solo = labels.index('solo') if 'solo' in labels else None
            if (first_solo is None or labels[0] != 'live'
                    or any(x != 'solo' for x in labels[first_solo:])):
                row['reasons'].append('solo-arm-not-a-live-prefix-then-decoder-free-chunks')
        row['bands'] = {}
        for band in sorted({c['band'] for c in counted}):
            rows = [c for c in counted if c['band'] == band]
            row['bands'][band] = dict(chunks=len(rows), wall_s=median([c['wall_s'] for c in rows]),
                                      open_interval_s=median([c.get('open_interval_s') for c in rows]),
                                      gap_before_s=median([c.get('gap_before_s') for c in rows]),
                                      residual_s=median([c['residual_s'] for c in rows]),
                                      components_s={k: median([c['components_s'][k] for c in rows])
                                                    for k in COMPONENTS},
                                      stock_rpcs=median([c['stock_rpcs'] for c in rows]),
                                      wait_s=median([c['wait_s'] for c in rows]),
                                      wait_sends=median([c['wait_sends'] for c in rows]),
                                      commands=median([c['commands'] for c in rows]))
        row['all_counted'] = dict(chunks=len(counted), wall_s=median([c['wall_s'] for c in counted]),
                                  open_interval_s=median([c.get('open_interval_s') for c in counted]),
                                  gap_before_s=median([c.get('gap_before_s') for c in counted]),
                                  residual_s=median([c['residual_s'] for c in counted]),
                                  components_s={k: median([c['components_s'][k] for c in counted])
                                                for k in COMPONENTS})
    # Every engine line belongs to exactly one arm: native work outside the arms
    # (or a line docker did not stamp) is not this run's declared traffic.
    return dict(flushes=len(flushes), engine_flushes=len(engine), worker_rows=len(workers),
                duplicate_worker_rows=duplicate, dropped_rows=dropped, problems=problems,
                unstamped_engine_flushes=sum(1 for f in engine if f['stamp'] is None),
                engine_flushes_outside_every_arm=sum(1 for f in engine if id(f) not in owners),
                engine_flushes_in_two_arms=sum(1 for names in owners.values() if len(names) > 1), arms=out)


def ratio(a, b):
    return a / b if type(a) in (int, float) and type(b) in (int, float) and b > 0 and math.isfinite(a) else None


def grade(arms, ingress, checks, logs):
    """Receipts and complete progress lines for all eight declared arms; speed reported, never a gate."""
    missing, failures = [], []
    declared = [label(*x) for x in ARM_ORDER]
    seen = [a.get('label') for a in arms]
    if sorted(x for x in seen if x) != sorted(declared) or len(seen) != len(declared):
        missing.append('declared-arms-missing-or-repeated')
    for r in arms:
        name = r.get('label') or 'unlabelled'
        if r.get('complete') is not True:
            missing.extend(name + ':' + x for x in (r.get('reasons') or ['incomplete']))
    for name, data in sorted(ingress.items()):
        if (data or {}).get('verdict') == 'FAIL':
            failures.append('ingress-' + name + ':INVALID')
        elif (data or {}).get('verdict') != 'PASS':
            missing.append('ingress-' + name + ':' + str((data or {}).get('verdict')))
    for name, data in sorted(checks.items()):
        if (data or {}).get('verdict') == 'FAIL':
            failures.append(name + ':FAIL')
        elif (data or {}).get('verdict') != 'PASS':
            missing.append(name + ':' + str((data or {}).get('verdict')))
    records, errors = read_lines(logs)
    receipts = progress_receipts([a for a in arms if a.get('label')], records, errors)
    if not records:
        missing.append('no-progress-lines-in-the-engine-log')
    for problem in receipts['problems']:
        missing.append('progress:' + problem)
    if receipts['dropped_rows'] or receipts['duplicate_worker_rows']:
        missing.append('progress:rows-dropped-or-duplicated')
    if receipts['unstamped_engine_flushes'] or receipts['engine_flushes_outside_every_arm']:
        missing.append('progress:engine-lines-outside-every-arm-span')
    if receipts['engine_flushes_in_two_arms']:
        missing.append('progress:engine-line-in-two-arm-spans')
    for name, row in sorted(receipts['arms'].items()):
        missing.extend('progress-%s:%s' % (name, x) for x in row['reasons'])
    summary = {}
    for mode in MODES:
        for cohort in COHORTS:
            rows = [receipts['arms'].get(label(mode, cohort, i)) or {} for i in (0, 1)]
            bands = sorted({b for r in rows for b in (r.get('bands') or {})})
            summary[mode + '-' + cohort] = {
                'chunk_wall_s': median([c['wall_s'] for r in rows for c in r.get('chunks') or []
                                        if c['label'] == r.get('counted_label')]),
                'bands': {b: {'wall_s': median([((r.get('bands') or {}).get(b) or {}).get('wall_s') for r in rows]),
                              'components_s': {k: median([(((r.get('bands') or {}).get(b) or {}).get('components_s')
                                                           or {}).get(k) for r in rows]) for k in COMPONENTS}}
                          for b in bands},
                'client_prefill_tokens_per_s': median([((a.get('client_basis') or {}).get('prefill_tokens_per_s'))
                                                       for a in arms if a.get('mode') == mode
                                                       and a.get('cohort') == cohort]),
                'client_ttft_s': median([(a.get('fresh') or {}).get('ttft_s') for a in arms
                                         if a.get('mode') == mode and a.get('cohort') == cohort])}
    for cohort in COHORTS:
        summary['burst_over_current_' + cohort] = {
            'chunk_wall': ratio(summary['burst-' + cohort]['chunk_wall_s'], summary['current-' + cohort]['chunk_wall_s']),
            'client_ttft': ratio(summary['burst-' + cohort]['client_ttft_s'], summary['current-' + cohort]['client_ttft_s'])}
    live_decode = {}
    for mode in MODES:
        values = []
        for a in arms:
            if a.get('mode') == mode and a.get('cohort') == 'live':
                only = (a.get('decode_only') or {}).get('aggregate_tokens_per_s')
                during = ((a.get('client_basis') or {}).get('decode_window') or {}).get('aggregate_tokens_per_s')
                values.append(ratio(during, only))
        live_decode[mode] = median(values)
    summary['live_decode_share_client'] = live_decode
    verdict = 'FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS'
    return {'schema': GRADE_SCHEMA, 'verdict': verdict, 'failures': failures, 'missing': sorted(set(missing)),
            'declared_arms': declared, 'arms': [{k: a.get(k) for k in (
                'label', 'mode', 'cohort', 'index', 'complete', 'reasons', 'progress_delta', 'solo_cancel',
                'fresh', 'append', 'mode_switch', 'metrics', 'open', 'notes')} for a in arms],
            'progress': receipts, 'summary': summary, 'speed_is_gate': False, 'target_is_gate': False,
            'all_declared_arms_retained': True, 'retry_selected': False, 'picked': False,
            'components': {'cadence': 'previous native apply -> this submit (owner progress cadence)',
                           'send': 'submit -> broadcast sent', 'fifo': 'sent -> critical worker handler enter',
                           'host': 'worker handler enter -> exit', 'hol': 'handler exit -> response MQ write',
                           'mq': 'MQ write -> engine receive', 'apply': 'engine receive -> owner apply'},
            'basis': 'host CLOCK_MONOTONIC stamps of the engine owner and the four workers (one container); '
                     'per chunk, the critical (last-replying) rank of each native command; live arms count '
                     'chunks begun with all 8 decoders, solo arms chunks begun with none',
            'scope': 'matched native progress timing (current vs burst, live vs solo) on one engine; '
                     'receipts gate, speed descriptive'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('arm')
    r.add_argument('--url', required=True)
    r.add_argument('--model', required=True)
    r.add_argument('--run-id', required=True)
    r.add_argument('--epoch', type=int, required=True)
    r.add_argument('--mode', choices=MODES, required=True)
    r.add_argument('--cohort', choices=COHORTS, required=True)
    r.add_argument('--index', type=int, choices=(0, 1), required=True)
    r.add_argument('--workload', required=True)
    r.add_argument('--control-timeout-s', type=float, default=120.0)
    r.add_argument('--out', required=True)
    g = sub.add_parser('grade')
    g.add_argument('--arm', action='append', required=True)
    g.add_argument('--ingress', action='append', default=[], help='label=path')
    g.add_argument('--check', action='append', default=[], help='label=path')
    g.add_argument('--log', action='append', default=[], help='followed engine log (stdout/stderr)')
    g.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if Path(a.out).exists():
        return 2
    if a.action == 'arm':
        try:
            data = json.loads(Path(a.workload).read_text())
            budget = standing_budget(data)
            problems, cfg = shared.validate_manifest(data)
            if problems or cfg['repeats'] != 1:
                raise ValueError('frozen-native-workload-invalid')
            cfg = dict(cfg, timeouts=dict(cfg['timeouts'], read_s=base.first_advance_read_timeout(cfg)))
            client = NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
            # The receipt poller's own loopback client: status only, same run and epoch.
            poll = NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
            result = arm(a, cfg, client, poll_client=poll, budget=budget)
        except Exception as exc:  # noqa: BLE001
            result = {'schema': ARM_SCHEMA, 'kind': 'auto', 'mode': a.mode, 'cohort': a.cohort, 'index': a.index,
                      'label': label(a.mode, a.cohort, a.index), 'complete': False,
                      'reasons': ['arm-refused'], 'error': exception_details(exc)}
        with open(a.out, 'x') as handle:
            json.dump(result, handle, sort_keys=True)
        print('NATIVE41_PROGRESS_ARM %s %s' % (result['label'], 'COMPLETE' if result.get('complete') else 'INCOMPLETE'))
        return 0 if result.get('complete') is True else 2
    arms = [base.load(x) or {'label': None, 'complete': False, 'reasons': ['arm-artifact-unreadable:' + Path(x).name]}
            for x in a.arm]
    pairs = lambda rows: {x.split('=', 1)[0]: base.load(x.split('=', 1)[1]) for x in rows}
    result = grade(arms, pairs(a.ingress), pairs(a.check), a.log)
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    print('NATIVE41_PROGRESS_GRADE ' + result['verdict'])
    return {'PASS': 0, 'FAIL': 1}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
