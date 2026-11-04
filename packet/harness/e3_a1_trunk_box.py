#!/usr/bin/env python3
"""q18a-a1-trunk box arms and grade: the returning-agent yield on and off, with returning agents in the live replay, over inwait's ps1 base. Discovery mode.

The arm is its own, from native41's arm (progress.arm) and A1's (claude/returning-append, e3_q18b_append_yield_box.run_arm): that arm's receipts assume one auto cycle of one request and 8 streams
open through the bulk, and a served yield breaks each of those on purpose. Same request shape otherwise: 8 standing decoders (greedy 0 and 1), burst, the decode-only window, the bulk, the stock append.
Control order: pause, native_append_yield {enabled}, resume at the idle FULL entry; then inwait's release (release_compose); status; attn_meta's end read. ACTIONS below is the list the router smoke holds.
A returning agent ends its turn and comes back: at the plan's time after the bulk is sent, one sampled standing stream is cancelled by the client, then the same prompt plus a suffix goes out
under the same salt, with a short reply. Nothing warms it, nothing adds a page; the prefix cache is what the stream left. Solo: the standing streams are cancelled at the first native chunk, no returner.

The grade is small. FAIL: an engine death or signature, a startup failure or an attn-meta leak, a yield selected with the yield off, a route generation that does not add up. INC: an arm missing or
incomplete, an on arm that served no yield, an ingress or check not PASS. Every speed and every budget line is a read (lever_read): discovery mode, numbers for the lead. Numerics are not covered.
"""
import argparse
import json
from pathlib import Path
import hashlib
import os
import statistics
import threading
import time

import a1_trunk_arms as T
import e3_inwait_box as iw
import e3_native39_w2_length as w2
import e3_native_refusal_client as refusal
import step_a_torn_lines as torn_lines

am, an, ap, base, c1, progress, sb = iw.am, iw.an, iw.ap, iw.base, iw.c1, iw.progress, iw.sb
shared, streams = progress.shared, progress.streams
exception_details = progress.exception_details
ARM_SCHEMA = 'a1-trunk-arm/1'
GRADE_SCHEMA = 'a1-trunk-grade/1'
YIELD_CONTROL = 'native_append_yield'
ACTIONS = iw.ACTIONS[:1] + ((YIELD_CONTROL, ('enabled',)),) + iw.ACTIONS[1:]    # the router smoke sends them in this order: the yield's control only before the release
ACTION_NAMES = frozenset(a for a, _f in ACTIONS)
CONTROL_FIELDS = ('enabled', 'previous', 'max_new', 'per_bulk', 'handback_steps', 'reentry_retries', 'route', 'route_generation', 'engine_epoch', 'observed_clock_ns')
PHASES = ('selected', 'drained', 'row_closed', 'admitted', 'handback', 'full', 'reenter', 'resume', 'bulk_reopened')
REFUSAL_COUNTERS = ('handback_expired', 'reentry_refused', 'append_refused')
SIGNATURES = an.sa.SIGNATURES + an.STAGE_SIGNATURES
HARD_MARKERS = ('Traceback', 'EngineDeadError', 'illegal memory access', 'E3ContractError', 'CUDA error')
LANE = 'native41-progress-'
JOIN_S = 300.0
NOT_COVERED = 'numerics: no fixed-schedule check; a yield under real progress_in_wait ticks was never exercised on CPU (composition read from the code), this box run is its first exercise'
SCOPE = 'the returning-agent yield on the trunk: selection, engagement, crash, OOM are gated; every speed, the budget lines and decode are reads; not a release assessment and not a numerics result'


class UsageInstrument:
    """correctness' per-event hook: keeps usage.prompt_tokens_details.cached_tokens (a number) and never raises."""

    def consume(self, state, event):
        usage = event.get('usage') if type(event) is dict else None
        details = usage.get('prompt_tokens_details') if type(usage) is dict else None
        cached = details.get('cached_tokens') if type(details) is dict else None
        if type(cached) is int and cached >= 0:
            state.cached_tokens = cached


def yield_view(status, history=False):
    """The engine's append_yield status row (no request UID): bounded fields, the history only when asked."""
    row = status.get('append_yield') if type(status) is dict else None
    if type(row) is not dict:
        return None
    out = {k: row.get(k) for k in ('schema', 'config', 'phase', 'seq', 'counters', 'end_reasons', 'arrivals', 'holds', 'refusals', 'lookups', 'split_holds')}
    current = row.get('current')
    out['current_seq'] = current.get('seq') if type(current) is dict else None
    if history:
        out['history'] = row.get('history') if type(row.get('history')) is list else None
    return out


def yield_control(client, enabled, record):
    """pause, native_append_yield, resume at the arm's idle FULL entry; the reply, its bounds and the clock proof (A1's)."""
    reply = client.pause()
    record['yield_pause'] = {'route': reply.get('route'), 'route_generation': reply.get('route_generation')}
    try:
        sent = time.monotonic_ns()
        got = client.call(YIELD_CONTROL, enabled=enabled)
        received = time.monotonic_ns()
    finally:
        record['yield_resume'] = client.resume().get('resumed')
    view = {k: got.get(k) for k in CONTROL_FIELDS}
    engine_ns = view.get('observed_clock_ns')
    view['clock_proof'] = dict(sent_ns=sent, received_ns=received, engine_ns=engine_ns, inside=type(engine_ns) is int and sent <= engine_ns <= received)
    view['confirmed'] = view.get('enabled') is enabled and {k: view.get(k) for k in T.LAUNCH} == T.LAUNCH and view.get('route') == 'full'
    record['yield_control'] = view
    if not view['confirmed']:
        raise ValueError('yield-control-not-confirmed')


def digest(state):
    """Length and a hash of the output, never the text."""
    tokens = list(getattr(state, 'tokens', ()))
    return dict(tokens=len(tokens), sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest())


def suffix_seed(label_, index):
    """Per arm and stream: a suffix an earlier arm's returner sent would hit its pages in the cache, and the arm would measure no new tokens."""
    return int(hashlib.sha256(('a1-trunk-suffix|%s|%d' % (label_, index)).encode()).hexdigest()[:12], 16)


def returner_text(prompt, label_, index, tokens):
    return prompt + '\n' + shared._prompt(suffix_seed(label_, index), 2, tokens)


def shaped_returner_text(a, prompt, label_, index, tokens, short, *, post=None, timeout=120):
    """The returner's text with the tail it needs, measured by the engine's own /tokenize; (text, record). Never sends a token id onward."""
    post = post or w2._post
    extra, tails = 0, []
    for _ in range(T.TAIL_ATTEMPTS):
        text = returner_text(prompt, label_, index, tokens + extra)
        got = post(a.url, '/tokenize', {'model': a.model, 'prompt': text}, timeout).get('tokens')
        if type(got) is not list:
            raise ValueError('tokenize-reply-shape')
        tail = T.tail_rows(len(got))
        tails.append(tail)
        if T.tail_fits(tail, short):
            return text, dict(prompt_tokens_tokenized=len(got), tail_rows=tail, tail_attempts=len(tails), short_tail=short)
        extra += (T.TAIL_TARGET[short] - tail) % T.PREFIX_UNIT
    raise ValueError('returner-tail-not-shaped: %s' % tails)


class Returner:
    """One returning agent, on its own thread: wait for its time, end the standing stream's turn, send the prompt plus the suffix under the same salt, read to the end."""

    def __init__(self, a, cfg, row, standing_state, salt, plan_row, label_, row_of_arm):
        self.a, self.cfg, self.prompt, self.standing, self.salt = a, cfg, row['prompt'], standing_state, salt
        self.index, self.tokens, self.at_s = plan_row
        self.label = label_
        self.short = T.keeps_short_tail(row_of_arm, self.index)
        self.text = None
        self.state = None
        self.row = dict(index=self.index, suffix_tokens=self.tokens, at_s=self.at_s, sent=False)
        self.thread = threading.Thread(target=self.run, name='a1-trunk-returner-%d' % self.index, daemon=True)

    def prepare(self):
        """Before the bulk: /tokenize doesn't reach the engine's step, and no decode window is open yet."""
        self.text, shape = shaped_returner_text(self.a, self.prompt, self.label, self.index, self.tokens, self.short)
        self.row.update(shape)

    def start(self, bulk_send_ns):
        self.bulk_send_ns = bulk_send_ns
        self.thread.start()

    def run(self):
        try:
            delay = self.bulk_send_ns + int(self.at_s * 1e9) - time.monotonic_ns()
            if delay > 0:
                time.sleep(delay / 1e9)
            workload = shared._workload_view(self.cfg, streams=1, generation_budget=T.REPLY_TOKENS, prompt_seed=suffix_seed(self.label, self.index),
                                             finish_timeout_s=self.cfg['timeouts']['finish_s'])
            self.row['turn_end_ns'] = time.monotonic_ns()
            self.row['readers_alive'] = streams._cancel_streams([self.standing], workload)
            text = self.text
            body = shared._request(self.a.model, text, workload, self.salt)
            self.state = streams._start_streams(self.a.url, self.a.model, [body], [{'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(), 'prompt_chars': len(text)}], workload,
                                                instrument=UsageInstrument())[0]
            self.state.cached_tokens = getattr(self.state, 'cached_tokens', None)
            self.row['sent'] = True
            first = shared._wait_first_advance(self.state, self.cfg['timeouts']['read_s'])
            finish = shared._finish_or_cancel([self.state], workload)
            state = self.state
            self.row.update(send_at_ns=state.send_at_ns, first_at_ns=first, finish_at_ns=state.finish_at_ns, ttft_s=(first - state.send_at_ns) / 1e9 if first and state.send_at_ns else None,
                            since_bulk_send_s=(state.send_at_ns - self.bulk_send_ns) / 1e9 if state.send_at_ns else None, prompt_tokens=state.prompt_tokens,
                            cached_tokens=getattr(state, 'cached_tokens', None), new_tokens=state.prompt_tokens - state.cached_tokens if type(state.prompt_tokens) is int and type(getattr(state, 'cached_tokens', None)) is int else None,
                            output_completion=state.count(), ok=base.stream_ok(state) and not finish.get('timed_out'), error=state.error, malformed=state.malformed, finish=finish, output=digest(state))
        except Exception as exc:  # noqa: BLE001 - retained, never retried
            self.row['exception'] = exception_details(exc)
            self.row['ok'] = False


def longest_gap_s(state, start_ns, end_ns):
    """The longest gap between two token arrivals of one stream overlapping [start, end]; None when the stream has no arrival on either side."""
    stamps = sorted(s for s in state.arrivals)
    before, after = [s for s in stamps if s <= start_ns], [s for s in stamps if s >= end_ns]
    if not before or not after:
        return None
    points = [before[-1]] + [s for s in stamps if start_ns < s < end_ns] + [after[0]]
    return round(max(b - a for a, b in zip(points, points[1:])) / 1e9, 4)


def yield_stalls(history, always_on, proven):
    """Per served yield: the longest decode gap of the always-on streams across the FULL return (handback..reenter) and the re-entry (reenter..resume). Needs the clock proof."""
    out = {}
    for h in history or ():
        times = h.get('times') or {}
        if not proven or not all(type(times.get(k)) is int for k in ('handback', 'reenter', 'resume')):
            continue
        gaps = lambda a, b: [longest_gap_s(s, times[a], times[b]) for s in always_on]
        full, re_entry = gaps('handback', 'reenter'), gaps('reenter', 'resume')
        out[str(h.get('seq'))] = dict(full_return_stall_s=None if None in full or not full else max(full), reentry_stall_s=None if None in re_entry or not re_entry else max(re_entry))
    return out


def load_config(a, band):
    data = json.loads(Path(a.workload).read_text())
    budget = ap.standing_budget(data, band) if band else None
    if budget is None:
        raise ValueError('round-index-without-a-band')
    problems, cfg = shared.validate_manifest(data)
    if problems or cfg['repeats'] != 1:
        raise ValueError('frozen-native-workload-invalid')
    return dict(cfg, timeouts=dict(cfg['timeouts'], read_s=base.first_advance_read_timeout(cfg))), budget


def arm(a, row, cfg, budget, client, poll_client, *, sleep=time.sleep):
    """One declared arm. Every exception is retained in the record, never retried."""
    name, cohort = row['label'], row['cohort']
    record = dict(schema=ARM_SCHEMA, kind='auto', label=name, cohort=cohort, index=row['index'], part=row['part'], c1=row['c1'], size=row['size'], band=row['band'], selection=row['selection'], complete=False,
                  reasons=[], statuses=[], progress=[], metrics={}, run_id=a.run_id, engine_epoch=client.epoch, yield_on=row['yield_on'], plan=[list(p) for p in row['plan']],
                  generation_requests_expected=row['generations'], manifest_sha256=cfg['manifest_sha256'], standing_generation_budget=budget, returners=[], outputs={})
    workload = shared._workload_view(cfg, generation_budget=budget)
    prompts = shared._prefill_prompts(cfg)
    salt = shared._cache_salt(a.run_id, LANE + name, 0)
    record['salt_sha256'] = hashlib.sha256(salt.encode()).hexdigest()
    standing, sampler, state, poller, returners = [], None, None, None, []
    mark = len(getattr(client, 'history', ()))
    record['wall_start_s'] = time.time()

    def status(tag, history=False):
        reply = client.call('status')
        record['statuses'].append(dict(base.status_view(tag, reply), append_yield=yield_view(reply, history)))
        record['progress'].append(dict(progress.progress_view(reply) or {}, label=tag, native_stage_local=iw.stage_view(reply)))
        return reply
    try:
        if shared._package_gate(os.environ).get('ok') is not True:
            raise ValueError('reviewed-package-readiness-not-pass')
        entry = status('entry')
        auto = entry.get('auto_arm')
        if type(auto) is not dict:
            raise ValueError('auto-arm-policy-not-installed')
        if auto.get('manual_owned') is not True or auto.get('active') is not False or entry.get('route') != 'full' or entry.get('scheduler_requests') != 0:
            raise ValueError('arm-entry-not-latched-idle-full')
        config = (entry.get('append_yield') or {}).get('config') or {}
        if {k: config.get(k) for k in T.LAUNCH} != T.LAUNCH or (entry.get('append_yield') or {}).get('phase') is not None:
            raise ValueError('append-yield-launch-bounds-differ')
        yield_control(client, row['yield_on'], record)
        rows = streams.build_cohort(workload)
        bodies = []
        for i, item in enumerate(rows):
            body = shared._request(a.model, item['prompt'], workload, shared._cache_salt(a.run_id, progress.STANDING, i))
            bodies.append(dict(body, temperature=T.GREEDY_TEMPERATURE) if i in T.STANDING_GREEDY else body)
        standing = streams._start_streams(a.url, a.model, bodies, rows, workload)
        opened = streams._wait_open(standing, workload)
        record['open'] = {k: opened.get(k) for k in ('complete', 'reasons')}
        if opened.get('complete') is not True:
            raise ValueError('standing-cohort-unopened')
        iw.release_compose(row)(row['size'], row['stage_local'])(client, record, 'burst')
        for rec_key, missing in (('attn_meta', 'attn-meta-select-not-sent'), ('inwait', 'inwait-field-not-sent')):
            if not record.get(rec_key):
                raise ValueError(missing)
        sampler = shared.CountSampler(standing)
        sampler.start()
        sleep(base.SETTLE_S)
        b0 = sampler.snapshot()
        sleep(cfg['window_s'])
        b1 = sampler.snapshot()
        record['decode_only'] = shared._stream_window(standing, sampler, b0, b1)
        before = status('before-fresh')
        before_auto = before.get('auto_arm') or {}
        if before_auto.get('manual_owned') is not False or before_auto.get('active') is not False:
            raise ValueError('arm-lease-state-before-fresh-differs')
        record['metrics']['before-fresh'] = progress.preemption_metrics(a.url)
        cancel = record['solo_cancel'] = {'triggered': False}

        def solo_cancel():
            cancel.update(triggered=True, started_ns=time.monotonic_ns())
            cancel['readers_alive'] = streams._cancel_streams(standing, workload)
            cancel['ended_ns'] = time.monotonic_ns()
        returners = [Returner(a, cfg, rows[p[0]], standing[p[0]], shared._cache_salt(a.run_id, progress.STANDING, p[0]), p, name, row) for p in row['plan']]
        for r in returners:
            r.prepare()
        spec = cfg['prefill']['fresh']
        poller = progress.SoloPoller(poll_client, on_native=solo_cancel if cohort == 'solo' else None)
        poller.start(base.compact(before))
        sampler.snapshot()
        state, fresh_workload = shared._send_prefill(a.url, a.model, cfg, LANE + name, 0, prompts['fresh'], spec['generation_budget'], spec['seed'], salt)
        for r in returners:
            r.start(state.send_at_ns if type(state.send_at_ns) is int else time.monotonic_ns())
        first = shared._wait_first_advance(state, cfg['timeouts']['read_s'])
        record['fresh_finish'] = shared._finish_or_cancel([state], fresh_workload)
        sampler.snapshot()
        record['fresh'] = base.request_row(state, first, prompts, 'fresh')
        record['outputs']['fresh'] = digest(state)
        # A returner that was held behind the bulk (yield off, or past the bound) ends after it.
        for r in returners:
            r.thread.join(JOIN_S)
        record['returners'] = [r.row for r in returners]
        record['returners_alive'] = [r.index for r in returners if r.thread.is_alive()]
        poller.closed.wait(base.CLOSE_TIMEOUT_S)
        polled = poller.finish(base.CLOSE_TIMEOUT_S)
        cancel['trigger'] = poller.fired
        poller = None
        after = status('after-fresh', history=True)
        record['metrics']['after-fresh'] = progress.preemption_metrics(a.url)
        record['polled'] = {k: polled.get(k) for k in ('samples_read', 'samples_kept', 'errors', 'alive_after_join', 'poll_s', 'poll_fast_s')}
        record['polled_actions'] = polled.get('actions')
        record['route_samples'] = [{k: s.get(k) for k in ('route', 'route_generation', 'observed_clock_ns')} for s in polled.get('samples') or []][:256]
        if not base.stream_ok(state) or record['fresh_finish'].get('timed_out'):
            raise ValueError('fresh-client-completion-unproved')
        if not base.geometry_ok(state):
            raise ValueError('fresh-prompt-token-geometry-unproven')
        always_on = [s for i, s in enumerate(standing) if i not in {p[0] for p in row['plan']}]
        record['always_on'] = [s.index for s in always_on]
        if type(first) is int and type(state.send_at_ns) is int and first > state.send_at_ns:
            record['client_basis'] = dict(prefill_tokens_per_s=state.prompt_tokens / ((first - state.send_at_ns) / 1e9), bulk_ttft_s=(first - state.send_at_ns) / 1e9, bulk_complete_s=(state.finish_at_ns - state.send_at_ns) / 1e9 if state.finish_at_ns else None)
            if cohort == 'live':
                record['client_basis']['decode_window'] = shared._stream_window(always_on, sampler, state.send_at_ns, first)
        view_after = yield_view(after, True) or {}
        proven = (record['yield_control'].get('clock_proof') or {}).get('inside') is True
        record['yield_stalls'] = yield_stalls(view_after.get('history'), always_on, proven)
        bad = [r.index for r in returners if r.row.get('ok') is not True]
        if bad:
            record['reasons'].append('returner-not-complete:%s' % ','.join(str(x) for x in bad))
        if record['returners_alive']:
            record['reasons'].append('returner-still-running')
        if polled.get('errors') or polled.get('alive_after_join'):
            record['reasons'].append('poller-not-clean')
        if not set(polled.get('actions') or ()) <= ACTION_NAMES:
            record['reasons'].append('poller-sent-a-non-receipt-action')
        views = {s.get('label'): s for s in record['progress']}
        before_view, after_view = views.get('before-fresh') or {}, views.get('after-fresh') or {}
        record['progress_delta'] = {k: progress.delta(after_view, before_view, k) for k in ('chunks', 'commands', 'direct_commands', 'stock_rpcs', 'waits', 'wait_sends', 'wait_capped', 'line_errors', 'trace_lines', 'trace_dropped')}
        if before_view.get('native_stage_local') and after_view.get('native_stage_local'):
            record['stage_delta'] = iw.stage_delta(after_view['native_stage_local'], before_view['native_stage_local'])
        if record['progress_delta'].get('line_errors') or record['progress_delta'].get('trace_dropped'):
            record['reasons'].append('progress-lines-incomplete')
        p0, p1 = (record['metrics']['before-fresh'] or {}).get('num_preemptions_total'), (record['metrics']['after-fresh'] or {}).get('num_preemptions_total')
        if type(p0) is not float or type(p1) is not float:
            record['reasons'].append('preemption-counter-unavailable')
        elif p1 != p0:
            record['reasons'].append('kv-preemption-during-the-fresh-prompt')
        if cohort == 'solo' and (cancel.get('triggered') is not True or cancel.get('readers_alive') != []):
            record['reasons'].append('solo-cancel-unproved')
        # The stock append: same conversation, below the policy minimum, stock in every arm by design.
        mark_status = status('before-append')
        spec = cfg['prefill']['append']
        app_state, app_workload = shared._send_prefill(a.url, a.model, cfg, LANE + name, 1, prompts['append'], spec['generation_budget'], spec['seed'], salt)
        app_first = shared._wait_first_advance(app_state, cfg['timeouts']['read_s'])
        record['append_finish'] = shared._finish_or_cancel([app_state], app_workload)
        record['append'] = base.request_row(app_state, app_first, prompts, 'append')
        record['outputs']['append'] = digest(app_state)
        if not base.stream_ok(app_state) or record['append_finish'].get('timed_out'):
            record['reasons'].append('append-client-completion-unproved')
        tail = status('after-append')
        if (tail.get('auto_arm') or {}).get('selection_seq') != (mark_status.get('auto_arm') or {}).get('selection_seq'):
            record['reasons'].append('append-was-selected')
        record['outputs']['standing'] = {str(s.index): digest(s) for s in standing}
        record['control_actions'] = sorted({h.get('action') for h in list(getattr(client, 'history', ()))[mark:]})
        if not set(record['control_actions']) <= ACTION_NAMES:
            record['reasons'].append('arm-sent-a-non-receipt-action')
    except Exception as exc:  # noqa: BLE001 - retained, never retried
        record['error'] = exception_details(exc, getattr(client, 'control', None))
        if getattr(exc, 'refusal', None) is not None:
            record['controller_refusal'] = exc.refusal
        record['reasons'].append('arm-error')
    finally:
        if poller is not None:
            record['polled_after_error'] = {k: v for k, v in poller.finish(5.0).items() if k != 'samples'}
        for r in returners:
            if r.thread.ident is not None:
                r.thread.join(5.0)
        if returners and not record['returners']:
            record['returners'] = [r.row for r in returners]
        if sampler:
            sampler.close()
        live = ([state] if state is not None else []) + [r.state for r in returners if r.state is not None]
        if live:
            streams._stop_streams({}, live, shared._workload_view(cfg, streams=1))
        if standing:
            streams._stop_streams({}, standing, workload)
        record['wall_end_s'] = time.time()
    record['complete'] = not record['reasons']
    return record


def run_arm(a):
    name = T.label(a.c1, a.size, a.cohort, a.index)
    row = T.by_label().get(name)
    client = None
    try:
        if row is None:
            raise ValueError('arm-not-in-the-declared-table')
        cfg, budget = load_config(a, row['band'])
        client = refusal.RefusalKeepingClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
        poll = refusal.RefusalKeepingClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
        record = arm(a, row, cfg, budget, client, poll)
    except Exception as exc:  # noqa: BLE001 - retained, never retried
        record = dict(schema=ARM_SCHEMA, kind='auto', label=name, complete=False, reasons=['arm-refused'], error=exception_details(exc))
    if row is not None and 'attn_meta' in record:
        block = record['attn_meta']
        block['end'] = am.end_read(a)
        read = (block['end'] or {}).get('read')
        block['deltas'] = am.deltas(block.get('select'), read)
        block['table'] = row['attn']
        record.setdefault('slots_fast', dict(requested=row['next_levers'], select=None))['calls'] = iw.slots_calls((record.get('slots_fast') or {}).get('select'), read)
        iw.rank_lever_counters(record, row, read)
    with open(a.out, 'x') as handle:
        json.dump(record, handle, sort_keys=True)
    print('A1_TRUNK_ARM_RECORD %s %s' % (name, 'COMPLETE' if record.get('complete') is True else 'INCOMPLETE'))
    return 0 if record.get('complete') is True else 2


# --- reads (pure, over the arm's record) ----------------------------------------------------------------------------

def _status(rec, tag):
    return next((s for s in rec.get('statuses') or () if s.get('label') == tag), None)


def _delta(after, before):
    after, before = after or {}, before or {}
    return {k: (after.get(k) or 0) - (before.get(k) or 0) for k in sorted(set(after) | set(before)) if type(after.get(k, 0)) is int and type(before.get(k, 0)) is int}


def _s(later, earlier):
    return round((later - earlier) / 1e9, 4) if type(later) is int and type(earlier) is int else None


def lookup_read(before, after):
    """Every prefix lookup through the manager in this arm: the yield's scan apart from the rest (stock's loop, the admission). max is the run's so far."""
    before, after = before or {}, after or {}
    if not after:
        return None
    gain = lambda key: after.get(key, 0) - before.get(key, 0)
    calls, scan = gain('calls'), gain('scan_calls')
    return dict(calls=calls, scan_calls=scan, other_calls=calls - scan, total_s=round(gain('ns') / 1e9, 3), scan_s=round(gain('scan_ns') / 1e9, 3),
                mean_ms=round(gain('ns') / calls / 1e6, 1) if calls else None, max_ms=round(after.get('max_ns', 0) / 1e6, 1))


def yield_read(rec):
    """Per yield of this arm, from the journal; per arm the counter deltas."""
    before, after = _status(rec, 'before-fresh') or {}, _status(rec, 'after-fresh') or {}
    y0, y1 = before.get('append_yield') or {}, after.get('append_yield') or {}
    seq0 = y0.get('seq') if type(y0.get('seq')) is int else None
    history = y1.get('history') if type(y1.get('history')) is list else []
    rows = []
    for h in history:
        if seq0 is None or type(h.get('seq')) is not int or h['seq'] <= seq0:
            continue
        times = h.get('times') or {}
        present = [p for p in PHASES if type(times.get(p)) is int]
        steps = {'%s>%s' % (a, b): _s(times[b], times[a]) for a, b in zip(present, present[1:])}
        served = 'full' in times
        row = dict(seq=h['seq'], reason=h.get('reason'), served=served, arrival=h.get('arrival'), open_chunks=h.get('open_chunks'), phases_seen=present, steps=steps,
                   selected_to_full_s=_s(times.get('full'), times.get('selected')), selected_to_reopened_s=_s(times.get('bulk_reopened'), times.get('selected')),
                   route_generation_delta=h['route_generation_end'] - h['route_generation_selected'] if type(h.get('route_generation_end')) is int and type(h.get('route_generation_selected')) is int else None)
        row.update((rec.get('yield_stalls') or {}).get(str(h['seq'])) or {})
        rows.append(row)
    g0, g1 = before.get('route_generation'), after.get('route_generation')
    return dict(yields=rows, served=sum(1 for r in rows if r['served']), journal=len(rows), counters=_delta(y1.get('counters'), y0.get('counters')), end_reasons=_delta(y1.get('end_reasons'), y0.get('end_reasons')),
                arrivals=_delta(y1.get('arrivals'), y0.get('arrivals')), holds=_delta(y1.get('holds'), y0.get('holds')), route_generation_window=g1 - g0 if type(g0) is int and type(g1) is int else None,
                refusals=_delta(y1.get('refusals'), y0.get('refusals')), lookups=lookup_read(y0.get('lookups'), y1.get('lookups')), split_holds=_delta(y1.get('split_holds'), y0.get('split_holds')),
                current_seq=y1.get('current_seq'), clock_proven=((rec.get('yield_control') or {}).get('clock_proof') or {}).get('inside'))


def arm_read(rec, decode_only_rate):
    basis, window = rec.get('client_basis') or {}, (rec.get('client_basis') or {}).get('decode_window') or {}
    rows = window.get('streams') if type(window.get('streams')) is list else []
    per_stream = round(window['aggregate_tokens_per_s'] / len(rows), 3) if rows and type(window.get('aggregate_tokens_per_s')) in (int, float) else None
    reference = T.FIXED_REFERENCE_TOK_S / T.STANDING_DECODERS
    own = (rec.get('decode_only') or {}).get('aggregate_tokens_per_s')
    return dict(prefill_tok_s=round(basis['prefill_tokens_per_s'], 1) if type(basis.get('prefill_tokens_per_s')) in (int, float) else None, bulk_ttft_s=basis.get('bulk_ttft_s'), bulk_complete_s=basis.get('bulk_complete_s'),
                always_on=rec.get('always_on'), decode_per_stream_tok_s=per_stream, decode_vs_reference_per_stream=round(per_stream / reference, 4) if per_stream else None,
                decode_only_8_tok_s=own, decode_only_vs_reference=round(own / T.FIXED_REFERENCE_TOK_S, 4) if type(own) in (int, float) else None, decode_only_vs_this_run=round(own / decode_only_rate, 4) if type(own) in (int, float) and type(decode_only_rate) in (int, float) and decode_only_rate else None,
                returners=[{k: r.get(k) for k in ('index', 'suffix_tokens', 'tail_rows', 'short_tail', 'tail_attempts', 'prompt_tokens_tokenized', 'at_s', 'ttft_s', 'since_bulk_send_s', 'cached_tokens', 'new_tokens', 'prompt_tokens', 'output_completion', 'ok')} for r in rec.get('returners') or ()],
                stock_append_ttft_s=(rec.get('append') or {}).get('ttft_s'), progress=rec.get('progress_delta'), inwait={k: (rec.get('stage_delta') or {}).get('step_iw_' + k) for k in iw.IW_DELTA}, **{'yield': yield_read(rec)})


def _median(values):
    values = [v for v in values if type(v) in (int, float)]
    return round(statistics.median(values), 4) if values else None


def lever_read(arms_read):
    """The operator's budget, line by line; a read. On arms against their matched off arm. A line whose numbers are missing says so."""
    lines = []
    for row in T.arms():
        off_label = T.matched_off(row)
        on = arms_read.get(row['label'])
        if row['yield_on'] and row['cohort'] == 'live' and on:
            for r in on['returners']:
                budget = T.BUDGET_FIRST_OUTPUT_S[r['suffix_tokens']]
                lines.append(dict(line='first-output', arm=row['label'], index=r['index'], suffix_tokens=r['suffix_tokens'], ttft_s=r['ttft_s'], budget_s=budget, within=r['ttft_s'] <= budget if type(r['ttft_s']) in (int, float) else None))
        off = arms_read.get(off_label) if off_label else None
        if not on or not off:
            continue
        a, b = on.get('bulk_ttft_s'), off.get('bulk_ttft_s')
        lines.append(dict(line='bulk-complete', arm=row['label'], off=off_label, on_s=a, off_s=b, slack_s=T.BUDGET_BULK_SLACK_S, within=a <= b + T.BUDGET_BULK_SLACK_S if None not in (a, b) else None))
        c, d = on.get('decode_per_stream_tok_s'), off.get('decode_per_stream_tok_s')
        ratio = round(c / d, 4) if type(c) in (int, float) and type(d) in (int, float) and d else None
        lines.append(dict(line='decode', arm=row['label'], off=off_label, on_tok_s=c, off_tok_s=d, ratio=ratio, min_ratio=T.BUDGET_DECODE_RATIO, within=ratio >= T.BUDGET_DECODE_RATIO if ratio is not None else None))
    return lines


def engine_failures(arms, logs):
    """Signatures and hard markers in each arm's span (Step A's and C0's lists, torn-line aware)."""
    rows = ap.read_stamped(logs, lambda text: True)
    out = {}
    for rec in arms:
        hits = set()
        for stamp, text in rows:
            if torn_lines.in_span(stamp, rec):
                found = torn_lines.signature_of(text, SIGNATURES) or next((m for m in HARD_MARKERS if m in text), None)
                if found:
                    hits.add(found)
        if hits:
            out[rec['label']] = sorted(hits)
    return out


def grade(arms, ingress, checks, logs, watches=None, part='core'):
    known = [a for a in arms if a.get('label')]
    decode_records = {a['label']: a for a in known if a.get('kind') == 'decode-only'}
    prefill = [a for a in known if a.get('kind') != 'decode-only']
    by = {a['label']: a for a in prefill}
    want = T.arms()
    failures, missing = set(), set()
    for a in arms:
        if not a.get('label'):
            missing.add('arm-record-unreadable')
    afail, amiss, _classes, attn, start = iw.attn_gate(prefill, decode_records, ingress, watches, logs, want)
    failures |= afail
    missing |= amiss
    for name, watch in (watches or {}).items():
        if watch.get('verdict') == 'DEAD':
            failures.add('engine-death-watch:' + name)
    for name, hits in engine_failures(known, logs).items():
        failures.add('engine-signature:%s:%s' % (name, ','.join(hits)))
    for name, check in (checks or {}).items():
        if check.get('verdict') != 'PASS':
            missing.add('check-%s-%s' % (name, check.get('verdict') or 'absent'))
    rate = ((decode_records.get(T.REFERENCE_ARM) or {}).get('decode_only') or {}).get('aggregate_tokens_per_s')
    reads = {}
    for row in want:
        name = row['label']
        rec = by.get(name)
        if rec is None:
            missing.add(name + ':no-arm-record')
            continue
        if rec.get('complete') is not True or rec.get('reasons'):
            missing.add('%s:incomplete:%s' % (name, ','.join(rec.get('reasons') or ['no-reason'])))
        if (ingress or {}).get(name, {}).get('verdict') not in ('PASS', None):
            missing.add(name + ':ingress-not-the-arm')
        reads[name] = read = arm_read(rec, rate)
        y = read['yield']
        control = rec.get('yield_control') or {}
        if control.get('enabled') is not row['yield_on']:
            missing.add(name + ':yield-selection-not-acknowledged')
        if y['current_seq'] is not None:
            failures.add(name + ':yield-open-at-the-arm-end')
        if not row['yield_on'] and (y['journal'] or y['counters'].get('selected')):
            failures.add(name + ':yield-off-selected-a-yield')
        if row['yield_on'] and row['plan'] and not y['served']:
            missing.add(name + ':yield-on-served-none')
        if y['route_generation_window'] != 2 + 2 * y['served'] and rec.get('complete') is True:
            failures.add('%s:route-generations-%s-for-%d-served' % (name, y['route_generation_window'], y['served']))
        if any(y['counters'].get(k) for k in REFUSAL_COUNTERS):
            missing.add(name + ':a-yield-fell-back')
    verdict = 'FAIL' if failures else 'INC' if missing else 'PASS'
    cells = {}
    for row in want:
        if row['label'] in reads:
            cells.setdefault('%s %s' % (row['c1'], row['cohort']), []).append(reads[row['label']])
    medians = {name: dict(prefill_tok_s=_median(r['prefill_tok_s'] for r in rows), bulk_ttft_s=_median(r['bulk_ttft_s'] for r in rows), decode_per_stream_tok_s=_median(r['decode_per_stream_tok_s'] for r in rows),
                          ttft_500_s=_median(t['ttft_s'] for r in rows for t in r['returners'] if t['suffix_tokens'] == 500), ttft_1500_s=_median(t['ttft_s'] for r in rows for t in r['returners'] if t['suffix_tokens'] == 1500),
                          selected_to_full_s=_median(y['selected_to_full_s'] for r in rows for y in r['yield']['yields']), selected_to_drained_s=_median(y['steps'].get('selected>drained') for r in rows for y in r['yield']['yields']),
                          served=[r['yield']['served'] for r in rows]) for name, rows in cells.items()}
    return dict(schema=GRADE_SCHEMA, verdict=verdict, failures=sorted(failures), missing=sorted(missing), part=part, scope=SCOPE, not_covered=NOT_COVERED, is_budget_gate=False, lever_read=lever_read(reads),
                cells=medians, arms=reads, decode_only={k: (v.get('decode_only') or {}).get('aggregate_tokens_per_s') for k, v in decode_records.items()}, fixed_reference_tok_s=T.FIXED_REFERENCE_TOK_S, attn_meta=attn, startup=start)


def report(result):
    def cell(v, digits=2):
        return '-' if v is None else ('%.*f' % (digits, v) if type(v) is float else str(v))
    lines = ['# q18a-a1-trunk report', '', 'Verdict %s. %s.' % (result['verdict'], result['scope']), '', 'Not covered: %s.' % result['not_covered'], '', 'Failures: %s' % (result['failures'] or 'none'), '', 'Missing: %s' % (result['missing'] or 'none'), '',
             'Decode-only tok/s: %s; fixed reference %s tok/s for 8 streams.' % (json.dumps(result['decode_only'], sort_keys=True), result['fixed_reference_tok_s']), '',
             '## Budget lines (a read)', '', '| line | arm | figures | within |', '|---|---|---|---|']
    for row in result['lever_read']:
        figures = {k: v for k, v in row.items() if k not in ('line', 'arm', 'within')}
        lines.append('| %s | %s | %s | %s |' % (row['line'], row['arm'], json.dumps(figures, sort_keys=True), row['within']))
    lines += ['', '## Arms', '', '| arm | prefill tok/s | bulk first s | decode tok/s per always-on stream | vs reference | served | arrivals | end reasons | applied in wait |', '|---|---|---|---|---|---|---|---|---|']
    for name, r in result['arms'].items():
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (name, cell(r['prefill_tok_s'], 1), cell(r['bulk_ttft_s']), cell(r['decode_per_stream_tok_s'], 1), cell(r['decode_vs_reference_per_stream'], 3), r['yield']['served'],
                                                                       json.dumps(r['yield']['arrivals'], sort_keys=True), json.dumps(r['yield']['end_reasons'], sort_keys=True), cell(r['inwait'].get('applied'))))
    lines += ['', 'Decode per always-on stream is not an 8-stream figure: fewer streams run after a turn ends, so compare on against off first.', '', '## Yields', '', '| arm | seq | arrival | open chunks | reason | selected to full s | steps s | full-return stall s | re-entry stall s |', '|---|---|---|---|---|---|---|---|---|']
    for name, r in result['arms'].items():
        for y in r['yield']['yields']:
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (name, y['seq'], y['arrival'], y['open_chunks'], y['reason'], cell(y['selected_to_full_s'], 3), json.dumps(y['steps'], sort_keys=True), cell(y.get('full_return_stall_s'), 3), cell(y.get('reentry_stall_s'), 3)))
    lines += ['', '## Refusals and lookups', '', 'Every prefix lookup through the manager in the arm, off arms too. `other` is stock\'s loop and the admission: if an off arm shows many of about 265 ms, stock taxes a held request on its own.', '', '| arm | refusals | calls | yield scan | other | total s | scan s | mean ms | max ms so far | stock holds |', '|---|---|---|---|---|---|---|---|---|---|']
    for name, r in result['arms'].items():
        l = r['yield'].get('lookups') or {}
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (name, json.dumps(r['yield'].get('refusals'), sort_keys=True), l.get('calls'), l.get('scan_calls'), l.get('other_calls'), l.get('total_s'), l.get('scan_s'), l.get('mean_ms'), l.get('max_ms'), json.dumps(r['yield'].get('split_holds'), sort_keys=True)))
    lines += ['', '## Returners', '', '| arm | stream | suffix | tail rows | kept short | sent after bulk s | first output s | cached | new |', '|---|---|---|---|---|---|---|---|---|']
    for name, r in result['arms'].items():
        for t in r['returners']:
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (name, t['index'], t['suffix_tokens'], t.get('tail_rows'), t.get('short_tail'), cell(t['since_bulk_send_s']), cell(t['ttft_s'], 3), t['cached_tokens'], t['new_tokens']))
    return '\n'.join(lines) + '\n'


# --- the exact-path smoke -----------------------------------------------------------------------------------------

def smoke_check(rec, row=None):
    """Arm 1's facts: real request shape, real control dispatch, positive native engagement, one complete observation, a served yield. (ok, reasons); the queue aborts the table on not ok."""
    row = row or T.by_label()[T.SMOKE_ARM]
    if not isinstance(rec, dict):
        return False, ['smoke-record-unreadable']
    reasons = []
    if rec.get('label') != row['label'] or rec.get('complete') is not True or rec.get('reasons'):
        reasons.append('smoke-arm-not-complete')
    outputs = rec.get('outputs') or {}
    if not (outputs.get('fresh') and outputs.get('append') and outputs.get('standing')):
        reasons.append('smoke-request-shape-not-observed')
    if rec.get('generation_requests_expected') != row['generations']:
        reasons.append('smoke-request-shape-differs')
    control = rec.get('yield_control') or {}
    if control.get('confirmed') is not True or control.get('enabled') is not row['yield_on']:
        reasons.append('smoke-yield-control-not-acknowledged')
    if (rec.get('inwait') or {}).get('acknowledged') is not row['inwait'] or (rec.get('inwait') or {}).get('early_acknowledged') is not row['next_levers']:
        reasons.append('smoke-control-not-acknowledged')
    if (rec.get('attn_meta') or {}).get('select') is None:
        reasons.append('smoke-select-not-sent')
    delta = rec.get('progress_delta') or {}
    if type(delta.get('chunks')) is not int or delta['chunks'] < 1 or delta.get('line_errors'):
        reasons.append('smoke-no-native-chunks')
    y = yield_read(rec)
    if not y['served']:
        reasons.append('smoke-no-yield-served')
    if any(y['counters'].get(k) for k in REFUSAL_COUNTERS):
        reasons.append('smoke-a-yield-fell-back')
    returners = rec.get('returners') or ()
    if len(returners) != len(row['plan']) or any(r.get('ok') is not True for r in returners):
        reasons.append('smoke-a-returner-did-not-finish')
    return not reasons, sorted(set(reasons))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('arm')
    r.add_argument('--url', required=True)
    r.add_argument('--model', required=True)
    r.add_argument('--run-id', required=True)
    r.add_argument('--epoch', type=int, required=True)
    r.add_argument('--c1', choices=T.SELECTIONS, required=True, help="this arm's selection (inherited flag name): the yield and its returners; inwait's ps1 levers in every arm")
    r.add_argument('--size', type=int, choices=T.SIZES, required=True)
    r.add_argument('--cohort', choices=progress.COHORTS, required=True)
    r.add_argument('--index', type=int, choices=T.INDEXES, required=True)
    r.add_argument('--workload', required=True)
    r.add_argument('--control-timeout-s', type=float, default=120.0)
    r.add_argument('--out', required=True)
    k = sub.add_parser('smoke', help='the exact-path smoke: 0 engaged, 2 not (the queue aborts the table)')
    k.add_argument('--arm', required=True)
    k.add_argument('--out', required=True)
    d = sub.add_parser('decode-arm')
    for flag in ('--url', '--model', '--run-id', '--workload', '--out'):
        d.add_argument(flag, required=True)
    d.add_argument('--epoch', type=int, required=True)
    d.add_argument('--label', choices=tuple(x['label'] for x in T.DECODE_ARMS), required=True)
    d.add_argument('--attn', choices=('on', 'off'), required=True)
    d.add_argument('--control-timeout-s', type=float, default=120.0)
    for action, text in (('grade', 'the grade: the queue verdict'), ('read', 'the same over every arm: a read only')):
        g = sub.add_parser(action, help=text)
        g.add_argument('--arm', action='append', required=True)
        g.add_argument('--ingress', action='append', default=[], help='label=path')
        g.add_argument('--watch', action='append', default=[], help='label=engine-watch record path')
        g.add_argument('--check', action='append', default=[], help='label=path')
        g.add_argument('--log', action='append', default=[], help='followed engine log (stdout/stderr)')
        g.add_argument('--report', help='write the report here')
        g.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if Path(a.out).exists():
        return 2
    if a.action == 'arm':
        return run_arm(a)
    if a.action == 'decode-arm':
        return am.run_decode_arm(a)
    if a.action == 'smoke':
        ok, reasons = smoke_check(base.load(a.arm))
        with open(a.out, 'x') as handle:
            json.dump(dict(schema='a1-trunk-smoke/1', engaged=ok, reasons=reasons), handle, sort_keys=True)
        print('A1_TRUNK_SMOKE %s %s' % ('ENGAGED' if ok else 'NOT_ENGAGED', ','.join(reasons)))
        return 0 if ok else 2
    arms = [base.load(x) or {'label': None, 'complete': False, 'reasons': ['arm-artifact-unreadable:' + Path(x).name]} for x in a.arm]
    result = grade(arms, ap.load_pairs(a.ingress), ap.load_pairs(a.check), a.log, ap.load_pairs(a.watch), part='core' if a.action == 'grade' else 'all')
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    if a.report:
        with open(a.report, 'x') as handle:
            handle.write(report(result))
    print('A1_TRUNK_%s %s' % ('GRADE' if a.action == 'grade' else 'READ', result['verdict']))
    if a.action == 'read':
        return 0 if result['verdict'] == 'PASS' else 2      # a read: never 1, the queue does not count it
    return {'PASS': 0, 'FAIL': 1}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
