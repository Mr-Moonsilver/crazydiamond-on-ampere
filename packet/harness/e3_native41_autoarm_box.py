#!/usr/bin/env python3
"""Native41 box arms: plain W2 traffic with the manual lease held (stock) or released (auto).

Both arms send the frozen q18a W2 requests through the same plain streaming
clients held-b used: eight standing decoders, then the fresh prompt and its
append. No prepare, admission, prefill_begin or other native command is sent
around the measured prefill. The stock arm keeps the manual lease (auto-arm
suppressed). The auto arm pauses, releases and resumes once while the standing
cohort is open, before its decode-only window. The engine policy alone then
decides whether the fresh prompt runs natively; the arm reads only status and
the read-only auto window. Receipts hold counts, hashes and identities, never
prompts or auth.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import e3_decode_correctness as streams
import e3_native_performance as perf
import e3_overlap_performance as shared
from e3_native_qualification_driver import exception_details
from e3_native_runtime_client import NativeRuntimeClient
from native_overlap_metrics import throughput

ARM_SCHEMA = 'native41-autoarm-arm/1'
GRADE_SCHEMA = 'native41-autoarm-grade/1'
KINDS = ('stock', 'auto')
STANDING = 'native41-standing'
SETTLE_S = 5.0
CLOSE_TIMEOUT_S = 60.0
CLOSE_POLL_S = 0.5
# Generation requests one arm sends: eight standing decoders, fresh, append.
ARM_GENERATIONS = 10


def auto_view(status):
    auto = status.get('auto_arm') if type(status) is dict else None
    if type(auto) is not dict:
        return None
    armed = auto.get('last_armed')
    armed_status = armed.get('status') if type(armed) is dict else None
    return {k: auto.get(k) for k in ('phase', 'active', 'manual_owned', 'selection_seq', 'completed',
                                     'request_uid', 'minimum_native_rows', 'max_cohort')} | {
        'last_armed': None if type(armed) is not dict else {
            'selection_seq': armed.get('selection_seq'),
            'route': (armed_status or {}).get('route'),
            'route_generation': (armed_status or {}).get('route_generation'),
            'observed_clock_ns': (armed_status or {}).get('observed_clock_ns')},
        'last_refusal': auto.get('last_refusal'), 'last_close': auto.get('last_close'),
        'last_unselected': auto.get('last_unselected')}


def status_view(label, status):
    admission = status.get('native_admission') or {}
    return {'label': label, 'route': status.get('route'), 'route_generation': status.get('route_generation'),
            'paused': status.get('paused'), 'scheduler_requests': status.get('scheduler_requests'),
            'transition_pending': status.get('transition_pending'),
            'admission_intent': status.get('admission_intent'),
            'observed_clock_ns': status.get('observed_clock_ns'),
            'native_admission': {k: admission.get(k) for k in ('admitted', 'refused', 'alloc_failures',
                                                               'split_stock_deferrals', 'refusal_seq')},
            'auto_arm': auto_view(status)}


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


def wait_closed(client, before, record, *, clock=time.monotonic, sleep=time.sleep):
    """After the fresh first token: status reads until the cycle that started after `before` closed."""
    deadline = clock() + CLOSE_TIMEOUT_S
    while True:
        status = client.call('status')
        auto = status.get('auto_arm') or {}
        if (auto.get('active') is False and type(auto.get('completed')) is int
                and auto['completed'] > before.get('completed', 0)):
            record['statuses'].append(status_view('cycle-closed', status))
            return status
        if clock() >= deadline:
            record['statuses'].append(status_view('cycle-not-closed', status))
            return status
        sleep(CLOSE_POLL_S)


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


def auto_proof(window_end, bracket, epoch, prompt_tokens):
    """perf.native_window over the auto window, plus the binding to this plain request."""
    record = {'begin': {'engine_epoch': (window_end.get('begin') or {}).get('engine_epoch'),
                        'measure_start_ns': window_end.get('measure_start_ns'),
                        'snapshot': window_end.get('begin')},
              'end': {'engine_epoch': window_end.get('engine_epoch'),
                      'measure_end_ns': window_end.get('measure_end_ns'),
                      'raw_journal_evidence': window_end.get('raw_journal_evidence'),
                      'clock': window_end.get('clock')},
              'client_bracket': bracket}
    proof = perf.native_window(record, 'fresh', epoch)
    reasons = list(proof['reasons']) + list(window_end.get('auto_reasons') or [])
    if window_end.get('complete') is not True:
        reasons.append('auto-window-incomplete')
    group = proof.get('group') or {}
    view = proof.get('view') or {}
    segments = view.get('segments') or []
    uids = {s.get('request') for s in segments}
    intent = ((window_end.get('begin') or {}).get('admission_intent') or {})
    if len(uids) != 1 or intent.get('request_uid') not in uids:
        reasons.append('auto-segments-not-the-armed-request')
    # The driver's prompt_end is request.num_tokens (scheduler_hooks), so the
    # whole fresh prompt, as the client's usage counts it.
    if type(prompt_tokens) is not int or group.get('prompt_end') != prompt_tokens:
        reasons.append('auto-group-not-this-prompt')
    if not segments or any(s.get('transport') != 'native-split' for s in segments):
        reasons.append('auto-segments-not-native-split')
    if sum(1 for s in segments if s.get('terminal')) != 1:
        reasons.append('auto-terminal-chunk-not-unique')
    return {k: v for k, v in proof.items() if k != 'view'} | {
        'reasons': sorted(set(reasons)), 'complete': not reasons, 'segments': len(segments),
        'request_uid_sha256': (hashlib.sha256(intent['request_uid'].encode()).hexdigest()
                               if type(intent.get('request_uid')) is str else None),
        'origin': intent.get('origin'), 'selection_seq': intent.get('selection_seq')}


def returns_ms(window_end):
    """Last native STATE_WRITE (final commit) to the all-rank FULL ack, engine clock."""
    raw = window_end.get('raw_journal_evidence') or {}
    segments = (raw.get('measurement') or {}).get('segments') or []
    acks = raw.get('transition_acks') or []
    ends = [s.get('end_monotonic_ns') for s in segments]
    seen = {a.get('observed_ns') for a in acks}
    if not ends or not all(type(n) is int for n in ends) or len(seen) != 1 or type(next(iter(seen))) is not int:
        return {'final_commit_to_all_rank_full_ack_ms': None, 'reason': 'return-stamps-unavailable'}
    ack = seen.pop()
    return {'final_commit_to_all_rank_full_ack_ms': (ack - max(ends)) / 1e6, 'acks': len(acks),
            'basis': 'engine host-monotonic: max segment end_monotonic_ns -> shared all-rank FULL ack observed_ns'}


def arm(a, cfg, client, *, sleep=time.sleep):
    kind, index = a.kind, a.index
    record = {'schema': ARM_SCHEMA, 'kind': kind, 'index': index, 'complete': False, 'reasons': [],
              'statuses': [], 'run_id': a.run_id, 'engine_epoch': client.epoch,
              'generation_requests_expected': ARM_GENERATIONS, 'manifest_sha256': cfg['manifest_sha256'],
              'native_commands_around_prefill': [] if kind == 'stock' else ['auto_window_begin', 'auto_window_end']}
    workload = shared._workload_view(cfg)
    prompts = shared._prefill_prompts(cfg)
    salt = shared._cache_salt(a.run_id, 'native41-%s-%d' % (kind, index), 0)
    record['salt_sha256'] = hashlib.sha256(salt.encode()).hexdigest()
    standing, sampler, state = [], None, None
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
        record['reasons'] += ['decode-only:' + r for r in stream_reasons(record['decode_only'], standing, b1)]
        before = client.call('status')
        record['statuses'].append(status_view('before-fresh', before))
        before_auto = before.get('auto_arm') or {}
        if before_auto.get('manual_owned') is not (kind == 'stock') or before_auto.get('active') is not False:
            raise ValueError('arm-lease-state-before-fresh-differs')
        local_start = time.monotonic_ns()
        if kind == 'auto':
            record['window_begin'] = {k: v for k, v in client.call('auto_window_begin').items() if k != 'snapshot'}
        sampler.snapshot()
        spec = cfg['prefill']['fresh']
        state, fresh_workload = shared._send_prefill(a.url, a.model, cfg, 'native41-' + kind, 0, prompts['fresh'],
                                                     spec['generation_budget'], spec['seed'], salt)
        first = shared._wait_first_advance(state, cfg['timeouts']['read_s'])
        record['fresh_finish'] = shared._finish_or_cancel([state], fresh_workload)
        sampler.snapshot()
        closed = wait_closed(client, before_auto, record, sleep=sleep) if kind == 'auto' else None
        if kind == 'auto':
            window_end = client.call('auto_window_end')
            local_end = time.monotonic_ns()
            record['window_end'] = window_end
        else:
            local_end = time.monotonic_ns()
            after = client.call('status')
            record['statuses'].append(status_view('after-fresh', after))
        bracket = {'start_ns': local_start, 'end_ns': local_end}
        record['client_bracket'] = bracket
        record['fresh'] = request_row(state, first, prompts, 'fresh')
        if not perf._stream_ok(state) or record['fresh_finish'].get('timed_out'):
            raise ValueError('fresh-client-completion-unproved')
        if not geometry_ok(state):
            raise ValueError('fresh-prompt-token-geometry-unproven')
        ttft = record['fresh']['ttft_s']
        record['client_basis'] = {
            'prefill_tokens_per_s': throughput(state.prompt_tokens, ttft),
            'basis': 'observed fresh prompt tokens / client TTFT (held-b serving basis)',
            'decode_window': shared._stream_window(standing, sampler, state.send_at_ns, first)}
        record['reasons'] += ['client-window:' + r for r in
                              stream_reasons(record['client_basis']['decode_window'], standing, first)]
        if kind == 'auto':
            proof = auto_proof(window_end, bracket, client.epoch, state.prompt_tokens)
            record['auto_proof'] = proof
            group = proof.get('group') or {}
            if proof['complete'] and type(group.get('start_ns')) is int and type(group.get('end_ns')) is int:
                window = shared._stream_window(standing, sampler, group['start_ns'], group['end_ns'])
                record['engine_basis'] = {'prefill_tokens_per_s': proof.get('prefill_tokens_per_s'),
                                          'basis': proof.get('headline_basis'), 'decode_window': window}
                record['reasons'] += ['engine-window:' + r for r in stream_reasons(window, standing, group['end_ns'])]
            else:
                record['reasons'] += ['auto-not-proven:' + r for r in proof['reasons']] or ['auto-not-proven']
            record['returns'] = returns_ms(window_end)
            closed_auto = (closed or {}).get('auto_arm') or {}
            if closed_auto.get('completed') != before_auto.get('completed', -2) + 1:
                record['reasons'].append('auto-cycle-not-closed-once')
        else:
            after_auto = (after.get('auto_arm') or {})
            if (after_auto.get('selection_seq') != before_auto.get('selection_seq')
                    or after_auto.get('manual_owned') is not True):
                record['reasons'].append('stock-arm-auto-selection-or-release-observed')
        # The append is the same conversation; its remaining prefill is below the
        # policy minimum, so it stays stock in both arms by design.
        mark = client.call('status')
        record['statuses'].append(status_view('before-append', mark))
        spec = cfg['prefill']['append']
        app_state, app_workload = shared._send_prefill(a.url, a.model, cfg, 'native41-' + kind, 1, prompts['append'],
                                                       spec['generation_budget'], spec['seed'], salt)
        app_first = shared._wait_first_advance(app_state, cfg['timeouts']['read_s'])
        record['append_finish'] = shared._finish_or_cancel([app_state], app_workload)
        record['append'] = request_row(app_state, app_first, prompts, 'append')
        if not perf._stream_ok(app_state) or record['append_finish'].get('timed_out'):
            record['reasons'].append('append-client-completion-unproved')
        tail = client.call('status')
        record['statuses'].append(status_view('after-append', tail))
        if ((tail.get('auto_arm') or {}).get('selection_seq') != (mark.get('auto_arm') or {}).get('selection_seq')):
            record['reasons'].append('append-was-selected')
        record['complete'] = not record['reasons']
    except Exception as exc:  # noqa: BLE001 - retained, never retried
        record['error'] = exception_details(exc, getattr(client, 'control', None))
        record['reasons'].append('arm-error')
        record['complete'] = False
        if 'window_begin' in record and 'window_end' not in record:
            # Close the read-only window so the session does not keep it open.
            try:
                record['window_end_after_error'] = {k: v for k, v in client.call('auto_window_end').items()
                                                    if k in ('complete', 'auto_reasons', 'route')}
            except Exception as close:  # noqa: BLE001 - recorded only
                record['window_end_after_error'] = {'error': exception_details(close, getattr(client, 'control', None))}
    finally:
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
    engine = row.get('engine_basis') or {}
    during_c = (client.get('decode_window') or {}).get('aggregate_tokens_per_s')
    during_e = (engine.get('decode_window') or {}).get('aggregate_tokens_per_s')
    return {'kind': row.get('kind'), 'index': row.get('index'), 'complete': row.get('complete'),
            'decode_only_tokens_per_s': only,
            'client': {'prefill_tokens_per_s': client.get('prefill_tokens_per_s'),
                       'decode_during_prefill_tokens_per_s': during_c, 'decode_share': ratio(during_c, only),
                       'ttft_s': (row.get('fresh') or {}).get('ttft_s')},
            'engine': None if not engine else {
                'prefill_tokens_per_s': engine.get('prefill_tokens_per_s'),
                'decode_during_prefill_tokens_per_s': during_e, 'decode_share': ratio(during_e, only)},
            'returns_ms': (row.get('returns') or {}).get('final_commit_to_all_rank_full_ack_ms'),
            'append_ttft_s': (row.get('append') or {}).get('ttft_s')}


def grade(arms, ingress, checks):
    """Auto engagement is graded; speed is reported beside the same-queue stock arms, never a gate."""
    missing, failures = [], []
    rows = [arm_metrics(r) for r in arms]
    for r in arms:
        label = '%s-%s' % (r.get('kind'), r.get('index'))
        if r.get('complete') is not True:
            missing.extend(label + ':' + x for x in (r.get('reasons') or ['incomplete']))
        if r.get('kind') == 'auto' and (r.get('auto_proof') or {}).get('complete') is not True:
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
                      'client_decode_share': median(kind, ('client', 'decode_share')),
                      'decode_only_tokens_per_s': median(kind, ('decode_only_tokens_per_s',)),
                      'ttft_s': median(kind, ('client', 'ttft_s'))} for kind in KINDS}
    summary['auto']['engine_prefill_tokens_per_s'] = median('auto', ('engine', 'prefill_tokens_per_s'))
    summary['auto']['engine_decode_share'] = median('auto', ('engine', 'decode_share'))
    summary['auto_over_stock'] = {
        'client_prefill': ratio(summary['auto']['client_prefill_tokens_per_s'], summary['stock']['client_prefill_tokens_per_s']),
        'client_decode_share': ratio(summary['auto']['client_decode_share'], summary['stock']['client_decode_share'])}
    verdict = 'FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS'
    return {'schema': GRADE_SCHEMA, 'verdict': verdict, 'failures': failures, 'missing': sorted(set(missing)),
            'auto_engaged': [bool((r.get('auto_proof') or {}).get('complete')) for r in arms if r.get('kind') == 'auto'],
            'arms': rows, 'summary': summary, 'speed_is_gate': False, 'target_is_gate': False,
            'all_declared_arms_retained': True, 'retry_selected': False, 'picked': False,
            'held_b_comparison': 'hub-side: held-b overlap-round native windows and serving TTFT beside these arms',
            'scope': 'auto-arm engagement from plain W2 traffic; descriptive speed beside same-queue stock arms'}


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
            cfg = dict(cfg, timeouts=dict(cfg['timeouts'], read_s=perf.first_advance_read_timeout(cfg)))
            client = NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
            result = arm(a, cfg, client)
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
