#!/usr/bin/env python3
"""Decode interference while a fresh long prompt prefills (W2), with raw per-stream events and hard verdicts.

STREAMS streams (PROMPT tokens, distinct salts, forced long output) generate; after all are in steady decode and
BASE seconds have passed, one fresh LONG-token prompt (unique salt, 1 output token, streamed) arrives. Every
streamed chunk of a decoder carries its cumulative completion-token count (continuous_usage_stats).

Measurement semantics:
  * monotonic clock for every timestamp (wall-clock anchor saved once);
  * an "advance" is an event whose cumulative count grew; events without growth are kept raw but ignored;
    tokens per advance > 1 means the client received coalesced tokens (gap = client-visible, not engine);
  * window gaps: every gap between consecutive advances whose END lies in the window, plus each stream's open
    gap at the window end (lower bound), so a silent stream counts as one long gap;
  * per stream p50/p95/p99/max gap and tok/s; the worst stream is reported separately from pooled numbers;
  * newcomer: TTFT (client), plus engine queue / prefill time from /metrics histogram deltas.
Fails (exit 1) on any thread error, a stream that stops before the newcomer finishes, a missing newcomer token,
too few advances, or injected faults (INJECT=missing|error|stop, to prove the checks work).
usage: stall_trace2.py URL MODEL STREAMS PROMPT LONG LABEL OUT.json      env BASE=20 AFTER=10 INJECT=
       stall_trace2.py --recompute OUT.json                              (recomputes the report from raw events)
"""
import json, os, random, re, sys, threading, time, urllib.request, uuid


def pct(v, q):
    return round(sorted(v)[min(len(v) - 1, int(q * len(v)))] * 1000, 1) if v else None


def stats(gaps):
    return {'n': len(gaps), 'p50': pct(gaps, .5), 'p95': pct(gaps, .95), 'p99': pct(gaps, .99), 'max': pct(gaps, 1)}


def advances(ev):
    out, last = [], 0
    for t, n in ev:
        if n > last:
            out.append((t, n, n - last)); last = n
    return out


def window(raw, t0, t1):
    per = []
    for ev in raw['events']:
        adv = advances(ev)
        a = max((n for t, n, _ in adv if t <= t0), default=0)
        b = max((n for t, n, _ in adv if t <= t1), default=0)
        times = [t for t, _, _ in adv]
        gaps = [y - x for x, y in zip(times, times[1:]) if t0 < y <= t1]
        last = max((t for t in times if t <= t1), default=None)
        if last is not None:
            gaps.append(t1 - max(last, t0))  # open gap at the window end (lower bound)
        tpe = [k for t, _, k in adv if t0 < t <= t1]
        per.append({'tps': round((b - a) / (t1 - t0), 2), 'gap_ms': stats(gaps),
                    'tokens_per_advance_max': max(tpe, default=0), 'coalesced_advances': sum(k > 1 for k in tpe),
                    '_gaps': gaps})
    pooled = [g for p in per for g in p['_gaps']]
    worst = max(range(len(per)), key=lambda i: per[i]['gap_ms']['p99'] or 0)
    for p in per:
        del p['_gaps']
    return {'s': round(t1 - t0, 2), 'agg_tps': round(sum(p['tps'] for p in per), 1),
            'min_stream_tps': min(p['tps'] for p in per), 'pooled_gap_ms': stats(pooled),
            'worst_stream': worst, 'worst_stream_gap_ms': per[worst]['gap_ms'], 'per_stream': per}


def report(raw):
    m = raw['marks']
    res = {k: raw[k] for k in ('label', 'streams', 'stream_prompt', 'long_prompt', 'wall_anchor', 'errors')}
    res.update(before=window(raw, m['b0'], m['b1']), during=window(raw, m['p0'], m['p1']),
               after=window(raw, m['p1'], m['a1']))
    res['newcomer'] = {'client_ttft_s': round(m['p1'] - m['p0'], 2),
                       'client_prompt_tps': round(raw['long_prompt'] / (m['p1'] - m['p0'])),
                       **raw.get('engine_newcomer', {})}
    fails = list(raw['errors'])
    for i, ev in enumerate(raw['events']):
        adv = advances(ev)
        if not any(t > m['p1'] for t, _, _ in adv):
            fails.append(f'stream {i} made no progress after the newcomer finished (stopped or stalled)')
    if not raw.get('newcomer_token'):
        fails.append('newcomer returned no token')
    if res['during']['pooled_gap_ms']['n'] < raw['streams'] * 5:
        fails.append(f"too few advances during prefill ({res['during']['pooled_gap_ms']['n']})")
    res['fails'] = fails
    res['verdict'] = 'FAIL' if fails else 'PASS'
    return res


def short(res):
    out = {k: v for k, v in res.items() if k not in ('before', 'during', 'after')}
    out.update({w: {k: v for k, v in res[w].items() if k != 'per_stream'} for w in ('before', 'during', 'after') if w in res})
    return out


if sys.argv[1] == '--recompute':
    raw = json.load(open(sys.argv[2]))['raw']
    if not raw.get('marks') or 'a1' not in raw['marks']:
        print(json.dumps({'verdict': 'FAIL', 'fails': ['incomplete run'] + raw.get('errors', [])})); sys.exit(1)
    res = report(raw)
    print(json.dumps(short(res)))
    sys.exit(0 if res['verdict'] == 'PASS' else 1)

URL, MODEL, N, PT, LONG, LABEL, OUT = (sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                                       int(sys.argv[5]), sys.argv[6], sys.argv[7])
BASE, AFTER, INJECT = float(os.environ.get('BASE', 20)), float(os.environ.get('AFTER', 10)), os.environ.get('INJECT', '')
corpus = open('/home/<service-user>/p2p-tests/corpus.txt', encoding='utf-8', errors='ignore').read()


def post(path, body, timeout=3600):
    return urllib.request.urlopen(urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                                         headers={'Content-Type': 'application/json'}), timeout=timeout)


def engine_hist():
    text = urllib.request.urlopen(URL + '/metrics', timeout=30).read().decode()
    out = {}
    for name in ('request_queue_time_seconds', 'request_prefill_time_seconds', 'time_to_first_token_seconds'):
        for kind in ('sum', 'count'):
            v = re.findall(rf'^vllm:{name}_{kind}{{[^}}]*}} ([0-9.e+-]+)', text, re.M)
            out[f'{name}_{kind}'] = sum(float(x) for x in v)
    return out


base_tokens = json.load(post('/tokenize', {'model': MODEL, 'prompt': corpus[:max(PT, LONG) * 6]}, 300))['tokens']
events = [[] for _ in range(N)]
errors, stop, lock = [], threading.Event(), threading.Lock()


def stream(i):
    try:
        start = random.Random(i).randrange(0, len(base_tokens) // 3)
        tokens = (base_tokens[start:] + base_tokens[:start])[:PT]
        body = {'model': MODEL, 'prompt': tokens, 'max_tokens': 16384, 'min_tokens': 16384, 'ignore_eos': True,
                'temperature': 0.7, 'seed': i, 'stream': True, 'cache_salt': uuid.uuid4().hex,
                'stream_options': {'include_usage': True, 'continuous_usage_stats': True}}
        if INJECT == 'error' and i == 0:
            body['model'] = 'no-such-model'
        with post('/v1/completions', body) as r:
            for line in r:
                if stop.is_set() or (INJECT == 'stop' and i == 0 and len(events[i]) > 50):
                    break
                if line.startswith(b'data: {'):
                    u = json.loads(line[6:]).get('usage')
                    if u:
                        events[i].append((time.monotonic(), u['completion_tokens']))
    except Exception as e:  # noqa: BLE001 - fails the run
        with lock:
            errors.append(f'stream {i}: {type(e).__name__} {e}')


wall_anchor = (time.time(), time.monotonic())
threads = [threading.Thread(target=stream, args=(i,), daemon=True) for i in range(N)]
[t.start() for t in threads]
deadline = time.monotonic() + 900
while not all(events) and time.monotonic() < deadline and not errors:
    time.sleep(0.5)
marks, newcomer_token, eng = {}, False, {}


def newcomer_run():
    global newcomer_token, eng
    time.sleep(3)
    marks['b0'] = time.monotonic(); time.sleep(BASE); marks['b1'] = time.monotonic()
    long_tokens = (base_tokens * (LONG // len(base_tokens) + 2))[:LONG]
    h0 = engine_hist()
    marks['p0'] = time.monotonic()
    with post('/v1/completions', {'model': MODEL, 'prompt': long_tokens, 'max_tokens': 1, 'temperature': 0,
                                  'stream': True, 'cache_salt': uuid.uuid4().hex,
                                  'stream_options': {'include_usage': True, 'continuous_usage_stats': True}}) as r:
        for line in r:
            if line.startswith(b'data: {'):
                u = json.loads(line[6:]).get('usage') or {}
                if u.get('completion_tokens', 0) >= 1:   # a validated token advance, not just an event
                    newcomer_token = INJECT != 'missing'
                    marks['p1'] = time.monotonic()
                    break
    marks.setdefault('p1', time.monotonic())
    h1 = engine_hist()
    # attributable only if the newcomer is the one request whose queue/prefill histogram entry landed in the window
    if h1['request_queue_time_seconds_count'] - h0['request_queue_time_seconds_count'] == 1:
        eng = {'engine_queue_s': round(h1['request_queue_time_seconds_sum'] - h0['request_queue_time_seconds_sum'], 3),
               'engine_prefill_s': round(h1['request_prefill_time_seconds_sum'] - h0['request_prefill_time_seconds_sum'], 3)}
    else:
        eng = {'engine_metrics': 'not attributable (other requests entered the histograms in the window)'}
    time.sleep(AFTER)
    marks['a1'] = time.monotonic()


if all(events) and not errors:
    try:
        newcomer_run()
    except Exception as e:  # noqa: BLE001 - fails the run, artifact still saved
        errors.append(f'foreground: {type(e).__name__} {e}')
else:
    errors.append('streams did not all start')
stop.set()
for t in threads:
    t.join(timeout=30)
raw = {'label': LABEL, 'streams': N, 'stream_prompt': PT, 'long_prompt': LONG, 'wall_anchor': wall_anchor,
       'errors': errors, 'marks': marks, 'events': events, 'newcomer_token': newcomer_token, 'engine_newcomer': eng}
if not marks or 'a1' not in marks:
    json.dump({'verdict': 'FAIL', 'fails': errors or ['incomplete run'], 'raw': raw}, open(OUT, 'w'))
    print('FAIL', errors); sys.exit(1)
res = report(raw)
json.dump({**res, 'raw': raw}, open(OUT, 'w'))
print(json.dumps(short(res)))
sys.exit(0 if res['verdict'] == 'PASS' else 1)
