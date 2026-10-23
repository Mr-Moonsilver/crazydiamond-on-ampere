#!/usr/bin/env python3
"""Decode stall during a fresh long prefill, with per-stream token timelines.

STREAMS streams (PROMPT tokens each, distinct salts, forced long output) generate; after BASE seconds one
fresh LONG-token prompt (1 output token) arrives. Every streamed chunk carries the cumulative completion
token count (continuous_usage_stats), so each stream's token arrival times are known. Reports per window
(before / during the newcomer's prefill / after): aggregate tok/s, the slowest stream's tok/s, and the
inter-arrival gaps of all streams (p50 / p95 / p99 / max, ms), plus the newcomer's time to its token.
usage: stall_trace.py URL MODEL STREAMS PROMPT LONG LABEL OUT.json   (env BASE=20 AFTER=10)
"""
import json, os, random, sys, threading, time, urllib.request, uuid

URL, MODEL, N, PT, LONG, LABEL, OUT = (sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                                       int(sys.argv[5]), sys.argv[6], sys.argv[7])
BASE, AFTER = float(os.environ.get('BASE', 20)), float(os.environ.get('AFTER', 10))
corpus = open('/home/<service-user>/p2p-tests/corpus.txt', encoding='utf-8', errors='ignore').read()


def post(path, body, timeout=3600):
    return urllib.request.urlopen(urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                                         headers={'Content-Type': 'application/json'}), timeout=timeout)


base_tokens = json.load(post('/tokenize', {'model': MODEL, 'prompt': corpus[:max(PT, LONG) * 6]}, 300))['tokens']
events = [[] for _ in range(N)]   # (time, cumulative completion tokens)
errors, stop = [], threading.Event()


def stream(i):
    try:
        start = random.Random(i).randrange(0, len(base_tokens) // 3)
        tokens = (base_tokens[start:] + base_tokens[:start])[:PT]
        body = {'model': MODEL, 'prompt': tokens, 'max_tokens': 16384, 'min_tokens': 16384, 'ignore_eos': True,
                'temperature': 0.7, 'seed': i, 'stream': True, 'cache_salt': uuid.uuid4().hex,
                'stream_options': {'include_usage': True, 'continuous_usage_stats': True}}
        with post('/v1/completions', body) as r:
            for line in r:
                if stop.is_set():
                    break
                if line.startswith(b'data: {'):
                    u = json.loads(line[6:]).get('usage')
                    if u:
                        events[i].append((time.time(), u['completion_tokens']))
    except Exception as e:  # noqa: BLE001 - reported in the result
        errors.append(f'stream {i}: {type(e).__name__} {e}')


def pct(v, q):
    return round(sorted(v)[min(len(v) - 1, int(q * len(v)))] * 1000, 1) if v else None


def window(t0, t1):
    per, gaps = [], []
    for ev in events:
        a = max((n for t, n in ev if t <= t0), default=0)
        b = max((n for t, n in ev if t <= t1), default=0)
        per.append((b - a) / (t1 - t0))
        ts = [t0] + [t for t, _ in ev if t0 < t <= t1] + [t1]   # include the edges: a silent stream is one long gap
        gaps += [y - x for x, y in zip(ts, ts[1:])]
    return {'s': round(t1 - t0, 1), 'agg_tps': round(sum(per), 1), 'min_stream_tps': round(min(per), 2),
            'gap_ms': {'p50': pct(gaps, .5), 'p95': pct(gaps, .95), 'p99': pct(gaps, .99), 'max': pct(gaps, 1)}}


threads = [threading.Thread(target=stream, args=(i,), daemon=True) for i in range(N)]
[t.start() for t in threads]
deadline = time.time() + 900
while not all(events) and time.time() < deadline and not errors:
    time.sleep(0.5)
if not all(events):
    print(json.dumps({'label': LABEL, 'error': 'streams did not start', 'errors': errors})); sys.exit(1)
time.sleep(3)
b0 = time.time(); time.sleep(BASE); b1 = time.time()
long_tokens = (base_tokens * (LONG // len(base_tokens) + 2))[:LONG]
p0 = time.time()
post('/v1/completions', {'model': MODEL, 'prompt': long_tokens, 'max_tokens': 1, 'temperature': 0,
                         'cache_salt': uuid.uuid4().hex}).read()
p1 = time.time()
time.sleep(AFTER)
a1 = time.time()
stop.set()
res = {'label': LABEL, 'streams': N, 'stream_prompt': PT, 'long_prompt': LONG,
       'before': window(b0, b1), 'during': window(p0, p1), 'after': window(p1, a1),
       'newcomer_s': round(p1 - p0, 1), 'newcomer_tps': round(LONG / (p1 - p0)), 'errors': errors}
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(res))
