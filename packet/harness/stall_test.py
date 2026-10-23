#!/usr/bin/env python3
"""Decode stall during someone else's long prefill (agentic-coding shape).

Starts STREAMS streaming requests (PROMPT tokens, long forced output). Once all of them generate, measures
their aggregate rate for BASE seconds, then submits one fresh LONG-token prompt (1 output token) and measures
the running streams' aggregate rate while that prefill runs, plus the prefill's own time to first token.
usage: stall_test.py URL MODEL STREAMS PROMPT LONG OUT.json
"""
import json, random, sys, threading, time, urllib.request, uuid

URL, MODEL, N, PT, LONG, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
BASE = 20.0
corpus = open('/home/<service-user>/p2p-tests/corpus.txt', encoding='utf-8', errors='ignore').read()


def tokenize(text):
    req = urllib.request.Request(URL + '/tokenize', data=json.dumps({'model': MODEL, 'prompt': text}).encode(),
                                 headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=300))['tokens']


base_tokens = tokenize(corpus[:max(PT, LONG) * 6])
events = [[] for _ in range(N)]
stop = threading.Event()


def stream(i):
    start = random.Random(i).randrange(0, len(base_tokens) // 3)
    tokens = (base_tokens[start:] + base_tokens[:start])[:PT]
    body = {'model': MODEL, 'prompt': tokens, 'max_tokens': 16384, 'min_tokens': 16384, 'ignore_eos': True,
            'temperature': 0.7, 'seed': i, 'stream': True,
            'stream_options': {'include_usage': True, 'continuous_usage_stats': True}, 'cache_salt': uuid.uuid4().hex}
    req = urllib.request.Request(URL + '/v1/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if stop.is_set():
                break
            line = line.strip()
            if line.startswith(b'data: ') and line != b'data: [DONE]':
                u = json.loads(line[6:]).get('usage')
                if u:
                    events[i].append((time.time(), u['completion_tokens']))


def tokens_between(t0, t1):
    total = 0
    for ev in events:
        a = max((n for t, n in ev if t <= t0), default=0)
        b = max((n for t, n in ev if t <= t1), default=0)
        total += b - a
    return total


threads = [threading.Thread(target=stream, args=(i,), daemon=True) for i in range(N)]
[t.start() for t in threads]
while not all(events):
    time.sleep(0.5)
time.sleep(3)
b0 = time.time(); time.sleep(BASE); b1 = time.time()
long_tokens = (base_tokens * (LONG // len(base_tokens) + 2))[:LONG]
body = {'model': MODEL, 'prompt': long_tokens, 'max_tokens': 1, 'temperature': 0, 'cache_salt': uuid.uuid4().hex}
p0 = time.time()
urllib.request.urlopen(urllib.request.Request(URL + '/v1/completions', data=json.dumps(body).encode(),
                                              headers={'Content-Type': 'application/json'}), timeout=3600).read()
p1 = time.time()
time.sleep(1)
a0 = time.time(); time.sleep(BASE); a1 = time.time()  # recovery window after the long prefill
stop.set()
res = {'streams': N, 'stream_prompt': PT, 'long_prompt': LONG,
       'baseline_tps': round(tokens_between(b0, b1) / (b1 - b0), 1),
       'during_prefill_tps': round(tokens_between(p0, p1) / (p1 - p0), 1),
       'long_prefill_s': round(p1 - p0, 1), 'long_prefill_tps': round(LONG / (p1 - p0), 1),
       'after_prefill_tps': round(tokens_between(a0, a1) / (a1 - a0), 1),
       'first_5s_after_tps': round(tokens_between(p1, p1 + 5) / 5, 1)}
res['slowdown_factor'] = round(res['baseline_tps'] / max(res['during_prefill_tps'], 0.1), 1)
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(res))
