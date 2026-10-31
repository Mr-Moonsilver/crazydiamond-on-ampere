#!/usr/bin/env python3
"""q18b-w1ab W1 client: the 24 Oct stall_trace2.py W1 phase, W1 only, repeated windows.

Derived by copy from fn-main/harness/stall_trace2.py (sha256 8c6237fd...). Unchanged from it:
corpus, /tokenize, per-stream prompt rotation (random.Random(i)), fresh uuid4 cache salts,
request body (max_tokens=min_tokens=16384, ignore_eos, temperature 0.7, seed i, streamed with
continuous_usage_stats), "all streams have events, sleep 3 s, then window", and the window()
semantics (advances, gaps ending in the window plus each stream's open gap, worst stream by p99).
Changed: no newcomer / small-arrival request; WINDOWS consecutive windows of WINDOW_S seconds
(repeats) on the same 8 decoding streams; the bearer is taken from VLLM_API_KEY in this process
environment (as the P-c W1 client did) and sent only to a loopback URL; the key is never printed
or stored. Each window's wall-clock bounds are stored so the grader can match engine log lines.
usage: w1_trace.py URL MODEL STREAMS PROMPT LABEL OUT.json   env WINDOWS=3 WINDOW_S=20 SETTLE_S=3
       CORPUS=/home/<service-user>/p2p-tests/corpus.txt START_DEADLINE_S=900
       w1_trace.py --recompute OUT.json
"""
import json, os, random, sys, threading, time, urllib.parse, urllib.request, uuid


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


def wall(raw, t):
    w, m = raw['wall_anchor']
    return round(w + (t - m), 6)


def report(raw):
    res = {k: raw[k] for k in ('label', 'streams', 'stream_prompt', 'wall_anchor', 'errors')}
    res['windows'] = []
    for i, (t0, t1) in enumerate(raw['marks']['windows']):
        row = window(raw, t0, t1)
        row.update(repeat=i, wall_t0=wall(raw, t0), wall_t1=wall(raw, t1))
        res['windows'].append(row)
    res['first_event_s'] = sorted(round(ev[0][0] - raw['wall_anchor'][1], 1) for ev in raw['events'] if ev)
    fails = list(raw['errors'])
    for row in res['windows']:
        if row['pooled_gap_ms']['n'] < raw['streams'] * 5:
            fails.append(f"too few advances in window {row['repeat']} ({row['pooled_gap_ms']['n']})")
    if len(res['windows']) != raw['repeats']:
        fails.append('incomplete windows')
    res['fails'] = fails
    res['verdict'] = 'FAIL' if fails else 'PASS'
    return res


def short(res):
    out = {k: v for k, v in res.items() if k != 'windows'}
    out['windows'] = [{k: v for k, v in w.items() if k != 'per_stream'} for w in res.get('windows', [])]
    return out


if len(sys.argv) > 1 and sys.argv[1] == '--recompute':
    raw = json.load(open(sys.argv[2]))['raw']
    res = report(raw)
    print(json.dumps(short(res)))
    sys.exit(0 if res['verdict'] == 'PASS' else 1)

URL, MODEL, N, PT, LABEL, OUT = (sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                                 sys.argv[5], sys.argv[6])
WINDOWS, WINDOW_S = int(os.environ.get('WINDOWS', 3)), float(os.environ.get('WINDOW_S', 20))
SETTLE_S = float(os.environ.get('SETTLE_S', 3))
START_DEADLINE_S = float(os.environ.get('START_DEADLINE_S', 900))
CORPUS = os.environ.get('CORPUS', '/home/<service-user>/p2p-tests/corpus.txt')
_host = urllib.parse.urlsplit(URL).hostname
if _host not in ('127.0.0.1', 'localhost', '::1'):
    print('refused: bearer is only sent to a loopback URL'); sys.exit(2)
KEY = os.environ.get('VLLM_API_KEY') or ''
HEADERS = {'Content-Type': 'application/json'}
if KEY:
    HEADERS['Authorization'] = 'Bearer ' + KEY
del KEY
corpus = open(CORPUS, encoding='utf-8', errors='ignore').read()


def post(path, body, timeout=3600):
    return urllib.request.urlopen(urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                                         headers=HEADERS), timeout=timeout)


# same tokenized corpus slice as the 24 Oct runs (stall_trace2 LONG=200000), so prompts are identical
LONG_REF = 200000
base_tokens = json.load(post('/tokenize', {'model': MODEL, 'prompt': corpus[:max(PT, LONG_REF) * 6]}, 300))['tokens']
events = [[] for _ in range(N)]
salts = [uuid.uuid4().hex for _ in range(N)]


def stream_tokens(i):
    start = random.Random(i).randrange(0, len(base_tokens) // 3)
    return (base_tokens[start:] + base_tokens[:start])[:PT]
errors, stop, lock = [], threading.Event(), threading.Lock()


def scrub(text):
    # an exception text never carries the bearer, but keep the artifact free of it regardless
    return text.replace(HEADERS.get('Authorization', '\0'), '***')


def stream(i):
    try:
        tokens = stream_tokens(i)
        body = {'model': MODEL, 'prompt': tokens, 'max_tokens': 16384, 'min_tokens': 16384, 'ignore_eos': True,
                'temperature': 0.7, 'seed': i, 'stream': True, 'cache_salt': salts[i],
                'stream_options': {'include_usage': True, 'continuous_usage_stats': True}}
        with post('/v1/completions', body) as r:
            for line in r:
                if stop.is_set():
                    break
                if line.startswith(b'data: {'):
                    u = json.loads(line[6:]).get('usage')
                    if u:
                        events[i].append((time.monotonic(), u['completion_tokens']))
    except Exception as e:  # noqa: BLE001 - fails the run
        with lock:
            errors.append(scrub(f'stream {i}: {type(e).__name__} {e}'))


wall_anchor = (time.time(), time.monotonic())
threads = [threading.Thread(target=stream, args=(i,), daemon=True) for i in range(N)]
[t.start() for t in threads]
deadline = time.monotonic() + START_DEADLINE_S
while not all(events) and time.monotonic() < deadline and not errors:
    time.sleep(0.5)
marks = {'windows': []}
if all(events) and not errors:
    marks['all_started'] = time.monotonic()
    time.sleep(SETTLE_S)
    for _ in range(WINDOWS):
        t0 = time.monotonic(); time.sleep(WINDOW_S); t1 = time.monotonic()
        marks['windows'].append((t0, t1))
        if errors:
            break
else:
    errors.append('streams did not all start')
stop.set()
for t in threads:
    t.join(timeout=30)
raw = {'label': LABEL, 'streams': N, 'stream_prompt': PT, 'repeats': WINDOWS, 'window_s': WINDOW_S,
       'wall_anchor': wall_anchor, 'errors': errors, 'marks': marks, 'events': events}
res = report(raw)
with open(OUT, 'w') as f:
    json.dump({**res, 'raw': raw}, f)
print(json.dumps(short(res)))
sys.exit(0 if res['verdict'] == 'PASS' else 1)
