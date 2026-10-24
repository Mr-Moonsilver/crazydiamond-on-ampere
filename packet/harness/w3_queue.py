#!/usr/bin/env python3
"""W3: all request slots decoding, extra requests waiting for a slot -- does decode keep its speed?

SLOTS streams (PROMPT tokens, distinct salts, forced long output) fill every request slot. After all decode, a
BASE-second baseline window is measured; then EXTRA more requests are submitted (they cannot be admitted) and a
second window of the same length is measured while they wait. With a correct scheduler both windows decode at the
same rate; a policy that counts unadmittable queue entries as prefill work empties every other step (about half).
Waiting is proven from the engine (vllm:num_requests_waiting sampled every second in the window must stay >= EXTRA),
not from the absence of output. Gaps are between token advances (repeated counts ignored), plus each stream's open
gap at the window end. FAIL (exit 1) on any worker error (also after the windows), a stream that stops, extra requests
not proven waiting, or decode in the waiting window < MIN_RATIO x baseline. The artifact is always written.
usage: w3_queue.py URL MODEL SLOTS PROMPT EXTRA OUT.json   env BASE=30 MIN_RATIO=0.9
"""
import json, os, random, sys, threading, time, urllib.request, uuid

URL, MODEL, SLOTS, PT, EXTRA, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
BASE, MIN_RATIO = float(os.environ.get('BASE', 30)), float(os.environ.get('MIN_RATIO', 0.9))
corpus = open('/home/<service-user>/p2p-tests/corpus.txt', encoding='utf-8', errors='ignore').read()


def post(path, body, timeout=3600):
    return urllib.request.urlopen(urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                                         headers={'Content-Type': 'application/json'}), timeout=timeout)


base = json.load(post('/tokenize', {'model': MODEL, 'prompt': corpus[:PT * 6]}, 300))['tokens']
events = [[] for _ in range(SLOTS + EXTRA)]
errors, stop, lock = [], threading.Event(), threading.Lock()


def stream(i):
    try:
        s = random.Random(i).randrange(0, len(base) // 3)
        body = {'model': MODEL, 'prompt': (base[s:] + base[:s])[:PT], 'max_tokens': 16384, 'min_tokens': 16384,
                'ignore_eos': True, 'temperature': 0.7, 'seed': i, 'stream': True, 'cache_salt': uuid.uuid4().hex,
                'stream_options': {'include_usage': True, 'continuous_usage_stats': True}}
        with post('/v1/completions', body) as r:
            for line in r:
                if stop.is_set():
                    break
                if line.startswith(b'data: {'):
                    u = json.loads(line[6:]).get('usage')
                    if u and u.get('completion_tokens'):
                        events[i].append((time.monotonic(), u['completion_tokens']))
    except Exception as e:  # noqa: BLE001 - fails the run
        with lock:
            errors.append(f'stream {i}: {type(e).__name__} {e}')


def waiting_now():
    text = urllib.request.urlopen(URL + '/metrics', timeout=10).read().decode()
    vals = [float(l.split()[-1]) for l in text.splitlines() if l.startswith('vllm:num_requests_waiting{')]
    return sum(vals)


def advances(ev):
    out, last = [], 0
    for t, n in ev:
        if n > last:
            out.append((t, n)); last = n
    return out


def window(t0, t1, idx):
    per = []
    for i in idx:
        ev = advances(events[i])
        a = max((n for t, n in ev if t <= t0), default=0)
        b = max((n for t, n in ev if t <= t1), default=0)
        times = [t for t, _ in ev]
        gaps = [y - x for x, y in zip(times, times[1:]) if t0 < y <= t1]
        last = max((t for t in times if t <= t1), default=None)
        if last is not None:
            gaps.append(t1 - max(last, t0))
        gaps.sort()
        per.append({'tps': round((b - a) / (t1 - t0), 2),
                    'gap_p50_ms': round(gaps[len(gaps) // 2] * 1000, 1) if gaps else None,
                    'gap_max_ms': round(gaps[-1] * 1000, 1) if gaps else None})
    return {'agg_tps': round(sum(p['tps'] for p in per), 1), 'min_stream_tps': min(p['tps'] for p in per),
            'per_stream': per}


ths = [threading.Thread(target=stream, args=(i,), daemon=True) for i in range(SLOTS)]
[t.start() for t in ths]
deadline = time.monotonic() + 1800
while not all(events[:SLOTS]) and time.monotonic() < deadline and not errors:
    time.sleep(0.5)
fails = []
res: dict = {'slots': SLOTS, 'prompt': PT, 'extra': EXTRA}
extra_threads = []


def measure():
    time.sleep(5)
    b0 = time.monotonic(); time.sleep(BASE); b1 = time.monotonic()
    for j in range(SLOTS, SLOTS + EXTRA):
        t = threading.Thread(target=stream, args=(j,), daemon=True); t.start(); extra_threads.append(t)
    time.sleep(2)
    w0 = time.monotonic(); samples = []
    while time.monotonic() < w0 + BASE:
        samples.append(waiting_now()); time.sleep(1)
    w1 = time.monotonic()
    res['waiting_samples'] = samples
    if not samples or min(samples) < EXTRA:
        fails.append(f'extra requests not proven waiting (engine waiting counts {samples[:5]}...)')
    res['baseline'] = window(b0, b1, range(SLOTS))
    res['waiting'] = window(w0, w1, range(SLOTS))
    res['ratio'] = round(res['waiting']['agg_tps'] / res['baseline']['agg_tps'], 3)
    for i in range(SLOTS):
        if not any(t > w1 - 2 for t, _ in events[i]):
            fails.append(f'stream {i} stopped')
    if res['ratio'] < MIN_RATIO:
        fails.append(f"decode while requests wait is {res['ratio']} x baseline (< {MIN_RATIO})")


if all(events[:SLOTS]) and not errors:
    try:
        measure()
    except Exception as e:  # noqa: BLE001 - fails the run, artifact still written
        fails.append(f'foreground: {type(e).__name__} {e}')
else:
    fails.append('streams did not all start')
stop.set()
for t in ths + extra_threads:
    t.join(timeout=30)
fails = list(errors) + fails   # worker errors from any point of the run
res['errors'], res['fails'] = errors, fails
res['verdict'] = 'FAIL' if fails else 'PASS'
json.dump({**res, 'raw': events}, open(OUT, 'w'))
print(json.dumps({k: (v if k not in ('baseline', 'waiting') else {x: v[x] for x in ('agg_tps', 'min_stream_tps')})
                  for k, v in res.items()}))
sys.exit(0 if res['verdict'] == 'PASS' else 1)
