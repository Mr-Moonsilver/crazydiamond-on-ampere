#!/usr/bin/env python3
"""Background KV-pool pressure for lifetime tests (supersedes pressure.py for the v3 gate; pressure.py is unchanged for
the queues that ran it): N long streams with unique prompts (unique salts, nothing shared) loop fresh requests until
SECONDS have passed or the file STOP exists, so cached prefix blocks get evicted and physical blocks reused while a
leak test runs next to it. Differences from pressure.py: the per-request token bound is PRESSURE_MAX_TOKENS (default
4096 = the old behaviour; the v3 gate sets 128 so every stream completes requests inside the phase) and is recorded in
OUT.json; KV occupancy (kv_cache_usage_perc), running/waiting request counts and the preemption counter are sampled
every 2 s and summarised (min/max/mean, samples) so a shorter generation cannot silently remove the pressure; the
progress file (PRESSURE_READY) also carries the per-stream completed counts and last token times for a bounded
completion wait (pressure_wait.py). Records the engine's counters at start and end, plus any stream errors.
usage: pressure2.py URL MODEL N PROMPT SECONDS OUT.json [STOPFILE]   exit 0 if no stream errored
"""
import json, os, random, sys, threading, time, urllib.request, uuid

URL, MODEL, N, PT, SECS, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
STOP = sys.argv[7] if len(sys.argv) > 7 else None
MAX_TOKENS = int(os.environ.get('PRESSURE_MAX_TOKENS', '4096'))
samples = {'kv_cache_usage_perc': [], 'num_requests_running': [], 'num_requests_waiting': [], 'num_preemptions_total': []}
corpus = open('/home/<service-user>/p2p-tests/corpus.txt', encoding='utf-8', errors='ignore').read()
errors, done = [], threading.Event()


def post(path, body, timeout=3600):
    return urllib.request.urlopen(urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                                         headers={'Content-Type': 'application/json'}), timeout=timeout)


def metrics():
    text = urllib.request.urlopen(URL + '/metrics', timeout=10).read().decode()
    out = {}
    for l in text.splitlines():
        for k in ('num_preemptions_total', 'kv_cache_usage_perc', 'prefix_cache_queries_total', 'prefix_cache_hits_total',
                  'num_requests_running', 'num_requests_waiting'):
            if l.startswith(f'vllm:{k}{{') or l.startswith(f'vllm:{k} '):
                out[k] = out.get(k, 0) + float(l.split()[-1])
    return out


base = json.load(post('/tokenize', {'model': MODEL, 'prompt': corpus[:PT * 6]}, 600))['tokens']
if len(base) < PT:
    raise SystemExit('pressure corpus is shorter than requested prompt')
counts = [0] * N
first_tokens = [None] * N
last_tokens = [None] * N
issued = [0] * N
started_wall = time.time()
progress_path = os.environ.get('PRESSURE_READY')


def save_progress():
    if progress_path:
        tmp = progress_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'first_tokens': first_tokens, 'last_tokens': last_tokens, 'issued': issued,
                       'completed_requests': counts, 'started_wall': started_wall, 'wall': time.time(),
                       'max_tokens': MAX_TOKENS, 'samples': {k: len(v) for k, v in samples.items()}}, f)
        os.replace(tmp, progress_path)


def sample():
    try:
        m = metrics()
    except Exception as e:  # noqa: BLE001
        errors.append(f'metrics sample: {type(e).__name__} {e}'); return
    for k, v in samples.items():
        if k in m:
            v.append(m[k])


def summary(v):
    return {'samples': len(v), 'min': min(v), 'max': max(v), 'mean': sum(v) / len(v)} if v else {'samples': 0}



def stream(i):
    k = 0
    while not done.is_set():
        try:
            s = random.Random(i * 1000 + k).randrange(0, len(base) // 2)
            body = {'model': MODEL, 'prompt': (base[s:] + base[:s])[:PT], 'max_tokens': MAX_TOKENS, 'temperature': 0.7,
                    'stream': True, 'cache_salt': uuid.uuid4().hex}
            issued[i] += 1
            terminal = False
            with post('/v1/completions', body) as r:
                for raw in r:
                    if done.is_set():
                        return
                    if raw.startswith(b'data: ') and raw.strip() != b'data: [DONE]':
                        data = json.loads(raw[6:])
                        for choice in data.get('choices', []):
                            if choice.get('text'):
                                now = time.time()
                                if first_tokens[i] is None:
                                    first_tokens[i] = now
                                last_tokens[i] = now
                            terminal |= choice.get('finish_reason') is not None
            if not terminal:
                raise RuntimeError('pressure stream ended without a finish reason')
            counts[i] += 1; k += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f'pressure {i}: {type(e).__name__} {e}'); time.sleep(5)


m0 = metrics()
ths = [threading.Thread(target=stream, args=(i,), daemon=True) for i in range(N)]
[t.start() for t in ths]
end = time.monotonic() + SECS
while time.monotonic() < end and not (STOP and os.path.exists(STOP)):
    sample()
    save_progress()
    time.sleep(2)
done.set()
save_progress()
m1 = metrics()
res = {'started_wall': started_wall, 'ended_wall': time.time(), 'first_tokens': first_tokens,
       'last_tokens': last_tokens, 'issued': issued, 'streams': N, 'prompt': PT, 'max_tokens': MAX_TOKENS,
       'completed_requests': counts, 'metrics_start': m0, 'metrics_end': m1,
       'occupancy': {k: summary(v) for k, v in samples.items()},
       'prefix_cache_hits_delta': m1.get('prefix_cache_hits_total', 0) - m0.get('prefix_cache_hits_total', 0),
       'preemptions': m1.get('num_preemptions_total', 0) - m0.get('num_preemptions_total', 0), 'errors': errors}
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('metrics_start',)}))
sys.exit(1 if errors else 0)
