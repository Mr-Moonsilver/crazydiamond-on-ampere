#!/usr/bin/env python3
"""Drive a torch-profiler capture of mixed prefill/decode steps.

STREAMS decoding streams (PROMPT tokens, distinct salts) run; a fresh LONG-token prompt arrives; DELAY seconds
into its prefill the profiler is started (POST /start_profile, the server records its configured number of
active iterations), then stopped after HOLD seconds. Requests are cancelled at the end.
usage: prof_drive.py URL MODEL STREAMS PROMPT LONG [DELAY HOLD]
"""
import json, random, sys, threading, time, urllib.request, uuid

URL, MODEL, N, PT, LONG = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
DELAY, HOLD = (float(sys.argv[6]), float(sys.argv[7])) if len(sys.argv) > 7 else (6.0, 8.0)
corpus = open('/home/<service-user>/p2p-tests/corpus.txt', encoding='utf-8', errors='ignore').read()
stop = threading.Event()


def post(path, body=None, timeout=3600):
    data = json.dumps(body).encode() if body is not None else b''
    return urllib.request.urlopen(urllib.request.Request(URL + path, data=data, headers={'Content-Type': 'application/json'},
                                                         method='POST'), timeout=timeout)


base = json.load(post('/tokenize', {'model': MODEL, 'prompt': corpus[:max(PT, LONG) * 6]}, 300))['tokens']
started = [False] * N


def stream(i):
    s = random.Random(i).randrange(0, len(base) // 3)
    body = {'model': MODEL, 'prompt': (base[s:] + base[:s])[:PT], 'max_tokens': 16384, 'min_tokens': 16384, 'ignore_eos': True,
            'stream': True, 'cache_salt': uuid.uuid4().hex}
    with post('/v1/completions', body) as r:
        for _ in r:
            started[i] = True
            if stop.is_set():
                return


def newcomer():
    try:
        post('/v1/completions', {'model': MODEL, 'prompt': (base * (LONG // len(base) + 2))[:LONG], 'max_tokens': 1,
                                 'cache_salt': uuid.uuid4().hex}).read()
    except Exception:  # noqa: BLE001 - cancelled at the end
        pass


for i in range(N):
    threading.Thread(target=stream, args=(i,), daemon=True).start()
t0 = time.time()
while not all(started):
    if time.time() - t0 > 600:
        sys.exit('streams did not start')
    time.sleep(0.5)
time.sleep(3)
threading.Thread(target=newcomer, daemon=True).start()
time.sleep(DELAY)
print('start_profile', post('/start_profile').status, flush=True)
time.sleep(HOLD)
print('stop_profile', post('/stop_profile', timeout=600).status, flush=True)
stop.set()
