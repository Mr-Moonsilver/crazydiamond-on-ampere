#!/usr/bin/env python3
"""Quick concurrent decode probe: N streams, fixed prompt length, forced output length.

Reports aggregate decode tok/s over the interval in which every stream is generating
(0.5 s trimmed at each end), per-stream rate, TTFT spread, and actual prompt tokens.
usage: quick_c8.py URL MODEL STREAMS PROMPT_TOKENS OUTPUT_TOKENS LABEL OUT.json
"""
import json, os, sys, threading, time, urllib.request, uuid, random

URL, MODEL = sys.argv[1], sys.argv[2]
N, PT, OT = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
LABEL, OUT = sys.argv[6], sys.argv[7]
CORPUS = open('/home/<service-user>/p2p-tests/corpus.txt', encoding='utf-8', errors='ignore').read()


def tokenize(text):
    req = urllib.request.Request(URL + '/tokenize', data=json.dumps({'model': MODEL, 'prompt': text}).encode(),
                                 headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=120))['tokens']


base = tokenize(CORPUS[:PT * 6])[:PT]
events = [[] for _ in range(N)]
# Optional early stop: once every stream has produced a token, keep going for WINDOW seconds, then
# disconnect (vLLM aborts the requests). The common all-stream window is all that is measured, so the
# tail after it only costs time.
WINDOW = float(os.environ.get('QUICK_WINDOW_S', '0'))
all_started = [None]
meta = [{} for _ in range(N)]
barrier = threading.Barrier(N)


def run(i):
    rnd = random.Random(i)
    start = rnd.randrange(0, len(base) // 2)
    tokens = base[start:] + base[:start]  # distinct rotation per stream: no shared prefix
    body = {'model': MODEL, 'prompt': tokens, 'max_tokens': OT, 'min_tokens': OT, 'ignore_eos': True,
            'temperature': 0.7, 'seed': i, 'stream': True,
            'stream_options': {'include_usage': True, 'continuous_usage_stats': True},
            'cache_salt': uuid.uuid4().hex}
    req = urllib.request.Request(URL + '/v1/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    barrier.wait()
    t0 = time.time()
    meta[i]['t0'] = t0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b'data: ') or line == b'data: [DONE]':
                continue
            d = json.loads(line[6:])
            if d.get('choices'):
                meta[i].setdefault('text', []).append(d['choices'][0].get('text') or '')
            u = d.get('usage')
            if u:
                events[i].append((time.time(), u['completion_tokens']))
                meta[i]['prompt_tokens'] = u['prompt_tokens']
            if WINDOW:
                if all_started[0] is None and all(events):
                    all_started[0] = time.time()
                if all_started[0] is not None and time.time() > all_started[0] + WINDOW + 1:
                    break


ths = [threading.Thread(target=run, args=(i,)) for i in range(N)]
[t.start() for t in ths]
[t.join() for t in ths]

first = [ev[0][0] for ev in events]
last = [ev[-1][0] for ev in events]
lo, hi = max(first) + 0.5, min(last) - 0.5


def count_at(ev, t):
    c = 0
    for ts, n in ev:
        if ts <= t:
            c = n
        else:
            break
    return c


rates = [(count_at(ev, hi) - count_at(ev, lo)) / (hi - lo) for ev in events] if hi > lo else None
res = {'label': LABEL, 'streams': N, 'prompt_tokens': [m.get('prompt_tokens') for m in meta], 'output_tokens': OT,
       'ttft_s': [round(f - m['t0'], 2) for f, m in zip(first, meta)],
       'burst_s': round(max(last) - min(m['t0'] for m in meta), 1),
       'common_window_s': round(hi - lo, 1),
       'aggregate_decode_tps': round(sum(rates), 1) if rates else None,
       'per_stream_tps': [round(r, 1) for r in rates] if rates else None}
def repetition(text):
    # share of 8-word windows that already occurred earlier in the same output (1.0 = pure loop)
    words = text.split()
    seen, repeats = set(), 0
    grams = [tuple(words[k:k + 8]) for k in range(max(0, len(words) - 7))]
    for g in grams:
        repeats += g in seen
        seen.add(g)
    return round(repeats / len(grams), 3) if grams else None


texts = [''.join(m.get('text', [])) for m in meta]
res['repetition_8gram'] = [repetition(t) for t in texts]
res['text_head'] = [t[:300] for t in texts]
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('prompt_tokens', 'text_head')}))
