#!/usr/bin/env python3
"""TTFT of one fresh LONG-token prompt with nothing else running (chunk-by-load's case), REPEATS times, unique salts.
usage: fresh_ttft.py URL MODEL LONG OUT.json   (env REPEATS=2)   prints the TTFTs; exit 1 on an error"""
import json, os, sys, time, urllib.request, uuid
URL, MODEL, LONG, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
corpus = open('/home/<service-user>/p2p-tests/corpus.txt', encoding='utf-8', errors='ignore').read()


def post(path, body):
    return urllib.request.urlopen(urllib.request.Request(URL + path, json.dumps(body).encode(),
                                                         {'Content-Type': 'application/json'}), timeout=3600)


toks = json.load(post('/tokenize', {'model': MODEL, 'prompt': corpus[:LONG * 6]}))['tokens']
toks = (toks * (LONG // len(toks) + 2))[:LONG]
res = []
for _ in range(int(os.environ.get('REPEATS', 2))):
    t0 = time.monotonic()
    with post('/v1/completions', {'model': MODEL, 'prompt': toks, 'max_tokens': 1, 'temperature': 0, 'stream': True,
                                  'cache_salt': uuid.uuid4().hex}) as r:
        for line in r:
            if line.startswith(b'data: {') and json.loads(line[6:]).get('choices'):
                break
    res.append(round(time.monotonic() - t0, 2))
json.dump({'long': LONG, 'ttft_s': res}, open(OUT, 'w'))
print('fresh_ttft', LONG, res)
