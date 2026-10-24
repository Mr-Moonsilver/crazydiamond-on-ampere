#!/usr/bin/env python3
"""Whole-block (4,048-token) shared-state test: the path of the two unexplained warm-only failures (J-inside r1 s8 and
release-v3 after r1 s9, both cached_tokens = 4048).

Each round: SHARED = one full first block (4,048 tokens) identical for all sessions; every session sends ONE request:
SHARED + its own suffix (~SUFFIX tokens of text with a unique fact) + a question, generating LONG tokens first in a
streamed prelude request so producers keep running while others hit block 0. Each graded prompt is asked warm (shared
cache -> its hit is exactly the first block plus nothing else) and cold (unique salt). If the cached first block
(GPU recurrent-state checkpoint + host QSA page) is intact, warm-only and cold-only failures are equally likely;
a significant excess of warm-only failures (one-sided binomial p < 0.01 over the discordant pairs) fails the test.
Foreign facts in any answer fail it outright.
usage: block0_test.py URL MODEL SESSIONS ROUNDS OUT.json   env SUFFIX=2500 LONG=400 SEED=5
exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE (fewer than 90 % of answers on a 4,048 hit)
"""
import json, math, os, random, sys, threading, urllib.request, uuid

URL, MODEL, N, ROUNDS, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
SUFFIX, LONG = int(os.environ.get('SUFFIX', 2500)), int(os.environ.get('LONG', 400))
rnd = random.Random(int(os.environ.get('SEED', 5)))
corpus = open('/home/<service-user>/p2p-tests/corpus.txt', encoding='utf-8', errors='ignore').read()
BLOCK = 4048


def post(path, body, timeout=900):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def tok(t):
    return post('/tokenize', {'model': MODEL, 'prompt': t, 'add_special_tokens': False})['tokens']


def complete(prompt, n, salt=None):
    body = {'model': MODEL, 'prompt': prompt, 'max_tokens': n, 'min_tokens': n, 'ignore_eos': True, 'temperature': 0}
    if salt:
        body['cache_salt'] = salt
    d = post('/v1/completions', body)
    return d['choices'][0]['text'], ((d.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens')


Q = tok("\n\nQuestion: Earlier in this document an archive note gave a vault code. What exactly was that vault code? "
        "Answer: The vault code is")
results, errors, codes_all, lock = [], [], set(), threading.Lock()


def session(rd, i, shared, code):
    try:
        off = random.Random(rd * 1000 + i).randrange(0, len(corpus) - 100000)
        filler = tok(corpus[off:off + SUFFIX * 8])[:SUFFIX]
        fact = tok(f"\n\nArchive note: the vault code for this document is {code}. (End of archive note.)\n\n")
        a = SUFFIX // 3
        prompt = shared + filler[:a] + fact + filler[a:] + Q
        # producer: the shared block + unrelated text, generating while this session's graded request hits the block
        o2 = random.Random(rd * 7777 + i).randrange(0, len(corpus) - 100000)
        prod = threading.Thread(target=complete, args=(shared + tok(corpus[o2:o2 + 4000]), LONG), daemon=True)
        prod.start()
        threading.Event().wait(random.Random(rd * 31 + i).uniform(0.2, 2.0))
        warm, hit = complete(prompt, 12)
        cold, _ = complete(prompt, 12, salt=uuid.uuid4().hex)
        prod.join(timeout=600)
        with lock:
            results.append({'round': rd, 'session': i, 'code': code, 'warm': warm, 'cold': cold, 'hit': hit})
    except Exception as e:  # noqa: BLE001
        with lock:
            errors.append(f'r{rd} s{i}: {type(e).__name__} {e}')


for rd in range(ROUNDS):
    o = rnd.randrange(0, len(corpus) - 100000)
    shared = tok(f'[block0 round {rd} {uuid.uuid4().hex}]\n' + corpus[o:o + BLOCK * 6])[:BLOCK]
    codes = []
    while len(codes) < N:
        c = f"{rnd.choice(['AMBER', 'COBALT', 'JUNIPER', 'OSPREY', 'TUNDRA', 'VIOLET'])}-{rnd.randint(1000, 9999)}"
        if c not in codes_all:
            codes.append(c); codes_all.add(c)
    th = [threading.Thread(target=session, args=(rd, i, shared, codes[i])) for i in range(N)]
    [t.start() for t in th]; [t.join() for t in th]

warm_only = cold_only = both_bad = foreign = on_block = 0
for r in results:
    w, c = r['code'] in r['warm'], r['code'] in r['cold']
    f = [x for x in codes_all if x != r['code'] and (x in r['warm'] or x in r['cold'])]
    foreign += bool(f)
    warm_only += (not w and c); cold_only += (w and not c); both_bad += (not w and not c)
    on_block += r['hit'] == BLOCK
k, n = warm_only, warm_only + cold_only
p = sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n if n else 1.0   # P(>= k warm-only | fair coin)
fails = list(errors)
if len(results) != N * ROUNDS:
    fails.append(f'{len(results)} of {N * ROUNDS} results')
if foreign:
    fails.append(f'{foreign} answers name a foreign code')
if p < 0.01:
    fails.append(f'warm-only failures {warm_only} vs cold-only {cold_only} (one-sided p={p:.4f})')
inconclusive = [] if on_block >= 0.9 * max(len(results), 1) else [f'only {on_block} of {len(results)} warm answers hit exactly {BLOCK}']
res = {'answers': len(results), 'warm_only': warm_only, 'cold_only': cold_only, 'both_bad': both_bad, 'p_one_sided': round(p, 4),
       'on_block_hits': on_block, 'hits_seen': sorted({r['hit'] for r in results}), 'fails': fails, 'inconclusive': inconclusive,
       'verdict': 'FAIL' if fails else ('INCONCLUSIVE' if inconclusive else 'PASS'), 'results': results, 'errors': errors}
json.dump(res, open(OUT, 'w'))
print(json.dumps({k: v for k, v in res.items() if k not in ('results',)}))
sys.exit({'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[res['verdict']])
