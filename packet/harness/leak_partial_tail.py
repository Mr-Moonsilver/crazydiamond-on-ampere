#!/usr/bin/env python3
"""Cross-session / partial-copy leak test shaped like the 2026-10-23 agent incidents (partial prefix-cache hits).

With --prefix-match-unit U < block size, a request whose prompt ends < U tokens past a hash boundary B registers a
partial-tail entry at B; the next request of the conversation hits B inside a block, and the scheduler gives it a
private copy (CoW) of that block. MODE=inside (default) checks that the copied history survives: each session's
prompt = SHARED (first full block, identical for all) + session-unique filler with a secret fact at ~FACT tokens
(inside the copied part [block start, B)) + filler up to exactly B + a short tail; request 1 generates GEN tokens,
request 2 asks for the fact, which is never repeated after B. MODE=after puts the secret after B instead (cross-session
shared tail entry; the fact is recomputed, so it only tests state hand-over, not the copy).
Each request 2 also runs cold (unique cache_salt). Verdicts: LEAK = names another session's code (any round);
COPYFAIL = warm wrong while cold right; MISS = both wrong (model). Exit 1 on LEAK, COPYFAIL, errors or missing results.
usage: leak_partial_tail.py URL MODEL SESSIONS ROUNDS [B] [OUT.json]   env: MODE=inside GEN=300 UNIT=176 FACT=5500 SEED=1
"""
import json, os, random, re, sys, threading, time, urllib.request, uuid

URL, MODEL, N, ROUNDS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
B = int(sys.argv[5]) if len(sys.argv) > 5 else 43 * 176
OUT = sys.argv[6] if len(sys.argv) > 6 else None
GEN, UNIT, COLD = int(os.environ.get("GEN", "300")), int(os.environ.get("UNIT", "176")), os.environ.get("COLD", "1") == "1"
MODE, FACT, BLOCK = os.environ.get("MODE", "inside"), int(os.environ.get("FACT", "5000")), int(os.environ.get("BLOCK", "4048"))
FACT2 = int(os.environ.get("FACT2", "6800"))  # MODE=inside: 2nd fact, asked only by request 3 (chained copies)
rnd = random.Random(int(os.environ.get("SEED", "1")))
corpus = open('/home/<service-user>/p2p-tests/corpus.txt', encoding='utf-8', errors='ignore').read()


def post(path, body, timeout=900):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def tok(text):
    return post('/tokenize', {'model': MODEL, 'prompt': text, 'add_special_tokens': False})['tokens']


def complete(prompt, max_tokens, salt=None, ignore_eos=False):
    body = {'model': MODEL, 'prompt': prompt, 'max_tokens': max_tokens, 'temperature': 0, 'return_token_ids': True}
    if ignore_eos:
        body.update(ignore_eos=True, min_tokens=max_tokens)
    if salt:
        body['cache_salt'] = salt
    d = post('/v1/completions', body)
    c = d['choices'][0]
    return c['text'], c.get('token_ids') or tok(c['text']), (d.get('usage') or {}).get('prompt_tokens_details') or {}


NAMES = ['AMBER', 'COBALT', 'JUNIPER', 'OSPREY', 'TUNDRA', 'VIOLET', 'MAGPIE', 'QUARTZ', 'SIENNA', 'FALCON', 'HAZEL', 'RIVET']
QUESTION = tok("\n\nQuestion: What is the secret code of this session, exactly as stated in the session note? "
               "Answer: The secret code is") if MODE == "after" else tok(
    "\n\nQuestion: Earlier in this document an archive note gave a vault code. What exactly was that vault code? "
    "Answer: The vault code is")
TAIL = tok("\n\nNow continue the text:\n")
QUESTION2 = tok("\n\nQuestion: Earlier in this document a registry note gave a locker number. What exactly was that "
                "locker number? Answer: The locker number is")
results, errors, lock = [], [], threading.Lock()


def build_inside(shared, i, rd, code, locker):
    # session-unique history from the first block boundary on, facts at ~FACT and ~FACT2, filler up to exactly B
    off = random.Random(rd * 1000 + i).randrange(0, len(corpus) - 200000)
    filler = tok(corpus[off:off + (B - BLOCK) * 8])
    fact = tok(f"\n\nArchive note: the vault code for this document is {code}. (End of archive note.)\n\n")
    fact2 = tok(f"\n\nRegistry note: the locker number for this document is {locker}. (End of registry note.)\n\n")
    a, b = FACT - BLOCK, FACT2 - BLOCK
    body = (filler[:a] + fact + filler[a:b] + fact2 + filler[b:])[:B - BLOCK]
    text = post('/detokenize', {'model': MODEL, 'tokens': body})['prompt']
    assert len(body) == B - BLOCK and code in text and locker in text
    return shared[:BLOCK] + body + TAIL


def locker_of(code):
    return f"L{int(code[-4:]) * 7 % 90000 + 10000}"


def run_session(rd, i, shared, code, delay):
    time.sleep(delay)
    if MODE == "after":
        priv = tok(f"\n\nSession note: the secret code of this session is {code}. Keep it in mind. Now continue the text:\n")
        assert len(priv) < UNIT - 8, len(priv)
        p1 = shared + priv
    else:
        p1 = build_inside(shared, i, rd, code, locker_of(code))
        assert len(p1) - B < UNIT - 8
    _, ids1, _ = complete(p1, GEN + rnd.randrange(0, GEN), ignore_eos=True)
    p2 = p1 + list(ids1) + QUESTION
    a, ids_a, det = complete(p2, 12, ignore_eos=True)
    cold = complete(p2, 12, salt=uuid.uuid4().hex, ignore_eos=True)[0] if COLD else None
    rec = {'round': rd, 'session': i, 'code': code, 'answer': a, 'cold': cold, 'cached': det.get('cached_tokens'),
           'p2_len': len(p2), '_p2': p2}
    if MODE == "inside":
        # request 3: continue from request 2 (its answer may repeat the code, so ask for the 2nd fact)
        p3 = p2 + list(ids_a) + tok("\n\n") + list(ids1[:200]) + QUESTION2
        a3, _, det3 = complete(p3, 12, ignore_eos=True)
        cold3 = complete(p3, 12, salt=uuid.uuid4().hex, ignore_eos=True)[0] if COLD else None
        rec.update(locker=locker_of(code), answer3=a3, cold3=cold3, cached3=det3.get('cached_tokens'), _p3=p3)
    with lock:
        results.append(rec)


def session(rd, i, shared, code, delay):
    try:
        run_session(rd, i, shared, code, delay)
    except Exception as e:  # noqa: BLE001 - counted as a failure below
        with lock:
            errors.append(f"r{rd} s{i}: {type(e).__name__}: {e}")


codes_all = []
for rd in range(ROUNDS):
    off = rnd.randrange(0, len(corpus) - 200000)
    shared = tok(f"[round {rd} {uuid.uuid4().hex}]\n" + corpus[off:off + B * 6])[:B]
    assert len(shared) == B
    codes = []
    while len(codes) < N:
        c = f"{rnd.choice(NAMES)}-{rnd.randint(1000, 9999)}"
        if c not in codes and all(c not in cs for cs in codes_all):
            codes.append(c)
    codes_all.append(codes)
    # MODE=after: warm the shared prefix once so every session's 1st request hits the same full blocks
    if MODE == 'after':
        complete(shared + tok("\n"), 1)
    th = [threading.Thread(target=session, args=(rd, i, shared, codes[i], i * 1.5)) for i in range(N)]
    [t.start() for t in th]
    [t.join() for t in th]

def code_of(text):
    m = re.search(r"[A-Z]{4,8}-\d{4}", text)
    return m.group(0) if m else text.strip()[:20]


every_code = [c for cs in codes_all for c in cs]


def settle(prompt, want):
    # A warm-wrong / cold-right disagreement: re-ask both paths 3 more times. COPYFAIL only if the cached path fails
    # every retry while the cold path mostly succeeds; anything mixed is NOISE (reported, never dropped).
    warm, cold = [], []
    for _ in range(3):
        t, _, d = complete(prompt, 12, ignore_eos=True)
        warm.append((want in t, d.get('cached_tokens')))
        cold.append(want in complete(prompt, 12, salt=uuid.uuid4().hex, ignore_eos=True)[0])
    return ('COPYFAIL' if not any(ok for ok, _ in warm) and sum(cold) >= 2 else 'NOISE'), warm, cold


leak = miss = div = copyfail = noise = 0
for r in results:
    own = r['code'] in r['answer']
    foreign = [c for c in every_code if c != r['code'] and c in r['answer']]
    cold_ok = COLD and r['code'] in (r['cold'] or '')
    r['verdict'] = 'LEAK' if foreign else ('OK' if own else ('COPYFAIL' if cold_ok else 'MISS'))
    if r['verdict'] == 'COPYFAIL':
        r['verdict'], r['retry_warm'], r['retry_cold'] = settle(r['_p2'], r['code'])
    leak += bool(foreign); miss += r['verdict'] == 'MISS'; copyfail += r['verdict'] == 'COPYFAIL'
    noise += r['verdict'] == 'NOISE'
    if COLD and r['cold'] is not None and code_of(r['cold']) != code_of(r['answer']):
        div += 1; r['diverged'] = True
    if 'answer3' in r:
        own3 = r['locker'] in r['answer3']
        foreign3 = [locker_of(c) for c in every_code if c != r['code'] and locker_of(c) in r['answer3']]
        r['verdict3'] = 'LEAK' if foreign3 else ('OK' if own3 else ('COPYFAIL' if r['locker'] in (r['cold3'] or '') else 'MISS'))
        if r['verdict3'] == 'COPYFAIL':
            r['verdict3'], r['retry3_warm'], r['retry3_cold'] = settle(r['_p3'], r['locker'])
        leak += bool(foreign3); miss += r['verdict3'] == 'MISS'; copyfail += r['verdict3'] == 'COPYFAIL'
        noise += r['verdict3'] == 'NOISE'
        if r['verdict3'] != 'OK':
            print(f"r{r['round']} s{r['session']} req3 {r['verdict3']} cached={r['cached3']} locker={r['locker']} "
                  f"warm={r['answer3'][:40]!r} cold={(r['cold3'] or '')[:40]!r}", flush=True)
    if r['verdict'] != 'OK' or r.get('diverged'):
        print(f"r{r['round']} s{r['session']} {r['verdict']} {r.get('retry_warm', '')}{r.get('retry_cold', '')} cached={r['cached']} code={r['code']} "
              f"warm={r['answer'][:40]!r} cold={(r['cold'] or '')[:40]!r}", flush=True)
cached = sorted({r['cached'] for r in results})
for e in errors:
    print("ERROR", e, flush=True)
c3 = sorted({r['cached3'] for r in results if 'cached3' in r})
print(f"MODE {MODE} sessions {N} rounds {ROUNDS} B {B}: answers {len(results)}/{N * ROUNDS} "
      f"(+{sum('answer3' in r for r in results)} 3rd requests, cached {c3[:6]}) errors {len(errors)} "
      f"LEAK {leak} COPYFAIL {copyfail} NOISE {noise} MISS {miss} warm/cold diverged {div}; "
      f"cached_tokens seen {cached[:6]}{'...' if len(cached) > 6 else ''}")
if OUT:
    for r in results:
        r.pop('_p2', None); r.pop('_p3', None)
    json.dump({'MODE': MODE, 'B': B, 'N': N, 'ROUNDS': ROUNDS, 'GEN': GEN, 'results': results, 'errors': errors},
              open(OUT, 'w'), indent=1)
sys.exit(1 if leak or copyfail or errors or len(results) != N * ROUNDS else 0)
