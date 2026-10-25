#!/usr/bin/env python3
"""Cross-session / partial-copy leak test shaped like the 2026-10-23 agent incidents (partial prefix-cache hits).

With --prefix-match-unit U < block size, a request whose prompt ends < U tokens past a hash boundary B registers a
partial-tail entry at B; the next request of the conversation hits B inside a block, and the scheduler gives it a
private copy (CoW) of that block. MODE=inside (default) checks that the copied history survives: each session's
prompt = SHARED (first full block, identical for all) + session-unique filler with a secret fact at ~FACT tokens
(inside the copied part [block start, B)) + filler up to exactly B + a short tail; request 1 generates GEN tokens,
request 2 asks for the fact, which is never repeated after B. MODE=after puts the secret after B instead (cross-session
shared tail entry; the fact is recomputed, so it only tests state hand-over, not the copy).
Each graded request also runs cold (unique cache_salt). Verdicts and the path-recreating retries: see leakgrade.py.
The output JSON keeps every prompt's token ids, answers, hit lengths, retries and the engine container identity at
start and end. Exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE.
UNIQUE_PREFIX=1 (MODE=inside): every session gets its own first block, so long runs fill the KV pool (preemption).
CANCEL_FRAC=f: that share of sessions first opens request 2 as a stream and drops the connection 0-300 ms in (an
abort around the copy submission), then asks again; the re-asked answer is graded. NEED_PARTIAL=0 for whole-block
control runs (B a multiple of BLOCK: no partial copy expected).
usage: leak_partial_tail.py URL MODEL SESSIONS ROUNDS [B] [OUT.json]
       env: MODE=inside GEN=300 UNIT=176 BLOCK=4048 FACT=5000 FACT2=6800 SEED=1 CANCEL_FRAC=0 NEED_PARTIAL=1 TIMEOUT=900
Note: if request 1's output crosses the page end (p1 + gen1 > next block boundary), the stock hybrid lookup returns the
previous full block instead of the partial tail (cache-efficiency gap, 24 Oct); keep GEN small (e.g. 200 for B=7568)
when partial-copy coverage is the point of the run.
"""
import http.client, json, os, random, subprocess, sys, threading, time, urllib.parse, urllib.request, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leakgrade  # noqa: E402

URL, MODEL, N, ROUNDS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
B = int(sys.argv[5]) if len(sys.argv) > 5 else 43 * 176
OUT = sys.argv[6] if len(sys.argv) > 6 else None
GEN, UNIT, COLD = int(os.environ.get("GEN", "300")), int(os.environ.get("UNIT", "176")), os.environ.get("COLD", "1") == "1"
MODE, FACT, BLOCK = os.environ.get("MODE", "inside"), int(os.environ.get("FACT", "5000")), int(os.environ.get("BLOCK", "4048"))
FACT2 = int(os.environ.get("FACT2", "6800"))  # MODE=inside: 2nd fact, asked only by request 3 (chained copies)
CANCEL_FRAC = float(os.environ.get("CANCEL_FRAC", "0"))
NEED_PARTIAL = os.environ.get("NEED_PARTIAL", "1") == "1"
UNIQUE_PREFIX = os.environ.get("UNIQUE_PREFIX", "0") == "1"  # MODE=inside: no shared first block (fills the pool)
CACHE_NS = os.environ.get("CACHE_NS", "")  # warm requests of a session share cache_salt CACHE_NS-round-session: identical
# token streams (same SEED) in two runs with different CACHE_NS values are equivalent fresh namespaces (matched CoW sets)
cancelled = []
rnd = random.Random(int(os.environ.get("SEED", "1")))
corpus = open('/home/<service-user>/p2p-tests/corpus.txt', encoding='utf-8', errors='ignore').read()


HTTP_TIMEOUT = float(os.environ.get("TIMEOUT", "900"))


def post(path, body, timeout=None):
    timeout = timeout or HTTP_TIMEOUT
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


LOCKERS = {}  # code -> unique locker id, drawn when the code is drawn


def locker_of(code):
    return LOCKERS[code]


def engine_identity():
    try:
        return subprocess.run(['docker', 'inspect', 'qwen-flash-prod', '--format', '{{.Id}} {{.State.StartedAt}}'],
                              capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:  # noqa: BLE001 - identity unknown is recorded as None
        return None


def abort_after(prompt, seconds, salt=None):
    # send a streamed request and drop the connection `seconds` after submission, without waiting for headers
    # (the engine sees a client abort while the request is being scheduled / its copy is being submitted)
    u = urllib.parse.urlparse(URL)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=900)
    body = {'model': MODEL, 'prompt': prompt, 'max_tokens': 12, 'temperature': 0, 'stream': True}
    if salt:
        body['cache_salt'] = salt
    body = json.dumps(body)
    conn.request('POST', '/v1/completions', body, {'Content-Type': 'application/json'})
    time.sleep(seconds)
    conn.close()


def run_session(rd, i, shared, code, delay):
    time.sleep(delay)
    if MODE == "after":
        priv = tok(f"\n\nSession note: the secret code of this session is {code}. Keep it in mind. Now continue the text:\n")
        assert len(priv) < UNIT - 8, len(priv)
        p1 = shared + priv
    else:
        if UNIQUE_PREFIX:
            off = random.Random(rd * 31337 + i).randrange(0, len(corpus) - B * 6 - 1)
            shared = tok(f"[round {rd} session {i} {uuid.uuid4().hex}]\n" + corpus[off:off + B * 6])[:B]
        p1 = build_inside(shared, i, rd, code, locker_of(code))
        assert len(p1) - B < UNIT - 8
    ns = f'{CACHE_NS}-{rd}-{i}' if CACHE_NS else None
    # matched mode: the generation budget is per-session deterministic (the shared rnd stream is thread-order dependent)
    budget = GEN + (random.Random(rd * 48611 + i).randrange(0, GEN) if CACHE_NS else rnd.randrange(0, GEN))
    gen1, ids1, _ = complete(p1, budget, salt=ns, ignore_eos=True)
    p2 = p1 + list(ids1) + QUESTION
    if random.Random(rd * 7919 + i).random() < CANCEL_FRAC:
        abort_after(p2, random.Random(rd * 104729 + i).uniform(0, 0.3), salt=ns)
        with lock:
            cancelled.append((rd, i))
    a, ids_a, det = complete(p2, 12, salt=ns, ignore_eos=True)
    cold = complete(p2, 12, salt=uuid.uuid4().hex, ignore_eos=True)[0] if COLD else None
    rec = {'round': rd, 'session': i, 'code': code, 'answer': a, 'cold': cold, 'cached': det.get('cached_tokens'),
           'p2_len': len(p2), 'p1_ids': p1, 'gen1': gen1, 'gen1_ids': list(ids1), 'p2': p2}
    if MODE == "inside":
        # request 3: continue from request 2 (its answer may repeat the code, so ask for the 2nd fact)
        p3 = p2 + list(ids_a) + tok("\n\n") + list(ids1[:200]) + QUESTION2
        a3, _, det3 = complete(p3, 12, salt=ns, ignore_eos=True)
        cold3 = complete(p3, 12, salt=uuid.uuid4().hex, ignore_eos=True)[0] if COLD else None
        rec.update(locker=locker_of(code), answer3=a3, cold3=cold3, cached3=det3.get('cached_tokens'), p3=p3)
    with lock:
        results.append(rec)


def session(rd, i, shared, code, delay):
    try:
        run_session(rd, i, shared, code, delay)
    except Exception as e:  # noqa: BLE001 - counted as a failure below
        with lock:
            errors.append(f"r{rd} s{i}: {type(e).__name__}: {e}")


codes_all = []
identity_start = engine_identity()
for rd in range(ROUNDS):
    off = rnd.randrange(0, len(corpus) - 200000)
    shared = tok(f"[round {rd} {uuid.uuid4().hex}]\n" + corpus[off:off + B * 6])[:B]
    assert len(shared) == B
    codes = []
    while len(codes) < N:
        c = f"{rnd.choice(NAMES)}-{rnd.randint(1000, 9999)}"
        if c not in codes and all(c not in cs for cs in codes_all):
            codes.append(c)
            while True:
                lk = f"L{rnd.randrange(16**6):06x}"
                if lk not in LOCKERS.values():
                    LOCKERS[c] = lk
                    break
    codes_all.append(codes)
    # MODE=after: warm the shared prefix once so every session's 1st request hits the same full blocks
    if MODE == 'after':
        complete(shared + tok("\n"), 1)
    th = [threading.Thread(target=session, args=(rd, i, shared, codes[i], i * 1.5)) for i in range(N)]
    [t.start() for t in th]
    [t.join() for t in th]

identity_end = engine_identity()
every_code = [c for cs in codes_all for c in cs]


def retries_for(prompt, hit, n=3):
    # recreate the original cache path: prime a fresh salt with prompt[:hit+1], then ask under that salt
    out = []
    for _ in range(n):
        salt = uuid.uuid4().hex
        if hit:
            complete(prompt[:hit + 1], 1, salt=salt)
        warm, _, d = complete(prompt, 12, salt=salt, ignore_eos=True)
        cold = complete(prompt, 12, salt=uuid.uuid4().hex, ignore_eos=True)[0]
        out.append({'warm': warm, 'warm_cached': d.get('cached_tokens'), 'cold': cold})
    return out


def grade_one(r, ans_key, cold_key, cached_key, prompt_key, own, ids):
    foreign = [x for x in ids if x != own]
    first = leakgrade.first_verdict(r[ans_key], own, foreign, r.get(cold_key))
    case = {'first': first, 'settled': None, 'partial': bool(r.get(cached_key)) and r[cached_key] % BLOCK != 0}
    if first == 'SUSPECT':
        r[ans_key + '_retries'] = retries_for(r[prompt_key], r.get(cached_key) or 0)
        case['settled'] = leakgrade.settle(r[ans_key + '_retries'], own, foreign, r.get(cached_key))
    r[ans_key + '_first'], r[ans_key + '_settled'] = first, case['settled']
    return case


cases = []
every_id = every_code + [locker_of(c) for c in every_code]
for r in sorted(results, key=lambda x: (x['round'], x['session'])):
    # request 1's own continuation is generated text too: a foreign identifier there is a leak
    own_ids = (r['code'], locker_of(r['code']))
    bad = leakgrade.foreign_in(r.get('gen1'), [x for x in every_id if x not in own_ids])
    if bad:
        errors.append(f"r{r['round']} s{r['session']}: LEAK in request 1 output {bad}")
    cases.append(grade_one(r, 'answer', 'cold', 'cached', 'p2', r['code'], every_code))
    if 'answer3' in r:
        cases.append(grade_one(r, 'answer3', 'cold3', 'cached3', 'p3', r['locker'], [locker_of(c) for c in every_code]))
    for k in ('answer', 'answer3'):
        if r.get(k + '_first') not in (None, 'OK'):
            print(f"r{r['round']} s{r['session']} {k} first={r[k + '_first']} settled={r[k + '_settled']} "
                  f"cached={r.get('cached' if k == 'answer' else 'cached3')} warm={r[k][:40]!r} "
                  f"cold={(r.get('cold' if k == 'answer' else 'cold3') or '')[:40]!r}", flush=True)
expected = N * ROUNDS * (2 if MODE == 'inside' else 1)
run = leakgrade.run_verdict(cases, expected, errors, restarted=identity_start != identity_end or identity_start is None,
                            need_partial=MODE == 'inside' and NEED_PARTIAL)
run['cancelled'] = len(cancelled)
if CANCEL_FRAC and not cancelled:
    run['inconclusive'].append('no request was cancelled')
    run['verdict'] = 'INCONCLUSIVE' if run['verdict'] == 'PASS' else run['verdict']
for e in errors:
    print("ERROR", e, flush=True)
seen = sorted({r['cached'] for r in results}) + sorted({r['cached3'] for r in results if 'cached3' in r})
print(f"MODE {MODE} sessions {N} rounds {ROUNDS} B {B}: {run['verdict']} answers {len(cases)}/{expected} "
      f"{run['counts']} errors {len(errors)}; hits seen {seen[:8]}; {run['fails']} {run['inconclusive']}")
if OUT:
    json.dump({'MODE': MODE, 'B': B, 'N': N, 'ROUNDS': ROUNDS, 'GEN': GEN, 'identity_start': identity_start,
               'identity_end': identity_end, 'run': run, 'results': results, 'errors': errors}, open(OUT, 'w'))
sys.exit({'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[run['verdict']])
