#!/usr/bin/env python3
"""Image qualification against the engine (run inside CT <ct-id> with a vision-enabled engine).

Modes (all write OUT.json, exit 0 PASS / 1 FAIL / 2 INCONCLUSIVE):
  quality      every manifest item (screenshot small text, table cell, diagram order, natural photos) answered
               correctly; records prompt tokens per image
  bound        big.png (4000 x 3000) must stay within MAX_VISUAL visual tokens, also when the request itself asks for
               mm_processor_kwargs max_pixels 16,777,216 (a per-request override must not lift the bound)
  followup     two-turn conversation on table.png; turn 2 asks another cell, must be right and reuse cache
  isolation    8 concurrent requests, identical text, each with its own code image, two rounds; every answer must
               name its own code and no other
  uuid         the same client-supplied uuid on two different images: a stale answer (first image's code) FAILS;
               PASS needs the second image answered correctly or the uuid request rejected
  interference 8 text decoders (PROMPT tokens) run; after BASE s, IMAGES image requests arrive (concurrently); reports
               per-stream gap p50/p99/max before vs during, and image latency
usage: vision_test.py URL MODEL ASSETS MODE OUT.json   env MAX_VISUAL=1100 PROMPT=60000 BASE=20 IMAGES=4
"""
import base64, json, os, random, sys, threading, time, urllib.error, urllib.request, uuid

URL, MODEL, ASSETS, MODE, OUT = sys.argv[1:6]
MAX_VISUAL = int(os.environ.get('MAX_VISUAL', 1100))
man = json.load(open(os.path.join(ASSETS, 'manifest.json')))


def post(path, body, timeout=900):
    return json.load(urllib.request.urlopen(urllib.request.Request(URL + path, json.dumps(body).encode(),
                                                                   {'Content-Type': 'application/json'}), timeout=timeout))


def img_part(name, extra=None):
    mime = 'image/jpeg' if name.endswith('.jpg') else 'image/png'
    data = base64.b64encode(open(os.path.join(ASSETS, name), 'rb').read()).decode()
    part = {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{data}'}}
    if extra:
        part.update(extra)
    return part


def ask(messages, max_tokens=40, **kw):
    body = {'model': MODEL, 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0,
            'chat_template_kwargs': {'enable_thinking': False}, **kw}
    t0 = time.monotonic(); d = post('/v1/chat/completions', body); t1 = time.monotonic()
    u = d.get('usage') or {}
    return (d['choices'][0]['message']['content'] or '').strip(), u.get('prompt_tokens'), \
        ((u.get('prompt_tokens_details') or {}).get('cached_tokens')), round(t1 - t0, 2)


def norm(s):
    return ''.join(ch for ch in s.lower() if ch.isalnum() or ch == '>')


def text_only_tokens(q):
    return ask([{'role': 'user', 'content': q}], max_tokens=1)[1]


res: dict = {"mode": MODE}
fails: list = []
inconclusive: list = []
try:
    if MODE == 'quality':
        items = []
        for name in ('screenshot.png', 'table.png', 'diagram.png', 'penguin.jpg', 'zebra.jpg'):
            m = man[name]
            a, pt, _, lat = ask([{'role': 'user', 'content': [img_part(name), {'type': 'text', 'text': m['q']}]}])
            ok = norm(m['a']) in norm(a)
            items.append({'image': name, 'expected': m['a'], 'answer': a, 'ok': ok, 'prompt_tokens': pt,
                          'visual_tokens': pt - text_only_tokens(m['q']), 'latency_s': lat})
            if not ok:
                fails.append(f'{name}: expected {m["a"]!r}, got {a!r}')
        res['items'] = items
    elif MODE == 'bound':
        m = man['big.png']; base = text_only_tokens(m['q'])
        a, pt, _, _ = ask([{'role': 'user', 'content': [img_part('big.png'), {'type': 'text', 'text': m['q']}]}])
        res['default'] = {'answer': a, 'visual_tokens': pt - base, 'ok': norm(m['a']) in norm(a)}
        try:
            a2, pt2, _, _ = ask([{'role': 'user', 'content': [img_part('big.png'), {'type': 'text', 'text': m['q']}]}],
                                mm_processor_kwargs={'max_pixels': 16777216})
            res['override'] = {'answer': a2, 'visual_tokens': pt2 - base}
        except urllib.error.HTTPError as e:
            res['override'] = {'rejected': e.code}
        if res['default']['visual_tokens'] > MAX_VISUAL:
            fails.append(f"default visual tokens {res['default']['visual_tokens']} > {MAX_VISUAL}")
        if res['override'].get('visual_tokens', 0) > MAX_VISUAL:
            fails.append(f"per-request max_pixels override lifted the bound to {res['override']['visual_tokens']} visual tokens")
    elif MODE == 'followup':
        q1, q2 = man['table.png']['q'], man['table.png']['followup_q']
        msgs = [{'role': 'user', 'content': [img_part('table.png'), {'type': 'text', 'text': q1}]}]
        a1, pt1, c1, _ = ask(msgs)
        msgs += [{'role': 'assistant', 'content': a1}, {'role': 'user', 'content': q2}]
        a2, pt2, c2, _ = ask(msgs)
        res.update(turn1={'answer': a1, 'ok': man['table.png']['a'] in a1, 'prompt_tokens': pt1},
                   turn2={'answer': a2, 'prompt_tokens': pt2, 'cached_tokens': c2})
        if not res['turn1']['ok']:
            fails.append(f'turn 1 wrong: {a1!r}')
        res['turn2']['expected'] = man['table.png']['followup_a']
        if a2.strip() != man['table.png']['followup_a']:
            fails.append(f"turn 2 wrong: expected {man['table.png']['followup_a']!r}, got {a2!r}")
        if not c2:
            inconclusive.append('turn 2 reused no cache')
    elif MODE == 'isolation':
        iso = man['isolation']; codes = [x['code'] for x in iso]; rounds = []
        for rd in range(2):
            out = [None] * len(iso)

            def one(k):
                try:
                    out[k] = ask([{'role': 'user', 'content': [img_part(iso[k]['file']), {'type': 'text', 'text':
                                   'What is the private image code? Answer with the code only.'}]}])[0]
                except Exception as e:  # noqa: BLE001
                    out[k] = f'ERR {type(e).__name__} {e}'
            th = [threading.Thread(target=one, args=(k,)) for k in range(len(iso))]
            [t.start() for t in th]; [t.join() for t in th]
            for k, a in enumerate(out):
                foreign = [c for c in codes if c != codes[k] and c in (a or '')]
                if foreign:
                    fails.append(f'round {rd} image {k}: foreign code {foreign} in {a!r}')
                elif codes[k] not in (a or ''):
                    fails.append(f'round {rd} image {k}: expected {codes[k]}, got {a!r}')
            rounds.append(out)
        res['rounds'] = rounds; res['codes'] = codes
    elif MODE == 'uuid':
        iso = man['isolation']; u = 'flash-test-' + uuid.uuid4().hex; q = 'What is the private image code? Answer with the code only.'
        r = []
        for k in (0, 1):
            try:
                a = ask([{'role': 'user', 'content': [img_part(iso[k]['file'], {'uuid': u}), {'type': 'text', 'text': q}]}])[0]
            except urllib.error.HTTPError as e:
                a = f'HTTP {e.code}'
            r.append(a)
        res.update(answers=r, expected=[iso[0]['code'], iso[1]['code']], stale_reuse=iso[0]['code'] in r[1])
        if res['stale_reuse']:
            fails.append('a client-supplied uuid made the engine answer for a different, earlier image (stale reuse)')
        elif r[1].startswith('HTTP '):
            res['rejected'] = r[1]
        elif iso[1]['code'] not in r[1]:
            fails.append(f'second image answered {r[1]!r}, expected {iso[1]["code"]}')
    elif MODE == 'interference':
        PT, BASE, NIMG = int(os.environ.get('PROMPT', 60000)), float(os.environ.get('BASE', 20)), int(os.environ.get('IMAGES', 4))
        corpus = open('/home/<service-user>/p2p-tests/corpus.txt', encoding='utf-8', errors='ignore').read()
        base_tok = post('/tokenize', {'model': MODEL, 'prompt': corpus[:PT * 6]}, 600)['tokens']
        events = [[] for _ in range(8)]; stop = threading.Event(); errs = []

        def stream(i):
            try:
                s = random.Random(i).randrange(0, len(base_tok) // 3)
                body = {'model': MODEL, 'prompt': (base_tok[s:] + base_tok[:s])[:PT], 'max_tokens': 8192, 'min_tokens': 8192,
                        'ignore_eos': True, 'temperature': 0.7, 'seed': i, 'stream': True, 'cache_salt': uuid.uuid4().hex,
                        'stream_options': {'include_usage': True, 'continuous_usage_stats': True}}
                with urllib.request.urlopen(urllib.request.Request(URL + '/v1/completions', json.dumps(body).encode(),
                                                                   {'Content-Type': 'application/json'}), timeout=3600) as r:
                    for line in r:
                        if stop.is_set():
                            return
                        if line.startswith(b'data: {'):
                            u = json.loads(line[6:]).get('usage') or {}
                            if u.get('completion_tokens'):
                                events[i].append((time.monotonic(), u['completion_tokens']))
            except Exception as e:  # noqa: BLE001
                errs.append(f'stream {i}: {e}')
        th = [threading.Thread(target=stream, args=(i,), daemon=True) for i in range(8)]; [t.start() for t in th]
        end = time.monotonic() + 1200
        while not all(events) and time.monotonic() < end and not errs:
            time.sleep(0.5)
        time.sleep(3)

        def gaps(t0, t1):
            # advance-based: only events whose token count grew; the gap that crosses t0 and the open gap to t1 count
            per = []
            for ev in events:
                adv, last = [], -1
                for t, n in ev:
                    if n > last:
                        adv.append(t); last = n
                pre = [t for t in adv if t <= t0][-1:]
                ts = pre + [t for t in adv if t0 < t <= t1]
                g = [y - x for x, y in zip(ts, ts[1:])]
                if ts:
                    g.append(t1 - ts[-1])
                g = sorted(g)
                per.append({'p50': round(g[len(g) // 2] * 1000, 1), 'p99': round(g[min(len(g) - 1, int(.99 * len(g)))] * 1000, 1),
                            'max': round(g[-1] * 1000, 1)} if g else None)
            return per
        b0 = time.monotonic(); time.sleep(BASE); b1 = time.monotonic()
        names = ['screenshot.png', 'table.png', 'diagram.png', 'penguin.jpg', 'zebra.jpg']
        lat = [None] * NIMG

        def img(k):
            try:
                nm = names[k % len(names)]
                lat[k] = ask([{'role': 'user', 'content': [img_part(nm), {'type': 'text', 'text': man[nm]['q']}]}])[3]
            except Exception as e:  # noqa: BLE001
                errs.append(f'image {k}: {type(e).__name__} {e}')
        i0 = time.monotonic(); it = [threading.Thread(target=img, args=(k,)) for k in range(NIMG)]
        [t.start() for t in it]; [t.join() for t in it]; i1 = time.monotonic()
        stop.set(); [t.join(timeout=30) for t in th]
        if any(x is None for x in lat):
            errs.append(f'{sum(x is None for x in lat)} image requests without a latency')
        # a stream must keep advancing through the image window (an open gap to the window end counts)
        for i, ev in enumerate(events):
            ts = [t for t, _ in ev if t <= i1]
            if not ts or i1 - ts[-1] > 2.0:
                errs.append(f'stream {i} stalled: no token in the last {round(i1 - (ts[-1] if ts else b0), 1)} s of the image window')
        res.update(before=gaps(b0, b1), during=gaps(i0, i1), image_window_s=round(i1 - i0, 2), image_latency_s=lat, errors=errs)
        worst = lambda per, k: max(p[k] for p in per if p)
        res['worst_before'] = {k: worst(res['before'], k) for k in ('p50', 'p99', 'max')}
        res['worst_during'] = {k: worst(res['during'], k) for k in ('p50', 'p99', 'max')}
        fails += errs
    else:
        fails.append(f'unknown mode {MODE}')
except Exception as e:  # noqa: BLE001
    fails.append(f'foreground: {type(e).__name__} {e}')
res.update(fails=fails, inconclusive=inconclusive, verdict='FAIL' if fails else ('INCONCLUSIVE' if inconclusive else 'PASS'))
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(res)[:1500])
sys.exit({'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[res['verdict']])
