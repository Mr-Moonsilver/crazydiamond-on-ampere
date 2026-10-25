#!/usr/bin/env python3
"""E2 exactness driver: one fresh >=200k token prompt, then a long warm-prefix append on the same cache_salt.

Gives the per-worker oracle (FLASH_E2_VERIFY=1) its coverage: a dense >=200k multibucket staged history on every
QSA layer (fresh run) and a prefix-hit staged invocation (append: the cached prefix is restarted with computed > 0).
Token ids come from /tokenize of the local corpus (as fresh_probe.py), so the prompt length is exact. max_tokens is
small on purpose; this is a coverage driver, never a performance measurement (the oracle synchronizes every bucket).
usage: e2_exact_drive.py URL MODEL OUT.json [FRESH_TOKENS=201000] [APPEND_TOKENS=3000]
exit 0 when the usage evidence proves both shapes (fresh: cached 0, exact length; append: cached >= fresh-4096),
1 otherwise (never a verdict on bytes: that is e2_exact_grade.py on the worker sidecars).
"""
import json, sys, time, urllib.request, uuid
from pathlib import Path

CORPUS = '/home/<service-user>/p2p-tests/corpus.txt'


def post(url, path, body, timeout=1800):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def completion(url, model, toks, salt):
    t = time.monotonic()
    r = post(url, '/v1/completions', {'model': model, 'prompt': toks, 'max_tokens': 8, 'min_tokens': 8,
                                      'ignore_eos': True, 'temperature': 0, 'cache_salt': salt})
    u = r['usage']
    return dict(seconds=round(time.monotonic() - t, 3), prompt_tokens=u['prompt_tokens'],
                completion_tokens=u['completion_tokens'],
                cached_tokens=(u.get('prompt_tokens_details') or {}).get('cached_tokens'),
                finish_reason=r['choices'][0]['finish_reason'])


def main(url, model, out, fresh_n='201000', append_n='3000'):
    fresh_n, append_n = int(fresh_n), int(append_n)
    out = Path(out)
    if out.exists():
        raise SystemExit('output exists; use a new run directory')
    text = Path(CORPUS).read_text()
    toks = post(url, '/tokenize', {'model': model, 'prompt': text[:1200000]})['tokens']
    assert toks
    toks = (toks * ((fresh_n + append_n) // len(toks) + 1))[:fresh_n + append_n]
    salt = 'e2-exact-' + uuid.uuid4().hex
    fresh = completion(url, model, toks[:fresh_n], salt)
    append = completion(url, model, toks, salt)
    why = []
    if fresh['prompt_tokens'] != fresh_n or fresh['cached_tokens'] != 0 or fresh['completion_tokens'] != 8:
        why.append('fresh run not a cold exact-length prefill')
    if (append['prompt_tokens'] != fresh_n + append_n or append['cached_tokens'] is None or
            append['cached_tokens'] < fresh_n - 4096 or append['cached_tokens'] > fresh_n or append['completion_tokens'] != 8):
        why.append('append did not start from the cached fresh prefix')
    rec = dict(fresh_tokens=fresh_n, append_tokens=append_n, fresh=fresh, append=append, ok=not why, why=why,
               scope='coverage driver for the E2 byte oracle; no timing claim')
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))
    raise SystemExit(0 if not why else 1)


if __name__ == '__main__':
    main(*sys.argv[1:])
