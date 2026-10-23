#!/usr/bin/env python3
"""Prefix-cache correctness check: per conversation, turn 1 (haystack + question) warms the cache, turn 2 (history +
new question) is sent twice with greedy decoding — once normally (served from the prefix cache) and once with a
random cache_salt (computed from scratch). The two answers must match and be correct. Prompt lengths are random so
the cache boundaries land at arbitrary offsets inside a block. Prints cached-token counts when the server reports
them (--enable-prompt-tokens-details).
usage: salt_ab.py URL MODEL CONVERSATIONS RECORDS"""
import json, random, sys, urllib.request

URL, MODEL, CONV, RECORDS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])


def chat(messages, salt=None):
    body = {"model": MODEL, "messages": messages, "max_tokens": 48, "temperature": 0, "seed": 1,
            "chat_template_kwargs": {"enable_thinking": False}}
    if salt:
        body["cache_salt"] = salt
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    u = d.get("usage") or {}
    return d["choices"][0]["message"]["content"] or "", u.get("prompt_tokens"), (u.get("prompt_tokens_details") or {}).get("cached_tokens")


bad = 0
for c in range(CONV):
    rnd = random.Random(1000 + c)
    n = RECORDS + rnd.randint(0, RECORDS // 3)
    salt = rnd.randint(1, 99999)
    code = lambda r: f"C{(r * 7919 + salt) % 1000003:07d}"
    hay = "\n".join(f"record {r}: code {code(r)}" for r in range(n))
    q1, q2 = rnd.randrange(n), rnd.randrange(n)
    m = [{"role": "user", "content": hay + f"\n\nWhat is the code of record {q1}? Answer with the code only."}]
    a1, p1, _ = chat(m)
    m += [{"role": "assistant", "content": a1}, {"role": "user", "content": f"And record {q2}? Answer with the code only."}]
    warm, p2, cached = chat(m)
    cold, _, cached_cold = chat(m, salt=f"ab-{c}-{random.random()}")
    ok = warm.strip() == cold.strip() and code(q2) in warm and code(q1) in a1
    bad += not ok
    print(f"conv {c}: prompt {p1} -> {p2} tok, cached warm {cached} cold {cached_cold}; turn1 {'ok' if code(q1) in a1 else 'WRONG'}, "
          f"warm {'ok' if code(q2) in warm else 'WRONG'}, warm==cold {warm.strip() == cold.strip()}", flush=True)
print("RESULT", "PASS" if not bad else f"FAIL ({bad} of {CONV})")
sys.exit(1 if bad else 0)
