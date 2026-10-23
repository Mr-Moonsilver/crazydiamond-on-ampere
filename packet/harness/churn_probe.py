#!/usr/bin/env python3
"""Does the prefix cache turn over per REQUEST rather than per token? Caches a ~TOKENS probe, then sends N tiny unique
requests (~300 tokens, max_tokens 1, 4 at a time), then re-sends the probe WAIT s after its first send and prints
cached tokens. Run with N=0 as the control.
usage: churn_probe.py URL MODEL TOKENS WAIT N [N ...]"""
import concurrent.futures, json, random, sys, time, urllib.request

URL, MODEL, TOKENS, WAIT = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])


def send(text):
    body = {"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": 1, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    u = json.load(urllib.request.urlopen(req, timeout=1800)).get("usage") or {}
    return u.get("prompt_tokens"), (u.get("prompt_tokens_details") or {}).get("cached_tokens")


def text(tag, rows):
    return f"probe {tag}\n" + "\n".join(f"line {r}: value {(r * 7919 + hash(tag)) % 1000003}" for r in range(rows))


for n in map(int, sys.argv[5:]):
    tag = f"{random.randrange(1 << 30)}"
    p = text(tag, TOKENS // 12)
    t0 = time.time(); pt, _ = send(p)
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda i: send(text(f"{tag}-tiny-{i}", 25)), range(n)))
    burst = time.time() - t0
    time.sleep(max(0, t0 + WAIT - time.time()))
    _, ct = send(p)
    print(f"N={n:3} tiny requests (burst took {burst:.0f} s): probe re-sent after {time.time() - t0:.0f} s -> cached {ct} of {pt}", flush=True)
