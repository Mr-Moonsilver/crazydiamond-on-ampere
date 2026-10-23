#!/usr/bin/env python3
"""How long does an idle prefix survive in the prefix cache under the current load? Sends N unique prompts
(~TOKENS each, max_tokens 1), then re-sends prompt i once after DELAYS[i] seconds and reads cached_tokens
(server must run with --enable-prompt-tokens-details). Each prompt is re-sent only once, so the first send's
blocks age untouched.
usage: retention_probe.py URL MODEL TOKENS DELAY [DELAY ...]"""
import json, random, sys, threading, time, urllib.request

URL, MODEL, TOKENS = sys.argv[1], sys.argv[2], int(sys.argv[3])
DELAYS = [float(x) for x in sys.argv[4:]]


def send(text):
    body = {"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": 1, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    u = json.load(urllib.request.urlopen(req, timeout=1800)).get("usage") or {}
    return u.get("prompt_tokens"), (u.get("prompt_tokens_details") or {}).get("cached_tokens")


tag = random.randrange(1 << 30)
prompts = [f"probe {tag}-{i}\n" + "\n".join(f"line {r}: value {(r * 7919 + i * 31 + tag) % 1000003}" for r in range(TOKENS // 12))
           for i in range(len(DELAYS))]
done = {}
for i, p in enumerate(prompts):
    pt, ct = send(p)
    done[i] = time.time()
    print(f"prompt {i}: {pt} tokens sent (cached {ct}) at {time.strftime('%H:%M:%S', time.gmtime())}", flush=True)


def resend(i):
    time.sleep(max(0, done[i] + DELAYS[i] - time.time()))
    pt, ct = send(prompts[i])
    print(f"prompt {i}: re-sent after {time.time() - done[i]:.0f} s -> cached {ct} of {pt}", flush=True)


ts = [threading.Thread(target=resend, args=(i,)) for i in range(len(DELAYS))]
[t.start() for t in ts]
[t.join() for t in ts]
