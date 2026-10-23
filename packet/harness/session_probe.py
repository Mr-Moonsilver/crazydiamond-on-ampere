#!/usr/bin/env python3
"""Agent-like retention test. A probe session sends turn 1 (~BASE tokens) and goes idle; K background sessions keep
taking follow-up turns (each turn adds ~STEP tokens to its own history) until N background turns have run; then the
probe sends turn 2 and we print how much of its history was still cached. Repeats for each N.
usage: session_probe.py URL MODEL K BASE STEP N [N ...]"""
import json, random, sys, threading, time, urllib.request

URL, MODEL, K, BASE, STEP = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])


def send(msgs):
    body = {"model": MODEL, "messages": msgs, "max_tokens": 1, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    u = json.load(urllib.request.urlopen(req, timeout=1800)).get("usage") or {}
    return u.get("prompt_tokens"), (u.get("prompt_tokens_details") or {}).get("cached_tokens")


def blob(tag, n):
    return "\n".join(f"{tag} row {r}: value {(r * 7919 + hash(tag)) % 1000003}" for r in range(n // 12))


class Session:
    def __init__(self, tag):
        self.tag, self.turn = tag, 0
        self.msgs = [{"role": "system", "content": f"session {tag}"}, {"role": "user", "content": blob(tag, BASE)}]

    def step(self):
        pt, ct = send(self.msgs)
        self.turn += 1
        self.msgs += [{"role": "assistant", "content": f"ack {self.tag} {self.turn}"},
                      {"role": "user", "content": blob(f"{self.tag}-{self.turn}", STEP)}]
        return pt, ct


for n in map(int, sys.argv[6:]):
    run = random.randrange(1 << 30)
    probe = Session(f"probe{run}")
    bg = [Session(f"bg{run}-{i}") for i in range(K)]
    for s in bg:
        s.step()
    probe.step()
    t0, done, lock, hits = time.time(), [0], threading.Lock(), []

    def worker(s):
        while True:
            with lock:
                if done[0] >= n:
                    return
                done[0] += 1
            pt, ct = s.step()
            hits.append((ct or 0) / pt)

    ts = [threading.Thread(target=worker, args=(s,)) for s in bg]
    [t.start() for t in ts]; [t.join() for t in ts]
    pt, ct = probe.step()
    print(f"N={n:3} background turns over {time.time() - t0:.0f} s (their mean cached share {sum(hits) / max(1, len(hits)):.2f}): "
          f"probe turn 2 cached {ct} of {pt}", flush=True)
