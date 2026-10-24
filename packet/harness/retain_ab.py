#!/usr/bin/env python3
"""Output equivalence for S3 (retained history): the same greedy requests against two engine configs.

Cases, sent one at a time in this order (prompts are seeded word streams, unique per case except C, which extends
B's prompt so it starts from B's cached prefix):
  A  ~60k-token fresh prompt   (history <= 23 pages: old and new paths both attend in one dense bucket -> identical
                                 arithmetic expected, tokens and logprobs exact)
  B  ~150k-token fresh prompt  (old path: sparse union / several buckets; new: one dense bucket -> summation order
                                 differs, small logprob differences allowed)
  C  B + ~3k appended tokens   (prefix-cache hit, 3 chunks)
  record: retain_ab.py URL MODEL OUT.json
  compare: retain_ab.py --compare BASE.json CAND.json   -> PASS / FAIL with the first divergence per case
  in-process: retain_ab.py --inproc URL MODEL SWITCHFILE OUT.json
     one engine with QSA_RETAIN_HISTORY=1; per case: retention on, then off twice (SWITCHFILE present), each with
     its own cache_salt so every run prefills from scratch. Effect = on vs off-a, noise = off-a vs off-b (the
     engine is not bit-deterministic across runs). PASS: per case, the on/off-a token prefix is as long as the
     off-a/off-b one or >= 32, and max |dlogprob| on the common prefix <= max(2 x noise, 0.05).
"""
import json, random, sys, time, urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec "
         "romeo sierra tango uniform victor whiskey xray yankee zulu north south east west river stone cedar maple "
         "copper silver amber violet").split()


def text(seed, n_words):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) + (str(r.randint(0, 999)) if r.random() < 0.3 else "") for _ in range(n_words))


def ask(url, model, prompt, salt=None):
    body = {"model": model, "messages": [{"role": "user", "content": prompt + "\n\nList the last ten words above, "
            "then continue the sequence for a while."}], "max_tokens": 64, "temperature": 0, "logprobs": True,
            "top_logprobs": 5, "chat_template_kwargs": {"enable_thinking": False}}
    if salt:
        body["cache_salt"] = salt
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    lp = d["choices"][0]["logprobs"]["content"]
    return {"tokens": [x["token"] for x in lp], "logprobs": [x["logprob"] for x in lp],
            "prompt_tokens": d["usage"]["prompt_tokens"], "seconds": round(time.time() - t, 2),
            "cached_tokens": (d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")}


def record(url, model, out):
    b = text(2002, 67000)
    cases = {"A": text(1001, 27000), "B": b, "C": b + " " + text(3003, 1400)}
    res = {}
    for k, p in cases.items():
        res[k] = ask(url, model, p)
        print(k, res[k]["prompt_tokens"], "tokens", res[k]["seconds"], "s", "".join(res[k]["tokens"][:12])[:60])
    json.dump(res, open(out, "w"))


def cases():
    b = text(2002, 67000)
    return {"A": text(1001, 27000), "B": b, "C": b + " " + text(3003, 1400)}


def diff(x, y):
    div = next((i for i, (s, t) in enumerate(zip(x["tokens"], y["tokens"])) if s != t), None)
    n = div if div is not None else min(len(x["tokens"]), len(y["tokens"]))
    return n, max((abs(p - q) for p, q in zip(x["logprobs"][:n], y["logprobs"][:n])), default=0.0)


def inproc(url, model, switch, out):
    import os
    res, ok = {}, True
    tag = str(int(time.time()))
    for k, p in cases().items():
        # C must extend B's cached prefix in the same mode, so C reuses B's salt per mode
        salt = {m: f"{m}-{'B' if k == 'C' else k}-{tag}" for m in ("on", "offa", "offb")}
        if os.path.exists(switch):
            os.remove(switch)
        on = ask(url, model, p, salt["on"])
        open(switch, "w").close()
        offa = ask(url, model, p, salt["offa"])
        offb = ask(url, model, p, salt["offb"])
        os.remove(switch)
        res[k] = {"on": on, "offa": offa, "offb": offb}
        n_eff, d_eff = diff(on, offa)
        n_noise, d_noise = diff(offa, offb)
        good = (n_eff >= min(n_noise, 32)) and d_eff <= max(2 * d_noise, 0.05)
        # every run must prefill from scratch (A, B) or from its own mode's B prefix (C): a salt that is ignored
        # would serve off-runs from the on-run's cache and compare nothing
        cached = [r.get("cached_tokens") for r in (on, offa, offb)]
        limit = 0 if k != "C" else res["B"]["on"]["prompt_tokens"]
        if any(c is None or c > limit + 4096 for c in cached):
            good = False
            print(f"{k}: cache isolation not proven, cached tokens {cached}", flush=True)
        ok &= good
        print(f"{k}: {'ok' if good else 'BAD'} prompt {on['prompt_tokens']} | on vs off: same prefix {n_eff}, max |d| "
              f"{d_eff:.4f} | off vs off (noise): prefix {n_noise}, max |d| {d_noise:.4f} | seconds on/off "
              f"{on['seconds']}/{offa['seconds']}", flush=True)
    json.dump(res, open(out, "w"))
    print("retain_ab inproc:", "PASS" if ok else "FAIL")
    return ok


def compare(base, cand):
    a, b = json.load(open(base)), json.load(open(cand))
    ok = True
    for k in a:
        x, y = a[k], b[k]
        div = next((i for i, (s, t) in enumerate(zip(x["tokens"], y["tokens"])) if s != t), None)
        n = div if div is not None else min(len(x["tokens"]), len(y["tokens"]))
        dl = max((abs(p - q) for p, q in zip(x["logprobs"][:n], y["logprobs"][:n])), default=0.0)
        strict = k == "A"
        good = (div is None and dl == 0.0) if strict else ((div is None or div >= 32) and dl < 0.05)
        ok &= good
        print(f"{k}: {'ok' if good else 'BAD'} first divergence {div}, max |dlogprob| before it {dl:.4f}"
              f" ({'exact required' if strict else 'tolerance: no divergence before token 32, |d| < 0.05'})")
    print("retain_ab:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "--inproc":
        sys.exit(0 if inproc(*sys.argv[2:6]) else 1)
    if sys.argv[1] == "--compare":
        sys.exit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
    record(*sys.argv[1:4])
