#!/usr/bin/env python3
"""Teacher-forced batch-invariance / equivalence test (STALL-PLAN goal item 6; replaces free-running comparisons).

Every probe is one fixed token sequence: context (CTX tokens, seeded) + its own tail (TAIL tokens, seeded per probe),
sent as token ids to /v1/completions with max_tokens=1, logprobs=20 and a fresh cache_salt, so every run of a probe
is a cold prefill of the same tokens and the compared distribution sits at the same position with the same history.
(Prompt logprobs would give more positions per request, but this vLLM turns prefix-cache reads off for them and
computes full-vocabulary logits per chunk -- up to ~2 GB transient on the last rank, too close to the VRAM limit.)

Conditions (each runs every probe once):
  alone-a, alone-b   sequential, no other load: alone-b vs alone-a is the noise floor at identical positions
  reordered          probes in reverse order
  concurrent         probes while LOAD streams decode on their own (warm) contexts
  switch-on/off      only with SWITCH=<file inside the container's /work>: the S3 switch file absent (on) / present
                     (off), touched through docker exec by the caller-supplied SWITCH_CMD_ON / SWITCH_CMD_OFF
Per pair of conditions: top-1 agreement, max |logprob(top-1 of A) in A - same token in B|, and max over the shared
top-20 of |Δ logprob|. Verdict for each candidate pair (reordered, concurrent, switch-on vs switch-off) against the
noise pair: PASS if top-1 agreement >= noise agreement and max |Δ| <= max(TOL_X * noise max, TOL_ABS); FAIL if
exceeded; INCONCLUSIVE when fewer than MIN_PROBES probes returned complete top-20 evidence or a probe was served
from cache (cached_tokens > 0). Exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE.
usage: teacher_forced.py URL MODEL OUT.json  (env: PROBES=12 CTX=60000 TAIL=2500 LOAD=6 SEED=7 TOL_X=1.5
       TOL_ABS=0.02 MIN_PROBES=10 [SWITCH_CMD_ON=... SWITCH_CMD_OFF=...])
"""
import json, os, random, subprocess, sys, threading, time, urllib.request

E = lambda k, d: type(d)(os.environ.get(k, d))
PROBES, CTX, TAIL, LOAD, SEED = E("PROBES", 12), E("CTX", 60000), E("TAIL", 2500), E("LOAD", 6), E("SEED", 7)
TOL_X, TOL_ABS, MIN_PROBES = E("TOL_X", 1.5), E("TOL_ABS", 0.02), E("MIN_PROBES", 10)
SW_ON, SW_OFF = os.environ.get("SWITCH_CMD_ON"), os.environ.get("SWITCH_CMD_OFF")


def post(url, path, body, timeout=3600):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def token_ids(url, model, rng, n):
    """n valid token ids: tokenize seeded words, then cut to n (ids, not text, so positions are exact)."""
    words = [rng.choice("alpha beta gamma delta epsilon kernel page block cache stream layer token "
                        "commit merge diff patch queue batch thread".split()) + str(rng.randint(0, 999))
             for _ in range(int(n / 2.0) + 64)]
    ids = post(url, "/tokenize", {"model": model, "prompt": " ".join(words), "add_special_tokens": False})["tokens"]
    if len(ids) < n:
        raise SystemExit(f"tokenizer gave {len(ids)} < {n} ids")
    return ids[:n]


def probe(url, model, ids, salt):
    t0 = time.time()
    r = post(url, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                                      "logprobs": 20, "return_tokens_as_token_ids": True, "cache_salt": salt})
    lp = r["choices"][0]["logprobs"]
    top = (lp.get("top_logprobs") or [None])[0] or {}
    return {"top": {k: float(v) for k, v in top.items()}, "first": (lp.get("tokens") or [None])[0],
            "cached": (r.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
            "prompt_tokens": r.get("usage", {}).get("prompt_tokens"), "seconds": round(time.time() - t0, 2)}


def run_condition(url, model, probes, name, order):
    out = {}
    for i in order:
        out[i] = probe(url, model, probes[i], f"tf-{name}-{i}-{time.time_ns()}")
        print(f"  {name} probe {i}: first {out[i]['first']} cached {out[i]['cached']} {out[i]['seconds']} s", flush=True)
    return out


def load_streams(url, model, rng, stop):
    """LOAD decoding streams on their own warm contexts (short, so they are cached after the first round)."""
    ctxs = [token_ids(url, model, rng, 8000) for _ in range(LOAD)]

    def one(c):
        while not stop.is_set():
            try:
                post(url, "/v1/completions", {"model": model, "prompt": c, "max_tokens": 400, "min_tokens": 400,
                                              "temperature": 0.7})
            except Exception as e:
                print("load stream error", repr(e)[:200], flush=True)
                time.sleep(1)
    th = [threading.Thread(target=one, args=(c,), daemon=True) for c in ctxs]
    for t in th:
        t.start()
    return th


def compare(a, b):
    """Over probes present in both with full top-20: (n, top-1 agreement, max |Δ| on A's top-1, max |Δ| shared)."""
    n, agree, d1, dall = 0, 0, 0.0, 0.0
    for i in a:
        if i not in b or len(a[i]["top"]) < 20 or len(b[i]["top"]) < 20:
            continue
        n += 1
        ta, tb = a[i]["top"], b[i]["top"]
        t1 = max(ta, key=ta.get)
        agree += t1 == max(tb, key=tb.get)
        d1 = max(d1, abs(ta[t1] - tb[t1]) if t1 in tb else float("inf"))
        shared = set(ta) & set(tb)
        dall = max(dall, max(abs(ta[k] - tb[k]) for k in shared) if shared else float("inf"))
    return {"n": n, "top1_agree": agree / n if n else None, "max_d_top1": round(d1, 5), "max_d_shared": round(dall, 5)}


def verdict(cand, noise):
    if cand["n"] < MIN_PROBES or noise["n"] < MIN_PROBES:
        return "INCONCLUSIVE"
    lim = max(TOL_X * noise["max_d_top1"], TOL_ABS)
    if cand["top1_agree"] < noise["top1_agree"] or cand["max_d_top1"] > lim:
        return "FAIL"
    return "PASS"


def main():
    url, model, out_path = sys.argv[1:4]
    if bool(SW_ON) != bool(SW_OFF):
        raise SystemExit("set both SWITCH_CMD_ON and SWITCH_CMD_OFF, or neither")
    rng = random.Random(SEED)
    ctx = token_ids(url, model, rng, CTX)
    probes = {i: ctx + token_ids(url, model, rng, TAIL) for i in range(PROBES)}
    fwd, rev = list(range(PROBES)), list(range(PROBES))[::-1]
    conds = {}
    if SW_ON:
        subprocess.run(str(SW_ON), shell=True, check=True)
    conds["alone-a"] = run_condition(url, model, probes, "alone-a", fwd)
    conds["alone-b"] = run_condition(url, model, probes, "alone-b", fwd)
    conds["reordered"] = run_condition(url, model, probes, "reordered", rev)
    stop = threading.Event()
    th = load_streams(url, model, random.Random(SEED + 1), stop)
    time.sleep(E("TF_SETTLE", 20.0))  # load contexts prefilled and decoding
    conds["concurrent"] = run_condition(url, model, probes, "concurrent", fwd)
    stop.set()
    for t in th:
        t.join(timeout=600)
    if SW_OFF:
        subprocess.run(SW_OFF, shell=True, check=True)
        conds["switch-off"] = run_condition(url, model, probes, "switch-off", fwd)
        subprocess.run(str(SW_ON), shell=True, check=True)
    bad_cache = [(c, i) for c, d in conds.items() for i, r in d.items() if r["cached"] not in (0, None)
                 or r["prompt_tokens"] != CTX + TAIL]
    noise = compare(conds["alone-a"], conds["alone-b"])
    pairs = {"reordered": compare(conds["alone-a"], conds["reordered"]),
             "concurrent": compare(conds["alone-a"], conds["concurrent"])}
    if "switch-off" in conds:
        pairs["switch"] = compare(conds["alone-a"], conds["switch-off"])
    vs = {k: verdict(v, noise) for k, v in pairs.items()}
    if bad_cache:
        vs = {k: "INCONCLUSIVE" if v == "PASS" else v for k, v in vs.items()}
    final = "FAIL" if "FAIL" in vs.values() else "PASS" if all(v == "PASS" for v in vs.values()) else "INCONCLUSIVE"
    summ = {"noise": noise, "pairs": pairs, "verdicts": vs, "bad_cache_or_length": bad_cache[:10], "final": final,
            "config": {"PROBES": PROBES, "CTX": CTX, "TAIL": TAIL, "LOAD": LOAD, "SEED": SEED, "TOL_X": TOL_X,
                       "TOL_ABS": TOL_ABS, "MIN_PROBES": MIN_PROBES, "switch": bool(SW_OFF)}}
    json.dump({"summary": summ, "conditions": conds}, open(out_path, "w"), indent=1)
    print("teacher_forced", json.dumps(summ))
    sys.exit({"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[final])


if __name__ == "__main__":
    main()
