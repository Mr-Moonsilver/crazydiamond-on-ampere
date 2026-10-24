#!/usr/bin/env python3
"""Decode rows under a concurrent prefill vs alone (gate for the row cache in mixed steps, QSA_ROWCACHE_MIXED).

N prompts (own seeded 8k-token contexts, fresh cache_salt per run) are decoded greedily for LEN tokens with
logprobs=2: alone-a and alone-b one after another (noise control), then all N at once while a fresh LONG-token
prefill runs, so their decode steps are mixed steps. Per prompt, up to the first divergence from alone-a, |Δ
logprob| of the emitted token must stay <= TOL (0.375, teacher_forced rule v3); a divergence must happen at a
near-tie (alone-a's top-1 margin <= NEAR_TIE 0.375); any other divergence or a larger Δ fails. A leak (another
session's rows) shows as a large Δ or an early non-near-tie divergence. INCONCLUSIVE when the evidence is incomplete
(length, logprobs, errors), when alone-b vs alone-a already fails the rule, or when a concurrent decode overlapped the
prefill for less than half of its duration. Exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE.
usage: decode_ab.py URL MODEL OUT.json   (env N=6 LEN=128 LONG=200000 SEED=11)
"""
import json, os, random, sys, threading, time, urllib.request, uuid

E = lambda k, d: type(d)(os.environ.get(k, d))
N, LEN, LONG, SEED = E("N", 6), E("LEN", 128), E("LONG", 200000), E("SEED", 11)
TOL, NEAR_TIE = 0.375, 0.375


def post(url, path, body, timeout=3600):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def token_ids(url, model, rng, n):
    words = [rng.choice("alpha beta gamma delta kernel page block cache stream layer token commit merge diff patch "
                        "queue batch thread".split()) + str(rng.randint(0, 999)) for _ in range(int(n / 2.0) + 64)]
    ids = post(url, "/tokenize", {"model": model, "prompt": " ".join(words), "add_special_tokens": False})["tokens"]
    if len(ids) < n:
        raise SystemExit(f"tokenizer gave {len(ids)} < {n} ids")
    return ids[:n]


def decode(url, model, ids):
    t0 = time.time()
    try:
        r = post(url, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": LEN, "min_tokens": LEN,
                                          "temperature": 0, "logprobs": 2, "return_tokens_as_token_ids": True,
                                          "cache_salt": uuid.uuid4().hex})
        lp = r["choices"][0]["logprobs"]
        return {"tokens": lp.get("tokens") or [], "lp": lp.get("token_logprobs") or [],
                "top": [{k: float(v) for k, v in (t or {}).items()} for t in (lp.get("top_logprobs") or [])],
                "t0": t0, "t1": time.time()}
    except Exception as e:  # noqa: BLE001 - incomplete evidence
        return {"error": repr(e)[:200], "t0": t0, "t1": time.time()}


def compare(a, b):
    """Common prefix, max |Δ| of the emitted token up to it, and whether the divergence (if any) is a near-tie."""
    k, d = 0, 0.0
    while k < len(a["tokens"]) and k < len(b["tokens"]) and a["tokens"][k] == b["tokens"][k]:
        d = max(d, abs(a["lp"][k] - b["lp"][k]))
        k += 1
    div = None
    if k < LEN:
        vals = sorted(a["top"][k].values()) if k < len(a["top"]) else []
        margin = vals[-1] - vals[-2] if len(vals) >= 2 else float("inf")
        div = {"pos": k, "margin": round(margin, 4), "near_tie": margin <= NEAR_TIE}
    ok = d <= TOL and (div is None or div["near_tie"])
    return {"prefix": k, "max_d": round(d, 5), "divergence": div, "ok": ok}


def valid(r):
    return ("error" not in r and len(r["tokens"]) == LEN and len(r["lp"]) == LEN and len(r["top"]) == LEN
            and all(isinstance(x, (int, float)) for x in r["lp"]))


def main():
    url, model, out = sys.argv[1:4]
    rng = random.Random(SEED)
    prompts = [token_ids(url, model, rng, 8000) for _ in range(N)]
    runs = {"alone-a": [decode(url, model, p) for p in prompts], "alone-b": [decode(url, model, p) for p in prompts]}
    long_ids = (prompts[0] * (LONG // 8000 + 1))[:LONG]  # own cache_salt: shares nothing with the decodes
    pre = {}

    def prefill():
        pre["t0"] = time.time()
        try:
            post(url, "/v1/completions", {"model": model, "prompt": long_ids, "max_tokens": 1, "temperature": 0,
                                          "cache_salt": uuid.uuid4().hex})
        except Exception as e:  # noqa: BLE001
            pre["error"] = repr(e)[:200]
        pre["t1"] = time.time()
    th = threading.Thread(target=prefill)
    th.start()
    time.sleep(3)
    res = [None] * N

    def one(i):
        res[i] = decode(url, model, prompts[i])
    ts = [threading.Thread(target=one, args=(i,)) for i in range(N)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    th.join()
    runs["under-prefill"] = res
    probs: list = [(c, i, r.get("error", "incomplete")) for c, rs in runs.items() for i, r in enumerate(rs) if not valid(r)]
    if "error" in pre:
        probs.append(("prefill", pre["error"]))
    overlap = []
    for r in res:
        o = max(0.0, min(r["t1"], pre.get("t1", 0)) - max(r["t0"], pre.get("t0", 0))) / max(r["t1"] - r["t0"], 1e-9)
        overlap.append(round(o, 2))
    if any(o < 0.5 for o in overlap):
        probs.append(("overlap", overlap))
    summ = {"config": {"N": N, "LEN": LEN, "LONG": LONG, "SEED": SEED, "TOL": TOL, "NEAR_TIE": NEAR_TIE},
            "overlap": overlap, "prefill_s": round(pre.get("t1", 0) - pre.get("t0", 0), 2), "invalid": probs[:20]}
    if probs:
        summ["final"] = "INCONCLUSIVE"
    else:
        noise = [compare(a, b) for a, b in zip(runs["alone-a"], runs["alone-b"])]
        cand = [compare(a, c) for a, c in zip(runs["alone-a"], res)]
        summ.update(noise=noise, under_prefill=cand)
        summ["final"] = ("INCONCLUSIVE" if not all(x["ok"] for x in noise) else
                         "PASS" if all(x["ok"] for x in cand) else "FAIL")
    json.dump({"summary": summ, "runs": runs, "prefill": pre}, open(out, "w"), indent=1)
    print("decode_ab", json.dumps({k: summ[k] for k in ("final", "overlap", "prefill_s", "invalid")}),
          "| prefixes", [x["prefix"] for x in summ.get("under_prefill", [])],
          "max_d", max((x["max_d"] for x in summ.get("under_prefill", [])), default=None))
    sys.exit({"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[summ["final"]])


if __name__ == "__main__":
    main()
