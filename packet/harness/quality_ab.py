#!/usr/bin/env python3
"""Model-quality A/B across engine configurations (FP8 PLE phase: BF16 vs FP8 n-gram table) on real text.

run:     quality_ab.py run URL MODEL CORPUS.txt OUT.json
  CORPUS is real text (q7 builds it from the image's own Python sources and docs, sorted, so every run sees the same
  bytes; the corpus SHA256 is recorded). It is tokenized once through /tokenize; PROBES seeded positions p are drawn
  with context lengths from CTXS; each probe sends ids[p-L:p] (max_tokens=1, temperature 0, logprobs=20, fresh
  cache_salt: a cold prefill on every engine) and records the top-20 map, the emitted token, the true next token
  ids[p] and its rank (1..20, or 0 = not in the top 20). Every probe must come back complete (20 finite logprobs,
  prompt_tokens = L, cached_tokens = 0), else the run is marked incomplete.
compare: quality_ab.py compare BASE_A.json BASE_B.json CAND.json OUT.json
  BASE_A / BASE_B: two runs of the reference engine (noise floor, same probes); CAND: the candidate engine.
  Per pair (A vs B = noise, A vs CAND = candidate): top-1 agreement; KL lower bound over A's top 20 (a token missing
  from the other map is charged at that map's 20th logprob, an upper bound on its logprob, so the KL is a lower bound);
  |Δ logprob| of A's winner; next-token accuracy (rank 1) of both sides, paired: n10 (A right, other wrong), n01.
Verdict rule v1 (frozen 24 Oct before any FP8 data; the yardstick is the measured noise, the slack is fixed):
  INCONCLUSIVE  any run incomplete, probe sets or corpus differ, or the noise pair agrees on top-1 for < 0.97 of the
                probes (the reference is too noisy to judge against)
  FAIL          candidate top-1 agreement < noise agreement - 0.03, or
                mean KL_lb(candidate) > 3 x mean KL_lb(noise) + 0.005 nats, or
                paired accuracy loss n10 - n01 > 0.02 x PROBES and > 2 x sqrt(n10 + n01) (more than noise)
  PASS          otherwise: "no quality loss beyond the noise floor seen on PROBES real-text positions"
Exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE (compare); run exits 0 when complete, 2 when incomplete.
env (run): PROBES=600 CTXS=1024,4096,16384 SEED=5
"""
import hashlib, json, math, os, random, sys, urllib.request, uuid

E = lambda k, d: type(d)(os.environ.get(k, d))
PROBES, SEED = E("PROBES", 600), E("SEED", 5)
CTXS = [int(x) for x in os.environ.get("CTXS", "1024,4096,16384").split(",")]
RULE = {"version": 1, "MIN_NOISE_AGREE": 0.97, "AGREE_SLACK": 0.03, "KL_FACTOR": 3.0, "KL_SLACK": 0.005,
        "ACC_LOSS_SHARE": 0.02, "ACC_SIGMAS": 2.0}
fin = lambda x: isinstance(x, (int, float)) and math.isfinite(x)


def post(url, path, body, timeout=1800):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tid(t):
    return t if isinstance(t, int) else int(str(t).split(":")[-1])


def run(url, model, corpus, out):
    text = open(corpus, encoding="utf-8", errors="replace").read()
    ids, piece = [], []  # tokenized in ~100 KB pieces cut at line ends (an endpoint may cap one request's length)
    for line in text.splitlines(True):
        piece.append(line)
        if sum(map(len, piece)) > 100_000:
            ids += post(url, "/tokenize", {"model": model, "prompt": "".join(piece), "add_special_tokens": False})["tokens"]
            piece = []
    if piece:
        ids += post(url, "/tokenize", {"model": model, "prompt": "".join(piece), "add_special_tokens": False})["tokens"]
    if len(ids) < max(CTXS) + PROBES + 1:
        raise SystemExit(f"corpus too short: {len(ids)} tokens")
    rng = random.Random(SEED)
    probes = []
    for _ in range(PROBES):
        L = rng.choice(CTXS)
        probes.append((rng.randrange(L, len(ids) - 1), L))
    recs, complete = [], True
    for i, (p, L) in enumerate(probes):
        r = {"p": p, "L": L, "true": ids[p]}
        try:
            c = post(url, "/v1/completions", {"model": model, "prompt": ids[p - L:p], "max_tokens": 1, "temperature": 0,
                                              "logprobs": 20, "return_tokens_as_token_ids": True,
                                              "cache_salt": uuid.uuid4().hex})
            lp = c["choices"][0]["logprobs"]
            top = {tid(k): v for k, v in (lp["top_logprobs"][0] or {}).items()}
            u = c.get("usage") or {}
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
            r.update(top=top, emitted=tid(lp["tokens"][0]))
            ranked = sorted(top, key=top.get, reverse=True)
            r["rank"] = ranked.index(ids[p]) + 1 if ids[p] in top else 0
            ok = len(top) == 20 and all(fin(v) for v in top.values()) and u.get("prompt_tokens") == L and cached == 0
        except Exception as e:  # noqa: BLE001 - incomplete evidence
            r["error"], ok = repr(e)[:200], False
        r["ok"] = ok
        complete &= ok
        recs.append(r)
        if i % 100 == 99:
            print(f"{i + 1}/{PROBES} probes", flush=True)
    meta = {"corpus_sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest(), "corpus_tokens": len(ids),
            "PROBES": PROBES, "CTXS": CTXS, "SEED": SEED, "complete": complete}
    json.dump({"meta": meta, "probes": recs}, open(out, "w"))
    acc = sum(r.get("rank") == 1 for r in recs) / len(recs)
    print("quality_ab run", json.dumps({"complete": complete, "probes": len(recs), "next_token_acc": round(acc, 4)}))
    sys.exit(0 if complete else 2)


def pair(a, b):
    agree = kl = dwin = 0.0
    n10 = n01 = 0
    for x, y in zip(a, b):
        A, B = {int(k): v for k, v in x["top"].items()}, {int(k): v for k, v in y["top"].items()}
        wa, wb = max(A, key=A.get), max(B, key=B.get)
        agree += wa == wb
        floor = min(B.values())
        kl += sum(math.exp(lp) * (lp - B.get(t, floor)) for t, lp in A.items())
        dwin += abs(A[wa] - B.get(wa, floor))
        ra, rb = x["rank"] == 1, y["rank"] == 1
        n10 += ra and not rb
        n01 += rb and not ra
    n = len(a)
    return {"top1_agree": round(agree / n, 4), "kl_lb_mean": round(kl / n, 6), "dwin_mean": round(dwin / n, 5),
            "n10": n10, "n01": n01, "acc_a": round(sum(x["rank"] == 1 for x in a) / n, 4),
            "acc_b": round(sum(y["rank"] == 1 for y in b) / n, 4)}


def compare(pa, pb, pc, out):
    runs = [json.load(open(p)) for p in (pa, pb, pc)]
    why = []
    for name, r in zip(("base_a", "base_b", "cand"), runs):
        if not r["meta"]["complete"]:
            why.append(f"{name} incomplete")
    keys = [[(x["p"], x["L"], x["true"]) for x in r["probes"]] for r in runs]
    if keys[0] != keys[1] or keys[0] != keys[2] or len({r["meta"]["corpus_sha256"] for r in runs}) != 1:
        why.append("probe sets or corpus differ")
    res = {"rule": RULE, "inconclusive": why}
    if not why:
        a, b, c = (r["probes"] for r in runs)
        noise, cand = pair(a, b), pair(a, c)
        res.update(noise=noise, candidate=cand)
        n = len(a)
        fails = []
        if noise["top1_agree"] < RULE["MIN_NOISE_AGREE"]:
            why.append(f"noise top-1 agreement {noise['top1_agree']} < {RULE['MIN_NOISE_AGREE']}")
        if cand["top1_agree"] < noise["top1_agree"] - RULE["AGREE_SLACK"]:
            fails.append(f"top-1 agreement {cand['top1_agree']} < noise {noise['top1_agree']} - {RULE['AGREE_SLACK']}")
        if cand["kl_lb_mean"] > RULE["KL_FACTOR"] * noise["kl_lb_mean"] + RULE["KL_SLACK"]:
            fails.append(f"KL_lb {cand['kl_lb_mean']} > {RULE['KL_FACTOR']} x noise {noise['kl_lb_mean']} + {RULE['KL_SLACK']}")
        loss = cand["n10"] - cand["n01"]
        if loss > RULE["ACC_LOSS_SHARE"] * n and loss > RULE["ACC_SIGMAS"] * math.sqrt(cand["n10"] + cand["n01"]):
            fails.append(f"accuracy loss {loss} of {n} (n10 {cand['n10']}, n01 {cand['n01']})")
        res["fails"] = fails
    res["final"] = "INCONCLUSIVE" if why else "FAIL" if res["fails"] else "PASS"
    json.dump(res, open(out, "w"), indent=1)
    print("quality_ab", json.dumps({k: res.get(k) for k in ("final", "inconclusive", "fails", "noise", "candidate")}))
    sys.exit({"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[res["final"]])


if __name__ == "__main__":
    {"run": lambda: run(*sys.argv[2:6]), "compare": lambda: compare(*sys.argv[2:6])}[sys.argv[1]]()
