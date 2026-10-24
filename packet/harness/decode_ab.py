#!/usr/bin/env python3
"""Decode rows under a concurrent prefill vs alone (gate for the row cache in mixed steps, QSA_ROWCACHE_MIXED).

N prompts (own seeded 8k-token contexts, fresh cache_salt per request) are decoded greedily for LEN tokens, streamed
with logprobs=TOPK and a client arrival time per token: alone-a and alone-b one after another (noise control), then
all N at once; once every one of them has streamed its first token, a fresh LONG-token prefill is submitted, so
their later decode steps share steps with its chunks (client-observed: a token counts as "mixed" when it arrives
inside the prefill request's interval minus PRE_LEAD at the start and PRE_TAIL at the end).

Codex round 4 rules (a near-tie must not admit an unrelated winner, absent evidence is never a PASS):
  * every record must be complete and finite: LEN tokens, logprobs and top maps (>= 2 finite entries), the emitted
    token is in its top map with the maximal score (greedy), usage shows the whole prompt and 0 cached tokens;
    the long prefill reports LONG prompt tokens, 0 cached
  * up to the first differing token, |Δ logprob| of the emitted token <= TOL
  * at a divergence both winners are judged in both distributions: each must score within NEAR_TIE of the other's
    winner; a winner missing from a top map with a margin > NEAR_TIE is a FAIL, a winner missing otherwise is
    INCONCLUSIVE (no score); everything after an accepted divergence is unexamined
  * the candidate must have >= MIN_MIXED compared (common-prefix) positions that arrived in the mixed window, per
    prompt; fewer (queued-only run, early divergence) is INCONCLUSIVE
alone-b vs alone-a failing the same rule (noise) -> INCONCLUSIVE. Exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE.
usage: decode_ab.py URL MODEL OUT.json  (env N=6 LEN=384 LONG=200000 SEED=11 TOPK=5 MIN_MIXED=64 PRE_LEAD=1 PRE_TAIL=0.5)
"""
import json, math, os, random, sys, threading, time, urllib.request, uuid

E = lambda k, d: type(d)(os.environ.get(k, d))
N, LEN, LONG, SEED, TOPK = E("N", 6), E("LEN", 384), E("LONG", 200000), E("SEED", 11), E("TOPK", 5)
MIN_MIXED, PRE_LEAD, PRE_TAIL = E("MIN_MIXED", 64), E("PRE_LEAD", 1.0), E("PRE_TAIL", 0.5)
PLEN = E("PLEN", 8000)
TOL, NEAR_TIE = 0.375, 0.375
fin = lambda x: isinstance(x, (int, float)) and math.isfinite(x)


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


def decode(url, model, ids, first=None):
    """Streamed greedy decode; per token: id, logprob, top map, arrival time. `first` is set at the first token."""
    r = {"tokens": [], "lp": [], "top": [], "ts": [], "t0": time.time()}
    body = {"model": model, "prompt": ids, "max_tokens": LEN, "min_tokens": LEN, "temperature": 0, "logprobs": TOPK,
            "return_tokens_as_token_ids": True, "cache_salt": uuid.uuid4().hex, "stream": True,
            "stream_options": {"include_usage": True}}
    try:
        req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for line in resp:
                if not line.startswith(b"data: {"):
                    continue
                c = json.loads(line[6:])
                if c.get("usage"):
                    r["usage"] = c["usage"]
                for ch in c.get("choices") or []:
                    lp = ch.get("logprobs") or {}
                    now = time.time()
                    for t, l, tp in zip(lp.get("tokens") or [], lp.get("token_logprobs") or [],
                                        lp.get("top_logprobs") or []):
                        r["tokens"].append(t); r["lp"].append(l); r["ts"].append(now)
                        r["top"].append({k: v for k, v in (tp or {}).items()})
                    if first is not None and r["tokens"]:
                        first.set()
    except Exception as e:  # noqa: BLE001 - incomplete evidence
        r["error"] = repr(e)[:200]
    r["t1"] = time.time()
    return r


def problems(r, plen):
    """Why a record is not usable evidence ([] when it is)."""
    if "error" in r:
        return [r["error"]]
    p = [f"{k} {len(r[k])}/{LEN}" for k in ("tokens", "lp", "top", "ts") if len(r[k]) != LEN]
    if p:
        return p
    for i, (t, l, tp) in enumerate(zip(r["tokens"], r["lp"], r["top"])):
        if not fin(l) or len(tp) < 2 or not all(fin(v) for v in tp.values()):
            return [f"pos {i}: nonfinite or short top map"]
        if t not in tp or abs(tp[t] - l) > 1e-3 or tp[t] < max(tp.values()) - 1e-3:
            return [f"pos {i}: emitted token not the scored top-1"]
    u = r.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    if u.get("prompt_tokens") != plen or cached != 0:
        return [f"usage prompt {u.get('prompt_tokens')} cached {cached} (want {plen}, 0)"]
    return []


def at_divergence(a, b, k):
    """'ok' (both winners within NEAR_TIE in both distributions), 'fail' or 'inconclusive' (a score is missing)."""
    ta, tb, A, B = a["tokens"][k], b["tokens"][k], a["top"][k], b["top"][k]
    verdict = "ok"
    for W, mine, other in ((A, ta, tb), (B, tb, ta)):
        if other in W:
            if W[mine] - W[other] > NEAR_TIE:
                return "fail", W[mine] - W[other]
        else:
            second = sorted(W.values())[-2]
            if W[mine] - second > NEAR_TIE:  # the other winner scores <= second here, so it is not a near-tie
                return "fail", W[mine] - second
            verdict = "inconclusive"
    return verdict, None


def compare(a, b, window=None):
    """Common prefix, max |Δ| of the emitted token, divergence verdict, mixed positions in the prefix (b's arrivals)."""
    k, d = 0, 0.0
    while k < LEN and a["tokens"][k] == b["tokens"][k]:
        d = max(d, abs(a["lp"][k] - b["lp"][k]))
        k += 1
    div = None
    if k < LEN:
        v, gap = at_divergence(a, b, k)
        div = {"pos": k, "verdict": v, "gap": None if gap is None else round(gap, 4)}
    mixed = sum(1 for t in b["ts"][:k] if window and window[0] <= t <= window[1])
    status = ("fail" if d > TOL or (div and div["verdict"] == "fail") else
              "inconclusive" if div and div["verdict"] == "inconclusive" else "ok")
    return {"prefix": k, "max_d": round(d, 5), "divergence": div, "mixed": mixed, "status": status}


def main():
    url, model, out = sys.argv[1:4]
    rng = random.Random(SEED)
    prompts = [token_ids(url, model, rng, PLEN) for _ in range(N)]
    runs = {"alone-a": [decode(url, model, p) for p in prompts], "alone-b": [decode(url, model, p) for p in prompts]}
    long_ids = (prompts[0] * (LONG // PLEN + 1))[:LONG]  # own cache_salt: shares nothing with the decodes
    pre, res, firsts = {}, [None] * N, [threading.Event() for _ in range(N)]

    def prefill():
        pre["t0"] = time.time()
        try:
            r = post(url, "/v1/completions", {"model": model, "prompt": long_ids, "max_tokens": 1, "temperature": 0,
                                              "cache_salt": uuid.uuid4().hex})
            pre["usage"] = r.get("usage")
        except Exception as e:  # noqa: BLE001
            pre["error"] = repr(e)[:200]
        pre["t1"] = time.time()

    def one(i):
        res[i] = decode(url, model, prompts[i], firsts[i])
    ts = [threading.Thread(target=one, args=(i,)) for i in range(N)]
    for t in ts:
        t.start()
    started = all(f.wait(600) for f in firsts)  # decoding before the long prompt arrives
    th = threading.Thread(target=prefill)
    th.start()
    for t in ts:
        t.join()
    th.join()
    runs["under-prefill"] = res
    probs: list = [(c, i, p) for c, rs in runs.items() for i, r in enumerate(rs) for p in problems(r, PLEN)[:1]]
    if not started:
        probs.append(("first tokens", "not every decode streamed before the prefill"))
    pu = pre.get("usage") or {}
    pc = (pu.get("prompt_tokens_details") or {}).get("cached_tokens")
    if "error" in pre or pu.get("prompt_tokens") != LONG or pc != 0:
        probs.append(("prefill", pre.get("error") or f"usage prompt {pu.get('prompt_tokens')} cached {pc}"))
    window = (pre.get("t0", 0) + PRE_LEAD, pre.get("t1", 0) - PRE_TAIL)
    summ = {"config": {"N": N, "LEN": LEN, "LONG": LONG, "SEED": SEED, "TOPK": TOPK, "TOL": TOL, "NEAR_TIE": NEAR_TIE,
                       "MIN_MIXED": MIN_MIXED, "PRE_LEAD": PRE_LEAD, "PRE_TAIL": PRE_TAIL},
            "prefill_s": round(pre.get("t1", 0) - pre.get("t0", 0), 2), "invalid": probs[:20]}
    if probs:
        summ["final"] = "INCONCLUSIVE"
    else:
        noise = [compare(a, b) for a, b in zip(runs["alone-a"], runs["alone-b"])]
        cand = [compare(a, c, window) for a, c in zip(runs["alone-a"], res)]
        summ.update(noise=noise, under_prefill=cand)
        summ["final"] = ("FAIL" if any(x["status"] == "fail" for x in cand) else
                         "INCONCLUSIVE" if any(x["status"] != "ok" for x in noise + cand)
                         or any(x["mixed"] < MIN_MIXED for x in cand) else "PASS")
    json.dump({"summary": summ, "runs": runs, "prefill": pre}, open(out, "w"), indent=1)
    up = summ.get("under_prefill", [])
    print("decode_ab", json.dumps({k: summ[k] for k in ("final", "prefill_s", "invalid")}),
          "| prefixes", [x["prefix"] for x in up], "mixed", [x["mixed"] for x in up],
          "max_d", max((x["max_d"] for x in up), default=None),
          "| noise", [x["status"] for x in summ.get("noise", [])])
    sys.exit({"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[summ["final"]])


if __name__ == "__main__":
    main()
