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
Grading rule v3 (Codex round 2; frozen 24 Oct from the q2 pilot -- noise pair: winners within 0.125, A's top 5
within 0.134, one near-tie flip at margin 0.125; bf16 logprob steps are 0.125 here -- before any new data). Fixed
absolute tolerances, never scaled by the run's own noise: for every probe, |Δ logprob| of BOTH conditions' top-1
tokens <= TOL_WIN (0.375) and of A's top 5 <= TOL_TOP5 (0.5); a token missing from the other condition's top 20
counts as unbounded (evidence incomplete -> fail of that probe). Every top-1 flip is reported; a flip whose margin
in A is <= NEAR_TIE (0.375) is labelled near-tie, any other flip fails. The noise pair (alone-b vs alone-a) is held
to the same rule: if it fails, the baseline is noisier than the rule and the run is INCONCLUSIVE (characterise the
baseline; the gate is not widened). Evidence must be complete: every probe in every condition with 20 finite
logprobs, prompt length CTX+TAIL and cached_tokens reported and 0 -- else INCONCLUSIVE (also on --regrade, which
re-validates against the artifact's recorded configuration). PROBES=12 on one shared context is a smoke test: PASS
means "no regression seen in 12 probes", not population equivalence (0/12 gives a 95 % upper bound of 22 % on the
event rate). Exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE.
usage: teacher_forced.py URL MODEL OUT.json  (env: PROBES=12 CTX=60000 TAIL=2500 LOAD=6 SEED=7
       [SWITCH_CMD_ON=... SWITCH_CMD_OFF=...])      teacher_forced.py --regrade OUT.json
"""
import json, math, os, random, subprocess, sys, threading, time, urllib.request

E = lambda k, d: type(d)(os.environ.get(k, d))
PROBES, CTX, TAIL, LOAD, SEED = E("PROBES", 12), E("CTX", 60000), E("TAIL", 2500), E("LOAD", 6), E("SEED", 7)
RULE = {"version": 3, "TOL_WIN": 0.375, "TOL_TOP5": 0.5, "NEAR_TIE": 0.375}  # frozen; see docstring
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


def compare(a, b, rule=RULE):
    """Per-probe rule v3 over probes present in a: symmetric winners, A's top 5, every flip (near-tie or not)."""
    n, bad, flips, dwin, d5, dshared = 0, [], [], 0.0, 0.0, 0.0
    for i in sorted(a):
        n += 1
        ta, tb = a[i]["top"], b.get(i, {}).get("top", {})
        wa, wb = max(ta, key=ta.get), max(tb, key=tb.get) if tb else None
        dw = max(abs(ta[t] - tb[t]) if t in ta and t in tb else math.inf for t in {wa, wb})
        top5 = sorted(ta, key=ta.get)[-5:]
        d5i = max(abs(ta[k] - tb[k]) if k in tb else math.inf for k in top5)
        dwin, d5 = max(dwin, dw), max(d5, d5i)
        shared = set(ta) & set(tb)
        dshared = max(dshared, max(abs(ta[k] - tb[k]) for k in shared) if shared else math.inf)
        if wa != wb:
            margin = ta[wa] - ta[wb] if wb in ta else math.inf
            flips.append({"probe": i, "margin": round(margin, 4), "near_tie": margin <= rule["NEAR_TIE"]})
        if dw > rule["TOL_WIN"] or d5i > rule["TOL_TOP5"] or (wa != wb and not flips[-1]["near_tie"]):
            bad.append(i)
    r = lambda x: round(x, 5) if math.isfinite(x) else "inf"
    return {"n": n, "fail_probes": bad, "flips": flips, "max_d_winners": r(dwin), "max_d_top5": r(d5),
            "max_d_shared_info": r(dshared)}


def validate(conds, ctx, tail):
    """Complete, finite, cold evidence at the right length; returns the list of problems (empty = valid)."""
    probs = []
    ids = set(conds.get("alone-a", {}))
    for c, d in conds.items():
        if set(d) != ids or not ids:
            probs.append((c, "probe set"))
        for i, r in d.items():
            top = r.get("top") or {}
            if len(top) < 20 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in top.values()):
                probs.append((c, i, "top20"))
            if r.get("cached") != 0:
                probs.append((c, i, f"cached={r.get('cached')}"))
            if r.get("prompt_tokens") != ctx + tail:
                probs.append((c, i, f"len={r.get('prompt_tokens')}"))
    return probs


def grade(conds, ctx, tail):
    probs = validate(conds, ctx, tail)
    noise = compare(conds["alone-a"], conds["alone-b"])
    pairs = {"reordered": compare(conds["alone-a"], conds["reordered"]),
             "concurrent": compare(conds["alone-a"], conds["concurrent"])}
    if "switch-off" in conds:
        pairs["switch"] = compare(conds["alone-a"], conds["switch-off"])
    if probs or noise["fail_probes"]:
        vs = {k: "INCONCLUSIVE" for k in pairs}
    else:
        vs = {k: "FAIL" if v["fail_probes"] else "PASS" for k, v in pairs.items()}
    final = "FAIL" if "FAIL" in vs.values() else "PASS" if all(v == "PASS" for v in vs.values()) else "INCONCLUSIVE"
    return {"noise": noise, "pairs": pairs, "verdicts": vs, "invalid_evidence": probs[:20], "final": final,
            "rule": RULE, "claim": f"smoke test, {noise['n']} probes on one shared context"}


def main():
    if sys.argv[1] == "--regrade":  # re-grade a saved run with the current rule, validated against its own config
        d = json.load(open(sys.argv[2]))
        cfg = d["summary"]["config"]
        conds = {c: {int(k): v for k, v in r.items()} for c, r in d["conditions"].items()}
        summ = grade(conds, cfg["CTX"], cfg["TAIL"])
        print("teacher_forced regrade", json.dumps(summ))
        sys.exit({"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[summ["final"]])
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
    summ = grade(conds, CTX, TAIL)
    summ["config"] = {"PROBES": PROBES, "CTX": CTX, "TAIL": TAIL, "LOAD": LOAD, "SEED": SEED, "switch": bool(SW_OFF)}
    final = summ["final"]
    json.dump({"summary": summ, "conditions": conds}, open(out_path, "w"), indent=1)
    print("teacher_forced", json.dumps(summ))
    sys.exit({"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[final])


if __name__ == "__main__":
    main()
