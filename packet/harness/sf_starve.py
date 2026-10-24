#!/usr/bin/env python3
"""Short-first starvation test (Codex 24 Oct: aging relaxes the filter every FLASH_SHORT_FIRST_MAX_RUN deferring steps
but reserves no service for the waiting long prompt -- a sustained stream of short arrivals must not starve it).

K workers submit fresh short prompts (SHORT token ids, own cache_salt each, MAXTOK output tokens) back to back; after
WARM seconds one fresh LONG-token prompt (max_tokens 1) is submitted; the shorts keep arriving until it returns.
Reported: the long prompt's TTFT, the shorts started / completed while it waited and their TTFT p50/p95, and the
short arrival rate (the load actually sustained). Run it on a short-first engine and on the same engine without
short-first (control): the ratio of the two long-prompt TTFTs is the cost of prioritisation under this load.
Verdict: FAIL when the long prompt has not returned after MAX_S seconds (starved) or errors; INCONCLUSIVE when the
shorts did not keep arriving during the whole wait (fewer than MIN_SHORTS completed, or a gap between short
completions longer than 20 s) or a short request failed; else PASS (completed under sustained short load).
usage: sf_starve.py URL MODEL OUT.json   (env K=6 SHORT=2048 MAXTOK=32 LONG=200000 WARM=10 MAX_S=900 MIN_SHORTS=20)
"""
import json, os, random, sys, threading, time, urllib.request, uuid

E = lambda k, d: type(d)(os.environ.get(k, d))
K, SHORT, MAXTOK, LONG = E("K", 6), E("SHORT", 2048), E("MAXTOK", 32), E("LONG", 200000)
WARM, MAX_S, MIN_SHORTS = E("WARM", 10.0), E("MAX_S", 900.0), E("MIN_SHORTS", 20)


def post(url, body, timeout):
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    url, model, out = sys.argv[1:4]
    rng = random.Random(7)
    stop, shorts, errors, lock = threading.Event(), [], [], threading.Lock()

    def worker(w):
        r = random.Random(100 + w)
        while not stop.is_set():
            ids = [r.randint(1000, 150000) for _ in range(SHORT)]
            t0 = time.time()
            try:
                post(url, {"model": model, "prompt": ids, "max_tokens": MAXTOK, "min_tokens": MAXTOK, "temperature": 0,
                           "cache_salt": uuid.uuid4().hex}, 600)
                with lock:
                    shorts.append((t0, time.time()))
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(repr(e)[:160])
                time.sleep(1)
    ws = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(K)]
    for t in ws:
        t.start()
    time.sleep(WARM)
    lng: dict = {"t0": time.time()}
    long_ids = [rng.randint(1000, 150000) for _ in range(LONG)]

    def long_req():
        try:
            r = post(url, {"model": model, "prompt": long_ids, "max_tokens": 1, "temperature": 0,
                           "cache_salt": uuid.uuid4().hex}, MAX_S + 600)
            lng["usage"] = r.get("usage")
        except Exception as e:  # noqa: BLE001
            lng["error"] = repr(e)[:160]
        lng["t1"] = time.time()
    lt = threading.Thread(target=long_req, daemon=True)
    lt.start()
    lt.join(MAX_S)
    starved = lt.is_alive()
    stop.set()
    t_end = lng.get("t1", time.time())
    with lock:
        during = [(a, b) for a, b in shorts if a >= lng["t0"] and b <= t_end]
        done_t = sorted(b for a, b in shorts if lng["t0"] <= b <= t_end)
    ttfts = sorted(b - a for a, b in during)
    pct = lambda q: round(ttfts[min(len(ttfts) - 1, int(len(ttfts) * q))], 2) if ttfts else None
    gaps = [y - x for x, y in zip([lng["t0"]] + done_t, done_t + [t_end])]
    summ = {"config": {"K": K, "SHORT": SHORT, "MAXTOK": MAXTOK, "LONG": LONG, "WARM": WARM, "MAX_S": MAX_S,
                       "MIN_SHORTS": MIN_SHORTS},
            "long_ttft_s": None if starved or "t1" not in lng else round(lng["t1"] - lng["t0"], 2),
            "long_error": lng.get("error"), "long_usage": lng.get("usage"),
            "shorts_completed_during": len(during), "short_ttft_p50_s": pct(0.5), "short_ttft_p95_s": pct(0.95),
            "short_rate_per_s": round(len(done_t) / max(t_end - lng["t0"], 1e-9), 2),
            "max_gap_between_short_completions_s": round(max(gaps), 2), "short_errors": errors[:5]}
    if starved or "error" in lng:
        summ["final"] = "FAIL"
    elif errors or len(during) < MIN_SHORTS or max(gaps) > 20:
        summ["final"] = "INCONCLUSIVE"
    else:
        summ["final"] = "PASS"
    json.dump({"summary": summ, "shorts": shorts, "long": lng}, open(out, "w"), indent=1)
    print("sf_starve", json.dumps({k: summ[k] for k in ("final", "long_ttft_s", "shorts_completed_during",
                                                         "short_ttft_p50_s", "short_ttft_p95_s", "short_rate_per_s",
                                                         "max_gap_between_short_completions_s", "long_error")}))
    sys.exit({"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[summ["final"]])


if __name__ == "__main__":
    main()
