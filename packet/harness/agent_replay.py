#!/usr/bin/env python3
"""Agent replay benchmark: the ship gate for agentic work (STALL-PLAN 3b.6).

N agents run concurrently against the engine, each a coding-agent-like loop on its own growing conversation:
  turn = (history so far) + an appended tool result of APPEND_MIN..APPEND_MAX tokens -> streamed reply of
         REPLY_MIN..REPLY_MAX tokens -> tool pause TOOL_MIN..TOOL_MAX s -> next turn.
Each agent starts from a seeded history of START_MIN..START_MAX tokens (its first turn is the cold prefill and is
reported separately). When a history passes ROLL_AT tokens the agent starts a new session (a fresh long prompt, as a
compacted or new task would), so fresh prefills keep arriving at a realistic rate. Every number is seeded: two runs
send the same sequence of requests (timing aside).

Metrics (warm turns only, after WARMUP s): completed turns per hour, turn time p50/p95 (send -> last token), TTFT
p50/p95, per-turn decode rate p50/p5, prefix-cache hit rate (usage.prompt_tokens_details.cached_tokens / prompt
tokens), errors. Per-turn records go to the JSON.
usage: agent_replay.py URL MODEL OUT.json   (env: AGENTS=8 DURATION=1200 WARMUP=300 SEED=1 + the ranges below)
"""
import json, os, random, statistics, sys, threading, time, urllib.request

E = lambda k, d: type(d)(os.environ.get(k, d))
AGENTS, DURATION, WARMUP, SEED = E("AGENTS", 8), E("DURATION", 1200.0), E("WARMUP", 300.0), E("SEED", 1)
START_MIN, START_MAX = E("START_MIN", 60000), E("START_MAX", 150000)
APPEND_MIN, APPEND_MAX = E("APPEND_MIN", 500), E("APPEND_MAX", 5000)
REPLY_MIN, REPLY_MAX = E("REPLY_MIN", 200), E("REPLY_MAX", 1000)
TOOL_MIN, TOOL_MAX = E("TOOL_MIN", 1.0), E("TOOL_MAX", 5.0)
ROLL_AT = E("ROLL_AT", 230000)
TOKENS_PER_WORD = 2.2  # measured with this tokenizer on the word stream below (24 Oct)

WORDS = ("def class return import self value result error file path test assert config request response token cache "
         "block layer index buffer stream kernel queue batch thread lock state update commit branch merge diff patch "
         "line column module package build deploy").split()


def text(rng, n_tokens):
    n = max(1, int(n_tokens / TOKENS_PER_WORD))
    return " ".join(rng.choice(WORDS) + (str(rng.randint(0, 9999)) if rng.random() < 0.3 else "") for _ in range(n))


def stream_chat(url, model, messages, max_tokens):
    """Streamed completion; returns dict(ttft, seconds, tokens, text, prompt_tokens, cached_tokens) or raises."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "min_tokens": max_tokens,
            "temperature": 0.7, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    ttft, out, usage, n = None, [], {}, 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    out.append(piece)
                    n += 1
    return {"ttft": ttft, "seconds": time.time() - t0, "chunks": n, "text": "".join(out),
            "completion_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}


def agent(i, url, model, t_start, t_end, records, lock):
    rng = random.Random(SEED * 1000 + i)
    session, history_tokens = 0, 0
    messages = []
    while time.time() < t_end:
        if not messages or history_tokens > ROLL_AT:
            session += 1
            start = rng.randint(START_MIN, START_MAX)
            messages = [{"role": "system", "content": f"You are coding agent {i}, session {session}."},
                        {"role": "user", "content": "Repository context:\n" + text(rng, start)}]
            history_tokens, kind = start, "fresh"
        else:
            add = rng.randint(APPEND_MIN, APPEND_MAX)
            messages.append({"role": "user", "content": "Tool output:\n" + text(rng, add)})
            history_tokens += add
            kind = "append"
        reply = rng.randint(REPLY_MIN, REPLY_MAX)
        t_send = time.time()
        rec = {"agent": i, "session": session, "kind": kind, "t": round(t_send - t_start, 2),
               "history_est": history_tokens, "reply_target": reply}
        try:
            r = stream_chat(url, model, messages, reply)
            rec.update({k: r[k] for k in ("ttft", "seconds", "completion_tokens", "prompt_tokens", "cached_tokens")})
            messages.append({"role": "assistant", "content": r["text"]})
            history_tokens = (r["prompt_tokens"] or history_tokens) + (r["completion_tokens"] or reply)
        except Exception as e:  # recorded, the agent continues with a new session
            rec["error"] = repr(e)[:300]
            messages = []
        with lock:
            records.append(rec)
        pause = rng.uniform(TOOL_MIN, TOOL_MAX)
        time.sleep(max(0.0, min(pause, t_end - time.time())))


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 3) if xs else None


def summarize(records, warmup, duration):
    warm = [r for r in records if r["t"] >= warmup and "error" not in r and r.get("ttft") is not None]
    span_h = max(1e-9, (duration - warmup) / 3600)
    appends = [r for r in warm if r["kind"] == "append"]
    fresh = [r for r in warm if r["kind"] == "fresh"]
    rate = lambda r: (r["completion_tokens"] - 1) / (r["seconds"] - r["ttft"]) if (
        r.get("completion_tokens") and r["seconds"] > r["ttft"]) else None
    hit = (sum(r["cached_tokens"] or 0 for r in appends) / max(1, sum(r["prompt_tokens"] or 0 for r in appends)))
    return {
        "turns_per_hour": round(len(warm) / span_h, 1),
        "turn_s_p50": pct([r["seconds"] for r in warm], 0.5), "turn_s_p95": pct([r["seconds"] for r in warm], 0.95),
        "append_ttft_p50": pct([r["ttft"] for r in appends], 0.5),
        "append_ttft_p95": pct([r["ttft"] for r in appends], 0.95),
        "fresh_ttft_p50": pct([r["ttft"] for r in fresh], 0.5), "fresh_turns": len(fresh),
        "decode_tps_p50": pct([rate(r) for r in warm], 0.5), "decode_tps_p5": pct([rate(r) for r in warm], 0.05),
        "output_tokens_per_s": round(sum(r.get("completion_tokens") or 0 for r in warm) / (duration - warmup), 1),
        "prefix_hit_rate_appends": round(hit, 4),
        "errors": sum(1 for r in records if "error" in r),
        "warm_turns": len(warm),
    }


def main():
    url, model, out = sys.argv[1:4]
    records, lock = [], threading.Lock()
    t_start = time.time()
    t_end = t_start + DURATION
    th = [threading.Thread(target=agent, args=(i, url, model, t_start, t_end, records, lock), daemon=True)
          for i in range(AGENTS)]
    for t in th:
        t.start()
        time.sleep(0.5)
    for t in th:
        t.join(timeout=DURATION + 3600)
    summ = summarize(records, WARMUP, DURATION)
    cfg = {k: globals()[k] for k in ("AGENTS", "DURATION", "WARMUP", "SEED", "START_MIN", "START_MAX", "APPEND_MIN",
                                     "APPEND_MAX", "REPLY_MIN", "REPLY_MAX", "TOOL_MIN", "TOOL_MAX", "ROLL_AT")}
    json.dump({"config": cfg, "summary": summ, "turns": records}, open(out, "w"), indent=1)
    print("agent_replay", json.dumps(summ))
    sys.exit(1 if summ["errors"] or not summ["warm_turns"] else 0)


if __name__ == "__main__":
    main()
