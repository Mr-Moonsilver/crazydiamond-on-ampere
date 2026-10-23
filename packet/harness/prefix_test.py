#!/usr/bin/env python3
"""Multi-turn prefix-reuse test, shaped like a chat client: each conversation sends a long haystack with
question 1, then the full history (haystack, question 1, the model's answer) plus question 2, then the same
again with question 3. Reports time to first token per turn (streamed) and checks every answer.
usage: prefix_test.py URL MODEL CONVERSATIONS RECORDS OUT.json   (RECORDS 8000 ~ 150k tokens)"""
import concurrent.futures, json, random, sys, time, urllib.request

URL, MODEL, CONV, RECORDS, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]


def chat(messages, max_tokens=48):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, ttft, text, usage = time.time(), None, "", None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                ttft = ttft or time.time() - t0
                text += d["choices"][0]["delta"]["content"]
            usage = d.get("usage") or usage
    return text, ttft, time.time() - t0, usage


def conversation(c):
    rnd = random.Random(500 + c)
    salt = rnd.randint(1, 99999)
    code = lambda r: f"C{(r * 7919 + salt) % 1000003:07d}"
    hay = "\n".join(f"Record {r:05d}: code = {code(r)}." for r in range(RECORDS))
    messages, turns = [], []
    for t in range(3):
        rec = rnd.randrange(RECORDS)
        q = (hay + "\n\n" if t == 0 else "") + f"What is the code of record {rec:05d}? Answer with the code only."
        messages.append({"role": "user", "content": q})
        text, ttft, total, usage = chat(messages)
        messages.append({"role": "assistant", "content": text})
        turns.append({"turn": t + 1, "ttft_s": round(ttft or -1, 2), "total_s": round(total, 2),
                      "prompt_tokens": (usage or {}).get("prompt_tokens"),
                      "ok": code(rec) in text, "text": text[:40]})
    return {"conversation": c, "turns": turns}


with concurrent.futures.ThreadPoolExecutor(max_workers=CONV) as pool:
    rows = list(pool.map(conversation, range(CONV)))
json.dump(rows, open(OUT, "w"), indent=1)
summary = {f"turn{t}": {"ttft_s": [r["turns"][t - 1]["ttft_s"] for r in rows],
                        "ok": sum(r["turns"][t - 1]["ok"] for r in rows)} for t in (1, 2, 3)}
summary["prompt_tokens_turn1"] = rows[0]["turns"][0]["prompt_tokens"]
print(json.dumps(summary))
