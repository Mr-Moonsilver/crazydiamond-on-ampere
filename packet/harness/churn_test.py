#!/usr/bin/env python3
"""Churn retrieval test: more concurrent requests than the server admits, varied lengths, so requests
queue, finish at different times and vLLM compacts its batch. Each request asks for the codes of five
random records in its own haystack; every code is checked. A cache serving another request's rows
shows up as wrong codes.
usage: churn_test.py URL MODEL REQUESTS CONCURRENCY OUT.json"""
import concurrent.futures, json, random, re, sys, urllib.request

URL, MODEL, N, C, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]


def case(i):
    rnd = random.Random(1000 + i)
    n_records = rnd.randint(1000, 4000)
    salt = rnd.randint(1, 99999)
    code = lambda r: f"C{(r * 7919 + salt) % 1000003:07d}"
    hay = "\n".join(f"Record {r:05d}: code = {code(r)}." for r in range(n_records))
    asked = rnd.sample(range(n_records), 5)
    question = (f"{hay}\n\nGive the codes of records {', '.join(f'{r:05d}' for r in asked)}, in that order, "
                "as a JSON list of strings and nothing else.")
    return question, [code(r) for r in asked], rnd.choice([64, 128, 256])


def run(i):
    question, expected, max_tokens = case(i)
    body = {"model": MODEL, "messages": [{"role": "user", "content": question}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    text = r["choices"][0]["message"]["content"] or ""
    got = re.findall(r"C\d{7}", text)
    correct = sum(g == e for g, e in zip(got, expected))
    return {"i": i, "prompt_tokens": r["usage"]["prompt_tokens"], "correct": correct, "expected": expected, "got": got}


with concurrent.futures.ThreadPoolExecutor(max_workers=C) as pool:
    rows = list(pool.map(run, range(N)))
json.dump(rows, open(OUT, "w"), indent=1)
total = sum(r["correct"] for r in rows)
print(json.dumps({"requests": N, "codes_correct": total, "codes_asked": 5 * N,
                  "requests_all_correct": sum(r["correct"] == 5 for r in rows),
                  "prompt_tokens_range": [min(r["prompt_tokens"] for r in rows), max(r["prompt_tokens"] for r in rows)]}))
