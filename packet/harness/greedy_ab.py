#!/usr/bin/env python3
"""Greedy outputs for a fixed prompt set, one request at a time (batch composition fixed).
usage: greedy_ab.py URL MODEL OUT.json   -> prints a one-line summary; compare two OUT files with --compare A B
"""
import json, sys, urllib.request

if sys.argv[1] == "--compare":
    a, b = (json.load(open(p)) for p in sys.argv[2:4])
    same = [k for k in a if a[k]["text"] == b[k]["text"]]
    diff = [k for k in a if k not in same]
    first = {}
    for k in diff:
        ta, tb = a[k]["text"], b[k]["text"]
        n = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        first[k] = n
    print(json.dumps({"identical": same, "different": diff, "first_diff_char": first,
                      "retrieval": {k: (a[k].get("ok"), b[k].get("ok")) for k in a if "ok" in a[k]}}))
    sys.exit(0)

URL, MODEL, OUT = sys.argv[1:4]
corpus = open("/home/<service-user>/p2p-tests/corpus.txt", encoding="utf-8", errors="ignore").read()
records = "\n".join(f"Record {i:05d}: code = CODE-{(i * 7919 + 101) % 100000:05d}." for i in range(3000))
cases = {
    "retrieval_30k": (records + "\nWhat is the code in record 01731? Answer with the code only.", 16,
                      f"CODE-{(1731 * 7919 + 101) % 100000:05d}"),
    "retrieval_30k_b": (records + "\nWhat is the code in record 02904? Answer with the code only.", 16,
                        f"CODE-{(2904 * 7919 + 101) % 100000:05d}"),
    "arith": ("What is (137 * 29) + 58? Give only the number.", 8, "4031"),
    "translate": ("Translate into French, returning only the translation: The meeting takes place on Tuesday in Lausanne.", 32, None),
    "continue_16k": (corpus[200000:200000 + 64000] + "\n\nSummarise the passage above in three sentences.", 256, None),
}
out = {}
for name, (prompt, max_tokens, expect) in cases.items():
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "seed": 1, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1200))
    text = r["choices"][0]["message"]["content"] or ""
    row = {"text": text, "prompt_tokens": r["usage"]["prompt_tokens"]}
    if expect:
        row["ok"] = expect in text
    out[name] = row
json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: {"prompt_tokens": v["prompt_tokens"], "ok": v.get("ok"), "head": v["text"][:60]} for k, v in out.items()},
                 ensure_ascii=False))
