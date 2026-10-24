#!/usr/bin/env python3
"""Per-worker coverage gate over engine log lines on stdin (Codex round 3: the last line of any worker is not enough).
  hostmeta: "QSA host meta coverage: host H fallback F verified V mismatches X" -- every worker of WORKERS must report;
            PASS needs, per worker (last line), H > 0 and V > 0; X == 0 on every line; no "host meta MISMATCH" line.
  rowmix:   "QSA rowcache mixed steps: S seen, C with cached decode rows" -- per worker (last line) S > 0 and C > 0.
Lines carry vLLM's "(Worker_PPx_TPy pid=N)" prefix. Prints one summary line; exit 0 PASS, 1 FAIL.
usage: docker logs C 2>&1 | cov_gate.py hostmeta|rowmix [WORKERS=4]"""
import re, sys
kind, need = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4
pat = {"hostmeta": r"QSA host meta coverage: host (\d+) fallback (\d+) verified (\d+) mismatches (\d+)",
       "rowmix": r"QSA rowcache mixed steps: (\d+) seen, (\d+) with cached decode rows"}[kind]
last, bad, mism = {}, [], 0
for line in sys.stdin:
    if "host meta MISMATCH" in line:
        mism += 1
    m = re.search(pat, line)
    if not m:
        continue
    w = re.search(r"\((Worker_PP\d+_TP\d+)", line)
    w = w.group(1) if w else "?"
    v = tuple(map(int, m.groups()))
    last[w] = v
    if kind == "hostmeta" and v[3]:
        bad.append(f"{w} mismatches {v[3]}")
if len(last) < need:
    bad.append(f"{len(last)} of {need} workers reported")
for w, v in sorted(last.items()):
    if kind == "hostmeta" and (v[0] == 0 or v[2] == 0):
        bad.append(f"{w} host {v[0]} verified {v[2]}")
    if kind == "rowmix" and (v[0] == 0 or v[1] == 0):
        bad.append(f"{w} seen {v[0]} cached {v[1]}")
if mism:
    bad.append(f"{mism} MISMATCH lines")
print(kind, "PASS" if not bad else "FAIL", {w: v for w, v in sorted(last.items())}, "|", "; ".join(bad))
sys.exit(1 if bad else 0)
