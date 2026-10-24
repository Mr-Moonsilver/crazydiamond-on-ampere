#!/usr/bin/env python3
"""Per-worker coverage gate over engine log lines on stdin (Codex round 3: the last line of any worker is not enough).
  hostmeta: "QSA host meta coverage: host H fallback F verified V mismatches X" -- every worker of WORKERS must report;
            PASS needs, per worker (last line), H > 0 and V > 0; X == 0 on every line; no "host meta MISMATCH" line.
  hostmeta-used: the same lines in a performance run without verification: per worker H > 0, X == 0 everywhere.
  rowmix:   "QSA rowcache mixed steps: S seen, C with cached decode rows" -- per worker (last line) S > 0 and C > 0.
            (C counts mixed dispatches with cached decode rows (lead > 0), not verified cache hits.)
  retain:   S3 retained history with QSA_RETAIN_VERIFY=1: "QSA retained history active" from every worker, the stage
            timing's cumulative "verify n=V bad=B" per worker (last) V > 0, B == 0 on every line; no "retained history
            MISMATCH" and no "retained history disabled" line.
  retain-used: S3 retained history in a performance run (no verify): every worker active, >= 1 stage-timing window
            with retained reads ("ret=R", R > 0), no "retained history disabled" line.
BASELINE (Codex round 4: coverage must come from the test interval): an earlier snapshot of the same container's log.
Cumulative counters must then have advanced since it on every worker (delta > 0), windowed ones (ret=) must have new
windows; MISMATCH / disabled lines in the baseline count too (the engine is the same). Lines carry vLLM's
"(Worker_PPx_TPy pid=N)" prefix. Prints one summary line; exit 0 PASS, 1 FAIL.
usage: docker logs C 2>&1 | cov_gate.py hostmeta|rowmix|retain|retain-used [WORKERS=4] [BASELINE_LOG]"""
import re, sys

kind = sys.argv[1]
need = int(sys.argv[2]) if len(sys.argv) > 2 else 4
base_f = sys.argv[3] if len(sys.argv) > 3 else None
PAT = {"hostmeta": r"QSA host meta coverage: host (\d+) fallback (\d+) verified (\d+) mismatches (\d+)",
       "hostmeta-used": r"QSA host meta coverage: host (\d+) fallback (\d+) verified (\d+) mismatches (\d+)",
       "rowmix": r"QSA rowcache mixed steps: (\d+) seen, (\d+) with cached decode rows",
       "retain": r"QSA stage timing .*verify n=(\d+) bad=(\d+)",
       "retain-used": r"QSA stage timing .* ret=([1-9]\d*) "}[kind]
BAD_LINES = {"hostmeta": ("host meta MISMATCH",), "hostmeta-used": ("host meta MISMATCH",),
             "retain": ("retained history MISMATCH", "retained history disabled"),
             "retain-used": ("retained history disabled",)}.get(kind, ())
worker = lambda line: (re.search(r"\((Worker_PP\d+_TP\d+)", line) or [None, "?"])[1]


def scan(lines):
    last, windows, active, bad = {}, {}, set(), []
    for line in lines:
        for b in BAD_LINES:
            if b in line:
                bad.append(f"{worker(line)}: {b}")
        if "QSA retained history active" in line:
            active.add(worker(line))
        m = re.search(PAT, line)
        if not m:
            continue
        w, v = worker(line), tuple(map(int, m.groups()))
        last[w], windows[w] = v, windows.get(w, 0) + 1
        if kind.startswith("hostmeta") and v[3]:
            bad.append(f"{w} mismatches {v[3]}")
        if kind == "retain" and v[1]:
            bad.append(f"{w} verify bad {v[1]}")
    return last, windows, active, bad


last, windows, active, bad = scan(sys.stdin)
b_last, b_windows, _, b_bad = scan(open(base_f)) if base_f else ({}, {}, set(), [])
bad = sorted(set(bad) | set(b_bad))
delta = {w: tuple(x - y for x, y in zip(v, b_last.get(w, (0,) * len(v)))) for w, v in last.items()}
if len(last) < need:
    bad.append(f"{len(last)} of {need} workers reported")
if kind.startswith("retain") and len(active) < need:
    bad.append(f"retention active on {len(active)} of {need} workers")
for w, d in sorted(delta.items()):
    if kind == "hostmeta" and (d[0] <= 0 or d[2] <= 0):
        bad.append(f"{w} host +{d[0]} verified +{d[2]}")
    if kind == "hostmeta-used" and d[0] <= 0:
        bad.append(f"{w} host +{d[0]}")
    if kind == "rowmix" and (d[0] <= 0 or d[1] <= 0):
        bad.append(f"{w} seen +{d[0]} cached +{d[1]}")
    if kind == "retain" and d[0] <= 0:
        bad.append(f"{w} verified +{d[0]}")
    if kind == "retain-used" and windows[w] - b_windows.get(w, 0) <= 0:
        bad.append(f"{w} no new windows with retained reads")
shown = ({w: windows[w] - b_windows.get(w, 0) for w in sorted(windows)} if kind == "retain-used" else
         {w: delta[w] for w in sorted(delta)})
print(kind, "PASS" if not bad else "FAIL", ("delta" if base_f else "total"), shown, "|", "; ".join(bad[:8]))
sys.exit(1 if bad else 0)
