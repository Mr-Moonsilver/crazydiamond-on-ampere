#!/usr/bin/env python3
"""alpha-next-c1 engine self-run, b: tools/alpha_next_c1_engine_selfrun.py (which chain a1 ran; superseded here, never edited) with
its judge reading the full case records (HARNESS, chain a1, 3 Nov 03:32 CET).

Why: a1's engine self-run had all seven containers PASS (rc 0, 1909 tests, skips 0, 65 cases PASS) and still FAILED: run() merged
each part's record ``cases`` (step_a.case_row summaries: sha256, checks, verdict) and judged C1's pairs (c1_pairs: request_tokens,
native, per_rank_segments_sha256) and C1's early open (c1.early_open: overlap.counters.early_opens) on them, so every C1 mode
'differed' and early_opens read None. C1's own tool (alpha_c1_engine_selfrun.run_stage) judges the container's full case dicts.
Recomputed over a1's case files: differ [], early_opens 31.

What: run() is the original's source with one exact-once swap (SWAP below): the pairs and the early open are judged on
connected_records(), the case files <part>/<mode>.json each part's container wrote (part_of(), the comparison's own map; a missing
file reads {}, so it fails closed). Its globals are a copy of the original module's (nothing there is mutated). The negative
(bounded, fail-closed), the comparison and every constant are the original's, unchanged; same CLI.
"""
import argparse
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_next_c1_engine_selfrun as base  # noqa: E402

SWAP = ("    pairs, differ = c1_pairs(connected)\n"
        "    early = c1.early_open(connected)\n",
        "    records = connected_records(out, connected)\n"
        "    pairs, differ = c1_pairs(records)\n"
        "    early = c1.early_open(records)\n")


def connected_records(out, connected):
    """The full case record of every mode the C1 pairs and the early open read (C1_PAIRS, their partners, EARLY_OPEN_CASE)."""
    need = set(base.C1_PAIRS) | {against for against, _fields in base.C1_PAIRS.values()} | {base.EARLY_OPEN_CASE}
    records = {}
    for mode in sorted(need & set(connected)):
        path = Path(out) / base.part_of(mode) / (mode + '.json')
        records[mode] = json.loads(path.read_text()) if path.is_file() else {}
    return records


def derive_run(module=base):
    text = inspect.getsource(module.run)
    old, new = SWAP
    assert text.count(old) == 1, 'run() changed: ' + old[:60]
    return text.replace(old, new)


RUN_TEXT = derive_run()
_ns = dict(base.__dict__, connected_records=connected_records)
exec(compile(RUN_TEXT, 'alpha_next_c1_engine_selfrun_b:run', 'exec'), _ns)
run = _ns['run']
negative = base.negative
compare = base.compare
expected_cases = base.expected_cases

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--spec', type=Path)
    p.add_argument('--tree', type=Path)
    p.add_argument('--bundle', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name')
    p.add_argument('--overlay', type=Path)
    p.add_argument('--sources', type=Path)
    p.add_argument('--negative', action='store_true')
    p.add_argument('--observe', action='store_true', help='negative only: record the outcomes, verdict OBSERVED, rc 1 (INC: rc 4)')
    p.add_argument('--compare', type=Path, help='candidate engine self-run directory')
    a = p.parse_args()
    if a.compare:
        raise SystemExit(compare(a.compare, a.out))
    if a.negative:
        raise SystemExit(negative(a.bundle, a.overlay, a.sources, a.out, a.name, a.observe))
    raise SystemExit(run(a.spec, a.tree, a.bundle, a.out, a.name))
