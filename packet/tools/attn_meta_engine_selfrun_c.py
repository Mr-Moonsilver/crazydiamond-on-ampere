#!/usr/bin/env python3
"""q18a-attn-meta engine self-run, negative fixed: tools/attn_meta_engine_selfrun_b.py (chain b ran it; never edited; superseded by name,
lead 3 Nov ~20:00). CPU only, no network.

The defect (chain b, 3 Nov 19:18:37 CET, rc 1, HARNESS, no negative observed): c1's negative() assembles `modes = PARENT_PASS + CASES`.
c1 has two tuples; tools/attn_meta_engine_selfrun.py holds PARENT_PASS as a list and CASES=(), so the assembly raised TypeError before
any container ran. Fix, one name: PARENT_PASS is a tuple again, the type c1 hands to the same text and to step_a.container. Same
members in the same order. CASES stays () (attn-meta pins no probe case of its own).

run(), compare() and every other name are _b's objects, unchanged. judge, negative_judgement and negative are c1's texts through _b's
derivation, re-executed in this namespace so that they read the fixed name. harness/test_attn_meta_engine_selfrun_c.py runs negative()
offline through both phases and the pin writer; the old _b assembly is its mutant.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attn_meta_engine_selfrun_b as b  # noqa: E402

am = b.am
base = b.base
SCHEMA = b.SCHEMA
PARTS = b.PARTS
ATTN_SUITES = b.ATTN_SUITES
NEGATIVE_SUITES = b.NEGATIVE_SUITES
PINS = b.PINS
OVERLAP_SUITES = b.OVERLAP_SUITES
PARENT_PASS = tuple(b.PARENT_PASS)
sha = b.sha
part_of = b.part_of
comparison_rules = b.comparison_rules
expected_cases = b.expected_cases
frozen = b.frozen
run_attn = b.run_attn
run = b.run
compare = b.compare

_ns = dict(b._ns, PARENT_PASS=PARENT_PASS)
for _name, _text in (('judge', am.JUDGE_TEXT), ('negative_judgement', am.NEGATIVE_JUDGEMENT_TEXT), ('negative', am.NEGATIVE_TEXT)):
    exec(compile(_text, 'attn_meta_engine_selfrun_c:' + _name, 'exec'), _ns)
judge = _ns['judge']
negative = _ns['negative']

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
