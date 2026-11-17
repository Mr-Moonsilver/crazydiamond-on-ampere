#!/usr/bin/env python3
"""q18a-attn-meta engine self-run on the c1b base: tools/attn_meta_engine_selfrun.py (never edited; superseded by name) with c1b's frozen
record and c1b's comparison. CPU only, no network.

  * OVERLAP_SUITES = c1b's (c1's plus test_alpha_next_c1b_queue), as c1b's own self-run; c1's seven parts and 65 connected cases, the
    attn/ part unchanged plus test_attn_meta_box_b;
  * frozen(): alpha-next-c1b b2's engine record, tools/attn_meta_parent_b.json (sha-pinned);
  * --compare: c1b's compare (every case in its own part: both PASS and the canonical trace byte-equal; a case whose frozen trace has
    no canonical form, the four receipts-* cases, the same verdict and check count) with attn-meta's schema: with the switch unset
    nothing moves. tools/attn_meta_engine_selfrun.py's base.compare was never exercised (chain a stopped before it);
  * --negative: two-phase, c1's judge, attn-meta's pins (tools/attn_meta_negative_pins.json, now hub-side: tools/attn_meta_spec_b.py).
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attn_meta_engine_selfrun as am  # noqa: E402
import alpha_next_c1b_engine_selfrun as c1b_run  # noqa: E402

base = am.base
SCHEMA = am.SCHEMA
PARTS = am.PARTS
ATTN_SUITES = am.ATTN_SUITES + ('test_attn_meta_box_b',)
NEGATIVE_SUITES = ATTN_SUITES
PINS = am.PINS
PARENT_PASS = am.PARENT_PASS
OVERLAP_SUITES = c1b_run.OVERLAP_SUITES
sha = am.sha
part_of = am.part_of
comparison_rules = am.comparison_rules
expected_cases = am.expected_cases
COMPARE_SWAPS = (("schema='alpha-next-c1b-engine-comparison/1'", "schema='attn-meta-default-off-comparison/1'"),
                 ("meaning='c1b moves no engine byte: every connected case, overlap on included, has alpha-next-c1\\'s exact canonical '",
                  "meaning='with the attn-meta switch unset every connected case, overlap on included, has alpha-next-c1b\\'s exact canonical '"))


def frozen():
    import attn_meta_parent_pin_b as parent_pin
    pins = parent_pin.load()
    assert 'engine_selfrun' in pins, 'the parent pin has no engine self-run (a development parent cannot be compared)'
    place = Path(pins['engine_selfrun'])
    assert sha(place / 'result.json') == pins['engine_selfrun_sha256'], 'c1b engine record changed'
    return place


def run_attn(bundle, out, name):
    with base_bound(am, dict(ATTN_SUITES=ATTN_SUITES)):
        return am.run_attn(bundle, out, name)


class base_bound:
    def __init__(self, module, names):
        self.module, self.names, self.saved = module, names, {}

    def __enter__(self):
        for key, value in self.names.items():
            self.saved[key] = getattr(self.module, key)
            setattr(self.module, key, value)

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            setattr(self.module, key, value)


_ns = dict(am._ns, ATTN_SUITES=ATTN_SUITES, NEGATIVE_SUITES=NEGATIVE_SUITES, run_attn=run_attn, OVERLAP_SUITES=OVERLAP_SUITES,
           frozen=frozen)
_run_ns = dict(_ns, CASES=base.CASES)
exec(compile(am.RUN_TEXT, 'attn_meta_engine_selfrun_b:run', 'exec'), _run_ns)
exec(compile(inspect.getsource(base.run_overlap), 'attn_meta_engine_selfrun_b:run_overlap', 'exec'), _run_ns)
for name, text in (('judge', am.JUDGE_TEXT), ('negative_judgement', am.NEGATIVE_JUDGEMENT_TEXT), ('negative', am.NEGATIVE_TEXT)):
    exec(compile(text, 'attn_meta_engine_selfrun_b:' + name, 'exec'), _ns)
_compare_text = inspect.getsource(c1b_run.compare)
for _old, _new in COMPARE_SWAPS:
    assert _compare_text.count(_old) == 1, ('c1b compare changed', _old)
    _compare_text = _compare_text.replace(_old, _new)
_compare_ns = dict(c1b_run.__dict__, frozen=frozen, expected_cases=expected_cases)
exec(compile(_compare_text, 'attn_meta_engine_selfrun_b:compare', 'exec'), _compare_ns)
run = _run_ns['run']
judge = _ns['judge']
negative = _ns['negative']
compare = _compare_ns['compare']

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
