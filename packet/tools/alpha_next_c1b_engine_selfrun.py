#!/usr/bin/env python3
"""alpha-next-c1b engine self-run: tools/alpha_next_c1_engine_selfrun_b.py's run (c1's seven containers, the same 65 connected cases,
the full-record judge), its negative and its bounded wall guard, imported and re-bound, never copy-edited. CPU only, no network.

What differs from c1's, and only this (each function below is the c1 _b / base source exec'd in a namespace that is a copy of the
base module's globals with these names re-bound; schema strings swapped exactly once each):
  * SCHEMA 'alpha-next-c1b-selfrun/1' (the frozen c1b runner's swap), the overlap part 'alpha-next-c1b-overlap-selfrun/1', the
    negative 'alpha-next-c1b-negative/1';
  * OVERLAP_SUITES = c1's plus test_alpha_next_c1b_queue (as c1 added its own queue suite to the overlap container); the cases are
    c1's, unchanged (expected_cases() is c1's);
  * frozen(): alpha-next-c1's delivered engine record (tools/alpha_next_c1b_parent.json engine_selfrun, sha-pinned);
  * PINS: tools/alpha_next_c1b_negative_pins.json.

Comparison (--compare, c1b's own): the engine bytes do not move, so NO engine case may differ from alpha-next-c1's frozen record.
The candidate record's case set, the frozen record's and expected_cases() must be equal (65). Every case is compared in its own
part against the frozen record's file of the same case: both PASS, and the canonical trace (held_f.CANONICAL, c1's comparison's
sameness) byte-equal; a case whose frozen trace has no canonical form (the four receipts-* cases) must have the same verdict
(PASS) and the same check count. Overlap-on cases are included (their canonical traces were equal across c1's own runs a1, a2, b).

Negative (--negative, two-phase as c1's): the c1b harness over alpha-next-c1's regenerated overlay and sources (build OUT/parent,
byte-identical to c1's). Since c1b moves no engine byte, the expectation is that nothing bites: every alpha-c1 probe case PASSes
with c1's frozen trace (the stub pins them ['PASS', None, None]), every parent case PASSes with its frozen trace, and the suites'
non-passes are whatever the one observation shows (pinned with reasons by tools/alpha_next_c1b_negative_pins_write.py; zero is
expected and is pinned explicitly as no_failing_tests, which lifts c1's 'negative-suites-not-run' refusal only when the suites
reported outcomes). The wall guard and its INC rule are c1's, unchanged.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_next_c1_engine_selfrun_b as c1_b  # noqa: E402  (installs c1's and alpha-next's command lines, as c1's chain did)

base = c1_b.base
SCHEMA = 'alpha-next-c1b-selfrun/1'
OVERLAP_SCHEMA = ("schema='alpha-next-c1-overlap-selfrun/1'", "schema='alpha-next-c1b-overlap-selfrun/1'")
NEGATIVE_SCHEMA = ("schema='alpha-next-c1-negative/1'", "schema='alpha-next-c1b-negative/1'")
NEGATIVE_KIND = ("negative_kind='alpha-next-parent'", "negative_kind='alpha-next-c1-parent'")
QUEUE_SUITE = 'test_alpha_next_c1b_queue'
OVERLAP_SUITES = tuple(base.OVERLAP_SUITES) + (QUEUE_SUITE,)
NEGATIVE_SUITES = OVERLAP_SUITES
CASES = base.CASES
PINS = Path(__file__).resolve().with_name('alpha_next_c1b_negative_pins.json')
sha = base.sha
canonical = base.canonical
part_of = base.part_of
expected_cases = base.expected_cases


def _pin():
    import alpha_next_c1b_parent_pin as parent_pin
    return parent_pin.load()


def frozen():
    """alpha-next-c1's frozen engine record, read from the c1b parent file and pinned there."""
    pins = _pin()
    assert 'engine_selfrun' in pins, 'the parent pin has no engine self-run (a development parent cannot be compared)'
    place = Path(pins['engine_selfrun'])
    assert sha(place / 'result.json') == pins['engine_selfrun_sha256'], 'alpha-next-c1 engine record changed'
    return place


def judge(pins, outcomes, cases, place, observe=False):
    """c1's judge, plus: the alpha-c1 probe cases' traces equal the frozen c1 record's, and an explicit no_failing_tests pin
    (one observation with zero non-passes) lifts 'negative-suites-not-run' when the suites did report outcomes."""
    reasons = base.judge(pins, outcomes, cases, place, observe)
    if pins.get('no_failing_tests') is True and outcomes and 'negative-suites-not-run' in reasons:
        reasons.remove('negative-suites-not-run')
    for mode in CASES:
        frozen_case = place / part_of(mode) / (mode + '.json')
        theirs = canonical(json.loads(frozen_case.read_text())) if frozen_case.is_file() else None
        if not cases.get(mode) or theirs is None or canonical(cases[mode]) != theirs:
            reasons.append('negative-trace-differs-from-frozen:' + mode)
    return reasons


def _swap(text, pair):
    assert text.count(pair[0]) == 1, ('c1 source changed', pair[0])
    return text.replace(pair[0], pair[1])


SOURCES = dict(run_overlap=(inspect.getsource(base.run_overlap), [OVERLAP_SCHEMA]),
               run=(c1_b.RUN_TEXT, []),
               negative_judgement=(inspect.getsource(base.negative_judgement), []),
               negative=(inspect.getsource(base.negative), [NEGATIVE_SCHEMA, NEGATIVE_KIND]))
NS = dict(base.__dict__, SCHEMA=SCHEMA, OVERLAP_SUITES=OVERLAP_SUITES, NEGATIVE_SUITES=NEGATIVE_SUITES, PINS=PINS,
          frozen=frozen, _pin=_pin, judge=judge, connected_records=c1_b.connected_records)
for _name, (_text, _pairs) in SOURCES.items():
    for _pair in _pairs:
        _text = _swap(_text, _pair)
    exec(compile(_text, 'alpha_next_c1b_engine_selfrun:' + _name, 'exec'), NS)
run, run_overlap, negative = NS['run'], NS['run_overlap'], NS['negative']


def compare(candidate, out):
    """Every engine case of the candidate equals alpha-next-c1's frozen record's (see the module docstring)."""
    candidate, place = Path(candidate).resolve(), frozen()
    ours_rec = json.loads((candidate / 'result.json').read_text())
    theirs_rec = json.loads((place / 'result.json').read_text())
    want = expected_cases()
    problems = []
    case_sets = dict(candidate=sorted(ours_rec.get('cases') or {}), frozen=sorted(theirs_rec.get('cases') or {}), expected=want)
    if not case_sets['candidate'] == case_sets['frozen'] == want:
        problems.append('case-set-differs')
    rows = {}
    for mode in want:
        part = part_of(mode)
        mine, theirs = candidate / part / (mode + '.json'), place / part / (mode + '.json')
        if not mine.is_file() or not theirs.is_file():
            rows[mode] = dict(same=False, why='case file missing', part=part)
            problems.append(mode)
            continue
        ours, frozen_case = json.loads(mine.read_text()), json.loads(theirs.read_text())
        a, b = canonical(ours), canonical(frozen_case)
        verdicts = ours.get('verdict') == frozen_case.get('verdict') == 'PASS'
        if b is None:
            ours_row, theirs_row = (ours_rec.get('cases') or {}).get(mode) or {}, (theirs_rec.get('cases') or {}).get(mode) or {}
            same = verdicts and a is None and ours_row.get('checks') == theirs_row.get('checks') is not None
            rows[mode] = dict(same=same, part=part, by='verdict-and-checks (no canonical trace in the frozen record)',
                              checks=ours_row.get('checks'))
        else:
            same = verdicts and a == b
            rows[mode] = dict(same=same, part=part, by='canonical-trace',
                              canonical_sha256=hashlib.sha256(b).hexdigest() if same else None)
        if not same:
            problems.append(mode)
    result = dict(schema='alpha-next-c1b-engine-comparison/1', verdict='FAIL' if problems else 'PASS', differs=problems,
                  case_sets=case_sets, fields=list(base.held_f.CANONICAL), modes=rows,
                  compared=sum(1 for v in rows.values() if v.get('by') == 'canonical-trace'),
                  verdict_and_checks=sum(1 for v in rows.values() if str(v.get('by', '')).startswith('verdict-and-checks')),
                  parent_record=str(place / 'result.json'), parent_record_sha256=sha(place / 'result.json'),
                  candidate=str(candidate), candidate_record_sha256=sha(candidate / 'result.json'),
                  meaning='c1b moves no engine byte: every connected case, overlap on included, has alpha-next-c1\'s exact canonical '
                          'trace (or, without one, its verdict and check count)')
    with Path(out).open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2)
        f.write('\n')
    print(result['verdict'], problems)
    return 0 if not problems else 1


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
