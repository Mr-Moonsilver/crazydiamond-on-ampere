#!/usr/bin/env python3
"""Write tools/alpha_next_c1_negative_pins.json from ONE observation of alpha-next-c1's negative control (tools/alpha_next_negative_pins_write.py,
ported: the probe cases are C1's, the parent is alpha-next, the frozen record is alpha-next's pinned engine self-run).

The observation is alpha_next_c1_engine_selfrun.py --negative --observe over the chain's own build: the candidate harness (the
overlap suites, the alpha-c1 probe cases, alpha-next's parent cases) over alpha-next's regenerated overlay and sources, the candidate
hidden. The lead's conditions carried from the sweep: observe exactly once, never again for a cleaner set; pin every
non-pass with a reason and a class; any ENGINE non-pass stops and gets a write-up, not a pin.

The reasons come from a reviewed classification file (kept with the evidence, bound here by sha), a JSON list of
[test id or case prefix, regex on the outcome text, reason, class]. The first matching row explains a non-pass. Classes:
  HARNESS  the candidate's test or probe mode needs what only the candidate has (C1's stage_overlap module, the overlap switch, a
           candidate harness change), which alpha-next's engine lacks: the control biting as designed;
  ENGINE   anything else that points at alpha-next's engine itself (a crash, an alpha-next behaviour that differs): refused here.
Tests: every non-pass of the observed suites is pinned [kind, text] with its reason and class. Cases: every alpha-c1 probe
mode must FAIL in the observation; its [verdict, error_type, error] is read from the raw case file next to the record (the record's
rows cut the error at 400 characters) and must be explained by a row (the case name is matched against the prefix, the
"error_type: error" text against the regex). A case that passes is refused (the control would not bite). Unexplained non-passes are
refused. The observation must be OBSERVED, have no skips, every parent case PASS, and no reason other than a probe case differing
from the stub's expectation (those are recorded under cases_vs_expected). The pins are bound to the frozen alpha-next record the
self-run reads (frozen_record_sha256: null in the stub, which is written before the parent is pinned; filled here from the
pinned record, or asserted equal when the stub carries it).

The pins file is committed as a stub (status starting AWAITING-OBSERVATION, observation null); this tool replaces the stub once and refuses a
second write (one observation only). --check recomputes the pins from the same inputs and compares them byte for byte with the file.
The tool is committed before the observation, so the pins commit holds the pins file only. No GPU, no box action; reads files only.
usage: alpha_next_c1_negative_pins_write.py --observed OBS/result.json --log OBS_STDOUT --classify FILE --discovery TEXT [--check]
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
OUT = ROOT / 'tools/alpha_next_c1_negative_pins.json'
SCHEMA = 'alpha-next-c1-negative-pins/1'
OBSERVED_SCHEMA = 'alpha-next-c1-negative/1'
AWAITING = 'AWAITING-OBSERVATION'
CLASSES = ('HARNESS', 'ENGINE')
ALLOWED_REASON = 'negative-case-differs:'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path):
    rows = json.loads(Path(path).read_text())
    assert isinstance(rows, list) and rows, 'classification rows'
    for row in rows:
        assert isinstance(row, list) and len(row) == 4 and all(isinstance(x, str) and x for x in row) and row[3] in CLASSES, row
        re.compile(row[1])
    return rows


def explain(rows, name, text):
    for prefix, pattern, reason, klass in rows:
        if name.startswith(prefix) and re.search(pattern, text):
            return reason, klass
    return None


def rec_frozen(observed):
    return json.loads(Path(observed).read_text()).get('frozen_record_sha256')


def base_of(current):
    """The stub's persistent fields and its expected case rows (a written file keeps them under expected_cases)."""
    assert current['schema'] == SCHEMA, current.get('schema')
    if str(current.get('status')).startswith(AWAITING):
        assert current.get('observation') is None and not current.get('tests'), 'stub carries observed fields'
        expected = current['cases']
    else:
        assert current.get('status') == 'OBSERVED' and current.get('observation'), current.get('status')
        expected = current['expected_cases']
    return dict(frozen_record_sha256=current['frozen_record_sha256'], cases_from=current['cases_from']), expected


def pins(current, observed, log, classify, discovery):
    import alpha_next_c1_engine_selfrun as selfrun
    base, expected = base_of(current)
    frozen_sha = sha(selfrun.frozen() / 'result.json')
    assert base['frozen_record_sha256'] in (None, frozen_sha), 'the stub was taken against another frozen record'
    base['frozen_record_sha256'] = frozen_sha
    assert rec_frozen(observed) == frozen_sha, 'the observation ran against another frozen record'
    rec = json.loads(Path(observed).read_text())
    assert rec['schema'] == OBSERVED_SCHEMA and rec['verdict'] == 'OBSERVED' and rec['negative_control'] is True, \
        (rec.get('schema'), rec.get('verdict'))
    assert rec['pins_sha256'] is None and rec['pinned_tests'] == 0, 'not an --observe record'
    assert rec['suites'].get('skips') == 0, 'observation has skips'
    assert rec['suite_names'] == list(selfrun.NEGATIVE_SUITES), 'observed suites are not this self-run\'s'
    other = [r for r in rec['reasons'] if not r.startswith(ALLOWED_REASON)]
    assert not other, ('observation reasons beyond the probe cases: write up, do not pin', other)
    probe = tuple(selfrun.CASES)
    assert sorted(expected) == sorted(probe), 'stub cases are not the probe cases'
    bad = sorted(m for m in selfrun.PARENT_PASS if (rec['cases'].get(m) or {}).get('verdict') != 'PASS')
    assert not bad, ('parent cases not PASS in the observation', bad)
    rows = load_rows(classify)
    unexplained, engine = [], []

    def classify_one(name, kind, text, into):
        got = explain(rows, name, text)
        if got is None:
            unexplained.append((name, kind, text))
        elif got[1] == 'ENGINE':
            engine.append((name, kind, text, got[0]))
        else:
            into[name] = got

    tests, test_why = {}, {}
    for test, (kind, text) in sorted(rec['observed'].items()):
        classify_one(test, kind, text, test_why)
        if test in test_why:
            tests[test] = [kind, text]
    cases, case_why, passing = {}, {}, []
    place = Path(observed).resolve().parent
    for mode in sorted(probe):
        raw = json.loads((place / (mode + '.json')).read_text()) if (place / (mode + '.json')).exists() else {}
        row = [raw.get('verdict'), raw.get('error_type'), raw.get('error')]
        assert row[0] == (rec['cases'].get(mode) or {}).get('verdict'), ('raw case and record disagree', mode)
        if row[0] != 'FAIL' or not row[1]:
            passing.append((mode, row))
            continue
        classify_one(mode, row[0], '%s: %s' % (row[1], row[2]), case_why)
        if mode in case_why:
            cases[mode] = row
    assert not passing, ('probe cases that do not fail on the parent: the control would not bite; write up', passing)
    assert not engine, ('ENGINE non-passes: write up, do not pin', engine)
    assert not unexplained, ('non-passes no reviewed row explains: write up, do not pin', unexplained)
    assert tests, 'nothing to pin: the negative control would not bite on the suites'
    return dict(schema=SCHEMA, status='OBSERVED', discovery=discovery, frozen_record_sha256=base['frozen_record_sha256'],
                cases_from=base['cases_from'], tests=tests,
                reasons={t: r for t, (r, _) in test_why.items()}, classes={t: k for t, (_, k) in test_why.items()},
                cases=cases, case_reasons={m: r for m, (r, _) in case_why.items()},
                case_classes={m: k for m, (_, k) in case_why.items()},
                expected_cases=expected, cases_vs_expected={m: dict(expected=expected[m], observed=cases[m])
                                                            for m in sorted(cases) if cases[m] != expected[m]},
                why=dict(rule='every non-pass of one observation over the chain\'s own build, each with its reviewed reason and class; '
                              'negative() refuses any other non-pass, any pinned test that passes or changes kind or text, and any '
                              'probe case whose verdict, error type or error differs from its pin',
                         classes={'HARNESS': 'the candidate\'s test or probe needs what only the candidate has; the control biting '
                                             'as designed',
                                  'ENGINE': 'points at alpha-next\'s engine itself; refused, written up'}),
                classification=dict(path=str(Path(classify).resolve()), sha256=sha(classify), rows=rows),
                observation=dict(observed_json=str(Path(observed).resolve()), observed_sha256=sha(observed),
                                 log=str(Path(log).resolve()), log_sha256=sha(log), tests_run=rec['suites'].get('tests'),
                                 not_passing=len(tests), cases_failing=len(cases), skips=rec['suites'].get('skips'),
                                 overlay_manifest=rec['overlay_manifest'], bundle_manifest=rec['manifest'],
                                 container=rec['container'], reasons=rec['reasons']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--observed', type=Path, required=True)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--classify', type=Path, required=True)
    p.add_argument('--discovery', required=True)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    current = json.loads(OUT.read_text())
    text = json.dumps(pins(current, a.observed, a.log, a.classify, a.discovery), indent=1, sort_keys=True) + '\n'
    if a.check:
        assert current.get('status') == 'OBSERVED', 'pins not written yet'
        assert OUT.read_text() == text, 'pins differ from the observation'
        print('negative pins verified')
    else:
        assert str(current.get('status')).startswith(AWAITING), 'pins already written: one observation only'
        OUT.write_text(text)
        print('pinned %d tests, %d cases' % (len(json.loads(text)['tests']), len(json.loads(text)['cases'])))
