#!/usr/bin/env python3
"""Write tools/alpha_next_c1b_negative_pins.json from ONE observation of alpha-next-c1b's negative control
(tools/alpha_next_c1_negative_pins_write.py's rules, for a harness-only package).

The observation is alpha_next_c1b_engine_selfrun.py --negative --observe over the chain's own build: the c1b harness (c1's overlap
suites plus test_alpha_next_c1b_queue, the alpha-c1 probe cases, alpha-next's parent cases) over alpha-next-c1's regenerated
overlay and sources (byte-identical to c1's). c1b moves no engine byte, so the expectation differs from c1's: every alpha-c1 probe
case must PASS (row ['PASS', None, None], as the stub carries it; a failing probe case is refused here and written up, it would
mean the c1b harness depends on something c1's engine lacks); every parent case PASS. Suite non-passes, if any, are pinned
[kind, text] each with a reviewed reason and class from a classification file (rows [test prefix, regex, reason, class]; HARNESS:
the c1b test needs what only the c1b bundle has, e.g. its own review; ENGINE: refused, written up). Zero non-passes is the expected
outcome and is pinned explicitly (no_failing_tests: true, tests {}); then --classify may be omitted.
Rules kept from c1's writer: observe exactly once (the stub is replaced once, a second write is refused); the observation must be
OBSERVED, from --observe (pins_sha256 null, pinned_tests 0), skips 0, its suites this self-run's, its frozen record the pinned c1
record, and its reasons limited to probe cases differing from the stub's expectation (those are refused here anyway).
--check recomputes and compares byte for byte. No GPU, no box action; reads files only.
usage: alpha_next_c1b_negative_pins_write.py --observed OBS/result.json --log OBS_STDOUT [--classify FILE] --discovery TEXT [--check]
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import alpha_next_c1_negative_pins_write as c1_writer  # noqa: E402  (load_rows, explain: unchanged)

OUT = ROOT / 'tools/alpha_next_c1b_negative_pins.json'
SCHEMA = 'alpha-next-c1b-negative-pins/1'
OBSERVED_SCHEMA = 'alpha-next-c1b-negative/1'
AWAITING = 'AWAITING-OBSERVATION'
ALLOWED_REASON = 'negative-case-differs:'
PASS_ROW = ['PASS', None, None]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_of(current):
    assert current['schema'] == SCHEMA, current.get('schema')
    if str(current.get('status')).startswith(AWAITING):
        assert current.get('observation') is None and not current.get('tests'), 'stub carries observed fields'
        expected = current['cases']
    else:
        assert current.get('status') == 'OBSERVED' and current.get('observation'), current.get('status')
        expected = current['expected_cases']
    return dict(frozen_record_sha256=current['frozen_record_sha256'], cases_from=current['cases_from']), expected


def pins(current, observed, log, classify, discovery, selfrun=None):
    if selfrun is None:
        import alpha_next_c1b_engine_selfrun as selfrun
    base, expected = base_of(current)
    frozen_sha = sha(selfrun.frozen() / 'result.json')
    assert base['frozen_record_sha256'] in (None, frozen_sha), 'the stub was taken against another frozen record'
    base['frozen_record_sha256'] = frozen_sha
    rec = json.loads(Path(observed).read_text())
    assert rec.get('frozen_record_sha256') == frozen_sha, 'the observation ran against another frozen record'
    assert rec['schema'] == OBSERVED_SCHEMA and rec['verdict'] == 'OBSERVED' and rec['negative_control'] is True, \
        (rec.get('schema'), rec.get('verdict'))
    assert rec['pins_sha256'] is None and rec['pinned_tests'] == 0, 'not an --observe record'
    assert rec['suites'].get('skips') == 0, 'observation has skips'
    assert rec['suite_names'] == list(selfrun.NEGATIVE_SUITES), 'observed suites are not this self-run\'s'
    other = [r for r in rec['reasons'] if not r.startswith(ALLOWED_REASON)]
    assert not other, ('observation reasons beyond the probe cases: write up, do not pin', other)
    assert sorted(expected) == sorted(selfrun.CASES) and all(expected[m] == PASS_ROW for m in expected), 'stub cases'
    bad = sorted(m for m in selfrun.PARENT_PASS if (rec['cases'].get(m) or {}).get('verdict') != 'PASS')
    assert not bad, ('parent cases not PASS in the observation', bad)
    place = Path(observed).resolve().parent
    cases, failing = {}, []
    for mode in sorted(selfrun.CASES):
        raw = json.loads((place / (mode + '.json')).read_text()) if (place / (mode + '.json')).exists() else {}
        row = [raw.get('verdict'), raw.get('error_type'), raw.get('error')]
        assert row[0] == (rec['cases'].get(mode) or {}).get('verdict'), ('raw case and record disagree', mode)
        if row != PASS_ROW:
            failing.append((mode, row))
        cases[mode] = row
    assert not failing, ('probe cases that do not PASS over c1\'s engine: the c1b harness depends on an engine change; write up',
                         failing)
    observed_tests = sorted((rec.get('observed') or {}).items())
    tests, why = {}, {}
    if observed_tests:
        assert classify is not None, 'non-passes observed: a reviewed --classify file is required'
        rows = c1_writer.load_rows(classify)
        unexplained, engine = [], []
        for test, (kind, text) in observed_tests:
            got = c1_writer.explain(rows, test, text)
            if got is None:
                unexplained.append((test, kind, text))
            elif got[1] == 'ENGINE':
                engine.append((test, kind, text, got[0]))
            else:
                tests[test], why[test] = [kind, text], got
        assert not engine, ('ENGINE non-passes: write up, do not pin', engine)
        assert not unexplained, ('non-passes no reviewed row explains: write up, do not pin', unexplained)
    return dict(schema=SCHEMA, status='OBSERVED', discovery=discovery, frozen_record_sha256=frozen_sha,
                cases_from=base['cases_from'], tests=tests, no_failing_tests=not tests,
                reasons={t: r for t, (r, _) in why.items()}, classes={t: k for t, (_, k) in why.items()},
                cases=cases, expected_cases=expected,
                why=dict(rule='one observation over the chain\'s own build; every alpha-c1 probe case and parent case PASS over '
                              'alpha-next-c1\'s engine; every suite non-pass pinned with its reviewed reason and class (none '
                              'expected: no_failing_tests); negative() refuses any other non-pass, a pinned test that passes or '
                              'changes, and any probe case not PASS or whose trace differs from c1\'s frozen record'),
                classification=(dict(path=str(Path(classify).resolve()), sha256=sha(classify)) if classify else None),
                observation=dict(observed_json=str(Path(observed).resolve()), observed_sha256=sha(observed),
                                 log=str(Path(log).resolve()), log_sha256=sha(log), tests_run=rec['suites'].get('tests'),
                                 not_passing=len(tests), skips=rec['suites'].get('skips'),
                                 overlay_manifest=rec['overlay_manifest'], bundle_manifest=rec['manifest'],
                                 container=rec['container'], reasons=rec['reasons']))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--observed', type=Path, required=True)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--classify', type=Path)
    p.add_argument('--discovery', required=True)
    p.add_argument('--check', action='store_true')
    a = p.parse_args(argv)
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
    return 0


if __name__ == '__main__':
    sys.exit(main())
