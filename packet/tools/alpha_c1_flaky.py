#!/usr/bin/env python3
"""Is a failed C1 gate step explained by the declared flaky tests alone? (the lead, 2 Nov: one unchanged rerun of the whole step)

The list (FLAKY_TESTS, FLAKY_CLASSES) is the lead's; nothing else qualifies. A step qualifies only when every failure is parsed
from a unittest FAIL:/ERROR: line and names a listed test or a test of a listed class. Anything else is not flaky: no parsed
failure (a crash, a kill, a timeout), a failing check that is not a unittest line, a failed selftest suite with no parsed line, or a
failed engine self-run whose cases, clock split or dense clock-bound list are not all PASS.

  strict LOG     harness/selftest.py output: a suite's lines name __main__; the "selftest suite: {...}" line after them names the
                 file. Every top-level "FAIL <check>" line must follow a failed suite whose failures are all flaky.
  unittest LOG   unittest output with module-qualified ids (the focused step through clock_bound.py).
  engine DIR     an engine self-run out dir: result.json, and every suites.log under DIR.
Prints one JSON line; exit 0 = flaky only (rerun once), 1 = not flaky (stop), 2 = usage.
"""
import json
from pathlib import Path
import re
import sys

FLAKY_TESTS = ('test_e3_probe_session.DriverCompositionTests.test_real_live_driver_uses_session_controller_and_validator',
               'test_native43_le_bounded_enqueue.GeneratedSitesTests.test_the_owned_route_fails_loud_and_poisons_the_transport')
FLAKY_CLASSES = ('test_native38_plain_bind.FatalReplies', 'test_e3_native_reply_hol.DirectReplies')
LINE = re.compile(r'^(FAIL|ERROR): (\S+) \(([\w.]+)\)')
SUITE = re.compile(r'^selftest suite: (\{.*\})\s*$')


def flaky(test_id):
    return test_id in FLAKY_TESTS or any(test_id == c or test_id.startswith(c + '.') for c in FLAKY_CLASSES)


def failures(lines):
    """(kind, id) per FAIL:/ERROR: line; a setUpClass-style row (name not the id's last part) keeps the class id."""
    return [(m.group(1), m.group(3)) for m in map(LINE.match, lines) if m]


def strict(text):
    problems, found, pending, last_suite = [], [], [], None
    for line in text.splitlines():
        row = LINE.match(line)
        if row:
            pending.append(row.group(3))
            continue
        suite = SUITE.match(line)
        if suite:
            try:
                info = json.loads(suite.group(1))
            except ValueError:
                problems.append('unparseable-suite-line')
                continue
            module = str(info.get('suite', '')).removesuffix('.py')
            ids = [module + t[len('__main__'):] if t.startswith('__main__.') else t for t in pending]
            last_suite = dict(rc=info.get('returncode'), ids=ids, module=module)
            found += ids
            pending = []
            continue
        if line.startswith('FAIL ') and not line.startswith('FAIL:'):
            if last_suite is None or last_suite['rc'] in (0, None):
                problems.append('failed-check-not-a-suite:' + line[5:80])
            elif not last_suite['ids']:
                problems.append('failed-suite-without-parsed-failure:' + last_suite['module'])
            last_suite = None
        elif line.startswith('ok '):
            last_suite = None
    if pending:
        problems.append('failure-lines-without-suite')
    return found, problems


def unittest_log(text):
    return [i for _, i in failures(text.splitlines())], []


def engine(out):
    out = Path(out)
    problems, ids = [], []
    try:
        record = json.loads((out / 'result.json').read_text())
    except (OSError, ValueError):
        return [], ['no-engine-record']
    cases = record.get('cases') or {}
    if not cases or any((c or {}).get('verdict') != 'PASS' for c in cases.values()):
        problems.append('engine-case-not-pass')
    if (record.get('clock_split') or {}).get('verdict') != 'PASS':
        problems.append('clock-split-not-pass')
    if (record.get('dense_clock_bound') or {}).get('ok') is False:
        problems.append('dense-clock-bound-list-differs')
    for log in sorted(out.rglob('suites.log')):
        ids += [i for _, i in failures(log.read_text(errors='replace').splitlines())]
    return ids, problems


def main(argv):
    if len(argv) != 3 or argv[1] not in ('strict', 'unittest', 'engine'):
        print(__doc__, file=sys.stderr)
        return 2
    mode, target = argv[1], argv[2]
    if mode == 'engine':
        ids, problems = engine(target)
    else:
        text = Path(target).read_text(errors='replace')
        ids, problems = strict(text) if mode == 'strict' else unittest_log(text)
    if not ids:
        problems.append('no-parsed-failure')
    others = sorted(set(i for i in ids if not flaky(i)))
    if others:
        problems.append('not-listed:' + ','.join(others))
    ok = not problems
    print(json.dumps(dict(mode=mode, target=str(target), flaky_only=ok, failures=sorted(set(ids)), problems=problems),
                     sort_keys=True))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
