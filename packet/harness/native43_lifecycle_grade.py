"""Native43 lifecycle reading: native lifetime (cancel/drain/reuse) and the typed FULL return, short scope.

The short-scope grade over the q18a-native43-lifecycle queue's retained records.
It applies native35_lifetime_grade's lifecycle, typed-close, reuse, hygiene and
engine-crash checks unchanged (the same functions, called on the same records) and
does not grade the leak or pressure oracles: the short queue does not run them, so
they are reported NOT_RUN and are never counted as PASS.  A leak record present in
a short-scope queue directory is a FAIL (the queue was not the declared one).
No leak, pressure, performance, placement or numerical claim.
"""
import argparse
import json
from pathlib import Path

import native35_lifetime_grade as l

NOT_RUN = ('cross-session-leak', 'partial-leak', 'pressure-completed', 'pressure-lifetime')
FOREIGN = ('cross-session-leak.log', 'partial-leak.grade.json', 'pressure.completion.json', 'pressure.coverage.json')


def grade(queue):
    full = l.grade(queue)
    q = Path(queue)
    prefixes = tuple(name + ':' for name in NOT_RUN)
    failures = [f for f in full['failures'] if not f.startswith(prefixes)]
    missing = [m for m in full['missing'] if not m.startswith(prefixes)]
    foreign = [name for name in FOREIGN if (q / name).exists()]
    failures += ['short-scope-foreign-record:' + name for name in foreign]
    records = {k: v for k, v in full['records'].items()
               if k in ('lifecycle', 'hygiene_full')}
    return dict(schema='native43-lifecycle-reading/1', verdict='FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS',
                failures=failures, missing=missing, records=records, not_run=list(NOT_RUN), release_qualified=False,
                scope='native cancel/drain/reuse on the native43 dense engine plus the typed FULL return; the leak and '
                      'pressure oracles are not run (short scope); no leak, pressure, performance, placement or '
                      'numerical claim')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--queue', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    result = grade(a.queue)
    with Path(a.out).open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2)
        f.write('\n')
    print('NATIVE43_LIFECYCLE ' + result['verdict'])
    return {'PASS': 0, 'FAIL': 1}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
