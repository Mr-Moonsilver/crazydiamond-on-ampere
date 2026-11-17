#!/usr/bin/env python3
"""Native43 l-f row1 qualification: e3_q18a_qualification with the policy-aware row1 grade.

``main`` is ``e3_q18a_qualification.main`` with the same argv and record, plus one leading option,
``--qsa-policy {current,dense}`` (required; the queue passes the policy its preflight asserted in EXTRA_ENV).
For the duration of the call the qualification module's ``_grade_row1`` name is bound to the same function
over native43_row1_grade.grade with that policy, and restored afterwards.  The row1 phase, its client, the
worker records and every other base check are unchanged; only the route coverage rule is the policy's
(native43_row1_grade).  Any other phase is refused: this wrapper is for row1 only.
"""
import argparse
import sys

import native43_row1_grade


def policy_grade_row1(policy):
    def _grade_row1(g, evidence, *, run_id, engine_epoch):
        # The worker's flat records are the only source (as e3_q18a_qualification._grade_row1).
        result = native43_row1_grade.grade(evidence, run_id=run_id, engine_epoch=engine_epoch, policy=policy)
        g.checks.update({'row1:' + key: value for key, value in result['checks'].items()})
        g.checks['row1:policy-' + policy] = True
        for value in result['failures']:
            g.fail('row1:' + value)
        for value in result['missing']:
            g.absent('row1:' + value)
    return _grade_row1


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--qsa-policy', required=True, choices=native43_row1_grade.POLICIES)
    known, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    phase = [rest[i + 1] for i, word in enumerate(rest[:-1]) if word == '--phase']
    if phase != ['row1']:
        raise SystemExit('native43_row1_qualification runs the row1 phase only')
    import e3_q18a_qualification as qualification
    original = qualification._grade_row1
    qualification._grade_row1 = policy_grade_row1(known.qsa_policy)
    print('native43 row1 qualification: route coverage for policy %s' % known.qsa_policy, file=sys.stderr)
    try:
        return qualification.main(rest)
    finally:
        qualification._grade_row1 = original


if __name__ == '__main__':
    sys.exit(main())
