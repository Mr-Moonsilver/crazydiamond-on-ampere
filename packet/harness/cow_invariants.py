#!/usr/bin/env python3
"""Check the runner's QSA CoW verify lines (QSA_COW_VERIFY=1) for every invariant, not only destination equality.

Input: the "QSA host-page CoW verify" log lines of one run. PASS needs: lines from all 4 ranks; every line "after 0"
(the copied prefix equal after the copy); in each rank's last invariants dict src_mutated, controls_changed,
src_changed_by_copy and prefix_unknown all present, integers and 0; src_rechecks and controls integers > 0 on every
rank (the checks ran). Exactly the ranks in RANKS; a missing or mistyped counter is a FAIL (24 Oct review: a line
without mutation counters was accepted as clean).
usage: cow_invariants.py VERIFY.log   prints one line "PASS|FAIL ..."; exit 0 PASS, 1 FAIL
"""
import ast, re, sys

VIOLATIONS = ('src_mutated', 'controls_changed', 'src_changed_by_copy', 'prefix_unknown')
COVERAGE = ('src_rechecks', 'controls')
RANKS = {'Worker_PP0_TP0', 'Worker_PP0_TP1', 'Worker_PP1_TP0', 'Worker_PP1_TP1'}


def check(lines):
    bad_after = [l for l in lines if not re.search(r', after 0 \(', l)]
    last = {}
    for l in lines:
        w = re.search(r'Worker_PP\d+_TP\d+', l); inv = re.search(r'invariants (\{.*\})', l)
        if w and inv:
            last[w.group(0)] = ast.literal_eval(inv.group(1))
    schema = {w: sorted(k for k in VIOLATIONS + COVERAGE if not isinstance(d.get(k), int) or isinstance(d.get(k), bool))
              for w, d in last.items()}
    schema = {w: v for w, v in schema.items() if v}
    bad = {w: {k: v for k, v in d.items() if k in VIOLATIONS and v} for w, d in last.items()}
    cov = {w: tuple(d.get(k) if isinstance(d.get(k), int) else 0 for k in ('src_rechecks', 'src_reallocated', 'controls'))
           for w, d in last.items()}
    ok = (set(last) == RANKS and not bad_after and not schema and not any(bad.values())
          and all(r > 0 and c > 0 for r, _, c in cov.values()))
    if schema:
        return False, f'missing or non-integer counters {schema}; ranks {sorted(last)}'
    return ok, (f'lines {len(lines)} ranks {len(last)} unequal-after-lines {len(bad_after)} violations {bad} '
                f'coverage(rechecks,reallocated,controls) {cov}')


if __name__ == '__main__':
    ok, msg = check(open(sys.argv[1]).read().splitlines())
    print('PASS' if ok else 'FAIL', msg)
    sys.exit(0 if ok else 1)
