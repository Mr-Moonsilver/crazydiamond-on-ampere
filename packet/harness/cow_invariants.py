#!/usr/bin/env python3
"""Check the runner's QSA CoW verify lines (QSA_COW_VERIFY=1) for every invariant, not only destination equality.

Input: the "QSA host-page CoW verify" log lines of one run. PASS needs: lines from all 4 ranks; every line "after 0"
(the copied prefix equal after the copy); in each rank's last invariants dict src_mutated, controls_changed,
src_changed_by_copy and prefix_unknown all 0; src_rechecks > 0 and controls > 0 on every rank (the checks ran).
usage: cow_invariants.py VERIFY.log   prints one line "PASS|FAIL ..."; exit 0 PASS, 1 FAIL
"""
import ast, re, sys

VIOLATIONS = ('src_mutated', 'controls_changed', 'src_changed_by_copy', 'prefix_unknown')


def check(lines):
    bad_after = [l for l in lines if not re.search(r', after 0 \(', l)]
    last = {}
    for l in lines:
        w = re.search(r'Worker_PP\d+_TP\d+', l); inv = re.search(r'invariants (\{.*\})', l)
        if w and inv:
            last[w.group(0)] = ast.literal_eval(inv.group(1))
    bad = {w: {k: v for k, v in d.items() if k in VIOLATIONS and v} for w, d in last.items()}
    cov = {w: (d.get('src_rechecks', 0), d.get('src_reallocated', 0), d.get('controls', 0)) for w, d in last.items()}
    ok = len(last) == 4 and not bad_after and not any(bad.values()) and all(r > 0 and c > 0 for r, _, c in cov.values())
    return ok, (f'lines {len(lines)} ranks {len(last)} unequal-after-lines {len(bad_after)} violations {bad} '
                f'coverage(rechecks,reallocated,controls) {cov}')


if __name__ == '__main__':
    ok, msg = check(open(sys.argv[1]).read().splitlines())
    print('PASS' if ok else 'FAIL', msg)
    sys.exit(0 if ok else 1)
