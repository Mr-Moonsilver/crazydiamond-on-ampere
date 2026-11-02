#!/usr/bin/env python3
"""alpha-next negative control: a bounded wall for its container, and a judge that fails closed (lead, 2 Nov ~23:35 CET).

Why: g5's negative observation hung (a candidate witness waited forever over Step B's overlay; fixed in 780b09ef) and nothing
bounded it; after the container was stopped, the self-run still wrote verdict OBSERVED (container rc 137, suites {}).

  * run_bounded(): runs the container client with a relative wall limit (no clock window). On expiry it calls on_timeout (the
    negative stops the container: docker stop -t 30), waits a grace period, then kills the client. The result says TIMEOUT
    with the rc, the limit and the wall.
  * inc_reasons(): OBSERVED (or PASS/FAIL against pins) is allowed only when the container rc is one the negative expects
    (inner.sh records every step's rc in a file and exits 0), every listed negative suite ran and reported (an outcome under
    the suite, its load error or its _FailedTest), and skips are 0 (None counts as not reported). Anything else (killed, timed
    out, crashed, empty) is INC with its reasons. A TIMEOUT is never an expected FAIL.
  * verdict(): INC first, then OBSERVED / PASS / FAIL.
Pure functions plus one subprocess; no GPU, no box action.
"""
import re
import subprocess
import time
from pathlib import Path

EXPECTED_CONTAINER_RC = (0,)
WALL_LIMIT_S = 2700      # the negative's container; ~3.3x the largest container wall of g4/g5 (progress, 820 s at load ~40)
STOP_GRACE_S = 90        # after on_timeout: docker stop -t 30 plus the client's exit
TIMEOUT = 'TIMEOUT'
INC = 'INC'


def run_bounded(cmd, stdout, limit_s, on_timeout, grace_s=STOP_GRACE_S):
    """subprocess.call(cmd) with a wall limit; returns dict(rc, timed_out, limit_s, wall_s, outcome)."""
    start = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=stdout, stderr=subprocess.STDOUT)
    timed_out = False
    try:
        rc = proc.wait(timeout=limit_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            on_timeout()
        finally:
            try:
                rc = proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
    return dict(rc=rc, timed_out=timed_out, limit_s=limit_s, wall_s=round(time.monotonic() - start, 3),
                outcome=TIMEOUT if timed_out else 'exited')


def suite_reported(outcomes, name):
    return any(k == name or k.startswith(name + '.') or k == 'unittest.loader._FailedTest.' + name for k in outcomes)


def inc_reasons(container_rc, result, suite_names, guard=None, expected_rc=EXPECTED_CONTAINER_RC):
    """Why this negative container cannot be judged (empty: it can)."""
    reasons = []
    if guard and guard.get('timed_out'):
        reasons.append('%s: wall limit %ss exceeded (wall %ss), rc %s' % (TIMEOUT, guard.get('limit_s'), guard.get('wall_s'),
                                                                         guard.get('rc')))
    if container_rc not in expected_rc:
        reasons.append('container-rc-unexpected:%s (expected %s)' % (container_rc, ','.join(map(str, expected_rc))))
    result = result or {}
    outcomes = result.get('outcomes')
    if not isinstance(outcomes, dict) or not outcomes:
        reasons.append('suites-not-reported')
        outcomes = {}
    missing = [n for n in suite_names if not suite_reported(outcomes, n)]
    reasons += ['suite-not-reported:' + n for n in missing]
    if result.get('skips') != 0:
        reasons.append('skips:%r' % (result.get('skips'),))
    return reasons


def verdict(inc, observe, ok):
    if inc:
        return INC
    return 'OBSERVED' if observe else 'PASS' if ok else 'FAIL'


def in_flight(out):
    """For the record of a stopped container: the last test line of suites.log and the probe cases with no rc file."""
    out = Path(out)
    log = out / 'suites.log'
    tests = re.findall(r'^(\S+ \([^)]*\)) \.\.\. ', log.read_text(errors='replace'), re.M) if log.is_file() else []
    script = out / 'inner.sh'
    rcs = re.findall(r'^echo \$\? > /out/(\S+)\.rc$', script.read_text(), re.M) if script.is_file() else []
    return dict(last_test=tests[-1] if tests else None, steps_without_rc=[s for s in rcs if not (out / (s + '.rc')).exists()])
