"""Native43 lifecycle window: the unchanged weekend bounds and the short lifecycle queue's own derived budget.

NOT_BEFORE and HARD_END stay native34_gate's (via native39_lifecycle_gate), unchanged.
The native43 lifecycle queue is the box-run native39 lifecycle queue (held-a run 1)
in short scope: startup, bind receipt, W2 length guard, row1 (lane proof), the
lifetime gates, the two instruments receipts, the native lifetime (cancel/drain/reuse)
and the typed FULL hygiene return, then the unchanged ending.  The leaktest, the
pressure2/partial-tail leak, pressure completion and pressure lifetime rows and the
pool-telemetry record are not run, so their rows are dropped; every kept row is taken
by name from native39_lifecycle_gate with its value unchanged.  The cleanup derivation
is native39_lifecycle_gate's, whose last row (the lifetime grade's engine-stream crash
scan) is the native43 lifecycle grade's identical scan.
``lifecycle_window_ok(time.time())`` is the single executable check used by the
lifecycle draft's slot and the lifecycle owner runner.
"""
import datetime

import native39_lifecycle_gate as parent
from native39_lifecycle_gate import HARD_END, NOT_BEFORE, SNAPSHOT_CAP, host_receipt


def _rows(source, names):
    found = {row[0]: row for row in source}
    return tuple(found[name] for name in names)


KEPT = ('lifetime-gates', 'instruments-x2', 'native-lifetime', 'hygiene-full')
DROPPED = ('pool-telemetry', 'cross-session-leak', 'pressure-before-log', 'pressure-ready', 'partial-leak',
           'pressure-completed', 'pressure-stop-and-after-log', 'pressure-lifetime-grade')
assert set(KEPT) | set(DROPPED) == {row[0] for row in parent.LIFECYCLE_DERIVATION} and not set(KEPT) & set(DROPPED)
STARTUP_DERIVATION = parent.STARTUP_DERIVATION
LIFECYCLE_DERIVATION = _rows(parent.LIFECYCLE_DERIVATION, KEPT)
CLEANUP_DERIVATION = parent.CLEANUP_DERIVATION
ENDING_DERIVATION = parent.ENDING_DERIVATION
ACTIVE_DERIVATION = STARTUP_DERIVATION + LIFECYCLE_DERIVATION + ENDING_DERIVATION
LIFECYCLE_ACTIVE_S = 5000
LIFECYCLE_CLEANUP_MARGIN_S = parent.LIFECYCLE_CLEANUP_MARGIN_S
LIFECYCLE_ACTIVE_AND_CLEANUP = LIFECYCLE_ACTIVE_S + LIFECYCLE_CLEANUP_MARGIN_S
assert sum(seconds for _, seconds, _ in ACTIVE_DERIVATION) <= LIFECYCLE_ACTIVE_S
assert sum(seconds for _, seconds, _ in CLEANUP_DERIVATION) <= LIFECYCLE_CLEANUP_MARGIN_S
# The last instant the queue's slot (and the runner's arm check) accepts: HARD_END minus the active budget and the
# cleanup margin.  2026-11-02T04:21:40Z.
LATEST_ENTRY = HARD_END - LIFECYCLE_ACTIVE_AND_CLEANUP
LATEST_ENTRY_UTC = datetime.datetime.fromtimestamp(LATEST_ENTRY, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
assert LATEST_ENTRY_UTC == '2026-11-02T04:21:40Z'
EXPECTED_RUNTIME_S = {'nominal': 2400, 'basis': 'native39 lifecycle held-a run 1 (03:32-04:16Z, 44 min) without its '
                                                'leaktest, partial-leak and pressure phases (~25 min there): startup ~10 '
                                                'min, row1 ~7 min, lifecycle ~5 min, ending ~10 min'}
__all__ = ('ACTIVE_DERIVATION', 'CLEANUP_DERIVATION', 'DROPPED', 'ENDING_DERIVATION', 'HARD_END', 'KEPT', 'LATEST_ENTRY',
           'LATEST_ENTRY_UTC', 'LIFECYCLE_ACTIVE_AND_CLEANUP', 'LIFECYCLE_ACTIVE_S', 'LIFECYCLE_CLEANUP_MARGIN_S',
           'LIFECYCLE_DERIVATION', 'NOT_BEFORE', 'SNAPSHOT_CAP', 'STARTUP_DERIVATION', 'host_receipt',
           'lifecycle_window_ok')


def lifecycle_window_ok(now):
    return NOT_BEFORE <= now and now + LIFECYCLE_ACTIVE_AND_CLEANUP <= HARD_END
