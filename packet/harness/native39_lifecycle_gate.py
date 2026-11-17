"""Native39 lifecycle window: the unchanged weekend bounds and the lifecycle queue's own derived budget.

NOT_BEFORE and HARD_END stay native34_gate's (via native39_trunk_gate), unchanged.
The lifecycle queue is the trunk queue's startup (native28 startup, bind receipt,
W2 length guard), row1 as the lane proof, then the native28/native35-L lifecycle
block (native cancel/drain/reuse, typed FULL hygiene, leaktest 8 x 3, pressure2
2 x 200000 beside the 7568 partial-tail leak at 30 % cancels, pressure completion
and lifetime) between two of the trunk's instruments receipts (observer and offers
off on the actual rank processes, before the block and at its end), after a
pool-telemetry record (the declared FLASH_POOL_STATS=200 delta as the engine runs it),
and the trunk's unchanged ending.  The active budget is derived row
by row: the startup and row1 rows are P's (native35p_gate.ACTIVE_DERIVATION) and
the trunk's (native39_trunk_gate.TRUNK_DERIVATION), taken by name with their values
unchanged; the lifecycle rows are the tools' own caps or a stated 2x estimate (no
lifecycle phase has run on a native engine: native28 and native35-L both stopped
before it).  The cleanup derivation is P's with the native35 lifetime grade added
beside the localize grade; its margin is the trunk's.
``lifecycle_window_ok(time.time())`` is the single executable check used by the
lifecycle draft's slot and the lifecycle owner runner.
"""
import datetime

from native35p_gate import ACTIVE_DERIVATION as P_ACTIVE_DERIVATION, CLEANUP_DERIVATION as P_CLEANUP_DERIVATION
from native39_trunk_gate import (HARD_END, NOT_BEFORE, SNAPSHOT_CAP, TRUNK_CLEANUP_MARGIN_S, TRUNK_DERIVATION,
                                 host_receipt)


def _rows(source, names):
    found = {row[0]: row for row in source}
    return tuple(found[name] for name in names)


STARTUP_DERIVATION = (_rows(P_ACTIVE_DERIVATION, ('restart-to-health', 'speculation-and-sampler', 'identity-start',
                                                  'startup-ledger-headroom-plan'))
                      + _rows(TRUNK_DERIVATION, ('bind-receipt', 'w2-length-guard'))
                      + _rows(P_ACTIVE_DERIVATION, ('row1',)))
LIFECYCLE_DERIVATION = (
    ('pool-telemetry', 30, 'cap: one read of the identity-start capture and the followed engine log so far (startup and '
                           'row1); the declared FLASH_POOL_STATS=200 telemetry delta, record only'),
    ('lifetime-gates', 5, 'pure CPU: gates.json from the startup and row1 verdicts'),
    ('instruments-x2', 210, "cap: 2 x the trunk's instruments row (pause + resume at --control-timeout-s 30, 4 x docker "
                            'exec at --exec-timeout-s 10, 5 s start): before the lifecycle block and at its end'),
    ('native-lifetime', 900, 'caps: in-flight observation read_s 180 + drain_s 120 + reuse first advance read_s 180 '
                             '(q18a-w2-overlap-native39 timeouts) = 480 s, plus 420 s for its 13 control calls '
                             '(P allows 30 s controller per W1 repeat; each call is bounded at 120 s)'),
    ('hygiene-full', 240, 'cap: 2 control calls (prepare full, resume) at the 120 s client timeout'),
    ('cross-session-leak', 780, 'cap: leaktest 3 rounds x (240 s request timeout + GAP 20 s max sleep); 8 sessions in parallel'),
    ('pressure-before-log', 120, 'docker logs of <= 2 GiB retained json log (P capture-logs basis)'),
    ('pressure-ready', 180, 'cap: the queue ready wait, 180 s monotonic'),
    ('partial-leak', 2400, "estimate: 2x the E2 gate plan's CoW/leak ~20 min (docs/E2-CANDIDATE-GATE.md); 10 sessions "
                           'x 4 rounds, GEN 200, 30 % cancels beside 2 x 200k pressure; not measured on a native engine; '
                           'its 900 s per-request TIMEOUT is a ceiling, not a sleep'),
    ('pressure-completed', 900, 'cap: pressure_wait bound 900 s'),
    ('pressure-stop-and-after-log', 150, 'stop_pressure TERM + 30 s kill-after, then docker logs <= 2 GiB (120 s)'),
    ('pressure-lifetime-grade', 30, 'pure CPU over the two retained logs'),
)
# Cleanup margin: owned cleanup after TERM before KILL (on_exit order).  P's rows,
# plus the native35 lifetime grade (L's on_exit line) beside the localize grade.
CLEANUP_DERIVATION = P_CLEANUP_DERIVATION + (
    ('lifetime-grade', 150, 'native35 lifetime grade: engine-stream.log/err crash scan at the localize grade rate, ~2 GiB'),)
ENDING_DERIVATION = (_rows(P_ACTIVE_DERIVATION, ('identity-end',))
                     + (('normal-path-cleanup', sum(s for _, s, _ in CLEANUP_DERIVATION),
                         'same worst case as the timeout cleanup below'),))
ACTIVE_DERIVATION = STARTUP_DERIVATION + LIFECYCLE_DERIVATION + ENDING_DERIVATION
LIFECYCLE_ACTIVE_S = 9600
LIFECYCLE_CLEANUP_MARGIN_S = TRUNK_CLEANUP_MARGIN_S
LIFECYCLE_ACTIVE_AND_CLEANUP = LIFECYCLE_ACTIVE_S + LIFECYCLE_CLEANUP_MARGIN_S
assert sum(seconds for _, seconds, _ in ACTIVE_DERIVATION) <= LIFECYCLE_ACTIVE_S
assert sum(seconds for _, seconds, _ in CLEANUP_DERIVATION) <= LIFECYCLE_CLEANUP_MARGIN_S
# The last instant the queue's slot (and the runner's arm check) accepts: HARD_END minus the active budget and the
# cleanup margin.  2026-11-02T03:05:00Z.
LATEST_ENTRY = HARD_END - LIFECYCLE_ACTIVE_AND_CLEANUP
LATEST_ENTRY_UTC = datetime.datetime.fromtimestamp(LATEST_ENTRY, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
assert LATEST_ENTRY_UTC == '2026-11-02T03:05:00Z'
EXPECTED_RUNTIME_S = {'nominal': 4500, 'basis': 'startup ~10 min and row1 ~7 min as on P-c; lifecycle ~5 min, leaktest '
                                                '~5-10 min, partial-leak ~20 min, pressure wait ~2 min, ending ~10 min'}
__all__ = ('ACTIVE_DERIVATION', 'CLEANUP_DERIVATION', 'ENDING_DERIVATION', 'HARD_END', 'LATEST_ENTRY', 'LATEST_ENTRY_UTC',
           'LIFECYCLE_ACTIVE_AND_CLEANUP', 'LIFECYCLE_ACTIVE_S', 'LIFECYCLE_CLEANUP_MARGIN_S', 'LIFECYCLE_DERIVATION',
           'NOT_BEFORE', 'SNAPSHOT_CAP', 'STARTUP_DERIVATION', 'host_receipt', 'lifecycle_window_ok')


def lifecycle_window_ok(now):
    return NOT_BEFORE <= now and now + LIFECYCLE_ACTIVE_AND_CLEANUP <= HARD_END
