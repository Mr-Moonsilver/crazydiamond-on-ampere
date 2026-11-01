"""Native39 trunk window: native35p_gate's weekend bounds and P budget plus the trunk's added phases.

NOT_BEFORE and HARD_END stay native34_gate's (via native35p_gate), unchanged.
The trunk queue is the P-c queue plus a plain W1 control, an idle wait, the W2
length guard and the bind receipt; its declared active budget adds their caps
to P's derivation (native35p_gate.ACTIVE_DERIVATION, unchanged).  The native39
blocker phases (instrument receipts, typed FULL + idle, route check, access-log
ingress, per-round idle, return reader) add theirs, and row4's 900 s is
released because row4 is declared NOT_RUN on the observer-off speed engine.  The restart
health loop samples every second until the same 540 s wall deadline P's 36 x
(5 s curl + 10 s sleep) allowed, so its row is unchanged.
``trunk_window_ok(time.time())`` is the single executable check used by the
trunk draft's slot and the trunk owner runner.
"""
from native35p_gate import (ACTIVE_DERIVATION as P_ACTIVE_DERIVATION, CLEANUP_DERIVATION, HARD_END, NOT_BEFORE,
                            SNAPSHOT_CAP, SPEED_CLEANUP_MARGIN_S, host_receipt)

TRUNK_DERIVATION = (
    ('bind-receipt', 30, 'cap: --wait-s 30 for the followed log line'),
    ('w2-length-guard', 240, 'cap: 2 x 120 s /tokenize timeout (fresh and append prompt texts)'),
    ('w1-plain-control', 1530, 'cap: client timeout 1500 s + 30 s kill-after (q18b-w1ab2 box values; 8 x 153,537 '
                               'cold opens took 21.5-172.5 s there)'),
    ('w1-plain-idle', 120, 'cap: --timeout-s 120 for running + waiting to reach 0'),
    ('trunk-summary', 5, 'pure CPU over retained artifacts'),
    # Native39 blockers (Codex 1c334bc review, 1 Nov 2026).
    ('row4-declared-not-run', -900, 'P row4-placement 900 s released: row4 is declared NOT_RUN (observer and offers off)'),
    ('instruments-x4', 420, 'cap: 4 x (pause + resume at --control-timeout-s 30, 4 x docker exec at --exec-timeout-s 10, '
                            '5 s start)'),
    ('w1-plain-full-idle', 420, 'cap: prepare, 2 status, pause, resume at --control-timeout-s 60 plus --timeout-s 120 idle'),
    ('w1-plain-route-check', 60, 'cap: one status at --control-timeout-s 60'),
    ('ingress-x4', 240, 'budget: 4 x 60 s, one pass over the followed log each (no inner timeout; the outer one bounds it)'),
    ('w2-idle-x2', 240, 'cap: 2 x --timeout-s 120 for running + waiting to reach 0 before each round'),
    ('returns', 5, 'pure CPU over retained artifacts'),
    ('w1-drain-waits', 180, 'cap: 6 repeats x e3_native39_w1.DRAIN_TIMEOUT_S 30 s for the previous cohort to leave'),
)
ACTIVE_DERIVATION = P_ACTIVE_DERIVATION + TRUNK_DERIVATION
# The declared row4 NOT_RUN (lead option for the observer-off speed engine), written by the queue's
# PY_GATES and read by e3_native39_round; one text in one place.
ROW4_NOT_RUN_REASON = ('speed engine runs observer and offers off; row4 placement needs both '
                       '(lead option: dropped from this run)')
TRUNK_ACTIVE_S = 16200
TRUNK_CLEANUP_MARGIN_S = SPEED_CLEANUP_MARGIN_S
TRUNK_ACTIVE_AND_CLEANUP = TRUNK_ACTIVE_S + TRUNK_CLEANUP_MARGIN_S
assert sum(seconds for _, seconds, _ in ACTIVE_DERIVATION) <= TRUNK_ACTIVE_S
assert sum(seconds for _, seconds, _ in CLEANUP_DERIVATION) <= TRUNK_CLEANUP_MARGIN_S
__all__ = ('ACTIVE_DERIVATION', 'CLEANUP_DERIVATION', 'HARD_END', 'NOT_BEFORE', 'ROW4_NOT_RUN_REASON', 'SNAPSHOT_CAP',
           'TRUNK_ACTIVE_S',
           'TRUNK_ACTIVE_AND_CLEANUP', 'TRUNK_CLEANUP_MARGIN_S', 'TRUNK_DERIVATION', 'host_receipt', 'trunk_window_ok')


def trunk_window_ok(now):
    return NOT_BEFORE <= now and now + TRUNK_ACTIVE_AND_CLEANUP <= HARD_END
