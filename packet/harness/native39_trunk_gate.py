"""Native39 trunk window: native35p_gate's weekend bounds and P budget plus the trunk's added phases.

NOT_BEFORE and HARD_END stay native34_gate's (via native35p_gate), unchanged.
The trunk queue is the P-c queue plus a plain W1 control, an idle wait, the W2
length guard and the bind receipt; its declared active budget adds their caps
to P's derivation (native35p_gate.ACTIVE_DERIVATION, unchanged).  The restart
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
)
ACTIVE_DERIVATION = P_ACTIVE_DERIVATION + TRUNK_DERIVATION
TRUNK_ACTIVE_S = 16200
TRUNK_CLEANUP_MARGIN_S = SPEED_CLEANUP_MARGIN_S
TRUNK_ACTIVE_AND_CLEANUP = TRUNK_ACTIVE_S + TRUNK_CLEANUP_MARGIN_S
assert sum(seconds for _, seconds, _ in ACTIVE_DERIVATION) <= TRUNK_ACTIVE_S
assert sum(seconds for _, seconds, _ in CLEANUP_DERIVATION) <= TRUNK_CLEANUP_MARGIN_S
__all__ = ('ACTIVE_DERIVATION', 'CLEANUP_DERIVATION', 'HARD_END', 'NOT_BEFORE', 'SNAPSHOT_CAP', 'TRUNK_ACTIVE_S',
           'TRUNK_ACTIVE_AND_CLEANUP', 'TRUNK_CLEANUP_MARGIN_S', 'TRUNK_DERIVATION', 'host_receipt', 'trunk_window_ok')


def trunk_window_ok(now):
    return NOT_BEFORE <= now and now + TRUNK_ACTIVE_AND_CLEANUP <= HARD_END
