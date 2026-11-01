"""Native40 short box queue: 40 min active, existing 15 min owned cleanup.

The trunk sequence is retained. Its individual phase maxima do not fit this
short diagnostic's active allowance: the existing outer TERM watchdog cuts
the active sequence at 2400s. Incomplete W1/W2 can never count as speed PASS.
"""
from native39_trunk_gate import NOT_BEFORE, HARD_END

ACTIVE_S = 2400
CLEANUP_S = 900


def window_ok(now):
    return NOT_BEFORE <= now and now + ACTIVE_S + CLEANUP_S <= HARD_END
