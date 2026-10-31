"""Native35 window: the unchanged native34 weekend bounds plus the lifetime-arm budget.

R and O arms keep native34_gate.window_ok (2100 s active + 420 s cleanup). The
L arm (native lifetime, hygiene and leak gates) needs a longer active budget;
its slot still ends before the same HARD_END with the same cleanup margin.
"""
from native30_gate import ACTIVE_AND_CLEANUP,SNAPSHOT_CAP,host_receipt
from native34_gate import NOT_BEFORE,HARD_END,window_ok

LIFETIME_ACTIVE_S=7200
CLEANUP_MARGIN_S=420
LIFETIME_ACTIVE_AND_CLEANUP=LIFETIME_ACTIVE_S+CLEANUP_MARGIN_S


def lifetime_window_ok(now):
    return NOT_BEFORE<=now and now+LIFETIME_ACTIVE_AND_CLEANUP<=HARD_END
