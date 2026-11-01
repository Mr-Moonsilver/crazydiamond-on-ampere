"""One source for pair/arm wall budgets and the owner's weekend window."""
from native34_gate import NOT_BEFORE, HARD_END

ARM_PHASES = dict(preflight=120, startup=900, identity_start=360, length_guard=240,
                  boundary=380, identity_end=480, stop_and_restore=300)
ARM_ACTIVE_S = sum(ARM_PHASES.values())
ARM_KILL_MARGIN_S = 600
PAIR_ACTIVE_S = 2 * (ARM_ACTIVE_S + ARM_KILL_MARGIN_S)
PAIR_KILL_MARGIN_S = 600


def arm_window_ok(now):
    return NOT_BEFORE <= now and now + ARM_ACTIVE_S + ARM_KILL_MARGIN_S <= HARD_END


def window_ok(now):
    return NOT_BEFORE <= now and now + PAIR_ACTIVE_S + PAIR_KILL_MARGIN_S <= HARD_END
