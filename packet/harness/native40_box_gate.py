"""One phase budget for the targeted queue watchdog and owner window."""
from native39_trunk_gate import NOT_BEFORE, HARD_END
PHASES = {
    'preflight_admin': 120,
    'restart': 120,
    'health': 540,
    'startup_logs_sampler': 60,
    'identity': 360,
    'bootstrap': 120,
    'guards': 120,
    'unarmed': 180,
    'row1_plan': 60,
    'row1': 600,
    'instruments': 240,  # two 120s process/map proofs
    'w2_fresh': 600,     # standing cohort + one native fresh request + cleanup
    'idle_before_w2': 120,
    'report': 60,
    'end_identity': 360,
}
ACTIVE_S = sum(PHASES.values())
CLEANUP_S = 900  # inherited bounded stop/restores; production never started

def window_ok(now):
    return NOT_BEFORE <= now and now + ACTIVE_S + CLEANUP_S <= HARD_END
