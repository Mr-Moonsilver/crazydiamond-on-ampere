"""Native41 auto-arm box queue: one phase budget for the watchdog, the owner window and the draft.

The queue is native40's startup, bind and unarmed check (native40's phase caps,
unchanged), then four plain W2 arms (stock, auto, stock, auto) with their entries,
instruments and ingress receipts, a typed FULL and the native41 grade, then
native40's end identity and owned cleanup. Each native41 cap below is the
timeout the draft wraps that step in. ACTIVE_S is their sum; CLEANUP_S is
native40's. ``window_ok(time.time())`` is the single entry check used by the
draft and the owner runner: the last arm is HARD_END minus active plus cleanup.
"""
from native40_box_gate import CLEANUP_S, HARD_END, NOT_BEFORE, PHASES as NATIVE40_PHASES

# Lead, 1 Nov 2026: last arm before Mon 06:00Z minus budget (HARD_END is native34's 2026-11-02T06:00Z).
ARMS = 4
ARM_TIMEOUT_S = 1500
ARM_KILL_AFTER_S = 30
IDLE_CAP_S = 150        # idle --timeout-s 120 plus its /metrics reads
FULL_IDLE_CAP_S = 450   # full-idle: prepare, status, pause, resume at --control-timeout-s 60 plus idle 120
INSTRUMENTS_CAP_S = 120  # native40's instruments cap (pause + resume at 30 s, 4 docker exec at 10 s)
INGRESS_CAP_S = 60      # one pass over the followed access log
GRADE_CAP_S = 60
KEPT = ('preflight_admin', 'restart', 'health', 'startup_logs_sampler', 'identity', 'bootstrap', 'guards',
        'unarmed', 'report', 'end_identity')
PHASES = {name: NATIVE40_PHASES[name] for name in KEPT}
PHASES.update({
    'instruments': 2 * INSTRUMENTS_CAP_S,                 # start and end
    'arm_entries': 3 * IDLE_CAP_S + FULL_IDLE_CAP_S,      # stock-0, auto-0, auto-1 idle; stock-1 typed FULL
    'w2_arms': ARMS * (ARM_TIMEOUT_S + ARM_KILL_AFTER_S),
    'ingress': ARMS * INGRESS_CAP_S,
    'end_full_idle': FULL_IDLE_CAP_S,
    'grade': GRADE_CAP_S,
})
DROPPED = ('row1_plan', 'row1', 'w2_fresh', 'idle_before_w2')  # native40's row1 and fresh-W2 regression
assert set(PHASES) | set(DROPPED) == set(NATIVE40_PHASES) | {
    'arm_entries', 'w2_arms', 'ingress', 'end_full_idle', 'grade'}
ACTIVE_S = sum(PHASES.values())
EXPECTED_RUNTIME_S = {'nominal': 3600, 'basis': 'native40 startup through the unarmed check ~15-20 min; four arms '
                                                 '~6-9 min each (8 x 153,537 open, 25 s decode-only, fresh, close, '
                                                 'append); receipts, typed FULL, end identity ~5 min'}


def latest_arm():
    return HARD_END - ACTIVE_S - CLEANUP_S


def window_ok(now):
    return NOT_BEFORE <= now and now + ACTIVE_S + CLEANUP_S <= HARD_END
