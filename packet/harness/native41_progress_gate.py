"""Held-f native progress box queue: one phase budget for the watchdog, the owner window and the draft.

The queue is held-e's (native40's startup, bind and unarmed check with native40's
phase caps, unchanged), then eight auto-only arms (current | burst) x (live | solo)
x 2 in ABBA order, each behind a typed all-rank FULL entry, with instruments and
ingress receipts, a typed FULL at the end and the progress grade, then native40's
end identity and owned cleanup. Each cap below is the timeout the draft wraps
that step in. ACTIVE_S is their sum; CLEANUP_S is native40's.
``window_ok(time.time())`` is the single entry check used by the draft and the
owner runner: the last arm is HARD_END minus active plus cleanup.
"""
from native41_box_gate import (ARM_KILL_AFTER_S, ARM_TIMEOUT_S, FULL_IDLE_CAP_S, INGRESS_CAP_S,
                               INSTRUMENTS_CAP_S, KEPT)
from native40_box_gate import CLEANUP_S, HARD_END, NOT_BEFORE, PHASES as NATIVE40_PHASES

ARMS = 8
# The grade reads every progress line of the followed engine log (tens of MB).
GRADE_CAP_S = 300
PHASES = {name: NATIVE40_PHASES[name] for name in KEPT}
PHASES.update({
    'instruments': 2 * INSTRUMENTS_CAP_S,                 # start and end
    'arm_entries': ARMS * FULL_IDLE_CAP_S,                # a typed all-rank FULL before every arm
    'progress_arms': ARMS * (ARM_TIMEOUT_S + ARM_KILL_AFTER_S),
    'ingress': ARMS * INGRESS_CAP_S,
    'end_full_idle': FULL_IDLE_CAP_S,
    'grade': GRADE_CAP_S,
})
DROPPED = ('row1_plan', 'row1', 'w2_fresh', 'idle_before_w2')  # native40's row1 and fresh-W2 regression
assert set(PHASES) | set(DROPPED) == set(NATIVE40_PHASES) | {
    'arm_entries', 'progress_arms', 'ingress', 'end_full_idle', 'grade'}
ACTIVE_S = sum(PHASES.values())
EXPECTED_RUNTIME_S = {'nominal': 4500, 'basis': 'native40 startup through the unarmed check ~15-20 min; eight arms '
                                                 '~4-6 min each (8 x 153,537 open, cold only in the first arm; 25 s '
                                                 'decode-only; native fresh 65,445 tokens ~64 chunks at ~2.8 s live '
                                                 'or ~0.8 s solo; append; close) plus a typed FULL entry each; '
                                                 'receipts, typed FULL, grade, end identity ~6 min'}


def latest_arm():
    return HARD_END - ACTIVE_S - CLEANUP_S


def window_ok(now):
    return NOT_BEFORE <= now and now + ACTIVE_S + CLEANUP_S <= HARD_END
