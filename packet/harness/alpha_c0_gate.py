"""C0 box queue: one phase budget for the watchdog, the owner window and the draft.

alpha_perf_gate's rule over C0's eight required arms: startup, bind and the unarmed check are held-f's (native40's
phase caps, unchanged). Then eight band 1 arms, each behind a typed all-rank FULL entry, with
instruments and ingress receipts, a typed FULL at the end and the grade, then native40's end identity and owned cleanup.

ACTIVE_S is the outer watchdog, not the sum of the per-step caps: the box budget is 5,400 s including cleanup. The draft
guards each arm entry with LAST_ARM_ENTRY_S: an arm starts only when that arm's own worst case (entry, arm, ingress) and
the whole tail (instruments, typed FULL, grade, report, end identity) still fit ACTIVE_S. An arm past that point is
NOT_RUN and INC. NOT_BEFORE and CLEANUP_S are native40_box_gate's. The hard end is C0's own: 2026-11-02T07:00Z, the operator's
instruction of 2 Nov (production comes back at 08:00 CET); native40_box_gate.HARD_END (06:00Z) and every other frozen gate are
unchanged.
``window_ok(time.time())`` is the single entry check used by the draft and the owner runner.
"""
from native41_box_gate import (ARM_KILL_AFTER_S, ARM_TIMEOUT_S, FULL_IDLE_CAP_S, INGRESS_CAP_S,
                               INSTRUMENTS_CAP_S, KEPT)
from native40_box_gate import CLEANUP_S, HARD_END as NATIVE40_HARD_END, NOT_BEFORE, PHASES as NATIVE40_PHASES

HARD_END = 1793602800.0                                    # 2026-11-02T07:00Z, C0 only (operator, 2 Nov)
assert HARD_END == NATIVE40_HARD_END + 3600

BAND1_ARMS = 8
ARMS = BAND1_ARMS
GRADE_CAP_S = 600
PHASES = {name: NATIVE40_PHASES[name] for name in KEPT}
PHASES.update({
    'instruments': 2 * INSTRUMENTS_CAP_S,                 # start and end
    'arm_entries': ARMS * FULL_IDLE_CAP_S,                # a typed all-rank FULL before every arm
    'alpha_c0_arms': ARMS * (ARM_TIMEOUT_S + ARM_KILL_AFTER_S),
    'ingress': ARMS * INGRESS_CAP_S,
    'end_full_idle': FULL_IDLE_CAP_S,
    'grade': GRADE_CAP_S,
})
WORST_CASE_SUM_S = sum(PHASES.values())
BOX_BUDGET_S = 5400                                       # eight arms, about 40 min nominal; 90 min including cleanup
ACTIVE_S = BOX_BUDGET_S - CLEANUP_S
ACTIVE_AND_CLEANUP = ACTIVE_S + CLEANUP_S
ARM_RESERVE_S = FULL_IDLE_CAP_S + ARM_TIMEOUT_S + ARM_KILL_AFTER_S + INGRESS_CAP_S
TAIL_RESERVE_S = (INSTRUMENTS_CAP_S + FULL_IDLE_CAP_S + GRADE_CAP_S + PHASES['report'] + PHASES['end_identity'])
LAST_ARM_ENTRY_S = ACTIVE_S - ARM_RESERVE_S - TAIL_RESERVE_S
assert LAST_ARM_ENTRY_S > 0
EXPECTED_RUNTIME_S = {
    'nominal': 2400,
    'worst_case_budget': BOX_BUDGET_S,
    'per_arm_s': {'serial-solo-0 (cold cohort)': 245, 'local-solo': 75, 'serial-solo-1': 80, 'local-live': 145, 'serial-live': 145},
    'basis': 'band 1 per-arm wall is held-f run 2 for burst. local speed is unmeasured on the box: this run is the first. About '
             '35 min for startup, bind, the unarmed check and the eight arms, plus the tail: about 40 min. The 5,400 s budget '
             'holds a slower local path, and an arm past LAST_ARM_ENTRY_S is NOT_RUN and INC.'}


def latest_arm():
    return HARD_END - ACTIVE_S - CLEANUP_S


def window_ok(now):
    return NOT_BEFORE <= now and now + ACTIVE_S + CLEANUP_S <= HARD_END


def arm_fits(elapsed_s):
    """The draft's arm entry rule (it compares bash SECONDS with LAST_ARM_ENTRY_S), for the tests."""
    return elapsed_s <= LAST_ARM_ENTRY_S
