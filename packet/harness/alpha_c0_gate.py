"""C0 box queue: one phase budget for the watchdog and the draft; no absolute-time window.

alpha_perf_gate's rule over C0's eight required arms: startup, bind and the unarmed check are held-f's (native40's
phase caps, unchanged). Then eight band 1 arms, each behind a typed all-rank FULL entry, with
instruments and ingress receipts, a typed FULL at the end and the grade, then native40's end identity and owned cleanup.

ACTIVE_S is the outer watchdog, not the sum of the per-step caps: the box budget is 9,000 s including cleanup. The draft
guards each arm entry with LAST_ARM_ENTRY_S: an arm starts only when that arm's own worst case (entry, arm, ingress) and
the whole tail (instruments, typed FULL, grade, report, end identity) still fit ACTIVE_S. An arm past that point is
NOT_RUN and INC. CLEANUP_S is native40_box_gate's.

No absolute-time window (operator, 2 Nov: "we don't put in code what you govern"; launch time is the lead's). There is no
NOT_BEFORE, HARD_END, latest start or window_ok here, and neither the draft nor the owner runner refuses a start by wall clock.
The only time limits are relative, counted from the run's own start: the draft's outer timeout (ACTIVE_S, then CLEANUP_S for
owned cleanup) and LAST_ARM_ENTRY_S before each arm (bash SECONDS). native40_box_gate and every other frozen gate are unchanged.
"""
from native41_box_gate import (ARM_KILL_AFTER_S, ARM_TIMEOUT_S, FULL_IDLE_CAP_S, INGRESS_CAP_S,
                               INSTRUMENTS_CAP_S, KEPT)
from native40_box_gate import CLEANUP_S, PHASES as NATIVE40_PHASES

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
BOX_BUDGET_S = 9000                                       # alpha_perf_gate's; eight arms, about 40 min nominal
# 5,400 s left LAST_ARM_ENTRY_S at 870 s from queue start, so only arms 1-3 could enter (first arm entry is about
# +480 s on the box); 9,000 s gives 4,470 s, the parent's own rule (lead's go, 2 Nov).
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
             '35 min for startup, bind, the unarmed check and the eight arms, plus the tail: about 40 min. On the box the first arm '
             'enters about 480 s after queue start (alpha-perf, 1 Nov). The 9,000 s budget gives LAST_ARM_ENTRY_S 4,470 s, '
             'which holds all eight arms and a slower local path; an arm past it is NOT_RUN and INC.'}


def arm_fits(elapsed_s):
    """The draft's arm entry rule (it compares bash SECONDS with LAST_ARM_ENTRY_S), for the tests."""
    return elapsed_s <= LAST_ARM_ENTRY_S
