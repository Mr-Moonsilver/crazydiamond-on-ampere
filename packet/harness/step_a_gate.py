"""Step A box queue: one phase budget for the watchdog, the owner window and the draft.

alpha_perf_gate's rule over Step A's twelve required arms: startup, bind and the unarmed check are held-f's (native40's
phase caps, unchanged). Then eight band 1 arms and four band 2 arms, each behind a typed all-rank FULL entry, with
instruments and ingress receipts, a typed FULL at the end and the grade, then native40's end identity and owned cleanup.

ACTIVE_S is the outer watchdog, not the sum of the per-step caps: the box budget is 9,000 s including cleanup. The draft
guards each arm entry with LAST_ARM_ENTRY_S: an arm starts only when that arm's own worst case (entry, arm, ingress) and
the whole tail (instruments, typed FULL, grade, report, end identity) still fit ACTIVE_S. An arm past that point is
NOT_RUN and INC. The hard end is inherited from native40_box_gate and not moved here.
``window_ok(time.time())`` is the single entry check used by the draft and the owner runner.
"""
from native41_box_gate import (ARM_KILL_AFTER_S, ARM_TIMEOUT_S, FULL_IDLE_CAP_S, INGRESS_CAP_S,
                               INSTRUMENTS_CAP_S, KEPT)
from native40_box_gate import CLEANUP_S, HARD_END, NOT_BEFORE, PHASES as NATIVE40_PHASES

BAND1_ARMS = 8
BAND2_ARMS = 4
ARMS = BAND1_ARMS + BAND2_ARMS
GRADE_CAP_S = 600
PHASES = {name: NATIVE40_PHASES[name] for name in KEPT}
PHASES.update({
    'instruments': 2 * INSTRUMENTS_CAP_S,                 # start and end
    'arm_entries': ARMS * FULL_IDLE_CAP_S,                # a typed all-rank FULL before every arm
    'step_a_arms': ARMS * (ARM_TIMEOUT_S + ARM_KILL_AFTER_S),
    'ingress': ARMS * INGRESS_CAP_S,
    'end_full_idle': FULL_IDLE_CAP_S,
    'grade': GRADE_CAP_S,
})
WORST_CASE_SUM_S = sum(PHASES.values())
BOX_BUDGET_S = 9000                                       # lead: at most 9,000 s, cleanup included
ACTIVE_S = BOX_BUDGET_S - CLEANUP_S
ACTIVE_AND_CLEANUP = ACTIVE_S + CLEANUP_S
ARM_RESERVE_S = FULL_IDLE_CAP_S + ARM_TIMEOUT_S + ARM_KILL_AFTER_S + INGRESS_CAP_S
TAIL_RESERVE_S = (INSTRUMENTS_CAP_S + FULL_IDLE_CAP_S + GRADE_CAP_S + PHASES['report'] + PHASES['end_identity'])
LAST_ARM_ENTRY_S = ACTIVE_S - ARM_RESERVE_S - TAIL_RESERVE_S
assert LAST_ARM_ENTRY_S > 0
EXPECTED_RUNTIME_S = {
    'nominal': 4800,
    'worst_case_budget': BOX_BUDGET_S,
    'per_arm_s': {'band 1 stepped-solo-0 (cold cohort)': 245, 'band 1 fused-solo': 75, 'band 1 stepped-solo-1': 80,
                  'band 1 fused-live': 145, 'band 1 stepped-live': 220,
                  'band 2 solo': 150, 'band 2 live': 420},
    'basis': 'the lead planned about 80 min nominal. Band 1 per-arm wall is held-f run 2 (burst stepped as its burst arms; '
             'the stepped arms here are burst, so the 220 s and 245 s are the slow side). Band 2 is alpha-perf\'s estimate, '
             'never measured on the box at this writing, and fused speed is unmeasured on the box: this run is the first. '
             'Sum: about 35 min for startup, bind, the unarmed check and band 1, about 19 min for band 2, entries and tail; '
             'about 60 to 80 min. The 9,000 s budget holds the worst case.'}


def latest_arm():
    return HARD_END - ACTIVE_S - CLEANUP_S


def window_ok(now):
    return NOT_BEFORE <= now and now + ACTIVE_S + CLEANUP_S <= HARD_END


def arm_fits(elapsed_s):
    """The draft's arm entry rule (it compares bash SECONDS with LAST_ARM_ENTRY_S), for the tests."""
    return elapsed_s <= LAST_ARM_ENTRY_S
