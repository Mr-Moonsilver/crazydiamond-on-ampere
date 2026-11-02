"""Step B box queue: one phase budget for the watchdog, the owner window and the draft.

step_a_gate's rule over Step B's sixteen required arms: startup, bind and the unarmed check are held-f's. Then twelve band 1
arms (sizes 1, 2, 4) and four band 2 arms (sizes 1, 2), each behind a typed all-rank FULL entry, with instruments and ingress
receipts, a typed FULL at the end and the grade, then native40's end identity and owned cleanup.

ACTIVE_S is the outer watchdog: the box budget is 9,000 s including cleanup. An arm starts only inside LAST_ARM_ENTRY_S, Step A's
number, since each arm's worst case and the tail are unchanged. TRIM_G4_REPEAT_S is the lead's order of what goes first: a size 4
repeat enters only while band 2's nominal time still fits, so a slow run gives up those two arms before it gives up a band 2
arm. A trimmed arm is NOT_RUN and the run is INC, never PASS. HARD_END is Step B's own: 2026-11-02T07:00Z, the operator's move of
2 Nov (production comes back at 08:00 CET); native40_box_gate's 06:00Z and every other gate stay as they are. NOT_BEFORE and CLEANUP_S
stay native40's.
``window_ok(time.time())`` is the single entry check used by the draft and the owner runner.
"""
from native41_box_gate import ARM_KILL_AFTER_S, ARM_TIMEOUT_S, FULL_IDLE_CAP_S, INGRESS_CAP_S, INSTRUMENTS_CAP_S, KEPT
import datetime

from native40_box_gate import CLEANUP_S, NOT_BEFORE, PHASES as NATIVE40_PHASES
from step_a_gate import ACTIVE_S, BOX_BUDGET_S, GRADE_CAP_S, LAST_ARM_ENTRY_S

HARD_END = datetime.datetime(2026, 11, 2, 7, tzinfo=datetime.timezone.utc).timestamp()   # operator, 2 Nov: production at 08:00 CET
BAND1_ARMS = 12
BAND2_ARMS = 4
ARMS = BAND1_ARMS + BAND2_ARMS
PHASES = {name: NATIVE40_PHASES[name] for name in KEPT}
PHASES.update({
    'instruments': 2 * INSTRUMENTS_CAP_S,                 # start and end
    'arm_entries': ARMS * FULL_IDLE_CAP_S,                # a typed all-rank FULL before every arm
    'step_b_arms': ARMS * (ARM_TIMEOUT_S + ARM_KILL_AFTER_S),
    'ingress': ARMS * INGRESS_CAP_S,
    'end_full_idle': FULL_IDLE_CAP_S,
    'grade': GRADE_CAP_S,
})
WORST_CASE_SUM_S = sum(PHASES.values())
ACTIVE_AND_CLEANUP = ACTIVE_S + CLEANUP_S
assert ACTIVE_AND_CLEANUP == BOX_BUDGET_S
ARM_RESERVE_S = FULL_IDLE_CAP_S + ARM_TIMEOUT_S + ARM_KILL_AFTER_S + INGRESS_CAP_S
TAIL_RESERVE_S = (INSTRUMENTS_CAP_S + FULL_IDLE_CAP_S + GRADE_CAP_S + PHASES['report'] + PHASES['end_identity'])
assert LAST_ARM_ENTRY_S == ACTIVE_S - ARM_RESERVE_S - TAIL_RESERVE_S
BAND2_NOMINAL_S = 2 * 420 + 2 * 150 + 4 * 30              # two live, two solo, four entries; alpha-perf's estimate
BAND1_REST_S = 2 * 145 + 2 * 75                           # what can still follow the first size 4 repeat in band 1
TRIM_G4_REPEAT_S = LAST_ARM_ENTRY_S - BAND2_NOMINAL_S - BAND1_REST_S
assert TRIM_G4_REPEAT_S > 0
EXPECTED_RUNTIME_S = {
    'nominal': 4800,
    'worst_case_budget': BOX_BUDGET_S,
    'per_arm_s': {'band 1 solo, first (cold cohort)': 245, 'band 1 solo': 75, 'band 1 live': 145, 'band 2 solo': 150, 'band 2 live': 420},
    'basis': 'Band 1 per-arm wall is held-f run 2 as Step A planned it, on the slow side of fused. Band 2 is alpha-perf\'s estimate, '
             'never measured on the box at this writing, and neither group speed nor fused speed is measured on the box: this is '
             'an estimate. Sum: about 15 min for startup, bind and the unarmed check, 25 min for twelve band 1 arms, 19 min for '
             'band 2, 5 min for entries and tail: about 65 min. The lead planned 80 to 90 min. The 9,000 s budget holds the worst case.'}


def latest_arm():
    return HARD_END - ACTIVE_S - CLEANUP_S


def window_ok(now):
    return NOT_BEFORE <= now and now + ACTIVE_S + CLEANUP_S <= HARD_END


def arm_fits(elapsed_s):
    """The draft's arm entry rule (it compares bash SECONDS with LAST_ARM_ENTRY_S), for the tests."""
    return elapsed_s <= LAST_ARM_ENTRY_S


def g4_repeat_fits(elapsed_s):
    """The draft's trim rule for g4-live-1 and g4-solo-1, for the tests."""
    return elapsed_s <= TRIM_G4_REPEAT_S
