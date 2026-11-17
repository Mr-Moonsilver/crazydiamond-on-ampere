"""C1 box queue: one phase budget for the watchdog and the draft; no absolute-time window.

alpha_c0_gate's rule over C1's sixteen required arms: startup, bind and the unarmed check are held-f's (native40's phase caps,
unchanged). Then twelve band 1 arms and four band 2 arms (alpha-perf's band 2, 131,094 tokens), each behind a typed all-rank
FULL entry, with instruments and ingress receipts, a typed FULL at the end and the grade, then native40's end identity and
owned cleanup. Every arm, band 2 included, runs under native41's arm caps, as alpha-perf's band 2 did.

ACTIVE_S is the outer watchdog, not the sum of the per-step caps: the box budget is 9,000 s including cleanup. The draft
guards each arm entry with LAST_ARM_ENTRY_S: an arm starts only when that arm's own worst case (entry, arm, ingress) and
the whole tail (instruments, typed FULL, grade, report, end identity) still fit ACTIVE_S. An arm past that point is
NOT_RUN and INC (a budget stop is read as a budget stop, never as an engine result). CLEANUP_S is native40_box_gate's.

Why 9,000 s and not c1/DESIGN.md's 7,200 s: on alpha-perf's box walls (1 Nov, read with tools/boxro.sh from the file times
under /home/<service-user>/s0/q18a-alpha-perf: first arm entry +480 s; cold solo 246 s, burst solo 74-81 s, burst live 146-148 s,
band 2 burst solo 125-131 s, band 2 burst live 272-274 s) the sixteenth arm enters about +2,517 s from queue start. 7,200 s
gives LAST_ARM_ENTRY_S 2,670 s, about 150 s of slack with the local and overlap speeds unmeasured on the box: C0's 870 s
defect class (only arms 1-3 could enter). 9,000 s is C0 l's and alpha-perf's budget and gives 4,470 s, which holds all
sixteen arms even if every arm took 1.5 times its serial wall (``arm_start_offsets``).

No absolute-time window (operator, 2 Nov: "we don't put in code what you govern"; launch time is the lead's). There is no
NOT_BEFORE, HARD_END, latest start or window_ok here, and neither the draft nor the owner runner refuses a start by wall clock.
The only time limits are relative, counted from the run's own start: the draft's outer timeout (ACTIVE_S, then CLEANUP_S for
owned cleanup) and LAST_ARM_ENTRY_S before each arm (bash SECONDS). native40_box_gate and every other frozen gate are unchanged.
"""
from native41_box_gate import (ARM_KILL_AFTER_S, ARM_TIMEOUT_S, FULL_IDLE_CAP_S, INGRESS_CAP_S,
                               INSTRUMENTS_CAP_S, KEPT)
from native40_box_gate import CLEANUP_S, PHASES as NATIVE40_PHASES

BAND1_ARMS = 12
BAND2_ARMS = 4
ARMS = BAND1_ARMS + BAND2_ARMS
GRADE_CAP_S = 600
PHASES = {name: NATIVE40_PHASES[name] for name in KEPT}
PHASES.update({
    'instruments': 2 * INSTRUMENTS_CAP_S,                 # start and end
    'arm_entries': ARMS * FULL_IDLE_CAP_S,                # a typed all-rank FULL before every arm
    'alpha_c1_arms': ARMS * (ARM_TIMEOUT_S + ARM_KILL_AFTER_S),
    'ingress': ARMS * INGRESS_CAP_S,
    'end_full_idle': FULL_IDLE_CAP_S,
    'grade': GRADE_CAP_S,
})
WORST_CASE_SUM_S = sum(PHASES.values())
BOX_BUDGET_S = 9000                                       # C0 l's and alpha_perf_gate's; DESIGN.md's 7,200 s left ~150 s slack
ACTIVE_S = BOX_BUDGET_S - CLEANUP_S
ACTIVE_AND_CLEANUP = ACTIVE_S + CLEANUP_S
ARM_RESERVE_S = FULL_IDLE_CAP_S + ARM_TIMEOUT_S + ARM_KILL_AFTER_S + INGRESS_CAP_S
TAIL_RESERVE_S = (INSTRUMENTS_CAP_S + FULL_IDLE_CAP_S + GRADE_CAP_S + PHASES['report'] + PHASES['end_identity'])
LAST_ARM_ENTRY_S = ACTIVE_S - ARM_RESERVE_S - TAIL_RESERVE_S
assert LAST_ARM_ENTRY_S > 0
# Measured on the box (q18a-alpha-perf, 1 Nov, file times): start of one arm to the start of the next, each including the arm's
# ingress check and the next typed FULL entry; the first arm's entry ends 480 s after queue start.
BOX_WALLS_S = dict(first_entry=480, cold_solo=246, solo=76, live=147, band2_solo=128, band2_live=273)
EXPECTED_RUNTIME_S = {
    'nominal': 3100,
    'worst_case_budget': BOX_BUDGET_S,
    'per_arm_s': {'serial-solo-0 (cold cohort)': 246, 'band 1 solo': 76, 'band 1 live': 147, 'band 2 solo': 128,
                  'band 2 live': 273},
    'basis': 'per-arm walls are alpha-perf\'s box run of 1 Nov (burst arms); local and overlap speed is unmeasured on the box, so '
             'each is set equal to serial. About 2,790 s to the end of the sixteenth arm (first entry +480 s), plus the tail: '
             'about 52 min. The 9,000 s budget gives LAST_ARM_ENTRY_S 4,470 s; the sixteenth arm is expected to enter at about '
             '+2,517 s, and still enters with every arm 1.5 times slower. DESIGN.md\'s 7,200 s would give 2,670 s, about 150 s of '
             'slack (C0\'s 870 s defect class); an arm past it is NOT_RUN and INC.'}


def arm_fits(elapsed_s):
    """The draft's arm entry rule (it compares bash SECONDS with LAST_ARM_ENTRY_S), for the tests."""
    return elapsed_s <= LAST_ARM_ENTRY_S


def arm_start_offsets(order, slow=1.0):
    """Expected start offset (s from queue start) of each arm of ``order`` ((selection, cohort, index) rows) on BOX_WALLS_S,
    every arm's wall multiplied by ``slow``; the first arm is the cold one."""
    offsets, at = [], float(BOX_WALLS_S['first_entry'])
    for n, (_selection, cohort, index) in enumerate(order):
        offsets.append(round(at))
        if n == 0:
            wall = BOX_WALLS_S['cold_solo'] if cohort == 'solo' else BOX_WALLS_S['live']
        else:
            wall = BOX_WALLS_S[('band2_' if index >= 2 else '') + cohort]
        at += wall * slow
    return offsets
