"""Alpha-perf box queue: one phase budget for the watchdog, the owner window and the draft.

Startup, bind and the unarmed check are held-f's (native40's phase caps, unchanged). Then held-f's eight
band 1 arms and up to eight band 2 arms, each behind a typed all-rank FULL entry, with instruments and ingress
receipts, a typed FULL at the end and the grade, then native40's end identity and owned cleanup.

Unlike held-f, ACTIVE_S is not the sum of the per-step caps: the lead's budget for this queue is about 2.5 h of box
time in total, cleanup included, and the worst-case sum of sixteen arm caps is far beyond it. ACTIVE_S is the outer
watchdog. The draft guards each arm entry with LAST_ARM_ENTRY_S: an arm starts only when that arm's own worst case
(entry, arm, ingress) and the whole tail (instruments, typed FULL, grade, report, end identity) still fit ACTIVE_S.
A band 1 or required band 2 arm past that point is NOT_RUN and INCONCLUSIVE; an optional band 2 arm past it is
NOT_RUN by budget, declared to the grade, and changes no verdict. No per-step cap of held-f is changed except the
grade's, which reads about twice held-f's progress lines.
``window_ok(time.time())`` is the single entry check used by the draft and the owner runner.
"""
from native41_box_gate import (ARM_KILL_AFTER_S, ARM_TIMEOUT_S, FULL_IDLE_CAP_S, INGRESS_CAP_S,
                               INSTRUMENTS_CAP_S, KEPT)
from native40_box_gate import CLEANUP_S, HARD_END, NOT_BEFORE, PHASES as NATIVE40_PHASES

BAND1_ARMS = 8
BAND2_ARMS = 8
BAND2_REQUIRED_ARMS = 4
ARMS = BAND1_ARMS + BAND2_ARMS
GRADE_CAP_S = 600
PHASES = {name: NATIVE40_PHASES[name] for name in KEPT}
PHASES.update({
    'instruments': 2 * INSTRUMENTS_CAP_S,                 # start and end
    'arm_entries': ARMS * FULL_IDLE_CAP_S,                # a typed all-rank FULL before every arm
    'progress_arms': ARMS * (ARM_TIMEOUT_S + ARM_KILL_AFTER_S),
    'ingress': ARMS * INGRESS_CAP_S,
    'end_full_idle': FULL_IDLE_CAP_S,
    'grade': GRADE_CAP_S,
})
WORST_CASE_SUM_S = sum(PHASES.values())
BOX_BUDGET_S = 9000                                       # lead: at most about 2.5 h of box time in total
ACTIVE_S = BOX_BUDGET_S - CLEANUP_S
ACTIVE_AND_CLEANUP = ACTIVE_S + CLEANUP_S
ARM_RESERVE_S = FULL_IDLE_CAP_S + ARM_TIMEOUT_S + ARM_KILL_AFTER_S + INGRESS_CAP_S
TAIL_RESERVE_S = (INSTRUMENTS_CAP_S + FULL_IDLE_CAP_S + GRADE_CAP_S + PHASES['report'] + PHASES['end_identity'])
LAST_ARM_ENTRY_S = ACTIVE_S - ARM_RESERVE_S - TAIL_RESERVE_S
assert LAST_ARM_ENTRY_S > 0
EXPECTED_RUNTIME_S = {
    'nominal': 4800,
    'worst_case_budget': BOX_BUDGET_S,
    'per_arm_s': {'band 1 current-solo-0 (cold cohort)': 245, 'band 1 burst-solo': 75, 'band 1 current-solo-1': 80,
                  'band 1 burst-live': 145, 'band 1 current-live': 220,
                  'band 2 burst-solo': 150, 'band 2 burst-live': 420},
    'basis': 'band 1 per-arm wall from held-f run 2 (arm wall_start_s/wall_end_s, 1,199 s for the eight arms; that '
             'whole queue took 35 min). Band 2 is an estimate: about 130 native chunks of 1,024, solo about 0.85 s per '
             'chunk (q18a-native43-dense-boundary run 2 past the boundary), live 1.77 s per chunk below the boundary '
             '(held-f run 2) and an assumed 2-4 s past it (never measured: this run is the first), plus cohort reopen, '
             '25 s decode-only, append and close. 35 min + 8 band 2 arms (4 x 7 + 4 x 2.5 min) + entries about 80 min.'}


def latest_arm():
    return HARD_END - ACTIVE_S - CLEANUP_S


def window_ok(now):
    return NOT_BEFORE <= now and now + ACTIVE_S + CLEANUP_S <= HARD_END


def arm_fits(elapsed_s):
    """The draft's arm entry rule (it compares bash SECONDS with LAST_ARM_ENTRY_S), for the tests."""
    return elapsed_s <= LAST_ARM_ENTRY_S
