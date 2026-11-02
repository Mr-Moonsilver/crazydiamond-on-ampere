"""q18a-alpha-next box queue: one phase budget for the watchdog and the draft, relative to the run's own start only.

Startup, bind and the unarmed check are held-f's (native40's phase caps, unchanged). Then the core arms (alpha_next_arms, parts
'core' and 'extra'), each behind a typed all-rank FULL entry with instruments and ingress receipts; a typed FULL and the core
grade (the queue's verdict); then the g4 tail arms (part 'tail', reads only), each behind its own typed FULL; then
instruments-end and the end FULL (counted: leak accounting covers the tail), the read over every arm, native40's end identity
and owned cleanup.

Every limit counts from the run's own start: ACTIVE_S is the outer watchdog (the box budget is 9,000 s including cleanup, the
lead's cap), and an arm, core or tail, enters only while bash SECONDS <= LAST_ARM_ENTRY_S, so that arm's own worst case and the
whole remaining tail still fit ACTIVE_S (core and extra arms: LAST_ARM_ENTRY_S, which still reserves the core FULL and grade;
tail arms: LAST_TAIL_ENTRY_S, the same rule once the core grade is written). A core arm past it is NOT_RUN and the queue is INC;
an extra or tail arm past it is NOT_RUN and not counted (the extra round is "if the budget allows", lead 13:35). There is no absolute time here: launch time is governed by the lead at launch (operator, 2 Nov 2026: "we
don't put in code what you govern"), so this gate has no hard end, no not-before, no latest start and no window, and imports no
clock. EXPECTED_RUNTIME_S is descriptive, for the lead's planning.
"""
from native41_box_gate import ARM_KILL_AFTER_S, ARM_TIMEOUT_S, FULL_IDLE_CAP_S, INGRESS_CAP_S, INSTRUMENTS_CAP_S, KEPT
from native40_box_gate import CLEANUP_S, PHASES as NATIVE40_PHASES
import alpha_next_arms as table

CORE_ARMS = len(table.arms('core'))
EXTRA_ARMS = len(table.arms('extra'))
TAIL_ARMS = len(table.arms('tail'))
ARMS = CORE_ARMS + EXTRA_ARMS + TAIL_ARMS
GRADE_CAP_S = 600                                         # the core grade and the read, each (Step A's grade cap)
PHASES = {name: NATIVE40_PHASES[name] for name in KEPT}
PHASES.update({
    'instruments': 2 * INSTRUMENTS_CAP_S,                 # start and end
    'arm_entries': ARMS * FULL_IDLE_CAP_S,                # a typed all-rank FULL before every arm
    'alpha_next_arms': ARMS * (ARM_TIMEOUT_S + ARM_KILL_AFTER_S),
    'ingress': ARMS * INGRESS_CAP_S,
    'core_full_idle': FULL_IDLE_CAP_S,                    # the typed FULL before the core grade
    'grade': GRADE_CAP_S,                                 # the core grade
    'end_full_idle': FULL_IDLE_CAP_S,
    'read': GRADE_CAP_S,                                  # the read over every arm
})
WORST_CASE_SUM_S = sum(PHASES.values())
BOX_BUDGET_S = 9000                                       # the lead's cap, cleanup included, relative to the run's start
ACTIVE_S = BOX_BUDGET_S - CLEANUP_S
ACTIVE_AND_CLEANUP = ACTIVE_S + CLEANUP_S
ARM_RESERVE_S = FULL_IDLE_CAP_S + ARM_TIMEOUT_S + ARM_KILL_AFTER_S + INGRESS_CAP_S
# What must still fit after an arm: after a core arm the core FULL and grade, then the end; after a tail arm only the end.
END_RESERVE_S = INSTRUMENTS_CAP_S + PHASES['end_full_idle'] + PHASES['read'] + PHASES['report'] + PHASES['end_identity']
TAIL_RESERVE_S = PHASES['core_full_idle'] + PHASES['grade'] + END_RESERVE_S
LAST_ARM_ENTRY_S = ACTIVE_S - ARM_RESERVE_S - TAIL_RESERVE_S        # core arms
LAST_TAIL_ENTRY_S = ACTIVE_S - ARM_RESERVE_S - END_RESERVE_S        # tail arms: the core grade is already written
assert 0 < LAST_ARM_ENTRY_S < LAST_TAIL_ENTRY_S
ENTRY_LIMIT_S = {name: globals()[name] for name, _ in table.ENTRY_LIMIT.values()}     # the arm table's limits, in seconds

# Descriptive only. Step B's box times (2 Nov, q18a-step-b-e): first arm about 680 s after the start, band 1 arms about 119 s,
# band 2 about 166 s. The lead's per-arm planning figures: Step B about 1.6 min, C0 about 3.6 min (stepped base).
FIRST_ARM_S = 680
PER_ARM_S = {1: 119, 2: 166}
BETWEEN_S = 150                                           # core FULL and core grade, an estimate
END_S = 600                                               # instruments-end, end FULL, read, logs, end identity, stop, restores
PER_ARM_MIN = {'step_b': 1.6, 'c0': 3.6}


def _estimate(per_arm):
    """Walk the table at a per-arm wall time; an arm past its entry limit is NOT_RUN and takes no time, as in the draft."""
    entered, clock, tail_started = [], FIRST_ARM_S, False
    for row in table.arms():
        if row['part'] == 'tail' and not tail_started:
            clock += BETWEEN_S
            tail_started = True
        if clock > ENTRY_LIMIT_S[row['entry_limit']]:
            continue
        entered.append(dict(label=row['label'], part=row['part'], enters_s=clock))
        clock += per_arm(row)
    count = lambda part: sum(1 for r in entered if r['part'] == part)
    return dict(core_arms_entering=count('core'), extra_arms_entering=count('extra'), tail_arms_entering=count('tail'),
                arms_entering=len(entered), last_arm_enters_s=entered[-1]['enters_s'] if entered else None,
                total_s=clock + END_S, total_min=round((clock + END_S) / 60, 1))


EXPECTED_RUNTIME_S = {
    'arms': ARMS, 'core_arms': CORE_ARMS, 'extra_arms': EXTRA_ARMS, 'tail_arms': TAIL_ARMS,
    'by_band': _estimate(lambda row: PER_ARM_S[row['band']]),
    'at_step_b_rate': _estimate(lambda row: PER_ARM_MIN['step_b'] * 60),
    'at_c0_rate': _estimate(lambda row: PER_ARM_MIN['c0'] * 60),
    'worst_case_budget': BOX_BUDGET_S,
    'last_arm_entry_s': LAST_ARM_ENTRY_S,
    'last_tail_entry_s': LAST_TAIL_ENTRY_S,
    'basis': 'estimate: first arm entry, per-arm wall by band and the end from Step B run e (2 Nov); the two rate rows use the '
             'lead\'s per-arm minutes (Step B about 1.6, C0 about 3.6). C0 on the fused, grouped engine and every band 2 C0 '
             'arm are unmeasured on the box. An extra or tail arm that would enter after its limit is NOT_RUN and not counted. '
             'The 9,000 s budget holds the worst case.',
}


def arm_fits(elapsed_s, part='core'):
    """The draft's arm entry rule (bash SECONDS against LAST_ARM_ENTRY_S or LAST_TAIL_ENTRY_S), for the tests."""
    return elapsed_s <= ENTRY_LIMIT_S[table.ENTRY_LIMIT[part][0]]
