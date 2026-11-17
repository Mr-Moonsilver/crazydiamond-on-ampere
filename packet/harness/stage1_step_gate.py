"""q18a-stage1-step box queue: one phase budget for the watchdog and the draft, relative to the run's own start only.

alpha-next-c1's gate (alpha_next_c1_gate) with this queue's arm table (stage1_step_arms: 18 core arms, no tail). Startup, bind and
the unarmed check are held-f's (native40's phase caps, unchanged); every arm cap, the grade and read caps and both entry limits follow
alpha-next's formulas over this budget (the entry limits do not depend on the arm count: an arm enters only while its own worst case
and everything after it still fit ACTIVE_S). The box budget is 10,200 s including cleanup (CLEANUP_S 900), relative to the run's start
(see BOX_BUDGET_S). There is no absolute time here: no hard end, no not-before, no latest start, no window, and
no clock import.

EXPECTED_RUNTIME_S is descriptive, from the lead's box walls (2 Nov, 16:40): the first arm about 480 s after the start; band 1 solo
75 s, live 147 s, the cold first arm 246 s; band 2 solo 128 s, live 273 s. The selected arms are unmeasured on the box; their walls are set
equal to the base. ``slower`` is the same walk at 1.5x every arm wall.
"""
from native41_box_gate import ARM_KILL_AFTER_S, ARM_TIMEOUT_S, FULL_IDLE_CAP_S, INGRESS_CAP_S, INSTRUMENTS_CAP_S, KEPT
from native40_box_gate import CLEANUP_S, PHASES as NATIVE40_PHASES
import stage1_step_arms as table

CORE_ARMS = len(table.arms('core'))
EXTRA_ARMS = 0                                            # no extra part in this table
TAIL_ARMS = len(table.arms('tail'))                       # 0: no tail in this table
ARMS = CORE_ARMS + TAIL_ARMS
GRADE_CAP_S = 600                                         # the core grade and the read, each (alpha-next's)
PHASES = {name: NATIVE40_PHASES[name] for name in KEPT}
PHASES.update({
    'instruments': 2 * INSTRUMENTS_CAP_S,                 # start and end
    'arm_entries': ARMS * FULL_IDLE_CAP_S,                # a typed all-rank FULL before every arm
    'stage1_step_arms': ARMS * (ARM_TIMEOUT_S + ARM_KILL_AFTER_S),
    'ingress': ARMS * INGRESS_CAP_S,
    'core_full_idle': FULL_IDLE_CAP_S,                    # the typed FULL before the core grade
    'grade': GRADE_CAP_S,                                 # the core grade
    'end_full_idle': FULL_IDLE_CAP_S,
    'read': GRADE_CAP_S,                                  # the read over every arm
})
WORST_CASE_SUM_S = sum(PHASES.values())
BOX_BUDGET_S = 10200                                       # cleanup included, relative to the run's start (lead's cap >= this + staging). The
# Fixed so every arm enters at 1.5x the walls. The always pair is last, so a shorter budget costs it first.
ACTIVE_S = BOX_BUDGET_S - CLEANUP_S
ACTIVE_AND_CLEANUP = ACTIVE_S + CLEANUP_S
ARM_RESERVE_S = FULL_IDLE_CAP_S + ARM_TIMEOUT_S + ARM_KILL_AFTER_S + INGRESS_CAP_S
# What must still fit after an arm: after a core arm the core FULL and grade, then the end; after a tail arm only the end.
END_RESERVE_S = INSTRUMENTS_CAP_S + PHASES['end_full_idle'] + PHASES['read'] + PHASES['report'] + PHASES['end_identity']
TAIL_RESERVE_S = PHASES['core_full_idle'] + PHASES['grade'] + END_RESERVE_S
LAST_ARM_ENTRY_S = ACTIVE_S - ARM_RESERVE_S - TAIL_RESERVE_S        # core arms
LAST_TAIL_ENTRY_S = ACTIVE_S - ARM_RESERVE_S - END_RESERVE_S        # no tail rows; the draft's budget function reads the name
assert 0 < LAST_ARM_ENTRY_S < LAST_TAIL_ENTRY_S
ENTRY_LIMIT_S = {name: globals()[name] for name, _ in table.ENTRY_LIMIT.values()}     # the arm table's limits, in seconds

# Descriptive only: the lead's box walls (2 Nov, 16:40), entry to entry.
FIRST_ARM_S = 480
COLD_FIRST_S = 246                                        # the run's first arm (cold)
WALL_S = {(1, 'solo'): 75, (1, 'live'): 147, (2, 'solo'): 128, (2, 'live'): 273}
BETWEEN_S = 150                                           # core FULL and core grade (alpha-next's estimate)
END_S = 600                                               # instruments-end, end FULL, read, logs, end identity, stop, restores
SLOWER = 1.5


def arm_wall(row, first=False):
    """One arm's expected wall (every selected arm as the base)."""
    return COLD_FIRST_S if first else WALL_S[(row['band'], row['cohort'])]


def slower_wall(row, first=False):
    return SLOWER * arm_wall(row, first)


def _estimate(per_arm):
    """Walk the table at a per-arm wall time; an arm past its entry limit is NOT_RUN and takes no time, as in the draft."""
    entered, clock = [], FIRST_ARM_S
    rows = table.arms()
    for k, row in enumerate(rows):
        if clock > ENTRY_LIMIT_S[row['entry_limit']]:
            continue
        wall = per_arm(row, k == 0)
        entered.append(dict(label=row['label'], part=row['part'], enters_s=clock, wall_s=wall,
                            margin_s=ENTRY_LIMIT_S[row['entry_limit']] - clock))
        clock += wall
    return dict(not_run=[r['label'] for r in rows if r['label'] not in {e['label'] for e in entered}],
                core_arms_entering=len(entered), extra_arms_entering=0, tail_arms_entering=0, arms_entering=len(entered),
                last_arm_enters_s=entered[-1]['enters_s'] if entered else None,
                last_arm_ends_s=entered[-1]['enters_s'] + entered[-1]['wall_s'] if entered else None,
                min_margin_s=min((r['margin_s'] for r in entered), default=None), arms=entered,
                total_s=clock + BETWEEN_S + END_S, total_min=round((clock + BETWEEN_S + END_S) / 60, 1))


EXPECTED_RUNTIME_S = {
    'arms': ARMS, 'core_arms': CORE_ARMS, 'extra_arms': EXTRA_ARMS, 'tail_arms': TAIL_ARMS,
    'nominal': _estimate(arm_wall),
    'slower': _estimate(slower_wall),
    'worst_case_budget': BOX_BUDGET_S,
    'last_arm_entry_s': LAST_ARM_ENTRY_S,
    'last_tail_entry_s': LAST_TAIL_ENTRY_S,
    'basis': 'estimate from the lead\'s box walls (2 Nov 16:40): first arm +480 s, the cold first arm 246 s, band 1 solo 75 / live 147 s, '
             'band 2 solo 128 / live 273 s, the selected arms as the base (unmeasured on the box); core FULL and grade 150 s, the end '
             '600 s. An arm that would enter after its limit is NOT_RUN. The 10,200 s budget holds the worst case of what enters.',
}


def rows():
    return table.arms()


def arm_fits(elapsed_s, part='core'):
    """The draft's arm entry rule (bash SECONDS against LAST_ARM_ENTRY_S), for the tests."""
    return elapsed_s <= ENTRY_LIMIT_S[table.ENTRY_LIMIT[part][0]]
