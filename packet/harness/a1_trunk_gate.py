"""q18a-a1-trunk box queue: inwait_gate's formulas with this table (a1_trunk_arms: 8 core arms, no tail, one decode-only baseline run first). One phase budget for the watchdog and the draft, relative to the run's own start only.

EXPECTED_RUNTIME_S is descriptive, from c1's walls; the on arm is set equal to the off arm. ``slower`` is the same walk at 1.5x every wall.
"""
from native41_box_gate import ARM_KILL_AFTER_S, ARM_TIMEOUT_S, FULL_IDLE_CAP_S, INGRESS_CAP_S, INSTRUMENTS_CAP_S, KEPT
from native40_box_gate import CLEANUP_S, PHASES as NATIVE40_PHASES
import a1_trunk_arms as table

DECODE_ARMS = len(table.DECODE_ARMS)
CORE_ARMS = len(table.arms('core'))
EXTRA_ARMS = 0
TAIL_ARMS = len(table.arms('tail'))
ARMS = DECODE_ARMS + CORE_ARMS + TAIL_ARMS
GRADE_CAP_S = 600
PHASES = {name: NATIVE40_PHASES[name] for name in KEPT}
PHASES.update({
    'instruments': 2 * INSTRUMENTS_CAP_S,
    'arm_entries': ARMS * FULL_IDLE_CAP_S,
    'a1_trunk_arms': ARMS * (ARM_TIMEOUT_S + ARM_KILL_AFTER_S),
    'ingress': ARMS * INGRESS_CAP_S,
    'core_full_idle': FULL_IDLE_CAP_S,
    'grade': GRADE_CAP_S,
    'end_full_idle': FULL_IDLE_CAP_S,
    'read': GRADE_CAP_S,
})
WORST_CASE_SUM_S = sum(PHASES.values())
BOX_BUDGET_S = 8700
ACTIVE_S = BOX_BUDGET_S - CLEANUP_S
ACTIVE_AND_CLEANUP = ACTIVE_S + CLEANUP_S
ARM_RESERVE_S = FULL_IDLE_CAP_S + ARM_TIMEOUT_S + ARM_KILL_AFTER_S + INGRESS_CAP_S
END_RESERVE_S = INSTRUMENTS_CAP_S + PHASES['end_full_idle'] + PHASES['read'] + PHASES['report'] + PHASES['end_identity']
TAIL_RESERVE_S = PHASES['core_full_idle'] + PHASES['grade'] + END_RESERVE_S
LAST_ARM_ENTRY_S = ACTIVE_S - ARM_RESERVE_S - TAIL_RESERVE_S        # decode and core arms
LAST_TAIL_ENTRY_S = ACTIVE_S - ARM_RESERVE_S - END_RESERVE_S        # tail arms: the core grade is already written
assert 0 < LAST_ARM_ENTRY_S < LAST_TAIL_ENTRY_S
ENTRY_LIMIT_S = {name: globals()[name] for name, _ in table.ENTRY_LIMIT.values()}

FIRST_ARM_S = 780                                        # build-3b's box: launch to the first prefill arm, 13-14 min
COLD_FIRST_S = 70                                         # the run's first prefill arm (cold)
WALL_S = {(1, 'solo'): 50, (1, 'live'): 62}                # build-3b's band 1 walls, plus returners' tails on live arms (a guess until the first run)
DECODE_WALL_S = 60
BETWEEN_S = 30
END_S = 240
SLOWER = 1.5
PER_ARM_ALL_IN_MIN = 2.1                                  # c1's old all-in basis, kept for the record


def rows():
    """Every arm in run order: the exact-path smoke, the decode-only baselines, then the rest of the table."""
    rows_ = table.arms()
    return rows_[:1] + [dict(d, entry_limit='LAST_ARM_ENTRY_S', band=1) for d in table.DECODE_ARMS] + rows_[1:]


def arm_wall(row, first=False):
    if row['part'] == 'decode':
        return DECODE_WALL_S
    return COLD_FIRST_S if first else WALL_S[(row['band'], row['cohort'])]


def _estimate(scale):
    entered, clock, tail_started, first = [], FIRST_ARM_S, False, True
    for row in rows():
        if row['part'] == 'tail' and not tail_started:
            clock += BETWEEN_S
            tail_started = True
        if clock > ENTRY_LIMIT_S[row['entry_limit']]:
            continue
        wall = scale * arm_wall(row, first and row['part'] != 'decode')
        if row['part'] != 'decode':
            first = False
        entered.append(dict(label=row['label'], part=row['part'], enters_s=clock, wall_s=wall,
                            margin_s=ENTRY_LIMIT_S[row['entry_limit']] - clock))
        clock += wall
    return dict(not_run=[r['label'] for r in rows() if r['label'] not in {e['label'] for e in entered}],
                arms_entering=len(entered), core_arms_entering=sum(1 for r in entered if r['part'] == 'core'),
                decode_arms_entering=sum(1 for r in entered if r['part'] == 'decode'),
                tail_arms_entering=sum(1 for r in entered if r['part'] == 'tail'),
                last_core_arm_enters_s=max((r['enters_s'] for r in entered if r['part'] == 'core'), default=None),
                last_arm_ends_s=entered[-1]['enters_s'] + entered[-1]['wall_s'] if entered else None,
                min_margin_s=min((r['margin_s'] for r in entered), default=None), arms=entered,
                total_s=clock + BETWEEN_S + END_S, total_min=round((clock + BETWEEN_S + END_S) / 60, 1))


EXPECTED_RUNTIME_S = {
    'arms': ARMS, 'decode_arms': DECODE_ARMS, 'core_arms': CORE_ARMS, 'extra_arms': EXTRA_ARMS, 'tail_arms': TAIL_ARMS,
    'nominal': _estimate(1.0),
    'slower': _estimate(SLOWER),
    'per_arm_all_in_min': round(ARMS * PER_ARM_ALL_IN_MIN, 1),
    'worst_case_budget': BOX_BUDGET_S,
    'last_arm_entry_s': LAST_ARM_ENTRY_S,
    'last_tail_entry_s': LAST_TAIL_ENTRY_S,
    'basis': "measured on build-3b's box (4 Nov, 12:08-12:33): 13-14 min to the first prefill arm, 47-52 s a band 1 arm, 3 min at the end; band 2 estimated at about twice band 1",
}


def arm_fits(elapsed_s, part='core'):
    return elapsed_s <= ENTRY_LIMIT_S[table.ENTRY_LIMIT['core' if part == 'decode' else part][0]]
