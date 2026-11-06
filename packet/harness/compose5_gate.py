"""q18a-compose-soak box queue: a1-trunk's gate formulas with this table (compose5_arms: the smoke, one decode-only baseline, the one-stream arm, four soak arms on dnr's turn mix, ABBA, no tail). One budget for the watchdog and the draft, relative to the run's own start only.

soak (lead, 6 Nov 04:10): four arms of 480 s traffic plus a 60 s tail, hard cap 1.3x nominal (~74 min), cleanup included; the run ends by 07:15 CET. The entry limits come from the nominal walls at 1.5x, not from the worst-case caps (those sum past the cap); a step that overruns meets the budget's TERM, whose path is the clean stop (tmp/decode-phase/STOP-dnr.md).

EXPECTED_RUNTIME_S is descriptive. The startup, the smoke and the end are build-3b's walls; the replay arm's wall is its table, DURATION and the turns it lets finish. ``slower`` is the same walk at 1.5x every wall except a replay's, whose length is its clock. The slot limit is the operator's script's; this budget holds the watchdog.
"""
from native41_box_gate import ARM_KILL_AFTER_S, ARM_TIMEOUT_S, FULL_IDLE_CAP_S, INGRESS_CAP_S, INSTRUMENTS_CAP_S, KEPT
from native40_box_gate import PHASES as NATIVE40_PHASES
import compose5_arms as table

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
    'compose5_arms': ARMS * (ARM_TIMEOUT_S + ARM_KILL_AFTER_S),
    'ingress': ARMS * INGRESS_CAP_S,
    'core_full_idle': FULL_IDLE_CAP_S,
    'grade': GRADE_CAP_S,
    'end_full_idle': FULL_IDLE_CAP_S,
    'read': GRADE_CAP_S,
})
WORST_CASE_SUM_S = sum(PHASES.values())                  # descriptive: every cap at once, past the budget by design

FIRST_ARM_S = 600                                        # dnr's box: 450 s from the restart to sm's entry, plus the preflight
COLD_FIRST_S = 220                                       # dnr's sm: 216 s
ONE_STREAM_WALL_S = 60                                   # dnr's dec1: 26 s and its entry
REPLAY_TAIL_S = 60                                       # dnr: 10-34 s past DURATION for the open turns; nr's native turns run longer (lead 04:10: 60 s)
ENTRY_S = 10                                             # dnr's entries: 0.3-8 s
DECODE_WALL_S = 60                                       # dnr's decode-am1: 29 s
BETWEEN_S = 30
END_S = 240
SLOWER = 1.5
CAP = 1.3                                                # the lead's hard cap over nominal (6 Nov 04:10)


def _nominal_s():
    """The table walked at its nominal walls with no entry limit: the cap's basis."""
    walls = FIRST_ARM_S + COLD_FIRST_S + DECODE_ARMS * DECODE_WALL_S + BETWEEN_S + END_S
    for row in table.arms()[1:]:
        walls += ENTRY_S + (table.REPLAY['DURATION'] + REPLAY_TAIL_S if row['kind'] == 'replay' else ONE_STREAM_WALL_S)
    return walls


NOMINAL_S = int(_nominal_s())
CLEANUP_S = 600                                          # this queue's own: STOP-dnr.md read 1.5-3 min of cleanup; native40's 900 is a third of this run
BOX_BUDGET_S = int(CAP * NOMINAL_S)                      # the hard cap, cleanup included: the outer timeout's KILL
ACTIVE_S = BOX_BUDGET_S - CLEANUP_S                      # the outer timeout's TERM, the clean stop's path
ACTIVE_AND_CLEANUP = ACTIVE_S + CLEANUP_S
ARM_RESERVE_S = int(CAP * (ENTRY_S + table.REPLAY['DURATION'] + REPLAY_TAIL_S))
END_RESERVE_S = int(CAP * END_S)
TAIL_RESERVE_S = int(CAP * BETWEEN_S) + END_RESERVE_S
LAST_ARM_ENTRY_S = ACTIVE_S - ARM_RESERVE_S - TAIL_RESERVE_S        # decode and core arms
LAST_TAIL_ENTRY_S = ACTIVE_S - ARM_RESERVE_S - END_RESERVE_S        # tail arms (none in this table)
assert 0 < LAST_ARM_ENTRY_S < LAST_TAIL_ENTRY_S
ENTRY_LIMIT_S = {name: globals()[name] for name, _ in table.ENTRY_LIMIT.values()}
PER_ARM_ALL_IN_MIN = 2.1                                  # c1's old all-in basis, kept for the record


def rows():
    """Every arm in run order: the exact-path smoke, the decode-only baselines, then the rest of the table."""
    rows_ = table.arms()
    return rows_[:1] + [dict(d, entry_limit='LAST_ARM_ENTRY_S', band=1) for d in table.DECODE_ARMS] + rows_[1:]


def arm_wall(row, first=False):
    if row['part'] == 'decode':
        return DECODE_WALL_S
    if row['kind'] == 'replay':
        return table.REPLAY['DURATION'] + REPLAY_TAIL_S
    return COLD_FIRST_S if row['kind'] == 'live-arm' else ONE_STREAM_WALL_S


def _estimate(scale):
    entered, clock, tail_started, first = [], FIRST_ARM_S, False, True
    for row in rows():
        if row['part'] == 'tail' and not tail_started:
            clock += BETWEEN_S
            tail_started = True
        if clock > ENTRY_LIMIT_S[row['entry_limit']]:
            continue
        wall = arm_wall(row, first and row['part'] != 'decode') + (ENTRY_S if row['part'] != 'decode' and not first else 0)
        wall = wall if row.get('kind') == 'replay' else scale * wall
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
    'nominal_s': NOMINAL_S, 'cap': CAP,
    'basis': "dnr's box (5 Nov): 450 s from the restart to sm, sm 216 s, decode-am1 29 s, dec1 26 s, entries 0.3-8 s, a replay arm DURATION plus 10-34 s; the end is compose-4's 4 min",
}


def arm_fits(elapsed_s, part='core'):
    return elapsed_s <= ENTRY_LIMIT_S[table.ENTRY_LIMIT['core' if part == 'decode' else part][0]]
