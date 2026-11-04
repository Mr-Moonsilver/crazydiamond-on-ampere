"""q18a-a1-trunk arm table: the one declared table (a1-trunk, 4 Nov 2026). Change it here and nowhere else.

The returning-agent yield (A1 with the review's draining phase, results/hub/a1-trunk/DESIGN.md) over inwait's build-3b base (ps1: C1 defer, attn-meta, L1 + L2, in-wait, early successor,
fast slot mappings, PREPARE in slices; no fast CoW copy, no reserved cap). One axis, selected per arm by the typed control native_append_yield at the arm's paused idle FULL entry.
A returning agent: one standing stream ends its turn (the client cancels it, a real slot frees) and comes back with its own prompt plus a suffix under its own salt.
  sm   yield on, the exact-path smoke: arm 1, cold, two returners; if no yield is served nothing else runs
  on   yield on, two returners inside the bulk: 500 tokens at about bulk + 5 s, 1,500 at about bulk + 12 s
  off  the same with the yield off
  mn   yield on, three returners at once at about bulk + 6 s (500, 500, 1,500)
  mo   the same with the yield off
Order: sm live | decode-am1 | on off live | off on live (reversed) | mn mo live | on solo, no returner | the grade.
"""
import sys

import inwait_arms as _inwait

STANDING_DECODERS = _inwait.STANDING_DECODERS
STANDING_GREEDY = _inwait.STANDING_GREEDY
GREEDY_TEMPERATURE = _inwait.GREEDY_TEMPERATURE
STANDING_SAMPLED = _inwait.STANDING_SAMPLED
BAND1_INDEXES = (0, 1)
BAND_SIZE = {'core': {1: 2}, 'tail': {1: 2}}
SMOKE_ROW = ('core', 'sm', 'live', 0)
ARM_TABLE = (SMOKE_ROW, ('core', 'on', 'live', 0), ('core', 'off', 'live', 0), ('core', 'off', 'live', 1), ('core', 'on', 'live', 1),
             ('core', 'mn', 'live', 0), ('core', 'mo', 'live', 0), ('core', 'on', 'solo', 0))
DECODE_ARMS = _inwait.DECODE_ARMS
REFERENCE_ARM = _inwait.REFERENCE_ARM
FIXED_REFERENCE_TOK_S = _inwait.FIXED_REFERENCE_TOK_S
PARTS = _inwait.PARTS
ENTRY_LIMIT = _inwait.ENTRY_LIMIT
GRADED = _inwait.GRADED
SELECTIONS = ('sm', 'on', 'off', 'mn', 'mo')
OFF = 'off'
YIELD = {'sm': True, 'on': True, 'off': False, 'mn': True, 'mo': False}
MATCHED_OFF = {'on': 'off', 'mn': 'mo'}
LAUNCH = {'max_new': 2048, 'per_bulk': 4, 'handback_steps': 16, 'reentry_retries': 4}
LAUNCH_ENV = {'FLASH_S5_APPEND_YIELD_MAX_NEW': 2048, 'FLASH_S5_APPEND_YIELD_PER_BULK': 4, 'FLASH_S5_APPEND_YIELD_HANDBACK_STEPS': 16, 'FLASH_S5_APPEND_YIELD_REENTRY_RETRIES': 4}
# (standing index that returns, suffix tokens, seconds after the bulk is sent). Never a greedy stream.
PLANS = {'pair': ((7, 500, 5.0), (6, 1500, 12.0)), 'several': ((7, 500, 6.0), (6, 500, 6.0), (5, 1500, 6.0)), 'none': ()}
PLAN = {'sm': 'pair', 'on': 'pair', 'off': 'pair', 'mn': 'several', 'mo': 'several'}
REPLY_TOKENS = 64
BUDGET_FIRST_OUTPUT_S = {500: 2.0, 1500: 3.0}   # operator, 3 Nov 12:41 CET: first output of a short returning append during bulk prefill
BUDGET_BULK_SLACK_S = 3.0                       # bulk complete <= yield-off + 3.0 s
BUDGET_DECODE_RATIO = 0.97                      # standing decode >= 0.97 of yield-off
ARM_GENERATIONS = 10                            # 8 standing, the bulk, the stock append; the returners come on top
COHORTS = ('live', 'solo')
INDEXES = BAND1_INDEXES
SIZES = (1, 2, 4)
COLD_ARM = 'sm.g2-live-0'
SMOKE_ARM = COLD_ARM


def band_of_index(index):
    return 1 if index in BAND1_INDEXES else None


def selection(sel, size):
    return '%s.g%d' % (sel, size)


def label(sel, size, cohort, index):
    return '%s-%s-%d' % (selection(sel, size), cohort, index)


def plan_of(sel, cohort):
    return PLANS[PLAN[sel]] if cohort == 'live' else PLANS['none']


def arms(part=None):
    parts = (part,) if isinstance(part, str) else part
    rows = []
    for row_part, sel, cohort, index in ARM_TABLE:
        if parts is not None and row_part not in parts:
            continue
        band = band_of_index(index)
        plan = plan_of(sel, cohort)
        # inwait's selection ps1: the levers release_compose reads.
        rows.append(dict(label=label(sel, 2, cohort, index), selection=selection(sel, 2), part=row_part, c1=sel, attn=True, lazy_recv=True, decode_merge='chunk', inwait=True, next_levers=True,
                         prepare_slices=True, score_warm=False, cow_fast=False, reserved_cap=False, overlap=True, defer_bind=True, stage_local=True, size=2, cohort=cohort, index=index, band=band,
                         extra=False, entry_limit=ENTRY_LIMIT[row_part][0], yield_on=YIELD[sel], plan=plan, generations=ARM_GENERATIONS + len(plan)))
    return rows


def labels(part=None):
    return [row['label'] for row in arms(part)]


def by_label():
    return {row['label']: row for row in arms()}


def matched_off(row):
    """The yield-off arm of a live on arm at the same round; None for the rest. mn's pair has one round."""
    off = MATCHED_OFF.get(row.get('c1'))
    if off is None or row.get('cohort') != 'live':
        return None
    want = label(off, row['size'], row['cohort'], row['index'])
    return want if want in labels() else None


def check():
    names = labels()
    assert len(names) == len(set(names)) == 8, 'eight distinct prefill arm labels'
    assert names[0] == COLD_ARM and labels('tail') == [], 'the smoke first, no tail'
    assert set(YIELD) == set(PLAN) == set(SELECTIONS)
    assert [d['label'] for d in DECODE_ARMS] == [REFERENCE_ARM]
    for plan in PLANS.values():
        assert len({row[0] for row in plan}) == len(plan) and {row[0] for row in plan} <= set(STANDING_SAMPLED), 'a returner is one sampled standing stream'
        assert all(row[1] in BUDGET_FIRST_OUTPUT_S for row in plan)
        assert all(row[1] + 1 + 176 <= LAUNCH['max_new'] for row in plan), 'a returner stays under the launch bound'
    for row in arms():
        if row['yield_on'] and row['cohort'] == 'live' and row['c1'] != 'sm':
            assert matched_off(row), 'on arm without its matched off arm: ' + row['label']
    return True


check()

if __name__ == '__main__':
    # The queue's ingress check: python3 a1_trunk_arms.py generations LABEL
    if len(sys.argv) != 3 or sys.argv[1] != 'generations':
        raise SystemExit(2)
    print(by_label()[sys.argv[2]]['generations'])
