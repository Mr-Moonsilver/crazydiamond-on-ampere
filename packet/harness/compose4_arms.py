"""q18a-compose-4 arm table: the one declared table (compose-4, 5 Nov 2026). Change it here and nowhere else.

The four lever tracks on one engine, at one joint point, for the first time. One launch: CUDA_DEVICE_MAX_CONNECTIONS=32 and the launch switches hold for the whole run, so both arms carry connections 32 and their ratio can't read it.
Selected per arm by typed control at the arm's paused idle FULL entry:
  sm    the a1 live arm's shape with every lever on (5d full, private PLE buffer, the yield): the exact-path smoke; its failure stops the table
  dec1  one standing stream, decode only, about 20 s, the composed levers on (the operator's question, 5 Nov): beside decode-am1's eight
  cmp   composed: 5d's full + the private PLE buffer + the yield, under the closed-loop agent replay
  ctl   control: 5d's full only (private PLE buffer off, the yield off), the same replay, the same seed; the salts differ, so neither hits the other's cache
Order: sm | decode-am1 | dec1 | cmp | ctl | the grade.
The replay's numbers (lead's rulings, 5 Nov, QUESTION-lead-1.md and after): 8 agents, 18 minutes each, the same draws in both arms; fresh 60-131k, no prompt above 135,000 (the largest stock append that has run under the cap).
"""
import sys

import a1_trunk_arms as _a1
import inwait_arms as _inwait

STANDING_DECODERS = _inwait.STANDING_DECODERS
STANDING_GREEDY = _inwait.STANDING_GREEDY
GREEDY_TEMPERATURE = _inwait.GREEDY_TEMPERATURE
STANDING_SAMPLED = _inwait.STANDING_SAMPLED
BAND1_INDEXES = (0,)
BAND_SIZE = {'core': {1: 2}, 'tail': {1: 2}}
SMOKE_ROW = ('core', 'sm', 'live', 0)
ARM_TABLE = (SMOKE_ROW, ('core', 'dec1', 'live', 0), ('core', 'cmp', 'live', 0), ('core', 'ctl', 'live', 0))
DECODE_ARMS = _inwait.DECODE_ARMS
REFERENCE_ARM = _inwait.REFERENCE_ARM
FIXED_REFERENCE_TOK_S = _inwait.FIXED_REFERENCE_TOK_S
PARTS = _inwait.PARTS
ENTRY_LIMIT = _inwait.ENTRY_LIMIT
GRADED = _inwait.GRADED
SELECTIONS = ('sm', 'dec1', 'cmp', 'ctl')
KIND = {'sm': 'live-arm', 'dec1': 'one-stream', 'cmp': 'replay', 'ctl': 'replay'}
YIELD = {'sm': True, 'dec1': True, 'cmp': True, 'ctl': False}
PLE_PRIVATE = {'sm': True, 'dec1': True, 'cmp': True, 'ctl': False}
PLE_WITNESS = dict.fromkeys(SELECTIONS, False)     # the witness moves timing (READY ple-private); it ran already, 0 unequal in 526
OFF, ON = 'ctl', 'cmp'
LAUNCH = _a1.LAUNCH
LAUNCH_ENV = _a1.LAUNCH_ENV
CONNECTIONS_ENV = 'CUDA_DEVICE_MAX_CONNECTIONS=32'
PLE_ENV = 'FLASH_S5_PLE_PRIVATE=1 FLASH_S5_PLE_WITNESS=1'    # installed for the run, as q18a-ple-private-1024; each arm selects them, and no arm selects the witness
PLANS = {'pair': _a1.PLANS['pair'], 'none': ()}
PLAN = {'sm': 'pair', 'dec1': 'none', 'cmp': 'none', 'ctl': 'none'}
ARM_GENERATIONS = 10
ONE_STREAM_WINDOW_S = 20.0

# The replay. A prompt above MAX_PROMPT_TOKENS never goes out: the agent rolls to a new session instead (READY, the cap's arithmetic).
REPLAY = dict(AGENTS=8, DURATION=1080.0, WARMUP=300.0, SEED=1, START_MIN=60000, START_MAX=131000, APPEND_MIN=500, APPEND_MAX=5000, REPLY_MIN=200, REPLY_MAX=1000, TOOL_MIN=1.0, TOOL_MAX=5.0)
MAX_PROMPT_TOKENS = 135000
PROMPT_SLACK_TOKENS = 1000                          # the replay's token count is an estimate until usage comes back
LAUNCH_MAX_NEW, LAUNCH_PER_BULK = LAUNCH['max_new'], LAUNCH['per_bulk']
MIN_NATIVE_ROWS = 32768                             # native_auto_arm.MIN_NATIVE_ROWS

# The rule's numbers, fixed before the run (brief, 5 Nov): compose4_rule.py reads them.
RULE_PREFILL_TOK_S = 3300.0
RULE_DECODE_VS_FIXED = 0.60
RULE_MIN_NATIVE_BULKS = 3
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


def generations_of(sel):
    """A live arm's exact request count; None for a replay, whose count is in its own record, and read back by the ingress check."""
    if KIND[sel] == 'live-arm':
        return ARM_GENERATIONS + len(PLANS[PLAN[sel]])
    return 1 if KIND[sel] == 'one-stream' else None


def arms(part=None):
    parts = (part,) if isinstance(part, str) else part
    rows = []
    for row_part, sel, cohort, index in ARM_TABLE:
        if parts is not None and row_part not in parts:
            continue
        # 5d's full on every arm: the exact successor, the reserved cap, the warm block on every rank, each from a flush; cf and hf with the byte check on.
        rows.append(dict(label=label(sel, 2, cohort, index), selection=selection(sel, 2), part=row_part, c1=sel, kind=KIND[sel], attn=True, lazy_recv=True, decode_merge='chunk', inwait=True, next_levers=True,
                         prepare_slices=True, score_warm=True, score_warm_rank0=False, allocator_flush=True, cow_fast=True, reserved_cap=True, cow_host_filter=True, cow_verify=True, exact_successor=True,
                         ple_private=PLE_PRIVATE[sel], ple_witness=PLE_WITNESS[sel], overlap=True, defer_bind=True, c0='on', stage_local=True, size=2, cohort=cohort, index=index, band=band_of_index(index),
                         extra=False, entry_limit=ENTRY_LIMIT[row_part][0], yield_on=YIELD[sel], plan=PLANS[PLAN[sel]], generations=generations_of(sel)))
    return rows


def labels(part=None):
    return [row['label'] for row in arms(part)]


def by_label():
    return {row['label']: row for row in arms()}


def matched_off(row):
    """c1's name: cmp's control; None for the rest."""
    return label(OFF, row['size'], row['cohort'], row['index']) if row.get('c1') == ON else None


def matched_on(row):
    return None


def band1_glob():
    return '[%s]' % ''.join(str(i) for i in BAND1_INDEXES)


def check():
    names = labels()
    assert names == ['sm.g2-live-0', 'dec1.g2-live-0', 'cmp.g2-live-0', 'ctl.g2-live-0'] and names[0] == COLD_ARM and labels('tail') == []
    assert set(KIND) == set(YIELD) == set(PLE_PRIVATE) == set(PLE_WITNESS) == set(PLAN) == set(SELECTIONS)
    assert REPLAY['START_MAX'] + PROMPT_SLACK_TOKENS < MAX_PROMPT_TOKENS, 'a fresh prompt fits under the ceiling'
    rows = by_label()
    on, off = rows['cmp.g2-live-0'], rows['ctl.g2-live-0']
    assert sorted(k for k in on if on[k] != off[k]) == ['c1', 'label', 'ple_private', 'selection', 'yield_on'], 'the two arms differ by flags only'
    assert on['ple_private'] and on['yield_on'] and not off['ple_private'] and not off['yield_on']
    assert all(r['score_warm'] and r['reserved_cap'] and r['exact_successor'] and r['allocator_flush'] for r in rows.values()), "5d's full on every arm"
    assert not any(r['ple_witness'] for r in rows.values()), 'no arm carries the witness'
    return True


check()

if __name__ == '__main__':
    # The queue's ingress check: python3 compose4_arms.py generations LABEL RECORD (a replay's count is the record's own; none readable, none printed: the check reads that as unknown)
    if len(sys.argv) != 4 or sys.argv[1] != 'generations':
        raise SystemExit(2)
    count = by_label()[sys.argv[2]]['generations']
    if count is None:
        import json
        try:
            count = json.load(open(sys.argv[3])).get('replay_requests_sent')
        except (OSError, ValueError):
            count = None
    if type(count) is int:
        print(count)
