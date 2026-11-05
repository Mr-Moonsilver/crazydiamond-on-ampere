"""q18a-compose-5 arm table: the one declared table (compose-5, 5 Nov 2026; tmp/compose-4/DESIGN-compose-5.md). Change it here and nowhere else.

compose-4's run with the yield priced: the replay pair differs in one lever, and a frozen pair gives the price under control with its noise. c4b's pair differed in the yield and in the private PLE buffer.
One launch, connections 32 and the launch switches for the whole run, as compose-4. Selected per arm by typed control at the arm's paused idle FULL entry; every arm carries 5d's full and the private PLE buffer:
  sm    the a1 live arm's shape, the exact-path smoke; its failure stops the table. Its memory is the run's first read of r1 under yields
  dec1  one standing stream, decode only, as compose-4
  fon   frozen, yield on: a1's live arm, 8 standing, the bulk, two returners
  foff  the same, yield off
  cmp   the yield on, under the closed-loop agent replay: the arm the rule reads
  ny    the yield off, the same replay, the same seed; the salts differ, so neither hits the other's cache
Order: sm | decode-am1 | dec1 | fon foff | foff fon (reversed, on index 1) | cmp | ny | the grade.
Lost, on the lead's ruling: compose-4's replay read of the private PLE buffer on and off.
b13 on every arm, from dp8d (READING-box-dp8d.md, confirmed at 32): native chunks at 2048 rows below 13 Mamba blocks, 1024 from there, FLASH_S5_CHUNK_TOKENS=2048 for the whole run. hf (cow_host_filter) on every arm,
adopted by dp8d, as its b13h. What dp8d did not run with them: 5d's warm block and reserved cap, the exact successor, the private PLE buffer and the yield; those are compose-4's rows.
5f is out, for compose-6 (lead, 5 Nov 08:42): no score_split, no stock_rescue; the rule reads their counters as none.
"""
import sys

import a1_trunk_arms as _a1
import compose4_arms as _c4
import inwait_arms as _inwait

STANDING_DECODERS = _inwait.STANDING_DECODERS
STANDING_GREEDY = _inwait.STANDING_GREEDY
GREEDY_TEMPERATURE = _inwait.GREEDY_TEMPERATURE
STANDING_SAMPLED = _inwait.STANDING_SAMPLED
BAND1_INDEXES = (0, 1)
BAND_SIZE = _c4.BAND_SIZE
SMOKE_ROW = ('core', 'sm', 'live', 0)
ARM_TABLE = (SMOKE_ROW, ('core', 'dec1', 'live', 0), ('core', 'fon', 'live', 0), ('core', 'foff', 'live', 0), ('core', 'foff', 'live', 1), ('core', 'fon', 'live', 1),
             ('core', 'cmp', 'live', 0), ('core', 'ny', 'live', 0))
DECODE_ARMS = _inwait.DECODE_ARMS
REFERENCE_ARM = _inwait.REFERENCE_ARM
FIXED_REFERENCE_TOK_S = _inwait.FIXED_REFERENCE_TOK_S
PARTS = _inwait.PARTS
ENTRY_LIMIT = _inwait.ENTRY_LIMIT
GRADED = _inwait.GRADED
SELECTIONS = ('sm', 'dec1', 'fon', 'foff', 'cmp', 'ny')
KIND = {'sm': 'live-arm', 'dec1': 'one-stream', 'fon': 'live-arm', 'foff': 'live-arm', 'cmp': 'replay', 'ny': 'replay'}
YIELD = {'sm': True, 'dec1': True, 'fon': True, 'foff': False, 'cmp': True, 'ny': False}
PLE_PRIVATE = dict.fromkeys(SELECTIONS, True)
PLE_WITNESS = dict.fromkeys(SELECTIONS, False)
MATCHED_OFF = {'fon': 'foff', 'cmp': 'ny'}
OFF, ON = 'ny', 'cmp'
LAUNCH = _a1.LAUNCH
LAUNCH_ENV = _a1.LAUNCH_ENV
CONNECTIONS_ENV = _c4.CONNECTIONS_ENV
PLE_ENV = _c4.PLE_ENV
PLANS = _c4.PLANS
PLAN = {'sm': 'pair', 'dec1': 'none', 'fon': 'pair', 'foff': 'pair', 'cmp': 'none', 'ny': 'none'}
CHUNK_TOKENS = 2048
CHUNK_ENV = 'FLASH_S5_CHUNK_TOKENS=%d' % CHUNK_TOKENS
MAMBA_BLOCK_ROWS, HASH_BLOCK_ROWS = 4048, 176     # read from this model's chunks on the box (dp5): ends 4048 x k, last stop prompt-1 // 176 x 176
B13_BOUND = [13 * MAMBA_BLOCK_ROWS, 1024]         # dp8d's b13 and b13h: chunk_bound on native_stage_local, [end, rows]
ARM_GENERATIONS = _c4.ARM_GENERATIONS
ONE_STREAM_WINDOW_S = _c4.ONE_STREAM_WINDOW_S

# The replay and the ceiling stay compose-4's, so c4b reads across (lead, 5 Nov 07:49: the ceiling stays 135,000 here).
REPLAY = _c4.REPLAY
MAX_PROMPT_TOKENS = _c4.MAX_PROMPT_TOKENS
PROMPT_SLACK_TOKENS = _c4.PROMPT_SLACK_TOKENS
LAUNCH_MAX_NEW, LAUNCH_PER_BULK = LAUNCH['max_new'], LAUNCH['per_bulk']
MIN_NATIVE_ROWS = _c4.MIN_NATIVE_ROWS

RULE_PREFILL_TOK_S = _c4.RULE_PREFILL_TOK_S
RULE_DECODE_VS_FIXED = _c4.RULE_DECODE_VS_FIXED
RULE_MIN_NATIVE_BULKS = _c4.RULE_MIN_NATIVE_BULKS
COHORTS = _c4.COHORTS
INDEXES = BAND1_INDEXES
SIZES = _c4.SIZES
COLD_ARM = 'sm.g2-live-0'
SMOKE_ARM = COLD_ARM


def planned_chunks(prompt_tokens, cap=CHUNK_TOKENS, bound=None):
    """A fresh prompt's native chunks: the largest stock split at each position, as staged_chunk_plan finds it (dp6)."""
    end = prompt_tokens - 1
    stop = end // HASH_BLOCK_ROWS * HASH_BLOCK_ROWS
    position, sizes = 0, []
    while position < end:
        limit = cap if bound is None or position < bound[0] else min(cap, bound[1])
        following = min(position + limit, end)
        if following < end and following // MAMBA_BLOCK_ROWS * MAMBA_BLOCK_ROWS > position:
            following = following // MAMBA_BLOCK_ROWS * MAMBA_BLOCK_ROWS
        if position < stop < following:
            following = stop
        sizes.append(following - position)
        position = following
    return sizes


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
        # As compose-4's rows: 5d's full on every arm. The private PLE buffer is on every arm here, so the yield is the one axis.
        rows.append(dict(label=label(sel, 2, cohort, index), selection=selection(sel, 2), part=row_part, c1=sel, kind=KIND[sel], attn=True, lazy_recv=True, decode_merge='chunk', inwait=True, next_levers=True,
                         prepare_slices=True, score_warm=True, score_warm_rank0=False, allocator_flush=True, cow_fast=True, reserved_cap=True, cow_host_filter=True, cow_verify=True, exact_successor=True,
                         ple_private=PLE_PRIVATE[sel], ple_witness=PLE_WITNESS[sel], overlap=True, defer_bind=True, c0='on', stage_local=True, size=2, cohort=cohort, index=index, band=band_of_index(index),
                         extra=False, entry_limit=ENTRY_LIMIT[row_part][0], yield_on=YIELD[sel], plan=PLANS[PLAN[sel]], generations=generations_of(sel), chunk_bound=list(B13_BOUND)))
    return rows


def labels(part=None):
    return [row['label'] for row in arms(part)]


def by_label():
    return {row['label']: row for row in arms()}


def matched_off(row):
    """c1's name: an on arm's control at the same index; None for the rest."""
    off = MATCHED_OFF.get(row.get('c1'))
    return label(off, row['size'], row['cohort'], row['index']) if off else None


def matched_on(row):
    return None


def price_pairs():
    """(on, off) labels of the frozen pair, by index: fon foff on 0, foff fon on 1."""
    return [(label('fon', 2, 'live', index), label('foff', 2, 'live', index)) for index in BAND1_INDEXES]


def band1_glob():
    return '[%s]' % ''.join(str(i) for i in BAND1_INDEXES)


def check():
    names = labels()
    assert names == ['sm.g2-live-0', 'dec1.g2-live-0', 'fon.g2-live-0', 'foff.g2-live-0', 'foff.g2-live-1', 'fon.g2-live-1', 'cmp.g2-live-0', 'ny.g2-live-0'] and names[0] == COLD_ARM and labels('tail') == []
    assert set(KIND) == set(YIELD) == set(PLE_PRIVATE) == set(PLE_WITNESS) == set(PLAN) == set(SELECTIONS)
    assert REPLAY['START_MAX'] + PROMPT_SLACK_TOKENS < MAX_PROMPT_TOKENS, 'a fresh prompt fits under the ceiling'
    rows = by_label()
    for on, off in (('cmp.g2-live-0', 'ny.g2-live-0'), ('fon.g2-live-0', 'foff.g2-live-0')):
        on, off = rows[on], rows[off]
        assert sorted(k for k in on if on[k] != off[k]) == ['c1', 'label', 'selection', 'yield_on'], 'each pair differs in the yield alone'
        assert on['yield_on'] and not off['yield_on']
    assert rows['fon.g2-live-1']['index'] == rows['foff.g2-live-1']['index'] == 1
    assert all(r['score_warm'] and r['reserved_cap'] and r['exact_successor'] and r['allocator_flush'] and r['ple_private'] for r in rows.values()), "5d's full and the private PLE buffer on every arm"
    assert not any(r['ple_witness'] for r in rows.values()), 'no arm carries the witness'
    assert all(r['chunk_bound'] == B13_BOUND and r['cow_host_filter'] and r['cow_fast'] for r in rows.values()), "dp8d's b13h on every arm"
    assert B13_BOUND == [52624, 1024] and len(planned_chunks(65445, CHUNK_TOKENS, B13_BOUND)) == 40, "dp8d's 40 chunks at band 1"
    return True


check()

if __name__ == '__main__':
    # The queue's ingress check: python3 compose5_arms.py generations LABEL RECORD (as compose-4's).
    if len(sys.argv) != 4 or sys.argv[1] != 'generations':
        raise SystemExit(2)
    count = by_label()[sys.argv[2]]['generations']
    if count is None:
        import json
        try:
            count = json.load(open(sys.argv[3])).get('replay_requests_sent')
        except (OSError, ValueError):
            count = None
    print(count if type(count) is int else 0)
