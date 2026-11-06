"""q18a-compose-residue arm table: the one declared table (coverage-residue, 6 Nov 2026; tmp/a1-trunk/READY-coverage-residue.md). Change it here and nowhere else.

The operator's rule (5 Nov 23:10): the engine takes 100 % of prefill on the native lane while any decoder is live. nw is the hold build as
q18a-compose-hold runs it (b13h, native_hold 600). nr is nw with native_residue on: the hold's stock residue goes native (the shared-prefix
junction, a hit leaving under 64 rows, a refused selection held for a changed cohort, a lost race not counted as a refusal, and idle prefill:
held and selected with an empty cohort). Every arm counts its prefill rows by path and site, decoder live or idle (the engine's coverage
counter); any stock prefill row on nr is a FAIL in the grade (lead, 6 Nov 01:10: 100 % of all prefill rows). Forked from the hold at
5879e0e3. Discovery mode. One launch, connections 32 and FLASH_S5_CHUNK_TOKENS=2048 for the whole run. Per arm, by typed control at entry:
  sm    the a1 live arm's shape, the exact-path smoke, b13; its failure stops the table
  dec1  one standing stream, decode only, as compose-5
  nw    b13h under the replay with the hold, yield off, private PLE buffer off, as the hold's nw
  nr    the same with the residue on
Order: sm | decode-am1 | dec1 | nw nr (seed 1, index 0) | nr nw (seed 2, index 1) | nw nr (seed 3, index 2) | the grade.
Lean (the operator, 5 Nov): each replay arm is REPLAY_S seconds, every agent starting at once. The reserved cap and the warm block are off on
every arm, as in dnr. Each arm's salt comes from its label, so no arm hits another's cache.
"""
import sys

import a1_trunk_arms as _a1
import compose4_arms as _c4
import inwait_arms as _inwait

STANDING_DECODERS = _inwait.STANDING_DECODERS
STANDING_GREEDY = _inwait.STANDING_GREEDY
GREEDY_TEMPERATURE = _inwait.GREEDY_TEMPERATURE
STANDING_SAMPLED = _inwait.STANDING_SAMPLED
PAIRS = 3                                          # lead, 5 Nov 23:00: about 30 min nominal with startup, 3 matched pairs
BAND1_INDEXES = tuple(range(PAIRS))                # a replay has no band; sm, the one live arm, is index 0
BAND_SIZE = _c4.BAND_SIZE
SMOKE_ROW = ('core', 'sm', 'live', 0)
ARM_TABLE = (SMOKE_ROW, ('core', 'dec1', 'live', 0)) + tuple(('core', sel, 'live', index) for index in BAND1_INDEXES for sel in (('nw', 'nr') if index % 2 == 0 else ('nr', 'nw')))
DECODE_ARMS = _inwait.DECODE_ARMS
REFERENCE_ARM = _inwait.REFERENCE_ARM
FIXED_REFERENCE_TOK_S = _inwait.FIXED_REFERENCE_TOK_S
PARTS = _inwait.PARTS
ENTRY_LIMIT = _inwait.ENTRY_LIMIT
GRADED = _inwait.GRADED
SELECTIONS = ('sm', 'dec1', 'nw', 'nr')
KIND = {'sm': 'live-arm', 'dec1': 'one-stream', 'nw': 'replay', 'nr': 'replay'}
YIELD = {'sm': True, 'dec1': True, 'nw': False, 'nr': False}
PLE_PRIVATE = {'sm': True, 'dec1': True, 'nw': False, 'nr': False}
PLE_WITNESS = dict.fromkeys(SELECTIONS, False)
MATCHED_OFF = {'nr': 'nw'}
OFF, ON = 'nw', 'nr'
LAUNCH = _a1.LAUNCH
LAUNCH_ENV = _a1.LAUNCH_ENV
CONNECTIONS_ENV = _c4.CONNECTIONS_ENV
PLE_ENV = _c4.PLE_ENV
PLANS = _c4.PLANS
PLAN = {'sm': 'pair', 'dec1': 'none', 'nw': 'none', 'nr': 'none'}
CHUNK_TOKENS = 2048
CHUNK_ENV = 'FLASH_S5_CHUNK_TOKENS=%d' % CHUNK_TOKENS
MAMBA_BLOCK_ROWS, HASH_BLOCK_ROWS = 4048, 176     # read from this model's chunks on the box (dp5): ends 4048 x k, last stop prompt-1 // 176 x 176
B13_BOUND = [13 * MAMBA_BLOCK_ROWS, 1024]         # dp8d's b13 and b13h: chunk_bound on native_stage_local, [end, rows]
C22_BOUND = [22 * MAMBA_BLOCK_ROWS, 1024]         # dn's c22h: largest score buffer 170.5 MiB at 85008-87056; cf died at 194 at 97152
BOUND = {'sm': B13_BOUND, 'dec1': B13_BOUND, 'nw': B13_BOUND, 'nr': B13_BOUND}
HOLD_S = 600                                      # the engine's valve on a stuck lane, not a wait the arm expects: longer than an arm
NATIVE_HOLD = {'sm': 0, 'dec1': 0, 'nw': HOLD_S, 'nr': HOLD_S}
NATIVE_RESIDUE = {'sm': 0, 'dec1': 0, 'nw': 0, 'nr': 1}     # coverage-residue: on the stage command, needs the hold; the one axis of a pair
ARM_GENERATIONS = _c4.ARM_GENERATIONS
ONE_STREAM_WINDOW_S = _c4.ONE_STREAM_WINDOW_S

# The replay and the ceiling stay compose-4's, so c4b reads across (lead, 5 Nov 07:49: the ceiling stays 135,000 here).
REPLAY_S = 90.0                                   # lean: about 30 min nominal for the run (compose5_gate)
# Fresh prompts 20-40k rows, not 60-131k: eight of dnr's at once are ~760k rows, more than a 90 s arm prefills on either lane, and the
# arm would be the burst alone. At these sizes the burst is ~240k rows and most of the arm is appends, the bulk of the turns. Every
# other draw is compose-4's; the comparison is within this run, nw against nr on the same seed (the hold's own sizes).
START_MIN, START_MAX = 20000, 40000
REPLAY = dict(_c4.REPLAY, DURATION=REPLAY_S, WARMUP=0.0, START_MIN=START_MIN, START_MAX=START_MAX)
REPLAY_SEEDS = {index: index + 1 for index in BAND1_INDEXES}      # index 0 is c4b's and compose-5's seed; both arms of a pair get the same draws


def replay_config(row):
    """This row's replay: compose-4's, with its index's seed."""
    return dict(REPLAY, SEED=REPLAY_SEEDS[row['index']])


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
        # compose-5's rows less the cap (the module's note); the residue is the one axis between nw and nr.
        rows.append(dict(label=label(sel, 2, cohort, index), selection=selection(sel, 2), part=row_part, c1=sel, kind=KIND[sel], attn=True, lazy_recv=True, decode_merge='chunk', inwait=True, next_levers=True,
                         prepare_slices=True, score_warm=False, score_warm_rank0=False, allocator_flush=True, cow_fast=True, reserved_cap=False, cow_host_filter=True, cow_verify=True, exact_successor=True,
                         ple_private=PLE_PRIVATE[sel], ple_witness=PLE_WITNESS[sel], overlap=True, defer_bind=True, c0='on', stage_local=True, size=2, cohort=cohort, index=index, band=band_of_index(index),
                         extra=False, entry_limit=ENTRY_LIMIT[row_part][0], yield_on=YIELD[sel], plan=PLANS[PLAN[sel]], generations=generations_of(sel), chunk_bound=list(BOUND[sel]),
                         native_hold=NATIVE_HOLD[sel], native_residue=NATIVE_RESIDUE[sel]))
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


def band1_glob():
    return '[%s]' % ''.join(str(i) for i in BAND1_INDEXES)


def check():
    names = labels()
    assert names[:2] == ['sm.g2-live-0', 'dec1.g2-live-0'] and len(names) == 2 + 2 * PAIRS and names[0] == COLD_ARM and labels('tail') == []
    assert set(KIND) == set(YIELD) == set(PLE_PRIVATE) == set(PLE_WITNESS) == set(PLAN) == set(NATIVE_HOLD) == set(NATIVE_RESIDUE) == set(SELECTIONS)
    assert REPLAY['START_MAX'] + PROMPT_SLACK_TOKENS < MAX_PROMPT_TOKENS, 'a fresh prompt fits under the ceiling'
    rows = by_label()
    for index in REPLAY_SEEDS:
        on, off = rows['nr.g2-live-%d' % index], rows['nw.g2-live-%d' % index]
        assert sorted(k for k in on if on[k] != off[k]) == ['c1', 'label', 'native_residue', 'selection'], 'each pair differs in the residue alone'
        assert on['chunk_bound'] == off['chunk_bound'] == B13_BOUND and on['native_hold'] == off['native_hold'] == HOLD_S
        assert on['native_residue'] == 1 and off['native_residue'] == 0
        assert not on['yield_on'] and not off['yield_on'] and not on['ple_private'] and not off['ple_private'], "c4b's ctl"
    order = [r['label'] for r in rows.values() if r['kind'] == 'replay']
    assert order[:4] == ['nw.g2-live-0', 'nr.g2-live-0', 'nr.g2-live-1', 'nw.g2-live-1'] and len(order) == 2 * PAIRS, 'the order alternates; a stop on the first nr keeps the first nw'
    assert all(r['native_hold'] == 0 for r in rows.values() if r['kind'] != 'replay'), 'the hold on the replay arms alone'
    assert all(r['native_residue'] == (r['c1'] == 'nr') for r in rows.values()), 'the residue on nr alone'
    assert len(set(REPLAY_SEEDS.values())) == PAIRS and replay_config(rows[order[0]])['SEED'] == REPLAY['SEED'] and all(index < 10 for index in REPLAY_SEEDS), "one seed a pair, index 0 c4b's, one digit in a label"
    assert all(not r['score_warm'] and not r['reserved_cap'] and r['exact_successor'] and r['allocator_flush'] for r in rows.values()), "5d's full less the cap and the warm block on every arm"
    assert not any(r['ple_witness'] for r in rows.values()), 'no arm carries the witness'
    assert all(r['cow_host_filter'] and r['cow_fast'] for r in rows.values()), 'hf on every arm'
    assert B13_BOUND == [52624, 1024] and len(planned_chunks(65445, CHUNK_TOKENS, B13_BOUND)) == 40, "dp8d's 40 chunks at band 1"
    assert C22_BOUND == [89056, 1024] and len(planned_chunks(65445, CHUNK_TOKENS, C22_BOUND)) == 34 and len(planned_chunks(131094, CHUNK_TOKENS, C22_BOUND)) == 87, "dn's 34 and 87 chunks"
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
