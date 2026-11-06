"""q18a-compose-soak-fix arm table: the one declared table (coverage-soak, 6 Nov 2026; tmp/a1-trunk/READY-coverage-soak-fix.md). Change it here and nowhere else.

The lead (6 Nov 03:55): the residue build met coverage on 90 s arms of 20-40k fresh prompts; the operator's goal is an engine that takes 100 %
of calls and works all the time. This is a lean soak on the real turn mix: dnr's closed-loop agent replay (compose-4's REPLAY: 8 agents,
fresh 60-131k, appends 500-5000 with cache hits, replies 200-1000, the 135k ceiling rolling a session), engine bytes as build-3 of
q18a-compose-residue, unchanged. Forked from claude/coverage-residue at f4a2e996. Discovery mode. One launch, connections 32 and
FLASH_S5_CHUNK_TOKENS=2048 for the whole run. Per arm, by typed control at entry:
  sm    the a1 live arm's shape, the exact-path smoke, b13; its failure stops the table
  dec1  one standing stream, decode only, as compose-5
  nr    b13h under the replay, the hold (600) and the residue on: the candidate
  ny    b13h under the replay as dnr ran it: no hold, no residue (stock-mixed serving on the same mix, the control)
Order: sm | decode-am1 | dec1 | nr ny (seed 1, index 0) | ny nr (seed 2, index 1) | the grade: ABBA.
Every arm counts its prefill rows by path and site (the engine's coverage counter); any stock prefill row on nr is a FAIL in the grade, the
one-row handback graded as the lead decided (6 Nov 02:00). Each replay arm is REPLAY_S of traffic, every agent starting at once, then up to
the replay's FINISH_BOUND_S for the open turns (a turn still open then is a hang, named). The reserved cap and the warm block are off on
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
PAIRS = 2                                          # lead, 6 Nov 03:55: ABBA nr ny ny nr
BAND1_INDEXES = tuple(range(PAIRS))                # a replay has no band; sm, the one live arm, is index 0
BAND_SIZE = _c4.BAND_SIZE
SMOKE_ROW = ('core', 'sm', 'live', 0)
ARM_TABLE = (SMOKE_ROW, ('core', 'dec1', 'live', 0)) + tuple(('core', sel, 'live', index) for index in BAND1_INDEXES for sel in (('nr', 'ny') if index % 2 == 0 else ('ny', 'nr')))
DECODE_ARMS = _inwait.DECODE_ARMS
REFERENCE_ARM = _inwait.REFERENCE_ARM
FIXED_REFERENCE_TOK_S = _inwait.FIXED_REFERENCE_TOK_S
PARTS = _inwait.PARTS
ENTRY_LIMIT = _inwait.ENTRY_LIMIT
GRADED = _inwait.GRADED
SELECTIONS = ('sm', 'dec1', 'ny', 'nr')
KIND = {'sm': 'live-arm', 'dec1': 'one-stream', 'ny': 'replay', 'nr': 'replay'}
YIELD = {'sm': True, 'dec1': True, 'ny': False, 'nr': False}
PLE_PRIVATE = {'sm': True, 'dec1': True, 'ny': False, 'nr': False}
PLE_WITNESS = dict.fromkeys(SELECTIONS, False)
MATCHED_OFF = {'nr': 'ny'}
OFF, ON = 'ny', 'nr'
LAUNCH = _a1.LAUNCH
LAUNCH_ENV = _a1.LAUNCH_ENV
CONNECTIONS_ENV = _c4.CONNECTIONS_ENV
PLE_ENV = _c4.PLE_ENV
PLANS = _c4.PLANS
PLAN = {'sm': 'pair', 'dec1': 'none', 'ny': 'none', 'nr': 'none'}
CHUNK_TOKENS = 2048
CHUNK_ENV = 'FLASH_S5_CHUNK_TOKENS=%d' % CHUNK_TOKENS
MAMBA_BLOCK_ROWS, HASH_BLOCK_ROWS = 4048, 176     # read from this model's chunks on the box (dp5): ends 4048 x k, last stop prompt-1 // 176 x 176
B13_BOUND = [13 * MAMBA_BLOCK_ROWS, 1024]         # dp8d's b13 and b13h: chunk_bound on native_stage_local, [end, rows]
C22_BOUND = [22 * MAMBA_BLOCK_ROWS, 1024]         # dn's c22h: largest score buffer 170.5 MiB at 85008-87056; cf died at 194 at 97152
BOUND = {'sm': B13_BOUND, 'dec1': B13_BOUND, 'ny': B13_BOUND, 'nr': B13_BOUND}
HOLD_S = 600                                      # the engine's valve on a stuck lane, not a wait the arm expects: longer than an arm
NATIVE_HOLD = {'sm': 0, 'dec1': 0, 'ny': 0, 'nr': HOLD_S}
NATIVE_RESIDUE = {'sm': 0, 'dec1': 0, 'ny': 0, 'nr': 1}     # the residue needs the hold; ny has neither, as dnr ran it
ARM_GENERATIONS = _c4.ARM_GENERATIONS
ONE_STREAM_WINDOW_S = _c4.ONE_STREAM_WINDOW_S

# The replay is dnr's (compose-4's REPLAY: its sizes, draws, ceiling) with the soak's own length and no warmup cut: every turn is graded.
REPLAY_S = 480.0                                  # 8 min of traffic an arm (lead 03:55: about 10; the 07:15 end sets it, compose5_gate)
REPLAY = dict(_c4.REPLAY, DURATION=REPLAY_S, WARMUP=0.0)
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
        on, off = rows['nr.g2-live-%d' % index], rows['ny.g2-live-%d' % index]
        assert sorted(k for k in on if on[k] != off[k]) == ['c1', 'label', 'native_hold', 'native_residue', 'selection'], 'each pair differs in the hold and the residue alone'
        assert on['chunk_bound'] == off['chunk_bound'] == B13_BOUND and on['native_hold'] == HOLD_S and off['native_hold'] == 0
        assert on['native_residue'] == 1 and off['native_residue'] == 0
        assert not on['yield_on'] and not off['yield_on'] and not on['ple_private'] and not off['ple_private'], "c4b's ctl"
    order = [r['label'] for r in rows.values() if r['kind'] == 'replay']
    assert order == ['nr.g2-live-0', 'ny.g2-live-0', 'ny.g2-live-1', 'nr.g2-live-1'], 'ABBA: nr ny ny nr'
    assert all(r['native_hold'] == (HOLD_S if r['c1'] == 'nr' else 0) for r in rows.values()), 'the hold on nr alone'
    assert {k: REPLAY[k] for k in _c4.REPLAY if k not in ('DURATION', 'WARMUP')} == {k: v for k, v in _c4.REPLAY.items() if k not in ('DURATION', 'WARMUP')}, "dnr's turn mix"
    assert REPLAY['DURATION'] == REPLAY_S and REPLAY['WARMUP'] == 0.0
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
