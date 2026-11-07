"""q18a-parity-na2 arm table: the one declared table (parity-na2, 7 Nov 2026; tmp/decode-phase/READY-parity-na2.md). Change it here and nowhere else.

parity-na2 (lead, 7 Nov 15:07): a lean repeat of q18a-parity-o3's na against nt and sp (n1: na 753.8 turns/h, nt 727.5, sp 956.2). The same
engine, launch and rows; the pair arms pf po pg and the replay arm no are out of the table (O3 stays installed by FLASH_S5_O3=1 and every arm
sends o3_split False). Order: sm | decode-am1 | dec1 | na nt sp (seed 1) | sp nt na (seed 2). O3_ARM_TABLE keeps q18a-parity-o3's order for
the O3 grade's own tests. The q18a-parity-o3 text follows.

q18a-parity-o3 arm table (lane-o3, 7 Nov 2026; tmp/decode-phase/READY-parity-o3.md).

lane-o3 (lead, 7 Nov 10:24, DESIGN-multi-prompt-lane O3: stay split across runs): parity-2's engine, launch and levers with O3 installed
for the whole run (FLASH_S5_O3=1: the policy's capability and the workers' handback classification) and selected per arm by
``o3_split`` on native_stage_local (acknowledged; a command without it turns it off). Every row is parity-2's with its own o3_split:
  pf, po, pg  the design's first box test (kind o3-pair): O3_PAIR_DECODERS standing decoders live and O3_PAIR_NATIVES native requests
              (the band's fresh prompt, one salt each) sent at once, so the lane takes them back to back; greedy, top-20 logprobs over
              O3_PAIR_TOKENS positions. nt's levers; po O3 on, pf and pg off, in that order: pf|pg is the control's self-disagreement
              (numerical gate rule v4's shape, off vs off on the same prompts), po against both is the stop rule.
  no          nt with O3 on, under the replay
  na          nt with ahead_nosync (parity-ahead 9daa2ac3, lead 7 Nov ~12:10: the prepare-ahead's successor copies without the stream sync,
              installed for the whole run by FLASH_S5_AHEAD_NOSYNC=1, AHEAD_NOSYNC_ENV; selected per arm), under the replay
  nt          parity-2's, O3 off
  sp          parity-base's "stock, plain" (lead, 7 Nov 11:55: sp replaces ny, the acceptance bar): ny's row with the lane off for the whole
              arm (the box's keep_lease after the release: the policy declines 'gate-manual-owned'), attn-meta, lazy receive and decode merge off.
              ny stays in the selection tables as sp's reference row only; it is not an arm.
Order: sm | decode-am1 | dec1 | pf po pg | no na nt sp (seed 1) | sp nt na no (seed 2): each pair of the four replay selections ABBA; one
effect per arm against nt (no: O3, na: ahead_nosync, sp: the lane off).
The parity-2 text follows.

q18a-compose-throughput-p2 arm table (parity-2 build-1, 7 Nov 2026; tmp/decode-phase/READY-parity-2.md).

parity-2 (lead, 6 Nov ~23:35; tmp/decode-phase/READING-lane-pace.md): build-2b24c64's table, engine and launch unchanged, with fix A on nt:
the terminal successor (``terminal_successor`` on native_stage_local, per arm; it needs the exact successor). On c64 the terminal chunk was
refused as a successor on 193 of 207 second-to-last chunks (``successor-may-be-terminal``), edge 267-324 ms against ~40. nt sends True, every
other row sends False; the grade fails an arm whose reply does not echo its value. ABBA nt ny ny nt, seeds 1/2, 480 s, as c64. Build-2b24c64's
text follows.

q18a-compose-throughput-b24c64 (coverage-throughput build-2b24c64, 6 Nov 2026; tmp/decode-phase/READY-coverage-throughput-b2b24c64.md).

coverage-throughput build-2b24c64 (lead, 6 Nov ~17:12): build-2b24's table with build-2b's full ABBA back (nt ny ny nt, 480 s, the same seeds
and positions) and the indexer logits cap VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 for the whole run (LOGITS_ENV, both preflights assert it).
b24 on nt died at 16:52 on a 190 MiB logits ask (fragmentation, READING-b2b24-oom.md); the cap keeps every logits slice <= 64 MiB. The
grade fails any allocator retry over 64 MiB, per rank. nt keeps b24, ny b13. Build-2b24's text follows.

coverage-throughput build-2b24 (lead, 6 Nov ~15:35): build-2b's engine and table unchanged, except nt's chunk_bound: b24's [97152, 1024]
(2048-row native chunks up to 97152, 1024 past; q18a-o4-b24 ran it clean, OOM 0, 15:01-15:23). ny keeps b13 [52624, 1024]. One pair, nt then ny,
480 s each (lead, ~15:40: the slot ends 17:20), not ABBA. Build-2's text follows.

coverage-throughput build-2 (lead, 6 Nov): build-1's table and arms unchanged (ABBA nt ny ny nt), over an engine with (a) prepare-ahead,
the safe variant (a later chunk's PREPARE slices 1-2 staged after the live chunk's state_finalize, on a second builder set; launch setting
FLASH_S5_PREPARE_AHEAD=1 for the whole run, used only by native chunks, so ny never stages), (b) K2 (the next run selected at the scheduled
handback, no cooldown) and (c) the geometry tail at the prompt's last prefix unit. Chunk geometry and the indexer logits cap unchanged.
Build-1's text follows.

coverage-throughput (lead, 6 Nov ~09:00): the soak-fix table with its candidate renamed nt: nr (the hold and the residue) plus the append yield,
over an engine whose exact successor plans over the admission's view (the soak's in-run pause) and whose yield plans as admit() does.
ABBA nt ny ny nt; ny unchanged. The soak-fix text follows.


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
PAIRS = 2                                          # build-2b24c64 (lead, 6 Nov ~17:12): build-2b's ABBA nt ny ny nt again (build-2b24 ran 1)
# coverage-soak-fix build-2 (lead, 6 Nov 08:50: name and fix the Worker_PP RSS owner): the CoW byte check at release
# (lane_chunk.cow_host_verify) reads each host CoW pair's src and dst page prefix through the registered QSA host pool on the
# CPU, which maps those pool pages into the worker: the soak's worker RSS growth on nr (native appends only). Off on every arm,
# as the inwait control arm 'cn' ran it; the pair stays matched. tmp/a1-trunk/READY-coverage-soak-fix-2.md.
COW_VERIFY = False
BAND1_INDEXES = tuple(range(PAIRS))                # a replay has no band; sm, the one live arm, is index 0
BAND_SIZE = _c4.BAND_SIZE
SMOKE_ROW = ('core', 'sm', 'live', 0)
O3_PAIR = ('pf', 'po', 'pg')                       # lane-o3 arm 1: off, on, off
O3_REPLAY_ORDER = ('no', 'na', 'nt', 'sp')         # lane-o3 (+ na): index 0 in this order, index 1 reversed
O3_ARM_TABLE = (SMOKE_ROW, ('core', 'dec1', 'live', 0)) + tuple(('core', sel, 'live', 0) for sel in O3_PAIR) + tuple(('core', sel, 'live', index) for index in BAND1_INDEXES for sel in (O3_REPLAY_ORDER if index % 2 == 0 else O3_REPLAY_ORDER[::-1]))
REPLAY_ORDER = ('na', 'nt', 'sp')                  # parity-na2: no pair arms and no 'no'; index 0 in this order, index 1 reversed
ARM_TABLE = (SMOKE_ROW, ('core', 'dec1', 'live', 0)) + tuple(('core', sel, 'live', index) for index in BAND1_INDEXES for sel in (REPLAY_ORDER if index % 2 == 0 else REPLAY_ORDER[::-1]))
DECODE_ARMS = _inwait.DECODE_ARMS
REFERENCE_ARM = _inwait.REFERENCE_ARM
FIXED_REFERENCE_TOK_S = _inwait.FIXED_REFERENCE_TOK_S
PARTS = _inwait.PARTS
ENTRY_LIMIT = _inwait.ENTRY_LIMIT
GRADED = _inwait.GRADED
SELECTIONS = ('sm', 'dec1', 'ny', 'nt', 'no', 'na', 'sp') + O3_PAIR
REFERENCE_ONLY = ('ny',)                           # lane-o3: sp's reference row, not an arm
# lane-o3: no, na and the pair arms are nt's row (_AS_NT) but for o3_split (no, po) or ahead_nosync (na)
_AS_NT = ('no', 'na') + O3_PAIR


def _nt(table):
    """no and the pair arms take nt's value; sp takes ny's (parity-base: sp is ny's row but the lane off)."""
    return dict(table, sp=table['ny'], **{sel: table['nt'] for sel in _AS_NT})


KIND = dict(_nt({'sm': 'live-arm', 'dec1': 'one-stream', 'ny': 'replay', 'nt': 'replay'}), **dict.fromkeys(O3_PAIR, 'o3-pair'))
YIELD = _nt({'sm': True, 'dec1': True, 'ny': False, 'nt': True})     # coverage-throughput: lever 1, the yield on the candidate
PLE_PRIVATE = _nt({'sm': True, 'dec1': True, 'ny': False, 'nt': False})
PLE_WITNESS = dict.fromkeys(SELECTIONS, False)
MATCHED_OFF = {'nt': 'sp', 'no': 'nt', 'na': 'nt', 'po': 'pf'}
OFF, ON = 'sp', 'nt'
# parity-base: sp, the parity baseline. NATIVE_OFF keeps the manual lease for the whole arm (the box's keep_lease, after the release); ATTN,
# LAZY_RECV and DECODE_MERGE are the row's attn-meta selection and the stage command's two decode fields (every other arm: on, on, 'chunk').
BASELINE = 'sp'
NATIVE_OFF = dict(dict.fromkeys(SELECTIONS, False), sp=True)
ATTN = dict(dict.fromkeys(SELECTIONS, True), sp=False)
LAZY_RECV = dict(dict.fromkeys(SELECTIONS, True), sp=False)
DECODE_MERGE = dict(dict.fromkeys(SELECTIONS, 'chunk'), sp='off')
ONS = ('nt', 'no', 'na')                           # lane-o3: the candidate's rules (open requests, RSS) on every nt row
# coverage-throughput: the yield takes an append of up to YIELD_MAX_NEW new rows. a1's 2048 covered about one append in seven on this
# mix (native appends compute 1.3-6.6k rows, the soak's mean 3.9k); the other three bounds stay a1's.
YIELD_MAX_NEW = 8192
LAUNCH = dict(_a1.LAUNCH, max_new=YIELD_MAX_NEW)
LAUNCH_ENV = dict(_a1.LAUNCH_ENV, FLASH_S5_APPEND_YIELD_MAX_NEW=YIELD_MAX_NEW)
CONNECTIONS_ENV = _c4.CONNECTIONS_ENV
PLE_ENV = _c4.PLE_ENV
PLANS = _c4.PLANS
PLAN = _nt({'sm': 'pair', 'dec1': 'none', 'ny': 'none', 'nt': 'none'})
CHUNK_TOKENS = 2048
CHUNK_ENV = 'FLASH_S5_CHUNK_TOKENS=%d' % CHUNK_TOKENS
LOGITS_CAP_MB = 64                                 # build-2b24c64: the indexer splits a chunk's rows so one score block stays under this (stock 512)
LOGITS_ENV = 'VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=%d' % LOGITS_CAP_MB
PREPARE_AHEAD_ENV = 'FLASH_S5_PREPARE_AHEAD=1'     # build-2 (a): installed for the whole run; needs FLASH_S5_STAGE_OVERLAP=1 (the settings carry it)
MAMBA_BLOCK_ROWS, HASH_BLOCK_ROWS = 4048, 176     # read from this model's chunks on the box (dp5): ends 4048 x k, last stop prompt-1 // 176 x 176
B13_BOUND = [13 * MAMBA_BLOCK_ROWS, 1024]         # dp8d's b13 and b13h: chunk_bound on native_stage_local, [end, rows]
B24_BOUND = [24 * MAMBA_BLOCK_ROWS, 1024]         # build-2b24: o4's b24 (q18a-o4-b24), 2048 rows up to 97152, then 1024
C22_BOUND = [22 * MAMBA_BLOCK_ROWS, 1024]         # dn's c22h: largest score buffer 170.5 MiB at 85008-87056; cf died at 194 at 97152
BOUND = _nt({'sm': B13_BOUND, 'dec1': B13_BOUND, 'ny': B13_BOUND, 'nt': B24_BOUND})     # build-2b24: nt alone on b24
HOLD_S = 600                                      # the engine's valve on a stuck lane, not a wait the arm expects: longer than an arm
NATIVE_HOLD = _nt({'sm': 0, 'dec1': 0, 'ny': 0, 'nt': HOLD_S})
NATIVE_RESIDUE = _nt({'sm': 0, 'dec1': 0, 'ny': 0, 'nt': 1})     # the residue needs the hold; ny has neither, as dnr ran it
TERMINAL_SUCCESSOR = _nt({'sm': False, 'dec1': False, 'ny': False, 'nt': True})     # parity-2 A: nt alone; every row sends its value
# lane-o3: on no and po alone; every row sends its value. FLASH_S5_O3=1 is in EXTRA_ENV for the whole run (O3_ENV; both preflights assert it).
O3_SPLIT = dict(dict.fromkeys(SELECTIONS, False), no=True, po=True)
O3_ENV = 'FLASH_S5_O3=1'
# parity-ahead: on na alone; every row sends its value. FLASH_S5_AHEAD_NOSYNC=1 for the whole run (needs FLASH_S5_PREPARE_AHEAD=1; both
# preflights assert it); True needs prepare-ahead, which every row has.
AHEAD_NOSYNC = dict(dict.fromkeys(SELECTIONS, False), na=True)
AHEAD_NOSYNC_ENV = 'FLASH_S5_AHEAD_NOSYNC=1'
O3_PAIR_DECODERS = 2                                # the design's first box test: 2 decoders live
O3_PAIR_NATIVES = 3                                 # and 3 back-to-back native requests (each >= MIN_NATIVE_ROWS new rows)
O3_PAIR_TOKENS = 16                                 # greedy positions captured per native, top-20 each; position 0 is the handback's token
O3_LOGPROBS = 20
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
    if KIND[sel] == 'o3-pair':
        return O3_PAIR_DECODERS + O3_PAIR_NATIVES
    return 1 if KIND[sel] == 'one-stream' else None


def arms(part=None):
    parts = (part,) if isinstance(part, str) else part
    rows = []
    for row_part, sel, cohort, index in ARM_TABLE:
        if parts is not None and row_part not in parts:
            continue
        # compose-5's rows less the cap (the module's note); the residue is the one axis between nw and nr.
        rows.append(dict(label=label(sel, 2, cohort, index), selection=selection(sel, 2), part=row_part, c1=sel, kind=KIND[sel], attn=ATTN[sel], lazy_recv=LAZY_RECV[sel], decode_merge=DECODE_MERGE[sel], inwait=True, next_levers=True,
                         prepare_slices=True, score_warm=False, score_warm_rank0=False, allocator_flush=True, cow_fast=True, reserved_cap=False, cow_host_filter=True, cow_verify=COW_VERIFY, exact_successor=True,
                         ple_private=PLE_PRIVATE[sel], ple_witness=PLE_WITNESS[sel], overlap=True, defer_bind=True, c0='on', stage_local=True, size=2, cohort=cohort, index=index, band=band_of_index(index),
                         extra=False, entry_limit=ENTRY_LIMIT[row_part][0], yield_on=YIELD[sel], plan=PLANS[PLAN[sel]], generations=generations_of(sel), chunk_bound=list(BOUND[sel]),
                         native_hold=NATIVE_HOLD[sel], native_residue=NATIVE_RESIDUE[sel],
                         terminal_successor=TERMINAL_SUCCESSOR[sel], o3_split=O3_SPLIT[sel], ahead_nosync=AHEAD_NOSYNC[sel], native_off=NATIVE_OFF[sel]))
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
    o3 = ARM_TABLE == O3_ARM_TABLE                 # q18a-parity-o3's table (the O3 grade's tests select it); else parity-na2's
    head = ['sm.g2-live-0', 'dec1.g2-live-0'] + (['pf.g2-live-0', 'po.g2-live-0', 'pg.g2-live-0'] if o3 else [])
    replay = O3_REPLAY_ORDER if o3 else REPLAY_ORDER
    assert names[:len(head)] == head and len(names) == len(head) + len(replay) * PAIRS and names[0] == COLD_ARM and labels('tail') == []
    assert set(KIND) == set(YIELD) == set(PLE_PRIVATE) == set(PLE_WITNESS) == set(PLAN) == set(NATIVE_HOLD) == set(NATIVE_RESIDUE) == set(TERMINAL_SUCCESSOR) == set(O3_SPLIT) == set(AHEAD_NOSYNC) == set(NATIVE_OFF) == set(ATTN) == set(LAZY_RECV) == set(DECODE_MERGE) == set(SELECTIONS)
    assert REPLAY['START_MAX'] + PROMPT_SLACK_TOKENS < MAX_PROMPT_TOKENS, 'a fresh prompt fits under the ceiling'
    rows = by_label()
    for index in REPLAY_SEEDS:
        on, off = rows['nt.g2-live-%d' % index], rows['sp.g2-live-%d' % index]
        assert sorted(k for k in on if on[k] != off[k]) == ['attn', 'c1', 'chunk_bound', 'decode_merge', 'label', 'lazy_recv', 'native_hold', 'native_off', 'native_residue', 'selection', 'terminal_successor', 'yield_on'], 'nt against sp: the levers and the lane'
        assert on['chunk_bound'] == B24_BOUND and off['chunk_bound'] == B13_BOUND and on['native_hold'] == HOLD_S and off['native_hold'] == 0
        assert on['native_residue'] == 1 and off['native_residue'] == 0
        assert on['terminal_successor'] is True and off['terminal_successor'] is False
        assert on['yield_on'] and not off['yield_on'] and not on['ple_private'] and not off['ple_private'], "the yield on nt alone; c4b's ple off"
        ny = {name: table[sel] for name, table, sel in ((k, globals()[v], 'ny') for k, v in (('yield_on', 'YIELD'), ('ple_private', 'PLE_PRIVATE'), ('chunk_bound', 'BOUND'),
              ('native_hold', 'NATIVE_HOLD'), ('native_residue', 'NATIVE_RESIDUE'), ('terminal_successor', 'TERMINAL_SUCCESSOR'), ('o3_split', 'O3_SPLIT'), ('ahead_nosync', 'AHEAD_NOSYNC'), ('kind', 'KIND')))}
        assert {k: off[k] for k in ny} == dict(ny, chunk_bound=list(ny['chunk_bound'])), "sp is ny's row but the lane off (parity-base)"
        assert off['native_off'] and not off['attn'] and not off['lazy_recv'] and off['decode_merge'] == 'off'
        if o3:
            no = rows['no.g2-live-%d' % index]
            assert sorted(k for k in no if no[k] != on[k]) == ['c1', 'label', 'o3_split', 'selection'] and no['o3_split'] is True and on['o3_split'] is False, 'no is nt but for O3'
        na = rows['na.g2-live-%d' % index]
        assert sorted(k for k in na if na[k] != on[k]) == ['ahead_nosync', 'c1', 'label', 'selection'] and na['ahead_nosync'] is True and on['ahead_nosync'] is False, 'na is nt but for ahead_nosync'
    for sel in (O3_PAIR if o3 else ()):
        pair, nt0 = rows['%s.g2-live-0' % sel], rows['nt.g2-live-0']
        assert sorted(k for k in pair if pair[k] != nt0[k]) == sorted(['c1', 'generations', 'kind', 'label', 'selection'] + (['o3_split'] if sel == 'po' else [])), 'a pair arm is nt but for its kind and O3'
    assert not o3 or [rows['%s.g2-live-0' % sel]['o3_split'] for sel in O3_PAIR] == [False, True, False], 'off, on, off: the control pair brackets the candidate'
    assert o3 or not any(r['o3_split'] or r['kind'] == 'o3-pair' for r in rows.values()), 'parity-na2: O3 installed, selected on no arm'
    assert all(r['generations'] == O3_PAIR_DECODERS + O3_PAIR_NATIVES for r in rows.values() if r['kind'] == 'o3-pair')
    assert all(r['o3_split'] is (r['c1'] in ('no', 'po')) for r in rows.values()), 'O3 on no and po alone'
    assert all(r['ahead_nosync'] is (r['c1'] == 'na') for r in rows.values()), 'ahead_nosync on na alone'
    assert AHEAD_NOSYNC_ENV == 'FLASH_S5_AHEAD_NOSYNC=1' and PREPARE_AHEAD_ENV == 'FLASH_S5_PREPARE_AHEAD=1', 'the nosync needs prepare-ahead, installed for the whole run'
    assert O3_ENV == 'FLASH_S5_O3=1' and O3_PAIR_DECODERS == 2 and O3_PAIR_NATIVES == 3 and O3_LOGPROBS == 20
    order = [r['label'] for r in rows.values() if r['kind'] == 'replay']
    assert order == (['no.g2-live-0', 'na.g2-live-0', 'nt.g2-live-0', 'sp.g2-live-0', 'sp.g2-live-1', 'nt.g2-live-1', 'na.g2-live-1', 'no.g2-live-1'] if o3 else
                     ['na.g2-live-0', 'nt.g2-live-0', 'sp.g2-live-0', 'sp.g2-live-1', 'nt.g2-live-1', 'na.g2-live-1']), 'each pair ABBA'
    assert [r['label'] for r in rows.values() if r['native_off']] == ['sp.g2-live-0', 'sp.g2-live-1'] and BASELINE == 'sp', 'the lane off on sp alone'
    assert all(r['attn'] and r['lazy_recv'] and r['decode_merge'] == 'chunk' for r in rows.values() if not r['native_off']), "every other arm's decode features as parity-2"
    assert not any(r['c1'] in REFERENCE_ONLY for r in rows.values()), 'ny is a reference row, not an arm'
    assert all(r['native_hold'] == (HOLD_S if r['c1'] in ('nt',) + _AS_NT else 0) for r in rows.values()), 'the hold on the nt rows alone'
    assert {k: REPLAY[k] for k in _c4.REPLAY if k not in ('DURATION', 'WARMUP')} == {k: v for k, v in _c4.REPLAY.items() if k not in ('DURATION', 'WARMUP')}, "dnr's turn mix"
    assert REPLAY['DURATION'] == REPLAY_S and REPLAY['WARMUP'] == 0.0
    assert all(r['native_residue'] == (r['c1'] in ('nt',) + _AS_NT) for r in rows.values()), 'the residue on the nt rows alone'
    assert all(r['terminal_successor'] is (r['c1'] in ('nt',) + _AS_NT) for r in rows.values()), 'the terminal successor on the nt rows alone, needing the exact successor'
    assert len(set(REPLAY_SEEDS.values())) == PAIRS and replay_config(rows[order[0]])['SEED'] == REPLAY['SEED'] and all(index < 10 for index in REPLAY_SEEDS), "one seed a pair, index 0 c4b's, one digit in a label"
    assert all(not r['score_warm'] and not r['reserved_cap'] and r['exact_successor'] and r['allocator_flush'] for r in rows.values()), "5d's full less the cap and the warm block on every arm"
    assert not any(r['ple_witness'] for r in rows.values()), 'no arm carries the witness'
    assert all(r['cow_host_filter'] and r['cow_fast'] for r in rows.values()), 'hf on every arm'
    assert LAUNCH == dict(_a1.LAUNCH, max_new=8192) and _a1.LAUNCH['max_new'] == 2048, "the yield's new-row bound raised here alone"
    assert PREPARE_AHEAD_ENV == 'FLASH_S5_PREPARE_AHEAD=1' and CHUNK_TOKENS == 2048 and CHUNK_ENV == 'FLASH_S5_CHUNK_TOKENS=2048', 'build-2: prepare-ahead on, chunk geometry unchanged'
    assert B13_BOUND == [52624, 1024] and len(planned_chunks(65445, CHUNK_TOKENS, B13_BOUND)) == 40, "dp8d's 40 chunks at band 1"
    assert B24_BOUND == [97152, 1024] and len(planned_chunks(131094, CHUNK_TOKENS, B24_BOUND)) == 83, "b24's 83 chunks for the 131,094-token fresh (READY-o4-b24)"
    assert LOGITS_CAP_MB == 64 and LOGITS_ENV == 'VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64', 'build-2b24c64: the logits cap for the whole run'
    assert all(r['chunk_bound'] == (B24_BOUND if r['c1'] in ('nt',) + _AS_NT else B13_BOUND) for r in rows.values()), 'b24 on the nt rows alone'
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
