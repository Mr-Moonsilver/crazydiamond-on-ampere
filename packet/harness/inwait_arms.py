"""q18a-inwait arm table: the one declared table (DESIGN-5d, 4 Nov, build-5d). Change it here and nowhere else.

Build-5d (results/hub/inwait/DESIGN-5d.md): the stack that goes forward, at band 1 and band 2, with the cap fix, and cf with the byte check off for the fair hf read.
The engine is build-5c's plus these behind one default-off flag, FLASH_S5_INWAIT, installed for the whole run. Selected per arm by typed control while paused idle FULL:
  hf    the base, no cap: cf, the host pass filtered to QSA targets
  full  hf, the exact successor plan, the reserved cap that keeps 512 MiB free and the warm score block on every rank
  cn    cf: the host pass unfiltered and the byte check at release off
  sm    full as the exact-path smoke: arm 1, cold, graded as any arm, no matched pair; if it does not engage, nothing else runs
Order: sm live | decode-am1 | round 0: hf full solo, hf cn full live | round 1: full cn hf live (reversed) | the core grade | hf then full live at band 2, last and a read.
Band 1's fresh and append requests carry top-20 logprobs (READING-box-5c).
"""
import alpha_next_arms as _alpha_next
import attn_meta_arms as _attn_meta

STANDING_DECODERS = _alpha_next.STANDING_DECODERS
STANDING_GREEDY = _alpha_next.STANDING_GREEDY
GREEDY_TEMPERATURE = _alpha_next.GREEDY_TEMPERATURE
STANDING_SAMPLED = _alpha_next.STANDING_SAMPLED
BAND1_ROUNDS = 2
BAND1_REQUIRED_ROUNDS = 2
BAND1_INDEXES = (0, 1)
BAND2_INDEX = 2                           # 131,094 tokens fresh; the selected config's pair
TAIL_INDEX = 0                            # no row uses it: the inherited code reads the name
BAND_SIZE = {'core': {1: 2, 2: 2}, 'tail': {1: 2, 2: 2}}
ROUND0 = (('hf', 'solo'), ('full', 'solo'), ('hf', 'live'), ('cn', 'live'), ('full', 'live'))
ROUND1 = (('full', 'live'), ('cn', 'live'), ('hf', 'live'))
SMOKE_ROW = ('core', 'sm', 'live', 0)
BAND2_ROWS = (('tail', 'hf', 'live', BAND2_INDEX), ('tail', 'full', 'live', BAND2_INDEX))    # after the core grade, a read: an OOM here costs no pair and does not change the grade
ARM_TABLE = ((SMOKE_ROW,) + tuple([('core', s, c, 0) for s, c in ROUND0] + [('core', s, c, 1) for s, c in ROUND1]) + BAND2_ROWS)
DECODE_ARMS = tuple(d for d in _attn_meta.DECODE_ARMS if d['attn'])
REFERENCE_ARM = 'decode-am1'              # this run's decode-only baseline; FIXED_REFERENCE below is the lead's fixed number
FIXED_REFERENCE_TOK_S = 452.8             # the lead's fixed reference (brief, 4 Nov); compose-2's own decode-am1 read 456.598
PARTS = ('core', 'tail')
ENTRY_LIMIT = {'core': ('LAST_ARM_ENTRY_S', 'last_arm_entry'), 'tail': ('LAST_TAIL_ENTRY_S', 'last_tail_entry')}
GRADED = ('core',)
SELECTIONS = ('sm', 'hf', 'full', 'cn')
OFF = 'hf'                                  # the matched off arm of every lever
ON = 'full'
ATTN = dict.fromkeys(SELECTIONS, True)
LAZY_RECV = dict.fromkeys(SELECTIONS, True)
DECODE_MERGE = dict.fromkeys(SELECTIONS, 'chunk')
INWAIT = dict.fromkeys(SELECTIONS, True)
NEXT = dict.fromkeys(SELECTIONS, True)      # early_successor on native_stage_local, slots_fast on attn_meta_select
SLICES = dict.fromkeys(SELECTIONS, True)    # prepare_slices on attn_meta_select
WARM = {'sm': True, 'hf': False, 'full': True, 'cn': False}    # score_warm on attn_meta_select
WARM_RANK0 = dict.fromkeys(SELECTIONS, False)    # score_warm_rank0 on attn_meta_select
FLUSH = dict.fromkeys(SELECTIONS, True)     # allocator_flush on attn_meta_select: every prefill arm, so the pairs start matched
COW_FAST = dict.fromkeys(SELECTIONS, True)    # cow_fast on attn_meta_select
RESERVED_CAP = {'sm': True, 'hf': False, 'full': True, 'cn': False}    # reserved_cap on attn_meta_select
HOST_FILTER = {'sm': True, 'hf': True, 'full': True, 'cn': False}    # cow_host_filter on attn_meta_select
COW_VERIFY = {'sm': True, 'hf': True, 'full': True, 'cn': False}    # cow_verify on attn_meta_select
EXACT = {'sm': True, 'hf': False, 'full': True, 'cn': False}    # exact_successor on native_stage_local
C1 = dict.fromkeys(SELECTIONS, True)
DEFER_BIND = dict.fromkeys(SELECTIONS, True)
STAGE_LOCAL = True
COHORTS = ('live', 'solo')
INDEXES = tuple(sorted(BAND1_INDEXES + (BAND2_INDEX,)))
SIZES = (1, 2, 4)
COLD_ARM = 'sm.g2-live-0'
SMOKE_ARM = COLD_ARM


MAMBA_BLOCK_ROWS, HASH_BLOCK_ROWS = 4048, 176     # read from this model's chunks on the box (dp5); expected_chunks is claude/ple-private-b's, for test_dp5_chunk


def expected_chunks(prompt_tokens, cap):
    """How many native chunks a fresh prompt runs in at this cap: the largest stock split at each position, as staged_chunk_plan finds it.

    At cap 2048 every 4048-row block takes at least two chunks, so a count equal to this one also proves each block's chunks are 2000-2048 rows: one chunk at 1024 makes it larger.
    """
    end = prompt_tokens - 1
    stop = end // HASH_BLOCK_ROWS * HASH_BLOCK_ROWS
    position, count = 0, 0
    while position < end:
        following = min(position + cap, end)
        if following < end and following // MAMBA_BLOCK_ROWS * MAMBA_BLOCK_ROWS > position:
            following = following // MAMBA_BLOCK_ROWS * MAMBA_BLOCK_ROWS
        if position < stop < following:
            following = stop
        position, count = following, count + 1
    return count


def band_of_index(index):
    return 1 if index in BAND1_INDEXES else 2 if index == BAND2_INDEX else None


def selection(sel, size):
    return '%s.g%d' % (sel, size)


def label(sel, size, cohort, index):
    return '%s-%s-%d' % (selection(sel, size), cohort, index)


def parse_selection(token):
    head, dot, tail = str(token).partition('.')
    if not dot or head not in ATTN or not tail.startswith('g') or not tail[1:].isdigit():
        return None
    return head, int(tail[1:])


def arms(part=None):
    parts = (part,) if isinstance(part, str) else part
    rows = []
    for row_part, sel, cohort, index in ARM_TABLE:
        if parts is not None and row_part not in parts:
            continue
        band = band_of_index(index)
        size = BAND_SIZE[row_part][band]
        rows.append(dict(label=label(sel, size, cohort, index), selection=selection(sel, size), part=row_part, c1=sel, attn=ATTN[sel], lazy_recv=LAZY_RECV[sel], decode_merge=DECODE_MERGE[sel],
                         inwait=INWAIT[sel], next_levers=NEXT[sel], prepare_slices=SLICES[sel], score_warm=WARM[sel], score_warm_rank0=WARM_RANK0[sel], allocator_flush=FLUSH[sel], cow_fast=COW_FAST[sel], reserved_cap=RESERVED_CAP[sel], cow_host_filter=HOST_FILTER[sel], cow_verify=COW_VERIFY[sel], exact_successor=EXACT[sel], overlap=C1[sel], defer_bind=DEFER_BIND[sel], c0='on', stage_local=STAGE_LOCAL, size=size, cohort=cohort,
                         index=index, band=band, extra=False, entry_limit=ENTRY_LIMIT[row_part][0]))
    return rows


def labels(part=None):
    return [row['label'] for row in arms(part)]


def by_label():
    return {row['label']: row for row in arms()}


def matched_off(row):
    """The ps1 arm at the same size, cohort and round; None for a ps1 arm, the smoke, and band 2, which has no off arm."""
    if row.get('part') not in GRADED or row.get('c1') in (OFF, 'sm'):
        return None
    want = label(OFF, row['size'], row['cohort'], row['index'])
    return want if want in labels(GRADED) else None


def matched_on(row):
    """c1's name (a C1-defer arm's C1-on partner): this table has no C1-on arm."""
    return None


def band1_glob():
    return '[%s]' % ''.join(str(i) for i in BAND1_INDEXES)


def check():
    assert set(BAND_SIZE) == set(PARTS) == set(ENTRY_LIMIT)
    assert BAND1_REQUIRED_ROUNDS == BAND1_ROUNDS == len(BAND1_INDEXES)
    names = labels()
    assert len(names) == len(set(names)) == 11, 'eleven distinct prefill arm labels'
    assert labels('tail') == ['hf.g2-live-2', 'full.g2-live-2'] and labels('core') == names[:-2], 'two tail arms, last, the stack last of all'
    assert names[0] == COLD_ARM, 'the cold arm is the first prefill arm'
    assert [d['label'] for d in DECODE_ARMS] == [REFERENCE_ARM] and [d['attn'] for d in DECODE_ARMS] == [True]
    assert ROUND1 == ROUND0[:1:-1], 'round 1 is round 0 live reversed'
    assert set(ATTN) == set(LAZY_RECV) == set(DECODE_MERGE) == set(INWAIT) == set(NEXT) == set(SLICES) == set(WARM) == set(WARM_RANK0) == set(FLUSH) == set(COW_FAST) == set(RESERVED_CAP) == set(HOST_FILTER) == set(EXACT) == set(C1) == set(DEFER_BIND) == set(SELECTIONS)
    for row in arms():
        assert row['overlap'] is True and row['defer_bind'] is True and row['stage_local'] is True, 'C1 defer in every arm'
        assert row['attn'] is True and row['lazy_recv'] is True and row['decode_merge'] == 'chunk', 'attn-meta, L1 and L2 in every arm'
        assert row['size'] == 2, 'group size 2 in every arm'
        assert row['allocator_flush'] is True, 'every prefill arm starts flushed'
        if row['c1'] not in (OFF, 'sm') and row['part'] in GRADED and row['band'] == 1:
            assert matched_off(row), 'on arm without its matched off arm: ' + row['label']
    for index in BAND1_INDEXES:
        assert {OFF, ON, 'cn'} <= {r['c1'] for r in arms() if r['index'] == index and r['cohort'] == 'live'}, ('live pair', index)
    return True


check()
