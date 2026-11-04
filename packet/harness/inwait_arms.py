"""q18a-inwait arm table: the one declared table (lead, 4 Nov 12:41 CET, build-4). Change it here and nowhere else.

Build-4 (results/hub/inwait/DESIGN-4.md): in-wait, the early successor, fast slot mappings and PREPARE in slices are on in every arm (build-3b's ps1).
The engine is build-3b's plus these levers behind one default-off flag, FLASH_S5_INWAIT, installed for the whole run. The only axis, selected per arm by typed control while paused, idle and FULL:
  ps1  build-3b's base
  cf   ps1 and the fast CoW copy at the residue chunk
  sm   cf as the exact-path smoke: arm 1, cold, graded as any arm, no matched pair; if it does not engage, nothing else runs
  cap  ps1 and a reserved cap that keeps 512 MiB free; flushes the allocator cache at its selection
Order in build-4c: sm live | decode-am1 | round 0: ps1 cf solo, ps1 cf live | round 1: cf ps1 live (reversed) | band 2: cf live, solo | the core grade | cap live.

Decode-phase build-3 (READING-box-dp2.md; lead, 4 Nov 15:4x CET): the same engine at another decode SM count, a launch setting (S5_REQUESTED_DECODE_SMS, one launch one value, DEFAULT_DECODE_SMS first).
Same labels as build-4c, whose cf arms at 44 are the control: sm live | decode-am1 | cf solo | cf live x2 | cfa live (cf with merge always, last). No ps1, band 2 or cap arm.

Decode-phase dp5 (DESIGN-dp5.md; lead, 4 Nov 19:3x CET): native chunks above 1024 rows, a launch setting (S5_REQUESTED_CHUNK_TOKENS, FLASH_S5_CHUNK_TOKENS on the engine, DEFAULT_CHUNK_TOKENS first), at 44 SMs. Control: build-4c's cf arms again.
sm live | decode-am1 | cf solo | cf live x2 | cf live band 2, last: memory is the likeliest failure and band 2 its worst case, so an OOM there still leaves band 1's numbers. The smoke stops the table unless the prompt ran in exactly expected_chunks chunks.
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
ROUND0 = (('ps1', 'solo'), ('cf', 'solo'), ('ps1', 'live'), ('cf', 'live'))
ROUND1 = (('cf', 'live'), ('ps1', 'live'))
SMOKE_ROW = ('core', 'sm', 'live', 0)
BAND2_ROWS = (('core', 'cf', 'live', BAND2_INDEX), ('core', 'cf', 'solo', BAND2_INDEX))
CAP_ROW = ('tail', 'cap', 'live', 0)
# Build-4c's table, not run here: the control's arms and the fixture of test_inwait_box, whose off-arm and cap rules stay in the code.
BUILD4C_TABLE = ((SMOKE_ROW,) + tuple([('core', s, c, 0) for s, c in ROUND0] + [('core', s, c, 1) for s, c in ROUND1]) + BAND2_ROWS + (CAP_ROW,))
ARM_TABLE = (SMOKE_ROW, ('core', 'cf', 'solo', 0), ('core', 'cf', 'live', 0), ('core', 'cf', 'live', 1), ('core', 'cf', 'live', BAND2_INDEX))
DEFAULT_DECODE_SMS = '44'
DEFAULT_CHUNK_TOKENS = '2048'
LEGAL_CHUNK_TOKENS = ('1024', '1536', '2048')     # 1024 the rerun on this bundle, 1536 the fallback if 2048 fails on memory alone
MAMBA_BLOCK_ROWS, HASH_BLOCK_ROWS = 4048, 176     # read from this model's chunks on the box: ends 4048 x k, last stop prompt-1 // 176 x 176
MIN_DECODE_SMS, MAX_DECODE_SMS = 24, 60
DECODE_ARMS = tuple(d for d in _attn_meta.DECODE_ARMS if d['attn'])
REFERENCE_ARM = 'decode-am1'              # this run's decode-only baseline; FIXED_REFERENCE below is the lead's fixed number
FIXED_REFERENCE_TOK_S = 452.8             # the lead's fixed reference (brief, 4 Nov); compose-2's own decode-am1 read 456.598
PARTS = ('core', 'tail')
ENTRY_LIMIT = {'core': ('LAST_ARM_ENTRY_S', 'last_arm_entry'), 'tail': ('LAST_TAIL_ENTRY_S', 'last_tail_entry')}
GRADED = ('core',)
SELECTIONS = ('sm', 'ps1', 'cf', 'cap', 'cfa')
OFF = 'ps1'                                 # the matched off arm of every lever
ATTN = dict.fromkeys(SELECTIONS, True)
LAZY_RECV = dict.fromkeys(SELECTIONS, True)
DECODE_MERGE = dict(dict.fromkeys(SELECTIONS, 'chunk'), cfa='always')
INWAIT = dict.fromkeys(SELECTIONS, True)
NEXT = dict.fromkeys(SELECTIONS, True)      # early_successor on native_stage_local, slots_fast on attn_meta_select
SLICES = dict.fromkeys(SELECTIONS, True)    # prepare_slices on attn_meta_select
WARM = dict.fromkeys(SELECTIONS, False)     # score_warm on attn_meta_select: build-3's pw, dropped (READING-box-3b)
COW_FAST = {'sm': True, 'ps1': False, 'cf': True, 'cap': False, 'cfa': True}     # cow_fast on attn_meta_select
RESERVED_CAP = {'sm': False, 'ps1': False, 'cf': False, 'cap': True, 'cfa': False}    # reserved_cap on attn_meta_select
C1 = dict.fromkeys(SELECTIONS, True)
DEFER_BIND = dict.fromkeys(SELECTIONS, True)
STAGE_LOCAL = True
COHORTS = ('live', 'solo')
INDEXES = tuple(sorted(BAND1_INDEXES + (BAND2_INDEX,)))
SIZES = (1, 2, 4)
COLD_ARM = 'sm.g2-live-0'
SMOKE_ARM = COLD_ARM


def legal_decode_sms(value):
    """The launch's decode SM count, a string: even, MIN..MAX. Anything else raises."""
    text = str(value)
    if not (text.isascii() and text.isdigit()) or int(text) % 2 or not MIN_DECODE_SMS <= int(text) <= MAX_DECODE_SMS or text != str(int(text)):
        raise ValueError('decode SM count must be even, %d..%d' % (MIN_DECODE_SMS, MAX_DECODE_SMS))
    return text


def legal_chunk_tokens(value):
    """The launch's native chunk rows, a string. Anything else raises."""
    text = str(value)
    if text not in LEGAL_CHUNK_TOKENS:
        raise ValueError('native chunk rows must be one of ' + ', '.join(LEGAL_CHUNK_TOKENS))
    return text


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
                         inwait=INWAIT[sel], next_levers=NEXT[sel], prepare_slices=SLICES[sel], score_warm=WARM[sel], cow_fast=COW_FAST[sel], reserved_cap=RESERVED_CAP[sel], overlap=C1[sel], defer_bind=DEFER_BIND[sel], c0='on', stage_local=STAGE_LOCAL, size=size, cohort=cohort,
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
    assert len(names) == len(set(names)) == 5, 'five distinct prefill arm labels'
    assert labels('tail') == [] and labels('core') == names, 'no tail'
    assert names[0] == COLD_ARM, 'the cold arm is the first prefill arm'
    assert names[-1] == label('cf', 2, 'live', BAND2_INDEX), 'band 2 is last'
    assert set(names[1:-1]) < {label(s, BAND_SIZE[p][band_of_index(i)], c, i) for p, s, c, i in BUILD4C_TABLE}, 'the chunk arms carry the labels of the control'
    assert [d['label'] for d in DECODE_ARMS] == [REFERENCE_ARM] and [d['attn'] for d in DECODE_ARMS] == [True]
    assert ROUND1 == ROUND0[:1:-1], 'round 1 is round 0 live reversed'
    assert set(ATTN) == set(LAZY_RECV) == set(DECODE_MERGE) == set(INWAIT) == set(NEXT) == set(SLICES) == set(WARM) == set(COW_FAST) == set(RESERVED_CAP) == set(C1) == set(DEFER_BIND) == set(SELECTIONS)
    for row in arms():
        assert row['overlap'] is True and row['defer_bind'] is True and row['stage_local'] is True, 'C1 defer in every arm'
        assert row['attn'] is True and row['lazy_recv'] is True and row['decode_merge'] == ('always' if row['c1'] == 'cfa' else 'chunk'), 'attn-meta, L1 and L2 in every arm'
        assert row['size'] == 2, 'group size 2 in every arm'
    return True


check()
