"""q18a-inwait arm table: the one declared table (lead, 4 Nov ~05:30 CET). Change it here and nowhere else.

Build-3 (results/hub/inwait/DESIGN-3.md): in-wait, the early successor and fast slot mappings are on in every arm (build-2's nx1, the matched base).
The engine is build-2's plus these levers behind one default-off flag, FLASH_S5_INWAIT, installed for the whole run. Every prefill arm is C1 defer, attn-meta on,
stage1-step's L1 + L2 selected; the only axis, selected per arm by typed control while paused, idle and FULL:
  ps0  PREPARE whole (build-2's nx1)
  ps1  PREPARE in slices
  pw   PREPARE in slices and the warm score block; the first pw arm flushes the allocator cache at its selection
Band 1 only: ps0 and ps1 live x2 and solo x1 each, then pw solo x1 and live x2. Before them one decode-only baseline with attn-meta on (decode-am1).

Order: decode-am1 | round 0: ps0 ps1 solo, ps0 ps1 live | round 1: ps1 ps0 live (reversed) | pw solo, live, live. The first prefill arm, ps0.g2-solo-0, is the cold arm.
pw comes last so that no off arm follows a flushed cache: its memory read starts from the flush and its speed reads against ps1.
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
BAND2_INDEX = 2                           # no row uses it: the inherited code reads the name
TAIL_INDEX = 0                            # no row uses it: the inherited code reads the name
BAND_SIZE = {'core': {1: 2, 2: 2}, 'tail': {1: 2, 2: 2}}
ROUND0 = (('ps0', 'solo'), ('ps1', 'solo'), ('ps0', 'live'), ('ps1', 'live'))
ROUND1 = (('ps1', 'live'), ('ps0', 'live'))
WARM_ROWS = (('core', 'pw', 'solo', 0), ('core', 'pw', 'live', 0), ('core', 'pw', 'live', 1))
ARM_TABLE = tuple([('core', s, c, 0) for s, c in ROUND0] + [('core', s, c, 1) for s, c in ROUND1]) + WARM_ROWS
DECODE_ARMS = tuple(d for d in _attn_meta.DECODE_ARMS if d['attn'])
REFERENCE_ARM = 'decode-am1'              # this run's decode-only baseline; FIXED_REFERENCE below is the lead's fixed number
FIXED_REFERENCE_TOK_S = 452.8             # the lead's fixed reference (brief, 4 Nov); compose-2's own decode-am1 read 456.598
PARTS = ('core', 'tail')
ENTRY_LIMIT = {'core': ('LAST_ARM_ENTRY_S', 'last_arm_entry'), 'tail': ('LAST_TAIL_ENTRY_S', 'last_tail_entry')}
GRADED = ('core',)
SELECTIONS = ('ps0', 'ps1', 'pw')
ATTN = dict.fromkeys(SELECTIONS, True)
LAZY_RECV = dict.fromkeys(SELECTIONS, True)
DECODE_MERGE = dict.fromkeys(SELECTIONS, 'chunk')
INWAIT = dict.fromkeys(SELECTIONS, True)
NEXT = dict.fromkeys(SELECTIONS, True)      # early_successor on native_stage_local, slots_fast on attn_meta_select
SLICES = {'ps0': False, 'ps1': True, 'pw': True}    # prepare_slices on attn_meta_select
WARM = {'ps0': False, 'ps1': False, 'pw': True}     # score_warm on attn_meta_select
C1 = dict.fromkeys(SELECTIONS, True)
DEFER_BIND = dict.fromkeys(SELECTIONS, True)
STAGE_LOCAL = True
COHORTS = ('live', 'solo')
INDEXES = tuple(sorted(BAND1_INDEXES))
SIZES = (1, 2, 4)
COLD_ARM = 'ps0.g2-solo-0'


def band_of_index(index):
    return 1 if index in BAND1_INDEXES else None


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
                         inwait=INWAIT[sel], next_levers=NEXT[sel], prepare_slices=SLICES[sel], score_warm=WARM[sel], overlap=C1[sel], defer_bind=DEFER_BIND[sel], c0='on', stage_local=STAGE_LOCAL, size=size, cohort=cohort,
                         index=index, band=band, extra=False, entry_limit=ENTRY_LIMIT[row_part][0]))
    return rows


def labels(part=None):
    return [row['label'] for row in arms(part)]


def by_label():
    return {row['label']: row for row in arms()}


def matched_off(row):
    """The ps0 arm at the same size, cohort and round; None for an ps0 arm."""
    if row.get('part') not in GRADED or row.get('c1') == 'ps0':
        return None
    want = label('ps0', row['size'], row['cohort'], row['index'])
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
    assert len(names) == len(set(names)) == 9, 'nine distinct prefill arm labels'
    assert labels('tail') == [] and labels('core') == names, 'no tail'
    assert names[0] == COLD_ARM, 'the cold arm is the first prefill arm'
    assert [d['label'] for d in DECODE_ARMS] == [REFERENCE_ARM] and [d['attn'] for d in DECODE_ARMS] == [True]
    assert ROUND1 == ROUND0[:1:-1], 'round 1 is round 0 live reversed'
    assert set(ATTN) == set(LAZY_RECV) == set(DECODE_MERGE) == set(INWAIT) == set(NEXT) == set(SLICES) == set(WARM) == set(C1) == set(DEFER_BIND) == set(SELECTIONS)
    for row in arms():
        assert row['overlap'] is True and row['defer_bind'] is True and row['stage_local'] is True, 'C1 defer in every arm'
        assert row['attn'] is True and row['lazy_recv'] is True and row['decode_merge'] == 'chunk', 'attn-meta, L1 and L2 in every arm'
        assert row['size'] == 2, 'group size 2 in every arm'
        if row['c1'] != 'ps0':
            assert matched_off(row), 'on arm without its matched off arm: ' + row['label']
    for index in BAND1_INDEXES:
        assert {r['c1'] for r in arms() if r['index'] == index and r['cohort'] == 'live'} == set(SELECTIONS), ('live pair', index)
    return True


check()
