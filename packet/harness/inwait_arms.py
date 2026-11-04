"""q18a-inwait arm table: the one declared table (DESIGN-5b, 4 Nov, build-5b). Change it here and nowhere else.

Build-5b (results/hub/inwait/DESIGN-5b.md): in-wait, the early successor, fast slot mappings, PREPARE in slices, the fast CoW copy and the host pass filtered to QSA targets are on in every arm (build-5's hf).
The engine is build-5's plus these levers behind one default-off flag, FLASH_S5_INWAIT, installed for the whole run. The only axis, selected per arm by typed control while paused, idle and FULL; every prefill arm flushes the allocator at its selection, so each lever is the only difference:
  hf   the base
  both hf and the exact successor plan
  cap  hf and the reserved cap that keeps 512 MiB free
  w0   cap and the warm score block on rank 0 only
  wa   cap and the warm score block on every rank
  sm   both and w0 as the exact-path smoke: arm 1, cold, graded as any arm, no matched pair; if it does not engage, nothing else runs
Order: sm live | decode-am1 | round 0: hf both solo, hf both cap w0 wa live | round 1: wa w0 cap both hf live (reversed) | the core grade | w0 live at band 2, last and a read, so an OOM costs nothing else.
Band 1's fresh and append requests carry top-20 logprobs, so hf and both can be compared token vector by token vector (READING-box-5's flag).
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
ROUND0 = (('hf', 'solo'), ('both', 'solo'), ('hf', 'live'), ('both', 'live'), ('cap', 'live'), ('w0', 'live'), ('wa', 'live'))
ROUND1 = (('wa', 'live'), ('w0', 'live'), ('cap', 'live'), ('both', 'live'), ('hf', 'live'))
SMOKE_ROW = ('core', 'sm', 'live', 0)
BAND2_ROWS = (('tail', 'w0', 'live', BAND2_INDEX),)    # after the core grade, a read: an OOM here costs no pair and does not change the grade
ARM_TABLE = ((SMOKE_ROW,) + tuple([('core', s, c, 0) for s, c in ROUND0] + [('core', s, c, 1) for s, c in ROUND1]) + BAND2_ROWS)
DECODE_ARMS = tuple(d for d in _attn_meta.DECODE_ARMS if d['attn'])
REFERENCE_ARM = 'decode-am1'              # this run's decode-only baseline; FIXED_REFERENCE below is the lead's fixed number
FIXED_REFERENCE_TOK_S = 452.8             # the lead's fixed reference (brief, 4 Nov); compose-2's own decode-am1 read 456.598
PARTS = ('core', 'tail')
ENTRY_LIMIT = {'core': ('LAST_ARM_ENTRY_S', 'last_arm_entry'), 'tail': ('LAST_TAIL_ENTRY_S', 'last_tail_entry')}
GRADED = ('core',)
SELECTIONS = ('sm', 'hf', 'both', 'cap', 'w0', 'wa')
OFF = 'hf'                                  # the matched off arm of every lever
ON = 'both'
ATTN = dict.fromkeys(SELECTIONS, True)
LAZY_RECV = dict.fromkeys(SELECTIONS, True)
DECODE_MERGE = dict.fromkeys(SELECTIONS, 'chunk')
INWAIT = dict.fromkeys(SELECTIONS, True)
NEXT = dict.fromkeys(SELECTIONS, True)      # early_successor on native_stage_local, slots_fast on attn_meta_select
SLICES = dict.fromkeys(SELECTIONS, True)    # prepare_slices on attn_meta_select
WARM = {'sm': True, 'hf': False, 'both': False, 'cap': False, 'w0': True, 'wa': True}    # score_warm on attn_meta_select
WARM_RANK0 = {'sm': True, 'hf': False, 'both': False, 'cap': False, 'w0': True, 'wa': False}    # score_warm_rank0 on attn_meta_select
FLUSH = dict.fromkeys(SELECTIONS, True)     # allocator_flush on attn_meta_select: every prefill arm, so the pairs start matched
COW_FAST = dict.fromkeys(SELECTIONS, True)    # cow_fast on attn_meta_select
RESERVED_CAP = {'sm': True, 'hf': False, 'both': False, 'cap': True, 'w0': True, 'wa': True}    # reserved_cap on attn_meta_select
HOST_FILTER = dict.fromkeys(SELECTIONS, True)    # cow_host_filter on attn_meta_select
EXACT = {'sm': True, 'hf': False, 'both': True, 'cap': False, 'w0': False, 'wa': False}    # exact_successor on native_stage_local
C1 = dict.fromkeys(SELECTIONS, True)
DEFER_BIND = dict.fromkeys(SELECTIONS, True)
STAGE_LOCAL = True
COHORTS = ('live', 'solo')
INDEXES = tuple(sorted(BAND1_INDEXES + (BAND2_INDEX,)))
SIZES = (1, 2, 4)
COLD_ARM = 'sm.g2-live-0'
SMOKE_ARM = COLD_ARM


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
                         inwait=INWAIT[sel], next_levers=NEXT[sel], prepare_slices=SLICES[sel], score_warm=WARM[sel], score_warm_rank0=WARM_RANK0[sel], allocator_flush=FLUSH[sel], cow_fast=COW_FAST[sel], reserved_cap=RESERVED_CAP[sel], cow_host_filter=HOST_FILTER[sel], exact_successor=EXACT[sel], overlap=C1[sel], defer_bind=DEFER_BIND[sel], c0='on', stage_local=STAGE_LOCAL, size=size, cohort=cohort,
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
    assert len(names) == len(set(names)) == 14, 'fourteen distinct prefill arm labels'
    assert labels('tail') == ['w0.g2-live-2'] and labels('core') == names[:-1], 'one tail arm, last'
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
        assert {OFF, ON, 'cap', 'w0', 'wa'} <= {r['c1'] for r in arms() if r['index'] == index and r['cohort'] == 'live'}, ('live pair', index)
    return True


check()
