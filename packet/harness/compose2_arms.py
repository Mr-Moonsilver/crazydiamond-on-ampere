"""q18a-compose-2 arm table: the one declared table (lead, 4 Nov ~01:20 CET). Change it here and nowhere else.

The engine is alpha-next-c1b's plus two levers, each behind its own default-off flag and installed for the whole run: attn-meta (FLASH_ATTN_META_GDN_BATCH) and stage1-step's
L1 lazy PP receive and L2 decode merge (FLASH_S5_LAZY_RECV, FLASH_S5_DECODE_MERGE). Every prefill arm is C1 defer, group size 2, fused, burst, and selects each lever by typed
control while paused, idle and FULL:
  b0   both off (the matched base)
  am   attn-meta only
  mg   L1 + L2 only (lazy_recv true, decode_merge 'chunk': 8 decoders as one batch during the policy's native phase, 4 + 4 otherwise)
  amg  both
Band 1: live x2 rounds each, solo x1 for b0 and amg. Band 2: live b0 and amg. Before them the two decode-only baselines, attn_meta_arms.DECODE_ARMS as they ran in q18a-attn-meta (8 standing
decoders, no bulk, attn-meta off then on, group size 1, C1 off, L1 and L2 at their off defaults); decode-am1 is the fixed reference for the decode retention.

Order: decode-am0 decode-am1 | round 0: b0 amg solo, b0 am mg amg live | round 1: amg mg am b0 live (round 0 reversed) | band 2: amg b0 live. The first prefill arm, b0.g2-solo-0, is the cold arm
(c1's handling, read as such), so the cold cost falls on the solo control and never on a live pair. To shorten the run, drop am-only round 1 first (the lead's order).
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
BAND2_INDEX = 2
TAIL_INDEX = 0                            # no row uses it: the inherited code reads the name
BAND_SIZE = {'core': {1: 2, 2: 2}, 'tail': {1: 2, 2: 2}}
ROUND0 = (('b0', 'solo'), ('amg', 'solo'), ('b0', 'live'), ('am', 'live'), ('mg', 'live'), ('amg', 'live'))
ROUND1 = (('amg', 'live'), ('mg', 'live'), ('am', 'live'), ('b0', 'live'))
BAND2 = (('amg', 'live'), ('b0', 'live'))
ARM_TABLE = tuple([('core', s, c, 0) for s, c in ROUND0] + [('core', s, c, 1) for s, c in ROUND1] + [('core', s, c, BAND2_INDEX) for s, c in BAND2])
DECODE_ARMS = _attn_meta.DECODE_ARMS
REFERENCE_ARM = 'decode-am1'              # the fixed decode reference: decode-only with attn-meta on
PARTS = ('core', 'tail')
ENTRY_LIMIT = {'core': ('LAST_ARM_ENTRY_S', 'last_arm_entry'), 'tail': ('LAST_TAIL_ENTRY_S', 'last_tail_entry')}
GRADED = ('core',)
SELECTIONS = ('b0', 'am', 'mg', 'amg')
ATTN = {'b0': False, 'am': True, 'mg': False, 'amg': True}
LAZY_RECV = {'b0': False, 'am': False, 'mg': True, 'amg': True}
DECODE_MERGE = {'b0': 'off', 'am': 'off', 'mg': 'chunk', 'amg': 'chunk'}
C1 = dict.fromkeys(SELECTIONS, True)
DEFER_BIND = dict.fromkeys(SELECTIONS, True)
STAGE_LOCAL = True
COHORTS = ('live', 'solo')
INDEXES = tuple(sorted(BAND1_INDEXES + (BAND2_INDEX,)))
SIZES = (1, 2, 4)
COLD_ARM = 'b0.g2-solo-0'


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
        rows.append(dict(label=label(sel, size, cohort, index), selection=selection(sel, size), part=row_part, c1=sel, attn=ATTN[sel], lazy_recv=LAZY_RECV[sel],
                         decode_merge=DECODE_MERGE[sel], overlap=C1[sel], defer_bind=DEFER_BIND[sel], c0='on', stage_local=STAGE_LOCAL, size=size, cohort=cohort,
                         index=index, band=band, extra=False, entry_limit=ENTRY_LIMIT[row_part][0]))
    return rows


def labels(part=None):
    return [row['label'] for row in arms(part)]


def by_label():
    return {row['label']: row for row in arms()}


def matched_off(row):
    """The b0 arm at the same size, cohort and round; None for a b0 arm."""
    if row.get('part') not in GRADED or row.get('c1') == 'b0':
        return None
    want = label('b0', row['size'], row['cohort'], row['index'])
    return want if want in labels(GRADED) else None


def matched_on(row):
    """c1's name (a C1-defer arm's C1-on partner): this table has no C1-on arm."""
    return None


def band1_glob():
    return '[%s]' % ''.join(str(i) for i in BAND1_INDEXES)


def check():
    assert set(BAND_SIZE) == set(PARTS) == set(ENTRY_LIMIT)
    assert BAND1_REQUIRED_ROUNDS == BAND1_ROUNDS == len(BAND1_INDEXES) and BAND2_INDEX not in BAND1_INDEXES
    names = labels()
    assert len(names) == len(set(names)) == 12, 'twelve distinct prefill arm labels'
    assert labels('tail') == [] and labels('core') == names, 'no tail'
    assert names[0] == COLD_ARM, 'the cold arm is the first prefill arm'
    assert [d['label'] for d in DECODE_ARMS] == ['decode-am0', REFERENCE_ARM] and [d['attn'] for d in DECODE_ARMS] == [False, True]
    assert ROUND1 == ROUND0[:1:-1], 'round 1 is round 0 live reversed'
    assert set(ATTN) == set(LAZY_RECV) == set(DECODE_MERGE) == set(C1) == set(DEFER_BIND) == set(SELECTIONS)
    for row in arms():
        assert row['overlap'] is True and row['defer_bind'] is True and row['stage_local'] is True, 'C1 defer in every arm'
        assert row['size'] == 2, 'group size 2 in every arm'
        assert (row['decode_merge'] == 'chunk') is row['lazy_recv'], 'L1 and L2 go together here'
        if row['c1'] != 'b0':
            assert matched_off(row), 'lever arm without its matched base arm: ' + row['label']
    for index in BAND1_INDEXES:
        assert {r['c1'] for r in arms() if r['index'] == index and r['cohort'] == 'live'} == set(SELECTIONS), ('live four', index)
    return True


check()
