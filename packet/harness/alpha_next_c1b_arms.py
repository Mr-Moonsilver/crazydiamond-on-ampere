"""q18a-alpha-next-c1b arm table: the fixed-schedule numerics queue for C0 and C1 defer (lead, 3 Nov ~10:10 CET; DESIGN-c1-next (c),
band 1 only). The one declared table: change it here and nowhere else.

The engine is alpha-next-c1's, byte for byte (its delivered build 0c33ca841620, the same switches for the whole run:
FLASH_S5_STAGE_OVERLAP=1, FLASH_S5_STAGE_OVERLAP_DEFER_BIND=1, FLASH_S5_STAGE_OVERLAP_OBSERVE=1). Each arm selects, by typed control
while paused, idle and FULL, native_stage_local {local, overlap, defer_bind}:
  'serial'  {local: false, overlap: false, defer_bind: false}  C0 off (the serial staged path): the numerics controls S1 S2 S3;
  'off'     {local: true,  overlap: false, defer_bind: false}  C0 local, C1 off: the C0 candidates L1 L2;
  'defer'   {local: true,  overlap: true,  defer_bind: true}   C1 defer (two chunks in flight, the deferred PP bind): D.
Serial is selectable per arm on this engine: stage_progress._stage_control takes local False; stage_overlap's control forces
overlap and defer_bind False when local is False (patches/s5_engine/stage_overlap.py OverlapSegmentDriver._stage_control).

Every arm is solo (the 8 standing decoders are cancelled at the first native chunk, held-f), band 1 (the fixed 65,445-token fresh
prompt of alpha-next's band 1 manifest, recorded by sha), group size 2, fused and burst progress selected. The fresh request is
the numerics probe: temperature 0, 256 generated tokens (min = max, ignore_eos), top-20 logprobs per position
(e3_alpha_next_c1b_box.NUMERICS). Its cache salt is the arm's own (the label is in it), so no arm reads another arm's prefix cache.

Order (lead): S1 S2 L1 L2 D S3. A row is (part, selection, cohort, round index); band 1's round indexes 0, 1 and 3 (2 is band 2's
index in alpha-next's rule and is not used here). The numerics role of each row is ROLE. No tail.
"""
import alpha_next_arms as _alpha_next

STANDING_DECODERS = _alpha_next.STANDING_DECODERS
STANDING_GREEDY = _alpha_next.STANDING_GREEDY
GREEDY_TEMPERATURE = _alpha_next.GREEDY_TEMPERATURE
STANDING_SAMPLED = _alpha_next.STANDING_SAMPLED
BAND1_INDEXES = (0, 1, 3)
BAND1_ROUNDS = len(BAND1_INDEXES)
BAND1_REQUIRED_ROUNDS = BAND1_ROUNDS      # every row is core
BAND2_INDEX = 2                           # alpha-next's band 2 index: no row uses it
TAIL_INDEX = 0
BAND_SIZE = {'core': {1: 2, 2: 2}, 'tail': {1: 4, 2: 4}}
ARM_TABLE = (('core', 'serial', 'solo', 0), ('core', 'serial', 'solo', 1), ('core', 'off', 'solo', 0),
             ('core', 'off', 'solo', 1), ('core', 'defer', 'solo', 0), ('core', 'serial', 'solo', 3))
ROLES = ('S1', 'S2', 'L1', 'L2', 'D', 'S3')
PARTS = ('core', 'tail')
ENTRY_LIMIT = {'core': ('LAST_ARM_ENTRY_S', 'last_arm_entry'), 'tail': ('LAST_TAIL_ENTRY_S', 'last_tail_entry')}
GRADED = ('core',)
C1 = {'serial': False, 'off': False, 'defer': True}          # native_stage_local overlap per selection
DEFER_BIND = {'serial': False, 'off': False, 'defer': True}  # native_stage_local defer_bind per selection (sent in every arm)
LOCAL = {'serial': False, 'off': True, 'defer': True}        # native_stage_local local per selection (C0)
PREFIX = {'serial': 'c0off', 'off': 'c1off', 'defer': 'c1defer'}
COHORTS = ('solo',)
INDEXES = tuple(sorted(BAND1_INDEXES))
SIZES = (1, 2, 4)


def band_of_index(index):
    return 1 if index in BAND1_INDEXES else 2 if index == BAND2_INDEX else None


def selection(c1, size):
    """The arm's selection token (``c0off.g2``, ``c1off.g2``, ``c1defer.g2``)."""
    return '%s.g%d' % (PREFIX[c1], size)


def label(c1, size, cohort, index):
    return '%s-%s-%d' % (selection(c1, size), cohort, index)


def parse_selection(token):
    """(selection, size) of a selection token, or None."""
    head, dot, tail = str(token).partition('.')
    names = {v: k for k, v in PREFIX.items()}
    if not dot or head not in names or not tail.startswith('g') or not tail[1:].isdigit():
        return None
    return names[head], int(tail[1:])


def arms(part=None):
    parts = (part,) if isinstance(part, str) else part
    rows = []
    for (row_part, c1, cohort, index), role in zip(ARM_TABLE, ROLES):
        if parts is not None and row_part not in parts:
            continue
        band = band_of_index(index)
        size = BAND_SIZE[row_part][band]
        rows.append(dict(label=label(c1, size, cohort, index), selection=selection(c1, size), part=row_part, c1=c1,
                         overlap=C1[c1], defer_bind=DEFER_BIND[c1], c0='on' if LOCAL[c1] else 'off', stage_local=LOCAL[c1],
                         size=size, cohort=cohort, index=index, band=band, role=role, extra=False,
                         entry_limit=ENTRY_LIMIT[row_part][0]))
    return rows


def labels(part=None):
    return [row['label'] for row in arms(part)]


def by_label():
    return {row['label']: row for row in arms()}


def by_role():
    return {row['role']: row for row in arms()}


def matched_off(row):
    """C1's pairing (e3_alpha_next_c1_box reads it): the graded C1-off (C0 local) arm of a C1-defer arm, same size, cohort and
    round. None for the other rows."""
    if row.get('part') not in GRADED or row.get('c1') != 'defer':
        return None
    want = label('off', row['size'], row['cohort'], row['index'])
    return want if want in labels(GRADED) else None


def matched_on(row):
    """No C1-on arm in this table."""
    return None


def matched_serial(row):
    """alpha-next's pairing on the C0 axis (the grade's view): the serial arm of a C0-local or C1-defer arm, same round."""
    if row.get('part') not in GRADED or row.get('c1') == 'serial':
        return None
    want = label('serial', row['size'], row['cohort'], row['index'])
    return want if want in labels(GRADED) else None


def band1_glob():
    return '[%s]' % ''.join(str(i) for i in BAND1_INDEXES)


def check():
    assert len(ARM_TABLE) == len(ROLES) == 6 and ROLES == ('S1', 'S2', 'L1', 'L2', 'D', 'S3')
    assert [c for _, c, _, _ in ARM_TABLE] == ['serial', 'serial', 'off', 'off', 'defer', 'serial'], 'S1 S2 L1 L2 D S3'
    assert all(p == 'core' and cohort == 'solo' and band_of_index(i) == 1 for p, _, cohort, i in ARM_TABLE), 'band 1, solo, core'
    assert BAND2_INDEX not in BAND1_INDEXES and all(0 <= i <= 9 for i in INDEXES)
    names = labels()
    assert len(names) == len(set(names)), 'repeated arm label'
    assert len({r['size'] for r in arms()}) == 1, 'one group size'
    for row in arms():
        assert row['defer_bind'] is (row['c1'] == 'defer') and row['overlap'] is (row['c1'] == 'defer')
        assert row['stage_local'] is (row['c1'] != 'serial')
        assert parse_selection(row['selection']) == (row['c1'], row['size'])
    d = by_role()['D']
    assert matched_off(d) == by_role()['L1']['label'] and matched_serial(d) == by_role()['S1']['label']
    return True


check()
