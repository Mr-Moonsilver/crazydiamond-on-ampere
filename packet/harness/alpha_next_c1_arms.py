"""q18a-alpha-next-c1 arm table: the one declared table (lead approved 2 Nov about 16:40 CET: 14 arms; C1 rebind folded in 2 Nov
20:55-21:0x CET: overlap+defer arms added, every arm sets defer_bind). Change it here and nowhere else.

The engine is alpha-next's (fused QSA, layer groups and C0 stage-local installed) with FLASH_S5_STAGE_OVERLAP=1 (C1),
FLASH_S5_STAGE_OVERLAP_DEFER_BIND=1 (the deferred PP bind installed) and FLASH_S5_STAGE_OVERLAP_OBSERVE=1 (the overlap observation)
added for the whole run; every arm selects stage-local, its C1 selection and its defer_bind, explicitly, by typed control while
paused, idle and FULL: native_stage_local {local: true, overlap: false, defer_bind: false} (C1 off: C0's local path),
{local: true, overlap: true, defer_bind: false} (C1 on: two chunks in flight) or {local: true, overlap: true, defer_bind: true}
(C1 defer: two chunks in flight with the deferred PP bind). C0 local is selected in every arm, so the control of this queue is
alpha-next's candidate. Fused QSA and burst progress
are selected in every arm; the group size comes from BAND_SIZE[part][band] by the typed native_layer_group control.

A row is (part, C1 selection, cohort, round index):
  part          'core' arms make the queue grade (band 1 and band 2). 'tail' arms run after the core grade is written and are
                reads only: the queue verdict does not depend on them, except for crash accounting (the death watch and scan cover
                every arm) and leak and OOM accounting (instruments-end and the end FULL run after the tail and are counted; an
                OOM in a tail arm fails that arm's read and is reported). There is no 'extra' part in this table.
  C1 selection  'on' selects the overlap path, 'defer' the overlap path with the deferred PP bind, 'off' C0's local path; all
                with C0 stage-local selected; defer_bind is false on 'on' and 'off', true on 'defer' (DEFER_BIND).
  cohort        'live' keeps the 8 standing decoders through the prefill, 'solo' cancels them at the first native chunk (held-f).
  round index   0 and 1 are band 1 (65,445 fresh, then the append), 2 is band 2 (131,094 fresh, alpha-next's band 2 workload,
                past the 97,152-row arena), as in alpha-next. The index is in the label and the label is in the cache salt.

Order (lead): core first, tail last. Band 1 at g2, ABBA over two rounds, each C1-on arm with its matched C1-off arm at the same
cohort and round, and each C1-defer arm beside its matched C1-on arm, mirrored like the others (ABBA against C1 on per cohort):
off-solo-0 on-solo-0 defer-solo-0 defer-live-0 on-live-0 off-live-0 | off-live-1 on-live-1 defer-live-1 defer-solo-1 on-solo-1
off-solo-1. Band 2 at g2, matched: off-solo on-solo defer-solo defer-live on-live off-live (BAND2_DEFER, lead 22:5x: the
defer arms between on-solo and on-live; the budget is alpha_next_c1_gate's). Then the g4 tail, C1 on, band 1 round 0: on-solo-g4 on-live-g4,
reads only, never matched (its memory is reported only). The tail stays last: ``group_layers_max`` is a lifetime high-water mark (alpha-next's reason).

Standing decoders: alpha-next's (8; streams STANDING_GREEDY greedy at temperature 0 carry the numerics read, the others keep the
manifest's sampling and carry the decode read).

The API is alpha-next's (alpha_next_arms), so alpha-next's arm and grade code reads this table through a view
(e3_alpha_next_c1_box.table_view); the C1 selection is the only axis that replaces alpha-next's C0 axis.
"""
import alpha_next_arms as _alpha_next

STANDING_DECODERS = _alpha_next.STANDING_DECODERS
STANDING_GREEDY = _alpha_next.STANDING_GREEDY
GREEDY_TEMPERATURE = _alpha_next.GREEDY_TEMPERATURE
STANDING_SAMPLED = _alpha_next.STANDING_SAMPLED
BAND1_ROUNDS = 2                          # band 1 core rounds (lead, 16:40: two, ABBA)
BAND1_REQUIRED_ROUNDS = 2                 # every band 1 round is core
BAND1_INDEXES = (0, 1)
BAND2_INDEX = 2
TAIL_INDEX = 0                            # the g4 tail runs band 1, round 0 (its own label: the size differs)
BAND_SIZE = {'core': {1: 2, 2: 2},        # layer group size per part and band, sent per arm through native_layer_group {size}
             'tail': {1: 4, 2: 4}}
ORDER_A = (('off', 'solo'), ('on', 'solo'), ('defer', 'solo'), ('defer', 'live'), ('on', 'live'), ('off', 'live'))  # B = A reversed
# Band 2's C1-defer arms (lead, 21:0x: "only if the budget allows", core limit at 1.5x the walls). At the 9,000 s budget they did
# not fit; lead 2 Nov 22:5x CET (box free until ~09:45): added, with the queue budget raised to 9,750 s (alpha_next_c1_gate), at
# which every arm still enters at 1.5x the walls.
BAND2_DEFER = True
ORDER_BAND2 = ((('off', 'solo'), ('on', 'solo'), ('defer', 'solo'), ('defer', 'live'), ('on', 'live'), ('off', 'live'))
               if BAND2_DEFER else (('off', 'solo'), ('on', 'solo'), ('on', 'live'), ('off', 'live')))
ORDER_TAIL = (('on', 'solo'), ('on', 'live'))


def _table():
    rows = []
    for k, index in enumerate(BAND1_INDEXES):
        rows += [('core', c1, cohort, index) for c1, cohort in (ORDER_A if k % 2 == 0 else ORDER_A[::-1])]
    rows += [('core', c1, cohort, BAND2_INDEX) for c1, cohort in ORDER_BAND2]
    rows += [('tail', c1, cohort, TAIL_INDEX) for c1, cohort in ORDER_TAIL]
    return tuple(rows)


ARM_TABLE = _table()

PARTS = ('core', 'tail')
# Each part's arm entry limit, relative to the run's own start (bash SECONDS), never a clock: the alpha_next_c1_gate name and
# the draft's native41_budget key. Core arms reserve the core FULL, the core grade and the end; tail arms only the end.
ENTRY_LIMIT = {'core': ('LAST_ARM_ENTRY_S', 'last_arm_entry'), 'tail': ('LAST_TAIL_ENTRY_S', 'last_tail_entry')}
GRADED = ('core',)                        # the parts the core grade (the queue verdict) covers
C1 = {'on': True, 'off': False, 'defer': True}           # the overlap value of native_stage_local per C1 selection
DEFER_BIND = {'on': False, 'off': False, 'defer': True}  # the defer_bind value of native_stage_local per C1 selection (every arm)
STAGE_LOCAL = True                        # C0 local in every arm
COHORTS = ('live', 'solo')
INDEXES = tuple(sorted(BAND1_INDEXES + (BAND2_INDEX,)))
SIZES = (1, 2, 4)                         # the sizes Step B's engagement rule knows (e3_step_b_box.SIZES)


def band_of_index(index):
    """Band 1 for BAND1_INDEXES, band 2 for index 2 (alpha-next's rule)."""
    return 1 if index in BAND1_INDEXES else 2 if index == BAND2_INDEX else None


def selection(c1, size):
    """The arm's selection token: C1 and group size, no '-' (``c1on.g2``)."""
    return 'c1%s.g%d' % (c1, size)


def label(c1, size, cohort, index):
    return '%s-%s-%d' % (selection(c1, size), cohort, index)


def parse_selection(token):
    """(c1, size) of a selection token, or None."""
    head, dot, tail = str(token).partition('.')
    if not dot or head not in tuple('c1' + c for c in C1) or not tail.startswith('g') or not tail[1:].isdigit():
        return None
    return head[2:], int(tail[1:])


def arms(part=None):
    """The table as rows, in order; one part (or a tuple of parts) only when ``part`` is given. A part this table does not have
    (alpha-next's 'extra') gives no rows."""
    parts = (part,) if isinstance(part, str) else part
    rows = []
    for row_part, c1, cohort, index in ARM_TABLE:
        if parts is not None and row_part not in parts:
            continue
        band = band_of_index(index)
        size = BAND_SIZE[row_part][band]
        rows.append(dict(label=label(c1, size, cohort, index), selection=selection(c1, size), part=row_part, c1=c1,
                         overlap=C1[c1], defer_bind=DEFER_BIND[c1], c0='on', stage_local=STAGE_LOCAL, size=size, cohort=cohort, index=index, band=band,
                         extra=False, entry_limit=ENTRY_LIMIT[row_part][0]))
    return rows


def labels(part=None):
    return [row['label'] for row in arms(part)]


def by_label():
    return {row['label']: row for row in arms()}


def matched_off(row):
    """The graded C1-off arm matched to a graded C1-on or C1-defer arm: same size, cohort and round. None for a tail arm or when
    the table has none."""
    if row.get('part') not in GRADED:
        return None
    want = label('off', row['size'], row['cohort'], row['index'])
    return want if want in labels(GRADED) else None


def matched_on(row):
    """The graded C1-on arm matched to a graded C1-defer arm (its ABBA partner): same size, cohort and round."""
    if row.get('part') not in GRADED or row.get('c1') != 'defer':
        return None
    want = label('on', row['size'], row['cohort'], row['index'])
    return want if want in labels(GRADED) else None


def band1_glob():
    """The bash pattern of the band 1 round indexes (the draft's band 1 manifest choice)."""
    return '[%s]' % ''.join(str(i) for i in BAND1_INDEXES)


def check():
    """The table's own invariants; raises on a malformed table so no queue text is generated from one."""
    assert set(BAND_SIZE) == set(PARTS) == set(ENTRY_LIMIT), BAND_SIZE
    for sizes in BAND_SIZE.values():
        assert set(sizes) == {1, 2} and all(size in SIZES for size in sizes.values()), BAND_SIZE
    assert BAND1_REQUIRED_ROUNDS == BAND1_ROUNDS == len(BAND1_INDEXES) and BAND2_INDEX not in BAND1_INDEXES
    assert TAIL_INDEX in BAND1_INDEXES
    assert all(0 <= i <= 9 for i in INDEXES), 'round indexes are one digit (the draft matches them with one bash glob)'
    order = [PARTS.index(part) for part, _, _, _ in ARM_TABLE]
    assert order == sorted(order), 'parts in order: core, then the tail (the core grade runs before the tail)'
    names = labels()
    assert len(names) == len(set(names)), 'repeated arm label'
    for row in arms(GRADED):
        if row['c1'] in ('on', 'defer'):
            assert matched_off(row), 'graded C1-on arm without its matched C1-off arm: ' + row['label']
        if row['c1'] == 'defer':
            assert matched_on(row), 'graded C1-defer arm without its matched C1-on arm: ' + row['label']
    assert [r for r in arms('core') if r['c1'] == 'defer' and r['band'] == 1], 'band 1 carries the C1-defer arms'
    for index in BAND1_INDEXES:
        # ABBA against C1 on per cohort: the defer arm sits beside its on arm, before it in one round and after it in the other.
        order = [(r['c1'], r['cohort']) for r in arms('core') if r['index'] == index]
        for cohort in COHORTS:
            on, defer = order.index(('on', cohort)), order.index(('defer', cohort))
            assert abs(on - defer) == 1, (index, cohort, order)
    for cohort in COHORTS:
        firsts = []
        for index in BAND1_INDEXES:
            order = [(r['c1'], r['cohort']) for r in arms('core') if r['index'] == index]
            firsts.append(order.index(('defer', cohort)) < order.index(('on', cohort)))
        assert firsts == [firsts[0], not firsts[0]], ('ABBA against C1 on', cohort, firsts)
    assert all(r['c1'] == 'on' for r in arms('tail')), 'the tail is C1 on only'
    assert [r for r in arms('core') if r['c1'] == 'off'], 'the core needs C1-off controls'
    sizes = [r['size'] for r in arms()]
    assert sizes == sorted(sizes), 'the largest group is a lifetime high-water mark: sizes never decrease along the table'
    assert all(r['stage_local'] is True for r in arms()), 'C0 local in every arm'
    assert all(r['defer_bind'] is (r['c1'] == 'defer') and type(r['defer_bind']) is bool for r in arms()), 'defer_bind on every arm'
    return True


check()
