"""q18a-attn-meta arm table: the one declared table (lead amendment A, 3 Nov 12:37 CET; decision A: per-arm typed control).
Change it here and nowhere else.

The engine is alpha-next-c1's (fused QSA, layer groups, C0 stage-local, C1 with the deferred PP bind installed) with
FLASH_ATTN_META_GDN_BATCH=1 added for the whole run: attn_meta_select installs the Selector (not selected) at startup on
every rank. Every arm selects, by typed control while paused idle FULL, its C1 selection and defer_bind (native_stage_local,
c1's rule), its group size (native_layer_group) and its attn-meta selection (the qualification worker action
attn_meta_select {on}).

Cells (amendment A):
  R    attn-meta off  C1 off    G1   band 1 live x2, solo x1
  M    attn-meta on   C1 off    G1   band 1 live x2, solo x1
  P    attn-meta off  C1 defer  G1   band 1 live x2, solo x1
  MP   attn-meta on   C1 defer  G1   band 1 live x2, solo x1, band 2 live x1
  P2   attn-meta off  C1 defer  G2   band 1 live x1
  MP2  attn-meta on   C1 defer  G2   band 1 live x1
plus the standalone decode baselines: decode-only, the 8 standing streams, no bulk, one arm attn-meta off, one on (G1, C1 off).

The c1 API. alpha-next-c1's arm and grade code (e3_alpha_next_c1_box, unchanged) reads this table in place of
alpha_next_c1_arms (bound for one call). Its label is (C1 selection, size, cohort, round index), so the attn-meta selection
lives in the round index: attn-meta off rounds are 0 and 1, attn-meta on rounds 3 and 4 (band 1, the same 65,445-token
workload), band 2 is index 2 (MP only, attn-meta on). Labels stay unique (c1off.g1-live-0 is R, c1off.g1-live-3 is M), the
cache salt differs per arm, and c1's matched C1-off partner of a C1-defer arm is the same attn-meta selection: P's is R,
MP's is M (same size, cohort and index). ``ATTN`` maps each index to its selection; ``attn_of(row)`` reads it.

Parts as c1: 'core' (graded: the G1 cells and MP band 2) then 'tail' (P2, MP2 at G2; reads only, crash, leak and OOM
accounted), because group_layers_max is a lifetime high-water mark (sizes never decrease along the table). The decode-only
baselines are not in ARM_TABLE (c1's arm needs a prefill); they are DECODE_ARMS, run first, graded by e3_attn_meta_box.

Order (amendment A: interleave the cells so drift does not fall on one cell; P2/MP2 last):
  decode-off decode-on | live round A: R M P MP | live round B: MP P M R | solo: M R MP P | MP band 2 live | P2 MP2
"""
STANDING_DECODERS = 8
STANDING_GREEDY = (0, 1)                  # alpha-next's (alpha_next_arms), restated; test_attn_meta_arms compares them
GREEDY_TEMPERATURE = 0.0
STANDING_SAMPLED = (2, 3, 4, 5, 6, 7)

CELLS = {'R': (False, 'off', 1), 'M': (True, 'off', 1), 'P': (False, 'defer', 1), 'MP': (True, 'defer', 1),
         'P2': (False, 'defer', 2), 'MP2': (True, 'defer', 2)}
ROUNDS = {False: (0, 1), True: (3, 4)}    # band 1 round indexes per attn-meta selection
BAND2_INDEX = 2                           # MP only (attn-meta on)
ATTN = {0: False, 1: False, 2: True, 3: True, 4: True}
BAND1_INDEXES = (0, 1, 3, 4)
BAND1_ROUNDS = 2                          # per attn-meta selection
BAND1_REQUIRED_ROUNDS = 2
INDEXES = (0, 1, 2, 3, 4)
SIZES = (1, 2, 4)                         # the sizes the engagement rule knows (e3_step_b_box.SIZES)
BAND_SIZE = {'core': {1: 1, 2: 1}, 'tail': {1: 2, 2: 2}}
PARTS = ('core', 'tail')
GRADED = ('core',)
ENTRY_LIMIT = {'core': ('LAST_ARM_ENTRY_S', 'last_arm_entry'), 'tail': ('LAST_TAIL_ENTRY_S', 'last_tail_entry')}
C1 = {'off': False, 'defer': True}        # native_stage_local overlap per C1 selection (no plain C1-on arm here)
DEFER_BIND = {'off': False, 'defer': True}
STAGE_LOCAL = True
COHORTS = ('live', 'solo')

# (part, cell, cohort, round): round 0 / 1 is the cell's first / second band 1 round, 'b2' band 2.
_LIVE_A = ('R', 'M', 'P', 'MP')
ORDER = tuple([('core', c, 'live', 0) for c in _LIVE_A] + [('core', c, 'live', 1) for c in _LIVE_A[::-1]]
              + [('core', c, 'solo', 0) for c in ('M', 'R', 'MP', 'P')] + [('core', 'MP', 'live', 'b2')]
              + [('tail', 'P2', 'live', 0), ('tail', 'MP2', 'live', 0)])
DECODE_ARMS = (dict(label='decode-am0', attn=False, c1='off', overlap=False, defer_bind=False, size=1, cohort='decode',
                    part='decode', cell='R'),
               dict(label='decode-am1', attn=True, c1='off', overlap=False, defer_bind=False, size=1, cohort='decode',
                    part='decode', cell='M'))


def index_of(cell, rnd):
    attn = CELLS[cell][0]
    return BAND2_INDEX if rnd == 'b2' else ROUNDS[attn][rnd]


def _table():
    return tuple((part, CELLS[cell][1], cohort, index_of(cell, rnd)) for part, cell, cohort, rnd in ORDER)


ARM_TABLE = _table()


def band_of_index(index):
    return 1 if index in BAND1_INDEXES else 2 if index == BAND2_INDEX else None


def selection(c1, size):
    return 'c1%s.g%d' % (c1, size)


def label(c1, size, cohort, index):
    return '%s-%s-%d' % (selection(c1, size), cohort, index)


def parse_selection(token):
    head, dot, tail = str(token).partition('.')
    if not dot or head not in tuple('c1' + c for c in C1) or not tail.startswith('g') or not tail[1:].isdigit():
        return None
    return head[2:], int(tail[1:])


def cell_of(c1, size, attn):
    for name, value in CELLS.items():
        if value == (attn, c1, size):
            return name
    return None


def arms(part=None):
    parts = (part,) if isinstance(part, str) else part
    rows = []
    for row_part, c1, cohort, index in ARM_TABLE:
        if parts is not None and row_part not in parts:
            continue
        band = band_of_index(index)
        size = BAND_SIZE[row_part][band]
        attn = ATTN[index]
        rows.append(dict(label=label(c1, size, cohort, index), selection=selection(c1, size), part=row_part, c1=c1,
                         overlap=C1[c1], defer_bind=DEFER_BIND[c1], c0='on', stage_local=STAGE_LOCAL, size=size,
                         cohort=cohort, index=index, band=band, extra=False, entry_limit=ENTRY_LIMIT[row_part][0],
                         attn=attn, cell=cell_of(c1, size, attn)))
    return rows


def labels(part=None):
    return [row['label'] for row in arms(part)]


def by_label():
    return {row['label']: row for row in arms()}


def attn_of(row):
    return ATTN.get(row.get('index'))


def matched_off(row):
    """The graded C1-off arm at the same size, cohort, index (so the same attn-meta selection): P -> R, MP -> M."""
    if row.get('part') not in GRADED:
        return None
    want = label('off', row['size'], row['cohort'], row['index'])
    return want if want in labels(GRADED) else None


def matched_on(row):
    """No plain C1-on arm in this table (``row`` kept for c1's call shape)."""
    del row
    return None


def matched_attn_off(row):
    """The arm at the same C1 selection, size, cohort and round with attn-meta off (M -> R, MP -> P, MP2 -> P2); None for
    an attn-meta-off arm and for MP band 2 (no partner)."""
    index = row.get('index')
    if not ATTN.get(index) or index == BAND2_INDEX:
        return None
    want = label(row['c1'], row['size'], row['cohort'], ROUNDS[False][ROUNDS[True].index(index)])
    return want if want in labels() else None


def band1_glob():
    return '[%s]' % ''.join(str(i) for i in BAND1_INDEXES)


def check():
    assert set(BAND_SIZE) == set(PARTS) == set(ENTRY_LIMIT)
    assert set(ATTN) == set(INDEXES) and all(0 <= i <= 9 for i in INDEXES)
    assert set(BAND1_INDEXES) == set(ROUNDS[False]) | set(ROUNDS[True]) and BAND2_INDEX not in BAND1_INDEXES
    assert all(ATTN[i] is a for a, rounds in ROUNDS.items() for i in rounds)
    order = [PARTS.index(part) for part, _, _, _ in ARM_TABLE]
    assert order == sorted(order), 'parts in order: core, then the tail'
    names = labels() + [d['label'] for d in DECODE_ARMS]
    assert len(names) == len(set(names)), 'repeated arm label'
    rows = arms()
    count = {}
    for r in rows:
        assert r['cell'] in CELLS, r
        key = (r['cell'], r['cohort'], r['band'])
        count[key] = count.get(key, 0) + 1
    want = {(c, 'live', 1): 2 for c in ('R', 'M', 'P', 'MP')}
    want.update({(c, 'solo', 1): 1 for c in ('R', 'M', 'P', 'MP')})
    want.update({('MP', 'live', 2): 1, ('P2', 'live', 1): 1, ('MP2', 'live', 1): 1})
    assert count == want, ('amendment A cells', count)
    for r in arms(GRADED):
        if r['c1'] == 'defer' and r['index'] != BAND2_INDEX:     # MP band 2 has no partner by design (amendment A)
            assert matched_off(r), 'graded C1-defer arm without its matched C1-off arm: ' + r['label']
            assert attn_of(by_label()[matched_off(r)]) is r['attn'], 'matched C1-off arm differs in attn-meta: ' + r['label']
        if r['attn'] and r['index'] != BAND2_INDEX:
            partner = matched_attn_off(r)
            assert partner and by_label()[partner]['cell'] == {'M': 'R', 'MP': 'P'}[r['cell']], r['label']
    assert matched_attn_off(by_label()[label('defer', 2, 'live', 3)]) == label('defer', 2, 'live', 0)
    sizes = [r['size'] for r in rows]
    assert sizes == sorted(sizes), 'the largest group is a lifetime high-water mark: sizes never decrease along the table'
    assert all(r['size'] == 2 for r in arms('tail')) and all(r['size'] == 1 for r in arms('core'))
    assert all(d['size'] == 1 and d['c1'] == 'off' for d in DECODE_ARMS) and [d['attn'] for d in DECODE_ARMS] == [False, True]
    # interleaving: within each band 1 live round every G1 cell once; round B reverses round A
    for rnd in (0, 1):
        cells = [cell for part, cell, cohort, r in ORDER if cohort == 'live' and r == rnd and part == 'core']
        assert sorted(cells) == sorted(_LIVE_A), cells
    a = [c for p, c, k, r in ORDER if k == 'live' and r == 0 and p == 'core']
    b = [c for p, c, k, r in ORDER if k == 'live' and r == 1 and p == 'core']
    assert b == a[::-1]
    assert [c for p, c, k, r in ORDER][-2:] == ['P2', 'MP2'], 'P2/MP2 last'
    return True


check()
