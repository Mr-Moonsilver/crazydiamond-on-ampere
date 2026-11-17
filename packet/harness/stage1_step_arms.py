"""q18a-stage1-step arm table: the one declared table (lead's build go, 3 Nov ~22:5x CET; results/hub/stage1-step/DESIGN-stage1-step.md section 6).
Change it here and nowhere else.

The engine is alpha-next-c1b's with L1 lazy PP receive (FLASH_S5_LAZY_RECV=1) and L2 decode merge (FLASH_S5_DECODE_MERGE=1) installed for the whole run, next to c1's switches. Every
arm is C1 defer (local, overlap and defer_bind true) and selects, explicitly, by typed control while paused, idle and FULL:
native_stage_local {local: true, overlap: true, defer_bind: true, lazy_recv: bool, decode_merge: 'off'|'chunk'|'always'}. Tokens, none of them c1's 'off' or 'on' (OBSERVE_ROLE):
  soff   lazy_recv false, decode_merge off: the base.
  sl1    lazy_recv true,  decode_merge off: L1.
  sl12   lazy_recv true,  decode_merge chunk: L1 + L2, decoders merged while the policy's native phase runs, 4 + 4 otherwise, so decode-only stays the ratio's baseline.
  sl12a  lazy_recv true,  decode_merge always: L1 + L2 for the whole arm, which gives the pure-decode rate at batch 8. Its decode-only baseline is batch 8 too: read absolute decode tok/s.
Fused QSA and burst progress are selected in every arm; the group size is 2 in every arm (native_layer_group).

A row is (part, selection, cohort, round index):
  part       'core' only: every arm makes the queue grade. No tail (the 'tail' part stays in PARTS for the inherited code; it has no rows).
  cohort     'live' keeps the 8 standing decoders through the prefill, 'solo' cancels them at the first native chunk (held-f).
  round      0, 1 and 3 are band 1 (65,445 fresh, then the append), 2 is band 2 (131,094 fresh, past the 97,152-row arena), as in alpha-next and alpha-next-c1: the same prompts and
             streams as the c1 record. The label is in the cache salt, so no arm hits another's prefix.

Order: matched within the same band, cohort and round. Round 0: soff sl1 sl12 solo, sl12 sl1 soff live. Round 1 the live trio reversed against round 0 (soff sl1 sl12), then solo sl12 soff
(solo has no decode during the prefill, so the solo arms are controls and sl1 solo runs once in band 1 and not in band 2). Band 2: solo soff sl12, live sl12 sl1 soff. Then the 'always' pair, last so a short
budget costs it and never a core comparison: round 3, band 1 live, soff then sl12a. The first arm of the run, soff.g2-solo-0, is the cold arm: c1's handling, unchanged and read as such (COLD_ARM).

Standing decoders and the API are alpha-next's (alpha_next_arms), through a view (e3_stage1_step_box.table_view). No arm asks for logprobs: native_auto_arm refuses them and the arm goes
stock (q18a-alpha-next-c1b's lesson).
"""
import alpha_next_arms as _alpha_next

STANDING_DECODERS = _alpha_next.STANDING_DECODERS
STANDING_GREEDY = _alpha_next.STANDING_GREEDY
GREEDY_TEMPERATURE = _alpha_next.GREEDY_TEMPERATURE
STANDING_SAMPLED = _alpha_next.STANDING_SAMPLED
BAND1_ROUNDS = 3
BAND1_REQUIRED_ROUNDS = 3
CORE_ROUNDS = (0, 1)                      # the matched rounds of band 1
ALWAYS_ROUND = 3                          # the soff/sl12a pair
BAND1_INDEXES = CORE_ROUNDS + (ALWAYS_ROUND,)
BAND2_INDEX = 2
TAIL_INDEX = 0                            # no row uses it: the inherited code reads the name
BAND_SIZE = {'core': {1: 2, 2: 2}, 'tail': {1: 2, 2: 2}}
ROUND0 = (('soff', 'solo'), ('sl1', 'solo'), ('sl12', 'solo'), ('sl12', 'live'), ('sl1', 'live'), ('soff', 'live'))
ROUND1 = (('soff', 'live'), ('sl1', 'live'), ('sl12', 'live'), ('sl12', 'solo'), ('soff', 'solo'))
BAND2 = (('soff', 'solo'), ('sl12', 'solo'), ('sl12', 'live'), ('sl1', 'live'), ('soff', 'live'))
ORDER_ALWAYS = (('soff', 'live'), ('sl12a', 'live'))


def _table():
    rows = [('core', s, cohort, 0) for s, cohort in ROUND0]
    rows += [('core', s, cohort, 1) for s, cohort in ROUND1]
    rows += [('core', s, cohort, BAND2_INDEX) for s, cohort in BAND2]
    rows += [('core', s, cohort, ALWAYS_ROUND) for s, cohort in ORDER_ALWAYS]
    return tuple(rows)


ARM_TABLE = _table()
PARTS = ('core', 'tail')
ENTRY_LIMIT = {'core': ('LAST_ARM_ENTRY_S', 'last_arm_entry'), 'tail': ('LAST_TAIL_ENTRY_S', 'last_tail_entry')}
GRADED = ('core',)
SELECTIONS = ('soff', 'sl1', 'sl12', 'sl12a')
LAZY_RECV = {'soff': False, 'sl1': True, 'sl12': True, 'sl12a': True}            # the lazy_recv value of native_stage_local per selection (every arm sends it)
DECODE_MERGE = {'soff': 'off', 'sl1': 'off', 'sl12': 'chunk', 'sl12a': 'always'}  # the decode_merge value (every arm sends it)
C1 = dict.fromkeys(SELECTIONS, True)                       # the overlap value per selection: C1 defer in every arm
DEFER_BIND = dict.fromkeys(SELECTIONS, True)               # the defer_bind value per selection: C1 defer in every arm
STAGE_LOCAL = True
COHORTS = ('live', 'solo')
INDEXES = tuple(sorted(BAND1_INDEXES + (BAND2_INDEX,)))
SIZES = (1, 2, 4)
COLD_ARM = 'soff.g2-solo-0'                # the run's first arm, c1's cold warm-up: read, never a verdict input


def band_of_index(index):
    return 1 if index in BAND1_INDEXES else 2 if index == BAND2_INDEX else None


def selection(sel, size):
    """The arm's selection token: selection and group size, no '-' (``sl1.g2``)."""
    return '%s.g%d' % (sel, size)


def label(sel, size, cohort, index):
    return '%s-%s-%d' % (selection(sel, size), cohort, index)


def parse_selection(token):
    """(selection, size) of a selection token, or None."""
    head, dot, tail = str(token).partition('.')
    if not dot or head not in LAZY_RECV or not tail.startswith('g') or not tail[1:].isdigit():
        return None
    return head, int(tail[1:])


def arms(part=None):
    """The table as rows, in order; one part (or a tuple of parts) only when ``part`` is given."""
    parts = (part,) if isinstance(part, str) else part
    rows = []
    for row_part, sel, cohort, index in ARM_TABLE:
        if parts is not None and row_part not in parts:
            continue
        band = band_of_index(index)
        size = BAND_SIZE[row_part][band]
        rows.append(dict(label=label(sel, size, cohort, index), selection=selection(sel, size), part=row_part, c1=sel,
                         lazy_recv=LAZY_RECV[sel], decode_merge=DECODE_MERGE[sel], overlap=C1[sel], defer_bind=DEFER_BIND[sel], c0='on',
                         stage_local=STAGE_LOCAL, size=size, cohort=cohort, index=index, band=band, extra=False,
                         entry_limit=ENTRY_LIMIT[row_part][0]))
    return rows


def labels(part=None):
    return [row['label'] for row in arms(part)]


def by_label():
    return {row['label']: row for row in arms()}


def matched_off(row):
    """The base arm matched to a selected arm: same size, cohort and round. None for a base arm, or when the table holds no base arm in that cell."""
    if row.get('part') not in GRADED or row.get('c1') == 'soff':
        return None
    want = label('soff', row['size'], row['cohort'], row['index'])
    return want if want in labels(GRADED) else None


def matched_on(row):
    """c1's name (a C1-defer arm's C1-on partner): this table has no C1-on arm."""
    return None


def band1_glob():
    return '[%s]' % ''.join(str(i) for i in BAND1_INDEXES)


def check():
    """The table's own invariants; raises on a malformed table so no queue text is generated from one."""
    assert set(BAND_SIZE) == set(PARTS) == set(ENTRY_LIMIT), BAND_SIZE
    assert BAND1_REQUIRED_ROUNDS == BAND1_ROUNDS == len(BAND1_INDEXES) and BAND2_INDEX not in BAND1_INDEXES
    names = labels()
    assert len(names) == len(set(names)) == 18, 'eighteen distinct arm labels'
    assert labels('tail') == [] and labels('core') == names, 'no tail'
    assert names[0] == COLD_ARM, 'the cold arm is the first arm'
    assert set(LAZY_RECV) == set(DECODE_MERGE) == set(C1) == set(DEFER_BIND) == set(SELECTIONS)
    assert [(r['c1'], r['cohort'], r['band']) for r in arms() if r['index'] == ALWAYS_ROUND] == [('soff', 'live', 1), ('sl12a', 'live', 1)], 'the always pair'
    assert names[-1] == label('sl12a', 2, 'live', ALWAYS_ROUND), 'the always arm is last'
    assert not any(r['c1'] == 'sl12a' and r['index'] != ALWAYS_ROUND for r in arms()), 'sl12a is the one arm'
    for row in arms():
        assert row['overlap'] is True and row['defer_bind'] is True and row['stage_local'] is True, 'C1 defer in every arm'
        assert type(row['lazy_recv']) is bool and row['decode_merge'] in ('off', 'chunk', 'always')
        assert row['size'] == 2, 'group size 2 in every arm'
        if row['c1'] != 'soff':
            assert matched_off(row), 'selected arm without its matched base arm: ' + row['label']
    for index in CORE_ROUNDS + (BAND2_INDEX,):
        assert {(r['c1']) for r in arms() if r['index'] == index and r['cohort'] == 'live'} == {'soff', 'sl1', 'sl12'}, ('live trio', index)
        assert {'soff', 'sl12'} <= {r['c1'] for r in arms() if r['index'] == index and r['cohort'] == 'solo'}, ('solo controls', index)
    return True


check()
