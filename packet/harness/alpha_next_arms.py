"""q18a-alpha-next arm table: the one declared table (lead approved 2 Nov 13:28 CET, rounds and greedy streams 13:35). Change it
here and nowhere else.

BAND1_ROUNDS, the orders and BAND_SIZE make ARM_TABLE, the arm order. Every other place reads them: the arm CLI and its refusal of
an arm not in the table (e3_alpha_next_box), the grade's declared labels and matched pairs, the gate's arm count and expected
runtime (alpha_next_gate), the draft's loop list and band 1 round pattern (tools/alpha_next_queue.py regenerates the draft from
this file; ``--check`` and test_alpha_next_queue.Derivation fail on a stale draft), the analyser's groups and the brief's table.

A row is (part, C0 selection, cohort, round index):
  part          'core' arms make the queue grade. 'extra' arms are core arms of the band 1 rounds past BAND1_REQUIRED_ROUNDS:
                entered only while the core entry limit allows (lead, 13:35: "if the budget allows"); one that ran is graded
                exactly as a core arm, one not entered for the budget is a read-only NOT_RUN and is not required. 'tail' arms run
                after the core grade is written and are reads only: the queue verdict does not depend on them, except for crash
                accounting (the death watch and scan cover every arm) and leak accounting (instruments-end and end-full-idle run
                after the tail and are counted).
  C0 selection  'on' selects the stage-local path (typed native_stage_local {local: true}), 'off' the serial path {local: false}.
  cohort        'live' keeps the standing decoders through the prefill, 'solo' cancels them at the first native chunk (held-f).
  round index   BAND1_INDEXES are band 1 (65,445 fresh, then the append), 2 is band 2 (131,094 fresh, past the 97,152-row arena),
                as in Step A and Step B; band 1's third round takes index 3 because 2 is band 2 in every Step A/B label. The index
                is in the label and the label is in the cache salt: no arm hits another's prefix.
Fused QSA and burst progress are selected in every arm. The group size comes from BAND_SIZE[part][band] and is sent per arm by
the typed native_layer_group control, so parts and bands may run different sizes; the label carries the size.

Order: band 1 ABBA C0-on against C0-off, live and solo, rounds 0 and 1 at size 2; band 2 matched at size 2; then the extra band 1
round(s) at size 2 (C0's on/off effect at g2 is about 4 % of a chunk, so a third round if the budget allows; it goes after band 2
so a short budget costs the extra round, never band 2); then the g4 tail, C0-on only and last: band 1 live and solo over rounds 0
and 1, then band 2 live and solo (never run before). The tail must stay last: ``group_layers_max`` in the engine's status is a
lifetime high-water mark (q18a-step-b-e: every g2 arm after a g4 arm read largest-group-is-not-the-size).

Standing decoders (lead, 13:35): 8, the band manifests' cohort. Streams STANDING_GREEDY run greedy (temperature 0, the arm sets it
on those streams' request bodies only) and carry the numerics digest check; the others keep the manifest's sampling (temperature
0.7, fixed seed) and carry the decode-ratio reading. Not included: a solo idle=wait, replies=lean arm (Step B's arm sends
native_progress {mode} only; see docs/e3-briefs/q18a-alpha-next-arms.md).
"""
STANDING_DECODERS = 8                     # the band manifests' standing cohort (held-f's), not set here
STANDING_GREEDY = (0, 1)                  # greedy streams: the digest check
GREEDY_TEMPERATURE = 0.0
STANDING_SAMPLED = tuple(i for i in range(STANDING_DECODERS) if i not in STANDING_GREEDY)   # the decode-ratio reading
BAND1_ROUNDS = 3                          # band 1 core rounds (lead, 13:35: 3 if the budget allows)
BAND1_REQUIRED_ROUNDS = 2                 # rounds past this are 'extra'
BAND1_INDEXES = (0, 1, 3, 4, 5)[:BAND1_ROUNDS]
BAND2_INDEX = 2
BAND_SIZE = {'core': {1: 2, 2: 2},        # layer group size per part and band, sent per arm through native_layer_group {size}
             'extra': {1: 2, 2: 2},
             'tail': {1: 4, 2: 4}}
ORDER_A = (('off', 'solo'), ('on', 'solo'), ('on', 'live'), ('off', 'live'))     # ABBA: A, then B = A reversed
ORDER_BAND2 = (('on', 'live'), ('off', 'live'), ('off', 'solo'), ('on', 'solo'))
ORDER_TAIL = ((0, 'live'), (0, 'solo'), (1, 'solo'), (1, 'live'), (BAND2_INDEX, 'live'), (BAND2_INDEX, 'solo'))


def _table():
    rows = []
    for k, index in enumerate(BAND1_INDEXES[:BAND1_REQUIRED_ROUNDS]):
        rows += [('core', c0, cohort, index) for c0, cohort in (ORDER_A if k % 2 == 0 else ORDER_A[::-1])]
    rows += [('core', c0, cohort, BAND2_INDEX) for c0, cohort in ORDER_BAND2]
    for k, index in enumerate(BAND1_INDEXES[BAND1_REQUIRED_ROUNDS:], BAND1_REQUIRED_ROUNDS):
        rows += [('extra', c0, cohort, index) for c0, cohort in (ORDER_A if k % 2 == 0 else ORDER_A[::-1])]
    rows += [('tail', 'on', cohort, index) for index, cohort in ORDER_TAIL]
    return tuple(rows)


ARM_TABLE = _table()

PARTS = ('core', 'extra', 'tail')
GRADED = ('core', 'extra')                # the parts the core grade (the queue verdict) covers
C0 = {'on': True, 'off': False}
COHORTS = ('live', 'solo')
INDEXES = tuple(sorted(BAND1_INDEXES + (BAND2_INDEX,)))
SIZES = (1, 2, 4)                         # the sizes Step B's engagement rule knows (e3_step_b_box.SIZES)


def band_of_index(index):
    """Band 1 for BAND1_INDEXES, band 2 for index 2 (Step A's rule, extended to band 1's third round)."""
    return 1 if index in BAND1_INDEXES else 2 if index == BAND2_INDEX else None


def selection(c0, size):
    """The arm's selection token: C0 and group size, no '-' (``c0on.g2``)."""
    return 'c0%s.g%d' % (c0, size)


def label(c0, size, cohort, index):
    return '%s-%s-%d' % (selection(c0, size), cohort, index)


def parse_selection(token):
    """(c0, size) of a selection token, or None."""
    head, dot, tail = str(token).partition('.')
    if not dot or head not in ('c0on', 'c0off') or not tail.startswith('g') or not tail[1:].isdigit():
        return None
    return head[2:], int(tail[1:])


def arms(part=None):
    """The table as rows, in order; one part (or a tuple of parts) only when ``part`` is given."""
    parts = (part,) if isinstance(part, str) else part
    rows = []
    for row_part, c0, cohort, index in ARM_TABLE:
        if parts is not None and row_part not in parts:
            continue
        band = band_of_index(index)
        size = BAND_SIZE[row_part][band]
        rows.append(dict(label=label(c0, size, cohort, index), selection=selection(c0, size), part=row_part, c0=c0,
                         stage_local=C0[c0], size=size, cohort=cohort, index=index, band=band, extra=row_part == 'extra'))
    return rows


def labels(part=None):
    return [row['label'] for row in arms(part)]


def by_label():
    return {row['label']: row for row in arms()}


def matched_off(row):
    """The graded C0-off arm matched to a C0-on arm: same size, cohort and round. None when the table has none."""
    want = label('off', row['size'], row['cohort'], row['index'])
    return want if want in labels(GRADED) else None


def band1_glob():
    """The bash pattern of the band 1 round indexes (the draft's band 1 manifest choice)."""
    return '[%s]' % ''.join(str(i) for i in BAND1_INDEXES)


def check():
    """The table's own invariants; raises on a malformed table so no queue text is generated from one."""
    assert set(BAND_SIZE) == set(PARTS), BAND_SIZE
    for sizes in BAND_SIZE.values():
        assert set(sizes) == {1, 2} and all(size in SIZES for size in sizes.values()), BAND_SIZE
    assert 1 <= BAND1_REQUIRED_ROUNDS <= BAND1_ROUNDS <= 5 and BAND2_INDEX not in BAND1_INDEXES
    assert all(0 <= i <= 9 for i in INDEXES), 'round indexes are one digit (the draft matches them with one bash glob)'
    assert set(STANDING_GREEDY) < set(range(STANDING_DECODERS)) and STANDING_GREEDY and STANDING_SAMPLED
    order = [PARTS.index(part) for part, _, _, _ in ARM_TABLE]
    assert order == sorted(order), 'parts in order: core, then extra, then the tail (the core grade runs before the tail)'
    names = labels()
    assert len(names) == len(set(names)), 'repeated arm label (a tail arm at a core size and selection?)'
    for row in arms(GRADED):
        if row['c0'] == 'on':
            assert matched_off(row), 'graded C0-on arm without its matched C0-off arm: ' + row['label']
    assert [r for r in arms('core') if r['c0'] == 'off'], 'the core needs C0-off controls'
    sizes = [r['size'] for r in arms()]
    assert sizes == sorted(sizes), 'the largest group is a lifetime high-water mark: sizes never decrease along the table'
    return True


check()
