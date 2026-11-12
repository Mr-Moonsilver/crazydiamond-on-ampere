"""decode-merge probe table (12 Nov 2026): j2's table plus one probe selection, ABBA on one seed.

Exploratory. On top of the j2 table (the bundle's ``n3j2_packet.configure``, unchanged and run first) this adds one
selection, a copy of ``n3``'s row in every per-selection table but ``DECODE_MERGE``:
  * variant ``off``:      ``nm`` with ``decode_merge='off'`` (stock's two parity cohorts during the native phase too);
  * variant ``offsmall``: ``ns`` with ``decode_merge='off-small'`` (merged only while a piece of >= 1536 rows is open).
The replay is ``sp | n3 X X n3`` with X the probe selection, all five arms on seed 1 (index 1 is mapped to seed 1, so the
second chunk and probe arms get distinct labels and the same draws). The head (smoke, pf po pg) is j2's. Every row but the
declared probe rows keeps ``chunk``; ``sp`` keeps ``off`` (native off). No other field differs from ``n3``.
"""

VARIANTS = {'off': ('nm', 'off'), 'offsmall': ('ns', 'off-small')}
PROBE_SEED = 1
SELECTION_TABLES = ('KIND', 'YIELD', 'PLE_PRIVATE', 'PLE_WITNESS', 'PLAN', 'NATIVE_HOLD', 'NATIVE_RESIDUE',
                    'TERMINAL_SUCCESSOR', 'O3_SPLIT', 'AHEAD_NOSYNC', 'K2_SETTLE', 'NATIVE_OFF', 'ATTN', 'LAZY_RECV',
                    'DECODE_MERGE', 'BOUND', 'MATCHED_OFF')


def probe(variant):
    assert variant in VARIANTS, variant
    return VARIANTS[variant]


def replay_rows(sel):
    return (('core', 'sp', 'live', 0), ('core', 'n3', 'live', 0), ('core', sel, 'live', 0),
            ('core', sel, 'live', 1), ('core', 'n3', 'live', 1))


def extend(table, variant):
    """Add the probe selection to the configured j2 table (module ``compose5_arms``) once, then (every call: j2's
    configure resets ARM_TABLE each time it runs) set the ABBA rows and the seed map."""
    sel, mode = probe(variant)
    if sel not in table.SELECTIONS:
        _add_selection(table, sel, mode)
    assert table.DECODE_MERGE[sel] == mode
    head = tuple(r for r in table.ARM_TABLE if r[1] not in ('sp', 'n3', sel))
    table.ARM_TABLE = head + replay_rows(sel)
    table.REPLAY_SEEDS = {0: PROBE_SEED, 1: PROBE_SEED}
    check(table, variant)
    return table


def _add_selection(table, sel, mode):
    for name in SELECTION_TABLES:
        values = getattr(table, name)
        assert 'n3' in values, name
        setattr(table, name, dict(values, **{sel: values['n3']}))
    table.DECODE_MERGE = dict(table.DECODE_MERGE, **{sel: mode})
    table.SELECTIONS = tuple(table.SELECTIONS) + (sel,)
    table.ONS = tuple(table.ONS) + (sel,)


def check(table, variant):
    sel, mode = probe(variant)
    rows = table.arms()
    labels = [r['label'] for r in rows]
    assert len(labels) == len(set(labels)), 'distinct labels'
    replay = [r for r in rows if r['kind'] == 'replay']
    assert [r['label'] for r in replay] == ['sp.g2-live-0', 'n3.g2-live-0', sel + '.g2-live-0', sel + '.g2-live-1',
                                           'n3.g2-live-1'], 'sp | n3 X X n3'
    assert {table.replay_config(r)['SEED'] for r in replay} == {PROBE_SEED}, 'one seed for every replay arm'
    assert [r['label'] for r in rows[:4]] == [table.COLD_ARM, 'pf.g2-live-0', 'po.g2-live-0', 'pg.g2-live-0'], "j2's head"
    n3 = table.reference_row('n3', 0)
    x = table.reference_row(sel, 0)
    differ = sorted(k for k in n3 if n3[k] != x[k])
    assert differ == ['c1', 'decode_merge', 'label', 'selection'], ('the probe row is n3 but decode_merge', differ)
    for r in rows:
        if r['c1'] == sel:
            assert r['decode_merge'] == mode
        elif r['native_off']:
            assert r['decode_merge'] == 'off'
        else:
            assert r['decode_merge'] == 'chunk', ('chunk on every other native row', r['label'])
    assert table.REPLAY['DURATION'] == 480 and table.REPLAY['AGENTS'] == 8
    return True


def wrap(configure, variant):
    """The bundle's n3j2_packet.configure, then the probe; safe to call again (configure runs from several modules)."""
    def configured():
        return extend(configure(), variant)
    configured.n3dmp_variant = variant
    return configured
