"""Packet-level decode groups for an ordinary four-arm j2 packet (composition branch, lead 12 Nov 2026).

``FLASH_S5_DECODE_GROUPS`` = ``chunk`` (default; also unset or empty) | ``off`` | ``off-small`` sets the ``n3``
selection's ``decode_merge`` field, the per-arm field the engine's ``native_stage_local`` already carries (the name
``FLASH_S5_DECODE_MERGE`` is the engine's install switch and stays as it is):
  * ``chunk``: today's table, untouched (the configure's own table object, ``n3`` at ``chunk``);
  * ``off``: stock's two parity cohorts during the native phase too;
  * ``off-small``: merged only while a piece of at least ``stage_overlap.OFF_SMALL_MIN_ROWS`` (1536) rows is open.
Only ``n3`` moves; ``sp`` stays ``off`` (native off) and every other selection keeps its value. The arm list, seeds,
launch and budget are j2's: unlike the dmp probe (``n3dmp_packet``) this adds no selection and no arm.

A packet applies it through ``wrap_env`` on the bundle's ``n3j2_packet.configure`` (a marked tail line written by the
packet builder), and the draft exports the setting before its first python, so every process the draft starts reads
the same value; the settings line carries it too (the engine identity records it; the engine itself never reads it).
An unknown value refuses at the first configure call, never silently.
"""
import os

ENV = 'FLASH_S5_DECODE_GROUPS'
MODES = ('chunk', 'off', 'off-small')
DEFAULT = 'chunk'
SELECTION = 'n3'


def mode_of(environ=None):
    value = (os.environ if environ is None else environ).get(ENV, '') or DEFAULT
    if value not in MODES:
        raise ValueError('%s must be one of %s, not %r' % (ENV, '|'.join(MODES), value))
    return value


def apply(table, mode):
    """Set the n3 selection's decode_merge; ``chunk`` leaves an unmodified table as it is."""
    assert mode in MODES, mode
    assert table.DECODE_MERGE.get(SELECTION) in MODES, ('n3 decode_merge', table.DECODE_MERGE.get(SELECTION))
    if table.DECODE_MERGE[SELECTION] != mode:
        # the configure does not reset DECODE_MERGE, so the value is set (also back to chunk) on every call
        table.DECODE_MERGE = dict(table.DECODE_MERGE, **{SELECTION: mode})
    return table


def check(table, mode):
    rows = table.arms()
    for r in rows:
        if r['c1'] == SELECTION:
            assert r['decode_merge'] == mode, (r['label'], r['decode_merge'], mode)
        elif r['native_off']:
            assert r['decode_merge'] == 'off', r['label']
    return True


def wrap_env(configure, environ=None):
    """The bundle's n3j2_packet.configure, then the decode-group mode named by FLASH_S5_DECODE_GROUPS."""
    def configured():
        mode = mode_of(environ)
        table = apply(configure(), mode)
        check(table, mode)
        return table
    configured.decode_groups_env = ENV
    configured.decode_groups_inner = configure
    return configured
