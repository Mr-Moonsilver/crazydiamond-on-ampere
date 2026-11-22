"""Host-gap batch (FLASH_S5_SMALL_PIECE_BATCH), observational only: keep the engine's block on an arm status row.

The arm status rows keep a fixed key list, so the engine's ``small_piece_batch`` status block (present only when the
flag is 1) would be dropped from the record. ``persist`` copies it onto the row unchanged. When the engine has no
block (the flag is 0, or an older engine) the row is returned untouched, so an off record keeps its exact shape. A read
of an already-fetched reply: no engine call, no sync, no write beyond the row the caller already keeps.
"""

KEY = 'small_piece_batch'


def persist(row, reply):
    """``row`` with the reply's small_piece_batch block beside its fixed keys; untouched without one."""
    if type(row) is dict and type(reply) is dict and KEY in reply:
        row[KEY] = reply[KEY]
    return row


__all__ = ['KEY', 'persist']
