"""The n3j2 table: inherited traffic/numerics, without two redundant decode heads."""
import compose5_arms as table

ARM_TABLE = (table.SMOKE_ROW,) + tuple(('core', name, 'live', 0) for name in table.O3_PAIR) + (
    ('core', 'sp', 'live', 0), ('core', 'n3', 'live', 0),
    ('core', 'n3', 'live', 1), ('core', 'sp', 'live', 1))


def configure():
    table.ARM_TABLE = ARM_TABLE
    table.DECODE_ARMS = ()
    table.LAUNCH = dict(table.LAUNCH, per_bulk=8)
    table.LAUNCH_ENV = dict(table.LAUNCH_ENV, FLASH_S5_APPEND_YIELD_PER_BULK=8)
    table.LAUNCH_PER_BULK = 8
    table.REPLAY = dict(table.REPLAY, REQUEST_IDS=True)
    assert table.REPLAY['DURATION'] == 480 and table.REPLAY['AGENTS'] == 8
    assert table.B24_BOUND == [97152, 2024] and table.DECODE_SMS == 24
    assert table.ROWCACHE_SLOTS == 6144
    # compose5_arms.check() describes the historical full table/per_bulk4.
    # This restricted wrapper owns the j2 shape and launch identity instead.
    assert table.LAUNCH == dict(max_new=8192,per_bulk=8,handback_steps=16,reentry_retries=4)
    assert table.LAUNCH_ENV['FLASH_S5_APPEND_YIELD_PER_BULK'] == 8
    return table

# composition branch (q2024p-j2-cx): FLASH_S5_DECODE_GROUPS sets the n3 selection's decode_merge
# (harness/n3_decode_groups; chunk, the default, is j2's table unchanged). The draft exports it.
import n3_decode_groups as _n3dg
configure = _n3dg.wrap_env(configure)

# brief 12 (q2024p-j2-cx): REPLAY_SEED_ORDER sets the seed each replay pair carries (harness/fixed_work_packet;
# plain, the default, is j2's dict unchanged). The draft exports it.
import fixed_work_packet as _fwp
configure = _fwp.wrap_env(configure)
