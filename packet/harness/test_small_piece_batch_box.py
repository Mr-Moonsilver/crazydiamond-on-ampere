#!/usr/bin/env python3
"""Host-gap batch pair, bundle-only: the arm record keeps the engine's small_piece_batch block through the real path.

Runs only inside the q2024p-j2-hgb bundle (it needs that bundle's patched n3j2_box.py; the repository has no copy), so
it is listed in the bundle's frozen CPU suites and not in selftest.py. Path: the batch driver's real _status_record
(the inherited base record is the declared leaf ``{'route': 'full'}``) -> n3j2_box.status_row (patched: marked
persistence line) -> compose5 -> compose4 rows -> tools/q2024p_j2_hgb_reader.batch_delta. compose5's _bound() is the
binding the replay arms run under; the off driver's row has no key at all.
"""
from pathlib import Path
import sys
import unittest
from unittest import mock

import test_small_piece_batch as spbt  # first: its imports put the engine sources on the path

HERE = Path(__file__).resolve().parent
for extra in (HERE.parent / 'n3tools', HERE.parent / 'tools'):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.append(str(extra))

import n3j2_box  # noqa: E402
import e3_compose4_box  # noqa: E402
import e3_compose5_box  # noqa: E402
import q2024p_j2_hgb_reader as reader  # noqa: E402

so = spbt.so


def reply(network):
    with mock.patch.object(so.OverlapSegmentDriver, '_status_record', lambda driver, now=None: {'route': 'full'}):
        return network.driver._status_record()


class BoxRecord(spbt.Base):
    def test_patched_status_row_is_the_replay_arm_binding(self):
        self.assertIn('persist_small_piece_batch(row, reply)', Path(n3j2_box.__file__).read_text())
        self.assertIs(e3_compose5_box.status_row, n3j2_box.status_row)
        with e3_compose5_box._bound():
            self.assertIs(e3_compose4_box.status_row, n3j2_box.status_row)

    def test_on_arm_rows_carry_the_block_and_the_reader_reads_its_delta(self):
        on = self.batched()
        before = n3j2_box.status_row('before', reply(on))
        self.run_overlapped(on)
        after = n3j2_box.status_row('after', reply(on))
        status = on.driver.small_batch_status()
        self.assertEqual(after['small_piece_batch'], status)
        self.assertEqual(before['small_piece_batch']['sent'], 0)
        delta = reader.batch_delta(before['small_piece_batch'], after['small_piece_batch'])
        self.assertEqual(delta['sent'], status['sent'])
        self.assertGreater(delta['sent'], 0)
        self.assertEqual(delta['commands_per_submission'], 2.0)
        self.assertEqual(delta['both_ran_without_yield'], status['second_applied'])
        self.assertEqual(delta['failures'], 0)
        self.assertEqual(delta['piece_stage']['batched']['n'] + delta['piece_stage']['unbatched']['n'],
                         sum(status['pieces'].values()))

    def test_off_arm_rows_have_no_key(self):
        off = self.plain()
        self.run_overlapped(off)
        row = n3j2_box.status_row('after', reply(off))
        self.assertNotIn('small_piece_batch', row)
        self.assertEqual(row, dict(e3_compose5_box._status_row('after', {'route': 'full'}),
                                   append_yield=row['append_yield'], route_generation=None, native_admission=None))


if __name__ == '__main__':
    unittest.main()
