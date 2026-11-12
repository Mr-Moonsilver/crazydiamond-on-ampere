#!/usr/bin/env python3
"""Composition branch (lead 12 Nov 2026): lever pairs that share a code path, run together on the real engine code.

Y1 (FLASH_S5_YIELD_NO_APPEND_OWNER) x Y2 (FLASH_S5_YIELD_WINDOW), on test_yield_window's rig (the real auto-arm
policy, engine driver and scheduler hooks over the CPU worker stack; O3 not installed): Y1 decides at the yield offer,
before Y2's window exists, from the owner's rows left past the transaction frontier.
  * owner rows left > max_new: Y1 lets the yield through, and the window runs exactly as with Y1 off (same window
    counters, the same append-yield event kinds, the same receipts), Y1 counting no junction;
  * owner rows left <= max_new: Y1 refuses the yield, so no window opens (Y2's counters stay zero), the junction is
    counted once, and the owner keeps its lane.
The rig's bulk has 7 rows left at its first junction (prompt 12, frontier 4, native end 11); max_new 6 and 8 put it
on either side. CPU only; no box, GPU or torch device.
"""
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
for extra in (HERE, HERE.parent / 'patches' / 's5_engine'):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.append(str(extra))

import test_yield_window as yw  # noqa: E402


class Y1WithY2(unittest.TestCase):
    def flow(self, y1, max_new):
        log = yw.Lines(self)
        rig = yw.window_rig(max_new=max_new)
        rig.c.yield_no_append_owner = y1
        yw.served_first(rig)
        yw.to_window_append(rig)
        # receipts without their clock stamps (wall / monotonic values) and measured gaps
        receipts = [{k: v for k, v in r.items() if k != 'gap_ms' and not (type(v) in (int, float) and v > 10 ** 6)}
                    for r in log.receipts()]
        return rig, yw.counts(rig), yw.kinds(rig), receipts

    def test_owner_past_max_new_window_unchanged(self):
        _, off_counts, off_kinds, off_receipts = self.flow(False, 6)
        rig, on_counts, on_kinds, on_receipts = self.flow(True, 6)
        self.assertEqual((off_counts['appends'], off_counts['reentries_avoided']), (1, 1), off_counts)   # in the window
        self.assertEqual(on_counts, off_counts)
        self.assertEqual(on_kinds, off_kinds)
        self.assertEqual(on_receipts, off_receipts)
        self.assertEqual(rig.c.yield_owner_counts['junctions'], 0, rig.c.yield_owner_counts)
        self.assertEqual(rig.d.append_yield_phase(), 'append')

    def test_owner_within_max_new_no_window(self):
        yw.Lines(self)
        rig = yw.window_rig(max_new=8)
        rig.c.yield_no_append_owner = True
        yw.ay.run_to_yield(rig)
        with self.assertRaises(AssertionError):
            yw.ay.to_append(rig)                  # the yield never starts: Y1 refuses at every offer
        self.assertIsNone(rig.d.append_yield_phase())
        self.assertEqual(rig.c.yield_owner_counts['junctions'], 1, rig.c.yield_owner_counts)
        self.assertEqual(rig.c.yield_owner_counts['errors'], 0)
        self.assertEqual({k: yw.counts(rig)[k] for k in ('windows', 'appends', 'reentries_avoided')},
                         dict(windows=0, appends=0, reentries_avoided=0))

    def test_control_without_y1_the_same_rig_yields(self):
        yw.Lines(self)
        rig = yw.window_rig(max_new=8)
        rig.c.yield_no_append_owner = False
        yw.served_first(rig)
        yw.to_window_append(rig)
        self.assertEqual(yw.counts(rig)['appends'], 1)


if __name__ == '__main__':
    unittest.main()
