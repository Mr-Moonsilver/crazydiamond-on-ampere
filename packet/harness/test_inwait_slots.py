"""inwait build-2: LaneTables.slot_mappings_fast against the loop, value for value, with the real torch (pinned image); the switch and its default. DESIGN-2."""
import itertools
import types
import unittest

import torch

import lane_chunk as lc


def tables(kbs, bpk, enabled):
    t = lc.LaneTables.__new__(lc.LaneTables)
    t.torch, t.num_groups = torch, len(kbs)
    t.kernel_block_sizes, t.blocks_per_kv_block, t.slot_mapping_enabled = tuple(kbs), tuple(bpk), tuple(enabled)
    return t


def admission(start, end, rows):
    return types.SimpleNamespace(chunk_start=start, chunk_end=end, num_tokens=end - start, block_tables=tuple(rows))


class Switch(unittest.TestCase):
    def setUp(self):
        self.addCleanup(lc.select_slots_fast, False)
        lc.select_slots_fast(False)

    def both(self, t, a, pad):
        lc.select_slots_fast(False)
        calls = lc.SLOTS_FAST['calls']
        slow = t.slot_mappings(a, pad)
        self.assertEqual(lc.SLOTS_FAST['calls'], calls, 'off: the loop')
        lc.select_slots_fast(True)
        fast = t.slot_mappings(a, pad)
        self.assertEqual(lc.SLOTS_FAST['calls'], calls + 1, 'on: the tensor path')
        return slow, fast

    def test_equal_to_the_loop(self):
        shapes = [((16, 1), (1, 1), (True, False)), ((4, 4, 64), (1, 2, 4), (True, True, True)), ((1,), (1,), (True,)), ((1024, 16), (2, 1), (False, True))]
        spans = [(0, 1024), (1024, 2048), (65000, 65445), (3, 4), (15, 17)]
        for (kbs, bpk, enabled), (start, end), pad in itertools.product(shapes, spans, (-1, 0)):
            rows = [[7 + g + 3 * i for i in range(end // max(k, 1) // b + 2)] for g, (k, b) in enumerate(zip(kbs, bpk))]
            with self.subTest(kbs=kbs, bpk=bpk, start=start, end=end, pad=pad):
                slow, fast = self.both(tables(kbs, bpk, enabled), admission(start, end, rows), pad)
                self.assertEqual(slow.dtype, fast.dtype)
                self.assertEqual(slow.shape, fast.shape)
                self.assertTrue(torch.equal(slow, fast))

    def test_a_short_row_raises_the_same_error(self):
        t, a = tables((4,), (1,), (True,)), admission(0, 64, [[1, 2]])
        for on in (False, True):
            lc.select_slots_fast(on)
            with self.assertRaisesRegex(lc.LaneChunkError, 'too short'):
                t.slot_mappings(a, -1)

    def test_a_disabled_group_needs_no_row(self):
        slow, fast = self.both(tables((4,), (1,), (False,)), admission(0, 8, [[]]), -1)
        self.assertTrue(torch.equal(slow, fast))
        self.assertEqual(fast.tolist(), [[-1] * 8])

    def test_off_by_default_and_typed(self):
        self.assertIs(lc.SLOTS_FAST['selected'], False)
        for bad in (1, 'on', None):
            with self.assertRaises(lc.LaneChunkError):
                lc.select_slots_fast(bad)


if __name__ == '__main__':
    unittest.main()
