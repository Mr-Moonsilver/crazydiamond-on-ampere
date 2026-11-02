#!/usr/bin/env python3
"""C0 ring reading (assessment 4C.6): time unable to submit and the reader that limits reclamation, beside headroom_refused.

Real StagedSegmentDriver on harness/stage_net.RingStack (leaves: stage_fixture_leaves.json).  The ring readers are the
per-rank read positions of the leaf ring; freezing ranks 2 and 3 (stage 1) makes them the readers owing reads.  The new
fields are receipts only: counters in stage_status()['ring'] and one more field on the per-chunk stage line.  CPU only.
"""
import re
import unittest
from unittest import mock

from test_e3_stage_local import Base  # puts the engine sources on sys.path
import stage_progress as sp  # noqa: E402


class Ring(Base):
    def blocked(self, frozen):
        network = self.network()
        self.reach(network, lambda: network.driver._frames and self.stage(network, 1)['index'] >= 1)
        network.frozen = set(frozen)
        clock = iter(range(0, 10 ** 12, 1000))
        with mock.patch.object(sp.time, 'monotonic_ns', lambda: next(clock)):
            for _ in range(40):
                network.tick()
            network.frozen = set()
            for _ in range(20):
                network.tick()
        return network

    def test_blocked_time_and_limiting_readers_are_recorded(self):
        network = self.blocked({2, 3})
        driver = network.driver
        ring = driver.stage_status()['ring']
        refused = driver.stage_counters['headroom_refused']
        self.assertGreater(refused, 0)
        self.assertGreaterEqual(ring['episodes'], 1)
        self.assertGreater(ring['blocked_ns'], 0)
        self.assertLessEqual(ring['blocked_max_ns'], ring['blocked_ns'])
        # Only the frozen stage owes reads: every refusal names rank 2 or 3, never stage 0.
        self.assertTrue(set(ring['limiting']) <= {2, 3}, ring)
        self.assertEqual(max(ring['limiting'].values()), refused)
        self.assertFalse(ring['blocked_open'])

    def test_one_frozen_reader_is_the_sole_limit(self):
        network = self.blocked({3})
        ring = network.driver.stage_status()['ring']
        if network.driver.stage_counters['headroom_refused']:
            self.assertEqual(set(ring['limiting']), {3})
            self.assertEqual(ring['limiting_sole'], {3: network.driver.stage_counters['headroom_refused']})

    def test_no_refusal_no_ring_time(self):
        network = self.network()
        network.run_request()
        ring = network.driver.stage_status()['ring']
        self.assertEqual(network.driver.stage_counters['headroom_refused'], 0)
        self.assertEqual((ring['blocked_ns'], ring['episodes'], ring['limiting']), (0, 0, {}))

    def test_stage_line_keeps_the_c0_grader_prefix(self):
        import e3_alpha_c0_box as box
        line = ('flash-s5 stage-local txn=1 chunk=0-4 commands=6,6 both_stages_ns=0 first_ns=1,2 last_ns=3,4 '
                'local_release_ns=5,6 headroom_refused=2 headroom_blocked_ns=1500')
        self.assertIsNotNone(box.STAGE_RE.search(line))
        self.assertEqual(int(re.search(r'headroom_blocked_ns=(\d+)', line).group(1)), 1500)

    def test_the_reading_never_decides(self):
        """Mutant parity: the limiting-reader query raising changes nothing the driver does."""
        clean = self.blocked({2, 3})
        with mock.patch.object(sp, 'ring_limiting_readers', side_effect=RuntimeError('leaf')):
            broken = self.blocked({2, 3})
        def counts(network):   # both_stages_ns is real elapsed time before the leaf clock; not a decision
            return {k: v for k, v in network.driver.stage_counters.items() if k != 'both_stages_ns'}
        self.assertEqual(counts(clean), counts(broken))
        self.assertEqual(clean.driver.controller.commits, broken.driver.controller.commits)
        self.assertEqual(broken.driver.stage_status()['ring']['limiting'], {})


if __name__ == '__main__':
    unittest.main()
