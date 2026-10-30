"""Real frontier/controller/allocation hook, synthetic allocator event inputs.

The separate pinned-image proof exercises real BlockPool/Mamba allocation.
"""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'patches/s5_engine'))
import async_handoff as ah
import scheduler_hooks as sh
from e3_types import TransactionId, E3ContractError, E3Pending


class FrontierRetirementTests(unittest.TestCase):
    def setUp(self):
        self.scheduler, self.pool = NS(), NS()
        self.guard = sh.SchedulerLeases().attach(self.scheduler, self.pool)
        self.frontier = ah.CommittedFrontier()
        self.controller = sh.PrefillAdmissionController(
            self.scheduler, leases=self.guard, frontier=self.frontier,
            engine_epoch=1, chunk_tokens=1024, ranks=(0, 1, 2, 3)).attach()

    def tx(self, seq):
        return TransactionId(1, 'request', 1, (seq-1)*1024, seq*1024, seq)

    def publish(self, seq, ids):
        self.frontier.reserve(self.tx(seq), ids)
        self.frontier.publish(self.tx(seq))

    def allocate(self, ids):
        sh.note_allocated(self.pool, [NS(block_id=i) for i in ids])

    def test_recycled_page_is_reserved_and_published_by_new_transaction(self):
        self.publish(1, (7, 8))
        self.allocate((7,))
        self.assertEqual(self.frontier.committed_blocks(), (8,))
        self.frontier.reserve(self.tx(2), (7,))
        with self.assertRaises(E3Pending):
            self.frontier.shareable((7,))
        self.frontier.publish(self.tx(2))
        self.assertEqual(self.frontier.shareable((7, 8)), (7, 8))

    def test_repeated_recycling_removes_old_transaction_entries(self):
        for seq in range(1, 98):
            self.allocate((7,))
            self.publish(seq, (7,))
            self.assertEqual(len(self.frontier._epoch), 1)
            self.assertEqual(len(self.frontier._published), 1)

    def test_cached_touch_does_not_retire_publication(self):
        self.publish(1, (7,))
        sh.before_touch(self.pool, [NS(block_id=7)])
        self.assertEqual(self.frontier.shareable((7,)), (7,))

    def test_held_page_refuses_entire_batch_without_retirement(self):
        self.publish(1, (7, 8))
        self.guard.reserve_blocks(self.tx(2), ah.LeaseCategory.SOURCE_HISTORY, (8,))
        with self.assertRaisesRegex(E3ContractError, 'leased block'):
            self.allocate((7, 8))
        self.assertEqual(self.frontier.committed_blocks(), (7, 8))

    def test_unpublished_page_refuses_entire_batch_without_retirement(self):
        self.publish(1, (7,))
        self.frontier.reserve(self.tx(2), (8,))
        with self.assertRaisesRegex(E3Pending, 'unpublished'):
            self.allocate((7, 8))
        self.assertEqual(self.frontier.committed_blocks(), (7,))
        self.assertEqual(self.frontier._epoch[self.tx(2)], (8,))

    def test_foreign_owner_refuses_without_retirement(self):
        self.publish(1, (7,))
        self.controller.leases = sh.SchedulerLeases()
        with self.assertRaisesRegex(E3ContractError, 'another owner'):
            self.allocate((7,))
        self.assertEqual(self.frontier.committed_blocks(), (7,))

    def test_invalid_id_refuses_without_partial_retirement(self):
        for bad in (None, True, 0, -1, '8'):
            with self.subTest(bad=bad):
                self.frontier = ah.CommittedFrontier()
                self.controller.frontier = self.frontier
                self.publish(1, (7,))
                with self.assertRaisesRegex(E3ContractError, 'positive ints'):
                    self.allocate((7, bad))
                self.assertEqual(self.frontier.committed_blocks(), (7,))

    def test_empty_allocation_preserves_publications(self):
        self.publish(1, (7,))
        self.allocate(())
        self.assertEqual(self.frontier.committed_blocks(), (7,))

    def test_detached_controller_is_not_used(self):
        self.publish(1, (7,))
        self.controller.detach()
        self.allocate((7,))
        self.assertEqual(self.frontier.committed_blocks(), (7,))
        self.controller.attach()
        self.allocate((7,))
        self.assertEqual(self.frontier.committed_blocks(), ())

    def test_default_off_pool_is_inert(self):
        self.publish(1, (7,))
        sh.note_allocated(NS(), [NS(block_id=7)])
        self.assertEqual(self.frontier.committed_blocks(), (7,))


if __name__ == '__main__':
    unittest.main()
