"""compose-2 build-2: a PP receipt refused by the ring must not strand the chunk (DIAG-box-1.md).

q18a-compose-2 run 1, arm b0.g2-solo-0: txn 34's receipt to sender 1 never left the engine.  ``_next_receipt`` set
``_receipt_inflight`` before ``_submit``; C0's headroom check raised Backpressure, ``advance`` swallowed it for the next
tick's retry, and every later tick found the guard set and answered "sent" with nothing in flight.  So: no commit, the
successor's deferred bind waits for ever, the client reads to its timeout, the auto-arm cycle owns the scheduler (every
pause refused), and the stop burns its rounds to ``stop_bound_exhausted``.

Real composition (``stage_net``, the C1 overlap stack); the refusal is the real ``_check_headroom`` with the ring's
answer forced False once, for one sender's receipt.  CPU only.
"""
import unittest
from unittest import mock

from reply_transport import Backpressure
import stage_progress
from test_e3_stage_local import Base
from test_e3_stage_overlap import OverlapBase


def refuse_receipt_once(driver, sender):
    """The next receipt to ``sender`` meets a ring without headroom; every other check is real."""
    seen = dict(refused=0)
    real = driver._check_headroom

    def check(ring, envelope):
        if not seen['refused'] and envelope[0] == 'flash_s5_receipt' and envelope[3] == sender:
            seen['refused'] += 1
            with mock.patch.object(stage_progress, 'ring_has_headroom', return_value=False), \
                    mock.patch.object(stage_progress, 'ring_limiting_readers', return_value=()):
                return real(ring, envelope)
        return real(ring, envelope)

    driver._check_headroom = check
    return seen


class ReceiptRefused(OverlapBase):
    def reach_refusal(self, network, seen):
        self.reach(network, lambda: seen['refused'] == 1)
        self.assertIsNone(network.driver._frames[0].chunk['_receipt_inflight'],
                          'a refused receipt is not in flight: nothing was sent, no reply will come')

    def test_the_refusal_is_the_ring_backpressure(self):
        network = self.network()
        real = network.driver._check_headroom
        with mock.patch.object(stage_progress, 'ring_has_headroom', return_value=False), \
                mock.patch.object(stage_progress, 'ring_limiting_readers', return_value=()):
            with self.assertRaises(Backpressure):
                real(None, ('flash_s5_receipt', (), {}, 0))

    def test_a_refused_receipt_is_retried_and_the_chunk_commits(self):
        for sender in (0, 1):
            with self.subTest(sender=sender):
                network = self.network()
                seen = refuse_receipt_once(network.driver, sender)
                network.run_request()
                self.assertEqual(seen['refused'], 1)
                self.assertEqual(network.worker_errors, [])
                self.assertEqual(network.controller.commits, network.expected_commits)
                self.assert_closed(network)

    def test_c0_alone_has_the_same_guard(self):
        # Stage-local with no overlap (the C0 driver): no successor, the same receipt path.
        for sender in (0, 1):
            with self.subTest(sender=sender):
                network = Base.network(self)
                seen = refuse_receipt_once(network.driver, sender)
                network.run_request()
                self.assertEqual(seen['refused'], 1)
                self.assertEqual(network.worker_errors, [])
                self.assertEqual(network.controller.commits, network.expected_commits)

    def test_the_frozen_driver_retries_too(self):
        # The frozen driver has no headroom check; a full ring reaches the same line through the sender's own check.
        for sender in (0, 1):
            with self.subTest(sender=sender):
                network = self.serial()
                driver = network.driver
                seen = dict(refused=0)
                real = driver._submit

                def submit(kind, envelope, ranks, **kwargs):
                    if kind == 'receipt' and envelope[3] == sender and not seen['refused']:
                        seen['refused'] += 1
                        raise Backpressure('ring full')
                    return real(kind, envelope, ranks, **kwargs)
                driver._submit = submit
                network.run_request()
                self.assertEqual(seen['refused'], 1)
                self.assertEqual(network.controller.commits, network.expected_commits)

    def test_a_stop_after_the_refusal_closes_within_its_bound(self):
        for sender in (0, 1):
            with self.subTest(sender=sender):
                network = self.network()
                driver = network.driver
                seen = refuse_receipt_once(driver, sender)
                self.reach_refusal(network, seen)
                commits = network.controller.commits
                driver.stop('abort')
                for _ in range(3000):
                    if not driver.active:
                        break
                    network.tick()
                self.assertEqual(network.controller.commits, commits)
                self.assert_closed(network)



if __name__ == '__main__':
    unittest.main()
