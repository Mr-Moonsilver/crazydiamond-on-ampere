#!/usr/bin/env python3
"""Stop after a refused continuation open, on the real driver, bridges, dispatchers and transports.

ENGINE finding from the C0 builder (1 Nov 2026): a row is retained on the workers after chunk n, the ring refuses chunk
n+1's open, then a stop arrives.  On c3f1f64 the driver cancels n+1, each worker answers quarantined because it still holds
n's row, and native service needs a restart.  The fix closes the retained row first, then the chunk retires as never
submitted through the ordinary cancel.  Controls show the neighbouring paths unchanged.  Leaves:
harness/alpha_stopfix_fixture_leaves.json.  No GPU, CUDA, NCCL, MQ or model forward; no pass here is native evidence.
"""
import unittest

from alpha_stopfix_net import RingStack, ed, stage_of_command

STOPS = ('abort', 'cancel', 'pause')


class Base(unittest.TestCase):
    def network(self, **kwargs):
        network = RingStack(**kwargs)
        self.addCleanup(network.close_network)
        return network

    def reach(self, network, until, limit=1000):
        for _ in range(limit):
            if until():
                return
            network.tick()
        self.fail('network did not reach the expected state')

    def between_chunks(self):
        network = self.network()
        self.reach(network, lambda: network.controller.commits == 1)
        self.assertIsNotNone(network.driver._retained)
        network.admit_next()
        return network

    def fill_ring(self, network):
        network.frozen = {0, 1, 2, 3}
        while network.ring.writable():
            network.stock_send()

    def refused_open(self):
        """The finding's state: chunk 2 begun, its open refused, chunk 1's row retained on every worker."""
        network = self.between_chunks()
        self.fill_ring(network)
        driver = network.driver
        driver.advance()
        self.assertEqual((driver.phase, driver.tx.txn_seq, driver._inflight, driver.error), ('open', 2, {}, None))
        self.assertEqual(driver._retained['transaction'].txn_seq, 1)
        return network

    def finish(self, network, limit=300):
        network.frozen = set()
        network.pump()
        driver = network.driver
        for _ in range(limit):
            if not driver.active:
                return
            network.tick()
        self.fail('the stop did not finish')

    def assert_closed(self, network):
        driver = network.driver
        self.assertEqual(network.worker_errors, [])
        self.assertIsNone(driver.error)
        self.assertFalse(driver.active)
        self.assertIsNone(driver.tx)
        self.assertEqual(driver._inflight, {})
        self.assertIsNone(driver._retained)
        self.assertIsNone(network.controller.parked)
        self.assertTrue(network.controller.leases.shutdown()['clean'])
        self.assertEqual(network.transport._calls, {})
        for rank, worker in network.workers.items():
            bridge = worker._flash_s5_bridge
            self.assertIsNone(bridge.transaction, rank)
            self.assertIsNone(bridge._retained, rank)
            self.assertIsNone(worker.dispatcher.transaction, rank)
            self.assertIsNone(worker.dispatcher.quarantined, rank)

    def commands(self, network):
        """The S5 commands on the ring in order, as (method, transaction)."""
        rows = []
        for envelope in network.ring.history:
            method, message, _ = stage_of_command(envelope)
            if method in ('flash_s5_open', 'flash_s5_cancel'):
                transaction = message.admission.transaction if method == 'flash_s5_open' else message.transaction
                rows.append((method, transaction.txn_seq))
        return rows

    def after_row_one(self, rows):
        """Commands from the last open of chunk 1 on."""
        last = max(i for i, row in enumerate(rows) if row == ('flash_s5_open', 1))
        return rows[last + 1:]


class Fix(Base):
    def test_each_stop_closes_the_retained_row_then_retires_the_chunk_as_never_submitted(self):
        for kind in STOPS:
            with self.subTest(stop=kind):
                network = self.refused_open()
                driver = network.driver
                driver.stop(kind)
                self.assertIsNone(driver.error)
                self.finish(network)
                self.assert_closed(network)
                # No worker ever saw chunk 2's open; the row closes before chunk 2's own cancel.
                rows = self.after_row_one(self.commands(network))
                self.assertNotIn(('flash_s5_open', 2), rows)
                self.assertEqual(rows[0], ('flash_s5_cancel', 1))
                self.assertEqual(set(rows), {('flash_s5_cancel', 1), ('flash_s5_cancel', 2)})
                self.assertLess(max(i for i, row in enumerate(rows) if row[1] == 1),
                                min(i for i, row in enumerate(rows) if row[1] == 2))
                kinds = [entry['kind'] for entry in driver._journal]
                self.assertIn('request_row_closed', kinds)
                self.assertNotIn('request_row_continued', kinds[kinds.index('request_row_closed'):])
                self.assertEqual((network.controller.commits, network.controller.cancels), (1, 1))

    def test_final_state_equals_a_stop_before_the_chunk_began(self):
        """The existing between-chunks path (tx unset) and this one leave the same persistent state."""
        for kind in STOPS:
            with self.subTest(stop=kind):
                before = self.between_chunks()
                before.driver.stop(kind)
                self.finish(before)
                self.assert_closed(before)
                refused = self.refused_open()
                refused.driver.stop(kind)
                self.finish(refused)
                self.assert_closed(refused)
                self.assertEqual(refused.persistent_state(), before.persistent_state())

    def test_a_ring_still_full_refuses_the_row_close_without_losing_it(self):
        network = self.refused_open()
        driver = network.driver
        driver.stop('abort')
        for _ in range(5):
            driver.advance()
        self.assertEqual((driver._inflight, driver.error), ({}, None))
        self.assertEqual(driver._retained['transaction'].txn_seq, 1)
        self.assertEqual(driver.tx.txn_seq, 2)
        self.finish(network)
        self.assert_closed(network)

    def test_a_refusal_after_the_row_closed_still_retires_the_chunk(self):
        """The later boundary: the row has closed and the ring refuses the chunk's own cancel."""
        network = self.refused_open()
        driver = network.driver
        driver.stop('abort')
        network.frozen = set()
        network.pump()
        self.reach(network, lambda: driver._retained is None)
        self.assertEqual(driver.tx.txn_seq, 2)
        self.fill_ring(network)
        for _ in range(5):
            driver.advance()
        self.assertEqual((driver._inflight, driver.error, driver.tx.txn_seq), ({}, None, 2))
        self.finish(network)
        self.assert_closed(network)
        self.assertEqual(network.controller.cancels, 1)

    def test_a_quarantined_row_close_still_fails_closed_with_ownership_kept(self):
        network = self.refused_open()
        driver = network.driver

        def refuse():
            raise RuntimeError('close refused')
        network.workers[0].dispatcher.close_request = refuse
        driver.stop('abort')
        network.frozen = set()
        network.pump()
        with self.assertRaisesRegex(ed.DriverError, 'worker quarantined the retained row; ownership kept'):
            for _ in range(50):
                network.tick()
        self.assertEqual(driver.error, 'native driver failed; ownership retained')
        self.assertIsNotNone(driver._retained)
        self.assertTrue(driver.active)


class Controls(Base):
    """Neighbouring paths the new branch must not reach; each also passes on c3f1f64."""

    def test_a_first_chunk_refused_open_closes_through_never_open(self):
        network = self.network()
        driver = network.driver
        # Run until the split route's reply is ready, so the next advance begins the chunk with the ring full.
        self.reach(network, lambda: driver._bootstrapped and driver._route_exchange() is not None)
        network.pump()
        self.fill_ring(network)
        driver.advance()
        self.assertEqual((driver.phase, driver.tx.txn_seq, driver._inflight), ('open', 1, {}))
        self.assertIsNone(driver._retained)
        driver.stop('abort')
        self.finish(network)
        self.assert_closed(network)
        self.assertNotIn('flash_s5_open', [row[0] for row in self.commands(network)])
        self.assertEqual(network.controller.commits, 0)

    def test_a_sent_open_then_a_stop_continues_the_row_and_cancels_normally(self):
        network = self.between_chunks()
        driver = network.driver
        network.frozen = {0, 1, 2, 3}
        driver.advance()
        self.assertEqual(driver.phase, 'open')
        self.assertEqual(list(driver._inflight.values()), ['open'])
        self.assertIsNotNone(driver._retained)
        driver.stop('abort')
        self.assertEqual(list(driver._inflight.values()), ['open'])
        self.finish(network)
        self.assert_closed(network)
        kinds = [entry['kind'] for entry in driver._journal]
        self.assertIn('request_row_continued', kinds)
        self.assertNotIn('request_row_closed', kinds)
        self.assertEqual(self.after_row_one(self.commands(network))[0], ('flash_s5_open', 2))
        self.assertNotIn(('flash_s5_cancel', 1), self.commands(network))

    def test_a_refused_open_with_no_stop_retries_and_ends_like_a_clean_run(self):
        clean = self.network()
        clean.run_request()
        network = self.refused_open()
        self.assertEqual(network.driver._retained['transaction'].txn_seq, 1)
        network.frozen = set()
        network.pump()
        network.run_request()
        self.assert_closed(network)
        self.assertEqual(network.controller.commits, network.expected_commits)
        self.assertEqual(network.persistent_state(), clean.persistent_state())


if __name__ == '__main__':
    unittest.main()
