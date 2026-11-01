#!/usr/bin/env python3
"""Stop while one TP peer accepted a segment and the other answered pending (ENGINE finding 2).

C0/C1 builder, 1 Nov 2026 (tmp/c0c1/ENGINE-FINDING.md section 2): on c3f1f64 a stop after a partial PREPARE cancelled every
rank and then refused retirement ('accepted operations exist without a bound producer'), keeping ownership.  The fix
replays the exact retained command to its targets under the stop-round bound, then cancels, as C1's
OverlapSegmentDriver._stop_partial_segment does.  Real driver, controller, bridges, dispatchers and transports; the one
injection is a target rank's dispatcher answering typed pending.  Leaves: harness/alpha_stopfix_fixture_leaves.json.
No GPU, CUDA, NCCL, MQ or model forward: whether an accepted TP peer would wait in a collective is not shown here.
"""
from unittest import mock
import unittest

from alpha_stopfix_net import RingStack, ed, stage_of_command
from e3_types import E3Pending

STOPS = ('abort', 'cancel', 'pause')
# Non-PP kinds and the rank that answers pending (stage 0 ranks 0/1, stage 1 ranks 2/3).
REPLAYED = (('prepare', 1), ('prefix', 1), ('layer', 1), ('suffix', 3), ('state_finalize', 1))


class Base(unittest.TestCase):
    def network(self, **kwargs):
        network = RingStack(tokens=33, chunk=4, **kwargs)
        self.addCleanup(network.close_network)
        return network

    def busy(self, network, kind, rank, limit):
        """The target rank answers typed pending to the first ``limit`` submits of ``kind`` in the first chunk."""
        dispatcher = network.workers[rank].dispatcher
        real = dispatcher.submit
        state = dict(tx=None, pending=0)

        def submit(command):
            if state['tx'] is None:
                state['tx'] = command.transaction
            if command.transaction == state['tx'] and command.kind.value == kind and state['pending'] < limit:
                state['pending'] += 1
                raise E3Pending('test: peer busy')
            return real(command)
        dispatcher.submit = submit
        return state

    def partial(self, kind='prepare', rank=1, limit=3, **kwargs):
        """Tick until a target peer accepted the command and another answered pending; return (network, retained)."""
        network = self.network(**kwargs)
        state = self.busy(network, kind, rank, limit)
        driver = network.driver
        for _ in range(4000):
            if state['pending'] and driver._retry is not None:
                break
            network.admit_next()
            network.tick()
        else:
            self.fail('never reached a partial acceptance')
        command = driver._retry
        accepted = {r for r, op in driver.accepted_tickets if op == command.op_seq and r in driver._targets}
        self.assertEqual(command.kind.value, kind)
        self.assertTrue(accepted and not set(driver._targets) <= accepted, (accepted, driver._targets))
        return network, command, state

    def finish(self, network, limit=3000):
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
        self.assertIsNone(network.controller.parked)
        self.assertTrue(network.controller.leases.shutdown()['clean'])
        self.assertEqual(network.transport._calls, {})
        for rank, worker in network.workers.items():
            self.assertIsNone(worker._flash_s5_bridge.transaction, rank)
            self.assertIsNone(worker.dispatcher.transaction, rank)
            self.assertIsNone(worker.dispatcher.quarantined, rank)

    def journal(self, network, kind):
        return [entry for entry in network.driver._journal if entry['kind'] == kind]

    def submits_after(self, network, mark, op_seq):
        rows = []
        for envelope in network.ring.history[mark:]:
            method, message, _ = stage_of_command(envelope)
            if method == 'flash_s5_submit' and message.op_seq == op_seq:
                rows.append(message)
            if method == 'flash_s5_cancel':
                rows.append('cancel')
        return rows


class Fix(Base):
    def test_partial_prepare_then_each_stop_replays_the_same_command_then_cancels(self):
        for kind in STOPS:
            with self.subTest(stop=kind):
                network, command, state = self.partial('prepare')
                driver = network.driver
                mark = len(network.ring.history)
                driver.stop(kind)
                self.finish(network)
                self.assert_closed(network)
                rows = self.submits_after(network, mark, command.op_seq)
                replays = [row for row in rows if row != 'cancel']
                self.assertTrue(replays)
                self.assertTrue(all(row == command for row in replays))  # the same command, never a new one
                self.assertLess(rows.index(replays[-1]), rows.index('cancel'))  # every replay before the cancel
                records = self.journal(network, 'stop_partial_replay')
                self.assertTrue(records)
                self.assertEqual({(r['segment_kind'], tuple(r['accepted_ranks']), tuple(r['pending_ranks']), r['op_seq'])
                                  for r in records}, {('prepare', (0,), (1,), command.op_seq)})
                # The producer bound through the ordinary acceptance; the cancel is the bound path, not never-submitted.
                self.assertEqual((network.controller.cancels, network.controller.commits), (1, 0))
                cancels = self.journal(network, 'cancel')
                self.assertEqual([c['never_submitted'] for c in cancels], [False])
                self.assertEqual(state['pending'], 3)

    def test_each_partially_accepted_kind_replays_before_the_cancel(self):
        for kind, rank in REPLAYED:
            with self.subTest(kind=kind):
                network, command, _ = self.partial(kind, rank)
                network.driver.stop('abort')
                self.finish(network)
                self.assert_closed(network)
                records = self.journal(network, 'stop_partial_replay')
                self.assertTrue(records)
                self.assertEqual({r['segment_kind'] for r in records}, {kind})
                self.assertEqual({tuple(r['pending_ranks']) for r in records}, {(rank,)})

    def test_pp_partials_stay_with_their_own_obligation(self):
        for kind, rank in (('pp_send', 1), ('pp_receive', 3)):
            with self.subTest(kind=kind):
                network, _, _ = self.partial(kind, rank)
                network.driver.stop('abort')
                self.finish(network)
                self.assert_closed(network)
                self.assertEqual(self.journal(network, 'stop_partial_replay'), [])

    def test_a_peer_that_never_accepts_stops_at_the_bound_with_ownership_kept(self):
        network, _, _ = self.partial('prepare', limit=10 ** 6, stop_budget=12)
        driver = network.driver
        driver.stop('abort')
        with self.assertRaisesRegex(ed.DriverError, 'cancellation did not close worker ownership within the bound'):
            for _ in range(500):
                network.tick()
        self.assertEqual(driver.error, 'native driver failed; ownership retained')
        self.assertTrue(driver.active)
        self.assertEqual(len(self.journal(network, 'stop_partial_replay')), 12)

    def test_without_the_replay_the_fault_returns(self):
        """Mutant: the replay removed, the same stop ends in finding 2's refusal with ownership kept."""
        network, _, _ = self.partial('prepare')
        driver = network.driver
        with mock.patch.object(ed.NativeSegmentDriver, '_drive_partial_stop_submission', lambda self: False):
            driver.stop('abort')
            with self.assertRaisesRegex(ed.DriverError, 'accepted operations exist without a bound producer'):
                for _ in range(500):
                    network.tick()
        self.assertEqual(driver.error, 'native driver failed; ownership retained')
        self.assertTrue(driver.active)


class Controls(Base):
    """Paths the new branch must not reach."""

    def test_a_pending_peer_with_no_stop_ends_like_a_clean_run(self):
        clean = self.network()
        clean.run_request()
        for kind, rank in REPLAYED:
            with self.subTest(kind=kind):
                network, _, _ = self.partial(kind, rank)
                network.run_request()
                self.assertEqual(self.journal(network, 'stop_partial_replay'), [])
                self.assertEqual(network.controller.commits, network.expected_commits)
                self.assertEqual(network.persistent_state(), clean.persistent_state())

    def test_both_targets_pending_is_not_a_partial(self):
        network = self.network()
        state0 = self.busy(network, 'prepare', 0, 10 ** 6)
        state1 = self.busy(network, 'prepare', 1, 10 ** 6)
        driver = network.driver
        for _ in range(4000):
            if state0['pending'] and state1['pending'] and driver._retry is not None:
                break
            network.admit_next()
            network.tick()
        self.assertEqual({r for r, _ in driver.accepted_tickets}, set())
        driver.stop('abort')
        self.finish(network)
        self.assert_closed(network)
        self.assertEqual(self.journal(network, 'stop_partial_replay'), [])
        # The never-submitted retirement: no producer bound, nothing accepted, no 'cancel' journal row on that path.
        self.assertEqual((network.controller.cancels, network.controller.commits), (1, 0))
        self.assertEqual(self.journal(network, 'cancel'), [])

    def test_a_stop_with_no_partial_command_is_unchanged(self):
        network = self.network()
        driver = network.driver
        for _ in range(4000):
            if driver.phase == 'segments' and driver.index >= 3 and driver._retry is None:
                break
            network.admit_next()
            network.tick()
        driver.stop('abort')
        self.finish(network)
        self.assert_closed(network)
        self.assertEqual(self.journal(network, 'stop_partial_replay'), [])


if __name__ == '__main__':
    unittest.main()
