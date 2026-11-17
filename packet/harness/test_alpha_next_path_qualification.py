#!/usr/bin/env python3
"""alpha-next 4C.5: fused and grouped path qualification on the composed tree (CPU; Step A/B's declared leaves).

Assessment 4C item 5 asks, for the fused path (Step A) and the grouped path (Step B, sizes 2 and 4):
  (a) an automatically admitted fused or grouped chunk;   (b) proof that it engaged;   (c) the numerical contract;
  (d) a cancel after partial TP acceptance with bucket consumers in flight;            (e) reuse after drain.

(a)-(c) and the halves of (d) already have tests in the inherited suites; QUALIFICATION names them by id and the first test here
loads every one (a renamed or dropped test refuses). This module adds the two that did not exist:
  * (d) whole, fused: a QSA LAYER command accepted by one TP peer and pending on the other while the accepted peer's earlier
    buckets are unresolved (consumers in flight). The stop replays the same command to the pending peer, nothing is released
    while device work is outstanding, and the chunk closes once it drains.
    Grouped (sizes 2 and 4): Step B's run-ahead bound never sends the QSA group while an earlier group's buckets are in flight
    on a peer, so that state cannot arise; the test asserts the bound holds for 120 ticks with rank 0 held, then the stop meets
    the bounded state, releases nothing early and closes. In this fixture rank 0's held work before the QSA group carries no
    sparse bucket events, so grouped bucket consumers in flight at a stop are not reached on CPU: owed to the VM run.
  * (e) at the worker boundary: after that drained cancel, every rank's same bridge and dispatcher accept a fresh fused or
    grouped open with no stale op_seq, group size or selection, and a stepped open after a drain carries size 1.
    Mechanism: a partially accepted LAYER closes through Step A's TP tail obligation (the replay), not stop fix 2 (PREPARE retry;
    its witnesses are test_alpha_next_stopfix_matrix's), so disabling fix 2 leaves this test green (checked, by design).
    A whole second admitted chunk on the same connected network is not reachable in this fixture (the leaf scheduler drops the
    cancelled request); declared, owed to the VM run.
"""
import dataclasses
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_e3_native_layer_groups as g  # noqa: E402
import test_e3_native_qsa_fused as f  # noqa: E402

w = f.w

QUALIFICATION = {
    'a-auto-admitted': (
        'test_e3_native_qsa_fused.OneBoundedCommand.test_multi_bucket_layer_is_one_command_with_every_device_event_unresolved',
        'test_e3_native_qsa_fused.OneBoundedCommand.test_manual_cycles_never_run_fused',
        'test_e3_native_layer_groups.OneCommand.test_one_command_per_group_with_every_device_event_unresolved',
        'test_e3_native_layer_groups.DefaultOff.test_manual_cycles_never_group',
    ),
    'b-engaged': (
        'test_e3_native_qsa_fused.OneBoundedCommand.test_same_device_work_as_the_stepped_path_with_far_fewer_commands',
        'test_e3_native_layer_groups.OneCommand.test_commands_per_chunk_shrink_by_the_plan',
        'test_e3_native_layer_groups.OneCommand.test_a_qsa_stage_one_grants_and_runs_fused',
    ),
    'c-numerical-contract': (
        'test_e3_native_qsa_fused.Commitment.test_real_ledger_agrees_on_order_communicator_and_shape',
        'test_e3_native_qsa_fused.Commitment.test_equal_counts_with_a_different_shape_fail_closed_without_release',
        'test_e3_native_qsa_fused.ConsecutiveLayers.test_second_layer_first_copy_waits_on_the_first_layers_last_consumer',
        'test_e3_native_layer_groups.OneCommand.test_same_device_work_as_size_one',
        'test_e3_native_layer_groups.StepAIdentity.test_size_one_is_exactly_step_as_command_stream',
        'test_e3_native_layer_groups.Ledger.test_the_ledger_is_per_group_and_is_compared_on_every_group',
    ),
    'd-partial-acceptance-halves': (
        'test_e3_native_qsa_fused.Cancel.test_cancel_at_every_internal_boundary_with_one_peer_unresolved',
        'test_e3_native_qsa_fused.Cancel.test_cancel_between_the_peers_acceptances_replays_the_exact_command',
        'test_e3_native_layer_groups.Cancel.test_cancel_at_every_command_boundary_with_one_peer_unresolved',
        'test_e3_native_layer_groups.Cancel.test_cancel_between_acceptances_in_a_group_with_qsa',
        'test_alpha_next_stopfix_matrix.FusedGrouped.test_partial_prepare_replays_then_cancels_on_fused_and_each_group_size',
    ),
    'd-whole-and-e-reuse': (
        'test_alpha_next_path_qualification.PartialWithBucketsInFlight.test_layer_partial_acceptance_with_device_work_outstanding',
        'test_alpha_next_path_qualification.PartialWithBucketsInFlight.test_reuse_after_the_drained_cancel',
    ),
}


class Inventory(unittest.TestCase):
    def test_every_declared_test_loads(self):
        loader = unittest.defaultTestLoader
        for part, names in QUALIFICATION.items():
            for name in names:
                with self.subTest(part=part, test=name):
                    loaded = list(loader.loadTestsFromName(name))
                    self.assertEqual([t.id() for t in loaded], [name])


class PartialWithBucketsInFlight(g.GroupCase):
    SIZES = (None, 2, 4)        # None: Step A fused without groups
    # The first LAYER command that holds a QSA layer (bucket consumers): Step B's Cancel.test_cancel_between_acceptances_in_a_group_with_qsa.
    QSA_GROUP_START = {None: 0, 2: 2, 4: 0}

    def network(self, size):
        return g.GroupNetwork(self, size=size) if size else f.FusedNetwork(self)

    def drained_cancel(self, size):
        network = self.network(size)
        network.install_tp()
        start = self.QSA_GROUP_START[size]
        target = lambda command: command.kind is w.SegmentKind.LAYER and command.descriptor.layer_start == start
        endpoint = g.PendingPeer(network.worker(1), target)
        network.transport.network.workers[1] = endpoint
        driver = network.driver
        # Rank 0 never completes device work (the inherited Cancel tests' ready=(other,)): its buckets stay in flight.
        held = network.qsa[0]
        outstanding = lambda: sum(1 for event in held.device.events if not event.ready)
        partial = lambda: (driver._tail_submission is not None and driver._tail_submission.accepted == {0}
                           and driver._tail_submission.command.descriptor.layer_start == start)
        if size is None:
            network.run_until(partial, ready=(1,), limit=120)
            obligation = driver._tail_submission
            self.assertNotIn(obligation.command.op_seq, network.qsa[1].dispatcher.accepted)
        else:
            # Step B's run-ahead bound: with an earlier group's buckets in flight the QSA group is never sent, so a partially
            # accepted QSA group with consumers in flight cannot arise; the stop meets the bounded state instead.
            for _ in range(120):
                network.tick(ready=(1,))
                self.assertFalse(partial())
            self.assertEqual(network.sent(w.SegmentKind.LAYER, start), [])
            obligation = None
        self.assertTrue(driver._tx_fused)
        self.assertEqual(driver._tx_group, size or 1)
        if size is None:
            self.assertGreater(outstanding(), 0, 'the accepted peer has device work outstanding')
        before = [h.released for h in held.holds]
        endpoint.defer = False
        driver.stop('abort')
        for _ in range(5):
            driver.pump_stop(1)
        # While rank 0's work is in flight, the stop releases nothing it holds (no hold flips, no operation release).
        if size is None:
            self.assertGreater(outstanding(), 0)
        self.assertEqual([h.released for h in held.holds][:len(before)], before)
        self.assertFalse(any(h.released for h in held.holds[len(before):]))
        self.assertFalse(held.operation is not None and held.operation._released)
        self.assertEqual([a[0] for a in network.sender.envelopes].count('flash_s5_release'), 0)
        network.pump_stop()
        self.assert_closed(network)
        if obligation is not None:
            # The pending peer got the same command before any cancel; both entered the one collective.
            sent = network.sent(w.SegmentKind.LAYER, start)
            self.assertGreaterEqual(len(sent), 2)
            self.assertTrue(all(item == obligation.command for item in sent))
            self.assertEqual(obligation.accepted, {0, 1})
            self.assertEqual(network.tp[0].collectives, network.tp[1].collectives)
        cancels = [e for e in driver._journal if e['kind'] == 'cancel']
        self.assertEqual([c['never_submitted'] for c in cancels], [False])
        return network

    def test_layer_partial_acceptance_with_device_work_outstanding(self):
        for size in self.SIZES:
            with self.subTest(size=size):
                self.drained_cancel(size)

    def test_reuse_after_the_drained_cancel(self):
        for size in self.SIZES:
            with self.subTest(size=size):
                network = self.drained_cancel(size)
                record = network.record
                fresh = dataclasses.replace(record, transaction=dataclasses.replace(
                    record.transaction, txn_seq=record.transaction.txn_seq + 1))
                for rank, stack in sorted(network.qsa.items()):
                    dispatcher = stack.dispatcher
                    # Drained: nothing of the cancelled transaction is left on the worker.
                    self.assertEqual((dispatcher.accepted, dispatcher.transaction, stack.bridge.transaction), ({}, None, None))
                    self.assertEqual((dispatcher.layer_group, dispatcher.qsa_fused), (1, False))
                    ack = stack.bridge.open(w.OpenRequest(fresh, network.driver.ranks, fresh.state_epoch, 1,
                                                          qsa_fused=True, layer_group=size or 1))
                    self.assertTrue(ack.accepted, rank)
                    self.assertEqual((dispatcher.layer_group, dispatcher.qsa_fused), (size or 1, True))
                    self.assertEqual(dispatcher.accepted, {})
                    self.assertEqual(w.OpSeqTracker().propose(w.SegmentDescriptor(fresh.transaction, w.SegmentKind.PREPARE,
                                                                                  stack.stage if hasattr(stack, 'stage')
                                                                                  else rank // 2)), 0)


if __name__ == '__main__':
    unittest.main()
