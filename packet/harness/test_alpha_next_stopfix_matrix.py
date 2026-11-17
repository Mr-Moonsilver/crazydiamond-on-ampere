#!/usr/bin/env python3
"""Stop-fix matrix on the composed alpha-next drivers (assessment 4C.4): each repair, a witness per driver.

Repairs (both from claude/alpha-trunk-stopfix):
  fix 1  a stop after a refused continuation open closes the retained row first (engine_driver._drive_stop);
  fix 2  a stop with one TP peer accepted replays the retained command, then cancels
         (engine_driver._drive_partial_stop_submission).

Drivers on this tree:
  stepped  NativeSegmentDriver, serial (harness/test_e3_stopfix_partial_segment.py and
           test_e3_stopfix_refused_open.py run it; this file adds the receipts read-back);
  fused    Step A, FLASH_S5_QSA_FUSED, on test_e3_native_qsa_fused.FusedNetwork;
  grouped  Step B, layer groups of 2 and 4, on test_e3_native_layer_groups.GroupNetwork;
  C0       StagedSegmentDriver (FLASH_S5_STAGE_LOCAL), on harness/stage_net.RingStack(staged=True).
           C0 overrides the stop path, so fix 2 was missing there (the partial-PREPARE fault, assessment 1);
           alpha-next adds the replay to StagedSegmentDriver._stop_obligations, per stage, in the frozen order
           (tails, PP, receipts, partial replay, then retained rows and cancel).
  C1       not in this build (claude/c1 is composed later).

Each witness: the repaired driver closes; the same scenario with the repair removed (a mutant, patched in memory)
ends in finding 2's refusal 'accepted operations exist without a bound producer' with ownership kept.  Fix 1 on
fused/grouped: fix 1 acts at a chunk OPEN before any segment exists, on the shared NativeSegmentDriver._drive_stop;
the only fused-dependent code on that path is _stop_round's bound, so the stepped refused-open scenario is run with
the chunk's fused latch set (declared leaf 'fused-latch-at-refused-open').  Fix 1 on C0 is C0's own
_stop_retained_rows (test_e3_stage_local.PartialOpen); it is re-run here as the C0 cell.

Receipts (PLAN section 1, item 1): every replay writes a journal row, bumps the status counter
stop_receipts.partial_replays and, where progress receipts are installed, one 'flash-s5 stop ' line; an exhausted
stop bound writes stop_bound_exhausted the same three ways, before the refusal.

Leaves: harness/alpha_next_stopfix_fixture_leaves.json (inherits the stop-fix, C0 stage and held-g leaves).  No GPU,
CUDA, NCCL, MQ or model forward: whether an accepted TP peer waits in a collective is the box arm's question.
"""
import json
import logging
from unittest import mock
import unittest

import alpha_stopfix_net as asn
import stage_net
from e3_types import E3Pending
import test_e3_native_layer_groups as g
import test_e3_native_qsa_fused as f
import test_e3_stopfix_refused_open as ro

ed = asn.ed
w = asn.w
MUTANT = 'accepted operations exist without a bound producer'
STOP_PREFIX = ed.STOP_RECEIPT_LINE


def no_replay():
    """Mutant: fix 2 removed on every driver (the staged one inherits the method)."""
    return mock.patch.object(ed.NativeSegmentDriver, '_drive_partial_stop_submission', lambda self: False)


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def stop_lines(capture):
    return [json.loads(line[len(STOP_PREFIX):]) for line in capture.lines if line.startswith(STOP_PREFIX)]


def capture_receipts(test):
    """The engine's stop receipt lines, read where the driver writes them (its progress logger)."""
    capture = Capture()
    logger = logging.getLogger('alpha_next_stop_receipts')
    logger.addHandler(capture)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    test.addCleanup(logger.removeHandler, capture)
    patcher = mock.patch.object(ed, '_progress_logger', lambda: logger)
    patcher.start()
    test.addCleanup(patcher.stop)
    return capture


# ---------------------------------------------------------------- fused and grouped
class FusedGrouped(g.GroupCase):
    SIZES = (None, 2, 4)        # None: Step A fused without groups

    def network(self, size):
        return g.GroupNetwork(self, size=size) if size else f.FusedNetwork(self)

    def partial(self, size, kind=w.SegmentKind.PREPARE, rank=1):
        network = self.network(size)
        endpoint = g.PendingPeer(network.worker(rank), lambda command: command.kind is kind)
        network.transport.network.workers[rank] = endpoint
        driver = network.driver
        network.run_until(lambda: driver._partial_stop_state() is not None, limit=80)
        command, accepted, pending = driver._partial_stop_state()
        self.assertIs(command.kind, kind)
        self.assertEqual(pending, [rank])
        return network, endpoint, command, accepted

    def test_partial_prepare_replays_then_cancels_on_fused_and_each_group_size(self):
        for size in self.SIZES:
            with self.subTest(size=size):
                network, endpoint, command, accepted = self.partial(size)
                driver = network.driver
                self.assertTrue(driver._tx_fused)
                self.assertEqual(driver._tx_group, size or 1)
                driver.install_progress('trace')
                capture = capture_receipts(self)
                endpoint.defer = False
                driver.stop('abort')
                network.pump_stop()
                self.assert_closed(network)
                rows = [e for e in driver._journal if e['kind'] == 'stop_partial_replay']
                self.assertTrue(rows)
                self.assertEqual({(r['segment_kind'], tuple(r['accepted_ranks']), tuple(r['pending_ranks']),
                                   r['op_seq'], r['reason']) for r in rows},
                                 {('prepare', tuple(accepted), (1,), command.op_seq, 'abort')})
                cancels = [e for e in driver._journal if e['kind'] == 'cancel']
                self.assertEqual([c['never_submitted'] for c in cancels], [False])
                # Every replay is the retained command itself, sent before the cancel.
                sent = [args[0] for method, args, _, _ in network.sender.envelopes
                        if method == 'flash_s5_submit' and args[0].op_seq == command.op_seq
                        and args[0].kind is w.SegmentKind.PREPARE and args[0].stage == command.stage]
                self.assertGreaterEqual(len(sent), 2)
                self.assertTrue(all(item == command for item in sent))
                # Receipts: journal, status counter and one line per replay, the same rows.
                self.assertEqual(driver.stop_receipts(), dict(partial_replays=len(rows), bound_exhausted=0))
                lines = stop_lines(capture)
                self.assertEqual([l['kind'] for l in lines], ['stop_partial_replay'] * len(rows))
                for line, row in zip(lines, rows):
                    self.assertEqual(line['schema'], 's5-stop-receipt/1')
                    for key in ('transaction', 'op_seq', 'segment_kind', 'accepted_ranks', 'pending_ranks', 'reason'):
                        self.assertEqual(line[key], row[key], key)

    def test_witness_without_the_replay_the_partial_prepare_fault_returns(self):
        for size in self.SIZES:
            with self.subTest(size=size):
                network, endpoint, _, _ = self.partial(size)
                driver = network.driver
                endpoint.defer = False
                with no_replay(), self.assertRaisesRegex(ed.DriverError, MUTANT):
                    driver.stop('abort')
                    network.pump_stop()
                self.assertEqual(driver.error, 'native driver failed; ownership retained')
                self.assertTrue(driver.active)

    def test_a_peer_that_never_accepts_stops_at_the_fused_bound_with_its_receipt(self):
        network, endpoint, command, accepted = self.partial(2)
        driver = network.driver
        driver.install_progress('trace')
        capture = capture_receipts(self)
        driver.qsa_fused.stall_s = 0.0     # the fused stop bound is time; zero makes the first late round exhaust it
        clock = iter(range(10 ** 6))
        with mock.patch.object(driver, 'clock', lambda: float(next(clock))):
            with self.assertRaisesRegex(ed.DriverError, 'cancellation did not close worker ownership within the bound'):
                driver.stop('abort')
                network.pump_stop()
        self.assertTrue(driver.active)
        rows = [e for e in driver._journal if e['kind'] == 'stop_bound_exhausted']
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]['op_seq'], rows[0]['segment_kind'], rows[0]['pending_ranks'], rows[0]['fused']),
                         (command.op_seq, 'prepare', [1], True))
        self.assertEqual(driver.stop_receipts()['bound_exhausted'], 1)
        self.assertEqual([l['kind'] for l in stop_lines(capture)][-1], 'stop_bound_exhausted')

    def test_a_pending_peer_with_no_stop_completes_without_a_replay(self):
        for size in self.SIZES:
            with self.subTest(size=size):
                network, endpoint, _, _ = self.partial(size)
                endpoint.defer = False
                network.finish()
                self.assertEqual([e for e in network.driver._journal if e['kind'] == 'stop_partial_replay'], [])
                self.assertIsNone(network.driver.error)


class FusedRefusedOpen(ro.Base):
    """Fix 1 with the chunk's fused latch set at the refused open (leaf fused-latch-at-refused-open)."""

    def test_refused_open_with_the_fused_latch_closes_the_row_then_retires(self):
        for kind in ro.STOPS:
            with self.subTest(stop=kind):
                network = self.refused_open()
                driver = network.driver
                driver.qsa_fused = ed.FusedQsa()
                driver._tx_fused = True
                driver.stop(kind)
                self.assertIsNone(driver.error)
                self.finish(network)
                self.assert_closed(network)
                rows = self.after_row_one(self.commands(network))
                self.assertEqual(rows[0], ('flash_s5_cancel', 1))
                self.assertNotIn(('flash_s5_open', 2), rows)
                self.assertGreaterEqual(driver.qsa_fused.counters['stops'], 1)  # the fused bound was the one counted

    def test_reach_fix_1_is_the_branch_acting_with_the_fused_latch(self):
        """Reach, not a witness: the fix-1 witness proper is test_e3_stopfix_refused_open on c3f1f64 (no fused code on the path)."""
        network = self.refused_open()
        driver = network.driver
        driver.qsa_fused = ed.FusedQsa()
        driver._tx_fused = True
        with mock.patch.object(ed.NativeSegmentDriver, '_drive_retained_cancel',
                               lambda self: (_ for _ in ()).throw(AssertionError('fix 1 reached'))):
            # The branch is the fix; removing the branch's call shows it is the one acting at this state.
            with self.assertRaisesRegex(AssertionError, 'fix 1 reached'):
                driver.stop('abort')


class GroupedRefusedOpen(ro.Base):
    """Fix 1 with the chunk's fused latch and a group size latched at the refused open (sizes 2 and 4; leaf fused-latch-at-refused-open).

    The grouped column of the assessment's matrix. The C0 column's fix-1 equivalent is C0's own _stop_retained_rows
    (test_e3_stage_local.PartialOpen), not the stepped branch.
    """

    def test_refused_open_with_a_group_latched_closes_the_row_then_retires(self):
        for size in (2, 4):
            for kind in ro.STOPS:
                with self.subTest(size=size, stop=kind):
                    network = self.refused_open()
                    driver = network.driver
                    driver.qsa_fused = ed.FusedQsa()
                    driver.install_layer_groups().size = size
                    driver._tx_fused = True
                    driver._tx_group = size
                    driver.stop(kind)
                    self.assertIsNone(driver.error)
                    self.finish(network)
                    self.assert_closed(network)
                    rows = self.after_row_one(self.commands(network))
                    self.assertEqual(rows[0], ('flash_s5_cancel', 1))
                    self.assertNotIn(('flash_s5_open', 2), rows)

    def test_reach_fix_1_acts_with_a_group_latched(self):
        network = self.refused_open()
        driver = network.driver
        driver.qsa_fused = ed.FusedQsa()
        driver.install_layer_groups().size = 2
        driver._tx_fused, driver._tx_group = True, 2
        with mock.patch.object(ed.NativeSegmentDriver, '_drive_retained_cancel',
                               lambda self: (_ for _ in ()).throw(AssertionError('fix 1 reached'))):
            with self.assertRaisesRegex(AssertionError, 'fix 1 reached'):
                driver.stop('abort')


# ---------------------------------------------------------------- C0 staged
REPLAYED = (('prepare', 1), ('prefix', 1), ('layer', 1), ('suffix', 3), ('state_finalize', 1), ('prepare', 3))


class C0Base(unittest.TestCase):
    def network(self, **kwargs):
        network = stage_net.RingStack(staged=True, tokens=33, chunk=4, **kwargs)
        self.addCleanup(network.close_network)
        return network

    def busy(self, network, kind, rank, limit):
        dispatcher = network.workers[rank].dispatcher
        real = dispatcher.submit
        state = dict(tx=None, pending=0, took=[])

        def submit(command):
            if state['tx'] is None:
                state['tx'] = command.transaction
            if command.transaction == state['tx'] and command.kind.value == kind:
                if state['pending'] < limit:
                    state['pending'] += 1
                    raise E3Pending('test: peer busy')
                state['took'].append((command.op_seq, command.kind, command.descriptor))
            return real(command)
        dispatcher.submit = submit
        return state

    @staticmethod
    def staged_partial(driver):
        for frame in driver._frames:
            for stage in (0, 1):
                with driver._scope(frame, stage):
                    state = driver._partial_stop_state()
                    if state is not None:
                        return stage, state
        return None

    def partial(self, kind, rank, limit=3, **kwargs):
        network = self.network(**kwargs)
        state = self.busy(network, kind, rank, limit)
        driver = network.driver
        found = None
        for _ in range(4000):
            found = self.staged_partial(driver) if driver._frames else None
            if state['pending'] and found:
                break
            network.admit_next()
            network.tick()
        else:
            self.fail('never reached a partial acceptance')
        stage, (command, accepted, pending) = found
        self.assertEqual(command.kind.value, kind)
        self.assertEqual(pending, [rank])
        return network, stage, command, accepted, state

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
        self.assertEqual(driver._frames, [])
        self.assertEqual(driver._inflight, {})
        self.assertIsNone(network.controller.parked)
        self.assertTrue(network.controller.leases.shutdown()['clean'])
        for rank, worker in network.workers.items():
            self.assertIsNone(worker._flash_s5_bridge.transaction, rank)
            self.assertIsNone(worker.dispatcher.transaction, rank)
            self.assertIsNone(worker.dispatcher.quarantined, rank)

    def journal(self, network, kind):
        return [entry for entry in network.driver._journal if entry['kind'] == kind]


class C0Partial(C0Base):
    def test_each_partially_accepted_stage_segment_replays_before_the_cancel(self):
        for kind, rank in REPLAYED:
            for stop in ('abort', 'cancel', 'pause'):
                with self.subTest(kind=kind, rank=rank, stop=stop):
                    network, stage, command, accepted, state = self.partial(kind, rank)
                    driver = network.driver
                    driver.stop(stop)
                    self.finish(network)
                    self.assert_closed(network)
                    rows = self.journal(network, 'stop_partial_replay')
                    self.assertTrue(rows)
                    self.assertEqual({(r['segment_kind'], tuple(r['accepted_ranks']), tuple(r['pending_ranks']),
                                       r['op_seq'], r['reason']) for r in rows},
                                     {(kind, tuple(accepted), (rank,), command.op_seq, stop)})
                    kinds = [e['kind'] for e in driver._journal]
                    self.assertLess(max(i for i, k in enumerate(kinds) if k == 'stop_partial_replay'),
                                    kinds.index('cancel'))
                    self.assertEqual([c['never_submitted'] for c in self.journal(network, 'cancel')], [False])
                    self.assertEqual((network.controller.cancels, network.controller.commits), (1, 0))
                    # The pending peer took the retained command itself.
                    self.assertIn((command.op_seq, command.kind, command.descriptor), state['took'])
                    self.assertEqual(driver.stop_receipts()['partial_replays'], len(rows))

    def test_witness_c0_partial_prepare_fault_without_the_replay(self):
        """Assessment 1: C0's overridden stop path kept the partial-PREPARE fault; the stage 0 PREPARE shows it."""
        network, stage, _, _, _ = self.partial('prepare', 1)
        self.assertEqual(stage, 0)
        driver = network.driver
        with no_replay(), self.assertRaisesRegex(ed.DriverError, MUTANT):
            driver.stop('abort')
            for _ in range(500):
                network.tick()
        self.assertEqual(driver.error, 'native driver failed; ownership retained')
        self.assertTrue(driver.active)

    def test_c0_peer_that_never_accepts_stops_at_the_bound_with_its_receipt(self):
        network, _, command, _, _ = self.partial('prepare', 1, limit=10 ** 6, stop_budget=12)
        driver = network.driver
        driver.stop('abort')
        with self.assertRaisesRegex(ed.DriverError, 'cancellation did not close worker ownership within the bound'):
            for _ in range(500):
                network.tick()
        self.assertTrue(driver.active)
        exhausted = self.journal(network, 'stop_bound_exhausted')
        self.assertEqual(len(exhausted), 1)
        self.assertEqual(driver.stop_receipts(),
                         dict(partial_replays=len(self.journal(network, 'stop_partial_replay')), bound_exhausted=1))

    def test_c0_pending_peer_with_no_stop_ends_like_a_clean_run(self):
        clean = self.network()
        clean.run_request()
        for kind, rank in REPLAYED:
            with self.subTest(kind=kind, rank=rank):
                network, _, _, _, _ = self.partial(kind, rank)
                network.run_request()
                self.assertEqual(self.journal(network, 'stop_partial_replay'), [])
                self.assertEqual(network.controller.commits, network.expected_commits)
                self.assertEqual(network.persistent_state(), clean.persistent_state())


# ---------------------------------------------------------------- stepped receipts
class SteppedReceipts(unittest.TestCase):
    """The stepped driver's replay rows reach the status counter (the line needs progress installed: fused cell)."""

    def test_stepped_replay_counted_in_status(self):
        network = asn.RingStack(tokens=33, chunk=4)
        self.addCleanup(network.close_network)
        dispatcher = network.workers[1].dispatcher
        real = dispatcher.submit
        state = dict(tx=None, pending=0)

        def submit(command):
            if state['tx'] is None:
                state['tx'] = command.transaction
            if command.transaction == state['tx'] and command.kind.value == 'prepare' and state['pending'] < 3:
                state['pending'] += 1
                raise E3Pending('test: peer busy')
            return real(command)
        dispatcher.submit = submit
        driver = network.driver
        for _ in range(4000):
            if state['pending'] and driver._partial_stop_state() is not None:
                break
            network.admit_next()
            network.tick()
        driver.stop('abort')
        for _ in range(3000):
            if not driver.active:
                break
            network.tick()
        self.assertFalse(driver.active)
        rows = [e for e in driver._journal if e['kind'] == 'stop_partial_replay']
        self.assertTrue(rows)
        self.assertEqual(driver.stop_receipts(), dict(partial_replays=len(rows), bound_exhausted=0))
        self.assertIsNone(driver.progress)   # no line without progress receipts installed (default off)


if __name__ == '__main__':
    unittest.main()
