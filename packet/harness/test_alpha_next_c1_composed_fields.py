#!/usr/bin/env python3
"""alpha-next-c1: C1's two chunks in flight on alpha-next's composed per-chunk state (fused QSA, layer groups, lean replies).

Two engine faults found while composing claude/c1 onto claude/alpha-next (ENGINE, fixed in stage_overlap.py before any box run):
  1. alpha-next's base driver keeps per-chunk fields on ``self`` (``_tx_group``, ``_tx_fused``, ``_replies``, ``_lean_streams``,
     ``_fused_mark``, ``_fused_mark_at``); C0's frame swap (stage_progress._CHUNK_FIELDS) does not carry them, so with two frames
     n+1's begin and n's reset overwrote the other chunk's values.  Fix: ``COMPOSED_CHUNK_FIELDS`` kept per frame.
  2. C1 began the successor with the progress recorder set aside, so ``_open_replies()`` gave ``full`` for n+1 on a lean arm.
     Fix: the successor's frozen ``_begin`` reads the real recorder through ``_ProgressNoBegin`` (only ``begin`` is a no-op).

Connected scenario: C1's ring fixture (test_e3_stage_overlap.OverlapStack: real StagedAdmissionController, OverlapSegmentDriver,
four NativeWorkerBridge/LaneDispatcher workers, BoundedSender ring, PP channel), stage 1 slow so stage 0 runs n+1 while stage 1
is on n, with the composed selections installed on the real driver and latched by the real dispatchers at OPEN (fused, a layer
group; the dispatcher refuses any LAYER command whose ``layer_group`` differs from its chunk's).  Per band a fresh fixture
incarnation at its own group size (g2, g4); the per-arm switch itself is alpha-next's typed control, tested there.

The ring model has two layers per stage, so g2 and g4 both send one LAYER command per stage; what differs per band is the size
latched at OPEN and carried on every command, which is the field the swap must keep per frame.

Mutants (``Mutants``): the swap reverted to C0's, and the recorder view reverted to ``None``, each fail here.  Dropping one field
from the swap is caught for ``_replies``, ``_lean_streams``, ``_tx_fused`` and ``_tx_group`` (the per-field sweep is in
tmp/alpha-next-c1/STATUS.md).  ``_fused_mark``/``_fused_mark_at`` (the fused stall watchdog's no-progress mark) are kept per frame
so each chunk's watchdog reads its own tickets and reports, as serially; with both shared the stall below is still caught at the
same tick, so no mutant here separates them (declared in the leaves file, ``cannot_prove``).

Declared leaves (harness/alpha_next_c1_engine_fixture_leaves.json, inheriting C1's, Step A's and Step B's): auto-arm ownership, the
no-QSA layer driver, the fake lane's stage end, the injected watchdog clock.  No GPU, CUDA, NCCL or model forward.
"""
import types
import unittest
from unittest import mock

import test_e3_stage_overlap as t  # first: its imports put the engine sources on the path
import stage_overlap as so
import stage_progress as sp
import test_e3_native_qsa_fused as fq
from stage_net import stage_of_command
from reply_transport import TransportError
from engine_driver import DriverError

w = t.w


class Fixture(t.OverlapBase):
    def composed(self, *, group=None, lean=False, fused=True, network=None):
        network = self.network() if network is None else network
        driver = network.driver
        # Leaf auto-arm-ownership: the auto-arm policy owns the chunk (real policy: test_native41_autoarm).
        driver.auto_arm = types.SimpleNamespace(manual_owned=False, phase='native', active=False)
        driver._admission_intent = {'origin': 'auto-arm', 'phase': 'native', 'seq': 1}
        for rank in (0, 1, 2, 3):
            dispatcher = network.worker(rank).dispatcher
            # Leaf no-qsa-layer-driver: the ring model's four layers are GDN only, so PREPARE's fused plan is empty.
            dispatcher.driver_factory = lambda operation, module, context: fq.NoQsaDriver()
            # Leaf fake-lane-stage-end: the real lane's end_layer is its stage's layer end.
            lane = dispatcher.native.lane_forward(None)
            lane.end_layer = max(lane.layers) + 1
        if fused:
            driver.install_qsa_fused().selected = True
        if group is not None:
            driver.install_layer_groups().size = group
        if lean:
            driver.install_progress('burst')
            driver.progress.replies = 'lean'
        return network

    def run_overlapped(self, network):
        driver = network.driver
        self.stage_one_held(network, lambda: self.finished(network), limit=4000)
        self.assertIsNone(driver.error)
        self.assertEqual(network.controller.commits, network.expected_commits)
        self.assertGreater(driver.overlap_counters['successor_frames'], 0)
        self.assertGreater(driver.overlap_counters['early_opens'], 0)
        self.assertLessEqual(max(network.inflight_peak.values()), 1)
        self.assert_closed(network)

    def finished(self, network):
        driver = network.driver
        return driver.error or (network.controller.commits >= network.expected_commits and not driver.live)

    def sent(self, network, method):
        return [message for name, message, _targets in map(stage_of_command, network.sent_envelopes) if name == method]

    def assert_groups(self, network, size):
        layers = [m for m in self.sent(network, 'flash_s5_submit') if m.descriptor.kind is w.SegmentKind.LAYER]
        self.assertTrue(layers)
        self.assertEqual({m.descriptor.layer_group for m in layers}, {size})
        opens = self.sent(network, 'flash_s5_open')
        self.assertEqual(len(opens), 2 * network.expected_commits)
        self.assertEqual({(o.qsa_fused, o.layer_group) for o in opens}, {(True, size)})
        counters = network.driver.layer_groups.counters
        self.assertEqual(counters['chunks'], network.expected_commits)
        self.assertEqual(counters['group_commands'], len({(m.descriptor.transaction, m.op_seq) for m in layers}))



class Composed(Fixture):
    def test_two_chunks_in_flight_per_group_size(self):
        for size in (2, 4):
            with self.subTest(size=size):
                network = self.composed(group=size)
                self.run_overlapped(network)
                self.assert_groups(network, size)

    def test_grouped_overlap_matches_grouped_serial(self):
        """Per group size: the same per-rank device ledgers, persistent state (KV allocation calls and sizes in order, cache
        calls, rows, frontier) as the grouped serial driver; one command in flight per stage; a successor that may be terminal is
        refused; every admitted successor became a frame and every promotion an overlapped commit."""
        for size in (2, 4):
            with self.subTest(size=size):
                off = self.composed(group=size, network=self.serial())
                off.run_request()
                on = self.composed(group=size)
                self.run_overlapped(on)
                self.assertEqual(off.worker_errors, [])
                self.assertEqual(on.worker_errors, [])
                for rank in (0, 1, 2, 3):
                    self.assertEqual(on.ledgers[rank], off.ledgers[rank], rank)
                self.assertEqual(on.persistent_state(), off.persistent_state())
                self.assertEqual(off.driver.layer_groups.counters['chunks'], on.driver.layer_groups.counters['chunks'])
                self.assertEqual(off.driver.layer_groups.counters['group_commands'],
                                 on.driver.layer_groups.counters['group_commands'])
                counters, ctl = on.driver.overlap_counters, on.controller.overlap_counters
                self.assertGreater(on.controller.overlap_refusals['successor-may-be-terminal'], 0)
                self.assertEqual(ctl['successors'], counters['successor_frames'])
                self.assertEqual(ctl['promoted'], counters['overlapped_commits'])
                self.assertLessEqual(max(on.inflight_peak.values()), 1, on.inflight_peak)

    def test_overlap_arms_emit_one_peak_receipt_per_chunk_per_rank(self):
        """alpha-next's per-chunk peak-allocated receipt under two chunks in flight: a bridge holds one transaction at a time, so
        each rank's open and release alternate and its one peak slot is the chunk's own; one line per chunk per rank, in order."""
        import native_measurement
        import test_alpha_next_peak_receipt as peak
        network = self.composed(group=2, lean=True)
        lines = []
        for rank in (0, 1, 2, 3):
            bridge = network.worker(rank)._flash_s5_bridge
            bridge._memory_receipts = True       # FLASH_S5_AUTO_ARM's receipts, as on the box's auto-armed arms
            bridge._memory_cuda = peak.PeakCuda()   # Leaf peak-allocator (the peak test's allocator counters)
        with mock.patch.object(native_measurement, 'emit_receipt_kv', lambda line, *rest, **kw: lines.append(line)):
            self.run_overlapped(network)
        rows = [peak.LINE.match(line).groups() for line in lines if line.startswith('flash-s5 memory-peak ')]
        ends = [min(4 * k, network.tokens - 1) for k in range(1, network.expected_commits + 1)]
        for rank in (0, 1, 2, 3):
            mine = [(int(txn), int(start), int(end)) for r, txn, start, end, *_ in rows if int(r) == rank]
            self.assertEqual([end for _txn, _start, end in mine], ends, rank)
            self.assertEqual([txn for txn, _start, _end in mine], sorted(txn for txn, _start, _end in mine), rank)
            self.assertEqual(getattr(network.worker(rank)._flash_s5_bridge, 'peak_receipt_errors', 0), 0)
        self.assertEqual(len(rows), 4 * network.expected_commits)

    def test_lean_arm_opens_every_chunk_lean_successors_included(self):
        network = self.composed(lean=True, fused=False)
        self.run_overlapped(network)
        opens = self.sent(network, 'flash_s5_open')
        self.assertEqual(len(opens), 2 * network.expected_commits)
        self.assertEqual({o.replies for o in opens}, {'lean'})
        self.assertEqual(network.driver.progress.counters['lean_chunks'], network.expected_commits)

    def test_lean_and_grouped_together(self):
        network = self.composed(group=2, lean=True)
        self.run_overlapped(network)
        self.assert_groups(network, 2)
        self.assertEqual({o.replies for o in self.sent(network, 'flash_s5_open')}, {'lean'})

    def test_each_frame_keeps_its_own_composed_fields(self):
        """Direct read of the swap: at every tick, each live frame's fields are its own chunk's (both frames grouped)."""
        network = self.composed(group=2, lean=True)
        driver = network.driver
        seen = []

        def check():
            for frame in driver._frames:
                seen.append(frame.composed['_tx_group'])
                self.assertEqual(frame.composed['_tx_group'], 2)
                self.assertIs(frame.composed['_tx_fused'], True)
                self.assertEqual(frame.composed['_replies'], 'lean')
            if driver._frames and driver._scoped_frame is None:
                self.assertEqual(driver._tx_group, driver._frames[0].composed['_tx_group'])
            return self.finished(network)
        self.stage_one_held(network, check, limit=4000)
        self.assertIsNone(driver.error)
        self.assertGreater(len(seen), network.expected_commits)

    def test_fused_stall_watchdog_with_two_frames_in_flight(self):
        """Stage 1 stops reading its ring with n and n+1 in flight: the chunk's watchdog fails closed just past the bound."""
        network = self.composed(group=2)
        driver = network.driver
        now = [1000.0]
        driver.clock = lambda: now[0]      # Leaf watchdog-clock: the injected monotonic clock the driver already takes.
        driver.qsa_fused.stall_s = 5.0
        self.stage_one_held(network, lambda: self.side_by_side(network), limit=4000)
        raised = None
        for _ in range(200):
            now[0] += 0.1
            network.admit_next()
            network.complete_everything()
            network.pump((0, 1))
            try:
                driver.advance()
            except Exception as exc:
                raised = exc
                break
        self.assertIsNotNone(raised)
        self.assertEqual(str(raised), 'tp-collective-stall')
        self.assertIsNotNone(driver.error)
        self.assertEqual(len(driver._frames), 2)
        self.assertTrue(5.0 < now[0] - 1000.0 < 5.5)

    def side_by_side(self, network):
        frames = network.driver._frames
        return (len(frames) == 2 and frames[1].stage[0]['phase'] == 'segments' and 1 not in frames[0].released)


class Mutants(Fixture):
    """Each fix reverted in memory: the connected scenario above fails."""

    def revert_swap(self):
        stack = mock.patch.multiple(so.OverlapSegmentDriver, _new_frame=sp.StagedSegmentDriver._new_frame,
                                    _scope=sp.StagedSegmentDriver._scope, _show_rest=sp.StagedSegmentDriver._show_rest)
        stack.start()
        self.addCleanup(stack.stop)

    def test_without_the_per_frame_fields_a_grouped_pair_fails(self):
        """C0's swap only: a chunk's begin or reset overwrites the other frame's group; stage 1's real dispatchers refuse."""
        self.revert_swap()
        network = self.composed(group=2)
        with self.assertRaises(TransportError):
            self.run_overlapped(network)
        refusals = {(rank, error) for rank, _method, _kind, error in network.worker_errors}
        self.assertIn('layer group differs from this transaction', {error for _rank, error in refusals})
        self.assertIsNotNone(network.driver.error)

    def test_each_composed_field_dropped_from_the_swap_fails(self):
        for field in ('_replies', '_lean_streams', '_tx_fused', '_tx_group'):
            with self.subTest(field=field):
                kept = tuple(name for name in so.COMPOSED_CHUNK_FIELDS if name != field)
                with mock.patch.object(so, 'COMPOSED_CHUNK_FIELDS', kept):
                    network = self.composed(group=2, lean=True)
                    with self.assertRaises((AssertionError, TransportError, DriverError)):
                        self.run_overlapped(network)
                        self.assert_groups(network, 2)
                        self.assertEqual({o.replies for o in self.sent(network, 'flash_s5_open')}, {'lean'})

    def test_without_the_recorder_view_a_successor_opens_full(self):
        """The successor begun with the recorder set aside (C1's form): every successor opens full on a lean arm."""
        with mock.patch.object(so, '_ProgressNoBegin', lambda progress: None):
            network = self.composed(lean=True, fused=False)
            self.run_overlapped(network)
        opens = self.sent(network, 'flash_s5_open')
        self.assertEqual({o.replies for o in opens}, {'lean', 'full'})
        full = {o.admission.transaction for o in opens if o.replies == 'full'}
        self.assertEqual(len(full), network.driver.overlap_counters['successor_frames'])


if __name__ == '__main__':
    unittest.main()
