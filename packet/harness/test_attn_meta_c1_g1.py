#!/usr/bin/env python3
"""C1 (two chunks in flight) with and without the deferred PP bind, composed at layer-group size 1 (the attn-meta package's
G1 cells P and MP; lead amendment A, 3 Nov: "confirm in code that C1 defer composes with group size 1").

Code reading (c1 fbb9123e): size 1 is in wire_v1.LAYER_GROUP_SIZES and engine_driver._layer_group_control accepts it without
fused; a chunk opened at size 1 carries ``_tx_group == 1`` (the ungrouped Step A command stream: _build_plan is per layer,
grouping only coalesces LAYER submits when ``_tx_group != 1``); stage_overlap's deferred bind keys on plan kinds (PP_SEND) and
bindings, never on the group; COMPOSED_CHUNK_FIELDS keeps ``_tx_group`` per frame. alpha-next-c1's composed-field suite runs
g2 / g4 only (and fused=False without groups); this suite closes the G1 gap with the same connected fixture
(test_alpha_next_c1_composed_fields.Fixture over test_e3_stage_overlap's ring: real StagedAdmissionController,
OverlapSegmentDriver, four NativeWorkerBridge / LaneDispatcher workers), layer groups installed and selected at size 1, fused
selected, as the box arms are (FLASH_S5_LAYER_GROUPS=1, FLASH_S5_QSA_FUSED=1, per-arm size 1).

Per defer setting (off: C1 on; on: C1 defer):
  - two chunks in flight, every commit made, one command in flight per stage, no driver error;
  - every OPEN carries (qsa_fused True, layer_group 1); every LAYER submit is ungrouped (layer_group 1); the grouped-chunk
    counters stay 0 (size 1 is not a grouped chunk);
  - the per-rank device ledgers and persistent state equal the serial driver's at size 1;
  - with defer on: deferred opens and binds happen (the deferred bind engaged); with defer off: none.
Mutant: a driver whose open request carries the selected size even at 1 as a grouped chunk (``_tx_group`` forced to the
installed size 2 while the arm says 1) is seen by the OPEN / submit assertions (layer_group 2 appears).
Declared leaves: those of alpha-next-c1's composed-field suite (harness/alpha_next_c1_engine_fixture_leaves.json).
"""
import unittest

import test_alpha_next_c1_composed_fields as cf


class Base(cf.Fixture):
    def g1(self, defer, network=None):
        network = self.composed(group=1, network=network)
        network.driver.defer_bind = defer
        return network

    def assert_g1(self, network):
        layers = [m for m in self.sent(network, 'flash_s5_submit') if m.descriptor.kind is cf.w.SegmentKind.LAYER]
        self.assertTrue(layers)
        self.assertEqual({m.descriptor.layer_group for m in layers}, {1})
        opens = self.sent(network, 'flash_s5_open')
        self.assertEqual(len(opens), 2 * network.expected_commits)
        self.assertEqual({(o.qsa_fused, o.layer_group) for o in opens}, {(True, 1)})
        self.assertEqual(network.driver.layer_groups.size, 1)
        self.assertEqual(network.driver.layer_groups.counters['chunks'], 0)


class G1(Base):
    def test_c1_at_g1_two_chunks_in_flight(self):
        for defer in (False, True):
            with self.subTest(defer=defer):
                network = self.g1(defer)
                self.run_overlapped(network)
                self.assert_g1(network)
                counters = network.driver.defer_counters
                if defer:
                    self.assertGreater(counters['deferred_opens'], 0)
                    self.assertGreater(counters['deferred_binds'], 0)
                else:
                    self.assertEqual(counters, dict(deferred_opens=0, deferred_binds=0))

    def test_c1_at_g1_matches_serial_at_g1(self):
        for defer in (False, True):
            with self.subTest(defer=defer):
                off = self.composed(group=1, network=self.serial())
                off.run_request()
                on = self.g1(defer)
                self.run_overlapped(on)
                self.assertEqual(off.worker_errors, [])
                self.assertEqual(on.worker_errors, [])
                for rank in (0, 1, 2, 3):
                    self.assertEqual(on.ledgers[rank], off.ledgers[rank], rank)
                self.assertEqual(on.persistent_state(), off.persistent_state())
                self.assertLessEqual(max(on.inflight_peak.values()), 1, on.inflight_peak)
                self.assertEqual(on.controller.overlap_counters['promoted'],
                                 on.driver.overlap_counters['overlapped_commits'])


class Mutants(Base):
    def test_grouped_at_size_one_is_seen(self):
        network = self.g1(True)
        network.driver.layer_groups.size = 2       # the mutant: chunks open grouped although the arm is G1
        self.run_overlapped(network)
        with self.assertRaises(AssertionError):
            self.assert_g1(network)


if __name__ == '__main__':
    unittest.main()
