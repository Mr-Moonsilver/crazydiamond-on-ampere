"""Native41 held-e: asynchronous TP completion skew through the real driver.

Box run 1 (held-c) died on ``DriverError('TP workers disagree on the layer
continuation')`` 2.5 s into the first automatic native cycle. These tests drive
the connected composition of ``test_e3_native_schedule_profile.ProfileNetwork``
(real ``NativeSegmentDriver``, ``NativeWorkerBridge``, ``LaneDispatcher``,
``QsaLayerDriver`` and ``QsaPrefillOperation`` on stage 0; r4 lanes on stage 1)
with per-rank asynchronous event completion, including the one property the
older all-at-once completion leaves never had: an event recorded on an idle
stream completes at once, as on a GPU.

Declared CPU leaves (``native41_fixture_leaves.json``): the event buses and
the fake sparse device replace CUDA events/streams; ``SkewNetwork`` marks the
stage-0 fake model as not the last PP stage, which the real stage-0 model of a
two-stage pipeline is not. No GPU, box or performance claim.
"""
import contextlib
from dataclasses import replace
import os
from pathlib import Path
import random
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'harness'), str(ROOT / 'patches/s5_engine')]
import test_e3_native_schedule_profile as sp  # noqa: E402
import test_e3_native_submission_credit as credit  # noqa: E402
import test_e3_r4_worker_bridge as r4  # noqa: E402
import test_e3_sparse_history_dma as sparse_test  # noqa: E402
import engine_driver as ed  # noqa: E402
import wire_v1 as w  # noqa: E402

FINALIZE = w.SegmentKind.STATE_FINALIZE
LAYER = w.SegmentKind.LAYER


class SkewNetwork(sp.ProfileNetwork):
    """ProfileNetwork with per-rank completion control and agreement capture."""

    def __init__(self, harness):
        super().__init__(harness)
        for stack in self.qsa.values():
            # Leaf: the real stage-0 model of a two-stage pipeline is not the
            # last PP stage, so its suffix produces the HC tensor PP_SEND needs.
            stack.model._flash_e3_is_last_stage = lambda: False
        self.held = set()
        self.idle = set()
        self.agreements = []
        driver = self.driver
        real = driver._apply_segments

        def capture(values):
            values = tuple(values)
            command = driver._command
            self.agreements.append(dict(
                kind=command.kind, stage=command.stage,
                layer=command.descriptor.layer_start, op_seq=command.op_seq,
                rows={v.worker_rank: (v.accepted, v.pending, v.paused, v.cursor)
                      for v in values if v.worker_rank in driver._targets}))
            return real(values)
        driver._apply_segments = capture

    # -- per-rank device progress -------------------------------------------
    def buses(self, rank):
        buses = [self.workers[rank].bus]
        if rank in self.qsa:
            buses.append(self.qsa[rank].bus)
        return buses

    def complete(self, ranks=(0, 1, 2, 3)):
        for rank in ranks:
            for bus in self.buses(rank):
                for event in bus.events:
                    if id(event) not in self.held:
                        event.complete = True
            if rank in self.qsa:
                for event in self.qsa[rank].device.events:
                    if id(event) not in self.held:
                        event.ready = True

    def hold_new_events(self, rank, call):
        """Run ``call`` and keep every event it created on ``rank`` incomplete."""
        before = [len(bus.events) for bus in self.buses(rank)]
        device = self.qsa[rank].device if rank in self.qsa else None
        device_before = len(device.events) if device is not None else 0
        try:
            return call()
        finally:
            for bus, start in zip(self.buses(rank), before):
                self.held.update(id(event) for event in bus.events[start:])
            if device is not None:
                self.held.update(id(event) for event in device.events[device_before:])

    def release_held(self):
        self.held.clear()

    def wrap_submit(self, rank, hook):
        bridge = self.qsa[rank].bridge if rank in self.qsa else self.workers[rank]._flash_s5_bridge
        real = bridge.submit
        bridge.submit = lambda command: hook(command, lambda: real(command))

    def methods(self):
        return [envelope[0] for envelope in self.sender.envelopes]


@contextlib.contextmanager
def idle_streams(network):
    """An event recorded on an idle stream (every earlier event done) completes."""
    def wrap(original):
        def record(event, stream=None):
            original(event, stream)
            if id(event.bus) in network.idle:
                earlier = event.bus.events[:event.bus.events.index(event)]
                if all(e.complete for e in earlier):
                    event.complete = True
        return record
    with mock.patch.object(r4.ControllableEvent, 'record', wrap(r4.ControllableEvent.record)), \
         mock.patch.object(credit.ControllableEvent, 'record',
                           wrap(credit.ControllableEvent.record)):
        yield


class TPCompletionSkew(credit.CreditHarness, unittest.TestCase):
    legacy_schedule_source = False
    query_rows = 64
    stack_type = sp.HistoryStack
    selection_counts = (4097, 4097)

    @classmethod
    def setUpClass(cls):
        if not credit.REFERENCE.is_dir():
            raise RuntimeError('required reference absent; connected proof NOT_RUN')
        super().setUpClass()

    def drive(self, network, until, limit=600, complete=None):
        for tick in range(limit):
            (complete or network.complete)()
            network.driver.advance(deliver_outputs=False)
            if until():
                return tick + 1
        self.fail('driver did not reach the expected state: phase=%s index=%s'
                  % (network.driver.phase, network.driver.index))

    def assert_clean_finish(self, network):
        self.assertIsNone(network.driver.error)
        self.assertFalse(network.driver.active)
        self.assertTrue(network.all_dispatchers_reset())
        self.assertTrue(network.controller.leases.shutdown()['clean'])
        self.assertEqual(network.transport._calls, {})

    # -- the box failure ----------------------------------------------------
    def test_state_finalize_completion_skew_is_one_continuation(self):
        """One rank's state event done at record (idle stream), its peer's not.

        The parent agreement refuses exactly as on the box. The candidate
        treats it as one continuation, and the release still waits for the
        held rank's real completion.
        """
        for stage, idle_rank, late_rank in ((0, 0, 1), (0, 1, 0), (1, 2, 3)):
            with self.subTest(stage=stage, idle=idle_rank, late=late_rank):
                network = SkewNetwork(self)
                with idle_streams(network):
                    def idle_hook(command, call, rank=idle_rank):
                        if command.kind is FINALIZE and command.stage == stage:
                            network.idle.update(id(bus) for bus in network.buses(rank))
                            try:
                                return call()
                            finally:
                                network.idle.clear()
                        return call()

                    def late_hook(command, call, rank=late_rank):
                        if command.kind is FINALIZE and command.stage == stage:
                            return network.hold_new_events(rank, call)
                        return call()
                    network.wrap_submit(idle_rank, idle_hook)
                    network.wrap_submit(late_rank, late_hook)
                    self.drive(network, lambda: any(
                        a['kind'] is FINALIZE and a['stage'] == stage
                        for a in network.agreements))
                    finalize = [a for a in network.agreements
                                if a['kind'] is FINALIZE and a['stage'] == stage][0]
                    phases = {rank: row[3].phase for rank, row in finalize['rows'].items()}
                    self.assertEqual(phases[idle_rank], 'commit')
                    self.assertEqual(phases[late_rank], 'state_finalize')
                    self.assertEqual({row[2] for row in finalize['rows'].values()}, {False})
                    self.assertEqual(len({(row[3].layer, row[3].bucket)
                                          for row in finalize['rows'].values()}), 1)
                    # The late rank's finalize work is held: the plan may run
                    # to its end, but the lane is never released before it.
                    released_before = self.methods_after(network, 'flash_s5_release')
                    for _ in range(40):
                        network.complete()
                        network.driver.advance(deliver_outputs=False)
                    self.assertIsNone(network.driver.error)
                    self.assertTrue(network.driver.active)
                    self.assertEqual(self.methods_after(network, 'flash_s5_release'),
                                     released_before)
                    network.release_held()
                    self.drive(network, lambda: not network.driver.active)
                self.assert_clean_finish(network)
                self.assertEqual(network.methods().count('flash_s5_release'), 1)

    @staticmethod
    def methods_after(network, method):
        return network.methods().count(method)

    # -- delayed peer: retained acceptance, no repeat, no early tail ---------
    def test_delayed_peer_retains_accepted_and_never_repeats(self):
        """Rank 1's device completes nothing for 120 polling rounds while rank 0
        completes everything. The engine retries one retained command; rank 0
        never repeats a submission, copy or continuation, and neither rank
        enters the layer's TP tail before both reach its boundary."""
        network = SkewNetwork(self)
        driver = network.driver

        def buckets():
            layers = driver.qsa_control.snapshot()['layers']
            return layers[0]['bucket_accepted_commands'] if layers else 0
        self.drive(network, lambda: buckets() >= 1)
        rank0 = network.qsa[0]

        def state(stack):
            return (tuple(sorted(stack.dispatcher.accepted)), dict(stack.operation.stats),
                    len(stack.device.events), len(stack.device.copies),
                    len(stack.device.page_copies), len(stack.bus.events),
                    sum(1 for r in stack.driver.records if r[0] == 'tail'))
        start = len(network.agreements)
        frozen = None
        for _ in range(120):
            network.complete(ranks=(0, 2, 3))
            driver.advance(deliver_outputs=False)
            self.assertIsNone(driver.error)
            if frozen is None and any(
                    a['kind'] is LAYER and a['stage'] == 0 and a['rows'][0][0] and a['rows'][1][1]
                    for a in network.agreements[start:]):
                frozen = state(rank0)
            if frozen is not None:
                self.assertEqual(state(rank0), frozen)
            for stack in network.qsa.values():
                self.assertFalse(any(r[0] == 'tail' and r[1] == 0 for r in stack.driver.records))
            self.assertEqual(driver.qsa_control.snapshot()['layers'][0]['tail_accepted_commands'], 0)
        self.assertIsNotNone(frozen, 'rank 1 never went pending beside an accepted rank 0')
        retries = [a for a in network.agreements[start:]
                   if a['kind'] is LAYER and a['rows'][0][0] and a['rows'][1][1]]
        self.assertGreater(len(retries), 10)
        # Every retry names the one retained command: no new op_seq while pending.
        self.assertEqual(len({a['op_seq'] for a in retries}), 1)
        self.drive(network, lambda: not driver.active)
        self.assert_clean_finish(network)
        for stack in network.qsa.values():
            self.assertEqual(sum(1 for r in stack.driver.records if r[0] == 'tail' and r[1] == 0), 1)

    # -- a real divergence still refuses -------------------------------------
    def divergent(self, kind, stage, mutate):
        network = SkewNetwork(self)
        driver = network.driver
        real = driver._apply_segments
        done = []

        def tamper(values):
            values = tuple(values)
            command = driver._command
            if (not done and command.kind is kind and command.stage == stage
                    and all(v.accepted for v in values if v.worker_rank in driver._targets)):
                done.append(command.op_seq)
                values = tuple(mutate(v) if v.worker_rank == driver._targets[-1] else v
                               for v in values)
            return real(values)
        driver._apply_segments = tamper
        with self.assertRaises(ed.DriverError) as caught:
            self.drive(network, lambda: not driver.active)
        self.assertTrue(done)
        return str(caught.exception)

    def test_divergent_cursor_or_pause_still_refuses(self):
        cursor = w.CompletionCursor
        cases = (
            ('layer-bucket', LAYER, 0, lambda v: replace(
                v, cursor=cursor(v.cursor.phase, v.cursor.layer, v.cursor.bucket + 1))),
            ('layer-index', LAYER, 0, lambda v: replace(
                v, cursor=cursor(v.cursor.phase, v.cursor.layer + 1, v.cursor.bucket))),
            ('layer-commit-is-not-normalized', LAYER, 0, lambda v: replace(
                v, cursor=cursor('commit', v.cursor.layer, v.cursor.bucket))),
            ('finalize-other-phase', FINALIZE, 0, lambda v: replace(
                v, cursor=cursor('layer', v.cursor.layer, v.cursor.bucket))),
            ('finalize-other-layer', FINALIZE, 1, lambda v: replace(
                v, cursor=cursor('commit', v.cursor.layer + 1, v.cursor.bucket))),
            ('suffix-commit', w.SegmentKind.SUFFIX, 0, lambda v: replace(
                v, cursor=cursor('commit', v.cursor.layer, v.cursor.bucket))),
        )
        for name, kind, stage, mutate in cases:
            with self.subTest(name):
                message = self.divergent(kind, stage, mutate)
                self.assertIn('TP workers disagree on the layer continuation', message)
                detail = message[message.index(' ['):]
                self.assertLessEqual(len(detail), 1024)
                self.assertIn('[command=%s stage=%d ' % (kind.value, stage), message)
                self.assertRegex(message, r'uid_sha=[0-9a-f]{16} ')
                ranks = (0, 1) if stage == 0 else (2, 3)
                for rank in ranks:
                    self.assertRegex(message, r'r%d:acc=1,pend=0,paused=[01],cursor=\w+/\d+/\d+,'
                                              r'profile=[01],tail=[01]' % rank)

    def test_paused_disagreement_still_refuses(self):
        def unpause(v):
            # Only a paused accepted layer may carry a boundary flag.
            return replace(v, paused=not v.paused, profile_boundary=False, tail_boundary=False)
        message = self.divergent(LAYER, 0, unpause)
        self.assertIn('TP workers disagree on the', message)
        self.assertIn('[command=layer stage=0 ', message)

    # -- bounded random sweep -----------------------------------------------
    def test_random_completion_skew_sweep(self):
        """In-order partial per-rank completion, idle-stream completion at
        record, and progress between the per-rank deliveries of one round."""
        mixed = 0
        for seed in range(32):
            with self.subTest(seed=seed):
                rnd = random.Random(4100 + seed)
                network = SkewNetwork(self)
                rate = {rank: rnd.choice((1.0, 0.6, 0.3, 0.1)) for rank in range(4)}

                def progress(rank):
                    lists = [bus.events for bus in network.buses(rank)]
                    for events in lists:
                        pending = [e for e in events if not e.complete]
                        count = sum(rnd.random() < rate[rank] for _ in pending)
                        for event in pending[:count]:
                            event.complete = True
                    if rank in network.qsa:
                        device = network.qsa[rank].device
                        for token in {e.token for e in device.events}:
                            pending = [e for e in device.events if e.token == token and not e.ready]
                            count = sum(rnd.random() < rate[rank] for _ in pending)
                            for event in pending[:count]:
                                event.ready = True

                def tick():
                    for rank in range(4):
                        progress(rank)
                for rank in range(4):
                    def hook(command, call, rank=rank):
                        for other in range(4):
                            if rnd.random() < 0.5:
                                progress(other)
                        idle = rnd.random() < 0.5
                        if idle:
                            network.idle.update(id(bus) for bus in network.buses(rank))
                        try:
                            return call()
                        finally:
                            network.idle.clear()
                    network.wrap_submit(rank, hook)
                with idle_streams(network):
                    self.drive(network, lambda: not network.driver.active, limit=4000,
                               complete=tick)
                self.assert_clean_finish(network)
                mixed += sum(1 for a in network.agreements if a['kind'] is FINALIZE
                             and len({row[3].phase for row in a['rows'].values()}) > 1)
        # The sweep actually exercised the skew the box hit.
        self.assertGreater(mixed, 0)


class FakeCuda:
    """Leaf for torch.cuda's host allocator/driver queries on one rank."""

    def __init__(self, rank, initialized=True, fail=False):
        self.rank = rank
        self.initialized = initialized
        self.fail = fail
        self.calls = 0

    def is_initialized(self):
        return self.initialized

    def memory_stats(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError('allocator query failed')
        base = 1000 * (self.rank + 1)
        return {'allocated_bytes.all.current': base + self.calls,
                'reserved_bytes.all.current': base + 500,
                'inactive_split_bytes.all.current': 7 + self.rank,
                'num_alloc_retries': self.calls,
                'num_ooms': 0}

    def mem_get_info(self):
        return (300 + self.rank, 24 * 1024)


class AutoArmMemoryReceipts(credit.CreditHarness, unittest.TestCase):
    """Per-rank allocator receipts at each chunk open and after its drained
    release, only with FLASH_S5_AUTO_ARM=1, never a control input."""
    legacy_schedule_source = False
    query_rows = 64
    stack_type = sp.HistoryStack
    selection_counts = (4097, 4097)

    @classmethod
    def setUpClass(cls):
        if not credit.REFERENCE.is_dir():
            raise RuntimeError('required reference absent; connected proof NOT_RUN')
        super().setUpClass()

    def network(self, switch, **cuda):
        with mock.patch.dict(os.environ, {'FLASH_S5_AUTO_ARM': switch}):
            network = SkewNetwork(self)
        bridges = {rank: (network.qsa[rank].bridge if rank in network.qsa
                          else network.workers[rank]._flash_s5_bridge) for rank in range(4)}
        fakes = {}
        for rank, bridge in bridges.items():
            fakes[rank] = bridge._memory_cuda = FakeCuda(rank, **cuda)
        return network, bridges, fakes

    def finish(self, network):
        for _ in range(600):
            network.complete()
            network.driver.advance(deliver_outputs=False)
            if not network.driver.active:
                break
        self.assertIsNone(network.driver.error)
        self.assertFalse(network.driver.active)
        self.assertTrue(network.all_dispatchers_reset())

    def test_switch_on_records_open_and_released_per_rank(self):
        network, bridges, fakes = self.network('1')
        tx = network.record.transaction
        with self.assertLogs('worker_bridge', level='INFO') as logs:
            self.finish(network)
        for rank, bridge in bridges.items():
            rows = bridge.memory_rows
            self.assertEqual([row['point'] for row in rows], ['open', 'released'])
            self.assertEqual(bridge.memory_receipt_errors, 0)
            for row in rows:
                self.assertEqual((row['rank'], row['txn_seq'], row['chunk_start'], row['chunk_end']),
                                 (rank, tx.txn_seq, tx.chunk_start, tx.chunk_end))
                self.assertEqual(set(row), {'point', 'rank', 'txn_seq', 'chunk_start', 'chunk_end',
                                            'allocated', 'reserved', 'inactive_split',
                                            'alloc_retries', 'ooms', 'free', 'total'})
                self.assertEqual((row['free'], row['total']), (300 + rank, 24 * 1024))
                self.assertEqual(row['inactive_split'], 7 + rank)
            self.assertEqual(fakes[rank].calls, 2)
        lines = [line for line in logs.output if 'flash-s5 memory ' in line]
        self.assertEqual(len(lines), 8)
        self.assertTrue(all(len(line) < 400 for line in lines))

    def test_switch_off_never_queries_and_keeps_the_same_commands(self):
        on, _, _ = self.network('1')
        self.finish(on)
        off, bridges, fakes = self.network('0')
        self.finish(off)
        for rank, bridge in bridges.items():
            self.assertEqual(bridge.memory_rows, [])
            self.assertEqual(fakes[rank].calls, 0)
        # The receipt changes no RPC: both runs send the same method sequence.
        self.assertEqual(on.methods(), off.methods())

    def test_uninitialized_or_failing_cuda_is_counted_never_raised(self):
        network, bridges, fakes = self.network('1', initialized=False)
        self.finish(network)
        for rank, bridge in bridges.items():
            self.assertEqual((bridge.memory_rows, bridge.memory_receipt_errors, fakes[rank].calls),
                             ([], 0, 0))
        network, bridges, fakes = self.network('1', fail=True)
        self.finish(network)
        for bridge in bridges.values():
            self.assertEqual(bridge.memory_rows, [])
            self.assertEqual(bridge.memory_receipt_errors, 2)


if __name__ == '__main__':
    unittest.main()
