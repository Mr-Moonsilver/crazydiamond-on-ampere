"""Connected host control proof with declared CPU model/device leaves.

Current helper -> operation -> layer/lane/dispatcher/bridge -> engine profile
agreement. Engine reply-validation uses projected envelopes without completion
reports: no engine startup/publication/healthy request-error claim is made here.
No runtime method/ready flag is patched. These leaves are not box evidence.
"""
import copy
from dataclasses import replace
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'harness'), str(ROOT / 'patches/s5_engine')]
import test_e3_native_submission_credit as c
import test_e3_native_selection_asymmetry as old
import wire_v1 as w
import e3_types as et
import test_e3_r4_worker_bridge as r4
import engine_driver as ed


class ProfileServer(c.CreditServer):
    """Real lane storage lifecycle; tensor/model/stream leaves remain synthetic."""
    def __init__(self, stack, state):
        super().__init__(stack)
        self.runtime = r4.FakeRuntime(state)

    def open(self, message):
        state = self.runtime.state
        state.declare_request(message.transaction.request_uid,
                              message.transaction.request_generation)
        state.prefill.reserve(message.transaction, metadata=message.metadata)
        state.prefill.ready()
        self.stack.lane = state.prefill.handle.lane_id
        return super().open(message)


class ProfileNetwork(r4.Stack):
    """Real coordinator/bridge/QSA pipeline; explicit CPU device and MQ leaves.

    This reuses r4's fake graph/PP/model owners, not a hardware certificate.
    The scheduler/controller can instead be supplied by the pinned-image proof.
    """
    def __init__(self, harness, scheduler=None, controller=None):
        self.h = harness
        self.scheduler = scheduler or r4.Scheduler(r4.Manager(), [r4.Request(tokens=65)])
        self.controller = controller or r4.make_controller(
            self.scheduler, chunk=64, ranks=(0, 1, 2, 3))
        self.record = self.controller.admit(self.scheduler)
        if self.record is None:
            raise AssertionError('connected profile request was not admitted')
        self.sender = r4.RoutingSender()
        self.transport = r4.DirectTransport(self.sender)
        self.executor = r4.FakeExecutor(self.sender, self.transport)
        self.driver = ed.NativeSegmentDriver(
            self.controller, executor=self.executor, scheduler=self.scheduler,
            engine=types.SimpleNamespace(), layer_count=4, layer_split=2,
            stage_of=((0, 0), (1, 0), (2, 1), (3, 1)),
            reply_timeout_s=5., stop_budget=64).install()
        self.workers = {}
        self.qsa = {}
        for rank, stage in ((0, 0), (1, 0), (2, 1), (3, 1)):
            super()._add_worker(rank, stage)
            if stage:
                continue
            worker = self.workers[rank]
            stack = HistoryStack(harness, rank=rank, route='sparse',
                                 transaction=self.record.transaction, ranks=(0, 1))
            # Keep the real constructed WorkerState/route owners from r4.
            # The two event buses are independent, explicitly queried leaves.
            stack.native.split = worker.split
            stack.split = worker.split
            stack.native.controller = worker.controller
            stack.native.router = worker.router
            stack.native.comms.pp = worker.channel
            stack.server = ProfileServer(stack, worker.state)
            stack.dispatcher = c.LaneDispatcher(
                'runner-%d' % rank, stack.native, stack.factory,
                server=stack.server, driver_factory=stack._driver_factory,
                torch_module=stack.torch, clock=lambda: 1)
            stack.bridge = c.worker_bridge.NativeWorkerBridge(
                stack.dispatcher, server=stack.server, source_binder=stack._source_binder)
            worker.dispatcher = stack.dispatcher
            worker._flash_s5_bridge = stack.bridge
            worker.server = stack.server
            worker.factory = stack.factory
            stack.bridge.attach(worker)
            self.qsa[rank] = stack
        self.transport.network = types.SimpleNamespace(workers=self.workers)
        for rank in (2, 3):
            self.worker(rank).channel.ready = True

    def complete_everything(self):
        super().complete_everything()
        for stack in self.qsa.values():
            stack.complete_all()

    def drain_profile_refusal(self):
        for tick in range(400):
            self.complete_everything()
            self.driver.advance(deliver_outputs=False)
            if (self.driver._profile_failed_request is not None
                    and self.driver._return_pending is None
                    and self.driver.stop_kind is None and not self.driver._inflight):
                return tick + 1
        raise AssertionError('profile refusal did not finish its typed all-rank drain')


class HistoryStack(c.Stack):
    def __init__(self, harness, **kwargs):
        super().__init__(harness, **kwargs)
        self.mapping = tuple(range(1, 51))
        self.versions = tuple(range(1, 52))
        self.source = c.sparse_test.T((51, 1, c.PAGE_TOKENS, c.ROW_WIDTH),
                                     device='cpu', dtype='torch.bfloat16', pinned=True)
        self.attention._qsa_host_kv = self.source
        self.attention._qsa_stage_min_rows = 64
        self.main._qsa_host_meta = ((0, self.tokens), (202400,))
        self.main.seq_lens.values = [202400]
        count = harness.selection_counts[self.rank]
        width = (count + self.tokens - 1) // self.tokens + 1
        values = []
        for row in range(self.tokens):
            values.extend(2 * i if i < count else -1
                          for i in range(row * (width - 1), (row + 1) * (width - 1)))
            values.append(-1)
        self.attention.topk_indices_buffer = c.sparse_test.T(
            (self.tokens, width), device='cuda:0', dtype='torch.int32', values=values)
        self.attention.indexer = c.FakeIndexer(self.attention.topk_indices_buffer)


class ScheduleProfile(c.CreditHarness, unittest.TestCase):
    legacy_schedule_source = False
    query_rows = 64
    stack_type = HistoryStack
    selection_counts = (4097, 4097)

    @classmethod
    def setUpClass(cls):
        if not c.REFERENCE.is_dir():
            raise RuntimeError('required reference absent; connected proof NOT_RUN')
        super().setUpClass()

    def boundary(self, ready=(False, False)):
        world = self.world(route='sparse', ranks=(0, 1))
        self.do(world, 'prepare', source=world.lease)
        self.do(world, 'prefix', paused=False)
        self.do(world, 'layer', layer=0, paused=True)
        world.complete_all()
        for rank in world.ranks:
            world.by_rank[rank].device.auto_ready = ready[rank]
        trackers = copy.deepcopy(world.trackers)
        command, acks = world.send('layer', layer=0)
        self.assertTrue(all(a.accepted and a.paused and a.profile_boundary for a in acks.values()))
        self.assertEqual({a.cursor.bucket for a in acks.values()}, {0})
        self.assertEqual([s.operation.stats['sparse_buckets'] for s in world.stacks], [0, 0])
        driver = old.SelectionAsymmetry.driver(self, world, command, trackers)
        driver.admission = world.admission
        driver.plan = [{'kind': w.SegmentKind.LAYER, 'stage': 0, 'layer': 0}]
        skipped = tuple(w.SegmentAcceptance(world.txn, r, command.kind, command.stage,
            command.op_seq, skipped=True) for r in (2, 3))
        driver._apply_segments(tuple(replace(a, report=None) for a in acks.values()) + skipped)
        self.assertEqual(driver.phase, 'profile')
        return world, driver, command

    def query(self, world, driver):
        query = driver._profile_query
        result = tuple(world.by_rank[r].bridge.poll(query) for r in world.ranks)
        return result + tuple(w.QsaProfileObservation(query, r, 'skipped') for r in (2, 3))

    def agree(self, world, driver):
        world.complete_all()
        driver._apply_profile(self.query(world, driver))
        self.assertEqual(driver.phase, 'segments')
        return driver._profile_grants[(0, 0)]

    def test_both_skew_directions_boundary_agreement_and_cached_retry(self):
        for ready in ((True, False), (False, True)):
            with self.subTest(ready=ready):
                world, driver, command = self.boundary(ready)
                observations = self.query(world, driver)
                self.assertEqual([o.status for o in observations[:2]],
                                 ['ready' if flag else 'pending' for flag in ready])
                driver._apply_profile(observations)
                self.assertEqual(driver.phase, 'profile')
                grant = self.agree(world, driver)
                self.assertEqual(grant.shape.bucket_waves, (17,))
                next_command, acks = world.send('layer', layer=0, profile=grant)
                self.assertTrue(all(a.accepted and a.paused for a in acks.values()))
                self.assertEqual([s.operation.stats['copy_events'] for s in world.stacks], [1, 1])
                before = self.submission_state(world)
                for _ in range(1100):
                    self.assertTrue(all(a.accepted for a in world.deliver(next_command).values()))
                self.assertEqual(self.submission_state(world), before)

    def test_missing_grant_is_pending_before_mutation_or_quarantine(self):
        world, driver, boundary = self.boundary((True, True))
        world.complete_all()
        command, acks = world.send('layer', layer=0)
        self.assertTrue(all(a.pending for a in acks.values()))
        before = self.submission_state(world)
        for _ in range(1100):
            self.assertTrue(all(a.pending for a in world.deliver(command).values()))
        self.assertEqual(self.submission_state(world), before)
        self.assertTrue(all(s.dispatcher.quarantined is None for s in world.stacks))

    def test_equal_profile_completes_every_wave_and_retires_source_lifetime(self):
        world, driver, boundary = self.boundary((False, True))
        grant = self.agree(world, driver)
        calls = 0
        while True:
            world.complete_all()
            command, acks = world.send('layer', layer=0, profile=grant)
            self.expect_accepted(world, command, acks)
            calls += 1
            self.assertLess(calls, 30)
            if not next(iter(acks.values())).paused:
                break
        # Seventeen copy submissions, with the attention work on the final
        # wave, then one host-credit pull to finish the layer after its event.
        self.assertEqual(calls, 18)
        self.assertEqual([s.operation.stats['copy_events'] for s in world.stacks], [17, 17])
        self.finish(world)
        self.complete_and_release(world)
        for stack in world.stacks:
            self.assertTrue(stack.operation._released)
            self.assertTrue(all(hold.released for hold in stack.holds))
            self.assertIsNone(stack.bridge.transaction)
            self.assertIsNone(stack.dispatcher.transaction)

    def test_ready_profile_cannot_change_or_regress_while_peer_is_pending(self):
        for mutate in ('pending', 'different'):
            world, driver, boundary = self.boundary((True, False))
            observations = self.query(world, driver)
            driver._apply_profile(observations)
            first = observations[0]
            if mutate == 'pending':
                first = w.QsaProfileObservation(first.query, first.worker_rank, 'pending')
            else:
                first = replace(first, shape=w.QsaScheduleShape('sparse_dma', (16,), (16,)))
            with self.assertRaisesRegex(ed.DriverError, 'schedule changed'):
                driver._apply_profile((first,) + observations[1:])
            self.assertEqual([s.operation.stats['sparse_buckets'] for s in world.stacks], [0, 0])

    def test_actual_4097_97153_refusal_before_bucket_and_typed_worker_cancel(self):
        self.selection_counts = (4097, 97153)
        world, driver, boundary = self.boundary((True, True))
        observations = self.query(world, driver)
        self.assertEqual([o.shape.bucket_waves for o in observations[:2]], [(17,), (380, 1)])
        driver._apply_profile(observations)
        self.assertEqual(driver._profile_failure[1], 'unsupported-native-schedule-profile')
        self.assertEqual(driver._profile_mismatches, 1)
        self.assertEqual(driver.stop_kind, 'abort')
        self.assertIsNone(driver.error)
        self.assertEqual([s.operation.stats['sparse_buckets'] for s in world.stacks], [0, 0])
        cancel = w.CancelRequest(world.txn, world.admission.state_epoch, 'abort')
        world.complete_all()
        for stack in world.stacks:
            self.assertTrue(stack.bridge.cancel(cancel).closed)
            self.assertTrue(stack.operation._released)
            self.assertTrue(all(hold.released for hold in stack.holds))
            self.assertIsNone(stack.bridge.transaction)
            self.assertIsNone(stack.dispatcher.transaction)

    def test_stale_query_or_grant_cannot_authorize_current_plan(self):
        world, driver, boundary = self.boundary((True, True))
        query = driver._profile_query
        for bad in (replace(query, state_epoch=2), replace(query, boundary_op_seq=2),
                    replace(query, layer=1)):
            with self.assertRaises(et.E3ContractError):
                world.by_rank[0].bridge.poll(bad)
        grant = self.agree(world, driver)
        wrong = w.QsaProfileGrant(replace(grant.query, state_epoch=2), grant.shape)
        with self.assertRaises(et.E3ContractError):
            world.send('layer', layer=0, profile=wrong)
        self.assertEqual([s.operation.stats['sparse_buckets'] for s in world.stacks], [0, 0])

    def test_connected_mismatch_cancels_all_ranks_returns_full_and_releases_credits(self):
        self.selection_counts = (4097, 97153)
        network = ProfileNetwork(self)
        ticks = network.drain_profile_refusal()
        self.assertLess(ticks, 400)
        self.assertEqual(network.driver._profile_failure[1], 'unsupported-native-schedule-profile')
        self.assertEqual(network.driver._profile_mismatches, 1)
        self.assertIsNone(network.driver.error)
        self.assertFalse(network.driver._split_ready)
        self.assertFalse(network.controller.active)
        self.assertTrue(network.controller.leases.shutdown()['clean'])
        self.assertEqual(network.transport._calls, {})
        self.assertTrue(network.all_dispatchers_reset())
        self.assertTrue(network.driver._last_native_closure['complete'])
        for stack in network.qsa.values():
            self.assertEqual(stack.operation.stats['sparse_buckets'], 0)
            self.assertTrue(stack.operation._released)
            self.assertTrue(all(hold.released for hold in stack.holds))


if __name__ == '__main__':
    unittest.main()
