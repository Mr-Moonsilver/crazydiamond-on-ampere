"""C0 over the real QSA driver and operation: sparse profile agreement, buckets,
TP tail entry, stop at every tick.  Stage 0 is the real generated QSA stack of
``test_e3_native_schedule_profile``; leaves as declared there and in
``stage_fixture_leaves.json``.  CPU only; no pass here is native evidence.
"""
import types
import unittest

import test_e3_native_schedule_profile as profile
from stage_net import RingNetworkMixin, ed, r4, w
from stage_progress import StagedSegmentDriver
from e3_types import E3Pending


class RingProfile(RingNetworkMixin, profile.ProfileNetwork):
    """ProfileNetwork's real QSA stack on the per-rank ring and real reply transport."""

    def __init__(self, harness, staged=True):
        self.h = harness
        self.direct = False
        self.expected_commits = 1
        self.scheduler = r4.Scheduler(r4.Manager(), [r4.Request(tokens=65)])
        self.controller = r4.make_controller(self.scheduler, chunk=64, ranks=(0, 1, 2, 3))
        self.record = self.controller.admit(self.scheduler)
        self.install_network()
        driver_class = StagedSegmentDriver if staged else ed.NativeSegmentDriver
        self.driver = driver_class(
            self.controller, executor=self.executor, scheduler=self.scheduler,
            engine=types.SimpleNamespace(), layer_count=4, layer_split=2,
            stage_of=((0, 0), (1, 0), (2, 1), (3, 1)),
            reply_timeout_s=30., stop_budget=256).install()
        self.workers = {}
        self.qsa = {}
        for rank, stage in ((0, 0), (1, 0), (2, 1), (3, 1)):
            r4.Stack._add_worker(self, rank, stage)
            if stage:
                continue
            worker = self.workers[rank]
            stack = profile.HistoryStack(harness, rank=rank, route='sparse',
                                         transaction=self.record.transaction, ranks=(0, 1))
            stack.native.split = worker.split
            stack.split = worker.split
            stack.native.controller = worker.controller
            stack.native.router = worker.router
            stack.native.comms.pp = worker.channel
            stack.server = profile.ProfileServer(stack, worker.state)
            stack.dispatcher = profile.c.LaneDispatcher(
                'runner-%d' % rank, stack.native, stack.factory,
                server=stack.server, driver_factory=stack._driver_factory,
                torch_module=stack.torch, clock=lambda: 1)
            stack.bridge = profile.c.worker_bridge.NativeWorkerBridge(
                stack.dispatcher, server=stack.server, source_binder=stack._source_binder)
            # Leaf as in SkewNetwork: stage 0 of a two-stage model is not the last stage.
            stack.model._flash_e3_is_last_stage = lambda: False
            worker.dispatcher = stack.dispatcher
            worker._flash_s5_bridge = stack.bridge
            worker.server = stack.server
            worker.factory = stack.factory
            stack.bridge.attach(worker)
            self.qsa[rank] = stack
        for rank in (2, 3):
            self.worker(rank).channel.ready = True

    def tick(self, complete=True):
        if complete:
            self.complete_everything()
        self.pump()
        return self.driver.advance(deliver_outputs=False)


class StagedQsa(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        profile.ScheduleProfile.setUpClass()

    def network(self, staged=True):
        case = profile.ScheduleProfile('runTest')
        case.setUp()
        self.addCleanup(case.doCleanups)
        network = RingProfile(case, staged=staged)
        self.addCleanup(network.close_network)
        return network

    def counters(self, network):
        rows = network.driver.qsa_control.snapshot()['layers']
        keys = ('stage', 'layer', 'boundary_commands', 'agreed_buckets', 'agreed_copy_waves',
                'bucket_commands', 'bucket_accepted_commands', 'tail_boundaries', 'tail_commands',
                'tail_accepted_commands', 'layer_commands')
        return [{key: row[key] for key in keys} for row in rows]

    def copies(self, network):
        return {rank: dict(copies=stack.operation.stats['copy_events'],
                           buckets=stack.operation.stats['sparse_buckets'])
                for rank, stack in network.qsa.items()}

    def test_real_qsa_chunk_commits_with_trunk_counts(self):
        off = self.network(staged=False)
        off.run_request(limit=800)
        on = self.network(staged=True)
        on.run_request(limit=800)
        self.assertEqual(off.worker_errors, [])
        self.assertEqual(on.worker_errors, [])
        self.assertEqual(self.counters(on), self.counters(off))
        self.assertEqual(self.copies(on), self.copies(off))
        self.assertEqual(self.copies(on)[0]['copies'], 17)
        self.assertEqual(on.controller.commits, 1)
        self.assertEqual(on.scheduler.kv_cache_manager.cache_calls,
                         off.scheduler.kv_cache_manager.cache_calls)
        self.assertEqual(on.driver.measurement_census._faults, [])

    def test_only_active_stage_answers_profile_and_tail(self):
        network = self.network()
        network.run_request(limit=800)
        queries = [args[0] for method, args, _k, _r in network.sent_envelopes
                   if method == 'flash_s5_poll' and type(args[0].message) is w.QsaProfileQuery]
        self.assertTrue(queries)
        self.assertEqual({scoped.ranks for scoped in queries}, {(0, 1)})
        layers = [args[0] for method, args, _k, _r in network.sent_envelopes
                  if method == 'flash_s5_submit'
                  and args[0].message.kind is w.SegmentKind.LAYER]
        self.assertEqual({(s.ranks, s.message.stage) for s in layers},
                         {((0, 1), 0), ((2, 3), 1)})

    def test_stop_at_every_tick(self):
        probe = self.network()
        count = [0]

        def tick():
            count[0] += 1
            return probe.tick()
        probe.run_request(tick=tick, limit=800)
        total = count[0]
        self.assertGreater(total, 30)
        for cut in range(0, total):
            with self.subTest(cut=cut):
                network = self.network()
                for _ in range(cut):
                    network.tick()
                commits = network.controller.commits
                network.driver.stop('abort')
                for _ in range(800):
                    if not network.driver.active:
                        break
                    network.tick()
                self.assertFalse(network.driver.active)
                self.assertEqual(network.worker_errors, [])
                self.assertEqual(network.controller.commits, commits)
                self.assertIsNone(network.driver.error)
                self.assertTrue(network.controller.leases.shutdown()['clean'])
                self.assertEqual(network.transport._calls, {})
                self.assertIsNone(network.controller.parked)
                for rank, worker in network.workers.items():
                    self.assertIsNone(worker._flash_s5_bridge.transaction, rank)
                    self.assertIsNone(worker._flash_s5_bridge._retained, rank)

    def test_tail_peer_pending_is_replayed_before_cancel(self):
        network = self.network()
        driver = network.driver
        calls = {0: 0, 1: 0}
        for rank, stack in network.qsa.items():
            def counted(state, real=stack.attention._flash_e3_attention_post, rank=rank):
                calls[rank] += 1
                return real(state)
            stack.attention._flash_e3_attention_post = counted
        for _ in range(400):
            network.tick()
            if driver._frames and driver._frames[0].stage[0]['_tail_boundary'] is not None:
                break
        else:
            self.fail('no tail boundary')
        # The tail command is on the ring, read by no rank yet.
        self.assertEqual(driver._frames[0].stage[0]['_tail_submission'].accepted, set())
        # Peer 1 answers the tail command typed pending, before any mutation.
        dispatcher = network.qsa[1].dispatcher
        real_submit = dispatcher.submit
        hold = [True]

        def pending_tail(command, real=real_submit):
            if hold[0] and command.profile is not None:
                raise E3Pending('test: tail credit')
            return real(command)
        dispatcher.submit = pending_tail
        for _ in range(40):
            network.tick()
            obligation = driver._frames[0].stage[0]['_tail_submission']
            if obligation is not None and obligation.accepted == {0}:
                break
        else:
            self.fail('no partial tail acceptance')
        self.assertEqual(calls, {0: 1, 1: 0})
        command = obligation.command
        driver.stop('abort')
        for _ in range(6):
            network.tick()
        # Nothing cancels while the peer owes the tail: rank 0 entered its collective.
        self.assertEqual(obligation.accepted, {0})
        self.assertTrue(driver.active)
        self.assertIsNone(network.qsa[0].dispatcher.cancelled)
        self.assertIsNone(network.qsa[1].dispatcher.cancelled)
        hold[0] = False
        for _ in range(800):
            if not driver.active:
                break
            network.tick()
        self.assertFalse(driver.active)
        self.assertEqual(calls, {0: 1, 1: 1})
        self.assertEqual(network.worker_errors, [])
        self.assertIsNone(driver.error)
        self.assertTrue(driver._last_native_closure['complete'])
        replays = [args[0].message for method, args, _k, _r in network.sent_envelopes
                   if method == 'flash_s5_submit' and args[0].message.op_seq == command.op_seq]
        self.assertGreaterEqual(len(replays), 2)
        self.assertTrue(all(item == command for item in replays))


if __name__ == '__main__':
    unittest.main()
