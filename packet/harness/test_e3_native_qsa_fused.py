"""Step A connected host proof: one bounded dense QSA operation per layer.

Real NativeSegmentDriver, bridge, dispatcher, QsaLayerDriver, QsaPrefillOperation
and generated attention over the declared CPU leaves of
harness/step_a_fixture_leaves.json. Not a hardware certificate: real-GPU event
order and bytes are the V3 check.
"""
from dataclasses import replace
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'harness'), str(ROOT / 'patches/s5_engine')]
import test_e3_native_submission_credit as c
import test_e3_native_schedule_profile as sp
import test_e3_r4_worker_bridge as r4
import test_e3_sparse_history_dma as base
import e3_types as et
import engine_driver as ed
import lane_comm
import qsa_driver
import qsa_operation as qo
import wire_v1 as w

# Chunk end 202,400 is 50 pages of history: three dense buckets, the arena
# boundary crossed twice.
HISTORY = 202336
BUCKETS = 3


class NoQsaDriver(r4.FakeDriver):
    """Leaf: stage 1 owns no QSA layer in this model."""
    def fused_profiles(self, lane, tokens):
        return ()


class LeafPynccl:
    """Leaf: the PyNccl object under the real PrefillTpCommunicator."""
    disabled = False
    world_size = 2
    unique_name = 'tp-lane'

    def __init__(self, rank):
        self.rank = rank
        self.calls = 0

    def all_reduce(self, tensor, stream=None):
        self.calls += 1
        return tensor


# Leaf: the torch surface the communicator reads, the lane's current stream.
TP_TORCH = types.SimpleNamespace(cuda=types.SimpleNamespace(current_stream=lambda device: 'lane'))


class FusedNetwork(sp.ProfileNetwork):
    def __init__(self, harness, fused=True, owned=True):
        scheduler = r4.Scheduler(r4.Manager(), [r4.Request(tokens=HISTORY + 65, computed=HISTORY)])
        controller = r4.make_controller(scheduler, chunk=64, ranks=(0, 1, 2, 3),
                                        max_history=HISTORY + 64)
        super().__init__(harness, scheduler=scheduler, controller=controller)
        # Leaf: the auto-arm policy's ownership state (real one: test_native41_autoarm).
        self.driver.auto_arm = types.SimpleNamespace(
            manual_owned=not owned, phase='native', active=False)
        self.driver._admission_intent = {'origin': 'auto-arm', 'phase': 'native', 'seq': 1}
        self.fused = self.driver.install_qsa_fused()
        self.fused.selected = fused
        for stack in self.qsa.values():
            # attach_qsa_host_metadata: valid tokens are the admission's chunk end.
            end = stack.transaction.chunk_end
            stack.main._qsa_host_meta = ((0, stack.tokens), (end,))
            stack.main.seq_lens.values = [end]
            stack.device.native_qsa_policy = 'dense'
            # The stage 0 sender: its suffix produces the PP send tensor.
            stack.model._flash_e3_is_last_stage = lambda: False
        for rank in (2, 3):
            self.worker(rank).dispatcher.driver_factory = (
                lambda operation, module, context: NoQsaDriver())

    def tick(self, ready=(0, 1)):
        r4.Stack.complete_everything(self)
        for rank in ready:
            self.qsa[rank].complete_all()
        self.driver.advance(deliver_outputs=False)

    def run_until(self, done, *, ready=(0, 1), limit=200):
        for _ in range(limit):
            self.tick(ready)
            if done():
                return
        raise AssertionError('connected fused chunk did not reach the expected state; '
                             'phase=%s index=%s error=%s' % (self.driver.phase, self.driver.index,
                                                             self.driver.error))

    def finish(self, ready=(0, 1)):
        started = [self.driver.tx is not None]
        def done():
            started[0] = started[0] or self.driver.tx is not None
            return started[0] and self.driver.tx is None
        self.run_until(done, ready=ready)

    def pump_stop(self, ready=(0, 1), limit=200):
        for _ in range(limit):
            r4.Stack.complete_everything(self)
            for rank in ready:
                self.qsa[rank].complete_all()
            self.driver.pump_stop(1)
            if not self.driver.active and not self.driver._inflight:
                return
        raise AssertionError('stop did not close; phase=%s error=%s'
                             % (self.driver.phase, self.driver.error))

    def sent(self, kind=None, layer=None):
        return [args[0] for method, args, _, _ in self.sender.envelopes
                if method == 'flash_s5_submit'
                and (kind is None or args[0].kind is kind)
                and (layer is None or args[0].descriptor.layer_start == layer)]

    def profile_polls(self):
        return [args[0] for method, args, _, _ in self.sender.envelopes
                if method == 'flash_s5_poll' and type(args[0]) is w.QsaProfileQuery]

    def layer_accepted(self, rank, layer=0):
        return any(stored[0].kind == 'layer' and stored[0].layer_start == layer
                   for stored in self.qsa[rank].dispatcher.accepted.values())

    def install_tp(self):
        """The real communicator and ledger; the post seam enters it as o_proj does."""
        self.tp = {}
        for rank, stack in self.qsa.items():
            leaf = LeafPynccl(rank % 2)
            comm = lane_comm.PrefillTpCommunicator(leaf, 'cuda:0', torch_module=TP_TORCH)
            stack.native.comms.tp = comm
            self.tp[rank] = comm
            original = stack.attention._flash_e3_attention_post
            def post(state, comm=comm, stack=stack, original=original):
                comm.all_reduce(stack.output)
                return original(state)
            stack.attention._flash_e3_attention_post = post


class FusedCase(c.CreditHarness, unittest.TestCase):
    legacy_schedule_source = False
    query_rows = 64
    stack_type = sp.HistoryStack
    selection_counts = (0, 0)

    @classmethod
    def setUpClass(cls):
        if not c.REFERENCE.is_dir():
            raise RuntimeError('required reference absent; connected proof NOT_RUN')
        super().setUpClass()

    def assert_closed(self, network):
        driver = network.driver
        self.assertIsNone(driver.error)
        self.assertFalse(driver.active)
        self.assertFalse(driver._inflight)
        self.assertEqual(network.transport._calls, {})
        self.assertTrue(driver._last_native_closure['complete'])
        self.assertTrue(network.controller.leases.shutdown()['clean'])
        for stack in network.qsa.values():
            self.assertIsNone(stack.bridge.transaction)
            self.assertIsNone(stack.dispatcher.transaction)
            # A hold is attached when its layer starts; that one must be released.
            if stack.operation is not None and stack.operation.stats['producers']:
                self.assertTrue(all(hold.released for hold in stack.holds))


class OneBoundedCommand(FusedCase):
    def test_multi_bucket_layer_is_one_command_with_every_device_event_unresolved(self):
        network = FusedNetwork(self)
        network.run_until(lambda: network.layer_accepted(0) and network.layer_accepted(1), ready=())
        for stack in network.qsa.values():
            stats = stack.operation.stats
            self.assertEqual((stats['buckets'], stats['copy_events'], stats['consumer_events']),
                             (BUCKETS, BUCKETS, BUCKETS))
            self.assertEqual((stack.device.host_reads, stack.device.synchronized), ([], 0))
            # The whole layer was submitted with nothing resolved on the host.
            self.assertFalse(any(event.ready for event in stack.device.events))
            device = stack.device
            copies = [e for e in device.events if e.token == device.copy]
            computes = [e for e in device.events if e.token == device.compute]
            consumers = [run.event for run in stack.operation._layers[c.LAYER].consumed]
            self.assertEqual(len(copies), BUCKETS)
            self.assertTrue(set(map(id, consumers)) <= set(map(id, computes)))
            for index in range(BUCKETS):
                self.assertTrue(any(t == device.compute and e is copies[index]
                                    for t, e in device.waits), index)
            # Bucket n+1's copy is ordered behind bucket n's consumer on the device.
            for index in range(BUCKETS - 1):
                self.assertTrue(any(t == device.copy and e is consumers[index]
                                    for t, e in device.waits), index)
        self.assertEqual(len(network.sent(w.SegmentKind.LAYER, 0)), 1)
        self.assertEqual(network.profile_polls(), [])
        network.finish()
        self.assertEqual(len(network.sent(w.SegmentKind.LAYER, 0)), 1)
        self.assertIsNone(network.driver.error)
        for stack in network.qsa.values():
            self.assertTrue(stack.operation._released)
            self.assertEqual(stack.operation.stats['sparse_source_released'], 1)
            self.assertTrue(all(hold.released for hold in stack.holds))
        snapshot = network.fused.snapshot()
        self.assertEqual((snapshot['chunks'], snapshot['grants'], snapshot['fused_commands']), (1, 1, 1))

    def test_same_device_work_as_the_stepped_path_with_far_fewer_commands(self):
        results = {}
        for fused in (False, True):
            network = FusedNetwork(self, fused=fused)
            network.finish()
            self.assertIsNone(network.driver.error)
            self.assertEqual(network.driver.open_request.qsa_fused if network.driver.tx else fused, fused)
            results[fused] = dict(
                layer=len(network.sent(w.SegmentKind.LAYER, 0)),
                polls=len(network.profile_polls()),
                stats=[{k: stack.operation.stats[k]
                        for k in ('producers', 'buckets', 'pieces', 'bytes',
                                  'copy_events', 'consumer_events')}
                       for stack in network.qsa.values()])
        self.assertEqual(results[True]['stats'], results[False]['stats'])
        self.assertEqual((results[True]['layer'], results[True]['polls']), (1, 0))
        self.assertGreaterEqual(results[False]['layer'], BUCKETS + 2)
        self.assertGreaterEqual(results[False]['polls'], 1)

    def test_default_off_replies_carry_nothing_new(self):
        network = FusedNetwork(self, fused=False)
        network.finish()
        for stack in network.qsa.values():
            self.assertFalse(stack.dispatcher.qsa_fused)
        self.assertEqual(network.fused.snapshot()['chunks'], 0)
        for command in network.sent(w.SegmentKind.LAYER):
            self.assertFalse(command.profile is not None and command.profile.fused)

    def test_manual_cycles_never_run_fused(self):
        # row1, lifecycle and hygiene are manual: auto-arm stands aside.
        network = FusedNetwork(self, owned=False)
        network.finish()
        self.assertEqual(network.fused.snapshot()['chunks'], 0)
        self.assertGreater(len(network.sent(w.SegmentKind.LAYER, 0)), 1)

    def test_duplicate_delivery_replays_without_new_device_work(self):
        network = FusedNetwork(self)
        network.run_until(lambda: network.layer_accepted(0) and network.layer_accepted(1), ready=())
        command = network.sent(w.SegmentKind.LAYER, 0)[-1]
        before = [(dict(s.operation.stats), len(s.device.events)) for s in network.qsa.values()]
        first = {r: network.worker(r).flash_s5_submit(command) for r in (0, 1)}
        second = {r: network.worker(r).flash_s5_submit(command) for r in (0, 1)}
        for rank in (0, 1):
            self.assertTrue(first[rank].accepted)
            self.assertEqual(first[rank].collectives, second[rank].collectives)
        self.assertEqual(before, [(dict(s.operation.stats), len(s.device.events))
                                  for s in network.qsa.values()])
        network.finish()
        self.assertIsNone(network.driver.error)


class PeerSkew(FusedCase):
    def test_one_peers_events_held_in_either_direction(self):
        for held in (0, 1):
            with self.subTest(held=held):
                other = 1 - held
                network = FusedNetwork(self)
                network.run_until(lambda: network.layer_accepted(0) and network.layer_accepted(1),
                                  ready=(other,))
                for _ in range(40):
                    network.tick((other,))
                self.assertIsNotNone(network.driver.tx)
                self.assertIsNone(network.driver.error)
                self.assertFalse(network.qsa[held].operation._released)
                self.assertFalse(any(hold.released for hold in network.qsa[held].holds))
                self.assertEqual(network.qsa[0].operation.stats['copy_events'],
                                 network.qsa[1].operation.stats['copy_events'])
                network.finish()
                self.assertIsNone(network.driver.error)

    def test_run_ahead_bound_polls_without_a_host_wait(self):
        network = FusedNetwork(self)
        network.fused.run_ahead = 1
        for _ in range(60):
            network.tick(ready=())
        snapshot = network.fused.snapshot()
        self.assertGreater(snapshot['run_ahead_polls'], 0)
        self.assertLess(network.driver.index, len(network.driver.plan))
        self.assertFalse(any(event.ready for event in network.qsa[0].device.events))
        self.assertIsNone(network.driver.error)
        network.finish()
        self.assertIsNone(network.driver.error)


class Cancel(FusedCase):
    def test_cancel_at_every_internal_boundary_with_one_peer_unresolved(self):
        reached = set()
        for boundary in range(14):
            for held in (0, 1):
                with self.subTest(boundary=boundary, held=held):
                    other = 1 - held
                    network = FusedNetwork(self)
                    network.run_until(lambda: network.driver.tx is not None
                                      and network.driver.index >= boundary, ready=(other,))
                    reached.add(network.driver.index)
                    network.driver.stop('abort')
                    # The held peer's device work outlives the cancel: no release yet.
                    for _ in range(20):
                        r4.Stack.complete_everything(network)
                        network.qsa[other].complete_all()
                        network.driver.pump_stop(1)
                    if network.qsa[held].device.events:
                        self.assertFalse(any(h.released for h in network.qsa[held].holds))
                    network.pump_stop()
                    self.assert_closed(network)
        self.assertGreaterEqual(len(reached), 10)

    def test_stop_with_the_layer_reply_in_flight_waits_for_it(self):
        network = FusedNetwork(self)
        network.run_until(lambda: network.driver.index == 2)  # PREFIX accepted
        network.transport.hold = True
        for _ in range(4):
            network.tick()
        self.assertTrue(network.layer_accepted(0) or network.driver._inflight)
        network.driver.stop('abort')
        network.driver.pump_stop(1)
        self.assertEqual([a[0] for a in network.sender.envelopes].count('flash_s5_cancel'), 0)
        network.transport.deliver_held()
        network.pump_stop()
        self.assert_closed(network)

    def test_cancel_between_the_peers_acceptances_replays_the_exact_command(self):
        class PendingPeer:
            """Leaf: a typed pending endpoint in front of the real peer bridge."""
            def __init__(self, worker):
                self.worker = worker
                self.rank = worker.rank
                self.defer = True

            def __getattr__(self, name):
                return getattr(self.worker, name)

            def flash_s5_submit(self, command):
                if (self.defer and command.kind is w.SegmentKind.LAYER
                        and command.profile is not None):
                    return w.SegmentAcceptance(command.transaction, self.worker.rank, command.kind,
                                               command.stage, command.op_seq, pending=True)
                return self.worker.flash_s5_submit(command)

        network = FusedNetwork(self)
        network.install_tp()
        endpoint = PendingPeer(network.worker(1))
        network.transport.network.workers[1] = endpoint
        network.run_until(lambda: network.driver._tail_submission is not None
                          and network.driver._tail_submission.accepted == {0})
        obligation = network.driver._tail_submission
        self.assertEqual((network.tp[0].collectives, network.tp[1].collectives), (1, 0))
        self.assertNotIn(obligation.command.op_seq, network.qsa[1].dispatcher.accepted)
        endpoint.defer = False
        network.driver.stop('abort')
        network.pump_stop()
        # The peer entered the same collective before anything was cancelled.
        self.assertEqual((network.tp[0].collectives, network.tp[1].collectives), (1, 1))
        self.assertEqual(obligation.accepted, {0, 1})
        sent = network.sent(w.SegmentKind.LAYER, 0)
        self.assertGreaterEqual(len(sent), 2)
        self.assertTrue(all(item == obligation.command for item in sent))
        self.assert_closed(network)

    def test_failure_in_one_peer_after_partial_submission_retains_ownership(self):
        network = FusedNetwork(self)
        device = network.qsa[1].device
        original = device.copy_pages
        calls = [0]
        def fail_second_bucket(*args):
            calls[0] += 1
            if calls[0] > 1:
                raise RuntimeError('leaf: device copy failure in a later bucket')
            return original(*args)
        device.copy_pages = fail_second_bucket
        # Leaf transport: a worker failure reaches the owner as the raised reply.
        with self.assertRaises(Exception) as failed:
            network.run_until(lambda: network.driver.index > 3, ready=())
        self.assertIn('quarantined', str(failed.exception))
        # No stepped fallback and no release of what the peers already submitted.
        self.assertEqual(len(network.sent(w.SegmentKind.LAYER, 0)), 1)
        self.assertEqual(network.profile_polls(), [])
        self.assertIsNotNone(network.qsa[1].dispatcher.quarantined)
        for stack in network.qsa.values():
            self.assertFalse(any(hold.released for hold in stack.holds))
            self.assertFalse(stack.operation._released)
        self.assertEqual([a[0] for a in network.sender.envelopes].count('flash_s5_release'), 0)


class Commitment(FusedCase):
    def fused_reply(self, network, mutate):
        endpoint = network.worker(1)
        original = endpoint.flash_s5_submit
        def submit(command):
            return mutate(command, original(command))
        endpoint.flash_s5_submit = submit

    def test_real_ledger_agrees_on_order_communicator_and_shape(self):
        network = FusedNetwork(self)
        network.install_tp()
        network.finish()
        self.assertIsNone(network.driver.error)
        self.assertEqual((network.tp[0].collectives, network.tp[1].collectives), (1, 1))
        for stack in network.qsa.values():
            self.assertIsNone(stack.native.comms.tp.ledger)
        self.assertGreater(network.fused.snapshot()['ledger_checks'], 10)

    def test_equal_counts_with_a_different_shape_fail_closed_without_release(self):
        network = FusedNetwork(self)
        network.install_tp()
        stack = network.qsa[1]
        comm = network.tp[1]
        other = c.sparse_test.T((stack.tokens + 1, 1), device='cuda:0', dtype='torch.bfloat16')
        stack.attention._flash_e3_attention_post = (
            lambda state, inner=stack.attention._flash_e3_attention_post: (
                comm.all_reduce(other), inner(state))[1])
        with self.assertRaises(ed.DriverError) as refused:
            network.run_until(lambda: network.driver.index > 3)
        self.assertIn('tp-ledger-mismatch', str(refused.exception))
        self.assertEqual([a[0] for a in network.sender.envelopes].count('flash_s5_release'), 0)

    def test_equal_counts_in_a_different_order_fail_closed(self):
        ledger = [lane_comm.CollectiveLedger() for _ in range(2)]
        leaf = LeafPynccl(0)
        a, b = (c.sparse_test.T((4,), device='cuda:0', dtype='torch.int32'),
                c.sparse_test.T((8,), device='cuda:0', dtype='torch.int32'))
        ledger[0].note('all_reduce', leaf, a)
        ledger[0].note('all_reduce', leaf, b)
        ledger[1].note('all_reduce', leaf, b)
        ledger[1].note('all_reduce', leaf, a)
        self.assertEqual(ledger[0].count, ledger[1].count)
        self.assertNotEqual(ledger[0].take(), ledger[1].take())

    def test_peer_plan_disagreement_refuses_before_any_layer(self):
        network = FusedNetwork(self)
        def mutate(command, reply):
            if command.kind is w.SegmentKind.PREPARE and reply.dense_profiles:
                layer, shape = reply.dense_profiles[0]
                return replace(reply, dense_profiles=((layer, w.QsaScheduleShape(
                    shape.route, shape.bucket_submissions + (1,), shape.bucket_waves + (1,))),))
            return reply
        self.fused_reply(network, mutate)
        with self.assertRaises(ed.DriverError) as refused:
            network.run_until(lambda: network.driver.index > 3)
        self.assertIn('fused-profile-disagreement', str(refused.exception))
        self.assertEqual(network.sent(w.SegmentKind.LAYER), [])

    def test_plan_differing_from_the_admission_refuses_before_any_layer(self):
        network = FusedNetwork(self)
        for rank in (0, 1):
            driver_factory = network.qsa[rank].dispatcher.driver_factory
            def factory(operation, module, context, inner=driver_factory):
                driver = inner(operation, module, context)
                real = driver.fused_profiles
                def forged(lane, tokens):
                    return tuple((layer, w.QsaScheduleShape(shape.route, (1,), (1,)))
                                 for layer, shape in real(lane, tokens))
                driver.fused_profiles = forged
                return driver
            network.qsa[rank].dispatcher.driver_factory = factory
        with self.assertRaises(ed.DriverError) as refused:
            network.run_until(lambda: network.driver.index > 3)
        self.assertIn('fused-profile-differs-from-the-admission', str(refused.exception))
        self.assertEqual(network.sent(w.SegmentKind.LAYER), [])

    def test_watchdog_fails_closed_and_never_releases(self):
        network = FusedNetwork(self)
        now = [100.]
        network.driver.clock = lambda: now[0]
        network.run_until(lambda: network.layer_accepted(0) and network.layer_accepted(1), ready=())
        for _ in range(30):
            network.tick(ready=())  # nothing completes; the mark settles
        now[0] += network.fused.stall_s + 1
        with self.assertRaises(ed.DriverError) as refused:
            for _ in range(30):
                network.tick(ready=())
        self.assertIn('tp-collective-stall', str(refused.exception))
        self.assertEqual([a[0] for a in network.sender.envelopes].count('flash_s5_release'), 0)
        for stack in network.qsa.values():
            self.assertFalse(stack.operation._released)
            self.assertFalse(any(hold.released for hold in stack.holds))
            self.assertFalse(any(event.ready for event in stack.device.events))


class Refusal(FusedCase):
    def refused(self, configure, text):
        network = FusedNetwork(self)
        configure(network)
        # Leaf transport: a worker failure reaches the owner as the raised reply.
        with self.assertRaises(Exception) as refused:
            network.run_until(lambda: network.driver.index > 3)
        self.assertIn(text, str(refused.exception.__cause__))
        self.assertEqual(network.sent(w.SegmentKind.LAYER), [])
        for stack in network.qsa.values():
            self.assertEqual(stack.operation.stats['copy_events'], 0)

    def test_policy_row1_and_diagnostics_refuse_at_prepare(self):
        def current(network):
            for stack in network.qsa.values():
                stack.device.native_qsa_policy = 'current'
        def row1(network):
            for stack in network.qsa.values():
                begin = stack.server.begin_prefill_operation
                def wrapped(runner, marker, begin=begin, stack=stack):
                    result = begin(runner, marker)
                    stack.operation.qualification = object()
                    return result
                stack.server.begin_prefill_operation = wrapped
        def boundary(network):
            for stack in network.qsa.values():
                stack.device.native_qsa_boundary = True
        for configure, reason in ((current, 'native-qsa-policy-not-dense'),
                                  (row1, 'row1-qualifier-attached'),
                                  (boundary, 'diagnostic-readback-attached')):
            with self.subTest(configure.__name__):
                self.refused(configure, reason)


class Selection(FusedCase):
    def test_grant_for_a_stepped_transaction_is_refused(self):
        network = FusedNetwork(self, fused=False)
        network.run_until(lambda: network.driver.index >= 2)
        worker = network.worker(0)
        query = w.QsaProfileQuery(network.driver.tx, network.driver.admission.state_epoch, 0, 0, 0, (0, 1))
        grant = w.QsaProfileGrant(query, w.QsaScheduleShape('dense_dma', (1,) * 3, (1,) * 3), fused=True)
        descriptor = network.sent(w.SegmentKind.PREFIX)[-1].descriptor
        layer = w.SegmentDescriptor(descriptor.transaction, w.SegmentKind.LAYER, 0,
                                    layer_start=0, layer_end=1)
        command = w.SegmentCommand(layer, 99, (0, 1), profile=grant)
        with self.assertRaises(Exception) as refused:
            network.qsa[0].bridge.submit(command)
        self.assertIn('fused grant differs from the reported PREPARE plan', str(refused.exception))

    def test_fused_flag_needs_dense_route(self):
        network = FusedNetwork(self)
        network.run_until(lambda: network.driver.tx is not None)
        query = w.QsaProfileQuery(network.driver.tx, network.driver.admission.state_epoch, 0, 0, 0, (0, 1))
        w.QsaProfileGrant(query, w.QsaScheduleShape('dense_dma', (1,), (1,)), fused=True)
        with self.assertRaises(Exception) as refused:
            w.QsaProfileGrant(query, w.QsaScheduleShape('sparse_dma', (1,), (1,)), fused=True)
        self.assertIn('dense', str(refused.exception))


class Control(unittest.TestCase):
    """The per-arm switch and its latch over the real AutoArmPolicy and driver."""
    def setUp(self):
        import native_auto_arm as auto
        import test_e3_native_overlap as native
        self.auto = auto
        self.stack = native.cold_stack()
        self.core, self.frontend = native.core_for(self.stack, paused=True)
        self.d = self.stack.driver
        self.policy = auto.AutoArmPolicy(self.core, self.d)
        self.d.auto_arm = self.policy
        self.fused = self.d.install_qsa_fused()

    def test_selected_only_paused_idle_full_and_taken_under_the_manual_lease(self):
        reply = self.d.submit_control({'action': 'native_qsa_fused', 'fused': True})
        self.assertEqual((reply['fused'], reply['previous'], reply['route']), (True, False, 'full'))
        self.assertTrue(self.fused.selected)
        self.assertTrue(self.policy.manual_owned)
        self.assertEqual(self.fused.counters['selections'], 1)
        status = self.d.submit_control({'action': 'status'})
        self.assertEqual(status['native_qsa_fused']['selected'], True)
        released = self.d.submit_control({'action': self.auto.RELEASE, 'engine_epoch': self.d.engine_epoch})
        self.assertTrue(released['released'])
        self.assertFalse(self.policy.manual_owned)

    def test_refused_while_running_or_with_bad_fields(self):
        for command, text in (({'action': 'native_qsa_fused', 'fused': 1}, 'native-qsa-fused-invalid'),
                              ({'action': 'native_qsa_fused'}, 'native-qsa-fused-invalid'),
                              ({'action': 'native_qsa_fused', 'fused': True, 'extra': 1},
                               'native-qsa-fused-command-fields-differ'),
                              ({'action': 'native_qsa_fused', 'fused': True, 'engine_epoch': -1},
                               'native-control-engine-epoch-differs')):
            with self.subTest(command=command), self.assertRaisesRegex(Exception, text):
                self.d.submit_control(command)
        self.core.paused = False
        with self.assertRaisesRegex(Exception, 'native-qsa-fused-requires-paused-idle-full'):
            self.d.submit_control({'action': 'native_qsa_fused', 'fused': True})
        self.assertFalse(self.fused.selected)

    def test_refused_during_an_owned_cycle(self):
        self.policy.phase = 'native'
        with self.assertRaisesRegex(Exception, 'cycle-owned'):
            self.d.submit_control({'action': 'native_qsa_fused', 'fused': True})
        self.assertFalse(self.fused.selected)

    def test_not_a_control_unless_installed(self):
        self.d.qsa_fused = None
        with self.assertRaisesRegex(Exception, 'native-control-unknown-action'):
            self.d.submit_control({'action': 'native_qsa_fused', 'fused': True})

    def test_needs_the_auto_arm_policy_and_installs_once(self):
        with self.assertRaisesRegex(ed.DriverError, 'already installed'):
            self.d.install_qsa_fused()
        self.d.qsa_fused = None
        self.d.auto_arm = None
        with self.assertRaisesRegex(ed.DriverError, 'auto-arm'):
            self.d.install_qsa_fused()


class Mutants(FusedCase):
    """The scenarios above discriminate: each mutant makes its scenario fail."""
    def fails(self, target, name, replacement, scenario):
        with mock.patch.object(target, name, replacement):
            result = unittest.TestResult()
            scenario.run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertTrue(result.failures or result.errors, 'mutant survived: ' + name)

    def test_no_ledger_comparison(self):
        self.fails(ed.NativeSegmentDriver, '_fused_agree', lambda *a, **k: None,
                   Commitment('test_equal_counts_with_a_different_shape_fail_closed_without_release'))

    def test_no_tp_obligation(self):
        real = ed.TPTailSubmissionObligation
        def mutant(self, entry, real=ed.NativeSegmentDriver._send_plan_entry):
            result = real(self, entry)
            if self._tx_fused and self._tail_boundary is None:
                self._tail_submission = None
            return result
        self.fails(ed.NativeSegmentDriver, '_send_plan_entry', mutant,
                   Cancel('test_cancel_between_the_peers_acceptances_replays_the_exact_command'))

    def test_no_watchdog(self):
        self.fails(ed.NativeSegmentDriver, '_fused_watchdog', lambda self: None,
                   Commitment('test_watchdog_fails_closed_and_never_releases'))

    def test_no_run_ahead_bound(self):
        self.fails(ed.NativeSegmentDriver, '_fused_hold', lambda self, entry: False,
                   PeerSkew('test_run_ahead_bound_polls_without_a_host_wait'))

    def test_host_waits_between_buckets(self):
        real = qo.QsaPrefillOperation.advance_fused
        def mutant(self):
            if self.pending_buckets:
                raise et.E3Pending('mutant: host-observed consumer before the next bucket')
            return real(self)
        self.fails(qo.QsaPrefillOperation, 'advance_fused', mutant,
                   OneBoundedCommand('test_multi_bucket_layer_is_one_command_with_every_device_event_unresolved'))

    def test_selection_ignores_ownership(self):
        self.fails(ed.NativeSegmentDriver, '_progress_owned', lambda self: True,
                   OneBoundedCommand('test_manual_cycles_never_run_fused'))


class ConsecutiveLayers(base._GeneratedHarness, unittest.TestCase):
    """Two dense layers sharing one arena, operation level, real generated attention."""
    for _name in ('make_operation', 'hold_for', 'bind', 'prepare', 'open_layer'):
        locals()[_name] = getattr(base.GeneratedSparseTests, _name)

    def fused_layer(self, operation, source, name, mapping):
        logical = base.T((8, 5), device='cuda:0', dtype='torch.int32',
                         values=[0, 1, 4048, 4049, 4] * 8)
        query = base.T((8, 6, 256), device='cuda:0')
        output = base.T((8, 6, 256), device='cuda:0')
        rows = base.T((8,), device='cuda:0', dtype='torch.int64', values=list(range(8)))
        generator = self.attention(query, source, logical, self.table, rows, len(mapping) - 1,
                                   self.arena, True, output, e3op=operation)
        operation.attach_continuation(generator)
        shape = operation.fused_shape(name, len(mapping) * c.PAGE_TOKENS, c.PAGE_TOKENS)
        query = w.QsaProfileQuery(operation.transaction, 1, 0, 0, 0, (operation.lane.worker_rank,))
        buckets = 0
        while True:
            try:
                value = operation.advance_fused()
            except StopIteration:
                break
            if getattr(value, 'qsa_schedule_boundary', False):
                operation.authorize_schedule(w.QsaProfileGrant(query, shape, fused=True))
            else:
                buckets += 1
        operation.end_layer(name)
        return buckets

    def test_second_layer_first_copy_waits_on_the_first_layers_last_consumer(self):
        self.device.native_qsa_policy = 'dense'
        self.device.auto_ready = False
        mapping = tuple(range(1, 51))
        operation, lane, tx, sources = self.prepare({'qsa.0': mapping, 'qsa.1': mapping})
        self.open_layer(operation, lane, tx, 'qsa.0', sources['qsa.0'], mapping, min_pages=24)
        first = self.fused_layer(operation, sources['qsa.0'], 'qsa.0', mapping)
        consumers = [e for e in self.device.events if e.token == self.device.compute]
        waits = len(self.device.waits)
        self.open_layer(operation, lane, tx, 'qsa.1', sources['qsa.1'], mapping, min_pages=24)
        second = self.fused_layer(operation, sources['qsa.1'], 'qsa.1', mapping)
        self.assertEqual((first, second), (BUCKETS, BUCKETS))
        later = self.device.waits[waits:]
        self.assertTrue(any(t == self.device.copy and e is operation._layers['qsa.0'].consumed[-1].event
                            for t, e in later))
        self.assertFalse(any(e.ready for e in self.device.events))
        self.assertEqual(self.device.synchronized, 0)


if __name__ == '__main__':
    unittest.main()
