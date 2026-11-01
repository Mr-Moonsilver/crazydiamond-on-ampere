"""Step B connected host proof: consecutive LAYER entries of a stage as one bounded command.

Real NativeSegmentDriver, bridge, dispatcher, LaneForward, QsaLayerDriver, QsaPrefillOperation and generated
attention on all four ranks, over the declared CPU leaves of harness/step_b_fixture_leaves.json. Unlike Step A's
FusedNetwork, this network has several QSA layers per stage and a real QSA stage 1. Not a hardware certificate:
real-GPU event order and bytes are the VM's check, TP and PP on real NCCL are the box's.
"""
import asyncio
import contextlib
from dataclasses import replace
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'harness'), str(ROOT / 'patches/s5_engine')]
import test_e3_native_qsa_fused as f
c = f.c
r4 = f.r4
sp = f.sp
ed = f.ed
w = f.w
lane_comm = f.lane_comm
qsa_driver = f.qsa_driver
s5_startup = f.s5_startup
import lane_dispatch as ld
import native_measurement as nm
import native_overlap_control as control
import probe_control
import test_native41_progress_queue as hf
import native_overlap_session as session
import sparse_history_dma as shd
import worker_bridge as wb

# The real model's QSA positions on a shorter stage: every fourth layer, last in its four.
REAL = dict(layers=(8, 8), qsa=((3, 7), (11, 15)))
ALL = (0, 1, 2, 3)
STAGE = {0: (0, 1), 1: (2, 3)}
SIZES = (1, 2, 4)


def names_of(layers):
    return {layer: 'qsa.%d' % index for index, layer in enumerate(layers)}


def equip_stack(harness, stack, low, high, qsa, stage):
    """One real model, operation and driver owning several QSA layers on this rank."""
    names = names_of(qsa)
    # The plan provider needs a table even on a stage that never binds one.
    table = tuple((name, stack.mapping) for name in names.values()) or ((c.LAYER, stack.mapping),)
    model = stack.model
    model.layers = [None] * high
    for layer in range(low, high):
        model.layers[layer] = c.FakeGdnLayer()
    model.start_layer, model.end_layer = low, high
    attentions = {}
    for layer, name in names.items():
        attention = c.FakeAttention(stack, stack.kv_cache)
        attention.layer_name = name
        attention._qsa_host_kv = stack.source
        attention._qsa_stage_min_rows = 64
        attention.topk_indices_buffer = stack.attention.topk_indices_buffer
        attention.indexer = stack.attention.indexer
        attentions[layer] = attention
        model.layers[layer] = c.FakeQsaLayer(attention, stack.tokens)
        stack.attn_metadata[name] = stack.main
    stack.attentions = attentions
    # A receiver is the last stage; the sender's suffix produces the PP send tensor.
    model._flash_e3_is_last_stage = lambda: bool(stage)

    def make_operation(transaction):
        provider = c.qo.metadata_plan_provider(
            c.sparse_test.Snapshot([c.sparse_test.Row()], tuple(range(260))), layer_tables=table)
        operation = c.qo.QsaPrefillOperation(transaction, stack.lane, device=stack.device,
                                             plan_provider=provider)
        operation.state_epoch = 1
        return operation
    stack.make_operation = make_operation

    def begin(runner, marker):
        operation = stack.server.operation
        for name in names.values():
            operation.register_source(name, stack.source, versions=stack.versions)
        operation.begin_forward(layers=tuple(names.values()))
    stack.server.begin_prefill_operation = begin

    def factory(operation, module, context):
        stack.operation = operation
        holds = {}
        for name in names.values():
            holds[name] = shd.SparseSourceHold(
                c.sparse_test.make_owner(stack.lane, name, stack.mapping, stack.main._qsa_host_meta[1][0],
                                         transaction=operation.transaction),
                allocator=c.sparse_test.FakeAllocator())
            stack.holds.append(holds[name])
        stack.driver = qsa_driver.QsaLayerDriver(
            operation, staged_operation=harness.helper_dense, forward_context=stack._forward_scope,
            torch_module=stack.torch, clock=lambda: 0, sparse_staged_operation=harness.helper_sparse,
            source_holds=holds)
        return stack.driver
    stack.dispatcher.driver_factory = factory


class GroupNetwork(f.FusedNetwork):
    """Real on all four ranks, QSA positions given per stage; ``size`` None leaves groups uninstalled."""

    def __init__(self, harness, size=2, layers=REAL['layers'], qsa=REAL['qsa'], fused=True, owned=True):
        super().__init__(harness, fused=fused, owned=owned)
        for rank in (2, 3):
            self._real_receiver(harness, rank)
        low = 0
        spans = {}
        for stage, count in enumerate(layers):
            spans[stage] = (low, low + count)
            low += count
        self.spans = spans
        for rank, stack in self.qsa.items():
            stage = 0 if rank < 2 else 1
            equip_stack(harness, stack, *spans[stage], qsa[stage], stage)
            end = stack.transaction.chunk_end
            stack.main._qsa_host_meta = ((0, stack.tokens), (end,))
            stack.main.seq_lens.values = [end]
            stack.device.native_qsa_policy = 'dense'
        driver = self.driver
        driver.stage_layers = spans
        driver.layer_count, driver.layer_split = low, layers[0]
        driver.qsa_control = nm.QsaControlCounters(driver.engine_epoch, spans)
        self.size = size
        self.groups = None
        if size is not None:
            self.groups = driver.install_layer_groups()
            self.groups.size = size

    def _real_receiver(self, harness, rank):
        """ProfileNetwork's real stack, on the receiving stage: Step A's connected test had none."""
        worker = self.workers[rank]
        stack = sp.HistoryStack(harness, rank=rank, route='sparse', transaction=self.record.transaction,
                                ranks=STAGE[1])
        stack.native.split = worker.split
        stack.split = worker.split
        stack.native.controller = worker.controller
        stack.native.router = worker.router
        stack.native.comms.pp = worker.channel
        stack.server = sp.ProfileServer(stack, worker.state)
        stack.dispatcher = c.LaneDispatcher('runner-%d' % rank, stack.native, stack.factory, server=stack.server,
                                            driver_factory=stack._driver_factory, torch_module=stack.torch,
                                            clock=lambda: 1)
        stack.bridge = c.worker_bridge.NativeWorkerBridge(stack.dispatcher, server=stack.server,
                                                          source_binder=stack._source_binder)
        worker.dispatcher = stack.dispatcher
        worker._flash_s5_bridge = stack.bridge
        worker.server = stack.server
        worker.factory = stack.factory
        stack.bridge.attach(worker)
        self.qsa[rank] = stack

    # -- driving ------------------------------------------------------------------
    def tick(self, ready=ALL):
        r4.Stack.complete_everything(self)
        for rank in ready:
            self.qsa[rank].complete_all()
        self.driver.advance(deliver_outputs=False)

    def run_until(self, done, *, ready=ALL, limit=500):
        for _ in range(limit):
            self.tick(ready)
            if done():
                return
        raise AssertionError('connected group chunk did not reach the expected state; phase=%s index=%s '
                             'error=%s' % (self.driver.phase, self.driver.index, self.driver.error))

    def finish(self, ready=ALL):
        started = [self.driver.tx is not None]
        def done():
            started[0] = started[0] or self.driver.tx is not None
            return started[0] and self.driver.tx is None
        self.run_until(done, ready=ready)

    def pump_stop(self, ready=ALL, limit=500):
        for _ in range(limit):
            r4.Stack.complete_everything(self)
            for rank in ready:
                self.qsa[rank].complete_all()
            self.driver.pump_stop(1)
            if not self.driver.active and not self.driver._inflight:
                return
        raise AssertionError('stop did not close; phase=%s error=%s' % (self.driver.phase, self.driver.error))

    def to_layers(self):
        """Run everything up to and including the send of this chunk's first LAYER command."""
        def done():
            driver = self.driver
            return (driver.tx is not None and driver.phase == 'segments' and driver.index < len(driver.plan)
                    and driver.plan[driver.index]['kind'] is w.SegmentKind.LAYER)
        self.run_until(done)

    def parked_before_layers(self):
        """Every stage 0 peer has PREFIX accepted and no LAYER command has gone out."""
        self.driver._fused_hold = lambda entry: entry['kind'] is w.SegmentKind.LAYER
        stepped = self.driver._send_plan_entry
        def held(entry):
            if entry['kind'] is w.SegmentKind.LAYER:
                raise AssertionError('parked before the layers')
            return stepped(entry)
        self.driver._send_plan_entry = held
        self.run_until(lambda: self.driver.tx is not None and self.driver.index >= 3
                       and self.qsa[0].dispatcher.lane is not None and self.qsa[0].dispatcher.lane.state == 'running'
                       and self.qsa[1].dispatcher.lane is not None and self.qsa[1].dispatcher.lane.state == 'running',
                       limit=12)

    def layer_commands(self, stage=None):
        return [command for command in self.sent(w.SegmentKind.LAYER) if stage is None or command.stage == stage]

    def ranges(self, stage):
        seen = []
        for command in self.layer_commands(stage):
            item = (command.descriptor.layer_start, command.descriptor.layer_end)
            if item not in seen:
                seen.append(item)
        return seen

    def accepted_layers(self, rank):
        return [(stored[0].layer_start, stored[0].layer_end)
                for stored in self.qsa[rank].dispatcher.accepted.values() if stored[0].kind == 'layer']

    def layer_entries_left(self):
        return sum(1 for entry in self.driver.plan[self.driver.index:] if entry['kind'] is w.SegmentKind.LAYER)

    def install_tp(self, gdn_collectives=2):
        """The real communicators and ledgers, entered at the seams the generated model enters them."""
        self.tp = {}
        self.calls = {rank: [] for rank in self.qsa}
        for rank, stack in self.qsa.items():
            leaf = f.LeafPynccl(rank % 2)
            comm = lane_comm.PrefillTpCommunicator(leaf, 'cuda:0', torch_module=f.TP_TORCH)
            stack.native.comms.tp = comm
            self.tp[rank] = comm
            group = types.SimpleNamespace(world_size=2, device_communicator=comm)
            torch = types.SimpleNamespace(int32='int32', cuda=f.TP_TORCH.cuda)
            calls = self.calls[rank]
            for attention in stack.attentions.values():
                original = attention._flash_e3_attention_post
                def post(state, comm=comm, stack=stack, original=original, group=group, torch=torch,
                         calls=calls, name=attention.layer_name):
                    selection = types.SimpleNamespace(shape=(stack.tokens, 5), dtype='int32', device='cuda:0',
                                                      is_contiguous=lambda: True)
                    lane_comm.canonical_selection(selection, group=group, torch_module=torch)
                    comm.all_reduce(stack.output)
                    calls.append(('qsa', name))
                    return original(state)
                attention._flash_e3_attention_post = post
            forward = stack.model._flash_e3_forward_layer
            def layer(index, *args, comm=comm, stack=stack, forward=forward, calls=calls):
                for _ in range(gdn_collectives):
                    comm.all_reduce(c.sparse_test.T((stack.tokens, 16), device='cuda:0', dtype='torch.bfloat16'))
                calls.append(('gdn', index))
                return forward(index, *args)
            stack.model._flash_e3_forward_layer = layer


class PendingPeer:
    """Leaf: a typed pending endpoint in front of the real peer bridge, for one chosen command."""
    def __init__(self, worker, wanted):
        self.worker = worker
        self.rank = worker.rank
        self.wanted = wanted
        self.defer = True

    def __getattr__(self, name):
        return getattr(self.worker, name)

    def flash_s5_submit(self, command):
        if self.defer and self.wanted(command):
            return w.SegmentAcceptance(command.transaction, self.worker.rank, command.kind, command.stage,
                                       command.op_seq, pending=True)
        return self.worker.flash_s5_submit(command)


class GroupCase(f.FusedCase):
    selection_counts = (0, 0, 0, 0)

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
            operation = stack.operation
            if operation is None:
                continue
            # A hold is attached when its layer starts; that one must be released.
            for name, run in operation._layers.items():
                if run.sparse_hold is not None:
                    self.assertTrue(run.sparse_hold.released, name)


class Plan(GroupCase):
    def test_groups_close_at_the_size_after_a_qsa_layer_and_at_the_stage_end(self):
        qsa_real = set(range(3, 24, 4))
        for args, expected in (
                ((0, 8, {3, 7}, 2), [(0, 2), (2, 4), (4, 6), (6, 8)]),
                ((0, 8, {3, 7}, 4), [(0, 4), (4, 8)]),
                # A QSA layer closes its group early.
                ((0, 8, {1}, 4), [(0, 2), (2, 6), (6, 8)]),
                ((0, 8, {0}, 2), [(0, 1), (1, 3), (3, 5), (5, 7), (7, 8)]),
                ((0, 3, {0, 1, 2}, 4), [(0, 1), (1, 2), (2, 3)]),
                # A leftover short group at the stage end.
                ((0, 5, set(), 2), [(0, 2), (2, 4), (4, 5)]),
                ((0, 6, set(), 4), [(0, 4), (4, 6)]),
                ((24, 27, {26}, 4), [(24, 27)]),
                # Size 1 is one command per layer.
                ((2, 5, {3}, 1), [(2, 3), (3, 4), (4, 5)])):
            with self.subTest(args=args):
                self.assertEqual(ed.layer_group_plan(*args), expected)
        # The real model, written out from the geometry.
        self.assertEqual(ed.layer_group_plan(0, 24, qsa_real, 4), [(k, k + 4) for k in range(0, 24, 4)])
        self.assertEqual(ed.layer_group_plan(0, 24, qsa_real, 2), [(k, k + 2) for k in range(0, 24, 2)])
        qsa_second = set(range(27, 48, 4))
        self.assertEqual(ed.layer_group_plan(24, 48, qsa_second, 4), [(k, k + 4) for k in range(24, 48, 4)])

    def test_each_group_holds_at_most_one_qsa_layer_last_and_covers_the_stage(self):
        for size in (2, 4):
            for qsa in (set(), {0}, {1}, {3}, {0, 1, 2}, {3, 7}, {2, 3, 7, 9}, set(range(10))):
                groups = ed.layer_group_plan(0, 10, qsa, size)
                self.assertEqual(groups[0][0], 0)
                self.assertEqual(groups[-1][1], 10)
                for (_, left), (right, _) in zip(groups, groups[1:]):
                    self.assertEqual(left, right)
                for start, end in groups:
                    self.assertLessEqual(end - start, size)
                    inside = [layer for layer in range(start, end) if layer in qsa]
                    self.assertIn(inside, ([], [end - 1]))

    def test_positions_come_from_the_prepare_plan_not_from_the_layer_number(self):
        # QSA at 1 on stage 0 and at 12 on stage 1: layer % 4 would put a QSA mid-group.
        network = GroupNetwork(self, size=4, qsa=((1,), (12,)))
        network.finish()
        self.assertIsNone(network.driver.error)
        self.assertEqual(network.ranges(0), [(0, 2), (2, 6), (6, 8)])
        self.assertEqual(network.ranges(1), [(8, 12), (12, 13), (13, 16)])
        self.assertEqual(network.groups.counters['plan_commands'], 6)
        self.assertEqual(network.groups.counters['group_commands'], 6)
        self.assertEqual(network.groups.counters['group_layers'], 16)

    def test_a_stage_without_qsa_groups_too(self):
        network = GroupNetwork(self, size=2, layers=(5, 3), qsa=((1,), ()))
        network.finish()
        self.assertIsNone(network.driver.error)
        self.assertEqual(network.ranges(0), [(0, 2), (2, 4), (4, 5)])
        self.assertEqual(network.ranges(1), [(5, 7), (7, 8)])

    def test_grant_rides_on_the_group_of_its_qsa_layer(self):
        network = GroupNetwork(self, size=2)
        network.finish()
        for command in network.layer_commands():
            descriptor = command.descriptor
            if descriptor.layer_end - 1 in (3, 7, 11, 15):
                self.assertTrue(command.profile.fused)
                self.assertEqual(command.profile.query.layer, descriptor.layer_end - 1)
            else:
                self.assertIsNone(command.profile)
            self.assertEqual(descriptor.layer_group, 2)


class OneCommand(GroupCase):
    def first_group_held(self, size):
        """The first group accepted with its own device events and every QSA event unresolved."""
        network = GroupNetwork(self, size=size)
        network.install_tp()
        network.to_layers()
        return network

    def test_one_command_per_group_with_every_device_event_unresolved(self):
        for size, expected in ((2, (2, 4)), (4, (0, 4))):
            with self.subTest(size=size):
                network = self.first_group_held(size)
                target = expected
                # The bus and device stay unresolved from here on.
                network.run_until(lambda: all(target in network.accepted_layers(rank) for rank in STAGE[0]), ready=())
                for rank in STAGE[0]:
                    stack = network.qsa[rank]
                    groups = network.accepted_layers(rank)
                    qsa_groups = sum(1 for _, end in groups if end - 1 in (3, 7))
                    self.assertEqual(stack.operation.stats['copy_events'], qsa_groups * f.BUCKETS)
                    self.assertEqual(stack.device.host_reads, [])
                    self.assertEqual(stack.device.synchronized, 0)
                    self.assertFalse(any(event.ready for event in stack.device.events))
                    # One op_seq per group, and the reply's cursor is the group end.
                    self.assertEqual(stack.dispatcher.cursor()[0], groups[-1][1])
                self.assertEqual(sum(1 for c_ in network.layer_commands(0)
                                     if (c_.descriptor.layer_start, c_.descriptor.layer_end) == expected), 1)
                self.assertEqual(network.profile_polls(), [])
                network.finish()
                self.assertIsNone(network.driver.error)

    def trace(self, size):
        network = GroupNetwork(self, size=size)
        network.install_tp()
        network.finish()
        self.assertIsNone(network.driver.error)
        rows = {}
        for rank, stack in network.qsa.items():
            device = stack.device
            rows[rank] = dict(
                layers=network.calls[rank],
                stats={k: stack.operation.stats[k] for k in ('producers', 'buckets', 'pieces', 'bytes',
                                                             'copy_events', 'consumer_events')},
                events=[event.token for event in device.events],
                waits=[(token, device.events.index(event)) for token, event in device.waits],
                collectives=network.tp[rank].collectives,
                broadcasts=[(item[1], item[2]) for item in network.tp[rank].pynccl_comm.broadcasts],
                reduces=network.tp[rank].pynccl_comm.calls,
                released=stack.operation._released)
        return rows, network

    def test_same_device_work_as_size_one(self):
        base, _ = self.trace(1)
        for rank, row in base.items():
            self.assertEqual(len(row['layers']), 8)
            self.assertEqual(row['stats']['buckets'], 2 * f.BUCKETS)
        for size in (2, 4):
            with self.subTest(size=size):
                self.assertEqual(self.trace(size)[0], base)

    def test_commands_per_chunk_shrink_by_the_plan(self):
        for size, count in ((1, 16), (2, 8), (4, 4)):
            with self.subTest(size=size):
                network = GroupNetwork(self, size=size)
                network.finish()
                self.assertEqual(len(network.layer_commands()), count)
                counters = network.groups.counters
                if size == 1:
                    self.assertEqual((counters['chunks'], counters['group_commands'], counters['group_layers'],
                                      counters['plan_commands']), (0, 0, 0, 0))
                else:
                    self.assertEqual((counters['chunks'], counters['group_commands'], counters['group_layers'],
                                      counters['plan_commands']), (1, count, 16, count))
                    self.assertEqual(counters['group_layers_max'], size)

    def test_a_qsa_stage_one_grants_and_runs_fused(self):
        network = GroupNetwork(self, size=2)
        network.finish()
        for rank in STAGE[1]:
            stats = network.qsa[rank].operation.stats
            self.assertEqual((stats['buckets'], stats['copy_events'], stats['consumer_events']),
                             (2 * f.BUCKETS, 2 * f.BUCKETS, 2 * f.BUCKETS))
            self.assertTrue(network.qsa[rank].operation._released)
        self.assertEqual(network.fused.snapshot()['grants'], 4)
        self.assertEqual(network.fused.snapshot()['fused_commands'], 4)

    def test_duplicate_delivery_replays_the_ack_with_no_device_work(self):
        for size in (2, 4):
            with self.subTest(size=size):
                network = self.first_group_held(size)
                network.run_until(lambda: all(network.accepted_layers(rank) for rank in STAGE[0]), ready=())
                command = network.layer_commands(0)[-1]
                before = [(dict(s.operation.stats), len(s.device.events), s.model.forward_layer_calls[:],
                           network.tp[r].collectives) for r, s in network.qsa.items()]
                first = {r: network.worker(r).flash_s5_submit(command) for r in STAGE[0]}
                second = {r: network.worker(r).flash_s5_submit(command) for r in STAGE[0]}
                for rank in STAGE[0]:
                    self.assertTrue(first[rank].accepted)
                    self.assertEqual(first[rank].collectives, second[rank].collectives)
                    self.assertEqual(first[rank].cursor, second[rank].cursor)
                self.assertEqual(before, [(dict(s.operation.stats), len(s.device.events), s.model.forward_layer_calls[:],
                                           network.tp[r].collectives) for r, s in network.qsa.items()])
                network.finish()
                self.assertIsNone(network.driver.error)

    def test_no_credit_query_can_pend_a_group(self):
        def refuse(self, layer):
            raise AssertionError('a group consulted submission credit')
        with mock.patch.object(qsa_driver.QsaLayerDriver, 'submission_credit', refuse):
            network = GroupNetwork(self, size=4)
            network.finish()
        self.assertIsNone(network.driver.error)


class StepAIdentity(GroupCase):
    @staticmethod
    def stream(network):
        rows = []
        for method, args, _, _ in network.sender.envelopes:
            if method != 'flash_s5_submit':
                rows.append(method)
                continue
            command = args[0]
            descriptor = command.descriptor
            profile = command.profile
            rows.append((descriptor.kind, descriptor.stage, descriptor.layer_start, descriptor.layer_end,
                         descriptor.layer_group, command.op_seq, command.ranks,
                         None if profile is None else (profile.fused, profile.shape, profile.query.layer)))
        return rows

    def test_size_one_is_exactly_step_as_command_stream(self):
        uninstalled = GroupNetwork(self, size=None)
        uninstalled.finish()
        selected = GroupNetwork(self, size=1)
        selected.finish()
        self.assertIsNone(uninstalled.groups)
        self.assertEqual(self.stream(selected), self.stream(uninstalled))
        self.assertTrue(all(command.descriptor.layer_group == 1 for command in selected.layer_commands()))
        self.assertTrue(all(command.descriptor.layer_end == command.descriptor.layer_start + 1
                            for command in selected.layer_commands()))

    def test_size_one_gives_a_lone_gdn_layer_no_obligation_as_step_a(self):
        network = GroupNetwork(self, size=1)
        seen = []
        apply = network.driver._apply_segments
        def spy(values):
            command = network.driver._command
            if command.kind is w.SegmentKind.LAYER:
                seen.append((command.descriptor.layer_start, network.driver._tail_submission is not None))
            return apply(values)
        network.driver._apply_segments = spy
        network.finish()
        rows = dict(seen)
        self.assertEqual(len(rows), 16)
        self.assertEqual({layer for layer, held in rows.items() if held}, {3, 7, 11, 15})

    def test_size_one_opens_with_no_group_and_size_two_opens_with_its_size(self):
        for size in (1, 2):
            with self.subTest(size=size):
                network = GroupNetwork(self, size=size)
                network.run_until(lambda: network.driver.tx is not None and network.driver.open_request is not None)
                self.assertEqual(network.driver.open_request.layer_group, size)
                self.assertEqual(network.driver._tx_group, size)
                network.finish()
                self.assertEqual(network.driver._tx_group, 1)


class DefaultOff(GroupCase):
    def descriptor(self, **kwargs):
        tx = self.transaction()
        return w.SegmentDescriptor(tx, w.SegmentKind.LAYER, 0, **kwargs)

    def transaction(self):
        return c.et.TransactionId(engine_epoch=1, request_uid='g', request_generation=0, chunk_start=0,
                                  chunk_end=64, txn_seq=1)

    def test_a_multi_layer_descriptor_needs_its_group_size(self):
        self.assertEqual(self.descriptor(layer_start=0, layer_end=1).layer_group, 1)
        with self.assertRaisesRegex(Exception, 'one layer per segment'):
            self.descriptor(layer_start=0, layer_end=2)
        with self.assertRaisesRegex(Exception, 'consecutive layers'):
            self.descriptor(layer_start=0, layer_end=3, layer_group=2)
        with self.assertRaisesRegex(Exception, 'consecutive layers'):
            self.descriptor(layer_start=2, layer_end=2, layer_group=2)
        with self.assertRaisesRegex(Exception, 'known group size'):
            self.descriptor(layer_start=0, layer_end=3, layer_group=3)
        with self.assertRaisesRegex(Exception, 'known group size'):
            w.SegmentDescriptor(self.transaction(), w.SegmentKind.SUFFIX, 0, layer_group=2)
        self.assertEqual(self.descriptor(layer_start=4, layer_end=6, layer_group=2).layers, (4, 5))
        self.assertEqual(self.descriptor(layer_start=4, layer_end=5).layers, (4,))

    def test_local_command_still_refuses_more_than_one_layer_without_a_group(self):
        tx = self.transaction()
        with self.assertRaisesRegex(ld.LaneDispatchError, 'exactly one layer'):
            ld.SegmentCommand(tx, 0, 'layer', 1, layer_start=0, layer_end=2)
        with self.assertRaisesRegex(ld.LaneDispatchError, 'longer than its size'):
            ld.SegmentCommand(tx, 0, 'layer', 1, layer_start=0, layer_end=3, layer_group=2)
        self.assertEqual(ld.SegmentCommand(tx, 0, 'layer', 1, layer_start=0, layer_end=2, layer_group=2).layer_end, 2)

    def forged(self, network, count, group):
        driver = network.driver
        start = network.qsa[0].dispatcher.lane.cursor
        descriptor = w.SegmentDescriptor(driver.tx, w.SegmentKind.LAYER, 0, layer_start=start,
                                         layer_end=start + count, layer_group=group)
        return w.SegmentCommand(descriptor, 99, STAGE[0])

    def test_transactions_not_selecting_groups_refuse_a_group_with_no_device_work(self):
        for label, network in (('fused size 1', GroupNetwork(self, size=1)),
                               ('stepped', GroupNetwork(self, size=None, fused=False)),
                               ('uninstalled', GroupNetwork(self, size=None))):
            with self.subTest(label=label):
                network.run_until(lambda: network.driver.index >= 3 and network.qsa[0].dispatcher.lane is not None
                                  and network.qsa[0].dispatcher.lane.state == 'running')
                before = len(network.qsa[0].model.forward_layer_calls)
                cursor = network.qsa[0].dispatcher.lane.cursor
                with self.assertRaisesRegex(Exception, 'layer group differs'):
                    network.qsa[0].bridge.submit(self.forged(network, 2, 2))
                self.assertEqual(len(network.qsa[0].model.forward_layer_calls), before)
                self.assertIsNone(network.qsa[0].dispatcher.quarantined)
                self.assertEqual(network.qsa[0].dispatcher.lane.cursor, cursor)

    def test_a_group_transaction_refuses_a_foreign_size(self):
        network = GroupNetwork(self, size=4)
        network.parked_before_layers()
        with self.assertRaisesRegex(Exception, 'layer group differs'):
            network.qsa[0].bridge.submit(self.forged(network, 2, 2))
        self.assertIsNone(network.qsa[0].dispatcher.quarantined)

    def test_open_refuses_a_group_without_the_fused_selection(self):
        network = GroupNetwork(self, size=2)
        network.run_until(lambda: network.driver.tx is not None and network.driver.open_request is not None)
        request = network.driver.open_request
        self.assertEqual(replace(request, layer_group=1).layer_group, 1)
        with self.assertRaisesRegex(Exception, 'fused selection'):
            replace(request, qsa_fused=False)
        with self.assertRaisesRegex(Exception, 'known size'):
            replace(request, layer_group=3)

    def test_dispatcher_selects_a_group_only_at_open_with_fused(self):
        network = GroupNetwork(self, size=2)
        network.run_until(lambda: network.driver.index >= 3)
        with self.assertRaisesRegex(ld.LaneDispatchError, 'selected at open'):
            network.qsa[0].dispatcher.select_layer_group(4)
        stepped = GroupNetwork(self, size=None, fused=False)
        stepped.run_until(lambda: stepped.driver.index >= 1 and stepped.qsa[0].dispatcher.transaction is not None)
        with self.assertRaisesRegex(ld.LaneDispatchError, 'selected at open'):
            stepped.qsa[0].dispatcher.select_layer_group(2)

    def test_manual_cycles_never_group(self):
        network = GroupNetwork(self, size=2, owned=False)
        network.finish()
        self.assertEqual(network.groups.counters['chunks'], 0)
        self.assertEqual(network.fused.snapshot()['chunks'], 0)
        self.assertTrue(all(command.descriptor.layer_group == 1 for command in network.layer_commands()))

    def test_fused_not_selected_never_groups_even_with_a_stale_size(self):
        network = GroupNetwork(self, size=2, fused=False)
        network.finish()
        self.assertEqual(network.groups.counters['chunks'], 0)
        self.assertTrue(all(command.descriptor.layer_group == 1 for command in network.layer_commands()))


class WorkerRefusal(GroupCase):
    """Refused before the first enqueue: nothing to quarantine, no collective reached."""
    def ready(self, size=4, **kwargs):
        network = GroupNetwork(self, size=size, **kwargs)
        network.install_tp()
        network.parked_before_layers()
        return network

    def command(self, network, start, end, group, grant_layer=None):
        driver = network.driver
        descriptor = w.SegmentDescriptor(driver.tx, w.SegmentKind.LAYER, 0, layer_start=start, layer_end=end,
                                         layer_group=group)
        profile = None if grant_layer is None else driver._profile_grants[(0, grant_layer)]
        return w.SegmentCommand(descriptor, 99, STAGE[0], profile=profile)

    def refused(self, network, command, text):
        stack = network.qsa[0]
        collectives = network.tp[0].collectives
        calls = list(network.calls[0])
        with self.assertRaisesRegex(Exception, text):
            stack.bridge.submit(command)
        self.assertIsNone(stack.dispatcher.quarantined)
        self.assertEqual(network.tp[0].collectives, collectives)
        self.assertEqual(network.calls[0], calls)

    def test_a_group_that_crosses_its_stage(self):
        network = self.ready(layers=(2, 8), qsa=((), (3,)))
        self.refused(network, self.command(network, 0, 4, 4), 'crosses its stage')

    def test_a_qsa_layer_not_last(self):
        network = self.ready(qsa=((1,), ()))
        self.refused(network, self.command(network, 0, 4, 4), 'QSA layer last')

    def test_two_qsa_layers(self):
        network = self.ready(qsa=((0, 1), ()))
        self.refused(network, self.command(network, 0, 2, 4, 1), 'QSA layer last')

    def test_a_qsa_group_without_its_grant_and_a_grant_without_a_qsa_layer(self):
        network = self.ready()
        self.refused(network, self.command(network, 0, 4, 4), 'carries the grant')
        grant = network.driver._profile_grants[(0, 3)]
        query = replace(grant.query, layer=1)
        descriptor = w.SegmentDescriptor(network.driver.tx, w.SegmentKind.LAYER, 0, layer_start=0, layer_end=2,
                                         layer_group=4)
        command = w.SegmentCommand(descriptor, 99, STAGE[0], profile=w.QsaProfileGrant(query, grant.shape, fused=True))
        with self.assertRaisesRegex(Exception, 'differs from the reported PREPARE plan'):
            network.qsa[0].bridge.submit(command)

    def test_a_wire_grant_names_the_groups_last_layer(self):
        network = self.ready()
        grant = network.driver._profile_grants[(0, 3)]
        descriptor = w.SegmentDescriptor(network.driver.tx, w.SegmentKind.LAYER, 0, layer_start=0, layer_end=2,
                                         layer_group=2)
        with self.assertRaisesRegex(Exception, 'differs from the layer command'):
            w.SegmentCommand(descriptor, 99, STAGE[0], profile=grant)

    def test_a_group_off_the_cursor(self):
        network = self.ready()
        self.refused(network, self.command(network, 4, 8, 4, 7), 'does not match cursor')


class RunAhead(GroupCase):
    def test_table_and_reason(self):
        self.assertEqual(ed.LayerGroups.RUN_AHEAD, {1: 4, 2: 3, 4: 2})
        self.assertEqual(ed.LayerGroups.RUN_AHEAD[1], ed.FusedQsa.RUN_AHEAD)

    def test_the_bound_per_size_with_a_peer_held_in_either_direction(self):
        for size, limit in ((1, 4), (2, 3), (4, 2)):
            for held in (0, 1):
                with self.subTest(size=size, held=held):
                    # SUFFIX is not held (Step A's rule), so the stage has more groups than the bound.
                    network = GroupNetwork(self, size=size, layers=(16, 8), qsa=((3, 7, 11, 15), (19, 23)))
                    network.parked_before_layers()
                    # Everything up to the first layer completed: from here the queue holds LAYER commands only.
                    for rank in ALL:
                        network.qsa[rank].complete_all()
                    del network.driver._fused_hold, network.driver._send_plan_entry
                    network.tick()
                    other = 1 - held
                    for _ in range(60):
                        network.tick((other, 2, 3))
                        stack = network.qsa[held]
                        pending = stack.dispatcher.poll()['pending_op_seqs']
                        self.assertLessEqual(len(pending), limit)
                    # The queue is full, in commands, on the held peer and nothing beyond it was accepted.
                    self.assertEqual(len(network.qsa[held].dispatcher.poll()['pending_op_seqs']), limit)
                    self.assertGreater(network.fused.counters['run_ahead_polls'], 0)
                    self.assertIsNone(network.driver.error)
                    self.assertEqual(len(network.qsa[held].dispatcher.poll()['pending_op_seqs']), limit)
                    network.finish()
                    self.assertIsNone(network.driver.error)

    def test_a_held_peer_keeps_its_owners_until_released(self):
        for held in (0, 1):
            with self.subTest(held=held):
                network = GroupNetwork(self, size=4)
                other = 1 - held
                network.to_layers()
                network.run_until(lambda: network.accepted_layers(held), ready=(other, 2, 3))
                for _ in range(30):
                    network.tick((other, 2, 3))
                self.assertIsNotNone(network.driver.tx)
                self.assertIsNone(network.driver.error)
                self.assertFalse(network.qsa[held].operation._released)
                self.assertEqual(network.qsa[0].operation.stats['copy_events'],
                                 network.qsa[1].operation.stats['copy_events'])
                network.finish()
                self.assertIsNone(network.driver.error)

    def test_only_layer_entries_are_held(self):
        network = GroupNetwork(self, size=2)
        network.run_until(lambda: network.driver.tx is not None)
        driver = network.driver
        limit = ed.LayerGroups.RUN_AHEAD[2]
        driver.reports = {0: types.SimpleNamespace(pending_op_seqs=tuple(range(limit))),
                          1: types.SimpleNamespace(pending_op_seqs=())}
        entry = dict(kind=w.SegmentKind.LAYER, stage=0, layer=0)
        self.assertTrue(driver._fused_hold(entry))
        self.assertFalse(driver._fused_hold(dict(entry, kind=w.SegmentKind.SUFFIX)))
        self.assertFalse(driver._fused_hold(dict(entry, stage=1)))
        driver.reports = {0: types.SimpleNamespace(pending_op_seqs=tuple(range(limit - 1))),
                          1: types.SimpleNamespace(pending_op_seqs=tuple(range(limit - 1)))}
        self.assertFalse(driver._fused_hold(entry))


class Cancel(GroupCase):
    def test_cancel_at_every_command_boundary_with_one_peer_unresolved(self):
        for size in (2, 4):
            probe = GroupNetwork(self, size=size)
            probe.finish()
            # PREPARE x2, PREFIX, groups, SUFFIX, PP_SEND, PP_RECEIVE, FINALIZE, PREFIX, groups, SUFFIX, FINALIZE.
            length = len(probe.layer_commands()) + 10
            for boundary in range(1, length):
                for held in STAGE[0]:
                    with self.subTest(size=size, boundary=boundary, held=held):
                        network = GroupNetwork(self, size=size)
                        network.run_until(lambda: network.driver.tx is not None
                                          and network.driver.phase == 'segments'
                                          and network.driver.index >= boundary)
                        network.driver.stop('abort')
                        others = tuple(rank for rank in ALL if rank != held)
                        for _ in range(20):
                            r4.Stack.complete_everything(network)
                            for rank in others:
                                network.qsa[rank].complete_all()
                            network.driver.pump_stop(1)
                        stack = network.qsa[held]
                        if stack.device.events and not all(event.ready for event in stack.device.events):
                            self.assertFalse(any(hold.released for hold in stack.holds))
                        network.pump_stop()
                        self.assert_closed(network)

    def test_stop_with_the_group_reply_in_flight_waits_for_it(self):
        for size in (2, 4):
            with self.subTest(size=size):
                network = GroupNetwork(self, size=size)
                network.to_layers()
                network.tick()
                network.transport.hold = True
                for _ in range(4):
                    network.tick()
                self.assertTrue(network.driver._inflight)
                self.assertEqual(tuple(network.driver._inflight.values()), ('segment',))
                network.driver.stop('abort')
                network.driver.pump_stop(1)
                self.assertEqual([a[0] for a in network.sender.envelopes].count('flash_s5_cancel'), 0)
                network.transport.deliver_held()
                network.transport.hold = False
                network.pump_stop()
                self.assert_closed(network)

    def between_acceptances(self, size, wanted, entered):
        """One peer accepted the group, the other answered pending, and then the stop arrives."""
        network = GroupNetwork(self, size=size)
        network.install_tp()
        endpoint = PendingPeer(network.worker(1), wanted)
        network.transport.network.workers[1] = endpoint
        network.run_until(lambda: network.driver._tail_submission is not None
                          and network.driver._tail_submission.accepted == {0})
        obligation = network.driver._tail_submission
        before = network.tp[1].collectives
        self.assertEqual(network.tp[0].collectives - before, entered)
        self.assertNotIn(obligation.command.op_seq, network.qsa[1].dispatcher.accepted)
        endpoint.defer = False
        network.driver.stop('abort')
        network.pump_stop()
        # The peer entered the same collectives before anything was cancelled.
        self.assertEqual(network.tp[1].collectives, network.tp[0].collectives)
        self.assertEqual(obligation.accepted, {0, 1})
        descriptor = obligation.command.descriptor
        sent = [item for item in network.layer_commands(0) if item.op_seq == obligation.command.op_seq]
        self.assertGreaterEqual(len(sent), 2)
        self.assertTrue(all(item == obligation.command for item in sent))
        self.assert_closed(network)
        return descriptor

    def test_cancel_between_acceptances_in_a_group_without_qsa(self):
        descriptor = self.between_acceptances(
            2, lambda command: command.kind is w.SegmentKind.LAYER and command.descriptor.layer_start == 0, 4)
        self.assertEqual((descriptor.layer_start, descriptor.layer_end), (0, 2))

    def test_cancel_between_acceptances_in_a_group_with_qsa(self):
        descriptor = self.between_acceptances(
            2, lambda command: command.kind is w.SegmentKind.LAYER and command.descriptor.layer_start == 2, 2 + 1)
        self.assertEqual((descriptor.layer_start, descriptor.layer_end), (2, 4))
        descriptor = self.between_acceptances(
            4, lambda command: command.kind is w.SegmentKind.LAYER and command.descriptor.layer_start == 0, 6 + 1)
        self.assertEqual((descriptor.layer_start, descriptor.layer_end), (0, 4))

    def test_a_retry_without_a_stop_finishes_the_same_command(self):
        network = GroupNetwork(self, size=2)
        network.install_tp()
        endpoint = PendingPeer(network.worker(1), lambda command: command.kind is w.SegmentKind.LAYER
                               and command.descriptor.layer_start == 0)
        network.transport.network.workers[1] = endpoint
        network.run_until(lambda: network.driver._tail_submission is not None
                          and network.driver._tail_submission.accepted == {0})
        endpoint.defer = False
        network.finish()
        self.assertIsNone(network.driver.error)
        self.assertEqual(network.tp[0].collectives, network.tp[1].collectives)

    def test_a_stop_outlasts_the_round_budget_while_groups_drain(self):
        network = GroupNetwork(self, size=4)
        network.to_layers()
        network.run_until(lambda: network.accepted_layers(0) and network.accepted_layers(1), ready=())
        network.driver.stop('abort')
        budget = network.driver.stop_budget
        for _ in range(3 * budget):
            r4.Stack.complete_everything(network)
            network.driver.pump_stop(1)
        self.assertIsNone(network.driver.error)
        self.assertGreater(network.driver._stop_ticks, budget)
        for stack in network.qsa.values():
            self.assertFalse(any(hold.released for hold in stack.holds))
        network.pump_stop()
        self.assert_closed(network)
        self.assertGreater(network.fused.snapshot()['stop_rounds_max'], budget)

    def test_a_stop_past_the_stall_bound_retains_ownership(self):
        now = [100.]
        network = GroupNetwork(self, size=4)
        network.driver.clock = lambda: now[0]
        network.to_layers()
        network.run_until(lambda: network.accepted_layers(0) and network.accepted_layers(1), ready=())
        network.driver.stop('abort')
        for _ in range(8):
            r4.Stack.complete_everything(network)
            network.driver.pump_stop(1)
        now[0] += network.fused.stall_s + 1
        with self.assertRaises(ed.DriverError) as refused:
            for _ in range(8):
                r4.Stack.complete_everything(network)
                network.driver.pump_stop(1)
        self.assertIn('within the bound', str(refused.exception))
        self.assertEqual([a[0] for a in network.sender.envelopes].count('flash_s5_release'), 0)
        for rank in STAGE[0]:
            self.assertFalse(network.qsa[rank].operation._released)


class Failure(GroupCase):
    def failing(self, network, rank, layer):
        stack = network.qsa[rank]
        forward = stack.model._flash_e3_forward_layer
        def fail(index, *args):
            if index == layer:
                raise RuntimeError('leaf: a layer failed inside a group')
            return forward(index, *args)
        stack.model._flash_e3_forward_layer = fail

    def assert_retained(self, network, failed, others=STAGE[0]):
        self.assertIsNotNone(network.qsa[failed].dispatcher.quarantined)
        self.assertEqual([a[0] for a in network.sender.envelopes].count('flash_s5_release'), 0)
        for rank in others:
            stack = network.qsa[rank]
            self.assertFalse(stack.operation._released)
            self.assertEqual(stack.device.synchronized, 0)
            self.assertEqual(stack.device.host_reads, [])
            self.assertFalse(any(hold.released for hold in stack.holds))
        # The failed peer refuses everything and cannot drain or release.
        stack = network.qsa[failed]
        self.assertFalse(stack.dispatcher.drain())
        with self.assertRaisesRegex(Exception, 'quarantined'):
            stack.dispatcher.release()

    def test_layer_two_of_four_raises_on_one_peer(self):
        for failed in STAGE[0]:
            with self.subTest(failed=failed):
                network = GroupNetwork(self, size=4)
                network.install_tp()
                self.failing(network, failed, 1)
                with self.assertRaises(Exception) as raised:
                    network.to_layers()
                    network.run_until(lambda: network.driver.index > 4, ready=())
                self.assertIn('quarantined', str(raised.exception))
                # Layer 0 reached the failing peer's collectives; no later layer ran there.
                self.assertEqual([row for row in network.calls[failed] if row[0] == 'gdn'], [('gdn', 0)])
                self.assert_retained(network, failed)

    def test_the_qsa_layer_of_a_group_raises_on_one_peer(self):
        network = GroupNetwork(self, size=4)
        device = network.qsa[1].device
        original = device.copy_pages
        calls = [0]
        def fail_second_bucket(*args):
            calls[0] += 1
            if calls[0] > 1:
                raise RuntimeError('leaf: device copy failure in a later bucket')
            return original(*args)
        device.copy_pages = fail_second_bucket
        with self.assertRaises(Exception) as raised:
            network.to_layers()
            network.run_until(lambda: network.driver.index > 4, ready=())
        self.assertIn('quarantined', str(raised.exception))
        self.assert_retained(network, 1)

    def test_a_layer_that_does_not_finish_quarantines(self):
        network = GroupNetwork(self, size=4)
        lane_layer = c.LaneForward.layer
        def paused(lane, index):
            record = lane_layer(lane, index)
            if index == 1 and lane.model is network.qsa[0].model:
                record.kind = 'layer-paused'
            return record
        with mock.patch.object(c.LaneForward, 'layer', paused), self.assertRaises(Exception) as raised:
            network.to_layers()
            network.run_until(lambda: network.driver.index > 4, ready=())
        self.assertIn('quarantined', str(raised.exception))
        self.assert_retained(network, 0)

    def test_a_peer_refusal_after_a_peer_accepted_fails_closed_on_the_owner(self):
        network = GroupNetwork(self, size=2)
        endpoint = network.worker(1)
        original = endpoint.flash_s5_submit
        def refuse(command):
            if command.kind is w.SegmentKind.LAYER:
                raise wb.WorkerBridgeError('leaf: peer refused before any enqueue')
            return original(command)
        endpoint.flash_s5_submit = refuse
        with self.assertRaisesRegex(Exception, 'peer refused'):
            network.run_until(lambda: network.driver.index > 4)
        self.assertEqual(network.accepted_layers(1), [])
        self.assertEqual([a[0] for a in network.sender.envelopes].count('flash_s5_release'), 0)


class Ledger(GroupCase):
    def test_equal_counts_with_a_different_shape_inside_a_group_fail_closed(self):
        for layer in (0, 1):
            with self.subTest(layer=layer):
                network = GroupNetwork(self, size=2)
                network.install_tp()
                stack, comm = network.qsa[1], network.tp[1]
                other = c.sparse_test.T((stack.tokens + 1, 16), device='cuda:0', dtype='torch.bfloat16')
                real = comm.all_reduce
                seen = [0]
                def reduce(tensor):
                    seen[0] += 1
                    # Layer ``layer``'s first collective; two per GDN layer.
                    return real(other if seen[0] == 2 * layer + 1 else tensor)
                comm.all_reduce = reduce
                with self.assertRaises(ed.DriverError) as refused:
                    network.run_until(lambda: network.driver.index > 4)
                self.assertIn('tp-ledger-mismatch', str(refused.exception))
                self.assertEqual([a[0] for a in network.sender.envelopes].count('flash_s5_release'), 0)

    def test_a_different_count_inside_a_group_fails_closed(self):
        network = GroupNetwork(self, size=4)
        network.install_tp()
        comm = network.tp[1]
        real = comm.all_reduce
        seen = [0]
        def reduce(tensor):
            seen[0] += 1
            if seen[0] == 3:
                return tensor  # one collective of layer 1 never reached the ledger
            return real(tensor)
        comm.all_reduce = reduce
        with self.assertRaises(ed.DriverError) as refused:
            network.run_until(lambda: network.driver.index > 4)
        self.assertIn('tp-ledger-mismatch', str(refused.exception))

    def test_equal_counts_in_a_different_order_across_a_groups_layers_fail_closed(self):
        network = GroupNetwork(self, size=2)
        network.install_tp(gdn_collectives=1)
        stack, comm = network.qsa[1], network.tp[1]
        real = comm.all_reduce
        small = c.sparse_test.T((stack.tokens, 8), device='cuda:0', dtype='torch.bfloat16')
        seen = [0]
        # The same two shapes on both peers, swapped between the group's layers.
        def peer1(tensor):
            seen[0] += 1
            return real(small if seen[0] == 1 else tensor)
        comm.all_reduce = peer1
        real0 = network.tp[0].all_reduce
        seen0 = [0]
        def peer0(tensor):
            seen0[0] += 1
            return real0(small if seen0[0] == 2 else tensor)
        network.tp[0].all_reduce = peer0
        with self.assertRaises(ed.DriverError) as refused:
            network.run_until(lambda: network.driver.index > 4)
        self.assertIn('tp-ledger-mismatch', str(refused.exception))

    def test_the_ledger_is_per_group_and_is_compared_on_every_group(self):
        network = GroupNetwork(self, size=2)
        network.install_tp()
        network.finish()
        self.assertIsNone(network.driver.error)
        self.assertEqual(network.tp[0].collectives, network.tp[1].collectives)
        self.assertEqual(network.tp[2].collectives, network.tp[3].collectives)
        for rank in ALL:
            accepted = network.qsa[rank].dispatcher
            self.assertIsNone(network.tp[rank].ledger)
            self.assertFalse(accepted.qsa_fused)
            self.assertEqual(accepted.layer_group, 1)
        # GDN group: 4 all-reduces. QSA group: 2 GDN all-reduces, one selection broadcast, one o_proj all-reduce.
        self.assertEqual(network.tp[0].pynccl_comm.calls, 4 + 3 + 4 + 3)
        self.assertEqual(len(network.tp[0].pynccl_comm.broadcasts), 2)

    def test_group_collective_counts_on_the_wire(self):
        network = GroupNetwork(self, size=4)
        network.install_tp()
        endpoint = network.worker(0)
        original = endpoint.flash_s5_submit
        seen = []
        def spy(command):
            reply = original(command)
            if command.kind is w.SegmentKind.LAYER and reply.accepted:
                seen.append((command.descriptor.layer_start, reply.collectives[0]))
            return reply
        endpoint.flash_s5_submit = spy
        network.finish()
        self.assertIsNone(network.driver.error)
        self.assertEqual(seen, [(0, 3 * 2 + 2), (4, 3 * 2 + 2)])


class Burst(GroupCase):
    def network(self, size=2):
        network = GroupNetwork(self, size=size)
        driver = network.driver
        progress = driver.install_progress('burst')
        progress.mode = 'burst'
        # Leaf: the worker seam pops the meta key before dispatch (executor_hooks); no worker seam here.
        patcher = mock.patch.object(ed.NativeProgress, 'with_meta', staticmethod(lambda envelope, meta: envelope))
        patcher.start()
        self.addCleanup(patcher.stop)
        # Leaf: direct replies change the wire path only (test_e3_native_reply_hol proves them).
        submit = network.transport.submit
        network.transport.submit = lambda *args, direct=False, **kwargs: submit(*args, **kwargs)
        return network, driver

    def test_a_poll_in_flight_with_a_group_queued_keeps_the_wait_permit(self):
        for size in (2, 4):
            with self.subTest(size=size):
                network, driver = self.network(size)
                network.run_until(lambda: network.fused.snapshot()['run_ahead_polls'] > 0, ready=())
                network.transport.hold = True
                for _ in range(4):
                    network.tick(ready=())
                    if tuple(driver._inflight.values()) == ('poll',):
                        break
                self.assertEqual(tuple(driver._inflight.values()), ('poll',))
                self.assertTrue(driver.progress.burst())
                self.assertTrue(driver._wait_permit())
                network.transport.deliver_held()
                network.transport.hold = False
                network.finish()
                self.assertIsNone(driver.error)

    def test_a_group_goes_out_inside_an_owner_wait(self):
        network, driver = self.network(2)
        sent = []
        def done():
            driver.progress.last_stock_kind = 'sample_tokens'
            before = network.layer_commands()
            count = driver.progress_in_wait() or 0
            if count and len(network.layer_commands()) > len(before):
                sent.append(network.layer_commands()[-1].descriptor.layer_end - network.layer_commands()[-1].descriptor.layer_start)
            return driver.tx is None and bool(network.layer_commands())
        network.run_until(done, limit=800)
        self.assertIsNone(driver.error)
        self.assertTrue(sent, 'no group was sent inside a wait')
        self.assertIn(2, sent)
        self.assertEqual(network.groups.counters['chunks'], 1)
        self.assertEqual(network.groups.counters['group_commands'], 8)


class Control(unittest.TestCase):
    """The per-arm size and its refusals over the real AutoArmPolicy and driver."""
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
        self.groups = self.d.install_layer_groups()

    def control(self, **fields):
        return self.d.submit_control(dict(action='native_layer_group', **fields))

    def test_selected_only_with_fused_paused_idle_full(self):
        with self.assertRaisesRegex(Exception, 'native-layer-group-requires-fused'):
            self.control(size=2)
        self.assertEqual(self.groups.size, 1)
        self.d.submit_control({'action': 'native_qsa_fused', 'fused': True})
        reply = self.control(size=2)
        self.assertEqual((reply['size'], reply['previous'], reply['route']), (2, 1, 'full'))
        self.assertEqual(self.groups.size, 2)
        self.assertEqual(self.groups.counters['selections'], 1)
        status = self.d.submit_control({'action': 'status'})
        self.assertEqual(status['native_layer_groups']['size'], 2)
        self.assertEqual(status['native_layer_groups']['run_ahead'], 3)
        self.assertTrue(self.policy.manual_owned)
        self.assertEqual(self.control(size=4)['previous'], 2)
        self.assertEqual(self.control(size=1)['previous'], 4)
        self.assertEqual(self.d.submit_control({'action': 'status'})['native_layer_groups']['run_ahead'], 4)

    def test_size_one_without_fused_is_step_a(self):
        self.assertEqual(self.control(size=1)['size'], 1)
        self.assertEqual(self.groups.counters['selections'], 0)

    def test_fused_cannot_be_deselected_under_a_group(self):
        self.d.submit_control({'action': 'native_qsa_fused', 'fused': True})
        self.control(size=2)
        with self.assertRaisesRegex(Exception, 'native-layer-group-requires-fused'):
            self.d.submit_control({'action': 'native_qsa_fused', 'fused': False})
        self.assertTrue(self.fused.selected)
        self.control(size=1)
        self.assertEqual(self.d.submit_control({'action': 'native_qsa_fused', 'fused': False})['fused'], False)

    def test_bad_fields_refused(self):
        self.d.submit_control({'action': 'native_qsa_fused', 'fused': True})
        for fields, text in ((dict(size=3), 'native-layer-group-invalid'),
                             (dict(size=0), 'native-layer-group-invalid'),
                             (dict(size='2'), 'native-layer-group-invalid'),
                             (dict(size=True), 'native-layer-group-invalid'),
                             (dict(size=2.0), 'native-layer-group-invalid'),
                             (dict(), 'native-layer-group-invalid'),
                             (dict(size=2, extra=1), 'native-layer-group-command-fields-differ'),
                             (dict(size=2, engine_epoch=-1), 'native-control-engine-epoch-differs')):
            with self.subTest(fields=fields), self.assertRaisesRegex(Exception, text):
                self.control(**fields)
        self.assertEqual(self.groups.size, 1)

    def test_refused_unless_paused(self):
        self.d.submit_control({'action': 'native_qsa_fused', 'fused': True})
        self.core.paused = False
        with self.assertRaisesRegex(Exception, 'native-layer-group-requires-paused-idle-full'):
            self.control(size=2)
        self.assertEqual(self.groups.size, 1)

    def test_refused_during_an_owned_cycle(self):
        self.d.submit_control({'action': 'native_qsa_fused', 'fused': True})
        self.policy.phase = 'native'
        with self.assertRaisesRegex(Exception, 'cycle-owned'):
            self.control(size=2)
        self.assertEqual(self.groups.size, 1)

    def test_not_a_control_unless_installed(self):
        self.d.layer_groups = None
        with self.assertRaisesRegex(Exception, 'native-control-unknown-action'):
            self.control(size=2)
        self.assertNotIn('native_layer_groups', self.d.submit_control({'action': 'status'}))

    def test_needs_the_fused_switch_and_installs_once(self):
        with self.assertRaisesRegex(ed.DriverError, 'already installed'):
            self.d.install_layer_groups()
        self.d.layer_groups = None
        self.d.qsa_fused = None
        with self.assertRaisesRegex(ed.DriverError, 'fused QSA'):
            self.d.install_layer_groups()


class Exposure(unittest.TestCase):
    ON = {'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_QSA_FUSED': '1', 'FLASH_S5_LAYER_GROUPS': '1'}

    def test_control_client_guard(self):
        for env, exposed in ((self.ON, True),
                             (dict(self.ON, FLASH_S5_LAYER_GROUPS='0'), False),
                             (dict(self.ON, FLASH_S5_QSA_FUSED='0'), False),
                             (dict(self.ON, FLASH_S5_AUTO_ARM='0'), False),
                             ({}, False)):
            with self.subTest(env=env), mock.patch.dict(os.environ, env, clear=True):
                self.assertIs(control._layer_groups(), exposed)

    def test_client_refuses_a_bad_size_before_any_call(self):
        client = control.NativeControlClient.__new__(control.NativeControlClient)
        for size in (0, 3, '2', True, None):
            with self.subTest(size=size), self.assertRaisesRegex(Exception, 'native-control-layer-group-invalid'):
                asyncio.run(client.native_layer_group(size))

    def test_client_call_refuses_before_the_utility_with_any_switch_off(self):
        """The client's own guard, behind the session's: nothing reaches the utility channel."""
        for env in (dict(self.ON, FLASH_S5_LAYER_GROUPS='0'), dict(self.ON, FLASH_S5_QSA_FUSED='0'),
                    dict(self.ON, FLASH_S5_AUTO_ARM='0'), {}):
            sent = []
            async def call_utility_async(utility, payload):
                sent.append(payload)
                return {'size': 2}
            client = control.NativeControlClient.__new__(control.NativeControlClient)
            client.core = types.SimpleNamespace(call_utility_async=call_utility_async)
            client.pending = None
            client.engine_epoch = 7
            with self.subTest(env=env), mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(Exception, 'native-control-unknown-action'):
                    asyncio.run(client._call({'action': 'native_layer_group', 'size': 2}))
                self.assertEqual(sent, [])

    def session_call(self, env, command):
        with mock.patch.dict(os.environ, env, clear=True):
            sess = session.NativeOverlapSession.__new__(session.NativeOverlapSession)
            sess.run_id = None
            sess.measure_start_ns = None
            seen = []
            async def native_layer_group(size):
                seen.append(size)
                return {'size': size}
            sess.control = types.SimpleNamespace(native_layer_group=native_layer_group)
            sess.lock = asyncio.Lock() if hasattr(sess, 'lock') else None
            return sess, seen, asyncio.run(self._execute(sess, command))

    @staticmethod
    async def _execute(sess, command):
        handler = getattr(sess, '_action_native_layer_group')
        return await handler(command)

    def test_session_refuses_during_a_measurement(self):
        sess = session.NativeOverlapSession.__new__(session.NativeOverlapSession)
        sess.measure_start_ns = 1
        with self.assertRaisesRegex(Exception, 'native-layer-group-with-open-measurement'):
            asyncio.run(sess._action_native_layer_group({'size': 2}))

    def test_session_passes_the_size(self):
        _, seen, reply = self.session_call(self.ON, {'size': 4})
        self.assertEqual((seen, reply), ([4], {'size': 4}))


ROUTER_ALL = {'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_NATIVE_PROGRESS': 'burst', 'FLASH_S5_QSA_FUSED': '1',
              'FLASH_S5_LAYER_GROUPS': '1'}


def router_attach(engine_client, values):
    """The real attach_native_router as the API server calls it (its env is the server's)."""
    env = {'FLASH_S5_OVERLAP': '1', 'VLLM_API_KEY': hf.held.KEY}
    env.update(values)
    app = hf.held.App(engine_client)
    with hf.held.fastapi_leaf():
        state = probe_control.attach_native_router(types.SimpleNamespace(api_key=None), app, env=env)
    (path, endpoint, methods), = app.routers[0].routes
    assert (path, methods) == (probe_control.NATIVE_PATH, ('POST',))
    return app, endpoint, state


@contextlib.contextmanager
def process_env(values):
    with mock.patch.dict(os.environ, values):
        for name in ROUTER_ALL:
            if name not in values:
                os.environ.pop(name, None)
        yield


class RouterCase(hf.RouterBase):
    """The real router, session, control client and driver. Leaves are test_native41_progress_queue's, as declared there."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.fused = self.driver.install_qsa_fused()
        self.groups = self.driver.install_layer_groups()

    def unset(self):
        self.assertEqual((self.groups.size, self.groups.counters['selections']), (1, 0))


class Router(RouterCase):
    async def test_all_settings_forward_the_control_to_the_real_session(self):
        with process_env(ROUTER_ALL):
            app, endpoint, _ = router_attach(self.frontend, ROUTER_ALL)
            # A running engine: the engine refuses. Its DriverError is a ValueError, which this router maps to its 400.
            self.fused.selected = True
            self.assertEqual(await self.post(endpoint, app, 'native_layer_group', size=2),
                             (400, {'error': 'invalid-command-json'}))
            self.unset()
            self.fused.selected = False
            self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
            for bad in (0, 3, '2', True, None):
                self.assertEqual(await self.post(endpoint, app, 'native_layer_group', size=bad),
                                 (409, {'error': 'native-control-layer-group-invalid'}))
            self.unset()
            # Fused not selected: size above 1 is refused, size 1 stays allowed.
            self.assertEqual(await self.post(endpoint, app, 'native_layer_group', size=2),
                             (400, {'error': 'invalid-command-json'}))
            self.unset()
            code, reply = await self.post(endpoint, app, 'native_layer_group', size=1)
            self.assertEqual((code, reply.get('previous')), (200, 1), reply)
            self.unset()
            self.assertEqual((await self.post(endpoint, app, 'native_qsa_fused', fused=True))[0], 200)
            code, reply = await self.post(endpoint, app, 'native_layer_group', size=2)
            self.assertEqual(code, 200, reply)
            self.assertEqual({k: reply[k] for k in ('action', 'size', 'previous', 'route')},
                             dict(action='native_layer_group', size=2, previous=1, route='full'))
            self.assertEqual(reply['engine_epoch'], self.driver.engine_epoch)
            self.assertEqual((self.groups.size, self.groups.counters['selections']), (2, 1))
            code, status = await self.post(endpoint, app, 'status')
            self.assertEqual(code, 200, status)
            self.assertEqual({k: status['native_layer_groups'][k] for k in ('size', 'run_ahead', 'selections', 'chunks')},
                             dict(size=2, run_ahead=3, selections=1, chunks=0))
            # Deselecting fused under a group size would leave a size no OPEN can honour.
            self.assertEqual(await self.post(endpoint, app, 'native_qsa_fused', fused=False),
                             (400, {'error': 'invalid-command-json'}))
            self.assertIs(self.fused.selected, True)
            # A manual control: the lease stays taken, and the same value counts no selection.
            self.assertIs(self.policy.manual_owned, True)
            code, reply = await self.post(endpoint, app, 'native_layer_group', size=2)
            self.assertEqual((code, reply.get('previous')), (200, 2))
            self.assertEqual(self.groups.counters['selections'], 1)

    async def test_every_other_setting_refuses_with_409(self):
        router_off = 'native-action-not-exposed'
        session_off = 'session-action-not-exposed'
        for router, process, error in (
                (dict(ROUTER_ALL, FLASH_S5_LAYER_GROUPS='0'), ROUTER_ALL, router_off),   # step A's own settings
                (dict(ROUTER_ALL, FLASH_S5_QSA_FUSED='0'), ROUTER_ALL, router_off),
                (dict(ROUTER_ALL, FLASH_S5_AUTO_ARM='0'), ROUTER_ALL, router_off),
                ({'FLASH_S5_LAYER_GROUPS': '1'}, ROUTER_ALL, router_off),
                (ROUTER_ALL, dict(ROUTER_ALL, FLASH_S5_LAYER_GROUPS='0'), session_off),  # the session reads its own process
                (ROUTER_ALL, dict(ROUTER_ALL, FLASH_S5_QSA_FUSED='0'), session_off),
                (ROUTER_ALL, dict(ROUTER_ALL, FLASH_S5_AUTO_ARM='0'), session_off)):
            with self.subTest(router=router, process=process), process_env(process):
                app, endpoint, _ = router_attach(self.frontend, router)
                self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
                self.assertEqual(await self.post(endpoint, app, 'native_layer_group', size=2),
                                 (409, {'error': error}))
                self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)
                self.unset()


class RouterNotInstalled(hf.RouterBase):
    async def test_an_engine_without_the_switch_refuses_and_its_status_has_no_block(self):
        self.driver.install_qsa_fused()
        with process_env(ROUTER_ALL):
            app, endpoint, _ = router_attach(self.frontend, ROUTER_ALL)
            self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
            self.assertEqual(await self.post(endpoint, app, 'native_layer_group', size=2),
                             (400, {'error': 'invalid-command-json'}))
            self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)
            code, status = await self.post(endpoint, app, 'status')
            self.assertEqual(code, 200)
            self.assertNotIn('native_layer_groups', status)


class Startup(unittest.TestCase):
    ON = dict(FLASH_S5_QSA_FUSED='1', FLASH_S5_AUTO_ARM='1', FLASH_S5_NATIVE_QSA_POLICY='dense',
              FLASH_S5_LAYER_GROUPS='1')

    def refused(self, env, text):
        # Leaf: _enabled reads the overlap composition's settings, which these guards come after.
        with mock.patch.object(s5_startup, '_enabled', lambda env: True), \
                self.assertRaises(s5_startup.StartupError) as refused:
            s5_startup.install_engine(types.SimpleNamespace(), env=env)
        self.assertIn(text, str(refused.exception))

    def test_needs_the_fused_switch(self):
        env = dict(self.ON)
        del env['FLASH_S5_QSA_FUSED']
        self.refused(env, 'needs FLASH_S5_QSA_FUSED=1')
        self.refused(dict(self.ON, FLASH_S5_QSA_FUSED='0'), 'needs FLASH_S5_QSA_FUSED=1')

    def test_only_a_boolean_setting(self):
        for value in ('2', 'true', ''):
            with self.subTest(value=value):
                self.refused(dict(self.ON, FLASH_S5_LAYER_GROUPS=value), 'invalid boolean setting: FLASH_S5_LAYER_GROUPS')


class Mutants(GroupCase):
    """Each new rule, faulted in place: the scenario named beside it must then fail. A survivor is a finding."""
    def fails(self, target, name, replacement, scenario):
        # A scenario class later in the file has not had its class setup yet.
        if hasattr(type(scenario), 'setUpClass'):
            type(scenario).setUpClass()
        with mock.patch.object(target, name, replacement):
            result = unittest.TestResult()
            scenario.run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertTrue(result.failures or result.errors, 'mutant survived: ' + name)
        # A mutant must die on the rule, not because the mutant itself is broken.
        for _, trace in result.failures + result.errors:
            self.assertNotRegex(trace.strip().splitlines()[-1], r'^(TypeError|AttributeError|NameError)', trace)

    def test_plan_groups_by_the_layer_number_not_the_prepare_plan(self):
        def by_number(low, high, qsa_layers, size):
            return ed_plan(low, high, {layer for layer in range(low, high) if layer % 4 == 3}, size)
        ed_plan = ed.layer_group_plan
        self.fails(ed, 'layer_group_plan', by_number,
                   Plan('test_positions_come_from_the_prepare_plan_not_from_the_layer_number'))

    def test_plan_does_not_close_after_a_qsa_layer(self):
        real = ed.layer_group_plan
        self.fails(ed, 'layer_group_plan', lambda low, high, qsa, size: real(low, high, set(), size),
                   Plan('test_groups_close_at_the_size_after_a_qsa_layer_and_at_the_stage_end'))
        self.fails(ed, 'layer_group_plan', lambda low, high, qsa, size: real(low, high, set(), size),
                   Plan('test_positions_come_from_the_prepare_plan_not_from_the_layer_number'))

    def test_plan_crosses_the_stage_end(self):
        real = ed.layer_group_plan
        def crossing(low, high, qsa, size):
            groups = real(low, high, qsa, size)
            return groups[:-1] + [(groups[-1][0], groups[-1][0] + size)]
        self.fails(ed, 'layer_group_plan', crossing,
                   Plan('test_a_stage_without_qsa_groups_too'))

    def test_worker_ignores_the_transactions_latch(self):
        real = ld.LaneDispatcher._do_layer
        def no_latch(self, command):
            self.layer_group = command.layer_group
            return real(self, command)
        self.fails(ld.LaneDispatcher, '_do_layer', no_latch,
                   DefaultOff('test_transactions_not_selecting_groups_refuse_a_group_with_no_device_work'))

    def test_worker_does_not_check_the_stage_end_before_enqueueing(self):
        real = ld.LaneDispatcher._do_layer_group
        def late(self, command):
            lane = self.lane
            end = lane.end_layer
            lane.end_layer = command.layer_end
            try:
                return real(self, command)
            finally:
                lane.end_layer = end
        self.fails(ld.LaneDispatcher, '_do_layer_group', late,
                   WorkerRefusal('test_a_group_that_crosses_its_stage'))

    def test_worker_runs_a_group_short_by_one(self):
        real = ld.LaneDispatcher._do_layer_group
        def short(self, command):
            bounded = types.SimpleNamespace(layer_start=command.layer_start, layer_end=command.layer_end - 1,
                                            profile=None)
            return real(self, bounded if bounded.layer_end > bounded.layer_start else command)
        self.fails(ld.LaneDispatcher, '_do_layer_group', short,
                   OneCommand('test_same_device_work_as_size_one'))

    def test_worker_accepts_a_layer_that_did_not_finish(self):
        import inspect
        source = inspect.getsource(ld.LaneDispatcher._do_layer_group)
        self.assertIn("if record.kind != 'layer':", source)
        mutated = source.replace("if record.kind != 'layer':", 'if False:')
        namespace = dict(vars(ld))
        exec('class Holder:\n' + mutated, namespace)
        self.fails(ld.LaneDispatcher, '_do_layer_group', namespace['Holder']._do_layer_group,
                   Failure('test_a_layer_that_does_not_finish_quarantines'))

    def test_no_obligation_for_a_group_without_qsa(self):
        real = ed.NativeSegmentDriver._send_plan_entry
        def mutant(self, entry):
            result = real(self, entry)
            if self._command is not None and self._command.profile is None:
                self._tail_submission = None
            return result
        self.fails(ed.NativeSegmentDriver, '_send_plan_entry', mutant,
                   Cancel('test_cancel_between_acceptances_in_a_group_without_qsa'))

    def test_no_obligation_for_a_group_with_qsa(self):
        real = ed.NativeSegmentDriver._send_plan_entry
        def mutant(self, entry):
            result = real(self, entry)
            if self._command is not None and self._command.profile is not None:
                self._tail_submission = None
            return result
        self.fails(ed.NativeSegmentDriver, '_send_plan_entry', mutant,
                   Cancel('test_cancel_between_acceptances_in_a_group_with_qsa'))

    def test_size_one_takes_a_group_obligation(self):
        real = ed.NativeSegmentDriver._send_plan_entry
        def mutant(self, entry):
            result = real(self, entry)
            if self._tx_fused and self._command is not None and self._command.kind is w.SegmentKind.LAYER \
                    and self._tail_submission is None and self._command.op_seq > 0:
                self._tail_submission = ed.TPTailSubmissionObligation(self._command, self._command.op_seq - 1,
                                                                      self.route_generation)
            return result
        self.fails(ed.NativeSegmentDriver, '_send_plan_entry', mutant,
                   StepAIdentity('test_size_one_gives_a_lone_gdn_layer_no_obligation_as_step_a'))

    def test_run_ahead_ignores_the_table(self):
        RunAhead.setUpClass()
        with mock.patch.dict(ed.LayerGroups.RUN_AHEAD, {2: 4, 4: 4}):
            result = unittest.TestResult()
            RunAhead('test_the_bound_per_size_with_a_peer_held_in_either_direction').run(result)
        self.assertTrue(result.failures or result.errors, 'mutant survived: run-ahead table')

    def test_run_ahead_reads_one_peer_only(self):
        def one_peer(self, entry):
            if entry['kind'] is not w.SegmentKind.LAYER:
                return False
            rank = self.stage_ranks[entry['stage']][0]
            limit = ed.LayerGroups.RUN_AHEAD[self._tx_group]
            return rank in self.reports and len(self.reports[rank].pending_op_seqs) >= limit
        self.fails(ed.NativeSegmentDriver, '_fused_hold', one_peer,
                   RunAhead('test_the_bound_per_size_with_a_peer_held_in_either_direction'))

    def test_latch_reads_the_live_size_in_a_manual_cycle(self):
        real = ed.NativeSegmentDriver._progress_owned
        self.fails(ed.NativeSegmentDriver, '_progress_owned', lambda self: True,
                   DefaultOff('test_manual_cycles_never_group'))
        self.assertIs(ed.NativeSegmentDriver._progress_owned, real)

    def test_ledger_stops_at_the_first_layer_of_a_group(self):
        real = lane_comm.CollectiveLedger.note
        def first_only(self, op, pynccl, tensor, root=None):
            if self.count < 2:
                return real(self, op, pynccl, tensor, root)
            self.count += 1
        self.fails(lane_comm.CollectiveLedger, 'note', first_only,
                   Ledger('test_equal_counts_with_a_different_shape_inside_a_group_fail_closed'))

    def test_group_counters_lie(self):
        real = ed.NativeSegmentDriver._fused_agree
        def no_counts(self, descriptor, targets, accepted):
            before = dict(self.layer_groups.counters) if self.layer_groups is not None else None
            result = real(self, descriptor, targets, accepted)
            if before is not None:
                self.layer_groups.counters['group_commands'] = before['group_commands']
            return result
        self.fails(ed.NativeSegmentDriver, '_fused_agree', no_counts,
                   OneCommand('test_commands_per_chunk_shrink_by_the_plan'))

    def test_control_accepts_a_size_without_fused(self):
        real = ed.NativeSegmentDriver._layer_group_control
        def mutant(self, command):
            selected = self.qsa_fused.selected
            self.qsa_fused.selected = True
            try:
                return real(self, command)
            finally:
                self.qsa_fused.selected = selected
        self.fails(ed.NativeSegmentDriver, '_layer_group_control', mutant,
                   Control('test_selected_only_with_fused_paused_idle_full'))

    def test_control_accepts_while_running(self):
        real = ed.NativeSegmentDriver._layer_group_control
        def mutant(self, command):
            with mock.patch.object(type(self), '_engine_paused', lambda self: True):
                return real(self, command)
        self.fails(ed.NativeSegmentDriver, '_layer_group_control', mutant,
                   Control('test_refused_unless_paused'))

    def test_fused_deselected_under_a_group(self):
        real = ed.NativeSegmentDriver._fused_control
        def mutant(self, command):
            groups, self.layer_groups = self.layer_groups, None
            try:
                return real(self, command)
            finally:
                self.layer_groups = groups
        self.fails(ed.NativeSegmentDriver, '_fused_control', mutant,
                   Control('test_fused_cannot_be_deselected_under_a_group'))

    def test_journal_row_loses_the_group(self):
        real = ed.NativeSegmentDriver._record
        def mutant(self, kind, **fields):
            fields.pop('layer_group', None)
            return real(self, kind, **fields)
        self.fails(ed.NativeSegmentDriver, '_record', mutant,
                   Journal('test_a_group_chunk_keeps_every_telemetry_row_and_journal_entry_valid'))

    def test_a_group_lifted_for_every_transaction_in_the_local_command(self):
        real = ld.SegmentCommand.__init__
        def mutant(self, *args, **kwargs):
            kwargs['layer_group'] = 4
            return real(self, *args, **kwargs)
        self.fails(ld.SegmentCommand, '__init__', mutant,
                   DefaultOff('test_local_command_still_refuses_more_than_one_layer_without_a_group'))

    def test_a_host_wait_inside_a_group(self):
        real = ld.LaneDispatcher._do_layer_group
        def mutant(self, command):
            result = real(self, command)
            self.native.split.prefill_stream.synchronize = True
            self.operation.device.synchronized += 1
            return result
        self.fails(ld.LaneDispatcher, '_do_layer_group', mutant,
                   OneCommand('test_one_command_per_group_with_every_device_event_unresolved'))

    def test_a_group_reruns_on_a_duplicate(self):
        real = ld.LaneDispatcher.submit
        def mutant(self, command):
            if isinstance(command, ld.SegmentCommand):
                return real(self, command)
            view = ld.SegmentCommand.from_descriptor(command)
            self.accepted.pop(view.op_seq, None)
            return real(self, view)
        self.fails(ld.LaneDispatcher, 'submit', mutant,
                   OneCommand('test_duplicate_delivery_replays_the_ack_with_no_device_work'))

    def test_startup_guard_gone(self):
        real = s5_startup.install_engine
        def mutant(engine, *, env=None, clock=None):
            env = dict(env)
            env['FLASH_S5_QSA_FUSED'] = '1' if env.get('FLASH_S5_QSA_FUSED') != '1' else '1'
            env['FLASH_S5_LAYER_GROUPS'] = env.get('FLASH_S5_LAYER_GROUPS') if env.get('FLASH_S5_LAYER_GROUPS') == '1' else '0'
            return real(engine, env=env)
        self.fails(s5_startup, 'install_engine', mutant, Startup('test_needs_the_fused_switch'))


class Journal(GroupCase):
    @staticmethod
    def reader():
        """The real journal reader's segment rule, on its bounded state alone."""
        reader = nm.NativeMeasurementCensus.__new__(nm.NativeMeasurementCensus)
        reader.pool_chunk_capacity = True
        reader.max_segments_per_chunk = 64
        reader.max_chunks_total = 4
        reader._segment_total = 0
        reader.ranks = ALL
        reader.stage_ranks = STAGE
        reader._conflict = lambda conflicts, reason: conflicts.append(reason)
        reader._overflow = lambda reason, chunk: None
        return reader

    @staticmethod
    def chunk():
        return dict(conflicts=[], commit=None, segments=dict(attrs={}, unique=0, events=0, first_ns=None, last_ns=None,
                                                             transport=None, serialized=False))

    def test_a_group_chunk_keeps_every_telemetry_row_and_journal_entry_valid(self):
        network = GroupNetwork(self, size=4)
        network.finish()
        self.assertEqual(network.driver.qsa_control.faults, 0)
        rows = [entry for entry in network.driver._journal
                if entry['kind'] == 'segment' and entry['segment_kind'] == 'layer']
        self.assertEqual(len(rows), 4)
        reader, chunk = self.reader(), self.chunk()
        for row in rows:
            self.assertEqual(row['layer_end'] - row['layer_start'], 4)
            self.assertEqual(reader._ingest_segment(row, chunk, row['observed_ns']), {'accepted': True})
        self.assertEqual(chunk['conflicts'], [])

    def test_size_one_journal_rows_carry_no_group_field(self):
        network = GroupNetwork(self, size=1)
        network.finish()
        rows = [entry for entry in network.driver._journal if entry['kind'] == 'segment']
        self.assertTrue(rows)
        self.assertTrue(all('layer_group' not in row for row in rows))

    def test_the_measurement_accepts_a_declared_group_and_nothing_longer(self):
        """The real journal reader: a one-layer row, a declared group, and a row longer than it declares."""
        reader = self.reader()
        def ingest(op_seq, start, end, **extra):
            chunk = self.chunk()
            entry = dict(op_seq=op_seq, stage=0, segment_kind='layer', targets=[0, 1], transport='native-split',
                         layer_start=start, layer_end=end, **extra)
            return reader._ingest_segment(entry, chunk, 1), chunk['conflicts']
        self.assertEqual(ingest(0, 0, 1), ({'accepted': True}, []))
        self.assertEqual(ingest(1, 0, 4, layer_group=4), ({'accepted': True}, []))
        self.assertEqual(ingest(2, 0, 3, layer_group=4), ({'accepted': True}, []))
        self.assertEqual(ingest(3, 0, 2)[1], ['segment-layer-shape-invalid'])
        self.assertEqual(ingest(4, 0, 5, layer_group=4)[1], ['segment-layer-shape-invalid'])
        self.assertEqual(ingest(5, 3, 3, layer_group=4)[1], ['segment-layer-shape-invalid'])


if __name__ == '__main__':
    unittest.main()
