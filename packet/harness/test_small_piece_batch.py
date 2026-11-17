#!/usr/bin/env python3
"""Host-gap batch for small prefill pieces (FLASH_S5_SMALL_PIECE_BATCH): CPU proofs.

Connected network: test_e3_stage_overlap's ring (real StagedAdmissionController, the batch driver = OverlapSegmentDriver
subclass, four real NativeWorkerBridge / LaneDispatcher workers, BoundedSender ring, ReplyTransport receivers, generated
worker hooks; every envelope and reply pickled) with alpha-next-c1's composed selections (fused selected, layer groups at
size 1, auto-arm owned).  The ring model has 2 GDN layers per stage, chunk 4 rows (a small piece), so each stage's two
LAYER commands are one batch.

Declared leaves (beyond those of test_alpha_next_c1_composed_fields): ``ring-probe`` — small_piece_batch.ring_waiting
reads the fake ring's per-rank lag (the stock MessageQueue peek is not reachable from the stage_net loop, whose frame is not
``_execute_worker_rpc``); ``live-pending`` patched only in the run-ahead test.  No GPU, CUDA, NCCL, MQ or model forward;
nothing here is device evidence (TP collectives and stream order are the box's).
"""
import pickle
import unittest
from unittest import mock

import test_e3_stage_overlap as t  # first: its imports put the engine sources on the path
import test_alpha_next_c1_composed_fields as cf
import small_piece_batch as spb
import stage_overlap as so
import stage_wire
import executor_hooks as eh
from engine_driver import DriverError
from e3_types import E3ContractError

w = t.w
LAYER = w.SegmentKind.LAYER


def batch_class():
    return spb.driver_class()


class BatchStack(t.OverlapStack):
    """OverlapStack constructing the batch driver (the class s5_startup picks with the setting on)."""

    def __init__(self, **kwargs):
        with mock.patch.object(t, 'OverlapSegmentDriver', batch_class()):
            super().__init__(**kwargs)


def messages(network, method):
    out = []
    for name, message, _targets in map(t.stage_of_command, network.sent_envelopes):
        if name == method:
            out.append(message)
    return out


class Base(cf.Fixture):
    def setUp(self):
        super().setUp()
        self.ring_override = {}
        self.current = None

        def probe(bridge):
            if bridge.rank in self.ring_override:
                return self.ring_override[bridge.rank]()
            return self.current.ring.lag(bridge.rank) > 0
        patcher = mock.patch.object(spb, 'ring_waiting', probe)
        patcher.start()
        self.addCleanup(patcher.stop)

    def batched(self, lean=False, **kwargs):
        network = BatchStack(**kwargs)
        self.addCleanup(network.close_network)
        self.current = network
        return self.composed(group=1, lean=lean, network=network)

    def plain(self, lean=False, **kwargs):
        network = self.network(**kwargs)
        self.current = network
        return self.composed(group=1, lean=lean, network=network)

    def assert_same(self, off, on):
        self.assertEqual(off.worker_errors, [])
        self.assertEqual(on.worker_errors, [])
        for rank in (0, 1, 2, 3):
            self.assertEqual(on.ledgers[rank], off.ledgers[rank], rank)
        self.assertEqual(on.persistent_state(), off.persistent_state())

    def accepted(self, network, rank):
        """(txn, op_seq, kind, layer) in this rank's accepted order, for the live transaction."""
        dispatcher = network.worker(rank).dispatcher
        return [(command.transaction.txn_seq, op_seq, command.kind, command.layer_start)
                for op_seq, (command, _ack) in sorted(dispatcher.accepted.items())]


# -- worker side, unit -----------------------------------------------------------------
class FakeBridge:
    def __init__(self, rank=0, pending=0, ring=False):
        self.rank = rank
        self.ring = ring
        self.pending = pending
        self.dispatcher = self

    def ring_waiting(self):
        return self.ring

    def poll(self):
        return {'pending_op_seqs': tuple(range(self.pending))}

    def _cursor(self):
        return w.CompletionCursor('layer', 0, 0)


def layer_commands(count=2, start=5):
    tx = w.TransactionId(7, 'req', 1, 0, 4, 1)
    commands = []
    for index in range(count):
        descriptor = w.SegmentDescriptor(tx, LAYER, 0, layer_start=index, layer_end=index + 1)
        commands.append(w.SegmentCommand(descriptor, start + index, (0, 1)))
    return tuple(commands)


def acceptance(command, rank=0, **flags):
    flags.setdefault('accepted', True)
    return w.SegmentAcceptance(command.transaction, rank, command.kind, command.stage, command.op_seq,
                               cursor=w.CompletionCursor('layer', 0, 0), **flags)


class WorkerBatch(unittest.TestCase):
    def setUp(self):
        try:
            self.commands = layer_commands()
        except E3ContractError as exc:   # pragma: no cover - fixture construction
            self.skipTest('wire fixture: %s' % exc)
        self.ran = []

    def dispatch(self, command):
        self.ran.append(command.op_seq)
        return acceptance(command)

    def test_both_run_in_order_without_a_waiting_message(self):
        reply = spb.run_batch(FakeBridge(), spb.SegmentBatch(self.commands, 4), self.dispatch)
        self.assertEqual(self.ran, [5, 6])
        self.assertEqual(reply.stop, 'complete')
        self.assertEqual([item.op_seq for item in reply.items], [5, 6])
        self.assertTrue(all(item.accepted for item in reply.items))
        pickle.loads(pickle.dumps(reply))

    def test_a_waiting_ring_message_stops_after_the_first_command(self):
        """The decode guarantee: with any message waiting, the worker runs no second command."""
        reply = spb.run_batch(FakeBridge(ring=True), spb.SegmentBatch(self.commands, 4), self.dispatch)
        self.assertEqual(self.ran, [5])
        self.assertEqual(reply.stop, 'ring')
        self.assertTrue(reply.items[1].pending and not reply.items[1].accepted)
        self.assertIsNone(reply.items[1].report)

    def test_the_run_ahead_bound_is_checked_live(self):
        for pending, ran in ((3, [5, 6]), (4, [5]), (9, [5])):
            with self.subTest(pending=pending):
                self.ran = []
                reply = spb.run_batch(FakeBridge(pending=pending), spb.SegmentBatch(self.commands, 4), self.dispatch)
                self.assertEqual(self.ran, ran)
                self.assertEqual(reply.stop, 'complete' if len(ran) == 2 else 'run_ahead')

    def test_a_failure_in_the_second_command_is_typed_and_nothing_after_it_runs(self):
        def dispatch(command):
            self.ran.append(command.op_seq)
            if command.op_seq == 6:
                raise RuntimeError('boom')
            return acceptance(command)
        reply = spb.run_batch(FakeBridge(), spb.SegmentBatch(self.commands, 4), dispatch)
        self.assertEqual(self.ran, [5, 6])
        self.assertEqual(reply.stop, 'failed')
        self.assertEqual(len(reply.items), 1)
        self.assertEqual((reply.failure.index, reply.failure.error), (1, 'RuntimeError'))

    def test_a_failure_in_the_first_command_takes_todays_path(self):
        def dispatch(command):
            raise RuntimeError('boom')
        with self.assertRaises(RuntimeError):
            spb.run_batch(FakeBridge(), spb.SegmentBatch(self.commands, 4), dispatch)

    def test_a_first_command_not_accepted_stops_with_no_marker(self):
        for flags in (dict(accepted=False, pending=True),):
            with self.subTest(flags=flags):
                self.ran = []

                def dispatch(command):
                    self.ran.append(command.op_seq)
                    return acceptance(command, **flags)
                reply = spb.run_batch(FakeBridge(), spb.SegmentBatch(self.commands, 4), dispatch)
                self.assertEqual(self.ran, [5])
                self.assertEqual((reply.stop, len(reply.items)), ('not_accepted', 1))

    def test_batch_record_refusals(self):
        commands = self.commands
        with self.assertRaises(E3ContractError):
            spb.SegmentBatch(commands[:1], 4)
        with self.assertRaises(E3ContractError):
            spb.SegmentBatch(layer_commands(3), 4)       # more than BATCH_MAX
        with self.assertRaises(E3ContractError):
            spb.SegmentBatch((commands[1], commands[0]), 4)
        with self.assertRaises(E3ContractError):
            spb.SegmentBatch(commands, 0)
        tx = commands[0].transaction
        prepare = w.SegmentCommand(w.SegmentDescriptor(tx, w.SegmentKind.SUFFIX, 0), 6, (0, 1))
        with self.assertRaises(E3ContractError):
            spb.SegmentBatch((commands[0], prepare), 4)

    def test_without_a_readable_ring_the_worker_yields(self):
        """No ring to read reads as 'a message waits': the safe default is today's one command."""
        class NoRing:
            rank = 0
            dispatcher = object()
        # The real function (this class does not patch it).
        self.assertTrue(spb.ring_waiting(NoRing()))
        self.ran = []
        bridge = FakeBridge()
        bridge.dispatcher = type('D', (), {'poll': FakeBridge.poll, 'pending': 0})()
        reply = spb.run_batch(bridge, spb.SegmentBatch(self.commands, 4), self.dispatch)
        self.assertEqual((self.ran, reply.stop), ([5], 'ring'))


class Setting(unittest.TestCase):
    def test_values(self):
        self.assertFalse(spb.small_piece_batch_enabled({}))
        self.assertFalse(spb.small_piece_batch_enabled({spb.SMALL_PIECE_BATCH_ENV: '0'}))
        self.assertTrue(spb.small_piece_batch_enabled({spb.SMALL_PIECE_BATCH_ENV: '1'}))
        for value in ('', 'on', '2', 'true'):
            with self.assertRaises(E3ContractError):
                spb.small_piece_batch_enabled({spb.SMALL_PIECE_BATCH_ENV: value})

    def test_startup_picks_the_batch_class_only_under_the_setting(self):
        import inspect
        import s5_startup
        source = inspect.getsource(s5_startup)
        gate = source.index("if env.get('FLASH_S5_SMALL_PIECE_BATCH', '0') != '0':")
        overlap = source.index('driver_class = OverlapSegmentDriver')
        self.assertLess(overlap, gate)
        self.assertIn('driver_class = small_piece_driver()', source[gate:gate + 600])

    def test_off_hooks_leave_every_existing_record_as_it_was(self):
        """Flag off: the wire map and the scoped submit answer exactly as before for every existing record."""
        commands = layer_commands()
        self.assertEqual(eh._wire_method(commands[0]), 'submit')
        self.assertIsNone(eh._wire_method(object()))
        self.assertEqual(eh._wire_method(spb.SegmentBatch(commands, 4)), 'submit')
        seen = []
        bridge = FakeBridge()

        def dispatch(message):
            seen.append(message)
            return 'answer'
        scoped = stage_wire.StageScoped(commands[0], (0, 1))
        self.assertEqual(stage_wire._scoped_answer(bridge, 'submit', scoped, dispatch), 'answer')
        self.assertEqual(seen, [commands[0]])
        self.assertIsNone(spb.scoped_submit(bridge, commands[0], dispatch))

    def test_the_frozen_driver_class_is_untouched(self):
        cls = batch_class()
        self.assertTrue(issubclass(cls, so.OverlapSegmentDriver))
        for name in ('_send_segment', '_send_plan_entry', '_apply', '_apply_segments', '_status_record'):
            self.assertNotIn(name, so.OverlapSegmentDriver.__dict__.get('__small_piece__', {}))
            self.assertIsNot(getattr(so.OverlapSegmentDriver, name), cls.__dict__[name])
        self.assertFalse(hasattr(so.OverlapSegmentDriver, 'small_batch_status'))


# -- connected engine + workers ------------------------------------------------------------
class Connected(Base):
    def test_identical_device_work_and_state_as_the_unbatched_driver(self):
        """Ordering, numerics and barriers: per-rank device ledgers (PREPARE/profile order, PP edges, state finalize,
        release) and persistent state equal the plain overlap driver's; the batches really ran two commands."""
        off = self.plain()
        self.run_overlapped(off)
        on = self.batched()
        self.run_overlapped(on)
        self.assert_same(off, on)
        for rank in (0, 1, 2, 3):
            self.assertEqual(self.accepted(on, rank), self.accepted(off, rank), rank)
        status = on.driver.small_batch_status()
        self.assertGreater(status['sent'], 0, status)
        self.assertEqual(status['failures'], 0)
        self.assertEqual(status['outstanding'], 0)
        self.assertGreater(status['worker_stop']['complete'], 0, status)
        self.assertGreater(status['second_applied'], 0, status)
        self.assertEqual(status['pieces']['batched'] + status['pieces']['unbatched'], 2 * on.expected_commits)
        with mock.patch.object(so.OverlapSegmentDriver, '_status_record', lambda driver, now=None: {}):
            self.assertEqual(on.driver._status_record()['small_piece_batch'], status)
        # Fewer submit RPCs than the unbatched driver, by the batches whose second command ran in the batch.
        self.assertLess(len(messages(on, 'flash_s5_submit')), len(messages(off, 'flash_s5_submit')))

    def test_lean_replies_keep_their_chain_through_batches_and_yields(self):
        """Lean reports: every sent delta is ingested in order (a yield marker carries none); some batches complete and
        some yield, and the result equals the unbatched lean run."""
        off = self.plain(lean=True)
        self.run_overlapped(off)
        on = self.batched(lean=True)
        flip = [0]

        def alternate():
            flip[0] += 1
            return flip[0] % 3 == 0
        self.ring_override[0] = alternate
        self.run_overlapped(on)
        self.assert_same(off, on)
        status = on.driver.small_batch_status()
        self.assertGreater(status['worker_stop']['complete'], 0, status)
        self.assertGreater(status['partial'] + status['second_not_run'], 0, status)
        self.assertGreater(on.driver.progress.counters['lean_reports'], 0)

    def test_overlap_observation_stays_whole_with_batches(self):
        """FLASH_S5_STAGE_OVERLAP_OBSERVE is on in the box arms: the observer bounds every accepted op of a batch from
        its submit and apply stamps, and emits its lines; equal device work to the observed unbatched run."""
        off = self.plain()
        off.driver.overlap_observer = so.OverlapObserver()
        self.run_overlapped(off)
        on = self.batched()
        on.driver.overlap_observer = so.OverlapObserver()
        self.run_overlapped(on)
        self.assert_same(off, on)
        self.assertGreater(on.driver.small_batch_status()['second_applied'], 0)
        counters_on, counters_off = on.driver.overlap_observer.counters, off.driver.overlap_observer.counters
        self.assertEqual(counters_on['ops'], counters_off['ops'])
        self.assertGreater(counters_on['lines'], 0)
        self.assertEqual(counters_on['submits_dropped'], 0)

    def test_only_layer_pairs_are_batched_and_every_other_kind_stays_single(self):
        on = self.batched()
        self.run_overlapped(on)
        batches = [m for m in messages(on, 'flash_s5_submit') if type(m) is spb.SegmentBatch]
        self.assertTrue(batches)
        for batch in batches:
            self.assertEqual([c.kind for c in batch.commands], [LAYER, LAYER])
            self.assertEqual(batch.commands[1].op_seq, batch.commands[0].op_seq + 1)
            self.assertEqual(batch.commands[1].descriptor.depends_on[0].op_seq, batch.commands[0].op_seq)
        singles = [m for m in messages(on, 'flash_s5_submit') if type(m) is w.SegmentCommand]
        kinds = {m.kind for m in singles}
        self.assertTrue({w.SegmentKind.PREPARE, w.SegmentKind.PREFIX, w.SegmentKind.SUFFIX, w.SegmentKind.PP_SEND,
                         w.SegmentKind.PP_RECEIVE, w.SegmentKind.STATE_FINALIZE} <= kinds)

    def test_a_large_piece_is_never_batched(self):
        with mock.patch.object(spb, 'SMALL_PIECE_ROWS', 3):
            on = self.batched()
            self.run_overlapped(on)
        self.assertEqual(on.driver.small_batch_status()['sent'], 0)
        self.assertFalse([m for m in messages(on, 'flash_s5_submit') if type(m) is spb.SegmentBatch])

    def test_unfused_transactions_are_not_batched(self):
        on = BatchStack()
        self.addCleanup(on.close_network)
        self.current = on
        self.composed(group=None, fused=False, network=on)
        self.reach(on, lambda: self.finished(on), limit=4000, tick=self.slow_tick(on))
        self.assertIsNone(on.driver.error)
        status = on.driver.small_batch_status()
        self.assertEqual(status['sent'], 0)
        self.assertGreater(status['refused']['not_fused'], 0)

    def first_batch_written(self, network):
        return any(type(m) is spb.SegmentBatch and m.commands[0].stage == 0
                   for m in messages(network, 'flash_s5_submit'))

    def drive_until_batch(self, network):
        """Tick (stage 1 slow) until the engine wrote a stage-0 batch the stage-0 ranks have not read yet."""
        tick_count = [0]
        for _ in range(400):
            network.admit_next()
            network.complete_everything()
            # Step everything except the stage-0 ranks' read of a just-written batch.
            for rank in (0, 1, 2, 3):
                while True:
                    head = self.peek(network, rank)
                    if head is not None and type(head) is spb.SegmentBatch and rank in (0, 1):
                        break
                    if not network.step(rank):
                        break
            network.settle()
            if all(type(self.peek(network, rank)) is spb.SegmentBatch for rank in (0, 1)):
                return
            network.driver.advance()
            tick_count[0] += 1
        self.fail('no stage-0 batch reached the ring')

    def peek(self, network, rank):
        ring = network.ring
        index = ring.readers[rank]
        slot = ring.ranks.index(rank)
        meta = ring.buffer.metadata[index]
        if not meta[0] or meta[1 + slot]:
            return None
        method, args, _kwargs, _rank = pickle.loads(ring.buffer.data[index])
        message = args[0] if args else None
        return message.message if type(message) is stage_wire.StageScoped else message

    def finish(self, network):
        self.reach(network, lambda: self.finished(network), limit=4000, tick=self.slow_tick(network))
        self.assertIsNone(network.driver.error)
        self.assertEqual(network.controller.commits, network.expected_commits)
        self.assert_closed(network)

    def test_a_stock_message_behind_a_batch_waits_for_one_native_command_only(self):
        """Decode-starvation guard: a stock RPC written after the batch is read after exactly one LAYER command."""
        on = self.batched()
        self.drive_until_batch(on)
        before = {rank: len(on.worker(rank).dispatcher.accepted) for rank in (0, 1)}
        on.stock_send()
        for rank in (0, 1):
            self.assertTrue(on.step(rank))          # the batch
            self.assertEqual(len(on.worker(rank).dispatcher.accepted) - before[rank], 1, rank)
            head = on.ring.read(rank)               # the very next message is the stock RPC
            self.assertEqual(head[0], 'stock_rpc')
        on.settle()
        on.driver.advance()
        status = on.driver.small_batch_status()
        self.assertGreaterEqual(status['worker_stop']['ring'], 2)
        self.assertGreaterEqual(status['second_not_run'], 1)
        self.finish(on)
        off = self.plain()
        self.run_overlapped(off)
        self.assert_same(off, on)

    def test_peers_stopping_at_different_commands_is_the_frozen_partial_retry(self):
        """TP divergence by one: rank 0 yields, rank 1 runs the second command.  Command 0 keeps its accounting, command 1
        is partially accepted (frozen _retry), the retry finishes it, and the result equals the unbatched run."""
        on = self.batched()
        self.drive_until_batch(on)
        self.ring_override[0] = lambda: True
        self.ring_override[1] = lambda: False
        on.step(0)
        on.step(1)
        on.settle()
        del self.ring_override[0], self.ring_override[1]
        driver = on.driver
        driver.advance()
        frame = driver._frames[0] if driver._frames[0].stage[0]['_retry'] is not None else driver._frames[-1]
        retry = frame.stage[0]['_retry']
        self.assertIsNotNone(retry)
        self.assertEqual(retry.kind, LAYER)
        accepted = {rank for rank, op in frame.chunk['accepted_tickets'] if op == retry.op_seq and rank in (0, 1)}
        self.assertEqual(accepted, {1})
        self.assertEqual(driver.small_batch_status()['partial'], 1)
        self.finish(on)
        off = self.plain()
        self.run_overlapped(off)
        self.assert_same(off, on)

    def test_a_stop_with_a_partially_accepted_second_command_replays_it_before_the_cancel(self):
        on = self.batched()
        self.drive_until_batch(on)
        self.ring_override[0] = lambda: True
        self.ring_override[1] = lambda: False
        on.step(0)
        on.step(1)
        on.settle()
        self.ring_override.clear()
        driver = on.driver
        # The stop comes before the reply is applied, so the frozen retry is never sent outside the stop path.
        driver.stop('cancel')
        for _ in range(400):
            if not driver.live:
                break
            on.complete_everything()
            on.pump()
            driver.pump_stop()
        self.assertEqual(on.worker_errors, [])
        self.assertGreaterEqual(driver._stop_partial_replays, 1)
        replays = [entry for entry in driver._journal if entry.get('kind') == 'stop_partial_replay']
        self.assertTrue(replays)
        self.assertEqual(replays[-1]['accepted_ranks'], [1])
        self.assertEqual(replays[-1]['pending_ranks'], [0])

    def test_a_failure_in_the_second_command_keeps_command_zero_and_never_applies_command_one(self):
        on = self.batched()
        self.drive_until_batch(on)
        dispatcher = on.worker(1).dispatcher
        real = dispatcher.submit
        target = self.peek(on, 1).commands[1].op_seq

        def failing(command):
            if command.op_seq == target:
                raise RuntimeError('injected')
            return real(command)
        dispatcher.submit = failing
        self.ring_override[0] = self.ring_override[1] = lambda: False
        on.step(0)
        on.step(1)
        on.settle()
        driver = on.driver
        with self.assertRaises(DriverError) as caught:
            driver.advance()
        self.assertIn('small-piece-batch-command-failed', str(caught.exception))
        frame = [f for f in driver._frames
                 if any(op == target - 1 for _r, op in f.chunk['accepted_tickets'])][0]
        tickets = frame.chunk['accepted_tickets']
        self.assertIn((0, target - 1), tickets)
        self.assertIn((1, target - 1), tickets)
        self.assertNotIn((0, target), tickets)
        self.assertNotIn((1, target), tickets)
        self.assertEqual(driver.small_batch_status()['failures'], 1)
        # Rank 0 ran it (as a peer of a failing command would have today); rank 1 ran nothing after it.
        self.assertIn(target, on.worker(0).dispatcher.accepted)
        self.assertNotIn(target + 1, on.worker(1).dispatcher.accepted)

    def test_the_run_ahead_bound_stops_the_second_command(self):
        on = self.batched()
        with mock.patch.object(spb, 'live_pending', lambda bridge: 99):
            self.run_overlapped(on)
        status = on.driver.small_batch_status()
        self.assertGreater(status['worker_stop']['run_ahead'], 0)
        self.assertEqual(status['second_applied'], 0)
        off = self.plain()
        self.run_overlapped(off)
        self.assert_same(off, on)

    def test_a_wrong_prediction_fails_closed(self):
        on = self.batched()
        cls = type(on.driver)
        real = cls._predict

        def wrong(driver, entry, previous, targets):
            command = real(driver, entry, previous, targets)
            descriptor = w.SegmentDescriptor(command.transaction, command.kind, command.stage, depends_on=(),
                                             layer_start=command.descriptor.layer_start,
                                             layer_end=command.descriptor.layer_end,
                                             layer_group=command.descriptor.layer_group)
            return w.SegmentCommand(descriptor, command.op_seq, command.ranks, profile=command.profile)
        with mock.patch.object(cls, '_predict', wrong):
            self.ring_override.update({rank: (lambda: False) for rank in (0, 1, 2, 3)})
            tick = self.slow_tick(on)
            for _ in range(4000):
                if on.driver.error is not None:
                    break
                try:
                    tick()
                except DriverError as exc:
                    self.assertIn('small-piece-batch', str(exc))
                    break
            else:
                self.fail('a wrong prediction was applied')


class StopBetweenSendAndApply(Base):
    """night-pack fix 7 (review-reentry-batch): a stop of the kinds the junction-yield path raises on a live chunk (the
    profile 'abort' of _yield_stop and a stock 'pause'; 'cancel' as the control) arriving after the batch was sent and
    BOTH stage-0 peers ran command 1, before the engine applied the batch reply. The batch reply is then applied under
    the stop: command 0 and command 1 are each applied once (no 'small-piece-batch-diverged'), the stop closes with
    no worker error and no live chunk, and the ranks' accepted order matches the unbatched commands."""

    drive_until_batch, peek = Connected.drive_until_batch, Connected.peek

    def run_stop(self, kind):
        on = self.batched()
        self.drive_until_batch(on)
        batch = self.peek(on, 0)
        self.ring_override[0] = self.ring_override[1] = lambda: False
        on.step(0)
        on.step(1)                                  # both peers run command 0 and command 1
        for rank in (0, 1):
            self.assertIn(batch.commands[1].op_seq, on.worker(rank).dispatcher.accepted, rank)
        self.ring_override.clear()
        driver = on.driver
        driver.stop(kind)                           # before the batch reply is read and applied
        on.settle()
        for _ in range(400):
            if not driver.live:
                break
            on.complete_everything()
            on.pump()
            driver.pump_stop()
        self.assertIsNone(driver.error)
        self.assertEqual(on.worker_errors, [])
        self.assertFalse(driver.live)
        status = driver.small_batch_status()
        self.assertEqual(status['failures'], 0)
        self.assertEqual(status['worker_stop'].get('complete', 0), 2, status['worker_stop'])
        self.assertEqual(status['second_applied'], 1, status)
        self.assertEqual(status['partial'], 0)
        return on

    def test_abort(self):
        self.run_stop('abort')

    def test_pause(self):
        self.run_stop('pause')

    def test_cancel(self):
        self.run_stop('cancel')


class Mutants(Base):
    """Each guard is load-bearing: the named mutant fails the named test."""

    def run_test(self, case, name):
        suite = unittest.TestSuite([case(name)])
        return unittest.TextTestRunner(stream=open('/dev/null', 'w'), verbosity=0).run(suite)

    def test_no_ring_check_starves_a_stock_message(self):
        with mock.patch.object(spb, '_yield_reason', lambda bridge, batch: None):
            result = self.run_test(Connected, 'test_a_stock_message_behind_a_batch_waits_for_one_native_command_only')
        self.assertTrue(result.failures or result.errors, 'mutant survived: no ring check')

    def test_unchecked_prediction_is_caught(self):
        cls = batch_class()
        real = cls._send_segment

        def lax(driver, command, targets):
            if driver._batch_capture is not None:
                driver._batch_capture = command
            return real(driver, command, targets)
        with mock.patch.object(cls, '_send_segment', lax):
            result = self.run_test(Connected, 'test_a_wrong_prediction_fails_closed')
        self.assertTrue(result.failures or result.errors, 'mutant survived: unchecked prediction')


if __name__ == '__main__':
    unittest.main()
