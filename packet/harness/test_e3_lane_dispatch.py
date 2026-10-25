"""CPU-only tests for the worker lane dispatcher (D4/D2/D3 seam).

The dispatcher owns wire ordering, bounded host operations, release proofs and
quarantine. Everything here uses fake streams/events/channels; passing is a
contract check, never native/GPU evidence.
"""
import contextlib
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

from e3_types import E3Pending, TransactionId  # noqa: E402
from lane_comm import LaneCommError  # noqa: E402
from lane_dispatch import LaneDispatchError, LaneDispatcher, SegmentCommand  # noqa: E402


TXN = TransactionId(1, 'test-request', 0, 0, 8, 1)
TXN2 = TransactionId(1, 'test-request', 0, 8, 16, 2)

class FakeStream:
    def __init__(self, token):
        self.cuda_stream = token
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)


class FakeEvent:
    def __init__(self, enable_timing=False):
        self.stream = None
        self.complete = True

    def record(self, stream=None):
        self.stream = stream

    def query(self):
        return self.complete


def fake_torch():
    torch = types.SimpleNamespace()
    torch.cuda = types.SimpleNamespace(Event=FakeEvent, stream=lambda s: contextlib.nullcontext(s))
    torch.inference_mode = lambda: contextlib.nullcontext()
    return torch


class FakeLane:
    def __init__(self, layers=(0, 1, 2)):
        self.cursor = None
        self.state = 'new'
        self.output = None
        self.is_sender = True
        self.layers = list(layers)
        self.pause_counts = {}
        self.calls = []

    def begin(self, intermediate_tensors=None):
        self.calls.append(('begin', intermediate_tensors))
        self.cursor = self.layers[0]
        self.state = 'running'
        return types.SimpleNamespace(kind='prefix')

    def layer(self, idx):
        self.calls.append(('layer', idx))
        if self.pause_counts.get(idx, 0) > 0:
            self.pause_counts[idx] -= 1
            return types.SimpleNamespace(kind='layer-paused')
        self.cursor = idx + 1
        return types.SimpleNamespace(kind='layer')

    def finish(self):
        self.calls.append(('finish',))
        self.state = 'finished'
        self.output = {'hidden_states': 'hc'} if self.is_sender else None
        return types.SimpleNamespace(kind='suffix')

    def abort(self, reason):
        self.state = 'aborted'
        self.calls.append(('abort', reason))


class FakeChannel:
    """Minimal PP channel with the release-proof API the dispatcher uses."""

    def __init__(self, is_sender):
        self.is_sender = is_sender
        self.calls = []
        self.ready = False
        self.send_error = None
        self.tickets = []
        self.released = []

    def _ticket(self, kind, transaction, stage, seq):
        ticket = types.SimpleNamespace(kind=kind, transaction=transaction, stage=stage, seq=seq,
                                       state='posted', consumer_event=None, event=FakeEvent())
        self.tickets.append(ticket)
        return ticket

    def post_receive(self, txn, stage, seq, n):
        self.calls.append(('post', stage, seq, n))
        return self._ticket('recv', txn, stage, seq)

    def poll_receive(self, ticket):
        self.calls.append(('poll', ticket.transaction, ticket.stage, ticket.seq))
        if self.ready:
            ticket.state = 'ready'
            return 'payload'
        return None

    def send(self, payload, txn, stage, seq):
        self.calls.append(('send', payload, stage, seq))
        ticket = self._ticket('send', txn, stage, seq)
        if self.send_error is not None:
            # Simulates a failure after the bounded enqueue already happened.
            raise self.send_error
        return ticket

    def consumer_done(self, ticket, consumer_event):
        self.calls.append(('consumer_done', ticket.seq))
        ticket.consumer_event = consumer_event
        return ticket

    def release_receive(self, ticket):
        if ticket.consumer_event is None or not ticket.consumer_event.query():
            raise RuntimeError('receive release before the consumer completed')
        self.calls.append(('release_receive', ticket.seq))
        self.released.append(ticket)
        return True

    def validate_consumer_proof(self, ticket, proof):
        if any(getattr(proof, name, None) != getattr(ticket, name)
               for name in ('transaction', 'stage', 'seq')):
            raise LaneDispatchError('proof identity mismatch')
        return True

    def release_send(self, ticket, proof):
        if proof is None:
            raise RuntimeError('send release needs the consumer proof')
        for name in ('transaction', 'stage', 'seq'):
            if getattr(proof, name, None) != getattr(ticket, name, None):
                raise RuntimeError('proof identity mismatch')
        if not ticket.event.query():
            raise RuntimeError('send release before local completion')
        self.calls.append(('release_send', ticket.seq))
        self.released.append(ticket)
        return True


class FakeDriver:
    def __init__(self):
        self.aborted = None

    def cursor(self):
        return ('qsa', 1, 0)

    def abort(self, reason):
        self.aborted = reason
        return reason


class FakeOperation:
    def __init__(self, drained=True):
        self.drained = drained
        self.drain_calls = 0
        self.released = False

    def cancel(self, reason):
        return reason

    def release(self):
        self.released = True

    def drain(self):
        self.drain_calls += 1
        return self.drained


def make(is_sender=True, torch=None, operation='op', drained=True):
    lane = FakeLane()
    lane.is_sender = is_sender
    chunk = types.SimpleNamespace(req_index=2, lane_inputs=lambda: {'positions': 'p'},
                                  forward_scope=lambda: contextlib.nullcontext(),
                                  finish_state=lambda: True)
    factory = types.SimpleNamespace(admit=lambda adm: chunk, release=lambda: 'req-a', calls=[])
    channel = FakeChannel(is_sender)
    native = types.SimpleNamespace(
        split=types.SimpleNamespace(prefill_stream=FakeStream(7)),
        comms=types.SimpleNamespace(pp=channel), copy_stream=FakeStream(8),
        lane_forward=lambda inputs, forward_context=None, attention_driver=None: lane)
    server = types.SimpleNamespace(calls=[])
    server.begin_prefill_operation = lambda runner, marker: server.calls.append(
        ('begin', marker.flash_e3_transaction))
    server.end_prefill_operation = lambda runner: server.calls.append(('end',))
    server.abort_prefill_operation = lambda runner: server.calls.append(('abort',))
    dispatcher = LaneDispatcher('runner', native, factory, server=server,
                                driver_factory=lambda op, mod, fc: FakeDriver(),
                                torch_module=torch, clock=lambda: 0)
    admission = types.SimpleNamespace(num_tokens=8, transaction=TXN)
    if operation in ('drainable', 'op'):
        operation = FakeOperation(drained)
    return dispatcher, lane, channel, server, admission, operation


def cmd(kind, op_seq, stage=0, **kw):
    if kind in ('pp_send', 'pp_receive') and 'seq' not in kw:
        kw['seq'] = 0
    return SegmentCommand(TXN, stage, kind, op_seq, **kw)


def proof_for(ticket, **changes):
    values = dict(transaction=ticket.transaction, stage=ticket.stage, seq=ticket.seq)
    values.update(changes)
    return types.SimpleNamespace(**values)


class CommandTests(unittest.TestCase):
    def test_validation_and_descriptor(self):
        with self.assertRaises(LaneDispatchError):
            cmd('dance', 1)
        with self.assertRaises(LaneDispatchError):
            cmd('layer', 1)
        with self.assertRaises(LaneDispatchError):
            cmd('layer', 1, layer_start=3, layer_end=3)
        with self.assertRaises(LaneDispatchError):
            cmd('layer', 1, layer_start=0, layer_end=2)  # exactly one layer for now
        with self.assertRaises(LaneDispatchError):
            cmd('prepare', 1, seq=0)  # non-PP kinds do not carry a transfer seq
        with self.assertRaises(LaneDispatchError):
            SegmentCommand(TXN, 0, 'pp_send', 1)  # no implicit seq=0
        with self.assertRaises(LaneDispatchError):
            cmd('pp_send', 1, seq='0')
        with self.assertRaises(LaneDispatchError):
            cmd('prepare', '1')
        desc = types.SimpleNamespace(transaction=TXN, stage=1, kind='layer', op_seq=4,
                                     layer_start=24, layer_end=25, seq=None)
        command = SegmentCommand.from_descriptor(desc)
        self.assertEqual((command.kind, command.layer_start, command.stage, command.seq),
                         ('layer', 24, 1, None))
        pp = types.SimpleNamespace(transaction=TXN, stage=1, kind='pp_receive', op_seq=5,
                                   layer_start=None, layer_end=None, seq=3)
        self.assertEqual(SegmentCommand.from_descriptor(pp).seq, 3)
        with self.assertRaises(LaneDispatchError):
            SegmentCommand.from_descriptor(types.SimpleNamespace(
                transaction=TXN, stage='1', kind='layer', op_seq=4,
                layer_start=0, layer_end=1))
        with self.assertRaises(LaneDispatchError):
            SegmentCommand.from_descriptor(types.SimpleNamespace(
                transaction=TXN, stage=1, kind=3, op_seq=4))
        with self.assertRaises(LaneDispatchError):
            SegmentCommand.from_descriptor(types.SimpleNamespace(
                transaction=TXN, stage=1, kind='pp_receive', op_seq=4))


class DispatcherStage0Tests(unittest.TestCase):
    def setUp(self):
        self.torch = fake_torch()
        self.d, self.lane, self.channel, self.server, self.admission, self.operation = \
            make(True, self.torch)
        self.d.open(TXN, self.admission, self.operation)

    def native_prefill_stream(self):
        return self.d.native.split.prefill_stream

    def _through_layers(self, first_op=3):
        self.d.submit(cmd('prepare', 1))
        self.d.submit(cmd('prefix', 2))
        for op, idx in zip(range(first_op, first_op + 3), (0, 1, 2)):
            self.d.submit(cmd('layer', op, layer_start=idx, layer_end=idx + 1))

    def test_full_stage0_flow(self):
        d = self.d
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('prefix', 1))  # before prepare
        ack = d.submit(cmd('prepare', 1))
        self.assertEqual(ack['result'], {'prepared': True, 'req_index': 2})
        self.assertEqual(self.server.calls, [('begin', TXN)])
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('prepare', 2))
        d.submit(cmd('prefix', 2))
        self.assertEqual(self.lane.calls[-1], ('begin', None))
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('layer', 3, layer_start=1, layer_end=2))  # cursor is 0
        self.lane.pause_counts = {1: 1}
        for op, idx in ((3, 0), (4, 1), (5, 1), (6, 2)):
            d.submit(cmd('layer', op, layer_start=idx, layer_end=idx + 1))
        self.assertEqual(len(d.accepted), 6)
        self.assertEqual(d.cursor(), (3, 'running', ('qsa', 1, 0)))
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('state_finalize', 7))  # before suffix
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('pp_send', 7))  # before suffix
        ack = d.submit(cmd('suffix', 7))
        self.assertTrue(ack['result']['has_output'])
        ack = d.submit(cmd('pp_send', 8, seq=3))
        self.assertEqual(self.channel.calls[-1], ('send', 'hc', 0, 3))
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('pp_send', 9, seq=4))
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('pp_receive', 9, seq=5))  # sender cannot receive
        d.submit(cmd('state_finalize', 9))
        self.assertEqual(self.server.calls[-1], ('end',))
        self.assertIsNotNone(d.state_event)
        self.assertEqual(len(self.native_prefill_stream().waited), 1,
                         'copy stream joined before STATE_WRITE')
        report = d.poll()
        self.assertEqual(report['completed_op_seqs'], (1, 2, 3, 4, 5, 6, 7, 8, 9))
        self.assertTrue(report['state_written'])
        # Sender release also needs the engine's identity-matched consumer proof.
        self.assertFalse(d.drain())
        with self.assertRaises(E3Pending):
            d.release()
        ticket = d.pp_send_ticket
        with self.assertRaises(LaneDispatchError):
            d.accept_pp_consumer_proof(proof_for(ticket, seq=99))
        self.assertTrue(d.accept_pp_consumer_proof(proof_for(ticket)))
        self.assertTrue(d.drain())
        self.assertTrue(d.release())
        self.assertIn(('release_send', 3), self.channel.calls)
        self.assertIsNone(d.transaction)
        self.assertEqual(d.close_request(), 'req-a')

    def test_pending_events_block_drain_and_release(self):
        d = self.d
        d.submit(cmd('prepare', 1))
        d.events[-1].event.complete = False
        report = d.poll()
        self.assertEqual(report['pending_op_seqs'], (1,))
        self.assertFalse(d.drain())
        with self.assertRaises(E3Pending):
            d.release()
        with self.assertRaises(LaneDispatchError):
            d.close_request()
        d.events[-1].event.complete = True
        self.assertFalse(d.drain(), 'no final state proof')
        d.cancel()
        self.assertTrue(d.drain())

    def test_operation_drain_is_required(self):
        d, lane, channel, server, admission, operation = make(
            True, self.torch, operation='drainable', drained=False)
        d.open(TXN, admission, operation)
        d.submit(cmd('prepare', 1))
        self.assertFalse(d.drain())
        operation.drained = True
        self.assertFalse(d.drain(), 'no final state proof')
        d.cancel()
        self.assertTrue(d.drain())
        self.assertGreaterEqual(operation.drain_calls, 2)

    def test_cancel_stops_submissions_and_aborts(self):
        d, lane, channel, server, admission, operation = make(True, self.torch)
        with self.assertRaises(LaneDispatchError):
            d.cancel()
        d.open(TXN, admission, operation)
        d.submit(cmd('prepare', 1))
        d.submit(cmd('prefix', 2))
        self.assertEqual(d.cancel('aborted'), 'aborted')
        self.assertEqual(lane.state, 'aborted')
        self.assertEqual(d.driver.aborted, 'aborted')
        self.assertEqual(server.calls[-1], ('abort',))
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('layer', 3, layer_start=0, layer_end=1))
        self.assertTrue(d.drain())
        self.assertTrue(d.release())

    def test_cancel_does_not_bypass_pp_release_proofs(self):
        d = self.d
        self._through_layers()
        d.submit(cmd('suffix', 6))
        d.submit(cmd('pp_send', 7, seq=2))
        d.pp_send_ticket.event.complete = False
        d.cancel('stop')
        # An already-accepted submission is still an idempotent replay after
        # cancel; a genuinely new segment is refused.
        calls = list(self.lane.calls)
        replay = d.submit(cmd('prefix', 2))
        self.assertEqual(replay['op_seq'], 2)
        self.assertEqual(self.lane.calls, calls)
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('state_finalize', 8))
        self.assertFalse(d.drain(), 'cancelled send is still outstanding')
        d.accept_pp_consumer_proof(proof_for(d.pp_send_ticket))
        self.assertFalse(d.drain(), 'local send completion is still pending')
        with self.assertRaises(E3Pending):
            d.release()
        d.pp_send_ticket.event.complete = True
        self.assertTrue(d.drain())
        self.assertTrue(d.release())
        self.assertIn(('release_send', 2), self.channel.calls)

    def test_wrong_stage_is_refused(self):
        d = self.d
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('prepare', 1, stage=1))
        self.assertEqual(d.accepted, {})
        self.assertEqual(d.events, [])

    def test_handler_failure_after_partial_enqueue_quarantines(self):
        d = self.d
        self._through_layers()
        d.submit(cmd('suffix', 6))
        self.channel.send_error = RuntimeError('enqueue failed')
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('pp_send', 7, seq=2))
        self.assertIn('pp_send', d.quarantined)
        self.assertIn('RuntimeError', d.quarantined)
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('state_finalize', 8))
        self.assertFalse(d.drain())
        with self.assertRaises(LaneDispatchError):
            d.release()

    def test_event_recording_failure_quarantines(self):
        d = self.d
        d.submit(cmd('prepare', 1))

        def boom(enable_timing=False):
            raise RuntimeError('event record failed')

        d.torch.cuda.Event = boom
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('prefix', 2))
        self.assertIn('RuntimeError', d.quarantined)
        self.assertFalse(d.drain())
        with self.assertRaises(LaneDispatchError):
            d.release()


class DispatcherStage1Tests(unittest.TestCase):
    def setUp(self):
        self.torch = fake_torch()
        self.d, self.lane, self.channel, self.server, self.admission, self.operation = \
            make(False, self.torch)
        self.d.open(TXN, self.admission, self.operation)

    def test_receive_before_prefix_and_no_suffix_output(self):
        d = self.d
        d.submit(cmd('prepare', 1, stage=1))
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('prefix', 2, stage=1))  # no receive posted
        d.submit(cmd('pp_receive', 2, stage=1, seq=3))
        self.assertEqual(self.channel.calls[-1], ('post', 0, 3, 8))
        with self.assertRaises(E3Pending):
            d.submit(cmd('prefix', 3, stage=1))  # receive not complete
        self.assertNotIn(3, d.accepted, 'pending submission consumes no op_seq')
        self.assertEqual(len(d.events), 2, 'pending submission records no event')
        self.channel.ready = True
        d.submit(cmd('prefix', 3, stage=1))  # the same op_seq may retry
        self.assertEqual(self.lane.calls[-1], ('begin', {'hidden_states': 'payload'}))
        self.assertTrue(d.poll()['pp_receive_ready'])
        for op, idx in ((4, 0), (5, 1), (6, 2)):
            d.submit(cmd('layer', op, stage=1, layer_start=idx, layer_end=idx + 1))
        # Keep HC through all tails and the final state fence.
        self.assertEqual(self.channel.calls.count(('consumer_done', 3)), 0)
        ack = d.submit(cmd('suffix', 7, stage=1))
        self.assertFalse(ack['result']['has_output'])
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('pp_send', 8, stage=1, seq=4))
        d.submit(cmd('state_finalize', 8, stage=1))
        self.assertTrue(d.drain())
        self.assertTrue(d.release())
        self.assertIn(('release_receive', 3), self.channel.calls)

    def test_receive_release_waits_for_the_first_layer_event(self):
        d = self.d
        d.submit(cmd('prepare', 1, stage=1))
        d.submit(cmd('pp_receive', 2, stage=1, seq=3))
        self.channel.ready = True
        d.submit(cmd('prefix', 3, stage=1))
        d.submit(cmd('layer', 4, stage=1, layer_start=0, layer_end=1))
        self.assertIsNone(d.pp_ticket.consumer_event)
        d.submit(cmd('layer', 5, stage=1, layer_start=1, layer_end=2))
        d.submit(cmd('layer', 6, stage=1, layer_start=2, layer_end=3))
        d.submit(cmd('suffix', 7, stage=1))
        d.submit(cmd('state_finalize', 8, stage=1))
        consumer = d.pp_ticket.consumer_event
        self.assertIs(consumer, d.events[-1].event)
        consumer.complete = False
        self.assertFalse(d.drain(), 'PP consumer must be host-queried complete')
        with self.assertRaises(E3Pending):
            d.release()
        consumer.complete = True
        self.assertTrue(d.drain())
        self.assertTrue(d.release())

    def test_cancel_with_pending_event_needs_a_query(self):
        d = self.d
        d.submit(cmd('prepare', 1, stage=1))
        d.submit(cmd('pp_receive', 2, stage=1, seq=3))
        self.channel.ready = True
        d.submit(cmd('prefix', 3, stage=1))
        d.submit(cmd('layer', 4, stage=1, layer_start=0, layer_end=1))
        d.events[-1].event.complete = False
        d.cancel('stop')
        self.assertFalse(d.drain(), 'cancelled copy/PP events still need a query')
        with self.assertRaises(E3Pending):
            d.release()
        d.events[-1].event.complete = True
        self.assertTrue(d.drain())
        self.assertTrue(d.release())

    def test_poll_receive_error_quarantines(self):
        d = self.d
        d.submit(cmd('prepare', 1, stage=1))
        d.submit(cmd('pp_receive', 2, stage=1, seq=3))

        def fail(ticket):
            raise LaneCommError('header mismatch')

        self.channel.poll_receive = fail
        d.poll()
        self.assertIn('LaneCommError', d.quarantined)
        self.assertFalse(d.drain())
        with self.assertRaises(LaneDispatchError):
            d.release()


class OrderingTests(unittest.TestCase):
    def setUp(self):
        self.torch = fake_torch()
        self.d, self.lane, self.channel, self.server, self.admission, self.operation = \
            make(True, self.torch)
        self.d.open(TXN, self.admission, self.operation)

    def test_repeated_paused_phases_and_idempotent_resubmission(self):
        d, lane = self.d, self.lane
        d.submit(cmd('prepare', 1))
        d.submit(cmd('prefix', 2))
        lane.pause_counts = {0: 2}
        ack3 = d.submit(cmd('layer', 3, layer_start=0, layer_end=1))
        ack4 = d.submit(cmd('layer', 4, layer_start=0, layer_end=1))
        self.assertTrue(ack3['result']['paused'])
        self.assertTrue(ack4['result']['paused'])
        d.submit(cmd('layer', 5, layer_start=0, layer_end=1))  # advances the cursor
        self.assertEqual(list(d.accepted), [1, 2, 3, 4, 5])
        calls = list(lane.calls)
        events = list(d.events)
        # Earlier phase's event is ready; the later one is still pending.
        d.events[3].event.complete = True
        d.events[4].event.complete = False
        # An identical resubmission is idempotent: stored ack, nothing executes.
        self.assertEqual(d.submit(cmd('layer', 4, layer_start=0, layer_end=1)), ack4)
        self.assertEqual(d.submit(cmd('layer', 3, layer_start=0, layer_end=1)), ack3)
        self.assertEqual(lane.calls, calls)
        self.assertEqual(d.events, events)
        # Reusing an accepted op_seq for a different command is refused.
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('layer', 3, layer_start=1, layer_end=2))
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('suffix', 0))  # lower and never accepted
        # A lower, never-accepted op_seq is refused.
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('suffix', 2))
        self.assertIn(5, d.poll()['pending_op_seqs'])
        self.assertIn(4, d.poll()['completed_op_seqs'])
        d.events[4].event.complete = True
        self.assertFalse(d.drain())
        d.cancel()
        self.assertTrue(d.drain())

    def test_second_transaction_resets_every_per_transaction_field(self):
        d = self.d
        d.submit(cmd('prepare', 1))
        d.submit(cmd('prefix', 2))
        for op, idx in ((3, 0), (4, 1), (5, 2)):
            d.submit(cmd('layer', op, layer_start=idx, layer_end=idx + 1))
        d.submit(cmd('suffix', 6))
        d.submit(cmd('pp_send', 7, seq=0))
        d.accept_pp_consumer_proof(proof_for(d.pp_send_ticket))
        d.submit(cmd('state_finalize', 8))
        self.assertTrue(d.release())
        self.assertEqual((d.events, d.accepted, d.pp_ticket, d.pp_send_ticket,
                          d.state_event, d.chunk, d.lane, d.driver, d.operation,
                          d.cancelled, d.quarantined, d.last_ns),
                         ([], {}, None, None, None, None, None, None, None, None, None, None))
        second = types.SimpleNamespace(num_tokens=8, transaction=TXN2)
        d.open(TXN2, second, FakeOperation())
        self.assertEqual((d.events, d.accepted, d.pp_ticket, d.pp_send_ticket,
                          d.state_event, d.chunk, d.cancelled), ([], {}, None, None, None, None, None))
        # op_seq ordering restarts for the new transaction.
        d.submit(SegmentCommand(TXN2, 0, 'prepare', 1))
        d.submit(SegmentCommand(TXN2, 0, 'prefix', 2))
        self.assertEqual(list(d.accepted), [1, 2])

    def test_open_checks_the_admission_transaction(self):
        with self.assertRaises(LaneDispatchError):
            self.d.open(TXN, types.SimpleNamespace(num_tokens=8, transaction='other'), 'op')
        with self.assertRaises(LaneDispatchError):
            self.d.open(TXN, types.SimpleNamespace(num_tokens=8), 'op')


class ReviewRegressions(unittest.TestCase):
    def make(self, sender=True):
        values = make(sender, fake_torch())
        values[0].open(TXN, values[4], values[5])
        return values

    def test_missing_drain_never_means_drained(self):
        d, _, _, _, adm, _ = make(True, fake_torch())
        with self.assertRaises(LaneDispatchError):
            d.open(TXN, adm, object())
        self.assertIsNone(d.transaction)

    def test_real_default_loads_torch_not_host_completion(self):
        from unittest.mock import patch
        torch = fake_torch()
        d, _, _, _, _, _ = make(True, torch)
        with patch.dict(sys.modules, {'torch': torch}):
            real_default = LaneDispatcher(d.runner, d.native, d.factory)
        self.assertIs(real_default.torch, torch)

    def test_admission_failure_quarantines_before_factory_returns(self):
        d, *_ = self.make()
        def fail(admission):
            raise RuntimeError('private diagnostic text')
        d.factory.admit = fail
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('prepare', 1))
        self.assertIsNotNone(d.quarantined)
        self.assertNotIn('private diagnostic text', d.quarantined)
        self.assertFalse(d.drain())
        with self.assertRaises(LaneDispatchError):
            d.release()

    def test_descriptor_rejects_hidden_seq_and_pp_layer_bounds(self):
        desc = types.SimpleNamespace(transaction=TXN, stage=0, kind='prepare', op_seq=1, seq=4)
        with self.assertRaises(LaneDispatchError):
            SegmentCommand.from_descriptor(desc)
        with self.assertRaises(LaneDispatchError):
            cmd('pp_send', 1, seq=0, layer_start=0, layer_end=1)
        with self.assertRaises(LaneDispatchError):
            cmd('prepare', 1, stage=2)

    def test_stored_ack_and_command_are_not_mutable_by_caller(self):
        d, lane, *_ = self.make()
        command = cmd('prepare', 1)
        ack = d.submit(command)
        ack['result']['prepared'] = False
        command.kind = 'prefix'
        self.assertTrue(d.submit(cmd('prepare', 1))['result']['prepared'])
        with self.assertRaises(LaneDispatchError):
            d.submit(command)
        self.assertEqual(lane.calls, [])

    def test_paused_first_receiver_layer_is_not_consumer_completion(self):
        d, lane, ch, *_ = self.make(False)
        d.submit(cmd('prepare', 1, stage=1))
        d.submit(cmd('pp_receive', 2, stage=1, seq=0))
        ch.ready = True
        d.submit(cmd('prefix', 3, stage=1))
        lane.pause_counts = {0: 1}
        d.submit(cmd('layer', 4, stage=1, layer_start=0, layer_end=1))
        self.assertIsNone(d.pp_ticket.consumer_event)
        self.assertFalse(d.drain())

    def test_cancel_before_receiver_layer_queries_receive_and_releases(self):
        d, _, ch, _, _, operation = self.make(False)
        d.submit(cmd('prepare', 1, stage=1))
        d.submit(cmd('pp_receive', 2, stage=1, seq=0))
        d.cancel()
        self.assertFalse(d.drain())
        ch.ready = True
        self.assertTrue(d.drain())
        d.release()
        self.assertTrue(operation.released)

    def test_query_error_quarantines(self):
        d, *_ = self.make()
        d.submit(cmd('prepare', 1))
        def fail():
            raise RuntimeError('event destroyed')
        d.events[0].event.query = fail
        with self.assertRaises(RuntimeError):
            d.poll()
        self.assertFalse(d.drain())

    def test_pending_after_possible_enqueue_quarantines(self):
        d, *_ = self.make()
        def fail(admission):
            raise E3Pending('ambiguous submission')
        d.factory.admit = fail
        with self.assertRaises(E3Pending):
            d.submit(cmd('prepare', 1))
        self.assertFalse(d.drain())
        self.assertIsNotNone(d.quarantined)

    def test_cancel_is_idempotent_and_retains_input_owners(self):
        d, lane, *_ = self.make()
        d.submit(cmd('prepare', 1))
        lane.output = object()
        owner = lane.output
        d.cancel()
        d.cancel()
        self.assertIn(owner, d._cancel_owners[1])
        d.release()
        self.assertEqual(d._cancel_owners, ())


if __name__ == '__main__':
    unittest.main()
