"""CPU-only tests for the worker lane dispatcher (D4/D2/D3 seam)."""
import contextlib
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

from e3_types import E3Pending  # noqa: E402
from lane_dispatch import LaneDispatchError, LaneDispatcher, SegmentCommand  # noqa: E402


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
        self.layers = list(layers)
        self.pause_once = set()
        self.calls = []

    def begin(self, intermediate_tensors=None):
        self.calls.append(('begin', intermediate_tensors))
        self.cursor = self.layers[0]
        self.state = 'running'
        return types.SimpleNamespace(kind='prefix')

    def layer(self, idx):
        self.calls.append(('layer', idx))
        if idx in self.pause_once:
            self.pause_once.discard(idx)
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
    def __init__(self, is_sender):
        self.is_sender = is_sender
        self.calls = []
        self.ready = False

    def post_receive(self, txn, stage, seq, n):
        self.calls.append(('post', stage, seq, n))
        return types.SimpleNamespace(state='posted')

    def poll_receive(self, ticket):
        if self.ready:
            ticket.state = 'ready'
            return 'payload'
        return None

    def send(self, payload, txn, stage, seq):
        self.calls.append(('send', payload, stage, seq))
        return types.SimpleNamespace(state='posted')


class FakeDriver:
    def __init__(self):
        self.aborted = None

    def cursor(self):
        return ('qsa', 1, 0)

    def abort(self, reason):
        self.aborted = reason
        return reason


def make(is_sender=True, torch=None):
    lane = FakeLane()
    lane.is_sender = is_sender
    chunk = types.SimpleNamespace(req_index=2, lane_inputs=lambda: {'positions': 'p'},
                                  forward_scope=lambda: contextlib.nullcontext(), finish_state=lambda: True,
                                  finished=[])
    factory = types.SimpleNamespace(admit=lambda adm: chunk, release=lambda: 'req-a', calls=[])
    channel = FakeChannel(is_sender)
    native = types.SimpleNamespace(
        split=types.SimpleNamespace(prefill_stream=FakeStream(7)),
        comms=types.SimpleNamespace(pp=channel), copy_stream=FakeStream(8),
        lane_forward=lambda inputs, forward_context=None, attention_driver=None: lane)
    server = types.SimpleNamespace(calls=[])
    server.begin_prefill_operation = lambda runner, marker: server.calls.append(('begin', marker.flash_e3_transaction))
    server.end_prefill_operation = lambda runner: server.calls.append(('end',))
    server.abort_prefill_operation = lambda runner: server.calls.append(('abort',))
    dispatcher = LaneDispatcher('runner', native, factory, server=server,
                                driver_factory=lambda op, mod, fc: FakeDriver(),
                                torch_module=torch, clock=lambda: 0)
    admission = types.SimpleNamespace(num_tokens=8)
    return dispatcher, lane, channel, server, admission


def cmd(kind, op_seq, stage=0, **kw):
    return SegmentCommand('txn', stage, kind, op_seq, **kw)


class CommandTests(unittest.TestCase):
    def test_validation_and_descriptor(self):
        with self.assertRaises(LaneDispatchError):
            cmd('dance', 1)
        with self.assertRaises(LaneDispatchError):
            cmd('layer', 1)
        with self.assertRaises(LaneDispatchError):
            cmd('layer', 1, layer_start=3, layer_end=3)
        desc = types.SimpleNamespace(transaction='txn', stage=1, kind='layer', op_seq=4,
                                     layer_start=24, layer_end=25, seq=0)
        command = SegmentCommand.from_descriptor(desc)
        self.assertEqual((command.kind, command.layer_start, command.stage), ('layer', 24, 1))


class DispatcherStage0Tests(unittest.TestCase):
    def setUp(self):
        self.torch = fake_torch()
        self.d, self.lane, self.channel, self.server, self.admission = make(True, self.torch)

    def test_full_stage0_flow(self):
        d = self.d
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('prefix', 1))
        d.open('txn', self.admission, 'op')
        with self.assertRaises(LaneDispatchError):
            d.open('txn', self.admission, 'op')
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('prefix', 1))  # before prepare
        ack = d.submit(cmd('prepare', 1))
        self.assertEqual(ack['result'], {'prepared': True, 'req_index': 2})
        self.assertEqual(self.server.calls, [('begin', 'txn')])
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('prepare', 2))
        d.submit(cmd('prefix', 2))
        self.assertEqual(self.lane.calls[-1], ('begin', None))
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('layer', 3, layer_start=1, layer_end=2))  # cursor is 0
        self.lane.pause_once.add(1)
        for op, idx in ((3, 0), (4, 1), (5, 1), (6, 2)):
            ack = d.submit(cmd('layer', op, layer_start=idx, layer_end=idx + 1))
        paused = [a for a in d.accepted]
        self.assertEqual(len(paused), 6)
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
            d.submit(cmd('pp_send', 9))
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('pp_receive', 9))  # sender cannot receive
        ack = d.submit(cmd('state_finalize', 9))
        self.assertEqual(self.server.calls[-1], ('end',))
        self.assertIn(d.state_event, self.native_events())
        self.assertEqual(len(self.native_prefill_stream().waited), 1, 'copy stream joined before STATE_WRITE')
        report = d.poll()
        self.assertEqual(report['completed_op_seqs'], (1, 2, 3, 4, 5, 6, 7, 8, 9))
        self.assertTrue(report['state_written'])
        self.assertTrue(d.drain())
        self.assertTrue(d.release())
        self.assertIsNone(d.transaction)
        self.assertEqual(d.close_request(), 'req-a')

    def native_prefill_stream(self):
        return self.d.native.split.prefill_stream

    def native_events(self):
        return [self.d.state_event]

    def test_pending_events_block_drain_and_release(self):
        d = self.d
        d.open('txn', self.admission, 'op')
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
        self.assertTrue(d.drain())

    def test_cancel_stops_submissions_and_aborts(self):
        d = self.d
        with self.assertRaises(LaneDispatchError):
            d.cancel()
        d.open('txn', self.admission, 'op')
        d.submit(cmd('prepare', 1))
        d.submit(cmd('prefix', 2))
        self.assertEqual(d.cancel('aborted'), 'aborted')
        self.assertEqual(self.lane.state, 'aborted')
        self.assertEqual(d.driver.aborted, 'aborted')
        self.assertEqual(self.server.calls[-1], ('abort',))
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('layer', 3, layer_start=0, layer_end=1))
        self.assertTrue(d.drain())
        self.assertTrue(d.release())


class DispatcherStage1Tests(unittest.TestCase):
    def test_receive_before_prefix_and_no_suffix_output(self):
        d, lane, channel, server, admission = make(False, fake_torch())
        d.open('txn', admission, 'op')
        d.submit(cmd('prepare', 1, stage=1))
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('prefix', 2, stage=1))  # no receive posted
        d.submit(cmd('pp_receive', 2, stage=1, seq=3))
        self.assertEqual(channel.calls[-1], ('post', 0, 3, 8))
        with self.assertRaises(E3Pending):
            d.submit(cmd('prefix', 3, stage=1))  # receive not complete
        channel.ready = True
        d.submit(cmd('prefix', 3, stage=1))
        self.assertEqual(lane.calls[-1], ('begin', {'hidden_states': 'payload'}))
        self.assertTrue(d.poll()['pp_receive_ready'])
        for op, idx in ((4, 0), (5, 1), (6, 2)):
            d.submit(cmd('layer', op, stage=1, layer_start=idx, layer_end=idx + 1))
        ack = d.submit(cmd('suffix', 7, stage=1))
        self.assertFalse(ack['result']['has_output'])
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd('pp_send', 8, stage=1))
        d.submit(cmd('state_finalize', 8, stage=1))
        self.assertTrue(d.drain())


if __name__ == '__main__':
    unittest.main()
