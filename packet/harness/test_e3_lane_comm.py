"""CPU-only tests for the E3 prefill-lane communicators (D4).

Fake PyNccl communicators share an in-process wire; fake torch supplies
streams, events and tensors. These exercise routing, header protocol, slot
credits and release ordering; they are not NCCL or GPU evidence.
"""
import contextlib
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import lane_comm  # noqa: E402
from e3_types import (TransactionId, LaneId, LaneRole, OperationTicket,
                      DependencyRole, CompletionKind, CompletionProof)  # noqa: E402
from lane_comm import (  # noqa: E402
    HEADER_MAGIC,
    PPConsumerReceipt,
    LaneCommError,
    PrefillCommunicators,
    PrefillPPChannel,
    PrefillTpCommunicator,
    header_values,
    tp_scope,
)


# --- fakes ------------------------------------------------------------------

class FakeStream:
    def __init__(self, token):
        self.cuda_stream = token
        self.synchronized = 0

    def synchronize(self):
        self.synchronized += 1


class FakeEvent:
    pending = []

    def __init__(self, enable_timing=False):
        self.stream = None
        self.complete = True

    def record(self, stream=None):
        self.stream = stream
        if FakeEvent.pending:
            self.complete = False
            FakeEvent.pending[0].append(self)

    def query(self):
        return self.complete


class FakeTensor:
    def __init__(self, values, shape, element_size=2, pinned=False):
        self.values = list(values)
        self.shape = tuple(shape)
        self._element_size = element_size
        self.pinned = pinned
        self.device = 'cuda:0'
        self.copies = 0

    def element_size(self):
        return self._element_size

    def numel(self):
        return len(self.values)

    def copy_(self, other, non_blocking=False):
        self.values[:len(other.values)] = list(other.values)
        self.copies += 1
        parent = getattr(self, 'parent', None)
        if parent is not None:
            base, offset = parent
            base.values[offset:offset + len(other.values)] = list(other.values)
        return self

    def __getitem__(self, item):
        if isinstance(item, slice):
            rows = range(*item.indices(self.shape[0]))
            width = self.shape[1] if len(self.shape) > 1 else 1
            view = FakeTensor(self.values[rows.start * width:rows.stop * width],
                              (len(rows),) + self.shape[1:], self._element_size)
            view.parent = (self, rows.start * width)
            return view
        raise TypeError(item)

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class FakeCuda:
    def __init__(self):
        self._stack = [FakeStream(1)]
        self.Event = FakeEvent

    def current_stream(self, device=None):
        return self._stack[-1]

    @contextlib.contextmanager
    def stream(self, stream):
        self._stack.append(stream)
        try:
            yield stream
        finally:
            self._stack.pop()


def fake_torch():
    torch = types.SimpleNamespace()
    torch.cuda = FakeCuda()
    torch.int64 = 'int64'
    torch.float32 = 'float32'
    torch.bfloat16 = 'bfloat16'

    def _numel(shape):
        n = 1
        for dim in shape:
            n *= dim
        return n

    def size_of(dtype):
        return 8 if dtype == 'int64' else 4 if dtype == 'float32' else 2

    torch.empty = lambda shape, dtype=None, device=None, pin_memory=False: FakeTensor(
        [0] * _numel(shape), shape, size_of(dtype), pin_memory)
    torch.ones = lambda shape, dtype=None, device=None: FakeTensor([1] * _numel(shape), shape, size_of(dtype))
    torch.full = lambda shape, fill, dtype=None, device=None: FakeTensor([fill] * _numel(shape), shape, size_of(dtype))
    torch.tensor = lambda values, dtype=None: FakeTensor(values, (len(values),), size_of(dtype))
    return torch


class FakeWire:
    def __init__(self):
        self.messages = []


class FakePyNccl:
    def __init__(self, rank, world_size, wire=None, disabled=False):
        self.rank = rank
        self.world_size = world_size
        self.wire = wire
        self.disabled = disabled
        self.calls = []

    def all_reduce(self, in_tensor, out_tensor=None, stream=None):
        self.calls.append(('all_reduce', in_tensor.numel(), stream.cuda_stream))
        return FakeTensor([v * self.world_size for v in in_tensor.values], in_tensor.shape)

    def send(self, tensor, dst, stream=None):
        self.calls.append(('send', tensor.numel(), dst, stream.cuda_stream))
        self.wire.messages.append(list(tensor.values))

    def recv(self, tensor, src, stream=None):
        self.calls.append(('recv', tensor.numel(), src, stream.cuda_stream))
        values = self.wire.messages.pop(0)
        if len(values) != tensor.numel():
            raise AssertionError('wire message size mismatch')
        tensor.values[:] = values
        parent = getattr(tensor, 'parent', None)
        if parent is not None:
            base, offset = parent
            base.values[offset:offset + len(values)] = values


def txn(seq=1, start=0, end=1024):
    return TransactionId(1, 'req', 0, start, end, seq)


# --- TP ---------------------------------------------------------------------

class TpCommunicatorTests(unittest.TestCase):
    def setUp(self):
        FakeEvent.pending = []
        self.torch = fake_torch()
        self.nccl = FakePyNccl(0, 2)

    def test_all_reduce_on_current_stream_and_refusals(self):
        comm = PrefillTpCommunicator(self.nccl, 'cuda:0', torch_module=self.torch)
        lane = FakeStream(77)
        with self.torch.cuda.stream(lane):
            out = comm.all_reduce(FakeTensor([1, 2], (2,)))
        self.assertEqual(out.values, [2, 4])
        self.assertEqual(self.nccl.calls[-1], ('all_reduce', 2, 77))
        self.assertEqual(comm.collectives, 1)
        for name in ('all_gather', 'reduce_scatter', 'broadcast', 'send', 'recv', 'dispatch'):
            with self.assertRaises(LaneCommError):
                getattr(comm, name)(None)
        self.assertIsNone(comm.ca_comm)
        with self.assertRaises(LaneCommError):
            PrefillTpCommunicator(FakePyNccl(0, 2, disabled=True), 'cuda:0', torch_module=self.torch)

    def test_warm_checks_sum(self):
        comm = PrefillTpCommunicator(self.nccl, 'cuda:0', torch_module=self.torch)
        lane = FakeStream(5)
        self.assertTrue(comm.warm(lane))
        self.assertTrue(comm.warmed)
        self.assertEqual(lane.synchronized, 1)
        bad = FakePyNccl(0, 3)
        bad.world_size = 3
        comm = PrefillTpCommunicator(bad, 'cuda:0', torch_module=self.torch)
        comm.world_size = 2  # expected sum 2, fake produces 3
        with self.assertRaises(LaneCommError):
            comm.warm(lane)

    def test_tp_scope_swaps_and_restores(self):
        comm = PrefillTpCommunicator(self.nccl, 'cuda:0', torch_module=self.torch)
        stock = object()
        group = types.SimpleNamespace(device_communicator=stock)
        with tp_scope(group, comm):
            self.assertIs(group.device_communicator, comm)
            with self.assertRaises(LaneCommError):
                with tp_scope(group, comm):
                    pass
        self.assertIs(group.device_communicator, stock)
        with self.assertRaises(RuntimeError):
            with tp_scope(group, comm):
                raise RuntimeError('segment failed')
        self.assertIs(group.device_communicator, stock)
        with self.assertRaises(LaneCommError):
            with tp_scope(group, object()):
                pass
        with self.assertRaises(LaneCommError):
            with tp_scope(types.SimpleNamespace(), comm):
                pass


# --- PP ---------------------------------------------------------------------

class PPChannelTests(unittest.TestCase):
    def setUp(self):
        FakeEvent.pending = []
        self.torch = fake_torch()
        self.wire = FakeWire()
        self.stream0 = FakeStream(10)
        self.stream1 = FakeStream(11)
        self.sender = PrefillPPChannel(FakePyNccl(0, 2, self.wire), 'cuda:0', self.stream0,
                                       max_tokens=4, hidden_width=3, dtype='bfloat16',
                                       is_sender=True, peer_rank=1, torch_module=self.torch)
        self.receiver = PrefillPPChannel(FakePyNccl(1, 2, self.wire), 'cuda:1', self.stream1,
                                         max_tokens=4, hidden_width=3, dtype='bfloat16',
                                         is_sender=False, peer_rank=0, torch_module=self.torch)

    def test_header_values(self):
        values = header_values(txn(7, 0, 1024), 0, 3, 1024)
        self.assertEqual(values[0], HEADER_MAGIC)
        self.assertEqual(values[2], 7)
        self.assertEqual(values[7], 1024)
        with self.assertRaises(LaneCommError):
            header_values(txn(), 0, 0, 0)
        with self.assertRaises(LaneCommError):
            header_values(txn(), 0, 0, 2048)
        with self.assertRaises(LaneCommError):
            header_values(object(), 0, 0, 1)
        for stage, seq in ((0, 1_000_000), (1, 0)):
            with self.assertRaises(LaneCommError):
                header_values(txn(), stage, seq, 1)

    def test_round_trip_validates_header_and_releases_in_order(self):
        payload = FakeTensor([1, 2, 3, 4, 5, 6], (2, 3))
        t = txn(1, 0, 4)
        ticket = self.sender.send(payload, t, 0, 0)
        self.assertEqual(ticket.state, 'posted')
        self.assertEqual([c[0] for c in self.sender.pynccl.calls], ['send', 'send'])
        self.assertEqual(self.sender.pynccl.calls[0][3], 10, 'sent on the lane stream')
        self.assertEqual(len(self.wire.messages), 2)
        # Receiver posts with a pending completion event: poll returns None.
        FakeEvent.pending = [[]]
        rticket = self.receiver.post_receive(t, 0, 0, 2)
        self.assertIsNone(self.receiver.poll_receive(rticket))
        for event in FakeEvent.pending[0]:
            event.complete = True
        FakeEvent.pending = []
        view = self.receiver.poll_receive(rticket)
        self.assertEqual(view.values, [1, 2, 3, 4, 5, 6])
        self.assertEqual(rticket.state, 'ready')
        self.assertEqual(self.receiver.poll_receive(rticket).values, view.values, 'ready polls are idempotent')
        with self.assertRaises(LaneCommError):
            self.receiver.release_receive(rticket)
        consumer = FakeEvent()
        consumer.complete = False
        self.receiver.consumer_done(rticket, consumer)
        with self.assertRaises(LaneCommError):
            self.receiver.release_receive(rticket)
        consumer.complete = True
        self.assertTrue(self.receiver.release_receive(rticket))
        self.assertEqual(self.receiver.outstanding(), 0)
        with self.assertRaises(LaneCommError):
            self.sender.release_send(ticket, None)
        # The typed consumer proof binds transfer, peer storage and route.
        peer = LaneId(2, LaneRole.PREFILL, 1, 7)
        with self.assertRaises(LaneCommError):
            self.sender.release_send(ticket, types.SimpleNamespace(transaction=t, stage=0, seq=0))
        with self.assertRaises(LaneCommError):
            self.sender.bind_consumer(peer, 2)  # live slot cannot be rebound
        self.sender.consumer_binding = (peer, 2)  # fixture pre-admission binding
        proof = PPConsumerReceipt(0, 0, 2, CompletionProof(
            OperationTicket(t, 2, peer, 9, DependencyRole.HANDOFF),
            CompletionKind.PP_CONSUMER, 1))
        for bad in (types.SimpleNamespace(transaction=txn(2, 0, 4), stage=0, seq=0),
                    types.SimpleNamespace(transaction=t, stage=1, seq=0),
                    types.SimpleNamespace(transaction=t, stage=0, seq=1),
                    'pp-consumer-proof'):
            with self.assertRaises(LaneCommError):
                self.sender.release_send(ticket, bad)
        ticket.event.complete = False
        with self.assertRaises(LaneCommError):
            self.sender.release_send(ticket, proof)
        ticket.event.complete = True
        self.assertTrue(self.sender.release_send(ticket, proof))
        self.assertEqual(self.sender.free_slots, [1, 0])

    def test_receipt_requires_real_kind_peer_epoch_and_route(self):
        peer = LaneId(2, LaneRole.PREFILL, 3, 10)
        self.sender.bind_consumer(peer, 7)
        t = txn(1, 0, 4)
        sent = self.sender.send(FakeTensor([0] * 6, (2, 3)), t, 0, 4)
        def receipt(lane=peer, route=7, kind=CompletionKind.PP_CONSUMER,
                    role=DependencyRole.HANDOFF):
            return PPConsumerReceipt(0, 4, route, CompletionProof(
                OperationTicket(t, lane.worker_rank, lane, 20, role), kind, 1))
        for lane, route in ((LaneId(2, LaneRole.PREFILL, 2, 10), 7),
                            (LaneId(3, LaneRole.PREFILL, 3, 10), 7), (peer, 6)):
            with self.assertRaises(LaneCommError):
                self.sender.release_send(sent, receipt(lane, route))
        for kind in (CompletionKind.GPU_EVENT, CompletionKind.STATE_WRITE):
            with self.assertRaises(LaneCommError):
                receipt(kind=kind)
        with self.assertRaises(LaneCommError):
            receipt(role=DependencyRole.PRODUCER)
        self.assertTrue(self.sender.release_send(sent, receipt()))

    def test_header_mismatch_quarantines(self):
        t = txn(1, 0, 4)
        self.sender.send(FakeTensor([0] * 6, (2, 3)), t, 0, 0)
        self.wire.messages[0][2] = 99  # tamper txn_seq on the wire
        rticket = self.receiver.post_receive(t, 0, 0, 2)
        with self.assertRaises(LaneCommError):
            self.receiver.poll_receive(rticket)
        self.assertEqual(rticket.state, 'quarantined')
        with self.assertRaises(LaneCommError):
            self.receiver.post_receive(t, 0, 1, 2)

    def test_backpressure_roles_and_shapes(self):
        t = txn(1, 0, 4)
        self.sender.send(FakeTensor([0] * 3, (1, 3)), t, 0, 0)
        self.sender.send(FakeTensor([0] * 3, (1, 3)), t, 0, 1)
        with self.assertRaises(LaneCommError):
            self.sender.send(FakeTensor([0] * 3, (1, 3)), t, 0, 2)
        with self.assertRaises(LaneCommError):
            self.sender.post_receive(t, 0, 0, 1)
        with self.assertRaises(LaneCommError):
            self.receiver.send(FakeTensor([0] * 3, (1, 3)), t, 0, 0)
        with self.assertRaises(LaneCommError):
            self.sender.send(FakeTensor([0] * 15, (5, 3)), txn(1, 0, 8), 0, 3)
        with self.assertRaises(LaneCommError):
            self.receiver.post_receive(t, 0, 0, 5)
        self.assertEqual(self.sender.describe()['free_slots'], 0)
        self.assertEqual(self.sender.bytes_owned(), 2 * (4 * 3 * 2 + 8 * 8))

    def test_warm_round_trip(self):
        self.assertTrue(self.sender.warm())
        self.assertTrue(self.receiver.warm())
        self.assertTrue(self.receiver.warmed)
        self.assertEqual(self.wire.messages, [])
        with self.assertRaises(LaneCommError):
            PrefillPPChannel(FakePyNccl(0, 2, self.wire, disabled=True), 'cuda:0', self.stream0,
                             max_tokens=4, hidden_width=3, dtype='bfloat16', is_sender=True,
                             peer_rank=1, torch_module=self.torch)


class PrefillCommunicatorsTests(unittest.TestCase):
    def setUp(self):
        FakeEvent.pending = []
        self.torch = fake_torch()
        self.wire = FakeWire()

    def groups(self, pp_rank):
        tp = types.SimpleNamespace(cpu_group='tp-cpu', world_size=2)
        pp = types.SimpleNamespace(cpu_group='pp-cpu', world_size=2, rank_in_group=pp_rank,
                                   is_first_rank=pp_rank == 0, is_last_rank=pp_rank == 1)
        return tp, pp

    def test_build_assigns_roles_and_warms_in_order(self):
        made = []
        wire = self.wire

        def pynccl_class(group, device):
            comm = FakePyNccl(0 if group == 'tp-cpu' else len(made), 2, wire)
            made.append(group)
            return comm

        tp, pp = self.groups(0)
        stream = FakeStream(20)
        sender = PrefillCommunicators.build(tp, pp, 'cuda:0', stream, max_tokens=4,
                                            hidden_width=3, dtype='bfloat16',
                                            pynccl_class=pynccl_class, torch_module=self.torch)
        self.assertEqual(made, ['tp-cpu', 'pp-cpu'], 'TP communicator created before PP')
        self.assertTrue(sender.pp.is_sender)
        self.assertEqual(sender.pp.peer_rank, 1)
        tp, pp = self.groups(1)
        receiver = PrefillCommunicators.build(tp, pp, 'cuda:1', FakeStream(21), max_tokens=4,
                                              hidden_width=3, dtype='bfloat16',
                                              pynccl_class=pynccl_class, torch_module=self.torch)
        self.assertFalse(receiver.pp.is_sender)
        self.assertEqual(receiver.pp.peer_rank, 0)
        self.assertTrue(sender.warm(stream))
        self.assertTrue(receiver.warm(FakeStream(21)))
        self.assertEqual(sender.tp.pynccl_comm.calls[0][0], 'all_reduce')
        self.assertEqual(sender.pp.pynccl.calls[0][0], 'send')
        self.assertTrue(receiver.describe()['pp']['role'] == 'receiver')
        self.assertTrue(sender.describe()['warmed'])

    def test_single_stage_has_no_pp_channel(self):
        tp = types.SimpleNamespace(cpu_group='tp-cpu', world_size=2)
        pp = types.SimpleNamespace(cpu_group='pp-cpu', world_size=1)
        comms = PrefillCommunicators.build(tp, pp, 'cuda:0', FakeStream(1), max_tokens=4,
                                           hidden_width=3, dtype='bfloat16',
                                           pynccl_class=lambda group, device: FakePyNccl(0, 2),
                                           torch_module=self.torch)
        self.assertIsNone(comms.pp)
        self.assertIsNone(comms.describe()['pp'])


if __name__ == '__main__':
    unittest.main()
