"""native35 metadata-generation lease through the real LaneDispatcher (CPU).

Composes the real ``LaneDispatcher`` (test_e3_lane_dispatch fakes for lane,
channel, operation, driver, server) with the real ``LaneChunkFactory`` and
``LaneChunk`` (test_e3_lane_chunk fake runner, torch and pinned ports). The
only fixture leaves are those two existing suites' fakes; the dispatcher,
factory, chunk admission and the native builder view are the shipped code.

Proves, for the lead's gate items 4 and 5:
* normal drain/release clears ``_active_metadata_chunk`` and the next
  transaction admits with no spurious refusal; the private builder view is
  allocated once (same objects, generation +1);
* cancel -> drain -> release clears the lease the same way and reuse admits;
* premature reuse refuses at every entry (factory.admit, dispatcher.open,
  a second prepare);
* a quarantine fails closed with explicit messages: release, close_request,
  factory.admit and factory.release all refuse and the lease stays held;
* the receiver stage (PP1) releases the lease on the same path.
"""
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'patches/s5_engine'))
sys.path.insert(0, str(HERE))

import test_e3_lane_chunk as chunk_fixture  # noqa: E402
import test_e3_lane_dispatch as dispatch_fixture  # noqa: E402
from e3_types import E3Pending, TransactionId  # noqa: E402
from lane_chunk import LaneChunkError, LaneChunkFactory  # noqa: E402
from lane_dispatch import LaneDispatchError, LaneDispatcher, SegmentCommand  # noqa: E402

REQUEST = 'test-request'


def txn(chunk_start, chunk_end, seq):
    return TransactionId(1, REQUEST, 0, chunk_start, chunk_end, seq)


FIRST = txn(0, 8, 1)
SECOND = txn(8, 11, 2)


def first_admission():
    return chunk_fixture.admission(request_uid=REQUEST)


def second_admission():
    return chunk_fixture.admission(request_uid=REQUEST, chunk_start=8, chunk_end=11, txn_seq=2,
                                   input_ids=(8, 9, 10),
                                   block_tables=((5, 6, 7, 8), (9, 10, 11), ()))


def cmd(transaction, kind, op_seq, stage=0, **kw):
    if kind in ('pp_send', 'pp_receive') and 'seq' not in kw:
        kw['seq'] = 0
    return SegmentCommand(transaction, stage, kind, op_seq, **kw)


class Composition:
    """Real dispatcher over a real factory; everything below them is a fake."""

    def __init__(self, is_sender=True):
        chunk_fixture.FakeMambaContext.instances = []
        self.calls = []
        self.chunk_torch = chunk_fixture.fake_torch()
        self.runner = chunk_fixture.fake_runner(self.chunk_torch, self.calls)
        self.ports = chunk_fixture.fake_ports(self.calls)
        self.factory = LaneChunkFactory(self.runner, ports=self.ports, torch_module=self.chunk_torch)
        self.dispatch_torch = dispatch_fixture.fake_torch()
        base, self.lane, self.channel, self.server, _, self.operation = dispatch_fixture.make(
            is_sender, self.dispatch_torch)
        self.dispatcher = LaneDispatcher(self.runner, base.native, self.factory, server=self.server,
                                         driver_factory=lambda op, mod, fc: dispatch_fixture.FakeDriver(),
                                         torch_module=self.dispatch_torch, clock=lambda: 0)

    def open(self, transaction, admission, operation=None):
        operation = dispatch_fixture.FakeOperation() if operation is None else operation
        return self.dispatcher.open(transaction, admission, operation)

    def run_stage0(self, transaction, ops=3):
        d = self.dispatcher
        d.submit(cmd(transaction, 'prepare', 1))
        d.submit(cmd(transaction, 'prefix', 2))
        op = 3
        for idx in range(ops):
            d.submit(cmd(transaction, 'layer', op, layer_start=idx, layer_end=idx + 1))
            op += 1
        d.submit(cmd(transaction, 'suffix', op))
        op += 1
        d.submit(cmd(transaction, 'pp_send', op, seq=transaction.txn_seq))
        d.accept_pp_consumer_proof(dispatch_fixture.proof_for(d.pp_send_ticket))
        op += 1
        d.submit(cmd(transaction, 'state_finalize', op))
        return op


class NormalFlow(unittest.TestCase):
    def test_release_clears_the_lease_and_the_next_chunk_admits(self):
        c = Composition()
        f, d = c.factory, c.dispatcher
        view = f.native_attn_groups
        private_builder = view[0][0].get_metadata_builder(0)
        self.assertIsNone(f._active_metadata_chunk)
        c.open(FIRST, first_admission())
        c.run_stage0(FIRST)
        chunk = d.chunk
        self.assertIs(f._active_metadata_chunk, chunk, 'prepare takes the single lease')
        self.assertEqual(chunk.native_metadata_generation, 1)
        self.assertTrue(f.inventory()['metadata_generation_in_flight'])
        self.assertTrue(d.drain())
        self.assertTrue(d.release())
        self.assertIsNone(f._active_metadata_chunk, 'release after drain frees the lease')
        self.assertIsNone(d.transaction)
        self.assertFalse(f.inventory()['metadata_generation_in_flight'])
        # Second chunk of the same request: no spurious refusal, same storage.
        c.open(SECOND, second_admission())
        ack = d.submit(cmd(SECOND, 'prepare', 1))
        self.assertEqual(ack['result']['prepared'], True)
        self.assertIs(f._active_metadata_chunk, d.chunk)
        self.assertEqual(d.chunk.native_metadata_generation, 2)
        self.assertIs(f.native_attn_groups, view, 'private view allocated once')
        self.assertIs(view[0][0].get_metadata_builder(0), private_builder)
        self.assertEqual(f.inventory()['chunks'], 2)
        # The stock builder never saw the native-only field.
        self.assertIsNone(c.runner.builder.mamba_aligned_state_indices)
        self.assertEqual(private_builder.mamba_aligned_state_indices, 'idx-g0')
        c.run_stage0(SECOND)
        self.assertTrue(d.drain())
        self.assertTrue(d.release())
        self.assertIsNone(f._active_metadata_chunk)
        self.assertEqual(d.close_request(), REQUEST)
        self.assertEqual(f.inventory()['metadata_generation'], 2)

    def test_receiver_stage_releases_the_lease_on_the_same_path(self):
        c = Composition(is_sender=False)
        f, d = c.factory, c.dispatcher
        c.open(FIRST, first_admission())
        d.submit(cmd(FIRST, 'prepare', 1, stage=1))
        self.assertIs(f._active_metadata_chunk, d.chunk)
        d.submit(cmd(FIRST, 'pp_receive', 2, stage=1, seq=3))
        c.channel.ready = True
        d.submit(cmd(FIRST, 'prefix', 3, stage=1))
        for op, idx in ((4, 0), (5, 1), (6, 2)):
            d.submit(cmd(FIRST, 'layer', op, stage=1, layer_start=idx, layer_end=idx + 1))
        d.submit(cmd(FIRST, 'suffix', 7, stage=1))
        d.submit(cmd(FIRST, 'state_finalize', 8, stage=1))
        self.assertTrue(d.drain())
        self.assertTrue(d.release())
        self.assertIsNone(f._active_metadata_chunk)
        c.open(SECOND, second_admission())
        d.submit(cmd(SECOND, 'prepare', 1, stage=1))
        self.assertEqual(d.chunk.native_metadata_generation, 2)


class CancelFlow(unittest.TestCase):
    def test_cancel_drain_release_then_reuse(self):
        c = Composition()
        f, d = c.factory, c.dispatcher
        c.open(FIRST, first_admission())
        d.submit(cmd(FIRST, 'prepare', 1))
        d.submit(cmd(FIRST, 'prefix', 2))
        chunk = d.chunk
        self.assertEqual(d.cancel('disconnect'), 'disconnect')
        self.assertIs(f._active_metadata_chunk, chunk, 'cancel keeps the lease until drain')
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd(FIRST, 'layer', 3, layer_start=0, layer_end=1))
        self.assertTrue(d.drain())
        self.assertTrue(d.release())
        self.assertIsNone(f._active_metadata_chunk, 'cancel -> drain -> release frees the lease')
        self.assertEqual(c.server.calls[-1], ('abort',))
        # Reuse: the next admitted chunk of the request is not refused.
        c.open(SECOND, second_admission())
        ack = d.submit(cmd(SECOND, 'prepare', 1))
        self.assertTrue(ack['result']['prepared'])
        self.assertEqual(d.chunk.native_metadata_generation, 2)
        self.assertEqual(d.cancel('abort'), 'abort')
        self.assertTrue(d.drain())
        self.assertTrue(d.release())
        self.assertIsNone(f._active_metadata_chunk)
        self.assertEqual(d.close_request(), REQUEST)
        self.assertIsNone(f.request.req_id)

    def test_cancel_with_outstanding_work_holds_the_lease_until_queried(self):
        c = Composition()
        f, d = c.factory, c.dispatcher
        c.open(FIRST, first_admission())
        d.submit(cmd(FIRST, 'prepare', 1))
        d.submit(cmd(FIRST, 'prefix', 2))
        d.events[-1].event.complete = False
        d.cancel('stop')
        self.assertFalse(d.drain())
        with self.assertRaises(E3Pending):
            d.release()
        self.assertIs(f._active_metadata_chunk, d.chunk, 'lease held while work is outstanding')
        d.events[-1].event.complete = True
        self.assertTrue(d.drain())
        self.assertTrue(d.release())
        self.assertIsNone(f._active_metadata_chunk)


class PrematureReuse(unittest.TestCase):
    def test_every_entry_refuses_while_the_lease_is_held(self):
        c = Composition()
        f, d = c.factory, c.dispatcher
        c.open(FIRST, first_admission())
        d.submit(cmd(FIRST, 'prepare', 1))
        with self.assertRaisesRegex(LaneChunkError, 'live consumers'):
            f.admit(second_admission())
        with self.assertRaisesRegex(LaneDispatchError, 'already open'):
            c.open(SECOND, second_admission())
        with self.assertRaisesRegex(LaneDispatchError, 'already prepared'):
            d.submit(cmd(FIRST, 'prepare', 2))
        with self.assertRaisesRegex(LaneChunkError, 'live metadata consumers'):
            f.release()
        with self.assertRaisesRegex(LaneDispatchError, 'transaction is open'):
            d.close_request()
        with self.assertRaisesRegex(LaneChunkError, 'not complete'):
            f.complete_chunk(d.chunk, consumers_complete=False)
        self.assertIs(f._active_metadata_chunk, d.chunk)


class Quarantine(unittest.TestCase):
    def test_quarantine_fails_closed_with_explicit_messages(self):
        c = Composition()
        f, d = c.factory, c.dispatcher
        c.open(FIRST, first_admission())
        d.submit(cmd(FIRST, 'prepare', 1))
        d.submit(cmd(FIRST, 'prefix', 2))
        for op, idx in ((3, 0), (4, 1), (5, 2)):
            d.submit(cmd(FIRST, 'layer', op, layer_start=idx, layer_end=idx + 1))
        d.submit(cmd(FIRST, 'suffix', 6))
        c.channel.send_error = RuntimeError('enqueue failed')
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd(FIRST, 'pp_send', 7, seq=1))
        self.assertIn('pp_send', d.quarantined)
        chunk = d.chunk
        self.assertIs(f._active_metadata_chunk, chunk, 'quarantine keeps the lease')
        self.assertFalse(d.drain())
        with self.assertRaisesRegex(LaneDispatchError, 'dispatcher is quarantined'):
            d.release()
        with self.assertRaisesRegex(LaneDispatchError, 'transaction is open'):
            d.close_request()
        with self.assertRaisesRegex(LaneChunkError, 'live consumers'):
            f.admit(second_admission())
        with self.assertRaisesRegex(LaneChunkError, 'live metadata consumers'):
            f.release()
        with self.assertRaisesRegex(LaneDispatchError, 'already open'):
            c.open(SECOND, second_admission())
        inventory = f.inventory()
        self.assertTrue(inventory['metadata_generation_in_flight'])
        self.assertEqual(inventory['metadata_generation'], 1)
        # Even a correct completion claim from outside the dispatcher cannot
        # revive a quarantined transaction: the dispatcher never calls it.
        self.assertIsNotNone(d.quarantined)

    def test_refused_admission_takes_no_lease_and_quarantines(self):
        c = Composition()
        f, d = c.factory, c.dispatcher
        c.open(FIRST, first_admission())
        c.run_stage0(FIRST)
        self.assertTrue(d.drain())
        self.assertTrue(d.release())
        # A foreign request on the registered row: admit refuses at the
        # request update, before the lease is taken.
        bad = chunk_fixture.admission(request_uid='other')
        c.open(bad.transaction, bad)
        with self.assertRaises(LaneDispatchError):
            d.submit(cmd(bad.transaction, 'prepare', 1))
        self.assertIsNone(f._active_metadata_chunk, 'a refused admission holds no lease')
        self.assertEqual(f.inventory()['metadata_generation'], 1)
        self.assertIsNotNone(d.quarantined)
        with self.assertRaisesRegex(LaneDispatchError, 'dispatcher is quarantined'):
            d.release()
        with self.assertRaisesRegex(LaneDispatchError, 'transaction is open'):
            d.close_request()


if __name__ == '__main__':
    unittest.main()
