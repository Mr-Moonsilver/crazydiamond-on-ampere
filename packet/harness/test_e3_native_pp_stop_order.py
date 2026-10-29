"""CPU counterexample: unmatched PP receive must not complete on cancellation.

Runtime driver/bridge/dispatcher/QSA methods are unchanged. Model, tensor,
allocator, transport and event leaves are synthetic (not box evidence).
The PP leaf models the required send/receive matching and stream FIFO edge,
not NCCL execution. Earlier fixtures completed this edge unconditionally.
"""
import json
from pathlib import Path
import sys
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'harness'), str(ROOT / 'patches/s5_engine')]
import test_e3_native_schedule_profile as p


class ClosingServer(p.ProfileServer):
    def clear_native(self, transaction):
        # Match NativeOperationServer.clear_native's handoff retirement. The
        # original CreditServer only logged 'clear' and left its owner live.
        result = super().clear_native(transaction)
        self.handoff = None
        self.operation = None
        return result


class MatchedPPNetwork(p.ProfileNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for rank, stack in self.qsa.items():
            worker = self.worker(rank)
            server = ClosingServer(stack, worker.state)
            stack.server = worker.server = stack.dispatcher.server = stack.bridge.server = server
        for rank in (2, 3):
            self.worker(rank).channel.ready = False

    def complete_everything(self):
        for rank in (0, 1):
            self.worker(rank).bus.complete_all(True)
        for stack in self.qsa.values():
            stack.complete_all()
        for rank in (2, 3):
            receiver = self.worker(rank)
            sender = self.worker(rank - 2)
            unmatched = []
            for ticket in receiver.channel.tickets:
                if ticket.kind != 'recv' or ticket.state == 'retired':
                    continue
                matched = any(
                    send.kind == 'send' and send.transaction == ticket.transaction
                    and send.stage == ticket.stage and send.seq == ticket.seq
                    for send in sender.channel.tickets)
                if not matched:
                    unmatched.append(ticket)
            receiver.channel.ready = bool(receiver.channel.tickets) and not unmatched
            # Pre-receive events may finish; receive and later events cannot
            # pass the unmatched NCCL receive on the same prefill stream.
            first_blocked = min((receiver.bus.events.index(t.event)
                                 for t in unmatched), default=len(receiver.bus.events))
            for index, event in enumerate(receiver.bus.events):
                event.complete = index < first_blocked


class PPStopOrder(p.c.CreditHarness, unittest.TestCase):
    legacy_schedule_source = False
    query_rows = 64
    stack_type = p.HistoryStack
    selection_counts = (4097, 4097)
    receipts = []

    @classmethod
    def setUpClass(cls):
        if not p.c.REFERENCE.is_dir():
            raise RuntimeError('reference required; counterexample NOT_RUN')
        super().setUpClass()

    def reproduce(self, kind):
        self.selection_counts = (4097, 97153) if kind == 'mismatch' else (4097, 4097)
        network = MatchedPPNetwork(self)
        requested_stop = False
        try:
            for tick in range(500):
                network.complete_everything()
                if kind == 'abort' and network.driver._profile_query is not None and not requested_stop:
                    requested_stop = True
                    network.driver.stop('abort')
                network.driver.advance(deliver_outputs=False)
            self.fail('counterexample did not reach bounded drain refusal')
        except p.ed.DriverError as exc:
            self.assertEqual(str(exc), 'cancellation did not close worker ownership within the bound')
            frames = traceback.extract_tb(exc.__traceback__)
            receipt = dict(
                variant=kind, classification='ENGINE', counterexample_confirmed=True,
                exception=dict(type=type(exc).__name__, message=str(exc), frames=[
                    dict(file=Path(f.filename).name, line=f.lineno, function=f.name)
                    for f in frames]), ticks=tick + 1, stop_rounds=network.driver._stop_ticks,
                sender_sends=[network.worker(r).channel.sends for r in (0, 1)],
                receiver_tickets=[len(network.worker(r).channel.tickets) for r in (2, 3)],
                receive_events_ready=[network.worker(r).dispatcher.pp_ticket.event.query()
                                      for r in (2, 3)],
                pending_receiver_ops=[list(network.worker(r).dispatcher.poll()['pending_op_seqs'])
                                      for r in (2, 3)],
                closed_ranks=sorted(network.driver._closed_rows()),
                mismatch_count=network.driver._profile_mismatches,
                timeout_count=network.driver._profile_timeouts,
                named_refusal_waiting=network.driver._profile_failure,
                native_closure=network.driver._last_native_closure,
                leases_clean=network.controller.leases.shutdown()['clean'],
                runtime_methods_patched=False, gpu_execution=False, box_evidence=False)
            self.assertEqual(receipt['sender_sends'], [0, 0])
            self.assertEqual(receipt['receiver_tickets'], [1, 1])
            self.assertEqual(receipt['receive_events_ready'], [False, False])
            self.assertTrue(all(receipt['pending_receiver_ops']))
            self.assertEqual(receipt['closed_ranks'], [0, 1])
            self.assertFalse(receipt['leases_clean'])
            self.assertIsNone(receipt['native_closure'])
            self.assertEqual(receipt['mismatch_count'], int(kind == 'mismatch'))
            self.receipts.append(receipt)
            return receipt

    def test_stage0_profile_mismatch_cannot_drain_unmatched_receive(self):
        self.reproduce('mismatch')

    def test_stage0_client_abort_cannot_drain_unmatched_receive(self):
        self.reproduce('abort')


if __name__ == '__main__':
    output = None
    if '--receipt' in sys.argv:
        index = sys.argv.index('--receipt')
        output = Path(sys.argv[index + 1])
        del sys.argv[index:index + 2]
    result = unittest.main(exit=False)
    if output is not None:
        with output.open('x') as stream:
            json.dump(dict(schema='native18-pp-stop-counterexample/1',
                           tests_run=result.result.testsRun,
                           skips=len(result.result.skipped),
                           tests_ok=result.result.wasSuccessful(),
                           cases=PPStopOrder.receipts), stream, indent=2, default=str)
            stream.write('\n')
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
