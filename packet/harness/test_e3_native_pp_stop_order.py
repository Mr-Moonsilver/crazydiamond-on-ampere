"""Native19 PP stop regression; ae5eb56 retains the native18 counterexamples.

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
        network = getattr(self, 'network_factory', MatchedPPNetwork)(self)
        requested_stop = False
        for tick in range(500):
            network.complete_everything()
            if kind == 'abort' and network.driver._profile_query is not None and not requested_stop:
                requested_stop = True
                network.driver.stop('abort')
            network.driver.advance(deliver_outputs=False)
            if (network.driver.tx is None and not network.controller.active
                    and network.driver._return_pending is None
                    and network.driver.stop_kind is None and not network.driver._inflight):
                break
        else:
            self.fail('stop did not finish its typed all-rank drain')
        receipt = dict(variant=kind, verdict='PASS', ticks=tick + 1,
            sender_sends=[network.worker(r).channel.sends for r in (0, 1)],
            receiver_tickets=[len(network.worker(r).channel.tickets) for r in (2, 3)],
            mismatch_count=network.driver._profile_mismatches,
            timeout_count=network.driver._profile_timeouts,
            named_refusal_waiting=network.driver._profile_failure,
            native_closure=network.driver._last_native_closure,
            leases_clean=network.controller.leases.shutdown()['clean'],
            credits_clean=not network.transport._calls,
            runtime_methods_patched=False, gpu_execution=False, box_evidence=False)
        self.assertEqual(receipt['sender_sends'], [0, 0])
        self.assertEqual(receipt['receiver_tickets'], [0, 0])
        self.assertTrue(receipt['leases_clean'])
        self.assertTrue(receipt['credits_clean'])
        self.assertTrue(receipt['native_closure']['complete'])
        self.assertFalse(network.driver._split_ready)
        self.assertEqual(receipt['mismatch_count'], int(kind == 'mismatch'))
        self.receipts.append(receipt)
        return network, receipt

    def test_stage0_profile_mismatch_drains_without_posting_receive(self):
        self.reproduce('mismatch')

    def test_stage0_client_abort_drains_without_posting_receive(self):
        self.reproduce('abort')


class TransferStopTests(unittest.TestCase):
    def drain(self, network):
        network.driver.stop('abort')
        for _ in range(300):
            network.complete_everything()
            network.driver.pump_stop(1)
            if not network.driver.active and not network.driver._inflight:
                break
        self.assertIsNone(network.driver.error)
        self.assertFalse(network.driver.active)
        self.assertFalse(network.controller.leases.held_blocks())
        self.assertFalse(network.transport._calls)
        self.assertTrue(network.driver._last_native_closure['complete'])
        self.assertFalse(network.driver._split_ready)

    def test_abort_at_each_transfer_command_edge(self):
        for edge in ('before_send', 'send_reply_inflight', 'receive_reply_inflight', 'after_receive'):
            with self.subTest(edge=edge):
                network = p.r4.Stack()
                for _ in range(150):
                    d = network.driver
                    obligation = d._pp_submission
                    if edge == 'before_send':
                        cmd = getattr(d, '_command', None)
                        ready = bool(cmd and cmd.kind is p.w.SegmentKind.SUFFIX and cmd.stage == 0 and d._inflight)
                    elif edge == 'send_reply_inflight':
                        ready = obligation is not None and obligation.receive is None and bool(d._inflight)
                    elif edge == 'receive_reply_inflight':
                        ready = obligation is not None and obligation.receive is not None and bool(d._inflight)
                    else:
                        ready = obligation is not None and obligation.receive_accepted == {2, 3}
                    if ready:
                        break
                    network.complete_everything()
                    d.advance()
                else:
                    self.fail('transfer edge not reached')
                self.drain(network)
                expected = 0 if edge == 'before_send' else 1
                self.assertEqual([network.worker(r).channel.sends for r in (0, 1)], [expected] * 2)
                self.assertEqual([len(network.worker(r).channel.tickets) for r in (2, 3)], [expected] * 2)
                self.assertTrue(all(t.state == 'retired' for w in network.workers.values() for t in w.channel.tickets))

    def test_receive_leaf_never_completes_without_matching_transfer(self):
        network = p.r4.Stack()
        tx = network.record.transaction
        receiver = network.worker(2)
        ticket = receiver.channel.post_receive(tx, 0, 19, 4)
        later = receiver.bus.event()
        receiver.bus.complete_all(True)
        self.assertFalse(ticket.event.query())
        self.assertFalse(later.query())
        network.worker(0).channel.send('fixture-payload', tx, 0, 18)
        self.assertFalse(ticket.event.query())
        network.worker(0).channel.send('fixture-payload', tx, 0, 19)
        self.assertTrue(ticket.event.query())
        self.assertTrue(later.query())

    def test_stop_finishes_partial_send_and_receive_acceptance_without_duplicate_enqueue(self):
        # Reply-endpoint admission leaf: a rank can answer typed pending before
        # its real bridge is called. No runtime method is replaced. This is a
        # protocol stress input, not a claim that NCCL itself returns pending.
        class DeferredEndpoint:
            def __init__(self, worker, kind):
                self.worker, self.kind, self.defer = worker, kind, True

            def __getattr__(self, name):
                return getattr(self.worker, name)

            def flash_s5_submit(self, command):
                if self.defer and command.kind is self.kind and self.worker.rank in command.ranks:
                    return p.w.SegmentAcceptance(command.transaction, self.worker.rank,
                        command.kind, command.stage, command.op_seq, pending=True)
                return self.worker.flash_s5_submit(command)

        for kind, rank in ((p.w.SegmentKind.PP_SEND, 1), (p.w.SegmentKind.PP_RECEIVE, 3)):
            with self.subTest(kind=kind):
                network = p.r4.Stack()
                endpoint = DeferredEndpoint(network.worker(rank), kind)
                network.transport.network.workers = dict(network.workers, **{})
                network.transport.network.workers[rank] = endpoint
                for _ in range(100):
                    network.complete_everything()
                    network.driver.advance()
                    if network.driver._retry is not None and network.driver._retry.kind is kind:
                        break
                else:
                    self.fail('partial acceptance not reached')
                obligation = network.driver._pp_submission
                self.assertEqual(obligation.send_accepted if rank == 1 else obligation.receive_accepted,
                                 {0} if rank == 1 else {2})
                endpoint.defer = False
                self.drain(network)
                self.assertEqual([network.worker(r).channel.sends for r in (0, 1)], [1, 1])
                self.assertEqual([len(network.worker(r).channel.tickets) for r in (2, 3)], [1, 1])
                for worker in network.workers.values():
                    self.assertTrue(all(t.state == 'retired' for t in worker.channel.tickets))


if __name__ == '__main__':
    output = None
    if '--receipt' in sys.argv:
        index = sys.argv.index('--receipt')
        output = Path(sys.argv[index + 1])
        del sys.argv[index:index + 2]
    result = unittest.main(exit=False)
    if output is not None:
        with output.open('x') as stream:
            json.dump(dict(schema='native19-pp-stop-regression/1',
                           tests_run=result.result.testsRun,
                           skips=len(result.result.skipped),
                           tests_ok=result.result.wasSuccessful(),
                           cases=PPStopOrder.receipts), stream, indent=2, default=str)
            stream.write('\n')
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
