"""C1 bind-gate witness: can stage 0 compute chunk n+1 while stage 1 still computes n?

The consolidated review (2 Nov 2026, section 2) asks for this connected test:

    the receiver's old final-layer / STATE_FINALIZE event stays genuinely
    incomplete, while the preceding PP receive and stage-0(n)'s local release are
    complete.  Can stage 0 of n+1 submit PREPARE/PREFIX and real layer work?

Everything is the C1 ring network of ``test_e3_stage_overlap`` (real
``OverlapSegmentDriver``, ``StagedAdmissionController``, ``LaneDispatcher``
completion semantics, ``NativeWorkerBridge``, the C0 stage wire, ``ReplyTransport``
and the generated worker-loop hooks), with its declared leaves.  This file adds
exactly one leaf, ``held-receiver-device-events`` (declared in
``c1_rebind_fixture_leaves.json``):

* On the receiving stage's ranks, the device events that ``LaneDispatcher._record``
  records for chunk n's last LAYER, its SUFFIX and its STATE_FINALIZE, and the
  ``state_event`` that ``_do_state_finalize`` records, report not complete on a
  host query until the test lets them go.  That is what ``cudaEventQuery`` answers
  while those kernels are still running on the prefill stream.
* Nothing logical is delayed: every RPC, reply, RELEASE and receipt flows at the
  network's normal rate.  The receiver's ``pp_consumer_complete()`` stays False
  for the real reason (its consumer event, the STATE_FINALIZE record's event, is
  incomplete), so its report carries no PP_CONSUMER proof.

The verdict is read from the stage-0 ranks' own device ledgers (the real
dispatcher handlers that ran, in order): ``OVERLAPS`` when any PREPARE, PREFIX or
LAYER of n+1 ran on a stage-0 rank while the hold was in force, else ``BLOCKED``.
CPU only; no pass or fail here is native evidence.
"""
import json
import os
import unittest

from test_e3_stage_overlap import OverlapBase, serial_stack

COMPUTE_KINDS = ('prepare', 'prefix', 'layer', 'suffix')
SETTLE_TICKS = 300


class ReceiverHold:
    """Leaf ``held-receiver-device-events``: stage 1's late events of chunk n stay incomplete."""

    def __init__(self, network, txn_seq):
        self.network = network
        self.txn_seq = txn_seq
        self.active = True
        self.events = {}
        driver = network.driver
        self.ranks = tuple(driver.stage_ranks[1])
        self.last_layer = driver.layer_count - 1
        for rank in self.ranks:
            self._install(rank)

    def applies(self, command):
        if command.transaction.txn_seq != self.txn_seq:
            return False
        if command.kind == 'layer':
            return command.layer_start == self.last_layer
        return command.kind in ('suffix', 'state_finalize')

    def _install(self, rank):
        worker = self.network.workers[rank]
        dispatcher, bus = worker.dispatcher, worker.bus
        held = self.events.setdefault(rank, [])
        record = dispatcher._record

        def held_record(command, record=record):
            out = record(command)
            if self.active and self.applies(command) and out.event is not None:
                held.append(out.event)
            return out
        dispatcher._record = held_record
        finalize = dispatcher._do_state_finalize

        def held_finalize(command, finalize=finalize, dispatcher=dispatcher):
            result = finalize(command)
            if self.active and self.applies(command) and dispatcher.state_event is not None:
                held.append(dispatcher.state_event)
            return result
        dispatcher._do_state_finalize = held_finalize
        complete_all = bus.complete_all

        def complete_all_but_held(complete=True, complete_all=complete_all):
            complete_all(complete)
            if self.active:
                for event in held:
                    event.complete = False
        bus.complete_all = complete_all_but_held

    def release(self):
        self.active = False


def stage_ops(network, ranks, txn_seq):
    """(rank, kind) of every device op the given ranks ran for one transaction, in order."""
    return [(rank, row[1]) for rank in ranks for row in network.ledgers[rank]
            if row[0] == 'op' and row[2] == txn_seq]


class Witness(OverlapBase):
    """The review's smallest next test, connected; see the module docstring."""

    def observe(self, configure=None):
        network = self.network()
        if configure is not None:
            configure(network)
        driver, controller = network.driver, network.controller
        n = controller.parked.transaction.txn_seq
        hold = ReceiverHold(network, n)
        receivers = hold.ranks
        senders = tuple(driver.stage_ranks[0])

        def finalize_accepted():
            return all(any(cmd.kind == 'state_finalize' and cmd.transaction.txn_seq == n
                           for cmd, _ack in network.workers[r].dispatcher.accepted.values())
                       for r in receivers)

        def successor_open():
            frames = driver._frames
            return (len(frames) == 2 and frames[0].chunk['tx'].txn_seq == n
                    and 0 in frames[1].opened)
        self.reach(network, lambda: finalize_accepted() and successor_open(), limit=2000)
        n1 = driver._frames[1].chunk['tx'].txn_seq
        for _ in range(SETTLE_TICKS):
            network.tick()
        older, newer = driver._frames[0], driver._frames[1]
        receiver_state = {}
        for rank in receivers:
            dispatcher = network.workers[rank].dispatcher
            ticket = dispatcher.pp_ticket
            receiver_state[rank] = dict(
                transaction=None if dispatcher.transaction is None else dispatcher.transaction.txn_seq,
                pp_receive_state=getattr(ticket, 'state', None),
                consumer_event_recorded=getattr(ticket, 'consumer_event', None) is not None,
                pp_consumer_complete=dispatcher.pp_consumer_complete(),
                held_events=len(hold.events[rank]),
                held_incomplete=sum(1 for e in hold.events[rank] if not e.query()))
        n1_ops = stage_ops(network, senders, n1)
        observation = dict(
            older_txn=n, successor_txn=n1, settle_ticks=SETTLE_TICKS,
            older_released=sorted(older.released), successor_opened=sorted(newer.opened),
            successor_phases=[newer.stage[s]['phase'] for s in (0, 1)],
            successor_early=bool(getattr(newer, 'early', False)),
            older_receipt_acked=sorted(older.chunk['_receipt_acked']),
            older_receipt_inflight=older.chunk['_receipt_inflight'],
            retained_sends={r: len(network.workers[r]._flash_s5_bridge.__dict__.get(
                '_flash_s5_retained_sends', {})) for r in senders},
            receivers=receiver_state,
            successor_stage0_ops=n1_ops,
            successor_stage0_compute=[op for op in n1_ops if op[1] in COMPUTE_KINDS],
            successor_stage0_pp_send=[op for op in n1_ops if op[1] == 'pp_send'],
            older_stage1_ops_after=[op for op in stage_ops(network, receivers, n)])
        observation['verdict'] = 'OVERLAPS' if observation['successor_stage0_compute'] else 'BLOCKED'
        # Preconditions the review names; a witness that misses them proves nothing.
        self.assertIn(0, older.released, observation)
        self.assertNotIn(1, older.released, observation)
        for rank, row in receiver_state.items():
            self.assertEqual(row['transaction'], n, observation)
            self.assertEqual(row['pp_receive_state'], 'ready', observation)
            self.assertTrue(row['consumer_event_recorded'], observation)
            self.assertFalse(row['pp_consumer_complete'], observation)
            self.assertGreater(row['held_incomplete'], 0, observation)
            self.assertEqual(row['held_incomplete'], row['held_events'], observation)
        self.assertEqual(network.worker_errors, [], observation)
        path = os.environ.get('C1_REBIND_WITNESS_OUT')
        if path:
            with open(path, 'a') as out:
                out.write(json.dumps(dict(test=self.id(), **observation), default=str) + '\n')
        # Let stage 1 finish n; the request must then complete and match the serial ledger.
        hold.release()
        network.run_request()
        off = serial_stack()
        self.addCleanup(off.close_network)
        off.run_request()
        for rank in (0, 1, 2, 3):
            self.assertEqual(network.ledgers[rank], off.ledgers[rank], rank)
        self.assertEqual(network.persistent_state(), off.persistent_state())
        return observation

    def test_c1_successor_compute_with_receiver_finalize_incomplete(self):
        """On C1 as published (cf120738) the bind gate holds n+1's compute: BLOCKED."""
        observation = self.observe()
        self.assertEqual(observation['verdict'], 'BLOCKED', observation)
        self.assertEqual(observation['successor_phases'][0], 'bind', observation)


if __name__ == '__main__':
    unittest.main()
