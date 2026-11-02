"""C1 bind-gate witness and the deferred-rebind repair (``FLASH_S5_STAGE_OVERLAP_DEFER_BIND``).

The consolidated review (2 Nov 2026, section 2) asks for this connected test:

    the receiver's old final-layer / STATE_FINALIZE event stays genuinely
    incomplete, while the preceding PP receive and stage-0(n)'s local release are
    complete.  Can stage 0 of n+1 submit PREPARE/PREFIX and real layer work?

Everything is the C1 ring network of ``test_e3_stage_overlap`` (real
``OverlapSegmentDriver``, ``StagedAdmissionController``, ``LaneDispatcher``
completion semantics, ``NativeWorkerBridge``, the C0 stage wire, ``ReplyTransport``
and the generated worker-loop hooks), with its declared leaves.  This file adds
one leaf, ``held-receiver-device-events``, and one observer, ``sender-channel-spy``
(both declared in ``c1_rebind_fixture_leaves.json``):

* On the receiving stage's ranks, the device events that ``LaneDispatcher._record``
  records for a held chunk's last LAYER, its SUFFIX and its STATE_FINALIZE, and the
  ``state_event`` that ``_do_state_finalize`` records, answer not complete on a host
  query until the test lets them go: what ``cudaEventQuery`` answers while those
  kernels still run.  Nothing logical is delayed: every RPC, reply, RELEASE and
  receipt flows at the network's normal rate.  The receiver's
  ``pp_consumer_complete()`` stays False for the real reason, so its report carries
  no PP_CONSUMER proof.
* The spy records, on each stage-0 channel, the binding at every send, bind and
  proof validation, and flags a proof checked under any binding but its own
  ticket's, a bind meeting a live ticket, and a send meeting a live send.

Verdicts are read from the stage-0 ranks' own device ledgers (the real dispatcher
handlers that ran, in order): ``OVERLAPS`` when any PREPARE, PREFIX, LAYER or
SUFFIX of n+1 ran on a stage-0 rank while n's hold was in force, else ``BLOCKED``.
CPU only; no pass or fail here is native evidence.
"""
import json
import os
import random
import unittest

from test_e3_stage_overlap import OverlapBase, serial_stack  # sets the engine path first
from e3_types import E3Pending  # noqa: E402
from lane_comm import PP_SLOTS  # noqa: E402
from stage_net import stage_of_command, w  # noqa: E402
from stage_wire import retained_sends  # noqa: E402

COMPUTE_KINDS = ('prepare', 'prefix', 'layer', 'suffix')
SETTLE_TICKS = 300


class ReceiverHold:
    """Leaf ``held-receiver-device-events``: stage 1's late events of held chunks stay incomplete.

    ``match(txn_seq)`` names the chunks held from the start (default: the first one);
    ``release(txn_seq)`` lets one go, ``release()`` every one.
    """

    def __init__(self, network, match):
        self.network = network
        self.match = match
        self.released = set()
        self.records = []       # (rank, txn_seq, event)
        driver = network.driver
        self.ranks = tuple(driver.stage_ranks[1])
        self.last_layer = driver.layer_count - 1
        self.all_released = False
        for rank in self.ranks:
            self._install(rank)

    def held(self, txn_seq):
        return not self.all_released and txn_seq not in self.released and self.match(txn_seq)

    def held_txns(self):
        return sorted({txn for _rank, txn, _event in self.records if self.held(txn)})

    def applies(self, command):
        if not self.held(command.transaction.txn_seq):
            return False
        if command.kind == 'layer':
            return command.layer_start == self.last_layer
        return command.kind in ('suffix', 'state_finalize')

    def _install(self, rank):
        worker = self.network.workers[rank]
        dispatcher, bus = worker.dispatcher, worker.bus
        record = dispatcher._record

        def held_record(command, record=record):
            out = record(command)
            if self.applies(command) and out.event is not None:
                self.records.append((rank, command.transaction.txn_seq, out.event))
            return out
        dispatcher._record = held_record
        finalize = dispatcher._do_state_finalize

        def held_finalize(command, finalize=finalize, dispatcher=dispatcher):
            result = finalize(command)
            if self.applies(command) and dispatcher.state_event is not None:
                self.records.append((rank, command.transaction.txn_seq, dispatcher.state_event))
            return result
        dispatcher._do_state_finalize = held_finalize
        complete_all = bus.complete_all

        def complete_all_but_held(complete=True, complete_all=complete_all):
            complete_all(complete)
            for held_rank, txn, event in self.records:
                if held_rank == rank and self.held(txn):
                    event.complete = False
        bus.complete_all = complete_all_but_held

    def events(self, rank, txn_seq):
        return [event for r, txn, event in self.records if r == rank and txn == txn_seq]

    def release(self, txn_seq=None):
        if txn_seq is None:
            self.all_released = True
        else:
            self.released.add(txn_seq)


class SenderSpy:
    """Observer ``sender-channel-spy`` on each stage-0 PP channel; changes no answer."""

    def __init__(self, network):
        self.violations = []
        self.sends = []
        self.binds = []
        self.validations = []
        self.max_live_sends = 0
        for rank in network.driver.stage_ranks[0]:
            self._install(rank, network.workers[rank].channel)

    @staticmethod
    def live(channel, kind=None):
        return [t for t in channel.tickets
                if t.state != 'retired' and (kind is None or t.kind == kind)]

    def _install(self, rank, channel):
        bind = channel.bind_consumer

        def spy_bind(lane, route_generation, bind=bind):
            live = self.live(channel)
            self.binds.append((rank, lane, route_generation, len(live)))
            if live:
                self.violations.append(('bind-with-live-ticket', rank,
                                        [t.transaction.txn_seq for t in live]))
            return bind(lane, route_generation)
        channel.bind_consumer = spy_bind
        send = channel.send

        def spy_send(payload, transaction, stage, seq, send=send):
            live = self.live(channel, 'send')
            if live:
                self.violations.append(('send-with-live-send', rank, transaction.txn_seq,
                                        [t.transaction.txn_seq for t in live]))
            ticket = send(payload, transaction, stage, seq)
            ticket.bound_at_send = channel.consumer_binding
            self.sends.append((rank, transaction.txn_seq, channel.consumer_binding))
            self.max_live_sends = max(self.max_live_sends, len(self.live(channel, 'send')))
            return ticket
        channel.send = spy_send
        validate = channel.validate_consumer_proof

        def spy_validate(ticket, receipt, validate=validate):
            own = getattr(ticket, 'bound_at_send', None)
            current = channel.consumer_binding
            self.validations.append((rank, ticket.transaction.txn_seq, own, current))
            if own is None or current != own:
                self.violations.append(('proof-under-foreign-binding', rank,
                                        ticket.transaction.txn_seq, own, current))
            elif receipt.proof.ticket.lane != own[0]:
                self.violations.append(('proof-from-another-lane', rank,
                                        ticket.transaction.txn_seq))
            return validate(ticket, receipt)
        channel.validate_consumer_proof = spy_validate


def stage_ops(network, ranks, txn_seq):
    """(rank, kind) of every device op the given ranks ran for one transaction, in order."""
    return [(rank, row[1]) for rank in ranks for row in network.ledgers[rank]
            if row[0] == 'op' and row[2] == txn_seq]


def successor_of(driver, txn_seq):
    for frame in driver._frames:
        older = getattr(frame, 'successor_of', None)
        if older is not None and older.chunk['tx'].txn_seq == txn_seq:
            return frame
    return None


class RebindBase(OverlapBase):
    defer = False

    def network(self, defer=None, **kwargs):
        network = super().network(**kwargs)
        network.driver.defer_bind = self.defer if defer is None else defer
        return network

    def first_txn(self, network):
        return network.controller.parked.transaction.txn_seq

    def assert_serial(self, network, **kwargs):
        off = serial_stack(**kwargs)
        self.addCleanup(off.close_network)
        off.run_request()
        self.assertEqual(network.worker_errors, [])
        for rank in (0, 1, 2, 3):
            self.assertEqual(network.ledgers[rank], off.ledgers[rank], rank)
        self.assertEqual(network.persistent_state(), off.persistent_state())

    def late_consumer_tick(self, network, hold, patience=3, cap=80):
        """Each held chunk's consumer finishes late: after its successor's stage 0 waits at its
        PP_SEND for ``patience`` ticks (deferred), or after ``cap`` ticks (no such wait)."""
        ages = {}

        def tick():
            driver = network.driver
            for txn in hold.held_txns():
                ages[txn] = ages.get(txn, 0) + 1
                frame = successor_of(driver, txn)
                waiting = (frame is not None and getattr(frame, 'deferred_bind', False)
                           and frame.stage[0]['phase'] == 'bind')
                if (waiting and ages[txn] >= patience) or ages[txn] >= cap:
                    hold.release(txn)
            out = network.tick()
            return out
        return tick

    def held_successor(self, defer=None):
        """Chunk n's late stage-1 events held; the successor's stage 0 open.  Returns the stack."""
        network = self.network(defer=defer)
        n = self.first_txn(network)
        hold = ReceiverHold(network, lambda txn: txn == n)
        spy = SenderSpy(network)
        driver = network.driver
        receivers = hold.ranks

        def finalize_accepted():
            return all(any(cmd.kind == 'state_finalize' and cmd.transaction.txn_seq == n
                           for cmd, _ack in network.workers[r].dispatcher.accepted.values())
                       for r in receivers)
        self.reach(network, lambda: finalize_accepted() and successor_of(driver, n) is not None
                   and 0 in successor_of(driver, n).opened, limit=2000)
        return network, hold, spy, n


class Witness(RebindBase):
    """The review's smallest next test, connected; see the module docstring."""

    def observe(self, defer):
        network, hold, spy, n = self.held_successor(defer=defer)
        driver = network.driver
        receivers, senders = hold.ranks, tuple(driver.stage_ranks[0])
        n1 = successor_of(driver, n).chunk['tx'].txn_seq
        for _ in range(SETTLE_TICKS):
            network.tick()
        older, newer = driver._frames[0], driver._frames[1]
        receiver_state = {}
        for rank in receivers:
            dispatcher = network.workers[rank].dispatcher
            ticket = dispatcher.pp_ticket
            events = hold.events(rank, n)
            receiver_state[rank] = dict(
                transaction=None if dispatcher.transaction is None else dispatcher.transaction.txn_seq,
                pp_receive_state=getattr(ticket, 'state', None),
                consumer_event_recorded=getattr(ticket, 'consumer_event', None) is not None,
                pp_consumer_complete=dispatcher.pp_consumer_complete(),
                held_events=len(events), held_incomplete=sum(1 for e in events if not e.query()))
        n1_ops = stage_ops(network, senders, n1)
        observation = dict(
            defer_bind=driver.defer_bind, older_txn=n, successor_txn=n1, settle_ticks=SETTLE_TICKS,
            older_released=sorted(older.released), successor_opened=sorted(newer.opened),
            successor_phases=[newer.stage[s]['phase'] for s in (0, 1)],
            successor_early=bool(getattr(newer, 'early', False)),
            successor_bindings=sorted(newer.chunk['_bindings']),
            older_receipt_acked=sorted(older.chunk['_receipt_acked']),
            older_receipt_inflight=older.chunk['_receipt_inflight'],
            retained_sends={r: len(retained_sends(network.workers[r]._flash_s5_bridge)) for r in senders},
            live_send_tickets={r: len(SenderSpy.live(network.workers[r].channel, 'send')) for r in senders},
            receivers=receiver_state,
            successor_stage0_ops=n1_ops,
            successor_stage0_compute=[op for op in n1_ops if op[1] in COMPUTE_KINDS],
            successor_stage0_pp_send=[op for op in n1_ops if op[1] == 'pp_send'],
            older_stage1_ops=stage_ops(network, receivers, n),
            defer_counters=dict(driver.defer_counters))
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
        self.assert_serial(network)
        self.assertEqual(spy.violations, [], observation)
        return observation

    def test_c1_successor_compute_with_receiver_finalize_incomplete(self):
        """Flag off (C1 as published, cf120738): the bind gate holds n+1's compute: BLOCKED."""
        observation = self.observe(defer=False)
        self.assertEqual(observation['verdict'], 'BLOCKED', observation)
        self.assertEqual(observation['successor_phases'][0], 'bind', observation)
        self.assertEqual(observation['successor_stage0_ops'], [], observation)

    def test_deferred_bind_successor_computes_with_receiver_finalize_incomplete(self):
        """Flag on: n+1's PREPARE, PREFIX, layers and SUFFIX run on both stage-0 ranks while n's
        late stage-1 events are incomplete; its PP_SEND waits in ``bind`` for n's proof."""
        observation = self.observe(defer=True)
        self.assertEqual(observation['verdict'], 'OVERLAPS', observation)
        for rank in (0, 1):
            kinds = [kind for r, kind in observation['successor_stage0_compute'] if r == rank]
            self.assertEqual(kinds, ['prepare', 'prefix', 'layer', 'layer', 'suffix'], observation)
        self.assertEqual(observation['successor_stage0_pp_send'], [], observation)
        self.assertEqual(observation['successor_phases'][0], 'bind', observation)
        self.assertEqual(observation['successor_bindings'], [], observation)
        self.assertEqual(observation['live_send_tickets'], {0: 1, 1: 1}, observation)


class PeerIdentity(RebindBase):
    """Every PP proof is validated under its own ticket's binding, never a newer one."""
    defer = True

    def check(self, network, spy):
        self.assertEqual(spy.violations, [])
        self.assertTrue(spy.sends)
        validated = {(rank, txn) for rank, txn, _own, _cur in spy.validations}
        self.assertEqual(validated, {(rank, txn) for rank, txn, _b in spy.sends})
        self.assertLessEqual(spy.max_live_sends, 1)
        self.assertTrue(all(live == 0 for _r, _l, _g, live in spy.binds))

    def test_late_consumer_every_chunk(self):
        network = self.network()
        hold = ReceiverHold(network, lambda txn: True)
        spy = SenderSpy(network)
        network.run_request(tick=self.late_consumer_tick(network, hold))
        self.check(network, spy)
        self.assertGreater(network.driver.defer_counters['deferred_binds'], 3)
        # Each bind names the lane the receiver actually opened for that transaction.
        lanes = {}
        for rank, txn, binding in spy.sends:
            lanes.setdefault(binding[0], set()).add(txn)
        self.assertEqual(sum(len(v) for v in lanes.values()), len(spy.sends))
        self.assert_serial(network)

    def test_staggered_receivers(self):
        """Rank 2's consumer finishes long before rank 3's: sender 1 must still wait for its own."""
        network = self.network()
        n = self.first_txn(network)
        hold = ReceiverHold(network, lambda txn: txn == n)
        spy = SenderSpy(network)
        driver = network.driver
        self.reach(network, lambda: successor_of(driver, n) is not None
                   and successor_of(driver, n).stage[0]['phase'] == 'bind'
                   and any(k == 'suffix' for _r, k in stage_ops(network, (0, 1), n + 1)), limit=2000)
        # Release rank 2's events only.
        for rank, txn, event in hold.records:
            if rank == 2:
                event.complete = True
        hold.records = [row for row in hold.records if row[0] != 2]
        for _ in range(60):
            network.tick()
        self.assertEqual(stage_ops(network, (1,), n + 1).count((1, 'pp_send')), 0)
        self.assertNotIn(1, successor_of(driver, n).chunk['_bindings'])
        hold.release()
        network.run_request()
        self.check(network, spy)
        self.assert_serial(network)

    def test_adversarial_stage_timing(self):
        for seed in range(6):
            rng = random.Random(seed)
            network = self.network()
            hold = ReceiverHold(network, lambda txn, rng=rng: rng.random() < 0.6)
            spy = SenderSpy(network)
            late = self.late_consumer_tick(network, hold, patience=rng.randint(1, 6), cap=60)

            def tick(network=network, rng=rng, late=late):
                behind = rng.choice(((), (0, 1), (2, 3), (1,), (3,)))
                network.admit_next()
                network.complete_everything()
                for rank in (0, 1, 2, 3):
                    if rank in behind and rng.random() < 0.8:
                        continue
                    for _ in range(rng.randint(1, 4)):
                        network.step(rank)
                network.settle()
                return late()
            with self.subTest(seed=seed):
                network.run_request(tick=tick)
                self.check(network, spy)
                self.assert_serial(network)


class DelayedConsumption(RebindBase):
    """n's consumer finishes after n+1's compute; n+1's PP_SEND waits for it, bounded, no live slot."""
    defer = True

    def test_successor_waits_at_its_send_until_the_late_proof(self):
        network, hold, spy, n = self.held_successor()
        driver = network.driver
        self.reach(network, lambda: successor_of(driver, n).stage[0]['phase'] == 'bind'
                   and (0, 'suffix') in stage_ops(network, (0,), n + 1)
                   and (1, 'suffix') in stage_ops(network, (1,), n + 1))
        frame = successor_of(driver, n)
        sent = len(network.sent_envelopes)
        stage0 = [stage_of_command(env) for env in network.sent_envelopes]
        for _ in range(100):
            network.tick()
        # Waiting: no send, no binding, n's send still owns its slot, and stage 0 sends nothing.
        self.assertEqual([op for op in stage_ops(network, (0, 1), n + 1) if op[1] == 'pp_send'], [])
        self.assertEqual(frame.chunk['_bindings'], {})
        for rank in (0, 1):
            live = SenderSpy.live(network.workers[rank].channel, 'send')
            self.assertEqual([t.transaction.txn_seq for t in live], [n])
            self.assertLessEqual(len(live), PP_SLOTS - 1)
        later = [stage_of_command(env) for env in network.sent_envelopes[sent:]]
        self.assertEqual([m for m in later if m[2] is not None and set(m[2]) & {0, 1}], [])
        self.assertGreater(len(network.sent_envelopes), sent)   # stage 1's polls continue
        del stage0
        # The proof arrives: the send follows within a bounded number of ticks.
        hold.release()
        ticks = 0
        while not any(op[1] == 'pp_send' for op in stage_ops(network, (0, 1), n + 1)):
            ticks += 1
            self.assertLess(ticks, 40, network.status_text())
            network.tick()
        self.assertEqual(spy.violations, [])
        self.assertEqual(sorted(frame.chunk['_bindings']), [0, 1])
        network.run_request()
        self.assertEqual(spy.violations, [])
        self.assertLessEqual(spy.max_live_sends, 1)
        self.assert_serial(network)

    def test_flag_off_never_computes_before_the_proof(self):
        network, hold, _spy, n = self.held_successor(defer=False)
        for _ in range(100):
            network.tick()
        self.assertEqual(stage_ops(network, (0, 1), n + 1), [])
        hold.release()
        network.run_request()
        self.assert_serial(network)


class Bounded(RebindBase):
    """At most one live send per sender (of PP_SLOTS), two frames, one retained send; no growth."""
    defer = True

    def test_no_growth_over_chunks(self):
        network = self.network(tokens=65)
        hold = ReceiverHold(network, lambda txn: True)
        spy = SenderSpy(network)
        late = self.late_consumer_tick(network, hold)
        driver = network.driver
        samples = []

        def tick():
            out = late()
            samples.append((driver.controller.commits, dict(
                frames=len(driver._frames), routes=len(driver._seq_route), inflight=len(driver._inflight),
                live_sends=max(len(SenderSpy.live(network.workers[r].channel, 'send')) for r in (0, 1)),
                live_recvs=max(len(SenderSpy.live(network.workers[r].channel, 'recv')) for r in (2, 3)),
                retained=max(len(retained_sends(network.workers[r]._flash_s5_bridge)) for r in (0, 1)))))
            return out
        network.run_request(tick=tick)
        self.assertEqual(spy.violations, [])
        caps = dict(frames=2, routes=4, inflight=4, live_sends=1, live_recvs=1, retained=1)
        for _commits, row in samples:
            for key, cap in caps.items():
                self.assertLessEqual(row[key], cap, (key, row))
        half = network.controller.commits // 2
        first = {k: max(r[k] for c, r in samples if c < half) for k in caps}
        second = {k: max(r[k] for c, r in samples if c >= half) for k in caps}
        self.assertEqual(first, second)
        self.assertGreater(driver.defer_counters['deferred_binds'], 8)
        self.assert_serial(network, tokens=65)


class Stops(RebindBase):
    """Cancellation and stop with stage 0 a chunk ahead of stage 1 (flag on)."""
    defer = True

    def finish(self, network, hold, kind='abort', held_ticks=10):
        driver = network.driver
        commits = network.controller.commits
        driver.stop(kind)
        network.frozen = set()
        for _ in range(held_ticks):
            network.tick()
        hold.release()
        for _ in range(3000):
            if not driver.active:
                break
            network.tick()
        self.assertEqual(network.controller.commits, commits)
        self.assert_closed(network)

    def test_stop_at_each_successor_command_with_stage_one_behind(self):
        cuts = 0
        for k in range(6):          # 0..5 stage-0 ops of n+1 on rank 0; 5 = waiting at its PP_SEND
            for kind in ('abort', 'pause', 'reset'):
                with self.subTest(k=k, kind=kind):
                    network = self.network()
                    n = self.first_txn(network)
                    hold = ReceiverHold(network, lambda txn, n=n: txn == n)
                    spy = SenderSpy(network)
                    self.reach(network, lambda: len(stage_ops(network, (0,), n + 1)) >= k
                               and successor_of(network.driver, n) is not None
                               and 0 in successor_of(network.driver, n).opened, limit=2000)
                    if k == 5:
                        self.reach(network, lambda: successor_of(network.driver, n).stage[0]['phase'] == 'bind')
                    self.assertNotIn(1, network.driver._frames[0].released)
                    self.finish(network, hold, kind)
                    self.assertEqual(spy.violations, [])
                    self.assertEqual([op for op in stage_ops(network, (0, 1), n + 1) if op[1] == 'pp_send'], [])
                    cuts += 1
        self.assertEqual(cuts, 18)

    def pending_peer(self, network, txn, kind):
        """Rank 1 answers typed pending for ``txn``'s first ``kind`` command (twice); rank 0 accepts."""
        dispatcher = network.workers[1].dispatcher
        real = dispatcher.submit
        pending, took = [], []

        def busy(command, real=real):
            if command.transaction.txn_seq == txn and command.kind == kind:
                if len(pending) < 2 and (not pending or pending[0] == command.op_seq):
                    pending.append(command.op_seq)
                    raise E3Pending('test: peer busy')
                took.append((command.op_seq, command.kind, command.descriptor))
            return real(command)
        dispatcher.submit = busy
        return pending, took

    def partial_then_stop(self, kind, slot):
        network = self.network()
        n = self.first_txn(network)
        hold = ReceiverHold(network, lambda txn: txn == n)
        driver = network.driver
        pending, took = self.pending_peer(network, n + 1, kind)
        self.reach(network, lambda: pending and successor_of(driver, n) is not None
                   and successor_of(driver, n).stage[0]['_retry'] is not None, limit=2000)
        frame = successor_of(driver, n)
        retry = frame.stage[0]['_retry']
        self.assertEqual(retry.kind, slot)
        self.assertEqual({r for r, op in frame.chunk['accepted_tickets'] if op == retry.op_seq}, {0})
        self.assertNotIn(1, driver._frames[0].released)
        replays = driver.overlap_counters['partial_replays']
        before = len(network.sent_envelopes)
        self.finish(network, hold)
        self.assertGreater(driver.overlap_counters['partial_replays'], replays)
        self.assertIn((retry.op_seq, retry.kind, retry.descriptor), took)
        replayed = [(m, ranks) for method, m, ranks in map(stage_of_command, network.sent_envelopes[before:])
                    if method == 'flash_s5_submit' and m.op_seq == retry.op_seq
                    and m.transaction == retry.transaction]
        self.assertTrue(replayed)
        # The replay goes to the command's own TP targets only (stage 0), same identity.
        self.assertEqual({ranks for _m, ranks in replayed}, {(0, 1)})
        self.assertEqual({(m.kind, m.stage, m.descriptor) for m, _ in replayed},
                         {(retry.kind, retry.stage, retry.descriptor)})

    def test_partially_accepted_successor_prepare_then_stop(self):
        self.partial_then_stop('prepare', w.SegmentKind.PREPARE)

    def test_partially_accepted_successor_layer_then_stop(self):
        self.partial_then_stop('layer', w.SegmentKind.LAYER)

    def test_refused_successor_open_then_stop(self):
        network = self.network()
        n = self.first_txn(network)
        hold = ReceiverHold(network, lambda txn: txn == n)
        driver, controller = network.driver, network.controller

        def no_admission_tick():
            network.complete_everything()
            network.pump()
            return driver.advance()
        self.reach(network, lambda: driver._frames and 0 in driver._frames[0].released
                   and controller.successor is None and not driver._frames[0].dead,
                   limit=2000, tick=no_admission_tick)
        network.admit_next()
        self.assertIsNotNone(controller.successor)
        network.frozen = {0, 1, 2, 3}
        while network.ring.writable():
            network.stock_send()
        refused = driver.stage_counters['headroom_refused']
        driver.advance()            # Begins the successor's frame; the ring refuses its open.
        self.assertEqual(len(driver._frames), 2)
        self.assertEqual(driver._frames[1].opened, set())
        self.assertGreater(driver.stage_counters['headroom_refused'], refused)
        self.finish(network, hold)


class Selection(unittest.TestCase):
    def test_switch_values(self):
        from engine_driver import DriverError
        from stage_overlap import DEFER_BIND_ENV, stage_overlap_defer_bind_enabled as on
        base = {'FLASH_S5_STAGE_LOCAL': '1', 'FLASH_S5_STAGE_OVERLAP': '1'}
        self.assertFalse(on({}))
        self.assertFalse(on(base))
        self.assertTrue(on(dict(base, **{DEFER_BIND_ENV: '1'})))
        for bad in ({DEFER_BIND_ENV: '1'}, {DEFER_BIND_ENV: '1', 'FLASH_S5_STAGE_LOCAL': '1'},
                    dict(base, **{DEFER_BIND_ENV: 'yes'}), dict(base, **{DEFER_BIND_ENV: ''})):
            with self.subTest(bad=bad), self.assertRaises(DriverError):
                on(bad)


if __name__ == '__main__':
    unittest.main()
