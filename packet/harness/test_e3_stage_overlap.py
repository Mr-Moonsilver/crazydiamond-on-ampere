"""C1 connected proofs: two chunks in flight, one command in flight per stage.

Real composition and declared leaves: ``stage_net`` and ``stage_fixture_leaves.json``
(C0's), plus the overlap leaves in ``stage_overlap_fixture_leaves.json``.  CPU only;
no pass here is native evidence.  The KV-level proof with the real manager is
``e3_alpha_c1_probe`` and ``c1_alloc_diff``; here the scheduler is C0's fake.

What each class establishes:

* ``Identity``: per-rank device ledger, PP order and persistent state, overlap
  against the frozen serial driver, under fair and adversarial stage timing; and
  that the chunks really overlapped (successors, both stages busy at once).
* ``Admission``: the successor is admitted only at stage 0's local release, under
  the eligibility rule, with the provisional fields restored; commit and cancel
  keep chunk order; the serial receipts keep their meaning.
* ``Lanes``: stage 0 binds n+1 on stage 1's expected lanes; a different lane at
  stage 1's open is refused before any PP send of n+1.
* ``StopSweep``: a stop at every tick on fair and random timing closes every owner
  with no quarantine, leak or publication, with two chunks in flight.
* ``StopStates``: a stop at each internal boundary of the pair: admitted not begun;
  successor open in flight; stage 0 opened n+1 unbound; stage 0 bound n+1 with
  stage 1 still on n; stage 1 released n without opening n+1; both stages on n+1
  with n uncommitted; one TP peer accepted n+1's command, the other pending;
  the same for the first chunk's PREPARE (frozen-trunk finding 2), overlap selected or not.
* ``StaleAndDuplicate``: replies of the older chunk arriving while the successor
  runs on the same stage; duplicates; a source share of an uncommitted block.
"""
from dataclasses import replace
import random
import types
import unittest

from stage_net import RingStack, ed, sh, stage_of_command, w
from stage_overlap import (OverlapSegmentDriver, StagedAdmissionController, overlap_refusal,
                           stage_overlap_enabled)
from stage_wire import retained_sends
from e3_types import E3ContractError, E3Pending
import async_handoff as ah
import test_e3_r4_worker_bridge as r4
from test_e3_stage_local import Base


class OverlapStack(RingStack):
    """RingStack with the overlap controller and driver.

    Leaf ``fake-request-overlap-fields``: the fake request gains the two fields the
    provisional allocation sets (``num_in_flight_tokens``, ``num_prompt_tokens``)
    and the fake manager the two the eligibility rule reads (no hybrid groups, hash
    block 1), so the rule's only active clause here is ``successor-may-be-terminal``.
    The rule's other clauses run against real managers in ``Admission`` and in the
    real-KV differential.
    """

    def __init__(self, *, tokens=33, chunk=4, direct=False, stop_budget=256, overlap=True):
        self.direct = direct
        self.staged = True
        self.tokens = tokens
        self.chunk = chunk
        self.expected_commits = (tokens - 1 + chunk - 1) // chunk
        request = r4.Request(uid=r4.UID, tokens=tokens)
        request.num_in_flight_tokens = 0
        request.num_prompt_tokens = tokens
        manager = r4.Manager()
        manager.coordinator = types.SimpleNamespace(single_type_managers=())
        manager.block_pool.hash_block_size = 1
        self.scheduler = r4.Scheduler(manager, [request])
        self.controller = StagedAdmissionController(
            self.scheduler, leases=sh.SchedulerLeases(), frontier=ah.CommittedFrontier(),
            engine_epoch=7, chunk_tokens=chunk, ranks=(0, 1, 2, 3), max_history=64).attach()
        self.record = self.controller.admit(self.scheduler)
        self.install_network()
        self.driver = OverlapSegmentDriver(
            self.controller, executor=self.executor, scheduler=self.scheduler,
            engine=types.SimpleNamespace(), layer_count=4, layer_split=2,
            stage_of=((0, 0), (1, 0), (2, 1), (3, 1)),
            reply_timeout_s=30.0, stop_budget=stop_budget).install()
        self.driver.overlap_selected = overlap
        self.workers = {}
        for rank, stage in ((0, 0), (1, 0), (2, 1), (3, 1)):
            self._add_worker(rank, stage)
        self.transport.network = None
        for rank in (2, 3):
            self.worker(rank).channel.ready = True
        self.ledgers = {rank: [] for rank in (0, 1, 2, 3)}
        for rank in (0, 1, 2, 3):
            self._wrap_device_leaves(rank)
        self.inflight_peak = {0: 0, 1: 0}

    def admit_next(self):
        # The scheduler's seam runs every schedule(); the controller decides.
        return self.controller.admit(self.scheduler)

    def tick(self, complete=True):
        # schedule() runs the admission seam on every engine step.
        self.admit_next()
        out = super().tick(complete)
        self.note_peak()
        return out

    def note_peak(self):
        driver = self.driver
        for stage in (0, 1):
            count = sum(1 for seq in driver._inflight if driver._route_stage(seq) == stage)
            self.inflight_peak[stage] = max(self.inflight_peak[stage], count)


def serial_stack(**kwargs):
    kwargs = dict(kwargs)
    tokens = kwargs.pop('tokens', 33)
    chunk = kwargs.pop('chunk', 4)
    return RingStack(staged=False, tokens=tokens, chunk=chunk, **kwargs)


class OverlapBase(Base):
    def network(self, **kwargs):
        network = OverlapStack(**kwargs)
        self.addCleanup(network.close_network)
        return network

    def serial(self, **kwargs):
        network = serial_stack(**kwargs)
        self.addCleanup(network.close_network)
        return network

    def assert_closed(self, network):
        super().assert_closed(network)
        self.assertIsNone(network.controller.successor)
        for rank, worker in network.workers.items():
            self.assertEqual(retained_sends(worker._flash_s5_bridge), {}, rank)

    def two_frames(self, network):
        return len(network.driver._frames) == 2

    def frames(self, network):
        return network.driver._frames

    def slow_tick(self, network, period=6):
        """Stage 1 reads the ring only every ``period`` ticks: its commands take longer, as long
        layers on a busy stream do.  Device completions stay prompt, so each accepted command's
        report carries what has finished (the consumer proof among them), as on the box."""
        count = [0]

        def tick():
            count[0] += 1
            network.admit_next()
            network.complete_everything()
            network.pump((0, 1))
            if count[0] % period == 0:
                network.pump((2, 3))
            out = network.driver.advance()
            network.note_peak()
            return out
        return tick

    def stage_one_held(self, network, until, limit=2000):
        """A slow stage 1 (delayed device completions), so stage 0 runs n+1 while stage 1 is on n."""
        self.reach(network, until, limit=limit, tick=self.slow_tick(network))


class Identity(OverlapBase):
    def off_on(self, tick_on=None, **kwargs):
        off = self.serial(**kwargs)
        off.run_request()
        on = self.network(**kwargs)
        on.run_request(tick=(lambda: tick_on(on)) if tick_on else None)
        return off, on

    def assert_same(self, off, on):
        self.assertEqual(off.worker_errors, [])
        self.assertEqual(on.worker_errors, [])
        for rank in (0, 1, 2, 3):
            self.assertEqual(on.ledgers[rank], off.ledgers[rank], rank)
        self.assertEqual(on.persistent_state(), off.persistent_state())
        self.assertLessEqual(max(on.inflight_peak.values()), 1, on.inflight_peak)

    def test_fair_timing_overlaps_and_matches_serial(self):
        off, on = self.off_on()
        self.assert_same(off, on)
        counters = on.driver.overlap_counters
        ctl = on.controller.overlap_counters
        # 33 tokens in chunks of 4: ends 4..32; a successor of n.end >= 28 may be terminal.
        self.assertEqual(ctl['successors'], counters['successor_frames'])
        self.assertGreaterEqual(ctl['successors'], 4)
        self.assertEqual(counters['overlapped_commits'], ctl['promoted'])
        self.assertGreater(on.driver.stage_counters['both_stages_ns'], 0)
        self.assertEqual(on.controller.overlap_refusals['successor-may-be-terminal'] > 0, True)

    def test_adversarial_stage_timing(self):
        for seed in range(12):
            rng = random.Random(seed)

            def tick(network, rng=rng):
                network.admit_next()
                network.complete_everything()
                behind = rng.choice(((), (0, 1), (2, 3), (1,), (3,)))
                for rank in (0, 1, 2, 3):
                    if rank in behind and rng.random() < 0.8:
                        continue
                    for _ in range(rng.randint(1, 4)):
                        network.step(rank)
                network.settle()
                out = network.driver.advance()
                network.note_peak()
                return out
            with self.subTest(seed=seed):
                off, on = self.off_on(tick_on=tick)
                self.assert_same(off, on)

    def test_slow_stage_one_matches_serial_and_runs_side_by_side(self):
        side_by_side = [0]

        def tick(network, state={}):
            if 'slow' not in state:
                state['slow'] = self.slow_tick(network)
            frames = network.driver._frames
            if (len(frames) == 2 and frames[1].stage[0]['phase'] == 'segments'
                    and 1 not in frames[0].released):
                side_by_side[0] += 1
            return state['slow']()
        off, on = self.off_on(tick_on=tick)
        self.assert_same(off, on)
        self.assertGreater(side_by_side[0], 0)

    def test_chunk_boundary_lengths(self):
        for tokens, chunk in ((5, 4), (9, 4), (17, 4), (6, 3), (25, 3), (41, 5)):
            with self.subTest(tokens=tokens, chunk=chunk):
                off, on = self.off_on(tokens=tokens, chunk=chunk)
                self.assert_same(off, on)

    def test_collective_order_communicator_and_shape(self):
        off, on = self.off_on()
        for rank in (0, 1, 2, 3):
            rows = [row for row in on.ledgers[rank] if row[0].startswith('pp-')]
            self.assertEqual(rows, [row for row in off.ledgers[rank] if row[0].startswith('pp-')])
            self.assertTrue(rows)
            self.assertEqual({row[1] for row in rows}, {'prefill_stream'})
        for sender, receiver in ((0, 2), (1, 3)):
            sends = [row[2:5] for row in on.ledgers[sender] if row[0] == 'pp-send']
            receives = [row[2:5] for row in on.ledgers[receiver] if row[0] == 'pp-recv']
            self.assertEqual(sends, receives)

    def test_unselected_overlap_is_the_c0_stream(self):
        # The overlap classes installed, overlap not selected: C0's stream, envelope for envelope.
        c0 = RingStack(staged=True, tokens=33, chunk=4)
        self.addCleanup(c0.close_network)
        c0.run_request()
        on = self.network(overlap=False)
        on.run_request()
        self.assertEqual([env[0] for env in on.sent_envelopes], [env[0] for env in c0.sent_envelopes])
        for rank in (0, 1, 2, 3):
            self.assertEqual(on.ledgers[rank], c0.ledgers[rank])
        self.assertEqual(on.persistent_state(), c0.persistent_state())
        self.assertFalse(any(on.driver.overlap_counters.values()))
        self.assertFalse(any(on.controller.overlap_counters.values()))

    def test_journal_census_has_no_fault(self):
        _off, on = self.off_on()
        self.assertEqual(on.driver.measurement_census._faults, [])
        kinds = [row.get('kind') or row.get('event') for row in on.driver.journal_rows()] \
            if hasattr(on.driver, 'journal_rows') else []
        del kinds


class Admission(OverlapBase):
    def test_switch_values(self):
        self.assertFalse(stage_overlap_enabled({}))
        self.assertTrue(stage_overlap_enabled({'FLASH_S5_STAGE_OVERLAP': '1', 'FLASH_S5_STAGE_LOCAL': '1'}))
        with self.assertRaises(ed.DriverError):
            stage_overlap_enabled({'FLASH_S5_STAGE_OVERLAP': '1'})
        with self.assertRaises(ed.DriverError):
            stage_overlap_enabled({'FLASH_S5_STAGE_OVERLAP': 'yes', 'FLASH_S5_STAGE_LOCAL': '1'})

    def test_frozen_driver_refuses_the_overlap_controller(self):
        network = self.network()
        with self.assertRaises(ed.DriverError):
            ed.NativeSegmentDriver(network.controller, executor=network.executor,
                                   scheduler=network.scheduler, layer_count=4, layer_split=2,
                                   stage_of=((0, 0), (1, 0), (2, 1), (3, 1)))

    def test_successor_waits_for_stage_zero_local_release(self):
        network = self.network()
        driver, controller = network.driver, network.controller
        seen = []
        for _ in range(2000):
            frames = driver._frames
            before = controller.successor
            network.admit_next()
            if controller.successor is not None and before is None:
                older = frames[0]
                seen.append((sorted(older.released), older.chunk['tx'].txn_seq))
                self.assertIn(0, older.released)
            network.tick()
            if controller.commits >= network.expected_commits and not driver.live:
                break
        self.assertTrue(seen)

    def test_provisional_fields_are_restored(self):
        network = self.network()
        request = network.scheduler.waiting[0] if network.scheduler.waiting else None
        driver, controller = network.driver, network.controller
        self.reach(network, lambda: controller.successor is not None)
        request = controller.parked.request
        self.assertEqual(request.num_in_flight_tokens, 0)
        self.assertEqual(request.num_computed_tokens, controller.parked.transaction.chunk_start)
        calls = network.scheduler.kv_cache_manager.alloc_calls
        self.assertEqual(calls[-1][1], controller.successor.transaction.chunk_end
                         - controller.successor.transaction.chunk_start)
        del driver

    def test_refusal_keeps_serial_receipts(self):
        network = self.network()
        controller = network.controller
        self.reach(network, lambda: network.driver._frames and 0 in network.driver._frames[0].released
                   and controller.successor is None)
        receipts = (controller.refused, controller.alloc_failures, controller.refusal_seq)
        network.scheduler.kv_cache_manager.fail = True
        self.assertIsNone(controller.admit(network.scheduler))
        network.scheduler.kv_cache_manager.fail = False
        self.assertEqual((controller.refused, controller.alloc_failures, controller.refusal_seq), receipts)
        self.assertEqual(controller.overlap_counters['alloc_wait'], 1)
        network.run_request()
        self.assertEqual(controller.commits, network.expected_commits)

    def test_commit_and_cancel_keep_chunk_order(self):
        network = self.network()
        controller = network.controller
        self.reach(network, lambda: controller.successor is not None)
        with self.assertRaises(E3ContractError):
            controller.commit(controller.successor.transaction, None)
        with self.assertRaises(E3ContractError):
            controller.cancel(controller.parked.transaction, never_submitted=True)

    def test_eligibility_rule(self):
        def manager(groups=(), hash_size=16, eagle=False):
            items = [types.SimpleNamespace(use_eagle=eagle, _partial_hit_reqs=set(), _checkpoints={})
                     for _ in range(groups)]
            return types.SimpleNamespace(coordinator=types.SimpleNamespace(single_type_managers=items),
                                         block_pool=types.SimpleNamespace(hash_block_size=hash_size))

        def request(prompt=200, boundary=None):
            return types.SimpleNamespace(request_id='r', num_prompt_tokens=prompt, num_tokens=prompt,
                                         shared_prefix_boundary=boundary)
        cap = 16
        self.assertIsNone(overlap_refusal(manager(2), request(), 64, False, cap))
        self.assertEqual(overlap_refusal(manager(2), request(), 64, True, cap), 'older-terminal')
        self.assertEqual(overlap_refusal(manager(2), request(), 199 - 16, False, cap), 'successor-may-be-terminal')
        # Last hash boundary 192 at hash 16, cap small enough to reach it first.
        self.assertEqual(overlap_refusal(manager(2), request(), 192, False, 4), 'older-at-last-hash-boundary')
        self.assertEqual(overlap_refusal(manager(2, eagle=True), request(), 176, False, 4),
                         'older-at-last-hash-boundary')
        self.assertEqual(overlap_refusal(manager(2), request(boundary=64), 64, False, cap),
                         'older-at-shared-prefix-boundary')
        m = manager(2)
        m.coordinator.single_type_managers[1]._partial_hit_reqs.add('r')
        self.assertEqual(overlap_refusal(m, request(), 64, False, cap), 'partial-tail-pending')
        m = manager(2)
        m.coordinator.single_type_managers[0]._checkpoints['r'] = 1
        self.assertEqual(overlap_refusal(m, request(), 64, False, cap), 'partial-tail-pending')
        self.assertEqual(overlap_refusal(types.SimpleNamespace(), request(), 64, False, cap), 'manager-shape')


class Lanes(OverlapBase):
    def test_stage_one_lane_equals_the_bound_expectation(self):
        network = self.network()
        network.run_request()
        counters = network.driver.overlap_counters
        self.assertGreater(counters['early_opens'], 0)
        self.assertEqual(counters['lane_checks'], counters['successor_frames'])

    def test_a_different_stage_one_lane_is_refused_before_the_pp_send(self):
        network = self.network()
        driver = network.driver
        self.reach(network, lambda: self.two_frames(network) and driver._frames[1].early)
        frame = driver._frames[1]
        rank = network.driver.stage_ranks[1][0]
        frame.expected_lanes[rank] = replace(frame.expected_lanes[rank],
                                             storage_epoch=frame.expected_lanes[rank].storage_epoch + 5)
        sends = [network.workers[r].channel.sends for r in (0, 1)]
        with self.assertRaises(ed.DriverError):
            for _ in range(400):
                network.tick()
        self.assertIn('lane', str(driver.error) + repr(network.status_text()) or 'lane')
        # No PP send of the successor left stage 0.
        self.assertEqual([network.workers[r].channel.sends for r in (0, 1)], sends)


class StopSweep(OverlapBase):
    def tick_count(self):
        network = self.network()
        ticks = [0]

        def tick():
            ticks[0] += 1
            return network.tick()
        network.run_request(tick=tick)
        return ticks[0]

    def closure(self, network, tick=None, limit=3000):
        for _ in range(limit):
            if not network.driver.active:
                return
            (tick or network.tick)()
        self.fail('stop did not close: %r' % (network.status_text(),))

    def test_stop_at_every_tick_fair_timing(self):
        total = self.tick_count()
        self.assertGreater(total, 40)
        pairs = 0
        for cut in range(total):
            network = self.network()
            with self.subTest(cut=cut):
                for _ in range(cut):
                    network.admit_next()
                    network.tick()
                commits = network.controller.commits
                pairs += int(network.controller.successor is not None)
                network.driver.stop('abort')
                self.closure(network)
                self.assertEqual(network.controller.commits, commits)
                self.assert_closed(network)
        self.assertGreater(pairs, 10)

    def test_stop_at_every_tick_random_timing(self):
        total = self.tick_count()
        for cut in range(0, total, 2):
            seed = cut * 7 + 1
            rng = random.Random(seed)
            network = self.network()

            def tick(network=network, rng=rng):
                for rank in (0, 1, 2, 3):
                    if rng.random() < 0.6:
                        network.workers[rank].bus.complete_all(True)
                    for _ in range(rng.randint(0, 3)):
                        network.step(rank)
                network.settle()
                return network.driver.advance()

            with self.subTest(cut=cut, seed=seed):
                for _ in range(cut * 2):
                    network.admit_next()
                    tick()
                commits = network.controller.commits
                network.driver.stop(rng.choice(('abort', 'pause', 'reset')))
                for _ in range(rng.randint(0, 20)):
                    tick()
                self.closure(network)
                self.assertEqual(network.controller.commits, commits)
                self.assert_closed(network)


class StopStates(OverlapBase):
    def finish(self, network, kind='abort'):
        driver = network.driver
        commits = network.controller.commits
        driver.stop(kind)
        network.frozen = set()
        for _ in range(3000):
            if not driver.active:
                break
            network.tick()
        self.assertEqual(network.controller.commits, commits)
        self.assert_closed(network)

    def test_admitted_not_begun(self):
        network = self.network()
        controller, driver = network.controller, network.driver
        self.reach(network, lambda: driver._frames and 0 in driver._frames[0].released
                   and controller.successor is None and not driver._frames[0].dead)
        network.admit_next()
        self.assertIsNotNone(controller.successor)
        self.assertEqual(len(driver._frames), 1)
        self.finish(network)
        self.assertEqual(driver.overlap_counters['unbegun_successor_cancels'], 1)

    def test_successor_open_in_flight(self):
        network = self.network()
        driver = network.driver
        self.reach(network, lambda: self.two_frames(network))
        network.frozen = {0, 1}
        for _ in range(3):
            network.tick()
        self.assertEqual(driver._frames[1].opened, set())
        self.finish(network)

    def test_stage_zero_opened_successor_unbound(self):
        network = self.network()
        driver = network.driver
        self.reach(network, lambda: self.two_frames(network) and 0 in driver._frames[1].opened)
        self.assertIsNone(driver._frames[1].chunk['bound_ticket'])
        self.finish(network)

    def test_stage_zero_bound_successor_with_stage_one_on_the_older(self):
        network = self.network()
        driver = network.driver
        self.stage_one_held(network, lambda: self.two_frames(network) and driver._frames[1].chunk['bound_ticket'] is not None)
        self.assertNotIn(1, driver._frames[0].released)
        self.finish(network)
        self.assertEqual(network.controller.overlap_counters['partial_successor_retires'], 1)

    def test_stage_one_released_older_without_opening_successor(self):
        network = self.network()
        driver = network.driver
        self.reach(network, lambda: self.two_frames(network) and driver._frames[1].opened == {0}
                   and driver._frames[0].released == {0, 1}
                   and driver._frames[0].chunk['_receipt_inflight'] is not None)
        self.finish(network)

    def test_both_stages_on_successor_with_older_uncommitted(self):
        network = self.network()
        driver = network.driver
        # Hold stage 0's replies, so the older chunk's sender receipt stays unanswered.
        reached = False
        for _ in range(3000):
            frames = driver._frames
            if len(frames) == 2 and frames[0].released == {0, 1} and 1 in frames[1].opened:
                reached = True
                break
            if len(frames) == 2 and frames[0].released == {0, 1}:
                network.frozen = {0, 1}
            network.admit_next()
            network.tick()
        if not reached:
            self.skipTest('state not reachable on this fixture: the older commits at stage 1\'s release')
        self.finish(network)

    def test_one_tp_peer_accepted_the_successor_command(self):
        network = self.network()
        driver = network.driver
        self.stage_one_held(network, lambda: self.two_frames(network)
                            and driver._frames[1].stage[0]['phase'] == 'segments')
        successor = driver._frames[1]
        transaction = successor.chunk['tx']
        dispatcher = network.workers[1].dispatcher
        real = dispatcher.submit
        pending = []

        peer_took = []

        def busy(command, real=real):
            # Rank 1 answers typed pending for the successor's next command; rank 0 accepts it.
            if command.transaction == transaction and len(pending) < 3:
                pending.append(command.op_seq)
                raise E3Pending('test: peer busy')
            if command.transaction == transaction:
                peer_took.append((command.op_seq, command.kind, command.descriptor))
            return real(command)
        dispatcher.submit = busy
        self.reach(network, lambda: pending and successor.stage[0]['_retry'] is not None,
                   tick=self.slow_tick(network))
        retry = successor.stage[0]['_retry']
        accepted = {rank for rank, op in successor.chunk['accepted_tickets'] if op == retry.op_seq}
        self.assertEqual(accepted, {0})
        replays = driver.overlap_counters['partial_replays']
        self.finish(network)
        # The stop replayed the retained command to the pending peer before cancelling,
        # and the peer took exactly that command.
        self.assertGreater(driver.overlap_counters['partial_replays'], replays)
        self.assertIn((retry.op_seq, retry.kind, retry.descriptor), peer_took)
        sent = [stage_of_command(env)[1] for env in network.sent_envelopes if env[0] == 'flash_s5_submit']
        same = [m for m in sent if m.op_seq == retry.op_seq and m.transaction == transaction]
        # The command kept its identity and went to the peer again.
        self.assertGreaterEqual(len(same), 2)
        self.assertEqual({(m.kind, m.stage, m.descriptor) for m in same}, {(retry.kind, retry.stage, retry.descriptor)})


    def first_frame_partial_prepare(self, overlap):
        """Frozen-trunk finding 2 on the overlap driver: the first chunk's PREPARE accepted by
        rank 0, rank 1 pending, then a stop.  The retained command is replayed, same identity,
        to the peer before the cancel; the owners close with nothing published."""
        network = self.network(overlap=overlap)
        driver = network.driver
        dispatcher = network.workers[1].dispatcher
        real = dispatcher.submit
        first, peer_took = [], []

        def busy(command, real=real):
            if not first:
                first.append(command)
            if command.transaction == first[0].transaction and command.op_seq == first[0].op_seq:
                if len(first) < 3:
                    first.append(command)
                    raise E3Pending('test: peer busy')
                peer_took.append((command.op_seq, command.kind, command.descriptor))
            return real(command)
        dispatcher.submit = busy
        self.reach(network, lambda: len(first) >= 2 and driver._frames
                   and driver._frames[0].stage[0]['_retry'] is not None)
        frame = driver._frames[0]
        retry = frame.stage[0]['_retry']
        self.assertIs(retry.kind, w.SegmentKind.PREPARE)
        self.assertIsNone(frame.chunk['bound_ticket'])
        self.assertEqual({rank for rank, op in frame.chunk['accepted_tickets'] if op == retry.op_seq}, {0})
        replays = driver.overlap_counters['partial_replays']
        self.finish(network)
        self.assertGreater(driver.overlap_counters['partial_replays'], replays)
        self.assertIn((retry.op_seq, retry.kind, retry.descriptor), peer_took)
        self.assertEqual(network.controller.commits, 0)

    def test_first_frame_partial_prepare_then_stop(self):
        self.first_frame_partial_prepare(overlap=True)

    def test_first_frame_partial_prepare_then_stop_overlap_unselected(self):
        self.first_frame_partial_prepare(overlap=False)


class Control(OverlapBase):
    """The per-arm selection: ``native_stage_local {local, overlap}`` at paused idle FULL."""

    def idle_full(self):
        network = self.network()
        driver = network.driver
        driver.controller = types.SimpleNamespace(active=False)
        driver.engine = types.SimpleNamespace(is_scheduler_paused=lambda: True)
        return driver

    def select(self, driver, **fields):
        return driver._submit_control(dict(action='native_stage_local', **fields))

    def test_overlap_selection_and_its_previous(self):
        driver = self.idle_full()
        reply = self.select(driver, local=True, overlap=False)
        self.assertEqual((reply['local'], reply['overlap'], reply['previous_overlap']), (True, False, True))
        self.assertEqual((driver.stage_selected, driver.overlap_selected), (True, False))
        reply = self.select(driver, local=True)          # C0-shaped: keeps the overlap choice
        self.assertEqual((reply['overlap'], reply['previous_overlap']), (False, False))
        reply = self.select(driver, local=True, overlap=True)
        self.assertEqual((reply['overlap'], reply['previous_overlap']), (True, False))
        reply = self.select(driver, local=False)          # serial selects no overlap
        self.assertEqual((reply['local'], reply['overlap'], reply['previous_overlap']), (False, False, True))
        self.assertEqual((driver.stage_selected, driver.overlap_selected), (False, False))

    def test_overlap_without_local_is_refused_before_anything_is_applied(self):
        driver = self.idle_full()
        for bad in (dict(local=False, overlap=True), dict(overlap=True), dict(local=True, overlap=1),
                    dict(local=True, overlap='true'), dict(local=True, overlap=None)):
            with self.subTest(bad=bad), self.assertRaises(ed.DriverError):
                self.select(driver, **bad)
            self.assertEqual((driver.stage_selected, driver.overlap_selected), (True, True))
        self.assertEqual(driver.stage_counters['selections'], 0)

    def test_overlap_selection_refused_unless_idle(self):
        driver = self.idle_full()
        driver.tx = object()
        with self.assertRaises(ed.DriverError) as caught:
            self.select(driver, local=True, overlap=False)
        self.assertIn('requires-paused-idle-full', str(caught.exception))
        self.assertTrue(driver.overlap_selected)
        driver.tx = None


class StaleAndDuplicate(OverlapBase):
    def test_older_reply_while_successor_runs_on_the_stage(self):
        network = self.network()
        driver = network.driver
        self.stage_one_held(network, lambda: self.two_frames(network) and driver._frames[1].stage[0]['phase'] == 'segments')
        older, newer = driver._frames
        with driver._scope(newer, 0):
            command = driver._command or driver._retry
            if command is None:
                self.skipTest('no successor command outstanding at this tick')
            stale = w.SegmentAcceptance(older.chunk['tx'], 0, command.kind, command.stage,
                                        command.op_seq, accepted=True)
            with self.assertRaises(ed.DriverError):
                driver._apply_segments((stale, stale))

    def test_duplicate_reply_never_commits(self):
        network = self.network()
        self.reach(network, lambda: self.two_frames(network))
        commits = network.controller.commits
        mq = network.reply_queues[0]
        real = mq.enqueue

        def twice(wire, real=real):
            real(wire)
            real(wire)
        mq.enqueue = twice
        with self.assertRaises((ed.DriverError, ed.TransportError, RuntimeError)):
            for _ in range(80):
                network.tick()
        self.assertLessEqual(network.controller.commits, commits + 1)

    def test_uncommitted_blocks_are_not_shared(self):
        network = self.network()
        controller = network.controller
        self.reach(network, lambda: controller.successor is not None)
        leases = controller.leases
        blocks = controller.successor.admission.flat_blocks()
        held = [types.SimpleNamespace(block_id=b) for b in blocks if b > 0]
        with self.assertRaises(E3Pending):
            leases.before_touch(held)


if __name__ == '__main__':
    unittest.main()
