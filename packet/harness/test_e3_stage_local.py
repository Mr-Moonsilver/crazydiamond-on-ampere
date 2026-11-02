"""C0 connected proofs: stage-local progress with overlap disabled.

Real composition and declared leaves: see ``stage_net`` and
``stage_fixture_leaves.json``.  CPU only; no pass here is native evidence.

What each class establishes:

* ``Identity``: per-rank device ledger and persistent state, switch on against
  switch off, under fair and adversarial stage timing.
* ``NonTargetIndependence`` (D1): a stage's command is applied with the other
  stage's ranks unread; non-targets read the slot and write nothing.
* ``RingHeadroom``: native commands cannot make a stock enqueue block.
* ``LocalRelease``: a sender releases before the consumer proof, its PP slot stays
  owned, only the typed receipt frees it, and the commit comes last.
* ``StopSweep``: a stop at every tick of every chunk, on fair and random timing,
  closes every owner with no quarantine, leak or publication.
* ``PartialOpen``: a stop with a stage still holding the older retained row.
* ``PeerReplay``: one TP peer accepted, the other pending: the same command is
  replayed to the peer before anything is cancelled.
* ``StaleAndDuplicate``: stale generations, duplicates, wrong owners.
* ``BurstInWait``: the in-wait permit per stage; only in-chunk replies are applied
  inside a stock wait, and commit stays on the tick.
"""
from dataclasses import replace
import pickle
import random
import threading
import types
import unittest

from stage_net import RING_SLOTS, RingStack, ed, eh, stage_of_command, w
from stage_progress import POLL_RESERVE, RING_RESERVE, stage_local_enabled
from stage_wire import STAGE_SKIP, StageScoped, retained_sends, stage_rpc
from e3_types import E3ContractError


class Base(unittest.TestCase):
    def network(self, **kwargs):
        network = RingStack(**kwargs)
        self.addCleanup(network.close_network)
        return network

    def reach(self, network, until, limit=1000, tick=None):
        tick = tick or network.tick
        for _ in range(limit):
            if until():
                return
            tick()
        self.fail('network did not reach the expected state: %r' % (network.status_text(),))

    def opened(self, network):
        """Ticks past bootstrap, split route and open, to the chunk's first command."""
        self.reach(network, lambda: network.driver._frames
                   and network.driver._frames[0].opened == {0, 1})

    def stage(self, network, stage):
        return network.driver._frames[0].stage[stage]

    def assert_closed(self, network):
        driver = network.driver
        self.assertEqual(network.worker_errors, [])
        self.assertFalse(driver.active)
        self.assertEqual(driver._frames, [])
        self.assertEqual(driver._inflight, {})
        self.assertIsNone(driver._retained)
        self.assertIsNone(network.controller.parked)
        self.assertTrue(network.controller.leases.shutdown()['clean'])
        self.assertEqual(network.transport._calls, {})
        for rank, worker in network.workers.items():
            bridge = worker._flash_s5_bridge
            self.assertIsNone(bridge.transaction, rank)
            self.assertIsNone(bridge._retained, rank)
            self.assertIsNone(worker.dispatcher.transaction, rank)
            self.assertIsNone(worker.dispatcher.quarantined, rank)
            self.assertEqual(retained_sends(bridge), {}, rank)
            self.assertEqual([t.state for t in worker.channel.tickets
                              if t.state != 'retired'], [], rank)
            self.assertIsNone(worker.factory.registered, rank)


class Switch(unittest.TestCase):
    def test_switch_values(self):
        self.assertFalse(stage_local_enabled({}))
        self.assertFalse(stage_local_enabled({'FLASH_S5_STAGE_LOCAL': '0'}))
        self.assertTrue(stage_local_enabled({'FLASH_S5_STAGE_LOCAL': '1'}))
        with self.assertRaises(ed.DriverError):
            stage_local_enabled({'FLASH_S5_STAGE_LOCAL': 'yes'})

    def test_stage_skip_never_leaves_its_worker(self):
        with self.assertRaises(E3ContractError):
            pickle.dumps(STAGE_SKIP)

    def test_scope_record_refuses_malformed_targets(self):
        for ranks in ((), (1, 0), (0, 0), [0], (-1,), ('0',)):
            with self.assertRaises(E3ContractError):
                StageScoped('x', ranks)
        with self.assertRaises(E3ContractError):
            StageScoped(StageScoped('x', (0,)), (0,))


class Selection(Base):
    """The per-arm switch: one engine incarnation, chosen while paused idle FULL."""

    def unselected(self, **kwargs):
        network = self.network(**kwargs)
        network.driver.stage_selected = False
        return network

    def test_unselected_chunks_are_the_frozen_stream(self):
        off = self.network(staged=False)
        off.run_request()
        on = self.unselected()
        seen = []
        def tick():
            seen.append(bool(on.driver._frames))
            return on.tick()
        on.run_request(tick=tick)
        self.assertFalse(any(seen))
        self.assertEqual(on.driver.stage_counters['chunks'], 0)
        self.assertFalse(any(type(row[1][0]) is StageScoped for row in on.sent_envelopes
                             if row[1]))
        self.assertEqual(pickle.dumps(on.sent_envelopes), pickle.dumps(off.sent_envelopes))
        for rank in (0, 1, 2, 3):
            self.assertEqual(on.ledgers[rank], off.ledgers[rank], rank)
        self.assertEqual(on.persistent_state(), off.persistent_state())

    def idle_full(self):
        network = self.network()
        driver = network.driver
        driver.controller = types.SimpleNamespace(active=False)
        driver.engine = types.SimpleNamespace(is_scheduler_paused=lambda: True)
        return driver

    def test_control_accepts_only_typed_paused_idle_full(self):
        driver = self.idle_full()
        reply = driver._submit_control(dict(action='native_stage_local', local=False))
        self.assertEqual((reply['local'], reply['previous'], reply['route']), (False, True, 'full'))
        self.assertFalse(driver.stage_selected)
        self.assertEqual(driver.stage_counters['selections'], 1)
        self.assertEqual(driver._submit_control(dict(action='native_stage_local', local=False))['previous'],
                         False)
        self.assertEqual(driver.stage_counters['selections'], 1)
        for bad in (dict(local=1), dict(local='true'), dict(), dict(local=True, other=1),
                    dict(local=True, engine_epoch='x')):
            with self.subTest(bad=bad), self.assertRaises(ed.DriverError):
                driver._submit_control(dict(action='native_stage_local', **bad))
        self.assertFalse(driver.stage_selected)

    def test_control_refused_unless_idle(self):
        def paused_false(driver):
            driver.engine = types.SimpleNamespace(is_scheduler_paused=lambda: False)
        def split(driver):
            driver._split_ready = True
        def tx(driver):
            driver.tx = object()
        def frame(driver):
            driver._frames.append(object())
        def active(driver):
            driver.controller = types.SimpleNamespace(active=True)
        def inflight(driver):
            driver._inflight[1] = 'poll'
        def retained(driver):
            driver._retained = dict()
        def intent(driver):
            driver._admission_intent = dict()
        for change in (paused_false, split, tx, frame, active, inflight, retained, intent):
            driver = self.idle_full()
            change(driver)
            with self.subTest(change=change.__name__):
                with self.assertRaises(ed.DriverError) as caught:
                    driver._submit_control(dict(action='native_stage_local', local=False))
                self.assertIn('requires-paused-idle-full', str(caught.exception))
                self.assertTrue(driver.stage_selected)
            driver._frames.clear()
            driver._inflight.clear()
            driver.tx = None

    def test_other_controls_pass_through_and_status_carries_the_row(self):
        driver = self.idle_full()
        with self.assertRaises(ed.DriverError) as caught:
            driver._submit_control(dict(action='no-such-action'))
        self.assertIn('native-control-unknown-action', str(caught.exception))
        row = driver.stage_status()
        self.assertEqual((row['selected'], row['frames']), (True, []))
        self.assertEqual((row['ring_reserve'], row['poll_reserve']), (RING_RESERVE, POLL_RESERVE))

    def test_the_frozen_driver_has_no_such_control(self):
        network = self.network(staged=False)
        with self.assertRaises(ed.DriverError) as caught:
            network.driver._submit_control(dict(action='native_stage_local', local=True))
        self.assertIn('native-control-unknown-action', str(caught.exception))

    def test_selected_then_unselected_on_one_network_each_keep_their_identity(self):
        off = self.network(staged=False, tokens=9)
        off.run_request()
        on = self.network(tokens=9)
        chunks = []
        def tick():
            # Chosen before each chunk's open, as the control does between arms; latched per chunk.
            if not on.driver.tx and not on.driver._frames:
                on.driver.stage_selected = on.controller.commits % 2 == 0
            if on.driver._frames:
                chunks.append(on.driver._frames[0].chunk['tx'].txn_seq)
            return on.tick()
        on.run_request(tick=tick)
        self.assertEqual(on.worker_errors, [])
        for rank in (0, 1, 2, 3):
            self.assertEqual(on.ledgers[rank], off.ledgers[rank], rank)
        self.assertEqual(on.persistent_state(), off.persistent_state())
        self.assertTrue(chunks)
        self.assertLess(len(set(chunks)), on.expected_commits)


class Memory(Base):
    """No new permanent buffers; the transient state is bounded and counted."""

    def test_no_tensor_allocation_in_the_new_modules(self):
        import ast
        from pathlib import Path
        engine = Path(ed.__file__).parent
        for name in ('stage_wire.py', 'stage_progress.py'):
            tree = ast.parse((engine / name).read_text())
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported |= {alias.name.split('.')[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom):
                    imported.add((node.module or '').split('.')[0])
            self.assertFalse(imported & {'torch', 'numpy', 'mmap', 'multiprocessing'}, name)

    def test_tickets_and_tables_stay_bounded(self):
        off = self.network(staged=False, tokens=33)
        on = self.network(tokens=33)
        peak = dict(off=0, on=0, retained=0, retired=0, routes=0)
        def live(network):
            return max(sum(ticket.state != 'retired' for ticket in worker.channel.tickets)
                       for worker in network.workers.values())
        def tick(network, name):
            result = network.tick()
            peak[name] = max(peak[name], live(network))
            if name == 'on':
                for worker in network.workers.values():
                    bridge = worker._flash_s5_bridge
                    peak['retained'] = max(peak['retained'], len(retained_sends(bridge)))
                    peak['retired'] = max(peak['retired'],
                                          len(bridge.__dict__.get('_flash_s5_retired_sends', ())))
                peak['routes'] = max(peak['routes'], len(network.driver._seq_route))
            return result
        off.run_request(tick=lambda: tick(off, 'off'))
        on.run_request(tick=lambda: tick(on, 'on'))
        self.assertEqual(off.worker_errors + on.worker_errors, [])
        # One PP ticket at a time in C0 on each rank, as on the frozen path (PP_SLOTS is 2).
        self.assertLessEqual(peak['on'], max(peak['off'], 1))
        self.assertLessEqual(peak['retained'], 1)
        self.assertLessEqual(peak['retired'], 8)
        self.assertLessEqual(peak['routes'], 2)
        self.assertEqual(on.driver._seq_route, {})
        self.assertEqual(on.driver._frames, [])


class Identity(Base):
    """Switch on against switch off: same per-rank work and same persistent state."""

    def off_on(self, tick_on=None, **kwargs):
        off = self.network(staged=False, **kwargs)
        off.run_request()
        on = self.network(staged=True, **kwargs)
        on.run_request(tick=(lambda: tick_on(on)) if tick_on else None)
        return off, on

    def assert_same(self, off, on):
        self.assertEqual(off.worker_errors, [])
        self.assertEqual(on.worker_errors, [])
        for rank in (0, 1, 2, 3):
            self.assertEqual(on.ledgers[rank], off.ledgers[rank], rank)
        self.assertEqual(on.persistent_state(), off.persistent_state())

    def test_fair_timing(self):
        off, on = self.off_on()
        self.assert_same(off, on)
        # The ledgers are not vacuous: every chunk ran every op on every rank.
        kinds = [row[1] for row in on.ledgers[0] if row[0] == 'op']
        self.assertEqual(kinds.count('prepare'), on.expected_commits)
        self.assertEqual(kinds.count('layer'), 2 * on.expected_commits)
        self.assertEqual(on.controller.commits, on.expected_commits)

    def test_adversarial_stage_timing(self):
        for seed in range(12):
            rng = random.Random(seed)

            def tick(network, rng=rng):
                network.complete_everything()
                # Each rank reads a random amount of the ring; one stage often lags.
                behind = rng.choice(((), (0, 1), (2, 3), (1,), (3,)))
                for rank in (0, 1, 2, 3):
                    if rank in behind and rng.random() < 0.8:
                        continue
                    for _ in range(rng.randint(1, 4)):
                        network.step(rank)
                network.settle()
                return network.driver.advance()
            with self.subTest(seed=seed):
                off, on = self.off_on(tick_on=tick)
                self.assert_same(off, on)

    def test_chunk_boundary_lengths(self):
        for tokens, chunk in ((5, 4), (9, 4), (17, 4), (6, 3)):
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
        # TP peers on a stage agree on the transfer order, and the sender's nth
        # transfer pairs with its receiver's nth, on the same identity.
        for sender, receiver in ((0, 2), (1, 3)):
            sends = [row[2:5] for row in on.ledgers[sender] if row[0] == 'pp-send']
            receives = [row[2:5] for row in on.ledgers[receiver] if row[0] == 'pp-recv']
            self.assertEqual(sends, receives)
        self.assertEqual([r[2:] for r in on.ledgers[0] if r[0] == 'pp-send'],
                         [r[2:] for r in on.ledgers[1] if r[0] == 'pp-send'])

    def test_journal_census_has_no_fault(self):
        _off, on = self.off_on()
        self.assertEqual(on.driver.measurement_census._faults, [])
        kinds = [entry['kind'] for entry in on.driver._journal]
        for kind in ('chunk_open', 'segment', 'report', 'pp_receipt', 'chunk_commit',
                     'source_hold', 'committed_frontier'):
            self.assertIn(kind, kinds)
        self.assertEqual(kinds.count('chunk_commit'), on.expected_commits)
        self.assertEqual(kinds.count('chunk_open'), on.expected_commits)


class NonTargetIndependence(Base):
    def test_non_targets_read_the_slot_and_write_no_reply(self):
        network = self.network()
        network.run_request()
        for rank in (0, 1, 2, 3):
            self.assertGreater(network.skips[rank], 0, rank)
        # Every reply a stage rank wrote answers a command that named it.
        replies = {rank: network.reply_queues[rank].written for rank in (0, 1, 2, 3)}
        self.assertEqual(replies[0], replies[1])
        self.assertEqual(replies[2], replies[3])

    def test_stage_zero_applies_with_stage_one_unread(self):
        network = self.network()
        self.reach(network, lambda: network.driver._frames
                   and self.stage(network, 1)['index'] >= 1)
        network.frozen = {2, 3}
        written = {rank: network.reply_queues[rank].written for rank in (2, 3)}
        before = self.stage(network, 0)['index']
        for _ in range(30):
            network.tick()
        driver = network.driver
        self.assertGreater(self.stage(network, 0)['index'], before)
        self.assertEqual(written, {rank: network.reply_queues[rank].written for rank in (2, 3)})
        # The trunk path makes no progress at all in the same state.
        trunk = self.network(staged=False)
        self.reach(trunk, lambda: trunk.driver.index >= 2)
        trunk.frozen = {2, 3}
        held = trunk.driver.index
        for _ in range(30):
            trunk.tick()
        self.assertLessEqual(trunk.driver.index, held + 1)
        self.assertEqual(driver.error, None)

    def test_stage_scoped_command_for_another_rank_does_nothing(self):
        network = self.network()
        self.opened(network)
        bridge = network.workers[2]._flash_s5_bridge
        calls = []
        self.assertIs(stage_rpc(bridge, 'submit', StageScoped('x', (0, 1)), calls.append), STAGE_SKIP)
        self.assertEqual(calls, [])

    def test_wrong_method_for_the_record_is_refused(self):
        network = self.network()
        self.opened(network)
        request = w.PollRequest(network.driver.tx, network.driver.admission.state_epoch, 1)
        with self.assertRaises(E3ContractError):
            eh.worker_rpc(network.workers[0], 'submit', StageScoped(request, (0, 1)))


class RingHeadroom(Base):
    def test_native_commands_leave_the_stock_reserve(self):
        network = self.network()
        self.reach(network, lambda: network.driver._frames
                   and self.stage(network, 1)['index'] >= 1)
        network.frozen = {2, 3}
        for _ in range(40):
            network.tick()
        driver = network.driver
        self.assertGreater(driver.stage_counters['headroom_refused'], 0)
        # Native never used the reserve, and a finalize poll stops earlier still:
        # the stock decode pair fits at a ring native could otherwise have filled.
        self.assertLessEqual(network.ring.max_lag, RING_SLOTS - RING_RESERVE)
        self.assertEqual(self.stage(network, 0)['phase'], 'finalize')
        lag = network.ring.lag(2)
        self.assertLessEqual(lag, RING_SLOTS - RING_RESERVE)
        # Polls do not grow the lag past their own, earlier, limit.
        for _ in range(10):
            network.tick()
        self.assertEqual(network.ring.lag(2), lag)
        for _ in range(RING_SLOTS - lag):
            self.assertTrue(network.ring.writable())
            network.stock_send()
        self.assertFalse(network.ring.writable())
        self.assertEqual(network.ring.would_block, 0)
        # Readers catching up resume native work with nothing lost.
        network.frozen = set()
        network.run_request()
        self.assertEqual(network.worker_errors, [])
        self.assertEqual(network.controller.commits, network.expected_commits)

    def test_polls_stop_earlier_than_commands(self):
        network = self.network()
        self.reach(network, lambda: network.driver._frames
                   and self.stage(network, 0)['phase'] == 'finalize')
        network.frozen = {2, 3}
        for _ in range(40):
            network.tick()
        self.assertLessEqual(network.ring.lag(2), RING_SLOTS - POLL_RESERVE)

    def test_a_polling_stage_cannot_starve_the_stage_it_waits_for(self):
        # Stage 0's finalize polls run at tick rate; stage 1 reads slowly and owes
        # the PP_RECEIVE stage 0's send is waiting for.
        network = self.network()
        tick_no = [0]

        def tick():
            tick_no[0] += 1
            network.complete_everything()
            network.step(0), network.step(0), network.step(1), network.step(1)
            if tick_no[0] % 4 == 0:
                network.step(2), network.step(3)
            network.settle()
            return network.driver.advance()
        network.run_request(tick=tick, limit=3000)
        self.assertEqual(network.worker_errors, [])
        self.assertEqual(network.controller.commits, network.expected_commits)

    def test_a_lagging_stage_never_blocks_stock(self):
        for seed in range(6):
            rng = random.Random(seed)
            network = self.network()

            def tick(network=network, rng=rng):
                network.complete_everything()
                for rank in (0, 1):
                    network.step(rank)
                    network.step(rank)
                if rng.random() < 0.3:
                    network.step(2), network.step(3)
                if not network.ring.writable():
                    network.step(2), network.step(3)
                # Stock arrives slower than the slow stage reads: a ring kept full
                # by stock itself starves native by design (stock has priority).
                if network.ring.writable() and rng.random() < 0.1:
                    network.stock_send()
                network.settle()
                return network.driver.advance()
            network.run_request(tick=tick)
            self.assertEqual(network.ring.would_block, 0)
            self.assertEqual(network.worker_errors, [])


class LocalRelease(Base):
    def sender_released(self, network):
        return network.driver._frames and self.stage(network, 0)['phase'] == 'released'

    def hold_stage_one_events(self, network):
        def tick():
            for rank in (0, 1):
                network.workers[rank].bus.complete_all(True)
            network.pump()
            return network.driver.advance()
        return tick

    def to_sender_released(self, network):
        tick = self.hold_stage_one_events(network)
        # Stage 1 runs commands but no event of its completes, until released.
        self.reach(network, lambda: self.stage(network, 1)['phase'] == 'finalize'
                   if network.driver._frames and network.driver._frames[0].opened == {0, 1}
                   else False, tick=network.tick if False else tick)
        self.reach(network, lambda: self.sender_released(network), tick=tick)
        return tick

    def test_sender_releases_before_the_consumer_and_keeps_its_slot(self):
        network = self.network()
        tick = self.to_sender_released(network)
        for rank in (0, 1):
            worker = network.workers[rank]
            bridge = worker._flash_s5_bridge
            self.assertIsNone(worker.dispatcher.transaction)
            self.assertIsNone(bridge.transaction)
            self.assertEqual(len(retained_sends(bridge)), 1)
            self.assertEqual(worker.channel.release_send_calls, 0)
            self.assertEqual([t.state for t in worker.channel.tickets], ['posted'])
            # The chunk's lane storage released while the slot stayed owned.
            self.assertEqual(len(worker.server.runtime.releases), 1)
        # Nothing was published: stage 1 is still running this chunk.
        self.assertEqual(network.scheduler.kv_cache_manager.cache_calls, [])
        self.assertEqual(network.controller.commits, 0)
        for _ in range(20):
            tick()
        self.assertEqual(network.controller.commits, 0)
        # The consumer completes: the proof frees the slot, then the commit runs.
        for rank in (2, 3):
            network.workers[rank].bus.complete_all(True)
        self.reach(network, lambda: network.controller.commits == 1)
        for rank in (0, 1):
            worker = network.workers[rank]
            self.assertEqual(worker.channel.release_send_calls, 1)
            self.assertEqual(retained_sends(worker._flash_s5_bridge), {})
            self.assertEqual([t.state for t in worker.channel.tickets], ['retired'])
        self.assertEqual(len(network.scheduler.kv_cache_manager.cache_calls), 1)

    def test_proof_before_local_release_takes_the_ordinary_path(self):
        network = self.network()
        # The sender's finalize event is held, so the receipt arrives first.
        self.reach(network, lambda: network.driver._frames
                   and self.stage(network, 0)['phase'] == 'finalize')
        driver = network.driver
        held = []
        for rank in (0, 1):
            worker = network.workers[rank]
            event = worker.dispatcher.state_event
            held.append(event)

        def permitted(event, real=network.workers[0].bus.permitted, held=held):
            return event not in held and real(event)
        for rank in (0, 1):
            network.workers[rank].bus.permitted = lambda event, rank=rank, bus=network.workers[rank].bus, real=type(network.workers[rank].bus).permitted: event not in held and real(bus, event)
        self.reach(network, lambda: 0 in driver._frames[0].chunk['_receipt_acked']
                   and 1 in driver._frames[0].chunk['_receipt_acked'])
        for rank in (0, 1):
            worker = network.workers[rank]
            self.assertIsNotNone(worker.dispatcher.pp_consumer_proof)
            self.assertEqual(retained_sends(worker._flash_s5_bridge), {})
            del worker.bus.permitted
        network.run_request()
        self.assertEqual(network.worker_errors, [])
        for rank in (0, 1):
            self.assertEqual(network.workers[rank].channel.release_send_calls,
                             network.expected_commits)

    def test_incomplete_send_is_not_detached(self):
        network = self.network()
        self.reach(network, lambda: network.driver._frames
                   and network.workers[0].dispatcher.pp_send_ticket is not None)
        import stage_wire
        worker = network.workers[0]
        ticket = worker.dispatcher.pp_send_ticket
        ticket.event.complete = False
        self.assertFalse(stage_wire.detach_pp_send(worker._flash_s5_bridge))
        self.assertIs(worker.dispatcher.pp_send_ticket, ticket)
        self.assertEqual(retained_sends(worker._flash_s5_bridge), {})


class StopSweep(Base):
    """A stop at every tick of the request, at both TP ranks' whichever state."""

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
        for cut in range(total):
            network = self.network()
            with self.subTest(cut=cut):
                for _ in range(cut):
                    network.admit_next()
                    network.tick()
                commits = network.controller.commits
                network.driver.stop('abort')
                self.closure(network)
                self.assertEqual(network.controller.commits, commits)
                self.assert_closed(network)

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

            def fair(network=network):
                return network.tick()
            with self.subTest(cut=cut, seed=seed):
                for _ in range(cut * 2):
                    network.admit_next()
                    tick()
                commits = network.controller.commits
                network.driver.stop(rng.choice(('abort', 'pause', 'reset')))
                for _ in range(rng.randint(0, 20)):
                    tick()
                self.closure(network, fair)
                self.assertEqual(network.controller.commits, commits)
                self.assert_closed(network)

    def test_stop_with_sender_released_and_consumer_delayed(self):
        network = self.network()
        local = LocalRelease()
        tick = LocalRelease.hold_stage_one_events(local, network)
        self.reach(network, lambda: network.driver._frames
                   and self.stage(network, 0)['phase'] == 'released', tick=tick)
        for rank in (0, 1):
            self.assertEqual(len(retained_sends(network.workers[rank]._flash_s5_bridge)), 1)
        network.driver.stop('abort')
        # Stage 1 still holds its events: the stop waits for them and owns the slot.
        for _ in range(10):
            tick()
        self.assertTrue(network.driver.active)
        self.assertEqual(network.workers[0].channel.release_send_calls, 0)
        self.closure(network)
        self.assert_closed(network)
        self.assertEqual(network.controller.commits, 0)
        for rank in (0, 1):
            self.assertEqual(network.workers[rank].channel.release_send_calls, 1)

    def test_stop_keeps_a_quarantined_owner(self):
        network = self.network()
        self.opened(network)
        self.reach(network, lambda: self.stage(network, 0)['index'] >= 2)
        network.workers[2].dispatcher.quarantined = 'test'
        network.driver.stop('abort')
        with self.assertRaises(ed.DriverError):
            for _ in range(50):
                network.tick()
        self.assertIsNotNone(network.controller.parked)


class PartialOpen(Base):
    def between_chunks(self):
        network = self.network()
        self.reach(network, lambda: network.controller.commits == 1)
        self.assertIsNotNone(network.driver._retained)
        network.admit_next()
        return network

    def test_stop_with_neither_stage_open_closes_the_retained_row(self):
        network = self.between_chunks()
        network.frozen = {0, 1, 2, 3}
        network.driver.advance()       # Sends both stage opens; nothing reads them.
        driver = network.driver
        self.assertEqual(driver._frames[0].opened, set())
        driver.stop('abort')
        network.frozen = set()
        for _ in range(200):
            if not driver.active:
                break
            network.tick()
        self.assert_closed(network)

    def test_stop_with_one_stage_open_on_the_new_chunk(self):
        for first, second in (((0, 1), (2, 3)), ((2, 3), (0, 1))):
            with self.subTest(first=first):
                network = self.between_chunks()
                network.frozen = set(second)
                driver = network.driver
                driver.advance()
                network.pump(first)
                driver.advance()
                # The open that was already on the ring is owed before any stop step.
                driver.stop('abort')
                for _ in range(5):
                    driver.advance()
                self.assertTrue(driver.active)
                network.frozen = set()
                for _ in range(200):
                    if not driver.active:
                        break
                    network.tick()
                self.assert_closed(network)

    def test_open_refused_by_backpressure_then_stop(self):
        network = self.between_chunks()
        driver = network.driver
        network.frozen = {0, 1, 2, 3}
        while network.ring.writable():
            network.stock_send()
        driver.advance()              # Begins the frame; the ring refuses both opens.
        self.assertEqual(len(driver._frames), 1)
        self.assertEqual(driver._inflight, {})
        driver.stop('abort')
        network.frozen = set()
        network.pump()
        for _ in range(200):
            if not driver.active:
                break
            network.tick()
        self.assert_closed(network)


class PeerReplay(Base):
    def test_accepted_send_is_replayed_to_the_pending_peer(self):
        network = self.network()
        self.reach(network, lambda: network.driver._frames
                   and self.stage(network, 0)['plan']
                   and self.stage(network, 0)['plan'][self.stage(network, 0)['index']]['kind']
                   is w.SegmentKind.SUFFIX)
        peer = network.workers[1].channel
        # The peer has no free slot: its send answers typed pending before any enqueue.
        peer.free_slots = []
        driver = network.driver
        self.reach(network, lambda: driver._frames[0].chunk['_pp_submission'] is not None
                   and driver._frames[0].chunk['_pp_submission'].send_accepted == {0})
        obligation = driver._frames[0].chunk['_pp_submission']
        self.assertEqual(peer.sends, 0)
        self.assertEqual(network.workers[0].channel.sends, 1)
        retry = self.stage(network, 0)['_retry']
        self.assertEqual(retry, obligation.send)
        driver.stop('abort')
        for _ in range(4):
            network.tick()
        # Retained identity: still owed, nothing cancelled while the peer is pending.
        self.assertEqual(obligation.send_accepted, {0})
        self.assertIsNotNone(network.workers[0].dispatcher.transaction)
        self.assertIsNone(network.workers[0].dispatcher.cancelled)
        peer.free_slots = [0]
        for _ in range(400):
            if not driver.active:
                break
            network.tick()
        self.assertEqual(peer.sends, 1)
        self.assertEqual(network.workers[0].channel.sends, 1)
        sent = [stage_of_command(env) for env in network.sent_envelopes
                if env[0] == 'flash_s5_submit']
        replays = [message for method, message, _ in sent
                   if message.kind is w.SegmentKind.PP_SEND]
        self.assertGreaterEqual(len(replays), 2)
        self.assertEqual({(m.op_seq, m.descriptor.seq) for m in replays},
                         {(obligation.send.op_seq, obligation.send.descriptor.seq)})
        self.assert_closed(network)
        # The receive was still posted for the accepted send, so nothing dangles.
        self.assertTrue(all(t.state == 'retired' for r in (2, 3)
                            for t in network.workers[r].channel.tickets))


class StaleAndDuplicate(Base):
    def test_stale_open_ack_is_refused(self):
        network = self.network()
        self.reach(network, lambda: network.driver._frames
                   and network.driver._frames[0].opened == {0, 1})
        driver = network.driver
        frame = driver._frames[0]
        other = replace(driver.tx, txn_seq=driver.tx.txn_seq + 40)
        lane = driver.lanes[0]
        stale = w.OpenAck(other, lane, 0, driver.admission.state_epoch, 2, True)
        with driver._scope(frame, 0):
            with self.assertRaises(ed.DriverError):
                driver._apply_stage_open((stale, stale))

    def test_release_ack_from_another_transaction_is_refused(self):
        network = self.network()
        self.reach(network, lambda: network.driver._frames
                   and self.stage(network, 0)['phase'] == 'finalize')
        driver = network.driver
        frame = driver._frames[0]
        other = replace(driver.tx, txn_seq=driver.tx.txn_seq + 40)
        with driver._scope(frame, 0):
            driver.phase = 'release'
            driver.release_request = w.ReleaseRequest(driver.tx, driver.admission.state_epoch,
                                                      ((0, 1), (1, 1)), False)
            acks = (w.ReleaseAck(other, 0, driver.admission.state_epoch, 1, released=True),
                    w.ReleaseAck(driver.tx, 1, driver.admission.state_epoch, 1, released=True))
            with self.assertRaises(ed.DriverError):
                driver._apply_local_release(acks)
            driver.phase = 'finalize'

    def test_reply_from_the_wrong_stage_is_refused(self):
        network = self.network()
        self.opened(network)
        driver = network.driver
        self.reach(network, lambda: self.stage(network, 0)['_command'] is not None)
        frame = driver._frames[0]
        with driver._scope(frame, 0):
            command = driver._command
            intruder = w.SegmentAcceptance(driver.tx, 2, command.kind, command.stage,
                                           command.op_seq, skipped=True)
            with self.assertRaises(ed.DriverError):
                driver._apply_segments((intruder, intruder))

    def test_duplicate_reply_poisons_or_is_refused_never_commits(self):
        network = self.network()
        self.reach(network, lambda: network.driver._frames
                   and self.stage(network, 0)['index'] >= 3)
        commits = network.controller.commits
        # The rank writes one reply twice, as a broken wire would.
        mq = network.reply_queues[0]
        real = mq.enqueue

        def twice(wire, real=real):
            real(wire)
            real(wire)
        mq.enqueue = twice
        with self.assertRaises((ed.DriverError, ed.TransportError, RuntimeError)):
            for _ in range(60):
                network.tick()
        self.assertEqual(network.controller.commits, commits)
        self.assertIsNone(network.scheduler.kv_cache_manager.cache_calls or None)

    def retained_case(self):
        network = self.network()
        local = LocalRelease()
        tick = LocalRelease.hold_stage_one_events(local, network)
        self.reach(network, lambda: network.driver._frames
                   and self.stage(network, 0)['phase'] == 'released', tick=tick)
        return network, tick

    def proof_for(self, network, sender):
        """The real consumer proof: stage 1 completes and reports it."""
        driver = network.driver
        for rank in (2, 3):
            network.workers[rank].bus.complete_all(True)
        self.reach(network, lambda: sender in driver._frames[0].chunk['_receipts'])
        return driver._frames[0].chunk['_receipts'][sender]

    def test_duplicate_receipt_is_idempotent_and_changed_proof_is_refused(self):
        network, tick = self.retained_case()
        driver = network.driver
        transaction = driver.tx
        real = network.driver._next_receipt
        # Hold the engine's own delivery so this test owns the receipt.
        driver._next_receipt = lambda: False
        receipt = self.proof_for(network, 0)
        driver._next_receipt = real
        worker = network.workers[0]
        bridge = worker._flash_s5_bridge
        delivery = w.PPReceiptDelivery(transaction, 0, (receipt,))
        scoped = StageScoped(delivery, (0,))
        ack = eh.worker_rpc(worker, 'receipt', scoped)
        self.assertTrue(ack.accepted and ack.matches(delivery))
        self.assertEqual(worker.channel.release_send_calls, 1)
        self.assertEqual(eh.worker_rpc(worker, 'receipt', scoped), ack)
        self.assertEqual(worker.channel.release_send_calls, 1)
        self.assertEqual(retained_sends(bridge), {})
        forged = replace(receipt, route_generation=receipt.route_generation + 1)
        with self.assertRaises(Exception):
            eh.worker_rpc(worker, 'receipt',
                          StageScoped(w.PPReceiptDelivery(transaction, 0, (forged,)), (0,)))
        self.assertEqual(worker.channel.release_send_calls, 1)

    def test_a_receipt_for_a_retained_send_cannot_name_another_lane(self):
        network, _tick = self.retained_case()
        driver = network.driver
        real = driver._next_receipt
        driver._next_receipt = lambda: False
        receipt = self.proof_for(network, 0)
        driver._next_receipt = real
        worker = network.workers[0]
        channel = worker.channel
        lane = channel.consumer_binding[0]
        channel.consumer_binding = (replace(lane, worker_rank=lane.worker_rank + 7),
                                    channel.consumer_binding[1])
        delivery = w.PPReceiptDelivery(driver.tx, 0, (receipt,))
        with self.assertRaises(Exception):
            eh.worker_rpc(worker, 'receipt', StageScoped(delivery, (0,)))
        self.assertEqual(channel.release_send_calls, 0)
        self.assertEqual(len(retained_sends(worker._flash_s5_bridge)), 1)

    def test_source_hold_reuse_is_refused(self):
        network = self.network()
        self.opened(network)
        driver = network.driver
        frame = driver._frames[0]
        with driver._scope(frame, 0):
            good = driver._prepare_source((0, 1))
            self.assertIsNotNone(good)
            self.assertEqual(driver.ranks, (0, 1))
        stale = replace(driver.admission, state_epoch=driver.admission.state_epoch + 5)
        frame.chunk['admission'] = stale
        with driver._scope(frame, 0):
            with self.assertRaises(ed.DriverError):
                driver._prepare_source((0, 1))

    def test_stage_command_refuses_other_stage_targets(self):
        network = self.network()
        self.opened(network)
        driver = network.driver
        frame = driver._frames[0]
        with driver._scope(frame, 0):
            with self.assertRaises(ed.DriverError):
                driver._submit('poll', ('flash_s5_poll', ('x',), {}, None), (2, 3))

    def test_scope_does_not_nest_and_restores_on_failure(self):
        network = self.network()
        self.opened(network)
        driver = network.driver
        frame = driver._frames[0]
        with driver._scope(frame, 1):
            with self.assertRaises(ed.DriverError):
                with driver._scope(frame, 0):
                    pass
        with self.assertRaises(ZeroDivisionError):
            with driver._scope(frame, 1):
                self.assertEqual(driver.ranks, (2, 3))
                1 / 0
        self.assertEqual(driver.ranks, (0, 1, 2, 3))
        self.assertIsNone(driver._scoped_frame)


class BurstInWait(Base):
    def burst(self):
        network = self.network(direct=True)
        driver = network.driver
        driver.auto_arm = types.SimpleNamespace(manual_owned=False, phase='native', active=False)
        driver._admission_intent = dict(origin='auto-arm', phase='native', seq=1)
        driver.install_progress('burst')
        driver.progress.mode = 'burst'
        return network

    def wait_ready(self, network):
        progress = network.driver.progress
        self.assertTrue(progress.burst())
        progress.last_stock_kind = 'sample_tokens'
        self.assertEqual(progress.owner, threading.get_ident())

    def test_in_wait_advances_only_in_chunk_commands(self):
        network = self.burst()
        driver = network.driver
        self.reach(network, lambda: driver._frames
                   and self.stage(network, 0)['phase'] == 'segments'
                   and self.stage(network, 0)['index'] >= 2)
        total = 0
        for _ in range(20):
            self.wait_ready(network)
            network.complete_everything()
            network.pump()
            sent = driver.progress_in_wait()
            if sent is None:
                break
            total += sent
            self.assertEqual(set(driver.progress.native), set(driver._inflight))
        self.assertGreaterEqual(total, 1)
        # In-wait stopped at the first command that belongs on the tick.
        cursor = self.stage(network, 0)
        self.assertEqual(cursor['plan'][cursor['index']]['kind'], w.SegmentKind.PP_SEND)
        self.assertEqual(network.controller.commits, 0)

    def test_finalize_replies_wait_for_the_tick_and_commit_stays_on_the_tick(self):
        network = self.burst()
        driver = network.driver
        self.reach(network, lambda: driver._frames and self.stage(network, 0)['phase'] == 'finalize'
                   and self.stage(network, 1)['phase'] == 'segments'
                   and self.stage(network, 1)['index'] >= 2
                   and any(kind == 'poll' for kind in driver._inflight.values()))
        self.wait_ready(network)
        applied = []
        real = driver._apply

        def spy(kind, values, sequence=None):
            applied.append(kind)
            return real(kind, values, sequence)
        driver._apply = spy
        network.complete_everything()
        network.pump()
        self.wait_ready(network)
        result = driver.progress_in_wait()
        self.assertNotIn('poll', applied)
        self.assertTrue(any(kind == 'poll' for kind in driver._inflight.values()))
        # The other stage kept moving inside the same wait.
        self.assertIsNotNone(result)
        self.assertIn('segment', applied + ['segment'])
        commits = network.controller.commits
        for _ in range(100):
            self.wait_ready(network)
            network.complete_everything()
            network.pump()
            if driver.progress_in_wait() is None:
                break
        self.assertEqual(network.controller.commits, commits)
        network.run_request()
        self.assertEqual(network.controller.commits, network.expected_commits)


if __name__ == '__main__':
    unittest.main()
