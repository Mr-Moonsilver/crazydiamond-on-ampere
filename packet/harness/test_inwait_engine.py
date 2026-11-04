#!/usr/bin/env python3
"""inwait: native handoffs inside the stock wait (FLASH_S5_INWAIT, default off; results/hub/inwait/DESIGN.md). Discovery mode: the new logic only.

Real composition: test_e3_stage_overlap.OverlapStack on the burst path (test_alpha_next_c1_send_gate): real controller, overlap driver, reply transport, worker bridge, dispatcher, PP channel.
"""
import logging
import os
import random
import time
import types
import unittest
from unittest import mock

import test_alpha_next_c1_composed_fields as cf
import test_alpha_next_c1_send_gate as sg
import test_e3_stage_overlap as c1
from stage_net import stage_of_command

import engine_driver as ed
import executor_hooks as eh
import native_measurement as nm
import stage_overlap as so
import stage_progress as sp
from reply_transport import Backpressure
from wire_v1 import SegmentKind

FLAG = {sp.STAGE_LOCAL_ENV: '1', so.STAGE_OVERLAP_ENV: '1', so.DEFER_BIND_ENV: '1', nm.INWAIT_ENV: '1'}


class FlagOn:
    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ, FLAG)
        patcher.start()
        self.addCleanup(patcher.stop)


class Selected:
    select = True

    def network(self, **kwargs):
        network = super().network(**kwargs)
        driver = network.driver
        driver.defer_bind = True
        driver.inwait_installed = True
        driver.inwait = self.select and bool(driver.overlap_selected)
        return network


class IdentityOn(Selected, c1.Identity):
    pass


class StopSweepOn(Selected, c1.StopSweep):
    pass


class StopStatesOn(Selected, c1.StopStates):
    pass


class Base(FlagOn, Selected, sg.Base):
    def run_in_wait(self, select=True, **kwargs):
        network = self.burst(self.network(direct=True))
        network.driver.inwait = select
        network.run_request(tick=self.in_wait_tick(network, **kwargs))
        return network

    def same_as_serial(self, network):
        self.assertIsNone(network.driver.error)
        self.assertEqual(network.controller.commits, network.expected_commits)
        self.assert_serial(network)


class Widened(Base):
    def test_every_handoff_kind_runs_in_the_wait_and_the_ledgers_equal_serial(self):
        network = self.burst(self.network(direct=True))
        spies = sg.Spies(network.driver)
        network.run_request(tick=self.in_wait_tick(network))
        counters = network.driver.inwait_counters
        applied = set(counters['wait_applied'])
        self.assertLessEqual({'prepare', 'pp_send', 'pp_receive', 'state_finalize', 'open', 'bind', 'receipt', 'release'}, applied, counters)
        self.assertLessEqual({'prepare', 'pp_send', 'pp_receive', 'state_finalize'}, set(counters['wait_sent']), counters)
        self.same_as_serial(network)
        self.assertLessEqual(max(network.inflight_peak.values()), 1)
        self.assertTrue(spies.sends)
        self.assertEqual(spies.unbound_sends(), [])

    def test_commit_and_successor_begin_stay_on_the_tick(self):
        network = self.burst(self.network(direct=True))
        driver = network.driver
        rows = []
        commit, begin, maybe = network.controller.commit, driver._begin_successor, driver._maybe_commit

        def spy(name, real):
            def call(*args, **kwargs):
                rows.append((name, driver._in_wait))
                return real(*args, **kwargs)
            return call
        network.controller.commit = spy('commit', commit)
        driver._begin_successor = spy('begin', begin)
        driver._maybe_commit = spy('maybe', maybe)
        network.run_request(tick=self.in_wait_tick(network))
        self.assertEqual({row[0] for row in rows}, {'commit', 'begin', 'maybe'})
        self.assertEqual([row for row in rows if row[1]], [])

    def test_stage_zero_prepare_binds_the_real_controller_in_the_wait(self):
        network = self.burst(self.network(direct=True))
        driver = network.driver
        rows = []
        real = network.controller.bind

        def bind(ticket):
            rows.append(driver._in_wait)
            return real(ticket)
        network.controller.bind = bind
        network.run_request(tick=self.in_wait_tick(network))
        self.assertIn(True, rows)
        self.same_as_serial(network)

    def test_a_wait_send_keeps_each_gate(self):
        """PP_SEND still waits for stage 1's PREPARE reply; PP_RECEIVE for the accepted sends; n+1's bind for n's receipts."""
        network = self.burst(self.network(direct=True))
        spies = sg.Spies(network.driver)
        network.run_request(tick=self.in_wait_tick(network, waits=8, period=2))
        self.same_as_serial(network)
        self.assertTrue(spies.sends)
        self.assertEqual(spies.unbound_sends(), [])

    def test_adversarial_timing_equals_serial(self):
        for seed in range(6):
            rng = random.Random(seed)
            network = self.burst(self.network(direct=True))
            driver = network.driver

            def tick(network=network, rng=rng, driver=driver):
                network.admit_next()
                for rank in (0, 1, 2, 3):
                    if rng.random() < 0.7:
                        network.workers[rank].bus.complete_all(True)
                    for _ in range(rng.randint(0, 2)):
                        network.step(rank)
                network.settle()
                for _ in range(rng.randint(0, 4)):
                    if not driver._frames:
                        break
                    self.wait_ready(network)
                    if driver.progress_in_wait() is None:
                        break
                return driver.advance()
            with self.subTest(seed=seed):
                network.run_request(limit=8000, tick=tick)
                self.same_as_serial(network)

    def test_stop_at_every_tick_of_the_wait_path_closes(self):
        for stop_at in (5, 11, 17, 23, 31, 41, 53, 67, 83, 101, 131, 167):
            for kind in ('abort', 'cancel'):
                with self.subTest(stop_at=stop_at, kind=kind):
                    network = self.burst(self.network(direct=True))
                    driver = network.driver
                    tick = self.in_wait_tick(network)
                    count = dict(n=0)

                    def reached():
                        count['n'] += 1
                        return count['n'] >= stop_at
                    self.reach(network, reached, limit=2000, tick=tick)
                    driver.stop(kind)
                    for _ in range(3000):
                        if not driver.active:
                            break
                        tick()
                    self.assertFalse(driver.active, network.status_text())
                    self.assert_closed(network)


class Direct(Base):
    def wire(self, network):
        rows = []
        for envelope in network.sent_envelopes:
            method, _message, ranks = stage_of_command(envelope)
            if method not in ('flash_s5_open', 'flash_s5_release'):
                continue
            kwargs = envelope[2]
            direct = type(kwargs) is dict and nm.PROGRESS_META_KEY in kwargs and nm.decode_progress_meta(kwargs[nm.PROGRESS_META_KEY])[1]
            rows.append((method, tuple(ranks) if ranks is not None else None, bool(direct)))
        return rows

    def test_stage_one_open_and_release_only(self):
        network = self.run_in_wait()
        rows, counters = self.wire(network), network.driver.inwait_counters
        direct = [row for row in rows if row[2]]
        self.assertEqual({row[1] for row in direct}, {(2, 3)})
        self.assertEqual(sum(row[0] == 'flash_s5_open' for row in direct), counters['direct_opens'])
        self.assertEqual(sum(row[0] == 'flash_s5_release' for row in direct), counters['direct_releases'])
        self.assertGreater(counters['direct_releases'], 0)
        self.assertLess(counters['direct_releases'], network.expected_commits, 'the terminal release stays ordered')
        self.assertFalse(rows[-1][2], rows[-1])
        self.same_as_serial(network)

    def test_off_nothing_new_is_direct(self):
        network = self.run_in_wait(select=False)
        self.assertEqual([row for row in self.wire(network) if row[2]], [])
        self.same_as_serial(network)

    def test_worker_refuses_the_bit_without_the_flag(self):
        proc = types.SimpleNamespace(rank=2)
        meta = {nm.PROGRESS_META_KEY: nm.encode_progress_meta(7, True)}
        with mock.patch.dict(os.environ, {nm.INWAIT_ENV: '0'}):
            with self.assertRaises(ValueError):
                eh.worker_rpc_begin(proc, 'flash_s5_open', dict(meta))
        with mock.patch.dict(os.environ, {nm.INWAIT_ENV: '1'}):
            with self.assertRaises(ValueError):
                eh.worker_rpc_begin(proc, 'flash_s5_cancel', dict(meta))
            self.assertTrue(eh.worker_rpc_begin(proc, 'flash_s5_open', dict(meta))[4])


class Refused(Base):
    def test_a_receipt_refused_in_the_wait_is_retried_counted_and_logged(self):
        for sender in (0, 1):
            with self.subTest(sender=sender):
                network = self.burst(self.network(direct=True))
                driver = network.driver
                real = driver._check_headroom
                seen = dict(refused=0, in_wait=0)

                def check(ring, envelope):
                    if not seen['refused'] and envelope[0] == 'flash_s5_receipt' and envelope[3] == sender and driver._in_wait:
                        seen['refused'] = seen['in_wait'] = 1
                        with mock.patch.object(sp, 'ring_has_headroom', return_value=False), mock.patch.object(sp, 'ring_limiting_readers', return_value=()):
                            return real(ring, envelope)
                    return real(ring, envelope)
                driver._check_headroom = check
                with self.assertLogs(level=logging.INFO) as logs:
                    network.run_request(tick=self.in_wait_tick(network))
                self.assertEqual(seen['in_wait'], 1, 'the receipt went in the wait')
                # The ring model's slow stage 1 refuses other sends by itself, so these are lower bounds.
                self.assertGreaterEqual(driver.inwait_refused.get('flash_s5_receipt', 0), 1)
                self.assertGreaterEqual(driver.inwait_counters['refused_in_wait'], 1)
                self.assertTrue(any('flash-s5 inwait' in line and 'flash_s5_receipt' in line for line in logs.output), logs.output[-3:])
                self.same_as_serial(network)

    def test_a_refused_release_burns_no_attempt(self):
        network = self.burst(self.network(direct=True))
        driver = network.driver
        real = driver._check_headroom
        seen = dict(refused=0)

        def check(ring, envelope):
            if envelope[0] == 'flash_s5_release' and seen['refused'] < 2 * driver.stop_budget:
                seen['refused'] += 1
                raise Backpressure('ring_headroom')
            return real(ring, envelope)
        driver._check_headroom = check
        network.run_request(limit=20000, tick=self.in_wait_tick(network))
        self.assertEqual(seen['refused'], 2 * driver.stop_budget)
        self.assertGreater(driver.inwait_counters['refused_releases'], 0)
        self.same_as_serial(network)

    def test_a_refusal_does_not_end_the_wait(self):
        network = self.burst(self.network(direct=True))
        driver = network.driver
        self.reach(network, lambda: bool(driver._frames) and driver._frames[0].stage[0]['phase'] == 'segments', limit=2000, tick=self.in_wait_tick(network))
        self.wait_ready(network)
        with mock.patch.object(type(driver), '_drive_stage', side_effect=Backpressure('ring')):
            result = driver.progress_in_wait()
        self.assertEqual(result, 0)
        self.assertIsNone(driver.error)


class Pacing(Base):
    def test_a_stage_polls_at_most_once_per_gap_and_polls_are_not_sends(self):
        network = self.burst(self.network(direct=True))
        driver = network.driver
        stamps = {0: [], 1: []}
        real = driver._submit_observed

        def submit(kind, envelope, ranks, **kwargs):
            sequence = real(kind, envelope, ranks, **kwargs)
            if kind == 'poll' and driver._in_wait:
                stamps[driver._scoped_stage].append(time.monotonic_ns())
            return sequence
        driver._submit_observed = submit
        network.run_request(tick=self.in_wait_tick(network))
        for rows in stamps.values():
            self.assertTrue(all(b - a >= so.INWAIT_POLL_GAP_NS for a, b in zip(rows, rows[1:])))
        self.same_as_serial(network)

    def test_a_paced_poll_returns_no_send(self):
        network = self.burst(self.network(direct=True))
        driver = network.driver
        self.reach(network, lambda: any(f.stage[s]['phase'] == 'finalize' and not any(driver._route_stage(q) == s for q in driver._inflight) for f in driver._frames for s in (0, 1)),
                   limit=4000, tick=self.in_wait_tick(network))
        driver._inwait_poll_ns = {0: time.monotonic_ns() + 10 ** 10, 1: time.monotonic_ns() + 10 ** 10}
        polls = sum(kind == 'poll' for kind in driver._inflight.values())
        self.wait_ready(network)
        result = driver.progress_in_wait()
        self.assertIsNotNone(result)
        self.assertLessEqual(sum(kind == 'poll' for kind in driver._inflight.values()), polls)


class Control(FlagOn, sg.Base):
    def setup_idle(self):
        network = self.network()
        network.driver.controller = types.SimpleNamespace(active=False, overlap_counters={}, overlap_refusals={})
        network.driver.engine = types.SimpleNamespace(is_scheduler_paused=lambda: True)
        return network, network.driver

    def test_field_rules(self):
        network, driver = self.setup_idle()
        base = dict(action=sp.STAGE_CONTROL, local=True, overlap=True, defer_bind=True)
        reply = driver._stage_control(dict(base, inwait=True))
        self.assertIs(reply['inwait'], True)
        self.assertIs(driver.inwait, True)
        self.assertIs(driver._stage_control(dict(base, inwait=False))['inwait'], False)
        for bad in (dict(base, inwait=1), dict(base, inwait=True, defer_bind=False), dict(base, inwait=True, overlap=False)):
            with self.assertRaises(ed.DriverError):
                driver._stage_control(bad)
        self.assertIs(driver.inwait, False)
        self.assertTrue(driver.stage_status()['overlap']['inwait']['installed'])

    def test_not_installed_refused_and_no_status_block(self):
        with mock.patch.dict(os.environ, {nm.INWAIT_ENV: '0'}):
            network, driver = self.setup_idle()
            self.assertFalse(driver.inwait_installed)
            self.assertNotIn('inwait', driver.stage_status()['overlap'])
            with self.assertRaises(ed.DriverError) as caught:
                driver._stage_control(dict(action=sp.STAGE_CONTROL, local=True, overlap=True, defer_bind=True, inwait=True))
            self.assertIn('not-installed', str(caught.exception))

    def test_bad_flag_is_refused(self):
        with mock.patch.dict(os.environ, {nm.INWAIT_ENV: '2'}):
            with self.assertRaises(ed.DriverError):
                so.inwait_enabled(os.environ)
        with mock.patch.dict(os.environ, {nm.INWAIT_ENV: '1', so.DEFER_BIND_ENV: '0'}):
            with self.assertRaises(ed.DriverError):
                so.inwait_enabled(os.environ)


class ComposedArm(Base):
    """Fused, layer group 2, as every box arm runs."""
    GROUP = 2

    def network(self, **kwargs):
        return cf.Fixture.composed(self, group=self.GROUP, network=super().network(**kwargs))

    def assert_serial(self, network, **kwargs):
        self.assertEqual(network.worker_errors, [])
        self.assert_closed(network)

    def test_the_arm_equals_serial(self):
        self.same_as_serial(self.run_in_wait())


if __name__ == '__main__':
    unittest.main()
