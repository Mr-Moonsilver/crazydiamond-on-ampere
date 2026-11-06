#!/usr/bin/env python3
"""parity-2 A (tmp/decode-phase/READING-lane-pace.md §2): the terminal successor admitted beside commit(n). Discovery mode.

On c64 the stage-overlap successor was refused on 193 second-to-last chunks (the successor is the turn's terminal chunk):
stage 0 waited for commit on both stages, edge 267-324 ms against ~40. With ``terminal_successor`` selected (it needs the
exact successor), the exact plan's terminal answer admits n+1 too. It commits after n, alone, on the ordinary path: its
release closes the request row and its commit hands back on FULL, as a serial terminal chunk's.

Real composition: test_e3_stage_overlap.OverlapStack (real controller, overlap driver, reply transport, worker bridge, PP
channel) with inwait, the early successor and the exact successor selected, as every compose-5 arm runs. No GPU.
"""
import types
import unittest

import stage_overlap as so
import stage_progress as sp
import test_e3_stage_overlap as c1
import test_inwait_engine as ie
import test_inwait_exact as ix
from stage_net import ed


class Terminal:
    """The overlap network with the exact successor and, per class, the terminal successor."""
    exact = True
    terminal = True

    def network(self, **kwargs):
        network = super().network(**kwargs)
        network.driver.exact_successor = self.exact
        network.driver.terminal_successor = self.terminal
        return network


class Connected(Terminal, ie.FlagOn, ie.Early, c1.OverlapBase):
    def run_it(self, terminal, slow=False, **kwargs):
        self.terminal = terminal
        network = self.network(**kwargs)
        seen = []
        tick = self.slow_tick(network) if slow else network.tick

        def spy():
            out = tick()
            record = network.controller.successor
            if record is not None and record.admission is not None and record.admission.terminal and len(network.driver._frames) == 2:
                seen.append(network.driver._frames[1].chunk['tx'].chunk_end)
            return out
        network.run_request(tick=spy)
        network.terminal_in_flight = seen
        return network

    def assert_same_as_serial(self, network, **kwargs):
        serial = self.serial(**kwargs)
        serial.run_request()
        self.assertEqual(network.worker_errors, [])
        for rank in (0, 1, 2, 3):
            self.assertEqual(network.ledgers[rank], serial.ledgers[rank], rank)
        self.assertEqual(network.persistent_state(), serial.persistent_state())
        self.assertLessEqual(max(network.inflight_peak.values()), 1, network.inflight_peak)
        self.assertEqual(network.controller.commits, serial.controller.commits)

    def test_on_the_terminal_successor_overlaps_and_the_bytes_are_serial(self):
        network = self.run_it(True)
        self.assert_same_as_serial(network)
        self.assertEqual(network.controller.terminal_counters, dict(terminal_admits=1, terminal_mismatch=0))
        self.assertEqual(set(network.terminal_in_flight), {32}, 'the terminal chunk ran as the second frame')
        self.assert_closed(network)

    def test_on_a_slow_stage_one_also_matches_serial(self):
        network = self.run_it(True, slow=True)
        self.assert_same_as_serial(network)
        self.assertEqual(network.controller.terminal_counters['terminal_admits'], 1)

    def test_off_the_terminal_chunk_waits_and_admits_one_successor_less(self):
        on, off = self.run_it(True), self.run_it(False)
        self.assertEqual(off.controller.terminal_counters, dict(terminal_admits=0, terminal_mismatch=0))
        self.assertEqual(off.terminal_in_flight, [])
        self.assertEqual(on.controller.overlap_counters['successors'], off.controller.overlap_counters['successors'] + 1)
        self.assertEqual(on.controller.exact_counters['exact_terminal'], 1, 'the plan names the terminal chunk once, and it is admitted')
        self.assertGreater(off.controller.exact_counters['exact_terminal'], 1, 'off, the parked chunk asks again every tick')
        self.assert_same_as_serial(off)

    def test_the_handback_follows_the_terminal_commit_once(self):
        for terminal in (True, False):
            with self.subTest(terminal=terminal):
                network = self.run_it(terminal)
                queued = [r for r in network.scheduler.waiting.items if r.request_id == c1.r4.UID]
                self.assertEqual(len(queued), 1, 'one handback, after the terminal commit')
                self.assertEqual(queued[0].num_computed_tokens, network.tokens - 1)
                self.assertIsNone(network.controller.parked)
                self.assertIsNone(network.controller.successor)

    def test_without_the_exact_successor_nothing_changes(self):
        self.exact = False
        on = self.run_it(True)
        self.exact = False
        off = self.run_it(False)
        self.assertEqual(on.controller.terminal_counters, off.controller.terminal_counters)
        self.assertEqual(on.controller.overlap_counters, off.controller.overlap_counters)
        self.assertEqual(on.controller.overlap_refusals, off.controller.overlap_refusals)

    def test_a_plan_that_names_a_nonterminal_chunk_terminal_is_counted_not_raised(self):
        self.terminal = True
        network = self.network()
        ix.Exact.shaped(types.SimpleNamespace(hash_size=1), network)
        network.controller._successor_reaches_end = lambda *args: True
        network.run_request()
        counters = network.controller.terminal_counters
        self.assertGreater(counters['terminal_mismatch'], 0)
        self.assertEqual(counters['terminal_admits'], 1, 'the real terminal chunk is still named so')
        self.assertEqual(network.worker_errors, [])


class ShapedConnected(Terminal, ie.FlagOn, ie.Early, c1.OverlapBase):
    """inwait build-5's stock-shaped split: a last hash boundary at 30 refuses the terminal successor before this rule."""

    def network(self, **kwargs):
        network = super().network(**dict(dict(tokens=ix.TOKENS, chunk=ix.CAP), **kwargs))
        network.scheduler._mamba_block_aligned_split = ix.Split()
        network.scheduler.kv_cache_manager.block_pool.hash_block_size = self.hash_size
        network.expected_commits = 9
        return network

    hash_size = 10

    def test_the_last_hash_boundary_still_refuses_the_terminal_successor(self):
        network = self.network()
        network.run_request()
        self.assertEqual(network.controller.terminal_counters, dict(terminal_admits=0, terminal_mismatch=0))
        self.assertGreater(network.controller.overlap_refusals['older-at-last-hash-boundary'], 0)

    def test_without_the_boundary_the_terminal_successor_is_admitted(self):
        self.hash_size = 1
        network = self.network()
        network.run_request()
        self.assertEqual(network.controller.terminal_counters, dict(terminal_admits=1, terminal_mismatch=0))
        self.assertEqual(network.controller.exact_counters['exact_admits'], 1)
        self.assertEqual(network.worker_errors, [])


class StopSweepTerminal(Terminal, ie.FlagOn, ie.Early, c1.StopSweep):
    def test_stops_land_while_the_terminal_successor_is_in_flight(self):
        total = self.tick_count()
        landed = 0
        for cut in range(total):
            network = self.network()
            for _ in range(cut):
                network.admit_next()
                network.tick()
            record = network.controller.successor
            if record is None or record.admission is None or not record.admission.terminal:
                continue
            landed += 1
            with self.subTest(cut=cut):
                commits = network.controller.commits
                network.driver.stop('abort')
                self.closure(network)
                self.assertEqual(network.controller.commits, commits)
                self.assert_closed(network)
        self.assertGreater(landed, 3, 'the sweep reached the terminal successor')


class StopStatesTerminal(Terminal, ie.FlagOn, ie.StopStatesEarly):
    """The pair's internal boundaries (the early classes' own, with the terminal successor on), and two reached with the
    terminal chunk as n+1. Both stages on a terminal n+1 with n uncommitted is the inherited test's state (skipped there when
    this fixture commits n at stage 1's release)."""

    def reach_terminal(self, network, predicate, tick=None):
        driver = network.driver
        self.reach(network, lambda: self.two_frames(network) and driver._frames[1].chunk['admission'] is not None
                   and driver._frames[1].chunk['admission'].terminal and predicate(), tick=tick)

    def test_terminal_successor_open_in_flight(self):
        network = self.network()
        self.reach_terminal(network, lambda: True)
        network.frozen = {0, 1}
        for _ in range(3):
            network.tick()
        self.finish(network)

    def test_terminal_successor_bound_with_stage_one_on_the_older(self):
        network = self.network()
        driver = network.driver
        self.reach_terminal(network, lambda: driver._frames[1].chunk['bound_ticket'] is not None and 1 not in driver._frames[0].released,
                            tick=self.slow_tick(network))
        self.finish(network)


class Selection(ie.FlagOn, ie.Early, c1.OverlapBase):
    def idle(self):
        network = self.network()
        network.driver.controller = types.SimpleNamespace(active=False, overlap_counters={}, overlap_refusals={},
                                                          exact_counters=dict(exact_admits=0, exact_terminal=0),
                                                          terminal_counters=dict(terminal_admits=0, terminal_mismatch=0))
        network.driver.engine = types.SimpleNamespace(is_scheduler_paused=lambda: True)
        return network.driver

    base = dict(action=sp.STAGE_CONTROL, local=True, overlap=True, defer_bind=True, inwait=True, early_successor=True)

    def test_selected_by_the_stage_command_and_read_in_status(self):
        driver = self.idle()
        self.assertIs(driver.terminal_successor, False)
        reply = driver._stage_control(dict(self.base, exact_successor=True, terminal_successor=True))
        self.assertIs(reply['terminal_successor'], True)
        self.assertIs(driver.terminal_successor, True)
        status = driver.stage_status()['overlap']
        self.assertEqual(status['terminal_successor'], dict(selected=True, counters=dict(terminal_admits=0, terminal_mismatch=0)))
        self.assertEqual({k: status['inwait']['counters'][k] for k in ('terminal_admits', 'terminal_mismatch')}, dict(terminal_admits=0, terminal_mismatch=0),
                         'the arm records read them from the inwait counters')
        self.assertIs(driver._stage_control(dict(self.base, exact_successor=True))['terminal_successor'], False, 'a command without the field turns it off')

    def test_off_with_the_exact_successor_off(self):
        driver = self.idle()
        driver._stage_control(dict(self.base, exact_successor=True, terminal_successor=True))
        self.assertIs(driver._stage_control(dict(self.base, exact_successor=False, terminal_successor=False))['terminal_successor'], False)

    def test_refused_when_invalid_or_without_the_exact_successor(self):
        driver = self.idle()
        for bad in (dict(self.base, exact_successor=True, terminal_successor=1),
                    dict(self.base, terminal_successor=True),
                    dict(self.base, exact_successor=False, terminal_successor=True)):
            with self.subTest(bad=bad):
                with self.assertRaises(ed.DriverError):
                    driver._stage_control(bad)
        self.assertIs(driver.terminal_successor, False, 'nothing applied on a refusal')

    def test_not_installed_refused(self):
        driver = self.idle()
        driver.inwait_installed = False
        with self.assertRaises(ed.DriverError):
            driver._stage_control(dict(action=sp.STAGE_CONTROL, local=True, overlap=True, defer_bind=True, terminal_successor=False))


class Rule(unittest.TestCase):
    """The admission's read on its own: off is today's refusal, on is an admission with the other clauses still read."""

    def test_the_refusal_names_and_order_are_unchanged(self):
        self.assertEqual(so.REFUSALS, ('older-terminal', 'successor-may-be-terminal', 'older-at-last-hash-boundary',
                                       'older-at-shared-prefix-boundary', 'partial-tail-pending', 'manager-shape'))

    def test_a_terminal_older_still_refuses(self):
        manager = types.SimpleNamespace(coordinator=types.SimpleNamespace(single_type_managers=()), block_pool=types.SimpleNamespace(hash_block_size=1))
        request = types.SimpleNamespace(request_id='r', num_prompt_tokens=33, num_tokens=33, shared_prefix_boundary=None)
        self.assertEqual(so.overlap_refusal(manager, request, 32, True, 0), 'older-terminal')


if __name__ == '__main__':
    unittest.main()
