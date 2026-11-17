#!/usr/bin/env python3
"""inwait build-5 (results/hub/inwait/DESIGN-5.md): the exact successor plan in place of the "may be terminal" stand-in. Discovery mode.

Runs the real overlap network with a stock-shaped split: chunks stop at a last hash boundary, so a successor with ``remaining <= cap`` is not always terminal.
No GPU; what it saves, and whether the real Mamba manager's ``partial-tail-pending`` clause lets it through, is the box's question.
"""
import types
import unittest

import stage_overlap as so
import test_e3_stage_overlap as c1
import test_inwait_engine as ie
from stage_net import ed
import stage_progress as sp
from e3_types import E3ContractError

TOKENS, CAP, BOUNDARY = 33, 4, 30    # ends 4..28, then 30 (the boundary), then 32 (terminal)


class Split:
    """Stock's _mamba_block_aligned_split: a chunk that would cross the last hash boundary stops there."""

    def __call__(self, request, num_new_tokens, num_new_local_computed_tokens=0):
        start = request.num_computed_tokens + num_new_local_computed_tokens
        end = start + num_new_tokens
        return BOUNDARY - start if start < BOUNDARY < end else num_new_tokens


class Exact:
    """The network with a stock-shaped split and a last hash boundary at 30, so the chunk ending at 28 has a successor that is short of the end."""
    exact = True
    hash_size = 10

    def shaped(self, network):
        network.scheduler._mamba_block_aligned_split = Split()
        network.scheduler.kv_cache_manager.block_pool.hash_block_size = self.hash_size
        network.expected_commits = 9
        return network

    def network(self, **kwargs):
        network = self.shaped(super().network(**dict(dict(tokens=TOKENS, chunk=CAP), **kwargs)))
        network.driver.exact_successor = self.exact
        return network


class Connected(Exact, ie.FlagOn, ie.Early, c1.OverlapBase):
    def run_it(self, exact, slow=False):
        self.exact = exact
        network = self.network()
        network.run_request(tick=self.slow_tick(network) if slow else None)
        return network

    def serial_run(self):
        network = self.shaped(self.serial(tokens=TOKENS, chunk=CAP))
        network.run_request()
        return network

    def assert_same_as_serial(self, network):
        serial = self.serial_run()
        self.assertEqual(network.worker_errors, [])
        for rank in (0, 1, 2, 3):
            self.assertEqual(network.ledgers[rank], serial.ledgers[rank], rank)
        self.assertEqual(network.persistent_state(), serial.persistent_state())
        self.assertLessEqual(max(network.inflight_peak.values()), 1, network.inflight_peak)

    def test_on_the_short_successor_overlaps_and_the_bytes_are_serial(self):
        network = self.run_it(True)
        self.assert_same_as_serial(network)
        self.assertEqual(network.controller.exact_counters['exact_admits'], 1)
        self.assertGreater(network.controller.exact_counters['exact_terminal'], 0, 'the terminal chunk still waits')

    def test_on_a_slow_stage_one_also_matches_serial(self):
        network = self.run_it(True, slow=True)
        self.assert_same_as_serial(network)

    def test_on_admits_one_more_successor_than_off(self):
        on, off = self.run_it(True), self.run_it(False)
        self.assertEqual(off.controller.exact_counters, dict(exact_admits=0, exact_terminal=0))
        self.assertEqual(on.controller.overlap_counters['successors'], off.controller.overlap_counters['successors'] + 1)
        self.assert_same_as_serial(off)

    def test_the_last_hash_boundary_still_refuses(self):
        # The chunk ending at 30 reaches the terminal successor: the plan refuses it first. Force it past the plan: the boundary clause reads next.
        controller = so.overlap_refusal
        manager = types.SimpleNamespace(coordinator=types.SimpleNamespace(single_type_managers=()), block_pool=types.SimpleNamespace(hash_block_size=10))
        request = types.SimpleNamespace(request_id='r', num_prompt_tokens=TOKENS, num_tokens=TOKENS, shared_prefix_boundary=None)
        self.assertEqual(controller(manager, request, 30, False, CAP), 'successor-may-be-terminal')
        self.assertEqual(controller(manager, request, 30, False, 0), 'older-at-last-hash-boundary')
        self.assertIsNone(controller(manager, request, 28, False, 0))

    def test_a_plan_that_disagrees_with_the_admission_stops_loudly(self):
        self.exact = True
        self.hash_size = 1      # no boundary clause in the way
        network = self.network()
        network.controller._successor_reaches_end = lambda *args: False
        with self.assertRaises(E3ContractError):
            network.run_request()


class StopSweepExact(Exact, ie.FlagOn, ie.Early, c1.StopSweep):
    pass


class Selection(ie.FlagOn, ie.Early, c1.OverlapBase):
    def idle(self):
        network = self.network()
        network.driver.controller = types.SimpleNamespace(active=False, overlap_counters={}, overlap_refusals={}, exact_counters=dict(exact_admits=0, exact_terminal=0))
        network.driver.engine = types.SimpleNamespace(is_scheduler_paused=lambda: True)
        return network.driver

    def test_selected_by_the_stage_command_and_read_in_status(self):
        driver = self.idle()
        self.assertIs(driver.exact_successor, False)
        base = dict(action=sp.STAGE_CONTROL, local=True, overlap=True, defer_bind=True, inwait=True, early_successor=True)
        self.assertIs(driver._stage_control(dict(base, exact_successor=True))['exact_successor'], True)
        self.assertIs(driver.exact_successor, True)
        status = driver.stage_status()['overlap']['inwait']
        self.assertIs(status['exact_successor'], True)
        self.assertEqual((status['counters']['exact_admits'], status['counters']['exact_terminal']), (0, 0))
        self.assertIs(driver._stage_control(base)['exact_successor'], False, 'a command without the field turns it off')

    def test_refused_when_invalid_or_without_inwait(self):
        driver = self.idle()
        base = dict(action=sp.STAGE_CONTROL, local=True, overlap=True, defer_bind=True)
        for bad in (dict(base, inwait=True, exact_successor=1), dict(base, exact_successor=True), dict(base, inwait=False, exact_successor=True)):
            with self.assertRaises(ed.DriverError):
                driver._stage_control(bad)

    def test_not_installed_refused(self):
        driver = self.idle()
        driver.inwait_installed = False
        with self.assertRaises(ed.DriverError):
            driver._stage_control(dict(action=sp.STAGE_CONTROL, local=True, overlap=True, defer_bind=True, exact_successor=False))


class Rule(unittest.TestCase):
    """The controller's read on its own: the plan is pure, and only the stand-in clause is replaced."""

    def controller(self, split=None, minimum=1):
        controller = so.StagedAdmissionController.__new__(so.StagedAdmissionController)
        controller.minimum_stage_rows = minimum
        controller.exact_counters = dict(exact_admits=0, exact_terminal=0)
        controller.native_hold_s, controller.native_residue = 0, 0      # coverage-soak-fix's native_end reads both
        return controller, types.SimpleNamespace(_mamba_block_aligned_split=split or Split())

    def request(self, computed=28):
        return types.SimpleNamespace(request_id='r', num_tokens=TOKENS, num_prompt_tokens=TOKENS, num_computed_tokens=computed)

    def test_a_successor_stopped_by_the_boundary_is_not_terminal(self):
        controller, scheduler = self.controller()
        self.assertFalse(controller._successor_reaches_end(scheduler, self.request(), 28, CAP))

    def test_a_successor_reaching_the_end_is_terminal(self):
        controller, scheduler = self.controller()
        self.assertTrue(controller._successor_reaches_end(scheduler, self.request(), 30, CAP))

    def test_no_split_hook_means_the_plain_cap_decides(self):
        controller, _ = self.controller()
        scheduler = types.SimpleNamespace()
        self.assertTrue(controller._successor_reaches_end(scheduler, self.request(), 28, CAP))

    def test_the_staged_plan_decides_when_rows_have_a_minimum(self):
        controller, scheduler = self.controller(minimum=2)
        self.assertFalse(controller._successor_reaches_end(scheduler, self.request(), 28, CAP), 'plan 2, 2')
        self.assertTrue(controller._successor_reaches_end(scheduler, self.request(), 30, CAP), 'plan 2')

    def test_no_plan_is_treated_as_terminal(self):
        controller, scheduler = self.controller(minimum=3)
        self.assertTrue(controller._successor_reaches_end(scheduler, self.request(), 28, CAP), '2 then 2 is under the minimum, 4 crosses the boundary')

    def test_the_plan_leaves_the_request_unchanged(self):
        controller, scheduler = self.controller()
        request = self.request()
        before = dict(vars(request))
        controller._successor_reaches_end(scheduler, request, 28, CAP)
        self.assertEqual(vars(request), before)


class HoldView(unittest.TestCase):
    """coverage-throughput: under the hold the exact plan reads the admission's view (plan_view), at production geometry.

    Soak (q18a-compose-soak, ANSWERS Q1): the chunk before a short hash tail planned its successor over the raw request, which has
    no plan there; the search ran to its 65,536-query bound on every engine tick (~250 ms a step, ~4.3 s a chunk) and named a
    non-terminal successor terminal, so the early successor never opened.

    Build-2 (c), option A, ends such a turn at its last 176-row unit (a geometry tail), so the hold no longer moves this stop.
    The pause-fix checks run under the block rule (build-1's geometry_tail_rows, ``block_rule``), where it still does; the
    unit tail's own answer is in ``test_unit_tail_*``.
    """
    PROMPT, START = 37362, 36432        # Q1's txn 478 -> its successor: 929 rows left, tail 49 rows past the last 176-row unit

    def setUp_block_rule(self):
        import scheduler_hooks as sh
        from unittest import mock

        def block_rule(scheduler, request, minimum):
            block = scheduler.cache_config.block_size
            r = request.num_tokens % block
            return r if 2 <= r <= min(minimum, sh.GEOMETRY_TAIL_MAX) and request.num_tokens - r > 0 else 0
        patcher = mock.patch.object(sh, 'geometry_tail_rows', block_rule)
        patcher.start()
        self.addCleanup(patcher.stop)

    def setUp(self):
        import test_coverage_residue as tcr
        host = types.SimpleNamespace(**tcr.split_host())
        self.queries = 0
        def split(request, num_new_tokens, num_new_local_computed_tokens=0):
            self.queries += 1
            return tcr.SPLIT(host, request, num_new_tokens, num_new_local_computed_tokens=num_new_local_computed_tokens)
        self.scheduler = types.SimpleNamespace(_mamba_block_aligned_split=split, **vars(host))
        self.cap = so.bounded_limit([52624, 1024], self.START, so.native_chunk_cap(2048, host.max_num_scheduled_tokens))

    def controller(self, hold):
        controller = so.StagedAdmissionController.__new__(so.StagedAdmissionController)
        controller.minimum_stage_rows, controller.chunk_bound = 64, [52624, 1024]
        controller.native_hold_s, controller.native_residue = hold, 1
        controller.exact_counters = dict(exact_admits=0, exact_terminal=0)
        return controller

    def request(self):
        return types.SimpleNamespace(request_id='r', num_tokens=self.PROMPT, num_prompt_tokens=self.PROMPT, num_computed_tokens=0, shared_prefix_boundary=0)

    def test_under_the_hold_the_successor_is_not_terminal_and_the_plan_is_cheap(self):
        self.setUp_block_rule()
        self.assertFalse(self.controller(600)._successor_reaches_end(self.scheduler, self.request(), self.START, self.cap))
        self.assertLess(self.queries, 16, 'the moved view plans in a few queries, not the 65,536 bound')

    def test_the_answer_is_the_plan_admission_makes(self):
        self.setUp_block_rule()
        controller, request = self.controller(600), self.request()
        view = controller.plan_view(self.scheduler, request)
        self.assertIsNot(view, request, 'the hold moves the stop on a short tail')
        end = controller.native_end(self.scheduler, request)
        plan = so.staged_chunk_plan(self.START, end, self.cap, 64, lambda p, l: controller._aligned_split(
            self.scheduler, so._ProjectedPrefillRequest(view, p), so.bounded_limit(controller.chunk_bound, p, l)))
        self.assertEqual(plan, (704, 225))
        self.assertFalse(controller._successor_reaches_end(self.scheduler, request, self.START, self.cap))

    def test_hold_off_keeps_the_raw_view(self):
        controller, request = self.controller(0), self.request()
        self.assertIs(controller.plan_view(self.scheduler, request), request)

    def test_the_same_inputs_are_answered_once(self):
        self.setUp_block_rule()
        controller, request = self.controller(600), self.request()
        first = controller._successor_reaches_end(self.scheduler, request, self.START, self.cap)
        asked = self.queries
        self.assertEqual(controller._successor_reaches_end(self.scheduler, request, self.START, self.cap), first)
        self.assertEqual(self.queries, asked, 'the parked chunk asks every tick; the second ask is the kept answer')
        self.assertTrue(controller._successor_reaches_end(self.scheduler, request, 37136, self.cap), 'a new start plans again: 225 rows, terminal')
        self.assertGreater(self.queries, asked)
        other = self.request()
        controller._successor_reaches_end(self.scheduler, other, self.START, self.cap)
        self.assertGreater(self.queries, asked + 1, 'another request object plans again')

    def test_unit_tail_ends_the_turn_at_the_last_unit_and_the_view_stays(self):
        import scheduler_hooks as sh
        controller, request = self.controller(600), self.request()
        self.assertEqual(sh.geometry_tail_rows(self.scheduler, request, 64), self.PROMPT % 176)
        self.assertEqual(controller.native_end(self.scheduler, request), self.PROMPT - self.PROMPT % 176)
        self.assertIs(controller.plan_view(self.scheduler, request), request, 'a geometry tail: the stop is not moved')

    def test_unit_tail_successor_is_the_one_chunk_to_the_last_unit_and_cheap(self):
        controller, request = self.controller(600), self.request()
        end = controller.native_end(self.scheduler, request)
        plan = so.staged_chunk_plan(self.START, end, self.cap, 64, lambda p, l: controller._aligned_split(
            self.scheduler, so._ProjectedPrefillRequest(request, p), so.bounded_limit(controller.chunk_bound, p, l)))
        self.assertEqual(plan, (end - self.START,))
        self.assertTrue(controller._successor_reaches_end(self.scheduler, request, self.START, self.cap))
        self.assertLess(self.queries, 16)


if __name__ == '__main__':
    unittest.main()
