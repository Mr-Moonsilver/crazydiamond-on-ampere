"""lane-o3 (7 Nov 2026, DESIGN-multi-prompt-lane O3): stay split across runs, on the append-yield Rig (real policy, real
NativeSegmentDriver / StagedSegmentDriver, real controller and scheduler seams, four real bridge workers).

With FLASH_S5_O3=1 (policy.o3) a native run whose terminal commit leaves one prompt row keeps the split route:
  - the driver owes no FULL return and disarms the run's intent (engine_driver._o3_keep_split);
  - the positive-frontier seam lets exactly that row through under split (scheduler_hooks, driver.o3_handback_permits);
  - the policy's split tail waits for stock to schedule it, then selects the next head on the same route, drains the
    in-flight handback step, proves the cohort with a census round trip on split (route split, same generation, live rows
    == cohort, prefill complete) and arms; no barrier, identity, enter_split or FULL transition;
  - the worker classifies that step's batch decode-shaped (graph_hooks.handback_uniform), so it dispatches FULL at n+1.
Refusals keep today's FULL cycle: off, geometry tails, yields, counts not proven, a ledger miss, no next head (FULL after the
handback step), a census mismatch, a handback left unscheduled past O3_TAIL_TICKS.

python3 -m unittest test_lane_o3 (from harness/). No GPU claim: the descriptor's numerics and stream order are the box's.
"""
from types import SimpleNamespace as NS
import os
import unittest
from unittest import mock

import numpy as np

import test_append_yield as ay
import test_parity2_terminal as p2t
import stage_overlap as so
from test_append_yield import BULK_UID, Pause, Rig, ed, sh
import test_tail_select as ts
import graph_hooks

auto = ay.auto
NEXT_UID = 'o3-next'
THIRD_UID = 'o3-third'
COUNTS = (1, 2, 3, 4)


def o3_rig(*, o3=True, selected=None, counts=COUNTS, waiting=(NEXT_UID,)):
    """The Rig's bulk armed on split as by a proven entry: the residency's generation, its (empty) proven cohort and the
    identity's decode counts are what the policy's 'split' phase records (native_auto_arm tick, phase 'split')."""
    rig = Rig(cached={}, enabled=False)
    ts.mirror_stock_pause(rig)
    rig.c.native_hold_s = 30
    p, d = rig.p, rig.d
    p.o3 = o3
    rig.c.o3_selected = o3 if selected is None else selected     # the arm's o3_split (stage_overlap._stage_control)
    p.o3_split_generation, p.o3_ledger = d.route_generation, frozenset()
    p.decode_counts = frozenset(counts)
    # The rig enters split by a manual enter_split control; the policy's own entry hands the route to the policy
    # (native_auto_arm tick, phase 'split': d._control_route = None), as on the box.
    d._control_route = None
    held = [rig.add_held(uid, 12, queue='waiting') for uid in waiting]
    return rig, held


def run_native_to_terminal(rig, request):
    """The selected request's native chunks to its terminal commit (the last prompt row left to stock)."""
    txns = []
    while request.num_computed_tokens < request.num_tokens - 1:
        txn = rig.admit()
        assert txn is not None and txn.request_uid == request.request_id, (txn, rig.c.last_refusal)
        txns.append(txn)
        rig.finish_chunk()
    return txns


def transitions(rig):
    return [e for e in rig.d._journal if e['kind'] == 'transition']


def split_handback_step(rig, request):
    """The stock step that schedules the kept row on split: the positive-frontier seam must let it through (the real
    seam function), then the pinned stock leaf schedules it (rig.stock_handback) and the step is in flight."""
    assert sh.defer_stock_positive_frontier(rig.s, request) is False
    ts.scheduled(rig, request)


class StaySplit(unittest.TestCase):

    def setUp(self):
        self.log = ts.Log()
        patcher = mock.patch.object(ed, '_progress_logger', lambda: self.log)
        patcher.start()
        self.addCleanup(patcher.stop)

    def to_split_tail(self, rig):
        run_native_to_terminal(rig, rig.bulk)
        rig.tick(lambda: rig.p.phase == 'tail')
        return rig

    def test_two_back_to_back_runs_stay_split(self):
        rig, (nxt,) = o3_rig()
        p, d = rig.p, rig.d
        generation, moves = d.route_generation, len(transitions(rig))
        self.to_split_tail(rig)
        # the terminal commit kept the route: no return owed, the run's intent disarmed, the row kept for the split bank
        self.assertTrue(d._split_ready)
        self.assertIsNone(d._return_pending)
        self.assertIsNone(d.admission_intent())
        self.assertEqual(d.o3_handback_uid(), BULK_UID)
        self.assertTrue(p.o3_tail)
        self.assertEqual(d.o3_counters['kept'], 1)
        self.assertIn('o3_split_handback', rig.kinds())
        # the seam: only the kept row passes; a fresh arrival stays held under split
        self.assertTrue(sh.defer_stock_prefix_hit(rig.s, NS(blocks=()), request=nxt))
        ay.policy_ticks(rig, 2)
        self.assertEqual(p.phase, 'tail')               # not scheduled yet: the split tail waits
        split_handback_step(rig, rig.bulk)
        ay.policy_ticks(rig, 1)
        self.assertEqual(p.phase, 'o3_drain')
        self.assertIs(p.request_ref(), nxt)
        self.assertEqual(p.cohort, (BULK_UID,))
        self.assertEqual(rig.s.pause_state, Pause.PAUSED_ALL)
        self.assertIsNone(d.o3_handback_uid())
        ay.policy_ticks(rig, 3)
        self.assertEqual(p.phase, 'o3_drain')           # the in-flight handback step lands first
        ts.landed(rig, rig.bulk)
        rig.tick(lambda: p.phase == 'native')
        self.assertTrue(d._split_ready)
        self.assertEqual(d.route_generation, generation)
        self.assertEqual(len(transitions(rig)), moves, 'no FULL transition and no enter_split')
        self.assertNotIn('worker_barrier', rig.kinds())
        self.assertEqual(p.o3_counts['armed'], 1)
        gap = p.o3_record()['gap']
        self.assertEqual(list(gap), ['o3'])                       # the run's end to the next arm, on the O3 path
        self.assertEqual(gap['o3'][0], 1)
        self.assertTrue(gap['o3'][1] >= gap['o3'][2] >= 0)
        self.assertIsNone(p._o3_gap_start)
        self.assertEqual(p.o3_ledger, frozenset({BULK_UID}))
        self.assertEqual(rig.s.pause_state, Pause.UNPAUSED)
        self.assertEqual(d.admission_intent()['request_uid'], NEXT_UID)
        txn = rig.admit()
        self.assertEqual(txn.request_uid, NEXT_UID)
        rig.finish_chunk()
        # the second run's own terminal keeps the route as well: cohort 1 + its handback = 2 rows, proven
        run_native_to_terminal(rig, nxt)
        self.assertTrue(d._split_ready)
        self.assertIsNone(d._return_pending)
        self.assertEqual(d.o3_handback_uid(), NEXT_UID)
        self.assertEqual(d.o3_counters['kept'], 2)
        self.assertEqual(len(transitions(rig)), moves)

    def test_off_keeps_todays_full_return(self):
        rig, _ = o3_rig(o3=False)
        moves = len(transitions(rig))
        run_native_to_terminal(rig, rig.bulk)
        self.assertEqual(rig.d._return_pending, 'handback')
        self.assertIsNone(rig.d.o3_handback_uid())
        ay.to_full(rig)
        self.assertEqual(len(transitions(rig)), moves + 1)
        self.assertEqual(rig.d.o3_counters['kept'], 0)
        self.assertFalse(sh.defer_stock_positive_frontier(rig.s, rig.bulk))     # on FULL: stock as today
        self.assertEqual(rig.p._o3_gap_start[1], 'full', "the run's end starts today's gap")
        full = rig.d.o3_counters['transitions'].get('full')
        self.assertIsNotNone(full)
        self.assertEqual(full[0], 1)                               # the FULL return, timed send -> every ack
        self.assertTrue(full[1] >= full[2] >= 0)

    def test_installed_but_not_selected_keeps_todays_full_return(self):
        rig, _ = o3_rig(o3=True, selected=False)       # the box's off arms: FLASH_S5_O3=1 for the run, o3_split False
        run_native_to_terminal(rig, rig.bulk)
        self.assertEqual(rig.d._return_pending, 'handback')
        self.assertIsNone(rig.d.o3_handback_uid())
        self.assertEqual(rig.d.o3_counters['full'], {'off': 1})
        self.assertEqual(rig.p.o3_record()['selected'], 0)
        self.assertFalse(rig.d.o3_handback_permits(rig.bulk, 'positive-frontier'))

    def test_counts_not_proven_keep_the_full_return(self):
        rig, _ = o3_rig(counts=())
        run_native_to_terminal(rig, rig.bulk)
        self.assertEqual(rig.d._return_pending, 'handback')
        self.assertEqual(rig.d.o3_counters['full'], {'counts': 1})

    def test_refusal_reasons(self):
        rig, _ = o3_rig()
        p = rig.p
        self.assertIsNone(p.o3_keeps_split(BULK_UID, 1))
        self.assertEqual(p.o3_keeps_split(BULK_UID, 2), 'tail-rows')            # a geometry tail: FULL as today
        self.assertEqual(p.o3_keeps_split('someone-else', 1), 'not-the-selection')
        p.o3_split_generation += 1
        self.assertEqual(p.o3_keeps_split(BULK_UID, 1), 'generation')
        p.o3_split_generation -= 1
        stranger = NS(request_id='stranger', is_prefill_chunk=False, num_computed_tokens=5, num_prompt_tokens=5)
        rig.s.running.append(stranger)
        self.assertEqual(p.o3_keeps_split(BULK_UID, 1), 'ledger')               # a decoder the route never proved
        p.o3_ledger = frozenset({'stranger'})
        self.assertIsNone(p.o3_keeps_split(BULK_UID, 1))
        p.decode_counts = frozenset({1})
        self.assertEqual(p.o3_keeps_split(BULK_UID, 1), 'counts')               # 1 decoder + the row = 2 rows
        with mock.patch.object(rig.d, 'append_yield_active', lambda: True):
            self.assertEqual(p.o3_keeps_split(BULK_UID, 1), 'yield')
        p.o3 = False
        self.assertEqual(p.o3_keeps_split(BULK_UID, 1), 'off')

    def test_the_seam_passes_only_the_kept_row(self):
        rig, (nxt,) = o3_rig()
        self.to_split_tail(rig)
        d = rig.d
        self.assertTrue(d.o3_handback_permits(rig.bulk, 'positive-frontier'))
        self.assertFalse(d.o3_handback_permits(rig.bulk, 'running-prefill'))
        self.assertFalse(d.o3_handback_permits(nxt, 'positive-frontier'))
        nxt.num_computed_tokens = 4                     # a part-computed request at the positive-frontier seam: held
        self.assertTrue(sh.defer_stock_positive_frontier(rig.s, nxt))
        rig.bulk.num_computed_tokens -= 1               # two rows left: not the one-row handback
        self.assertFalse(d.o3_handback_permits(rig.bulk, 'positive-frontier'))
        self.assertTrue(sh.defer_stock_positive_frontier(rig.s, rig.bulk))

    def test_no_next_head_returns_full_after_the_handback_step(self):
        rig, _ = o3_rig(waiting=())
        self.to_split_tail(rig)
        moves = len(transitions(rig))
        split_handback_step(rig, rig.bulk)
        ay.policy_ticks(rig, 1)
        self.assertEqual(rig.p.phase, 'rollback_drain')
        self.assertTrue(rig.d._split_ready)             # the handback step runs on split first
        ts.landed(rig, rig.bulk)
        rig.tick(lambda: rig.p.phase == 'idle')
        self.assertFalse(rig.d._split_ready)
        self.assertEqual(len(transitions(rig)), moves + 1)
        self.assertEqual(rig.p.rollbacks, {'o3-no-next-head': 1})       # leave_split, then the frozen close
        self.assertEqual(rig.p.o3_counts['full'], {'no-next-head': 1})
        self.assertEqual(rig.s.pause_state, Pause.UNPAUSED)

    def test_a_census_mismatch_returns_full(self):
        rig, _ = o3_rig()
        self.to_split_tail(rig)
        split_handback_step(rig, rig.bulk)
        ay.policy_ticks(rig, 1)
        # leaf: one rank's runner lost the handed-back row (a census the scheduler's cohort does not match)
        worker = rig.stack.workers[2]
        worker.model_runner.req_states.req_id_to_index.pop(BULK_UID)
        ts.landed(rig, rig.bulk)
        rig.tick(lambda: rig.p.phase == 'idle')
        self.assertFalse(rig.d._split_ready)
        self.assertEqual(rig.p.o3_counts['refused'], {'census': 1})
        self.assertEqual(rig.p.o3_counts['armed'], 0)
        self.assertEqual(rig.p.rollbacks, {'o3-census': 1})

    def test_an_unscheduled_handback_returns_full_at_the_bound(self):
        rig, _ = o3_rig()
        self.to_split_tail(rig)
        ay.policy_ticks(rig, auto.O3_TAIL_TICKS + 1)
        self.assertEqual(rig.p.o3_counts['tail_bound'], 1)
        self.assertIsNone(rig.d.o3_handback_uid())
        ay.to_full(rig)
        self.assertFalse(rig.d._split_ready)
        self.assertFalse(rig.p.o3_tail)
        self.assertFalse(sh.defer_stock_positive_frontier(rig.s, rig.bulk))     # FULL: stock schedules it as today
        ts.scheduled(rig, rig.bulk)
        ay.policy_ticks(rig, 1)
        self.assertEqual(rig.p.restores.get('full-tail-scheduled'), 1)        # today's K2 tail from there on


class StaySplitStaged(ay.StagedDriver, StaySplit):
    """The same on the box's StagedSegmentDriver."""

    def setUp(self):
        ay.StagedDriver.setUp(self)
        StaySplit.setUp(self)


def batch(rows):
    """A BatchReqState with the fields the pinned gather_batch_req_state fills (model_runner.py:1386-1416)."""
    sched = np.array([r[0] for r in rows], dtype=np.int32)
    done = np.array([r[1] for r in rows], dtype=np.int32)
    need = np.array([r[2] for r in rows], dtype=np.int32)
    prefilling = done < need
    return NS(num_scheduled_tokens=sched, num_tokens=int(sched.sum()), num_computed_prefill_tokens_np=done,
              prefill_len_np=need, is_prefilling_np=prefilling, has_prefill=bool(prefilling.any()))


P = 100197
DECODER = (1, P + 5, P)
HANDBACK = (1, P - 1, P)


class HandbackUniform(unittest.TestCase):
    """graph_hooks.handback_uniform: 1 only for the split route's one-row handback batch, with FLASH_S5_O3=1."""

    def setUp(self):
        self.router = NS(current_mode='decode-split')
        self.runner = NS(decode_query_len=1, _flash_s5_native_census=NS(o3_handbacks=0))
        for patcher in (mock.patch.object(graph_hooks, '_router', lambda runner: self.router),
                        mock.patch.object(graph_hooks, '_native_decode', lambda router: router is self.router),
                        mock.patch.dict(os.environ, {'FLASH_S5_O3': '1'})):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.output = NS(flash_e3_transaction=None, scheduled_spec_decode_tokens={})

    def uniform(self, rows, stock=None):
        return graph_hooks.handback_uniform(self.runner, self.output, batch(rows), stock)

    def test_shapes(self):
        f = lambda rows: graph_hooks.handback_shaped(batch(rows))
        for n in (0, 1, 2, 7, 8):
            self.assertTrue(f([DECODER] * n + [HANDBACK]), n)
        self.assertFalse(f([DECODER] * 2))                              # no prefill row: stock's own uniform count
        self.assertFalse(f([DECODER, HANDBACK, HANDBACK]))              # two handbacks in one step
        self.assertFalse(f([DECODER, (3, P - 3, P)]))                   # a geometry tail
        self.assertFalse(f([DECODER, (1, P - 2, P)]))                   # one row of a two-row tail
        self.assertFalse(f([(2, P + 5, P), HANDBACK]))                  # a non-decode row beside it

    def test_uniform_on_the_split_route_only(self):
        self.assertEqual(self.uniform([DECODER] * 2 + [HANDBACK]), 1)
        self.assertEqual(self.runner._flash_s5_native_census.o3_handbacks, 1)
        self.assertEqual(self.uniform([DECODER] * 2, stock=1), 1)                # stock's count untouched
        self.assertIsNone(self.uniform([DECODER, (3, P - 3, P)]))
        self.router.current_mode = 'ordinary'
        self.assertIsNone(self.uniform([DECODER] * 2 + [HANDBACK]))      # FULL route: today's eager handback
        self.router.current_mode = 'decode-split'
        self.output.flash_e3_transaction = object()
        self.assertIsNone(self.uniform([DECODER] * 2 + [HANDBACK]))      # an E3 lane batch: never
        self.output.flash_e3_transaction = None
        self.output.scheduled_spec_decode_tokens = {'x': [1]}
        self.assertIsNone(self.uniform([DECODER] * 2 + [HANDBACK]))
        self.output.scheduled_spec_decode_tokens = {}
        with mock.patch.dict(os.environ, {'FLASH_S5_O3': '0'}):
            self.assertIsNone(self.uniform([DECODER] * 2 + [HANDBACK]))
        with mock.patch.dict(os.environ, {'FLASH_S5_O3': 'yes'}), self.assertRaises(Exception):
            self.uniform([DECODER] * 2 + [HANDBACK])
        self.assertEqual(self.runner._flash_s5_native_census.o3_handbacks, 1)

    def test_has_prefill_is_left_for_the_prompt_token(self):
        state = batch([DECODER, HANDBACK])
        self.assertEqual(graph_hooks.handback_uniform(self.runner, self.output, state, None), 1)
        self.assertTrue(state.has_prefill)              # prepare_inputs still runs prepare_prefill_inputs for the row


class Install(unittest.TestCase):
    def install(self, env):
        engine = NS(_flash_s5_startup_ready=True, scheduler=NS())
        driver = NS(controller=NS(), auto_arm=None)
        return auto.install(engine, driver, env)

    def test_install_reads_the_switch(self):
        self.assertTrue(self.install({'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_O3': '1'}).o3)
        self.assertFalse(self.install({'FLASH_S5_AUTO_ARM': '1'}).o3)
        self.assertIsNone(self.install({'FLASH_S5_O3': '1'}))          # O3 lives on the auto-arm policy only

    def test_env(self):
        self.assertFalse(auto.o3_enabled({}))
        self.assertTrue(auto.o3_enabled({'FLASH_S5_O3': '1'}))
        with self.assertRaises(ay.E3ContractError):
            auto.o3_enabled({'FLASH_S5_O3': 'true'})
        self.assertFalse(auto.AutoArmPolicy.o3)


MUTANT_LOG = []


class O3Selection(p2t.ie.FlagOn, p2t.ie.Early, p2t.c1.OverlapBase):
    """The arm's o3_split on the real stage command (stage_overlap._stage_control): selected, echoed, in status; a command
    without it turns it off; refused when not a bool or not installed (FLASH_S5_O3 unset)."""
    idle = p2t.Selection.idle
    base = p2t.Selection.base

    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ, {'FLASH_S5_O3': '1'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_selected_by_the_stage_command_and_read_in_status(self):
        driver = self.idle()
        self.assertIs(driver._stage_control(dict(self.base, o3_split=True))['o3_split'], True)
        self.assertIs(driver.controller.o3_selected, True)
        self.assertEqual(driver.stage_status()['overlap']['o3'], dict(installed=True, selected=True))
        self.assertIs(driver._stage_control(dict(self.base))['o3_split'], False, 'a command without the field turns it off')
        self.assertIs(driver.controller.o3_selected, False)
        self.assertEqual(driver.stage_status()['overlap']['o3'], dict(installed=True, selected=False))

    def test_refused_when_invalid_or_not_installed(self):
        driver = self.idle()
        with self.assertRaises(ed.DriverError):
            driver._stage_control(dict(self.base, o3_split=1))
        with mock.patch.dict(os.environ, {'FLASH_S5_O3': '0'}):
            with self.assertRaises(ed.DriverError):
                driver._stage_control(dict(self.base, o3_split=True))
            self.assertIs(driver._stage_control(dict(self.base))['o3_split'], False)
        self.assertIs(getattr(driver.controller, 'o3_selected', False), False, 'nothing applied on a refusal')


class Mutants(unittest.TestCase):
    """One engine mutation per new test; each must fail it (lead's rule: a mutant per new test)."""

    def fails(self, test, *mutation):
        case = test.__self__ if hasattr(test, '__self__') else None
        with ay.mutant(*mutation), self.assertRaises(ay.MUTANT_FAILURES) as caught:
            test()
        MUTANT_LOG.append((test.__name__, mutation[1], type(caught.exception).__name__))

    def run_test(self, cls, name):
        case = cls(name)
        def go():
            case.setUp()
            try:
                getattr(case, name)()
            finally:
                case.doCleanups()
        go.__name__ = name
        return go

    def test_two_runs_driver_owes_full(self):
        self.fails(self.run_test(StaySplit, 'test_two_back_to_back_runs_stay_split'),
                   ed.NativeSegmentDriver, '_note_terminal_route',
                   ("if reason == 'handback' and self._o3_keep_split(transaction, admission):", 'if False:'))

    def test_two_runs_seam_holds_the_row(self):
        self.fails(self.run_test(StaySplit, 'test_two_back_to_back_runs_stay_split'),
                   sh, '_defer_stock_while_split_armed',
                   ('    if callable(permits) and permits(request, kind):\n        return False\n', ''))

    def test_two_runs_ledger_not_grown(self):
        self.fails(self.run_test(StaySplit, 'test_two_back_to_back_runs_stay_split'),
                   auto.AutoArmPolicy, '_o3_arm', ('        self.o3_ledger = frozenset(self.cohort)\n', ''))

    def test_two_runs_intent_left_armed(self):
        self.fails(self.run_test(StaySplit, 'test_two_back_to_back_runs_stay_split'),
                   ed.NativeSegmentDriver, '_o3_keep_split', ('        if intent is not None:', '        if False:'))

    def test_off_ignored(self):
        self.fails(self.run_test(StaySplit, 'test_off_keeps_todays_full_return'),
                   auto.AutoArmPolicy, '_o3_keeps_split', ("            if not self.o3 or getattr(self.controller, 'o3_selected', False) is not True:\n", "            if False:\n"))

    def test_gap_not_closed(self):
        self.fails(self.run_test(StaySplit, 'test_two_back_to_back_runs_stay_split'),
                   auto.AutoArmPolicy, '_o3_arm', ("        self._o3_gap_close()\n", ''))

    def test_transitions_not_timed(self):
        self.fails(self.run_test(StaySplit, 'test_off_keeps_todays_full_return'),
                   ed.NativeSegmentDriver, '_apply',
                   ("            self.o3_counters['transitions'][bank] = [n + 1, round(total + ms, 3), round(max(worst, ms), 3)]\n", ''))

    def test_selection_ignored(self):
        self.fails(self.run_test(StaySplit, 'test_installed_but_not_selected_keeps_todays_full_return'),
                   auto.AutoArmPolicy, '_o3_keeps_split', ("            if not self.o3 or getattr(self.controller, 'o3_selected', False) is not True:\n", "            if not self.o3:\n"))

    def test_stage_drops_the_selection(self):
        self.fails(self.run_test(O3Selection, 'test_selected_by_the_stage_command_and_read_in_status'),
                   so.OverlapSegmentDriver, '_stage_control',
                   ('        self.controller.o3_selected = o3_split\n', '        self.controller.o3_selected = False\n'))

    def test_stage_without_install_check(self):
        self.fails(self.run_test(O3Selection, 'test_refused_when_invalid_or_not_installed'),
                   so.OverlapSegmentDriver, '_stage_control',
                   ("            if os.environ.get('FLASH_S5_O3', '0') != '1':\n                raise DriverError('native-stage-o3-not-installed')\n", ''))

    def test_counts_ignored(self):
        self.fails(self.run_test(StaySplit, 'test_counts_not_proven_keep_the_full_return'),
                   auto.AutoArmPolicy, '_o3_keeps_split',
                   ("            if not self._o3_covers(rows):\n                return 'counts'\n", ''))

    def test_tail_rows_ignored(self):
        self.fails(self.run_test(StaySplit, 'test_refusal_reasons'),
                   auto.AutoArmPolicy, '_o3_keeps_split', ("            if remaining != 1:", "            if False:"))

    def test_seam_any_row_count(self):
        self.fails(self.run_test(StaySplit, 'test_the_seam_passes_only_the_kept_row'),
                   ed.NativeSegmentDriver, 'o3_handback_permits',
                   ('                  and request.num_tokens - request.num_computed_tokens == 1)', '                  )'))

    def test_no_head_without_pause(self):
        self.fails(self.run_test(StaySplit, 'test_no_next_head_returns_full_after_the_handback_step'),
                   auto.AutoArmPolicy, '_o3_tail',
                   ("        self.phase = 'rollback_drain'\n        return False\n",
                    "        self._rollback('o3-no-next-head')\n        return False\n"))

    def test_census_without_live_rows(self):
        self.fails(self.run_test(StaySplit, 'test_a_census_mismatch_returns_full'),
                   auto.AutoArmPolicy, '_o3_proof',
                   ("                    or sorted(live.get('live_request_uids', ())) != list(self.cohort)\n"
                    "                    or row.get('decode_eligible') != len(self.cohort)):",
                    "                    ):"))

    def test_tail_unbounded(self):
        self.fails(self.run_test(StaySplit, 'test_an_unscheduled_handback_returns_full_at_the_bound'),
                   auto.AutoArmPolicy, '_o3_tail', ("                d.o3_return_full('tail-bound')\n", ''))

    def test_shape_two_handbacks(self):
        self.fails(self.run_test(HandbackUniform, 'test_shapes'),
                   graph_hooks, 'handback_shaped', ('return len(index) == 1 and', 'return len(index) >= 1 and'))

    def test_uniform_on_full_route(self):
        self.fails(self.run_test(HandbackUniform, 'test_uniform_on_the_split_route_only'),
                   graph_hooks, 'handback_uniform', ("            or router.current_mode == 'ordinary'\n", ''))

    def test_uniform_clears_has_prefill(self):
        self.fails(self.run_test(HandbackUniform, 'test_has_prefill_is_left_for_the_prompt_token'),
                   graph_hooks, 'handback_uniform', ('    return 1\n', '    batch_state.has_prefill = False\n    return 1\n'))

    def test_install_without_switch(self):
        self.fails(self.run_test(Install, 'test_install_reads_the_switch'),
                   auto, 'install', ('    policy.o3 = o3_enabled(env)\n', ''))

    def test_env_any_value(self):
        self.fails(self.run_test(Install, 'test_env'),
                   auto, 'o3_enabled', ("    if value not in ('0', '1'):", "    if False:"))


if __name__ == '__main__':
    unittest.main()
