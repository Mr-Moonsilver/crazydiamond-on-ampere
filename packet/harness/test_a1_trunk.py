#!/usr/bin/env python3
"""a1-trunk: the returning-append yield on the C1 driver, with the review's contract (4 Nov 2026, section 8). CPU only.

Once a yield is pending: no further bulk successor; admitted work drains to a common valid state on both stages; the short request
is served; the bulk resumes. Real StagedAdmissionController, OverlapSegmentDriver, AutoArmPolicy and four bridge workers
(test_append_yield.Rig on the overlap classes), through the real admission seam.
Leaves beyond Rig's: the manager fields overlap_refusal reads (no hybrid groups, hash block 1), as OverlapStack declares them.
Discovery mode: tests for the new logic only. Chunk timing, decode, and the yield's speed are the box's.
"""
import os
import types
import unittest
from unittest import mock

import test_append_yield as ay
from test_append_yield import APPEND_UID, BULK_UID, Rig, auto_arm_workers
import test_e3_r4_worker_bridge as bridge_suite
import test_e3_native_overlap as native
import async_handoff as ah
import engine_driver as ed
import native_auto_arm as auto
import scheduler_hooks as sh
import stage_overlap as so
import stage_progress as sp
import native_measurement as nm
import stage_net

NS = types.SimpleNamespace
FLAG = {sp.STAGE_LOCAL_ENV: '1', so.STAGE_OVERLAP_ENV: '1', so.DEFER_BIND_ENV: '1', nm.INWAIT_ENV: '1'}
TOKENS = 33      # chunks of 4: enough for several successors


def overlap_cold_stack(tokens=TOKENS):
    real_request = bridge_suite.Request

    def make_controller(scheduler, *, chunk=4, ranks=(0, 1, 2, 3), max_history=64, leases=None, frontier=None):
        return so.StagedAdmissionController(scheduler, leases=leases or sh.SchedulerLeases(),
                                            frontier=frontier or ah.CommittedFrontier(), engine_epoch=7,
                                            chunk_tokens=chunk, ranks=ranks, max_history=max_history).attach()

    def request(uid, tokens=tokens, **kwargs):
        made = real_request(uid=uid, tokens=TOKENS, **kwargs)
        made.num_in_flight_tokens = 0
        made.num_prompt_tokens = TOKENS
        return made

    with mock.patch.dict(os.environ, FLAG), \
            mock.patch.object(bridge_suite, 'DirectTransport', ay.StagedDirectTransport), \
            mock.patch.object(bridge_suite.ed, 'NativeSegmentDriver', so.OverlapSegmentDriver), \
            mock.patch.object(bridge_suite, 'make_controller', make_controller), \
            mock.patch.object(bridge_suite, 'Request', request):
        stack = native.cold_stack()
    stack.sender.queue = stage_net.FakeRing((0, 1, 2, 3))
    stack.driver.defer_bind = True
    return stack


ay.STACKS['overlap'] = overlap_cold_stack
ay.DRIVERS['overlap'] = so.OverlapSegmentDriver


class C1Rig(Rig):
    def __init__(self, *, early=False, inwait=False, **kwargs):
        kwargs.setdefault('cached', {})
        kwargs.setdefault('max_new', 8)
        with mock.patch.dict(os.environ, FLAG):
            super().__init__(driver='overlap', **kwargs)
        manager = self.s.kv_cache_manager
        manager.coordinator.single_type_managers = ()
        manager.block_pool.hash_block_size = 1
        ay.upgrade(self.bulk)
        self.bulk.num_in_flight_tokens = 0
        self.bulk.num_prompt_tokens = self.bulk.num_tokens
        self.d.inwait = inwait
        self.d.early_successor = early

    # one engine step: the admission seam runs each schedule(), the tick after it
    def step(self):
        record = sh.flash_s5_admit(self.s)
        self.d.advance()
        return record

    def steps(self, until, limit=500, complete=True):
        for _ in range(limit):
            if until():
                return
            self.step()
            if complete:
                self.stack.complete_everything()
        raise AssertionError('no state; frames=%r yield=%r journal=%r' % (
            [(f.chunk['tx'].chunk_start, sorted(f.opened), sorted(f.released)) for f in self.d._frames],
            self.d.append_yield_phase(), self.kinds()[-6:]))

    def ticks(self, until, limit=500):
        """Engine ticks with no admission pass: the state just before the seam would admit."""
        for _ in range(limit):
            if until():
                return
            self.d.advance()
            self.stack.complete_everything()
        raise AssertionError('no state; frames=%r' % ([(sorted(f.opened), sorted(f.released)) for f in self.d._frames],))

    def stage0_released(self):
        return bool(self.d._frames) and 0 in self.d._frames[0].released


class Base(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, FLAG)
        patcher.start()
        self.addCleanup(patcher.stop)

    def rig(self, **kwargs):
        return C1Rig(**kwargs)

    def assert_common_state(self, rig, frontier):
        """The contract's state: nothing admitted, both stages at the frontier, the row retained on every rank."""
        d, c = rig.d, rig.c
        self.assertIsNone(c.parked)
        self.assertIsNone(c.successor)
        self.assertEqual(d._frames, [])
        self.assertIsNone(d.tx)
        self.assertEqual(rig.bulk.num_computed_tokens, frontier)
        self.assertEqual(d._retained['transaction'].chunk_end, frontier)
        self.assertEqual(d._retained['transaction'].request_uid, BULK_UID)
        self.assertTrue(all(row['retained_rows'] == 1 for row in ay.worker_rows(rig).values()))
        self.assertFalse(any(ledger.outstanding(d._retained['transaction']) for ledger in d.pp_ledgers.values()))

    def serve_and_resume(self, rig, append):
        """From closing to the reopen: the rest of the flow on the real path."""
        d = rig.d
        ay.to_append(rig)
        self.assertIsNone(d._retained)
        frontier = d._yield['frontier']
        ay.run_append(rig)
        self.assertEqual(d.append_yield_phase(), 'handback')
        ay.to_full(rig)
        rig.stock_handback(append)
        ay.to_resume(rig)
        self.assertEqual(rig.p.plan['start'], frontier)
        txn = rig.admit()
        self.assertEqual((txn.request_uid, txn.chunk_start), (BULK_UID, frontier))
        rig.drive(lambda: d.tx is not None)
        self.assertIsNone(d.append_yield_phase())
        return frontier


class Arrival(Base):
    """The review's four decisive arrivals, each classified by the engine and drained by the contract."""

    def contract(self, rig, arrival, *, frontier):
        """After the append arrived: no further admission, a common state at ``frontier``, then the rest of the flow."""
        d, c = rig.d, rig.c
        successors = c.overlap_counters['successors']
        append = d._yield['append'] if d._yield else None
        self.assertIsNotNone(append, 'the yield is pending')
        self.assertEqual(d.append_yield_record()['arrivals'][arrival], 1, d.append_yield_record()['arrivals'])
        rig.steps(lambda: d.append_yield_phase() != 'draining')
        self.assertEqual(c.overlap_counters['successors'], successors, 'a pending yield admitted a bulk successor')
        self.assertEqual(d.append_yield_phase(), 'closing')
        self.assert_common_state(rig, frontier)
        self.serve_and_resume(rig, append)
        self.assertEqual(d.append_yield_counters['reopened'], 1)
        self.assertEqual(rig.bulk.num_computed_tokens, frontier)
        return rig

    def test_before_admission(self):
        rig = self.rig()
        rig.admit()
        rig.finish_chunk()                      # chunk 1 committed, nothing admitted
        self.assertIsNone(rig.c.parked)
        rig.add_held(APPEND_UID, 11, cached=8)
        rig.d.advance()
        self.contract(rig, 'before-admission', frontier=4)

    def test_before_admission_seen_in_the_admission_pass(self):
        # A request that arrived since the tick is seen before the lookahead admits n+1.
        rig = self.rig()
        rig.admit()
        rig.ticks(rig.stage0_released)
        rig.add_held(APPEND_UID, 11, cached=8)
        self.assertIsNone(sh.flash_s5_admit(rig.s))
        self.assertIsNone(rig.c.successor)
        self.assertEqual(rig.d.append_yield_phase(), 'draining')
        self.contract(rig, 'transfer-retirement', frontier=4)

    def test_after_reservation_before_worker_open(self):
        rig = self.rig()
        rig.admit()
        rig.ticks(rig.stage0_released)
        self.assertIsNotNone(sh.flash_s5_admit(rig.s))      # n+1 reserved; no frame, no worker open
        self.assertIsNotNone(rig.c.successor)
        self.assertEqual(len(rig.d._frames), 1)
        rig.add_held(APPEND_UID, 11, cached=8)
        self.assertIsNone(sh.flash_s5_admit(rig.s), 'the one successor slot is taken')
        self.assertIsNone(rig.d.append_yield_phase())
        self.assertTrue(rig.d._yield_offer())                # the tick's offer, before _drive begins the frame
        self.contract(rig, 'reserved-unopened', frontier=8)

    def test_stage_zero_ahead_of_stage_one(self):
        rig = self.rig()
        rig.admit()

        def stage_zero_on_successor():
            frames = rig.d._frames
            return len(frames) == 2 and 0 in frames[1].opened and 1 not in frames[0].released
        # Stage 0's device work completes, stage 1's does not: stage 1 stays on n while stage 0 runs n+1.
        for _ in range(2000):
            if stage_zero_on_successor():
                break
            rig.step()
            for rank in (0, 1):
                rig.stack.worker(rank).bus.complete_all(True)
        self.assertTrue(stage_zero_on_successor(), [(sorted(f.opened), sorted(f.released)) for f in rig.d._frames])
        rig.add_held(APPEND_UID, 11, cached=8)
        self.assertTrue(rig.d._yield_offer())
        self.contract(rig, 'stage0-ahead', frontier=8)

    def test_during_transfer_retirement(self):
        rig = self.rig()
        rig.admit()

        def retiring():
            frames = rig.d._frames
            return (len(frames) == 1 and frames[0].released == {0, 1}
                    and set(frames[0].chunk['_receipt_acked']) != set(rig.d.pp_peers))
        # Stage 1 released n; stage 0's receipts still outstanding. Keep the lookahead out by reading the state before admitting.
        for _ in range(2000):
            if retiring() or (rig.d._frames and rig.d._frames[0].released):
                break
            rig.d.advance()
            rig.stack.complete_everything()
        self.assertTrue(rig.d._frames and rig.d._frames[0].released, 'a local release was reached')
        rig.add_held(APPEND_UID, 11, cached=8)
        self.assertIsNone(sh.flash_s5_admit(rig.s))
        self.contract(rig, 'transfer-retirement', frontier=4)

    def test_in_chunk(self):
        rig = self.rig()
        rig.admit()
        rig.d.advance()
        rig.add_held(APPEND_UID, 11, cached=8)
        rig.d.advance()
        self.contract(rig, 'reserved-unopened' if not rig.d._frames[0].opened else 'in-chunk', frontier=4)


class EarlySuccessor(Arrival):
    """The same arrivals with the early successor and in-wait selected (the box's arm)."""

    def rig(self, **kwargs):
        return C1Rig(early=True, inwait=True, **kwargs)


class Blocked(Base):
    def test_pending_blocks_the_lookahead_and_the_next_chunk(self):
        rig = self.rig()
        rig.admit()
        rig.ticks(rig.stage0_released)
        rig.add_held(APPEND_UID, 11, cached=8)
        self.assertIsNone(sh.flash_s5_admit(rig.s))
        self.assertFalse(rig.d.successor_admissible(rig.c.parked.transaction))
        rig.steps(lambda: rig.d.append_yield_phase() != 'draining')
        self.assertIsNone(sh.flash_s5_admit(rig.s), 'closing admits nothing')

    def test_mutant_without_the_block_admits_a_successor(self):
        # The contract's first clause removed: the same arrival admits n+1 while the yield is pending.
        def check():
            rig = self.rig()
            rig.admit()
            rig.ticks(rig.stage0_released)
            rig.add_held(APPEND_UID, 11, cached=8)
            self.assertIsNone(sh.flash_s5_admit(rig.s))
            self.assertIsNone(rig.c.successor)
        check()
        # Three layers hold it: the lookahead's own gate, the policy's permit and the controller's target rule. Each alone
        # is enough; all removed, it admits.
        gate = ("        if self._yield is not None and not (self._yield['phase'] == 'append'\n"
                "                                            and transaction.request_uid == self._yield['append'].request_id):\n"
                "            return False\n", "")
        permit = ("            return self.driver.yield_permits(request)\n", "            return request is self._request()\n")
        with ay.mutant(so.OverlapSegmentDriver, 'successor_admissible', gate):
            check()
        with ay.mutant(auto.AutoArmPolicy, 'permits', permit):
            check()
        target = ("            target = target_query() if callable(target_query) else None\n", "            target = None\n")
        with ay.mutant(sh.PrefillAdmissionController, '_admit_observed', target):
            check()
        with ay.mutant(so.OverlapSegmentDriver, 'successor_admissible', gate), ay.mutant(auto.AutoArmPolicy, 'permits', permit), \
                ay.mutant(sh.PrefillAdmissionController, '_admit_observed', target), self.assertRaises(AssertionError):
            check()

    def test_no_yield_on_a_terminal_chunk(self):
        rig = self.rig()
        for _ in range(3000):
            if rig.c.parked is not None and rig.c.parked.admission is not None and rig.c.parked.admission.terminal:
                break
            rig.step()
            rig.stack.complete_everything()
        self.assertTrue(rig.c.parked.admission.terminal, rig.kinds()[-5:])
        rig.add_held(APPEND_UID, 11, cached=8)
        rig.d.advance()
        self.assertIsNone(rig.d.append_yield_phase())

    def test_off_never_drains(self):
        rig = self.rig(enabled=False)
        rig.admit()
        rig.add_held(APPEND_UID, 11, cached=8)
        rig.steps(lambda: rig.c.overlap_counters['successors'] >= 1)
        self.assertIsNone(rig.d.append_yield_phase())

    def test_common_state_check_refuses_a_counter_ahead(self):
        # Never roll back only the counter: a frontier the row does not carry stops the yield, ownership retained.
        rig = self.rig()
        rig.admit()
        rig.finish_chunk()
        rig.bulk.num_computed_tokens += 1
        rig.add_held(APPEND_UID, 11, cached=8)
        with self.assertRaisesRegex(ed.DriverError, 'not common to both stages'):
            rig.d._yield_offer()


class Drain(Base):
    def pending_with_two_frames(self):
        rig = self.rig()
        rig.admit()
        rig.steps(lambda: len(rig.d._frames) == 2)
        rig.add_held(APPEND_UID, 11, cached=8)
        self.assertTrue(rig.d._yield_offer())
        self.assertEqual(rig.d.append_yield_phase(), 'draining')
        return rig

    def test_append_left_during_the_drain(self):
        rig = self.pending_with_two_frames()
        append = rig.d._yield['append']
        rig.stock_abort(append)
        rig.steps(lambda: rig.d.append_yield_phase() is None)
        self.assertEqual(rig.d.append_yield_end_reasons, {'drain-no-append': 1})
        self.assertEqual(rig.d.append_yield_counters['abandoned'], 0)
        self.assertEqual(rig.d.append_yield_counters['rows_closed'], 0)
        self.assertEqual(rig.d._yield_counts[(BULK_UID, rig.d._retained['transaction'].request_generation)], 0, 'nothing served, the per-bulk count is refunded')
        self.assertTrue(rig.d._split_ready)
        # the bulk simply goes on, no route change
        before = rig.d.route_generation
        txn = rig.admit()
        self.assertEqual((txn.request_uid, txn.chunk_start), (BULK_UID, 8))
        self.assertEqual(rig.d.route_generation, before)

    def test_next_held_takes_the_turn(self):
        rig = self.pending_with_two_frames()
        first = rig.d._yield['append']
        second = rig.add_held('ra-second', 11, cached=8)
        rig.stock_abort(first)
        rig.steps(lambda: rig.d.append_yield_phase() != 'draining')
        self.assertIs(rig.d._yield['append'], second)

    def test_stop_during_the_drain(self):
        for kind in ('abort', 'cancel', 'reset'):
            with self.subTest(kind=kind):
                rig = self.pending_with_two_frames()
                rig.d.stop(kind)
                self.assertIsNone(rig.d.append_yield_phase())
                self.assertEqual(rig.d.append_yield_end_reasons, {'stop-' + kind: 1})
                for _ in range(500):
                    if not rig.d._frames and rig.d.stop_kind is None:
                        break
                    rig.d.advance()
                    rig.stack.complete_everything()
                self.assertEqual(rig.d._frames, [])
                self.assertIsNone(rig.c.successor)
                self.assertIsNone(rig.d.error)


class Several(Base):
    def run_all(self, rig, uids):
        """Every returning agent held at once: bulk chunks committed between yields, and the bulk still completes."""
        d = rig.d
        between, last, served = [], None, []
        for _ in range(4000):
            if rig.bulk.num_computed_tokens >= rig.bulk.num_tokens - 1:
                break
            phase = d.append_yield_phase()
            if phase == 'handback' and not d._split_ready and not d.active:
                append = d._yield['append']
                if append.num_output_tokens == 0 and append.request_id not in served:
                    served.append(append.request_id)
                    rig.stock_handback(append)
            with auto_arm_workers():
                rig.p.tick()
                sh.flash_s5_admit(rig.s)
                d.advance()
                rig.stack.complete_everything()
            ends = d.append_yield_counters['selected']
            if ends != last:
                between.append(rig.bulk.num_computed_tokens)
                last = ends
        self.assertIsNone(d.error)
        self.assertEqual(rig.bulk.num_computed_tokens, rig.bulk.num_tokens - 1, ('bulk starved', d.append_yield_phase(), rig.kinds()[-8:]))
        return between, served

    def test_several_at_once_one_at_a_time_and_the_bulk_completes(self):
        rig = self.rig(per_bulk=4)
        uids = ['ra-%d' % i for i in range(3)]
        rig.admit()
        rig.ticks(rig.stage0_released)
        for uid in uids:
            rig.add_held(uid, 11, cached=8)
        between, served = self.run_all(rig, uids)
        self.assertEqual(served, uids, 'FCFS, each served once')
        self.assertEqual(rig.d.append_yield_counters['reopened'], 3)
        self.assertEqual(rig.d.append_yield_counters['abandoned'], 0)
        self.assertEqual(len(rig.d.append_yield_end_reasons), 1)
        # a yield starts only after the last reopened bulk chunk was admitted
        self.assertEqual(sorted(between), between)
        self.assertEqual(len(set(between)), len(between), 'two yields with no bulk chunk between them')

    def test_per_bulk_bounds_the_bulk_loss(self):
        rig = self.rig(per_bulk=2)
        uids = ['ra-%d' % i for i in range(5)]
        rig.admit()
        rig.ticks(rig.stage0_released)
        for uid in uids:
            rig.add_held(uid, 11, cached=8)
        _, served = self.run_all(rig, uids)
        self.assertEqual(served, uids[:2])
        self.assertEqual(rig.d.append_yield_counters['selected'], 2)
        self.assertIn(rig.s.requests['ra-2'], rig.s.skipped_waiting.items)


class Scan(Base):
    def test_each_held_request_looked_up_once_per_bulk(self):
        """Box 1: a 151k lookup stalled the engine about 0.265 s, per held request, per chunk, when the scan was per chunk."""
        rig = self.rig()
        rig.admit()
        rig.add_held(APPEND_UID, 2000, cached=0)        # over max_new: never a candidate
        before = len(rig.lookups)
        for _ in range(5):
            rig.d._yield_offer()
        self.assertEqual(len(rig.lookups) - before, 1)
        rig.finish_chunk()
        rig.admit()                                     # a new newest chunk, the same bulk
        rig.d._yield_offer()
        self.assertEqual(len(rig.lookups) - before, 1)
        self.assertEqual(rig.c.yield_refusals, {'max-new': 1})

    def test_the_drain_does_not_look_up_a_request_still_held(self):
        rig = self.rig()
        rig.admit()
        rig.add_held(APPEND_UID, 11, cached=8)
        rig.d._yield_offer()
        self.assertEqual(rig.d.append_yield_phase(), 'draining')
        before = len(rig.lookups)
        rig.finish_chunk()
        rig.d._maybe_yield()
        self.assertEqual(rig.d.append_yield_phase(), 'closing')
        self.assertEqual(len(rig.lookups), before)

    def test_timer_counts_every_caller_also_with_the_control_off(self):
        rig = self.rig()
        rig.admit()
        rig.c.append_yield.enabled = False
        rig.d._yield_offer()
        rig.add_held(APPEND_UID, 2000, cached=0)
        rig.s.kv_cache_manager.get_computed_blocks(rig.s.requests[APPEND_UID])      # as stock's waiting loop does
        record = rig.d.append_yield_record()['lookups']
        self.assertEqual((record['calls'], record['scan_calls']), (1, 0))
        self.assertGreaterEqual(record['max_ns'], 0)
        rig.c.append_yield.enabled = True
        rig.d._yield_offer()
        record = rig.d.append_yield_record()['lookups']
        self.assertEqual((record['calls'], record['scan_calls']), (2, 1))

    def test_no_timer_without_the_bounds(self):
        rig = self.rig()
        rig.c.append_yield = None
        real = rig.s.kv_cache_manager.get_computed_blocks
        rig.d._yield_offer()
        self.assertIs(rig.s.kv_cache_manager.get_computed_blocks, real)


class ShortTail(Base):
    """Box 1, q18a-a1-trunk: returners of 151905 and 152651 tokens, stock's stop at the last hash boundary, a minimum of 64 rows."""
    UNIT, MINIMUM = 176, 64

    def stock_split(self, request, limit, num_new_local_computed_tokens=0):
        """The stop of scheduler._mamba_block_aligned_split that matters here; like stock, it reads num_prompt_tokens. The real text: scratch/box1d/real_split.py."""
        start = request.num_computed_tokens + num_new_local_computed_tokens
        boundary = request.num_prompt_tokens // self.UNIT * self.UNIT
        return boundary - start if start < boundary < start + limit else limit

    def plan(self, tokens, start, *, moved=True):
        rig = getattr(self, '_rig', None) or self.rig()
        self._rig = rig
        rig.s.hash_block_size, rig.s.mamba_partial_cache_hit = self.UNIT, True
        rig.c.minimum_stage_rows = self.MINIMUM
        request = NS(num_tokens=tokens, num_prompt_tokens=tokens, num_computed_tokens=0)
        view = rig.c._append_yield_split_view(rig.s, request) if moved else request

        def split(position, limit):
            return self.stock_split(sh._ProjectedPrefillRequest(view, position), limit)
        return sh.staged_chunk_plan(start, tokens - 1, 1024, self.MINIMUM, split)

    def test_the_box_returners_have_a_plan_once_the_stop_moves(self):
        self.assertEqual(self.plan(151905, 151360, moved=False), ())
        self.assertEqual(self.plan(152651, 151008, moved=False), ())
        self.assertEqual(self.plan(151905, 151360), (352, 192))
        self.assertEqual(self.plan(152651, 151008), (1024, 384, 234))

    def test_a_tail_that_fits_keeps_stocks_stop(self):
        self.assertEqual(self.plan(151965, 151360), (528, 76), 'box 1c, yield 3')
        self.assertEqual(self.plan(65445, 64768), (528, 148), 'the bulk tail')
        rig = self._rig
        request = NS(num_tokens=151965, num_prompt_tokens=151965)
        self.assertIs(rig.c._append_yield_split_view(rig.s, request), request)

    def test_every_tail_at_every_start_has_a_plan_over_the_whole_prompt(self):
        boundary = 860 * self.UNIT
        for tokens in range(boundary, boundary + self.UNIT + 1):
            for start in (boundary - 3 * self.UNIT, boundary - self.UNIT):
                plan = self.plan(tokens, start)
                if tokens - 1 - start < self.MINIMUM:
                    continue
                self.assertEqual(sum(plan), tokens - 1 - start, tokens)
                self.assertGreaterEqual(min(plan), self.MINIMUM, (tokens, plan))

    def test_only_a_hit_inside_the_last_unit_stays_without_a_plan(self):
        # New rows all past the last boundary, under the minimum: no earlier state to resume from.
        self.assertEqual(self.plan(176 * 3 + 17, 176 * 3), ())
        self.assertEqual(sum(self.plan(176 * 3 + 17, 176 * 2)), 176 + 16, 'one unit back from a hit crosses the boundary')

    def test_the_view_is_read_only_and_keeps_the_prefill_end(self):
        request = NS(num_tokens=151905, num_prompt_tokens=151905, num_computed_tokens=151360, marker=1)
        view = sh._ShortTailPrefillRequest(request, 151888)
        self.assertEqual((view.num_prompt_tokens, view.num_tokens, view.num_computed_tokens, view.marker), (151887, 151905, 151360, 1))
        self.assertEqual(max(view.num_prompt_tokens, view.num_tokens - 1), 151904)
        with self.assertRaises(sh.E3ContractError):
            view.num_prompt_tokens = 1

    def test_no_stop_to_move_when_stock_has_none(self):
        rig = self.rig()
        rig.s.hash_block_size, rig.s.mamba_partial_cache_hit = self.UNIT, False
        rig.c.minimum_stage_rows = self.MINIMUM
        self.assertFalse(rig.c._append_yield_short_tail(rig.s, 176 * 3 + 17))
        request = NS(num_tokens=176 * 3 + 17, num_prompt_tokens=176 * 3 + 17)
        self.assertIs(rig.c._append_yield_split_view(rig.s, request), request)

    def test_under_one_unit_nothing_moves(self):
        rig = self.rig()
        rig.s.hash_block_size, rig.s.mamba_partial_cache_hit = self.UNIT, True
        rig.c.minimum_stage_rows = self.MINIMUM
        request = NS(num_tokens=20, num_prompt_tokens=20)
        self.assertIs(rig.c._append_yield_split_view(rig.s, request), request)


class ShortTailFlow(Base):
    """The real path, small numbers: unit 8, minimum 3, chunks of 4. 26 tokens: last boundary 24, one row past it."""
    UNIT, MINIMUM, TOKENS = 8, 3, 26

    def stock_split(self, request, limit, num_new_local_computed_tokens=0):
        start = request.num_computed_tokens + num_new_local_computed_tokens
        boundary = request.num_prompt_tokens // self.UNIT * self.UNIT
        self.seen.append((request.request_id, boundary))
        return boundary - start if start < boundary < start + limit else limit

    def setup_rig(self, tokens=None, cached=8):
        self.seen = []
        rig = self.rig(max_new=64)
        rig.admit()
        rig.s.hash_block_size, rig.s.mamba_partial_cache_hit = self.UNIT, True
        rig.c.minimum_stage_rows = self.MINIMUM
        rig.s._mamba_block_aligned_split = self.stock_split
        rig.ticks(rig.stage0_released)
        append = rig.add_held(APPEND_UID, self.TOKENS if tokens is None else tokens, cached=cached)
        append.num_prompt_tokens = append.num_tokens
        return rig, append

    def test_a_short_tail_is_selected_and_its_last_chunk_crosses_the_old_boundary(self):
        rig, append = self.setup_rig()
        rig.d._yield_offer()
        self.assertIsNotNone(rig.d.append_yield_phase())
        self.assertFalse(rig.c.yield_refusals)
        ay.to_append(rig)
        self.seen.clear()
        starts = []
        real = rig.c._aligned_split

        def spy(scheduler, request, n, num_new_local_computed_tokens=0):
            starts.append(request.num_computed_tokens + num_new_local_computed_tokens)
            return real(scheduler, request, n, num_new_local_computed_tokens=num_new_local_computed_tokens)
        rig.c._aligned_split = spy
        ay.run_append(rig)
        rig.c._aligned_split = real
        self.assertIsNone(rig.d.error)
        self.assertEqual(append.num_computed_tokens, self.TOKENS - 1, 'native to the last row but one: the prompt is whole before handback')
        boundaries = {boundary for uid, boundary in self.seen if uid == APPEND_UID}
        self.assertEqual(boundaries, {16}, 'every chunk of it planned over the moved stop')
        self.assertFalse(rig.c.yield_refusals)

    def test_the_bulks_own_split_never_sees_the_view(self):
        rig, _append = self.setup_rig()
        rig.d._yield_offer()
        ay.to_append(rig)
        self.assertTrue(rig.d.append_yield_is_append(rig.d._yield['append']))
        self.assertFalse(rig.d.append_yield_is_append(rig.bulk))
        ay.run_append(rig)
        self.assertFalse(rig.d.append_yield_is_append(rig.d._yield['append'] if rig.d._yield else None))
        self.assertFalse(any(uid == BULK_UID and boundary != 32 for uid, boundary in self.seen), self.seen[:6])

    def test_a_hit_inside_the_last_unit_is_refused_and_named(self):
        rig, _append = self.setup_rig(cached=24)
        rig.d._yield_offer()
        self.assertIsNone(rig.d.append_yield_phase())
        self.assertEqual(rig.c.yield_refusals, {'no-plan': 1})


class Merge(Base):
    def test_merge_is_off_in_the_full_window(self):
        rig = self.rig()
        rig.d.decode_merge = 'chunk'
        rig.admit()
        rig.finish_chunk()
        self.assertTrue(rig.d._decode_merge_wanted())
        rig.add_held(APPEND_UID, 11, cached=8)
        rig.d.advance()
        ay.to_append(rig)
        ay.run_append(rig)
        self.assertEqual(rig.d.append_yield_phase(), 'handback')
        self.assertFalse(rig.d._decode_merge_wanted())


class InWait(Base):
    """A yield under real progress_in_wait calls inside the stock wait: the commit, the admission and the phases stay on the tick, the frames advance in the wait."""

    def run_with_waits(self, rig, *, hold_at_chunk=2, early=True):
        d, c = rig.d, rig.c
        # Leaf: the executor hook strips this meta on the real path; the rig's fake workers take the call directly.
        patcher = mock.patch.object(ed.NativeProgress, 'with_meta', staticmethod(lambda envelope, _meta: envelope))
        patcher.start()
        self.addCleanup(patcher.stop)
        real_submit = ay.StagedDirectTransport.submit

        def submit(transport, kind, send, *args, direct=False, **kwargs):      # leaf: the rig's direct transport has one path
            return real_submit(transport, kind, send, *args, **kwargs)
        patch_submit = mock.patch.object(ay.StagedDirectTransport, 'submit', submit)
        patch_submit.start()
        self.addCleanup(patch_submit.stop)
        d.install_progress('burst')
        d.progress.mode = 'burst'
        d.inwait_installed = True
        d.inwait = True
        d.early_successor = early
        rig.admit()
        served, moved, held, append = False, [], False, None
        for _ in range(4000):
            if rig.bulk.num_computed_tokens >= rig.bulk.num_tokens - 1:
                break
            if not held and rig.bulk.num_computed_tokens >= 4 * hold_at_chunk and d._frames:
                held = True
                append = rig.add_held(APPEND_UID, 11, cached=8)
            phase = d.append_yield_phase()
            if phase == 'handback' and not d._split_ready and not d.active and not served:
                served = True
                rig.stock_handback(d._yield['append'])
            with auto_arm_workers():
                rig.p.tick()
                sh.flash_s5_admit(rig.s)
                for _call in range(3):
                    if not d._frames:
                        break
                    d.progress.last_stock_kind = 'sample_tokens'
                    before = d.append_yield_phase(), c.overlap_counters['successors'], rig.bulk.num_computed_tokens
                    sent = d.progress_in_wait()
                    self.assertEqual((d.append_yield_phase(), c.overlap_counters['successors'], rig.bulk.num_computed_tokens), before, 'the wait moved a phase, an admission or a commit')
                    if sent is None:
                        break
                    moved.append(phase)
                    rig.stack.complete_everything()
                d.advance()
                rig.stack.complete_everything()
        self.assertIsNone(d.error)
        self.assertTrue(held and served, (held, served, d.append_yield_phase(), rig.kinds()[-8:]))
        self.assertEqual(d.append_yield_counters['reopened'], 1, d.append_yield_counters)
        self.assertEqual(rig.bulk.num_computed_tokens, rig.bulk.num_tokens - 1)
        return moved

    def test_the_wait_drives_the_drain_and_the_flow_ends_where_it_should(self):
        rig = self.rig()
        moved = self.run_with_waits(rig)
        self.assertIn('draining', moved, 'in-wait advanced a frame while the yield drained')
        applied = sum((rig.d.inwait_counters.get('wait_applied') or {}).values()) + sum((rig.d.inwait_counters.get('wait_sent') or {}).values())
        self.assertGreater(applied, 0, rig.d.inwait_counters)
        self.assertEqual(rig.d.append_yield_arrivals['in-chunk'] + sum(v for k, v in rig.d.append_yield_arrivals.items() if k != 'in-chunk'), 1)

    def test_without_the_early_successor(self):
        moved = self.run_with_waits(self.rig(), early=False)
        self.assertIn(None, moved, 'in-wait ran with no yield too')


if __name__ == '__main__':
    unittest.main()
