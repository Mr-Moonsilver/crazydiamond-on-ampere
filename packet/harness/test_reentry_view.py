"""coverage-throughput build-2b: a yielded bulk re-enters over the view its admission reads, and is never handed to stock.

Build-1b's box run (q18a-compose-throughput, 6 Nov): 19 yields closed 'yielded-bulk-has-no-reentry-plan' and stock carried
the bulks at the positive-frontier seam, as sites 'cooldown' (14 turns) and 'no-seam'. Every one of the 14 cooldown turns has
a short hash tail (10-60 rows past its last 176-row unit). _reentry_plan split the remainder over the raw request; _plan,
admission and the exact successor read controller.plan_view (the hold's moved stop). Here:
  Shapes  the 14 record prompts, at every chunk boundary of the plan their admission makes: the re-entry plan is that plan's
          remainder (raw-view control: no plan, the box failure).
  Held    a forced no-plan close: the bulk is held at the positive-frontier seam (the cooldown tick and after), counted in
          the decline record's yield_bulk_hold and named 'yield-no-reentry-plan'; at the hold's limit stock may take it.
python3 -m unittest test_reentry_view (from harness/).
"""
import unittest

import test_a1_trunk as a1
import test_append_yield as ay
from test_append_yield import Req, Rig
import scheduler_hooks as sh

UNIT, MINIMUM, CHUNK = 176, 64, 2048
BOUND = [52624, 1024]                       # the table's B13 chunk bound
# The prompts of build-1b's 14 'decline site=cooldown' lines (engine-stream.log), nt arms 0 and 1.
RECORD = (121639, 114764, 94202, 126589, 92062, 101589, 105614, 128350, 125179, 59521, 131848, 122507, 103197, 115475)


def stock_split(request, limit, num_new_local_computed_tokens=0):
    """Leaf: scheduler._mamba_block_aligned_split's stop at the prompt's last hash boundary (test_a1_trunk.ShortTail's)."""
    start = request.num_computed_tokens + num_new_local_computed_tokens
    boundary = request.num_prompt_tokens // UNIT * UNIT
    return boundary - start if start < boundary < start + limit else limit


class Shapes(a1.Base):

    def setUp(self):
        super().setUp()
        rig = self.rig()
        rig.s.hash_block_size, rig.s.mamba_partial_cache_hit = UNIT, True
        rig.s._mamba_block_aligned_split = stock_split
        rig.s.max_num_scheduled_tokens = 1 << 20
        rig.c.minimum_stage_rows, rig.c.chunk_tokens, rig.c.chunk_bound = MINIMUM, CHUNK, BOUND
        rig.c.native_hold_s = 600              # nt's hold: plan_view is the moved stop
        self.rig_ = rig

    def bulk(self, tokens, frontier):
        request = Req('rb-bulk-%d' % tokens, tokens)
        request.num_computed_tokens = frontier
        self.rig_.s.requests[request.request_id] = request
        return request

    def admission_plan(self, request, start):
        """The plan the bulk's admission makes from its hit (plan_view, the cap and the bound, as _plan)."""
        rig = self.rig_
        view = rig.c.plan_view(rig.s, request)
        cap = sh.native_chunk_cap(rig.c.chunk_tokens, rig.s.max_num_scheduled_tokens)

        def split(position, limit):
            limit = sh.bounded_limit(rig.c.chunk_bound, position, limit)
            projected = view if position == start else sh._ProjectedPrefillRequest(view, position)
            return rig.c._aligned_split(rig.s, projected, limit)
        return sh.staged_chunk_plan(start, request.num_tokens - 1, cap, MINIMUM, split)

    def test_every_record_shape_re_enters_at_every_boundary_of_its_plan(self):
        for tokens in RECORD:
            self.assertTrue(self.rig_.c._append_yield_short_tail(self.rig_.s, tokens), tokens)
            hit = (tokens - 20000) // UNIT * UNIT              # an append's hit: the earlier turn, at a unit
            request = self.bulk(tokens, 0)
            plan = self.admission_plan(request, hit)
            self.assertTrue(plan, tokens)
            position, checked = hit, 0
            for k, rows in enumerate(plan[:-1]):
                position += rows
                request.num_computed_tokens = position         # the yield's frontier: a chunk boundary
                got = self.rig_.p._reentry_plan(request)
                self.assertIsNotNone(got, (tokens, position))
                self.assertEqual(got['plan'], tuple(plan[k + 1:]), (tokens, position))
                checked += 1
            self.assertGreater(checked, 3, tokens)

    def test_control_the_raw_view_has_no_plan_on_these_shapes(self):
        rig = self.rig_
        bulks = []
        for tokens in RECORD:
            hit = (tokens - 20000) // UNIT * UNIT
            request = self.bulk(tokens, 0)
            plan = self.admission_plan(request, hit)            # the admission's plan, over the moved stop
            request.num_computed_tokens = hit + plan[0]
            bulks.append(request)
        rig.c.plan_view = lambda _scheduler, request: request    # build-1b's re-entry: the raw request
        self.assertEqual([r.num_tokens for r in bulks if rig.p._reentry_plan(r) is None], list(RECORD))


class Held(unittest.TestCase):
    """The lead (6 Nov): a yielded bulk with no re-entry plan waits for native, never continues on stock."""

    def no_plan_close(self, hold_s=600):
        rig = Rig(cached={})
        now = [1000.0]
        rig.d.clock = lambda: now[0]
        rig.c.native_hold_s = hold_s
        append = ay.handback_window(rig)
        rig.stock_handback(append)
        rig.bulk.num_computed_tokens = rig.bulk.num_tokens - 1      # forced: nothing left for a native chunk
        ay.policy_idle(rig)
        self.assertEqual(rig.journal('auto_closed')[-1]['reason'], 'yielded-bulk-has-no-reentry-plan')
        self.assertIn(rig.bulk, rig.s.waiting.items)
        return rig, now

    def test_the_bulk_is_held_at_the_positive_frontier_through_the_cooldown_and_after(self):
        rig, now = self.no_plan_close()
        p = rig.p
        self.assertTrue(p.cooldown)
        for _ in range(3):                                     # the cooldown tick ('cooldown'), then idle ticks ('no-seam')
            self.assertTrue(sh.defer_stock_positive_frontier(rig.s, rig.bulk))
            ay.policy_ticks(rig, 1)
        self.assertTrue(sh.defer_stock_positive_frontier(rig.s, rig.bulk))
        self.assertEqual(p.decline_record()['yield_bulk_hold'], dict(bulks=1, deferrals=4, expired=0, errors=0, waiting=1))
        self.assertEqual(p.stock_site(rig.bulk), 'yield-no-reentry-plan')

    def test_at_the_holds_limit_stock_may_take_it_named(self):
        rig, now = self.no_plan_close()
        self.assertTrue(sh.defer_stock_positive_frontier(rig.s, rig.bulk))
        now[0] += 601
        self.assertFalse(sh.defer_stock_positive_frontier(rig.s, rig.bulk))
        self.assertEqual(rig.p.bulk_hold['expired'], 1)
        self.assertEqual(rig.p.stock_site(rig.bulk), 'yield-no-reentry-plan')
        self.assertFalse(sh.defer_stock_positive_frontier(rig.s, rig.bulk))

    def test_hold_off_and_other_requests_are_unchanged(self):
        rig, _now = self.no_plan_close(hold_s=0)
        self.assertFalse(sh.defer_stock_positive_frontier(rig.s, rig.bulk))
        rig, _now = self.no_plan_close()
        other = Req('rb-other', 12)
        other.num_computed_tokens = 4
        rig.s.requests[other.request_id] = other
        self.assertFalse(sh.defer_stock_positive_frontier(rig.s, other))
        self.assertFalse(sh.defer_stock_positive_frontier(rig.s, None))

    def test_a_failing_read_never_raises(self):
        rig, _now = self.no_plan_close()
        rig.d.clock = None
        self.assertFalse(sh.defer_stock_positive_frontier(rig.s, rig.bulk))
        self.assertEqual(rig.p.bulk_hold['errors'], 1)


class HeldStaged(ay.StagedDriver, Held):
    """The same on the box's StagedSegmentDriver."""


if __name__ == '__main__':
    unittest.main()
