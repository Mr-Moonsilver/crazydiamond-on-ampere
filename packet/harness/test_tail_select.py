"""coverage-throughput build-2 (b) K2: two back-to-back auto-arm runs on the append-yield Rig (real policy, real driver,
real bridge workers).

The lead (6 Nov): in the tail phase, once stock has scheduled every handback row of the run's request, the policy closes
with no cooldown and selects the next waiting head in the same tick. The handed-back request is then a decoder in the
next run's cohort, proven by the same drain, barrier and identity. Also: the two receipt lines (FULL ack, tail end) that
split the unlogged ~144 ms on the box. Unchanged: output already out or request gone -> the old close with its cooldown;
a preempted (WAITING) or partly scheduled tail keeps waiting.

python3 -m unittest test_tail_select (from harness/).
"""
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import test_append_yield as ay
from test_append_yield import BULK_UID, Pause, Rig, ed

NEXT_UID = 'rb-next'


class Log:
    """Leaf: vLLM's init_logger logger behind engine_driver._progress_logger; keeps the formatted lines."""

    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args)

    warning = info


def mirror_stock_pause(rig):
    """Leaf: stock Scheduler.set_pause_state (scheduler.py.v9:2803) sets the state that pause_state reads back;
    the rig's core_for only records it."""
    core = rig.core

    def set_pause_state(state):
        core.pause_states.append(state)
        core.paused = state != 'UNPAUSED'
        rig.s.pause_state = state
    rig.s.set_pause_state = set_pause_state


def scheduled(rig, request, rows=None):
    """Leaf: the pinned stock step on FULL that schedules the request's last prompt rows (all of them, or ``rows``
    of a two-step geometry tail); the step is in flight (batch_queue holds it), no output token yet."""
    if rows is None:
        rig.stock_handback(request)                     # queues -> running, worker rows, computed = num_tokens
        request.num_output_tokens = 0
    else:
        request.num_computed_tokens += rows
        request.is_prefill_chunk = True
        request.status = NS(name='RUNNING')
    rig.core.batch_queue = ['stock-handback-step']


def landed(rig, request):
    """Leaf: the in-flight step's output is processed: batch_queue drains, the first token is out."""
    rig.core.batch_queue = []
    request.num_output_tokens = 1


class BackToBack(unittest.TestCase):

    def setUp(self):
        self.log = Log()
        patcher = mock.patch.object(ed, '_progress_logger', lambda: self.log)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_to_tail(self, waiting_next=True):
        rig = Rig(cached={}, enabled=False)
        mirror_stock_pause(rig)
        rig.c.native_hold_s = 30         # the box's native hold (decode-phase): the plan floor is the staged minimum
        nxt = rig.add_held(NEXT_UID, 12, queue='waiting') if waiting_next else None
        ay.finish_bulk(rig)
        rig.tick(lambda: rig.p.phase == 'tail')
        self.assertEqual(rig.bulk.num_output_tokens, 0)
        self.assertLess(rig.bulk.num_computed_tokens, rig.bulk.num_prompt_tokens)
        return rig, nxt

    def test_select_at_the_scheduled_handback_and_arm_with_it_in_the_cohort(self):
        rig, nxt = self.run_to_tail()
        p = rig.p
        selection = p.selection_seq
        self.assertEqual([l for l in self.log.lines if 'tail full-ack' in l],
                         ['flash-s5 tail full-ack selection=%d computed=%d prompt=%d'
                          % (selection, rig.bulk.num_computed_tokens, rig.bulk.num_prompt_tokens)])
        ay.policy_ticks(rig, 2)                         # the handback row not yet scheduled: the tail waits
        self.assertEqual(p.phase, 'tail')
        scheduled(rig, rig.bulk)
        ay.policy_ticks(rig, 1)
        # one tick: closed without cooldown, the next head selected, the cohort is the handed-back request
        self.assertEqual(p.phase, 'drain')
        self.assertEqual(p.restores, {'full-tail-scheduled': 1})
        self.assertIs(p.request_ref(), nxt)
        self.assertEqual(p.cohort, (BULK_UID,))
        self.assertEqual(p.selection_seq, selection + 1)
        self.assertEqual(p.old_pause, Pause.UNPAUSED)  # read after the close put the old state back
        self.assertEqual(rig.s.pause_state, Pause.PAUSED_ALL)
        self.assertEqual(p.decline_ticks['cooldown'], 0)
        self.assertEqual([l for l in self.log.lines if 'tail end' in l],
                         ['flash-s5 tail end reason=full-tail-scheduled selection=%d ticks=4 output_tokens=0' % selection])   # entry tick + 2 + this
        ay.policy_ticks(rig, 3)                         # the drain waits for the in-flight handback step
        self.assertEqual(p.phase, 'drain')
        landed(rig, rig.bulk)
        rig.tick(lambda: p.phase == 'native')
        self.assertTrue(rig.d._split_ready)
        self.assertEqual(p.cohort, (BULK_UID,))
        self.assertEqual(p.decline_ticks['cooldown'], 0)
        self.assertEqual(p.decline_totals['armed'], 1)
        self.assertEqual(rig.s.pause_state, Pause.UNPAUSED)
        txn = rig.admit()                               # the next run's first chunk is the selected head's
        self.assertIsNotNone(txn)
        self.assertEqual(txn.request_uid, NEXT_UID)
        self.assertIn(rig.bulk, rig.s.running)

    def test_a_geometry_tail_over_two_steps_selects_after_the_second(self):
        rig, nxt = self.run_to_tail()
        missing = rig.bulk.num_prompt_tokens - rig.bulk.num_computed_tokens
        self.assertEqual(missing, 1)                    # the rig's handback is one row; split it as two steps
        rig.bulk.num_computed_tokens -= 1               # leaf: a 2-row geometry tail, one row per step
        scheduled(rig, rig.bulk, rows=1)
        ay.policy_ticks(rig, 2)
        self.assertEqual(rig.p.phase, 'tail', 'one row left: not every handback row is scheduled')
        rig.bulk.is_prefill_chunk = False
        scheduled(rig, rig.bulk)
        ay.policy_ticks(rig, 1)
        self.assertEqual((rig.p.phase, rig.p.request_ref()), ('drain', nxt))

    def test_a_preempted_handback_waits(self):
        rig, _ = self.run_to_tail()
        scheduled(rig, rig.bulk)
        rig.bulk.status = NS(name='WAITING')            # leaf: stock preempted it before the step ran
        ay.policy_ticks(rig, 3)
        self.assertEqual(rig.p.phase, 'tail')
        self.assertEqual(rig.p.restores, {})

    def test_output_already_out_keeps_the_old_close_and_its_cooldown(self):
        rig, nxt = self.run_to_tail()
        scheduled(rig, rig.bulk)
        landed(rig, rig.bulk)
        ay.policy_ticks(rig, 1)
        self.assertEqual(rig.p.restores, {'full-tail-completed': 1})
        self.assertEqual(rig.p.phase, 'idle')
        self.assertTrue(rig.p.cooldown)
        self.assertIn('flash-s5 tail end reason=full-tail-completed', ' '.join(self.log.lines))
        ay.policy_ticks(rig, 1)
        self.assertEqual(rig.p.decline_ticks['cooldown'], 1)

    def test_no_head_to_select_leaves_the_policy_idle_without_cooldown(self):
        rig, _ = self.run_to_tail(waiting_next=False)
        scheduled(rig, rig.bulk)
        ay.policy_ticks(rig, 1)
        self.assertEqual((rig.p.phase, rig.p.restores, rig.p.cooldown), ('idle', {'full-tail-scheduled': 1}, False))
        self.assertEqual(rig.s.pause_state, Pause.UNPAUSED)
        ay.policy_ticks(rig, 2)
        self.assertEqual(rig.p.decline_ticks['cooldown'], 0)

    def test_a_logger_failure_never_stops_the_tick(self):
        rig, nxt = self.run_to_tail()
        with mock.patch.object(ed, '_progress_logger', side_effect=RuntimeError('logger')):
            scheduled(rig, rig.bulk)
            ay.policy_ticks(rig, 1)
        self.assertEqual((rig.p.phase, rig.p.request_ref()), ('drain', nxt))
        self.assertGreaterEqual(rig.p.decline_totals['read_errors'], 1)


class Predicate(unittest.TestCase):
    """_handback_scheduled on request states. Stock sets is_prefill_chunk = computed < num_tokens + placeholders when it
    schedules (scheduler.py:1698), so 'RUNNING, no chunk, computed short' is not a stock state; the computed check is a
    guard for it and is held here on its own."""

    def req(self, status, chunk, computed, prompt=12):
        return NS(status=NS(name=status), is_prefill_chunk=chunk, num_computed_tokens=computed, num_prompt_tokens=prompt)

    def test_states(self):
        f = ay.auto.AutoArmPolicy._handback_scheduled
        self.assertTrue(f(self.req('RUNNING', False, 12)))
        self.assertTrue(f(self.req('RUNNING', False, 13)))         # a decode step scheduled after it
        self.assertFalse(f(self.req('RUNNING', True, 11)))         # the first step of a two-step tail
        self.assertFalse(f(self.req('WAITING', False, 12)))        # preempted
        self.assertFalse(f(self.req('WAITING', False, 11)))        # not yet scheduled (the native run's end)
        self.assertFalse(f(self.req('RUNNING', False, 11)))        # the guard (not a stock state)


class StagedBackToBack(ay.StagedDriver, BackToBack):
    """The same on the box's StagedSegmentDriver."""

    def setUp(self):
        ay.StagedDriver.setUp(self)
        BackToBack.setUp(self)


if __name__ == '__main__':
    unittest.main()
