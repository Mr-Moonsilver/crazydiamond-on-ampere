"""decode-phase hold: a waiting request the native lane can take stays off stock for its arm's native_hold_s; the last row's
handback goes ahead of what the split held. CPU only: the real AutoArmPolicy over compose-5's stand-in scheduler, the real
stock seam, and the real PrefillAdmissionController's terminal commit."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'patches/s5_engine')]

import native_auto_arm as auto  # noqa: E402
import scheduler_hooks as sh  # noqa: E402
import test_compose5_declines as rig  # noqa: E402
import test_e3_admission_native as adm  # noqa: E402


def stage(*heads, hold=30, clock=None):
    s = rig.Stage(*heads)
    s.controller.native_hold_s = hold
    s.now = [0.0]
    s.driver.clock = clock or (lambda: s.now[0])
    return s


class Hold(unittest.TestCase):
    def test_off_is_todays_behaviour(self):
        s = stage(rig.req(), hold=0)
        self.assertFalse(s.policy.holds_from_stock(s.queue[0]))
        self.assertEqual(s.policy.decline_record()['holds']['held'], 0)

    def test_a_plannable_fresh_is_held_then_released_at_its_time(self):
        s = stage(rig.req())
        r = s.queue[0]
        self.assertTrue(s.policy.holds_from_stock(r))
        s.now[0] = 30.0
        self.assertTrue(s.policy.holds_from_stock(r))
        s.now[0] = 30.5
        self.assertFalse(s.policy.holds_from_stock(r))     # stock takes it
        s.now[0] = 31.0
        self.assertFalse(s.policy.holds_from_stock(r))     # and keeps it: never held twice
        holds = s.policy.decline_record()['holds']
        self.assertEqual((holds['held'], holds['expired'], holds['waiting'], holds['limit_s']), (1, 1, 0, 30))

    def test_never_held_with_nothing_running(self):
        s = stage(rig.req())
        s.scheduler.running = []
        self.assertFalse(s.policy.holds_from_stock(s.queue[0]))

    def test_an_append_waits_too_the_floor_is_the_staged_minimum(self):
        s = stage(rig.req(tokens=70000))
        s.controller.minimum_stage_rows = 64
        s.scheduler._sf_uncached = lambda r: 2000           # an append: a cached prefix and 2000 new rows
        self.assertTrue(s.policy.holds_from_stock(s.queue[0]))
        self.assertTrue(s.policy.tick())                    # and native plans it
        self.assertEqual(s.policy.decline_record()['holds']['selected'], 1)

    def test_under_the_staged_minimum_goes_stock_by_name(self):
        s = stage(rig.req(tokens=70000))
        s.controller.minimum_stage_rows = 64
        s.scheduler._sf_uncached = lambda r: 60
        self.assertFalse(s.policy.holds_from_stock(s.queue[0]))
        self.assertEqual(s.policy.decline_record()['holds']['released'], {'below-minimum': 1})

    def test_off_the_floor_is_todays(self):
        s = stage(rig.req(tokens=2000), hold=0)
        self.assertFalse(s.policy.tick())
        self.assertEqual(s.policy.decline_record()['ticks']['plan-below-floor'], 1)
        self.assertEqual(s.policy.snapshot()['minimum_native_rows'], auto.MIN_NATIVE_ROWS)

    def test_a_capability_native_lacks_goes_stock_by_name(self):
        s = stage(rig.req(sampling_params=rig.NS(prompt_logprobs=1, logprobs=None)))
        self.assertFalse(s.policy.holds_from_stock(s.queue[0]))
        self.assertEqual(s.policy.decline_record()['holds']['released'], {'plan-state': 1})

    def test_manual_lease_goes_stock(self):
        s = stage(rig.req())
        s.policy.manual_owned = True
        self.assertFalse(s.policy.holds_from_stock(s.queue[0]))

    def test_a_refused_selection_is_retried_then_goes_stock(self):
        s = stage(rig.req())
        r = s.queue[0]
        self.assertTrue(s.policy.holds_from_stock(r))
        for attempt in range(auto.MAX_RETRIES):
            s.policy.rejected[r.request_id] = auto.weakref.ref(r)
            self.assertTrue(s.policy.holds_from_stock(r))
            s.policy.phase = 'idle'
            self.assertTrue(s.policy.tick())                # selected again
            s.policy.phase, s.policy.cooldown = 'idle', False
            s.policy.holds[r.request_id] = (auto.weakref.ref(r), 0.0)
        s.policy.rejected[r.request_id] = auto.weakref.ref(r)
        self.assertFalse(s.policy.tick())
        self.assertFalse(s.policy.holds_from_stock(r))
        holds = s.policy.decline_record()['holds']
        self.assertEqual((holds['retried'], holds['retries_spent'], holds['released']), (auto.MAX_RETRIES, 1, {'rejected': 1}))

    def test_kv_blocks_wait_rather_than_go_stock(self):
        s = stage(rig.req())
        r = s.queue[0]
        s.policy.holds_from_stock(r)
        s.free = 0
        self.assertFalse(s.policy.tick())
        self.assertTrue(s.policy.holds_from_stock(r))
        s.free = 10 ** 7
        self.assertTrue(s.policy.tick())

    def test_a_held_request_selected_natively_counts_once(self):
        s = stage(rig.req())
        r = s.queue[0]
        s.policy.holds_from_stock(r)
        self.assertTrue(s.policy.tick())
        self.assertIn(s.policy.phase, ('drain', 'barrier'))     # selected: the cycle is the policy's now
        holds = s.policy.decline_record()['holds']
        self.assertEqual((holds['held'], holds['selected'], holds['waiting']), (1, 1, 0))

    def test_a_held_request_whose_plan_fails_goes_stock_next_step(self):
        s = stage(rig.req())
        r = s.queue[0]
        s.policy.holds_from_stock(r)
        s.shared = 1                                       # plan-shared at selection
        self.assertFalse(s.policy.tick())
        self.assertFalse(s.policy.holds_from_stock(r))
        self.assertEqual(s.policy.decline_record()['holds']['unplannable'], 1)

    def test_while_a_stock_prefill_runs_it_waits_then_goes_native(self):
        s = stage(rig.req())
        r = s.queue[0]
        s.scheduler.running = [rig.decoder(), rig.decoder('p1', is_prefill_chunk=True, num_computed_tokens=5, num_prompt_tokens=9000)]
        self.assertTrue(s.policy.holds_from_stock(r))
        self.assertFalse(s.policy.tick())                  # cohort-prefill: today stock would take it here
        self.assertTrue(s.policy.holds_from_stock(r))      # held instead
        s.scheduler.running = [rig.decoder(), rig.decoder('p1')]
        self.assertTrue(s.policy.tick())
        self.assertEqual(s.policy.decline_record()['holds']['selected'], 1)

    def test_an_error_lets_stock_take_it(self):
        s = stage(rig.req(), clock=lambda: 1 / 0)
        self.assertFalse(s.policy.holds_from_stock(s.queue[0]))
        self.assertEqual(s.policy.decline_record()['holds']['errors'], 1)

    def test_holds_are_bounded(self):
        heads = [rig.req('r%d' % i) for i in range(auto.MAX_HOLDS + 3)]
        s = stage(*heads)
        held = [s.policy.holds_from_stock(r) for r in heads]
        self.assertEqual(sum(held), auto.MAX_HOLDS)


class Seam(unittest.TestCase):
    def test_the_fresh_seam_defers_a_held_request_and_counts_it(self):
        s = stage(rig.req())
        controller = NS(native_hold_deferrals=0)
        scheduler = NS(_flash_s5_driver=NS(auto_arm=s.policy), _flash_s5_admission=controller)
        self.assertTrue(sh._native_hold(scheduler, s.queue[0]))
        self.assertEqual(controller.native_hold_deferrals, 1)
        s.controller.native_hold_s = 0
        self.assertFalse(sh._native_hold(scheduler, s.queue[0]))

    def test_the_stock_loop_seam_reaches_it(self):
        s = stage(rig.req())
        scheduler = NS(_flash_s5_driver=NS(auto_arm=s.policy))
        self.assertTrue(sh.defer_stock_prefix_hit(scheduler, None, request=s.queue[0]))
        s.controller.native_hold_s = 0
        self.assertFalse(sh.defer_stock_prefix_hit(scheduler, None, request=s.queue[0]))

    def test_no_policy_no_hold(self):
        self.assertFalse(sh._native_hold(NS(), rig.req()))
        self.assertFalse(sh._native_hold(NS(_flash_s5_driver=NS(auto_arm=None)), rig.req()))


class Handback(unittest.TestCase):
    def terminal(self, hold):
        request = adm.Request(tokens=5)
        other = adm.Request(uid='held-by-split', tokens=5)
        scheduler = adm.Scheduler(adm.Manager(), [request])
        scheduler.skipped_waiting = adm.Queue([other])
        controller = adm.make_controller(scheduler, chunk=8)
        controller.native_hold_s = hold
        record = controller.admit(scheduler)
        self.assertEqual(record.remaining_tokens, 1)
        transaction = record.transaction
        controller.bind(adm.make_ticket(transaction, 0, 0, adm.DependencyRole.PRODUCER))
        for report in adm.reports_for(transaction):
            controller.note_report(report)
            controller.note_worker_release(adm.release_ack(transaction, report, epoch=record.state_epoch, terminal=True))
        controller.commit(transaction, adm.make_proof(transaction, 0, 0, adm.DependencyRole.PRODUCER, adm.CompletionKind.GPU_EVENT))
        return request, other, scheduler, controller

    def test_off_the_last_row_returns_where_it_was(self):
        request, other, scheduler, controller = self.terminal(0)
        self.assertEqual(scheduler.waiting.items, [request])
        self.assertEqual(scheduler.skipped_waiting.items, [other])
        self.assertEqual(controller.handback_front, 0)

    def test_on_the_last_row_goes_ahead_of_what_the_split_held(self):
        request, other, scheduler, controller = self.terminal(30)
        self.assertEqual(scheduler.skipped_waiting.items, [request, other])
        self.assertFalse(scheduler.waiting)
        self.assertEqual(controller.handback_front, 1)


if __name__ == '__main__':
    unittest.main()
