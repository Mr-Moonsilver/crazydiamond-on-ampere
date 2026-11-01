"""Native40: physical split-bank prefill exclusion. CPU leaves declared separately."""
import gc
import sys
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import weakref

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'patches/s5_engine'),str(ROOT/'harness')]
import scheduler_hooks as sh
from e3_types import E3ContractError
import test_native39_trunk as prior

class UnarmedHold(prior.Session):
    async def split(self):
        await self.call('prepare',target_route='split',phase='split-w1-decode')
        self.assertIsNone(self.driver.admission_intent())
        self.scheduler.requests={self.request.request_id:self.request}

    async def test_all_stock_prefill_branches_hold_without_native_admission(self):
        await self.split()
        running=prior.adm.Request(uid='n40-running',tokens=40,computed=8)
        running.is_prefill_chunk=True
        resumed=prior.adm.Request(uid='n40-resumed',tokens=40,computed=8)
        self.scheduler.running.append(running)
        self.scheduler.waiting.items.append(resumed)
        self.scheduler.requests.update({r.request_id:r for r in (running,resumed)})
        admitted_before=self.controller.admitted
        for _ in range(3):
            admission,held=self.schedule_pass(('fresh',self.request),('positive-frontier',resumed),('running-prefill',running))
            self.assertIsNone(admission)
            self.assertEqual(held,[True,True,True])
        self.assertEqual((self.request.num_computed_tokens,resumed.num_computed_tokens,running.num_computed_tokens),(0,8,8))
        self.assertIsNone(self.controller.parked)
        self.assertEqual(self.controller._schedule_seq,3)
        self.assertEqual(self.controller.admitted,admitted_before)

    async def test_unarmed_timeout_returns_full_without_inventing_native_refusal(self):
        await self.split()
        before=self.status()['refused']
        self.controller.split_hold_bound_ns=0
        self.assertEqual(self.schedule_pass(('fresh',self.request)),(None,[True]))
        await prior.settle(lambda:not self.driver._split_ready)
        status=self.status()
        self.assertEqual(status['refused'],before)
        self.assertEqual(status['last_split_hold_timeout']['reason'],'unarmed-stock-prefill-hold-timeout')
        self.assertFalse(status['last_split_hold_timeout']['admission_armed'])
        self.assertEqual(status['split_hold_timeouts'],1)
        self.assertFalse(sh.defer_stock_prefix_hit(self.scheduler,None,request=self.request))
        self.assertEqual(self.request.num_computed_tokens,0)
        self.assertIsNone(self.driver.admission_intent())

    async def test_same_generation_disarm_does_not_reset_the_hold_age(self):
        await self.split()
        anchor=time.monotonic_ns()
        now=[anchor]
        self.controller._now_ns=lambda:now[0]
        self.driver._monotonic_ns=lambda:now[0]
        self.controller.minimum_stage_rows=sh.QSA_STAGE_MIN_ROWS
        self.schedule_pass(('fresh',self.request))
        since=self.controller._split_hold['since_ns']
        self.core.paused=True
        self.driver.submit_control({'action':'arm_admission','phase':'split-prefill-solo'})
        self.core.paused=False
        now[0]+=10_000_000_000
        self.schedule_pass(('fresh',self.request))
        self.core.paused=True
        self.driver.submit_control({'action':'disarm_admission'})
        self.core.paused=False
        now[0]+=10_000_000_000
        self.schedule_pass(('fresh',self.request))
        self.assertEqual(self.controller._split_hold['since_ns'],since)
        self.assertFalse(self.controller._split_hold['admission_armed'])
        self.assertFalse(self.driver._split_hold_timeout())
        now[0]+=11_000_000_000
        self.assertTrue(self.driver._split_hold_timeout())
        await prior.settle(lambda:not self.driver._split_ready)

    async def test_aborted_or_replaced_uid_does_not_return_live_decoders_to_full(self):
        for replaced in (False,True):
            with self.subTest(replaced=replaced):
                await self.split()
                self.schedule_pass(('fresh',self.request))
                self.controller.split_hold_bound_ns=0
                decode=prior.adm.Request(uid='n40-live-decode',tokens=40,computed=39)
                self.scheduler.requests={decode.request_id:decode}
                if replaced:
                    other=prior.adm.Request(uid=self.request.request_id,tokens=40)
                    self.scheduler.requests[other.request_id]=other
                self.assertFalse(self.driver._split_hold_timeout())
                self.assertTrue(self.driver._split_ready)
                self.assertIsNone(self.controller._split_hold)

    async def test_running_prefill_that_became_decode_is_not_a_stale_hold(self):
        await self.split()
        running=prior.adm.Request(uid='n40-running',tokens=40,computed=8)
        running.is_prefill_chunk=True
        self.scheduler.running.append(running)
        self.scheduler.requests[running.request_id]=running
        self.schedule_pass(('running-prefill',running))
        self.controller.split_hold_bound_ns=0
        running.is_prefill_chunk=False
        self.assertFalse(self.driver._split_hold_timeout())
        self.assertTrue(self.driver._split_ready)

    async def test_every_native_ownership_boundary_excludes_timeout(self):
        await self.split()
        self.schedule_pass(('fresh',self.request))
        self.controller.split_hold_bound_ns=0
        guards={'tx':object(),'_retained':object(),'stop_kind':'abort','_control':object(),
                '_inflight':{1:'poll'},'_return_pending':'handback','_transition':object()}
        for field,value in guards.items():
            with self.subTest(field=field),patch.object(self.driver,field,value):
                self.assertFalse(self.driver._split_hold_timeout())
        with patch.object(self.controller,'parked',object()):
            self.assertFalse(self.driver._split_hold_timeout())
        self.core.paused=True
        self.assertFalse(self.driver._split_hold_timeout())
        self.core.paused=False
        self.assertTrue(self.driver._split_hold_timeout())
        await prior.settle(lambda:not self.driver._split_ready)

    async def test_no_hold_visit_and_a_new_route_generation_end_the_old_budget(self):
        await self.split()
        self.schedule_pass(('fresh',self.request))
        self.controller.split_hold_bound_ns=0
        sh.flash_s5_admit(self.scheduler)
        self.assertFalse(self.driver._split_hold_timeout())
        self.schedule_pass(('fresh',self.request))
        with patch.object(self.driver,'route_generation',self.driver.route_generation+1):
            self.assertFalse(self.driver._split_hold_timeout())

    async def test_full_and_absent_owner_stay_stock_foreign_owner_refuses(self):
        self.assertFalse(sh.defer_stock_prefix_hit(self.scheduler,None))
        await self.split()
        with patch.object(self.scheduler,'_flash_s5_driver',None):
            self.assertFalse(sh.defer_stock_prefix_hit(self.scheduler,None))
        with patch.object(self.driver,'engine_epoch',self.driver.engine_epoch+1):
            with self.assertRaises(E3ContractError):sh.defer_stock_prefix_hit(self.scheduler,None)
        await self.call('leave_full')
        self.assertFalse(sh.defer_stock_positive_frontier(self.scheduler,self.request))

    async def test_later_arrival_after_all_held_clients_abort_gets_a_new_budget(self):
        await self.split()
        now=[time.monotonic_ns()]
        self.controller._now_ns=lambda:now[0]
        self.schedule_pass(('fresh',self.request))
        old_since=self.controller._split_hold['since_ns']
        self.scheduler.requests.clear()
        self.scheduler.waiting.items.clear()
        self.assertFalse(self.driver._split_hold_timeout())
        now[0]+=31_000_000_000
        later=prior.adm.Request(uid='n40-later',tokens=40)
        self.scheduler.requests[later.request_id]=later
        self.scheduler.waiting.items.append(later)
        self.schedule_pass(('fresh',later))
        self.assertGreater(self.controller._split_hold['since_ns'],old_since)
        self.assertEqual(self.controller._split_hold['since_ns'],now[0])

    async def test_hold_identity_is_weak(self):
        await self.split()
        request=prior.adm.Request(uid='n40-gone',tokens=40,computed=8)
        self.scheduler.waiting.items.append(request)
        self.scheduler.requests[request.request_id]=request
        self.schedule_pass(('positive-frontier',request))
        ref=weakref.ref(request)
        self.scheduler.waiting.items.remove(request)
        self.scheduler.requests.pop(request.request_id)
        del request
        gc.collect()
        self.assertIsNone(ref())
        self.controller.split_hold_bound_ns=0
        self.assertFalse(self.driver._split_hold_timeout())

if __name__=='__main__':unittest.main()
