"""Native41 host control/ownership/default-off proofs. Device leaves declared."""
import ast
import gc
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as NS
import weakref
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'patches/s5_engine'),str(ROOT/'harness')]
import native_auto_arm as auto
import engine_driver as ed
import executor_hooks as eh
import scheduler_hooks as sh
from e3_types import E3ContractError
import test_e3_native_overlap as native
import test_e3_admission_native as adm
import test_native39_trunk as prior


class SettingAndStartup(unittest.TestCase):
    def test_zero_is_inert_without_touching_engine(self):
        class Untouched:
            def __getattribute__(self,name):raise AssertionError(name)
        for env in ({},{auto.ENV:'0'}):self.assertIsNone(auto.install(Untouched(),Untouched(),env))
        for bad in ('true','',1,None):
            with self.subTest(bad=bad),self.assertRaises(E3ContractError):auto.enabled({auto.ENV:bad})

    def test_real_startup_order_and_no_eager_scheduling_during_install(self):
        source=(ROOT/'reference/vllm-src/v1/engine/core.py').read_text()
        generated=eh.generate_core(source)
        cls=next(n for n in ast.parse(generated).body if isinstance(n,ast.ClassDef) and n.name=='EngineCore')
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        after=[n.lineno for n in ast.walk(init) if isinstance(n,ast.Call) and ast.unparse(n.func)=='_flash_s5_hooks.after_engine_startup']
        queues=[n.lineno for n in ast.walk(init) if isinstance(n,(ast.Assign,ast.AnnAssign)) and any(ast.unparse(t)=='self.batch_queue' for t in (n.targets if isinstance(n,ast.Assign) else [n.target]))]
        self.assertEqual(len(after),1);self.assertGreater(after[0],max(queues))
        engine=NS(scheduler=object(),_flash_s5_startup_ready={'engine_epoch':7})
        driver=NS(controller=object())
        policy=auto.install(engine,driver,{auto.ENV:'1'})
        self.assertIs(driver.auto_arm,policy)
        self.assertFalse(hasattr(engine,'batch_queue'))
        self.assertFalse(policy.active)

    def test_auto_is_bound_after_real_startup_fence(self):
        tree=ast.parse((ROOT/'patches/s5_engine/s5_startup.py').read_text())
        install=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install_engine')
        calls=[(n.lineno,ast.unparse(n.func)) for n in ast.walk(install) if isinstance(n,ast.Call)]
        self.assertEqual(len([n for n in ast.walk(install) if isinstance(n,ast.ImportFrom) and n.module=='native_auto_arm']),1)
        self.assertLess(next(line for line,name in calls if name=='bootstrap_full_startup'),max(line for line,name in calls if name=='install'))
        with self.assertRaises(E3ContractError):auto.install(NS(),NS(),{auto.ENV:'1'})


class Ownership(unittest.TestCase):
    def setUp(self):
        self.stack=native.cold_stack()
        self.core,self.frontend=native.core_for(self.stack,paused=True)
        self.s=self.stack.scheduler;self.d=self.stack.driver
        self.request=self.s.waiting.peek_request()
        self.s.requests={self.request.request_id:self.request}
        self.control({'action':'enter_split'})
        self.p=auto.AutoArmPolicy(self.core,self.d)
        self.d.auto_arm=self.p
        self.p.request_ref=weakref.ref(self.request)
        self.p.selection_seq=9

    def control(self,command):
        result=self.d.submit_control(command)
        for _ in range(300):
            if not hasattr(result,'done') or result.done():return result.result() if hasattr(result,'result') else result
            self.d.control_tick()
            self.stack.complete_everything()
        self.fail('control never settled')

    def test_cycle_refuses_mutating_manual_control_and_pause_but_keeps_health_status(self):
        self.p.phase='native'
        for action in ('enter_split','leave_split','arm_admission','disarm_admission','barrier','cancel',auto.RELEASE):
            with self.subTest(action=action),self.assertRaisesRegex(E3ContractError,'cycle-owned'):
                self.d.submit_control({'action':action})
        status=self.d.submit_control({'action':'status'})
        self.assertTrue(status['auto_arm']['active'])
        with self.assertRaisesRegex(E3ContractError,'cycle-owned'):eh.before_pause(self.core,'keep')
        self.assertIsNone(self.d.stop_kind)
        self.assertFalse(self.p.manual_owned)

    def test_manual_lease_survives_full_until_explicit_safe_release(self):
        self.control({'action':'leave_split'})
        self.assertTrue(self.p.manual_owned)
        self.assertFalse(self.d._split_ready)
        self.core.paused=False
        with self.assertRaisesRegex(E3ContractError,'paused-idle-full'):self.d.submit_control({'action':auto.RELEASE,'engine_epoch':self.d.engine_epoch})
        self.core.paused=True
        self.core.batch_queue=[]
        result=self.d.submit_control({'action':auto.RELEASE,'engine_epoch':self.d.engine_epoch})
        self.assertTrue(result['released']);self.assertFalse(self.p.manual_owned)
        eh.before_pause(self.core,'keep')
        self.assertTrue(self.p.manual_owned)

    def test_stale_manual_release_cannot_enable_another_engine(self):
        self.control({'action':'leave_split'})
        self.core.batch_queue=[]
        with self.assertRaisesRegex(E3ContractError,'epoch-invalid'):
            self.d.submit_control({'action':auto.RELEASE,'engine_epoch':self.d.engine_epoch+1})
        self.assertTrue(self.p.manual_owned)

    def test_only_private_policy_can_create_auto_intent(self):
        with self.assertRaises(ed.DriverError):self.d.submit_control({'action':'arm_admission','phase':auto.PHASE})
        self.assertFalse(self.p.manual_owned)
        self.p.phase='split';self.p._internal=True
        try:intent=self.d._arm_admission(auto.PHASE)
        finally:self.p._internal=False
        self.p.phase='native'
        self.assertEqual(intent['origin'],'auto-arm');self.assertEqual(intent['selection_seq'],9)
        self.assertEqual(intent['request_uid'],self.request.request_id)
        self.assertFalse(self.stack.controller._native_admission_disabled(self.s))
        other=adm.Request(uid=self.request.request_id,tokens=12)
        self.s.requests[other.request_id]=other
        self.assertFalse(self.p.permits(other))
        with self.assertRaises(E3ContractError):self.stack.controller._native_admission_disabled(self.s)

    def test_foreign_head_is_not_admitted_into_an_auto_cycle(self):
        self.p.phase='native'
        other=adm.Request(uid='foreign',tokens=12)
        self.s.waiting.prepend_request(other)
        self.s.requests[other.request_id]=other
        before=self.stack.controller.admitted
        self.assertIsNone(self.stack.controller.admit(self.s))
        self.assertIsNone(self.stack.controller.parked)
        self.assertEqual(self.stack.controller.admitted,before)

    def test_auto_phase_keeps_paused_core_awake(self):
        self.p.phase='drain'
        self.assertTrue(eh.native_work_pending(self.core))
        self.p.phase='idle'
        self.assertFalse(eh.native_work_pending(self.core))

    def test_rejection_memo_is_weak_bounded_and_does_not_bind_reused_uid(self):
        r=adm.Request(uid='gone',tokens=12)
        ref=weakref.ref(r);self.p.request_ref=ref;self.p._remember()
        del r;gc.collect();self.assertIsNone(ref())
        self.p._remember();self.assertNotIn('gone',self.p.rejected)
        for i in range(auto.MAX_REJECTED+3):
            r=adm.Request(uid=str(i),tokens=12);self.p.request_ref=weakref.ref(r);self.p._remember()
        self.assertLessEqual(len(self.p.rejected),auto.MAX_REJECTED)

class ApiOwnership(prior.Session):
    async def test_public_manual_session_can_release_only_while_enabled_and_paused_full(self):
        await self.call('prepare',target_route='full',phase='full-prefill-solo')
        policy=auto.AutoArmPolicy(self.core,self.driver)
        self.driver.auto_arm=policy
        policy.manual_owned=True
        with patch.dict('os.environ',{'FLASH_S5_AUTO_ARM':'0'}):
            with self.assertRaisesRegex(prior.ns.NativeOverlapSessionError,'session-action-not-exposed'):
                await self.call('release_auto_arm')
        await self.call('pause')
        with patch.dict('os.environ',{'FLASH_S5_AUTO_ARM':'1'}):
            reply=await self.call('release_auto_arm')
        self.assertTrue(reply['released'])
        self.assertFalse(policy.manual_owned)
        self.assertTrue(self.core.paused)
        self.assertFalse(self.driver._split_ready)

if __name__=='__main__':unittest.main()
