"""Actual API-owned teacher plan over NativeControlClient; CPU only."""
import sys
from pathlib import Path
import types
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'patches/s5_engine'),str(ROOT/'harness')]
import native_qualification as q
from native_overlap_control import NativeControlClient, NativeControlError

class Processor:
    def process_inputs(self,*args,**kwargs):
        raise AssertionError('an empty plan configuration must not submit a request')

class CoreClient:
    client_count=1
    client_index=0
    async def call_utility_async(self,*args):
        raise AssertionError('frontend ownership must not call the engine utility')

class ApiPlanTests(unittest.TestCase):
    def test_actual_api_client_owns_the_real_input_processor_plan(self):
        async def scheduling_forbidden(*args,**kwargs):
            raise AssertionError('plan ownership must never pause, resume or query scheduling')
        processor=Processor()
        engine=types.SimpleNamespace(engine_core=CoreClient(),input_processor=processor,
            is_paused=scheduling_forbidden,pause_generation=scheduling_forbidden,
            resume_generation=scheduling_forbidden)
        control=NativeControlClient(engine)
        owner=q.own_frontend_plan(control)
        self.assertIs(processor._flash_s5_teacher,owner)
        self.assertIs(q.own_frontend_plan(control),owner)
        result=q.configure_frontend(control,None)
        self.assertIs(result['enabled'],False)
        self.assertIs(q.frontend_status(control)['enabled'],False)
        self.assertFalse(hasattr(control,'driver'))

    def test_core_object_and_probe_shapes_are_refused(self):
        with self.assertRaises(NativeControlError):
            NativeControlClient(CoreClient())
        with self.assertRaises(q.NativeQualificationUnavailable):
            q.own_frontend_plan(types.SimpleNamespace(engine=types.SimpleNamespace(input_processor=Processor())))
        core=CoreClient();core._flash_s5_driver=object()
        with self.assertRaises(NativeControlError):
            NativeControlClient(types.SimpleNamespace(engine_core=core,input_processor=Processor()))

if __name__=='__main__':unittest.main()
