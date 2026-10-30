"""CPU native observer ownership and actual ledger composition, no GPU evidence."""
import hashlib
import inspect
from pathlib import Path
import sys
import time
import types
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'patches/s5_engine'),str(ROOT/'harness')]
import native_observation as obs
import native_prefill_support as support
import observer_runtime
import native_qualification as qualification
from native_startup import WorkerNative
import test_e3_native_accounting_source_closure as closure
from test_e3_native_qualification import make_runner,EPOCH
from test_e3_observer_runtime import Session,probe

class IdentityTests(unittest.TestCase):
    def test_primed_once_and_no_ambient_fallback(self):
        calls=[]
        def query(stream):
            calls.append(stream.cuda_stream);return support.stream_identity(stream)
        ids=obs.PrimedIdentities(query,[support.MAIN,support.MAIN,support.COPY])
        self.assertEqual(len(calls),2)
        ids(support.MAIN)['context_id']=99
        self.assertEqual(ids(support.MAIN),support.stream_identity(support.MAIN))
        self.assertEqual(len(calls),2)
        with self.assertRaisesRegex(obs.NativeObservationError,'not-primed'):ids(support.SE_AUX)
    def test_bad_identity_and_unknown_worker_refused(self):
        for row in ({'context_id':0,'stream_id':1},{'context_id':1}):
            with self.assertRaises(obs.NativeObservationError):obs.PrimedIdentities(lambda _:row,[support.MAIN])
        with self.assertRaisesRegex(obs.NativeObservationError,'worker-owner'):
            obs.NativeObservation(object(),object(),query=lambda _: {},pins={'x':'a'*64},clock=time.monotonic_ns)
    def test_native_clock_requires_actual_success(self):
        class Query:
            failed=False
            def __call__(self,output):output._obj.value=1234;return int(self.failed)
        query=Query();clock=obs.native_clock(types.SimpleNamespace(_lib=types.SimpleNamespace(cuptiGetTimestamp=query)))
        self.assertEqual(clock(),1234);query.failed=True
        with self.assertRaisesRegex(obs.NativeObservationError,'timestamp-unproved'):clock()

class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.bridge=support.FakeBridge();self.torch=support.make_torch()
        factory=self.torch.cuda.Event
        def event(**kwargs):
            item=factory(**kwargs);item.synchronize=lambda:None;return item
        self.torch.cuda.Event=event
        self.parts=support.build_lane(self.bridge,torch_module=self.torch)
        private=closure.LanePrivateLeafTests();private.setUp();self.private=private.bindings
        self.runner=types.SimpleNamespace(_flash_s5_dispatcher=self.parts['dispatcher'])
        owners=types.SimpleNamespace(state='ready',_in_scope=False,
            owners=[(self.parts['shared'],'_stream'),(self.parts['ple'],'_prefetch_stream')],
            replacements=[support.SE_AUX,support.PLE_AUX])
        self.native=WorkerNative(self.runner,types.SimpleNamespace(prefill_stream=support.MAIN),object(),
            backend=object(),controller=object(),comms=object(),ple_layer=0,ple_scratch=None,
            chunk_tokens=8,tp_group=None,torch=self.torch,state=object(),
            private_bindings=self.private,prefill_streams=owners)
        self.native.copy_stream=support.COPY
        targets=[self.parts[k] for k in ('dispatcher','lane','driver','operation','shared','ple')]
        # Explicit fake-boundary source pins, never runtime attestations.
        pins={inspect.getfile(type(t)):hashlib.sha256(Path(inspect.getfile(type(t))).read_bytes()).hexdigest() for t in targets}
        self.owner=obs.NativeObservation(self.native,self.runner,query=support.stream_identity,pins=pins,clock=time.monotonic_ns)
    def begin(self):
        self.owner.begin(self.bridge,worker='w0',rank=0,epoch=self.bridge.epoch,generation=self.bridge.generation,
                         storage_epoch=1,run_id='run-1',window_id='window-1')
    def test_real_ledgers_dynamic_sources_share_one_window(self):
        self.begin();self.assertIs(self.private._ledger,self.owner.ledger)
        for key,kind in (('lane','lane_forward'),('driver','qsa_driver'),('operation','qsa_operation')):
            self.native.observe_prefill(self.parts[key],kind)
        support.submit_all(self.parts['dispatcher'])
        decode,_=support.decode_ledger(self.bridge);support.run_decode_offers(decode,self.bridge);decode.deactivate()
        self.owner.before_end();self.assertIsNone(self.native.observation);self.assertIsNone(self.private._ledger)
        self.assertEqual(self.owner.attachments,{})
        self.bridge.close_window();decode.finish_window({'total':1,'graphs':{'g0':1}})
        result=self.owner.after_end(decode)
        self.assertEqual(result['native_overlap_accounting']['verdict'],'PASS',result['native_overlap_accounting'])
        self.assertGreater(result['native_prefill_accounting']['joined_records'],0)
        self.assertTrue(result['native_source_seams']);self.assertTrue(result['native_overlap_accounting']['raw_unchanged'])
        self.assertNotIn('composed_placement',result) # Raw graph proof remains separate.
    def test_pin_drift_invalidates_and_leaves_original_callable(self):
        self.begin();path=inspect.getfile(type(self.parts['operation']));self.owner.pins[path]='0'*64
        self.native.observe_prefill(self.parts['operation'],'qsa_operation')
        self.assertTrue(self.owner.ledger.window()['invalidated'])
        self.assertFalse(getattr(self.parts['operation'].fill_bucket,'_flash_s5_prefill_shadow',False))
    def test_idempotent_attachment_and_capacity_refusal(self):
        self.begin();count=len(self.owner.attachments)
        self.owner.attach(self.parts['dispatcher'],'lane_dispatcher');self.assertEqual(len(self.owner.attachments),count)
        self.owner.MAX_ATTACHMENTS=count;self.owner.attach(self.parts['operation'],'qsa_operation')
        self.assertIn('native-shadow-capacity',self.owner.ledger.window()['invalid'])

class FullReturnTests(unittest.TestCase):
    def test_window_ends_before_full_return_and_is_not_reended(self):
        session=Session();owner=observer_runtime.WorkerObserver(session,0);owner.start();p=probe(session);p.native=object()
        owner.begin(p,7);owner.native_offers=True;owner._native_probe=p
        owner.before_native_full_return(p.native)
        self.assertTrue(owner.closed);self.assertEqual(owner.completed_reply['route_generation'],1)
        self.assertEqual(session.events[-3:],['end','stop','close'])
        p.route='full';p.route_generation=2;owner.before_native_full_return(p.native)
        self.assertEqual(session.events.count('end'),1);self.assertEqual(owner.completed_reply['route_generation'],1)
    def test_later_worker_read_preserves_window_and_current_identity(self):
        runner=make_runner();session=Session()
        owner=observer_runtime.WorkerObserver(session,0,engine_offers=True,native_offers=True)
        owner.closed=True;owner.engine_epoch=EPOCH;owner.window_identity=('run','window')
        owner.completed_reply={'ack':'flash_s5_observer_end','rank':0,'route_generation':1,
                               'receipt':{'immutable':'raw'},'observer_stopped':True}
        runner._flash_s5_replay_observer=owner;runner.controller.route='full';runner.controller.route_generation=2
        result=qualification.handle_worker(runner,{'action':'observer_end','engine_epoch':EPOCH,
            'route_generation':2,'run_id':'run','window_id':'window'})
        self.assertEqual(result['route_generation'],1);self.assertEqual(result['current_route_generation'],2)
        self.assertEqual(result['current_route'],'full');self.assertEqual(result['receipt'],{'immutable':'raw'})
        self.assertEqual(owner.completed_reply['route_generation'],1)

class RouterCompositionTests(unittest.TestCase):
    def test_actual_default_native_registration_installs_three_narrow_routes(self):
        import probe_control
        import native_qualification_frontend as qf
        import test_e3_native_qualification_frontend as fixtures
        from unittest.mock import patch
        fastapi=types.ModuleType('fastapi');fastapi.APIRouter=fixtures.FakeAPIRouter;fastapi.Request=object
        responses=types.ModuleType('fastapi.responses');responses.JSONResponse=fixtures.FakeJSONResponse
        app=fixtures.FakeApp()
        with patch.dict(sys.modules,{'fastapi':fastapi,'fastapi.responses':responses}):
            state=probe_control.attach_router(types.SimpleNamespace(api_key=['test-only']),app,
                env={'FLASH_S5_OVERLAP':'1'})
        paths=[row[0] for router in app.routers for row in router.routes]
        self.assertEqual(paths,[probe_control.NATIVE_PATH,qf.FRONTEND_PATH,qf.WORKER_PATH])
        self.assertIsNotNone(state['qualification'])
        self.assertIsNone(state['control']) # Lazy setup performs no engine control.

class BridgeBoundaryTests(unittest.TestCase):
    def test_actual_typed_bridge_closes_before_full_and_exact_retry_does_not_repeat(self):
        import test_e3_r4_worker_bridge as b
        stack=b.Stack();helper=b.RouteCompletionTests();helper.bind_split(stack)
        calls=[]
        for rank in stack.driver.ranks:
            native=stack.worker(rank).dispatcher.native
            def before(target,rank=rank,native=native):
                calls.append((rank,target,native.controller.route,native.controller.route_generation))
            native.before_route_change=before
        command=helper.worker_full_command(stack)
        for rank in stack.driver.ranks:
            first=stack.worker(rank).flash_s5_transition(command)
            second=stack.worker(rank).flash_s5_transition(command)
            self.assertEqual(first,second)
        self.assertEqual(calls,[(rank,'full','split',1) for rank in stack.driver.ranks])
    def test_observer_failure_keeps_actual_bridge_route_and_quarantines(self):
        import test_e3_r4_worker_bridge as b
        stack=b.Stack();helper=b.RouteCompletionTests();helper.bind_split(stack)
        worker=stack.worker(0)
        def refuse(_):raise RuntimeError('observer drain failed')
        worker.dispatcher.native.before_route_change=refuse
        with self.assertRaisesRegex(Exception,'route switch failed'):
            worker.flash_s5_transition(helper.worker_full_command(stack))
        self.assertEqual(worker.controller.route,'split')
        self.assertIsNotNone(worker._flash_s5_bridge._route_failure)

if __name__=='__main__':unittest.main()
