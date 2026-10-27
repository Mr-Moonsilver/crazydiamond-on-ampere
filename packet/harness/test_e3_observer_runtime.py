"""Real q17b observer orchestration; only native work and AsyncLLM are faked."""
import hashlib
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from test_e3_probe_session import CountedEngine, SessionTests
from test_e3_probe_control import ProbeController, ProbeControlError
from probe_session import ProbeSession
import observer_runtime as runtime
import worker_hooks
from replay_observer import ReplayObserverError
from test_e3_replay_observer import ObserverTestBase, CONTEXT_ID, EXPECTED_SMS

class Session:
    def __init__(self):
        self.state='created'; self.events=[]; self._native_reserved=1024
        self.binding=types.SimpleNamespace(identity=lambda:{'abi':1}); self.fail_stop=False
    def start(self): self.events.append('start'); self.state='started'
    def bind_bank(self,probe): self.events.append('bind'); self.state='bound'
    def begin_window(self,**kw): self.events.append('begin'); self.state='window'
    def replay_census(self): return {'total':0,'graphs':{}}
    def end_window(self,proof):
        self.events.append('end'); self.state='ended'
        return {'schema':'s5ro-replay-observer/1','verdict':'PASS',
                'static_structure':{'verdict':'PASS'},'replay':{'verdict':'PASS'},'caller_proof':proof.as_record()}
    def stop(self):
        self.events.append('stop')
        if self.fail_stop: raise ReplayObserverError('cleanup_failed')
        self.state='stopped'
    def close(self): self.events.append('close'); self.state='closed'

def probe(session):
    def drain(): session.events.append('drain'); return 3
    return types.SimpleNamespace(route='split',route_generation=1,storage_epoch=1,in_flight=0,
        runner=types.SimpleNamespace(execute_model_state=None),
        controller=types.SimpleNamespace(admitting=True,drain=drain),_require_not_down=lambda:None)

class WorkerTests(unittest.TestCase):
    def test_real_window_order_and_worker_proof(self):
        session=Session(); owner=runtime.WorkerObserver(session,0); owner.start(); p=probe(session)
        begun=owner.begin(p,7); ended=owner.end(p,7)
        self.assertEqual(session.events,['start','drain','bind','begin','drain','end','stop','close'])
        self.assertEqual(ended['receipt']['caller_proof']['relevant_consumers'],3)
        self.assertEqual(ended['receipt']['caller_proof']['engine_incarnation'],'engine:7')
        self.assertEqual(begun['worker_incarnation'],ended['worker_incarnation'])
        self.assertTrue(owner.status()['closed'])
        with self.assertRaises(ReplayObserverError): owner.begin(p,7)
    def test_bad_drain_or_generation_never_completes(self):
        session=Session(); owner=runtime.WorkerObserver(session,1); owner.start(); p=probe(session); owner.begin(p,7)
        p.route_generation=2
        with self.assertRaises(ReplayObserverError): owner.end(p,7)
        p.route_generation=1; p.controller.drain=lambda:0
        with self.assertRaises(ReplayObserverError): owner.end(p,7)
        self.assertIsNone(owner.receipt); self.assertEqual(session.state,'window')
    def test_stop_failure_retains_owner(self):
        session=Session(); owner=runtime.WorkerObserver(session,2); owner.start(); p=probe(session)
        owner.begin(p,7); session.fail_stop=True
        with self.assertRaises(ReplayObserverError): owner.end(p,7)
        self.assertFalse(owner.closed); self.assertNotIn('close',session.events)
    def test_default_off_never_constructs_binding(self):
        def forbidden(*a): raise AssertionError('native construction')
        self.assertIsNone(runtime.start_worker_observer(object(),object(),{},binding_factory=forbidden))
    def test_start_owner_and_late_install_refusal(self):
        session=Session(); runner=types.SimpleNamespace(); worker=types.SimpleNamespace(rank=0)
        env={'FLASH_S5_REPLAY_OBSERVER':'1','FLASH_S5_DECODE_ONLY_PROBE':'1',
             'FLASH_S5_REPLAY_LIB':'/declared/helper.so','FLASH_S5_REPLAY_LIB_SHA256':'a'*64}
        owner=runtime.start_worker_observer(worker,runner,env,binding_factory=lambda *a:object(),session_factory=lambda b:session)
        self.assertIs(runner._flash_s5_replay_observer,owner); self.assertTrue(owner.started_before_split)
        with self.assertRaises(ReplayObserverError): runtime.start_worker_observer(worker,runner,env)
        with self.assertRaises(ReplayObserverError):
            runtime.start_worker_observer(worker,types.SimpleNamespace(_flash_s5_split=object()),env)
    def test_worker_hook_starts_observer_before_split(self):
        events=[]; root=Path(tempfile.mkdtemp(prefix='observer-runtime-')); lib=root/'green.so'
        lib.write_bytes(b'CPU fixture never loaded')
        env={'FLASH_S5_DECODE_ONLY_PROBE':'1','FLASH_S5_OVERLAP':'0','FLASH_S5_DECODE_SMS':'44',
             'FLASH_S5_GREEN_LIB':str(lib),'FLASH_S5_GREEN_LIB_SHA256':hashlib.sha256(lib.read_bytes()).hexdigest()}
        runner=types.SimpleNamespace(); worker=types.SimpleNamespace(rank=0,model_runner=runner,device=types.SimpleNamespace(index=0))
        split=types.SimpleNamespace(membership=lambda:types.SimpleNamespace(ok=True),describe=lambda:{},close=lambda:None)
        with patch('decode_probe.require_decode_probe_ready'), \
             patch('observer_runtime.start_worker_observer',side_effect=lambda *a:events.append('observer') or None), \
             patch('green_context.open_split',side_effect=lambda *a,**k:events.append('split') or split):
            worker_hooks.after_runner_init(worker,env)
        self.assertEqual(events,['observer','split'])


class NativeCompositionTests(ObserverTestBase):
    def test_owner_uses_actual_native_binding_and_retained_bank(self):
        session = self.new_session()
        owner = runtime.WorkerObserver(session, 0)
        owner.start()  # Native subscription really precedes fake green creation.
        self.make_bank()
        self.ctl.stub_emit_context(CONTEXT_ID, 1, EXPECTED_SMS, 0)
        self.probe.in_flight = 0
        self.probe.runner = types.SimpleNamespace(execute_model_state=None)
        drains = []
        self.probe.controller = types.SimpleNamespace(admitting=True,
            drain=lambda: drains.append('synchronized-tail-events') or 3)
        self.probe._require_not_down = lambda: None
        self.ctl.stub_set_timestamp(1000)
        begin = owner.begin(self.probe, 71)
        self.ctl.stub_set_timestamp(2000)
        self.emit_graph_kernels(0)
        self.probe.router.replays = 1
        self.probe.router.replays_by_descriptor[self.bank.pairs[0][1]] = 1
        self.ctl.stub_set_timestamp(3000)
        result = owner.end(self.probe, 71)
        self.assertEqual(result['receipt']['verdict'], 'PASS')
        self.assertEqual(result['receipt']['static_structure']['verdict'], 'PASS')
        self.assertEqual(result['receipt']['caller_proof']['engine_incarnation'], 'engine:71')
        self.assertEqual(result['receipt']['replay']['context']['num_multiprocessors'], EXPECTED_SMS)
        self.assertEqual(len(drains), 2)
        self.assertEqual(self.ctl.stub_has_subscriber(), 0)
        self.assertTrue(owner.closed)
        from e3_q17b_clients import replay_multiplicity, replay_capacity
        self.assertTrue(replay_multiplicity(begin, result, result['receipt']))
        self.assertEqual(replay_capacity(begin,result,result['receipt'])['verdict'],'PASS')
        # A second runtime replay with all its records lost would have passed
        # the former presence-only gate. Missing CUPTI drop metadata cannot
        # now turn that into complete evidence.
        result['replay_census'] = {'total':2,'graphs':{'graph:0':2,'graph:1':0}}
        self.assertFalse(replay_multiplicity(begin, result, result['receipt']))

class NativeLedgerOwnerTests(ObserverTestBase):
    def test_owned_window_drives_real_native_ledger_and_separate_composition(self):
        import ctypes
        import engine_offers as eo
        session = self.new_session()
        owner = runtime.WorkerObserver(session, 0, engine_offers=True)
        owner.start()
        self.make_bank()
        class Descriptor:
            def __init__(self, tokens):
                self.num_tokens=tokens; self.num_reqs=tokens
                self.uniform_token_count=1; self.num_ubatches=1
        for i, (old, local) in enumerate(self.bank.pairs):
            desc=Descriptor(i+1)
            self.bank.manager.graphs[desc]=self.bank.manager.graphs.pop(old)
            self.bank.pairs[i]=(desc,local);self.bank.descriptors[i]=desc
        target=self.bank.descriptors[0]
        stream=types.SimpleNamespace(cuda_stream=101)
        p=self.probe;p._last_decode_descriptor=target;p.in_flight=0
        p.runner=types.SimpleNamespace(execute_model_state=None,main_stream=stream,
            parallel_config=types.SimpleNamespace(pipeline_parallel_size=2),
            is_first_pp_rank=True,is_last_pp_rank=False)
        p.split.decode_stream=stream
        p.controller=types.SimpleNamespace(admitting=True,drain=lambda:3)
        p._require_not_down=lambda:None
        self.ctl.stub_emit_context(CONTEXT_ID,1,EXPECTED_SMS,0)
        def query_identity(stream):
            self.assertEqual(session.state, 'bound', 'setup query must precede the native window')
            return {'context_id':CONTEXT_ID,'stream_id':9}
        with patch.object(session.binding,'stream_identity',side_effect=query_identity):
            self.ctl.stub_set_timestamp(1000)
            begin=owner.begin(p,71,run_id='owner-fixture',window_id='short-placement')
        self.addCleanup(owner.ledger.deactivate)
        self.assertIs(eo.current_ledger(p.runner),owner.ledger)
        eager=self.ctl.stub_emit_correlated
        eager.argtypes=[ctypes.c_uint32,ctypes.c_uint64,ctypes.c_uint32,ctypes.c_uint32,
            ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint64,ctypes.c_uint32]
        graph=self.ctl.stub_emit_correlated_graph
        graph.argtypes=[ctypes.c_uint32,ctypes.c_uint64,ctypes.c_uint32,ctypes.c_uint32,
                       ctypes.c_uint32,ctypes.c_uint64,ctypes.c_uint64]
        with owner.ledger.step('execute'):
            with eo.leaf(p.runner,'request_state_table','prep_state'):
                offer=owner.bridge.offers[-1]
                eager(10,offer['id'],4,offer['thread'],1100,1400,CONTEXT_ID,0,0,1)
            with owner.ledger.offer_scope('forward_dispatch','graph_replay',
                    descriptor=eo.graph_descriptor(None,target),census={'kernel':2,'memcpy':0,'memset':0}):
                offer=owner.bridge.offers[-1]
                graph(11,offer['id'],offer['thread'],CONTEXT_ID,11,101,102)
        p.router.replays=1;p.router.replays_by_descriptor[self.bank.pairs[0][1]]=1
        self.ctl.stub_set_timestamp(3000)
        result=owner.end(p,71)
        self.assertEqual(result['receipt']['verdict'],'INCONCLUSIVE')
        self.assertEqual(result['engine_accounting']['verdict'],'PASS',result['engine_accounting'])
        self.assertEqual(result['composed_placement']['verdict'],'PASS',result['composed_placement'])
        binding=result['engine_accounting']['binding']
        self.assertEqual(binding['run_id'],'owner-fixture')
        self.assertEqual(binding['router_census']['total'],1)
        self.assertEqual(binding['pp_stage'],0)
        self.assertIsNone(eo.current_ledger(p.runner))
        self.assertTrue(owner.closed)

    def test_missing_target_refused_before_window(self):
        session=self.new_session();owner=runtime.WorkerObserver(session,0,engine_offers=True)
        owner.start();self.make_bank();p=self.probe;p.in_flight=0
        p.runner=types.SimpleNamespace(execute_model_state=None)
        p.controller=types.SimpleNamespace(admitting=True,drain=lambda:3);p._require_not_down=lambda:None
        with self.assertRaisesRegex(ReplayObserverError,'engine_target_unproved'):
            owner.begin(p,71,run_id='owner-fixture',window_id='short-placement')
        self.assertEqual(session.state,'bound')

class StreamIdentityTests(unittest.TestCase):
    def test_green_identity_uses_converted_context_and_ordinary_uses_primary(self):
        import ctypes as c
        from replay_observer import ReplayBinding
        for green in (0,55):
            seen=[]
            def query(stream,primary,out):
                c.cast(primary,c.POINTER(c.c_void_p))[0]=11
                c.cast(out,c.POINTER(c.c_void_p))[0]=green
                return 0
            def convert(out,handle):
                self.assertEqual(handle.value,55)
                c.cast(out,c.POINTER(c.c_void_p))[0]=22
                return 0
            def cid(context,out):
                seen.append(context.value);c.cast(out,c.POINTER(c.c_uint32))[0]=42 if green else 41
                return 0
            def sid(context,stream,flag,out):
                self.assertIsNone(context);self.assertEqual(flag,0)
                c.cast(out,c.POINTER(c.c_uint32))[0]=9
                return 0
            b=ReplayBinding.__new__(ReplayBinding);b._c=c
            b._lib=types.SimpleNamespace(cuStreamGetCtx_v2=query,cuCtxFromGreenCtx=convert,
                                        cuptiGetContextId=cid,cuptiGetStreamIdEx=sid)
            self.assertEqual(b.stream_identity(types.SimpleNamespace(cuda_stream=7)),
                             {'context_id':42 if green else 41,'stream_id':9})
            self.assertEqual(seen,[22 if green else 11])

class NativeLaunchComposition(ObserverTestBase):
    @patch("graph_router.descriptor_name",side_effect=lambda d:"descriptor:"+str(id(d)))
    def test_late_enable_full_sample_and_split_report_same_bank(self,_name):
        session=self.new_session();owner=runtime.WorkerObserver(session,0);owner.start();self.make_bank()
        self.bank.stock=self.bank.manager
        p=self.probe;p.route='full';p.in_flight=0;p.runner=types.SimpleNamespace(execute_model_state=None)
        p.controller=types.SimpleNamespace(admitting=True,drain=lambda:3);p._require_not_down=lambda:None
        self.ctl.stub_emit_context(CONTEXT_ID,1,EXPECTED_SMS,0)
        self.ctl.stub_set_timestamp(1000);owner.launch_begin(p,71)
        import ctypes
        token=ctypes.c_uint64()
        self.assertNotEqual(session.binding._lib.s5ro_begin_window(session._token,1,ctypes.byref(token)),0)
        self.assertEqual(token.value,0)
        self.ctl.stub_set_timestamp(2000);self.emit_graph_kernels(0)
        self.ctl.stub_set_timestamp(3000);owner.launch_end(p,71)
        p.route='split';self.ctl.stub_set_timestamp(4000);owner.begin(p,71)
        self.ctl.stub_set_timestamp(5000)
        self.emit_kernel(11,101,start=5000,end=5100,completed=5200);self.emit_kernel(11,102,start=5000,end=5100,completed=5200)
        self.ctl.stub_set_timestamp(6000);result=owner.end(p,71)
        self.assertEqual(result['launch_config']['verdict'],'PASS',result['launch_config'])
        self.assertFalse(result['launch_config']['gates_w1'])
        self.assertTrue(owner.closed)

class ObservedEngine(CountedEngine):
    def __init__(self):
        super().__init__(); self.phase='started'; self.obs_calls=[]; self.bad_proof=False; self.bad_incarnation=False
    async def collective_rpc(self,*args,**kwargs):
        command=kwargs['args'][0]; action=command['action']
        if action in ('launch_begin','launch_end'):
            return [dict(rank=r,ack='flash_s5_'+action,engine_epoch=command['engine_epoch'],drained_streams=3) for r in range(4)]
        if action.startswith('observer_'):
            assert self.paused and self.route=='split'
            self.obs_calls.append(action); end=action=='observer_end'; self.phase='closed' if end else 'window'; rows=[]
            for rank in range(4):
                row={'rank':rank,'ack':'flash_s5_'+action,'engine_epoch':command['engine_epoch'],
                     'route_generation':self.generation,'storage_epoch':1,'drained_streams':3,
                     'state':self.phase,'worker_incarnation':('foreign-' if self.bad_incarnation else 'worker-')+str(rank)}
                if end:
                    row.update(observer_stopped=True,receipt={'schema':'s5ro-replay-observer/1','verdict':'PASS',
                        'caller_proof':{'worker_rank':rank,'worker_incarnation':'worker-'+str(rank),
                        'engine_incarnation':'engine:'+str(command['engine_epoch']),'route_generation':self.generation,
                        'storage_epoch':1,'relevant_consumers':3,'controller_drained':not self.bad_proof}})
                rows.append(row)
            return rows
        rows=await super().collective_rpc(*args,**kwargs)
        if action=='status':
            for row in rows: row['inventory']['replay_observer']={'enabled':True,'state':self.phase,
                'closed':self.phase=='closed','started_before_split':True,
                'worker_incarnation':'worker-'+str(row['rank'])}
        return rows

class CompositionTests(SessionTests):
    async def asyncSetUp(self):
        await super().asyncSetUp(); self.engine=ObservedEngine()
        self.session=ProbeSession(ProbeController(self.engine,epoch=71))
    async def test_once_then_unprofiled_later_split_window(self):
        for repeat in range(2):
            self.ctx.update(repeat=repeat,cohort_id='cohort-'+str(repeat))
            await self.enter('split'); self.engine.steps+=5; end=await self.action('pause_end','split')
            if repeat==0:
                self.assertTrue(end['observer']['stopped']); self.assertEqual(len(end['observer']['ranks']),4)
            else: self.assertNotIn('observer',end)
            await self.action('leave_full','full'); await self.action('resume','full'); self.engine.unfinished=0
        self.assertEqual(self.engine.obs_calls,['observer_begin','observer_end'])
    async def test_changed_incarnation_cannot_receive_audit_ack(self):
        await self.action('prepare', 'full')
        self.engine.unfinished = 8
        self.engine.bad_incarnation = True
        with self.assertRaises(ProbeControlError):
            await self.action('pause_enter_split', 'split')
        self.assertTrue(self.session.controller.quarantined)
        self.assertTrue(self.engine.paused)

    async def test_bad_worker_proof_quarantines_without_resume(self):
        await self.enter('split'); self.engine.steps+=5; self.engine.bad_proof=True
        with self.assertRaises(ProbeControlError): await self.action('pause_end','split')
        self.assertTrue(self.session.controller.quarantined); self.assertTrue(self.engine.paused)

if __name__=='__main__': unittest.main()
