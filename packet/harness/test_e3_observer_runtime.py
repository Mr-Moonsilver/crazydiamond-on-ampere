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
        owner.begin(self.probe, 71)
        self.ctl.stub_set_timestamp(2000)
        self.emit_graph_kernels(0)
        self.ctl.stub_set_timestamp(3000)
        result = owner.end(self.probe, 71)
        self.assertEqual(result['receipt']['verdict'], 'PASS')
        self.assertEqual(result['receipt']['static_structure']['verdict'], 'PASS')
        self.assertEqual(result['receipt']['caller_proof']['engine_incarnation'], 'engine:71')
        self.assertEqual(result['receipt']['replay']['context']['num_multiprocessors'], EXPECTED_SMS)
        self.assertEqual(len(drains), 2)
        self.assertEqual(self.ctl.stub_has_subscriber(), 0)
        self.assertTrue(owner.closed)

class ObservedEngine(CountedEngine):
    def __init__(self):
        super().__init__(); self.phase='started'; self.obs_calls=[]; self.bad_proof=False; self.bad_incarnation=False
    async def collective_rpc(self,*args,**kwargs):
        command=kwargs['args'][0]; action=command['action']
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
