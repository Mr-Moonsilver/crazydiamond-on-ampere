"""CPU-only actual controller/route paths with a delayed fake AsyncLLM client."""
import asyncio
import copy
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
from probe_control import ProbeController, ProbeControlError, attach_router, PATH


class Engine:
    def __init__(self):
        self.engine_core = types.SimpleNamespace(client_count=1, client_index=0)
        self.vllm_config = types.SimpleNamespace(parallel_config=types.SimpleNamespace(
            tensor_parallel_size=2, pipeline_parallel_size=2, data_parallel_size=1))
        self.route, self.generation = 'full', 0
        self.paused = False
        self.calls = []
        self.unfinished, self.queued = 8, 0
        self.corrupt = None
        self.pause_gate = None
        self.rpc_failure = False
        self.last_decode = True

    def get_num_unfinished_requests(self): return self.unfinished
    def get_num_queued_tokens(self): return self.queued
    async def is_paused(self): return self.paused

    async def pause_generation(self, **kwargs):
        self.calls.append(('pause', kwargs))
        if self.pause_gate is not None:
            await self.pause_gate.wait()
        self.paused = True

    async def resume_generation(self):
        self.calls.append(('resume',))
        self.paused = False

    async def collective_rpc(self, method, *, timeout, args, kwargs):
        self.calls.append(('rpc', method, copy.deepcopy(args)))
        if self.rpc_failure:
            raise RuntimeError('private failure data')
        c = args[0]
        assert method == 'flash_s5_decode_probe' and kwargs == {}
        if c['action'] == 'status':
            rows = []
            for rank in range(4):
                identity = dict(route=self.route, route_generation=self.generation,
                                storage_epoch=1, mode='decode-only-probe')
                rows.append({'rank':rank, 'inventory':dict(identity),
                             'counters':dict(identity, state='ready', quarantined=None,
                                 in_flight=0, sampling_pending=False, transition_pending=False,
                                 batch_observations={'last':{'sequence':10, 'decode_only':self.last_decode,
                                 'full_graph':True, 'num_reqs':8, 'num_tokens':8}})})
        else:
            if c['action'] == 'transition':
                assert self.paused and c['expected_route_generation'] == self.generation
                self.route, self.generation = c['target'], c['route_generation']
            rows = [dict(rank=r, ack='flash_s5_transition' if c['action'] == 'transition' else 'flash_s5_probe_barrier',
                         mode='decode-only-probe', engine_epoch=c['engine_epoch'], storage_epoch=1,
                         route=self.route, route_generation=self.generation, drained_streams=3) for r in range(4)]
        if self.corrupt is not None:
            rows = self.corrupt(c, rows)
        return rows


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = Engine()
        self.ctl = ProbeController(self.engine, epoch=71)
        self.seq = 0
        self.opened = await self.ctl.execute({'action':'open','run_id':'fixture-run'})

    async def cmd(self, action, **kwargs):
        self.seq += 1
        return await self.ctl.execute(dict(action=action, run_id='fixture-run', engine_epoch=71,
                                           seq=self.seq, **kwargs))

    async def test_full_split_full_actual_rpc_order_and_typed_acks(self):
        pause = await self.cmd('pause')
        self.assertEqual(pause['pause'], {'mode':'keep','clear_cache':False,'completed':True})
        transition = await self.cmd('transition', target='split', expected_route_generation=0, cohort_size=8)
        self.assertEqual(len(transition['replies']), 4)
        self.assertTrue(transition['decode_eligibility']['proved'])
        self.assertTrue(self.engine.paused)
        await self.cmd('resume', cohort_size=8)
        self.assertFalse(self.engine.paused)
        await self.cmd('pause')
        left = await self.cmd('transition', target='full', expected_route_generation=1, cohort_size=8)
        self.assertEqual(left['snapshot']['route_generation'], 2)
        await self.cmd('resume', cohort_size=8)
        self.assertFalse(self.engine.paused)
        self.assertFalse(self.ctl.quarantined)
        pauses = [c for c in self.engine.calls if c[0] == 'pause']
        self.assertTrue(all(c[1] == {'mode':'keep','clear_cache':False} for c in pauses))

    async def test_pause_receipt_waits_for_engine_completion(self):
        self.engine.pause_gate = asyncio.Event()
        task = asyncio.create_task(self.cmd('pause'))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertFalse(self.ctl.paused)
        self.assertEqual(self.engine.calls[-1][0], 'pause')
        self.engine.pause_gate.set()
        result = await task
        self.assertTrue(result['pause']['completed'])

    async def test_no_transition_or_resume_without_completed_pause(self):
        with self.assertRaisesRegex(ProbeControlError, 'completed-pause'):
            await self.cmd('transition', target='split', expected_route_generation=0, cohort_size=8)
        self.assertEqual(self.engine.route, 'full')
        self.assertTrue(self.ctl.quarantined)

    async def test_stale_seq_epoch_and_arbitrary_rpc_refused(self):
        for bad in ({'seq':True}, {'engine_epoch':72}, {'seq':4}, {'run_id':'other'},
                    {'method':'arbitrary'}, {'action':'arbitrary'}):
            command = dict(action='status',run_id='fixture-run',engine_epoch=71,seq=1)
            command.update(bad)
            with self.assertRaises(ProbeControlError):
                await self.ctl.execute(command)
        self.assertEqual(self.ctl.seq, 0)

    async def test_duplicate_rank_and_mixed_generation_refused(self):
        for corrupt in (lambda c,rs:rs[:-1]+[rs[0]],
                        lambda c,rs:rs[:-1]+[dict(rs[-1], counters=dict(rs[-1]['counters'],route_generation=1))]):
            ctl = ProbeController(self.engine, epoch=72)
            self.engine.corrupt = corrupt
            with self.assertRaises(ProbeControlError):
                await ctl.execute({'action':'open','run_id':'new'})
            self.assertTrue(ctl.quarantined)
        self.engine.corrupt = None

    async def test_partial_transition_ack_never_resumes(self):
        await self.cmd('pause')
        self.engine.corrupt = lambda c,rs:rs[:-1] if c['action']=='transition' else rs
        with self.assertRaises(ProbeControlError):
            await self.cmd('transition',target='split',expected_route_generation=0,cohort_size=8)
        with self.assertRaisesRegex(ProbeControlError, 'quarantined'):
            await self.cmd('resume',cohort_size=8)
        self.assertTrue(self.engine.paused)
        self.assertFalse(any(c[0]=='resume' for c in self.engine.calls))

    async def test_decode_eligibility_observes_worker_and_live_client(self):
        for field,value in [('queued',1),('unfinished',7),('last_decode',False)]:
            engine = Engine(); setattr(engine, field, value)
            ctl = ProbeController(engine, epoch=72)
            await ctl.execute({'action':'open','run_id':'new'})
            await ctl.execute(dict(action='pause',run_id='new',engine_epoch=72,seq=1))
            with self.assertRaises(ProbeControlError):
                await ctl.execute(dict(action='transition',run_id='new',engine_epoch=72,seq=2,
                                       target='split',expected_route_generation=0,cohort_size=8))
            self.assertEqual(engine.route,'full')

    async def test_timeout_consumes_sequence_and_quarantines(self):
        self.ctl.timeout_s = .01
        self.engine.pause_gate = asyncio.Event()
        with self.assertRaisesRegex(ProbeControlError, 'TimeoutError'):
            await self.cmd('pause')
        self.assertEqual(self.ctl.seq,1)
        self.assertTrue(self.ctl.quarantined)
        with self.assertRaisesRegex(ProbeControlError,'quarantined'):
            await self.cmd('resume',cohort_size=8)

    async def test_full_control_resume_has_real_barrier_without_fake_transition(self):
        await self.cmd('pause')
        receipt = await self.cmd('resume',cohort_size=8)
        self.assertEqual(receipt['snapshot']['route_generation'],0)
        self.assertEqual(len(receipt['barriers']),4)
        self.assertFalse(any(c[0]=='rpc' and c[2][0]['action']=='transition' for c in self.engine.calls))

    async def test_response_is_detached_from_controller_state(self):
        result = await self.cmd('status')
        result['snapshot']['route'] = 'changed'
        self.assertEqual(self.ctl.last['snapshot']['route'],'full')

    async def test_remote_error_payload_not_exposed(self):
        self.engine.rpc_failure = True
        with self.assertRaises(ProbeControlError) as caught:
            await self.cmd('status')
        self.assertNotIn('private failure data',str(caught.exception))

    async def test_wrong_topology_or_multiple_frontends_refused(self):
        self.engine.engine_core.client_count = 2
        with self.assertRaises(ProbeControlError): ProbeController(self.engine)
        self.engine.engine_core.client_count = 1
        self.engine.vllm_config.parallel_config.pipeline_parallel_size = 1
        with self.assertRaises(ProbeControlError): ProbeController(self.engine)


class HttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        class Router:
            def add_api_route(self,path,fn,methods):
                if not hasattr(self, 'path'): self.path,self.fn,self.methods=path,fn,methods
        class Response:
            def __init__(self,value,status_code=200): self.value,self.status_code=value,status_code
        self.fake = types.ModuleType('fastapi'); self.fake.APIRouter=Router; self.fake.Request=object
        self.responses = types.ModuleType('fastapi.responses'); self.responses.JSONResponse=Response
        self.app = types.SimpleNamespace(state=types.SimpleNamespace(engine_client=Engine()))
        self.app.include_router = lambda router:setattr(self.app,'router',router)
        self.args = types.SimpleNamespace(api_key=['fixture-key'])
        self.env = {'FLASH_S5_DECODE_ONLY_PROBE':'1','FLASH_S5_OVERLAP':'0','FLASH_S5_DECODE_SMS':'44'}
        self.modules = patch.dict(sys.modules,{'fastapi':self.fake,'fastapi.responses':self.responses})
        self.modules.start(); self.addCleanup(self.modules.stop)
        self.state = attach_router(self.args,self.app,self.env)

    async def request(self, command, *, host='127.0.0.1', auth='Bearer fixture-key', raw=None):
        data = json.dumps(command).encode() if raw is None else raw
        async def stream():
            yield data[:3]; yield data[3:]
        request = types.SimpleNamespace(client=types.SimpleNamespace(host=host),headers={'authorization':auth},
                                        stream=stream,app=self.app,url=types.SimpleNamespace(path=PATH))
        return await self.app.router.fn(request)

    async def test_authenticated_local_open_uses_fixed_actual_controller(self):
        self.assertEqual(self.app.router.path, PATH)
        response = await self.request({'action':'open','run_id':'http-fixture'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(len(response.value['snapshot']['ranks']),4)

    async def test_wrong_auth_and_remote_client_do_not_construct_controller(self):
        for host,auth in [('127.0.0.1','Bearer wrong'),('192.0.2.1','Bearer fixture-key')]:
            response = await self.request({'action':'open','run_id':'x'},host=host,auth=auth)
            self.assertEqual(response.status_code,403)
        self.assertIsNone(self.state['controller'])

    async def test_invalid_or_large_json_is_refused(self):
        for data,code in [(b'{',400),(b' '*16385,409)]:
            response = await self.request(None,raw=data)
            self.assertEqual(response.status_code,code)
        self.assertIsNone(self.state['controller'])

    async def test_default_off_no_registration_and_missing_auth_refuses(self):
        app = types.SimpleNamespace()
        self.assertIsNone(attach_router(self.args,app,{}))
        with self.assertRaisesRegex(ProbeControlError,'api-auth'):
            attach_router(types.SimpleNamespace(api_key=None),app,self.env)


class SourceTests(unittest.TestCase):
    def test_pinned_frontend_transform_inverse_and_actual_added_call(self):
        import ast
        import hashlib
        import probe_control_hooks as hooks
        for path, digest in hooks.PINS.items():
            self.assertEqual(hashlib.sha256((ROOT / 'reference/vllm-src' / path).read_bytes()).hexdigest(), digest)
        original = (ROOT / 'reference/vllm-src' / hooks.PATH).read_text()
        generated = hooks.generate(original)
        self.assertEqual(hooks.revert(generated), original)
        with self.assertRaises(ValueError): hooks.generate(generated)
        call = next(n for n in ast.walk(ast.parse(generated))
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == '_flash_s5_probe_router')
        calls = []
        args, app = object(), object()
        ns = {'args':args,'app':app,'_flash_s5_probe_router':lambda *a:calls.append(a)}
        eval(compile(ast.Expression(call),'<generated frontend>','eval'),ns)
        self.assertEqual(calls,[(args,app)])

    def test_frontend_refuses_dev_mode_or_string_key_configuration(self):
        env = {'FLASH_S5_DECODE_ONLY_PROBE':'1','FLASH_S5_OVERLAP':'0','FLASH_S5_DECODE_SMS':'44'}
        with self.assertRaisesRegex(ProbeControlError,'dev-router'):
            attach_router(types.SimpleNamespace(api_key=['fixture-key']),object(),dict(env,VLLM_SERVER_DEV_MODE='1'))
        with self.assertRaisesRegex(ProbeControlError,'auth-shape'):
            attach_router(types.SimpleNamespace(api_key='fixture-key'),object(),env)


if __name__ == '__main__': unittest.main()
