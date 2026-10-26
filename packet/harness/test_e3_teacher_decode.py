"""CPU contracts for the probe-only sampler trace seam; no GPU claim."""
import asyncio
import copy
import hashlib
from pathlib import Path
import sys
import types
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
import teacher_decode as td
import probe_control_hooks

def plan():
    return {'schema':td.SCHEMA,'requests':[
        {'request_id':f'teacher-{i}','prompt_sha256':td.digest([i,11,12]),
         'continuation':[i+20]*td.STEPS} for i in range(8)]}
class Params(types.SimpleNamespace):
    def _validate_trace_replay(self,model,speculative):
        if speculative is not None or any(t>=model.get_vocab_size() for t in self.trace_decode_token_ids):
            raise ValueError('private payload must not escape')
def request(i=0):
    return types.SimpleNamespace(request_id=f'cmpl-teacher-{i}-0',prompt_token_ids=[i,11,12],
        sampling_params=Params(trace_decode_token_ids=None,n=1,max_tokens=td.STEPS,min_tokens=td.STEPS,
        temperature=0.,ignore_eos=True,logprobs=20,stop=[],stop_token_ids=[]),lora_request=None,mm_features=None)
class FrontendTests(unittest.TestCase):
    def setUp(self):
        self.events=[]
        def original(req):
            self.events.append('validated')
            if req.sampling_params.trace_decode_token_ids:raise ValueError('stock trace-mode rejection')
            return req
        async def original_async(req):return original(req)
        self.p=types.SimpleNamespace(process_inputs=original,process_inputs_async=original_async,
            model_config=types.SimpleNamespace(get_vocab_size=lambda:100),speculative_config=None)
        self.owner=td.FrontendPlan(self.p)
    def test_private_injection_preserves_every_other_setting(self):
        self.owner.configure(plan())
        for i in range(8):
            r=request(i); before=copy.deepcopy(vars(r.sampling_params))
            self.assertIs(self.p.process_inputs(r),r)
            self.assertEqual(r.sampling_params.trace_decode_token_ids,[20+i]*td.STEPS)
            after=vars(r.sampling_params).copy();after['trace_decode_token_ids']=None
            self.assertEqual(before,after)
            r.request_id+='-randomized'
        self.assertEqual(self.owner.status()['admitted'],8)
        self.assertNotIn('continuation',str(self.owner.status()))
        self.owner.configure(None)
        self.assertIsNone(self.p.process_inputs(request()).sampling_params.trace_decode_token_ids)
    def test_async_bound_method_is_not_a_bypass(self):
        self.owner.configure(plan())
        r=asyncio.run(self.p.process_inputs_async(request()))
        self.assertEqual(r.sampling_params.trace_decode_token_ids,[20]*td.STEPS)
        self.assertEqual(self.events,['validated'])
    def test_default_off_and_public_trace_rejection_unchanged(self):
        r=request();self.assertIs(self.p.process_inputs(r),r)
        r.sampling_params.trace_decode_token_ids=[20]*td.STEPS
        with self.assertRaisesRegex(ValueError,'stock trace-mode'):self.p.process_inputs(r)
    def test_wrong_prompt_foreign_id_duplicate_and_reconfiguration_refused(self):
        self.owner.configure(plan())
        r=request();r.prompt_token_ids=[0,11,13]
        with self.assertRaises(td.TeacherDecodeError):self.p.process_inputs(r)
        r=request();r.request_id='foreign'
        with self.assertRaises(td.TeacherDecodeError):self.p.process_inputs(r)
        self.p.process_inputs(request())
        with self.assertRaises(td.TeacherDecodeError):self.p.process_inputs(request())
        with self.assertRaises(td.TeacherDecodeError):self.owner.configure(plan())
        with self.assertRaises(td.TeacherDecodeError):self.owner.configure(None)
        self.assertEqual(self.owner.status()['pending'],7)
    def test_bounded_typed_plan_and_no_settings_normalization(self):
        for field,value in [('continuation',[1]*(td.STEPS+1)),('continuation',[True]*td.STEPS),
                            ('request_id','invalid/name'),('prompt_sha256','bad')]:
            p=plan();p['requests'][0][field]=value
            with self.assertRaises(td.TeacherDecodeError):td.validate_plan(p)
        self.owner.configure(plan())
        for key,value in [('max_tokens',255),('temperature',.7),('ignore_eos',False),('logprobs',19),('stop',['stop'])]:
            r=request();setattr(r.sampling_params,key,value)
            with self.assertRaises(td.TeacherDecodeError):self.p.process_inputs(r)
        self.p.speculative_config=object();r=request()
        with self.assertRaisesRegex(td.TeacherDecodeError,'compatibility-refused'):self.p.process_inputs(r)
        self.assertIsNone(r.sampling_params.trace_decode_token_ids)
class WorkerTests(unittest.TestCase):
    def test_temporary_sampler_state_restores_original_path(self):
        r=types.SimpleNamespace(is_last_pp_rank=True,_flash_s5_decode_probe=object(),req_states=types.SimpleNamespace(max_num_reqs=1024),
            sampler=types.SimpleNamespace(trace_replay_state=None,logprobs_mode='raw_logprobs'))
        made=[]
        def factory(states):made.append(states);return types.SimpleNamespace()
        self.assertEqual(td.configure_worker(r,True,factory=factory)['table_bytes'],1024*(td.STEPS+1)*4)
        td.configure_worker(r,True,factory=factory);self.assertEqual(len(made),1)
        td.configure_worker(r,False,factory=factory);self.assertIsNone(r.sampler.trace_replay_state)
        r.sampler.trace_replay_state=object()
        with self.assertRaises(td.TeacherDecodeError):td.configure_worker(r,True,factory=factory)
    def test_pp0_does_not_allocate_and_processed_logprobs_refused(self):
        def forbidden(states):raise AssertionError('allocation')
        r=types.SimpleNamespace(is_last_pp_rank=False,sampler=None)
        self.assertFalse(td.configure_worker(r,True,factory=forbidden)['sampler_owner'])
        r.is_last_pp_rank=True;r.sampler=types.SimpleNamespace(trace_replay_state=None,logprobs_mode='processed_logprobs')
        with self.assertRaises(td.TeacherDecodeError):td.configure_worker(r,True,factory=forbidden)
    def test_pinned_sampler_order_and_request_sources(self):
        ref=ROOT/'reference/vllm-src'
        for path,sha in probe_control_hooks.PINS.items():
            self.assertEqual(hashlib.sha256((ref/path).read_bytes()).hexdigest(),sha,path)
        s=(ref/'v1/worker/gpu/sample/sampler.py').read_text()
        self.assertLess(s.index('self.trace_replay_state.apply_trace(sampled, idx_mapping)'),
                        s.index('logprobs_tensors = compute_topk_scores('))
        s=(ref/'v1/worker/gpu/sample/trace_replay.py').read_text()
        self.assertIn('total_len - prompt_len',s)
        self.assertIn('trace_token_ids.stride(0)',s)



class ControllerCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_session_and_client_bind_plan_and_all_rank_ack(self):
        from test_e3_probe_session import CountedEngine
        from probe_control import ProbeController
        from probe_session import ProbeSession, receipt_core
        import e3_w1_decode_driver as driver
        class Engine(CountedEngine):
            async def collective_rpc(self,*a,**kw):
                c=kw['args'][0]
                if c['action']=='teacher_configure':
                    return [dict(rank=r,ack='flash_s5_teacher_configure',engine_epoch=c['engine_epoch'],
                        enabled=c['enabled'],sampler_owner=r>=2,table_bytes=1028 if r>=2 else 0,
                        drained_streams=3,snapshot_max_bytes=c['snapshot_max_bytes']) for r in range(4)]
                return await super().collective_rpc(*a,**kw)
        engine=Engine()
        engine.input_processor=types.SimpleNamespace(process_inputs=lambda r:r)
        session=ProbeSession(ProbeController(engine,epoch=71))
        context=dict(run_id='teacher-fixture',repeat=0,cohort_id='cohort-0',streams=8,
                     command_seq=0,prior_receipt=None,engine_epoch=None,storage_epoch=None,
                     route='full',route_generation=None)
        command=driver._next_command(context,'prepare','full')
        command['teacher_forced']=plan()
        reply=await session.execute(command)
        check=dict(context,action='prepare',command_seq=command['command_seq'])
        ok,reasons,_,accepted=driver.validate_receipt(reply,command,check)
        self.assertTrue(ok,reasons)
        core=driver._receipt_core(reply)
        self.assertEqual(core,receipt_core(reply))
        self.assertEqual(core['teacher_forced']['plan_sha256'],td.digest(plan()))
        self.assertEqual(len(core['teacher_forced']['ranks']),4)
        self.assertNotIn('continuation',str(core))
        self.assertFalse(engine._flash_s5_admission_seal.sealed)

    async def test_incomplete_worker_ack_does_not_enable_admission(self):
        from test_e3_probe_session import CountedEngine
        from probe_control import ProbeController,ProbeControlError
        class Engine(CountedEngine):
            async def collective_rpc(self,*a,**kw):
                if kw['args'][0]['action']=='teacher_configure':return []
                return await super().collective_rpc(*a,**kw)
        engine=Engine();ctl=ProbeController(engine,epoch=71)
        with self.assertRaises(ProbeControlError):await ctl.configure_teacher(plan())
        self.assertTrue(ctl.ingress.sealed)
        self.assertFalse(hasattr(ctl,'_teacher_plan'))


class GradeTests(unittest.TestCase):
    def evidence(self):
        import e3_teacher_decode_grade as g
        positions=list(range(1,257)); prefixes=['a'*64]*8
        rows=[dict(stream=s,position=p,worker_route='full',full_graph=True,decode_only=True,
                   forced_token_exact=True,route_generation=0,
                   top={f'token_id:{t}':-t*.2 for t in range(20)}) for s in range(8) for p in positions]
        runs={name:dict(route='split' if name=='split-B' else 'full',lifecycle_verdict='PASS',
              prefix_digests=prefixes,positions=copy.deepcopy(rows)) for name in g.RUNS}
        for row in runs['split-B']['positions']:row['worker_route']='split';row['route_generation']=1
        return dict(schema=td.SCHEMA,rule=g.FROZEN_RULE,baseline_sha256=g.BASELINE_SHA256,
                    fixed_positions=positions,prefix_digests=prefixes,runs=runs)
    def test_all_positions_stable_then_split_out_of_rule_and_control_noise(self):
        from e3_teacher_decode_grade import grade
        e=self.evidence();self.assertEqual(grade(e)['verdict'],'PASS')
        e['runs']['split-B']['positions'][0]['top']['token_id:0']=-1.
        self.assertEqual(grade(e)['verdict'],'FAIL')
        e['runs']['full-A2']['positions'][0]['top']['token_id:0']=-1.
        r=grade(e);self.assertEqual(r['verdict'],'INCONCLUSIVE')
        self.assertFalse(r['comparisons']['split']['comparable'])
    def test_missing_prefill_or_false_route_cannot_supply_decode_coverage(self):
        from e3_teacher_decode_grade import grade
        for key,value in [('decode_only',False),('worker_route','full'),('forced_token_exact',None)]:
            e=self.evidence();e['runs']['split-B']['positions'][0][key]=value
            self.assertEqual(grade(e)['verdict'],'INCONCLUSIVE')
        e=self.evidence();e['runs']['split-B']['positions'].pop()
        self.assertEqual(grade(e)['verdict'],'INCONCLUSIVE')
    def test_explicit_forced_token_mismatch_is_failure(self):
        from e3_teacher_decode_grade import grade
        e=self.evidence();e['runs']['split-B']['positions'][0]['forced_token_exact']=False
        self.assertEqual(grade(e)['verdict'],'FAIL')

    def test_rule_is_frozen_and_near_tie_flips_only(self):
        from e3_teacher_decode_grade import grade
        e=self.evidence();e['runs']['split-B']['positions'][0]['top']['token_id:1']=.01
        self.assertEqual(grade(e)['verdict'],'PASS')
        for name in e['runs']:
            e['runs'][name]['positions'][0]['top']['token_id:1']=-.4
        e['runs']['split-B']['positions'][0]['top']['token_id:0']=-.21
        e['runs']['split-B']['positions'][0]['top']['token_id:1']=-.19
        self.assertEqual(grade(e)['verdict'],'FAIL')
        e=self.evidence();e['rule']={**e['rule'],'TOL_WIN':1.}
        self.assertEqual(grade(e)['verdict'],'INCONCLUSIVE')


class ClientTests(unittest.TestCase):
    def test_real_instrument_cohort_survives_shared_result_assembly(self):
        from dataclasses import replace
        from unittest import mock
        import e3_decode_correctness as dc
        import e3_teacher_decode_client as client
        calls=[]
        def post(url,path,body):
            calls.append(path)
            return {'tokens':[len(calls)]*900}
        instrument=client.Instrument(ROOT/'harness/baselines/q15-teacher-forced-rule3.json',post=post)
        cfg=dc.config_from_env({'RUN_ID':'teacher-assembly-fixture','CONTROLLER_CMD':'false',
            'WORKLOAD':str(ROOT/'harness/workloads/q17b-teacher-decode-v3.json')},
            Path('unused-teacher-artifact.json'))
        cfg=replace(cfg,instrument=instrument)
        def measure(cfg,controller,ctx,index,label,route,cohort):
            return dc._not_run_record(index,label,route,'offline-no-engine')
        # Exercise the actual cohort, teacher finish, and shared artifact
        # assembly. Only the network/controller measurement is replaced.
        with mock.patch.object(dc,'measure_run',side_effect=measure):
            result=dc.run(cfg,'http://unused','fixture',object(),source={'verified':True})
        self.assertEqual(len(calls),16)
        self.assertEqual([r['stream'] for r in result['workload']['prompts']],list(range(8)))
        self.assertEqual(result['verdict'],'INCONCLUSIVE')
        self.assertIn('teacher_forced',result)
        self.assertTrue(all(r['not_run'] for r in result['runs']))
        bad=instrument.cohort('http://unused','fixture')
        bad[0].pop('stream')
        with mock.patch.object(instrument,'cohort',return_value=bad), \
             mock.patch.object(dc,'measure_run') as measure:
            with self.assertRaisesRegex(dc.ConfigError,'cohort artifact identities incomplete'):
                dc.run(cfg,'http://unused','fixture',object(),source={'verified':True})
        measure.assert_not_called()

    def instrument(self):
        import e3_teacher_decode_client as client
        i=client.Instrument.__new__(client.Instrument);i.snapshot_max_bytes=2*1024**3;i.plan_shas={}
        i.baseline={'sha256':__import__('e3_teacher_decode_grade').BASELINE_SHA256}
        i.prompts=[[n,11,12] for n in range(8)]
        i.traces=[[n+20]*client.TOTAL for n in range(8)]
        i.plan=None
        return i
    def test_api_original_logprobs_are_bound_to_fixed_token_positions(self):
        import e3_teacher_decode_client as client
        i=self.instrument();state=types.SimpleNamespace(index=0,token_count=client.ENTRY,teacher_top=[])
        top={f'token_id:{n}':-n/10 for n in range(20)}
        event={'choices':[{'token_ids':[20], 'logprobs':{'tokens':['token_id:20'],'top_logprobs':[top]}}]}
        i.consume(state,event)
        self.assertEqual(state.teacher_top,[{'position':client.ENTRY,'top':top}])
        event['choices'][0]['token_ids']=[21]
        with self.assertRaises(ValueError):i.consume(state,event)
    def test_requests_bind_private_plan_without_changing_sampling(self):
        import e3_teacher_decode_client as client
        import e3_decode_correctness as dc
        i=self.instrument()
        workload=types.SimpleNamespace(generation_budget=client.TOTAL,temperature=0.,top_p=1.,sampling_seed=909)
        bodies=[dc.request_body('model',p,workload,'salt') for p in i.prompts]
        before=[{k:v for k,v in b.items()} for b in bodies]
        i.requests(bodies,[{'prompt':p} for p in i.prompts],3)
        self.assertEqual(len(td.validate_plan(i.plan)),8)
        for a,b in zip(before,bodies):
            self.assertEqual(a,{k:v for k,v in b.items() if k not in ('request_id','logprobs','return_tokens_as_token_ids')})
    def test_fixed_worker_proof_join_then_missing_position_cannot_pass(self):
        import e3_teacher_decode_client as client
        i=self.instrument();runs=[]
        top={f'token_id:{n}':-n/10 for n in range(20)}
        for repeat,name in enumerate(('full-A1','full-A2','split-B','full-C')):
            route='split' if name=='split-B' else 'full';gen=1 if route=='split' else 0
            rows=[];streams=[]
            for stream,trace in enumerate(i.traces):
                rows.extend(dict(trace_sha256=client.digest(trace),position=p,worker_route=route,
                    route_generation=gen,full_graph=True,decode_only=True,forced_token_exact=True)
                    for p in range(client.ENTRY,client.ENTRY+client.MEASURE))
                streams.append(dict(stream=stream,_tokens=trace,
                    teacher_top=[dict(position=p,top=top) for p in range(client.ENTRY,client.ENTRY+client.MEASURE)]))
            workers=[dict(rank=r,sampler_owner=r>=2,engine_epoch=71,route_generation=gen,
                          overflow=False,route=route,drained_streams=3,positions=rows if r==2 else []) for r in range(4)]
            i.plan_shas[repeat]='a'*64
            proof=dict(pending=0,admitted=8,plan_sha256='a'*64,ranks=workers)
            core=dict(repeat=repeat,engine_epoch=71,route_generation_after=gen,teacher_forced=proof)
            runs.append(dict(label=name,route=route,complete=True,contract=False,streams=streams,
                calls=[dict(action='pause_end',lifecycle_ok=True,receipt={'teacher_forced':proof},
                            receipt_core=core,receipt_sha256=client.digest(core))]))
        self.assertEqual(i.finish(runs)['grade']['verdict'],'PASS')
        proof=runs[2]['calls'][0]['receipt']['teacher_forced']
        proof['ranks'][2]['positions'].pop()
        runs[2]['calls'][0]['receipt_sha256']=client.digest(runs[2]['calls'][0]['receipt_core'])
        self.assertEqual(i.finish(runs)['grade']['verdict'],'INCONCLUSIVE')

if __name__=='__main__':unittest.main()
