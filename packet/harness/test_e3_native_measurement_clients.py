"""Actual native client actions, source batch counters and window grading; CPU."""
import asyncio
import copy
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'harness'),str(ROOT/'patches/s5_engine')]
import native_overlap_worker as nw
import native_overlap_control as nc
import native_overlap_session as ns
import native_overlap_metrics as metrics
import graph_hooks
import e3_native_w1 as w1
import e3_native_performance as perf
import e3_native_startup_grade as startup
from e3_native_runtime_client import NativeRuntimeClient
import test_e3_native_overlap as native
import test_e3_native_measurement as source


def batch(count=4,prefill=False):
    return types.SimpleNamespace(has_prefill=prefill,num_scheduled_tokens=[1]*count,
                                 is_prefilling_np=[prefill]*count)


def census_row(rank=0,route='split',steps=5):
    census=nw.NativeCensus()
    for _ in range(steps):census.observe_dispatch(4,4,1,batch(),True)
    return {'worker_rank':rank,'engine_epoch':71,'engine_epoch_ok':True,
            'storage_epoch':1,'worker_incarnation':'worker-%d'%rank,'route':route,
            'route_generation':4,'prefill_requests':0,'decode_eligible':8,'live_decode_requests':8,
            'barrier_clean':True,'drained_streams':2,'pending_native_replays':0,
            'retained_rows':0,'pending_prefill_streams':0,'pending_output_copies':0,
            'live_decode':{'available':True,'census':census.record()}}


class DispatchCensus(unittest.TestCase):
    def test_actual_metadata_partial_batch_is_decode_and_counters_never_reset(self):
        c=nw.NativeCensus()
        c.observe_dispatch(4,4,1,batch(),True)
        c.observe_dispatch(2,2,1,batch(2),True)
        self.assertEqual(c.record()['decode_dispatches'],2)
        self.assertEqual(c.record()['non_decode_dispatches'],0)
        c.observe_dispatch(4,4,1,batch(prefill=True),True)
        self.assertEqual(c.record()['non_decode_dispatches'],1)
        self.assertEqual(c.record()['dispatches'],3)

    def test_missing_metadata_is_not_proven_decode(self):
        for state in (None,types.SimpleNamespace(has_prefill=False),batch(prefill=True)):
            c=nw.NativeCensus();c.observe_dispatch(4,4,1,state,True)
            self.assertEqual(c.record()['decode_dispatches'],0)
            self.assertEqual(c.record()['non_decode_dispatches'],1)

    def test_numpy_pinned_ints_and_bools_are_real_batch_metadata(self):
        import numpy as np
        c=nw.NativeCensus()
        c.observe_dispatch(4,4,1,types.SimpleNamespace(has_prefill=False,
            num_scheduled_tokens=np.ones(4,dtype=np.int32),is_prefilling_np=np.zeros(4,dtype=np.bool_)),True)
        self.assertEqual(c.record()['decode_dispatches'],1)

    def test_native_graph_route_seam_observes_the_dispatched_batch(self):
        c=nw.NativeCensus();runner=types.SimpleNamespace(_flash_s5_native_census=c)
        descriptor=types.SimpleNamespace(cg_mode='full')
        router=types.SimpleNamespace(native_decode_route=lambda value:'actual-route')
        with patch.object(graph_hooks,'_router',return_value=router),patch.object(graph_hooks,'_native_decode',return_value=True),\
             patch.object(graph_hooks,'_transaction',return_value=None),patch.object(graph_hooks,'_descriptor',return_value=descriptor):
            self.assertEqual(graph_hooks.check_route(runner,object(),object(),4,4,1,batch()),'actual-route')
        self.assertEqual(c.record()['decode_dispatches'],1)


class W1NativeIntervals(unittest.TestCase):
    def setUp(self):
        self.client=NativeRuntimeClient.__new__(NativeRuntimeClient);self.client.epoch=71
        self.before={'barrier':{'workers':[census_row(r) for r in range(4)]}}
        self.after={'barrier':{'workers':[census_row(r,steps=10) for r in range(4)]}}
    def grade(self):return w1.interval_proof(self.before,self.after,self.client,'split')
    def test_all_rank_partial_batches_and_actual_full_graph_dispatches(self):
        self.assertTrue(self.grade()['complete'])
    def test_one_mixed_batch_refuses_instead_of_reading_last_only(self):
        self.after['barrier']['workers'][1]['live_decode']['census']['non_decode_dispatches']=1
        self.assertFalse(self.grade()['complete'])
    def test_no_graph_dispatches_refuses(self):
        self.after['barrier']['workers'][2]['live_decode']['census']['full_graph_dispatches']=5
        self.assertFalse(self.grade()['complete'])
    def test_counter_reset_refuses(self):
        self.after['barrier']['workers'][0]['live_decode']['census']['dispatches']=0
        self.assertFalse(self.grade()['complete'])
    def test_route_generation_or_worker_change_refuses(self):
        for key,value in [('route_generation',5),('worker_incarnation','foreign'),('route','full')]:
            with self.subTest(key=key):
                saved=copy.deepcopy(self.after)
                self.after['barrier']['workers'][0][key]=value
                self.assertFalse(self.grade()['complete']);self.after=saved
    def test_live_count_and_prefill_are_not_inferred_from_token_count(self):
        self.after['barrier']['workers'][0]['prefill_requests']=1
        self.assertFalse(self.grade()['complete'])
    def test_unqueried_drain_or_missing_rank_refuses(self):
        self.after['barrier']['workers'].pop()
        self.assertFalse(self.grade()['complete'])


class NativeActions(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stack=native.idle_stack()
        self.core,self.frontend=native.core_for(self.stack)
        self.pump=native.Pump(self.core).start()
        self.control=nc.NativeControlClient(self.frontend,timeout_s=5)
        self.session=ns.NativeOverlapSession(self.frontend,control=self.control)
    async def asyncTearDown(self):
        await self.pump.stop()
        if self.pump.failure:raise self.pump.failure
    async def call(self,action,**fields):
        return await self.session.execute(dict(schema=1,action=action,run_id='native-client-test',**fields))
    async def test_pause_is_real_awaited_keep_and_actual_four_bridge_barrier(self):
        await self.call('prepare',target_route='split')
        result=await self.call('pause')
        self.assertEqual(result['pause']['mode'],'keep')
        self.assertFalse(result['pause']['clear_cache'])
        self.assertEqual(len(result['barrier']['workers']),4)
        self.assertTrue(all(r['drained_streams']>0 and r['barrier_clean'] for r in result['barrier']['workers']))
    async def test_actual_session_full_return_carries_its_bound_engine_epoch(self):
        prepared=await self.call('prepare',target_route='split')
        returned=await self.call('leave_full')
        self.assertEqual(returned['engine_epoch'],prepared['engine_epoch'])
        self.assertEqual(len(returned['transition_acks']),4)
        self.assertTrue(all(a['engine_epoch']==returned['engine_epoch'] for a in returned['transition_acks']))

    async def test_identity_uses_existing_ordered_worker_reply_path(self):
        await self.call('prepare',target_route='split')
        result=await self.call('identity')
        self.assertEqual({r['worker_rank'] for r in result['workers']},set(range(4)))
        self.assertEqual({r['action'] for r in result['workers']},{'identity'})
        self.assertTrue(all('native_inventory' in r for r in result['workers']))
    async def test_native_prefill_begin_needs_real_admission(self):
        await self.call('prepare',target_route='split')
        with self.assertRaisesRegex(ns.NativeOverlapSessionError,'admission-not-armed'):
            await self.call('prefill_begin',phase='split-prefill-solo')
    async def test_native_prefill_window_is_separate_and_no_decode_progress_invented(self):
        await self.call('prepare',target_route='split',phase='split-prefill-solo')
        begin=await self.call('prefill_begin',phase='split-prefill-solo')
        self.assertTrue(begin['clock_proof']['proved'])
        with self.assertRaisesRegex(ns.NativeOverlapSessionError,'already-open'):
            await self.call('prefill_begin',phase='split-prefill-solo')
        await self.control.census()  # actual queried idle census, no native work
        end=await self.call('prefill_end')
        self.assertFalse(end['complete'])
        self.assertEqual(end['measurement_binding']['reasons'],['measurement-route-generation-differs'])
        self.assertEqual(end['measurement']['route_generations'],[])
        self.assertNotIn('decode_progress',end)
        with self.assertRaisesRegex(ns.NativeOverlapSessionError,'without-begin'):
            await self.call('prefill_end')


class NativeWindowSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stack=source.make_long_stack(tokens=2050,layers=4)
        entries=source.run_journal(stack)
        start,stop=source.window_of(entries)
        raw=stack.driver.overlap_evidence(start,stop)
        measurement=raw['measurement']
        generation=measurement['route_generations'][0]
        begin={'schema':1,'engine_epoch':stack.driver.engine_epoch,'route':'split',
               'route_generation':generation,'ranks':list(stack.driver.ranks),
               'storage_epochs_by_rank':{r:stack.worker(r).state.decode_owner.storage_epoch for r in stack.driver.ranks}}
        # The fixture's final typed FULL return is the actual driver's proof.
        raw['start_ns'],raw['end_ns']=start,stop
        identity=metrics.clock_identity()
        proof=dict(identity,proved=True)
        cls.record={'begin':{'engine_epoch':stack.driver.engine_epoch,'snapshot':begin,'measure_start_ns':start},
            'end':{'engine_epoch':stack.driver.engine_epoch,'measure_end_ns':stop,'raw_journal_evidence':raw,
                   'clock':{'begin':proof,'end':proof}},'client_bracket':{'start_ns':start-1,'end_ns':stop+1}}
        cls.epoch=stack.driver.engine_epoch
    def test_actual_driver_source_packet_regrades_without_cached_flags(self):
        result=perf.native_window(self.record,'fresh',self.epoch)
        self.assertTrue(result['complete'],result['reasons'])
        self.assertGreater(result['prefill_tokens_per_s'],0)
    def test_clock_or_family_mismatch_stays_inconclusive(self):
        self.assertFalse(perf.native_window(self.record,'append',self.epoch)['complete'])
        record=copy.deepcopy(self.record);record['end']['clock']['end']['time_namespace']='foreign'
        self.assertFalse(perf.native_window(record,'fresh',self.epoch)['complete'])
    def test_changed_window_or_source_proof_is_refused(self):
        record=copy.deepcopy(self.record);record['begin']['measure_start_ns']+=1
        self.assertFalse(perf.native_window(record,'fresh',self.epoch)['complete'])
        record=copy.deepcopy(self.record);record['end']['raw_journal_evidence']['measurement']['completions'][0]['queried']=False
        self.assertFalse(perf.native_window(record,'fresh',self.epoch)['complete'])
    def test_raw_loss_is_visible_and_independent_census_still_required(self):
        record=copy.deepcopy(self.record)
        record['end']['raw_journal_evidence']['journal']['complete_in_window']=False
        record['end']['raw_journal_evidence']['journal']['dropped']=1000
        result=perf.native_window(record,'fresh',self.epoch)
        self.assertTrue(result['complete'],result['reasons'])
        self.assertTrue(result['raw_journal_retained'])
        self.assertEqual(record['end']['raw_journal_evidence']['journal']['dropped'],1000)
    def test_performance_never_uses_incomplete_survivors(self):
        result=perf.grade({'engine_epoch':self.epoch,'lanes':{'prefill_split':{'fresh':{'complete':True}}}})
        self.assertEqual(result['verdict'],'INCONCLUSIVE')
        self.assertIsNone(result['target']['decode_share'])



class PerformanceMetricContract(unittest.TestCase):
    def record(self):
        window={'start_ns':10,'end_ns':20,'aggregate_tokens_per_s':40,
                'stream_rates_tokens_per_s':[5.0]*8}
        lanes={lane:{family:{'complete':True,'request':{'ttft_s':2.0},
                            'decode_window':copy.deepcopy(window),
                            'serving_prefill_tokens_per_s':200}
                     for family in ('fresh','append')} for lane in perf.shared.LANES}
        w1={'summary':{'verdict':'COMPLETE','routes':{'full':{'aggregate_tokens_per_s_median':80}}},
            'repeats':[{'valid':True,'route':'full','metrics':{'stream_rates_tokens_per_s':[10.0]*8}}]}
        return {'engine_epoch':71,'lanes':lanes,'w1':w1}
    def proof(self,row,family,epoch):
        # Metric policy fixture only; actual source-packet regrades are covered
        # independently above and are never replaced in production.
        return {'complete':True,'reasons':[],
                'prefill_tokens_per_s':50 if row['decode_window']['aggregate_tokens_per_s']==40 else 100,
                'group':{'start_ns':10,'end_ns':20}}
    def test_80_80_shortfall_is_reported_without_changing_hard_gates(self):
        record=self.record()
        record['lanes']['prefill_split']['fresh']['decode_window']['aggregate_tokens_per_s']=80
        with patch.object(perf,'native_window',side_effect=self.proof):result=perf.grade(record)
        self.assertEqual(result['verdict'],'PASS',result['missing'])
        self.assertEqual(result['target']['status'],'SHORTFALL')
        self.assertFalse(result['target_met']);self.assertFalse(result['target_is_gate'])
        self.assertEqual(result['decode_shares_by_stream'],[.5]*8)
        self.assertEqual(result['target']['prefill_share'],.5)
    def test_incomplete_append_and_nan_stream_cannot_be_selected_away(self):
        for mutate in (lambda r:r['lanes']['overlap']['append'].update(complete=False),
                       lambda r:r['lanes']['overlap']['fresh']['decode_window']['stream_rates_tokens_per_s'].__setitem__(0,float('nan')),
                       lambda r:r.update(first_error={'error_type':'RuntimeError'})):
            record=self.record();mutate(record)
            with patch.object(perf,'native_window',side_effect=self.proof):result=perf.grade(record)
            self.assertEqual(result['verdict'],'INCONCLUSIVE')
            self.assertIsNone(result['target_met'])
            self.assertFalse(result['filtered_invalid'])


class NativeMemoryWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        NativeWindowSource.setUpClass()
    def fixture(self):
        import e3_native_memory_grade as memory
        row=copy.deepcopy(NativeWindowSource.record)
        proof=perf.native_window(row,'fresh',NativeWindowSource.epoch)
        start,stop=proof['group']['start_ns'],proof['group']['end_ns']
        row['drain']={'barrier':{'workers':[dict(census_row(r),engine_epoch=NativeWindowSource.epoch,
            allocator={'basis':'torch-cuda-allocator-queried-v1',
                       'allocated_peak_bytes':1024,'reserved_peak_bytes':2048}) for r in range(4)]}}
        duration=(stop-start)/1e9
        performance={'engine_epoch':NativeWindowSource.epoch,'wall_anchor_s':1000.,
            'wall_anchor_ns':start,'wall_anchor_uncertainty_ns':0,
            'wall_end_s':1001.+duration,'wall_end_ns':stop+10**9,
            'lanes':{'overlap':{'fresh':row}}}
        sample={'start_s':1000.+duration*.5,'end_s':1000.+duration*.5,
                'gpus':{str(r):{'total_mib':24576,'used_mib':23000} for r in range(4)}}
        return memory,performance,[sample]
    def test_actual_packet_and_physical_free_are_separate_from_allocator(self):
        memory,record,samples=self.fixture();result=memory.grade(record,samples)
        self.assertEqual(result['verdict'],'PASS',result['missing'])
        self.assertEqual(result['min_free_mib_under_pressure'],{r:1576 for r in range(4)})
        self.assertEqual(result['reserve_mib'],512)
        samples[0]['gpus']['2']['used_mib']=24100
        result=memory.grade(record,samples)
        self.assertEqual(result['verdict'],'FAIL')
        self.assertTrue(any('gpu2' in reason for reason in result['failures']))
    def test_missing_clock_or_rank_and_crossing_query_bracket_are_unproved(self):
        for mutate in (
            lambda r,s:r.update(wall_end_s=10000.),
            lambda r,s:r['lanes']['overlap']['fresh']['drain']['barrier']['workers'].pop(),
            lambda r,s:s[0].update(end_s=100000.),
            lambda r,s:s[0]['gpus'].pop('3'),
            lambda r,s:s[0].update(error='ClockStep')):
            memory,record,samples=self.fixture();mutate(record,samples)
            self.assertEqual(memory.grade(record,samples)['verdict'],'INCONCLUSIVE')


class ActualW1Reader(unittest.TestCase):
    def test_original_requests_and_fixed_repeats_over_live_sse_native_boundaries(self):
        import tempfile
        import test_e3_w1_decode_driver as fixture
        import e3_w1_decode_driver as old
        data=copy.deepcopy(fixture.BASE_MANIFEST)
        data.update(streams=8,open_timeout_s=2.0)
        data['request_shape']['temperature']=.7
        workload=old.validate_workload(data)[1]
        server=fixture.FakeServer(fixture.FakeConfig(streams=8))
        self.addCleanup(server.stop)
        control=NativeRuntimeClient.__new__(NativeRuntimeClient);control.epoch=71
        class Client:
            epoch=71
            control=None
            calls=[]
            route='full'
            pauses=0
            def prepare(self,route,phase):
                self.calls.append(('prepare',route,phase));self.route=route;return {'route':route}
            def pause(self):
                self.pauses+=1;self.calls.append(('pause',self.route))
                return {'barrier':{'workers':[census_row(r,route=self.route,steps=self.pauses*10) for r in range(4)]}}
            def workers(self,reply):return control.workers(reply)
            def enter(self,phase):self.calls.append(('enter',phase));self.route='split';return {}
            def resume(self):self.calls.append(('resume',));return {}
            def leave(self):self.calls.append(('leave',));self.route='full';return {}
        c=Client()
        cfg=old.Config(workload,'native-sse','',2,2,1,'unused','unused')
        result=w1.run(cfg,server.url,'model',c)
        self.assertEqual(result['summary']['verdict'],'COMPLETE',result['repeats'])
        self.assertEqual(len(server.measure_requests()),16)
        self.assertEqual([r['route'] for r in result['repeats']],['full','split'])
        self.assertTrue(all(r['interval']['complete'] for r in result['repeats']))
        self.assertFalse(any(call[0] in ('pause_full','pause_end') for call in c.calls))
        self.assertEqual(result['config']['WORKLOAD_SHA256'],workload.manifest_sha256)


class QueriedStartup(unittest.TestCase):
    def fixture(self):
        from native_startup import WorkerNative
        from lane_state import WorkerState
        from green_context import GreenSplit,MembershipResult
        from test_e3_green_native import FakeBinding,fake_torch
        rows=[]
        epoch=71
        acks=[]
        for rank in range(4):
            state=WorkerState(rank,engine_epoch=epoch)
            owner=state.retain_decode_storage(context_id=100+rank)
            split=GreenSplit(FakeBinding(),rank,44,torch_module=fake_torch(82)).create()
            split.membership_result=MembershipResult(range(44),range(44,82),82,True,3)
            n=WorkerNative.__new__(WorkerNative)
            n.split=split;n.state=state;n.decode_owner=owner
            n.controller=types.SimpleNamespace(route='full',route_generation=2)
            n.router=types.SimpleNamespace(native_census=lambda:{'engine_epoch':71,'storage_epoch':1,'pending':0})
            n.bank=types.SimpleNamespace(measured_bytes=1024)
            n.comms=types.SimpleNamespace(pp=types.SimpleNamespace(bytes_owned=lambda:1024),
                describe=lambda:{'tp_world_size':2,'tp_warmed':True})
            n.ple_scratch=types.SimpleNamespace(bytes=512);n.ple_layer=rank*12;n.chunk_tokens=1024
            n.prefill_streams=types.SimpleNamespace(record=lambda:{'state':'ready','basis':'prefill-green','original_owners_restored':True})
            n.private_bindings=types.SimpleNamespace(audit=lambda:{'isolation':'partial','readiness':'unclaimed',
                'workspace_prepared':True,'failed':False,'workspace_basis':'extra_last_lane','workspace_bytes':128,
                'uid_rows':10,'qsa_layers':[{'layer':rank*12}]})
            row=census_row(rank,route='full')
            row.update(worker_incarnation=state.incarnation,route_generation=2,native_inventory=nw._native_inventory(n),
                allocator={'basis':'torch-cuda-allocator-queried-v1'},
                memory={'basis':'torch-cuda-mem-get-info-v1','free_bytes':1024**3,'total_bytes':24*1024**3})
            rows.append(row)
            acks.append({'typed':'TransitionAck','worker_rank':rank,'engine_epoch':epoch,
                'applied_bank':0,'drained_streams':2,'route_generation':2,'applied_route_generation':2})
        from wire_v1 import ROUTE_BANK_FULL
        for ack in acks:ack['applied_bank']=ROUTE_BANK_FULL
        return {'typed_acks':True,'route':'full','transition_acks':acks},{'workers':rows}
    def test_actual_worker_inventory_shape_and_owner_incarnation(self):
        prepare,identity=self.fixture()
        result=startup.grade(prepare,identity,epoch=71)
        self.assertEqual(result['verdict'],'PASS',result)
        self.assertEqual(result['workers'][0]['native_inventory']['private_bindings']['isolation'],'partial')
    def test_source_owner_membership_capture_workspace_and_tp_evidence_required(self):
        prepare,identity=self.fixture()
        for section,key,value in [('split','basis','driver-reported'),('communicators','tp_warmed',False),
                                  ('private_bindings','workspace_prepared',False),('router','pending',1)]:
            bad=copy.deepcopy(identity);bad['workers'][0]['native_inventory'][section][key]=value
            self.assertNotEqual(startup.grade(prepare,bad,epoch=71)['verdict'],'PASS')
    def test_wrong_epoch_and_missing_rank_cannot_pass(self):
        prepare,identity=self.fixture()
        identity['workers'].pop()
        self.assertNotEqual(startup.grade(prepare,identity,epoch=71)['verdict'],'PASS')



class ActualCertifiedClosure(unittest.TestCase):
    def close(self):
        import e3_q18a_lifecycle as lifetime
        stack=native.active_stack()
        native.core_for(stack)
        stack.drive_to_finalize()
        before=stack.driver._status_record()
        stack.complete_everything()
        stack.driver.stop('abort')
        stack.drive(lambda:not stack.driver.active)
        after=stack.driver._status_record()
        return stack,before,after,lifetime.grade_close(before,after,stack.driver.engine_epoch)
    def test_real_driver_four_bridge_close_and_same_queried_reports(self):
        stack,before,after,grade=self.close()
        self.assertEqual(grade['verdict'],'PASS',grade)
        self.assertEqual(stack.controller.commits,0)
        self.assertEqual(stack.controller.cancels,1)
        self.assertFalse(stack.controller.leases.held_blocks())
        self.assertGreater(grade['completed_operation_count'],0)
        self.assertEqual({r['typed'] for r in after['last_native_closure']['rows']},{'CancelAck'})
    def test_reset_does_not_remove_the_certified_record_and_status_does_not_alias_it(self):
        stack,before,after,grade=self.close()
        self.assertIsNone(stack.driver.tx)
        after['last_native_closure']['rows'].clear()
        self.assertEqual(len(stack.driver._status_record()['last_native_closure']['rows']),4)
    def test_same_uid_generation_and_completed_operation_sets_are_required(self):
        import e3_q18a_lifecycle as lifetime
        stack,before,after,_=self.close()
        for mutate in (
            lambda row:row['last_native_closure'].update(request_generation=123),
            lambda row:row['last_native_closure'].update(request_uid_sha256='f'*64),
            lambda row:row['last_native_closure']['rows'][0]['report'].update(pending_op_seqs=[99]),
            lambda row:row['last_native_closure']['rows'][0]['report']['proofs'].clear(),
            lambda row:row['last_native_closure'].update(reason='pause'),
        ):
            bad=copy.deepcopy(after);mutate(bad)
            self.assertNotEqual(lifetime.grade_close(before,bad,stack.driver.engine_epoch)['verdict'],'PASS')


class FrozenRow1Plan(unittest.TestCase):
    def test_exact_prefix_tail_is_new_oracle_only_and_frozen_before_request(self):
        import e3_native_request_plans as plans
        from e3_native_qualification_driver import freeze_row1_plan
        template=json.loads((ROOT/'harness/workloads/q18a-native-row1-template.json').read_text())
        calls=[]
        def post(url,path,body):
            calls.append((path,body['model']))
            return {'tokens':[71]*(plans.EXACT_ROW1_TOKENS+3)}
        plan,receipt=plans.build(template,'http://127.0.0.1:8000','model','run',post=post)
        self.assertEqual(calls,[('/tokenize','model')])
        self.assertEqual(receipt['native_bulk_tail_tokens'],7)
        frozen=freeze_row1_plan(plan,run_id='run',model='model')
        self.assertEqual(len(frozen['bodies'][0]['prompt']),98312)
        self.assertEqual(frozen['bodies'][0]['prompt'],[71]*98312)
        self.assertEqual(receipt['request_sha256'],frozen['request_sha256'])
    def test_one_tokenization_refuses_insufficient_data_without_retry(self):
        import e3_native_request_plans as plans
        template=json.loads((ROOT/'harness/workloads/q18a-native-row1-template.json').read_text())
        calls=[]
        def post(*args):calls.append(1);return {'tokens':[7]}
        with self.assertRaisesRegex(ValueError,'single-tokenization-insufficient'):
            plans.build(template,'http://127.0.0.1:8000','model','run',post=post)
        self.assertEqual(calls,[1])
    def test_nominal_seeded_text_is_not_mislabeled_as_exact_token_count(self):
        from e3_native_qualification_driver import NativePlacementRunner
        import e3_decode_correctness as correctness
        state=correctness.StreamRecord(0,'f'*64,33,'b'*64)
        state.prompt_tokens=4112;state.prompt_token_ids_len=4112
        row=NativePlacementRunner._prefill_progress([state],types.SimpleNamespace(prompt_tokens=4096))[0]
        self.assertTrue(row['prompt_tokens_proven'])
        self.assertFalse(row['prompt_tokens_match_declared'])
        state.prompt_token_ids_len=4100
        row=NativePlacementRunner._prefill_progress([state],types.SimpleNamespace(prompt_tokens=4096))[0]
        self.assertFalse(row['prompt_tokens_proven'])



class ExactAuthentication(unittest.TestCase):
    def test_authentication_only_changes_the_declared_loopback_header(self):
        import e3_auth_exec as auth
        import http.client
        seen=[]
        def original(connection,method,url,body=None,headers=None,**kwargs):
            seen.append((connection.host,method,url,body,headers))
        with patch.object(http.client.HTTPConnection,'request',original):
            with auth.authentication('http://127.0.0.1:8000','fixture-only-credential'):
                c=http.client.HTTPConnection('127.0.0.1',8000)
                c.request('POST','/v1/completions',b'exact-body',{'Authorization':'Bearer old','Other':'retained'})
                c=http.client.HTTPConnection('127.0.0.1',8001)
                c.request('POST','/v1/completions',b'exact-body',{'Authorization':'Bearer old'})
            self.assertIs(http.client.HTTPConnection.request,original)
        self.assertEqual(seen[0][3],b'exact-body')
        self.assertEqual(seen[0][4],{'Other':'retained','Authorization':'Bearer fixture-only-credential'})
        self.assertEqual(seen[1][4],{'Authorization':'Bearer old'})
    def test_candidate_config_is_private_new_and_does_not_mutate_shared_settings(self):
        import canary_hub_candidate_runner as canary
        import e3_private_auth as auth
        import tempfile,os
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);os.chmod(root,0o700)
            auth.prepare(root/'client.env',{'VLLM_API_KEY':'fixture-only-credential'})
            original={'env':{'ANTHROPIC_BASE_URL':'http://<gateway-host>:8000','ANTHROPIC_AUTH_TOKEN':'legacy'},'permissions':{'allow':['Read']}}
            saved=copy.deepcopy(original)
            path=canary.prepare_client_config(original,str(root/'client.env'),root)
            self.assertEqual(original,saved)
            self.assertEqual(path.stat().st_mode&0o777,0o700)
            settings=path/'settings.json'
            self.assertEqual(settings.stat().st_mode&0o777,0o600)
            self.assertEqual(json.loads(settings.read_text())['env']['ANTHROPIC_AUTH_TOKEN'],'fixture-only-credential')
            with self.assertRaises(FileExistsError):canary.prepare_client_config(original,str(root/'client.env'),root)
    def test_candidate_canary_shape_and_script_are_unchanged_except_private_config(self):
        original=(ROOT/'harness/flash_canary.sh').read_text()
        candidate=(ROOT/'harness/flash_canary_candidate.sh').read_text()
        candidate=candidate.replace('\n[[ ${CANARY_CONFIG_DIR:-} == /* ]] || exit 2','')
        candidate=candidate.replace('CLAUDE_CONFIG_DIR="$CANARY_CONFIG_DIR"','CLAUDE_CONFIG_DIR=/home/<service-user>/.claude-local')
        candidate=candidate.replace('"$OUT" "$CANARY_CONFIG_DIR/projects"','"$OUT" /home/<service-user>/.claude-local/projects')
        self.assertEqual(candidate,original)
        runner=(ROOT/'harness/canary_hub_candidate_runner.py').read_text()
        self.assertIn("Relay('http://<gateway-host>:8000'",runner)
        self.assertIn("'flash_canary_candidate.sh'",runner)
        self.assertIn("env.pop('ANTHROPIC_AUTH_TOKEN',None)",runner)
        self.assertIn("(r['N'],r['ROUNDS'])!=(10,2)",runner)

    def test_candidate_canary_path_matches_owner_and_queue_contract(self):
        import canary_hub_candidate_runner as canary
        path='/home/<service-user>/.flashnext/canary-client-auth.env'
        self.assertEqual(canary.client_auth_path({'CANARY_CLIENT_AUTH_FILE':path}),path)
        for supplied in ({},{'CANARY_CLIENT_AUTH_FILE':path+'.other'},
                         {'CANARY_CLIENT_AUTH_FILE':'relative.env'}):
            with self.assertRaisesRegex(ValueError,'pinned owner contract'):
                canary.client_auth_path(supplied)
        settings=(ROOT/'harness/runs/q18a_overlap_settings.sh').read_text()
        declaration=(ROOT/'harness/runs/q18a_overlap_declare.sh').read_text()
        self.assertIn('CANARY_CLIENT_AUTH_FILE='+path,settings)
        self.assertIn("canary_client_auth_file='"+path+"'",declaration)

    def test_fresh_ct_bridge_process_reads_owned_file_without_secret_argv(self):
        import canary_hub_candidate_runner as canary
        import e3_auth_exec as auth
        import e3_private_auth as private
        import tempfile,http.client,sys
        seen=[]
        original_argv=sys.argv
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);root.chmod(0o700)
            private.prepare(root/'api.env',{'VLLM_API_KEY':'fixture-file-only-credential'})
            def execute(path,run_name):
                self.assertEqual(Path(path).name,'pp_canary_bridge.py')
                self.assertNotIn('fixture-file-only-credential',repr(sys.argv))
                http.client.HTTPConnection('127.0.0.1',8000).request('GET','/metrics')
            def request(connection,method,url,body=None,headers=None,**kwargs):
                seen.append(headers)
            with patch.object(auth.runpy,'run_path',execute),patch.object(http.client.HTTPConnection,'request',request):
                self.assertEqual(auth.main(['--env-file',str(root/'api.env'),
                    'http://127.0.0.1:8000',str(ROOT/'harness/pp_canary_bridge.py'),'idle','/owner/run']),0)
        self.assertIs(sys.argv,original_argv)
        self.assertEqual(seen,[{'Authorization':'Bearer fixture-file-only-credential'}])
        result=types.SimpleNamespace(returncode=0,stdout=b'{}')
        with patch.object(canary.subprocess,'run',return_value=result) as run:
            canary.remote('/home/<service-user>/harness/id','/home/<service-user>/s0/q18a-native1','idle')
        command=run.call_args.args[0][-1]
        self.assertIn('e3_auth_exec.py --env-file /home/<service-user>/s0/q18a-native1-requests/private/vllm-api.env',command)
        self.assertNotIn('VLLM_API_KEY=',command)

    def test_declared_snapshot_cap_reaches_the_actual_teacher_instrument(self):
        import tempfile
        import e3_native_qualification_driver as driver
        with tempfile.TemporaryDirectory() as tmp:
            value=driver.QualificationDriver(tmp,run_id='cpu-cap',model='fixture',engine_epoch=71,
                control=types.SimpleNamespace(),teacher_baseline='fixture-baseline',snapshot_max_bytes=64*1024**2)
            with patch.object(driver.teacher_client,'Instrument') as instrument:
                value._instrument()
            self.assertEqual(instrument.call_args.kwargs['snapshot_max_bytes'],64*1024**2)
            for cap in (True,1,2*1024**3+1):
                with self.assertRaises(driver.QualificationError):
                    driver.QualificationDriver(tmp,run_id='cpu-cap',model='fixture',engine_epoch=71,
                        control=types.SimpleNamespace(),snapshot_max_bytes=cap)


if __name__=='__main__':unittest.main()
