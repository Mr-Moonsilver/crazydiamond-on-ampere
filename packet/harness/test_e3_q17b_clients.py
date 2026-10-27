"""Full q17b client composition and placement gates, CPU only."""
import copy
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest import mock

import e3_q17b_clients as q
import e3_w1_decode_driver as w1
import e3_decode_correctness as c
from test_e3_decode_correctness import ComposedActualSessionTests

HERE = Path(__file__).resolve().parent


def audit_data():
    rows = []
    begins = []
    for rank in range(4):
        row = dict(rank=rank, ack='flash_s5_observer_end', state='closed',
                   observer_stopped=True, drained_streams=3,
                   engine_epoch=71, storage_epoch=1, route_generation=1,
                   worker_incarnation='worker-%d' % rank)
        row['receipt'] = dict(schema='s5ro-replay-observer/1', verdict='PASS', complete=True,
            static_structure={'verdict':'PASS','complete':True}, replay={'verdict':'PASS','complete':True},
            caller_proof=dict(worker_rank=rank, worker_incarnation=row['worker_incarnation'],
                 storage_epoch=1, route_generation=1, engine_incarnation='engine:71', relevant_consumers=3, controller_drained=True,
                 route_generation_matches=True, storage_epoch_matches=True))
        row['receipt']['replay']['graph_slots'] = [dict(label='graph:0',
            required=dict(kernel=2,memcpy=0,memset=0),
            kernel_executions=4,copy_executions=0,memset_executions=0,
            node_execution_bounds=dict(kernel=dict(min=2,max=2),
                                       memcpy=dict(min=0,max=0),memset=dict(min=0,max=0)))]
        row['receipt']['limits'] = dict(max_kernel_records=131072,max_copy_records=32768,max_graph_events=262144)
        row['receipt']['replay']['counts'] = dict(resource_events=10)
        row['replay_census'] = {'total':2,'graphs':{'graph:0':2}}
        rows.append(row)
        begins.append(dict({k: row[k] for k in ('rank','engine_epoch','storage_epoch','route_generation','worker_incarnation')}, ack='flash_s5_observer_begin',state='window',drained_streams=3))
        begins[-1]['replay_census'] = {'total':0,'graphs':{'graph:0':0}}
    receipt = dict(engine_epoch=71,storage_epoch=1,route_generation_after=1,
                   observer=dict(schema=1,complete=True,stopped=True,ranks=rows))
    start = dict(engine_epoch=71,storage_epoch=1,route_generation_after=1,observer=dict(schema=1,complete=False,stopped=False,ranks=begins))
    return {'verdict':'PASS','runs':[{'route':'split','calls':[
        bound_call('pause_enter_split',start), bound_call('pause_end',receipt)]}]}


def bound_call(action, receipt):
    receipt['observer_sha256']=w1._sha256_json(receipt['observer'])
    core=w1._receipt_core(receipt)
    return {'action':action,'lifecycle_ok':True,'receipt':receipt,'receipt_core':core,
            'receipt_sha256':w1._sha256_json(core),'retained_receipt_sha256':w1._sha256_json(receipt)}


class PlacementTests(unittest.TestCase):
    def test_teacher_exception_retains_placement_fail_and_writes_final_grade(self):
        data=audit_data()
        call=data['runs'][0]['calls'][1]
        call['receipt']['observer']['ranks'][0]['receipt']['replay']['verdict']='FAIL'
        data['runs'][0]['calls'][1]=bound_call('pause_end',call['receipt'])
        with tempfile.TemporaryDirectory() as tmp:
            args=types.SimpleNamespace(url='http://127.0.0.1:8000',model='fixture',
                run_id='teacher-exception',out=Path(tmp)/'out',api_source=Path(tmp),
                teacher_baseline=HERE/'baselines/q15-teacher-forced-rule3.json',
                teacher_workload=HERE/'workloads/q17b-teacher-decode-v3.json',
                snapshot_max_bytes=2*1024**3,
                correctness_workload=HERE/'workloads/q17b-correctness-precontrols.json',
                w1_workload=HERE/'workloads/q17b-w1.json')
            ctrl=types.SimpleNamespace(execute=mock.Mock(return_value=({},{})))
            with mock.patch.object(c,'verify_sources',return_value=({'verified':True},[])), \
                 mock.patch.object(c,'run',side_effect=[data,KeyError('sensitive-fixture-value')]), \
                 mock.patch.object(q,'final_status_grade',return_value={'verdict':'PASS'}), \
                 mock.patch.object(w1,'run') as perf:
                result=q.run(args,env={},controller=ctrl)
            perf.assert_not_called()
            ctrl.execute.assert_called_once()
            self.assertEqual(result['verdict'],'FAIL')
            self.assertEqual(result['outcomes']['placement'],'FAIL')
            self.assertEqual(result['outcomes']['teacher_forced'],'INCONCLUSIVE')
            self.assertEqual(result['outcomes']['performance'],'NOT_RUN')
            self.assertEqual(json.loads((args.out/'clients-grade.json').read_text()),result)
            teacher=json.loads((args.out/'teacher-forced.json').read_text())
            self.assertEqual(teacher['error_type'],'KeyError')
            self.assertNotIn('sensitive-fixture-value',json.dumps(teacher))

    def test_partial_or_duplicate_replays_cannot_pass_multiplicity(self):
        for kind in ('lost_replay','balanced_loss_duplicate','missing_census',
                     'extra_record','count_backwards','unknown_descriptor'):
            with self.subTest(kind=kind):
                data=audit_data();call=data['runs'][0]['calls'][1]
                row=call['receipt']['observer']['ranks'][0]
                slot=row['receipt']['replay']['graph_slots'][0]
                if kind=='lost_replay':
                    slot['kernel_executions']=2
                    slot['node_execution_bounds']['kernel']={'min':1,'max':1}
                elif kind=='balanced_loss_duplicate':
                    slot['node_execution_bounds']['kernel']={'min':1,'max':3}
                elif kind=='missing_census':row.pop('replay_census')
                elif kind=='extra_record':slot['kernel_executions']=5
                elif kind=='count_backwards':row['replay_census']['total']=-1
                else:row['replay_census']['graphs']['foreign']=1
                data['runs'][0]['calls'][1]=bound_call('pause_end',call['receipt'])
                grade=q.placement_grade(data)
                self.assertEqual(grade['verdict'],'INCONCLUSIVE',grade)
                self.assertIn('rank0: replay multiplicity incomplete',grade['missing'])

    def test_rule3_control_instability_cannot_unlock_w1_even_with_placement_pass(self):
        data=audit_data()
        data.update(verdict='INCONCLUSIVE', verdict_reasons=[
            'inconclusive:pre_split_controls_differ'])
        with tempfile.TemporaryDirectory() as tmp:
            args=types.SimpleNamespace(url='http://127.0.0.1:8000',
                model='fixture',run_id='precontrol-gate',out=Path(tmp)/'out',
                api_source=Path(tmp),
                teacher_baseline=HERE/'baselines/q15-teacher-forced-rule3.json',
                teacher_workload=HERE/'workloads/q17b-teacher-decode-v3.json',snapshot_max_bytes=2*1024**3,
                correctness_workload=HERE/'workloads/q17b-correctness-precontrols.json',
                w1_workload=HERE/'workloads/q17b-w1.json')
            ctrl=types.SimpleNamespace(execute=lambda *a:({},{}))
            with mock.patch.object(c,'verify_sources',return_value=({'verified':True},[])), \
                 mock.patch.object(c,'run',return_value=data), \
                 mock.patch.object(q,'final_status_grade',return_value={'verdict':'PASS'}), \
                 mock.patch.object(w1,'run') as perf:
                result=q.run(args,env={},controller=ctrl)
            perf.assert_not_called()
            self.assertEqual(result['verdict'],'INCONCLUSIVE')
            self.assertEqual(result['outcomes']['placement'],'PASS')
            self.assertEqual(result['outcomes']['performance'],'NOT_RUN')

    def test_graph_growth_exceeding_ring_cannot_pass(self):
        data=audit_data();call=data['runs'][0]['calls'][1]
        call['receipt']['observer']['ranks'][0]['receipt']['limits']['max_kernel_records']=3
        data['runs'][0]['calls'][1]=bound_call('pause_end',call['receipt'])
        result=q.placement_grade(data)
        self.assertEqual(result['verdict'],'INCONCLUSIVE')
        self.assertEqual(result['capacity']['0']['required_kernel_records'],4)
        self.assertEqual(result['capacity']['0']['largest_graph_replay_capacity'],1)

    def test_exact_four_closed_receipts(self):
        self.assertEqual(q.placement_grade(audit_data())['verdict'],'PASS')

    def test_failed_or_missing_audit_never_becomes_performance(self):
        for mutation, verdict in (
            (lambda o:o.update(stopped=False),'INCONCLUSIVE'),
            (lambda o:o['ranks'].pop(),'FAIL'),
            (lambda o:o['ranks'][0]['receipt'].update(verdict='FAIL'),'FAIL'),
            (lambda o:o['ranks'][0]['receipt']['replay'].update(verdict='INCONCLUSIVE'),'INCONCLUSIVE'),
            (lambda o:o['ranks'][0].update(route_generation=3),'FAIL')):
            data=audit_data();call=data['runs'][0]['calls'][1];mutation(call['receipt']['observer'])
            data['runs'][0]['calls'][1]=bound_call('pause_end',call['receipt'])
            self.assertEqual(q.placement_grade(data)['verdict'],verdict)

    def test_observer_hash_is_part_of_receipt_core_and_retained(self):
        observer=audit_data()['runs'][0]['calls'][1]['receipt']['observer']
        receipt={'schema':1,'observer':observer,'observer_sha256':w1._sha256_json(observer)}
        self.assertEqual(w1._receipt_core(receipt)['observer_sha256'],receipt['observer_sha256'])
        self.assertEqual(w1._sanitize_receipt(receipt)['observer'],observer)

    def test_workloads_are_explicit_and_valid(self):
        short=json.loads((HERE/'workloads/q17b-correctness.json').read_text())
        perf=json.loads((HERE/'workloads/q17b-w1.json').read_text())
        self.assertEqual(c.validate_manifest(short)[0],[])
        self.assertEqual(w1.validate_workload(perf)[0],[])
        self.assertEqual(short['streams'],8)
        self.assertEqual(perf['route_plan'],['full','split','full']*2)
        self.assertEqual(perf['prompt_tokens'],153537)
        pre=json.loads((HERE/'workloads/q17b-correctness-precontrols.json').read_text())
        self.assertEqual(c.validate_manifest(pre)[0],[])
        self.assertEqual(pre['route_plan'],['full','full','split','full'])


class EnginePlacementTests(unittest.TestCase):
    def fixture(self):
        from test_e3_engine_offers import positive, raw_receipt
        import engine_offers as eo
        data=audit_data()
        begins=[];ends=[]
        for rank in range(4):
            owner,bridge,ledger,builder=positive()
            engine=ledger.seal();ledger.deactivate()
            raw=raw_receipt(engine,verdict='INCONCLUSIVE',
                native_reasons=['unattributed_expected_context'],qualified=False)
            def bind(value):
                if type(value) is dict:
                    out={k:bind(v) for k,v in value.items()}
                    for k in ('rank','worker_rank'):
                        if k in out:out[k]=rank
                    if 'engine_incarnation' in out:out['engine_incarnation']='engine:3'
                    if 'worker_incarnation' in out:out['worker_incarnation']='worker-'+str(rank)
                    if 'window_id' in out:out['window_id']='short-placement'
                    return out
                if type(value) is list:return [bind(v) for v in value]
                return value
            engine,raw=bind(engine),bind(raw)
            composed=eo.compose_placement(raw,engine)
            self.assertEqual(composed['verdict'],'PASS',composed)
            row={'rank':rank,'engine_epoch':3,'storage_epoch':1,'route_generation':7,
                 'drained_streams':3,'worker_incarnation':'worker-'+str(rank)}
            begins.append(dict(row,ack='flash_s5_observer_begin',state='window',
                               replay_census={'total':0,'graphs':{'graph:0':0}}))
            ends.append(dict(row,ack='flash_s5_observer_end',state='closed',observer_stopped=True,
                replay_census={'total':1,'graphs':{'graph:0':1}},receipt=raw,
                engine_accounting=engine,composed_placement=composed))
        for call,rows,stopped in ((data['runs'][0]['calls'][0],begins,False),
                                  (data['runs'][0]['calls'][1],ends,True)):
            receipt=call['receipt'];receipt.update(run_id='run-1',engine_epoch=3,route_generation_after=7)
            receipt['observer'].update(ranks=rows,complete=False,stopped=stopped)
            call.update(bound_call(call['action'],receipt))
        return data

    def test_real_composer_discharges_only_accounted_guard_on_all_four_ranks(self):
        data=self.fixture();before=json.dumps(data,sort_keys=True)
        self.assertEqual(q.placement_grade(data)['verdict'],'INCONCLUSIVE')
        grade=q.placement_grade(data,engine_required=True)
        self.assertEqual(grade['verdict'],'PASS',grade)
        self.assertEqual(set(grade['engine_accounting']),{'0','1','2','3'})
        self.assertEqual(json.dumps(data,sort_keys=True),before)

    def test_missing_or_forged_receipt_cannot_use_engine_path(self):
        for kind,want in [('missing','INCONCLUSIVE'),('forged','FAIL'),('foreign','FAIL'),('raw_fail','FAIL')]:
            with self.subTest(kind=kind):
                data=self.fixture();call=data['runs'][0]['calls'][1]
                row=call['receipt']['observer']['ranks'][0]
                if kind=='missing':row.pop('engine_accounting');row.pop('composed_placement')
                elif kind=='forged':row['composed_placement']['verdict']='INCONCLUSIVE'
                elif kind=='foreign':row['engine_accounting']['binding']['run_id']='foreign'
                else:row['receipt']['verdict']='FAIL'
                call.update(bound_call('pause_end',call['receipt']))
                grade=q.placement_grade(data,engine_required=True)
                self.assertEqual(grade['verdict'],want,grade)

class SessionContinuation(ComposedActualSessionTests):
    # Inherit the real session + completion SSE fixture, not invented receipts.
    def test_positive_session_survives_delayed_window_reader(self):
        wait_window = c._wait_window
        def delayed(*args):
            time.sleep(0.12)
            return wait_window(*args)
        with mock.patch.object(c, '_wait_window', side_effect=delayed):
            data = self.run_client(controller=self.session_controller)
        self.assertEqual(data['verdict'], 'PASS', data['verdict_reasons'])

    def test_w1_continues_actual_correctness_session(self):
        context={}
        cfg0=self.config()
        data=c.run(cfg0,self.server.url, 'correctness-test-model',self.session_controller,
                   source=self.source,context=context)
        self.assertEqual(data['verdict'],'PASS',data['verdict_reasons'])
        old_seq=context['command_seq'];old_epoch=context['engine_epoch']
        cfg=types.SimpleNamespace(run_id=cfg0.run_id,
            workload=types.SimpleNamespace(streams=8,route_plan=('full',),cache_preparation=False,name='fixture',manifest_sha256='a'*64),
            out_path=str(Path(self.tmp.name)/'continued.json'))
        seen=[]
        def repeat(cfg,controller,ctx,repeat,route,*args):
            self.assertEqual(repeat,len(cfg0.workload.route_plan))
            self.assertEqual(ctx['command_seq'],old_seq)
            ctx.update(repeat=repeat,cohort_id='continued-cohort')
            # Real ProbeSession accepts the exact preceding canonical receipt.
            command=w1._next_command(ctx,'prepare','full')
            reply,_=controller.execute(command,2)
            seen.append(reply)
            self.assertEqual(reply['command_seq'],old_seq+1)
            self.assertEqual(reply['engine_epoch'],old_epoch)
            return {'streams':[], 'repeat':repeat,'route':route}
        with mock.patch.object(w1,'build_requests',return_value=([],{})), \
             mock.patch.object(w1,'build_request_set',return_value=([],[],'')), \
             mock.patch.object(w1,'_config_artifact',return_value={}), \
             mock.patch.object(w1,'measure_repeat',side_effect=repeat), \
             mock.patch.object(w1,'_apply_preparation_usage'), \
             mock.patch.object(w1,'interval_valid',return_value=(True,[],'valid')), \
             mock.patch.object(w1,'summarize',return_value={'verdict':'COMPLETE'}), \
             mock.patch.object(w1,'write_output'):
            w1.run(cfg,self.server.url,'correctness-test-model',self.session_controller,context=context)
        self.assertEqual(len(seen),1)


class PrecontrolSessionContinuation(SessionContinuation):
    manifest_overrides = {'prompt': {'seed': 17, 'tokens': 16},
                          'route_plan': ['full', 'full', 'split', 'full']}


if __name__=='__main__':unittest.main()
