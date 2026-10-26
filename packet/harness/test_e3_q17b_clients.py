"""Full q17b client composition and placement gates, CPU only."""
import copy
import json
from pathlib import Path
import tempfile
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
        rows.append(row)
        begins.append(dict({k: row[k] for k in ('rank','engine_epoch','storage_epoch','route_generation','worker_incarnation')}, ack='flash_s5_observer_begin',state='window',drained_streams=3))
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


class SessionContinuation(ComposedActualSessionTests):
    # Inherit the real session + completion SSE fixture, not invented receipts.
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
            self.assertEqual(repeat,3)
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


if __name__=='__main__':unittest.main()
