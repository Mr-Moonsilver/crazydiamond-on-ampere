"""Full-preconditioning and receipt-policy gates; no device or live service."""
import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import tempfile
import native30_host as host
import native30_policy as policy
from test_native30_host import fixture,Obj

HERE=Path(__file__).resolve().parent
class Policy(unittest.TestCase):
    def setUp(self):
        self.rows=[];self.previous=host._EMIT
        host._EMIT=lambda event,**fields:self.rows.append(dict(event=event,**fields))
    def tearDown(self):host._EMIT=self.previous

    def test_only_first_row4_admission_activates_and_rotations_stay_global(self):
        pool,runner,s=fixture();s['active']=False;obj=Obj();obj.runner=runner;obj.events=[]
        marker=object();calls=[]
        original=lambda *a:(calls.append(1) or marker)
        for _ in range(3):pool.copy_to_uva([1])
        self.assertIs(policy.qualification(original,runner,dict(action='teacher_configure',enabled=True)),marker)
        fields=dict(rank=1,label='FIRST_ADMISSION',call_id=1,chunk_start=0,chunk_end=1024)
        with policy.boundary(obj,fields):pass
        self.assertFalse(s['active']);self.assertFalse(any(x['event']=='host-policy-admission' for x in self.rows))
        policy.qualification(original,runner,dict(action='observer_begin'))
        with policy.boundary(obj,fields):pool.copy_to_uva([2])
        self.assertTrue(s['active']);self.assertEqual(host.pool_state(pool)['rotations'],4)
        self.assertEqual(len([x for x in self.rows if x['event']=='host-policy-admission']),1)
        with policy.boundary(obj,fields):pass
        self.assertEqual(len([x for x in self.rows if x['event']=='host-policy-admission']),1)
        self.assertEqual(calls,[1,1])

    def test_observer_end_keeps_receipts_for_30_seconds_and_disable_does_not_cut_tail(self):
        _,runner,s=fixture();w=policy.window(s);w.update(started=True,armed=True)
        timers=[]
        class Timer:
            def __init__(self,delay,callback):self.delay=delay;self.callback=callback;timers.append(self)
            def start(self):pass
        with patch.object(policy.threading,'Timer',Timer),patch.object(policy.time,'monotonic',return_value=100):
            marker=object();self.assertIs(policy.qualification(lambda *a:marker,runner,dict(action='observer_end')),marker)
            policy.qualification(lambda *a:marker,runner,dict(action='teacher_configure',enabled=False))
        self.assertTrue(s['active']);self.assertEqual(timers[0].delay,30)
        with patch.object(policy.time,'monotonic',return_value=129):policy.finish(s)
        self.assertTrue(s['active'])
        with patch.object(policy.time,'monotonic',return_value=130):timers[0].callback()
        self.assertFalse(s['active']);self.assertEqual(self.rows[-1]['seconds_after_join'],30)

    def test_budget_covers_measured_peak_and_preserves_each_line_bound(self):
        self.assertGreater(policy.BYTES_PER_RANK,7.165*1024**2*120)
        self.assertEqual(policy.RECORDS_PER_RANK,policy.BYTES_PER_RANK//512)
        self.assertEqual((host.MAX_LINE_BYTES,host.MAX_RECORD_BYTES,host.FRAGMENT_CHARS),(4096,128*1024,1400))

class Queue(unittest.TestCase):
    def test_full_native28_phase_block_is_exact_and_restore_bodies_match_d3(self):
        old=(HERE/'runs/q18a_native28.draft.sh').read_text();new=(HERE/'runs/q18a_native30_localize_e.draft.sh').read_text()
        start='python3 "$H/e3_native_request_plans.py" --url "$PROBE_URL"'
        block=start+old.split(start,1)[1].split('\nRULE4_CONTINUATION=0',1)[0]
        self.assertIn(block,new)
        self.assertEqual([x.split()[-1] for x in block.splitlines() if x.startswith('qualification_phase ')],['rule4_controls','rule4_candidate','row4'])
        self.assertIn('  qualification_phase row1',block)
        self.assertNotIn('qualification_ready=',new);self.assertNotIn('record w1_controls',new)
        d3=(HERE/'runs/q18a_native30_localize_d3.draft.sh').read_text()
        for name in ('stop_engine','restore_launcher','restore_defaults','write_cleanup'):
            body=lambda s:s.split(name+'() {',1)[1].split('\n}\n',1)[0]
            self.assertEqual(body(new),body(d3))
        self.assertIn('sleep 30',new);self.assertIn('2100s env',new)
        self.assertIn('--placement "$Q/qual-row4.json"',new)

    def test_generation_override_explains_64_without_changing_native28_requests(self):
        from e3_q18a_qualification import _workload_object
        from e3_native_qualification_driver import NativePlacementRunner
        workload=_workload_object(json.loads((HERE/'workloads/q17b-correctness-precontrols.json').read_text()))
        runner=NativePlacementRunner(workload=workload,url='http://127.0.0.1:1')
        self.assertEqual(workload.generation_budget,256);self.assertEqual(runner.workload.generation_budget,64)
        self.assertEqual(runner.workload.window_tokens,20)  # native35 P-c: 2 cohorts x 20 = PLACEMENT_REPLAYS 40

    def test_grade_cannot_pass_without_complete_host_tail_and_engine_fault_wins(self):
        from e3_native30_e_grade import evaluate
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'err';rows=[]
            for rank in range(4):
                for i,label in enumerate(('FIRST_ADMISSION','DECODE_STEP')):
                    for event in ('begin','end'):rows.append(dict(rank=rank,pid=rank,call_id=i,event=event,label=label))
                for event in ('host-policy-admission','host-policy-join','host-policy-end'):
                    rows.append(dict(rank=rank,event=event,seconds_after_join=30))
            def save(data):path.write_text(''.join('NATIVE30 '+json.dumps(x)+'\n' for x in data))
            save(rows);self.assertEqual(evaluate([path],{'verdict':'PASS'},False,root/'a')['verdict'],'PASS')
            save(rows[:-1]);self.assertEqual(evaluate([path],{'verdict':'PASS'},False,root/'b')['verdict'],'INCONCLUSIVE')
            save(rows+[dict(rank=1,event='host-receipt-limit')]);r=evaluate([path],{'verdict':'PASS'},False,root/'c')
            self.assertEqual((r['verdict'],r['failure_class']),('INCONCLUSIVE','HARNESS'))
            self.assertEqual(evaluate([path],{'verdict':'PASS'},True,root/'d')['failure_class'],'ENGINE')

if __name__=='__main__':unittest.main()
