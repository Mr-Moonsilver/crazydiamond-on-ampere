"""Teacher12 narrowing: real client orchestration, fake OS lifecycle, no GPU."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import e3_q17b_clients as q
import e3_decode_correctness as c
import e3_w1_decode_driver as w1
import test_e3_q17b_full_queue as queues
from test_e3_q17b_clients import audit_data

HERE=Path(__file__).resolve().parent

class Teacher12Clients(unittest.TestCase):
    def test_never_runs_w1_even_when_every_gate_passes_and_keeps_raw_verdicts(self):
        for teacher,placement,final,want in [
            ('PASS','PASS','PASS','PASS'),
            ('PASS','INCONCLUSIVE','PASS','INCONCLUSIVE'),
            ('INCONCLUSIVE','PASS','PASS','INCONCLUSIVE'),
            ('PASS','FAIL','PASS','FAIL'),
            ('PASS','PASS','FAIL','FAIL'),
        ]:
            with self.subTest(teacher=teacher,placement=placement,final=final), tempfile.TemporaryDirectory() as tmp:
                args=types.SimpleNamespace(url='http://127.0.0.1:8000',model='fixture',
                    run_id='teacher12-fixture',out=Path(tmp)/'out',api_source=Path(tmp),
                    teacher_baseline=HERE/'baselines/q15-teacher-forced-rule3.json',
                    teacher_workload=HERE/'workloads/q17b-teacher-decode-v3.json',
                    correctness_workload=HERE/'workloads/q17b-correctness-precontrols.json',
                    snapshot_max_bytes=2*1024**3,teacher_diagnostic=True)
                ctrl=types.SimpleNamespace(execute=mock.Mock(return_value=({},{})))
                raw={'verdict':placement,'failures':[],'missing':['unattributed_expected_context']}
                with mock.patch.object(c,'verify_sources',return_value=({'verified':True},[])), \
                     mock.patch.object(c,'run',side_effect=[audit_data(),{'verdict':teacher}]) as runs, \
                     mock.patch.object(q,'placement_grade',return_value=raw), \
                     mock.patch.object(q,'data_invariant_grade',return_value={'verdict':'PASS'}), \
                     mock.patch.object(q,'final_status_grade',return_value={'verdict':final}), \
                     mock.patch.object(w1,'run') as perf:
                    result=q.run(args,env={},controller=ctrl)
                perf.assert_not_called()
                self.assertEqual(runs.call_count,2) # short unchanged cohort then fixed four-run teacher plan
                self.assertEqual(result['verdict'],want)
                self.assertFalse(result['w1_eligible'])
                self.assertEqual(result['outcomes']['performance'],'NOT_RUN')
                self.assertEqual(json.loads((args.out/'placement.json').read_text()),raw)
                self.assertEqual(json.loads((args.out/'clients-grade.json').read_text()),result)

class Teacher12Queue(queues.Full11Queue):
    queue_name='q17b_teacher12.draft.sh'
    def make_harness(self):
        super().make_harness()
        p=self.h/'e3_q17b_clients.py'
        s=p.read_text().replace('import argparse,json,os','import argparse,json,os,sys\nassert "--teacher-diagnostic" in sys.argv')
        s=s.replace("(out/name).write_text('{}')", "(out/name).write_text(json.dumps({'verdict':'NOT_RUN'} if name=='performance.json' else {}))")
        s=s.replace('performance=verdict','performance="NOT_RUN",final_status=verdict')
        s=s.replace("schema=1,run_id=a.run_id", "scope='teacher-row-diagnostic',w1_eligible=False,schema=1,run_id=a.run_id")
        p.write_text(s);queues.old.write_manifest(self.h)

    def test_only_identity_changes_from_protocol_v2_queue(self):
        # New queue changes client selection/consistency only; cleanup stays byte-identical.
        before=(HERE/'runs/q17b_full11.draft.sh').read_text()
        after=(HERE/'runs'/self.queue_name).read_text()
        self.assertEqual(before[before.index('stop_sampler || true\ncapture_logs || true',before.index('crc=$?')):],
                         after[after.index('stop_sampler || true\ncapture_logs || true',after.index('crc=$?')):])
        self.assertIn('--teacher-diagnostic --url',after)
        declare=(HERE/'runs/q17b_teacher12_declare.sh').read_text()
        self.assertIn('QNAME=${QNAME:-q17b-teacher12}',declare)
        self.assertIn('q17b_teacher12.draft.sh',declare)
        self.assertNotIn('q17b_full11.draft.sh',declare)

    def test_accidental_performance_result_rejected(self):
        p=self.h/'e3_q17b_clients.py';p.write_text(p.read_text().replace('performance="NOT_RUN"','performance="PASS"'))
        queues.old.write_manifest(self.h)
        result=self.run_queue()
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertIn('STEP clients-grade-consistency FAIL',result.stdout)
        self.assertIn('STEP defaults-restore PASS',result.stdout)

if __name__=='__main__': unittest.main()
