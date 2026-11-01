"""Queue/client/grade CPU seams; hardware, HTTP peers explicitly declared."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'patches/s5_engine'),str(ROOT/'harness')]
import native43_boundary_client as client
import native43_boundary_clock as clock
import native43_boundary_gate as gate
import native43_dense_grade as grade


def derived():
    spec=importlib.util.spec_from_file_location('native43_queue',ROOT/'tools/native43_queue_drafts.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.derive()


def evidence(policy):
    view=dict(verdict='PASS',cohort_streams=0,fresh=True,prompt_sha256='prompt',paired_salt_id='pair')
    rows=[]
    for rank in range(4):
        for layer in range(6):
            rows.append(dict(policy=policy,rank=rank,layer='qsa.'+str(layer),chunk_start=100224,
                chunk_end=101200,route='dense_dma' if policy=='dense' else 'sparse_dma',
                timing_excluded='final-chunk-boundary-readback',gpu_timing_complete=True,
                selection_evidence=dict(sha256='selection',dtype='torch.int32',shape=[976,5],bytes=19520),
                output_evidence=dict(sha256='output',dtype='torch.bfloat16',shape=[976,6,256],bytes=2998272)))
    return view,rows


class QueueTests(unittest.TestCase):
    def test_all_seven_drafts_exact_and_shell_parsed(self):
        texts=derived();self.assertEqual(len(texts),7)
        for name,text in texts.items():
            self.assertEqual((ROOT/name).read_text(),text)
            subprocess.run(['bash','-n',str(ROOT/name)],check=True,capture_output=True)

    def test_arm_phase_has_no_extra_workloads_or_cohort_and_literal_env(self):
        for policy in ('current','dense'):
            text=(ROOT/f'harness/runs/q18a_native43_{policy}.draft.sh').read_text()
            settings=(ROOT/f'harness/runs/q18a_native43_{policy}_settings.sh').read_text()
            self.assertEqual(text.count('python3 "$H/native43_boundary_client.py"'),1)
            self.assertIn('--policy '+policy,text)
            self.assertIn('FLASH_S5_NATIVE_QSA_POLICY='+policy,settings)
            self.assertIn('FLASH_S5_REPLAY_OBSERVER=0 FLASH_S5_ENGINE_OFFERS=0',settings)
            self.assertNotIn('--round ',text)
            self.assertNotIn('python3 "$H/e3_q18a_qualification.py"',text)
            self.assertNotIn('python3 "$H/e3_native39_w1.py"',text)
            self.assertIn('from native43_boundary_gate import arm_window_ok',text)
            self.assertIn('restore_launcher || true',text)
            self.assertIn('restore_defaults || true',text)

    def test_window_uses_same_budget_real_wall_time_and_edges(self):
        self.assertEqual(gate.PAIR_ACTIVE_S,2*(gate.ARM_ACTIVE_S+gate.ARM_KILL_MARGIN_S))
        self.assertTrue(gate.window_ok(time.time()))
        self.assertFalse(gate.window_ok(gate.NOT_BEFORE-1))
        self.assertFalse(gate.window_ok(gate.HARD_END-gate.PAIR_ACTIVE_S-gate.PAIR_KILL_MARGIN_S+1))
        draft=(ROOT/'harness/runs/q18a_native43_dense_boundary.draft.sh').read_text()
        self.assertIn(str(gate.PAIR_ACTIVE_S)+'s',draft)
        self.assertNotIn('datetime',draft)

    def test_real_cli_exact_queue_args_reach_tokenize_after_real_manifest_validation(self):
        import e3_native39_w2_length as length
        calls=[]
        def refused(url,path,body,timeout):
            calls.append((url,path,body,timeout));raise OSError('CPU first-network boundary')
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'client.json'
            argv=['--url','http://127.0.0.1:1','--model','fixture-model','--run-id','fixture-run',
                  '--epoch','1','--pair-id','fixture-pair','--policy','dense',
                  '--workload',str(ROOT/'harness/workloads/q18a-w2-overlap-native39.json'),
                  '--timeout-s','300','--out',str(out)]
            with patch.object(length,'_post',side_effect=refused):
                self.assertEqual(client.main(argv),2)
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][1],'/tokenize')
            self.assertEqual(calls[0][2]['model'],'fixture-model')
            self.assertTrue(calls[0][2]['prompt'])
            self.assertEqual(json.loads(out.read_text())['verdict'],'INCONCLUSIVE')

    def test_real_engine_status_clock_bracket(self):
        from native43_clock_probe import engine_packet
        before=time.monotonic_ns();packet=engine_packet();after=time.monotonic_ns()
        proof=clock.prove(packet['status'],before,after)
        self.assertTrue(proof['proved'])
        self.assertFalse(proof['identity_equality_required'])
        self.assertFalse(clock.prove(packet['status'],after+1,after+2)['proved'])


class GradeTests(unittest.TestCase):
    def test_exact_complete_pair_only_can_pass(self):
        a,ar=evidence('current');b,br=evidence('dense')
        self.assertEqual(grade.grade(a,b,ar,br)['verdict'],'PASS')
        self.assertFalse(grade.grade(a,b,ar,br)['speed_claim'])

    def test_each_selection_or_output_difference_is_retained_failure(self):
        for field in ('selection_evidence','output_evidence'):
            a,ar=evidence('current');b,br=evidence('dense')
            br[0][field]['sha256']='different'
            result=grade.grade(a,b,ar,br)
            self.assertEqual(result['verdict'],'FAIL')
            self.assertTrue(any(field in x for x in result['failures']))

    def test_missing_layer_or_timing_or_client_cannot_pass(self):
        for kind in ('layer','timing','client'):
            a,ar=evidence('current');b,br=evidence('dense')
            if kind=='layer':br.pop()
            elif kind=='timing':br[0]['gpu_timing_complete']=False
            else:b['verdict']='INCONCLUSIVE'
            self.assertEqual(grade.grade(a,b,ar,br)['verdict'],'INCONCLUSIVE')

    def test_wrong_policy_overrun_or_prompt_pair_cannot_pass(self):
        for kind in ('policy','overrun','prompt'):
            a,ar=evidence('current');b,br=evidence('dense')
            if kind=='policy':br[0]['policy']='current'
            elif kind=='overrun':br[0]['chunk_end']=102224
            else:b['prompt_sha256']='different'
            self.assertEqual(grade.grade(a,b,ar,br)['verdict'],'FAIL')

if __name__=='__main__':unittest.main()
