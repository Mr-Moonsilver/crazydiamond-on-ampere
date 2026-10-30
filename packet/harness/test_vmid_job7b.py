"""Job7b must prove late tensor release and retain the native-crash veto."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import vmid_job7b as j
from test_vmid_job7 import good

class LifetimeTests(unittest.TestCase):
    def row(self):
        return dict(good(),native_lifetime={'native_release':'process-exit','native_retained':True},
                    late_tensor_release_after_close=True)
    def test_saved_job7_measurements_do_not_prove_fixed_teardown(self):
        self.assertEqual(j.case_grade(good())['verdict'],'INCONCLUSIVE')
        self.assertEqual(j.case_grade(self.row())['verdict'],'PASS')
        r=self.row();r['cleanup_ok']=False
        self.assertEqual(j.case_grade(r)['verdict'],'FAIL')
    def test_missing_late_free_or_released_context_cannot_pass(self):
        for key in ('late_tensor_release_after_close','native_lifetime'):
            r=self.row();r.pop(key)
            self.assertEqual(j.case_grade(r)['verdict'],'INCONCLUSIVE')
    def test_process_crash_overrides_completed_phase_receipts(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            raw={'cases':[dict(self.row(),family=f) for f in j.FAMILIES],
                 'regressions':{n:{'verdict':'PASS'} for n in ('job1','job2b_flat')}}
            (p/'probe.json').write_text(json.dumps(raw))
            (p/'monitor.json').write_text(json.dumps({'own_samples':1,'errors':0,'own_peak_mib':650}))
            self.assertEqual(j.finalize(p,139),1)
            self.assertIn('native_process_crash',json.loads((p/'result.json').read_text())['failures'])
    def test_fresh_bounded_queue_and_real_record_stream_reproducer(self):
        root=Path(__file__).resolve().parent
        q=root/'runs/vmid_job7b.sh'
        subprocess.run(['bash','-n',str(q)],check=True)
        for text in ('runs/job7b','flashnext-job7b','flock -n 9','540s'):
            self.assertIn(text,q.read_text())
        source=(root/'vmid_job7b.py').read_text()
        self.assertIn('late_tensor.record_stream(late_stream)',source)
        self.assertLess(source.index('if split:split.close()'),source.index('del late_tensor'))
        self.assertIn('torch.cuda.empty_cache()',source)
        self.assertIn('set_per_process_memory_fraction(.125,0)',source)

if __name__=='__main__':unittest.main()
