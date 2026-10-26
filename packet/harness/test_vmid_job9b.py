import copy
from pathlib import Path
import subprocess
import unittest
import vmid_job9b as j
from test_e3_activity_accounting import fixture

class Job9BTests(unittest.TestCase):
    def test_posix_and_system_spaces_are_explicit_and_checked_both_ways(self):
        a=j.account_module();r,o=fixture()
        self.assertNotEqual(o[0]['thread'],o[0]['system_thread'])
        self.assertEqual(a.grade(r,o,epoch=4,generation=2)['verdict'],'PASS')
        o[0]['thread']=o[0]['system_thread']
        self.assertNotEqual(a.grade(r,o,epoch=4,generation=2)['verdict'],'PASS')
        r,o=fixture();r['thread_id_type']='system-gettid'
        self.assertNotEqual(a.grade(r,o,epoch=4,generation=2)['verdict'],'PASS')
    def test_zero_work_state_query_is_positive_host_evidence_not_a_launch(self):
        a=j.account_module();r,o=fixture()
        r['apis'].append(dict(correlation_id=40,domain='driver',thread=1001,start=110,end=120,name='cuDevicePrimaryCtxGetState',return_value=0))
        self.assertEqual(a.grade(r,o,epoch=4,generation=2)['verdict'],'PASS')
        r['activities'][0]['correlation_id']=40
        self.assertNotEqual(a.grade(r,o,epoch=4,generation=2)['verdict'],'PASS')
    def test_two_wrappers_cannot_hide_a_lost_submission(self):
        a=j.account_module();r,o=fixture()
        r['apis'].append(dict(r['apis'][0],correlation_id=12))
        r['external'].append(dict(r['external'][0],correlation_id=12))
        self.assertIn('submission_without_unique_activity',a.grade(r,o,epoch=4,generation=2)['missing'])
    def test_explicit_wrapper_challenge_discriminates(self):
        a=j.account_module();r,o=fixture();o[0].update(epoch=1,generation=1,graph=True)
        r['apis']=r['apis'][:1];r['apis'][0]['name']='cudaGraphLaunch_v10000'
        r['external']=r['external'][:1];r['activities'][0].update(correlation_id=10,graph_id=7)
        g=j.wrapper_challenge(r,o,a.grade)
        self.assertEqual(g['verdict'],'PASS',g)
        self.assertEqual(g['native_wrappers'],0)
        self.assertEqual(g['native_work_api_domains'],['runtime'])
        self.assertEqual(g['duplicate_wrapper'],'INCONCLUSIVE')
        self.assertIn('not GPU-observed',g['basis'])
    def test_job9b_fresh_paths_and_shell(self):
        p=Path(__file__).parent/'runs/vmid_job9b.sh';s=p.read_text()
        subprocess.run(['bash','-n',str(p)],check=True)
        self.assertIn('Q=/opt/flashnext/runs/job9b',s)
        self.assertIn('flashnext-job9b',s)
        self.assertIn('flock -n 9',s)
        self.assertIn('set_per_process_memory_fraction(.125,0)',Path(j.__file__).read_text())
if __name__=='__main__':unittest.main()
