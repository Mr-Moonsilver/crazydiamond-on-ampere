"""Frozen VM7 verdict, exact ID join and native host-extension compile (CPU)."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
import vmid_job7 as j


def good():
    r={'family':'shared','failures':[],'missing':[], 'cleanup_ok':True,'owners_restored':True,
       'serialized_stable':True,'source_pins_verified':True,'overlap_selected':True,
       'source_scope':'real owner methods; synthetic small arithmetic and TP1/DP1 PLE fixture',
       'symbol_join':[{'node_index':8,'symbol':'kernel'}],'windows':{},
       'native_window':{'flags':{'known_wrong_context':True},'reasons':['known_wrong_context','missing_execution','incomplete_record'],
            'expected_context':{'is_green':True,'sm_matching':1},'counts':{'known_wrong_context_records':9},
            'graph_slots':[{'reasons':['known_wrong_context']},{},{'reasons':['known_wrong_context']}]},
       'repaired_slot':{'complete':True,'required':{'kernel':2,'memcpy':1,'memset':0},
            'kernel_executions':12,'copy_executions':6,'memset_executions':0,
            'node_execution_bounds':{'kernel':{'min':6,'max':6},'memcpy':{'min':6,'max':6}}}}
    for name,counts in [('before',{'expected_green':3,'expected_parent':3}),('repaired',{'expected_green':6}),('cross',{'expected_green':6}),('restored',{'expected_parent':6})]:
        r['windows'][name]={'output_exact':True,'replays':3,'complete':True,'context_counts':counts,'predicate':'FAIL' if 'expected_parent' in counts else 'PASS'}
    return r

class GradeTests(unittest.TestCase):
    def test_expected_negative_is_not_repaired_pass(self):
        r=good();self.assertEqual(j.case_grade(r)['verdict'],'PASS')
        r['windows']['repaired']['context_counts']={'expected_parent':6}
        self.assertEqual(j.case_grade(r)['verdict'],'FAIL')
    def test_native_repaired_copy_context_failure_stays_fail(self):
        r=good();r['repaired_slot']['complete']=False
        r['repaired_slot']['reasons']=['known_wrong_context']
        self.assertEqual(j.case_grade(r)['verdict'],'FAIL')

    def test_native_guard_must_reject_negative(self):
        r=good();r['native_window']['graph_slots'][0]['reasons']=[]
        self.assertEqual(j.case_grade(r)['verdict'],'INCONCLUSIVE')
    def test_missing_reproduction_not_pass(self):
        r=good();r['windows']['before']['context_counts']={'expected_green':6}
        self.assertEqual(j.case_grade(r)['verdict'],'INCONCLUSIVE')
    def test_loss_multiplicity_cleanup_fail_closed(self):
        r=good();r['native_window']['flags']['permanent_overflow']=True
        self.assertEqual(j.case_grade(r)['verdict'],'INCONCLUSIVE')
        r=good();r['repaired_slot']['node_execution_bounds']['kernel']['min']=5
        self.assertEqual(j.case_grade(r)['verdict'],'INCONCLUSIVE')
        r['cleanup_ok']=False;self.assertEqual(j.case_grade(r)['verdict'],'FAIL')
    def test_third_context_and_missing_regression(self):
        r=good();r['windows']['before']['context_counts']['other']=1
        self.assertEqual(j.case_grade(r)['verdict'],'FAIL')
        allrows={'cases':[dict(good(),family=f) for f in j.FAMILIES]}
        self.assertEqual(j.grade(allrows)['verdict'],'INCONCLUSIVE')
        allrows['regressions']={n:{'verdict':'PASS'} for n in ('job1','job2b_flat')}
        self.assertEqual(j.grade(allrows)['verdict'],'PASS')
    def test_authoritative_join_does_not_use_local_index_as_ordinal(self):
        from launch_diagnostic import FIELDS
        # Interleaved copy creates local index 8 for kernel ordinal 1.
        d={'nodes':[{'slot':0,'node_index':2,'graph_node_id':100},{'slot':0,'node_index':8,'graph_node_id':200}]}
        report={'complete':True,'aliases':[[11,1,1001,100],[11,1,1002,200]],'nodes':[
            {'graph':11,'node':1001,'name':'first','fields':[0]*len(FIELDS),'records':3,'complete':True},
            {'graph':11,'node':1002,'name':'second','fields':[0]*len(FIELDS),'records':3,'complete':True}]}
        rows=j.symbol_join(d,report,{'0':{'graph':1,'nodes':[200,100]}})
        self.assertEqual([(r['node_index'],r['symbol']) for r in rows],[(2,'first'),(8,'second')])
        d['nodes'].append({'slot':0,'node_index':9,'graph_node_id':300})
        with self.assertRaises(ValueError):j.symbol_join(d,report,{'0':{'graph':1,'nodes':[200,100]}})
    def test_queue_is_new_bounded_and_includes_unchanged_regressions(self):
        p=ROOT/'harness/runs/vmid_job7.sh';subprocess.run(['bash','-n',str(p)],check=True)
        s=p.read_text();self.assertIn('flock -n 9',s);self.assertIn('540s',s);self.assertIn('flashnext-job7',s)
        self.assertIn('vmid_graph_probe.cu',s)
        src=(ROOT/'harness/vmid_job7.py').read_text();self.assertIn('set_per_process_memory_fraction(.125,0)',src)
        self.assertIn("for name in ('job1','job2')",src)
    def test_native_host_extension_compiles_against_pinned_cuda13(self):
        source=(ROOT/'patches/s5_engine/native/flash_s5_green.cu').read_text()
        source=source[:source.index('// SM membership probe:')]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'green-host.cpp';p.write_text(source)
            result=subprocess.run(['g++','-std=c++17','-fsyntax-only','-I'+str(ROOT/'reference/cuda-runtime-13'),str(p)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
