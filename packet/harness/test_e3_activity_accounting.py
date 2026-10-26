import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'patches/s5_engine'))
import activity_accounting as a

def fixture():
    o=[dict(id=1,epoch=4,generation=2,thread=1010,system_thread=10,parent=0,family='pre',closed=True,graph=False,
            expected={'kernel':1},context_id=7,stream_id=9)]
    r=dict(schema=a.SCHEMA,thread_id_type="default-pthread-low32",complete=True,overflow=0,errors=0,open_scopes=0,boundary_records=0,activity_records=1,
        scopes=[dict(id=1,parent=0,thread=1010,system_thread=10,action=x) for x in ('push','pop')],
        apis=[dict(correlation_id=10,domain='runtime',name='cudaLaunchKernel_v7000',start=100,end=150,thread=1010,return_value=0),
              dict(correlation_id=11,domain='driver',name='cuLaunchKernel',start=110,end=140,thread=1010,return_value=0)],
        external=[dict(kind='custom0',correlation_id=c,external_id=1) for c in (10,11)],
        activities=[dict(correlation_id=11,kind='kernel',count=1,graph_id=0,context_id=7,stream_id=9,copy_kind=0)])
    return r,o

class JoinTests(unittest.TestCase):
    def grade(self,r,o):return a.grade(r,o,epoch=4,generation=2)
    def test_deferred_out_of_order_runtime_driver_dedup(self):
        r,o=fixture();g=self.grade(r,o);self.assertEqual(g['verdict'],'PASS',g)
        self.assertEqual(g['wrappers'],[dict(wrapper=10,activity_api=11)])
        r['apis'].reverse();r['external'].reverse()
        self.assertEqual(self.grade(r,o)['joined_records'],1)
    def test_missing_duplicate_foreign_external_and_generations_refused(self):
        for field in ('missing','duplicate','unknown','stale','thread','epoch','open','overflow'):
            r,o=fixture()
            if field=='missing':r['external'].pop()
            if field=='duplicate':r['external'].append(r['external'][0])
            if field=='unknown':r['external'][0]['external_id']=999
            if field=='stale':o[0]['generation']=1
            if field=='epoch':o[0]['epoch']=3
            if field=='thread':r['apis'][1]['thread']=20
            if field=='open':r['open_scopes']=1
            if field=='overflow':r['overflow']=1
            self.assertNotEqual(self.grade(r,o)['verdict'],'PASS',field)
    def test_no_nearest_timestamp_or_symbol_fallback(self):
        r,o=fixture();r['apis'][0].update(start=80,end=90)
        self.assertIn('submission_without_unique_activity',self.grade(r,o)['missing'])
        r,o=fixture();r['activities'][0]['correlation_id']=55
        self.assertIn('unoffered_activity',self.grade(r,o)['failures'])
    def test_runtime_referenced_or_driver_referenced_but_not_both_counted_twice(self):
        r,o=fixture();r['activities'][0]['correlation_id']=10
        self.assertEqual(self.grade(r,o)['verdict'],'PASS')
        r['activities'].append(dict(r['activities'][0],correlation_id=11));r['activity_records']=2
        self.assertIn('offer_count_mismatch',self.grade(r,o)['failures'])
    def test_wrong_context_direction_and_failed_api_refused(self):
        r,o=fixture();r['activities'][0]['context_id']=1
        self.assertIn('wrong_context',self.grade(r,o)['failures'])
        r,o=fixture();r['apis'][1]['return_value']=1
        self.assertIn('api_submission_failed',self.grade(r,o)['failures'])
        r,o=fixture();r['activities'][0].update(kind='memcpy',copy_kind=2)
        self.assertIn('copy_direction',self.grade(r,o)['failures'])
    def test_two_threads_parent_scopes_no_work_and_nested_exception_status(self):
        r,o=fixture();o.append(dict(o[0],id=2,thread=1020,system_thread=20,expected={},family='exception_parent'))
        r['scopes'] += [dict(id=2,parent=0,thread=1020,system_thread=20,action=x) for x in ('push','pop')]
        self.assertEqual(self.grade(r,o)['verdict'],'PASS')
        o[0]['parent']=2
        self.assertIn('offer_parent',self.grade(r,o)['missing'])
    def test_missing_api_completion_unknown_api_and_census_refused(self):
        r,o=fixture();r['apis'][1]['name']='cuUnknownLaunch'
        self.assertNotEqual(self.grade(r,o)['verdict'],'PASS')
        r,o=fixture();r['activity_records']=2
        self.assertIn('activity_census',self.grade(r,o)['missing'])

if __name__=='__main__':unittest.main()
