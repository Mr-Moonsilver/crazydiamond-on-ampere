import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'patches/s5_engine'))
import launch_diagnostic as d

def sample():
    fields=[0]*len(d.FIELDS)
    fields[d.FIELDS.index('gridX')]=4
    fields[d.FIELDS.index('blockX')]=32
    fields[d.FIELDS.index('accessPolicyWindow_base_private')]=1234567890123456
    return ({'complete':True,'aliases':[], 'nodes':[
        {'graph':11,'node':n,'complete':True,'stable':True,'name':'kernel'+str(n),
         'records':4,'fields':list(fields)} for n in (101,102)]},
        {'full:4':{'graph':11,'nodes':[102,101]}})
class LaunchTests(unittest.TestCase):
    def test_exact_fields_and_redacted_pointer_are_diagnostic_only(self):
        out=d.compare(sample(),sample());self.assertEqual(out['verdict'],'PASS')
        self.assertFalse(out['gates_w1']);self.assertEqual(len(out['kernels']),2)
        self.assertNotIn('1234567890123456',str(out))
        self.assertEqual(out['kernels'][0]['full_fields'][d.FIELDS.index('gridX')],4)
    def test_any_configuration_field_difference_is_reported(self):
        for index,field in enumerate(d.FIELDS):
            a,b=sample(),sample();b[0]['nodes'][0]['fields'][index]+=1
            out=d.compare(a,b);self.assertEqual(out['verdict'],'DIFFER',field)
            self.assertIn(field,out['kernels'][0]['differing_fields'])
    def test_incomplete_record_loss_or_missing_node_is_never_pass(self):
        for change in (lambda r:r.update(complete=False),lambda r:r['nodes'].pop(),
                       lambda r:r['nodes'][0].update(complete=False),
                       lambda r:r['nodes'][0].update(records=3)):
            a,b=sample(),sample();change(b[0]);self.assertEqual(d.compare(a,b)['verdict'],'INCONCLUSIVE')
    def test_clone_alias_mapping_and_cycle_refusal(self):
        a,b=sample(),sample();b[0]['aliases']=[[21,11,201,101],[21,11,202,102]]
        for row,n in zip(b[0]['nodes'],(201,202)):row.update(graph=21,node=n)
        self.assertEqual(d.compare(a,b)['verdict'],'PASS')
        b[0]['aliases'].append([11,21,0,0])
        self.assertEqual(d.compare(a,b)['verdict'],'INCONCLUSIVE')
    def test_within_mode_variation_and_symbol_change_are_differ(self):
        for field,value in [('stable',False),('name','other_kernel')]:
            b=sample();b[0]['nodes'][0][field]=value
            self.assertEqual(d.compare(sample(),b)['verdict'],'DIFFER')
if __name__=='__main__':unittest.main()
