import copy
from pathlib import Path
import subprocess
import unittest
import vmid_job9 as j
from test_e3_activity_accounting import fixture

class Job9Tests(unittest.TestCase):
    def test_mutations_all_refused(self):
        r,o=fixture();o[0].update(epoch=1,generation=1)
        g=j.account_module().grade(r,o,epoch=1,generation=1)
        self.assertEqual(g['verdict'],'PASS',g)
        x=j.injections(r,o,j.account_module().grade)
        self.assertEqual(len(x),13)
        self.assertTrue(all(a['rejected'] for a in x.values()),x)
    def test_missing_never_passes(self):
        self.assertEqual(j.grade({})['verdict'],'INCONCLUSIVE')
        self.assertNotEqual(j.case_grade({'mode':'positive','accounting':{'grade':{'verdict':'PASS'}}})['verdict'],'PASS')
    def test_two_threads_wrapper_and_native_guard_are_independent(self):
        r,o=fixture();o[0].update(epoch=1,generation=1)
        row={'mode':'positive','accounting':{'native':r,'offers':o},'window':{'reasons':[]}}
        g=j.case_grade(row)
        self.assertIn('two_submit_threads',g['missing']);self.assertIn('raw_graph_guard_missing',g['missing'])
    def test_native_unknown_cannot_be_excused_by_negative(self):
        r,o=fixture();o[0].update(epoch=1,generation=1);r['external']=[];r['complete']=False
        g=j.case_grade({'mode':'unoffered','accounting':{'native':r,'offers':o}})
        self.assertIn('negative_native_incomplete',g['missing'])
    def test_queue_owned_fresh_paths_caps_and_syntax(self):
        p=Path(__file__).parent/'runs/vmid_job9.sh';s=p.read_text()
        subprocess.run(['bash','-n',str(p)],check=True)
        for text in ('Q=/opt/flashnext/runs/job9','flock -n 9','flashnext-job9','--network none','540s','--memory 4g','max_split_size_mb:64'):
            self.assertIn(text,s)
        self.assertNotIn('vllm-flash',s)
        code=Path(j.__file__).read_text()
        self.assertIn('set_per_process_memory_fraction(.125,0)',code)
        self.assertIn('threading.Barrier(3,timeout=30)',code)
        self.assertIn("for mode in ('positive','unoffered','primary')",code)
    def test_complete_positive_and_exact_bank_multiplicity(self):
        r,o=fixture();o[0].update(epoch=1,generation=1)
        o.append(dict(o[0],id=2,thread=1020,system_thread=20,family='exception',expected={},body_exception=True))
        r['scopes'] += [dict(id=2,parent=0,thread=1020,system_thread=20,action=x) for x in ('push','pop')]
        o.append(dict(o[0],id=3,family='graph',graph=True,expected={k:n*40 for k,n in j.MIX.items()},copy_kinds=[1,2,8]))
        r['scopes'] += [dict(id=3,parent=0,thread=1010,system_thread=10,action=x) for x in ('push','pop')]
        r['apis'].append(dict(r['apis'][1],correlation_id=12,name='cuGraphLaunch',start=200,end=210))
        r['external'].append(dict(kind='custom0',correlation_id=12,external_id=3))
        r['activities'] += [dict(correlation_id=12,kind=k,count=n*40,graph_id=9,context_id=7,stream_id=9,copy_kind=1 if k=='memcpy' else 0) for k,n in j.MIX.items()]
        r['activity_records']=1+40*sum(j.MIX.values())
        counts={k:0 for k in ('dropped_records_total','buffers_refused','drop_query_errors','parse_errors','callback_errors','unknown_node_records','ambiguous_node_records','kind_mismatch_records','known_wrong_context_records','unknown_activity_records','unknown_resource_events')}
        counts.update(resource_events=194268,kernel_records=55441,copy_records=1320,ring_overflows={'kernel':0,'copy':0})
        slots=[]
        for i in range(7):
            n=40 if i==0 else 0
            slot=dict(required=j.MIX,node_execution_bounds={k:dict(min=n,max=n) for k in j.MIX})
            for k,f in (('kernel','kernel_executions'),('memcpy','copy_executions'),('memset','memset_executions')):slot[f]=n*j.MIX[k]
            slots.append(slot)
        row=dict(mode='positive',limits=j.LIMITS,pressure_graphs=8,replays=40,cleanup_ok=True,serialized_stable=True,output_exact=True,
            accounting=dict(native=r,offers=o),static={'complete':True},injections=j.injections(r,o,j.account_module().grade),
            window=dict(expected_context={'is_green':True,'sm_matching':1},reasons=['unattributed_expected_context'],counts=counts,graph_slots=slots))
        g=j.case_grade(row);self.assertEqual(g['verdict'],'PASS',g)
        slots[0]['node_execution_bounds']['kernel']['min']=39
        self.assertNotEqual(j.case_grade(row)['verdict'],'PASS')
    def test_api_dedup_ambiguous_never_guesses(self):
        r,o=fixture();r['apis'].append(dict(r['apis'][1],correlation_id=12))
        r['external'].append(dict(r['external'][1],correlation_id=12))
        r['activities'].append(dict(r['activities'][0],correlation_id=12));r['activity_records']=2;o[0]['expected']['kernel']=2
        g=j.account_module().grade(r,o,epoch=4,generation=2)
        self.assertIn('submission_without_unique_activity',g['missing'])
if __name__=='__main__':unittest.main()
