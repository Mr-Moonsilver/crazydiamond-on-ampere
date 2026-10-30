"""CPU accounting contracts; actual native query also tested by observer stubs."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import vmid_job8 as j

def census(rows,start=100,end=200):
    return dict(schema='s5ro-activity-census/1',complete=True,start=start,end=end,
        boundary_records=0,invalid_records=0,omitted_records=0,
        matched_records=sum(r['count'] for r in rows),rows=rows)

def row(kind='kernel',count=1,**kw):
    return dict(dict(kind=kind,count=count,context_id=2,stream_id=8,graph_id=0,node_id=0,
        copy_kind=0,name='vm8_sentinel' if kind=='kernel' else ''),**kw)

def fixture(rank=0):
    p=j.plan(rank);intervals=[];rows=[];clock=100
    for _ in range(j.REPLAYS):
        for name in ('h2d','eager_pre','graph','eager_post','d2d','d2h','memset','helper_sentinel'):
            expected=p['graph'] if name=='graph' else {j.FAMILIES[name][1]:p['families'][name]}
            rs=[]
            for kind,n in expected.items():
                if name=='graph':r=row(kind,n,graph_id=11,node_id={'kernel':1,'memcpy':2,'memset':3}[kind])
                else:
                    spec=j.FAMILIES[name];symbol={1:'vm8_pre',2:'vm8_post',3:'vm8_sentinel'}.get(spec[0],'')
                    r=row(kind,n,copy_kind=spec[2],name=symbol)
                rs.append(r)
            intervals.append(dict(family=name,expected=expected,census=census(rs,clock,clock+1)))
            rows+=rs;clock+=2
    pos=census(rows,99,clock);neg=census([row()],clock+1,clock+2)
    parent=census([row(context_id=1,stream_id=9)],clock+3,clock+4)
    whole=census(rows+neg['rows']+parent['rows'],98,clock+5)
    pcounts=j.totals(whole)
    counts={k:0 for k in ('dropped_records_total','buffers_refused','drop_query_errors','parse_errors','callback_errors',
        'unknown_node_records','ambiguous_node_records','kind_mismatch_records','known_wrong_context_records',
        'unknown_activity_records','unknown_resource_events')}
    counts.update(ring_overflows={k:0 for k in ('kernel','copy','aliases','resource','context')},kernel_records=pcounts['kernel'],
       copy_records=pcounts['memcpy']+pcounts['memset'],unattributed_expected_context=sum(p['families'].values())*40+1)
    slot=dict(required=p['graph'],node_execution_bounds={k:dict(min=40,max=40) for k in p['graph']})
    for k,f in [('kernel','kernel_executions'),('memcpy','copy_executions'),('memset','memset_executions')]:slot[f]=p['graph'][k]*40
    return dict(rank_equivalent=rank,plan=p,replays=40,output_exact=True,serialized_stable=True,cleanup_ok=True,
        context_id=2,stream_id=8,primary_context_id=1,static={'complete':True},positive=pos,intervals=intervals,
        negative_unoffered=neg,negative_primary=parent,all_activity=whole,
        window=dict(expected_context={'is_green':True,'sm_matching':1},reasons=['unattributed_expected_context'],
            flags={},counts=counts,graph_slots=[slot]))

class AccountingTests(unittest.TestCase):
    def test_all_four_shapes_positive_and_both_negatives(self):
        for rank in range(4):
            r=fixture(rank);g=j.case_grade(r)
            self.assertEqual(g['verdict'],'PASS',g)
            self.assertEqual(g['families']['eager_pre']['kernel'],3200)
    def test_no_graph_zero_blanket_acceptance(self):
        q=census([row()]);g=j.accounting_grade(q,[],2,8)
        self.assertIn('unassigned_or_duplicate_activity',g['failures'])
        for field,value in [('context_id',3),('stream_id',77),('graph_id',55),('name','foreign_kernel')]:
            r=row(**{field:value});c=census([r]);d=dict(family='helper_sentinel',expected={'kernel':1},census=c)
            self.assertEqual(j.accounting_grade(c,[d],2,8)['verdict'],'FAIL')
    def test_missing_direction_overlap_boundary_and_duplicate(self):
        c=census([row('memcpy',copy_kind=2)]);d=dict(family='h2d',expected={'memcpy':1},census=c)
        self.assertIn('h2d:kind_or_direction',j.accounting_grade(c,[d],2,8)['failures'])
        c=census([row()]);d=dict(family='helper_sentinel',expected={'kernel':1},census=c)
        self.assertEqual(j.accounting_grade(c,[d,d],2,8)['verdict'],'FAIL')
        c['complete']=False;c['boundary_records']=1
        self.assertEqual(j.accounting_grade(c,[d],2,8)['verdict'],'INCONCLUSIVE')
    def test_old_guard_loss_counts_and_exposure_not_relaxed(self):
        for change in ('guard','overflow','multiplicity','negative','omission','schedule'):
            r=fixture()
            if change=='guard':r['window']['reasons']=[]
            elif change=='overflow':r['window']['counts']['ring_overflows']['copy']=1
            elif change=='multiplicity':r['window']['graph_slots'][0]['kernel_executions']-=1
            elif change=='negative':r['negative_primary']['rows'][0]['context_id']=2
            elif change=='omission':r['all_activity']['omitted_records']=1
            elif change=='schedule':r['intervals'].pop()
            self.assertNotEqual(j.case_grade(r)['verdict'],'PASS',change)
    def test_saved_grade_never_trusted_and_native_crash_veto(self):
        raw=dict(cases=[fixture(i) for i in range(4)],regressions={n:{'verdict':'PASS'} for n in ('job1','job2b_flat')})
        self.assertEqual(j.grade(raw)['verdict'],'PASS')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'probe.json').write_text(json.dumps(raw));(p/'monitor.json').write_text(json.dumps(dict(own_samples=1,errors=0,own_peak_mib=500)))
            self.assertEqual(j.finalize(p,139),1)
    def test_queue_boundaries(self):
        p=Path(__file__).resolve().parent;q=p/'runs/vmid_job8.sh'
        subprocess.run(['bash','-n',str(q)],check=True)
        for x in ('runs/job8','flashnext-job8','flock -n 9','540s','--network none','vmid_eager_probe.cu'):
            self.assertIn(x,q.read_text())
        source=(p/'vmid_job8.py').read_text()
        self.assertIn('set_per_process_memory_fraction(.125,0)',source)
        self.assertIn('split.retain_allocator_context()',source)

if __name__=='__main__':unittest.main()
