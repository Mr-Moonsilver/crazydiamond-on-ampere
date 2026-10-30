"""CPU gates for the measured budget, split logs and operator window."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import native30_gate as gate
import e3_native30_grade as grade

HERE=Path(__file__).resolve().parent
class Gates(unittest.TestCase):
    def test_snapshot_headroom_boundary_and_cgroup_refusal(self):
        need=7*1024**3
        self.assertEqual(gate.evaluate(gate.SNAPSHOT_CAP,need,{})['verdict'],'PASS')
        self.assertEqual(gate.evaluate(gate.SNAPSHOT_CAP,need-1,{})['verdict'],'INCONCLUSIVE')
        self.assertEqual(gate.evaluate(gate.SNAPSHOT_CAP,13514409984,{})['verdict'],'PASS')
        self.assertEqual(gate.evaluate(gate.SNAPSHOT_CAP,need,{'memory.max':str(need),'memory.current':'1'})['verdict'],'INCONCLUSIVE')
        with self.assertRaises(ValueError):gate.evaluate(2*1024**3,need,{})
        self.assertGreater(gate.SNAPSHOT_CAP,10*gate.MEASURED_SNAPSHOT_MAX)

    def test_both_time_guards_reject_early_and_late_and_allow_boundaries(self):
        text=(HERE/'runs/q18a_native30_localize_c.draft.sh').read_text()
        body=text.split("<<'PY_SLOT'",1)[1].split("\n",1)[1].split('\nPY_SLOT',1)[0]
        owner=(HERE.parent/'tools/native30_owner_functions.py').read_text()
        fn=next(n for n in ast.parse(owner).body if isinstance(n,ast.FunctionDef) and n.name=='check_slot')
        import time
        def require(ok,reason):
            if not ok:raise AssertionError(reason)
        ns=dict(time=time,require=require);exec(ast.get_source_segment(owner,fn),ns)
        for now,allowed in ((gate.NOT_BEFORE-1,False),(gate.NOT_BEFORE,True),
                            (gate.HARD_END-gate.ACTIVE_AND_CLEANUP,True),
                            (gate.HARD_END-gate.ACTIVE_AND_CLEANUP+1,False)):
            with patch.object(time,'time',return_value=now):
                for action in (lambda:exec(body,{}),ns['check_slot']):
                    if allowed:action()
                    else:
                        with self.assertRaises(AssertionError):action()

    def test_stderr_only_and_split_pairs_are_graded(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'out',Path(d)/'err';rows=[]
            for rank in range(4):
                for call,label in enumerate(('FIRST_ADMISSION','DECODE_STEP')):
                    for event,t in (('begin',1),('end',2)):
                        rows.append(dict(rank=rank,pid=rank,call_id=call,event=event,label=label,monotonic_ns=t))
            fmt=lambda rs:'\n'.join('NATIVE30 '+json.dumps(x) for x in rs)
            a.write_text('');b.write_text(fmt(rows))
            self.assertEqual(grade.grade([a,b],{'verdict':'PASS'})['verdict'],'PASS')
            a.write_text(fmt([x for x in rows if x['event']=='end']))
            b.write_text(fmt([x for x in rows if x['event']=='begin']))
            result=grade.grade([a,b],{'verdict':'PASS'})
            self.assertEqual(result['verdict'],'PASS');self.assertTrue(result['all_begins_paired'])
            self.assertEqual(grade.grade([a,b,Path(d)/'missing'],{'verdict':'PASS'})['verdict'],'INCONCLUSIVE')

    def test_fault_in_stderr_overrides_completed_placement(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'out',Path(d)/'err';a.write_text('')
            rows=[dict(rank=1,pid=7,call_id=5,label='LAYER',event='begin',monotonic_ns=1),dict(rank=1,pid=7,call_id=5,event='exception',monotonic_ns=2)]
            b.write_text('\n'.join('NATIVE30 '+json.dumps(x) for x in rows))
            result=grade.grade([a,b],{'verdict':'PASS'})
            self.assertEqual((result['verdict'],result['failure_class']),('FAIL','ENGINE'))
            self.assertEqual(result['per_rank']['1']['last_unmatched_begin']['call_id'],5)
            a.write_text(b.read_text());self.assertEqual(grade.grade([a,b])['duplicate_receipts'],2)

    def test_c_queue_restores_unchanged_and_uses_both_logs_and_cap(self):
        old=(HERE/'runs/q18a_native30_localize_b.draft.sh').read_text();new=(HERE/'runs/q18a_native30_localize_c.draft.sh').read_text()
        for name in ('stop_engine','restore_launcher','restore_defaults','write_cleanup'):
            body=lambda s:s.split(name+'() {',1)[1].split('\n}\n',1)[0]
            self.assertEqual(body(old),body(new))
        self.assertIn('--log "$Q/engine-stream.log" --log "$Q/engine-stream.err"',new)
        self.assertIn('record=host_receipt(cap)',new)
        self.assertIn('cap==512*1024**2',(HERE/'runs/q18a_native30_localize_c_declare.sh').read_text())

if __name__=='__main__':unittest.main()
