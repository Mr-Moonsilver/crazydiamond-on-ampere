"""Observability-only CPU contracts; no GPU or performance qualification."""
import ast
import types
import unittest
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'patches/s5_engine'),str(ROOT/'harness')]
from native_measurement import ControlPathCounters,control_path_delta
import qsa_operation
import test_e3_sparse_history_dma as source
import test_e3_native_measurement as fixtures


class Tests(unittest.TestCase):
    def test_nested_timing_is_exclusive_and_preserves_exceptions(self):
        times=iter((0,10,30,50))
        c=ControlPathCounters(7,clock=lambda:next(times))
        with c.span('prepare_ns'):
            with c.span('serialize_ns'):pass
        self.assertEqual(c.values['prepare_ns'],30)
        self.assertEqual(c.values['serialize_ns'],20)
        times=iter((100,110))
        c.clock=lambda:next(times)
        with self.assertRaisesRegex(ValueError,'original'):
            with c.span('validate_ns'):raise ValueError('original')
        self.assertEqual(c.values['validate_ns'],10)
        self.assertEqual(c._stack,[])
    def test_snapshot_delta_has_no_readiness_authority(self):
        c=ControlPathCounters(7);a=c.snapshot()
        c.values.update(prepare_ns=13,poll_ns=5,validate_ns=9,serialize_ns=7,ops_count=2)
        b=c.snapshot();r=control_path_delta(a,b)
        self.assertTrue(r['available']);self.assertEqual(r['ops_count'],2)
        self.assertEqual(sum(r[k] for k in ('prepare_ns','poll_ns','validate_ns','serialize_ns')),34)
        for bad in (None,dict(a,engine_epoch=8),dict(a,poll_ns=True),dict(a,ops_count=3)):
            self.assertIs(control_path_delta(bad,b)['available'],False)
        self.assertNotIn('enabled',r);self.assertNotIn('native_ready',r)
    def test_real_generated_guard_logs_shape_before_same_failure(self):
        source._GeneratedHarness.setUpClass()
        for name,route in (('_flash_s5_staged_operation','dense_staged'),('_flash_s5_sparse_staged_operation','sparse_staged')):
            tree=ast.Module(body=[source._function_node(source._GeneratedHarness.generated_qsa,name)],type_ignores=[])
            ns={'torch':None};exec(compile(ast.fix_missing_locations(tree),'<actual-generated-helper>','exec'),ns)
            operation=object.__new__(qsa_operation.QsaPrefillOperation)
            operation.transaction=types.SimpleNamespace(txn_seq=99);operation.stats={}
            with self.assertLogs(qsa_operation.__name__,level='INFO') as captured:
                with self.assertRaisesRegex(RuntimeError,'requires the staged dense row threshold'):
                    next(ns[name](operation,types.SimpleNamespace(_qsa_kv_offload=True,_qsa_stage_min_rows=64),None,None,None,None,32,True,None))
            self.assertIn('rows=32 min_rows=64 route='+route+' chunk_index=99',captured.output[0])
    def test_actual_driver_counts_commands_and_keeps_source_census(self):
        s=fixtures.make_long_stack(tokens=65,layers=4,chunk=64)
        s.drive(lambda:s.driver._split_ready)
        begin=s.driver.control_path.snapshot()
        start=int(s.driver._monotonic_ns());s.finish_chunk();end=int(s.driver._monotonic_ns())+1
        evidence=s.driver.overlap_evidence(start,end)
        cost=control_path_delta(begin,evidence['control_path'])
        self.assertTrue(cost['available']);self.assertGreater(cost['ops_count'],0)
        for name in ('prepare_ns','poll_ns','validate_ns','serialize_ns'):self.assertGreater(cost[name],0)
        self.assertEqual(s.controller.commits,1)
        self.assertIsNone(s.driver.retained_row())


if __name__=='__main__':unittest.main()
