"""Native15 device-proof boundary receipts/source checks; CPU-only."""
import ast
import copy
import inspect
from pathlib import Path
import unittest
import e3_native_staged_plan_device_probe as p
import e3_native_sparse_waves_device_probe as old
import test_e3_native_device_path_probe as fixture
from e3_native_row1_grade import _digest


def record():
    original=fixture.record()
    r={k:copy.deepcopy(v) for k,v in original.items() if k!='sections'}
    r.update(schema=p.SCHEMA,minimum_rows=64,threshold_negatives=[1,7,32,63],cases={})
    for rows in p.CASE_ROWS:
        for route in ('dense_dma','sparse_dma'):
            row=copy.deepcopy(original['sections'][route]);row['geometry']={'rows':rows}
            output=row['records'][-1];output['rows']=rows;output['digest']=_digest(output)
            r['cases'][str(rows)+'_'+route]=row
    return r


class Tests(unittest.TestCase):
    def test_positive(self):self.assertEqual(p.grade(record())['verdict'],'PASS')
    def test_copied_device_body_has_only_declared_geometry_and_minimum_changes(self):
        source=inspect.getsource(old.qsa_case).strip()
        expected=source.replace('ROWS','rows').replace('def qsa_case(route, *, fragmented=False, cancel=False):',
            'def qsa_case(route, *, rows, fragmented=False, cancel=False):',1).replace(
            'layer._qsa_stage_min_rows, layer._qsa_stage_arena_bytes = 1,',
            'layer._qsa_stage_min_rows, layer._qsa_stage_arena_bytes = qsa._QSA_STAGE_MIN_ROWS,',1)
        self.assertEqual(inspect.getsource(p.qsa_case).strip(),expected)
    def test_first_case_threshold_attribute_is_an_actual_generated_module_name(self):
        import types
        source=(Path(__file__).resolve().parents[1]/'reference/vllm-src/models/qwen4_exp/nvidia/qsa.py').read_text()
        assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='_QSA_STAGE_MIN_ROWS' for t in n.targets))
        namespace={};exec(compile(ast.Module(body=[assignment],type_ignores=[]),'<pinned-qsa-constant>','exec'),namespace)
        module=types.ModuleType('pinned_qsa_source_constants')
        module.__dict__.update(namespace)
        body=ast.parse(inspect.getsource(p.qsa_case))
        assignment=next(n for n in ast.walk(body) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Tuple) and any(isinstance(a,ast.Attribute)
                and a.attr=='_qsa_stage_min_rows' for a in t.elts) for t in n.targets))
        value=ast.Expression(assignment.value.elts[0])
        self.assertEqual(eval(compile(value,'<actual-probe-first-case-attribute>','eval'),{'qsa':module}),64)
    def test_every_case_required(self):
        for key in record()['cases']:
            r=record();r['cases'].pop(key)
            self.assertNotEqual(p.grade(r)['verdict'],'PASS')
    def test_old_minimum1_is_not_accepted(self):
        r=record();r['minimum_rows']=1
        self.assertEqual(p.grade(r)['verdict'],'FAIL')
    def test_raw_digest_refused(self):
        r=record();r['cases']['64_dense_dma']['records'][0]['compared_bytes']+=1
        self.assertEqual(p.grade(r)['verdict'],'FAIL')
    def test_device_mismatch_is_fail(self):
        r=record();r['cases']['992_sparse_dma']['records'][-1]['equal']=False
        self.assertEqual(p.grade(r)['verdict'],'FAIL')
    def test_output_shape_must_match_query_rows(self):
        r=record();o=r['cases']['967_sparse_dma']['records'][-1];o['rows']=50;o['digest']=_digest(o)
        self.assertNotEqual(p.grade(r)['verdict'],'PASS')
    def test_release_and_state_finalize_are_required(self):
        for key in ('finish_state','finalized_state_bytes_exact','state_event_queried',
                'operation_drained','operation_released','source_registry_empty','ordinary_tables_unchanged'):
            r=record();r['cases']['64_dense_dma'][key]=False
            self.assertNotEqual(p.grade(r)['verdict'],'PASS')
    def test_layer_digest_refused(self):
        r=record();r['cases']['103_dense_dma']['layer_records'][0]['output_checks']=2
        self.assertEqual(p.grade(r)['verdict'],'FAIL')
    def test_peak_limit(self):
        r=record();r['gpu_allocator_peak_reserved']=2*1024**3+1
        self.assertEqual(p.grade(r)['verdict'],'FAIL')
    def test_pending_guard_negatives_cannot_be_omitted(self):
        r=record();r['threshold_negatives'].remove(32)
        self.assertNotEqual(p.grade(r)['verdict'],'PASS')


if __name__=='__main__':unittest.main()
