"""Execute the actual VM14 operation setup with real modules, CPU devices only."""
import ast
import copy
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'patches/s5_engine'),str(ROOT/'harness')]
import exactness
import test_e3_native_source_oracle as source
import test_vmid_job14 as fixture
import vmid_job14 as old
import vmid_job14b as job

def flat_green_record():
    record=fixture.green_record();record['schema']=job.SCHEMA
    for case in record['cases']:
        for layer in case['layers']:
            real=[]
            for row in layer['row1_records']:
                fields=dict(row);identity=job.ro.Row1Identity(**fields.pop('identity'))
                fields.update(compact_start=None,coverage_rows=fields['rows'],
                              coverage_bytes=fields['compared_bytes'],observed_ns=1)
                real.append(job.ro.Row1Comparison(identity=identity,**fields).as_dict())
            layer['row1_records']=real
    return record


class OperationSetupTests(unittest.TestCase):
    def test_actual_operation_backend_constructor_and_source_begin(self):
        names=tuple('qsa.%d'%i for i in range(job.LAYERS))
        tensors=[source.T((job.MAP_PAGES,1,job.PAGE,job.ns.ROW_WIDTH),
                 device='cpu',dtype='torch.bfloat16',pinned=True) for _ in names]
        runner=job.NativeRunner(names,tensors,[None]*len(names))
        device=source.sparse_test.FakeSparseDevice()
        probe=types.SimpleNamespace(runner=runner,device=device,route_generation=1)
        lane=job.make_lane(2)
        dispatcher,binding,_,_=job.build_binding(probe,job.make_transaction(),2,lane)
        fake_torch=types.ModuleType('torch')
        try:
            # The old submitted probe fails at precisely this source call.
            with patch.dict(sys.modules,{'torch':fake_torch}):
                with self.assertRaisesRegex(AttributeError,'TorchExactBackend'):
                    old._make_operation(probe,binding,lane)
                operation=job._make_operation(probe,binding,lane)
            self.assertIs(type(operation),job.qo.QsaPrefillOperation)
            self.assertIs(type(operation.qualification.backend),exactness.TorchExactBackend)
            self.assertIs(operation.qualification.backend.torch,fake_torch)
            self.assertIs(operation.transaction,binding.transaction)
            self.assertEqual(operation.state_epoch,binding.state_epoch)
            name=job._begin(probe,operation,binding,0)
            run=operation._layers[name]
            self.assertIs(run.source,binding.sources[name])
            self.assertIs(run.sparse_hold,binding.holds[name])
            operation.cancel('cpu-setup-smoke')
            device.ready_all()
            self.assertTrue(operation.drain())
            operation.release()
            self.assertTrue(binding.holds[name].released)
            self.assertTrue(operation.qualification.store.layer_records()[0].aborted)
            self.assertEqual(dispatcher.source_allocator.inventory()['live'],job.LAYERS-1)
        finally:
            job.close_binding(dispatcher,binding)
        self.assertEqual(dispatcher.source_allocator.inventory()['live'],0)

    def test_same_frozen_work_and_grade_with_fresh_identity(self):
        record=flat_green_record()
        self.assertEqual(job.grade(record)['verdict'],'PASS')
        record['cases'][0]['layers'][0]['row1_records'][0]['equal']=False
        self.assertEqual(job.grade(record)['verdict'],'FAIL')
        self.assertEqual(job.geometry(),old.geometry())
        self.assertEqual(job.expected_pins(),old.expected_pins())

    def test_actual_producer_flat_records_reach_runtime_collector_and_grader(self):
        record=flat_green_record()
        row=record['cases'][0]['layers'][0]['row1_records'][0]
        identity=job.ro.Row1Identity(**{key:row[key] for key in job.ro.Row1Identity.__dataclass_fields__})
        fields={key:row[key] for key in job.ro.Row1Comparison.__dataclass_fields__ if key!='identity'}
        comparison=job.ro.Row1Comparison(identity=identity,**fields)
        store=job.ro.Row1RecordStore(capacity=8);store.append_comparison(comparison)
        records,layers,failures,buckets,outputs=job.collect_row1(store)
        self.assertEqual(records,[comparison.as_dict()])
        self.assertEqual(buckets,records);self.assertEqual(outputs,[])
        self.assertNotIn('identity',records[0])
        nested=fixture.green_record();nested['schema']=job.SCHEMA
        self.assertEqual(job.grade(nested)['verdict'],'INCONCLUSIVE')
        self.assertEqual(job.grade(record)['verdict'],'PASS')

    def test_compute_geometry_and_generated_call_path_remain_unchanged(self):
        def functions(module):
            text=Path(module.__file__).read_text().replace('job14b','job14').replace('JOB14B','JOB14')
            return {n.name:ast.dump(n) for n in ast.parse(text).body
                    if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        previous,fixed=functions(old),functions(job)
        changed={name for name in set(previous)|set(fixed) if previous.get(name)!=fixed.get(name)}
        self.assertEqual(changed,{'_make_operation','run_layer','run_cancellation','collect_row1','grade','load_modules','install_ops_module'})
        fn=next(n for n in ast.parse(Path(job.__file__).read_text()).body
                if isinstance(n,ast.FunctionDef) and n.name=='run_cancellation')
        assignment=next(n for n in ast.walk(fn) if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=='allocator_refs_zero' for t in n.targets))
        close_line=min(n.lineno for n in ast.walk(fn) if isinstance(n,ast.Call)
                       and isinstance(n.func,ast.Name) and n.func.id=='close_binding')
        self.assertGreater(assignment.lineno,close_line)

    def test_actual_generated_helper_imports_and_calls_the_aliased_ops_module(self):
        class Buffer:
            shape=(51,1,4048,512);device='cuda:0'
            def __getitem__(self,index):return self
        calls=[];marker=object()
        def attention(*args,**kwargs):
            calls.append('attention');yield marker
        module=types.ModuleType('real_generated_ops_name')
        module.flash_s5_sparse_arena_attention=attention
        module.flash_s5_arena_attention=attention
        module.flash_e2_arena_attention=lambda *a,**k:None
        module.qsa_get_staging_arena=lambda *a,**k:types.SimpleNamespace(shape=(24,1,4048,512))
        canonical='models.qwen4_exp.nvidia.ops.qsa'
        with patch.dict(sys.modules):
            self.assertIs(job.install_ops_module(module),module)
            self.assertIs(sys.modules[canonical],module)
            self.assertEqual(module.__name__,'real_generated_ops_name')
            class Operation:
                def __getattr__(self,name):
                    def record(*args,**kwargs):calls.append(name)
                    return record
            text=job.operation_source.generate_layer(job.runner_hooks.qsa_staged(job.LAYER_SOURCE.read_text()))
            namespace={'__name__':'models.qwen4_exp.nvidia.qsa',
                       '__package__':'models.qwen4_exp.nvidia',
                       'torch':types.SimpleNamespace(int64='int64',arange=lambda *a,**k:range(a[0],a[1]))}
            exec(compile(job.helper_definitions(text,'qsa_native.py'),'qsa_native.py','exec'),namespace)
            metadata=types.SimpleNamespace(_qsa_host_meta=((0,100),(job.TOKENS,)),
                query_start_loc=types.SimpleNamespace(shape=(2,)),
                seq_lens=types.SimpleNamespace(shape=(1,)),
                block_table=types.SimpleNamespace())
            class Table:
                shape=(1,50)
                def __getitem__(self,index):return self
            metadata.block_table=Table()
            owner=types.SimpleNamespace(_qsa_kv_offload=True,_qsa_stage_min_rows=1,
                     _qsa_stage_arena_bytes=job.ARENA_ROWS*1024,layer_name='qsa.0')
            helper=namespace[job.operation_source.SPARSE_LAYER_HELPER]
            frame=helper(Operation(),owner,Buffer(),Buffer(),Buffer(),metadata,100,True,Buffer())
            self.assertIs(next(frame),marker)
            with self.assertRaises(StopIteration):next(frame)
        self.assertEqual(calls,['bind_plan','bind_sparse_context','bind_row1_uva',
                                'bind_row1_reference','attention','row1_output'])

    def test_new_queue_is_owned_and_syntax_checked(self):
        path=ROOT/'harness/runs/vmid_job14b.sh';text=path.read_text()
        for value in ('runs/job14b','flashnext-job14b','vmid_job14b.py','.job.lock','--network none'):
            self.assertIn(value,text)
        self.assertIn('flashnext-job14b',job.s.OWNED_CONTAINERS)
        self.assertIn('flashnext-job14b',job.s.LONG_RUN_CONTAINERS)
        subprocess.run(['bash','-n',str(path)],check=True)

if __name__=='__main__':unittest.main()
