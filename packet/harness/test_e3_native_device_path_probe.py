"""CPU safety/receipt mutations. No CUDA execution is claimed by this suite."""
import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import e3_native_device_path_probe as probe
from e3_native_row1_grade import _digest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('device_owner',ROOT/'tools/native12_device_path_owner.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)


def record():
    cases=[dict(slot=s,mode=m,kernel_executed=True,all_state_bytes_exact=True,completion_queried=True)
           for s in (0,2,9) for m in ('copy','self','unaligned')]
    finals=[dict(slot=s,actual_finish_state=True,staged_write_exact=True,state_bytes_exact=True,completion_queried=True) for s in (0,2,9)]
    sections={'alignment':dict(verdict='PASS',cases=cases,finish_state=finals,legacy_rejected=True,
                              methods_unpatched=True,kernels_unpatched=True),
              'backend_negative':dict(verdict='PASS',refused=True,device_differing_bytes=True,
                                     native9_negative_rejected=True,raw=[dict(equal=False,different_bytes=4)])}
    for route in ('dense_dma','sparse_dma'):
        comp=dict(operation='qsa_output_before_moe',equal=True,compared_bytes=100,finite=True,route=route)
        comp['digest']=_digest(comp)
        layer=dict(passed=True,hold_released=True,route=route,bucket_checks=2 if route=='dense_dma' else 1,output_checks=1)
        layer['digest']=_digest(layer,layer=True)
        buckets=[]
        for i in range(2 if route=='dense_dma' else 1):
            b=dict(operation='dense_page_bucket' if route=='dense_dma' else 'sparse_row_bucket',bucket=i,equal=True,compared_bytes=4096,route=route)
            b['digest']=_digest(b);buckets.append(b)
        row=dict(verdict='PASS',records=buckets+[comp],layer_records=[layer],allocator_live=0,failure_records=[])
        for key in ('kernel_executed','actual_helper','host_plan_bound','uva_pointer_verified','finish_state',
                    'finalized_state_bytes_exact','native10_negative_rejected','state_event_queried','copy_event_joined',
                    'operation_drained','operation_released','source_registry_empty','ordinary_tables_unchanged',
                    'methods_unpatched','arena_provider_unpatched'):row[key]=True
        sections[route]=row
    return dict(jit_mount=dict(filesystem='tmpfs',noexec=False,readonly=False),schema=probe.SCHEMA,sections=sections,imports_verified=True,exceptions={},gpu_allocator_peak_reserved=1024,
                mocks_used=False,patched_methods=False,patched_kernels=False,engine_qualified=False,full_row1_qualified=False)


class DeviceProofTests(unittest.TestCase):
    def test_receipt_control(self):self.assertEqual(probe.grade(record())['verdict'],'PASS')
    def test_prelaunch_cpu_claim_is_not_gpu_proof(self):
        r=record();r['sections']['alignment']['cases'][0]['kernel_executed']=False
        self.assertNotEqual(probe.grade(r)['verdict'],'PASS')
    def test_missing_slot_refused(self):
        r=record();r['sections']['alignment']['cases'].pop()
        self.assertNotEqual(probe.grade(r)['verdict'],'PASS')
    def test_finalize_query_required(self):
        r=record();r['sections']['alignment']['finish_state'][0]['completion_queried']=False
        self.assertNotEqual(probe.grade(r)['verdict'],'PASS')
    def test_each_local_closure_fact_required(self):
        for key in ('state_event_queried','operation_drained','operation_released','source_registry_empty',
                    'finalized_state_bytes_exact','native10_negative_rejected','arena_provider_unpatched'):
            with self.subTest(key=key):
                r=record();r['sections']['dense_dma'][key]=False
                self.assertNotEqual(probe.grade(r)['verdict'],'PASS')
    def test_raw_mismatch_is_failure(self):
        r=record();r['sections']['dense_dma']['records'][0]['equal']=False
        self.assertEqual(probe.grade(r)['verdict'],'FAIL')
    def test_layer_tamper_refused(self):
        r=record();r['sections']['sparse_dma']['layer_records'][0]['bucket_bytes']=10
        self.assertEqual(probe.grade(r)['verdict'],'FAIL')
    def test_live_allocator_refused(self):
        r=record();r['sections']['sparse_dma']['allocator_live']=1
        self.assertEqual(probe.grade(r)['verdict'],'FAIL')
    def test_budget_refused(self):
        r=record();r['gpu_allocator_peak_reserved']=2*1024**3+1
        self.assertEqual(probe.grade(r)['verdict'],'FAIL')
    def test_exception_not_hidden(self):
        r=record();r['exceptions']['alignment']={'error_type':'RuntimeError'}
        self.assertEqual(probe.grade(r)['verdict'],'FAIL')
    def test_empty_receipt_not_pass(self):self.assertNotEqual(probe.grade({})['verdict'],'PASS')
    def test_setup_failure_does_not_invent_device_contradictions(self):
        r=record();r['sections']={};r['imports_verified']=False
        r['exceptions']={'setup':{'error_type':'ImportError'}}
        r.pop('gpu_allocator_peak_reserved')
        graded=probe.grade(r)
        self.assertEqual(graded['verdict'],'FAIL')
        self.assertEqual(graded['failures'],['exception:setup'])
        self.assertTrue(set(probe.SECTIONS)<=set(graded['missing']))
    def test_partial_receipt_preserves_observed_failures(self):
        r=record();r['sections'].pop('sparse_dma')
        r['sections']['dense_dma']['allocator_live']=1
        graded=probe.grade(r)
        self.assertIn('dense_dma:source_or_oracle',graded['failures'])
        self.assertFalse(any(x.startswith('sparse_dma:') for x in graded['failures']))
        self.assertIn('sparse_dma',graded['missing'])
    def test_observed_patched_alignment_still_fails(self):
        r=record();r['sections']['alignment']['methods_unpatched']=False
        self.assertIn('alignment_patched',probe.grade(r)['failures'])
    def test_noexec_and_readonly_mounts_rejected(self):
        for key in ('noexec','readonly'):
            row=dict(filesystem='tmpfs',noexec=False,readonly=False);row[key]=True
            with self.subTest(key=key), self.assertRaises(RuntimeError):probe.require_jit_mount(row)
        self.assertTrue(probe.require_jit_mount(dict(filesystem='tmpfs',noexec=False,readonly=False)))
    def test_effective_mount_evidence_required(self):
        r=record();r.pop('jit_mount')
        self.assertIn('jit_mount',probe.grade(r)['missing'])

    def test_query_deadline_not_completion(self):
        with self.assertRaises(TimeoutError):probe.bounded_poll(lambda:False,seconds=0)
    def test_monitor_pid_separation(self):
        rows=owner.parse_apps('123, 600, python3\n456, 9000, <gpu-tenant>\n')
        self.assertEqual(sum(x['used_mib'] for x in rows if x['pid'] in {123}),600)
    def test_monitor_unknown_memory_refuses(self):
        with self.assertRaises(ValueError):owner.parse_apps('123, [N/A], python3\n')
    def test_visible_gpu_exact_mounts_retained_network_none(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'overlay').mkdir();(root/'overlay/a.py').write_text('')
            cmd=owner.docker_command(root,root/'out',{'mounts':[dict(active=True,source='a.py',destination='/actual/a.py')]})
            self.assertIn('NVIDIA_VISIBLE_DEVICES=0',cmd)
            self.assertIn('CUDA_VISIBLE_DEVICES=0',cmd)
            self.assertEqual(cmd[cmd.index('--network')+1],'none')
            self.assertNotIn('--rm',cmd)
            self.assertEqual(cmd[cmd.index('--tmpfs')+1],'/tmp:rw,exec,nosuid,nodev,size=1024m')
            self.assertEqual(cmd[cmd.index('--name')+1],'flashnext-native12-device-path-b')
            self.assertIn('type=bind,src='+str(root/'overlay/a.py')+',dst=/actual/a.py,readonly',cmd)
    def test_budget_constants(self):
        self.assertEqual(owner.CAP_MIB,3072);self.assertEqual(owner.STOP_MIB,2816)
        self.assertEqual(owner.TIMEOUT_S,900)
    def test_no_kernel_interception_or_fake_arena(self):
        tree=ast.parse(Path(probe.__file__).read_text())
        calls=[x for x in ast.walk(tree) if isinstance(x,ast.Call)]
        self.assertFalse(any(isinstance(x.func,ast.Attribute) and x.func.attr in ('settrace','setprofile') for x in calls))
        self.assertFalse(any(isinstance(x.func,ast.Name) and x.func.id in ('exec','eval') for x in calls))
        assignments=[x for x in ast.walk(tree) if isinstance(x,(ast.Assign,ast.AnnAssign))]
        attrs=[]
        for x in assignments:
            for target in x.targets if isinstance(x,ast.Assign) else [x.target]:
                attrs.extend(n.attr for n in ast.walk(target) if isinstance(n,ast.Attribute))
        self.assertNotIn('qsa_get_staging_arena',attrs)
        self.assertNotIn('run_fused_postprocess_align',attrs)
        self.assertNotIn('is_initialized',attrs)
        self.assertNotIn('postprocess_mamba_fused_kernel',attrs)

if __name__=='__main__':unittest.main()
