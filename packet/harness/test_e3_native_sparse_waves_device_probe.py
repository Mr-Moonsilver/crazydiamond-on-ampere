"""CPU mutations for native14 device proof; synthetic receipts, no GPU claim."""
import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import e3_native_sparse_waves_device_probe as probe
import test_e3_native_device_path_probe as prior
from e3_native_row1_grade import _digest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('waves_owner',ROOT/'tools/native14_device_waves_owner.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)


def record():
    r=prior.record();r['schema']=probe.SCHEMA
    for name in ('sparse_fragmented','refill_fragmented'):
        row=copy.deepcopy(r['sections']['sparse_dma'])
        row['fragmented']=True
        row['waves']=[dict(descriptors=n,rows=n,published=0,consumers=0) for n in range(256,4097,256)]
        row['sparse_stats']=dict(sparse_descriptor_high_water=256,sparse_descriptors=4097,sparse_buckets=1)
        b=row['records'][0];b.update(rows=4097,pieces=4097,coverage_rows=4097,coverage_bytes=24*4048*1024,compared_bytes=24*4048*1024);b['digest']=_digest(b)
        r['sections'][name]=row
    aborted=dict(passed=False,aborted='cancelled',hold_released=True)
    aborted['digest']=_digest(aborted,layer=True)
    row=dict(verdict='PASS',allocator_live=0,layer_records=[aborted])
    for k in ('cancelled','no_partial_publish','no_attention_output','queried_drain',
              'source_registry_empty','advance_refused','methods_unpatched'):row[k]=True
    r['sections']['cancel_fragmented']=row
    return r


class ProofTests(unittest.TestCase):
    def test_control(self):self.assertEqual(probe.grade(record())['verdict'],'PASS')
    def test_partial_publish_refused(self):
        r=record();r['sections']['sparse_fragmented']['waves'][0]['published']=1
        self.assertNotEqual(probe.grade(r)['verdict'],'PASS')
    def test_unbounded_wave_refused(self):
        r=record();r['sections']['sparse_fragmented']['sparse_stats']['sparse_descriptor_high_water']=257
        self.assertNotEqual(probe.grade(r)['verdict'],'PASS')
    def test_old_nonfragmented_proof_not_sufficient(self):
        r=record();r['sections']['sparse_fragmented']=copy.deepcopy(r['sections']['sparse_dma'])
        self.assertNotEqual(probe.grade(r)['verdict'],'PASS')
    def test_raw_fragmented_rows_required(self):
        r=record();b=r['sections']['sparse_fragmented']['records'][0]
        b['rows']=1000;b['digest']=_digest(b)
        self.assertNotEqual(probe.grade(r)['verdict'],'PASS')
    def test_output_mismatch_fail(self):
        r=record();r['sections']['refill_fragmented']['records'][-1]['equal']=False
        self.assertEqual(probe.grade(r)['verdict'],'FAIL')
    def test_abort_cannot_claim_pass(self):
        r=record();r['sections']['cancel_fragmented']['layer_records'][0]['passed']=True
        self.assertNotEqual(probe.grade(r)['verdict'],'PASS')
    def test_cancel_requires_queried_drain(self):
        r=record();r['sections']['cancel_fragmented']['queried_drain']=False
        self.assertNotEqual(probe.grade(r)['verdict'],'PASS')
    def test_no_refill_no_pass(self):
        r=record();r['sections'].pop('refill_fragmented')
        self.assertNotEqual(probe.grade(r)['verdict'],'PASS')
    def test_abort_digest_checked(self):
        r=record();r['sections']['cancel_fragmented']['layer_records'][0]['digest']='x'
        self.assertEqual(probe.grade(r)['verdict'],'FAIL')
    def test_retained_actual_gpu_receipt(self):
        path=Path('<hub-dir>/agents/flash-next/fn-main/results/vm<vm-id>/native14-device-waves/probe.json')
        if not path.is_file(): self.skipTest('retained VM receipt unavailable')
        import json,hashlib
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         '27cc1851d19d3eee848e5f2544b0f8e83a06306e82450557675a8d9540669911')
        raw=json.loads(path.read_text())
        self.assertEqual(raw['verdict'],'INCONCLUSIVE')
        self.assertEqual(set(raw['missing']),{'sparse_fragmented:fragmented_raw_coverage','refill_fragmented:fragmented_raw_coverage'})
        self.assertEqual(probe.grade(raw)['verdict'],'PASS')
        for field in ('coverage_rows','coverage_bytes','compared_bytes','pieces'):
            changed=copy.deepcopy(raw);bucket=changed['sections']['sparse_fragmented']['records'][0]
            bucket[field]-=1;bucket['digest']=_digest(bucket)
            self.assertNotEqual(probe.grade(changed)['verdict'],'PASS')

    def test_owner_delta_only_names(self):
        old=(ROOT/'tools/native12_device_path_owner.py').read_text()
        new=(ROOT/'tools/native14_device_waves_owner.py').read_text()
        expected=old.replace('flashnext-native12-device-path-b','flashnext-native14-device-waves').replace('e3_native_device_path_probe.py','e3_native_sparse_waves_device_probe.py').replace("schema='native12-device-owner/1'", "schema='native14-device-owner/1'")
        self.assertEqual(new,expected)
        self.assertEqual(owner.CAP_MIB,3072);self.assertEqual(owner.STOP_MIB,2816)
    def test_no_patch_or_interception(self):
        tree=ast.parse(Path(probe.__file__).read_text())
        methods={'run_fused_postprocess_align','is_initialized','postprocess_mamba_fused_kernel',
                 'qsa_get_staging_arena','advance','fill_sparse_bucket','_fill_sparse_wave'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Attribute) and isinstance(node.ctx,ast.Store):
                self.assertNotIn(node.attr,methods)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                self.assertNotIn(node.func.id,('exec','eval'))


if __name__=='__main__':unittest.main()
