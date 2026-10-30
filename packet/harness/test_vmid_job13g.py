"""CPU fixtures for VM13g owner discovery; actual vLLM class loading is VM-only."""
import copy
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'patches/s5_engine'),str(ROOT/'harness')]
import model_hooks
import native_startup
import test_e3_lane_private as lp
import test_vmid_job13f as prior
import vmid_job13g as job


class ModuleLeaf:
    def __init__(self):self.module_initialized=True


def pinned_module_double():
    """Only imported classes are doubled; use real pinned constructor source."""
    module=types.ModuleType('vllm.models.qwen4_exp.nvidia.model')
    module.__file__=str(ROOT/'reference/vllm-src/models/qwen4_exp/nvidia/model.py')
    def forbidden_constructor(self,*a,**k):raise AssertionError('no model constructor allowed')
    def forbidden_forward(self,*a,**k):raise AssertionError('no model arithmetic allowed')
    for name in ('Qwen4ExpModel','Qwen4ExpForCausalLM','Qwen4ExpForConditionalGeneration'):
        setattr(module,name,type(name,(),{'__module__':module.__name__,
            '__init__':forbidden_constructor,'forward':forbidden_forward}))
    return module


class OwnershipConstructorTests(unittest.TestCase):
    def build(self):
        prior._PRODUCER['bridge']=None
        torch=prior.cpu_torch();torch.nn=types.SimpleNamespace(Module=ModuleLeaf)
        workspace=lp.FakeWorkspaceModule(lanes=2,torch=torch)
        module=pinned_module_double()
        original=job.importlib.import_module
        def load(name,*a,**k):
            return module if name==module.__name__ else original(name,*a,**k)
        with patch.object(job.importlib,'import_module',side_effect=load):
            return job.build_bindings(torch,workspace_module=workspace,
                prefill_stream=lp.FakeStream(700)),module

    def test_real_probe_constructor_resolves_all_consumers_on_a_wrapped_tree(self):
        bundle,module=self.build()
        root,model=bundle['root_model'],bundle['model']
        self.assertIs(root.language_model.model,model)
        self.assertIs(bundle['runner'].model,root)
        self.assertIs(model_hooks.local_text_model(root),model)
        self.assertIs(type(root),module.Qwen4ExpForConditionalGeneration)
        self.assertIs(type(root.language_model),module.Qwen4ExpForCausalLM)
        self.assertIs(type(model),module.Qwen4ExpModel)
        self.assertEqual(native_startup._hc_width(model),10240)
        with self.assertRaises(native_startup.NativeStartupError):native_startup._hc_width(root)
        self.assertEqual(job.ownership_grade(bundle['model_ownership'])['verdict'],'PASS')
        self.assertEqual([x.module for x in bundle['bindings'].owners],list(bundle['attn']))

    def test_all_positive_claims_are_required(self):
        bundle,_=self.build();positive=bundle['model_ownership']
        for key,value in positive.items():
            if value is True:
                bad=copy.deepcopy(positive);bad[key]=False
                self.assertEqual(job.ownership_grade(bad)['verdict'],'FAIL',key)
                bad.pop(key)
                self.assertEqual(job.ownership_grade(bad)['verdict'],'INCONCLUSIVE',key)

    def test_width_mismatch_is_a_failure(self):
        bundle,_=self.build();proof=bundle['model_ownership'];proof['hc_width']+=1
        self.assertEqual(job.ownership_grade(proof)['verdict'],'FAIL')

    def test_missing_proof_is_never_a_pass(self):
        self.assertEqual(job.ownership_grade(None)['verdict'],'INCONCLUSIVE')
        grade=job.case_grade({'failures':[],'missing':[]})
        self.assertNotEqual(grade['verdict'],'PASS')
        self.assertIn('model_ownership:proof_missing',grade['missing'])

    def test_real_class_module_is_checked_before_constructing_shells(self):
        module=pinned_module_double();module.Qwen4ExpModel.__module__='foreign'
        torch=types.SimpleNamespace(nn=types.SimpleNamespace(Module=ModuleLeaf))
        with patch.object(job.importlib,'import_module',return_value=module):
            with self.assertRaisesRegex(RuntimeError,'model_class_module'):
                job.pinned_owner_shells(torch,types.SimpleNamespace())

    def test_generated_native_methods_do_not_replace_stock_forward(self):
        bundle,module=self.build()
        self.assertIn('_flash_e3_forward_prefix',vars(bundle['model']))
        self.assertNotIn('forward',vars(bundle['model']))
        self.assertTrue(bundle['model_ownership']['generated_methods_exact_inverse'])
        with self.assertRaisesRegex(AssertionError,'no model arithmetic'):
            module.Qwen4ExpModel.forward(bundle['model'])

    def test_vm_contract_has_fraction_sidecar_fresh_names_and_no_engine(self):
        text=(ROOT/'harness/runs/vmid_job13g.sh').read_text()
        for word in ('/opt/flashnext/runs/job13g','flashnext-job13g','flashnext-job13g-build',
                     '/opt/flashnext/.job.lock','PYTORCH_CUDA_ALLOC_CONF'):
            self.assertIn(word,text)
        self.assertIn('set_per_process_memory_fraction(.125, 0)',
                      (ROOT/'harness/vmid_job13g.py').read_text())


if __name__=='__main__':unittest.main()
