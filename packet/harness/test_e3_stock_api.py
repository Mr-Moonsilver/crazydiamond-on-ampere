"""Source contracts/negative controls; real image import proof is separate.

These tests never provide a replacement vLLM class or claim image execution.
The mandatory real-import result is retained separately by the owner on VM<vm-id>.
"""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT/'harness'))
import audit_native_stock_api as audit
import e3_stock_api_probe as probe


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def contract():return json.loads((ROOT/'harness/e3_stock_api_contract.json').read_text())


class StockApiContractTests(unittest.TestCase):
    def test_each_pristine_source_pin_matches_its_explicit_layer(self):
        c=contract()
        for spec in c['classes']:
            pristine=ROOT/'reference/vllm-stock-ad067db5'/spec['source']
            p=pristine if pristine.is_file() else ROOT/'reference/vllm-src'/spec['source']
            self.assertEqual(sha(p),spec['source_sha256'],spec['class'])

    def test_pristine_and_historic_baseline_are_distinct_not_an_allowlist(self):
        c=contract();self.assertEqual(len(c['derived_baseline_shas']),3)
        for name,digest in c['derived_baseline_shas'].items():
            self.assertEqual(digest,sha(ROOT/'reference/vllm-src'/name))
            self.assertNotEqual(digest,c['source_shas'][name])
            self.assertIsInstance(c['source_shas'][name],str)

    def test_every_method_has_explicit_signature_and_await_contract(self):
        for spec in contract()['classes']:
            for method in spec['methods']:
                self.assertIs(type(method['async']),bool)
                self.assertTrue(method['signature'])
                sigtext=method.get('signature_ast',method.get('source_signature'))
                if sigtext:
                    sig=audit.signature_from_ast(sigtext)
                    self.assertEqual(probe.signature_shape(lambda:None),[])
                    self.assertEqual([p.name for p in sig.parameters.values()],
                                     [p['name'] for p in method['signature']])

    def test_actual_current_frontend_calls_join_the_stock_contract(self):
        source=(ROOT/'patches/s5_engine/native_overlap_control.py').read_text()
        index=audit.Index('patches/s5_engine/native_overlap_control.py',source,
                          audit.contract_classes(contract()))
        index.visit(ast.parse(source))
        rows=[x for x in index.sites if x['stock_role']=='AsyncLLM']
        self.assertTrue(rows)
        self.assertFalse([x for x in rows if 'missing' in x['binding']['status']])
        self.assertTrue(any(x['member']=='is_paused' for x in rows))
        self.assertFalse(any(x['member']=='is_scheduler_paused' for x in rows))

    def test_exact_native6_source_exposes_the_missing_frontend_method(self):
        old=subprocess.check_output(['git','show','777ad4c:patches/s5_engine/native_overlap_control.py'],cwd=ROOT,text=True)
        index=audit.Index('patches/s5_engine/native_overlap_control.py',old,
                          audit.contract_classes(contract()))
        index.visit(ast.parse(old))
        bad=[x for x in index.sites if x['member']=='is_scheduler_paused'
             and x['binding']['status']=='required-stock-member-missing']
        self.assertGreaterEqual(len(bad),3)

    def test_core_scheduler_query_spelling_stays_synchronous_and_distinct(self):
        classes=audit.contract_classes(contract())
        core=audit.methods(classes['EngineCore'])['is_scheduler_paused']
        frontend=audit.methods(classes['AsyncLLM'])['is_paused']
        self.assertFalse(core['async']);self.assertTrue(frontend['async'])
        self.assertIn('is_scheduler_paused',audit.optional(classes['AsyncLLM']))

    def test_wrong_call_signature_refuses_without_invoking_a_stock_object(self):
        text='async def invalid(self):\n await self.engine.is_paused(123)\n'
        index=audit.Index('patches/s5_engine/native_overlap_control.py',text,
                          audit.contract_classes(contract()))
        index.visit(ast.parse(text))
        self.assertTrue(any(x['binding']['status']=='required-call-signature-mismatch'
                            for x in index.sites))

    def test_optional_absence_is_named_with_fallback_reason(self):
        classes=audit.contract_classes(contract())
        item=next(x for x in classes['GPUModelRunner']['optional_absent'] if x['name']=='rank')
        self.assertTrue(item['reason'])
        text='def rank(runner):\n return getattr(runner,"rank",None)\n'
        index=audit.Index('patches/s5_engine/native_qualification.py',text,classes)
        index.visit(ast.parse(text))
        self.assertTrue(any(x['binding']['status']=='reviewed-optional-absence'
                            for x in index.sites))

    def test_real_import_probe_has_no_mock_or_owner_construction(self):
        source=(ROOT/'harness/e3_stock_api_probe.py').read_text();tree=ast.parse(source)
        names=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertIn('importlib.import_module',names)
        self.assertIn('inspect.getattr_static',names)
        self.assertNotIn('cls',names)
        self.assertFalse(any('mock' in n or n.endswith('.cuda') for n in names))
        self.assertIn('cuda.is_initialized',names)

    def test_finite_dynamic_frontend_methods_are_expanded_not_waived(self):
        text='def check(engine):\n for name in ("is_paused","pause_generation","resume_generation"):\n  getattr(engine,name,None)\n'
        index=audit.Index('patches/s5_engine/native_overlap_control.py',text,
                          audit.contract_classes(contract()))
        index.visit(ast.parse(text))
        rows=[x for x in index.sites if x['kind']=='getattr']
        self.assertEqual({x['member'] for x in rows},{'is_paused','pause_generation','resume_generation'})
        self.assertFalse(any(x['binding']['status']=='dynamic-stock-name-unresolved' for x in rows))


if __name__=='__main__':unittest.main()
