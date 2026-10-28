"""E3 row1 backend ownership/protocol provenance; tensor execution is VM CPU proof."""
import ast
import hashlib
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
import row1_exactness as rx
import make_s5_overlap as maker
import decode_probe
import s5_startup
REQUIRED=('snapshot','equal','nbytes','differing_bytes','finite','gather_rows','gather_reference')
def methods(path, name):
 tree=ast.parse(path.read_text());c=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==name)
 return {x.name:x for x in c.body if isinstance(x,ast.FunctionDef)}
class Row1BackendOwnership(unittest.TestCase):
 def test_explicit_e3_inventory_and_provider_owner(self):
  self.assertIn('row1_exactness',maker.E3_MODULES)
  self.assertIn('row1_exactness',decode_probe.GENERATED_SOURCES)
  backend,qualifier=s5_startup.row1_qualification_types()
  self.assertIs(backend,rx.TorchRow1Backend)
  self.assertTrue(all(callable(getattr(backend,n,None)) for n in REQUIRED))
 def test_exact_vm14b_reference_method_bodies(self):
  old=methods(ROOT/'patches/s5_engine/exactness.py','TorchExactBackend')
  new=methods(ROOT/'patches/s5_engine/row1_exactness.py','TorchRow1Backend')
  self.assertEqual(set(new),set(REQUIRED)|{'__init__'})
  for n in new:
   self.assertEqual(ast.dump(old[n],include_attributes=False),ast.dump(new[n],include_attributes=False),n)
 def test_frozen_e2_counterexample_is_really_missing_the_method(self):
  p=ROOT/'harness/fixtures/frozen_e2_exactness.py'
  self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'baf79d18af1b1c97f5b61472c7bac93d9d937d9f72adcbe517212718c1f10545')
  self.assertNotIn('gather_rows',methods(p,'TorchExactBackend'))
 def test_packaged_helper_has_only_the_explicit_e3_owner(self):
  s=maker.package((ROOT/'patches/s5_engine/s5_startup.py').read_text())
  self.assertIn('from .row1_exactness import TorchRow1Backend',s)
  tree=ast.parse(s);helper=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='row1_qualification_types')
  self.assertNotIn('flash_e2',ast.unparse(helper))
 def test_provider_constructs_the_same_backend_it_returned(self):
  tree=ast.parse((ROOT/'patches/s5_engine/s5_startup.py').read_text())
  factory=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='row1_factory')
  self.assertIn('backend=TorchRow1Backend()',ast.unparse(factory))
 def test_backend_import_constructs_no_owners(self):
  tree=ast.parse((ROOT/'patches/s5_engine/row1_exactness.py').read_text())
  self.assertFalse(any(isinstance(x,(ast.Import,ast.ImportFrom)) for x in tree.body))
  self.assertFalse(any(isinstance(x,ast.Call) for x in tree.body))
if __name__=='__main__':unittest.main()
