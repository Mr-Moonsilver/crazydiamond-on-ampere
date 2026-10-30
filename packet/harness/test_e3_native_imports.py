"""Namespace ownership tests; actual-image provider import proof is separate."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
import make_s5_overlap as m
import make_e2_arena as e2


class ImportOwnershipTests(unittest.TestCase):
    def test_e2_owned_from_import_is_qualified_after_packaging(self):
        source='def provider():\n from exactness import TorchExactBackend\n return TorchExactBackend\n'
        generated=m.package(source);self.assertIn('from ..flash_e2.exactness import',generated)
        self.assertNotIn('from .exactness import',generated)

    def test_bare_e2_import_and_mixed_import_keep_owner_and_alias(self):
        s=m.package('def f():\n import os, exactness as ex, row1_oracle as ro\n')
        self.assertIn('from ..flash_e2 import exactness as ex',s)
        self.assertIn('from . import row1_oracle as ro',s)
        self.assertIn('import os',s);ast.parse(s)

    def test_e3_peers_stay_same_package(self):
        self.assertIn('from .row1_oracle import',m.package('from row1_oracle import Row1Qualifier\n'))

    def test_provider_uses_real_import_helper_after_existing_fences(self):
        tree=ast.parse((ROOT/'patches/s5_engine/s5_startup.py').read_text())
        helper=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='row1_qualification_types')
        self.assertEqual([type(x).__name__ for x in helper.body[1:]],['ImportFrom','ImportFrom','Return'])
        install=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='install_native_lane')
        provider=next(x for x in ast.walk(install) if isinstance(x,ast.FunctionDef) and x.name=='row1_provider')
        calls=[(x.lineno,ast.unparse(x.func)) for x in ast.walk(provider) if isinstance(x,ast.Call)]
        point=next(line for line,name in calls if name=='row1_qualification_types')
        self.assertTrue(all(line<point for line,name in calls if name in ('controller.drain','native.drain_prefill_streams')))

    def test_forward_context_import_is_separate_from_unweakened_gpu_guard(self):
        tree=ast.parse((ROOT/'patches/s5_engine/native_resources.py').read_text())
        functions={x.name:x for x in tree.body if isinstance(x,ast.FunctionDef)}
        pure=ast.unparse(functions['_import_forward_context'])
        guard=ast.unparse(functions['_forward_context_module'])
        self.assertIn("importlib.import_module('vllm.forward_context')",pure)
        self.assertNotIn('QSA_REQ_UID',pure)
        self.assertIn("hasattr(module, 'QSA_REQ_UID')",guard)
        self.assertIn("hasattr(module, '_forward_context')",guard)
        self.assertIn('raise NativeResourcesError',guard)

    def test_isolated_real_generated_packages_reject_the_old_bare_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            for rel in ('candidate','candidate/v1','candidate/v1/flash_e2','candidate/v1/flash_e3'):
                d=base/rel;d.mkdir(exist_ok=True);(d/'__init__.py').write_text('')
            for name in m.E3_MODULES:
                (base/'candidate/v1/flash_e3'/(name+'.py')).write_text(m.package((ROOT/'patches/s5_engine'/(name+'.py')).read_text()))
            for name in e2.RUNTIME:
                (base/'candidate/v1/flash_e2'/(name+'.py')).write_text(e2.package((ROOT/'patches/s5_engine'/(name+'.py')).read_text()))
            frozen=ROOT/'harness/fixtures/frozen_e2_exactness.py'
            (base/'candidate/v1/flash_e2/exactness.py').write_bytes(frozen.read_bytes())
            program='''import importlib,sys
sys.path.insert(0,sys.argv[1])
s=importlib.import_module('candidate.v1.flash_e3.s5_startup')
b,q=s.row1_qualification_types()
assert b.__module__=='candidate.v1.flash_e3.row1_exactness'
assert q.__module__=='candidate.v1.flash_e3.row1_oracle'
assert all(callable(getattr(b,n,None)) for n in ('snapshot','equal','nbytes','differing_bytes','finite','gather_rows','gather_reference'))
old=importlib.import_module('candidate.v1.flash_e2.exactness').TorchExactBackend
assert not hasattr(old,'gather_rows')
try:exec('from exactness import TorchExactBackend',dict(s.__dict__))
except ModuleNotFoundError as e:assert e.name=='exactness'
else:raise AssertionError('bare source path leaked')
f=importlib.import_module('candidate.v1.flash_e3.native_qualification_frontend')
assert f._peer_module('native_qualification').__name__=='candidate.v1.flash_e3.native_qualification'
try:f._peer_module('json')
except ModuleNotFoundError as e:assert e.name=='candidate.v1.flash_e3.json'
else:raise AssertionError('packaged import error masked by bare fallback')
'''
            p=subprocess.run([sys.executable,'-I','-B','-c',program,str(base)],capture_output=True,text=True,timeout=30)
            self.assertEqual(p.returncode,0,p.stderr)


if __name__=='__main__':unittest.main()
