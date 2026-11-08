"""CPU packaging gates. Fixtures retained; no launcher execution or engine access.

Leaves: overlay bytes/manifests are tiny synthetic files. Runner tests execute
only the real require/sha/manifest/inventory/verify_sources functions; the
post-pin native_readiness.verify_finalized_package is a no-op leaf. Actual
launcher identity/inventory/manifest guards and runner derivation are real.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

TOOLS=Path(__file__).resolve().parent
sys.path.insert(0,str(TOOLS))
import build_inwait as iw

b=iw.b
LAUNCHER='run-flash-s5-test.sh'


class LauncherOverlay(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='inwait-launcher-'))
        self.parent=self.root/'parent';self.old=self.parent/'native30_overlay';self.new=self.root/'new'
        self.old.mkdir(parents=True);self.new.mkdir()
        for name,old,new in ((LAUNCHER,'#!/bin/bash\ns5_log=(--log-opt max-file=4)\n',
                              '#!/bin/bash\ns5_log=(--log-opt max-file=4)\ns5=(-v "$OVERLAY/ops_qsa_indexer.py:/vllm/ops/qsa_indexer.py:ro")\n'),
                             ('module.py','def run(): return 1\n','def run(): return 2\n')):
            (self.old/name).write_text(old);(self.new/name).write_text(new)
        b.manifest(self.old);b.manifest(self.new)

    def spec(self):
        names=[LAUNCHER,'module.py']
        return dict(engine_changed=names,overlay_definitions={n:iw.overlay_declaration(n,self.old/n,self.new/n) for n in names})

    def proof(self,spec):
        with patch.object(b,'PARENT',self.parent),patch.object(iw,'NEW_OVERLAY',()):
            return iw.overlay_level(spec,self.new,b.verify(self.old))

    def test_allowlisted_shell_has_exact_sha_pins_and_python_keeps_ast(self):
        proof=self.proof(self.spec())
        self.assertEqual(proof['definitions'][LAUNCHER],dict(format='non-ast-sha256/1',
                         old_sha256=b.sha(self.old/LAUNCHER),new_sha256=b.sha(self.new/LAUNCHER),changed=['non-ast:sha256']))
        self.assertEqual(proof['definitions']['module.py']['changed'],['run'])
        with self.assertRaises(SyntaxError):b.changed_definitions((self.old/LAUNCHER).read_text(),(self.new/LAUNCHER).read_text())

    def test_launcher_pin_mismatch_refuses(self):
        for pin in ('old_sha256','new_sha256'):
            spec=self.spec();spec['overlay_definitions'][LAUNCHER][pin]='0'*64
            with self.subTest(pin=pin),self.assertRaisesRegex(AssertionError,'non-AST overlay pin differs'):
                self.proof(spec)

    def test_other_non_python_files_refuse_even_when_python_parseable(self):
        for name in ('other.sh','nested/run-flash-s5-test.sh','notes.bin'):
            old=self.old/name;new=self.new/name
            old.parent.mkdir(parents=True,exist_ok=True);new.parent.mkdir(parents=True,exist_ok=True)
            old.write_text('x = 1\n');new.write_text('x = 2\n')
            with self.subTest(name=name),self.assertRaisesRegex(AssertionError,'not allowlisted'):
                iw.overlay_declaration(name,old,new)
        b.manifest(self.old);b.manifest(self.new)
        spec=self.spec()
        for name in ('other.sh','nested/run-flash-s5-test.sh','notes.bin'):
            spec['engine_changed'].append(name)
            spec['overlay_definitions'][name]=dict(format='non-ast-sha256/1',old_sha256=b.sha(self.old/name),new_sha256=b.sha(self.new/name))
        with self.assertRaisesRegex(AssertionError,'not allowlisted'):self.proof(spec)

    def derived(self):
        sources=b.verify(b.PARENT/'native30_overlay')
        sources[LAUNCHER]=b.sha(self.new/LAUNCHER)
        owner,diff=b.derive_runner('q18a-codex-launcher-proof',b.PARENT_TESTS,[],'1'*64,'2'*64,sources,False)
        return owner,sources,diff

    def test_runner_derives_both_pins_and_keeps_the_real_guard_body(self):
        owner,sources,diff=self.derived();tree=ast.parse(owner)
        assignments={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('LAUNCHER','EXPECTED_SOURCES')}
        self.assertEqual(assignments['LAUNCHER'],sources[LAUNCHER])
        self.assertEqual(assignments['EXPECTED_SOURCES'],sources)
        original=next(n for n in ast.parse(b.PARENT_RUNNER.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='verify_sources')
        updated=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_sources')
        self.assertEqual(ast.dump(original),ast.dump(updated))
        self.assertTrue(any(line.startswith('+ LAUNCHER=') for line in diff))

    def test_real_runner_refuses_launcher_mismatch_even_with_fresh_manifest(self):
        owner,_,_=self.derived()
        names={'require','sha','manifest','inventory','verify_sources'}
        body=[n for n in ast.parse(owner).body if isinstance(n,ast.FunctionDef) and n.name in names]
        class Refused(Exception):pass
        ns=dict(Path=Path,hashlib=hashlib,json=json,re=re,Refused=Refused)
        exec(compile(ast.Module(body=body,type_ignores=[]),'real-derived-runner-guards','exec'),ns)
        target=self.root/'runner-check';target.mkdir();(target/'e3').mkdir()
        (target/LAUNCHER).write_bytes((self.new/LAUNCHER).read_bytes())
        (target/'e3/native-implementation-closure.json').write_text(json.dumps(dict(review_sha256='2'*64,generated_source_shas={})))
        b.manifest(target)
        ns.update(EXPECTED_SOURCES=b.verify(target),LAUNCHER=b.sha(self.new/LAUNCHER),REVIEW='2'*64)
        readiness=ModuleType('native_readiness');readiness.verify_finalized_package=lambda *args:None
        with patch.dict(sys.modules,{'native_readiness':readiness}):
            ns['verify_sources'](target)
            (target/LAUNCHER).write_text('#!/bin/bash\necho changed\n')
            b.manifest(target);ns['EXPECTED_SOURCES']=b.verify(target)
            with self.assertRaisesRegex(Refused,'trusted-launcher-pin'):ns['verify_sources'](target)


if __name__=='__main__':unittest.main()
