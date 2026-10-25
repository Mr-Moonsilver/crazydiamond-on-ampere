"""Generator boundary tests; full pinned-source transform when reference exists.

Fixtures are NEW synthetic overlays, never declarations for a running engine.
No GPU, Docker, host writes, or import of the generated vLLM runner.
"""
import ast
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
import make_e2_arena as g


def manifest(folder):
    files=[p for p in sorted(folder.rglob('*')) if p.is_file() and p!=folder/'SHA256SUMS']
    (folder/'SHA256SUMS').write_text(''.join(g.sha(p)+'  '+p.relative_to(folder).as_posix()+'\n' for p in files))
    return g.sha(folder/'SHA256SUMS')


class PackageTests(unittest.TestCase):
    def fixture(self):
        root=Path(tempfile.mkdtemp(prefix='e2-package-',dir=ROOT.parent/"tmp"));base=root/'base';ref=root/'reference'
        for name in g.PINS:
            p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fake '+name+'\n')
        for name in g.REFERENCE_PINS:
            p=ref/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fake '+name+'\n')
        (base/'unchanged.py').write_text('UNCHANGED = True\n')
        (base/'core/untouched.py').write_text('UNCHANGED_CORE = True\n')
        core=manifest(base/'core');whole=manifest(base)
        launch=root/'launcher';launch.write_text('#!/bin/bash\nexit 0\n')
        return root,base,ref,whole,core,launch

    def stubs(self,base,ref):
        stack=ExitStack()
        stack.enter_context(patch.object(g,'PINS',{n:g.sha(base/n) for n in g.PINS}))
        stack.enter_context(patch.object(g,'REFERENCE_PINS',{n:g.sha(ref/n) for n in g.REFERENCE_PINS}))
        for module,name in ((g.arena_attention,'generate'),(g.runner_hooks,'qsa_staged'),
                (g.runner_hooks,'runner_execution'),(g.runner_hooks,'runner_startup'),
                (g.mirror_hooks,'block_pool'),(g.mirror_hooks,'scheduler'),
                (g.mirror_hooks,'scheduler_output'),(g.mirror_hooks,'block_tables'),
                (g.telemetry_hooks,'block_pool'),(g.telemetry_hooks,'scheduler')):
            stack.enter_context(patch.object(module,name,side_effect=lambda s:s+'# transformed\n'))
        stack.enter_context(patch.object(g,'launcher',side_effect=lambda s:s))
        return stack

    def test_complete_manifest_package_import_and_untouched_sources(self):
        root,b,r,w,c,l=self.fixture();before={str(p.relative_to(b)):p.read_bytes() for p in b.rglob('*') if p.is_file()}
        previous_umask=os.umask(0o077)
        try:
            with self.stubs(b,r):result=g.build(b,root/'out',r,w,l,c)
        finally:os.umask(previous_umask)
        out=root/'out';self.assertEqual(result['manifest_sha256'],g.sha(out/'SHA256SUMS'))
        self.assertEqual(before,{str(p.relative_to(b)):p.read_bytes() for p in b.rglob('*') if p.is_file()})
        self.assertEqual((out/'unchanged.py').read_bytes(),before['unchanged.py'])
        self.assertEqual((out/'core/untouched.py').read_bytes(),before['core/untouched.py'])
        self.assertEqual(set(g.manifest(out)),{str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p!=out/'SHA256SUMS'})
        g.manifest(out/'core')
        self.assertEqual((out/'run-flash-e2-test.sh').stat().st_mode&0o777,0o755)
        self.assertEqual(out.stat().st_mode&0o777,0o755)
        for p in out.rglob('*'):
            self.assertEqual(p.stat().st_mode&0o555,0o555 if p.is_dir() else (0o444 | (p.stat().st_mode&0o111)))
        subprocess.run([sys.executable,'-I','-c',
            'import sys;sys.path.insert(0,sys.argv[1]);import e2.startup,e2.exactness;'
            'assert e2.startup.CpuTables is __import__("e2.block_mirror",fromlist=["CpuTables"]).CpuTables',str(out)],check=True)

    def test_declared_root_and_core_hashes_are_independent(self):
        for which in ('root','core'):
            root,b,r,w,c,l=self.fixture()
            with self.stubs(b,r),self.assertRaisesRegex(ValueError,'manifest differs'):
                g.build(b,root/'out',r,'0'*64 if which=='root' else w,l,'0'*64 if which=='core' else c)
            self.assertFalse((root/'out').exists())

    def vision_fixture(self):
        root,b,r,w,c,l=self.fixture()
        # Deliberately add AFTER root/core manifests, matching the real v16.
        v=b/'vision';v.mkdir();(v/'model.py').write_text('# inert vision fixture\n')
        vs=manifest(v)
        return root,b,r,w,c,l,vs

    def test_third_vision_manifest_verified_preserved_and_provenanced(self):
        root,b,r,w,c,l,v=self.vision_fixture()
        before={p.name:p.read_bytes() for p in (b/'vision').iterdir()}
        with self.stubs(b,r):g.build(b,root/'out',r,w,l,c,v)
        out=root/'out'
        self.assertEqual(before,{p.name:p.read_bytes() for p in (out/'vision').iterdir()})
        self.assertEqual(before,{p.name:p.read_bytes() for p in (b/'vision').iterdir()})
        meta=json.loads((out/'E2-GENERATION.json').read_text())
        self.assertEqual(meta['vision_manifest_sha256'],v)
        self.assertEqual(meta['preserved_submanifests'],{'vision/SHA256SUMS':v})
        self.assertEqual(meta['base_files']['vision/model.py'],g.sha(b/'vision/model.py'))
        self.assertIn('vision/SHA256SUMS',g.manifest(out));g.manifest(out/'vision')

    def test_vision_pin_content_and_coverage_fail_before_output(self):
        for defect in ('pin','content','extra','link'):
            root,b,r,w,c,l,v=self.vision_fixture()
            if defect=='pin':v='0'*64
            elif defect=='content':(b/'vision/model.py').write_text('# changed\n')
            elif defect=='extra':(b/'vision/extra.py').write_text('# unmanifested\n')
            else:(b/'vision/link').symlink_to(b/'vision/model.py')
            with self.stubs(b,r),self.subTest(defect=defect),self.assertRaises(ValueError):
                g.build(b,root/'out',r,w,l,c,v)
            self.assertFalse((root/'out').exists())

    def test_unmanifested_or_symlink_entry_rejected_before_output(self):
        for kind in ('file','link'):
            root,b,r,w,c,l=self.fixture()
            if kind=='file':(b/'unexpected.py').write_text('x=1\n')
            else:(b/'link').symlink_to(b/'unchanged.py')
            with self.stubs(b,r),self.assertRaises(ValueError):g.build(b,root/'out',r,w,l,c)
            self.assertFalse((root/'out').exists())

    def test_manifest_paths_and_changed_source_pins_fail_closed(self):
        for defect in ('traversal','duplicate','changed','reference'):
            root,b,r,w,c,l=self.fixture()
            with self.stubs(b,r):
                if defect in ('traversal','duplicate'):
                    text=(b/'SHA256SUMS').read_text()
                    extra=g.sha(l)+'  ../launcher\n' if defect=='traversal' else text.splitlines()[0]+'\n'
                    (b/'SHA256SUMS').write_text(text+extra);w=g.sha(b/'SHA256SUMS')
                elif defect=='changed':
                    (b/'qsa.py').write_text('# drift\n');w=manifest(b)
                else:(r/'config/compilation.py').write_text('# drift\n')
                with self.subTest(defect=defect),self.assertRaises(ValueError):g.build(b,root/'out',r,w,l,c)
            self.assertFalse((root/'out').exists())

    def test_existing_output_or_descendant_never_overwritten(self):
        root,b,r,w,c,l=self.fixture()
        for out in (b,root,b/'new'):
            with self.stubs(b,r),self.assertRaises(ValueError):g.build(b,out,r,w,l,c)

    def test_launcher_mounts_package_and_sources_without_installing(self):
        source=(ROOT/'deploy/run-flash-prod.sh').read_text();new=g.launcher(source)
        import hashlib
        import identity_declare
        self.assertEqual(hashlib.sha256(new.encode()).hexdigest(),identity_declare.E2_LAUNCHER_SHA)
        root=Path(tempfile.mkdtemp(prefix='e2-launcher-',dir=ROOT.parent/"tmp"));p=root/'launcher';p.write_text(new)
        subprocess.run(['bash','-n',str(p)],check=True)
        for name in g.RUNTIME:self.assertIn('e2/'+name+'.py',new)
        for dest in ('flash_e2','core/sched/output.py','worker/gpu/block_table.py'):
            self.assertIn(dest,new)
        self.assertIn('"${runner[@]}" "${e2[@]}"',new)
        with self.assertRaises(ValueError):g.launcher(source+'\n')

    def test_sparse_switch_is_the_only_change_to_q16b_launcher(self):
        # Reverse the one reviewed q16c edit and recover the independently
        # pinned q16b output byte-for-byte (no golden generated file needed).
        import hashlib
        source=(ROOT/'deploy/run-flash-prod.sh').read_text()
        new=g.launcher(source)
        variable='-e VLLM_QSA_KVO_SPARSE_PREFILL="${SPARSE_PREFILL:-1}"'
        self.assertEqual(new.count('VLLM_QSA_KVO_SPARSE_PREFILL='),1)
        self.assertEqual(new.count(variable),1)
        old=new.replace(variable,'-e VLLM_QSA_KVO_SPARSE_PREFILL=1')
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),
                         'd0a5967e58336e8694161b887fcada6103ed4ad270984875c6201ef3b4507491')

    def test_full_pinned_source_transform_and_real_config_enums(self):
        ref=ROOT/'reference/vllm-src'
        mapping={'model_runner_v2.py':'v1/worker/gpu/model_runner.py',
                 'qsa.py':'models/qwen4_exp/nvidia/qsa.py','ops_qsa.py':'models/qwen4_exp/nvidia/ops/qsa.py',
                 'core/block_pool.py':'v1/core/block_pool.py','core/scheduler.py':'v1/core/sched/scheduler.py'}
        if not all((ref/n).is_file() for n in (*mapping.values(),*g.REFERENCE_PINS)):
            self.skipTest('read-only image reference absent; full-source integration NOT verified')
        root=Path(tempfile.mkdtemp(prefix='e2-pinned-package-',dir=ROOT.parent/"tmp"));base=root/'base'
        for name,source in mapping.items():
            p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ref/source).read_bytes())
        c=manifest(base/'core');w=manifest(base)
        result=g.build(base,root/'out',ref,w,ROOT/'deploy/run-flash-prod.sh',c)
        for name in result['generated_files']:
            if name.endswith('.py'):ast.parse((root/'out'/name).read_text())
        runner=(root/'out/model_runner_v2.py').read_text()
        self.assertEqual(runner.count('_install_flash_e2(self, is_profiling)'),1)
        self.assertEqual(runner.count('@execute_with_e2'),1)
        # Execute only enum definitions from the pinned config, not vLLM imports.
        src=(ref/'config/compilation.py').read_text();tree=ast.parse(src)
        selected=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('CompilationMode','CUDAGraphMode')]
        import enum
        ns={'enum':enum}
        exec('from __future__ import annotations\n'+ast.unparse(ast.Module(body=selected,type_ignores=[])),ns)
        self.assertEqual(int(ns['CompilationMode'].NONE),0)
        self.assertIs(ns['CUDAGraphMode'].FULL_DECODE_ONLY.mixed_mode(),ns['CUDAGraphMode'].NONE)
        self.assertIs(ns['CUDAGraphMode'].FULL_DECODE_ONLY.decode_mode(),ns['CUDAGraphMode'].FULL)

    def test_telemetry_matches_reviewed_pool_and_scheduler_delta_is_reversible(self):
        ref=ROOT/'reference/vllm-src'
        if not (ref/'v1/core/block_pool.py').is_file():self.skipTest('image reference absent')
        original=(ref/'v1/core/block_pool.py').read_text()
        self.assertEqual(g.telemetry_hooks.block_pool(original),
                         (ROOT/'patches/ppgrouping/overlay/core/block_pool.py').read_text())
        original=(ref/'v1/core/sched/scheduler.py').read_text()
        changed=g.telemetry_hooks.scheduler(original)
        self.assertEqual(changed.replace(g.telemetry_hooks.LAYOUT,'',1).replace(g.telemetry_hooks.STEP,'',1),original)
        self.assertNotIn('get_kv_cache_groups',g.telemetry_hooks.LAYOUT)

    def test_layout_names_actual_state_groups_and_does_not_mutate_them(self):
        import textwrap
        from types import SimpleNamespace as NS
        MambaSpec=type('MambaSpec',(),{});state=MambaSpec();state.block_size=4048
        attention=NS(block_size=4048)
        groups=[NS(kv_cache_spec=NS(first_spec=state,block_size=4048),layer_names=['model.layers.0.gdn']),
                NS(kv_cache_spec=attention,layer_names=['model.layers.3.attn'])]
        rows=[];obj=NS(kv_cache_manager=NS(block_pool=NS(_flash_pool_stats_every=200)),
                       kv_cache_config=NS(num_blocks=1340,kv_cache_groups=groups))
        ns=dict(self=obj,logger=NS(info=lambda fmt,record:rows.append(json.loads(record))))
        exec(textwrap.dedent(g.telemetry_hooks.LAYOUT),ns)
        self.assertEqual(rows[0]['groups'][0]['kind'],'MambaSpec')
        self.assertEqual(rows[0]['groups'][0]['id'],0)
        self.assertFalse(rows[0]['enabled']);self.assertIs(obj.kv_cache_config.kv_cache_groups,groups)
        obj.kv_cache_manager.block_pool._flash_pool_stats_every=0
        exec(textwrap.dedent(g.telemetry_hooks.LAYOUT),ns);self.assertEqual(len(rows),1)


if __name__=='__main__':unittest.main()
