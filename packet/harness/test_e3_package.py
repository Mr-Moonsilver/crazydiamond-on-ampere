"""S5 overlap generator boundary tests.

The fixtures are NEW offline directories. Nothing is mounted, installed or
started. The full-source case uses the read-only pinned reference tree; the
fast cases patch transforms so manifest and composition rules are checked
without depending on image availability.
"""
import ast
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import executor_hooks as eh
import make_e2_arena as e2
import make_s5_overlap as g
import scheduler_hooks as sh


def manifest(folder, skip=()):
    folder = Path(folder)
    files = [p for p in sorted(folder.rglob('*'))
             if p.is_file() and p != folder / 'SHA256SUMS'
             and p.relative_to(folder).as_posix() not in skip]
    (folder / 'SHA256SUMS').write_text(
        ''.join(g.sha(p) + '  ' + p.relative_to(folder).as_posix() + '\n' for p in files))
    return g.sha(folder / 'SHA256SUMS')


class SyntheticTests(unittest.TestCase):
    def fixture(self):
        root = Path(tempfile.mkdtemp(prefix='s5-package-', dir=ROOT.parent / 'tmp'))
        base, ref = root / 'base', root / 'reference'
        for name, text in (('model_runner_v2.py', '# runner\n'),
                           ('core/scheduler.py', '# scheduler\n'),
                           ('core/block_pool.py', '# pool\n'),
                           ('e2/__init__.py', ''),
                           ('e2/startup.py', 'STARTUP = True\n'),
                           ('unchanged.py', 'UNCHANGED = True\n')):
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        (base / 'E2-GENERATION.json').write_text(json.dumps(
            {'default_feature': 'FLASH_E2_ARENA=0', 'incremental_kv_bytes': 0}))
        launcher = e2.launcher((ROOT / 'deploy/run-flash-prod.sh').read_text())
        (base / 'run-flash-e2-test.sh').write_text(launcher)
        for name in eh.PINS:
            path = ref / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# ref ' + name + '\n')
        core = manifest(base / 'core')
        whole = manifest(base)
        return root, base, ref, whole, core

    def stubs(self, base, ref):
        stack = ExitStack()
        stack.enter_context(patch.object(g, 'E2_INPUTS', {
            name: g.sha(base / name) for name in g.E2_INPUTS}))
        stack.enter_context(patch.object(g, 'REFERENCE_INPUTS', {
            name: g.sha(ref / name) for name in eh.PINS}))
        stack.enter_context(patch.object(g, 'compose_runner',
                                         side_effect=lambda s: s + '# A+C builder\n'))
        stack.enter_context(patch.object(g.scheduler_hooks, 'generate_scheduler',
                                         side_effect=lambda s: s + '# S5 scheduler\n'))
        stack.enter_context(patch.object(g.scheduler_hooks, 'generate_block_pool',
                                         side_effect=lambda s: s + '# S5 pool\n'))
        for name in ('generate_executor', 'generate_core', 'generate_worker'):
            stack.enter_context(patch.object(g.executor_hooks, name,
                                             side_effect=lambda s: s + '# S5 source\n'))
        return stack

    def build(self):
        root, base, ref, whole, core = self.fixture()
        with self.stubs(base, ref):
            result = g.build(base, root / 'out', ref, whole)
        return root, base, ref, result

    def test_composes_new_package_and_preserves_baseline_bytes(self):
        root, base, ref, result = self.build()
        out = root / 'out'
        for name in ('unchanged.py', 'e2/startup.py', 'e2/__init__.py',
                     'E2-GENERATION.json', 'run-flash-e2-test.sh'):
            self.assertEqual((base / name).read_bytes(), (out / name).read_bytes(), name)
        self.assertEqual(result['manifest_sha256'], g.sha(out / 'SHA256SUMS'))
        listed = set(g.manifest(out))
        self.assertEqual(listed, {p.relative_to(out).as_posix() for p in out.rglob('*')
                                  if p.is_file() and p != out / 'SHA256SUMS'})
        g.manifest(out / 'core')
        for name in g.E3_MODULES:
            self.assertIn('e3/' + name + '.py', listed)
        for name in ('e3_sources/executor.py', 'e3_sources/core.py',
                     'e3_sources/worker.py', 'run-flash-s5-test.sh',
                     'S5-OVERLAP-GENERATION.json'):
            self.assertIn(name, listed)
        self.assertEqual((out / 'model_runner_v2.py').read_text(),
                         '# runner\n# A+C builder\n')
        # The package rewrite is required for a separately mounted e3 package.
        text = (out / 'e3/async_handoff.py').read_text()
        self.assertIn('from .e3_types import', text)

    def test_default_off_settings_budget_and_launcher_guard(self):
        root, base, ref, result = self.build()
        out = root / 'out'
        record = json.loads((out / 'S5-OVERLAP-GENERATION.json').read_text())
        self.assertEqual(record['default_feature'], 'FLASH_S5_OVERLAP=0')
        self.assertEqual(record['installed_settings']['FLASH_S5_OVERLAP'], '0')
        self.assertEqual(record['installed_settings']['FLASH_S5_TRANSPORT'], 'dense_dma')
        self.assertEqual(record['installed_settings']['FLASH_S5_SPARSE_OVERLAP'], '0')
        self.assertEqual(record['incremental_kv_bytes'], 0)
        self.assertIn('no claim on the unmet 512-MiB reserve',
                      record['allocation_budget']['funding'])
        self.assertIn('prefill_uid_buffer', record['allocation_budget']['unmeasured'])
        self.assertGreater(record['allocation_budget']['new_bytes'], 0)
        self.assertEqual(record['base_generation_sha256'],
                         g.sha(base / 'E2-GENERATION.json'))
        launcher = (out / 'run-flash-s5-test.sh').read_text()
        self.assertIn('[[ ${S5:-0} == 1 ]]', launcher)
        self.assertIn('"${runner[@]}" "${e2[@]}" "${s5[@]}"', launcher)
        for name in g.E3_MODULES:
            self.assertIn('e3/' + name + '.py', launcher)
        subprocess.run(['bash', '-n', str(out / 'run-flash-s5-test.sh')], check=True)
        self.assertEqual((out / 'run-flash-e2-test.sh').read_text(),
                         (base / 'run-flash-e2-test.sh').read_text())

    def test_graph_transform_is_applied_last_through_declared_pins(self):
        root, base, ref, whole, core = self.fixture()
        composed = '# runner\n# A+C builder\n'
        stub = GraphCompositionTests.Stub(
            'model_runner_v2.py', composed + '# B graph\n',
            {'model_runner_v2.py': g._sha_text(composed)})
        with self.stubs(base, ref), patch.object(g, 'load_graph', return_value=stub):
            result = g.build(base, root / 'out', ref, whole, graph=root / 'graph.py')
        out = root / 'out'
        self.assertEqual((out / 'model_runner_v2.py').read_text(),
                         composed + '# B graph\n')
        record = json.loads((out / 'S5-OVERLAP-GENERATION.json').read_text())
        self.assertEqual(record['graph_composition']['module'], 'graph.py')
        self.assertEqual(record['graph_composition']['source_pins'],
                         {'model_runner_v2.py': g._sha_text(composed)})
        self.assertIn('model_runner_v2.py', result['generated_files'])

    def test_generated_package_imports_without_vllm(self):
        root, base, ref, result = self.build()
        out = root / 'out'
        subprocess.run([sys.executable, '-I', '-c',
                        'import sys;sys.path.insert(0,sys.argv[1]);'
                        'import e3.async_handoff,e3.scheduler_hooks,e3.s5_startup;'
                        'assert e3.s5_startup.Composition is not None',
                        str(out)], check=True)

    def test_modes_normalized_on_new_output_only(self):
        root, base, ref, whole, core = self.fixture()
        before = {p: p.stat().st_mode & 0o777 for p in base.rglob('*')}
        old = os.umask(0o077)
        try:
            with self.stubs(base, ref):
                g.build(base, root / 'out', ref, whole)
        finally:
            os.umask(old)
        out = root / 'out'
        self.assertEqual(out.stat().st_mode & 0o777, 0o755)
        for path in out.rglob('*'):
            expected = 0o755 if path.is_dir() else 0o644 | (path.stat().st_mode & 0o111)
            self.assertEqual(path.stat().st_mode & 0o777, expected)
        self.assertEqual(before, {p: p.stat().st_mode & 0o777 for p in base.rglob('*')})

    def test_preconditions_fail_before_output_exists(self):
        cases = ('root-manifest', 'e2-record', 'e2-default',
                 'output-exists', 'inside-base', 'launcher-s5')
        for case in cases:
            root, base, ref, whole, core = self.fixture()
            kwargs = {}
            if case == 'root-manifest':
                whole = '0' * 64
            elif case == 'e2-record':
                (base / 'E2-GENERATION.json').unlink()
                whole = manifest(base)
            elif case == 'e2-default':
                (base / 'E2-GENERATION.json').write_text(json.dumps(
                    {'default_feature': 'FLASH_E2_ARENA=1', 'incremental_kv_bytes': 0}))
                whole = manifest(base)
            elif case == 'output-exists':
                (root / 'out').mkdir()
            elif case == 'inside-base':
                kwargs['out'] = base / 'child'
            elif case == 'launcher-s5':
                text = (base / 'run-flash-e2-test.sh').read_text()
                (base / 'run-flash-e2-test.sh').write_text(
                    text + '\n# S5 needs the declared overlap launcher\n')
                whole = manifest(base)
            out = kwargs.pop('out', root / 'out')
            with self.subTest(case=case):
                if case == 'e2-default':
                    with self.assertRaisesRegex(ValueError, 'not default-off'):
                        g.build(base, out, ref, whole)
                else:
                    with self.assertRaises(ValueError):
                        g.build(base, out, ref, whole)
                self.assertFalse(out.exists() and case != 'output-exists')

    def test_manifest_path_traversal_and_symlinks_refused(self):
        for defect in ('traversal', 'duplicate', 'symlink', 'unmanifested'):
            root, base, ref, whole, core = self.fixture()
            with self.subTest(defect=defect):
                if defect == 'traversal':
                    (base / 'SHA256SUMS').write_text(
                        (base / 'SHA256SUMS').read_text() +
                        g.sha(base / 'unchanged.py') + '  ../escape.py\n')
                    whole = g.sha(base / 'SHA256SUMS')
                elif defect == 'duplicate':
                    line = (base / 'SHA256SUMS').read_text().splitlines()[0]
                    (base / 'SHA256SUMS').write_text(
                        (base / 'SHA256SUMS').read_text() + line + '\n')
                    whole = g.sha(base / 'SHA256SUMS')
                elif defect == 'symlink':
                    (base / 'link.py').symlink_to(base / 'unchanged.py')
                    whole = manifest(base)
                else:
                    (base / 'extra.py').write_text('X = 1\n')
                with self.assertRaises(ValueError):
                    with self.stubs(base, ref):
                        g.build(base, root / 'out', ref, whole)
                self.assertFalse((root / 'out').exists())


class GraphCompositionTests(unittest.TestCase):
    class Stub:
        def __init__(self, name, output, pins):
            self.name = name
            self.output = output
            self.SOURCE_PINS = pins

        def generate(self, files):
            return {self.name: self.output}

    def test_graph_may_only_replace_its_pinned_input(self):
        files = {'model_runner_v2.py': 'RUNNER = 1\n', 'core/scheduler.py': 'S = 1\n'}
        pins = {'model_runner_v2.py': g._sha_text(files['model_runner_v2.py'])}
        stub = self.Stub('model_runner_v2.py', 'RUNNER = 2\n', pins)
        out = g.compose_graph(stub, files)
        self.assertEqual(out['model_runner_v2.py'], 'RUNNER = 2\n')
        self.assertEqual(out['core/scheduler.py'], 'S = 1\n')
        with self.assertRaises(ValueError):
            g.compose_graph(self.Stub('core/scheduler.py', 'S = 2\n', pins), files)
        with self.assertRaises(ValueError):
            g.compose_graph(self.Stub('model_runner_v2.py', 'S = 2\n',
                                      {'model_runner_v2.py': '0' * 64}), files)
        with self.assertRaises(ValueError):
            g.compose_graph(self.Stub('extra.py', 'X = 1\n', {'extra.py': '0' * 64}), files)
        with self.assertRaises(ValueError):
            g.compose_graph(self.Stub('model_runner_v2.py', 'def (:\n', pins), files)

    def test_graph_module_requires_the_declared_interface(self):
        root = Path(tempfile.mkdtemp(prefix='s5-graph-', dir=ROOT.parent / 'tmp'))
        for text in ('def generate(files): return files\n',
                     'SOURCE_PINS={"x.py": "0"*64}\n',
                     'SOURCE_PINS={}\ndef generate(files): return files\n'):
            path = root / 'graph.py'
            path.write_text(text)
            with self.assertRaises(ValueError):
                g.load_graph(path)
        path = root / 'graph.py'
        path.write_text('SOURCE_PINS={"x.py":"0"*64}\n'
                        'def generate(files): return {"x.py": "# graph\\n"}\n')
        module = g.load_graph(path)
        self.assertEqual(module.SOURCE_PINS['x.py'], '0' * 64)


class FullPinnedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ref = ROOT / 'reference/vllm-src'
        needed = [ref / eh.EXECUTOR_PATH, ref / eh.CORE_PATH, ref / eh.WORKER_PATH,
                  ref / 'v1/worker/gpu/model_runner.py',
                  ref / 'v1/core/sched/scheduler.py', ref / 'v1/core/block_pool.py',
                  ROOT / 'deploy/run-flash-prod.sh']
        if not all(path.is_file() for path in needed):
            raise unittest.SkipTest('pinned reference tree absent')
        root = Path(tempfile.mkdtemp(prefix='s5-full-', dir=ROOT.parent / 'tmp'))
        base = root / 'base'
        sources = {'model_runner_v2.py': 'v1/worker/gpu/model_runner.py',
                   'core/scheduler.py': 'v1/core/sched/scheduler.py',
                   'core/block_pool.py': 'v1/core/block_pool.py'}
        for name, source in sources.items():
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            text = (ref / source).read_text()
            if name == 'model_runner_v2.py':
                text = e2.runner_hooks.runner_startup(e2.runner_hooks.runner_execution(text))
            elif name.endswith('scheduler.py'):
                text = e2.mirror_hooks.scheduler(e2.telemetry_hooks.scheduler(text))
            else:
                text = e2.mirror_hooks.block_pool(e2.telemetry_hooks.block_pool(text))
            path.write_text(text)
        (base / 'e2').mkdir()
        (base / 'e2/__init__.py').write_text('')
        (base / 'e2/startup.py').write_text('"""inert E2 fixture"""\n')
        (base / 'E2-GENERATION.json').write_text(json.dumps(
            {'default_feature': 'FLASH_E2_ARENA=0', 'incremental_kv_bytes': 0}))
        launcher = e2.launcher((ROOT / 'deploy/run-flash-prod.sh').read_text())
        (base / 'run-flash-e2-test.sh').write_text(launcher)
        core = manifest(base / 'core')
        whole = manifest(base)
        cls.root = root
        cls.base = base
        cls.out = root / 'out'
        cls.whole = whole
        cls.result = g.build(base, cls.out, ref, whole)

    def test_runner_composes_e2_lane_then_startup_builder(self):
        text = (self.out / 'model_runner_v2.py').read_text()
        install = text.index('_install_flash_e2(self, is_profiling)')
        builder = text.index('_flash_s5_startup.install_builder(self, kv_caches_dict, is_profiling)')
        lane = text.index('_flash_e3_hooks.init_lane(self, kv_caches_dict, is_profiling)')
        self.assertLess(install, builder)
        self.assertLess(builder, lane)
        self.assertEqual(text.count('_flash_s5_startup.install_builder'), 1)
        self.assertIn('import vllm.v1.flash_e3.s5_startup as _flash_s5_startup', text)
        # C's reverse path proves only C's insertion is removed before A's.
        self.assertEqual(g.revert_runner(text, (self.base / 'model_runner_v2.py').read_text()),
                         (self.base / 'model_runner_v2.py').read_text())

    def test_scheduler_and_pool_leases_revert_to_reviewed_e2(self):
        scheduler = (self.out / 'core/scheduler.py').read_text()
        pool = (self.out / 'core/block_pool.py').read_text()
        self.assertIn('_flash_s5_hooks.defer_request_free', scheduler)
        self.assertIn('_flash_s5_hooks.blocked_deferred', scheduler)
        self.assertIn('_flash_s5_hooks.hold_leased_blocks', pool)
        self.assertIn('_flash_s5_hooks.block_held', pool)
        self.assertEqual(sh.revert_scheduler(scheduler),
                         (self.base / 'core/scheduler.py').read_text())
        self.assertEqual(sh.revert_block_pool(pool),
                         (self.base / 'core/block_pool.py').read_text())

    def test_executor_core_worker_exact_and_reversible(self):
        ref = ROOT / 'reference/vllm-src'
        executor = (self.out / 'e3_sources/executor.py').read_text()
        core = (self.out / 'e3_sources/core.py').read_text()
        worker = (self.out / 'e3_sources/worker.py').read_text()
        self.assertIn('_flash_s5_hooks.collective_rpc', executor)
        self.assertIn('_flash_s5_stock_forbidden', executor)
        self.assertIn('_flash_s5_hooks.before_step_with_batch_queue', core)
        self.assertIn('_flash_s5_hooks.before_engine_shutdown', core)
        for name in ('open', 'submit', 'poll', 'commit', 'cancel'):
            self.assertIn('def flash_s5_' + name + '(', worker)
        self.assertEqual(eh.revert_executor(executor, (ref / eh.EXECUTOR_PATH).read_text()),
                         (ref / eh.EXECUTOR_PATH).read_text())
        self.assertEqual(eh.revert_core(core, (ref / eh.CORE_PATH).read_text()),
                         (ref / eh.CORE_PATH).read_text())
        self.assertEqual(eh.revert_worker(worker, (ref / eh.WORKER_PATH).read_text()),
                         (ref / eh.WORKER_PATH).read_text())

    def test_baseline_e2_payload_preserved_and_manifest_complete(self):
        self.assertEqual((self.out / 'e2/startup.py').read_bytes(),
                         (self.base / 'e2/startup.py').read_bytes())
        self.assertEqual((self.out / 'E2-GENERATION.json').read_bytes(),
                         (self.base / 'E2-GENERATION.json').read_bytes())
        listed = set(g.manifest(self.out))
        self.assertEqual(listed, {p.relative_to(self.out).as_posix()
                                  for p in self.out.rglob('*')
                                  if p.is_file() and p != self.out / 'SHA256SUMS'})
        g.manifest(self.out / 'core')
        record = json.loads((self.out / 'S5-OVERLAP-GENERATION.json').read_text())
        self.assertEqual(record['e2_composed_pins'], g.E2_INPUTS)
        self.assertEqual(record['reference_source_pins'], eh.PINS)
        self.assertIsNone(record['graph_composition'])

    def test_package_imports_and_no_old_reader_is_exposed(self):
        subprocess.run([sys.executable, '-I', '-c',
                        'import sys;sys.path.insert(0,sys.argv[1]);'
                        'import e3.executor_hooks,e3.async_handoff;'
                        'assert hasattr(e3.executor_hooks,"LocalFuture")',
                        str(self.out)], check=True)
        executor = (self.out / 'e3_sources/executor.py').read_text()
        # The stock FutureWrapper text stays for byte-exact inert behavior, but
        # is unreachable once the owned receiver is installed.
        self.assertIn('_flash_s5_stock_forbidden', executor)
        self.assertIn('stock collective_rpc response reader is disabled', executor)
        self.assertEqual(executor.count('_flash_s5_hooks.collective_rpc('), 1)

    def test_e2_composed_source_drift_is_refused(self):
        root = Path(tempfile.mkdtemp(prefix='s5-drift-', dir=ROOT.parent / 'tmp'))
        base = root / 'base'
        shutil.copytree(self.base, base)
        (base / 'core/scheduler.py').write_text(
            (base / 'core/scheduler.py').read_text() + '\n# drift\n')
        manifest(base / 'core')
        whole = manifest(base)
        with self.assertRaisesRegex(ValueError, 'unreviewed E2-composed source'):
            g.build(base, root / 'out', ROOT / 'reference/vllm-src', whole)
        self.assertFalse((root / 'out').exists())

    def test_output_never_overwrites_an_existing_directory(self):
        with self.assertRaises(ValueError):
            g.build(self.base, self.out, ROOT / 'reference/vllm-src', self.whole)
        with self.assertRaises(ValueError):
            g.build(self.base, self.base / 'child', ROOT / 'reference/vllm-src',
                    self.whole)


if __name__ == '__main__':
    unittest.main()
