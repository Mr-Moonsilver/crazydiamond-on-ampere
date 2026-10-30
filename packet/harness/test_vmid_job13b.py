"""CPU ABI regression: replay/close own Graphs; serialization owns cudaGraph_t."""
import ast
import copy
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
import vmid_job13 as failed_job
import vmid_job13b as job
import test_vmid_job13 as grade_fixture

ROOT = Path(__file__).resolve().parents[1]
OWNER, GRAPH = 0x100010, 0x200020


class NativeABISpy:
    def __init__(self):
        self.replayed = []
        self.closed = []

    def vmg_replay(self, owner):
        if owner != OWNER:
            raise AssertionError('raw graph passed where Graphs owner is required')
        self.replayed.append(owner)
        return 0

    def vmg_close(self, owner):
        if owner != OWNER:
            raise AssertionError('raw graph passed to close')
        self.closed.append(owner)
        return 0


def case_ast(module):
    tree = ast.parse(Path(module.__file__).read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'case')


def native_replay_sites(module):
    return [n for n in ast.walk(case_ast(module)) if isinstance(n, ast.Call)
            and ((isinstance(n.func, ast.Attribute) and n.func.attr == 'vmg_replay')
                 or (isinstance(n.func, ast.Name) and n.func.id == 'replay_owned_graph'))]


class ReplayABI(unittest.TestCase):
    def test_actual_warm_and_measured_calls_use_owner(self):
        spy = NativeABISpy()
        sites = native_replay_sites(job)
        self.assertEqual(len(sites), 2)
        env = dict(gl=types.SimpleNamespace(lib=spy), graph_owner=OWNER,
                   graph_handle=GRAPH, replay_owned_graph=job.replay_owned_graph)
        for site in sites:
            expression = ast.Expression(body=site)
            ast.fix_missing_locations(expression)
            self.assertEqual(eval(compile(expression, job.__file__, 'eval'), env), 0)
        self.assertEqual(spy.replayed, [OWNER, OWNER])

    def test_fixture_rejects_both_calls_in_the_failed_job13(self):
        # Same source-level evaluation, no CUDA, confirms the test would have
        # rejected the delivered bug before the first native launch.
        sites = native_replay_sites(failed_job)
        self.assertEqual(len(sites), 2)
        spy = NativeABISpy()
        env = dict(gl=types.SimpleNamespace(lib=spy), graph_owner=OWNER,
                   graph_handle=GRAPH)
        for site in sites:
            with self.assertRaisesRegex(AssertionError, 'raw graph'):
                eval(compile(ast.Expression(body=site), failed_job.__file__, 'eval'), env)

    def test_only_raw_graph_handle_is_serialized(self):
        seen = []
        def graph_bytes(lib, handle):
            self.assertEqual(handle, GRAPH)
            seen.append(handle)
            return b'cuda graph bytes'
        sites = [n for n in ast.walk(case_ast(job)) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'graph_bytes']
        self.assertEqual(len(sites), 2)
        for node in sites:
            self.assertEqual(eval(compile(ast.Expression(body=node), job.__file__, 'eval'),
                dict(s=types.SimpleNamespace(graph_bytes=graph_bytes), lib=object(),
                     graph_handle=GRAPH, graph_owner=OWNER)), b'cuda graph bytes')
        self.assertEqual(seen, [GRAPH, GRAPH])

    def test_actual_close_call_uses_owner(self):
        spy = NativeABISpy()
        sites = [n for n in ast.walk(case_ast(job)) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'vmg_close']
        self.assertEqual(len(sites), 1)
        eval(compile(ast.Expression(body=sites[0]), job.__file__, 'eval'),
             dict(gl=types.SimpleNamespace(lib=spy), graph_owner=OWNER, graph_handle=GRAPH))
        self.assertEqual(spy.closed, [OWNER])

    def test_unproved_owner_is_refused_before_native_call(self):
        spy = NativeABISpy()
        for owner, graph in ((0, GRAPH), (OWNER, 0), (GRAPH, GRAPH),
                             (True, GRAPH), (OWNER, ctypes.c_uint64(GRAPH))):
            with self.subTest(owner=owner, graph=graph):
                with self.assertRaisesRegex(RuntimeError, 'owner_identity_unproved'):
                    job.replay_owned_graph(types.SimpleNamespace(lib=spy), owner, graph)
        self.assertEqual(spy.replayed, [])

    def test_pinned_helper_declares_separate_owner_and_graph_outputs(self):
        text = (ROOT / 'harness/native/vmid_graph_probe.cu').read_text()
        self.assertIn('uint64_t *owner, uint64_t *graph)', text)
        self.assertIn('int vmg_replay(uint64_t owner)', text)
        self.assertIn('reinterpret_cast<Graphs*>(owner)', text)
        self.assertIn('cudaGraphLaunch(p->exec, p->stream)', text)


class RetainedEvidence(unittest.TestCase):
    def test_progress_is_append_only_and_names_before_native_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            job.progress(p, 'positive:before_warm_replay')
            job.progress(p, 'positive:after_warm_replay')
            rows = [json.loads(line) for line in (p / 'progress.jsonl').read_text().splitlines()]
            self.assertEqual([r['phase'] for r in rows],
                ['positive:before_warm_replay', 'positive:after_warm_replay'])
        source = Path(job.__file__).read_text()
        for name in ('before_warm_replay', 'after_warm_replay',
                     'before_measured_replay', 'after_measured_replay',
                     'before_bind_bank', 'before_window_end', 'after_window_end'):
            self.assertIn(name, source)

    def test_existing_grades_and_failure_precedence_are_unchanged(self):
        for record in (grade_fixture.positive_record(), {},
                       grade_fixture.NegativeControlTests()._record()):
            self.assertEqual(job.case_grade(copy.deepcopy(record)),
                             failed_job.case_grade(copy.deepcopy(record)))
        record = grade_fixture.positive_record()
        record['native_report']['overflow'] = 1
        self.assertEqual(job.case_grade(copy.deepcopy(record)), failed_job.case_grade(record))

    def test_fresh_job_names_and_same_resource_guards(self):
        script = ROOT / 'harness/runs/vmid_job13b.sh'
        source = script.read_text()
        self.assertIn('Q=/opt/flashnext/runs/job13b', source)
        for name in ('flashnext-job13b', 'flashnext-job13b-build',
                     '--network none', 'vmid_job13b.py', '/opt/flashnext/.job.lock'):
            self.assertIn(name, source)
        self.assertEqual(job.s.CAP_MIB, 3072)
        self.assertEqual(job.s.STOP_MIB, 2816)
        self.assertIn('set_per_process_memory_fraction(.125, 0)', Path(job.__file__).read_text())
        subprocess.run(['bash', '-n', str(script)], check=True)
        import vmid_support
        self.assertIn('flashnext-job13b', vmid_support.OWNED_CONTAINERS)
        self.assertIn('flashnext-job13b-build', vmid_support.OWNED_CONTAINERS)
        self.assertIn('flashnext-job13b', vmid_support.LONG_RUN_CONTAINERS)


if __name__ == '__main__':
    unittest.main()
