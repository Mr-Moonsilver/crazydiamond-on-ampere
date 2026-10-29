"""Time-only preparation tests. No box, Docker or GPU calls in this suite."""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rewindow', ROOT / 'tools/native18_rewindow.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RewindowTests(unittest.TestCase):
    def setUp(self):
        self.end = m.utc_end('2030-01-02T07:00:00Z')
        self.queue = (ROOT / 'harness/runs/q18a_native18.draft.sh').read_text()
        self.runner = (ROOT / 'tools/q18a_native18_owner_run.py').read_text()

    def test_real_guard_exact_boundary_and_unchanged_budget(self):
        q = m.window_text(self.queue, self.end)
        r = m.window_text(self.runner, self.end, runner=True)
        result = m.guard_boundary_check(q, r, self.end)
        self.assertEqual([x['queue_accepts'] for x in result], [True, True, False])
        self.assertIn('--kill-after=900s 14400s', q)
        self.assertEqual(q.count('14400+900'), self.queue.count('14400+900'))

    def test_runner_only_deadline_docstring_and_pin_values(self):
        revised = m.replace_pins(m.window_text(self.runner, self.end, runner=True), {
            'H': "Path('/fresh')", 'BUNDLE': "'fresh'", 'REVIEW': "'new'", 'EXPECTED_SOURCES': '{}'})
        def scrub(s):
            tree = ast.parse(s)
            tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign)
                         and any(getattr(t, 'id', None) in m.PIN_NAMES for t in n.targets))]
            tree.body = tree.body[1:]  # module docstring only
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'check_slot':
                    node.body[0].value = ast.Constant('HARD_END')
            return ast.dump(tree, include_attributes=False)
        self.assertEqual(scrub(revised), scrub(self.runner))

    def test_queue_bytes_outside_window_block_unchanged(self):
        revised = m.window_text(self.queue, self.end)
        a = self.queue.index('# Operator order 28 Oct night:')
        b = self.queue.index('\nPY_SLOT\n')
        c = revised.index('# Prepared hard end ')
        d = revised.index('\nPY_SLOT\n')
        self.assertEqual(self.queue[:a], revised[:c])
        self.assertEqual(self.queue[b:], revised[d:])

    def test_ambiguous_dates_and_changed_templates_refused(self):
        for value in ('2030-01-02', '2030-01-02T07:00:00', '2030-01-02T07:00:00+01:00',
                      '2030-02-30T07:00:00Z'):
            with self.assertRaises(ValueError):
                m.utc_end(value)
        with self.assertRaises(ValueError):
            m.window_text(self.queue.replace(m.OLD_DATE, '0'), self.end)
        with self.assertRaises(ValueError):
            m.window_text(self.runner + m.OLD_DATE, self.end, runner=True)

    def test_held_manifest_and_runner_are_still_pinned(self):
        self.assertEqual(len(m.inventory(m.HELD, m.HELD_SHA)), 889)
        self.assertEqual(m.sha(ROOT / 'tools/q18a_native18_owner_run.py'), m.RUNNER_SHA)


if __name__ == '__main__':
    unittest.main()
