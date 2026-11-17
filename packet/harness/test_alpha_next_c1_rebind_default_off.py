"""tools/alpha_next_c1_rebind_default_off.py (lead 2 Nov 21:0x CET, point 4): the proof holds on this tree, writes its receipt
once, and every check in it is live (a dropped projection, an added statement or a failing refusal test turns it FAIL)."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools' / 'alpha_next_c1_rebind_default_off.py'
sys.path.insert(0, str(ROOT / 'tools'))
import alpha_next_c1_rebind_default_off as proof  # noqa: E402


def quiet(**patches):
    """main() under patches; returns (rc, receipt or None, error)."""
    out = io.StringIO()
    with contextlib.ExitStack() as stack:
        for name, value in patches.items():
            stack.enter_context(mock.patch.object(proof, name, value))
        stack.enter_context(contextlib.redirect_stdout(out))
        try:
            rc = proof.main([])
        except SystemExit as exc:
            return 'exit', None, str(exc)
    return rc, json.loads(out.getvalue()), None


class Holds(unittest.TestCase):
    def test_pass_on_this_tree_and_receipt_written_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'receipt.json')
            first = subprocess.run([sys.executable, '-B', str(TOOL), '--out', out], capture_output=True, text=True,
                                   timeout=900)
            self.assertEqual(first.returncode, 0, first.stderr[-2000:])
            receipt = json.loads(Path(out).read_text())
            self.assertEqual(receipt['verdict'], 'PASS')
            self.assertIs(receipt['passed'], True)
            self.assertEqual(receipt['other_engine_files_differing'], [])
            self.assertTrue(all(row['projected_ast_equals_base'] for row in receipt['files'].values()))
            self.assertEqual(sorted(receipt['files']), sorted(['stage_overlap.py', *proof.OTHERS]))
            self.assertEqual(receipt['flag_unset_field_refused']['rc'], 0)
            self.assertGreater(receipt['flag_unset_field_refused']['ok_lines'], 0)
            again = subprocess.run([sys.executable, '-B', str(TOOL), '--out', out], capture_output=True, text=True,
                                   timeout=900)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn('FileExistsError', again.stderr)
            self.assertEqual(json.loads(Path(out).read_text()), receipt)


class Live(unittest.TestCase):
    def assertFails(self, **patches):
        rc, receipt, error = quiet(**patches)
        if rc == 'exit':
            self.assertIn('does not match exactly once', error)
            return
        self.assertEqual(rc, 1)
        self.assertEqual(receipt['verdict'], 'FAIL')

    def test_each_per_arm_projection_is_needed(self):
        for index in range(len(proof.PER_ARM)):
            with self.subTest(index=index):
                self.assertFails(PER_ARM=proof.PER_ARM[:index] + proof.PER_ARM[index + 1:])

    def test_each_other_projection_is_needed(self):
        for name, pairs in proof.OTHERS.items():
            for index in range(len(pairs)):
                with self.subTest(name=name, index=index):
                    others = dict(proof.OTHERS)
                    others[name] = pairs[:index] + pairs[index + 1:]
                    self.assertFails(OTHERS=others)

    def test_an_added_statement_breaks_the_ast_compare(self):
        old, _ = proof.PER_ARM[0]
        self.assertFails(PER_ARM=[(old, '        self.extra = 1\n')] + proof.PER_ARM[1:])
        name = 's5_startup.py'
        old, _ = proof.OTHERS[name][0]
        self.assertFails(OTHERS={**proof.OTHERS, name: [(old, '    extra = 1\n')]})

    def test_a_differing_engine_file_fails(self):
        real = proof.git_show

        def show(rev, path):
            data = real(rev, path)
            return data + b'\n' if path.endswith('/engine_driver.py') else data
        self.assertFails(git_show=show)

    def test_a_failing_refusal_test_fails(self):
        self.assertFails(REFUSAL_TESTS=('test_alpha_next_c1_defer_control.NoSuchClass',))

    def test_refusals_run_with_the_switches_removed(self):
        seen = {}

        def run(cmd, **kwargs):
            seen.update(kwargs['env'])
            return subprocess.CompletedProcess(cmd, 0, '', 'test (x) ... ok\nOK\n')
        with mock.patch.dict(os.environ, {name: '1' for name in proof.SWITCHES}), \
                mock.patch.object(proof.subprocess, 'run', run):
            row = proof.refusals()
        self.assertEqual(row['rc'], 0)
        self.assertFalse(set(proof.SWITCHES) & set(seen))


if __name__ == '__main__':
    unittest.main()
