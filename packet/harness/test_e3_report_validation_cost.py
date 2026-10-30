"""Report validator equivalence and loop placement, CPU only."""
import ast
import hashlib
import inspect
from pathlib import Path
import sys
import textwrap
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
import wire_v1 as w
from e3_report_validation_check import check,BASELINE_SHA


class ValidationTests(unittest.TestCase):
    def test_frozen_acceptance_rejections_and_messages_identical(self):
        j=check(w,ROOT/'harness/fixtures/native13-worker-report-post-init.py.txt')
        self.assertEqual(j['status'],'PASS')
        self.assertEqual(len(j['cases']),19)
        print('report-validation-benchmark',j['timings'])

    def test_union_computed_once_outside_timestamp_loop(self):
        tree=ast.parse(textwrap.dedent(inspect.getsource(w.WorkerCompletionReport.__post_init__)))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='union']
        self.assertEqual(len(calls),1)
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.iter)=='self.host_submit_ns')
        self.assertFalse(any(n is calls[0] for n in ast.walk(loop)))

    def test_only_the_union_hoist_differs_from_frozen_method(self):
        old=(ROOT/'harness/fixtures/native13-worker-report-post-init.py.txt').read_text()
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),BASELINE_SHA)
        expected=old.replace('    last = -1\n','    accepted_ops = done.union(self.pending_op_seqs)\n    last = -1\n').replace('if op_seq not in done.union(self.pending_op_seqs):','if op_seq not in accepted_ops:')
        actual=textwrap.dedent(inspect.getsource(w.WorkerCompletionReport.__post_init__))
        self.assertEqual(ast.dump(ast.parse(expected)),ast.dump(ast.parse(actual)))


if __name__=='__main__':unittest.main()
