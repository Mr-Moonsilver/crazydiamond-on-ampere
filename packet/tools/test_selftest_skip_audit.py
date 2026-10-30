"""Run the actual selftest audit function on success and missing-coverage summaries."""
import ast
import contextlib
import hashlib
import io
from pathlib import Path
import re
import types
import unittest

class SkipAudit(unittest.TestCase):
    def test_actual_audit_returncode_and_observations(self):
        source=Path(__file__).resolve().parents[1]/'harness/selftest.py'
        fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='audit_suite')
        scope=dict(re=re,hashlib=hashlib,json=__import__('json'),unittest_observations=[])
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual-selftest-audit','exec'),scope)
        for output,rc,want,skips in [('Ran 5 tests in 0.1s\nOK\n',0,0,0),
                ('Ran 0 tests in 0.0s\nOK (skipped=1)\n',0,1,1),
                ('Ran 8 tests in 0.1s\nOK (skipped=3)\n',0,1,3),
                ('Ran 8 tests in 0.1s\nFAILED (errors=1, skipped=2)\n',1,1,2),
                ('source fixture printed skipped=1 outside summary\nOK\n',0,0,0)]:
            with contextlib.redirect_stdout(io.StringIO()):
                got=scope['audit_suite']('fixture',types.SimpleNamespace(stdout='',stderr=output,returncode=rc))
            self.assertEqual(got,want);self.assertEqual(scope['unittest_observations'][-1]['skipped'],skips)

if __name__=='__main__':unittest.main()
