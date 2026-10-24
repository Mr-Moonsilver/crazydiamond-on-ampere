#!/usr/bin/env python3
"""Offline counterexamples for the round-7 review, not fixes to deployed queues.

These tests assert the current defects so the review is reproducible. Turn them
into desired-behavior regression tests when adopting the recommended fixes.
"""
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
TMP = HERE.parent.parent / 'tmp'


class ReviewTests(unittest.TestCase):
    def setUp(self):
        TMP.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='round7-review-', dir=TMP))
        spec = importlib.util.spec_from_file_location('review_decode_ab', HERE / 'decode_ab.py')
        self.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.m)
        self.m.N, self.m.LEN, self.m.PLEN, self.m.LONG, self.m.MIN_MIXED = 2, 70, 1, 1, 1
        self.pre = {'t0': 0, 't1': 10, 'usage': {'prompt_tokens': 1, 'prompt_tokens_details': {'cached_tokens': 0}}}

    def record(self):
        return {'tokens': ['token_id:7'] * 70, 'lp': [-.01] * 70,
                'top': [{'token_id:7': -.01, 'token_id:8': -2.} for _ in range(70)],
                'ts': [2.] * 70, 't0': 0., 't1': 3.,
                'usage': {'prompt_tokens': 1, 'prompt_tokens_details': {'cached_tokens': 0}}}

    def finish(self, runs):
        out = self.root / 'evidence.json'
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ex:
            self.m.finish(runs, self.pre, True, out)
        return ex.exception.code, json.loads(out.read_text())

    def runs(self):
        return {k: [self.record(), self.record()] for k in ('alone-a', 'alone-b', 'under-prefill')}

    def test_valid_noise_prefix_cannot_vouch_for_failure_it_actually_covers(self):
        runs = self.runs()
        # Noise agrees on positions 0..29, then accepts a symmetric near tie.
        a, b = runs['alone-a'][0], runs['alone-b'][0]
        a['top'][30] = {'token_id:7': -.01, 'token_id:8': -.02}
        b['tokens'][30] = 'token_id:8'
        b['top'][30] = {'token_id:8': -.01, 'token_id:7': -.02}
        c = runs['under-prefill'][0]
        c['top'][30] = copy.deepcopy(a['top'][30])
        # Candidate shares all 70 tokens but fails at position 5, well inside the
        # 30-position clean control. Requiring coverage of all 70 hides this FAIL.
        c['lp'][5] = -.5; c['top'][5]['token_id:7'] = -.5
        rc, d = self.finish(runs)
        self.assertEqual(d['summary']['noise'][0]['prefix'], 30)
        self.assertEqual(d['summary']['noise'][0]['status'], 'ok')
        self.assertEqual(d['summary']['under_prefill'][0]['status'], 'fail')
        self.assertEqual(rc, 2, 'current defect: expected fix is FAIL for the covered failure at position 5')

    def test_infinite_prefill_window_currently_false_passes(self):
        self.pre.update(t0=float('-inf'), t1=float('inf'))
        rc, _ = self.finish(self.runs())
        self.assertEqual(rc, 0, 'current defect: nonfinite prefill endpoints must be INCONCLUSIVE')

    def test_arrivals_outside_own_request_currently_false_pass(self):
        runs = self.runs()
        for c in runs:
            for r in runs[c]:
                r.update(t0=100., t1=101.)  # all recorded arrivals are 2.0
        rc, _ = self.finish(runs)
        self.assertEqual(rc, 0, 'current defect: arrival bounds must be validated')

    def test_q8_interrupt_bail_currently_exits_success(self):
        # Only evaluate the exact bail function with a no-op restore and qlib's
        # accounting. Never run q8, systemctl, docker or any box path.
        src = (HERE / 'runs/q8.sh').read_text()
        bail = re.search(r'^bail\(\) \{.*\}$', src, re.M).group(0)
        shell = 'source "$1"\nCHANGED=1\nrestore() { :; }\n' + bail + '\nbail\n'
        r = subprocess.run(['bash', '-c', shell, 'review', str(HERE / 'qlib.sh')], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertIn('fails=0 inconclusive=0', r.stdout)

    def test_q8_failed_setcfg_can_still_report_restore_pass(self):
        src = (HERE / 'runs/q8.sh').read_text()
        restore = re.search(r'^restore\(\) \{\n.*?^\}', src, re.M | re.S).group(0)
        # The one absolute executable is substituted with a no-op fake. Other
        # names are shell functions. The production path is never executed.
        restore = restore.replace('/usr/local/sbin/flash-reconfig.sh', 'mock_reconfig')
        shell = '''source "$1"
P=/fixture; MD=/fixture; Q=$2
python3() { return 1; }
systemctl() { :; }
mock_reconfig() { return 0; }
installed() { return 0; }
model_is() { return 0; }
''' + restore + '\nrestore\n'
        r = subprocess.run(['bash', '-c', shell, 'review', str(HERE / 'qlib.sh'), str(self.root)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertIn('STEP restart-final PASS', r.stdout)


if __name__ == '__main__':
    unittest.main()
