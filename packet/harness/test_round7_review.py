#!/usr/bin/env python3
"""Round-7 review cases (Codex), turned into regression tests for decode_ab v5 and q8 v2."""
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
TMP = HERE.parent.parent / 'tmp'


class ReviewTests(unittest.TestCase):
    def setUp(self):
        TMP.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='round7-review-', dir=TMP))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
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
        self.assertEqual(rc, 1, 'decode_ab v5: a covered failure stays FAIL')

    def test_infinite_prefill_window_currently_false_passes(self):
        self.pre.update(t0=float('-inf'), t1=float('inf'))
        rc, _ = self.finish(self.runs())
        self.assertEqual(rc, 2, 'decode_ab v5: nonfinite prefill endpoints are INCONCLUSIVE')

    def test_arrivals_outside_own_request_currently_false_pass(self):
        runs = self.runs()
        for c in runs:
            for r in runs[c]:
                r.update(t0=100., t1=101.)  # all recorded arrivals are 2.0
        rc, _ = self.finish(runs)
        self.assertEqual(rc, 2, 'decode_ab v5: arrivals must lie inside their own request')

    def test_q8_interrupt_is_recorded_as_failure(self):  # q8 v2: on_exit replaces bail
        src = (HERE / 'runs/q8.sh').read_text()
        on_exit = re.search(r'^on_exit\(\) \{\n.*?^\}', src, re.M | re.S).group(0)
        shell = 'source "$1"\nCHANGED=1; DONE=0; STAGE=s5\nrestore() { :; }\n' + on_exit + '\non_exit\n'
        r = subprocess.run(['bash', '-c', shell, 'review', str(HERE / 'qlib.sh')], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn('STEP interrupted FAIL', r.stdout)

    def test_q8_restore_leaves_engine_stopped_when_gpus_not_free(self):
        src = (HERE / 'runs/q8.sh').read_text()
        restore = re.search(r'^restore\(\) \{\n.*?^\}', src, re.M | re.S).group(0)
        restore = restore.replace('/usr/local/sbin/flash-reconfig.sh', 'mock_reconfig')
        shell = 'source "$1"\nQ=$2\ns5_clean() { return 1; }\nnvidia-smi() { echo 9000; }\npython3() { echo STARTED; }\n' \
                'mock_reconfig() { echo STARTED; }\n' + restore + '\nrestore\n'
        r = subprocess.run(['bash', '-c', shell, 'review', str(HERE / 'qlib.sh'), str(self.root)], capture_output=True, text=True)
        self.assertIn('STEP gpu-handoff FAIL', r.stdout)
        self.assertNotIn('STARTED', r.stdout)

    def test_q8_failed_setcfg_can_still_report_restore_pass(self):
        src = (HERE / 'runs/q8.sh').read_text()
        restore = re.search(r'^restore\(\) \{\n.*?^\}', src, re.M | re.S).group(0)
        # The one absolute executable is substituted with a no-op fake. Other
        # names are shell functions. The production path is never executed.
        restore = restore.replace('/usr/local/sbin/flash-reconfig.sh', 'mock_reconfig')
        shell = '''source "$1"
P=/fixture; MD=/fixture; Q=$2
python3() { return 1; }
s5_clean() { return 0; }
systemctl() { :; }
mock_reconfig() { return 0; }
installed() { return 0; }
model_is() { return 0; }
''' + restore + '\nrestore\n'
        r = subprocess.run(['bash', '-c', shell, 'review', str(HERE / 'qlib.sh'), str(self.root)], capture_output=True, text=True)
        self.assertIn('STEP restart-final FAIL', r.stdout)
        self.assertIn('settings write failed', r.stdout)


if __name__ == '__main__':
    unittest.main()
