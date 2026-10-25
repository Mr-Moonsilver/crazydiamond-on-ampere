"""CPU regressions: bounded failures, issue evidence, exact diagnostic selection."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from s5_runtime import Progress, Watchdog
from s5_clock import Inconclusive
from s5_segment import parse_modes, grade_pair
from test_s5_segment import fixture

HERE = Path(__file__).resolve().parent


class RuntimeTests(unittest.TestCase):
    def directory(self):
        return Path(tempfile.mkdtemp(prefix='s5-runtime-', dir=os.environ.get('TMPDIR')))

    def test_issue_order_and_phase_survive(self):
        out = self.directory()/'pair.json'
        p = Progress(out, 1)
        p.update('prio:warmup')
        p.event('prefill','collective-enter',step=0,collective=1)
        p.event('decode','graph-issue',step=0)
        rows = [json.loads(x) for x in p.events.read_text().splitlines()]
        self.assertEqual([x['seq'] for x in rows], [1,2,3])
        self.assertEqual([x.get('lane') for x in rows[1:]], ['prefill','decode'])
        self.assertEqual(json.loads(p.path.read_text())['phase'], 'prio:warmup')

    def test_python_watchdog_raises_and_disarms(self):
        import time
        out = self.directory()/'pair.json'
        w = Watchdog(Progress(out,0),out,0)
        try:
            w.arm(.03,'blocked-join')
            with self.assertRaises(Inconclusive):
                time.sleep(1)
        finally:
            w.close()
        self.assertEqual(w.child.returncode, 0)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0],0)

    def test_hard_watchdog_preserves_rank_failure_and_stops_only_child(self):
        out = self.directory()/'pair.json'
        code = '''import signal, time, sys
from s5_runtime import Progress, Watchdog
w = Watchdog(Progress(sys.argv[1], 1), sys.argv[1], 1, grace=.05)
w.arm(.05, 'prio:measurement')
signal.setitimer(signal.ITIMER_REAL, 0) # simulate Python unable to handle alarm
time.sleep(20)
'''
        p = subprocess.run([sys.executable,'-c',code,str(out)],cwd=HERE,capture_output=True,timeout=5)
        self.assertEqual(p.returncode, -signal.SIGKILL)
        record = json.loads(Path(str(out)+'.rank1.error.json').read_text())
        self.assertEqual(record['grade']['verdict'], 'INCONCLUSIVE')
        self.assertEqual(record['partial_rank']['phase'], 'prio:measurement')
        self.assertIn(b'File',p.stderr) # faulthandler dump

    def test_modes_exact_and_subset_cannot_pass(self):
        self.assertEqual(parse_modes('prio,green24'), ['prio','green24'])
        for invalid in ('','prio,prio','bogus','prio,'):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_modes(invalid)
        d = fixture()
        for r in d['ranks']:
            r['modes'] = {'prio':r['modes']['prio']}
        self.assertEqual(grade_pair(d)['verdict'],'INCONCLUSIVE')

    def test_cpu_plan_filter(self):
        p = subprocess.run([sys.executable,str(HERE/'s5_segment.py'),'--plan','--modes','prio'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout)['modes'], ['prio'])

    def test_conv_and_lane_timing_contract(self):
        src = (HERE/'s5_segment_gpu.py').read_text()
        self.assertIn('compute_causal_conv1d_metadata(starts,',src)
        self.assertIn('metadata=self.conv_metadata',src)
        self.assertNotIn('stamps.cpu()',src)
        self.assertIn('host_stamps.copy_(stamps, non_blocking=True)',src)
        self.assertIn('while not complete.query()',src)
        self.assertIn('backend="pynccl"',src)
        self.assertIn('os._exit(EXIT[verdict])',src)

if __name__ == '__main__':
    unittest.main()
