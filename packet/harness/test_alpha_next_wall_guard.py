#!/usr/bin/env python3
"""alpha-next negative control guard (harness/alpha_next_wall_guard.py): a hanging suite is stopped at its wall limit and
recorded TIMEOUT with its rc; the judge fails closed: rc 137 with empty suites, a timeout, a missing suite or skips present is
INC, never OBSERVED (and a TIMEOUT never an expected FAIL). CPU, subprocesses only."""
import os
from pathlib import Path
import signal
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_next_wall_guard as guard  # noqa: E402

SUITES = ['test_a', 'test_b', 'test_c']
COMPLETE = dict(skips=0, tests=3, outcomes={'test_a.T.test_one': ['fail', 'AssertionError'],
                                            'unittest.loader._FailedTest.test_b': ['error', "ModuleNotFoundError: No module named 'x'"],
                                            'test_c': ['error', 'ImportError: y']})
HANG = textwrap.dedent('''
    import os, sys, unittest
    class Hang(unittest.TestCase):
        def test_waits_on_a_pipe_nobody_writes(self):
            r, w = os.pipe()
            open(sys.argv[-1], 'w').write(str(os.getpid()))
            os.read(r, 1)          # blocks forever, as the g5 witness did
''')


class HangingSuite(unittest.TestCase):
    def run_hanging(self, stop):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'test_hang_mod.py').write_text(HANG)
            pidfile = Path(tmp, 'pid')
            code = ('import sys, unittest; sys.path.insert(0, %r); sys.argv.append(%r); '
                    'unittest.main(module="test_hang_mod", argv=["x"])' % (tmp, str(pidfile)))
            with open(Path(tmp, 'log'), 'w') as log:
                result = guard.run_bounded([sys.executable, '-B', '-c', code], log, 3, lambda: stop(pidfile), grace_s=10)
            return result

    def test_a_hanging_suite_is_stopped_and_recorded_timeout_with_its_rc(self):
        def stop(pidfile):          # what docker stop does to the container's process
            os.kill(int(pidfile.read_text()), signal.SIGTERM)
        result = self.run_hanging(stop)
        self.assertEqual((result['timed_out'], result['outcome'], result['rc'], result['limit_s']), (True, 'TIMEOUT', -15, 3))
        self.assertLess(result['wall_s'], 3 + 10)
        reasons = guard.inc_reasons(result['rc'], {}, SUITES, result)
        self.assertTrue(reasons[0].startswith('TIMEOUT: wall limit 3s exceeded'), reasons)
        self.assertEqual(guard.verdict(reasons, observe=True, ok=False), 'INC')

    def test_a_stop_that_does_not_stop_still_ends_the_client(self):
        result = self.run_hanging(lambda pidfile: None)
        self.assertEqual((result['timed_out'], result['rc']), (True, -signal.SIGKILL))


class FailClosed(unittest.TestCase):
    def test_rc_137_with_empty_suites_is_inc(self):       # r2's record: container stopped, suites {}
        reasons = guard.inc_reasons(137, {}, SUITES)
        self.assertEqual(reasons, ['container-rc-unexpected:137 (expected 0)', 'suites-not-reported',
                                   'suite-not-reported:test_a', 'suite-not-reported:test_b', 'suite-not-reported:test_c',
                                   'skips:None'])
        self.assertEqual(guard.verdict(reasons, observe=True, ok=False), 'INC')

    def test_a_timeout_is_inc_never_an_expected_fail(self):
        reasons = guard.inc_reasons(0, COMPLETE, SUITES, dict(timed_out=True, limit_s=2700, wall_s=2790.1, rc=-15))
        self.assertEqual(len(reasons), 1)
        self.assertEqual(guard.verdict(reasons, observe=True, ok=False), 'INC')
        self.assertEqual(guard.verdict(reasons, observe=False, ok=False), 'INC')

    def test_a_missing_suite_is_inc(self):
        result = dict(COMPLETE, outcomes={k: v for k, v in COMPLETE['outcomes'].items() if not k.endswith('test_b')})
        reasons = guard.inc_reasons(0, result, SUITES)
        self.assertEqual(reasons, ['suite-not-reported:test_b'])
        self.assertEqual(guard.verdict(reasons, observe=True, ok=False), 'INC')

    def test_skips_present_is_inc(self):
        reasons = guard.inc_reasons(0, dict(COMPLETE, skips=2), SUITES)
        self.assertEqual(reasons, ['skips:2'])
        self.assertEqual(guard.verdict(reasons, observe=True, ok=False), 'INC')

    def test_a_complete_observation_is_observed(self):
        reasons = guard.inc_reasons(0, COMPLETE, SUITES, dict(timed_out=False, rc=0))
        self.assertEqual(reasons, [])
        self.assertEqual(guard.verdict(reasons, observe=True, ok=False), 'OBSERVED')
        self.assertEqual(guard.verdict(reasons, observe=False, ok=True), 'PASS')
        self.assertEqual(guard.verdict(reasons, observe=False, ok=False), 'FAIL')

    def test_in_flight_names_the_hung_test_and_the_steps_without_rc(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'suites.log').write_text('test_x (m.C.test_x) ... ok\ntest_y (m.C.test_y) ... Process ForkProcess-1:\n')
            (out / 'inner.sh').write_text('cd /source/harness\npython3 suites.py\necho $? > /out/suites.rc\n'
                                          'python3 p.py\necho $? > /out/case-a.rc\n')
            self.assertEqual(guard.in_flight(out), dict(last_test='test_y (m.C.test_y)', steps_without_rc=['suites', 'case-a']))


if __name__ == '__main__':
    unittest.main()
