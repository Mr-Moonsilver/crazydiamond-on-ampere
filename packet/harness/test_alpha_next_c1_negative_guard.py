#!/usr/bin/env python3
"""alpha-next-c1 negative control, bounded and fail-closed (lead 2 Nov ~23:50 CET; HARNESS, after alpha-next's g5 negative hung).

tools/alpha_next_c1_engine_selfrun.py runs the negative's container client under alpha-next's shared guard
(harness/alpha_next_wall_guard.py, 55b21ae5) and judges INC first:
  * a deliberately hanging suite (a unittest blocked on a pipe nobody writes, as the g5 witness was) run through the real
    guarded container call is stopped at the wall limit, recorded TIMEOUT, and judged INC, never OBSERVED or an expected FAIL;
  * the judge fails closed: container rc 137 with suites {} (alpha-next's r2 defect), a timeout, a missing suite, skips present
    or unknown, no guard record, or an unreadable container record: INC; only a complete record is OBSERVED / PASS / FAIL;
  * end to end, negative() with the container replaced writes verdict INC, classification INC, and exits INC_RC (4), with or
    without --observe; the gate chain stops INC on rc 4.
CPU only: subprocesses of this interpreter, no docker, no box. Leaves: alpha_next_c1_engine_fixture_leaves.json
(negative-guard-fakes).
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(HERE))
import alpha_next_c1_engine_selfrun as sr  # noqa: E402
import alpha_next_wall_guard as wall_guard  # noqa: E402

SUITES = list(sr.NEGATIVE_SUITES)
HANG = textwrap.dedent('''
    import os, sys, unittest
    PID_FILE = sys.argv[1]
    class Hang(unittest.TestCase):
        def test_waits_on_a_pipe_nobody_writes(self):
            r, w = os.pipe()
            open(PID_FILE, 'w').write(str(os.getpid()))
            os.read(r, 1)          # blocks forever, as the g5 witness did
    unittest.main(argv=['hang'])
''')


def complete(skips=0):
    """A container result every negative suite reported in (import failures as the loader reports them, the queue suite ran)."""
    outcomes = {'unittest.loader._FailedTest.' + n: ['error', "ModuleNotFoundError: No module named 'stage_overlap'"]
                for n in SUITES[:-1]}
    outcomes[SUITES[-1] + '.Table.test_one'] = ['ok', '']
    return dict(skips=skips, tests=len(outcomes), outcomes=outcomes)


EXITED = dict(rc=0, timed_out=False, limit_s=sr.NEGATIVE_WALL_LIMIT_S, wall_s=12.0, outcome='exited')


class HangingSuite(unittest.TestCase):
    """The real guarded call (sr.guarded_container -> step_a.container's subprocess.call) over a suite that never returns."""

    def run_hang(self, stop):
        with tempfile.TemporaryDirectory() as tmp:
            script, pid_file, log = Path(tmp, 'hang.py'), Path(tmp, 'pid'), Path(tmp, 'suites.log')
            script.write_text(HANG)
            stopped = []

            def container(*args, **kwargs):
                # step_a.container's own call shape: the client with --name, stdout to the log, stderr folded in.
                with open(log, 'w') as f:
                    rc = sr.step_a.subprocess.call([sys.executable, str(script), str(pid_file), '--name', 'c1-neg-test'],
                                                   stdout=f, stderr=subprocess.STDOUT)
                return rc, {}, {}, ['client'], 0.0

            def stop_container(name):
                stopped.append(name)
                stop(pid_file)
            start = time.monotonic()
            with mock.patch.object(sr.step_a, 'container', container), \
                    mock.patch.object(sr, 'NEGATIVE_WALL_LIMIT_S', 2), mock.patch.object(sr, 'NEGATIVE_STOP_GRACE_S', 2), \
                    mock.patch.object(sr, '_stop_container', stop_container):
                got, guard, error = sr.guarded_container()
            wall = time.monotonic() - start
            self.assertIs(sr.step_a.subprocess, subprocess)        # the module's subprocess is restored
            self.assertTrue(pid_file.exists(), 'the hanging test never started')
            return got, guard, error, stopped, wall

    def assert_inc_timeout(self, got, guard, error):
        rc, result = got[0], got[1]
        verdict, classification, inc, ok = sr.negative_judgement(rc, result, guard, [], observe=True, error=error)
        self.assertEqual((verdict, classification, ok), ('INC', 'INC', False))
        self.assertTrue(inc[0].startswith('TIMEOUT: wall limit 2s exceeded'), inc)
        self.assertIn('suites-not-reported', inc)
        verdict, _, _, _ = sr.negative_judgement(rc, result, guard, [], observe=False, error=error)
        self.assertEqual(verdict, 'INC')                           # never an expected FAIL

    def test_stopped_at_the_limit_by_the_container_stop(self):
        def stop(pid_file):
            os.kill(int(pid_file.read_text()), signal.SIGTERM)     # what docker stop does to the container's process
        got, guard, error, stopped, wall = self.run_hang(stop)
        self.assertEqual(stopped, ['c1-neg-test'])
        self.assertEqual((guard['timed_out'], guard['outcome'], guard['limit_s']), (True, wall_guard.TIMEOUT, 2))
        self.assertNotEqual(guard['rc'], 0)
        self.assertLess(wall, 30)
        self.assertIsNone(error)
        self.assert_inc_timeout(got, guard, error)

    def test_client_killed_after_the_grace_when_the_stop_does_nothing(self):
        got, guard, error, stopped, wall = self.run_hang(lambda pid_file: None)
        self.assertEqual(stopped, ['c1-neg-test'])
        self.assertEqual((guard['timed_out'], guard['rc']), (True, -signal.SIGKILL))
        self.assertLess(wall, 30)
        self.assert_inc_timeout(got, guard, error)


class Judge(unittest.TestCase):
    def judge(self, rc, result, guard=EXITED, reasons=(), observe=True, error=None):
        return sr.negative_judgement(rc, result, guard, list(reasons), observe, error)

    def assertInc(self, got, reason):
        verdict, classification, inc, ok = got
        self.assertEqual((verdict, classification, ok), ('INC', 'INC', False))
        self.assertTrue(any(r.startswith(reason) for r in inc), (reason, inc))

    def test_rc_137_with_empty_suites_is_inc_not_observed(self):     # alpha-next's r2 defect
        for result in ({}, dict(skips=0, outcomes={}), dict(skips=None, outcomes=None)):
            got = self.judge(137, result, guard=dict(EXITED, rc=137))
            self.assertInc(got, 'container-rc-unexpected:137')
            self.assertIn('suites-not-reported', got[2])

    def test_timeout_is_inc_even_with_a_complete_record(self):
        timed = dict(rc=-9, timed_out=True, limit_s=2700, wall_s=2790.0, outcome=wall_guard.TIMEOUT)
        self.assertInc(self.judge(0, complete(), guard=timed), 'TIMEOUT')
        self.assertInc(self.judge(0, complete(), guard=timed, observe=False), 'TIMEOUT')

    def test_a_missing_suite_is_inc(self):
        result = complete()
        del result['outcomes']['unittest.loader._FailedTest.' + SUITES[0]]
        self.assertInc(self.judge(0, result), 'suite-not-reported:' + SUITES[0])

    def test_skips_present_or_unknown_are_inc(self):
        self.assertInc(self.judge(0, complete(skips=1)), 'skips:1')
        self.assertInc(self.judge(0, complete(skips=None)), 'skips:None')

    def test_no_guard_record_or_an_unreadable_container_record_is_inc(self):
        self.assertInc(self.judge(0, complete(), guard=None), 'container-not-run-under-the-wall-guard')
        self.assertInc(self.judge(0, complete(), error='CalledProcessError: docker inspect'), 'container-record-unreadable')

    def test_only_a_complete_record_is_judged(self):
        self.assertEqual(self.judge(0, complete())[:3], ('OBSERVED', None, []))
        self.assertEqual(self.judge(0, complete(), observe=False)[:4], ('PASS', None, [], True))
        self.assertEqual(self.judge(0, complete(), reasons=['negative-test-differs:x'], observe=False)[:3],
                         ('FAIL', 'HARNESS-OR-ENGINE-UNCLASSIFIED', []))


class EndToEnd(unittest.TestCase):
    """negative() with only the container (and the parent's frozen record) replaced: the record and the exit code."""

    def run_negative(self, container, observe):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for d in ('bundle', 'overlay', 'sources', 'out', 'place'):
                (tmp / d).mkdir()
            for d in ('bundle', 'overlay'):
                (tmp / d / 'SHA256SUMS').write_text('0' * 64 + '  x\n')
            (tmp / 'place' / 'result.json').write_text('{}\n')
            pins = tmp / 'pins.json'
            pins.write_text(json.dumps(dict(status='AWAITING-OBSERVATION', frozen_record_sha256=sr.sha(tmp / 'place/result.json'),
                                            tests={}, cases={})))
            with mock.patch.object(sr, 'frozen', lambda: tmp / 'place'), mock.patch.object(sr, 'PINS', pins), \
                    mock.patch.object(sr.step_a, 'container', container), \
                    contextlib_redirect():
                rc = sr.negative(tmp / 'bundle', tmp / 'overlay', tmp / 'sources', tmp / 'out', 'c1-neg-test', observe)
            return rc, json.loads((tmp / 'out' / 'result.json').read_text())

    def fake(self, rc, result, raise_after=None):
        def container(bundle, overlay, sources, out, name, modes, suites, hide_candidate=False):
            self.assertEqual((tuple(suites), hide_candidate), (sr.NEGATIVE_SUITES, True))
            with open(Path(out) / 'suites.log', 'w') as f:
                got = sr.step_a.subprocess.call([sys.executable, '-c', 'import sys; sys.exit(%d)' % rc, '--name', name],
                                                stdout=f, stderr=subprocess.STDOUT)
            if raise_after:
                raise subprocess.CalledProcessError(1, ['sudo', '-n', 'docker', 'inspect', name])
            return got, result, {}, ['client'], 0.5
        return container

    def test_rc_137_with_suites_empty_writes_inc_and_exits_4(self):
        for observe in (True, False):
            with self.subTest(observe=observe):
                rc, record = self.run_negative(self.fake(137, {}), observe)
                self.assertEqual((rc, record['verdict'], record['classification']), (sr.INC_RC, 'INC', 'INC'))
                self.assertIn('container-rc-unexpected:137 (expected 0)', record['inc_reasons'])
                self.assertIn('suites-not-reported', record['inc_reasons'])
                self.assertEqual((record['wall_guard']['rc'], record['wall_guard']['timed_out']), (137, False))
                self.assertNotIn('observed', record)

    def test_an_unreadable_container_record_writes_inc(self):
        rc, record = self.run_negative(self.fake(0, None, raise_after=True), True)
        self.assertEqual((rc, record['verdict']), (sr.INC_RC, 'INC'))
        self.assertTrue(any(r.startswith('container-record-unreadable:CalledProcessError') for r in record['inc_reasons']))

    def test_a_complete_observation_is_observed(self):
        rc, record = self.run_negative(self.fake(0, complete()), True)
        self.assertEqual((rc, record['verdict'], record['inc_reasons'], record['classification']), (1, 'OBSERVED', [], None))

    def test_the_chain_stops_inc_on_rc_4(self):
        text = (ROOT / 'tools/alpha_next_c1_gates.sh').read_text()
        self.assertIn('if [[ $ORC == 4 ]]; then', text)
        self.assertIn('step engine-negative-observe 4; fi', text)
        self.assertIn("[[ $RC == 4 ]] && echo 'engine-negative INC", text)


def contextlib_redirect():
    import contextlib
    return contextlib.redirect_stdout(open(os.devnull, 'w'))


if __name__ == '__main__':
    unittest.main()
