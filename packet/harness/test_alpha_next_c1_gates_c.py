#!/usr/bin/env python3
"""tools/alpha_next_c1_gates_c.sh (lead, 3 Nov ~05:45 CET, rule 5: before each container step docker ps must show no
returning-append-* container; after chain a1's overlap with returning-append r1).

gates_c is gates_b with a header, the wait_clear block, and one wait_clear call before each container step (bundle self-run, engine
self-run, its rerun, the negative observation, the negative); every other line is gates_b's, so no step is added, dropped or
shortened. The wait_clear block itself, cut from the script and run in bash with docker ps replaced: it returns at once when no
returning-append container is up (other containers do not block), waits while one is up or while docker ps fails, and writes
waiting-before / clear-before lines with no rc field to the steps file. CPU only, temporary files; no docker.
"""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / 'tools/alpha_next_c1_gates_b.sh'
C = ROOT / 'tools/alpha_next_c1_gates_c.sh'
CONTAINER_STEPS = ['bundle-selfrun', 'engine-selfrun', 'engine-selfrun-2', 'engine-negative-observe', 'engine-negative']


def block():
    text = C.read_text()
    start = text.index('# >>> wait_clear')
    end = text.index('# <<< wait_clear')
    return text[start:end]


def run_wait(outputs, step='bundle-selfrun'):
    """wait_clear STEP with clear_ps replaced: each call prints the next entry of OUTPUTS (None: docker ps fails)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, out in enumerate(outputs):
            (tmp / ('ps%d' % i)).write_text('FAIL' if out is None else out)
        script = block() + '''
echo 0 >"$PS_DIR/n"
clear_ps() {  # runs in wait_clear's command substitution (a subshell): the call count lives in a file
  local n; n=$(cat "$PS_DIR/n"); echo $((n + 1)) >"$PS_DIR/n"
  [[ $(cat "$PS_DIR/ps$n") == FAIL ]] && { echo 'sudo: a password is required' >&2; return 1; }; cat "$PS_DIR/ps$n"; }
wait_clear "$1"
echo "calls=$(cat "$PS_DIR/n")"
'''
        steps = tmp / 'steps'
        got = subprocess.run(['bash', '-c', script, 'wait', step], capture_output=True, text=True, timeout=60,
                             env=dict(STEPS=str(steps), PS_DIR=str(tmp), CLEAR_POLL_S='0', PATH='/usr/bin:/bin'))
        return got, steps.read_text().splitlines() if steps.exists() else []


class Derivation(unittest.TestCase):
    def test_gates_c_is_gates_b_plus_the_header_the_block_and_one_wait_per_container_step(self):
        b = B.read_text().splitlines()
        c = C.read_text().splitlines()
        self.assertEqual(c[0], b[0])
        self.assertTrue(all(line.startswith('#') for line in c[1:6]))
        body = c[6:]
        start = body.index('# >>> wait_clear (tools/alpha_next_c1_gates_c.sh; harness/test_alpha_next_c1_gates_c.py runs this block)')
        end = body.index('# <<< wait_clear')
        body = body[:start] + body[end + 1:]
        waits = [line for line in body if line.strip().startswith('wait_clear ')]
        self.assertEqual([w.split()[1] for w in waits], ['bundle-selfrun', 'engine-selfrun', 'engine-negative-observe',
                                                         'engine-negative'])
        rest = [line for line in body if not line.strip().startswith('wait_clear ')]
        rerun_b = [line for line in b if 'engine engine-selfrun-2 engine-2' in line]
        rerun_c = [line for line in rest if 'engine engine-selfrun-2 engine-2' in line]
        self.assertEqual(len(rerun_b), 1)
        self.assertEqual(rerun_c, [rerun_b[0].replace('engine engine-selfrun-2 engine-2;',
                                                      'wait_clear engine-selfrun-2; engine engine-selfrun-2 engine-2;')])
        self.assertEqual([line.replace('wait_clear engine-selfrun-2; ', '') for line in rest], b[1:])
        self.assertEqual(subprocess.run(['bash', '-n', str(C)]).returncode, 0)

    def test_each_wait_comes_right_before_its_container_launch(self):
        c = C.read_text().splitlines()
        launches = {'bundle-selfrun': 'tools/native43_bundle_selfrun.py', 'engine-selfrun': 'engine engine-selfrun engine',
                    'engine-negative-observe': '--negative --observe', 'engine-negative': '--negative --bundle'}
        for i, line in enumerate(c):
            if line.strip().startswith('wait_clear '):
                self.assertIn(launches[line.split()[1]], c[i + 1])
        self.assertEqual([line for line in c if 'sudo -n docker' in line],                           # the chain's only docker call
                         ["clear_ps() { sudo -n docker ps --format '{{.Names}}'; }"])


class WaitClear(unittest.TestCase):
    def test_returns_at_once_when_no_returning_append_container_is_up(self):
        got, steps = run_wait(['alpha-next-c1-hub-cpu-x\ncoolify-proxy\nreturning-appendix\n'])
        self.assertEqual(got.returncode, 0, got.stderr)
        self.assertIn('calls=1', got.stdout)
        self.assertEqual(len(steps), 1)
        self.assertRegex(steps[0], r'^\d\d:\d\d:\d\dZ clear-before bundle-selfrun waited \d+s loadavg [\d.]+ [\d.]+ [\d.]+$')

    def test_waits_while_returning_append_is_up_and_while_docker_ps_fails(self):
        got, steps = run_wait(['returning-append-hub-cpu-1\nforgejo\n', 'returning-append-hub-cpu-1\n', None, None,
                               'returning-append-engine-cpu-1-dense\n', 'forgejo\n'], step='engine-negative')
        self.assertEqual(got.returncode, 0, got.stderr)
        self.assertIn('calls=6', got.stdout)
        self.assertEqual([re.sub(r'^\S+ ', '', s) for s in steps[:-1]],
                         ['waiting-before engine-negative: returning-append-hub-cpu-1 ',
                          'waiting-before engine-negative: docker-ps-failed',
                          'waiting-before engine-negative: returning-append-engine-cpu-1-dense '])
        self.assertIn(' clear-before engine-negative waited ', steps[-1])
        self.assertFalse(any(' rc=' in s for s in steps))     # the steps file's rc lines stay the chain's own

    def test_every_container_step_is_named(self):
        for step in CONTAINER_STEPS:
            got, steps = run_wait([''], step=step)
            self.assertEqual(got.returncode, 0, got.stderr)
            self.assertIn(' clear-before %s waited ' % step, steps[0])


if __name__ == '__main__':
    unittest.main()
