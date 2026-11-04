#!/usr/bin/env python3
"""tools/attn_meta_gates.sh: the q18a-attn-meta gate chain (c1's chain over the attn-meta tool family).

Text: bash -n clean; the steps in c1's order with attn-meta's two proofs (base-proof, default-off) where c1's composition and rebind
default-off stood, the two selfrun5 container steps after the negative, the stop line last; every container step right after its
wait_clear; the chain's only docker call is clear_ps; focused and the composition test go through the superseded runner with the
base-tree receipt; no chain tool of c1's family is called (each is attn-meta's or a shared unchanged one).
The wait_clear block, cut from the script and run in bash with docker ps replaced: it waits while a container whose name contains
alpha-next-c1 (c1b's included), append-yield or returning-append is up, while docker ps fails, and while the clearance file is
absent (the lead's hold); other containers do not block; waiting-before / clear-before lines carry no rc field. CPU only, temporary
files; no docker.
"""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / 'tools/attn_meta_gates.sh'
ORDER = ['parent-pin', 'base-proof', 'default-off', 'review-pins', 'strict', 'focused', 'composition-test', 'prepare-review',
         'review-commit', 'build', 'bundle-selfrun', 'queue-pins-write', 'queue-pins-check', 'engine-selfrun',
         'engine-comparison', 'engine-negative-observe', 'engine-negative-awaiting-pins', 'engine-negative', '"attn-$MODE"',
         'overlay-tamper', 'clock-scan', 'compile-proof', 'precheck-conf', 'precheck', 'staging-check']
WAITS = ['bundle-selfrun', 'engine-selfrun', 'engine-negative-observe', 'engine-negative', '"attn-$MODE"']
CONTAINER_STEPS = ['bundle-selfrun', 'engine-selfrun', 'engine-selfrun-2', 'engine-negative-observe', 'engine-negative',
                   'attn-tests-c', 'attn-probe-c']


def block():
    text = G.read_text()
    return text[text.index('# >>> wait_clear'):text.index('# <<< wait_clear')]


def run_wait(outputs, step='bundle-selfrun', clear_after=0):
    """wait_clear STEP with clear_ps replaced: call i prints OUTPUTS[i] (None: docker ps fails); the clearance file appears
    after CLEAR_AFTER calls (0: present from the start)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, out in enumerate(outputs):
            (tmp / ('ps%d' % i)).write_text('FAIL' if out is None else out)
        clear = tmp / 'cleared'
        script = block() + '''
echo 0 >"$PS_DIR/n"
[[ $CLEAR_AFTER == 0 ]] && : >"$CLEAR"
clear_ps() {
  local n; n=$(cat "$PS_DIR/n"); echo $((n + 1)) >"$PS_DIR/n"
  [[ $((n + 1)) -ge $CLEAR_AFTER && $CLEAR_AFTER != 0 ]] && : >"$CLEAR"
  [[ $(cat "$PS_DIR/ps$n") == FAIL ]] && { echo 'sudo: a password is required' >&2; return 1; }; cat "$PS_DIR/ps$n"; }
wait_clear "$1"
echo "calls=$(cat "$PS_DIR/n")"
'''
        steps = tmp / 'steps'
        got = subprocess.run(['bash', '-c', script, 'wait', step], capture_output=True, text=True, timeout=60,
                             env=dict(STEPS=str(steps), PS_DIR=str(tmp), CLEAR=str(clear), CLEAR_AFTER=str(clear_after),
                                      CLEAR_POLL_S='0', PATH='/usr/bin:/bin'))
        return got, steps.read_text().splitlines() if steps.exists() else []


class Text(unittest.TestCase):
    def setUp(self):
        self.text = G.read_text()
        self.lines = self.text.splitlines()

    def test_bash_n_and_the_steps_in_order(self):
        self.assertEqual(subprocess.run(['bash', '-n', str(G)]).returncode, 0)
        named = []
        for line in self.lines:
            if line.lstrip().startswith('#'):
                continue
            for name in re.findall(r'(?:^|[;\s])step (\S+) ', line):
                if name not in named and name not in ('strict-2', 'focused-2', 'engine-selfrun-2'):
                    named.append(name)
        self.assertEqual(named, ORDER)
        self.assertEqual(self.lines[-2:], ["echo 'stopped-before-delivery: VM run pending' >>\"$STEPS\"", 'exit 0'])

    def test_each_wait_right_before_its_container_launch(self):
        launches = {'bundle-selfrun': 'tools/native43_bundle_selfrun.py', 'engine-selfrun': 'engine engine-selfrun engine',
                    'engine-negative-observe': '--negative --observe', 'engine-negative': '--negative --bundle',
                    '"attn-$MODE"': 'tools/attn_meta_selfrun5.py'}
        waits = []
        for i, line in enumerate(self.lines):
            if line.strip().startswith('wait_clear '):
                name = line.split()[1]
                waits.append(name)
                self.assertIn(launches[name], self.lines[i + 1])
        self.assertEqual(waits, WAITS)
        self.assertIn('wait_clear engine-selfrun-2; engine engine-selfrun-2 engine-2;', self.text)
        self.assertEqual([line for line in self.lines if 'sudo -n docker' in line],
                         ["clear_ps() { sudo -n docker ps --format '{{.Names}}'; }"])

    def test_the_superseded_runner_carries_focused_and_composition_with_the_receipt(self):
        runs = [line for line in self.lines if 'robust_swap.py' in line and not line.lstrip().startswith('#')]
        self.assertEqual(len(runs), 2)
        self.assertTrue(all('attn_meta_robust_swap.py --base-receipt "$BASE_RECEIPT"' in r for r in runs), runs)
        self.assertIn('BASE_RECEIPT=$EVID/base-proof-$TAG.json', self.text)
        self.assertIn('python3 -B tools/attn_meta_base_proof.py --out "$BASE_RECEIPT"', self.text)
        self.assertIn('python3 -B tools/attn_meta_default_off.py --out "$EVID/default-off-$TAG.json"', self.text)

    def test_no_c1_chain_tool_is_called(self):
        called = set(re.findall(r'tools/([a-z0-9_]+\.(?:py|sh))', '\n'.join(l for l in self.lines if not l.startswith('#'))))
        own = {n for n in called if n.startswith(('attn_meta_', 'build_attn_meta'))}
        shared = called - own
        self.assertEqual(shared, {'alpha_c1_flaky.py', 'native43_bundle_selfrun.py', 'native35_compile_proof.py'})
        for name in called:
            self.assertTrue((ROOT / 'tools' / name).is_file(), name)
        self.assertNotIn('tools/alpha_next_c1_', '\n'.join(l for l in self.lines if not l.lstrip().startswith('#')))


class WaitClear(unittest.TestCase):
    def test_returns_at_once_when_clear_and_the_file_is_there(self):
        got, steps = run_wait(['coolify-proxy\nforgejo\nreturning-x\nalpha-next-c0-x\n'])
        self.assertEqual(got.returncode, 0, got.stderr)
        self.assertIn('calls=1', got.stdout)
        self.assertEqual(len(steps), 1)
        self.assertRegex(steps[0], r'^\d\d:\d\d:\d\dZ clear-before bundle-selfrun waited \d+s loadavg [\d.]+ [\d.]+ [\d.]+$')

    def test_waits_on_each_name_rule_and_on_a_failing_docker_ps(self):
        got, steps = run_wait(['alpha-next-c1b-hub-cpu-1\nforgejo\n', 'append-yield-x\n', None,
                               'returning-append-engine-cpu-1-dense\n', 'forgejo\n'], step='engine-negative')
        self.assertEqual(got.returncode, 0, got.stderr)
        self.assertIn('calls=5', got.stdout)
        self.assertEqual([re.sub(r'^\S+ ', '', s) for s in steps[:-1]],
                         ['waiting-before engine-negative: alpha-next-c1b-hub-cpu-1 ',
                          'waiting-before engine-negative: append-yield-x ',
                          'waiting-before engine-negative: docker-ps-failed',
                          'waiting-before engine-negative: returning-append-engine-cpu-1-dense '])
        self.assertIn(' clear-before engine-negative waited ', steps[-1])
        self.assertFalse(any(' rc=' in s for s in steps))

    def test_holds_until_the_clearance_file_exists(self):
        got, steps = run_wait(['', '', '', ''], clear_after=3)
        self.assertEqual(got.returncode, 0, got.stderr)
        self.assertIn('calls=3', got.stdout)
        self.assertEqual([re.sub(r'^\S+ ', '', s) for s in steps[:-1]], ['waiting-before bundle-selfrun: hold:no-clearance-file'])
        self.assertIn(' clear-before bundle-selfrun waited ', steps[-1])

    def test_every_container_step_is_named(self):
        for step in CONTAINER_STEPS:
            got, steps = run_wait([''], step=step)
            self.assertEqual(got.returncode, 0, got.stderr)
            self.assertIn(' clear-before %s waited ' % step, steps[0])


if __name__ == '__main__':
    unittest.main()
