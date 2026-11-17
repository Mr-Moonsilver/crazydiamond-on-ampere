#!/usr/bin/env python3
"""tools/alpha_next_c1b_gates_b.sh: tools/alpha_next_c1b_gates.sh (chains a and a2) with one rule change, after the 3 Nov container
overlap (a container named flashnext-alpha-c0-append-yield-...-stage-cpu-hub-cpu passed the prefix match): wait_clear blocks while
any container whose name CONTAINS returning-append or append-yield is up. Checked: the derivation (a comment header, the block's
marker line and the one grep line; nothing else), and the wait_clear block cut from the script and run in bash with docker ps
replaced (each fake output is one docker ps call). CPU only, temporary files; no docker.
"""
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_alpha_next_c1b_gates as base  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'tools/alpha_next_c1b_gates.sh'
B = ROOT / 'tools/alpha_next_c1b_gates_b.sh'
SWAPS = [
    ('# >>> wait_clear (tools/alpha_next_c1b_gates.sh; harness/test_alpha_next_c1b_gates.py runs this block)\n',
     '# >>> wait_clear (tools/alpha_next_c1b_gates_b.sh; harness/test_alpha_next_c1b_gates_b.py runs this block)\n'),
    ("      names=$(grep -E '^(returning-append-|append-yield-)' <<<\"$names\" | tr '\\n' ' ')\n",
     "      names=$(grep -E '(returning-append|append-yield)' <<<\"$names\" | tr '\\n' ' ')\n"),
]


def run_wait(outputs, step='bundle-selfrun'):
    """base.run_wait over gates_b's block."""
    original = base.G
    base.G = B
    try:
        return base.run_wait(outputs, step)
    finally:
        base.G = original


class Derivation(unittest.TestCase):
    def test_b_is_a_with_the_one_rule_change(self):
        a, b = A.read_text(), B.read_text()
        for old, new in SWAPS:
            self.assertEqual((a.count(old), b.count(new), b.count(old)), (1, 1, 0), old[:60])
            a = a.replace(old, new)
        head_a, head_b = a.split('\n', 1), b.split('\n', 1)
        self.assertEqual(head_a[0], '#!/usr/bin/env bash')
        extra = head_b[1][:len(head_b[1]) - len(head_a[1])]
        self.assertEqual(head_b[1][len(extra):], head_a[1])                  # the rest byte for byte
        self.assertTrue(extra and all(line.startswith('#') for line in extra.splitlines()), 'header is comments only')
        self.assertEqual(base.body(B), base.body(A).replace(*SWAPS[0]).replace(*SWAPS[1]))

    def test_the_container_steps_and_their_waits_are_as(self):
        c = B.read_text().splitlines()
        self.assertEqual([line.split()[1] for line in c if line.strip().startswith('wait_clear ')],
                         ['bundle-selfrun', 'engine-selfrun', 'engine-negative-observe', 'engine-negative'])
        self.assertEqual([line for line in c if 'sudo -n docker' in line],
                         ["clear_ps() { sudo -n docker ps --format '{{.Names}}'; }"])


class WaitClear(unittest.TestCase):
    def test_returns_at_once_when_no_name_contains_either(self):
        got, steps = run_wait(['alpha-next-c1b-hub-cpu-x\ncoolify-proxy\nforgejo\nappend_yield-underscore\nreturning_append\n'])
        self.assertEqual(got.returncode, 0, got.stderr)
        self.assertIn('calls=1', got.stdout)
        self.assertEqual(len(steps), 1)
        self.assertRegex(steps[0], r'^\d\d:\d\d:\d\dZ clear-before bundle-selfrun waited \d+s loadavg [\d.]+ [\d.]+ [\d.]+$')

    def test_blocks_on_the_name_the_prefix_match_missed(self):
        missed = 'flashnext-alpha-c0-append-yield-07d5f1a5968d-stage-cpu-hub-cpu'
        got, steps = run_wait([missed + '\nforgejo\n', missed + '\n', 'forgejo\n'], step='engine-negative-observe')
        self.assertEqual(got.returncode, 0, got.stderr)
        self.assertIn('calls=3', got.stdout)
        self.assertEqual([re.sub(r'^\S+ ', '', s) for s in steps[:-1]],
                         ['waiting-before engine-negative-observe: %s ' % missed])
        self.assertIn(' clear-before engine-negative-observe waited ', steps[-1])

    def test_blocks_anywhere_in_the_name_and_while_docker_ps_fails(self):
        got, steps = run_wait(['x-returning-append-r2\n', None, 'append-yield-engine-cpu-1-dense\nq-returning-append\n', ''],
                              step='engine-selfrun')
        self.assertEqual(got.returncode, 0, got.stderr)
        self.assertIn('calls=4', got.stdout)
        self.assertEqual([re.sub(r'^\S+ ', '', s) for s in steps[:-1]],
                         ['waiting-before engine-selfrun: x-returning-append-r2 ',
                          'waiting-before engine-selfrun: docker-ps-failed',
                          'waiting-before engine-selfrun: append-yield-engine-cpu-1-dense q-returning-append '])
        self.assertFalse(any(' rc=' in s for s in steps))

    def test_the_old_prefix_cases_still_block(self):
        got, steps = run_wait(['returning-append-hub-cpu-1\n', 'append-yield-hub-cpu-1\n', 'forgejo\n'], step='engine-negative')
        self.assertEqual(got.returncode, 0, got.stderr)
        self.assertIn('calls=3', got.stdout)


if __name__ == '__main__':
    unittest.main()
