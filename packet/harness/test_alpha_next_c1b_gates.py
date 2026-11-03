#!/usr/bin/env python3
"""tools/alpha_next_c1b_gates.sh (chain a for alpha-next-c1b): derivation from tools/alpha_next_c1_gates_c.sh and the wait_clear block.

The c1b chain is gates_c's body (from 'set -u') with exactly the declared swaps below (each matching exactly once in gates_c and its
result exactly once in c1b), the vm-carry step inserted right after the bundle names are fixed (after build), and the delivery call
replaced by a refusal (no c1b delivery tool). The header is new and all comments. Also checked: every container launch is preceded
by its wait_clear, the chain's only docker call is clear_ps, the focused delta is tools/alpha_next_c1b_spec.FOCUSED_DELTA (read from
the text), vm-carry stops the chain on any non-zero rc, and the wait_clear block itself (cut from the script and run in bash with
docker ps replaced) blocks while a returning-append-* OR an append-yield-* container is up, or docker ps fails, and returns when
neither is (other names, e.g. returning-appendix, do not block). CPU only, temporary files; no docker.
"""
import ast
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / 'tools/alpha_next_c1_gates_c.sh'
G = ROOT / 'tools/alpha_next_c1b_gates.sh'
CONTAINER_STEPS = ['bundle-selfrun', 'engine-selfrun', 'engine-selfrun-2', 'engine-negative-observe', 'engine-negative']
SWAPS = [('unset ALPHA_NEXT_C1_PARENT   # development pins are for tests and dry runs only; the chain reads the committed pin\n',
          'unset ALPHA_NEXT_C1_PARENT ALPHA_NEXT_C1B_PARENT   # development pins are for tests and dry runs only; the chain reads the '
          'committed pins\n'),
         ('# >>> wait_clear (tools/alpha_next_c1_gates_c.sh; harness/test_alpha_next_c1_gates_c.py runs this block)\n',
          '# >>> wait_clear (tools/alpha_next_c1b_gates.sh; harness/test_alpha_next_c1b_gates.py runs this block)\n'),
         ('      names=$(grep \'^returning-append-\' <<<"$names" | tr \'\\n\' \' \')\n',
          '      names=$(grep -E \'^(returning-append-|append-yield-)\' <<<"$names" | tr \'\\n\' \' \')\n'),
         ('python3 -B tools/alpha_next_c1_parent_pin.py --pinned', 'python3 -B tools/alpha_next_c1b_parent_pin.py --pinned'),
         ('python3 -B tools/alpha_next_c1_review_pins.py --check', 'python3 -B tools/alpha_next_c1b_review_pins.py --check'),
         ('    test_e3_probe_session test_e3_native_startup_binding test_e3_wire_v1 test_e3_report_validation_cost 2>&1; echo "rc=$?"; '
          '} >"$1"\n',
          '    test_e3_probe_session test_e3_native_startup_binding test_e3_wire_v1 test_e3_report_validation_cost \\\n'
          '    test_e3_c1_open_rows test_alpha_next_c1b_queue test_alpha_next_c1_send_gate test_alpha_next_c1_defer_control_b 2>&1; '
          'echo "rc=$?"; } >"$1"\n'),
         ('python3 -B -m unittest -v test_alpha_next_composition test_alpha_next_c1_pack 2>&1',
          'python3 -B -m unittest -v test_alpha_next_composition test_alpha_next_c1_pack test_alpha_next_c1b_pack 2>&1'),
         ('python3 -B tools/build_alpha_next_c1.py --spec "$SPEC" --prepare-review',
          'python3 -B tools/build_alpha_next_c1b.py --spec "$SPEC" --prepare-review'),
         ('git commit -q -m "alpha-next-c1 build ($TAG): review prepared over $HEAD',
          'git commit -q -m "alpha-next-c1b build ($TAG): review prepared over $HEAD'),
         ('python3 -B tools/build_alpha_next_c1.py --spec "$SPEC" --out "$OUT"',
          'python3 -B tools/build_alpha_next_c1b.py --spec "$SPEC" --out "$OUT"'),
         ('HELD=$OUT/q18a-alpha-next-c1-bundle-$P12\nRUNNER=$OUT/q18a-alpha-next-c1-owner-run-$P12.py\n',
          'HELD=$OUT/q18a-alpha-next-c1b-bundle-$P12\nRUNNER=$OUT/q18a-alpha-next-c1b-owner-run-$P12.py\n'),
         ('--name "alpha-next-c1-hub-cpu-$P12"', '--name "alpha-next-c1b-hub-cpu-$P12"'),
         ('# Hub-side queue pins (tools/alpha_next_c1_queue_pins.json)',
          '# Hub-side queue pins (tools/alpha_next_c1b_queue_pins.json)'),
         ('python3 -B tools/alpha_next_c1_queue_pins_write.py --out "$OUT" ${PINS_SUPERSEDE',
          'python3 -B tools/alpha_next_c1b_queue_pins_write.py --out "$OUT" ${PINS_SUPERSEDE'),
         ('python3 -B tools/alpha_next_c1_queue_pins_write.py --out "$OUT" --check',
          'python3 -B tools/alpha_next_c1b_queue_pins_write.py --out "$OUT" --check'),
         ('# engine DIR LABEL: one engine self-run attempt into $OUT/DIR, container alpha-next-c1-LABEL-cpu-<pin>',
          '# engine DIR LABEL: one engine self-run attempt into $OUT/DIR, container alpha-next-c1b-LABEL-cpu-<pin>'),
         ('engine() { python3 -B tools/alpha_next_c1_engine_selfrun_b.py --spec "$SPEC" --tree "$WT" --bundle "$HELD" --out "$OUT/$1" '
          '\\\n'
          '    --name "alpha-next-c1-$2-cpu-$P12"',
          'engine() { python3 -B tools/alpha_next_c1b_engine_selfrun.py --spec "$SPEC" --tree "$WT" --bundle "$HELD" --out "$OUT/$1" '
          '\\\n'
          '    --name "alpha-next-c1b-$2-cpu-$P12"'),
         ('python3 -B tools/alpha_next_c1_engine_selfrun_b.py --compare "$ENGINE"',
          'python3 -B tools/alpha_next_c1b_engine_selfrun.py --compare "$ENGINE"'),
         ('    tools/alpha_next_c1_negative_pins.json; then', '    tools/alpha_next_c1b_negative_pins.json; then'),
         ('  python3 -B tools/alpha_next_c1_engine_selfrun_b.py --negative --observe',
          '  python3 -B tools/alpha_next_c1b_engine_selfrun.py --negative --observe'),
         ('--name "alpha-next-c1-negative-observe-cpu-$P12"', '--name "alpha-next-c1b-negative-observe-cpu-$P12"'),
         ('echo "pins awaiting review: tools/alpha_next_c1_negative_pins_write.py --observed $OUT/engine-negative-observe/result.json '
          '--log $OUT/engine-negative-observe-stdout.log --classify REVIEWED.json --discovery TEXT"',
          'echo "pins awaiting review: tools/alpha_next_c1b_negative_pins_write.py --observed $OUT/engine-negative-observe/result.json '
          '--log $OUT/engine-negative-observe-stdout.log [--classify REVIEWED.json] --discovery TEXT"'),
         ('python3 -B tools/alpha_next_c1_engine_selfrun_b.py --negative --bundle',
          'python3 -B tools/alpha_next_c1b_engine_selfrun.py --negative --bundle'),
         ('--name "alpha-next-c1-negative-cpu-$P12"', '--name "alpha-next-c1b-negative-cpu-$P12"'),
         ('python3 -B tools/alpha_next_c1_overlay_tamper.py', 'python3 -B tools/alpha_next_c1b_overlay_tamper.py'),
         ('python3 -B tools/alpha_next_c1_clock_scan.py', 'python3 -B tools/alpha_next_c1b_clock_scan.py'),
         ('python3 -B tools/alpha_next_c1_precheck_conf.py', 'python3 -B tools/alpha_next_c1b_precheck_conf.py'),
         ('python3 -B tools/alpha_next_c1_staging_check.py', 'python3 -B tools/alpha_next_c1b_staging_check.py')]
VM_CARRY_ANCHOR = 'HELD=$OUT/q18a-alpha-next-c1b-bundle-$P12\nRUNNER=$OUT/q18a-alpha-next-c1b-owner-run-$P12.py\n'
VM_CARRY = ('\n'
 "# VM carry-over (lead's rule): every module VM <vm-id> imported from c1's mounted overlay (72, plain = stage) is byte-identical "
 'in this\n'
 '# bundle at the origin the launcher mounts it. Any differing, unmapped or missing row stops the chain: tell the lead.\n'
 'python3 -B tools/alpha_next_c1b_vm_carry.py --bundle "$HELD" --out "$OUT/vm-carry.json" >"$OUT/vm-carry-stdout.log" 2>&1\n'
 'RC=$?; [[ $RC == 0 ]] || echo "vm-carry FAILED (rc $RC): see $OUT/vm-carry.json problems; the lead must be told" >>"$STEPS"\n'
 'step vm-carry "$RC"\n')
DELIVERY_REFUSAL = ("# No alpha-next-c1b delivery tool exists in this pack (tools/alpha_next_c1_delivery.py reads c1's freeze record); refuse.\n"
 'echo \'delivery refused: no alpha-next-c1b delivery tool yet (VM_RECORD set)\' >>"$STEPS"\n'
 'step delivery 6\n')


def body(path):
    text = path.read_text()
    return text[text.index('set -u\n'):]


def derive():
    text = body(C)
    for old, new in SWAPS:
        assert text.count(old) == 1, old[:80]
        text = text.replace(old, new)
    assert text.count(VM_CARRY_ANCHOR) == 1
    text = text.replace(VM_CARRY_ANCHOR, VM_CARRY_ANCHOR + VM_CARRY)
    start = text.index('python3 -B tools/alpha_next_c1_delivery.py')
    assert text[start:].endswith('step delivery $?\necho DONE >>"$STEPS"\n')
    return text[:start] + DELIVERY_REFUSAL


def block():
    text = G.read_text()
    return text[text.index('# >>> wait_clear'):text.index('# <<< wait_clear')]


def run_wait(outputs, step='bundle-selfrun'):
    """wait_clear STEP with clear_ps replaced: each call prints the next entry of OUTPUTS (None: docker ps fails)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, out in enumerate(outputs):
            (tmp / ('ps%d' % i)).write_text('FAIL' if out is None else out)
        script = block() + '''
echo 0 >"$PS_DIR/n"
clear_ps() {
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
    def test_c1b_is_gates_c_with_the_declared_swaps_vm_carry_and_the_delivery_refusal(self):
        self.assertEqual(body(G), derive())
        for old, new in SWAPS:
            self.assertEqual(body(G).count(new), 1, new[:80])
        head = G.read_text()[:G.read_text().index('set -u\n')].splitlines()
        self.assertEqual(head[0], '#!/usr/bin/env bash')
        self.assertTrue(all(line.startswith('#') for line in head[1:]))
        self.assertEqual(subprocess.run(['bash', '-n', str(G)]).returncode, 0)

    def test_no_c1_tool_left_but_the_two_declared_and_the_shared_ones(self):
        code = '\n'.join(l for l in body(G).splitlines() if not l.lstrip().startswith('#'))   # what runs, comments aside
        tools = set(re.findall(r'tools/([A-Za-z0-9_./-]+\.(?:py|sh|json))', code))
        c1 = sorted(t for t in tools if t.startswith('alpha_next_c1_'))
        self.assertEqual(c1, ['alpha_next_c1_composition.py', 'alpha_next_c1_rebind_default_off.py'])
        self.assertNotIn('alpha_next_c1_engine_selfrun_b.py', tools)

    def test_focused_delta_is_the_specs(self):
        tree = ast.parse((ROOT / 'tools/alpha_next_c1b_spec.py').read_text())
        delta = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and getattr(n.targets[0], 'id', None) == 'FOCUSED_DELTA'][0]
        line = [l for l in body(G).splitlines() if 'test_e3_c1_open_rows' in l]
        self.assertEqual(len(line), 1)
        self.assertEqual(line[0].split()[:len(delta)], delta)

    def test_vm_carry_right_after_build_and_before_any_container(self):
        lines = body(G).splitlines()
        steps = [re.match(r'\s*step (\S+)', l).group(1) for l in lines if re.match(r'\s*step \S+', l)]
        self.assertEqual(steps[steps.index('build') + 1], 'vm-carry')
        first_wait = min(i for i, l in enumerate(lines) if l.strip().startswith('wait_clear '))
        self.assertLess(min(i for i, l in enumerate(lines) if 'alpha_next_c1b_vm_carry.py' in l), first_wait)

    def test_each_wait_comes_right_before_its_container_launch(self):
        c = G.read_text().splitlines()
        launches = {'bundle-selfrun': 'tools/native43_bundle_selfrun.py', 'engine-selfrun': 'engine engine-selfrun engine',
                    'engine-negative-observe': '--negative --observe', 'engine-negative': '--negative --bundle'}
        waits = [line.split()[1] for line in c if line.strip().startswith('wait_clear ')]
        self.assertEqual(waits, ['bundle-selfrun', 'engine-selfrun', 'engine-negative-observe', 'engine-negative'])
        for i, line in enumerate(c):
            if line.strip().startswith('wait_clear '):
                self.assertIn(launches[line.split()[1]], c[i + 1])
        self.assertEqual(len([l for l in c if 'wait_clear engine-selfrun-2; engine engine-selfrun-2 engine-2;' in l]), 1)
        self.assertEqual([line for line in c if 'sudo -n docker' in line],
                         ["clear_ps() { sudo -n docker ps --format '{{.Names}}'; }"])


class WaitClear(unittest.TestCase):
    def test_returns_at_once_when_neither_prefix_is_up(self):
        got, steps = run_wait(['alpha-next-c1b-hub-cpu-x\ncoolify-proxy\nreturning-appendix\nappend-yieldx\nmy-append-yield-1\n'])
        self.assertEqual(got.returncode, 0, got.stderr)
        self.assertIn('calls=1', got.stdout)
        self.assertEqual(len(steps), 1)
        self.assertRegex(steps[0], r'^\d\d:\d\d:\d\dZ clear-before bundle-selfrun waited \d+s loadavg [\d.]+ [\d.]+ [\d.]+$')

    def test_blocks_on_append_yield_alone(self):
        got, steps = run_wait(['append-yield-hub-cpu-1\nforgejo\n', 'append-yield-hub-cpu-1\n', 'forgejo\n'], step='engine-selfrun')
        self.assertEqual(got.returncode, 0, got.stderr)
        self.assertIn('calls=3', got.stdout)
        self.assertEqual([re.sub(r'^\S+ ', '', s) for s in steps[:-1]],
                         ['waiting-before engine-selfrun: append-yield-hub-cpu-1 '])
        self.assertIn(' clear-before engine-selfrun waited ', steps[-1])

    def test_blocks_on_returning_append_then_append_yield_and_while_docker_ps_fails(self):
        got, steps = run_wait(['returning-append-hub-cpu-1\nforgejo\n', None,
                               'returning-append-r2\nappend-yield-engine-cpu-1-dense\n', 'append-yield-engine-cpu-1-dense\n',
                               'forgejo\n'], step='engine-negative')
        self.assertEqual(got.returncode, 0, got.stderr)
        self.assertIn('calls=5', got.stdout)
        self.assertEqual([re.sub(r'^\S+ ', '', s) for s in steps[:-1]],
                         ['waiting-before engine-negative: returning-append-hub-cpu-1 ',
                          'waiting-before engine-negative: docker-ps-failed',
                          'waiting-before engine-negative: returning-append-r2 append-yield-engine-cpu-1-dense ',
                          'waiting-before engine-negative: append-yield-engine-cpu-1-dense '])
        self.assertIn(' clear-before engine-negative waited ', steps[-1])
        self.assertFalse(any(' rc=' in s for s in steps))

    def test_every_container_step_is_named(self):
        for step in CONTAINER_STEPS:
            got, steps = run_wait([''], step=step)
            self.assertEqual(got.returncode, 0, got.stderr)
            self.assertIn(' clear-before %s waited ' % step, steps[0])


if __name__ == '__main__':
    unittest.main()
