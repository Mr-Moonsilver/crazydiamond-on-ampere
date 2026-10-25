#!/usr/bin/env python3
"""CPU-only tests for the standalone E3 green probe and its unarmed queue.

No torch, no CUDA, no docker and no engine. The frozen verdict is driven with
synthetic device records; the build recipe is the reviewed s5_clock command; the
queue is parsed with ``bash -n`` and checked for its pinned mounts and final line.
These tests cannot establish native readiness, SM containment or GPU behaviour.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

from e3_green_probe import build_command, verdict
from s5_clock import Inconclusive

HERE = Path(__file__).resolve().parent
QUEUE = HERE / 'runs' / 'q17_green_probe.draft.sh'


def record(device=0, requested=44, total=82, **overrides):
    """A complete, passing synthetic device record; callers mutate one field."""
    membership = {'decode_seen': requested, 'prefill_seen': total - requested,
                  'total': total, 'disjoint': True, 'ok': True, 'rounds': [[1]]}
    build = {'available': True, 'abi': 1, 'rc': 0, 'error': None,
             'source_sha256': 'a' * 64, 'library_sha256': 'b' * 64,
             'nvcc_version': 'nvcc: mock', 'elf_needed': ['libcudart.so.12'],
             'log_tail': '', 'command': ['nvcc']}
    describe = {'requested_decode_sms': requested, 'decode_sms': requested,
                'prefill_sms': total - requested, 'total_sms': total,
                'basis': 'measured', 'membership': membership,
                'helper': {'library_sha256': 'b' * 64, 'abi': 1,
                           'helper_source_sha256': 'a' * 64}}
    timings = {'shape': [4096, 4096, 4096], 'dtype': 'bfloat16', 'iterations': 20,
               'ordinary': {'ms': [1.0] * 20, 'total_ms': 20.0, 'median_ms': 1.0},
               'decode_stream': {'ms': [1.0] * 20, 'total_ms': 20.0, 'median_ms': 1.0},
               'prefill_stream': {'ms': [1.0] * 20, 'total_ms': 20.0, 'median_ms': 1.0},
               'concurrent': {'wall_ms': 30.0, 'decode_stream_ms': 25.0,
                              'prefill_stream_ms': 24.0, 'iterations': 20}}
    base = {'schema': 1, 'device': device, 'status': 'ok',
            'requested_decode_sms': requested, 'capability': [8, 6],
            'build': build, 'describe': describe, 'membership': membership,
            'timings': timings, 'errors': []}
    base.update(overrides)
    return base


class VerdictRules(unittest.TestCase):
    def assertVerdict(self, records, want):
        result = verdict(records)
        self.assertEqual(result['verdict'], want, result['reasons'])
        if want != 'PASS':
            self.assertTrue(result['reasons'])
        return result

    def test_pass_requires_every_requested_device(self):
        self.assertVerdict([record(0), record(1), record(2), record(3)], 'PASS')

    def test_no_records_is_inconclusive(self):
        self.assertVerdict([], 'INCONCLUSIVE')

    def test_membership_not_ok_fails(self):
        bad = record()
        bad['membership']['ok'] = False
        self.assertVerdict([bad], 'FAIL')

    def test_count_mismatch_fails_without_tolerance(self):
        bad = record()
        bad['describe']['decode_sms'] -= 1
        self.assertVerdict([bad], 'FAIL')
        bad = record()
        bad['describe']['prefill_sms'] += 1
        self.assertVerdict([bad], 'FAIL')
        bad = record()
        bad['requested_decode_sms'] = 43  # describe still reports 44
        self.assertVerdict([bad], 'FAIL')

    def test_membership_counts_or_disjointness_off_fails(self):
        bad = record()
        bad['membership']['decode_seen'] -= 1
        self.assertVerdict([bad], 'FAIL')
        bad = record()
        bad['membership']['disjoint'] = False
        self.assertVerdict([bad], 'FAIL')

    def test_cuda_launch_and_membership_errors_fail(self):
        for kind in ('cuda', 'launch', 'membership'):
            bad = record()
            bad['errors'] = [{'kind': kind, 'message': 'synthetic'}]
            self.assertVerdict([bad], 'FAIL')

    def test_missing_timings_fail(self):
        for key in ('ordinary', 'decode_stream', 'prefill_stream', 'concurrent'):
            bad = record()
            del bad['timings'][key]
            self.assertVerdict([bad], 'FAIL')

    def test_build_failure_or_refusal_is_inconclusive(self):
        for build in ({'available': False, 'abi': None, 'error': 'nvcc failed'},
                      {'available': False, 'abi': None, 'refused': True,
                       'error': 'refusing to replace existing library'}):
            bad = record()
            bad['build'] = build
            self.assertVerdict([bad], 'INCONCLUSIVE')

    def test_wrong_abi_is_inconclusive(self):
        bad = record()
        bad['build']['abi'] = 2
        self.assertVerdict([bad], 'INCONCLUSIVE')

    def test_capability_mismatch_is_inconclusive(self):
        for capability in ([8, 0], None, (7, 5)):
            bad = record()
            bad['capability'] = capability
            self.assertVerdict([bad], 'INCONCLUSIVE')

    def test_missing_device_is_inconclusive(self):
        bad = record(1)
        bad['status'] = 'missing'
        bad['capability'] = None
        self.assertVerdict([record(0), bad], 'INCONCLUSIVE')

    def test_interruption_is_inconclusive(self):
        bad = record()
        bad['status'] = 'interrupted'
        bad['errors'] = [{'kind': 'interrupted', 'message': 'synthetic'}]
        self.assertVerdict([bad], 'INCONCLUSIVE')

    def test_internal_probe_error_is_inconclusive_not_fail(self):
        bad = record()
        bad['errors'] = [{'kind': 'internal', 'message': 'TypeError'}]
        self.assertVerdict([bad], 'INCONCLUSIVE')

    def test_hard_failure_outranks_an_inconclusive_device(self):
        good = record(0)
        bad = record(1)
        bad['membership']['ok'] = False
        missing = record(2)
        missing['status'] = 'missing'
        missing['capability'] = None
        self.assertVerdict([good, bad, missing], 'FAIL')

    def test_loaded_helper_identity_must_match_the_build(self):
        for field, value in (('library_sha256', 'c' * 64),
                             ('helper_source_sha256', 'c' * 64)):
            bad = record()
            bad['describe']['helper'][field] = value
            self.assertVerdict([bad], 'FAIL')
        bad = record()
        bad['describe']['helper']['abi'] = 2
        self.assertVerdict([bad], 'FAIL')


class BuildRecipe(unittest.TestCase):
    def test_exact_reviewed_sm86_recipe(self):
        command = build_command('flash_s5_green.cu', 'flash_s5_green.so', (8, 6))
        self.assertEqual(command, [
            'nvcc', '-O2', '-std=c++17', '-shared', '-Xcompiler=-fPIC',
            '--generate-code=arch=compute_86,code=sm_86', '--cudart=shared',
            '-lcuda', 'flash_s5_green.cu', '-o', 'flash_s5_green.so'])
        with self.assertRaises(Inconclusive):
            build_command('x.cu', 'x.so', (7, 5))

    def test_plan_is_cpu_only_and_shows_the_same_recipe(self):
        done = subprocess.run([sys.executable, str(HERE / 'e3_green_probe.py'), '--plan'],
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        plan = json.loads(done.stdout)
        self.assertEqual(plan['command'], build_command(
            'native/flash_s5_green.cu', 'flash_s5_green.so', (8, 6)))
        self.assertEqual(plan['devices'], [0, 1, 2, 3])
        self.assertEqual(plan['rounds'], 3)
        self.assertEqual(plan['matmul'], [4096, 4096, 4096])
        self.assertEqual(plan['iterations'], 20)
        self.assertEqual(plan['decode_sms_env'], 'FLASH_S5_DECODE_SMS')

    def test_import_has_no_torch_or_green_context_side_effect(self):
        code = ('import sys; sys.path.insert(0, %r); import e3_green_probe; '
                'assert "torch" not in sys.modules; '
                'assert "green_context" not in sys.modules' % str(HERE))
        done = subprocess.run([sys.executable, '-c', code],
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)


class QueueScript(unittest.TestCase):
    NEEDLES = (
        'fail_early engine-active',
        'echo "INCONCLUSIVE $1"',
        'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b',
        'systemctl show vllm-flash -p ActiveState --value',
        '[[ ! -e $Q ]]',
        'timeout --signal=TERM --kill-after=30s 2700s env Q17_BOUNDED=1 bash "$0"',
        '--gpus all',
        '--network none',
        '-v "$H:/h:ro"',
        '-v "$E3_NATIVE:/n:ro"',
        '-v "$Q:/q"',
        'PYTHONPATH=/n:/h',
        '-e FLASH_S5_DECODE_SMS="$FLASH_S5_DECODE_SMS"',
        'python3 /h/e3_green_probe.py --native /n --out /q --build /q/flash_s5_green.so',
        'Q17_GREEN_PROBE DONE rc=$rc engine-untouched containers-retained',
    )

    def test_bash_syntax(self):
        done = subprocess.run(['bash', '-n', str(QUEUE)],
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_pinned_inputs_mounts_and_final_line(self):
        text = QUEUE.read_text()
        for needle in self.NEEDLES:
            self.assertIn(needle, text)
        for forbidden in ('docker rm', 'docker start', 'systemctl start',
                          'systemctl stop', 'systemctl restart', 'systemctl enable'):
            self.assertNotIn(forbidden, text)

    def test_shellcheck_when_available(self):
        shellcheck = shutil.which('shellcheck')
        if shellcheck is None:
            self.skipTest('shellcheck not installed')
        done = subprocess.run([shellcheck, '-S', 'warning', str(QUEUE)],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_missing_input_prints_the_final_line_without_touching_the_box(self):
        env = {key: value for key, value in os.environ.items()
               if key not in ('H', 'Q', 'E3_NATIVE', 'FLASH_S5_DECODE_SMS')}
        env['Q17_BOUNDED'] = '0'
        done = subprocess.run(['bash', str(QUEUE)], capture_output=True, text=True,
                              env=env, timeout=30)
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertIn('INCONCLUSIVE missing-input:', done.stdout)
        self.assertEqual(done.stdout.strip().splitlines()[-1],
                         'Q17_GREEN_PROBE DONE rc=2 engine-untouched containers-retained')


if __name__ == '__main__':
    unittest.main(verbosity=2)
