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
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from e3_green_probe import build_command, verdict, verify_native, NATIVE_PINS, write_json, measure_timings
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
    timings = {'shape': [4096, 4096, 4096], 'dtype': 'bfloat16', 'iterations': 20, 'outputs_finite': True,
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

    def test_driver_rounding_is_recorded_not_a_parameter_change(self):
        good = record(requested=44)
        good['describe'].update(decode_sms=48, prefill_sms=34)
        good['membership'].update(decode_seen=48, prefill_seen=34)
        self.assertVerdict([good], 'PASS')

    def test_missing_duplicate_devices_and_invalid_samples_cannot_pass(self):
        self.assertEqual(verdict([record(0)], [0, 1])['verdict'], 'INCONCLUSIVE')
        self.assertEqual(verdict([record(0), record(0)])['verdict'], 'INCONCLUSIVE')
        for x in (float('nan'), float('inf'), 0, -1, True):
            bad = record()
            bad['timings']['ordinary']['ms'][0] = x
            self.assertVerdict([bad], 'FAIL')
        bad = record()
        bad['timings']['ordinary']['ms'].pop()
        self.assertVerdict([bad], 'FAIL')
        bad = record()
        bad['timings']['outputs_finite'] = False
        self.assertVerdict([bad], 'FAIL')

    def test_close_failure_survives_interruption_same_device(self):
        bad = record(status='interrupted')
        bad['errors'] = [{'kind': 'cuda', 'message': 'close failed'}]
        self.assertVerdict([bad], 'FAIL')


class ArtifactIdentity(unittest.TestCase):
    def native_copy(self, root):
        for name in NATIVE_PINS:
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Immutable q17a/q17c pins: do not replace them with a new helper ABI
            # extension or newer additive type. The pinned bytes stay under fixtures.
            frozen = {'green_context.py': 'q17c_original_green_context.py',
                      'e3_types.py': 'q17c_original_e3_types.py',
                      'native/flash_s5_green.cu': 'q17a_original_flash_s5_green.cu'}
            source = HERE / 'fixtures' / frozen[name] if name in frozen else HERE.parent / 'patches/s5_engine' / name
            shutil.copyfile(source, dest)
        (root / 'SHA256SUMS').write_text(''.join(f'{sha}  {name}\n' for name, sha in sorted(NATIVE_PINS.items())))

    def test_exact_native_pins_extra_file_and_changed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.native_copy(root)
            self.assertEqual(verify_native(root), NATIVE_PINS)
            (root / 'extra.py').write_text('')
            with self.assertRaisesRegex(ValueError, 'inventory'):
                verify_native(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.native_copy(root)
            (root / 'green_context.py').write_text('different')
            with self.assertRaisesRegex(ValueError, 'source pin'):
                verify_native(root)

    def test_symlink_native_source_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.native_copy(root)
            (root / 'extra.py').symlink_to(root / 'green_context.py')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                verify_native(root)

    def test_json_publication_does_not_clobber_a_racing_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'result.json'
            original = os.link
            def race(src, dst):
                target.write_text('other writer')
                return original(src, dst)
            with patch('e3_green_probe.os.link', side_effect=race):
                with self.assertRaises(FileExistsError):
                    write_json(target, {'ok': True})
            self.assertEqual(target.read_text(), 'other writer')


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
        '[[ ! -e $Q && ! -L $Q ]]',
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


class TimingOwnership(unittest.TestCase):
    def test_inputs_ready_outputs_private_and_submissions_interleaved(self):
        log, current = [], ['ordinary']
        class Stream:
            def __init__(self, name): self.name = name
            def synchronize(self): log.append(('sync', self.name))
        streams = {name: Stream(name) for name in ('ordinary', 'decode_stream', 'prefill_stream')}
        @contextmanager
        def stream(value):
            old, current[0] = current[0], value.name
            try: yield
            finally: current[0] = old
        class Event:
            def __init__(self, **kw): pass
            def record(self): self.lane = current[0]
            def elapsed_time(self, other):
                assert self.lane == other.lane
                return 1.0
        def matmul(a, b, *, out): log.append(('matmul', current[0], out))
        cuda = SimpleNamespace(current_stream=lambda d: streams['ordinary'], stream=stream,
                               synchronize=lambda d: log.append(('sync', 'device')), Event=Event)
        finite = SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True))
        torch = SimpleNamespace(cuda=cuda, bfloat16='bf16', randn=lambda *a, **k: object(),
                                empty_like=lambda a: object(), matmul=matmul, isfinite=lambda a: finite)
        split = SimpleNamespace(decode_stream=streams['decode_stream'], prefill_stream=streams['prefill_stream'])
        result = measure_timings(torch, split, 0, 4, 3)
        self.assertTrue(result['outputs_finite'])
        self.assertEqual(log[0], ('sync', 'device'))
        calls = [entry for entry in log if entry[0] == 'matmul']
        ids = {name: {id(out) for _, lane, out in calls if lane == name} for name in streams}
        self.assertTrue(all(len(values) == 1 for values in ids.values()))
        self.assertEqual(len(set.union(*ids.values())), 3)
        self.assertEqual([entry[1] for entry in calls[-6:]], ['decode_stream', 'prefill_stream'] * 3)


class QueueLifecycle(unittest.TestCase):
    def test_missing_cid_uses_full_id_and_stops_only_owned_container(self):
        # Execute the real shell cleanup function with a fake docker executable.
        text = QUEUE.read_text()
        function = text[text.index('cleanup_own() {'):text.index('on_exit() {')]
        docker = r"""#!/usr/bin/env python3
import os, sys
from pathlib import Path
args = sys.argv[1:]
root = Path(os.environ['FAKE_ROOT'])
with (root / 'calls').open('a') as out: out.write(' '.join(args) + '\n')
if args[0] == 'ps':
    assert '--no-trunc' in args
    print('a' * 64)
elif args[0] == 'inspect':
    assert args[1] == 'a' * 64
    if 'Labels' in args[-1]: print('owned')
    else: print('false' if (root / 'stopped').exists() else 'true')
elif args[0] == 'stop':
    assert args[-1] == 'a' * 64
    (root / 'stopped').write_text('1')
else: raise AssertionError(args)
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / 'docker'
            exe.write_text(docker)
            exe.chmod(0o755)
            env = dict(os.environ, FAKE_ROOT=tmp, Q=tmp, PATH=tmp + ':' + os.environ['PATH'])
            done = subprocess.run(['bash', '-c', 'set -euo pipefail; active=""; token=owned; name=own-name; '
                                   + function + '\ncleanup_own'], env=env, text=True,
                                  capture_output=True, timeout=30)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertTrue((root / 'stopped').exists())
            self.assertNotIn('rm ', (root / 'calls').read_text())

    def test_summary_launch_consistency_and_abort(self):
        text = QUEUE.read_text()
        code = text.split("<<'PYGRADE'\n", 1)[1].split('\nPYGRADE', 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'summary.json'
            for verdict_name, raw, expected in [('PASS', 0, 0), ('PASS', 1, 2), ('PASS', 124, 2),
                                              ('FAIL', 0, 1), ('FAIL', 137, 1),
                                              ('INCONCLUSIVE', 0, 2), ('INCONCLUSIVE', 2, 2)]:
                path.write_text(json.dumps({'verdict': verdict_name, 'requested_devices': [0,1,2,3],
                                            'records': ['gpu-%d.json' % i for i in range(4)]}))
                done = subprocess.run([sys.executable, '-c', code, str(path), str(raw)],
                                      capture_output=True, text=True, timeout=10)
                self.assertEqual(done.returncode, expected, (verdict_name, raw, done.stderr))
            path.write_text('{}')
            done = subprocess.run([sys.executable, '-c', code, str(path), '0'],
                                  capture_output=True, timeout=10)
            self.assertEqual(done.returncode, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
