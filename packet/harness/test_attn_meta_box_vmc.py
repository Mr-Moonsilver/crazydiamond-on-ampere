"""Offline tests for tools/attn_meta_box_vmc.py (attn-meta run c on the box): conf and input refusals, GPU-busy refusals, the
wrapper's verdict rule end to end, probe c as the one entry, exact-name container handling. docker, nvidia-smi and systemctl are stub
programs on PATH (the real subprocess path of the tool and of the wrapper it execs); no container, no GPU, no box.
"""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools/attn_meta_box_vmc.py'
sys.path.insert(0, str(ROOT / 'tools'))
import attn_meta_box_vmc as box  # noqa: E402
import attn_meta_vm_input_c as vm_input  # noqa: E402

IMAGE = 'sha256:' + 'a' * 64
NAME = 'attn-meta-vm-c'
OWN = 'flashnext-' + NAME
ENGINE_FILES = ('attn_meta_batch.py', 'attn_meta_select.py', 'e3_types.py', 'lane_chunk.py')
STUB_DOCKER = r'''#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
state = Path(os.environ['STUB_DIR'])
args = sys.argv[1:]
with (state / 'docker.log').open('a') as log:
    log.write(json.dumps(args) + '\n')
cfg = json.loads((state / 'cfg.json').read_text())
cmd = args[0]
fmt = args[args.index('--format') + 1] if '--format' in args else ''
if cmd == 'ps':
    if cfg.get('ps_fail'):
        sys.exit(1)
    if '-a' in args:
        print('\n'.join(cfg.get('all_names', [])))
    else:
        print('\n'.join(cfg.get('running', [])))
elif cmd == 'image':
    if cfg.get('image') is None:
        sys.exit(1)
    print(cfg['image'])
elif cmd == 'inspect':
    if 'HostConfig' in fmt:
        print(cfg.get('inspect', 'runc null'))
    elif fmt == '{{.State.Running}}':
        print('false')
    elif fmt == '{{json .State}}':
        print(json.dumps({'Running': False, 'ExitCode': cfg.get('probe_rc', 0)}))
    else:
        print('/%s %s' % (args[-1], cfg.get('own_running', 'false')))
elif cmd == 'top':
    print('PID\n4242')
elif cmd == 'run':
    out = next(a for a in args if a.startswith('type=bind,') and a.split(',')[2] == 'dst=/out').split(',')[1].split('=', 1)[1]
    (state / 'running').write_text('x')
    deadline = time.monotonic() + cfg.get('run_s', 1.0)
    while time.monotonic() < deadline and not (state / 'stopped').exists():
        time.sleep(0.05)
    if cfg.get('probe') is not None:
        Path(out, 'probe.json').write_text(json.dumps(cfg['probe']))
    if not cfg.get('card_stuck'):
        (state / 'running').unlink()
    sys.exit(cfg.get('probe_rc', 0))
elif cmd in ('stop', 'kill'):
    (state / 'stopped').write_text('x')
elif cmd == 'rm':
    pass
else:
    sys.exit(9)
'''
STUB_SMI = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
state = Path(os.environ['STUB_DIR'])
args = ' '.join(sys.argv[1:])
with (state / 'nvidia.log').open('a') as log:
    log.write(args + '\n')
cfg = json.loads((state / 'cfg.json').read_text())
if cfg.get('smi_fail'):
    sys.exit(1)
used = cfg.get('card_used', 100)
if (state / 'running').exists():
    used += cfg.get('card_rise', 300)
if 'query-gpu=memory.used,memory.total' in args:
    print('%d, 48000' % used)
elif 'query-gpu=name,memory.total,memory.used' in args:
    print('Stub GPU, 48000, %d' % used)
elif 'query-compute-apps=pid,process_name' in args:
    print('\n'.join(cfg.get('apps', [])))
elif 'query-compute-apps=pid,used_gpu_memory,process_name' in args:
    if (state / 'running').exists() and cfg.get('own_row', True):
        print('4242, [N/A], [Not Found]')
'''
STUB_SYSTEMCTL = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
state = Path(os.environ['STUB_DIR'])
with (state / 'systemctl.log').open('a') as log:
    log.write(' '.join(sys.argv[1:]) + '\n')
print(json.loads((state / 'cfg.json').read_text()).get('unit', 'inactive'))
sys.exit(3)
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sums(root):
    rows = sorted('%s  %s' % (sha(p), p.relative_to(root).as_posix()) for p in root.rglob('*')
                  if p.is_file() and p.name != 'SHA256SUMS')
    (root / 'SHA256SUMS').write_text('\n'.join(rows) + '\n')


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.d = Path(self._tmp.name)
        self.stub = self.d / 'stub'
        self.stub.mkdir()
        self.bin = self.d / 'bin'
        self.bin.mkdir()
        for name, text in (('docker', STUB_DOCKER), ('nvidia-smi', STUB_SMI), ('systemctl', STUB_SYSTEMCTL)):
            path = self.bin / name
            path.write_text(text)
            path.chmod(path.stat().st_mode | stat.S_IEXEC)
        self.input = self.d / (NAME + '-input')
        (self.input / 'w/tools').mkdir(parents=True)
        (self.input / 'w/tools/attn_meta_vm_probe_c.py').write_text('# stub\n')
        (self.input / 'w/overlay-qsa.py').write_text('# qsa\n')
        (self.input / 'run_gpu.py').write_text((ROOT / 'tools/attn_meta_vm_run_gpu_c.py').read_text().replace('@NAME@', NAME))
        self.bundle = self.d / 'bundle'
        (self.bundle / 'e3_builder').mkdir(parents=True)
        (self.bundle / 'native30_overlay').mkdir()
        (self.bundle / 'native30_overlay/qsa.py').write_text('# qsa\n')
        engine = {}
        for name in ENGINE_FILES:
            (self.bundle / 'e3_builder' / name).write_text('# %s\n' % name)
            engine['patches/s5_engine/' + name] = sha(self.bundle / 'e3_builder' / name)
        (self.input / 'provenance.json').write_text(json.dumps(dict(schema='attn-meta-vm-input-c/1', name=NAME, engine_bytes=engine)))
        sums(self.input)
        sums(self.bundle)
        self.out_root = self.d / 'run'
        self.out_root.mkdir()
        self.conf = self.d / 'box.conf'
        self.values = dict(INPUT_DIR=str(self.input), INPUT_MANIFEST=sha(self.input / 'SHA256SUMS'), BUNDLE_DIR=str(self.bundle),
                           BUNDLE_MANIFEST=sha(self.bundle / 'SHA256SUMS'), GPUS='2', OUT_ROOT=str(self.out_root), IMAGE=IMAGE)
        self.cfg(probe=dict(verdict='PASS'))

    def cfg(self, **kw):
        kw.setdefault('image', IMAGE)
        (self.stub / 'cfg.json').write_text(json.dumps(kw))

    def write_conf(self, **change):
        rows = dict(self.values, **change)
        self.conf.write_text(''.join('%s=%s\n' % kv for kv in rows.items() if kv[1] is not None))

    def env(self):
        return dict(os.environ, PATH='%s:%s' % (self.bin, os.environ['PATH']), STUB_DIR=str(self.stub))

    def tool(self, action='run', **change):
        self.write_conf(**change)
        return subprocess.run([sys.executable, '-B', str(TOOL), action, '--conf', str(self.conf)], env=self.env(),
                              capture_output=True, text=True, timeout=120)

    def docker_calls(self):
        path = self.stub / 'docker.log'
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    def refused(self, got, reason):
        self.assertEqual(got.returncode, 3, got.stderr[-600:])
        self.assertIn(reason, got.stderr)
        self.assertFalse((self.out_root / NAME).exists())
        self.assertFalse([c for c in self.docker_calls() if c[0] in ('run', 'stop', 'kill', 'rm')])

    def record(self):
        return json.loads((self.out_root / NAME / 'result.json').read_text())


class ConfRefusals(Base):
    def test_every_required_key(self):
        for key in box.REQUIRED:
            self.refused(self.tool(**{key: None}), 'conf-incomplete:' + key)

    def test_placeholder_unknown_duplicate_and_forms(self):
        self.refused(self.tool(OUT_ROOT='@ABSOLUTE/run-scoped-dir@'), 'conf-incomplete:OUT_ROOT')
        self.refused(self.tool(NOT_A_KEY='1'), 'conf-unknown-line')
        self.refused(self.tool(OUT_ROOT='run'), 'conf-path-not-absolute:OUT_ROOT')
        self.refused(self.tool(INPUT_MANIFEST='abc'), 'conf-not-a-sha256:INPUT_MANIFEST')
        self.refused(self.tool(IMAGE='flash:latest'), 'conf-image-not-a-digest')
        self.refused(self.tool(GPUS='0,1'), 'conf-gpus-must-name-one-card')
        self.refused(self.tool(GPUS='all'), 'conf-gpus-must-name-one-card')
        self.write_conf()
        self.conf.write_text(self.conf.read_text() + 'GPUS=3\n')
        got = subprocess.run([sys.executable, '-B', str(TOOL), 'run', '--conf', str(self.conf)], env=self.env(), capture_output=True,
                             text=True)
        self.refused(got, 'conf-duplicate-key:GPUS')

    def test_no_conf_file_and_the_example_refuses_as_shipped(self):
        got = subprocess.run([sys.executable, '-B', str(TOOL), 'run', '--conf', str(self.d / 'nope')], env=self.env(),
                             capture_output=True, text=True)
        self.refused(got, 'conf-unreadable')
        got = subprocess.run([sys.executable, '-B', str(TOOL), 'run', '--conf', str(ROOT / 'tools/attn_meta_box_vmc.conf.example')],
                             env=self.env(), capture_output=True, text=True)
        self.refused(got, 'conf-incomplete')


class InputRefusals(Base):
    def test_a_changed_input_file_is_refused_before_any_of_its_text_runs(self):
        (self.input / 'run_gpu.py').write_text('raise SystemExit(77)\n')
        self.refused(self.tool(), 'sums-mismatch:run_gpu.py')

    def test_an_extra_input_file_is_refused(self):
        (self.input / 'w/extra.py').write_text('x')
        self.refused(self.tool(), 'sums-file-set-differs')

    def test_manifests(self):
        self.refused(self.tool(INPUT_MANIFEST='0' * 64), 'input-manifest-differs')
        self.refused(self.tool(BUNDLE_MANIFEST='0' * 64), 'bundle-manifest-differs')

    def test_engine_bytes_and_qsa_bound_to_the_bundle(self):
        (self.bundle / 'e3_builder/lane_chunk.py').write_text('# moved\n')
        sums(self.bundle)
        self.refused(self.tool(BUNDLE_MANIFEST=sha(self.bundle / 'SHA256SUMS')), 'engine-bytes-not-the-bundles:e3_builder/lane_chunk.py')
        (self.bundle / 'e3_builder/lane_chunk.py').write_text('# lane_chunk.py\n')
        (self.bundle / 'native30_overlay/qsa.py').write_text('# other\n')
        sums(self.bundle)
        self.refused(self.tool(BUNDLE_MANIFEST=sha(self.bundle / 'SHA256SUMS')), 'overlay-qsa-not-the-bundles')

    def test_same_bytes_pass_here_and_in_the_unchanged_check_bind(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(vm_input.check_bind(self.input, self.bundle / 'e3_builder'), 0)
        self.assertEqual(self.tool('check').returncode, 0)

    def test_output_exists_image_missing_name_used(self):
        self.cfg(image=None)
        self.refused(self.tool(), 'image-missing')
        self.cfg(all_names=[OWN])
        self.refused(self.tool(), 'container-name-already-used')
        self.cfg()
        (self.out_root / NAME).mkdir()
        got = self.tool()
        self.assertEqual(got.returncode, 3)
        self.assertIn('output-exists', got.stderr)
        self.assertFalse([c for c in self.docker_calls() if c[0] == 'run'])


class GpuBusyRefusals(Base):
    def test_the_engine_unit_active(self):
        for state in ('active', 'activating', 'deactivating'):
            self.cfg(unit=state)
            self.refused(self.tool(), 'unit:vllm-flash:' + state)

    def test_engine_containers_by_name(self):
        for name in ('vllm-flash', 'qwen-flash-prod', 'alpha-next-c1b-engine-cpu-x'):
            self.cfg(running=['abc ' + name])
            self.refused(self.tool(), 'container-name:' + name)

    def test_any_container_with_gpu_access(self):
        self.cfg(running=['abc quiet'], inspect='runc [{"Driver":"","DeviceIDs":["0"]}]')
        self.refused(self.tool(), 'container-gpu-access:quiet')
        self.cfg(running=['abc quiet'], inspect='nvidia null')
        self.refused(self.tool(), 'container-gpu-access:quiet')

    def test_a_cpu_container_does_not_refuse(self):
        self.cfg(running=['abc forgejo'], inspect='runc null')
        self.assertEqual(self.tool('check').returncode, 0)

    def test_compute_process_and_a_card_not_idle(self):
        self.cfg(apps=['99, python3'])
        self.refused(self.tool(), 'compute-process:99 python3')
        self.cfg(card_used=513)
        self.refused(self.tool(), 'card-not-idle:2:513 MiB')

    def test_failing_queries_refuse(self):
        self.cfg(ps_fail=True)
        self.refused(self.tool(), 'docker-ps-failed')
        self.cfg(smi_fail=True)
        self.refused(self.tool(), 'nvidia-smi-compute-apps-failed')
        (self.bin / 'systemctl').unlink()
        (self.bin / 'python3').symlink_to(sys.executable)       # the stubs' interpreter, nothing else on this PATH
        self.cfg()
        got = subprocess.run([sys.executable, '-B', str(TOOL), 'check', '--conf', str(self.conf)],
                             env=dict(self.env(), PATH=str(self.bin)), capture_output=True, text=True)
        self.assertEqual(got.returncode, 3)
        self.assertIn('query-failed:systemctl', got.stderr)

    def test_check_is_read_only(self):
        got = self.tool('check')
        self.assertEqual(got.returncode, 0, got.stderr)
        self.assertEqual(json.loads(got.stdout)['verdict'], 'CLEAR')
        self.assertFalse((self.out_root / NAME).exists())
        self.assertFalse([c for c in self.docker_calls() if c[0] in ('run', 'stop', 'kill', 'rm', 'start', 'restart')])


class TheRun(Base):
    def run_args(self):
        return next(c for c in self.docker_calls() if c[0] == 'run')

    def test_pass_rc_0_on_the_named_card_with_probe_c(self):
        got = self.tool()
        self.assertEqual(got.returncode, 0, got.stderr[-800:])
        rec = self.record()
        self.assertEqual((rec['verdict'], rec['rc'], rec['schema'], rec['box_action']), ('PASS', 0, box.SCHEMA, True))
        self.assertEqual(rec['own_peak_mib'], 300)          # the card's rise: the stub's compute row has no memory field
        args = self.run_args()
        self.assertEqual(args[args.index('--entrypoint') + 1:args.index('--entrypoint') + 4], ['python3', IMAGE, '-B'])
        self.assertEqual(args.count('/w/tools/attn_meta_vm_probe_c.py'), 1)
        self.assertEqual(args[args.index('--gpus') + 1], 'device=2')
        self.assertNotIn('--runtime', args)
        self.assertNotIn('NVIDIA_VISIBLE_DEVICES=0', args)
        self.assertIn('CUDA_VISIBLE_DEVICES=0', args)
        self.assertIn('--pull=never', args)
        self.assertEqual(args[args.index('--name') + 1], OWN)
        run = json.loads((self.out_root / NAME / 'box-run.json').read_text())
        self.assertEqual((run['rc'], run['gpu'], run['card_baseline_mib']), (0, '2', 100))
        self.assertEqual(run['tool_sha256'], sha(TOOL))

    def test_samples_come_from_the_card_when_no_pid_row_shows(self):
        self.cfg(probe=dict(verdict='PASS'), own_row=False)
        self.assertEqual(self.tool().returncode, 0)
        rec = self.record()
        self.assertGreater(rec['own_samples'], 0)
        self.assertEqual(rec['own_peak_mib'], 300)

    def test_probe_fail_is_rc_1(self):
        self.cfg(probe=dict(verdict='FAIL', failures=['opcount:x']), probe_rc=1)
        self.assertEqual(self.tool().returncode, 1)
        rec = self.record()
        self.assertEqual(rec['verdict'], 'FAIL')
        self.assertIn('opcount:x', rec['failures'])
        self.assertIn('process_rc:1', rec['failures'])

    def test_no_probe_record_is_rc_2(self):
        self.cfg(probe=None)
        self.assertEqual(self.tool().returncode, 2)
        rec = self.record()
        self.assertEqual(rec['verdict'], 'INCONCLUSIVE')
        self.assertIn('probe_result', rec['missing'])

    def test_no_memory_evidence_is_rc_2(self):
        self.cfg(probe=dict(verdict='PASS'), card_rise=0, own_row=False)
        self.assertEqual(self.tool().returncode, 2)
        self.assertIn('own_gpu_samples', self.record()['missing'])

    def test_the_memory_stop_stops_its_own_container_and_fails(self):
        self.cfg(probe=dict(verdict='PASS'), card_rise=3000, run_s=20)
        self.assertEqual(self.tool().returncode, 1)
        rec = self.record()
        self.assertIn('own_gpu_stop', rec['failures'])
        self.assertEqual([c[-1] for c in self.docker_calls() if c[0] == 'stop'], [OWN])

    def test_a_card_not_released_is_rc_2(self):
        self.cfg(probe=dict(verdict='PASS'), card_stuck=True)
        self.write_conf()
        saved_argv, saved_env, saved_wait = sys.argv, dict(os.environ), box.CARD_RELEASE_WAIT_S
        os.environ.update(self.env())
        box.CARD_RELEASE_WAIT_S = 1
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(box.main(['run', '--conf', str(self.conf)]), 2)
        finally:
            sys.argv, box.CARD_RELEASE_WAIT_S = saved_argv, saved_wait
            os.environ.clear()
            os.environ.update(saved_env)
        self.assertIn('own_gpu_cleanup', self.record()['missing'])

    def test_every_docker_call_is_declared_and_exact_name_only(self):
        self.cfg(probe=dict(verdict='PASS'), card_rise=3000, run_s=20)
        self.tool()
        calls = self.docker_calls()
        self.assertLessEqual({c[0] for c in calls}, {'ps', 'image', 'inspect', 'top', 'run', 'stop', 'kill', 'rm'})
        for c in calls:
            if c[0] in ('stop', 'kill', 'rm', 'top'):
                self.assertIn(OWN, c)
        self.assertEqual((self.stub / 'systemctl.log').read_text().split('\n')[0], 'is-active vllm-flash')
        self.assertEqual(set((self.stub / 'systemctl.log').read_text().split()), {'is-active', 'vllm-flash'})


class Cleanup(Base):
    def test_refused_without_the_record(self):
        self.refused(self.tool('cleanup'), 'cleanup-no-record')

    def test_exact_name_rm_after_the_record_never_force(self):
        self.assertEqual(self.tool().returncode, 0)
        self.assertFalse([c for c in self.docker_calls() if c[0] == 'rm'])      # the run itself retains the container
        got = self.tool('cleanup')
        self.assertEqual(got.returncode, 0, got.stderr)
        self.assertEqual([c for c in self.docker_calls() if c[0] == 'rm'], [['rm', OWN]])

    def test_a_running_container_is_not_removed(self):
        self.assertEqual(self.tool().returncode, 0)
        self.cfg(own_running='true')
        got = self.tool('cleanup')
        self.assertEqual(got.returncode, 3)
        self.assertFalse([c for c in self.docker_calls() if c[0] == 'rm'])


class TheWrapperIsTheInputs(unittest.TestCase):
    TEXT = (ROOT / 'tools/attn_meta_vm_run_gpu_c.py').read_text()

    def test_each_swap_matches_the_unchanged_wrapper_exactly_once(self):
        for old, _ in box.WRAPPER_SWAPS:
            self.assertEqual(self.TEXT.count(old), 1, old)
        compile(box.swap_wrapper(self.TEXT), 'swapped', 'exec')

    def test_a_moved_wrapper_is_refused(self):
        with self.assertRaises(box.Refused):
            box.swap_wrapper(self.TEXT.replace("'VM lane occupied'", "'other'"))

    def test_the_verdict_rule_is_the_wrappers_but_for_the_declared_cleanup_condition(self):
        swapped = box.swap_wrapper(self.TEXT)

        def rule(text):
            return text[text.index('    failures, missing = [], []\n'):text.index("    record = dict(schema=")]
        self.assertEqual(rule(swapped).replace(" and BOX['released']()", ''), rule(self.TEXT))

    def test_probe_c_is_what_the_input_tool_ships_and_the_wrapper_runs(self):
        self.assertEqual(self.TEXT.count("'/w/tools/attn_meta_vm_probe_c.py'"), 1)
        self.assertEqual(vm_input.FILES['w/tools/attn_meta_vm_probe_c.py'], 'tools/attn_meta_vm_probe_c.py')
        self.assertEqual(box.PROBE, '/w/tools/attn_meta_vm_probe_c.py')

    def test_the_tool_holds_no_engine_control_auth_or_window_text(self):
        tree = ast.parse(TOOL.read_text())
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr)
                and isinstance(n.body[0].value, ast.Constant)}
        strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]
        for bad in ('restart', 'canary', 'auth', '/tmp/', 'setcfg', 'stall2', '/data/vllm', '/home/<service-user>', '192.168', 'ssh'):
            self.assertFalse([s for s in strings if bad in s], bad)
        self.assertFalse({'start', 'enable', 'disable', 'pull'} & set(strings))
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertLessEqual(imported, {'argparse', 'csv', 'fcntl', 'hashlib', 'io', 'json', 'os', 're', 'subprocess', 'sys', 'time'})


if __name__ == '__main__':
    unittest.main()
