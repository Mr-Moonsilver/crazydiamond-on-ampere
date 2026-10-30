#!/usr/bin/env python3
"""CPU-only offline execution of the q17s startup/capture queue.

Fake PATH commands, a temporary task/defaults tree, a real loopback HTTP
transport and a fake identity executable are used. No root, docker daemon,
GPU, engine, real /etc/default or /data path is touched. These tests prove the
queue lifecycle and verdict plumbing only; they do not prove GPU startup.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from test_e3_startup_grade import rank_row

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QUEUE = HERE / 'runs' / 'q17_decode_startup.draft.sh'
SETTINGS = HERE / 'runs' / 'q17_startup_settings.sh'
IMAGE = 'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
AUTH = 'offline-fixture-key-1234567890'
LAUNCHER_SHA = 'c812eb3d610a963919c537d4963b49c54f25ba8d7d518448dbb81ca18e860d98'
CU = b'// fixture native green helper source\n'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(root):
    lines = []
    for path in sorted(Path(root).rglob('*')):
        if path.is_file() and path != Path(root) / 'SHA256SUMS':
            lines.append(sha(path) + '  ' + path.relative_to(root).as_posix() + '\n')
    (Path(root) / 'SHA256SUMS').write_text(''.join(lines))


def full11_launcher_text():
    sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
    import make_e2_arena, make_s5_overlap
    text = make_s5_overlap._s5_launcher(make_e2_arena.launcher(
        (ROOT / 'deploy/run-flash-prod.sh').read_text()))
    text = without_native28_log_cap(text)
    # The legacy full11/teacher12/startup launchers predate these engine-side
    # modules; strip them so the frozen queues still reconstruct byte-identically.
    for value in ('e3/activity_accounting.py ', 'e3/engine_offers.py ',
                  'e3/admission_metadata.py ', 'e3/worker_bridge.py ',
                  'e3/sparse_history_dma.py ', 'e3/native_source.py ', 'e3/row1_oracle.py ', 'e3/row1_exactness.py ',
                  'e3/native_overlap_metrics.py ', 'e3/native_overlap_worker.py ',
                  'e3/native_overlap_control.py ', 'e3/native_overlap_session.py ',
                  'e3/native_prefill_offers.py ', 'e3/native_placement.py ', 'e3/native_measurement.py ', 'e3/native_qualification.py ', 'e3/native_qualification_frontend.py ', 'e3/native_observation.py ', 'e3/native_graph_placement.py ',
                  'e3/native_readiness.py ',
                  'e3_sources/sampler.py ', 'e3_sources/async_utils.py '):
        text = text.replace(value, '')
    text = text.replace('      -v "$OVERLAY/e3_sources/sampler.py:$V/worker/gpu/sample/sampler.py:ro"\n', '')
    text = text.replace('      -v "$OVERLAY/e3_sources/async_utils.py:$V/worker/gpu/async_utils.py:ro"\n', '')
    return text


def without_native28_log_cap(text):
    """Rebuild immutable pre-native28 queue fixture bytes from the new generator."""
    text = text.replace('s5=(); s5_auth=(); s5_log=()', 's5=(); s5_auth=()')
    text = text.replace(
        "  # Bound Docker's ephemeral json-file log; the queue follows the full stream\n"
        "  # into its own retained host artifact before engine startup.\n"
        "  s5_log=(--log-opt max-size=512m --log-opt max-file=4)\n", '')
    return text.replace('"${extra_env[@]}" "${s5_auth[@]}" "${s5_log[@]}" "${mps[@]}"',
                        '"${extra_env[@]}" "${s5_auth[@]}" "${mps[@]}"')

def launcher_text():
    # Historical immutable queues pin the pre-teacher module inventory.
    sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
    import make_e2_arena
    import make_s5_overlap
    return full11_launcher_text().replace('e3/capture_streams.py ', '').replace('e3/teacher_decode.py ', '').replace('e3/route_invariant.py ', '').replace('e3/launch_diagnostic.py ', '').replace('e3/lane_private.py ', '').replace('e3/native/flash_s5_graph_bytes.cpp ', '')


FAKE_SYSTEMCTL = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
state = Path(os.environ['FAKE_STATE'])
d = json.loads(state.read_text()) if state.exists() else {'active': 'inactive', 'restarts': 0}
args = sys.argv[1:]
cmd = args[0] if args else ''
def save(): state.write_text(json.dumps(d))
def fail(name): return Path(os.environ['FAKE_BIN'], name).exists()
with open(os.environ['FAKE_CALLS'], 'a') as f: f.write(' '.join(args) + '\n')
if cmd == 'show':
    key = args[args.index('-p') + 1]
    print(d['active'] if key == 'ActiveState' else d['restarts'])
    raise SystemExit(0)
if cmd == 'is-active': raise SystemExit(0 if d['active'] == 'active' else 1)
if cmd == 'is-failed': raise SystemExit(1)
if cmd == 'reset-failed': raise SystemExit(0)
if cmd == 'restart':
    if fail('fail-restart'): raise SystemExit(1)
    d['active'] = 'active'
    if fail('increment-restarts'): d['restarts'] += 1
    save()
    mode = oct(os.stat(os.environ['FAKE_DEFAULTS']).st_mode & 0o777)
    with Path(os.environ['FAKE_LOG']).open('a') as f: f.write('restarted engine startup line\n')
    with open(os.environ['FAKE_CALLS'], 'a') as f: f.write('restart defaults-mode ' + mode + '\n')
    raise SystemExit(0)
if cmd == 'stop':
    if fail('fail-stop'): raise SystemExit(1)
    d['active'] = 'inactive'; save(); raise SystemExit(0)
raise SystemExit(2)
'''

FAKE_DOCKER = r'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['FAKE_CALLS'], 'a') as f: f.write('docker ' + ' '.join(args) + '\n')
if args[:2] == ['image', 'inspect']:
    print(os.environ['FAKE_IMAGE']); raise SystemExit(0)
if args and args[0] == 'ps':
    raise SystemExit(0)
if args and args[0] == 'logs':
    print(Path(os.environ['FAKE_LOG']).read_text(), end=''); raise SystemExit(0)
if args and args[0] == 'inspect':
    fmt = args[args.index('--format') + 1] if '--format' in args else ''
    print(fmt.replace('{{.Id}}', 'c' * 64).replace(
        '{{.State.StartedAt}}', '2026-10-26T00:00:00Z'))
    raise SystemExit(0)
raise SystemExit(2)
'''

FAKE_NVIDIA = r'''#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
query = next((x for x in args if x.startswith('--query-')), '')
with open(os.environ['FAKE_CALLS'], 'a') as f: f.write('nvidia-smi ' + ' '.join(args) + '\n')
if 'compute-apps' in query: raise SystemExit(0)
if query.startswith('--query-gpu=index,memory.used,memory.total'):
    for i in range(4): print(f'{i}, 100, 1000')
    raise SystemExit(0)
if query.startswith('--query-gpu=memory.used'):
    for _ in range(4): print('100')
    raise SystemExit(0)
raise SystemExit(2)
'''

FAKE_CURL = r'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
with open(os.environ['FAKE_CALLS'], 'a') as f: f.write('curl ' + ' '.join(sys.argv[1:]) + '\n')
raise SystemExit(7 if Path(os.environ['FAKE_BIN'], 'fail-health').exists() else 0)
'''

FAKE_SETCFG = r'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
path = Path(os.environ['FAKE_DEFAULTS'])
lines = path.read_text().splitlines()
values, order = {}, []
for line in lines:
    if '=' not in line or line.lstrip().startswith('#'): continue
    k, v = line.split('=', 1)
    if k not in values: order.append(k)
    values[k] = v.strip().strip('"')
for word in sys.argv[1:]:
    k, sep, v = word.partition('=')
    assert sep
    if v == '-':
        values.pop(k, None)
        if k in order: order.remove(k)
    else:
        if k not in values: order.append(k)
        values[k] = v
out = []
for k in order:
    out.append(k + '="' + values[k].replace('\\', '\\\\').replace('"', '\\"') + '"')
Path(str(path) + '.next').write_text('\n'.join(out) + '\n')
# The real reference setcfg prints its whole override mapping, private auth
# included. Mirror that so the queue must suppress the stream.
print('next:', {w.partition('=')[0]: w.partition('=')[2] for w in sys.argv[1:]})
'''

FAKE_IDENTITY = r'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
mode, out = sys.argv[1], Path(sys.argv[2])
if mode == 'capture':
    if Path(os.environ['FAKE_BIN'], 'fail-identity').exists(): raise SystemExit(2)
    with out.open('x') as f: f.write('{}\n')
    raise SystemExit(0)
if mode == 'check':
    raise SystemExit(1 if Path(os.environ['FAKE_BIN'], 'fail-identity-check').exists() else 0)
raise SystemExit(2)
'''


def make_rank_rows(helper, lib):
    rows = [rank_row(i) for i in range(4)]
    for row in rows:
        inv = row['inventory']
        inv['helper_source_sha256'] = helper
        inv['library_sha256'] = lib
        inv['binding_identity']['helper_source_sha256'] = helper
        inv['binding_identity']['library_sha256'] = lib
        inv['split']['helper']['helper_source_sha256'] = helper
        inv['split']['helper']['library_sha256'] = lib
    return rows


def make_server(rows, state):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get('Content-Length', 0))
            try:
                command = json.loads(self.rfile.read(length))
            except Exception:
                self.send_response(400); self.end_headers(); return
            with open(state['calls'], 'a') as f:
                f.write(json.dumps(command, sort_keys=True) + '\n')
            if state.get('http_error'):
                self.send_response(500); self.end_headers(); return
            if self.headers.get('Authorization') != 'Bearer ' + AUTH:
                self.send_response(403); self.end_headers(); return
            seq = command.get('seq', 0)
            action = command.get('action')
            for row in rows:
                if 'memory' in row:
                    row['memory'].update(wall_s=time.time(), sample_end_wall_s=time.time(),
                                         monotonic_ns=time.monotonic_ns(), sample_end_monotonic_ns=time.monotonic_ns())
            body = {'schema': 1, 'run_id': command.get('run_id'),
                    'engine_epoch': 9, 'command_seq': seq, 'action': action,
                    'quarantined': False, 'paused': False,
                    'snapshot': {'route': 'full', 'route_generation': 7,
                                 'storage_epoch': 1,
                                 'ranks': rows}}
            if action == 'open':
                body.pop('command_seq'); body['command_seq'] = 0
            data = json.dumps(body).encode()
            if state.get('bad_json'):
                data = b'{not json'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, format, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


class QueueRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bin = self.root / 'bin'; self.bin.mkdir()
        self.task = self.root / 'task'; self.task.mkdir()
        (self.task / 'run-flash-prod.sh').write_text('#!/bin/sh\n# found production launcher\n')
        (self.task / 'run-flash-prod.sh').chmod(0o755)
        self.defaults = self.root / 'defaults'
        self.defaults.write_text('GPU_KV_BYTES="1800000000"\nEXTRA_ENV="OLD=1"\nOVERLAY="/old"\nS5="0"\n')
        self.image_file = self.root / 'image.txt'; self.image_file.write_text(IMAGE + '\n')
        self.setcfg = self.root / 'setcfg.py'; self.setcfg.write_text(FAKE_SETCFG)
        self.h = self.root / 'harness'; self.h.mkdir()
        self.r = self.root / 'requests'; self.r.mkdir()
        self.q = self.root / 'out'
        private = self.root / 'private'; private.mkdir(mode=0o700)
        self.auth_file = private / 'api.env'
        from e3_private_auth import prepare
        prepare(self.auth_file, {'VLLM_API_KEY': AUTH})
        self.overlay = self.root / 'overlay'; self.overlay.mkdir()
        self.calls = self.root / 'calls.log'; self.calls.write_text('')
        self.log = self.root / 'engine.log'; self.log.write_text('old startup line\n')
        self.children = []
        self.make_harness()
        self.make_overlay()
        self.make_requests()
        self.state = {'calls': str(self.root / 'rpc-calls.log'), 'http_error': False,
                      'bad_json': False}
        self.server, self.thread = make_server(make_rank_rows(self.helper, self.lib), self.state)
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown(); self.server.server_close()
        for p in self.children:
            p.kill(); p.wait()
        self.tmp.cleanup()

    def make_harness(self):
        for name in ('e3_startup_grade.py', 'e3_probe_rpc.py', 'e2_memory_attribution.py',
                     'qlib.sh', 'e3_startup_mount_policy.json'):
            shutil.copyfile(HERE / name, self.h / name)
        (self.h / 'runs').mkdir()
        shutil.copyfile(QUEUE, self.h / 'runs/q17_decode_startup.draft.sh')
        shutil.copyfile(SETTINGS, self.h / 'runs/q17_startup_settings.sh')
        (self.h / 'engine_identity.py').write_text(FAKE_IDENTITY)
        (self.h / 'e3_build_closure.json').write_text(json.dumps({
            'schema': 1,
            'builder_sources': {name: 'd' * 64 for name in (
                'make_s5_overlap.py', 'green_context.py', 'decode_probe.py',
                'operation_source.py')} | {
                'native/flash_s5_green.cu': hashlib.sha256(CU).hexdigest()},
            'reference_pins': {'v1/worker/gpu_worker.py': 'c' * 64},
        }))
        write_manifest(self.h)

    def make_overlay(self):
        green = b'# fixture green context\n'
        (self.overlay / 'e3').mkdir()
        (self.overlay / 'e3/native').mkdir()
        (self.overlay / 'e3/green_context.py').write_bytes(green)
        self.helper = hashlib.sha256(green).hexdigest()
        runtime = {'decode_probe': b'# decode probe\n', 'probe_control': b'# probe control\n'}
        build_only = {'operation_source': b'# operation source\n'}
        for name, data in {**runtime, **build_only}.items():
            (self.overlay / ('e3/' + name + '.py')).write_bytes(data)
        lib = b'\x7fELF fixture helper bytes\n'
        (self.overlay / 'e3/native/flash_s5_green.so').write_bytes(lib)
        self.lib = hashlib.sha256(lib).hexdigest()
        (self.overlay / 'e3/native/flash_s5_green.cu').write_bytes(CU)
        (self.overlay / 'e3/native/build_green_helper.sh').write_text('#!/bin/sh\n')
        evidence = {
            'pins': {'v1/worker/gpu_worker.py': 'c' * 64},
            'helper_source_sha256': self.helper,
            'generated_source_shas': {name: hashlib.sha256(data).hexdigest()
                                      for name, data in runtime.items()},
            'build_only_source_shas': {name: hashlib.sha256(data).hexdigest()
                                       for name, data in build_only.items()},
            'tests': ['test_e3_decode_probe.py'],
        }
        (self.overlay / 'e3/decode-probe-evidence.json').write_text(json.dumps(evidence))
        declaration = {
            'declared': True, 'qualifies': 'diagnostic_modes.decode_only',
            'diagnostic_only': True,
            'scope': 'separately declared diagnostic mode; it does not qualify the full FLASH_S5_OVERLAP package, which stays armable:false',
            'enabled_by': {'FLASH_S5_DECODE_ONLY_PROBE': '1'},
            'requires': {},
            'gate': 's5_startup.require_decode_probe_ready',
            'startup': 'native_startup.install_decode_probe',
            'constructs': [], 'excluded': [],
            'module_inventory': {'runtime': list(runtime), 'build_only': list(build_only),
                                 'basis': 'build-only text generator not imported by runtime'},
            'evidence': evidence,
        }
        generation = {
            'schema': 1, 'kind': 'e3-s5-overlap', 'armable': False,
            'default_feature': 'FLASH_S5_OVERLAP=0', 'incremental_kv_bytes': 0,
            'reference_source_pins': {'v1/worker/gpu_worker.py': 'c' * 64},
            'generated_files': ['run-flash-s5-test.sh', 'e3/decode-probe-evidence.json'],
            'launcher': {'name': 'run-flash-s5-test.sh', 'mounts_require': 'S5=1',
                         'status': 'offline-generated; not mounted or started'},
            'diagnostic_modes': {'decode_only': declaration},
        }
        (self.overlay / 'S5-OVERLAP-GENERATION.json').write_text(json.dumps(generation))
        (self.overlay / 'run-flash-s5-test.sh').write_text(launcher_text())
        (self.overlay / 'run-flash-s5-test.sh').chmod(0o755)
        (self.overlay / 'core').mkdir()
        (self.overlay / 'core/block_pool.py').write_text('# fixture core\n')
        (self.overlay / 'core/SHA256SUMS').write_text(
            sha(self.overlay / 'core/block_pool.py') + '  block_pool.py\n')
        write_manifest(self.overlay)

    def make_requests(self):
        identity = {
            'image': {'id': IMAGE},
            'defaults': {'S5': '1', 'OVERLAY': str(self.overlay),
                         'CORE_OVERLAY': '1', 'PP_LAYER_PARTITION': '24,24'},
            'env': {'FLASH_S5_DECODE_ONLY_PROBE': '1', 'FLASH_S5_OVERLAP': '0',
                    'FLASH_S5_DECODE_SMS': '44',
                    'FLASH_S5_GREEN_LIB_SHA256': self.lib,
                    'VLLM_API_KEY': {'redacted_length': len(AUTH)}},
            'mounts': [
                {'source': str(self.overlay / 'run-flash-s5-test.sh'),
                 'destination': '/work', 'rw': True,
                 'content': {'entries': [{'path': 'prod/run-flash-prod.sh',
                                          'sha256': LAUNCHER_SHA}]}},
                {'source': str(self.overlay / 'e3'), 'destination': '/vllm/v1/flash_e3',
                 'rw': False},
            ],
        }
        (self.r / 'startup.json').write_text(json.dumps({'schema': 1, 'identity': identity}))
        shutil.copyfile(self.task / 'run-flash-prod.sh', self.r / 'launcher.found')
        pins = [self.r / 'startup.json', self.r / 'launcher.found',
                self.h / 'runs/q17_decode_startup.draft.sh',
                self.h / 'runs/q17_startup_settings.sh',
                self.h / 'e3_startup_grade.py', self.h / 'e3_startup_mount_policy.json',
                self.h / 'e3_build_closure.json', self.h / 'e3_probe_rpc.py',
                self.h / 'engine_identity.py', self.h / 'qlib.sh',
                self.overlay / 'SHA256SUMS', self.overlay / 'core/SHA256SUMS',
                self.overlay / 'S5-OVERLAP-GENERATION.json',
                self.overlay / 'run-flash-s5-test.sh',
                self.overlay / 'e3/decode-probe-evidence.json',
                self.overlay / 'e3/green_context.py',
                self.overlay / 'e3/native/flash_s5_green.so',
                self.setcfg, self.image_file]
        (self.r / 'SHA256SUMS').write_text(
            ''.join(sha(p) + '  ' + str(p.absolute()) + '\n' for p in pins))

    def fake(self, name, text, mode=0o755):
        p = self.bin / name; p.write_text(text); p.chmod(mode)

    def install_fakes(self):
        self.fake('systemctl', FAKE_SYSTEMCTL)
        self.fake('docker', FAKE_DOCKER)
        self.fake('nvidia-smi', FAKE_NVIDIA)
        self.fake('curl', FAKE_CURL)
        state = self.root / 'systemctl.json'
        state.write_text(json.dumps({'active': 'inactive', 'restarts': 0}))
        env = dict(os.environ)
        env.update({
            'PATH': str(self.bin) + os.pathsep + os.environ['PATH'],
            'FAKE_BIN': str(self.bin), 'FAKE_STATE': str(state),
            'FAKE_CALLS': str(self.calls), 'FAKE_IMAGE': IMAGE,
            'FAKE_LOG': str(self.log), 'FAKE_DEFAULTS': str(self.defaults),
            'HARNESS_DIR': str(self.h), 'REQUESTS': str(self.r),
            'S5_OVERLAY': str(self.overlay), 'S5_GREEN_LIB_SHA256': self.lib,
            'S5_AUTH_ENV_FILE': str(self.auth_file),
            'VLLM_API_KEY': AUTH, 'PROBE_URL': 'http://127.0.0.1:%d' % self.port,
            'Q17_TEST_OFFLINE': '1', 'Q17_TEST_TASK_BASE': str(self.task),
            'Q17_TEST_DEFAULTS_FILE': str(self.defaults),
            'Q17_TEST_SETCFG': str(self.setcfg),
            'Q17_TEST_IMAGE_FILE': str(self.image_file),
            'Q17_TEST_SERVICE': 'vllm-flash', 'Q17_TEST_CONTAINER': 'qwen-flash-prod',
            'Q17_TEST_OWNER': os.environ.get('USER', '<service-user>'),
            'Q17_TEST_GROUP': os.environ.get('USER', '<service-user>'),
            'Q17_TEST_SERVICE_USER': os.environ.get('USER', '<service-user>'),
            'Q17_STARTUP_BOUNDED': '1',
        })
        for key in ('FLASH_S5_DECODE_SMS', 'FLASH_S5_OVERLAP',
                    'FLASH_S5_DECODE_ONLY_PROBE', 'EXTRA_ENV', 'GPU_KV_BYTES'):
            env.pop(key, None)
        return env

    def run_queue(self):
        return subprocess.run(['bash', str(self.h / 'runs/q17_decode_startup.draft.sh'), str(self.q)],
                              capture_output=True, text=True, timeout=120,
                              env=self.install_fakes())

    def assert_no_secret(self, text):
        self.assertNotIn(AUTH, text)

    def test_standalone_entrypoints_refuse_before_any_box_action(self):
        env = dict(os.environ)
        env.pop('Q17_TEST_OFFLINE', None)
        for name in ('q17_decode_startup.draft.sh', 'q17s_declare.sh'):
            result = subprocess.run(['bash', str(HERE / 'runs' / name)],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('standalone run superseded', result.stdout)

    def test_success_runs_the_whole_lifecycle_and_restores(self):
        done = self.run_queue()
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assert_no_secret(done.stdout + done.stderr)
        self.assertIn('Q17S-STARTUP DONE rc=0 scope=startup/capture-only', done.stdout)
        self.assertIn('engine-state=inactive production-never-started', done.stdout)
        self.assertIn('verdict=PASS', done.stdout)
        self.assertEqual(done.stdout.count('Q17S-STARTUP DONE'), 1)
        self.assertTrue((self.q / 'startup-grade.json').is_file())
        grade = json.loads((self.q / 'startup-grade.json').read_text())
        self.assertEqual(grade['verdict'], 'PASS')
        self.assertEqual(grade['measured']['requested_decode_sms'], 44)
        self.assertEqual(grade['measured']['decode_sms'], 48)
        self.assertEqual(self.defaults.read_text().splitlines(),
                         ['GPU_KV_BYTES="1800000000"', 'EXTRA_ENV="OLD=1"',
                          'OVERLAY="/old"', 'S5="0"'])
        self.assertEqual((self.task / 'run-flash-prod.sh').read_text(),
                         '#!/bin/sh\n# found production launcher\n')
        calls = self.calls.read_text()
        self.assertIn('restart', calls)
        self.assertIn('stop', calls)
        self.assertIn('restart defaults-mode 0o600', calls,
                      'private auth in the installed test defaults must stay root-only')
        self.assertFalse((self.q / 'setcfg.log').exists(),
                         'setcfg output must not be captured: it echoes private auth')
        self.assertFalse([line for line in calls.splitlines()
                          if line.split()[:1] == ['start']], 'queue must never start the engine')
        for path in self.q.rglob('*'):
            if path.is_file():
                self.assert_no_secret(path.read_text(errors='ignore'))
        rpc = [json.loads(line) for line in (self.root / 'rpc-calls.log').read_text().splitlines()]
        self.assertEqual([c['action'] for c in rpc], ['open', 'status'])
        self.assertEqual(rpc[1]['seq'], 1)
        self.assertFalse((self.root / 'systemctl.json').read_text().find('"active": "active"') >= 0)

    def test_restart_count_increase_is_failure_and_cleanup_still_runs(self):
        (self.bin / 'increment-restarts').write_text('')
        done = self.run_queue()
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn('STEP restart FAIL', done.stdout)
        self.assertIn('STEP launcher-restore PASS', done.stdout)
        self.assertIn('STEP defaults-restore PASS', done.stdout)
        self.assertIn('"active": "inactive"', (self.root / 'systemctl.json').read_text())
        self.assertEqual(self.defaults.read_text().splitlines()[-1], 'S5="0"')

    def test_status_transport_failure_is_inconclusive_not_pass(self):
        self.state['http_error'] = True
        done = self.run_queue()
        self.assertIn('STEP status-open INCONCLUSIVE', done.stdout)
        self.assertIn('Q17S-STARTUP DONE rc=2', done.stdout)
        self.assertIn('verdict=INCONCLUSIVE', done.stdout)
        self.assertEqual(done.returncode, 2)
        self.assertEqual(json.loads((self.q / 'startup-grade.json').read_text())['verdict'],
                         'INCONCLUSIVE')
        self.assertIn('STEP engine-stop PASS', done.stdout)
        self.assertIn('STEP defaults-restore PASS', done.stdout)

    def test_identity_failure_is_inconclusive_and_still_stops(self):
        (self.bin / 'fail-identity').write_text('')
        done = self.run_queue()
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertIn('STEP engine.identity-start INCONCLUSIVE', done.stdout)
        self.assertIn('verdict=INCONCLUSIVE', done.stdout)
        self.assertIn('STEP engine-stop PASS', done.stdout)
        self.assertIn('STEP launcher-restore PASS', done.stdout)
        self.assertEqual(self.task.joinpath('run-flash-prod.sh').read_text(),
                         '#!/bin/sh\n# found production launcher\n')

    def test_stop_failure_is_fail_and_restore_is_withheld(self):
        (self.bin / 'fail-stop').write_text('')
        done = self.run_queue()
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn('STEP engine-stop FAIL', done.stdout)
        self.assertIn('STEP engine-inactive FAIL', done.stdout)
        self.assertIn('withheld: engine state active', done.stdout)
        self.assertIn('STEP launcher-restore FAIL', done.stdout)
        self.assertIn('STEP defaults-restore FAIL', done.stdout)
        self.assertIn('engine-state=active production-never-started', done.stdout)
        self.assertIn('verdict=FAIL', done.stdout)
        # The running test engine keeps its test launcher/defaults for operator
        # recovery; production bytes are not restored under a live engine.
        self.assertEqual(sha(self.task / 'run-flash-prod.sh'), LAUNCHER_SHA)
        self.assertIn('S5="1"', self.defaults.read_text())

    def test_restore_failure_is_reported(self):
        # Restore via the fake install would be hard to fail selectively; a
        # pre-created restore path is refused at preflight and must not clobber.
        (self.task / 'run-flash-prod.sh.q17s-restore').write_text('foreign')
        done = self.run_queue()
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertIn('INCONCLUSIVE', done.stdout)
        self.assertEqual((self.task / 'run-flash-prod.sh.q17s-restore').read_text(), 'foreign')

    def test_duplicate_status_rank_is_hard_failure(self):
        self.server.shutdown(); self.server.server_close()
        rows = make_rank_rows(self.helper, self.lib)
        rows[3]['rank'] = 2
        self.server, self.thread = make_server(rows, self.state)
        self.port = self.server.server_address[1]
        done = self.run_queue()
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn('STEP startup-grade FAIL', done.stdout)
        self.assertEqual(json.loads((self.q / 'startup-grade.json').read_text())['verdict'], 'FAIL')

    def test_empty_capture_is_hard_failure(self):
        self.server.shutdown(); self.server.server_close()
        rows = make_rank_rows(self.helper, self.lib)
        for row in rows:
            row['inventory']['descriptors'] = []
            row['inventory']['descriptors_captured'] = 0
            row['inventory']['capture_stream_checks'] = []
        self.server, self.thread = make_server(rows, self.state)
        self.port = self.server.server_address[1]
        done = self.run_queue()
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertEqual(json.loads((self.q / 'startup-grade.json').read_text())['verdict'], 'FAIL')

    def test_unknown_memory_is_inconclusive(self):
        self.server.shutdown(); self.server.server_close()
        rows = make_rank_rows(self.helper, self.lib)
        rows[0]['memory']['allocator']['reserved_peak_bytes'] = None
        self.server, self.thread = make_server(rows, self.state)
        self.port = self.server.server_address[1]
        done = self.run_queue()
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertEqual(json.loads((self.q / 'startup-grade.json').read_text())['verdict'],
                         'INCONCLUSIVE')

    def test_health_failure_does_not_start_a_fallback(self):
        (self.bin / 'fail-health').write_text('')
        (self.bin / 'increment-restarts').write_text('')
        done = self.run_queue()
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn('STEP restart FAIL', done.stdout)
        self.assertIn('STEP engine-stop PASS', done.stdout)
        self.assertIn('STEP defaults-restore PASS', done.stdout)

    def test_existing_output_is_refused_and_not_clobbered(self):
        self.q.mkdir()
        (self.q / 'marker').write_text('keep')
        done = self.run_queue()
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertIn('INCONCLUSIVE output-exists', done.stdout)
        self.assertEqual((self.q / 'marker').read_text(), 'keep')
        self.assertEqual(done.stdout.count('Q17S-STARTUP DONE'), 1)

    def test_missing_input_prints_final_line_and_touches_nothing(self):
        env = self.install_fakes()
        env.pop('REQUESTS'); env.pop('R', None)
        done = subprocess.run(['bash', str(self.h / 'runs/q17_decode_startup.draft.sh'), str(self.q)],
                              capture_output=True, text=True, timeout=30, env=env)
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertEqual(done.stdout.strip().splitlines()[-1],
                         'Q17S-STARTUP DONE rc=2 scope=startup/capture-only engine-state=unverified production-never-started full-e3-armable=false verdict=INCONCLUSIVE')
        self.assertFalse(self.q.exists())


if __name__ == '__main__':
    unittest.main()
