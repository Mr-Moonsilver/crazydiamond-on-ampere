"""Native39 trunk queue: the exact derivation from the P-c texts, the trunk window at real wall clock,
and the queue's own new segments executed under bash against loopback leaves.  CPU only, no box action.

Connected compositions (the queue's bytes with the real checks, guard, client, observer and grade):
  RestartBind  restart -> /health loop -> bind receipt: the queue segment, the real host observer
               following a fake docker CLI, real curl against a loopback /health leaf.
  Tail         native W1 -> plain W1 -> idle -> gates -> W2 rounds -> grade -> trunk summary: the queue
               segment, the real w1_trace client and idle check against a loopback engine leaf, the
               real native35 grade and trunk summary, fake W1/W2/memory phase executables.
  Guard        the queue's w2-length lines with the real guard against a loopback /tokenize leaf.
  Launcher     the trunk settings file sourced, the real setcfg algorithm, the real S5 launcher render
               and declaration projection, then the queue's host check on the declared identity.
  BoxFormats   the checks on retained box extracts (harness/fixtures/native39_box_extract.json,
               tools/native39_box_extract.py): docker --timestamps lines, host-disk rows, the P-c
               capture's --host, the w1ab2 served-host line, the P-c grade and the w1ab2 client output.
Fixture leaves: native39_fixture_leaves.json (the queue-* entries).
"""
import contextlib
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import e3_native39_trunk_checks as checks
import e3_native39_w2_length as length
import native35p_gate
import native39_trunk_gate as gate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / 'runs'
TRUNK = RUNS / 'q18a_native39_trunk.draft.sh'
SETTINGS = RUNS / 'q18a_native39_trunk_settings.sh'
DECLARE = RUNS / 'q18a_native39_trunk_declare.sh'
# The P-c texts as held in bundle q18a-native35-p-c-bundle-fe194172a6c8 and run on the box (31 Oct 2026).
P_C_SHA = {'q18a_native35_p_c.draft.sh': '1bc7dd61a36462d0bd040299a4b24e29e5630ae6da528da7dab13769753408f2',
           'q18a_native35_p_c_settings.sh': '0a8ffda9f40deb70b2e61a67600494a20fe83a5142699d65050deefc3ec1b1c5',
           'q18a_native35_p_c_declare.sh': 'a0faa2e77804b27e1955e7992e3eb6e28f3cec3fb9cbced8a6ceb8eea67bf9d7'}
# q18b-w1ab2's client as committed in fn-main 199ef23 and run on the box.
W1_TRACE_SHA = '7960fd3784f023e74896580f117751d3d4f4bb03d82848d79ac14deb9823c06c'
EXTRACT = json.loads((HERE / 'fixtures/native39_box_extract.json').read_text())
PROD_EXTRA = '--prefix-match-unit 176 --enable-prompt-tokens-details'
L_ENDING = ('stop_pressure || true\nstop_sampler || true\ncapture_logs || true\ncapture_end "$Q18A_IDENTITY_TIMEOUT_S" || true\n'
            'stop_engine || true\nrestore_launcher || true\nrestore_defaults || true\narchive_owned_next || true\nwrite_cleanup\nDONE=1\n'
            '[[ $FAILS == 0 ]] || exit 1\n[[ $INCS == 0 ]] || exit 2\nexit 0\n')
SLOT_RECEIPTS = []
OLD_ID, NEW_ID, OTHER_ID = 'a' * 64, 'b' * 64, 'c' * 64
PROXY = ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY')


def drafts_tool():
    spec = importlib.util.spec_from_file_location('native39_trunk_queue_drafts', ROOT / 'tools/native39_trunk_queue_drafts.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def env(**extra):
    out = {k: v for k, v in os.environ.items() if k not in PROXY}
    out.update(PYTHONDONTWRITEBYTECODE='1', **extra)
    return out


def functions(text, names):
    return ''.join(text[text.index(name + '() {\n'):text.index('\n}\n', text.index(name + '() {\n')) + 3] for name in names)


def segment(text, begin, end):
    assert text.count(begin) == 1 and text.count(end) == 1, (begin[:50], end[:50])
    start = text.index(begin)
    return text[start:text.index(end, start) + len(end)]


def phases(stdout):
    return dict(line[3:].split('=', 1) for line in stdout.splitlines() if line.startswith('PV '))


def utc(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + '000Z'


class Leaf:
    """Loopback engine leaf: /health, /metrics, /tokenize and streamed /v1/completions."""

    def __init__(self, tokens=None, stuck=False):
        self.ready, self.stuck, self.tokens = False, stuck, tokens
        self.first_200 = threading.Event()
        self.lock = threading.Lock()
        self.active, self.health, self.completions, self.tokenized, self.auth = 0, [], [], [], set()
        leaf = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, body=b'', kind='application/json'):
                self.send_response(code)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

            def body(self):
                leaf.auth.add(self.headers.get('Authorization'))
                return json.loads(self.rfile.read(int(self.headers['Content-Length'])))

            def do_GET(self):
                if self.path == '/health':
                    code = 200 if leaf.ready else 503
                    leaf.health.append(code)
                    self.reply(code)
                    if code == 200:
                        leaf.first_200.set()
                elif self.path == '/metrics':
                    running = leaf.active + (1 if leaf.stuck else 0)
                    text = ('vllm:num_requests_running{engine="0",model_name="m"} %d.0\n'
                            'vllm:num_requests_waiting{engine="0",model_name="m"} 0.0\n' % running)
                    self.reply(200, text.encode(), 'text/plain')
                else:
                    self.reply(404)

            def do_POST(self):
                data = self.body()
                if self.path == '/tokenize':
                    n = leaf.tokens(data['prompt'])
                    leaf.tokenized.append(dict(model=data.get('model'), chars=len(data['prompt']), tokens=n))
                    self.reply(200, json.dumps({'count': n, 'max_model_len': 262144, 'tokens': [0] * n}).encode())
                    return
                if self.path != '/v1/completions':
                    self.reply(404)
                    return
                leaf.completions.append({k: (len(v) if k == 'prompt' else v) for k, v in data.items()})
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                with leaf.lock:
                    leaf.active += 1
                try:
                    for k in range(1, 100000):
                        self.wfile.write(b'data: ' + json.dumps({'usage': {'completion_tokens': k}}).encode() + b'\n\n')
                        self.wfile.flush()
                        time.sleep(0.01)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with leaf.lock:
                        leaf.active -= 1

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.url = 'http://127.0.0.1:%d' % self.server.server_address[1]

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


FAKE_DOCKER = r'''
import datetime, os, sys, time
from pathlib import Path
fake = Path(os.environ['FAKE']); argv = sys.argv[1:]
def current():
    p = fake / 'current-id'
    return p.read_text().strip() if p.exists() else None
if argv[0] == 'inspect':
    cid = current()
    if cid is None: sys.exit(1)
    fmt = argv[argv.index('--format') + 1]
    if fmt == '{{.Id}}': print(cid)
    elif fmt == '{{.State.StartedAt}}': print((fake / 'started-at').read_text().strip())
    elif fmt == '{{.SizeRw}}': print(0)
    elif fmt.startswith('{{.Id}}|'): print('%s|true|%s|%s' % (cid, (fake / 'started-at').read_text().strip(), fake / 'json.log'))
    else: sys.exit(2)
    sys.exit(0)
if argv[0] == 'logs':
    # The json-file driver stamps a line when the daemon reads it: stamp at read time.
    raw, done = fake / 'engine.raw', 0
    while True:
        data = raw.read_bytes() if raw.exists() else b''
        while b'\n' in data[done:]:
            end = data.index(b'\n', done) + 1
            t = time.time()
            stamp = datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + '000Z '
            sys.stdout.buffer.write(stamp.encode() + data[done:end]); sys.stdout.flush(); done = end
        time.sleep(0.02)
sys.exit(2)
'''
FAKE_SYSTEMCTL = ('#!/bin/bash\ncase $1 in\n'
                  '  restart) date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$FAKE/started-at"; printf %s "$NEW_ID" > "$FAKE/current-id"; exit 0;;\n'
                  '  is-failed) exit 1;;\n  show) echo 4;;\n  *) exit 3;;\nesac\n')
BIND_LINE = ('(EngineCore pid=87) INFO 11-01 00:00:00 [s5_startup.py:390] flash-s5 startup owner bound engine_epoch=7 '
             'ranks=[0, 1, 2, 3] route_after=full route_generation_after=1\n')
SERVED_LINE = '(APIServer pid=52) INFO 11-01 00:00:00 [entry.py:139] Starting vLLM server on http://127.0.0.1:8000\n'


class RestartBind(unittest.TestCase):
    """The queue's observer start, restart, /health loop and bind receipt, from its own bytes."""

    def run_restart(self, scenario, deadline_s=20):
        trunk = TRUNK.read_text()
        seg = segment(trunk, "OLD_CONTAINER_ID=$(docker inspect --format '{{.Id}}' \"$CONTAINER\" 2>/dev/null) || OLD_CONTAINER_ID=''\n",
                      "record bind-receipt \"$?\" \"first 'flash-s5 startup owner bound' line before the first /health 200 request\"\n")
        with tempfile.TemporaryDirectory() as d, Leaf() as leaf:
            d = Path(d)
            fake, q, bindir = d / 'fake', d / 'q', d / 'bin'
            for p in (fake, q, bindir):
                p.mkdir()
            (fake / 'current-id').write_text(OLD_ID)
            (fake / 'started-at').write_text(utc(time.time() - 3600))
            (fake / 'json.log').write_text('')
            (bindir / 'docker').write_text('#!' + sys.executable + '\n' + FAKE_DOCKER)
            # An executable, not a shell function: the queue calls it through timeout(1).
            (bindir / 'systemctl').write_text(FAKE_SYSTEMCTL)
            for name in ('docker', 'systemctl'):
                (bindir / name).chmod(0o755)
            raw = fake / 'engine.raw'

            def emit(line):
                with raw.open('a') as f:
                    f.write(line)

            def persisted(text):
                p = q / 'engine-stream.log'
                return p.exists() and text in p.read_text(errors='replace')

            def wait(predicate, limit=15):
                end = time.monotonic() + limit
                while not predicate():
                    if time.monotonic() > end:
                        raise AssertionError('scenario wait timed out')
                    time.sleep(0.05)

            def attached():
                p = q / 'host-disk.jsonl'
                return (fake / 'current-id').read_text() == NEW_ID and p.exists() and p.stat().st_size > 0

            def drive():
                if scenario == 'never-ready':
                    return
                wait(attached)
                emit('(APIServer pid=52) INFO starting\n')
                if scenario in ('pass', 'other'):
                    emit(BIND_LINE)
                    wait(lambda: persisted('startup owner bound'))
                    time.sleep(0.3)
                    if scenario == 'other':
                        (fake / 'current-id').write_text(OTHER_ID)
                    leaf.ready = True
                elif scenario == 'late':
                    leaf.ready = True
                    leaf.first_200.wait(15)
                    time.sleep(0.5)
                    emit(BIND_LINE)
                else:
                    leaf.ready = True
            driver = threading.Thread(target=drive, daemon=True)
            driver.start()
            script = ('set -uo pipefail\nFAILS=0; INCS=0; N0=4; DONE=0; CHANGED=0; STARTED=0\ndeclare -A PHASE_VERDICT=()\n'
                      + functions(trunk, ('step', 'record'))
                      + 'finish() { kill -TERM "$HOST_OBSERVER_PID" 2>/dev/null; wait "$HOST_OBSERVER_PID" 2>/dev/null\n'
                        '  for k in "${!PHASE_VERDICT[@]}"; do echo "PV $k=${PHASE_VERDICT[$k]}"; done; }\ntrap finish EXIT\n'
                      + seg)
            done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=120, cwd=str(d),
                                  env=env(PATH=str(bindir) + os.pathsep + os.environ['PATH'], FAKE=str(fake), H=str(HERE),
                                          Q=str(q), CONTAINER='qwen-flash-prod', SERVICE='vllm-flash', NEW_ID=NEW_ID,
                                          PROBE_URL=leaf.url, HEALTH_DEADLINE_S=str(deadline_s), HEALTH_SLEEP='0.2',
                                          BIND_WAIT_S='2'))
            driver.join(20)
            receipt = json.loads((q / 'bind-receipt.json').read_text()) if (q / 'bind-receipt.json').exists() else None
            health = (q / 'health.tsv').read_text().splitlines()
            summary = json.loads((q / 'host-observer-summary.json').read_text())
            return done, phases(done.stdout), receipt, health, summary, leaf.health

    def common(self, done, health, summary, codes):
        self.assertIn('STEP restart PASS', done.stdout, done.stdout + done.stderr)
        self.assertEqual(health[0].split()[1], 'restart-begin')
        self.assertEqual(health[1].split()[1:], ['restart-returned', 'rc=0'])
        rows = [line.split() for line in health[2:]]
        self.assertEqual([r[2] for r in rows], [str(c) for c in codes])  # one row per request the leaf served
        self.assertEqual(rows[-1][2], '200')
        for sent, returned, _ in rows:
            self.assertLessEqual(float(sent), float(returned))
        self.assertEqual(summary['container_id'], NEW_ID[:12])  # the observer excluded the old incarnation

    def test_bind_line_before_the_first_200_request_passes(self):
        done, pv, receipt, health, summary, codes = self.run_restart('pass')
        self.common(done, health, summary, codes)
        self.assertEqual(pv['bind-receipt'], 'PASS', receipt)
        self.assertEqual(receipt['observed_containers'], [NEW_ID[:12]])
        self.assertEqual(len(receipt['bind_lines']), 1)
        self.assertEqual(receipt['bind_lines'][0]['engine_epoch'], 7)
        self.assertLess(receipt['bind_t'], receipt['first_200']['sent'])
        self.assertGreaterEqual(receipt['bind_t'], receipt['restart_begin'])
        self.assertGreater(len(codes), 1)

    def test_bind_line_after_the_first_200_returned_fails(self):
        done, pv, receipt, health, summary, codes = self.run_restart('late')
        self.common(done, health, summary, codes)
        self.assertEqual(pv['bind-receipt'], 'FAIL', receipt)
        self.assertGreater(receipt['bind_t'], receipt['first_200']['returned'])
        self.assertIn('STEP bind-receipt FAIL', done.stdout)

    def test_no_bind_line_is_inconclusive(self):
        done, pv, receipt, health, summary, codes = self.run_restart('none')
        self.common(done, health, summary, codes)
        self.assertEqual(pv['bind-receipt'], 'INCONCLUSIVE', receipt)
        self.assertEqual(receipt['bind_lines'], [])

    def test_a_log_of_another_container_is_inconclusive(self):
        done, pv, receipt, health, summary, codes = self.run_restart('other')
        self.common(done, health, summary, codes)
        self.assertEqual(pv['bind-receipt'], 'INCONCLUSIVE', receipt)
        self.assertEqual((receipt['current_container'], receipt['observed_containers']), (OTHER_ID[:12], [NEW_ID[:12]]))

    def test_health_deadline_stops_the_loop_and_the_queue(self):
        done, pv, receipt, health, summary, codes = self.run_restart('never-ready', deadline_s=2)
        self.assertIn('STEP restart FAIL', done.stdout)
        self.assertIsNone(receipt)
        self.assertNotIn('bind-receipt', pv)
        self.assertTrue(codes and set(codes) == {503})
        self.assertEqual(len(health) - 2, len(codes))


FAKE_PHASE = r'''
import json,os,sys
from pathlib import Path
fake=Path(os.environ['FAKE']);name=Path(__file__).stem
with open(fake/'calls.jsonl','a') as f:f.write(json.dumps({'name':name,'argv':sys.argv[1:]})+'\n')
def arg(key):return sys.argv[sys.argv.index(key)+1]
if name=='e3_native_w1':
    Path(sys.argv[3]).write_text(json.dumps({'schema':2,'summary':{'verdict':'COMPLETE'}}));sys.exit(0)
if name=='e3_native35_performance':
    r=int(arg('--round'))
    Path(arg('--out')).write_text(json.dumps({'grade':{'verdict':'PASS'},'native35_round':{'round':r}}));sys.exit(0)
if name=='e3_native_memory_grade':
    Path(arg('--out')).write_text(json.dumps({'verdict':'PASS'}));sys.exit(0)
sys.exit(2)
'''
QUALIFIED = {'startup': 'PASS', 'row1': 'PASS', 'rule4_controls': 'PASS', 'rule4_candidate': 'PASS', 'row4': 'PASS'}


class Tail(unittest.TestCase):
    """From the rule4 continuation through the trunk summary: the queue's own bytes."""

    def run_tail(self, verdicts, stuck=False):
        trunk = TRUNK.read_text()
        tail = trunk[trunk.index('RULE4_CONTINUATION=0\n'):]
        self.assertTrue(tail.endswith(L_ENDING))
        tail = tail[:-len(L_ENDING)]
        with tempfile.TemporaryDirectory() as d, Leaf(tokens=lambda prompt: 200000, stuck=stuck) as leaf:
            d = Path(d)
            fake, h, q = d / 'fake', d / 'h', d / 'q'
            for p in (fake, h, q):
                p.mkdir()
            for name in ('e3_native_w1', 'e3_native35_performance', 'e3_native_memory_grade', 'e3_native28_rule4_continuation'):
                (h / (name + '.py')).write_text(FAKE_PHASE)
            for name in ('w1_trace.py', 'e3_native39_trunk_checks.py', 'native35_performance_grade.py'):
                shutil.copyfile(HERE / name, h / name)  # the real client, checks and grade
            bindir = d / 'bin'
            bindir.mkdir()
            (bindir / 'systemctl').write_text(FAKE_SYSTEMCTL)
            (bindir / 'systemctl').chmod(0o755)
            corpus = d / 'corpus.txt'
            corpus.write_text('fixture corpus text ' * 64)
            script = ('set -uo pipefail\nFAILS=0; INCS=0; ENGINE_CRASHED=0; N0=4; SERVICE=vllm-flash\ndeclare -A PHASE_VERDICT=()\n'
                      + functions(trunk, ('step', 'record'))
                      + 'health_ok() { return 0; }\nhost_crash_detected() { return 1; }\nsystemctl() { echo 4; }\n'
                      + ''.join('PHASE_VERDICT[%s]=%s\n' % item for item in verdicts.items())
                      + tail + 'for k in "${!PHASE_VERDICT[@]}"; do echo "PV $k=${PHASE_VERDICT[$k]}"; done\n'
                        'echo "TOTAL FAILS=$FAILS INCS=$INCS"\n')
            done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=300, cwd=str(d),
                                  env=env(PATH=str(bindir) + os.pathsep + os.environ['PATH'], NEW_ID=NEW_ID,
                                          FAKE=str(fake), H=str(h), Q=str(q), PROBE_URL=leaf.url, MODEL='m',
                                          RUN_ID='q18a-native39-trunk-cpu-1', EPOCH='7', Q18A_FIRST_ADVANCE_READ_TIMEOUT_S='1200',
                                          Q18A_READINESS='/overlay/e3/native-overlap-readiness.json', S5_OVERLAY='/overlay',
                                          S5_LAUNCHER_NAME='run-flash-s5-test.sh', CORPUS=str(corpus),
                                          W1_PLAIN_WINDOW_S='1', W1_PLAIN_SETTLE_S='0.2', W1_PLAIN_START_DEADLINE_S='20',
                                          W1_PLAIN_TIMEOUT_S='60', IDLE_TIMEOUT_S='5',
                                          PYTHONPATH=os.pathsep.join((str(HERE), str(ROOT / 'patches/s5_engine')))))
            calls = [json.loads(line)['name'] for line in (fake / 'calls.jsonl').read_text().splitlines()] \
                if (fake / 'calls.jsonl').exists() else []
            arts = {name: json.loads((q / name).read_text()) for name in
                    ('gates.json', 'w1-plain.json', 'w1-plain-idle.json', 'trunk-summary.json', 'native35-performance-grade.json')
                    if (q / name).exists()}
            return done, phases(done.stdout), calls, arts, leaf

    def test_plain_w1_beside_native_w1_then_idle_then_both_rounds_and_the_summary(self):
        done, pv, calls, arts, leaf = self.run_tail(QUALIFIED)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(calls, ['e3_native_w1', 'e3_native35_performance', 'e3_native_memory_grade',
                                 'e3_native35_performance', 'e3_native_memory_grade'])
        self.assertEqual({k: pv[k] for k in ('w1_controls', 'w1_plain', 'w1-plain-idle', 'overlap-round-0', 'overlap-round-1')},
                         dict.fromkeys(('w1_controls', 'w1_plain', 'w1-plain-idle', 'overlap-round-0', 'overlap-round-1'), 'PASS'))
        # The 24 Oct client's own requests: 8 streams of 153537 tokens, fresh salts, one /tokenize.
        self.assertEqual(len(leaf.tokenized), 1)
        self.assertEqual(len(leaf.completions), 8)
        self.assertEqual({c['prompt'] for c in leaf.completions}, {153537})
        self.assertEqual(len({c['cache_salt'] for c in leaf.completions}), 8)
        self.assertEqual({(c['max_tokens'], c['min_tokens'], c['ignore_eos'], c['stream']) for c in leaf.completions},
                         {(16384, 16384, True, True)})
        plain = arts['w1-plain.json']
        self.assertEqual((plain['verdict'], plain['streams'], plain['stream_prompt'], len(plain['windows'])), ('PASS', 8, 153537, 3))
        self.assertEqual(leaf.active, 0)  # the client's requests ended on disconnect
        self.assertEqual(arts['w1-plain-idle.json']['verdict'], 'PASS')
        # The gates producer keeps P-c's exact outcome keys (perf.gates_allow_run requires them).
        self.assertEqual(set(arts['gates.json']['outcomes']), {'startup', 'row1', 'rule4_controls', 'rule4_candidate', 'row4', 'w1_controls'})
        summary = arts['trunk-summary.json']
        windows = [w['agg_tps'] for w in plain['windows']]
        self.assertEqual(summary['plain_w1']['window_agg_tps'], windows)
        self.assertEqual(summary['plain_w1']['median'], sorted(windows)[1])
        self.assertEqual([r['round'] for r in summary['rounds']], [0, 1])
        self.assertFalse(summary['target_is_gate'])
        self.assertNotIn('w2-idle-precondition', done.stdout)

    def test_engine_not_idle_after_the_plain_w1_withholds_w2(self):
        done, pv, calls, arts, leaf = self.run_tail(QUALIFIED, stuck=True)
        self.assertEqual(calls, ['e3_native_w1'])
        self.assertEqual((pv['w1_plain'], pv['w1-plain-idle'], pv['overlap']), ('PASS', 'INCONCLUSIVE', 'INCONCLUSIVE'))
        self.assertIn('STEP w2-idle-precondition INCONCLUSIVE', done.stdout)
        self.assertIn('STEP overlap-gates PASS', done.stdout)
        self.assertEqual([r['verdict'] for r in arts['native35-performance-grade.json']['rounds']], ['NOT_RUN', 'NOT_RUN'])
        self.assertEqual(arts['trunk-summary.json']['w1_plain_idle']['verdict'], 'INCONCLUSIVE')

    def test_unqualified_row4_withholds_both_w1_clients_and_w2(self):
        done, pv, calls, arts, leaf = self.run_tail(dict(QUALIFIED, row4='FAIL'))
        self.assertEqual(calls, [])
        self.assertEqual((leaf.completions, leaf.tokenized), ([], []))
        self.assertEqual((pv['w1_controls'], pv['w1_plain'], pv['overlap']), ('INCONCLUSIVE', 'INCONCLUSIVE', 'INCONCLUSIVE'))
        self.assertNotIn('w1-plain-idle', pv)
        self.assertEqual(arts['gates.json']['outcomes']['row4'], 'FAIL')
        self.assertIn('trunk-summary.json', arts)


def model_count(prompt):
    return length.model_tokens(prompt) if '\n' not in prompt else sum(
        length.model_tokens(part) for part in prompt.split('\n')) + prompt.count('\n')


class Guard(unittest.TestCase):
    """The queue's w2-length lines with the real guard; the /tokenize leaf counts with the declared model."""

    def run_guard(self, tokens=model_count, workload=None, url=None):
        trunk = TRUNK.read_text()
        seg = segment(trunk, "# Native39: the W2 counts the trunk manifest declares, from this engine's /tokenize, must plan.\n",
                      "rc=$?; record w2-length \"$rc\" 'declared W2 fresh/append counts equal /tokenize and both have a native plan'\n")
        with tempfile.TemporaryDirectory() as d, Leaf(tokens=tokens) as leaf:
            d = Path(d)
            h, q = d / 'h', d / 'q'
            q.mkdir()
            (h / 'workloads').mkdir(parents=True)
            # The real guard (it resolves its imports from its own directory) and the trunk manifest, or a tampered copy.
            (h / 'e3_native39_w2_length.py').symlink_to(HERE / 'e3_native39_w2_length.py')
            manifest = h / 'workloads/q18a-w2-overlap-native39.json'
            if workload is None:
                shutil.copyfile(HERE / 'workloads/q18a-w2-overlap-native39.json', manifest)
            else:
                manifest.write_text(json.dumps(workload))
            script = ('set -uo pipefail\nFAILS=0; INCS=0\ndeclare -A PHASE_VERDICT=()\n' + functions(trunk, ('step', 'record')) + seg
                      + 'for k in "${!PHASE_VERDICT[@]}"; do echo "PV $k=${PHASE_VERDICT[$k]}"; done\n')
            done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=300,
                                  env=env(H=str(h), Q=str(q), PROBE_URL=url or leaf.url, MODEL='qwen38-w8a8-tp4',
                                          VLLM_API_KEY='fixture-bearer-value'))
            out = (q / 'w2-length.json').read_text()
            return phases(done.stdout), json.loads(out), out + (q / 'w2-length.log').read_text(), leaf

    def test_declared_counts_from_tokenize_pass_and_the_bearer_is_sent_not_kept(self):
        pv, receipt, text, leaf = self.run_guard()
        manifest = json.loads((HERE / 'workloads/q18a-w2-overlap-native39.json').read_text())['length_declaration']
        self.assertEqual(pv['w2-length'], 'PASS', receipt)
        self.assertEqual({k: receipt['observed'][k]['tokens'] for k in ('fresh', 'append')},
                         {k: manifest[k]['tokens'] for k in ('fresh', 'append')})
        self.assertEqual([t['tokens'] for t in leaf.tokenized], [manifest['fresh']['tokens'], manifest['append']['tokens']])
        self.assertEqual(leaf.auth, {'Bearer fixture-bearer-value'})
        self.assertNotIn('fixture-bearer-value', text)

    def test_a_count_off_by_one_fails(self):
        pv, receipt, _, _ = self.run_guard(tokens=lambda prompt: model_count(prompt) + 1)
        self.assertEqual(pv['w2-length'], 'FAIL')
        self.assertEqual(receipt['failures'], ['fresh-count-differs-from-declaration', 'append-count-differs-from-declaration'])

    def test_a_count_in_the_refused_band_fails(self):
        manifest = json.loads((HERE / 'workloads/q18a-w2-overlap-native39.json').read_text())['length_declaration']
        shift = next(s for s in range(1, 176) if not length.band(manifest['fresh']['tokens'] + s)['plans'])
        pv, receipt, _, _ = self.run_guard(tokens=lambda prompt: model_count(prompt) + shift)
        self.assertEqual(pv['w2-length'], 'FAIL')
        self.assertIn('fresh-has-no-native-plan', receipt['failures'])

    def test_a_tampered_declaration_fails_before_any_request(self):
        data = json.loads((HERE / 'workloads/q18a-w2-overlap-native39.json').read_text())
        data['length_declaration']['fresh']['tokens'] += 1
        pv, receipt, _, leaf = self.run_guard(workload=data)
        self.assertEqual((pv['w2-length'], receipt['failures']), ('FAIL', ['manifest-length-declaration-differs-from-derivation']))
        self.assertEqual(leaf.tokenized, [])

    def test_no_engine_is_inconclusive(self):
        pv, receipt, _, _ = self.run_guard(url='http://127.0.0.1:9')
        self.assertEqual((pv['w2-length'], receipt['rc']), ('INCONCLUSIVE', 2))


class Launcher(unittest.TestCase):
    """Sourced trunk settings -> real setcfg -> real S5 launcher render -> declared identity -> host check."""

    def declared(self, settings_path):
        import test_identity_declare as tid
        import identity_declare as dec
        sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
        import make_e2_arena as e2g
        import make_s5_overlap as s5g
        from e3_private_auth import prepare
        fixture = tid.DeclareTests('test_pair_only_verifier_diff')
        fixture.setUp()
        self.addCleanup(shutil.rmtree, fixture.root, True)
        with fixture.defaults.open('a') as f:  # the found production EXTRA_ARGS (P-c capture)
            f.write('EXTRA_ARGS="%s"\n' % PROD_EXTRA)
        launch = fixture.root / 's5-launcher'
        launch.write_text(s5g._s5_launcher(e2g.launcher((ROOT / 'deploy/run-flash-prod.sh').read_text())))
        self.assertEqual(dec.sha(launch), dec.S5_LAUNCHER_SHA)
        (fixture.task / 'prod/run-flash-prod.sh').write_bytes((ROOT / 'deploy/run-flash-prod.sh').read_bytes())  # the found launcher
        names = ['e3/__init__.py', 'e3/decode-probe-evidence.json', *['e3/%s.py' % n for n in s5g.E3_MODULES],
                 *['e3/' + n for n in s5g.NATIVE_FILES], 'e3/native/build_green_helper.sh',
                 *['e3_sources/' + n for n in ('executor.py', 'core.py', 'worker.py', 'cudagraph_utils.py', 'sampler.py',
                                               'async_utils.py', 'model.py', 'api_routers.py')]]
        with (fixture.ovl / 'SHA256SUMS').open('a') as manifest:
            for name in names:
                p = fixture.ovl / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('# fixture ' + name + '\n')
                manifest.write(dec.sha(p) + '  ' + name + '\n')
        fixture.a.launcher = str(launch)
        private = fixture.root / 'auth'
        private.mkdir(mode=0o700)
        envfile = private / 'api.env'
        prepare(envfile, {'VLLM_API_KEY': tid.DUMMY})
        script = ('S5_OVERLAY=$2 S5_AUTH_ENV_FILE=$3 S5_GREEN_LIB_SHA256=$4 S5_REPLAY_LIB_SHA256=$5 '
                  'Q18A_TARGET_LABEL=s5-overlap-80-80; source "$1"; printf "%s\\0" "${S5_SETTINGS[@]}"')
        rendered = subprocess.run(['bash', '-uc', script, 'fixture', str(settings_path), dec.TASK + '/prod/overlay-v16g',
                                   str(envfile), 'a' * 64, 'b' * 64], capture_output=True, check=True)
        overrides = dec.settings(rendered.stdout.decode().rstrip('\0').split('\0'))
        policy = json.loads((HERE / 'e3_startup_mount_policy.json').read_text())
        declared, _ = dec.declare(fixture.a, overrides, fixture.image, policy, fixture.root / 'declared', install_launcher=True)
        self.assertNotIn(tid.DUMMY, json.dumps(declared))
        path = fixture.root / 'startup.json'
        path.write_text(json.dumps(declared))
        return declared, path

    def host_check(self, path):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'effective-host.json'
            rc = checks.main(['host', '--identity', str(path), '--expected-extra-args', PROD_EXTRA + ' --host 127.0.0.1',
                              '--out', str(out)])
            return rc, json.loads(out.read_text())

    def test_trunk_settings_declare_a_loopback_api_server_and_p_c_settings_do_not(self):
        declared, path = self.declared(SETTINGS)
        args = declared['identity']['command']['args']
        hosts = [args[j + 1] for j, x in enumerate(args[:-1]) if x == '--host']
        self.assertEqual(hosts, ['0.0.0.0', '127.0.0.1'])  # the launcher's own, then EXTRA_ARGS last
        self.assertEqual(args[-5:], PROD_EXTRA.split() + ['--host', '127.0.0.1'])
        self.assertEqual(declared['identity']['defaults']['EXTRA_ARGS'], PROD_EXTRA.split() + ['--host', '127.0.0.1'])
        rc, receipt = self.host_check(path)
        self.assertEqual((rc, receipt['verdict']), (0, 'PASS'), receipt)
        # The trunk declare script's own assertion on the same identity.
        decl = DECLARE.read_text()
        code = segment(decl, "args=d['identity']['command']['args']\n", "'declared API server must bind 127.0.0.1 (last --host)'\n")
        exec(compile(code, 'declare-host', 'exec'), {'d': declared})
        _, pc_path = self.declared(RUNS / 'q18a_native35_p_c_settings.sh')
        rc, receipt = self.host_check(pc_path)
        self.assertEqual((rc, receipt['identities'][0]['effective_host']), (1, '0.0.0.0'))
        with self.assertRaises(AssertionError):
            exec(compile(code, 'declare-host', 'exec'), {'d': json.loads(pc_path.read_text())})


class BoxFormats(unittest.TestCase):
    """The checks on retained box records (native39_box_extract.json)."""

    def write_q(self, d, lines, ids):
        q = Path(d)
        (q / 'engine-stream.log').write_text(''.join(line + '\n' for line in lines))
        (q / 'engine-stream.err').write_text('')
        (q / 'host-disk.jsonl').write_text(''.join(json.dumps({'container_id': i}) + '\n' for i in ids))
        return q

    def test_extract_names_its_sources(self):
        self.assertEqual(EXTRACT['schema'], 'native39-box-extract/1')
        for source in EXTRACT['sources'].values():
            self.assertRegex(source['sha256'], '^[0-9a-f]{64}$')
        self.assertEqual(hashlib.sha256((HERE / 'w1_trace.py').read_bytes()).hexdigest(), W1_TRACE_SHA)

    def test_docker_timestamps_parse_and_the_p_c_log_has_no_bind_line(self):
        pc = EXTRACT['p_c']
        first = pc['engine_stream']['head'][0]
        self.assertEqual(checks.stamp(first), datetime.datetime(2026, 10, 31, 17, 38, 26, 453470,
                                                                tzinfo=datetime.timezone.utc).timestamp())
        self.assertEqual(pc['engine_stream']['bind_lines'], 0)
        self.assertEqual(len(pc['host_disk']['container_ids']), 1)
        with tempfile.TemporaryDirectory() as d:
            q = self.write_q(d, pc['engine_stream']['head'] + pc['engine_stream']['served'], pc['host_disk']['container_ids'])
            self.assertEqual(checks.bind_lines(q), [])
            self.assertEqual(checks.observed_containers(q), set(pc['host_disk']['container_ids']))

    def test_served_host_line_is_loopback_on_w1ab2_and_not_on_p_c(self):
        b0, pc = EXTRACT['w1ab2_b0']['follow'], EXTRACT['p_c']
        self.assertEqual(len(b0['duplicate_host_warning']), 1)  # vLLM saw both --host values; the last won
        for lines, host, expected in ((b0['served'], ['127.0.0.1'], 0), (pc['engine_stream']['served'], pc['identity']['host_args'], 1)):
            with tempfile.TemporaryDirectory() as d:
                q = self.write_q(d, lines, ['x'])
                ident = Path(d) / 'identity.json'
                ident.write_text(json.dumps({'identity': {'command': {'args': ['serve', '--host', host[-1], '--port', '8000']},
                                                          'defaults': {'EXTRA_ARGS': PROD_EXTRA.split() + ['--host', '127.0.0.1']}}}))
                out = Path(d) / 'out.json'
                rc = checks.main(['host', '--identity', str(ident), '--expected-extra-args', PROD_EXTRA + ' --host 127.0.0.1',
                                  '--q', str(q), '--out', str(out)])
                self.assertEqual(rc, expected, out.read_text())
        self.assertEqual(pc['identity']['defaults_extra_args'], PROD_EXTRA.split())  # the found value the trunk carries

    def test_summary_reads_the_real_p_c_grade_and_w1ab2_client_output(self):
        with tempfile.TemporaryDirectory() as d:
            q = Path(d)
            (q / 'native35-performance-grade.json').write_text(json.dumps(EXTRACT['p_c']['grade']))
            (q / 'w1-plain.json').write_text(json.dumps(EXTRACT['w1ab2_b0']['w1']))
            self.assertEqual(checks.main(['summary', '--q', str(q), '--out', str(q / 's.json')]), 0)
            s = json.loads((q / 's.json').read_text())
        self.assertEqual(s['plain_w1']['window_agg_tps'], [391.8, 384.0, 394.4])
        self.assertEqual(s['plain_w1']['median'], 391.8)
        self.assertEqual((s['native_w1_full']['median'], s['native_w1_full']['denominator']), (227.049, 227.199))
        self.assertAlmostEqual(s['native_w1_full_over_plain_w1'], 227.199 / 391.8)
        self.assertEqual([r['verdict'] for r in s['rounds']], ['INCONCLUSIVE', 'NOT_RUN'])
        self.assertIsNone(s['rounds'][0]['decode_tokens_per_s_during_fresh_prefill'])  # P-c fresh had no native plan
        self.assertEqual(s['bind_receipt'], None)

    def test_bind_epoch_against_the_p_c_startup_epoch(self):
        epoch = EXTRACT['p_c']['startup_engine_epoch']
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'startup.json').write_text(json.dumps({'engine_epoch': epoch, 'verdict': 'PASS'}))
            for line_epoch, rc in ((epoch, 0), (epoch + 1, 1)):
                (d / 'bind.json').write_text(json.dumps({'verdict': 'PASS', 'bind_lines': [{'engine_epoch': line_epoch}]}))
                out = d / ('epoch-%d.json' % rc)
                self.assertEqual(checks.main(['bind-epoch', '--bind', str(d / 'bind.json'), '--startup', str(d / 'startup.json'),
                                              '--out', str(out)]), rc)

    def test_found_extra_args_is_compared_silently(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for text, rc in (('EXTRA_ARGS="%s"\nHF_TOKEN="fixture-private-value"\n' % PROD_EXTRA, 0),
                             ('EXTRA_ARGS="%s --max-num-seqs 9"\n' % PROD_EXTRA, 1)):
                (d / 'defaults').write_text(text)
                out = d / ('found-%d.json' % rc)
                self.assertEqual(checks.main(['found-extra-args', '--defaults', str(d / 'defaults'), '--expected', PROD_EXTRA,
                                              '--out', str(out)]), rc)
                self.assertNotIn('max-num-seqs', out.read_text())
                self.assertNotIn('fixture-private-value', out.read_text())


class Derivation(unittest.TestCase):
    def test_trunk_texts_are_the_declared_derivation_of_the_p_c_texts(self):
        for name, digest in P_C_SHA.items():
            self.assertEqual(hashlib.sha256((RUNS / name).read_bytes()).hexdigest(), digest, name)
        for path, text in drafts_tool().generate().items():
            self.assertEqual(path.read_text(), text, path.name)
        self.assertEqual(hashlib.sha256((HERE / 'w1_trace.py').read_bytes()).hexdigest(), W1_TRACE_SHA)

    def test_every_p_c_phase_stays_and_only_the_declared_phases_are_added(self):
        import re

        def names(text):
            return set(re.findall(r'(?:^|[;\s])(?:record|step) "?([A-Za-z0-9_.$\-{}]+)"? ', text, re.M))
        pc, trunk = names((RUNS / 'q18a_native35_p_c.draft.sh').read_text()), names(TRUNK.read_text())
        self.assertLessEqual(pc, trunk)
        self.assertEqual(trunk - pc, {'bind-receipt', 'effective-host', 'bind-epoch', 'w2-length', 'w1_plain', 'w1-plain-idle',
                                      'w2-idle-precondition', 'trunk-summary'})

    def test_scripts_parse_and_name_only_the_trunk(self):
        for path in (TRUNK, SETTINGS, DECLARE):
            done = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            text = path.read_text()
            self.assertNotIn('q18a-native35-p-c', text)
            self.assertNotIn('workloads/q18a-w2-overlap.json', text)
        self.assertIn('--kill-after=900s 16200s env', TRUNK.read_text())


class Window(unittest.TestCase):
    def test_bounds_unchanged_and_the_trunk_budget_fits(self):
        self.assertEqual((gate.NOT_BEFORE, gate.HARD_END), (native35p_gate.NOT_BEFORE, native35p_gate.HARD_END))
        self.assertEqual(gate.HARD_END, datetime.datetime(2026, 11, 2, 6, tzinfo=datetime.timezone.utc).timestamp())
        self.assertEqual((gate.TRUNK_ACTIVE_S, gate.TRUNK_CLEANUP_MARGIN_S, gate.TRUNK_ACTIVE_AND_CLEANUP), (16200, 900, 17100))
        self.assertEqual(gate.ACTIVE_DERIVATION[:len(native35p_gate.ACTIVE_DERIVATION)], native35p_gate.ACTIVE_DERIVATION)
        self.assertLessEqual(sum(s for _, s, _ in gate.ACTIVE_DERIVATION), gate.TRUNK_ACTIVE_S)
        self.assertFalse(gate.trunk_window_ok(gate.NOT_BEFORE - 1))
        self.assertTrue(gate.trunk_window_ok(gate.HARD_END - 17100))
        self.assertFalse(gate.trunk_window_ok(gate.HARD_END - 17100 + 1))
        self.assertEqual(datetime.datetime.fromtimestamp(gate.HARD_END - 17100, datetime.timezone.utc).isoformat(),
                         '2026-11-02T01:15:00+00:00')

    def test_trunk_draft_window_slot_executes_at_real_wall_clock(self):
        source = TRUNK.read_text()
        actual = segment(source, "python3 - <<'PY_SLOT'", '\nPY_SLOT\n')
        self.assertIn('from native39_trunk_gate import trunk_window_ok as window_ok', actual)
        done = subprocess.run(['bash', '-c', 'fail_early() { return 2; }\n' + actual], env=env(HARNESS_DIR=str(HERE)),
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        SLOT_RECEIPTS.append(dict(draft=TRUNK.name, verdict='PASS', rc=0, slot_sha256=hashlib.sha256(actual.encode()).hexdigest(),
                                  real_wall_clock=True, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  engine_contact=False, active_s=gate.TRUNK_ACTIVE_S, cleanup_margin_s=gate.TRUNK_CLEANUP_MARGIN_S))


if __name__ == '__main__':
    unittest.main()
