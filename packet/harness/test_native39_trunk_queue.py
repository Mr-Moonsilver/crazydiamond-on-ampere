"""Native39 trunk queue: the exact derivation from the P-c texts, the trunk window at real wall clock,
and the queue's own new segments executed under bash against loopback leaves.  CPU only, no box action.

Connected compositions (the queue's bytes with the real checks, guard, client, observer and grade):
  RestartBind  restart -> /health loop -> bind receipt: the queue segment, the real host observer
               following a fake docker CLI, real curl against a loopback /health leaf.
  Tail         row4 declared NOT_RUN -> instruments (per-rank maps) -> sealed native W1 -> ingress ->
               typed FULL + idle -> plain W1 -> route check -> ingress -> gates -> per round idle,
               instruments, W2 round, ingress -> grade -> instruments end -> returns -> trunk summary: the
               queue segment with the real sealed native W1 (e3_native39_w1 over e3_w1_decode_driver and
               the real session client), checks, w1_trace client and grade against a loopback engine leaf
               (native session, chat and completion streams, access log); fake W2 round and memory grade.
  RoundWrapper the real e3_native39_round CLI over the unchanged same-queue gate: only the declared row4
               NOT_RUN shape reaches it as PASS.
  Guard        the queue's w2-length lines with the real guard against a loopback /tokenize leaf.
  Launcher     the trunk settings file sourced, the real setcfg algorithm, the real S5 launcher render
               and declaration projection, then the queue's host check on the declared identity.
  BoxFormats   the checks on retained box extracts (harness/fixtures/native39_box_extract.json,
               tools/native39_box_extract.py): docker --timestamps lines, host-disk rows, the P-c
               capture's --host, the w1ab2 served-host line, the P-c grade and the w1ab2 client output;
               real uvicorn access lines (native39_box_access_extract.json,
               tools/native39_box_access_extract.py): the w1ab2 loopback plain run and the P-c 0.0.0.0 log.
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
ACCESS = json.loads((HERE / 'fixtures/native39_box_access_extract.json').read_text())
PROD_EXTRA = '--prefix-match-unit 176 --enable-prompt-tokens-details'
PROD_LAUNCHER_SHA = '98d6d2a21927b8943f1ae9b0b43616757e80adac0c628e65f482e4332e3cf126'  # deploy/run-flash-prod.sh
L_ENDING = ('stop_pressure || true\nstop_sampler || true\ncapture_logs || true\ncapture_end "$Q18A_IDENTITY_TIMEOUT_S" || true\n'
            'stop_engine || true\nrestore_launcher || true\nrestore_defaults || true\narchive_owned_next || true\nwrite_cleanup\nDONE=1\n'
            '[[ $FAILS == 0 ]] || exit 1\n[[ $INCS == 0 ]] || exit 2\nexit 0\n')
SLOT_RECEIPTS = []
QUEUE_RECEIPTS = []  # one row per connected case, appended only after its assertions (the bundle probe asserts the set)


def declare_sha():
    import identity_declare
    return identity_declare.S5_LAUNCHER_SHA


def record_case(case, **fields):
    QUEUE_RECEIPTS.append(dict(case=case, verdict='PASS', **fields))
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


class NativeSession:
    """Engine double for the native session endpoint and the chat cohort (fixture leaf queue-native-session).

    Replaces NativeOverlapSession (patches/s5_engine/native_overlap_session.py) over the engine driver's
    status record (engine_driver._status_record) and the worker rows (native_overlap_worker: row fields and
    live_decode census).  Equivalent for the queue because it answers the same actions with the same field
    names the harness reads: typed TransitionAck rows (engine_driver.transition_ack_rows), the keep-pause
    barrier rows with live_request_uids = the engine ids of the open chat streams (SSE id + '-<8 hex>', as
    the pinned input processor assigns them), scheduler_requests = live + waiting while running and 0 under
    pause (the pinned PAUSED_ALL behaviour, engine_driver.py:2994), and route generations that advance on
    every transition.  Census counters advance only while resumed with live decoders.
    """

    def __init__(self, mode=''):
        import wire_v1
        self.full_bank, self.split_bank = wire_v1.ROUTE_BANK_FULL, wire_v1.ROUTE_BANK_SPLIT
        self.mode, self.lock = mode, threading.Lock()
        self.epoch, self.route, self.generation, self.paused = 7, 'full', 1, False
        self.live, self.hidden, self.foreign = {}, [], set()
        self.dispatches, self.ran, self.ack_ns = 0, False, time.monotonic_ns()
        self.calls, self.prepares, self.split_entries, self.stuck, self.chats = [], 0, 0, False, 0
        self.injected = False
        self.log = None
        clock = {'basis': 'CLOCK_MONOTONIC', 'proc': 'engine'}
        clock['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        clock['time_namespace'] = os.readlink('/proc/self/ns/time')
        self.clock = clock

    def foreign_line(self):
        if self.log:
            self.log('127.0.0.1', 1, 'POST', '/v1/chat/completions', 200)

    def rows(self, now):
        uids = sorted(self.live)
        c = self.dispatches
        census = {'dispatches': c, 'decode_dispatches': c, 'non_decode_dispatches': 0, 'full_graph_dispatches': c,
                  'dispatch_errors': 0, 'rejected': 0}
        return [{'schema': 1, 'action': 'barrier', 'worker_rank': r, 'pid': 1001 + r, 'engine_epoch': self.epoch,
                 'engine_epoch_ok': True, 'storage_epoch': 1, 'worker_incarnation': 'worker-%d' % r, 'route': self.route,
                 'route_generation': self.generation, 'observed_clock_ns': now, 'clock_basis': 'CLOCK_MONOTONIC',
                 'prefill_requests': 0, 'decode_eligible': len(uids), 'live_decode_requests': len(uids),
                 'running_requests': len(uids), 'barrier_clean': True, 'drained_streams': 1, 'pending_native_replays': 0,
                 'retained_rows': 0, 'pending_prefill_streams': 0, 'pending_output_copies': 0,
                 'live_decode': {'basis': 'worker-native-live-decode-v1', 'available': True, 'live_requests': len(uids),
                                 'prefill_requests': 0, 'decode_eligible': len(uids), 'live_request_uids': uids[:64],
                                 'census': dict(census)}} for r in range(4)]

    def acks(self, bank):
        self.ack_ns = time.monotonic_ns()
        return [{'typed': 'TransitionAck', 'worker_rank': r, 'engine_epoch': self.epoch,
                 'expected_route_generation': self.generation, 'route_generation': self.generation,
                 'applied_route_generation': self.generation, 'target_bank': bank, 'applied_bank': bank,
                 'applied_storage_epochs': [1], 'decode_replay_watermark': 0, 'drained_streams': 1,
                 'observed_ns': self.ack_ns, 'clock_basis': 'CLOCK_MONOTONIC'} for r in range(4)]

    def status(self, now):
        unfinished = 0 if self.paused else len(self.live) + len(self.hidden)
        return {'schema': 1, 'mode': 'native-overlap', 'engine_epoch': self.epoch, 'route': self.route,
                'route_generation': self.generation, 'paused': self.paused, 'transition_pending': False,
                'control_pending': None, 'return_pending': None, 'scheduler_requests': unfinished,
                'observed_clock_ns': now, 'clock': dict(self.clock, sample_ns=now), 'workers': self.rows(now)}

    def pause(self):
        if not self.paused and self.ran:
            self.dispatches += 10
        self.paused, self.ran = True, False
        return {'paused': True, 'mode': 'keep', 'clear_cache': False}

    def resume(self):
        self.paused = False
        self.live.update((uid, True) for uid in self.hidden)
        self.hidden = []
        self.ran = bool(self.live)
        return {'resumed': True}

    def transition(self, route):
        if self.route != route:
            self.route, self.generation = route, self.generation + 1
        return self.acks(self.full_bank if route == 'full' else self.split_bank)

    def act(self, command):
        """One session action; returns (http status, reply)."""
        with self.lock:
            action, now = command.get('action'), time.monotonic_ns()
            self.calls.append(action)
            base = {'schema': 1, 'action': action, 'run_id': command.get('run_id'), 'engine_epoch': self.epoch}
            if self.stuck and action in ('prepare', 'leave_full'):
                return 409, {'error': 'stuck-split'}
            entry = self.prepares == 2 and len(self.live) >= 8 and self.route == 'full' and not self.injected
            if action == 'status':
                if 'foreign-entry' in self.mode and entry:  # a foreign decoder registered just before the entry read
                    uid = 'chatcmpl-f0f0f0f0f0f0f0f0-0badc0de'
                    self.live[uid], self.injected = True, True
                    self.foreign.add(uid)
                    self.foreign_line()
                if 'swap-entry' in self.mode and entry:  # one cohort request gone engine-side, a foreign one in its place
                    self.live.pop(sorted(self.live)[0])
                    uid = 'chatcmpl-d0d0d0d0d0d0d0d0-0badbeef'
                    self.live[uid], self.injected = True, True
                    self.foreign.add(uid)
                    self.foreign_line()
                return 200, dict(self.status(now), **base)
            if action == 'pause':
                pause = self.pause()
                if 'hidden-arrival' in self.mode and entry:
                    uid = 'chatcmpl-e0e0e0e0e0e0e0e0-0badf00d'
                    self.hidden.append(uid)  # admitted after the status read: waiting, invisible under PAUSED_ALL
                    self.foreign.add(uid)
                    self.injected = True
                    self.foreign_line()
                return 200, dict(base, pause=pause, barrier={'workers': self.rows(now)}, snapshot=self.status(now),
                                 route=self.route, route_generation=self.generation)
            if action == 'resume':
                if 'transient-foreign' in self.mode and self.route == 'split' and not self.injected:
                    self.injected = True  # a foreign request inside the split window that ends before the end barrier
                    self.foreign_line()
                return 200, dict(base, **self.resume())
            if action == 'prepare':
                self.prepares += 1
                pause = self.pause()
                acks = self.transition(command.get('target_route'))
                return 200, dict(base, target_route=command.get('target_route'), route=self.route, phase=command.get('phase'),
                                 paused=pause, typed_acks=True, transition_acks=acks, resumed=self.resume())
            if action == 'pause_enter_split':
                pause = self.pause()
                acks = self.transition('split')
                self.split_entries += 1
                if 'stuck-split' in self.mode:
                    self.stuck = True
                return 200, dict(base, route='split', pause=pause, phase=command.get('phase'), typed_acks=True,
                                 transition_acks=acks)
            if action == 'leave_full':
                pause = self.pause()
                census = {'workers': self.rows(time.monotonic_ns())}
                acks = self.transition('full')
                for uid in list(self.foreign):  # the foreign requests end with the cohort's window
                    self.live.pop(uid, None)
                self.foreign = set()
                drain = [{'worker_rank': a['worker_rank'], 'drained_streams': 1,
                          'applied_route_generation': a['applied_route_generation'], 'decode_replay_watermark': 0}
                         for a in acks]
                return 200, dict(base, route='full', pause=pause, census_before_return=census, transition_acks=acks,
                                 worker_drain=drain, typed_acks=True, resume=self.resume())
            return 404, {'error': 'action-not-exposed'}

    def open_chat(self):
        with self.lock:
            sid = 'chatcmpl-%s' % os.urandom(8).hex()
            uid = '%s-%s' % (sid, os.urandom(4).hex())
            self.live[uid] = False
            self.chats += 1
            if not self.paused:
                self.ran = True
            return sid, uid

    def close_chat(self, uid):
        with self.lock:
            self.live.pop(uid, None)


class Leaf:
    """Loopback engine leaf: /health, /metrics, /tokenize, streamed /v1/completions; with a NativeSession also
    the native session endpoint and streamed /v1/chat/completions with SSE ids (fixture leaf queue-chat-stream);
    with a log path the uvicorn
    access lines docker would stamp (fixture leaf queue-access-log: streaming lines at response start,
    others at completion, as uvicorn logs them)."""

    def __init__(self, tokens=None, stuck=False, session=None, log=None):
        self.ready, self.stuck, self.tokens = False, stuck, tokens
        self.session, self.log_path = session, log
        self.first_200 = threading.Event()
        self.lock = threading.Lock()
        self.active, self.health, self.completions, self.tokenized, self.auth = 0, [], [], [], set()
        leaf = self
        if session is not None:
            session.log = self.access

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def access(self, code):
                leaf.access(self.client_address[0], self.client_address[1], self.command, self.path, code)

            def reply(self, code, body=b'', kind='application/json'):
                self.send_response(code)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                self.access(code)

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
                    live = len(leaf.session.live) if leaf.session else 0
                    waiting = len(leaf.session.hidden) if leaf.session else 0
                    running = leaf.active + live + (1 if leaf.stuck else 0)
                    text = ('vllm:num_requests_running{engine="0",model_name="m"} %d.0\n'
                            'vllm:num_requests_waiting{engine="0",model_name="m"} %d.0\n' % (running, waiting))
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
                if self.path == '/v1/flash_s5_native_overlap' and leaf.session is not None:
                    code, reply = leaf.session.act(data)
                    self.reply(code, json.dumps(reply).encode())
                    return
                if self.path == '/v1/chat/completions' and leaf.session is not None:
                    self.chat()
                    return
                if self.path != '/v1/completions':
                    self.reply(404)
                    return
                leaf.completions.append({k: (len(v) if k == 'prompt' else v) for k, v in data.items()})
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.access(200)
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

            def chat(self):
                session = leaf.session
                sid, uid = session.open_chat()
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.access(200)
                k = 0
                try:
                    while True:
                        if session.paused:
                            self.wfile.write(b': paused\n\n')  # no token under a keep-pause; detects disconnect
                        else:
                            k += 1
                            event = {'id': sid, 'object': 'chat.completion.chunk',
                                     'choices': [{'index': 0, 'delta': {'content': 't'}, 'finish_reason': None}],
                                     'usage': {'prompt_tokens': 40, 'completion_tokens': k, 'total_tokens': 40 + k}}
                            self.wfile.write(b'data: ' + json.dumps(event).encode() + b'\n\n')
                        self.wfile.flush()
                        time.sleep(0.01)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    session.close_chat(uid)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.server.request_queue_size = 64
        self.url = 'http://127.0.0.1:%d' % self.server.server_address[1]

    def access(self, addr, port, method, path, code):
        if self.log_path is None:
            return
        line = '%s (APIServer pid=52) INFO:     %s:%d - "%s %s HTTP/1.1" %d OK\n' % (
            utc(time.time()), addr, port, method, path, code)
        with self.lock, open(self.log_path, 'a') as f:
            f.write(line)

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
if argv[0] == 'exec':
    # Fixture leaf queue-docker-exec-maps: `docker exec <id> cat /proc/<pid>/maps` reads FAKE/maps/<pid>.
    p = fake / 'maps' / argv[-1].split('/')[2]
    if argv[2:4] != ['cat', argv[-1]] or argv[1] != current() or not p.exists(): sys.exit(1)
    sys.stdout.write(p.read_text()); sys.exit(0)
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
        record_case('bind-before-first-200', bind=receipt['verdict'], health_samples=len(codes), real_host_observer=True)

    def test_bind_line_after_the_first_200_returned_fails(self):
        done, pv, receipt, health, summary, codes = self.run_restart('late')
        self.common(done, health, summary, codes)
        self.assertEqual(pv['bind-receipt'], 'FAIL', receipt)
        self.assertGreater(receipt['bind_t'], receipt['first_200']['returned'])
        self.assertIn('STEP bind-receipt FAIL', done.stdout)
        record_case('bind-after-first-200', bind=receipt['verdict'], health_samples=len(codes), real_host_observer=True)

    def test_no_bind_line_is_inconclusive(self):
        done, pv, receipt, health, summary, codes = self.run_restart('none')
        self.common(done, health, summary, codes)
        self.assertEqual(pv['bind-receipt'], 'INCONCLUSIVE', receipt)
        self.assertEqual(receipt['bind_lines'], [])
        record_case('bind-absent', bind=receipt['verdict'], health_samples=len(codes), real_host_observer=True)

    def test_a_log_of_another_container_is_inconclusive(self):
        done, pv, receipt, health, summary, codes = self.run_restart('other')
        self.common(done, health, summary, codes)
        self.assertEqual(pv['bind-receipt'], 'INCONCLUSIVE', receipt)
        self.assertEqual((receipt['current_container'], receipt['observed_containers']), (OTHER_ID[:12], [NEW_ID[:12]]))
        record_case('bind-other-container', bind=receipt['verdict'], health_samples=len(codes), real_host_observer=True)

    def test_health_deadline_stops_the_loop_and_the_queue(self):
        done, pv, receipt, health, summary, codes = self.run_restart('never-ready', deadline_s=2)
        self.assertIn('STEP restart FAIL', done.stdout)
        self.assertIsNone(receipt)
        self.assertNotIn('bind-receipt', pv)
        self.assertTrue(codes and set(codes) == {503})
        self.assertEqual(len(health) - 2, len(codes))
        record_case('health-deadline', bind=None, health_samples=len(codes), real_host_observer=True)


FAKE_PHASE = r'''
import json,os,sys,time,urllib.request
from pathlib import Path
fake=Path(os.environ['FAKE']);name=Path(__file__).stem
with open(fake/'calls.jsonl','a') as f:f.write(json.dumps({'name':name,'argv':sys.argv[1:]})+'\n')
def arg(key):return sys.argv[sys.argv.index(key)+1]
if name=='e3_native39_round':
    # Fixture leaf queue-phase-executables: the round's 14 generation requests (3 lanes x fresh/append + 8 standing
    # decoders), each opened and closed, inside the record's wall span; record shape as e3_native_performance.run.
    r=int(arg('--round'));t0=time.time();url=os.environ['PROBE_URL']
    for i in range(14):
        req=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps({'model':'m','messages':[],'stream':True}).encode(),
            headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['VLLM_API_KEY']})
        with urllib.request.urlopen(req,timeout=10) as resp:resp.readline()
    lanes={lane:{family:{'request':{'send_at_ns':time.monotonic_ns()},'complete':True} for family in ('fresh','append')}
           for lane in ('prefill_full','prefill_split','overlap')}
    for family in ('fresh','append'):lanes['overlap'][family]['decode_window']={'streams':[{'stream':i} for i in range(8)]}
    Path(arg('--out')).write_text(json.dumps({'grade':{'verdict':'PASS'},'native35_round':{'round':r},'lanes':lanes,
        'wall_anchor_s':t0,'wall_end_s':time.time()}));sys.exit(0)
if name=='e3_native_memory_grade':
    Path(arg('--out')).write_text(json.dumps({'verdict':'PASS'}));sys.exit(0)
sys.exit(2)
'''
QUALIFIED = {'startup': 'PASS', 'row1': 'PASS', 'rule4_controls': 'PASS', 'rule4_candidate': 'PASS'}
# Fixture leaf queue-w1-test-workload: the q17b-w1 manifest's shape with 8 streams, full then split, 0.4 s
# windows and 40-token prompts (test_e3_w1_decode_driver.BASE_MANIFEST counts); the driver code path is the same.
TEST_W1 = {'schema': 1, 'workload': 'test-small', 'streams': 8, 'prompt_tokens': 40, 'seed': 1, 'generation_budget': 100000,
           'repeats': 2, 'route_plan': ['full', 'split'],
           'request_shape': {'temperature': 0.7, 'min_tokens': 'equal_max', 'include_usage': True,
                             'continuous_usage_stats': True, 'enable_thinking': False},
           'window_s': 0.4, 'open_timeout_s': 10.0, 'pause_timeout_s': 5.0, 'cache_preparation': False,
           'completion_policy': {'per_request_delta_min': 1, 'per_request_delta_max': 100000, 'per_request_start_min': 0,
                                 'per_request_start_max': 100000, 'matched_across_routes': True},
           'context_position_policy': 'same prompt, seed and context position per stream across routes', 'block_size': 16}
CLEAN_MAPS = '7f00-7f01 r-xp 0 00:00 0 /usr/lib/x86_64-linux-gnu/libcuda.so.580.95.05\n7f02-7f03 r-xp 0 00:00 0 /usr/lib/libcupti.so.13\n'
OBSERVER_MAPS = CLEAN_MAPS + ('7f04-7f05 r-xp 0 00:00 0 /usr/local/lib/python3.12/dist-packages/vllm/v1/flash_e3/native/'
                              'flash_s5_replay_observer.so\n')
ARTS = ('gates.json', 'w1.json', 'w1-plain.json', 'w1-plain-idle.json', 'w1-plain-full-idle.json', 'w1-plain-route.json',
        'instruments-w1.json', 'instruments-round-0.json', 'instruments-round-1.json', 'instruments-end.json',
        'ingress-w1.json', 'ingress-plain.json', 'ingress-round-0.json', 'ingress-round-1.json', 'returns.json',
        'trunk-summary.json', 'native35-performance-grade.json')


class Tail(unittest.TestCase):
    """From the row4 declaration through the trunk summary: the queue's own bytes, with the real sealed native
    W1, checks (instruments, ingress, full-idle, route-check, idle, returns, summary), plain client and grade
    against the loopback engine leaf (native session, chat and completion streams, access log)."""

    def run_tail(self, verdicts=QUALIFIED, mode='', maps=CLEAN_MAPS, mapped_rank=None, stuck=False):
        trunk = TRUNK.read_text()
        declared = segment(trunk, 'ROW4_DECLARED_NOT_RUN=1\n', 'export ROW4_DECLARED_NOT_RUN\n')
        tail = trunk[trunk.index('RULE4_CONTINUATION=0\n'):]
        self.assertTrue(tail.endswith(L_ENDING))
        tail = tail[:-len(L_ENDING)]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fake, h, q, bindir = d / 'fake', d / 'h', d / 'q', d / 'bin'
            for p in (fake, h, q, bindir, fake / 'maps', h / 'workloads'):
                p.mkdir()
            session = NativeSession(mode)
            with Leaf(tokens=lambda prompt: 200000, stuck=stuck, session=session, log=q / 'engine-stream.log') as leaf:
                for name in ('e3_native39_round', 'e3_native_memory_grade', 'e3_native28_rule4_continuation'):
                    (h / (name + '.py')).write_text(FAKE_PHASE)
                for name in ('w1_trace.py', 'e3_native39_trunk_checks.py', 'native35_performance_grade.py', 'e3_native39_w1.py'):
                    shutil.copyfile(HERE / name, h / name)  # the real sealed W1, client, checks and grade
                (h / 'workloads/q17b-w1.json').write_text(json.dumps(TEST_W1))
                (fake / 'current-id').write_text(NEW_ID)
                for rank in range(4):
                    text = maps if mapped_rank is None or rank != mapped_rank else OBSERVER_MAPS
                    if text is not None:
                        (fake / 'maps' / str(1001 + rank)).write_text(text)
                (bindir / 'docker').write_text('#!' + sys.executable + '\n' + FAKE_DOCKER)
                (bindir / 'systemctl').write_text(FAKE_SYSTEMCTL)
                for name in ('docker', 'systemctl'):
                    (bindir / name).chmod(0o755)
                corpus = d / 'corpus.txt'
                corpus.write_text('fixture corpus text ' * 64)
                script = ('set -uo pipefail\nFAILS=0; INCS=0; ENGINE_CRASHED=0; N0=4; SERVICE=vllm-flash\ndeclare -A PHASE_VERDICT=()\n'
                          + functions(trunk, ('step', 'record'))
                          + 'health_ok() { return 0; }\nhost_crash_detected() { return 1; }\nsystemctl() { echo 4; }\n'
                          + ''.join('PHASE_VERDICT[%s]=%s\n' % item for item in verdicts.items())
                          + declared + tail + 'for k in "${!PHASE_VERDICT[@]}"; do echo "PV $k=${PHASE_VERDICT[$k]}"; done\n'
                            'echo "TOTAL FAILS=$FAILS INCS=$INCS"\n')
                done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=300, cwd=str(d),
                                      env=env(PATH=str(bindir) + os.pathsep + os.environ['PATH'], NEW_ID=NEW_ID,
                                              FAKE=str(fake), H=str(h), Q=str(q), PROBE_URL=leaf.url, MODEL='m',
                                              CONTAINER='qwen-flash-prod', VLLM_API_KEY='fixture-key-not-a-secret',
                                              RUN_ID='q18a-native39-trunk-cpu-1', EPOCH='7', Q18A_FIRST_ADVANCE_READ_TIMEOUT_S='1200',
                                              Q18A_READINESS='/overlay/e3/native-overlap-readiness.json', S5_OVERLAY='/overlay',
                                              S5_LAUNCHER_NAME='run-flash-s5-test.sh', CORPUS=str(corpus),
                                              W1_PLAIN_WINDOW_S='1', W1_PLAIN_SETTLE_S='0.2', W1_PLAIN_START_DEADLINE_S='20',
                                              W1_PLAIN_TIMEOUT_S='60', IDLE_TIMEOUT_S='5',
                                              PYTHONPATH=os.pathsep.join((str(HERE), str(ROOT / 'patches/s5_engine')))))
            calls = [json.loads(line)['name'] for line in (fake / 'calls.jsonl').read_text().splitlines()] \
                if (fake / 'calls.jsonl').exists() else []
            arts = {name: json.loads((q / name).read_text()) for name in ARTS if (q / name).exists()}
            arts['_q'] = str(q)
            return done, phases(done.stdout), calls, arts, leaf, session

    def test_observer_off_sealed_w1_plain_on_typed_full_ingress_rounds_returns_and_summary(self):
        done, pv, calls, arts, leaf, session = self.run_tail()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(calls, ['e3_native39_round', 'e3_native_memory_grade', 'e3_native39_round', 'e3_native_memory_grade'])
        names = ('instruments-w1', 'w1_controls', 'ingress-w1', 'w1-plain-full-idle', 'w1_plain', 'w1-plain-route',
                 'ingress-plain', 'w1-plain-idle', 'w2-idle-round-0', 'instruments-round-0', 'overlap-round-0',
                 'ingress-round-0', 'w2-idle-round-1', 'instruments-round-1', 'overlap-round-1', 'ingress-round-1',
                 'instruments-end')
        self.assertEqual({k: pv.get(k) for k in names}, dict.fromkeys(names, 'PASS'), done.stdout)
        self.assertEqual(pv['row4'], 'NOT_RUN')
        self.assertIn('STEP row4 NOT_RUN declared', done.stdout)
        self.assertIn('STEP returns PASS', done.stdout)
        # Observer and offers off on the actual rank processes, the same four throughout.
        for name in ('instruments-w1', 'instruments-round-0', 'instruments-round-1', 'instruments-end'):
            ranks = arts[name + '.json']['ranks']
            self.assertEqual({r['observer_mapped'] for r in ranks.values()}, {False}, name)
            self.assertEqual(sorted(r['pid'] for r in ranks.values()), [1001, 1002, 1003, 1004])
        # The sealed W1: exact cohort ids at entry, begin and end of both repeats; split entered once.
        w1 = arts['w1.json']
        self.assertEqual(w1['summary']['verdict'], 'COMPLETE', [r['reasons'] for r in w1['repeats']])
        self.assertEqual([r['seal'] for r in w1['repeats']], [{'entry': [], 'begin': [], 'end': []}] * 2)
        self.assertEqual(session.split_entries, 1)
        self.assertEqual([len(set(r['cohort']['client_ports'])) for r in w1['repeats']], [8, 8])
        self.assertEqual(arts['ingress-w1.json']['measurement'], 'VALID')
        # The plain W1 ran on a typed FULL that stayed at one generation; 8 x 153537 from the real client.
        full = arts['w1-plain-full-idle.json']
        self.assertEqual((full['prepare']['typed_acks'], full['status']['route'], full['status']['scheduler_requests']),
                         (True, 'full', 0))
        self.assertEqual(arts['w1-plain-route.json']['status']['route_generation'], full['route_generation'])
        self.assertEqual(len(leaf.completions), 8)
        self.assertEqual({c['prompt'] for c in leaf.completions}, {153537})
        plain = arts['w1-plain.json']
        self.assertEqual((plain['verdict'], plain['streams'], len(plain['windows'])), ('PASS', 8, 3))
        self.assertEqual(arts['ingress-plain.json']['spans'][0]['generation_lines'], 8)
        self.assertEqual([arts['ingress-round-%d.json' % i]['spans'][0]['generation_lines'] for i in (0, 1)], [14, 14])
        # Gates carry the declared row4 NOT_RUN; the round wrapper reads exactly that shape.
        gates = arts['gates.json']
        self.assertEqual(set(gates['outcomes']), {'startup', 'row1', 'rule4_controls', 'rule4_candidate', 'row4', 'w1_controls'})
        self.assertEqual((gates['outcomes']['row4'], gates['row4_declared_not_run']), ('NOT_RUN', True))
        import e3_native39_round as round39
        self.assertEqual(gates['row4_not_run_reason'], round39.ROW4_NOT_RUN_REASON)
        self.assertTrue(round39.gates_allow_run(gates, arts['_q'] + '/gates.json'))
        # Return timing: the split repeat's FULL return, per-rank acks, missing stamps named, no observation window.
        ret = arts['returns.json']
        self.assertEqual([r['source'] for r in ret['rows']], ['w1:repeat1'])
        row = ret['rows'][0]
        self.assertEqual(sorted(row['ranks']), ['0', '1', '2', '3'])
        self.assertTrue(row['clock_shared'])
        self.assertGreater(row['client_send_to_all_rank_ack_ms'], 0)
        self.assertIn(row['band'], ('at-or-below-row1-1.5ms', 'between-row1-and-row4'))
        self.assertEqual(row['missing_per_rank'], ['full_send', 'arrival', 'before_route_change_done', 'applied'])
        self.assertTrue(ret['no_observation_window_at_every_return'])
        # The summary's plain ratios exist because every validity condition holds.
        summary = arts['trunk-summary.json']
        self.assertTrue(summary['plain_w1']['accepted'], summary['plain_w1'])
        windows = [w['agg_tps'] for w in plain['windows']]
        self.assertEqual(summary['plain_w1']['median'], sorted(windows)[1])
        self.assertTrue(summary['native_w1_full']['instruments_off_and_sealed'])
        self.assertEqual([r['instruments_off_and_sealed'] for r in summary['rounds']], [True, True])
        self.assertTrue(summary['clean_speed_reading'])
        self.assertFalse(summary['target_is_gate'])
        record_case('tail-observer-off-sealed-w1-plain-on-full-rounds', calls=calls, rounds=[0, 1],
                    split_entries=session.split_entries, plain_requests=len(leaf.completions), real_sealed_w1=True,
                    real_client=True, returns=len(ret['rows']))

    def test_observer_mapped_on_one_rank_withholds_w1_plain_and_w2(self):
        done, pv, calls, arts, leaf, session = self.run_tail(mapped_rank=2)
        self.assertEqual(pv['instruments-w1'], 'FAIL')
        self.assertTrue(arts['instruments-w1.json']['ranks']['2']['observer_mapped'])
        self.assertEqual((pv['w1_controls'], pv['w1_plain'], pv['overlap']), ('INCONCLUSIVE',) * 3)
        self.assertEqual((calls, session.chats, leaf.completions, leaf.tokenized), ([], 0, [], []))
        self.assertNotIn('instruments-end', pv)
        self.assertFalse(arts['trunk-summary.json']['plain_w1']['accepted'])
        record_case('tail-observer-mapped-withholds', calls=calls, instruments='FAIL', rounds=[])

    def test_unreadable_maps_are_inconclusive_and_withhold(self):
        done, pv, calls, arts, leaf, session = self.run_tail(maps=None)
        self.assertEqual(pv['instruments-w1'], 'INCONCLUSIVE')
        self.assertEqual((pv['w1_controls'], pv['w1_plain'], pv['overlap']), ('INCONCLUSIVE',) * 3)
        self.assertEqual((calls, session.chats, leaf.completions), ([], 0, []))
        record_case('tail-maps-unreadable-withholds', calls=calls, instruments='INCONCLUSIVE', rounds=[])

    def test_split_left_unrestorable_withholds_the_plain_w1(self):
        done, pv, calls, arts, leaf, session = self.run_tail(mode='stuck-split')
        w1 = arts['w1.json']
        self.assertEqual(w1['summary']['verdict'], 'INCONCLUSIVE')
        cleanup = w1['repeats'][1]['cleanup']
        self.assertEqual((cleanup['route_after'], cleanup['full_restored'], cleanup['resume']), ('split', False, None))
        self.assertEqual(pv['w1-plain-full-idle'], 'INCONCLUSIVE')
        self.assertEqual((pv['w1_plain'], pv['overlap']), ('INCONCLUSIVE', 'INCONCLUSIVE'))
        self.assertEqual((leaf.completions, leaf.tokenized), ([], []))  # no plain request reached the engine
        self.assertNotIn('w1-plain.json', arts)
        self.assertEqual(calls, [])
        record_case('tail-unrestored-split-withholds-plain', calls=calls, full_idle='INCONCLUSIVE', plain_requests=0)

    def test_foreign_request_at_split_entry_never_enters_split(self):
        done, pv, calls, arts, leaf, session = self.run_tail(mode='foreign-entry')
        w1 = arts['w1.json']
        seal = w1['repeats'][1]['seal']['entry']
        self.assertIn('entry-status:scheduler-requests', seal)
        self.assertTrue(any(r.endswith(':foreign-live-uid') for r in seal), seal)
        self.assertEqual(session.split_entries, 0)
        self.assertIn('native-w1-unsealed:unsealed-entry', w1['repeats'][1]['reasons'])
        self.assertTrue(w1['repeats'][1]['cleanup']['full_restored'])
        self.assertEqual((pv['w1_controls'], pv['ingress-w1']), ('INCONCLUSIVE', 'FAIL'))
        self.assertEqual(arts['ingress-w1.json']['measurement'], 'INVALID')
        self.assertEqual(pv['overlap'], 'INCONCLUSIVE')
        self.assertEqual(calls, [])
        record_case('tail-foreign-at-entry-no-split', split_entries=0, ingress='FAIL', rounds=[])

    def test_a_swapped_cohort_member_at_the_same_count_never_enters_split(self):
        # A count-only census sees 8; the identity seal sees a foreign request and a missing cohort member.
        done, pv, calls, arts, leaf, session = self.run_tail(mode='swap-entry')
        w1 = arts['w1.json']
        seal = w1['repeats'][1]['seal']['entry']
        self.assertNotIn('entry-status:scheduler-requests', seal)
        self.assertTrue(any(r.endswith(':foreign-live-uid') for r in seal), seal)
        self.assertTrue(any(r.endswith(':cohort-uids-not-exact') for r in seal), seal)
        self.assertEqual(session.split_entries, 0)
        self.assertIn('native-w1-unsealed:unsealed-entry', w1['repeats'][1]['reasons'])
        self.assertTrue(w1['repeats'][1]['cleanup']['full_restored'])
        self.assertEqual(pv['w1_controls'], 'INCONCLUSIVE')
        self.assertEqual(pv['overlap'], 'INCONCLUSIVE')
        self.assertEqual(calls, [])
        record_case('tail-swapped-member-no-split', split_entries=0, rounds=[])

    def test_hidden_arrival_after_the_entry_read_is_invalid_and_withholds_w2(self):
        done, pv, calls, arts, leaf, session = self.run_tail(mode='hidden-arrival')
        w1 = arts['w1.json']
        repeat = w1['repeats'][1]
        # The stated residual: invisible under PAUSED_ALL at the entry seal, caught afterwards.
        self.assertEqual((repeat['seal']['entry'], repeat['seal']['begin']), ([], []))
        self.assertTrue(any(r.endswith(':foreign-live-uid') for r in repeat['seal']['end']), repeat['seal'])
        self.assertFalse(repeat['valid'])
        self.assertEqual((pv['w1_controls'], pv['ingress-w1']), ('INCONCLUSIVE', 'FAIL'))
        self.assertIn('repeat1:generation-count 9 != 8', arts['ingress-w1.json']['invalid'])
        self.assertIn('STEP overlap-gates INCONCLUSIVE', done.stdout)
        self.assertEqual(calls, [])
        record_case('tail-hidden-arrival-invalid', seal_entry='PASS', seal_end='FAIL', ingress='FAIL', rounds=[])

    def test_a_foreign_request_inside_a_sealed_window_is_invalid_by_the_access_log_alone(self):
        done, pv, calls, arts, leaf, session = self.run_tail(mode='transient-foreign')
        self.assertEqual(arts['w1.json']['summary']['verdict'], 'COMPLETE')  # every barrier was clean
        self.assertEqual((pv['w1_controls'], pv['ingress-w1']), ('PASS', 'FAIL'))
        self.assertEqual(arts['ingress-w1.json']['measurement'], 'INVALID')
        self.assertIn('repeat1:generation-count 9 != 8', arts['ingress-w1.json']['invalid'])
        self.assertIn('STEP overlap-gates PASS', done.stdout)
        self.assertIn('STEP w2-ingress-precondition INCONCLUSIVE', done.stdout)
        self.assertEqual(calls, [])
        self.assertFalse(arts['trunk-summary.json']['native_w1_full']['instruments_off_and_sealed'])
        record_case('tail-transient-foreign-ingress-invalid', w1='COMPLETE', ingress='FAIL', rounds=[])

    def test_engine_not_idle_after_the_plain_w1_withholds_w2(self):
        done, pv, calls, arts, leaf, session = self.run_tail(stuck=True)
        self.assertEqual(calls, [])
        self.assertEqual(pv['w1-plain-full-idle'], 'INCONCLUSIVE')  # the stuck request is visible before the plain W1
        self.assertEqual((pv['w1_plain'], pv['overlap']), ('INCONCLUSIVE', 'INCONCLUSIVE'))
        self.assertEqual(leaf.completions, [])
        self.assertIn('STEP w2-idle-precondition INCONCLUSIVE', done.stdout)
        self.assertEqual([r['verdict'] for r in arts['native35-performance-grade.json']['rounds']], ['NOT_RUN', 'NOT_RUN'])
        record_case('tail-not-idle-withholds-w2', calls=calls, idle='INCONCLUSIVE', rounds=[])

    def test_unqualified_rule4_withholds_both_w1_clients_and_w2(self):
        done, pv, calls, arts, leaf, session = self.run_tail(dict(QUALIFIED, rule4_candidate='FAIL'))
        self.assertEqual(calls, [])
        self.assertEqual((leaf.completions, leaf.tokenized, session.chats), ([], [], 0))
        self.assertEqual((pv['w1_controls'], pv['w1_plain'], pv['overlap']), ('INCONCLUSIVE', 'INCONCLUSIVE', 'INCONCLUSIVE'))
        self.assertNotIn('instruments-w1', pv)
        self.assertEqual(arts['gates.json']['outcomes']['row4'], 'NOT_RUN')
        self.assertIn('trunk-summary.json', arts)
        record_case('tail-unqualified-withholds-all', calls=calls, rounds=[])


class RoundWrapper(unittest.TestCase):
    """The real W2 round CLI behind e3_native39_round: the declared row4 NOT_RUN passes the unchanged gate and
    nothing else does.  Gates as the queue's PY_GATES writes them."""

    def gates(self, **change):
        import e3_native39_round as round39
        base = {'schema': 1, 'outcomes': {'startup': 'PASS', 'row1': 'PASS', 'rule4_controls': 'PASS',
                                          'rule4_candidate': 'PASS', 'row4': 'NOT_RUN', 'w1_controls': 'PASS'},
                'no_retry': True, 'calibration_frozen': True, 'rule4_control_continuation': False,
                'rule4_continuation_sha256': None, 'row4_capacity_continuation': False,
                'row4_capacity_continuation_sha256': None, 'row4_capacity_matched': [],
                'row4_declared_not_run': True, 'row4_not_run_reason': round39.ROW4_NOT_RUN_REASON}
        outcomes = change.pop('outcomes', {})
        base.update(change)
        base['outcomes'].update(outcomes)
        return base

    def run_round(self, gates):
        import e3_native39_round as round39
        import e3_native35_performance as round35
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'gates.json').write_text(json.dumps(gates))
            for name in ('w', 'w1'):
                (d / (name + '.json')).write_text('{}')
            argv = ['--url', 'http://127.0.0.1:9', '--model', 'm', '--run-id', 'r', '--epoch', '7', '--round', '0',
                    '--workload', str(d / 'w.json'), '--w1', str(d / 'w1.json'), '--gates', str(d / 'gates.json'),
                    '--out', str(d / 'out.json')]
            saved = round35.gates_allow_run
            try:
                with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
                    rc = round39.main(argv)
            finally:
                round35.gates_allow_run = saved
            return rc, json.loads((d / 'out.json').read_text())

    def test_only_the_declared_not_run_view_passes_the_unchanged_gate(self):
        import e3_native39_round as round39
        rc, out = self.run_round(self.gates())
        self.assertEqual((rc, out['error']['error_message']), (2, 'reviewed-package-readiness-not-pass'))  # past the gate
        refused = {'undeclared': self.gates(row4_declared_not_run=False, row4_not_run_reason=None),
                   'reason-differs': self.gates(row4_not_run_reason='other'),
                   'row4-fail-declared': self.gates(outcomes={'row4': 'FAIL'}),
                   'with-capacity-continuation': self.gates(row4_capacity_continuation=True),
                   'w1-not-pass': self.gates(outcomes={'w1_controls': 'INCONCLUSIVE'})}
        for label, gates in refused.items():
            with self.subTest(label=label):
                self.assertFalse(round39.gates_allow_run(gates, '/nonexistent/gates.json'))
                rc, out = self.run_round(gates)
                self.assertEqual((rc, out['error']['error_message']), (2, 'same-queue-qualification-gates-not-pass'))
        self.assertTrue(round39.gates_allow_run(self.gates(row4_declared_not_run=False, row4_not_run_reason=None,
                                                           outcomes={'row4': 'PASS'}), '/nonexistent/gates.json'))
        record_case('round-wrapper-declared-not-run-only', accepted=['declared'], refused=sorted(refused))

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
        record_case('guard-declared-counts', guard=receipt['verdict'], tokens={k: receipt['observed'][k]['tokens'] for k in ('fresh', 'append')})

    def test_a_count_off_by_one_fails(self):
        pv, receipt, _, _ = self.run_guard(tokens=lambda prompt: model_count(prompt) + 1)
        self.assertEqual(pv['w2-length'], 'FAIL')
        self.assertEqual(receipt['failures'], ['fresh-count-differs-from-declaration', 'append-count-differs-from-declaration'])
        record_case('guard-count-differs', guard=receipt['verdict'])

    def test_a_count_in_the_refused_band_fails(self):
        manifest = json.loads((HERE / 'workloads/q18a-w2-overlap-native39.json').read_text())['length_declaration']
        shift = next(s for s in range(1, 176) if not length.band(manifest['fresh']['tokens'] + s)['plans'])
        pv, receipt, _, _ = self.run_guard(tokens=lambda prompt: model_count(prompt) + shift)
        self.assertEqual(pv['w2-length'], 'FAIL')
        self.assertIn('fresh-has-no-native-plan', receipt['failures'])
        record_case('guard-refused-band', guard=receipt['verdict'], shift=shift)

    def test_a_tampered_declaration_fails_before_any_request(self):
        data = json.loads((HERE / 'workloads/q18a-w2-overlap-native39.json').read_text())
        data['length_declaration']['fresh']['tokens'] += 1
        pv, receipt, _, leaf = self.run_guard(workload=data)
        self.assertEqual((pv['w2-length'], receipt['failures']), ('FAIL', ['manifest-length-declaration-differs-from-derivation']))
        self.assertEqual(leaf.tokenized, [])
        record_case('guard-tampered-declaration', guard=receipt['verdict'])

    def test_no_engine_is_inconclusive(self):
        pv, receipt, _, _ = self.run_guard(url='http://127.0.0.1:9')
        self.assertEqual((pv['w2-length'], receipt['rc']), ('INCONCLUSIVE', 2))
        record_case('guard-no-engine', guard=receipt['verdict'])


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
        # deploy/ in a git tree; a bundle carries the same bytes as run-flash-prod.sh.q9 and setcfg.reference.py.
        prod = next(p for p in (ROOT / 'deploy/run-flash-prod.sh', HERE / 'run-flash-prod.sh.q9') if p.is_file())
        self.assertEqual(dec.sha(prod), PROD_LAUNCHER_SHA)
        fixture.a.setcfg = str(next(p for p in (ROOT / 'deploy/setcfg.py', HERE / 'setcfg.reference.py') if p.is_file()))
        self.assertEqual(dec.sha(fixture.a.setcfg), dec.SETCFG_SHA)
        launch = fixture.root / 's5-launcher'
        launch.write_text(s5g._s5_launcher(e2g.launcher(prod.read_text())))
        self.assertEqual(dec.sha(launch), dec.S5_LAUNCHER_SHA)
        (fixture.task / 'prod/run-flash-prod.sh').write_bytes(prod.read_bytes())  # the found launcher
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
        record_case('launcher-loopback-host', trunk_hosts=hosts, p_c_effective_host='0.0.0.0', launcher_sha256=declare_sha())


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
            reads = []

            def read():
                out = q / ('s%d.json' % len(reads))  # the checks never overwrite a record
                reads.append(out)
                self.assertEqual(checks.main(['summary', '--q', str(q), '--out', str(out)]), 0)
                return json.loads(out.read_text())
            # The w1ab2 B0 client output alone, without the trunk's FULL, route, ingress and instrument receipts:
            # its numbers stay visible as raw samples and are never a denominator (Codex item 4).
            s = read()
            self.assertEqual((s['plain_w1']['accepted'], s['plain_w1']['median'], s['plain_w1']['window_agg_tps']),
                             (False, None, None))
            self.assertEqual(s['plain_w1']['raw_invalid_samples']['window_agg_tps'], [391.8, 384.0, 394.4])
            self.assertEqual(s['plain_w1']['invalid_reasons'], ['w1-plain-full-idle-missing', 'w1-plain-route-missing',
                                                                'ingress-plain-missing', 'instruments-w1-missing'])
            self.assertIsNone(s['native_w1_full_over_plain_w1'])
            for name in ('w1-plain-full-idle', 'w1-plain-route', 'ingress-plain', 'instruments-w1'):
                (q / (name + '.json')).write_text(json.dumps({'verdict': 'PASS'}))
            s = read()
            self.assertEqual((s['plain_w1']['accepted'], s['plain_w1']['invalid_reasons']), (True, []))
            self.assertEqual(s['plain_w1']['window_agg_tps'], [391.8, 384.0, 394.4])
            self.assertEqual(s['plain_w1']['median'], 391.8)
            self.assertIsNone(s['plain_w1']['raw_invalid_samples'])
            self.assertEqual((s['native_w1_full']['median'], s['native_w1_full']['denominator']), (227.049, 227.199))
            self.assertAlmostEqual(s['native_w1_full_over_plain_w1'], 227.199 / 391.8)
            self.assertEqual([r['verdict'] for r in s['rounds']], ['INCONCLUSIVE', 'NOT_RUN'])
            self.assertIsNone(s['rounds'][0]['decode_tokens_per_s_during_fresh_prefill'])  # P-c fresh had no native plan
            self.assertEqual(s['bind_receipt'], None)
            # Every receipt PASS but the client verdict, a window or a stream not valid: never a denominator.
            plain = EXTRACT['w1ab2_b0']['w1']
            for change, reason in ((dict(verdict='INCONCLUSIVE'), 'plain-verdict-INCONCLUSIVE'),
                                   (dict(errors=['reader: stream 3 ended']), 'plain-errors'),
                                   (dict(windows=plain['windows'][:2]), 'windows-2-of-3'),
                                   (dict(windows=[dict(plain['windows'][0], per_stream=plain['windows'][0]['per_stream'][:7])]
                                         + plain['windows'][1:]), 'window0-eight-live-streams')):
                (q / 'w1-plain.json').write_text(json.dumps(dict(plain, **change)))
                s = read()
                self.assertEqual((s['plain_w1']['accepted'], s['plain_w1']['median']), (False, None), reason)
                self.assertIn(reason, s['plain_w1']['invalid_reasons'])
                self.assertIsNone(s['native_w1_full_over_plain_w1'])
                self.assertTrue(all(r['decode_share_of_plain_w1_median'] is None for r in s['rounds']))

    def test_real_access_lines_parse_and_only_the_loopback_plain_run_is_the_harness_s_own(self):
        self.assertEqual(ACCESS['schema'], 'native39-box-access-extract/1')
        for source in ACCESS['sources'].values():
            self.assertRegex(source['sha256'], '^[0-9a-f]{64}$')
        self.assertEqual(ACCESS['access_re'], checks.ACCESS_RE.pattern)  # the recorded counts are this parser's
        b0, pc = ACCESS['w1ab2_b0'], ACCESS['p_c']
        self.assertEqual((b0['parsed_lines'], pc['parsed_lines']), (b0['shaped_lines'], pc['shaped_lines']))
        self.assertEqual((b0['parsed_lines'], pc['parsed_lines']), (10, 6174))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'b0').mkdir()
            q = self.write_q(d / 'b0', b0['lines'], ['x'])
            rows = checks.access_lines(q)
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(r['t'] is not None and r['addr'] == checks.LOOPBACK for r in rows))
            start, end = b0['wall_anchor_s'], b0['windows'][-1]['wall_t1'] + checks.INGRESS_LAG_S
            for expect, rc in ((b0['streams'], 0), (b0['streams'] + 1, 1)):
                out = d / ('plain-%d.json' % expect)
                self.assertEqual(checks.main(['ingress', '--q', str(q), '--label', 'plain', '--start', repr(start),
                                              '--end', repr(end), '--generation', str(expect), '--out', str(out)]), rc)
            plain = json.loads((d / 'plain-8.json').read_text())
            self.assertEqual((plain['measurement'], plain['spans'][0]['generation_lines']), ('VALID', 8))
            # The P-c log (--host 0.0.0.0): every path the engine logged during the queue is classified, and
            # its LAN scrapes and unauthenticated LAN requests make any span holding them INVALID.
            (d / 'pc').mkdir()
            q = self.write_q(d / 'pc', pc['first_of_kind'], ['x'])
            rows = checks.access_lines(q)
            self.assertEqual(len(rows), len(pc['kinds']))
            self.assertNotIn('other', {r['kind'] for r in rows})
            generation = sum(n for k, n in pc['kinds'].items() if k.split()[2] in checks.GENERATION_PATHS)
            self.assertEqual(generation, 48 + 45 + 10)
            whole = checks.span(rows, min(r['t'] for r in rows), max(r['t'] for r in rows), 3)
            self.assertEqual(whole['invalid'], ['non-loopback-client', 'unauthenticated-request', 'generation-not-200'])
            loop = [r for r in rows if r['addr'] == checks.LOOPBACK]
            only = checks.span(loop, min(r['t'] for r in loop), max(r['t'] for r in loop), 2)
            self.assertEqual((only['invalid'], only['unknown']), ([], []))  # the control path's 500 is not ingress
        record_case('box-access-lines', b0_parsed=10, p_c_parsed=6174, plain='VALID', p_c_lan='INVALID')

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
                                      'w2-idle-precondition', 'trunk-summary',
                                      # Codex queue review (1c334bc) items 1-3 and the lead's return timing
                                      'instruments-w1', 'ingress-w1', 'w1-plain-full-idle', 'w1-plain-route', 'ingress-plain',
                                      'w2-ingress-precondition', 'w2-idle-round-$_round', 'instruments-round-$_round',
                                      'ingress-round-$_round', 'instruments-end', 'returns'})

    def test_every_leaf_this_suite_names_is_declared_with_its_test(self):
        import re
        declared = {l['name']: l for l in json.loads((HERE / 'native39_fixture_leaves.json').read_text())['leaves']}
        named = set(re.findall(r'[Ff]ixture leaf (queue-[a-z0-9-]+)', Path(__file__).read_text()))
        self.assertLessEqual(named, set(declared))
        for name, leaf in declared.items():
            if not name.startswith('queue-'):
                continue
            self.assertTrue(leaf['replaces'] and leaf['equivalence'] and leaf['cannot_prove'], name)
            for test in re.findall(r'test_native39_trunk_queue\.(\w+)', leaf['test']):
                self.assertIn(test, globals(), name)

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
