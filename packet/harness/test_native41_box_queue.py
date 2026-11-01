"""Native41 queue: router, plain-W2 arms over the real control plane, grade, commands, derivation. CPU only.

Parents: 9838f54 (Codex's checkpoint d9899cb with the byte-identical launcher)
and native40 f6d1db9 (the frozen negative control: candidate harness over
native40's overlay and sources). On native40, Router fails at the missing
AUTO_ARM_ACTIONS in the session and Arm at the latch (native40 has no manual
lease, so prepare never sets manual_owned). Classes:

* Router (fails on the parent: release_auto_arm reached the box as 409
  native-action-not-exposed): the API route probe_control.attach_native_router
  forwards release_auto_arm and the read-only auto window to the real
  NativeOverlapSession only with FLASH_S5_AUTO_ARM=1; off, the same three
  actions get exactly that 409 and the lease stays latched.
* Arm (fails on the parent: the same 409 through the router): the queue's exact
  arm command line runs e3_native41_autoarm_box.main through NativeRuntimeClient
  and LoopbackControl over loopback HTTP into the real endpoint, session,
  control client, driver and policy of a native38 startup-bound stack. The
  stock arm completes and never releases; the auto arm sends only status,
  pause, release_auto_arm, resume and the auto window (no prepare, no
  prefill_begin, no split entry) and, because nothing in this stack links the
  HTTP prompt to the scheduler, is incomplete with auto-not-proven. A real
  automatic cycle through the same window is e3_native41_window_probe.py on the
  connected Core (container).
* Grade: an INVALID ingress fails, an unproven auto arm is INCONCLUSIVE, speed
  never gates.
* Commands: every python3 command of the native41 body parses with its own
  module's argparse.
* Derivation/Window: the derived texts are on disk and parse; the switch is in
  EXTRA_ENV, declaration and preflight; the window slot executes at real wall
  clock; the active cap is the declared derivation.

Real objects: the endpoint, NativeOverlapSession, NativeControlClient, the
driver, AutoArmPolicy, bootstrap_full_startup, NativeRuntimeClient,
LoopbackControl, the W2 stream readers, sampler and windows, the arm runner.
Fixture leaves: native41_fixture_leaves.json (box-queue entries).
"""
import argparse
import asyncio
import contextlib
import copy
import io
import json
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'tools'), str(ROOT / 'patches/s5_engine')]
import native_auto_arm as auto  # noqa: E402
import probe_control as pc  # noqa: E402
import e3_native41_autoarm_box as box  # noqa: E402
import native41_box_gate as gate  # noqa: E402
import native41_box_queue as queue  # noqa: E402
import test_e3_native_overlap as native  # noqa: E402
import test_native38_plain_bind as native38  # noqa: E402
import test_native39_trunk as prior  # noqa: E402

SLOT_RECEIPTS = []
KEY = 'native41-fixture-key'
DRAFT = 'harness/runs/q18a_native41_autoarm.draft.sh'
FORBIDDEN = {'prepare', 'prefill_begin', 'prefill_end', 'pause_enter_split', 'enter_split', 'leave_full',
             'arm_admission', 'identity', 'barrier'}


# --- leaves -----------------------------------------------------------------

class JSONResponse:
    """Leaf fastapi.responses.JSONResponse: the endpoint's content and status code."""

    def __init__(self, content, status_code=200):
        self.content, self.status_code = content, status_code


class APIRouter:
    """Leaf fastapi.APIRouter: records the one POST route the endpoint registers."""

    def __init__(self):
        self.routes = []

    def add_api_route(self, path, endpoint, methods):
        self.routes.append((path, endpoint, tuple(methods)))


class App:
    """Leaf FastAPI app: state.engine_client (the AsyncLLM), include_router, the health middleware."""

    def __init__(self, engine_client):
        self.state = SimpleNamespace(engine_client=engine_client)
        self.routers = []

    def include_router(self, router):
        self.routers.append(router)

    def middleware(self, kind):
        return lambda fn: fn


class Request:
    """Leaf starlette Request: loopback client host, the Authorization header and the body stream."""

    def __init__(self, app, body, auth):
        self.app, self._body = app, body
        self.client = SimpleNamespace(host='127.0.0.1')
        self.headers = {'authorization': auth}
        self.url = SimpleNamespace(path=pc.NATIVE_PATH)

    async def stream(self):
        yield self._body


@contextlib.contextmanager
def fastapi_leaf():
    saved = {name: sys.modules.get(name) for name in ('fastapi', 'fastapi.responses')}
    fastapi, responses = ModuleType('fastapi'), ModuleType('fastapi.responses')
    fastapi.APIRouter, fastapi.Request, responses.JSONResponse = APIRouter, object, JSONResponse
    sys.modules.update({'fastapi': fastapi, 'fastapi.responses': responses})
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def pinned_reads(scheduler):
    """Leaf: two pinned Scheduler reads the cold-stack scheduler lacks.

    policy (scheduler.py:199, SchedulingPolicy(FCFS) on the box): AutoArmPolicy._select reads its name.
    get_num_unfinished_requests (scheduler.py:2825): 0 under PAUSED_ALL, else waiting + skipped + running
    (no streaming input here); the driver status reports it as scheduler_requests.
    """
    scheduler.policy = SimpleNamespace(name='FCFS')

    def unfinished():
        state = getattr(scheduler, 'pause_state', None)
        if getattr(state, 'name', state) == 'PAUSED_ALL' or str(state) == 'PAUSED_ALL':
            return 0
        return len(scheduler.waiting) + len(getattr(scheduler, 'skipped_waiting', ())) + len(scheduler.running)
    scheduler.get_num_unfinished_requests = unfinished


def attach(engine_client, switch):
    """The real attach_native_router as the API server calls it (its env is the server's)."""
    env = {'FLASH_S5_OVERLAP': '1', 'VLLM_API_KEY': KEY, 'FLASH_S5_AUTO_ARM': switch}
    app = App(engine_client)
    with fastapi_leaf():
        state = pc.attach_native_router(SimpleNamespace(api_key=None), app, env=env)
    (path, endpoint, methods), = app.routers[0].routes
    assert (path, methods) == (pc.NATIVE_PATH, ('POST',))
    return app, endpoint, state


def tokens(prompt):
    """Leaf tokenizer: the W2 token model (one token per word plus one per digit)."""
    return len(prompt.split()) + sum(c.isdigit() for c in prompt)


class Server:
    """Loopback API server leaf.

    POST /v1/flash_s5_native_overlap hands the body, client host and
    Authorization header to the real endpoint coroutine on the engine loop
    and writes its JSONResponse (leaf for uvicorn/fastapi plumbing).
    POST /v1/completions is the serving leaf: the pinned streaming completion
    shape the W2 readers parse (choices[0].token_ids, prompt_token_ids on the
    first chunk, cumulative usage, finish_reason 'length' at max_tokens, [DONE]);
    one token per 4 ms, first token after PREFILL_S for a prompt over 256 tokens.
    """
    PREFILL_S = 0.3

    def __init__(self, loop, app, endpoint):
        self.loop, self.app, self.endpoint = loop, app, endpoint
        self.completions, self.session_actions, self.lock = [], [], threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                if self.path == pc.NATIVE_PATH:
                    request = Request(server.app, body, self.headers.get('Authorization', ''))
                    with server.lock:
                        server.session_actions.append(json.loads(body).get('action'))
                    reply = server.loop.run(server.endpoint(request), 30)
                    data = json.dumps(reply.content).encode()
                    self.send_response(reply.status_code)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                assert self.path == '/v1/completions', self.path
                self.complete(json.loads(body))

            def complete(self, data):
                n, budget = tokens(data['prompt']), data['max_tokens']
                with server.lock:
                    server.completions.append(dict(tokens=n, max_tokens=budget, cache_salt=data.get('cache_salt'),
                                                   auth=self.headers.get('Authorization') == 'Bearer ' + KEY,
                                                   stream=data.get('stream'), ids=data.get('return_token_ids')))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.close_connection = True
                if n > 256:
                    time.sleep(server.PREFILL_S)
                try:
                    for k in range(1, budget + 1):
                        choice = {'index': 0, 'text': 't', 'token_ids': [7],
                                  'finish_reason': 'length' if k == budget else None}
                        if k == 1:
                            choice['prompt_token_ids'] = [1] * n
                        event = {'choices': [choice], 'usage': {'prompt_tokens': n, 'completion_tokens': k,
                                                                'total_tokens': n + k}}
                        self.wfile.write(b'data: ' + json.dumps(event).encode() + b'\n\n')
                        self.wfile.flush()
                        time.sleep(0.004)
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.http.daemon_threads = True
        self.url = 'http://127.0.0.1:%d' % self.http.server_address[1]

    def __enter__(self):
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.http.shutdown()
        self.http.server_close()


class EngineLoop:
    """Leaf: the API server event loop that also runs the engine ticks (as test_native39_trunk.ThreadControl)."""

    def __init__(self, core):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

        async def start():
            return native.Pump(core).start()
        self.pump = self.run(start(), 5)

    def run(self, coroutine, timeout_s):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout_s)

    def close(self):
        self.run(self.pump.stop(), 5)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()
        if self.pump.failure:
            raise self.pump.failure


def body_commands(text):
    """Every python3 command line of the native41 body, continuations joined, redirections dropped."""
    body = text[text.index(queue.BODY.splitlines()[0]):]
    body = body[:body.index("record native41-autoarm")]
    joined = re.sub(r'\\\n\s*', ' ', body)
    rows = []
    for line in joined.splitlines():
        match = re.search(r'python3 "\$H/([a-z0-9_]+)\.py"(.*)$', line)
        if match:
            args = re.split(r'\s+>\s*"', match.group(2))[0]
            rows.append((match.group(1), args.strip()))
    return rows


def expand(args, values):
    """Shell words with the queue variables substituted (leaf for bash expansion)."""
    for name, value in values.items():
        args = args.replace('"${%s[@]}"' % name, value).replace('${%s}' % name, value).replace('$' + name, value)
    args = args.replace('${_cid:-none}', 'none')
    return shlex.split(args)


QUEUE_VALUES = {'PROBE_URL': 'http://127.0.0.1:9', 'MODEL': 'fixture-model', 'RUN_ID': 'native41-fixture',
                'EPOCH': '7', '_kind': 'stock', '_pair': '0', 'H': '/h', 'Q': '/q', '_label': 'stock-0',
                'IDLE_TIMEOUT_S': '120', '_s': '1.5', '_e': '2.5',
                'GRADE_ARGS': '--arm /q/arm-stock-0.json --ingress stock-0=/q/ingress-stock-0.json'}


def arm_argv(**values):
    """The arm command exactly as the derived draft writes it."""
    draft = queue.derive()[DRAFT]
    (args,) = [a for m, a in body_commands(draft) if m == 'e3_native41_autoarm_box' and a.startswith('arm ')]
    values = dict(QUEUE_VALUES, **values)
    values['_label'] = '%s-%s' % (values['_kind'], values['_pair'])
    return expand(args, values)


def workload(tmp):
    """Test-scale W2 manifest: the frozen q18a W2 file with only sizes and times reduced (leaf)."""
    data = json.loads((HERE / 'workloads/q18a-w2-overlap-native39.json').read_text())
    data['decode'].update(prompt_tokens=64, generation_budget=3000, open_timeout_s=10.0)
    data['prefill']['fresh']['prompt_tokens'] = 600
    data['prefill']['append']['prompt_tokens'] = 64
    data['timeouts'] = dict(controller_s=10.0, read_s=10.0, drain_s=5.0, join_s=5.0, finish_s=10.0)
    data['window_s'] = 0.5
    path = Path(tmp) / 'w2-fixture.json'
    path.write_text(json.dumps(data))
    return path


# --- router -----------------------------------------------------------------

class Router(prior.Session):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        pinned_reads(self.scheduler)
        await self.call('prepare', target_route='full', phase='full-prefill-solo')
        # The installed policy object (auto.install adds only the bound-startup
        # check and the attribute); the manual prepare above latched the lease.
        self.policy = auto.AutoArmPolicy(self.core, self.driver)
        self.driver.auto_arm = self.policy
        self.policy.manual_owned = True

    async def post(self, endpoint, app, action):
        body = json.dumps(dict(schema=1, action=action, run_id='native41-router')).encode()
        reply = await endpoint(Request(app, body, 'Bearer ' + KEY))
        return reply.status_code, reply.content

    async def test_switch_on_forwards_release_and_the_auto_window_to_the_real_session(self):
        import native_overlap_session as ns
        self.assertEqual(getattr(ns, 'AUTO_ARM_ACTIONS', None), box_actions())
        with mock.patch.dict(os.environ, {'FLASH_S5_AUTO_ARM': '1'}):
            app, endpoint, _ = attach(self.frontend, '1')
            code, reply = await self.post(endpoint, app, 'auto_window_begin')
            self.assertEqual((code, reply), (409, {'error': 'auto-window-requires-released-idle-full'}))
            self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
            code, reply = await self.post(endpoint, app, 'release_auto_arm')
            self.assertEqual(code, 200, reply)
            self.assertIs(reply['released'], True)
            self.assertIs(self.policy.manual_owned, False)
            self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)
            self.assertIs(self.policy.manual_owned, False)  # resume never re-takes the lease
            code, begin = await self.post(endpoint, app, 'auto_window_begin')
            self.assertEqual(code, 200, begin)
            self.assertEqual((begin['phase'], begin['route']), ('auto-prefill', 'full'))
            self.assertIs(begin['snapshot']['auto_arm']['manual_owned'], False)
            self.assertEqual(await self.post(endpoint, app, 'auto_window_begin'),
                             (409, {'error': 'auto-window-already-open'}))
            code, end = await self.post(endpoint, app, 'auto_window_end')
            self.assertEqual(code, 200, end)
            self.assertEqual(end['auto_reasons'], ['auto-window-journaled-no-native-work',
                                                   'auto-window-not-exactly-one-closed-cycle',
                                                   'auto-window-armed-status-missing'])
            self.assertIs(end['complete'], False)
            self.assertIsNone(end['raw_journal_evidence'])
            self.assertEqual(end['route'], 'full')
            self.assertEqual(await self.post(endpoint, app, 'auto_window_end'),
                             (409, {'error': 'auto-window-end-without-begin'}))
            self.assertIs(self.policy.manual_owned, False)  # window reads never take the lease
            self.assertEqual(self.policy.selection_seq, 0)

    async def test_switch_off_refuses_the_three_actions_exactly_and_keeps_the_lease(self):
        for router, process, error in (('0', '0', 'native-action-not-exposed'),
                                       ('1', '0', 'session-action-not-exposed')):
            with self.subTest(router=router, process=process), \
                    mock.patch.dict(os.environ, {'FLASH_S5_AUTO_ARM': process}):
                app, endpoint, _ = attach(self.frontend, router)
                await self.post(endpoint, app, 'pause')
                for action in box_actions():
                    self.assertEqual(await self.post(endpoint, app, action), (409, {'error': error}))
                await self.post(endpoint, app, 'resume')
                self.assertIs(self.policy.manual_owned, True)


def box_actions():
    """The three actions the native41 arm sends besides status/pause/resume."""
    return ('release_auto_arm', 'auto_window_begin', 'auto_window_end')


# --- arms over the real control plane -----------------------------------------

class Arm(unittest.TestCase):
    def setUp(self):
        # Leaf (as test_native39_trunk.StartupStep): vllm.logger for the startup receipt line.
        self._logger = sys.modules.get('vllm.logger')
        sys.modules['vllm.logger'] = SimpleNamespace(init_logger=logging.getLogger)
        self.stack, self.core = native38.OwnerStartup().stack()
        pinned_reads(self.stack.scheduler)
        self.ready = native38.bootstrap_full_startup(self.core, self.stack.driver, timeout_s=2)
        self.driver = self.stack.driver
        self.policy = auto.install(self.core, self.driver, {auto.ENV: '1'})
        self.env = mock.patch.dict(os.environ, {'FLASH_S5_AUTO_ARM': '1', 'VLLM_API_KEY': KEY})
        self.env.start()
        self.loop = EngineLoop(self.core)
        app, endpoint, _ = attach(native.Frontend(self.core), '1')
        self.server = Server(self.loop, app, endpoint).__enter__()
        self.tmp = tempfile.TemporaryDirectory()
        self.workload = workload(self.tmp.name)

    def tearDown(self):
        self.server.__exit__()
        self.loop.close()
        self.env.stop()
        self.tmp.cleanup()
        if self._logger is None:
            sys.modules.pop('vllm.logger', None)
        else:
            sys.modules['vllm.logger'] = self._logger

    def latch(self):
        """The queue's typed full-idle entry: prepare FULL then resume (takes the manual lease)."""
        from e3_native_runtime_client import NativeRuntimeClient
        client = NativeRuntimeClient(self.server.url, 'native41-fixture', epoch=self.driver.engine_epoch, timeout_s=10)
        client.prepare('full', 'full-prefill-solo')
        client.resume()
        self.assertIs(self.policy.manual_owned, True)

    def run_arm(self, kind, index=0, package=True):
        out = Path(self.tmp.name) / ('arm-%s-%d.json' % (kind, index))
        argv = arm_argv(PROBE_URL=self.server.url, EPOCH=str(self.driver.engine_epoch), _kind=kind,
                        _pair=str(index), Q=self.tmp.name, H=str(HERE))
        argv[argv.index('--workload') + 1] = str(self.workload)
        mark = len(self.server.session_actions)
        with mock.patch.object(box.shared, '_package_gate', return_value={'ok': package}), \
                mock.patch.object(box, 'SETTLE_S', 0.2), mock.patch.object(box, 'CLOSE_TIMEOUT_S', 1.0), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = box.main(argv)
        return rc, json.loads(out.read_text()), self.server.session_actions[mark:]

    def test_plain_w2_stock_then_auto_arm_through_the_exact_cli(self):
        self.latch()
        rc, stock, actions = self.run_arm('stock')
        self.assertEqual(rc, 0, (stock.get('reasons'), stock.get('error')))
        self.assertIs(stock['complete'], True)
        self.assertEqual(set(actions), {'status'})
        self.assertIs(self.policy.manual_owned, True)
        self.assertEqual(stock['generation_requests_expected'], len(self.server.completions))
        standing, fresh, append = (self.server.completions[:8], self.server.completions[8],
                                   self.server.completions[9])
        self.assertTrue(all(c['stream'] and c['ids'] and c['auth'] for c in self.server.completions))
        self.assertEqual(len({c['cache_salt'] for c in standing}), 8)
        self.assertEqual(fresh['cache_salt'], append['cache_salt'])
        self.assertNotIn(fresh['cache_salt'], {c['cache_salt'] for c in standing})
        self.assertEqual((fresh['max_tokens'], append['max_tokens']), (1, 1))
        self.assertEqual(stock['fresh']['prompt_tokens_observed'], fresh['tokens'])
        self.assertGreater(stock['client_basis']['prefill_tokens_per_s'], 0)
        metrics = box.arm_metrics(stock)
        self.assertGreater(metrics['decode_only_tokens_per_s'], 0)
        self.assertGreater(metrics['client']['decode_share'], 0)

        rc, auto_arm, actions = self.run_arm('auto')
        self.assertEqual(rc, 2)
        self.assertIs(auto_arm['complete'], False)
        self.assertNotIn('error', auto_arm, auto_arm.get('error'))
        self.assertFalse(FORBIDDEN & set(actions), actions)
        self.assertEqual(actions[:5], ['status', 'pause', 'release_auto_arm', 'resume', 'status'])
        self.assertEqual(actions.count('auto_window_begin'), 1)
        self.assertEqual(actions.count('auto_window_end'), 1)
        self.assertLess(actions.index('auto_window_begin'), actions.index('auto_window_end'))
        self.assertIs(auto_arm['release']['released'], True)
        self.assertIs(self.policy.manual_owned, False)
        self.assertEqual(self.policy.selection_seq, 0)
        self.assertIn('auto-cycle-not-closed-once', auto_arm['reasons'])
        self.assertIn('auto-not-proven:auto-window-not-exactly-one-closed-cycle', auto_arm['reasons'])
        self.assertIs(auto_arm['auto_proof']['complete'], False)
        self.assertEqual([s['label'] for s in auto_arm['statuses']],
                         ['entry', 'before-fresh', 'cycle-not-closed', 'before-append', 'after-append'])

        grade = box.grade([stock, auto_arm], {'stock-0': {'verdict': 'PASS'}, 'auto-0': {'verdict': 'PASS'}},
                          {'end-full-idle': {'verdict': 'PASS'}})
        self.assertEqual(grade['verdict'], 'INCONCLUSIVE')
        self.assertIn('auto-0:auto-engagement-unproven', grade['missing'])
        self.assertEqual(grade['auto_engaged'], [False])
        self.assertIsNotNone(grade['summary']['stock']['client_prefill_tokens_per_s'])

    def test_entries_that_are_not_the_declared_lease_state_are_refused(self):
        # Released lease at a stock entry: the startup prepare bound the session run, then an
        # explicit release (as an auto arm leaves it).
        self.latch()
        from e3_native_runtime_client import NativeRuntimeClient
        client = NativeRuntimeClient(self.server.url, 'native41-fixture', epoch=self.driver.engine_epoch, timeout_s=10)
        box.release(client, {})
        self.assertIs(self.policy.manual_owned, False)
        rc, record, actions = self.run_arm('stock')
        self.assertEqual((rc, record['reasons']), (2, ['arm-error']))
        self.assertIn('arm-entry-not-latched-idle-full', json.dumps(record['error']))
        self.assertEqual(actions, ['status'])
        self.assertEqual(self.server.completions, [])
        # Package readiness not PASS: refused before any engine or serving call.
        self.latch()
        rc, record, actions = self.run_arm('auto', package=False)
        self.assertEqual(rc, 2)
        self.assertIn('reviewed-package-readiness-not-pass', json.dumps(record['error']))
        self.assertEqual((actions, self.server.completions), ([], []))
        self.assertIs(self.policy.manual_owned, True)


# --- grade ------------------------------------------------------------------

def row(kind, index, complete=True, proof=True, prefill=1000.0, share=0.9):
    """A retained arm record reduced to the fields arm_metrics/grade read (shape of box.arm)."""
    r = {'kind': kind, 'index': index, 'complete': complete, 'reasons': [] if complete else ['x'],
         'decode_only': {'aggregate_tokens_per_s': 100.0},
         'client_basis': {'prefill_tokens_per_s': prefill, 'decode_window': {'aggregate_tokens_per_s': 100.0 * share}},
         'fresh': {'ttft_s': 2.0}, 'append': {'ttft_s': 0.1}}
    if kind == 'auto':
        r['auto_proof'] = {'complete': proof}
        r['engine_basis'] = {'prefill_tokens_per_s': prefill * 1.1, 'decode_window': {'aggregate_tokens_per_s': 95.0}}
        r['returns'] = {'final_commit_to_all_rank_full_ack_ms': 3.0}
    return r


class Grade(unittest.TestCase):
    PASS = {'verdict': 'PASS'}

    def arms(self, **auto_fields):
        return [row('stock', 0), row('auto', 0, **auto_fields), row('stock', 1), row('auto', 1, **auto_fields)]

    def test_engaged_arms_pass_with_speed_reported_never_gated(self):
        slow = [row('stock', 0), row('auto', 0, prefill=10.0, share=0.1), row('stock', 1),
                row('auto', 1, prefill=10.0, share=0.1)]
        result = box.grade(slow, {k: self.PASS for k in ('stock-0', 'auto-0', 'stock-1', 'auto-1')},
                           {'end-full-idle': self.PASS})
        self.assertEqual(result['verdict'], 'PASS')
        self.assertEqual(result['auto_engaged'], [True, True])
        self.assertAlmostEqual(result['summary']['auto_over_stock']['client_prefill'], 0.01)
        self.assertIs(result['speed_is_gate'], False)

    def test_invalid_ingress_or_failed_check_fails_and_unproven_auto_is_inconclusive(self):
        ingress = {k: self.PASS for k in ('stock-0', 'auto-0', 'stock-1', 'auto-1')}
        self.assertEqual(box.grade(self.arms(), dict(ingress, **{'auto-1': {'verdict': 'FAIL'}}), {})['failures'],
                         ['ingress-auto-1:INVALID'])
        self.assertEqual(box.grade(self.arms(), ingress, {'end-full-idle': {'verdict': 'FAIL'}})['verdict'], 'FAIL')
        unproven = box.grade(self.arms(proof=False), ingress, {})
        self.assertEqual(unproven['verdict'], 'INCONCLUSIVE')
        self.assertEqual(unproven['missing'], ['auto-0:auto-engagement-unproven', 'auto-1:auto-engagement-unproven'])
        self.assertEqual(box.grade([row('stock', 0)], {'stock-0': self.PASS}, {})['missing'],
                         ['declared-arm-pair-missing'])
        missing = box.grade(self.arms(), dict(ingress, **{'stock-1': None}), {'unarmed-stock-prefill': None})
        self.assertEqual(missing['missing'], ['ingress-stock-1:None', 'unarmed-stock-prefill:None'])


# --- commands, derivation, window ----------------------------------------------

class Parsed(Exception):
    pass


class Commands(unittest.TestCase):
    def test_every_body_command_parses_with_its_module(self):
        import e3_native39_trunk_checks as checks
        modules = {'e3_native41_autoarm_box': box, 'e3_native39_trunk_checks': checks}
        rows = body_commands(queue.derive()[DRAFT])
        self.assertEqual(sorted({m for m, _ in rows}), sorted(modules))
        actions = []
        original = argparse.ArgumentParser.parse_args

        def capture(parser, args=None, namespace=None):
            raise Parsed(original(parser, args, namespace))
        for module, args in rows:
            argv = expand(args, QUEUE_VALUES)
            with self.subTest(command=' '.join(argv[:1])), mock.patch.object(argparse.ArgumentParser,
                                                                          'parse_args', capture):
                with self.assertRaises(Parsed) as parsed, contextlib.redirect_stderr(io.StringIO()):
                    modules[module].main(argv)
                actions.append((module, argv[0], vars(parsed.exception.args[0])))
        self.assertEqual([a[1] for a in actions], ['instruments', 'full-idle', 'idle', 'arm', 'ingress',
                                                  'instruments', 'full-idle', 'grade'])
        arm = next(ns for m, a, ns in actions if a == 'arm')
        self.assertEqual((arm['kind'], arm['index'], arm['control_timeout_s']), ('stock', 0, 60.0))
        self.assertTrue(arm['workload'].endswith('workloads/q18a-w2-overlap-native39.json'))
        ingress = next(ns for m, a, ns in actions if a == 'ingress')
        self.assertEqual(ingress['generation'], box.ARM_GENERATIONS)
        grade = actions[-1][2]
        self.assertEqual(sorted(c.split('=')[0] for c in grade['check']),
                         ['end-full-idle', 'instruments-end', 'instruments-start', 'unarmed-stock-prefill'])


class Derivation(unittest.TestCase):
    def test_derived_texts_are_on_disk_parse_and_carry_the_switch(self):
        derived = queue.derive() | queue.derive_tools()
        for name, text in derived.items():
            with self.subTest(name=name):
                self.assertEqual((ROOT / name).read_text(), text)
        for name in queue.derive():
            self.assertEqual(subprocess.run(['bash', '-n', str(ROOT / name)], capture_output=True).returncode, 0)
        settings = derived['harness/runs/q18a_native41_autoarm_settings.sh']
        extra = [line for line in settings.splitlines() if line.strip().startswith('"EXTRA_ENV=')]
        self.assertEqual(len(extra), 1)
        self.assertEqual(extra[0].split().count('FLASH_S5_AUTO_ARM=1'), 1)
        self.assertIn('FLASH_S5_REPLAY_OBSERVER=0', extra[0])
        self.assertIn('FLASH_S5_ENGINE_OFFERS=0', extra[0])
        draft = derived[DRAFT]
        self.assertIn("assert env.get('FLASH_S5_AUTO_ARM') == '1'", draft)
        self.assertIn("assert env.get('FLASH_S5_AUTO_ARM')=='1'", derived['harness/runs/q18a_native41_autoarm_declare.sh'])
        self.assertIn('exec timeout --signal=TERM --kill-after="$(native41_budget cleanup)s" '
                      '"$(native41_budget active)s"', draft)
        self.assertNotIn('native40_budget', draft)
        self.assertIn('from native41_box_gate import window_ok', draft)
        self.assertNotIn('native40_box_gate import window_ok', draft)
        # The grade's unarmed check is the inherited native40 step's own receipt.
        self.assertIn('--out "$Q/unarmed-stock-prefill.json"', draft)
        self.assertLess(draft.index('--out "$Q/unarmed-stock-prefill.json"'), draft.index(queue.BODY.splitlines()[0]))
        self.assertEqual(draft.count('e3_native41_autoarm_box.py" arm'), 1)
        self.assertNotIn('prefill_begin', queue.BODY)
        # The replaced trunk body never runs; its tools stay only in the inherited interface list.
        self.assertNotIn(queue.BODY_START, draft)
        self.assertNotIn('qualification_phase row1', draft)


class Window(unittest.TestCase):
    def test_one_phase_budget_caps_every_native41_step_and_keeps_native40_caps(self):
        import native40_box_gate as n40
        self.assertEqual(gate.ACTIVE_S, sum(gate.PHASES.values()))
        self.assertEqual(gate.CLEANUP_S, n40.CLEANUP_S)
        self.assertEqual((gate.NOT_BEFORE, gate.HARD_END), (n40.NOT_BEFORE, n40.HARD_END))
        for name in gate.KEPT:
            self.assertEqual(gate.PHASES[name], n40.PHASES[name], name)
        self.assertFalse(set(gate.DROPPED) & set(gate.PHASES))
        body = queue.BODY
        caps = {'instruments': ('timeout 120s python3 "$H/e3_native39_trunk_checks.py" instruments', 2, 2),
                'idle': ('timeout 150s python3 "$H/e3_native39_trunk_checks.py" idle', 1, 3),
                'full-idle': ('timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle', 2, 2),
                'arm': ('timeout --signal=TERM --kill-after=30s 1500s python3 "$H/e3_native41_autoarm_box.py" arm', 1, 4),
                'ingress': ('timeout 60s python3 "$H/e3_native39_trunk_checks.py" ingress', 1, 4),
                'grade': ('timeout 60s python3 "$H/e3_native41_autoarm_box.py" grade', 1, 1)}
        for name, (text, lines, _) in caps.items():
            self.assertEqual(body.count(text), lines, name)
        self.assertEqual(len(re.findall(r'^\s*python3 "\$H/', body, flags=re.M)), 0)  # every tool call is capped
        self.assertEqual((gate.PHASES['instruments'], gate.INSTRUMENTS_CAP_S), (2 * 120, 120))
        self.assertEqual(gate.PHASES['arm_entries'], 3 * gate.IDLE_CAP_S + gate.FULL_IDLE_CAP_S)
        self.assertEqual((gate.IDLE_CAP_S, gate.FULL_IDLE_CAP_S, gate.INGRESS_CAP_S, gate.GRADE_CAP_S),
                         (150, 450, 60, 60))
        self.assertEqual(gate.PHASES['w2_arms'], gate.ARMS * (gate.ARM_TIMEOUT_S + gate.ARM_KILL_AFTER_S))
        self.assertEqual((gate.ARM_TIMEOUT_S, gate.ARM_KILL_AFTER_S), (1500, 30))
        self.assertEqual(gate.PHASES['ingress'], gate.ARMS * gate.INGRESS_CAP_S)
        self.assertEqual(gate.PHASES['end_full_idle'], gate.FULL_IDLE_CAP_S)
        self.assertEqual(gate.latest_arm(), gate.HARD_END - gate.ACTIVE_S - gate.CLEANUP_S)
        self.assertTrue(gate.window_ok(gate.latest_arm()))
        self.assertFalse(gate.window_ok(gate.latest_arm() + 1))
        self.assertFalse(gate.window_ok(gate.NOT_BEFORE - 1))
        # The draft's budget function prints exactly these values.
        draft = queue.derive()[DRAFT]
        start = draft.index('native41_budget() {')
        func = draft[start:draft.index('\n}\n', start) + 3]
        for key, value in (('active', gate.ACTIVE_S), ('cleanup', gate.CLEANUP_S), ('health', gate.PHASES['health']),
                           ('end_identity', gate.PHASES['end_identity'])):
            done = subprocess.run(['bash', '-c', func + 'native41_budget ' + key],
                                  env=dict(os.environ, HARNESS_DIR=str(HERE)), capture_output=True, text=True)
            self.assertEqual((done.returncode, done.stdout.strip()), (0, str(value)), done.stderr)

    def test_draft_window_slot_executes_at_real_wall_clock(self):
        draft = queue.derive()[DRAFT]
        start = draft.index("python3 - <<'PY_SLOT'")
        slot = draft[start:draft.index('\nPY_SLOT\n', start) + len('\nPY_SLOT\n')]
        done = subprocess.run(['bash', '-c', 'fail_early() { return 2; }\n' + slot],
                              env=dict(os.environ, HARNESS_DIR=str(HERE)), capture_output=True, text=True)
        now = time.time()
        expected = 0 if gate.window_ok(now) else 2
        self.assertEqual(done.returncode, expected, done.stderr)
        SLOT_RECEIPTS.append(dict(draft='q18a_native41_autoarm.draft.sh', real_wall_clock=True,
                                  verdict='PASS' if done.returncode == 0 else 'OUTSIDE-WINDOW',
                                  active_s=gate.ACTIVE_S, cleanup_margin_s=gate.CLEANUP_S))


if __name__ == '__main__':
    unittest.main()
