"""Held-f queue: router, progress arms over the real control plane, receipts grade, commands, derivation. CPU only.

Parent: held-e 818a4a7 (bundle fc9fee4268dd). On held-e the API router never
forwards native_progress (409 native-action-not-exposed), so every held-f arm
stops at its mode switch: the Router class fails there, and Arm reproduces it
(test_held_e_router_stops_the_arm_at_the_mode_switch). Classes:

* Router: probe_control.attach_native_router forwards native_progress to the
  real NativeOverlapSession only with FLASH_S5_AUTO_ARM=1 and
  FLASH_S5_NATIVE_PROGRESS=burst; every other setting refuses it exactly as
  held-e did (409). The engine refuses it unless burst is installed and the
  engine is paused, idle and FULL. native_progress is a manual control: it
  holds the lease, so the arm sends it before release_auto_arm.
* Arm: the queue's exact arm command line, all four mode x cohort, through two
  NativeRuntimeClients over loopback HTTP into the real endpoint, session,
  control client, driver, AutoArmPolicy and NativeProgress of a native38
  startup-bound stack. Each arm sends status, pause, native_progress {mode},
  release_auto_arm, resume and nothing else; the engine mode is the arm's; the
  standing decoders carry the declared progress budget; the solo arm cancels
  them at the first observed native chunk with no reader alive; /metrics is
  read before and after the fresh prompt. Nothing here links the HTTP prompt
  to the scheduler, so no native chunk runs and each arm is incomplete by its
  own receipts (as held-e's auto arm here). The native chunk receipts on the
  connected Core are e3_native41_progress_probe.py.
* Receipts: the grade over lines written by the real ProgressTrace emitters
  (engine and four workers, the worker flush at each release begin) in the
  box's followed-log format: PASS for eight complete arms with the exact
  per-component decomposition, INCONCLUSIVE for each completeness defect,
  FAIL for an INVALID ingress.
* Commands / QueueArgparse: every python3 call of the held-f texts parses with
  its own module's real parser (real bash, the sourced held-f settings, the
  draft's budget function); the label split and span snippets run in bash.
* Derivation / Window: derived texts on disk and parse; both switches in
  EXTRA_ENV, declaration and preflight; the manifest; one phase budget; the
  window slot at real wall clock.

Real objects: the endpoint, NativeOverlapSession, NativeControlClient, the
driver, AutoArmPolicy, NativeProgress, bootstrap_full_startup,
NativeRuntimeClient, LoopbackControl, the W2 stream readers, the arm runner and
grade, nm.ProgressTrace. Fixture leaves: native41_progress_fixture_leaves.json
(queue entries) beside held-e's native41_fixture_leaves.json (box-queue).
"""
import argparse
import contextlib
import difflib
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
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'tools'), str(ROOT / 'patches/s5_engine')]
import native_auto_arm as auto  # noqa: E402
import native_measurement as nm  # noqa: E402
import probe_control as pc  # noqa: E402
import e3_native41_progress_box as box  # noqa: E402
import native41_progress_gate as gate  # noqa: E402
import native41_progress_queue as queue  # noqa: E402
import test_e3_native_overlap as native  # noqa: E402
import test_native38_plain_bind as native38  # noqa: E402
import test_native39_trunk as prior  # noqa: E402
import test_native41_box_queue as held  # noqa: E402

SLOT_RECEIPTS = []
ARGPARSE_RECEIPTS = []
KEY = held.KEY
DRAFT = 'harness/runs/q18a_native41_progress.draft.sh'
LABELS = [box.label(*x) for x in box.ARM_ORDER]
ARM_ACTIONS = ['status', 'pause', 'native_progress', 'release_auto_arm', 'resume']
BOTH = {'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_NATIVE_PROGRESS': 'burst'}


def attach(engine_client, auto_arm, progress):
    """The real attach_native_router as the API server calls it (its env is the server's)."""
    env = {'FLASH_S5_OVERLAP': '1', 'VLLM_API_KEY': KEY, 'FLASH_S5_AUTO_ARM': auto_arm}
    if progress is not None:
        env['FLASH_S5_NATIVE_PROGRESS'] = progress
    app = held.App(engine_client)
    with held.fastapi_leaf():
        state = pc.attach_native_router(SimpleNamespace(api_key=None), app, env=env)
    (path, endpoint, methods), = app.routers[0].routes
    assert (path, methods) == (pc.NATIVE_PATH, ('POST',))
    return app, endpoint, state


@contextlib.contextmanager
def process_env(values):
    """The engine/API process environment for one case (restored on exit)."""
    with mock.patch.dict(os.environ, values):
        for name in ('FLASH_S5_AUTO_ARM', 'FLASH_S5_NATIVE_PROGRESS'):
            if name not in values:
                os.environ.pop(name, None)
        yield


# --- router -----------------------------------------------------------------

class RouterBase(prior.Session):
    INSTALLED = 'burst'

    async def asyncSetUp(self):
        await super().asyncSetUp()
        held.pinned_reads(self.scheduler)
        await self.call('prepare', target_route='full', phase='full-prefill-solo')
        # The installed policy object (auto.install adds only the bound-startup
        # check and the attribute); the manual prepare above latched the lease.
        self.policy = auto.AutoArmPolicy(self.core, self.driver)
        self.driver.auto_arm = self.policy
        self.policy.manual_owned = True
        # The startup seam s5_startup.install_engine calls after the policy install.
        self.progress = self.driver.install_progress(self.INSTALLED)

    async def post(self, endpoint, app, action, **fields):
        body = json.dumps(dict(schema=1, action=action, run_id='held-f-router', **fields)).encode()
        reply = await endpoint(held.Request(app, body, 'Bearer ' + KEY))
        return reply.status_code, reply.content

    def unchanged(self):
        self.assertEqual((self.progress.mode, self.progress.counters['mode_changes']), ('current', 0))


class Router(RouterBase):
    async def test_both_switches_forward_native_progress_to_the_real_session(self):
        with process_env(BOTH):
            app, endpoint, _ = attach(self.frontend, '1', 'burst')
            # A running engine: the engine refuses (paused, idle FULL only). Its
            # DriverError is a ValueError, which this router maps to its generic 400.
            self.assertEqual(await self.post(endpoint, app, 'native_progress', mode='burst'),
                             (400, {'error': 'invalid-command-json'}))
            self.unchanged()
            self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
            self.assertEqual(await self.post(endpoint, app, 'native_progress', mode='fast'),
                             (409, {'error': 'native-control-progress-mode-invalid'}))
            self.unchanged()
            code, reply = await self.post(endpoint, app, 'native_progress', mode='burst')
            self.assertEqual(code, 200, reply)
            self.assertEqual({k: reply[k] for k in ('action', 'mode', 'previous', 'installed', 'route')},
                             dict(action='native_progress', mode='burst', previous='current', installed='burst',
                                  route='full'))
            self.assertEqual(reply['engine_epoch'], self.driver.engine_epoch)
            self.assertEqual((self.progress.mode, self.progress.counters['mode_changes']), ('burst', 1))
            code, status = await self.post(endpoint, app, 'status')
            self.assertEqual(code, 200, status)
            view = box.progress_view(status)
            self.assertEqual({k: view[k] for k in ('installed', 'mode', 'recording', 'mode_changes', 'chunks')},
                             dict(installed='burst', mode='burst', recording=False, mode_changes=1, chunks=0))
            # A manual control: the lease stays taken until the explicit release.
            self.assertIs(self.policy.manual_owned, True)
            code, reply = await self.post(endpoint, app, 'release_auto_arm')
            self.assertEqual((code, reply.get('released')), (200, True), reply)
            self.assertIs(self.policy.manual_owned, False)
            self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)
            # Sent after a release it takes the lease back: the arm sends it first.
            self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
            code, reply = await self.post(endpoint, app, 'native_progress', mode='current')
            self.assertEqual((code, reply.get('mode'), reply.get('previous')), (200, 'current', 'burst'))
            self.assertIs(self.policy.manual_owned, True)
            self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)
            self.assertEqual((self.progress.mode, self.progress.counters['mode_changes']), ('current', 2))

    async def test_every_other_setting_refuses_native_progress_exactly_as_held_e(self):
        cases = (('1', None, BOTH, 'native-action-not-exposed'),        # held-e's router
                 ('0', 'burst', BOTH, 'native-action-not-exposed'),
                 ('1', 'trace', BOTH, 'native-action-not-exposed'),
                 ('1', 'burst', {'FLASH_S5_AUTO_ARM': '1'}, 'session-action-not-exposed'),
                 ('1', 'burst', {'FLASH_S5_AUTO_ARM': '0', 'FLASH_S5_NATIVE_PROGRESS': 'burst'},
                  'session-action-not-exposed'))
        for router_auto, router_progress, process, error in cases:
            with self.subTest(router=(router_auto, router_progress), process=process), process_env(process):
                app, endpoint, _ = attach(self.frontend, router_auto, router_progress)
                self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
                for mode in ('burst', 'current'):
                    self.assertEqual(await self.post(endpoint, app, 'native_progress', mode=mode),
                                     (409, {'error': error}))
                self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)
                self.unchanged()
                self.assertIs(self.policy.manual_owned, True)


class RouterTraceInstalled(RouterBase):
    INSTALLED = 'trace'

    async def test_an_engine_without_burst_refuses_the_switch(self):
        with process_env(BOTH):
            app, endpoint, _ = attach(self.frontend, '1', 'burst')
            self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
            self.assertEqual(await self.post(endpoint, app, 'native_progress', mode='burst'),
                             (400, {'error': 'invalid-command-json'}))
            self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)
            code, status = await self.post(endpoint, app, 'status')
            self.assertEqual((box.progress_view(status)['installed'], box.progress_view(status)['mode']),
                             ('trace', 'current'))
            self.unchanged()


# --- arms over the real control plane -----------------------------------------

METRICS = '''# HELP vllm:num_preemptions_total Cumulative number of preemption from the engine.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="fixture-model"} %r
# HELP vllm:num_preemptions_created Cumulative number of preemption from the engine.
# TYPE vllm:num_preemptions_created gauge
vllm:num_preemptions_created{engine="0",model_name="fixture-model"} 1.7935341903309684e+09
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="fixture-model"} 0.42
'''
NATIVE_WORK = {'request_uid_sha256': 'f' * 64, 'request_generation': 1, 'transaction': 1,
               'chunk_start': 0, 'chunk_end': 1024}


class ProgressServer(held.Server):
    """held.Server (the declared serving and API leaves) plus two leaves.

    metrics-exposition: GET /metrics is the image's prometheus exposition of
    vllm:num_preemptions_total (with its _created sample) and
    vllm:kv_cache_usage_perc; the counter is ``preemptions``, raised by
    ``preempt_on_fresh`` when a fresh prompt arrives.
    native-work-identity: when a fresh prompt (a long prompt with a new cache
    salt) arrives, the next status reply carries the native_work_identity the
    real driver status reports while a native chunk is open; the prompt's
    prefill holds until that status was served.
    """

    def __init__(self, loop, app, endpoint):
        self.inner, self.replies = endpoint, []
        super().__init__(loop, app, self.endpoint_leaf)
        self.preemptions, self.preempt_on_fresh, self.metrics_reads = 0.0, 0.0, 0
        self.salts, self.native_open, self.native_seen = set(), threading.Event(), threading.Event()
        self.native_lock = threading.Lock()
        server, parent = self, self.http.RequestHandlerClass

        class Handler(parent):
            def do_GET(self):
                assert self.path == '/metrics', self.path
                with server.lock:
                    server.metrics_reads += 1
                    data = (METRICS % server.preemptions).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def complete(self, data):
                with server.lock:
                    fresh = held.tokens(data['prompt']) > 256 and data.get('cache_salt') not in server.salts
                    server.salts.add(data.get('cache_salt'))
                    if fresh:
                        server.preemptions += server.preempt_on_fresh
                if fresh:
                    server.native_seen.clear()
                    server.native_open.set()
                    server.native_seen.wait(5)
                super().complete(data)
        self.http.RequestHandlerClass = Handler

    async def endpoint_leaf(self, request):
        reply = await self.inner(request)
        self.replies.append((reply.status_code, reply.content))
        if type(reply.content) is dict and reply.content.get('action') == 'status':
            with self.native_lock:
                if self.native_open.is_set():
                    self.native_open.clear()
                    reply.content = dict(reply.content, native_work_identity=dict(NATIVE_WORK))
                    self.native_seen.set()
        return reply


def body_commands(text):
    """Every python3 command line of the held-f body, continuations joined, redirections dropped."""
    body = text[text.index(queue.BODY.splitlines()[0]):]
    body = body[:body.index('record native41-progress')]
    joined = re.sub(r'\\\n\s*', ' ', body)
    rows = []
    for line in joined.splitlines():
        match = re.search(r'python3 "\$H/([a-z0-9_]+)\.py"(.*)$', line)
        if match:
            args = re.split(r'\s+>\s*"', match.group(2))[0]
            rows.append((match.group(1), args.strip()))
    return rows


QUEUE_VALUES = dict(held.QUEUE_VALUES, RUN_ID='held-f-fixture', _mode='current', _cohort='solo', _pair='0',
                    _label='current-solo-0',
                    GRADE_ARGS='--arm /q/arm-current-solo-0.json --ingress current-solo-0=/q/ingress-current-solo-0.json')


def arm_argv(**values):
    """The arm command exactly as the derived draft writes it."""
    (args,) = [a for m, a in body_commands(queue.derive()[DRAFT]) if m == 'e3_native41_progress_box'
               and a.startswith('arm ')]
    values = dict(QUEUE_VALUES, **values)
    values['_label'] = '%s-%s-%s' % (values['_mode'], values['_cohort'], values['_pair'])
    return held.expand(args, values)


def workload(tmp):
    """Test-scale progress manifest: the frozen held-f file with only sizes and times reduced (leaf).

    The standing budget (3000) differs from decode.generation_budget (2000), so
    the serving leaf shows which one the arm sends.
    """
    data = json.loads((HERE / 'workloads/q18a-native41-progress.json').read_text())
    data['decode'].update(prompt_tokens=64, generation_budget=2000, open_timeout_s=10.0)
    data['progress']['standing_generation_budget'] = 3000
    data['prefill']['fresh']['prompt_tokens'] = 600
    data['prefill']['append']['prompt_tokens'] = 64
    data['timeouts'] = dict(controller_s=10.0, read_s=10.0, drain_s=5.0, join_s=5.0, finish_s=10.0)
    data['window_s'] = 0.5
    path = Path(tmp) / 'progress-fixture.json'
    path.write_text(json.dumps(data))
    return path


class Arm(unittest.TestCase):
    def setUp(self):
        # Leaf (as test_native39_trunk.StartupStep): vllm.logger for the startup receipt line.
        self._logger = sys.modules.get('vllm.logger')
        sys.modules['vllm.logger'] = SimpleNamespace(init_logger=logging.getLogger)
        self.stack, self.core = native38.OwnerStartup().stack()
        held.pinned_reads(self.stack.scheduler)
        # A fixture hang bound, not a gate: the first startup fence in a fresh
        # process imports cold on the shared hub (over 2 s seen).
        self.ready = native38.bootstrap_full_startup(self.core, self.stack.driver, timeout_s=10)
        self.driver = self.stack.driver
        # s5_startup.install_engine's order: the auto-arm policy, then progress.
        self.policy = auto.install(self.core, self.driver, {auto.ENV: '1'})
        self.progress = self.driver.install_progress('burst')
        self.env = mock.patch.dict(os.environ, dict(BOTH, VLLM_API_KEY=KEY))
        self.env.start()
        self.loop = held.EngineLoop(self.core)
        app, endpoint, _ = attach(native.Frontend(self.core), '1', 'burst')
        self.server = ProgressServer(self.loop, app, endpoint).__enter__()
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
        client = NativeRuntimeClient(self.server.url, 'held-f-fixture', epoch=self.driver.engine_epoch, timeout_s=10)
        client.prepare('full', 'full-prefill-solo')
        client.resume()
        self.assertIs(self.policy.manual_owned, True)

    def run_arm(self, mode, cohort, index=0):
        name = box.label(mode, cohort, index)
        out = Path(self.tmp.name) / ('arm-%s.json' % name)
        argv = arm_argv(PROBE_URL=self.server.url, EPOCH=str(self.driver.engine_epoch), _mode=mode, _cohort=cohort,
                        _pair=str(index), Q=self.tmp.name, H=str(HERE))
        argv[argv.index('--workload') + 1] = str(self.workload)
        mark, served = len(self.server.session_actions), len(self.server.completions)
        with mock.patch.object(box.shared, '_package_gate', return_value={'ok': True}), \
                mock.patch.object(box.base, 'SETTLE_S', 0.2), mock.patch.object(box.base, 'CLOSE_TIMEOUT_S', 1.0), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = box.main(argv)
        return (rc, json.loads(out.read_text()), self.server.session_actions[mark:],
                self.server.completions[served:])

    def test_mode_and_cohort_arms_through_the_exact_cli(self):
        records, previous = {}, 'current'
        for mode, cohort in (('current', 'solo'), ('burst', 'solo'), ('burst', 'live'), ('current', 'live')):
            with self.subTest(mode=mode, cohort=cohort):
                self.latch()
                rc, record, actions, served = self.run_arm(mode, cohort)
                records[record['label']] = record
                self.assertEqual(rc, 2)
                self.assertNotIn('error', record, record.get('error'))
                self.assertIs(record['complete'], False)
                self.assertEqual(record['label'], box.label(mode, cohort, 0))
                # Main client then the status poller: nothing but the five receipt actions.
                self.assertFalse(held.FORBIDDEN & set(actions), actions)
                self.assertEqual(actions[:5], ARM_ACTIONS)
                self.assertEqual(set(actions), set(ARM_ACTIONS))
                self.assertEqual([actions.count(a) for a in ARM_ACTIONS[1:]], [1, 1, 1, 1])
                self.assertEqual(record['control_history'], ARM_ACTIONS + ['status', 'status'])
                self.assertEqual(record['control_actions'], sorted(ARM_ACTIONS))
                # The engine took this arm's mode, then the policy got the lease.
                self.assertEqual({k: record['mode_switch'][k] for k in ('mode', 'previous', 'installed', 'route')},
                                 dict(mode=mode, previous=previous, installed='burst', route='full'))
                self.assertEqual(self.progress.mode, mode)
                self.assertIs(record['release']['released'], True)
                self.assertIs(self.policy.manual_owned, False)
                self.assertEqual([p['label'] for p in record['progress']],
                                 ['entry', 'before-fresh', 'after-fresh', 'before-append', 'after-append'])
                self.assertEqual([p['mode'] for p in record['progress']], [previous] + [mode] * 4)
                self.assertTrue(all(p['installed'] == 'burst' and p['recording'] is False
                                    for p in record['progress']))
                previous = mode
                # Eight standing decoders with the declared progress budget, then fresh and append.
                self.assertEqual(len(served), box.base.ARM_GENERATIONS)
                self.assertEqual([c['max_tokens'] for c in served], [3000] * 8 + [1, 1])
                self.assertEqual(record['standing_generation_budget'], 3000)
                self.assertEqual(served[8]['cache_salt'], served[9]['cache_salt'])
                self.assertNotIn(served[8]['cache_salt'], {c['cache_salt'] for c in served[:8]})
                # /metrics before and after the fresh prompt; no preemption.
                for tag in ('before-fresh', 'after-fresh'):
                    self.assertEqual(record['metrics'][tag], {'num_preemptions_total': 0.0,
                                                              'kv_cache_usage_perc': 0.42})
                reasons = set(record['reasons'])
                # No native chunk ran (nothing here links the prompt to the scheduler).
                self.assertIn('progress-chunks-not-recorded', reasons)
                self.assertTrue(any(r.startswith('auto-not-proven:') for r in reasons), reasons)
                self.assertEqual(record['progress_delta']['chunks'], 0)
                self.assertFalse(reasons & {'control-sequence-not-pause-native-progress-release-resume',
                                            'arm-sent-a-non-receipt-action', 'progress-status-missing',
                                            'progress-mode-not-this-arm', 'progress-lines-incomplete',
                                            'preemption-counter-unavailable', 'kv-preemption-during-the-fresh-prompt',
                                            'current-used-the-candidate', 'solo-cancel-unproved',
                                            'live-cohort-finished-inside-the-prefill', 'append-was-selected'},
                                 reasons)
                self.assertEqual('burst-sent-no-direct-command' in reasons, mode == 'burst')
                self.assertEqual('burst-live-sent-nothing-inside-a-wait' in reasons,
                                 (mode, cohort) == ('burst', 'live'))
                share = record['client_basis']['all_open_share_of_span']
                if cohort == 'solo':
                    # The status poller saw the native work and cancelled all eight; none survived.
                    cancel = record['solo_cancel']
                    self.assertIs(cancel['triggered'], True)
                    self.assertEqual(cancel['readers_alive'], [])
                    self.assertEqual((cancel['trigger']['chunk_start'], cancel['trigger']['chunk_end']), (0, 1024))
                    self.assertLess(cancel['started_ns'], cancel['ended_ns'])
                    self.assertLess(share, 1.0)
                    self.assertNotIn('decode_window', record['client_basis'])
                else:
                    self.assertIs(record['solo_cancel']['triggered'], False)
                    self.assertEqual(share, 1.0)
                    self.assertGreater(record['client_basis']['decode_window']['aggregate_tokens_per_s'], 0)
        self.assertEqual(len({r['salt_sha256'] for r in records.values()}), 4)
        self.assertEqual(self.progress.counters['mode_changes'], 2)
        self.assertEqual(self.server.metrics_reads, 8)

    def test_a_preemption_over_the_fresh_prompt_is_a_reason(self):
        self.latch()
        self.server.preempt_on_fresh = 1.0
        rc, record, _, _ = self.run_arm('burst', 'live', 1)
        self.assertEqual(rc, 2)
        self.assertIn('kv-preemption-during-the-fresh-prompt', record['reasons'])
        self.assertEqual([record['metrics'][t]['num_preemptions_total'] for t in ('before-fresh', 'after-fresh')],
                         [0.0, 1.0])

    def test_held_e_router_stops_the_arm_at_the_mode_switch(self):
        # The parent's API router (FLASH_S5_AUTO_ARM=1 only) in front of the same engine.
        app, self.server.inner, _ = attach(native.Frontend(self.core), '1', None)
        self.server.app = app
        self.latch()
        mark = len(self.server.replies)
        rc, record, actions, served = self.run_arm('burst', 'solo')
        self.assertEqual((rc, record['reasons']), (2, ['arm-error']))
        # LoopbackControl reports any 409 as a refusal; the route's own reply is the held-e one.
        self.assertEqual((record['error']['error_type'], record['error']['error_message']),
                         ('QualificationRefused', 'controller-refusal'))
        self.assertEqual(actions, ['status', 'pause', 'native_progress', 'resume'])
        self.assertEqual(self.server.replies[mark:][2], (409, {'error': 'native-action-not-exposed'}))
        self.assertEqual(len(served), 8)                  # the standing cohort only: no fresh prompt
        self.assertIs(self.policy.manual_owned, True)    # never released
        self.assertEqual((self.progress.mode, self.progress.counters['mode_changes']), ('current', 0))


# --- receipts: real ProgressTrace lines through the box log path ---------------

WALL0 = 1793534190.0
CHUNK_ROWS = 1000
CHUNKS = 3


class Emit:
    """The box's progress line producers over one synthetic run, with the real emitters.

    Engine: one nm.ProgressTrace('engine'), one flush per chunk after its
    release apply (NativeProgress.end), identity as NativeProgress.begin plus
    mode and open_rows. Workers: one nm.ProgressTrace('worker', rank) per rank,
    flushed at each flash_s5_release begin (executor_hooks.worker_rpc_begin),
    so a chunk's release row rides on the next flush and the run's last stays
    unflushed. Native commands reply on all four ranks, rank 3 last; stock
    RPCs reply on output rank 2. Lines leave in the box's followed-log format
    (leaf progress-log-lines): the docker --timestamps RFC 3339 prefix, the
    vLLM process prefix and record header, then the message; rank 3 on stderr.
    Every stamp is a whole microsecond, so each component is exact.
    """
    SEND, FIFO, HOST, HOL, MQ, APPLY, CADENCE = 10, 20, 500, 70, 50, 50, 50   # microseconds

    def __init__(self):
        self.engine = nm.ProgressTrace('engine')
        self.workers = {rank: nm.ProgressTrace('worker', rank) for rank in range(4)}
        self.now = self.mono0 = 5_000_000_000_000
        self.seq = self.txn = 0
        self.lines = []

    def wall(self, ns):
        return WALL0 + (ns - self.mono0) / 1e9

    def log(self, ns, who, text):
        whole, frac = divmod(ns - self.mono0, 10 ** 9)
        stamp = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(int(WALL0) + whole)) + '.%09dZ' % frac
        prefix = ('(EngineCore_DP0 pid=101) INFO 11-01 12:00:00 [engine_driver.py:940]' if who == 'engine'
                  else '(Worker_TP%d pid=%d) INFO 11-01 12:00:00 [executor_hooks.py:1230]' % (who, 201 + who))
        self.lines.append(['err' if who == 3 else 'log', '%s %s %s\n' % (stamp, prefix, text)])

    def tick(self, us):
        self.now += us * 1000
        return self.now

    def command(self, kind, segment, origin, direct):
        self.seq += 1
        submit = self.now
        sent = submit + self.SEND * 1000
        recv = {}
        method = 'flash_s5_release' if kind == 'release' else 'flash_s5_submit'
        for rank, trace in self.workers.items():
            enter = sent + self.FIFO * 1000
            if kind == 'release':
                for text in trace.lines():
                    self.log(enter, rank, text)
            done = enter + self.HOST * 1000
            emit = done + (self.HOL - (3 - rank)) * 1000
            recv[rank] = emit + self.MQ * 1000
            trace.record('r', self.seq, method, enter, done, emit, direct, 1)
        apply = recv[3] + self.APPLY * 1000
        self.engine.record('n', self.seq, kind, segment, self.seq if kind == 'segment' else None,
                           0 if kind == 'segment' else None, 0 if kind == 'segment' else None, origin, submit, sent,
                           recv[0], recv[1], recv[2], recv[3], apply, 'ok', direct)
        self.now = apply

    def stock(self, kind):
        self.seq += 1
        submit = self.now
        for rank, trace in self.workers.items():
            out = rank == 2
            trace.record('r', self.seq, kind, submit + 40_000, submit + 900_000, submit + 950_000 if out else None,
                         0, int(out))
        self.engine.record('s', self.seq, kind, submit, submit + 1_000_000, submit + 1_100_000)

    def chunk(self, start, end, *, mode, live, segments=3, waits=0):
        self.txn += 1
        burst = mode == 'burst'
        kinds = [('open', None)] + [('segment', ('prefix', 'layer', 'suffix')[i % 3]) for i in range(segments)]
        for i, (kind, segment) in enumerate(kinds + [('release', None)]):
            if i:
                self.tick(self.CADENCE)
            origin = 'wait' if kind == 'segment' and i <= waits else 'tick'
            self.command(kind, segment, origin, int(burst and kind == 'segment'))
            if i == 1 and live:
                # Two stock RPCs of the live decode batch, sampled inside the owner wait.
                self.stock('execute_model')
                self.stock('sample_tokens')
                if waits:
                    self.engine.record('w', self.seq, 'sample_tokens', self.now, self.now + 300_000, waits, 0)
        for text in self.engine.lines(txn=self.txn, request_generation=1, chunk_start=start, chunk_end=end,
                                      decode_live=live, mode=mode, open_rows=0):
            self.log(self.tick(100), 'engine', text)

    @staticmethod
    def commands(segments):
        return segments + 2                      # open, the segments, release


def synthetic_run(override=None, big=('burst-live-0', 1, 300)):
    """Eight arms in ARM_ORDER, CHUNKS native chunks of CHUNK_ROWS rows each, 5 s apart."""
    emit, arms, override = Emit(), [], override or {}
    for mode, cohort, index in box.ARM_ORDER:
        name = box.label(mode, cohort, index)
        spec = override.get(name, {})
        start = emit.tick(5_000_000)
        emit.tick(500_000)
        lives = spec.get('live', [8] * CHUNKS if cohort == 'live' else [8] + [0] * (CHUNKS - 1))
        for c, live in enumerate(lives):
            segments = big[2] if (name, c) == big[:2] else 3
            emit.chunk(c * CHUNK_ROWS, (c + 1) * CHUNK_ROWS, mode=spec.get('mode', mode), live=live,
                       segments=segments, waits=2 if (mode, cohort) == ('burst', 'live') else 0)
        end = emit.tick(1_000_000)
        arms.append(dict(label=name, mode=mode, cohort=cohort, index=index, complete=True, reasons=[],
                         wall_start_s=emit.wall(start), wall_end_s=emit.wall(end),
                         progress_delta={'chunks': len(lives)},
                         fresh={'prompt_tokens_observed': CHUNKS * CHUNK_ROWS + 1, 'ttft_s': 3.0},
                         decode_only={'aggregate_tokens_per_s': 400.0},
                         client_basis={'prefill_tokens_per_s': 1000.0,
                                       'decode_window': {'aggregate_tokens_per_s': 360.0}}))
    return emit, arms


PASS = {'verdict': 'PASS'}
CHECKS = {k: PASS for k in ('instruments-start', 'instruments-end', 'end-full-idle', 'unarmed-stock-prefill')}


class Receipts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def grade(self, emit, arms, ingress=None, lines=None):
        q = Path(tempfile.mkdtemp(dir=self.tmp.name))
        for stream in ('log', 'err'):
            (q / ('engine-stream.' + stream)).write_text(''.join(t for s, t in (lines or emit.lines) if s == stream))
        ingress = {a['label']: PASS for a in arms} if ingress is None else ingress
        return box.grade(arms, ingress, CHECKS, [str(q / 'engine-stream.log'), str(q / 'engine-stream.err')])

    def test_schema_and_line_path_are_the_emitters(self):
        self.assertEqual(box.ROWS, nm.PROGRESS_ROWS)
        self.assertEqual((box.PROGRESS_LINE, box.PROGRESS_SCHEMA), (nm.PROGRESS_LINE, nm.PROGRESS_SCHEMA))
        # docker logs splits entries at 16 KiB; every line plus its prefixes stays below.
        emit, _ = synthetic_run()
        self.assertTrue(all(len(t.encode()) < 16384 for _, t in emit.lines))
        self.assertTrue(all(len(t[t.index(nm.PROGRESS_LINE):].rstrip('\n')) < nm.PROGRESS_LINE_BYTES
                            for _, t in emit.lines))
        parts = [json.loads(t[t.index(nm.PROGRESS_LINE) + len(nm.PROGRESS_LINE):]).get('parts') for _, t in emit.lines]
        self.assertGreater(max(parts), 1)          # the multi-part path is exercised

    def test_eight_complete_arms_pass_with_the_exact_decomposition(self):
        emit, arms = synthetic_run()
        result = self.grade(emit, arms)
        self.assertEqual((result['verdict'], result['missing'], result['failures']), ('PASS', [], []))
        receipts = result['progress']
        self.assertEqual((receipts['engine_flushes'], receipts['dropped_rows'], receipts['duplicate_worker_rows'],
                          receipts['engine_flushes_outside_every_arm'], receipts['engine_flushes_in_two_arms'],
                          receipts['unstamped_engine_flushes']), (8 * CHUNKS, 0, 0, 0, 0, 0))
        for name in LABELS:
            row = receipts['arms'][name]
            self.assertEqual(row['reasons'], [], name)
            self.assertEqual(row['chunks_logged'], CHUNKS)
            live = row['counted_label'] == 'live'
            self.assertEqual(row['counted_chunks'], CHUNKS if live else CHUNKS - 1)
            self.assertEqual(row['missing_worker_rows'], 0)
            # Only the run's very last release reply row stays unflushed (critical rank 3).
            self.assertEqual(row['unflushed_final_release_rows'], int(name == LABELS[-1]))
            self.assertEqual(list(row['bands']), ['0-16384'])
        e = Emit
        for name, chunk in (('current-solo-0', 1), ('burst-live-0', 1), ('burst-live-1', 2)):
            info = receipts['arms'][name]['chunks'][chunk]
            n = e.commands(300 if (name, chunk) == ('burst-live-0', 1) else 3)
            expected = dict(cadence=e.CADENCE * (n - 1), send=e.SEND * n, fifo=e.FIFO * n, host=e.HOST * n,
                            hol=e.HOL * n, mq=e.MQ * n, apply=e.APPLY * n)
            with self.subTest(arm=name, chunk=chunk):
                self.assertEqual(info['commands'], n)
                for k, us in expected.items():
                    self.assertAlmostEqual(info['components_s'][k], us / 1e6, places=9, msg=k)
                self.assertAlmostEqual(info['wall_s'], sum(expected.values()) / 1e6, places=9)
                self.assertAlmostEqual(info['residual_s'], 0.0, places=9)
                self.assertEqual(info['missing_worker_rows'], [])
                self.assertEqual(info['direct_commands'], n - 2 if name.startswith('burst') else 0)
                live = name.split('-')[1] == 'live'
                self.assertEqual(info['stock_rpcs'], 2 if live else 0)
                if live:
                    self.assertEqual(info['stock_by_kind'], {'execute_model': 1, 'sample_tokens': 1})
                    self.assertAlmostEqual(info['stock_replying_host_s']['sample_tokens'], 0.00086, places=9)
                self.assertEqual((info['waits'], info['wait_sends']), (1, 2) if name.startswith('burst-live')
                                 else (0, 0))
                self.assertEqual(info['origins'].get('wait', 0), 2 if name.startswith('burst-live') else 0)
        # Speed is reported per mode and cohort, never a gate.
        self.assertIs(result['speed_is_gate'], False)
        self.assertIsNotNone(result['summary']['burst-live']['chunk_wall_s'])
        self.assertAlmostEqual(result['summary']['live_decode_share_client']['burst'], 0.9)
        self.assertEqual(result['declared_arms'], LABELS)

    def assertInconclusive(self, result, *items):
        self.assertEqual(result['verdict'], 'INCONCLUSIVE', result['missing'])
        for item in items:
            self.assertIn(item, result['missing'])

    def test_each_completeness_defect_is_inconclusive(self):
        emit, arms = synthetic_run()
        engine = [i for i, (_, t) in enumerate(emit.lines) if '"side":"engine"' in t]
        multi = [i for i in engine if '"parts":1,' not in emit.lines[i][1]]
        worker3 = [i for i, (s, t) in enumerate(emit.lines) if s == 'err']
        cases = []
        # One part of the multi-part chunk line lost.
        lines = [x for i, x in enumerate(emit.lines) if i != multi[1]]
        number = json.loads(emit.lines[multi[1]][1].split(nm.PROGRESS_LINE, 1)[1])['line']
        cases.append(('part-lost', lines, arms, ['progress:line-parts-incomplete:engine:None:%d' % number,
                                                 'progress-burst-live-0:progress-lines-differ-from-the-chunk-counter',
                                                 'progress-burst-live-0:native-chunks-not-row-0-through-n-minus-1']))
        # The critical rank's flush inside burst-solo-0 lost: its reply rows are missing.
        span = [a for a in arms if a['label'] == 'burst-solo-0'][0]
        inside = [i for i in worker3 if span['wall_start_s'] <= box.docker_stamp(emit.lines[i][1]) <= span['wall_end_s']]
        lines = [x for i, x in enumerate(emit.lines) if i != inside[1]]
        cases.append(('worker-flush-lost', lines, arms, ['progress-burst-solo-0:worker-rows-missing']))
        # A line cut as a log split would cut it.
        lines = [list(x) for x in emit.lines]
        lines[engine[4]][1] = lines[engine[4]][1][:300] + '\n'
        cases.append(('line-cut', lines, arms, ['progress:unparsable-progress-line']))
        # A line without the docker stamp.
        lines = [list(x) for x in emit.lines]
        lines[engine[5]][1] = lines[engine[5]][1].split(' ', 1)[1]
        cases.append(('unstamped', lines, arms, ['progress:engine-lines-outside-every-arm-span']))
        # The engine counter says one more chunk than the log holds.
        changed = [dict(a, progress_delta={'chunks': CHUNKS + 1}) if a['label'] == 'current-live-1' else a
                   for a in arms]
        cases.append(('counter', None, changed, ['progress-current-live-1:progress-lines-differ-from-the-chunk-counter']))
        # A declared arm without its record: its lines belong to no arm.
        cases.append(('arm-missing', None, arms[:-1], ['declared-arms-missing-or-repeated',
                                                       'progress:engine-lines-outside-every-arm-span']))
        for label, lines, rows, items in cases:
            with self.subTest(case=label):
                self.assertInconclusive(self.grade(emit, rows, lines=lines), *items)

    def test_each_wrong_arm_shape_is_inconclusive(self):
        cases = (
            ('mode', {'burst-solo-0': {'mode': 'current'}}, ['progress-burst-solo-0:progress-line-mode-not-this-arm']),
            ('live-transition', {'burst-live-1': {'live': [8, 7, 8]}},
             ['progress-burst-live-1:live-arm-chunk-without-all-decoders']),
            ('solo-relive', {'current-solo-0': {'live': [8, 0, 8]}},
             ['progress-current-solo-0:solo-arm-not-a-live-prefix-then-decoder-free-chunks']),
            ('solo-cold', {'current-solo-1': {'live': [0, 0, 0]}},
             ['progress-current-solo-1:solo-arm-not-a-live-prefix-then-decoder-free-chunks']),
        )
        for label, override, items in cases:
            with self.subTest(case=label):
                emit, arms = synthetic_run(override)
                result = self.grade(emit, arms)
                self.assertInconclusive(result, *items)
                self.assertEqual(result['missing'], items)
        # A solo arm with a transition chunk between its live head and the solo
        # chunks is the expected cancel shape, not a defect.
        emit, arms = synthetic_run({'burst-solo-1': {'live': [8, 3, 0]}})
        result = self.grade(emit, arms)
        self.assertEqual(result['verdict'], 'PASS', result['missing'])
        self.assertEqual(result['progress']['arms']['burst-solo-1']['counted_chunks'], 1)

    def test_dropped_rows_stray_lines_and_ingress(self):
        # A trace that hit its bound drops and counts: the flush reports it.
        emit = Emit()
        emit.seq = 10 ** 6                      # its own transport sequences: no worker-row collision
        emit.engine.limit = 3
        emit.chunk(0, CHUNK_ROWS, mode='current', live=8)
        emit.engine.limit = nm.PROGRESS_MAX_ROWS
        emit2, arms = synthetic_run()
        lines = emit.lines + emit2.lines
        self.assertInconclusive(self.grade(emit2, arms, lines=lines), 'progress:rows-dropped-or-duplicated',
                                'progress:engine-lines-outside-every-arm-span')
        # Native work between two arms is not declared traffic.
        emit, arms = synthetic_run()
        gap = arms[3]['wall_end_s'] + box.LOG_LAG_S + 0.5
        emit.now = emit.mono0 + int(round((gap - WALL0) * 1e9))
        emit.chunk(0, CHUNK_ROWS, mode='current', live=8)
        self.assertInconclusive(self.grade(emit, arms), 'progress:engine-lines-outside-every-arm-span')
        # An arm span that reaches into the next arm's lines.
        emit, arms = synthetic_run()
        arms = [dict(a, wall_end_s=arms[i + 1]['wall_end_s']) if i == 2 else a for i, a in enumerate(arms)]
        self.assertInconclusive(self.grade(emit, arms), 'progress:engine-line-in-two-arm-spans')
        # An INVALID ingress fails the run.
        emit, arms = synthetic_run()
        result = self.grade(emit, arms, ingress=dict({a['label']: PASS for a in arms}, **{'burst-live-0': {
            'verdict': 'FAIL'}}))
        self.assertEqual((result['verdict'], result['failures']), ('FAIL', ['ingress-burst-live-0:INVALID']))

    def test_grade_cli_reads_both_streams_and_never_overwrites(self):
        emit, arms = synthetic_run()
        q = Path(self.tmp.name)
        args = []
        for a in arms:
            (q / ('arm-%s.json' % a['label'])).write_text(json.dumps(a))
            (q / ('ingress-%s.json' % a['label'])).write_text(json.dumps(PASS))
            args += ['--arm', str(q / ('arm-%s.json' % a['label'])),
                     '--ingress', '%s=%s' % (a['label'], q / ('ingress-%s.json' % a['label']))]
        for name in CHECKS:
            (q / (name + '.json')).write_text(json.dumps(PASS))
            args += ['--check', '%s=%s' % (name, q / (name + '.json'))]
        for stream in ('log', 'err'):
            (q / ('engine-stream.' + stream)).write_text(''.join(t for s, t in emit.lines if s == stream))
        args += ['--log', str(q / 'engine-stream.log'), '--log', str(q / 'engine-stream.err'),
                 '--out', str(q / 'grade.json')]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(box.main(['grade'] + args), 0)
        self.assertEqual(out.getvalue().strip(), 'NATIVE41_PROGRESS_GRADE PASS')
        self.assertEqual(json.loads((q / 'grade.json').read_text())['verdict'], 'PASS')
        self.assertEqual(box.main(['grade'] + args), 2)                  # an existing output is never replaced
        # Without the stderr stream the critical rank's rows are missing.
        args[args.index(str(q / 'engine-stream.err')) - 1:args.index(str(q / 'engine-stream.err')) + 1] = []
        args[-1] = str(q / 'grade-stdout-only.json')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(box.main(['grade'] + args), 2)
        self.assertEqual(json.loads((q / 'grade-stdout-only.json').read_text())['missing'],
                         sorted('progress-%s:worker-rows-missing' % name for name in LABELS))


# --- commands, derivation, window ----------------------------------------------

class Commands(unittest.TestCase):
    def test_every_body_command_parses_with_its_module(self):
        import e3_native39_trunk_checks as checks
        modules = {'e3_native41_progress_box': box, 'e3_native39_trunk_checks': checks}
        rows = body_commands(queue.derive()[DRAFT])
        self.assertEqual(sorted({m for m, _ in rows}), sorted(modules))
        actions = []
        original = argparse.ArgumentParser.parse_args

        def capture(parser, args=None, namespace=None):
            raise held.Parsed(original(parser, args, namespace))
        for module, args in rows:
            argv = held.expand(args, QUEUE_VALUES)
            with self.subTest(command=' '.join(argv[:1])), mock.patch.object(argparse.ArgumentParser,
                                                                          'parse_args', capture):
                with self.assertRaises(held.Parsed) as parsed, contextlib.redirect_stderr(io.StringIO()):
                    modules[module].main(argv)
                actions.append((module, argv[0], vars(parsed.exception.args[0])))
        self.assertEqual([a[1] for a in actions], ['instruments', 'full-idle', 'arm', 'ingress', 'instruments',
                                                  'full-idle', 'grade'])
        arm = next(ns for m, a, ns in actions if a == 'arm')
        self.assertEqual((arm['mode'], arm['cohort'], arm['index'], arm['control_timeout_s']),
                         ('current', 'solo', 0, 60.0))
        self.assertTrue(arm['workload'].endswith('workloads/q18a-native41-progress.json'))
        self.assertTrue(arm['out'].endswith('/arm-current-solo-0.json'))
        ingress = next(ns for m, a, ns in actions if a == 'ingress')
        self.assertEqual((ingress['generation'], ingress['label']), (box.base.ARM_GENERATIONS, 'current-solo-0'))
        grade = actions[-1][2]
        self.assertEqual(sorted(c.split('=')[0] for c in grade['check']),
                         ['end-full-idle', 'instruments-end', 'instruments-start', 'unarmed-stock-prefill'])
        self.assertEqual(grade['log'], ['/q/engine-stream.log', '/q/engine-stream.err'])
        self.assertEqual(grade['out'], '/q/native41-progress-grade.json')

    def test_label_split_and_span_snippets_run_in_bash(self):
        draft = queue.derive()[DRAFT]
        (loop,) = re.findall(r'^\s*for _label in ([a-z0-9 -]+); do$', draft, flags=re.M)
        self.assertEqual(loop.split(), LABELS)
        self.assertEqual(list(queue.ARMS), LABELS)
        split = draft[draft.index('    IFS=- read -r _mode _cohort _pair <<< "$_label"'):].splitlines()[0]
        script = 'for _label in %s; do\n%s\nprintf "%%s %%s %%s\\n" "$_mode" "$_cohort" "$_pair"\ndone' % (loop, split)
        done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual([tuple(line.split()) for line in done.stdout.splitlines()],
                         [(m, c, str(i)) for m, c, i in box.ARM_ORDER])
        start = draft.index('    _span=$(python3 - "$Q/arm-$_label.json" <<\'PY_SPAN\'')
        snippet = draft[start:draft.index('\nPY_SPAN\n)\n', start) + len('\nPY_SPAN\n)\n')]
        with tempfile.TemporaryDirectory() as q:
            Path(q, 'arm-burst-live-0.json').write_text(json.dumps({'wall_start_s': 1793534190.123456,
                                                                    'wall_end_s': 1793534490.5}))
            Path(q, 'arm-burst-solo-0.json').write_text(json.dumps({'wall_start_s': None}))
            Path(q, 'arm-current-live-0.json').write_text('{not json')
            for name, expected in (('burst-live-0', '1793534190.123456 1793534490.5'), ('burst-solo-0', ''),
                                   ('current-live-0', ''), ('current-solo-1', '')):
                body = 'Q=%s\n_label=%s\n%s\nprintf "%%s" "$_span"' % (shlex.quote(q), name, snippet)
                done = subprocess.run(['bash', '-c', body], capture_output=True, text=True, timeout=30)
                self.assertEqual((done.returncode, done.stdout), (0, expected), (name, done.stderr))


ARGPARSE_VALUES = dict(held.ARGPARSE_VALUES, QNAME='q18a-native41-progress', RUN_ID='q18a-native41-progress-fixture',
                       _mode='current', _cohort='solo', _pair='0', _label='current-solo-0')


def bash_words(args, harness, draft, values=None):
    """Expand one call's argument text with the real bash (set -u), held-f settings sourced."""
    values = dict(ARGPARSE_VALUES, **(values or {}))
    budget = draft[draft.index('native41_budget() {'):]
    budget = budget[:budget.index('\n}\n') + 3]
    lines = ['set -u', 'fail_early() { :; }', 'fail() { :; }', 'H=%s' % shlex.quote(str(harness))]
    lines += ['%s=%s' % (k, shlex.quote(v)) for k, v in values.items()]
    lines += ['GRADE_ARGS=(--arm "$Q/arm-current-solo-0.json" --ingress "current-solo-0=$Q/ingress-current-solo-0.json")',
              'source "$H/runs/q18a_native41_progress_settings.sh" || exit 97', budget,
              "printf '%s\\0' " + args]
    done = subprocess.run(['bash', '-c', '\n'.join(lines)], capture_output=True, timeout=60)
    return done.returncode, done.stdout.decode().split('\0')[:-1], done.stderr.decode()[-400:]


class QueueArgparse(unittest.TestCase):
    """Every python3 call of the held-f queue texts that names a script runs
    here, expanded by the real bash (set -u, the sourced held-f settings, the
    draft's own budget function), against that script's real parser."""

    def test_every_queue_python_call_parses_with_its_real_parser(self):
        texts = queue.derive()
        draft = texts[DRAFT]
        calls = held.queue_python_calls(texts, (HERE / 'qlib.sh').read_text())
        scripts = sorted({c[1] for c in calls})
        self.assertEqual(scripts, sorted([
            'SETCFG', 'e2_memory_attribution.py', 'e3_native28_host_observer.py', 'e3_native30_e_grade.py',
            'e3_native39_trunk_checks.py', 'e3_native39_w2_length.py', 'e3_native41_progress_box.py',
            'e3_native41_unarmed_box.py', 'e3_native_speculation_gate.py', 'e3_native_startup_grade.py',
            'e3_private_auth.py', 'engine_identity.py', 'native35_ledger_grade.py']))
        setcfg = next(p for p in (ROOT / 'deploy/setcfg.py', HERE / 'setcfg.reference.py') if p.is_file())
        rows = []
        for name, script, args, form in calls:
            with self.subTest(file=name.split('/')[-1], script=script, args=args[:60]):
                rc, words, err = bash_words(args, HERE, draft)
                self.assertEqual(rc, 0, err)
                if form != 'python3':
                    words = held.QLIB_CALLS[form][1](words)
                if script == 'SETCFG':
                    verdict, detail = held.setcfg_parse(setcfg, words)
                else:
                    verdict, detail = held.real_parse(HERE / script, words)
                self.assertEqual(verdict, 'parsed', (script, words[:4], detail))
                rows.append(dict(file=name.split('/')[-1], script=script, form=form,
                                 action=words[0] if words and not words[0].startswith('-') else None,
                                 words=len(words), verdict=verdict))
        self.assertEqual(len(rows), len(calls))
        # Held-e's calls with its body's replaced by held-f's (7: no unarmed idle entry).
        parent = queue.held_e.derive()
        parent_calls = held.queue_python_calls(parent, (HERE / 'qlib.sh').read_text())
        mine = body_commands(draft)
        self.assertEqual(len(mine), 7)
        self.assertEqual(len(rows), len(parent_calls) - len(held.body_commands(parent[held.DRAFT])) + len(mine))
        ARGPARSE_RECEIPTS[:] = rows
        # The length guard reads the manifest the arms run.
        guard = [a for n, s, a, f in calls if s == 'e3_native39_w2_length.py']
        self.assertEqual(len(guard), 1)
        self.assertIn('--workload "$H/workloads/q18a-native41-progress.json"', guard[0])

        def first(script, action):
            return next(a for n, s, a, f in calls if s == script and a.split()[0] == action)
        arm, grade = first('e3_native41_progress_box.py', 'arm'), first('e3_native41_progress_box.py', 'grade')
        controls = (
            ('arm-unknown-option', arm, ('--control-timeout-s', '--control-timeout-sec')),
            ('arm-undeclared-mode', arm, ('--mode "$_mode"', '--mode fast')),
            ('arm-without-cohort', arm, ('--cohort "$_cohort"', '')),
            ('grade-without-out', grade, ('--out "$Q/native41-progress-grade.json"', '')),
        )
        for label, original, (old, new) in controls:
            with self.subTest(control=label):
                self.assertEqual(original.count(old), 1)
                rc, words, err = bash_words(original.replace(old, new), HERE, draft)
                self.assertEqual(rc, 0, err)
                verdict, detail = held.real_parse(HERE / 'e3_native41_progress_box.py', words)
                self.assertEqual(verdict, 'refused', (label, detail))
        rc, _, err = bash_words('--out "$NOT_DECLARED_ANYWHERE"', HERE, draft)
        self.assertNotEqual(rc, 0)
        self.assertIn('NOT_DECLARED_ANYWHERE', err)


class Derivation(unittest.TestCase):
    def test_derived_texts_are_on_disk_parse_and_carry_both_switches(self):
        derived = queue.derive() | queue.derive_tools()
        for name, text in derived.items():
            with self.subTest(name=name):
                self.assertEqual((ROOT / name).read_text(), text)
        for name in queue.derive():
            self.assertEqual(subprocess.run(['bash', '-n', str(ROOT / name)], capture_output=True).returncode, 0)
        settings = derived['harness/runs/q18a_native41_progress_settings.sh']
        extra = [line for line in settings.splitlines() if line.strip().startswith('"EXTRA_ENV=')]
        self.assertEqual(len(extra), 1)
        words = extra[0].split()
        self.assertEqual((words.count('FLASH_S5_AUTO_ARM=1'), words.count('FLASH_S5_NATIVE_PROGRESS=burst')), (1, 1))
        self.assertIn('FLASH_S5_REPLAY_OBSERVER=0', extra[0])
        self.assertIn('FLASH_S5_ENGINE_OFFERS=0', extra[0])
        # Against held-e's settings: that one token and its comment, nothing else.
        old = (ROOT / 'harness/runs/q18a_native41_autoarm_settings.sh').read_text()
        old = old.replace(queue.OLD, queue.NEW).replace(queue.OLD_Q, queue.NEW_Q)
        diff = [line for line in difflib.ndiff(old.splitlines(), settings.splitlines()) if line[:1] in '+-']
        self.assertEqual([line[2:].replace(' ' + queue.SWITCH, '') for line in diff if line[0] == '+'
                          and 'EXTRA_ENV' in line], [line[2:] for line in diff if line[0] == '-'])
        self.assertTrue(all(line.startswith('+ #') for line in diff if line[0] == '+' and 'EXTRA_ENV' not in line))
        draft = derived[DRAFT]
        declare = derived['harness/runs/q18a_native41_progress_declare.sh']
        self.assertIn("assert env.get('FLASH_S5_AUTO_ARM') == '1'", draft)
        self.assertIn("assert env.get('FLASH_S5_NATIVE_PROGRESS') == 'burst'", draft)
        self.assertIn("assert env.get('FLASH_S5_AUTO_ARM')=='1'", declare)
        self.assertIn("assert env.get('FLASH_S5_NATIVE_PROGRESS')=='burst'", declare)
        self.assertIn('from native41_progress_gate import window_ok', draft)
        self.assertIn('from native41_progress_gate import ACTIVE_S,CLEANUP_S,PHASES', draft)
        self.assertNotIn('native41_box_gate import window_ok', draft)
        self.assertNotIn('native40_budget', draft)
        self.assertEqual(draft.count('e3_native41_progress_box.py" arm'), 1)
        self.assertNotIn('e3_native41_autoarm_box.py" arm', draft)
        self.assertNotIn(queue.held_e.BODY, draft)
        self.assertIn(queue.GUARD_NEW, draft)
        self.assertNotIn(queue.GUARD_OLD, draft)
        self.assertEqual(draft.count('--workload "$H/workloads/q18a-native41-progress.json"'), 2)
        self.assertIn('  e3_native41_progress_box.py native41_progress_gate.py; do', draft)
        self.assertIn('-r $H/workloads/q18a-native41-progress.json &&', draft)
        self.assertIn("perf=p.parent/'native41-progress-grade.json'", draft)
        self.assertNotIn('native41-autoarm-grade', draft)
        self.assertIn('--out "$Q/unarmed-stock-prefill.json"', draft)
        self.assertLess(draft.index('--out "$Q/unarmed-stock-prefill.json"'), draft.index(queue.BODY.splitlines()[0]))
        self.assertNotIn(queue.held_e.BODY_START, draft)
        self.assertNotIn('prefill_begin', queue.BODY)

    def test_workload_is_the_declared_sub_threshold_progress_manifest(self):
        import e3_native39_w2_length as length
        data = json.loads((HERE / 'workloads/q18a-native41-progress.json').read_text())
        w2 = json.loads((HERE / 'workloads/q18a-w2-overlap-native39.json').read_text())
        problems, cfg = box.shared.validate_manifest(data)
        self.assertEqual((problems, cfg['repeats']), ([], 1))
        self.assertEqual(box.standing_budget(data), box.MAX_STANDING_BUDGET)
        self.assertEqual(data['decode']['generation_budget'], w2['decode']['generation_budget'])
        self.assertEqual(data['length_declaration'], length.declaration(data))
        # Only the fresh prompt size, the declared progress block and its length declaration differ from W2.
        own = ('workload', 'notes', 'progress', 'length_declaration', 'prefill')
        self.assertEqual(data['workload'], 'q18a-native41-progress')
        self.assertEqual({k: v for k, v in data.items() if k not in own},
                         {k: v for k, v in w2.items() if k not in own})
        self.assertEqual(data['prefill']['append'], w2['prefill']['append'])
        self.assertEqual({k: v for k, v in data['prefill']['fresh'].items() if k != 'prompt_tokens'},
                         {k: v for k, v in w2['prefill']['fresh'].items() if k != 'prompt_tokens'})
        *_, fresh, append = length.counts(data['prefill']['fresh'], data['prefill']['append'])
        *_, _, w2_append = length.counts(w2['prefill']['fresh'], w2['prefill']['append'])
        self.assertEqual((fresh, append), (65445, 69436))         # the counts the guard checks on /tokenize
        self.assertLess(fresh, box.BANDS[-1][1])                  # below the 96k cliff band
        prompt = data['decode']['prompt_tokens']
        self.assertLessEqual(prompt + box.MAX_STANDING_BUDGET, box.MODEL_LEN)
        # KV at the append (it extends the fresh conversation): below held-e's W2 arm.
        kv = 8 * (prompt + box.MAX_STANDING_BUDGET) + append
        self.assertLess(kv, 8 * (prompt + w2['decode']['generation_budget']) + w2_append)


class Window(unittest.TestCase):
    def test_one_phase_budget_caps_every_held_f_step_and_keeps_native40_caps(self):
        import native40_box_gate as n40
        import native41_box_gate as n41
        self.assertEqual(gate.ACTIVE_S, sum(gate.PHASES.values()))
        self.assertEqual((gate.CLEANUP_S, gate.NOT_BEFORE, gate.HARD_END), (n40.CLEANUP_S, n40.NOT_BEFORE, n40.HARD_END))
        for name in n41.KEPT:
            self.assertEqual(gate.PHASES[name], n40.PHASES[name], name)
        self.assertFalse(set(gate.DROPPED) & set(gate.PHASES))
        body = queue.BODY
        caps = {'instruments': ('timeout 120s python3 "$H/e3_native39_trunk_checks.py" instruments', 2),
                'full-idle': ('timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle', 2),
                'arm': ('timeout --signal=TERM --kill-after=30s 1500s python3 "$H/e3_native41_progress_box.py" arm', 1),
                'ingress': ('timeout 60s python3 "$H/e3_native39_trunk_checks.py" ingress', 1),
                'grade': ('timeout 300s python3 "$H/e3_native41_progress_box.py" grade', 1)}
        for name, (text, lines) in caps.items():
            self.assertEqual(body.count(text), lines, name)
        self.assertEqual(len(re.findall(r'^\s*python3 "\$H/', body, flags=re.M)), 0)  # every tool call is capped
        self.assertEqual((gate.ARMS, len(queue.ARMS)), (8, 8))
        self.assertEqual(gate.PHASES['instruments'], 2 * n41.INSTRUMENTS_CAP_S)
        self.assertEqual((n41.INSTRUMENTS_CAP_S, n41.FULL_IDLE_CAP_S, n41.INGRESS_CAP_S, gate.GRADE_CAP_S),
                         (120, 450, 60, 300))
        self.assertEqual(gate.PHASES['arm_entries'], 8 * n41.FULL_IDLE_CAP_S)
        self.assertEqual(gate.PHASES['progress_arms'], 8 * (n41.ARM_TIMEOUT_S + n41.ARM_KILL_AFTER_S))
        self.assertEqual((n41.ARM_TIMEOUT_S, n41.ARM_KILL_AFTER_S), (1500, 30))
        self.assertEqual(gate.PHASES['ingress'], 8 * n41.INGRESS_CAP_S)
        self.assertEqual((gate.PHASES['end_full_idle'], gate.PHASES['grade']), (n41.FULL_IDLE_CAP_S, 300))
        self.assertEqual(gate.latest_arm(), gate.HARD_END - gate.ACTIVE_S - gate.CLEANUP_S)
        self.assertTrue(gate.window_ok(gate.latest_arm()))
        self.assertFalse(gate.window_ok(gate.latest_arm() + 1))
        self.assertFalse(gate.window_ok(gate.NOT_BEFORE - 1))
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
        self.assertIn('native41_progress_gate', slot)
        done = subprocess.run(['bash', '-c', 'fail_early() { return 2; }\n' + slot],
                              env=dict(os.environ, HARNESS_DIR=str(HERE)), capture_output=True, text=True)
        now = time.time()
        expected = 0 if gate.window_ok(now) else 2
        self.assertEqual(done.returncode, expected, done.stderr)
        SLOT_RECEIPTS.append(dict(draft='q18a_native41_progress.draft.sh', real_wall_clock=True,
                                  verdict='PASS' if done.returncode == 0 else 'OUTSIDE-WINDOW',
                                  active_s=gate.ACTIVE_S, cleanup_margin_s=gate.CLEANUP_S))


if __name__ == '__main__':
    unittest.main()
