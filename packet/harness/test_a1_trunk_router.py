"""a1-trunk: one real-router smoke for every action the box client sends, and the arm itself end to end over it (the 409 lesson).

The box's client (e3_native_refusal_client over loopback HTTP) goes into the real probe_control.attach_native_router and native_qualification_frontend.attach_router, NativeOverlapSession, NativeControlClient, AutoArmPolicy,
the utility channel and the overlap driver's control, with the router and process env read from the settings file the box launches with. Only fastapi's request and response objects are leaves (test_inwait_router's Base).
  * Exposure: every action in box.ACTIONS, with its fields; any that does not return 200 fails by name. The mutant without the row gives the 409.
  * BoxCall: box.yield_control on and off gets a 200 and a confirmed record.
  * ArmEnd: box.main arm, the exact CLI, for each kind of row: the real control sequence through the real router, streams from the fixture's serving leaf (held.Server). No native chunk runs here, by construction.
Nothing here runs a yield; test_a1_trunk and test_append_yield do, without the router.
"""
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import test_inwait_router as iwr    # first: its imports put the engine sources on the path
import test_alpha_perf_queue as tap
import test_native41_progress_queue as hf
import a1_trunk_arms as T
import e3_a1_trunk_box as box
import e3_native_refusal_client as rc
import native_qualification_frontend as nqf
import probe_control
import scheduler_hooks as sh

held = iwr.held
HERE = Path(__file__).resolve().parent
SETTINGS = HERE / 'runs' / 'q18a_a1_trunk_settings.sh'
RUN = 'a1-trunk-router'
ROW = ("    if auto_actions and all(env.get(name) is not None for name in APPEND_YIELD_ENV):\n"
       "        auto_actions += ('native_append_yield',)\n")
SAMPLE = dict(iwr.SAMPLE, enabled=True)
QUICK = {'pair': ((7, 64, 0.02), (6, 64, 0.05)), 'several': ((7, 64, 0.02), (6, 64, 0.02), (5, 64, 0.02))}


def without_row():
    """A private probe_control whose router lacks the row: the bytes the q18a-append-yield-h box ran, for this action."""
    source = Path(probe_control.__file__).read_text()
    assert source.count(ROW) == 1
    module = types.ModuleType('probe_control_without_the_row')
    module.__file__ = probe_control.__file__
    exec(compile(source.replace(ROW, ''), probe_control.__file__ + ' (mutant)', 'exec'), module.__dict__)
    return module


class Base(iwr.Base):
    async def asyncSetUp(self):
        with mock.patch.object(iwr, 'SETTINGS', SETTINGS):
            await super().asyncSetUp()
        self.driver.controller.append_yield = sh.AppendYieldConfig(2048, 4, 16, 4)

    def client(self):
        return rc.RefusalKeepingClient(self.bridge.url, RUN, epoch=self.driver.engine_epoch, timeout_s=60)


class Exposure(Base):
    async def test_settings_install_the_bounds(self):
        for name, value in T.LAUNCH_ENV.items():
            self.assertEqual(self.env_box.get(name), str(value), name)
        self.assertEqual(self.env_box.get('FLASH_S5_INWAIT'), '1')

    async def test_every_box_action_gets_a_200(self):
        def smoke():
            client = self.client()
            client.prepare('full', 'full-prefill-solo')
            worker = iwr.am.Worker(client.control, RUN, client.epoch)
            out = {}
            for action, fields in box.ACTIONS:
                try:
                    if action == 'qualification_worker':
                        out[action] = worker.call(iwr.am.SELECT, {'on': True, 'slots_fast': True, 'prepare_slices': True, 'score_warm': False, 'cow_fast': True, 'reserved_cap': False, 'cow_host_filter': True})
                    else:
                        out[action] = client.call(action, **{name: SAMPLE[name] for name in fields})
                except Exception as exc:  # noqa: BLE001 - the test reads it
                    out[action] = repr(exc)
            return out
        out = await self.off_loop(smoke)
        self.assertEqual({k: v for k, v in out.items() if type(v) is str}, {})
        self.assertEqual(set(out), {a for a, _f in box.ACTIONS})
        self.assertIs(out['native_append_yield']['enabled'], True)

    async def test_the_router_without_the_row_is_the_boxs_409(self):
        module = without_row()
        router = dict(self.env_box, VLLM_API_KEY=held.KEY)
        app = held.App(self.frontend)
        with held.fastapi_leaf():
            module.attach_native_router(types.SimpleNamespace(api_key=None), app, env=router)
        (_path, endpoint, _methods), = app.routers[0].routes
        reply = await endpoint(held.Request(app, json.dumps(dict(schema=1, action='native_append_yield', run_id=RUN, enabled=True)).encode(), 'Bearer ' + held.KEY))
        self.assertEqual(reply.status_code, 409, reply.content)


class BoxCall(Base):
    async def test_yield_control_confirms_on_and_off(self):
        def one(enabled):
            client = self.client()
            client.prepare('full', 'full-prefill-solo')
            record = {}
            box.yield_control(client, enabled, record)
            return record
        for enabled in (True, False, True):
            record = await self.off_loop(lambda enabled=enabled: one(enabled))
            view = record['yield_control']
            self.assertIs(view['confirmed'], True, view)
            self.assertIs(view['enabled'], enabled)
            self.assertIs(view['clock_proof']['inside'], True, view['clock_proof'])
            self.assertEqual({k: view[k] for k in T.LAUNCH}, T.LAUNCH)
            self.assertIs(record['yield_resume'], True)
            self.assertIs(self.driver.controller.append_yield.enabled, enabled)


class ArmEnd(Base):
    """box.main's arm through the real router; the completions, /metrics and the native-work line are held-f's serving leaves."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        loop = asyncio.get_running_loop()
        native = self.endpoints[probe_control.NATIVE_PATH]
        leaf = types.SimpleNamespace(run=lambda coro, timeout: asyncio.run_coroutine_threadsafe(coro, loop).result(timeout))
        server = self.server = hf.ProgressServer(leaf, self.app, native)
        others = {nqf.FRONTEND_PATH: self.endpoints[nqf.FRONTEND_PATH], nqf.WORKER_PATH: self.endpoints[nqf.WORKER_PATH]}
        app = self.app

        class Handler(server.http.RequestHandlerClass):
            def do_POST(self):
                if self.path == '/tokenize':                # the image's route; the fixture counts words. On the box: w2-length's call, 200 in run 1
                    text = json.loads(self.rfile.read(int(self.headers['Content-Length'])))['prompt']
                    content = json.dumps(dict(tokens=[0] * len(text.split()))).encode()
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    return None
                if self.path not in others:
                    return super().do_POST()
                data = self.rfile.read(int(self.headers['Content-Length']))
                request = held.Request(app, data, self.headers.get('Authorization', ''))
                request.url = types.SimpleNamespace(path=self.path)
                reply = leaf.run(others[self.path](request), 60)
                content = json.dumps(reply.content).encode()
                self.send_response(reply.status_code)
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return None
        server.http.RequestHandlerClass = Handler
        server.__enter__()
        self.addCleanup(server.__exit__)
        self.workload = str(tap.scale_workload(self.tmp.name, 'q18a-native41-progress'))

    async def arm(self, sel, cohort, index=0, argv_extra=()):
        def run():
            client = self.client()
            client.prepare('full', 'full-prefill-solo')
            client.resume()
            name = T.label(sel, 2, cohort, index)
            out = Path(self.tmp.name) / ('arm-%s.json' % name)
            argv = ['arm', '--url', self.server.url, '--model', 'fixture-model', '--run-id', RUN, '--epoch', str(self.driver.engine_epoch), '--c1', sel, '--size', '2', '--cohort', cohort, '--index', str(index),
                    '--workload', self.workload, '--control-timeout-s', '30', '--out', str(out)]
            mark, served = len(self.server.session_actions), len(self.server.completions)
            with mock.patch.object(box.shared, '_package_gate', return_value={'ok': True}), mock.patch.object(box.base, 'SETTLE_S', 0.2), mock.patch.object(box.base, 'CLOSE_TIMEOUT_S', 1.0), \
                    mock.patch.dict(T.PLANS, QUICK), contextlib.redirect_stdout(io.StringIO()):
                rc_ = box.main(argv)
            return rc_, json.loads(out.read_text()), self.server.session_actions[mark:], self.server.completions[served:]
        return await self.off_loop(run)

    async def test_a_live_on_arm_runs_its_control_and_its_returners(self):
        rc_, record, actions, served = await self.arm('sm', 'live')
        self.assertNotIn('error', record, record.get('error'))
        self.assertEqual(rc_, 0, record['reasons'])
        self.assertEqual(record['reasons'], [])
        self.assertEqual(actions[:3], ['status', 'pause', 'native_append_yield'], actions[:12])
        self.assertLess(actions.index('native_append_yield'), actions.index('native_stage_local'))
        self.assertLess(actions.index('native_stage_local'), actions.index('release_auto_arm'))
        self.assertIs(record['yield_control']['enabled'], True)
        self.assertEqual(record['always_on'], [0, 1, 2, 3, 4, 5])
        self.assertEqual(len(served), T.by_label()['sm.g2-live-0']['generations'])
        self.assertEqual([c['max_tokens'] for c in served].count(T.REPLY_TOKENS), 2)
        returned = [c['cache_salt'] for c in served if c['max_tokens'] == T.REPLY_TOKENS]
        self.assertEqual(set(returned), {box.shared._cache_salt(RUN, box.progress.STANDING, i) for i in (7, 6)}, 'a returner comes back under its own salt')
        self.assertEqual([r['index'] for r in record['returners']], [7, 6])
        for row in record['returners']:
            self.assertIs(row['ok'], True, row)
            self.assertEqual(row['readers_alive'], [])
            self.assertGreater(row['ttft_s'], 0)
        ok, reasons = box.smoke_check(record)
        self.assertEqual(reasons, ['smoke-cow-fast-not-engaged', 'smoke-host-filter-not-engaged', 'smoke-no-native-chunks', 'smoke-no-yield-served'], 'everything but the native chunks, the yield and the copies they would make, which this fixture does not run')
        self.assertFalse(ok)

    async def test_an_off_arm_sends_enabled_false_and_several_return_at_once(self):
        rc_, record, _actions, served = await self.arm('mo', 'live')
        self.assertNotIn('error', record, record.get('error'))
        self.assertEqual(rc_, 0, record['reasons'])
        self.assertIs(record['yield_control']['enabled'], False)
        self.assertIs(self.driver.controller.append_yield.enabled, False)
        self.assertEqual(sorted(r['index'] for r in record['returners']), [5, 6, 7])
        self.assertEqual(len(served), 13)

    async def test_a_solo_arm_has_no_returner_and_cancels_the_standing_streams(self):
        rc_, record, _actions, served = await self.arm('on', 'solo')
        self.assertNotIn('error', record, record.get('error'))
        self.assertEqual(rc_, 0, record['reasons'])
        self.assertEqual(record['returners'], [])
        self.assertIs(record['solo_cancel']['triggered'], True)
        self.assertEqual(len(served), 10)


if __name__ == '__main__':
    unittest.main()
