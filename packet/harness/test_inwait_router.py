"""inwait: one real-router smoke for every action the box client sends, with the three flags installed as the settings file installs them.

The box's client (e3_native_runtime_client.NativeRuntimeClient, e3_inwait_box.release_compose, e3_attn_meta_box.end_read) goes over real loopback HTTP into the real probe_control.attach_native_router and
native_qualification_frontend.attach_router, NativeOverlapSession, NativeControlClient, AutoArmPolicy, the utility channel, the overlap driver's control and each rank's native_overlap_worker. Leaves: fastapi's
own request and response objects; each rank's attn_meta_select.Selector stands over a counting batch (as test_attn_meta_control's). Control path only: nothing here runs a chunk.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import asyncio
import json
import os
from pathlib import Path
import re
import threading
import types
import unittest
from unittest import mock

import test_alpha_next_c1_defer_control as dctl   # first: its imports put the engine sources on the path
import test_attn_meta_control as ac
import test_native41_box_queue as held
import inwait_arms as T
import e3_attn_meta_box as am
import e3_inwait_box as box
import e3_native_runtime_client as rc
import native_qualification_frontend as nqf
import probe_control

HERE = Path(__file__).resolve().parent
SETTINGS = HERE / 'runs' / 'q18a_inwait_settings.sh'
RUN = 'inwait-router'
SAMPLE = dict(mode='burst', fused=True, size=2, local=True, overlap=True, defer_bind=True, lazy_recv=True, decode_merge='chunk', inwait=True, early_successor=True)


def settings_env():
    """The FLASH values the box launches with (paths to auth files stay out)."""
    found = dict(re.findall(r'\b(FLASH_(?:S5|ATTN)_[A-Z0-9_]+)=([^\s",;]*)', SETTINGS.read_text()))
    found.pop('FLASH_S5_AUTH_ENV_FILE', None)
    return found


class Bridge:
    """Real loopback HTTP into the routers' endpoints, by path."""

    def __init__(self, app, endpoints, loop):
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def do_POST(self):
                data = self.rfile.read(int(self.headers['Content-Length']))
                request = held.Request(app, data, self.headers.get('Authorization', ''))
                request.url = types.SimpleNamespace(path=self.path)
                reply = asyncio.run_coroutine_threadsafe(endpoints[self.path](request), loop).result(60)
                content = json.dumps(reply.content).encode()
                self.send_response(reply.status_code)
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def log_message(self, *_):
                pass
        self.http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = 'http://127.0.0.1:%d' % self.http.server_address[1]
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def close(self):
        self.http.shutdown()
        self.http.server_close()


class Base(dctl.RouterBase):
    async def asyncSetUp(self):
        self.env_box = settings_env()
        patch = mock.patch.dict(os.environ, dict(self.env_box, VLLM_API_KEY=held.KEY))
        patch.start()
        self.addCleanup(patch.stop)
        await super().asyncSetUp()
        self.driver.install_progress('burst')
        self.driver.install_qsa_fused()
        self.driver.install_layer_groups()
        self.selectors = {rank: ac.put_selector(self.stack.worker(rank).model_runner) for rank in (0, 1, 2, 3)}
        router = dict(self.env_box, VLLM_API_KEY=held.KEY)
        self.app = held.App(self.frontend)
        with held.fastapi_leaf():
            probe_control.attach_native_router(types.SimpleNamespace(api_key=None), self.app, env=router)
            nqf.attach_router(types.SimpleNamespace(api_key=None), self.app, env=router)
        self.endpoints = {path: endpoint for r in self.app.routers for path, endpoint, _methods in r.routes}
        self.bridge = Bridge(self.app, self.endpoints, asyncio.get_running_loop())
        self.addCleanup(self.bridge.close)

    def env(self):
        return mock.patch.dict(os.environ, {})

    def client(self):
        return rc.NativeRuntimeClient(self.bridge.url, RUN, epoch=self.driver.engine_epoch, timeout_s=60)

    async def off_loop(self, fn):
        return await asyncio.get_running_loop().run_in_executor(None, fn)


class Exposure(Base):
    async def test_settings_install_the_three_flags(self):
        for name in ('FLASH_S5_LAZY_RECV', 'FLASH_S5_DECODE_MERGE', 'FLASH_ATTN_META_GDN_BATCH', 'FLASH_S5_INWAIT'):
            self.assertEqual(self.env_box.get(name), '1', name)
        self.assertEqual((self.driver.lazy_recv_installed, self.driver.decode_merge_installed, self.driver.inwait_installed), (True, True, True))
        self.assertEqual(set(self.endpoints), {probe_control.NATIVE_PATH, nqf.FRONTEND_PATH, nqf.WORKER_PATH})

    async def test_every_box_action_gets_a_200(self):
        """Every action in box.ACTIONS, with its fields, in the box's order; any that does not return 200 fails by name."""

        def smoke():
            client = self.client()
            client.prepare('full', 'full-prefill-solo')
            worker = am.Worker(client.control, RUN, client.epoch)
            out = {}
            for action, fields in box.ACTIONS:
                try:
                    if action == 'qualification_worker':
                        out[am.SELECT] = worker.call(am.SELECT, {'on': True})
                        out['slots_fast'] = worker.call(am.SELECT, {'on': True, 'slots_fast': False})
                        out[am.READ] = worker.call(am.READ, {})
                    else:
                        out[action] = client.call(action, **{name: SAMPLE[name] for name in fields})
                except Exception as exc:  # noqa: BLE001 - the test reads it
                    out[action] = repr(exc)
            return out
        out = await self.off_loop(smoke)
        self.assertEqual({k: v for k, v in out.items() if type(v) is str}, {})
        self.assertEqual(set(out), {a for a, _f in box.ACTIONS} - {'qualification_worker'} | {am.SELECT, am.READ, 'slots_fast'})


class BoxCalls(Base):
    async def test_each_selection_through_the_release_and_the_end_read(self):
        """The box's own release for each selection in turn (each switch moves from the last): replies, driver, status view, all ranks' selectors."""
        rows = {r['c1']: r for r in T.arms()}

        def one(row):
            client = self.client()
            client.prepare('full', 'full-prefill-solo')
            record = {}
            try:
                box.release_compose(row)(2, True)(client, record, 'burst')
            except Exception as exc:  # noqa: BLE001 - the test reads it
                return record, exc, None, None
            client.pause()
            status = client.call('status')
            client.resume()
            args = types.SimpleNamespace(url=self.bridge.url, run_id=RUN, epoch=client.epoch, control_timeout_s=60)
            return record, None, status, am.end_read(args)
        for sel in ('nx1', 'nx0', 'nx1'):
            row = rows[sel]
            with self.subTest(sel=sel):
                record, exc, status, end = await self.off_loop(lambda row=row: one(row))
                self.assertIsNone(exc, (record, exc))
                self.assertIs(record['release_resume'], True)
                self.assertEqual((record['step']['acknowledged_lazy'], record['step']['acknowledged_merge']), (row['lazy_recv'], row['decode_merge']))
                self.assertEqual((self.driver.lazy_recv, self.driver.decode_merge), (row['lazy_recv'], row['decode_merge']))
                self.assertIs(self.driver.defer_bind, True)
                self.assertEqual((record['inwait']['acknowledged'], self.driver.inwait), (row['inwait'], row['inwait']))
                self.assertEqual((record['inwait']['early_acknowledged'], self.driver.early_successor), (row['next_levers'], row['next_levers']))
                self.assertEqual({r['after']['slots_fast']['selected'] for r in record['slots_fast']['select'].values()}, {row['next_levers']})
                self.assertEqual({r['counters']['slots_fast']['selected'] for r in end['read'].values()}, {row['next_levers']})
                self.assertEqual(box.slots_calls(record['slots_fast']['select'], end['read']), dict.fromkeys(('0', '1', '2', '3'), 0))
                view = box.stage_view(status)
                self.assertIs(view['step_iw_selected'], row['inwait'])
                self.assertIs(view['step_iw_installed'], True)
                self.assertEqual((view['step_lazy'], view['step_merge']), (row['lazy_recv'], row['decode_merge']))
                for rank in self.selectors:
                    self.assertIs(record['attn_meta']['select'][str(rank)]['after']['selected'], row['attn'], rank)
                self.assertNotIn('error', end)
                self.assertEqual((end['paused'], end['resumed']), (True, True))
                self.assertEqual(sorted(end['read']), ['0', '1', '2', '3'])


if __name__ == '__main__':
    unittest.main()
