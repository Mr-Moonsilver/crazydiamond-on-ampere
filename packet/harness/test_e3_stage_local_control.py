#!/usr/bin/env python3
"""C0's per-arm switch over the real router, session, control client and policy.

``native_stage_local {local}`` reaches the driver only with FLASH_S5_STAGE_LOCAL=1 and
FLASH_S5_AUTO_ARM=1 in both the API server's and the engine's environment.  Every other
setting refuses it exactly as the frozen engine refuses an unknown action.

Real objects: the endpoint, NativeOverlapSession, NativeControlClient, the utility channel
pump, AutoArmPolicy.  The driver is the cold stack's driver with the staged class's control
methods; the staged class's chunk path is covered by test_e3_stage_local.py.
Fixture leaves: as test_native41_progress_queue.py's router cases.
"""
import contextlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'tools'), str(ROOT / 'patches/s5_engine')]
import engine_driver as ed  # noqa: E402
import native_auto_arm as auto  # noqa: E402
import probe_control as pc  # noqa: E402
from stage_progress import StagedSegmentDriver  # noqa: E402
import test_native39_trunk as prior  # noqa: E402
import test_native41_box_queue as held  # noqa: E402

KEY = held.KEY
BOTH = {'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_STAGE_LOCAL': '1'}


def attach(engine_client, auto_arm, stage):
    env = {'FLASH_S5_OVERLAP': '1', 'VLLM_API_KEY': KEY, 'FLASH_S5_AUTO_ARM': auto_arm}
    if stage is not None:
        env['FLASH_S5_STAGE_LOCAL'] = stage
    app = held.App(engine_client)
    with held.fastapi_leaf():
        pc.attach_native_router(SimpleNamespace(api_key=None), app, env=env)
    (path, endpoint, methods), = app.routers[0].routes
    assert (path, methods) == (pc.NATIVE_PATH, ('POST',))
    return app, endpoint


@contextlib.contextmanager
def process_env(values):
    with mock.patch.dict(os.environ, values):
        for name in BOTH:
            if name not in values:
                os.environ.pop(name, None)
        yield


class Base(prior.Session):
    async def asyncSetUp(self):
        with mock.patch.object(ed, 'NativeSegmentDriver', StagedSegmentDriver):
            await super().asyncSetUp()
        held.pinned_reads(self.scheduler)
        await self.call('prepare', target_route='full', phase='full-prefill-solo')
        self.policy = auto.AutoArmPolicy(self.core, self.driver)
        self.driver.auto_arm = self.policy
        self.policy.manual_owned = True

    async def post(self, endpoint, app, action, **fields):
        body = json.dumps(dict(schema=1, action=action, run_id='c0-router', **fields)).encode()
        reply = await endpoint(held.Request(app, body, 'Bearer ' + KEY))
        return reply.status_code, reply.content


class Router(Base):
    async def test_the_switch_reaches_the_driver_only_paused_idle_full(self):
        self.assertIsInstance(self.driver, StagedSegmentDriver)
        with process_env(BOTH):
            app, endpoint = attach(self.frontend, '1', '1')
            self.assertEqual(await self.post(endpoint, app, 'native_stage_local', local=False),
                             (400, {'error': 'invalid-command-json'}))
            self.assertIs(self.driver.stage_selected, True)
            self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
            self.assertEqual(await self.post(endpoint, app, 'native_stage_local', local=1),
                             (409, {'error': 'native-control-stage-local-invalid'}))
            self.assertIs(self.driver.stage_selected, True)
            code, reply = await self.post(endpoint, app, 'native_stage_local', local=False)
            self.assertEqual(code, 200, reply)
            self.assertEqual({k: reply[k] for k in ('action', 'local', 'previous', 'route')},
                             dict(action='native_stage_local', local=False, previous=True, route='full'))
            self.assertEqual(reply['engine_epoch'], self.driver.engine_epoch)
            self.assertIs(self.driver.stage_selected, False)
            code, status = await self.post(endpoint, app, 'status')
            self.assertEqual(code, 200, status)
            row = status['native_stage_local']
            self.assertEqual((row['selected'], row['counters']['selections']), (False, 1))
            # A manual control: the lease stays until the explicit release.
            self.assertIs(self.policy.manual_owned, True)
            self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)

    async def test_every_other_setting_refuses_it(self):
        cases = (('1', None, BOTH, 'native-action-not-exposed'),
                 ('0', '1', BOTH, 'native-action-not-exposed'),
                 ('1', '0', BOTH, 'native-action-not-exposed'),
                 ('1', '1', {'FLASH_S5_AUTO_ARM': '1'}, 'session-action-not-exposed'),
                 ('1', '1', {'FLASH_S5_AUTO_ARM': '0', 'FLASH_S5_STAGE_LOCAL': '1'},
                  'session-action-not-exposed'))
        for router_auto, router_stage, process, error in cases:
            with self.subTest(router=(router_auto, router_stage), process=process), process_env(process):
                app, endpoint = attach(self.frontend, router_auto, router_stage)
                self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
                self.assertEqual(await self.post(endpoint, app, 'native_stage_local', local=False),
                                 (409, {'error': error}))
                self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)
                self.assertIs(self.driver.stage_selected, True)


if __name__ == '__main__':
    unittest.main()
