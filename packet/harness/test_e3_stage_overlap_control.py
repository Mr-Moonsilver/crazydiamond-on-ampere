#!/usr/bin/env python3
"""C1's per-arm field over the real router, session, control client and policy.

``native_stage_local {local, overlap}``: the ``overlap`` field reaches the driver only where
FLASH_S5_STAGE_OVERLAP=1 is installed beside C0's two switches.  Without it the control client
refuses the field before anything is sent, and a C0-shaped command is unchanged.

Real objects: the endpoint, NativeOverlapSession, NativeControlClient, the utility channel
pump, AutoArmPolicy.  The driver is the cold stack's driver with the overlap class and the
overlap controller (both real); the chunk path is covered by test_e3_stage_overlap.py.
Fixture leaves: as test_e3_stage_local_control.py (test_native41_progress_queue.py's router cases).
"""
from unittest import mock
import unittest

import test_e3_stage_local_control as c0
from test_e3_stage_local_control import attach, process_env
import engine_driver as ed
import scheduler_hooks as sh
from stage_overlap import OverlapSegmentDriver, StagedAdmissionController

ALL = dict(c0.BOTH, FLASH_S5_STAGE_OVERLAP='1')


class Base(c0.Base):
    async def asyncSetUp(self):
        with mock.patch.object(ed, 'NativeSegmentDriver', OverlapSegmentDriver), \
                mock.patch.object(sh, 'PrefillAdmissionController', StagedAdmissionController), \
                mock.patch.object(c0, 'StagedSegmentDriver', OverlapSegmentDriver):
            await super().asyncSetUp()


class Router(Base):
    async def test_overlap_field_reaches_the_driver_only_where_installed(self):
        self.assertIsInstance(self.driver, OverlapSegmentDriver)
        self.assertIsInstance(self.controller, StagedAdmissionController)
        with process_env(ALL), mock.patch.dict('os.environ', {'FLASH_S5_STAGE_OVERLAP': '1'}):
            app, endpoint = attach(self.frontend, '1', '1')
            self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
            self.assertEqual(await self.post(endpoint, app, 'native_stage_local', local=True, overlap=1),
                             (409, {'error': 'native-control-stage-overlap-invalid'}))
            # An engine-side refusal reaches the router as C0's unpaused refusal does.
            self.assertEqual(await self.post(endpoint, app, 'native_stage_local', local=False, overlap=True),
                             (400, {'error': 'invalid-command-json'}))
            self.assertEqual((self.driver.stage_selected, self.driver.overlap_selected), (True, True))
            code, reply = await self.post(endpoint, app, 'native_stage_local', local=True, overlap=False)
            self.assertEqual(code, 200, reply)
            self.assertEqual({k: reply[k] for k in ('local', 'overlap', 'previous_overlap', 'route')},
                             dict(local=True, overlap=False, previous_overlap=True, route='full'))
            self.assertEqual((self.driver.stage_selected, self.driver.overlap_selected), (True, False))
            code, status = await self.post(endpoint, app, 'status')
            self.assertEqual(code, 200, status)
            self.assertIs(status['native_stage_local']['overlap']['selected'], False)
            code, reply = await self.post(endpoint, app, 'native_stage_local', local=False)
            self.assertEqual(code, 200, reply)
            self.assertEqual((reply['local'], reply['overlap']), (False, False))
            self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)

    async def test_without_the_overlap_switch_the_field_is_refused_and_c0_commands_unchanged(self):
        with process_env(c0.BOTH), mock.patch.dict('os.environ', {'FLASH_S5_STAGE_OVERLAP': '0'}):
            app, endpoint = attach(self.frontend, '1', '1')
            self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
            # The API server without the switch sends C0's command, as before C1: the field never
            # reaches the engine, so the engine's choice stays and its reply says so (a runner that
            # checks reply['overlap'] against its request sees the difference).
            code, reply = await self.post(endpoint, app, 'native_stage_local', local=True, overlap=False)
            self.assertEqual(code, 200, reply)
            self.assertIs(reply.get('overlap'), True)
            self.assertEqual((self.driver.stage_selected, self.driver.overlap_selected), (True, True))
            with self.assertRaises(Exception) as caught:
                await self.control.native_stage_overlap(True, False)
            self.assertIn('native-control-stage-overlap-not-installed', str(caught.exception))
            code, reply = await self.post(endpoint, app, 'native_stage_local', local=False)
            self.assertEqual(code, 200, reply)
            self.assertIs(self.driver.stage_selected, False)
            self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)


if __name__ == '__main__':
    unittest.main()
