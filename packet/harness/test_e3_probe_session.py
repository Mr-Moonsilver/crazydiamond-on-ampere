"""Actual W1 validator -> session -> controller, with only AsyncLLM/CUDA faked."""
import copy
import unittest

from test_e3_probe_control import Engine, ProbeController, ProbeControlError
from probe_session import ProbeSession, interval
import e3_w1_decode_driver as driver


class CountedEngine(Engine):
    def __init__(self):
        super().__init__()
        self.unfinished = 0
        self.steps = 10
        self.preempted = 0
        self.unknown = 0
        self.non_decode = 0

    async def collective_rpc(self, *args, **kwargs):
        rows = await super().collective_rpc(*args, **kwargs)
        if kwargs['args'][0]['action'] == 'status':
            for row in rows:
                b = row['counters']['batch_observations']
                b.update(decode_batches=self.steps, non_decode_batches=self.non_decode,
                         non_full_dispatches=0, empty_batches=0, preempted_requests=self.preempted,
                         unknown_preemption_observations=self.unknown, scheduler_observations=self.steps)
        return rows


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = CountedEngine()
        self.session = ProbeSession(ProbeController(self.engine, epoch=71))
        self.ctx = dict(run_id='composed-run', repeat=0, cohort_id='cohort-0', command_seq=0,
                        prior_receipt=None, streams=8, engine_epoch=None, storage_epoch=None,
                        route='full', route_generation=None)

    async def action(self, action, route):
        cmd = driver._next_command(self.ctx, action, route)
        receipt = await self.session.execute(cmd)
        check = dict(self.ctx, action=action, command_seq=cmd['command_seq'])
        ok, reasons, severity, accepted = driver.validate_receipt(receipt, cmd, check)
        self.assertTrue(ok, (action, reasons, severity))
        self.ctx.update(accepted)
        self.ctx['command_seq'] = cmd['command_seq']
        self.ctx['prior_receipt'] = dict(command_seq=cmd['command_seq'],
                                        receipt_sha256=driver._sha256_json(receipt))
        return receipt

    async def enter(self, route):
        await self.action('prepare', 'full')
        self.engine.unfinished = 8
        await self.action('pause_enter_split' if route == 'split' else 'pause_full', route)
        await self.action('resume', route)

    async def test_actual_client_validator_accepts_full_split_full_and_next_repeat(self):
        for repeat, route in enumerate(('full', 'split')):
            self.ctx.update(repeat=repeat, cohort_id=f'cohort-{repeat}')
            await self.enter(route)
            self.engine.steps += 5
            end = await self.action('pause_end', route)
            self.assertEqual(end['interval']['counters_after'][0]['decode_steps'], self.engine.steps)
            left = await self.action('leave_full', 'full')
            if route == 'full':
                self.assertEqual(left['route_generation_after'], 0)
                self.assertNotIn('transition_ack', left['ranks'][0])
            else:
                self.assertEqual(left['route_generation_after'], 2)
                self.assertEqual(left['ranks'][0]['transition_ack']['ack'], 'flash_s5_transition')
            await self.action('resume', 'full')
            self.engine.unfinished = 0
        self.assertFalse(self.session.failed)

    async def test_previous_receipt_hash_and_sequence_are_required(self):
        await self.action('prepare', 'full')
        cmd = driver._next_command(self.ctx, 'pause_full', 'full')
        for change in (dict(prior_receipt=None), dict(command_seq=4), dict(cohort_size=7)):
            with self.assertRaises(ProbeControlError):
                await self.session.execute(dict(cmd, **change))
        self.assertEqual(self.session.phase, 'prepared')

    async def test_unknown_end_proof_quarantines_and_never_leaves_or_resumes(self):
        await self.enter('split')
        self.engine.steps += 1
        self.engine.unknown += 1
        with self.assertRaisesRegex(ProbeControlError, 'preemption-proof'):
            await self.session.execute(driver._next_command(self.ctx, 'pause_end', 'split'))
        self.assertTrue(self.engine.paused)
        self.assertTrue(self.session.failed)
        with self.assertRaisesRegex(ProbeControlError, 'quarantined'):
            await self.session.execute(driver._next_command(self.ctx, 'leave_full', 'full'))
        self.assertEqual(self.engine.route, 'split')

    async def test_partial_real_transition_replies_do_not_create_receipt(self):
        await self.action('prepare', 'full')
        self.engine.unfinished = 8
        self.engine.corrupt = lambda c,rs: rs[:-1] if c['action'] == 'transition' else rs
        with self.assertRaises(ProbeControlError):
            await self.session.execute(driver._next_command(self.ctx, 'pause_enter_split', 'split'))
        self.assertTrue(self.engine.paused)
        self.assertTrue(self.session.failed)
        self.assertFalse(any(c[0] == 'resume' for c in self.engine.calls))

    async def test_present_preemption_is_preserved_and_client_refuses_interval(self):
        await self.enter('full')
        self.engine.steps += 3
        self.engine.preempted += 1
        cmd = driver._next_command(self.ctx, 'pause_end', 'full')
        receipt = await self.session.execute(cmd)
        ok, reasons, _, _ = driver.validate_receipt(receipt, cmd,
            dict(self.ctx, action='pause_end', command_seq=cmd['command_seq']))
        self.assertFalse(ok)
        self.assertIn('contract:interval_invalidated:preempted', reasons)
        self.assertEqual(receipt['interval']['invalidation']['preempted'], 4)

    async def test_missing_counter_not_default_zero(self):
        await self.enter('full')
        snapshot = self.session.before
        after = copy.deepcopy(snapshot)
        del after['ranks'][0]['counters']['batch_observations']['preempted_requests']
        with self.assertRaises(ProbeControlError): interval(snapshot, after)

    async def test_old_cohort_cannot_overlap_new_prepare(self):
        await self.action('prepare', 'full')
        self.engine.unfinished = 8
        cmd = driver._next_command(dict(self.ctx, repeat=1,cohort_id='cohort-1'), 'prepare','full')
        with self.assertRaisesRegex(ProbeControlError, 'repeat-boundary'):
            await self.session.execute(cmd)



class DriverCompositionTests(unittest.TestCase):
    def test_real_live_driver_uses_session_controller_and_validator(self):
        import asyncio
        import json
        from pathlib import Path
        import tempfile
        from test_e3_w1_decode_driver import BASE_MANIFEST, FakeConfig, FakeServer
        engine = CountedEngine()
        loop = asyncio.new_event_loop()
        async def create(): return ProbeSession(ProbeController(engine, epoch=71))
        session = loop.run_until_complete(create())
        seen = []
        class Adapter:
            def execute(self, command, timeout_s):
                seen.append(command['action'])
                engine.unfinished = 0 if command['action'] == 'prepare' else 8
                if command['action'] == 'pause_end': engine.steps += 5
                receipt = loop.run_until_complete(session.execute(command))
                return receipt, {'kind':'actual-session/fake-AsyncLLM', 'rc':0}
        server = FakeServer(FakeConfig(streams=8))
        try:
            with tempfile.TemporaryDirectory() as folder:
                manifest = copy.deepcopy(BASE_MANIFEST)
                manifest['streams'] = 8
                path = Path(folder)/'manifest.json'; path.write_text(json.dumps(manifest))
                out = Path(folder)/'result.json'
                cfg = driver.config_from_env({'WORKLOAD':str(path),'RUN_ID':'composed-driver'},str(out))
                result = driver.run(cfg,server.url,'fake-model',Adapter())
                self.assertEqual(result['summary']['verdict'], 'COMPLETE',result['summary']['invalid'])
                self.assertEqual(len(server.measure_requests()),16)
                self.assertEqual(seen, ['prepare','pause_full','resume','pause_end','leave_full','resume',
                                       'prepare','pause_enter_split','resume','pause_end','leave_full','resume'])
                self.assertEqual(engine.route,'full')
        finally:
            server.stop(); loop.close()

if __name__ == '__main__': unittest.main()
