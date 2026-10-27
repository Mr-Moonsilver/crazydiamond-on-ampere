"""Actual W1 validator -> session -> controller, with only AsyncLLM/CUDA faked."""
import copy
import json
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
        self.empty_batches = 0
        self.drop_status_keys = ()

    async def collective_rpc(self, *args, **kwargs):
        rows = await super().collective_rpc(*args, **kwargs)
        if kwargs['args'][0]['action'] == 'status':
            for row in rows:
                b = row['counters']['batch_observations']
                b.update(decode_batches=self.steps, non_decode_batches=self.non_decode,
                         non_full_dispatches=0, empty_batches=self.empty_batches,
                         preempted_requests=self.preempted,
                         unknown_preemption_observations=self.unknown,
                         scheduler_observations=self.steps)
                for key in self.drop_status_keys:
                    b.pop(key, None)
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
        self.ctx['prior_receipt'] = dict(
            command_seq=cmd['command_seq'],
            receipt_sha256=driver._sha256_json(driver._receipt_core(receipt)))
        return receipt

    async def enter(self, route):
        await self.action('prepare', 'full')
        self.engine.unfinished = 8
        await self.action('pause_enter_split' if route == 'split' else 'pause_full', route)
        await self.action('resume', route)

    async def test_partial_pp_batch_composes_with_live_census_and_client(self):
        def partial(command, rows):
            if command['action'] == 'status':
                for row in rows:
                    row['counters']['batch_observations']['last'].update(num_reqs=4,num_tokens=4)
                    row['counters']['live_decode']=dict(basis='worker-cpu-live-decode-v1',
                        available=True,live_requests=8,prefill_requests=0,
                        batch_is_live_subset=True,last_batch_member_count=4)
            return rows
        self.engine.corrupt=partial
        await self.enter('split')
        self.engine.steps += 5
        await self.action('pause_end','split')
        await self.action('leave_full','full')
        await self.action('resume','full')
        self.assertFalse(self.session.failed)

    async def test_partial_batch_failure_snapshot_survives_client_cleanup(self):
        def partial(command, rows):
            if command['action'] == 'status':
                for row in rows:
                    row['counters']['batch_observations']['last'].update(num_reqs=4,num_tokens=4)
            return rows
        self.engine.corrupt=partial
        await self.action('prepare','full'); self.engine.unfinished=8
        command=driver._next_command(self.ctx,'pause_full','full')
        with self.assertRaises(ProbeControlError): await self.session.execute(command)
        failure=copy.deepcopy(self.session.controller.eligibility_failure)
        self.assertEqual(failure['batch']['num_reqs'],4)
        self.engine.unfinished=0
        def cleared(command,rows):
            if command['action']=='status':
                for row in rows:row['counters']['batch_observations']['last']=None
            return rows
        self.engine.corrupt=cleared
        status=await self.session.controller._observe()
        self.assertEqual(status['eligibility_failure'],failure)
        self.assertIsNone(status['workers'][0]['counters']['batch_observations']['last'])

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

    async def _action_lifecycle(self, action, route):
        """Send/advance the chain using independent lifecycle proof."""
        cmd = driver._next_command(self.ctx, action, route)
        receipt = await self.session.execute(cmd)
        check = dict(self.ctx, action=action, command_seq=cmd['command_seq'])
        ok, reasons, severity, accepted = driver.validate_lifecycle_receipt(
            receipt, cmd, check)
        self.assertTrue(ok, (action, reasons, severity))
        self.ctx.update(accepted)
        self.ctx['command_seq'] = cmd['command_seq']
        self.ctx['prior_receipt'] = dict(
            command_seq=cmd['command_seq'],
            receipt_sha256=driver._sha256_json(driver._receipt_core(receipt)))
        return receipt, cmd, check

    async def test_unknown_end_proof_is_unavailable_evidence_but_cleanup_continues(self):
        await self.enter('split')
        self.engine.steps += 1
        self.engine.unknown += 1
        receipt, cmd, check = await self._action_lifecycle('pause_end', 'split')
        ok, reasons, severity, _ = driver.validate_receipt(receipt, cmd, check)
        self.assertFalse(ok)
        self.assertTrue(any('preemption-proof-unavailable' in reason
                            for reason in reasons))
        self.assertFalse(receipt['interval']['valid'])
        self.assertTrue(receipt['interval']['available'])
        self.assertTrue(receipt['lifecycle_proven'])
        # Measurement invalidity is retained, but the session must not
        # quarantine a proven-safe route or refuse the cleanup commands.
        self.assertFalse(self.session.failed)
        self.assertFalse(self.session.controller.quarantined)
        left, _, _ = await self._action_lifecycle('leave_full', 'full')
        self.assertEqual(left['route_after'], 'full')
        await self.action('resume', 'full')
        self.assertEqual(self.engine.route, 'full')
        self.assertFalse(self.engine.paused)

    async def test_missing_counter_not_default_zero(self):
        await self.enter('full')
        snapshot = self.session.before
        after = copy.deepcopy(snapshot)
        del after['ranks'][0]['counters']['batch_observations']['preempted_requests']
        result = interval(snapshot, after)
        self.assertFalse(result['available'])
        self.assertFalse(result['valid'])
        self.assertIn('interval-counter-unavailable-or-regressed', result['reasons'])
        self.assertIn('preempted_requests:0', result['unavailable'])
        # Only the counter that disappeared becomes unknown; the observed step
        # count is a separate actual value and must not be discarded or zeroed.
        self.assertEqual(result['counters_after'][0]['decode_steps'], 10)
        self.assertIsNone(
            result['counters_after'][0]['counters']['batch_observations'].get(
                'preempted_requests'))

    async def test_receipt_core_hash_round_trips_over_retained_bytes(self):
        from probe_session import receipt_core
        await self.action('prepare', 'full')
        self.engine.unfinished = 8
        await self.action('pause_full', 'full')
        await self.action('resume', 'full')
        self.engine.steps += 5
        receipt = await self.action('pause_end', 'full')
        session_core = receipt_core(receipt)
        driver_core = driver._receipt_core(receipt)
        self.assertEqual(session_core, driver_core)
        retained = driver._sanitize_receipt(receipt)
        # The separately retained canonical core binds narrative bytes by
        # digest; display redaction does not discard chain evidence.
        recomputed = json.loads(json.dumps(driver_core))
        self.assertEqual(recomputed, session_core)
        self.assertEqual(driver._sha256_json(recomputed),
                         self.session.prior['receipt_sha256'])
        self.assertEqual(driver._sha256_json(retained),
                         driver._sha256_json(driver._sanitize_receipt(receipt)))
        # Boundary, ingress and per-rank interval metadata survive sanitation.
        self.assertIn('observed_clock_ns', retained)
        self.assertIn('clock_basis', retained)
        self.assertEqual(retained['ingress']['sealed'], True)
        self.assertEqual(len(retained['barrier_replies']), 4)
        self.assertIn('counters_before', retained['interval'])
        counters = retained['interval']['counters_before'][0]['counters']
        self.assertIn('preempted_requests', counters['batch_observations'])
        self.assertEqual(counters['batch_observations']['last']['sequence'], 10)

    async def test_timeout_with_unknown_remote_completion_quarantines(self):
        import asyncio
        # The admission seam can only be claimed once, so shorten the existing
        # controller instead of wrapping the engine a second time.
        self.session.controller.timeout_s = 0.05
        await self.action('prepare', 'full')
        self.engine.unfinished = 8
        self.engine.pause_gate = asyncio.Event()
        cmd = driver._next_command(self.ctx, 'pause_full', 'full')
        with self.assertRaisesRegex(ProbeControlError, 'TimeoutError'):
            await self.session.execute(cmd)
        self.assertTrue(self.session.failed)
        self.assertTrue(self.session.controller.quarantined)
        self.assertFalse(any(call[0] == 'resume' for call in self.engine.calls))
        self.assertEqual(self.engine.route, 'full')
        with self.assertRaisesRegex(ProbeControlError, 'quarantined'):
            await self.session.execute(driver._next_command(
                self.ctx, 'leave_full', 'full'))

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
        # Positive invalidation is a failed measurement, not a route failure:
        # the proven lifecycle still allows the required cleanup commands.
        self.assertTrue(receipt['lifecycle_proven'])
        self.assertFalse(self.session.failed)
        # This test sent pause_end directly, so advance the local chain exactly
        # as the client does before the next command.
        self.ctx['command_seq'] = cmd['command_seq']
        self.ctx['prior_receipt'] = dict(
            command_seq=cmd['command_seq'],
            receipt_sha256=driver._sha256_json(driver._receipt_core(receipt)))
        left, _, _ = await self._action_lifecycle('leave_full', 'full')
        self.assertEqual(left['route_after'], 'full')
        await self.action('resume', 'full')
        self.assertFalse(self.engine.paused)

    async def test_positive_preemption_survives_missing_unrelated_counter(self):
        await self.enter('split')
        self.engine.steps += 4
        self.engine.preempted += 1
        # Every rank proves one preemption; an unrelated dispatch counter is
        # absent from the end snapshot.  The known invalidation must survive.
        self.engine.drop_status_keys = ('non_full_dispatches',)
        receipt, cmd, check = await self._action_lifecycle('pause_end', 'split')
        ok, reasons, severity, _ = driver.validate_receipt(receipt, cmd, check)
        self.assertFalse(ok)
        self.assertEqual(severity, 'contract')
        self.assertTrue(any('interval_invalidated:preempted' in reason
                            for reason in reasons), reasons)
        self.assertTrue(any('interval_invalidation_incomplete:unsupported' in reason
                            for reason in reasons), reasons)
        sample = receipt['interval']
        self.assertEqual(sample['invalidation']['preempted'], 4)
        self.assertEqual(sample['invalidation_complete']['preempted'], True)
        self.assertEqual(sample['invalidation_complete']['unsupported'], False)
        self.assertEqual(sample['invalidation_lower_bounds'], ['unsupported'])
        self.assertFalse(sample['available'])
        self.assertFalse(sample['valid'])
        # A proved lifecycle still permits the legal cleanup commands.
        self.assertTrue(receipt['lifecycle_proven'])
        self.assertFalse(self.session.failed)
        left, _, _ = await self._action_lifecycle('leave_full', 'full')
        self.assertEqual(left['route_after'], 'full')
        await self.action('resume', 'full')
        self.assertEqual(self.engine.route, 'full')
        self.assertFalse(self.engine.paused)

    async def test_known_part_of_unsupported_survives_missing_sibling(self):
        await self.enter('split')
        self.engine.steps += 4
        self.engine.empty_batches += 1
        self.engine.drop_status_keys = ('non_full_dispatches',)
        receipt, cmd, check = await self._action_lifecycle('pause_end', 'split')
        ok, reasons, severity, _ = driver.validate_receipt(receipt, cmd, check)
        self.assertFalse(ok)
        self.assertEqual(severity, 'contract')
        self.assertEqual(receipt['interval']['invalidation']['unsupported'], 4)
        self.assertFalse(receipt['interval']['invalidation_complete']['unsupported'])
        self.assertIn('contract:interval_invalidated:unsupported', reasons)
        await self._action_lifecycle('leave_full', 'full')
        await self.action('resume', 'full')
        self.assertFalse(self.engine.paused)

    async def test_narrative_core_binds_original_bytes_without_payload(self):
        from probe_session import receipt_core
        import hashlib
        for value in ('private-fixture-' + 'x' * 600, '   ', None, 7):
            receipt = {'clock_basis': value,
                       'decode_eligibility': {'reason': value},
                       'interval': {'observer_basis': value, 'invalidation_basis': value},
                       'ingress': {'basis': value}}
            core = receipt_core(receipt)
            self.assertEqual(core, driver._receipt_core(receipt))
            self.assertEqual(core, json.loads(json.dumps(core)))
            if type(value) is str:
                self.assertEqual(core['clock_basis']['sha256'],
                                 hashlib.sha256(value.encode()).hexdigest())
                self.assertEqual(core['clock_basis']['utf8_bytes'], len(value.encode()))
                self.assertEqual(core['clock_basis']['nonblank'], bool(value.strip()))
                self.assertNotIn('private-fixture', json.dumps(core))
            else:
                self.assertEqual(core['clock_basis']['kind'],
                                 'null' if value is None else 'invalid_type')

    async def test_positive_nondecode_survives_regressed_unrelated_counter(self):
        # The resume snapshot observes 5 empty batches; the end snapshot sees
        # 0 (a regression) while a real non-decode batch was also observed.
        self.engine.empty_batches = 5
        await self.enter('split')
        self.engine.empty_batches = 0
        self.engine.steps += 4
        self.engine.non_decode += 1
        receipt, cmd, check = await self._action_lifecycle('pause_end', 'split')
        ok, reasons, severity, _ = driver.validate_receipt(receipt, cmd, check)
        self.assertFalse(ok)
        self.assertEqual(severity, 'contract')
        self.assertTrue(any('interval_invalidated:prefill' in reason
                            for reason in reasons), reasons)
        self.assertTrue(any('interval_invalidation_incomplete:unsupported' in reason
                            for reason in reasons), reasons)
        sample = receipt['interval']
        self.assertEqual(sample['invalidation']['prefill'], 4)
        self.assertEqual(sample['invalidation']['mixed'], 4)
        self.assertEqual(sample['invalidation_complete']['prefill'], True)
        self.assertEqual(sample['invalidation_complete']['unsupported'], False)
        self.assertTrue(any('empty_batches' in item for item in sample['unavailable']))
        self.assertFalse(sample['available'])
        self.assertTrue(receipt['lifecycle_proven'])
        self.assertFalse(self.session.failed)
        left, _, _ = await self._action_lifecycle('leave_full', 'full')
        self.assertEqual(left['route_after'], 'full')
        await self.action('resume', 'full')
        self.assertFalse(self.engine.paused)

    async def test_missing_only_evidence_is_inconclusive_and_cleanup_continues(self):
        await self.enter('split')
        self.engine.steps += 4
        self.engine.drop_status_keys = ('empty_batches',)
        receipt, cmd, check = await self._action_lifecycle('pause_end', 'split')
        ok, reasons, severity, _ = driver.validate_receipt(receipt, cmd, check)
        self.assertFalse(ok)
        # Missing-only evidence is INCONCLUSIVE, not a positive invalidation.
        self.assertEqual(severity, 'inconclusive')
        self.assertFalse(any('interval_invalidated' in reason for reason in reasons))
        self.assertTrue(any('interval_evidence_unavailable' in reason
                            for reason in reasons), reasons)
        sample = receipt['interval']
        self.assertEqual(sample['invalidation'],
                         {'prefill': 0, 'mixed': 0, 'unsupported': 0, 'preempted': 0})
        self.assertEqual(sample['invalidation_complete']['unsupported'], False)
        self.assertTrue(receipt['lifecycle_proven'])
        self.assertFalse(self.session.failed)
        await self._action_lifecycle('leave_full', 'full')
        await self.action('resume', 'full')
        self.assertFalse(self.engine.paused)

    async def test_old_cohort_cannot_overlap_new_prepare(self):
        await self.action('prepare', 'full')
        self.engine.unfinished = 8
        cmd = driver._next_command(dict(self.ctx, repeat=1,cohort_id='cohort-1'), 'prepare','full')
        with self.assertRaisesRegex(ProbeControlError, 'repeat-boundary'):
            await self.session.execute(cmd)



class DropEndCounterEngine(CountedEngine):
    """Counters disappear from the end snapshot: evidence failure, not route loss."""

    def __init__(self):
        super().__init__()
        self.drop_end = False

    async def collective_rpc(self, *args, **kwargs):
        rows = await super().collective_rpc(*args, **kwargs)
        if self.drop_end and kwargs['args'][0]['action'] == 'status':
            for row in rows:
                row['counters']['batch_observations'].pop('decode_batches', None)
        return rows


class DriverCompositionTests(unittest.TestCase):
    def test_measurement_invalid_end_still_proves_cleanup_and_next_repeat(self):
        self._measurement_invalid_end_cleanup()

    def test_eight_stream_cohort_survives_bounded_slow_socket_accept(self):
        import time
        from unittest.mock import patch
        from test_e3_w1_decode_driver import ThreadingHTTPServer
        original = ThreadingHTTPServer.process_request
        def delayed(server, *args, **kwargs):
            time.sleep(0.025)
            return original(server, *args, **kwargs)
        with patch.object(ThreadingHTTPServer, 'process_request', delayed):
            self._measurement_invalid_end_cleanup()

    def _measurement_invalid_end_cleanup(self):
        import asyncio
        import json
        from pathlib import Path
        import tempfile
        from test_e3_w1_decode_driver import BASE_MANIFEST, FakeConfig, FakeServer
        engine = DropEndCounterEngine()
        loop = asyncio.new_event_loop()
        async def create(): return ProbeSession(ProbeController(engine, epoch=71))
        session = loop.run_until_complete(create())
        seen = []
        ended = [False]
        class Adapter:
            def execute(self, command, timeout_s):
                seen.append(command['action'])
                engine.unfinished = 0 if command['action'] == 'prepare' else 8
                if command['action'] == 'pause_end':
                    engine.steps += 5
                    engine.drop_end = not ended[0]
                try:
                    receipt = loop.run_until_complete(session.execute(command))
                finally:
                    engine.drop_end = False
                if command['action'] == 'pause_end':
                    ended[0] = True
                return receipt, {'kind':'actual-session/fake-AsyncLLM', 'rc':0}
        server = FakeServer(FakeConfig(streams=8))
        try:
            with tempfile.TemporaryDirectory() as folder:
                manifest = copy.deepcopy(BASE_MANIFEST)
                manifest['streams'] = 8
                manifest['repeats'] = 2
                manifest['route_plan'] = ['full', 'split']
                path = Path(folder)/'manifest.json'; path.write_text(json.dumps(manifest))
                out = Path(folder)/'result.json'
                cfg = driver.config_from_env(
                    {'WORKLOAD':str(path),'RUN_ID':'composed-driver'},str(out))
                result = driver.run(cfg,server.url,'fake-model',Adapter())
                first = result['repeats'][0]
                self.assertIn('measurement', first, {key: first.get(key)
                    for key in ('phase', 'reasons', 'open')})
                self.assertFalse(first['measurement']['valid'])
                self.assertTrue(first['measurement']['lifecycle_ok'])
                self.assertTrue(first['cleanup']['attempted'])
                self.assertTrue(first['cleanup']['restore_before_drain'])
                self.assertFalse(first['abort_run'])
                self.assertTrue(any('interval-counter-unavailable' in reason
                                    for reason in first['reasons']))
                self.assertFalse(first['valid'])
                # The proven cleanup allowed the next required repeat to run.
                self.assertEqual(result['summary']['totals']['attempted'], 2)
                self.assertTrue(result['repeats'][1]['valid'])
                self.assertEqual(result['summary']['totals']['valid'], 1)
                self.assertEqual(result['summary']['verdict'], 'INCONCLUSIVE')
                self.assertEqual(engine.route,'full')
                self.assertIn('leave_full', seen)
                self.assertEqual(seen.count('pause_end'), 2)
        finally:
            server.stop(); loop.close()

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
