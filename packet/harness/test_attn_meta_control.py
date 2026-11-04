#!/usr/bin/env python3
"""attn-meta per-arm switch through the typed native control (lead decision A, conditions 2 and 3; CPU only).

Path under test, all real code: NativeControlClient.qualification('attn_meta_select' | 'attn_meta_read', payload) ->
the engine driver's qualification intent (``_admit_control``: the attn-meta select is admitted only paused idle FULL) ->
the ordered worker RPC ``flash_s5_native_overlap`` on every rank -> native_overlap_worker.handle -> attn_meta_select
.select / .read on the rank's runner -> typed census rows with the counters in the ``qualification`` mapping. Fixture:
test_e3_native_overlap's real driver and four real bridge workers (idle_stack / active_stack, core_for, pump_until); on
each worker runner a real attn_meta_select.Selector over a counting stand-in batch (leaf: the GdnBatch itself and the
real model state are tested in the pinned image, harness/test_attn_meta_select.py and test_attn_meta_overlay.py).

Covers: exposure (switch unset: the client, the HTTP route's action list, the driver and the worker know no attn-meta
action; set: they do); select on / off / read round trip on all four ranks with per-rank counters and wrapper order;
refusals with typed reasons: engine not paused (mid-decode), native work active (a parked chunk), the split route, an
armed admission, an active auto-arm policy cycle, a recording progress window; worker-side: not installed, non-bool,
extra fields. Mutants (each must make a refusal test fail, run here through the same checks): the attn-meta admission
clause removed; the paused check skipped; the worker ignoring ``on``.
"""
import asyncio
import os
import types
import unittest
from unittest import mock

import test_e3_native_overlap as native   # first: its imports put the engine sources on the path
import attn_meta_select as S
import engine_driver as ed
import native_overlap_control as nc
import native_overlap_worker as nw

ON = {'FLASH_ATTN_META_GDN_BATCH': '1'}


class _Batch:
    """Leaf: a counting stand-in for attn_meta_batch.GdnBatch (same counters() schema)."""
    def __init__(self):
        self.batched = 0

    def __call__(self, *a, **k):
        self.batched += 1
        return 'batched'

    def counters(self):
        return dict(schema='attn-meta-gdn-batch/1', batched=self.batched, fallback={'mode': 0, 'capture': 0})


def put_selector(runner):
    mstate = types.SimpleNamespace()
    sel = S.Selector(mstate, _Batch(), lambda m, *a, **k: 'stock')
    setattr(mstate, S._MARK, sel)
    mstate.prepare_attn = sel
    runner.model_state = mstate
    return sel


class Base(unittest.IsolatedAsyncioTestCase):
    def stack(self, active=False):
        stack = native.active_stack() if active else native.idle_stack()
        core, frontend = native.core_for(stack)
        self.sels = {rank: put_selector(stack.worker(rank).model_runner) for rank in (0, 1, 2, 3)}
        return stack, core, nc.NativeControlClient(frontend, timeout_s=5.0)

    async def settle(self, core, coroutine):
        task = asyncio.get_running_loop().create_task(coroutine)
        await native.pump_until(lambda: task.done(), core)
        return task

    async def select(self, core, control, on):
        return (await self.settle(core, control.qualification('attn_meta_select', {'on': on}))).result()

    async def refused(self, core, control, text, payload=None):
        task = await self.settle(core, control.qualification('attn_meta_select', payload or {'on': True}))
        with self.assertRaisesRegex(Exception, text):
            task.result()


class Exposure(unittest.TestCase):
    def test_switch_unset_knows_no_action(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(nc.attn_meta_actions(), ())
            self.assertEqual(nw.attn_meta_actions(), ())
            client = nc.NativeControlClient.__new__(nc.NativeControlClient)
            for action in nc.ATTN_META_ACTIONS:
                with self.assertRaisesRegex(Exception, 'native-control-qualification-action-unknown'):
                    asyncio.run(client.qualification(action, {'on': True}))
            with self.assertRaisesRegex(Exception, 'native-overlap-unknown-action'):
                nw.handle(types.SimpleNamespace(), {'action': 'attn_meta_read', 'engine_epoch': 1, 'route_generation': 0})

    def test_switch_set_knows_both(self):
        with mock.patch.dict(os.environ, ON, clear=True):
            self.assertEqual(nc.attn_meta_actions(), ('attn_meta_select', 'attn_meta_read'))
            self.assertEqual(nw.attn_meta_actions(), ('attn_meta_select', 'attn_meta_read'))
        with mock.patch.dict(os.environ, {'FLASH_ATTN_META_GDN_BATCH': '0'}, clear=True):
            self.assertEqual(nc.attn_meta_actions(), ())

    def test_qualification_tuples_unchanged(self):
        self.assertEqual(nc.QUALIFICATION_ACTIONS, nw.QUALIFICATION_ACTIONS)
        self.assertNotIn('attn_meta_select', nw.QUALIFICATION_ACTIONS)

    def test_http_route_uses_the_gate(self):
        import inspect
        import probe_control
        src = inspect.getsource(probe_control)
        self.assertIn('if action in QUALIFICATION_ACTIONS or action in attn_meta_actions():', src)


class Startup(unittest.TestCase):
    """s5_startup.install_builder calls attn_meta_select.startup first (before the probe / overlap branches), only when
    the switch is set and not profiling; a refusal propagates (startup fails, never a skipped wrapper)."""

    def run_builder(self, env, is_profiling=False):
        import s5_startup
        calls = []
        runner = types.SimpleNamespace()
        with mock.patch.object(S, 'startup', lambda r, env=None: calls.append((r, dict(env)))):
            s5_startup.install_builder(runner, {}, is_profiling, env=env)
        return calls, runner

    def test_on_calls_startup_with_the_runner(self):
        calls, runner = self.run_builder({'FLASH_ATTN_META_GDN_BATCH': '1'})
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], runner)

    def test_off_and_profiling_call_nothing(self):
        for env, prof in (({}, False), ({'FLASH_ATTN_META_GDN_BATCH': '0'}, False),
                          ({'FLASH_ATTN_META_GDN_BATCH': '1'}, True)):
            with self.subTest(env=env, prof=prof):
                self.assertEqual(self.run_builder(env, prof)[0], [])

    def test_refusal_propagates(self):
        import s5_startup

        def refuse(r, env=None):
            raise S.AttnMetaSelectError('selector already installed')
        with mock.patch.object(S, 'startup', refuse), self.assertRaisesRegex(S.AttnMetaSelectError, 'already'):
            s5_startup.install_builder(types.SimpleNamespace(), {}, False, env={'FLASH_ATTN_META_GDN_BATCH': '1'})

    def test_bad_value_refuses_at_startup(self):
        import s5_startup
        with self.assertRaisesRegex(S.AttnMetaSelectError, 'must be 0 or 1'):
            s5_startup.install_builder(types.SimpleNamespace(), {}, False, env={'FLASH_ATTN_META_GDN_BATCH': 'yes'})

    def test_hook_precedes_every_other_branch(self):
        import inspect
        import s5_startup
        src = inspect.getsource(s5_startup.install_builder)
        self.assertLess(src.index('_attn_meta_startup(runner, env=env)'), src.index('probe = mode_settings(env)'))


class RoundTrip(Base):
    async def test_select_read_on_every_rank(self):
        with mock.patch.dict(os.environ, ON):
            stack, core, control = self.stack()
            reply = await self.select(core, control, True)
            rows = reply['qualification']
            self.assertEqual([r['worker_rank'] for r in rows], [0, 1, 2, 3])
            for row in rows:
                q = row['qualification']
                self.assertEqual((q['worker_action'], q['before']['selected'], q['after']['selected']),
                                 ('attn_meta_select', False, True))
                self.assertEqual(q['order'], dict(outer='Selector', selector_outer=True, wrapped_by_other=False,
                                                  ok=True))
                self.assertIs(q['engine_supplied_verdict'], False)
            for sel in self.sels.values():
                self.assertEqual(sel(1), 'batched')
                self.assertEqual(sel(1), 'batched')
            read = (await self.settle(core, control.qualification('attn_meta_read', {}))).result()
            self.assertEqual([r['qualification']['counters']['batched'] for r in read['qualification']], [2] * 4)
            off = await self.select(core, control, False)
            self.assertEqual({r['qualification']['after']['selected'] for r in off['qualification']}, {False})
            for sel in self.sels.values():
                self.assertEqual(sel(1), 'stock')
            read = (await self.settle(core, control.qualification('attn_meta_read', {}))).result()
            counters = [r['qualification']['counters'] for r in read['qualification']]
            self.assertEqual([(c['batched'], c['off_calls'], c['selections']) for c in counters], [(2, 1, 2)] * 4)
            self.assertIsNone(stack.driver.error)


class WorkerPath(Base):
    """The box's route: native_qualification_frontend.handle_worker (the reviewed WORKER_PATH adapter, action
    'qualification_worker') over the same control; identity fields in the reply; exact payloads; switch unset refuses."""

    def command(self, worker_action, payload, seq=1, epoch=None):
        return dict(schema=1, action='qualification_worker', run_id='r', command_seq=seq, client_id='c',
                    engine_epoch=epoch, worker_action=worker_action, payload=payload)

    async def test_select_and_read_through_the_adapter(self):
        import native_qualification_frontend as qf
        with mock.patch.dict(os.environ, ON):
            stack, core, control = self.stack()
            epoch = stack.driver.engine_epoch
            task = await self.settle(core, qf.handle_worker(self.command('attn_meta_select', {'on': True}, 1, epoch),
                                                            control=control))
            reply = task.result()
            self.assertEqual((reply['schema'], reply['action'], reply['run_id'], reply['command_seq'], reply['client_id'],
                              reply['engine_epoch'], reply['worker_action']),
                             (1, 'qualification_worker', 'r', 1, 'c', epoch, 'attn_meta_select'))
            rows = reply['worker_reply']['qualification']
            self.assertEqual([r['qualification']['after']['selected'] for r in rows], [True] * 4)
            task = await self.settle(core, qf.handle_worker(self.command('attn_meta_read', {}, 2, epoch), control=control))
            self.assertEqual([r['qualification']['counters']['selected'] for r in task.result()['worker_reply']['qualification']],
                             [True] * 4)

    def test_payloads_exact(self):
        import native_qualification_frontend as qf
        with mock.patch.dict(os.environ, ON):
            qf.validate_worker(self.command('attn_meta_select', {'on': False}, epoch=1))
            qf.validate_worker(self.command('attn_meta_read', {}, epoch=1))
            for action, payload, text in (('attn_meta_select', {}, 'fields-differ'),
                                          ('attn_meta_select', {'on': True, 'x': 1}, 'fields-differ'),
                                          ('attn_meta_select', {'on': 1}, 'on-not-a-bool'),
                                          ('attn_meta_read', {'on': True}, 'fields-differ')):
                with self.subTest(action=action, payload=payload), self.assertRaisesRegex(qf.NativeQualificationFrontendError, text):
                    qf.validate_worker(self.command(action, payload, epoch=1))

    def test_unknown_action_refused_with_the_switch_on(self):
        import native_qualification_frontend as qf
        with mock.patch.dict(os.environ, ON):
            for action in ('attn_meta_write', 'attn_meta', 'ATTN_META_SELECT'):
                with self.subTest(action=action), self.assertRaisesRegex(qf.NativeQualificationFrontendError, 'not-allowlisted'):
                    qf.validate_worker(self.command(action, {'on': True}, epoch=1))

    def test_mutants_caught(self):
        """Each adapter mutant must make one of this class's refusal tests fail."""
        import native_qualification_frontend as qf
        real = qf.validate_worker
        mutants = dict(
            any_attn_action=lambda c: (real(dict(c, worker_action='attn_meta_read', payload={}))
                                       if str(c.get('worker_action', '')).startswith('attn_meta') else real(c)),
            payload_unchecked=lambda c: real(dict(c, payload={'on': True} if c.get('worker_action') == 'attn_meta_select'
                                                  else {})) if c.get('worker_action') in qf.ATTN_META_PAYLOAD_FIELDS else real(c),
            switch_ignored=lambda c: (mock.patch.dict(os.environ, ON).__enter__(), real(c))[1])
        tests = ('test_unknown_action_refused_with_the_switch_on', 'test_payloads_exact', 'test_switch_unset_refuses')
        for name, mutant in mutants.items():
            with self.subTest(mutant=name):
                saved = dict(os.environ)
                try:
                    with mock.patch.object(qf, 'validate_worker', mutant):
                        results = []
                        for test in tests:
                            outcome = unittest.TestResult()
                            WorkerPath(test).run(outcome)
                            results.append(outcome.wasSuccessful())
                finally:
                    os.environ.clear()
                    os.environ.update(saved)
                self.assertIn(False, results, name + ' not caught')

    def test_switch_unset_refuses(self):
        import native_qualification_frontend as qf
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(qf.attn_meta_actions(), ())
            for action in ('attn_meta_select', 'attn_meta_read'):
                with self.assertRaisesRegex(qf.NativeQualificationFrontendError, 'not-allowlisted'):
                    qf.validate_worker(self.command(action, {'on': True} if action == 'attn_meta_select' else {}, epoch=1))
            self.assertNotIn('attn_meta_select', qf.WORKER_ACTIONS)


class Refusals(Base):
    """Admission is synchronous in the driver (submit_control): the client always pauses first, so the driver's own
    rule is what a mid-decode or native-active switch meets."""

    def admit(self, stack, on=True):
        return stack.driver.submit_control({'action': 'qualification', 'worker_action': 'attn_meta_select',
                                            'payload': {'on': on}})

    def refuse(self, stack, text):
        with self.assertRaisesRegex(ed.E3ContractError, text):   # DriverError, or the policy's refusal (its base)
            self.admit(stack)
        self.assertEqual({s.selected for s in self.sels.values()}, {False})

    def scenarios(self):
        import native_auto_arm as auto

        def policy(stack, core):
            stack.driver.auto_arm = auto.AutoArmPolicy(core, stack.driver)
            stack.driver.auto_arm.phase = 'native'

        def recording(stack, core):
            stack.driver.auto_arm = auto.AutoArmPolicy(core, stack.driver)
            stack.driver.install_progress('burst')
            stack.driver.progress.recording = True
        return (
            ('mid-decode', False, lambda stack, core: setattr(core, 'paused', False),
             'native-control-requires-paused-engine'),
            ('native-active', True, None,
             'native-control-route-exchange-outstanding|native-control-requires-idle-driver'),
            ('split', False, lambda stack, core: setattr(stack.driver, '_split_ready', True),
             'native-attn-meta-requires-paused-idle-full'),
            ('admission', False, lambda stack, core: setattr(stack.driver, '_admission_intent',
                                                              {'origin': 'manual', 'seq': 1}),
             'native-attn-meta-requires-paused-idle-full'),
            ('policy-cycle', False, policy, 'cycle-owned|native-attn-meta-requires-paused-idle-full'),
            ('recording', False, recording, 'native-attn-meta-requires-paused-idle-full'),
        )

    async def test_refused_unless_paused_idle_full(self):
        with mock.patch.dict(os.environ, ON):
            for name, active, apply, text in self.scenarios():
                with self.subTest(name=name):
                    stack, core, control = self.stack(active=active)
                    if apply is not None:
                        apply(stack, core)
                    self.refuse(stack, text)

    async def test_admitted_paused_idle_full(self):
        with mock.patch.dict(os.environ, ON):
            stack, core, control = self.stack()
            future = self.admit(stack)
            await native.pump_until(lambda: future.done(), core)
            self.assertEqual({r['qualification']['after']['selected'] for r in future.result()['qualification']},
                             {True})

    async def test_read_is_admitted_while_paused(self):
        with mock.patch.dict(os.environ, ON):
            stack, core, control = self.stack()
            stack.driver._admission_intent = {'origin': 'manual', 'seq': 1}
            read = (await self.settle(core, control.qualification('attn_meta_read', {}))).result()
            self.assertEqual(len(read['qualification']), 4)

    async def test_worker_refusals(self):
        with mock.patch.dict(os.environ, ON):
            stack, core, control = self.stack()
            await self.refused(core, control, 'native-attn-meta-select-not-bool', {'on': 1})
            await self.refused(core, control, 'native-attn-meta-fields-differ', {'on': True, 'extra': 1})
            del stack.worker(2).model_runner.model_state
            await self.refused(core, control, 'native-attn-meta-not-installed')


class Mutants(Base):
    """Each mutant must make the matching refusal check fail (the check is run and its failure observed)."""

    async def check(self, apply, scenario):
        try:
            await scenario()
        except AssertionError:
            return True
        return False

    async def test_mutants_caught(self):
        caught = {}
        real_admit = ed.NativeSegmentDriver._admit_control

        def no_attn_clause(self, action, command):
            if action == 'qualification' and command.get('worker_action') == 'attn_meta_select':
                if self.live or self._inflight:
                    raise ed.DriverError('native-control-requires-idle-driver')
                if not self._engine_paused():
                    raise ed.DriverError('native-control-requires-paused-engine')
                return None
            return real_admit(self, action, command)

        def no_paused_check(self, action, command):
            if action == 'qualification' and command.get('worker_action') == 'attn_meta_select':
                return None
            return real_admit(self, action, command)

        def driver_refuses(stack, text):
            with self.assertRaisesRegex(ed.DriverError, text):
                stack.driver.submit_control({'action': 'qualification', 'worker_action': 'attn_meta_select',
                                             'payload': {'on': True}})

        async def admission_scenario():
            stack, core, control = self.stack()
            stack.driver._admission_intent = {'origin': 'manual', 'seq': 1}
            driver_refuses(stack, 'native-attn-meta-requires-paused-idle-full')

        async def paused_scenario():
            stack, core, control = self.stack()
            core.paused = False
            driver_refuses(stack, 'native-control-requires-paused-engine')

        with mock.patch.dict(os.environ, ON):
            with mock.patch.object(ed.NativeSegmentDriver, '_admit_control', no_attn_clause):
                caught['admission-clause-removed'] = await self.check(None, admission_scenario)
            with mock.patch.object(ed.NativeSegmentDriver, '_admit_control', no_paused_check):
                caught['paused-check-skipped'] = await self.check(None, paused_scenario)
            real_select = S.select

            def ignores_on(runner, on):
                return real_select(runner, False)

            async def roundtrip():
                stack, core, control = self.stack()
                reply = await self.select(core, control, True)
                self.assertEqual({r['qualification']['after']['selected'] for r in reply['qualification']}, {True})
            with mock.patch.object(S, 'select', ignores_on):
                caught['worker-ignores-on'] = await self.check(None, roundtrip)
        self.assertEqual(caught, dict.fromkeys(caught, True), caught)


if __name__ == '__main__':
    unittest.main()
