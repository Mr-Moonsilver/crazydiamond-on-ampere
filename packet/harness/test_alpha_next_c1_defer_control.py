#!/usr/bin/env python3
"""alpha-next-c1: the per-arm deferred bind and the fail-closed C1 rebind switches (lead 2 Nov 21:0x CET).

The C1 rebind (claude/c1-rebind 70b0896d) reads FLASH_S5_STAGE_OVERLAP_DEFER_BIND once, at driver construction. One engine
incarnation per queue alternates C1 arms and C1+defer arms, so here the startup flag only installs the deferred bind (selected,
as the rebind has it) and the typed per-arm field ``defer_bind`` of ``native_stage_local`` selects it, as fused QSA and layer
groups are selected:

* accepted only paused idle FULL (C0's rule): refused mid-chunk, with a retained row, and never accepted while a PP send or
  receive ticket is live (swept over every tick of a connected run);
* refused where not installed, refused with overlap false or absent, refused when not a bool, nothing applied on a refusal;
* the reply acknowledges the selected value; the router session refuses the field where the flag is not installed and the
  control client refuses an unacknowledged value;
* s5_startup refuses DEFER_BIND=1 or OBSERVE=1 without FLASH_S5_STAGE_OVERLAP=1 (before, they were ignored).

Connected switch (``Switch``): one ring incarnation (test_e3_stage_overlap.OverlapStack: real controller, OverlapSegmentDriver,
NativeWorkerBridge/LaneDispatcher workers, PP channel) serves four requests with defer off, on, off, on selected between them
through the driver's real control path. After every arm the per-rank ledgers and persistent state equal a serial stack's over the
same requests; at every switch no frame, binding, retained row, retained send or live PP ticket is left; in each arm stage 1's
STATE_FINALIZE of the arm's first chunk is held genuinely incomplete and the rebind's question is asked: BLOCKED in the off arms,
OVERLAPS in the on arms, of the one incarnation.

Mutants (``Mutants``): the idle rule skipped, the install check removed, the overlap check removed, the startup guard removed,
the session refusal removed; each must fail a test here. Leaves: alpha_next_c1_engine_fixture_leaves.json (paused-engine,
second-request, installed-by-env, held-finalize) beside C1's, the rebind's and C0's control leaves.
"""
import inspect
import os
import textwrap
import types
import unittest
from unittest import mock

import test_e3_stage_overlap as t  # first: its imports put the engine sources on the path
import test_e3_admission_native as adm
import test_e3_stage_local_control as c0ctl
from test_e3_stage_local_control import attach, process_env
import engine_driver as ed
import native_overlap_session as nos
import s5_startup
import scheduler_hooks as sh
import stage_overlap as so
import stage_progress as sp
from stage_overlap import OverlapSegmentDriver, StagedAdmissionController
from stage_wire import retained_sends

OVERLAP_ENV = {'FLASH_S5_STAGE_LOCAL': '1', 'FLASH_S5_STAGE_OVERLAP': '1'}
DEFER_ENV = dict(OVERLAP_ENV, FLASH_S5_STAGE_OVERLAP_DEFER_BIND='1')
IDLE_REFUSAL = 'native-stage-local-requires-paused-idle-full'


def live_tickets(network):
    """PP tickets not yet retired, on every rank's channel (sender and receiver)."""
    return {rank: [ticket for ticket in getattr(worker.channel, 'tickets', ()) if ticket.state != 'retired']
            for rank, worker in network.workers.items()}


class Fixture(t.OverlapBase):
    def network(self, installed=True, **kwargs):
        # Leaf installed-by-env: the driver reads the startup flags from the process environment at construction.
        with mock.patch.dict(os.environ, DEFER_ENV if installed else OVERLAP_ENV):
            os.environ.pop('FLASH_S5_STAGE_OVERLAP_DEFER_BIND', None) if not installed else None
            return super().network(**kwargs)

    def control(self, network, **fields):
        """The driver's real control path. Leaf paused-engine: the scheduler reads paused for this call only."""
        driver = network.driver
        engine = driver.engine
        driver.engine = types.SimpleNamespace(is_scheduler_paused=lambda: True)
        try:
            return driver._submit_control(dict(action='native_stage_local', **fields))
        finally:
            driver.engine = engine

    def selection(self, driver):
        return (driver.stage_selected, driver.overlap_selected, driver.defer_bind, driver.stage_counters['selections'])

    def add_request(self, network, uid):
        """Leaf second-request: the next request of the same incarnation, queued as the scheduler would queue it."""
        request = adm.Request(uid=uid, tokens=network.tokens)
        request.num_in_flight_tokens = 0
        request.num_prompt_tokens = network.tokens
        network.scheduler.waiting.prepend_request(request)
        network.expected_commits = network.controller.commits + (network.tokens - 1 + network.chunk - 1) // network.chunk

    def idle_state(self, network):
        driver = network.driver
        return dict(frames=len(driver._frames), tx=driver.tx is not None, retained=driver._retained is not None,
                    inflight=len(driver._inflight), bindings=dict(getattr(driver, '_bindings', {}) or {}),
                    pending_binding=getattr(driver, '_pending_binding', None) is not None,
                    retained_sends={rank: retained_sends(worker._flash_s5_bridge) for rank, worker in network.workers.items()},
                    live_tickets={rank: len(v) for rank, v in live_tickets(network).items()})

    def assert_idle(self, network):
        state = self.idle_state(network)
        self.assertEqual(state, dict(frames=0, tx=False, retained=False, inflight=0, bindings={}, pending_binding=False,
                                     retained_sends={rank: {} for rank in network.workers},
                                     live_tickets={rank: 0 for rank in network.workers}))


class Install(Fixture):
    def network(self, installed=True, **kwargs):
        """C1's Control.idle_full: the cold stack with its admission controller read idle (no request admitted)."""
        network = super().network(installed=installed, **kwargs)
        network.driver.controller = types.SimpleNamespace(active=False, overlap_counters={}, overlap_refusals={})
        return network

    def test_installed_by_the_startup_flag_and_selected_as_the_rebind_has_it(self):
        on, off = self.network(), self.network(installed=False)
        self.assertEqual((on.driver.defer_bind_installed, on.driver.defer_bind), (True, True))
        self.assertEqual((off.driver.defer_bind_installed, off.driver.defer_bind), (False, False))
        self.assertEqual(on.driver.stage_status()['overlap']['defer_bind']['installed'], True)
        self.assertNotIn('defer_bind', off.driver.stage_status()['overlap'])

    def test_selection_its_previous_and_its_acknowledgement(self):
        network = self.network()
        driver = network.driver
        reply = self.control(network, local=True, overlap=True, defer_bind=False)
        self.assertEqual((reply['overlap'], reply['defer_bind'], reply['previous_defer_bind']), (True, False, True))
        self.assertIs(driver.defer_bind, False)
        self.assertIs(driver.stage_status()['overlap']['defer_bind']['selected'], False)
        reply = self.control(network, local=True, overlap=True)        # without the field: keeps the choice
        self.assertEqual((reply['defer_bind'], reply['previous_defer_bind']), (False, False))
        reply = self.control(network, local=True, overlap=True, defer_bind=True)
        self.assertEqual((reply['defer_bind'], reply['previous_defer_bind']), (True, False))
        reply = self.control(network, local=True, overlap=False)       # no overlap selects no deferred bind
        self.assertEqual((reply['overlap'], reply['defer_bind'], reply['previous_defer_bind']), (False, False, True))
        reply = self.control(network, local=False)
        self.assertEqual((reply['local'], reply['overlap'], reply['defer_bind']), (False, False, False))

    def test_refused_where_not_installed(self):
        network = self.network(installed=False)
        before = self.selection(network.driver)
        for value in (True, False):
            with self.subTest(value=value), self.assertRaises(ed.DriverError) as caught:
                self.control(network, local=True, overlap=True, defer_bind=value)
            self.assertIn('native-stage-overlap-defer-bind-not-installed', str(caught.exception))
        self.assertEqual(self.selection(network.driver), before)
        reply = self.control(network, local=True, overlap=True)        # C1's command, unchanged, has no such field
        self.assertNotIn('defer_bind', reply)

    def test_refused_with_overlap_false_or_absent_and_nothing_applied(self):
        network = self.network()
        self.control(network, local=True, overlap=True, defer_bind=False)
        before = self.selection(network.driver)
        for bad in (dict(local=True, overlap=False, defer_bind=True), dict(local=True, defer_bind=True),
                    dict(local=False, defer_bind=True)):
            with self.subTest(bad=bad), self.assertRaises(ed.DriverError) as caught:
                self.control(network, **bad)
            self.assertIn('native-stage-overlap-defer-bind-needs-overlap', str(caught.exception))
            self.assertEqual(self.selection(network.driver), before)

    def test_only_a_bool(self):
        network = self.network()
        before = self.selection(network.driver)
        for bad in (1, 'true', None, 0):
            with self.subTest(bad=bad), self.assertRaises(ed.DriverError) as caught:
                self.control(network, local=True, overlap=True, defer_bind=bad)
            self.assertIn('native-stage-overlap-defer-bind-invalid', str(caught.exception))
        self.assertEqual(self.selection(network.driver), before)


class Idle(Fixture):
    def idle_network(self):
        network = self.network()
        network.driver.controller = types.SimpleNamespace(active=False, overlap_counters={}, overlap_refusals={})   # C1's Control.idle_full
        return network

    def refused_with(self, network, **attrs):
        driver = network.driver
        saved = {name: getattr(driver, name) for name in attrs}
        before = self.selection(driver)
        try:
            for name, value in attrs.items():
                setattr(driver, name, value)
            with self.assertRaises(ed.DriverError) as caught:
                self.control(network, local=True, overlap=True, defer_bind=not driver.defer_bind)
            self.assertIn(IDLE_REFUSAL, str(caught.exception))
        finally:
            for name, value in saved.items():
                setattr(driver, name, value)
        self.assertEqual(self.selection(driver), before)

    def test_refused_mid_transaction_and_with_a_retained_row(self):
        network = self.idle_network()
        self.refused_with(network, tx=object())
        self.refused_with(network, _retained={'transaction': object()})
        self.refused_with(network, _frames=[object()])
        self.refused_with(network, _inflight={1: object()})

    def test_never_accepted_while_a_chunk_or_a_pp_ticket_is_live(self):
        """At every tick of a slow-stage-1 run, try to flip the selection: every acceptance is at a fully idle point (no
        frame, row, binding, retained send or live PP ticket); every refusal is the idle rule; a ticket was live meanwhile."""
        network = self.network()
        driver = network.driver
        seen = dict(accepted=0, refused=0, ticket_live_ticks=0)

        def probe():
            live = any(live_tickets(network).values())
            seen['ticket_live_ticks'] += int(live)
            state = self.idle_state(network)
            current = driver.defer_bind
            try:
                self.control(network, local=True, overlap=True, defer_bind=not current)
            except ed.DriverError as exc:
                self.assertIn(IDLE_REFUSAL, str(exc))
                seen['refused'] += 1
            else:
                seen['accepted'] += 1
                self.assertFalse(live)
                self.assertEqual((state['frames'], state['tx'], state['retained'], state['inflight']), (0, False, False, 0))
                self.control(network, local=True, overlap=True, defer_bind=current)   # put it back
            return network.controller.commits >= network.expected_commits and not driver.live
        self.stage_one_held(network, probe, limit=4000)
        self.assertIsNone(driver.error)
        self.assertEqual(network.controller.commits, network.expected_commits)
        self.assertGreater(seen['refused'], 0)
        self.assertGreater(seen['ticket_live_ticks'], 0)
        self.assert_idle(network)
        self.control(network, local=True, overlap=True, defer_bind=False)   # accepted once idle


class Switch(Fixture):
    """off -> on -> off -> on in one incarnation; see the module docstring."""
    HOLD = 300

    def held_arm(self, network, defer):
        """Hold the arm's first chunk's stage-1 STATE_FINALIZE incomplete (leaf held-finalize); ask the rebind's question."""
        driver = network.driver
        first = None
        held = []

        def complete_but_held():
            for worker in network.workers.values():
                for event in worker.bus.events:
                    event.complete = not any(event is h for h in held)

        def ops(rank, txn):
            return {e[1] for e in network.ledgers[rank] if e[0] == 'op' and e[2] == txn}

        def tick():
            network.admit_next()
            complete_but_held()
            network.pump()
            out = driver.advance()
            network.note_peak()
            return out
        for _ in range(4000):
            tick()
            self.assertIsNone(driver.error)
            if first is None and driver._frames and driver._frames[0].chunk['tx'] is not None:
                first = driver._frames[0].chunk['tx'].txn_seq
            if first is not None:
                for rank in (2, 3):
                    dispatcher = network.workers[rank].dispatcher
                    if dispatcher.transaction is not None and dispatcher.transaction.txn_seq == first:
                        held += [r.event for r in dispatcher.events if r.kind == 'state_finalize'
                                 and not any(r.event is h for h in held)]
                if len(held) == 2:
                    break
        self.assertEqual(len(held), 2)
        for event in held:
            event.complete = False
        for _ in range(self.HOLD):
            tick()
            self.assertIsNone(driver.error)
        self.assertTrue(all(not event.query() for event in held))
        successor_ops = ops(0, first + 1) | ops(1, first + 1)
        verdict = 'OVERLAPS' if 'layer' in successor_ops else 'BLOCKED'
        del held[:]
        network.run_request(tick=tick)
        return verdict, first

    def test_off_on_off_on_in_one_incarnation(self):
        on = self.network()
        off = self.serial()
        driver = on.driver
        verdicts, deferred = [], []
        # The stack's first request is admitted at construction, before any control can be accepted (C0's idle rule): it
        # runs at the installed default (selected), then the four arms off, on, off, on follow in the same incarnation.
        self.assertIs(driver.defer_bind, True)
        on.run_request()
        off.run_request()
        for rank in (0, 1, 2, 3):
            self.assertEqual(on.ledgers[rank], off.ledgers[rank], rank)
        for index, defer in enumerate((False, True, False, True)):
            with self.subTest(arm=index, defer=defer):
                self.add_request(on, 'req-arm-%d' % index)
                self.add_request(off, 'req-arm-%d' % index)
                self.assert_idle(on)
                reply = self.control(on, local=True, overlap=True, defer_bind=defer)
                self.assertIs(reply['defer_bind'], defer)
                self.assertIs(driver.stage_status()['overlap']['defer_bind']['selected'], defer)
                opens = driver.defer_counters['deferred_opens']
                verdict, _first = self.held_arm(on, defer)
                off.run_request()
                verdicts.append(verdict)
                deferred.append(driver.defer_counters['deferred_opens'] - opens)
                self.assertEqual(on.controller.commits, off.controller.commits)
                self.assertEqual(on.worker_errors, [])
                for rank in (0, 1, 2, 3):
                    self.assertEqual(on.ledgers[rank], off.ledgers[rank], (index, rank))
                self.assertEqual(on.persistent_state(), off.persistent_state())
                self.assert_idle(on)
                self.assertLessEqual(max(on.inflight_peak.values()), 1)
        self.assertEqual(verdicts, ['BLOCKED', 'OVERLAPS', 'BLOCKED', 'OVERLAPS'])
        self.assertEqual([n > 0 for n in deferred], [False, True, False, True])


class Startup(unittest.TestCase):
    def refused(self, env, text):
        # Leaf (test_e3_native_layer_groups.Startup's): _enabled reads the composition's settings; these guards come first.
        with mock.patch.object(s5_startup, '_enabled', lambda env: True), \
                self.assertRaises(s5_startup.StartupError) as refused:
            s5_startup.install_engine(types.SimpleNamespace(), env=env)
        self.assertIn(text, str(refused.exception))

    def test_rebind_switches_need_the_overlap_switch(self):
        for name in ('FLASH_S5_STAGE_OVERLAP_DEFER_BIND', 'FLASH_S5_STAGE_OVERLAP_OBSERVE'):
            with self.subTest(name=name):
                self.refused({name: '1'}, name + ' needs FLASH_S5_STAGE_OVERLAP=1')
                self.refused({name: '1', 'FLASH_S5_STAGE_LOCAL': '1', 'FLASH_S5_STAGE_OVERLAP': '0'},
                             name + ' needs FLASH_S5_STAGE_OVERLAP=1')

    def test_only_a_boolean_setting(self):
        for name in ('FLASH_S5_STAGE_OVERLAP_DEFER_BIND', 'FLASH_S5_STAGE_OVERLAP_OBSERVE'):
            for value in ('2', 'yes', ''):
                with self.subTest(name=name, value=value):
                    self.refused(dict(OVERLAP_ENV, **{name: value}), 'invalid boolean setting: ' + name)


ROUTER_ALL = dict(c0ctl.BOTH, FLASH_S5_STAGE_OVERLAP='1')


class RouterBase(c0ctl.Base):
    INSTALLED = True

    async def asyncSetUp(self):
        env = dict(FLASH_S5_STAGE_LOCAL='1', FLASH_S5_STAGE_OVERLAP='1',
                   FLASH_S5_STAGE_OVERLAP_DEFER_BIND='1' if self.INSTALLED else '0')
        with mock.patch.object(ed, 'NativeSegmentDriver', OverlapSegmentDriver), \
                mock.patch.object(sh, 'PrefillAdmissionController', StagedAdmissionController), \
                mock.patch.object(c0ctl, 'StagedSegmentDriver', OverlapSegmentDriver), \
                mock.patch.dict(os.environ, env):
            await super().asyncSetUp()

    def env(self):
        return mock.patch.dict('os.environ', {'FLASH_S5_STAGE_OVERLAP': '1',
                                              'FLASH_S5_STAGE_OVERLAP_DEFER_BIND': '1' if self.INSTALLED else '0'})


class Router(RouterBase):
    async def test_field_reaches_the_driver_and_is_acknowledged(self):
        self.assertTrue(self.driver.defer_bind_installed)
        with process_env(ROUTER_ALL), self.env():
            app, endpoint = attach(self.frontend, '1', '1')
            self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
            code, reply = await self.post(endpoint, app, 'native_stage_local', local=True, overlap=True, defer_bind=False)
            self.assertEqual(code, 200, reply)
            self.assertEqual((reply['overlap'], reply['defer_bind'], reply['previous_defer_bind']), (True, False, True))
            self.assertIs(self.driver.defer_bind, False)
            code, status = await self.post(endpoint, app, 'status')
            self.assertIs(status['native_stage_local']['overlap']['defer_bind']['selected'], False)
            self.assertEqual(await self.post(endpoint, app, 'native_stage_local', local=True, overlap=True, defer_bind=1),
                             (409, {'error': 'native-control-stage-overlap-defer-bind-invalid'}))
            # The engine-side refusal (needs overlap) reaches the router as C1's refusals do.
            self.assertEqual(await self.post(endpoint, app, 'native_stage_local', local=True, overlap=False, defer_bind=True),
                             (400, {'error': 'invalid-command-json'}))
            self.assertIs(self.driver.defer_bind, False)
            code, reply = await self.post(endpoint, app, 'native_stage_local', local=True, overlap=True, defer_bind=True)
            self.assertEqual((code, reply['defer_bind']), (200, True))
            self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)

    async def test_an_unacknowledged_value_is_refused_by_the_client(self):
        with process_env(ROUTER_ALL), self.env():
            real = self.control._call

            async def drops_the_field(command, real=real):
                return await real({k: v for k, v in command.items() if k != 'defer_bind'})
            app, endpoint = attach(self.frontend, '1', '1')
            self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
            with mock.patch.object(self.control, '_call', drops_the_field), self.assertRaises(Exception) as caught:
                await self.control.native_stage_overlap(True, True, False)
            self.assertIn('native-control-stage-overlap-defer-bind-not-set', str(caught.exception))
            self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)


class RouterNotInstalled(RouterBase):
    INSTALLED = False

    async def test_field_refused_where_not_installed(self):
        self.assertFalse(self.driver.defer_bind_installed)
        with process_env(ROUTER_ALL), self.env():
            app, endpoint = attach(self.frontend, '1', '1')
            self.assertEqual((await self.post(endpoint, app, 'pause'))[0], 200)
            code, reply = await self.post(endpoint, app, 'native_stage_local', local=True, overlap=True, defer_bind=False)
            self.assertNotEqual(code, 200, reply)
            self.assertIn('native-stage-overlap-defer-bind-not-installed', str(reply))
            with self.assertRaises(Exception) as caught:
                await self.control.native_stage_overlap(True, True, True)
            self.assertIn('native-control-stage-overlap-defer-bind-not-installed', str(caught.exception))
            code, reply = await self.post(endpoint, app, 'native_stage_local', local=True, overlap=False)
            self.assertEqual(code, 200, reply)                  # C1's command without the field: unchanged
            self.assertNotIn('defer_bind', reply)
            self.assertEqual((await self.post(endpoint, app, 'resume'))[0], 200)


# ------------------------------------------------------------------------------------------------- mutants
class Mutants(unittest.TestCase):
    """Each new rule, faulted in place: the scenario named beside it must then fail. A survivor is a finding."""

    def fails(self, target, name, replacement, scenario):
        with mock.patch.object(target, name, replacement):
            result = unittest.TestResult()
            scenario.run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertTrue(result.failures or result.errors, 'mutant survived: ' + name)

    def test_idle_rule_skipped(self):
        def no_idle(self, command):
            # C0's control without its paused idle FULL rule.
            previous = self.stage_selected
            self.stage_selected = command['local']
            return {'schema': 1, 'action': sp.STAGE_CONTROL, 'local': command['local'], 'previous': previous}
        self.fails(sp.StagedSegmentDriver, '_stage_control', no_idle,
                   Idle('test_never_accepted_while_a_chunk_or_a_pp_ticket_is_live'))
        self.fails(sp.StagedSegmentDriver, '_stage_control', no_idle,
                   Idle('test_refused_mid_transaction_and_with_a_retained_row'))

    def overlap_control(self, old, new):
        source = textwrap.dedent(inspect.getsource(so.OverlapSegmentDriver._stage_control))
        assert source.count(old) == 1, old
        # zero-argument super() inside: compile as a method of a subclass body so the class cell exists.
        body = 'class _M(OverlapSegmentDriver):\n' + textwrap.indent(source.replace(old, new), '    ')
        namespace = dict(vars(so))
        exec(compile(body, '<mutant _stage_control>', 'exec'), namespace)
        method = namespace['_M'].__dict__['_stage_control']

        def mutant(self, command):
            return method(self, command)
        return mutant

    def test_install_check_removed(self):
        mutant = self.overlap_control("        if not self.defer_bind_installed:\n"
                                      "            raise DriverError('native-stage-overlap-defer-bind-not-installed')\n", '')
        self.fails(so.OverlapSegmentDriver, '_stage_control', mutant, Install('test_refused_where_not_installed'))

    def test_overlap_check_removed(self):
        mutant = self.overlap_control("        if defer and overlap is not True:\n"
                                      "            raise DriverError('native-stage-overlap-defer-bind-needs-overlap')\n", '')
        self.fails(so.OverlapSegmentDriver, '_stage_control', mutant,
                   Install('test_refused_with_overlap_false_or_absent_and_nothing_applied'))

    def test_startup_guard_removed(self):
        source = textwrap.dedent(inspect.getsource(s5_startup.install_engine))
        old = "        if value == '1' and env.get('FLASH_S5_STAGE_OVERLAP', '0') != '1':\n"
        assert source.count(old) == 1
        namespace = dict(vars(s5_startup))
        exec(compile(source.replace(old, "        if False:\n"), '<mutant install_engine>', 'exec'), namespace)
        self.fails(s5_startup, 'install_engine', namespace['install_engine'],
                   Startup('test_rebind_switches_need_the_overlap_switch'))

    def test_session_refusal_removed(self):
        source = textwrap.dedent(inspect.getsource(nos.NativeOverlapSession._action_native_stage_local))
        old = "            raise NativeOverlapSessionError('native-stage-overlap-defer-bind-not-installed')\n"
        old = old[4:]   # dedented with the method
        assert source.count(old) == 1
        body = 'class _M:\n' + textwrap.indent(source.replace(old, '        pass\n'), '    ')
        namespace = dict(vars(nos))
        exec(compile(body, '<mutant session>', 'exec'), namespace)
        self.fails(nos.NativeOverlapSession, '_action_native_stage_local', namespace['_M'].__dict__['_action_native_stage_local'],
                   RouterNotInstalled('test_field_refused_where_not_installed'))


if __name__ == '__main__':
    unittest.main()
