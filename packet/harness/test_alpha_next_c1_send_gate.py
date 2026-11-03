#!/usr/bin/env python3
"""alpha-next-c1b: the deferred frame's PP_SEND gate (``OverlapSegmentDriver._segment_gate``), reachability and contract.

READY (alpha-next-c1, "Known test gaps" 1): the mutant ``send-gate-removed`` (the overlap gate replaced by C0's
``StagedSegmentDriver._segment_gate``) failed no test of test_c1_rebind_witness, test_c1_rebind_on or
test_alpha_next_c1_defer_control. The gate's own clause holds stage 0 of a deferred frame when its next command is its
PP_SEND and its senders are not all bound ("armed" below: ``_deferred_send_next`` on the scoped cursor). Normally
``_deferred_bind_due`` moves that cursor into ``bind`` first; it declines in two cases with the send next: (i) a stage-0
route of the frame still in flight, (ii) another frame scoped (``_scoped_frame`` set).

Finding (``Reachability``, asserted): on this engine the armed gate is NOT reachable through the driver's real paths.

* Tick (``advance`` -> ``_drain`` -> ``OverlapSegmentDriver._drive``): ``_deferred_bind_due`` runs for every frame
  before any ``_drive_stage``, and between the two nothing applies a reply (``_drive_stage`` only submits), so a frame
  armed at the gate was armed at the bind-due, which then declined:
  (i) a stage-0 route of the frame in flight: that route is in the scoped ``_inflight`` (routes are a subset of it), so
      ``_drive_stage`` returns at ``if self._inflight`` before the gate;
  (ii) ``_scoped_frame`` set: ``_drive`` is never called in a scope (its only callers are ``advance`` and, with no
      frame, the frozen ``progress_in_wait``), and if it were, ``_drive_stage`` raises ``stage scopes do not nest``
      before the gate.
* In-wait (``StagedSegmentDriver.progress_in_wait``, where a frame IS scoped around each drive): ``_deferred_bind_due``
  never runs, and the gate is consulted only when ``_stage_wait_ok`` held, which for an idle segments cursor needs its
  next entry in PREFIX/LAYER/SUFFIX (``_WAIT_SEGMENT_KINDS``); PP_SEND is not, so the clause is false there.
* Moreover state (i) does not arise naturally: one command per stage, and the SUFFIX reply that makes the send next is
  that stage's only route; the next tick's bind-due then fires. Here the route is planted (a real stage-0 poll
  submitted through the driver's own ``_submit``), the one declared leaf of this file.

So the gate is defence in depth, and ``send-gate-removed`` is an equivalent mutant on the unfaulted paths. It stays
(no engine change). ``GateHolds`` tests its contract: directly, in a real state where C0's gate would let the send
go, and connected with the one fault that makes it the only guard (the bind-due never firing): no PP_SEND of the
frame leaves before its own binding on the tick or in-wait paths. Both fail under ``send-gate-removed``.

Fixtures: test_e3_stage_overlap.OverlapStack (real StagedAdmissionController, OverlapSegmentDriver, workers, PP
channel, ReplyTransport), test_c1_rebind_witness's RebindBase / ReceiverHold / SenderSpy, test_e3_stage_local's
burst in-wait setup. Spies wrap the instance and call the class attribute, so a class-level mutant is what runs.
CPU only; no pass here is native evidence.
"""
import ast
import inspect
import random
import threading
import types
import unittest
from unittest import mock

import test_e3_stage_overlap  # noqa: F401  first: its imports put the engine sources on the path
import test_c1_rebind_witness as rw
import engine_driver as ed
import stage_overlap as so
import stage_progress as sp
from stage_net import stage_of_command, w

PP_SEND = w.SegmentKind.PP_SEND
SUFFIX = w.SegmentKind.SUFFIX


def never_due(self, frame):
    """The one fault ``GateHolds`` injects: the deferred bind never becomes due (rebind_mutants_3's bind-never-due)."""
    return False


class Spies:
    """Records every ``_segment_gate``, ``_deferred_bind_due`` and segment send of one driver; changes no answer."""

    def __init__(self, driver):
        self.driver = driver
        self.gate, self.due, self.sends = [], [], []
        cls = type(driver)
        senders = len(driver.stage_ranks[0])

        def gate(frame, stage):
            armed = bool(stage == 0 and getattr(frame, 'deferred_bind', False)
                         and driver._deferred_send_next(driver.__dict__, driver.__dict__))
            result = cls._segment_gate(driver, frame, stage)
            self.gate.append(dict(frame=frame, stage=stage, deferred=bool(getattr(frame, 'deferred_bind', False)),
                                  armed=armed, in_wait=bool(driver._in_wait), result=result))
            return result

        def due(frame):
            cursor = frame.stage[0]
            armed = bool(not frame.dead and getattr(frame, 'deferred_bind', False) and cursor['phase'] == 'segments'
                         and driver._deferred_send_next(cursor, frame.chunk))
            row = dict(frame=frame, armed=armed, scoped=driver._scoped_frame is not None,
                       route=any(route == (frame, 0) for route in driver._seq_route.values()),
                       in_wait=bool(driver._in_wait))
            row['result'] = cls._deferred_bind_due(driver, frame)
            self.due.append(row)
            return row['result']

        def send_segment(command, targets):
            frame = driver._scoped_frame
            if command.kind is PP_SEND:
                self.sends.append(dict(frame=frame, txn=command.transaction.txn_seq,
                                       deferred=bool(getattr(frame, 'deferred_bind', False)),
                                       bound=len(driver._bindings), senders=senders, in_wait=bool(driver._in_wait)))
            return cls._send_segment(driver, command, targets)
        driver._segment_gate = gate
        driver._deferred_bind_due = due
        driver._send_segment = send_segment

    def armed_gate(self, frame=None):
        return [row for row in self.gate if row['armed'] and (frame is None or row['frame'] is frame)]

    def unbound_sends(self):
        return [row for row in self.sends if row['bound'] < row['senders']]


class Base(rw.RebindBase):
    defer = True

    def burst(self, network):
        """test_e3_stage_local.BurstInWait.burst on the overlap stack: the held-f burst permit on the owner thread."""
        driver = network.driver
        driver.auto_arm = types.SimpleNamespace(manual_owned=False, phase='native', active=False)
        driver._admission_intent = dict(origin='auto-arm', phase='native', seq=1)
        driver.install_progress('burst')
        driver.progress.mode = 'burst'
        return network

    def wait_ready(self, network):
        progress = network.driver.progress
        self.assertTrue(progress.burst())
        progress.last_stock_kind = 'sample_tokens'
        self.assertEqual(progress.owner, threading.get_ident())

    def in_wait_tick(self, network, waits=3, period=6):
        """test_e3_stage_overlap's slow stage 1 (its ranks read the ring every ``period`` ticks, so successors open
        early and defer), with up to ``waits`` in-wait progress calls inside the stock wait before the engine's tick."""
        driver = network.driver
        count = [0]

        def tick():
            count[0] += 1
            network.admit_next()
            network.complete_everything()
            network.pump((0, 1))
            if count[0] % period == 0:
                network.pump((2, 3))
            for _ in range(waits):
                if not driver._frames:
                    break
                self.wait_ready(network)
                if driver.progress_in_wait() is None:
                    break
                network.complete_everything()
                network.pump((0, 1))
            out = driver.advance()
            network.note_peak()
            return out
        return tick

    def deferred_successor(self, driver):
        """The live frame that opened deferred (an early-opened successor; it may be the oldest once n committed)."""
        for frame in driver._frames:
            if getattr(frame, 'deferred_bind', False) and not frame.dead:
                return frame
        return None

    def armed_after_drain(self, network, tick=None):
        """A deferred successor whose stage-0 SUFFIX is in flight, then its reply applied by the real ``_drain`` alone:
        stage 0's next command is its PP_SEND, no binding, phase still ``segments`` (no ``_drive`` ran yet)."""
        driver = network.driver

        def suffix_in_flight():
            frame = self.deferred_successor(driver)
            if frame is None:
                return False
            command = frame.stage[0]['_command']
            return (frame.stage[0]['phase'] == 'segments' and command is not None and command.kind is SUFFIX
                    and any(route == (frame, 0) for route in driver._seq_route.values()))
        self.reach(network, suffix_in_flight, limit=2000, tick=tick)
        frame = self.deferred_successor(driver)
        network.complete_everything()
        network.pump()
        driver._drain()
        self.assert_armed(driver, frame)
        return frame

    def assert_armed(self, driver, frame):
        cursor = frame.stage[0]
        self.assertTrue(frame.deferred_bind)
        self.assertEqual(cursor['phase'], 'segments')
        self.assertIsNone(cursor['_retry'])
        self.assertIs(cursor['plan'][cursor['index']]['kind'], PP_SEND)
        self.assertEqual(frame.chunk['_bindings'], {})
        self.assertTrue(driver._deferred_send_next(cursor, frame.chunk))

    def pp_sends_of(self, network, txn):
        out = []
        for envelope in network.sent_envelopes:
            method, message, _ranks = stage_of_command(envelope)
            if method == 'flash_s5_submit' and message.kind is PP_SEND and message.transaction.txn_seq == txn:
                out.append(message)
        return out


# ------------------------------------------------------------------------------------------------ reachability
class Reachability(Base):
    """The armed gate is not reached through the real tick or in-wait paths; see the module docstring."""

    def check_run(self, network, spies, in_wait=False):
        driver = network.driver
        self.assertIsNone(driver.error)
        self.assertEqual(network.controller.commits, network.expected_commits)
        self.assert_serial(network)
        opens = driver.defer_counters['deferred_opens']
        self.assertGreater(opens, 0)
        # Every deferred frame went into bind through _deferred_bind_due (none bypassed it).
        self.assertEqual(driver.defer_counters['deferred_binds'], opens)
        deferred = [row for row in spies.gate if row['deferred'] and row['stage'] == 0]
        self.assertTrue(deferred, 'the gate was never consulted for a deferred frame: vacuous')
        if in_wait:
            self.assertTrue([row for row in deferred if row['in_wait']], 'no in-wait gate call: vacuous')
        self.assertEqual(spies.armed_gate(), [])
        # Whenever the bind-due saw the armed state it fired: conditions (i) and (ii) never met it.
        self.assertEqual([row for row in spies.due if row['armed'] and not row['result']], [])
        self.assertEqual([row for row in spies.due if row['scoped'] or row['in_wait']], [])
        self.assertTrue([row for row in spies.due if row['armed'] and row['result']])
        self.assertTrue(spies.sends)
        self.assertEqual(spies.unbound_sends(), [])

    def test_fair_and_slow_stage_one_runs_never_arm_the_gate(self):
        for name in ('fair', 'slow'):
            with self.subTest(timing=name):
                network = self.network()
                spies = Spies(network.driver)
                network.run_request(tick=self.slow_tick(network) if name == 'slow' else None)
                self.check_run(network, spies)

    def test_late_consumer_every_chunk_never_arms_the_gate(self):
        network = self.network()
        hold = rw.ReceiverHold(network, lambda txn: True)
        spy = rw.SenderSpy(network)
        spies = Spies(network.driver)
        network.run_request(tick=self.late_consumer_tick(network, hold))
        self.check_run(network, spies)
        self.assertEqual(spy.violations, [])

    def test_adversarial_timing_never_arms_the_gate(self):
        """test_c1_rebind_witness.PeerIdentity's adversarial timing (random held consumers, ranks behind)."""
        for seed in range(6):
            rng = random.Random(seed)
            network = self.network()
            hold = rw.ReceiverHold(network, lambda txn, rng=rng: rng.random() < 0.6)
            spies = Spies(network.driver)
            late = self.late_consumer_tick(network, hold, patience=rng.randint(1, 6), cap=60)

            def tick(network=network, rng=rng, late=late):
                behind = rng.choice(((), (0, 1), (2, 3), (1,), (3,)))
                network.admit_next()
                network.complete_everything()
                for rank in (0, 1, 2, 3):
                    if rank in behind and rng.random() < 0.8:
                        continue
                    for _ in range(rng.randint(1, 4)):
                        network.step(rank)
                network.settle()
                return late()
            with self.subTest(seed=seed):
                network.run_request(tick=tick)
                self.check_run(network, spies)

    def test_in_wait_runs_never_arm_the_gate(self):
        network = self.burst(self.network(direct=True))
        spies = Spies(network.driver)
        network.run_request(tick=self.in_wait_tick(network))
        self.check_run(network, spies, in_wait=True)

    def test_case_i_stage_zero_route_in_flight(self):
        """(i) Armed, with a stage-0 route of the frame in flight (planted: a real poll through the driver's _submit).
        The tick's bind-due declines (route), and the gate is not reached: _drive_stage returns at its in-flight check."""
        network = self.network()
        driver = network.driver
        frame = self.armed_after_drain(network)
        with driver._scope(frame, 0):
            planted = driver._submit('poll', ('flash_s5_poll', (sp.PollRequest(driver.tx, driver.admission.state_epoch,
                                                                                driver.poll_budget),), {}, None),
                                     driver.ranks)
        self.assertEqual(driver._seq_route[planted], (frame, 0))
        spies = Spies(driver)
        txn = frame.chunk['tx'].txn_seq
        driver.advance()               # no pump: the planted poll stays unanswered
        self.assertEqual([(row['armed'], row['route'], row['scoped'], row['result'])
                          for row in spies.due if row['frame'] is frame], [(True, True, False, False)])
        self.assertEqual([row for row in spies.gate if row['frame'] is frame and row['stage'] == 0], [])
        self.assertEqual(frame.stage[0]['phase'], 'segments')
        self.assertEqual(self.pp_sends_of(network, txn), [])
        # The poll answers; the next tick's bind-due fires and the gate is never armed.
        network.tick()
        self.assertNotIn(planted, driver._seq_route)
        self.assertEqual(frame.stage[0]['phase'] if not frame.dead else 'bind', 'bind')
        network.run_request()
        self.assertEqual(spies.armed_gate(), [])
        self.assertEqual(spies.unbound_sends(), [])
        self.assert_serial(network)

    def test_case_i_does_not_arise_unplanted(self):
        """Without the planted route the next tick after the SUFFIX reply moves the frame into bind before any drive."""
        network = self.network()
        driver = network.driver
        frame = self.armed_after_drain(network)
        spies = Spies(driver)
        driver.advance()
        self.assertEqual([(row['armed'], row['route'], row['scoped'], row['result'])
                          for row in spies.due if row['frame'] is frame], [(True, False, False, True)])
        self.assertEqual(spies.armed_gate(), [])
        network.run_request()
        self.assertEqual(spies.armed_gate(), [])
        self.assert_serial(network)

    def test_case_ii_in_wait_with_the_send_next(self):
        """(ii) In-wait, where a frame is scoped around each drive: the bind-due never runs, and the armed stage is not
        wait-ok (PP_SEND is not a wait segment kind), so the gate is not consulted for it and nothing is sent."""
        network = self.burst(self.network(direct=True))
        driver = network.driver
        # Reached on the slow-stage-1 tick (an in-wait run applies the SUFFIX reply and binds within one tick).
        frame = self.armed_after_drain(network, tick=self.slow_tick(network))
        spies = Spies(driver)
        wait_ok = []
        real = type(driver)._stage_wait_ok

        def stage_wait_ok(f, stage):
            result = real(driver, f, stage)
            wait_ok.append((f is frame, stage, result))
            return result
        driver._stage_wait_ok = stage_wait_ok
        txn = frame.chunk['tx'].txn_seq
        self.wait_ready(network)
        sent = driver.progress_in_wait()
        del driver._stage_wait_ok
        self.assertIsNotNone(sent)                  # the permit held: the per-stage loop ran
        self.assertIn((True, 0, False), wait_ok)
        self.assertNotIn((True, 0, True), wait_ok)
        self.assertEqual(spies.due, [])
        self.assertEqual([row for row in spies.gate if row['frame'] is frame and row['stage'] == 0], [])
        self.assertEqual(frame.stage[0]['phase'], 'segments')
        self.assertEqual(self.pp_sends_of(network, txn), [])
        # The tick then binds and sends.
        network.run_request(tick=self.in_wait_tick(network))
        self.assertEqual(spies.armed_gate(), [])
        self.assertEqual(spies.unbound_sends(), [])
        self.assert_serial(network)

    def test_case_ii_tick_under_another_frames_scope(self):
        """(ii) on the tick: were _drive ever called inside a scope, the bind-due declines (scoped) and the first
        _drive_stage refuses the nested scope before any gate. No engine path does this (see the call-site test)."""
        network = self.network()
        driver = network.driver
        frame = self.armed_after_drain(network)
        older = driver._frames[0]
        spies = Spies(driver)
        with self.assertRaises(ed.DriverError) as caught:
            with driver._scope(older, 1):
                driver._drive()
        self.assertIn('stage scopes do not nest', str(caught.exception))
        self.assertEqual([(row['armed'], row['scoped'], row['result']) for row in spies.due if row['frame'] is frame],
                         [(True, True, False)])
        self.assertEqual(spies.gate, [])
        self.assertIsNone(driver.error)
        self.assertEqual(frame.stage[0]['phase'], 'segments')
        network.run_request()
        self.assertEqual(spies.armed_gate(), [])
        self.assert_serial(network)

    def test_call_sites(self):
        """The structure the finding rests on, read from the engine sources: who calls the gate, the bind-due, the
        per-stage drive and the overlap ``_drive``. A new call site fails here and reopens the question."""
        sites = {}
        for module in (ed, sp, so):
            tree = ast.parse(inspect.getsource(module))
            for cls in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
                for fn in [node for node in cls.body if isinstance(node, ast.FunctionDef)]:
                    for node in ast.walk(fn):
                        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                                and node.func.attr in ('_segment_gate', '_deferred_bind_due', '_drive_stage', '_drive',
                                                       '_stage_wait_ok')):
                            receiver = node.func.value
                            via = ('self' if isinstance(receiver, ast.Name) and receiver.id == 'self'
                                   else 'super' if isinstance(receiver, ast.Call) and getattr(receiver.func, 'id', '') == 'super'
                                   else 'other')
                            sites.setdefault(node.func.attr, set()).add((cls.name, fn.name, via))
        self.assertEqual(sites['_segment_gate'], {('StagedSegmentDriver', '_drive_stage', 'self'),
                                                  ('StagedSegmentDriver', 'progress_in_wait', 'self'),
                                                  ('OverlapSegmentDriver', '_segment_gate', 'super')})
        self.assertEqual(sites['_deferred_bind_due'], {('OverlapSegmentDriver', '_drive', 'self')})
        self.assertEqual(sites['_drive_stage'], {('StagedSegmentDriver', '_drive', 'self'),
                                                 ('OverlapSegmentDriver', '_drive', 'self')})
        self.assertEqual(sites['_stage_wait_ok'], {('StagedSegmentDriver', '_wait_permit', 'self'),
                                                   ('StagedSegmentDriver', 'progress_in_wait', 'self')})
        # self._drive(): advance and the frozen progress_in_wait (overridden by the staged one whenever a frame
        # exists); every in-scope drive is super()._drive(), which resolves to NativeSegmentDriver._drive.
        self.assertEqual({site for site in sites['_drive'] if site[2] == 'self'},
                         {('NativeSegmentDriver', 'advance', 'self'),
                          ('NativeSegmentDriver', 'progress_in_wait', 'self')})
        self.assertEqual({site for site in sites['_drive'] if site[2] == 'super'},
                         {('StagedSegmentDriver', '_drive', 'super'), ('StagedSegmentDriver', '_drive_stage', 'super'),
                          ('StagedSegmentDriver', 'progress_in_wait', 'super'), ('OverlapSegmentDriver', '_drive', 'super')})
        self.assertIs(so.OverlapSegmentDriver.__mro__[2], ed.NativeSegmentDriver)
        self.assertNotIn('_segment_gate', vars(ed.NativeSegmentDriver))
        self.assertNotIn('_deferred_bind_due', vars(ed.NativeSegmentDriver))
        self.assertNotIn(PP_SEND, ed._WAIT_SEGMENT_KINDS)


# ------------------------------------------------------------------------------------------------ the contract
class GateHolds(Base):
    """The gate's own clause holds a deferred frame's PP_SEND until that frame's binding."""

    def faulted_wait(self, network, tick=None, ticks=120, spies=None):
        """With the bind-due faulted to never fire: run until a deferred successor's stage 0 waits at its unbound
        PP_SEND while its stage 1 has opened and moved on (C0's gate would let the send go), then ``ticks`` more.
        With ``spies``, every tick is checked: no PP_SEND left unbound (a failure names it, before any later crash)."""
        driver = network.driver
        plain = tick or network.tick

        def tick():
            out = plain()
            if spies is not None:
                self.assertEqual(spies.unbound_sends(), [], 'a PP_SEND left before its own binding')
            return out

        def waiting():
            frame = self.deferred_successor(driver)
            return (frame is not None and frame.stage[0]['phase'] == 'segments'
                    and driver._deferred_send_next(frame.stage[0], frame.chunk)
                    and frame.stage[1]['phase'] == 'segments' and frame.stage[1]['index'] > 0)
        self.reach(network, waiting, limit=3000, tick=tick)
        frame = self.deferred_successor(driver)
        for _ in range(ticks):
            tick()
            self.assertIsNone(driver.error)
        return frame

    def test_direct_contract(self):
        network = self.network()
        driver = network.driver
        with mock.patch.object(so.OverlapSegmentDriver, '_deferred_bind_due', never_due):
            frame = self.faulted_wait(network, ticks=0)
        self.assert_armed(driver, frame)
        c0 = sp.StagedSegmentDriver._segment_gate
        with driver._scope(frame, 0):
            self.assertIs(c0(driver, frame, 0), False)          # C0's gate alone would let the send go
            self.assertIs(driver._segment_gate(frame, 0), True)  # the deferred clause holds it
            bindings = driver._bindings
            driver._bindings = {rank: object() for rank in driver.stage_ranks[0]}
            self.assertIs(driver._segment_gate(frame, 0), False)  # bound: C0's answer
            driver._bindings = {driver.stage_ranks[0][0]: object()}
            self.assertIs(driver._segment_gate(frame, 0), True)   # one sender of two bound: still held
            driver._bindings = bindings
            frame.deferred_bind = False
            self.assertIs(driver._segment_gate(frame, 0), False)  # not a deferred frame: C0's answer
            frame.deferred_bind = True
        with driver._scope(frame, 1):
            self.assertIs(driver._segment_gate(frame, 1), c0(driver, frame, 1))   # stage 1: C0's gate, unchanged
        # Restored: the run completes (bind-due back) and equals the serial ledger.
        network.run_request()
        self.assert_serial(network)

    def test_natural_armed_state(self):
        """The armed state as the real path leaves it (the SUFFIX reply applied by ``_drain``, the next ``_drive`` not
        yet run; fair timing, n already committed): stage 1 has opened the frame and waits at its PP_RECEIVE, so C0's
        gate would let the send go; only the bind-due at the top of the next ``_drive``, or this gate, holds it."""
        network = self.network()
        driver = network.driver
        frame = self.armed_after_drain(network)
        self.assertEqual((frame.stage[1]['phase'], frame.stage[1]['index']), ('segments', 1))
        with driver._scope(frame, 0):
            self.assertIs(sp.StagedSegmentDriver._segment_gate(driver, frame, 0), False)
            self.assertIs(driver._segment_gate(frame, 0), True)
        network.run_request()
        self.assert_serial(network)

    def faulted_run(self, in_wait):
        network = self.network(direct=in_wait)
        if in_wait:
            self.burst(network)
        driver = network.driver
        spies = Spies(driver)
        tick = self.in_wait_tick(network) if in_wait else None
        with mock.patch.object(so.OverlapSegmentDriver, '_deferred_bind_due', never_due):
            try:
                frame = self.faulted_wait(network, tick=tick, spies=spies)
            except Exception as exc:
                # The engine failed (e.g. a consumer proof for a sender with no binding): name the cause if it is ours.
                self.assertEqual(spies.unbound_sends(), [], 'engine failed after an unbound PP_SEND: %r' % (exc,))
                raise
            txn = frame.chunk['tx'].txn_seq
            held = spies.armed_gate(frame)
            self.assertTrue(held)
            self.assertTrue(all(row['result'] is True for row in held), held)
            if in_wait:
                # In-wait never asks the gate about the send (not a wait kind): the tick's gate is what held it.
                self.assertEqual([row for row in held if row['in_wait']], [])
            self.assertEqual(self.pp_sends_of(network, txn), [])
            self.assertEqual([op for op in rw.stage_ops(network, (0, 1), txn) if op[1] == 'pp_send'], [])
            self.assertEqual(spies.unbound_sends(), [])
            self.assertEqual(frame.chunk['_bindings'], {})
        # The fault lifted: the bind-due fires, the frame binds, then sends; the request equals the serial one.
        network.run_request(tick=tick)
        self.assertEqual(spies.unbound_sends(), [])
        sends = [row for row in spies.sends if row['txn'] == txn]
        self.assertTrue(sends)
        self.assertTrue(all(row['bound'] == row['senders'] for row in sends))
        self.assertIsNone(driver.error)
        self.assert_serial(network)

    def test_no_send_before_its_binding_on_the_tick(self):
        self.faulted_run(in_wait=False)

    def test_no_send_before_its_binding_with_in_wait(self):
        self.faulted_run(in_wait=True)


if __name__ == '__main__':
    unittest.main()
