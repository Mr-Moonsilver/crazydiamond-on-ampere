"""TP completion skew at STATE_FINALIZE, through the real native driver.

Box run q18a-native41-autoarm run 1 died on ``DriverError('TP workers disagree
on the layer continuation')``. ``worker_bridge._cursor`` derives a finished
lane's phase from that rank's own ``state_event.query()``: 'commit' once its
own state write completed, 'state_finalize' before that. When two TP peers
accept the same STATE_FINALIZE while one stream is idle (so its event completes
at record) and the other is still busy, the old full-cursor agreement refused a
valid continuation. ``engine_driver._continuation`` now treats 'commit' and
'state_finalize' as one continuation for a STATE_FINALIZE acceptance, and
nowhere else.

The tests drive the connected composition of
``test_e3_native_schedule_profile.ProfileNetwork``: the real
``NativeSegmentDriver``, ``NativeWorkerBridge``, ``LaneDispatcher``,
``QsaLayerDriver`` and ``QsaPrefillOperation`` on stage 0, and r4 lanes on
stage 1, with per-rank asynchronous, in-order event completion.

- (a) The skew is one continuation. Nothing that needs device completion runs
  before both ranks' own state events have completed: the release RPC, a
  worker's release, its dispatcher's drain and release, the handback, any route
  transition or lane open after finalize, or any further segment on that
  stage. ``CompletionGuard`` checks each action at the moment it runs.
- (b) A delayed peer: the accepted rank keeps its retained acceptance and
  never repeats a submission, copy or continuation. No rank enters the TP tail
  or any completion-dependent action early.
- (c) At the first agreement of every (kind, stage), every pair of distinct
  phases refuses, as does a missing cursor and any layer or bucket
  difference. At a paused layer, a paused, profile or tail difference also
  refuses. The one exception is 'commit' beside 'state_finalize' at
  STATE_FINALIZE.
- (d) A bounded random sweep. With the fix there are 0 refusals and 0 guard
  violations. Under ``PARENT_KEY``, the old full-cursor agreement, the same
  seeds refuse with the box's text.

Declared CPU leaves (fixture leaf -> the real engine object it replaces):
- The harness event buses and the fake sparse device (``ProfileNetwork``)
  stand in for the CUDA streams and events. Completion is in order per stream,
  as on a GPU, and each rank progresses on its own.
- ``ControllableEvent.record`` (the credit and r4 harness events), patched by
  ``idle_streams``, stands in for cudaEventRecord. An event recorded on a
  stream whose earlier events have all completed completes at once, as a CUDA
  event recorded on an idle stream does. The older all-at-once leaves never
  produced this, which is why no earlier CPU test reached the refusal.
- ``SkewNetwork`` marks the stage-0 fake model as not the last PP stage. The
  real stage-0 model of a two-stage pipeline is not the last stage, so its
  suffix produces the HC tensor that PP_SEND needs.
- The (c) probes call the real ``_apply_segments`` on the real driver, at the
  real outstanding command, with reply fields replaced on one or both TP
  peers. What runs before the agreement check is the retry path's idempotent
  bookkeeping. Every probed run must still finish clean on its own replies,
  which proves that the probes left no state behind.
No GPU, box or performance claim.
"""
import contextlib
from dataclasses import replace
import itertools
from pathlib import Path
import random
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'harness'), str(ROOT / 'patches/s5_engine')]
import test_e3_native_schedule_profile as sp  # noqa: E402
import test_e3_native_submission_credit as credit  # noqa: E402
import test_e3_r4_worker_bridge as r4  # noqa: E402
import engine_driver as ed  # noqa: E402
import wire_v1 as w  # noqa: E402

FINALIZE = w.SegmentKind.STATE_FINALIZE
LAYER = w.SegmentKind.LAYER
CONTINUATION = 'TP workers disagree on the layer continuation'
BENIGN = frozenset(('commit', 'state_finalize'))


def PARENT_KEY(kind, cursor):
    """The agreement key before the fix: the whole cursor."""
    return None if cursor is None else (cursor.phase, cursor.layer, cursor.bucket)


class SkewNetwork(sp.ProfileNetwork):
    """ProfileNetwork with per-rank completion control and agreement capture."""

    def __init__(self, harness):
        super().__init__(harness)
        for stack in self.qsa.values():
            # Leaf: the real stage-0 model of a two-stage pipeline is not the
            # last PP stage, so its suffix produces the HC tensor PP_SEND needs.
            stack.model._flash_e3_is_last_stage = lambda: False
        self.held = set()
        self.idle = set()
        self.agreements = []
        self.guard = None
        driver = self.driver
        real = driver._apply_segments

        def capture(values):
            values = tuple(values)
            command = driver._command
            self.agreements.append(dict(
                kind=command.kind, stage=command.stage,
                layer=command.descriptor.layer_start, op_seq=command.op_seq,
                rows={v.worker_rank: (v.accepted, v.pending, v.paused, v.cursor)
                      for v in values if v.worker_rank in driver._targets}))
            return real(values)
        driver._apply_segments = capture

    def bridge(self, rank):
        return self.qsa[rank].bridge if rank in self.qsa else self.workers[rank]._flash_s5_bridge

    # -- per-rank device progress -------------------------------------------
    def buses(self, rank):
        buses = [self.workers[rank].bus]
        if rank in self.qsa:
            buses.append(self.qsa[rank].bus)
        return buses

    def complete(self, ranks=(0, 1, 2, 3)):
        for rank in ranks:
            for bus in self.buses(rank):
                for event in bus.events:
                    if id(event) not in self.held:
                        event.complete = True
            if rank in self.qsa:
                for event in self.qsa[rank].device.events:
                    if id(event) not in self.held:
                        event.ready = True

    def hold_new_events(self, rank, call):
        """Run ``call`` and keep every event it created on ``rank`` incomplete."""
        before = [len(bus.events) for bus in self.buses(rank)]
        device = self.qsa[rank].device if rank in self.qsa else None
        device_before = len(device.events) if device is not None else 0
        try:
            return call()
        finally:
            for bus, start in zip(self.buses(rank), before):
                self.held.update(id(event) for event in bus.events[start:])
            if device is not None:
                self.held.update(id(event) for event in device.events[device_before:])

    def release_held(self):
        self.held.clear()

    def wrap_submit(self, rank, hook):
        bridge = self.bridge(rank)
        real = bridge.submit
        bridge.submit = lambda command: hook(command, lambda: real(command))

    def methods(self):
        return [envelope[0] for envelope in self.sender.envelopes]


@contextlib.contextmanager
def idle_streams(network):
    """An event recorded on an idle stream (every earlier event done) completes."""
    def wrap(original):
        def record(event, stream=None):
            original(event, stream)
            if id(event.bus) in network.idle:
                earlier = event.bus.events[:event.bus.events.index(event)]
                if all(e.complete for e in earlier):
                    event.complete = True
        return record
    with mock.patch.object(r4.ControllableEvent, 'record', wrap(r4.ControllableEvent.record)), \
         mock.patch.object(credit.ControllableEvent, 'record',
                           wrap(credit.ControllableEvent.record)):
        yield


class _Envelopes(list):
    def __init__(self, items, hook):
        super().__init__(items)
        self._hook = hook

    def append(self, envelope):
        self._hook(envelope)
        super().append(envelope)


class CompletionGuard:
    """Every completion-dependent action, checked at the moment it runs.

    Each rank's own state event is latched once it has completed. The network
    drives one transaction, so the latch never needs a reset.

    Release scope needs every rank's state event to have completed:
    - the driver's release (``_send_release`` and the ``flash_s5_release`` RPC);
    - a worker's ``release``;
    - its dispatcher's successful ``drain`` and its ``release``;
    - ``_note_terminal_route``;
    - any ``flash_s5_transition`` or ``flash_s5_open`` sent after the first
      accepted STATE_FINALIZE (the handback, or the next chunk's open/copy).

    Stage scope needs both TP peers' events: any ``flash_s5_submit`` for a stage
    after that stage's STATE_FINALIZE was accepted, which would be a TP tail
    collective or the next chunk's copy.
    """

    def __init__(self, network):
        self.network = network
        self.driver = network.driver
        self.done = set()
        self.finalized = set()
        self.actions = []
        self.violations = []
        network.guard = self
        driver = self.driver
        inner = driver._apply_segments

        def apply(values):
            result = inner(values)
            command = driver._command
            if command is not None and command.kind is FINALIZE and driver._retry is None:
                self.finalized.add(command.stage)
            return result
        driver._apply_segments = apply
        self._wrap(driver, '_send_release', lambda *a: ('driver.release', None, self.ranks()))
        self._wrap(driver, '_note_terminal_route',
                   lambda *a: ('driver.terminal-route', None, self.ranks()))
        for rank in self.ranks():
            bridge = network.bridge(rank)
            self._wrap(bridge, 'release', lambda *a, r=rank: ('bridge.release', r, self.ranks()))
            dispatcher = bridge.dispatcher
            self._wrap(dispatcher, 'release',
                       lambda *a, r=rank: ('dispatcher.release', r, self.ranks()))
            self._wrap_drain(dispatcher, rank)
        network.sender.envelopes = _Envelopes(network.sender.envelopes, self._envelope)

    def ranks(self):
        return tuple(self.driver.ranks)

    def observe(self):
        for rank in self.ranks():
            event = self.network.bridge(rank).dispatcher.state_event
            if event is not None and event.complete:
                self.done.add(rank)

    def _check(self, name, rank, needed):
        self.observe()
        missing = sorted(set(needed) - self.done)
        self.actions.append((name, rank))
        if missing:
            self.violations.append((name, rank, tuple(missing)))

    def _wrap(self, owner, attribute, describe):
        real = getattr(owner, attribute)

        def call(*args, **kwargs):
            self._check(*describe(*args))
            return real(*args, **kwargs)
        setattr(owner, attribute, call)

    def _wrap_drain(self, dispatcher, rank):
        real = dispatcher.drain

        def drain():
            self.observe()
            drained = real()
            if drained:
                self._check('dispatcher.drain', rank, self.ranks())
            return drained
        dispatcher.drain = drain

    def _envelope(self, envelope):
        method, args = envelope[0], envelope[1]
        if method == 'flash_s5_release':
            self._check('rpc.release', None, self.ranks())
        elif method in ('flash_s5_transition', 'flash_s5_open') and self.finalized:
            self._check('rpc.' + method[len('flash_s5_'):], None, self.ranks())
        elif method == 'flash_s5_submit':
            command = args[0]
            if command.stage in self.finalized:
                self._check('rpc.submit-after-finalize', command.stage,
                            self.driver.stage_ranks[command.stage])

    def named(self, *names):
        return [action for action in self.actions if action[0] in names]

    RELEASE_SCOPE = ('driver.release', 'driver.terminal-route', 'bridge.release',
                     'dispatcher.release', 'dispatcher.drain', 'rpc.release',
                     'rpc.transition', 'rpc.open')


class TPFinalizeSkew(credit.CreditHarness, unittest.TestCase):
    legacy_schedule_source = False
    query_rows = 64
    stack_type = sp.HistoryStack
    selection_counts = (4097, 4097)

    @classmethod
    def setUpClass(cls):
        if not credit.REFERENCE.is_dir():
            raise RuntimeError('required reference absent; connected proof NOT_RUN')
        super().setUpClass()

    def drive(self, network, until, limit=600, complete=None):
        for tick in range(limit):
            (complete or network.complete)()
            network.driver.advance(deliver_outputs=False)
            if network.guard is not None:
                network.guard.observe()
            if until():
                return tick + 1
        self.fail('driver did not reach the expected state: phase=%s index=%s'
                  % (network.driver.phase, network.driver.index))

    def assert_clean_finish(self, network):
        self.assertIsNone(network.driver.error)
        self.assertFalse(network.driver.active)
        self.assertTrue(network.all_dispatchers_reset())
        self.assertTrue(network.controller.leases.shutdown()['clean'])
        self.assertEqual(network.transport._calls, {})

    def assert_guard_clean(self, guard):
        self.assertEqual(guard.violations, [])
        # The guard saw each completion-dependent action, on every rank.
        for name in ('bridge.release', 'dispatcher.drain', 'dispatcher.release'):
            self.assertEqual({rank for n, rank in guard.actions if n == name},
                             set(guard.ranks()), name)
        self.assertTrue(guard.named('driver.release'))
        self.assertTrue(guard.named('rpc.release'))
        self.assertTrue(guard.named('rpc.transition'))

    def assert_diagnostic(self, message, kind, stage):
        detail = message[message.index(' ['):]
        self.assertLessEqual(len(detail), 1024)
        self.assertIn('[command=%s stage=%d ' % (kind.value, stage), message)
        self.assertRegex(message, r'uid_sha=[0-9a-f]{16} ')
        for rank in ((0, 1) if stage == 0 else (2, 3)):
            self.assertRegex(message, r'r%d:acc=1,pend=0,paused=[01],cursor=(\w+/\d+/\d+|none),'
                                      r'profile=[01],tail=[01]' % rank)

    # -- (a) the box failure -------------------------------------------------
    def skew_finalize(self, network, stage, idle_rank, late_rank):
        """``idle_rank``'s finalize events complete at record (idle stream);
        ``late_rank``'s stay incomplete until ``release_held``."""
        def idle_hook(command, call):
            if command.kind is FINALIZE and command.stage == stage:
                network.idle.update(id(bus) for bus in network.buses(idle_rank))
                try:
                    return call()
                finally:
                    network.idle.clear()
            return call()

        def late_hook(command, call):
            if command.kind is FINALIZE and command.stage == stage:
                return network.hold_new_events(late_rank, call)
            return call()
        network.wrap_submit(idle_rank, idle_hook)
        network.wrap_submit(late_rank, late_hook)

    def finalize_agreement(self, network, stage):
        found = [a for a in network.agreements if a['kind'] is FINALIZE and a['stage'] == stage]
        return found[0] if found else None

    def test_a_state_finalize_completion_skew_is_one_continuation(self):
        """One rank's state event completes at record, its peer's does not.

        Under the parent key the same schedule refuses with the box's text.
        With the fix the acceptance is one continuation, and no
        completion-dependent action runs before the late rank's own state
        event has completed.
        """
        for stage, idle_rank, late_rank in ((0, 0, 1), (0, 1, 0), (1, 2, 3), (1, 3, 2)):
            with self.subTest(stage=stage, idle=idle_rank, late=late_rank):
                parent = SkewNetwork(self)
                with idle_streams(parent), mock.patch.object(ed, '_continuation', PARENT_KEY):
                    self.skew_finalize(parent, stage, idle_rank, late_rank)
                    with self.assertRaises(ed.DriverError) as caught:
                        self.drive(parent, lambda: not parent.driver.active)
                message = str(caught.exception)
                self.assertTrue(message.startswith(CONTINUATION), message)
                self.assert_diagnostic(message, FINALIZE, stage)
                self.assertIn('r%d:acc=1,pend=0,paused=0,cursor=commit/' % idle_rank, message)
                self.assertIn('r%d:acc=1,pend=0,paused=0,cursor=state_finalize/' % late_rank,
                              message)

                network = SkewNetwork(self)
                guard = CompletionGuard(network)
                driver = network.driver
                late = network.bridge(late_rank).dispatcher
                with idle_streams(network):
                    self.skew_finalize(network, stage, idle_rank, late_rank)
                    self.drive(network, lambda: self.finalize_agreement(network, stage))
                    finalize = self.finalize_agreement(network, stage)
                    phases = {rank: row[3].phase for rank, row in finalize['rows'].items()}
                    self.assertEqual(phases, {idle_rank: 'commit', late_rank: 'state_finalize'})
                    self.assertEqual({row[2] for row in finalize['rows'].values()}, {False})
                    self.assertEqual(len({(row[3].layer, row[3].bucket)
                                          for row in finalize['rows'].values()}), 1)
                    self.assertIn(stage, guard.finalized)
                    # The late rank's state event is held. The plan may run to
                    # its end, but nothing completion-dependent may run.
                    for _ in range(40):
                        network.complete()
                        driver.advance(deliver_outputs=False)
                        guard.observe()
                        self.assertIsNone(driver.error)
                        self.assertTrue(driver.active)
                        self.assertFalse(late.state_event.complete)
                        self.assertNotIn(late_rank, guard.done)
                        self.assertFalse(driver._drained())
                        self.assertEqual(guard.named(*CompletionGuard.RELEASE_SCOPE), [])
                        self.assertEqual(guard.named('rpc.submit-after-finalize'), [])
                    network.release_held()
                    self.drive(network, lambda: not driver.active)
                self.assert_clean_finish(network)
                self.assert_guard_clean(guard)
                self.assertEqual(network.methods().count('flash_s5_release'), 1)
                self.assertEqual(network.methods().count('flash_s5_open'), 1)
                self.assertEqual(guard.named('rpc.submit-after-finalize'), [])

    # -- (b) delayed peer: retained acceptance, no repeat, no early tail -----
    def test_b_delayed_peer_retains_accepted_and_never_repeats(self):
        """Rank 1 completes nothing for 120 polling rounds while rank 0
        completes everything. The engine retries one retained command. Rank 0
        never repeats a submission, copy or continuation, and no rank enters
        the layer's TP tail or any completion-dependent action before both
        ranks reach the boundary."""
        network = SkewNetwork(self)
        guard = CompletionGuard(network)
        driver = network.driver

        def buckets():
            layers = driver.qsa_control.snapshot()['layers']
            return layers[0]['bucket_accepted_commands'] if layers else 0
        self.drive(network, lambda: buckets() >= 1)
        rank0 = network.qsa[0]

        def state(stack):
            return (tuple(sorted(stack.dispatcher.accepted)), dict(stack.operation.stats),
                    len(stack.device.events), len(stack.device.copies),
                    len(stack.device.page_copies), len(stack.bus.events),
                    sum(1 for r in stack.driver.records if r[0] == 'tail'))
        start = len(network.agreements)
        methods = len(network.methods())
        frozen = None
        for _ in range(120):
            network.complete(ranks=(0, 2, 3))
            driver.advance(deliver_outputs=False)
            guard.observe()
            self.assertIsNone(driver.error)
            if frozen is None and any(
                    a['kind'] is LAYER and a['stage'] == 0 and a['rows'][0][0] and a['rows'][1][1]
                    for a in network.agreements[start:]):
                frozen = state(rank0)
            if frozen is not None:
                self.assertEqual(state(rank0), frozen)
            for stack in network.qsa.values():
                self.assertFalse(any(r[0] == 'tail' and r[1] == 0 for r in stack.driver.records))
            self.assertEqual(driver.qsa_control.snapshot()['layers'][0]['tail_accepted_commands'], 0)
            self.assertEqual(guard.named(*CompletionGuard.RELEASE_SCOPE), [])
            self.assertFalse(guard.finalized)
        self.assertIsNotNone(frozen, 'rank 1 never went pending beside an accepted rank 0')
        retries = [a for a in network.agreements[start:]
                   if a['kind'] is LAYER and a['rows'][0][0] and a['rows'][1][1]]
        self.assertGreater(len(retries), 10)
        # Every retry names the one retained command: no new op_seq while pending.
        self.assertEqual(len({a['op_seq'] for a in retries}), 1)
        self.assertNotIn('flash_s5_release', network.methods()[methods:])
        self.drive(network, lambda: not driver.active)
        self.assert_clean_finish(network)
        self.assert_guard_clean(guard)
        for stack in network.qsa.values():
            self.assertEqual(sum(1 for r in stack.driver.records if r[0] == 'tail' and r[1] == 0), 1)

    # -- (c) every phase pair and every field difference ---------------------
    def test_c_every_phase_pair_and_field_difference_refuses(self):
        """Every (kind, stage) agreement of a real transaction, probed on the
        real driver. The only accepted difference is 'commit' beside
        'state_finalize' at STATE_FINALIZE, in both rank orders."""
        cursor = w.CompletionCursor
        pairs = list(itertools.combinations(w.CURSOR_PHASES, 2))
        self.assertEqual(len(pairs), 36)

        def phase(name):
            return lambda v: replace(v, cursor=cursor(name, v.cursor.layer, v.cursor.bucket))

        def same(v):
            return v

        def layer(v):
            return replace(v, cursor=cursor(v.cursor.phase, v.cursor.layer + 1, v.cursor.bucket))

        def bucket(v):
            return replace(v, cursor=cursor(v.cursor.phase, v.cursor.layer, v.cursor.bucket + 1))

        def then(first, second):
            return lambda v: second(first(v))

        for order in (0, 1):
            with self.subTest(order=order):
                network = SkewNetwork(self)
                guard = CompletionGuard(network)
                driver = network.driver
                real = driver._apply_segments
                outcomes = {}
                unexpected = []

                def probe(values):
                    values = tuple(values)
                    command = driver._command
                    targets = tuple(driver._targets)
                    rows = [v for v in values if v.worker_rank in targets]
                    if len(rows) != 2 or not all(v.accepted for v in rows):
                        return real(values)
                    a, b = targets
                    point = (command.kind, command.stage)
                    paused = all(v.paused and not v.profile_boundary and not v.tail_boundary
                                 for v in rows)
                    cases = []
                    if point not in outcomes:
                        outcomes[point] = {}
                        for p, q in pairs:
                            if command.kind is FINALIZE and {p, q} == BENIGN:
                                continue
                            cases.append((('phase', p, q), phase(p), phase(q)))
                        cases += [('layer', same, layer), ('bucket', same, bucket),
                                  ('cursor-none', same, lambda v: replace(v, cursor=None))]
                        if command.kind is FINALIZE:
                            cases += [('commit-sf-layer', phase('commit'),
                                       then(phase('state_finalize'), layer)),
                                      ('commit-sf-bucket', phase('commit'),
                                       then(phase('state_finalize'), bucket))]
                    if paused and ('paused', command.stage) not in outcomes:
                        outcomes[('paused', command.stage)] = {}
                        cases += [
                            ('unpause', same, lambda v: replace(v, paused=False)),
                            ('profile', same, lambda v: replace(v, profile_boundary=True)),
                            ('tail', same, lambda v: replace(v, tail_boundary=True))]
                    for name, ma, mb in cases:
                        key = ('paused', command.stage) if name in ('unpause', 'profile', 'tail') else point
                        tampered = tuple(ma(v) if v.worker_rank == a else
                                         mb(v) if v.worker_rank == b else v for v in values)
                        try:
                            real(tampered)
                        except ed.DriverError as exc:
                            outcomes[key][name] = str(exc)
                        else:
                            unexpected.append((point, name))
                            raise AssertionError('probe %r at %r was accepted' % (name, point))
                    if command.kind is FINALIZE:
                        first, second = (('commit', 'state_finalize')
                                         if (order == 0) == (command.stage == 0)
                                         else ('state_finalize', 'commit'))
                        values = tuple(phase(first)(v) if v.worker_rank == a else
                                       phase(second)(v) if v.worker_rank == b else v
                                       for v in values)
                        outcomes[point]['benign'] = (first, second)
                    return real(values)
                driver._apply_segments = probe
                self.drive(network, lambda: not driver.active)
                self.assertEqual(unexpected, [])
                self.assert_clean_finish(network)
                self.assert_guard_clean(guard)
                points = {p for p in outcomes if p[0] != 'paused'}
                self.assertEqual(points, {
                    (w.SegmentKind.PREPARE, 0), (w.SegmentKind.PREFIX, 0), (LAYER, 0),
                    (w.SegmentKind.SUFFIX, 0), (w.SegmentKind.PP_SEND, 0), (FINALIZE, 0),
                    (w.SegmentKind.PREPARE, 1), (w.SegmentKind.PP_RECEIVE, 1),
                    (w.SegmentKind.PREFIX, 1), (LAYER, 1), (w.SegmentKind.SUFFIX, 1),
                    (FINALIZE, 1)})
                for point in points:
                    kind, stage = point
                    found = outcomes[point]
                    phases = {name[1:] for name in found if type(name) is tuple}
                    expected = {pair for pair in pairs
                                if not (kind is FINALIZE and set(pair) == BENIGN)}
                    self.assertEqual(phases, expected, point)
                    self.assertTrue({'layer', 'bucket', 'cursor-none'} <= set(found), point)
                    for name, message in found.items():
                        if name == 'benign':
                            continue
                        self.assertTrue(message.startswith(CONTINUATION), (point, name, message))
                        self.assert_diagnostic(message, kind, stage)
                    if kind is FINALIZE:
                        self.assertTrue({'commit-sf-layer', 'commit-sf-bucket'} <= set(found))
                        self.assertEqual(set(found['benign']), BENIGN)
                    else:
                        # Off STATE_FINALIZE, commit beside state_finalize refuses.
                        self.assertIn(('phase', 'state_finalize', 'commit'), found)
                paused = outcomes.get(('paused', 0)) or outcomes.get(('paused', 1))
                self.assertIsNotNone(paused, 'no paused QSA layer agreement was probed')
                self.assertTrue(paused['unpause'].startswith(CONTINUATION))
                self.assertTrue(paused['profile'].startswith(
                    'TP workers disagree on the QSA submission boundary'))
                self.assertTrue(paused['tail'].startswith(
                    'TP workers disagree on the QSA tail boundary'))
                for message in paused.values():
                    self.assertIn('[command=layer stage=', message)
                benign = sorted(outcomes[(FINALIZE, s)]['benign'] for s in (0, 1))
                self.assertEqual(sorted(benign), [('commit', 'state_finalize'),
                                                  ('state_finalize', 'commit')])

    def divergent(self, kind, stage, mutate):
        network = SkewNetwork(self)
        driver = network.driver
        real = driver._apply_segments
        done = []

        def tamper(values):
            values = tuple(values)
            command = driver._command
            if (not done and command.kind is kind and command.stage == stage
                    and all(v.accepted for v in values if v.worker_rank in driver._targets)):
                done.append(command.op_seq)
                values = tuple(mutate(v) if v.worker_rank == driver._targets[-1] else v
                               for v in values)
            return real(values)
        driver._apply_segments = tamper
        with self.assertRaises(ed.DriverError) as caught:
            self.drive(network, lambda: not driver.active)
        self.assertTrue(done)
        return str(caught.exception)

    def test_c_divergence_surfaces_from_advance_with_its_diagnostic(self):
        """A real divergence still stops the driver, end to end, with the
        bounded diagnostic in the DriverError text."""
        cursor = w.CompletionCursor
        cases = (
            ('layer-bucket', LAYER, 0, lambda v: replace(
                v, cursor=cursor(v.cursor.phase, v.cursor.layer, v.cursor.bucket + 1))),
            ('finalize-other-phase', FINALIZE, 0, lambda v: replace(
                v, cursor=cursor('layer', v.cursor.layer, v.cursor.bucket))),
            ('finalize-other-layer', FINALIZE, 1, lambda v: replace(
                v, cursor=cursor('commit', v.cursor.layer + 1, v.cursor.bucket))),
            ('suffix-commit', w.SegmentKind.SUFFIX, 0, lambda v: replace(
                v, cursor=cursor('commit', v.cursor.layer, v.cursor.bucket))),
        )
        for name, kind, stage, mutate in cases:
            with self.subTest(name):
                message = self.divergent(kind, stage, mutate)
                self.assertTrue(message.startswith(CONTINUATION), message)
                self.assert_diagnostic(message, kind, stage)

    # -- (d) bounded random sweep -------------------------------------------
    def sweep(self, seed, parent):
        """In-order partial per-rank completion, idle-stream completion at
        record, and progress between the per-rank deliveries of one round."""
        rnd = random.Random(4100 + seed)
        network = SkewNetwork(self)
        guard = CompletionGuard(network)
        rate = {rank: rnd.choice((1.0, 0.6, 0.3, 0.1)) for rank in range(4)}

        def progress(rank):
            for events in [bus.events for bus in network.buses(rank)]:
                pending = [e for e in events if not e.complete]
                count = sum(rnd.random() < rate[rank] for _ in pending)
                for event in pending[:count]:
                    event.complete = True
            if rank in network.qsa:
                device = network.qsa[rank].device
                for token in {e.token for e in device.events}:
                    pending = [e for e in device.events if e.token == token and not e.ready]
                    count = sum(rnd.random() < rate[rank] for _ in pending)
                    for event in pending[:count]:
                        event.ready = True

        def tick():
            for rank in range(4):
                progress(rank)
        for rank in range(4):
            def hook(command, call, rank=rank):
                for other in range(4):
                    if rnd.random() < 0.5:
                        progress(other)
                if rnd.random() < 0.5:
                    network.idle.update(id(bus) for bus in network.buses(rank))
                try:
                    return call()
                finally:
                    network.idle.clear()
            network.wrap_submit(rank, hook)
        key = PARENT_KEY if parent else ed._continuation
        error = None
        with idle_streams(network), mock.patch.object(ed, '_continuation', key):
            try:
                self.drive(network, lambda: not network.driver.active, limit=4000, complete=tick)
            except ed.DriverError as exc:
                error = str(exc)
        return network, guard, error

    def test_d_random_completion_skew_sweep(self):
        mixed = 0
        refused = 0
        for seed in range(32):
            with self.subTest(seed=seed):
                network, guard, error = self.sweep(seed, parent=False)
                self.assertIsNone(error)
                self.assert_clean_finish(network)
                self.assert_guard_clean(guard)
                mixed += sum(1 for a in network.agreements if a['kind'] is FINALIZE
                             and len({row[3].phase for row in a['rows'].values()}) > 1)
                _, parent_guard, parent_error = self.sweep(seed, parent=True)
                self.assertEqual(parent_guard.violations, [])
                if parent_error is not None:
                    self.assertTrue(parent_error.startswith(CONTINUATION), parent_error)
                    self.assertIn('[command=state_finalize ', parent_error)
                    refused += 1
        # The sweep exercised the skew the box hit, and the parent key refuses it.
        self.assertGreater(mixed, 0)
        self.assertGreater(refused, 0)


if __name__ == '__main__':
    unittest.main()
