"""CPU-only tests for the E3 decode-only diagnostic probe.

No torch, no vLLM, no CUDA, no engine. The driver, torch, the stock cudagraph
manager and the device events are fakes with explicit boundaries. These tests
exercise construction, capture/replay/transition ordering, invalidation,
quarantine, counters, inventory and generated-package bytes. They cannot
establish native readiness, SM containment, GPU progress or a performance
result; the full package stays ``armable: False``.
"""
import contextlib
import inspect
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import decode_probe  # noqa: E402
import green_context  # noqa: E402
import graph_hooks  # noqa: E402
import native_startup  # noqa: E402
import s5_startup  # noqa: E402
import worker_hooks  # noqa: E402
from decode_probe import (  # noqa: E402
    EVIDENCE_KEYS,
    REQUIRED_TESTS,
    DecodeProbeError,
    DecodeProbeInvalidated,
    DecodeProbeNotReady,
    DecodeProbeQuarantined,
    mode_settings,
    require_decode_probe_ready,
)
from graph_router import (  # noqa: E402
    DecodeProbeRouter,
    ExecutionDescriptor,
    GraphRoutingError,
    NoPrefillRoute,
)
from test_e3_green_native import (  # noqa: E402
    FULL,
    FakeBinding,
    FakeDesc,
    FakeEvent,
    FakeManager,
    fake_runner,
    fake_torch,
    install_fake_vllm,
    remove_fake_vllm,
)

SMS = 34
PROBE_ENV = decode_probe.PROBE_ENV
DECODE_SMS_ENV = decode_probe.DECODE_SMS_ENV


def desc(num_tokens=8, num_reqs=None, uniform=1, mode='full', ubatches=1):
    return ExecutionDescriptor(mode, num_tokens,
                               num_tokens if num_reqs is None else num_reqs,
                               uniform, None, 0, ubatches)


class ProbeFixture(unittest.TestCase):
    """One fake worker with the split created and the probe installed."""

    def setUp(self):
        FakeManager.instances.clear()
        self.graph_contexts = []
        install_fake_vllm(self.graph_contexts)
        self.torch = fake_torch()
        self.runner = fake_runner(self.torch)
        self.binding = FakeBinding()
        self.split = green_context.open_split(
            0, env={DECODE_SMS_ENV: str(SMS)},
            torch_module=self.torch, binding=self.binding)
        self.runner._flash_s5_split = self.split
        self.extra_splits = []
        self.env = {PROBE_ENV: '1', 'FLASH_S5_OVERLAP': '0', DECODE_SMS_ENV: str(SMS)}
        self.probe = native_startup.install_decode_probe(
            self.runner, env=self.env, torch_module=self.torch)

    def tearDown(self):
        for split in [self.split, *self.extra_splits]:
            if split.state == 'open':
                split.close()
        remove_fake_vllm()

    def new_split(self):
        split = green_context.open_split(
            0, env={DECODE_SMS_ENV: str(SMS)},
            torch_module=self.torch, binding=FakeBinding())
        self.extra_splits.append(split)
        return split

    @property
    def controller(self):
        return self.runner._flash_s5_route

    @property
    def stock(self):
        return self.runner.cudagraph_manager


# --- flags ------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    def test_both_flags_refused_before_allocation(self):
        env = {PROBE_ENV: '1', 'FLASH_S5_OVERLAP': '1', DECODE_SMS_ENV: '44'}
        with self.assertRaises(DecodeProbeError):
            mode_settings(env)
        allocated = []
        model_runner = types.SimpleNamespace()
        worker = types.SimpleNamespace(model_runner=model_runner)
        with patch.object(green_context, 'open_split',
                          side_effect=lambda *a, **k: allocated.append(1)):
            with self.assertRaises(DecodeProbeError):
                worker_hooks.after_runner_init(worker, env=env)
        self.assertEqual(allocated, [], 'the conflicting flags allocated nothing')
        self.assertIsNone(getattr(model_runner, '_flash_s5_split', None))
        runner = types.SimpleNamespace()
        with patch.object(native_startup, 'install_split_bank',
                          side_effect=AssertionError('allocated the split bank')):
            with self.assertRaises(DecodeProbeError):
                native_startup.install_decode_probe(runner, env=env, torch_module=object())
        with self.assertRaises(DecodeProbeError):
            s5_startup.install_builder(types.SimpleNamespace(), None, env=env)
        self.assertIsNone(getattr(runner, '_flash_s5_split_bank', None))

    def test_probe_declares_its_own_settings(self):
        self.assertIsNone(mode_settings({}))
        self.assertIsNone(mode_settings({PROBE_ENV: '0', 'FLASH_S5_OVERLAP': '0'}))
        self.assertIsNone(mode_settings({'FLASH_S5_OVERLAP': '1'}))
        with self.assertRaises(green_context.GreenContextError):
            mode_settings({PROBE_ENV: '1'})
        with self.assertRaises(DecodeProbeError):
            mode_settings({PROBE_ENV: 'maybe'})
        settings = mode_settings({PROBE_ENV: '1', DECODE_SMS_ENV: '44'})
        self.assertEqual(settings.decode_sms, 44)
        self.assertEqual(settings.storage_epoch, decode_probe.STORAGE_EPOCH)
        with self.assertRaises(DecodeProbeError):
            decode_probe.DecodeProbeSettings(0)

    def test_no_environment_bypass(self):
        # The gate takes evidence only; there is no unsafe/bypass flag to set.
        self.assertEqual(set(inspect.signature(require_decode_probe_ready).parameters),
                         {'evidence'})
        names = {name for name in dir(decode_probe) if name.isupper()}
        self.assertNotIn('BYPASS', ' '.join(names))
        self.assertNotIn('UNSAFE', ' '.join(names))


# --- gate -------------------------------------------------------------------

class GateTests(unittest.TestCase):
    def test_require_decode_probe_ready_requires_complete_evidence(self):
        evidence = require_decode_probe_ready()
        self.assertEqual(set(evidence), set(EVIDENCE_KEYS))
        self.assertEqual(evidence['helper_source_sha256'],
                         green_context.helper_source_sha256())
        self.assertIn('decode_probe', evidence['generated_source_shas'])
        self.assertEqual(evidence['pins']['v1/worker/gpu_worker.py'],
                         worker_hooks.PINS['v1/worker/gpu_worker.py'])
        self.assertEqual(evidence['pins']['v1/worker/gpu/model_runner.py'],
                         graph_hooks.PINNED_SHA256)

        for key in EVIDENCE_KEYS:
            with self.subTest(missing=key):
                broken = dict(evidence)
                del broken[key]
                with self.assertRaises(DecodeProbeNotReady):
                    require_decode_probe_ready(broken)

        def tamper(field, mutate):
            copy = dict(evidence)
            copy[field] = mutate(dict(evidence[field])
                                 if isinstance(evidence[field], dict)
                                 else evidence[field])
            with self.assertRaises(DecodeProbeNotReady):
                require_decode_probe_ready(copy)

        tamper('pins', lambda pins: pins.update({'v1/worker/gpu_worker.py': '0' * 64}) or pins)
        tamper('pins', lambda pins: pins.update({'unknown.py': '0' * 64}) or pins)
        tamper('helper_source_sha256', lambda _: '0' * 64)
        tamper('generated_source_shas',
               lambda shas: shas.update({'decode_probe': '0' * 64}) or shas)
        tamper('generated_source_shas', lambda shas: shas.pop('graph_hooks') and shas)
        tamper('tests', lambda _: ('other.Test',))
        with self.assertRaises(DecodeProbeNotReady):
            require_decode_probe_ready('not-a-mapping')

    def test_declared_test_names_exist(self):
        module = sys.modules[__name__]
        for name in REQUIRED_TESTS:
            owner, method = name.split('.')
            cls = getattr(module, owner)
            self.assertTrue(issubclass(cls, unittest.TestCase), name)
            self.assertTrue(callable(getattr(cls, method, None)), name)

    def test_native_gate_unchanged(self):
        with self.assertRaises(s5_startup.StartupError):
            s5_startup.require_native_ready()
        self.assertIs(s5_startup.require_decode_probe_ready, require_decode_probe_ready)
        runner = types.SimpleNamespace()
        self.assertIsNone(s5_startup.install_builder(runner, None, env={
            PROBE_ENV: '1', 'FLASH_S5_OVERLAP': '0', DECODE_SMS_ENV: '44'}))
        self.assertIsNone(getattr(runner, '_flash_e3_build_state', None))
        with self.assertRaises(s5_startup.StartupError):
            s5_startup.install_builder(runner, None, env={'FLASH_S5_OVERLAP': '1'})
        self.assertIsNone(s5_startup.install_engine(
            types.SimpleNamespace(), env={PROBE_ENV: '1', DECODE_SMS_ENV: '44'}))


# --- typed route guard ------------------------------------------------------

class RouteGuardTests(ProbeFixture):
    def test_no_prefill_guard_and_captured_descriptor_routing(self):
        output = types.SimpleNamespace()
        self.assertIsNone(graph_hooks.check_route(
            self.runner, output, FakeDesc(FULL, 8, 8), 8, 8, 1))
        self.assertEqual(self.stock.replayed, [], 'full route keeps the stock manager')
        with self.assertRaises(GraphRoutingError):
            graph_hooks._transaction(
                types.SimpleNamespace(flash_e3_transaction=object()), self.probe.router)

        record = self.probe.transition('split')
        self.assertEqual(record.drained_streams, 3)
        decision = graph_hooks.check_route(
            self.runner, types.SimpleNamespace(), FakeDesc(FULL, 8, 8), 8, 8, 1)
        self.assertEqual(decision.mode, 'split')
        self.assertEqual(decision.descriptor, desc(8, 8, 1))
        guard = self.probe.router.no_prefill_guard()
        self.assertIsInstance(guard, NoPrefillRoute)
        self.assertEqual(guard.route_generation, 1)
        self.assertEqual(guard.storage_epoch, decode_probe.STORAGE_EPOCH)
        with self.assertRaises(GraphRoutingError):
            self.probe.router.supported_route(types.SimpleNamespace(), desc(8, 8, 1))
        self.probe.transition('full')
        self.assertEqual(self.probe.route, 'full')

    def test_unsupported_descriptor_invalidates_without_fallback(self):
        reasons = {
            'non_full_graph': desc(8, 8, None, mode='piecewise'),
            'empty_batch': desc(0, 0, 1),
            'prefill_or_mixed_batch': desc(8, 2, 4),
            'multi_ubatch': desc(8, 8, 1, ubatches=2),
        }
        for reason, descriptor in reasons.items():
            self.assertEqual(self.probe._unsupported_reason(descriptor), reason)
        self.assertIsNone(self.probe._unsupported_reason(desc(8, 8, 1)))
        self.assertIsNone(self.probe._unsupported_reason(desc(8, 8, None)))

        self.probe.transition('split')
        with self.assertRaises(DecodeProbeInvalidated):
            graph_hooks.check_route(self.runner, types.SimpleNamespace(),
                                    FakeDesc(FULL, 16, 16), 16, 16, 1)
        counters = self.probe.counters()
        self.assertEqual(counters['interval_invalidations'], 1)
        self.assertEqual(counters['invalidation_reasons'], ['unsupported_descriptor'])
        self.assertEqual(counters['interval'], 'invalidated')
        self.assertEqual(self.stock.replayed, [])
        with self.assertRaises(DecodeProbeInvalidated):
            graph_hooks.replay_fullgraph(self.runner, types.SimpleNamespace(),
                                         FakeDesc(FULL, 16, 16),
                                         self.stock.run_fullgraph)
        self.assertEqual(self.stock.replayed, [],
                         'an unsupported dispatch never falls back to the full GPU bank')
        with self.assertRaises(DecodeProbeInvalidated):
            graph_hooks.check_route(self.runner, types.SimpleNamespace(),
                                    FakeDesc(FULL, 0, 0), 0, 0, 1)

    def test_capture_replay_and_warmup_through_the_split_bank(self):
        self.assertEqual(len(self.probe.router.handles), 2)
        self.probe.transition('split')
        with self.torch.cuda.stream(self.split.decode_stream):
            output = self.probe.replay(desc(8, 8, 1))
        self.assertEqual(output, ('hidden', 8))
        self.assertEqual(self.stock.replayed, [])
        counters = self.probe.counters()
        self.assertEqual(counters['replays'], 1)
        self.assertEqual(counters['replays_by_bank'], {'decode-split': 1})
        self.assertEqual(counters['replays_by_stream'], {self.split.decode_stream_token: 1})
        self.assertEqual(list(counters['replays_by_descriptor'].values()), [1])
        self.assertEqual(counters['warmup_replays'], 0)
        with self.torch.cuda.stream(self.split.decode_stream):
            self.assertEqual(self.probe.warmup(), 2)
        self.assertEqual(self.probe.counters()['warmup_replays'], 2)
        self.assertEqual(self.probe.counters()['replays'], 1,
                         'warmup replays are counted apart from measured replays')


# --- bank wiring ------------------------------------------------------------

class BankWiringTests(ProbeFixture):
    def test_separate_pool_and_measured_bytes(self):
        bank = self.runner._flash_s5_split_bank
        self.assertIsNot(bank.manager, self.stock)
        self.assertNotEqual(bank.pool, self.stock.pool)
        self.assertIsNotNone(bank.pool)
        self.assertEqual(bank.measured_bytes, 4096)
        self.assertEqual(self.probe.router.state, 'ready')
        inventory = self.probe.inventory()
        self.assertEqual(inventory['split_pool_bytes'], 4096)
        self.assertEqual(inventory['split_pool_basis'], 'measured')
        self.assertEqual(inventory['descriptors_captured'], 2)
        self.assertEqual(inventory['split_pool_id'], self.probe.router.context.pool_id)
        self.assertEqual(inventory['split_pool_context_id'], self.split.decode_context_id)
        self.assertEqual(inventory['storage_epoch'], decode_probe.STORAGE_EPOCH)
        self.assertEqual(inventory['route_generation'], 0)
        self.assertIn('prefill lane', inventory['excluded'])
        for name in ('_flash_s5_native', '_flash_s5_comms', '_flash_s5_ple_scratch',
                     '_flash_s5_lane_forward', '_flash_s5_dispatcher'):
            self.assertIsNone(getattr(self.runner, name, None), name)

    def test_install_needs_the_declared_split_and_no_lane(self):
        with patch.object(native_startup, 'install_worker_native',
                          side_effect=AssertionError('native lane installed')):
            runner = fake_runner(self.torch)
            with self.assertRaises(native_startup.NativeStartupError):
                native_startup.install_decode_probe(runner, env=self.env,
                                                    torch_module=self.torch)
            runner._flash_s5_split = self.new_split()
            with self.assertRaises(native_startup.NativeStartupError):
                native_startup.install_decode_probe(
                    runner, env=dict(self.env, **{DECODE_SMS_ENV: '44'}),
                    torch_module=self.torch)
            runner._flash_s5_native = object()
            with self.assertRaises(native_startup.NativeStartupError):
                native_startup.install_decode_probe(runner, env=self.env,
                                                    torch_module=self.torch)
        with self.assertRaises(native_startup.NativeStartupError):
            native_startup.install_decode_probe(self.runner, env=self.env,
                                                torch_module=self.torch)
        with self.assertRaises(green_context.GreenContextError):
            native_startup.install_decode_probe(fake_runner(self.torch), env={
                PROBE_ENV: '1', 'FLASH_S5_OVERLAP': '0'}, torch_module=self.torch)

    def test_capture_banks_installs_the_probe_at_capture_time(self):
        runner = fake_runner(self.torch)
        runner._flash_s5_split = self.new_split()
        with patch.dict(os.environ, self.env), patch.object(
                native_startup, '_torch', lambda: self.torch):
            router = graph_hooks.capture_banks(runner)
        self.assertIsInstance(router, DecodeProbeRouter)
        self.assertIs(runner._flash_e3_graph_router, router)
        self.assertIsNotNone(runner._flash_s5_decode_probe)
        self.assertTrue(runner._flash_e3_graph_captured)
        self.assertEqual(runner._flash_s5_split_bank.measured_bytes, 4096)
        with self.assertRaises(GraphRoutingError):
            graph_hooks.capture_banks(runner)

    def test_split_descriptors_are_the_stock_full_descriptors(self):
        self.assertEqual(self.runner._flash_s5_split_descriptors, (desc(8, 8, 1), desc(4, 4, 1)))
        self.assertEqual(self.stock.replayed, [])


# --- transitions ------------------------------------------------------------

class TransitionTests(ProbeFixture):
    def test_drained_transitions_and_refusal_while_unresolved(self):
        seen = []

        class TrackingEvent(FakeEvent):
            def record(self, stream=None):
                seen.append(stream)
                return super().record(stream)

        self.torch.cuda.Event = TrackingEvent
        record = self.probe.transition('split')
        self.assertEqual(record.drained_streams, len(self.controller.consumer_streams()))
        self.assertEqual(len(seen), record.drained_streams)
        for stream in (self.controller.full_main_stream,
                       self.runner.output_copy_stream,
                       self.runner.pp_handler.broadcast_stream):
            self.assertIn(stream, seen)
        self.assertIs(self.runner.main_stream, self.split.decode_stream)
        self.assertIs(self.runner.pp_handler.main_stream, self.split.decode_stream)
        self.assertIs(self.runner._flash_s5_decode_stream, self.split.decode_stream)
        self.assertEqual(self.controller.route_generation, 1)
        self.assertEqual(self.probe.router.route_generation, 1)
        self.assertEqual(self.probe.interval, 'active')
        self.assertEqual(self.probe.counters()['interval_index'], 1)
        with self.assertRaises(DecodeProbeError):
            self.probe.transition('split')
        self.assertIsNone(self.probe.quarantined)

        self.probe.in_flight = 1
        with self.assertRaises(DecodeProbeError):
            self.probe.transition('full')
        self.assertEqual(self.probe.route, 'split')
        self.assertIsNone(self.probe.quarantined)
        self.probe.in_flight = 0

        self.controller.admitting = False
        with self.assertRaises(DecodeProbeError):
            self.probe.transition('full')
        self.assertEqual(self.probe.route, 'split')
        self.controller.admitting = True

        record = self.probe.transition('full')
        self.assertEqual(record.drained_streams, 3)
        self.assertIs(self.runner.main_stream, self.controller.full_main_stream)
        self.assertIsNone(self.runner._flash_s5_decode_stream)
        self.assertEqual(self.probe.route_generation, 2)
        self.assertEqual(self.probe.counters()['transition_count'], 2)
        self.assertEqual(len(self.probe.counters()['transition_latencies_s']), 2)

    def test_failed_transition_quarantines_and_keeps_route(self):
        self.controller.drain = lambda: (_ for _ in ()).throw(RuntimeError('drain failed'))
        with self.assertRaises(RuntimeError):
            self.probe.transition('split')
        self.assertEqual(self.controller.route, 'full')
        self.assertIs(self.runner.main_stream, self.controller.full_main_stream)
        self.assertIsNone(self.runner._flash_s5_decode_stream)
        self.assertIn('transition to split failed', self.probe.quarantined)
        kinds = [event.kind for event in self.probe.events]
        self.assertIn('transition_failed', kinds)
        self.assertEqual(self.probe.counters()['quarantined'], self.probe.quarantined)
        with self.assertRaises(DecodeProbeQuarantined):
            self.probe.transition('split')
        # Decode continues on the route it was on: the stock full path.
        self.assertIsNone(self.probe.check_route(desc(8, 8, 1)))
        self.assertEqual(self.probe.route_generation, 0)

    def test_probe_transition_rpc_validates_and_acks_after_draining(self):
        base = {'engine_epoch': 3, 'expected_route_generation': 0, 'route_generation': 1,
                'target': native_startup.SPLIT_BANK_NAME, 'storage_epoch': 1}
        with self.assertRaises(native_startup.NativeStartupError):
            native_startup.probe_transition_rpc(
                self.runner, dict(base, engine_epoch=4), engine_epoch=3)
        with self.assertRaises(native_startup.NativeStartupError):
            native_startup.probe_transition_rpc(
                self.runner, dict(base, expected_route_generation=5))
        with self.assertRaises(native_startup.NativeStartupError):
            native_startup.probe_transition_rpc(
                self.runner, dict(base, route_generation=2))
        with self.assertRaises(native_startup.NativeStartupError):
            native_startup.probe_transition_rpc(
                self.runner, dict(base, storage_epoch=9))
        with self.assertRaises(native_startup.NativeStartupError):
            native_startup.probe_transition_rpc(
                self.runner, dict(base, authorization={'kind': 'x'}))
        with self.assertRaises(native_startup.NativeStartupError):
            native_startup.probe_transition_rpc(self.runner, {'target': 'split'})
        self.assertEqual(self.probe.route, 'full')

        ack = native_startup.probe_transition_rpc(self.runner, base, engine_epoch=3)
        self.assertEqual(ack['ack'], 'flash_s5_transition')
        self.assertEqual(ack['mode'], 'decode-only-probe')
        self.assertEqual((ack['route'], ack['route_generation']), ('split', 1))
        self.assertEqual(ack['drained_streams'], 3)
        self.assertEqual(ack['storage_epoch'], decode_probe.STORAGE_EPOCH)
        leave = {'engine_epoch': 3, 'expected_route_generation': 1, 'route_generation': 2,
                 'target': native_startup.FULL_ROUTE, 'storage_epoch': 1}
        ack = native_startup.probe_transition_rpc(self.runner, leave, engine_epoch=3)
        self.assertEqual((ack['route'], ack['route_generation']), ('full', 2))


# --- inventory and shutdown -------------------------------------------------

class ShutdownTests(ProbeFixture):
    def test_inventory_helper_sha_and_shutdown_order(self):
        inventory = self.runner._flash_s5_decode_probe_inventory
        self.assertEqual(inventory['mode'], 'decode-only-probe')
        self.assertEqual(inventory['helper_source_sha256'],
                         green_context.helper_source_sha256())
        self.assertEqual(inventory['library_sha256'], 'f' * 64)
        self.assertEqual(inventory['split_pool_bytes'], 4096)
        self.assertIs(inventory['split_pool_bytes'], self.runner._flash_s5_split_bank.measured_bytes)
        self.assertEqual(inventory['communicators'], 'none')
        self.assertEqual(inventory['pp_channel_bytes'], 0)
        self.assertEqual(inventory['ple_scratch_bytes'], 0)
        self.assertEqual(len(inventory['descriptors']), 2)
        self.assertIs(self.runner._flash_s5_inventory.__self__, self.probe)
        self.assertIs(self.runner._flash_s5_decode_probe_counters.__self__, self.probe)

        self.probe.transition('split')
        record = graph_hooks.shutdown_banks(self.runner)
        self.assertEqual(record['order'][0], 'leave-split')
        self.assertEqual(record['order'][1], 'drain:3')
        self.assertEqual(record['order'][2:4], ('graph', 'graph'))
        self.assertEqual(record['order'][-2:], ('bank', 'split'))
        self.assertEqual(record['graphs'], 2)
        self.assertEqual(record['route'], 'full')
        self.assertEqual(self.split.state, 'closed')
        self.assertEqual(self.controller.route, 'full')
        self.assertEqual(self.probe.router.state, 'destroyed')
        self.assertEqual(self.runner._flash_s5_split_bank.handles, {})
        self.assertIsNone(self.runner._flash_s5_split_bank.manager)
        self.assertIsNone(self.runner._flash_s5_split_bank.pool)
        self.assertIs(graph_hooks.shutdown_banks(self.runner), None)
        self.assertEqual(self.probe.shutdown()['order'], record['order'])

    def test_shutdown_without_entering_split_still_closes_bank_and_split(self):
        record = self.probe.shutdown()
        self.assertEqual(record['order'][0], 'drain:3')
        self.assertNotIn('leave-split', record['order'])
        self.assertEqual(record['route'], 'full')
        self.assertEqual(self.split.state, 'closed')
        self.assertEqual(self.binding.closed, self.binding.created)


# --- default-off byte path --------------------------------------------------

class BytePathTests(unittest.TestCase):
    def test_default_off_generated_sources_are_probe_free(self):
        from test_e3_package import SyntheticTests

        import make_s5_overlap as g

        def build_once(env, prune=False):
            case = SyntheticTests('test_composes_new_package_and_preserves_baseline_bytes')
            root, base, ref, whole, _core = case.fixture()
            stack = case.stubs(base, ref)
            try:
                stack.__enter__()
                patches = [patch.dict(os.environ, env, clear=False)]
                if prune:
                    patches.append(patch.object(
                        g, 'E3_MODULES',
                        tuple(m for m in g.E3_MODULES if m != 'decode_probe')))
                    patches.append(patch.object(
                        g, '_decode_probe_declaration', lambda files: {'declared': False}))
                with contextlib.ExitStack() as inner:
                    for item in patches:
                        inner.enter_context(item)
                    result = g.build(base, root / 'out', ref, whole)
            finally:
                stack.__exit__(None, None, None)
            return root / 'out', result

        off = {PROBE_ENV: '0', 'FLASH_S5_OVERLAP': '0'}
        out, _result = build_once(off)
        pruned, _result = build_once({}, prune=True)

        # Composition bytes are exactly the pre-probe output: adding the
        # diagnostic module changes no other generated file.
        for name in ('model_runner_v2.py', 'qsa.py', 'ops_qsa.py',
                     'e3_sources/executor.py', 'e3_sources/core.py',
                     'e3_sources/worker.py', 'e3_sources/cudagraph_utils.py',
                     'e3_sources/model.py', 'E2-GENERATION.json'):
            self.assertEqual((out / name).read_bytes(), (pruned / name).read_bytes(), name)
        probe_importers = {
            name for name in g.E3_MODULES
            if 'from decode_probe import' in
            (ROOT / 'patches/s5_engine' / (name + '.py')).read_text()}
        self.assertEqual(probe_importers, {'graph_hooks', 'native_startup',
                                           's5_startup', 'worker_hooks'})
        for name in g.E3_MODULES:
            if name == 'decode_probe' or name in probe_importers:
                continue
            self.assertEqual((out / 'e3' / (name + '.py')).read_bytes(),
                             (pruned / 'e3' / (name + '.py')).read_bytes(), name)
        # Only the package's relative-import rewrite differs in the modules
        # that consult the diagnostic mode.
        self.assertIn('from .decode_probe import',
                      (out / 'e3/s5_startup.py').read_text())
        self.assertIn('from decode_probe import',
                      (pruned / 'e3/s5_startup.py').read_text())
        self.assertTrue((out / 'e3/decode_probe.py').is_file())
        self.assertFalse((pruned / 'e3/decode_probe.py').exists())

        # The transformed runner/worker sources carry no probe-only code path.
        for name in ('model_runner_v2.py', 'qsa.py', 'ops_qsa.py',
                     'e3_sources/executor.py', 'e3_sources/core.py',
                     'e3_sources/worker.py', 'e3_sources/cudagraph_utils.py',
                     'e3_sources/model.py'):
            text = (out / name).read_text()
            self.assertNotIn('FLASH_S5_DECODE_ONLY_PROBE', text, name)
            self.assertNotIn('install_decode_probe', text, name)

        record = json.loads((out / 'S5-OVERLAP-GENERATION.json').read_text())
        self.assertIn('e3/decode_probe.py', (out / 'run-flash-s5-test.sh').read_text())
        self.assertIs(record['armable'], False)
        self.assertEqual(record['default_feature'], 'FLASH_S5_OVERLAP=0')
        self.assertEqual(record['installed_settings'][PROBE_ENV], '0')
        declaration = record['diagnostic_modes']['decode_only']
        self.assertTrue(declaration['declared'])
        self.assertTrue(declaration['diagnostic_only'])
        self.assertEqual(declaration['qualifies'], 'diagnostic_modes.decode_only')
        self.assertEqual(declaration['gate'], 's5_startup.require_decode_probe_ready')
        self.assertEqual(declaration['startup'], 'native_startup.install_decode_probe')
        self.assertIn('prefill lane', declaration['excluded'])
        self.assertIn('does not qualify', declaration['scope'])
        evidence = declaration['evidence']
        self.assertEqual(set(evidence), set(EVIDENCE_KEYS))
        self.assertEqual(list(evidence['tests']), list(REQUIRED_TESTS))
        self.assertEqual(len(evidence['pins']['v1/worker/gpu_worker.py']), 64)
        self.assertEqual(evidence['helper_source_sha256'],
                         green_context.helper_source_sha256())
        self.assertEqual(set(evidence['generated_source_shas']),
                         set(decode_probe.GENERATED_SOURCES))

        # The runtime seams are inert with both flags unset.
        self.assertIsNone(mode_settings(off))
        self.assertFalse(worker_hooks._split_enabled(off))
        worker = types.SimpleNamespace(model_runner=types.SimpleNamespace())
        with patch.object(green_context, 'open_split',
                          side_effect=AssertionError('allocated a split')):
            self.assertIsNone(worker_hooks.after_runner_init(worker, env=off))
        quiet = types.SimpleNamespace()
        with patch.dict(os.environ, off):
            self.assertIsNone(graph_hooks.capture_banks(quiet))
            self.assertIsNone(graph_hooks.shutdown_banks(quiet))


if __name__ == '__main__':
    unittest.main()
