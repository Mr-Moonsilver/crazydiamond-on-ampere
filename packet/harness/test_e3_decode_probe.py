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
import tempfile
import subprocess
import shutil
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



def fake_evidence(test):
    root = Path(tempfile.mkdtemp(prefix='probe-evidence-'))
    # Preserve fixture artifacts; no worktree/runtime manifest is mutated.
    source = ROOT / 'patches/s5_engine'
    for name in (*decode_probe.GENERATED_SOURCES, *decode_probe.BUILD_ONLY_SOURCES):
        shutil.copyfile(source / (name + '.py'), root / (name + '.py'))
    (root / decode_probe.EVIDENCE_FILE).write_text(json.dumps(decode_probe.declared_evidence()))
    fixture = patch.object(decode_probe, '_module_dir', lambda: str(root))
    fixture.start()
    test.addCleanup(fixture.stop)
    return root


def batch_state(n, prefill=False):
    return types.SimpleNamespace(has_prefill=prefill, num_scheduled_tokens=[1]*n,
                                 is_prefilling_np=[prefill]*n)

class ProbeFixture(unittest.TestCase):
    """One fake worker with the split created and the probe installed."""

    def setUp(self):
        fake_evidence(self)
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
        self.env = {PROBE_ENV: '1', 'FLASH_S5_OVERLAP': '0', DECODE_SMS_ENV: str(SMS),
                    'FLASH_S5_GREEN_LIB_SHA256': 'f' * 64}
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
    def setUp(self):
        self.evidence_root = fake_evidence(self)

    def test_require_decode_probe_ready_requires_complete_evidence(self):
        evidence = require_decode_probe_ready()
        self.assertEqual(set(evidence), set(EVIDENCE_KEYS))
        self.assertEqual(evidence['helper_source_sha256'],
                         green_context.helper_source_sha256())
        self.assertIn('decode_probe', evidence['generated_source_shas'])
        self.assertNotIn('operation_source', evidence['generated_source_shas'])
        self.assertEqual(set(evidence['build_only_source_shas']),
                         set(decode_probe.BUILD_ONLY_SOURCES))
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
        tamper('build_only_source_shas', lambda shas: shas.update({'operation_source': '0' * 64}) or shas)
        tamper('build_only_source_shas', lambda shas: shas.pop('operation_source') and shas)
        tamper('build_only_source_shas', lambda shas: shas.update({'decode_probe': '0' * 64}) or shas)
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
        self.assertIsNone(s5_startup.install_builder(runner, None, is_profiling=True, env={
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
            self.runner, output, FakeDesc(FULL, 8, 8), 8, 8, 1, batch_state(8)))
        self.assertEqual(self.stock.replayed, [], 'full route keeps the stock manager')
        with self.assertRaises(GraphRoutingError):
            graph_hooks._transaction(
                types.SimpleNamespace(flash_e3_transaction=object()), self.probe.router)

        record = self.probe.transition('split')
        self.assertEqual(record.drained_streams, 3)
        decision = graph_hooks.check_route(
            self.runner, types.SimpleNamespace(), FakeDesc(FULL, 8, 8), 8, 8, 1, batch_state(8))
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
                                    FakeDesc(FULL, 16, 16), 16, 16, 1, batch_state(16))
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
        self.assertEqual(self.probe.warmup(), 2)
        self.probe.transition('split')
        with self.torch.cuda.stream(self.split.decode_stream):
            output = self.probe.replay(desc(8, 8, 1))
        self.assertEqual(output, ('hidden', 8))
        self.assertEqual(self.stock.replayed, [])
        counters = self.probe.counters()
        self.assertEqual(counters['replays'], 1)
        self.assertEqual(counters['replays_by_bank'], {'decode-split': 1})
        self.assertEqual(counters['replays_by_stream'], {'decode-split-stream': 1})
        self.assertTrue(counters['replay_streams_match_owner'])
        self.assertEqual(list(counters['replays_by_descriptor'].values()), [1])
        self.assertEqual(counters['warmup_replays'], 2)
        with self.assertRaises(DecodeProbeError):
            self.probe.warmup()
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
        self.assertIn('device-free delta', inventory['split_pool_basis'])
        self.assertEqual(inventory['descriptors_captured'], 2)
        self.assertEqual(inventory['split_pool_label'], 'decode-split-pool')
        self.assertEqual(inventory['split_context_label'], 'decode-split-context')
        self.assertNotIn('split_pool_context_id', inventory)
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
        alloc = types.ModuleType('vllm.distributed.device_communicators.pynccl_allocator')
        restored = []
        alloc.set_graph_pool_id = restored.append
        alloc._graph_pool_id = 'previous-active-pool'
        alloc_parent = types.ModuleType('vllm.distributed.device_communicators')
        alloc_parent.pynccl_allocator = alloc
        with patch.dict(sys.modules, {alloc.__name__: alloc, alloc_parent.__name__:alloc_parent}), patch.dict(os.environ, self.env), patch.object(
                native_startup, '_torch', lambda: self.torch):
            router = graph_hooks.capture_banks(runner)
        self.assertEqual(restored, ['previous-active-pool'])
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
        # A failed switch cannot silently resume a measured full-bank interval.
        with self.assertRaises(DecodeProbeQuarantined):
            self.probe.check_route(desc(8, 8, 1))
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
        self.assertEqual(inventory['communicators'], 'stock TP/PP retained; no additional prefill communicators')
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

        case = SyntheticTests('test_composes_new_package_and_preserves_baseline_bytes')
        root, base, ref, whole, _core = case.fixture()
        with case.stubs(base, ref):
            g.build(base, root / 'out', ref, whole)
        out = root / 'out'
        off = {PROBE_ENV: '0', 'FLASH_S5_OVERLAP': '0'}
        # Exact inverse is tested against real pinned files by package/source suites.
        # Here verify that the real packaged declaration pins packaged module bytes.
        evidence = json.loads((out / 'e3' / decode_probe.EVIDENCE_FILE).read_text())
        with patch.object(decode_probe, '_module_dir', lambda: str(out / 'e3')):
            self.assertEqual(require_decode_probe_ready(), evidence)
        self.assertEqual((out / 'E2-GENERATION.json').read_bytes(),
                         (base / 'E2-GENERATION.json').read_bytes())
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
        self.assertEqual(set(evidence['build_only_source_shas']),
                         set(decode_probe.BUILD_ONLY_SOURCES))
        self.assertNotIn('operation_source',
                         declaration['module_inventory']['runtime'])
        self.assertEqual(declaration['module_inventory']['build_only'],
                         list(decode_probe.BUILD_ONLY_SOURCES))
        self.assertIn('e3/operation_source.py',
                      (out / 'run-flash-s5-test.sh').read_text())

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



class ReviewRegressions(ProbeFixture):
    def test_actual_prefill_cannot_hide_in_a_captured_full_descriptor(self):
        self.probe.transition('split')
        mixed = batch_state(8, True)
        with self.assertRaises(DecodeProbeInvalidated):
            graph_hooks.check_route(self.runner, types.SimpleNamespace(), FakeDesc(FULL,8,8),
                                    8, 8, 1, mixed)
        self.assertEqual(self.probe.counters()['replays'], 0)
        self.assertEqual(self.stock.replayed, [])

    def test_missing_cpu_batch_proof_is_not_decode(self):
        self.probe.transition('split')
        with self.assertRaises(DecodeProbeInvalidated):
            graph_hooks.check_route(self.runner, types.SimpleNamespace(), FakeDesc(FULL,8,8), 8,8,1)

    def test_empty_early_return_invalidates(self):
        self.probe.transition('split')
        with self.assertRaises(DecodeProbeInvalidated):
            graph_hooks.check_empty(self.runner, types.SimpleNamespace(total_num_scheduled_tokens=0))
        self.assertEqual(self.probe.last_invalidation, 'empty_batch')

    def test_no_prefill_guard_binds_context_and_pool(self):
        from dataclasses import replace
        self.probe.transition('split')
        guard = self.probe.router.no_prefill_guard()
        for bad in (replace(guard, pool_id=guard.pool_id+1),
                    replace(guard, context_id=guard.context_id+1)):
            with self.assertRaises(GraphRoutingError):
                self.probe.router.supported_route(bad, desc(8,8,1))

    def test_pending_sampling_refuses_transition(self):
        self.runner.execute_model_state = object()
        with self.assertRaises(DecodeProbeError):
            self.probe.transition('split')
        self.assertEqual(self.probe.route, 'full')
        self.assertIsNone(self.probe.quarantined)

    def test_probe_rpc_reaches_real_transition_and_binds_epoch(self):
        worker = types.SimpleNamespace(model_runner=self.runner, rank=0)
        command = dict(action='transition', engine_epoch=7, expected_route_generation=0,
                       route_generation=1, storage_epoch=1, target='split')
        with patch.dict(os.environ, self.env):
            ack = worker_hooks.decode_probe_rpc(worker, command)
            self.assertEqual((ack['rank'], ack['route'], ack['engine_epoch']), (0,'split',7))
            self.assertEqual(decode_probe.validate_transition_acks(command, [ack], [0]), (0,))
            with self.assertRaises(DecodeProbeError):
                decode_probe.validate_transition_acks(command, [ack], [0,1,2,3])
            with self.assertRaises(DecodeProbeError):
                decode_probe.validate_transition_acks(command, [ack,ack], [0,1])
            with self.assertRaises(native_startup.NativeStartupError):
                worker_hooks.decode_probe_rpc(worker, dict(command, target='full', engine_epoch=8,
                                                           expected_route_generation=1, route_generation=2))
            with self.assertRaises(native_startup.NativeStartupError):
                worker_hooks.decode_probe_rpc(worker, dict(command, expected_route_generation=True))
            status = worker_hooks.decode_probe_rpc(worker, {'action':'status'})
            self.assertEqual(status['counters']['route'], 'split')
            self.assertEqual(self.probe.route_generation, 1)

    def test_generated_worker_method_invokes_probe_bridge(self):
        import ast
        import executor_hooks
        path = ROOT / 'reference/vllm-src/v1/worker/gpu_worker.py'
        source = worker_hooks.generate_one('v1/worker/gpu_worker.py',
                                          executor_hooks.generate_worker(path.read_text()))
        tree = ast.parse(source)
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == 'flash_s5_decode_probe')
        namespace = {'_flash_s5_worker':worker_hooks}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),
                     '<generated probe RPC>', 'exec'), namespace)
        worker = types.SimpleNamespace(model_runner=self.runner, rank=0)
        with patch.dict(os.environ, self.env):
            status = namespace['flash_s5_decode_probe'](worker, {'action':'status'})
        self.assertEqual(status['rank'], 0)

    def test_split_graph_retained_and_instantiated_stock_unchanged(self):
        calls = []
        def graph(**kw):
            calls.append(kw)
            return types.SimpleNamespace(instantiate=lambda:calls.append('instantiate'))
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(CUDAGraph=graph))
        split_manager = types.SimpleNamespace(_flash_s5_capture_context=object())
        with patch.dict(os.environ, self.env):
            handle = worker_hooks.capture_graph(split_manager, torch_module=torch)
            worker_hooks.finish_graph(split_manager, handle)
            stock = types.SimpleNamespace()
            handle = worker_hooks.capture_graph(stock, torch_module=torch)
            worker_hooks.finish_graph(stock, handle)
        self.assertEqual(calls, [{'keep_graph':True}, 'instantiate', {}])
        calls.clear()
        with patch.dict(os.environ, {PROBE_ENV:'0'}):
            worker_hooks.capture_graph(types.SimpleNamespace(_flash_s5_capture_context=object()),
                                       torch_module=torch)
        self.assertEqual(calls,[{}])

    def test_wrong_compiled_library_pin_refused_before_bank_creation(self):
        runner = fake_runner(self.torch)
        runner._flash_s5_split = self.new_split()
        with patch.object(native_startup, 'install_split_bank', side_effect=AssertionError('allocated')):
            with self.assertRaisesRegex(native_startup.NativeStartupError, 'library'):
                native_startup.install_decode_probe(runner, env=dict(self.env, FLASH_S5_GREEN_LIB_SHA256='0'*64),
                                                    torch_module=self.torch)

    def test_profile_capture_does_not_install_a_second_bank(self):
        with patch.dict(os.environ, self.env), patch.object(native_startup, 'install_decode_probe',
                                                            side_effect=AssertionError('capture')):
            self.assertIsNone(graph_hooks.capture_banks(types.SimpleNamespace(), profile_only=True))

    def test_flag_conflict_refused_before_workspace_lane_allocation(self):
        with self.assertRaises(DecodeProbeError):
            worker_hooks.workspace_lanes(2, {PROBE_ENV:'1', 'FLASH_S5_OVERLAP':'1'})
        self.assertEqual(worker_hooks.workspace_lanes(2, self.env), 2)


class PersistedEvidenceTests(unittest.TestCase):
    def test_default_runtime_gate_requires_persisted_declaration(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(decode_probe, '_module_dir', lambda:tmp):
            with self.assertRaises(DecodeProbeNotReady):
                require_decode_probe_ready()

    def test_tamper_after_declaration_is_detected(self):
        root = fake_evidence(self)
        require_decode_probe_ready()
        (root / 'lane_state.py').write_text('modified')
        with self.assertRaisesRegex(DecodeProbeNotReady, 'lane_state'):
            require_decode_probe_ready()

    def test_helper_recipe_is_q17a_sm86_shared_and_never_replaces(self):
        import make_s5_overlap as g
        from s5_clock import build_command
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / 'build.sh'
            script.write_text(g.BUILD_GREEN_HELPER)
            self.assertEqual(subprocess.run(['bash','-n',str(script)],capture_output=True).returncode,0)
            # Verify the exact nvcc words in the generated shell, independent of
            # line wrapping. No docker/CUDA/compiler invocation on the hub.
            import shlex
            words = shlex.split(g.BUILD_GREEN_HELPER.replace('\\\n',' '))
            start = words.index('nvcc')
            expected = build_command('/native/flash_s5_green.cu','/native/flash_s5_green.so',(8,6),'shared')
            self.assertEqual(words[start:start+len(expected)], expected)
            self.assertNotIn('--rm', words)
            self.assertIn('--network',words)
            (root/'flash_s5_green.so').write_bytes(b'existing')
            done = subprocess.run(['bash',str(script),
                'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b',tmp],
                capture_output=True,timeout=10)
            self.assertNotEqual(done.returncode,0)
            self.assertEqual((root/'flash_s5_green.so').read_bytes(),b'existing')

    def test_no_full_graph_configuration_refused_at_startup(self):
        fake_evidence(self)
        runner = types.SimpleNamespace(cudagraph_manager=types.SimpleNamespace(_capture_descs={}))
        with self.assertRaises(Exception) as caught:
            s5_startup.install_builder(runner, None, env={PROBE_ENV:'1', DECODE_SMS_ENV:'44'})
        self.assertIn('no FULL', str(caught.exception))

def noop_scheduler_output():
    return types.SimpleNamespace(total_num_scheduled_tokens=0,
        scheduled_new_reqs=[], num_scheduled_tokens={}, finished_req_ids=set(),
        scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
        free_encoder_mm_hashes=[], preempted_req_ids=None,
        new_block_ids_to_zero=None, kv_cache_block_copies=None,
        block_table_updates=None, kv_connector_metadata=None,
        kv_connector_block_state=None, ec_connector_metadata=None,
        ec_manager_metadata=None, has_sync_kv_loads=False,
        scheduled_cached_reqs=types.SimpleNamespace(req_ids=[], resumed_req_ids=set(),
            new_token_ids=[], all_token_ids={}, new_block_ids=[],
            num_computed_tokens=[], num_output_tokens=[]))


def pinned_paused_scheduler():
    """Real pinned schedule(), with only cache-manager boundaries faked on CPU."""
    import ast
    import itertools
    import time
    source = ROOT / 'reference/vllm-src/v1/core/sched/output.py'
    tree = ast.parse(source.read_text())
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ('CachedRequestData', 'SchedulerOutput'):
            node.body = [part for part in node.body if isinstance(part, ast.AnnAssign)
                         or isinstance(part, ast.FunctionDef) and part.name == 'make_empty']
            classes.append(node)
    module = ast.parse('from __future__ import annotations\nfrom dataclasses import dataclass, field\n')
    module.body.extend(classes)
    namespace = {'__name__': __name__, 'time': time, 'itertools': itertools,
                 'record_function_or_nullcontext': lambda *a: contextlib.nullcontext(),
                 'PauseState': types.SimpleNamespace(PAUSED_ALL=1, UNPAUSED=0)}
    source = ROOT / 'reference/vllm-src/v1/core/sched/scheduler.py'
    scheduler = next(n for n in ast.parse(source.read_text()).body
                     if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    scheduler.bases = []; scheduler.decorator_list = []
    scheduler.body = [n for n in scheduler.body if isinstance(n, ast.FunctionDef)
                      and n.name in ('schedule', '_make_cached_request_data',
                                     '_get_new_block_ids_to_zero', '_update_after_schedule')]
    module.body.append(scheduler)
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    obj = namespace['Scheduler']()
    attrs = dict(current_step=0, max_num_scheduled_tokens=1024,
        vllm_config=types.SimpleNamespace(speculative_config=None),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=1024, async_scheduling=True),
        _pause_state=1, max_num_encoder_input_tokens=0,
        running=[types.SimpleNamespace(request_id='fixture', is_prefill_chunk=False)],
        waiting=[], skipped_waiting=[], pd_alternate=False, parity_balance=False,
        _pd_share=None, _sf_max=0, lora_config=None, _sched_stats_every=0,
        max_num_running_reqs=11, use_v2_model_runner=True, model_uses_mrope=False,
        kv_cache_config=types.SimpleNamespace(kv_cache_groups=[object()]),
        connector=None, ec_connector=None, num_spec_tokens=0, dynamic_sd_lookup=None,
        log_stats=False, reset_preempted_req_ids=set(), finished_req_ids=set(),
        needs_kv_cache_zeroing=False, defer_block_free=True, sched_step_seq=17,
        enable_return_routed_experts=False)
    for key, value in attrs.items():
        setattr(obj, key, value)
    obj.kv_cache_manager = types.SimpleNamespace(new_step_starts=lambda:None,
        get_num_common_prefix_blocks=lambda rid:[0], take_boundary_state_offloads=lambda:{},
        take_kv_cache_block_copies=lambda:([], []), take_new_block_ids=lambda:[])
    obj.encoder_cache_manager = types.SimpleNamespace(get_freed_mm_hashes=lambda:[],
                                                       get_manager_metadata=lambda:None)
    return obj


class BatchAndBarrierTests(ProbeFixture):
    def test_actual_paused_schedule_with_running_unscheduled_request(self):
        scheduler = pinned_paused_scheduler()
        self.probe.check_batch(8, 8, 1, batch_state(8)); self.probe.check_route(desc())
        for _ in range(3):
            output = scheduler.schedule()
            self.assertEqual(output.total_num_scheduled_tokens, 0)
            self.assertEqual(output.scheduled_cached_reqs.req_ids, [])
            self.assertIsNone(decode_probe.DecodeProbe._empty_noop_rejection(output))
            self.probe.observe_empty(output)
        self.assertEqual(scheduler.sched_step_seq, 17)
        self.assertTrue(self.probe.batch_observations['last']['full_graph'])
        self.assertEqual(self.probe.batch_observations['noop_empty_batches'], 3)

    def test_paused_schedule_finished_and_zeroing_rejections_are_identified(self):
        scheduler = pinned_paused_scheduler()
        scheduler.finished_req_ids = {'fixture-finished'}
        self.probe.observe_empty(scheduler.schedule())
        self.assertEqual(self.probe.batch_observations['noop_rejections'], {'finished_req_ids':1})
        scheduler.needs_kv_cache_zeroing = True
        scheduler._skip_zero_block_ids = set()
        scheduler.kv_cache_manager.take_new_block_ids = lambda:[123]
        self.probe.observe_empty(scheduler.schedule())
        self.assertEqual(self.probe.batch_observations['last_noop_rejection'],
                         {'field':'new_block_ids_to_zero', 'empty_sequence':2})

    def test_rejections_are_bounded_first_fields_with_independent_snapshots(self):
        output = noop_scheduler_output()
        output.finished_req_ids = {'never-emit-this-id'}
        output.new_block_ids_to_zero = [987654321]
        self.probe.observe_empty(output)
        saved = self.probe.counters()['batch_observations']
        self.probe.observe_empty(output)
        self.assertEqual(saved['noop_rejections'], {'finished_req_ids':1})
        self.assertEqual(saved['last_noop_rejection']['empty_sequence'], 1)
        self.assertEqual(self.probe.batch_observations['noop_rejections'], {'finished_req_ids':2})
        output = noop_scheduler_output()
        output.scheduled_cached_reqs.req_ids = ['never-emit-this-id']
        self.probe.observe_empty(output)
        self.assertEqual(self.probe.batch_observations['last_noop_rejection']['field'],
                         'scheduled_cached_reqs.req_ids')
        output = noop_scheduler_output()
        del output.num_scheduled_tokens
        self.probe.observe_empty(output)
        self.assertEqual(self.probe.batch_observations['last_noop_rejection']['field'],
                         'num_scheduled_tokens')
        class DictSubclass(dict): pass
        output = noop_scheduler_output(); output.num_scheduled_tokens = DictSubclass()
        self.probe.observe_empty(output)
        self.assertEqual(self.probe.batch_observations['noop_rejections']['num_scheduled_tokens'],2)
        data = json.dumps(self.probe.counters())
        self.assertNotIn('never-emit-this-id', data)
        self.assertNotIn('987654321', data)

    def test_noop_matches_pinned_scheduler_output_make_empty(self):
        import ast
        source=ROOT/'reference/vllm-src/v1/core/sched/output.py'
        tree=ast.parse(source.read_text())
        classes=[]
        for node in tree.body:
            if isinstance(node,ast.ClassDef) and node.name in ('CachedRequestData','SchedulerOutput'):
                node.body=[part for part in node.body if isinstance(part,ast.AnnAssign)
                           or isinstance(part,ast.FunctionDef) and part.name=='make_empty']
                classes.append(node)
        module=ast.parse('from __future__ import annotations\nfrom dataclasses import dataclass, field\n')
        module.body.extend(classes)
        namespace={'__name__':__name__}
        exec(compile(ast.fix_missing_locations(module),str(source),'exec'),namespace)
        output=namespace['SchedulerOutput'].make_empty()
        self.assertTrue(decode_probe.DecodeProbe._empty_is_noop(output))
        output.finished_req_ids.add('fixture')
        self.assertFalse(decode_probe.DecodeProbe._empty_is_noop(output))

    def test_keep_pause_noop_preserves_actual_decode_evidence(self):
        self.probe.check_batch(8, 8, 1, batch_state(8))
        self.probe.check_route(desc())
        before = self.probe.counters()['batch_observations']['last']
        for _ in range(4):
            graph_hooks.check_empty(self.runner, noop_scheduler_output())
        self.assertEqual(self.probe.counters()['batch_observations']['last'], before)
        self.assertTrue(before['decode_only']); self.assertTrue(before['full_graph'])
        self.assertEqual(self.probe.batch_observations['empty_batches'], 4)
        from probe_control import ProbeController, ProbeControlError
        owner=types.SimpleNamespace(engine=types.SimpleNamespace(
            get_num_unfinished_requests=lambda:8, get_num_queued_tokens=lambda:0))
        snapshot={'ranks':[{'rank':r,'counters':self.probe.counters()} for r in range(4)]}
        self.assertTrue(ProbeController._eligibility(owner,snapshot,8)['proved'])
        owner.engine.get_num_unfinished_requests=lambda:7
        with self.assertRaises(ProbeControlError):
            ProbeController._eligibility(owner,snapshot,8)

    def test_empty_mutations_and_missing_fields_clear_evidence(self):
        mutations = [('finished_req_ids', {'fixture'}), ('scheduled_new_reqs', [object()]),
            ('preempted_req_ids', {'fixture'}), ('kv_cache_block_copies', [object()]),
            ('new_block_ids_to_zero', [1]), ('block_table_updates', {'fixture': ()}),
            ('kv_connector_metadata', object()), ('has_sync_kv_loads', True)]
        for name, value in mutations:
            with self.subTest(name=name):
                self.probe.check_batch(8,8,1,batch_state(8)); self.probe.check_route(desc())
                output=noop_scheduler_output();setattr(output,name,value)
                graph_hooks.check_empty(self.runner,output)
                self.assertIsNone(self.probe.batch_observations['last'])
                graph_hooks.check_empty(self.runner,noop_scheduler_output())
                self.assertIsNone(self.probe.batch_observations['last'])
        self.probe.check_batch(8,8,1,batch_state(8))
        output=noop_scheduler_output();del output.finished_req_ids
        graph_hooks.check_empty(self.runner,output)
        self.assertIsNone(self.probe.batch_observations['last'])

    def test_noop_does_not_hide_prefill_or_nonfull_dispatch(self):
        self.probe.check_batch(8,8,1,batch_state(8, prefill=True))
        self.probe.check_route(desc())
        graph_hooks.check_empty(self.runner,noop_scheduler_output())
        self.assertFalse(self.probe.batch_observations['last']['decode_only'])
        self.probe.check_batch(8,8,1,batch_state(8))
        self.probe.check_route(desc(mode='none'))
        graph_hooks.check_empty(self.runner,noop_scheduler_output())
        self.assertFalse(self.probe.batch_observations['last']['full_graph'])

    def test_split_empty_still_invalidates(self):
        self.probe.transition('split')
        with self.assertRaises(DecodeProbeInvalidated):
            graph_hooks.check_empty(self.runner,noop_scheduler_output())
        self.assertEqual(self.probe.last_invalidation,'empty_batch')

    def test_scheduler_preemption_is_counted_on_full_and_refused_on_split(self):
        worker = types.SimpleNamespace(model_runner=self.runner)
        worker_hooks.observe_scheduler(worker, types.SimpleNamespace(preempted_req_ids={'opaque'}))
        self.assertEqual(self.probe.batch_observations['preempted_requests'], 1)
        self.probe.transition('split')
        with self.assertRaises(DecodeProbeInvalidated):
            worker_hooks.observe_scheduler(worker, types.SimpleNamespace(preempted_req_ids={'opaque'}))
        self.assertEqual(self.probe.last_invalidation, 'preempted_request')

    def test_missing_scheduler_metadata_is_unknown_not_zero(self):
        self.probe.observe_scheduler(types.SimpleNamespace())
        self.assertEqual(self.probe.batch_observations['unknown_preemption_observations'], 1)
        self.probe.observe_scheduler(types.SimpleNamespace(preempted_req_ids=None))
        self.assertEqual(self.probe.batch_observations['unknown_preemption_observations'], 1)
        self.assertEqual(self.probe.batch_observations['scheduler_observations'], 2)

    def test_full_route_records_prefill_and_decode_without_refusing_stock(self):
        self.probe.check_batch(8, 8, 1, batch_state(8, prefill=True))
        self.probe.check_route(desc())
        before = self.probe.counters()['batch_observations']
        self.assertEqual(before['non_decode_batches'], 1)
        self.assertFalse(before['last']['decode_only'])
        self.probe.check_batch(8, 8, 1, batch_state(8))
        self.probe.check_route(desc())
        after = self.probe.counters()['batch_observations']
        self.assertEqual(after['decode_batches'], 1)
        self.assertTrue(after['last']['decode_only'])
        self.assertTrue(after['last']['full_graph'])
        before['last']['decode_only'] = 'tampered'
        self.assertTrue(self.probe.counters()['batch_observations']['last']['decode_only'])
        self.assertEqual(self.probe.route, 'full')

    def test_empty_full_batch_clears_stale_eligibility(self):
        self.probe.check_batch(8, 8, 1, batch_state(8))
        graph_hooks.check_empty(self.runner, types.SimpleNamespace(total_num_scheduled_tokens=0))
        self.assertIsNone(self.probe.counters()['batch_observations']['last'])
        self.assertEqual(self.probe.batch_observations['empty_batches'], 1)

    def test_real_worker_barrier_binds_epoch_and_drains_without_transition(self):
        worker = types.SimpleNamespace(model_runner=self.runner, rank=0)
        with patch.dict(os.environ, self.env):
            ack = worker_hooks.decode_probe_rpc(worker, {'action':'barrier','engine_epoch':9})
            self.assertGreater(ack['drained_streams'], 0)
            self.assertEqual(ack['ack'], 'flash_s5_probe_barrier')
            self.assertEqual(ack['route_generation'], 0)
            self.assertEqual(self.probe.route, 'full')
            with self.assertRaises(native_startup.NativeStartupError):
                worker_hooks.decode_probe_rpc(worker, {'action':'barrier','engine_epoch':10})

    def test_barrier_pending_sampling_refuses_and_drain_failure_quarantines(self):
        worker = types.SimpleNamespace(model_runner=self.runner, rank=0)
        with patch.dict(os.environ, self.env):
            self.runner.execute_model_state = object()
            with self.assertRaises(native_startup.NativeStartupError):
                worker_hooks.decode_probe_rpc(worker, {'action':'barrier','engine_epoch':9})
            self.runner.execute_model_state = None
            with patch.object(self.probe.controller, 'drain', side_effect=RuntimeError('drain failed')):
                with self.assertRaises(RuntimeError):
                    worker_hooks.decode_probe_rpc(worker, {'action':'barrier','engine_epoch':9})
            self.assertEqual(self.probe.quarantined, 'diagnostic barrier failed')


class CaptureObservationTests(unittest.TestCase):
    def setUp(self):
        self.env = {PROBE_ENV: '1'}
        self.stream = types.SimpleNamespace(cuda_stream=891234567890123)
        self.owner = types.SimpleNamespace(state='open', decode_stream=self.stream,
                                          decode_stream_token=self.stream.cuda_stream)
        self.manager = types.SimpleNamespace(device='cuda:0',
                    _flash_s5_capture_context=object(), _flash_s5_capture_owner=self.owner)
        self.descriptor = FakeDesc(FULL, 8, 8)
        self.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
                    current_stream=lambda device:self.stream))

    def check(self, stream=None):
        with patch.dict(os.environ, self.env):
            return worker_hooks.checked_capture_stream(self.manager,
                       self.stream if stream is None else stream, self.descriptor,
                       torch_module=self.torch)

    def test_matching_stream_records_metadata_without_native_handles(self):
        self.assertIs(self.check(), self.stream)
        checks = self.manager._flash_s5_capture_stream_checks
        record = next(iter(checks.values()))
        self.assertTrue(record['torch_matches_owner'])
        self.assertTrue(record['vllm_matches_owner'])
        self.assertNotIn(str(self.stream.cuda_stream), json.dumps(record))
        with self.assertRaisesRegex(DecodeProbeError, 'duplicate'):
            self.check()

    def test_each_observed_stream_must_match_the_owner(self):
        wrong = types.SimpleNamespace(cuda_stream=42)
        with self.assertRaisesRegex(DecodeProbeError, 'agreement'):
            self.check(wrong)
        self.torch.cuda.current_stream = lambda device: wrong
        with self.assertRaisesRegex(DecodeProbeError, 'agreement'):
            self.check()
        self.torch.cuda.current_stream = lambda device:self.stream
        self.owner.decode_stream = wrong
        with self.assertRaisesRegex(DecodeProbeError, 'agreement'):
            self.check()
        self.assertFalse(hasattr(self.manager, '_flash_s5_capture_stream_checks'))

    def test_missing_closed_or_boolean_owner_is_refused(self):
        for state, token in [('closed', self.stream.cuda_stream), ('open', True),
                             ('open', None), ('open', 0)]:
            self.owner.state, self.owner.decode_stream_token = state, token
            with self.assertRaises(DecodeProbeError):
                self.check()
        self.manager._flash_s5_capture_owner = None
        with self.assertRaises(DecodeProbeError):
            self.check()

    def test_stock_and_default_off_do_not_query_torch(self):
        def forbidden(*args):
            raise AssertionError('stock query')
        self.torch.cuda.current_stream = forbidden
        with patch.dict(os.environ, {PROBE_ENV:'0'}):
            self.assertIs(worker_hooks.checked_capture_stream(self.manager, self.stream,
                          self.descriptor, torch_module=self.torch), self.stream)
        with patch.dict(os.environ, self.env):
            self.assertIs(worker_hooks.checked_capture_stream(types.SimpleNamespace(),
                          self.stream, self.descriptor, torch_module=self.torch), self.stream)

    def test_generated_graph_stream_argument_executes_the_guard(self):
        import ast
        rel = 'v1/worker/gpu/cudagraph_utils.py'
        original = (ROOT / 'reference/vllm-src' / rel).read_text()
        generated = worker_hooks.generate_one(rel, original)
        self.assertEqual(worker_hooks.revert_one(rel, generated), original)
        tree = ast.parse(generated)
        stream_expr = next(n.value for n in ast.walk(tree)
                           if isinstance(n, ast.keyword) and n.arg == 'stream'
                           and isinstance(n.value, ast.Call)
                           and isinstance(n.value.func, ast.Name)
                           and n.value.func.id == '_flash_s5_checked_capture_stream')
        calls = []
        def current_stream():
            calls.append('vllm')
            return self.stream
        def guard(manager, stream, descriptor):
            return worker_hooks.checked_capture_stream(manager, stream, descriptor,
                                                       torch_module=self.torch)
        ns = {'self':self.manager, 'desc':self.descriptor,
              'current_stream':current_stream, '_flash_s5_checked_capture_stream':guard}
        with patch.dict(os.environ, self.env):
            value = eval(compile(ast.Expression(stream_expr), '<actual stream argument>', 'eval'), ns)
        self.assertIs(value, self.stream)
        self.assertEqual(calls, ['vllm'])
        # Actual graph() construction occurs only after its argument has passed.
        self.assertEqual(len(self.manager._flash_s5_capture_stream_checks), 1)

    def test_helper_source_behavior_is_pinned_and_packaged(self):
        import hashlib
        path = ROOT / 'reference/vllm-src/utils/torch_utils.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         worker_hooks.CAPTURE_REFERENCE_PINS['utils/torch_utils.py'])
        tree = __import__('ast').parse(path.read_text())
        fn = next(n for n in tree.body if getattr(n, 'name', None) == 'current_stream')
        self.assertEqual(len(fn.args.args), 0)
        evidence = decode_probe.declared_evidence()
        evidence['pins'].pop('utils/torch_utils.py')
        with self.assertRaisesRegex(DecodeProbeNotReady, 'known pin'):
            require_decode_probe_ready(evidence)


class MemoryObservationTests(ProbeFixture):
    def test_status_samples_labeled_memory_without_sync_or_reset(self):
        keys = ('allocated_bytes.all.current', 'reserved_bytes.all.current',
                'allocated_bytes.all.peak', 'reserved_bytes.all.peak',
                'inactive_split_bytes.all.current', 'inactive_split_bytes.all.peak')
        self.torch.cuda.memory_stats = lambda device: dict(zip(keys, [10,20,30,40,5,6]))
        self.torch.cuda.mem_get_info = lambda device: (50,100)
        def forbidden(*args):
            raise AssertionError('sync/reset called')
        self.torch.cuda.synchronize = forbidden
        self.torch.cuda.reset_peak_memory_stats = forbidden
        self.torch.accelerator.synchronize = forbidden
        worker = types.SimpleNamespace(model_runner=self.runner, rank=0)
        with patch.dict(os.environ, self.env):
            record = worker_hooks.decode_probe_rpc(worker, {'action':'status'})
        m = record['memory']
        self.assertEqual(m['allocator']['reserved_peak_bytes'], 40)
        self.assertEqual(m['device_free_bytes'], 50)
        self.assertEqual(m['errors'], [])
        self.assertFalse(m['synchronized_by_snapshot'])
        self.assertIn('lifetime', m['peak_scope'])
        self.assertGreaterEqual(m['sample_end_monotonic_ns'], m['monotonic_ns'])
        self.assertEqual(record['inventory']['capture_memory']['delta_bytes'], 4096)

    def test_unknown_and_error_values_are_not_zero_or_error_payloads(self):
        def failed(device):
            raise RuntimeError('private memory payload')
        self.torch.cuda.memory_stats = failed
        self.torch.cuda.mem_get_info = failed
        m = self.probe.memory_snapshot()
        self.assertTrue(all(v is None for v in m['allocator'].values()))
        self.assertIsNone(m['device_free_bytes'])
        self.assertNotIn('private memory payload', json.dumps(m))
        self.assertIn('allocator:RuntimeError', m['errors'])

    def test_invalid_counter_types_and_device_pair_remain_unknown(self):
        self.torch.cuda.memory_stats = lambda device:{'allocated_bytes.all.current':True,
                                                     'reserved_bytes.all.peak':-1}
        self.torch.cuda.mem_get_info = lambda device:(101,100)
        m = self.probe.memory_snapshot()
        self.assertIsNone(m['allocator']['allocated_current_bytes'])
        self.assertIsNone(m['allocator']['reserved_peak_bytes'])
        self.assertIsNone(m['device_total_bytes'])
        self.assertIn('device:ValueError', m['errors'])


if __name__ == '__main__':
    unittest.main()
