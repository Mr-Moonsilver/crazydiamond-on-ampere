"""CPU-only tests for native prefill side-stream ownership (addendum).

Covers exact SharedExperts/PLE owner discovery, sharing, prefill-green role
checks, exact finally restoration, failed partial construction, allocator
lifetime retention and the native transition/release drain hooks. Fake CUDA
wrappers and a fake driver binding only: no GPU, no vLLM process, no native
readiness claim.
"""
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
sys.path.insert(0, str(ROOT / 'harness'))

from green_context import GreenContextError, GreenSplit, _PROCESS_RETAINED_SPLITS  # noqa: E402
from model_hooks import (  # noqa: E402
    LaneForward,
    ModelHookError,
    PlePrivateScratch,
    PrefillStreamOwners,
)
from test_e3_green_native import FakeBinding, FakeStream, fake_torch  # noqa: E402


class Buffer:
    def __init__(self, shape, dtype='bf16'):
        self.shape = tuple(shape)
        self.dtype = dtype

    def numel(self):
        n = 1
        for dim in self.shape:
            n *= dim
        return n

    def element_size(self):
        return 2


def scratch_torch():
    torch = fake_torch()
    torch.empty = lambda shape, dtype=None, device=None: Buffer(shape, dtype)
    torch.cuda.Stream = lambda device=None: FakeStream(900, 'private')
    return torch


class Embedding:
    supports_prefetch = True
    etp_data_parallel_size = 1

    def __init__(self, token=91):
        self._prefetch_buffer = Buffer((4096, 4, 128))
        self._prefetch_stream = FakeStream(token, 'stock-ple')


class Owner:
    def __init__(self, token=None):
        self._stream = FakeStream(token, 'stock-shared') if token is not None else None


def make_split(binding=None):
    torch = scratch_torch()
    binding = binding if binding is not None else FakeBinding()
    return torch, binding, GreenSplit(binding, 0, 44, torch_module=torch).create()


class SplitCase(unittest.TestCase):
    def setUp(self):
        self.torch, self.binding, self.split = make_split()

    def owners(self, entries):
        return PrefillStreamOwners(self.split, entries)


class SharingTests(SplitCase):
    def test_prepare_shares_one_aux_per_original_and_restores_exactly(self):
        first = FakeStream(90, 'stock-shared')
        stock_ple = FakeStream(91, 'stock-ple')
        a, b = Owner(90), Owner(90)
        a._stream = b._stream = first
        ple = types.SimpleNamespace(_prefetch_stream=stock_ple)
        owners = self.owners([(a, '_stream'), (b, '_stream'), (ple, '_prefetch_stream')])
        owners.prepare()
        record = owners.record()
        self.assertEqual(record['state'], 'ready')
        self.assertEqual(record['distinct_original_streams'], 2)
        self.assertEqual(record['auxiliary_streams'], 2)
        self.assertEqual(record['shared_owners'], 2)
        self.assertEqual(record['ple_owners'], 1)
        with owners.scope():
            self.assertIs(a._stream, b._stream)
            self.assertIsNot(a._stream, first)
            self.assertIsNot(a._stream, ple._prefetch_stream)
            self.assertNotIn(a._stream.cuda_stream,
                             (first.cuda_stream, stock_ple.cuda_stream))
            with self.assertRaises(ModelHookError):
                with owners.scope():
                    pass
        self.assertIs(a._stream, first)
        self.assertIs(ple._prefetch_stream, stock_ple)
        # A foreign mutation between segments is refused, never silently
        # overwritten by the next swap.
        b._stream = FakeStream(95, 'foreign')
        with self.assertRaises(ModelHookError):
            with owners.scope():
                pass
        self.assertEqual(b._stream.cuda_stream, 95)

    def test_drain_synchronizes_every_owned_auxiliary(self):
        owners = self.owners([(Owner(90), '_stream'), (Embedding(91), '_prefetch_stream')])
        self.assertRaises(ModelHookError, owners.drain)  # not prepared yet
        owners.prepare()
        self.assertEqual(owners.drain(), 2)
        self.assertEqual([stream.synchronized for stream in owners.streams], [1, 1])


class RefusalTests(SplitCase):
    def test_role_context_check_rejects_the_other_pool(self):
        owners = self.owners([(Owner(90), '_stream')])
        owners.prepare()
        decode = self.split.create_decode_aux()
        self.assertRaises(GreenContextError, self.split.check_prefill_stream, decode)
        prefill = owners.streams[0]
        self.assertRaises(GreenContextError, self.split.check_decode_stream, prefill)

    def test_foreign_prefill_context_refuses_and_destroys_the_created_aux(self):
        self.binding.wrong_prefill = True
        owners = self.owners([(Owner(90), '_stream')])
        with self.assertRaises(GreenContextError):
            owners.prepare()
        self.assertEqual(owners.state, 'failed')
        self.assertEqual(owners.streams, [])
        self.assertEqual(getattr(self.binding, 'prefill_tokens', set()), set())
        self.assertNotIn(self.split, _PROCESS_RETAINED_SPLITS)

    def test_failed_partial_construction_destroys_earlier_auxiliaries(self):
        class SecondFails(FakeBinding):
            def prefill_aux_create(self, handle):
                self.calls = getattr(self, 'calls', 0) + 1
                if self.calls == 2:
                    raise GreenContextError('helper create failed')
                return FakeBinding.prefill_aux_create(self, handle)

        self.binding = SecondFails()
        self.split = GreenSplit(self.binding, 0, 44, torch_module=self.torch).create()
        owners = self.owners([(Owner(90), '_stream'), (Embedding(91), '_prefetch_stream')])
        with self.assertRaises(GreenContextError):
            owners.prepare()
        self.assertEqual(owners.state, 'failed')
        self.assertEqual(self.binding.prefill_tokens, set())

    def test_exposed_split_refuses_native_destroy(self):
        owners = self.owners([(Owner(90), '_stream')])
        owners.prepare()
        self.assertTrue(self.split.lifetime_record()['allocator_exposed'])
        self.assertRaises(GreenContextError, self.split.destroy_prefill_aux,
                          owners.streams[0])

    def test_allocator_exposed_construction_failure_retains_earlier_auxiliaries(self):
        class SecondFails(FakeBinding):
            def prefill_aux_create(self, handle):
                self.calls = getattr(self, 'calls', 0) + 1
                if self.calls == 2:
                    raise GreenContextError('helper create failed')
                return FakeBinding.prefill_aux_create(self, handle)

        self.binding = SecondFails()
        self.split = GreenSplit(self.binding, 0, 44, torch_module=self.torch).create()
        # The decoder capture already exposed allocator records for this split.
        self.split.retain_allocator_context()
        owners = self.owners([(Owner(90), '_stream'), (Embedding(91), '_prefetch_stream')])
        with self.assertRaises(GreenContextError):
            owners.prepare()
        self.assertEqual(len(self.binding.prefill_tokens), 1)
        self.assertTrue(self.split.lifetime_record()['allocator_exposed'])

    def test_close_logically_drains_and_retains_every_auxiliary(self):
        owners = self.owners([(Owner(90), '_stream'), (Embedding(91), '_prefetch_stream')])
        owners.prepare()
        self.assertIn(self.split, _PROCESS_RETAINED_SPLITS)
        self.assertTrue(self.split.close())
        self.assertTrue(self.split.native_retained)
        self.assertEqual([stream.synchronized for stream in owners.streams], [1, 1])
        self.assertEqual(len(self.binding.prefill_tokens), 2)


class PleScratchTests(SplitCase):
    def test_default_off_keeps_a_primary_context_stream(self):
        scratch = PlePrivateScratch(Embedding(91), 1024, torch_module=self.torch,
                                    device='cuda:0')
        self.assertEqual(scratch.stream_basis, 'primary-context')
        self.assertIsNot(scratch.stream, scratch.embedding._prefetch_stream)

    def test_supplied_stream_is_the_owned_auxiliary(self):
        owners = self.owners([(Embedding(91), '_prefetch_stream')])
        owners.prepare()
        embedding = owners.owners[0][0]
        stream = owners.replacement(embedding, '_prefetch_stream')
        scratch = PlePrivateScratch(embedding, 1024, torch_module=self.torch,
                                    device='cuda:0', stream=stream)
        self.assertEqual(scratch.stream_basis, 'prefill-green')
        self.assertIs(scratch.stream, owners.streams[0])
        with scratch.scope():
            self.assertIs(embedding._prefetch_stream, stream)
        self.assertIsNot(embedding._prefetch_stream, stream)

    def test_non_stream_wrapper_is_refused(self):
        with self.assertRaises(ModelHookError):
            PlePrivateScratch(Embedding(91), 1024, torch_module=self.torch,
                              device='cuda:0', stream=object())


class LaneScopeTests(SplitCase):
    def build(self):
        embedding = Embedding(91)
        shared = Owner(90)
        model = types.SimpleNamespace(
            layers=[types.SimpleNamespace(ple=types.SimpleNamespace(ngram_embedding=embedding))],
            start_layer=0, end_layer=1, trace=[])

        def prefix(*args):
            model.trace.append(('prefix', shared._stream.cuda_stream,
                                embedding._prefetch_stream.cuda_stream))
            return 'h'

        def layer(idx, *args):
            model.trace.append(('layer', shared._stream.cuda_stream,
                                embedding._prefetch_stream.cuda_stream))
            return ('h1', 'b', 'i')

        def suffix(*args):
            model.trace.append(('suffix', shared._stream.cuda_stream,
                                embedding._prefetch_stream.cuda_stream))
            return {'hidden_states': 'hc'}

        model._flash_e3_forward_prefix = prefix
        model._flash_e3_forward_layer = layer
        model._flash_e3_forward_suffix = suffix
        model._flash_e3_is_last_stage = lambda: False
        owners = self.owners([(shared, '_stream'), (embedding, '_prefetch_stream')])
        owners.prepare()
        scratch = PlePrivateScratch(embedding, 1024, torch_module=self.torch,
                                    device='cuda:0',
                                    stream=owners.replacement(embedding, '_prefetch_stream'))
        return model, shared, embedding, owners, scratch

    def test_every_segment_runs_on_owned_prefill_streams_and_restores(self):
        model, shared, embedding, owners, scratch = self.build()
        inputs = dict(positions='p', input_ids='i', query_start_loc='q',
                      ngram_context='n')
        lane = LaneForward(model, inputs, stream=self.split.prefill_stream,
                           ple_scratch=scratch, prefill_owners=owners,
                           torch_module=self.torch, clock=lambda: 0)
        lane.begin()
        lane.layer(0)
        lane.finish()
        aux_shared = owners.replacements[0].cuda_stream
        aux_ple = owners.replacements[1].cuda_stream
        self.assertEqual([entry[0] for entry in model.trace],
                         ['prefix', 'layer', 'suffix'])
        for _, seen_shared, seen_ple in model.trace:
            self.assertEqual(seen_shared, aux_shared)
            self.assertEqual(seen_ple, aux_ple)
        self.assertEqual(shared._stream.cuda_stream, 90)
        self.assertEqual(embedding._prefetch_stream.cuda_stream, 91)
        self.assertEqual(lane.state, 'finished')

    def test_scope_refuses_an_unprepared_owners_object(self):
        model, shared, embedding, owners, scratch = self.build()
        unprepared = self.owners([(shared, '_stream')])
        lane = LaneForward(model, dict(positions='p', input_ids='i',
                                       query_start_loc='q', ngram_context='n'),
                           stream=self.split.prefill_stream, ple_scratch=scratch,
                           prefill_owners=unprepared, torch_module=self.torch,
                           clock=lambda: 0)
        with self.assertRaises(ModelHookError):
            lane.begin()


class DiscoveryTests(SplitCase):
    def test_exact_types_sharing_and_disabled_stream(self):
        class Shared:
            pass

        class Ple:
            pass

        Model = type('Qwen4ExpModel', (), {
            '__module__': 'vllm.models.qwen4_exp.nvidia.model'})
        model = Model()
        model.start_layer = 0
        model.end_layer = 3
        model.config = types.SimpleNamespace(ple_layer_ids=[2])
        a, b, c = Shared(), Shared(), Shared()
        a._stream = b._stream = FakeStream(42)
        c._stream = None  # stock flag disables the shared-experts stream
        ple = Ple()
        ple._prefetch_stream = FakeStream(43)
        model.modules = lambda: [model, a, b, c, ple]
        owners = PrefillStreamOwners.discover(
            types.SimpleNamespace(model=model), self.split, pinned=(Shared, Ple))
        self.assertEqual(owners.record()['owners'], 4)
        owners.prepare()
        self.assertEqual(owners.record()['auxiliary_streams'], 2)
        self.assertEqual(owners.replacements[2], None)
        with owners.scope():
            self.assertIs(c._stream, None)
            self.assertIs(a._stream, b._stream)
        model.end_layer = 4
        with self.assertRaises(ModelHookError):
            PrefillStreamOwners.discover(
                types.SimpleNamespace(model=model), self.split, pinned=(Shared, Ple))

    def test_ple_owner_without_a_stream_is_refused(self):
        embedding = Embedding(91)
        embedding._prefetch_stream = None
        with self.assertRaises(ModelHookError):
            self.owners([(embedding, '_prefetch_stream')])


class BridgeDrainTests(unittest.TestCase):
    def test_release_drain_hook_calls_the_owned_prefill_streams(self):
        from worker_bridge import NativeWorkerBridge

        _, _, split = make_split()
        owners = PrefillStreamOwners(
            split, [(Owner(90), '_stream'), (Embedding(91), '_prefetch_stream')])
        owners.prepare()
        bridge = NativeWorkerBridge.__new__(NativeWorkerBridge)
        bridge._native = lambda: types.SimpleNamespace(prefill_streams=owners)
        self.assertEqual(bridge._drain_prefill_streams(), 2)
        self.assertEqual([stream.synchronized for stream in owners.streams], [1, 1])
        bridge._native = lambda: types.SimpleNamespace()
        self.assertEqual(bridge._drain_prefill_streams(), 0)


if __name__ == '__main__':
    unittest.main()
