"""CPU-only tests for the E3 worker-native startup composition and transition RPC (D4)."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
sys.path.insert(0, str(ROOT / 'harness'))

import native_startup  # noqa: E402
import lane_state  # noqa: E402
from green_context import GreenSplit  # noqa: E402
from native_startup import NativeStartupError, WorkerNative, install_worker_native, transition_rpc  # noqa: E402
from test_e3_green_native import FakeBinding, FakeManager, FakeTensor, fake_runner, fake_torch, install_fake_vllm, remove_fake_vllm  # noqa: E402


class FakeNccl:
    def __init__(self, group, device):
        self.group = group
        self.disabled = False
        self.world_size = 2
        self.rank = 0
        self.calls = []

    def all_reduce(self, t, stream=None):
        self.calls.append('all_reduce')
        return types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(tolist=lambda: [2.0] * t.numel()))

    def send(self, tensor, dst, stream=None):
        self.calls.append('send')

    def recv(self, tensor, src, stream=None):
        self.calls.append('recv')
        tensor.values[:] = [0] * tensor.numel()


class StartTorch:
    """fake_torch plus the tensor helpers the lane modules need."""

    @staticmethod
    def make(total_sms=82):
        torch = fake_torch(total_sms)
        torch.int64 = 'int64'
        torch.float32 = 'float32'

        def _numel(shape):
            n = 1
            for d in shape:
                n *= d
            return n

        class Tensor(FakeTensor):
            def __init__(self, n, fill, shape, dtype):
                super().__init__(n, fill)
                self.shape = shape
                self.dtype = dtype
                self.device = 'cuda:0'

            def element_size(self):
                return 8 if self.dtype == 'int64' else 2

            def copy_(self, other, non_blocking=False):
                return self

            def __getitem__(self, item):
                return self

        torch.empty = lambda shape, dtype=None, device=None, pin_memory=False: Tensor(
            _numel(shape), 0, tuple(shape), dtype)
        torch.ones = lambda shape, dtype=None, device=None: Tensor(_numel(shape), 1, tuple(shape), dtype)
        torch.full = lambda shape, fill, dtype=None, device=None: Tensor(_numel(shape), fill, tuple(shape), dtype)
        torch.tensor = lambda values, dtype=None: Tensor(len(values), 0, (len(values),), dtype)
        torch.cuda.Stream = lambda device=None: types.SimpleNamespace(cuda_stream=900, synchronize=lambda: None)
        return torch


def fake_layers(count=4, ple_at=1):
    layers = []
    for idx in range(count):
        embedding = None
        if idx == ple_at:
            embedding = types.SimpleNamespace(
                supports_prefetch=True, etp_data_parallel_size=1,
                _prefetch_buffer=types.SimpleNamespace(shape=(4096, 4, 128), dtype='bf16'),
                _prefetch_stream=types.SimpleNamespace(cuda_stream=50))
        layers.append(types.SimpleNamespace(
            ple=types.SimpleNamespace(ngram_embedding=embedding) if embedding else None))
    return layers


class StartupTests(unittest.TestCase):
    def setUp(self):
        FakeManager.instances = []
        FakeTensor.registry.clear()
        self.torch = StartTorch.make()
        self.markers = []
        install_fake_vllm(self.markers)
        self.runner = fake_runner(self.torch)
        self.runner.vllm_config = types.SimpleNamespace(model_config=types.SimpleNamespace(dtype='bf16'))
        self.runner.model = types.SimpleNamespace(
            torch=self.torch, layers=fake_layers(), start_layer=0, end_layer=4,
            config=types.SimpleNamespace(hc_count=4, hidden_size=2560), _mtp_hidden_buffer=None,
            _flash_e3_forward_prefix=lambda *a: 'h0',
            _flash_e3_forward_layer=lambda idx, *a: (f'h{idx + 1}', 'b', 'i'),
            _flash_e3_forward_suffix=lambda *a: {'hidden_states': 'hc'},
            _flash_e3_is_last_stage=lambda: False)
        self.split = GreenSplit(FakeBinding(), 0, 44, torch_module=self.torch).create()
        self.runner._flash_s5_split = self.split
        # The worker runtime state is what build_worker_runtime returns: one
        # persistent decode storage owner per worker incarnation.
        self.state = lane_state.WorkerState(0, engine_epoch=0, engine_bound=False)
        self.runner._flash_s5_worker = types.SimpleNamespace(
            runtime=types.SimpleNamespace(state=self.state), operation_device=None)
        self.tp = types.SimpleNamespace(cpu_group='tp', world_size=2, device_communicator=object())
        self.pp = types.SimpleNamespace(cpu_group='pp', world_size=2, rank_in_group=0,
                                        is_first_rank=True, is_last_rank=False)
        # Owner discovery walks the pinned vLLM model family, which this CPU
        # fixture does not construct. Use the real PrefillStreamOwners API over
        # the fixture's actual PLE embedding; discovery itself is covered in
        # test_e3_prefill_streams.py.
        import model_hooks
        embedding = self.runner.model.layers[1].ple.ngram_embedding
        self.ple_embedding = embedding
        self.ple_origin_stream = embedding._prefetch_stream

        def discover(runner, split, *, pinned=None):
            return model_hooks.PrefillStreamOwners(split, [(embedding, '_prefetch_stream')])

        self._discover = patch.object(model_hooks.PrefillStreamOwners, 'discover', discover)
        self._discover.start()

    def tearDown(self):
        self._discover.stop()
        remove_fake_vllm()

    def install(self):
        return install_worker_native(self.runner, chunk_tokens=1024, torch_module=self.torch,
                                     pynccl_class=FakeNccl, tp_group=self.tp, pp_group=self.pp)

    def test_install_composes_in_order_and_reports_inventory(self):
        native = self.install()
        self.assertIsInstance(native, WorkerNative)
        self.assertIs(self.runner._flash_s5_native, native)
        self.assertTrue(native.comms.tp.warmed)
        self.assertTrue(native.comms.pp.warmed)
        self.assertEqual(native.comms.pp.max_tokens, 1024)
        self.assertEqual(native.comms.pp.hidden_width, 4 * 2560)
        self.assertEqual(native.ple_layer, 1)
        inventory = native.inventory()
        self.assertEqual(inventory['pp_channel_bytes'], 2 * (1024 * 10240 * 2 + 64))
        self.assertEqual(inventory['ple_scratch_bytes'], 1024 * 4 * 128 * 2)
        self.assertEqual(inventory['split_bank_basis'], 'uncaptured')
        self.assertEqual(inventory['route'], 'full')
        self.assertEqual(inventory['split']['decode_sms'], 44)
        # The PLE prefetch stream is the owned prefill-green auxiliary, not a
        # private primary-context stream.
        self.assertEqual(native.prefill_streams.record()['basis'], 'prefill-green')
        self.assertEqual(inventory['prefill_streams']['auxiliary_streams'], 1)
        self.assertEqual(native.ple_scratch.stream_basis, 'prefill-green')
        self.assertIs(native.ple_scratch.stream,
                      native.prefill_streams.replacements[0])
        self.assertEqual(native.drain_prefill_streams(), 1)
        with self.assertRaises(NativeStartupError):
            self.install()
        lane = native.lane_forward(dict(positions='p', input_ids='i', query_start_loc='q', ngram_context='n'),
                                   clock=lambda: 0)
        lane.begin()
        for idx in range(4):
            lane.layer(idx)
        lane.finish()
        self.assertEqual(lane.output, {'hidden_states': 'hc'})
        # Every segment restored the stock owner exactly.
        self.assertIs(self.ple_embedding._prefetch_stream, self.ple_origin_stream)

    def test_install_refuses_without_the_prefill_helper_extension(self):
        from green_context import GreenContextError

        class NoPrefill(FakeBinding):
            prefill_aux_create = None

        self.runner._flash_s5_split = GreenSplit(
            NoPrefill(), 0, 44, torch_module=self.torch).create()
        with self.assertRaises(GreenContextError):
            self.install()

    def test_install_refusals(self):
        with self.assertRaises(NativeStartupError):
            install_worker_native(self.runner, chunk_tokens=0, torch_module=self.torch,
                                  pynccl_class=FakeNccl, tp_group=self.tp, pp_group=self.pp)
        self.runner._flash_s5_split = None
        with self.assertRaises(NativeStartupError):
            self.install()
        self.runner._flash_s5_split = self.split
        self.runner.model.config = types.SimpleNamespace()
        with self.assertRaises(NativeStartupError):
            self.install()

    def test_graph_builder_registers_split_bank(self):
        native = self.install()
        built = {}

        class Router:
            def __init__(self, state, handle, compilation, **kwargs):
                built['args'] = (state, handle, compilation, kwargs)
                self.banks = {}

            def add_bank(self, name, context, descriptors, *, context_factory):
                built['factory'] = context_factory
                self.banks[name] = (context, descriptors)

        handle = self.state.decode.handle
        self.assertIsNotNone(self.state.decode_owner)
        self.assertEqual(handle.lane_id, self.state.decode_owner.lane_id)
        build = native.graph_builder(router_class=Router, full_context_id=0, full_pool_id=0,
                                     full_allocated_bytes=5)
        router = build(self.runner)
        self.assertIs(built['args'][0], self.state)
        self.assertIs(built['args'][1], handle)
        self.assertIs(built['args'][2], native.compilation)
        self.assertIs(built['args'][3]['backend'], native.backend)
        context, descriptors = router.banks['decode-split']
        self.assertEqual(context.measured_sm, 44)
        self.assertEqual(len(descriptors), 2)
        # The refreshed factory must describe the same bank identity and agree
        # with the graph router's own non-mutating verification.
        refreshed = built['factory']()
        self.assertEqual((refreshed.context_id, refreshed.pool_id, refreshed.stream_token),
                         (context.context_id, context.pool_id, context.stream_token))
        with self.torch.cuda.stream(self.split.decode_stream):
            self.assertEqual(router.probe(), (self.split.decode_context_id, self.split.decode_stream_token))
        with self.assertRaises(NativeStartupError):
            build(self.runner)
        # Without the retained runtime state the builder refuses rather than
        # accepting a guessed context/pool identity.
        self.runner._flash_e3_graph_builder_used = False
        saved, native.state = native.state, None
        server, self.runner._flash_s5_worker = self.runner._flash_s5_worker, None
        try:
            with self.assertRaises(NativeStartupError):
                build(self.runner)
        finally:
            native.state = saved
            self.runner._flash_s5_worker = server

    def test_real_graph_router_builder_registers_the_retained_bank(self):
        native = self.install()
        built = native.graph_builder(
            full_context_id=native.full_identity.context_id,
            full_pool_id=native.full_identity.pool_id,
            full_allocated_bytes=native.full_identity.allocated_bytes)
        router = built(self.runner)
        self.assertIs(router, native.router)
        self.assertIs(self.runner._flash_e3_graph_router_split, native)
        self.assertEqual(router.current_mode, 'ordinary')
        self.assertIn('decode-split', router.banks)
        bank = router.banks['decode-split']
        self.assertEqual(bank.context.context_id, self.split.decode_context_id)
        self.assertEqual(bank.context.stream_token, self.split.decode_stream_token)
        self.assertFalse(router.engine_owner_bound)
        # Until the engine bootstrap binds this worker, no route may switch.
        from graph_router import GraphRoutingError, TransitionAuthorization
        with self.assertRaises(GraphRoutingError):
            router.validate_transition(
                'decode-split',
                TransitionAuthorization(1, self.split.decode_context_id, 1, 1, 'test'))

    def test_transition_rpc_validates_and_acks_after_switch(self):
        native = self.install()
        native.bank.capture()
        command = dict(engine_epoch=1, expected_route_generation=0, route_generation=1,
                       target='decode-split', storage_epoch=None, decode_replay_watermark=17)
        with self.assertRaises(NativeStartupError):
            transition_rpc(self.runner, dict(command, engine_epoch=2), engine_epoch=1)
        with self.assertRaises(NativeStartupError):
            transition_rpc(self.runner, dict(command, expected_route_generation=5))
        with self.assertRaises(NativeStartupError):
            transition_rpc(self.runner, dict(command, route_generation=2))
        with self.assertRaises(NativeStartupError):
            transition_rpc(self.runner, dict(command, target='sideways'))
        ack = transition_rpc(self.runner, command, engine_epoch=1)
        self.assertEqual((ack['route'], ack['route_generation']), ('split', 1))
        self.assertEqual(ack['drained_streams'], 3)
        self.assertEqual(ack['decode_replay_watermark'], 17)
        self.assertIs(self.runner.main_stream, self.split.decode_stream)
        with self.assertRaises(NativeStartupError):
            transition_rpc(self.runner, command, engine_epoch=1)  # stale generation
        back = dict(engine_epoch=1, expected_route_generation=1, route_generation=2, target='full')
        ack = transition_rpc(self.runner, back, engine_epoch=1)
        self.assertEqual((ack['route'], ack['route_generation']), ('full', 2))
        self.assertIsNone(self.runner._flash_s5_decode_stream)
        with self.assertRaises(NativeStartupError):
            transition_rpc(types.SimpleNamespace(), back)
        with self.assertRaises(NativeStartupError):
            transition_rpc(self.runner, 'not a mapping')


if __name__ == '__main__':
    unittest.main()
