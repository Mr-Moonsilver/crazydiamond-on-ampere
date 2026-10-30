"""CPU-only tests for the E3 lane forward transform and driver (D4/D2 boundary).

No torch, no vLLM. A fake model carries the generated method names so the
driver's ordering, scopes and bindings are exercised; the real transform is
checked against the pinned model source with an exact inverse.
"""
import ast
import contextlib
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import model_hooks  # noqa: E402
from lane_comm import PrefillTpCommunicator  # noqa: E402
from model_hooks import (  # noqa: E402
    AttributeSwap,
    LaneForward,
    ModelHookError,
    PlePrivateScratch,
    ple_embedding,
)

REFERENCE = ROOT / 'reference/vllm-src'


class ModelOwnershipTests(unittest.TestCase):
    def backbone(self):
        return types.SimpleNamespace(layers=[object()]*4, start_layer=0, end_layer=4)

    def test_only_the_three_pinned_owner_paths_resolve_without_mutation(self):
        backbone = self.backbone()
        causal = types.SimpleNamespace(model=backbone)
        conditional = types.SimpleNamespace(language_model=causal,
                                             config=types.SimpleNamespace(text_config=object()))
        for root in (backbone, causal, conditional):
            self.assertIs(model_hooks.local_text_model(root), backbone)
        self.assertIs(conditional.language_model, causal)
        self.assertIs(causal.model, backbone)

    def test_source_constructor_ownership_assignments_are_the_pinned_paths(self):
        source = (REFERENCE/'models/qwen4_exp/nvidia/model.py').read_text()
        self.assertEqual(model_hooks._sha(source), model_hooks.PINS['models/qwen4_exp/nvidia/model.py'])
        tree = ast.parse(source)
        expected = {'Qwen4ExpForCausalLM': ('model', 'Qwen4ExpModel'),
                    'Qwen4ExpForConditionalGeneration': ('language_model', 'Qwen4ExpForCausalLM')}
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in expected:
                attr, constructor = expected[node.name]
                init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
                assignments = [n for n in ast.walk(init) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                            and t.value.id == 'self' and t.attr == attr for t in n.targets)]
                self.assertEqual(len(assignments), 1)
                self.assertIsInstance(assignments[0].value, ast.Call)
                self.assertEqual(assignments[0].value.func.id, constructor)
                del expected[node.name]
        self.assertFalse(expected)

    def test_missing_or_arbitrary_recursive_owner_paths_refuse(self):
        for root in (None, object(), types.SimpleNamespace(config=object()),
                     types.SimpleNamespace(other=types.SimpleNamespace(model=self.backbone())),
                     types.SimpleNamespace(model=types.SimpleNamespace(model=self.backbone()))):
            with self.subTest(root_type=type(root).__name__), self.assertRaises(ModelHookError):
                model_hooks.local_text_model(root)

    def test_ambiguous_or_invalid_local_layer_owners_refuse(self):
        first, second = self.backbone(), self.backbone()
        first.model = second
        with self.assertRaisesRegex(ModelHookError, 'one local text backbone'):
            model_hooks.local_text_model(first)
        for start, end in ((True, 4), (0, 0), (-1, 4), (0, 5)):
            with self.subTest(start=start,end=end), self.assertRaises(ModelHookError):
                model_hooks.local_text_model(types.SimpleNamespace(layers=[1]*4,
                    start_layer=start,end_layer=end))

    def test_alias_of_one_backbone_is_not_two_distinct_owners(self):
        backbone = self.backbone();backbone.model = backbone
        self.assertIs(model_hooks.local_text_model(backbone), backbone)


class FakeStream:
    def __init__(self, token):
        self.cuda_stream = token


class FakeEvent:
    def __init__(self, enable_timing=False):
        self.stream = None

    def record(self, stream=None):
        self.stream = stream


class FakeTensor:
    def __init__(self, shape, dtype='bf16'):
        self.shape = tuple(shape)
        self.dtype = dtype

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def element_size(self):
        return 2


class FakeCuda:
    def __init__(self):
        self._stack = [FakeStream(1)]
        self.Event = FakeEvent

    def current_stream(self, device=None):
        return self._stack[-1]

    @contextlib.contextmanager
    def stream(self, stream):
        self._stack.append(stream)
        try:
            yield stream
        finally:
            self._stack.pop()

    def Stream(self, device=None):
        return FakeStream(900)


def fake_torch():
    torch = types.SimpleNamespace()
    torch.cuda = FakeCuda()
    torch.empty = lambda shape, dtype=None, device=None: FakeTensor(shape, dtype)
    return torch


class FakeEmbedding:
    supports_prefetch = True
    etp_data_parallel_size = 1

    def __init__(self):
        self._prefetch_buffer = FakeTensor((4096, 4, 128))
        self._prefetch_stream = FakeStream(50)


class FakeLayer:
    def __init__(self, idx, ple=False):
        self.idx = idx
        self.ple = types.SimpleNamespace(ngram_embedding=FakeEmbedding()) if ple else None
        self.mlp_hyper_connection = types.SimpleNamespace(
            combine=lambda h, b, i: ('combined', h, b, i))


class FakeModel:
    def __init__(self, start=0, end=4, last_stage=False, torch=None, tp_group=None, trace=None):
        self.layers = [FakeLayer(i, ple=(i == 1)) for i in range(end)]
        self.start_layer = start
        self.end_layer = end
        self._mtp_hidden_buffer = None
        self._flash_e3_is_last_stage = lambda: last_stage
        self.torch = torch
        self.tp_group = tp_group
        self.trace = trace if trace is not None else []
        self.layer_results = {}

    def _observe(self, what, layer_idx=None):
        current = self.torch.cuda.current_stream().cuda_stream if self.torch else None
        comm = type(self.tp_group.device_communicator).__name__ if self.tp_group else None
        ple = None
        if layer_idx is not None and self.layers[layer_idx].ple is not None:
            ple = self.layers[layer_idx].ple.ngram_embedding._prefetch_stream.cuda_stream
        self.trace.append((what, layer_idx, current, comm, ple))

    def _flash_e3_forward_prefix(self, input_ids, intermediate_tensors, inputs_embeds=None):
        self._observe('prefix')
        return 'h0'

    def _flash_e3_forward_layer(self, layer_idx, hidden_states, block_output, injection,
                                positions, input_ids, query_start_loc, ngram_context):
        self._observe('layer', layer_idx)
        return (f'h{layer_idx + 1}', f'b{layer_idx}', f'i{layer_idx}')

    def _flash_e3_forward_suffix(self, hidden_states, block_output, injection, last_layer_idx):
        self._observe('suffix', last_layer_idx)
        return {'hidden_states': ('combined', hidden_states, block_output, injection)}


class FakeNccl:
    disabled = False
    world_size = 2
    rank = 0

    def all_reduce(self, t, stream=None):
        return t


def inputs():
    return dict(positions='pos', input_ids='ids', query_start_loc='qsl', ngram_context='ng')


class TransformTests(unittest.TestCase):
    def setUp(self):
        if not REFERENCE.is_dir():
            self.skipTest('reference/vllm-src is not available on this host')
        self.path = 'models/qwen4_exp/nvidia/model.py'
        self.source = (REFERENCE / self.path).read_text()

    def test_pin_generate_and_inverse(self):
        import hashlib

        self.assertEqual(hashlib.sha256(self.source.encode()).hexdigest(), model_hooks.PINS[self.path])
        out = model_hooks.generate({self.path: self.source})[self.path]
        ast.parse(out)
        self.assertEqual(model_hooks.revert_one(self.path, out, expected=self.source), self.source)
        with self.assertRaises(ValueError):
            model_hooks.generate_one(self.path, out)
        with self.assertRaises(ValueError):
            model_hooks.generate_one(self.path, self.source + '\n# drift\n')
        with self.assertRaises(ValueError):
            model_hooks.revert_one(self.path, self.source)
        with self.assertRaises(ValueError):
            model_hooks.generate({})

    def test_stock_forward_unchanged_and_methods_added(self):
        out = model_hooks.generate({self.path: self.source})[self.path]
        tree = ast.parse(out)
        model_cls = next(node for node in ast.walk(tree)
                         if isinstance(node, ast.ClassDef) and node.name == 'Qwen4ExpModel')
        names = [node.name for node in model_cls.body if isinstance(node, ast.FunctionDef)]
        for name in ('_flash_e3_forward_prefix', '_flash_e3_forward_layer', '_flash_e3_forward_suffix'):
            self.assertIn(name, names)
        layer_cls = next(node for node in ast.walk(tree)
                         if isinstance(node, ast.ClassDef) and node.name == 'Qwen4ExpDecoderLayer')
        layer_names = [node.name for node in layer_cls.body if isinstance(node, ast.FunctionDef)]
        self.assertIn('_flash_e3_layer_pre', layer_names)
        self.assertIn('_flash_e3_layer_post', layer_names)
        stock = ast.parse(self.source)
        stock_cls = next(node for node in ast.walk(stock)
                         if isinstance(node, ast.ClassDef) and node.name == 'Qwen4ExpModel')
        stock_forward = next(n for n in stock_cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        new_forward = next(n for n in model_cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        self.assertEqual(ast.dump(stock_forward), ast.dump(new_forward))
        lane_layer = next(n for n in model_cls.body if isinstance(n, ast.FunctionDef)
                          and n.name == '_flash_e3_forward_layer')
        dumped = ast.dump(lane_layer)
        self.assertIn('_start_layer_ple_prefetch', dumped)
        self.assertNotIn('deepstack', dumped)
        self.assertNotIn('layer_idx + 1', ast.unparse(lane_layer), 'no PLE look-ahead in the lane')


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.torch = fake_torch()
        self.stream = FakeStream(77)
        self.group = types.SimpleNamespace(device_communicator=types.SimpleNamespace())
        self.comm = PrefillTpCommunicator(FakeNccl(), 'cuda:0', torch_module=self.torch)
        self.model = FakeModel(torch=self.torch, tp_group=self.group)
        self.scratch = PlePrivateScratch(self.model.layers[1].ple.ngram_embedding, 1024,
                                         torch_module=self.torch, device='cuda:0')
        self.contexts = []

        @contextlib.contextmanager
        def fc():
            self.contexts.append('enter')
            yield
            self.contexts.append('exit')

        self.lane = LaneForward(self.model, inputs(), stream=self.stream, tp_group=self.group,
                                tp_comm=self.comm, forward_context=fc, ple_scratch=self.scratch,
                                torch_module=self.torch, clock=lambda: 0)

    def test_segments_run_in_order_under_scopes(self):
        lane = self.lane
        self.assertEqual(lane.remaining(), (0, 1, 2, 3))
        record = lane.begin()
        self.assertEqual(record.kind, 'prefix')
        self.assertIs(record.event.stream, self.stream)
        with self.assertRaises(ModelHookError):
            lane.layer(1)
        for idx in range(4):
            record = lane.layer(idx)
            self.assertEqual((record.kind, record.layer_idx), ('layer', idx))
            self.assertIs(record.event.stream, self.stream)
        with self.assertRaises(ModelHookError):
            lane.layer(4)
        self.assertEqual(lane.remaining(), ())
        record = lane.finish()
        self.assertEqual(record.kind, 'suffix')
        self.assertEqual(lane.output['hidden_states'], ('combined', 'h4', 'b3', 'i3'))
        self.assertEqual(lane.state, 'finished')
        # Every segment ran on the lane stream with the lane communicator installed,
        # and only layer 1 saw the private PLE prefetch stream.
        kinds = [t[0] for t in self.model.trace]
        self.assertEqual(kinds, ['prefix', 'layer', 'layer', 'layer', 'layer', 'suffix'])
        for what, idx, current, comm, ple in self.model.trace:
            self.assertEqual(current, 77)
            self.assertEqual(comm, 'PrefillTpCommunicator')
            if what == 'layer' and idx == 1:
                self.assertEqual(ple, 900, 'private prefetch stream bound during the PLE layer')
        self.assertIs(self.group.device_communicator.__class__, types.SimpleNamespace, 'restored')
        embedding = self.model.layers[1].ple.ngram_embedding
        self.assertEqual(embedding._prefetch_stream.cuda_stream, 50, 'stock PLE stream restored')
        self.assertEqual(self.contexts.count('enter'), 6)
        self.assertEqual(self.contexts.count('exit'), 6)
        self.assertEqual(self.scratch.bytes, 1024 * 4 * 128 * 2)

    def test_last_stage_has_no_suffix_output(self):
        model = FakeModel(start=2, end=4, last_stage=True, torch=self.torch)
        lane = LaneForward(model, inputs(), stream=self.stream, torch_module=self.torch, clock=lambda: 0)
        lane.begin(intermediate_tensors={'hidden_states': 'from-stage-0'})
        with self.assertRaises(ModelHookError):
            lane.layer(0)
        lane.layer(2)
        with self.assertRaises(ModelHookError):
            lane.finish()
        lane.layer(3)
        lane.finish()
        self.assertIsNone(lane.output)
        self.assertEqual([t[0] for t in model.trace], ['prefix', 'layer', 'layer'])

    def test_refusals(self):
        bad = dict(inputs())
        del bad['ngram_context']
        with self.assertRaises(ModelHookError):
            LaneForward(self.model, bad, stream=self.stream)
        with self.assertRaises(ModelHookError):
            LaneForward(self.model, dict(inputs(), deepstack_input_embeds='x'), stream=self.stream)
        self.model._mtp_hidden_buffer = 'buf'
        with self.assertRaises(ModelHookError):
            LaneForward(self.model, inputs(), stream=self.stream)
        self.model._mtp_hidden_buffer = None
        with self.assertRaises(ModelHookError):
            LaneForward(self.model, inputs(), stream=self.stream, tp_group=self.group)
        with self.assertRaises(ModelHookError):
            LaneForward(types.SimpleNamespace(), inputs(), stream=self.stream)
        lane = LaneForward(self.model, inputs(), stream=self.stream, torch_module=self.torch)
        with self.assertRaises(ModelHookError):
            lane.layer(0)
        with self.assertRaises(ModelHookError):
            lane.finish()
        lane.begin()
        with self.assertRaises(ModelHookError):
            lane.begin()
        self.assertEqual(lane.abort('cancelled'), 'cancelled')
        self.assertEqual(lane.state, 'aborted')

    def test_attention_driver_can_pause_a_layer(self):
        calls = []

        class Driver:
            def owns(self, model, layer_idx):
                return layer_idx == 3

            def run_layer(self, lane, layer_idx):
                calls.append(layer_idx)
                if len(calls) == 1:
                    return None  # paused at a bucket boundary
                return ('h4', 'b3', 'i3')

        lane = LaneForward(self.model, inputs(), stream=self.stream, torch_module=self.torch,
                           attention_driver=Driver(), clock=lambda: 0)
        lane.begin()
        for idx in range(3):
            lane.layer(idx)
        record = lane.layer(3)
        self.assertEqual(record.kind, 'layer-paused')
        self.assertEqual(lane.cursor, 3)
        record = lane.layer(3)
        self.assertEqual(record.kind, 'layer')
        self.assertEqual(lane.cursor, 4)
        self.assertEqual(calls, [3, 3])
        lane.finish()

    def test_attribute_swap_and_ple_lookup(self):
        owner = types.SimpleNamespace(a=1, b=2)
        swap = AttributeSwap(owner, {'a': 10})
        with swap:
            self.assertEqual(owner.a, 10)
            with self.assertRaises(ModelHookError):
                with swap:
                    pass
        self.assertEqual(owner.a, 1)
        with self.assertRaises(ModelHookError):
            with AttributeSwap(owner, {'missing': 1}):
                pass
        self.assertIsNone(ple_embedding(self.model, 0))
        self.assertIsNotNone(ple_embedding(self.model, 1))
        with self.assertRaises(ModelHookError):
            PlePrivateScratch(types.SimpleNamespace(), 1024, torch_module=self.torch, device='cuda:0')


if __name__ == '__main__':
    unittest.main()
