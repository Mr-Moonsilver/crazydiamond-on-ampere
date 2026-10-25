"""CPU-only tests for the E3 green-context split, split decode bank and worker seams (D4).

No torch, no vLLM, no CUDA. The driver helper, torch and the stock cudagraph
manager are fakes with explicit boundaries; they exercise construction,
capture/replay/transition ordering, pins and inverses. They cannot establish
native readiness, SM containment or cross-context progress.
"""
import ast
import contextlib
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import green_context  # noqa: E402
import split_bank  # noqa: E402
import worker_hooks  # noqa: E402
from graph_router import ExecutionDescriptor  # noqa: E402
from green_context import DriverBinding, GreenContextError, GreenSplit  # noqa: E402
from split_bank import (  # noqa: E402
    RouteController,
    SplitBankError,
    SplitCaptureBackend,
    SplitDecodeBank,
)

REFERENCE = ROOT / 'reference/vllm-src'


# --- fakes ------------------------------------------------------------------

class FakeStream:
    def __init__(self, token, name='stream'):
        self.cuda_stream = token
        self.name = name
        self.synchronized = 0

    def synchronize(self):
        self.synchronized += 1


class FakeEvent:
    def __init__(self, enable_timing=False):
        self.stream = None
        self.synchronized = 0

    def record(self, stream=None):
        self.stream = stream

    def synchronize(self):
        self.synchronized += 1

    def query(self):
        return True


class FakeTensor:
    registry = {}

    def __init__(self, n, fill):
        self.values = [fill] * n
        self.ptr = 1000 + len(FakeTensor.registry)
        FakeTensor.registry[self.ptr] = self

    def data_ptr(self):
        return self.ptr

    def numel(self):
        return len(self.values)

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class FakeCuda:
    def __init__(self, total_sms, default_stream):
        self.total_sms = total_sms
        self._stack = [default_stream]
        self.pools = 0
        self.Event = FakeEvent

    def get_device_properties(self, index):
        return types.SimpleNamespace(multi_processor_count=self.total_sms)

    def ExternalStream(self, token, device=None):
        return FakeStream(token, 'external')

    def synchronize(self, index=None):
        return None

    def current_stream(self, device=None):
        return self._stack[-1]

    def current_device(self):
        return 0

    @contextlib.contextmanager
    def stream(self, stream):
        self._stack.append(stream)
        try:
            yield stream
        finally:
            self._stack.pop()

    def graph_pool_handle(self):
        self.pools += 1
        return (7, self.pools)


class FakeAccelerator:
    def __init__(self):
        self.free = 10_000_000
        self.synchronized = 0

    def synchronize(self):
        self.synchronized += 1

    def get_memory_info(self):
        return (self.free, 24_000_000_000)


def fake_torch(total_sms=82, default_stream=None):
    default_stream = default_stream or FakeStream(1, 'default')
    torch = types.SimpleNamespace()
    torch.cuda = FakeCuda(total_sms, default_stream)
    torch.accelerator = FakeAccelerator()
    torch.int32 = 'int32'
    torch.device = lambda kind, index=0: types.SimpleNamespace(type=kind, index=index)
    torch.full = lambda shape, fill, dtype=None, device=None: FakeTensor(shape[0], fill)
    return torch


class FakeBinding:
    """Driver boundary: returns the split counts and marks probe SMs."""

    def __init__(self, total=82, overlap=False, alias=False, lie=False):
        self.total = total
        self.overlap = overlap
        self.alias = alias
        self.lie = lie
        self.created = []
        self.closed = []
        self.streams = {}

    def create(self, device, requested):
        dc = requested
        pc = self.total - requested
        if self.lie:
            pc -= 1
        handle = object()
        ds, ps = 500 + len(self.created) * 2, 501 + len(self.created) * 2
        if self.alias:
            ps = ds
        self.streams[ds] = ('decode', dc)
        self.streams[ps] = ('prefill', dc)
        self.created.append(handle)
        return handle, ds, ps, dc, pc, 9000 + ds, 9000 + ps

    def close(self, handle):
        self.closed.append(handle)
        return 0

    def probe(self, pointer, blocks, stream_token, cycles):
        role, dc = self.streams[stream_token]
        tensor = FakeTensor.registry[pointer]
        if role == 'decode':
            sms = list(range(dc))
        else:
            sms = list(range(dc, self.total))
            if self.overlap:
                sms[0] = 0
        for i in range(blocks):
            tensor.values[i] = sms[i % len(sms)]
        return 0

    def identity(self):
        return {'library_sha256': 'f' * 64, 'abi': 1}


class FakeMode:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


FULL = FakeMode('FULL')
PIECEWISE = FakeMode('PIECEWISE')


class FakeDesc:
    def __init__(self, mode, num_tokens, num_reqs):
        self.cg_mode = mode
        self.num_tokens = num_tokens
        self.num_reqs = num_reqs
        self.uniform_token_count = 1 if mode is FULL else None
        self.max_query_len = None
        self.num_active_loras = 0
        self.num_ubatches = 1

    def __hash__(self):
        return hash((self.cg_mode.name, self.num_tokens))

    def __eq__(self, other):
        return isinstance(other, FakeDesc) and (self.cg_mode.name, self.num_tokens) == (
            other.cg_mode.name, other.num_tokens)


class FakeGraph:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class FakeManager:
    """Stands in for the pinned ModelCudaGraphManager after worker_hooks' transform."""

    instances = []

    def __init__(self, vllm_config, device, cudagraph_mode, decode_query_len,
                 lora_capture_cases=None, varlen_decode=False):
        self.vllm_config = vllm_config
        self.device = device
        self.cudagraph_mode = cudagraph_mode
        self.decode_query_len = decode_query_len
        self.lora_capture_cases = lora_capture_cases
        self.varlen_decode = varlen_decode
        self.pool = ('stock', 0)
        self._capture_descs = {}
        self.graphs = {}
        self.capture_calls = []
        self.capture_contexts = []
        self.replayed = []
        self.capture_bytes = 4096
        self.fail_capture = False
        FakeManager.instances.append(self)

    def capture(self, **kwargs):
        self.capture_calls.append(kwargs)
        # Transformed cudagraph_utils asks the manager for its capture context.
        self.capture_contexts.append(worker_hooks.capture_context(self))
        if self.fail_capture:
            raise RuntimeError('capture failed')
        for mode, descs in self._capture_descs.items():
            for desc in descs:
                self.graphs[desc] = FakeGraph()
        kwargs['model'].torch.accelerator.free -= self.capture_bytes

    def run_fullgraph(self, desc):
        self.replayed.append(desc)
        return ('hidden', desc.num_tokens)


def fake_runner(torch, stock_descs=None):
    stock = FakeManager('cfg', 'dev', FakeMode('FULL_AND_PIECEWISE'), 1)
    stock._capture_descs = {
        PIECEWISE: [FakeDesc(PIECEWISE, 512, None)],
        FULL: stock_descs or [FakeDesc(FULL, 8, 8), FakeDesc(FULL, 4, 4)],
    }
    main = torch.cuda._stack[0]
    runner = types.SimpleNamespace(
        vllm_config='cfg', device=types.SimpleNamespace(index=0), cudagraph_manager=stock,
        lora_config=None, model=types.SimpleNamespace(torch=torch), model_state='state',
        input_buffers='inputs', intermediate_tensors=None, block_tables='tables',
        attn_groups=[], kv_cache_config='kv', pcp_manager=None,
        main_stream=main, output_copy_stream=FakeStream(2, 'copy'),
        pp_handler=types.SimpleNamespace(main_stream=main, broadcast_stream=FakeStream(3, 'bcast')),
    )
    return runner


def install_fake_vllm(stream_marker):
    module = types.ModuleType('vllm.distributed.parallel_state')

    class GraphCaptureContext:
        def __init__(self, stream):
            self.stream = stream
            stream_marker.append(stream)

    module.GraphCaptureContext = GraphCaptureContext
    pkg = types.ModuleType('vllm')
    dist = types.ModuleType('vllm.distributed')
    sys.modules['vllm'] = pkg
    sys.modules['vllm.distributed'] = dist
    sys.modules['vllm.distributed.parallel_state'] = module


def remove_fake_vllm():
    for name in ('vllm.distributed.parallel_state', 'vllm.distributed', 'vllm'):
        sys.modules.pop(name, None)


# --- green context ----------------------------------------------------------

class GreenSplitTests(unittest.TestCase):
    def setUp(self):
        FakeTensor.registry.clear()
        self.torch = fake_torch()

    def test_create_measures_and_describes(self):
        binding = FakeBinding()
        split = GreenSplit(binding, 0, 44, torch_module=self.torch).create()
        self.assertEqual((split.decode_sms, split.prefill_sms, split.total_sms), (44, 38, 82))
        self.assertNotEqual(split.decode_stream_token, split.prefill_stream_token)
        self.assertNotEqual(split.decode_context_id, split.prefill_context_id)
        self.assertEqual(split.decode_stream.cuda_stream, split.decode_stream_token)
        record = split.describe()
        self.assertEqual(record['basis'], 'driver-reported')
        self.assertEqual(record['helper']['abi'], 1)
        result = split.membership()
        self.assertTrue(result.ok)
        self.assertEqual(len(result.decode_sms), 44)
        self.assertEqual(split.describe()['basis'], 'measured')
        self.assertTrue(split.close())
        self.assertEqual(split.state, 'closed')
        self.assertEqual(binding.closed, binding.created)
        with self.assertRaises(GreenContextError):
            split.close()

    def test_membership_detects_overlap(self):
        split = GreenSplit(FakeBinding(overlap=True), 0, 34, torch_module=self.torch).create()
        self.assertFalse(split.membership().ok)
        self.assertEqual(split.describe()['basis'], 'driver-reported')
        self.assertFalse(split.describe()['membership']['disjoint'])

    def test_bad_requests_fail_closed(self):
        with self.assertRaises(GreenContextError):
            GreenSplit(FakeBinding(), 0, 82, torch_module=self.torch).create()
        with self.assertRaises(GreenContextError):
            GreenSplit(FakeBinding(), 0, 0, torch_module=self.torch)
        with self.assertRaises(GreenContextError):
            GreenSplit(object(), 0, 44, torch_module=self.torch)
        lying = FakeBinding(lie=True)
        with self.assertRaises(GreenContextError):
            GreenSplit(lying, 0, 44, torch_module=self.torch).create()
        self.assertEqual(len(lying.closed), 1, 'a lying split is unwound')
        aliased = FakeBinding(alias=True)
        with self.assertRaises(GreenContextError):
            GreenSplit(aliased, 0, 44, torch_module=self.torch).create()
        self.assertEqual(len(aliased.closed), 1)

    def test_declared_split_and_open(self):
        with self.assertRaises(GreenContextError):
            green_context.declared_decode_sms({})
        with self.assertRaises(GreenContextError):
            green_context.declared_decode_sms({'FLASH_S5_DECODE_SMS': 'many'})
        self.assertEqual(green_context.declared_decode_sms({'FLASH_S5_DECODE_SMS': '44'}), 44)
        with self.assertRaises(GreenContextError):
            green_context.open_split(0, env={'FLASH_S5_DECODE_SMS': '44'}, torch_module=self.torch)
        split = green_context.open_split(0, env={'FLASH_S5_DECODE_SMS': '34'},
                                         torch_module=self.torch, binding=FakeBinding())
        self.assertEqual(split.decode_sms, 34)

    def test_driver_binding_checks_abi_and_file(self):
        class FakeCtypes:
            c_uint64 = c_int = c_uint = c_void_p = object
            POINTER = staticmethod(lambda t: t)

            def __init__(self, abi):
                self.abi = abi

            def CDLL(self, path):
                lib = types.SimpleNamespace()
                lib.s5_abi = lambda: self.abi
                lib.s5_create = lib.s5_close = lib.s5_probe = lambda *a: 0
                return lib

        with self.assertRaises(GreenContextError):
            DriverBinding('/nonexistent/flash_s5_green.so', ctypes_module=FakeCtypes(1))
        with tempfile.NamedTemporaryFile(suffix='.so') as handle:
            handle.write(b'not a library')
            handle.flush()
            with self.assertRaises(GreenContextError):
                DriverBinding(handle.name, ctypes_module=FakeCtypes(2))
            binding = DriverBinding(handle.name, ctypes_module=FakeCtypes(1))
            identity = binding.identity()
            self.assertEqual(identity['abi'], 1)
            self.assertEqual(len(identity['library_sha256']), 64)
            self.assertEqual(len(identity['helper_source_sha256']), 64)

    def test_helper_source_declares_abi(self):
        source = (ROOT / 'patches/s5_engine/native/flash_s5_green.cu').read_text()
        self.assertIn('#define FLASH_S5_GREEN_ABI %d' % green_context.ABI_VERSION, source)
        for symbol in ('s5_abi', 's5_create', 's5_close', 's5_probe'):
            self.assertIn('extern "C" int %s(' % symbol, source)
        self.assertIn('cuGreenCtxCreate', source)
        self.assertIn('cuDevSmResourceSplitByCount', source)


# --- split bank -------------------------------------------------------------

class SplitBankTests(unittest.TestCase):
    def setUp(self):
        FakeManager.instances = []
        self.torch = fake_torch()
        self.markers = []
        install_fake_vllm(self.markers)
        self.split = GreenSplit(FakeBinding(), 0, 44, torch_module=self.torch).create()
        self.runner = fake_runner(self.torch)
        self.restored = []

    def tearDown(self):
        remove_fake_vllm()

    def bank(self):
        return SplitDecodeBank(self.runner, self.split, torch_module=self.torch)

    def test_manager_is_full_only_with_own_pool_on_green_stream(self):
        bank = self.bank()
        manager = bank.build_manager()
        self.assertIsNot(manager, self.runner.cudagraph_manager)
        self.assertEqual(list(manager._capture_descs), [FULL])
        self.assertNotEqual(manager.pool, self.runner.cudagraph_manager.pool)
        context = worker_hooks.capture_context(manager)
        self.assertIs(context.stream, self.split.decode_stream)
        self.assertIsNone(worker_hooks.capture_context(self.runner.cudagraph_manager))
        with self.assertRaises(SplitBankError):
            bank.build_manager()

    def test_capture_measures_bytes_and_restores_stock_pool(self):
        bank = self.bank()
        measured = bank.capture(set_graph_pool_id=self.restored.append)
        self.assertEqual(measured, 4096)
        self.assertEqual(self.restored, [self.runner.cudagraph_manager.pool])
        self.assertEqual(len(bank.handles), 2)
        self.assertEqual(bank.manager.capture_contexts[0].stream, self.split.decode_stream)
        kwargs = bank.manager.capture_calls[0]
        self.assertEqual(kwargs['model'], self.runner.model)
        self.assertFalse(kwargs['has_lora'])
        context = bank.execution_context()
        self.assertEqual(context.basis, 'measured')
        self.assertEqual(context.reserved_bytes, 4096)
        self.assertEqual(context.measured_sm, 44)
        self.assertEqual(context.stream_token, self.split.decode_stream_token)
        self.assertEqual(context.context_id, self.split.decode_context_id)
        with self.assertRaises(SplitBankError):
            bank.capture()

    def test_capture_failure_restores_pool_and_stays_uncaptured(self):
        bank = self.bank()
        bank.build_manager().fail_capture = True
        with self.assertRaises(RuntimeError):
            bank.capture(set_graph_pool_id=self.restored.append)
        self.assertFalse(bank.captured)
        self.assertEqual(self.restored, [self.runner.cudagraph_manager.pool])

    def test_lora_and_missing_full_descriptors_refused(self):
        self.runner.lora_config = object()
        with self.assertRaises(SplitBankError):
            self.bank()
        self.runner.lora_config = None
        self.runner.cudagraph_manager._capture_descs = {PIECEWISE: [FakeDesc(PIECEWISE, 8, None)]}
        with self.assertRaises(SplitBankError):
            self.bank()

    def test_backend_captures_once_and_guards_replay_stream(self):
        bank = self.bank()
        backend = SplitCaptureBackend(bank, set_graph_pool_id=self.restored.append)
        context = bank.execution_context()
        key = types.SimpleNamespace(descriptor=bank.descriptors()[0])
        handle = backend.capture('decode-split', key, 'lane', context)
        self.assertTrue(bank.captured)
        other = backend.capture('decode-split', types.SimpleNamespace(descriptor=bank.descriptors()[1]),
                                'lane', context)
        self.assertEqual(len(bank.manager.capture_calls), 1, 'one capture for the whole bank')
        self.assertIsNot(handle, other)
        wrong = ExecutionDescriptor('full', 999, 999, 1)
        with self.assertRaises(SplitBankError):
            backend.capture('decode-split', types.SimpleNamespace(descriptor=wrong), 'lane', context)
        bad_context = types.SimpleNamespace(stream_token=1, context_id=context.context_id)
        with self.assertRaises(SplitBankError):
            backend.capture('decode-split', key, 'lane', bad_context)
        # Replay must run with the green decode stream current.
        with self.assertRaises(SplitBankError):
            backend.replay(handle, key, 'lane', context)
        with self.torch.cuda.stream(self.split.decode_stream):
            output = backend.replay(handle, key, 'lane', context)
        self.assertEqual(output, ('hidden', 8))
        self.assertEqual(handle.replays, 1)
        with self.assertRaises(SplitBankError):
            backend.replay(handle, types.SimpleNamespace(descriptor=bank.descriptors()[1]), 'lane', context)
        self.assertTrue(backend.destroy_graph(handle))
        self.assertTrue(handle.destroyed)
        with self.assertRaises(SplitBankError):
            backend.destroy_graph(handle)
        with self.assertRaises(SplitBankError):
            backend.destroy_event(object())

    def test_route_transitions_drain_and_swap_streams(self):
        bank = self.bank()
        controller = RouteController(self.runner, bank, torch_module=self.torch)
        with self.assertRaises(SplitBankError):
            controller.enter_split()  # not captured yet
        bank.capture()
        full_main = self.runner.main_stream
        record = controller.enter_split()
        self.assertEqual((record.generation, record.source, record.target), (1, 'full', 'split'))
        self.assertEqual(record.drained_streams, 3, 'main, output copy and PP broadcast')
        self.assertIs(self.runner.main_stream, self.split.decode_stream)
        self.assertIs(self.runner.pp_handler.main_stream, self.split.decode_stream)
        self.assertIs(self.runner._flash_s5_decode_stream, self.split.decode_stream)
        with self.assertRaises(SplitBankError):
            controller.enter_split()
        with split_bank.decode_scope(self.runner):
            self.assertIs(self.torch.cuda.current_stream(), self.split.decode_stream)
        with self.assertRaises(SplitBankError):
            split_bank.shutdown(self.runner)
        record = controller.leave_split()
        self.assertEqual((record.generation, record.source, record.target), (2, 'split', 'full'))
        self.assertIs(self.runner.main_stream, full_main)
        self.assertIs(self.runner.pp_handler.main_stream, full_main)
        self.assertIsNone(self.runner._flash_s5_decode_stream)
        self.assertEqual(len(controller.transitions), 2)
        self.assertTrue(controller.admitting)

    def test_install_and_shutdown_order(self):
        bank, backend, controller = split_bank.install(self.runner, self.split, torch_module=self.torch)
        self.assertEqual(self.runner._flash_s5_graph_capture, backend.capture)
        self.assertEqual(len(self.runner._flash_s5_split_descriptors), 2)
        with self.assertRaises(SplitBankError):
            split_bank.install(self.runner, self.split, torch_module=self.torch)
        bank.capture()
        graphs = list(bank.manager.graphs.values())
        self.assertTrue(split_bank.shutdown(self.runner))
        self.assertEqual([g.reset_calls for g in graphs], [1, 1])
        self.assertEqual(self.split.state, 'closed')
        self.assertEqual(bank.handles, {})

    def test_prefill_scope_uses_green_prefill_stream(self):
        self.runner._flash_s5_split = self.split
        self.runner._flash_s5_torch = self.torch
        with split_bank.prefill_scope(self.runner) as stream:
            self.assertIs(stream, self.split.prefill_stream)
            self.assertIs(self.torch.cuda.current_stream(), self.split.prefill_stream)
        self.runner._flash_s5_split = None
        with self.assertRaises(SplitBankError):
            with split_bank.prefill_scope(self.runner):
                pass


# --- worker hooks -----------------------------------------------------------

class WorkerHookSeamTests(unittest.TestCase):
    def test_workspace_lanes(self):
        self.assertEqual(worker_hooks.workspace_lanes(1, {}), 1)
        self.assertEqual(worker_hooks.workspace_lanes(1, {'FLASH_S5_OVERLAP': '1'}), 2)
        self.assertEqual(worker_hooks.prefill_workspace_lane(1), 1)
        with self.assertRaises(ValueError):
            worker_hooks.workspace_lanes(0, {})

    def test_after_runner_init_off_and_on(self):
        worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(),
                                       device=types.SimpleNamespace(index=0))
        self.assertIsNone(worker_hooks.after_runner_init(worker, env={}))
        self.assertFalse(hasattr(worker.model_runner, '_flash_s5_split'))
        torch = fake_torch()
        original = green_context.open_split
        green_context.open_split = lambda index, env=None: GreenSplit(
            FakeBinding(), index, 44, torch_module=torch).create()
        try:
            split = worker_hooks.after_runner_init(worker, env={'FLASH_S5_OVERLAP': '1'})
        finally:
            green_context.open_split = original
        self.assertIs(worker.model_runner._flash_s5_split, split)
        self.assertEqual(worker.model_runner._flash_s5_native_identity['basis'], 'measured')
        with self.assertRaises(ValueError):
            worker_hooks.after_runner_init(worker, env={'FLASH_S5_OVERLAP': '1'})

    def test_decode_scope_is_null_when_off(self):
        worker = types.SimpleNamespace(model_runner=types.SimpleNamespace())
        with worker_hooks.decode_scope(worker):
            pass
        worker = types.SimpleNamespace(model_runner=None)
        with worker_hooks.decode_scope(worker):
            pass


class WorkerHookTransformTests(unittest.TestCase):
    def setUp(self):
        if not REFERENCE.is_dir():
            self.skipTest('reference/vllm-src is not available on this host')
        self.files = {path: (REFERENCE / path).read_text() for path in worker_hooks.PINS}

    def test_pins_match_reference(self):
        import hashlib

        for path, digest in worker_hooks.PINS.items():
            self.assertEqual(hashlib.sha256(self.files[path].encode()).hexdigest(), digest, path)

    def test_generate_and_exact_inverse(self):
        out = worker_hooks.generate(self.files)
        self.assertEqual(set(out), set(worker_hooks.PINS))
        for path, text in out.items():
            ast.parse(text)
            self.assertNotEqual(text, self.files[path])
            self.assertEqual(worker_hooks.revert_one(path, text, expected=self.files[path]),
                             self.files[path])
            with self.assertRaises(ValueError):
                worker_hooks.generate_one(path, text)
            with self.assertRaises(ValueError):
                worker_hooks.generate_one(path, self.files[path] + '\n# drift\n')
            with self.assertRaises(ValueError):
                worker_hooks.revert_one(path, self.files[path])

    def test_worker_call_sites(self):
        worker = worker_hooks.generate(self.files)['v1/worker/gpu_worker.py']
        self.assertEqual(worker.count('_flash_s5_worker.decode_scope(self)'), 2)
        self.assertEqual(worker.count('def _flash_s5_stock_execute_model('), 1)
        self.assertEqual(worker.count('def execute_model('), 1)
        self.assertEqual(worker.count('_flash_s5_worker.after_runner_init(self)'), 1)
        self.assertIn('_flash_s5_worker.workspace_lanes(', worker)
        tree = ast.parse(worker)
        names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertIn('_flash_s5_stock_execute_model', names)
        self.assertIn('execute_model', names)
        graphs = worker_hooks.generate(self.files)['v1/worker/gpu/cudagraph_utils.py']
        self.assertEqual(graphs.count('graph_capture_context=_flash_s5_capture_context(self)'), 1)
        self.assertNotIn('with graph_capture(device=self.device):', graphs)

    def test_unknown_path_refused(self):
        with self.assertRaises(ValueError):
            worker_hooks.generate_one('v1/engine/core.py', 'x = 1\n')
        with self.assertRaises(ValueError):
            worker_hooks.generate({})


if __name__ == '__main__':
    unittest.main()
