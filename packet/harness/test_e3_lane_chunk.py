"""CPU-only tests for the lane's worker-side chunk admission and private inputs (D4).

Fake runner, torch and pinned ports; these check call order, row-scoped
registration, CPU slot arithmetic and the metadata/forward-context flow. They
are not GPU evidence and cannot audit builder scratch sharing.
"""
import ast
import contextlib
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import numpy as np  # noqa: E402

import lane_chunk  # noqa: E402
from e3_types import TransactionId  # noqa: E402
from lane_chunk import ChunkAdmission, LaneChunkError, LaneChunkFactory, LaneTables, Ports  # noqa: E402


# --- fakes ------------------------------------------------------------------

class T:
    """Minimal tensor: list storage, shape, dtype, device, 2-D row views."""

    def __init__(self, values, shape=None, dtype=None, device='cpu'):
        self.values = list(values)
        self.shape = tuple(shape) if shape is not None else (len(self.values),)
        self.dtype = dtype
        self.device = device
        self.parent = None
        self.log = []

    def tolist(self):
        if self.device != 'cpu':
            raise AssertionError('device readback forbidden in lane host plan')
        return list(self.values)

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def element_size(self):
        return 4

    def to(self, device):
        self.device = device
        return self

    def view(self, *shape):
        self.shape = tuple(shape)
        return self

    def _sync_parent(self):
        if self.parent is not None:
            base, offset = self.parent
            base.values[offset:offset + len(self.values)] = self.values
            base._sync_parent()

    def copy_(self, other, non_blocking=False):
        self.values[:len(other.values)] = list(other.values)
        self._sync_parent()
        return self

    def zero_(self):
        self.values = [0] * len(self.values)
        self._sync_parent()
        return self

    def fill_(self, value):
        self.values = [value] * len(self.values)
        self._sync_parent()
        return self

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row, cols = item
            width = self.shape[1]
            base = row * width
            if isinstance(cols, slice):
                lo, hi, _ = cols.indices(width)
                view = T(self.values[base + lo:base + hi], (hi - lo,), self.dtype, self.device)
                view.parent = (self, base + lo)
                return view
            view = T([self.values[base + cols]], (1,), self.dtype, self.device)
            view.parent = (self, base + cols)
            return view
        if isinstance(item, slice):
            lo, hi, _ = item.indices(self.shape[0])
            width = self.numel() // self.shape[0] if self.shape[0] else 0
            view = T(self.values[lo * width:hi * width], (hi - lo,) + self.shape[1:], self.dtype, self.device)
            view.parent = (self, lo * width)
            return view
        view = T([self.values[item]], (1,), self.dtype, self.device)
        view.parent = (self, item)
        return view


def fake_torch():
    torch = types.SimpleNamespace(int32='i32', int64='i64', bool='bool')

    def _numel(shape):
        n = 1
        for d in shape:
            n *= d
        return n

    torch.tensor = lambda values, dtype=None, device='cpu': T(
        [v for row in values for v in row] if values and isinstance(values[0], list) else values,
        (len(values), len(values[0])) if values and isinstance(values[0], list) else None, dtype, device)
    torch.zeros = lambda shape, dtype=None, device='cpu': T([0] * _numel(shape), shape, dtype, device)
    torch.ones = lambda shape, dtype=None, device='cpu': T([1] * _numel(shape), shape, dtype, device)
    torch.arange = lambda lo, hi, dtype=None, device='cpu': T(list(range(lo, hi)), None, dtype, device)
    return torch


class FakeStates:
    def __init__(self, calls):
        self.req_id_to_index = {}
        self.free = [3, 2]
        self.calls = calls
        self.num_computed_tokens_np = np.zeros(8, dtype=np.int32)
        self.num_computed_prefill_tokens = np.zeros(8, dtype=np.int32)
        self.num_computed_tokens = types.SimpleNamespace(
            gpu='num-computed-gpu', stage_write_elem=lambda i, v: self.calls.append(('stage-computed', i, v)))

    def add_request(self, req_id, prompt_len, all_token_ids, num_computed_tokens, max_tokens):
        index = self.free.pop()
        self.req_id_to_index[req_id] = index
        self.calls.append(('add', req_id, prompt_len, len(all_token_ids), num_computed_tokens))

    def apply_staged_writes(self):
        self.calls.append(('apply-states',))


class FakeBlockTables:
    def __init__(self, calls):
        self.calls = calls
        self.num_kv_cache_groups = 3
        self.blocks_per_kv_block = [1, 2, 1]
        self.kernel_block_sizes = [4, 2, 4]
        self._slot_mapping_enabled = [True, True, False]
        self.block_tables = [types.SimpleNamespace(gpu=types.SimpleNamespace(shape=(8, 6))) for _ in range(3)]

    def append_block_ids(self, index, block_ids, overwrite):
        self.calls.append(('append', index, tuple(tuple(r) for r in block_ids), overwrite))

    def apply_staged_writes(self):
        self.calls.append(('apply-tables',))


class FakeCtx:
    def __init__(self):
        self.calls = []
        self.tables = None
        self.is_initialized = False

    def initialize_from_forward_context(self, config, fc, funcs, tables):
        self.tables = tables
        self.is_initialized = True

    def run_fused_precopy(self, n, state_idx, src_col, src_off, idx):
        self.calls.append(('precopy', n))

    def compute_aligned_state_indices(self, seq_lens, n):
        self.calls.append(('aligned', n))
        return {0: 'idx-g0', 1: 'idx-g1'}

    def run_fused_postprocess_align(self, n, accepted, state_idx, new_computed, idx):
        self.calls.append(('postprocess', n))


class FakeMambaContext:
    instances = []

    @classmethod
    def create(cls, max_num_reqs, kv_cache_config, copy_funcs, device, make_buffer):
        ctx = FakeCtx()
        ctx.max_num_reqs = max_num_reqs
        cls.instances.append(ctx)
        return ctx


def fake_runner(torch, calls):
    builder = types.SimpleNamespace(mamba_aligned_state_indices=None)
    group = types.SimpleNamespace(get_metadata_builder=lambda i: builder)
    state = types.SimpleNamespace(
        _align_mode=True, _mamba_state_copy_funcs={'gdn': 1},
        _get_mamba_group_info=lambda cfg: ([0, 1], types.SimpleNamespace(block_size=4)),
        _mamba_state_idx_gpu=T([0] * 8), _mamba_src_col_gpu='src-col', _mamba_src_off_gpu='src-off',
        num_accepted_tokens_gpu=T([1] * 8),
        add_request=lambda index, data: calls.append(('model-add', index, data.num_computed_tokens)),
        apply_staged_writes=lambda: calls.append(('apply-model',)))
    runner = types.SimpleNamespace(
        device='cuda:0', req_states=FakeStates(calls), block_tables=FakeBlockTables(calls), model_state=state,
        attn_groups=[[group], [group], []], kv_cache_config=types.SimpleNamespace(num_blocks=64),
        vllm_config=types.SimpleNamespace(compilation_config=types.SimpleNamespace(static_forward_context={})),
        kv_caches='kv-caches', _qsa_host_views=['view0'],
        kv_block_zeroer=types.SimpleNamespace(zero_block_ids=lambda ids: calls.append(('zero', tuple(ids)))),
        _qsa_new_uid=lambda index: calls.append(('uid', index)),
        _remove_request=lambda req_id: calls.append(('remove', req_id)))
    runner.builder = builder
    qsa = types.SimpleNamespace(_qsa_kv_offload=True, layer_name='layer', _qsa_host_kv='pinned-source')
    runner.model = types.SimpleNamespace(modules=lambda: (qsa,))
    return runner


def fake_ports(calls):
    @contextlib.contextmanager
    def override(ctx):
        calls.append(('override-enter', ctx))
        yield ctx
        calls.append(('override-exit',))

    return Ports(
        InputBatch=lambda **kw: types.SimpleNamespace(**kw),
        BatchDescriptor=lambda num_tokens, num_reqs, uniform: ('desc', num_tokens, num_reqs),
        MambaHybridAttnMetadata=lambda **kw: ('mamba', kw['is_prefilling'].values),
        build_attn_metadata=lambda **kw: calls.append(('build', kw['num_tokens'], kw['max_seq_len'])) or {'layer': types.SimpleNamespace(query_start_loc=kw['query_start_loc_gpu'], seq_lens=kw['seq_lens'], num_actual_tokens=kw['num_tokens'])},
        build_slot_mappings_by_layer=lambda m, cfg: {'layer': m},
        create_forward_context=lambda md, cfg, **kw: ('fc', md, kw['batch_descriptor']),
        override_forward_context=override, cudagraph_none='NONE',
        align_preprocess=lambda *a, **kw: calls.append(('align-pre', a[7], a[8])),
        mamba_context=FakeMambaContext,
        copy_blocks_inplace=lambda caches, n, copies: calls.append(('cow', caches, len(copies))),
        block_copy=lambda src, dst: (src, dst),
        new_request=lambda **kw: types.SimpleNamespace(**kw), pad_slot_id=-1,
        make_buffer=lambda device: ('make-buffer', device),
        mamba_groups=lambda cfg: {type('Spec', (), {'mamba_type': 'gdn',
                                                   'shapes': ((2,), (2,))}): [0, 1]},
        validate_mamba_copy_funcs=lambda groups, funcs: None)


def admission(**overrides):
    txn = {name: overrides.pop(name) for name in
           ('engine_epoch', 'request_uid', 'request_generation', 'chunk_start', 'chunk_end',
            'txn_seq') if name in overrides}
    start, end = txn.get('chunk_start', 0), txn.get('chunk_end', 8)
    txn.setdefault('engine_epoch', 1)
    txn.setdefault('request_uid', 'req-a')
    txn.setdefault('request_generation', 0)
    txn.setdefault('chunk_start', start)
    txn.setdefault('chunk_end', end)
    txn.setdefault('txn_seq', 1)
    base = dict(transaction=TransactionId(**txn), prompt_end=12,
                input_ids=tuple(range(start, end)), ngram_context=(1, 2),
                block_tables=((5, 6, 7), (9, 10), ()), new_block_ids_to_zero=(),
                cow_pairs=(), state_epoch=1)
    base.update(overrides)
    return ChunkAdmission(**base)


# --- tests ------------------------------------------------------------------

class AdmissionTests(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(admission().num_tokens, 8)
        with self.assertRaises(LaneChunkError):
            admission(chunk_end=12)  # must leave the final token
        with self.assertRaises(LaneChunkError):
            admission(input_ids=(1, 2))
        with self.assertRaises(LaneChunkError):
            admission(cow_pairs=((1,),))
        with self.assertRaises(LaneChunkError):
            admission(block_tables=())
        with self.assertRaises(LaneChunkError):
            Ports(InputBatch=None)

    def test_identity_is_derived_from_the_canonical_transaction(self):
        adm = admission()
        self.assertIs(type(adm.transaction), TransactionId)
        self.assertEqual((adm.request_uid, adm.request_generation, adm.chunk_start,
                          adm.chunk_end, adm.num_tokens), ('req-a', 0, 0, 8, 8))
        second = admission(chunk_start=8, chunk_end=11, txn_seq=2)
        self.assertEqual((second.request_uid, second.chunk_start, second.chunk_end,
                          second.num_tokens), ('req-a', 8, 11, 3))
        with self.assertRaises(TypeError):
            ChunkAdmission(request_uid='req-a', prompt_end=12,
                           input_ids=tuple(range(8)), ngram_context=(),
                           block_tables=((5,),))  # no duplicated identity fields
        for bad in ('txn', object(), None):
            with self.assertRaises(LaneChunkError):
                admission(transaction=bad)

    def test_repr_redacts_request_tokens_and_block_tables(self):
        text = repr(admission())
        self.assertNotIn('req-a', text, 'request uid must not be logged')
        self.assertIn('TransactionId(', text)
        self.assertIn('input_ids=<8 ids>', text)
        self.assertIn('block_tables=<3 groups>', text)
        self.assertIn('cow_pairs=<0 pairs>', text)
        self.assertNotIn('block_tables=(', text)
        self.assertNotIn('input_ids=(', text)
        self.assertNotIn('5, 6, 7', text)


class TablesTests(unittest.TestCase):
    def setUp(self):
        self.torch = fake_torch()
        self.calls = []
        self.runner = fake_runner(self.torch, self.calls)
        self.tables = LaneTables(self.runner, torch_module=self.torch)

    def test_expansion_write_and_slots(self):
        self.assertEqual(self.tables.expanded(1, (9, 10)), [18, 19, 20, 21])
        views = self.tables.write(admission())
        self.assertEqual(len(views), 3)
        self.assertEqual(self.tables.rows[0].values[:4], [5, 6, 7, 0])
        self.assertEqual(self.tables.rows[1].values[:5], [18, 19, 20, 21, 0])
        slots = self.tables.slot_mappings(admission(), -1)
        self.assertEqual(slots.shape, (3, 8))
        # group 0: kbs 4, blocks 5,6 -> 20..23, 24..27
        self.assertEqual(slots.values[:8], [20, 21, 22, 23, 24, 25, 26, 27])
        # group 1: kbs 2, expanded blocks 18,19,20,21
        self.assertEqual(slots.values[8:16], [36, 37, 38, 39, 40, 41, 42, 43])
        # group 2: slot mapping disabled -> pad
        self.assertEqual(slots.values[16:], [-1] * 8)
        self.assertEqual(self.tables.bytes, 3 * 6 * 4)
        with self.assertRaises(LaneChunkError):
            self.tables.slot_mappings(admission(block_tables=((5,), (9, 10), ())), -1)
        with self.assertRaises(LaneChunkError):
            self.tables.write(admission(block_tables=((5,), (9,))))


class HostPlanTests(unittest.TestCase):
    def fixture(self):
        from lane_chunk import attach_qsa_host_metadata
        md = types.SimpleNamespace(query_start_loc=T([0, 8], device='cuda:0'),
                                   seq_lens=T([8], device='cuda:0'), num_actual_tokens=8)
        return attach_qsa_host_metadata, md, T([0, 8]), T([8])

    def test_cpu_plan_is_copied_and_bound_to_the_same_metadata(self):
        bind, md, starts, lengths = self.fixture()
        host = bind({'layer': md, 'other': 'untouched'}, ('layer',), starts, lengths, admission())
        self.assertIs(md._qsa_host_meta, host)
        self.assertEqual(host, ([0, 8], [8]))
        starts.values[1] = 3
        lengths.values[0] = 3
        self.assertEqual(host, ([0, 8], [8]))

    def test_wrong_or_device_plan_is_refused_without_readback(self):
        bind, md, starts, lengths = self.fixture()
        for first, second in ((T([0, 7]), lengths), (starts, T([7])),
                              (T([0, 8], device='cuda:0'), lengths), (None, lengths)):
            with self.assertRaises(LaneChunkError):
                bind({'layer': md}, ('layer',), first, second, admission())

    def test_missing_stale_and_wrong_shape_metadata_refuse(self):
        bind, md, starts, lengths = self.fixture()
        with self.assertRaises(LaneChunkError):
            bind({}, ('layer',), starts, lengths, admission())
        md._qsa_host_meta = ([0, 8], [9])
        with self.assertRaises(LaneChunkError):
            bind({'layer': md}, ('layer',), starts, lengths, admission())
        del md._qsa_host_meta
        md.seq_lens = T([8, 8], device='cuda:0')
        with self.assertRaises(LaneChunkError):
            bind({'layer': md}, ('layer',), starts, lengths, admission())


class FactoryTests(unittest.TestCase):
    def setUp(self):
        FakeMambaContext.instances = []
        self.torch = fake_torch()
        self.calls = []
        self.runner = fake_runner(self.torch, self.calls)
        self.ports = fake_ports(self.calls)
        self.factory = LaneChunkFactory(self.runner, ports=self.ports, torch_module=self.torch)

    def test_align_context_bound_to_lane_tables(self):
        ctx = FakeMambaContext.instances[0]
        self.assertEqual(ctx.max_num_reqs, 1)
        self.assertEqual(len(ctx.tables), 2)
        self.assertIs(ctx.tables[0], self.factory.tables.rows[0])
        self.assertIs(ctx.tables[1], self.factory.tables.rows[1])
        self.assertTrue(self.factory.inventory()['align'])

    def test_first_chunk_registers_then_builds_in_order(self):
        chunk = self.factory.admit(admission())
        tags = [c[0] for c in self.calls]
        self.assertEqual(tags[:7], ['add', 'uid', 'model-add', 'append', 'apply-states', 'apply-model', 'apply-tables'])
        self.assertEqual(self.calls[0][1:], ('req-a', 12, 12, 0))
        self.assertEqual(self.calls[3][3], True, 'first chunk overwrites the rows')
        self.assertEqual(chunk.req_index, 2)
        self.assertIn('align-pre', tags)
        self.assertLess(tags.index('align-pre'), tags.index('build'), 'pre-copy before metadata')
        self.assertEqual(self.runner.builder.mamba_aligned_state_indices, 'idx-g1')
        self.assertEqual(FakeMambaContext.instances[0].calls, [('precopy', 1), ('aligned', 1)])
        self.assertIs(chunk.attn_metadata['layer']._qsa_host_meta, chunk.qsa_host_plan)
        self.assertEqual(chunk.qsa_host_plan, ([0, 8], [8]))
        self.assertEqual(chunk.forward_context[2], ('desc', 8, 1))
        self.assertEqual(chunk.batch.num_tokens, 8)
        self.assertEqual(chunk.batch.idx_mapping.values, [2])
        self.assertTrue(bool(chunk.batch.is_prefilling_np[0]))
        inputs = chunk.lane_inputs()
        self.assertEqual(sorted(inputs), ['input_ids', 'ngram_context', 'positions', 'query_start_loc'])
        self.assertEqual(inputs['positions'].values, list(range(0, 8)))
        with chunk.forward_scope() as fc:
            self.assertIs(fc, chunk.forward_context)
        self.assertEqual(self.calls[-2][0], 'override-enter')
        self.assertTrue(chunk.finish_state())
        self.assertEqual(FakeMambaContext.instances[0].calls[-1], ('postprocess', 1))
        self.assertIn(('stage-computed', 2, 8), self.calls)
        self.assertEqual(self.runner.model_state.num_accepted_tokens_gpu.values[2], 1)
        self.assertTrue(self.factory.inventory()['registered'])

    def test_later_chunk_updates_zeroes_and_copies(self):
        self.factory.admit(admission())
        self.calls.clear()
        second = admission(chunk_start=8, chunk_end=11, input_ids=(8, 9, 10),
                           block_tables=((5, 6, 7, 8), (9, 10, 11), ()), new_block_ids_to_zero=(8, 11),
                           cow_pairs=((1, 8),))
        chunk = self.factory.admit(second)
        tags = [c[0] for c in self.calls]
        self.assertEqual(tags[:6], ['stage-computed', 'append', 'apply-states', 'apply-tables', 'zero', 'cow'])
        self.assertEqual(self.calls[0], ('stage-computed', 2, 8))
        self.assertEqual(self.runner.req_states.num_computed_tokens_np[2], 8)
        self.assertEqual(self.calls[4], ('zero', (8, 11)))
        self.assertEqual(self.calls[5], ('cow', 'kv-caches', 1))
        self.assertEqual(self.calls[6], ('cow', ['view0'], 1), 'QSA host pages copied too')
        self.assertEqual(chunk.positions.values, [8, 9, 10])
        self.assertEqual(self.factory.chunks, 2)
        with self.assertRaises(LaneChunkError):
            self.factory.admit(admission(request_uid='other'))
        self.assertEqual(self.factory.release(), 'req-a')
        self.assertEqual(self.calls[-1], ('remove', 'req-a'))
        self.assertIsNone(self.factory.release())

    def test_duplicate_registration_refused(self):
        self.runner.req_states.req_id_to_index['req-a'] = 5
        with self.assertRaises(LaneChunkError):
            self.factory.admit(admission())
        with self.assertRaises(LaneChunkError):
            self.factory.admit('not an admission')


def pinned_function(path, name, *, class_name=None, namespace=None):
    """Execute the pinned CPU method body, with device constructors doubled."""
    tree = ast.parse(path.read_text())
    owner = tree if class_name is None else next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    function = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])
    env = {} if namespace is None else dict(namespace)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), env)
    return env[name]


class LazyAlignTests(unittest.TestCase):
    """The stock startup state is lazy; lane construction must not require a batch."""

    def setUp(self):
        self.torch = fake_torch()
        self.runner = fake_runner(self.torch, [])
        self.state = self.runner.model_state
        self.state._mamba_state_copy_funcs = None
        self.state._mamba_ctx = None
        self.state.max_num_reqs = 8
        self.state.device = self.runner.device
        self.state.vllm_config = self.runner.vllm_config
        self.ports = fake_ports([])
        self.functions = {'gdn': (lambda: None, lambda: None)}
        self.getter_types = []
        def getter(types):
            self.getter_types.append(types)
            return self.functions
        self.state.model = types.SimpleNamespace(get_mamba_state_copy_funcs=getter)
        self.ports.validate_mamba_copy_funcs = pinned_function(
            ROOT/'reference/vllm-src/v1/worker/mamba_utils.py',
            'validate_mamba_state_copy_funcs')
        self.groups = self.ports.mamba_groups(self.runner.kv_cache_config)
        self.ports.mamba_groups = lambda cfg: self.groups
        self.tables = LaneTables(self.runner, torch_module=self.torch)
        FakeMambaContext.instances = []

    def construct(self):
        return lane_chunk.LaneAlign(self.runner, self.ports, self.tables,
                                    torch_module=self.torch)

    def test_none_cache_uses_real_model_descriptors_and_stock_validator(self):
        align = self.construct()
        self.assertTrue(align.enabled)
        self.assertTrue(align.ctx.is_initialized)
        self.assertEqual(self.getter_types, [{'gdn'}])
        self.assertIsNone(self.state._mamba_state_copy_funcs)
        self.assertIsNone(self.state._mamba_ctx)
        self.assertEqual(align.ctx.max_num_reqs, 1)
        self.assertEqual(align.ctx.tables, [self.tables.rows[0], self.tables.rows[1]])

    def test_cached_ordinary_descriptors_are_reused_without_model_query(self):
        self.state._mamba_state_copy_funcs = self.functions
        ordinary_ctx = object()
        self.state._mamba_ctx = ordinary_ctx
        self.construct()
        self.assertEqual(self.getter_types, [])
        self.assertIs(self.state._mamba_state_copy_funcs, self.functions)
        self.assertIs(self.state._mamba_ctx, ordinary_ctx)

    def test_stock_lazy_initializer_later_binds_ordinary_tables_independently(self):
        lane = self.construct()
        method = pinned_function(
            ROOT/'reference/vllm-src/v1/worker/gpu/model_states/mamba_hybrid.py',
            '_ensure_align_ctx', class_name='MambaHybridModelState', namespace={
                'get_mamba_groups': self.ports.mamba_groups,
                'validate_mamba_state_copy_funcs': self.ports.validate_mamba_copy_funcs,
                'MambaSpecDecodeGPUContext': FakeMambaContext,
                'CpuGpuBuffer': lambda *a, **kw: None})
        ordinary_tables = tuple(object() for _ in range(3))
        ordinary = method(self.state, self.runner.kv_cache_config, [0, 1], ordinary_tables)
        self.assertIsNot(ordinary, lane.ctx)
        self.assertEqual(ordinary.max_num_reqs, 8)
        self.assertEqual(ordinary.tables, list(ordinary_tables[:2]))
        self.assertIs(method(self.state, self.runner.kv_cache_config, [0, 1],
                             ordinary_tables), ordinary)
        self.assertEqual(len(self.getter_types), 2)
        self.assertIs(self.state._mamba_state_copy_funcs, self.functions)
        self.assertEqual(lane.ctx.tables, [self.tables.rows[0], self.tables.rows[1]])

    def test_missing_copy_family_refuses_before_context_allocation(self):
        self.functions = {}
        with self.assertRaises(AssertionError):
            self.construct()
        self.assertEqual(FakeMambaContext.instances, [])
        self.assertIsNone(self.state._mamba_state_copy_funcs)

    def test_invalid_copy_prefix_refuses_before_context_allocation(self):
        for functions in ({'gdn': ()}, {'gdn': (1, 2, 3)}):
            self.functions = functions
            with self.assertRaises(AssertionError):
                self.construct()
        self.assertEqual(FakeMambaContext.instances, [])

    def test_align_off_remains_inert(self):
        self.state._align_mode = False
        align = self.construct()
        self.assertFalse(align.enabled)
        self.assertEqual(self.getter_types, [])
        self.assertEqual(FakeMambaContext.instances, [])


if __name__ == '__main__':
    unittest.main()
