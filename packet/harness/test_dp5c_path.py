"""Decode-phase dp5c: the native prefill path's Python at 2048 rows, with 1024 as the control, CPU tensors, kernels stubbed.

Real pinned code wherever it can run without a GPU, so its shape assertions and slices run at the size: QSAMetadataBuilder and its
CPU metadata kernel, QSAIndexer.forward, qsa_pre_indexer, qsa_select_paged_prefill and expand_qsa_block_indices. A stub replaces
only a launch and records what it was handed. test_the_box_crash is dp5b's crash of 4 Nov 22:27 CET, on the unscaled clone.
What the kernels do with the rows is not exercised here; that needs the box.
"""
from contextlib import nullcontext
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'patches/s5_engine'), str(ROOT / 'harness')]
from vllm.models.qwen4_exp.nvidia import model as _model  # noqa: F401  first: qsa <-> model import cycle
from vllm.models.qwen4_exp.common import qsa_cache as qc
from vllm.models.qwen4_exp.common.qsa_cache import QSAMetadataBuilder
from vllm.models.qwen4_exp.nvidia import indexer_qsa as ix
from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as qi
from vllm.models.qwen4_exp.nvidia.ops import qsa_pre_indexer as qp
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import CircularBufferSpec, MLAAttentionSpec
from vllm.v1.worker.utils import AttentionGroup
import lane_chunk as lc
import model_hooks as mh
import qsa_driver

NS = types.SimpleNamespace
STOCK_ROWS, REQUESTS, RATIO = 1024, 10, 4
Q_HEADS, HEAD_DIM, TOPK = 16, 128, 2048
RAW = 'model.layers.3.self_attn.indexer.raw_key_cache'
COMPRESSED = 'model.layers.3.self_attn.indexer.compressed_key_cache'
SIZES = (1024, 1352, 1536, 2048)


class Kernel:
    """A launch: records the grid and what it was handed, never runs."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
        return run


def stock_groups():
    cfg = NS(scheduler_config=NS(max_num_batched_tokens=STOCK_ROWS, max_num_seqs=REQUESTS), speculative_config=None,
             parallel_config=NS(decode_context_parallel_size=1), model_config=NS(is_mm_prefix_lm=False))
    raw_spec = CircularBufferSpec(block_size=4, num_kv_heads=1, head_size=HEAD_DIM, dtype=torch.bfloat16)
    comp_spec = MLAAttentionSpec(block_size=64, num_kv_heads=1, head_size=HEAD_DIM, dtype=torch.bfloat16,
                                 tokens_per_state=RATIO)
    device = torch.device('cpu')
    raw = QSAMetadataBuilder(raw_spec, [RAW], cfg, device)
    comp = QSAMetadataBuilder(comp_spec, [COMPRESSED], cfg, device)
    return [[AttentionGroup(backend=object, layer_names=[RAW], kv_cache_spec=raw_spec, kv_cache_group_id=0,
                            metadata_builders=[raw])],
            [AttentionGroup(backend=object, layer_names=[COMPRESSED], kv_cache_spec=comp_spec, kv_cache_group_id=1,
                            metadata_builders=[comp])]]


def common(rows, chunk_start, pages=128):
    """A native chunk's real common metadata: one request, rows from chunk_start, as LaneChunk passes it."""
    starts = torch.tensor([0, rows], dtype=torch.int32)
    owner = CommonAttentionMetadata(starts, starts, torch.tensor([chunk_start + rows], dtype=torch.int32), 1, rows, rows,
                                    chunk_start + rows, torch.arange(pages, dtype=torch.int32).reshape(1, pages),
                                    torch.arange(rows, dtype=torch.int64))
    owner._token_to_req_indices_cache = torch.zeros(rows, dtype=torch.int32)
    return owner


def build(groups, rows, chunk_start=0, device_path=True):
    """Both side caches' metadata out of the real builders, keyed as the forward context keys them.

    device_path: the real build_qsa_metadata_triton, what a GPU runs, launch stubbed: past a buffer's end its slices are short and
    nothing raises, as on the box. Otherwise the real CPU kernel, which fills values but refuses short buffers itself."""
    chosen = qc.build_qsa_metadata_triton if device_path else qc._build_qsa_metadata_torch
    with mock.patch.object(qc, 'build_qsa_metadata', chosen), mock.patch.object(qc, '_build_qsa_metadata_kernel', Kernel()), \
            mock.patch.object(qc, '_metadata_launch_pdl', lambda: 0, create=True):
        metadata = {group.layer_names[0]: group.metadata_builders[0].build(0, common(rows, chunk_start))
                    for row in groups for group in row}
    if device_path:  # the launch didn't run; forward compares the two caches' positions on CPU
        for md in metadata.values():
            md.logical_positions.zero_()
    return metadata


def indexer():
    """The real QSAIndexer with its __init__'s weights and config replaced; forward and everything below it are the pinned code."""
    module = ix.QSAIndexer.__new__(ix.QSAIndexer)
    torch.nn.Module.__init__(module)
    module.index_n_heads, module.index_kv_heads, module.index_head_dim = Q_HEADS, 1, HEAD_DIM
    module.token_topk, module.compress_ratio = TOPK, RATIO
    module.use_fused_pre_indexer, module.skip_topk = True, False
    module.indexer_dtype = torch.bfloat16
    module.rotary_emb = NS(cos_sin_cache=torch.zeros((8, HEAD_DIM // 2)), mrope_section=None)
    module.q_layernorm = NS(weight=torch.ones(HEAD_DIM), variance_epsilon=1e-6)
    module.k_layernorm = NS(weight=torch.ones(HEAD_DIM), variance_epsilon=1e-6)
    module.raw_key_cache = NS(prefix=RAW, kv_cache=torch.zeros((8, 4, 1, HEAD_DIM), dtype=torch.bfloat16),
                              rope_position_cache=None, rope_position_offset=None)
    module.compressed_key_cache = NS(prefix=COMPRESSED, kv_cache=torch.zeros((8, 64, 1, HEAD_DIM), dtype=torch.bfloat16))
    return module


def select(metadata, rows, chunk_start=0):
    """QSAIndexer.forward on this chunk's rows; returns the output and the launches it made."""
    module = indexer()
    qk = torch.zeros((rows, (Q_HEADS + 1) * HEAD_DIM), dtype=torch.bfloat16)
    positions = torch.arange(chunk_start, chunk_start + rows, dtype=torch.int64)
    out = torch.empty((rows, module.packed_output_width), dtype=torch.int32)
    launches = dict(pre=Kernel(), prefill=Kernel(), expand=Kernel(), topk=[])

    def topk(logits, visible_blocks, token_topk, compress_ratio, block_indices, workspace):
        launches['topk'].append((tuple(logits.shape), tuple(visible_blocks.shape), tuple(block_indices.shape)))
    with mock.patch.object(ix, 'get_forward_context', lambda: NS(attn_metadata=metadata)), \
            mock.patch.object(qp, '_qsa_pre_indexer_kernel', launches['pre']), \
            mock.patch.object(qi, '_qsa_mqa_paged_prefill_kernel', launches['prefill']), \
            mock.patch.object(qi, '_expand_qsa_indices_kernel', launches['expand']), \
            mock.patch.object(qi, '_topk', topk):
        return module(qk, positions, out), launches


class Builders(unittest.TestCase):
    def test_every_row_field_holds_the_chunks_rows(self):
        for rows in SIZES:
            capacity = None if rows <= 1024 else (STOCK_ROWS, rows)
            with self.subTest(rows=rows):
                private, _, _, _ = lc.private_attention_groups(stock_groups(), torch_module=torch, row_capacity=capacity)
                metadata = build(private, rows, device_path=False)
                lc.check_metadata_rows(metadata, rows)
                self.assertEqual(metadata[COMPRESSED].logical_positions[-1].item(), rows - 1)
                lc.check_metadata_rows(build(private, rows), rows)
                for md in metadata.values():
                    for field in lc._QSA_ROW_METADATA:
                        self.assertEqual(tuple(getattr(md, field).shape), (rows,), field)

    def test_k_work_rows_reach_the_chunk(self):
        for rows in SIZES:
            capacity = None if rows <= 1024 else (STOCK_ROWS, rows)
            with self.subTest(rows=rows):
                private, _, _, _ = lc.private_attention_groups(stock_groups(), torch_module=torch, row_capacity=capacity)
                comp = private[1][0].metadata_builders[0]
                self.assertEqual(comp.k_work_metadata_buffer.shape, ((rows + (RATIO - 1) * REQUESTS) // RATIO, 2))
                self.assertEqual(private[0][0].metadata_builders[0].k_work_metadata_buffer.numel(), 0)
                self.assertEqual(build(private, rows)[COMPRESSED].k_work_metadata.shape,
                                 ((rows + (RATIO - 1)) // RATIO, 2))

    def test_stock_stays_at_the_budget_and_shares_nothing(self):
        stock = stock_groups()
        private, _, _, _ = lc.private_attention_groups(stock, torch_module=torch, row_capacity=(STOCK_ROWS, 2048))
        for s_row, p_row in zip(stock, private):
            s, p = s_row[0].metadata_builders[0], p_row[0].metadata_builders[0]
            for name in lc._QSA_ROW_BUFFERS:
                self.assertEqual(tuple(getattr(s, name).shape), (STOCK_ROWS,))
                self.assertEqual(tuple(getattr(p, name).shape), (2048,))
                self.assertEqual(getattr(p, name).dtype, getattr(s, name).dtype)
                self.assertNotEqual(getattr(p, name).data_ptr(), getattr(s, name).data_ptr())

    def test_ledger_counts_the_scaled_bytes(self):
        private, nbytes, count, inventory = lc.private_attention_groups(stock_groups(), torch_module=torch,
                                                                        row_capacity=(STOCK_ROWS, 2048))
        expected = sum(int(t.numel()) * int(t.element_size()) for row in private for t in vars(row[0].metadata_builders[0]).values()
                       if isinstance(t, torch.Tensor))
        self.assertEqual(nbytes, expected)
        self.assertEqual(sum(f['tensor_bytes'] for b in inventory for f in b['fields']), nbytes)
        self.assertEqual(count, 10)

    def test_byte_cap_holds_on_scaled_storage(self):
        with self.assertRaisesRegex(lc.LaneChunkError, 'byte per-rank cap'):
            lc.private_attention_groups(stock_groups(), torch_module=torch, max_bytes=1000, row_capacity=(STOCK_ROWS, 2048))

    def test_an_unknown_row_sized_tensor_refuses_at_startup(self):
        groups = stock_groups()
        groups[0][0].metadata_builders[0].surprise_rows = torch.empty(STOCK_ROWS, dtype=torch.int32)
        with self.assertRaisesRegex(lc.LaneChunkError, 'not scaled: surprise_rows'):
            lc.private_attention_groups(groups, torch_module=torch, row_capacity=(STOCK_ROWS, 2048))

    def test_1024_control_is_a_plain_clone(self):
        """row_capacity None, what the factory passes at or below 1024: same shapes, same bytes, same inventory as stock's fields."""
        stock = stock_groups()
        private, nbytes, count, _ = lc.private_attention_groups(stock, torch_module=torch)
        for s_row, p_row in zip(stock, private):
            s, p = s_row[0].metadata_builders[0], p_row[0].metadata_builders[0]
            for name in lc._QSA_ROW_BUFFERS + ('k_work_metadata_buffer',):
                self.assertEqual(getattr(p, name).shape, getattr(s, name).shape)
        self.assertEqual(nbytes, 2 * (4096 + 8192 + 8192 + 4096) + ((1024 + 3 * REQUESTS) // RATIO) * 8)
        self.assertEqual(count, 10)

    def test_factory_at_1024_never_reads_the_budget(self):
        seen = []

        def groups(attn_groups, *, torch_module, row_capacity=None):
            seen.append(row_capacity)
            return [], 0, 0, []
        runner = NS(block_tables=NS(num_kv_cache_groups=0, blocks_per_kv_block=(), kernel_block_sizes=(), _slot_mapping_enabled=(),
                                    block_tables=()), attn_groups=[], device='cpu')
        for chunk, want in ((1024, None), (2048, (STOCK_ROWS, 2048))):
            if want:
                runner.scheduler_config = NS(max_num_batched_tokens=STOCK_ROWS)
            with mock.patch.object(lc, 'private_attention_groups', groups), mock.patch.object(lc, 'LaneAlign'), \
                    mock.patch.object(lc, 'LaneRequest'):
                lc.LaneChunkFactory(runner, ports=object(), torch_module=torch, chunk_tokens=chunk)
            self.assertEqual(seen.pop(), want)


class RowCheck(unittest.TestCase):
    def test_a_short_slice_is_named_where_it_happens(self):
        private, _, _, _ = lc.private_attention_groups(stock_groups(), torch_module=torch)
        with self.assertRaisesRegex(lc.LaneChunkError, r'token_to_req holds \(1024,\), not 2048 rows'):
            lc.check_metadata_rows(build(private, 2048), 2048)

    def test_metadata_without_row_fields_passes(self):
        lc.check_metadata_rows({'layers.1.ple': NS(), 'layers.0.linear_attn': NS(visible_blocks=None)}, 2048)


class Indexer(unittest.TestCase):
    def test_the_box_crash(self):
        """The unscaled clone at 2048 stops where the box stopped: qsa_indexer.py:393."""
        private, _, _, _ = lc.private_attention_groups(stock_groups(), torch_module=torch)
        try:
            select(build(private, 2048), 2048)
        except AssertionError as caught:
            tb, frames = caught.__traceback__, []
            while tb:
                frames.append(tb.tb_frame.f_code.co_name)
                tb = tb.tb_next
        else:
            self.fail('the unscaled clone passed at 2048')
        self.assertEqual(frames[-1], '_prefill_logits')

    def test_forward_at_each_size(self):
        for rows in SIZES:
            for start in (0, rows):  # the first chunk and a later one
                capacity = None if rows <= 1024 else (STOCK_ROWS, rows)
                with self.subTest(rows=rows, chunk_start=start):
                    private, _, _, _ = lc.private_attention_groups(stock_groups(), torch_module=torch, row_capacity=capacity)
                    out, launches = select(build(private, rows, start), rows, start)
                    self.assertEqual(tuple(out.shape), (rows, TOPK + RATIO))
                    self.assertEqual(len(launches['pre'].calls), 1)
                    self.assertTrue(launches['prefill'].calls)
                    self.assertEqual(len(launches['expand'].calls), 1)
                    self.assertEqual(sum(shape[0] for shape, _, _ in launches['topk']), rows)
                    for _, visible, indices in launches['topk']:
                        self.assertEqual(visible[0], indices[0])


# --- The two stages at each size: real LaneForward, QsaLayerDriver, composed decoder layers, attention pre/post, staged generators, PLE, PP channel.
HIDDEN, HC_COUNT, HEADS, HEAD = 16, 2, 12, 64
PAGE = 64


class HC:
    """Row-checking stand-in for the hyper-connection modules; real in the next tier if it can be."""

    def __init__(self, rows):
        self.rows = rows

    def _check(self, *tensors):
        for t in tensors:
            if t.shape[0] != self.rows:
                raise AssertionError(f'HC got {t.shape[0]} rows, not {self.rows}')

    def mix(self, h):
        self._check(h)
        return h, h[:, :HIDDEN], h[:, :HIDDEN]

    def combine(self, h, block, injection):
        self._check(h, block, injection)
        return h

    def combine_and_mix(self, h, block, injection):
        self._check(h, block, injection)
        return h, h[:, :HIDDEN], h[:, :HIDDEN]


class Operation:
    """The transaction operation's surface: what the real generators and driver call; records rows, never touches a device."""

    def __init__(self, sparse=False):
        self.sparse = sparse
        self.continuation = None
        self.memory_receipts = False
        self.run = NS(schedule_boundary=None)
        self.buckets = 0
        self.fills = []
        self.layers = []

    def observe_chunk_shape(self, rows, min_rows, route):
        self.shape = (rows, route)

    def begin_layer(self, name, source):
        self.layers.append(name)
        self.run = NS(schedule_boundary=None)

    def boundary_evidence(self, tensor, *, selection=False):
        return None

    def producer_submitted(self, name, *owners):
        pass

    def has_sparse_source(self, name):
        return self.sparse

    def attach_continuation(self, generator):
        self.continuation = generator

    def advance(self):
        return next(self.continuation)

    advance_fused = advance

    def authorize_schedule(self, grant):
        self.granted = grant

    def end_layer(self, name):
        pass

    def _current_run(self):
        return self.run

    def schedule_boundary(self, selection=None):
        self.run.schedule_boundary = NS(qsa_schedule_boundary=True, sparse_selection_pending=False, sparse_copy_pending=False)
        return self.run.schedule_boundary

    def bind_plan(self, name, row, pages, seq_len, arena):
        self.plan = (name, pages, seq_len)

    def fill_bucket(self, staged, lo, hi, row):
        self.fills.append((lo, hi))

    def retain_attention(self, q, indices, rows, acc, max_state, norm_state, out):
        for t in (q, indices, acc, max_state, norm_state, out):
            if t.shape[0] != rows.numel():
                raise AssertionError(f'attention bucket holds {t.shape[0]} rows, not {rows.numel()}')

    def after_attention_bucket(self, *args):
        self.buckets += 1
        return NS(qsa_schedule_boundary=False, sparse_selection_pending=False, sparse_copy_pending=False)

    # sparse route
    def bind_sparse_context(self, name, arena, row, min_pages=0):
        pass

    def bind_row1_uva(self, name, kv_cache):
        pass

    def bind_row1_reference(self, name, function):
        pass

    def sparse_selection_eligible(self, pages, arena_pages):
        return True

    def sparse_union_scratch(self, capacity, logical_indices, rows):
        groups = (capacity + 1023) // 1024
        ints = lambda n: torch.zeros((n,), dtype=torch.int32)  # noqa: E731  sizes as alloc_union_scratch
        return NS(flags=ints(capacity), ranks=ints(capacity), tokens=ints(capacity), totals=ints(groups), offsets=ints(groups),
                  info=ints(2), remapped=logical_indices.clone())

    def stage_sparse_selection(self, tokens, info):
        self.selection = (tokens, info)
        return 'selection'

    def sparse_selection_ready(self, selection):
        return True

    def sparse_selection_pending(self, selection):
        raise AssertionError('ready at once here')

    def sparse_selection_count(self, selection):
        return self.selection[0].numel() // 2

    def declare_sparse_route(self, route):
        self.route = route

    def fill_sparse_bucket(self, *args, **kwargs):
        self.fills.append('sparse')

    def row1_output(self, *args, **kwargs):
        pass


def attention(rows, name, oq):
    """The real composed attention class's pre/post on stubbed projections, over the real indexer."""
    from vllm.models.qwen4_exp.nvidia import qsa as cq
    attn = cq.Qwen4ExpQSAAttention.__new__(cq.Qwen4ExpQSAAttention)
    torch.nn.Module.__init__(attn)
    attn.layer_name, attn.num_heads, attn.num_kv_heads, attn.head_dim = name, HEADS, 1, HEAD
    attn.qkv_proj = lambda h: (h, None)
    attn._project_qkv_gate = lambda qkv, positions: (torch.zeros((qkv.shape[0], HEADS * HEAD), dtype=torch.bfloat16),
                                                     torch.zeros((qkv.shape[0], HEAD), dtype=torch.bfloat16),
                                                     torch.zeros((qkv.shape[0], HEAD), dtype=torch.bfloat16), None)
    attn.o_proj = lambda flat: (flat[:, :HIDDEN], None)
    attn.indexer = indexer()
    attn.indexer.index_qk_proj = lambda h: (torch.zeros((h.shape[0], (Q_HEADS + 1) * HEAD_DIM), dtype=torch.bfloat16), None)
    attn.topk_indices_buffer = torch.full((rows, attn.indexer.packed_output_width), -1, dtype=torch.int32)
    attn.kv_cache = torch.zeros((128, 1, PAGE, 2 * HEAD), dtype=torch.bfloat16)
    attn._qsa_host_kv = 'host'
    attn._qsa_kv_offload, attn._qsa_stage_min_rows, attn._qsa_stage_arena_bytes = True, 1, 1
    attn._max_decode_query_len = 1
    attn.impl = NS(do_kv_cache_update=lambda *a: None)
    return attn


def decoder(kind, rows, attn=None, ple=None):
    layer = _model.Qwen4ExpDecoderLayer.__new__(_model.Qwen4ExpDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.layer_type = kind
    layer.ple = ple
    layer.attn_hyper_connection = layer.mlp_hyper_connection = HC(rows)

    def per_row(hidden_states=None, **kwargs):
        if hidden_states.shape[0] != rows:
            raise AssertionError(f'mixer got {hidden_states.shape[0]} rows, not {rows}')
        return hidden_states
    layer.linear_attn = per_row
    layer.mlp = lambda block: per_row(block)
    if attn is not None:
        layer.self_attn = attn
    return layer


def stage(kinds, rows, first):
    model = _model.Qwen4ExpModel.__new__(_model.Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model.layers = kinds
    model.start_layer, model.end_layer = 0, len(kinds)
    model.config = NS(hc_count=HC_COUNT)
    model.embed_input_ids = lambda ids: torch.zeros((ids.shape[0], HIDDEN), dtype=torch.bfloat16)
    model._mtp_hidden_buffer = None
    return model


class Stages(unittest.TestCase):
    def run_stage(self, rows, start, *, first, sparse, intermediate=None, oq=None, cq=None):
        """Dense goes the fused route, the box's: one command per QSA layer. Sparse goes bucket by bucket."""
        names = ('qsa-a', 'qsa-b')
        attn = attention(rows, names[0], oq)
        layers = [decoder('linear_attention', rows), decoder('full_attention', rows, attn=attn), decoder('linear_attention', rows)]
        model = stage(layers, rows, first)
        owner = common(rows, start)
        private, _, _, _ = lc.private_attention_groups(stock_groups(), torch_module=torch,
                                                       row_capacity=None if rows <= 1024 else (STOCK_ROWS, rows))
        metadata = build(private, rows, start)
        metadata[names[0]] = NS(num_actual_tokens=rows, slot_mapping=torch.zeros(rows, dtype=torch.int64), max_query_len=rows,
                                block_table=owner.block_table_tensor, query_start_loc=owner.query_start_loc, seq_lens=owner.seq_lens)
        lc.check_metadata_rows(metadata, rows)
        lc.attach_qsa_host_metadata({names[0]: metadata[names[0]]}, (names[0],), owner.query_start_loc_cpu, owner.seq_lens,
                                    NS(num_tokens=rows, chunk_end=start + rows))
        operation = Operation(sparse=sparse)
        context = lambda: NS(attn_metadata=metadata)  # noqa: E731
        driver = qsa_driver.QsaLayerDriver(operation, staged_operation=cq._flash_s5_staged_operation,
                                           sparse_staged_operation=cq._flash_s5_sparse_staged_operation, forward_context=context)
        inputs = dict(positions=torch.arange(start, start + rows, dtype=torch.int64), input_ids=torch.zeros(rows, dtype=torch.int32),
                      query_start_loc=owner.query_start_loc, ngram_context=torch.zeros((1, 8), dtype=torch.int64))
        lane = mh.LaneForward(model, inputs, stream=None, attention_driver=driver)
        lane.begin(intermediate_tensors=intermediate)
        self.assertEqual(lane.hidden_states.shape[0], rows)
        for index in range(len(layers)):
            if not sparse and driver.owns(model, index):
                driver.arm_fused(index, NS(query=NS(layer=index), fused=True))
            for _ in range(1000):
                if lane.layer(index).kind == 'layer':
                    break
            else:
                self.fail('layer never finished')
        lane.finish()
        self.assertGreater(operation.buckets, 0)
        self.assertEqual(operation.shape[0], rows)
        return lane, operation

    def test_s0_then_pp_then_s1_at_each_size(self):
        from vllm.models.qwen4_exp.nvidia import qsa as cq
        from vllm.models.qwen4_exp.nvidia.ops import qsa as oq
        launches = {name: Kernel() for name in ('_qsa_sparse_staged_gqa_kernel', '_s6_mark', '_s6_scan', '_s6_offsets', '_s6_pack',
                                                '_s6_remap')}
        patches = [mock.patch.object(oq, name, kernel) for name, kernel in launches.items()]
        patches.append(mock.patch.object(oq, 'qsa_get_staging_arena',
                                         lambda kv, nbytes: torch.zeros((23, 1, PAGE, 2 * HEAD), dtype=torch.bfloat16)))
        patches.append(mock.patch('vllm.distributed.parallel_state.get_tp_group', lambda: NS(world_size=1)))
        # Triton is disabled without a GPU: its one plain helper this path calls.
        patches.append(mock.patch.object(oq.triton, 'next_power_of_2', lambda n: 1 << (int(n) - 1).bit_length(), create=True))
        patches += [mock.patch.object(qp, '_qsa_pre_indexer_kernel', Kernel()), mock.patch.object(qi, '_qsa_mqa_paged_prefill_kernel', Kernel()),
                    mock.patch.object(qi, '_expand_qsa_indices_kernel', Kernel()), mock.patch.object(qi, '_topk', lambda *a: None)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        for rows in SIZES:
            for start in (0, rows):
                for sparse in (False, True):
                    with self.subTest(rows=rows, chunk_start=start, sparse=sparse):
                        # S0: not last, so the suffix materializes the PP tensor.
                        s0_group = NS(is_first_rank=True, is_last_rank=False)
                        with mock.patch.object(_model, 'get_pp_group', lambda: s0_group), \
                                mock.patch('vllm.distributed.parallel_state.get_pp_group', lambda: s0_group), \
                                mock.patch.object(ix, 'get_forward_context', self.metadata_getter(rows, start)):
                            lane, _ = self.run_stage(rows, start, first=True, sparse=sparse, oq=oq, cq=cq)
                        hidden = lane.output['hidden_states']
                        self.assertEqual(hidden.shape, (rows, HIDDEN * HC_COUNT))
                        # PP: the real channel at this size, sender and receiver, comm stubbed.
                        received = self.through_pp(hidden, rows, start)
                        self.assertEqual(received.shape, hidden.shape)
                        # S1: the last stage has no suffix (bulk chunks are not sampled).
                        s1_group = NS(is_first_rank=False, is_last_rank=True)
                        with mock.patch.object(_model, 'get_pp_group', lambda: s1_group), \
                                mock.patch('vllm.distributed.parallel_state.get_pp_group', lambda: s1_group), \
                                mock.patch.object(ix, 'get_forward_context', self.metadata_getter(rows, start)):
                            lane, _ = self.run_stage(rows, start, first=False, sparse=sparse, oq=oq, cq=cq,
                                                     intermediate={'hidden_states': received})
                        self.assertIsNone(lane.output)
        self.assertTrue(launches['_qsa_sparse_staged_gqa_kernel'].calls)
        self.assertTrue(launches['_s6_mark'].calls)

    def metadata_getter(self, rows, start):
        private, _, _, _ = lc.private_attention_groups(stock_groups(), torch_module=torch,
                                                       row_capacity=None if rows <= 1024 else (STOCK_ROWS, rows))
        metadata = build(private, rows, start)
        return lambda: NS(attn_metadata=metadata)

    def through_pp(self, hidden, rows, start):
        import lane_comm
        from e3_types import TransactionId
        sent = []

        class Comm:
            disabled = False

            def send(self, tensor, rank, stream=None):
                sent.append(tensor)

            def recv(self, tensor, rank, stream=None):
                tensor.copy_(sent.pop(0))
        comm = Comm()
        ticket_event = NS(record=lambda stream: None, query=lambda: True)
        txn = TransactionId(1, 'request', 0, start, start + rows, 0)
        with mock.patch.object(torch.cuda, 'stream', lambda stream: nullcontext()), \
                mock.patch.object(torch.cuda, 'Event', lambda enable_timing=False: ticket_event), \
                mock.patch.object(torch, 'empty', pinless(torch.empty)):
            sender = lane_comm.PrefillPPChannel(comm, 'cpu', None, max_tokens=rows, hidden_width=HIDDEN * HC_COUNT, dtype=torch.bfloat16,
                                                is_sender=True, peer_rank=1, slots=2, torch_module=torch)
            receiver = lane_comm.PrefillPPChannel(comm, 'cpu', None, max_tokens=rows, hidden_width=HIDDEN * HC_COUNT, dtype=torch.bfloat16,
                                                  is_sender=False, peer_rank=0, slots=2, torch_module=torch)
            sender.send(hidden, txn, 0, 0)
            ticket = receiver.post_receive(txn, 0, 0, rows)
            payload = receiver.poll_receive(ticket)
        self.assertEqual(sender.describe()['max_tokens'], rows)
        return payload


def pinless(empty):
    """No CUDA here, so pin_memory has nothing to pin; the channel's other arguments stay."""
    def call(*args, **kwargs):
        kwargs.pop('pin_memory', None)
        return empty(*args, **kwargs)
    return call


if __name__ == '__main__':
    unittest.main()
