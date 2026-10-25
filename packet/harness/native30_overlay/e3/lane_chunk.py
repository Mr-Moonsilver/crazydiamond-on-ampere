"""Worker-side admission of one parked chunk and its private forward inputs (D4).

The parked request never goes through the stock ``execute_model``, so the lane
does on this worker what ``add_requests``/``update_requests`` would have done
for it, but only the request-row-scoped parts, and builds every batch-scoped
buffer privately:

* request registration in the runner's ``req_states`` (canonical QSA UID via
  ``_qsa_new_uid``, model-state registration, block-table rows staged and
  applied) and per-chunk updates (computed count, full block-table rows,
  zeroing of new blocks, CoW copies incl. the QSA host pages);
* private input tensors (ids, positions, query_start_loc, seq_lens, padding),
  private gathered block tables with stable addresses (``LaneTables``) and
  CPU-computed slot mappings; nothing reads or writes ``runner.input_buffers``;
* a lane-private Mamba align context bound to those lane tables, so the align
  pre-copy, aligned state indices and post-process touch only this request's
  rows (``MambaSpecDecodeGPUContext`` created once per worker);
* the per-layer attention metadata through the pinned ``build_attn_metadata``
  and a ``ForwardContext`` the lane installs per segment.

All pinned vLLM entry points come through a ``Ports`` object so the control
flow is CPU-testable with fakes; ``PinnedPorts.load()`` binds the real ones in
the worker. Nothing here synchronizes. The metadata builders are the shared
group builders (used synchronously inside one host operation, as decode uses
them inside its RPC); whether any builder keeps device scratch that a captured
decode graph still reads is an open audit item for the GPU smoke.
"""
from contextlib import contextmanager
from dataclasses import dataclass

from e3_types import E3ContractError, TransactionId

PAD_SLOT_ID_DEFAULT = -1


class LaneChunkError(E3ContractError):
    pass


def _ints(name, values, *, allow_empty=False):
    if type(values) is not tuple:
        raise LaneChunkError(f'{name} must be a tuple')
    if not values and not allow_empty:
        raise LaneChunkError(f'{name} must not be empty')
    for value in values:
        if type(value) is not int or value < 0:
            raise LaneChunkError(f'{name} must hold non-negative ints')
    return values


@dataclass(frozen=True)
class ChunkAdmission:
    """What the lane consumes from the engine's admission record (frozen CPU values).

    Identity comes from the canonical :class:`e3_types.TransactionId`: the
    request UID/generation and chunk bounds are derived from it, never
    duplicated as independent fields.
    """

    transaction: TransactionId
    prompt_end: int
    input_ids: tuple
    ngram_context: tuple
    block_tables: tuple
    new_block_ids_to_zero: tuple = ()
    cow_pairs: tuple = ()
    state_epoch: int = 0

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise LaneChunkError('transaction must be the canonical TransactionId')
        if type(self.prompt_end) is not int or self.prompt_end < 0:
            raise LaneChunkError('prompt_end must be a non-negative int')
        if type(self.state_epoch) is not int or self.state_epoch < 0:
            raise LaneChunkError('state_epoch must be a non-negative int')
        if not (self.chunk_start < self.chunk_end <= self.prompt_end - 1):
            raise LaneChunkError('chunk must lie inside the prompt and leave the final token')
        _ints('input_ids', self.input_ids)
        if len(self.input_ids) != self.num_tokens:
            raise LaneChunkError('input_ids length differs from the chunk')
        _ints('ngram_context', self.ngram_context, allow_empty=True)
        if type(self.block_tables) is not tuple or not self.block_tables:
            raise LaneChunkError('block_tables must be a non-empty tuple of per-group rows')
        for row in self.block_tables:
            _ints('block_tables row', row, allow_empty=True)
        _ints('new_block_ids_to_zero', self.new_block_ids_to_zero, allow_empty=True)
        if type(self.cow_pairs) is not tuple:
            raise LaneChunkError('cow_pairs must be a tuple')
        for pair in self.cow_pairs:
            if type(pair) is not tuple or len(pair) != 2 or any(type(v) is not int or v < 0 for v in pair):
                raise LaneChunkError('cow_pairs must hold (src, dst) int pairs')

    @property
    def request_uid(self):
        return self.transaction.request_uid

    @property
    def request_generation(self):
        return self.transaction.request_generation

    @property
    def chunk_start(self):
        return self.transaction.chunk_start

    @property
    def chunk_end(self):
        return self.transaction.chunk_end

    @property
    def num_tokens(self):
        return self.chunk_end - self.chunk_start

    def __repr__(self):
        """Never log request ids, token ids or block tables; counts only."""
        return (f'ChunkAdmission(transaction={self.transaction!r}, '
                f'prompt_end={self.prompt_end}, '
                f'input_ids=<{len(self.input_ids)} ids>, '
                f'ngram_context=<{len(self.ngram_context)} ids>, '
                f'block_tables=<{len(self.block_tables)} groups>, '
                f'new_block_ids_to_zero=<{len(self.new_block_ids_to_zero)} ids>, '
                f'cow_pairs=<{len(self.cow_pairs)} pairs>, '
                f'state_epoch={self.state_epoch})')


class Ports:
    """Pinned entry points the chunk needs; see ``PinnedPorts.load`` for the real ones."""

    required = ('InputBatch', 'BatchDescriptor', 'MambaHybridAttnMetadata', 'build_attn_metadata',
                'build_slot_mappings_by_layer', 'create_forward_context', 'override_forward_context',
                'cudagraph_none', 'align_preprocess', 'mamba_context', 'copy_blocks_inplace',
                'block_copy', 'new_request', 'pad_slot_id', 'make_buffer')

    def __init__(self, **values):
        missing = [name for name in self.required if name not in values]
        if missing:
            raise LaneChunkError('ports missing: ' + ','.join(missing))
        for name, value in values.items():
            setattr(self, name, value)


class PinnedPorts:
    @staticmethod
    def load():
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import (BatchDescriptor, create_forward_context,
                                          override_forward_context)
        from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
        from vllm.v1.core.sched.output import NewRequestData
        from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, build_slot_mappings_by_layer
        from vllm.v1.worker.gpu.block_table import PAD_SLOT_ID
        from vllm.v1.worker.gpu.input_batch import InputBatch
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
            MambaHybridAttnMetadata, preprocess_mamba_align_fused_kernel)
        from vllm.v1.utils import CpuGpuBuffer
        from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext
        from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

        def make_buffer(device):
            return lambda n, dtype: CpuGpuBuffer(n, dtype=dtype, device=device)

        def align_preprocess(idx_mapping, state_idx, num_computed, query_start_loc, accepted,
                             src_col, src_off, num_reqs, block_size):
            block = 256
            preprocess_mamba_align_fused_kernel[((num_reqs + block - 1) // block,)](
                idx_mapping, state_idx, num_computed, query_start_loc, accepted, src_col, src_off,
                num_reqs, BLOCK_SIZE=block, MAMBA_BLOCK_SIZE=block_size)

        return Ports(InputBatch=InputBatch, BatchDescriptor=BatchDescriptor,
                     MambaHybridAttnMetadata=MambaHybridAttnMetadata,
                     build_attn_metadata=build_attn_metadata,
                     build_slot_mappings_by_layer=build_slot_mappings_by_layer,
                     create_forward_context=create_forward_context,
                     override_forward_context=override_forward_context,
                     cudagraph_none=CUDAGraphMode.NONE, align_preprocess=align_preprocess,
                     mamba_context=MambaSpecDecodeGPUContext,
                     copy_blocks_inplace=copy_kv_cache_blocks_inplace,
                     block_copy=KVCacheBlockCopy, new_request=NewRequestData,
                     pad_slot_id=PAD_SLOT_ID, make_buffer=make_buffer)


# --- persistent lane tables ----------------------------------------------------------

class LaneTables:
    """Per-group gathered block-table rows with stable addresses for the lane's one request."""

    def __init__(self, runner, *, torch_module):
        tables = runner.block_tables
        self.torch = torch_module
        self.num_groups = int(tables.num_kv_cache_groups)
        self.blocks_per_kv_block = tuple(int(v) for v in tables.blocks_per_kv_block)
        self.kernel_block_sizes = tuple(int(v) for v in tables.kernel_block_sizes)
        self.slot_mapping_enabled = tuple(bool(v) for v in tables._slot_mapping_enabled)
        self.rows = []
        for group in range(self.num_groups):
            capacity = int(tables.block_tables[group].gpu.shape[1])
            self.rows.append(torch_module.zeros((1, capacity), dtype=torch_module.int32,
                                                device=runner.device))
        self.bytes = sum(int(row.numel() * row.element_size()) for row in self.rows)

    def expanded(self, group, row):
        bpk = self.blocks_per_kv_block[group]
        if bpk > 1:
            return [b * bpk + k for b in row for k in range(bpk)]
        return list(row)

    def write(self, admission):
        if len(admission.block_tables) != self.num_groups:
            raise LaneChunkError('admission block tables do not cover every KV group')
        views = []
        for group, row in enumerate(admission.block_tables):
            ids = self.expanded(group, row)
            target = self.rows[group]
            if len(ids) > int(target.shape[1]):
                raise LaneChunkError('admission row exceeds the lane table capacity')
            if ids:
                target[0, :len(ids)].copy_(self.torch.tensor(ids, dtype=self.torch.int32),
                                           non_blocking=True)
            if len(ids) < int(target.shape[1]):
                target[0, len(ids):].zero_()
            views.append(target[:1])
        return tuple(views)

    def slot_mappings(self, admission, pad):
        """[num_groups, num_tokens] slot ids computed on the CPU from the admission rows."""
        n = admission.num_tokens
        rows = []
        for group, row in enumerate(admission.block_tables):
            ids = self.expanded(group, row)
            kbs = self.kernel_block_sizes[group]
            slots = []
            for position in range(admission.chunk_start, admission.chunk_end):
                if not self.slot_mapping_enabled[group]:
                    slots.append(pad)
                    continue
                logical = position // kbs
                if logical >= len(ids):
                    raise LaneChunkError('admission row too short for the chunk positions')
                slots.append(ids[logical] * kbs + position % kbs)
            rows.append(slots)
        return self.torch.tensor(rows, dtype=self.torch.int64).view(self.num_groups, n)


# --- request registration ------------------------------------------------------------

class LaneRequest:
    """The lane's one registered request on this worker (request-row-scoped state)."""

    def __init__(self, runner, ports, *, torch_module):
        self.runner = runner
        self.ports = ports
        self.torch = torch_module
        self.req_id = None
        self.req_index = None
        self.generation = None
        self.chunks = 0

    def _new_request(self, admission, block_ids):
        tokens = ([0] * admission.chunk_start + list(admission.input_ids)
                  + [0] * (admission.prompt_end - admission.chunk_end))
        return self.ports.new_request(
            req_id=admission.request_uid, prompt_token_ids=tokens, prefill_token_ids=tokens,
            mm_features=[], sampling_params=None, pooling_params=None, block_ids=block_ids,
            num_computed_tokens=admission.chunk_start, lora_request=None)

    def register(self, admission):
        if self.req_id is not None:
            raise LaneChunkError('lane request already registered')
        runner = self.runner
        states = runner.req_states
        if admission.request_uid in states.req_id_to_index:
            raise LaneChunkError('request is already present in the runner state')
        block_ids = tuple(list(row) for row in admission.block_tables)
        data = self._new_request(admission, block_ids)
        states.add_request(req_id=admission.request_uid, prompt_len=admission.prompt_end,
                           all_token_ids=list(data.prefill_token_ids),
                           num_computed_tokens=admission.chunk_start, max_tokens=1)
        req_index = int(states.req_id_to_index[admission.request_uid])
        runner._qsa_new_uid(req_index)
        runner.model_state.add_request(req_index, data)
        runner.block_tables.append_block_ids(req_index, block_ids, overwrite=True)
        states.apply_staged_writes()
        runner.model_state.apply_staged_writes()
        runner.block_tables.apply_staged_writes()
        self.req_id = admission.request_uid
        self.req_index = req_index
        self.generation = admission.request_generation
        return req_index

    def update(self, admission):
        if self.req_id != admission.request_uid or self.generation != admission.request_generation:
            raise LaneChunkError('admission names another request/generation than the registered one')
        runner = self.runner
        states = runner.req_states
        index = self.req_index
        states.num_computed_tokens_np[index] = admission.chunk_start
        states.num_computed_prefill_tokens[index] = admission.chunk_start
        states.num_computed_tokens.stage_write_elem(index, admission.chunk_start)
        runner.block_tables.append_block_ids(
            index, tuple(list(row) for row in admission.block_tables), overwrite=True)
        states.apply_staged_writes()
        runner.block_tables.apply_staged_writes()
        if admission.new_block_ids_to_zero:
            zeroer = getattr(runner, 'kv_block_zeroer', None)
            if zeroer is None:
                raise LaneChunkError('runner has no KV block zeroer for the new blocks')
            zeroer.zero_block_ids(list(admission.new_block_ids_to_zero))
        if admission.cow_pairs:
            copies = [self.ports.block_copy(src, dst) for src, dst in admission.cow_pairs]
            self.ports.copy_blocks_inplace(runner.kv_caches, runner.kv_cache_config.num_blocks, copies)
            views = getattr(runner, '_qsa_host_views', None)
            if views:
                self.ports.copy_blocks_inplace(views, runner.kv_cache_config.num_blocks, copies)
        self.chunks += 1
        return index

    def release(self):
        if self.req_id is None:
            return None
        remover = getattr(self.runner, '_remove_request', None)
        if not callable(remover):
            raise LaneChunkError('runner has no _remove_request for the lane release')
        remover(self.req_id)
        released = self.req_id
        self.req_id = self.req_index = self.generation = None
        return released


# --- the chunk ---------------------------------------------------------------------------

class LaneAlign:
    """Lane-private Mamba align context over the lane tables (created once per worker)."""

    def __init__(self, runner, ports, tables, *, torch_module):
        state = runner.model_state
        self.runner = runner
        self.ports = ports
        self.torch = torch_module
        self.enabled = bool(getattr(state, '_align_mode', False))
        self.ctx = None
        self.block_size = None
        if not self.enabled:
            return
        group_ids, spec = state._get_mamba_group_info(runner.kv_cache_config)
        self.group_ids = tuple(int(g) for g in group_ids)
        self.block_size = int(spec.block_size)
        copy_funcs = getattr(state, '_mamba_state_copy_funcs', None)
        if copy_funcs is None:
            raise LaneChunkError('runner model state has no Mamba copy functions yet')
        ctx = ports.mamba_context.create(max_num_reqs=1, kv_cache_config=runner.kv_cache_config,
                                         copy_funcs=copy_funcs, device=runner.device,
                                         make_buffer=ports.make_buffer(runner.device))
        forward_context = runner.vllm_config.compilation_config.static_forward_context
        ctx.initialize_from_forward_context(runner.kv_cache_config, forward_context, copy_funcs,
                                            [tables.rows[g] for g in self.group_ids])
        self.ctx = ctx

    def precopy(self, chunk):
        if not self.enabled:
            return False
        state = self.runner.model_state
        self.ports.align_preprocess(chunk.idx_mapping, state._mamba_state_idx_gpu,
                                    self.runner.req_states.num_computed_tokens.gpu,
                                    chunk.query_start_loc, state.num_accepted_tokens_gpu,
                                    state._mamba_src_col_gpu, state._mamba_src_off_gpu, 1,
                                    self.block_size)
        self.ctx.run_fused_precopy(1, state._mamba_state_idx_gpu, state._mamba_src_col_gpu,
                                   state._mamba_src_off_gpu, chunk.idx_mapping)
        return True

    def aligned_indices(self, chunk):
        if not self.enabled:
            return None
        return self.ctx.compute_aligned_state_indices(chunk.seq_lens, 1)

    def postprocess(self, chunk):
        """After the chunk's last state writer: advance the aligned state position."""
        if not self.enabled:
            return False
        state = self.runner.model_state
        states = self.runner.req_states
        state.num_accepted_tokens_gpu[chunk.req_index].fill_(1)
        states.num_computed_tokens.stage_write_elem(chunk.req_index, chunk.admission.chunk_end)
        states.apply_staged_writes()
        self.ctx.run_fused_postprocess_align(1, state.num_accepted_tokens_gpu,
                                             state._mamba_state_idx_gpu,
                                             states.num_computed_tokens.gpu, chunk.idx_mapping)
        return True


class LaneChunk:
    """Private inputs, metadata and forward context for one admitted chunk."""

    def __init__(self, runner, ports, tables, align, admission, req_index, *, torch_module):
        torch = torch_module
        device = runner.device
        n = admission.num_tokens
        self.runner = runner
        self.ports = ports
        self.admission = admission
        self.req_index = req_index
        self.torch = torch
        self.input_ids = torch.tensor(list(admission.input_ids), dtype=torch.int32, device=device)
        self.positions = torch.arange(admission.chunk_start, admission.chunk_end, dtype=torch.int64,
                                      device=device)
        self.query_start_loc = torch.tensor([0, n], dtype=torch.int32, device=device)
        self.query_start_loc_cpu = torch.tensor([0, n], dtype=torch.int32)
        self.seq_lens = torch.tensor([admission.chunk_end], dtype=torch.int32, device=device)
        self.seq_lens_cpu = torch.tensor([admission.chunk_end], dtype=torch.int32)
        self.idx_mapping = torch.tensor([req_index], dtype=torch.int64, device=device)
        self.is_padding = torch.zeros((n,), dtype=torch.bool, device=device)
        self.ngram_context = torch.tensor(list(admission.ngram_context), dtype=torch.int64,
                                          device=device)
        self.block_tables = tables.write(admission)
        mappings = tables.slot_mappings(admission, ports.pad_slot_id)
        self.slot_mappings = mappings.to(device) if hasattr(mappings, 'to') else mappings
        self.tables = tables
        self.align = align
        self.attn_metadata = None
        self.forward_context = None
        self.batch = None

    def input_batch(self):
        n = self.admission.num_tokens
        torch = self.torch
        import numpy as np

        batch = self.ports.InputBatch(
            req_ids=[self.admission.request_uid], num_reqs=1, num_reqs_after_padding=1,
            idx_mapping=self.idx_mapping, idx_mapping_np=np.array([self.req_index], dtype=np.intp),
            expanded_idx_mapping=self.idx_mapping,
            expanded_local_pos=torch.zeros((1,), dtype=torch.int32, device=self.runner.device),
            num_scheduled_tokens=np.array([n], dtype=np.int32), num_tokens=n,
            num_tokens_after_padding=n, num_draft_tokens=0, num_draft_tokens_per_req=None,
            query_start_loc=self.query_start_loc, query_start_loc_np=np.array([0, n], dtype=np.int32),
            seq_lens=self.seq_lens, seq_lens_cpu_upper_bound=self.seq_lens_cpu,
            dcp_local_seq_lens=None,
            num_computed_tokens_np=np.array([self.admission.chunk_start], dtype=np.int32),
            prefill_len_np=np.array([self.admission.prompt_end], dtype=np.int32),
            num_computed_prefill_tokens_np=np.array([self.admission.chunk_start], dtype=np.int32),
            is_prefilling_np=np.array([True]), has_prefill=True, input_ids=self.input_ids,
            positions=self.positions, is_padding=self.is_padding,
            logits_indices=torch.tensor([n - 1], dtype=torch.int64, device=self.runner.device),
            cu_num_logits=torch.tensor([0, 1], dtype=torch.int32, device=self.runner.device),
            cu_num_logits_np=np.array([0, 1], dtype=np.int32), has_structured_output_reqs=False,
            prompt_lens=None)
        self.batch = batch
        return batch

    def build_metadata(self):
        """Align pre-copy, aligned indices, then the per-layer attention metadata."""
        runner = self.runner
        ports = self.ports
        torch = self.torch
        self.align.precopy(self)
        is_prefilling = torch.ones((1,), dtype=torch.bool)
        mamba = ports.MambaHybridAttnMetadata(is_prefilling=is_prefilling, num_accepted_tokens=None,
                                              num_decode_draft_tokens_cpu=None)
        indices = self.align.aligned_indices(self)
        if indices is not None:
            for group_idx, gid in enumerate(self.align.group_ids):
                for group in runner.attn_groups[gid]:
                    builder = group.get_metadata_builder(0)
                    if hasattr(builder, 'mamba_aligned_state_indices'):
                        builder.mamba_aligned_state_indices = indices[group_idx]
        n = self.admission.num_tokens
        self.attn_metadata = ports.build_attn_metadata(
            attn_groups=runner.attn_groups, num_reqs=1, num_tokens=n,
            query_start_loc_gpu=self.query_start_loc, query_start_loc_cpu=self.query_start_loc_cpu,
            max_query_len=n, seq_lens=self.seq_lens, max_seq_len=self.admission.chunk_end,
            block_tables=self.block_tables, slot_mappings=self.slot_mappings,
            kv_cache_config=runner.kv_cache_config, seq_lens_cpu_upper_bound=self.seq_lens_cpu,
            dcp_local_seq_lens=None, positions=self.positions, model_specific_attn_metadata=mamba,
            for_cudagraph_capture=False, rswa_prefix_lens=None)
        self.forward_context = ports.create_forward_context(
            self.attn_metadata, runner.vllm_config, cudagraph_runtime_mode=ports.cudagraph_none,
            batch_descriptor=ports.BatchDescriptor(num_tokens=n, num_reqs=1, uniform=False),
            slot_mapping=ports.build_slot_mappings_by_layer(self.slot_mappings, runner.kv_cache_config),
            is_padding=self.is_padding)
        return self.attn_metadata

    @contextmanager
    def forward_scope(self):
        if self.forward_context is None:
            raise LaneChunkError('build_metadata before installing the lane forward context')
        with self.ports.override_forward_context(self.forward_context):
            yield self.forward_context

    def lane_inputs(self):
        return dict(positions=self.positions, input_ids=self.input_ids,
                    query_start_loc=self.query_start_loc, ngram_context=self.ngram_context)

    def finish_state(self):
        """Advance the request's computed count/align position after the chunk's writers."""
        return self.align.postprocess(self)


class LaneChunkFactory:
    """Per-worker owner of the lane tables, align context and the one registered request."""

    def __init__(self, runner, *, ports=None, torch_module=None):
        if torch_module is None:
            import torch as torch_module
        self.runner = runner
        self.ports = ports if ports is not None else PinnedPorts.load()
        self.torch = torch_module
        self.tables = LaneTables(runner, torch_module=torch_module)
        self.align = LaneAlign(runner, self.ports, self.tables, torch_module=torch_module)
        self.request = LaneRequest(runner, self.ports, torch_module=torch_module)
        self.chunks = 0

    def admit(self, admission):
        """Register (first chunk) or update the request, then build the private chunk."""
        if type(admission) is not ChunkAdmission:
            raise LaneChunkError('admit needs a ChunkAdmission')
        if self.request.req_id is None:
            index = self.request.register(admission)
        else:
            index = self.request.update(admission)
        chunk = LaneChunk(self.runner, self.ports, self.tables, self.align, admission, index,
                          torch_module=self.torch)
        chunk.input_batch()
        chunk.build_metadata()
        self.chunks += 1
        return chunk

    def release(self):
        return self.request.release()

    def inventory(self):
        return {'lane_tables_bytes': self.tables.bytes, 'align': self.align.enabled,
                'chunks': self.chunks, 'registered': self.request.req_id is not None}


__all__ = ['ChunkAdmission', 'LaneAlign', 'LaneChunk', 'LaneChunkError', 'LaneChunkFactory',
           'LaneRequest', 'LaneTables', 'PinnedPorts', 'Ports']
