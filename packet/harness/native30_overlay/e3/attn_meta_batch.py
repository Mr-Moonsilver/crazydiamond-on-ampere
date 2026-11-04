"""Stock decode-step attention metadata: GDN group-batched staging (attn-meta option A), default off.

Switch ``FLASH_ATTN_META_GDN_BATCH``: unset or ``0`` off, ``1`` on, anything else refuses at install. With the
switch off :func:`install` returns before importing torch or vLLM and sets nothing.

What it replaces. On every stock FULL decode step ``MambaHybridModelState.prepare_attn`` builds each of the rank's
18 one-layer GDN KV-cache groups separately: ``mamba_get_block_table_tensor`` (align mode: sub, floor_divide,
clamp_, arange, add, to(int64), gather), two ``copy_`` into the builder's persistent
``non_spec_state_indices_tensor`` / ``non_spec_query_start_loc``, two empty-range fills and one
``GDNAttentionMetadata``: 162 device kernels per step, six of the nine identical across the groups. The FULL graph
reads only those two persistent buffers (``docs/attn-meta/DESIGN.md`` §1).

What it does instead, only on the narrow eligible case (:func:`eligible_groups`): stock ``prepare_attn`` for every
other group (PLE, QSA ring, QSA compressed, FA) with the GDN groups masked out, then the start index once, one
stacked gather, and two ``_foreach_copy_`` into every GDN builder's OWN persistent buffers (same tensors, same
offsets, same values as stock), and per-layer ``GDNAttentionMetadata`` equal to stock's in stock's dict order.
Every other case calls the stock function unchanged.

Where it sits. :func:`install` sets one instance attribute, ``runner.model_state.prepare_attn`` (never a class),
before any seam wrapper (handler_anatomy wraps whatever the instance holds, engine_offers looks the attribute up
at each call). It refuses an instance that already carries a ``prepare_attn`` attribute (a wrapper installed first
would be bypassed) and a model state whose class overrides ``prepare_attn`` (the batched path re-implements only
the stock ``MambaHybridModelState.prepare_attn``).

Native lane (native35 rule). The native lane builds through ``attn_utils.build_attn_metadata`` with private cloned
builders (``lane_chunk.private_attention_groups``) and never calls ``model_state.prepare_attn``. This module adds no
builder field, keeps no reference to any builder between calls, and writes only the builders of the
``attn_groups`` argument the stock step passes. No buffer is shared between native and stock.

Bounded: a fixed counter table (batched steps, stock fallbacks by reason), no logging, no allocation that
outlives the call except the returned metadata.
"""
import os

ENV = 'FLASH_ATTN_META_GDN_BATCH'
SCHEMA = 'attn-meta-gdn-batch/1'
_MARK = '_flash_attn_meta_gdn_batch'

# Fallback reasons, a fixed set (the counter table never grows).
REASONS = ('mode', 'capture', 'ubatch', 'spec', 'cache-mode', 'recoverssm', 'dcp', 'empty', 'not-all-decode',
           'builder', 'block-size', 'width', 'few-groups')


class AttnMetaBatchError(RuntimeError):
    pass


def enabled(env=None):
    env = os.environ if env is None else env
    value = env.get(ENV, '0')
    if value not in ('0', '1'):
        raise AttnMetaBatchError('invalid boolean setting: ' + ENV)
    return value == '1'


def _gdn_cls():
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    return GDNAttentionMetadataBuilder


def eligible_groups(mstate, input_batch, cudagraph_mode, attn_groups, for_capture, ubatch_idx=0):
    """(group ids to batch, None) or (None, reason): stock runs for any reason."""
    from vllm.config.compilation import CUDAGraphMode
    if cudagraph_mode != CUDAGraphMode.FULL:
        return None, 'mode'
    if for_capture:
        return None, 'capture'
    if ubatch_idx != 0:
        return None, 'ubatch'
    cfg = mstate.vllm_config
    if cfg.num_speculative_tokens != 0:
        return None, 'spec'
    if cfg.cache_config.mamba_cache_mode != 'align':
        return None, 'cache-mode'
    if getattr(mstate, 'recoverssm', None) is not None:
        return None, 'recoverssm'
    if getattr(input_batch, 'dcp_local_seq_lens', None) is not None:
        return None, 'dcp'
    n = input_batch.num_reqs
    if n == 0:
        return None, 'empty'
    if int(input_batch.num_scheduled_tokens.max()) != 1 or bool(input_batch.is_prefilling_np[:n].any()):
        return None, 'not-all-decode'
    R = input_batch.num_reqs_after_padding
    gdn = _gdn_cls()
    ids, block = [], None
    for gid, row in enumerate(attn_groups):
        if len(row) != 1 or len(row[0].metadata_builders) != 1:
            continue
        b = row[0].metadata_builders[0]
        if type(b) is not gdn:
            continue
        spec = b.kv_cache_spec
        if (b.use_spec_decode or not b.use_full_cuda_graph or R > b.decode_cudagraph_max_bs
                or spec.num_speculative_blocks != 0):
            return None, 'builder'
        if block is None:
            block = spec.block_size
        if spec.block_size != block:
            return None, 'block-size'
        ids.append(gid)
    if len(ids) < 2:
        return None, 'few-groups'
    return ids, None


class GdnBatch:
    """The per-process batched prepare_attn for one model state (instance-installed)."""

    def __init__(self, mstate, stock):
        self.mstate = mstate
        self.stock = stock                  # the class function, called with the instance
        self.batched = 0
        self.fallback = dict.fromkeys(REASONS, 0)

    def counters(self):
        return dict(schema=SCHEMA, batched=self.batched, fallback=dict(self.fallback))

    def __call__(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config,
                 for_capture=False, ubatch_idx=0):
        mstate = self.mstate
        ids, reason = eligible_groups(mstate, input_batch, cudagraph_mode, attn_groups, for_capture, ubatch_idx)
        if ids is not None and len({block_tables[i].shape[1] for i in ids}) != 1:
            ids, reason = None, 'width'
        if ids is None:
            self.fallback[reason] += 1
            return self.stock(mstate, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups,
                              kv_cache_config, for_capture=for_capture, ubatch_idx=ubatch_idx)
        self.batched += 1
        return batched_prepare_attn(self.stock, mstate, input_batch, cudagraph_mode, block_tables, slot_mappings,
                                    attn_groups, kv_cache_config, ids)


def batched_prepare_attn(stock, mstate, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups,
                         kv_cache_config, ids):
    import torch
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    idset = set(ids)
    masked = [[] if i in idset else row for i, row in enumerate(attn_groups)]
    rest = stock(mstate, input_batch, cudagraph_mode, block_tables, slot_mappings, masked, kv_cache_config,
                 for_capture=False, ubatch_idx=0)
    R = input_batch.num_reqs_after_padding          # FULL: stock's num_reqs
    T = input_batch.num_tokens_after_padding        # FULL: stock's num_actual_tokens
    builders = [attn_groups[i][0].metadata_builders[0] for i in ids]
    bs = builders[0].kv_cache_spec.block_size
    seq_lens = input_batch.seq_lens[:R]
    qsl = input_batch.query_start_loc               # stock copies the whole [R+1] view
    # mamba_get_block_table_tensor's align arithmetic, once instead of len(ids) times
    # (arange(1 + num_speculative_blocks) is arange(1) here: offset 0).
    start = (seq_lens - 1) // bs
    start.clamp_(min=0)
    index = start.to(torch.int64).view(1, R, 1).expand(len(ids), R, 1)
    states = torch.gather(torch.stack([block_tables[i] for i in ids]), 2, index).squeeze(2)
    dst_states = [b.non_spec_state_indices_tensor[:R] for b in builders]
    dst_qsl = [b.non_spec_query_start_loc[:R + 1] for b in builders]
    torch._foreach_copy_(dst_states, list(states.unbind(0)), non_blocking=True)
    torch._foreach_copy_(dst_qsl, [qsl] * len(builders), non_blocking=True)
    # Stock's padding fills cover [num_decodes:batch_size] and [num_decodes+1:batch_size+1]; here
    # num_decodes == batch_size == R, so both ranges are empty: nothing to write.
    gdn = {}
    for i, st, ql in zip(ids, dst_states, dst_qsl):
        md = GDNAttentionMetadata(
            num_prefills=0, num_prefill_tokens=0, num_decodes=R, num_decode_tokens=T,
            num_spec_decodes=0, num_spec_decode_tokens=0, num_actual_tokens=T,
            has_initial_state=None, spec_query_start_loc=None, non_spec_query_start_loc=ql,
            spec_state_indices_tensor=None, non_spec_state_indices_tensor=st,
            spec_sequence_masks=None, spec_token_indx=None, non_spec_token_indx=None,
            num_accepted_tokens=None)
        for name in attn_groups[i][0].layer_names:
            gdn[name] = md
    out = {}
    for i, row in enumerate(attn_groups):
        for g in row:
            for name in g.layer_names:
                out[name] = gdn[name] if i in idset else rest[name]
    return out


def install(runner, *, env=None):
    """Set the batched prepare_attn on this runner's model state (instance attribute), or nothing when off."""
    if not enabled(env):
        return None
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    mstate = getattr(runner, 'model_state', None)
    if not isinstance(mstate, MambaHybridModelState):
        raise AttnMetaBatchError('model state is not a MambaHybridModelState: ' + type(mstate).__name__)
    if type(mstate).prepare_attn is not MambaHybridModelState.prepare_attn:
        raise AttnMetaBatchError('prepare_attn is overridden by ' + type(mstate).__qualname__)
    if getattr(mstate, _MARK, None) is not None:
        raise AttnMetaBatchError('already installed')
    if 'prepare_attn' in vars(mstate):
        raise AttnMetaBatchError('model state already carries an instance prepare_attn (install this first)')
    batch = GdnBatch(mstate, MambaHybridModelState.prepare_attn)
    setattr(mstate, _MARK, batch)
    mstate.prepare_attn = batch
    return batch


def installed(runner):
    return getattr(getattr(runner, 'model_state', None), _MARK, None)


__all__ = ['ENV', 'REASONS', 'SCHEMA', 'AttnMetaBatchError', 'GdnBatch', 'batched_prepare_attn', 'eligible_groups',
           'enabled', 'install', 'installed']
