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
* the per-layer attention metadata through a native-only view of the pinned
  ``build_attn_metadata`` groups, with private persistent builder storage,
  and a ``ForwardContext`` the lane installs per segment.

All pinned vLLM entry points come through a ``Ports`` object so the control
flow is CPU-testable with fakes; ``PinnedPorts.load()`` binds the real ones in
the worker. Nothing here synchronizes. The stock groups/builders remain
unchanged for ordinary decode. The native factory owns a shallow group view
whose builders have private mutable tensor buffers and container state.
"""
from contextlib import contextmanager
import copy
from dataclasses import dataclass
import enum
import time

from e3_types import E3ContractError, TransactionId

PAD_SLOT_ID_DEFAULT = -1


class LaneChunkError(E3ContractError):
    pass


# inwait build-2 (DESIGN-2): per rank, selected per arm through attn_meta_select's control while paused idle FULL.  Off until selected.
SLOTS_FAST = dict(selected=False, calls=0)


def select_slots_fast(on):
    if type(on) is not bool:
        raise LaneChunkError('slots-fast-not-bool')
    SLOTS_FAST['selected'] = on
    return dict(SLOTS_FAST)


# inwait build-3 (DESIGN-3): per rank, selected the same way. PREPARE's host work in slices, so a stock RPC waiting in the
# rank's ring runs between them, and one cached score block so the indexer's growing buffers stop stacking up in the cache.
PREPARE_SLICES = dict(selected=False, sliced=0, yields=0, peek_missing=0)
# coverage-throughput build-2 (a), prepare-ahead: a launch setting (the factory builds a second private builder set at startup).
# staged: slices 1-2 of the next chunk built early; adopted: a PREPARE that ran slice 3 only; dropped: a staged chunk not used;
# refused: a stage request that did not qualify (counted by reason); errors: a staging that raised (nothing staged).
PREPARE_AHEAD = dict(selected=False, staged=0, adopted=0, dropped=0, refused={}, errors=0)
PREPARE_AHEAD_ENV = 'FLASH_S5_PREPARE_AHEAD'
AHEAD_LINE = 'flash-s5 prepare-ahead '
AHEAD_EVERY = 128            # decisions between receipt lines (the first decision always has one)


def ahead_line():
    """The worker's cumulative counters as one receipt line (a reader takes an arm's delta per worker pid)."""
    import json, os
    row = dict(PREPARE_AHEAD, refused=dict(sorted(PREPARE_AHEAD['refused'].items())), pid=os.getpid())
    return AHEAD_LINE + json.dumps(row, sort_keys=True, separators=(',', ':'))


def ahead_count(key, reason=None, emit=None):
    """Build-2 (a): one prepare-ahead decision counted; the receipt line at the first and every AHEAD_EVERY-th. Never raises."""
    try:
        if key == 'refused':
            PREPARE_AHEAD['refused'][reason] = PREPARE_AHEAD['refused'].get(reason, 0) + 1
        else:
            PREPARE_AHEAD[key] += 1
        decisions = (PREPARE_AHEAD['staged'] + PREPARE_AHEAD['adopted'] + PREPARE_AHEAD['dropped'] + PREPARE_AHEAD['errors']
                     + sum(PREPARE_AHEAD['refused'].values()))
        if decisions == 1 or decisions % AHEAD_EVERY == 0:
            if emit is None:
                from executor_hooks import _progress_logger
                emit = _progress_logger().info
            emit(ahead_line())
    except Exception:
        pass


def prepare_ahead_from_env(env):
    """The launch setting, read the same way by the engine and every worker; unset or 0 is today's one builder set."""
    value = env.get(PREPARE_AHEAD_ENV, '0')
    if value not in ('0', '1'):
        raise LaneChunkError(PREPARE_AHEAD_ENV + ' must be 0 or 1')
    return value == '1'


# parity-ahead (READING-parity-gaps-20261107.md A.2/A.5): the staged chunk's host-to-device copies without a synchronize.
# Today LaneChunk's ``torch.tensor(..., device=cuda)`` (a pageable copy with non_blocking False) waits for the prefill stream
# to drain the live chunk's tail on the worker's RPC thread (~45 ms p50 on stage 1). With the per-arm selection on, the ahead
# builds each small tensor in pinned host memory and copies it with non_blocking=True on the caller's current stream, the lane
# prefill stream, so every kernel that reads it later on that stream runs after the copy. The pinned sources are kept with one
# event recorded on that stream after the last copy, and dropped only once that event is queried complete (``H2DKeep``).
# Nothing synchronizes. Counters for the receipt line only; the off path never touches them.
AHEAD_NOSYNC = dict(staged=0, copies=0, bytes=0, kept=0, retired=0, keep_max=0, errors=0)


class H2DStaging:
    """One staged chunk's host-to-device copies: pinned sources, non-blocking copies on the current (prefill) stream."""

    def __init__(self, torch_module, device):
        self.torch = torch_module
        self.device = device
        self.hosts = []
        self.copies = 0
        self.bytes = 0

    def _pinned(self, host):
        if not host.is_pinned():
            host = host.pin_memory()
        return host

    def upload(self, host):
        """A device copy of a CPU tensor: pinned source, ``non_blocking=True``, on the current stream; the source is kept."""
        torch = self.torch
        dev = torch.empty(tuple(host.shape), dtype=host.dtype, device=self.device)
        if host.numel() == 0:
            return dev
        host = self._pinned(host)
        dev.copy_(host, non_blocking=True)
        self.hosts.append(host)
        self.copies += 1
        self.bytes += int(host.numel() * host.element_size())
        return dev

    def tensor(self, values, dtype):
        """``torch.tensor(values, dtype=dtype, device=device)`` without the blocking copy."""
        return self.upload(self.torch.tensor(values, dtype=dtype, pin_memory=True))

    def copy_into(self, target, values, dtype):
        """``target.copy_(torch.tensor(values))`` from a pinned source (the in-place LaneTables row write)."""
        host = self.torch.tensor(values, dtype=dtype, pin_memory=True)
        target.copy_(host, non_blocking=True)
        self.hosts.append(host)
        self.copies += 1
        self.bytes += int(host.numel() * host.element_size())
        return target


class H2DKeep:
    """The staged copies' pinned sources, each set with the event recorded after its last copy; never waits."""

    def __init__(self, torch_module):
        self.torch = torch_module
        self.entries = []

    def seal(self, staging):
        """After the last copy is enqueued: one event on the current stream, the sources kept until it completes."""
        if not staging.hosts:
            return None
        event = self.torch.cuda.Event()
        event.record()
        self.entries.append((event, tuple(staging.hosts)))
        AHEAD_NOSYNC['kept'] += 1
        AHEAD_NOSYNC['keep_max'] = max(AHEAD_NOSYNC['keep_max'], len(self.entries))
        return event

    def prune(self):
        """Drop the sources whose event is complete (a query, never a wait); the rest stay."""
        keep = []
        for entry in self.entries:
            if bool(entry[0].query()):
                AHEAD_NOSYNC['retired'] += 1
            else:
                keep.append(entry)
        self.entries = keep
        return len(keep)
SCORE_WARM = dict(selected=False, bytes=0, allocs=0, flushes=0, rewarms=0, errors=0, retries_seen=None)
SCORE_WARM_SLACK_BYTES = 16 * 1024 * 1024
SCORE_WARM_STEP_BYTES = 32 * 1024 * 1024
# inwait build-5b: the warm block on rank 0 only (off: every rank), and a flush at each selection so arms start matched.
SCORE_WARM_RANK0 = dict(selected=False, skipped=0)
ALLOCATOR_FLUSH = dict(selected=False, flushes=0)
_WORKER_RANK = [None]
SCORE_ROWS = 1024
SCORE_COMPRESS = 4


def select_prepare_slices(on):
    if type(on) is not bool:
        raise LaneChunkError('prepare-slices-not-bool')
    PREPARE_SLICES['selected'] = on
    return dict(PREPARE_SLICES)


def _flush_allocator():
    import torch
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def select_score_warm(on):
    """Off to on, while paused idle: flush the cache, so the arm starts without an earlier arm's dead blocks."""
    if type(on) is not bool:
        raise LaneChunkError('score-warm-not-bool')
    if on and not SCORE_WARM['selected']:
        _flush_allocator()
        SCORE_WARM.update(flushes=SCORE_WARM['flushes'] + 1, bytes=0, retries_seen=None)
    SCORE_WARM['selected'] = on
    return dict(SCORE_WARM)


def note_worker_rank(rank):
    _WORKER_RANK[0] = rank


def select_score_warm_rank0(on):
    if type(on) is not bool:
        raise LaneChunkError('score-warm-rank0-not-bool')
    SCORE_WARM_RANK0['selected'] = on
    return dict(SCORE_WARM_RANK0)


def select_allocator_flush(on):
    """On, while paused idle: flush at every selection, not only off to on, and the warm block goes with it, so it is warmed again."""
    if type(on) is not bool:
        raise LaneChunkError('allocator-flush-not-bool')
    if on:
        _flush_allocator()
        ALLOCATOR_FLUSH['flushes'] += 1
        SCORE_WARM.update(bytes=0, retries_seen=None)
    ALLOCATOR_FLUSH['selected'] = on
    return dict(ALLOCATOR_FLUSH)


def score_warm_bytes(prompt_end):
    """One score buffer for this prompt (rows x width x 4, as qsa_operation.indexer_score_shape), plus slack, on a 32 MiB step."""
    width = -(-(-(-prompt_end // SCORE_COMPRESS)) // 64) * 64
    need = SCORE_ROWS * max(64, width) * 4 + SCORE_WARM_SLACK_BYTES
    return -(-need // SCORE_WARM_STEP_BYTES) * SCORE_WARM_STEP_BYTES


# inwait build-4 (DESIGN-4): per rank, selected the same way. The partial-tail CoW copy as one slice copy per block, with no
# advanced index and no temporary; and an allocator cap that keeps a declared amount of the device free. One row per chunk
# that zeroes or copies pages times both, on or off, so the box names what the residue chunk's PREPARE does on each TP rank.
COW_FAST = dict(selected=False, calls=0, pairs=0, fallbacks=0, rows=0, row_errors=0, verify_pairs=0, verify_unequal=0, pages_errors=0)
COW_HOST_FILTER = dict(selected=False, kept=0, skipped=0, unfiltered=0)
RESERVED_CAP = dict(selected=False, cap_bytes=0, fraction=0.0, flushes=0, overdrafts=0, overdraft_refused=0)
RESERVED_CAP_FREE_BYTES = 512 * 1024 * 1024
RESERVED_CAP_OVERDRAFT_BYTES = 160 * 1024 * 1024    # the largest score block band 2 warms (score_warm_bytes(131094))
COW_VERIFY = dict(selected=True, skipped=0)         # the byte check at release: on unless an arm selects it off (DESIGN-5d)
_COW_ROW = [None]


def select_cow_fast(on):
    if type(on) is not bool:
        raise LaneChunkError('cow-fast-not-bool')
    COW_FAST['selected'] = on
    return dict(COW_FAST)


def reserved_cap_bytes(total, free, reserved, keep=RESERVED_CAP_FREE_BYTES):
    """The most this process may reserve so that ``keep`` stays free: what is outside the allocator is read, not assumed."""
    other = total - free - reserved
    cap = total - other - keep
    if other < 0 or cap <= 0:
        raise LaneChunkError('reserved-cap-out-of-range')
    return cap


def select_reserved_cap(on):
    """Off to on, while paused idle: flush the cache, read what is outside the allocator, cap reserved. On to off: no cap."""
    if type(on) is not bool:
        raise LaneChunkError('reserved-cap-not-bool')
    if on is not RESERVED_CAP['selected']:
        import torch
        device = torch.cuda.current_device()
        if on:
            _flush_allocator()
            free, total = torch.cuda.mem_get_info(device)
            cap = reserved_cap_bytes(total, free, torch.cuda.memory_reserved(device))
            fraction = cap / total
            torch.cuda.set_per_process_memory_fraction(fraction, device)
            RESERVED_CAP.update(cap_bytes=cap, fraction=fraction, flushes=RESERVED_CAP['flushes'] + 1)
        else:
            torch.cuda.set_per_process_memory_fraction(1.0, device)
            RESERVED_CAP.update(cap_bytes=0, fraction=0.0)
    RESERVED_CAP['selected'] = on
    return dict(RESERVED_CAP)


def score_under_cap(call):
    """The indexer under the reserved cap (READING-box-5c): a score buffer that cannot grow by a few MiB must not kill the engine with 600 MiB free on the device.
    One retry, with reserved allowed at most RESERVED_CAP_OVERDRAFT_BYTES past the cap, then the cap again; what it reserved stays cached and serves the next chunk.
    The retry repeats only the indexer's cache-row writes, with the same values, before the allocation. Counted, so a run that used it shows it, and its min free shows the price.
    With no cap selected, or too little free on the device, the error stands."""
    import torch
    try:
        return call()
    except torch.OutOfMemoryError:
        if not RESERVED_CAP['selected'] or not RESERVED_CAP['cap_bytes']:
            raise
    # Outside the except: the traceback's frames, and the tensors they hold, are gone.
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    if free < RESERVED_CAP_OVERDRAFT_BYTES:
        RESERVED_CAP['overdraft_refused'] += 1
        return call()
    RESERVED_CAP['overdrafts'] += 1
    torch.cuda.set_per_process_memory_fraction(min(1.0, (RESERVED_CAP['cap_bytes'] + RESERVED_CAP_OVERDRAFT_BYTES) / total), device)
    try:
        return call()
    finally:
        torch.cuda.set_per_process_memory_fraction(RESERVED_CAP['fraction'], device)


def select_cow_verify(on):
    """inwait build-5d: off for the fair hf read, so a non-hf arm doesn't carry the check at release (READING-box-5c)."""
    if type(on) is not bool:
        raise LaneChunkError('cow_verify selection must be a bool')
    COW_VERIFY['selected'] = on
    return dict(COW_VERIFY)


def select_cow_host_filter(on):
    """inwait build-5: the host-view pass copies only pairs whose target is in a QSA group's row, stock's filter. Selected per arm while paused."""
    if type(on) is not bool:
        raise LaneChunkError('cow_host_filter selection must be a bool')
    COW_HOST_FILTER['selected'] = on
    return dict(COW_HOST_FILTER)


def qsa_host_pairs(runner, block_tables, pairs):
    """The pairs whose target is in a QSA group's row of this request; stock's _qsa_copy_host_pages keeps the same ones. Pairs unchanged when the groups are unknown, counted."""
    groups = getattr(runner, '_qsa_host_groups', None)
    if groups is None:
        COW_HOST_FILTER['unfiltered'] += 1
        return pairs
    targets = {block for group in groups for block in block_tables[group]}
    kept = tuple(pair for pair in pairs if pair[1] in targets)
    COW_HOST_FILTER['kept'] += len(kept)
    COW_HOST_FILTER['skipped'] += len(pairs) - len(kept)
    return kept


def take_cow_row():
    """This chunk's row for the worker's release line, once; None when it zeroed and copied nothing."""
    row, _COW_ROW[0] = _COW_ROW[0], None
    return row


def cow_row_times(row):
    """Device and host microseconds from a row, after its chunk drained. An unfinished pair reads None; never waits."""
    marks = row.pop('marks', None) or ()
    out = dict(zero_dev_us=None, copy_dev_us=None, host_view_dev_us=None)
    if len(marks) == 4 and all(mark is not None for mark in marks) and all(bool(mark.query()) for mark in marks):
        out.update(zero_dev_us=int(round(marks[0].elapsed_time(marks[1]) * 1e3)),
                   copy_dev_us=int(round(marks[1].elapsed_time(marks[2]) * 1e3)),
                   host_view_dev_us=int(round(marks[2].elapsed_time(marks[3]) * 1e3)))
    row.update(out)
    return row


def _kv_arenas(runner):
    """The known KV arenas as (start, end, class), the pools before the allocations that could hold them. Pointers stay in here; never logged."""
    def span(tensor, name):
        start = tensor.data_ptr()
        return start, start + tensor.numel() * tensor.element_size(), name
    layers = runner.compilation_config.static_forward_context.values()
    arenas = [span(layer._qsa_host_kv, 'host') for layer in layers if getattr(layer, '_qsa_host_kv', None) is not None]
    arenas += [span(layer._qsa_gpu_slots, 'slot') for layer in layers if getattr(layer, '_qsa_gpu_slots', None) is not None]
    for cache in runner.kv_caches:
        storage = cache.untyped_storage()
        arenas.append((storage.data_ptr(), storage.data_ptr() + storage.nbytes(), 'device'))
    return arenas


def cow_residency(runner, views, pairs, host_pairs=None):
    """Where each pair's pages sit, per copy pass (k: runner.kv_caches, h: the QSA host views) and per distinct view: counts and bytes by source and target class.

    host: a registered ``_qsa_host_kv`` pool, so UVA from the GPU's side. slot: a layer's GPU placeholder. device: another KV allocation. other: none of them.
    """
    arenas = _kv_arenas(runner)

    def class_of(pointer):
        for start, end, name in arenas:
            if start <= pointer < end:
                return name
        return 'other'
    out = {}
    num_blocks = runner.kv_cache_config.num_blocks
    for tag, caches, chosen in (('k', runner.kv_caches, pairs), ('h', views or (), pairs if host_pairs is None else host_pairs)):
        for blocks in _block_views(caches, num_blocks):
            for src, dst in chosen:
                source, target = blocks[src], blocks[dst]
                name = '%s_%s_%s' % (tag, class_of(source.data_ptr()), class_of(target.data_ptr()))
                out[name + '_pairs'] = out.get(name + '_pairs', 0) + 1
                out[name + '_bytes'] = out.get(name + '_bytes', 0) + target.numel() * target.element_size()
    return out


def cow_host_verify(runner, pairs, chunk_start):
    """Host bytes, after the lane drained: each pair's copied prefix against its source, on every QSA pool. Raw bytes, so a NaN equals itself."""
    import torch
    compared, unequal, prefix = 0, 0, None
    for layer in runner.compilation_config.static_forward_context.values():
        host = getattr(layer, '_qsa_host_kv', None)
        if host is None:
            continue
        prefix = chunk_start % host.shape[2]
        for src, dst in pairs:
            compared += 1
            unequal += not torch.equal(host[src, :, :prefix].view(torch.uint8), host[dst, :, :prefix].view(torch.uint8))
    return dict(verify_pairs=compared, verify_unequal=unequal, verify_prefix=prefix)


def cow_row_pages(row, chunk_start):
    """Residency and the byte check for a row, at release so the chunk's path pays nothing. A failure is one count in the line, the times stay."""
    pages = row.pop('pages', None)
    if pages and pages['pairs'] and not COW_VERIFY['selected']:
        COW_VERIFY['skipped'] += 1
    elif pages and pages['pairs']:
        try:
            host_pairs = pages.get('host_pairs', pages['pairs'])
            row.update(cow_residency(pages['runner'], pages['views'], pages['pairs'], host_pairs))
            row.update(cow_host_verify(pages['runner'], host_pairs, chunk_start))
            COW_FAST['verify_pairs'] += row['verify_pairs']
            COW_FAST['verify_unequal'] += row['verify_unequal']
        except Exception:
            row['pages_error'] = 1
            COW_FAST['pages_errors'] += 1
    return row


def _block_views(caches, num_blocks):
    """Stock's walk: each distinct view once, each storage's byte view once."""
    seen, storages = set(), set()
    for cache in caches:
        key = (cache.device, cache.data_ptr())
        if key in seen:
            continue
        seen.add(key)
        blocks = _blocks_of(cache, num_blocks, storages)
        if blocks is not None:
            yield blocks


def _blocks_of(cache, num_blocks, storages):
    """Stock ``copy_kv_cache_blocks_inplace``'s view of one cache with dim 0 in scheduler blocks; None when the storage was copied."""
    import torch
    kernel_blocks, remainder = divmod(cache.shape[0], num_blocks)
    if remainder:
        raise LaneChunkError('kernel blocks not divisible by scheduler blocks')
    storage = cache.untyped_storage()
    stride = cache.stride(0) * cache.element_size() * kernel_blocks
    if storage.nbytes() == num_blocks * stride:
        key = (cache.device, storage.data_ptr())
        if key in storages:
            return None
        storages.add(key)
        blocks = torch.empty(0, dtype=torch.uint8, device=cache.device)
        blocks.set_(storage)
        return blocks.view(num_blocks, -1)
    return cache.unflatten(0, (num_blocks, kernel_blocks))


def copy_blocks_fast(caches, num_blocks, pairs):
    """The same bytes as stock's ``blocks[dst] = blocks[src]``, one ``copy_`` per block per distinct view, on the current stream.

    Stock reads every source before it writes. Pairs here run in order, so a pair list where order could matter goes to stock:
    returns None, nothing copied. Else {'views': distinct views copied, 'bytes': bytes copied}.
    """
    sources, targets = [src for src, _ in pairs], [dst for _, dst in pairs]
    if len(set(targets)) != len(targets) or set(sources) & set(targets):
        return None
    views, size = 0, 0
    for blocks in _block_views(caches, num_blocks):
        views += 1
        for src, dst in pairs:
            target = blocks[dst]
            target.copy_(blocks[src], non_blocking=True)
            size += target.numel() * target.element_size()
    return dict(views=views, bytes=size)


MAX_NATIVE_BUILDER_BYTES = 256 * 1024 * 1024
_SHARED_CONFIG_FIELDS = {
    'kv_cache_spec', 'vllm_config', 'device', 'backend',
    'model_config', 'parallel_config', 'cache_config', 'compilation_config',
    'attention_config', 'speculative_config',
}
_NUMPY_ALIAS_FIELDS = ('mm_prefix_query_ranges_cpu', 'mm_prefix_query_ranges_np')
_MAMBA_CONTEXT_BUFFER_FIELDS = (
    'state_base_addrs', 'state_block_strides', 'state_elem_sizes', 'state_inner_sizes',
    'state_conv_widths', 'state_group_indices', 'state_dim_row_count',
    'state_dim_row_stride', 'block_table_ptrs', 'aligned_state_indices',
    'num_accepted_tokens_out', 'mamba_state_idx_buf', 'num_scheduled_tokens_buf',
    'num_computed_tokens_buf', 'num_draft_tokens_buf', 'precopy_src_col_buf',
    'precopy_token_bias_buf')


def _torch_immutable(value):
    return (type(value).__module__ == 'torch'
            and type(value).__name__ in ('device', 'dtype', 'layout', 'memory_format', 'qscheme'))


def _is_vllm_config(value):
    return type(value).__module__ == 'vllm.config' or type(value).__module__.startswith('vllm.config.')


def _is_numpy_array(value):
    return (type(value).__module__.split('.', 1)[0] == 'numpy'
            and hasattr(value, '__array_interface__'))


def _clone_builder_value(value, torch_module, copied, field, max_bytes):
    """Copy builder-owned mutable storage while retaining immutable config.

    Attention metadata builders are long-lived and may own GPU scratch. The
    native lane calls the same metadata dispatcher as decode, so each native
    builder view must own its tensor scratch and Python containers. Config
    objects (for example VllmConfig/KVCacheSpec) remain shared read-only.
    """
    if _is_vllm_config(value) or field.rsplit('.', 1)[-1] in _SHARED_CONFIG_FIELDS:
        return value
    if _is_numpy_array(value):
        raise LaneChunkError('unclassified native metadata builder NumPy array: ' + field)
    if isinstance(value, torch_module.Tensor):
        nbytes = int(value.numel()) * int(value.element_size())
        copied_bytes = sum(size for size, _ in copied)
        if copied_bytes + nbytes > max_bytes:
            raise LaneChunkError(
                f'native metadata builder storage exceeds {max_bytes} byte per-rank cap '
                f'(would require {copied_bytes + nbytes} bytes)')
        result = value.clone()
        if int(value.numel()) and result.data_ptr() == value.data_ptr():
            raise LaneChunkError('native metadata tensor clone aliases source')
        copied.append((int(value.numel()) * int(value.element_size()), value))
        return result
    if type(value) is list:
        return [_clone_builder_value(item, torch_module, copied, field+'[]', max_bytes) for item in value]
    if type(value) is tuple:
        return tuple(_clone_builder_value(item, torch_module, copied, field+'()', max_bytes) for item in value)
    if type(value) is dict:
        if any(type(key) not in (str, bytes, int, float, bool, tuple) for key in value):
            raise LaneChunkError('unclassified native metadata builder key: ' + field)
        return {key: _clone_builder_value(item, torch_module, copied, field+'.value', max_bytes)
                for key, item in value.items()}
    if type(value) is set:
        if any(type(item) not in (str, bytes, int, float, bool, tuple) for item in value):
            raise LaneChunkError('unclassified native metadata builder set: ' + field)
        return set(value)
    if (value is None or type(value) in (str, bytes, int, float, bool)
            or isinstance(value, enum.Enum) or isinstance(value, type)
            or callable(value) or _torch_immutable(value)):
        return value
    raise LaneChunkError('unclassified native metadata builder value: ' + field)


def _builder_field_class(name, value, torch_module):
    if _is_vllm_config(value) or name in _SHARED_CONFIG_FIELDS:
        return 'shared-read-only'
    if name in _NUMPY_ALIAS_FIELDS and _is_numpy_array(value):
        return 'shared-read-only-numpy-alias'
    if _is_numpy_array(value):
        raise LaneChunkError('unclassified native metadata builder NumPy array: ' + name)
    if isinstance(value, torch_module.Tensor):
        return 'private-tensor'
    if type(value) in (list, tuple, dict, set):
        return 'private-container'
    if (value is None or type(value) in (str, bytes, int, float, bool)
            or isinstance(value, enum.Enum) or isinstance(value, type)
            or callable(value) or _torch_immutable(value)):
        return 'shared-read-only'
    raise LaneChunkError('unclassified native metadata builder field: ' + name)


# decode-phase dp5c: QSAMetadataBuilder's persistent row buffers (common/qsa_cache.py:624-650), sized by the stock budget.
_QSA_ROW_BUFFERS = ('token_to_req_buffer', 'slot_mapping_buffer', 'logical_positions_buffer',
                    'visible_blocks_buffer')
_QSA_ROW_METADATA = ('token_to_req', 'slot_mapping', 'logical_positions', 'visible_blocks')


def _scaled_builder_shape(builder, name, value, row_capacity, torch_module):
    """The private shape of a stock-row-sized builder tensor above the stock budget, else None (a plain clone).

    Nothing else may be sized by the stock rows: an unknown tensor of that length refuses at startup, because
    the stock builder's ``[:num_tokens]`` on it would return too few rows without an error.
    """
    if row_capacity is None or not isinstance(value, torch_module.Tensor):
        return None
    stock_rows, chunk_tokens = row_capacity
    if name in _QSA_ROW_BUFFERS:
        if tuple(value.shape) != (stock_rows,):
            raise LaneChunkError(f'native metadata row buffer {name} is not the stock {stock_rows} rows')
        return (chunk_tokens,)
    if name == 'k_work_metadata_buffer':
        if int(value.numel()) == 0:
            return None
        ratio, capacity = builder.compress_ratio, builder.request_capacity
        if tuple(value.shape) != ((stock_rows + (ratio - 1) * capacity) // ratio, 2):
            raise LaneChunkError('native metadata k-work buffer is not the stock shape')
        return ((chunk_tokens + (ratio - 1) * capacity) // ratio, 2)
    if int(value.dim()) > 0 and int(value.shape[0]) == stock_rows:
        raise LaneChunkError('stock-row-sized native metadata builder field is not scaled: ' + name)
    return None


def _scaled_builder_value(value, shape, torch_module, copied, name, max_bytes):
    nbytes = int(value.element_size())
    for size in shape:
        nbytes *= int(size)
    if sum(size for size, _ in copied) + nbytes > max_bytes:
        raise LaneChunkError(f'native metadata builder storage exceeds {max_bytes} byte per-rank cap at {name}')
    result = torch_module.empty(shape, dtype=value.dtype, device=value.device)
    copied.append((nbytes, value))
    return result


def check_metadata_rows(metadata, num_tokens):
    """decode-phase dp5c: a builder slices a persistent buffer with ``[:num_tokens]``, and past its end that returns
    fewer rows with no error; the next stage then fails far away, or a kernel takes the short view. Every QSA row
    field of this chunk's metadata must hold the chunk's rows. Host shapes only."""
    for name, md in metadata.items():
        if getattr(md, 'visible_blocks', None) is None:
            continue
        for field in _QSA_ROW_METADATA:
            rows = tuple(getattr(md, field).shape)
            if len(rows) != 1 or rows[0] != num_tokens:
                raise LaneChunkError(f'native metadata {name}.{field} holds {rows}, not {num_tokens} rows')


def private_attention_groups(attn_groups, *, torch_module, max_bytes=MAX_NATIVE_BUILDER_BYTES, row_capacity=None):
    """Create a native-only builder view; stock groups/builders are untouched.

    Every builder tensor and container is copied once per LaneChunkFactory,
    not once per chunk. This isolates persistent output/work buffers (including
    QSA's token map, logical positions, visible blocks, slot mapping and
    k-work rows) as well as mutable builder fields such as Mamba-aligned state
    indices. Non-container configuration objects are shared read-only.

    decode-phase dp5c: ``row_capacity`` is (stock rows, native chunk rows), set only above the stock budget; the
    row-sized buffers are then allocated at the chunk's size instead of cloned. None, as always: plain clones.
    """
    copied = []
    result = []
    inventory = []
    for group_row in attn_groups:
        private_row = []
        for group in group_row:
            private_group = copy.copy(group)
            private_builders = []
            for builder in group.metadata_builders:
                private_builder = copy.copy(builder)
                fields = []
                builder_fields = vars(builder)
                present_numpy_aliases = [name for name in _NUMPY_ALIAS_FIELDS if name in builder_fields]
                alias_values = [builder_fields[name] for name in present_numpy_aliases]
                if any(value is not None for value in alias_values):
                    if len(present_numpy_aliases) != 2 or not all(
                            _is_numpy_array(builder_fields[name]) for name in _NUMPY_ALIAS_FIELDS):
                        raise LaneChunkError('incomplete native metadata builder NumPy alias pair: '
                                             +type(builder).__name__+' '
                                             +','.join(present_numpy_aliases))
                    import numpy as np
                    if not np.shares_memory(builder_fields[_NUMPY_ALIAS_FIELDS[0]],
                                            builder_fields[_NUMPY_ALIAS_FIELDS[1]]):
                        raise LaneChunkError('native metadata builder NumPy fields do not alias')
                for name, value in builder_fields.items():
                    field_class = _builder_field_class(name, value, torch_module)
                    bytes_before = sum(size for size, _ in copied)
                    shape = _scaled_builder_shape(builder, name, value, row_capacity, torch_module)
                    if shape is not None:
                        cloned = _scaled_builder_value(value, shape, torch_module, copied, name, max_bytes)
                    else:
                        cloned = (value if field_class.startswith('shared-read-only') else
                                  _clone_builder_value(value, torch_module, copied, name, max_bytes))
                    setattr(private_builder, name, cloned)
                    copied_bytes = sum(size for size, _ in copied) - bytes_before
                    fields.append(dict(name=name, classification=field_class,
                                       tensor_bytes=copied_bytes))
                private_builders.append(private_builder)
                inventory.append(dict(builder=type(builder).__module__+'.'+type(builder).__name__,
                                      layers=list(getattr(group, 'layer_names', ())),fields=fields))
            private_group.metadata_builders = private_builders
            private_row.append(private_group)
        result.append(private_row)
    return result, sum(size for size, _ in copied), len(copied), inventory


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
    native_end: int = 0     # coverage-soak-fix: wire_v1 AdmissionRecord.native_end; 0 is prompt_end - 1

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise LaneChunkError('transaction must be the canonical TransactionId')
        if type(self.native_end) is not int or self.native_end < 0:
            raise LaneChunkError('native_end must be a non-negative int')
        if self.native_end == 0 and type(self.prompt_end) is int:
            object.__setattr__(self, 'native_end', self.prompt_end - 1)
        if type(self.prompt_end) is not int or self.prompt_end < 0:
            raise LaneChunkError('prompt_end must be a non-negative int')
        if type(self.state_epoch) is not int or self.state_epoch < 0:
            raise LaneChunkError('state_epoch must be a non-negative int')
        if not (self.chunk_start < self.chunk_end <= self.native_end <= self.prompt_end - 1):
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


def _no_mark():
    return None


class Ports:
    """Pinned entry points the chunk needs; see ``PinnedPorts.load`` for the real ones."""

    required = ('InputBatch', 'BatchDescriptor', 'MambaHybridAttnMetadata', 'build_attn_metadata',
                'build_slot_mappings_by_layer', 'create_forward_context', 'override_forward_context',
                'cudagraph_none', 'align_preprocess', 'mamba_context', 'copy_blocks_inplace',
                'block_copy', 'new_request', 'pad_slot_id', 'make_buffer',
                'mamba_groups', 'validate_mamba_copy_funcs')

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
        from vllm.v1.worker.mamba_utils import (
            MambaSpecDecodeGPUContext, get_mamba_groups, validate_mamba_state_copy_funcs)
        from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

        def make_buffer(device):
            return lambda n, dtype: CpuGpuBuffer(n, dtype=dtype, device=device)

        def align_preprocess(idx_mapping, state_idx, num_computed, query_start_loc, accepted,
                             src_col, src_off, num_reqs, block_size):
            block = 256
            preprocess_mamba_align_fused_kernel[((num_reqs + block - 1) // block,)](
                idx_mapping, state_idx, num_computed, query_start_loc, accepted, src_col, src_off,
                num_reqs, BLOCK_SIZE=block, MAMBA_BLOCK_SIZE=block_size)

        def cow_mark():
            # One timing event on this scope's stream (the prefill stream); read at release.
            import torch
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event

        return Ports(InputBatch=InputBatch, BatchDescriptor=BatchDescriptor, cow_mark=cow_mark,
                     MambaHybridAttnMetadata=MambaHybridAttnMetadata,
                     build_attn_metadata=build_attn_metadata,
                     build_slot_mappings_by_layer=build_slot_mappings_by_layer,
                     create_forward_context=create_forward_context,
                     override_forward_context=override_forward_context,
                     cudagraph_none=CUDAGraphMode.NONE, align_preprocess=align_preprocess,
                     mamba_context=MambaSpecDecodeGPUContext,
                     copy_blocks_inplace=copy_kv_cache_blocks_inplace,
                     block_copy=KVCacheBlockCopy, new_request=NewRequestData,
                     pad_slot_id=PAD_SLOT_ID, make_buffer=make_buffer,
                     mamba_groups=get_mamba_groups,
                     validate_mamba_copy_funcs=validate_mamba_state_copy_funcs)


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

    def write(self, admission, staging=None):
        if len(admission.block_tables) != self.num_groups:
            raise LaneChunkError('admission block tables do not cover every KV group')
        views = []
        for group, row in enumerate(admission.block_tables):
            ids = self.expanded(group, row)
            target = self.rows[group]
            if len(ids) > int(target.shape[1]):
                raise LaneChunkError('admission row exceeds the lane table capacity')
            if ids and staging is not None:
                # parity-ahead: the same in-place write from a pinned source (stream-ordered after the live chunk's readers).
                staging.copy_into(target[0, :len(ids)], ids, self.torch.int32)
            elif ids:
                target[0, :len(ids)].copy_(self.torch.tensor(ids, dtype=self.torch.int32),
                                           non_blocking=True)
            if len(ids) < int(target.shape[1]):
                target[0, len(ids):].zero_()
            views.append(target[:1])
        return tuple(views)

    def slot_mappings_fast(self, admission, pad):
        """slot_mappings' values by tensor arithmetic, one group at a time; inwait build-2 (DESIGN-2)."""
        torch = self.torch
        positions = torch.arange(admission.chunk_start, admission.chunk_end, dtype=torch.int64)
        out = torch.full((self.num_groups, admission.num_tokens), pad, dtype=torch.int64)
        for group, row in enumerate(admission.block_tables):
            if not self.slot_mapping_enabled[group]:
                continue
            kbs = self.kernel_block_sizes[group]
            ids = self.expanded(group, row)
            first, last = admission.chunk_start // kbs, (admission.chunk_end - 1) // kbs
            if last >= len(ids):
                raise LaneChunkError('admission row too short for the chunk positions')
            table = torch.tensor(ids[first:last + 1], dtype=torch.int64)
            out[group] = table[positions // kbs - first] * kbs + positions % kbs
        SLOTS_FAST['calls'] += 1
        return out

    def slot_mappings(self, admission, pad):
        """[num_groups, num_tokens] slot ids computed on the CPU from the admission rows."""
        n = admission.num_tokens
        if (SLOTS_FAST['selected'] and n > 0 and len(admission.block_tables) == self.num_groups
                and admission.chunk_end - admission.chunk_start == n):
            return self.slot_mappings_fast(admission, pad)
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
        # The first chunk is allocated exactly like a later one: its new pages
        # must be zeroed and its CoW destinations copied before any layer reads
        # them. Registration alone is not admission.
        self._apply_allocated_pages(admission)
        self.req_id = admission.request_uid
        self.req_index = req_index
        self.generation = admission.request_generation
        return req_index

    def _apply_allocated_pages(self, admission):
        """Zero allocation-owned pages and apply CoW copies (shared by all chunks)."""
        runner = self.runner
        zero_ids, pairs = admission.new_block_ids_to_zero, admission.cow_pairs
        if not zero_ids and not pairs:
            return
        mark = getattr(self.ports, 'cow_mark', None) or _no_mark
        views = getattr(runner, '_qsa_host_views', None)
        marks = [mark()]
        started = time.monotonic_ns()
        if zero_ids:
            zeroer = getattr(runner, 'kv_block_zeroer', None)
            if zeroer is None:
                raise LaneChunkError('runner has no KV block zeroer for the new blocks')
            zeroer.zero_block_ids(list(zero_ids))
        marks.append(mark())
        zeroed = time.monotonic_ns()
        fast = None
        if pairs:
            num_blocks = runner.kv_cache_config.num_blocks
            if COW_FAST['selected']:
                fast = copy_blocks_fast(runner.kv_caches, num_blocks, pairs)
                COW_FAST['fallbacks'] += int(fast is None)
            if fast is None:
                copies = [self.ports.block_copy(src, dst) for src, dst in pairs]
                self.ports.copy_blocks_inplace(runner.kv_caches, num_blocks, copies)
            else:
                COW_FAST['calls'] += 1
                COW_FAST['pairs'] += len(pairs)
        marks.append(mark())
        copied = time.monotonic_ns()
        host_fast = None
        host_pairs = pairs
        if pairs and views and COW_HOST_FILTER['selected']:
            host_pairs = qsa_host_pairs(runner, admission.block_tables, pairs)
        if host_pairs and views:
            # The QSA pages themselves: pinned host RAM, these are their UVA views. runner.kv_caches holds the layers' GPU placeholders.
            # coverage-soak-fix build-3 (lead, 6 Nov 11:45): always stock's copy, as stock's runner does. copy_blocks_fast here was a
            # host-to-host copy_ per block that ran on the CPU (~9.6 ms a pair), mapping the src and dst pool pages into the worker for
            # good: the nr worker RssShmem growth with the byte check off. Stock's indexed copy runs on the GPU; ny stayed flat on it.
            # host_fast stays None (host_fast 0 in the row); the device copy above keeps its fast path.
            copies = [self.ports.block_copy(src, dst) for src, dst in host_pairs]
            self.ports.copy_blocks_inplace(views, runner.kv_cache_config.num_blocks, copies)
        marks.append(mark())
        ended = time.monotonic_ns()
        _COW_ROW[0] = dict(marks=marks, zero_ids=len(zero_ids or ()), pairs=len(pairs or ()),
                           fast=int(fast is not None), views=(fast or {}).get('views'), bytes=(fast or {}).get('bytes'),
                           host_views=len(views or ()), host_fast=int(host_fast is not None),
                           host_bytes=(host_fast or {}).get('bytes'), host_pairs=len(host_pairs or ()),
                           pages=dict(runner=runner, views=views, pairs=list(pairs or ()), host_pairs=list(host_pairs or ())),
                           zero_host_us=(zeroed - started) // 1000,
                           copy_host_us=(copied - zeroed) // 1000, host_view_us=(ended - copied) // 1000)
        COW_FAST['rows'] += 1

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
        self._apply_allocated_pages(admission)
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
            # Stock MambaHybridModelState initializes this cache only in the
            # first real preprocess_state, not initialize_kv_cache. Resolve
            # the same model descriptors and run the same stock validator
            # (mamba_hybrid.py:_ensure_align_ctx), but keep them lane-owned:
            # never initialize/rebind the ordinary context to private tables
            # and never populate its lazy cache on behalf of ordinary decode.
            groups = ports.mamba_groups(runner.kv_cache_config)
            copy_funcs = state.model.get_mamba_state_copy_funcs(
                {group.mamba_type for group in groups})
            ports.validate_mamba_copy_funcs(groups, copy_funcs)
        # The block tables are batch ordered (one admitted row), but the
        # decision arrays below are request-slot ordered. Stock V2 snapshots
        # the *whole* accepted-token array before resolving idx_mapping; its
        # private snapshot must have that same capacity even for one request.
        # Narrowing only accepted tokens would misindex nonzero request slots.
        shape = tuple(state.num_accepted_tokens_gpu.shape)
        if len(shape) != 1 or type(shape[0]) is not int or shape[0] < 1:
            raise LaneChunkError('Mamba accepted-token buffer has invalid request-slot capacity')
        self.request_slot_capacity = shape[0]
        for value in (state._mamba_state_idx_gpu, state._mamba_src_col_gpu,
                      state._mamba_src_off_gpu,
                      runner.req_states.num_computed_tokens.gpu):
            if tuple(value.shape) != shape:
                raise LaneChunkError('Mamba request-slot decision buffer capacities differ')
        ctx = ports.mamba_context.create(max_num_reqs=self.request_slot_capacity,
                                         kv_cache_config=runner.kv_cache_config,
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


def attach_qsa_host_metadata(metadata, layer_names, query_start_loc_cpu,
                             seq_lens_cpu, admission):
    """Carry this admitted chunk's private CPU plan onto its QSA metadata.

    Production QSA's builder copies these same CPU arrays to lists. Native
    execution requires the plan independently of the ordinary-path opt-in.
    Device arrays are inspected for shape only; never read back or substituted.
    """
    for value in (query_start_loc_cpu, seq_lens_cpu):
        if value is None or str(getattr(value, 'device', None)) != 'cpu':
            raise LaneChunkError('QSA host plan requires owned CPU metadata buffers')
    host = (query_start_loc_cpu.tolist(), seq_lens_cpu.tolist())
    expected = ([0, admission.num_tokens], [admission.chunk_end])
    if host != expected or any(type(x) is not int for row in host for x in row):
        raise LaneChunkError('QSA host plan differs from the admitted chunk')
    if not isinstance(metadata, dict) or not layer_names:
        raise LaneChunkError('QSA host plan needs the actual named layer metadata')
    for name in layer_names:
        md = metadata.get(name)
        starts = getattr(md, 'query_start_loc', None)
        lengths = getattr(md, 'seq_lens', None)
        if (starts is None or lengths is None
                or tuple(starts.shape) != (2,) or tuple(lengths.shape) != (1,)
                or getattr(md, 'num_actual_tokens', None) != admission.num_tokens):
            raise LaneChunkError('QSA main metadata does not describe the admitted chunk')
        prior = getattr(md, '_qsa_host_meta', None)
        if prior is not None and prior != host:
            raise LaneChunkError('QSA main metadata already carries a different CPU plan')
        md._qsa_host_meta = host
    return host


class LaneChunk:
    """Private inputs, metadata and forward context for one admitted chunk."""

    def __init__(self, runner, ports, tables, align, admission, req_index, *, torch_module,
                 attn_groups=None, staging=None, probe=None):
        torch = torch_module
        device = runner.device
        n = admission.num_tokens
        self.runner = runner
        self.ports = ports
        self.admission = admission
        self.req_index = req_index
        self.attn_groups = runner.attn_groups if attn_groups is None else attn_groups
        self.torch = torch
        # parity-ahead: ``staging`` (an H2DStaging) only for a staged chunk with the per-arm selection on; ``probe`` only
        # with the receipt installed: it stamps the end of the first device tensor (today's wait for the stream).
        self._staging = staging
        if staging is None:
            self.input_ids = torch.tensor(list(admission.input_ids), dtype=torch.int32, device=device)
            if probe is not None:
                probe()
            self.positions = torch.arange(admission.chunk_start, admission.chunk_end, dtype=torch.int64,
                                          device=device)
            self.query_start_loc = torch.tensor([0, n], dtype=torch.int32, device=device)
            self.query_start_loc_cpu = torch.tensor([0, n], dtype=torch.int32, device='cpu')
            self.seq_lens = torch.tensor([admission.chunk_end], dtype=torch.int32, device=device)
            self.seq_lens_cpu = torch.tensor([admission.chunk_end], dtype=torch.int32, device='cpu')
            self.idx_mapping = torch.tensor([req_index], dtype=torch.int64, device=device)
            self.is_padding = torch.zeros((n,), dtype=torch.bool, device=device)
            # The pinned PLE/ngram path consumes ``[num_reqs, context_len]`` and
            # slices ``[:num_reqs]``; this private chunk carries exactly one row.
            self.ngram_context = torch.tensor([list(admission.ngram_context)],
                                              dtype=torch.int64, device=device)
            self.block_tables = tables.write(admission)
            mappings = tables.slot_mappings(admission, ports.pad_slot_id)
            self.slot_mappings = mappings.to(device) if hasattr(mappings, 'to') else mappings
        else:
            # The same tensors, values and shapes; every host value reaches the device by a pinned non-blocking copy.
            self.input_ids = staging.tensor(list(admission.input_ids), torch.int32)
            if probe is not None:
                probe()
            self.positions = torch.arange(admission.chunk_start, admission.chunk_end, dtype=torch.int64,
                                          device=device)
            self.query_start_loc = staging.tensor([0, n], torch.int32)
            self.query_start_loc_cpu = torch.tensor([0, n], dtype=torch.int32, device='cpu')
            self.seq_lens = staging.tensor([admission.chunk_end], torch.int32)
            self.seq_lens_cpu = torch.tensor([admission.chunk_end], dtype=torch.int32, device='cpu')
            self.idx_mapping = staging.tensor([req_index], torch.int64)
            self.is_padding = torch.zeros((n,), dtype=torch.bool, device=device)
            self.ngram_context = staging.tensor([list(admission.ngram_context)], torch.int64)
            self.block_tables = tables.write(admission, staging=staging)
            mappings = tables.slot_mappings(admission, ports.pad_slot_id)
            self.slot_mappings = staging.upload(mappings) if hasattr(mappings, 'to') else mappings
        self.tables = tables
        self.align = align
        self.attn_metadata = None
        self.forward_context = None
        self.batch = None

    def input_batch(self):
        n = self.admission.num_tokens
        torch = self.torch
        import numpy as np

        # parity-ahead: with ``staging`` (a staged chunk, selection on) the two small device tensors come from pinned sources.
        staging = self._staging
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
            logits_indices=(torch.tensor([n - 1], dtype=torch.int64, device=self.runner.device) if staging is None
                            else staging.tensor([n - 1], torch.int64)),
            cu_num_logits=(torch.tensor([0, 1], dtype=torch.int32, device=self.runner.device) if staging is None
                           else staging.tensor([0, 1], torch.int32)),
            cu_num_logits_np=np.array([0, 1], dtype=np.int32), has_structured_output_reqs=False,
            prompt_lens=None)
        self.batch = batch
        return batch

    def build_metadata(self, precopy=True):
        """Align pre-copy, aligned indices, then the per-layer attention metadata.

        ``precopy=False``, build-3's slices: everything here but the pre-copy reads only this chunk and the lane's private
        tables and builders, and the pre-copy writes neither, so the caller runs it afterwards, in PREPARE's last slice.
        """
        runner = self.runner
        ports = self.ports
        torch = self.torch
        if precopy:
            self.align.precopy(self)
        is_prefilling = torch.ones((1,), dtype=torch.bool)
        mamba = ports.MambaHybridAttnMetadata(is_prefilling=is_prefilling, num_accepted_tokens=None,
                                              num_decode_draft_tokens_cpu=None)
        indices = self.align.aligned_indices(self)
        if indices is not None:
            for group_idx, gid in enumerate(self.align.group_ids):
                for group in self.attn_groups[gid]:
                    builder = group.get_metadata_builder(0)
                    if hasattr(builder, 'mamba_aligned_state_indices'):
                        builder.mamba_aligned_state_indices = indices[group_idx]
        n = self.admission.num_tokens
        self.attn_metadata = ports.build_attn_metadata(
            attn_groups=self.attn_groups, num_reqs=1, num_tokens=n,
            query_start_loc_gpu=self.query_start_loc, query_start_loc_cpu=self.query_start_loc_cpu,
            max_query_len=n, seq_lens=self.seq_lens, max_seq_len=self.admission.chunk_end,
            block_tables=self.block_tables, slot_mappings=self.slot_mappings,
            kv_cache_config=runner.kv_cache_config, seq_lens_cpu_upper_bound=self.seq_lens_cpu,
            dcp_local_seq_lens=None, positions=self.positions, model_specific_attn_metadata=mamba,
            for_cudagraph_capture=False, rswa_prefix_lens=None)
        check_metadata_rows(self.attn_metadata, n)
        from native_resources import _qsa_layers
        self.qsa_host_plan = attach_qsa_host_metadata(
            self.attn_metadata, tuple(name for name, _ in _qsa_layers(runner)),
            self.query_start_loc_cpu, self.seq_lens_cpu, self.admission)
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

    def __init__(self, runner, *, ports=None, torch_module=None, chunk_tokens=1024, prepare_ahead=False):
        if torch_module is None:
            import torch as torch_module
        if type(prepare_ahead) is not bool:
            raise LaneChunkError('prepare-ahead-not-bool')
        self.runner = runner
        self.ports = ports if ports is not None else PinnedPorts.load()
        # decode-phase dp5c: at or below 1024 the builders stay plain clones, without reading the stock budget.
        row_capacity = None
        if chunk_tokens > 1024:
            row_capacity = (int(runner.scheduler_config.max_num_batched_tokens), int(chunk_tokens))
        self.torch = torch_module
        self.tables = LaneTables(runner, torch_module=torch_module)
        self.align = LaneAlign(runner, self.ports, self.tables, torch_module=torch_module)
        self.request = LaneRequest(runner, self.ports, torch_module=torch_module)
        (self.native_attn_groups, self.native_builder_bytes,
         self.native_builder_tensor_count, self.native_builder_fields) = (
            private_attention_groups(runner.attn_groups, torch_module=torch_module, row_capacity=row_capacity)
        )
        # Build-2 (a): with prepare-ahead, a second private set, so the next chunk's metadata is built while this chunk's
        # kernels still read the first. Off: one set, today's bytes; native_builder_* keep describing one set.
        self.builder_sets = [self.native_attn_groups]
        if prepare_ahead:
            self.builder_sets.append(private_attention_groups(runner.attn_groups, torch_module=torch_module,
                                                              row_capacity=row_capacity)[0])
        self.native_builder_sets = len(self.builder_sets)
        self._last_set = None
        self._staged = None
        self.chunks = 0
        self._active_metadata_chunk = None
        self._metadata_generation = 0
        # parity-ahead: the pinned sources of non-blocking staged copies, until their event completes (selection on only).
        self.h2d_keep = None

    def _next_set(self):
        """The builder set for the next chunk: the one the last admitted chunk does not hold; with one set, always it."""
        if len(self.builder_sets) == 1 or self._last_set is None:
            return 0
        return 1 - self._last_set

    def _new_chunk(self, admission, req_index, staging=None, probe=None):
        index = self._next_set()
        if staging is None and probe is None:
            chunk = LaneChunk(self.runner, self.ports, self.tables, self.align, admission, req_index,
                              torch_module=self.torch, attn_groups=self.builder_sets[index])
        else:
            chunk = LaneChunk(self.runner, self.ports, self.tables, self.align, admission, req_index,
                              torch_module=self.torch, attn_groups=self.builder_sets[index], staging=staging, probe=probe)
        chunk.builder_set = index
        return chunk

    def _refuse_ahead(self, reason):
        ahead_count('refused', reason)
        return dict(staged=False, why=reason)

    def stage_ahead(self, admission, nosync=False, probe=None):
        """Build-2 (a): slices 1-2 of a later chunk of the registered request, into the set the live chunk does not hold.

        The caller holds the lane prefill stream, after every command of the live chunk is enqueued there (the safe variant:
        after its state_finalize), so the LaneTables row write and the aligned-index kernel run after the live chunk's
        readers. Nothing shared moves: the request row, new pages, zeroing, CoW, the metadata gate and the pre-copy stay in
        slice 3, at the real PREPARE. Never raises; a failure stages nothing.

        parity-ahead: ``nosync`` (the per-arm selection) builds the same chunk with every host value copied from pinned
        memory, non-blocking, on that same stream; the order above is the stream's, so it holds without the wait. ``probe``
        (the receipt) is called once the chunk's first device tensor exists.
        """
        if nosync is not False or probe is not None:
            return self._stage_ahead_timed(admission, nosync is True, probe)
        if len(self.builder_sets) < 2:
            return self._refuse_ahead('not-selected')
        if type(admission) is not ChunkAdmission:
            return self._refuse_ahead('not-an-admission')
        if (self.request.req_id is None or self.request.req_id != admission.request_uid
                or self.request.generation != admission.request_generation or admission.chunk_start == 0):
            return self._refuse_ahead('not-a-later-chunk')
        live = self._active_metadata_chunk
        if live is not None and getattr(live, 'builder_set', None) == self._next_set():
            return self._refuse_ahead('set-held')
        if self._staged is not None:
            if self._staged[0] == admission:
                return dict(staged=True, why='already')
            self.drop_ahead()
        try:
            chunk = self._new_chunk(admission, self.request.req_index)
            chunk.input_batch()
            chunk.build_metadata(precopy=False)
        except Exception:
            ahead_count('errors')
            return dict(staged=False, why='error')
        self._staged = (admission, chunk)
        ahead_count('staged')
        return dict(staged=True, why='staged')

    def _stage_ahead_timed(self, admission, nosync, probe):
        """stage_ahead's checks and build with the probe, and (nosync) the pinned non-blocking copies kept until their event."""
        if self.h2d_keep is not None:
            self.h2d_keep.prune()
        if len(self.builder_sets) < 2:
            return self._refuse_ahead('not-selected')
        if type(admission) is not ChunkAdmission:
            return self._refuse_ahead('not-an-admission')
        if (self.request.req_id is None or self.request.req_id != admission.request_uid
                or self.request.generation != admission.request_generation or admission.chunk_start == 0):
            return self._refuse_ahead('not-a-later-chunk')
        live = self._active_metadata_chunk
        if live is not None and getattr(live, 'builder_set', None) == self._next_set():
            return self._refuse_ahead('set-held')
        if self._staged is not None:
            if self._staged[0] == admission:
                return dict(staged=True, why='already')
            self.drop_ahead()
        staging = H2DStaging(self.torch, self.runner.device) if nosync else None
        try:
            chunk = self._new_chunk(admission, self.request.req_index, staging=staging, probe=probe)
            chunk.input_batch()
            chunk.build_metadata(precopy=False)
        except Exception:
            ahead_count('errors')
            if staging is not None:
                AHEAD_NOSYNC['errors'] += 1
            return dict(staged=False, why='error')
        finally:
            if staging is not None:
                # Copies already enqueued keep their sources until the event, the failed build's too.
                if self.h2d_keep is None:
                    self.h2d_keep = H2DKeep(self.torch)
                self.h2d_keep.seal(staging)
        if staging is not None:
            chunk._staging = None
            AHEAD_NOSYNC['staged'] += 1
            AHEAD_NOSYNC['copies'] += staging.copies
            AHEAD_NOSYNC['bytes'] += staging.bytes
        self._staged = (admission, chunk)
        ahead_count('staged')
        return dict(staged=True, why='staged', nosync=nosync,
                    copies=0 if staging is None else staging.copies, bytes=0 if staging is None else staging.bytes)

    def drop_ahead(self):
        if self._staged is not None:
            self._staged = None
            ahead_count('dropped')
            return True
        return False

    def adopt(self, admission):
        """The staged chunk for exactly this admission, finished by slice 3 alone; None (and the staged chunk dropped) otherwise."""
        staged, self._staged = self._staged, None
        if staged is None:
            return None
        if staged[0] != admission or not self.can_slice(admission) or staged[1].builder_set != self._next_set():
            ahead_count('dropped')
            return None
        chunk = self._finish(admission, staged[1])
        ahead_count('adopted')
        return chunk

    def _finish(self, admission, chunk):
        """Slice 3: the request row, new pages, the metadata gate, the align pre-copy."""
        if self._active_metadata_chunk is not None:
            raise LaneChunkError('native metadata generation still has live consumers')
        if self.request.update(admission) != chunk.req_index:
            raise LaneChunkError('the request row moved between PREPARE slices')
        self._active_metadata_chunk = chunk
        self._metadata_generation += 1
        chunk.native_metadata_generation = self._metadata_generation
        self._last_set = chunk.builder_set
        self.align.precopy(chunk)
        self.chunks += 1
        return chunk

    def admit(self, admission):
        """Register (first chunk) or update the request, then build the private chunk."""
        if type(admission) is not ChunkAdmission:
            raise LaneChunkError('admit needs a ChunkAdmission')
        if self._active_metadata_chunk is not None:
            raise LaneChunkError('native metadata generation still has live consumers')
        if self.request.req_id is None:
            index = self.request.register(admission)
        else:
            index = self.request.update(admission)
        chunk = self._new_chunk(admission, index)
        self._active_metadata_chunk = chunk
        self._metadata_generation += 1
        chunk.native_metadata_generation = self._metadata_generation
        self._last_set = chunk.builder_set
        chunk.input_batch()
        chunk.build_metadata()
        self.chunks += 1
        return chunk

    def can_slice(self, admission):
        """A later chunk of the registered request: its row index is known, and nothing shared moves before the last slice."""
        return (type(admission) is ChunkAdmission and self.request.req_id is not None
                and self.request.req_id == admission.request_uid
                and self.request.generation == admission.request_generation
                and self._active_metadata_chunk is None)

    def admit_steps(self, admission):
        """``admit`` in three slices, for a later chunk (``can_slice``); each ``next`` is one slice, the last returns the chunk.

        The first two touch only this chunk and the lane's private tables and builders, so a stop between slices leaves
        nothing to undo. All the rest is the last slice: the request row, new pages, the metadata gate, the align pre-copy.
        """
        chunk = self._new_chunk(admission, self.request.req_index)
        chunk.input_batch()
        yield
        chunk.build_metadata(precopy=False)
        yield
        return self._finish(admission, chunk)

    def warm_scores(self, admission):
        """One cached block on the prefill stream (the caller's scope) big enough for this prompt's score buffers.

        From the second chunk on, so the first chunk's working set does not split it. A failure is counted, never raised:
        without the block the allocator behaves as it does today.
        """
        if not SCORE_WARM['selected'] or admission.chunk_start == 0:
            return False
        if SCORE_WARM_RANK0['selected'] and _WORKER_RANK[0] != 0:
            # An unknown rank warms nothing: rank 0's allocs stay 0 and the box says so.
            SCORE_WARM_RANK0['skipped'] += 1
            return False
        torch = self.torch
        try:
            device = self.runner.device
            retries = torch.cuda.memory_stats(device).get('num_alloc_retries', 0)
            if SCORE_WARM['retries_seen'] != retries:
                # A retry emptied the cache, this block with it.
                SCORE_WARM['rewarms'] += int(SCORE_WARM['retries_seen'] is not None)
                SCORE_WARM['bytes'] = 0
            need = score_warm_bytes(admission.prompt_end)
            if need > SCORE_WARM['bytes']:
                block = torch.empty((need,), dtype=torch.uint8, device=device)
                del block
                SCORE_WARM['bytes'] = need
                SCORE_WARM['allocs'] += 1
                retries = torch.cuda.memory_stats(device).get('num_alloc_retries', 0)
            SCORE_WARM['retries_seen'] = retries
            return True
        except Exception:
            SCORE_WARM['errors'] += 1
            return False

    def complete_chunk(self, chunk, *, consumers_complete):
        """Release the single-generation reuse gate after dispatcher drain.

        LaneDispatcher calls this only after all operation events and the
        operation's own consumer frontier have been queried complete.
        """
        if chunk is not self._active_metadata_chunk:
            raise LaneChunkError('native metadata completion does not match active generation')
        if consumers_complete is not True:
            raise LaneChunkError('native metadata consumers are not complete')
        self._active_metadata_chunk = None
        return True

    def release(self):
        if self._active_metadata_chunk is not None:
            raise LaneChunkError('cannot release request with live metadata consumers')
        self.drop_ahead()
        return self.request.release()

    def _align_inventory(self):
        ctx = self.align.ctx
        if ctx is None:
            return dict(enabled=False, gpu_bytes=0, host_bytes=0, buffers=[])
        known = set(_MAMBA_CONTEXT_BUFFER_FIELDS) | {
            'block_size', 'num_states', 'mamba_group_ids', 'num_groups',
            'block_table_stride_req', 'max_num_reqs', 'is_initialized',
            # Names used only by the CPU FakeMambaContext seam.
            'calls', 'tables'}
        buffers = []
        gpu_bytes = host_bytes = 0
        for name, value in vars(ctx).items():
            if name not in known:
                raise LaneChunkError('unclassified native GDN/linear state: ' + name)
            if value is None:
                continue
            tensors = []
            if isinstance(value, self.torch.Tensor):
                tensors.append(('gpu', value))
            else:
                for side in ('gpu', 'np'):
                    tensor = getattr(value, side, None)
                    if tensor is not None:
                        tensors.append((side, tensor))
            for side, tensor in tensors:
                nbytes = (int(tensor.numel()) * int(tensor.element_size())
                          if callable(getattr(tensor, 'numel', None)) else int(tensor.nbytes))
                if side == 'gpu':
                    gpu_bytes += nbytes
                else:
                    host_bytes += nbytes
                buffers.append(dict(field=name,side=side,bytes=nbytes,
                                    shape=list(getattr(tensor,'shape',()))))
        stock = getattr(self.runner.model_state, '_mamba_ctx', None)
        stock_ptrs = set()
        if stock is not None:
            for name in _MAMBA_CONTEXT_BUFFER_FIELDS:
                value = getattr(stock, name, None)
                for side in ('', 'gpu'):
                    tensor = value if not side else getattr(value, side, None)
                    if callable(getattr(tensor, 'data_ptr', None)):
                        stock_ptrs.add(int(tensor.data_ptr()))
        for row in buffers:
            if row['side'] == 'gpu':
                value = getattr(ctx, row['field'])
                if not callable(getattr(value, 'data_ptr', None)):
                    value = getattr(value, 'gpu')
                if int(value.data_ptr()) in stock_ptrs:
                    raise LaneChunkError('native GDN context aliases stock alignment storage')
        shared_state = []
        for name, tensor in (
                ('num_accepted_tokens_gpu', self.runner.model_state.num_accepted_tokens_gpu),
                ('_mamba_state_idx_gpu', self.runner.model_state._mamba_state_idx_gpu),
                ('_mamba_src_col_gpu', self.runner.model_state._mamba_src_col_gpu),
                ('_mamba_src_off_gpu', self.runner.model_state._mamba_src_off_gpu),
                ('num_computed_tokens_gpu', self.runner.req_states.num_computed_tokens.gpu)):
            shared_state.append(dict(field=name,shape=list(tensor.shape),
                bytes=int(tensor.numel())*int(tensor.element_size()),data_ptr=int(tensor.data_ptr()),
                ownership='existing request-slot state; lane writes only its admitted row'))
        return dict(enabled=True,gpu_bytes=gpu_bytes,host_bytes=host_bytes,buffers=buffers,
                    shared_request_state_buffers=shared_state,
                    stock_context_present=stock is not None,stock_aliases=0)

    def inventory(self):
        return {'lane_tables_bytes': self.tables.bytes, 'align': self.align.enabled,
                'chunks': self.chunks, 'registered': self.request.req_id is not None,
                'native_builder_bytes': self.native_builder_bytes,
                'native_builder_tensor_count': self.native_builder_tensor_count,
                'native_builder_fields': self.native_builder_fields,
                'gdn_linear_alignment': self._align_inventory(),
                'metadata_generation_in_flight': self._active_metadata_chunk is not None,
                'metadata_generation': self._metadata_generation}


__all__ = ['ChunkAdmission', 'LaneAlign', 'LaneChunk', 'LaneChunkError', 'LaneChunkFactory',
           'LaneRequest', 'LaneTables', 'PinnedPorts', 'Ports']
