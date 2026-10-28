"""Native admission metadata from the pinned allocation result.

The scheduler's ``flash_s5_admit`` hook runs before the stock waiting loop and
calls ``KVCacheManager.allocate_slots(..., delay_cache_blocks=True)`` itself.
That call also queues three side effects on the per-group managers:

* ``new_block_ids`` -- blocks the worker must zero (attention groups only),
* ``_pending_cow_copies`` -- retained ``(source, cow)`` endpoint objects for a
  partial prefix hit (and the Mamba "align" running-request relocation),
* ``_pending_boundary_state_offloads`` -- connector-only Mamba boundary offers.

The stock scheduler drains those global queues at the end of the same schedule
step. If admission returns an ordinary scheduler output, the stock drains also
see unrelated already-queued work; the metadata provider must not steal or
reorder it. This provider snapshots every manager queue before the allocation,
takes only the suffix the allocation appended (using each manager's own drain
method under a temporary list swap), and then removes that suffix while leaving
the untouched prefix in place, so stock semantics remain exact for everything
else.

The returned four fields are allocation-specific:

``new_block_ids_to_zero``
    The exact suffix stock would drain, honouring ``needs_kv_cache_zeroing``
    and ``_skip_zero_block_ids``. Empty carries explicit source evidence.
``cow_pairs``
    Ordered ``(source, destination)`` ids with unique destinations. The actual
    endpoint objects (not merely their ids) are retained in the journal and
    released exactly once through the controller's verified lifecycle.
``ngram_context``
    Flat ``ngram_size - 1`` token ids ending at the chunk start, left-padded
    with the configured EOS. Empty only when the model declares no PLE layers.
``unpublished_blocks``
    Pages this admission allocated (the pinned ``KVCacheBlocks`` result),
    plus CoW destinations that left the returned row (Mamba "align" relocates
    the hash before the copy finishes). Cached prefix pages, null placeholders
    and a previous chunk's partially filled tail are not owned again. Block
    hash state is never used as ownership evidence.

Boundary-state offers are audited, not invented: with no connector the pinned
scheduler drains and discards them exactly as stock does, and they stay in the
manager queue for that drain. With a connector they carry a consumer that the
four-field admission record cannot describe; the provider raises after seeing
such an offer, and the controller parks/quarantines the allocation with every
queued side effect and retained endpoint still journaled.

Install/capture/abort/retire call sites are documented in
``s5_startup.install_engine`` and ``PrefillAdmissionController``:

* provider installation: :func:`install_admission_metadata` from
  ``install_engine`` only;
* ``begin`` before ``allocate_slots``; ``capture`` after a successful
  allocation; ``abort`` on a failed allocation;
* ``AllocationCapture.release_retained`` only from controller commit or
  cancel, after the existing worker-release/commit authority has spoken.
  It decrements retained allocator references; it is not a completion proof.
"""
import hashlib
from collections.abc import Sequence

from e3_types import E3ContractError

SINGLE_TYPE_PATH = 'v1/core/single_type_kv_cache_manager.py'
KV_CACHE_MANAGER_PATH = 'v1/core/kv_cache_manager.py'
MODEL_STATE_PATH = 'models/qwen4_exp/nvidia/model_state.py'
BLOCK_POOL_PATH = 'v1/core/block_pool.py'
SCHEDULER_PATH = 'v1/core/sched/scheduler.py'

# Verified pinned bytes this provider reasons about. Any drift is refused.
SINGLE_TYPE_SHA256 = 'c49dcf97240537443c956c249b9c0e2baf3c5c77a6afeec28eb0a2a48de2ac18'
KV_CACHE_MANAGER_SHA256 = '58800535eda854580d84e1025bf32fe01b2a43ce0bba501b4f3d125e4f2c8afa'
MODEL_STATE_SHA256 = '1b25dc38e570473a4a9058e8cc8a66d1f0b15f338ed9e1e7ba16e34cf16bed4c'
BLOCK_POOL_SHA256 = '7f781378710e616624f87d95ccc7b245e1e741c85f3751268e672c2a262038e8'
SCHEDULER_SHA256 = '4d797e5ab6c5abaa06229dd88035ceede7562f1dbe054bdf7ad59be659c04c87'

_PINNED = (
    (SINGLE_TYPE_PATH, SINGLE_TYPE_SHA256),
    (KV_CACHE_MANAGER_PATH, KV_CACHE_MANAGER_SHA256),
    (MODEL_STATE_PATH, MODEL_STATE_SHA256),
    (BLOCK_POOL_PATH, BLOCK_POOL_SHA256),
    (SCHEDULER_PATH, SCHEDULER_SHA256),
)

_QUEUE_NEW = 'new_block_ids'
_QUEUE_COW = '_pending_cow_copies'
_QUEUE_BOUNDARY = '_pending_boundary_state_offloads'
_QUEUES = (_QUEUE_NEW, _QUEUE_COW, _QUEUE_BOUNDARY)

_MISSING = object()


def verify_sources(reference_root):
    """Refuse allocation metadata reasoned from unreviewed pinned bytes."""
    from pathlib import Path
    root = Path(reference_root)
    for relative, expected in _PINNED:
        path = root / relative
        source = path.read_bytes()
        got = hashlib.sha256(source).hexdigest()
        if got != expected:
            raise E3ContractError(f'unreviewed pinned source {relative}: {got}')
    return True


def _same_item(left, right):
    """Identity-preserving comparison for queue prefix entries."""
    if left is right:
        return True
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _same_item(a, b) for a, b in zip(left, right))
    if hasattr(left, 'block_id') or hasattr(right, 'block_id'):
        # Queue prefixes hold the actual endpoint objects; a same-id copy is a
        # different incarnation and must not be treated as unchanged.
        return left is right
    return type(left) is type(right) and left == right


def _unique(block_ids):
    seen = set()
    unique = []
    for block_id in block_ids:
        if block_id in seen:
            continue
        seen.add(block_id)
        unique.append(block_id)
    return tuple(unique)


def _block_id(block, what):
    if getattr(block, 'is_null', False):
        raise E3ContractError(what + ' is the null block')
    block_id = getattr(block, 'block_id', None)
    if type(block_id) is not int or block_id <= 0:
        raise E3ContractError(what + ' is not a positive physical block id')
    return block_id


def _list_attr(owner, name, what):
    value = getattr(owner, name, _MISSING)
    if type(value) is not list:
        raise E3ContractError(f'{what} does not expose a {name} list')
    return value


def _text_config(scheduler):
    config = getattr(scheduler, 'vllm_config', None)
    model_config = getattr(config, 'model_config', None)
    hf_config = getattr(model_config, 'hf_text_config', None)
    if hf_config is None:
        hf_config = getattr(model_config, 'hf_config', None)
    if hf_config is None:
        raise E3ContractError(
            'admission metadata cannot prove the PLE path without the model text config')
    return hf_config


def _ple_contract(scheduler):
    """Return ``(context_len, eos)``; ``(0, None)`` only with a declared empty path."""
    hf_config = _text_config(scheduler)
    layers = getattr(hf_config, 'ple_layer_ids', _MISSING)
    if layers is _MISSING:
        raise E3ContractError(
            'model text config does not declare ple_layer_ids; absence of the '
            'attribute is not evidence that the PLE path is absent')
    if not layers:
        return 0, None
    size = getattr(hf_config, 'ngram_size', None)
    eos = getattr(hf_config, 'eos_token_id', None)
    if type(size) is not int or size < 2:
        raise E3ContractError('PLE layers need an integer ngram_size >= 2')
    if type(eos) is not int or eos < 0:
        raise E3ContractError('PLE layers need a nonnegative integer eos_token_id')
    return size - 1, eos


_SLOT_TO_ATTR = {'new': _QUEUE_NEW, 'cow': _QUEUE_COW,
                 'boundary': _QUEUE_BOUNDARY}


class _ManagerQueues:
    """One manager's queue snapshot and suffix helpers."""

    __slots__ = ('manager', 'group_id', 'new', 'cow', 'boundary',
                 'new_prefix', 'cow_prefix', 'boundary_prefix')

    def __init__(self, manager, group_id):
        self.manager = manager
        self.group_id = group_id
        self.new = _list_attr(manager, _QUEUE_NEW, 'manager')
        self.cow = _list_attr(manager, _QUEUE_COW, 'manager')
        self.boundary = _list_attr(manager, _QUEUE_BOUNDARY, 'manager')
        self.new_prefix = tuple(self.new)
        self.cow_prefix = tuple(self.cow)
        self.boundary_prefix = tuple(self.boundary)

    def suffix(self, name):
        attr = _SLOT_TO_ATTR[name]
        original = getattr(self, name)
        prefix = getattr(self, name + '_prefix')
        current = getattr(self.manager, attr)
        if current is not original:
            raise E3ContractError(
                f'allocator replaced its {attr} list during admission')
        if len(current) < len(prefix):
            raise E3ContractError('allocator queue lost entries during admission')
        for before, after in zip(prefix, current):
            if not _same_item(before, after):
                raise E3ContractError(
                    'allocator queue prefix changed during admission; unrelated '
                    'stock-ordered work cannot be attributed')
        return list(current[len(prefix):])

    def truncate(self, name):
        original = getattr(self, name)
        prefix = getattr(self, name + '_prefix')
        del original[len(prefix):]

    def zero_ids(self, suffix):
        """Apply the manager's own drain semantics to exactly this suffix."""
        if not suffix:
            return ()
        original = self.manager.new_block_ids
        self.manager.new_block_ids = list(suffix)
        try:
            drained = self.manager.take_new_block_ids()
        finally:
            self.manager.new_block_ids = original
        if type(drained) is not list:
            raise E3ContractError('take_new_block_ids did not return a list')
        return tuple(drained)

    def cow_endpoints(self, suffix):
        """Apply the manager's own drain semantics to exactly this suffix."""
        if not suffix:
            return ()
        original = self.manager._pending_cow_copies
        self.manager._pending_cow_copies = list(suffix)
        try:
            drained = self.manager.take_pending_cow_copies()
        finally:
            self.manager._pending_cow_copies = original
        if type(drained) is not list or len(drained) != len(suffix):
            raise E3ContractError('take_pending_cow_copies changed this allocation')
        for before, after in zip(suffix, drained):
            if before is not after:
                raise E3ContractError('copy queue returned a different endpoint object')
        return tuple(drained)


class AllocationCapture:
    """Immutable-ish metadata plus the real endpoint objects it retains."""

    __slots__ = ('zero_ids', 'cow_pairs', 'retained_blocks', 'unpublished',
                 'ngram_context', 'evidence', 'orphan', 'error', '_released')

    def __init__(self, *, zero_ids, cow_pairs, retained_blocks, unpublished,
                 ngram_context, evidence, orphan=False, error=None):
        self.zero_ids = tuple(zero_ids)
        self.cow_pairs = tuple(cow_pairs)
        self.retained_blocks = tuple(retained_blocks)
        self.unpublished = tuple(unpublished)
        self.ngram_context = tuple(ngram_context)
        self.evidence = dict(evidence)
        self.orphan = bool(orphan)
        self.error = error
        self._released = False

    @property
    def details(self):
        return {
            'new_block_ids_to_zero': self.zero_ids,
            'cow_pairs': self.cow_pairs,
            'ngram_context': self.ngram_context,
            'unpublished_blocks': self.unpublished,
        }

    @property
    def released(self):
        return self._released

    def release_retained(self, scheduler):
        """Return the copy retentions once, through the stock deferred route.

        Callers may only invoke this after the controller's own commit/cancel
        authority accepted the chunk. It decrements allocator references; it
        never certifies that a GPU copy completed.
        """
        if self.orphan:
            # The failed allocation's copy pairs are still in the manager
            # queue, so the stock drain owns their references. Releasing here
            # as well would decrement them twice.
            raise E3ContractError(
                'an aborted allocation keeps its copy endpoints in the engine '
                'queue; only that queue may return them')
        if self._released:
            raise E3ContractError('retained copy endpoints were already released')
        blocks = list(self.retained_blocks)
        if not blocks:
            self._released = True
            return ()
        free = getattr(scheduler, '_free_cow_retained_blocks', None)
        if callable(free):
            # Stock releases a copy retention at the first processed step at or
            # after the step that runs the copy. The controller has already
            # observed every worker release, so the current fence is safe and
            # the pinned deferred-free route is preserved.
            fence = getattr(scheduler, 'sched_step_seq', 0)
            if type(fence) is not int or fence < 0:
                raise E3ContractError('scheduler step fence is invalid')
            free(blocks, fence)
        else:
            pool = getattr(getattr(scheduler, 'kv_cache_manager', None),
                           'block_pool', None)
            if pool is None or not callable(getattr(pool, 'free_blocks', None)):
                raise E3ContractError('no block pool accepts the retained endpoints')
            pool.free_blocks(blocks)
        self._released = True
        return tuple(block.block_id for block in blocks)


class NativeAdmissionMetadata:
    """The production ``_flash_s5_admission_metadata`` provider."""

    def __init__(self, scheduler):
        manager = getattr(scheduler, 'kv_cache_manager', None)
        coordinator = getattr(manager, 'coordinator', None)
        managers = getattr(coordinator, 'single_type_managers', None)
        if manager is None or coordinator is None or not managers:
            raise E3ContractError(
                'admission metadata needs the pinned KV manager and single-type groups')
        if getattr(manager, 'block_pool', None) is None:
            raise E3ContractError('admission metadata needs the block pool')
        self._scheduler = scheduler
        self._manager = manager
        self._queues = []
        for index, group in enumerate(managers):
            group_id = getattr(group, 'kv_cache_group_id', _MISSING)
            if group_id is _MISSING:
                group_id = index
            if type(group_id) is not int or group_id != index:
                raise E3ContractError('KV group ids must match allocator order')
            for name in _QUEUES:
                _list_attr(group, name, 'KV group manager')
            if not isinstance(getattr(group, 'req_to_blocks', None), dict):
                raise E3ContractError('KV group manager has no block rows')
            if not isinstance(getattr(group, 'num_cached_block', None), dict):
                raise E3ContractError('KV group manager has no publication frontier')
            if not callable(getattr(group, 'take_new_block_ids', None)):
                raise E3ContractError('KV group manager cannot drain zero ids')
            if not callable(getattr(group, 'take_pending_cow_copies', None)):
                raise E3ContractError('KV group manager cannot drain copy endpoints')
            if not callable(getattr(group, 'take_pending_boundary_state_offloads', None)):
                raise E3ContractError('KV group manager cannot drain boundary offers')
            if callable(getattr(group, 'take_host_cow_copies', None)):
                # HiSparseSourceManager keeps copies for its host coordinator and
                # its take_new_block_ids drops ids: neither is representable in
                # the four-field device admission record. Refuse before mutation.
                raise E3ContractError(
                    'host-tier HiSparse KV groups are not representable in the '
                    'admission metadata record; a host-copy/offload schema is required')
        self.contract_len, self.eos_token_id = _ple_contract(scheduler)
        needs_zeroing = getattr(scheduler, 'needs_kv_cache_zeroing', _MISSING)
        if type(needs_zeroing) is not bool:
            raise E3ContractError('scheduler does not declare needs_kv_cache_zeroing')
        self._scope = None
        self.captures = 0
        self.aborts = 0

    # -- installation -----------------------------------------------------
    def install(self):
        scheduler = self._scheduler
        if getattr(scheduler, '_flash_s5_admission_metadata', None) is not None:
            raise E3ContractError('scheduler already has an admission metadata provider')
        scheduler._flash_s5_admission_metadata = self
        return self

    # -- scoped capture ---------------------------------------------------
    def begin(self, scheduler, request):
        if scheduler is not self._scheduler:
            raise E3ContractError('admission metadata provider is installed elsewhere')
        if self._scope is not None:
            raise E3ContractError('admission metadata capture is already active')
        request_id = getattr(request, 'request_id', None)
        if not isinstance(request_id, str) or not request_id:
            raise E3ContractError('admission metadata needs a request id')
        self._scope = (request, tuple(
            _ManagerQueues(group, index)
            for index, group in enumerate(
                self._manager.coordinator.single_type_managers)))
        return self._scope

    def _take_scope(self, request=None):
        scope = self._scope
        if scope is None:
            raise E3ContractError('admission metadata capture was not begun')
        self._scope = None
        if request is not None and request is not scope[0]:
            raise E3ContractError('admission metadata belongs to another request incarnation')
        return scope

    def _unpublished(self, allocated_blocks, scope_queues, request_id, cow_pairs):
        """Allocation-owned publication obligations, including copy destinations.

        The pinned allocation result is the block set this admission owns;
        earlier row entries are another transaction's obligation (a partially
        filled tail block from a previous chunk is already recorded).  Block
        hash state is deliberately not consulted: ``"align"`` migrates a
        published hash onto the copy destination before the copy runs.
        """
        groups = getattr(allocated_blocks, 'blocks', None)
        if not isinstance(groups, tuple):
            raise E3ContractError(
                'allocation result does not expose per-group blocks')
        if len(groups) > len(scope_queues):
            raise E3ContractError(
                'allocation result has more KV groups than the allocator')
        cow_ids = {destination for _, destination in cow_pairs}
        owned = []
        for index, queues in enumerate(scope_queues):
            blocks = groups[index] if index < len(groups) else ()
            row = queues.manager.req_to_blocks.get(request_id, ())
            row_ids = {block.block_id for block in row
                       if not getattr(block, 'is_null', False)
                       and type(getattr(block, 'block_id', None)) is int}
            for block in blocks:
                if getattr(block, 'is_null', False):
                    continue
                block_id = _block_id(block, 'allocated block')
                if block_id not in row_ids and block_id not in cow_ids:
                    raise E3ContractError(
                        'allocated page is outside this request row and copy '
                        'destinations')
                owned.append(block_id)
        owned.extend(destination for _, destination in cow_pairs)
        return _unique(owned)

    def _row_unpublished(self, scope_queues, request_id, cow_pairs):
        """Conservative row-frontier journal for an aborted allocation.

        The allocation result does not exist after a failure, so the full
        non-null row suffix stands in for ownership; over-reporting is safe
        because the controller quarantines the request instead of reserving a
        frontier entry for it.
        """
        unpublished = []
        for queues in scope_queues:
            group = queues.manager
            row = group.req_to_blocks.get(request_id)
            if row is None:
                continue
            frontier = group.num_cached_block.get(request_id, 0)
            if type(frontier) is not int or frontier < 0 or frontier > len(row):
                raise E3ContractError('KV group publication frontier is invalid')
            for block in row[frontier:]:
                if getattr(block, 'is_null', False):
                    continue
                unpublished.append(_block_id(block, 'unpublished row block'))
        # Mamba "align" relocates the source hash onto the CoW destination and
        # leaves that destination out of the returned row. The copy's bytes are
        # a publication obligation even though the id is not in block_tables.
        for pair in cow_pairs:
            unpublished.append(pair[1])
        return _unique(unpublished)

    def _ngram_context(self, request, chunk_start):
        length = self.contract_len
        if length == 0:
            return (), 0
        tokens = getattr(request, 'all_token_ids', None)
        if tokens is None:
            tokens = getattr(request, 'prompt_token_ids', None)
        # Request.all_token_ids is the pinned read-only ConstantList (a
        # Sequence), not its mutable backing list. Never borrow prompt ids if
        # that live view is present but invalid, nor infer unavailable tokens.
        if (not isinstance(tokens, Sequence)
                or isinstance(tokens, (str, bytes, bytearray, memoryview))):
            raise E3ContractError('request tokens are not an integer token sequence')
        if len(tokens) < chunk_start:
            raise E3ContractError('request tokens do not cover the chunk start')
        begin = chunk_start - length
        if begin < 0:
            padding = -begin
            context = [self.eos_token_id] * padding + list(tokens[:chunk_start])
        else:
            padding = 0
            context = list(tokens[begin:chunk_start])
        if len(context) != length:
            raise E3ContractError('PLE context does not have its configured length')
        if any(type(token) is not int or token < 0 for token in context):
            raise E3ContractError('PLE context tokens must be nonnegative ints')
        return tuple(context), padding

    @staticmethod
    def _validate_cow(endpoints):
        pairs = []
        seen_destinations = set()
        retained = []
        for source, destination in endpoints:
            source_id = _block_id(source, 'copy source')
            destination_id = _block_id(destination, 'copy destination')
            if source_id == destination_id:
                raise E3ContractError('a copy pair must use two physical blocks')
            if destination_id in seen_destinations:
                raise E3ContractError('two copy pairs share a destination block')
            seen_destinations.add(destination_id)
            pairs.append((source_id, destination_id))
            retained.extend((source, destination))
        return tuple(pairs), tuple(retained)

    @staticmethod
    def _validate_boundary(scope_queues, suffixes, request_id, scheduler):
        entries = []
        for queues, suffix in zip(scope_queues, suffixes):
            for entry in suffix:
                if not isinstance(entry, tuple) or len(entry) != 4:
                    raise E3ContractError('boundary offload entry has the wrong shape')
                owner, group_id, block, tokens = entry
                if owner != request_id:
                    raise E3ContractError(
                        'boundary offload entry belongs to another request')
                if group_id != queues.group_id:
                    raise E3ContractError('boundary offload group does not match its manager')
                block_id = _block_id(block, 'boundary offload block')
                if type(tokens) is not int or tokens < 0:
                    raise E3ContractError('boundary offload token boundary is invalid')
                entries.append((group_id, block_id, tokens))
        if entries and getattr(scheduler, 'connector', None) is not None:
            raise E3ContractError(
                'Mamba "align" queued boundary-state offloads for a KV connector '
                '(MambaManager.allocate_new_blocks/cache_blocks); the four-field '
                'admission record cannot carry (group_id, block_id, '
                'boundary_tokens). A reviewed admission schema plus connector '
                'consumer is required')
        return tuple(entries)

    def capture(self, request, allocated_blocks, chunk_start, chunk_end):
        """Take this allocation's side-effect suffix; never steal stock work."""
        scope = self._take_scope(request)
        try:
            return self._capture(scope, request, allocated_blocks,
                                 chunk_start, chunk_end)
        except BaseException:
            # Nothing was truncated unless every validation passed. Keep the
            # scope so the controller's abort() journals the untouched suffix
            # (endpoints, zero ids, boundary offers) for the quarantined request.
            self._scope = scope
            raise

    def _capture(self, scope, request, allocated_blocks, chunk_start, chunk_end):
        request_id = request.request_id
        if allocated_blocks is None:
            raise E3ContractError('admission capture has no allocation result')
        if type(chunk_start) is not int or type(chunk_end) is not int:
            raise E3ContractError('admission capture needs integer token bounds')
        if not 0 <= chunk_start < chunk_end:
            raise E3ContractError('admission capture bounds are invalid')

        suffixes = [queues.suffix('new') for queues in scope[1]]
        cow_suffixes = [queues.suffix('cow') for queues in scope[1]]
        boundary_suffixes = [queues.suffix('boundary') for queues in scope[1]]

        zero_ids = []
        for queues, suffix in zip(scope[1], suffixes):
            zero_ids.extend(queues.zero_ids(suffix))
        endpoint_groups = [queues.cow_endpoints(suffix)
                           for queues, suffix in zip(scope[1], cow_suffixes)]
        endpoints = [pair for group in endpoint_groups for pair in group]
        cow_pairs, retained = self._validate_cow(endpoints)
        boundary = self._validate_boundary(scope[1], boundary_suffixes,
                                           request_id, self._scheduler)
        unpublished = self._unpublished(allocated_blocks, scope[1], request_id,
                                        cow_pairs)
        context, padding = self._ngram_context(request, chunk_start)

        if any(type(block_id) is not int or block_id <= 0 for block_id in zero_ids):
            raise E3ContractError('zeroing ids must be positive physical block ids')
        if len(set(zero_ids)) != len(zero_ids):
            raise E3ContractError('this allocation recorded a duplicate zeroing id')
        rows = self._manager.get_block_ids(request_id)
        row_ids = {block for row in rows for block in row if type(block) is int and block > 0}
        if not set(zero_ids) <= row_ids:
            raise E3ContractError('zeroing ids are outside this allocation rows')

        needs_zeroing = self._scheduler.needs_kv_cache_zeroing
        evidence = {
            'needs_kv_cache_zeroing': needs_zeroing,
            'records_new_block_ids': tuple(
                bool(getattr(queues.manager, 'records_new_block_ids', False))
                for queues in scope[1]),
            'drained_zero_ids': len(zero_ids),
            'recording_groups': tuple(
                queues.group_id for queues in scope[1]
                if getattr(queues.manager, 'records_new_block_ids', False)),
            'skipped_zero_ids': (),
            'boundary_offloads': boundary,
            'boundary_consumer': 'stock-connector-less drain',
            'cow_pairs': len(cow_pairs),
            'unpublished_blocks': len(unpublished),
            'unpublished_source': 'allocation result + copy destinations',
            'ngram_context_length': self.contract_len,
            'ngram_padding': padding,
        }
        if not needs_zeroing:
            # Stock drains and discards these ids when zeroing is disabled.
            evidence['discarded_zero_ids'] = len(zero_ids)
            zero_ids = []
        else:
            skip = getattr(self._scheduler, '_skip_zero_block_ids', None)
            if skip is not None:
                if not isinstance(skip, set):
                    raise E3ContractError('scheduler zero-skip ids are not a set')
                skipped = tuple(block_id for block_id in zero_ids if block_id in skip)
                if skipped:
                    zero_ids = [block_id for block_id in zero_ids
                                if block_id not in skip]
                    # Stock clears exactly these entries during its own drain;
                    # remove only ids this allocation consumed, leaving entries
                    # that belong to unrelated work in stock order.
                    for block_id in skipped:
                        skip.discard(block_id)
                    evidence['skipped_zero_ids'] = skipped
        if not needs_zeroing:
            evidence['zeroing_reason'] = 'needs_kv_cache_zeroing is false'
        elif zero_ids:
            evidence['zeroing_reason'] = 'drained new attention block ids'
        else:
            evidence['zeroing_reason'] = (
                'allocation recorded no zero work (cached hits and/or non-zeroed groups)')

        # Commit the extraction only after every validation succeeded.
        for queues, suffix in zip(scope[1], suffixes):
            queues.truncate('new')
        for queues, suffix in zip(scope[1], cow_suffixes):
            queues.truncate('cow')
        # Boundary offers stay in the manager queue so the pinned end-of-schedule
        # drain observes them exactly once, in stock order. With no connector the
        # stock path has no device consumer; the retained CoW destination keeps
        # the state block alive until the controller retires it.
        self.captures += 1
        return AllocationCapture(
            zero_ids=zero_ids, cow_pairs=cow_pairs,
            retained_blocks=retained, unpublished=unpublished,
            ngram_context=context, evidence=evidence)

    def abort(self):
        """Record a failed allocation's side effects without consuming them.

        Nothing is removed from the manager queues: an allocation that raised
        (or returned ``None`` after appending work) must stay visible to the
        engine and any later free. Returns ``None`` when the failed allocation
        appended nothing. This method is total: a journal failure returns an
        error record so the controller can still quarantine the request.
        """
        error = None
        zero_ids = ()
        cow_pairs = ()
        retained = ()
        boundary = ()
        unpublished = ()
        context = ()
        padding = 0
        raw_boundary = ()
        try:
            scope = self._take_scope()
            request = scope[0]
            request_id = request.request_id
            suffixes = [queues.suffix('new') for queues in scope[1]]
            cow_suffixes = [queues.suffix('cow') for queues in scope[1]]
            boundary_suffixes = [queues.suffix('boundary') for queues in scope[1]]
            zero = []
            for queues, suffix in zip(scope[1], suffixes):
                zero.extend(queues.zero_ids(suffix))
            zero_ids = tuple(zero)
            endpoints = [
                pair for queues, suffix in zip(scope[1], cow_suffixes)
                for pair in queues.cow_endpoints(suffix)]
            # Journal the raw endpoints before validation so a refused pair
            # (null/duplicate) is still recorded rather than dropped.
            retained = tuple(block for pair in endpoints for block in pair)
            cow_pairs = tuple(
                (getattr(source, 'block_id', None),
                 getattr(destination, 'block_id', None))
                for source, destination in endpoints)
            raw_boundary = tuple(
                entry for suffix in boundary_suffixes for entry in suffix)
            cow_pairs, retained = self._validate_cow(endpoints)
            boundary = self._validate_boundary(scope[1], boundary_suffixes,
                                               request_id, self._scheduler)
            unpublished = self._row_unpublished(scope[1], request_id, cow_pairs)
            context, padding = self._ngram_context(
                request, getattr(request, 'num_computed_tokens', 0))
            if not (any(suffixes) or any(cow_suffixes)
                    or any(boundary_suffixes)):
                self.aborts += 1
                return None
        except BaseException as exc:
            # The provider still returns the journal; the controller removes the
            # request and refuses to retry this allocation.
            error = f'{type(exc).__name__}: {exc}'
            self.aborts += 1
            return AllocationCapture(
                zero_ids=zero_ids, cow_pairs=cow_pairs, retained_blocks=retained,
                unpublished=unpublished, ngram_context=context,
                evidence={'abort_error': error,
                          'boundary_offloads_raw': raw_boundary},
                orphan=True, error=error)
        self.aborts += 1
        return AllocationCapture(
            zero_ids=zero_ids, cow_pairs=cow_pairs, retained_blocks=retained,
            unpublished=unpublished, ngram_context=context,
            evidence={'orphan': True, 'boundary_offloads': boundary,
                      'ngram_padding': padding,
                      'unpublished_source': 'row frontier + copy destinations '
                                            '(aborted allocation)'},
            orphan=True)

    def __call__(self, request, allocated_blocks, chunk_start, chunk_end):
        """Legacy four-field interface; still requires an explicit begin scope."""
        return self.capture(request, allocated_blocks, chunk_start, chunk_end).details


def install_admission_metadata(scheduler):
    """Install the production provider once at enabled engine startup."""
    return NativeAdmissionMetadata(scheduler).install()


__all__ = [
    'AllocationCapture', 'BLOCK_POOL_SHA256', 'KV_CACHE_MANAGER_SHA256',
    'MODEL_STATE_SHA256', 'NativeAdmissionMetadata', 'SCHEDULER_SHA256',
    'SINGLE_TYPE_SHA256', 'install_admission_metadata', 'verify_sources',
]
