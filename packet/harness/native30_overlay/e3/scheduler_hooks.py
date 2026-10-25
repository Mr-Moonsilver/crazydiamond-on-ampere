"""C: scheduler/block-pool leases for asynchronous prefill publication.

E2 protects its serial fences; E3 must not simply delete them. This module adds
an explicit lease guard on top of the reviewed E2 scheduler and block pool:

* A parked prefill transaction leases the blocks it may still write/read.
  ``free_blocks`` defers those blocks instead of returning them to the pool.
  Refcounts are decremented exactly once, at the real free, so
  ``is_block_writable`` and CoW counts keep their stock meaning.
* A lease never allocates and never copies. ``lease_caused_cow`` must stay 0.
* Prefix sharing can only see the :class:`~async_handoff.CommittedFrontier`
  (committed blocks). An uncommitted lease is refused by ``before_touch`` and
  truncated from a lookup so the caller recomputes.
* Release requires the consumer completion proof C records. Until then the
  blocks stay out of the free queue and out of prefix-cache publication.

The transforms target the E2-composed ``core/scheduler.py`` and
``core/block_pool.py`` (mirror + telemetry hooks already applied). Passing the
unmodified upstream scheduler is refused. No torch/vLLM import is needed here.
"""
import hashlib

from e3_types import (
    CompletionKind,
    CompletionProof,
    E3ContractError,
    E3Pending,
    OperationTicket,
)

from async_handoff import LeaseCategory


SCHEDULER_PATH = 'v1/core/sched/scheduler.py'
BLOCK_POOL_PATH = 'v1/core/block_pool.py'

# Pinned reference digests (unchanged inputs) and the exact E2-composed outputs.
SCHEDULER_PINNED_SHA256 = '4d797e5ab6c5abaa06229dd88035ceede7562f1dbe054bdf7ad59be659c04c87'
SCHEDULER_E2_SHA256 = 'c767a87918d48a4dced38d42367a958e5c9f81b18737ba385f9d88b7019cf6eb'
BLOCK_POOL_PINNED_SHA256 = '7f781378710e616624f87d95ccc7b245e1e741c85f3751268e672c2a262038e8'
BLOCK_POOL_E2_SHA256 = '3f2d030c0ff7217ed796080ad2588c07692b8b0970ab8018d7fb748c06d5dfb1'

_IMPORT = '''import vllm.v1.flash_e3.scheduler_hooks as _flash_s5_hooks
'''


def _sha(source):
    return hashlib.sha256(source.encode()).hexdigest()


def _replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('source anchor changed or duplicated: ' +
                         old[:100].replace('\n', '\\n'))
    return text.replace(old, new, 1)


def _require_e2(source, digest, path, marker):
    got = _sha(source)
    if got != digest:
        raise ValueError(f'unreviewed E2-composed source {path}: {got}')
    if marker not in source:
        raise ValueError(f'{path} is not the reviewed E2 composition')
    if '_flash_s5' in source:
        raise ValueError(f'{path} already carries E3 scheduler hooks')
    return source


_SCHEDULER_FORWARD = (
    ('logger = init_logger(__name__)\n', 'logger = init_logger(__name__)\n\n' + _IMPORT),
    ('''        if _e2_journal is not None:
            scheduler_output.flash_e2_frame = _e2_journal.frame(
                scheduler_output, self.kv_cache_manager.get_block_ids)
        return scheduler_output
''', '''        if _e2_journal is not None:
            scheduler_output.flash_e2_frame = _e2_journal.frame(
                scheduler_output, self.kv_cache_manager.get_block_ids)
        _flash_s5_hooks.after_schedule(self, scheduler_output)
        return scheduler_output
'''),
    ('''    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        if self._request_blocks_can_be_freed(request):
''', '''    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        if _flash_s5_hooks.defer_request_free(self, request):
            return
        if self._request_blocks_can_be_freed(request):
'''),
    ('''        if not self.defer_block_free or fence_seq <= self.processed_step_seq:
            self.kv_cache_manager.block_pool.free_blocks(blocks)
            return
''', '''        _flash_s5_hooks.note_cow_retained(self, blocks)
        if not self.defer_block_free or fence_seq <= self.processed_step_seq:
            self.kv_cache_manager.block_pool.free_blocks(blocks)
            return
'''),
    ('''        while self.deferred_frees:
            fence, _ = self.deferred_frees[0]
            if fence > self.processed_step_seq:
                break
''', '''        while self.deferred_frees:
            fence, blocks = self.deferred_frees[0]
            if _flash_s5_hooks.blocked_deferred(self, blocks):
                break
            if fence > self.processed_step_seq:
                break
'''),
    ('''    def shutdown(self) -> None:
        logger.debug_once("[shutdown] Scheduler: start")
''', '''    def shutdown(self) -> None:
        _flash_s5_hooks.before_scheduler_shutdown(self)
        logger.debug_once("[shutdown] Scheduler: start")
'''),
)

_SCHEDULER_REVERSE = (
    # (new, old) pairs in forward order; revert walks in reverse.
    ('''        while self.deferred_frees:
            fence, blocks = self.deferred_frees[0]
            if _flash_s5_hooks.blocked_deferred(self, blocks):
                break
            if fence > self.processed_step_seq:
                break
''', '''        while self.deferred_frees:
            fence, _ = self.deferred_frees[0]
            if fence > self.processed_step_seq:
                break
'''),
    ('''        _flash_s5_hooks.note_cow_retained(self, blocks)
        if not self.defer_block_free or fence_seq <= self.processed_step_seq:
''', '''        if not self.defer_block_free or fence_seq <= self.processed_step_seq:
'''),
    ('''        if _flash_s5_hooks.defer_request_free(self, request):
            return
        if self._request_blocks_can_be_freed(request):
''', '''        if self._request_blocks_can_be_freed(request):
'''),
    ('''        _flash_s5_hooks.after_schedule(self, scheduler_output)
        return scheduler_output
''', '''        return scheduler_output
'''),
    ('''    def shutdown(self) -> None:
        _flash_s5_hooks.before_scheduler_shutdown(self)
''', '''    def shutdown(self) -> None:
'''),
    ('logger = init_logger(__name__)\n\n' + _IMPORT,
     'logger = init_logger(__name__)\n'),
)

_BLOCK_POOL_FORWARD = (
    ('logger = init_logger(__name__)\n', 'logger = init_logger(__name__)\n\n' + _IMPORT),
    ('''        if self._flash_e2_journal is not None:
            self._flash_e2_journal.allocated(b.block_id for b in ret)
        return ret
''', '''        if self._flash_e2_journal is not None:
            self._flash_e2_journal.allocated(b.block_id for b in ret)
        _flash_s5_hooks.note_allocated(self, ret)
        return ret
'''),
    ('''        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
''', '''        _flash_s5_hooks.before_touch(self, blocks)
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
'''),
    ('''        # Identify blocks with hash (LRU cache) and without it (never match APC)
        blocks_to_evict_last = []
''', '''        ordered_blocks = _flash_s5_hooks.hold_leased_blocks(self, ordered_blocks)
        # Identify blocks with hash (LRU cache) and without it (never match APC)
        blocks_to_evict_last = []
'''),
)

_BLOCK_POOL_REVERSE = (
    ('''        ordered_blocks = _flash_s5_hooks.hold_leased_blocks(self, ordered_blocks)
        # Identify blocks with hash (LRU cache) and without it (never match APC)
''', '''        # Identify blocks with hash (LRU cache) and without it (never match APC)
'''),
    ('''        _flash_s5_hooks.before_touch(self, blocks)
        for block in blocks:
''', '''        for block in blocks:
'''),
    ('''        _flash_s5_hooks.note_allocated(self, ret)
        return ret
''', '''        return ret
'''),
    ('logger = init_logger(__name__)\n\n' + _IMPORT,
     'logger = init_logger(__name__)\n'),
)


def generate_scheduler(source):
    source = _require_e2(source, SCHEDULER_E2_SHA256, SCHEDULER_PATH, '_flash_e2_journal')
    text = source
    for old, new in _SCHEDULER_FORWARD:
        text = _replace(text, old, new)
    import ast
    ast.parse(text)
    return text


def revert_scheduler(source, expected=None):
    text = source
    for new, old in reversed(_SCHEDULER_REVERSE):
        text = _replace(text, new, old)
    if expected is not None and text != expected:
        raise ValueError('scheduler revert did not recover the supplied source')
    if expected is None and _sha(text) != SCHEDULER_E2_SHA256:
        raise ValueError('scheduler revert did not recover the E2 composition')
    return text


def generate_block_pool(source):
    source = _require_e2(source, BLOCK_POOL_E2_SHA256, BLOCK_POOL_PATH, '_flash_e2_journal')
    text = source
    for old, new in _BLOCK_POOL_FORWARD:
        text = _replace(text, old, new)
    import ast
    ast.parse(text)
    return text


def revert_block_pool(source, expected=None):
    text = source
    for new, old in reversed(_BLOCK_POOL_REVERSE):
        text = _replace(text, new, old)
    if expected is not None and text != expected:
        raise ValueError('block pool revert did not recover the supplied source')
    if expected is None and _sha(text) != BLOCK_POOL_E2_SHA256:
        raise ValueError('block pool revert did not recover the E2 composition')
    return text


# --- runtime guard ----------------------------------------------------------

def _flatten_ids(value):
    if value is None:
        return ()
    if type(value) is int:
        return (value,)
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_flatten_ids(item))
        return tuple(result)
    if isinstance(value, (list, tuple, set, frozenset)):
        result = []
        for item in value:
            result.extend(_flatten_ids(item))
        return tuple(result)
    if isinstance(value, str):
        return ()
    result = []
    for name in ('block_id', 'block_ids', 'new_block_ids', 'dst_block_ids',
                 'src_block_ids'):
        if hasattr(value, name):
            result.extend(_flatten_ids(getattr(value, name)))
    return tuple(result)


class SchedulerLeases:
    """Per-block leases, deferred free and CoW accounting.

    Installed on the scheduler and block pool by C's startup composition. Every
    hook is inert when no guard is installed, which keeps the default-off
    serial path byte-identical.
    """

    def __init__(self):
        self._held = {}
        self._pool = None
        self._scheduler = None
        self._bypass = False
        self.deferred_requests = {}
        self.cow_requested = 0
        self.cow_completed = 0
        self.lease_caused_cow = 0
        self.held_releases = 0
        self.pending_free = {}
        self._free_order = 0
        self.rejected_late = 0

    # -- attach -----------------------------------------------------------
    def attach(self, scheduler, block_pool):
        if getattr(scheduler, '_flash_s5_leases', None) is not None:
            raise E3ContractError('scheduler already has E3 leases')
        if getattr(block_pool, '_flash_s5_leases', None) is not None:
            raise E3ContractError('block pool already has E3 leases')
        scheduler._flash_s5_leases = self
        block_pool._flash_s5_leases = self
        self._scheduler, self._pool = scheduler, block_pool
        return self

    # -- lease lifecycle --------------------------------------------------
    def lease_blocks(self, ticket, category, block_ids, *, committed=False):
        if type(ticket) is not OperationTicket:
            raise E3ContractError('lease needs an OperationTicket')
        if type(category) is not LeaseCategory:
            raise E3ContractError('lease category is invalid')
        if category not in (LeaseCategory.SOURCE_HISTORY, LeaseCategory.ARENA):
            raise E3ContractError('block leases are source-history or arena leases')
        ids = _flatten_ids(block_ids)
        if not ids or any(type(b) is not int or b <= 0 for b in ids):
            raise E3ContractError('lease block ids must be positive, non-null ints')
        if len(set(ids)) != len(ids):
            raise E3ContractError('duplicate block in a lease')
        if any(e['ticket'] == ticket and e['category'] is category
               for entries in self._held.values() for e in entries):
            raise E3ContractError('duplicate lease ticket/category')
        entry = {'ticket': ticket, 'category': category, 'blocks': tuple(ids),
                 'committed': bool(committed), 'released': False}
        for block in ids:
            self._held.setdefault(block, []).append(entry)
        return entry

    def publish(self, ticket):
        entries = [e for item in self._held.values() for e in item
                   if e['ticket'] == ticket]
        if not entries:
            raise E3ContractError('no lease to publish')
        for entry in entries:
            entry['committed'] = True
        return ticket

    def held_blocks(self, ticket=None):
        return tuple(sorted({
            block for block, entries in self._held.items()
            for entry in entries
            if not entry['released'] and (ticket is None or entry['ticket'] == ticket)
        }))

    def _held_by(self, ticket):
        return set(self.held_blocks(ticket))

    def _held_by_transaction(self, transaction):
        if transaction is None:
            return set()
        return {block for block, entries in self._held.items()
                for entry in entries
                if not entry['released']
                and entry['ticket'].transaction == transaction}

    def _committed(self):
        return {block for block, entries in self._held.items()
                if any(e['committed'] and not e['released'] for e in entries)}

    def release(self, ticket, category, proof):
        """Release a lease only with its consumer completion proof."""
        if type(proof) is not CompletionProof:
            raise E3ContractError('lease release needs a CompletionProof')
        if proof.ticket != ticket:
            raise E3ContractError('release proof belongs to another operation')
        allowed = {
            LeaseCategory.SOURCE_HISTORY: (CompletionKind.COPY_EVENT,),
            LeaseCategory.ARENA: (CompletionKind.GPU_EVENT,),
        }
        if proof.kind not in allowed[category]:
            raise E3ContractError('release proof kind does not consume this lease')
        entries = [e for items in list(self._held.values()) for e in items
                   if e['ticket'] == ticket and e['category'] is category and not e['released']]
        if not entries:
            self.rejected_late += 1
            raise E3ContractError('release proof has no outstanding lease')
        released = set()
        for entry in entries:
            entry['released'] = True
            released.update(entry['blocks'])
        for block in list(self._held):
            self._held[block] = [e for e in self._held[block] if not e['released']]
            if not self._held[block]:
                del self._held[block]
        self.held_releases += 1
        self._free_released(tuple(sorted(released)))
        self._retry_requests()
        return tuple(sorted(released))

    def _free_released(self, blocks):
        pool = self._pool
        if pool is None or not blocks:
            return
        pending = []
        for block_id in blocks:
            if block_id in self._held:
                continue
            item = self.pending_free.get(block_id)
            if item is not None:
                pending.append((block_id, item))
        if not pending:
            return
        pending.sort(key=lambda item: item[1][2])
        free = []
        for _, (block, count, _) in pending:
            free.extend([block] * count)
        self._bypass = True
        try:
            pool.free_blocks(free)
            for block_id, _ in pending:
                del self.pending_free[block_id]
        finally:
            self._bypass = False

    # -- block-pool hooks -------------------------------------------------
    def hold_leased_blocks(self, ordered_blocks):
        blocks = tuple(ordered_blocks)
        if self._bypass:
            return blocks
        free_now, held = [], []
        for block in blocks:
            block_id = getattr(block, 'block_id', None)
            if getattr(block, 'is_null', False):
                free_now.append(block)
                continue
            if block_id in self._held:
                held.append(block)
            else:
                free_now.append(block)
        if held:
            for block in held:
                block_id = getattr(block, 'block_id', None)
                item = self.pending_free.get(block_id)
                if item is None:
                    self._free_order += 1
                    self.pending_free[block_id] = [block, 1, self._free_order]
                else:
                    item[1] += 1
            if self._scheduler is not None:
                self._scheduler._flash_s5_deferred = len(self.pending_free)
        return tuple(free_now)

    def block_held(self, block):
        return getattr(block, 'block_id', None) in self._held

    def before_touch(self, blocks):
        for block in blocks:
            if not self.block_held(block):
                continue
            block_id = getattr(block, 'block_id', None)
            committed = all(e['committed'] and not e['released']
                            for e in self._held.get(block_id, ()))
            if not committed:
                raise E3Pending('an uncommitted prefill block cannot be shared')

    def note_allocated(self, blocks):
        for block in blocks:
            if self.block_held(block):
                raise E3ContractError('allocator handed out a leased block')

    # -- scheduler hooks --------------------------------------------------
    def defer_request_free(self, scheduler, request):
        request_id = getattr(request, 'request_id', None)
        if request_id is None:
            raise E3ContractError('deferred free needs a request id')
        manager = getattr(scheduler, 'kv_cache_manager', None)
        getter = getattr(manager, 'get_block_ids', None)
        if getter is None:
            return False
        blocks = _flatten_ids(getter(request_id))
        if not any(block in self._held for block in blocks):
            return False
        self.deferred_requests[request_id] = (request, blocks)
        return True

    def _retry_requests(self):
        for request_id, (request, blocks) in tuple(self.deferred_requests.items()):
            if any(block in self._held for block in blocks):
                continue
            # The stock scheduler may defer further on its own RPC fence.
            # Keep our record if that call raises.
            self._scheduler._free_request_blocks(request)
            del self.deferred_requests[request_id]

    def blocked_deferred(self, blocks):
        return any(getattr(block, 'block_id', None) in self._held
                   for block in blocks)

    def mutation_guard(self, scheduler_output, transaction):
        """Validate block mutations against outstanding leases.

        Only block fields are inspected; no payload or token text is read.
        Destination blocks may not belong to any *other* outstanding lease, and
        a copy may only read blocks this transaction already holds or blocks
        that are already committed. The guard never creates a CoW or changes a
        refcount.
        """
        destinations = _flatten_ids(getattr(scheduler_output, 'block_table_updates', ()))
        destinations += _flatten_ids(getattr(scheduler_output, 'new_block_ids_to_zero', ()))
        copies = getattr(scheduler_output, 'kv_cache_block_copies', ()) or ()
        sources = ()
        for copy in copies:
            destination = getattr(copy, 'dst_block_id', None)
            source = getattr(copy, 'src_block_id', None)
            if type(destination) is int:
                destinations += (destination,)
            if type(source) is int:
                sources += (source,)
        cached = getattr(scheduler_output, 'scheduled_cached_reqs', None)
        destinations += _flatten_ids(getattr(cached, 'new_block_ids', ()))
        mine = self._held_by_transaction(transaction)
        # Even the same transaction cannot overwrite a still-consumed page.
        if set(destinations) & set(self._held):
            return False
        if sources and not set(sources) <= (mine | self._committed()):
            return False
        if copies:
            self.cow_requested += len(copies)
            # The copy itself is the scheduler's, not a lease artifact.
            # Submission is not proof that the scheduled copies completed.
        return True

    def note_cow_retained(self, blocks):
        # Retention does not complete a copy; matched accounting is checked at
        # the drain boundary by LifetimeLedger/assert semantics.
        return tuple(blocks)

    def shutdown(self):
        if self._held or self.pending_free or self.deferred_requests:
            return dict(clean=False, held=len(self._held),
                        pending_free=len(self.pending_free), deferred_requests=len(self.deferred_requests))
        return dict(clean=True, held=0, pending_free=0)


# --- generated-source hook functions ---------------------------------------

def _guard(owner):
    guard = getattr(owner, '_flash_s5_leases', None)
    if guard is not None and not isinstance(guard, SchedulerLeases):
        raise E3ContractError('owner E3 lease guard is invalid')
    return guard


def after_schedule(scheduler, scheduler_output):
    guard = _guard(scheduler)
    if guard is None:
        return None
    dispatcher = getattr(scheduler, '_flash_s5_handoff', None)
    lease = getattr(dispatcher, 'active', None)
    transaction = getattr(lease, 'transaction', None) if lease is not None else None
    if not guard.mutation_guard(scheduler_output, transaction):
        raise E3Pending('scheduled mutation conflicts with an outstanding E3 lease')
    return True


def defer_request_free(scheduler, request):
    guard = _guard(scheduler)
    return False if guard is None else guard.defer_request_free(scheduler, request)


def note_cow_retained(scheduler, blocks):
    guard = _guard(scheduler)
    return None if guard is None else guard.note_cow_retained(blocks)


def blocked_deferred(scheduler, blocks):
    guard = _guard(scheduler)
    return False if guard is None else guard.blocked_deferred(blocks)


def before_scheduler_shutdown(scheduler):
    guard = _guard(scheduler)
    if guard is None:
        return None
    report = guard.shutdown()
    if not report['clean']:
        raise E3ContractError('scheduler shutdown with outstanding E3 leases')
    return report


def note_allocated(block_pool, blocks):
    guard = _guard(block_pool)
    if guard is not None:
        guard.note_allocated(blocks)


def before_touch(block_pool, blocks):
    guard = _guard(block_pool)
    if guard is not None:
        guard.before_touch(blocks)


def block_held(block_pool, block):
    guard = _guard(block_pool)
    return False if guard is None else guard.block_held(block)


def hold_leased_blocks(block_pool, ordered_blocks):
    guard = _guard(block_pool)
    return tuple(ordered_blocks) if guard is None else guard.hold_leased_blocks(ordered_blocks)


__all__ = [
    'BLOCK_POOL_E2_SHA256', 'BLOCK_POOL_PATH', 'BLOCK_POOL_PINNED_SHA256',
    'SCHEDULER_E2_SHA256', 'SCHEDULER_PATH', 'SCHEDULER_PINNED_SHA256',
    'SchedulerLeases', 'after_schedule', 'before_scheduler_shutdown',
    'before_touch', 'block_held', 'blocked_deferred', 'defer_request_free',
    'generate_block_pool', 'generate_scheduler', 'hold_leased_blocks',
    'note_allocated', 'note_cow_retained', 'revert_block_pool',
    'revert_scheduler',
]
