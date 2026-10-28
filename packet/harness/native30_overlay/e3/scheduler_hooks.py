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
    MAX_ACTIVE_PREFILL,
    CompletionKind,
    CompletionProof,
    E3ContractError,
    E3Pending,
    LaneId,
    LaneRole,
    OperationTicket,
    TransactionId,
)

from async_handoff import LeaseCategory

try:  # The generated package must import before wire_v1 is registered (owner).
    from wire_v1 import AdmissionRecord, WorkerCompletionReport
except ImportError:  # pragma: no cover - package build without the new module
    AdmissionRecord = None
    WorkerCompletionReport = None


def _wire_admission():
    if AdmissionRecord is None:
        raise E3ContractError('wire-v1 is not packaged; E3 admission cannot run')
    return AdmissionRecord


def _wire_report():
    if WorkerCompletionReport is None:
        raise E3ContractError('wire-v1 is not packaged; E3 reporting cannot run')
    return WorkerCompletionReport


SCHEDULER_PATH = 'v1/core/sched/scheduler.py'
BLOCK_POOL_PATH = 'v1/core/block_pool.py'

# Native admission is an explicit, EngineCore-owned phase intent -- never an
# environment/config switch.  Stock full-route preparation, W1 decode-only
# preparation (on either route) and the split decode-only phase keep it off;
# only the matched split prefill solo, the row1 native qualification and the
# W2 overlap phase arm it.  The driver is the sole owner of the typed record;
# this module only reads the validated query at the policy seam.
ADMISSION_PHASES = ('full-prefill-solo', 'split-prefill-solo',
                    'split-w1-decode', 'row1-native', 'w2-overlap')
NATIVE_ADMISSION_PHASES = ('split-prefill-solo', 'row1-native', 'w2-overlap')

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
    # Pre-schedule admission runs before any request mutation in the step.
    ('''        self.kv_cache_manager.new_step_starts()
''', '''        self.kv_cache_manager.new_step_starts()
        _flash_s5_hooks.flash_s5_admit(self)
'''),
    # A parked chunk holds a model-runner slot until it retires.
    ('''                num_running = len(self.running) + self.num_waiting_for_streaming_input
                if num_running >= self.max_num_running_reqs:
''', '''                num_running = (len(self.running)
                               + self.num_waiting_for_streaming_input
                               + _flash_s5_hooks.parked_count(self))
                if num_running >= self.max_num_running_reqs:
'''),
    ('''        return num_waiting + len(self.running)
''', '''        return num_waiting + len(self.running) + _flash_s5_hooks.parked_count(self)
'''),
)

_SCHEDULER_REVERSE = (
    # (new, old) pairs in forward order; revert walks in reverse.
    ('''        return num_waiting + len(self.running) + _flash_s5_hooks.parked_count(self)
''', '''        return num_waiting + len(self.running)
'''),
    ('''                num_running = (len(self.running)
                               + self.num_waiting_for_streaming_input
                               + _flash_s5_hooks.parked_count(self))
''', '''                num_running = len(self.running) + self.num_waiting_for_streaming_input
'''),
    ('''        self.kv_cache_manager.new_step_starts()
        _flash_s5_hooks.flash_s5_admit(self)
''', '''        self.kv_cache_manager.new_step_starts()
'''),
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
        entry = {'ticket': ticket, 'transaction': ticket.transaction,
                 'category': category, 'blocks': tuple(ids), 'objects': (),
                 'committed': bool(committed), 'released': False, 'reservation': False}
        for block in ids:
            self._held.setdefault(block, []).append(entry)
        return entry

    def reserve_blocks(self, transaction, category, block_ids, *, blocks=None):
        """Pre-lease an allocation whose operation identity is not bound yet.

        Admission happens before any worker lane is opened. The blocks are held
        from that moment (deferred free, no sharing) and the reservation is
        bound to the real :class:`OperationTicket` by :meth:`bind`. Nothing here
        allocates or copies.

        ``blocks`` are the allocator's real block objects for those ids. They
        are retained with the reservation so the hold can later prove the same
        physical incarnation is still allocated; a CPU mirror of the ids alone
        is not a scheduler pin.
        """
        if type(transaction) is not TransactionId:
            raise E3ContractError('reservation needs a TransactionId')
        if type(category) is not LeaseCategory:
            raise E3ContractError('lease category is invalid')
        if category not in (LeaseCategory.SOURCE_HISTORY, LeaseCategory.ARENA):
            raise E3ContractError('block leases are source-history or arena leases')
        ids = _flatten_ids(block_ids)
        if not ids or any(type(b) is not int or b <= 0 for b in ids):
            raise E3ContractError('reservation block ids must be positive ints')
        if len(set(ids)) != len(ids):
            raise E3ContractError('duplicate block in a reservation')
        objects = ()
        if blocks is not None:
            objects = tuple(blocks)
            if len(objects) != len(ids):
                raise E3ContractError('reservation objects must cover every block id')
            if tuple(getattr(block, 'block_id', None) for block in objects) != tuple(ids):
                raise E3ContractError('reservation objects differ from the block ids')
        for entries in self._held.values():
            for item in entries:
                if (item['transaction'] == transaction
                        and item['category'] is category and not item['released']):
                    raise E3ContractError('duplicate reservation for a transaction')
        entry = {'ticket': None, 'transaction': transaction, 'category': category,
                 'blocks': tuple(ids), 'objects': objects, 'committed': False,
                 'released': False, 'reservation': True}
        for block in ids:
            self._held.setdefault(block, []).append(entry)
        return entry

    def bind(self, ticket, category=LeaseCategory.ARENA):
        """Attach the accepted operation identity to a pre-lease."""
        if type(ticket) is not OperationTicket:
            raise E3ContractError('bind needs an OperationTicket')
        entries = [e for item in self._held.values() for e in item
                   if e['transaction'] == ticket.transaction
                   and e['category'] is category and e['reservation']
                   and not e['released']]
        seen = set()
        for entry in entries:
            if id(entry) in seen:
                continue
            seen.add(id(entry))
            entry['ticket'] = ticket
            entry['reservation'] = False
        if not entries:
            raise E3ContractError('no reservation to bind')
        return ticket

    def cancel_reservation(self, transaction, category=LeaseCategory.ARENA):
        """Free a pre-lease only when no operation was ever submitted for it.

        This is not a completion shortcut: the caller must pass the driver's
        assertion that zero operations were accepted. Once any operation was
        accepted, :meth:`release` with its real proof is the only exit.
        """
        if type(transaction) is not TransactionId:
            raise E3ContractError('cancel_reservation needs a TransactionId')
        freed = set()
        for block, entries in list(self._held.items()):
            for entry in entries:
                if (entry['transaction'] == transaction
                        and entry['category'] is category
                        and entry['reservation'] and not entry['released']):
                    entry['released'] = True
                    freed.update(entry['blocks'])
            self._held[block] = [e for e in entries if not e['released']]
            if not self._held[block]:
                del self._held[block]
        if not freed:
            raise E3ContractError('no outstanding reservation to cancel')
        self._free_released(tuple(sorted(freed)))
        self._retry_requests()
        return tuple(sorted(freed))

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

    def held_transaction(self, transaction, category=None):
        """Block ids still unreleased for ``transaction`` (optionally one category)."""
        if type(transaction) is not TransactionId:
            raise E3ContractError('held_transaction needs a TransactionId')
        return tuple(sorted({
            block for block, entries in self._held.items()
            for entry in entries
            if not entry['released'] and entry['transaction'] == transaction
            and (category is None or entry['category'] is category)
        }))

    def _held_by(self, ticket):
        return set(self.held_blocks(ticket))

    def _held_by_transaction(self, transaction):
        if transaction is None:
            return set()
        return {block for block, entries in self._held.items()
                for entry in entries
                if not entry['released']
                and entry['transaction'] == transaction}

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

    def release_after_workers(self, ticket, acknowledgements, ranks):
        """Conservative whole-chunk release AFTER every worker drained/released.

        This is not a fabricated COPY_EVENT. The per-bucket release API remains
        unchanged. Callers must first correlate acknowledgements to queried
        final reports and their issued commands.
        """
        from wire_v1 import ReleaseAck
        if set(acknowledgements) != set(ranks):
            raise E3Pending('source release lacks an all-rank barrier')
        for rank, ack in acknowledgements.items():
            if (type(ack) is not ReleaseAck or ack.worker_rank != rank
                    or ack.transaction != ticket.transaction):
                raise E3ContractError('source release identity mismatch')
        entries = [e for items in self._held.values() for e in items
                   if e['ticket'] == ticket and e['category'] is LeaseCategory.SOURCE_HISTORY
                   and not e['released']]
        if not entries:
            raise E3ContractError('no source lease at the terminal barrier')
        blocks = set()
        for entry in entries:
            entry['released'] = True
            blocks.update(entry['blocks'])
        for block in tuple(self._held):
            self._held[block] = [e for e in self._held[block] if not e['released']]
            if not self._held[block]:
                del self._held[block]
        self.held_releases += 1
        self._free_released(tuple(sorted(blocks)))
        self._retry_requests()

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
        # get_new_blocks has selected and evicted these pages. Validate the
        # entire batch before changing any frontier entry: a later leased page
        # must not partially retire an earlier page's publication record.
        blocks = tuple(blocks)
        for block in blocks:
            if self.block_held(block):
                raise E3ContractError('allocator handed out a leased block')
        controller = getattr(self._scheduler, '_flash_s5_admission', None)
        if controller is not None:
            if (controller.leases is not self
                    or controller._scheduler is not self._scheduler):
                raise E3ContractError('allocator frontier belongs to another owner')
            # A free cached page remains shareable until actual reallocation.
            # Neither free_blocks nor commit is an incarnation boundary. The
            # frontier validates all unpublished conflicts before mutating.
            ids = tuple(getattr(block, 'block_id', None) for block in blocks)
            if any(type(block_id) is not int or block_id <= 0 for block_id in ids):
                raise E3ContractError('allocated frontier ids must be positive ints')
            controller.frontier.retire_blocks(ids)

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


def _pool_block_objects(pool, block_ids):
    """Resolve the allocator's real block objects for ``block_ids``.

    The pool keeps the authoritative ``KVCacheBlock`` list indexed by id
    (``pool.blocks``), so a page id addresses one physical incarnation. A CPU
    mirror of the ids is not a scheduler pin and is refused here.
    """
    blocks = getattr(pool, 'blocks', None)
    if blocks is None:
        raise E3ContractError('source hold needs the allocator block list')
    try:
        size = len(blocks)
    except TypeError:
        raise E3ContractError('allocator block list is not addressable') from None
    objects = []
    for block_id in block_ids:
        if block_id >= size:
            raise E3ContractError('source page is outside the allocator block list')
        block = blocks[block_id]
        if getattr(block, 'block_id', None) != block_id:
            raise E3ContractError('allocator block list is not indexed by block id')
        objects.append(block)
    return tuple(objects)


class SourceHold:
    """The scheduler-owned proof that a chunk's source pages are still valid.

    ``validate`` proves, for one physical page: the chunk transaction/state
    epoch is the parked one, the request rows still contain that page, the
    allocator object is the same incarnation admitted at pre-schedule time, and
    the source lease still holds it. Nothing Bool-valued stands in for those
    checks, and validation never mutates allocator state.

    ``release`` needs the correlated copy proof (``COPY_EVENT`` on the bound
    producer ticket) *and* every worker's release acknowledgment for this
    transaction/state epoch. A whole-chunk release conservatively reached after
    the all-rank barrier is recorded as such (``mode``) and never claims that a
    copy completed; callers that hold a real copy proof should pass it.

    ``descriptor`` exposes the transport-safe lease identity so a worker can
    validate the scheduler's hold against its own admitted frame; the engine
    keeps ownership of the real block objects throughout.
    """

    COPY = 'copy-event+release-barrier'
    BARRIER = 'all-rank-release-barrier'

    def __init__(self, leases, *, transaction, admission, pool, blocks, ranks,
                 is_current=None):
        if type(leases) is not SchedulerLeases:
            raise E3ContractError('source hold needs the scheduler leases')
        if type(transaction) is not TransactionId:
            raise E3ContractError('source hold needs a TransactionId')
        if admission is None or admission.transaction != transaction:
            raise E3ContractError('source hold admission belongs to another transaction')
        ranks = tuple(ranks)
        if not ranks or len(set(ranks)) != len(ranks):
            raise E3ContractError('source hold needs distinct worker ranks')
        pages = {block for block in admission.flat_blocks() if block > 0}
        pages.update(block for pair in admission.cow_pairs for block in pair)
        pages = tuple(sorted(pages))
        if not pages:
            raise E3ContractError('source hold covers no physical pages')
        objects = tuple(blocks)
        if tuple(getattr(block, 'block_id', None) for block in objects) != pages:
            raise E3ContractError('source hold objects do not cover the admitted pages')
        if len({id(block) for block in objects}) != len(objects):
            raise E3ContractError('duplicate allocator block in the source hold')
        if is_current is not None and not callable(is_current):
            raise E3ContractError('source hold currency must be callable')
        self._leases = leases
        self.transaction = transaction
        self.state_epoch = admission.state_epoch
        self._admission = admission
        self._pool = pool
        self._objects = objects
        self._by_id = {block.block_id: block for block in objects}
        self._pages = pages
        self._ranks = ranks
        self._is_current = is_current
        self._owners = None
        self._layer_count = None
        self._ticket = None
        self._released = False
        self._retired = False
        self._mode = None
        self.validations = 0
        self.releases = 0

    # -- identities -------------------------------------------------------
    @property
    def pages(self):
        return self._pages

    @property
    def ranks(self):
        return self._ranks

    @property
    def released(self):
        return self._released

    @property
    def release_mode(self):
        return self._mode

    @property
    def owners(self):
        return self._owners

    def bind_ticket(self, ticket):
        """Correlate the hold with the accepted producer operation."""
        if type(ticket) is not OperationTicket:
            raise E3ContractError('source hold binds to an OperationTicket')
        if ticket.transaction != self.transaction:
            raise E3ContractError('source hold ticket belongs to another transaction')
        if self._ticket is not None and self._ticket != ticket:
            raise E3ContractError('source hold is bound to another operation')
        if self._released or self._retired:
            raise E3ContractError('source hold is already retired')
        self._ticket = ticket
        return ticket

    def bind_owners(self, lanes, layer_count):
        """Record the lanes and layer range admitted for this chunk.

        ``lanes`` is the driver's rank -> opened :class:`LaneId` map; every
        owner must be one of them, so a page can only be validated for a lane
        that actually received this admitted frame.
        """
        if type(lanes) is not dict:
            raise E3ContractError('source owners need the opened lane map')
        if set(lanes) != set(self._ranks):
            raise E3ContractError('source owners do not cover every rank')
        owners = []
        for rank in self._ranks:
            lane = lanes[rank]
            if type(lane) is not LaneId:
                raise E3ContractError('source owner is not an opened LaneId')
            if lane.worker_rank != rank or lane.role is not LaneRole.PREFILL:
                raise E3ContractError('source owner is not this rank prefill lane')
            owners.append(lane)
        if type(layer_count) is not int or layer_count < 1:
            raise E3ContractError('source owners need the admitted layer count')
        owners = tuple(owners)
        if self._owners is not None:
            if self._owners != owners or self._layer_count != layer_count:
                raise E3ContractError('source owners changed after admission')
            return self._owners
        self._owners = owners
        self._layer_count = layer_count
        return owners

    # -- worker-facing lease ---------------------------------------------
    def descriptor(self):
        """Transport-safe lease identity; the engine keeps the block objects."""
        if self._owners is None:
            raise E3ContractError('source descriptor needs the admitted lanes')
        if self._released or self._retired:
            raise E3ContractError('source hold is already retired')
        from wire_v1 import SourceLeaseDescriptor
        return SourceLeaseDescriptor(self.transaction, self.state_epoch,
                                     self._pages, self._ranks)

    def _allocator_object(self, block_id):
        block = _pool_block_objects(self._pool, (block_id,))[0]
        return block

    def _held_pages(self):
        return set(self._leases.held_transaction(
            self.transaction, LeaseCategory.SOURCE_HISTORY))

    def _current(self):
        if self._retired:
            raise E3ContractError('source hold was retired without a submission')
        if self._released:
            raise E3ContractError('source hold was released')
        if self._is_current is not None and not self._is_current():
            raise E3ContractError('source hold belongs to a retired chunk')

    def validate(self, owner, layer, physical_pages):
        """Prove this chunk may still read ``physical_pages`` for ``owner``."""
        if type(owner) is not LaneId:
            raise E3ContractError('source validation needs an opened lane owner')
        if self._owners is None:
            raise E3ContractError('source validation needs the admitted lanes')
        if owner not in self._owners:
            raise E3ContractError('source validation owner was not admitted')
        if type(layer) is not int or not 0 <= layer < self._layer_count:
            raise E3ContractError('source validation layer is outside the model')
        pages = _flatten_ids(physical_pages)
        if not pages or any(type(page) is not int or page <= 0 for page in pages):
            raise E3ContractError('source validation needs physical page ids')
        if len(set(pages)) != len(pages):
            raise E3ContractError('duplicate physical page in validation')
        self._current()
        if not set(pages) <= set(self._pages):
            raise E3ContractError('physical page is outside this source hold')
        rows = {block for row in self._admission.block_tables
                for block in row if block > 0}
        # A CoW source is no longer in the request row (its destination is),
        # but the admitted frame still names it as one side of the copy.
        rows.update(block for pair in self._admission.cow_pairs for block in pair)
        held = self._held_pages()
        for page in pages:
            if page not in rows:
                raise E3ContractError('physical page left the admitted frame')
            block = self._by_id[page]
            if self._allocator_object(page) is not block:
                raise E3ContractError('physical page changed allocator incarnation')
            if page not in held:
                raise E3ContractError('allocator no longer holds the source page')
            if not getattr(block, 'is_null', False):
                count = getattr(block, 'ref_cnt', None)
                if type(count) is not int or count <= 0:
                    raise E3ContractError('allocator released the source page')
        self.validations += 1
        return True

    # -- release ----------------------------------------------------------
    def _check_barrier(self, acknowledgements, ranks, *, normal_release):
        from wire_v1 import ReleaseAck
        ranks = tuple(ranks)
        if ranks != self._ranks:
            raise E3ContractError('source release ranks differ from the hold')
        if type(acknowledgements) is not dict or set(acknowledgements) != set(ranks):
            raise E3ContractError('source release needs the all-rank barrier')
        for rank in ranks:
            ack = acknowledgements[rank]
            if type(ack) is not ReleaseAck or ack.worker_rank != rank:
                raise E3ContractError('source release acknowledgment is mistyped')
            if ack.transaction != self.transaction or ack.state_epoch != self.state_epoch:
                raise E3ContractError('source release acknowledgment is another chunk')
            if not ack.released:
                raise E3ContractError('source release acknowledgment did not release')
            if ack.request_closed and not self._admission.terminal:
                # A normal nonterminal release keeps the row for the next
                # chunk; a cancellation closes it before acknowledging.
                if normal_release:
                    raise E3ContractError(
                        'nonterminal release closed the request row')
                continue
            if normal_release and ack.request_closed != self._admission.terminal:
                raise E3ContractError(
                    'source release row closure differs from terminality')
            if not normal_release and self._admission.terminal and not ack.request_closed:
                raise E3ContractError(
                    'terminal source release kept the request row')

    def release(self, proof, acknowledgements, ranks):
        """Release the hold with the correlated copy proof and rank barrier."""
        if self._released:
            raise E3ContractError('source hold was already released')
        if self._retired:
            raise E3ContractError('source hold was retired without a submission')
        if self._ticket is None:
            raise E3ContractError('source release needs the bound operation ticket')
        if type(proof) is not CompletionProof:
            raise E3ContractError('source release needs a CompletionProof')
        if proof.ticket != self._ticket:
            raise E3ContractError('copy proof belongs to another operation')
        if proof.kind is not CompletionKind.COPY_EVENT:
            raise E3ContractError('source release needs the copy completion proof')
        self._check_barrier(acknowledgements, ranks, normal_release=True)
        self._current()
        self._leases.release(self._ticket, LeaseCategory.SOURCE_HISTORY, proof)
        self._released = True
        self._mode = self.COPY
        self.releases += 1
        return True

    def note_all_rank_release(self, acknowledgements, ranks):
        """Record the conservative whole-chunk barrier (no copy claim)."""
        self._check_barrier(acknowledgements, ranks, normal_release=False)
        if self._released:
            if self._mode != self.COPY:
                raise E3ContractError('source hold release was recorded twice')
            return self.BARRIER
        if self._retired:
            raise E3ContractError('source hold was retired without a submission')
        self._released = True
        self._mode = self.BARRIER
        self.releases += 1
        return self.BARRIER

    def retire_unsubmitted(self):
        """Retire a hold for an allocation that never became an operation."""
        if self._ticket is not None or self._released or self._retired:
            raise E3ContractError('a submitted source hold needs its completion proof')
        self._retired = True
        return True

    def describe(self):
        """Evidence summary; never contains payloads, hashes or secrets."""
        return dict(transaction_seq=self.transaction.txn_seq,
                    state_epoch=self.state_epoch,
                    request_uid=self.transaction.request_uid,
                    pages=len(self._pages), ranks=len(self._ranks),
                    bound=self._ticket is not None,
                    owners=0 if self._owners is None else len(self._owners),
                    released=self._released, mode=self._mode,
                    validations=self.validations, releases=self.releases)


class ParkedPrefill:
    """One scheduler-admitted chunk parked outside the running batch."""

    def __init__(self, *, request, queue, transaction, admission):
        self.request = request
        self.queue = queue
        self.transaction = transaction
        self.admission = admission
        self.ticket = None
        self.reports = {}
        self.pp_sent = set()
        self.pp_consumed = set()
        self.bound = False
        self.cache_published = False
        self.worker_releases = {}
        self.failed = False
        # AllocationCapture from the source-backed provider, if installed. It
        # owns the real CoW endpoint objects until commit/cancel retires them.
        self.capture = None
        # Scheduler-owned proof over the real allocator objects admitted here.
        self.source_hold = None

    @property
    def request_id(self):
        return self.transaction.request_uid


class PrefillAdmissionController:
    """Pre-schedule admission, delayed publication and one-token handback.

    ``admit`` runs at the top of ``schedule()``, before the stock waiting loop
    mutates anything. It performs the same prefix lookup and allocation the
    stock path would (``allocate_slots(delay_cache_blocks=True)``), removes the
    request from the waiting queue, and pre-leases every allocated block. The
    stock loop never sees the request, so it cannot publish optimistic storage.

    The chunk becomes cache-visible exactly once, from :meth:`commit`, after
    every rank reported a host-queried ``STATE_WRITE`` and every PP transfer
    has its consumer proof. Then the request is re-prepended to the waiting
    queue with ``num_computed_tokens`` advanced; the pinned waiting branch that
    handles ``num_computed_tokens > 0`` schedules the final token in stock
    decode (the one-token handback).

    Fail-closed: a request with a connector, encoder inputs, speculative
    lookahead, a blocked status, or only the last token left is left to stock.
    """

    def __init__(self, scheduler, *, leases, frontier, engine_epoch, chunk_tokens,
                 ranks, max_active=MAX_ACTIVE_PREFILL, max_history=64, clock=None):
        if type(leases) is not SchedulerLeases:
            raise E3ContractError('admission needs the E3 scheduler leases')
        if type(max_history) is not int or max_history < 1:
            raise E3ContractError('max_history must be a positive int')
        if type(engine_epoch) is not int or engine_epoch < 0:
            raise E3ContractError('engine_epoch must be a nonnegative int')
        if type(chunk_tokens) is not int or chunk_tokens < 1:
            raise E3ContractError('chunk_tokens must be a positive int')
        ranks = tuple(ranks)
        if not ranks or any(type(r) is not int or r < 0 for r in ranks):
            raise E3ContractError('admission needs the worker ranks')
        if len(set(ranks)) != len(ranks):
            raise E3ContractError('duplicate worker rank')
        if max_active != MAX_ACTIVE_PREFILL:
            raise E3ContractError('the fixed split runs one prefill at a time')
        self._scheduler = scheduler
        self.leases = leases
        self.frontier = frontier
        self.engine_epoch = engine_epoch
        self.chunk_tokens = chunk_tokens
        self.ranks = ranks
        self.max_active = max_active
        self.max_history = max_history
        self.clock = clock
        self.parked = None
        self.txn_seq = 0
        self._committed = {}
        self.admitted = 0
        self.alloc_failures = 0
        self.refused = 0
        self.commits = 0
        self.cancels = 0
        self.retained_releases = 0
        self.state_epoch_rejections = 0
        self._request_identities = {}
        self._next_generation = 0
        self._next_state_epoch = 0

    # -- installation -----------------------------------------------------
    def attach(self):
        scheduler = self._scheduler
        if getattr(scheduler, '_flash_s5_admission', None) is not None:
            raise E3ContractError('scheduler already has an E3 admission controller')
        scheduler._flash_s5_admission = self
        return self

    def detach(self):
        if getattr(self._scheduler, '_flash_s5_admission', None) is self:
            self._scheduler._flash_s5_admission = None

    def parked_count(self):
        return 0 if self.parked is None else 1

    @property
    def active(self):
        return self.parked is not None

    def _source_is_current(self, transaction):
        parked = self.parked
        return (parked is not None and parked.transaction == transaction
                and not parked.failed and not parked.cache_published)

    # -- admission --------------------------------------------------------
    def _queue(self, scheduler):
        select = getattr(scheduler, '_select_waiting_queue_for_scheduling', None)
        if callable(select):
            return select()
        return getattr(scheduler, 'waiting', None) or None

    def _aligned_split(self, scheduler, request, num_new_tokens,
                       num_new_local_computed_tokens=0):
        splitter = getattr(scheduler, '_mamba_block_aligned_split', None)
        if not callable(splitter):
            return num_new_tokens
        value = splitter(request, num_new_tokens,
                         num_new_local_computed_tokens=num_new_local_computed_tokens)
        if type(value) is not int or not 0 <= value <= num_new_tokens:
            raise E3ContractError('invalid aligned chunk size')
        return value

    def _environment_refuses(self, scheduler):
        if getattr(scheduler, 'connector', None) is not None:
            return True
        pause = getattr(scheduler, '_pause_state', None)
        if pause is not None and getattr(pause, 'name', 'UNPAUSED') != 'UNPAUSED':
            return True
        config = getattr(scheduler, 'vllm_config', None)
        if getattr(config, 'speculative_config', None) is not None:
            return True
        return False

    def _request_refuses(self, scheduler, request):
        if getattr(request, 'has_encoder_inputs', False):
            return True
        if getattr(request, 'kv_transfer_params', None):
            return True
        blocked = getattr(scheduler, '_is_blocked_waiting_status', None)
        if callable(blocked) and blocked(getattr(request, 'status', None)):
            return True
        return False

    def _native_admission_disabled(self, scheduler):
        """Read the EngineCore-owned admission intent at the policy seam.

        The generated policy seam defaults to stock when no driver is bound.
        Unit/qualification code can still call controller.admit directly. A
        bound driver is the sole owner of the intent: an absent intent keeps every
        request on the stock route, and a malformed, foreign-epoch or
        non-native one is refused rather than silently parked.
        """
        owner = getattr(scheduler, '_flash_s5_driver', None)
        query = getattr(owner, 'admission_intent', None)
        if owner is None:
            return True
        if not callable(query):
            raise E3ContractError('native admission owner has no intent query')
        intent = query()
        if intent is None:
            return True
        if type(intent) is not dict:
            raise E3ContractError('native admission intent is not a typed record')
        if intent.get('armed') is not True:
            raise E3ContractError('native admission intent is not armed')
        if intent.get('owner') != 'engine-core':
            raise E3ContractError('native admission intent has no engine owner')
        if intent.get('phase') not in NATIVE_ADMISSION_PHASES:
            raise E3ContractError('native admission intent names no native phase')
        if intent.get('engine_epoch') != self.engine_epoch:
            raise E3ContractError('native admission intent names another engine epoch')
        if intent.get('route') != 'split':
            raise E3ContractError('native admission intent needs the split route')
        generation = intent.get('route_generation')
        if (type(generation) is not int
                or generation != getattr(owner, 'route_generation', None)
                or getattr(owner, '_split_ready', None) is not True):
            raise E3ContractError('native admission intent has a stale route generation')
        if (getattr(owner, 'controller', None) is not self
                or getattr(owner, 'engine_epoch', None) != self.engine_epoch):
            raise E3ContractError('native admission intent has a foreign driver owner')
        return False

    def _next_transaction(self, request, start, chunk_end):
        self.txn_seq += 1
        # Object identity distinguishes reuse of a request UID in this engine.
        # Keep the object while registered; never infer generation from a
        # nonexistent vLLM request field or from Python id() reuse.
        identity = self._request_identities.get(request.request_id)
        if identity is None or identity[0] is not request:
            self._next_generation += 1
            identity = (request, self._next_generation)
            self._request_identities[request.request_id] = identity
        self._next_state_epoch += 1
        return TransactionId(self.engine_epoch, request.request_id,
                             identity[1], start, chunk_end, self.txn_seq)

    def _park_failed(self, request, queue, start, chunk_end, capture):
        """Quarantine a request whose allocation mutated and then failed.

        The capture (when the provider could build one) retains the real copy
        endpoint objects and every queued side effect. No worker operation was
        submitted, so no completion proof exists or is fabricated; the record
        stays parked and blocks any retry.
        """
        transaction = self._next_transaction(request, start, chunk_end)
        queue.remove_request(request)
        parked = ParkedPrefill(request=request, queue=queue,
                               transaction=transaction, admission=None)
        parked.capture = capture
        parked.failed = True
        self.parked = parked
        return parked

    def admit(self, scheduler):
        """Attempt one pre-schedule admission; return its record or ``None``."""
        if scheduler is not self._scheduler:
            raise E3ContractError('admission controller is attached elsewhere')
        if self.parked is not None or self._environment_refuses(scheduler):
            return None
        if (len(getattr(scheduler, 'running', ())) +
                getattr(scheduler, 'num_waiting_for_streaming_input', 0) + 1) > getattr(
                scheduler, 'max_num_running_reqs', 1):
            return None
        queue = self._queue(scheduler)
        if queue is None:
            return None
        request = queue.peek_request()
        owner = getattr(scheduler, '_flash_s5_driver', None)
        retained_query = getattr(owner, 'retained_request', None)
        retained = retained_query() if callable(retained_query) else None
        if retained is not None:
            identity = self._request_identities.get(request.request_id)
            if (request.request_id != retained.request_uid
                    or identity is None or identity[0] is not request
                    or identity[1] != retained.request_generation):
                # No allocation/parking/UID generation change for a request
                # whose worker row is still owned by another request frame.
                return None
        if self._request_refuses(scheduler, request):
            self.refused += 1
            return None
        manager = scheduler.kv_cache_manager
        # Native allocation also owns CoW endpoints, zeroing and boundary
        # state work. No fabricated empty defaults: the pinned allocation
        # adapter must capture and retain those side effects explicitly.
        metadata = getattr(scheduler, '_flash_s5_admission_metadata', None)
        if not callable(metadata):
            raise E3ContractError('native admission metadata adapter is not installed')
        prompt_end = getattr(request, 'num_tokens', None)
        start = getattr(request, 'num_computed_tokens', 0)
        if type(prompt_end) is not int or type(start) is not int or prompt_end < 2:
            self.refused += 1
            return None
        if getattr(request, 'num_output_tokens', 0) != 0:
            return None
        if prompt_end - start <= 1:
            return None

        new_computed_blocks = None
        num_new_computed = 0
        if start == 0:
            new_computed_blocks, num_new_computed, _ = (
                manager.get_computed_blocks(request))
            start = num_new_computed
            if prompt_end - start <= 1:
                # Everything except the final token is cached: stock handback.
                return None
        budget = getattr(scheduler, 'max_num_scheduled_tokens', None)
        if type(budget) is not int or budget < 1:
            raise E3ContractError('scheduler token budget is not declared')
        num_new = min(self.chunk_tokens, budget, prompt_end - 1 - start)
        num_new = self._aligned_split(
            scheduler, request, num_new,
            num_new_local_computed_tokens=num_new_computed)
        if num_new < 1:
            self.refused += 1
            return None
        chunk_end = start + num_new

        tokens = getattr(request, 'all_token_ids', None)
        if tokens is None:
            tokens = getattr(request, 'prompt_token_ids', None)
        if tokens is None:
            self.refused += 1
            return None
        input_ids = tuple(tokens[start:chunk_end])
        if len(input_ids) != num_new or any(type(t) is not int for t in input_ids):
            self.refused += 1
            return None

        # A source-backed provider scopes the global manager queues around
        # exactly this allocation; a legacy callable keeps the old contract.
        scoped = (callable(getattr(metadata, 'begin', None))
                  and callable(getattr(metadata, 'capture', None)))
        if scoped:
            metadata.begin(scheduler, request)
        try:
            blocks = manager.allocate_slots(
                request, num_new,
                num_new_computed_tokens=num_new_computed,
                new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=0,
                delay_cache_blocks=True,
                full_sequence_must_fit=getattr(scheduler, 'scheduler_reserve_full_isl', False),
                reserved_blocks=0,
                has_scheduled_reqs=bool(getattr(scheduler, 'running', ())))
        except BaseException:
            if scoped:
                # Keep every partially queued side effect visible and refuse a
                # retry of this request; never treat a failed allocation as a
                # stock fallback after the allocator mutated.
                self._park_failed(request, queue, start, chunk_end,
                                  metadata.abort())
            raise
        if blocks is None:
            capture = metadata.abort() if scoped else None
            if capture is not None:
                self._park_failed(request, queue, start, chunk_end, capture)
                raise E3ContractError(
                    'allocation failed after recording allocator side effects')
            self.alloc_failures += 1
            return None

        transaction = self._next_transaction(request, start, chunk_end)
        # Park BEFORE any post-allocation validation can fail. Allocation is
        # side-effectful, so a failure retains the request and blocks; it is
        # never an ordinary scheduling fallback.
        queue.remove_request(request)
        self.parked = ParkedPrefill(request=request, queue=queue,
                                   transaction=transaction, admission=None)
        try:
            if scoped:
                capture = metadata.capture(request, blocks, start, chunk_end)
                self.parked.capture = capture
                details = capture.details
            else:
                details = metadata(request, blocks, start, chunk_end)
            rows = manager.get_block_ids(request.request_id)
            block_tables = tuple(tuple(row) for row in rows)
            if not block_tables:
                raise E3ContractError('admission allocated no block rows')
            required = {'new_block_ids_to_zero', 'cow_pairs', 'ngram_context',
                        'unpublished_blocks'}
            if type(details) is not dict or set(details) != required:
                raise E3ContractError('admission metadata is incomplete')
            record = _wire_admission()(
                transaction=transaction, prompt_end=prompt_end,
                state_epoch=self._next_state_epoch, committed_end=start,
                block_tables=block_tables, slot_mappings=(),
                new_block_ids_to_zero=details['new_block_ids_to_zero'],
                cow_pairs=details['cow_pairs'], input_ids=input_ids,
                ngram_context=details['ngram_context'], sparse_active=False)
            self.parked.admission = record
            row_blocks = {b for b in record.flat_blocks() if b > 0}
            if not set(details['new_block_ids_to_zero']) <= row_blocks:
                raise E3ContractError('zeroing ids are outside the admission rows')
            flat = tuple(sorted(row_blocks
                                | {b for pair in record.cow_pairs for b in pair}))
            pool = getattr(manager, 'block_pool', None)
            try:
                objects = _pool_block_objects(pool, flat)
            except BaseException:
                # The allocation already happened. Keep its reservation so the
                # failed request can be retired by the never-submitted path.
                self.leases.reserve_blocks(
                    transaction, LeaseCategory.SOURCE_HISTORY, flat)
                raise
            self.leases.reserve_blocks(
                transaction, LeaseCategory.SOURCE_HISTORY, flat, blocks=objects)
            self.parked.source_hold = SourceHold(
                self.leases, transaction=transaction, admission=record,
                pool=pool, blocks=objects, ranks=self.ranks,
                is_current=lambda: self._source_is_current(transaction))
            # Only allocation-owned new/unpublished pages, never cached prefix
            # pages merely because this local frontier has not seen them yet.
            unpublished = details['unpublished_blocks']
            if type(unpublished) is not tuple or not set(unpublished) <= set(flat):
                raise E3ContractError('unpublished pages are outside the source lease')
            self.frontier.reserve(transaction, unpublished)
        except BaseException:
            self.parked.failed = True
            if scoped and self.parked.capture is None:
                # The provider refused inside capture(): its scope is intact
                # and the queue suffix untouched, so journal it (total, never
                # raises) instead of quarantining without the obligations.
                self.parked.capture = metadata.abort()
            raise
        self.admitted += 1
        return record

    # -- driver-facing state ----------------------------------------------
    def admission(self, transaction=None):
        parked = self.parked
        if parked is None:
            return None
        if transaction is not None and parked.transaction != transaction:
            raise E3ContractError('admission belongs to another transaction')
        return parked.admission

    def request(self, transaction=None):
        parked = self.parked
        if parked is None:
            return None
        if transaction is not None and parked.transaction != transaction:
            raise E3ContractError('admission belongs to another transaction')
        return parked.request

    def source_hold(self, transaction=None):
        """The parked chunk's scheduler-owned source proof, if any."""
        parked = self.parked
        if parked is None:
            return None
        if transaction is not None and parked.transaction != transaction:
            raise E3ContractError('source hold belongs to another transaction')
        return parked.source_hold

    def note_source_owners(self, transaction, lanes, layer_count):
        """Bind the admitted lanes/layers that may validate this source hold."""
        parked = self._parked_for(transaction)
        if parked.source_hold is None:
            raise E3ContractError('this chunk has no source hold')
        parked.source_hold.bind_owners(lanes, layer_count)
        return parked.source_hold

    def bind(self, ticket):
        """Bind the pre-lease to the accepted producer operation identity."""
        parked = self._parked_for(ticket.transaction)
        if parked.transaction != ticket.transaction:
            raise E3ContractError('lease ticket belongs to another transaction')
        self.leases.bind(ticket, LeaseCategory.SOURCE_HISTORY)
        if parked.source_hold is not None:
            parked.source_hold.bind_ticket(ticket)
        parked.ticket = ticket
        parked.bound = True
        return ticket

    def note_pp_send(self, transaction, seq):
        parked = self._parked_for(transaction)
        if type(seq) is not int or seq < 0:
            raise E3ContractError('PP transfer seq must be a nonnegative int')
        parked.pp_sent.add(seq)

    def note_pp_consumer(self, transaction, seq, stage):
        """Record the receiver's consumer proof identity (stage 1, exact seq)."""
        if stage != 1:
            raise E3ContractError('PP consumer proof belongs to stage 1')
        parked = self._parked_for(transaction)
        if seq not in parked.pp_sent:
            raise E3ContractError('consumer proof for an unsent transfer seq')
        parked.pp_consumed.add(seq)

    def note_report(self, report):
        """Record one rank's host-queried completion report."""
        if type(report) is not _wire_report():
            raise E3ContractError('admission needs a WorkerCompletionReport')
        parked = self._parked_for(report.transaction)
        if report.worker_rank not in self.ranks:
            raise E3ContractError('report comes from an unknown worker rank')
        if report.state_epoch != parked.admission.state_epoch:
            self.state_epoch_rejections += 1
            raise E3ContractError('report state_epoch does not match the chunk')
        prior = parked.reports.get(report.worker_rank)
        if prior is not None:
            if (report.completed_sequence < prior.completed_sequence
                    or not set(prior.completed_op_seqs) <= set(report.completed_op_seqs)):
                raise E3ContractError('completion report regressed')
        parked.reports[report.worker_rank] = report
        return report

    def ready_to_release(self, transaction):
        parked = self._parked_for(transaction)
        if set(parked.reports) != set(self.ranks):
            return False
        if parked.failed or any(r.failed or r.pending_op_seqs for r in parked.reports.values()):
            return False
        if not all(r.state_written() for r in parked.reports.values()):
            return False
        return parked.pp_sent <= parked.pp_consumed

    def ready_to_commit(self, transaction):
        parked = self._parked_for(transaction)
        return (self.ready_to_release(transaction)
                and set(parked.worker_releases) == set(self.ranks))

    def note_worker_release(self, acknowledgement):
        from wire_v1 import ReleaseAck
        if type(acknowledgement) is not ReleaseAck:
            raise E3ContractError('release needs a typed worker acknowledgment')
        parked = self._parked_for(acknowledgement.transaction)
        rank = acknowledgement.worker_rank
        if rank not in self.ranks or acknowledgement.state_epoch != parked.admission.state_epoch:
            raise E3ContractError('release identity differs from the admission')
        if not acknowledgement.released:
            raise E3ContractError('worker has not released this chunk')
        report = parked.reports.get(rank)
        if (report is None or report.pending_op_seqs or report.failed
                or acknowledgement.completed_sequence != report.completed_sequence):
            raise E3ContractError('release is not bound to the final completion report')
        parked.worker_releases[rank] = acknowledgement

    def commit(self, transaction, proof):
        """Publish the chunk once: cache blocks, leases, frontier, handback."""
        parked = self._parked_for(transaction)
        if parked.cache_published:
            raise E3ContractError('chunk was already published')
        if not parked.bound:
            raise E3ContractError('commit needs the accepted producer ticket')
        if not self.ready_to_commit(transaction):
            raise E3Pending('completion proofs are incomplete')
        if type(proof) is not CompletionProof:
            raise E3ContractError('commit needs the consumer completion proof')
        if proof.ticket != parked.ticket:
            raise E3ContractError('commit ticket differs from bound source reservation')

        admission = parked.admission
        # Validate terminal row ownership before cache publication or returning
        # any scheduler hold. Cancellation may close a nonterminal row, but a
        # normal commit must preserve it for the next chunk exactly as admitted.
        if any(ack.request_closed != admission.terminal
               for ack in parked.worker_releases.values()):
            raise E3ContractError('release row closure differs from the admitted chunk')
        hold = parked.source_hold
        strict = hold is not None and hold.released
        if strict and hold.release_mode != SourceHold.COPY:
            # A conservative barrier is recorded only by commit/cancel; seeing
            # it here means the hold was retired through a foreign path.
            raise E3ContractError('source hold was released without a copy proof')
        manager = self._scheduler.kv_cache_manager
        # Delayed allocation skipped this call; publish exactly once now.
        manager.cache_blocks(parked.request, transaction.chunk_end)
        parked.cache_published = True
        if strict:
            # The worker already returned the correlated copy proof and the
            # all-rank barrier; the strict release freed the pages once, so
            # there is no outstanding lease left to mark publishable.
            pass
        else:
            self.leases.publish(parked.ticket)
        self.frontier.publish(transaction)
        if not strict:
            self.leases.release_after_workers(
                parked.ticket, parked.worker_releases, self.ranks)
            if hold is not None:
                hold.note_all_rank_release(parked.worker_releases, self.ranks)
        # Copy retentions are allocator references, not completion evidence.
        # Every worker released and the chunk is being published, so the
        # journal may return them through the stock deferred-free route.
        if parked.capture is not None:
            if parked.capture.release_retained(self._scheduler):
                self.retained_releases += 1
        parked.request.num_computed_tokens = transaction.chunk_end
        self._committed[transaction] = dict(
            committed_end=transaction.chunk_end,
            remaining=admission.remaining_tokens,
            state_epoch=admission.state_epoch,
            ranks=tuple(sorted(parked.reports)),
            pp_seqs=tuple(sorted(parked.pp_sent)))
        while len(self._committed) > self.max_history:
            self._committed.pop(next(iter(self._committed)))
        self.parked = None
        if admission.remaining_tokens == 1:
            self._request_identities.pop(transaction.request_uid, None)
        self.commits += 1
        if admission.remaining_tokens > 0:
            # One-token handback: the pinned waiting branch schedules the rest.
            parked.queue.prepend_request(parked.request)
        return parked.request

    def cancel(self, transaction, *, proof=None, never_submitted=False):
        """Retire a parked chunk without publishing it.

        ``never_submitted`` is only valid when no operation was accepted and no
        PP transfer left the engine (the driver asserts it). Otherwise the
        normal consumer proof is mandatory -- cancellation never bypasses it.

        The request is returned unqueued: the engine's abort/finish path owns
        its final disposition. If that path runs first, the scheduler's
        deferred free waits for this lease and is retried here.
        """
        parked = self._parked_for(transaction)
        if not parked.pp_sent <= parked.pp_consumed:
            raise E3ContractError('a cancelled chunk still owns PP transfers')
        if never_submitted:
            if parked.bound:
                raise E3ContractError('a bound chunk had accepted operations')
            self.leases.cancel_reservation(transaction, LeaseCategory.SOURCE_HISTORY)
            if parked.source_hold is not None:
                parked.source_hold.retire_unsubmitted()
        else:
            if not parked.bound:
                raise E3ContractError('cancellation needs the accepted ticket')
            if type(proof) is not CompletionProof:
                raise E3ContractError('cancellation needs the consumer proof')
            if proof.ticket != parked.ticket:
                raise E3ContractError('cancellation ticket differs from source reservation')
            if set(parked.worker_releases) != set(self.ranks):
                raise E3Pending('cancellation waits for all worker releases')
            hold = parked.source_hold
            if hold is not None and hold.released:
                if hold.release_mode != SourceHold.COPY:
                    raise E3ContractError(
                        'source hold was released without a copy proof')
            else:
                self.leases.release_after_workers(
                    parked.ticket, parked.worker_releases, self.ranks)
                if hold is not None:
                    hold.note_all_rank_release(parked.worker_releases, self.ranks)
        self.frontier.cancel(transaction)
        if parked.capture is not None:
            # No metadata record certifies GPU completion; cancellation reached
            # this point only through never_submitted or the full release barrier.
            if parked.capture.release_retained(self._scheduler):
                self.retained_releases += 1
        self.parked = None
        self._request_identities.pop(transaction.request_uid, None)
        self.cancels += 1
        return parked.request

    def _parked_for(self, transaction):
        parked = self.parked
        if parked is None:
            raise E3ContractError('no chunk is parked')
        if parked.transaction != transaction:
            raise E3ContractError('transaction is not the parked chunk')
        return parked

    def summary(self):
        return dict(admitted=self.admitted, alloc_failures=self.alloc_failures,
                    refused=self.refused, commits=self.commits,
                    cancels=self.cancels, parked=self.parked_count(),
                    retained_releases=self.retained_releases,
                    state_epoch_rejections=self.state_epoch_rejections,
                    transactions=len(self._committed))


# --- generated-source hook functions ---------------------------------------

def _guard(owner):
    guard = getattr(owner, '_flash_s5_leases', None)
    if guard is not None and not isinstance(guard, SchedulerLeases):
        raise E3ContractError('owner E3 lease guard is invalid')
    return guard


def flash_s5_admit(scheduler):
    """Generated pre-schedule seam: admit at most one chunk, before any mutation.

    This is the policy seam as well as the parking seam: with no explicit
    EngineCore-owned native intent the request stays on the stock route.  A
    direct ``controller.admit`` call is the lower-level reviewed primitive used
    by construction/qualification code and is not re-gated here.
    """
    controller = getattr(scheduler, '_flash_s5_admission', None)
    if controller is None:
        return None
    if not isinstance(controller, PrefillAdmissionController):
        raise E3ContractError('scheduler E3 admission controller is invalid')
    if controller._native_admission_disabled(scheduler):
        return None
    return controller.admit(scheduler)


def parked_count(scheduler):
    """Parked chunks hold a model-runner slot and keep the engine alive."""
    controller = getattr(scheduler, '_flash_s5_admission', None)
    if controller is None:
        return 0
    if not isinstance(controller, PrefillAdmissionController):
        raise E3ContractError('scheduler E3 admission controller is invalid')
    return controller.parked_count()


def source_hold(scheduler, transaction=None):
    """Generated seam: the parked chunk's scheduler-owned source proof."""
    controller = getattr(scheduler, '_flash_s5_admission', None)
    if controller is None:
        return None
    if not isinstance(controller, PrefillAdmissionController):
        raise E3ContractError('scheduler E3 admission controller is invalid')
    return controller.source_hold(transaction)


def note_source_owners(scheduler, transaction, lanes, layer_count):
    """Generated seam: admit the lanes/layers that may validate the source."""
    controller = getattr(scheduler, '_flash_s5_admission', None)
    if controller is None:
        return None
    if not isinstance(controller, PrefillAdmissionController):
        raise E3ContractError('scheduler E3 admission controller is invalid')
    return controller.note_source_owners(transaction, lanes, layer_count)


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
    'ParkedPrefill', 'PrefillAdmissionController',
    'SCHEDULER_E2_SHA256', 'SCHEDULER_PATH', 'SCHEDULER_PINNED_SHA256',
    'SchedulerLeases', 'SourceHold', 'after_schedule',
    'before_scheduler_shutdown', 'before_touch', 'block_held',
    'blocked_deferred', 'defer_request_free', 'flash_s5_admit',
    'generate_block_pool', 'generate_scheduler', 'hold_leased_blocks',
    'note_allocated', 'note_cow_retained', 'note_source_owners',
    'parked_count', 'revert_block_pool', 'revert_scheduler', 'source_hold',
]
