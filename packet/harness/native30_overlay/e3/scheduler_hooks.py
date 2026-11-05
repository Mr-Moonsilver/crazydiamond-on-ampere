"""C: scheduler/block-pool leases for asynchronous prefill publication.

E2 protects its serial fences; E3 must not simply delete them. This module adds
an explicit lease guard on top of the reviewed E2 scheduler and block pool:

* A parked prefill transaction leases the blocks it may still write/read.
  ``free_blocks`` defers those blocks instead of returning them to the pool.
  Refcounts are decremented exactly once, at the real free, so
  ``is_block_writable`` and CoW counts keep their stock meaning.
* A lease never allocates and never copies. ``lease_caused_cow`` must stay 0.
* Prefix sharing can only see the :class:`~async_handoff.CommittedFrontier`
  (committed blocks). A stock request whose cached hit includes an uncommitted
  E3 source is deferred before allocation and retried after lease retirement;
  ``before_touch`` remains the final refusal guard.
* Release requires the consumer completion proof C records. Until then the
  blocks stay out of the free queue and out of prefix-cache publication.

The transforms target the E2-composed ``core/scheduler.py`` and
``core/block_pool.py`` (mirror + telemetry hooks already applied). Passing the
unmodified upstream scheduler is refused. No torch/vLLM import is needed here.
"""
import hashlib
import time
import weakref

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
    _text,
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
    ('''            dec_cap = (ndec_all + 1) // 2 if ndec_all > 1 else None
        ndec_sched = 0
''', '''            dec_cap = (ndec_all + 1) // 2 if ndec_all > 1 else None
        dec_cap = _flash_s5_hooks.decode_merge_cap(self, dec_cap)
        ndec_sched = 0
'''),
    ('''                        request.shared_prefix_boundary,
                        hit_diverged,
                    ) = self._get_local_prefix_cache_hit(request)

                    # Get externally-cached tokens if using a KVConnector.
''', '''                        shared_prefix_boundary,
                        hit_diverged,
                    ) = self._get_local_prefix_cache_hit(request)
                    # A cached block may also be a native prefill source. Keep
                    # the waiting request intact until that source is committed
                    # or released; touching it during allocate_slots would
                    # raise after the scheduler has begun this admission.
                    if _flash_s5_hooks.defer_stock_prefix_hit(self, new_computed_blocks, request=request):
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue
                    request.shared_prefix_boundary = shared_prefix_boundary

                    # Get externally-cached tokens if using a KVConnector.
'''),
    # native39: a waiting request with a positive frontier is a stock prefill
    # chunk too; the split route cannot run one.
    ('''                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens
''', '''                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens
                    # A positive-frontier stock prefill (KV receive, streaming
                    # resume, a native request between chunks) may not run on an
                    # armed split route either.
                    if _flash_s5_hooks.defer_stock_positive_frontier(self, request):
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue
'''),
    # native39: an in-progress stock prefill chunk of a RUNNING request.
    ('''            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue
''', '''            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue

            if request.is_prefill_chunk and _flash_s5_hooks.defer_stock_running_prefill(
                    self, request):
                # An in-progress stock prefill chunk waits while a native intent
                # is armed on the split route; decodes still run.
                req_index += 1
                continue
'''),
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
    ('''        dec_cap = _flash_s5_hooks.decode_merge_cap(self, dec_cap)
        ndec_sched = 0
''', '''        ndec_sched = 0
'''),
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
    ('''            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue

            if request.is_prefill_chunk and _flash_s5_hooks.defer_stock_running_prefill(
                    self, request):
                # An in-progress stock prefill chunk waits while a native intent
                # is armed on the split route; decodes still run.
                req_index += 1
                continue
''', '''            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue
'''),
    ('''                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens
                    # A positive-frontier stock prefill (KV receive, streaming
                    # resume, a native request between chunks) may not run on an
                    # armed split route either.
                    if _flash_s5_hooks.defer_stock_positive_frontier(self, request):
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue
''', '''                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens
'''),
    ('''                        shared_prefix_boundary,
                        hit_diverged,
                    ) = self._get_local_prefix_cache_hit(request)
                    # A cached block may also be a native prefill source. Keep
                    # the waiting request intact until that source is committed
                    # or released; touching it during allocate_slots would
                    # raise after the scheduler has begun this admission.
                    if _flash_s5_hooks.defer_stock_prefix_hit(self, new_computed_blocks, request=request):
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue
                    request.shared_prefix_boundary = shared_prefix_boundary

                    # Get externally-cached tokens if using a KVConnector.
''', '''                        request.shared_prefix_boundary,
                        hit_diverged,
                    ) = self._get_local_prefix_cache_hit(request)

                    # Get externally-cached tokens if using a KVConnector.
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


def decode_merge_cap(scheduler, dec_cap):
    """stage1-step L2 (FLASH_S5_DECODE_MERGE, default off): while the engine says so, one batch of all decoders.

    The decoder cap of this step.  Unset (always, with the switch off) the stock value.  Set: no cap on even steps, a
    cap of zero on odd ones, so an odd cohort stays eligible, waits one step and joins the even one.  An odd step is an
    empty step: its execute_model still consumes the PP handler's slot, so the sampled-token ring keeps its cadence.
    Unset again, the parity cap re-forms two cohorts within a step.
    """
    if getattr(scheduler, 'flash_s5_decode_merge', False) is not True:
        return dec_cap
    return None if scheduler.current_step % 2 == 0 else 0


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
        self.stock_prefix_deferrals = 0

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

    def defer_stock_prefix_hit(self, computed_blocks):
        """Pure pre-allocation check over the exact stock cached block objects.

        The scheduler keeps the request in its skipped-waiting queue for this
        one step. Its next schedule pass repeats the stock lookup, so a later
        source commit/release neither loses nor fabricates a cache hit.
        """
        groups = getattr(computed_blocks, 'blocks', None)
        if type(groups) is not tuple:
            raise E3ContractError('stock cached blocks have no typed groups')
        for group in groups:
            for block in group:
                entries = self._held.get(getattr(block, 'block_id', None), ())
                if any(not entry['committed'] and not entry['released']
                       for entry in entries):
                    self.stock_prefix_deferrals += 1
                    return True
        return False

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
        an E3-held copy source must belong to this transaction or already be
        committed. Unleased sources remain under the stock scheduler's CoW
        retention fence. The guard never creates a CoW or changes a refcount.
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
        leased_sources = set(sources) & set(self._held)
        if not leased_sources <= (mine | self._committed()):
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


QSA_STAGE_MIN_ROWS = 64  # pinned qsa.py: _QSA_STAGE_MIN_ROWS; no setting override
MAX_PLAN_REFUSAL_MEMO = 16  # refused-plan receipts kept per controller
# native39: how long stock work may be held on an armed split route while no
# native work is in flight before the engine returns FULL on its own (typed
# 'split-hold-timeout').  A constant, not a setting.
SPLIT_HOLD_BOUND_NS = 30_000_000_000
SPLIT_HOLD_KINDS = ('fresh', 'positive-frontier', 'running-prefill')
FURTHEST_BASIS = ('furthest end of any chunk the bounded search accepted, backtracked '
                  'edges included; descriptive, not a certified stock-forced stop')
# returning-append (Option A): a short held request may run natively between two
# chunks of a bulk native prefill.  Both bounds are launch settings; absent, no
# config object exists and every path below is the frozen one.
APPEND_YIELD_ENV = ('FLASH_S5_APPEND_YIELD_MAX_NEW', 'FLASH_S5_APPEND_YIELD_PER_BULK',
                    'FLASH_S5_APPEND_YIELD_HANDBACK_STEPS', 'FLASH_S5_APPEND_YIELD_REENTRY_RETRIES')


class AppendYieldConfig:
    """Launch-time bounds of the chunk-boundary yield to a short native request.

    ``max_new``: the most new (uncached) prompt tokens a yielded request may
    prefill natively.  ``per_bulk``: the most yields one bulk request grants.
    ``handback_steps``: the most engine steps the FULL window waits, on FULL,
    for the yielded request's one-token handback to be scheduled; then the
    yield ends and the bulk re-enters (the request is held as without a yield).
    ``reentry_retries``: the most scheduler passes the re-armed split waits for
    the bulk's reopen; then the yield ends and the bulk follows the existing
    refused-admission rule (FULL return, stock tail).  Both are counts of engine
    steps, never a clock.  ``enabled`` is the typed on/off control
    (``native_append_yield``); it starts on when the bounds are given at launch.
    """
    __slots__ = ('max_new', 'per_bulk', 'handback_steps', 'reentry_retries', 'enabled')
    FIELDS = ('max_new', 'per_bulk', 'handback_steps', 'reentry_retries')

    def __init__(self, max_new, per_bulk, handback_steps, reentry_retries, *, enabled=True):
        for name, value in zip(self.FIELDS, (max_new, per_bulk, handback_steps, reentry_retries)):
            if type(value) is not int or value < 1:
                raise E3ContractError('append yield %s must be a positive int' % name)
        if type(enabled) is not bool:
            raise E3ContractError('append yield enabled must be a bool')
        self.max_new, self.per_bulk = max_new, per_bulk
        self.handback_steps, self.reentry_retries = handback_steps, reentry_retries
        self.enabled = enabled

    @classmethod
    def from_env(cls, env):
        """``None`` when no bound is set; all four or none, else refused."""
        values = [env.get(name) for name in APPEND_YIELD_ENV]
        if values == [None] * len(APPEND_YIELD_ENV):
            return None
        if None in values:
            raise E3ContractError('append yield needs all four launch bounds or none')
        parsed = []
        for name, value in zip(APPEND_YIELD_ENV, values):
            if type(value) is not str or not value.isdigit() or value != str(int(value)):
                raise E3ContractError('%s must be a positive decimal int' % name)
            parsed.append(int(value))
        return cls(*parsed)

    def record(self):
        return {'max_new': self.max_new, 'per_bulk': self.per_bulk,
                'handback_steps': self.handback_steps, 'reentry_retries': self.reentry_retries,
                'enabled': self.enabled}


class _ProjectedPrefillRequest:
    """Read-only view for the stock, CPU-only Mamba split calculation."""
    __slots__ = ('_request', 'num_computed_tokens')

    def __init__(self, request, start):
        object.__setattr__(self, '_request', request)
        object.__setattr__(self, 'num_computed_tokens', start)

    def __getattr__(self, name):
        return getattr(self._request, name)

    def __setattr__(self, name, value):
        raise E3ContractError('projected prefill request is read-only')


class _ShortTailPrefillRequest:
    """Read-only view: stock's split sees the prompt's last hash boundary one unit lower.

    a1-trunk build-1d.  A chunk can't cross ``num_prompt_tokens // unit * unit``, so a prompt ending under the staged
    minimum past it has no native plan.  Below the boundary the stop lands one unit lower and the last chunk crosses
    the old one, carrying unit + tail rows.  ``num_tokens`` is untouched, so the prefill end stays.
    """
    __slots__ = ('_request', 'num_prompt_tokens')

    def __init__(self, request, boundary):
        object.__setattr__(self, '_request', request)
        object.__setattr__(self, 'num_prompt_tokens', boundary - 1)

    def __getattr__(self, name):
        return getattr(self._request, name)

    def __setattr__(self, name, value):
        raise E3ContractError('short tail prefill request is read-only')


def staged_chunk_plan(start, end, cap, minimum, split, *, query_limit=65536,
                      trace=None):
    """Find a complete native prefix partition before allocating any chunk.

    ``split(position, limit)`` is the stock alignment calculation, not an
    allocator. Search largest permitted chunks first, backtracking only when
    a later short tail cannot execute the unchanged staged attention path.
    Every returned chunk is an exact stock split, within the unchanged cap.
    A bounded search refusal leaves the request unallocated; on the FULL route
    stock owns it, on an armed split route it waits (``defer_stock_prefix_hit``).

    ``trace``, when a dict, receives the search receipt: split queries made,
    whether the query bound ended the search, and the furthest end of any
    chunk the search accepted.  That boundary is descriptive: it includes
    edges the search later backtracked from and stops early when the query
    bound is exhausted, so it is not certified as a stock-forced stop
    (native39, Codex P2).  It never changes the search or its result.
    """
    if trace is not None and type(trace) is not dict:
        raise E3ContractError('staged chunk planning trace must be a dict')
    if any(type(v) is not int for v in (start, end, cap, minimum, query_limit)):
        raise E3ContractError('staged chunk planning needs integer bounds')
    if start < 0 or end <= start or minimum < 1 or cap < 1 or query_limit < 1:
        raise E3ContractError('invalid staged chunk planning bounds')
    if cap < minimum or end - start < minimum:
        return _plan_result((), trace, 0, False, start)
    # Stack frames retain the next smaller candidate, so Python recursion and
    # per-token allocation are unnecessary even for a long context.
    stack = [[start, min(cap, end - start)]]
    path, impossible, queries = [], set(), 0
    furthest = start
    while stack:
        position, limit = stack[-1]
        if limit < minimum:
            impossible.add(position)
            stack.pop()
            if path:
                path.pop()
            continue
        queries += 1
        if queries > query_limit:
            return _plan_result((), trace, query_limit, True, furthest)
        size = split(position, limit)
        if type(size) is not int or not 0 <= size <= limit:
            raise E3ContractError('invalid projected aligned chunk size')
        stack[-1][1] = size - 1
        if size < minimum:
            continue
        following = position + size
        if following > furthest:
            furthest = following
        if following == end:
            return _plan_result(tuple(path + [size]), trace, queries, False, following)
        if following in impossible or end - following < minimum:
            continue
        path.append(size)
        stack.append([following, min(cap, end - following)])
    return _plan_result((), trace, queries, False, furthest)


def _plan_result(plan, trace, queries, exhausted, furthest):
    if trace is not None:
        trace.update(queries=queries, exhausted=exhausted, furthest_boundary=furthest)
    return plan


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
                 ranks, max_active=MAX_ACTIVE_PREFILL, max_history=64, clock=None,
                 minimum_stage_rows=1):
        if type(leases) is not SchedulerLeases:
            raise E3ContractError('admission needs the E3 scheduler leases')
        if type(max_history) is not int or max_history < 1:
            raise E3ContractError('max_history must be a positive int')
        if type(engine_epoch) is not int or engine_epoch < 0:
            raise E3ContractError('engine_epoch must be a nonnegative int')
        if type(chunk_tokens) is not int or chunk_tokens < 1:
            raise E3ContractError('chunk_tokens must be a positive int')
        if type(minimum_stage_rows) is not int or minimum_stage_rows < 1:
            raise E3ContractError('minimum_stage_rows must be a positive int')
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
        self.minimum_stage_rows = minimum_stage_rows
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
        # Typed refusal receipts (native37).  ``refused`` keeps its meaning:
        # one count per refused admission attempt.  A refused staged plan is
        # memoised per request object and planning inputs, so a request that
        # waits on an armed split route is not re-searched on every schedule().
        self.plan_searches = 0
        self.plan_memo_hits = 0
        self.split_stock_deferrals = 0
        self.refusal_seq = 0
        self.last_refusal = None
        self._plan_refusals = {}
        # native39: holds on the armed split route are bounded.  A hold counts
        # only while no native work is in flight (nothing parked, no retained
        # worker row) and every schedule() call keeps holding; past the bound
        # the driver returns FULL with a typed 'split-hold-timeout'.
        self.split_hold_bound_ns = SPLIT_HOLD_BOUND_NS
        self.split_hold_timeouts = 0
        self.split_stock_deferrals_by_kind = dict.fromkeys(SPLIT_HOLD_KINDS, 0)
        self._schedule_seq = 0
        self._split_hold = None
        self._split_hold_requests = {}
        self.last_split_hold_timeout = None

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
        policy = getattr(owner, 'auto_arm', None)
        if intent.get('phase') == 'auto-prefill' and policy is not None:
            if not policy.bound_intent(intent):
                raise E3ContractError('auto admission intent lost its exact request scope')
        elif intent.get('phase') not in NATIVE_ADMISSION_PHASES:
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

    def _refuse(self, request, reason, *, counted=True, **fields):
        """Count one refused attempt and keep its typed receipt; never raises.

        ``counted=False`` keeps the receipt for an outcome counted elsewhere
        (``alloc_failures``) without changing ``refused``.
        """
        if counted:
            self.refused += 1
        self.refusal_seq += 1
        identity = self._request_identities.get(getattr(request, 'request_id', None))
        record = {'schema': 's5-native-admission-refusal/1', 'seq': self.refusal_seq,
                  'reason': reason, 'engine_epoch': self.engine_epoch,
                  'request_uid': getattr(request, 'request_id', None),
                  'request_generation': (identity[1] if identity is not None
                                         and identity[0] is request else None),
                  'observed_ns': self._now_ns()}
        record.update(fields)
        self.last_refusal = record
        return None

    @staticmethod
    def _now_ns():
        # The driver journal's basis (CLOCK_MONOTONIC, ns).  ``self.clock`` is
        # the controller's seconds clock and is not used for receipts.
        return time.monotonic_ns()

    def _plan_key(self, scheduler, request, start, prompt_end, cap):
        owner = getattr(scheduler, '_flash_s5_driver', None)
        return (start, prompt_end, cap, self.minimum_stage_rows,
                getattr(request, 'num_prompt_tokens', None),
                getattr(request, 'shared_prefix_boundary', None),
                getattr(owner, 'route_generation', None))

    def _memoised_plan_refusal(self, request, key):
        entry = self._plan_refusals.get(getattr(request, 'request_id', None))
        if entry is None or entry[0]() is not request or entry[1] != key:
            return None
        return entry[2]

    def _remember_plan_refusal(self, request, key, receipt):
        # Bounded and weak (native39): the memo never keeps a finished request
        # (its token lists, host state) alive.  Identity stays exact: a dead or
        # foreign reference only ever fails to match.
        memo = self._plan_refusals
        memo.pop(request.request_id, None)
        for uid in [uid for uid, entry in memo.items() if entry[0]() is None]:
            memo.pop(uid)
        while len(memo) >= MAX_PLAN_REFUSAL_MEMO:
            memo.pop(next(iter(memo)))
        memo[request.request_id] = (weakref.ref(request), key, receipt)

    def forget_request(self, request):
        """Drop the refusal memo for a finished or aborted request (native39)."""
        uid = getattr(request, 'request_id', None)
        entry = self._plan_refusals.get(uid)
        if entry is not None and entry[0]() in (request, None):
            self._plan_refusals.pop(uid)
            return True
        return False

    # -- native40: bounded holds on the physical split route -------------------
    def _stock_hold_owner(self, scheduler):
        """Stock prefill safety follows the applied bank, independently of admission."""
        owner = getattr(scheduler, '_flash_s5_driver', None)
        if owner is None:
            return None
        if (getattr(owner, 'controller', None) is not self
                or getattr(owner, 'scheduler', None) is not scheduler
                or getattr(owner, 'engine_epoch', None) != self.engine_epoch
                or type(getattr(owner, '_split_ready', None)) is not bool
                or type(getattr(owner, 'route_generation', None)) is not int):
            raise E3ContractError('stock prefill hold has a foreign route owner')
        if not owner._split_ready:
            return None
        # Keep strict validation for an existing intent; absence is safe to
        # observe, but no longer authorizes stock prefill on a split bank.
        if owner.admission_intent() is not None:
            self._native_admission_disabled(scheduler)
        return owner

    def _note_split_hold(self, scheduler, kind, request=None):
        """Count held stock work, with a clock owned by the physical route."""
        if kind not in SPLIT_HOLD_KINDS:
            raise E3ContractError('unknown split hold kind')
        owner = self._stock_hold_owner(scheduler)
        if owner is None:
            return
        self.split_stock_deferrals += 1
        self.split_stock_deferrals_by_kind[kind] += 1
        if self.parked is not None or owner.tx is not None or owner.retained_request() is not None:
            self._split_hold = None
            self._split_hold_requests.clear()
            return
        # returning-append: a yield owns this split route between the bulk's rows
        # (closing / append / resume); no idle-hold budget runs while it does.
        yielding = getattr(owner, 'append_yield_active', None)
        if callable(yielding) and yielding():
            self._split_hold = None
            self._split_hold_requests.clear()
            return
        if request is None:
            queue = self._queue(scheduler)
            request = queue.peek_request() if queue else None
        if request is None:
            return
        intent = owner.admission_intent()
        hold = self._split_hold
        now = self._now_ns()
        if (hold is None or hold['engine_epoch'] != owner.engine_epoch
                or hold['route_generation'] != owner.route_generation
                or hold['schedule_seq'] < self._schedule_seq - 1
                or (hold['schedule_seq'] != self._schedule_seq
                    and not self._current_stock_holds(scheduler))):
            hold = self._split_hold = {
                'since_ns': now, 'engine_epoch': owner.engine_epoch,
                'route_generation': owner.route_generation,
                'schedule_seq': self._schedule_seq, 'holds': 0,
                'kinds': dict.fromkeys(SPLIT_HOLD_KINDS, 0)}
            self._split_hold_requests.clear()
        elif hold['schedule_seq'] != self._schedule_seq:
            self._split_hold_requests.clear()
        self._split_hold_requests[id(request)] = (weakref.ref(request), kind)
        hold.update(schedule_seq=self._schedule_seq,
                    intent_seq=None if intent is None else intent['seq'],
                    admission_armed=intent is not None, last_ns=now)
        hold['holds'] += 1
        hold['kinds'][kind] += 1

    def _current_stock_holds(self, scheduler):
        """Only the same live request objects still in a held scheduling branch."""
        requests = getattr(scheduler, 'requests', None)
        if type(requests) is not dict:
            return []
        def contains(queue, request):
            if queue is None:
                return False
            if hasattr(queue, '__iter__'):
                return any(item is request for item in queue)
            # RequestQueue's head interface is sufficient for head-only ports.
            return bool(queue) and queue.peek_request() is request
        out = []
        for ref, kind in self._split_hold_requests.values():
            request = ref()
            if request is None or requests.get(request.request_id) is not request:
                continue
            if kind == 'running-prefill':
                live = (contains(getattr(scheduler, 'running', None), request)
                        and getattr(request, 'is_prefill_chunk', False) is True)
            else:
                live = any(contains(getattr(scheduler, name, None), request)
                           for name in ('waiting', 'skipped_waiting'))
            if live:
                out.append(request)
        return out

    def expired_split_hold(self, now_ns, engine_epoch, route_generation):
        """An expired, still-current physical-route hold, armed or unarmed."""
        hold = self._split_hold
        if (hold is None or self.parked is not None
                or hold['engine_epoch'] != engine_epoch
                or hold['route_generation'] != route_generation
                or hold['schedule_seq'] != self._schedule_seq):
            return None
        if not self._current_stock_holds(self._scheduler):
            self._split_hold = None
            self._split_hold_requests.clear()
            return None
        if now_ns - hold['since_ns'] < self.split_hold_bound_ns:
            return None
        return dict(hold, kinds=dict(hold['kinds']))

    def note_split_hold_timeout(self, scheduler, hold, now_ns, intent):
        """Record policy recovery; unarmed stock work is not a native refusal."""
        requests = self._current_stock_holds(scheduler)
        request = requests[0] if requests else None
        self.split_hold_timeouts += 1
        self._split_hold = None
        self._split_hold_requests.clear()
        fields = dict(held_since_ns=hold['since_ns'], held_ns=now_ns - hold['since_ns'],
                      bound_ns=self.split_hold_bound_ns, holds=hold['holds'],
                      kinds=dict(hold['kinds']), engine_epoch=hold['engine_epoch'],
                      route_generation=hold['route_generation'],
                      intent_seq=None if intent is None else intent['seq'],
                      admission_armed=intent is not None, action='return-full')
        reason = 'split-hold-timeout' if intent is not None else 'unarmed-stock-prefill-hold-timeout'
        self.last_split_hold_timeout = dict(schema='s5-stock-prefill-hold-timeout/1',
                                           reason=reason, observed_ns=now_ns, **fields)
        if intent is not None:
            self._refuse(request, reason, **fields)
        return dict(self.last_split_hold_timeout)

    # returning-append: launch-time yield bounds (AppendYieldConfig) or None.
    append_yield = None
    yield_stock_deferrals_by_kind = None
    yield_refusals = None
    yield_lookups = None
    _yield_scanning = False

    def note_yield_hold(self, kind):
        """Count a stock prefill held in a yield's FULL window (no hold clock)."""
        if self.yield_stock_deferrals_by_kind is None:
            self.yield_stock_deferrals_by_kind = {name: 0 for name in SPLIT_HOLD_KINDS}
        self.yield_stock_deferrals_by_kind[kind] += 1

    def _waiting_object(self, scheduler, queue, target):
        """The queue holding exactly ``target`` (the yield's append or bulk), or None."""
        if queue.peek_request() is target:
            return queue, target
        for candidate in (queue, getattr(scheduler, 'skipped_waiting', None),
                          getattr(scheduler, 'waiting', None)):
            if candidate is None or not hasattr(candidate, '__iter__'):
                continue
            if any(item is target for item in candidate):
                return candidate, target
        return None

    def yield_lookup_timer(self, scheduler):
        """A read: every prefix lookup through the manager, whoever calls it, counted and timed (box 1: about 0.265 s on a 151k prompt).

        Installed once the first time the driver offers, also with the control off, so an off arm says what stock's own lookups cost.
        """
        manager = getattr(scheduler, 'kv_cache_manager', None)
        real = getattr(manager, 'get_computed_blocks', None)
        if real is None or getattr(real, 'yield_timed', False):
            return
        stats = self.yield_lookups = dict(calls=0, ns=0, max_ns=0, scan_calls=0, scan_ns=0)

        def timed(request):
            started = time.monotonic_ns()
            try:
                return real(request)
            finally:
                took = time.monotonic_ns() - started
                stats['calls'] += 1
                stats['ns'] += took
                stats['max_ns'] = max(stats['max_ns'], took)
                if self._yield_scanning:
                    stats['scan_calls'] += 1
                    stats['scan_ns'] += took
        timed.yield_timed = True
        manager.get_computed_blocks = timed

    def _yield_refuse(self, reason):
        """Count why a held request isn't taken; a declined yield leaves no other record."""
        if self.yield_refusals is None:
            self.yield_refusals = {}
        self.yield_refusals[reason] = self.yield_refusals.get(reason, 0) + 1

    def _append_yield_gate(self, scheduler, decode_counts):
        """The rules on the engine, before any request is read."""
        config = self.append_yield
        if config is None or not config.enabled:
            return False
        running = len(getattr(scheduler, 'running', ()))
        # The append's slot plus the slot auto-arm keeps for the bulk (its re-entry
        # cohort is at most max_num_running_reqs - 1, as at selection).
        if running + 2 > getattr(scheduler, 'max_num_running_reqs', 1):
            self._yield_refuse('slots')
            return False
        # The re-entry identity proof needs every cohort size 1..running+1.
        if decode_counts is not None and not set(range(1, running + 2)) <= set(decode_counts):
            self._yield_refuse('decode-cover')
            return False
        if self._environment_refuses(scheduler):
            self._yield_refuse('environment')
            return False
        return True

    def append_yield_still_held(self, scheduler, request, decode_counts=None):
        """The drain's re-check: the selected request is still takeable.  No lookup; the admission does its own."""
        return (self._append_yield_gate(scheduler, decode_counts)
                and self._append_yield_eligible(scheduler, request, getattr(scheduler, 'requests', None)))

    def append_yield_candidate(self, scheduler, bulk, decode_counts=None, exclude=(), seen=None):
        """The first held request (FCFS) that may run natively between two bulk chunks.

        The same refusals as auto-arm's plan (native_auto_arm._plan) except the
        bulk-size floor: waiting, nothing computed or generated, no LoRA, pooling,
        structured output, logprobs, connector or shared-prefix junction; then at
        most ``max_new`` new tokens beyond its cached prefix and a complete staged
        plan.  With ``decode_counts`` (the worker's proven decode cohort sizes) the
        running set plus every smaller cohort must be proven.  ``exclude``: request
        ids already refused for this bulk.  ``seen`` (a set the caller keeps): request ids
        looked up already, skipped and added to.  Read-only: a lookup, never an allocation.
        """
        if not self._append_yield_gate(scheduler, decode_counts):
            return None
        config = self.append_yield
        requests = getattr(scheduler, 'requests', None)
        looked = set()
        for queue in (getattr(scheduler, 'skipped_waiting', None), getattr(scheduler, 'waiting', None)):
            if queue is None or not hasattr(queue, '__iter__'):
                continue
            for request in list(queue):
                uid = getattr(request, 'request_id', None)
                if request is bulk or id(request) in looked or uid in exclude:
                    continue
                looked.add(id(request))
                if seen is not None:
                    if uid in seen:
                        continue
                    seen.add(uid)
                if self._append_yield_plan(scheduler, request, requests, config) is not None:
                    return request
        return None

    def _append_yield_short_tail(self, scheduler, prompt_end):
        """True when stock's split leaves a last chunk under the staged minimum, at any start.

        Stock must stop at the prompt's last hash boundary, so a chunk can't cross it (box 1: tails of 16 and 58 rows).
        0 is not short: the boundary is the native end itself, a chunk may end on it.
        """
        unit = getattr(scheduler, 'hash_block_size', None)
        if getattr(scheduler, 'mamba_partial_cache_hit', False) is not True or type(unit) is not int or unit < 1:
            return False
        return 0 < prompt_end - 1 - prompt_end // unit * unit < self.minimum_stage_rows

    def _append_yield_split_view(self, scheduler, request):
        """The request stock's split reads for a yield's short request: a short tail moves the stop one hash unit back.

        Cost: no partial-tail entry at the old boundary, so this request's next turn resumes one registered state
        earlier.  Under one unit of prompt there is nothing to move back to; the plan refuses it as before.
        """
        prompt_end = request.num_tokens
        if not self._append_yield_short_tail(scheduler, prompt_end):
            return request
        boundary = prompt_end // scheduler.hash_block_size * scheduler.hash_block_size
        if boundary < scheduler.hash_block_size:
            return request
        return _ShortTailPrefillRequest(request, boundary)

    def _append_yield_eligible(self, scheduler, request, requests):
        sampling = getattr(request, 'sampling_params', None)
        if (type(requests) is not dict or requests.get(getattr(request, 'request_id', None)) is not request
                or getattr(getattr(request, 'status', None), 'name', None) != 'WAITING'
                or getattr(request, 'num_computed_tokens', None) != 0
                or getattr(request, 'num_output_tokens', None) != 0
                or getattr(request, 'num_preemptions', 0) != 0 or getattr(request, 'resumable', False)
                or getattr(request, 'lora_request', None) is not None
                or getattr(request, 'pooling_params', None) is not None
                or getattr(request, 'structured_output_request', None) is not None
                or getattr(request, 'ec_transfer_params', None)
                or sampling is None or getattr(sampling, 'prompt_logprobs', None) is not None
                or getattr(sampling, 'logprobs', None) is not None
                or self._request_refuses(scheduler, request)):
            self._yield_refuse('request')
            return False
        try:
            _text('request_uid', request.request_id)
        except E3ContractError:
            self._yield_refuse('request')
            return False
        prompt_end = getattr(request, 'num_tokens', None)
        if type(prompt_end) is not int or prompt_end < 2:
            self._yield_refuse('request')
            return False
        return True

    def _append_yield_plan(self, scheduler, request, requests, config):
        if not self._append_yield_eligible(scheduler, request, requests):
            return None
        prompt_end = request.num_tokens
        manager = scheduler.kv_cache_manager
        self._yield_scanning = True
        try:
            blocks, start, shared = manager.get_computed_blocks(request)
        finally:
            self._yield_scanning = False
        if shared or type(start) is not int or start < 0:
            self._yield_refuse('shared-junction')
            return None
        end = prompt_end - 1
        if end - start < 1 or end - start > config.max_new:
            self._yield_refuse('max-new')
            return None
        # auto-arm _plan's free-block check: the whole request must fit now.
        needed = manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id, num_tokens=prompt_end,
            new_computed_blocks=blocks.blocks, num_encoder_tokens=0,
            total_computed_tokens=start, num_local_computed_tokens=start,
            num_tokens_main_model=prompt_end, apply_admission_cap=True)
        if needed + manager.watermark_blocks > manager.block_pool.get_num_free_blocks():
            self._yield_refuse('blocks')
            return None
        cap = min(self.chunk_tokens, getattr(scheduler, 'max_num_scheduled_tokens', 0))
        if cap < 1:
            self._yield_refuse('cap')
            return None
        view = self._append_yield_split_view(scheduler, request)
        def split(position, limit):
            projected = view if position == start else _ProjectedPrefillRequest(view, position)
            return self._aligned_split(scheduler, projected, limit,
                                       num_new_local_computed_tokens=start if position == start else 0)
        plan = staged_chunk_plan(start, end, cap, self.minimum_stage_rows, split)
        if not plan:
            self._yield_refuse('no-plan')
        return plan or None

    def _retained_waiting(self, scheduler, queue, retained):
        """The retained request and the queue holding it, wherever it waits.

        A worker row is owned by exactly one request object; that request
        continues first even when a held or refused request sits ahead of it
        (native39: a held head must not strand a retained row).
        """
        identity = self._request_identities.get(retained.request_uid)
        if identity is None or identity[1] != retained.request_generation:
            return None
        target = identity[0]
        if queue.peek_request() is target:
            return queue, target
        for candidate in (queue, getattr(scheduler, 'skipped_waiting', None),
                          getattr(scheduler, 'waiting', None)):
            if candidate is None or not hasattr(candidate, '__iter__'):
                continue
            if any(item is target for item in candidate):
                return candidate, target
        return None

    def native_admission_record(self):
        """Engine-lifetime admission counts and the newest typed refusal."""
        return {'schema': 's5-native-admission/1',
                'admitted': self.admitted, 'refused': self.refused,
                'alloc_failures': self.alloc_failures,
                'plan_searches': self.plan_searches,
                'plan_memo_hits': self.plan_memo_hits,
                'split_stock_deferrals': self.split_stock_deferrals,
                'split_stock_deferrals_by_kind': dict(self.split_stock_deferrals_by_kind),
                'split_hold_timeouts': self.split_hold_timeouts,
                'last_split_hold_timeout': (None if self.last_split_hold_timeout is None
                                            else dict(self.last_split_hold_timeout)),
                'split_hold_bound_ns': self.split_hold_bound_ns,
                'split_hold': (None if self._split_hold is None
                               else dict(self._split_hold,
                                         kinds=dict(self._split_hold['kinds']))),
                'refusal_seq': self.refusal_seq,
                'last_refusal': (None if self.last_refusal is None
                                 else dict(self.last_refusal)),
                'basis': ('engine-lifetime monotonic counts; subtract matching status '
                          'snapshots for a window')}

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
        owner = getattr(scheduler, '_flash_s5_driver', None)
        cost = getattr(owner, 'control_path', None)
        if cost is not None:
            with cost.span('prepare_ns'):
                return self._admit_observed(scheduler)
        return self._admit_observed(scheduler)

    def _admit_observed(self, scheduler):
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
            # No allocation/parking/UID generation change for a request whose
            # worker row is still owned by another request frame: only the
            # retained request itself may continue, from whichever queue holds it.
            found = self._retained_waiting(scheduler, queue, retained)
            if found is None:
                return None
            queue, request = found
        else:
            # returning-append: while a yield runs, only its current target (the
            # short request, then the bulk's reopen) may be admitted; while the
            # bulk row is being closed nothing is.
            target_query = getattr(owner, 'append_yield_target', None)
            target = target_query() if callable(target_query) else None
            if target is not None:
                phase, wanted = target
                if wanted is None:
                    return None
                found = self._waiting_object(scheduler, queue, wanted)
                if found is None:
                    owner.append_yield_target_lost(phase)
                    return None
                queue, request = found
        policy = getattr(owner, 'auto_arm', None)
        if policy is not None and policy.active and not policy.permits(request):
            return None
        if self._request_refuses(scheduler, request):
            return self._refuse(request, 'request-refused')
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
            return self._refuse(request, 'prompt-geometry',
                                prompt_tokens=prompt_end if type(prompt_end) is int else None)
        if getattr(request, 'num_output_tokens', 0) != 0:
            # A resumed (preempted) request recomputes prompt and output; native
            # admission plans prompts only.  Typed, so a hold on it is visible.
            return self._refuse(request, 'resumed-with-output', prompt_tokens=prompt_end,
                                start=start,
                                output_tokens=getattr(request, 'num_output_tokens', None))
        if prompt_end - start <= 1:
            return None

        new_computed_blocks = None
        num_new_computed = 0
        if start == 0:
            new_computed_blocks, num_new_computed, _ = (
                manager.get_computed_blocks(request))
            start = num_new_computed
            if prompt_end - start <= 1:
                # Everything except the final token is cached: stock handback,
                # which only the FULL route can run.  Typed (native39).
                return self._refuse(request, 'full-prefix-hit', prompt_tokens=prompt_end,
                                    cached_tokens=num_new_computed)
        budget = getattr(scheduler, 'max_num_scheduled_tokens', None)
        if type(budget) is not int or budget < 1:
            raise E3ContractError('scheduler token budget is not declared')
        cap = min(self.chunk_tokens, budget)
        if self.minimum_stage_rows > 1:
            # a1-trunk 1d: every chunk of a yield's short request plans over the same view, flag off this is the request.
            is_append = getattr(owner, 'append_yield_is_append', None)
            view = (self._append_yield_split_view(scheduler, request)
                    if callable(is_append) and is_append(request) else request)
            def projected_split(position, limit):
                if position == start:
                    return self._aligned_split(
                        scheduler, view, limit,
                        num_new_local_computed_tokens=num_new_computed)
                return self._aligned_split(
                    scheduler, _ProjectedPrefillRequest(view, position), limit)
            key = self._plan_key(scheduler, request, start, prompt_end, cap)
            receipt = (None if retained is not None
                       else self._memoised_plan_refusal(request, key))
            if receipt is not None:
                # Same request object, same planning inputs: the bounded search
                # is deterministic, so its refusal is reused, not repeated.
                self.plan_memo_hits += 1
                return self._refuse(request, 'no-staged-plan', memoised=True, **receipt)
            trace = {}
            self.plan_searches += 1
            plan = staged_chunk_plan(start, prompt_end - 1, cap,
                                     self.minimum_stage_rows, projected_split,
                                     trace=trace)
            if not plan:
                if retained is not None:
                    # Returning None here would let stock touch a still-owned
                    # private row. Keep the existing ownership refusal instead.
                    raise E3ContractError('retained native request has no staged chunk plan')
                furthest = trace.get('furthest_boundary')
                receipt = {'prompt_tokens': prompt_end, 'start': start,
                           'native_end': prompt_end - 1, 'cap': cap,
                           'minimum_rows': self.minimum_stage_rows,
                           'furthest_boundary': furthest,
                           'furthest_basis': FURTHEST_BASIS,
                           'gap': (prompt_end - 1 - furthest
                                   if type(furthest) is int else None),
                           'queries': trace.get('queries'),
                           'exhausted': trace.get('exhausted'),
                           'route_generation': key[-1]}
                self._remember_plan_refusal(request, key, receipt)
                return self._refuse(request, 'no-staged-plan', memoised=False, **receipt)
            num_new = plan[0]
        else:
            num_new = self._aligned_split(
                scheduler, request, min(cap, prompt_end - 1 - start),
                num_new_local_computed_tokens=num_new_computed)
        if num_new < 1:
            return self._refuse(request, 'no-aligned-chunk', prompt_tokens=prompt_end,
                                start=start, cap=cap)
        chunk_end = start + num_new

        tokens = getattr(request, 'all_token_ids', None)
        if tokens is None:
            tokens = getattr(request, 'prompt_token_ids', None)
        if tokens is None:
            return self._refuse(request, 'token-ids-unavailable')
        input_ids = tuple(tokens[start:chunk_end])
        if len(input_ids) != num_new or any(type(t) is not int for t in input_ids):
            return self._refuse(request, 'token-ids-invalid')

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
            return self._refuse(request, 'allocation-failed', counted=False,
                                prompt_tokens=prompt_end, start=start,
                                chunk_tokens=num_new)

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
            check = getattr(owner, 'append_yield_check_blocks', None)
            if callable(check):
                check(request, set(unpublished) | set(details['new_block_ids_to_zero'])
                      | {pair[1] for pair in record.cow_pairs})
            self.frontier.reserve(transaction, unpublished)
        except BaseException:
            self.parked.failed = True
            if scoped and self.parked.capture is None:
                # The provider refused inside capture(): its scope is intact
                # and the queue suffix untouched, so journal it (total, never
                # raises) instead of quarantining without the obligations.
                self.parked.capture = metadata.abort()
            raise
        self._split_hold = None
        self._split_hold_requests.clear()
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
    # Every schedule visit counts, even when native admission is disarmed.
    controller._schedule_seq += 1
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
    controller = getattr(scheduler, '_flash_s5_admission', None)
    if isinstance(controller, PrefillAdmissionController):
        # A finished or aborted request leaves the refusal memo (native39).
        controller.forget_request(request)
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


def defer_stock_prefix_hit(scheduler, computed_blocks, *, request=None):
    """Generated stock-loop seam for a new request (``num_computed_tokens == 0``).

    True keeps the request in skipped-waiting for this step, unallocated.  It
    defers a stock prefix hit on a still-leased native page, and (native37)
    every new stock prefill while the physical engine route is split: stock cannot dispatch a non-FULL batch there (the worker's
    route guard raises), so a request native admission refused waits until the
    route is FULL again instead of reaching the workers.
    """
    guard = _guard(scheduler)
    if guard is not None and guard.defer_stock_prefix_hit(computed_blocks):
        return True
    # Independent of the lease guard: the applied route decides the hold.
    return _defer_stock_while_split_armed(scheduler, 'fresh', request)


def defer_stock_positive_frontier(scheduler, request):
    """Generated stock-loop seam for a waiting request with ``num_computed_tokens > 0``.

    native39 (Codex P1): KV receives, streaming resumes and a native request
    between chunks take this branch, past the fresh-request seam.  Their stock
    chunk is a prefill as well, so it is held on a physical split route too.
    The native one-token handback never needs an exception: the driver skips
    every stock step from the terminal commit until the FULL ack, and that
    ack clears the intent (engine_driver ``_drive_route``/transition ack).
    """
    return _defer_stock_while_split_armed(scheduler, 'positive-frontier', request)


def defer_stock_running_prefill(scheduler, request):
    """Generated running-loop seam for an in-progress stock prefill chunk.

    A request that was mid-prefill on FULL when the split was armed would
    otherwise schedule its next chunk on the split route (native39).
    """
    return _defer_stock_while_split_armed(scheduler, 'running-prefill', request)


def _defer_stock_while_split_armed(scheduler, kind, request=None):
    controller = getattr(scheduler, '_flash_s5_admission', None)
    if controller is None:
        return False
    if not isinstance(controller, PrefillAdmissionController):
        raise E3ContractError('scheduler E3 admission controller is invalid')
    # Historical helper name; stock safety is independent of admission policy.
    if controller._stock_hold_owner(scheduler) is None:
        # returning-append A1: the yield's FULL window (the append's handback and
        # the bulk's re-entry) keeps every stock prefill held but that handback.
        owner = getattr(scheduler, '_flash_s5_driver', None)
        holds = getattr(owner, 'append_yield_holds', None)
        if not (callable(holds) and holds(request, kind)):
            return False
        controller.note_yield_hold(kind)
        return True
    controller._note_split_hold(scheduler, kind, request)
    return True


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
    'blocked_deferred', 'decode_merge_cap', 'defer_request_free', 'defer_stock_positive_frontier',
    'defer_stock_prefix_hit', 'defer_stock_running_prefill',
    'flash_s5_admit',
    'generate_block_pool', 'generate_scheduler', 'hold_leased_blocks',
    'note_allocated', 'note_cow_retained', 'note_source_owners',
    'parked_count', 'revert_block_pool', 'revert_scheduler', 'source_hold',
]
