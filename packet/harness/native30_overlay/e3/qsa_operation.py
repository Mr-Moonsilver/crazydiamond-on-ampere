"""Bounded resumable QSA prefill operation: producer -> CPU-source DMA -> attention.

This is the worker-side operation the generated layer call sites drive.  It is
deliberately not a wrapper around the whole model forward:

* ``begin_layer`` / ``producer_submitted`` / ``end_layer`` are the producer
  boundary (``index_and_store``) in the pinned QSA layer.
* ``bind_plan`` + ``fill_bucket`` are the bounded CPU-source DMA boundary
  (``history_ready``).  Every copy is at most :data:`MAX_PAGES_PER_COPY` pages
  and at most one :data:`ARENA_PAGES`-page bucket is in flight per rank.
  Its credit returns only after the attention event is host-queried complete.
* ``after_attention_bucket`` records the attention consumer event
  (``attend_and_project``) after the pinned kernel was enqueued.

Nothing here synchronizes: copies are enqueued on a private copy stream, the
attention stream waits on the copy event, and :meth:`QsaPrefillOperation.poll`
only calls ``event.query()``.  A poll never launches the remaining forward and
never reads a tensor back.  The operation retains the actual native tensor
owners (registered CPU pool, existing arena, block-table row, producer inputs
and attention outputs), not address integers alone; they are dropped only by
:meth:`release` after every consumer event has been observed.

The explicit API for the later scheduler/transport worker is::

    attach_continuation -> retains generated attention frame
    advance              -> at most one bucket enqueue; pending keeps frame intact
    submit (fill_bucket) -> accepted OperationTicket after bounded enqueue
    poll                 -> specific completed stage proofs (nonblocking)
    resume               -> validate transaction/lane/allocation/source epochs
    cancel               -> stop new submissions, retain consumers until drained
    release              -> only after drain; drops tensor owners

A stage ticket is *not* an RPC acknowledgment and its completion is *not* a
commit.  Only :class:`e3_types.CompletionKind` values are emitted; this module
never invents a state-write or PP consumer.

This module imports neither torch nor vLLM; the device port imports torch only
when the real :class:`NativeQsaDevice` is constructed, so the CPU hub can
execute the real state machine with delayed fake events.
"""
from dataclasses import dataclass, field
from enum import Enum
import math
import time

from e3_types import (
    ARENA_PAGES,
    CompletionKind,
    CompletionProof,
    DependencyRole,
    E3ContractError,
    E3Pending,
    LaneId,
    OperationTicket,
    TensorFacts,
    TransactionId,
)

# Bounds from docs/e3-briefs/worker-native-operations.md. One bucket (24 pages)
# outstanding per rank, at most four pages per DMA copy.
MAX_PAGES_PER_COPY = 4
MAX_BUCKET_PAGES = ARENA_PAGES
MAX_OUTSTANDING_BUCKETS = 1
MAX_LAYER_PAGES = 128
MAX_STAGES_PER_LAYER = 1 + 2 * math.ceil(MAX_LAYER_PAGES / MAX_BUCKET_PAGES)
MAX_TRACKED_TICKETS = 512
# Generated runner method the worker server calls for one parked chunk; kept
# here (an E3-packaged module) so the build-time generator and the runtime
# server share one name.
RUNNER_SEAM = '_flash_s5_execute_prefill_operation'


class OperationError(E3ContractError):
    """The operation contract was violated (identity, ordering or geometry)."""


class OperationCancelled(OperationError):
    """A cancelled operation refuses new submissions."""


class Stage(str, Enum):
    """The three QSA boundaries the design names, in submission order."""
    PRODUCER = 'index_and_store'
    HISTORY_READY = 'history_ready'
    ATTENTION = 'attend_and_project'


_STAGE_ROLE = {
    Stage.PRODUCER: DependencyRole.PRODUCER,
    Stage.HISTORY_READY: DependencyRole.PRODUCER,   # writes the arena
    Stage.ATTENTION: DependencyRole.CONSUMER,       # reads the arena
}
_STAGE_PROOF = {
    Stage.PRODUCER: CompletionKind.GPU_EVENT,
    Stage.HISTORY_READY: CompletionKind.COPY_EVENT,
    Stage.ATTENTION: CompletionKind.GPU_EVENT,
}


@dataclass
class StageRecord:
    """One bounded submission plus the native owners that keep it valid."""
    ticket: OperationTicket
    stage: Stage
    layer_name: str
    bucket: int
    event: object = field(default=None, repr=False)
    complete: bool = False
    failed: object = field(default=None, repr=False)
    owners: tuple = field(default=(), repr=False)
    pages: int = 0
    page_bytes: int = 0
    pieces: tuple = ()

    @property
    def bytes(self):
        return self.pages * self.page_bytes


@dataclass(frozen=True)
class BucketPlan:
    index: int
    lo: int
    hi: int
    sources: tuple = field(repr=False)  # physical pages in logical order
    pieces: tuple = field(repr=False)   # (source_page, arena_page, pages)

    @property
    def pages(self):
        return self.hi - self.lo


@dataclass
class LayerRun:
    layer_name: str
    source: object = field(repr=False)
    source_facts: TensorFacts
    allocation_epoch: int
    source_epoch: int
    buckets: tuple = ()
    arena: object = field(default=None, repr=False)
    arena_facts: object = None
    table: object = field(default=None, repr=False)
    table_facts: object = None
    producer: object = None
    producer_submitted: bool = False
    next_bucket: int = 0
    active_bucket: object = None
    consumed: list = field(default_factory=list)
    finished: bool = False
    _pending_consumer: object = None


def plan_buckets(pages, *, bucket_pages=MAX_BUCKET_PAGES,
                 max_piece=MAX_PAGES_PER_COPY):
    """Split a dense logical page list into bounded buckets and copies.

    Coalesce only adjacent physical source pages into one copy.  Real block
    tables are fragmented, so the piece count is data-dependent; 50 pages at
    200k gives the baseline 24/24/2 bucket order.
    """
    if type(bucket_pages) is not int or bucket_pages < 1 or bucket_pages > MAX_BUCKET_PAGES:
        raise OperationError('invalid bucket page bound')
    if type(max_piece) is not int or max_piece < 1 or max_piece > MAX_PAGES_PER_COPY:
        raise OperationError('invalid per-copy page bound')
    pages = tuple(pages)
    if not pages or len(pages) > MAX_LAYER_PAGES or any(type(p) is not int or p < 0 for p in pages):
        raise OperationError('dense page plan must be non-empty nonnegative ints')
    buckets = []
    for index, lo in enumerate(range(0, len(pages), bucket_pages)):
        hi = min(lo + bucket_pages, len(pages))
        span = pages[lo:hi]
        pieces = []
        i = 0
        while i < len(span):
            count = 1
            while (i + count < len(span) and count < max_piece
                   and span[i + count] == span[i] + count):
                count += 1
            pieces.append((span[i], i, count))
            i += count
        buckets.append(BucketPlan(index, lo, hi, span, tuple(pieces)))
    return tuple(buckets)


def metadata_plan_provider(snapshot, *, layer_tables=None):
    """Bind explicit complete per-layer CPU tables; never guess from flat IDs.

    The engine/worker admission adapter must discover the QSA KV group and
    supply its complete logical table for each local layer. The old snapshot's
    flat block_ids may include state/arena groups and is not a history map.
    Tables alone do not prove source leases/incarnations; that native admission
    binding remains required before enabled startup can be allowed.
    """
    if snapshot is None or len(getattr(snapshot, 'rows', ()) or ()) != 1:
        raise OperationError('CPU page plan needs a single-row admission snapshot')
    if type(layer_tables) is not tuple or not layer_tables:
        raise OperationError('explicit per-layer CPU tables required; flat block ids refused')
    bound = {}
    for entry in layer_tables:
        if type(entry) is not tuple or len(entry) != 2:
            raise OperationError('invalid per-layer table entry')
        name, pages = entry
        if (type(name) is not str or not name or name in bound
                or type(pages) is not tuple or not pages
                or len(pages) > MAX_LAYER_PAGES
                or any(type(p) is not int or p < 0 for p in pages)):
            raise OperationError('invalid or duplicate per-layer CPU table')
        bound[name] = pages

    def provide(layer_name, num_pages):
        if type(num_pages) is not int or not 1 <= num_pages <= MAX_LAYER_PAGES:
            raise OperationError('invalid dense page count')
        pages = bound.get(layer_name)
        if pages is None or len(pages) < num_pages:
            raise OperationError('per-layer CPU table does not cover dense history')
        return pages[:num_pages]
    return provide


class NativeQsaDevice:
    """torch device port used in production; never synchronizes.

    ``wait_event`` installs a stream dependency (``Stream.wait_event``), not a
    host wait.  ``copy_pages`` issues one bounded H2D copy from the registered
    CPU pool to the existing arena on the private copy stream.
    """

    compute = 'compute'
    copy = 'copy'

    def __init__(self, device, *, clock=time.monotonic):
        import torch
        self.torch = torch
        # Accept the string/device forms the runner reports; constructing a
        # torch.device on a string is unnecessary and would make the CPU
        # fixtures' minimal torch module part of the contract.
        self.device = str(device)
        if not self.device.startswith('cuda:'):
            raise OperationError('explicit CUDA device ordinal required')
        self.clock = clock
        self._copy_stream = None
        self._compute_stream = None

    def bind_streams(self, compute, copy):
        """Bind actual lane streams once; never fall back to caller's current stream."""
        if self._compute_stream is not None or self._copy_stream is not None:
            raise OperationError('operation streams already bound')
        for stream in (compute, copy):
            if (str(getattr(stream, 'device', None)) != self.device
                    or type(getattr(stream, 'cuda_stream', None)) is not int):
                raise OperationError('explicit lane stream on the owning device required')
        if compute.cuda_stream == copy.cuda_stream:
            raise OperationError('compute and copy streams must be distinct')
        self._compute_stream, self._copy_stream = compute, copy

    def _stream(self, token):
        if self._compute_stream is None or self._copy_stream is None:
            raise OperationError('native lane streams have not been bound')
        if token == self.compute:
            return self._compute_stream
        if token == self.copy:
            return self._copy_stream
        raise OperationError('unknown operation stream token')

    def validate_compute(self):
        if self.torch.cuda.current_stream(self.device).cuda_stream != self._stream(self.compute).cuda_stream:
            raise OperationError('operation called outside its owning compute stream')

    def stream_token(self, token):
        stream = self._stream(token)
        value = getattr(stream, 'cuda_stream', None)
        if type(value) is not int:
            raise OperationError('stream has no device token')
        return value

    def facts(self, tensor):
        return TensorFacts(str(tensor.device), tuple(tensor.shape),
                           tuple(tensor.stride()), str(tensor.dtype),
                           tensor.element_size(), tensor.is_pinned(),
                           tensor.is_contiguous(), tensor.data_ptr())

    def record(self, token):
        event = self.torch.cuda.Event(enable_timing=False)
        with self.torch.cuda.device(self.device):
            event.record(self._stream(token))
        return event

    def query(self, event):
        if str(event.device) != self.device:
            raise OperationError('completion event belongs to another device')
        return bool(event.query())

    def wait_event(self, token, event):
        if str(event.device) != self.device:
            raise OperationError('stream wait on a foreign-device event')
        self._stream(token).wait_event(event)

    def copy_pages(self, source, arena, source_page, arena_page, pages):
        if type(pages) is not int or not 1 <= pages <= MAX_PAGES_PER_COPY:
            raise OperationError('copy exceeds the four-page bound')
        with self.torch.cuda.device(self.device), self.torch.cuda.stream(self._stream(self.copy)):
            dst = arena[arena_page:arena_page + pages]
            src = source[source_page:source_page + pages]
            dst.copy_(src, non_blocking=True)


class QsaPrefillOperation:
    """One transaction's bounded QSA work, driven by generated call sites."""

    def __init__(self, transaction, lane, *, device, clock=time.monotonic,
                 plan_provider=None, allocation_epoch=None, source_epochs=None,
                 max_tickets=MAX_TRACKED_TICKETS):
        if type(transaction) is not TransactionId:
            raise OperationError('operation needs a TransactionId')
        if type(lane) is not LaneId:
            raise OperationError('operation needs an immutable LaneId')
        if device is None:
            raise OperationError('QSA operation needs a device port')
        if type(max_tickets) is not int or max_tickets < 1 or max_tickets > MAX_TRACKED_TICKETS:
            raise OperationError('invalid tracked-ticket bound')
        self.transaction = transaction
        self.lane = lane
        self.device = device
        self.clock = clock
        self.plan_provider = plan_provider
        self.allocation_epoch = (lane.storage_epoch if allocation_epoch is None
                                 else allocation_epoch)
        if type(self.allocation_epoch) is not int or self.allocation_epoch != lane.storage_epoch:
            raise OperationError('allocation epoch must match the lane storage epoch')
        self.source_epochs = dict(source_epochs or {})
        for name, epoch in self.source_epochs.items():
            if not isinstance(name, str) or not name or type(epoch) is not int or epoch < 0:
                raise OperationError('invalid source epoch entry')
        self.max_tickets = max_tickets
        self._records = []
        self._owners = {}
        self._layers = {}
        self._op_seq = 0
        self._completion_seq = 0
        self._ready_proofs = []
        self._failed = False
        self._released = False
        self._continuation = None
        self._cancelled = None
        self._forward_open = False
        self._forward_layers = []
        self._last_arena_consumer = None
        self._started_s = clock()
        self.stats = dict(producers=0, buckets=0, pieces=0, bytes=0,
                          copy_events=0, consumer_events=0, waits=0)

    # --- identity / ownership validation ---------------------------------
    def __repr__(self):
        return (f'QsaPrefillOperation(txn_seq={self.transaction.txn_seq}, '
                f'rank={self.lane.worker_rank}, storage_epoch={self.lane.storage_epoch}, '
                f'layers={len(self._layers)}, pending={self.pending})')

    def _check_identity(self, *, layer_name=None, lane=None,
                        allocation_epoch=None, source_epoch=None):
        validate = getattr(self.device, 'validate_compute', None)
        if callable(validate):
            validate()
        if self._released or self._failed:
            raise OperationError('operation released or quarantined')
        if self._cancelled is not None:
            raise OperationCancelled('operation was cancelled')
        if lane is not None and lane != self.lane:
            raise OperationError('stale lane identity')
        if self.lane.storage_epoch != self.allocation_epoch:
            raise OperationError('lane storage epoch changed after binding')
        if allocation_epoch is not None and allocation_epoch != self.allocation_epoch:
            raise OperationError('stale allocation epoch')
        if layer_name is not None and source_epoch is not None:
            if layer_name not in self.source_epochs:
                raise OperationError('unknown source epoch for ' + layer_name)
            if source_epoch != self.source_epochs[layer_name]:
                raise OperationError('stale source epoch for ' + layer_name)
        if layer_name is not None and layer_name not in self._layers:
            raise OperationError('operation layer is not bound: ' + layer_name)

    def register_source(self, layer_name, source, *, source_epoch=0):
        """Bind the real registered CPU pool for a layer; keep the tensor owner."""
        if not isinstance(layer_name, str) or not layer_name:
            raise OperationError('layer name required')
        if type(source_epoch) is not int or source_epoch < 0:
            raise OperationError('source epoch must be a nonnegative int')
        self._check_identity()
        if layer_name in self.source_epochs and self.source_epochs[layer_name] != source_epoch:
            raise OperationError('source epoch disagrees with admission')
        facts = self.device.facts(source)
        if facts.device != 'cpu' or not facts.pinned or facts.dtype != 'torch.bfloat16':
            raise OperationError('QSA operation source must be a pinned BF16 CPU pool')
        if (len(facts.shape) != 4 or facts.shape[1:] != (1, 4048, 512)
                or facts.shape[0] < 1 or facts.item_bytes != 2 or not facts.contiguous):
            raise OperationError('invalid QSA source geometry')
        previous = self._layers.get(layer_name)
        if previous is not None:
            if previous.source_facts != facts or previous.source_epoch != source_epoch:
                raise OperationError('source allocation changed after binding')
            previous.source = source
            return previous
        run = LayerRun(layer_name, source, facts, self.allocation_epoch, source_epoch)
        self._layers[layer_name] = run
        self.source_epochs[layer_name] = source_epoch
        self._owners[('source', layer_name)] = (source,)
        return run

    def _layer(self, layer_name):
        run = self._layers.get(layer_name)
        if run is None:
            raise OperationError('operation layer is not registered: ' + str(layer_name))
        return run

    def _new_ticket(self, stage, layer_name, bucket):
        if len(self._records) >= self.max_tickets:
            raise OperationError('bounded operation-ticket budget exhausted')
        role = _STAGE_ROLE[stage]
        ticket = OperationTicket(self.transaction, self.lane.worker_rank,
                                 self.lane, self._op_seq, role)
        self._op_seq += 1
        record = StageRecord(ticket, stage, layer_name, bucket)
        if len(self._records) >= self.max_tickets:
            raise OperationError('bounded operation-ticket budget exhausted')
        self._records.append(record)
        return record

    # --- forward / layer boundaries --------------------------------------
    def begin_forward(self, *, layers=()):
        """Mark the start of one host submission; no work is launched here."""
        self._check_identity()
        if self._forward_open:
            raise OperationError('a prefill host submission is already open')
        if self._cancelled is not None:
            raise OperationCancelled('cancelled operation cannot begin a forward')
        if self._records:
            raise OperationError("operation forward cannot be restarted")
        names = tuple(layers)
        if names:
            for name in names:
                self._layer(name)
            if set(names) != set(self._layers):
                raise OperationError('host submission layers differ from bound QSA layers')
        self._forward_open = True
        self._forward_layers = []
        for run in self._layers.values():
            run.finished = False
        return self

    def begin_layer(self, layer_name, source=None, *, lane=None,
                    allocation_epoch=None, source_epoch=None):
        """Producer boundary: validate and reserve a producer ticket.

        The pinned layer performs ``index_and_store`` immediately afterwards;
        ``producer_submitted`` records its completion event.
        """
        self._check_identity(layer_name=layer_name, lane=lane,
                             allocation_epoch=allocation_epoch,
                             source_epoch=source_epoch)
        run = self._layers.get(layer_name)
        if source is not None:
            run = self.register_source(layer_name, source, source_epoch=run.source_epoch)
        elif run is None:
            raise OperationError('producer boundary needs the registered CPU source')
        if not self._forward_open:
            raise OperationError('layer began outside an open host submission')
        if self._forward_layers and not self._current_run().finished:
            raise OperationError('previous layer is unfinished')
        if run.finished or run.producer is not None:
            raise OperationError('layer already ran in this transaction')
        if layer_name in self._forward_layers:
            raise OperationError('layer submitted twice in one forward')
        run.producer = self._new_ticket(Stage.PRODUCER, layer_name, -1)
        self._forward_layers.append(layer_name)
        return run.producer.ticket

    def producer_submitted(self, layer_name, *owners):
        """Record the compute-stream event after index_and_store was enqueued."""
        run = self._layer(layer_name)
        if run.producer is None or run.producer_submitted:
            raise OperationError('producer event without a reserved producer ticket')
        if not any(o is not None for o in owners):
            raise OperationError('producer completion must name its native owners')
        run.producer.owners = tuple(o for o in owners if o is not None)
        try:
            run.producer.event = self.device.record(self.device.compute)
        except BaseException:
            self._failed = True
            raise
        run.producer_submitted = True
        self.stats['producers'] += 1
        return run.producer.ticket

    def bind_plan(self, layer_name, block_table_row, num_pages, valid_tokens,
                  arena, *, lane=None, allocation_epoch=None, source_epoch=None):
        """Bind the dense CPU page plan, arena and GPU block-table row.

        The physical pages must come from a CPU-owned mapping (admission); the
        GPU table row is bound only as an identity/geometry owner.  Reading it
        back would synchronize and is refused.
        """
        self._check_identity(layer_name=layer_name, lane=lane,
                             allocation_epoch=allocation_epoch,
                             source_epoch=source_epoch)
        run = self._layer(layer_name)
        if run.producer is None or not run.producer_submitted:
            raise OperationError('history plan needs a submitted producer')
        if type(num_pages) is not int or not 1 <= num_pages <= MAX_LAYER_PAGES:
            raise OperationError('dense plan page count is out of bounds')
        if type(valid_tokens) is not int or valid_tokens < 1:
            raise OperationError('valid token count must be positive')
        if (valid_tokens + 4048 - 1) // 4048 != num_pages:
            raise OperationError('page plan disagrees with the valid token count')
        arena_facts = self.device.facts(arena)
        if (not arena_facts.device.startswith('cuda:')
                or arena_facts.dtype != 'torch.bfloat16'
                or len(arena_facts.shape) != 4
                or arena_facts.shape != (MAX_BUCKET_PAGES, 1, 4048, 512)
                or arena_facts.item_bytes != 2
                or arena_facts.device != self.device.device
                or arena_facts.shape[2] != 4048
                or not arena_facts.contiguous):
            raise OperationError('invalid existing arena')
        table_facts = self.device.facts(block_table_row)
        if (table_facts.device != arena_facts.device
                or len(table_facts.shape) != 1
                or table_facts.shape[0] < num_pages
                or table_facts.dtype not in ('torch.int32', 'torch.int64')
                or table_facts.strides != (1,) or not table_facts.contiguous):
            raise OperationError('invalid worker GPU block-table row')
        pages = self._plan(layer_name, num_pages)
        if run.buckets:
            if run.arena_facts != arena_facts or run.table_facts != table_facts:
                raise OperationError('arena or block-table allocation changed')
            if run.buckets != plan_buckets(pages):
                raise OperationError('page plan changed during the layer')
            return run.buckets
        for page in pages:
            if not 0 <= page < run.source_facts.shape[0]:
                raise OperationError('physical source page outside the pinned pool')
        run.arena, run.arena_facts = arena, arena_facts
        run.table, run.table_facts = block_table_row, table_facts
        run.buckets = plan_buckets(pages)
        if len(run.buckets) > MAX_STAGES_PER_LAYER:
            raise OperationError('layer exceeds the bounded stage budget')
        self._owners[('arena', layer_name)] = (arena,)
        self._owners[('table', layer_name)] = (block_table_row,)
        return run.buckets

    def _plan(self, layer_name, num_pages):
        if self.plan_provider is None:
            raise OperationError(
                'no CPU-owned physical page mapping for ' + layer_name +
                '; reading the GPU block table would synchronize')
        pages = tuple(self.plan_provider(layer_name, num_pages))
        if len(pages) != num_pages:
            raise OperationError('plan provider returned the wrong page count')
        if any(type(p) is not int or p < 0 for p in pages):
            raise OperationError('plan provider returned invalid pages')
        return pages

    # --- bounded DMA / attention boundaries -------------------------------
    def fill_bucket(self, staged, page_lo, page_hi, block_table_row):
        """Enqueue one bounded bucket of <=4-page copies; never synchronizes.

        Called from the generated attention function in place of the stock
        ``torch.index_select``.  The copy stream is ordered behind the producer
        event and behind the previous arena consumer before the arena is
        reused; the attention stream is ordered behind the copy event.
        """
        run = self._current_run()
        self._check_identity(layer_name=run.layer_name)
        if self.device.facts(run.source) != run.source_facts:
            raise OperationError("source storage changed after binding")
        if not run.producer_submitted:
            raise OperationError('bucket fill before the producer boundary')
        # Only one bucket may be outstanding per rank: the next fill is a
        # retry-after-completion, not a host wait or an unbounded queue.
        self._refresh()
        if self.pending_buckets >= MAX_OUTSTANDING_BUCKETS:
            raise E3Pending('one arena bucket is still outstanding')
        index = run.next_bucket
        if index >= len(run.buckets):
            raise OperationError('more buckets than the bound plan')
        bucket = run.buckets[index]
        if (bucket.lo, bucket.hi) != (page_lo, page_hi):
            raise OperationError('attention bucket order differs from the plan')
        if self.device.facts(block_table_row) != run.table_facts:
            raise OperationError('GPU block-table view changed')
        facts = self.device.facts(staged)
        if (facts.device != run.arena_facts.device
                or facts.shape != (bucket.pages, *run.arena_facts.shape[1:])
                or facts.strides != run.arena_facts.strides
                or facts.dtype != run.arena_facts.dtype
                or not facts.contiguous
                or facts.address != run.arena_facts.address):
            raise OperationError('wrong dense bucket destination')
        page_bytes = (run.arena_facts.shape[1] * run.arena_facts.shape[2]
                      * run.arena_facts.shape[3] * run.arena_facts.item_bytes)
        if len(self._records) + 2 > self.max_tickets:
            raise OperationError('bounded operation-ticket budget exhausted')
        copy = self._new_ticket(Stage.HISTORY_READY, run.layer_name, index)
        consumer = self._new_ticket(Stage.ATTENTION, run.layer_name, index)
        copy.pages = bucket.pages
        copy.page_bytes = page_bytes
        copy.pieces = bucket.pieces
        copy.owners = (run.source, run.arena, run.table)
        run.active_bucket = index
        run._pending_consumer = consumer
        try:
            # Producer visibility and arena reuse are GPU dependencies, not host
            # waits.  The first bucket waits for the store and for the last
            # consumer of this shared arena; later buckets wait for the previous
            # consumer of this layer.
            if index == 0:
                self.device.wait_event(self.device.copy, run.producer.event)
                self.stats['waits'] += 1
                if self._last_arena_consumer is not None:
                    self.device.wait_event(self.device.copy, self._last_arena_consumer)
                    self.stats['waits'] += 1
            else:
                self.device.wait_event(self.device.copy, run.consumed[-1].event)
                self.stats['waits'] += 1
            for source_page, arena_page, pages in bucket.pieces:
                self.device.copy_pages(run.source, run.arena, source_page,
                                       arena_page, pages)
            copy.event = self.device.record(self.device.copy)
            self.stats['buckets'] += 1
            self.stats['pieces'] += len(bucket.pieces)
            self.stats['bytes'] += copy.bytes
            self.stats['copy_events'] += 1
            self.device.wait_event(self.device.compute, copy.event)
            self.stats['waits'] += 1
        except BaseException:
            self._failed = True
            raise
        return copy.ticket

    def retain_attention(self, *owners):
        """Retain kernel arguments before launch, including on launch failure."""
        run = self._current_run()
        if run._pending_consumer is None:
            raise OperationError('attention has no filled bucket')
        run._pending_consumer.owners = tuple(owners)

    def after_attention_bucket(self, *owners):
        """Record the consumer event immediately after the pinned kernel launch.

        A cancel stops *new* submissions but never strands work that was
        already submitted: the consumer event of a filled bucket may still be
        recorded so the arena can be retired by polling.
        """
        run = self._current_run()
        if self._cancelled is None:
            self._check_identity(layer_name=run.layer_name)
        elif run._pending_consumer is None:
            raise OperationCancelled(
                'cancelled operation cannot submit another attention bucket')
        consumer = run._pending_consumer
        if run.active_bucket is None or consumer is None:
            raise OperationError('consumer event without a filled bucket')
        if not any(o is not None for o in owners):
            raise OperationError('consumer completion must name its native owners')
        consumer.owners = tuple(o for o in owners if o is not None)
        try:
            consumer.event = self.device.record(self.device.compute)
        except BaseException:
            self._failed = True
            raise
        run.consumed.append(consumer)
        run.active_bucket = None
        run._pending_consumer = None
        run.next_bucket += 1
        self._last_arena_consumer = consumer.event
        self.stats['consumer_events'] += 1
        return consumer.ticket

    def end_layer(self, layer_name):
        """Close a layer after attention submission; no completion claim.

        Closing is bookkeeping, not a submission, so a cancelled operation may
        still close a layer whose every bucket was already retired.  It may
        never close one with an outstanding bucket or consumer event.
        """
        run = self._layer(layer_name)
        if self._cancelled is None:
            self._check_identity(layer_name=layer_name)
        elif (run.active_bucket is not None or run._pending_consumer is not None
              or not run.buckets or run.next_bucket != len(run.buckets)):
            raise OperationCancelled(
                'cancelled layer cannot close before its buckets retire')
        if run.producer is None or not run.producer_submitted:
            raise OperationError('layer ended without its producer boundary')
        if not run.consumed or run.next_bucket != len(run.buckets):
            raise OperationError('layer ended before every bucket was consumed')
        if run.active_bucket is not None or run._pending_consumer is not None:
            raise OperationError('layer ended with an outstanding consumer')
        run.finished = True
        return run

    def end_forward(self):
        """Every bound layer must have run exactly once; submission is bounded."""
        if not self._forward_open:
            raise OperationError('no host submission is open')
        missing = [name for name, run in self._layers.items() if not run.finished]
        if missing:
            self._failed = True
            raise OperationError('layer(s) did not run through the operation: '
                                 + ','.join(sorted(missing)))
        self._forward_open = False
        return tuple(self._forward_layers)

    # --- explicit continuation API ----------------------------------------
    def resume(self, layer_name, stage, *, lane=None, allocation_epoch=None,
               source_epoch=None):
        """Validate identity/allocation/source epochs before advancing a stage.

        The generated stream-ordered path advances inside the pinned layer;
        this method is the contract the later host dispatcher uses when it
        needs an explicit poll/resume boundary.  It never launches the
        remaining forward: it only admits the named next stage for this layer.
        """
        self._check_identity(layer_name=layer_name, lane=lane,
                             allocation_epoch=allocation_epoch,
                             source_epoch=source_epoch)
        run = self._layer(layer_name)
        if type(stage) is not Stage:
            raise OperationError('resume needs an OperationStage')
        if stage is Stage.HISTORY_READY:
            if not run.producer_submitted:
                raise E3Pending('producer has not been submitted')
            if run.producer.event is None or not self.device.query(run.producer.event):
                raise E3Pending('producer completion is not observed yet')
            if run.active_bucket is not None:
                raise E3Pending('a bucket is still filled or consumed')
            return run.next_bucket, run.buckets[run.next_bucket:run.next_bucket + 1]
        if stage is Stage.ATTENTION:
            if run.active_bucket is None:
                raise OperationError('no filled bucket to attend')
            return run.active_bucket, run._pending_consumer.ticket
        raise OperationError('unknown resume stage')

    def attach_continuation(self, continuation):
        """Own the generated attention frame until completion/cancel retirement."""
        import types
        self._check_identity()
        if self._continuation is not None or type(continuation) is not types.GeneratorType:
            raise OperationError('one generated continuation required')
        self._continuation = continuation

    def advance(self):
        """Submit at most one bucket; pending credit leaves the frame untouched.

        The caller must reinstall the lane's forward/stream context each time.
        Poll never calls this method. Completion of the final bucket still
        needs a subsequent observed proof before StopIteration is returned.
        """
        self._check_identity()
        self._refresh()
        if self.pending_buckets:
            raise E3Pending('previous bucket consumer has not completed')
        if self._continuation is None:
            raise OperationError('no generated continuation attached')
        try:
            return next(self._continuation)
        except StopIteration:
            self._continuation = None
            raise
        except BaseException:
            self._failed = True
            raise

    # --- poll / cancel / drain / release ----------------------------------
    def _refresh(self):
        """Nonblocking completion query.  Never synchronizes or launches work."""
        completed = []
        if self._failed or self._released:
            return ()
        for record in self._records:
            if record.complete or record.failed is not None:
                continue
            if record.event is None:
                continue
            try:
                ready = self.device.query(record.event)
            except BaseException:
                self._failed = True
                raise
            if ready:
                record.complete = True
                self._completion_seq += 1
                completed.append(CompletionProof(record.ticket,
                                                 _STAGE_PROOF[record.stage],
                                                 self._completion_seq))
        self._ready_proofs.extend(completed)

    def poll(self):
        self._refresh()
        proofs, self._ready_proofs = tuple(self._ready_proofs), []
        return proofs

    @property
    def pending(self):
        return sum(1 for record in self._records if not record.complete)

    @property
    def pending_buckets(self):
        return int(any(record.stage is Stage.ATTENTION and not record.complete
                       for record in self._records))

    @property
    def cancelled(self):
        return self._cancelled

    def cancel(self, reason='cancelled'):
        """Stop new submissions; submitted consumers are retained until drained."""
        if not isinstance(reason, str) or not reason:
            raise OperationError('cancel reason required')
        if self._cancelled is None:
            self._cancelled = reason
        self._forward_open = False
        return self._cancelled

    def drain(self):
        """Query outstanding events without waiting; True once everything is done."""
        self._refresh()
        return (not self._failed and self.pending == 0
                and (self._cancelled is not None or not self._forward_open))

    def release(self):
        """Drop retained native owners only after every consumer completed."""
        self._refresh()
        if self._failed or self.pending:
            raise E3Pending('cannot release outstanding or quarantined operation')
        if self._forward_open and self._cancelled is None:
            raise E3Pending('cannot release an open forward')
        opened = [name for name, run in self._layers.items()
                  if run.producer is not None and not run.finished]
        if opened and self._cancelled is None:
            raise E3Pending('cannot release an unfinished layer: ' + ','.join(sorted(opened)))
        dropped = len(self._owners)
        self._released = True
        if self._continuation is not None:
            self._continuation.close()
            self._continuation = None
        self._owners.clear()
        for record in self._records:
            record.owners = ()
        for run in self._layers.values():
            run.source = run.arena = run.table = None
        return dropped

    # --- inventory --------------------------------------------------------
    def inventory(self):
        """Metadata-only estimates; no device query and no readback."""
        page_bytes = 0
        for run in self._layers.values():
            if run.arena_facts is not None:
                page_bytes = (run.arena_facts.shape[1] * run.arena_facts.shape[2]
                              * run.arena_facts.shape[3] * run.arena_facts.item_bytes)
                break
        return dict(
            layers=len(self._layers), tickets=len(self._records),
            pending=self.pending, buckets=self.stats['buckets'],
            pieces=self.stats['pieces'], bytes=self.stats['bytes'],
            page_bytes=page_bytes,
            max_pages_per_copy=MAX_PAGES_PER_COPY,
            max_bucket_pages=MAX_BUCKET_PAGES,
            max_outstanding_buckets=MAX_OUTSTANDING_BUCKETS,
            max_layer_pages=MAX_LAYER_PAGES,
            basis='estimate; no transfer timing, SM measurement or readback',
        )

    def _current_run(self):
        if len(self._forward_layers) == 0:
            raise OperationError('no layer is being submitted')
        return self._layers[self._forward_layers[-1]]


__all__ = [
    'MAX_BUCKET_PAGES', 'MAX_LAYER_PAGES', 'MAX_OUTSTANDING_BUCKETS',
    'MAX_PAGES_PER_COPY', 'MAX_STAGES_PER_LAYER', 'BucketPlan', 'LayerRun',
    'NativeQsaDevice', 'OperationCancelled', 'OperationError',
    'QsaPrefillOperation', 'RUNNER_SEAM', 'Stage', 'StageRecord',
    'metadata_plan_provider', 'plan_buckets',
]
