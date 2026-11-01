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
import os
import json
import hashlib
import logging
from functools import wraps

from sparse_history_dma import (
    MAX_DESCRIPTORS,
    MAX_LAYER_ROWS,
    MAX_PIECE_ROWS,
    PAGE_BYTES,
    PAGE_ROWS,
    SourceRefused,
    SparseFallbackRequired,
    SparsePlanError,
    SparsePlanRefused,
    SparseRefusal,
    SparseSourceHold,
    SparseStreamBucket,
    make_sparse_plan,
    stream_sparse_plan,
)
from row1_oracle import Row1Bucket, Row1OracleError
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
# Explicit byte bound for the one operation-owned union buffer set: three
# int32 vectors over the dense selection capacity, two int32 block totals and
# the two-word count/route record, plus the inverse-remap buffer.  A geometry
# that would exceed it is refused, never silently allocated.
MAX_UNION_SCRATCH_BYTES = 64 * 1024 * 1024
UNION_BLOCK = 1024
# Pinned readback bound: the whole bounded selection plus the two-word info
# record.  Host slicing below a count read from completed pinned memory is
# allowed; the allocation itself never depends on a device value.
SELECTION_HOST_BYTES = MAX_LAYER_ROWS * 4 + 2 * 4
# Generated runner method the worker server calls for one parked chunk; kept
# here (an E3-packaged module) so the build-time generator and the runtime
# server share one name.
RUNNER_SEAM = '_flash_s5_execute_prefill_operation'


class OperationError(E3ContractError):
    """The operation contract was violated (identity, ordering or geometry)."""


class OperationCancelled(OperationError):
    """A cancelled operation refuses new submissions."""


class Stage(str, Enum):
    """The QSA boundaries the design names, in submission order."""
    PRODUCER = 'index_and_store'
    SELECTION = 'selection_readback'
    HISTORY_READY = 'history_ready'
    ATTENTION = 'attend_and_project'


_STAGE_ROLE = {
    Stage.PRODUCER: DependencyRole.PRODUCER,
    # The readback copies the actual GPU selection to pinned CPU scratch.  It
    # is a handoff, not a storage release and not a commit.
    Stage.SELECTION: DependencyRole.HANDOFF,
    Stage.HISTORY_READY: DependencyRole.PRODUCER,   # writes the arena
    Stage.ATTENTION: DependencyRole.CONSUMER,       # reads the arena
}
_STAGE_PROOF = {
    Stage.PRODUCER: CompletionKind.GPU_EVENT,
    Stage.SELECTION: CompletionKind.COPY_EVENT,
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
    # Sparse transport accounting (zero on the dense route).
    sparse: bool = False
    rows: int = 0
    descriptors: int = 0
    nbytes: int = 0

    @property
    def bytes(self):
        return self.pages * self.page_bytes

    @property
    def moved_bytes(self):
        return self.nbytes if self.nbytes else self.bytes


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
    # Dense plan identity retained for the sparse route's source-map proof.
    pages: tuple = field(default=(), repr=False)
    valid_tokens: int = 0
    source_versions: tuple = field(default=(), repr=False)
    sparse_hold: object = field(default=None, repr=False)
    sparse: object = field(default=None, repr=False)
    # Native row1 qualification (serial, opt-in; never shipping performance).
    state_epoch: object = None
    mapping_version: object = None
    uva: object = field(default=None, repr=False)
    uva_facts: object = None
    row1_reference: object = field(default=None, repr=False)
    row1_failed: object = None
    qualification_bucket: object = None
    schedule_boundary: object = None
    schedule_shape: object = None
    schedule_grant: object = None


@dataclass(frozen=True)
class SparseUnionScratch:
    """One operation-owned fixed-capacity union buffer set.

    Every shape is host-known before launch: ``capacity`` is the dense
    selection range, ``rows``/``width`` are the attention row geometry.  The
    buffers are never resized from a device value and never shared between
    operations.  ``nbytes`` is the explicit bound charged to the operation.
    """
    capacity: int
    rows: int
    width: int
    dtype: str
    nbytes: int
    flags: object = field(repr=False)
    ranks: object = field(repr=False)
    tokens: object = field(repr=False)
    totals: object = field(repr=False)
    offsets: object = field(repr=False)
    info: object = field(repr=False)
    remapped: object = field(repr=False)


@dataclass(frozen=True)
class SelectionToken:
    """Opaque handle for one staged selection readback; carries no storage."""
    layer_name: str
    ticket: OperationTicket


@dataclass(frozen=True)
class SparseSelectionPending:
    """Yielded while the bounded D2H selection readback is still in flight."""
    ticket: OperationTicket

    @property
    def sparse_selection_pending(self):
        return True


@dataclass(frozen=True)
class QsaScheduleBoundary:
    """A submitted boundary, never a claim that its copy completed."""
    ticket: OperationTicket
    qsa_schedule_boundary: bool = True


@dataclass(frozen=True)
class SubmissionCredit:
    """Host-queried answer to "may the next bounded submission run now?".

    ``can_submit`` is the only decision; the two counts are the exact pending
    predicates it was derived from, so a caller can report why it is waiting
    without guessing.  A false answer means the lane must return typed pending
    for the unchanged command and mutate nothing.
    """
    can_submit: bool
    reason: str
    pending_buckets: int
    pending_selection: int
    pending_copy_waves: int = 0


@dataclass(frozen=True)
class SparseCopyPending:
    """A bounded copy wave was submitted, but its bucket is unpublished."""
    ticket: OperationTicket
    sparse_copy_pending: bool = True


@dataclass
class SparseLayerRun:
    """Per-layer sparse plan/readback state; built after the producer boundary."""
    mapping: tuple = field(repr=False)
    versions: tuple = field(repr=False)
    epoch: int
    valid_tokens: int
    min_pages: int
    hold: object = field(repr=False)
    null_page: object = None
    plan: object = None
    refusal: object = None
    empty: bool = False
    union: object = field(default=None, repr=False)
    selection: object = None
    selection_facts: object = None
    selection_count: object = None       # distinct selected rows (host-read)
    selection_sparse: object = None      # device threshold flag (host-read)
    selection_token: object = None
    host_tokens: object = field(default=None, repr=False)
    host_info: object = field(default=None, repr=False)
    next_bucket: int = 0
    active_bucket: object = None
    route: object = None                 # 'sparse_dma'|'dense_dma'|'empty'
    wave_iterator: object = field(default=None, repr=False)
    wave_copy: object = None
    wave_event: object = None
    wave_ready: bool = False
    wave_rows: int = 0
    wave_descriptors: int = 0


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

    def __init__(self, device, *, clock=time.monotonic, env=None):
        import torch
        self.torch = torch
        self.native_qsa_policy, self.native_qsa_trace = native_qsa_settings(env)
        declared = os.environ if env is None else env
        self.native_qsa_boundary = declared.get('FLASH_S5_NATIVE_QSA_STOP_AFTER', '0') == '101200'
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

    def boundary_digest(self, tensor, *, selection=False):
        # Diagnostic-only D2H at the final test chunk. No device-wide fence.
        # These hashes are kept outside timed chunks and contain no payloads.
        if tensor.numel() * tensor.element_size() > 64 * 1024 * 1024:
            raise OperationError('native43 boundary tensor exceeds declared 64MiB readback cap')
        started = time.monotonic_ns()
        host = tensor.detach().to(device='cpu').contiguous()
        raw = host.view(self.torch.uint8).numpy()
        result = dict(sha256=hashlib.sha256(memoryview(raw)).hexdigest(),
                      shape=list(host.shape), dtype=str(host.dtype),
                      bytes=host.numel() * host.element_size())
        if selection:
            import numpy as np
            values = host.numpy()[:, :-1]
            valid = values[values >= 0]
            result['selected_count'] = int(np.unique(valid).size)
        result['readback_cpu_ns'] = time.monotonic_ns() - started
        return result

    def timing_event(self, token):
        event = self.torch.cuda.Event(enable_timing=True)
        event.record(self._stream(token))
        return event

    def copy_pages(self, source, arena, source_page, arena_page, pages):
        if type(pages) is not int or not 1 <= pages <= MAX_PAGES_PER_COPY:
            raise OperationError('copy exceeds the four-page bound')
        with self.torch.cuda.device(self.device), self.torch.cuda.stream(self._stream(self.copy)):
            dst = arena[arena_page:arena_page + pages]
            src = source[source_page:source_page + pages]
            dst.copy_(src, non_blocking=True)

    # --- sparse-history transport port (fixed-capacity union, pinned readback)
    def alloc_union_scratch(self, capacity, rows, width, dtype, remapped_like):
        """Allocate the fixed-capacity union buffers on the compute stream.

        No buffer size depends on a device value.  The inverse-remap buffer is
        a clone of the host-known selection buffer (preserving the trailing
        count column exactly as the pinned ``logical_indices.clone()`` did).
        """
        if type(capacity) is not int or not 1 <= capacity <= MAX_LAYER_ROWS:
            raise OperationError('invalid union capacity')
        if type(rows) is not int or not 1 <= rows <= MAX_LAYER_ROWS:
            raise OperationError('invalid union row count')
        if type(width) is not int or width < 1:
            raise OperationError('invalid union width')
        if dtype not in ('torch.int32', 'torch.int64'):
            raise OperationError('union indices must be int32/int64')
        target = self.torch.int32 if dtype == 'torch.int32' else self.torch.int64
        groups = (capacity + UNION_BLOCK - 1) // UNION_BLOCK
        with self.torch.cuda.device(self.device), self.torch.cuda.stream(self._stream(self.compute)):
            flags = self.torch.zeros((capacity,), dtype=self.torch.int32, device=self.device)
            ranks = self.torch.empty((capacity,), dtype=self.torch.int32, device=self.device)
            tokens = self.torch.empty((capacity,), dtype=self.torch.int32, device=self.device)
            totals = self.torch.empty((groups,), dtype=self.torch.int32, device=self.device)
            offsets = self.torch.empty((groups,), dtype=self.torch.int32, device=self.device)
            info = self.torch.empty((2,), dtype=self.torch.int32, device=self.device)
            remapped = remapped_like.clone()
        return dict(flags=flags, ranks=ranks, tokens=tokens, totals=totals,
                    offsets=offsets, info=info, remapped=remapped)

    def alloc_host_indices(self, capacity, dtype):
        """One bounded pinned CPU index pool, reused by every layer."""
        if type(capacity) is not int or not 1 <= capacity <= MAX_LAYER_ROWS:
            raise OperationError('invalid selection scratch capacity')
        if dtype not in ('torch.int32', 'torch.int64'):
            raise OperationError('selection scratch must be int32/int64')
        target = self.torch.int32 if dtype == 'torch.int32' else self.torch.int64
        with self.torch.cuda.device(self.device):
            return self.torch.empty((capacity,), dtype=target, pin_memory=True)

    def copy_indices_to_host(self, source, dest, count):
        """Bounded D2H on the copy stream; ordered by the caller's stream wait."""
        if type(count) is not int or not 1 <= count <= MAX_LAYER_ROWS:
            raise OperationError('selection readback exceeds the bounded scratch')
        with self.torch.cuda.device(self.device), self.torch.cuda.stream(self._stream(self.copy)):
            dest[:count].copy_(source[:count], non_blocking=True)

    def host_indices(self, buffer, count):
        """Read the pinned selection; valid only after its copy event is queried."""
        if type(count) is not int or not 1 <= count <= MAX_LAYER_ROWS:
            raise OperationError('invalid staged selection count')
        if not buffer.is_pinned() or str(buffer.device) != 'cpu':
            raise OperationError('selection readback is not a pinned CPU buffer')
        return tuple(int(value) for value in buffer[:count].tolist())

    def copy_rows(self, source, arena, source_row, arena_row, rows):
        """One bounded compact-row H2D copy; ``rows`` spans <= 4 pages."""
        if type(rows) is not int or not 1 <= rows <= MAX_PIECE_ROWS:
            raise OperationError('row copy exceeds the four-page bound')
        if (type(source_row) is not int or source_row < 0
                or type(arena_row) is not int or arena_row < 0):
            raise OperationError('invalid sparse row offset')
        if source_row + rows > source.shape[0] * PAGE_ROWS:
            raise OperationError('row copy exceeds the pinned source pool')
        if arena_row + rows > arena.shape[0] * PAGE_ROWS:
            raise OperationError('row copy exceeds the arena')
        with self.torch.cuda.device(self.device), self.torch.cuda.stream(self._stream(self.copy)):
            width = source.shape[-1]
            src = source.reshape(-1, width)
            dst = arena.reshape(-1, width)
            dst[arena_row:arena_row + rows].copy_(
                src[source_row:source_row + rows], non_blocking=True)



NATIVE_QSA_POLICY_ENV = 'FLASH_S5_NATIVE_QSA_POLICY'
NATIVE_QSA_TRACE_ENV = 'FLASH_S5_NATIVE_QSA_TRACE'


def native_qsa_settings(env=None):
    """Latch an explicit policy once at worker construction, before ready."""
    env = os.environ if env is None else env
    policy = env.get(NATIVE_QSA_POLICY_ENV, 'current')
    trace = env.get(NATIVE_QSA_TRACE_ENV, '0')
    if policy not in ('current', 'dense') or trace not in ('0', '1'):
        raise OperationError('invalid native QSA startup policy/trace declaration')
    return policy, trace == '1'


class NativeQsaTrace:
    """Bounded per-operation receipts; event elapsed times only after query.

    No CUDA synchronize, scalar readback or wall/device clock conversion.
    Event spans include the stream's submitted work between markers. CPU
    times are method costs, inclusive of validation; they are not GPU time.
    """
    MAX_MARKERS = 512

    def __init__(self, operation):
        self.operation = operation
        self.open_ns = time.monotonic_ns()
        self.rows = {}
        self.pairs = []
        self.opened = {}
        self.markers = 0
        self.dropped = 0

    def row(self):
        name = self.operation._current_run().layer_name
        return self.rows.setdefault(name, dict(layer=name, cpu_plan_ns=0,
            cpu_submit_ns=0, descriptors=0, waves=0, bytes=0,
            union_ms=0.0, copy_ms=0.0, attention_ms=0.0))

    def begin(self, phase, stream):
        if self.markers + 2 > self.MAX_MARKERS:
            self.dropped += 1
            return None
        event = self.operation.device.timing_event(stream)
        self.markers += 2  # Reserve both endpoints, so an end cannot overflow.
        return self.row()['layer'], phase, stream, event

    def end(self, token):
        if token is not None:
            name, phase, stream, start = token
            end = self.operation.device.timing_event(stream)
            self.pairs.append((name, phase, start, end))

    def emit(self):
        incomplete = self.dropped + len(self.opened)
        for name, phase, start, end in self.pairs:
            if self.operation.device.query(end):
                self.rows[name][phase + '_ms'] += float(start.elapsed_time(end))
            else:
                incomplete += 1
        op = self.operation
        released_ns = time.monotonic_ns()
        for name, row in self.rows.items():
            run = op._layers[name]
            sparse = run.sparse
            row.update(schema='native43-qsa-layer/1', rank=op.lane.worker_rank,
                transaction=op.transaction.txn_seq,
                chunk_start=op.transaction.chunk_start,
                chunk_end=op.transaction.chunk_end, history=run.valid_tokens,
                policy=op.transport_policy, operation_open_ns=self.open_ns,
                operation_released_ns=released_ns,
                route=sparse.route if sparse is not None else 'dense_dma',
                selected_count=sparse.selection_count if sparse is not None else None,
                selected_count_basis='union-readback' if sparse is not None
                    and sparse.selection_count is not None else 'NOT_OBSERVED:no-union',
                gpu_timing_complete=incomplete == 0,
                timing_markers=self.markers, dropped_markers=self.dropped,
                timing_basis='stream-event brackets include host launch/idle gaps; not GPU busy time',
                counters_basis='operation-cumulative', counters=dict(op.stats))
            logging.getLogger(__name__).info('flash-s5 native43_qsa %s',
                json.dumps(row, sort_keys=True, separators=(',', ':')))


def sparse_active_for_trace(run):
    return run.sparse is not None and run.sparse.route == 'sparse_dma'


def trace_cpu(field):
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            if self.trace is None:
                return method(self, *args, **kwargs)
            row = self.trace.row()
            started = time.monotonic_ns()
            try:
                return method(self, *args, **kwargs)
            finally:
                row[field] += time.monotonic_ns() - started
        return wrapped
    return decorate

class QsaPrefillOperation:
    """One transaction's bounded QSA work, driven by generated call sites."""

    def __init__(self, transaction, lane, *, device, clock=time.monotonic,
                 plan_provider=None, allocation_epoch=None, source_epochs=None,
                 source_versions=None, max_tickets=MAX_TRACKED_TICKETS):
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
        self.transport_policy = getattr(device, 'native_qsa_policy', 'current')
        if self.transport_policy not in ('current', 'dense'):
            raise OperationError('invalid latched native QSA policy')
        self.trace = NativeQsaTrace(self) if getattr(device, 'native_qsa_trace', False) else None
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
        self.source_versions = {}
        for name, versions in dict(source_versions or {}).items():
            self._check_versions(name, versions)
            self.source_versions[name] = tuple(versions)
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
        self._index_scratch = {}
        self._union_scratch = None
        self._union_scratch_layer = None
        self._union_scratch_key = None
        self._fallback = None
        self._source_release = []
        # The opt-in native row1 byte oracle. ``None`` means qualification is
        # off (shipping): no UVA reference, snapshot or comparison is created.
        self.qualification = None
        self.state_epoch = None
        self.stats = dict(producers=0, buckets=0, pieces=0, bytes=0,
                          copy_events=0, consumer_events=0, waits=0,
                          selection_copies=0, selection_bytes=0,
                          sparse_buckets=0, sparse_rows=0, sparse_pieces=0,
                          sparse_bytes=0, sparse_descriptors=0,
                          sparse_descriptor_high_water=0, sparse_planned=0,
                          sparse_empty=0, sparse_fallback=0,
                          sparse_fallback_rows=0, sparse_fallback_bytes=0,
                          sparse_routes=0, sparse_source_released=0,
                          union_scratch_bytes=0)

    def boundary_evidence(self, tensor, *, selection=False):
        if (self.trace is None or self.transaction.chunk_end != 101200
                or not getattr(self.device, 'native_qsa_boundary', False)):
            return
        row = self.trace.row()
        key = 'selection_evidence' if selection else 'output_evidence'
        if key in row:
            raise OperationError('native43 boundary evidence repeated')
        row[key] = self.device.boundary_digest(tensor, selection=selection)
        row['timing_excluded'] = 'final-chunk-boundary-readback'

    def _check_versions(self, layer_name, versions):
        if (not isinstance(layer_name, str) or not layer_name
                or type(versions) is not tuple or not versions
                or any(type(value) is not int or value < 0 for value in versions)):
            raise OperationError('invalid per-physical-page source versions')
        return tuple(versions)

    # --- identity / ownership validation ---------------------------------
    def observe_chunk_shape(self, rows, min_rows, route):
        """Host-only diagnostic at helper entry; no token/UID/page data."""
        import logging
        try:
            logging.getLogger(__name__).info(
                'flash-s5 chunk_shape rows=%d min_rows=%d route=%s chunk_index=%d',
                rows, min_rows, route, self.transaction.txn_seq)
        except Exception:
            # A broken diagnostic handler cannot change an operation's route,
            # admission guard, completion or failure. Keep a count, not values.
            self.stats['shape_log_errors'] = self.stats.get('shape_log_errors', 0) + 1

    def __repr__(self):
        return (f'QsaPrefillOperation(txn_seq={self.transaction.txn_seq}, '
                f'rank={self.lane.worker_rank}, storage_epoch={self.lane.storage_epoch}, '
                f'layers={len(self._layers)}, pending={self.pending})')

    def _check_identity(self, *, layer_name=None, lane=None,
                        allocation_epoch=None, source_epoch=None,
                        compute_required=True):
        # The worker RPC thread may query an accepted profile while the lane
        # continuation is paused at its submission boundary. Querying an
        # event and reading the completed pinned plan are host operations;
        # they do not own the lane's compute stream. Keep the stream check for
        # every mutation that binds, advances or enqueues device work.
        if compute_required:
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

    def register_source(self, layer_name, source, *, source_epoch=0, versions=None):
        """Bind the real registered CPU pool for a layer; keep the tensor owner."""
        if not isinstance(layer_name, str) or not layer_name:
            raise OperationError('layer name required')
        if type(source_epoch) is not int or source_epoch < 0:
            raise OperationError('source epoch must be a nonnegative int')
        if versions is not None:
            versions = self._check_versions(layer_name, versions)
        self._check_identity()
        if layer_name in self.source_epochs and self.source_epochs[layer_name] != source_epoch:
            raise OperationError('source epoch disagrees with admission')
        previous_versions = self.source_versions.get(layer_name)
        if previous_versions is not None and versions is not None and previous_versions != versions:
            raise OperationError('source version table changed after admission')
        if versions is not None:
            self.source_versions[layer_name] = versions
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
            if self.source_versions.get(layer_name) is not None:
                previous.source_versions = self.source_versions[layer_name]
            return previous
        run = LayerRun(layer_name, source, facts, self.allocation_epoch, source_epoch)
        run.source_versions = self.source_versions.get(layer_name, ())
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

    @trace_cpu('cpu_plan_ns')
    def bind_plan(self, layer_name, block_table_row, num_pages, valid_tokens,
                  arena, *, lane=None, allocation_epoch=None, source_epoch=None,
                  versions=None):
        """Bind the dense CPU page plan, arena and GPU block-table row.

        The physical pages must come from a CPU-owned mapping (admission); the
        GPU table row is bound only as an identity/geometry owner.  Reading it
        back would synchronize and is refused.  The same CPU pages are the
        sparse route's logical->physical map, so the optional version table is
        validated here before any selection is known.
        """
        self._check_identity(layer_name=layer_name, lane=lane,
                             allocation_epoch=allocation_epoch,
                             source_epoch=source_epoch)
        if versions is not None:
            versions = self._check_versions(layer_name, versions)
            prior = self.source_versions.get(layer_name)
            if prior is not None and prior != versions:
                raise OperationError('source version table changed after admission')
            self.source_versions[layer_name] = versions
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
        versions = self.source_versions.get(layer_name, run.source_versions)
        if versions:
            if any(page >= len(versions) for page in pages):
                raise OperationError('physical page exceeds the source version table')
        if run.buckets:
            if run.arena_facts != arena_facts or run.table_facts != table_facts:
                raise OperationError('arena or block-table allocation changed')
            if run.pages != tuple(pages) or run.valid_tokens != valid_tokens:
                raise OperationError('page plan changed during the layer')
            if run.buckets != plan_buckets(pages):
                raise OperationError('page plan changed during the layer')
            return run.buckets
        for page in pages:
            if page < 0 or page >= run.source_facts.shape[0]:
                raise OperationError('physical source page outside the pinned pool')
        run.arena, run.arena_facts = arena, arena_facts
        run.table, run.table_facts = block_table_row, table_facts
        run.pages = tuple(pages)
        run.valid_tokens = valid_tokens
        run.source_versions = tuple(versions or ())
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

    # --- sparse-history admission and asynchronous selection --------------
    def attach_sparse_source(self, layer_name, hold):
        """Attach the admission-owned source hold before the sparse helper runs."""
        if type(hold) is not SparseSourceHold:
            raise OperationError('sparse source hold required')
        self._check_identity(layer_name=layer_name)
        run = self._layer(layer_name)
        owner = hold.owner
        if owner.transaction != self.transaction:
            raise SourceRefused('source hold belongs to another transaction')
        if owner.lane != self.lane:
            raise SourceRefused('source hold belongs to another lane')
        if owner.storage_epoch != self.allocation_epoch:
            raise SourceRefused('source hold storage epoch is stale')
        if owner.source_epoch != run.source_epoch:
            raise SourceRefused('source hold source epoch is stale')
        if run.sparse_hold is not None and run.sparse_hold is not hold:
            raise OperationError('layer already has a different sparse source hold')
        if run.pages and tuple(owner.physical_pages) != run.pages:
            raise SourceRefused('source hold map differs from the bound plan')
        if type(self.state_epoch) is not int or self.state_epoch < 1:
            raise SourceRefused('sparse source attached without the admitted state epoch')
        run.sparse_hold = hold
        run.state_epoch = self.state_epoch
        run.mapping_version = getattr(owner, 'mapping_version', None)
        return hold

    def has_sparse_source(self, layer_name):
        run = self._layers.get(layer_name)
        return run is not None and run.sparse_hold is not None

    def bind_sparse_context(self, layer_name, arena, block_table_row, *,
                            versions=None, min_pages=MAX_BUCKET_PAGES,
                            null_page=None, lane=None, allocation_epoch=None,
                            source_epoch=None):
        """Bind the compact-row context after the dense plan; no selection yet."""
        self._check_identity(layer_name=layer_name, lane=lane,
                             allocation_epoch=allocation_epoch,
                             source_epoch=source_epoch)
        run = self._layer(layer_name)
        if not run.pages or not run.buckets:
            raise OperationError('sparse context needs the bound dense plan')
        if run.sparse_hold is None:
            raise OperationError('sparse context needs an admitted source hold')
        if (self.device.facts(arena) != run.arena_facts
                or self.device.facts(block_table_row) != run.table_facts):
            raise OperationError('sparse context arena or block-table changed')
        if versions is None:
            versions = self.source_versions.get(layer_name, run.source_versions)
        versions = self._check_versions(layer_name, versions)
        if any(page >= len(versions) for page in run.pages):
            raise OperationError('source map exceeds the version table')
        if any(page >= run.source_facts.shape[0] for page in run.pages):
            raise OperationError('source map exceeds the pinned pool')
        if type(min_pages) is not int or not 1 <= min_pages <= MAX_LAYER_PAGES:
            raise OperationError('invalid sparse eligibility threshold')
        if null_page is not None and (type(null_page) is not int or null_page < 0):
            raise OperationError('invalid null physical page')
        owner = run.sparse_hold.owner
        if owner.valid_tokens != run.valid_tokens:
            raise SourceRefused('valid-token frontier differs from the admission')
        if tuple(owner.physical_pages) != run.pages:
            raise SourceRefused('source hold map differs from the bound plan')
        state = run.sparse
        if state is not None:
            if (state.mapping, state.versions, state.epoch, state.valid_tokens,
                    state.min_pages, state.null_page) != (
                        run.pages, versions, self.allocation_epoch,
                        run.valid_tokens, min_pages, null_page):
                raise OperationError('sparse context changed during the layer')
            return state
        state = SparseLayerRun(run.pages, versions, self.allocation_epoch,
                               run.valid_tokens, min_pages, run.sparse_hold,
                               null_page=null_page)
        run.sparse = state
        self._owners[('sparse_source', layer_name)] = (run.sparse_hold,)
        return state

    # --- native row1 qualification (opt-in, serial) -----------------------
    def bind_row1_uva(self, layer_name, uva):
        """Bind the original UVA view the reference gather must read.

        Qualification only; the native route's own source is the pinned CPU
        pool. The reference tensor must have the admitted source's exact
        geometry so a compact row offset addresses the same logical row.
        """
        if self.qualification is None:
            return None
        self._check_identity(layer_name=layer_name)
        run = self._layer(layer_name)
        if run.sparse_hold is None:
            raise OperationError('row1 UVA binding needs the admitted source hold')
        facts = self.device.facts(uva)
        if (len(facts.shape) != 4 or facts.shape != run.source_facts.shape
                or facts.dtype != run.source_facts.dtype
                or facts.item_bytes != run.source_facts.item_bytes
                or not facts.contiguous):
            raise SourceRefused(
                'row1 UVA view does not match the admitted source geometry')
        if run.uva is not None and self.device.facts(run.uva) != facts:
            raise SourceRefused('row1 UVA view changed during the layer')
        run.uva = uva
        run.uva_facts = facts
        return facts

    def bind_row1_reference(self, layer_name, reference):
        """Bind the original attention function used as the byte oracle."""
        if self.qualification is None:
            return None
        self._check_identity(layer_name=layer_name)
        if not callable(reference):
            raise OperationError('row1 reference must be the original function')
        run = self._layer(layer_name)
        if run.row1_reference is not None and run.row1_reference is not reference:
            raise OperationError('row1 reference changed during the layer')
        run.row1_reference = reference
        return reference

    def _mark_qualification_bucket(self, run, index, *, route, rows, pieces,
                                   compact_start, nbytes, segments):
        if self.qualification is None:
            return None
        bucket = Row1Bucket(index=index, route=route, rows=rows, pieces=pieces,
                            compact_start=compact_start, nbytes=nbytes,
                            segments=tuple(segments))
        run.qualification_bucket = (index, bucket)
        return bucket

    def _qualify_pending_bucket(self):
        """Compare a completed bucket's actual arena bytes with the UVA gather.

        Runs strictly after the attention consumer event was host-queried
        complete and before the next fill can reuse the arena.
        """
        qualifier = self.qualification
        if qualifier is None:
            return
        for run in list(self._layers.values()):
            pending = run.qualification_bucket
            if pending is None:
                continue
            index, bucket = pending
            record = None
            for candidate in run.consumed:
                if candidate.stage is Stage.ATTENTION and candidate.bucket == index:
                    record = candidate
                    break
            if record is None or not record.complete:
                continue
            run.qualification_bucket = None
            qualifier.bucket(run, bucket, run.arena)

    def row1_output(self, layer_name, q, kv_cache, logical_indices,
                    block_table_row, rows, max_logical_page, arena,
                    use_prefill_config, out):
        """Compare the actual QSA output before MoE with the original function.

        The original runs on the same query/arena/output storage; it may use
        its own bulk UVA gather. This is serial qualification, never shipping
        performance, and it writes no second history GPU buffer.
        """
        if self.qualification is None:
            return None
        self._check_identity(layer_name=layer_name)
        run = self._layer(layer_name)
        if run.uva is None or run.uva_facts is None:
            raise OperationError('row1 output without the bound UVA view')
        if run.row1_reference is None:
            raise OperationError('row1 output without the bound original function')
        if self.device.facts(kv_cache) != run.uva_facts:
            raise SourceRefused('row1 output UVA view changed after binding')
        if self.device.facts(arena) != run.arena_facts:
            raise OperationError('row1 output arena changed after binding')
        sparse = run.sparse
        if sparse is None or sparse.route is None:
            raise OperationError('row1 output before a declared route')
        if run.qualification_bucket is not None:
            raise OperationError('row1 output with an uncompared bucket')
        expected = {'sparse_dma': 'sparse_uva', 'dense_dma': 'dense_dma',
                    'empty': 'empty'}[sparse.route]
        expected_fills = len(run.buckets) if expected == 'dense_dma' else 0
        session = self.qualification.reference_session(run)
        reference = run.row1_reference

        def original():
            reference(q, kv_cache, logical_indices, block_table_row, rows,
                      max_logical_page, arena, use_prefill_config, out,
                      e2=session)

        self.qualification.output(run, session, original, out,
                                  expected_route=expected,
                                  expected_fills=expected_fills)
        return True

    def _sparse_state(self):
        run = self._current_run()
        if run.sparse is None:
            raise OperationError('sparse route is not bound for ' + run.layer_name)
        return run, run.sparse

    def sparse_selection_eligible(self, num_pages, pages_in_arena):
        """Pinned eligibility, made explicit: large selections need admission.

        Small contexts keep the dense plan.  A large selection in a layer with
        no admitted sparse source refuses instead of falling back to a bulk UVA
        gather beside decode.
        """
        run = self._current_run()
        self._check_identity(layer_name=run.layer_name)
        if type(num_pages) is not int or num_pages < 1:
            raise OperationError('invalid dense/sparse page count')
        if type(pages_in_arena) is not int or pages_in_arena < 1:
            raise OperationError('invalid arena page count')
        threshold = (run.sparse.min_pages if run.sparse is not None
                     else pages_in_arena)
        if num_pages <= threshold:
            return False
        if run.sparse is None or run.sparse_hold is None:
            self.stats['sparse_fallback'] += 1
            pages = num_pages
            refusal = SparseRefusal(
                'sparse_not_admitted', 0, 0, 0, pages,
                pages * PAGE_BYTES, self.allocation_epoch)
            self.stats['sparse_fallback_bytes'] += refusal.nbytes
            self._fallback = refusal
            raise SparseFallbackRequired(refusal)
        return not (self.transport_policy == 'dense' and num_pages > pages_in_arena)

    def declare_sparse_route(self, route):
        """Generated-code route claim; the plan must already be built."""
        run, sparse = self._sparse_state()
        if route == 'sparse_dma':
            if sparse.plan is None or sparse.plan.refused:
                raise OperationError('sparse route declared before an admitted plan')
        elif route == 'dense_dma':
            if sparse.plan is not None:
                raise OperationError('dense route declared with a sparse plan')
        else:
            raise OperationError('unknown sparse route declaration')
        sparse.route = route
        self.stats['sparse_routes'] += 1
        return route

    def sparse_empty(self):
        """Pinned empty-union exit: no DMA, no credit, no source hold use."""
        run, sparse = self._sparse_state()
        self._check_identity(layer_name=run.layer_name)
        if sparse.selection_count != 0:
            raise OperationError('empty route without a completed empty selection')
        sparse.empty = True
        sparse.route = 'empty'
        self.stats['sparse_empty'] += 1
        return 0

    def sparse_union_scratch(self, capacity, logical_indices, rows):
        """Return the operation-owned fixed-capacity union buffers.

        Capacity, row count and width are host-known; nothing is sized by a
        device value.  One buffer set is reused across layers with the same
        geometry and replaced only after the previous layer finished.
        """
        run, sparse = self._sparse_state()
        self._check_identity(layer_name=run.layer_name)
        if sparse.plan is not None or sparse.selection is not None:
            raise OperationError('union scratch requested after selection staging')
        if type(capacity) is not int or not 1 <= capacity <= MAX_LAYER_ROWS:
            raise OperationError('union capacity is out of bounds')
        if not run.pages or capacity % len(run.pages) != 0:
            raise OperationError('union capacity disagrees with the dense page plan')
        page_size = capacity // len(run.pages)
        if page_size != PAGE_ROWS:
            raise OperationError('union capacity is not the QSA page geometry')
        if (run.valid_tokens + page_size - 1) // page_size != len(run.pages):
            raise OperationError('union capacity disagrees with the valid frontier')
        index_facts = self.device.facts(logical_indices)
        if (index_facts.device != self.device.device
                or index_facts.dtype not in ('torch.int32', 'torch.int64')
                or len(index_facts.shape) != 2 or index_facts.shape[1] < 2
                or not index_facts.contiguous or index_facts.address <= 0):
            raise OperationError('invalid selection index buffer')
        row_facts = self.device.facts(rows)
        if (row_facts.device != self.device.device
                or row_facts.dtype not in ('torch.int32', 'torch.int64')
                or len(row_facts.shape) != 1 or not row_facts.contiguous
                or row_facts.address <= 0
                or row_facts.shape[0] != index_facts.shape[0]):
            raise OperationError('selection rows must cover the index buffer')
        width = index_facts.shape[1] - 1
        index_rows = index_facts.shape[0]
        key = (capacity, index_rows, width, index_facts.dtype)
        if sparse.union is not None:
            if (sparse.union.capacity, sparse.union.rows,
                    sparse.union.width, sparse.union.dtype) != key:
                raise OperationError('union scratch changed within the layer')
            return sparse.union
        groups = (capacity + UNION_BLOCK - 1) // UNION_BLOCK
        expected_bytes = (12 * capacity + 8 * groups + 8
                          + index_facts.item_bytes * index_rows * (width + 1))
        if expected_bytes > MAX_UNION_SCRATCH_BYTES:
            raise OperationError('union scratch exceeds the explicit byte bound')
        if self._union_scratch is not None and self._union_scratch_layer != run.layer_name:
            previous = self._layers.get(self._union_scratch_layer)
            if previous is None or not previous.finished:
                raise OperationError('previous layer still owns the union scratch')
            self._owners.pop(('union_scratch', self._union_scratch_layer), None)
            self._union_scratch = None
        if self._union_scratch is not None and self._union_scratch_key != key:
            raise OperationError('union scratch geometry changed within the layer')
        if self._union_scratch is None:
            buffers = self.device.alloc_union_scratch(
                capacity, index_rows, width, index_facts.dtype, logical_indices)
            expected = dict(
                flags=('torch.int32', (capacity,)),
                ranks=('torch.int32', (capacity,)),
                tokens=('torch.int32', (capacity,)),
                totals=('torch.int32', (groups,)),
                offsets=('torch.int32', (groups,)),
                info=('torch.int32', (2,)),
                remapped=(index_facts.dtype, index_facts.shape),
            )
            if type(buffers) is not dict or set(buffers) != set(expected):
                raise OperationError('device returned an incomplete union scratch')
            nbytes = 0
            for name, (dtype, shape) in expected.items():
                facts = self.device.facts(buffers[name])
                if (facts.device != self.device.device or facts.dtype != dtype
                        or facts.shape != shape or not facts.contiguous
                        or facts.address <= 0):
                    raise OperationError('device returned an invalid union buffer')
                nbytes += facts.nbytes
            if nbytes > MAX_UNION_SCRATCH_BYTES:
                raise OperationError('union scratch exceeds the explicit byte bound')
            self._union_scratch = SparseUnionScratch(
                capacity, index_rows, width, index_facts.dtype, nbytes, **buffers)
            self._union_scratch_layer = run.layer_name
            self._union_scratch_key = key
            self._owners[('union_scratch', run.layer_name)] = tuple(buffers.values())
            self.stats['union_scratch_bytes'] = max(
                self.stats['union_scratch_bytes'], nbytes)
        if self.trace is not None:
            self.trace.opened['union'] = self.trace.begin('union', self.device.compute)
        sparse.union = self._union_scratch
        return sparse.union

    def stage_sparse_selection(self, tokens, info):
        """Enqueue the bounded D2H of the device count and sorted token union.

        The copy stream waits on a compute event recorded after the fixed-
        capacity union kernels.  Nothing is read on the host here; the caller
        yields until :meth:`sparse_selection_ready` observes that copy event.
        """
        run, sparse = self._sparse_state()
        if self.trace is not None:
            self.trace.end(self.trace.opened.pop('union', None))
        if self._cancelled is not None:
            raise OperationCancelled('cancelled operation cannot stage a selection')
        if sparse.union is None:
            raise OperationError('selection staged before its union scratch')
        if sparse.selection is not None:
            raise OperationError('selection already staged for this layer')
        if tokens is not sparse.union.tokens or info is not sparse.union.info:
            raise OperationError('staged selection is not the bound union scratch')
        capacity = sparse.union.capacity
        token_facts = self.device.facts(tokens)
        info_facts = self.device.facts(info)
        if (token_facts.device != self.device.device
                or token_facts.dtype != 'torch.int32'
                or token_facts.shape != (capacity,)
                or not token_facts.contiguous or token_facts.address <= 0):
            raise OperationError('invalid staged token union')
        if (info_facts.device != self.device.device
                or info_facts.dtype != 'torch.int32'
                or info_facts.shape != (2,)
                or not info_facts.contiguous or info_facts.address <= 0):
            raise OperationError('invalid staged union info record')
        scratch = self._index_scratch.get('torch.int32')
        if scratch is None:
            scratch = self.device.alloc_host_indices(MAX_LAYER_ROWS, 'torch.int32')
            facts = self.device.facts(scratch)
            if (facts.device != 'cpu' or not facts.pinned
                    or facts.dtype != 'torch.int32'
                    or facts.shape != (MAX_LAYER_ROWS,)
                    or not facts.contiguous):
                raise OperationError('selection scratch must be a pinned CPU index pool')
            self._index_scratch['torch.int32'] = scratch
            self._owners[('selection_scratch', 'torch.int32')] = (scratch,)
        host_info = self._index_scratch.get('info')
        if host_info is None:
            host_info = self.device.alloc_host_indices(2, 'torch.int32')
            facts = self.device.facts(host_info)
            if (facts.device != 'cpu' or not facts.pinned
                    or facts.dtype != 'torch.int32'
                    or facts.shape != (2,) or not facts.contiguous):
                raise OperationError('selection info must be pinned CPU scratch')
            self._index_scratch['info'] = host_info
            self._owners[('selection_scratch', 'info')] = (host_info,)
        if self.pending_selection:
            raise E3Pending('a selection readback is still in flight')
        if len(self._records) + 1 > self.max_tickets:
            raise OperationError('bounded operation-ticket budget exhausted')
        record = self._new_ticket(Stage.SELECTION, run.layer_name, -1)
        record.rows = capacity
        record.nbytes = token_facts.nbytes + info_facts.nbytes
        record.owners = (tokens, info, scratch, host_info)
        try:
            source_event = self.device.record(self.device.compute)
            self.device.wait_event(self.device.copy, source_event)
            self.stats['waits'] += 1
            self.device.copy_indices_to_host(tokens, scratch, capacity)
            self.device.copy_indices_to_host(info, host_info, 2)
            record.event = self.device.record(self.device.copy)
        except BaseException:
            self._failed = True
            raise
        sparse.selection = record
        sparse.selection_facts = token_facts
        sparse.selection_count = None
        sparse.selection_sparse = None
        sparse.host_tokens = scratch
        sparse.host_info = host_info
        sparse.selection_token = SelectionToken(run.layer_name, record.ticket)
        self.stats['selection_copies'] += 1
        self.stats['selection_bytes'] += record.nbytes
        return sparse.selection_token

    def schedule_boundary(self, token=None):
        """Yield exactly once after selection submission or dense plan binding.

        No query chooses whether this boundary exists. In particular a copy
        that already completed cannot pull the first bucket into this call.
        """
        run = self._current_run()
        self._check_identity(layer_name=run.layer_name)
        if run.schedule_boundary is not None or not run.producer_submitted:
            raise OperationError('QSA schedule boundary repeated or before producer')
        if not run.buckets or run.next_bucket or run.active_bucket is not None:
            raise OperationError('QSA schedule boundary outside the initial bound plan')
        if token is not None:
            sparse = run.sparse
            if sparse is None or token is not sparse.selection_token:
                raise OperationError('QSA schedule boundary for a foreign selection')
            ticket = token.ticket
        else:
            if run.sparse is not None and run.sparse.selection is not None:
                raise OperationError('selection schedule needs its submitted token')
            ticket = run.producer.ticket
        run.schedule_boundary = QsaScheduleBoundary(ticket)
        return run.schedule_boundary

    def schedule_profile(self):
        """Read-only host query of the plan, before any bucket submission."""
        from wire_v1 import QsaScheduleShape
        run = self._current_run()
        self._check_identity(layer_name=run.layer_name, compute_required=False)
        if run.schedule_boundary is None:
            raise OperationError('QSA profile requested before its submission boundary')
        if run.schedule_shape is not None:
            return run.schedule_shape
        sparse = run.sparse
        if sparse is not None and sparse.selection is not None:
            if not self.sparse_selection_ready(sparse.selection_token):
                return None
            if sparse.selection_count == 0:
                shape = QsaScheduleShape('empty', (), ())
            elif sparse.selection_sparse:
                plan = sparse.plan
                if plan is None or plan.refused:
                    raise OperationError('queried sparse profile has no admitted plan')
                waves = tuple((b.descriptors + MAX_DESCRIPTORS - 1)
                              // MAX_DESCRIPTORS for b in plan.buckets)
                shape = QsaScheduleShape('sparse_dma', waves, waves)
            else:
                counts = (1,) * len(run.buckets)
                shape = QsaScheduleShape('dense_dma', counts, counts)
        else:
            counts = (1,) * len(run.buckets)
            shape = QsaScheduleShape('dense_dma', counts, counts)
        run.schedule_shape = shape
        return shape

    def authorize_schedule(self, grant):
        """Bind the engine's typed agreement to this live queried local plan."""
        from wire_v1 import QsaProfileGrant
        run = self._current_run()
        if type(grant) is not QsaProfileGrant:
            raise OperationError('QSA schedule requires a typed engine grant')
        if (grant.query.transaction != self.transaction
                or self.lane.worker_rank not in grant.query.ranks):
            raise OperationError('QSA schedule grant has a foreign owner')
        shape = self.schedule_profile()
        if shape is None:
            raise E3Pending('QSA schedule selection has not completed')
        if grant.shape != shape:
            raise OperationError('QSA schedule grant differs from the queried local plan')
        if run.schedule_grant is not None and run.schedule_grant != grant:
            raise OperationError('QSA schedule grant changed after authorization')
        run.schedule_grant = grant

    def sparse_selection_pending(self, token):
        run, sparse = self._sparse_state()
        if token is not sparse.selection_token:
            raise OperationError('pending marker for a foreign selection token')
        return SparseSelectionPending(token.ticket)

    @trace_cpu('cpu_plan_ns')
    def sparse_selection_ready(self, token):
        """Poll the readback copy; build the CPU row plan exactly once.

        This is the separate operation phase the dispatcher can retry.  A copy
        whose event is not host-queried complete yields to the lane; it is
        never waited on here.  Once queried, reading the pinned scratch is a
        host read, not a device synchronization, and the count/route are
        decoded from that completed memory.
        """
        run, sparse = self._sparse_state()
        if token is not sparse.selection_token:
            raise OperationError('readiness for a foreign selection token')
        if sparse.selection is None:
            raise OperationError('selection was never staged')
        self._refresh()
        if not sparse.selection.complete:
            return False
        if sparse.selection_count is not None:
            return True
        info = self.device.host_indices(sparse.host_info, 2)
        if len(info) != 2 or any(type(value) is not int for value in info):
            self._failed = True
            raise OperationError('staged union info could not be decoded')
        count, flag = info
        capacity = sparse.union.capacity
        if not 0 <= count <= capacity:
            self._failed = True
            raise OperationError('staged union count is out of bounds')
        expected_flag = int(count * 5 < capacity * 4)
        if flag not in (0, 1) or flag != expected_flag:
            self._failed = True
            raise OperationError('staged union route flag is inconsistent')
        # The count is now valid host data read from completed pinned memory.
        sparse.selection_count = count
        sparse.selection_sparse = bool(flag)
        if count and flag:
            values = tuple(self.device.host_indices(sparse.host_tokens, count))
            if (len(values) != count
                    or any(type(value) is not int for value in values)
                    or any(a >= b for a, b in zip(values, values[1:]))):
                self._failed = True
                raise OperationError('staged token union is not sorted distinct')
            try:
                plan = make_sparse_plan(
                    values, sparse.mapping, sparse.versions, sparse.epoch,
                    page_rows=PAGE_ROWS, arena_pages=ARENA_PAGES,
                    max_descriptors=MAX_DESCRIPTORS,
                    valid_tokens=sparse.valid_tokens, null_page=sparse.null_page)
            except SparsePlanError as exc:
                self._failed = True
                raise OperationError('sparse plan could not be formed') from exc
            if plan.refused and plan.fallback.reason == 'descriptor_cap':
                plan = stream_sparse_plan(plan)
            sparse.plan = plan
            self.stats['sparse_planned'] += 1
            if plan.refused:
                sparse.refusal = plan.fallback
                self.stats['sparse_fallback'] += 1
                self.stats['sparse_fallback_rows'] += plan.fallback.rows
                self.stats['sparse_fallback_bytes'] += plan.fallback.nbytes
                self._fallback = plan.fallback
                raise SparsePlanRefused(plan.fallback)
            self.stats['sparse_descriptor_high_water'] = max(
                self.stats['sparse_descriptor_high_water'],
                plan.descriptor_high_water)
        return True

    def sparse_selection_count(self, token):
        """Distinct selected rows, valid only after the queried readback."""
        run, sparse = self._sparse_state()
        if token is not sparse.selection_token:
            raise OperationError('count for a foreign selection token')
        if sparse.selection_count is None:
            raise OperationError('selection count requested before completion')
        return sparse.selection_count

    @trace_cpu('cpu_submit_ns')
    def fill_sparse_bucket(self, staged, selected, compact_start, block_table_row):
        """Enqueue one bounded compact-row bucket; never synchronizes.

        The plan was built from the staged selection itself.  ``selected`` is
        identified by its exact base address and offset into the staged union,
        so a different or reordered GPU tensor cannot silently fill the arena.
        """
        run, sparse = self._sparse_state()
        self._check_identity(layer_name=run.layer_name)
        if sparse.plan is None or sparse.plan.refused:
            raise OperationError('sparse bucket fill before an admitted plan')
        if sparse.selection is None or not sparse.selection.complete:
            raise OperationError('sparse bucket fill before the selection completion')
        if self.device.facts(run.source) != run.source_facts:
            raise OperationError('source storage changed after binding')
        if not run.producer_submitted:
            raise OperationError('bucket fill before the producer boundary')
        if self.device.facts(block_table_row) != run.table_facts:
            raise OperationError('GPU block-table view changed')
        if type(compact_start) is not int or compact_start < 0:
            raise OperationError('invalid compact destination offset')
        facts = self.device.facts(selected)
        if len(facts.shape) != 1:
            raise OperationError('selected rows must be one compact vector')
        count = facts.shape[0]
        index = sparse.next_bucket
        if index >= len(sparse.plan.buckets):
            raise OperationError('more sparse buckets than the bound plan')
        bucket = sparse.plan.buckets[index]
        if compact_start != bucket.compact_start or count != bucket.rows:
            raise OperationError('sparse bucket order differs from the loaded plan')
        if compact_start + count > sparse.union.capacity:
            raise OperationError('selected rows exceed the union scratch')
        expected = (sparse.selection_facts.address
                    + compact_start * sparse.selection_facts.item_bytes)
        if (facts.device != self.device.device
                or facts.dtype != sparse.selection_facts.dtype
                or not facts.contiguous or facts.address != expected):
            raise OperationError('selected rows are not the staged selection slice')
        pages = (bucket.rows + PAGE_ROWS - 1) // PAGE_ROWS
        staged_facts = self.device.facts(staged)
        if (staged_facts.device != run.arena_facts.device
                or staged_facts.shape != (pages, *run.arena_facts.shape[1:])
                or staged_facts.strides != run.arena_facts.strides
                or staged_facts.dtype != run.arena_facts.dtype
                or not staged_facts.contiguous
                or staged_facts.address != run.arena_facts.address):
            raise OperationError('wrong sparse bucket destination')
        # Producer readiness plus the shared source-hold proof, before any copy.
        sparse.hold.validate(sparse.hold.owner, run.layer_name, sparse.mapping)
        self._refresh()
        if self.pending_buckets >= MAX_OUTSTANDING_BUCKETS:
            raise E3Pending('one arena bucket is still outstanding')
        needed = 1 if sparse.wave_copy is not None else 2
        if len(self._records) + needed > self.max_tickets:
            raise OperationError('bounded operation-ticket budget exhausted')
        if type(bucket) is SparseStreamBucket:
            return self._fill_sparse_wave(run, sparse, bucket, pages)
        copy = self._new_ticket(Stage.HISTORY_READY, run.layer_name, index)
        consumer = self._new_ticket(Stage.ATTENTION, run.layer_name, index)
        copy.sparse = True
        consumer.sparse = True
        copy.rows = bucket.rows
        copy.descriptors = bucket.descriptors
        copy.nbytes = bucket.nbytes
        copy.pages = pages
        copy.page_bytes = (run.arena_facts.shape[1] * run.arena_facts.shape[2]
                           * run.arena_facts.shape[3] * run.arena_facts.item_bytes)
        copy.pieces = tuple((piece.source_row, piece.arena_row, piece.rows)
                            for piece in bucket.pieces)
        copy.owners = (run.source, run.arena, run.table)
        self._mark_qualification_bucket(
            run, index, route='sparse_dma', rows=bucket.rows,
            pieces=bucket.descriptors, compact_start=bucket.compact_start,
            nbytes=bucket.nbytes,
            segments=((piece.source_row, piece.arena_row, piece.rows)
                      for piece in bucket.pieces))
        sparse.active_bucket = index
        run._pending_consumer = consumer
        sparse.hold.begin_bucket(index, bucket.rows, bucket.descriptors, bucket.nbytes)
        try:
            if index == 0:
                self.device.wait_event(self.device.copy, run.producer.event)
                self.stats['waits'] += 1
                if self._last_arena_consumer is not None:
                    self.device.wait_event(self.device.copy, self._last_arena_consumer)
                    self.stats['waits'] += 1
            else:
                self.device.wait_event(self.device.copy, run.consumed[-1].event)
                self.stats['waits'] += 1
            timing = self.trace.begin('copy', self.device.copy) if self.trace is not None else None
            for source_row, arena_row, rows in copy.pieces:
                self.device.copy_rows(run.source, run.arena, source_row,
                                      arena_row, rows)
            if self.trace is not None:
                self.trace.end(timing)
            copy.event = self.device.record(self.device.copy)
            sparse.hold.copy_recorded(index)
            sparse.hold.publish(index)
            if self.trace is not None and sparse.wave_copy is None:
                self.trace.row()['waves'] += 1
            self.stats['sparse_buckets'] += 1
            self.stats['sparse_rows'] += bucket.rows
            self.stats['sparse_pieces'] += bucket.descriptors
            self.stats['sparse_bytes'] += bucket.nbytes
            self.stats['sparse_descriptors'] += bucket.descriptors
            self.stats['copy_events'] += 1
            self.device.wait_event(self.device.compute, copy.event)
            self.stats['waits'] += 1
            if self.trace is not None:
                row = self.trace.row()
                row['descriptors'] += copy.descriptors if sparse_active_for_trace(run) else len(copy.pieces)
                row['waves'] += 1 if not sparse_active_for_trace(run) else 0
                row['bytes'] += copy.nbytes if sparse_active_for_trace(run) else copy.bytes
                self.trace.opened['attention'] = self.trace.begin('attention', self.device.compute)
        except BaseException:
            self._failed = True
            raise
        return copy.ticket

    def _fill_sparse_wave(self, run, sparse, bucket, pages):
        """One <=256-descriptor submission, inside fill_sparse_bucket's offer.

        Intermediate events grant host-queried submission credit only. The
        logical copy proof and attention consumer are created/published only
        once the full, unchanged compact bucket has been enqueued.
        """
        index = bucket.index
        if sparse.wave_copy is not None and not sparse.wave_ready:
            raise E3Pending('sparse copy wave not observed')
        try:
            if sparse.wave_copy is None:
                copy = self._new_ticket(Stage.HISTORY_READY, run.layer_name, index)
                copy.rows, copy.descriptors = bucket.rows, bucket.descriptors
                copy.nbytes, copy.pages = bucket.nbytes, pages
                copy.owners = (run.source, run.arena, run.table)
                sparse.wave_copy = copy
                sparse.wave_iterator = bucket.waves()
                sparse.wave_rows = sparse.wave_descriptors = 0
                sparse.hold.begin_stream_bucket(index, bucket.rows,
                                                 bucket.descriptors, bucket.nbytes)
                self.device.wait_event(self.device.copy, run.producer.event)
                self.stats['waits'] += 1
                previous = (run.consumed[-1].event if index
                            else self._last_arena_consumer)
                if previous is not None:
                    self.device.wait_event(self.device.copy, previous)
                    self.stats['waits'] += 1
            copy = sparse.wave_copy
            wave = next(sparse.wave_iterator)
            timing = self.trace.begin('copy', self.device.copy) if self.trace is not None else None
            for piece in wave:
                self.device.copy_rows(run.source, run.arena, piece.source_row,
                                      piece.arena_row, piece.rows)
            if self.trace is not None:
                self.trace.end(timing)
            sparse.wave_event = self.device.record(self.device.copy)
            sparse.wave_ready = False
            rows = sum(piece.rows for piece in wave)
            sparse.hold.stream_wave_recorded(index, rows, len(wave))
            if self.trace is not None:
                self.trace.row()['waves'] += 1
            sparse.wave_rows += rows
            sparse.wave_descriptors += len(wave)
            self.stats['copy_events'] += 1
            if sparse.wave_rows < bucket.rows:
                return SparseCopyPending(copy.ticket)
            if (sparse.wave_rows != bucket.rows
                    or sparse.wave_descriptors != bucket.descriptors):
                raise OperationError('streamed sparse bucket accounting disagrees')
            copy.event = sparse.wave_event
            copy.sparse = True
            consumer = self._new_ticket(Stage.ATTENTION, run.layer_name, index)
            consumer.sparse = True
            run._pending_consumer = consumer
            sparse.active_bucket = index
            self._mark_qualification_bucket(
                run, index, route='sparse_dma', rows=bucket.rows,
                pieces=bucket.descriptors, compact_start=bucket.compact_start,
                nbytes=bucket.nbytes,
                segments=((p.source_row, p.arena_row, p.rows)
                          for p in bucket.iter_pieces()))
            sparse.hold.copy_recorded(index)
            sparse.hold.publish(index)
            self.stats['sparse_buckets'] += 1
            self.stats['sparse_rows'] += bucket.rows
            self.stats['sparse_pieces'] += bucket.descriptors
            self.stats['sparse_bytes'] += bucket.nbytes
            self.stats['sparse_descriptors'] += bucket.descriptors
            self.device.wait_event(self.device.compute, copy.event)
            self.stats['waits'] += 1
            if self.trace is not None:
                row = self.trace.row()
                row['descriptors'] += copy.descriptors if sparse_active_for_trace(run) else len(copy.pieces)
                row['waves'] += 1 if not sparse_active_for_trace(run) else 0
                row['bytes'] += copy.nbytes if sparse_active_for_trace(run) else copy.bytes
                self.trace.opened['attention'] = self.trace.begin('attention', self.device.compute)
            sparse.wave_copy = sparse.wave_iterator = sparse.wave_event = None
            return copy.ticket
        except BaseException:
            self._failed = True
            raise

    def take_fallback(self):
        """Typed refusal for the caller's serial route; never an overlap pass."""
        return self._fallback

    # --- bounded DMA / attention boundaries -------------------------------
    @trace_cpu('cpu_submit_ns')
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
        if run.sparse_hold is not None:
            run.sparse_hold.validate(run.sparse_hold.owner, run.layer_name, run.pages)
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
        self._mark_qualification_bucket(
            run, index, route='dense_dma', rows=bucket.pages * PAGE_ROWS,
            pieces=len(bucket.pieces), compact_start=None, nbytes=copy.bytes,
            segments=((source_page * PAGE_ROWS, arena_page * PAGE_ROWS,
                       pages * PAGE_ROWS)
                      for source_page, arena_page, pages in bucket.pieces))
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
            timing = self.trace.begin('copy', self.device.copy) if self.trace is not None else None
            for source_page, arena_page, pages in bucket.pieces:
                self.device.copy_pages(run.source, run.arena, source_page,
                                       arena_page, pages)
            if self.trace is not None:
                self.trace.end(timing)
            copy.event = self.device.record(self.device.copy)
            self.stats['buckets'] += 1
            self.stats['pieces'] += len(bucket.pieces)
            self.stats['bytes'] += copy.bytes
            self.stats['copy_events'] += 1
            self.device.wait_event(self.device.compute, copy.event)
            self.stats['waits'] += 1
            if self.trace is not None:
                row = self.trace.row()
                row['descriptors'] += copy.descriptors if sparse_active_for_trace(run) else len(copy.pieces)
                row['waves'] += 1 if not sparse_active_for_trace(run) else 0
                row['bytes'] += copy.nbytes if sparse_active_for_trace(run) else copy.bytes
                self.trace.opened['attention'] = self.trace.begin('attention', self.device.compute)
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
        if self.trace is not None:
            self.trace.end(self.trace.opened.pop('attention', None))
        consumer = run._pending_consumer
        sparse_active = (run.sparse is not None
                         and run.sparse.active_bucket is not None)
        if consumer is None or (not sparse_active and run.active_bucket is None):
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
        if run.sparse is not None and run.sparse.active_bucket is not None:
            run.sparse.active_bucket = None
            run.sparse.next_bucket += 1
        else:
            run.active_bucket = None
            run.next_bucket += 1
        run._pending_consumer = None
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
        sparse = run.sparse
        sparse_route = (sparse is not None
                        and (sparse.plan is not None or sparse.empty))
        if sparse_route:
            total = 0 if sparse.empty else len(sparse.plan.buckets)
            done = sparse.next_bucket
            active = (sparse.active_bucket is not None
                      or run._pending_consumer is not None)
        else:
            total = len(run.buckets)
            done = run.next_bucket
            active = (run.active_bucket is not None
                      or run._pending_consumer is not None)
        if self._cancelled is None:
            self._check_identity(layer_name=layer_name)
        elif active or not total or done != total:
            raise OperationCancelled(
                'cancelled layer cannot close before its buckets retire')
        if run.producer is None or not run.producer_submitted:
            raise OperationError('layer ended without its producer boundary')
        if done != total or (total and not run.consumed):
            raise OperationError('layer ended before every bucket was consumed')
        if active:
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
        if stage is Stage.SELECTION:
            sparse = run.sparse
            if sparse is None or sparse.selection is None:
                raise OperationError('no selection readback was staged')
            self._refresh()
            if not sparse.selection.complete:
                raise E3Pending('selection readback is not observed yet')
            return sparse.selection_token
        if stage is Stage.HISTORY_READY:
            if not run.producer_submitted:
                raise E3Pending('producer has not been submitted')
            if run.producer.event is None or not self.device.query(run.producer.event):
                raise E3Pending('producer completion is not observed yet')
            if run.sparse is not None and run.sparse.plan is not None:
                if (run.sparse.selection is not None
                        and not run.sparse.selection.complete):
                    raise E3Pending('selection readback is not observed yet')
                if run.sparse.active_bucket is not None:
                    raise E3Pending('a bucket is still filled or consumed')
                index = run.sparse.next_bucket
                return index, run.sparse.plan.buckets[index:index + 1]
            if run.active_bucket is not None:
                raise E3Pending('a bucket is still filled or consumed')
            return run.next_bucket, run.buckets[run.next_bucket:run.next_bucket + 1]
        if stage is Stage.ATTENTION:
            if run.sparse is not None and run.sparse.active_bucket is not None:
                return run.sparse.active_bucket, run._pending_consumer.ticket
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

    def submission_credit(self):
        """Host-query-only credit for the next ``advance``.

        This is the same evidence ``advance`` itself consults before it pulls
        the continuation: queried completion events (``_refresh``), the
        outstanding attention consumers and the staged selection readback.
        It allocates nothing, records no event, enqueues no copy or kernel,
        never synchronizes and never runs the row1 byte oracle.  A caller that
        sees ``can_submit is False`` returns typed pending for the unchanged
        command, so polling credit cannot consume an accepted operation.
        """
        self._check_identity(compute_required=False)
        self._refresh()
        pending_buckets = self.pending_buckets
        pending_selection = self.pending_selection
        pending_waves = self.pending_copy_waves
        if pending_waves:
            return SubmissionCredit(False, 'sparse copy wave not observed',
                                    pending_buckets, pending_selection, pending_waves)
        if pending_buckets:
            return SubmissionCredit(False, 'attention consumer not observed',
                                    pending_buckets, pending_selection)
        if pending_selection:
            return SubmissionCredit(False, 'selection readback not observed',
                                    pending_buckets, pending_selection)
        return SubmissionCredit(True, 'credit available',
                                pending_buckets, pending_selection)

    def advance(self):
        """Submit at most one bucket; pending credit leaves the frame untouched.

        The caller must reinstall the lane's forward/stream context each time.
        Poll never calls this method. Completion of the final bucket still
        needs a subsequent observed proof before StopIteration is returned.
        """
        self._check_identity()
        run = self._current_run()
        if run.schedule_boundary is not None and run.schedule_grant is None:
            raise E3Pending('QSA schedule awaits active-stage TP agreement')
        self._refresh()
        try:
            # The previous bucket's actual bytes are compared while they still
            # occupy the arena, before the next fill may reuse it.
            self._qualify_pending_bucket()
        except BaseException:
            self._failed = True
            raise
        if self.pending_buckets:
            raise E3Pending('previous bucket consumer has not completed')
        if self.pending_copy_waves:
            raise E3Pending('sparse copy wave not observed')
        if self._continuation is None:
            raise OperationError('no generated continuation attached')
        try:
            return next(self._continuation)
        except StopIteration:
            self._continuation = None
            raise
        except (SparsePlanRefused, SparseFallbackRequired):
            # Typed admission refusal: the layer parks for the serial route.
            # This is not a transport failure and never counts as overlap.
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
        for run in self._layers.values():
            sparse = run.sparse
            if sparse is None or sparse.wave_copy is None or sparse.wave_event is None:
                continue
            try:
                if not sparse.wave_ready:
                    sparse.wave_ready = bool(self.device.query(sparse.wave_event))
                if self._cancelled is not None and sparse.wave_ready:
                    # Only the accepted prefix is retired; it is never a
                    # published bucket or an attention-completion claim.
                    sparse.hold.abandon_stream_after_drain(sparse.next_bucket)
                    sparse.wave_copy.rows = sparse.wave_rows
                    sparse.wave_copy.descriptors = sparse.wave_descriptors
                    sparse.wave_copy.nbytes = sparse.wave_rows * (PAGE_BYTES // PAGE_ROWS)
                    sparse.wave_copy.event = sparse.wave_event
                    sparse.wave_copy = sparse.wave_iterator = sparse.wave_event = None
            except BaseException:
                self._failed = True
                raise
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
                if record.sparse:
                    run = self._layers.get(record.layer_name)
                    sparse = None if run is None else run.sparse
                    if sparse is not None and sparse.hold is not None:
                        try:
                            if record.stage is Stage.HISTORY_READY:
                                sparse.hold.copy_completed(record.bucket)
                            elif record.stage is Stage.ATTENTION:
                                sparse.hold.consumer_completed(record.bucket)
                        except BaseException:
                            self._failed = True
                            raise
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
    def pending_selection(self):
        return int(any(record.stage is Stage.SELECTION and not record.complete
                       for record in self._records))

    @property
    def pending_copy_waves(self):
        return int(any(run.sparse is not None
                       and run.sparse.wave_copy is not None
                       and not run.sparse.wave_ready
                       for run in self._layers.values()))

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
        # SOURCE_HISTORY may be released only now, after every accepted copy
        # and attention consumer was observed; a partial bucketed source keeps
        # release pending and quarantines ownership. The row1 output comparison
        # must also have completed (or been aborted by a cancel) first.
        qualifier = self.qualification
        cancelled = self._cancelled is not None
        if qualifier is not None:
            for run in self._layers.values():
                if run.sparse_hold is not None or run.row1_reference is not None:
                    qualifier.check_releasable(run, cancelled=cancelled)
        for run in self._layers.values():
            if run.sparse_hold is not None:
                info = run.sparse_hold.release()
                self._source_release.append(info)
                self.stats['sparse_source_released'] += 1
                if qualifier is not None:
                    qualifier.finish_layer(run, hold_stats=info)
        if self.trace is not None:
            self.trace.emit()
            self.trace = None
        dropped = len(self._owners)
        self._released = True
        if self._continuation is not None:
            self._continuation.close()
            self._continuation = None
        self._owners.clear()
        self._index_scratch.clear()
        self._union_scratch = None
        self._union_scratch_layer = None
        self._union_scratch_key = None
        for record in self._records:
            record.owners = ()
        for run in self._layers.values():
            run.source = run.arena = run.table = None
            run.uva = None
            run.row1_reference = None
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
            # Sparse history: distinct route, bounded descriptor cap, reported
            # fallback counts/bytes.  No sparse-UVA byte is counted as overlap.
            sparse_buckets=self.stats['sparse_buckets'],
            sparse_rows=self.stats['sparse_rows'],
            sparse_pieces=self.stats['sparse_pieces'],
            sparse_bytes=self.stats['sparse_bytes'],
            sparse_descriptors=self.stats['sparse_descriptors'],
            sparse_descriptor_high_water=self.stats['sparse_descriptor_high_water'],
            sparse_descriptor_cap=MAX_DESCRIPTORS,
            sparse_planned=self.stats['sparse_planned'],
            sparse_empty=self.stats['sparse_empty'],
            sparse_fallback=self.stats['sparse_fallback'],
            sparse_fallback_rows=self.stats['sparse_fallback_rows'],
            sparse_fallback_bytes=self.stats['sparse_fallback_bytes'],
            sparse_routes=self.stats['sparse_routes'],
            selection_copies=self.stats['selection_copies'],
            selection_bytes=self.stats['selection_bytes'],
            selection_scratch_bytes=(SELECTION_HOST_BYTES
                                     if self._index_scratch else 0),
            selection_scratch_bound=SELECTION_HOST_BYTES,
            union_scratch_bytes=self.stats['union_scratch_bytes'],
            union_scratch_bound=MAX_UNION_SCRATCH_BYTES,
            sparse_source_released=self.stats['sparse_source_released'],
            sparse_fallback_reason=(None if self._fallback is None
                                    else self._fallback.reason),
            row1_enabled=self.qualification is not None,
            row1_buckets=(0 if self.qualification is None
                          else int(self.qualification.stats['buckets'])),
            row1_bucket_bytes=(0 if self.qualification is None
                               else int(self.qualification.stats['bucket_bytes'])),
            row1_outputs=(0 if self.qualification is None
                          else int(self.qualification.stats['outputs'])),
            row1_failures=(0 if self.qualification is None
                           else int(self.qualification.stats['failures'])),
            row1_layers=(0 if self.qualification is None
                         else int(self.qualification.stats['layers'])),
            basis='estimate; no transfer timing, SM measurement or readback',
        )

    def _current_run(self):
        if len(self._forward_layers) == 0:
            raise OperationError('no layer is being submitted')
        return self._layers[self._forward_layers[-1]]


__all__ = [
    'MAX_BUCKET_PAGES', 'MAX_LAYER_PAGES', 'MAX_OUTSTANDING_BUCKETS',
    'MAX_PAGES_PER_COPY', 'MAX_STAGES_PER_LAYER', 'MAX_UNION_SCRATCH_BYTES',
    'SELECTION_HOST_BYTES', 'BucketPlan', 'LayerRun',
    'NativeQsaDevice', 'OperationCancelled', 'OperationError',
    'QsaPrefillOperation', 'RUNNER_SEAM', 'SelectionToken', 'SparseLayerRun',
    'SparseSelectionPending', 'SparseUnionScratch', 'Stage', 'StageRecord',
    'SubmissionCredit', 'metadata_plan_provider', 'plan_buckets',
]
