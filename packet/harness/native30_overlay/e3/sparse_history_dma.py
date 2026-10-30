"""Bounded sparse-history row transport for the native prefill lane.

The pinned QSA prefill compacts a selection of history tokens into a sorted
union and gathers the selected K/V rows into the existing staging arena.  For
the E3 lane that gather is replaced by bounded host->device copies:

* one compact-row plan per layer, built from the *actual* staged selection
  (never a recomputed or reused guess);
* a piece merges only source-and-destination-contiguous rows, spans at most
  :data:`MAX_PIECE_PAGES` physical source pages and at most that many pages of
  bytes (including an unaligned start);
* a bucket holds at most :data:`ARENA_PAGES` compact pages, exactly the arena;
* descriptor overflow (more than :data:`MAX_DESCRIPTORS` pieces in a bucket),
  an unsupported source map or a stale source is a *typed* admission refusal,
  never a silent sparse-UVA gather inside a claimed overlap interval.

This module is CPU-only: it imports neither torch nor vLLM, does no I/O and
never synchronizes.  The owning operation issues the copies through its device
port; this module owns plan arithmetic, the source-hold contract and the
refusal/lifecycle accounting.

Source-hold interface shared with the admission/allocator worker::

    hold.validate(owner, layer, physical_pages)

checks the admitted transaction/request/storage/source incarnation, the exact
logical->physical page map and a retained allocator reference (a reference
object, never a boolean pin).  SOURCE_HISTORY stays held until a bucket's copy
*and* attention consumer have both completed; a partial bucket is never
published.  Cancellation stops new submissions, the accepted copies/consumers
drain, and only then may the hold be released.
"""
from dataclasses import dataclass, field
from itertools import islice

from e3_types import (
    ARENA_PAGES,
    E3ContractError,
    E3Pending,
    LaneId,
    PAGE_BYTES,
    PAGE_TOKENS,
    TransactionId,
)

# One compact row is one 1,024-byte main QSA KV row.
PAGE_ROWS = PAGE_TOKENS
ROW_BYTES = PAGE_BYTES // PAGE_TOKENS                 # 1024
ARENA_ROWS = ARENA_PAGES * PAGE_ROWS                 # 97,152
MAX_PIECE_PAGES = 4
MAX_PIECE_ROWS = MAX_PIECE_PAGES * PAGE_ROWS
MAX_DESCRIPTORS = 256                                # tested diagnostic bound
MAX_LAYER_PAGES = 128                                # mirrors qsa_operation
MAX_LAYER_ROWS = MAX_LAYER_PAGES * PAGE_ROWS


class SparseTransportError(E3ContractError):
    """The sparse transport contract was violated."""


class SparsePlanError(SparseTransportError):
    """A plan cannot be formed from the supplied map or selection."""


class SparsePlanRefused(SparseTransportError):
    """Typed admission refusal: finish this layer outside the overlap window."""

    def __init__(self, refusal):
        if type(refusal) is not SparseRefusal:
            raise SparseTransportError('refusal record required')
        self.refusal = refusal
        super().__init__('sparse history refused: ' + refusal.reason)


class SparseFallbackRequired(SparseTransportError):
    """The caller must park this layer for the serial route; no UVA overlap."""

    def __init__(self, refusal):
        if type(refusal) is not SparseRefusal:
            raise SparseTransportError('refusal record required')
        self.refusal = refusal
        super().__init__('sparse history needs the serial route: ' + refusal.reason)


class SourceRefused(SparseTransportError):
    """The admitted source, incarnation, map or allocator reference is stale."""


@dataclass(frozen=True)
class SparseRefusal:
    """Reported fallback metadata; carries no storage and no request identity."""
    reason: str
    bucket: int
    descriptors: int
    rows: int
    pages: int
    nbytes: int
    epoch: int

    def __post_init__(self):
        if type(self.reason) is not str or not self.reason:
            raise SparseTransportError('refusal reason required')
        for name in ('bucket', 'descriptors', 'rows', 'pages', 'nbytes', 'epoch'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SparseTransportError('invalid refusal ' + name)


@dataclass(frozen=True)
class RowPiece:
    """One bounded copy: ``rows`` compact rows from a physical source row."""
    source_row: int
    arena_row: int
    rows: int

    def __post_init__(self):
        if (type(self.source_row) is not int or self.source_row < 0
                or type(self.arena_row) is not int or self.arena_row < 0
                or type(self.rows) is not int or not 1 <= self.rows <= MAX_PIECE_ROWS):
            raise SparsePlanError('invalid sparse row piece')
        if self.rows * ROW_BYTES > MAX_PIECE_PAGES * PAGE_BYTES:
            raise SparsePlanError('sparse piece exceeds four pages of bytes')
        first = self.source_row // PAGE_ROWS
        last = (self.source_row + self.rows - 1) // PAGE_ROWS
        if last - first + 1 > MAX_PIECE_PAGES:
            raise SparsePlanError('sparse piece spans more than four physical pages')

    @property
    def source_pages(self):
        return (self.source_row + self.rows - 1) // PAGE_ROWS - self.source_row // PAGE_ROWS + 1

    @property
    def nbytes(self):
        return self.rows * ROW_BYTES

    @property
    def arena_end(self):
        return self.arena_row + self.rows


@dataclass(frozen=True)
class SparseBucket:
    """One arena-sized compact batch; destination rows are 0..rows-1."""
    index: int
    compact_start: int
    rows: int
    pieces: tuple

    def __post_init__(self):
        if (type(self.index) is not int or self.index < 0
                or type(self.compact_start) is not int or self.compact_start < 0
                or type(self.rows) is not int or not 1 <= self.rows <= ARENA_ROWS
                or type(self.pieces) is not tuple or not self.pieces):
            raise SparsePlanError('invalid sparse bucket')
        cursor = 0
        for piece in self.pieces:
            if type(piece) is not RowPiece or piece.arena_row != cursor:
                raise SparsePlanError('sparse bucket destination is not contiguous')
            cursor = piece.arena_end
        if cursor != self.rows:
            raise SparsePlanError('sparse bucket piece rows disagree')
        if len(self.pieces) > MAX_DESCRIPTORS:
            raise SparsePlanError('sparse bucket exceeds the descriptor cap')

    @property
    def compact_end(self):
        return self.compact_start + self.rows

    @property
    def descriptors(self):
        return len(self.pieces)

    @property
    def nbytes(self):
        return self.rows * ROW_BYTES


@dataclass(frozen=True)
class SparseHistoryPlan:
    """Immutable plan bound to one layer epoch and one exact source map."""
    epoch: int
    mapping: tuple = field(repr=False)
    versions: tuple = field(repr=False)
    selected: tuple = field(repr=False)     # actual compact-order token ids
    buckets: tuple
    fallback: object = None                 # SparseRefusal or None

    @property
    def refused(self):
        return self.fallback is not None

    @property
    def rows(self):
        return len(self.selected)

    @property
    def descriptors(self):
        return sum(bucket.descriptors for bucket in self.buckets)

    @property
    def nbytes(self):
        return self.rows * ROW_BYTES

    @property
    def descriptor_high_water(self):
        return max((min(bucket.descriptors, MAX_DESCRIPTORS)
                    for bucket in self.buckets), default=0)

    @property
    def bucket_rows(self):
        return tuple(bucket.rows for bucket in self.buckets)


@dataclass(frozen=True)
class SparseStreamBucket:
    """Same compact attention bucket, filled in bounded descriptor waves.

    Only selected integers and a map are retained, not a per-row descriptor
    array. Each iteration materializes at most MAX_DESCRIPTORS RowPieces.
    """
    index: int
    compact_start: int
    rows: int
    selected: tuple = field(repr=False)
    mapping: tuple = field(repr=False)
    descriptors: int

    @property
    def compact_end(self):
        return self.compact_start + self.rows

    @property
    def nbytes(self):
        return self.rows * ROW_BYTES

    def iter_pieces(self):
        prior = None
        for dst in range(self.rows):
            token = self.selected[self.compact_start + dst]
            source = self.mapping[token // PAGE_ROWS] * PAGE_ROWS + token % PAGE_ROWS
            if (prior is not None and prior.source_row + prior.rows == source
                    and prior.rows < MAX_PIECE_ROWS
                    and source // PAGE_ROWS - prior.source_row // PAGE_ROWS < MAX_PIECE_PAGES):
                prior = RowPiece(prior.source_row, prior.arena_row, prior.rows + 1)
            else:
                if prior is not None:
                    yield prior
                prior = RowPiece(source, dst, 1)
        if prior is not None:
            yield prior

    def waves(self):
        pieces = self.iter_pieces()
        while True:
            wave = tuple(islice(pieces, MAX_DESCRIPTORS))
            if not wave:
                return
            yield wave


def stream_sparse_plan(plan):
    """Recover only descriptor overflow, without changing selection or buckets.

    The strict planner remains the validator and still refuses at descriptor
    257. Unsupported mappings and stale identities are never routed here.
    """
    if (type(plan) is not SparseHistoryPlan or not plan.refused
            or plan.fallback.reason != 'descriptor_cap'):
        raise SparsePlanError('streaming requires a descriptor-cap refusal')
    buckets = []
    for lo in range(0, len(plan.selected), ARENA_ROWS):
        rows = min(ARENA_ROWS, len(plan.selected) - lo)
        candidate = SparseStreamBucket(len(buckets), lo, rows,
                                       plan.selected, plan.mapping, 0)
        count = sum(1 for _ in candidate.iter_pieces())
        buckets.append(SparseStreamBucket(len(buckets), lo, rows,
                                         plan.selected, plan.mapping, count))
    return SparseHistoryPlan(plan.epoch, plan.mapping, plan.versions,
                             plan.selected, tuple(buckets))


def _int_tuple(values, name, *, minimum=0, maximum=None):
    if type(values) is not tuple or not values:
        raise SparsePlanError(name + ' must be a non-empty tuple')
    if any(type(value) is not int or value < minimum
           or (maximum is not None and value > maximum) for value in values):
        raise SparsePlanError('invalid ' + name)
    return values


def make_sparse_plan(selected, mapping, versions, epoch, *, page_rows=PAGE_ROWS,
                     arena_pages=ARENA_PAGES, max_descriptors=MAX_DESCRIPTORS,
                     valid_tokens=None, null_page=None):
    """Build the compact-row plan from the actual selected token ids.

    Selection order is preserved exactly; a bucket never mixes compact rows.
    The whole plan is prevalidated: one over-cap or unsupported bucket refuses
    the complete selection before any copy is submitted.
    """
    if type(epoch) is not int or epoch < 1:
        raise SparsePlanError('invalid source epoch')
    if page_rows != PAGE_ROWS:
        raise SparsePlanError('sparse transport is fixed at 4,048-row pages')
    if type(arena_pages) is not int or not 1 <= arena_pages <= ARENA_PAGES:
        raise SparsePlanError('invalid arena page bound')
    if type(max_descriptors) is not int or not 1 <= max_descriptors <= MAX_DESCRIPTORS:
        raise SparsePlanError('invalid descriptor bound')
    mapping = _int_tuple(mapping, 'physical page map')
    versions = _int_tuple(versions, 'source versions', maximum=None)
    if any(page >= len(versions) for page in mapping):
        raise SparsePlanError('source map exceeds the version table')
    if null_page is not None and (type(null_page) is not int or null_page < 0):
        raise SparsePlanError('invalid null page')
    if type(selected) is not tuple:
        try:
            selected = tuple(selected)
        except TypeError as exc:
            raise SparsePlanError('selection must be an ordered sequence') from exc
    if len(selected) > MAX_LAYER_ROWS:
        raise SparsePlanError('selection exceeds the bounded layer frontier')
    frontier = len(mapping) * page_rows if valid_tokens is None else valid_tokens
    if type(frontier) is not int or not 1 <= frontier <= MAX_LAYER_ROWS:
        raise SparsePlanError('invalid valid-token frontier')
    if any(type(token) is not int or token < 0 or token >= frontier
           for token in selected):
        raise SparsePlanError('selection token outside the valid frontier')
    if any(first >= second for first, second in zip(selected, selected[1:])):
        raise SparsePlanError('selection must be sorted, unique and unpadded')
    if null_page is not None:
        used = {mapping[token // page_rows] for token in selected}
        if null_page in used:
            return SparseHistoryPlan(
                epoch, mapping, versions, selected, (),
                SparseRefusal('unsupported_mapping', 0, 0, len(selected),
                              (len(selected) + page_rows - 1) // page_rows,
                              len(selected) * ROW_BYTES, epoch))

    buckets = []
    limit = arena_pages * page_rows
    for lo in range(0, len(selected), limit):
        part = selected[lo:lo + limit]
        pieces = []
        for dst, token in enumerate(part):
            source = mapping[token // page_rows] * page_rows + token % page_rows
            if pieces:
                prior = pieces[-1]
                span = source // page_rows - prior.source_row // page_rows
                if (prior.source_row + prior.rows == source
                        and prior.arena_row + prior.rows == dst
                        and prior.rows < MAX_PIECE_ROWS
                        and span < MAX_PIECE_PAGES):
                    pieces[-1] = RowPiece(prior.source_row, prior.arena_row,
                                          prior.rows + 1)
                    continue
            pieces.append(RowPiece(source, dst, 1))
            if len(pieces) > max_descriptors:
                return SparseHistoryPlan(
                    epoch, mapping, versions, selected, (),
                    SparseRefusal('descriptor_cap', len(buckets), len(pieces),
                                  len(selected), (len(selected) + page_rows - 1) // page_rows,
                                  len(selected) * ROW_BYTES, epoch))
        buckets.append(SparseBucket(len(buckets), lo, len(part), tuple(pieces)))
    return SparseHistoryPlan(epoch, mapping, versions, selected, tuple(buckets), None)


def validate_plan_source(plan, mapping, versions, epoch):
    """Re-check the plan was built for this exact source incarnation and map."""
    if type(plan) is not SparseHistoryPlan:
        raise SparsePlanError('sparse plan object required')
    mapping, versions = tuple(mapping), tuple(versions)
    if (type(epoch) is not int or plan.epoch != epoch or plan.mapping != mapping
            or plan.versions != versions):
        raise SourceRefused('stale source epoch, mapping or version table')
    if plan.refused:
        raise SparsePlanRefused(plan.fallback)
    return plan


def boundary_cases():
    """The 12 VM<vm-id> boundaries, kept as runtime qualification fixtures."""
    p = PAGE_ROWS
    identity = tuple(range(50))
    fragmented = tuple((i * 17 + 3) % 52 for i in range(50))
    return [
        ('empty', (), identity, False),
        ('one_row', (p + 17,), fragmented, False),
        ('page_edges', (0, p - 1, p, p + 1, 2 * p - 1), fragmented, False),
        ('four_page_piece_cap', tuple(range(8 * p)), identity, False),
        ('unaligned_four_page_piece_cap', tuple(range(1, 8 * p + 3)), identity, False),
        ('arena_23_pages', tuple(range(23 * p)), fragmented, False),
        ('arena_24_pages', tuple(range(24 * p)), fragmented, False),
        ('arena_25_pages_tail', tuple(range(25 * p + 13)), fragmented, False),
        ('bounded_subpage_runs', tuple(t for i in range(128) for t in range(i * 64, i * 64 + 32)), fragmented, False),
        ('aliased_physical_pages', tuple(range(2 * p)), (3, 3) + identity[2:], False),
        ('near_sparse_threshold', tuple(range(40 * p - 1)), fragmented, False),
        ('fragmentation_fallback', tuple(range(0, 1024, 2)), fragmented, True),
    ]


# --- source hold shared with admission/allocator ----------------------------

@dataclass(frozen=True)
class SourceOwner:
    """Admission-owned source record; the allocator retains the storage."""
    transaction: TransactionId
    lane: LaneId
    layer: str
    storage_epoch: int
    source_epoch: int
    source_incarnation: int
    allocation_incarnation: int
    physical_pages: tuple = field(repr=False)
    valid_tokens: int
    mapping_version: int

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise SourceRefused('owner transaction must be a TransactionId')
        if type(self.lane) is not LaneId:
            raise SourceRefused('owner lane must be a LaneId')
        if type(self.layer) is not str or not self.layer:
            raise SourceRefused('owner layer required')
        for name in ('storage_epoch', 'source_epoch', 'source_incarnation',
                     'allocation_incarnation', 'valid_tokens', 'mapping_version'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SourceRefused('invalid owner ' + name)
        _int_tuple(self.physical_pages, 'owner physical pages')
        if self.valid_tokens < 1:
            raise SourceRefused('owner valid-token frontier must be positive')


class SparseSourceHold:
    """Retained source reference plus per-bucket copy/consumer accounting.

    Source history is released only after every started bucket has completed
    both its copy and its attention consumer (or was never published and its
    copies drained after cancellation).  A started bucket is never reusable.
    """

    def __init__(self, owner, *, allocator):
        if type(owner) is not SourceOwner:
            raise SourceRefused('source hold needs an admitted SourceOwner')
        if allocator is None or not all(callable(getattr(allocator, name, None))
                                        for name in ('retain', 'assert_retained', 'release')):
            raise SourceRefused('allocator lease interface is incomplete')
        self._owner = owner
        self._allocator = allocator
        self._reference = allocator.retain(owner)
        # A Boolean flag is not a retained allocator reference.
        if self._reference is None or type(self._reference) is bool:
            raise SourceRefused('allocator did not retain a reference object')
        self._buckets = {}
        self._released = False
        self.stats = dict(buckets=0, copied=0, published=0, consumed=0,
                          abandoned=0, rows=0, descriptors=0, nbytes=0)

    @property
    def owner(self):
        return self._owner

    @property
    def released(self):
        return self._released

    @property
    def reference(self):
        """Identity of the retained allocator reference (no storage/address)."""
        return self._reference

    def validate(self, owner, layer, physical_pages):
        """The shared source-hold proof.  Re-checked before every DMA bucket."""
        if self._released:
            raise SourceRefused('source hold is already released')
        if type(owner) is not SourceOwner or owner is not self._owner:
            raise SourceRefused('foreign admission owner')
        admitted = self._owner
        checks = (
            ('request/transaction', owner.transaction, admitted.transaction),
            ('lane', owner.lane, admitted.lane),
            ('layer', layer, admitted.layer),
            ('storage epoch', owner.storage_epoch, admitted.storage_epoch),
            ('source epoch', owner.source_epoch, admitted.source_epoch),
            ('source incarnation', owner.source_incarnation,
             admitted.source_incarnation),
            ('allocation incarnation', owner.allocation_incarnation,
             admitted.allocation_incarnation),
            ('valid-token frontier', owner.valid_tokens, admitted.valid_tokens),
            ('mapping version', owner.mapping_version, admitted.mapping_version),
        )
        for name, got, want in checks:
            if got != want:
                raise SourceRefused('stale source ' + name)
        pages = tuple(physical_pages)
        if (any(type(page) is not int or page < 0 for page in pages)
                or pages != admitted.physical_pages):
            raise SourceRefused('physical source map differs from the admission')
        # Must be a live reference, not a synthetic/boolean readiness flag.
        self._allocator.assert_retained(self._reference)
        return pages

    def begin_bucket(self, index, rows, descriptors, nbytes):
        if self._released:
            raise SourceRefused('source hold is already released')
        if type(index) is not int or index < 0 or index in self._buckets:
            raise SourceRefused('invalid or repeated bucket index')
        if (type(rows) is not int or not 1 <= rows <= ARENA_ROWS
                or type(descriptors) is not int or not 1 <= descriptors <= MAX_DESCRIPTORS
                or type(nbytes) is not int or nbytes != rows * ROW_BYTES):
            raise SourceRefused('invalid bucket accounting')
        self._buckets[index] = dict(state='filling', rows=rows,
                                    descriptors=descriptors, nbytes=nbytes,
                                    copied=False, copy_recorded=False)
        self.stats['buckets'] += 1
        self.stats['rows'] += rows
        self.stats['descriptors'] += descriptors
        self.stats['nbytes'] += nbytes

    def _bucket(self, index):
        entry = self._buckets.get(index)
        if entry is None:
            raise SourceRefused('bucket was never started')
        return entry

    def begin_stream_bucket(self, index, rows, descriptors, nbytes):
        """Retain a logical bucket; no partial prefix is publishable."""
        if (self._released or type(index) is not int or index < 0
                or index in self._buckets or type(rows) is not int
                or not 1 <= rows <= ARENA_ROWS or type(descriptors) is not int
                or not 1 <= descriptors <= rows or nbytes != rows * ROW_BYTES):
            raise SourceRefused('invalid streaming bucket accounting')
        self._buckets[index] = dict(state='filling', rows=rows,
            descriptors=descriptors, nbytes=nbytes, copied=False,
            copy_recorded=False, stream_rows=0, stream_descriptors=0)
        self.stats['buckets'] += 1
        self.stats['rows'] += rows
        self.stats['descriptors'] += descriptors
        self.stats['nbytes'] += nbytes

    def stream_wave_recorded(self, index, rows, descriptors):
        entry = self._bucket(index)
        if (entry['state'] != 'filling' or 'stream_rows' not in entry
                or type(rows) is not int or rows < 1
                or type(descriptors) is not int or not 1 <= descriptors <= MAX_DESCRIPTORS
                or entry['stream_rows'] + rows > entry['rows']
                or entry['stream_descriptors'] + descriptors > entry['descriptors']):
            raise SourceRefused('invalid streaming wave accounting')
        entry['stream_rows'] += rows
        entry['stream_descriptors'] += descriptors

    def abandon_stream_after_drain(self, index):
        """Cancel an unpublished prefix only after its actual copy event query.

        This does not certify the complete bucket, publication, or a consumer.
        The operation owns and queries the last accepted wave's event.
        """
        entry = self._bucket(index)
        if (entry['state'] != 'filling' or not entry.get('stream_rows')
                or entry['copy_recorded']):
            raise SourceRefused('no unpublished streaming prefix to retire')
        entry['state'] = 'abandoned'
        self.stats['abandoned'] += 1

    def copy_recorded(self, index):
        """All bounded copies for the bucket are enqueued behind one event."""
        entry = self._bucket(index)
        if entry['state'] != 'filling' or entry['copy_recorded']:
            raise SourceRefused('copy was not outstanding for this bucket')
        if ('stream_rows' in entry
                and (entry['stream_rows'] != entry['rows']
                     or entry['stream_descriptors'] != entry['descriptors'])):
            raise SourceRefused('streaming bucket is only partially copied')
        entry['copy_recorded'] = True
        entry['state'] = 'copied'
        self.stats['copied'] += 1

    def publish(self, index):
        """The whole bucket may be read by attention; never a partial bucket."""
        entry = self._bucket(index)
        if entry['state'] != 'copied' or not entry['copy_recorded']:
            raise SourceRefused('cannot publish before every copy is enqueued')
        entry['state'] = 'published'
        self.stats['published'] += 1

    def copy_completed(self, index):
        entry = self._bucket(index)
        if entry['state'] == 'filling' or not entry['copy_recorded']:
            raise SourceRefused('copy completion without a copied bucket')
        entry['copied'] = True

    def consumer_completed(self, index):
        """A queried attention event also proves its stream-ordered copy."""
        entry = self._bucket(index)
        if entry['state'] not in ('published', 'consumed'):
            raise SourceRefused('attention completed for an unpublished bucket')
        entry['copied'] = True
        if entry['state'] != 'consumed':
            entry['state'] = 'consumed'
            self.stats['consumed'] += 1

    def abandon(self, index):
        """Cancelled bucket with no consumer: only after its copies drained.

        A ``published`` bucket was fully copied but never read by attention
        (the cancellation stopped new launches), so draining its copies makes
        it abandonable.  A bucket still ``filling`` may hold partial bytes and
        is never abandonable.
        """
        entry = self._bucket(index)
        if entry['state'] not in ('copied', 'published') or not entry['copied']:
            raise SourceRefused('copies have not drained')
        entry['state'] = 'abandoned'
        self.stats['abandoned'] += 1

    @property
    def releasable(self):
        return (not self._released
                and all(entry['state'] in ('consumed', 'abandoned')
                        for entry in self._buckets.values()))

    def release(self):
        if self._released:
            raise SourceRefused('source hold was already released')
        if not self.releasable:
            raise E3Pending('sparse source has outstanding copies or consumers')
        self._allocator.release(self._reference)
        self._released = True
        return dict(self.stats, released=True)


# --- worker lease adapter ---------------------------------------------------
#
# The engine owns B's ``scheduler_hooks.SourceHold`` over real allocator block
# objects; its serialisable half is ``wire_v1.SourceLeaseDescriptor``.  This
# worker-side factory proves that descriptor against the immutable admitted
# frame and the authoritative local history registration, then builds this
# module's ``SparseSourceHold`` over the local allocator's real retained
# reference.  It never calls, imports or mutates a live engine Python object
# and it never frees the engine's scheduler hold: only the engine's validated
# all-rank final consumer/release barrier may do that.  Local completion
# queries may release local views through ``SparseSourceHold.release``.

SOURCE_LEASE_FIELDS = ('transaction', 'state_epoch', 'block_ids', 'consumer_ranks')


@dataclass(frozen=True)
class AllocationRegistration:
    """What the local allocator returns when it registers a source map.

    ``allocation_incarnation`` and ``versions`` are assigned by the allocator
    that owns the retained page storage; the adapter never invents them.
    ``versions`` is indexed by physical page id and must cover every page.
    """
    allocation_incarnation: int
    versions: tuple


@dataclass(frozen=True)
class SourceRegistration:
    """One authoritative local history registration, derived from its event."""
    layer: str
    group: int
    transaction: TransactionId
    state_epoch: int
    source_epoch: int
    source_incarnation: int
    allocation_incarnation: int
    mapping_version: int
    valid_tokens: int
    physical_pages: tuple = field(repr=False)
    versions: tuple = field(repr=False)
    source: object = field(repr=False)

    def owner(self, lane):
        return SourceOwner(
            self.transaction, lane, self.layer, lane.storage_epoch,
            self.source_epoch, self.source_incarnation,
            self.allocation_incarnation, self.physical_pages,
            self.valid_tokens, self.mapping_version)


@dataclass(frozen=True)
class OpenedSparseSource:
    """A locally retained source plus the exact descriptor that authorised it."""
    hold: object = field(repr=False)
    registration: SourceRegistration
    descriptor: object
    local_rank: int

    @property
    def source(self):
        return self.registration.source

    @property
    def source_epoch(self):
        return self.registration.source_epoch

    @property
    def versions(self):
        return self.registration.versions

    @property
    def physical_pages(self):
        return self.registration.physical_pages


class LocalSourceRegistry:
    """Authoritative local history registrations; never a guessed identity.

    ``register`` is called only when the worker's local history storage
    actually registers one (layer, group, source) map.  The source epoch,
    source incarnation and mapping version are counters of those registration
    events.  A changed identity while a local lease is open is refused rather
    than silently rebound.
    """

    def __init__(self, lane, *, ranks=None, blocks_per_kv_block=1):
        if type(lane) is not LaneId:
            raise SourceRefused('registry needs the worker LaneId')
        names = tuple(sorted(set(ranks if ranks is not None
                                  else (lane.worker_rank,))))
        if (not names or any(type(rank) is not int or rank < 0 for rank in names)
                or lane.worker_rank not in names):
            raise SourceRefused('registry ranks must name the local worker')
        if (type(blocks_per_kv_block) is not int or blocks_per_kv_block < 1):
            raise SourceRefused('registry needs the local KV-block factor')
        self.lane = lane
        self.ranks = names
        self.blocks_per_kv_block = blocks_per_kv_block
        self._records = {}
        self._active = {}
        self._source_seq = 0
        self._epoch_seq = 0

    def rebind_lane(self, lane):
        """Advance a drained prefill lane without losing source registrations.

        The per-chunk storage epoch advances at WorkerState.reserve. Source
        registrations and their versions live across chunks; an old live hold
        must never be relabelled with the next chunk's lane identity.
        """
        if type(lane) is not LaneId:
            raise SourceRefused('registry rebind needs the worker LaneId')
        old = self.lane
        if lane == old:
            return self
        if self._active:
            raise SourceRefused('source lane changed while a local lease is open')
        if (lane.worker_rank != old.worker_rank or lane.role != old.role
                or lane.context_id != old.context_id
                or lane.storage_epoch <= old.storage_epoch):
            raise SourceRefused('source lane rebind is foreign or stale')
        self.lane = lane
        return self

    def current(self, layer, group):
        return self._records.get((layer, group))

    def _check_key(self, layer, group):
        if type(layer) is not str or not layer:
            raise SourceRefused('registration needs a layer name')
        if type(group) is not int or group < 0:
            raise SourceRefused('registration needs a local KV group')

    def register(self, *, layer, group, transaction, state_epoch, valid_tokens,
                 physical_pages, source, allocator):
        """Record one actual local source registration.

        The allocator must implement ``register_source(source, pages)`` and
        return an :class:`AllocationRegistration`.  A changed registration for
        a key that still has an open local lease is refused.
        """
        self._check_key(layer, group)
        if type(transaction) is not TransactionId:
            raise SourceRefused('registration needs a TransactionId')
        if type(state_epoch) is not int or state_epoch < 0:
            raise SourceRefused('registration state epoch must be nonnegative')
        if type(valid_tokens) is not int or valid_tokens < 1:
            raise SourceRefused('registration valid frontier must be positive')
        pages = tuple(physical_pages)
        if (not pages or any(type(page) is not int or page < 0 for page in pages)
                or len(set(pages)) != len(pages)):
            raise SourceRefused('registration pages must be unique nonnegative ints')
        if (valid_tokens + PAGE_ROWS - 1) // PAGE_ROWS != len(pages):
            raise SourceRefused('registration map disagrees with its valid frontier')
        method = getattr(allocator, 'register_source', None)
        if not callable(method):
            raise SourceRefused('local allocator cannot register source storage')
        allocation = method(source, pages)
        if type(allocation) is not AllocationRegistration:
            raise SourceRefused('allocator did not return a registration record')
        versions = tuple(allocation.versions)
        if (len(versions) <= max(pages)
                or any(type(value) is not int or value < 0 for value in versions)):
            raise SourceRefused('allocation versions do not cover the source map')
        key = (layer, group)
        previous = self._records.get(key)
        changed = (previous is None or previous.source is not source
                   or previous.physical_pages != pages
                   or previous.valid_tokens != valid_tokens
                   or previous.versions != versions
                   or previous.allocation_incarnation
                   != allocation.allocation_incarnation
                   or previous.transaction != transaction
                   or previous.state_epoch != state_epoch)
        if not changed:
            return previous
        if key in self._active:
            raise SourceRefused('source was rebound while a local lease is open')
        self._source_seq += 1
        if previous is None or previous.source is not source:
            self._epoch_seq += 1
            source_epoch = self._epoch_seq
        else:
            source_epoch = previous.source_epoch
        record = SourceRegistration(
            layer, group, transaction, state_epoch, source_epoch,
            self._source_seq, allocation.allocation_incarnation,
            (1 if previous is None else previous.mapping_version + 1),
            valid_tokens, pages, versions, source)
        self._records[key] = record
        return record

    def _expanded(self, row):
        if self.blocks_per_kv_block > 1:
            return tuple(block * self.blocks_per_kv_block + index
                         for block in row
                         for index in range(self.blocks_per_kv_block))
        return tuple(row)

    def open(self, descriptor, admission, *, layer, group, allocator,
             registration=None, local_rank=None):
        """Prove descriptor + admitted frame + local registration, then retain.

        Refuses before any DMA on a missing/foreign/wrong descriptor, a wrong
        transaction/state epoch/rank/group/map, a released or rebound lease,
        or a stale local registration.  The engine's scheduler hold is not
        touched here.
        """
        from wire_v1 import SourceLeaseDescriptor

        self._check_key(layer, group)
        if type(descriptor) is not SourceLeaseDescriptor:
            raise SourceRefused('a WireV1 SourceLeaseDescriptor is required')
        if (admission is None
                or type(getattr(admission, 'transaction', None)) is not TransactionId
                or type(getattr(admission, 'state_epoch', None)) is not int
                or type(getattr(admission, 'block_tables', None)) is not tuple
                or type(getattr(admission, 'cow_pairs', None)) is not tuple):
            raise SourceRefused('the immutable admitted frame is required')
        rank = self.lane.worker_rank if local_rank is None else local_rank
        if type(rank) is not int or rank != self.lane.worker_rank:
            raise SourceRefused('source lease opened for another local rank')
        if rank not in descriptor.consumer_ranks:
            raise SourceRefused('descriptor does not name the local rank')
        if not set(self.ranks) <= set(descriptor.consumer_ranks):
            raise SourceRefused('descriptor does not name every admitted rank')
        record = self._records.get((layer, group))
        if record is None:
            raise SourceRefused('no authoritative local source registration')
        if registration is not None and registration is not record:
            raise SourceRefused('stale local source registration')
        if (record.transaction != descriptor.transaction
                or record.transaction != admission.transaction):
            raise SourceRefused('descriptor or frame transaction differs from registration')
        if (record.state_epoch != descriptor.state_epoch
                or record.state_epoch != admission.state_epoch):
            raise SourceRefused('descriptor or frame state epoch differs from registration')
        if group >= len(admission.block_tables):
            raise SourceRefused('local KV group is outside the admitted frame')
        expanded = self._expanded(admission.block_tables[group])
        pages = record.physical_pages
        if len(pages) > len(expanded):
            raise SourceRefused('registered map exceeds the admitted group row')
        if tuple(expanded[:len(pages)]) != pages:
            raise SourceRefused('registered map differs from the admitted group row')
        if not set(pages) <= set(descriptor.block_ids):
            raise SourceRefused('descriptor omits a local source page')
        if (record.valid_tokens + PAGE_ROWS - 1) // PAGE_ROWS != len(pages):
            raise SourceRefused('registered frontier disagrees with its map')
        # B's real frame proof: transaction, epoch, every page inside the
        # admitted rows or CoW pairs, and every admitted rank named.  A frame
        # mismatch is this adapter's typed refusal, not an untyped contract
        # error escaping into the worker bridge.
        try:
            descriptor.validate_frame(admission, self.ranks)
        except E3ContractError as exc:
            raise SourceRefused(
                'source lease frame differs from the admitted chunk') from exc
        if (layer, group) in self._active:
            raise SourceRefused('a local source lease is already open')
        hold = SparseSourceHold(record.owner(self.lane), allocator=allocator)
        # Prove the local lease is live now, before any bucket is started.
        hold.validate(hold.owner, layer, pages)
        self._active[(layer, group)] = hold
        return OpenedSparseSource(hold, record, descriptor, rank)

    def close(self, opened):
        """Drop the registry's view after the local hold was released."""
        if type(opened) is not OpenedSparseSource:
            raise SourceRefused('close needs an opened source')
        key = (opened.registration.layer, opened.registration.group)
        if self._active.get(key) is not opened.hold:
            raise SourceRefused('local source lease is not the open one')
        if not opened.hold.released:
            raise SourceRefused('local source lease is still retained')
        del self._active[key]


__all__ = [
    'ARENA_ROWS', 'AllocationRegistration', 'LocalSourceRegistry',
    'MAX_DESCRIPTORS', 'MAX_LAYER_ROWS', 'MAX_PIECE_PAGES',
    'MAX_PIECE_ROWS', 'OpenedSparseSource', 'PAGE_ROWS', 'ROW_BYTES',
    'SOURCE_LEASE_FIELDS', 'RowPiece', 'SourceOwner', 'SourceRefused',
    'SourceRegistration', 'SparseBucket', 'SparseFallbackRequired',
    'SparseHistoryPlan', 'SparsePlanError', 'SparsePlanRefused',
    'SparseRefusal', 'SparseSourceHold', 'SparseTransportError',
    'boundary_cases', 'make_sparse_plan', 'validate_plan_source',
]
