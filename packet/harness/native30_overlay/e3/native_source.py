"""Worker-local binding of the engine's source lease to the actual pinned host KV.

The engine owns B's :class:`scheduler_hooks.SourceHold` over real scheduler block
objects; its serialisable half is :class:`wire_v1.SourceLeaseDescriptor`.  This
module is the worker half:

* :func:`verify_kv_layout` proves the *admitted* QSA group row's layout from the
  pinned ``BlockTables`` (``blocks_per_kv_block``/``block_sizes``/
  ``kernel_block_sizes``) and the KV-cache spec.  An expanded mapping is refused;
  factor 1 is never assumed.
* :class:`NativeHostSourceAllocator` registers the actual ``_qsa_host_kv`` CPU
  tensor per local layer.  Its reference object is the worker's *local* storage
  ownership.  It is deliberately not the engine's scheduler reference: the
  scheduler keeps the block objects, this keeps the pinned tensor alive.
* :func:`bind_local_sources` turns one admitted frame plus one descriptor into
  per-layer :class:`SparseSourceHold` objects and the exact per-layer CPU page
  tables the operation's planner needs.

Nothing here allocates a GPU cache, invents a page identity or records a
completion.  The versions are an allocation-frame counter assigned when a page
first appears in an admitted frame; they are not copy/consumer completions.
"""
from dataclasses import dataclass, field

from .e3_types import E3ContractError, LaneId, LaneRole, TransactionId
from .sparse_history_dma import (
    AllocationRegistration,
    LocalSourceRegistry,
    PAGE_ROWS,
    SourceRefused,
    SourceOwner,
)

# ``_qsa_host_kv`` is [blocks, 1, 4048, 2*head_dim] bf16; head_dim is 256, so a
# row is 512 bf16 values (1024 bytes).  These are the pinned QSA geometry.
ROW_WIDTH = 512
MAX_LAYER_PAGES = 128


class NativeSourceError(E3ContractError):
    """The worker could not prove the engine lease against its own storage."""


class NativeLayoutError(NativeSourceError):
    """The admitted block-table/KV layout is not a supported QSA mapping."""


@dataclass(frozen=True)
class LocalKvLayout:
    """The verified local mapping from QSA layer name to KV group."""
    groups: tuple                 # ((layer_name, group), ...) sorted by name
    block_sizes: tuple
    kernel_block_sizes: tuple
    blocks_per_kv_block: tuple
    page_rows: int

    def group(self, layer_name):
        for name, group in self.groups:
            if name == layer_name:
                return group
        raise NativeLayoutError('QSA layer is not in a verified KV group: '
                                + str(layer_name))


def _ints(name, values):
    if type(values) not in (tuple, list) or not values:
        raise NativeLayoutError(f'{name} is not a non-empty pinned tuple')
    if any(type(value) is not int or value < 1 for value in values):
        raise NativeLayoutError(f'{name} must hold positive ints')
    return tuple(values)


def verify_kv_layout(runner, layer_names):
    """Prove the local QSA layout from the pinned block tables, never assume 1.

    The scheduler block ids handed to a worker are scheduler pages.  The
    transport reads ``kernel_block_size`` pages directly from the pinned host
    pool, so a KV block expanded by ``blocks_per_kv_block > 1`` would need an id
    mapping this transport does not implement.  That mapping is refused here
    instead of being silently treated as factor 1.
    """
    names = tuple(layer_names)
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise NativeLayoutError('QSA layer names must be distinct non-empty strings')
    config = getattr(runner, 'kv_cache_config', None)
    groups = getattr(config, 'kv_cache_groups', None)
    if type(groups) not in (tuple, list) or not groups:
        raise NativeLayoutError('runner has no pinned KV cache groups')
    tables = getattr(runner, 'block_tables', None)
    bpk = getattr(tables, 'blocks_per_kv_block', None)
    block_sizes = getattr(tables, 'block_sizes', None)
    kernel = getattr(tables, 'kernel_block_sizes', None)
    if bpk is None or block_sizes is None or kernel is None:
        raise NativeLayoutError(
            'pinned BlockTables layout is unavailable; blocks_per_kv_block '
            'must be verified, not assumed')
    bpk = _ints('blocks_per_kv_block', bpk)
    block_sizes = _ints('block_sizes', block_sizes)
    kernel = _ints('kernel_block_sizes', kernel)
    if not (len(bpk) == len(block_sizes) == len(kernel) == len(groups)):
        raise NativeLayoutError('block table layout disagrees with the KV groups')

    found = {}
    for group, spec in enumerate(groups):
        layer_names_in_group = getattr(spec, 'layer_names', None)
        if not isinstance(layer_names_in_group, (tuple, list)):
            raise NativeLayoutError('KV group has no pinned layer_names')
        # GDN/Mamba/circular groups keep their own geometry. Only the actual
        # local QSA source groups are transport pages.
        if not set(names).intersection(layer_names_in_group):
            continue
        cache_spec = getattr(spec, 'kv_cache_spec', None)
        block_size = getattr(cache_spec, 'block_size', None)
        if block_size != PAGE_ROWS:
            raise NativeLayoutError(
                'QSA scheduler page size is not the 4048-row transport page')
        if kernel[group] != PAGE_ROWS:
            raise NativeLayoutError(
                'QSA kernel block size is not the 4048-row transport page')
        expected = block_sizes[group] // kernel[group]
        if expected != bpk[group]:
            raise NativeLayoutError(
                'blocks_per_kv_block disagrees with the pinned block sizes')
        for name in layer_names_in_group:
            if name not in names:
                continue
            if name in found:
                raise NativeLayoutError('QSA layer appears in multiple KV groups')
            if bpk[group] != 1:
                raise NativeLayoutError(
                    'expanded QSA block tables are unsupported: '
                    'blocks_per_kv_block != 1')
            found[name] = group
    missing = sorted(set(names) - set(found))
    if missing:
        raise NativeLayoutError('QSA layer is not in the admitted KV layout: '
                                + ','.join(missing))
    return LocalKvLayout(tuple(sorted(found.items())), block_sizes, kernel,
                         bpk, PAGE_ROWS)


def source_geometry(source):
    """Validate the actual pinned CPU K/V pool; return bounded facts (no address)."""
    if str(getattr(source, 'device', None)) != 'cpu':
        raise NativeSourceError('QSA host source must be CPU storage')
    shape = tuple(getattr(source, 'shape', ()))
    if len(shape) != 4 or shape[1:] != (1, PAGE_ROWS, ROW_WIDTH) or shape[0] < 1:
        raise NativeSourceError('QSA host source has an invalid geometry')
    dtype = str(getattr(source, 'dtype', ''))
    if dtype not in ('torch.bfloat16', 'bfloat16'):
        raise NativeSourceError('QSA host source must be bf16')
    pinned = getattr(source, 'is_pinned', None)
    contiguous = getattr(source, 'is_contiguous', None)
    if not callable(pinned) or not callable(contiguous):
        raise NativeSourceError('QSA host source has no pinned/contiguity proof')
    if not bool(pinned()):
        raise NativeSourceError('QSA host source is not pinned CPU storage')
    if not bool(contiguous()):
        raise NativeSourceError('QSA host source is not contiguous')
    return shape


def storage_identity(source):
    """The actual storage allocation identity, never a geometry-only guess.

    ``untyped_storage().data_ptr()`` is the pinned host allocation behind the
    tensor; a UVA accelerator view shares it.  The value is an identity only and
    is never written to a report.
    """
    storage = getattr(source, 'untyped_storage', None)
    if callable(storage):
        try:
            raw = storage()
        except BaseException:
            raw = None
        pointer = getattr(raw, 'data_ptr', None)
        if callable(pointer):
            try:
                value = pointer()
            except BaseException:
                value = None
            if type(value) is int and value > 0:
                return value
    pointer = getattr(source, 'data_ptr', None)
    if callable(pointer):
        try:
            value = pointer()
        except BaseException:
            value = None
        if type(value) is int and value > 0:
            return value
    raise NativeSourceError('QSA host source has no storage allocation identity')


@dataclass(frozen=True)
class _PoolFrame:
    storage: int
    shape: tuple
    allocation_incarnation: int
    pages: tuple
    versions: tuple
    version: int
    transaction: TransactionId
    state_epoch: int


@dataclass
class _LocalSourceReference:
    """The worker-local retained reference (not the scheduler's block lease)."""
    owner: SourceOwner
    source: object = field(repr=False)
    allocation_incarnation: int
    released: bool = False

    def __post_init__(self):
        if type(self.owner) is not SourceOwner:
            raise SourceRefused('local reference needs the admitted owner')
        if type(self.allocation_incarnation) is not int or self.allocation_incarnation < 1:
            raise SourceRefused('local reference needs an allocation incarnation')


class NativeHostSourceAllocator:
    """Worker-local retain/release over the actual pinned ``_qsa_host_kv`` tensors.

    It never allocates or frees GPU storage and never frees the engine's
    scheduler blocks.  ``retain`` hands out an opaque reference that keeps the
    real CPU tensor owner alive; ``assert_retained`` proves that exact object is
    still live before a bucket is copied.
    """

    def __init__(self):
        self._frames = {}          # storage id -> _PoolFrame
        self._sources = {}         # storage id -> the actual pinned tensor owner
        self._by_owner = {}        # (transaction, layer) -> _PoolFrame
        self._refs = {}            # id(reference) -> reference
        self._allocation_seq = 0
        self._version_seq = 0
        self.stats = dict(registrations=0, retained=0, released=0, frames=0)

    def begin_frame(self, source, *, layer, transaction, state_epoch, pages):
        """Record the admission frame of one real pinned pool.

        Re-entering the same (transaction, state_epoch) frame for the same
        storage and pages is inert: a retry can never renumber a live page.
        """
        if not isinstance(layer, str) or not layer:
            raise SourceRefused('source registration needs a layer name')
        if type(transaction) is not TransactionId:
            raise SourceRefused('source registration needs the admission transaction')
        if type(state_epoch) is not int or state_epoch < 1:
            raise SourceRefused('source registration needs the admitted state epoch')
        shape = source_geometry(source)
        storage = storage_identity(source)
        pages = tuple(pages)
        if (not pages or any(type(page) is not int or not 0 <= page < shape[0]
                             for page in pages)
                or len(set(pages)) != len(pages)):
            raise SourceRefused('source pages must be unique physical pool pages')
        previous = self._frames.get(storage)
        same_frame = (previous is not None
                      and previous.transaction == transaction
                      and previous.state_epoch == state_epoch
                      and previous.shape == shape
                      and previous.pages == pages)
        if same_frame:
            existing = self._by_owner.get((transaction, layer))
            if existing is not None and existing is not previous:
                raise SourceRefused('layer already registered another source pool')
            self._by_owner[(transaction, layer)] = previous
            self._sources[storage] = source
            return previous
        if previous is not None and previous.shape == shape:
            incarnation = previous.allocation_incarnation
            versions = list(previous.versions)
        else:
            self._allocation_seq += 1
            incarnation = self._allocation_seq
            versions = [0] * shape[0]
        self._version_seq += 1
        version = self._version_seq
        for page in pages:
            versions[page] = version
        frame = _PoolFrame(storage, shape, incarnation, pages, tuple(versions),
                           version, transaction, state_epoch)
        self._frames[storage] = frame
        self._sources[storage] = source
        self._by_owner[(transaction, layer)] = frame
        self.stats['registrations'] += 1
        self.stats['frames'] = len(self._frames)
        return frame

    def register_source(self, source, pages):
        """``LocalSourceRegistry`` hook: the actual allocation record for a map."""
        shape = source_geometry(source)
        storage = storage_identity(source)
        frame = self._frames.get(storage)
        if frame is None or frame.shape != shape:
            raise SourceRefused('source pool was not registered for this admission')
        pages = tuple(pages)
        if tuple(pages) != frame.pages:
            raise SourceRefused('registered pages differ from the admitted frame')
        if not frame.versions or max(pages) >= len(frame.versions):
            raise SourceRefused('version table does not cover the registered pages')
        return AllocationRegistration(frame.allocation_incarnation, frame.versions)

    def retain(self, owner):
        """Retain the real tensor behind an admitted owner; return the reference."""
        if type(owner) is not SourceOwner:
            raise SourceRefused('retain needs the admitted source owner')
        frame = self._by_owner.get((owner.transaction, owner.layer))
        if frame is None:
            raise SourceRefused('source owner has no local registration')
        if frame.pages != tuple(owner.physical_pages):
            raise SourceRefused('source owner pages differ from the local frame')
        if frame.allocation_incarnation != owner.allocation_incarnation:
            raise SourceRefused('source owner allocation incarnation is stale')
        source = self._sources.get(frame.storage)
        if source is None:
            raise SourceRefused('source owner has no registered tensor')
        reference = _LocalSourceReference(
            owner, source=source,
            allocation_incarnation=frame.allocation_incarnation)
        self._refs[id(reference)] = reference
        self.stats['retained'] += 1
        return reference

    def assert_retained(self, reference):
        """Prove this exact retained object is live before a bucket copy."""
        live = self._refs.get(id(reference))
        if live is not reference or reference.released:
            raise SourceRefused('allocator no longer retains the source reference')
        if reference.source is None:
            raise SourceRefused('retained source tensor owner was dropped')
        return True

    def release(self, reference):
        """Drop one local reference; the pinned tensor is owned by the model."""
        self.assert_retained(reference)
        reference.released = True
        reference.source = None
        del self._refs[id(reference)]
        owner_key = (reference.owner.transaction, reference.owner.layer)
        if not any((ref.owner.transaction, ref.owner.layer) == owner_key
                   for ref in self._refs.values()):
            self._by_owner.pop(owner_key, None)
        self.stats['released'] += 1
        return True

    def inventory(self):
        return dict(self.stats, live=len(self._refs),
                    allocations=self._allocation_seq, versions=self._version_seq)


class NativeSourceBinding:
    """One admitted frame's proven worker-local source ownership.

    ``holds`` are the per-layer :class:`SparseSourceHold` objects the operation
    and the layer driver consume; ``opened`` are their registry records, closed
    only after every hold was released by a validated operation release.
    """

    def __init__(self, descriptor, *, transaction, state_epoch, rank, holds,
                 registrations, opened, layer_tables, group_rows, sources,
                 source_epochs, versions):
        if type(transaction) is not TransactionId:
            raise NativeSourceError('binding needs the admission transaction')
        if descriptor is None or getattr(descriptor, 'transaction', None) != transaction:
            raise NativeSourceError('binding descriptor belongs to another transaction')
        if type(state_epoch) is not int or state_epoch < 1:
            raise NativeSourceError('binding needs the admitted state epoch')
        if type(rank) is not int or rank < 0:
            raise NativeSourceError('binding needs the local worker rank')
        self.descriptor = descriptor
        self.transaction = transaction
        self.state_epoch = state_epoch
        self.rank = rank
        self.holds = dict(holds)
        self.registrations = dict(registrations)
        self.opened = tuple(opened)
        self.layer_tables = tuple(layer_tables)
        self.group_rows = tuple(group_rows)
        self.sources = dict(sources)
        self.source_epochs = dict(source_epochs)
        self.versions = dict(versions)
        self._closed = False

    @property
    def layers(self):
        return tuple(sorted(self.holds))

    @property
    def closed(self):
        return self._closed

    def close(self, registry):
        """Drop registry views only after every local hold was released."""
        if self._closed:
            return True
        for opened in self.opened:
            if not opened.hold.released:
                raise SourceRefused(
                    'a local source hold is still retained at lane release')
            registry.close(opened)
        self._closed = True
        return True


def bind_local_sources(dispatcher, admission, descriptor, *, rank=None):
    """Bind one PREPARE source lease to the worker's actual pinned histories.

    Refuses a missing/foreign/stale descriptor, an unsupported KV layout, a
    missing/null admitted page, a mismatched pinned source tensor and any
    partial binding before a single DMA can be enqueued.
    """
    from .wire_v1 import SourceLeaseDescriptor

    if type(descriptor) is not SourceLeaseDescriptor:
        raise SourceRefused('a WireV1 SourceLeaseDescriptor is required')
    operation = getattr(dispatcher, 'operation', None)
    lane = getattr(operation, 'lane', None)
    if type(lane) is not LaneId:
        raise SourceRefused('source binding needs the opened worker LaneId')
    if lane.role is not LaneRole.PREFILL:
        raise SourceRefused('source binding needs the prefill lane')
    if rank is None:
        rank = lane.worker_rank
    if type(rank) is not int or rank != lane.worker_rank:
        raise SourceRefused('source lease opened for another local rank')
    if rank not in descriptor.consumer_ranks:
        raise SourceRefused('descriptor does not name the local rank')
    if (admission is None
            or type(getattr(admission, 'transaction', None)) is not TransactionId
            or type(getattr(admission, 'state_epoch', None)) is not int
            or type(getattr(admission, 'block_tables', None)) is not tuple
            or type(getattr(admission, 'cow_pairs', None)) is not tuple):
        raise SourceRefused('the immutable admitted frame is required')
    valid_tokens = getattr(admission, 'chunk_end', None)
    if type(valid_tokens) is not int or valid_tokens < 1:
        raise SourceRefused('admitted chunk has no positive token frontier')
    page_count = (valid_tokens + PAGE_ROWS - 1) // PAGE_ROWS
    if not 1 <= page_count <= MAX_LAYER_PAGES:
        raise SourceRefused('admitted chunk exceeds the bounded layer page map')

    from .native_resources import _qsa_layers

    runner = getattr(dispatcher, 'runner', None)
    if runner is None:
        raise SourceRefused('source binding needs the constructed runner')
    layers = _qsa_layers(runner)
    names = tuple(name for name, _ in layers)
    layout = verify_kv_layout(runner, names)

    registry = getattr(dispatcher, 'source_registry', None)
    if registry is None:
        registry = LocalSourceRegistry(lane, ranks=(rank,), blocks_per_kv_block=1)
        dispatcher.source_registry = registry
    elif type(registry) is not LocalSourceRegistry or registry.blocks_per_kv_block != 1:
        raise SourceRefused('worker source registry belongs to another lane layout')
    else:
        registry.rebind_lane(lane)
    allocator = getattr(dispatcher, 'source_allocator', None)
    if allocator is None:
        allocator = NativeHostSourceAllocator()
        dispatcher.source_allocator = allocator
    elif type(allocator) is not NativeHostSourceAllocator:
        raise SourceRefused('worker source allocator is not the local pinned allocator')

    holds = {}
    registrations = {}
    opened_list = []
    layer_tables = []
    group_rows = []
    sources = {}
    source_epochs = {}
    versions = {}
    try:
        for name, layer in layers:
            group = layout.group(name)
            if group >= len(admission.block_tables):
                raise SourceRefused('local KV group is outside the admitted frame')
            row = admission.block_tables[group]
            if type(row) is not tuple or len(row) < page_count:
                raise SourceRefused('admitted QSA group row is shorter than the chunk')
            pages = tuple(row[:page_count])
            if any(type(page) is not int or page <= 0 for page in pages):
                raise SourceRefused(
                    'admitted QSA group row has a null/invalid source page')
            if len(set(pages)) != len(pages):
                raise SourceRefused('admitted QSA group row repeats a source page')
            source = getattr(layer, '_qsa_host_kv', None)
            if source is None:
                raise SourceRefused('QSA layer has no pinned host KV pool')
            shape = source_geometry(source)
            if max(pages) >= shape[0]:
                raise SourceRefused('admitted page is outside the pinned host pool')
            allocator.begin_frame(
                source, layer=name, transaction=admission.transaction,
                state_epoch=admission.state_epoch, pages=pages)
            record = registry.register(
                layer=name, group=group, transaction=admission.transaction,
                state_epoch=admission.state_epoch, valid_tokens=valid_tokens,
                physical_pages=pages, source=source, allocator=allocator)
            opened = registry.open(descriptor, admission, layer=name, group=group,
                                   allocator=allocator, registration=record)
            holds[name] = opened.hold
            registrations[name] = record
            opened_list.append(opened)
            layer_tables.append((name, pages))
            group_rows.append((name, group, pages))
            sources[name] = source
            source_epochs[name] = opened.source_epoch
            versions[name] = opened.versions
    except BaseException:
        for opened in reversed(opened_list):
            try:
                if not opened.hold.released:
                    opened.hold.release()
                registry.close(opened)
            except BaseException:
                pass
        raise

    return NativeSourceBinding(
        descriptor, transaction=admission.transaction,
        state_epoch=admission.state_epoch, rank=rank, holds=holds,
        registrations=registrations, opened=opened_list,
        layer_tables=layer_tables, group_rows=group_rows, sources=sources,
        source_epochs=source_epochs, versions=versions)


__all__ = [
    'LocalKvLayout', 'MAX_LAYER_PAGES', 'NativeHostSourceAllocator',
    'NativeLayoutError', 'NativeSourceBinding', 'NativeSourceError', 'ROW_WIDTH',
    'bind_local_sources', 'source_geometry', 'storage_identity', 'verify_kv_layout',
]
