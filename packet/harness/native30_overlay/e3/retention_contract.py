"""E0 CPU ownership contract; no CUDA, vLLM adapter, allocation or performance claim.

Completion is injected by tests. An engine adapter must prove it with producer,
DMA and consumer events and hold real block references. Metadata is not a fence.
"""
from dataclasses import dataclass, field


class ContractError(RuntimeError): pass
class Pending(ContractError): pass


@dataclass(frozen=True)
class RequestKey:
    epoch: int
    uid: str
    generation: int


@dataclass(frozen=True)
class PageKey:
    epoch: int
    layer: str
    physical: int
    incarnation: int


@dataclass
class Page:
    key: PageKey
    valid: int
    version: int = 0
    readonly: bool = False
    holders: set = field(default_factory=set)
    readers: set = field(default_factory=set)
    writer: object = None


class Pool:
    """Explicit allocation incarnations and pins; deliberately stricter than raw IDs."""
    def __init__(self, epoch, page_tokens=4048):
        if epoch < 1 or page_tokens < 1: raise ValueError('positive pool dimensions')
        self.epoch, self.page_tokens = epoch, page_tokens
        self.pages = {}; self.incarnations = {}

    def allocate(self, layer, physical, valid=0, readonly=False):
        slot = (layer, physical)
        if not layer or physical <= 0 or not 0 <= valid <= self.page_tokens:
            raise ContractError('invalid allocation')
        if slot in self.pages: raise ContractError('physical block already allocated')
        generation = self.incarnations.get(slot, 0) + 1
        key = PageKey(self.epoch, layer, physical, generation)
        self.incarnations[slot] = generation
        self.pages[slot] = Page(key, valid, readonly=readonly)
        return key

    def page(self, key):
        page = self.pages.get((key.layer, key.physical))
        if page is None or page.key != key: raise ContractError('stale page allocation')
        return page

    def free(self, key):
        page = self.page(key)
        if page.holders or page.readers or page.writer: raise Pending('page still leased')
        del self.pages[(key.layer, key.physical)]

    def pin(self, keys, owner):
        pages = [self.page(k) for k in keys]  # Validate all before any mutation.
        if any(p.writer is not None for p in pages): raise Pending('producer pending')
        for p in pages: p.holders.add(owner)

    def unpin(self, keys, owner):
        pages = [self.page(k) for k in keys]
        if any(owner not in p.holders for p in pages): raise ContractError('unowned source')
        if any(p.writer is not None or p.readers for p in pages): raise Pending('source still consumed')
        for p in pages: p.holders.remove(owner)

    def held_count(self):
        return sum(len(p.holders) for p in self.pages.values())


@dataclass(frozen=True)
class Snapshot:
    page: PageKey
    version: int
    rows: int


@dataclass(frozen=True)
class Transfer:
    request: RequestKey
    chunk: int
    layer: str
    logical_start: int
    sources: tuple  # Snapshot values; whole valid pages, partial final page.
    row_bytes: int = 1024

    @property
    def bytes(self): return sum(s.rows for s in self.sources) * self.row_bytes


@dataclass(frozen=True)
class Ticket:
    request: RequestKey
    chunk: int
    sequence: int
    kind: str
    layer: str | None


@dataclass
class Work:
    ticket: Ticket
    keys: tuple
    transfer: Transfer | None = None
    new_valid: int | None = None


class RetentionBank:
    """One active request, exactly six local histories, fixed logical capacity.

    Ordinary serial E2 can implement this contract without async RPC plumbing.
    No tensor buffers are allocated here. Scope is main QSA history; finalize
    tickets stand for, but do not implement, GDN/conv/PLE/indexer completion.
    """
    def __init__(self, pool, rank, layers, capacity_pages=65):
        layers = tuple(layers)
        if len(layers) != 6 or len(set(layers)) != 6 or capacity_pages < 1:
            raise ValueError('six unique layers and positive capacity required')
        self.pool, self.rank, self.layers = pool, rank, layers
        self.capacity_pages = capacity_pages
        self.active = None; self.generations = {}

    @property
    def reserved_bytes(self): return len(self.layers)*self.capacity_pages*self.pool.page_tokens*1024

    def open(self, request, committed=0):
        if request.epoch != self.pool.epoch or not request.uid or request.generation < 1:
            raise ContractError('invalid request identity')
        if committed < 0 or committed > self.capacity_pages*self.pool.page_tokens:
            raise ContractError('initial frontier exceeds capacity')
        if self.active is not None and not self.active.retired:
            raise Pending('active request still owns retained slab')
        if request.generation <= self.generations.get(request.uid, 0):
            raise ContractError('stale request generation')
        self.generations[request.uid] = request.generation
        self.active = History(self, request, committed)
        return self.active


class History:
    def __init__(self, bank, request, committed):
        self.bank, self.pool, self.request = bank, bank.pool, request
        self.committed = committed; self.published = committed
        self.chunk = 0; self.end = None; self.tables = {}
        self.held = set(); self.copied = {}; self.pending = {}
        self.sequence = 0; self.attended = set(); self.finalized = False
        self.cancelled = False; self.quarantined = False; self.retired = False
        self.commit_count = 0

    def available(self):
        if self.retired or self.cancelled or self.quarantined:
            raise ContractError('history not submittable')

    def reserve(self, end, tables):
        self.available()
        if self.end is not None or self.pending: raise Pending('chunk already reserved')
        n = (end+self.pool.page_tokens-1)//self.pool.page_tokens
        if not self.committed < end <= self.bank.capacity_pages*self.pool.page_tokens:
            raise ContractError('invalid chunk frontier')
        if set(tables) != set(self.bank.layers): raise ContractError('missing or extra local history')
        frozen = {name: tuple(tables[name]) for name in self.bank.layers}
        keys = set()
        for layer, row in frozen.items():
            if len(row) != n or len(set(row)) != n: raise ContractError('invalid block table')
            for logical, key in enumerate(row):
                page = self.pool.page(key)
                prefix = max(0, min(self.pool.page_tokens, self.committed-logical*self.pool.page_tokens))
                if key.layer != layer or page.valid < prefix:
                    raise ContractError('mapping lacks committed source prefix')
                if page.writer is not None or page.readers: raise Pending('source is busy')
                keys.add(key)
        # The previous chunk has no consumers. Hold new sources before old release.
        added, removed = keys-self.held, self.held-keys
        for key in self.held:
            if self.request not in self.pool.page(key).holders:
                raise ContractError('previous source lease lost')
        for key in removed:
            page = self.pool.page(key)
            if page.writer is not None or page.readers: raise Pending('old source is busy')
        self.pool.pin(added, self.request)
        self.pool.unpin(removed, self.request)
        self.held = keys; self.tables = frozen; self.end = end; self.chunk += 1
        self.attended = set(); self.finalized = False
        # Retained identity includes physical incarnation; remapping never reuses stale data.
        self.copied = {(layer, i): snap for (layer, i), snap in self.copied.items()
                       if i < len(frozen[layer]) and snap.page == frozen[layer][i]}

    def required(self, logical):
        if self.end is None: raise ContractError('no active chunk')
        if type(logical) is not int or not 0 <= logical < (self.end+self.pool.page_tokens-1)//self.pool.page_tokens:
            raise ContractError('invalid logical page')
        return min(self.pool.page_tokens, self.end-logical*self.pool.page_tokens)

    def ticket(self, kind, layer, keys, **kwargs):
        self.sequence += 1
        t = Ticket(self.request, self.chunk, self.sequence, kind, layer)
        self.pending[t] = Work(t, tuple(keys), **kwargs)
        return t

    def store(self, layer, logical, valid):
        """Reserve a producer write. Completion advances version, not publication."""
        self.available()
        if self.end is None or layer in self.attended: raise ContractError('write outside layer phase')
        key = self.tables[layer][logical]; page = self.pool.page(key)
        if page.readonly or page.holders != {self.request}: raise ContractError('write requires private CoW destination')
        if page.writer is not None or page.readers: raise Pending('write conflicts with readers')
        if not page.valid < valid <= self.required(logical): raise ContractError('write must append valid rows')
        t = self.ticket('producer', layer, (key,), new_valid=valid)
        page.writer = t
        return t

    def desired(self, layer, logical):
        key = self.tables[layer][logical]; page = self.pool.page(key)
        rows = self.required(logical)
        if self.request not in page.holders: raise ContractError('source lease lost')
        if page.writer is not None or page.valid < rows: raise Pending('required source rows not yet visible')
        return Snapshot(key, page.version, rows)

    def plan(self, layer):
        """Conservative full-valid-page refresh; no data read and no state change."""
        self.available()
        if self.end is None: raise ContractError('no reserved chunk')
        dirty = []
        for i in range(len(self.tables[layer])):
            snap = self.desired(layer, i)
            if self.copied.get((layer, i)) != snap: dirty.append((i, snap))
        plans = []; group = []
        def flush():
            if group: plans.append(Transfer(self.request, self.chunk, layer, group[0][0], tuple(s for _, s in group)))
        for i, snap in dirty:
            if group and (len(group) == 4 or i != group[-1][0]+1 or
                          snap.page.physical != group[-1][1].page.physical+1 or
                          group[-1][1].rows != self.pool.page_tokens):
                flush(); group = []
            group.append((i, snap))
        flush()
        return tuple(plans)

    def copy(self, transfer):
        self.available()
        if any(w.ticket.kind == 'copy' for w in self.pending.values()):
            raise Pending('one outstanding DMA per rank across all layers')
        if transfer.request != self.request or transfer.chunk != self.chunk:
            raise ContractError('stale transfer identity')
        # Revalidate the entire transfer against current versions before any pin.
        if transfer not in self.plan(transfer.layer): raise ContractError('stale/forged DMA plan')
        keys = tuple(s.page for s in transfer.sources)
        t = self.ticket('copy', transfer.layer, keys, transfer=transfer)
        for key in keys: self.pool.page(key).readers.add(t)
        return t

    def attend(self, layer):
        self.available()
        if layer in self.attended or any(w.ticket.kind == 'attention' and w.ticket.layer == layer for w in self.pending.values()):
            raise ContractError('duplicate attention')
        if self.plan(layer): raise Pending('resident history not ready')
        if any(w.ticket.layer == layer for w in self.pending.values()): raise Pending('layer work unfinished')
        keys = self.tables[layer]
        t = self.ticket('attention', layer, keys)
        for key in keys: self.pool.page(key).readers.add(t)
        return t

    def finalize(self):
        self.available()
        if self.end is None or self.pending or self.attended != set(self.bank.layers):
            raise Pending('all six histories must finish before state finalization')
        if self.finalized: raise ContractError('duplicate state finalization')
        return self.ticket('finalize', None, ())

    def complete(self, ticket, success=True):
        """Inject an observed completion. Failure is quarantined, never released."""
        work = self.pending.get(ticket)
        if work is None or ticket.request != self.request or ticket.chunk != self.chunk:
            raise ContractError('stale/duplicate completion')
        if self.quarantined: raise ContractError('completion after worker/event loss requires recovery')
        if not success:
            self.quarantined = True
            return
        pages = [self.pool.page(k) for k in work.keys]
        if any(self.request not in p.holders for p in pages): raise ContractError('completion lost source lease')
        if ticket.kind == 'producer':
            if any(p.writer != ticket for p in pages): raise ContractError('producer ownership lost')
        elif ticket.kind in ('copy', 'attention'):
            if any(ticket not in p.readers for p in pages): raise ContractError('reader ownership lost')
            if ticket.kind == 'copy':
                for offset, snap in enumerate(work.transfer.sources):
                    if self.desired(ticket.layer, work.transfer.logical_start+offset) != snap:
                        raise ContractError('source changed under DMA')
        if ticket.kind == 'producer':
            for page in pages:
                page.valid = work.new_valid; page.version += 1; page.writer = None
        elif ticket.kind in ('copy', 'attention'):
            for page in pages: page.readers.remove(ticket)
            if ticket.kind == 'copy':
                for offset, snap in enumerate(work.transfer.sources):
                    self.copied[(ticket.layer, work.transfer.logical_start+offset)] = snap
            else: self.attended.add(ticket.layer)
        elif ticket.kind == 'finalize': self.finalized = True
        del self.pending[ticket]

    def ready(self):
        self.available()
        if self.end is None or self.pending or not self.finalized or self.attended != set(self.bank.layers):
            raise Pending('chunk not complete')
        for layer in self.bank.layers:
            if self.plan(layer): raise ContractError('source changed before commit')

    def cancel(self):
        if self.retired: raise ContractError('already retired')
        self.cancelled = True

    def retire(self):
        if self.pending or self.quarantined: raise Pending('retirement awaits proven quiescence')
        if self.end is not None and not self.cancelled: raise ContractError('uncommitted chunk must cancel')
        if self.retired: raise ContractError('duplicate retirement')
        self.pool.unpin(self.held, self.request)
        self.held.clear(); self.copied.clear(); self.retired = True

    def shutdown(self):
        self.cancel()
        if not self.pending and not self.quarantined: self.retire()
        return dict(clean=self.retired, pending=len(self.pending), held=len(self.held), quarantined=self.quarantined)


class Publication:
    """Four-rank commit barrier, CPU model only. No optimistic prefix publication."""
    def __init__(self, ranks=(0, 1, 2, 3)):
        if len(ranks) != 4 or len(set(ranks)) != 4: raise ValueError('four distinct ranks required')
        self.ranks = frozenset(ranks); self.records = []

    def commit(self, histories):
        histories = tuple(histories)
        if len(histories) != 4 or {h.bank.rank for h in histories} != self.ranks:
            raise ContractError('commit requires every rank exactly once')
        keys = {(h.request, h.chunk, h.committed, h.end) for h in histories}
        if len(keys) != 1: raise ContractError('rank chunk/frontier mismatch')
        for h in histories: h.ready()  # All preflight; no partial publication.
        record = next(iter(keys))
        if record in self.records: raise ContractError('duplicate commit')
        for h in histories:
            h.committed = h.published = h.end
            h.end = None; h.commit_count += 1
        self.records.append(record)
        return record
