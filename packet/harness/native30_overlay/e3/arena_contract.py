"""E0 single-arena ownership model. CPU only; not engine allocator integration.

Pool pins and completions here are injected models. An engine binding must hold
real block references and prove producer/DMA/consumer completion. No persistent
history slabs or cached destination-validity survive a bucket consumption.
"""
from dataclasses import dataclass

from retention_contract import ContractError, Pending, RequestKey, Snapshot


@dataclass(frozen=True)
class Owner:
    request: RequestKey
    chunk: int
    layer: str


@dataclass(frozen=True)
class Piece:
    source_page: int
    arena_page: int
    pages: int


@dataclass(frozen=True)
class Bucket:
    owner: Owner
    epoch: int
    lo: int
    hi: int
    sources: tuple
    pieces: tuple


@dataclass(frozen=True)
class Completion:
    owner: Owner
    epoch: int
    serial: int
    kind: str


class Arena:
    """One existing arena per rank; independent source history per local layer.

    Request uid should be a bounded worker-slot namespace; generation is the
    monotonic request incarnation for that slot. Logical request IDs live in the
    worker mirror. Retired-generation floors persist until this pool epoch ends.
    """
    def __init__(self, pool, layers, arena_pages=23, row_bytes=1024):
        self.layers=tuple(layers)
        if (len(self.layers)!=6 or len(set(self.layers))!=6 or
                type(arena_pages) is not int or arena_pages<1 or
                type(row_bytes) is not int or row_bytes<1):
            raise ValueError('six layers and positive arena dimensions required')
        self.pool=pool;self.arena_pages=arena_pages;self.row_bytes=row_bytes
        self.requests={};self.generations={};self.cancelled=set();self.completed={}
        self.owner=None;self.sources=();self.next_page=0;self.bucket=None
        self.epoch=0;self.serial=0;self.pending=None;self.state='idle';self.failure=None

    @property
    def storage_bytes(self):
        return self.arena_pages*self.pool.page_tokens*self.row_bytes

    @property
    def incremental_kv_bytes(self):
        return 0  # Caller supplies the already-accounted arena, never a new slab.

    def register(self, request):
        if (not isinstance(request,RequestKey) or request.epoch!=self.pool.epoch or
                not request.uid or type(request.generation) is not int or request.generation<1):
            raise ContractError('invalid request identity')
        if request.uid in self.requests:raise Pending('request slot occupied')
        if request.generation<=self.generations.get(request.uid,0):raise ContractError('retired generation')
        self.requests[request.uid]=request;self.generations[request.uid]=request.generation

    def _live(self, request, allow_cancel=False):
        if self.requests.get(request.uid)!=request:raise ContractError('foreign or retired request')
        if not allow_cancel and request in self.cancelled:raise ContractError('request cancelled')

    def _healthy(self):
        if self.state=='quarantined':raise Pending('unproven completion; worker recovery required')

    def _source(self, snap):
        page=self.pool.page(snap.page)
        if page.writer is not None or page.valid<snap.rows:raise Pending('source producer incomplete')
        if page.version!=snap.version:raise ContractError('source version changed')
        if self.owner not in page.holders:raise ContractError('source lease lost')
        return page

    def acquire(self, owner, keys, valid_tokens):
        self._healthy();self._live(owner.request)
        if self.state!='idle':raise Pending('arena occupied')
        keys=tuple(keys)
        if (owner.layer not in self.layers or type(owner.chunk) is not int or owner.chunk<0 or
                owner.chunk<=self.completed.get((owner.request,owner.layer),-1) or
                type(valid_tokens) is not int or valid_tokens<1 or
                len(keys)!=(valid_tokens+self.pool.page_tokens-1)//self.pool.page_tokens or
                len(keys)!=len(set(keys))):
            raise ContractError('invalid layer/chunk/table extent')
        snapshots=[]
        for i,key in enumerate(keys):
            p=self.pool.page(key);rows=min(self.pool.page_tokens,valid_tokens-i*self.pool.page_tokens)
            if key.layer!=owner.layer:raise ContractError('wrong source layer')
            if p.writer is not None or p.valid<rows:raise Pending('source producer incomplete')
            snapshots.append(Snapshot(key,p.version,rows))
        # Whole-layer source lease is conservative. Step 2 must bind it to real
        # allocator ownership; a Python set is not a scheduler-side pin.
        self.pool.pin(keys,owner)
        self.owner=owner;self.sources=tuple(snapshots);self.next_page=0;self.state='owned'

    def begin_fill(self):
        self._healthy()
        if self.owner is None:raise ContractError('no arena owner')
        self._live(self.owner.request)
        if self.state!='owned':raise Pending('previous fill/consumer unfinished')
        if self.next_page>=len(self.sources):raise ContractError('all buckets consumed')
        lo=self.next_page;hi=min(lo+self.arena_pages,len(self.sources))
        snapshots=self.sources[lo:hi]
        for snap in snapshots:self._source(snap)
        pieces=[];i=0
        while i<len(snapshots):
            n=1
            while (i+n<len(snapshots) and n<4 and
                   snapshots[i+n].page.physical==snapshots[i].page.physical+n):n+=1
            pieces.append(Piece(snapshots[i].page.physical,i,n));i+=n
        self.epoch+=1
        self.bucket=Bucket(self.owner,self.epoch,lo,hi,snapshots,tuple(pieces))
        self.serial+=1;self.pending=Completion(self.owner,self.epoch,self.serial,'copy')
        for snap in snapshots:self.pool.page(snap.page).readers.add(self.pending)
        self.state='filling'
        return self.bucket,self.pending

    def _ticket(self, ticket, kind):
        self._healthy()
        if ticket!=self.pending or ticket.kind!=kind or ticket.owner!=self.owner or ticket.epoch!=self.epoch:
            raise ContractError('stale, duplicate or foreign completion')

    def _failed(self, reason):
        self.failure=reason;self.state='quarantined'

    def complete_fill(self, ticket, success=True):
        self._ticket(ticket,'copy')
        if type(success) is not bool:raise ContractError('completion result must be boolean')
        if not success:self._failed('copy completion failed');return
        try:
            pages=[self._source(s) for s in self.bucket.sources]
            if any(ticket not in p.readers for p in pages):raise ContractError('DMA read lease lost')
        except (ContractError,Pending):
            self._failed('source changed during copy');raise
        for p in pages:p.readers.remove(ticket)
        self.pending=None;self.state='ready'

    def begin_attention(self, bucket):
        self._healthy()
        if self.owner is None:raise ContractError('no arena owner')
        self._live(self.owner.request)
        if bucket!=self.bucket:raise ContractError('stale arena validity')
        if self.state!='ready':raise Pending('arena bytes not ready')
        for snap in bucket.sources:self._source(snap)
        self.serial+=1;self.pending=Completion(self.owner,self.epoch,self.serial,'consume')
        self.state='consuming'
        return self.pending

    def complete_attention(self, ticket, success=True):
        self._ticket(ticket,'consume')
        if type(success) is not bool:raise ContractError('completion result must be boolean')
        if not success:self._failed('attention completion failed');return
        try:
            for snap in self.bucket.sources:self._source(snap)
        except (ContractError,Pending):
            self._failed('source changed during attention');raise
        self.next_page=self.bucket.hi
        # No 'copied' dictionary: arena validity dies after EACH consumer.
        self.bucket=None;self.pending=None;self.state='owned'

    def cancel(self, request):
        self._live(request,allow_cancel=True);self.cancelled.add(request)

    def release(self):
        self._healthy()
        if self.owner is None:raise ContractError('no arena owner')
        if self.pending is not None:raise Pending('DMA/consumer has not completed')
        cancelled=self.owner.request in self.cancelled
        if not cancelled and (self.state!='owned' or self.next_page!=len(self.sources)):
            raise Pending('layer still has unconsumed buckets')
        self.pool.unpin(tuple(s.page for s in self.sources),self.owner)
        if not cancelled:self.completed[(self.owner.request,self.owner.layer)]=self.owner.chunk
        self.owner=None;self.sources=();self.bucket=None;self.next_page=0;self.state='idle'

    def retire(self, request):
        self._live(request,allow_cancel=True)
        if self.owner is not None and self.owner.request==request:raise Pending('arena owner must drain before retirement')
        del self.requests[request.uid];self.cancelled.discard(request)
        self.completed={k:v for k,v in self.completed.items() if k[0]!=request}

    def shutdown(self):
        for request in self.requests.values():self.cancel(request)
        if self.owner is not None and self.pending is None and self.state!='quarantined':self.release()
        for request in tuple(self.requests.values()):
            if self.owner is None or self.owner.request!=request:self.retire(request)
        return self.counters()

    def counters(self):
        return dict(state=self.state,active_owners=int(self.owner is not None),
                    source_leases=len(self.sources),pending_dma=int(self.pending is not None and self.pending.kind=='copy'),
                    pending_consumers=int(self.pending is not None and self.pending.kind=='consume'),
                    valid_arena_pages=(self.bucket.hi-self.bucket.lo if self.state in ('ready','consuming') else 0),
                    arena_epoch=self.epoch,registered_requests=len(self.requests),
                    quarantined=self.state=='quarantined',incremental_kv_bytes=0,
                    reused_arena_bytes=self.storage_bytes)
