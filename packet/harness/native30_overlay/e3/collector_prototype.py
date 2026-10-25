"""Standalone E0 protocol model. No vLLM, CUDA, IPC or engine integration.

Fake transport try_recv is atomic/nonblocking, unlike stock MessageQueue's
overflow path. Worker RPC replies remain FIFO; GPU completions need not.
"""
from collections import deque
from dataclasses import dataclass, field


class NotReady(RuntimeError): pass
class ProtocolError(RuntimeError): pass
class WorkerError(RuntimeError): pass
class Backpressure(RuntimeError): pass


@dataclass(frozen=True)
class Transaction:
    epoch: int
    uid: str
    generation: int


@dataclass(frozen=True)
class Reply:
    call: int
    rank: int
    kind: str
    transaction: Transaction | None = None
    value: object = None
    failed: bool = False


@dataclass
class Call:
    id: int
    kind: str
    ranks: tuple
    transaction: Transaction | None
    aggregate: object
    replies: dict = field(default_factory=dict)
    ready: bool = False
    value: object = None
    failure: str | None = None


@dataclass
class Lease:
    submissions: set = field(default_factory=set)
    pending: set = field(default_factory=set)  # (submission call, worker rank)
    retired: set = field(default_factory=set)
    cancelled: bool = False
    committed: bool = False
    failed: bool = False


class Collector:
    """Single reader, bounded polling; result() NEVER drains or waits.

    The fake workers execute commands outside this class. Register an expected
    reply only for ranks that really respond (unique_reply_rank/aggregators).
    Typed replies and transaction generations are a proposed wire contract.
    """
    def __init__(self, transport, max_pending=16):
        if max_pending < 1: raise ValueError('positive pending budget required')
        transport.claim(self)
        self.transport = transport
        self.expected = {rank: deque() for rank in transport.ranks}
        self.calls = {}; self.leases = {}; self.next_id = 0
        self.max_pending = max_pending; self.cursor = 0
        self.closing = False; self.poisoned = False

    def open(self, transaction):
        if self.closing or self.poisoned: raise ProtocolError('collector unavailable')
        if transaction in self.leases: raise ProtocolError('duplicate transaction')
        for old, lease in self.leases.items():
            if (old.epoch, old.uid) == (transaction.epoch, transaction.uid):
                if transaction.generation <= old.generation:
                    raise ProtocolError('stale generation')
                if lease.pending or any(not self.calls[i].ready for i in lease.submissions):
                    raise ProtocolError('previous generation still owned')
                if not (lease.committed or lease.cancelled):
                    raise ProtocolError('previous generation not retired')
        self.leases[transaction] = Lease()

    def issue(self, kind, *, ranks=None, transaction=None, aggregate=None):
        if self.closing or self.poisoned: raise ProtocolError('collector unavailable')
        if kind not in ('decode', 'submit', 'poll'): raise ValueError('unknown command')
        ranks = tuple(self.expected) if ranks is None else tuple(ranks)
        if not ranks or len(set(ranks)) != len(ranks) or any(r not in self.expected for r in ranks):
            raise ValueError('invalid reply ranks')
        if sum(not j.ready for j in self.calls.values()) >= self.max_pending:
            raise Backpressure('pending RPC budget')
        lease = self.leases.get(transaction)
        if kind != 'decode' and lease is None: raise ProtocolError('unknown transaction')
        if lease and (lease.committed or (lease.cancelled and kind == 'submit')):
            raise ProtocolError('noncommittable transaction')
        i = self.next_id; self.next_id += 1
        job = Call(i, kind, ranks, transaction, aggregate)
        self.calls[i] = job
        for rank in ranks: self.expected[rank].append(i)
        if kind == 'submit':
            lease.submissions.add(i)
            lease.pending.update((i, rank) for rank in ranks)
        return i

    def _receive(self, reply, rank):
        if reply.rank != rank or not self.expected[rank] or self.expected[rank][0] != reply.call:
            raise ProtocolError('unexpected/duplicate/out-of-order reply')
        job = self.calls[reply.call]
        if reply.transaction != job.transaction or reply.kind != job.kind:
            raise ProtocolError('reply type/generation mismatch')
        # Validate the entire completion batch before releasing even one lease.
        completed = []
        if job.kind == 'poll' and not reply.failed:
            if type(reply.value) is not tuple: raise ProtocolError('invalid poll payload')
            completed = [(ticket, rank) for ticket in reply.value]
            lease = self.leases[job.transaction]
            if (any(type(t) is not int for t in reply.value) or
                len(set(completed)) != len(completed) or any(t not in lease.pending for t in completed)):
                raise ProtocolError('unknown/duplicate completed ticket')
        self.expected[rank].popleft(); job.replies[rank] = reply.value
        if reply.failed:
            job.failure = 'worker_failure'
            if job.transaction is not None: self.leases[job.transaction].failed = True
        if completed:
            lease.pending.difference_update(completed); lease.retired.update(completed)
        if len(job.replies) == len(job.ranks):
            job.ready = True
            if not job.failure:
                ordered = [job.replies[r] for r in job.ranks]
                try: job.value = job.aggregate(ordered) if job.aggregate else ordered
                except Exception: job.failure = 'aggregation_failure'

    def poll(self, budget=8):
        if type(budget) is not int or budget < 1: raise ValueError('positive poll budget')
        if self.poisoned: raise ProtocolError('collector quarantined')
        ranks = tuple(self.expected); count = 0; empty = 0
        while count < budget and empty < len(ranks):
            rank = ranks[self.cursor]; self.cursor = (self.cursor+1) % len(ranks)
            reply = self.transport.try_recv(rank, self)
            if reply is None: empty += 1; continue
            try: self._receive(reply, rank)
            except ProtocolError:
                self.poisoned = True
                raise
            count += 1; empty = 0
        return count

    def result(self, call):
        job = self.calls[call]
        if not job.ready: raise NotReady('reply pending; no implicit queue drain')
        if job.failure: raise WorkerError(job.failure)
        return job.value

    def cancel(self, transaction):
        lease = self.leases[transaction]
        if lease.committed: raise ProtocolError('already committed')
        lease.cancelled = True  # No premature source/block release.

    def commit(self, transaction):
        lease = self.leases[transaction]
        if self.poisoned or self.closing or lease.cancelled or lease.committed or lease.failed:
            raise ProtocolError('commit forbidden')
        if not lease.submissions or lease.pending:
            raise NotReady('GPU completion/leases pending')
        for i in lease.submissions:
            self.result(i)  # Submission acknowledgments are required too.
        lease.committed = True

    @property
    def leased(self): return sum(len(x.pending) for x in self.leases.values())

    def shutdown(self, polls=4):
        if type(polls) is not int or polls < 0: raise ValueError('bounded shutdown polls')
        self.closing = True
        for lease in self.leases.values():
            if not lease.committed: lease.cancelled = True
        for _ in range(polls):
            if self.poisoned: break
            try: self.poll()
            except ProtocolError: break
        pending = sum(len(q) for q in self.expected.values())
        return dict(clean=not (pending or self.leased or self.poisoned),
                    pending_replies=pending,quarantined_leases=self.leased,
                    protocol_error=self.poisoned,poll_bound=polls)


class FakeWorker:
    """FIFO async_output_busy_loop model; ready_at is a virtual time, not ms.

    enqueue_output waits for the head's get_output before dequeuing the next
    item, even if that next item is a plain, already-ready submit acknowledgment.
    """
    def __init__(self): self.async_outputs=deque(); self.wire=deque()

    def output(self, reply, ready_at=0): self.async_outputs.append((ready_at,reply))

    def advance(self, now):
        while self.async_outputs and self.async_outputs[0][0] <= now:
            self.wire.append(self.async_outputs.popleft()[1])


class FakeTransport:
    def __init__(self, ranks=(0,1,2,3)):
        self.ranks=tuple(ranks); self.workers={r:FakeWorker() for r in ranks}
        self.owner=None; self.reads=0

    def claim(self, owner):
        if self.owner is not None: raise ProtocolError('response reader already owned')
        self.owner=owner

    def try_recv(self, rank, owner):
        if owner is not self.owner: raise ProtocolError('second response reader')
        self.reads+=1; wire=self.workers[rank].wire
        return wire.popleft() if wire else None

    def reply(self, collector, call, rank, value=None, ready_at=0, failed=False):
        job=collector.calls[call]
        if rank not in job.ranks: raise ProtocolError('rank must not reply')
        self.workers[rank].output(Reply(call,rank,job.kind,job.transaction,value,failed),ready_at)

    def advance(self, now):
        for worker in self.workers.values(): worker.advance(now)


def head_of_line_demo():
    t=FakeTransport(); c=Collector(t); txn=Transaction(1,'synthetic',1); c.open(txn)
    decode=c.issue('decode',ranks=(3,))
    submit=c.issue('submit',transaction=txn)
    t.reply(c,decode,3,'synthetic decode result',ready_at=10)
    for rank in t.ranks: t.reply(c,submit,rank,'submitted',ready_at=0)
    t.advance(0); c.poll()
    before=dict(submit_ack_ready=c.calls[submit].ready,
                ready_ack_ranks=sorted(c.calls[submit].replies),leased=c.leased)
    t.advance(10); c.poll()
    after=dict(decode_ready=c.calls[decode].ready,submit_ack_ready=c.calls[submit].ready,
               leased=c.leased,committed=c.leases[txn].committed)
    return dict(before=before,after=after,virtual_decode_ready_at=10,
                verdict='HEAD_OF_LINE_CONFIRMED_IN_MODEL',
                limitation='no timing calibration, vLLM transport adapter, GPU or engine implementation')


if __name__=='__main__':
    import json
    print(json.dumps(head_of_line_demo(),sort_keys=True))
