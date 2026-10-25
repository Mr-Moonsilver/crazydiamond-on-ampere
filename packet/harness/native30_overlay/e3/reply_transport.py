"""E3 receiver checkpoint; not installed in the engine overlay yet.

One owned receiver thread per reply MQ. The dispatcher only polls local results;
it never calls MessageQueue.dequeue or FutureWrapper.result. Each expected FIFO
reply is registered BEFORE its broadcast. A timeout/error poisons the transport:
never retry dequeue after a possibly consumed overflow marker. Bounded credits
include ready-but-unclaimed payloads. GPU completion/publication is NOT inferred
from an RPC reply. The executor integration must route ALL response consumers
here at an idle handoff, and keep MQ objects alive after an unclean close.

No CUDA or vLLM imports. Thread handoff, ZMQ/native shmem, reply unpickling and
executor replacement require an integration test; exact-source CPU fixtures
establish ordering only. Worker async-output head-of-line blocking is unchanged.
"""
from collections import deque
from dataclasses import dataclass, field
import math
from queue import Empty, Queue
import threading
import time


class NotReady(RuntimeError): pass
class TransportError(RuntimeError): pass
class Backpressure(RuntimeError): pass


_CLAIM_LOCK = threading.Lock()


@dataclass
class _Call:
    sequence: int
    kind: str
    ranks: tuple
    registered_s: float
    deadline_s: float
    aggregate: object
    replies: dict = field(default_factory=dict)
    received_s: dict = field(default_factory=dict)
    value: object = None
    ready: bool = False


class ReplyTransport:
    """Dispatcher-owned submit/poll/take, with bounded per-worker receivers.

    queues maps rank -> exclusively handed-off MessageQueue-compatible reader.
    success_status must be the actual WorkerProc.ResponseStatus.SUCCESS enum.
    send is the existing broadcast enqueue; it may block on broadcast capacity,
    so this adapter makes REPLY reception nonblocking, not RPC submission itself.
    No stock FutureWrapper may retain access to these queues after handoff.
    """
    def __init__(self, queues, success_status, *, handoff_idle, max_calls=16,
                 clock=time.monotonic):
        if handoff_idle is not True:
            raise ValueError('idle executor handoff required')
        if not queues or any(type(r) is not int or r < 0 for r in queues):
            raise ValueError('nonempty rank map required')
        if len({id(q) for q in queues.values()}) != len(queues):
            raise ValueError('reply MQ shared between ranks')
        if any(not callable(getattr(q, 'dequeue', None)) for q in queues.values()):
            raise ValueError('reply readers must implement dequeue')
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError('positive credit budget required')
        self._owner_thread = threading.get_ident()
        self._clock = clock
        self._status = success_status
        self._max_calls = max_calls
        self._condition = threading.Condition()
        self._todo = {r: deque() for r in queues}
        self._queues = dict(queues)  # Retained even after unclean close.
        self._done = Queue()
        self._calls = {}
        self._next = 0
        self._failure = None
        self._closing = False
        self._inflight = set()
        self._threads = {}
        with _CLAIM_LOCK:
            if any(getattr(q, '_flash_s5_receiver_owner', None) is not None for q in queues.values()):
                raise TransportError('reply MQ already owned')
            claimed = []
            try:
                for q in queues.values():
                    q._flash_s5_receiver_owner = self
                    claimed.append(q)
            except BaseException:
                for q in claimed:
                    q._flash_s5_receiver_owner = None
                raise
        for rank in queues:
            t = threading.Thread(target=self._receive_loop, args=(rank,),
                                 name=f'flash-s5-reply-{rank}', daemon=True)
            self._threads[rank] = t
            t.start()

    def _owner(self):
        if threading.get_ident() != self._owner_thread:
            raise TransportError('dispatcher ownership violation')

    def _healthy(self):
        if self._failure is not None:
            raise TransportError(self._failure)
        if self._closing:
            raise TransportError('transport closing')

    def _poison(self, reason):
        # Caller holds condition. Never include worker exception text/payloads.
        if self._failure is None:
            self._failure = reason
        self._condition.notify_all()

    def submit(self, kind, send, *, ranks=None, timeout_s=300.0, aggregate=None):
        """Register before broadcasting, including unique-reply/aggregator ranks.

        send receives the local sequence solely for optional typed E3 envelopes.
        Stock wire replies lack IDs; FIFO is the only correlation until the E3
        worker methods carry/validate a transaction generation in their payload.
        """
        self._owner()
        if not isinstance(kind, str) or not kind:
            raise ValueError('reply kind required')
        ranks = tuple(self._queues) if ranks is None else tuple(ranks)
        if not ranks or len(set(ranks)) != len(ranks) or any(r not in self._queues for r in ranks):
            raise ValueError('invalid expected reply ranks')
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('positive finite reply timeout required')
        if not callable(send) or (aggregate is not None and not callable(aggregate)):
            raise ValueError('invalid callback')
        with self._condition:
            self._healthy()
            if len(self._calls) >= self._max_calls:
                raise Backpressure('unclaimed RPC credit budget')
            sequence = self._next; self._next += 1
            now = self._clock()
            call = _Call(sequence, kind, ranks, now, now + timeout_s, aggregate)
            self._calls[sequence] = call
            for rank in ranks:
                self._todo[rank].append((sequence, call.deadline_s))
            self._condition.notify_all()
        try:
            send(sequence)
        except BaseException:
            # Broadcast may have partially succeeded. No rollback/reuse of IDs.
            with self._condition:
                self._poison('broadcast_failed')
            raise
        return sequence

    def _receive_loop(self, rank):
        while True:
            with self._condition:
                while not self._todo[rank] and not self._closing and self._failure is None:
                    self._condition.wait()
                if self._closing or self._failure is not None:
                    return
                sequence, deadline = self._todo[rank].popleft()
                self._inflight.add((rank, sequence))
            try:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise TimeoutError
                # Once only. The source may consume an overflow marker before
                # timing out on the payload. Retrying could relabel next reply.
                wire = self._queues[rank].dequeue(timeout=remaining)
                received = self._clock()
                if received > deadline:
                    raise TimeoutError
                if not isinstance(wire, tuple) or len(wire) != 2:
                    raise ValueError
                status, value = wire
                if status != self._status:
                    with self._condition:
                        self._poison('worker_failure')
                    return
                with self._condition:
                    if not self._closing and self._failure is None:
                        self._done.put_nowait((sequence, rank, value, received))
            except BaseException as exc:
                with self._condition:
                    self._poison('reply_timeout' if isinstance(exc, TimeoutError) else 'reply_transport_failure')
                return
            finally:
                with self._condition:
                    self._inflight.discard((rank, sequence))
                    self._condition.notify_all()

    def poll(self, budget=8):
        """Bounded local processing. Aggregators run here, never in receivers.

        This uses short Python locks/GIL and is not a realtime/wait-free promise.
        Aggregate callbacks themselves must not perform blocking engine work.
        """
        self._owner()
        if type(budget) is not int or budget < 1:
            raise ValueError('positive poll budget required')
        with self._condition:
            self._healthy()
        n = 0
        while n < budget:
            try:
                sequence, rank, value, received = self._done.get_nowait()
            except Empty:
                break
            call = self._calls[sequence]
            if rank not in call.ranks or rank in call.replies:
                with self._condition:
                    self._poison('reply_correlation_failure')
                raise TransportError('reply_correlation_failure')
            call.replies[rank] = value; call.received_s[rank] = received
            if len(call.replies) == len(call.ranks):
                ordered = [call.replies[r] for r in call.ranks]
                try:
                    call.value = call.aggregate(ordered) if call.aggregate else ordered
                except BaseException:
                    with self._condition:
                        self._poison('aggregation_failed')
                    raise
                call.ready = True
            n += 1
        with self._condition:
            self._healthy()
        return n

    def take(self, sequence):
        """Consume once; no implicit poll/drain/wait, free the payload credit.

        Ack lag is register-to-receive, not GPU time or measured wire latency.
        No cancellation/lease retirement is inferred here.
        """
        self._owner()
        with self._condition:
            self._healthy()
            call = self._calls[sequence]
            if not call.ready:
                raise NotReady('reply pending')
            del self._calls[sequence]
        return call.value, {r: call.received_s[r] - call.registered_s for r in call.ranks}

    def close(self, timeout_s=0.0):
        """Bounded join, not forced cancellation or proof of GPU retirement.

        A blocked overflow receive may outlive this call. Return unclean and
        retain MQs/owned state; the executor must quarantine and terminate the
        owning process before destroying transport resources. No mq.shutdown()
        from this thread: it cannot reliably cancel the overflow socket poll.
        """
        self._owner()
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError('finite nonnegative close timeout required')
        deadline = time.monotonic() + timeout_s
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        for t in self._threads.values():
            t.join(max(0.0, deadline - time.monotonic()))
        with self._condition:
            live = [rank for rank, t in self._threads.items() if t.is_alive()]
            clean = not (live or self._calls or self._failure)
            if clean:
                with _CLAIM_LOCK:
                    for q in self._queues.values():
                        if q._flash_s5_receiver_owner is self:
                            q._flash_s5_receiver_owner = None
            return dict(clean=clean,
                        live_receivers=live, outstanding_calls=len(self._calls),
                        inflight_receives=len(self._inflight), failure=self._failure,
                        scope='transport only; no GPU completion or source-lease release implied')
