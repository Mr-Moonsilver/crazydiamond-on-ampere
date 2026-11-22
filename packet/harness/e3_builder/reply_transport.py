"""E3 process-owned reply transport with serialized cross-thread RPC submission.

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
import os
from queue import Empty, Queue
import threading
import time


class NotReady(RuntimeError): pass
class TransportError(RuntimeError): pass
class WorkerRPCError(TransportError):
    """Bounded metadata only; never serializes worker exception payloads."""
    def __init__(self, record):
        self.record = dict(record)
        super().__init__('worker-fatal rank=%s method=%s error=%s code=%s' % (
            record.get('rank'), record.get('method'), record.get('error_type'), record.get('code')))


class Backpressure(RuntimeError): pass


class DirectReply:
    """A worker reply written by its RPC thread ahead of pending async output.

    Only command-flow native kinds the engine registered as direct are ever
    wrapped. It carries the transport sequence the engine registered (the
    worker's progress meta): the receiver accepts it only as the exact head of
    that rank's direct FIFO and poisons on anything else. Any other wire
    matches the ordered FIFO. Pickles by reference to this module.
    """
    __slots__ = ('sequence', 'value')

    def __init__(self, sequence, value):
        self.sequence = sequence
        self.value = value

    def __reduce__(self):
        return (DirectReply, (self.sequence, self.value))


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
    """Process-owned submit/poll/take, with bounded per-worker receivers.

    queues maps rank -> exclusively handed-off MessageQueue-compatible reader.
    success_status must be the actual WorkerProc.ResponseStatus.SUCCESS enum.
    send is the existing broadcast enqueue; it may block on broadcast capacity,
    so this adapter makes REPLY reception nonblocking, not RPC submission itself.
    No stock FutureWrapper may retain access to these queues after handoff.
    """
    def __init__(self, queues, success_status, *, handoff_idle, max_calls=16,
                 clock=time.monotonic, direct_replies=False):
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
        self._owner_pid = os.getpid()
        # One lock covers registration through the complete broadcast write:
        # two submitting threads must not reverse the worker FIFO after assigning
        # sequence ids. Receivers use only _condition and keep draining replies.
        self._broker = threading.RLock()
        self._shutdown_lock = threading.RLock()
        self._sending_thread = None
        self._clock = clock
        self._status = success_status
        self._max_calls = max_calls
        self._condition = threading.Condition()
        self._todo = {r: deque() for r in queues}
        if direct_replies not in (False, True):
            raise ValueError('direct_replies must be a bool')
        # Held-f burst only: a second per-rank expectation FIFO for replies the
        # worker writes from its RPC thread (DirectReply). Off, it does not exist
        # and the receiver below is the unchanged single-FIFO loop.
        self._direct = {r: deque() for r in queues} if direct_replies else None
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
        loop = self._receive_loop if self._direct is None else self._receive_demux_loop
        for rank in queues:
            t = threading.Thread(target=loop, args=(rank,),
                                 name=f'flash-s5-reply-{rank}', daemon=True)
            self._threads[rank] = t
            t.start()

    def _owner(self):
        if os.getpid() != self._owner_pid:
            raise TransportError('transport process ownership violation')

    def _healthy(self):
        if self._failure is not None:
            if isinstance(self._failure, BaseException):
                raise self._failure
            raise TransportError(self._failure)
        if self._closing:
            raise TransportError('transport closing')

    def _poison(self, reason):
        # Caller holds condition. Never include worker exception text/payloads.
        if self._failure is None:
            self._failure = reason
        self._condition.notify_all()

    def fail(self, failure):
        """Latch an out-of-band worker failure before executor teardown."""
        if not isinstance(failure, TransportError):
            raise TypeError('typed transport failure required')
        with self._condition:
            self._poison(failure)

    def submit(self, kind, send, *, ranks=None, timeout_s=300.0, aggregate=None, precheck=None,
               direct=False):
        """Register before broadcasting, including unique-reply/aggregator ranks.

        send receives the local sequence solely for optional typed E3 envelopes.
        Stock wire replies lack IDs; FIFO is the only correlation until the E3
        worker methods carry/validate a transaction generation in their payload.
        ``direct`` registers in the direct FIFO (the worker writes that reply
        from its RPC thread); it exists only on a transport built with
        ``direct_replies=True``.
        """
        self._owner()
        if direct is not False and (direct is not True or self._direct is None):
            raise ValueError('direct replies are not enabled on this transport')
        if not isinstance(kind, str) or not kind:
            raise ValueError('reply kind required')
        ranks = tuple(self._queues) if ranks is None else tuple(ranks)
        if not ranks or len(set(ranks)) != len(ranks) or any(r not in self._queues for r in ranks):
            raise ValueError('invalid expected reply ranks')
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('positive finite reply timeout required')
        if not callable(send) or (aggregate is not None and not callable(aggregate)):
            raise ValueError('invalid callback')
        if precheck is not None and not callable(precheck):
            raise ValueError('invalid submission precheck')
        # Ordinary stock RPCs can submit from vLLM IO/health threads; this is
        # process ownership, not permission for a second MQ reader or writer.
        # Contention refuses before registration, with no retry or poisoning.
        if not self._broker.acquire(timeout=0.0 if precheck is not None else timeout_s):
            raise Backpressure('submission_serialization_busy')
        try:
            with self._condition:
                self._healthy()
                if self._sending_thread is not None:
                    raise TransportError('reentrant broadcast submission')
                if len(self._calls) >= self._max_calls:
                    raise Backpressure('unclaimed RPC credit budget')
            if precheck is not None:
                precheck()  # Same writer critical section, before registration.
            with self._condition:
                self._healthy()  # close may have begun during a precheck.
                sequence = self._next; self._next += 1
                now = self._clock()
                call = _Call(sequence, kind, ranks, now, now + timeout_s, aggregate)
                self._calls[sequence] = call
                self._sending_thread = threading.get_ident()
                fifo = self._direct if direct else self._todo
                for rank in ranks:
                    fifo[rank].append((sequence, call.deadline_s))
                self._condition.notify_all()
            try:
                send(sequence)
            except BaseException:
                # Broadcast may have partially succeeded. No rollback/reuse.
                with self._condition:
                    self._poison('broadcast_failed')
                raise
            finally:
                with self._condition:
                    self._sending_thread = None
                    self._condition.notify_all()
            return sequence
        finally:
            self._broker.release()

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

    def _receive_demux_loop(self, rank):
        """Two expectation FIFOs per rank, one dequeue per wire (held-f burst).

        The worker writes ordered replies in RPC order through its async output
        thread and DirectReply replies in RPC order from its RPC thread, so each
        class is FIFO on the wire. A DirectReply matches only when its carried
        sequence is exactly the head of this rank's direct FIFO; an ordered
        wire (stock replies carry no ID) matches the ordered head. A
        duplicate, stale, reordered or unexpected wire, or any non-SUCCESS
        status (before matching), poisons every pending call. The wait is
        bounded by the earlier head deadline; each match is checked against
        its own deadline.
        """
        ordered, direct = self._todo[rank], self._direct[rank]
        while True:
            with self._condition:
                while (not ordered and not direct and not self._closing
                       and self._failure is None):
                    self._condition.wait()
                if self._closing or self._failure is not None:
                    return
                deadline = min(fifo[0][1] for fifo in (ordered, direct) if fifo)
                marker = (rank, None)
                self._inflight.add(marker)
            try:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise TimeoutError
                wire = self._queues[rank].dequeue(timeout=remaining)
                received = self._clock()
                if not isinstance(wire, tuple) or len(wire) != 2:
                    raise ValueError
                status, value = wire
                if status != self._status:
                    with self._condition:
                        self._poison('worker_failure')
                    return
                with self._condition:
                    if isinstance(value, DirectReply):
                        tagged = value.sequence
                        if type(tagged) is not int or not direct or direct[0][0] != tagged:
                            self._poison('reply_correlation_failure')
                            return
                        fifo = direct
                    elif ordered:
                        fifo = ordered
                    else:
                        self._poison('reply_correlation_failure')
                        return
                    sequence, call_deadline = fifo.popleft()
                    if received > call_deadline:
                        raise TimeoutError
                    if fifo is direct:
                        value = value.value
                    if not self._closing and self._failure is None:
                        self._done.put_nowait((sequence, rank, value, received))
            except BaseException as exc:
                with self._condition:
                    self._poison('reply_timeout' if isinstance(exc, TimeoutError) else 'reply_transport_failure')
                return
            finally:
                with self._condition:
                    self._inflight.discard(marker)
                    self._condition.notify_all()

    def poll(self, budget=8):
        """Bounded local processing. Aggregators run here, never in receivers.

        A foreign writer/consumer makes this return zero, never wait for its
        broadcast. Short state locks/GIL remain; this is not a realtime promise.
        Aggregate callbacks themselves must not perform blocking engine work.
        """
        self._owner()
        if type(budget) is not int or budget < 1:
            raise ValueError('positive poll budget required')
        if not self._broker.acquire(blocking=False):
            with self._condition:
                self._healthy()
            return 0
        try:
            with self._condition:
                self._healthy()
            n = 0
            while n < budget:
                try:
                    sequence, rank, value, received = self._done.get_nowait()
                except Empty:
                    break
                with self._condition:
                    self._healthy()
                    call = self._calls.get(sequence)
                    if call is None or rank not in call.ranks or rank in call.replies:
                        self._poison('reply_correlation_failure')
                        raise TransportError('reply_correlation_failure')
                    call.replies[rank] = value; call.received_s[rank] = received
                    complete = len(call.replies) == len(call.ranks)
                    ordered = [call.replies[r] for r in call.ranks] if complete else None
                if complete:
                    try:
                        # Serialized with other consumers, outside the receiver
                        # condition: shutdown can still close/poison promptly.
                        result = call.aggregate(ordered) if call.aggregate else ordered
                    except BaseException:
                        with self._condition:
                            self._poison('aggregation_failed')
                        raise
                    with self._condition:
                        self._healthy()
                        call.value = result
                        call.ready = True
                n += 1
            with self._condition:
                self._healthy()
            return n
        finally:
            self._broker.release()

    def wait_landed(self, timeout_s):
        """Held-g L3: block up to ``timeout_s`` until a received reply waits for ``poll``.

        No MQ read, poll, claim or aggregation: receivers put each matched
        reply on the landed queue under the condition and notify after every
        receive, so a reply landing during the wait wakes it. Returns True
        when a landed reply is waiting; a failed or closing transport returns
        at once (its next poll raises as before).
        """
        self._owner()
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError('finite nonnegative wait required')
        with self._condition:
            if self._done.empty() and not self._closing and self._failure is None:
                self._condition.wait(timeout_s)
            return not self._done.empty()

    def call_ready(self, sequence):
        """Atomic local observation; no MQ read, implicit poll or claiming."""
        self._owner()
        with self._condition:
            self._healthy()
            call = self._calls.get(sequence)
            if call is None:
                raise TransportError('reply expectation disappeared')
            return call.ready

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
            if t is not threading.current_thread():
                t.join(max(0.0, deadline - time.monotonic()))
        with self._condition:
            live = [rank for rank, t in self._threads.items() if t.is_alive()]
            clean = not (live or self._calls or self._failure or self._sending_thread is not None)
            if clean:
                with _CLAIM_LOCK:
                    for q in self._queues.values():
                        if q._flash_s5_receiver_owner is self:
                            q._flash_s5_receiver_owner = None
            return dict(clean=clean,
                        live_receivers=live, outstanding_calls=len(self._calls),
                        inflight_receives=len(self._inflight), failure=self._failure,
                        active_broadcast=self._sending_thread is not None,
                        scope='transport only; no GPU completion or source-lease release implied')
