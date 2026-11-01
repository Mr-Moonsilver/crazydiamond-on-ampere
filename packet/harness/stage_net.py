"""Connected CPU network for the stage-local proofs (C0/C1).

Real objects: ``NativeSegmentDriver`` / ``StagedSegmentDriver``, the admission
controller and scheduler leases, ``LaneDispatcher``, ``NativeWorkerBridge``,
``WorkerState``, ``BoundedSender``, ``ReplyTransport`` (its receiver threads
included) and the generated worker-loop hooks in ``executor_hooks``.  Every
command and reply is pickled, as on the MQ, so each rank gets its own copy.

Each rank has its own read position on the ring and runs only when a test steps
it, so a stage can lag, stall or race the other one.  Leaves are declared in
``harness/stage_fixture_leaves.json``.  No GPU, CUDA, NCCL, MQ or model forward
is exercised; no pass here is native evidence.
"""
from contextlib import contextmanager
import pickle
import queue
import threading
import time
import types

import test_e3_r4_worker_bridge as r4

eh, ed, sh, w = r4.eh, r4.ed, r4.sh, r4.w
from reply_transport import ReplyTransport  # noqa: E402
from stage_progress import StagedSegmentDriver  # noqa: E402
from stage_wire import StageScoped  # noqa: E402

SUCCESS, FAILURE = 'SUCCESS', 'FAILURE'
RING_SLOTS = 10


class FakeBuffer:
    """Leaf for ``ShmRingBuffer``: per-slot written flag and per-reader read flags."""

    def __init__(self, n_reader, max_chunks):
        self.n_reader = n_reader
        self.max_chunks = max_chunks
        self.max_chunk_bytes = 1 << 22
        self.metadata = [[0] * (1 + n_reader) for _ in range(max_chunks)]
        self.data = [None] * max_chunks

    @contextmanager
    def get_metadata(self, index):
        yield self.metadata[index]


class FakeRing:
    """Leaf for the writer side of ``MessageQueue``, same slot rule as acquire_write."""

    def __init__(self, ranks, max_chunks=RING_SLOTS):
        self.ranks = tuple(ranks)
        self.buffer = FakeBuffer(len(self.ranks), max_chunks)
        self.current_idx = 0
        self._is_writer = True
        self.n_remote_reader = 0
        self.shutting_down = False
        self.readers = {rank: 0 for rank in self.ranks}
        self.would_block = 0
        self.history = []
        self.max_lag = 0
        self.sent = 0

    def writable(self):
        slots = self.buffer.metadata[self.current_idx]
        return not (slots[0] and sum(slots[1:]) != self.buffer.n_reader)

    def enqueue(self, envelope, timeout=None):
        if not self.writable():
            # The stock write would block the owner thread here.
            self.would_block += 1
            raise AssertionError('broadcast write on a full ring would block the owner')
        index = self.current_idx
        self.history.append(envelope)
        self.buffer.data[index] = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
        self.buffer.metadata[index] = [1] + [0] * self.buffer.n_reader
        self.current_idx = (index + 1) % self.buffer.max_chunks
        self.sent += 1
        self.max_lag = max(self.max_lag, max(self.lag(rank) for rank in self.ranks))

    def lag(self, rank):
        total = 0
        index = self.readers[rank]
        slot = self.ranks.index(rank)
        for _ in range(self.buffer.max_chunks):
            meta = self.buffer.metadata[index]
            if not meta[0] or meta[1 + slot]:
                break
            total += 1
            index = (index + 1) % self.buffer.max_chunks
        return total

    def read(self, rank):
        index = self.readers[rank]
        slot = self.ranks.index(rank)
        meta = self.buffer.metadata[index]
        if not meta[0] or meta[1 + slot]:
            return None
        envelope = pickle.loads(self.buffer.data[index])
        meta[1 + slot] = 1
        self.readers[rank] = (index + 1) % self.buffer.max_chunks
        return envelope


class ReplyQueue:
    """Leaf for one worker's response ``MessageQueue``; pickles like the wire."""

    def __init__(self):
        self.items = queue.Queue()
        self.written = 0

    def enqueue(self, wire):
        self.written += 1
        self.items.put(pickle.dumps(wire, protocol=pickle.HIGHEST_PROTOCOL))

    def dequeue(self, timeout=None):
        try:
            return pickle.loads(self.items.get(timeout=timeout))
        except queue.Empty:
            raise TimeoutError from None


class CountingDone(queue.Queue):
    """The transport's own done queue, counted so a test can wait for its receivers."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def put_nowait(self, item):
        self.count += 1
        super().put_nowait(item)


class FakeProc:
    """Leaf for ``WorkerProc``: the stock reply path without the async output thread."""

    def __init__(self, rank, mq):
        self.rank = rank
        self.worker_response_mq = mq

    def handle_output(self, output):
        self.enqueue_output(output)

    def enqueue_output(self, output):
        if isinstance(output, Exception):
            result = (FAILURE, type(output).__name__ + ': ' + str(output)[:200])
        else:
            result = (SUCCESS, output)
        eh.worker_enqueue_reply(self, result)


class RingNetworkMixin:
    """The per-rank ring, transport and worker loop; mixed into an r4 stack."""

    def install_network(self):
        self.ring = FakeRing((0, 1, 2, 3))
        self.sender = eh.BoundedSender(self.ring)
        self.reply_queues = {rank: ReplyQueue() for rank in (0, 1, 2, 3)}
        self.transport = ReplyTransport(self.reply_queues, SUCCESS, handoff_idle=True,
                                        max_calls=RING_SLOTS, direct_replies=self.direct)
        self.transport._done = CountingDone()
        self.executor = r4.FakeExecutor(self.sender, self.transport)
        self.procs = {rank: FakeProc(rank, self.reply_queues[rank]) for rank in (0, 1, 2, 3)}
        self.worker_errors = []
        self.skips = {rank: 0 for rank in (0, 1, 2, 3)}
        self.frozen = set()

    @property
    def sent_envelopes(self):
        return self.ring.history

    def close_network(self):
        self.transport.close(timeout_s=0.5)

    def _trace_commands(self, rank, method, args):
        message = args[0] if args else None
        if type(message) is StageScoped and rank not in message.ranks:
            self.skips[rank] += 1

    def step(self, rank):
        """One iteration of the generated ``worker_busy_loop`` on this rank."""
        if rank in self.frozen:
            return False
        envelope = self.ring.read(rank)
        if envelope is None:
            return False
        method, args, kwargs, output_rank = envelope
        self._trace_commands(rank, method, args)
        proc = self.procs[rank]
        rpc = None
        try:
            rpc = eh.worker_rpc_begin(proc, method, kwargs)
            if method == 'stock_rpc':
                output = 'stock'
            else:
                output = getattr(self.workers[rank], method)(*args, **kwargs)
            eh.worker_rpc_returned(proc, rpc, output_rank)
            if output_rank is None or rank == output_rank:
                eh.worker_handle_output(proc, rpc, output)
        except Exception as exc:
            self.worker_errors.append((rank, method, type(exc).__name__, str(exc)))
            if output_rank is None or rank == output_rank:
                eh.worker_handle_output(proc, rpc, exc)
        return True

    def pump(self, ranks=(0, 1, 2, 3)):
        worked = True
        while worked:
            worked = False
            for rank in ranks:
                worked = self.step(rank) or worked
        self.settle()

    def settle(self, timeout_s=5.0):
        """Wait for the real receiver threads to take every written reply."""
        transport = self.transport
        done = transport._done
        deadline = time.monotonic() + timeout_s
        while True:
            written = sum(mq.written for mq in self.reply_queues.values())
            unread = sum(mq.items.qsize() for mq in self.reply_queues.values())
            # A wire no expectation claims stays unread (a duplicate, for one).
            with transport._condition:
                reading = {marker[0] for marker in transport._inflight}
                claimed = any(mq.items.qsize() and (
                    rank in reading or transport._todo[rank]
                    or (transport._direct is not None and transport._direct[rank]))
                    for rank, mq in self.reply_queues.items())
            if not claimed and done.count >= written - unread:
                return
            if transport._failure is not None:
                return
            if time.monotonic() > deadline:
                raise AssertionError('reply receivers did not settle')
            time.sleep(0)

    def stock_send(self):
        """A stock decode RPC on the same ring: it must never block on native work.

        ``output_rank`` -1 is no rank: the stock future path is not under test.
        """
        self.ring.enqueue(('stock_rpc', (), {}, -1))

    # -- driving ----------------------------------------------------------
    def tick(self, complete=True):
        if complete:
            self.complete_everything()
        self.pump()
        return self.driver.advance()

    def admit_next(self):
        if self.controller.parked is None:
            return self.controller.admit(self.scheduler)
        return None

    def run_request(self, limit=4000, tick=None):
        """Drive every chunk of the request to its stock handback."""
        tick = tick or self.tick
        for _ in range(limit):
            if (self.controller.commits >= self.expected_commits and not self.driver.live):
                return
            self.admit_next()
            tick()
        raise AssertionError(
            f'request did not finish: phase={self.driver.phase} commits='
            f'{self.controller.commits} status={self.status_text()}')

    def status_text(self):
        status = getattr(self.driver, 'stage_status', None)
        return status() if callable(status) else None


class RingStack(RingNetworkMixin, r4.Stack):
    """r4's connected workers on the per-rank network; ``staged`` picks the driver."""

    def __init__(self, *, staged=True, tokens=13, chunk=4, direct=False, stop_budget=256):
        self.direct = direct
        self.staged = staged
        self.tokens = tokens
        self.chunk = chunk
        self.expected_commits = (tokens - 1 + chunk - 1) // chunk
        self.scheduler = r4.Scheduler(r4.Manager(), [r4.Request(uid=r4.UID, tokens=tokens)])
        self.controller = r4.make_controller(self.scheduler, chunk=chunk, ranks=(0, 1, 2, 3))
        self.record = self.controller.admit(self.scheduler)
        self.install_network()
        driver_class = StagedSegmentDriver if staged else ed.NativeSegmentDriver
        self.driver = driver_class(
            self.controller, executor=self.executor, scheduler=self.scheduler,
            engine=types.SimpleNamespace(), layer_count=4, layer_split=2,
            stage_of=((0, 0), (1, 0), (2, 1), (3, 1)),
            reply_timeout_s=30.0, stop_budget=stop_budget).install()
        self.workers = {}
        for rank, stage in ((0, 0), (1, 0), (2, 1), (3, 1)):
            self._add_worker(rank, stage)
        self.transport.network = None
        for rank in (2, 3):
            self.worker(rank).channel.ready = True
        self.ledgers = {rank: [] for rank in (0, 1, 2, 3)}
        for rank in (0, 1, 2, 3):
            self._wrap_device_leaves(rank)

    def _wrap_device_leaves(self, rank):
        """Per-rank ledger of device-affecting work, in the order the rank ran it."""
        worker = self.workers[rank]
        ledger = self.ledgers[rank]
        dispatcher = worker.dispatcher
        for kind in ('prepare', 'prefix', 'layer', 'suffix', 'pp_send', 'pp_receive',
                     'state_finalize'):
            original = getattr(dispatcher, '_do_' + kind)

            def wrapped(command, original=original, kind=kind):
                result = original(command)
                ledger.append(('op', kind, command.transaction.txn_seq, command.stage,
                               command.op_seq, command.layer_start, command.seq))
                return result
            setattr(dispatcher, '_do_' + kind, wrapped)
        channel = worker.channel
        if channel.is_sender:
            send = channel.send

            def traced_send(payload, transaction, stage, seq, send=send):
                ledger.append(('pp-send', 'prefill_stream', transaction.txn_seq, stage, seq))
                return send(payload, transaction, stage, seq)
            channel.send = traced_send
        else:
            post = channel.post_receive

            def traced_post(transaction, stage, seq, num_tokens, post=post):
                ledger.append(('pp-recv', 'prefill_stream', transaction.txn_seq, stage, seq,
                               num_tokens))
                return post(transaction, stage, seq, num_tokens)
            channel.post_receive = traced_post
        factory = worker.factory
        admit = factory.admit

        def traced_admit(admission, admit=admit):
            ledger.append(('admit', admission.transaction.txn_seq, admission.num_tokens))
            return admit(admission)
        factory.admit = traced_admit

    def persistent_state(self):
        """Every persistent fact the two paths must agree on after a request."""
        manager = self.scheduler.kv_cache_manager
        request = self.controller
        out = dict(cache_calls=list(manager.cache_calls),
                   rows={key: tuple(value) for key, value in manager.rows.items()},
                   alloc=[(call[0].request_id, call[1]) for call in manager.alloc_calls],
                   commits=request.commits, cancels=request.cancels,
                   frontier=sorted((tx.txn_seq, row['committed_end'], row['remaining'],
                                    row['state_epoch'], row['ranks'], row['pp_seqs'])
                                   for tx, row in request._committed.items()),
                   retained=None if self.driver._retained is None
                   else self.driver._retained['transaction'].txn_seq,
                   transfer_seq=self.driver.transfer_seq,
                   last_lane_sequences={rank: value[1] for rank, value
                                        in self.driver._last_lane_sequences.items()},
                   leases=request.leases.shutdown())
        for rank, worker in self.workers.items():
            out[rank] = dict(registers=worker.factory.registers, updates=worker.factory.updates,
                             released=worker.factory.released,
                             admissions=[(a.transaction.txn_seq, a.num_tokens)
                                         for a in worker.factory.admissions],
                             channel_sends=worker.channel.sends,
                             release_send=worker.channel.release_send_calls,
                             tickets=[t.state for t in worker.channel.tickets],
                             finished=[c.finished for c in worker.factory.chunks],
                             bridge_transaction=worker._flash_s5_bridge.transaction,
                             dispatcher_transaction=worker.dispatcher.transaction,
                             runtime_releases=[tx.txn_seq for tx
                                               in worker.server.runtime.releases])
        return out


def stage_of_command(envelope):
    """(method, message, targets) of one ring envelope, whichever path sent it."""
    method, args, _kwargs, _rank = envelope
    message = args[0] if args else None
    if type(message) is StageScoped:
        return method, message.message, message.ranks
    return method, message, None
