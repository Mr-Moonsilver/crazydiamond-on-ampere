"""CPU-only acceptance for the rung-4 worker bridge, typed RPCs and cancellation.

Four independent fake workers (ranks 0/1 stage 0, ranks 2/3 stage 1) each own a
real ``LaneDispatcher`` and a real ``NativeWorkerBridge``.  The generated
``flash_s5_*`` worker methods are invoked through ``executor_hooks.worker_rpc``
by a routing fake of the owned reply transport, and the real
``NativeSegmentDriver`` drives them through the real admission controller.

Every device event is a controllable fake.  This is a protocol/ownership test:
no GPU, CUDA, NCCL, MQ, model forward or real PP traffic is exercised, and no
pass here is native evidence.
"""
import contextlib
from dataclasses import replace
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import engine_driver as ed  # noqa: E402
import executor_hooks as eh  # noqa: E402
import lane_chunk  # noqa: E402
import s5_startup  # noqa: E402
import scheduler_hooks as sh  # noqa: E402
import wire_v1 as w  # noqa: E402
import worker_hooks  # noqa: E402
from e3_types import (  # noqa: E402
    CompletionKind,
    CompletionProof,
    DependencyRole,
    E3ContractError,
    E3Pending,
    LaneId,
    LaneRole,
    OperationTicket,
)
from graph_router import (  # noqa: E402
    CompilationIdentity,
    ExecutionContext,
    ExecutionDescriptor,
    GraphRouter,
)
from lane_chunk import ChunkAdmission  # noqa: E402
from lane_comm import PPConsumerReceipt  # noqa: E402
from lane_dispatch import LaneDispatcher  # noqa: E402
from lane_state import WorkerState  # noqa: E402
from split_bank import RouteController, SplitDecodeBank  # noqa: E402
from worker_bridge import NativeWorkerBridge, WorkerBridgeError  # noqa: E402

# Reuse the reviewed admission scheduler fixtures and the dispatcher's fakes.
from test_e3_admission_native import Manager, Request, Scheduler, make_controller  # noqa: E402
from test_e3_lane_dispatch import FakeDriver, FakeLane, FakeOperation, FakeStream  # noqa: E402

UID = 'req-r4-bridge'


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class ControllableEvent:
    """A fake device event whose host query is controlled by the test."""

    def __init__(self, bus):
        self.bus = bus
        self.stream = None
        self.complete = False
        bus.events.append(self)

    def record(self, stream=None):
        self.stream = stream

    def query(self):
        return self.complete and self.bus.permitted(self)

    def synchronize(self):
        # A real host wait blocks until the tail kernel completes.  The fake
        # records the wait order and completes immediately: the CPU evidence is
        # that every consumer stream got its own tail event and was waited on
        # before the route switched.
        self.complete = True
        self.bus.syncs.append(self)


class EventBus:
    def __init__(self):
        self.events = []
        self.syncs = []
        self.pp_channel = None

    def permitted(self, event):
        channel = self.pp_channel
        if channel is None:
            return True
        for ticket in channel.tickets:
            if (ticket.state != 'retired'
                    and not channel.matched(ticket)
                    and self.events.index(event) >= self.events.index(ticket.event)):
                return False
        return True

    def event(self):
        return ControllableEvent(self)

    def complete_all(self, complete=True):
        for event in self.events:
            event.complete = complete

    def sync_streams(self):
        return tuple(event.stream for event in self.syncs)


def bus_torch(bus):
    torch = types.SimpleNamespace()
    torch.cuda = types.SimpleNamespace(
        Event=lambda enable_timing=False: bus.event(),
        stream=lambda stream: contextlib.nullcontext(stream))
    torch.inference_mode = lambda: contextlib.nullcontext()
    return torch


class FakeChunk:
    def __init__(self, admission, req_index):
        self.admission = admission
        self.req_index = req_index
        self.finished = 0

    def lane_inputs(self):
        return {'positions': 'positions', 'input_ids': 'ids',
                'query_start_loc': 'qsl', 'ngram_context': 'ngram'}

    def forward_scope(self):
        return contextlib.nullcontext()

    def finish_state(self):
        self.finished += 1
        return True


class FakeFactory:
    """Per-worker chunk owner; models register-vs-update and the request row.

    ``register`` creates the row on the first chunk; every later chunk of the
    same request must reach ``update``; ``release`` is the only thing that
    closes the row, exactly like ``LaneChunkFactory``.
    """

    def __init__(self, rank):
        self.rank = rank
        self.admissions = []
        self.released = 0
        self.chunks = []
        self.registered = None
        self.registers = 0
        self.updates = 0

    def admit(self, admission):
        if not isinstance(admission, ChunkAdmission):
            raise AssertionError('bridge passed a non-local admission')
        self.admissions.append(admission)
        if self.registered is None:
            self.registered = admission.request_uid
            self.registers += 1
        else:
            if self.registered != admission.request_uid:
                raise AssertionError('a second request reused the lane row')
            self.updates += 1
        chunk = FakeChunk(admission, self.rank)
        self.chunks.append(chunk)
        return chunk

    def release(self):
        # ``LaneChunkFactory.release`` is a no-op when prepare never registered
        # a row (open only leases the server-side operation).
        if self.registered is None:
            return None
        self.registered = None
        self.released += 1
        return f'req-{self.rank}'


class FakePPChannel:
    """Bounded PP channel fake with a queried receipt/binding surface.

    The receiver's consumer event is the dispatcher's state-finalize segment
    event (its conservative boundary), so a delayed remote consumer is a
    delayed receiver segment query, not a separate fake proof.
    """

    def __init__(self, bus, *, is_sender, receiver_lane=None):
        self.bus = bus
        self.is_sender = is_sender
        self.receiver_lane = receiver_lane
        self.consumer_binding = None
        self.ready = False
        self.tickets = []
        self.payload = 'hc-received'
        self.sends = 0
        self.release_send_calls = 0
        self.peer = None
        self.bus.pp_channel = self

    def matched(self, ticket):
        return self.peer is not None and any(
            other.kind == ('recv' if self.is_sender else 'send')
            and other.transaction == ticket.transaction
            and other.stage == ticket.stage and other.seq == ticket.seq
            for other in self.peer.tickets)

    def bind_consumer(self, lane, route_generation):
        if any(ticket.state != 'retired' for ticket in self.tickets):
            raise AssertionError('live PP channel rebound')
        self.consumer_binding = (lane, route_generation)

    def _ticket(self, kind, transaction, stage, seq):
        ticket = types.SimpleNamespace(
            kind=kind, transaction=transaction, stage=stage, seq=seq,
            state='posted', consumer_event=None, event=self.bus.event())
        self.tickets.append(ticket)
        return ticket

    def post_receive(self, transaction, stage, seq, num_tokens):
        if self.is_sender:
            raise AssertionError('sender cannot receive')
        return self._ticket('recv', transaction, stage, seq)

    def poll_receive(self, ticket):
        if ticket.state == 'ready':
            return self.payload
        if not self.ready or not self.matched(ticket):
            return None
        ticket.event.complete = True
        ticket.state = 'ready'
        return self.payload

    def send(self, payload, transaction, stage, seq):
        if not self.is_sender:
            raise AssertionError('receiver cannot send')
        self.sends += 1
        return self._ticket('send', transaction, stage, seq)

    def consumer_done(self, ticket, consumer_event):
        if ticket.state != 'ready':
            raise AssertionError('consumer done before the receive was ready')
        ticket.consumer_event = consumer_event
        return ticket

    def validate_consumer_proof(self, ticket, receipt):
        if type(receipt) is not PPConsumerReceipt:
            raise AssertionError('proof is not a typed PPConsumerReceipt')
        if self.consumer_binding is None:
            raise AssertionError('receipt before peer binding')
        lane, route = self.consumer_binding
        if (receipt.transaction != ticket.transaction
                or receipt.stage != ticket.stage or receipt.seq != ticket.seq
                or receipt.route_generation != route
                or receipt.proof.ticket.lane != lane):
            raise AssertionError('receipt differs from transfer/lane/route')
        return True

    def release_send(self, ticket, proof):
        if proof is None:
            raise AssertionError('send release without the consumer proof')
        if not ticket.event.query():
            raise AssertionError('send release before local completion')
        if proof.proof.ticket.lane != self.consumer_binding[0]:
            raise AssertionError('send release for another consumer lane')
        self.release_send_calls += 1
        ticket.state = 'retired'
        return True

    def release_receive(self, ticket):
        if ticket.consumer_event is None or not ticket.consumer_event.query():
            raise AssertionError('receive release before the consumer completed')
        ticket.state = 'retired'
        return True


class FakeRuntime:
    def __init__(self, state):
        self.state = state
        self.releases = []

    def release(self, transaction):
        self.releases.append(transaction)
        lane = self.state.prefill
        if lane.state != 'free':
            lane.release()
        return True


class FakeSourceBinding:
    """Stand-in for the worker-local pinned source binding.

    The real binder's tensor/storage checks are exercised by
    ``test_e3_native_source_oracle``; here only the bridge's strict lease
    validation and its close-after-release ordering are under test.
    """

    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.transaction = descriptor.transaction
        self.state_epoch = descriptor.state_epoch
        self.holds = {}
        self.closed = False

    def close(self, registry):
        if self.closed:
            raise AssertionError('source binding closed twice')
        self.closed = True


def fake_source_binder(dispatcher, admission, descriptor, *, rank=None):
    descriptor.validate_frame(admission, (rank,))
    return FakeSourceBinding(descriptor)


class BridgeServer:
    """Worker handoff-server fake with the real open/clear fallback shape."""

    def __init__(self, state):
        self.runtime = FakeRuntime(state)
        self.operation = None
        self.handoff = None
        self.calls = []

    def attach_worker(self, worker):
        self.worker = worker
        return self

    def open(self, message):
        # The legacy server's OpenAck has no lane; the bridge must use the
        # runtime handle actually allocated by the worker's prefill state, so
        # this fake reserves the real LaneStorage row, exactly like the native
        # handoff server does.
        state = self.runtime.state
        state.declare_request(message.transaction.request_uid,
                              message.transaction.request_generation)
        state.prefill.reserve(message.transaction, metadata=message.metadata)
        state.prefill.ready()
        self.handoff = message.transaction
        self.operation = FakeOperation(drained=True)
        self.calls.append(('open', message.transaction, message.metadata))
        return types.SimpleNamespace(transaction=message.transaction, accepted=True)

    def begin_prefill_operation(self, runner, marker):
        self.calls.append(('begin', marker.flash_e3_transaction))

    def end_prefill_operation(self, runner):
        self.calls.append(('end',))

    def abort_prefill_operation(self, runner):
        self.calls.append(('abort',))

    def clear_native(self, transaction):
        if self.handoff != transaction:
            raise AssertionError('clear_native for another transaction')
        self.handoff = None
        self.operation = None
        self.calls.append(('clear', transaction))
        return True


class _Mode:
    """A stand-in for the pinned cudagraph mode enum (name-keyed)."""

    def __init__(self, name):
        self.name = name


class _VllmDescriptor:
    """The minimal pinned BatchExecutionDescriptor mirror B converts."""

    def __init__(self, tokens=1):
        self.cg_mode = _Mode('FULL')
        self.num_tokens = tokens
        self.num_reqs = 1
        self.uniform_token_count = tokens
        self.max_query_len = tokens
        self.num_active_loras = 0
        self.num_ubatches = 1


class _StockCudagraphManager:
    """Fake stock manager: one FULL descriptor, its own full-GPU pool."""

    pool = (2, 0)
    _capture_descs = {_Mode('FULL'): [_VllmDescriptor()]}


class _RouteBackend:
    """B's capture backend protocol over fake graphs (no device work)."""

    def __init__(self):
        self.captured = []

    def capture(self, name, key, lane, context):
        self.captured.append(key)
        return object()

    def replay(self, handle, key, lane, context):
        return object()

    def destroy_graph(self, handle):
        return True

    def destroy_event(self, token):
        return True

    def destroy_stream(self, token):
        return True


FULL_POOL_ID = 2 * 1000003
SPLIT_POOL_ID = 1 * 1000003


def route_objects(rank, state, bus, torch):
    """Real RouteController + real GraphRouter over fake streams/events.

    The split bank is a real ``SplitDecodeBank`` whose stock manager and capture
    are fakes (this suite never runs CUDA); the graph bank is declared and
    captured through the real router so a transition validates real bank
    readiness, context identity and pool/epoch fencing.
    """
    split = types.SimpleNamespace(
        decode_stream=FakeStream(1000 + rank),
        prefill_stream=FakeStream(2000 + rank),
        decode_stream_token=1000 + rank,
        decode_context_id=state.decode_owner.context_id,
        decode_sms=16,
        requested_decode_sms=16,
    )
    runner = types.SimpleNamespace(
        main_stream=FakeStream(3000 + rank),
        output_copy_stream=FakeStream(4000 + rank),
        pp_handler=types.SimpleNamespace(main_stream=FakeStream(5000 + rank),
                                         broadcast_stream=FakeStream(6000 + rank)),
        cudagraph_manager=_StockCudagraphManager(),
        lora_config=None,
        device=f'cuda:{rank}',
    )
    bank = SplitDecodeBank(runner, split, torch_module=torch)
    bank.captured = True
    bank.pool = (1, 0)
    controller = RouteController(runner, bank, torch_module=torch)
    backend = _RouteBackend()
    router = GraphRouter(
        state, state.decode.handle,
        CompilationIdentity('test-digest', 'full', (1,), 1),
        full_context_id=0, full_pool_id=FULL_POOL_ID, backend=backend)
    context = ExecutionContext(
        'decode-split', state.decode_owner.context_id, requested_sm=16,
        measured_sm=16, pool_id=SPLIT_POOL_ID, stream_token=1000 + rank,
        reserved_bytes=4096, basis='estimate')
    router.add_bank('decode-split', context,
                    descriptors=(ExecutionDescriptor('full', 1, 1, 1, 1),))
    router.capture_at_startup()
    return split, controller, router, backend


class FakeWorker:
    """The generated ``flash_s5_*`` methods over ``worker_rpc``."""

    def __init__(self, rank, stage, bridge, dispatcher):
        self.rank = rank
        self.stage = stage
        self._flash_s5_bridge = bridge
        self.dispatcher = dispatcher

    def flash_s5_open(self, message):
        return eh.worker_rpc(self, 'open', message)

    def flash_s5_submit(self, message):
        return eh.worker_rpc(self, 'submit', message)

    def flash_s5_poll(self, message):
        return eh.worker_rpc(self, 'poll', message)

    def flash_s5_commit(self, message):
        return eh.worker_rpc(self, 'commit', message)

    def flash_s5_release(self, message):
        return eh.worker_rpc(self, 'release', message)

    def flash_s5_bind(self, message):
        return eh.worker_rpc(self, 'bind', message)

    def flash_s5_receipt(self, message):
        return eh.worker_rpc(self, 'receipt', message)

    def flash_s5_cancel(self, message):
        return eh.worker_rpc(self, 'cancel', message)

    def flash_s5_transition(self, message):
        return eh.worker_rpc(self, 'transition', message)

    def flash_s5_bootstrap(self, message):
        return eh.worker_rpc(self, 'bootstrap', message)


class RoutingSender:
    def __init__(self):
        self.envelopes = []
        self.checks = []

    def check(self, envelope):
        self.checks.append(envelope)

    def __call__(self, envelope):
        self.envelopes.append(envelope)


class DirectTransport:
    """Synchronous, rank-routed stand-in for the owned reply transport.

    It keeps the driver's one-submit/one-drain contract: ``submit`` registers
    one expectation and immediately makes the routed worker call; the driver
    still consumes the reply through ``_calls``/``take`` on a later tick.
    """

    def __init__(self, sender):
        self.sender = sender
        self.network = None
        self._calls = {}
        self.ready = {}
        self.next = 0
        self.missing = set()
        self.silent_worker_errors = []
        # While held, replies are computed but not yet visible to the driver:
        # the reply is in flight on the owned transport.
        self.hold = False

    def submit(self, kind, send, ranks=None, timeout_s=300.0, aggregate=None, precheck=None):
        if precheck is not None:
            precheck()
        sequence = self.next
        self.next += 1
        self._calls[sequence] = types.SimpleNamespace(ready=False)
        send(sequence)
        envelope = self.sender.envelopes[-1]
        method, args, kwargs, output_rank = envelope
        everyone = tuple(sorted(self.network.workers))
        if output_rank is None:
            # Stock worker_busy_loop: every worker replies to an untargeted
            # broadcast, so expecting a subset would desynchronize the FIFO
            # reply queues of the ranks left out.
            if tuple(ranks) != everyone:
                raise AssertionError('untargeted broadcast must expect every rank')
            targets = everyone
        else:
            if tuple(ranks) != (output_rank,):
                raise AssertionError('targeted reply registered the wrong ranks')
            targets = (output_rank,)
        values = []
        failure = None
        # Every worker executes every broadcast command (stock MQ semantics);
        # only the reply ranks enqueue a value or a failure.
        for rank in everyone:
            if rank in self.missing:
                continue
            # A broadcast is one command fanned out to every rank: one worker's
            # local failure must not hide the command from its peers, exactly
            # as an MQ broadcast would not.
            try:
                value = getattr(self.network.workers[rank], method)(*args, **kwargs)
            except BaseException as exc:
                if rank not in targets:
                    self.silent_worker_errors.append((method, rank, type(exc).__name__))
                if rank in targets and failure is None:
                    failure = exc
                continue
            if rank in targets:
                values.append(value)
        if failure is not None:
            raise failure
        self.ready[sequence] = values
        self._calls[sequence].ready = not self.hold
        return sequence

    def deliver_held(self):
        self.hold = False
        for call in self._calls.values():
            call.ready = True

    def poll(self, budget=8):
        return 0

    def take(self, sequence):
        del self._calls[sequence]
        return self.ready.pop(sequence), {}

    def withhold(self, rank):
        self.missing.add(rank)


class FakeExecutor:
    def __init__(self, sender, transport):
        self.sender = sender
        self._flash_s5_bounded_send = sender
        self._flash_s5_transport = transport

    def last_method(self, index=-1):
        return self.sender.envelopes[index][0]


class Stack:
    """One controller, one real driver and four real bridge/dispatcher workers."""

    def __init__(self, *, chunks=4, stop_budget=64, stop_tick_budget=8):
        self.scheduler = Scheduler(Manager(), [Request(uid=UID, tokens=12)])
        self.controller = make_controller(self.scheduler, chunk=chunks,
                                          ranks=(0, 1, 2, 3))
        self.record = self.controller.admit(self.scheduler)
        self.sender = RoutingSender()
        self.transport = DirectTransport(self.sender)
        self.executor = FakeExecutor(self.sender, self.transport)
        self.driver = ed.NativeSegmentDriver(
            self.controller, executor=self.executor, scheduler=self.scheduler,
            engine=types.SimpleNamespace(), layer_count=4, layer_split=2,
            stage_of=((0, 0), (1, 0), (2, 1), (3, 1)),
            reply_timeout_s=5.0, stop_budget=stop_budget,
            stop_tick_budget=stop_tick_budget).install()
        self.workers = {}
        for rank, stage in ((0, 0), (1, 0), (2, 1), (3, 1)):
            self._add_worker(rank, stage)
        self.transport.network = types.SimpleNamespace(workers=self.workers)
        # A receiver's posted payload is available unless a test delays it.
        for rank in (2, 3):
            self.worker(rank).channel.ready = True

    def _add_worker(self, rank, stage):
        bus = EventBus()
        torch = bus_torch(bus)
        state = WorkerState(rank, engine_epoch=0, engine_bound=False)
        state.retain_decode_storage(context_id=100 + rank)
        split, controller, router, route_backend = route_objects(
            rank, state, bus, torch)
        sender = stage == 0
        if sender:
            receiver = self.workers.get(2 if rank == 0 else 3)
            receiver_lane = None
            if receiver is not None:
                receiver_lane = getattr(receiver.server.runtime.state.prefill,
                                        'handle', None)
            channel = FakePPChannel(bus, is_sender=True)
        else:
            channel = FakePPChannel(bus, is_sender=False)
            if rank - 2 in self.workers:
                peer = self.workers[rank - 2].channel
                channel.peer, peer.peer = peer, channel
        lane = FakeLane(layers=(0, 1) if stage == 0 else (2, 3))
        lane.is_sender = sender
        native = types.SimpleNamespace(
            split=split,
            comms=types.SimpleNamespace(pp=channel),
            copy_stream=FakeStream(100 + rank),
            controller=controller,
            router=router,
            lane_forward=lambda inputs, forward_context=None, attention_driver=None: lane)
        server = BridgeServer(state)
        factory = FakeFactory(rank)
        dispatcher = LaneDispatcher(
            f'runner-{rank}', native, factory, server=server,
            driver_factory=lambda operation, module, context: FakeDriver(),
            torch_module=torch, clock=lambda: 1000 + rank)
        bridge = NativeWorkerBridge(dispatcher, server=server,
                                    source_binder=fake_source_binder)
        worker = FakeWorker(rank, stage, bridge, dispatcher)
        worker.bus = bus
        worker.torch = torch
        worker.state = state
        worker.split = split
        worker.controller = controller
        worker.router = router
        worker.route_backend = route_backend
        worker.lane = lane
        worker.channel = channel
        worker.server = server
        worker.factory = factory
        bridge.attach(worker)
        self.workers[rank] = worker

    def worker(self, rank):
        return self.workers[rank]

    # -- driving -----------------------------------------------------------
    def advance(self, count=1):
        for _ in range(count):
            self.driver.advance()
        return self.driver.phase

    def drive(self, until, limit=200):
        for _ in range(limit):
            if until():
                return
            self.driver.advance()
        raise AssertionError(f'driver did not reach the expected state; '
                             f'phase={self.driver.phase} index={self.driver.index}')

    def drive_to_finalize(self):
        self.drive(lambda: self.driver.phase == 'finalize')

    def complete_everything(self):
        for worker in self.workers.values():
            worker.bus.complete_all(True)

    def all_dispatchers_reset(self):
        return all(worker.dispatcher.transaction is None
                   for worker in self.workers.values())

    def cache_calls(self):
        return list(self.scheduler.kv_cache_manager.cache_calls)


def segment_record(worker, kind, op_seq):
    for record in worker.dispatcher.events:
        if record.kind == kind and record.op_seq == op_seq:
            return record
    raise AssertionError(f'no {kind} event for op_seq {op_seq}')


def accepted_ops(worker, kind):
    return tuple(op for op, (command, _ack) in worker.dispatcher.accepted.items()
                 if command.kind == kind)


class ConversionTests(unittest.TestCase):
    def record(self):
        return w.AdmissionRecord(
            transaction=self.txn, prompt_end=12, state_epoch=9, committed_end=0,
            block_tables=((5, 6, 7), (9, 10), (), (0, 0)),
            slot_mappings=((0, 1, 2), (3, 4), (), ()),
            new_block_ids_to_zero=(8, 11), cow_pairs=((1, 8),),
            input_ids=tuple(range(8)), ngram_context=(70, 71, 72))

    def setUp(self):
        from e3_types import TransactionId
        self.txn = TransactionId(1, UID, 0, 0, 8, 1)

    def test_admission_fields_groups_and_2d_ple_are_preserved(self):
        record = self.record()
        from worker_bridge import admission_to_chunk
        converted = admission_to_chunk(record)
        self.assertIs(type(converted), ChunkAdmission)
        self.assertEqual(converted.transaction, self.txn)
        self.assertEqual(converted.prompt_end, 12)
        self.assertEqual(converted.chunk_start, 0)
        self.assertEqual(converted.input_ids, tuple(range(8)))
        self.assertEqual(converted.ngram_context, (70, 71, 72))
        self.assertEqual(converted.block_tables,
                         ((5, 6, 7), (9, 10), (), (0, 0)),
                         'empty and repeated-null group rows travel unchanged')
        self.assertEqual(converted.new_block_ids_to_zero, (8, 11))
        self.assertEqual(converted.cow_pairs, ((1, 8),))
        self.assertEqual(converted.state_epoch, 9)

    def test_first_chunk_zeroes_and_copies_then_later_chunk_does_not_repeat(self):
        # Real LaneChunkFactory over the reviewed fake runner/ports.
        from test_e3_lane_chunk import fake_ports, fake_runner, fake_torch
        calls = []
        factory = lane_chunk.LaneChunkFactory(
            fake_runner(fake_torch(), calls), ports=fake_ports(calls),
            torch_module=fake_torch())
        from worker_bridge import admission_to_chunk
        # The reviewed fake runner declares three KV groups; the bridge keeps
        # every row it is given, so this record mirrors that worker's QSA.
        first_record = w.AdmissionRecord(
            transaction=self.txn, prompt_end=12, state_epoch=9, committed_end=0,
            block_tables=((5, 6, 7), (9, 10), ()),
            slot_mappings=((0, 1, 2), (3, 4), ()),
            new_block_ids_to_zero=(8, 11), cow_pairs=((1, 8),),
            input_ids=tuple(range(8)), ngram_context=(70, 71, 72))
        first = admission_to_chunk(first_record)
        chunk = factory.admit(first)
        self.assertIn(('zero', (8, 11)), calls)
        self.assertIn(('cow', 'kv-caches', 1), calls)
        self.assertIn(('cow', ['view0'], 1), calls)
        self.assertEqual(chunk.ngram_context.shape, (1, 3),
                         'the pinned ngram path consumes [num_reqs, context_len]')
        self.assertEqual(chunk.ngram_context.values, [70, 71, 72])
        calls.clear()
        from e3_types import TransactionId
        second_txn = TransactionId(1, UID, 0, 8, 11, 2)
        second = w.AdmissionRecord(
            transaction=second_txn, prompt_end=12, state_epoch=9, committed_end=8,
            block_tables=((5, 6, 7, 8), (9, 10, 11), ()),
            slot_mappings=((0, 1, 2), (3, 4), ()),
            new_block_ids_to_zero=(), cow_pairs=(),
            input_ids=(8, 9, 10), ngram_context=(73,))
        factory.admit(admission_to_chunk(second))
        self.assertNotIn('zero', [c[0] for c in calls],
                         'a later chunk must not redo an older allocation zero')
        self.assertNotIn('cow', [c[0] for c in calls],
                         'a later chunk must not redo an older copy')
        self.assertEqual(factory.chunks, 2)


class FullPlanTests(unittest.TestCase):
    def test_targeted_pp_broadcasts_do_not_raise_on_nonreply_workers(self):
        stack = Stack()
        stack.complete_everything()
        for _ in range(250):
            stack.advance()
            stack.complete_everything()
            if stack.cache_calls():
                break
        self.assertTrue(stack.cache_calls())
        self.assertEqual(stack.transport.silent_worker_errors, [])

    def setUp(self):
        self.stack = Stack()
        self.open_lanes = {}

    def begin(self):
        stack = self.stack
        stack.drive_to_finalize()
        self.assertEqual(len(stack.driver.accepted_tickets), 28,
                         '14 plan commands on two stage ranks each')
        return stack

    def test_four_worker_plan_publishes_only_after_every_queried_event(self):
        stack = self.begin()
        self.assertEqual(stack.cache_calls(), [])
        self.assertTrue(stack.controller.leases.held_blocks())
        # Every completed op requires its own host query; at this point none
        # of the fake device events is complete, so nothing may publish.
        stack.advance()
        self.assertEqual(stack.executor.last_method(), 'flash_s5_poll')
        self.assertEqual(stack.cache_calls(), [])
        self.assertEqual(stack.driver.phase, 'finalize')
        # Complete every event except the two receivers' state-write events and
        # their state-finalize segment queries (the conservative consumer
        # boundary is that final segment event).
        for worker in stack.workers.values():
            worker.bus.complete_all(True)
        for rank in (2, 3):
            worker = stack.worker(rank)
            state_ops = accepted_ops(worker, 'state_finalize')
            self.assertEqual(len(state_ops), 1)
            segment_record(worker, 'state_finalize', state_ops[0]).event.complete = False
            worker.dispatcher.state_event.complete = False
        # An ordinary poll must not manufacture completion from incomplete
        # receiver queries, consume PP, release, or publish.
        stack.advance()
        self.assertEqual(stack.executor.last_method(), 'flash_s5_poll')
        self.assertEqual(stack.driver._receipts, {})
        self.assertFalse(stack.driver._pp_noted)
        self.assertEqual(stack.cache_calls(), [])
        self.assertEqual(stack.controller.commits, 0)
        self.assertTrue(stack.controller.leases.held_blocks())
        # Complete only the receiver state-finalize queries. This exposes the
        # PP_CONSUMER proof and the remote receipt, but STATE_WRITE is still
        # pending, so publication must not happen.
        for rank in (2, 3):
            worker = stack.worker(rank)
            op = accepted_ops(worker, 'state_finalize')[0]
            segment_record(worker, 'state_finalize', op).event.complete = True
        stack.drive(lambda: stack.driver._pp_noted)
        self.assertTrue(stack.driver._pp_noted)
        self.assertEqual(stack.cache_calls(), [],
                         'PP consumption alone must not publish')
        self.assertEqual(stack.controller.commits, 0)
        # The consumer receipt was accepted by both senders, but receiver
        # STATE_WRITE is still pending: no release and no publication.
        stack.advance()
        self.assertEqual(stack.executor.last_method(), 'flash_s5_poll')
        self.assertEqual(stack.cache_calls(), [])
        # Complete the state-write events; now the normal release may run.
        for rank in (2, 3):
            worker = stack.worker(rank)
            worker.dispatcher.state_event.complete = True
        stack.drive(lambda: stack.driver.tx is None)
        self.assertEqual(stack.cache_calls(), [(UID, 4)])
        self.assertEqual(stack.scheduler.kv_cache_manager.cache_calls, [(UID, 4)])
        self.assertFalse(stack.controller.leases.held_blocks())
        self.assertTrue(stack.all_dispatchers_reset())
        for rank, ledger in stack.driver.pp_ledgers.items():
            self.assertEqual(ledger.summary(), dict(sent=1, released=1))
        for worker in stack.workers.values():
            # This chunk still has tokens left, so its nonterminal release
            # keeps the request row for the next chunk's update().
            self.assertEqual(worker.factory.released, 0)
            self.assertEqual(worker.factory.registers, 1)
            self.assertIsNone(worker.server.handoff)
            self.assertEqual(worker.server.runtime.releases, [stack.record.transaction])
        # One publication, one handback; no second forward path was used.
        self.assertEqual(stack.controller.commits, 1)
        self.assertEqual(stack.controller.cancels, 0)
        self.assertEqual(stack.scheduler.waiting.items, [self.stack.scheduler.waiting.items[0]])

    def test_duplicate_and_reordered_reports_are_idempotent(self):
        stack = self.begin()
        stack.complete_everything()
        # Keep receiver STATE_WRITE pending so the terminal release cannot race
        # the duplicate delivery below.
        for rank in (2, 3):
            stack.worker(rank).dispatcher.state_event.complete = False
        stack.drive(lambda: stack.driver._pp_noted)
        # A repeated receipt delivery (same queried proof) must not double-free.
        for sender in (0, 1):
            binding = stack.driver._bindings[sender]
            proof = stack.driver._consumer_proofs[sender]
            delivery = w.PPReceiptDelivery(
                stack.driver.tx, sender,
                (PPConsumerReceipt(0, stack.driver.pp_send_seq,
                                   binding.route_generation, proof),))
            ack = stack.worker(sender).flash_s5_receipt(delivery)
            self.assertTrue(ack.accepted)
            self.assertEqual(stack.worker(sender).channel.release_send_calls, 0,
                             'a duplicate receipt must not free the send slot')
        for rank in (2, 3):
            stack.worker(rank).dispatcher.state_event.complete = True
        stack.drive(lambda: stack.driver.tx is None)
        self.assertEqual(stack.cache_calls(), [(UID, 4)])
        for sender in (0, 1):
            self.assertEqual(stack.worker(sender).channel.release_send_calls, 1)


class RetryAndCycleTests(unittest.TestCase):
    def test_repeated_paused_phases_reuse_the_layer_with_new_op_seqs(self):
        stack = Stack()
        # Both stage-0 TP ranks see the same paused QSA phase twice, so their
        # cursor agreement stays exact while the engine re-issues the layer.
        for rank in (0, 1):
            stack.worker(rank).lane.pause_counts[0] = 2
        stack.drive_to_finalize()
        for rank in (0, 1):
            dispatcher = stack.worker(rank).dispatcher
            layer0 = sorted(op for op, (command, _ack) in dispatcher.accepted.items()
                            if command.kind == 'layer'
                            and command.layer_start == 0)
            self.assertEqual(layer0, [layer0[0], layer0[0] + 1, layer0[0] + 2],
                             'one paused phase per accepted operation sequence')
            # Duplicate submission of a recorded paused command is the stored
            # acknowledgment, not a second enqueue.
            stored = dispatcher.accepted[layer0[0]][0]
            calls = list(stack.worker(rank).lane.calls)
            accepted = dict(dispatcher.accepted)
            events = list(dispatcher.events)
            first_ack = dispatcher.accepted[layer0[0]][1]
            ack = dispatcher.submit(stored)
            self.assertTrue(ack['result']['paused'])
            self.assertEqual(ack, first_ack)
            self.assertEqual(stack.worker(rank).lane.calls, calls)
            self.assertEqual(dispatcher.accepted, accepted)
            self.assertEqual(dispatcher.events, events)
        stack.complete_everything()
        stack.drive(lambda: stack.driver.tx is None)
        self.assertEqual(stack.cache_calls(), [(UID, 4)])
        self.assertEqual(stack.controller.commits, 1)
        self.assertTrue(stack.all_dispatchers_reset())

    def test_pending_prefix_retries_the_same_unaccepted_op_seq(self):
        stack = Stack()
        for rank in (2, 3):
            stack.worker(rank).channel.ready = False
        for _ in range(30):
            if stack.driver._retry is not None:
                break
            stack.driver.advance()
        retry = stack.driver._retry
        self.assertIsNotNone(retry, 'the receiver prefix must report pending')
        self.assertIs(retry.kind, w.SegmentKind.PREFIX)
        index = stack.driver.index
        for rank in (2, 3):
            self.assertNotIn(retry.op_seq, stack.worker(rank).dispatcher.accepted,
                             'a pending command is not accepted work')
        stack.driver.advance()
        method, args, _kwargs, _rank = stack.sender.envelopes[-1]
        self.assertEqual(method, 'flash_s5_submit')
        self.assertEqual(args[0].op_seq, retry.op_seq,
                         'a retry must reuse the unaccepted operation sequence')
        self.assertEqual(stack.driver.index, index)
        for rank in (2, 3):
            stack.worker(rank).channel.ready = True
        stack.drive(lambda: stack.driver.phase == 'finalize')
        for rank in (2, 3):
            prefix = accepted_ops(stack.worker(rank), 'prefix')
            self.assertEqual(len(prefix), 1, 'the retried op_seq advances the plan once')
        stack.complete_everything()
        stack.drive(lambda: stack.driver.tx is None)
        self.assertEqual(stack.cache_calls(), [(UID, 4)])

    def test_repeated_transactions_close_and_restart_bounded_history(self):
        stack = Stack()
        first = stack.record.transaction
        stack.drive_to_finalize()
        first_epochs = {rank: worker.state.prefill.storage_epoch
                        for rank, worker in stack.workers.items()}
        stack.complete_everything()
        stack.drive(lambda: stack.driver.tx is None)
        self.assertEqual(stack.cache_calls(), [(UID, 4)])
        second_record = stack.controller.admit(stack.scheduler)
        second = second_record.transaction
        self.assertEqual((second.chunk_start, second.chunk_end), (4, 8))
        self.assertEqual(second.txn_seq, first.txn_seq + 1)
        stack.drive_to_finalize()
        for rank, worker in stack.workers.items():
            ops = set(worker.dispatcher.accepted)
            # A prefill chunk owns its own lane reservation: the storage epoch
            # (and therefore the LaneId) advances per chunk, and the real
            # dispatcher resets its accepted history for that new lane epoch.
            # Restarted op sequences are a new lane, never an old replay.
            self.assertGreater(worker.state.prefill.storage_epoch,
                               first_epochs[rank])
            self.assertEqual(min(ops), 0)
            self.assertEqual(worker.dispatcher.transaction, second)
        stack.complete_everything()
        stack.drive(lambda: stack.driver.tx is None)
        self.assertEqual(stack.cache_calls(), [(UID, 4), (UID, 8)])
        self.assertEqual(stack.controller.commits, 2)
        self.assertTrue(stack.all_dispatchers_reset())
        for worker in stack.workers.values():
            self.assertEqual(worker.dispatcher.accepted, {})
            # Both chunks are nonterminal (the request keeps tokens after
            # chunk_end=8): the row registered once and was updated once.
            self.assertEqual(worker.factory.released, 0)
            self.assertEqual(worker.factory.registers, 1)
            self.assertEqual(worker.factory.updates, 1)
            self.assertEqual(worker.server.runtime.releases, [first, second])
            self.assertEqual(worker.server.calls[0][0], 'open')
            self.assertEqual(worker.server.calls[-1], ('clear', second))
        for sender in (0, 1):
            self.assertEqual(stack.worker(sender).channel.release_send_calls, 2)
        for ledger in stack.driver.pp_ledgers.values():
            self.assertEqual(ledger.summary(), dict(sent=2, released=2))


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.stack = Stack()
        # Bootstrap, split transition, open, then the first targeted peer bind.
        self.stack.drive(lambda: self.stack.driver._pending_binding is not None)
        self.tx = self.stack.record.transaction

    def test_bridge_rejects_wrong_transaction_epoch_stage_and_rank(self):
        worker = self.stack.worker(0)
        bridge = worker._flash_s5_bridge
        other = type(self.tx)(1, UID, 0, 0, 4, 99)
        request = w.PollRequest(other, 1, 1)
        with self.assertRaises(WorkerBridgeError):
            bridge.poll(request)
        with self.assertRaises(WorkerBridgeError):
            bridge.poll(w.PollRequest(self.tx, 999, 1))
        with self.assertRaises(WorkerBridgeError):
            bridge.bind_pp(w.PeerBinding(self.tx, 1, LaneId(
                2, LaneRole.PREFILL, 42, 0), 0))
        proof = CompletionProof(
            OperationTicket(self.tx, 2, LaneId(2, LaneRole.PREFILL, 42, 0), 0,
                            DependencyRole.HANDOFF),
            CompletionKind.PP_CONSUMER, 0)
        delivery = w.PPReceiptDelivery(
            self.tx, 1, (PPConsumerReceipt(0, 0, 0, proof),))
        with self.assertRaises(WorkerBridgeError):
            bridge.receipt(delivery)

    def test_dispatcher_rejects_wrong_stage_and_op_seq_reuse(self):
        worker = self.stack.worker(0)
        bridge = worker._flash_s5_bridge
        # The production bridge refuses a PREPARE without the engine's actual
        # lease, so the direct submissions here carry it explicitly.
        source = self.stack.controller.source_hold(self.tx).descriptor()
        command = w.SegmentCommand(
            w.SegmentDescriptor(self.tx, w.SegmentKind.PREPARE, 1), 1, (0, 1),
            source=source)
        with self.assertRaises(Exception):
            bridge.submit(command)
        good = w.SegmentCommand(
            w.SegmentDescriptor(self.tx, w.SegmentKind.PREPARE, 0), 1, (0, 1),
            source=source)
        self.assertTrue(bridge.submit(good).accepted)
        self.assertEqual(worker.dispatcher.source_binding.descriptor, source)
        stale = w.SegmentCommand(
            w.SegmentDescriptor(self.tx, w.SegmentKind.PREFIX, 0), 1, (0, 1))
        with self.assertRaises(Exception):
            bridge.submit(stale)
        self.assertFalse(worker.dispatcher.source_binding.closed)

    def test_driver_rejects_peer_binding_to_another_receiver_lane(self):
        driver = self.stack.driver
        binding = driver._pending_binding
        self.assertIsInstance(binding, w.PeerBinding)
        wrong = LaneId(2, LaneRole.PREFILL, 99, 0)
        # The ack matches the forged request, but that request names a lane
        # other than the receiver this driver opened.
        driver._pending_binding = replace(binding, receiver_lane=wrong)
        ack = w.PeerBindingAck(binding.transaction, binding.sender_rank, wrong, 0, True)
        with self.assertRaises(ed.DriverError):
            driver._apply_bind((ack,))

    def test_report_with_wrong_proof_role_or_watermark_is_refused(self):
        driver = self.stack.driver
        stack = self.stack
        stack.drive_to_finalize()
        stack.complete_everything()
        stack.drive(lambda: stack.driver._pp_noted or stack.driver.tx is None)
        # Re-ingesting a report whose proof names an impossible kind is refused.
        rank = 0
        report = driver.reports[rank]
        self.assertTrue(report.proofs)
        wrong = []
        for proof in report.proofs:
            kind = CompletionKind.STATE_WRITE if proof.kind is CompletionKind.GPU_EVENT \
                else CompletionKind.GPU_EVENT
            wrong.append(replace(proof, kind=kind))
        corrupted = replace(report, proofs=tuple(wrong))
        with self.assertRaises(ed.DriverError):
            driver._ingest_report(corrupted)


class CancellationTests(unittest.TestCase):
    def test_never_dispatched_admission_is_cancelled_without_a_worker_rpc(self):
        stack = Stack()
        before = len(stack.sender.envelopes)
        self.assertTrue(stack.driver.stop('abort'))
        self.assertEqual(len(stack.sender.envelopes), before)
        self.assertEqual(stack.controller.cancels, 1)
        self.assertEqual(stack.controller.commits, 0)
        self.assertFalse(stack.controller.leases.held_blocks())
        self.assertEqual(stack.cache_calls(), [])
        self.assertIsNone(stack.driver.tx)
        self.assertEqual(stack.scheduler.waiting.items, [],
                         'the engine abort path owns the returned request')

    def test_open_before_prepare_is_not_free_and_closes_every_request_row(self):
        stack = Stack()
        stack.drive(lambda: stack.driver.phase == 'bind')
        self.assertIsNone(stack.driver.bound_ticket)
        self.assertTrue(stack.driver.stop('abort'))
        self.assertEqual(stack.controller.cancels, 1)
        self.assertEqual(stack.controller.commits, 0)
        self.assertEqual(stack.cache_calls(), [])
        self.assertTrue(stack.all_dispatchers_reset())
        for worker in stack.workers.values():
            # No prepare ran, so the lane factory never registered a row; the
            # terminal close is the documented no-op, while the server-side
            # operation lease is cleared.
            self.assertEqual(worker.factory.released, 0)
            self.assertIsNone(worker.factory.registered)
            self.assertIsNone(worker.server.handoff)
            self.assertEqual(worker.server.runtime.releases, [stack.record.transaction])
        self.assertEqual(stack.driver.tx, None)

    def test_cancel_after_accepted_work_retires_without_publication(self):
        stack = Stack()
        stack.drive_to_finalize()
        stack.complete_everything()
        # Poll once so every report is queried, then cancel before release.
        stack.advance()
        self.assertEqual(stack.executor.last_method(), 'flash_s5_poll')
        self.assertTrue(stack.driver.stop('cancel'))
        self.assertEqual(stack.controller.commits, 0)
        self.assertEqual(stack.controller.cancels, 1)
        self.assertEqual(stack.cache_calls(), [])
        self.assertFalse(stack.controller.leases.held_blocks())
        self.assertTrue(stack.all_dispatchers_reset())
        self.assertEqual(stack.controller.frontier.committed_blocks(), ())

    def test_incomplete_device_work_retains_ownership_until_the_bound(self):
        stack = Stack(stop_budget=2, stop_tick_budget=1)
        stack.drive_to_finalize()  # events all incomplete
        stack.driver.stop('abort')
        with self.assertRaises(ed.DriverError):
            for _ in range(4):
                stack.driver.pump_stop(1)
        self.assertIsNotNone(stack.driver.error)
        self.assertEqual(stack.controller.commits, 0)
        self.assertEqual(stack.controller.cancels, 0)
        self.assertEqual(stack.cache_calls(), [])
        self.assertTrue(stack.controller.leases.held_blocks())
        # Sender and receiver still own their lane transactions and rows.
        self.assertTrue(all(worker.dispatcher.transaction is not None
                            for worker in stack.workers.values()))
        self.assertTrue(all(worker.factory.released == 0
                            for worker in stack.workers.values()))

    def test_stop_before_send_has_no_unmatched_receive(self):
        stack = Stack(stop_budget=64, stop_tick_budget=1)
        stack.drive(lambda: bool(stack.driver.plan) and stack.driver.index == 4)
        stack.complete_everything()
        stack.driver.stop('abort')
        for _ in range(100):
            stack.complete_everything()
            stack.driver.pump_stop(1)
            if not stack.driver.active:
                break
        self.assertFalse(stack.driver.active)
        self.assertTrue(stack.all_dispatchers_reset())
        self.assertFalse(stack.controller.leases.held_blocks())
        self.assertEqual([len(stack.worker(r).channel.tickets) for r in (2, 3)], [0, 0])
        self.assertEqual(stack.controller.commits, 0)
        self.assertEqual(stack.controller.cancels, 1)

    def test_partial_enqueue_quarantines_and_never_publishes(self):
        stack = Stack()
        # Stop before PP_SEND submission, independent of plan indices.
        stack.drive(lambda: getattr(stack.driver, '_command', None) is not None
                    and stack.driver._command.kind is w.SegmentKind.SUFFIX
                    and stack.driver._command.stage == 0 and bool(stack.driver._inflight))
        for rank in (0, 1):
            worker = stack.worker(rank)
            def fail(*args, **kwargs):
                raise RuntimeError('private enqueue text')
            worker.channel.send = fail
        with self.assertRaises(Exception):
            for _ in range(3):
                stack.driver.advance()
        self.assertIsNotNone(stack.driver.error)
        for rank in (0, 1):
            worker = stack.worker(rank)
            self.assertIn('pp_send', worker.dispatcher.quarantined or '')
            self.assertIsNotNone(worker.dispatcher.transaction)
        self.assertEqual(stack.cache_calls(), [])
        self.assertTrue(stack.controller.leases.held_blocks())
        self.assertEqual(stack.controller.commits, 0)


class ReviewFixTests(unittest.TestCase):
    """Regression tests added in review (release/stop races, partial closure)."""

    def finalized(self, **kwargs):
        stack = Stack(**kwargs)
        stack.drive_to_finalize()
        stack.complete_everything()
        return stack

    def test_stop_racing_a_closed_release_retires_without_publication(self):
        stack = self.finalized()
        stack.drive(lambda: stack.driver.phase == 'release')
        self.assertIn('release', stack.driver._inflight.values())
        stack.driver.stop('cancel')
        stack.driver.pump_stop(8)
        self.assertIsNone(stack.driver.error)
        self.assertIsNone(stack.driver.tx)
        self.assertEqual(stack.controller.commits, 0)
        self.assertEqual(stack.controller.cancels, 1)
        self.assertEqual(stack.cache_calls(), [])
        self.assertFalse(stack.controller.leases.held_blocks())
        self.assertTrue(stack.all_dispatchers_reset())

    def test_partial_release_retries_as_a_broadcast_and_keeps_closed_rows(self):
        stack = self.finalized()
        stack.worker(3).dispatcher.operation.drained = False
        stack.drive(lambda: set(stack.driver._release_acks) == {0, 1, 2})
        self.assertEqual(set(stack.driver._release_acks), {0, 1, 2})
        self.assertIsNotNone(stack.worker(3).dispatcher.transaction,
                             'the not-drained worker keeps its lane transaction')
        for _ in range(3):
            stack.advance()
        self.assertIsNone(stack.driver.error)
        self.assertEqual(stack.controller.commits, 0)
        self.assertEqual(stack.cache_calls(), [])
        stack.worker(3).dispatcher.operation.drained = True
        stack.drive(lambda: stack.driver.tx is None)
        self.assertEqual(stack.controller.commits, 1)
        self.assertEqual(stack.cache_calls(), [(UID, 4)])
        for worker in stack.workers.values():
            # Nonterminal chunk: the retried release never closes the row, and
            # no second release ever runs after the first accepted one.
            self.assertEqual(worker.factory.released, 0)
            self.assertEqual(worker.factory.registers, 1)

    def test_stop_after_partial_release_cancels_only_the_owner(self):
        stack = self.finalized()
        stack.worker(3).dispatcher.operation.drained = False
        stack.drive(lambda: set(stack.driver._release_acks) == {0, 1, 2})
        stack.driver.stop('abort')
        self.assertIsNone(stack.driver.error)
        self.assertIsNotNone(stack.driver.tx, 'rank 3 still owns work')
        stack.worker(3).dispatcher.operation.drained = True
        stack.driver.pump_stop(8)
        self.assertIsNone(stack.driver.error)
        self.assertIsNone(stack.driver.tx)
        self.assertEqual(stack.controller.commits, 0)
        self.assertEqual(stack.controller.cancels, 1)
        self.assertEqual(stack.cache_calls(), [])
        self.assertFalse(stack.controller.leases.held_blocks())
        for worker in stack.workers.values():
            self.assertEqual(worker.factory.released, 1)

    def test_waiting_for_an_inflight_cancel_is_not_a_stop_round(self):
        # Two targeted receipts and one cancel broadcast are three rounds; the
        # twenty waiting ticks below must not count against this bound.
        stack = self.finalized(stop_budget=3, stop_tick_budget=1)
        stack.advance()  # one queried poll
        stack.transport.hold = True
        stack.driver.stop('cancel')
        for _ in range(20):
            stack.driver.pump_stop(1)
        self.assertIsNone(stack.driver.error)
        self.assertTrue(stack.driver.active)
        self.assertEqual(stack.controller.cancels, 0)
        stack.transport.deliver_held()
        stack.driver.pump_stop(4)
        self.assertIsNone(stack.driver.error)
        self.assertIsNone(stack.driver.tx)
        self.assertEqual(stack.controller.cancels, 1)
        self.assertEqual(stack.controller.commits, 0)

    def test_partially_opened_worker_never_certifies_closure(self):
        stack = Stack()
        worker = stack.worker(0)
        bridge = worker._flash_s5_bridge
        # The worker must own its engine epoch before any request row.
        worker.flash_s5_bootstrap(w.BootstrapCommand(7, (0, 1, 2, 3)))
        # The server opened (storage leased) but the dispatcher refuses: the
        # bridge never binds the transaction.
        def refuse(*args, **kwargs):
            raise RuntimeError('dispatcher open failed')
        worker.dispatcher.open = refuse
        open_request = w.OpenRequest(stack.record, (0, 1, 2, 3),
                                     stack.record.state_epoch, 1)
        with self.assertRaises(RuntimeError):
            bridge.open(open_request)
        self.assertIsNone(bridge.transaction)
        self.assertEqual(worker.server.handoff, stack.record.transaction)
        ack = bridge.cancel(w.CancelRequest(stack.record.transaction,
                                            stack.record.state_epoch, 'abort'))
        self.assertEqual(ack.disposition, 'quarantined')
        self.assertFalse(ack.released)
        self.assertFalse(ack.closed)
        # A worker that truly never opened still answers never_open/closed.
        clean = stack.worker(1)._flash_s5_bridge.cancel(
            w.CancelRequest(stack.record.transaction, stack.record.state_epoch, 'abort'))
        self.assertEqual(clean.disposition, 'never_open')
        self.assertTrue(clean.closed)

    def test_release_retry_for_a_closed_transaction_is_idempotent_only_if_identical(self):
        stack = self.finalized()
        stack.drive(lambda: stack.driver.tx is None)
        bridge = stack.worker(0)._flash_s5_bridge
        request = stack.driver.release_request
        self.assertFalse(request.terminal)
        again = bridge.release(request)
        # Nonterminal: storage released, request row retained for update().
        self.assertTrue(again.released)
        self.assertFalse(again.closed)
        self.assertEqual(again, bridge._closed)
        self.assertEqual(again.completed_sequence,
                         dict(request.completed_sequences)[0])
        self.assertEqual(stack.worker(0).factory.released, 0)
        self.assertEqual(stack.worker(0).factory.registers, 1)
        wrong = w.ReleaseRequest(request.transaction, request.state_epoch,
                                 tuple((r, s + 1) for r, s in request.completed_sequences),
                                 request.terminal)
        with self.assertRaises(WorkerBridgeError):
            bridge.release(wrong)
        wrong_terminal = w.ReleaseRequest(
            request.transaction, request.state_epoch, request.completed_sequences,
            not request.terminal)
        with self.assertRaises(WorkerBridgeError):
            bridge.release(wrong_terminal)


class TerminalReleaseTests(unittest.TestCase):
    """The typed terminal field: row retained per chunk, closed at handback."""

    def finish_chunk(self, stack):
        stack.drive_to_finalize()
        stack.complete_everything()
        stack.drive(lambda: stack.driver.tx is None)
        self.assertIsNone(stack.driver.error)

    def test_two_chunks_reach_update_and_terminal_handback_closes_the_row(self):
        stack = Stack()
        # The Stack constructor admits the first chunk.
        records = [stack.record]
        transactions = [stack.record.transaction]
        self.assertEqual((records[0].transaction.chunk_start,
                          records[0].transaction.chunk_end), (0, 4))
        self.finish_chunk(stack)
        for expected in ((4, 8), (8, 11)):
            record = stack.controller.admit(stack.scheduler)
            self.assertIsNotNone(record)
            transaction = record.transaction
            self.assertEqual((transaction.chunk_start, transaction.chunk_end), expected)
            transactions.append(transaction)
            records.append(record)
            self.finish_chunk(stack)
            for worker in stack.workers.values():
                self.assertEqual(worker.factory.registers, 1,
                                 'the request row is registered exactly once')
        # First two chunks are nonterminal: each later chunk reached update().
        # The fixture only records counts, so three admissions in one row prove
        # the row survived both nonterminal releases.
        for worker in stack.workers.values():
            self.assertEqual(worker.factory.updates, 2)
            self.assertEqual(worker.factory.released, 1,
                             'only the terminal chunk closes the row')
            self.assertIsNone(worker.factory.registered)
            self.assertEqual(len(worker.factory.chunks), 3)
        self.assertEqual(stack.cache_calls(),
                         [(UID, 4), (UID, 8), (UID, 11)])
        self.assertEqual(stack.controller.commits, 3)
        self.assertEqual([r.terminal for r in records], [False, False, True])
        # The last chunk leaves the final prompt token to the ordinary path.
        self.assertEqual(len(stack.scheduler.waiting.items), 1)
        for rank in (0, 1, 2, 3):
            self.assertEqual(stack.worker(rank).server.runtime.releases,
                             transactions)

    def test_terminal_ack_requires_the_row_to_be_closed(self):
        stack = Stack(chunks=8)
        # prompt_end 12, chunk 0..8 leaves four tokens: nonterminal.
        self.assertFalse(stack.record.terminal)
        # A worker that drops the row early is refused by the engine. Patch
        # before the release is dispatched: the synchronous fixture transport
        # computes every reply at submit time.
        worker = stack.worker(0)
        real = worker.flash_s5_release

        def early(message):
            ack = real(message)
            return w.ReleaseAck(ack.transaction, ack.worker_rank, ack.state_epoch,
                                ack.completed_sequence, released=True,
                                request_closed=True)

        worker.flash_s5_release = early
        stack.drive_to_finalize()
        stack.complete_everything()
        stack.drive(lambda: stack.driver.phase == 'release')
        self.assertIs(stack.driver.release_request.terminal, False)
        with self.assertRaises(ed.DriverError):
            stack.driver.advance()
        self.assertEqual(stack.controller.commits, 0)
        self.assertEqual(stack.cache_calls(), [])

    def test_cancel_after_nonterminal_release_closes_the_retained_row(self):
        stack = Stack()
        self.finish_chunk(stack)
        bridge = stack.worker(0)._flash_s5_bridge
        retained = bridge._retained
        self.assertIsNotNone(retained)
        self.assertEqual(retained['transaction'], stack.record.transaction)
        self.assertFalse(stack.worker(0).factory.released)
        ack = bridge.cancel(w.CancelRequest(stack.record.transaction,
                                            stack.record.state_epoch, 'abort'))
        self.assertEqual(ack.disposition, 'released')
        self.assertTrue(ack.closed)
        self.assertIsNone(bridge._retained)
        self.assertEqual(stack.worker(0).factory.released, 1)
        # The cancellation row carries the worker's own final queried report.
        self.assertEqual(ack.report, retained['report'])
        self.assertEqual(ack.completed_sequence, retained['report'].completed_sequence)

    def test_cancel_of_another_chunk_quarantines_a_retained_row(self):
        stack = Stack()
        self.finish_chunk(stack)
        bridge = stack.worker(0)._flash_s5_bridge
        other = type(stack.record.transaction)(7, UID, 0, 4, 8, 99)
        ack = bridge.cancel(w.CancelRequest(other, stack.record.state_epoch + 1,
                                            'abort'))
        self.assertEqual(ack.disposition, 'quarantined')
        self.assertFalse(ack.closed)
        self.assertIsNotNone(bridge._retained)
        self.assertFalse(stack.worker(0).factory.released)


class DecodeProgressTests(unittest.TestCase):
    def test_bounded_cancel_ticks_do_not_steal_stock_batches_or_samples(self):
        stack = Stack(stop_tick_budget=1)
        stack.drive_to_finalize()
        stack.complete_everything()

        class Stock:
            def __init__(self):
                self.batches = [0, 1, 2, 3, 4]
                self.taken = []
                self.samples = []
                self.outputs = [None, 'a', None, 'b', 'c']

            def step(self):
                batch = self.batches.pop(0)
                self.taken.append(batch)
                out = self.outputs[batch]
                if out is not None:
                    self.samples.append(out)
                return out

        stock = Stock()
        ticks = 0
        while stack.driver.active and ticks < 40:
            stack.driver._drain()
            if stack.driver.stop_kind is None:
                stack.driver.stop('pause')
            else:
                stack.driver.pump_stop(1)
            if stock.batches:
                stock.step()
            ticks += 1
        # Drain the remaining ordinary batches once cancellation is closed.
        while stock.batches:
            stock.step()
        self.assertEqual(stock.taken, [0, 1, 2, 3, 4])
        self.assertEqual(stock.samples, ['a', 'b', 'c'])
        self.assertEqual(stack.controller.commits, 0)
        self.assertEqual(stack.cache_calls(), [])
        self.assertFalse(stack.controller.leases.held_blocks())
        self.assertTrue(stack.all_dispatchers_reset())


class RouteCompletionTests(unittest.TestCase):
    """Engine-owner bootstrap and the drained full <-> split route completion."""

    def finish_chunk(self, stack):
        stack.drive_to_finalize()
        stack.complete_everything()
        stack.drive(lambda: stack.driver.tx is None)
        self.assertIsNone(stack.driver.error)

    def full_command(self, stack, **overrides):
        values = dict(
            engine_epoch=stack.driver.engine_epoch,
            expected_route_generation=stack.driver.route_generation,
            route_generation=stack.driver.route_generation + 1,
            target_bank=w.ROUTE_BANK_FULL,
            storage_epochs=tuple(stack.driver._storage_epochs[rank]
                                 for rank in stack.driver.ranks),
            decode_replay_watermarks=tuple(stack.driver._decode_watermarks[rank]
                                           for rank in stack.driver.ranks))
        values.update(overrides)
        return w.TransitionCommand(**values)

    def bind_split(self, stack):
        """Bind and switch every worker without opening a request row."""
        epoch = stack.driver.engine_epoch
        ranks = stack.driver.ranks
        command = w.BootstrapCommand(epoch, ranks)
        for rank in ranks:
            self.assertTrue(stack.worker(rank).flash_s5_bootstrap(command).bound)
        storage = tuple(stack.worker(rank).state.decode.handle.storage_epoch
                        for rank in ranks)
        transition = w.TransitionCommand(
            epoch, 0, 1, w.ROUTE_BANK_SPLIT, storage, (0,) * len(ranks))
        for rank in ranks:
            ack = stack.worker(rank).flash_s5_transition(transition)
            self.assertEqual(ack.target_bank, w.ROUTE_BANK_SPLIT)
        return transition

    def worker_full_command(self, stack, **overrides):
        ranks = stack.driver.ranks
        values = dict(
            engine_epoch=stack.driver.engine_epoch,
            expected_route_generation=1, route_generation=2,
            target_bank=w.ROUTE_BANK_FULL,
            storage_epochs=tuple(stack.worker(rank).state.decode.handle.storage_epoch
                                 for rank in ranks),
            decode_replay_watermarks=tuple(stack.worker(rank).router.replay_watermark
                                           for rank in ranks))
        values.update(overrides)
        return w.TransitionCommand(**values)

    def test_bootstrap_and_split_are_applied_before_any_open(self):
        stack = Stack()
        stack.driver.advance()
        self.assertEqual([envelope[0] for envelope in stack.sender.envelopes],
                         ['flash_s5_bootstrap'])
        self.assertFalse(stack.driver._bootstrapped)
        for rank in stack.driver.ranks:
            worker = stack.worker(rank)
            # The worker executed the command; the engine has not consumed
            # the ACK, so no split/storage decision has been accepted yet.
            self.assertTrue(worker.state.engine_bound)
            self.assertTrue(worker.router.engine_owner_bound)
            self.assertEqual(worker.controller.route, 'full')

        stack.driver.advance()
        self.assertTrue(stack.driver._bootstrapped)
        self.assertFalse(stack.driver._split_ready)
        self.assertEqual([envelope[0] for envelope in stack.sender.envelopes],
                         ['flash_s5_bootstrap', 'flash_s5_transition'])
        command = stack.sender.envelopes[-1][1][0]
        self.assertIsInstance(command, w.TransitionCommand)
        self.assertEqual(command.target_bank, w.ROUTE_BANK_SPLIT)
        self.assertEqual(command.expected_route_generation, 0)
        for rank in stack.driver.ranks:
            worker = stack.worker(rank)
            self.assertEqual(worker.state.engine_epoch, stack.driver.engine_epoch)
            self.assertTrue(worker.state.engine_bound)
            # The switch was computed for the reply, but the engine still
            # treats the route as full and has admitted no open work.
            self.assertEqual(worker.controller.route, 'split')
            self.assertEqual(stack.driver.route_generation, 0)
            self.assertIsNone(stack.driver.tx)
        self.assertNotIn('flash_s5_open',
                         [envelope[0] for envelope in stack.sender.envelopes])

        stack.driver.advance()
        self.assertTrue(stack.driver._split_ready)
        self.assertEqual([envelope[0] for envelope in stack.sender.envelopes][:3],
                         ['flash_s5_bootstrap', 'flash_s5_transition', 'flash_s5_open'])
        for rank in stack.driver.ranks:
            worker = stack.worker(rank)
            self.assertEqual(worker.controller.route, 'split')
            self.assertEqual(worker.router.current_mode, 'decode-split')
            self.assertEqual(worker.controller.route_generation, 1)
            # The drained switch queried every local consumer stream.
            self.assertTrue(worker.bus.syncs)
            self.assertEqual(stack.driver._storage_epochs[rank],
                             worker.state.decode.handle.storage_epoch)
            self.assertEqual(stack.driver._decode_watermarks[rank],
                             worker.router.replay_watermark)

    def test_two_chunks_keep_one_persistent_decode_epoch_and_split_route(self):
        stack = Stack()
        self.finish_chunk(stack)
        self.assertTrue(stack.driver._split_ready)
        self.assertEqual(stack.driver.route_generation, 1)
        epochs = dict(stack.driver._storage_epochs)
        handles = {rank: stack.worker(rank).state.decode.handle
                   for rank in stack.driver.ranks}
        for rank in stack.driver.ranks:
            self.assertEqual(stack.worker(rank).controller.route, 'split')

        record = stack.controller.admit(stack.scheduler)
        self.assertIsNotNone(record)
        self.finish_chunk(stack)
        self.assertEqual(dict(stack.driver._storage_epochs), epochs)
        self.assertEqual(stack.driver.route_generation, 1)
        for rank in stack.driver.ranks:
            worker = stack.worker(rank)
            self.assertIs(worker.state.decode.handle, handles[rank])
            self.assertIs(worker.state.decode_owner, worker.router.worker.decode_owner)
            self.assertEqual(worker.controller.route, 'split')
            self.assertEqual(worker.controller.route_generation, 1)
        # No full-bank return happened between two nonterminal chunks.
        kinds = [envelope[0] for envelope in stack.sender.envelopes]
        self.assertEqual(kinds.count('flash_s5_transition'), 1)

    def test_delayed_transition_ack_blocks_open_and_stock_work(self):
        stack = Stack()
        stack.transport.hold = True
        stack.driver.advance()
        self.assertFalse(stack.driver._bootstrapped)
        stack.transport.deliver_held()
        stack.transport.hold = True
        stack.driver.advance()
        # Bootstrap applied, split issued, its reply is still on the wire.
        self.assertTrue(stack.driver._bootstrapped)
        self.assertEqual(stack.executor.last_method(), 'flash_s5_transition')
        self.assertEqual(stack.driver.advance(), (None, False))
        self.assertFalse(stack.driver._split_ready)
        self.assertIsNone(stack.driver.tx)
        self.assertEqual(stack.cache_calls(), [])
        for rank in stack.driver.ranks:
            worker = stack.worker(rank)
            # The worker computed the drained switch for its reply; the engine
            # has not seen the ACK and must not open or admit stock work.
            self.assertEqual(worker.controller.route, 'split')
            self.assertEqual(worker.router.current_mode, 'decode-split')
        self.assertNotIn('flash_s5_open',
                         [envelope[0] for envelope in stack.sender.envelopes])

        stack.transport.deliver_held()
        stack.driver.advance()
        self.assertTrue(stack.driver._split_ready)
        for rank in stack.driver.ranks:
            self.assertEqual(stack.worker(rank).controller.route, 'split')
            self.assertEqual(stack.worker(rank).router.current_mode, 'decode-split')

    def test_terminal_handback_holds_full_bank_until_the_drained_ack(self):
        stack = Stack()
        self.finish_chunk(stack)
        self.assertIsNotNone(stack.controller.admit(stack.scheduler))
        self.finish_chunk(stack)
        self.assertIsNotNone(stack.controller.admit(stack.scheduler))
        stack.drive_to_finalize()
        stack.complete_everything()
        stack.drive(lambda: stack.driver.phase == 'release')
        # Hold the all-rank FULL reply at the transport: the switch has been
        # computed and drained on every worker but is not visible yet.
        for rank in stack.driver.ranks:
            worker = stack.worker(rank)
            real = worker.flash_s5_transition

            def held(message, real=real):
                ack = real(message)
                if message.target_bank == w.ROUTE_BANK_FULL:
                    stack.transport.hold = True
                return ack

            worker.flash_s5_transition = held
        stack.drive(lambda: stack.driver.tx is None)
        opens = sum(1 for envelope in stack.sender.envelopes
                    if envelope[0] == 'flash_s5_open')
        self.assertEqual(stack.driver._return_pending, 'handback')
        self.assertIsNotNone(stack.driver._transition)
        self.assertEqual(stack.driver._transition.target_bank, w.ROUTE_BANK_FULL)
        self.assertTrue(stack.driver._split_ready)
        for rank in stack.driver.ranks:
            # The local switch is proven, but the engine has no ACK.
            self.assertEqual(stack.worker(rank).controller.route, 'full')
            self.assertEqual(stack.worker(rank).router.current_mode, 'ordinary')
        self.assertEqual(stack.executor.last_method(), 'flash_s5_transition')
        self.assertEqual(stack.driver.advance(), (None, False))
        self.assertEqual(sum(1 for envelope in stack.sender.envelopes
                             if envelope[0] == 'flash_s5_open'), opens)

        stack.transport.deliver_held()
        stack.driver.advance()
        self.assertFalse(stack.driver._split_ready)
        self.assertIsNone(stack.driver._return_pending)
        self.assertEqual(stack.driver.route_generation, 2)
        for rank in stack.driver.ranks:
            self.assertEqual(stack.worker(rank).controller.route, 'full')
        self.assertFalse(stack.driver.active)

    def test_cancel_before_prepare_returns_to_full_after_drain(self):
        stack = Stack()
        stack.drive(lambda: stack.driver.phase == 'open')
        self.assertTrue(stack.driver._split_ready)
        stack.driver.stop('abort')
        stack.drive(lambda: not stack.driver.active)
        self.assertIsNone(stack.driver.tx)
        self.assertFalse(stack.driver._split_ready)
        self.assertIsNone(stack.driver._return_pending)
        self.assertEqual(stack.driver.route_generation, 2)
        for rank in stack.driver.ranks:
            worker = stack.worker(rank)
            self.assertEqual(worker.controller.route, 'full')
            self.assertEqual(worker.router.current_mode, 'ordinary')
            self.assertIsNone(worker.dispatcher.transaction)
        self.assertEqual(stack.cache_calls(), [])
        self.assertEqual(stack.controller.commits, 0)

    def test_missing_worker_reply_never_advances_the_route(self):
        stack = Stack()
        stack.driver.advance()          # bootstrap
        stack.transport.withhold(3)
        stack.driver.advance()          # applies bootstrap, issues split
        with self.assertRaises(ed.DriverError):
            stack.driver.advance()
        self.assertEqual(stack.driver.error,
                         'native driver failed; ownership retained')
        self.assertFalse(stack.driver._split_ready)
        self.assertEqual(stack.driver.route_generation, 0)
        self.assertIsNone(stack.driver.tx)
        self.assertEqual(stack.cache_calls(), [])
        # Ranks 0-2 did switch locally; the engine never reports that as success.
        for rank in (0, 1, 2):
            self.assertEqual(stack.worker(rank).controller.route, 'split')
        self.assertEqual(stack.worker(3).controller.route, 'full')

    def test_partial_local_switch_quarantines_the_router(self):
        stack = Stack()
        self.bind_split(stack)
        worker = stack.worker(0)
        bridge = worker._flash_s5_bridge
        command = self.worker_full_command(stack)
        real = worker.controller.leave_split

        def ambiguous():
            real()
            raise RuntimeError('copy stream drain state is unclear')

        worker.controller.leave_split = ambiguous
        with self.assertRaises(E3ContractError):
            worker.flash_s5_transition(command)
        self.assertIn('ambiguous', bridge._route_failure)
        self.assertTrue(worker.router.unclean)
        self.assertEqual(worker.router.state, 'quarantined')
        # The controller did move, but no success was acknowledged.
        self.assertEqual(worker.controller.route, 'full')

    def test_foreign_late_bootstrap_never_rebinds_a_live_incarnation(self):
        stack = Stack()
        stack.drive(lambda: stack.driver._split_ready)
        worker = stack.worker(0)
        epoch = worker.state.engine_epoch
        # Retrying the same engine epoch is idempotent and re-reports the census.
        ack = worker.flash_s5_bootstrap(w.BootstrapCommand(epoch, (0, 1, 2, 3)))
        self.assertTrue(ack.bound)
        self.assertEqual(ack.engine_epoch, epoch)
        self.assertIs(ack.lane.role, LaneRole.DECODE)
        with self.assertRaises(E3ContractError):
            worker.flash_s5_bootstrap(w.BootstrapCommand(epoch + 1, (0, 1, 2, 3)))
        with self.assertRaises(E3ContractError):
            worker.flash_s5_bootstrap(w.BootstrapCommand(epoch, (1, 2, 3)))
        self.assertEqual(worker.state.engine_epoch, epoch)
        self.assertEqual(worker.controller.route, 'split')

    def test_stale_storage_generation_and_watermark_are_refused(self):
        stack = Stack()
        self.bind_split(stack)
        worker = stack.worker(0)
        bridge = worker._flash_s5_bridge
        command = self.worker_full_command(stack)
        owner = worker.state.decode_owner
        with self.assertRaises(E3ContractError):
            worker.state.retain_decode_storage(context_id=99999)
        self.assertIs(worker.state.decode_owner, owner)

        stale_storage = replace(command, storage_epochs=tuple(
            epoch + 1 for epoch in command.storage_epochs))
        with self.assertRaises(E3ContractError):
            worker.flash_s5_transition(stale_storage)
        wrong_engine = replace(command, engine_epoch=command.engine_epoch + 1)
        with self.assertRaises(E3ContractError):
            worker.flash_s5_transition(wrong_engine)
        stale_route = w.TransitionCommand(
            command.engine_epoch, 0, 1, w.ROUTE_BANK_FULL,
            command.storage_epochs, command.decode_replay_watermarks)
        with self.assertRaises(E3ContractError):
            worker.flash_s5_transition(stale_route)
        # A commanded watermark behind the census this worker acknowledged.
        bridge._ack_watermark = 5
        try:
            with self.assertRaises(E3ContractError):
                worker.flash_s5_transition(command)
        finally:
            bridge._ack_watermark = 0
        # A floor ahead of the worker's own queried census.
        ahead = replace(command, decode_replay_watermarks=tuple(
            value + 99 for value in command.decode_replay_watermarks))
        with self.assertRaises(E3ContractError):
            worker.flash_s5_transition(ahead)
        # Every refusal left both the controller and graph router on split.
        self.assertEqual(worker.controller.route, 'split')
        self.assertEqual(worker.router.current_mode, 'decode-split')
        self.assertEqual(worker.controller.route_generation, 1)


class ConstructionTests(unittest.TestCase):
    def test_package_import_and_default_off_identity_plus_native_refusal(self):
        self.assertEqual(worker_hooks.workspace_lanes(3, env={}), 3)
        self.assertFalse(worker_hooks._split_enabled(env={}))
        with self.assertRaises(s5_startup.StartupError):
            s5_startup.require_native_ready()
        import make_s5_overlap
        self.assertIn('worker_bridge', make_s5_overlap.E3_MODULES)
        import decode_probe
        # Build-only sources ship but are not runtime-imported; the declaration
        # is the generated package minus those (make_s5_overlap enforces it).
        self.assertEqual(list(decode_probe.GENERATED_SOURCES),
                         [n for n in make_s5_overlap.E3_MODULES
                          if n not in decode_probe.BUILD_ONLY_SOURCES])
        self.assertIn('admission_metadata', decode_probe.GENERATED_SOURCES)
        # The generated worker still carries every typed method the bridge serves.
        source = (ROOT / 'reference/vllm-src/v1/worker/gpu_worker.py').read_text()
        generated = eh.generate_worker(source)
        for name in ('flash_s5_open', 'flash_s5_submit', 'flash_s5_poll',
                     'flash_s5_release', 'flash_s5_bind', 'flash_s5_receipt',
                     'flash_s5_cancel', 'flash_s5_transition',
                     'flash_s5_bootstrap'):
            self.assertIn(f'def {name}(', generated)
        self.assertEqual(eh.revert_worker(generated), source)

    def test_install_native_lane_wires_the_real_bridge_objects(self):
        # Exercise the startup construction seam with a fake native builder and
        # a fake dispatcher constructor; the bridge itself is the real class.
        import native_startup
        from unittest.mock import patch
        server = types.SimpleNamespace(
            operation_device=None,
            runtime=types.SimpleNamespace(
                state=types.SimpleNamespace(worker_rank=0, prefill=None)))
        runner = types.SimpleNamespace(device='cuda:0', _flash_s5_worker=server)
        native = types.SimpleNamespace(
            split=types.SimpleNamespace(prefill_stream='prefill'),
            comms=types.SimpleNamespace(pp=None), copy_stream=None)
        created = {}

        class StartupDispatcher:
            def __init__(self, runner, native, factory, **kwargs):
                created['args'] = (runner, native, factory, kwargs)
                self.transaction = None
                self.channel = None

            def open(self, *args, **kwargs):
                return {'opened': True}

            def submit(self, command):
                return {'op_seq': command.op_seq}

        modules = {
            'vllm': types.ModuleType('vllm'),
            'vllm.models': types.ModuleType('vllm.models'),
            'vllm.models.qwen4_exp': types.ModuleType('vllm.models.qwen4_exp'),
            'vllm.models.qwen4_exp.nvidia': types.ModuleType('vllm.models.qwen4_exp.nvidia'),
            'vllm.models.qwen4_exp.nvidia.qsa': types.ModuleType(
                'vllm.models.qwen4_exp.nvidia.qsa'),
        }
        with patch.object(native_startup, 'install_worker_native',
                          lambda *a, **kw: native), \
             patch.object(lane_chunk, 'LaneChunkFactory', lambda runner: 'factory'), \
             patch('lane_dispatch.LaneDispatcher', StartupDispatcher), \
             patch.dict(sys.modules, modules):
            dispatcher = s5_startup.install_native_lane(
                runner, env={'FLASH_S5_OVERLAP': '0'})
        self.assertIs(dispatcher, server._flash_s5_dispatcher)
        self.assertIs(dispatcher, runner._flash_s5_dispatcher)
        self.assertIs(server._flash_s5_bridge.dispatcher, dispatcher)
        self.assertIs(server._flash_s5_bridge, runner._flash_s5_bridge)
        self.assertIsInstance(server._flash_s5_bridge, NativeWorkerBridge)
        self.assertIs(created['args'][0], runner)
        self.assertEqual(created['args'][1], native)
        self.assertEqual(created['args'][2], 'factory')


class SourceHoldIntegrationTests(unittest.TestCase):
    """The scheduler hold over real allocator objects, inside the real drive."""

    def test_hold_pins_pages_until_the_copy_proof_and_rank_barrier(self):
        stack = Stack(chunks=4)
        record = stack.record
        # Every segment is submitted; the workers still own their frames.
        stack.drive_to_finalize()
        hold = stack.controller.source_hold(record.transaction)
        self.assertIsInstance(hold, sh.SourceHold)
        owner = stack.driver.lanes[0]
        pages = hold.pages
        self.assertTrue(hold.validate(owner, 0, pages))
        self.assertTrue(hold.validate(owner, 3, pages))
        descriptor = hold.descriptor()
        # Every fake worker validates the same descriptor against the frame it
        # actually opened; the engine keeps the real block objects.
        for rank in stack.controller.ranks:
            worker = stack.worker(rank)
            frame = worker.dispatcher.admission
            self.assertIsInstance(frame, ChunkAdmission)
            self.assertEqual(frame.transaction, record.transaction)
            self.assertEqual(frame.state_epoch, record.state_epoch)
            self.assertTrue(descriptor.validate_frame(
                frame, stack.controller.ranks))
        self.assertEqual(stack.controller.leases.held_transaction(
            record.transaction), hold.pages)

        # Allocator reuse attempted before the proof is refused.
        pool = stack.scheduler.kv_cache_manager.block_pool
        page = pages[0]
        block = pool.block(page)
        pool.free_blocks([block])
        with self.assertRaises(E3ContractError):
            hold.validate(owner, 0, (page,))
        block.ref_cnt = 1

        # The final queried reports become clean now; the driver is about to
        # broadcast the release.
        stack.complete_everything()
        stack.drive(lambda: stack.controller.ready_to_release(record.transaction))

        # The worker side returns the correlated COPY_EVENT proof and the
        # all-rank barrier; the engine's own commit then must not release a
        # second time or re-publish the already-free lease.
        acks = {}
        for rank in stack.controller.ranks:
            report = stack.driver.reports[rank]
            ack = w.ReleaseAck(record.transaction, rank, record.state_epoch,
                               report.completed_sequence, released=True,
                               request_closed=False)
            stack.controller.note_worker_release(ack)
            acks[rank] = ack
        proof = CompletionProof(stack.driver.bound_ticket,
                                CompletionKind.COPY_EVENT,
                                max(a.completed_sequence for a in acks.values()))
        self.assertTrue(hold.release(proof, acks, stack.controller.ranks))
        self.assertEqual(hold.release_mode, sh.SourceHold.COPY)
        self.assertEqual(stack.controller.leases.held_transaction(
            record.transaction), ())
        # After the proof the source is no longer pinned for reads.
        with self.assertRaises(E3ContractError):
            hold.validate(owner, 0, (page,))

        stack.drive(lambda: stack.controller.commits == 1)
        self.assertEqual(stack.controller.commits, 1)
        self.assertEqual(stack.cache_calls(), [(UID, 4)])
        self.assertEqual(hold.release_mode, sh.SourceHold.COPY)
        self.assertEqual(stack.controller.leases.held_blocks(), ())


if __name__ == '__main__':
    unittest.main()
